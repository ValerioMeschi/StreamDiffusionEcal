import os
import sys
import time
import threading
from multiprocessing import Process, Queue, get_context
from multiprocessing.connection import Connection
from typing import List, Literal, Dict, Optional
import torch
from PIL import Image
import io
from streamdiffusion.image_utils import pil2tensor
import fire
import NDIlib as ndi
import numpy as np
from flask import Flask, Response, request
from collections import deque


app = Flask(__name__)

from utils.viewer import receive_images
from utils.wrapper import StreamDiffusionWrapper
from streamdiffusion.image_utils import postprocess_image

inputs = deque(maxlen=10)
top = 0
left = 0

# methodes pour streamer les frame en sortie. A threader dans main
def stream_frames(queue, convert=True):
    """ Continuously yields frames from the queue for MJPEG streaming. """
    while True:
        if not queue.empty():
            # convertir les images reçues dans la quue en byte array pour MJPEG

            if convert:
                pImage = postprocess_image(queue.get(block=False), output_type="pil")[0]
            else:
                pImage = queue.get(block=False)
            img_io = io.BytesIO()
            pImage.save(img_io, format="JPEG")
            img_io.seek(0)
            img_bytes = img_io.read()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img_bytes + b'\r\n')


import re
def get_string_between_parentheses(s):
    # Regular expression to find content inside parentheses
    match = re.search(r'\((.*?)\)', s)
    if match:
        return match.group(1)
    else:
        return None  # Return None if no match is found

def normalize(arr):
    """
    Normalisation linéaire vectorisée pour les canaux RGB,
    en laissant inchangé le canal alpha.
    """
    arr = arr.astype('float')
    rgb = arr[..., :3]
    min_vals = rgb.min(axis=(0, 1), keepdims=True)
    max_vals = rgb.max(axis=(0, 1), keepdims=True)
    diff = np.where((max_vals - min_vals) == 0, 1, max_vals - min_vals)
    rgb_normalized = (rgb - min_vals) * (255.0 / diff)
    arr[..., :3] = rgb_normalized
    return arr

def ndi_receiver(event: threading.Event,height: int = 512,
width: int = 512):
    global inputs
    if not ndi.initialize():
        return 0

    ndi_find = ndi.find_create_v2()

    if ndi_find is None:
        return 0

    sources = []
    while not event.is_set() and not sources:
        print('Looking for sources ...')
        ndi.find_wait_for_sources(ndi_find, 1000)
        sources = ndi.find_get_current_sources(ndi_find)

    selectedSourceindex = 0
    for idx, s in enumerate(sources):
        sourcename = get_string_between_parentheses(s.ndi_name)
        print(sourcename)
        if sourcename == "OBS":
            print("found obs")
            print(s)
            selectedSourceindex = idx
    print(selectedSourceindex)
    ndi_recv_create = ndi.RecvCreateV3()
    ndi_recv_create.color_format = ndi.RECV_COLOR_FORMAT_BGRX_BGRA
    ndi_recv = ndi.recv_create_v3(ndi_recv_create)

    if ndi_recv is None:
        return 0
    ndi.recv_connect(ndi_recv, sources[selectedSourceindex])
    ndi.find_destroy(ndi_find)
    #cv.startWindowThread()

    while True:
        if event.is_set():
            print("terminate read thread")
            break
        t, v, _, _ = ndi.recv_capture_v2(ndi_recv, 5000)
        if t == ndi.FRAME_TYPE_VIDEO:
            #print('Video data received (%dx%d).' % (v.xres, v.yres))
            frame = np.copy(v.data)
            pil_image = Image.fromarray(normalize(frame).astype('uint8'),'RGBA')
            rgb = pil_image.convert("RGB")
            #cv.imshow('ndi image', frame)
            #pil_image.show()
            pil_image.resize((height, width))
            inputs.append(pil2tensor(rgb))
            ndi.recv_free_video_v2(ndi_recv, v)

    ndi.recv_destroy(ndi_recv)
    ndi.destroy()
    #cv.destroyAllWindows()

    return 0



def image_generation_process(
    queue: Queue,
    fps_queue: Queue,
    close_queue: Queue,
    model_id_or_path: str,
    lora_dict: Optional[Dict[str, float]],
    prompt: str,
    negative_prompt: str,
    frame_buffer_size: int,
    width: int,
    height: int,
    acceleration: Literal["none", "xformers", "tensorrt"],
    use_denoising_batch: bool,
    seed: int,
    cfg_type: Literal["none", "full", "self", "initialize"],
    guidance_scale: float,
    delta: float,
    do_add_noise: bool,
    enable_similar_image_filter: bool,
    similar_image_filter_threshold: float,
    similar_image_filter_max_skip_frame: float,
    prompt_queue : Queue,
    inputs_queue: Queue,
    seed_queue: Queue,
) -> None:
    """
    Process for generating images based on a prompt using a specified model.

    Parameters
    ----------
    queue : Queue
        The queue to put the generated images in.
    fps_queue : Queue
        The queue to put the calculated fps.
    model_id_or_path : str
        The name of the model to use for image generation.
    lora_dict : Optional[Dict[str, float]], optional
        The lora_dict to load, by default None.
        Keys are the LoRA names and values are the LoRA scales.
        Example: {'LoRA_1' : 0.5 , 'LoRA_2' : 0.7 ,...}
    prompt : str
        The prompt to generate images from.
    negative_prompt : str, optional
        The negative prompt to use.
    frame_buffer_size : int, optional
        The frame buffer size for denoising batch, by default 1.
    width : int, optional
        The width of the image, by default 512.
    height : int, optional
        The height of the image, by default 512.
    acceleration : Literal["none", "xformers", "tensorrt"], optional
        The acceleration method, by default "tensorrt".
    use_denoising_batch : bool, optional
        Whether to use denoising batch or not, by default True.
    seed : int, optional
        The seed, by default 2. if -1, use random seed.
    cfg_type : Literal["none", "full", "self", "initialize"],
    optional
        The cfg_type for img2img mode, by default "self".
        You cannot use anything other than "none" for txt2img mode.
    guidance_scale : float, optional
        The CFG scale, by default 1.2.
    delta : float, optional
        The delta multiplier of virtual residual noise,
        by default 1.0.
    do_add_noise : bool, optional
        Whether to add noise for following denoising steps or not,
        by default True.
    enable_similar_image_filter : bool, optional
        Whether to enable similar image filter or not,
        by default False.
    similar_image_filter_threshold : float, optional
        The threshold for similar image filter, by default 0.98.
    similar_image_filter_max_skip_frame : int, optional
        The max skip frame for similar image filter, by default 10.
    """

    global inputs
    global app

    stream = StreamDiffusionWrapper(
        model_id_or_path=model_id_or_path,
        lora_dict=lora_dict,
        t_index_list=[32, 45],
        frame_buffer_size=frame_buffer_size,
        width=width,
        height=height,
        warmup=10,
        acceleration=acceleration,
        do_add_noise=do_add_noise,
        enable_similar_image_filter=enable_similar_image_filter,
        similar_image_filter_threshold=similar_image_filter_threshold,
        similar_image_filter_max_skip_frame=similar_image_filter_max_skip_frame,
        mode="img2img",
        use_denoising_batch=use_denoising_batch,
        cfg_type=cfg_type,
        seed=seed,
    )

    stream.prepare(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=50,
        guidance_scale=guidance_scale,
        delta=delta,
    )

    #monitor = monitor_receiver.recv()

    event = threading.Event()


    input_screen = threading.Thread(target=ndi_receiver, args=(event, height, width))
    input_screen.start()
    time.sleep(5)

    while True:
        try:
            if not close_queue.empty(): # closing check
                break

            # Check if there's a new seed
            if not seed_queue.empty():
                new_seed = seed_queue.get(block=False)
                print(f"Updating seed: {new_seed}")
                seed = int(new_seed)  # Ensure it's an integer
                # Create a new generator with the new seed
                new_generator = torch.Generator(device=stream.device).manual_seed(seed)
                # Update the generator in the diffusion stream
                stream.stream.generator = new_generator
                # Optionally reinitialize the initial noise using the new generator
                stream.stream.init_noise = torch.randn(
                    (stream.stream.batch_size, 4, stream.stream.latent_height, stream.stream.latent_width),
                    generator=new_generator,
                    device=stream.device,
                    dtype=stream.dtype
                )


            # Check if there's a new prompt
            if not prompt_queue.empty():
                newPrompt = prompt_queue.get(block=False)
                stream.stream.update_prompt(newPrompt)

            if len(inputs) < frame_buffer_size:
                time.sleep(0.005)
                continue
            start_time = time.time()
            sampled_inputs = []

            for i in range(frame_buffer_size):
                index = (len(inputs) // frame_buffer_size) * i
                sampled_inputs.append(inputs[len(inputs) - index - 1])
                inputs_queue.put(inputs[len(inputs) - index - 1])
            input_batch = torch.cat(sampled_inputs)
            inputs.clear()
            output_images = stream.stream(
                input_batch.to(device=stream.device, dtype=stream.dtype)
            ).cpu()
            if frame_buffer_size == 1:
                output_images = [output_images]
            for output_image in output_images:
                queue.put(output_image, block=False)

            fps = 1 / (time.time() - start_time)
            fps_queue.put(fps)
        except KeyboardInterrupt:
            break

    print("closing image_generation_process...")
    event.set() # stop capture thread
    input_screen.join()
    print(f"fps: {fps}")


def main(
    model_id_or_path: str = "Lykon/dreamshaper-8", #KBlueLeaf/kohaku-v2.1 Lykon/dreamshaper-8 dreamlike-art/dreamlike-photoreal-2.0
    lora_dict: Optional[Dict[str, float]] = None,
    prompt: str = "A portrait of a scary man, dark eyes, evil face, horror, scary, ancient, photography, victorian era, white hair",
    negative_prompt: str = "low quality, bad quality, blurry, low resolution",
    frame_buffer_size: int = 1,
    width: int = 512,
    height: int = 512,
    acceleration: Literal["none", "xformers", "tensorrt"] = "tensorrt",
    use_denoising_batch: bool = True,
    seed: int = 1,
    cfg_type: Literal["none", "full", "self", "initialize"] = "self",
    guidance_scale: float = 1.4,
    delta: float = .7,
    do_add_noise: bool = False,
    enable_similar_image_filter: bool = True,
    similar_image_filter_threshold: float = 0.99,
    similar_image_filter_max_skip_frame: float = 10,
) -> None:
    """
    Main function to start the image generation and viewer processes.
    """
    ctx = get_context('spawn')
    queue = ctx.Queue()
    inputs_queue = ctx.Queue()
    prompt_queue = ctx.Queue()
    fps_queue = ctx.Queue()
    seed_queue = ctx.Queue()
    close_queue = Queue()


    #monitor_sender, monitor_receiver = ctx.Pipe()
    # Check for FPS updates

    process1 = ctx.Process(
        target=image_generation_process,
        args=(
            queue,
            fps_queue,
            close_queue,
            model_id_or_path,
            lora_dict,
            prompt,
            negative_prompt,
            frame_buffer_size,
            width,
            height,
            acceleration,
            use_denoising_batch,
            seed,
            cfg_type,
            guidance_scale,
            delta,
            do_add_noise,
            enable_similar_image_filter,
            similar_image_filter_threshold,
            similar_image_filter_max_skip_frame,
            prompt_queue,
            inputs_queue,
            seed_queue
            ),
    )
    process1.start()

    @app.route('/output_feed')
    def output_feed():
        """ Flask route for MJPEG video streaming. """
        return Response(stream_frames(queue), mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/input_feed')
    def input_feed():
        """ Flask route for MJPEG video streaming. """
        return Response(stream_frames(inputs_queue), mimetype='multipart/x-mixed-replace; boundary=frame')

    # @app.route('/prompt', methods=['POST'])
    # def set_prompt():
    #     if 'prompt' not in request.form:
    #         return 'No prompt given', 400
    #     prompt_queue.put(request.form["prompt"])
    #     print(request.form["prompt"])
    #     return b"new Prompt set", 200

    @app.route('/set_params', methods=['POST'])
    def set_params():
        """API to update prompt and seed dynamically."""
        data = request.form.to_dict()

        if "prompt" in data and data["prompt"].strip():
            prompt_queue.put(data["prompt"])
            print(f"New prompt: {data['prompt']}")

        if "seed" in data:
            try:
                new_seed = int(data["seed"])  # Ensure it's an integer
                seed_queue.put(new_seed)  # Send new seed to the queue
                print(f"New seed: {new_seed}")
            except ValueError:
                return jsonify({"error": "Invalid seed value"}), 400

        return b"Parameters updated", 200  # Simple byte response


    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

    '''
    monitor_process = ctx.Process(
        target=monitor_setting_process,
        args=(
            width,
            height,
            monitor_sender,
            ),
    )
    monitor_process.start()
    monitor_process.join()
    '''

    #process2 = ctx.Process(target=receive_images, args=(queue, fps_queue))
    #process2.start()



    # terminate
    #process2.join()
    print("process2 terminated.")
    close_queue.put(True)
    print("process1 terminating...")
    process1.join(5) # with timeout
    if process1.is_alive():
        print("process1 still alive. force killing...")
        process1.terminate() # force kill...
    process1.join()
    print("process1 terminated.")


if __name__ == "__main__":
    fire.Fire(main)
