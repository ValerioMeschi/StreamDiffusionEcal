import os
import sys
import time
import threading
from multiprocessing import Process, Queue, get_context, shared_memory
from multiprocessing.connection import Connection
from typing import List, Literal, Dict, Optional
import torch
from PIL import Image, ImageDraw, ImageFont, ImageTk
import io

from transformers.image_transforms import NumpyToTensor
from streamdiffusion.image_utils import pil2tensor

from utils.shared_mem import create_shared_float, access_shared_float, cleanup_shared_float

import fire
import NDIlib as ndi
import numpy as np
from flask import Flask, Response, request
from flask_socketio import SocketIO
import base64
from collections import deque
import tkinter as tk
import cv2
import random



app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

from utils.viewer import receive_images
from utils.wrapper import StreamDiffusionWrapper
from streamdiffusion.image_utils import postprocess_image

top = 0
left = 0
use_ndi = False

frame_shape = (512, 512, 3)
frame_dtype = np.uint8

#sleep time between frames
sleep_time = 0.001

def file_to_array(file):
    # Read image from the uploaded file
    img = Image.open(io.BytesIO(file.read()))

    # Resize to 512x512
    img = img.resize((512, 512))

    # Convert to RGB if not already
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Convert to numpy array and ensure uint8 type
    img_array = np.array(img, dtype=np.uint8)

    return img_array

def tensor_to_rgb(tensor):
    # Convert to numpy and transpose dimensions to [H, W, C]
    rgb = tensor.detach().cpu().numpy()
    rgb = np.transpose(rgb, (1, 2, 0))

    # If values are in [0, 1], convert to [0, 255]
    if rgb.max() <= 1.0:
        rgb = (rgb * 255).astype(np.uint8)

    return rgb

def write_text_to_image(image, text):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    w, h = image.size
    bbox = draw.textbbox((0, 0), text, font=font)
    draw.text(((w-bbox[2])/2, (h-bbox[3])/2), text, font=font, fill='white')
    return image




# methodes pour streamer les frame en sortie. A threader dans main
def stream_frames(shared_name, convert=True):
    """ Continuously yields frames from the queue for MJPEG streaming. """
    shm = shared_memory.SharedMemory(name=shared_name)
    shared_frame = np.ndarray(frame_shape, dtype=frame_dtype, buffer=shm.buf)
    while True:
        frame = shared_frame.copy()  # Make a copy if needed
        pImage = Image.fromarray(frame)
        img_io = io.BytesIO()
        pImage.save(img_io, format="JPEG")
        img_io.seek(0)
        img_bytes = img_io.read()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + img_bytes + b'\r\n')

        time.sleep(sleep_time)


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

def ndi_receiver(height: int,
width: int, shared_frame_name):

    print("Starting NDI receiver thread...")

    if not ndi.initialize():
        return 0

    ndi_find = ndi.find_create_v2()

    if ndi_find is None:
        return 0

    sources = []
    while not sources:
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

    shm = shared_memory.SharedMemory(name=shared_frame_name)
    shared_frame = np.ndarray(frame_shape, dtype=frame_dtype, buffer=shm.buf)
    while True:
        try:
            t, v, _, _ = ndi.recv_capture_v2(ndi_recv, 5000)
            if t == ndi.FRAME_TYPE_VIDEO:
                frame = np.copy(v.data)
                frame = frame[...,:3]
                norm_frame = normalize(frame).astype('uint8')
                shared_frame[:] = norm_frame[:]
                ndi.recv_free_video_v2(ndi_recv, v)
                time.sleep(sleep_time)
        except KeyboardInterrupt:
            ndi.recv_destroy(ndi_recv)
            ndi.destroy()
            break
    return 0


def image_generation_process(
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
    shared_input_frame_name: str,
    shared_output_frame_name: str,
    shared_delta_name: str,
    shared_seed_name: str,
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

    global app

    print("Starting image generation thread..")

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

    shmIn = shared_memory.SharedMemory(name=shared_input_frame_name)
    shared_input_frame = np.ndarray(frame_shape, dtype=frame_dtype, buffer=shmIn.buf)

    shmOut = shared_memory.SharedMemory(name=shared_output_frame_name)
    shared_output_frame = np.ndarray(frame_shape, dtype=frame_dtype, buffer=shmOut.buf)

    shared_delta_array, shm_delta = access_shared_float(shared_delta_name)

    shared_seed_array, shm_seed = access_shared_float(shared_seed_name)


    time.sleep(2)

    last_seed_arr = shared_seed_array[:]

    while True:

        try:
            if not close_queue.empty(): # closing check
                break

            # Check if there's a new seed
            if shared_seed_array[0] != -1:
                #print("newSeed !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                seed = int(shared_seed_array[0])
                new_generator = torch.Generator(device=stream.device).manual_seed(seed)
                # Update the generator in the diffusion stream
                stream.stream.generator = new_generator
                shared_seed_array[0] = -1

            #update Delta
            stream.stream.delta = shared_delta_array[0]

            # Check if there's a new prompt
            if not prompt_queue.empty():
                newPrompt = prompt_queue.get(block=False)
                stream.stream.update_prompt(newPrompt)

            # Get input frame from shared memory and preprocess it
            new_frame = shared_input_frame.copy()
            tensor = torch.as_tensor(new_frame).permute(2, 0, 1)
            tensor = tensor.float() / 255.0
            input_batch = torch.cat([tensor])
            output_image = stream.stream(input_batch.to(device=stream.device, dtype=stream.dtype)).cpu()
            #process output and write to shared memory
            pp = postprocess_image(output_image, output_type="pt")[0]
            out_frame = tensor_to_rgb(pp);
            shared_output_frame[:] = out_frame[:]


            time.sleep(sleep_time)
        except KeyboardInterrupt:
            break

    print("closing image_generation_process...")


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
    guidance_scale: float = 1.3,
    delta: float = .8,
    do_add_noise: bool = True,
    enable_similar_image_filter: bool = False,
    similar_image_filter_threshold: float = 0.99,
    similar_image_filter_max_skip_frame: float = 10,
) -> None:
    """
    Main function to start the image generation and viewer processes.
    """
    ctx = get_context('spawn')
    prompt_queue = ctx.Queue()
    fps_queue = ctx.Queue()
    seed_queue = ctx.Queue()
    close_queue = Queue()


    shared_delta_name = create_shared_float(delta)
    shared_seed_name = create_shared_float(float(seed))

    in_frame = np.zeros(frame_shape, dtype=frame_dtype)

    shared_input_frame = shared_memory.SharedMemory(create=True, size=in_frame.nbytes)
    shared_output_frame = shared_memory.SharedMemory(create=True, size=in_frame.nbytes)


    shared_output_frame_buffer = np.ndarray(frame_shape, dtype=frame_dtype, buffer=shared_output_frame.buf)
    shared_input_frame_buffer = np.ndarray(frame_shape, dtype=frame_dtype, buffer=shared_input_frame.buf)

    # Empty image for no signal
    empty_image = write_text_to_image(Image.new('RGB', (512, 512), color='black'), "no signal")
    empty_image_arr = np.array(empty_image, dtype=np.uint8)
    # put the empty image in memory
    shared_output_frame_buffer[:] = empty_image_arr[:]
    shared_input_frame_buffer[:] = empty_image_arr[:]

    p1 = ctx.Process(
        target=image_generation_process,
        args=(
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
            shared_input_frame.name,
            shared_output_frame.name,
            shared_delta_name,
            shared_seed_name,
            ),
    )
    p1.start()

    #start NDI Process
    p2 = ctx.Process(target=ndi_receiver, args=(512,512,shared_input_frame.name))
    p2.start();

    @socketio.on('message')
    def handle_image(data):
        if data.startswith('data:image'):
            base64_data = data.split(',')[1]
            image_data = base64.b64decode(base64_data)
            imagefile =io.BytesIO(image_data)
            new_frame = file_to_array(imagefile)
            shared_input_frame_buffer[:] = new_frame[:]
            print("received")

    @app.route("/upload", methods=['POST'])
    def output_img():
        image_file = request.files['image']  # Get image from request
        if image_file:
            new_frame = file_to_array(image_file)
            shared_input_frame_buffer[:] = new_frame[:]
            return {"status": "success"}
        else:
            return {"status": "error", "message": "No image received"}

    @app.route('/output_feed')
    def output_feed():
        """ Flask route for MJPEG video streaming. """
        return Response(stream_frames(shared_output_frame.name), mimetype='multipart/x-mixed-replace; boundary=frame')


    @app.route('/input_feed')
    def input_feed():
        """ Flask route for MJPEG video streaming. """
        return Response(stream_frames(shared_input_frame.name), mimetype='multipart/x-mixed-replace; boundary=frame')

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
                shared_seed_arr, shm = access_shared_float(shared_seed_name)
                shared_seed_arr[0] = new_seed
                print(f"New seed: {new_seed}")
            except ValueError:
                return jsonify({"error": "Invalid seed value"}), 400

        if "delta" in data:
            try:
                new_delta = float(data["delta"])  # Ensure it's a float
                shared_delta_arr, shm = access_shared_float(shared_delta_name)
                shared_delta_arr[0] = new_delta
                print(f"New delta: {new_delta}")
            except ValueError:
                return jsonify({"error": "Invalid delta value"}), 400

        return b"Parameters updated", 200  # Simple byte response


    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

    # terminate
    #process2.join()
    print("process2 terminated.")
    close_queue.put(True)
    print("process1 terminating...")
    try:
        p1.join(5)
        p2.join(5)
    finally:
        # Clean up
        shared_input_frame.close()
        shared_input_frame.unlink()
        shared_output_frame.close()
        shared_output_frame.unlink()
        cleanup_shared_float(shared_delta_name)
        cleanup_shared_float(shared_seed_name)
        if p1.is_alive():
            print("process1 still alive. force killing...")
            p1.terminate() # force kill...
        p1.join()
        print("process1 terminated.")


if __name__ == "__main__":
    fire.Fire(main)
