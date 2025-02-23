# StreamDiffusion - Workshop ECAL

## Installation
### Creer l'environnement - (j'utilise miniconda)
```
conda create -n streamdiffusion python=3.10
conda activate streamdiffusion
```
### Installer les dep. spécifiques (versions qui marchent)
```bash
pip install torch==2.1.0 torchvision==0.16.0 xformers --index-url https://download.pytorch.org/whl/cu118
pip install numpy==1.26.4
pip install transformers==4.49.0
pip install huggingface-hub==0.29.1
```
### Fix l'import de HF-hub qui marche pas sur l'ancienne version de diffusers
Aller dans les site-packages de l'environnement
[YOUR ENV PATH]/lib/python3.10/site-packages/diffusers/utils/dynamic_modules_utils.py
effacer l'import de *cached_download*
### Installer streamdiffusion and tensorRT
```bash
python setup.py develop easy_install streamdiffusion[tensorrt]
python -m streamdiffusion.tools.install-tensorrt
```
### Start main script
```bash
python main.py
```
<ins>le premier boot va generer l'engine TensorRT donc ça va prendre des plombes, surveiller les log et le load sur la gpu / cpu pour être sur que ça avance</ins>
## Endpoints
### feed d'output - diffusion MJPEG
http://127.0.0.1:5000/output_feed
### feed d'input - passthrough du feed ndi en MJPEG
http://127.0.0.1:5000/input_feed
### [POST] - Form Multipart - paramètre "prompt"
http://127.0.0.1:5000/prompt
```js
const formData = new FormData();

formData.append('prompt', 'A giant castle overlooking a gloomy lake, incredible quality, 4k, photography, unreal engine');

fetch('http://127.0.0.1:5000/prompt', {
  method: 'POST',
  body: formData
})
```
