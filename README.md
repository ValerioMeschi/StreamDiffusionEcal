# StreamDiffusion - Workshop ECAL

## Installation

### Creer l'environnement

#### Avec Miniconda

```
conda create -n streamdiffusion python=3.10
conda activate streamdiffusion
```

#### Avec Venv (Windows)

```
# Vérifier la version de python (3.10.*)
python --version
# Installer python 3.10 https://www.python.org/downloads/release/python-3100/
# Cocher l'option "Add to path" au moment de l'installation

Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

python -m venv streamdiffusion-env

# Si plusieurs version de Pythons installées
py -3.10 -m venv streamdiffusion-env

# Activer la venv à chaque ouverture du terminal
streamdiffusion-env\Scripts\activate
```

### Installer les dep. spécifiques (versions qui marchent)

```bash
Set-ExecutionPolicy Bypass -Scope Process -Force # Windows

python -m pip install --upgrade pip
pip install cuda-python
pip install torch==2.1.0 torchvision==0.16.0 xformers --index-url https://download.pytorch.org/whl/cu118
pip install numpy==1.26.4
pip install transformers==4.49.0
pip install huggingface-hub==0.29.1
python setup.py develop easy_install streamdiffusion[tensorrt]
```

### Fix l'import de HF-hub qui marche pas sur l'ancienne version de diffusers

Aller dans les site-packages de l'environnement
[YOUR ENV PATH]/lib/python3.10/site-packages/diffusers/utils/dynamic\*modules_utils.py
[YOUR ENV PATH]\Lib\site-packages\diffusers-0.24.0-py3.10.egg\diffusers\utils\dynamic_modules_utils.py" # Windows

effacer l'import de \_cached_download\*

```python
from huggingface_hub import HfFolder, cached_download, hf_hub_download, model_info # Before
from huggingface_hub import HfFolder, hf_hub_download, model_info # After
```

### Installer streamdiffusion and tensorRT

```bash
python setup.py develop easy_install streamdiffusion[tensorrt]
python -m streamdiffusion.tools.install-tensorrt
```

### Installer les dépendances du projet

```bash
pip install -r .\requirements.txt
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

formData.append(
  "prompt",
  "A giant castle overlooking a gloomy lake, incredible quality, 4k, photography, unreal engine"
);

fetch("http://127.0.0.1:5000/prompt", {
  method: "POST",
  body: formData,
});
```

## Frontend

Exemple de frontend qui utilise l'API.

```bash
# Ouvrir une nouvelle fenêtre dans le terminal
# Pas nécessaire d'être dans la venv
cd frontend
npm i
npm run dev
```
