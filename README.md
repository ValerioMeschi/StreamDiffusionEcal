# StreamDiffusion

#### Workshop ECAL 2025

#### Valerio Meschi - Matthieu Minguet

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
python setup.py develop easy_install streamdiffusion[tensorrt] # Nécessaire pour l'étape suivante
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

http://127.0.0.1:5000/set_params

**Request Type:** `POST`
**Content-Type:** `multipart/form-data`

#### Parameters:

- `prompt` _(optional)_ - The text prompt for image generation.
- `seed` _(required)_ - The seed for randomization (integer).

#### Example (JavaScript):

```js
const formData = new FormData();

// Add prompt (optional)
formData.append(
  "prompt",
  "A giant castle overlooking a gloomy lake, incredible quality, 4k, photography, unreal engine"
);

// Add seed (required)
formData.append("seed", 12345);

fetch("http://127.0.0.1:5000/set_params", {
  method: "POST",
  body: formData,
})
  .then((response) => response.text())
  .then((data) => console.log("Response:", data))
  .catch((error) => console.error("Error:", error));
```

## Frontend Stream Diffusion Viewer

Exemple de frontend qui utilise l'API.

```bash
# Ouvrir une nouvelle fenêtre dans le terminal
# Pas nécessaire d'être dans la venv
cd frontend
npm i
npm run dev
```

## Changer le Modèle

Tous les modèles SD1.5 compatibles avec StableDiffusionPipeline de diffusers, disponibles sur Hugging Face, peuvent être utilisés avec TensorRT après optimisation.

### 1. Trouver un Modèle Compatible

Sur [Hugging Face](https://huggingface.co/), recherchez des modèles Stable Diffusion 1.5 compatibles avec diffusers et optimisés pour TensorRT en filtrant avec les tags sd-1.5, diffusers, et tensorrt.

Exemple:
https://huggingface.co/Lykon/dreamshaper-8

### 2. Modifier le Modèle dans le Script

Dans le fichier `main.py`, trouvez le paramètre `model_id_or_path` et remplacez-le par l'identifiant du modèle Hugging Face :

```python
def main(
      model_id_or_path: str = "Lykon/dreamshaper-8", #KBlueLeaf/kohaku-v2.1 Lykon/dreamshaper-8 dreamlike-art/dreamlike-photoreal-2.0

    ...
):
```
