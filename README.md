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

## Autres paramètres par défaut à configurer avant le chargement du modèle :

- prompt: Texte du prompt de base pour générer l'image.
- negative_prompt: Prompt négatif pour exclure certains aspects indésirables.
- width, height: Dimensions de l'image générée.
- use_denoising_batch: Active le débruitage par batch.
- seed: Valeur de seed pour la génération aléatoire (changer pour obtenir des variations).
- cfg_type: Type de guidance (ex: "self" pour img2img, "none" pour txt2img).
- guidance_scale: Facteur de guidance (CFG scale) ; pour img2img, typiquement entre 1.0 et 2.5.
- delta: Multiplicateur du bruit résiduel virtuel.
- do_add_noise: Active l'ajout de bruit pour les étapes de débruitage.
- enable_similar_image_filter: Active le filtre pour éviter des images trop similaires.
- similar_image_filter_threshold: Seuil du filtre (valeur entre 0 et 1).
- similar_image_filter_max_skip_frame: Nombre maximum de frames à sauter avec le filtre.

```python
def main(
    model_id_or_path: str = "Lykon/dreamshaper-8",  # Modèle SD1.5 (ex: "Lykon/dreamshaper-8")
    lora_dict: Optional[Dict[str, float]] = None,   # Dictionnaire des LoRA avec leur échelle (optionnel)
    prompt: str = "A portrait of a scary man, dark eyes, evil face, horror, scary, ancient, photography, victorian era, white hair",  # Prompt principal
    negative_prompt: str = "low quality, bad quality, blurry, low resolution",  # Prompt négatif
    frame_buffer_size: int = 1,  # Nombre d'images traitées en batch
    width: int = 512,            # Largeur de l'image générée
    height: int = 512,           # Hauteur de l'image générée
    acceleration: Literal["none", "xformers", "tensorrt"] = "tensorrt",  # Méthode d'accélération
    use_denoising_batch: bool = True,  # Active le débruitage par batch
    seed: int = 1,             # Seed pour la génération (changer pour varier le résultat)
    cfg_type: Literal["none", "full", "self", "initialize"] = "self",  # Type de guidance (ex: "self" pour img2img)
    guidance_scale: float = 1.4,  # Facteur de guidance (CFG scale), typiquement entre 1.0 et 2.5 pour img2img
    delta: float = .7,         # Multiplicateur du bruit résiduel virtuel
    do_add_noise: bool = False,  # Active l'ajout de bruit pour le débruitage
    enable_similar_image_filter: bool = True,  # Active le filtre d'images similaires
    similar_image_filter_threshold: float = 0.99,  # Seuil du filtre similaire (entre 0 et 1)
    similar_image_filter_max_skip_frame: float = 10,  # Nombre maximal de frames à sauter
) -> None:
    # ... le reste du code ...
```
