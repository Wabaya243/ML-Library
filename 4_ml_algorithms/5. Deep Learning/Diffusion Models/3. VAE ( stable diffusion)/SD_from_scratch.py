import torch, logging
from PIL import Image
from torchvision import transforms as tfms
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from IPython.display import display, HTML, clear_output
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import xformers

# Désactiver les warnings inutiles
logging.disable(logging.WARNING)

# Import des modèles CLIP et du diffuseur Stable Diffusion
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

# Crée un dossier pour sauvegarder les images intermédiaires
os.makedirs("steps2", exist_ok=True)
os.makedirs("Data", exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Charger l'image
def load_images(path):
    return Image.open(path).convert("RGB").resize((512, 512))

# Convertir un PIL Image en latent VAE
def pil_latent(image):
    """
    VAE produit une distribution de latents, et on en prélève un échantillon.
    En clair : on traduis l'image (512×512×3) en une “capsule d’information visuelle” (4×64×64).
    C’est l’entrée naturelle du U-Net, car la diffusion opère dans cet espace compressé.
    """
    init_image = tfms.ToTensor()(image).unsqueeze(0) * 2.0 - 1.0
    init_image = init_image.to(device, dtype=torch.float16)
    with torch.no_grad():
        init_latent = vae.encode(init_image).latent_dist.sample() * 0.18215 #On multiplie le résultat par 0.18215, un facteur d’échelle propre à Stable Diffusion (normalisation des latents).
    return init_latent

# Convertir une liste de latents en images PIL
def latent_to_pil(latents):
    """
    Fait l’inverse : reconvertit des latents en images visibles
    On multiplie le résultat par 0.18215, un facteur d’échelle propre à Stable Diffusion (normalisation des latents).
    """
    latents = (1 / 0.18215) * latents
    with torch.no_grad():
        image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(im) for im in images]
    return pil_images

# Encoder les textes en embeddings via CLIP
def text_enc(prompts, maxlen=None):
    '''
    encode les prompts texte avec CLIP, le modèle de vision-langage.
    prompts (List[str]) : liste d'invites textuelles à encoder. 
    maxlen (int, facultatif) : longueur maximale du texte encodé. 
    Le CLIPTextModel les convertit en un tenseur d’embeddings (taille 77×768).
    '''
    if maxlen is None:
        maxlen = tokenizer.model_max_length
    inp = tokenizer(prompts, padding="max_length", max_length=maxlen, truncation=True, return_tensors="pt")
    return text_encoder(inp.input_ids.to(device))[0].half()

# Convertir un prompt texte en image via le modèle de diffusion
'''
prompts (List[str]) : Textes à convertir en images. 
g (float) : Échelle de guidage. Des valeurs élevées renforcent le respect du texte. 
seed (int) : Valeur aléatoire pour la génération d'images. 
steps (int) : Nombre d'étapes de diffusion. 
dim (int) : Dimension des images générées. 
save_int (bool) : Enregistrement des images intermédiaires.
'''

def prompt_2_img(prompts, g=7.5, seed=42, steps=70, dim=512, save_int=True):
    bs = len(prompts)

    # Encoder le texte
    text = text_enc(prompts)
    uncond = text_enc([""] * bs, text.shape[1]) #uncond est le même encodeur, mais sur un texte vide ("").
    emb = torch.cat([uncond, text]) # On concatène les deux : [embedding_vide, embedding_prompt].
    #une technique qui aide le modèle à mieux coller au texte,
    #en comparant ce qu’il génère avec texte et sans texte.

    # Graine
    if seed:
        torch.manual_seed(seed)

    # Latents initiaux (bruit)
    channels = unet.config.in_channels          # = 4 pour Stable Diffusion
    latents = torch.randn((bs, channels, dim // 8, dim // 8), device=device, dtype=torch.float16) #dim//8 = réduction spatiale du VAE (ex : 512→64).
    scheduler.set_timesteps(steps) #définit la courbe de débruitage (combien d’étapes, comment le bruit décroît).
    latents *= scheduler.init_noise_sigma #On multiplie par scheduler.init_noise_sigma pour ajuster la variance.
    #C’est la “toile blanche” sur laquelle le modèle va peindre.

    print("Préprocessing du prompt :", prompts)
    print("Visualisation du latent initial :")
    latents_norm = torch.norm(latents.view(latents.shape[0], -1), dim=1).mean().item()
    print("Latent initial normalisé :", latents_norm)

    # Boucle de débruitage
    for i, ts in enumerate(tqdm(scheduler.timesteps)):
        # Mise à l’échelle des latents
        inp = scheduler.scale_model_input(torch.cat([latents] * 2), ts)

        # Prédiction du bruit résiduel par le U-Net
        with torch.no_grad():
            """
            On duplique le latent ([latents, latents]) pour calculer à la fois conditionné et non-conditionné.
            Le U-Net prédit le bruit à retirer.
            u → prédiction sans texte.
            t → prédiction avec texte.
            """
            u, t = unet(inp, ts, encoder_hidden_states=emb).sample.chunk(2)

        # Exécution du guidage
        pred = u + g * (t - u) # On combine les deux via la formule :


        # Mise à jour des latents
        latents = scheduler.step(pred, ts, latents).prev_sample #scheduler.step() met à jour le latent (retire un peu de bruit).
        #Ce cycle est répété steps fois, typiquement entre 20 et 100.

        # Normalisation intermédiaire
        latents_norm = torch.norm(latents.view(latents.shape[0], -1), dim=1).mean().item()
        print(f"Étape {i+1}/{steps} - Norme des latents : {latents_norm:.4f}")

        # Sauvegarde et affichage
        if save_int and (i % 10 == 0 or i == steps - 1):
            image_path = f"steps2/la_{i:04d}.jpeg"
            img = latent_to_pil(latents)[0] #Convertit les latents actuels en image via latent_to_pil().
            img.save(image_path) #Sauvegarde le rendu intermédiaire.
            clear_output(wait=True)
            display(img) #Affiche le résultat (utile pour suivre la progression du débruitage).

    return latent_to_pil(latents)



def traduire_texte(texte_fr, src_lang="fr_XX", tgt_lang="en_XX"):
    # Spécifier la langue source pour le tokenizer
    translator_tokenizer.src_lang = src_lang
    
    # Tokeniser le texte français
    inputs = translator_tokenizer(texte_fr, return_tensors="pt").to(device)
    
    # Forcer le token de début de phrase pour la langue cible (important !)
    forced_bos_token_id = translator_tokenizer.lang_code_to_id[tgt_lang]
    
    # Générer la traduction
    outputs = translator_model.generate(
        **inputs,
        forced_bos_token_id=forced_bos_token_id,
        max_length=128,
        num_beams=5,
        early_stopping=True,
    )
    
    # Décoder et retourner la traduction
    traduction = translator_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return traduction


"""
Le modèle de diffusion stable utilise l'entrée textuelle et une graine.
L'entrée textuelle est ensuite transmise au modèle CLIP pour générer un embedding textuel de taille 77x768. 
Cette graine est utilisée pour générer un bruit gaussien de taille 4x64x64, qui constitue la première représentation d'image latente.
L'U-Net débruite ensuite itérativement les représentations d'image latentes aléatoires en conditionnant les embeddings textuels.
Le résultat de l'U-Net est un résidu de bruit prédit, qui est ensuite utilisé pour calculer les latentes conditionnées via un algorithme d'ordonnancement.
Ce processus de débruitage et de conditionnement textuel est répété N fois (50 fois) afin d'obtenir une meilleure représentation d'image latente.
Une fois ce processus terminé, la représentation d'image latente (4x64x64) est décodée par le décodeur VAE pour obtenir l'image de sortie finale (3x512x512).
"""

# Initialisation des composants du pipeline Stable Diffusion
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14") #Prend le texte et le transforme en vecteurs (embeddings).
#Ce modèle CLIP (openai/clip-vit-large-patch14) a été entraîné à relier texte ↔ image.
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device) #Stable Diffusion s’en sert pour comprendre ce que tu veux générer :

# Le VAE encode une image 512×512 en une représentation latente 4×64×64.
vae = AutoencoderKL.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    subfolder="vae",
    torch_dtype=torch.float16
).to(device)

# Scheduler : planning du bruit
scheduler = LMSDiscreteScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    num_train_timesteps=1000
)
scheduler.set_timesteps(50) #→ il va faire 50 itérations de débruitage (au lieu des 1000 d’origine).


#spécialement entraîné à apprendre “comment enlever le bruit pour obtenir une image conforme au texte”.
unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet", torch_dtype=torch.float16).to(device)
unet.enable_xformers_memory_efficient_attention()

# Charger le modèle multilingue mBART (gère plusieurs langues)
model_name = "facebook/mbart-large-50-many-to-many-mmt"

translator_tokenizer  = AutoTokenizer.from_pretrained(model_name, use_fast=True)
translator_model  = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

# Exemple d'utilisation
prompt_en = traduire_texte("un chat portant un grand chapeau haut-de-forme sur la tête, pas une cravate, détaillé, réaliste, 4k")

# Exemple de génération
images = prompt_2_img(prompts=[prompt_en], g=10, save_int=True)
for img in images:
    display(img)

# Visualisation des étapes sauvegardées
def visualize_steps(folder="steps2"):
    files = sorted([f for f in os.listdir(folder) if f.lower().endswith(".jpeg")])
    if not files:
        print("Aucune image trouvée dans le dossier.")
        return
    num_steps = len(files)
    fig, axs = plt.subplots(1, num_steps, figsize=(20, 10))
    if num_steps == 1:
        axs = [axs]
    for i, (ax, file) in enumerate(zip(axs, files)):
        img = plt.imread(os.path.join(folder, file))
        ax.imshow(img)
        ax.axis("off")
        if i < num_steps - 1:
            ax.arrow(0.9, 0.5, 0.15, 0, head_width=0.1, head_length=0.05,
                     fc="k", ec="k", transform=ax.transAxes, clip_on=False)
    plt.tight_layout()
    plt.show()

visualize_steps()
