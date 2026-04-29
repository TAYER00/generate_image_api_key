import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverSinglestepScheduler


def load_model(model_name="sd-community/sdxl-flash"):
    """
    Charge et configure le modèle Stable Diffusion XL.

    Args:
        model_name (str): Nom du modèle Hugging Face.

    Returns:
        pipe: Pipeline prêt à générer des images.

    Raises:
        RuntimeError: Si le modèle ne peut pas être chargé.
    """

    try:
        # 🔍 Détection automatique du device (GPU ou CPU)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Device utilisé : {device}")

        # ⚙️ Choix du type de précision
        # float16 = plus rapide (GPU)
        # float32 = plus stable (CPU)
        dtype = torch.float16 if device == "cuda" else torch.float32

        print("[INFO] Chargement du modèle... (peut prendre du temps la première fois)")

        # 📥 Chargement du modèle depuis Hugging Face
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_name,
            torch_dtype=dtype
        )

        # 🚀 Envoi du modèle vers le device (GPU ou CPU)
        pipe = pipe.to(device)

        # ⚡ Optimisation du scheduler (meilleure vitesse pour SDXL Flash)
        pipe.scheduler = DPMSolverSinglestepScheduler.from_config(
            pipe.scheduler.config,
            timestep_spacing="trailing"
        )

        # 💡 Optimisations mémoire (utile si GPU limité)
        if device == "cuda":
            pipe.enable_attention_slicing()
            print("[INFO] Attention slicing activé (optimisation mémoire)")

        print("[SUCCESS] Modèle chargé avec succès !")

        return pipe

    except ImportError as e:
        print("[ERROR] Problème d'import des librairies.")
        print("👉 Vérifie que torch, diffusers sont installés.")
        raise e

    except OSError as e:
        print("[ERROR] Impossible de télécharger ou charger le modèle.")
        print("👉 Vérifie ta connexion internet ou le nom du modèle.")
        raise e

    except RuntimeError as e:
        print("[ERROR] Problème lié au GPU ou à la mémoire.")
        print("👉 Essaie de passer en CPU ou libérer de la VRAM.")
        raise e

    except Exception as e:
        print("[ERROR] Une erreur inattendue s'est produite.")
        raise e