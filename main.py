from models.sdxl import load_model
import os

# 📁 إنشاء dossier outputs إلا ما كانش
os.makedirs("outputs", exist_ok=True)

# 🧠 تحميل الموديل
pipe = load_model()

# ✍️ prompt
prompt = """A stunning blonde woman on a sunny beach, wearing an elegant and attractive swimsuit, posing confidently in a seductive and captivating stance, soft golden sunlight highlighting her silhouette, gentle ocean waves behind her, wind flowing through her hair, cinematic lighting, high fashion editorial style, glamorous atmosphere, ultra-realistic, 4k detail, shallow depth of field"""





print("[INFO] Génération de 3 images en une seule fois...")

# 🎨 توليد 3 صور مرة واحدة
images = pipe(
    prompt,
    num_inference_steps=7,
    guidance_scale=3,
    num_images_per_prompt=3
).images

# 💾 حفظ الصور
for i, image in enumerate(images):
    image_path = f"outputs/output_{i+1}.png"
    image.save(image_path)
    print(f"[SUCCESS] Image sauvegardée : {image_path}")

print("🎉 Toutes les images sont générées !")