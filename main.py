from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from models.sdxl import load_model
from utils.image_utils import build_output_stem, save_generated_images
from utils.logger import setup_logger


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PRODUCTS_FILE = BASE_DIR / "data" / "products.json"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs"
DEFAULT_ENVIRONMENT = "hospital office / workspace environment"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one or more SDXL product images from a JSON file."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_PRODUCTS_FILE),
        help="Path to the products JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where generated images are saved.",
    )
    parser.add_argument(
        "--images-per-product",
        type=int,
        default=1,
        help="Number of images to generate for each product.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=7,
        help="Number of diffusion inference steps.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=3.0,
        help="Guidance scale used by the diffusion pipeline.",
    )
    return parser.parse_args()


def load_products(json_path: Path) -> list[dict[str, Any]]:
    try:
        raw_content = json_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"JSON file not found: {json_path}") from exc
    except OSError as exc:
        raise OSError(f"Unable to read JSON file: {json_path}") from exc

    try:
        payload = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {json_path}: {exc}") from exc

    if isinstance(payload, list):
        products = payload
    elif isinstance(payload, dict):
        products = payload.get("products") or payload.get("entities")
        if products is None:
            raise ValueError(
                "JSON root object must contain a 'products' or 'entities' list."
            )
    else:
        raise ValueError("JSON root must be a list or an object containing a list.")

    if not isinstance(products, list):
        raise ValueError("Products payload must be a list.")

    invalid_entries = [index for index, item in enumerate(products, start=1) if not isinstance(item, dict)]
    if invalid_entries:
        raise ValueError(f"Products at indexes {invalid_entries} are not JSON objects.")

    return products


def _as_clean_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _join_or_default(values: list[str], default: str) -> str:
    return ", ".join(values) if values else default


def _extract_product_type(product: dict[str, Any]) -> str:
    candidates = [
        product.get("product_type"),
        product.get("category"),
        product.get("type"),
        product.get("name"),
    ]
    for candidate in candidates:
        if candidate and str(candidate).strip():
            return str(candidate).strip()
    raise ValueError("Missing required field: product_type/category/name")


def _extract_visual_description(product: dict[str, Any]) -> str:
    visual_description = str(product.get("visual_description", "")).strip()
    if visual_description:
        return visual_description

    description = str(product.get("description", "")).strip()
    dimensions = _join_or_default(_as_clean_list(product.get("dimensions")), "")
    features = _join_or_default(_as_clean_list(product.get("features")), "")

    parts = [part for part in [description, dimensions, features] if part]
    if not parts:
        raise ValueError("Missing required field: visual_description/description")

    return " ".join(parts)


def _extract_material_phrase(product: dict[str, Any]) -> str:
    materials = _as_clean_list(product.get("materials"))
    return _join_or_default(materials, "not specified")


def generate_prompt(product: dict[str, Any]) -> str:
    product_type = _extract_product_type(product)
    environment = str(product.get("environment", DEFAULT_ENVIRONMENT)).strip() or DEFAULT_ENVIRONMENT
    material_phrase = _extract_material_phrase(product)
    visual_description = _extract_visual_description(product)

    return (
        f"A ultra realistic commercial product photography of a {product_type} placed in a real {environment}. "
        f"The product must be physically correct, fully visible, and occupy most of the frame (at least 70%). "
        f"It is NOT symbolic or abstract, it is a real manufactured object with precise industrial design."
        f"\n\nOBJECT DETAILS:\n"
        f"- Materials: {material_phrase}\n"
        f"- Visual structure: {visual_description}\n"
        f"\n\nRENDERING REQUIREMENTS:\n"
        "The object is placed directly on a real floor surface with correct physical contact. "
        "Accurate proportions and engineering realism must be respected. "
        "All components described must be visible (legs, structure, storage, surfaces, edges). "
        "No missing parts, no simplified geometry."
        f"\n\nCAMERA & LIGHTING:\n"
        "Professional furniture photography, 35mm lens, eye-level perspective. "
        "Soft natural daylight mixed with studio lighting. "
        "Realistic shadows under the object. "
        "Shallow depth of field but entire product remains sharp."
        f"\n\nSCENE:\n"
        "Modern real-world interior (hospital office / workspace environment). "
        "No empty studio void. No floating object. No white isolated background. "
        "The environment must feel lived-in and realistic."
        f"\n\nNEGATIVE CONSTRAINTS (IMPORTANT):\n"
        "NO cartoon, NO CGI look, NO abstract rendering, NO minimal empty scene, "
        "NO cropped furniture, NO floating object, NO text, NO watermark, NO people."
        f"\n\nFINAL GOAL:\n"
        "A realistic furniture catalog product image suitable for architectural or commercial use."
    )


def generate_image(
    pipe: Any,
    prompt: str,
    num_images_per_prompt: int,
    num_inference_steps: int,
    guidance_scale: float,
) -> list[Any]:
    result = pipe(
        prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_images_per_prompt,
    )
    return result.images


def get_product_display_name(product: dict[str, Any], fallback_index: int) -> str:
    candidates = [
        product.get("name"),
        product.get("title"),
        product.get("product_type"),
        product.get("category"),
        product.get("code"),
    ]
    for candidate in candidates:
        if candidate and str(candidate).strip():
            return str(candidate).strip()
    return f"product_{fallback_index}"


def build_progress(current: int, total: int, width: int = 24) -> str:
    if total <= 0:
        return "[------------------------] 0/0"
    filled = int(width * current / total)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {current}/{total}"


def run() -> int:
    args = parse_args()
    logger = setup_logger()

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()

    if args.images_per_product < 1:
        logger.error("images-per-product must be at least 1.")
        return 1

    try:
        products = load_products(input_path)
    except Exception as exc:
        logger.error("Unable to load products: %s", exc)
        return 1

    if not products:
        logger.warning("No products found in %s", input_path)
        return 0

    logger.info("Loading SDXL model...")
    pipe = load_model()
    logger.info("Processing %s products from %s", len(products), input_path)

    success_count = 0
    skipped_count = 0

    for index, product in enumerate(products, start=1):
        product_name = get_product_display_name(product, index)
        progress = build_progress(index, len(products))
        logger.info("%s Generating images for '%s'", progress, product_name)

        try:
            prompt = generate_prompt(product)
            images = generate_image(
                pipe=pipe,
                prompt=prompt,
                num_images_per_prompt=args.images_per_product,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
            )
            output_stem = build_output_stem(product, index)
            saved_paths = save_generated_images(images, output_dir, output_stem)
            logger.info(
                "Saved %s image(s) for '%s': %s",
                len(saved_paths),
                product_name,
                ", ".join(str(path) for path in saved_paths),
            )
            success_count += 1
        except ValueError as exc:
            skipped_count += 1
            logger.warning("Skipping product '%s': %s", product_name, exc)
        except Exception:
            skipped_count += 1
            logger.exception("Image generation failed for '%s'", product_name)

    logger.info(
        "Completed batch generation. Success: %s, skipped/failed: %s.",
        success_count,
        skipped_count,
    )
    return 0 if success_count else 1


if __name__ == "__main__":
    raise SystemExit(run())
