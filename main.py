from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from models.dalle import generate_image, get_client
from utils.image_utils import build_output_stem, download_image
from utils.logger import setup_logger


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PRODUCTS_FILE = BASE_DIR / "data" / "products.json"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs"
DEFAULT_ENVIRONMENT = "hospital office / workspace environment"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one or more DALL-E 3 product images from a JSON file."
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
        help="Number of images to generate for each product via DALL-E 3.",
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
    material_phrase = _extract_material_phrase(product)
    visual_description = _extract_visual_description(product)

    return (
        f"Ultra realistic isolated product photography of a {product_type}. "
        f"The product is centered, fully visible, and occupies most of the image."

        f"\n\nPRODUCT DESCRIPTION:\n"
        f"- Materials: {material_phrase}\n"
        f"- Structure: {visual_description}\n"

        f"\n\nRENDER STYLE:\n"
        "Pure white seamless studio background. "
        "No environment, no room, no context, no furniture scene. "
        "Product only, isolated like an e-commerce catalog image. "
        "Soft studio lighting with minimal natural shadow under the product. "
        "High detail, sharp focus, professional commercial photography."

        f"\n\nSTRICT NEGATIVE CONSTRAINTS:\n"
        "No background objects, no room, no office, no interior, "
        "no furniture scene, no decoration, no people, no text, "
        "no watermark, no logo, no blur, no CGI, no cartoon style."

        f"\n\nFINAL GOAL:\n"
        "Pure e-commerce product image on white background for catalog use."
    )

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

    try:
        client = get_client()
    except Exception as exc:
        logger.error("Unable to initialize OpenAI client: %s", exc)
        return 1

    logger.info("Processing %s products from %s with DALL-E 3", len(products), input_path)

    success_count = 0
    skipped_count = 0

    for index, product in enumerate(products, start=1):
        product_name = get_product_display_name(product, index)
        progress = build_progress(index, len(products))
        logger.info("%s Generating images for '%s'", progress, product_name)

        try:
            prompt = generate_prompt(product)
            output_stem = build_output_stem(product, index)
            saved_paths = []
            for image_index in range(1, args.images_per_product + 1):
                image_url = generate_image(prompt=prompt, client=client)
                image_stem = output_stem if args.images_per_product == 1 else f"{output_stem}_{image_index}"
                saved_paths.append(download_image(image_url, output_dir, image_stem))
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
