from __future__ import annotations

import re
from urllib.parse import urlparse
from urllib.request import urlretrieve
from pathlib import Path
from typing import Any, Iterable


def ensure_directory(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "product"


def build_output_stem(product: dict[str, Any], index: int) -> str:
    name = (
        product.get("name")
        or product.get("title")
        or product.get("product_type")
        or product.get("category")
        or f"product_{index}"
    )
    code = str(product.get("code", "")).strip()
    stem = slugify(str(name))
    if code:
        stem = f"{stem}-{slugify(code)}"
    return stem


def save_generated_images(
    images: Iterable[Any],
    output_dir: Path,
    output_stem: str,
) -> list[Path]:
    ensure_directory(output_dir)

    saved_paths: list[Path] = []
    for image_index, image in enumerate(images, start=1):
        suffix = "" if image_index == 1 else f"_{image_index}"
        output_path = output_dir / f"{output_stem}{suffix}.png"
        image.save(output_path)
        saved_paths.append(output_path)

    return saved_paths


def download_image(image_url: str, output_dir: Path, output_stem: str) -> Path:
    if not image_url or not image_url.strip():
        raise ValueError("Image URL must not be empty.")

    ensure_directory(output_dir)

    parsed_url = urlparse(image_url)
    suffix = Path(parsed_url.path).suffix or ".png"
    output_path = output_dir / f"{output_stem}{suffix}"

    try:
        urlretrieve(image_url, output_path)
    except Exception as exc:
        raise RuntimeError(f"Unable to download image from URL: {image_url}") from exc

    return output_path
