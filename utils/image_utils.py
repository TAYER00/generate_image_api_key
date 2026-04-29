from __future__ import annotations

import re
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
