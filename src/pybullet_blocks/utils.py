"""Utility functions."""

import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def create_texture_with_letter(
    letter: str,
    tile_size: tuple[int, int] = (64, 64),
    grid_size: tuple[int, int] = (4, 4),
    font_size: int = 64,
    text_color: tuple[int, ...] = (0, 0, 0, 255),
    background_color: tuple[int, ...] = (255, 255, 255, 255),
) -> Path:
    """Create a texture with a letter on it that can be used on blocks."""
    texture_size = (tile_size[0] * grid_size[0], tile_size[1] * grid_size[1])
    img = Image.new("RGB", texture_size, color=background_color)
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default(font_size)
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            x = i * tile_size[0]
            y = j * tile_size[1]
            text_x = x + tile_size[0] // 2
            text_y = y + tile_size[1] // 2
            d.text((text_x, text_y), letter, font=font, fill=text_color, anchor="mm")
    filepath = Path(tempfile.NamedTemporaryFile("w", delete=False, suffix=".jpg").name)
    img = img.rotate(90)
    img.save(filepath)
    return filepath
