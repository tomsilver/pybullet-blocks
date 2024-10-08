"""Utility functions."""

import tempfile
from pathlib import Path

import pybullet as p
from PIL import Image, ImageDraw, ImageFont
from pybullet_helpers.utils import create_pybullet_block


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


def create_lettered_block(
    letter: str,
    half_extents: tuple[float, float, float],
    face_rgba: tuple[float, float, float, float],
    text_rgba: tuple[float, float, float, float],
    physics_client_id: int,
) -> int:
    """Create a block with a letter on all sides."""
    block_id = create_pybullet_block(
        (1, 1, 1, 1),  # NOTE: important to default to white for texture
        half_extents=half_extents,
        physics_client_id=physics_client_id,
    )
    text_color = tuple(map(lambda x: int(255 * x), text_rgba))
    background_color = tuple(map(lambda x: int(255 * x), face_rgba))
    filepath = create_texture_with_letter(
        letter,
        text_color=text_color,
        background_color=background_color,
    )
    texture_id = p.loadTexture(str(filepath), physicsClientId=physics_client_id)
    p.changeVisualShape(
        block_id,
        -1,
        textureUniqueId=texture_id,
        physicsClientId=physics_client_id,
    )
    return block_id
