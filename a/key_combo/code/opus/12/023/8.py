from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

TILE_ROWS = 4
TILE_COLS = 4
SEPARATOR_WIDTH = 1
SOLID_COLORS = (BLUE, RED, GREEN, YELLOW)
NOISE_COLORS = (BLACK, BLUE, RED, GREEN, YELLOW, 6, 8, 9)
MAX_GENERATION_ATTEMPTS = 400

# Canonical 0-degree sprite. Rotations of this mask encode the output rotation.
CANONICAL_SPRITE = np.array(
    [
        [1, 0],
        [1, 0],
        [1, 1],
    ],
    dtype=bool,
)


def _bands(length: int, separators: list[int]) -> list[tuple[int, int]]:
    start = 0
    bands = []
    for cut in separators:
        assert start < cut, "separator lines must leave non-empty tiles"
        bands.append((start, cut))
        start = cut + 1
    assert start < length, "separator lines must leave a final tile band"
    bands.append((start, length))
    return bands


def _find_tile_spans(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    sep_rows = [r for r in range(h) if np.all(g[r, :] == GREY)]
    sep_cols = [c for c in range(w) if np.all(g[:, c] == GREY)]
    assert len(sep_rows) == TILE_ROWS - 1 and len(sep_cols) == TILE_COLS - 1

    row_bands = _bands(h, sep_rows)
    col_bands = _bands(w, sep_cols)
    row_sizes = {r1 - r0 for r0, r1 in row_bands}
    col_sizes = {c1 - c0 for c0, c1 in col_bands}
    assert len(row_bands) == TILE_ROWS and len(col_bands) == TILE_COLS
    assert len(row_sizes) == 1 and len(col_sizes) == 1
    return row_bands, col_bands


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a non-empty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _solid_color(tile: np.ndarray) -> int | None:
    color = int(tile[0, 0])
    if color in (BLACK, GREY):
        return None
    if np.all(tile == color):
        return color
    return None


def _sprite_rotation_steps(tile: np.ndarray) -> int:
    orange_mask = tile == ORANGE
    assert orange_mask.any(), "sprite tile must contain orange pixels"
    cropped = _crop_mask(orange_mask)
    for steps in range(4):
        rotated = np.rot90(CANONICAL_SPRITE, k=-steps)
        if cropped.shape == rotated.shape and np.array_equal(cropped, rotated):
            return steps
    raise AssertionError("sprite orientation did not match any allowed rotation")


def _extract_payload(grid: np.ndarray) -> tuple[np.ndarray, int]:
    row_bands, col_bands = _find_tile_spans(grid)

    solid_tiles: dict[tuple[int, int], int] = {}
    sprite_tiles: list[np.ndarray] = []
    for tr, (r0, r1) in enumerate(row_bands):
        for tc, (c0, c1) in enumerate(col_bands):
            tile = grid[r0:r1, c0:c1]
            color = _solid_color(tile)
            if color is not None:
                solid_tiles[(tr, tc)] = color
            elif np.any(tile == ORANGE):
                sprite_tiles.append(tile.copy())

    assert solid_tiles, "expected at least one solid payload tile"
    assert len(sprite_tiles) == 1, "expected exactly one sprite tile"

    rows = sorted({r for r, _ in solid_tiles})
    cols = sorted({c for _, c in solid_tiles})
    r0, r1 = rows[0], rows[-1]
    c0, c1 = cols[0], cols[-1]

    out = np.empty((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for tr in range(r0, r1 + 1):
        for tc in range(c0, c1 + 1):
            assert (tr, tc) in solid_tiles, "solid tiles must form one filled rectangle"
            out[tr - r0, tc - c0] = solid_tiles[(tr, tc)]

    rotation_steps = _sprite_rotation_steps(sprite_tiles[0])
    return out, rotation_steps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the solid-color tile rectangle and rotate it by the orientation encoded in one orange sprite tile.

    Input structure:
    - Single-pixel grey separator rows and columns partition the grid into a 4x4 array of equal tiles.
    - Most tiles are multicolor noise tiles, some tiles are fully solid in one non-black non-grey color, and exactly one tile contains an orange sprite on a black background.
    - The solid tiles occupy one contiguous rectangle in tile coordinates, while the sprite tile sits elsewhere and its orange component is not rotationally symmetric.

    Rule:
    - Split the grid at the full grey separator lines to recover the 4x4 tile lattice.
    - Keep exactly the tiles whose full interior is one non-black non-grey color, preserve their lattice positions, and crop them to the smallest tile-aligned bounding rectangle.
    - Read the orange sprite tile, compare its cropped shape against the canonical sprite and its 90-degree clockwise rotations, and use the matching orientation as the rotation amount.
    - Rotate the cropped color rectangle clockwise by that amount and return the rotated color grid.

    Key insight:
    - The solid tiles and the orange sprite encode two independent channels: the solid rectangle gives the data grid, and the sprite orientation gives the rotation instruction that must be applied at the end.
    """
    payload, rotation_steps = _extract_payload(np.asarray(grid, dtype=int))
    return np.rot90(payload, k=-rotation_steps)


def _tile_origin(tile_r: int, tile_c: int, tile_h: int, tile_w: int) -> tuple[int, int]:
    return tile_r * (tile_h + SEPARATOR_WIDTH), tile_c * (tile_w + SEPARATOR_WIDTH)


def _make_noise_tile(tile_h: int, tile_w: int, seed: int) -> np.ndarray:
    palette = random.sample(NOISE_COLORS, random.randint(2, 4))
    rr, cc = np.indices((tile_h, tile_w))
    mode = seed % 4
    if mode == 0:
        idx = rr + 2 * cc + seed
    elif mode == 1:
        idx = 2 * rr + cc + seed
    elif mode == 2:
        idx = rr * cc + rr + seed
    else:
        idx = (rr % 2) + 2 * (cc % 2) + seed
    tile = np.asarray(palette, dtype=int)[idx % len(palette)]
    if len(np.unique(tile)) < 2:
        tile[:, ::2] = palette[0]
        tile[:, 1::2] = palette[1]
    if random.random() < 0.5:
        tile = np.fliplr(tile)
    if random.random() < 0.5:
        tile = np.flipud(tile)
    return tile.astype(int, copy=True)


def _data_rotations_distinct(data: np.ndarray) -> bool:
    seen = set()
    for steps in range(4):
        rotated = np.rot90(data, k=-steps)
        seen.add((rotated.shape, rotated.tobytes()))
    return len(seen) == 4


def _make_data_block(block_h: int, block_w: int) -> np.ndarray | None:
    for _ in range(120):
        data = np.random.choice(SOLID_COLORS, size=(block_h, block_w)).astype(int)
        if len(np.unique(data)) < 2:
            continue
        if _data_rotations_distinct(data):
            return data
    return None


def _place_sprite(tile_h: int, tile_w: int, rotation_steps: int) -> np.ndarray:
    tile = np.full((tile_h, tile_w), BLACK, dtype=int)
    sprite = np.rot90(CANONICAL_SPRITE, k=-rotation_steps)
    sprite_h, sprite_w = sprite.shape
    r0 = random.randint(0, tile_h - sprite_h)
    c0 = random.randint(0, tile_w - sprite_w)
    tile[r0 : r0 + sprite_h, c0 : c0 + sprite_w][sprite] = ORANGE
    return tile


def _render_grid(
    tile_h: int,
    tile_w: int,
    block_top: int,
    block_left: int,
    data: np.ndarray,
    sprite_pos: tuple[int, int],
    rotation_steps: int,
) -> np.ndarray:
    height = TILE_ROWS * tile_h + (TILE_ROWS - 1) * SEPARATOR_WIDTH
    width = TILE_COLS * tile_w + (TILE_COLS - 1) * SEPARATOR_WIDTH
    grid = np.full((height, width), GREY, dtype=int)

    block_h, block_w = data.shape
    salt = random.randint(0, 10_000)
    for tr in range(TILE_ROWS):
        for tc in range(TILE_COLS):
            r0, c0 = _tile_origin(tr, tc, tile_h, tile_w)
            if block_top <= tr < block_top + block_h and block_left <= tc < block_left + block_w:
                color = int(data[tr - block_top, tc - block_left])
                tile = np.full((tile_h, tile_w), color, dtype=int)
            elif (tr, tc) == sprite_pos:
                tile = _place_sprite(tile_h, tile_w, rotation_steps)
            else:
                tile = _make_noise_tile(tile_h, tile_w, salt + tr * TILE_COLS + tc)
            grid[r0 : r0 + tile_h, c0 : c0 + tile_w] = tile
    return grid


def _validate_structure(
    grid: np.ndarray,
    block_top: int,
    block_left: int,
    data: np.ndarray,
    sprite_pos: tuple[int, int],
    rotation_steps: int,
) -> bool:
    try:
        row_bands, col_bands = _find_tile_spans(grid)
    except AssertionError:
        return False

    solid_tiles = {}
    sprite_count = 0
    noise_count = 0
    for tr, (r0, r1) in enumerate(row_bands):
        for tc, (c0, c1) in enumerate(col_bands):
            tile = grid[r0:r1, c0:c1]
            color = _solid_color(tile)
            if color is not None:
                solid_tiles[(tr, tc)] = color
                if color not in SOLID_COLORS:
                    return False
            elif np.any(tile == ORANGE):
                sprite_count += 1
                if (tr, tc) != sprite_pos:
                    return False
                try:
                    if _sprite_rotation_steps(tile) != rotation_steps:
                        return False
                except AssertionError:
                    return False
                orange_mask = tile == ORANGE
                if np.any(tile[(~orange_mask)] != BLACK):
                    return False
            else:
                noise_count += 1
                if len(np.unique(tile)) < 2 or np.any(tile == ORANGE) or np.any(tile == GREY):
                    return False

    block_h, block_w = data.shape
    expected_solid = {
        (block_top + dr, block_left + dc): int(data[dr, dc])
        for dr in range(block_h)
        for dc in range(block_w)
    }
    if solid_tiles != expected_solid:
        return False
    if sprite_count != 1 or noise_count <= 0:
        return False
    if not _data_rotations_distinct(data):
        return False

    expected = np.rot90(data, k=-rotation_steps)
    try:
        actual = main(grid)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    tile_h = random.randint(3, 6)
    tile_w = random.randint(3, 6)

    block_h = random.randint(2, 3)
    block_w = random.randint(2, 3)
    data = _make_data_block(block_h, block_w)
    if data is None:
        return None

    block_top = random.randint(0, TILE_ROWS - block_h)
    block_left = random.randint(0, TILE_COLS - block_w)
    block_cells = {
        (block_top + dr, block_left + dc)
        for dr in range(block_h)
        for dc in range(block_w)
    }
    free_cells = [(tr, tc) for tr in range(TILE_ROWS) for tc in range(TILE_COLS) if (tr, tc) not in block_cells]
    sprite_pos = random.choice(free_cells)
    rotation_steps = random.randrange(4)

    grid = _render_grid(tile_h, tile_w, block_top, block_left, data, sprite_pos, rotation_steps)
    if not _validate_structure(grid, block_top, block_left, data, sprite_pos, rotation_steps):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
