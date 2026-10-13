from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
WALL = 1
TEXTURE_WIDTH = 3
COLOR_POOL = tuple(color for color in range(10) if color not in (BG, WALL))
LAYOUTS = (
    (1, 2),
    (1, 3),
    (1, 4),
    (2, 2),
    (2, 3),
    (2, 4),
    (3, 2),
    (3, 3),
    (3, 4),
)
ROOM_HEIGHT_RANGE = (3, 6)
ROOM_WIDTH_RANGE = (3, 6)
PALETTE_HEIGHT_CHOICES = (2, 3)
GENERATION_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            cells: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(cells)
    return components


def _tile_texture(texture: np.ndarray, height: int, width: int) -> np.ndarray:
    reps_r = (height + texture.shape[0] - 1) // texture.shape[0]
    reps_c = (width + texture.shape[1] - 1) // texture.shape[1]
    tiled = np.tile(texture, (reps_r, reps_c))
    return tiled[:height, :width]


def _parse_palette(grid: np.ndarray) -> tuple[int, dict[int, np.ndarray]]:
    wall_rows = np.flatnonzero((grid == WALL).any(axis=1))
    assert wall_rows.size > 0, "expected at least one wall row"
    wall_top = int(wall_rows[0])
    assert wall_top >= 2, "palette must occupy at least two rows"

    palette = grid[:wall_top]
    col_has_sample = np.any(palette != BG, axis=0)
    textures: dict[int, np.ndarray] = {}
    c = 0
    while c < grid.shape[1]:
        if not col_has_sample[c]:
            c += 1
            continue
        start = c
        while c < grid.shape[1] and col_has_sample[c]:
            c += 1
        end = c
        assert end - start == TEXTURE_WIDTH, "palette samples must be 3 columns wide"
        block = palette[:, start:end]
        active_rows = np.flatnonzero(np.any(block != BG, axis=1))
        assert active_rows.size >= 2, "palette samples must be at least 2 rows tall"
        assert int(active_rows[0]) == 0, "palette samples must be top-aligned"
        height = int(active_rows[-1]) + 1
        assert height in (2, 3), "palette samples must be 2x3 or 3x3"
        sample = block[:height, :].copy()
        assert np.all(sample != BG), "palette samples should be fully populated"
        index_color = int(sample[0, 0])
        assert index_color not in (BG, WALL), "index colors must be non-background and non-wall"
        assert index_color not in textures, "palette index colors must be unique"
        textures[index_color] = sample

    assert textures, "failed to find any palette samples"
    return wall_top, textures


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    wall_top, textures = _parse_palette(g)
    out = g.copy()
    room_mask = g[wall_top + 1 :, :] != WALL

    processed = 0
    for component in _components(room_mask):
        rows = [r for r, _ in component]
        cols = [c for _, c in component]
        r0 = wall_top + 1 + min(rows)
        r1 = wall_top + 1 + max(rows)
        c0 = min(cols)
        c1 = max(cols)
        room = g[r0 : r1 + 1, c0 : c1 + 1]
        if room.size != len(component):
            continue
        if np.any(room == WALL):
            continue

        seed_positions = np.argwhere((room != BG) & (room != WALL))
        if seed_positions.shape[0] != 1:
            continue
        seed_r, seed_c = map(int, seed_positions[0])
        seed_color = int(room[seed_r, seed_c])
        texture = textures.get(seed_color)
        if texture is None:
            continue

        out[r0 : r1 + 1, c0 : c1 + 1] = _tile_texture(texture, room.shape[0], room.shape[1])
        processed += 1

    assert processed > 0, "expected at least one seed-marked room"
    return out


def _make_texture(
    rng: random.Random,
    index_color: int,
    height: int,
    existing: list[np.ndarray],
) -> np.ndarray:
    while True:
        color_count = rng.randint(2, min(4, len(COLOR_POOL)))
        colors = [index_color]
        others = [color for color in COLOR_POOL if color != index_color]
        colors.extend(rng.sample(others, color_count - 1))
        texture = np.array(
            [[rng.choice(colors) for _ in range(TEXTURE_WIDTH)] for _ in range(height)],
            dtype=int,
        )
        texture[0, 0] = index_color
        if np.unique(texture).size < 2:
            continue
        if any(np.array_equal(texture, prev) for prev in existing):
            continue
        return texture


def _room_ranges(lengths: list[int], offset: int) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    cursor = offset
    for length in lengths:
        start = cursor + 1
        end = start + length - 1
        ranges.append((start, end))
        cursor = end + 1
    return ranges


def _assign_seed_colors(
    rng: random.Random,
    room_count: int,
    palette_indices: list[int],
) -> list[int]:
    if room_count >= len(palette_indices) and rng.random() < 0.5:
        chosen = palette_indices[:]
        while len(chosen) < room_count:
            chosen.append(rng.choice(palette_indices))
    else:
        min_active = 2 if room_count > 1 and len(palette_indices) > 1 else 1
        active_count = rng.randint(min_active, min(len(palette_indices), room_count))
        active = rng.sample(palette_indices, active_count)
        chosen = [rng.choice(active) for _ in range(room_count)]
        if active_count > 1 and len(set(chosen)) == 1:
            chosen[0] = active[0]
            chosen[-1] = active[1]
    rng.shuffle(chosen)
    return chosen


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    palette_height = rng.choice(PALETTE_HEIGHT_CHOICES)
    room_rows, room_cols = rng.choice(LAYOUTS)
    row_heights = [rng.randint(*ROOM_HEIGHT_RANGE) for _ in range(room_rows)]
    col_widths = [rng.randint(*ROOM_WIDTH_RANGE) for _ in range(room_cols)]

    grid_height = palette_height + 1 + sum(row_heights) + (room_rows - 1) + 1
    grid_width = 1 + sum(col_widths) + (room_cols - 1) + 1
    max_textures = min(5, (grid_width + 1) // 4)
    if max_textures < 2:
        return None

    texture_count = rng.randint(2, max_textures)
    palette_indices = rng.sample(list(COLOR_POOL), texture_count)

    textures: dict[int, np.ndarray] = {}
    built_textures: list[np.ndarray] = []
    if palette_height == 2:
        heights = [2] * texture_count
    else:
        heights = [rng.choice((2, 3)) for _ in range(texture_count)]
        if 3 not in heights and rng.random() < 0.5:
            heights[rng.randrange(texture_count)] = 3
    for index_color, height in zip(palette_indices, heights):
        texture = _make_texture(rng, index_color, height, built_textures)
        textures[index_color] = texture
        built_textures.append(texture)

    grid = np.zeros((grid_height, grid_width), dtype=int)

    top_wall = palette_height
    grid[top_wall, :] = WALL
    bottom_wall = grid_height - 1
    grid[bottom_wall, :] = WALL

    row_ranges = _room_ranges(row_heights, top_wall)
    col_ranges = _room_ranges(col_widths, 0)

    horizontal_walls = [top_wall]
    for r0, r1 in row_ranges:
        horizontal_walls.append(r1 + 1)
    for wall_row in horizontal_walls:
        grid[wall_row, :] = WALL

    vertical_walls = [0]
    for c0, c1 in col_ranges:
        vertical_walls.append(c1 + 1)
    for wall_col in vertical_walls:
        grid[top_wall:, wall_col] = WALL

    required_width = texture_count * TEXTURE_WIDTH + (texture_count - 1)
    extra_width = grid_width - required_width
    left_margin = rng.randint(0, extra_width)
    cursor = left_margin
    placement_order = list(textures.items())
    rng.shuffle(placement_order)
    for idx, texture in enumerate(placement_order):
        _, pattern = texture
        grid[: pattern.shape[0], cursor : cursor + TEXTURE_WIDTH] = pattern
        cursor += TEXTURE_WIDTH
        if idx != len(placement_order) - 1:
            cursor += 1

    room_specs = [(r0, r1, c0, c1) for r0, r1 in row_ranges for c0, c1 in col_ranges]
    seed_colors = _assign_seed_colors(rng, len(room_specs), list(textures))

    expected = grid.copy()
    for (r0, r1, c0, c1), seed_color in zip(room_specs, seed_colors):
        seed_r = rng.randint(r0, r1)
        seed_c = rng.randint(c0, c1)
        grid[seed_r, seed_c] = seed_color
        expected[r0 : r1 + 1, c0 : c1 + 1] = _tile_texture(
            textures[seed_color], r1 - r0 + 1, c1 - c0 + 1
        )

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATION_ATTEMPTS} attempts")
