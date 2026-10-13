from __future__ import annotations

import random

import numpy as np


BG = 0
SEP = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
THUMB_SLOTS = [(r, c) for r in range(3) for c in range(3) if (r, c) != (1, 1)]
MAX_GEN_ATTEMPTS = 300


def _find_separator_layout(grid: np.ndarray) -> tuple[int, list[int], list[int]]:
    h, w = grid.shape
    for color in map(int, np.unique(grid)):
        row_lines = [r for r in range(h) if np.all(grid[r, :] == color)]
        col_lines = [c for c in range(w) if np.all(grid[:, c] == color)]
        if len(row_lines) != 2 or len(col_lines) != 2:
            continue
        if row_lines[1] - row_lines[0] != row_lines[0] + 1:
            continue
        if col_lines[1] - col_lines[0] != col_lines[0] + 1:
            continue
        if 3 * row_lines[0] + 2 != h or 3 * col_lines[0] + 2 != w:
            continue
        return color, row_lines, col_lines
    raise AssertionError("expected two full separator rows and two full separator columns")


def _tile_slices(
    row_lines: list[int], col_lines: list[int]
) -> dict[tuple[int, int], tuple[slice, slice]]:
    row_starts = [0, row_lines[0] + 1, row_lines[1] + 1]
    col_starts = [0, col_lines[0] + 1, col_lines[1] + 1]
    tile_h = row_lines[0]
    tile_w = col_lines[0]
    return {
        (tr, tc): (slice(row_starts[tr], row_starts[tr] + tile_h), slice(col_starts[tc], col_starts[tc] + tile_w))
        for tr in range(3)
        for tc in range(3)
    }


def _infer_background(grid: np.ndarray, sep: int, row_lines: list[int], col_lines: list[int]) -> int:
    mask = np.ones(grid.shape, dtype=bool)
    mask[row_lines, :] = False
    mask[:, col_lines] = False
    values, counts = np.unique(grid[mask], return_counts=True)
    best = int(values[np.argmax(counts)])
    assert best != sep
    return best


def _read_thumbnail_map(tile: np.ndarray) -> np.ndarray:
    tile_h, tile_w = tile.shape
    assert tile_h % 3 == 0 and tile_w % 3 == 0
    block_h = tile_h // 3
    block_w = tile_w // 3
    thumb = np.zeros((3, 3), dtype=int)
    for r in range(3):
        for c in range(3):
            block = tile[r * block_h : (r + 1) * block_h, c * block_w : (c + 1) * block_w]
            colors = np.unique(block)
            assert len(colors) == 1, "thumbnail blocks must be solid"
            thumb[r, c] = int(colors[0])
    return thumb


def _is_thumbnail_tile(tile: np.ndarray, bg: int, sep: int) -> bool:
    if np.any(tile == bg):
        return False
    thumb = _read_thumbnail_map(tile)
    return int(np.sum(thumb == sep)) == 1 and len(set(map(int, thumb.flatten()))) >= 3


def _locate_thumbnail(
    grid: np.ndarray,
    tile_boxes: dict[tuple[int, int], tuple[slice, slice]],
    bg: int,
    sep: int,
) -> tuple[tuple[int, int], np.ndarray]:
    candidates: list[tuple[tuple[int, int], np.ndarray]] = []
    for slot, (rs, cs) in tile_boxes.items():
        tile = grid[rs, cs]
        if _is_thumbnail_tile(tile, bg, sep):
            candidates.append((slot, tile.copy()))
    assert len(candidates) == 1, "expected exactly one thumbnail tile"
    return candidates[0]


def _orient_thumbnail(thumb: np.ndarray, sep: int, thumb_slot: tuple[int, int]) -> np.ndarray:
    matches = []
    for rot in range(4):
        oriented = np.rot90(thumb, rot)
        grey_cells = np.argwhere(oriented == sep)
        assert len(grey_cells) == 1
        if tuple(map(int, grey_cells[0])) == thumb_slot:
            matches.append(oriented)
    assert len(matches) == 1, "thumbnail orientation must be uniquely determined"
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the self-referential thumbnail tile to recolor the other eight tile sprites.

    Input structure:
    - Grey separator lines split the canvas into a 3x3 array of equal tiles.
    - Eight tiles contain one monochrome sprite on a shared background.
    - The remaining tile is fully occupied by a 3x3 thumbnail map whose cells are solid color blocks,
      with exactly one grey block marking the thumbnail's own slot.

    Rule:
    - Find the unique tile that is fully filled by the thumbnail map rather than a sparse sprite.
    - Rotate that thumbnail until its grey block lands on the thumbnail tile's outer 3x3 slot.
    - Recolor every non-background pixel in each other tile to the oriented thumbnail color for that slot,
      while keeping separator lines, backgrounds, sprite geometry, and the thumbnail tile unchanged.

    Key insight:
    - The thumbnail has no external orientation cue; the only way to orient it is the self-reference
      that its grey cell must point back to the slot occupied by the thumbnail itself.
    """
    g = np.asarray(grid, dtype=int)
    sep, row_lines, col_lines = _find_separator_layout(g)
    bg = _infer_background(g, sep, row_lines, col_lines)
    tile_boxes = _tile_slices(row_lines, col_lines)

    thumb_slot, thumb_tile = _locate_thumbnail(g, tile_boxes, bg, sep)
    oriented_thumb = _orient_thumbnail(_read_thumbnail_map(thumb_tile), sep, thumb_slot)

    out = g.copy()
    for slot, (rs, cs) in tile_boxes.items():
        if slot == thumb_slot:
            continue
        tile = g[rs, cs]
        sprite_mask = tile != bg
        assert bool(sprite_mask.any()), "each non-thumbnail tile must contain a sprite"
        out_tile = out[rs, cs]
        out_tile[sprite_mask] = int(oriented_thumb[slot])

    return out


def _random_connected_mask(h: int, w: int, area: int) -> np.ndarray:
    cells = {(random.randrange(h), random.randrange(w))}
    while len(cells) < area:
        r, c = random.choice(tuple(cells))
        nbrs = []
        if r > 0:
            nbrs.append((r - 1, c))
        if r + 1 < h:
            nbrs.append((r + 1, c))
        if c > 0:
            nbrs.append((r, c - 1))
        if c + 1 < w:
            nbrs.append((r, c + 1))
        random.shuffle(nbrs)
        for nbr in nbrs:
            if nbr not in cells:
                cells.add(nbr)
                break

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _make_sprite(tile_h: int, tile_w: int, color: int) -> np.ndarray:
    max_h = min(5, tile_h - 2)
    max_w = min(5, tile_w - 2)
    bbox_h = random.randint(2, max_h)
    bbox_w = random.randint(2, max_w)
    max_area = min(bbox_h * bbox_w - 1, max(4, (bbox_h * bbox_w * 2) // 3))
    area = random.randint(3, max_area)
    mask = _random_connected_mask(bbox_h, bbox_w, area)

    if len(set(np.argwhere(mask)[:, 0])) == 1 and bbox_h > 2:
        extra_r = random.choice([r for r in range(bbox_h) if not mask[r, :].all()])
        mask[extra_r, random.randrange(bbox_w)] = True
    if len(set(np.argwhere(mask)[:, 1])) == 1 and bbox_w > 2:
        extra_c = random.choice([c for c in range(bbox_w) if not mask[:, c].all()])
        mask[random.randrange(bbox_h), extra_c] = True

    sprite = np.full((tile_h, tile_w), BG, dtype=int)
    top = random.randint(1, tile_h - bbox_h - 1)
    left = random.randint(1, tile_w - bbox_w - 1)
    sprite[top : top + bbox_h, left : left + bbox_w][mask] = color
    return sprite


def _sample_thumbnail_colors() -> np.ndarray:
    while True:
        palette = random.sample(PALETTE, k=random.randint(3, min(6, len(PALETTE))))
        colors = np.array([random.choice(palette) for _ in range(8)], dtype=int)
        if len(set(map(int, colors))) >= 3:
            return colors


def _render_thumbnail(base_map: np.ndarray, block_h: int, block_w: int, rot: int) -> np.ndarray:
    rotated = np.rot90(base_map, rot)
    return np.repeat(np.repeat(rotated, block_h, axis=0), block_w, axis=1)


def _build_input() -> np.ndarray:
    block_h = random.choice([2, 3])
    block_w = random.choice([2, 3])
    tile_h = 3 * block_h
    tile_w = 3 * block_w
    height = 3 * tile_h + 2
    width = 3 * tile_w + 2

    grid = np.full((height, width), BG, dtype=int)
    grid[tile_h, :] = SEP
    grid[2 * tile_h + 1, :] = SEP
    grid[:, tile_w] = SEP
    grid[:, 2 * tile_w + 1] = SEP

    row_lines = [tile_h, 2 * tile_h + 1]
    col_lines = [tile_w, 2 * tile_w + 1]
    tile_boxes = _tile_slices(row_lines, col_lines)

    thumb_slot = random.choice(THUMB_SLOTS)
    base_map = np.zeros((3, 3), dtype=int)
    base_map[thumb_slot] = SEP
    color_values = _sample_thumbnail_colors()
    idx = 0
    for r in range(3):
        for c in range(3):
            if (r, c) == thumb_slot:
                continue
            base_map[r, c] = int(color_values[idx])
            idx += 1

    input_rot = random.randrange(4)
    thumb_tile = _render_thumbnail(base_map, block_h, block_w, input_rot)
    rs, cs = tile_boxes[thumb_slot]
    grid[rs, cs] = thumb_tile

    for slot, (tile_rs, tile_cs) in tile_boxes.items():
        if slot == thumb_slot:
            continue
        target = int(base_map[slot])
        choices = [color for color in PALETTE if color != target]
        initial_color = random.choice(choices)
        sprite = _make_sprite(tile_h, tile_w, initial_color)
        grid[tile_rs, tile_cs] = sprite

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _build_input()
        try:
            out = main(grid)
        except Exception:
            continue

        if np.array_equal(out, grid):
            continue
        return grid
    raise RuntimeError("failed to generate a valid self-referential thumbnail puzzle")
