from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
TOP = 1
BOTTOM = 2
TRAIL = 5
TILE_COLORS = (3, 4, 6, 7, 8, 9)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
TILE_SHAPES = ((2, 3), (3, 2))
GEN_ATTEMPTS = 400


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        queue: deque[tuple[int, int]] = deque([(r, c)])
        coords: list[tuple[int, int]] = []
        while queue:
            rr, cc = queue.popleft()
            coords.append((rr, cc))
            for dr, dc in DIR4:
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        comps.append(np.array(coords, dtype=int))
    return comps


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste_clip(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    rr1 = min(dst.shape[0], r0 + h)
    cc1 = min(dst.shape[1], c0 + w)
    if r0 >= rr1 or c0 >= cc1:
        return
    dst[r0:rr1, c0:cc1] = patch[: rr1 - r0, : cc1 - c0]


def _slide_object(
    coords: np.ndarray,
    color: int,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    h, _w = shape
    r0, _c0, r1, _c1 = _bbox(coords)
    if color == TOP:
        dr = 1 - r0
        steps = range(0, -dr)
        delta = -1
    else:
        dr = (h - 2) - r1
        steps = range(0, dr)
        delta = 1
    assert dr != 0

    trail_cells: set[tuple[int, int]] = set()
    for step in steps:
        shifted = coords + np.array([step * delta, 0], dtype=int)
        trail_cells.update((int(r), int(c)) for r, c in shifted)

    final_coords = coords + np.array([dr, 0], dtype=int)
    final_set = {(int(r), int(c)) for r, c in final_coords}
    trail_cells.difference_update(final_set)
    assert trail_cells

    trail_coords = np.array(sorted(trail_cells), dtype=int)
    return final_coords, trail_coords


def _find_objects(grid: np.ndarray, color: int) -> list[dict]:
    comps = []
    for coords in _components(grid == color):
        box = _bbox(coords)
        r0, c0, r1, c1 = box
        if r0 == r1 and r0 in {0, grid.shape[0] - 1} and c0 == 0 and c1 == grid.shape[1] - 1:
            continue
        comps.append({"color": int(color), "coords": coords, "bbox": box})
    return comps


def _find_tiles(grid: np.ndarray, bg: int, top_color: int, bottom_color: int) -> list[dict]:
    mask = (~np.isin(grid, [bg, top_color, bottom_color]))
    tiles = []
    for coords in _components(mask):
        box = _bbox(coords)
        r0, c0, r1, c1 = box
        rel = coords - np.array([r0, c0], dtype=int)
        rect_mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        rect_mask[rel[:, 0], rel[:, 1]] = True
        assert bool(rect_mask.all())
        assert c0 == 0
        tiles.append({"coords": coords, "bbox": box, "patch": _crop(grid, box)})
    tiles.sort(key=lambda tile: tile["bbox"][0])
    return tiles


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each border-matched object vertically, keep its grey swept stripe, and repeat the uniquely overlapping left-edge tile by the stripe's row span.

    Input structure:
    - The background is black, with a full-width blue bar on the top row and a full-width red bar on the bottom row.
    - Interior blue or red 4-connected monochrome objects sit away from the bars and slide vertically toward the same-colored bar.
    - Small multicolor rectangular tiles are attached to the left edge; each valid object trail overlaps exactly one tile row band.

    Rule:
    - Move every blue object upward until its bbox top reaches row 1, and every red object downward until its bbox bottom reaches row H-2.
    - Paint the swept cells of each move grey, excluding the object's final resting cells.
    - For each moved object, measure the number of rows covered by its grey trail; find the unique left-edge tile whose row interval overlaps that trail interval.
    - Repeat that tile horizontally from its original position exactly that many times, clipping at the right edge, while keeping the bars, moved objects, trails, and original tiles together.

    Key insight:
    - The slide distance becomes visible as the vertical extent of the grey trail, and that geometric extent is reused as the repeat count for the matching edge tile.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 5 and w >= 5

    bg = BG
    top_color = int(g[0, 0])
    bottom_color = int(g[h - 1, 0])
    assert bool(np.all(g[0, :] == top_color))
    assert bool(np.all(g[h - 1, :] == bottom_color))
    assert top_color != bottom_color

    objects = _find_objects(g, top_color) + _find_objects(g, bottom_color)
    tiles = _find_tiles(g, bg, top_color, bottom_color)
    assert objects
    assert tiles

    out = g.copy()
    for obj in objects:
        coords = obj["coords"]
        out[coords[:, 0], coords[:, 1]] = bg

    used_tiles: set[int] = set()
    for obj in objects:
        final_coords, trail_coords = _slide_object(obj["coords"], obj["color"], g.shape)
        trail_r0, trail_r1 = int(trail_coords[:, 0].min()), int(trail_coords[:, 0].max())
        repeat_count = trail_r1 - trail_r0 + 1

        matches = []
        for idx, tile in enumerate(tiles):
            tile_r0, _tile_c0, tile_r1, _tile_c1 = tile["bbox"]
            if max(trail_r0, tile_r0) <= min(trail_r1, tile_r1):
                matches.append(idx)
        assert len(matches) == 1
        tile_idx = matches[0]
        assert tile_idx not in used_tiles
        used_tiles.add(tile_idx)

        out[trail_coords[:, 0], trail_coords[:, 1]] = TRAIL
        out[final_coords[:, 0], final_coords[:, 1]] = obj["color"]

        tile = tiles[tile_idx]
        tile_r0, tile_c0, _tile_r1, _tile_c1 = tile["bbox"]
        patch = tile["patch"]
        tile_w = patch.shape[1]
        for rep in range(repeat_count):
            _paste_clip(out, patch, tile_r0, tile_c0 + rep * tile_w)

    return out


def _make_tile(shape: tuple[int, int], used: set[tuple]) -> np.ndarray:
    h, w = shape
    area = h * w
    for _ in range(100):
        n_colors = random.randint(2, min(3, area))
        palette = random.sample(TILE_COLORS, n_colors)
        patch = np.array(random.choices(palette, k=area), dtype=int).reshape(h, w)
        for idx, color in enumerate(palette):
            patch.flat[idx] = color
        sig = (patch.shape, tuple(int(x) for x in patch.flat))
        if sig not in used:
            used.add(sig)
            return patch
    raise RuntimeError("failed to create a distinct tile")


def _build_attempt() -> np.ndarray | None:
    h = random.randint(14, 18)
    w = random.randint(18, 24)
    top_shape = random.choice(TILE_SHAPES)
    bottom_shape = random.choice(TILE_SHAPES)
    top_h, top_w = top_shape
    bottom_h, bottom_w = bottom_shape

    feasible_repeats = []
    for top_rep in range(top_h, 7):
        for bottom_rep in range(bottom_h, 7):
            if top_rep + bottom_rep > h - 5:
                continue
            if max(top_rep * top_w, bottom_rep * bottom_w) > w - 7:
                continue
            feasible_repeats.append((top_rep, bottom_rep))
    if not feasible_repeats:
        return None

    top_rep, bottom_rep = random.choice(feasible_repeats)
    blue_row = top_rep + 1
    red_row = h - 2 - bottom_rep
    if blue_row + 1 >= red_row:
        return None

    top_r0 = random.randint(2, blue_row - top_h + 1)
    bottom_r0 = random.randint(red_row, h - 3 - bottom_h + 1)
    if top_r0 + top_h - 1 >= red_row:
        return None
    if bottom_r0 <= blue_row:
        return None

    blue_len = random.randint(3, 5)
    red_len = random.randint(3, 5)
    min_col = max(top_rep * top_w, bottom_rep * bottom_w) + 2
    blue_choices = list(range(min_col, w - blue_len + 1))
    red_choices = list(range(min_col, w - red_len + 1))
    if not blue_choices or not red_choices:
        return None

    random.shuffle(blue_choices)
    random.shuffle(red_choices)
    blue_col = blue_choices[0]
    red_col = red_choices[0]
    if abs(blue_col - red_col) < 2 and len(red_choices) > 1:
        red_col = red_choices[1]

    grid = np.full((h, w), BG, dtype=int)
    grid[0, :] = TOP
    grid[h - 1, :] = BOTTOM

    used_tiles: set[tuple] = set()
    top_tile = _make_tile(top_shape, used_tiles)
    bottom_tile = _make_tile(bottom_shape, used_tiles)
    _paste_clip(grid, top_tile, top_r0, 0)
    _paste_clip(grid, bottom_tile, bottom_r0, 0)

    grid[blue_row, blue_col : blue_col + blue_len] = TOP
    grid[red_row, red_col : red_col + red_len] = BOTTOM

    try:
        out = main(grid)
    except AssertionError:
        return None

    assert out.shape == grid.shape
    assert max(out.shape) <= 30
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
