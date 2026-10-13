from __future__ import annotations

import itertools
import random
from collections import Counter

import numpy as np


CELL = 3
BG_COLOR = 0
DIVIDER_COLOR = 5
GLYPH_COLOR = 7
TILE_COLORS = (1, 2, 3, 4, 6, 8, 9)
GEN_ATTEMPTS = 500

Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


# Small helpers adapted from arc_utils so the file stays self-contained when
# loaded via SourceFileLoader.
def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _find_divider_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == DIVIDER_COLOR)]
    assert len(rows) == 1, "expected exactly one full-width gray divider row"
    return int(rows[0])


def _sorted_glyph_components(top: np.ndarray) -> list[np.ndarray]:
    comps = _components(top == GLYPH_COLOR)
    assert comps, "expected at least one orange glyph"
    comps.sort(key=lambda comp: (_bbox_coords(comp)[1], _bbox_coords(comp)[0]))
    return comps


def _find_tile_layout(top: np.ndarray, bottom: np.ndarray, n_tiles: int, bg: int) -> tuple[int, tuple[int, ...]]:
    glyph_boxes = [_bbox_coords(comp) for comp in _sorted_glyph_components(top)]
    assert len(glyph_boxes) == n_tiles

    tile_pixels = np.argwhere(bottom != bg)
    assert len(tile_pixels) > 0, "expected visible bottom tiles"

    h, w = bottom.shape
    max_gap = (w - CELL * n_tiles) // max(1, n_tiles - 1)
    candidates: list[tuple[int, tuple[int, ...]]] = []

    for tile_r0 in range(h - CELL + 1):
        if tile_pixels[:, 0].min() < tile_r0 or tile_pixels[:, 0].max() >= tile_r0 + CELL:
            continue
        for gap in range(1, max_gap + 1):
            pitch = CELL + gap
            total_width = CELL + (n_tiles - 1) * pitch
            for first_c0 in range(w - total_width + 1):
                slot_cols = tuple(first_c0 + i * pitch for i in range(n_tiles))

                if any(
                    not (slot_cols[i] <= box[1] and box[3] < slot_cols[i] + CELL)
                    for i, box in enumerate(glyph_boxes)
                ):
                    continue

                ok = True
                for rr, cc in tile_pixels:
                    if not any(slot_c0 <= cc < slot_c0 + CELL for slot_c0 in slot_cols):
                        ok = False
                        break
                if not ok:
                    continue

                if any(np.all(bottom[tile_r0 : tile_r0 + CELL, c0 : c0 + CELL] == bg) for c0 in slot_cols):
                    continue
                candidates.append((tile_r0, slot_cols))

    unique = list(dict.fromkeys(candidates))
    assert len(unique) == 1, f"expected a unique 3x3 tile layout, found {len(unique)}"
    return unique[0]


def _reorder_indices(areas: list[int]) -> tuple[int, ...]:
    return tuple(idx for idx, _area in sorted(enumerate(areas), key=lambda item: item[1]))


def _build_output(bottom: np.ndarray, tile_r0: int, slot_cols: tuple[int, ...], tiles: list[np.ndarray], areas: list[int], bg: int) -> np.ndarray:
    out = np.full(bottom.shape, bg, dtype=int)
    for src_idx, area in enumerate(areas):
        dst_idx = int(area) - 1
        c0 = int(slot_cols[dst_idx])
        out[tile_r0 : tile_r0 + CELL, c0 : c0 + CELL] = tiles[src_idx]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder the lower 3x3 tiles according to the pixel areas of the upper orange glyphs.

    Input structure:
    - A black-background grid is split by one full-width gray row into an upper glyph panel and a lower tile panel.
    - The upper panel contains 4 or 5 separated orange glyphs, one per conceptual 3x3 cell, laid out left to right at equal spacing.
    - The lower panel contains the same number of equally spaced 3x3 tiles with distinct visible patterns and colors.

    Rule:
    - Read the upper orange glyphs from left to right and count each glyph's orange pixels.
    - These counts form a permutation of 1..N; the i-th count says the i-th lower tile must move to output position `count`.
    - Keep every tile's internal 3x3 pattern unchanged and return only the lower panel after the tiles have been moved into their target slots.

    Key insight:
    - The upper shapes are distractors except for their areas: area is the external code that determines the lower-row permutation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    divider_r = _find_divider_row(g)
    top = g[:divider_r, :]
    bottom = g[divider_r + 1 :, :]

    glyphs = _sorted_glyph_components(top)
    n_tiles = len(glyphs)
    assert n_tiles in (4, 5), "expected 4 or 5 glyphs"
    areas = [int(len(comp)) for comp in glyphs]
    assert sorted(areas) == list(range(1, n_tiles + 1)), "glyph areas must be a permutation of 1..N"

    tile_r0, slot_cols = _find_tile_layout(top, bottom, n_tiles, bg)
    tiles = [bottom[tile_r0 : tile_r0 + CELL, c0 : c0 + CELL].copy() for c0 in slot_cols]
    return _build_output(bottom, tile_r0, slot_cols, tiles, areas, bg)


def _is_connected(mask: np.ndarray) -> bool:
    return mask.any() and len(_components(mask)) == 1


def _all_binary_masks() -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for bits in range(1, 1 << (CELL * CELL)):
        flat = np.array([(bits >> i) & 1 for i in range(CELL * CELL)], dtype=bool)
        masks.append(flat.reshape(CELL, CELL))
    return masks


def _build_glyph_catalog() -> dict[int, list[np.ndarray]]:
    by_area = {area: [] for area in range(1, 6)}
    for mask in _all_binary_masks():
        area = int(mask.sum())
        if area not in by_area:
            continue
        if _is_connected(mask):
            by_area[area].append(mask.copy())
    return by_area


def _build_tile_catalog() -> dict[int, list[np.ndarray]]:
    by_area = {area: [] for area in range(5, 10)}
    for mask in _all_binary_masks():
        area = int(mask.sum())
        if area not in by_area:
            continue
        if not mask.any(axis=0).all() or not mask.any(axis=1).all():
            continue
        by_area[area].append(mask.copy())
    return by_area


GLYPH_MASKS = _build_glyph_catalog()
TILE_MASKS = _build_tile_catalog()


def _paint_mask(color: int, mask: np.ndarray) -> np.ndarray:
    patch = np.zeros((CELL, CELL), dtype=int)
    patch[mask] = int(color)
    return patch


def _render_attempt(
    n_tiles: int,
    gap: int,
    left_pad: int,
    right_pad: int,
    top_head_pad: int,
    top_tail_pad: int,
    bottom_head_pad: int,
    bottom_tail_pad: int,
    areas: list[int],
    glyph_masks: list[np.ndarray],
    tile_patches: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    width = left_pad + n_tiles * CELL + (n_tiles - 1) * gap + right_pad
    top_h = top_head_pad + CELL + top_tail_pad
    bottom_h = bottom_head_pad + CELL + bottom_tail_pad
    divider_r = top_h
    tile_r0 = bottom_head_pad
    glyph_r0 = top_head_pad
    slot_cols = [left_pad + i * (CELL + gap) for i in range(n_tiles)]

    grid = np.full((top_h + 1 + bottom_h, width), BG_COLOR, dtype=int)
    grid[divider_r, :] = DIVIDER_COLOR

    for c0, glyph_mask in zip(slot_cols, glyph_masks):
        grid[glyph_r0 : glyph_r0 + CELL, c0 : c0 + CELL][glyph_mask] = GLYPH_COLOR
    for c0, tile in zip(slot_cols, tile_patches):
        grid[divider_r + 1 + tile_r0 : divider_r + 1 + tile_r0 + CELL, c0 : c0 + CELL] = tile

    expected_bottom = _build_output(
        grid[divider_r + 1 :, :],
        tile_r0,
        tuple(slot_cols),
        tile_patches,
        areas,
        BG_COLOR,
    )
    return grid, expected_bottom


def _semantic_ok(areas: list[int], tile_masks: list[np.ndarray]) -> bool:
    n_tiles = len(areas)
    output_order = _reorder_indices(areas)
    if output_order == tuple(range(n_tiles)):
        return False
    if output_order == tuple(range(n_tiles - 1, -1, -1)):
        return False

    tile_areas = [int(mask.sum()) for mask in tile_masks]
    assert len(set(tile_areas)) == len(tile_areas)
    asc_area_order = tuple(idx for idx, _area in sorted(enumerate(tile_areas), key=lambda item: item[1]))
    desc_area_order = tuple(idx for idx, _area in sorted(enumerate(tile_areas), key=lambda item: -item[1]))
    return output_order not in {asc_area_order, desc_area_order}


def _gen_attempt() -> np.ndarray | None:
    n_tiles = random.choice((4, 5))
    gap = random.choice((1, 2))
    left_pad = random.randint(0, 2)
    right_pad = random.randint(0, 2)
    top_head_pad = random.randint(0, 2)
    top_tail_pad = random.randint(0, 2)
    bottom_head_pad = random.randint(0, 2)
    bottom_tail_pad = random.randint(0, 2)

    areas = random.sample(list(range(1, n_tiles + 1)), n_tiles)
    glyph_masks = [random.choice(GLYPH_MASKS[area]).copy() for area in areas]

    tile_area_choices = random.sample(list(range(5, 10)), n_tiles)
    tile_masks = [random.choice(TILE_MASKS[area]).copy() for area in tile_area_choices]
    colors = random.sample(TILE_COLORS, n_tiles)
    tile_patches = [_paint_mask(color, mask) for color, mask in zip(colors, tile_masks)]

    if not _semantic_ok(areas, tile_masks):
        return None

    inp, expected_bottom = _render_attempt(
        n_tiles,
        gap,
        left_pad,
        right_pad,
        top_head_pad,
        top_tail_pad,
        bottom_head_pad,
        bottom_tail_pad,
        areas,
        glyph_masks,
        tile_patches,
    )

    try:
        actual = main(inp)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected_bottom):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
