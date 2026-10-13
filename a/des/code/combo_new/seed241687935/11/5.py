from __future__ import annotations

from collections import deque
from functools import lru_cache
from itertools import combinations
import random

import numpy as np


BG = 0
SEPARATOR_COLOR = 5
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)

# Generator configuration.
MARKER_COUNT_RANGE = (2, 4)
SEED_AREA_RANGE = (3, 5)
OFFSET_ROW_RANGE = (-6, 6)
OFFSET_COL_RANGE = (-8, 8)
TOP_EXTRA_HEIGHT = (0, 4)
BOTTOM_EXTRA_HEIGHT = (0, 6)
WIDTH_EXTRA = (0, 8)
MAX_GENERATION_ATTEMPTS = 400
SHAPE_CANVAS = 4


def _components_by_color(grid: np.ndarray, bg: int = BG) -> list[dict]:
    """Return 4-connected monochrome components for all non-background colors."""
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue

            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "area": int(arr.shape[0]),
                    "bbox": (r0, c0, r1, c1),
                }
            )

    return comps


def _shape_signature(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    """Normalize component cells to its bbox top-left and return a hashable signature."""
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def _separator_row(grid: np.ndarray, bg: int = BG) -> int:
    """Find the unique full-width non-background separator row."""
    g = np.asarray(grid, dtype=int)
    candidates = [
        int(r)
        for r in range(g.shape[0])
        if int(g[r, 0]) != bg and np.all(g[r] == int(g[r, 0]))
    ]
    assert len(candidates) == 1, "expected exactly one solid non-background separator row"
    return int(candidates[0])


def _paint_shape(canvas: np.ndarray, shape: tuple[tuple[int, int], ...], r0: int, c0: int, color: int) -> None:
    """Paint one normalized shape with its top-left corner at (r0, c0)."""
    for dr, dc in shape:
        canvas[int(r0 + dr), int(c0 + dc)] = int(color)


def _translated_cells(shape: tuple[tuple[int, int], ...], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(int(r + dr), int(c + dc)) for r, c in shape}


def _padded_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    """Dilate a cell set by one Chebyshev step to keep generated objects visually separated."""
    return {
        (int(r + rr), int(c + cc))
        for r, c in cells
        for rr in (-1, 0, 1)
        for cc in (-1, 0, 1)
    }


def _span(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [int(r) for r, _c in points]
    cols = [int(c) for _r, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the seed shape into the lower region using the top marker offsets and marker colors.

    Input structure:
    - The background is black, and one solid grey separator row splits the grid into an upper
      instruction area and a lower work area.
    - The upper area contains one small monochrome seed object plus several single-pixel markers,
      each marker having a distinct color different from the seed color.
    - The lower area contains one copy of the same seed object in the same color, plus one
      single-pixel marker for each upper marker color.

    Rule:
    - Find the seed shape in the upper area and record each upper marker's row/column offset from
      the seed bbox top-left corner.
    - In the lower area, locate the matching seed copy and place one seed-shaped copy at each
      recorded offset relative to that lower seed's bbox top-left corner.
    - Color each placed copy with the color of the marker that supplied the offset, erase the
      lower single-pixel markers, and keep the original lower seed copy.
    - Remove the upper area and the separator row; output only the transformed lower area.

    Key insight:
    - The upper region is a legend of relative positions, not part of the answer. The lower
      region reuses those relative offsets around a translated base seed, while marker colors
      determine the colors of the copied shapes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep = _separator_row(g, bg=BG)
    top = g[:sep]
    bottom = g[sep + 1 :]

    top_comps = _components_by_color(top, bg=BG)
    top_seed_candidates = [comp for comp in top_comps if int(comp["area"]) > 1]
    assert len(top_seed_candidates) == 1, "upper area must contain exactly one non-singleton seed"
    top_seed = top_seed_candidates[0]

    seed_color = int(top_seed["color"])
    seed_shape = _shape_signature(top_seed["coords"])
    seed_r0, seed_c0, _seed_r1, _seed_c1 = map(int, top_seed["bbox"])

    offsets_by_color: dict[int, tuple[int, int]] = {}
    for comp in top_comps:
        if int(comp["area"]) != 1:
            continue
        mr, mc = map(int, comp["coords"][0])
        color = int(comp["color"])
        assert color != seed_color, "marker colors must differ from the seed color"
        assert color not in offsets_by_color, "upper marker colors must be unique"
        offsets_by_color[color] = (int(mr - seed_r0), int(mc - seed_c0))
    assert offsets_by_color, "expected at least one upper marker"

    bottom_comps = _components_by_color(bottom, bg=BG)
    base_candidates = [
        comp
        for comp in bottom_comps
        if int(comp["area"]) > 1
        and int(comp["color"]) == seed_color
        and _shape_signature(comp["coords"]) == seed_shape
    ]
    assert len(base_candidates) == 1, "lower area must contain exactly one matching seed copy"
    base_seed = base_candidates[0]
    base_r0, base_c0, _base_r1, _base_c1 = map(int, base_seed["bbox"])

    lower_marker_cells: dict[int, tuple[int, int]] = {}
    for comp in bottom_comps:
        if int(comp["area"]) != 1:
            continue
        color = int(comp["color"])
        assert color not in lower_marker_cells, "lower marker colors must be unique"
        lower_marker_cells[color] = tuple(map(int, comp["coords"][0]))

    assert set(lower_marker_cells) == set(offsets_by_color), "upper/lower marker colors must match"

    out = bottom.copy()
    for r, c in lower_marker_cells.values():
        out[int(r), int(c)] = BG

    for color, (dr, dc) in offsets_by_color.items():
        _paint_shape(out, seed_shape, int(base_r0 + dr), int(base_c0 + dc), int(color))

    return out


@lru_cache(maxsize=None)
def _seed_shapes() -> tuple[tuple[tuple[int, int], ...], ...]:
    """Enumerate small connected seed shapes with an occupied bbox top-left cell."""
    seen: set[tuple[tuple[int, int], ...]] = set()
    shapes: list[tuple[tuple[int, int], ...]] = []

    for area in range(SEED_AREA_RANGE[0], SEED_AREA_RANGE[1] + 1):
        for cells in combinations(range(SHAPE_CANVAS * SHAPE_CANVAS), area):
            if 0 not in cells:
                continue

            pts = {(int(cell // SHAPE_CANVAS), int(cell % SHAPE_CANVAS)) for cell in cells}
            queue = deque([(0, 0)])
            visited = {(0, 0)}
            while queue:
                r, c = queue.popleft()
                for dr, dc in DIR4:
                    nxt = (int(r + dr), int(c + dc))
                    if nxt in pts and nxt not in visited:
                        visited.add(nxt)
                        queue.append(nxt)
            if len(visited) != len(pts):
                continue

            r0, c0, r1, c1 = _span(list(pts))
            sig = tuple(sorted((int(r - r0), int(c - c0)) for r, c in pts))
            if sig[0] != (0, 0):
                continue
            if max(r1 - r0 + 1, c1 - c0 + 1) > SHAPE_CANVAS:
                continue
            if sig not in seen:
                seen.add(sig)
                shapes.append(sig)

    return tuple(shapes)


def _sample_offsets(shape: tuple[tuple[int, int], ...], count: int) -> list[tuple[int, int]] | None:
    """Pick disjoint target offsets for the copied seed top-left corners."""
    base_cells = _translated_cells(shape, 0, 0)
    used_padded = _padded_cells(base_cells)

    candidates = [
        (int(dr), int(dc))
        for dr in range(OFFSET_ROW_RANGE[0], OFFSET_ROW_RANGE[1] + 1)
        for dc in range(OFFSET_COL_RANGE[0], OFFSET_COL_RANGE[1] + 1)
        if not (dr == 0 and dc == 0)
    ]
    random.shuffle(candidates)

    offsets: list[tuple[int, int]] = []
    for dr, dc in candidates:
        cells = _translated_cells(shape, dr, dc)
        if cells & used_padded:
            continue
        offsets.append((int(dr), int(dc)))
        used_padded |= _padded_cells(cells)
        if len(offsets) == count:
            break

    if len(offsets) != count:
        return None

    signs = {(0 if dr == 0 else (1 if dr > 0 else -1), 0 if dc == 0 else (1 if dc > 0 else -1)) for dr, dc in offsets}
    if len(signs) == 1:
        return None
    return offsets


def _render_input_output(
    shape: tuple[tuple[int, int], ...],
    seed_color: int,
    marker_colors: list[int],
    offsets: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Assemble one valid input grid plus its expected output."""
    top_points = list(shape) + offsets
    top_min_r, top_min_c, top_max_r, top_max_c = _span(top_points)

    lower_points: list[tuple[int, int]] = list(shape)
    for dr, dc in offsets:
        lower_points.extend((int(r + dr), int(c + dc)) for r, c in shape)
    bot_min_r, bot_min_c, bot_max_r, bot_max_c = _span(lower_points)

    top_min_h = int(top_max_r - top_min_r + 1)
    bot_min_h = int(bot_max_r - bot_min_r + 1)
    min_w = int(max(top_max_c - top_min_c + 1, bot_max_c - bot_min_c + 1))

    top_h = int(top_min_h + random.randint(*TOP_EXTRA_HEIGHT))
    bot_h = int(bot_min_h + random.randint(*BOTTOM_EXTRA_HEIGHT))
    if top_h + 1 + bot_h > 30:
        return None

    max_extra_w = min(WIDTH_EXTRA[1], 30 - min_w)
    min_extra_w = min(WIDTH_EXTRA[0], max_extra_w)
    if max_extra_w < 0:
        return None
    width = int(min_w + random.randint(min_extra_w, max_extra_w))
    if width > 30:
        return None

    top_r_lo = int(-top_min_r)
    top_r_hi = int(top_h - 1 - top_max_r)
    top_c_lo = int(-top_min_c)
    top_c_hi = int(width - 1 - top_max_c)
    bot_r_lo = int(-bot_min_r)
    bot_r_hi = int(bot_h - 1 - bot_max_r)
    bot_c_lo = int(-bot_min_c)
    bot_c_hi = int(width - 1 - bot_max_c)
    if top_r_lo > top_r_hi or top_c_lo > top_c_hi or bot_r_lo > bot_r_hi or bot_c_lo > bot_c_hi:
        return None

    top_seed_r0 = random.randint(top_r_lo, top_r_hi)
    top_seed_c0 = random.randint(top_c_lo, top_c_hi)
    bot_seed_r0 = random.randint(bot_r_lo, bot_r_hi)
    bot_seed_c0 = random.randint(bot_c_lo, bot_c_hi)

    top = np.full((top_h, width), BG, dtype=int)
    bottom_in = np.full((bot_h, width), BG, dtype=int)
    bottom_out = np.full((bot_h, width), BG, dtype=int)

    _paint_shape(top, shape, top_seed_r0, top_seed_c0, seed_color)
    _paint_shape(bottom_in, shape, bot_seed_r0, bot_seed_c0, seed_color)
    _paint_shape(bottom_out, shape, bot_seed_r0, bot_seed_c0, seed_color)

    for color, (dr, dc) in zip(marker_colors, offsets, strict=True):
        top[int(top_seed_r0 + dr), int(top_seed_c0 + dc)] = int(color)
        bottom_in[int(bot_seed_r0 + dr), int(bot_seed_c0 + dc)] = int(color)
        _paint_shape(bottom_out, shape, int(bot_seed_r0 + dr), int(bot_seed_c0 + dc), int(color))

    separator = np.full((1, width), SEPARATOR_COLOR, dtype=int)
    inp = np.vstack([top, separator, bottom_in])
    return inp, bottom_out


def _gen_attempt() -> np.ndarray | None:
    shape = random.choice(_seed_shapes())
    marker_count = random.randint(*MARKER_COUNT_RANGE)
    offsets = _sample_offsets(shape, marker_count)
    if offsets is None:
        return None

    palette = random.sample(COLOR_POOL, marker_count + 1)
    seed_color = int(palette[0])
    marker_colors = [int(color) for color in palette[1:]]

    rendered = _render_input_output(shape, seed_color, marker_colors, offsets)
    if rendered is None:
        return None
    inp, expected = rendered

    try:
        actual = main(inp)
    except Exception:
        return None

    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    """Generate a valid randomized input grid for this puzzle."""
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for combo_new/seed241687935/11/5")
