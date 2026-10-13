from __future__ import annotations

import random
from collections.abc import Iterable

import numpy as np


BLACK = 0
GREY = 5
PALETTE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

RECT_COUNT_RANGE = (3, 6)
GRID_SIDE_MAX = 30
GEN_RETRIES = 400


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_line_palette(line: np.ndarray) -> tuple[list[int], list[int]] | None:
    if np.any(line == GREY):
        return None

    idxs = np.flatnonzero((line != BLACK) & (line != GREY))
    if len(idxs) == 0:
        return None
    if idxs[-1] - idxs[0] + 1 != len(idxs):
        return None

    line = np.asarray(line, dtype=int)
    outside = np.ones(len(line), dtype=bool)
    outside[idxs[0] : idxs[-1] + 1] = False
    if np.any(line[outside] != BLACK):
        return None

    colors = [int(v) for v in line[idxs]]
    if len(set(colors)) != len(colors):
        return None
    return idxs.tolist(), colors


def _find_palette(grid: np.ndarray) -> tuple[str, list[tuple[int, int]], list[int]]:
    h, w = grid.shape
    candidates: list[tuple[str, list[tuple[int, int]], list[int]]] = []

    horizontal = _extract_line_palette(grid[h - 1, :])
    if horizontal is not None:
        cols, colors = horizontal
        candidates.append(("horizontal", [(h - 1, c) for c in cols], colors))

    vertical = _extract_line_palette(grid[:, w - 1])
    if vertical is not None:
        rows, colors = vertical
        candidates.append(("vertical", [(r, w - 1) for r in rows], colors))

    assert len(candidates) == 1, f"expected exactly one palette strip, got {len(candidates)}"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey rectangle by matching its area rank to the palette-strip order.

    Input structure:
    - The background is black.
    - Several disconnected solid grey rectangles of distinct areas are scattered in the grid.
    - The bottom row or the rightmost column contains one contiguous palette strip made of distinct single-pixel colors.

    Rule:
    - Detect the palette strip and read its colors in spatial order: left-to-right for a bottom-row strip, top-to-bottom for a right-column strip.
    - Remove the strip from the working grid, then find all grey connected components and verify that each is a solid rectangle.
    - Sort the grey rectangles by area from smallest to largest and assign the k-th palette color to the k-th rectangle in that ranking.
    - Recolor every pixel of each rectangle with its assigned color, and replace the original palette strip with black.

    Key insight:
    - The palette strip encodes an explicit order, while the rectangles encode an implicit order through their distinct areas; solving the task means aligning those two orderings.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    _orientation, palette_cells, palette_colors = _find_palette(g)

    work = g.copy()
    for r, c in palette_cells:
        work[r, c] = BLACK

    components = _find_components(work == GREY)
    assert len(components) == len(palette_colors), "rectangle count must match palette length"

    rectangles: list[dict[str, int]] = []
    for coords in components:
        r0, c0, r1, c1 = _bbox(coords)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        area = int(coords.shape[0])
        assert area == height * width, "grey component must be a solid rectangle"
        rectangles.append({"r0": r0, "c0": c0, "r1": r1, "c1": c1, "area": area})

    areas = [rect["area"] for rect in rectangles]
    assert len(set(areas)) == len(areas), "rectangle areas must be distinct"

    out = work.copy()
    for rect, color in zip(sorted(rectangles, key=lambda rect: rect["area"]), palette_colors):
        out[rect["r0"] : rect["r1"] + 1, rect["c0"] : rect["c1"] + 1] = int(color)

    return out


def _sample_grid_shape(rect_count: int, orientation: str) -> tuple[int, int]:
    base = 8 + 2 * rect_count
    h = random.randint(base, min(GRID_SIDE_MAX, base + 8))
    w = random.randint(base, min(GRID_SIDE_MAX, base + 8))

    if orientation == "horizontal":
        w = max(w, rect_count + 2)
    else:
        h = max(h, rect_count + 2)

    return h, w


def _sample_rectangle_dims(usable_h: int, usable_w: int, rect_count: int) -> list[tuple[int, int]] | None:
    max_h = min(6, usable_h)
    max_w = min(6, usable_w)
    candidates = [
        (h, w)
        for h in range(1, max_h + 1)
        for w in range(1, max_w + 1)
        if 2 <= h * w <= 24
    ]
    random.shuffle(candidates)

    dims: list[tuple[int, int]] = []
    used_areas: set[int] = set()
    for h, w in candidates:
        area = h * w
        if area in used_areas:
            continue
        dims.append((h, w))
        used_areas.add(area)
        if len(dims) == rect_count:
            break

    if len(dims) != rect_count:
        return None
    return dims


def _iter_positions(usable_h: int, usable_w: int, rect_h: int, rect_w: int) -> Iterable[tuple[int, int]]:
    for r0 in range(usable_h - rect_h + 1):
        for c0 in range(usable_w - rect_w + 1):
            yield r0, c0


def _place_rectangles(
    usable_h: int, usable_w: int, dims: list[tuple[int, int]]
) -> list[tuple[int, int, int, int]] | None:
    blocked = np.zeros((usable_h, usable_w), dtype=bool)
    placements: list[tuple[int, int, int, int]] = []

    for rect_h, rect_w in sorted(dims, key=lambda hw: hw[0] * hw[1], reverse=True):
        options: list[tuple[int, int]] = []
        for r0, c0 in _iter_positions(usable_h, usable_w, rect_h, rect_w):
            rs = max(0, r0 - 1)
            re = min(usable_h, r0 + rect_h + 1)
            cs = max(0, c0 - 1)
            ce = min(usable_w, c0 + rect_w + 1)
            if not blocked[rs:re, cs:ce].any():
                options.append((r0, c0))

        if not options:
            return None

        r0, c0 = random.choice(options)
        placements.append((r0, c0, rect_h, rect_w))
        rs = max(0, r0 - 1)
        re = min(usable_h, r0 + rect_h + 1)
        cs = max(0, c0 - 1)
        ce = min(usable_w, c0 + rect_w + 1)
        blocked[rs:re, cs:ce] = True

    random.shuffle(placements)
    return placements


def _paint_palette(grid: np.ndarray, orientation: str, colors: list[int]) -> None:
    count = len(colors)
    if orientation == "horizontal":
        start = random.randint(0, grid.shape[1] - count - 1)
        for offset, color in enumerate(colors):
            grid[grid.shape[0] - 1, start + offset] = int(color)
    else:
        start = random.randint(0, grid.shape[0] - count - 1)
        for offset, color in enumerate(colors):
            grid[start + offset, grid.shape[1] - 1] = int(color)


def _gen_attempt() -> np.ndarray | None:
    rect_count = random.randint(*RECT_COUNT_RANGE)
    orientation = random.choice(("horizontal", "vertical"))
    h, w = _sample_grid_shape(rect_count, orientation)

    usable_h = h - 1 if orientation == "horizontal" else h
    usable_w = w if orientation == "horizontal" else w - 1
    if usable_h < 3 or usable_w < 3:
        return None

    dims = _sample_rectangle_dims(usable_h, usable_w, rect_count)
    if dims is None:
        return None

    placements = _place_rectangles(usable_h, usable_w, dims)
    if placements is None:
        return None

    palette_colors = random.sample(PALETTE_COLORS, rect_count)
    grid = np.full((h, w), BLACK, dtype=int)

    for r0, c0, rect_h, rect_w in placements:
        grid[r0 : r0 + rect_h, c0 : c0 + rect_w] = GREY

    _paint_palette(grid, orientation, palette_colors)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
