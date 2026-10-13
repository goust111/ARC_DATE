from __future__ import annotations

import random
from itertools import combinations

import numpy as np


RED = 2

# --------------- solver helpers ---------------

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _checkerboard(shape: tuple[int, int], color0: int, color1: int, block: int, flip: int) -> np.ndarray:
    h, w = shape
    rows = (np.arange(h) // block)[:, None]
    cols = (np.arange(w) // block)[None, :]
    parity = (rows + cols + flip) & 1
    return np.where(parity == 0, color0, color1).astype(int)


def _infer_checkerboard(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    colors = [int(c) for c in np.unique(g)]
    assert len(colors) >= 3, "expected two board colors plus at least one contaminant color"

    best_key = None
    best_expected = None
    for color0, color1 in combinations(colors, 2):
        clean_mask = (g == color0) | (g == color1)
        clean_count = int(clean_mask.sum())
        if clean_count == 0:
            continue

        for block in range(1, min(h, w) + 1):
            for flip in (0, 1):
                expected = _checkerboard((h, w), color0, color1, block, flip)
                mismatches = int(np.count_nonzero(clean_mask & (g != expected)))
                key = (mismatches, -clean_count, block)
                if best_key is None or key < best_key:
                    best_key = key
                    best_expected = expected

    assert best_expected is not None, "failed to infer checkerboard background"
    return best_expected


def _component_bboxes(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    boxes: list[tuple[int, int, int, int]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            r0 = r1 = r
            c0 = c1 = c

            while stack:
                cr, cc = stack.pop()
                if cr < r0:
                    r0 = cr
                if cr > r1:
                    r1 = cr
                if cc < c0:
                    c0 = cc
                if cc > c1:
                    c1 = cc

                for dr, dc in DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            boxes.append((r0, c0, r1, c1))

    return boxes


def _draw_rings(base: np.ndarray, polluted_mask: np.ndarray) -> np.ndarray:
    out = np.asarray(base, dtype=int).copy()
    h, w = out.shape

    for r0, c0, r1, c1 in _component_bboxes(polluted_mask):
        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(h - 1, r1 + 1)
        cc1 = min(w - 1, c1 + 1)

        out[rr0 : rr1 + 1, cc0] = RED
        out[rr0 : rr1 + 1, cc1] = RED
        out[rr0, cc0 : cc1 + 1] = RED
        out[rr1, cc0 : cc1 + 1] = RED

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair checkerboard corruption and mark each polluted region with a red ring.

    Input structure:
    - The whole grid is a two-color checkerboard made of square blocks.
    - Some full blocks are replaced by a third color, forming one or more polluted regions.

    Rule:
    - Infer the two checkerboard colors and the block size from the visible pattern.
    - Restore every polluted pixel to its expected checkerboard color.
    - For each 4-connected polluted region, draw a one-pixel red rectangle around
      its bounding box, expanded outward by one pixel.
    """
    g = np.asarray(grid, dtype=int)
    expected = _infer_checkerboard(g)
    polluted_mask = g != expected
    return _draw_rings(expected, polluted_mask)


# --------------- generator helpers ---------------

CELL_SIZE_MIN = 1
CELL_SIZE_MAX = 4
CELL_ROWS_MIN = 5
CELL_ROWS_MAX = 9
CELL_COLS_MIN = 5
CELL_COLS_MAX = 9
REGION_COUNT_MIN = 1
REGION_COUNT_MAX = 4
GENERATION_RETRIES = 400
INPUT_COLORS = tuple(c for c in range(10) if c != RED)
SHAPE_KINDS = ("single", "rect", "bar", "l", "blob")


def _normalize_shape(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _shape_bbox(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _rotate_shape(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    return _normalize_shape({(c, -r) for r, c in cells})


def _make_blob(rng: random.Random) -> set[tuple[int, int]]:
    target = rng.randint(3, 6)
    cells = {(0, 0)}
    frontier = {(0, 1), (1, 0), (0, -1), (-1, 0)}

    while len(cells) < target and frontier:
        cand = rng.choice(tuple(frontier))
        frontier.remove(cand)
        trial = cells | {cand}
        nr = [r for r, _ in trial]
        nc = [c for _, c in trial]
        if max(nr) - min(nr) >= 4 or max(nc) - min(nc) >= 4:
            continue
        cells = trial
        for r, c in list(cells):
            for dr, dc in DIR4:
                nb = (r + dr, c + dc)
                if nb not in cells:
                    frontier.add(nb)

    return _normalize_shape(cells)


def _make_shape(rng: random.Random) -> set[tuple[int, int]]:
    kind = rng.choice(SHAPE_KINDS)
    if kind == "single":
        return {(0, 0)}
    if kind == "rect":
        height = rng.randint(1, 3)
        width = rng.randint(1, 3)
        return {(r, c) for r in range(height) for c in range(width)}
    if kind == "bar":
        length = rng.randint(2, 4)
        if rng.random() < 0.5:
            return {(0, c) for c in range(length)}
        return {(r, 0) for r in range(length)}
    if kind == "l":
        height = rng.randint(2, 4)
        width = rng.randint(2, 4)
        shape = {(r, 0) for r in range(height)} | {(0, c) for c in range(width)}
        for _ in range(rng.randrange(4)):
            shape = _rotate_shape(shape)
        return shape
    return _make_blob(rng)


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _place_regions(
    cell_rows: int,
    cell_cols: int,
    shapes: list[set[tuple[int, int]]],
    gap_cells: int,
    rng: random.Random,
) -> list[set[tuple[int, int]]] | None:
    placed_boxes: list[tuple[int, int, int, int]] = []
    regions: list[set[tuple[int, int]]] = []

    shapes = sorted(shapes, key=lambda cells: len(cells), reverse=True)
    for shape in shapes:
        sr0, sc0, sr1, sc1 = _shape_bbox(shape)
        height = sr1 - sr0 + 1
        width = sc1 - sc0 + 1
        candidates: list[tuple[int, int, set[tuple[int, int]], tuple[int, int, int, int]]] = []

        for r0 in range(1, cell_rows - height):
            for c0 in range(1, cell_cols - width):
                shifted = {(r0 + r, c0 + c) for r, c in shape}
                br0, bc0, br1, bc1 = _shape_bbox(shifted)
                test_box = (
                    br0 - gap_cells,
                    bc0 - gap_cells,
                    br1 + gap_cells,
                    bc1 + gap_cells,
                )
                if any(_boxes_overlap(test_box, box) for box in placed_boxes):
                    continue
                candidates.append((r0, c0, shifted, test_box))

        if not candidates:
            return None

        _r0, _c0, shifted, test_box = rng.choice(candidates)
        placed_boxes.append(test_box)
        regions.append(shifted)

    return regions


def _checkerboard_cells(rows: int, cols: int, color0: int, color1: int, flip: int) -> np.ndarray:
    rr = np.arange(rows)[:, None]
    cc = np.arange(cols)[None, :]
    return np.where(((rr + cc + flip) & 1) == 0, color0, color1).astype(int)


def _expand_cells(cell_grid: np.ndarray, block: int) -> np.ndarray:
    return np.repeat(np.repeat(cell_grid, block, axis=0), block, axis=1)


def _expected_output(clean_pixels: np.ndarray, polluted_pixels: np.ndarray) -> np.ndarray:
    polluted_mask = polluted_pixels != clean_pixels
    return _draw_rings(clean_pixels, polluted_mask)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    block = rng.randint(CELL_SIZE_MIN, CELL_SIZE_MAX)
    cell_rows = rng.randint(CELL_ROWS_MIN, CELL_ROWS_MAX)
    cell_cols = rng.randint(CELL_COLS_MIN, CELL_COLS_MAX)
    color0, color1, pollutant = rng.sample(INPUT_COLORS, 3)
    flip = rng.randrange(2)

    region_count = rng.randint(REGION_COUNT_MIN, REGION_COUNT_MAX)
    shapes = [_make_shape(rng) for _ in range(region_count)]

    polluted_cell_total = sum(len(shape) for shape in shapes)
    total_cells = cell_rows * cell_cols
    if polluted_cell_total == 0 or polluted_cell_total * 4 > total_cells:
        return None

    gap_cells = 2 if block == 1 else 1
    regions = _place_regions(cell_rows, cell_cols, shapes, gap_cells, rng)
    if regions is None:
        return None

    clean_cells = _checkerboard_cells(cell_rows, cell_cols, color0, color1, flip)
    polluted_cells = clean_cells.copy()
    for region in regions:
        for r, c in region:
            polluted_cells[r, c] = pollutant

    clean_pixels = _expand_cells(clean_cells, block)
    inp = _expand_cells(polluted_cells, block)
    out = _expected_output(clean_pixels, inp)

    # Keep the checkerboard colors dominant and both visible after corruption.
    vals, counts = np.unique(inp, return_counts=True)
    color_counts = {int(v): int(n) for v, n in zip(vals, counts)}
    if color_counts.get(pollutant, 0) >= min(color_counts.get(color0, 0), color_counts.get(color1, 0)):
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if np.array_equal(solved, out):
        return inp
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GENERATION_RETRIES} attempts")
