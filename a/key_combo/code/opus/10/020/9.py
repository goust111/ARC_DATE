from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5
MAROON = 9

MAX_GENERATION_ATTEMPTS = 300
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in NEIGHBORS4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _component_coords(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _extract_seed_mask(grid: np.ndarray) -> tuple[np.ndarray, int]:
    h, w = grid.shape
    for total in range(min(h, w, 7), 2, -1):
        if not np.all(grid[total - 1, :total] == MAROON):
            continue
        if not np.all(grid[:total, total - 1] == MAROON):
            continue
        seed = grid[: total - 1, : total - 1]
        if seed.shape[0] != seed.shape[1]:
            continue
        vals = set(map(int, np.unique(seed)))
        if not vals <= {BLACK, RED}:
            continue
        if not np.any(seed == RED) or not np.any(seed == BLACK):
            continue
        return seed == RED, total
    raise AssertionError("failed to find the top-left seed legend")


def _expanded_tile_mask(seed_mask: np.ndarray) -> np.ndarray:
    # One recursive level: each active seed cell expands into the seed mask itself.
    return np.kron(seed_mask.astype(int), seed_mask.astype(int)).astype(bool)


def _partition_bboxes(grid: np.ndarray, legend_total: int) -> list[tuple[int, int, int, int]]:
    mask = grid != BLUE
    mask[:legend_total, :legend_total] = False
    boxes: list[tuple[int, int, int, int]] = []
    for coords in _component_coords(mask):
        box = _bbox_from_coords(coords)
        r0, c0, r1, c1 = box
        region = mask[r0 : r1 + 1, c0 : c1 + 1]
        assert bool(region.all()), "each blue-separated partition must be rectangular"
        boxes.append(box)
    assert 2 <= len(boxes) <= 3, "expected 2-3 blue-separated partitions"
    return boxes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Tile each blue-separated region with a recursively expanded seed, using red inside each grey-content bbox and maroon outside it.

    Input structure:
    - The grid uses black as background, with a top-left square seed legend whose last row and last column are solid maroon.
    - Removing that maroon L-border leaves a red/black binary seed mask.
    - The rest of the grid is split by blue separator lines into 2-3 rectangular partitions, each containing one small 4-connected grey region.

    Rule:
    - Read the binary seed from the legend and expand it by one self-similar level: every seed foreground cell becomes a copy of the seed mask.
    - For each blue-separated partition, compute the bounding box of its grey cells.
    - Tile the expanded seed mask from the partition's top-left corner: active tile cells become red inside the grey bbox and maroon outside it; inactive tile cells stay black.
    - Grey cells are overwritten, while the seed legend and blue separator lines remain unchanged.

    Key insight:
    - The grey pixels do not survive; they only define a per-partition bounding box that switches the color scheme of the same recursively expanded tile.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    seed_mask, legend_total = _extract_seed_mask(g)
    tile_mask = _expanded_tile_mask(seed_mask)
    tile_h, tile_w = tile_mask.shape

    out = g.copy()
    for r0, c0, r1, c1 in _partition_bboxes(g, legend_total):
        part = g[r0 : r1 + 1, c0 : c1 + 1]
        grey_coords = [tuple(map(int, rc)) for rc in np.argwhere(part == GREY)]
        assert grey_coords, "each partition must contain grey content"
        gr0, gc0, gr1, gc1 = _bbox_from_coords(grey_coords)
        assert gr1 - gr0 + 1 < part.shape[0] and gc1 - gc0 + 1 < part.shape[1]

        rows = np.arange(part.shape[0])[:, None]
        cols = np.arange(part.shape[1])[None, :]
        active = tile_mask[rows % tile_h, cols % tile_w]

        inside_bbox = np.zeros(part.shape, dtype=bool)
        inside_bbox[gr0 : gr1 + 1, gc0 : gc1 + 1] = True

        painted = np.full(part.shape, BLACK, dtype=int)
        painted[active & ~inside_bbox] = MAROON
        painted[active & inside_bbox] = RED
        out[r0 : r1 + 1, c0 : c1 + 1] = painted

    return out


def _random_seed_mask(seed_size: int) -> np.ndarray:
    area = seed_size * seed_size
    for _ in range(200):
        n_red = random.randint(2, area - 1)
        chosen = random.sample(range(area), n_red)
        mask = np.zeros((seed_size, seed_size), dtype=bool)
        mask.reshape(-1)[chosen] = True
        expanded = _expanded_tile_mask(mask)
        repeated = np.tile(mask, (seed_size, seed_size))
        if np.array_equal(expanded, repeated):
            continue
        if expanded.sum() == 0 or expanded.sum() == expanded.size:
            continue
        return mask
    raise RuntimeError("failed to sample a non-degenerate seed mask")


def _partition_layout(seed_size: int):
    legend_total = seed_size + 1
    for _ in range(200):
        count = random.choice((2, 3))
        split_axis = random.choice(("rows", "cols"))
        mult_hi = 3 if count == 3 else 4
        strip_sizes = [seed_size * random.randint(2, mult_hi) for _ in range(count)]
        fixed_size = seed_size * random.randint(2, 5)

        boxes: list[tuple[int, int, int, int]] = []
        if split_axis == "rows":
            h = legend_total + 1 + sum(strip_sizes) + (count - 1)
            w = legend_total + 1 + fixed_size
            start_r = legend_total + 1
            c0 = legend_total + 1
            for part_h in strip_sizes:
                boxes.append((start_r, c0, start_r + part_h - 1, c0 + fixed_size - 1))
                start_r += part_h + 1
        else:
            h = legend_total + 1 + fixed_size
            w = legend_total + 1 + sum(strip_sizes) + (count - 1)
            start_c = legend_total + 1
            r0 = legend_total + 1
            for part_w in strip_sizes:
                boxes.append((r0, start_c, r0 + fixed_size - 1, start_c + part_w - 1))
                start_c += part_w + 1

        if h <= 30 and w <= 30:
            return h, w, boxes
    raise RuntimeError("failed to sample a valid partition layout")


def _shape_touches_all_sides(cells: set[tuple[int, int]], h: int, w: int) -> bool:
    rows = {r for r, _ in cells}
    cols = {c for _, c in cells}
    return 0 in rows and (h - 1) in rows and 0 in cols and (w - 1) in cols


def _connected_shape(bbox_h: int, bbox_w: int) -> set[tuple[int, int]]:
    max_area = min(8, bbox_h * bbox_w - 1)
    min_area = 3
    if max_area < min_area:
        raise RuntimeError("bbox too small for a non-full grey shape")

    for _ in range(400):
        target = random.randint(min_area, max_area)
        cells = {(random.randrange(bbox_h), random.randrange(bbox_w))}
        while len(cells) < target:
            frontier = {
                (nr, nc)
                for r, c in cells
                for nr, nc in _neighbors4(r, c, bbox_h, bbox_w)
                if (nr, nc) not in cells
            }
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))
        if len(cells) != target:
            continue
        if not _shape_touches_all_sides(cells, bbox_h, bbox_w):
            continue
        if len(cells) == bbox_h * bbox_w:
            continue
        return cells
    raise RuntimeError("failed to generate an irregular grey component")


def _place_grey_region(grid: np.ndarray, box: tuple[int, int, int, int]):
    r0, c0, r1, c1 = box
    part_h = r1 - r0 + 1
    part_w = c1 - c0 + 1

    bbox_h = random.randint(2, min(4, part_h - 1))
    bbox_w = random.randint(2, min(4, part_w - 1))
    shape = _connected_shape(bbox_h, bbox_w)

    top = random.randint(0, part_h - bbox_h)
    left = random.randint(0, part_w - bbox_w)
    for rr, cc in shape:
        grid[r0 + top + rr, c0 + left + cc] = GREY


def _render_input(seed_mask: np.ndarray, boxes: list[tuple[int, int, int, int]], shape: tuple[int, int]) -> np.ndarray:
    seed_size = seed_mask.shape[0]
    legend_total = seed_size + 1
    grid = np.full(shape, BLUE, dtype=int)

    grid[:legend_total, :legend_total] = BLACK
    seed_region = grid[:seed_size, :seed_size]
    seed_region[seed_mask] = RED
    grid[seed_size, :legend_total] = MAROON
    grid[:legend_total, seed_size] = MAROON

    for box in boxes:
        r0, c0, r1, c1 = box
        grid[r0 : r1 + 1, c0 : c1 + 1] = BLACK
        _place_grey_region(grid, box)
    return grid


def _attempt_input() -> np.ndarray | None:
    seed_size = random.choice((2, 3))
    seed_mask = _random_seed_mask(seed_size)
    h, w, boxes = _partition_layout(seed_size)
    grid = _render_input(seed_mask, boxes, (h, w))

    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 9")
