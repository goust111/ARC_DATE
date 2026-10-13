from __future__ import annotations

import random

import numpy as np


BG = 0
RED = 2
GREY = 5
PALETTE = [1, 3, 4, 6, 7, 8, 9]
MAX_RETRIES = 300
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _find_separator(g: np.ndarray) -> tuple[str, int]:
    rows = [r for r in range(g.shape[0]) if np.all(g[r] == GREY)]
    cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == GREY)]
    assert len(rows) + len(cols) == 1, "expected exactly one full grey separator"
    if rows:
        r = int(rows[0])
        assert r * 2 + 1 == g.shape[0], "separator must split grid into equal halves"
        return "horizontal", r
    c = int(cols[0])
    assert c * 2 + 1 == g.shape[1], "separator must split grid into equal halves"
    return "vertical", c


def _split_regions(g: np.ndarray, orient: str, axis: int) -> tuple[np.ndarray, np.ndarray]:
    if orient == "horizontal":
        top = g[:axis, :]
        bottom = g[axis + 1 :, :]
        assert top.shape == bottom.shape
        return top, bottom
    left = g[:, :axis]
    right = g[:, axis + 1 :]
    assert left.shape == right.shape
    return left, right


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compare the two separator-defined regions cell-by-cell in local coordinates.

    Shared foreground positions stay in their original foreground colors.
    Mismatched positions are highlighted red in both corresponding regions.
    Background stays black and the separator stays grey.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    orient, axis = _find_separator(g)
    region_a, region_b = _split_regions(g, orient, axis)

    mask_a = region_a != BG
    mask_a &= region_a != GREY
    mask_b = region_b != BG
    mask_b &= region_b != GREY

    out_a = np.full_like(region_a, BG)
    out_b = np.full_like(region_b, BG)

    both = mask_a & mask_b
    diff = mask_a ^ mask_b

    out_a[both] = region_a[both]
    out_b[both] = region_b[both]
    out_a[diff] = RED
    out_b[diff] = RED

    out = np.full_like(g, BG)
    if orient == "horizontal":
        out[:axis, :] = out_a
        out[axis, :] = GREY
        out[axis + 1 :, :] = out_b
    else:
        out[:, :axis] = out_a
        out[:, axis] = GREY
        out[:, axis + 1 :] = out_b
    return out


def _normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _grow_connected_shape(height: int, width: int, area: int) -> set[tuple[int, int]]:
    start = (random.randrange(height), random.randrange(width))
    cells = {start}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in NEIGHBORS4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < height and 0 <= cc < width and (rr, cc) not in cells:
                    frontier.append((rr, cc))
        assert frontier
        cells.add(random.choice(frontier))
    return _normalize(cells)


def _frontier(shape: set[tuple[int, int]], height: int, width: int) -> list[tuple[int, int]]:
    seen = set()
    frontier = []
    for r, c in shape:
        for dr, dc in NEIGHBORS4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < height and 0 <= cc < width and (rr, cc) not in shape and (rr, cc) not in seen:
                seen.add((rr, cc))
                frontier.append((rr, cc))
    return frontier


def _shift_within(shape: set[tuple[int, int]], height: int, width: int) -> set[tuple[int, int]]:
    max_r = max(r for r, _ in shape)
    max_c = max(c for _, c in shape)
    dr = random.randint(0, height - max_r - 1)
    dc = random.randint(0, width - max_c - 1)
    return {(r + dr, c + dc) for r, c in shape}


def _variant_from_shared(shared: set[tuple[int, int]], height: int, width: int, extra_count: int) -> set[tuple[int, int]]:
    shape = set(shared)
    candidates = _frontier(shape, height, width)
    random.shuffle(candidates)
    for cell in candidates[:extra_count]:
        shape.add(cell)
    return shape


def _make_pattern_pair(height: int, width: int) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    total = height * width
    shared_area = random.randint(max(2, total // 8), max(3, total // 4))
    shared_area = min(shared_area, total - 2)
    shared = _grow_connected_shape(height, width, shared_area)

    frontier = _frontier(shared, height, width)
    assert len(frontier) >= 2, "need room for asymmetric edits"
    random.shuffle(frontier)

    max_extra = min(len(frontier) // 2, max(1, total // 8), 4)
    extra_a = random.randint(1, max_extra)
    extra_b = random.randint(1, max_extra)
    shape_a = set(shared)
    shape_b = set(shared)

    for cell in frontier[:extra_a]:
        shape_a.add(cell)
    remaining = [cell for cell in frontier if cell not in shape_a]
    random.shuffle(remaining)
    if len(remaining) < extra_b:
        remaining = frontier
    for cell in remaining[:extra_b]:
        shape_b.add(cell)

    assert shape_a != shape_b
    assert shape_a & shape_b
    return shape_a, shape_b


def _paint_region(region: np.ndarray, shape: set[tuple[int, int]], color: int) -> None:
    for r, c in shape:
        region[r, c] = color


def _generate_case() -> np.ndarray:
    orient = random.choice(("horizontal", "vertical"))
    region_h = random.randint(5, 12)
    region_w = random.randint(5, 12)

    color_a = random.choice(PALETTE)
    same_color = random.random() < 0.5
    color_b = color_a if same_color else random.choice([c for c in PALETTE if c != color_a])

    local_h = random.randint(2, min(6, region_h))
    local_w = random.randint(2, min(6, region_w))
    shape_a, shape_b = _make_pattern_pair(local_h, local_w)
    shape_a = _shift_within(shape_a, region_h, region_w)
    shape_b = _shift_within(shape_b, region_h, region_w)

    region_a = np.full((region_h, region_w), BG, dtype=int)
    region_b = np.full((region_h, region_w), BG, dtype=int)
    _paint_region(region_a, shape_a, color_a)
    _paint_region(region_b, shape_b, color_b)

    if orient == "horizontal":
        sep = np.full((1, region_w), GREY, dtype=int)
        grid = np.vstack([region_a, sep, region_b])
    else:
        sep = np.full((region_h, 1), GREY, dtype=int)
        grid = np.hstack([region_a, sep, region_b])

    out = main(grid)
    assert np.any(out == RED), "expected at least one highlighted difference"
    assert np.any((out != BG) & (out != RED) & (out != GREY)), "expected at least one shared foreground cell"
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        try:
            return _generate_case()
        except AssertionError:
            continue
    raise RuntimeError(f"failed to generate valid input after {MAX_RETRIES} attempts")
