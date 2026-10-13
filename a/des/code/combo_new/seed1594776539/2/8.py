import random

import numpy as np


BG = 0
FRAME = 5
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in points]
    cols = [c for _r, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _bresenham(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc
    cr, cc = r0, c0
    while True:
        points.append((cr, cc))
        if cr == r1 and cc == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            cr += sr
        if e2 < dr:
            err += dr
            cc += sc
    return points


def _fill_cells(inner_h: int, inner_w: int, marker_count: int) -> set[tuple[int, int]]:
    if marker_count == 1:
        return {(r, c) for r in range(inner_h) for c in range(inner_w)}
    if marker_count == 2:
        return {
            (r, c)
            for r in range(inner_h)
            for c in range(inner_w)
            if r in (0, inner_h - 1) or c in (0, inner_w - 1)
        }
    if marker_count == 3:
        diag = set(_bresenham(0, 0, inner_h - 1, inner_w - 1))
        diag.update(_bresenham(0, inner_w - 1, inner_h - 1, 0))
        return diag
    raise AssertionError(f"unsupported marker count: {marker_count}")


def _extract_regions(grid: np.ndarray) -> list[dict[str, int]]:
    regions: list[dict[str, int]] = []
    for comp in _components(grid == FRAME):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 < 2 or c1 - c0 < 2:
            raise AssertionError("grey component is too small to enclose an interior")

        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                on_border = r in (r0, r1) or c in (c0, c1)
                if on_border:
                    assert int(grid[r, c]) == FRAME, "expected a rectangular grey loop"
                else:
                    assert int(grid[r, c]) != FRAME, "grey pixels may not appear inside the loop"

        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        marker_mask = (inner != BG) & (inner != FRAME)
        marker_points = np.argwhere(marker_mask)
        marker_count = int(marker_points.shape[0])
        assert marker_count in (1, 2, 3), "each interior must contain 1, 2, or 3 markers"

        colors = np.unique(inner[marker_mask])
        assert colors.size == 1, "markers inside one region must share a color"

        regions.append(
            {
                "r0": int(r0),
                "c0": int(c0),
                "r1": int(r1),
                "c1": int(c1),
                "inner_h": int(r1 - r0 - 1),
                "inner_w": int(c1 - c0 - 1),
                "count": marker_count,
                "color": int(colors[0]),
            }
        )

    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each grey enclosed rectangle with one of three patterns chosen by its marker count.

    Input structure:
    - The background is black.
    - Several disjoint grey 1-pixel-thick rectangular loops appear on the grid.
    - Each loop encloses a rectangular interior containing 1, 2, or 3 same-color marker pixels.
    - Marker colors are neither black nor grey, and each region uses only one marker color.

    Rule:
    - For every grey loop, read the unique interior marker color C and count its marker pixels K.
    - If K = 1, fill the entire interior rectangle with C.
    - If K = 2, fill only the interior border ring with C and leave deeper interior cells black.
    - If K = 3, fill only the two corner-to-corner diagonals of the interior rectangle with C.
    - Keep the grey loop unchanged and overwrite the original markers with the chosen fill pattern.

    Key insight:
    - The number of markers selects a discrete fill mode; it does not scale one continuous fill amount.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for region in _extract_regions(g):
        inner_r0 = region["r0"] + 1
        inner_c0 = region["c0"] + 1
        inner_r1 = region["r1"]
        inner_c1 = region["c1"]
        out[inner_r0:inner_r1, inner_c0:inner_c1] = BG

        for dr, dc in _fill_cells(region["inner_h"], region["inner_w"], region["count"]):
            out[inner_r0 + dr, inner_c0 + dc] = region["color"]

    return out


def _sample_region_specs() -> list[dict[str, int]]:
    n_regions = random.randint(3, 5)
    counts = [1, 2, 3] + [random.randint(1, 3) for _ in range(n_regions - 3)]
    random.shuffle(counts)
    colors = random.sample(MARKER_COLORS, n_regions)

    specs: list[dict[str, int]] = []
    for count, color in zip(counts, colors):
        inner_h = random.randint(3, 7)
        inner_w = random.randint(3, 7)
        if count == 3 and random.random() < 0.7 and inner_h == inner_w:
            if inner_w < 7:
                inner_w += 1
            elif inner_h > 3:
                inner_h -= 1
        specs.append(
            {
                "inner_h": inner_h,
                "inner_w": inner_w,
                "count": count,
                "color": color,
            }
        )
    return specs


def _place_regions(specs: list[dict[str, int]], height: int, width: int):
    occupancy = np.zeros((height, width), dtype=bool)
    placed: list[dict[str, int]] = []

    for spec in sorted(specs, key=lambda s: (s["inner_h"] + 2) * (s["inner_w"] + 2), reverse=True):
        box_h = spec["inner_h"] + 2
        box_w = spec["inner_w"] + 2
        if box_h + 2 > height or box_w + 2 > width:
            return None

        candidates: list[tuple[int, int]] = []
        for r0 in range(1, height - box_h):
            for c0 in range(1, width - box_w):
                r1 = r0 + box_h - 1
                c1 = c0 + box_w - 1
                pad = occupancy[max(0, r0 - 1) : min(height, r1 + 2), max(0, c0 - 1) : min(width, c1 + 2)]
                if not pad.any():
                    candidates.append((r0, c0))
        if not candidates:
            return None

        r0, c0 = random.choice(candidates)
        r1 = r0 + box_h - 1
        c1 = c0 + box_w - 1
        occupancy[max(0, r0 - 1) : min(height, r1 + 2), max(0, c0 - 1) : min(width, c1 + 2)] = True

        placed.append(
            {
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
                "inner_h": spec["inner_h"],
                "inner_w": spec["inner_w"],
                "count": spec["count"],
                "color": spec["color"],
            }
        )

    return placed


def _draw_input(placed: list[dict[str, int]], height: int, width: int) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)

    for region in placed:
        r0, c0, r1, c1 = region["r0"], region["c0"], region["r1"], region["c1"]
        grid[r0, c0 : c1 + 1] = FRAME
        grid[r1, c0 : c1 + 1] = FRAME
        grid[r0 : r1 + 1, c0] = FRAME
        grid[r0 : r1 + 1, c1] = FRAME

        fill_cells = sorted(_fill_cells(region["inner_h"], region["inner_w"], region["count"]))
        assert len(fill_cells) >= region["count"], "not enough cells to place markers"
        marker_cells = random.sample(fill_cells, region["count"])
        for dr, dc in marker_cells:
            grid[r0 + 1 + dr, c0 + 1 + dc] = region["color"]

    return grid


def _expected_output(grid: np.ndarray, placed: list[dict[str, int]]) -> np.ndarray:
    out = grid.copy()
    for region in placed:
        inner_r0 = region["r0"] + 1
        inner_c0 = region["c0"] + 1
        out[inner_r0 : region["r1"], inner_c0 : region["c1"]] = BG
        for dr, dc in _fill_cells(region["inner_h"], region["inner_w"], region["count"]):
            out[inner_r0 + dr, inner_c0 + dc] = region["color"]
    return out


def _gen_attempt() -> np.ndarray | None:
    specs = _sample_region_specs()
    height = random.randint(20, 30)
    width = random.randint(20, 30)

    padded_area = sum((spec["inner_h"] + 4) * (spec["inner_w"] + 4) for spec in specs)
    if padded_area > int(0.42 * height * width):
        return None

    placed = _place_regions(specs, height, width)
    if placed is None:
        return None

    grid = _draw_input(placed, height, width)
    expected = _expected_output(grid, placed)

    try:
        regions = _extract_regions(grid)
    except AssertionError:
        return None

    if len(regions) != len(placed):
        return None

    if not np.array_equal(main(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
