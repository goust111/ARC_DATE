import random
from typing import Iterable

import numpy as np


BG = 0
YELLOW = 4
FRAME_COLORS = [1, 2, 3, 5, 6, 7, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 400
REGION_PLACEMENT_RETRIES = 120
MARKER_RETRIES = 80


def _perimeter_positions(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    pts: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        pts.append((r0, c))
    for r in range(r0 + 1, r1):
        pts.append((r, c1))
    if r1 > r0:
        for c in range(c1, c0 - 1, -1):
            pts.append((r1, c))
    if c1 > c0:
        for r in range(r1 - 1, r0, -1):
            pts.append((r, c0))
    return pts


def _cross_cells(r: int, c: int) -> list[tuple[int, int]]:
    return [(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _chunk_points(points: list[tuple[int, int]], size: int) -> list[list[tuple[int, int]]]:
    return [points[i : i + size] for i in range(0, len(points), size)]


def _median_center(points: Iterable[tuple[int, int]]) -> tuple[int, int]:
    rows = sorted(r for r, _ in points)
    cols = sorted(c for _, c in points)
    mid = len(rows) // 2
    if len(rows) % 2 == 1:
        return rows[mid], cols[mid]
    return (rows[mid - 1] + rows[mid]) // 2, (cols[mid - 1] + cols[mid]) // 2


def _extract_regions(grid: np.ndarray) -> list[dict]:
    regions: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in (BG, YELLOW)):
        coords = np.argwhere(grid == color)
        if coords.size == 0:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        r0 = int(r0)
        c0 = int(c0)
        r1 = int(r1)
        c1 = int(c1)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue

        border = {tuple(map(int, pt)) for pt in coords}
        perimeter = _perimeter_positions(r0, c0, r1, c1)
        gap_count = sum((r, c) not in border for r, c in perimeter)
        markers = [
            (r, c)
            for r in range(r0 + 1, r1)
            for c in range(c0 + 1, c1)
            if int(grid[r, c]) == YELLOW
        ]
        regions.append(
            {
                "color": color,
                "bbox": (r0, c0, r1, c1),
                "gap_count": int(gap_count),
                "markers": markers,
            }
        )

    regions.sort(key=lambda region: region["bbox"])
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the one-cell gaps on each colored rectangular frame, use that count as
    the grouping size for the yellow markers inside the frame, then replace each
    group with a same-colored cross at the group's median center.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    out[out == YELLOW] = BG

    for region in _extract_regions(g):
        group_size = int(region["gap_count"])
        if group_size <= 0:
            continue

        r0, c0, r1, c1 = region["bbox"]
        color = int(region["color"])
        markers = list(region["markers"])
        groups = _chunk_points(markers, group_size)

        for group in groups:
            if not group:
                continue
            cr, cc = _median_center(group)
            for rr, cc2 in _cross_cells(cr, cc):
                if r0 < rr < r1 and c0 < cc2 < c1:
                    out[rr, cc2] = color

    return out


def _sample_gap_positions(r0: int, c0: int, r1: int, c1: int, gap_count: int, rng: random.Random) -> list[tuple[int, int]]:
    sides = {
        "top": [(r0, c) for c in range(c0 + 1, c1)],
        "bottom": [(r1, c) for c in range(c0 + 1, c1)],
        "left": [(r, c0) for r in range(r0 + 1, r1)],
        "right": [(r, c1) for r in range(r0 + 1, r1)],
    }
    chosen_sides = rng.sample(list(sides.keys()), gap_count)
    return [rng.choice(sides[side]) for side in chosen_sides]


def _sample_nonadjacent_markers(
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    marker_count: int,
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    candidates = [(r, c) for r in range(r0 + 2, r1 - 1) for c in range(c0 + 2, c1 - 1)]
    if len(candidates) < marker_count:
        return None

    for _ in range(MARKER_RETRIES):
        shuffled = candidates[:]
        rng.shuffle(shuffled)
        picked: list[tuple[int, int]] = []
        for r, c in shuffled:
            if all(max(abs(r - pr), abs(c - pc)) > 1 for pr, pc in picked):
                picked.append((r, c))
                if len(picked) == marker_count:
                    return picked
    return None


def _cross_footprints_ok(
    bbox: tuple[int, int, int, int],
    markers: list[tuple[int, int]],
    gap_count: int,
) -> bool:
    r0, c0, r1, c1 = bbox
    used: set[tuple[int, int]] = set()
    for group in _chunk_points(sorted(markers), gap_count):
        cr, cc = _median_center(group)
        cells = _cross_cells(cr, cc)
        if any(not (r0 < rr < r1 and c0 < cc2 < c1) for rr, cc2 in cells):
            return False
        if any((rr, cc2) in used for rr, cc2 in cells):
            return False
        used.update(cells)
    return True


def _can_place(occupied: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 2) -> bool:
    h, w = occupied.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    return not bool(occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _reserve(occupied: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 2) -> None:
    h, w = occupied.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _paint_region(
    grid: np.ndarray,
    color: int,
    bbox: tuple[int, int, int, int],
    gaps: list[tuple[int, int]],
    markers: list[tuple[int, int]],
) -> None:
    r0, c0, r1, c1 = bbox
    for r, c in _perimeter_positions(r0, c0, r1, c1):
        grid[r, c] = color
    for r, c in gaps:
        grid[r, c] = BG
    for r, c in markers:
        grid[r, c] = YELLOW


def _generated_grid_is_valid(grid: np.ndarray, specs: list[dict]) -> bool:
    regions = _extract_regions(grid)
    if len(regions) != len(specs):
        return False

    spec_by_color = {int(spec["color"]): spec for spec in specs}
    covered_markers: set[tuple[int, int]] = set()

    for region in regions:
        color = int(region["color"])
        if color not in spec_by_color:
            return False
        spec = spec_by_color[color]
        if tuple(region["bbox"]) != tuple(spec["bbox"]):
            return False
        if int(region["gap_count"]) != int(spec["gap_count"]):
            return False
        if len(region["markers"]) != int(spec["marker_count"]):
            return False
        covered_markers.update(region["markers"])
        if not _cross_footprints_ok(region["bbox"], region["markers"], region["gap_count"]):
            return False

    all_markers = {tuple(map(int, pt)) for pt in np.argwhere(grid == YELLOW)}
    if all_markers != covered_markers:
        return False

    try:
        out = main(grid)
    except Exception:
        return False
    if out.shape != grid.shape:
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    region_count = rng.randint(1, 4)
    min_side = 16 + region_count * 2
    h = rng.randint(min_side, 30)
    w = rng.randint(min_side, 30)
    grid = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    colors = rng.sample(FRAME_COLORS, region_count)
    size_caps = {1: 14, 2: 12, 3: 10, 4: 9}
    size_cap = size_caps[region_count]
    specs: list[dict] = []

    for color in colors:
        placed = False
        for _ in range(REGION_PLACEMENT_RETRIES):
            max_h = min(size_cap, h - 2)
            max_w = min(size_cap + 1, w - 2)
            if max_h < 7 or max_w < 7:
                return None
            rect_h = rng.randint(7, max_h)
            rect_w = rng.randint(7, max_w)
            if h - rect_h - 2 < 0 or w - rect_w - 2 < 0:
                continue

            r0 = rng.randint(1, h - rect_h - 1)
            c0 = rng.randint(1, w - rect_w - 1)
            r1 = r0 + rect_h - 1
            c1 = c0 + rect_w - 1
            bbox = (r0, c0, r1, c1)
            if not _can_place(occupied, r0, c0, r1, c1):
                continue

            gap_count = rng.randint(1, 4)
            usable_h = rect_h - 4
            usable_w = rect_w - 4
            marker_capacity = ((usable_h + 1) // 2) * ((usable_w + 1) // 2)
            cross_capacity = max(1, ((usable_h + 2) // 3) * ((usable_w + 2) // 3))
            max_groups = min(3, cross_capacity, 1 + (marker_capacity - 1) // gap_count)
            if max_groups <= 0:
                continue

            group_count = rng.randint(1, max_groups)
            max_last = min(gap_count, marker_capacity - gap_count * (group_count - 1))
            if max_last <= 0:
                continue
            last_group_size = rng.randint(1, max_last)
            marker_count = gap_count * (group_count - 1) + last_group_size

            gaps = _sample_gap_positions(r0, c0, r1, c1, gap_count, rng)
            markers = _sample_nonadjacent_markers(r0, c0, r1, c1, marker_count, rng)
            if markers is None:
                continue
            if not _cross_footprints_ok(bbox, markers, gap_count):
                continue

            _paint_region(grid, color, bbox, gaps, markers)
            _reserve(occupied, r0, c0, r1, c1)
            specs.append(
                {
                    "color": color,
                    "bbox": bbox,
                    "gap_count": gap_count,
                    "marker_count": marker_count,
                }
            )
            placed = True
            break

        if not placed:
            return None

    if not _generated_grid_is_valid(grid, specs):
        return None
    return grid


def generate_input() -> np.ndarray:
    """
    Generate a random grid with 1..4 distinct rectangular frames. Each frame has
    1..4 one-cell gaps and a set of non-adjacent yellow markers inside.
    """
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted retries")
