from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8

PLUS_OFFSETS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
RAY_DIRS = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
)
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
NEIGHBORS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
INPUT_MARKER_COLORS = (BLUE, GREEN, YELLOW, PINK, ORANGE, TEAL)
COLOR_SWAP = {
    BLUE: YELLOW,
    YELLOW: BLUE,
    GREEN: PINK,
    PINK: GREEN,
    ORANGE: TEAL,
    TEAL: ORANGE,
}

GEN_HEIGHT_RANGE = (13, 19)
GEN_WIDTH_RANGE = (13, 19)
GEN_MARKER_COUNT_RANGE = (3, 5)
GEN_RETRY_LIMIT = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _plus_cells(center: tuple[int, int]) -> list[tuple[int, int]]:
    cr, cc = center
    return [(cr + dr, cc + dc) for dr, dc in PLUS_OFFSETS]


def _find_plus_center(grid: np.ndarray) -> tuple[int, int]:
    red_cells = np.argwhere(grid == RED)
    assert red_cells.shape == (5, 2), "expected exactly one red five-pixel plus"
    coords = {(int(r), int(c)) for r, c in red_cells}
    r0, c0 = red_cells.min(axis=0)
    r1, c1 = red_cells.max(axis=0)
    assert (int(r1 - r0), int(c1 - c0)) == (2, 2), "red plus must occupy a 3x3 box"
    center = (int(r0 + 1), int(c0 + 1))
    assert coords == set(_plus_cells(center)), "red cells must form a plus"
    return center


def _build_partition(shape: tuple[int, int], center: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    base = np.full(shape, BG, dtype=int)
    blocked = np.zeros(shape, dtype=bool)

    for r, c in _plus_cells(center):
        assert _in_bounds(shape, r, c), "plus must fit in the grid"
        base[r, c] = RED
        blocked[r, c] = True

    cr, cc = center
    for dr, dc in RAY_DIRS:
        step = 2 if dr == 0 or dc == 0 else 1
        r = cr + step * dr
        c = cc + step * dc
        while _in_bounds(shape, r, c):
            assert not blocked[r, c], "ray unexpectedly overlaps the plus"
            base[r, c] = GREY
            blocked[r, c] = True
            r += dr
            c += dc

    return base, blocked


def _label_regions(passable: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    h, w = passable.shape
    labels = np.full((h, w), -1, dtype=int)
    regions: list[dict] = []

    for sr, sc in np.argwhere(passable):
        sr = int(sr)
        sc = int(sc)
        if labels[sr, sc] >= 0:
            continue

        region_id = len(regions)
        q = deque([(sr, sc)])
        labels[sr, sc] = region_id
        coords: list[tuple[int, int]] = []
        min_r = max_r = sr
        min_c = max_c = sc

        while q:
            r, c = q.popleft()
            coords.append((r, c))
            min_r = min(min_r, r)
            max_r = max(max_r, r)
            min_c = min(min_c, c)
            max_c = max(max_c, c)
            for dr, dc in NEIGHBORS4:
                nr = r + dr
                nc = c + dc
                if _in_bounds((h, w), nr, nc) and passable[nr, nc] and labels[nr, nc] < 0:
                    labels[nr, nc] = region_id
                    q.append((nr, nc))

        area = len(coords)
        bbox_area = (max_r - min_r + 1) * (max_c - min_c + 1)
        regions.append(
            {
                "coords": coords,
                "area": area,
                "bbox": (min_r, min_c, max_r, max_c),
                "bbox_area": bbox_area,
                "is_rect": area == bbox_area,
            }
        )

    return labels, regions


def _analyze_partition(shape: tuple[int, int], center: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    base, blocked = _build_partition(shape, center)
    labels, regions = _label_regions(~blocked)
    return base, blocked, labels, regions


def _is_isolated_marker(grid: np.ndarray, r: int, c: int) -> bool:
    for dr, dc in NEIGHBORS8:
        nr = r + dr
        nc = c + dc
        if _in_bounds(grid.shape, nr, nc) and grid[nr, nc] != BG:
            return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Partition the canvas with eight rays from a red plus, then fill marked regions by geometric type.

    Input structure:
    - Black background with exactly one red five-pixel plus.
    - Several isolated single-pixel markers use colors from the swap pairs blue/yellow, green/pink, and orange/teal.
    - The markers lie off the plus and off the future ray paths; the plus position is off-center so the ray partition contains both rectangular and non-rectangular regions.

    Rule:
    - Preserve the red plus and draw grey rays from the plus center in all eight directions until the grid edge, without repainting the plus itself.
    - Treat only the plus and rays as barriers, so each marker occupies one 4-connected region of the remaining cells; classify a region as rectangular when its area equals its inclusive bounding-box area, otherwise non-rectangular.
    - For each marked region, fill the whole region with the marker color if the region is rectangular, or with the fixed partner color if the region is non-rectangular: blue<->yellow, green<->pink, orange<->teal.
    - Regions without a marker stay black.

    Key insight:
    - The visible marker color alone is insufficient: the decisive signal is the geometry of the ray-created region that contains the marker, so the solver must reconstruct the partition before choosing whether to keep or swap the color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    center = _find_plus_center(g)
    marker_coords = np.argwhere((g != BG) & (g != RED))
    for r, c in marker_coords:
        color = int(g[int(r), int(c)])
        assert color in COLOR_SWAP, "unexpected marker color"
        assert _is_isolated_marker(g, int(r), int(c)), "markers must be isolated single pixels"

    base, blocked, labels, regions = _analyze_partition(g.shape, center)
    out = base.copy()

    region_to_marker: dict[int, int] = {}
    for r, c in marker_coords:
        r = int(r)
        c = int(c)
        assert not blocked[r, c], "marker cannot lie on the plus or a ray"
        region_id = int(labels[r, c])
        assert region_id >= 0, "marker must lie inside a region"
        assert region_id not in region_to_marker, "each marked region must contain only one marker"
        region_to_marker[region_id] = int(g[r, c])

    for region_id, marker_color in region_to_marker.items():
        fill_color = marker_color if regions[region_id]["is_rect"] else COLOR_SWAP[marker_color]
        for r, c in regions[region_id]["coords"]:
            out[r, c] = fill_color

    return out


def _center_is_off_axis(center: tuple[int, int], shape: tuple[int, int]) -> bool:
    cr, cc = center
    h, w = shape
    return abs(2 * cr - (h - 1)) >= 3 or abs(2 * cc - (w - 1)) >= 3


def _candidate_centers(h: int, w: int) -> list[tuple[int, int]]:
    centers: list[tuple[int, int]] = []

    def consider(center: tuple[int, int]) -> None:
        if not _center_is_off_axis(center, (h, w)):
            return
        _base, _blocked, _labels, regions = _analyze_partition((h, w), center)
        rect_sizes = [reg["area"] for reg in regions if reg["is_rect"]]
        nonrect_count = sum(not reg["is_rect"] for reg in regions)
        if len([size for size in rect_sizes if size >= 2]) >= 2 and nonrect_count >= 2 and len(regions) >= 5:
            centers.append(center)

    for c in range(3, w - 3):
        consider((1, c))
        consider((h - 2, c))
    for r in range(3, h - 3):
        consider((r, 1))
        consider((r, w - 2))

    return centers


def _candidate_marker_cells(
    region: dict,
    shape: tuple[int, int],
    plus_cells: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for r, c in region["coords"]:
        if (r, c) in plus_cells:
            continue
        ok = True
        for dr, dc in NEIGHBORS8:
            nr = r + dr
            nc = c + dc
            if _in_bounds(shape, nr, nc) and (nr, nc) in plus_cells:
                ok = False
                break
        if ok:
            cells.append((r, c))
    random.shuffle(cells)
    return cells


def _king_far_enough(cells: list[tuple[int, int]], r: int, c: int) -> bool:
    return all(max(abs(r - pr), abs(c - pc)) > 1 for pr, pc in cells)


def _choose_markers(
    regions: list[dict],
    shape: tuple[int, int],
    center: tuple[int, int],
) -> list[tuple[int, int, int, bool]]:
    plus_set = set(_plus_cells(center))
    usable_rect = []
    usable_nonrect = []
    for idx, region in enumerate(regions):
        candidates = _candidate_marker_cells(region, shape, plus_set)
        if not candidates:
            continue
        entry = (idx, candidates)
        if region["is_rect"]:
            usable_rect.append(entry)
        else:
            usable_nonrect.append(entry)

    max_markers = min(GEN_MARKER_COUNT_RANGE[1], len(usable_rect) + len(usable_nonrect), len(INPUT_MARKER_COLORS))
    min_markers = min(GEN_MARKER_COUNT_RANGE[0], max_markers)
    if max_markers < GEN_MARKER_COUNT_RANGE[0] or not usable_rect or not usable_nonrect:
        raise ValueError("not enough usable regions for markers")

    marker_count = random.randint(min_markers, max_markers)
    min_rect = max(1, marker_count - len(usable_nonrect))
    max_rect = min(len(usable_rect), marker_count - 1)
    rect_count = random.randint(min_rect, max_rect)
    nonrect_count = marker_count - rect_count

    chosen_regions = random.sample(usable_rect, rect_count) + random.sample(usable_nonrect, nonrect_count)
    random.shuffle(chosen_regions)
    chosen_colors = random.sample(INPUT_MARKER_COLORS, marker_count)

    placed: list[tuple[int, int, int, bool]] = []
    occupied: list[tuple[int, int]] = []
    for color, (region_id, candidates) in zip(chosen_colors, chosen_regions):
        pick = None
        for r, c in candidates:
            if _king_far_enough(occupied, r, c):
                pick = (r, c)
                break
        if pick is None:
            raise ValueError("failed to place isolated markers")
        occupied.append(pick)
        placed.append((pick[0], pick[1], color, regions[region_id]["is_rect"]))

    rect_hits = sum(is_rect for _r, _c, _color, is_rect in placed)
    assert 1 <= rect_hits < len(placed)
    return placed


def _verify_semantics(inp: np.ndarray, out: np.ndarray) -> bool:
    center = _find_plus_center(inp)
    base, blocked, labels, regions = _analyze_partition(inp.shape, center)

    if not np.array_equal(out[blocked], base[blocked]):
        return False

    marker_coords = np.argwhere((inp != BG) & (inp != RED))
    seen_regions: set[int] = set()
    for r, c in marker_coords:
        r = int(r)
        c = int(c)
        region_id = int(labels[r, c])
        if region_id < 0 or region_id in seen_regions:
            return False
        seen_regions.add(region_id)
        marker_color = int(inp[r, c])
        target = marker_color if regions[region_id]["is_rect"] else COLOR_SWAP[marker_color]
        for rr, cc in regions[region_id]["coords"]:
            if int(out[rr, cc]) != target:
                return False

    for region_id, region in enumerate(regions):
        if region_id in seen_regions:
            continue
        for r, c in region["coords"]:
            if int(out[r, c]) != BG:
                return False

    rect_count = sum(region["is_rect"] for region in regions)
    nonrect_count = len(regions) - rect_count
    rect_markers = sum(regions[int(labels[int(r), int(c)])]["is_rect"] for r, c in marker_coords)
    return rect_count >= 2 and nonrect_count >= 2 and 0 < rect_markers < len(marker_coords)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        h = random.randint(*GEN_HEIGHT_RANGE)
        w = random.randint(*GEN_WIDTH_RANGE)
        centers = _candidate_centers(h, w)
        if not centers:
            continue

        center = random.choice(centers)
        _base, _blocked, _labels, regions = _analyze_partition((h, w), center)
        try:
            markers = _choose_markers(regions, (h, w), center)
        except ValueError:
            continue

        inp = np.full((h, w), BG, dtype=int)
        for r, c in _plus_cells(center):
            inp[r, c] = RED
        for r, c, color, _is_rect in markers:
            inp[r, c] = color

        out = main(inp)
        if _verify_semantics(inp, out):
            return inp

    raise RuntimeError("failed to generate a valid ray-partition sample")
