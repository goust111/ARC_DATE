from __future__ import annotations

from collections import deque
from pathlib import Path
import random
import sys

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, neighbors4


MARKER_COLORS = (Color.BLUE, Color.RED, Color.GREEN)


def _row_intervals(row_mask: np.ndarray) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    c = 0
    width = int(row_mask.shape[0])
    while c < width:
        if not bool(row_mask[c]):
            c += 1
            continue
        start = c
        while c + 1 < width and bool(row_mask[c + 1]):
            c += 1
        intervals.append((start, c))
        c += 1
    return intervals


def _rectangles_from_gray(gray_mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    active: dict[tuple[int, int], int] = {}
    rects: list[tuple[int, int, int, int]] = []
    h, _w = gray_mask.shape

    for r in range(h):
        spans = _row_intervals(gray_mask[r])
        span_set = set(spans)

        for span, start_r in list(active.items()):
            if span not in span_set:
                c0, c1 = span
                rects.append((start_r, c0, r - start_r, c1 - c0 + 1))
                del active[span]

        for span in spans:
            if span not in active:
                active[span] = r

    for span, start_r in active.items():
        c0, c1 = span
        rects.append((start_r, c0, h - start_r, c1 - c0 + 1))

    rects = sorted(rects)
    assert 4 <= len(rects) <= 6, "expected 4-6 gray rectangles"
    for r0, c0, hh, ww in rects:
        assert hh >= 3 and ww >= 3, "each rectangle must be at least 3x3"
        assert np.all(gray_mask[r0 : r0 + hh, c0 : c0 + ww]), "rectangle must be solid"

    rebuilt = np.zeros_like(gray_mask, dtype=bool)
    for r0, c0, hh, ww in rects:
        rebuilt[r0 : r0 + hh, c0 : c0 + ww] = True
    assert np.array_equal(rebuilt, gray_mask), "gray mask must decompose exactly into rectangles"
    return rects


def _draw_walls(shape: tuple[int, int], rects: list[tuple[int, int, int, int]]) -> np.ndarray:
    walls = np.zeros(shape, dtype=int)
    for r0, c0, hh, ww in rects:
        r1 = r0 + hh - 1
        c1 = c0 + ww - 1
        walls[r0, c0 : c1 + 1] = Color.GREY
        walls[r1, c0 : c1 + 1] = Color.GREY
        walls[r0 : r1 + 1, c0] = Color.GREY
        walls[r0 : r1 + 1, c1] = Color.GREY
    return walls


def _find_markers(grid: np.ndarray) -> dict[int, tuple[int, int]]:
    markers: dict[int, tuple[int, int]] = {}
    for color in MARKER_COLORS:
        coords = np.argwhere(grid == color)
        assert coords.shape == (1, 2), "expected exactly one marker of each color"
        markers[int(color)] = (int(coords[0, 0]), int(coords[0, 1]))

    marker_cells = list(markers.values())
    for i, (r0, c0) in enumerate(marker_cells):
        for r1, c1 in marker_cells[i + 1 :]:
            assert abs(r0 - r1) + abs(c0 - c1) > 1, "markers must not be 4-neighbor adjacent"
    return markers


def _exterior_black(base: np.ndarray) -> np.ndarray:
    h, w = base.shape
    exterior = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    def enqueue(r: int, c: int) -> None:
        if 0 <= r < h and 0 <= c < w and not exterior[r, c] and int(base[r, c]) == Color.BLACK:
            exterior[r, c] = True
            queue.append((r, c))

    for r in range(h):
        enqueue(r, 0)
        enqueue(r, w - 1)
    for c in range(w):
        enqueue(0, c)
        enqueue(h - 1, c)

    while queue:
        r, c = queue.popleft()
        for rr, cc in neighbors4(r, c):
            enqueue(int(rr), int(cc))
    return exterior


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            queue: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    nr = int(nr)
                    nc = int(nc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _region_centroid(coords: np.ndarray) -> tuple[int, int]:
    mean_r = int(np.floor(coords[:, 0].mean()))
    mean_c = int(np.floor(coords[:, 1].mean()))
    return mean_r, mean_c


def _solve_with_analysis(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must be at most 30x30"
    assert set(map(int, np.unique(g))) <= {Color.BLACK, Color.BLUE, Color.RED, Color.GREEN, Color.GREY}

    markers = _find_markers(g)
    gray_mask = g == Color.GREY
    rects = _rectangles_from_gray(gray_mask)

    walls = _draw_walls(g.shape, rects)
    base = np.zeros_like(g)
    base[walls == Color.GREY] = Color.GREY
    for color, (r, c) in markers.items():
        assert int(base[r, c]) == Color.BLACK and int(g[r, c]) == color
        base[r, c] = int(color)

    exterior = _exterior_black(base)
    enclosed_mask = (base == Color.BLACK) & ~exterior
    regions = _components(enclosed_mask)
    assert len(regions) >= 2, "expected at least two enclosed regions"

    out = base.copy()
    region_infos: list[dict] = []
    for coords in regions:
        centroid = _region_centroid(coords)
        dists = {
            color: abs(centroid[0] - mr) + abs(centroid[1] - mc)
            for color, (mr, mc) in markers.items()
        }
        best_dist = min(dists.values())
        winners = [color for color, dist in dists.items() if dist == best_dist]
        assert len(winners) == 1, "nearest marker must be unique"
        fill_color = int(winners[0])
        out[coords[:, 0], coords[:, 1]] = fill_color
        region_infos.append(
            {
                "coords": coords,
                "coord_set": frozenset((int(r), int(c)) for r, c in coords),
                "centroid": centroid,
                "fill_color": fill_color,
                "distances": dists,
            }
        )

    return out, {"rectangles": rects, "markers": markers, "regions": region_infos, "walls": walls}


def main(grid: np.ndarray) -> np.ndarray:
    """Hollow each gray rectangle into a wall network, then fill every enclosed region with its nearest marker color.

    Input structure:
    - The grid has a black background, 4-6 solid gray axis-aligned rectangles, and exactly three single-pixel markers.
    - The markers are blue, red, and green, all placed in the exterior background and not 4-neighbor adjacent.
    - Some rectangles share horizontal boundary segments so that, once hollowed, their borders form one connected wall network with multiple enclosed black regions.

    Rule:
    - Treat every maximal gray row-span band as one solid rectangle and hollow it: keep only its one-cell-thick border and erase its strict interior to black.
    - Use the gray border pixels as walls and flood-fill from the grid boundary through 4-neighbor black cells to mark the exterior.
    - The remaining black connected components are enclosed regions.
    - For each enclosed region, compute the floor of its centroid and fill the whole region with the color of the uniquely nearest marker by Manhattan distance.
    - Keep the gray walls, the three markers, and the exterior background unchanged.

    Key insight:
    - The solid rectangles are only a precursor: after hollowing, their borders partition space, and the color decision is made per enclosed region rather than per rectangle.
    """
    out, _analysis = _solve_with_analysis(grid)
    return out


def _rectangle_interior(rect: tuple[int, int, int, int]) -> frozenset[tuple[int, int]]:
    r0, c0, hh, ww = rect
    return frozenset(
        (r, c)
        for r in range(r0 + 1, r0 + hh - 1)
        for c in range(c0 + 1, c0 + ww - 1)
    )


def _exterior_mask_for_input(grid: np.ndarray) -> np.ndarray:
    blocked = grid != Color.BLACK
    h, w = grid.shape
    exterior = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    def enqueue(r: int, c: int) -> None:
        if 0 <= r < h and 0 <= c < w and not exterior[r, c] and not bool(blocked[r, c]):
            exterior[r, c] = True
            queue.append((r, c))

    for r in range(h):
        enqueue(r, 0)
        enqueue(r, w - 1)
    for c in range(w):
        enqueue(0, c)
        enqueue(h - 1, c)

    while queue:
        r, c = queue.popleft()
        for rr, cc in neighbors4(r, c):
            enqueue(int(rr), int(cc))
    return exterior


def _candidate_marker_cells(
    exterior: np.ndarray,
    occupied: set[tuple[int, int]],
    region_name: str,
    bbox: tuple[int, int, int, int],
) -> list[tuple[int, int]]:
    h, w = exterior.shape
    r0, c0, r1, c1 = bbox
    center_r = (r0 + r1) // 2
    center_c = (c0 + c1) // 2
    cells: list[tuple[int, int]] = []

    for r in range(h):
        for c in range(w):
            if not bool(exterior[r, c]) or (r, c) in occupied:
                continue
            if region_name == "nw" and not (r < center_r and c < center_c):
                continue
            if region_name == "ne" and not (r < center_r and c > center_c):
                continue
            if region_name == "south" and not (r > center_r):
                continue
            if any(abs(r - rr) + abs(c - cc) <= 1 for rr, cc in occupied):
                continue
            if not all(
                0 <= nr < h and 0 <= nc < w and bool(exterior[nr, nc])
                for nr, nc in neighbors4(r, c)
            ):
                continue
            cells.append((r, c))
    random.shuffle(cells)
    return cells


def _place_markers(
    grid: np.ndarray,
    union_bbox: tuple[int, int, int, int],
) -> np.ndarray | None:
    exterior = _exterior_mask_for_input(grid)
    occupied: set[tuple[int, int]] = set()
    sectors = ("nw", "ne", "south")
    candidates = {
        sector: _candidate_marker_cells(exterior, occupied, sector, union_bbox)
        for sector in sectors
    }
    if any(not cells for cells in candidates.values()):
        return None

    sector_colors = list(MARKER_COLORS)
    random.shuffle(sector_colors)
    result = grid.copy()
    for sector, color in zip(sectors, sector_colors):
        choices = _candidate_marker_cells(exterior, occupied, sector, union_bbox)
        if not choices:
            return None
        cell = choices[0]
        occupied.add(cell)
        result[cell] = int(color)
    return result


def _region_set_from_box(r0: int, c0: int, r1: int, c1: int) -> frozenset[tuple[int, int]]:
    return frozenset((r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1))


def _build_layout() -> tuple[np.ndarray, list[tuple[int, int, int, int]], frozenset[tuple[int, int]]]:
    inner_h = random.randint(3, 5)
    inner_w = random.randint(3, 5)
    top_h = random.randint(3, 4)
    bottom_h = random.randint(3, 4)
    left_w = random.randint(3, 4)
    right_w = random.randint(3, 4)
    outer_w = left_w + inner_w + right_w
    outer_h = top_h + inner_h + bottom_h

    add_top = random.choice([False, True])
    add_bottom = random.choice([False, True])
    top_extra_h = random.randint(3, 4) if add_top else 0
    bottom_extra_h = random.randint(3, 4) if add_bottom else 0

    h = top_extra_h + outer_h + bottom_extra_h + random.randint(5, 8)
    w = outer_w + random.randint(7, 11)
    grid = np.zeros((h, w), dtype=int)

    margin_top = random.randint(2, 4)
    margin_left = random.randint(2, w - outer_w - 3)
    base_r = margin_top + top_extra_h
    base_c = margin_left

    rects = [
        (base_r, base_c, top_h, outer_w),
        (base_r + top_h + inner_h, base_c, bottom_h, outer_w),
        (base_r + top_h, base_c, inner_h, left_w),
        (base_r + top_h, base_c + left_w + inner_w, inner_h, right_w),
    ]

    if add_top:
        top_w = random.randint(3, outer_w - 2)
        top_c = base_c + random.randint(1, outer_w - top_w - 1)
        rects.append((base_r - top_extra_h, top_c, top_extra_h, top_w))
    if add_bottom:
        bottom_w = random.randint(3, outer_w - 2)
        bottom_c = base_c + random.randint(1, outer_w - bottom_w - 1)
        rects.append((base_r + outer_h, bottom_c, bottom_extra_h, bottom_w))

    for r0, c0, hh, ww in rects:
        grid[r0 : r0 + hh, c0 : c0 + ww] = Color.GREY

    central_gap = _region_set_from_box(
        base_r + top_h,
        base_c + left_w,
        base_r + top_h + inner_h - 1,
        base_c + left_w + inner_w - 1,
    )
    return grid, sorted(rects), central_gap


def _validate_generated(
    grid: np.ndarray,
    planned_rects: list[tuple[int, int, int, int]],
    central_gap: frozenset[tuple[int, int]],
) -> bool:
    try:
        out, analysis = _solve_with_analysis(grid)
    except AssertionError:
        return False

    if len(analysis["rectangles"]) != len(planned_rects):
        return False
    if analysis["rectangles"] != planned_rects:
        return False

    region_sets = [info["coord_set"] for info in analysis["regions"]]
    if len(region_sets) != len(planned_rects) + 1:
        return False
    if central_gap not in region_sets:
        return False

    for rect in planned_rects:
        if _rectangle_interior(rect) not in region_sets:
            return False

    if len({info["fill_color"] for info in analysis["regions"]}) != 3:
        return False

    non_rect_regions = [
        region
        for region in region_sets
        if region not in {_rectangle_interior(rect) for rect in planned_rects}
    ]
    if non_rect_regions != [central_gap]:
        return False

    recomputed = main(grid)
    if not np.array_equal(recomputed, out):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    solid_grid, rects, central_gap = _build_layout()
    rows, cols = np.nonzero(solid_grid == Color.GREY)
    bbox = (int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max()))

    for _ in range(40):
        grid = _place_markers(solid_grid, bbox)
        if grid is None:
            return None
        if _validate_generated(grid, rects, central_gap):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
