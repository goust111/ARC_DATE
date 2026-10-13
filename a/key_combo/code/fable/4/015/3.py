from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, bbox_h, bbox_w, find_components, in_bounds

Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

BG = Color.BLACK
DIRS: tuple[Coord, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
SIDES = ("top", "bottom", "left", "right")


def _neighbors8(r: int, c: int) -> list[Coord]:
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if (dr, dc) != (0, 0)
    ]


def _max_layer_depth(box: BBox) -> int:
    return (min(bbox_h(box), bbox_w(box)) - 1) // 2


def _paint_layer(grid: np.ndarray, box: BBox, depth: int, color: int) -> None:
    r0, c0, r1, c1 = box
    rows = np.arange(r0, r1 + 1)[:, None]
    cols = np.arange(c0, c1 + 1)[None, :]
    layer_depth = np.minimum(
        np.minimum(rows - r0, r1 - rows),
        np.minimum(cols - c0, c1 - cols),
    )
    region = grid[r0 : r1 + 1, c0 : c1 + 1]
    region[layer_depth == depth] = int(color)


def _parse_scene(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    components = find_components(
        grid,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )

    rectangles: list[dict] = []
    markers: list[dict] = []
    for component in components:
        color = int(component["color"])
        area = int(component["area"])
        box = tuple(map(int, component["bbox"]))
        if area == 1:
            coord = tuple(map(int, component["coords"][0]))
            markers.append({"coord": coord, "color": color})
            continue

        assert area == bbox_h(box) * bbox_w(box), "foreground objects must be solid rectangles or single-pixel markers"
        rectangles.append({"bbox": box, "color": color})

    assert rectangles, "expected at least one rectangle"
    assert markers, "expected at least one marker"
    return rectangles, markers


def _rect_cell_index(rectangles: list[dict]) -> dict[Coord, int]:
    mapping: dict[Coord, int] = {}
    for rect_idx, rect in enumerate(rectangles):
        r0, c0, r1, c1 = rect["bbox"]
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                mapping[(r, c)] = rect_idx
    return mapping


def _sightline_hits(grid: np.ndarray, coord: Coord, rect_by_cell: dict[Coord, int]) -> list[tuple[int, int]]:
    hits: list[tuple[int, int]] = []
    r, c = coord
    for dr, dc in DIRS:
        rr = r + dr
        cc = c + dc
        gap = 0
        while in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) == BG:
            gap += 1
            rr += dr
            cc += dc
        if gap == 0 or not in_bounds(grid.shape, rr, cc):
            continue
        rect_idx = rect_by_cell.get((rr, cc))
        if rect_idx is not None:
            hits.append((rect_idx, gap))
    return hits


def _recolor_plan(grid: np.ndarray) -> tuple[list[dict], list[Coord], dict[tuple[int, int], int]]:
    rectangles, markers = _parse_scene(grid)
    rect_by_cell = _rect_cell_index(rectangles)

    marker_coords: list[Coord] = []
    plan: dict[tuple[int, int], int] = {}
    for marker in markers:
        coord = marker["coord"]
        hits = _sightline_hits(grid, coord, rect_by_cell)
        assert len(hits) == 1, "each marker must see exactly one rectangle through a positive background gap"
        rect_idx, depth = hits[0]
        assert 1 <= depth <= _max_layer_depth(rectangles[rect_idx]["bbox"]), "marker gap must encode an existing layer"

        marker_coords.append(coord)
        key = (rect_idx, depth)
        color = int(marker["color"])
        previous = plan.get(key)
        assert previous is None or previous == color, "two markers cannot assign different colors to the same layer"
        plan[key] = color

    return rectangles, marker_coords, plan


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the interior ring of each solid rectangle using the gap length of its matched marker.

    Input structure:
    - A black-background grid contains several isolated solid monochrome rectangles.
    - Additional isolated single-pixel markers sit on the same row or column as exactly one rectangle edge.
    - Between each marker and its matched rectangle there are one or more consecutive background cells.

    Rule:
    - For each marker, measure the number of background cells in its straight sightline to the first rectangle it faces.
    - Interpret that gap length g as an interior depth, where the outer rectangle border is depth 0 and the next ring is depth 1.
    - Recolor the entire depth-g ring of the matched rectangle to the marker's color, then erase all markers.

    Key insight:
    - The empty space outside a rectangle is a ruler for selecting which concentric rectangle layer should be repainted inside.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    rectangles, marker_coords, plan = _recolor_plan(g)
    out = g.copy()

    for r, c in marker_coords:
        out[r, c] = BG

    for rect_idx, depth in sorted(plan):
        _paint_layer(out, rectangles[rect_idx]["bbox"], depth, plan[(rect_idx, depth)])

    return out


# --------------- generator ---------------

GEN_GRID_MIN = 18
GEN_GRID_MAX = 30
GEN_RECT_COUNT_CHOICES = (2, 3, 3)
GEN_RECT_SIDE_MIN = 5
GEN_RECT_SIDE_MAX = 9
GEN_MAX_DEPTH = 3
GEN_MAX_MARKERS_PER_RECT = 2
GEN_EDGE_MARGIN = GEN_MAX_DEPTH + 1
GEN_RECT_PADDING = 3
GEN_MIN_COVERAGE = 0.12
GEN_MAX_COVERAGE = 0.42
GEN_ATTEMPTS = 500


def _rect_fits(rect: BBox, placed: list[dict], pad: int) -> bool:
    r0, c0, r1, c1 = rect
    for other in placed:
        rr0, cc0, rr1, cc1 = other["bbox"]
        if not (r1 + pad < rr0 or rr1 + pad < r0 or c1 + pad < cc0 or cc1 + pad < c0):
            return False
    return True


def _valid_rect_positions(height: int, width: int, rect_h: int, rect_w: int, placed: list[dict]) -> list[Coord]:
    positions: list[Coord] = []
    row_stop = height - GEN_EDGE_MARGIN - rect_h
    col_stop = width - GEN_EDGE_MARGIN - rect_w
    for r0 in range(GEN_EDGE_MARGIN, row_stop + 1):
        for c0 in range(GEN_EDGE_MARGIN, col_stop + 1):
            rect = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
            if _rect_fits(rect, placed, GEN_RECT_PADDING):
                positions.append((r0, c0))
    return positions


def _sample_rectangles(height: int, width: int) -> list[dict] | None:
    rect_count = random.choice(GEN_RECT_COUNT_CHOICES)
    rect_colors = random.sample([c for c in range(1, 10)], rect_count)
    dims = [
        (
            random.randint(GEN_RECT_SIDE_MIN, min(GEN_RECT_SIDE_MAX, height - 2 * GEN_EDGE_MARGIN)),
            random.randint(GEN_RECT_SIDE_MIN, min(GEN_RECT_SIDE_MAX, width - 2 * GEN_EDGE_MARGIN)),
            color,
        )
        for color in rect_colors
    ]
    dims.sort(key=lambda item: item[0] * item[1], reverse=True)

    rectangles: list[dict] = []
    for rect_h, rect_w, color in dims:
        positions = _valid_rect_positions(height, width, rect_h, rect_w, rectangles)
        if not positions:
            return None
        r0, c0 = random.choice(positions)
        rectangles.append(
            {
                "bbox": (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1),
                "color": int(color),
            }
        )
    return rectangles


def _paint_rectangles(grid: np.ndarray, rectangles: list[dict]) -> None:
    for rect in rectangles:
        r0, c0, r1, c1 = rect["bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(rect["color"])


def _sample_marker_counts(rectangles: list[dict]) -> list[int]:
    counts = []
    promotable = []
    for rect_idx, rect in enumerate(rectangles):
        depth_limit = min(_max_layer_depth(rect["bbox"]), GEN_MAX_DEPTH)
        max_count = min(GEN_MAX_MARKERS_PER_RECT, depth_limit)
        count = random.randint(1, max_count)
        counts.append(count)
        if max_count >= 2:
            promotable.append(rect_idx)
    if promotable and all(count == 1 for count in counts):
        counts[random.choice(promotable)] = 2
    return counts


def _marker_slot_candidates(box: BBox, side: str, gap: int) -> list[Coord]:
    r0, c0, r1, c1 = box
    if side == "top":
        return [(r0 - gap - 1, c) for c in range(c0, c1 + 1)]
    if side == "bottom":
        return [(r1 + gap + 1, c) for c in range(c0, c1 + 1)]
    if side == "left":
        return [(r, c0 - gap - 1) for r in range(r0, r1 + 1)]
    if side == "right":
        return [(r, c1 + gap + 1) for r in range(r0, r1 + 1)]
    raise ValueError(f"unknown side: {side}")


def _corridor_clear(grid: np.ndarray, box: BBox, side: str, gap: int, coord: Coord) -> bool:
    r, c = coord
    r0, c0, r1, c1 = box
    if side == "top":
        cells = [(rr, c) for rr in range(r + 1, r0)]
    elif side == "bottom":
        cells = [(rr, c) for rr in range(r1 + 1, r)]
    elif side == "left":
        cells = [(r, cc) for cc in range(c + 1, c0)]
    else:
        cells = [(r, cc) for cc in range(c1 + 1, c)]
    return len(cells) == gap and all(int(grid[rr, cc]) == BG for rr, cc in cells)


def _isolated_marker_slot(grid: np.ndarray, coord: Coord) -> bool:
    r, c = coord
    if not in_bounds(grid.shape, r, c) or int(grid[r, c]) != BG:
        return False
    for nr, nc in _neighbors8(r, c):
        if in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) != BG:
            return False
    return True


def _candidate_marker_slots(
    grid: np.ndarray,
    rectangles: list[dict],
    rect_by_cell: dict[Coord, int],
    rect_idx: int,
    side: str,
    depth: int,
) -> list[Coord]:
    box = rectangles[rect_idx]["bbox"]
    candidates = []
    for coord in _marker_slot_candidates(box, side, depth):
        if not _isolated_marker_slot(grid, coord):
            continue
        if not _corridor_clear(grid, box, side, depth, coord):
            continue
        hits = _sightline_hits(grid, coord, rect_by_cell)
        if hits == [(rect_idx, depth)]:
            candidates.append(coord)
    return candidates


def _sample_marker_color(rect_color: int, used_colors: set[int]) -> int:
    options = [c for c in range(1, 10) if c != rect_color and c not in used_colors]
    if not options:
        options = [c for c in range(1, 10) if c != rect_color]
    return int(random.choice(options))


def _place_markers(grid: np.ndarray, rectangles: list[dict]) -> list[dict] | None:
    rect_by_cell = _rect_cell_index(rectangles)
    marker_counts = _sample_marker_counts(rectangles)
    markers: list[dict] = []

    for rect_idx, rect in enumerate(rectangles):
        depth_limit = min(_max_layer_depth(rect["bbox"]), GEN_MAX_DEPTH)
        depths = random.sample(list(range(1, depth_limit + 1)), marker_counts[rect_idx])
        used_sides: set[str] = set()
        used_colors: set[int] = set()

        for depth in depths:
            preferred_sides = random.sample(list(SIDES), len(SIDES))
            ordered_sides = [side for side in preferred_sides if side not in used_sides]
            placed = False
            for side in ordered_sides:
                candidates = _candidate_marker_slots(grid, rectangles, rect_by_cell, rect_idx, side, depth)
                if not candidates:
                    continue
                coord = random.choice(candidates)
                color = _sample_marker_color(int(rect["color"]), used_colors)
                grid[coord] = color
                markers.append(
                    {
                        "coord": coord,
                        "color": color,
                        "rect_idx": rect_idx,
                        "depth": depth,
                    }
                )
                used_sides.add(side)
                used_colors.add(color)
                placed = True
                break
            if not placed:
                return None

    return markers


def _expected_output(inp: np.ndarray, rectangles: list[dict], markers: list[dict]) -> np.ndarray:
    out = inp.copy()
    for marker in markers:
        r, c = marker["coord"]
        out[r, c] = BG
    for marker in markers:
        _paint_layer(out, rectangles[marker["rect_idx"]]["bbox"], int(marker["depth"]), int(marker["color"]))
    return out


def _coverage_ok(rectangles: list[dict], shape: tuple[int, int]) -> bool:
    total_area = 0
    for rect in rectangles:
        r0, c0, r1, c1 = rect["bbox"]
        total_area += (r1 - r0 + 1) * (c1 - c0 + 1)
    coverage = total_area / (shape[0] * shape[1])
    return GEN_MIN_COVERAGE <= coverage <= GEN_MAX_COVERAGE


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(GEN_GRID_MIN, GEN_GRID_MAX)
    width = random.randint(GEN_GRID_MIN, GEN_GRID_MAX)

    rectangles = _sample_rectangles(height, width)
    if rectangles is None or not _coverage_ok(rectangles, (height, width)):
        return None

    inp = np.full((height, width), BG, dtype=int)
    _paint_rectangles(inp, rectangles)

    markers = _place_markers(inp, rectangles)
    if markers is None:
        return None

    expected = _expected_output(inp, rectangles, markers)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
