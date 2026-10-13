from __future__ import annotations

from importlib.machinery import SourceFileLoader
from pathlib import Path
import random

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")

try:
    _arc_utils = SourceFileLoader("_arc_utils_015_1", str(_ARC_UTILS_PATH)).load_module()
    Color = _arc_utils.Color
    bbox_coords = _arc_utils.bbox_coords
    in_bounds = _arc_utils.in_bounds
    components_mask = _arc_utils.components_mask
except Exception:
    class Color:
        BLACK = 0
        RED = 2
        GREEN = 3
        YELLOW = 4
        GREY = 5
        ORANGE = 7

    def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        return int(r0), int(c0), int(r1), int(c1)

    def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
        h, w = map(int, shape)
        return 0 <= int(r) < h and 0 <= int(c) < w

    def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
        mask = np.asarray(mask, dtype=bool)
        h, w = mask.shape
        seen = np.zeros((h, w), dtype=bool)
        if int(connectivity) == 4:
            steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        else:
            steps = [
                (dr, dc)
                for dr in (-1, 0, 1)
                for dc in (-1, 0, 1)
                if (dr, dc) != (0, 0)
            ]

        comps: list[np.ndarray] = []
        for r in range(h):
            for c in range(w):
                if not mask[r, c] or seen[r, c]:
                    continue
                stack = [(r, c)]
                seen[r, c] = True
                coords: list[tuple[int, int]] = []
                while stack:
                    rr, cc = stack.pop()
                    coords.append((rr, cc))
                    for dr, dc in steps:
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
                comps.append(np.array(coords, dtype=int))
        return comps


BBox = tuple[int, int, int, int]
Coord = tuple[int, int]

BG = Color.BLACK
MARKER = Color.RED
RECT = Color.GREEN
DIAGONAL_FILL = Color.YELLOW
EDGE_MARKER = Color.GREY
ORTHOGONAL_FILL = Color.ORANGE

DIAGONALS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ORTHOGONALS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GRID_MIN_SIZE = 16
GRID_MAX_SIZE = 23
RECT_COUNT_CHOICES = (2, 3)
PRIMARY_RECT_H = (6, 9)
PRIMARY_RECT_W = (6, 8)
OTHER_RECT_H = (5, 9)
OTHER_RECT_W = (5, 8)
RECT_PAD = 1
GEN_RETRIES = 400
MARKER_LAYOUT_RETRIES = 80
MARKER_GAP = 3


def _depth_in_bbox(r: int, c: int, bbox: BBox) -> int:
    r0, c0, r1, c1 = bbox
    return int(min(r - r0, r1 - r, c - c0, c1 - c))


def _analyze_rectangles(grid: np.ndarray) -> list[dict[str, object]]:
    rects: list[dict[str, object]] = []
    for coords in components_mask(grid != BG, connectivity=4):
        bbox = bbox_coords(coords)
        r0, c0, r1, c1 = bbox
        region = grid[r0 : r1 + 1, c0 : c1 + 1]
        colors = set(map(int, np.unique(region)))
        assert colors <= {RECT, MARKER}, "expected only green rectangles with red markers"
        assert int(coords.shape[0]) == region.size, "each non-black component must fill its bbox"

        markers = [
            (int(r0 + rr), int(c0 + cc))
            for rr, cc in np.argwhere(region == MARKER)
        ]
        rects.append({"bbox": bbox, "markers": sorted(markers)})

    rects.sort(key=lambda item: item["bbox"])
    return rects


def _marker_records(grid: np.ndarray) -> list[tuple[int, int, BBox]]:
    records: list[tuple[int, int, BBox]] = []
    for rect in _analyze_rectangles(grid):
        bbox = rect["bbox"]
        for r, c in rect["markers"]:
            records.append((int(r), int(c), bbox))
    records.sort(key=lambda item: (item[0], item[1]))
    return records


def _expansion_plan(r: int, c: int, depth: int) -> tuple[int | None, list[Coord]]:
    if depth == 0:
        return None, []
    if depth % 2 == 1:
        deltas = DIAGONALS
        color = DIAGONAL_FILL
    else:
        deltas = ORTHOGONALS
        color = ORTHOGONAL_FILL
    return color, [(int(r + dr), int(c + dc)) for dr, dc in deltas]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer each marker's rectangular depth and expand it with the matching neighborhood geometry.

    Input structure:
    - The input is a black background containing 2 or 3 separated filled green rectangles.
    - Each rectangle may contain several isolated red marker pixels embedded inside the green fill.
    - Treat every non-black connected component as one solid axis-aligned rectangle with optional red cells inside it.

    Rule:
    - For each red marker, compute its depth from the nearest side of its containing rectangle.
    - Depth-0 markers turn gray and do not expand.
    - Positive odd-depth markers place yellow pixels on their four diagonal neighbors.
    - Positive even-depth markers place orange pixels on their four orthogonal neighbors.
    - Process markers in row-major order; only paint onto in-bounds green cells, so earlier expansions block later conflicting ones.

    Key insight:
    - The red marker alone does not say how to expand; the hidden concentric-layer depth inside its rectangle is the switch that chooses diagonal versus orthogonal growth.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for r, c, bbox in _marker_records(g):
        depth = _depth_in_bbox(r, c, bbox)
        if depth == 0:
            out[r, c] = EDGE_MARKER
            continue

        color, targets = _expansion_plan(r, c, depth)
        for rr, cc in targets:
            if in_bounds(out.shape, rr, cc) and int(out[rr, cc]) == RECT:
                out[rr, cc] = int(color)

    return out


def _rect_fits(rect: BBox, placed: list[BBox], pad: int = RECT_PAD) -> bool:
    r0, c0, r1, c1 = rect
    for pr0, pc0, pr1, pc1 in placed:
        if not (r1 + pad < pr0 or pr1 + pad < r0 or c1 + pad < pc0 or pc1 + pad < c0):
            return False
    return True


def _sample_rect_dims(rect_count: int) -> list[tuple[int, int]]:
    dims: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    primary = (random.randint(*PRIMARY_RECT_H), random.randint(*PRIMARY_RECT_W))
    dims.append(primary)
    seen.add(primary)

    while len(dims) < rect_count:
        dim = (random.randint(*OTHER_RECT_H), random.randint(*OTHER_RECT_W))
        if dim in seen:
            continue
        seen.add(dim)
        dims.append(dim)
    return dims


def _place_rectangles(height: int, width: int, dims: list[tuple[int, int]]) -> list[BBox] | None:
    placed: list[BBox] = []
    for h, w in sorted(dims, key=lambda hw: hw[0] * hw[1], reverse=True):
        candidates = []
        for r0 in range(height - h + 1):
            for c0 in range(width - w + 1):
                rect = (r0, c0, r0 + h - 1, c0 + w - 1)
                if _rect_fits(rect, placed):
                    candidates.append(rect)
        if not candidates:
            return None
        placed.append(random.choice(candidates))
    return placed


def _layer_cells(bbox: BBox) -> dict[int, list[Coord]]:
    r0, c0, r1, c1 = bbox
    layers: dict[int, list[Coord]] = {}
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            depth = _depth_in_bbox(r, c, bbox)
            layers.setdefault(depth, []).append((r, c))
    return layers


def _manhattan(a: Coord, b: Coord) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _far_enough(cell: Coord, chosen: list[Coord]) -> bool:
    return all(_manhattan(cell, other) >= MARKER_GAP for other in chosen)


def _choose_required_depths(bbox: BBox, required_depths: list[int]) -> list[Coord] | None:
    layers = _layer_cells(bbox)
    order = sorted(range(len(required_depths)), key=lambda i: len(layers.get(required_depths[i], [])))
    chosen_by_index: dict[int, Coord] = {}
    chosen_cells: list[Coord] = []

    def dfs(pos: int) -> bool:
        if pos == len(order):
            return True
        idx = order[pos]
        depth = required_depths[idx]
        candidates = list(layers.get(depth, []))
        random.shuffle(candidates)
        for cell in candidates:
            if cell in chosen_cells or not _far_enough(cell, chosen_cells):
                continue
            chosen_by_index[idx] = cell
            chosen_cells.append(cell)
            if dfs(pos + 1):
                return True
            chosen_cells.pop()
            del chosen_by_index[idx]
        return False

    if not dfs(0):
        return None
    return [chosen_by_index[i] for i in range(len(required_depths))]


def _add_extra_markers(bbox: BBox, chosen: list[Coord], target_count: int) -> bool:
    if len(chosen) >= target_count:
        return True

    layers = _layer_cells(bbox)
    used_depths = {_depth_in_bbox(r, c, bbox) for r, c in chosen}
    all_cells = [cell for depth in sorted(layers) for cell in layers[depth]]
    new_depth_first = [cell for cell in all_cells if _depth_in_bbox(cell[0], cell[1], bbox) not in used_depths]
    repeated_depths = [cell for cell in all_cells if _depth_in_bbox(cell[0], cell[1], bbox) in used_depths]
    pool = new_depth_first + repeated_depths

    for _ in range(MARKER_LAYOUT_RETRIES):
        work = list(chosen)
        candidates = list(pool)
        random.shuffle(candidates)
        for cell in candidates:
            if cell in work or not _far_enough(cell, work):
                continue
            work.append(cell)
            if len(work) == target_count:
                chosen[:] = work
                return True
    return False


def _marker_count_order(options: tuple[int, ...]) -> list[int]:
    first = random.choice(options)
    rest = [count for count in options if count != first]
    random.shuffle(rest)
    return [first] + rest


def _place_markers_in_rect(bbox: BBox, *, require_three_branches: bool) -> list[Coord] | None:
    if require_three_branches:
        required = [0, 1, 2]
        count_options = _marker_count_order((3, 4))
    else:
        required = [0, 1]
        count_options = _marker_count_order((2, 3, 4))

    for count in count_options:
        chosen = _choose_required_depths(bbox, required)
        if chosen is None:
            continue
        if _add_extra_markers(bbox, chosen, count):
            return sorted(chosen)
    return None


def _validate_generated_input(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        return False

    rects = _analyze_rectangles(g)
    if len(rects) not in RECT_COUNT_CHOICES:
        return False

    sizes = set()
    saw_depth0 = False
    saw_odd = False
    saw_pos_even = False

    for rect in rects:
        bbox = rect["bbox"]
        r0, c0, r1, c1 = bbox
        size = (r1 - r0 + 1, c1 - c0 + 1)
        if size in sizes:
            return False
        sizes.add(size)

        markers = rect["markers"]
        if not (2 <= len(markers) <= 4):
            return False

        depths = [_depth_in_bbox(r, c, bbox) for r, c in markers]
        if not any(depth % 2 == 0 for depth in depths):
            return False
        if not any(depth % 2 == 1 for depth in depths):
            return False

        for i, cell in enumerate(markers):
            for other in markers[i + 1 :]:
                if _manhattan(cell, other) < MARKER_GAP:
                    return False

        saw_depth0 |= any(depth == 0 for depth in depths)
        saw_odd |= any(depth > 0 and depth % 2 == 1 for depth in depths)
        saw_pos_even |= any(depth > 0 and depth % 2 == 0 for depth in depths)

    return saw_depth0 and saw_odd and saw_pos_even


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(GRID_MIN_SIZE, GRID_MAX_SIZE)
    width = random.randint(GRID_MIN_SIZE, GRID_MAX_SIZE)
    rect_count = random.choice(RECT_COUNT_CHOICES)

    dims = _sample_rect_dims(rect_count)
    rects = _place_rectangles(height, width, dims)
    if rects is None:
        return None

    rects = sorted(rects, key=lambda bbox: (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1), reverse=True)
    marker_sets: list[list[Coord]] = []
    for idx, bbox in enumerate(rects):
        markers = _place_markers_in_rect(bbox, require_three_branches=(idx == 0))
        if markers is None:
            return None
        marker_sets.append(markers)

    grid = np.full((height, width), BG, dtype=int)
    for bbox in rects:
        r0, c0, r1, c1 = bbox
        grid[r0 : r1 + 1, c0 : c1 + 1] = RECT
    for markers in marker_sets:
        for r, c in markers:
            grid[r, c] = MARKER

    if not _validate_generated_input(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is None:
            continue
        try:
            out = main(grid)
        except Exception:
            continue
        if out.shape == grid.shape and _validate_generated_input(grid):
            return grid
    raise RuntimeError("failed to generate a valid input")
