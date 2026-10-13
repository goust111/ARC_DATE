from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]
SIDES = ("top", "bottom", "left", "right")
OUTWARD_DIRS = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
PALETTE = tuple(range(10))


@dataclass(frozen=True)
class MarkerSpec:
    coord: Coord
    color: int


@dataclass
class CageSpec:
    bbox: BBox
    markers: list[MarkerSpec]
    gap: Coord | None
    side: str | None


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    height, width = map(int, shape)
    return 0 <= int(r) < height and 0 <= int(c) < width


def neighbors4(r: int, c: int):
    yield int(r) - 1, int(c)
    yield int(r) + 1, int(c)
    yield int(r), int(c) - 1
    yield int(r), int(c) + 1


def bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _color_components(grid: np.ndarray, bg: int) -> list[dict]:
    height, width = grid.shape
    seen = np.zeros((height, width), dtype=bool)
    components: list[dict] = []
    for r in range(height):
        for c in range(width):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if not in_bounds(grid.shape, nr, nc) or seen[nr, nc]:
                        continue
                    if int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            comp_coords = np.array(coords, dtype=int)
            components.append(
                {
                    "color": color,
                    "coords": comp_coords,
                    "bbox": bbox_coords(comp_coords),
                    "area": int(len(coords)),
                }
            )
    return components


def _border_set(bbox: BBox) -> set[Coord]:
    r0, c0, r1, c1 = bbox
    border: set[Coord] = set()
    for c in range(c0, c1 + 1):
        border.add((r0, c))
        border.add((r1, c))
    for r in range(r0, r1 + 1):
        border.add((r, c0))
        border.add((r, c1))
    return border


def _side_from_gap(bbox: BBox, gap: Coord) -> str | None:
    r0, c0, r1, c1 = bbox
    r, c = gap
    if r == r0 and c0 < c < c1:
        return "top"
    if r == r1 and c0 < c < c1:
        return "bottom"
    if c == c0 and r0 < r < r1:
        return "left"
    if c == c1 and r0 < r < r1:
        return "right"
    return None


def _classify_cage_component(grid: np.ndarray, component: dict, bg: int) -> dict | None:
    color = int(component["color"])
    coords = component["coords"]
    bbox = component["bbox"]
    r0, c0, r1, c1 = bbox
    if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
        return None

    present = {tuple(map(int, coord)) for coord in coords}
    border = _border_set(bbox)
    if any(cell not in border for cell in present):
        return None

    region = grid[r0 : r1 + 1, c0 : c1 + 1]
    same_color_mask = region == color
    border_mask = np.zeros(region.shape, dtype=bool)
    border_mask[0, :] = True
    border_mask[-1, :] = True
    border_mask[:, 0] = True
    border_mask[:, -1] = True
    if np.any(same_color_mask & ~border_mask):
        return None

    missing = sorted(border - present)
    if len(missing) > 1:
        return None

    gap = None
    side = None
    if missing:
        gap = missing[0]
        if int(grid[gap]) != bg:
            return None
        side = _side_from_gap(bbox, gap)
        if side is None:
            return None

    if len(present) != len(border) - len(missing):
        return None

    return {"color": color, "bbox": bbox, "gap": gap, "side": side}


def _extract_cages(grid: np.ndarray) -> tuple[int, int, list[CageSpec]]:
    bg = get_bg(grid)
    components = _color_components(grid, bg)
    by_color: dict[int, list[dict]] = {}
    for comp in components:
        by_color.setdefault(int(comp["color"]), []).append(comp)

    candidate_colors: list[int] = []
    classified_by_color: dict[int, list[dict]] = {}
    for color, comps in by_color.items():
        classified = []
        for comp in comps:
            cage = _classify_cage_component(grid, comp, bg)
            if cage is None:
                classified = []
                break
            classified.append(cage)
        if classified:
            candidate_colors.append(color)
            classified_by_color[color] = classified

    assert len(candidate_colors) == 1, "expected a unique cage-frame color"
    frame_color = int(candidate_colors[0])

    cages: list[CageSpec] = []
    for cage_info in sorted(classified_by_color[frame_color], key=lambda item: item["bbox"]):
        r0, c0, r1, c1 = cage_info["bbox"]
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        marker_coords = np.argwhere((interior != bg) & (interior != frame_color))
        assert marker_coords.size > 0, "every cage must contain markers"

        markers = []
        marker_colors = []
        for rr, cc in marker_coords:
            gr = int(r0 + 1 + rr)
            gc = int(c0 + 1 + cc)
            color = int(grid[gr, gc])
            marker_colors.append(color)
            markers.append(MarkerSpec((gr, gc), color))
        assert len(marker_colors) == len(set(marker_colors)), "markers inside a cage must have distinct colors"
        cages.append(CageSpec(cage_info["bbox"], markers, cage_info["gap"], cage_info["side"]))

    return bg, frame_color, cages


def _gap_distance_map(bbox: BBox, gap: Coord) -> dict[Coord, int]:
    r0, c0, r1, c1 = bbox
    walkable = {(gap[0], gap[1])}
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            walkable.add((r, c))

    dist = {gap: 0}
    queue = deque([gap])
    while queue:
        r, c = queue.popleft()
        for nr, nc in neighbors4(r, c):
            if (nr, nc) not in walkable or (nr, nc) in dist:
                continue
            dist[(nr, nc)] = dist[(r, c)] + 1
            queue.append((nr, nc))
    return dist


def _ray_cells(gap: Coord, side: str, length: int) -> list[Coord]:
    dr, dc = OUTWARD_DIRS[side]
    return [(int(gap[0] + dr * step), int(gap[1] + dc * step)) for step in range(1, length + 1)]


def _apply_leak_rule(grid: np.ndarray, bg: int, cages: list[CageSpec]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for cage in cages:
        if cage.gap is None:
            continue

        assert cage.side is not None
        distances = _gap_distance_map(cage.bbox, cage.gap)
        ranking = []
        for marker in cage.markers:
            assert marker.coord in distances, "marker must be reachable from the gap within the cage"
            ranking.append((int(distances[marker.coord]), marker))
        distance_values = [dist for dist, _marker in ranking]
        assert len(distance_values) == len(set(distance_values)), "gap distances must be unique within a cage"
        ranking.sort(key=lambda item: item[0])

        for marker in cage.markers:
            out[marker.coord] = bg

        ray_cells = _ray_cells(cage.gap, cage.side, len(ranking))
        for (dist, marker), cell in zip(ranking, ray_cells):
            rr, cc = cell
            assert in_bounds(out.shape, rr, cc), "emitted ray must stay in bounds"
            assert int(out[rr, cc]) == bg, f"ray target {cell} must be background"
            out[rr, cc] = int(marker.color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Leak the markers inside each broken cage into an outward ray ordered by distance to the gap.

    Input structure:
    - A background grid contains several hollow rectangular cages drawn in one shared frame color.
    - Some cages are fully closed, while others have exactly one background gap on one side.
    - Each cage contains distinct single-pixel colored markers inside the frame, and cages are well separated.

    Rule:
    - Closed cages stay unchanged.
    - For each gapped cage, erase all interior markers to background.
    - Starting from the cell just outside the gap, emit an outward ray whose length equals the marker count.
    - The j-th emitted color is the marker with the j-th smallest 4-neighbor in-cage distance to the gap.

    Key insight:
    - The gap is both the trigger and the ordering anchor: only broken cages leak, and the outside color sequence is a rank encoding of interior distances to that exact gap.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg, _frame_color, cages = _extract_cages(g)
    return _apply_leak_rule(g, bg, cages)


# --------------- generator ---------------

GRID_MIN = 22
GRID_MAX = 30
CAGE_COUNT_CHOICES = (2, 3)
BOX_SIDE_MIN = 6
BOX_SIDE_MAX = 9
OPEN_MARKERS = (3, 4)
CLOSED_MARKERS = (2, 4)
SCENE_ATTEMPTS = 350
CAGE_ATTEMPTS = 160
MARKER_ATTEMPTS = 240


def _expand_rect(bbox: BBox, shape: tuple[int, int], margin: int = 1) -> set[Coord]:
    height, width = shape
    r0, c0, r1, c1 = bbox
    cells: set[Coord] = set()
    for r in range(max(0, r0 - margin), min(height, r1 + margin + 1)):
        for c in range(max(0, c0 - margin), min(width, c1 + margin + 1)):
            cells.add((r, c))
    return cells


def _reserve_ray_cells(gap: Coord, side: str, length: int, shape: tuple[int, int]) -> set[Coord]:
    if length <= 0:
        return set()
    ray = _ray_cells(gap, side, length)
    if side in ("top", "bottom"):
        rows = [cell[0] for cell in ray]
        cols = [gap[1]]
    else:
        rows = [gap[0]]
        cols = [cell[1] for cell in ray]
    r0 = min(rows) - 1
    r1 = max(rows) + 1
    c0 = min(cols) - 1
    c1 = max(cols) + 1
    return _expand_rect((r0, c0, r1, c1), shape, margin=0)


def _pairwise_separated(cells: list[Coord]) -> bool:
    for i, (r0, c0) in enumerate(cells):
        for r1, c1 in cells[i + 1 :]:
            if abs(r0 - r1) + abs(c0 - c1) <= 1:
                return False
    return True


def _marker_metrics(cell: Coord, gap: Coord, side: str) -> tuple[int, int, int]:
    r, c = cell
    gr, gc = gap
    if side == "top":
        axial = r - gr
        lateral = abs(c - gc)
    elif side == "bottom":
        axial = gr - r
        lateral = abs(c - gc)
    elif side == "left":
        axial = c - gc
        lateral = abs(r - gr)
    else:
        axial = gc - c
        lateral = abs(r - gr)
    return int(axial), int(lateral), int(axial + lateral)


def _order_signature(cells: list[Coord], keys: list[tuple[int, ...]]) -> tuple[int, ...]:
    return tuple(sorted(range(len(cells)), key=lambda idx: keys[idx] + cells[idx]))


def _open_marker_layout_ok(cells: list[Coord], gap: Coord, side: str) -> bool:
    if not _pairwise_separated(cells):
        return False

    metrics = [_marker_metrics(cell, gap, side) for cell in cells]
    axials = [m[0] for m in metrics]
    laterals = [m[1] for m in metrics]
    dists = [m[2] for m in metrics]
    if len(set(dists)) != len(dists):
        return False
    if all(lat == 0 for lat in laterals):
        return False
    if len(set(axials)) < 2 or len(set(laterals)) < 2:
        return False

    dist_order = _order_signature(cells, [(d,) for d in dists])
    axial_order = _order_signature(cells, [(a, l) for a, l in zip(axials, laterals)])
    lateral_order = _order_signature(cells, [(l, a) for a, l in zip(axials, laterals)])
    row_order = _order_signature(cells, [(r, c) for r, c in cells])
    col_order = _order_signature(cells, [(c, r) for r, c in cells])
    if dist_order in {axial_order, lateral_order, row_order, col_order}:
        return False
    return True


def _choose_open_markers(bbox: BBox, gap: Coord, side: str, count: int) -> list[Coord] | None:
    r0, c0, r1, c1 = bbox
    interior = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    if len(interior) < count:
        return None
    for _ in range(MARKER_ATTEMPTS):
        cells = random.sample(interior, count)
        if _open_marker_layout_ok(cells, gap, side):
            return cells
    return None


def _choose_closed_markers(bbox: BBox, count: int) -> list[Coord] | None:
    r0, c0, r1, c1 = bbox
    interior = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    if len(interior) < count:
        return None
    for _ in range(MARKER_ATTEMPTS):
        cells = random.sample(interior, count)
        if _pairwise_separated(cells):
            return cells
    return None


def _gap_candidates(bbox: BBox, side: str) -> list[Coord]:
    r0, c0, r1, c1 = bbox
    if side == "top":
        return [(r0, c) for c in range(c0 + 1, c1)]
    if side == "bottom":
        return [(r1, c) for c in range(c0 + 1, c1)]
    if side == "left":
        return [(r, c0) for r in range(r0 + 1, r1)]
    return [(r, c1) for r in range(r0 + 1, r1)]


def _bbox_ranges(shape: tuple[int, int], h: int, w: int, side: str | None, ray_len: int) -> tuple[range, range] | None:
    height, width = shape
    min_r0 = 1
    max_r0 = height - h - 2
    min_c0 = 1
    max_c0 = width - w - 2
    if side == "top":
        min_r0 = max(min_r0, ray_len + 1)
    elif side == "bottom":
        max_r0 = min(max_r0, height - h - ray_len - 2)
    elif side == "left":
        min_c0 = max(min_c0, ray_len + 1)
    elif side == "right":
        max_c0 = min(max_c0, width - w - ray_len - 2)
    if min_r0 > max_r0 or min_c0 > max_c0:
        return None
    return range(min_r0, max_r0 + 1), range(min_c0, max_c0 + 1)


def _place_cage(
    shape: tuple[int, int],
    reserved: set[Coord],
    is_open: bool,
    marker_count: int,
    available_colors: list[int],
) -> CageSpec | None:
    height, width = shape
    side = random.choice(SIDES) if is_open else None
    ray_len = marker_count if is_open else 0

    for _ in range(CAGE_ATTEMPTS):
        h = random.randint(BOX_SIDE_MIN, BOX_SIDE_MAX)
        w = random.randint(BOX_SIDE_MIN, BOX_SIDE_MAX)
        ranges = _bbox_ranges(shape, h, w, side, ray_len)
        if ranges is None:
            continue
        row_range, col_range = ranges
        r0 = random.choice(tuple(row_range))
        c0 = random.choice(tuple(col_range))
        bbox = (r0, c0, r0 + h - 1, c0 + w - 1)

        bbox_reserved = _expand_rect(bbox, (height, width), margin=1)
        if bbox_reserved & reserved:
            continue

        if not is_open:
            marker_cells = _choose_closed_markers(bbox, marker_count)
            if marker_cells is None:
                continue
            colors = random.sample(available_colors, marker_count)
            return CageSpec(
                bbox=bbox,
                markers=[MarkerSpec(cell, color) for cell, color in zip(marker_cells, colors)],
                gap=None,
                side=None,
            )

        gaps = _gap_candidates(bbox, side)
        random.shuffle(gaps)
        for gap in gaps:
            ray_reserved = _reserve_ray_cells(gap, side, ray_len, (height, width))
            if ray_reserved & reserved:
                continue
            marker_cells = _choose_open_markers(bbox, gap, side, marker_count)
            if marker_cells is None:
                continue
            colors = random.sample(available_colors, marker_count)
            return CageSpec(
                bbox=bbox,
                markers=[MarkerSpec(cell, color) for cell, color in zip(marker_cells, colors)],
                gap=gap,
                side=side,
            )
    return None


def _paint_cage(grid: np.ndarray, frame_color: int, cage: CageSpec) -> None:
    r0, c0, r1, c1 = cage.bbox
    for c in range(c0, c1 + 1):
        grid[r0, c] = frame_color
        grid[r1, c] = frame_color
    for r in range(r0, r1 + 1):
        grid[r, c0] = frame_color
        grid[r, c1] = frame_color
    if cage.gap is not None:
        grid[cage.gap] = get_bg(grid)
    for marker in cage.markers:
        grid[marker.coord] = int(marker.color)


def _spec_signature(spec: CageSpec) -> tuple:
    return (
        spec.bbox,
        spec.gap,
        spec.side,
        tuple(sorted((marker.coord, int(marker.color)) for marker in spec.markers)),
    )


def _validate_scene(grid: np.ndarray, bg: int, frame_color: int, specs: list[CageSpec]) -> bool:
    if not any(spec.gap is None for spec in specs):
        return False
    if not any(spec.gap is not None for spec in specs):
        return False

    expected = _apply_leak_rule(grid, bg, specs)
    if np.array_equal(expected, grid):
        return False

    changed = int(np.count_nonzero(expected != grid))
    open_marker_total = sum(len(spec.markers) for spec in specs if spec.gap is not None)
    if changed != 2 * open_marker_total:
        return False

    for spec in specs:
        if spec.gap is None:
            for marker in spec.markers:
                if int(expected[marker.coord]) != int(marker.color):
                    return False
        else:
            assert spec.side is not None
            distances = _gap_distance_map(spec.bbox, spec.gap)
            ordered = sorted(spec.markers, key=lambda marker: distances[marker.coord])
            ray = _ray_cells(spec.gap, spec.side, len(spec.markers))
            for marker in spec.markers:
                if int(expected[marker.coord]) != bg:
                    return False
            for marker, cell in zip(ordered, ray):
                if int(expected[cell]) != int(marker.color):
                    return False

    try:
        parsed_bg, parsed_frame_color, parsed_specs = _extract_cages(grid)
    except AssertionError:
        return False
    if parsed_bg != bg or parsed_frame_color != frame_color:
        return False
    if len(parsed_specs) != len(specs):
        return False

    want = sorted(_spec_signature(spec) for spec in specs)
    got = sorted(_spec_signature(spec) for spec in parsed_specs)
    if got != want:
        return False

    try:
        actual = main(grid)
    except AssertionError:
        return False
    return np.array_equal(actual, expected)


def _build_scene() -> np.ndarray | None:
    height = random.randint(GRID_MIN, GRID_MAX)
    width = random.randint(GRID_MIN, GRID_MAX)
    bg, frame_color = random.sample(PALETTE, 2)
    grid = np.full((height, width), bg, dtype=int)
    available_colors = [color for color in PALETTE if color not in {bg, frame_color}]

    cage_count = random.choice(CAGE_COUNT_CHOICES)
    if cage_count == 2:
        open_count = 1
    else:
        open_count = random.choice((1, 2))
    statuses = [True] * open_count + [False] * (cage_count - open_count)
    random.shuffle(statuses)

    requests = []
    for is_open in statuses:
        if is_open:
            marker_count = random.randint(*OPEN_MARKERS)
        else:
            marker_count = random.randint(*CLOSED_MARKERS)
        requests.append((is_open, marker_count))
    requests.sort(key=lambda item: (item[0], item[1]), reverse=True)

    specs: list[CageSpec] = []
    reserved: set[Coord] = set()
    for is_open, marker_count in requests:
        spec = _place_cage(grid.shape, reserved, is_open, marker_count, available_colors)
        if spec is None:
            return None
        specs.append(spec)
        reserve = _expand_rect(spec.bbox, grid.shape, margin=1)
        if spec.gap is not None:
            assert spec.side is not None
            reserve |= _reserve_ray_cells(spec.gap, spec.side, len(spec.markers), grid.shape)
        reserved |= reserve

    for spec in specs:
        _paint_cage(grid, frame_color, spec)

    if not _validate_scene(grid, bg, frame_color, specs):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(SCENE_ATTEMPTS):
        grid = _build_scene()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid leak-cage scene")
