from __future__ import annotations

import hashlib
import random
from collections import deque

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
GREY = 5

Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

_GEN_COUNTER = 0
_SEEN_OUTPUT_SIGNATURES: set[str] = set()

_GRID_MIN = 15
_GRID_MAX = 22
_BOX_MIN = 6
_BOX_MAX = 9
_AREA_MIN = 3
_AREA_MAX = 8
_MAX_GEN_ATTEMPTS = 400


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = np.asarray(grid == int(color), dtype=bool)
    seen = np.zeros(mask.shape, dtype=bool)
    out: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        q: deque[Coord] = deque([(rr, cc)])
        seen[rr, cc] = True
        coords: list[Coord] = []
        while q:
            cr, cc2 = q.popleft()
            coords.append((cr, cc2))
            for nr, nc in neighbors4(cr, cc2):
                if not in_bounds(mask.shape, nr, nc):
                    continue
                if seen[nr, nc] or not mask[nr, nc]:
                    continue
                seen[nr, nc] = True
                q.append((int(nr), int(nc)))
        out.append(np.array(coords, dtype=int))
    return out


def _line_indices(center: int, length: int) -> list[int]:
    start = int(center) - int(length) // 2
    return list(range(start, start + int(length)))


def _segment_cells(center: Coord, axis: str, length: int) -> list[Coord]:
    r, c = map(int, center)
    if axis == "horizontal":
        return [(r, cc) for cc in _line_indices(c, length)]
    return [(rr, c) for rr in _line_indices(r, length)]


def _parallel_bar_cells(center: Coord, axis: str, length: int) -> list[Coord]:
    r, c = map(int, center)
    if axis == "horizontal":
        cols = _line_indices(c, length)
        return [(r - 1, cc) for cc in cols] + [(r + 1, cc) for cc in cols]
    rows = _line_indices(r, length)
    return [(rr, c - 1) for rr in rows] + [(rr, c + 1) for rr in rows]


def _missing_corner(grey_coords: np.ndarray) -> tuple[Coord, BBox]:
    r0, c0, r1, c1 = bbox_coords(grey_coords)
    assert r1 > r0 and c1 > c0
    assert (r1 - r0) == (c1 - c0), "grey markers must define a square bbox"
    corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    grey_set = {tuple(map(int, rc)) for rc in grey_coords}
    assert len(grey_set) == 3 and grey_set <= corners
    miss = corners - grey_set
    assert len(miss) == 1
    return miss.pop(), (r0, c0, r1, c1)


def _parse_scene(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(map(int, np.unique(g))) <= {BLACK, RED, GREEN, GREY}

    grey_coords = np.argwhere(g == GREY)
    assert grey_coords.shape == (3, 2), "expected exactly three grey corner markers"
    missing, box = _missing_corner(grey_coords)
    r0, c0, r1, c1 = box
    box_size = int(r1 - r0 + 1)

    red_components = _components_of_color(g, RED)
    assert len(red_components) == 2, "expected exactly two red guide bars"
    red_bboxes = [bbox_coords(comp) for comp in red_components]

    horizontal = all(rb[0] == rb[2] and rb[1] == c0 and rb[3] == c1 for rb in red_bboxes)
    vertical = all(rb[1] == rb[3] and rb[0] == r0 and rb[2] == r1 for rb in red_bboxes)
    assert horizontal ^ vertical, "guide bars must be parallel full-length lines"

    if horizontal:
        axis = "horizontal"
        bar_positions = sorted(int(rb[0]) for rb in red_bboxes)
        assert r0 < bar_positions[0] < bar_positions[1] < r1
    else:
        axis = "vertical"
        bar_positions = sorted(int(rb[1]) for rb in red_bboxes)
        assert c0 < bar_positions[0] < bar_positions[1] < c1

    green_components = _components_of_color(g, GREEN)
    assert len(green_components) == 1, "expected exactly one green object"
    green_coords = green_components[0]
    area = int(green_coords.shape[0])
    assert _AREA_MIN <= area <= _AREA_MAX
    gr0, gc0, gr1, gc1 = bbox_coords(green_coords)
    assert gr1 > gr0 and gc1 > gc0, "green object must be irregular, not a straight line"
    assert area < (gr1 - gr0 + 1) * (gc1 - gc0 + 1), "green object must not be a full rectangle"

    if axis == "horizontal":
        top_bar, bottom_bar = bar_positions
        assert bottom_bar - top_bar >= 3
        assert np.all((green_coords[:, 0] > top_bar) & (green_coords[:, 0] < bottom_bar))
        assert np.all((green_coords[:, 1] > c0) & (green_coords[:, 1] < c1))
    else:
        left_bar, right_bar = bar_positions
        assert right_bar - left_bar >= 3
        assert np.all((green_coords[:, 1] > left_bar) & (green_coords[:, 1] < right_bar))
        assert np.all((green_coords[:, 0] > r0) & (green_coords[:, 0] < r1))

    return {
        "box": box,
        "box_size": box_size,
        "missing": missing,
        "axis": axis,
        "length": area,
        "grey_coords": grey_coords,
        "green_coords": green_coords,
    }


def _paint_output(grid: np.ndarray, info: dict[str, object]) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    out[(out == RED) | (out == GREEN)] = BLACK

    center = info["missing"]
    axis = str(info["axis"])
    length = int(info["length"])
    segment = _segment_cells(center, axis, length)
    guides = _parallel_bar_cells(center, axis, length)

    for r, c in segment + guides:
        assert in_bounds(out.shape, r, c), "compressed output must fit on the canvas"

    grey_set = {tuple(map(int, rc)) for rc in np.asarray(info["grey_coords"], dtype=int)}
    for cell in segment + guides:
        assert cell not in grey_set, "compressed output must not overlap existing grey markers"

    for r, c in guides:
        out[r, c] = RED
    for r, c in segment:
        out[r, c] = GREEN
    for r, c in grey_set:
        out[r, c] = GREY
    return out


def _grid_sig(grid: np.ndarray) -> str:
    g = np.asarray(grid, dtype=int)
    return hashlib.md5(g.tobytes() + str(g.shape).encode()).hexdigest()


def _semantic_validate(grid: np.ndarray) -> None:
    info = _parse_scene(grid)
    out = _paint_output(grid, info)
    assert out.shape == grid.shape
    assert not np.array_equal(grid, out)

    true_sig = _grid_sig(out)
    center = tuple(map(int, info["missing"]))
    length = int(info["length"])
    axis = str(info["axis"])
    grey_coords = np.asarray(info["grey_coords"], dtype=int)
    box = info["box"]
    r0, c0, r1, c1 = map(int, box)
    corners = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]

    alt_axes = ["vertical"] if axis == "horizontal" else ["horizontal"]
    for alt_axis in alt_axes:
        alt_info = dict(info)
        alt_info["axis"] = alt_axis
        try:
            alt_out = _paint_output(grid, alt_info)
        except AssertionError:
            continue
        assert _grid_sig(alt_out) != true_sig, "axis must affect the result"

    for corner in corners:
        if corner == center:
            continue
        alt_info = dict(info)
        alt_info["missing"] = corner
        try:
            alt_out = _paint_output(grid, alt_info)
        except AssertionError:
            continue
        assert _grid_sig(alt_out) != true_sig, "missing-corner location must affect the result"

    for alt_length in (length - 1, length + 1):
        if not (_AREA_MIN <= alt_length <= _AREA_MAX and alt_length <= int(info["box_size"])):
            continue
        alt_info = dict(info)
        alt_info["length"] = alt_length
        try:
            alt_out = _paint_output(grid, alt_info)
        except AssertionError:
            continue
        assert _grid_sig(alt_out) != true_sig, "green area must affect the result"

    seg = _segment_cells(center, axis, length)
    guides = _parallel_bar_cells(center, axis, length)
    assert len(seg) == length
    assert len(guides) == 2 * length
    grey_set = {tuple(map(int, rc)) for rc in grey_coords}
    assert not grey_set.intersection(seg)
    assert not grey_set.intersection(guides)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress the inner green object into a guide-aligned line centered on the square's missing corner.

    Input structure:
    - The background is black, with exactly three grey pixels marking three corners of a square bbox and one corner missing.
    - Inside that square bbox are exactly two parallel full-length red guide bars, either horizontal or vertical.
    - A single irregular 4-connected green object lies between the red bars.

    Rule:
    - Infer the square bbox from the three grey corner markers and locate the missing corner.
    - Use the red bars to determine the compression axis, and count the green pixels to get the compressed line length.
    - Erase the original red bars and green object, then draw a centered green line through the missing-corner coordinate along that axis.
    - Draw two parallel red guide bars of the same length one cell to either side of the green line, and keep the three grey markers unchanged.

    Key insight:
    - Direction, length, and placement come from three different cues: the red guides give the axis, the green area gives the length, and the grey missing corner gives the new center.
    """
    info = _parse_scene(grid)
    return _paint_output(np.asarray(grid, dtype=int), info)


def _pick_bar_positions(box_size: int, axis: str, length: int) -> tuple[int, int]:
    options: list[tuple[int, int]] = []
    span = list(range(1, box_size - 1))
    for a in span:
        for b in span:
            if b <= a:
                continue
            gap = b - a - 1
            if gap < 2:
                continue
            strip_area = gap * (box_size - 2)
            if strip_area < length:
                continue
            options.append((a, b))
    assert options, "box must be large enough to hold both bars and the object"
    return random.choice(options)


def _random_connected_shape(cells: list[Coord], area: int) -> set[Coord] | None:
    cell_set = set(cells)
    for _ in range(120):
        start = random.choice(cells)
        shape = {start}
        frontier = {start}
        while len(shape) < area and frontier:
            boundary = []
            for r, c in frontier:
                for nr, nc in neighbors4(r, c):
                    nxt = (int(nr), int(nc))
                    if nxt in cell_set and nxt not in shape:
                        boundary.append(nxt)
            if not boundary:
                break
            nxt = random.choice(boundary)
            shape.add(nxt)
            frontier.add(nxt)
        if len(shape) != area:
            continue
        coords = np.array(sorted(shape), dtype=int)
        r0, c0, r1, c1 = bbox_coords(coords)
        if r1 == r0 or c1 == c0:
            continue
        if area == (r1 - r0 + 1) * (c1 - c0 + 1):
            continue
        return shape
    return None


def _make_case(call_index: int) -> np.ndarray | None:
    orientation = "horizontal" if call_index % 2 == 0 else "vertical"
    missing_idx = (call_index // 2) % 4
    want_even = ((call_index // 8) % 2) == 1

    grid_h = random.randint(_GRID_MIN, _GRID_MAX)
    grid_w = random.randint(_GRID_MIN, _GRID_MAX)

    box_size = random.randint(_BOX_MIN, min(_BOX_MAX, grid_h - 6, grid_w - 6))
    length_choices = [n for n in range(_AREA_MIN, min(_AREA_MAX, box_size) + 1) if (n % 2 == 0) == want_even]
    if not length_choices:
        length_choices = list(range(_AREA_MIN, min(_AREA_MAX, box_size) + 1))
    length = random.choice(length_choices)

    outward = length // 2 + 2
    inward = (length - 1) // 2 + 2
    pad = max(outward, inward, 2)
    if grid_h < box_size + 2 * pad or grid_w < box_size + 2 * pad:
        return None

    r0 = random.randint(pad, grid_h - box_size - pad)
    c0 = random.randint(pad, grid_w - box_size - pad)
    r1 = r0 + box_size - 1
    c1 = c0 + box_size - 1

    corners = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]
    missing = corners[missing_idx]
    grey = [corner for i, corner in enumerate(corners) if i != missing_idx]

    rel_a, rel_b = _pick_bar_positions(box_size, orientation, length)

    if orientation == "horizontal":
        bar_rows = [r0 + rel_a, r0 + rel_b]
        region = [
            (rr, cc)
            for rr in range(bar_rows[0] + 1, bar_rows[1])
            for cc in range(c0 + 1, c1)
        ]
    else:
        bar_cols = [c0 + rel_a, c0 + rel_b]
        region = [
            (rr, cc)
            for rr in range(r0 + 1, r1)
            for cc in range(bar_cols[0] + 1, bar_cols[1])
        ]

    shape = _random_connected_shape(region, length)
    if shape is None:
        return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    for r, c in grey:
        grid[r, c] = GREY

    if orientation == "horizontal":
        for br in bar_rows:
            grid[br, c0 : c1 + 1] = RED
    else:
        for bc in bar_cols:
            grid[r0 : r1 + 1, bc] = RED

    for r, c in shape:
        grid[r, c] = GREEN

    try:
        info = _parse_scene(grid)
        out = _paint_output(grid, info)
        _semantic_validate(grid)
    except AssertionError:
        return None

    out_sig = _grid_sig(out)
    if out_sig in _SEEN_OUTPUT_SIGNATURES:
        return None
    _SEEN_OUTPUT_SIGNATURES.add(out_sig)
    return grid


def generate_input() -> np.ndarray:
    global _GEN_COUNTER

    call_index = _GEN_COUNTER
    for attempt in range(_MAX_GEN_ATTEMPTS):
        grid = _make_case(call_index + attempt)
        if grid is not None:
            _GEN_COUNTER = call_index + attempt + 1
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
