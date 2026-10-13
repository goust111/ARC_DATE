from __future__ import annotations

import random

import numpy as np


BG = 0
MARKER = 7
RING_COLORS = (1, 2, 3, 4, 5, 6, 8, 9)
MAX_GENERATION_ATTEMPTS = 300
SIDE_NAMES = ("top", "right", "bottom", "left")

BBox = tuple[int, int, int, int]
Coord = tuple[int, int]


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_from_coords(coords: set[Coord]) -> BBox:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _border_cycle(bbox: BBox) -> list[Coord]:
    r0, c0, r1, c1 = bbox
    assert r1 - r0 + 1 >= 3 and c1 - c0 + 1 >= 3

    cycle: list[Coord] = []
    for c in range(c0, c1 + 1):
        cycle.append((r0, c))
    for r in range(r0 + 1, r1):
        cycle.append((r, c1))
    for c in range(c1, c0 - 1, -1):
        cycle.append((r1, c))
    for r in range(r1 - 1, r0, -1):
        cycle.append((r, c0))
    return cycle


def _gap_side_and_inward(bbox: BBox, coord: Coord) -> tuple[str, Coord]:
    r0, c0, r1, c1 = bbox
    r, c = coord
    if r == r0 and c0 < c < c1:
        return "top", (1, 0)
    if c == c1 and r0 < r < r1:
        return "right", (0, -1)
    if r == r1 and c0 < c < c1:
        return "bottom", (-1, 0)
    if c == c0 and r0 < r < r1:
        return "left", (0, 1)
    raise AssertionError("gap must be a non-corner border cell")


def _project_to_ring(gap: Coord, inward: Coord, ring_cells: set[Coord], shape: tuple[int, int]) -> Coord:
    rows, cols = shape
    dr, dc = inward
    r, c = gap
    while True:
        r += dr
        c += dc
        assert 0 <= r < rows and 0 <= c < cols, "radial projection missed the inner ring"
        if (r, c) in ring_cells:
            return (r, c)


def _travel_until_gap(start_idx: int, gap_idx: int, clockwise: bool, cycle_len: int) -> None:
    assert cycle_len >= 4
    step = 1 if clockwise else -1
    idx = start_idx
    for _ in range(cycle_len):
        idx = (idx + step) % cycle_len
        if idx == gap_idx:
            return
    raise AssertionError("failed to encounter the layer gap")


def _extract_rings(grid: np.ndarray) -> tuple[list[dict], Coord, int]:
    g = np.asarray(grid, dtype=int)
    bg = _dominant_color(g)

    marker_positions = np.argwhere(g == MARKER)
    assert len(marker_positions) == 1, "expected a unique orange marker"
    marker = tuple(map(int, marker_positions[0]))

    colors = [int(color) for color in np.unique(g) if color not in (bg, MARKER)]
    assert 2 <= len(colors) <= 4, "expected 2-4 ring colors"

    rings: list[dict] = []
    for color in colors:
        coords = {tuple(map(int, rc)) for rc in np.argwhere(g == color)}
        bbox = _bbox_from_coords(coords)
        cycle = _border_cycle(bbox)
        cycle_set = set(cycle)
        assert coords <= cycle_set, "ring pixels must lie on a rectangular border"
        rings.append(
            {
                "color": color,
                "bbox": bbox,
                "coords": coords,
                "cycle": cycle,
                "cycle_set": cycle_set,
                "index": {coord: idx for idx, coord in enumerate(cycle)},
            }
        )

    rings.sort(key=lambda ring: (ring["bbox"][0], ring["bbox"][1], -ring["bbox"][2], -ring["bbox"][3]))
    assert marker in rings[0]["cycle_set"], "marker must lie on the outermost ring"

    for i, ring in enumerate(rings):
        occ = set(ring["coords"])
        if i == 0:
            occ.add(marker)
        missing = [coord for coord in ring["cycle"] if coord not in occ]
        assert len(missing) == 1, "each ring must have exactly one visible gap"
        ring["gap"] = missing[0]
        ring["gap_idx"] = ring["index"][ring["gap"]]
        assert len(occ) == len(ring["cycle"]) - 1

    for outer, inner in zip(rings, rings[1:]):
        or0, oc0, or1, oc1 = outer["bbox"]
        ir0, ic0, ir1, ic1 = inner["bbox"]
        assert or0 < ir0 < ir1 < or1 and oc0 < ic0 < ic1 < oc1, "rings must be nested"

    return rings, marker, bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Track the orange marker through nested rectangular rings with single-pixel gaps.

    Odd-numbered layers rotate clockwise, even-numbered layers rotate counter-clockwise.
    At each layer the marker moves until that layer's gap, then jumps radially inward to
    the aligned pixel on the next ring. On the innermost ring it completes one loop and
    therefore stops where it entered that ring.
    """
    g = np.asarray(grid, dtype=int)
    rings, marker, _bg = _extract_rings(g)

    out = g.copy()
    out[marker] = rings[0]["color"]

    current = marker
    for layer_idx, ring in enumerate(rings[:-1], start=1):
        assert current in ring["index"], "marker must start on the current ring"
        clockwise = (layer_idx % 2) == 1
        _travel_until_gap(ring["index"][current], ring["gap_idx"], clockwise, len(ring["cycle"]))

        _side, inward = _gap_side_and_inward(ring["bbox"], ring["gap"])
        next_ring = rings[layer_idx]
        current = _project_to_ring(ring["gap"], inward, next_ring["cycle_set"], g.shape)
        assert current != next_ring["gap"], "radial entry cannot land on the next gap"

    out[current] = MARKER
    return out


def _gap_candidates(bbox: BBox, inner_bbox: BBox | None) -> list[Coord]:
    r0, c0, r1, c1 = bbox
    candidates: list[Coord] = []

    if inner_bbox is None:
        for c in range(c0 + 1, c1):
            candidates.append((r0, c))
            candidates.append((r1, c))
        for r in range(r0 + 1, r1):
            candidates.append((r, c0))
            candidates.append((r, c1))
        return candidates

    ir0, ic0, ir1, ic1 = inner_bbox
    for c in range(max(c0 + 1, ic0), min(c1 - 1, ic1) + 1):
        candidates.append((r0, c))
        candidates.append((r1, c))
    for r in range(max(r0 + 1, ir0), min(r1 - 1, ir1) + 1):
        candidates.append((r, c0))
        candidates.append((r, c1))
    return candidates


def _project_to_bbox(gap: Coord, outer_bbox: BBox, inner_bbox: BBox) -> Coord:
    side, _inward = _gap_side_and_inward(outer_bbox, gap)
    r, c = gap
    ir0, ic0, ir1, ic1 = inner_bbox
    if side == "top":
        assert ic0 <= c <= ic1
        return (ir0, c)
    if side == "right":
        assert ir0 <= r <= ir1
        return (r, ic1)
    if side == "bottom":
        assert ic0 <= c <= ic1
        return (ir1, c)
    assert side == "left"
    assert ir0 <= r <= ir1
    return (r, ic0)


def _random_bbox_chain() -> list[BBox]:
    layers = random.choice((2, 3, 4))
    deltas = [random.choice((2, 3)) for _ in range(layers - 1)]
    shrink_total = sum(deltas)

    outer_h = random.randint(5 + 2 * shrink_total, 28)
    outer_w = random.randint(5 + 2 * shrink_total, 28)

    margin_top = random.randint(1, 3)
    margin_bottom = random.randint(1, 3)
    margin_left = random.randint(1, 3)
    margin_right = random.randint(1, 3)

    r0 = margin_top
    c0 = margin_left
    r1 = r0 + outer_h - 1
    c1 = c0 + outer_w - 1

    boxes = [(r0, c0, r1, c1)]
    for delta in deltas:
        r0 += delta
        c0 += delta
        r1 -= delta
        c1 -= delta
        boxes.append((r0, c0, r1, c1))

    height = boxes[0][2] + margin_bottom + 1
    width = boxes[0][3] + margin_right + 1
    assert height <= 34 and width <= 34
    return boxes


def _sample_gaps(boxes: list[BBox]) -> tuple[list[Coord], list[Coord]]:
    gaps: list[Coord] = []
    entries: list[Coord] = []

    for i, bbox in enumerate(boxes):
        inner_bbox = boxes[i + 1] if i + 1 < len(boxes) else None
        candidates = _gap_candidates(bbox, inner_bbox)

        if i > 0:
            entry = entries[-1]
            candidates = [coord for coord in candidates if coord != entry]

        assert candidates, "expected at least one valid gap candidate"
        gap = random.choice(candidates)
        gaps.append(gap)

        if inner_bbox is not None:
            entries.append(_project_to_bbox(gap, bbox, inner_bbox))

    return gaps, entries


def _render_scene(boxes: list[BBox], gaps: list[Coord], colors: list[int], marker: Coord) -> np.ndarray:
    height = boxes[0][2] + random.randint(2, 4) - 1
    width = boxes[0][3] + random.randint(2, 4) - 1
    grid = np.full((height, width), BG, dtype=int)

    for bbox, gap, color in zip(boxes, gaps, colors):
        for coord in _border_cycle(bbox):
            if coord != gap:
                grid[coord] = color

    grid[marker] = MARKER
    return grid


def _gen_attempt() -> np.ndarray | None:
    boxes = _random_bbox_chain()
    colors = random.sample(RING_COLORS, len(boxes))
    gaps, entries = _sample_gaps(boxes)

    outer_cycle = [coord for coord in _border_cycle(boxes[0]) if coord != gaps[0]]
    marker = random.choice(outer_cycle)
    grid = _render_scene(boxes, gaps, colors, marker)

    if entries and entries[-1] == gaps[-1]:
        return None

    try:
        rings, parsed_marker, bg = _extract_rings(grid)
        assert bg == BG
        assert parsed_marker == marker
        assert len(rings) == len(boxes)
        main(grid)
    except Exception:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ring-track input")
