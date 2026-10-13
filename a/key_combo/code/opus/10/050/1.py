from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BACKGROUND = 0
FILL = 2
WALL = 3
MARKER = 4
SEPARATOR = 5
PANEL_SIZE = 7
GEN_ATTEMPTS = 400

BBox = tuple[int, int, int, int]
Coord = tuple[int, int]


def _neighbors4(r: int, c: int) -> Iterable[Coord]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_separator_row(grid: np.ndarray) -> int:
    g = np.asarray(grid, dtype=int)
    h, _w = g.shape
    rows = [
        r
        for r in range(1, h - 1)
        if r == h - r - 1 and int(g[r, 0]) != BACKGROUND and np.all(g[r] == g[r, 0])
    ]
    assert len(rows) == 1, "expected one solid separator row splitting equal panels"
    return int(rows[0])


def _enclosed_regions(top: np.ndarray) -> list[list[Coord]]:
    h, w = top.shape
    passable = top != WALL
    seen = np.zeros_like(passable, dtype=bool)
    regions: list[list[Coord]] = []

    for r in range(h):
        for c in range(w):
            if not passable[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[Coord] = []
            touches_edge = False
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                if rr in (0, h - 1) or cc in (0, w - 1):
                    touches_edge = True
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            if not touches_edge:
                regions.append(coords)
    return regions


def _render_top(boxes: list[BBox]) -> np.ndarray:
    top = np.zeros((PANEL_SIZE, PANEL_SIZE), dtype=int)
    for r0, c0, r1, c1 in boxes:
        top[r0, c0 : c1 + 1] = WALL
        top[r1, c0 : c1 + 1] = WALL
        top[r0 : r1 + 1, c0] = WALL
        top[r0 : r1 + 1, c1] = WALL
    return top


def _interior_coords(box: BBox) -> list[Coord]:
    r0, c0, r1, c1 = box
    return [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]


def _wall_coords(box: BBox) -> list[Coord]:
    r0, c0, r1, c1 = box
    coords: set[Coord] = set()
    for c in range(c0, c1 + 1):
        coords.add((r0, c))
        coords.add((r1, c))
    for r in range(r0, r1 + 1):
        coords.add((r, c0))
        coords.add((r, c1))
    return sorted(coords)


def _box_mask(boxes: list[BBox]) -> np.ndarray:
    mask = np.zeros((PANEL_SIZE, PANEL_SIZE), dtype=bool)
    for r0, c0, r1, c1 in boxes:
        mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay the lower marker panel onto the upper frame panel and fill only the enclosed regions that contain markers.

    Input structure:
    - The grid contains two equal-size panels separated by one solid non-black row.
    - The upper panel is black except for green 1-pixel rectangular frame walls that enclose black regions.
    - The lower panel is black except for yellow marker pixels placed at aligned positions.

    Rule:
    - Remove the separator row and align the two panels cell-by-cell.
    - In the upper panel, find every 4-connected non-wall region that is fully enclosed by green walls.
    - If any aligned yellow marker falls inside one enclosed region, fill that region red; markers on walls or outside all enclosed regions are ignored.
    - Keep the green walls unchanged and append one extra bottom row whose leftmost red run length equals the number of filled regions.

    Key insight:
    - The lower panel is only an activation signal: yellow matters only when it lands strictly inside a closed black region, not merely inside a frame's bounding box or on top of a wall.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep = _find_separator_row(g)
    top = g[:sep].copy()
    bottom = g[sep + 1 :].copy()
    assert top.shape == bottom.shape, "panels must be the same size"

    regions = _enclosed_regions(top)
    out = top.copy()
    activated = 0
    for coords in regions:
        if any(int(bottom[r, c]) == MARKER for r, c in coords):
            activated += 1
            for r, c in coords:
                out[r, c] = FILL

    bar = np.zeros((1, out.shape[1]), dtype=int)
    bar[0, :activated] = FILL
    return np.vstack([out, bar])


ALL_BOXES: tuple[BBox, ...] = tuple(
    (r0, c0, r0 + h - 1, c0 + w - 1)
    for h in range(3, 6)
    for w in range(3, 6)
    for r0 in range(PANEL_SIZE - h + 1)
    for c0 in range(PANEL_SIZE - w + 1)
)


def _interval_gap(a0: int, a1: int, b0: int, b1: int) -> int:
    if a1 < b0:
        return b0 - a1 - 1
    if b1 < a0:
        return a0 - b1 - 1
    return -1


def _boxes_compatible(a: BBox, b: BBox) -> bool:
    row_gap = _interval_gap(a[0], a[2], b[0], b[2])
    col_gap = _interval_gap(a[1], a[3], b[1], b[3])
    return row_gap >= 1 or col_gap >= 1


def _choose_layout(num_boxes: int) -> list[BBox] | None:
    boxes = list(ALL_BOXES)
    random.shuffle(boxes)
    chosen: list[BBox] = []

    def backtrack(start: int) -> bool:
        if len(chosen) == num_boxes:
            return True
        need = num_boxes - len(chosen)
        for i in range(start, len(boxes)):
            if len(boxes) - i < need:
                return False
            box = boxes[i]
            if all(_boxes_compatible(box, prev) for prev in chosen):
                chosen.append(box)
                if backtrack(i + 1):
                    return True
                chosen.pop()
        return False

    if backtrack(0):
        chosen.sort()
        return chosen
    return None


def _build_expected(top: np.ndarray, box_to_active: dict[BBox, bool]) -> np.ndarray:
    out = top.copy()
    active_count = 0
    for box, active in box_to_active.items():
        if active:
            active_count += 1
            for r, c in _interior_coords(box):
                out[r, c] = FILL
    bar = np.zeros((1, PANEL_SIZE), dtype=int)
    bar[0, :active_count] = FILL
    return np.vstack([out, bar])


def _fill_all(top: np.ndarray, boxes: list[BBox]) -> np.ndarray:
    out = top.copy()
    for box in boxes:
        for r, c in _interior_coords(box):
            out[r, c] = FILL
    return out


def _pick_unique(candidates: list[Coord], used: set[Coord]) -> Coord:
    available = [coord for coord in candidates if coord not in used]
    assert available, "expected at least one unused candidate"
    return random.choice(available)


def _validate_case(grid: np.ndarray, boxes: list[BBox], box_to_active: dict[BBox, bool], wall_neg: Coord, outside_neg: Coord) -> None:
    assert grid.shape == (PANEL_SIZE * 2 + 1, PANEL_SIZE)
    sep = _find_separator_row(grid)
    assert sep == PANEL_SIZE
    top = grid[:sep]
    bottom = grid[sep + 1 :]
    assert np.array_equal(top, _render_top(boxes))

    expected_region_sets = {frozenset(_interior_coords(box)): bool(active) for box, active in box_to_active.items()}
    actual_regions = _enclosed_regions(top)
    actual_region_sets = {frozenset(region) for region in actual_regions}
    assert actual_region_sets == set(expected_region_sets), "constructed boxes must match detected enclosed regions"

    for region in actual_regions:
        region_key = frozenset(region)
        should_activate = expected_region_sets[region_key]
        has_marker = any(int(bottom[r, c]) == MARKER for r, c in region)
        assert has_marker == should_activate, "marker placement must match chosen activation set"

    wr, wc = wall_neg
    orow, ocol = outside_neg
    assert top[wr, wc] == WALL
    assert top[orow, ocol] == BACKGROUND
    box_mask = _box_mask(boxes)
    assert not box_mask[orow, ocol], "outside counterexample must be outside every frame bbox"

    false_bbox_hits = 0
    for box, active in box_to_active.items():
        r0, c0, r1, c1 = box
        bbox_has_marker = bool(np.any(bottom[r0 : r1 + 1, c0 : c1 + 1] == MARKER))
        if bbox_has_marker and not active:
            false_bbox_hits += 1
    assert false_bbox_hits >= 1, "need a wall marker on an inactive frame as a counterexample"

    expected = _build_expected(top, box_to_active)
    assert np.array_equal(main(grid), expected), "solver output must match the semantic construction"
    assert not np.array_equal(_fill_all(top, boxes), expected[:-1]), "sample must include at least one inactive enclosed region"


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        num_boxes = random.choice((2, 2, 3, 3, 4))
        boxes = _choose_layout(num_boxes)
        if boxes is None:
            continue

        active_count = random.randint(1, num_boxes - 1)
        active_boxes = set(random.sample(boxes, active_count))
        box_to_active = {box: box in active_boxes for box in boxes}

        top = _render_top(boxes)
        bottom = np.zeros_like(top)
        used: set[Coord] = set()

        for box in boxes:
            if box_to_active[box]:
                pos = _pick_unique(_interior_coords(box), used)
                used.add(pos)
                bottom[pos] = MARKER

        inactive_boxes = [box for box in boxes if not box_to_active[box]]
        wall_candidates = [coord for box in inactive_boxes for coord in _wall_coords(box)]
        wall_neg = _pick_unique(wall_candidates, used)
        used.add(wall_neg)
        bottom[wall_neg] = MARKER

        outside_candidates = [tuple(map(int, rc)) for rc in np.argwhere(~_box_mask(boxes))]
        outside_neg = _pick_unique(outside_candidates, used)
        used.add(outside_neg)
        bottom[outside_neg] = MARKER

        min_markers = len(used)
        max_markers = min(6, PANEL_SIZE * PANEL_SIZE)
        marker_total = random.randint(max(3, min_markers), max_markers)

        extra_pool = list(
            {
                coord
                for box in boxes
                for coord in (_interior_coords(box) if box_to_active[box] else _wall_coords(box))
            }
            | set(outside_candidates)
        )
        while len(used) < marker_total:
            pos = _pick_unique(extra_pool, used)
            used.add(pos)
            bottom[pos] = MARKER

        grid = np.vstack(
            [
                top,
                np.full((1, PANEL_SIZE), SEPARATOR, dtype=int),
                bottom,
            ]
        )
        _validate_case(grid, boxes, box_to_active, wall_neg, outside_neg)
        return grid

    raise AssertionError("failed to generate a valid puzzle instance")
