from __future__ import annotations

import random
import numpy as np


BG = 0
TRACK = 8
MARKER = 9

OBJECT_COLORS = [1, 2, 3, 4, 5, 6, 7]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

MAX_ATTEMPTS = 400
PLACEMENT_TRIES = 120
BORDER_GAP = 1
RING_GAP = 2

GRID_SIDE_BY_COUNT = {
    2: (18, 24),
    3: (20, 28),
    4: (24, 30),
}

SHAPES_BY_SIZE = {
    2: [
        ((0, 0), (0, 1)),
    ],
    3: [
        ((0, 0), (0, 1), (0, 2)),
        ((0, 0), (1, 0), (1, 1)),
    ],
    4: [
        ((0, 0), (0, 1), (0, 2), (0, 3)),
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((0, 0), (1, 0), (2, 0), (2, 1)),
        ((0, 1), (1, 0), (1, 1), (1, 2)),
        ((0, 1), (0, 2), (1, 0), (1, 1)),
    ],
}


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
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(cells: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(cells: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    r0, c0, _r1, _c1 = _bbox(cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _shape_dims(cells: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    r0, c0, r1, c1 = _bbox(cells)
    return r1 - r0 + 1, c1 - c0 + 1


def _rotate_shape(cells: tuple[tuple[int, int], ...], k: int) -> tuple[tuple[int, int], ...]:
    out = list(cells)
    for _ in range(k % 4):
        out = [(c, -r) for r, c in out]
    return _normalize(out)


def _flip_shape(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return _normalize([(r, -c) for r, c in cells])


def _can_shift(
    cells: tuple[tuple[int, int], ...],
    dr: int,
    dc: int,
    inner_h: int,
    inner_w: int,
) -> bool:
    for r, c in cells:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < inner_h and 0 <= nc < inner_w):
            return False
    return True


def _shift_state(
    cells: tuple[tuple[int, int], ...],
    dr: int,
    dc: int,
) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((r + dr, c + dc) for r, c in cells))


def _step_direction(
    cells: tuple[tuple[int, int], ...],
    inner_h: int,
    inner_w: int,
    clockwise: bool,
) -> tuple[int, int]:
    touches_top = any(r == 0 for r, _ in cells)
    touches_bottom = any(r == inner_h - 1 for r, _ in cells)
    touches_left = any(c == 0 for _, c in cells)
    touches_right = any(c == inner_w - 1 for _, c in cells)

    if clockwise:
        candidates = [
            (touches_top, 0, 1),
            (touches_right, 1, 0),
            (touches_bottom, 0, -1),
            (touches_left, -1, 0),
        ]
    else:
        candidates = [
            (touches_top, 0, -1),
            (touches_left, 1, 0),
            (touches_bottom, 0, 1),
            (touches_right, -1, 0),
        ]

    for touches_side, dr, dc in candidates:
        if touches_side and _can_shift(cells, dr, dc, inner_h, inner_w):
            return dr, dc
    raise AssertionError("object must touch the inner wall and have a valid slide direction")


def _enumerate_cycle(
    start: tuple[tuple[int, int], ...],
    inner_h: int,
    inner_w: int,
    clockwise: bool,
) -> tuple[list[tuple[tuple[int, int], ...]], list[tuple[int, int]]]:
    seen: dict[tuple[tuple[int, int], ...], int] = {}
    states: list[tuple[tuple[int, int], ...]] = []
    dirs: list[tuple[int, int]] = []
    state = start

    while state not in seen:
        seen[state] = len(states)
        states.append(state)
        step = _step_direction(state, inner_h, inner_w, clockwise)
        dirs.append(step)
        state = _shift_state(state, *step)

    cycle_start = seen[state]
    cycle = states[cycle_start:]
    cycle_dirs = dirs[cycle_start:]
    assert cycle and state == cycle[0], "expected a closed boundary cycle"
    return cycle, cycle_dirs


def _perimeter_set(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    pts = set()
    for c in range(c0, c1 + 1):
        pts.add((r0, c))
        pts.add((r1, c))
    for r in range(r0, r1 + 1):
        pts.add((r, c0))
        pts.add((r, c1))
    return pts


def _marker_side(markers: list[tuple[int, int]], bbox: tuple[int, int, int, int]) -> str:
    r0, c0, r1, c1 = bbox
    if all(r == r0 and c0 < c < c1 for r, c in markers):
        return "top"
    if all(r == r1 and c0 < c < c1 for r, c in markers):
        return "bottom"
    if all(c == c0 and r0 < r < r1 for r, c in markers):
        return "left"
    if all(c == c1 and r0 < r < r1 for r, c in markers):
        return "right"
    raise AssertionError("marker pixels must lie on exactly one non-corner edge")


def _ring_records(grid: np.ndarray) -> list[dict]:
    mask = (grid == TRACK) | (grid == MARKER)
    comps = _components(mask)
    assert len(comps) >= 2, "expected multiple ring tracks"

    records: list[dict] = []
    for comp in comps:
        comp_set = set(comp)
        r0, c0, r1, c1 = _bbox(comp)
        assert r1 - r0 + 1 >= 6 and c1 - c0 + 1 >= 6, "ring bbox is too small"
        assert comp_set == _perimeter_set(r0, c0, r1, c1), "ring component must be a 1-pixel rectangle frame"

        markers = [(r, c) for r, c in comp if int(grid[r, c]) == MARKER]
        assert 1 <= len(markers) <= 4, "expected 1-4 marker pixels on each ring"
        side = _marker_side(markers, (r0, c0, r1, c1))

        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        obj_mask = (inner != BG) & (inner != TRACK) & (inner != MARKER)
        obj_comps = _components(obj_mask)
        assert len(obj_comps) == 1, "each ring must contain exactly one interior object"
        obj_local = tuple(sorted((r, c) for r, c in obj_comps[0]))
        colors = [int(inner[r, c]) for r, c in obj_local]

        records.append(
            {
                "bbox": (r0, c0, r1, c1),
                "marker_cells": markers,
                "steps": len(markers),
                "clockwise": side in {"top", "right"},
                "obj_local": obj_local,
                "obj_colors": colors,
                "inner_h": inner.shape[0],
                "inner_w": inner.shape[1],
            }
        )

    return records


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find each marked rectangular ring, read direction from the marker side and
    step count from the number of markers, then slide the interior object along
    the ring's inner wall.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    rings = _ring_records(g)

    for ring in rings:
        r0, c0, _r1, _c1 = ring["bbox"]
        inner_r0 = r0 + 1
        inner_c0 = c0 + 1
        state = ring["obj_local"]

        for _ in range(ring["steps"]):
            dr, dc = _step_direction(state, ring["inner_h"], ring["inner_w"], ring["clockwise"])
            state = _shift_state(state, dr, dc)

        for r, c in ring["marker_cells"]:
            out[r, c] = TRACK

        for r, c in ring["obj_local"]:
            out[inner_r0 + r, inner_c0 + c] = BG

        for (r, c), color in zip(state, ring["obj_colors"], strict=True):
            out[inner_r0 + r, inner_c0 + c] = color

    return out


def _random_shape() -> tuple[tuple[tuple[int, int], ...], list[int]]:
    size = random.randint(2, 4)
    shape = tuple(SHAPES_BY_SIZE[size][random.randrange(len(SHAPES_BY_SIZE[size]))])
    if random.random() < 0.5:
        shape = _flip_shape(shape)
    shape = _rotate_shape(shape, random.randrange(4))
    colors = random.sample(OBJECT_COLORS, len(shape))
    return shape, colors


def _build_ring_spec(clockwise: bool) -> dict | None:
    shape, colors = _random_shape()
    shape_h, shape_w = _shape_dims(shape)

    min_inner_h = max(shape_h + 2, 4)
    min_inner_w = max(shape_w + 2, 4)
    max_inner_h = min(10, min_inner_h + 3)
    max_inner_w = min(10, min_inner_w + 3)
    if min_inner_h > max_inner_h or min_inner_w > max_inner_w:
        return None

    inner_h = random.randint(min_inner_h, max_inner_h)
    inner_w = random.randint(min_inner_w, max_inner_w)

    start = shape
    cycle, cycle_dirs = _enumerate_cycle(start, inner_h, inner_w, clockwise)
    if len(cycle) <= 4:
        return None

    steps = random.randint(1, 4)
    valid_indices = [i for i in range(len(cycle)) if cycle[(i + steps) % len(cycle)] != cycle[i]]
    if not valid_indices:
        return None

    prefer_turn = steps > 1 and random.random() < 0.5
    if prefer_turn:
        turning = []
        for i in valid_indices:
            seen_turn = False
            prev_dir = cycle_dirs[i]
            for j in range(1, steps):
                cur_dir = cycle_dirs[(i + j) % len(cycle)]
                if cur_dir != prev_dir:
                    seen_turn = True
                    break
                prev_dir = cur_dir
            if seen_turn:
                turning.append(i)
        if turning:
            valid_indices = turning

    start_idx = random.choice(valid_indices)
    side_choices = ["top", "right"] if clockwise else ["bottom", "left"]
    marker_side = random.choice(side_choices)

    outer_h = inner_h + 2
    outer_w = inner_w + 2
    if marker_side in {"top", "bottom"} and outer_w - 2 < steps:
        return None
    if marker_side in {"left", "right"} and outer_h - 2 < steps:
        return None

    return {
        "clockwise": clockwise,
        "shape": shape,
        "colors": colors,
        "inner_h": inner_h,
        "inner_w": inner_w,
        "outer_h": outer_h,
        "outer_w": outer_w,
        "steps": steps,
        "marker_side": marker_side,
        "state": cycle[start_idx],
    }


def _marker_positions(outer_h: int, outer_w: int, side: str, count: int) -> list[tuple[int, int]]:
    if side == "top":
        coords = [(0, c) for c in range(1, outer_w - 1)]
    elif side == "bottom":
        coords = [(outer_h - 1, c) for c in range(1, outer_w - 1)]
    elif side == "left":
        coords = [(r, 0) for r in range(1, outer_h - 1)]
    elif side == "right":
        coords = [(r, outer_w - 1) for r in range(1, outer_h - 1)]
    else:
        raise AssertionError("unknown marker side")
    return random.sample(coords, count)


def _draw_ring(grid: np.ndarray, top: int, left: int, spec: dict) -> None:
    outer_h = spec["outer_h"]
    outer_w = spec["outer_w"]
    bottom = top + outer_h - 1
    right = left + outer_w - 1

    grid[top, left : right + 1] = TRACK
    grid[bottom, left : right + 1] = TRACK
    grid[top : bottom + 1, left] = TRACK
    grid[top : bottom + 1, right] = TRACK

    for r, c in _marker_positions(outer_h, outer_w, spec["marker_side"], spec["steps"]):
        grid[top + r, left + c] = MARKER

    inner_top = top + 1
    inner_left = left + 1
    for (r, c), color in zip(spec["state"], spec["colors"], strict=True):
        grid[inner_top + r, inner_left + c] = color


def _rect_clear(occupied: np.ndarray, top: int, left: int, h: int, w: int, pad: int) -> bool:
    h0, w0 = occupied.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h0, top + h + pad)
    c1 = min(w0, left + w + pad)
    return not occupied[r0:r1, c0:c1].any()


def _occupy_rect(occupied: np.ndarray, top: int, left: int, h: int, w: int, pad: int) -> None:
    h0, w0 = occupied.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h0, top + h + pad)
    c1 = min(w0, left + w + pad)
    occupied[r0:r1, c0:c1] = True


def _grid_shape(n_rings: int) -> tuple[int, int]:
    lo, hi = GRID_SIDE_BY_COUNT[n_rings]
    return random.randint(lo, hi), random.randint(lo, hi)


def _gen_attempt() -> np.ndarray | None:
    n_rings = random.randint(2, 4)
    directions = ["cw", "ccw"]
    while len(directions) < n_rings:
        directions.append(random.choice(["cw", "ccw"]))
    random.shuffle(directions)

    specs = []
    for direction in directions:
        spec = _build_ring_spec(direction == "cw")
        if spec is None:
            return None
        specs.append(spec)

    specs.sort(key=lambda item: item["outer_h"] * item["outer_w"], reverse=True)
    h, w = _grid_shape(n_rings)
    grid = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    for spec in specs:
        outer_h = spec["outer_h"]
        outer_w = spec["outer_w"]
        if outer_h + 2 * BORDER_GAP > h or outer_w + 2 * BORDER_GAP > w:
            return None

        placed = False
        for _ in range(PLACEMENT_TRIES):
            top = random.randint(BORDER_GAP, h - outer_h - BORDER_GAP)
            left = random.randint(BORDER_GAP, w - outer_w - BORDER_GAP)
            if not _rect_clear(occupied, top, left, outer_h, outer_w, RING_GAP):
                continue
            _draw_ring(grid, top, left, spec)
            _occupy_rect(occupied, top, left, outer_h, outer_w, RING_GAP)
            placed = True
            break
        if not placed:
            return None

    main(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
