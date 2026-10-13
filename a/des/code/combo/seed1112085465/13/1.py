from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
FRAGMENT_COLORS = (1, 3, 4, 5, 6, 7, 8, 9)

DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _unique_nearest_anchor(r: int, c: int, anchors: list[tuple[int, int]]) -> int | None:
    best_idx = -1
    best_dist = None
    tied = False
    for idx, (ar, ac) in enumerate(anchors):
        dist = abs(r - ar) + abs(c - ac)
        if best_dist is None or dist < best_dist:
            best_idx = idx
            best_dist = dist
            tied = False
        elif dist == best_dist:
            tied = True
    return None if tied else best_idx


def _fragment_direction(anchor: tuple[int, int], point: tuple[int, int]) -> tuple[str, int]:
    ar, ac = anchor
    r, c = point
    if r == ar and c != ac:
        return ("left" if c < ac else "right", abs(c - ac))
    if c == ac and r != ar:
        return ("up" if r < ar else "down", abs(r - ar))
    raise AssertionError("each fragment must share a row or column with its nearest anchor")


def _collect_assignments(
    g: np.ndarray,
) -> tuple[list[tuple[int, int]], dict[tuple[int, int], dict[str, list[tuple[int, int]]]]]:
    anchors = [tuple(map(int, rc)) for rc in np.argwhere(g == RED)]
    assert anchors, "expected at least one red anchor"

    groups = {anchor: {direction: [] for direction in DIRS} for anchor in anchors}
    fragments = np.argwhere((g != BLACK) & (g != RED))
    assert len(fragments) > 0, "expected at least one fragment"

    for r_raw, c_raw in fragments:
        r = int(r_raw)
        c = int(c_raw)
        anchor_idx = _unique_nearest_anchor(r, c, anchors)
        assert anchor_idx is not None, "each fragment must have a unique nearest anchor"
        anchor = anchors[anchor_idx]
        direction, distance = _fragment_direction(anchor, (r, c))
        groups[anchor][direction].append((distance, int(g[r, c])))

    return anchors, groups


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group single-pixel fragments by their unique nearest red anchor and rebuild
    each anchor's fragments into a compact point-symmetric pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    anchors, groups = _collect_assignments(g)
    h, w = g.shape

    out = np.full((h, w), BLACK, dtype=int)
    for ar, ac in anchors:
        out[ar, ac] = RED

    for ar, ac in anchors:
        for direction, (dr, dc) in DIRS.items():
            entries = sorted(groups[(ar, ac)][direction], key=lambda item: item[0])
            for step, (_distance, color) in enumerate(entries, start=1):
                for sign in (1, -1):
                    rr = ar + sign * dr * step
                    cc = ac + sign * dc * step
                    assert 0 <= rr < h and 0 <= cc < w, "rebuilt fragment must stay inside the grid"
                    current = int(out[rr, cc])
                    assert current in (BLACK, color), "rebuild cannot require conflicting colors"
                    out[rr, cc] = color

    return out


GRID_HEIGHT_RANGE = (11, 20)
GRID_WIDTH_RANGE = (11, 20)
ANCHOR_MARGIN = 3
MAX_ANCHORS = 4
MIN_ANCHOR_DISTANCE = 5
MAX_GROUP_LEN = 3
MAX_GENERATION_ATTEMPTS = 400


def _sample_anchors(h: int, w: int, count: int) -> list[tuple[int, int]] | None:
    row_choices = list(range(ANCHOR_MARGIN, h - ANCHOR_MARGIN))
    col_choices = list(range(ANCHOR_MARGIN, w - ANCHOR_MARGIN))
    if len(row_choices) < count or len(col_choices) < count:
        return None

    for _ in range(80):
        rows = random.sample(row_choices, count)
        cols = random.sample(col_choices, count)
        random.shuffle(cols)
        anchors = list(zip(rows, cols))
        if all(
            abs(r1 - r2) + abs(c1 - c2) >= MIN_ANCHOR_DISTANCE
            for i, (r1, c1) in enumerate(anchors)
            for r2, c2 in anchors[i + 1 :]
        ):
            return anchors
    return None


def _ray_valid_distances(
    anchor_idx: int,
    direction: str,
    anchors: list[tuple[int, int]],
    shape: tuple[int, int],
) -> list[int]:
    ar, ac = anchors[anchor_idx]
    dr, dc = DIRS[direction]
    h, w = shape

    valid: list[int] = []
    step = 1
    while True:
        rr = ar + dr * step
        cc = ac + dc * step
        if not (0 <= rr < h and 0 <= cc < w):
            break
        if _unique_nearest_anchor(rr, cc, anchors) == anchor_idx:
            valid.append(step)
        step += 1
    return valid


def _axis_capacity(anchor: tuple[int, int], shape: tuple[int, int], direction: str) -> int:
    ar, ac = anchor
    h, w = shape
    if direction in ("up", "down"):
        return min(ar, h - 1 - ar)
    return min(ac, w - 1 - ac)


def _output_cells(anchor: tuple[int, int], direction: str, count: int) -> list[tuple[int, int]]:
    ar, ac = anchor
    dr, dc = DIRS[direction]

    cells: list[tuple[int, int]] = []
    for step in range(1, count + 1):
        cells.append((ar + dr * step, ac + dc * step))
        cells.append((ar - dr * step, ac - dc * step))
    return cells


def _choose_plan_for_anchor(
    anchor_idx: int,
    anchors: list[tuple[int, int]],
    shape: tuple[int, int],
    occupied_input: set[tuple[int, int]],
    reserved_output: set[tuple[int, int]],
) -> tuple[dict[str, tuple[int, ...]], set[tuple[int, int]], set[tuple[int, int]]] | None:
    anchor = anchors[anchor_idx]
    valid: dict[str, list[int]] = {}

    for direction in DIRS:
        ray = _ray_valid_distances(anchor_idx, direction, anchors, shape)
        valid[direction] = [
            distance
            for distance in ray
            if (anchor[0] + DIRS[direction][0] * distance, anchor[1] + DIRS[direction][1] * distance)
            not in occupied_input
        ]

    vertical = [
        direction
        for direction in ("up", "down")
        if min(len(valid[direction]), _axis_capacity(anchor, shape, direction), MAX_GROUP_LEN) >= 1
    ]
    horizontal = [
        direction
        for direction in ("left", "right")
        if min(len(valid[direction]), _axis_capacity(anchor, shape, direction), MAX_GROUP_LEN) >= 1
    ]

    patterns: list[tuple[str, ...]] = []
    for direction in vertical:
        patterns.append((direction,))
    for direction in horizontal:
        patterns.append((direction,))
    for vertical_direction in vertical:
        for horizontal_direction in horizontal:
            patterns.append((vertical_direction, horizontal_direction))

    random.shuffle(patterns)
    for pattern in patterns:
        for _ in range(20):
            plan: dict[str, tuple[int, ...]] = {}
            local_input: set[tuple[int, int]] = set()
            local_output: set[tuple[int, int]] = set()
            ok = True

            for direction in pattern:
                candidates = [
                    distance
                    for distance in valid[direction]
                    if (anchor[0] + DIRS[direction][0] * distance, anchor[1] + DIRS[direction][1] * distance)
                    not in local_input
                ]
                max_count = min(len(candidates), _axis_capacity(anchor, shape, direction), MAX_GROUP_LEN)
                if max_count < 1:
                    ok = False
                    break

                count = random.randint(1, max_count)
                output_cells = _output_cells(anchor, direction, count)
                if any(cell in reserved_output or cell in local_output for cell in output_cells):
                    ok = False
                    break

                chosen = random.sample(candidates, count)
                plan[direction] = tuple(chosen)
                local_output.update(output_cells)
                for distance in chosen:
                    local_input.add((anchor[0] + DIRS[direction][0] * distance, anchor[1] + DIRS[direction][1] * distance))

            if ok:
                return plan, local_input, local_output

    return None


def _build_candidate() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    max_count = min(MAX_ANCHORS, h - 2 * ANCHOR_MARGIN, w - 2 * ANCHOR_MARGIN)
    if max_count < 1:
        return None

    anchor_count = random.randint(1, max_count)
    anchors = _sample_anchors(h, w, anchor_count)
    if anchors is None:
        return None

    occupied_input = set(anchors)
    reserved_output = set(anchors)
    plans: dict[int, dict[str, tuple[int, ...]]] = {}

    order = list(range(anchor_count))
    random.shuffle(order)
    for anchor_idx in order:
        chosen = _choose_plan_for_anchor(anchor_idx, anchors, (h, w), occupied_input, reserved_output)
        if chosen is None:
            return None
        plan, new_inputs, new_outputs = chosen
        plans[anchor_idx] = plan
        occupied_input.update(new_inputs)
        reserved_output.update(new_outputs)

    palette_size = random.randint(2, min(6, len(FRAGMENT_COLORS)))
    palette = random.sample(FRAGMENT_COLORS, palette_size)

    grid = np.full((h, w), BLACK, dtype=int)
    for ar, ac in anchors:
        grid[ar, ac] = RED

    for anchor_idx, plan in plans.items():
        ar, ac = anchors[anchor_idx]
        for direction, distances in plan.items():
            dr, dc = DIRS[direction]
            for distance in distances:
                rr = ar + dr * distance
                cc = ac + dc * distance
                grid[rr, cc] = random.choice(palette)

    return grid


def _is_valid_candidate(grid: np.ndarray) -> bool:
    try:
        anchors, groups = _collect_assignments(grid)
        out = main(grid)
    except Exception:
        return False

    fragment_count = int(np.count_nonzero((grid != BLACK) & (grid != RED)))
    if fragment_count == 0 or np.array_equal(grid, out):
        return False
    if int(np.count_nonzero((out != BLACK) & (out != RED))) != 2 * fragment_count:
        return False

    for anchor in anchors:
        directions = groups[anchor]
        if not any(directions.values()):
            return False
        if directions["up"] and directions["down"]:
            return False
        if directions["left"] and directions["right"]:
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        candidate = _build_candidate()
        if candidate is not None and _is_valid_candidate(candidate):
            return candidate
    raise RuntimeError("failed to generate a valid input")
