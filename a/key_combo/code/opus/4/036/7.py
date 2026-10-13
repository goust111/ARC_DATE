import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

DIRS4 = {
    UP: (-1, 0),
    RIGHT: (0, 1),
    DOWN: (1, 0),
    LEFT: (0, -1),
}


# Offsets are relative to the tip pixel. Every template is 4-connected and has
# a unique tip by construction, so the tip/direction can be recovered by exact
# template matching.
TEMPLATES = {
    "corner3": {
        UP: [(0, 0), (1, 0), (1, 1)],
        RIGHT: [(0, 0), (0, -1), (1, -1)],
        DOWN: [(0, 0), (-1, 0), (-1, -1)],
        LEFT: [(0, 0), (0, 1), (-1, 1)],
    },
    "tee4": {
        UP: [(0, 0), (1, -1), (1, 0), (1, 1)],
        RIGHT: [(0, 0), (-1, -1), (0, -1), (1, -1)],
        DOWN: [(0, 0), (-1, -1), (-1, 0), (-1, 1)],
        LEFT: [(0, 0), (-1, 1), (0, 1), (1, 1)],
    },
    "hook4": {
        UP: [(0, 0), (1, 0), (2, -1), (2, 0)],
        RIGHT: [(0, 0), (0, -1), (-1, -2), (0, -2)],
        DOWN: [(0, 0), (-1, 0), (-2, 1), (-2, 0)],
        LEFT: [(0, 0), (0, 1), (1, 2), (0, 2)],
    },
}


def _neighbors4(r, c):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_single_component(coords):
    coords = set(coords)
    if not coords:
        return set()
    start = next(iter(coords))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in coords and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return seen


def _sprite_cells(template_name, direction, tip):
    tr, tc = tip
    return {(tr + dr, tc + dc) for dr, dc in TEMPLATES[template_name][direction]}


def _sprite_fits(shape, template_name, direction, tip):
    h, w = shape
    for r, c in _sprite_cells(template_name, direction, tip):
        if not (0 <= r < h and 0 <= c < w):
            return False
    return True


def _identify_sprite(grid):
    red_coords = {tuple(map(int, rc)) for rc in np.argwhere(grid == RED)}
    assert len(red_coords) in (3, 4), "expected a 3- or 4-cell red sprite"
    assert _find_single_component(red_coords) == red_coords, "sprite must be 4-connected"

    matches = []
    for template_name, by_dir in TEMPLATES.items():
        for direction in by_dir:
            for tip in red_coords:
                if _sprite_cells(template_name, direction, tip) == red_coords:
                    matches.append((template_name, direction, tip))
    assert len(matches) == 1, "sprite tip and orientation must be uniquely identifiable"
    return matches[0]


def _forward_distance(src, direction, dst):
    sr, sc = src
    dr, dc = dst
    if direction == UP and sc == dc and dr < sr:
        return sr - dr
    if direction == RIGHT and sr == dr and dc > sc:
        return dc - sc
    if direction == DOWN and sc == dc and dr > sr:
        return dr - sr
    if direction == LEFT and sr == dr and dc < sc:
        return sc - dc
    return None


def _reachable_markers(tip, markers, shape, template_name):
    reachable = {}
    for direction in (UP, RIGHT, DOWN, LEFT):
        candidates = []
        for marker in markers:
            dist = _forward_distance(tip, direction, marker)
            if dist is None:
                continue
            if not _sprite_fits(shape, template_name, direction, marker):
                continue
            candidates.append((dist, marker))
        if candidates:
            candidates.sort()
            reachable[direction] = candidates
    return reachable


def _simulate_path(shape, template_name, initial_tip, initial_direction, markers):
    tip = tuple(map(int, initial_tip))
    current_direction = int(initial_direction)
    remaining = {tuple(map(int, m)) for m in markers}
    consumed = []

    max_steps = shape[0] * shape[1] + 1
    for _ in range(max_steps):
        reachable = _reachable_markers(tip, remaining, shape, template_name)
        if not reachable:
            break
        assert len(reachable) == 1, "expected exactly one reachable direction at each step"
        move_direction = next(iter(reachable))
        _dist, target = reachable[move_direction][0]
        consumed.append(target)
        remaining.remove(target)
        tip = target
        current_direction = move_direction

    return tip, current_direction, consumed, remaining


def _paint_halo(grid, center):
    h, w = grid.shape
    r, c = center
    r0 = max(0, r - 1)
    r1 = min(h, r + 2)
    c0 = max(0, c - 1)
    c1 = min(w, c + 2)
    grid[r0:r1, c0:c1] = BLUE


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Consume uniquely aligned grey markers one by one and leave a blue 3x3 trail at every consumed stop.

    Input structure:
    - The background is black.
    - There is one 3- or 4-cell red directional sprite and 3-5 grey singleton markers.
    - The red sprite is a small 4-connected template with a uniquely recoverable tip and facing.
    - Blue is absent from the input.

    Rule:
    - Recover the sprite's tip pixel and facing from its shape.
    - From the current tip, inspect the four axis-aligned half-lines; exactly one direction has reachable grey markers.
    - Move the sprite so its tip lands on the nearest reachable marker in that direction, consume that marker, and update the facing to the move direction.
    - At each consumed marker position, paint a blue 3x3 halo clipped by the grid boundary.
    - Repeat until no reachable marker remains.
    - Output only the accumulated blue halos plus the red sprite at its final position, with red overwriting blue where they overlap.

    Key insight:
    - The grey markers are not all used at once: straight runs require nearest-first consumption, and the overlapping blue halos reveal the full ordered path while the final red sprite identifies the endpoint.
    """
    grid = np.asarray(input_grid, dtype=int)
    assert grid.ndim == 2
    template_name, initial_direction, initial_tip = _identify_sprite(grid)

    markers = {tuple(map(int, rc)) for rc in np.argwhere(grid == GREY)}
    assert 3 <= len(markers) <= 5, "expected 3-5 grey markers"
    assert not np.any(grid == BLUE), "blue should be absent from the input"

    final_tip, final_direction, consumed, _remaining = _simulate_path(
        grid.shape, template_name, initial_tip, initial_direction, markers
    )

    out = np.zeros_like(grid)
    for center in consumed:
        _paint_halo(out, center)
    for r, c in _sprite_cells(template_name, final_direction, final_tip):
        out[r, c] = RED
    return out


def _segment_length_patterns(total_steps):
    patterns = []
    for first in range(1, total_steps):
        second = total_steps - first
        if max(first, second) > 1:
            patterns.append((first, second))
    for first in range(1, total_steps - 1):
        for second in range(1, total_steps - first):
            third = total_steps - first - second
            if third < 1:
                continue
            if max(first, second, third) > 1:
                patterns.append((first, second, third))
    return patterns


SEGMENT_PATTERNS = {steps: _segment_length_patterns(steps) for steps in range(3, 6)}


def _turn_left(direction):
    return (direction - 1) % 4


def _turn_right(direction):
    return (direction + 1) % 4


def _build_step_directions(rng, total_steps):
    pattern = SEGMENT_PATTERNS[total_steps][int(rng.integers(len(SEGMENT_PATTERNS[total_steps])))]
    first_direction = int(rng.integers(4))
    segment_directions = [first_direction]
    for _ in range(1, len(pattern)):
        prev = segment_directions[-1]
        segment_directions.append(int(rng.choice([_turn_left(prev), _turn_right(prev)])))
    step_directions = []
    for seg_len, seg_dir in zip(pattern, segment_directions):
        step_directions.extend([seg_dir] * seg_len)
    assert len(step_directions) == total_steps
    return pattern, step_directions


def _build_points(step_directions, distances):
    points = [(0, 0)]
    for direction, dist in zip(step_directions, distances):
        dr, dc = DIRS4[direction]
        pr, pc = points[-1]
        points.append((pr + dr * dist, pc + dc * dist))
    return points


def _semantic_checks(points, step_directions):
    used_nearest = False
    for idx, move_direction in enumerate(step_directions):
        state = points[idx]
        future = points[idx + 1 :]
        reachable = {}
        for direction in (UP, RIGHT, DOWN, LEFT):
            candidates = []
            for marker in future:
                dist = _forward_distance(state, direction, marker)
                if dist is not None:
                    candidates.append((dist, marker))
            if candidates:
                candidates.sort()
                reachable[direction] = candidates
        if set(reachable) != {move_direction}:
            return False
        if reachable[move_direction][0][1] != points[idx + 1]:
            return False
        if len(reachable[move_direction]) > 1:
            used_nearest = True
    has_turn = any(step_directions[i] != step_directions[i - 1] for i in range(1, len(step_directions)))
    return has_turn and used_nearest


def _state_orientations(step_directions):
    return [step_directions[0]] + list(step_directions)


def _states_bbox(template_name, points, orientations):
    rows = []
    cols = []
    for tip, direction in zip(points, orientations):
        for r, c in _sprite_cells(template_name, direction, tip):
            rows.append(r)
            cols.append(c)
    return min(rows), max(rows), min(cols), max(cols)


def _choose_layout(rng, template_name, points, step_directions):
    orientations = _state_orientations(step_directions)
    shape_min_r, shape_max_r, shape_min_c, shape_max_c = _states_bbox(template_name, points, orientations)
    marker_rows = [r for r, _ in points[1:]]
    marker_cols = [c for _, c in points[1:]]

    clip_sides = []
    if min(marker_rows) == shape_min_r:
        clip_sides.append("top")
    if max(marker_rows) == shape_max_r:
        clip_sides.append("bottom")
    if min(marker_cols) == shape_min_c:
        clip_sides.append("left")
    if max(marker_cols) == shape_max_c:
        clip_sides.append("right")

    base_h = shape_max_r - shape_min_r + 1
    base_w = shape_max_c - shape_min_c + 1

    for _ in range(80):
        top = int(rng.integers(1, 5))
        bottom = int(rng.integers(1, 5))
        left = int(rng.integers(1, 5))
        right = int(rng.integers(1, 5))

        if clip_sides and rng.random() < 0.4:
            clip_side = str(rng.choice(clip_sides))
            if clip_side == "top":
                top = 0
            elif clip_side == "bottom":
                bottom = 0
            elif clip_side == "left":
                left = 0
            else:
                right = 0

        h = base_h + top + bottom
        w = base_w + left + right
        if not (11 <= h <= 22 and 11 <= w <= 22):
            continue

        shift_r = top - shape_min_r
        shift_c = left - shape_min_c
        shifted_points = [(r + shift_r, c + shift_c) for r, c in points]
        if any(not _sprite_fits((h, w), template_name, direction, tip) for tip, direction in zip(shifted_points, orientations)):
            continue
        return (h, w), shifted_points
    return None


def _render_input(shape, template_name, initial_direction, initial_tip, markers):
    grid = np.zeros(shape, dtype=int)
    for r, c in markers:
        grid[r, c] = GREY
    for r, c in _sprite_cells(template_name, initial_direction, initial_tip):
        assert grid[r, c] == BLACK
        grid[r, c] = RED
    return grid


def _render_expected_output(shape, template_name, final_tip, final_direction, consumed):
    out = np.zeros(shape, dtype=int)
    for center in consumed:
        _paint_halo(out, center)
    for r, c in _sprite_cells(template_name, final_direction, final_tip):
        out[r, c] = RED
    return out


def _build_case(rng):
    template_name = str(rng.choice(list(TEMPLATES)))
    total_steps = int(rng.integers(3, 6))
    _pattern, step_directions = _build_step_directions(rng, total_steps)
    distances = [int(rng.choice([2, 2, 3, 3, 4, 5])) for _ in range(total_steps)]
    points = _build_points(step_directions, distances)

    if len(set(points)) != len(points):
        return None
    if not _semantic_checks(points, step_directions):
        return None

    layout = _choose_layout(rng, template_name, points, step_directions)
    if layout is None:
        return None
    shape, shifted_points = layout

    initial_tip = shifted_points[0]
    initial_direction = step_directions[0]
    markers = shifted_points[1:]
    if len(set(markers)) != len(markers):
        return None

    grid = _render_input(shape, template_name, initial_direction, initial_tip, markers)
    final_tip, final_direction, consumed, remaining = _simulate_path(
        shape, template_name, initial_tip, initial_direction, markers
    )

    if remaining:
        return None
    if consumed != markers:
        return None

    expected = _render_expected_output(shape, template_name, final_tip, final_direction, consumed)
    return {
        "input": grid,
        "output": expected,
        "template_name": template_name,
        "initial_tip": initial_tip,
        "initial_direction": initial_direction,
        "points": shifted_points,
        "step_directions": step_directions,
        "consumed": consumed,
    }


def _validate_case(case):
    grid = case["input"]
    out = case["output"]
    markers = case["points"][1:]
    step_directions = case["step_directions"]

    assert isinstance(grid, np.ndarray) and isinstance(out, np.ndarray)
    assert grid.ndim == 2 and out.ndim == 2
    assert max(grid.shape) <= 30 and max(out.shape) <= 30
    assert 3 <= len(markers) <= 5
    assert _semantic_checks(case["points"], step_directions)
    assert not np.any(grid == BLUE)
    assert np.count_nonzero(grid == RED) in (3, 4)
    assert np.count_nonzero(grid == GREY) == len(markers)
    assert np.array_equal(main(grid), out)
    assert np.any(out == BLUE), "trail must remain visible"
    assert np.any(out == RED), "final sprite must remain visible"


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(400):
        case = _build_case(rng)
        if case is None:
            continue
        _validate_case(case)
        return case["input"]
    raise RuntimeError("failed to generate a valid input")
