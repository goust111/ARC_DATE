from __future__ import annotations

import numpy as np

from arc_utils import Color, bbox_h, bbox_w, find_components, get_bg, paste


Block = dict


def _blocks(g: np.ndarray, bg: int) -> tuple[Block, list[Block]]:
    objects = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    assert objects, "expected colored 2x2 blocks"

    starts = []
    targets = []
    for obj in objects:
        color = int(obj["color"])
        assert color in {Color.RED, Color.TEAL}, "expected only red and teal input blocks"
        assert int(obj["area"]) == 4, "each colored object must have four pixels"
        assert bbox_h(obj["bbox"]) == 2 and bbox_w(obj["bbox"]) == 2, "each colored object must be a 2x2 block"
        if color == Color.RED:
            starts.append(obj)
        else:
            targets.append(obj)

    assert len(starts) == 1, "expected exactly one red start block"
    assert targets, "expected at least one teal target block"
    return starts[0], targets


def _visible_step(g: np.ndarray, bg: int, a: Block, b: Block) -> tuple[str, int] | None:
    ar0, ac0, ar1, ac1 = a["bbox"]
    br0, bc0, br1, bc1 = b["bbox"]

    if ar0 == br0 and ar1 == br1:
        c0 = min(ac1, bc1) + 1
        c1 = max(ac0, bc0) - 1
        if c0 <= c1 and not np.all(g[ar0 : ar1 + 1, c0 : c1 + 1] == bg):
            return None
        return "h", abs(bc0 - ac0)

    if ac0 == bc0 and ac1 == bc1:
        r0 = min(ar1, br1) + 1
        r1 = max(ar0, br0) - 1
        if r0 <= r1 and not np.all(g[r0 : r1 + 1, ac0 : ac1 + 1] == bg):
            return None
        return "v", abs(br0 - ar0)

    return None


def _paint_corridor(out: np.ndarray, a: Block, b: Block, axis: str) -> None:
    ar0, ac0, ar1, ac1 = a["bbox"]
    br0, bc0, br1, bc1 = b["bbox"]

    if axis == "h":
        c0 = min(ac1, bc1) + 1
        c1 = max(ac0, bc0) - 1
        if c0 <= c1:
            out[ar0 : ar1 + 1, c0 : c1 + 1] = Color.ORANGE
        return

    assert axis == "v", "corridor axis must be horizontal or vertical"
    r0 = min(ar1, br1) + 1
    r1 = max(ar0, br0) - 1
    if r0 <= r1:
        out[r0 : r1 + 1, ac0 : ac1 + 1] = Color.ORANGE


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw orange corridors along a nearest-visible-block walk from the red 2x2 block through teal 2x2 blocks.

    Input structure
    - The background contains one red 2x2 start block and multiple teal 2x2 target blocks.
    - Blocks may align in shared two-row or two-column bands; non-background pixels between aligned blocks block visibility.

    Rule
    - Start at the red block.
    - Among unvisited teal blocks visible in the same horizontal or vertical band, choose the unique nearest one.
    - Paint the background gap between the current block and that teal block orange, then continue from that teal block.
    - Stop when no unvisited teal block is visible from the current block.

    Key insight
    - The output is not the full alignment graph; it is a greedy nearest-neighbor walk whose line of sight is blocked by any existing colored block.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    current, unvisited = _blocks(g, bg)
    out = g.copy()

    while True:
        candidates = []
        for block in unvisited:
            step = _visible_step(g, bg, current, block)
            if step is None:
                continue
            axis, distance = step
            candidates.append((distance, axis, block))

        if not candidates:
            break

        min_distance = min(distance for distance, _axis, _block in candidates)
        nearest = [candidate for candidate in candidates if candidate[0] == min_distance]
        assert len(nearest) == 1, "expected a unique nearest visible teal block"

        _distance, axis, block = nearest[0]
        _paint_corridor(out, current, block, axis)
        unvisited = [candidate for candidate in unvisited if candidate is not block]
        current = block

    return out

# --------------- generator (o5x) ---------------
# Canvas and block geometry.
GEN_ROW_RANGE = (16, 27)  # rng.integers high is exclusive.
GEN_COL_RANGE = (18, 29)
GEN_EDGE_PADDING = 1
GEN_BLOCK_SIZE = 2

# Walk length and turn sampling.
GEN_STEP_VALUES = np.arange(1, 10)
GEN_STEP_WEIGHTS = np.array([8, 9, 11, 12, 13, 13, 13, 11, 10], dtype=float)
GEN_STEP_PROBS = GEN_STEP_WEIGHTS / GEN_STEP_WEIGHTS.sum()
GEN_MAX_TURNS = 6
GEN_HIGH_TURN_MIN = 3
GEN_HIGH_TURN_PROB = 0.7
GEN_HIGH_TURN_LOW = 2
GEN_TURN_OPTION_PROB = 0.75

# Move construction and placement retries.
GEN_AXES, GEN_MOVE_SIGNS = ("h", "v"), (-1, 1)
GEN_GAP_RANGE = range(1, 11)
GEN_START_TRIES = 80
GEN_MAX_ATTEMPTS = 300

# Distractor counts and total teal-block budget.
GEN_MAX_TEAL_BLOCKS = 11
GEN_MAX_DISTRACTORS = 3
GEN_SINGLE_STEP_MIN_DISTRACTORS = 1

# Render colors.
GEN_START_COLOR, GEN_TARGET_COLOR, GEN_CORRIDOR_COLOR = Color.RED, Color.TEAL, Color.ORANGE


def _gen_bbox(pos):
    r, c = pos
    return r, c, r + GEN_BLOCK_SIZE - 1, c + GEN_BLOCK_SIZE - 1


def _gen_position_range(length):
    return range(GEN_EDGE_PADDING, length - GEN_EDGE_PADDING - GEN_BLOCK_SIZE + 1)


def _gen_rects_overlap(a, b):
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _gen_blocks_have_gap(a, b):
    ar0, ac0, ar1, ac1 = _gen_bbox(a)
    br0, bc0, br1, bc1 = _gen_bbox(b)
    row_gap = max(br0 - ar1 - 1, ar0 - br1 - 1)
    col_gap = max(bc0 - ac1 - 1, ac0 - bc1 - 1)
    return row_gap >= 1 or col_gap >= 1


def _gen_far_from_all(pos, positions):
    return all(_gen_blocks_have_gap(pos, other) for other in positions)


def _gen_axis(a, b):
    if a[0] == b[0]:
        return "h"
    if a[1] == b[1]:
        return "v"
    return None


def _gen_distance(a, b):
    axis = _gen_axis(a, b)
    if axis == "h":
        return abs(a[1] - b[1])
    if axis == "v":
        return abs(a[0] - b[0])
    return None


def _gen_corridor(a, b):
    axis = _gen_axis(a, b)
    if axis == "h":
        r = a[0]
        c0 = min(a[1], b[1]) + GEN_BLOCK_SIZE
        c1 = max(a[1], b[1]) - 1
        if c0 <= c1:
            return (r, c0, r + GEN_BLOCK_SIZE - 1, c1), axis
    if axis == "v":
        c = a[1]
        r0 = min(a[0], b[0]) + GEN_BLOCK_SIZE
        r1 = max(a[0], b[0]) - 1
        if r0 <= r1:
            return (r0, c, r1, c + GEN_BLOCK_SIZE - 1), axis
    return None, None


def _gen_corridor_clear(a, b, positions):
    rect, axis = _gen_corridor(a, b)
    if axis is None:
        return False
    for pos in positions:
        if pos == a or pos == b:
            continue
        if _gen_rects_overlap(rect, _gen_bbox(pos)):
            return False
    return True


def _gen_visible_axis(a, b, positions):
    if _gen_axis(a, b) is None:
        return None
    if not _gen_corridor_clear(a, b, positions):
        return None
    return _gen_axis(a, b)


def _gen_turn_count(path):
    turns = 0
    prev = None
    for a, b in zip(path, path[1:]):
        axis = _gen_axis(a, b)
        if prev is not None and axis != prev:
            turns += 1
        prev = axis
    return turns


def _gen_walk_is_forced(path, distractors):
    positions = list(path) + list(distractors)
    for i, current in enumerate(path[:-1]):
        candidates = []
        for block in list(path[i + 1 :]) + list(distractors):
            axis = _gen_visible_axis(current, block, positions)
            if axis is not None:
                candidates.append((_gen_distance(current, block), block))
        if not candidates:
            return False
        nearest_distance = min(distance for distance, _block in candidates)
        nearest = [block for distance, block in candidates if distance == nearest_distance]
        if len(nearest) != 1 or nearest[0] != path[i + 1]:
            return False

    endpoint = path[-1]
    return all(_gen_visible_axis(endpoint, block, positions) is None for block in distractors)


def _gen_in_bounds(pos, rows, cols):
    r, c = pos
    return (
        GEN_EDGE_PADDING <= r <= rows - GEN_EDGE_PADDING - GEN_BLOCK_SIZE
        and GEN_EDGE_PADDING <= c <= cols - GEN_EDGE_PADDING - GEN_BLOCK_SIZE
    )


def _gen_shuffled(items, rng):
    items = list(items)
    if not items:
        return items
    return [items[i] for i in rng.permutation(len(items))]


def _gen_next_options(rows, cols, path, target_turns, rng):
    current = path[-1]
    prev_axis = _gen_axis(path[-2], path[-1]) if len(path) >= 2 else None
    turns_so_far = _gen_turn_count(path)
    moves_done = len(path) - 1
    options = []
    desired_turns, total_steps = target_turns
    moves_left_after = total_steps - (moves_done + 1)

    for axis in GEN_AXES:
        new_turns = turns_so_far + int(prev_axis is not None and axis != prev_axis)
        for sign in GEN_MOVE_SIGNS:
            for gap in GEN_GAP_RANGE:
                offset = sign * (gap + GEN_BLOCK_SIZE)
                if axis == "h":
                    candidate = (current[0], current[1] + offset)
                else:
                    candidate = (current[0] + offset, current[1])
                if not _gen_in_bounds(candidate, rows, cols):
                    continue
                if not _gen_far_from_all(candidate, path):
                    continue
                candidate_path = path + [candidate]
                if not (new_turns <= desired_turns <= new_turns + moves_left_after):
                    continue
                if not _gen_walk_is_forced(candidate_path, []):
                    continue
                options.append((axis, candidate))

    turn_options = [item for item in options if prev_axis is not None and item[0] != prev_axis]
    straight_options = [item for item in options if prev_axis is None or item[0] == prev_axis]
    if turn_options and turns_so_far < desired_turns and rng.random() < GEN_TURN_OPTION_PROB:
        return _gen_shuffled(turn_options, rng) + _gen_shuffled(straight_options, rng)
    return _gen_shuffled(options, rng)


def _gen_extend_path(rows, cols, steps, target_turns, path, rng):
    if len(path) == steps + 1:
        if _gen_turn_count(path) == target_turns and _gen_walk_is_forced(path, []):
            return path
        return None

    for _axis, candidate in _gen_next_options(rows, cols, path, (target_turns, steps), rng):
        found = _gen_extend_path(rows, cols, steps, target_turns, path + [candidate], rng)
        if found is not None:
            return found
    return None


def _gen_build_path(rows, cols, steps, target_turns, rng):
    starts = [(r, c) for r in _gen_position_range(rows) for c in _gen_position_range(cols)]
    for start in _gen_shuffled(starts, rng)[:GEN_START_TRIES]:
        found = _gen_extend_path(rows, cols, steps, target_turns, [start], rng)
        if found is not None:
            return found
    return None


def _gen_distractor_options(rows, cols, path, distractors):
    positions = list(path) + list(distractors)
    endpoint = path[-1]
    options = []
    for r in _gen_position_range(rows):
        for c in _gen_position_range(cols):
            pos = (r, c)
            if pos in positions:
                continue
            if r == endpoint[0] or c == endpoint[1]:
                continue
            if not _gen_far_from_all(pos, positions):
                continue
            if _gen_walk_is_forced(path, distractors + [pos]):
                options.append(pos)
    return options


def _gen_place_block(grid, pos, color):
    r, c = pos
    paste(grid, np.full((GEN_BLOCK_SIZE, GEN_BLOCK_SIZE), color, dtype=int), r, c)


def _gen_render(rows, cols, path, distractors):
    grid = np.zeros((rows, cols), dtype=int)
    _gen_place_block(grid, path[0], GEN_START_COLOR)
    for pos in path[1:]:
        _gen_place_block(grid, pos, GEN_TARGET_COLOR)
    for pos in distractors:
        _gen_place_block(grid, pos, GEN_TARGET_COLOR)

    out = grid.copy()
    for a, b in zip(path, path[1:]):
        rect, axis = _gen_corridor(a, b)
        if axis is None:
            return None, None
        r0, c0, r1, c1 = rect
        out[r0 : r1 + 1, c0 : c1 + 1] = GEN_CORRIDOR_COLOR
    return grid, out


def _gen_sample_walk_shape(rng):
    """Return the target number of path steps and turns."""
    steps = int(rng.choice(GEN_STEP_VALUES, p=GEN_STEP_PROBS))
    max_turns = min(GEN_MAX_TURNS, steps - 1)
    if max_turns == 0:
        target_turns = 0
    elif max_turns >= GEN_HIGH_TURN_MIN and rng.random() < GEN_HIGH_TURN_PROB:
        target_turns = int(rng.integers(GEN_HIGH_TURN_LOW, max_turns + 1))
    else:
        target_turns = int(rng.integers(0, max_turns + 1))
    return steps, target_turns


def _gen_sample_distractor_count(steps, rng):
    min_distractors = GEN_SINGLE_STEP_MIN_DISTRACTORS if steps == 1 else 0
    max_distractors = min(GEN_MAX_DISTRACTORS, GEN_MAX_TEAL_BLOCKS - steps)
    if max_distractors < min_distractors:
        return None
    return int(rng.integers(min_distractors, max_distractors + 1))


def _gen_place_distractors(rows, cols, path, distractor_count, rng):
    """Choose distractors one at a time while preserving the forced walk."""
    distractors = []
    for _ in range(distractor_count):
        options = _gen_distractor_options(rows, cols, path, distractors)
        if not options:
            return None
        distractors.append(_gen_shuffled(options, rng)[0])
    return distractors


def _gen_passes_oracle(grid, expected):
    try:
        actual = main(grid)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    rows, cols = int(rng.integers(*GEN_ROW_RANGE)), int(rng.integers(*GEN_COL_RANGE))
    steps, target_turns = _gen_sample_walk_shape(rng)

    path = _gen_build_path(rows, cols, steps, target_turns, rng)
    if path is None:
        return None

    distractor_count = _gen_sample_distractor_count(steps, rng)
    if distractor_count is None:
        return None
    distractors = _gen_place_distractors(rows, cols, path, distractor_count, rng)
    if distractors is None:
        return None

    grid, expected = _gen_render(rows, cols, path, distractors)
    if grid is None or not _gen_passes_oracle(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 3490cc26 input")
