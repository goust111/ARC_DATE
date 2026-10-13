from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np


_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, bbox_h, bbox_w, find_components, get_bg, paste


Block = dict[str, object]


BG_COLOR = Color.BLACK
START_COLOR = Color.RED
BLOCK_SIZE = 2
TARGET_COLORS = [
    Color.BLUE,
    Color.GREEN,
    Color.YELLOW,
    Color.GREY,
    Color.PINK,
    Color.ORANGE,
    Color.TEAL,
    Color.MAROON,
]

# Generator controls.
GEN_ROW_RANGE = (14, 25)
GEN_COL_RANGE = (14, 25)
GEN_EDGE_PADDING = 1
GEN_GAP_RANGE = range(1, 9)
GEN_STEP_VALUES = np.arange(1, 7)
GEN_STEP_WEIGHTS = np.array([5, 8, 9, 8, 6, 4], dtype=float)
GEN_STEP_PROBS = GEN_STEP_WEIGHTS / GEN_STEP_WEIGHTS.sum()
GEN_MAX_DISTRACTORS = 3
GEN_MAX_ATTEMPTS = 500
GEN_START_TRIES = 80
GEN_TURN_OPTION_PROB = 0.72


def _extract_blocks(grid: np.ndarray) -> tuple[int, Block, list[Block]]:
    bg = int(get_bg(grid))
    objects = find_components(grid, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    assert objects, "expected colored 2x2 blocks"

    start: Block | None = None
    targets: list[Block] = []
    seen_colors: set[int] = set()
    for obj in objects:
        color = int(obj["color"])
        assert int(obj["area"]) == 4, "every colored object must be a 2x2 block"
        assert bbox_h(obj["bbox"]) == BLOCK_SIZE and bbox_w(obj["bbox"]) == BLOCK_SIZE
        if color == START_COLOR:
            assert start is None, "expected exactly one red start block"
            start = obj
            continue
        assert color != bg, "foreground object cannot use the background color"
        assert color not in seen_colors, "target colors must be distinct"
        seen_colors.add(color)
        targets.append(obj)

    assert start is not None, "expected exactly one red start block"
    assert targets, "expected at least one target block"
    return bg, start, targets


def _visible_step(grid: np.ndarray, bg: int, a: Block, b: Block) -> tuple[str, int] | None:
    ar0, ac0, ar1, ac1 = a["bbox"]
    br0, bc0, br1, bc1 = b["bbox"]

    if ar0 == br0 and ar1 == br1:
        c0 = min(ac1, bc1) + 1
        c1 = max(ac0, bc0) - 1
        if c0 <= c1 and not np.all(grid[ar0 : ar1 + 1, c0 : c1 + 1] == bg):
            return None
        return "h", abs(int(bc0) - int(ac0))

    if ac0 == bc0 and ac1 == bc1:
        r0 = min(ar1, br1) + 1
        r1 = max(ar0, br0) - 1
        if r0 <= r1 and not np.all(grid[r0 : r1 + 1, ac0 : ac1 + 1] == bg):
            return None
        return "v", abs(int(br0) - int(ar0))

    return None


def _walk_order(grid: np.ndarray) -> list[int]:
    bg, current, unvisited = _extract_blocks(grid)
    order: list[int] = []

    while True:
        candidates: list[tuple[int, Block]] = []
        for block in unvisited:
            step = _visible_step(grid, bg, current, block)
            if step is None:
                continue
            _axis, distance = step
            candidates.append((distance, block))

        if not candidates:
            break

        nearest_distance = min(distance for distance, _block in candidates)
        nearest = [block for distance, block in candidates if distance == nearest_distance]
        assert len(nearest) == 1, "expected a unique nearest visible target"

        next_block = nearest[0]
        order.append(int(next_block["color"]))
        unvisited = [block for block in unvisited if block is not next_block]
        current = next_block

    return order


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Read off the color itinerary of the greedy visible-block tour that starts from the red 2x2 block.

    Input structure:
    - A black background contains exactly one red 2x2 start block and several other 2x2 target blocks.
    - Every target block has a distinct color, and alignment only matters when two blocks share the same two-row band or two-column band.
    - Colored blocks lying between aligned blocks block line of sight through that horizontal or vertical gap.

    Rule:
    - Start from the red block and repeatedly look for unvisited target blocks visible in the current block's row band or column band.
    - Among all visible unvisited targets, move to the unique nearest one and record its color.
    - Stop when no unvisited target is visible, then output the recorded colors as a 1xN row.

    Key insight:
    - The answer is not a drawn path; it is the hidden greedy visitation order induced by alignment, occlusion, and nearest-choice tie avoidance.
    """
    grid = np.asarray(input_grid, dtype=int)
    assert grid.ndim == 2, "expected a 2D grid"
    order = _walk_order(grid)
    return np.asarray([order], dtype=int)


def _gen_bbox(pos: tuple[int, int]) -> tuple[int, int, int, int]:
    r, c = pos
    return int(r), int(c), int(r + BLOCK_SIZE - 1), int(c + BLOCK_SIZE - 1)


def _gen_position_range(length: int) -> range:
    return range(GEN_EDGE_PADDING, length - GEN_EDGE_PADDING - BLOCK_SIZE + 1)


def _gen_rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _gen_blocks_have_gap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    ar0, ac0, ar1, ac1 = _gen_bbox(a)
    br0, bc0, br1, bc1 = _gen_bbox(b)
    row_gap = max(br0 - ar1 - 1, ar0 - br1 - 1)
    col_gap = max(bc0 - ac1 - 1, ac0 - bc1 - 1)
    return row_gap >= 1 or col_gap >= 1


def _gen_far_from_all(pos: tuple[int, int], positions: list[tuple[int, int]]) -> bool:
    return all(_gen_blocks_have_gap(pos, other) for other in positions)


def _gen_axis(a: tuple[int, int], b: tuple[int, int]) -> str | None:
    if a[0] == b[0]:
        return "h"
    if a[1] == b[1]:
        return "v"
    return None


def _gen_distance(a: tuple[int, int], b: tuple[int, int]) -> int | None:
    axis = _gen_axis(a, b)
    if axis == "h":
        return abs(int(a[1]) - int(b[1]))
    if axis == "v":
        return abs(int(a[0]) - int(b[0]))
    return None


def _gen_corridor(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int, int, int] | None, str | None]:
    axis = _gen_axis(a, b)
    if axis == "h":
        r = a[0]
        c0 = min(a[1], b[1]) + BLOCK_SIZE
        c1 = max(a[1], b[1]) - 1
        if c0 <= c1:
            return (r, c0, r + BLOCK_SIZE - 1, c1), axis
    if axis == "v":
        c = a[1]
        r0 = min(a[0], b[0]) + BLOCK_SIZE
        r1 = max(a[0], b[0]) - 1
        if r0 <= r1:
            return (r0, c, r1, c + BLOCK_SIZE - 1), axis
    return None, None


def _gen_corridor_clear(a: tuple[int, int], b: tuple[int, int], positions: list[tuple[int, int]]) -> bool:
    rect, axis = _gen_corridor(a, b)
    if axis is None or rect is None:
        return False
    for pos in positions:
        if pos == a or pos == b:
            continue
        if _gen_rects_overlap(rect, _gen_bbox(pos)):
            return False
    return True


def _gen_visible_axis(a: tuple[int, int], b: tuple[int, int], positions: list[tuple[int, int]]) -> str | None:
    axis = _gen_axis(a, b)
    if axis is None:
        return None
    if not _gen_corridor_clear(a, b, positions):
        return None
    return axis


def _gen_turn_count(path: list[tuple[int, int]]) -> int:
    turns = 0
    prev_axis = None
    for a, b in zip(path, path[1:]):
        axis = _gen_axis(a, b)
        if prev_axis is not None and axis != prev_axis:
            turns += 1
        prev_axis = axis
    return turns


def _gen_walk_is_forced(path: list[tuple[int, int]], distractors: list[tuple[int, int]]) -> bool:
    positions = list(path) + list(distractors)
    for i, current in enumerate(path[:-1]):
        candidates: list[tuple[int, tuple[int, int]]] = []
        for block in list(path[i + 1 :]) + list(distractors):
            axis = _gen_visible_axis(current, block, positions)
            if axis is None:
                continue
            distance = _gen_distance(current, block)
            assert distance is not None
            candidates.append((distance, block))
        if not candidates:
            return False
        nearest_distance = min(distance for distance, _block in candidates)
        nearest = [block for distance, block in candidates if distance == nearest_distance]
        if len(nearest) != 1 or nearest[0] != path[i + 1]:
            return False

    endpoint = path[-1]
    return all(_gen_visible_axis(endpoint, block, positions) is None for block in distractors)


def _gen_in_bounds(pos: tuple[int, int], rows: int, cols: int) -> bool:
    r, c = pos
    return (
        GEN_EDGE_PADDING <= r <= rows - GEN_EDGE_PADDING - BLOCK_SIZE
        and GEN_EDGE_PADDING <= c <= cols - GEN_EDGE_PADDING - BLOCK_SIZE
    )


def _gen_shuffled(items: list[tuple[int, int]]) -> list[tuple[int, int]]:
    items = list(items)
    random.shuffle(items)
    return items


def _gen_next_options(
    rows: int,
    cols: int,
    path: list[tuple[int, int]],
    desired_turns: int,
    total_steps: int,
) -> list[tuple[str, tuple[int, int]]]:
    current = path[-1]
    prev_axis = _gen_axis(path[-2], path[-1]) if len(path) >= 2 else None
    turns_so_far = _gen_turn_count(path)
    moves_done = len(path) - 1
    moves_left_after = total_steps - (moves_done + 1)
    options: list[tuple[str, tuple[int, int]]] = []

    for axis in ("h", "v"):
        new_turns = turns_so_far + int(prev_axis is not None and axis != prev_axis)
        if not (new_turns <= desired_turns <= new_turns + moves_left_after):
            continue
        for sign in (-1, 1):
            for gap in GEN_GAP_RANGE:
                offset = sign * (gap + BLOCK_SIZE)
                candidate = (current[0], current[1] + offset) if axis == "h" else (current[0] + offset, current[1])
                if not _gen_in_bounds(candidate, rows, cols):
                    continue
                if not _gen_far_from_all(candidate, path):
                    continue
                candidate_path = path + [candidate]
                if not _gen_walk_is_forced(candidate_path, []):
                    continue
                options.append((axis, candidate))

    turn_options = [item for item in options if prev_axis is not None and item[0] != prev_axis]
    straight_options = [item for item in options if prev_axis is None or item[0] == prev_axis]
    if turn_options and turns_so_far < desired_turns and random.random() < GEN_TURN_OPTION_PROB:
        return _gen_shuffled(turn_options) + _gen_shuffled(straight_options)
    return _gen_shuffled(options)


def _gen_extend_path(
    rows: int,
    cols: int,
    total_steps: int,
    desired_turns: int,
    path: list[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    if len(path) == total_steps + 1:
        if _gen_turn_count(path) == desired_turns and _gen_walk_is_forced(path, []):
            return path
        return None

    for _axis, candidate in _gen_next_options(rows, cols, path, desired_turns, total_steps):
        found = _gen_extend_path(rows, cols, total_steps, desired_turns, path + [candidate])
        if found is not None:
            return found
    return None


def _gen_build_path(rows: int, cols: int, total_steps: int, desired_turns: int) -> list[tuple[int, int]] | None:
    starts = [(r, c) for r in _gen_position_range(rows) for c in _gen_position_range(cols)]
    starts = _gen_shuffled(starts)
    for start in starts[:GEN_START_TRIES]:
        found = _gen_extend_path(rows, cols, total_steps, desired_turns, [start])
        if found is not None:
            return found
    return None


def _gen_distractor_options(
    rows: int,
    cols: int,
    path: list[tuple[int, int]],
    distractors: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    positions = list(path) + list(distractors)
    endpoint = path[-1]
    options: list[tuple[int, int]] = []
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


def _gen_distractor_kind(
    pos: tuple[int, int],
    path: list[tuple[int, int]],
    distractors: list[tuple[int, int]],
) -> str:
    positions = list(path) + list(distractors) + [pos]
    visible = False
    aligned = False
    for current in path:
        if _gen_axis(current, pos) is not None:
            aligned = True
        if _gen_visible_axis(current, pos, positions) is not None:
            visible = True
            break
    if visible:
        return "visible"
    if aligned:
        return "blocked"
    return "hidden"


def _gen_place_distractors(
    rows: int,
    cols: int,
    path: list[tuple[int, int]],
    distractor_count: int,
) -> list[tuple[int, int]] | None:
    distractors: list[tuple[int, int]] = []
    for _ in range(distractor_count):
        options = _gen_distractor_options(rows, cols, path, distractors)
        if not options:
            return None
        visible = [pos for pos in options if _gen_distractor_kind(pos, path, distractors) == "visible"]
        blocked = [pos for pos in options if _gen_distractor_kind(pos, path, distractors) == "blocked"]
        hidden = [pos for pos in options if _gen_distractor_kind(pos, path, distractors) == "hidden"]

        roll = random.random()
        if visible and roll < 0.45:
            pool = visible
        elif blocked and roll < 0.8:
            pool = blocked
        elif hidden:
            pool = hidden
        else:
            pool = options
        distractors.append(random.choice(pool))
    return distractors


def _gen_sample_walk_shape() -> tuple[int, int]:
    total_steps = int(np.random.choice(GEN_STEP_VALUES, p=GEN_STEP_PROBS))
    max_turns = total_steps - 1
    desired_turns = 0 if max_turns <= 0 else random.randint(0, max_turns)
    if total_steps >= 3 and random.random() < 0.55:
        desired_turns = random.randint(min(1, max_turns), max_turns)
    return total_steps, desired_turns


def _gen_sample_distractor_count(total_steps: int) -> int:
    max_distractors = min(GEN_MAX_DISTRACTORS, len(TARGET_COLORS) - total_steps)
    assert max_distractors >= 0
    if max_distractors == 0:
        return 0
    weights = [1] + [3] * max_distractors
    return random.choices(range(max_distractors + 1), weights=weights, k=1)[0]


def _gen_place_block(grid: np.ndarray, pos: tuple[int, int], color: int) -> None:
    patch = np.full((BLOCK_SIZE, BLOCK_SIZE), int(color), dtype=int)
    paste(grid, patch, pos[0], pos[1])


def _gen_assign_colors(path: list[tuple[int, int]], distractors: list[tuple[int, int]]) -> dict[tuple[int, int], int]:
    count = len(path) - 1 + len(distractors)
    colors = random.sample(TARGET_COLORS, count)
    assignments: dict[tuple[int, int], int] = {}
    cursor = 0
    for pos in path[1:]:
        assignments[pos] = colors[cursor]
        cursor += 1
    for pos in distractors:
        assignments[pos] = colors[cursor]
        cursor += 1
    return assignments


def _gen_render(
    rows: int,
    cols: int,
    path: list[tuple[int, int]],
    distractors: list[tuple[int, int]],
    color_map: dict[tuple[int, int], int],
) -> tuple[np.ndarray, np.ndarray]:
    grid = np.full((rows, cols), BG_COLOR, dtype=int)
    _gen_place_block(grid, path[0], START_COLOR)
    for pos in path[1:]:
        _gen_place_block(grid, pos, color_map[pos])
    for pos in distractors:
        _gen_place_block(grid, pos, color_map[pos])
    expected = np.asarray([[color_map[pos] for pos in path[1:]]], dtype=int)
    return grid, expected


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(*GEN_ROW_RANGE)
    cols = random.randint(*GEN_COL_RANGE)
    total_steps, desired_turns = _gen_sample_walk_shape()

    path = _gen_build_path(rows, cols, total_steps, desired_turns)
    if path is None:
        return None

    distractor_count = _gen_sample_distractor_count(total_steps)
    distractors = _gen_place_distractors(rows, cols, path, distractor_count)
    if distractors is None:
        return None

    color_map = _gen_assign_colors(path, distractors)
    grid, expected = _gen_render(rows, cols, path, distractors, color_map)

    try:
        actual = main(grid)
    except Exception:
        return None

    if not np.array_equal(actual, expected):
        return None
    if not _gen_walk_is_forced(path, distractors):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid touring scene")
