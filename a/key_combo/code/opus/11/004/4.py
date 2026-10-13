from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
START_COLOR = 2
CORRIDOR_COLORS = {(0, 0): 3, (0, 1): 4, (1, 0): 6, (1, 1): 9}
MARKER_COLOR = 7
TARGET_COLOR = 8
BLOCK_SIZE = 2

GRID_ROW_RANGE = (18, 28)
GRID_COL_RANGE = (18, 29)
ROW_POS_PADDING = 2
COL_POS_PADDING = 1
MAX_GEN_ATTEMPTS = 500
PATH_START_TRIES = 120
TURN_BIAS = 0.75
H_GAPS = tuple(range(1, 9))
V_GAPS = tuple(range(2, 9))
STEP_CHOICES = (3, 4, 5)
TOTAL_TARGET_MIN = 4
TOTAL_TARGET_MAX = 6


def _bbox_from_pos(pos: tuple[int, int]) -> tuple[int, int, int, int]:
    r, c = pos
    return r, c, r + BLOCK_SIZE - 1, c + BLOCK_SIZE - 1


def _marker_bbox(pos: tuple[int, int]) -> tuple[int, int, int, int]:
    r, c = pos
    return r - 2, c, r - 1, c + 1


def _bbox_h(bbox: tuple[int, int, int, int]) -> int:
    return bbox[2] - bbox[0] + 1


def _bbox_w(bbox: tuple[int, int, int, int]) -> int:
    return bbox[3] - bbox[1] + 1


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _blocks_have_gap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    ar0, ac0, ar1, ac1 = _bbox_from_pos(a)
    br0, bc0, br1, bc1 = _bbox_from_pos(b)
    row_gap = max(br0 - ar1 - 1, ar0 - br1 - 1)
    col_gap = max(bc0 - ac1 - 1, ac0 - bc1 - 1)
    return row_gap >= 1 or col_gap >= 1


def _far_from_all(pos: tuple[int, int], positions: list[tuple[int, int]]) -> bool:
    return all(_blocks_have_gap(pos, other) for other in positions)


def _axis(a: tuple[int, int], b: tuple[int, int]) -> str | None:
    if a[0] == b[0]:
        return "h"
    if a[1] == b[1]:
        return "v"
    return None


def _gap_len(a: tuple[int, int], b: tuple[int, int]) -> int | None:
    axis = _axis(a, b)
    if axis == "h":
        return abs(a[1] - b[1]) - BLOCK_SIZE
    if axis == "v":
        return abs(a[0] - b[0]) - BLOCK_SIZE
    return None


def _corridor_rect(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int, int, int] | None, str | None]:
    axis = _axis(a, b)
    if axis == "h":
        r = a[0]
        c0 = min(a[1], b[1]) + BLOCK_SIZE
        c1 = max(a[1], b[1]) - 1
        if c0 <= c1:
            return (r, c0, r + BLOCK_SIZE - 1, c1), axis
        return None, axis
    if axis == "v":
        c = a[1]
        r0 = min(a[0], b[0]) + BLOCK_SIZE
        r1 = max(a[0], b[0]) - 1
        if r0 <= r1:
            return (r0, c, r1, c + BLOCK_SIZE - 1), axis
        return None, axis
    return None, None


def _corridor_clear(a: tuple[int, int], b: tuple[int, int], positions: list[tuple[int, int]]) -> bool:
    rect, axis = _corridor_rect(a, b)
    if axis is None:
        return False
    if rect is None:
        return True
    for pos in positions:
        if pos == a or pos == b:
            continue
        if _rects_overlap(rect, _bbox_from_pos(pos)):
            return False
    return True


def _visible_step(a: tuple[int, int], b: tuple[int, int], positions: list[tuple[int, int]]) -> tuple[str, int] | None:
    axis = _axis(a, b)
    if axis is None or not _corridor_clear(a, b, positions):
        return None
    return axis, int(_gap_len(a, b))


def _paint_corridor(out: np.ndarray, a: tuple[int, int], b: tuple[int, int], color: int) -> None:
    rect, axis = _corridor_rect(a, b)
    if axis is None or rect is None:
        return
    r0, c0, r1, c1 = rect
    out[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _component_dicts(grid: np.ndarray, color: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            comps.append({"color": int(color), "coords": arr, "bbox": (r0, c0, r1, c1), "area": int(arr.shape[0])})
    return comps


def _extract_scene(grid: np.ndarray) -> tuple[tuple[int, int], list[tuple[int, int]], dict[tuple[int, int], tuple[int, int]]]:
    starts = []
    targets = []
    for comp in _component_dicts(grid, START_COLOR):
        if comp["area"] == 4 and _bbox_h(comp["bbox"]) == 2 and _bbox_w(comp["bbox"]) == 2:
            starts.append((comp["bbox"][0], comp["bbox"][1]))
    for comp in _component_dicts(grid, TARGET_COLOR):
        if comp["area"] == 4 and _bbox_h(comp["bbox"]) == 2 and _bbox_w(comp["bbox"]) == 2:
            targets.append((comp["bbox"][0], comp["bbox"][1]))

    assert len(starts) == 1, "expected exactly one red 2x2 start"
    assert 4 <= len(targets) <= 6, "expected 4-6 teal 2x2 targets"

    markers = []
    for comp in _component_dicts(grid, MARKER_COLOR):
        if comp["area"] == 3 and _bbox_h(comp["bbox"]) == 2 and _bbox_w(comp["bbox"]) == 2:
            markers.append(comp)
    assert len(markers) == len(targets), "expected one orange L marker per target"

    marker_by_target: dict[tuple[int, int], tuple[int, int]] = {}
    used = set()
    for pos in targets:
        tr0, tc0, _tr1, tc1 = _bbox_from_pos(pos)
        candidates = []
        for idx, marker in enumerate(markers):
            mr0, mc0, mr1, mc1 = marker["bbox"]
            if mr1 + 1 == tr0 and max(tc0, mc0) <= min(tc1, mc1):
                candidates.append((idx, marker))
        assert len(candidates) == 1, "each target must have one directly-above marker"
        idx, marker = candidates[0]
        assert idx not in used, "marker assignment must be unique"
        used.add(idx)
        patch = grid[marker["bbox"][0] : marker["bbox"][2] + 1, marker["bbox"][1] : marker["bbox"][3] + 1]
        missing = np.argwhere(patch == BG)
        assert missing.shape == (1, 2), "marker must be a 2x2 L triomino"
        marker_by_target[pos] = (int(missing[0, 0]), int(missing[0, 1]))
    return starts[0], sorted(targets), marker_by_target


def _walk(current: tuple[int, int], targets: list[tuple[int, int]]) -> list[tuple[tuple[int, int], str, int]]:
    all_positions = [current] + list(targets)
    remaining = list(targets)
    steps = []
    while True:
        candidates = []
        for pos in remaining:
            step = _visible_step(current, pos, all_positions)
            if step is None:
                continue
            axis, gap = step
            candidates.append((gap, axis, pos))
        if not candidates:
            break
        best_gap = min(item[0] for item in candidates)
        nearest = [item for item in candidates if item[0] == best_gap]
        assert len(nearest) == 1, "nearest visible target must be unique"
        gap, axis, pos = nearest[0]
        steps.append((pos, axis, gap))
        remaining.remove(pos)
        current = pos
    return steps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Walk greedily through visible cyan stations and color each corridor by the destination station's L-marker orientation.

    Input structure:
    - The grid has a black background, one red 2x2 start block, and 4-6 cyan 2x2 target blocks.
    - Each cyan block has a directly-above orange 2x2 L-triomino marker whose missing corner encodes a corridor color.
    - Target blocks may share a full two-row band or two-column band, creating possible visible corridors between block footprints.

    Rule:
    - Start at the red block and repeatedly choose the unique nearest unvisited cyan block that is visible in the same two-row or two-column band with no other block between the two 2x2 footprints.
    - Read the chosen target's marker orientation: missing upper-left -> green, upper-right -> yellow, lower-left -> pink, lower-right -> maroon.
    - Paint the full 2-cell-wide background gap between the current block and that target with the decoded color, then erase that target's orange marker to black.
    - Continue from the reached target until no unvisited visible target remains; preserve all start/target blocks and leave never-visited markers unchanged.

    Key insight:
    - The sight graph between 2x2 stations determines which station is reached next, while the tiny L-marker above that destination alone determines the color of the corridor that gets drawn for that step.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    start, targets, marker_by_target = _extract_scene(g)
    out = g.copy()

    remaining = list(targets)
    all_positions = [start] + list(targets)
    current = start
    while True:
        candidates = []
        for pos in remaining:
            step = _visible_step(current, pos, all_positions)
            if step is None:
                continue
            axis, gap = step
            candidates.append((gap, axis, pos))
        if not candidates:
            break

        best_gap = min(item[0] for item in candidates)
        nearest = [item for item in candidates if item[0] == best_gap]
        assert len(nearest) == 1, "expected a unique nearest visible target"
        _gap, _axis, target = nearest[0]
        _paint_corridor(out, current, target, CORRIDOR_COLORS[marker_by_target[target]])

        mr0, mc0, mr1, mc1 = _marker_bbox(target)
        out[mr0 : mr1 + 1, mc0 : mc1 + 1][out[mr0 : mr1 + 1, mc0 : mc1 + 1] == MARKER_COLOR] = BG
        remaining.remove(target)
        current = target

    return out


def _row_range(rows: int) -> range:
    return range(ROW_POS_PADDING, rows - BLOCK_SIZE)


def _col_range(cols: int) -> range:
    return range(COL_POS_PADDING, cols - BLOCK_SIZE)


def _in_bounds(pos: tuple[int, int], rows: int, cols: int) -> bool:
    r, c = pos
    return ROW_POS_PADDING <= r <= rows - BLOCK_SIZE - 1 and COL_POS_PADDING <= c <= cols - BLOCK_SIZE - 1


def _markers_clear(targets: list[tuple[int, int]], start: tuple[int, int] | None = None) -> bool:
    block_boxes = [_bbox_from_pos(pos) for pos in targets]
    if start is not None:
        block_boxes.append(_bbox_from_pos(start))
    marker_boxes = [_marker_bbox(pos) for pos in targets]
    for box in marker_boxes:
        if box[0] < 0:
            return False
        for block in block_boxes:
            if _rects_overlap(box, block):
                return False
    for i, box_a in enumerate(marker_boxes):
        for box_b in marker_boxes[i + 1 :]:
            if _rects_overlap(box_a, box_b):
                return False
    return True


def _walk_is_forced(path: list[tuple[int, int]], distractors: list[tuple[int, int]]) -> bool:
    positions = list(path) + list(distractors)
    for i, current in enumerate(path[:-1]):
        remaining = list(path[i + 1 :]) + list(distractors)
        candidates = []
        for pos in remaining:
            step = _visible_step(current, pos, positions)
            if step is None:
                continue
            _axis_name, gap = step
            candidates.append((gap, pos))
        if not candidates:
            return False
        best_gap = min(item[0] for item in candidates)
        nearest = [pos for gap, pos in candidates if gap == best_gap]
        if len(nearest) != 1 or nearest[0] != path[i + 1]:
            return False
    endpoint = path[-1]
    for pos in distractors:
        if _visible_step(endpoint, pos, positions) is not None:
            return False
    return True


def _turn_count(path: list[tuple[int, int]]) -> int:
    turns = 0
    prev = None
    for a, b in zip(path, path[1:]):
        cur = _axis(a, b)
        if prev is not None and cur != prev:
            turns += 1
        prev = cur
    return turns


def _shuffled(items) -> list:
    items = list(items)
    random.shuffle(items)
    return items


def _next_options(
    rows: int,
    cols: int,
    path: list[tuple[int, int]],
    total_steps: int,
    target_turns: int,
) -> list[tuple[str, tuple[int, int]]]:
    current = path[-1]
    prev_axis = _axis(path[-2], path[-1]) if len(path) >= 2 else None
    turns_so_far = _turn_count(path)
    moves_done = len(path) - 1
    moves_left_after = total_steps - (moves_done + 1)
    options = []

    for axis_name in ("h", "v"):
        gaps = H_GAPS if axis_name == "h" else V_GAPS
        new_turns = turns_so_far + int(prev_axis is not None and axis_name != prev_axis)
        if not (new_turns <= target_turns <= new_turns + moves_left_after):
            continue
        for sign in (-1, 1):
            for gap in gaps:
                offset = sign * (gap + BLOCK_SIZE)
                if axis_name == "h":
                    candidate = (current[0], current[1] + offset)
                else:
                    candidate = (current[0] + offset, current[1])
                if not _in_bounds(candidate, rows, cols):
                    continue
                if not _far_from_all(candidate, path):
                    continue
                candidate_targets = path[1:] + [candidate]
                if not _markers_clear(candidate_targets, start=path[0]):
                    continue
                candidate_path = path + [candidate]
                if not _walk_is_forced(candidate_path, []):
                    continue
                options.append((axis_name, candidate))

    turn_options = [item for item in options if prev_axis is not None and item[0] != prev_axis]
    straight_options = [item for item in options if prev_axis is None or item[0] == prev_axis]
    if turn_options and turns_so_far < target_turns and random.random() < TURN_BIAS:
        return _shuffled(turn_options) + _shuffled(straight_options)
    return _shuffled(options)


def _extend_path(
    rows: int,
    cols: int,
    steps: int,
    target_turns: int,
    path: list[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    if len(path) == steps + 1:
        if _turn_count(path) == target_turns and _walk_is_forced(path, []):
            return path
        return None
    for _axis_name, candidate in _next_options(rows, cols, path, steps, target_turns):
        found = _extend_path(rows, cols, steps, target_turns, path + [candidate])
        if found is not None:
            return found
    return None


def _build_path(rows: int, cols: int, steps: int, target_turns: int) -> list[tuple[int, int]] | None:
    starts = [(r, c) for r in _row_range(rows) for c in _col_range(cols)]
    for start in _shuffled(starts)[:PATH_START_TRIES]:
        found = _extend_path(rows, cols, steps, target_turns, [start])
        if found is not None:
            return found
    return None


def _distractor_options(
    rows: int,
    cols: int,
    path: list[tuple[int, int]],
    distractors: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    positions = list(path) + list(distractors)
    options = []
    for r in _row_range(rows):
        for c in _col_range(cols):
            pos = (r, c)
            if pos in positions:
                continue
            if not _far_from_all(pos, positions):
                continue
            if not _markers_clear(path[1:] + distractors + [pos], start=path[0]):
                continue
            if _walk_is_forced(path, distractors + [pos]):
                options.append(pos)
    return options


def _sample_turns(steps: int) -> int:
    max_turns = min(steps - 1, 4)
    if max_turns <= 0:
        return 0
    if max_turns >= 2 and random.random() < 0.7:
        return random.randint(1, max_turns)
    return random.randint(0, max_turns)


def _sample_orientations(visited_count: int, distractor_count: int) -> list[tuple[int, int]]:
    missing_cells = list(CORRIDOR_COLORS)
    pool_size = random.randint(2, min(4, visited_count))
    pool = random.sample(missing_cells, pool_size)
    visited = pool.copy()
    while len(visited) < visited_count:
        visited.append(random.choice(pool))
    random.shuffle(visited)
    distractors = [random.choice(missing_cells) for _ in range(distractor_count)]
    return visited + distractors


def _marker_patch(missing: tuple[int, int]) -> np.ndarray:
    patch = np.full((2, 2), MARKER_COLOR, dtype=int)
    patch[missing] = BG
    return patch


def _render_instance(
    rows: int,
    cols: int,
    path: list[tuple[int, int]],
    distractors: list[tuple[int, int]],
    missing_by_target: dict[tuple[int, int], tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    grid = np.zeros((rows, cols), dtype=int)
    sr, sc = path[0]
    grid[sr : sr + 2, sc : sc + 2] = START_COLOR
    for pos in path[1:] + distractors:
        r, c = pos
        grid[r : r + 2, c : c + 2] = TARGET_COLOR
        mr0, mc0, mr1, mc1 = _marker_bbox(pos)
        patch = _marker_patch(missing_by_target[pos])
        grid[mr0 : mr1 + 1, mc0 : mc1 + 1] = patch

    expected = grid.copy()
    current = path[0]
    for pos in path[1:]:
        _paint_corridor(expected, current, pos, CORRIDOR_COLORS[missing_by_target[pos]])
        mr0, mc0, mr1, mc1 = _marker_bbox(pos)
        expected[mr0 : mr1 + 1, mc0 : mc1 + 1][expected[mr0 : mr1 + 1, mc0 : mc1 + 1] == MARKER_COLOR] = BG
        current = pos
    return grid, expected


def _semantic_checks(
    grid: np.ndarray,
    expected: np.ndarray,
    path: list[tuple[int, int]],
    distractors: list[tuple[int, int]],
    missing_by_target: dict[tuple[int, int], tuple[int, int]],
) -> bool:
    try:
        parsed_start, parsed_targets, parsed_markers = _extract_scene(grid)
    except Exception:
        return False
    if parsed_start != path[0]:
        return False
    if set(parsed_targets) != set(path[1:] + distractors):
        return False
    if parsed_markers != missing_by_target:
        return False

    if len(path) - 1 < 3:
        return False
    if len(parsed_targets) < TOTAL_TARGET_MIN or len(parsed_targets) > TOTAL_TARGET_MAX:
        return False
    if len(set(missing_by_target[pos] for pos in path[1:])) < 2:
        return False
    if not _markers_clear(path[1:] + distractors, start=path[0]):
        return False
    if not _walk_is_forced(path, distractors):
        return False

    walk = _walk(path[0], path[1:] + distractors)
    walked_targets = [pos for pos, _axis_name, _gap in walk]
    if walked_targets != path[1:]:
        return False

    try:
        actual = main(grid)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False
    return not np.array_equal(grid, expected)


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(*GRID_ROW_RANGE)
    cols = random.randint(*GRID_COL_RANGE)
    steps = random.choice(STEP_CHOICES)
    target_turns = _sample_turns(steps)
    path = _build_path(rows, cols, steps, target_turns)
    if path is None:
        return None

    min_distractors = max(0, TOTAL_TARGET_MIN - steps)
    max_distractors = min(TOTAL_TARGET_MAX - steps, 2)
    if max_distractors < min_distractors:
        return None
    distractor_count = random.randint(min_distractors, max_distractors)
    distractors: list[tuple[int, int]] = []
    for _ in range(distractor_count):
        options = _distractor_options(rows, cols, path, distractors)
        if not options:
            return None
        distractors.append(random.choice(options))

    orientations = _sample_orientations(len(path) - 1, len(distractors))
    target_order = path[1:] + distractors
    missing_by_target = {pos: missing for pos, missing in zip(target_order, orientations)}

    grid, expected = _render_instance(rows, cols, path, distractors, missing_by_target)
    if not _semantic_checks(grid, expected, path, distractors, missing_by_target):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
