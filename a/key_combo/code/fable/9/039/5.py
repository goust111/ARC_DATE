from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
RED = 2
ORANGE = 7
TEAL = 8
PRISM_COLORS = {ORANGE, TEAL}

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left
CW = 1
CCW = -1

GRID_H_RANGE = (12, 22)
GRID_W_RANGE = (12, 22)
TURN_COUNT_RANGE = (2, 6)
MAX_GENERATION_ATTEMPTS = 400
MIN_SEGMENT_LEN = 2
MIN_FINAL_SEGMENT_LEN = 2
MAX_DISTRACTORS = 3


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _component_cells(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _detect_start(grid: np.ndarray) -> tuple[tuple[int, int], int]:
    starts = np.argwhere(grid == RED)
    assert len(starts) == 1, "expected exactly one red start pixel"
    r, c = map(int, starts[0])
    h, w = grid.shape
    on_top = r == 0
    on_bottom = r == h - 1
    on_left = c == 0
    on_right = c == w - 1
    assert on_top or on_bottom or on_left or on_right, "start pixel must lie on the grid edge"
    assert sum((on_top, on_bottom, on_left, on_right)) == 1, "start pixel must not be on a corner"
    if on_top:
        return (r, c), 2
    if on_right:
        return (r, c), 3
    if on_bottom:
        return (r, c), 0
    return (r, c), 1


def _detect_prisms(grid: np.ndarray) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], int]]:
    cell_to_color: dict[tuple[int, int], int] = {}
    cell_to_block: dict[tuple[int, int], int] = {}
    block_id = 0
    for color in (ORANGE, TEAL):
        comps = _component_cells(grid == color)
        for comp in comps:
            rows = [r for r, _c in comp]
            cols = [c for _r, c in comp]
            assert len(comp) == 4, "each prism must occupy four cells"
            assert max(rows) - min(rows) == 1 and max(cols) - min(cols) == 1, "each prism must be a solid 2x2 block"
            for cell in comp:
                cell_to_color[cell] = color
                cell_to_block[cell] = block_id
            block_id += 1
    assert block_id >= 1, "expected at least one prism block"
    return cell_to_color, cell_to_block


def _turn(dir_idx: int, prism_color: int) -> int:
    if prism_color == TEAL:
        return (dir_idx + 1) % 4
    assert prism_color == ORANGE, "unexpected prism color"
    return (dir_idx - 1) % 4


def _trace(grid: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must be at most 30x30"
    colors = set(map(int, np.unique(g)))
    assert colors <= {BLACK, RED, ORANGE, TEAL}, "unexpected input colors"

    start, dir_idx = _detect_start(g)
    prism_by_cell, prism_id_by_cell = _detect_prisms(g)

    out = g.copy()
    h, w = g.shape
    r, c = start
    trail_color = RED
    steps_in_segment = 0
    segment_lengths: list[int] = []
    hit_colors: list[int] = []
    turn_senses: list[int] = []
    hit_blocks: list[int] = []
    path_cells = [start]
    visited_states = {(r, c, dir_idx)}

    while True:
        dr, dc = DIRS[dir_idx]
        nr, nc = r + dr, c + dc
        if not _in_bounds((h, w), nr, nc):
            segment_lengths.append(steps_in_segment)
            break

        if (nr, nc) in prism_by_cell:
            prism_color = prism_by_cell[(nr, nc)]
            hit_colors.append(prism_color)
            turn_senses.append(CW if prism_color == TEAL else CCW)
            hit_blocks.append(prism_id_by_cell[(nr, nc)])
            segment_lengths.append(steps_in_segment)
            dir_idx = _turn(dir_idx, prism_color)
            trail_color = prism_color
            steps_in_segment = 0
            state = (r, c, dir_idx)
            assert state not in visited_states, "ray entered a cycle"
            visited_states.add(state)
            continue

        assert g[nr, nc] in {BLACK, RED}, "only background may lie on the traversed path"
        r, c = nr, nc
        path_cells.append((r, c))
        if g[r, c] == BLACK:
            out[r, c] = trail_color
        steps_in_segment += 1
        state = (r, c, dir_idx)
        assert state not in visited_states, "ray entered a cycle"
        visited_states.add(state)

    meta: dict[str, object] = {
        "path_cells": path_cells,
        "segment_lengths": segment_lengths,
        "hit_colors": hit_colors,
        "turn_senses": turn_senses,
        "hit_blocks": hit_blocks,
        "turn_count": len(hit_colors),
    }
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace a ray from the red edge pixel, turning at cyan and orange 2x2 prisms while repainting each segment with the latest prism color.

    Input structure:
    - The grid has a black background, exactly one red start pixel on a non-corner edge cell, and one or more interior solid 2x2 prisms.
    - Every prism is monochromatic: teal prisms and orange prisms are the only interior objects.
    - The intended ray always meets a prism head-on through the next cell in its current direction and eventually leaves the grid.

    Rule:
    - Launch the ray inward, perpendicular to the edge that contains the red start pixel; its initial trail color is red.
    - Whenever the next cell lies inside a prism, turn on the current cell instead of entering the prism: teal turns the ray 90 degrees clockwise, orange turns it 90 degrees counterclockwise.
    - After each turn, change the trail color to that prism's color; while moving, repaint only background cells on the traversed path and never alter prism cells.
    - Stop when the ray steps out of the grid.

    Key insight:
    - The 2x2 blocks are directional prisms rather than obstacles: their visible color encodes turn direction, and the output path color records the most recently hit prism.
    """
    out, _meta = _trace(grid)
    return out


def _step(pos: tuple[int, int], dir_idx: int, length: int) -> tuple[int, int]:
    dr, dc = DIRS[dir_idx]
    return pos[0] + dr * length, pos[1] + dc * length


def _path_cells(pos: tuple[int, int], dir_idx: int, length: int) -> list[tuple[int, int]]:
    dr, dc = DIRS[dir_idx]
    return [(pos[0] + dr * i, pos[1] + dc * i) for i in range(1, length + 1)]


def _cells_to_exit(shape: tuple[int, int], pos: tuple[int, int], dir_idx: int) -> list[tuple[int, int]]:
    dr, dc = DIRS[dir_idx]
    h, w = shape
    r, c = pos
    cells: list[tuple[int, int]] = []
    while True:
        r += dr
        c += dc
        if not _in_bounds((h, w), r, c):
            return cells
        cells.append((r, c))


def _prism_cells(bounce: tuple[int, int], incoming_dir: int, align: int) -> list[tuple[int, int]]:
    r, c = bounce
    if incoming_dir == 0:
        top, left = r - 2, c + align
    elif incoming_dir == 1:
        top, left = r + align, c + 1
    elif incoming_dir == 2:
        top, left = r + 1, c + align
    else:
        top, left = r + align, c - 2
    return [(top + dr, left + dc) for dr in range(2) for dc in range(2)]


def _prism_separated(prism: list[tuple[int, int]], block_cells: set[tuple[int, int]]) -> bool:
    prism_set = set(prism)
    for r, c in prism:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in block_cells and (nr, nc) not in prism_set:
                return False
    return True


def _choose_start(shape: tuple[int, int]) -> tuple[tuple[int, int], int]:
    h, w = shape
    side = random.choice(["top", "right", "bottom", "left"])
    if side == "top":
        return (0, random.randint(2, w - 3)), 2
    if side == "right":
        return (random.randint(2, h - 3), w - 1), 3
    if side == "bottom":
        return (h - 1, random.randint(2, w - 3)), 0
    return (random.randint(2, h - 3), 0), 1


def _sample_turns(turn_count: int) -> list[int]:
    turns = [random.choice([CW, CCW]) for _ in range(turn_count)]
    if turn_count >= 2 and random.random() < 0.75:
        turns[0] = CW
        turns[1] = CCW
        random.shuffle(turns)
    return turns


def _render_base_instance(
    shape: tuple[int, int],
    start: tuple[int, int],
    blocks: list[tuple[list[tuple[int, int]], int]],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    grid[start] = RED
    for cells, color in blocks:
        for r, c in cells:
            grid[r, c] = color
    return grid


def _attempt_path_layout(
    shape: tuple[int, int],
    start: tuple[int, int],
    start_dir: int,
    turn_senses: list[int],
) -> tuple[list[tuple[list[tuple[int, int]], int]], set[tuple[int, int]]] | None:
    occupied_trace = {start}
    occupied_blocks: set[tuple[int, int]] = set()
    placed_blocks: list[tuple[list[tuple[int, int]], int]] = []

    def dfs(
        pos: tuple[int, int],
        dir_idx: int,
        turn_idx: int,
        trace_cells: set[tuple[int, int]],
        block_cells: set[tuple[int, int]],
        blocks: list[tuple[list[tuple[int, int]], int]],
    ) -> tuple[list[tuple[list[tuple[int, int]], int]], set[tuple[int, int]]] | None:
        if turn_idx == len(turn_senses):
            tail = _cells_to_exit(shape, pos, dir_idx)
            if len(tail) < MIN_FINAL_SEGMENT_LEN:
                return None
            if any(cell in trace_cells or cell in block_cells for cell in tail):
                return None
            return blocks, trace_cells | set(tail)

        exit_cells = _cells_to_exit(shape, pos, dir_idx)
        lengths = list(range(MIN_SEGMENT_LEN, len(exit_cells)))
        random.shuffle(lengths)
        sense = turn_senses[turn_idx]

        for length in lengths:
            segment = _path_cells(pos, dir_idx, length)
            if any(cell in trace_cells or cell in block_cells for cell in segment):
                continue

            bounce = segment[-1]
            next_dir = (dir_idx + 1) % 4 if sense == CW else (dir_idx - 1) % 4
            alignments = [-1, 0]
            random.shuffle(alignments)
            for align in alignments:
                prism = _prism_cells(bounce, dir_idx, align)
                if any(not _in_bounds(shape, r, c) for r, c in prism):
                    continue
                if any((r, c) in trace_cells or (r, c) in block_cells for r, c in prism):
                    continue
                if not _prism_separated(prism, block_cells):
                    continue
                nr, nc = bounce[0] + DIRS[next_dir][0], bounce[1] + DIRS[next_dir][1]
                if not _in_bounds(shape, nr, nc):
                    continue
                if (nr, nc) in trace_cells or (nr, nc) in block_cells or (nr, nc) in prism:
                    continue

                color = TEAL if sense == CW else ORANGE
                result = dfs(
                    bounce,
                    next_dir,
                    turn_idx + 1,
                    trace_cells | set(segment),
                    block_cells | set(prism),
                    blocks + [(prism, color)],
                )
                if result is not None:
                    return result
        return None

    return dfs(start, start_dir, 0, occupied_trace, occupied_blocks, placed_blocks)


def _isolated_square_ok(grid: np.ndarray, top: int, left: int) -> bool:
    h, w = grid.shape
    if top <= 0 or left <= 0 or top + 1 >= h - 1 or left + 1 >= w - 1:
        return False
    for r in range(top - 1, top + 3):
        for c in range(left - 1, left + 3):
            if grid[r, c] != BLACK:
                return False
    return True


def _add_distractors(grid: np.ndarray, expected_out: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    result = grid.copy()
    target = random.randint(0, MAX_DISTRACTORS)
    if target == 0:
        return result

    candidates = [(r, c) for r in range(1, h - 2) for c in range(1, w - 2)]
    random.shuffle(candidates)
    added = 0
    for top, left in candidates:
        if added >= target:
            break
        if not _isolated_square_ok(result, top, left):
            continue
        color = random.choice([ORANGE, TEAL])
        trial = result.copy()
        trial[top : top + 2, left : left + 2] = color
        try:
            if np.array_equal(main(trial), expected_out):
                result = trial
                added += 1
        except Exception:
            continue
    return result


def _valid_instance(grid: np.ndarray, require_both_turns: bool) -> bool:
    try:
        out, meta = _trace(grid)
    except Exception:
        return False

    turn_count = int(meta["turn_count"])
    segment_lengths = list(meta["segment_lengths"])
    hit_colors = list(meta["hit_colors"])
    hit_blocks = list(meta["hit_blocks"])

    if turn_count < TURN_COUNT_RANGE[0]:
        return False
    if len(segment_lengths) != turn_count + 1:
        return False
    if any(length < MIN_SEGMENT_LEN for length in segment_lengths):
        return False
    if len(set(hit_blocks)) != len(hit_blocks):
        return False
    if len(set(tuple(cell) for cell in meta["path_cells"])) != len(meta["path_cells"]):
        return False
    if len(np.unique(out)) < 3:
        return False
    if require_both_turns and set(hit_colors) != {ORANGE, TEAL}:
        return False

    total_blocks = sum(1 for color in np.unique(grid) if int(color) in PRISM_COLORS)
    if total_blocks < 1:
        return False
    return True


def _attempt_instance() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    shape = (h, w)
    start, start_dir = _choose_start(shape)
    turn_count = random.randint(*TURN_COUNT_RANGE)
    turn_senses = _sample_turns(turn_count)
    require_both_turns = CW in turn_senses and CCW in turn_senses

    layout = _attempt_path_layout(shape, start, start_dir, turn_senses)
    if layout is None:
        return None
    blocks, trace_cells = layout

    grid = _render_base_instance(shape, start, blocks)
    expected_out = main(grid)
    if len(trace_cells) < 10:
        return None

    grid = _add_distractors(grid, expected_out)
    if not _valid_instance(grid, require_both_turns):
        return None
    if not np.array_equal(main(grid), expected_out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_instance()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid双色棱镜弹球 input")
