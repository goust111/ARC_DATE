from __future__ import annotations

import itertools
import random
from typing import Any

import numpy as np


BG = 0
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _foreground_role_counts(grid: np.ndarray) -> list[tuple[int, int]]:
    colors, counts = np.unique(grid[grid != BG], return_counts=True)
    assert len(colors) == 3, "expected exactly three foreground colors"
    ranked = sorted(((int(color), int(count)) for color, count in zip(colors, counts)), key=lambda item: -item[1])
    assert ranked[0][1] > ranked[1][1] > ranked[2][1], "foreground counts must be strictly ordered"
    return ranked


def _infer_roles(grid: np.ndarray) -> tuple[int, int, int]:
    ranked = _foreground_role_counts(grid)
    return ranked[0][0], ranked[1][0], ranked[2][0]


def _moving_run_info(grid: np.ndarray, row: int, moving_color: int) -> tuple[int, int] | None:
    cols = np.flatnonzero(grid[row] == moving_color)
    if len(cols) == 0:
        return None
    start = int(cols[0])
    end = int(cols[-1])
    assert end == grid.shape[1] - 1, "moving pixels must end at the right edge"
    assert len(cols) == end - start + 1, "moving pixels must form one contiguous run"
    return start, int(len(cols))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide the most frequent color's right-edge runs left against row obstacles, then recolor the rarest color by the moved rows.

    Input structure:
    - The background is black.
    - Exactly three foreground colors appear, and their global pixel counts are all different.
    - The most frequent foreground color forms contiguous horizontal runs on the right edge of several rows; the second-most frequent color is scattered as singleton obstacles; the least frequent color is scattered as singleton markers.

    Rule:
    - Rank the three foreground colors by total pixel count to determine moving, obstacle, and marker roles.
    - In each row containing the moving color, slide that whole right-edge run left until it touches the nearest obstacle in the same row, or the left edge if no obstacle exists; keep the run length and color unchanged and clear its old position to black.
    - Leave obstacle pixels fixed.
    - For every marker pixel, recolor it to the moving color if its row contains a moved run after the slide; otherwise erase it to black.

    Key insight:
    - Color identity and visible shape are decoys: only global frequency reveals which color moves, which blocks, and which gets recolored by row membership after the slide.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    moving_color, obstacle_color, marker_color = _infer_roles(g)
    out = g.copy()

    for r in range(g.shape[0]):
        info = _moving_run_info(g, r, moving_color)
        if info is None:
            continue

        start, run_len = info
        obstacle_cols = np.flatnonzero(g[r, :start] == obstacle_color)
        target_start = int(obstacle_cols[-1]) + 1 if len(obstacle_cols) else 0
        target_end = target_start + run_len
        assert target_end <= g.shape[1], "shifted moving run does not fit"
        assert np.all(g[r, target_start:target_end] != obstacle_color), "moving run cannot cover an obstacle"

        out[r, start:] = np.where(np.arange(start, g.shape[1]) < start + run_len, BG, out[r, start:])
        out[r, target_start:target_end] = moving_color

    moved_rows = set(int(r) for r in np.flatnonzero(np.any(out == moving_color, axis=1)))
    for r, c in np.argwhere(g == marker_color):
        out[r, c] = moving_color if int(r) in moved_rows else BG

    return out


GRID_H_RANGE = (9, 12)
GRID_W_RANGE = (13, 16)
RUN_LEN_RANGE = (3, 6)
MOVING_ROW_COUNT_RANGE = (4, 5)
MARKER_COUNT_RANGE = (3, 8)
BARRIER_COUNT_CAP = 14
RETRIES = 400


def _choose_nonadjacent_rows(height: int, count: int) -> tuple[int, ...] | None:
    combos = [rows for rows in itertools.combinations(range(height), count) if all(rows[i + 1] - rows[i] >= 2 for i in range(len(rows) - 1))]
    return random.choice(combos) if combos else None


def _isolated_zero(grid: np.ndarray, r: int, c: int) -> bool:
    if grid[r, c] != BG:
        return False
    for dr, dc in DIR4:
        nr, nc = r + dr, c + dc
        if _in_bounds(grid.shape, nr, nc) and grid[nr, nc] != BG:
            return False
    return True


def _pick_from_candidates(candidates: list[tuple[int, int]], grid: np.ndarray) -> tuple[int, int] | None:
    random.shuffle(candidates)
    for r, c in candidates:
        if _isolated_zero(grid, r, c):
            return int(r), int(c)
    return None


def _marker_candidates_for_row(
    grid: np.ndarray,
    row: int,
    moving_rows: set[int],
    input_starts: dict[int, int],
    output_starts: dict[int, int],
    run_lengths: dict[int, int],
) -> list[tuple[int, int]]:
    width = grid.shape[1]
    if row not in moving_rows:
        return [(row, c) for c in range(width)]

    input_start = input_starts[row]
    output_start = output_starts[row]
    output_end = output_start + run_lengths[row] - 1
    candidates = []
    for c in range(width):
        if c >= input_start - 1:
            continue
        if output_start <= c <= output_end:
            continue
        if output_start > 0 and c == output_start - 1:
            continue
        if c == output_end + 1:
            continue
        candidates.append((row, c))
    return candidates


def _allowed_barrier_cell(
    row: int,
    col: int,
    moving_rows: set[int],
    input_starts: dict[int, int],
    output_starts: dict[int, int],
) -> bool:
    if row not in moving_rows:
        return True
    if output_starts[row] == 0:
        return False
    return col <= output_starts[row] - 2 and col < input_starts[row]


def _render_expected_from_meta(inp: np.ndarray, meta: dict[str, Any]) -> np.ndarray:
    out = inp.copy()
    moving_color = int(meta["moving_color"])
    marker_color = int(meta["marker_color"])
    width = inp.shape[1]

    for row, run_len in meta["run_lengths"].items():
        row = int(row)
        run_len = int(run_len)
        input_start = width - run_len
        target_start = int(meta["output_starts"][row])
        out[row, input_start:width] = BG
        out[row, target_start : target_start + run_len] = moving_color

    moving_rows = set(int(r) for r in meta["moving_rows"])
    for r, c in np.argwhere(inp == marker_color):
        out[r, c] = moving_color if int(r) in moving_rows else BG

    return out


def _semantic_check(inp: np.ndarray, meta: dict[str, Any], out: np.ndarray) -> bool:
    ranked = _foreground_role_counts(inp)
    moving_color, barrier_color, marker_color = (ranked[0][0], ranked[1][0], ranked[2][0])
    if moving_color != meta["moving_color"] or barrier_color != meta["barrier_color"] or marker_color != meta["marker_color"]:
        return False
    if ranked[0][1] - ranked[1][1] < 3 or ranked[1][1] - ranked[2][1] < 3:
        return False

    moving_rows = set(int(r) for r in meta["moving_rows"])
    if not moving_rows or not meta["obstacle_rows"] or not meta["no_obstacle_rows"]:
        return False
    if not meta["marker_rows_with_mover"] or not meta["marker_rows_without_mover"]:
        return False

    for row in moving_rows:
        info = _moving_run_info(inp, row, moving_color)
        if info is None:
            return False
        input_start, run_len = info
        if input_start != inp.shape[1] - run_len:
            return False
        if run_len != meta["run_lengths"][row]:
            return False

    for r, c in meta["singleton_coords"]:
        if inp[r, c] == BG:
            return False
        neighbors = 0
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if _in_bounds(inp.shape, nr, nc) and inp[nr, nc] != BG:
                neighbors += 1
        if neighbors != 0:
            return False

    expected = _render_expected_from_meta(inp, meta)
    return np.array_equal(expected, out)


def _gen_attempt() -> tuple[np.ndarray, dict[str, Any]] | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    max_moving_rows = min(MOVING_ROW_COUNT_RANGE[1], (height + 1) // 2)
    min_moving_rows = min(MOVING_ROW_COUNT_RANGE[0], max_moving_rows)
    if min_moving_rows < 2:
        return None

    moving_row_count = random.randint(min_moving_rows, max_moving_rows)
    moving_rows_tuple = _choose_nonadjacent_rows(height, moving_row_count)
    if moving_rows_tuple is None:
        return None
    moving_rows = set(int(r) for r in moving_rows_tuple)
    nonmoving_rows = [r for r in range(height) if r not in moving_rows]
    if not nonmoving_rows:
        return None

    colors = random.sample(list(range(1, 10)), 3)
    moving_color, barrier_color, marker_color = colors

    run_lengths = {row: random.randint(*RUN_LEN_RANGE) for row in moving_rows}
    moving_total = sum(run_lengths.values())
    if moving_total < 15:
        return None

    obstacle_row_count = random.randint(1, len(moving_rows) - 1)
    obstacle_rows = set(random.sample(sorted(moving_rows), obstacle_row_count))
    no_obstacle_rows = moving_rows - obstacle_rows
    if not no_obstacle_rows:
        return None

    output_starts: dict[int, int] = {}
    input_starts: dict[int, int] = {}
    for row in moving_rows:
        run_len = run_lengths[row]
        input_start = width - run_len
        input_starts[row] = input_start
        if row in no_obstacle_rows:
            output_starts[row] = 0
        else:
            output_starts[row] = random.randint(1, input_start - 2)

    max_marker = min(MARKER_COUNT_RANGE[1], moving_total - 6)
    if max_marker < MARKER_COUNT_RANGE[0]:
        return None
    marker_target = random.randint(MARKER_COUNT_RANGE[0], max_marker)
    barrier_low = max(len(obstacle_rows), marker_target + 3, 6)
    barrier_high = min(BARRIER_COUNT_CAP, moving_total - 3)
    if barrier_low > barrier_high:
        return None
    barrier_target = random.randint(barrier_low, barrier_high)

    inp = np.zeros((height, width), dtype=int)
    for row in moving_rows:
        run_len = run_lengths[row]
        inp[row, width - run_len : width] = moving_color

    barrier_coords: list[tuple[int, int]] = []
    for row in sorted(obstacle_rows):
        col = output_starts[row] - 1
        if not _isolated_zero(inp, row, col):
            return None
        inp[row, col] = barrier_color
        barrier_coords.append((row, col))

    marker_coords: list[tuple[int, int]] = []
    preserve_row_options = sorted(no_obstacle_rows) + sorted(obstacle_rows)
    preserve_row = None
    preserve_pos = None
    for row in preserve_row_options:
        pos = _pick_from_candidates(
            _marker_candidates_for_row(inp, row, moving_rows, input_starts, output_starts, run_lengths),
            inp,
        )
        if pos is not None:
            preserve_row = row
            preserve_pos = pos
            break
    if preserve_pos is None:
        return None
    inp[preserve_pos] = marker_color
    marker_coords.append(preserve_pos)

    erase_row = random.choice(nonmoving_rows)
    erase_pos = _pick_from_candidates([(erase_row, c) for c in range(width)], inp)
    if erase_pos is None:
        return None
    inp[erase_pos] = marker_color
    marker_coords.append(erase_pos)

    extra_moving_marker_goal = random.randint(0, max(0, marker_target - 2))
    moving_marker_count = 1
    while len(marker_coords) < marker_target:
        prefer_moving = moving_marker_count <= extra_moving_marker_goal and bool(moving_rows)
        candidate_rows = list(sorted(moving_rows)) if prefer_moving else nonmoving_rows[:]
        random.shuffle(candidate_rows)
        placed = False
        for row in candidate_rows:
            if row in moving_rows:
                candidates = _marker_candidates_for_row(inp, row, moving_rows, input_starts, output_starts, run_lengths)
            else:
                candidates = [(row, c) for c in range(width)]
            pos = _pick_from_candidates(candidates, inp)
            if pos is None:
                continue
            inp[pos] = marker_color
            marker_coords.append(pos)
            if row in moving_rows:
                moving_marker_count += 1
            placed = True
            break
        if placed:
            continue

        fallback_rows = list(range(height))
        random.shuffle(fallback_rows)
        for row in fallback_rows:
            candidates = (
                _marker_candidates_for_row(inp, row, moving_rows, input_starts, output_starts, run_lengths)
                if row in moving_rows
                else [(row, c) for c in range(width)]
            )
            pos = _pick_from_candidates(candidates, inp)
            if pos is not None:
                inp[pos] = marker_color
                marker_coords.append(pos)
                if row in moving_rows:
                    moving_marker_count += 1
                placed = True
                break
        if not placed:
            return None

    while len(barrier_coords) < barrier_target:
        candidates = []
        for row in range(height):
            for col in range(width):
                if _allowed_barrier_cell(row, col, moving_rows, input_starts, output_starts):
                    candidates.append((row, col))
        pos = _pick_from_candidates(candidates, inp)
        if pos is None:
            return None
        inp[pos] = barrier_color
        barrier_coords.append(pos)

    meta = {
        "moving_color": moving_color,
        "barrier_color": barrier_color,
        "marker_color": marker_color,
        "moving_rows": sorted(moving_rows),
        "obstacle_rows": sorted(obstacle_rows),
        "no_obstacle_rows": sorted(no_obstacle_rows),
        "run_lengths": run_lengths,
        "output_starts": output_starts,
        "input_starts": input_starts,
        "marker_rows_with_mover": sorted({r for r, _ in marker_coords if r in moving_rows}),
        "marker_rows_without_mover": sorted({r for r, _ in marker_coords if r not in moving_rows}),
        "singleton_coords": barrier_coords + marker_coords,
    }

    out = _render_expected_from_meta(inp, meta)
    if not _semantic_check(inp, meta, out):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp, meta


def generate_input() -> np.ndarray:
    for _ in range(RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result[0]
    raise RuntimeError("failed to generate a valid input for frequency-ranked sliding recolor")
