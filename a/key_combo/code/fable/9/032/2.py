from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

RIBBON_PALETTE = (BLUE, RED, GREEN, YELLOW, PINK, TEAL, MAROON)

MIN_HEIGHT = 13
MAX_HEIGHT = 24
MIN_WIDTH = 12
MAX_WIDTH = 24
MIN_TURNS = 2
MAX_TURNS = 5
MAX_RETRIES = 600


def _components(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    if connectivity == 4:
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        dirs = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            q: deque[tuple[int, int]] = deque([(r, c)])
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _parse_ribbon(grid: np.ndarray) -> tuple[list[int], int]:
    row = np.asarray(grid[0], dtype=int)
    width = row.shape[0]
    length = 0
    while length < width and int(row[length]) != BLACK:
        length += 1
    assert length >= 2, "expected a non-empty ribbon with at least two cells"
    assert np.all(row[length:] == BLACK), "ribbon must be contiguous from column 0"
    colors = [int(v) for v in row[:length]]
    assert len(colors) == len(set(colors)), "ribbon colors must be distinct"
    return colors, length


def _find_start(grid: np.ndarray) -> tuple[int, int]:
    h, _w = grid.shape
    starts = np.argwhere(grid == GRAY)
    assert starts.shape[0] == 1, "expected exactly one gray start marker"
    r, c = map(int, starts[0])
    assert r == h - 1, "gray start marker must lie on the bottom edge"
    return r, c


def _orange_run_bounds(grid: np.ndarray, row: int, col: int) -> tuple[int, int]:
    h, w = grid.shape
    assert 0 <= row < h and 0 <= col < w
    assert int(grid[row, col]) == ORANGE
    left = col
    while left - 1 >= 0 and int(grid[row, left - 1]) == ORANGE:
        left -= 1
    right = col
    while right + 1 < w and int(grid[row, right + 1]) == ORANGE:
        right += 1
    return left, right


def _trace_route(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert g.ndim == 2
    assert h >= 3 and w >= 3

    ribbon, ribbon_len = _parse_ribbon(g)
    _start_r, start_c = _find_start(g)

    out = g.copy()
    current_r = h - 1
    current_c = start_c
    color_index = 0
    path_cells: list[tuple[int, int]] = []
    turn_dirs: list[str] = []
    turn_rows: list[int] = []
    probe_cols: dict[int, int] = {}

    while current_r > 1:
        probe_cols[current_r - 1] = current_c
        if int(g[current_r - 1, current_c]) == ORANGE:
            row = current_r - 1
            left, right = _orange_run_bounds(g, row, current_c)
            dist_left = current_c - left
            dist_right = right - current_c
            assert dist_left != dist_right, "obstacle ends must not be equidistant"

            color_index += 1
            assert color_index < len(ribbon), "route uses more colors than the ribbon provides"
            turn_color = int(ribbon[color_index])

            if dist_left < dist_right:
                target_c = left - 1
                step = -1
                turn_dirs.append("L")
            else:
                target_c = right + 1
                step = 1
                turn_dirs.append("R")
            assert 0 <= target_c < w, "detour exit must stay in bounds"

            for cc in range(current_c + step, target_c + step, step):
                assert int(g[current_r, cc]) == BLACK, "detour path must run on background"
                out[current_r, cc] = turn_color
                path_cells.append((current_r, cc))

            turn_rows.append(row)
            current_c = target_c
        else:
            current_r -= 1
            assert int(g[current_r, current_c]) == BLACK, "vertical path must run on background"
            out[current_r, current_c] = int(ribbon[color_index])
            path_cells.append((current_r, current_c))

    used_count = color_index + 1
    out[0, :ribbon_len] = BLACK
    out[0, :used_count] = np.array(ribbon[:used_count], dtype=int)
    return {
        "output": out,
        "used_count": used_count,
        "path_cells": path_cells,
        "turn_dirs": turn_dirs,
        "turn_rows": turn_rows,
        "probe_cols": probe_cols,
        "ribbon": ribbon,
        "ribbon_len": ribbon_len,
        "start_col": start_c,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw a vertical color-program path from the gray start, switching ribbon colors whenever the route detours around an orange bar.

    Input structure:
    - The top row starts at column 0 with a contiguous ribbon of distinct single-cell colors; these ribbon colors do not appear in the lower scene.
    - Below the ribbon, the background is black, the unique start marker is one gray pixel on the bottom edge, and every obstacle is a horizontal orange bar.
    - Orange bars are separated from each other and the route must infer its next color only from how many obstacle-detours have already happened.

    Rule:
    - Begin at the gray start column and draw upward on black cells, using the ribbon's first color; the gray start itself stays gray.
    - Whenever the cell directly above the current route position is orange, switch to the next ribbon color, move along the row just below that bar toward the nearer bar end, step one column outside that end, and then continue upward.
    - Continue until the route reaches row 1; keep all orange bars unchanged, keep only the ribbon cells whose colors were actually used, and erase every unused ribbon cell to black.

    Key insight:
    - Geometry chooses only the detour direction, while color is driven purely by event order: the k-th bypass consumes ribbon color k+1 for the entire following segment.
    """
    return np.asarray(_trace_route(np.asarray(grid, dtype=int))["output"], dtype=int)


def _sample_turn_rows(height: int, turn_count: int, rng: random.Random) -> list[int] | None:
    candidates = list(range(2, height - 2))
    if len(candidates) < turn_count:
        return None
    for _ in range(80):
        rows = sorted(rng.sample(candidates, turn_count), reverse=True)
        if all(rows[i] - rows[i + 1] >= 2 for i in range(len(rows) - 1)):
            return rows
    return None


def _sample_directions(turn_count: int, rng: random.Random) -> list[str]:
    while True:
        dirs = [rng.choice(["L", "R"]) for _ in range(turn_count)]
        if "L" in dirs and "R" in dirs:
            return dirs


def _build_turn_plan(
    width: int,
    start_col: int,
    directions: list[str],
    rng: random.Random,
) -> list[dict[str, int]] | None:
    current_c = start_col
    plan: list[dict[str, int]] = []
    for direction in directions:
        if direction == "R":
            step_max = min(3, current_c, width - 2 - current_c)
            if step_max < 1:
                return None
            step = rng.randint(1, step_max)
            near = step - 1
            far_min = step
            far_max = min(4, current_c)
            if far_min > far_max:
                return None
            far = rng.randint(far_min, far_max)
            left = current_c - far
            right = current_c + near
            exit_c = current_c + step
        else:
            step_max = min(3, current_c - 1, width - 1 - current_c)
            if step_max < 1:
                return None
            step = rng.randint(1, step_max)
            near = step - 1
            far_min = step
            far_max = min(4, width - 1 - current_c)
            if far_min > far_max:
                return None
            far = rng.randint(far_min, far_max)
            left = current_c - near
            right = current_c + far
            exit_c = current_c - step

        if not (0 <= left <= current_c <= right < width):
            return None
        if right - left + 1 > 7:
            return None

        plan.append(
            {
                "entry_c": current_c,
                "left": left,
                "right": right,
                "exit_c": exit_c,
            }
        )
        current_c = exit_c
    return plan


def _place_distractors(grid: np.ndarray, trace: dict[str, object], rng: random.Random) -> None:
    h, w = grid.shape
    used_rows = set(map(int, trace["turn_rows"]))
    probe_cols = {int(r): int(c) for r, c in dict(trace["probe_cols"]).items()}
    candidates = [
        row
        for row in range(2, h - 2)
        if all(abs(row - other) >= 2 for other in used_rows)
    ]
    rng.shuffle(candidates)
    extra_target = rng.randint(1, min(3, len(candidates))) if candidates else 0
    placed = 0

    for row in candidates:
        probe = probe_cols.get(row)
        options: list[tuple[int, int]] = []
        max_len = min(6, w)
        for length in range(2, max_len + 1):
            for left in range(0, w - length + 1):
                right = left + length - 1
                if probe is not None and left <= probe <= right:
                    continue
                options.append((left, right))
        rng.shuffle(options)
        for left, right in options:
            if np.any(grid[row, left : right + 1] != BLACK):
                continue
            grid[row, left : right + 1] = ORANGE
            used_rows.add(row)
            placed += 1
            break
        if placed >= extra_target:
            return


def _validate_generated_case(grid: np.ndarray) -> bool:
    try:
        g = np.asarray(grid, dtype=int)
        h, w = g.shape
        if g.ndim != 2 or h > 30 or w > 30:
            return False

        ribbon, ribbon_len = _parse_ribbon(g)
        if set(ribbon) & {BLACK, GRAY, ORANGE}:
            return False
        if any(int(v) in set(ribbon) for v in g[1:].flat):
            return False

        start_positions = np.argwhere(g == GRAY)
        if start_positions.shape[0] != 1:
            return False
        if tuple(map(int, start_positions[0]))[0] != h - 1:
            return False

        orange_comps = _components(g == ORANGE, connectivity=4)
        if len(orange_comps) < MIN_TURNS:
            return False
        for comp in orange_comps:
            rows = {r for r, _c in comp}
            if len(rows) != 1:
                return False
        orange_mask = g == ORANGE
        for r in range(h):
            for c in range(w):
                if not orange_mask[r, c]:
                    continue
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < h and 0 <= cc < w and orange_mask[rr, cc]:
                            if abs(dr) == 1 and abs(dc) == 1:
                                return False

        trace = _trace_route(g)
        out = np.asarray(trace["output"], dtype=int)
        if not np.array_equal(main(g), out):
            return False

        turn_dirs = list(trace["turn_dirs"])
        used_count = int(trace["used_count"])
        path_cells = [tuple(map(int, rc)) for rc in trace["path_cells"]]
        if len(path_cells) != len(set(path_cells)):
            return False
        if used_count >= ribbon_len:
            return False
        if len(turn_dirs) != used_count - 1:
            return False
        if "L" not in turn_dirs or "R" not in turn_dirs:
            return False
        if 1 not in {r for r, _c in path_cells}:
            return False
        if h - 2 not in {r for r, _c in path_cells}:
            return False

        used_colors = ribbon[:used_count]
        for color in used_colors:
            if not np.any(out == int(color)):
                return False
        if not np.all(out[0, used_count:ribbon_len] == BLACK):
            return False
        if not np.array_equal(g[g == ORANGE], out[g == ORANGE]):
            return False
        if int(out[h - 1, int(trace["start_col"])]) != GRAY:
            return False
    except Exception:
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    turn_count = rng.randint(MIN_TURNS, MAX_TURNS)
    height = rng.randint(max(MIN_HEIGHT, 2 * turn_count + 3), MAX_HEIGHT)
    width = rng.randint(MIN_WIDTH, MAX_WIDTH)
    ribbon_len = turn_count + 2

    rows = _sample_turn_rows(height, turn_count, rng)
    if rows is None:
        return None

    ribbon = rng.sample(list(RIBBON_PALETTE), ribbon_len)
    directions = _sample_directions(turn_count, rng)

    for _ in range(80):
        start_col = rng.randint(3, width - 4)
        bars = _build_turn_plan(width, start_col, directions, rng)
        if bars is None:
            continue

        grid = np.zeros((height, width), dtype=int)
        grid[0, :ribbon_len] = np.array(ribbon, dtype=int)
        grid[-1, start_col] = GRAY
        for row, bar in zip(rows, bars):
            grid[row, bar["left"] : bar["right"] + 1] = ORANGE

        try:
            base_trace = _trace_route(grid)
        except Exception:
            continue
        if len(base_trace["turn_dirs"]) != turn_count:
            continue
        if list(base_trace["turn_dirs"]) != directions:
            continue

        expected = np.asarray(base_trace["output"], dtype=int)
        _place_distractors(grid, base_trace, rng)

        try:
            solved = main(grid)
        except Exception:
            continue
        if not np.array_equal(solved, expected):
            continue
        if not _validate_generated_case(grid):
            continue
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(random.Random())
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid obstacle-ribbon input")
