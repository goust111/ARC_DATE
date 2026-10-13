from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5

RIGHT_TURN = {
    (0, 1): (1, 0),
    (1, 0): (0, -1),
    (0, -1): (-1, 0),
    (-1, 0): (0, 1),
}
LEFT_TURN = {
    (0, 1): (-1, 0),
    (-1, 0): (0, -1),
    (0, -1): (1, 0),
    (1, 0): (0, 1),
}

MIN_SIZE = 8
MAX_SIZE = 18
MIN_GAP = 2
MAX_RETRIES = 300


def _find_start(grid: np.ndarray) -> tuple[int, int]:
    h, w = grid.shape
    starts: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w - 1):
            if int(grid[r, c]) != GREEN or int(grid[r, c + 1]) != GREEN:
                continue
            if c > 0 and int(grid[r, c - 1]) == GREEN:
                continue
            starts.append((int(r), int(c)))

    assert len(starts) == 1, "expected one green start marker with a right-facing green pointer"
    return starts[0]


def _paint_path(grid: np.ndarray, path: list[tuple[int, int]]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for r, c in path:
        if int(out[r, c]) not in (RED, BLUE):
            out[r, c] = GREY
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw the grey route whose turns are triggered by marker row/column alignment.

    Input structure:
    - Background is black.
    - A green start pixel has a green pixel immediately to its right, so the
      initial direction is rightward.
    - Red and blue singleton markers sit away from the start row/column.

    Rule:
    - Move one step at a time, starting from the start pixel.
    - While moving horizontally, the first unused marker whose column matches the
      current column triggers a turn at the current cell.
    - While moving vertically, the first unused marker whose row matches the
      current row triggers a turn at the current cell.
    - Red means turn right; blue means turn left.
    - Used markers do not trigger again.
    - Stop once the next step would leave the grid.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    r, c = _find_start(g)
    dr, dc = 0, 1

    unused = {
        (int(rr), int(cc)): int(g[rr, cc])
        for rr, cc in np.argwhere((g == RED) | (g == BLUE))
    }
    path = [(int(r), int(c))]

    while True:
        nr, nc = int(r + dr), int(c + dc)
        if not (0 <= nr < h and 0 <= nc < w):
            break

        r, c = nr, nc
        path.append((int(r), int(c)))

        if dc != 0:
            hits = sorted(pos for pos in unused if int(pos[1]) == int(c))
        else:
            hits = sorted(pos for pos in unused if int(pos[0]) == int(r))

        if not hits:
            continue

        marker = hits[0]
        color = int(unused.pop(marker))
        if color == RED:
            dr, dc = RIGHT_TURN[(int(dr), int(dc))]
        else:
            dr, dc = LEFT_TURN[(int(dr), int(dc))]

    return _paint_path(g, path)


def _sample_increasing(count: int, lo: int, hi: int, min_gap: int = MIN_GAP) -> list[int] | None:
    if count == 0:
        return []

    span = int(hi - lo + 1)
    need = int(count + (count - 1) * (min_gap - 1))
    if span < need:
        return None

    slack = span - need
    extras = [0] * (count + 1)
    for _ in range(slack):
        extras[random.randrange(count + 1)] += 1

    values = [int(lo + extras[0])]
    for i in range(1, count):
        values.append(int(values[-1] + min_gap + extras[i]))
    return values


def _build_path(
    h: int,
    w: int,
    final_turn: str,
    cols: list[int],
    rows: list[int],
) -> tuple[list[tuple[int, int]], list[dict[str, object]]]:
    """
    Build a monotone staircase from the top-left, then escape with one final
    inward-safe turn. The generator uses this shape because any marker placed
    above/left of the current frontier would be triggered too early.
    """
    r = c = 0
    direction = (0, 1)
    path = [(0, 0)]
    turns: list[dict[str, object]] = []
    col_idx = row_idx = 0
    turn_count = len(cols) + len(rows)

    def extend_to(target_r: int, target_c: int) -> None:
        nonlocal r, c
        step_r = 0 if target_r == r else (1 if target_r > r else -1)
        step_c = 0 if target_c == c else (1 if target_c > c else -1)
        while (r, c) != (target_r, target_c):
            r += step_r
            c += step_c
            path.append((int(r), int(c)))

    for turn_idx in range(turn_count):
        if direction == (0, 1):
            x = int(cols[col_idx])
            col_idx += 1
            extend_to(r, x)
            if turn_idx == turn_count - 1:
                next_dir = (-1, 0)
                color = BLUE
            else:
                next_dir = (1, 0)
                color = RED
            turns.append(
                {
                    "orientation": "horizontal",
                    "turn_at": (int(r), int(c)),
                    "color": int(color),
                    "next_dir": (int(next_dir[0]), int(next_dir[1])),
                }
            )
            direction = next_dir
            continue

        y = int(rows[row_idx])
        row_idx += 1
        extend_to(y, c)
        if turn_idx == turn_count - 1:
            next_dir = (0, -1)
            color = RED
        else:
            next_dir = (0, 1)
            color = BLUE
        turns.append(
            {
                "orientation": "vertical",
                "turn_at": (int(r), int(c)),
                "color": int(color),
                "next_dir": (int(next_dir[0]), int(next_dir[1])),
            }
        )
        direction = next_dir

    if direction == (0, 1):
        extend_to(r, w - 1)
    elif direction == (0, -1):
        extend_to(r, 0)
    elif direction == (1, 0):
        extend_to(h - 1, c)
    else:
        extend_to(0, c)

    assert final_turn in {"left_exit", "up_exit"}
    return path, turns


def _choose_markers(
    h: int,
    w: int,
    path: list[tuple[int, int]],
    turns: list[dict[str, object]],
) -> list[tuple[int, int, int]] | None:
    path_rows: dict[int, int] = {}
    path_cols: dict[int, int] = {}
    for r, c in path:
        path_rows[int(r)] = max(path_rows.get(int(r), -1), int(c))
        path_cols[int(c)] = max(path_cols.get(int(c), -1), int(r))

    markers: list[tuple[int, int, int]] = []
    used: set[tuple[int, int]] = set()
    for turn in turns:
        r, c = turn["turn_at"]  # type: ignore[misc]
        color = int(turn["color"])  # type: ignore[arg-type]

        if turn["orientation"] == "horizontal":
            start_r = int(path_cols[int(c)] + 1)
            choices = [
                (rr, int(c))
                for rr in range(max(1, start_r), h)
                if (int(rr), int(c)) not in used
            ]
        else:
            start_c = int(path_rows[int(r)] + 1)
            choices = [
                (int(r), cc)
                for cc in range(max(1, start_c), w)
                if (int(r), int(cc)) not in used
            ]

        if not choices:
            return None

        mr, mc = random.choice(choices)
        used.add((int(mr), int(mc)))
        markers.append((int(mr), int(mc), int(color)))

    return markers


def _gen_attempt() -> np.ndarray | None:
    final_turn = random.choice(["left_exit", "up_exit"])
    if final_turn == "up_exit":
        vertical_turns = random.randint(1, 3)
        horizontal_turns = vertical_turns + 1
    else:
        horizontal_turns = vertical_turns = random.randint(1, 4)

    h_min = max(MIN_SIZE, 2 * vertical_turns + 4)
    w_min = max(MIN_SIZE, 2 * horizontal_turns + 4)
    h = random.randint(h_min, MAX_SIZE)
    w = random.randint(w_min, MAX_SIZE)

    rows = _sample_increasing(vertical_turns, 1, h - 2)
    cols = _sample_increasing(horizontal_turns, 2, w - 2)
    if rows is None or cols is None:
        return None

    path, turns = _build_path(h, w, final_turn, cols, rows)
    markers = _choose_markers(h, w, path, turns)
    if markers is None:
        return None

    grid = np.full((h, w), BLACK, dtype=int)
    grid[0, 0] = GREEN
    grid[0, 1] = GREEN
    for r, c, color in markers:
        grid[r, c] = color

    expected = _paint_path(grid, path)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
