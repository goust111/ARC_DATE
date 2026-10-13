from __future__ import annotations

import random

import numpy as np


PATH_COLORS = (1, 2, 3, 4)
BACKGROUND = 0

GRID_HEIGHT_RANGE = (16, 22)
GRID_WIDTH_RANGE = (16, 24)
PAIR_COUNT_CHOICES = (3, 4)
START_COL_MAX = 4
ROW_GAP_AFTER_CLEAR = 2
MAX_GENERATION_ATTEMPTS = 500


def _extract_pairs(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    colors = [int(color) for color in np.unique(g) if int(color) != BACKGROUND]
    assert 3 <= len(colors) <= 4, "expected 3-4 colored marker pairs"

    pairs: list[dict] = []
    for color in colors:
        coords = [tuple(map(int, rc)) for rc in np.argwhere(g == color)]
        assert len(coords) == 2, "each color must appear exactly twice"
        coords.sort(key=lambda rc: (rc[1], rc[0]))
        start, end = coords
        assert start[1] < end[1], "start marker must be left of end marker"
        assert start[0] != end[0], "start and end rows must differ"
        pairs.append({"color": color, "start": start, "end": end})

    start_rows = [pair["start"][0] for pair in pairs]
    assert len(set(start_rows)) == len(start_rows), "start rows must be unique"
    return sorted(pairs, key=lambda pair: pair["start"][0])


def _simulate_paths(grid: np.ndarray, pairs: list[dict]) -> tuple[np.ndarray, list[dict]]:
    out = np.asarray(grid, dtype=int).copy()
    path_mask = np.zeros(out.shape, dtype=bool)
    events: list[dict] = []

    for pair in pairs:
        color = int(pair["color"])
        start = tuple(pair["start"])
        end = tuple(pair["end"])
        r, c = start
        end_r, end_c = end
        route = [start]

        dc = 1 if end_c > c else -1
        collided_h = False
        while c != end_c:
            next_c = c + dc
            if path_mask[r, next_c]:
                collided_h = True
                break
            c = next_c
            route.append((r, c))
            if out[r, c] == BACKGROUND:
                out[r, c] = color
                path_mask[r, c] = True

        dr = 1 if end_r > r else -1
        stopped_v = False
        while r != end_r:
            next_r = r + dr
            if path_mask[next_r, c]:
                stopped_v = True
                break
            r = next_r
            route.append((r, c))
            if out[r, c] == BACKGROUND:
                out[r, c] = color
                path_mask[r, c] = True

        events.append(
            {
                "pair": pair,
                "collided_h": collided_h,
                "stopped_v": stopped_v,
                "turn_col": c,
                "route": route,
            }
        )

    return out, events


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw each same-color marker pair as a horizontal-then-vertical L path, with earlier paths blocking later ones.

    Input structure:
    - The grid is mostly black background, plus 3-4 colors that each appear exactly twice.
    - For each color, the left marker is the start and the right marker is the end.
    - Start rows are all distinct, so they define a unique top-to-bottom drawing order.

    Rule:
    - Sort the color pairs by the start marker row from top to bottom.
    - For each pair in that order, extend a path horizontally from the start toward the end column.
    - If the next horizontal cell already belongs to a previously drawn path, turn immediately at the current cell instead of entering the obstacle.
    - Then extend vertically toward the end row, painting only black cells and stopping if the next vertical cell already belongs to a previously drawn path. Keep all start/end marker pixels unchanged.

    Key insight:
    - The visible markers only specify the endpoints; the decisive hidden dependency is that already drawn paths become obstacles, so the final shape can only be determined by processing the pairs sequentially in start-row order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    pairs = _extract_pairs(g)
    out, _events = _simulate_paths(g, pairs)
    return out


def _choose_start_col(width: int, turn_limit: int | None = None) -> int:
    hi = min(START_COL_MAX, width // 2 - 2)
    if turn_limit is not None:
        hi = min(hi, turn_limit - 2)
    if hi < 1:
        raise ValueError("no valid start column")
    return random.randint(1, hi)


def _choose_end_col(width: int, min_col: int, forbidden: set[int]) -> int | None:
    lo = max(width // 2 + 1, min_col)
    candidates = [c for c in range(lo, width - 1) if c not in forbidden]
    if not candidates:
        return None
    return random.choice(candidates)


def _marker_grid(height: int, width: int, pairs: list[dict]) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for pair in pairs:
        color = int(pair["color"])
        sr, sc = pair["start"]
        er, ec = pair["end"]
        grid[sr, sc] = color
        grid[er, ec] = color
    return grid


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        pairs = _extract_pairs(grid)
        solved, events = _simulate_paths(grid, pairs)
    except Exception:
        return False

    height, width = grid.shape
    marker_cells = {
        tuple(pair["start"]): (idx, "start")
        for idx, pair in enumerate(pairs)
    }
    marker_cells.update(
        {
            tuple(pair["end"]): (idx, "end")
            for idx, pair in enumerate(pairs)
        }
    )

    if max(height, width) > 30:
        return False
    if not any(event["collided_h"] for event in events):
        return False
    if not any(not event["collided_h"] for event in events):
        return False

    start_rows = []
    for event in events:
        pair = event["pair"]
        sr, sc = pair["start"]
        er, ec = pair["end"]
        if not (sc < width // 2 and ec > width // 2):
            return False
        if sr >= er:
            return False
        start_rows.append(sr)
        for cell in event["route"]:
            owner = marker_cells.get(cell)
            if owner is None:
                continue
            if cell not in (tuple(pair["start"]), tuple(pair["end"])):
                return False

    if len(set(start_rows)) != len(start_rows):
        return False

    reversed_out, _ = _simulate_paths(grid, list(reversed(pairs)))
    if np.array_equal(solved, reversed_out):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    pair_count = random.choice(PAIR_COUNT_CHOICES)
    collision_prefix = random.randint(1, pair_count - 1)
    colors = random.sample(PATH_COLORS, pair_count)

    pairs: list[dict] = []
    used_end_cols: set[int] = set()
    turn_cols: list[int] = []

    start_row = random.randint(1, 3)
    start_col = _choose_start_col(width)
    end_col = _choose_end_col(width, width // 2 + 1, used_end_cols)
    if end_col is None:
        return None
    end_row_max = min(height - 2, start_row + 7)
    if end_row_max < start_row + 4:
        return None
    end_row = random.randint(start_row + 4, end_row_max)
    pairs.append({"color": colors[0], "start": (start_row, start_col), "end": (end_row, end_col)})
    used_end_cols.add(end_col)
    turn_cols.append(end_col)

    previous_start = start_row
    previous_end = end_row
    current_turn = end_col
    max_end_so_far = end_row

    for idx in range(1, collision_prefix + 1):
        if current_turn <= 2:
            return None
        row_lo = previous_start + 1
        row_hi = previous_end - 1
        if row_hi < row_lo:
            return None
        start_row = random.randint(row_lo, row_hi)
        try:
            start_col = _choose_start_col(width, turn_limit=current_turn)
        except ValueError:
            return None
        end_col = _choose_end_col(width, current_turn + 2, used_end_cols)
        if end_col is None:
            return None
        end_row_max = min(height - 2, start_row + 7)
        if end_row_max < start_row + 3:
            return None
        end_row = random.randint(start_row + 3, end_row_max)
        pairs.append({"color": colors[idx], "start": (start_row, start_col), "end": (end_row, end_col)})
        used_end_cols.add(end_col)
        previous_start = start_row
        previous_end = end_row
        current_turn -= 1
        turn_cols.append(current_turn)
        max_end_so_far = max(max_end_so_far, end_row)

    row_floor = max_end_so_far + ROW_GAP_AFTER_CLEAR
    for idx in range(collision_prefix + 1, pair_count):
        remaining_after = pair_count - idx - 1
        max_start = height - 5 - remaining_after * 4
        if row_floor > max_start:
            return None
        start_row = random.randint(row_floor, max_start)
        start_col = _choose_start_col(width)
        forbidden = used_end_cols | set(turn_cols)
        end_col = _choose_end_col(width, width // 2 + 1, forbidden)
        if end_col is None:
            return None
        end_row_max = min(height - 2, start_row + 6)
        if end_row_max < start_row + 2:
            return None
        end_row = random.randint(start_row + 2, end_row_max)
        pairs.append({"color": colors[idx], "start": (start_row, start_col), "end": (end_row, end_col)})
        used_end_cols.add(end_col)
        turn_cols.append(end_col)
        row_floor = end_row + ROW_GAP_AFTER_CLEAR

    grid = _marker_grid(height, width, pairs)
    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid non-degenerate puzzle")
