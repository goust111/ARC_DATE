from __future__ import annotations

from importlib.machinery import SourceFileLoader
import random

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "arc_utils_fable_8_028_1", "/home/lixinyu/snap/new/solvers/arc_utils.py"
).load_module()
Color = _ARC_UTILS.Color


BG = Color.BLACK
GREY = Color.GREY
STRIKER_COLORS = [c for c in Color.NOT_BLACK if c != GREY]

MIN_HEIGHT = 10
MAX_HEIGHT = 16
MIN_WIDTH = 11
MAX_WIDTH = 16
MIN_STRIKERS = 3
MAX_STRIKERS = 4
MAX_GENERATION_ATTEMPTS = 400


def _find_strikers(grid: np.ndarray) -> list[tuple[int, int, int]]:
    """Return (row, target_col, striker_color) for each right-edge striker."""
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    events: list[tuple[int, int, int]] = []
    for r in range(height):
        color = int(g[r, width - 1])
        if color in (BG, GREY):
            continue
        target_col = None
        for c in range(width - 2, -1, -1):
            if int(g[r, c]) == GREY:
                target_col = c
                break
        assert target_col is not None, "each striker row must contain at least one grey block"
        events.append((r, target_col, color))
    return events


def _landing_row(static_grid: np.ndarray, start_r: int, col: int) -> int:
    """Drop from (start_r, col) inside a grid where the knocked block is already removed."""
    below = np.flatnonzero(static_grid[start_r + 1 :, col] == GREY)
    if below.size == 0:
        return int(static_grid.shape[0] - 1)
    return int(start_r + below[0])


def _expected_from_metadata(
    grid: np.ndarray, events: list[dict[str, object]]
) -> np.ndarray:
    """Independent oracle used by the generator: apply the planned events directly."""
    out = np.asarray(grid, dtype=int).copy()

    for evt in events:
        r = int(evt["row"])
        c = int(evt["target_col"])
        out[r, -1] = BG
        out[r, c] = BG

    for evt in events:
        r = int(evt["row"])
        c = int(evt["target_col"])
        color = int(evt["color"])
        landing_r = _landing_row(out, r, c)
        assert int(out[landing_r, c]) == BG, "landing cell must be empty after removing the struck block"
        out[landing_r, c] = color

    return out


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Slide each right-edge striker into the first grey block on its row, then drop and recolor that block.

    Input structure:
    - The background is black.
    - Grey cells are scattered single-cell blocks, possibly sharing columns but not acting as strikers.
    - Some rows end with one non-grey colored pixel on the far-right edge; each such row also contains at least one grey block to its left.

    Rule:
    - For every right-edge colored striker, scan left on the same row and find the first grey block it would hit.
    - Remove the striker, remove that grey block from its row, and let the struck block fall straight down in its own column until it reaches the bottom edge or the cell just above the next grey block below.
    - Recolor the fallen block with the striker's color and leave every other cell unchanged.

    Key insight:
    - The moving object is not the striker pixel itself: the striker only selects a row, transfers its color, and knocks the first same-row grey block downward in that block's original column.
    """
    g = np.asarray(input_grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    events = _find_strikers(g)
    out = g.copy()

    for r, target_c, _color in events:
        out[r, out.shape[1] - 1] = BG
        out[r, target_c] = BG

    for r, target_c, color in events:
        landing_r = _landing_row(out, r, target_c)
        out[landing_r, target_c] = int(color)

    return out


def _choose_columns(width: int, count: int) -> list[int]:
    candidates = list(range(2, width - 1))
    for _ in range(200):
        cols = sorted(random.sample(candidates, count))
        if all(b - a >= 2 for a, b in zip(cols, cols[1:])):
            return cols
    raise RuntimeError("failed to sample separated target columns")


def _can_add_grey(greys: set[tuple[int, int]], r: int, c: int, height: int, width: int) -> bool:
    if not (0 <= r < height and 0 <= c < width - 1):
        return False
    if (r, c) in greys:
        return False
    for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
        if (nr, nc) in greys:
            return False
    return True


def _pick_support_rows(
    greys: set[tuple[int, int]],
    striker_rows: set[int],
    height: int,
    width: int,
    row: int,
    col: int,
) -> list[int] | None:
    candidates = [rr for rr in range(row + 2, height) if rr not in striker_rows]
    random.shuffle(candidates)
    for support_r in candidates:
        if _can_add_grey(greys, support_r, col, height, width):
            rows = [support_r]
            if random.random() < 0.45:
                deeper = [
                    rr
                    for rr in range(support_r + 2, height)
                    if rr not in striker_rows and _can_add_grey(greys, rr, col, height, width)
                ]
                if deeper:
                    rows.append(random.choice(deeper))
            return sorted(rows)
    return None


def _add_left_distractors(
    greys: set[tuple[int, int]],
    height: int,
    width: int,
    row: int,
    target_col: int,
    count_range: tuple[int, int],
) -> int:
    placed = 0
    count = random.randint(*count_range)
    candidates = list(range(0, target_col - 1))
    random.shuffle(candidates)
    for c in candidates:
        if _can_add_grey(greys, row, c, height, width):
            greys.add((row, c))
            placed += 1
            if placed == count:
                break
    return placed


def _place_global_distractors(
    greys: set[tuple[int, int]],
    height: int,
    width: int,
    striker_rows: set[int],
    target_columns: set[int],
) -> None:
    for _ in range(random.randint(1, 5)):
        candidates = [
            (r, c)
            for r in range(height)
            for c in range(width - 1)
            if r not in striker_rows and c not in target_columns and _can_add_grey(greys, r, c, height, width)
        ]
        if not candidates:
            return
        greys.add(random.choice(candidates))


def _metadata_from_grid(grid: np.ndarray) -> list[dict[str, object]]:
    events = []
    for r, target_c, color in _find_strikers(grid):
        events.append({"row": r, "target_col": target_c, "color": color})
    return events


def _build_candidate() -> tuple[np.ndarray, np.ndarray] | None:
    height = random.randint(MIN_HEIGHT, MAX_HEIGHT)
    width = random.randint(MIN_WIDTH, MAX_WIDTH)
    striker_count = random.randint(MIN_STRIKERS, MAX_STRIKERS)

    bottom_row = height - 1
    stack_row = random.randint(1, height - 5)
    other_rows_pool = [r for r in range(height - 1) if r != stack_row]
    extra_rows = random.sample(other_rows_pool, striker_count - 2)
    cols = _choose_columns(width, striker_count)
    colors = random.sample(STRIKER_COLORS, striker_count)

    events: list[dict[str, object]] = []
    events.append(
        {
            "row": bottom_row,
            "target_col": cols[0],
            "color": colors[0],
            "branch": "stay_on_floor",
            "supports": [],
        }
    )
    events.append(
        {
            "row": stack_row,
            "target_col": cols[1],
            "color": colors[1],
            "branch": "stack_stop",
            "supports": [],
        }
    )
    remaining_rows = extra_rows
    remaining_cols = cols[2:]
    remaining_colors = colors[2:]
    remaining_branches = ["fall_to_floor"]
    while len(remaining_branches) < len(remaining_rows):
        remaining_branches.append(random.choice(["stack_stop", "fall_to_floor"]))
    random.shuffle(remaining_branches)
    for row, col, color, branch in zip(remaining_rows, remaining_cols, remaining_colors, remaining_branches):
        events.append({"row": row, "target_col": col, "color": color, "branch": branch, "supports": []})

    greys: set[tuple[int, int]] = set()
    striker_row_set = {int(evt["row"]) for evt in events}
    for evt in events:
        r = int(evt["row"])
        c = int(evt["target_col"])
        if not _can_add_grey(greys, r, c, height, width):
            return None
        greys.add((r, c))

    for evt in events:
        if evt["branch"] != "stack_stop":
            continue
        support_rows = _pick_support_rows(
            greys=greys,
            striker_rows=striker_row_set,
            height=height,
            width=width,
            row=int(evt["row"]),
            col=int(evt["target_col"]),
        )
        if support_rows is None:
            return None
        for support_r in support_rows:
            greys.add((support_r, int(evt["target_col"])))
        evt["supports"] = support_rows

    extra_hit_rows = 0
    for evt in events:
        placed = _add_left_distractors(
            greys=greys,
            height=height,
            width=width,
            row=int(evt["row"]),
            target_col=int(evt["target_col"]),
            count_range=(0, 2),
        )
        if placed > 0:
            extra_hit_rows += 1
    if extra_hit_rows == 0:
        anchor_evt = random.choice(events)
        if _add_left_distractors(
            greys=greys,
            height=height,
            width=width,
            row=int(anchor_evt["row"]),
            target_col=int(anchor_evt["target_col"]),
            count_range=(1, 1),
        ) == 0:
            return None

    _place_global_distractors(
        greys=greys,
        height=height,
        width=width,
        striker_rows=striker_row_set,
        target_columns={int(evt["target_col"]) for evt in events},
    )

    grid = np.full((height, width), BG, dtype=int)
    for r, c in greys:
        grid[r, c] = GREY
    for evt in events:
        grid[int(evt["row"]), width - 1] = int(evt["color"])

    metadata = _metadata_from_grid(grid)
    if len(metadata) != striker_count:
        return None

    branch_hits = {"stay": 0, "stack": 0, "floor": 0}
    for evt in metadata:
        r = int(evt["row"])
        c = int(evt["target_col"])
        row_greys = [cc for cc in range(width - 1) if int(grid[r, cc]) == GREY]
        if not row_greys or c != max(row_greys):
            return None

        static = grid.copy()
        static[r, width - 1] = BG
        static[r, c] = BG
        landing_r = _landing_row(static, r, c)
        if r == height - 1 and landing_r == r:
            branch_hits["stay"] += 1
        elif landing_r == height - 1:
            branch_hits["floor"] += 1
        else:
            branch_hits["stack"] += 1

    if min(branch_hits.values()) < 1:
        return None

    expected = _expected_from_metadata(grid, metadata)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(actual, grid):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        candidate = _build_candidate()
        if candidate is not None:
            grid, _expected = candidate
            return grid
    raise RuntimeError("failed to generate a valid input for the striker-chain puzzle")
