from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, get_bg  # type: ignore


BG = Color.BLACK
PALETTE = (
    Color.BLUE,
    Color.RED,
    Color.GREEN,
    Color.YELLOW,
    Color.GREY,
    Color.PINK,
    Color.ORANGE,
    Color.TEAL,
    Color.MAROON,
)

HEIGHT_RANGE = (10, 19)
WIDTH_RANGE = (8, 18)
MAX_GENERATION_ATTEMPTS = 500


RowSegment = tuple[int, int, int, int]


def _validate_input_structure(grid: np.ndarray, bg: int) -> None:
    h, w = grid.shape
    assert w >= 2
    for r in range(h):
        cols = np.flatnonzero(grid[r] != bg)
        if len(cols) == 0:
            continue
        assert len(cols) == 2, "every marked row must have exactly two visible markers"
        assert int(cols[0]) == 0 and int(cols[1]) == w - 1, "markers must sit on both horizontal borders"


def _solve(grid: np.ndarray, *, with_trace: bool = False):
    g = np.asarray(grid, dtype=int).copy()
    assert g.ndim == 2 and g.size > 0
    bg = int(get_bg(g))
    assert bg == BG
    _validate_input_structure(g, bg)

    h, w = g.shape
    row_segments: list[RowSegment] = []
    row_segment_keys: set[RowSegment] = set()

    def add_row_segment(r: int, c0: int, c1: int, color: int) -> bool:
        key = (int(r), int(c0), int(c1), int(color))
        if key in row_segment_keys:
            return False
        row_segment_keys.add(key)
        row_segments.append(key)
        return True

    seed_rows: list[tuple[int, int]] = []
    initial_row_paint = False
    for r in range(h):
        if g[r, 0] == bg and g[r, w - 1] == bg:
            continue
        left = int(g[r, 0])
        right = int(g[r, w - 1])
        if left == right and left != bg:
            seed_rows.append((int(r), left))
            mask = g[r] == bg
            if np.any(mask):
                g[r, mask] = left
                initial_row_paint = True
            add_row_segment(r, 0, w - 1, left)

    after_seed_rows = g.copy()
    painted_columns: set[int] = set()
    painted_rows: set[int] = set()
    rounds = 0
    saw_column_paint = False
    saw_row_paint_after_column = False

    while True:
        changed = False

        for c in range(w):
            endpoint_rows = sorted({seg[0] for seg in row_segments if seg[1] == c or seg[2] == c})
            if len(endpoint_rows) < 2:
                continue
            top = int(endpoint_rows[0])
            bottom = int(endpoint_rows[-1])
            color = int(g[top, c])
            if color == bg or int(g[bottom, c]) != color:
                continue
            col = g[top : bottom + 1, c]
            mask = col == bg
            if np.any(mask):
                col[mask] = color
                painted_columns.add(int(c))
                saw_column_paint = True
                changed = True

        for r in range(h):
            cols = np.flatnonzero(g[r] != bg)
            if len(cols) < 2:
                continue
            c0 = int(cols[0])
            c1 = int(cols[-1])
            color = int(g[r, c0])
            if int(g[r, c1]) != color:
                continue

            row = g[r, c0 : c1 + 1]
            mask = row == bg
            if np.any(mask):
                row[mask] = color
                painted_rows.add(int(r))
                if saw_column_paint:
                    saw_row_paint_after_column = True
                changed = True

            if add_row_segment(r, c0, c1, color):
                changed = True

        if not changed:
            break
        rounds += 1

    if not with_trace:
        return g

    return g, {
        "seed_rows": list(seed_rows),
        "after_seed_rows": after_seed_rows,
        "painted_columns": sorted(painted_columns),
        "painted_rows": sorted(painted_rows),
        "rounds": int(rounds),
        "initial_row_paint": bool(initial_row_paint),
        "saw_column_paint": bool(saw_column_paint),
        "saw_row_paint_after_column": bool(saw_row_paint_after_column),
        "row_segments": list(row_segments),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Cascade same-color border equations until stable by alternating row bars and column bars.

    Input structure:
    - The background is the dominant color.
    - The only initial non-background pixels are paired row markers on the left and right canvas borders, with either zero or two visible markers on each row.
    - No column is pre-marked; every later vertical cue must be induced by a previously authorized horizontal bar.

    Rule:
    - First fill every row whose two border markers have the same color.
    - Treat each filled horizontal bar as creating column endpoints only at its two ends; if some column has a topmost and bottommost such endpoint of the same color, fill the background cells between them in that column.
    - After each column step, any row whose leftmost and rightmost visible non-background pixels now match fills between those two pixels; never overwrite already-colored cells, and alternate until no new cells or row bars appear.

    Key insight:
    - Equality lives at segment ends, not at isolated markers: a solved row manufactures vertical evidence at its two ends, those columns expose new left-right matches on other rows, and the final picture is the fixpoint of that alternating endpoint logic rather than a one-pass scan.
    """
    return _solve(grid, with_trace=False)


def _rows_only(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int).copy()
    bg = int(get_bg(g))
    h, w = g.shape
    for r in range(h):
        if g[r, 0] != bg and int(g[r, 0]) == int(g[r, w - 1]):
            g[r, :] = int(g[r, 0])
    return g


def _boundary_only(grid: np.ndarray) -> np.ndarray:
    g = _rows_only(grid)
    h, w = g.shape
    active_rows = [r for r in range(h) if np.all(g[r] == g[r, 0]) and int(g[r, 0]) != BG]
    if len(active_rows) < 2:
        return g
    for c in (0, w - 1):
        top = int(active_rows[0])
        bottom = int(active_rows[-1])
        color = int(g[top, c])
        if color != BG and int(g[bottom, c]) == color:
            col = g[top : bottom + 1, c]
            col[col == BG] = color
    return g


def _outer_rectangle_shortcut(grid: np.ndarray) -> np.ndarray:
    g = _rows_only(grid)
    h, w = g.shape
    active = [(r, int(g[r, 0])) for r in range(h) if np.all(g[r] == g[r, 0]) and int(g[r, 0]) != BG]
    if len(active) < 2:
        return g
    top_r, top_color = active[0]
    bottom_r, bottom_color = active[-1]
    if top_color != bottom_color:
        return g
    patch = g[top_r : bottom_r + 1, :]
    patch[patch == BG] = top_color
    return g


def _distinct_colors(exclude: set[int], count: int) -> list[int]:
    choices = [color for color in PALETTE if color not in exclude]
    random.shuffle(choices)
    return choices[:count]


def _choose_blocker_pair(chain_color: int) -> tuple[int, int]:
    mode = random.choice(("one_side_chain", "no_chain"))
    if mode == "one_side_chain":
        other = random.choice([c for c in PALETTE if c != chain_color])
        if random.random() < 0.5:
            return int(chain_color), int(other)
        return int(other), int(chain_color)
    left, right = random.sample([c for c in PALETTE if c != chain_color], 2)
    return int(left), int(right)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*HEIGHT_RANGE)
    w = random.randint(*WIDTH_RANGE)
    grid = np.full((h, w), BG, dtype=int)

    chain_color = random.choice(PALETTE)
    outer_top = random.randint(1, h - 6)
    outer_bottom = random.randint(outer_top + 4, h - 2)
    inside_rows = list(range(outer_top + 1, outer_bottom))
    if len(inside_rows) < 3:
        return None

    blocker_count = random.randint(1, min(3, len(inside_rows) - 1))
    blocker_rows = sorted(random.sample(inside_rows, blocker_count))
    remaining = [r for r in inside_rows if r not in blocker_rows]
    if not remaining:
        return None

    inner_same_rows: list[int] = []
    if len(remaining) >= 2 and random.random() < 0.35:
        chosen = random.choice(remaining)
        inner_same_rows = [int(chosen)]
        remaining = [r for r in remaining if r != chosen]

    inner_other_match_rows: list[int] = []
    max_other_match = min(2, max(0, len(remaining) - 1))
    if max_other_match > 0 and random.random() < 0.8:
        count = random.randint(1, max_other_match)
        inner_other_match_rows = sorted(random.sample(remaining, count))
        remaining = [r for r in remaining if r not in inner_other_match_rows]

    plain_rows = [r for r in range(outer_top + 1, outer_bottom) if r not in blocker_rows and r not in inner_same_rows and r not in inner_other_match_rows]
    if not plain_rows:
        return None

    outside_rows = [r for r in range(1, outer_top)] + [r for r in range(outer_bottom + 1, h - 1)]
    random.shuffle(outside_rows)
    outer_noise_rows = sorted(outside_rows[: random.randint(0, min(3, len(outside_rows)))])

    assignments: dict[int, tuple[int, int]] = {
        int(outer_top): (int(chain_color), int(chain_color)),
        int(outer_bottom): (int(chain_color), int(chain_color)),
    }
    for r in inner_same_rows:
        assignments[int(r)] = (int(chain_color), int(chain_color))

    other_match_colors = _distinct_colors({chain_color}, len(inner_other_match_rows))
    if len(other_match_colors) != len(inner_other_match_rows):
        return None
    for r, color in zip(inner_other_match_rows, other_match_colors):
        assignments[int(r)] = (int(color), int(color))

    for r in blocker_rows:
        assignments[int(r)] = _choose_blocker_pair(chain_color)
    for r in outer_noise_rows:
        left, right = random.sample(PALETTE, 2)
        assignments[int(r)] = (int(left), int(right))

    for r, (left, right) in assignments.items():
        grid[int(r), 0] = int(left)
        grid[int(r), w - 1] = int(right)

    out, trace = _solve(grid, with_trace=True)

    if trace["painted_columns"] != [0, w - 1]:
        return None
    if not trace["saw_column_paint"] or not trace["saw_row_paint_after_column"]:
        return None
    if not set(plain_rows) & set(trace["painted_rows"]):
        return None
    if not any(np.all(out[r, 1 : w - 1] == BG) for r in blocker_rows):
        return None

    rows_only = _rows_only(grid)
    boundary_only = _boundary_only(grid)
    outer_rectangle = _outer_rectangle_shortcut(grid)
    if np.array_equal(out, rows_only):
        return None
    if np.array_equal(out, boundary_only):
        return None
    if np.array_equal(out, outer_rectangle):
        return None

    if not any(np.all(out[r] == chain_color) for r in plain_rows):
        return None
    if np.count_nonzero(out == BG) >= np.count_nonzero(grid == BG):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid chained-equations input")
