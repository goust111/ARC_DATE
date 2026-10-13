from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, get_bg


BG = Color.BLACK
PATTERN_COLOR = Color.GREY
RECOLOR_SEED = Color.GREEN
INVERT_SEED = Color.RED
SEED_COLORS = {RECOLOR_SEED, INVERT_SEED}
ALLOWED_INPUT_COLORS = {BG, PATTERN_COLOR, RECOLOR_SEED, INVERT_SEED}

# --------------- solver helpers ---------------


def _seed_rows(g: np.ndarray) -> list[tuple[int, int]]:
    rows: list[tuple[int, int]] = []
    for r in range(g.shape[0]):
        vals = {int(v) for v in np.unique(g[r])}
        if len(vals) == 1 and next(iter(vals)) in SEED_COLORS:
            rows.append((r, next(iter(vals))))
            continue

        assert vals <= {BG, PATTERN_COLOR}, "non-seed rows must use only black and gray"
        assert PATTERN_COLOR in vals, "every non-seed row must contain a gray pattern"

    assert len(rows) >= 2, "expected at least two seed bars"
    assert {color for _r, color in rows} == SEED_COLORS, "both seed colors must appear"
    return rows


def _row_operation(seed_rows: list[tuple[int, int]], r: int) -> tuple[str, int | None, bool]:
    nearest_dist = min(abs(r - sr) for sr, _color in seed_rows)
    nearest = [(sr, color) for sr, color in seed_rows if abs(r - sr) == nearest_dist]
    colors = {color for _sr, color in nearest}
    if len(colors) > 1:
        return "clear", None, True
    color = next(iter(colors))
    return ("recolor" if color == RECOLOR_SEED else "invert"), color, len(nearest) > 1


def _expected_output(g: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    assert g.ndim == 2, "expected a 2D grid"
    assert get_bg(g) == BG, "expected black background"
    assert {int(v) for v in np.unique(g)} <= ALLOWED_INPUT_COLORS, "unexpected input colors"

    seed_rows = _seed_rows(g)
    seed_row_ids = {r for r, _color in seed_rows}
    seen_patterns: set[tuple[int, ...]] = set()

    out = np.full_like(g, BG)
    stats = {
        "recolor": 0,
        "invert": 0,
        "clear": 0,
        "same_tie": 0,
        "diff_tie": 0,
    }

    for r, color in seed_rows:
        out[r, :] = color

    for r in range(g.shape[0]):
        if r in seed_row_ids:
            continue

        pattern_cols = tuple(int(c) for c in np.flatnonzero(g[r] == PATTERN_COLOR))
        assert pattern_cols, "every non-seed row must have a non-empty gray pattern"
        assert len(pattern_cols) < g.shape[1], "gray pattern may not occupy the whole row"
        assert pattern_cols not in seen_patterns, "row patterns must be pairwise distinct"
        seen_patterns.add(pattern_cols)

        op, _color, is_tie = _row_operation(seed_rows, r)
        if op == "clear":
            stats["clear"] += 1
            stats["diff_tie"] += 1
            continue

        if is_tie:
            stats["same_tie"] += 1

        if op == "recolor":
            out[r, list(pattern_cols)] = RECOLOR_SEED
            stats["recolor"] += 1
        else:
            out[r, :] = INVERT_SEED
            out[r, list(pattern_cols)] = BG
            stats["invert"] += 1

    return out, stats


def _naive_recolor_only(g: np.ndarray) -> np.ndarray:
    seed_rows = _seed_rows(g)
    seed_row_ids = {r for r, _color in seed_rows}
    out = np.full_like(g, BG)
    for r, color in seed_rows:
        out[r, :] = color
    for r in range(g.shape[0]):
        if r in seed_row_ids:
            continue
        op, color, _is_tie = _row_operation(seed_rows, r)
        if op == "clear":
            continue
        out[r, g[r] == PATTERN_COLOR] = color
    return out


def _naive_full_fill(g: np.ndarray) -> np.ndarray:
    seed_rows = _seed_rows(g)
    seed_row_ids = {r for r, _color in seed_rows}
    out = np.full_like(g, BG)
    for r, color in seed_rows:
        out[r, :] = color
    for r in range(g.shape[0]):
        if r in seed_row_ids:
            continue
        op, color, _is_tie = _row_operation(seed_rows, r)
        if op == "clear":
            continue
        out[r, :] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Partition rows by nearest seed bars, with green recoloring, red inversion, and opposite-color midlines erased.

    Input structure
    - The background is black.
    - Some full-width seed rows are solid green or red bars.
    - Every other row contains a unique sparse gray pattern and no other foreground color.

    Rule
    - Preserve each seed bar row unchanged.
    - For every non-bar row, find the nearest seed bar by vertical distance.
    - If the nearest bars are tied and have different colors, clear the whole row to black.
    - Otherwise use the unique nearest color, or the tied shared color if both nearest bars match.
    - Green selects recolor: copy the gray pattern shape in green on black.
    - Red selects inversion: turn gray pattern cells to black and fill every other cell in that row with red.

    Key insight
    - Seed color chooses the row operation rather than merely the paint color, so one nearest-seed partition yields sparse copies, dense complements, and blank separator rows.
    """
    out, _stats = _expected_output(np.asarray(grid, dtype=int))
    return out


# --------------- generator ---------------

WIDTH_RANGE = (10, 18)
BAR_COUNT_RANGE = (4, 5)
EDGE_MARGIN_RANGE = (1, 3)
GAP_RANGE = (1, 4)  # number of non-bar rows between adjacent seed bars
PATTERN_SPAN_MAX = 8
PATTERN_PIXELS_MAX = 5
GEN_ATTEMPTS = 500


def _sample_seed_colors(count: int) -> list[int]:
    while True:
        colors = [random.choice((RECOLOR_SEED, INVERT_SEED)) for _ in range(count)]
        if len(set(colors)) < 2:
            continue
        same_adj = any(colors[i] == colors[i + 1] for i in range(count - 1))
        diff_adj = any(colors[i] != colors[i + 1] for i in range(count - 1))
        if same_adj and diff_adj:
            return colors


def _sample_gaps(colors: list[int]) -> list[int]:
    while True:
        gaps = [random.randint(*GAP_RANGE) for _ in range(len(colors) - 1)]
        same_odd = any(
            gaps[i] % 2 == 1 and colors[i] == colors[i + 1]
            for i in range(len(gaps))
        )
        diff_odd = any(
            gaps[i] % 2 == 1 and colors[i] != colors[i + 1]
            for i in range(len(gaps))
        )
        if same_odd and diff_odd:
            return gaps


def _bar_rows(top_margin: int, gaps: list[int]) -> list[int]:
    rows = [top_margin]
    for gap in gaps:
        rows.append(rows[-1] + gap + 1)
    return rows


def _sample_pattern(width: int, used: set[tuple[int, ...]]) -> tuple[int, ...] | None:
    for _ in range(300):
        max_span = min(PATTERN_SPAN_MAX, width)
        span = random.randint(4, max_span)
        start_min = 1 if width - span >= 2 else 0
        start_max = width - span - 1 if width - span >= 2 else width - span
        start = random.randint(start_min, start_max)
        cells = random.randint(2, min(PATTERN_PIXELS_MAX, span - 1))
        cols = tuple(sorted(random.sample(range(start, start + span), cells)))
        if cols[-1] - cols[0] + 1 == len(cols):
            continue
        if cols in used:
            continue
        return cols
    return None


def _render_input(width: int, bar_rows: list[int], bar_colors: list[int], height: int) -> np.ndarray | None:
    grid = np.full((height, width), BG, dtype=int)
    for r, color in zip(bar_rows, bar_colors):
        grid[r, :] = color

    used_patterns: set[tuple[int, ...]] = set()
    bar_row_ids = set(bar_rows)
    for r in range(height):
        if r in bar_row_ids:
            continue
        pattern = _sample_pattern(width, used_patterns)
        if pattern is None:
            return None
        used_patterns.add(pattern)
        grid[r, list(pattern)] = PATTERN_COLOR

    return grid


def _validate_semantics(inp: np.ndarray) -> bool:
    try:
        out, stats = _expected_output(inp)
    except AssertionError:
        return False

    if stats["recolor"] < 2 or stats["invert"] < 2:
        return False
    if stats["clear"] < 1 or stats["same_tie"] < 1 or stats["diff_tie"] < 1:
        return False

    if np.array_equal(out, _naive_recolor_only(inp)):
        return False
    if np.array_equal(out, _naive_full_fill(inp)):
        return False

    seed_rows = {r for r, _color in _seed_rows(inp)}
    saw_sparse_green = False
    saw_dense_red = False
    for r in range(inp.shape[0]):
        if r in seed_rows:
            if not np.array_equal(inp[r], out[r]):
                return False
            continue
        gray_count = int(np.sum(inp[r] == PATTERN_COLOR))
        out_count = int(np.sum(out[r] != BG))
        op, _color, is_tie = _row_operation(_seed_rows(inp), r)
        if op == "clear":
            if out_count != 0:
                return False
            continue
        if op == "recolor":
            if out_count != gray_count:
                return False
            saw_sparse_green = True
        else:
            if out_count != inp.shape[1] - gray_count:
                return False
            saw_dense_red = True
        if is_tie and op == "recolor" and np.any(out[r] == INVERT_SEED):
            return False

    if not (saw_sparse_green and saw_dense_red):
        return False

    row_signatures = {tuple(int(v) for v in row) for row in out}
    return len(row_signatures) >= 6


def _gen_attempt() -> np.ndarray | None:
    width = random.randint(*WIDTH_RANGE)
    bar_count = random.randint(*BAR_COUNT_RANGE)
    top_margin = random.randint(*EDGE_MARGIN_RANGE)
    bottom_margin = random.randint(*EDGE_MARGIN_RANGE)
    bar_colors = _sample_seed_colors(bar_count)
    gaps = _sample_gaps(bar_colors)
    bar_rows = _bar_rows(top_margin, gaps)
    height = bar_rows[-1] + 1 + bottom_margin
    if height > 30:
        return None

    inp = _render_input(width, bar_rows, bar_colors, height)
    if inp is None:
        return None
    if not _validate_semantics(inp):
        return None
    if not np.array_equal(main(inp), _expected_output(inp)[0]):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
