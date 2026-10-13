import numpy as np

from arc_utils import colors_present, get_bg
import random


def _palette_columns(g: np.ndarray, bg: int) -> tuple[list[int], list[int]]:
    h, w = g.shape
    assert h > 0 and w > 0
    cols: list[int] = []
    colors: list[int] = []
    for c in range(w):
        color = int(g[0, c])
        if color != bg and np.all(g[:, c] == color):
            cols.append(c)
            colors.append(color)

    assert cols, "expected at least one full-height palette column"
    for left, right in zip(cols, cols[1:]):
        assert np.all(g[:, left + 1 : right] == bg)
    assert len(colors) == len(cols)
    return cols, colors


def _target_color(g: np.ndarray, bg: int, palette: list[int]) -> int:
    foreground = set(colors_present(g, exclude={bg}))
    target_colors = foreground - set(palette)
    assert len(target_colors) == 1, "expected one non-palette drawing color"
    return target_colors.pop()


def _column_interval(mask: np.ndarray, c: int) -> tuple[int, int]:
    rows = np.flatnonzero(mask[:, c])
    assert rows.size > 0
    r0, r1 = int(rows[0]), int(rows[-1])
    assert np.array_equal(rows, np.arange(r0, r1 + 1))
    return r0, r1


def _target_runs(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    target_cols = [int(c) for c in np.flatnonzero(mask.any(axis=0))]
    assert target_cols, "expected target columns"

    runs: list[tuple[int, int, int, int]] = []
    cur_r0, cur_r1 = _column_interval(mask, target_cols[0])
    cur_c0 = cur_c1 = target_cols[0]

    for c in target_cols[1:]:
        r0, r1 = _column_interval(mask, c)
        if c == cur_c1 + 1 and (r0, r1) == (cur_r0, cur_r1):
            cur_c1 = c
        else:
            runs.append((cur_r0, cur_c0, cur_r1, cur_c1))
            cur_r0, cur_r1 = r0, r1
            cur_c0 = cur_c1 = c

    runs.append((cur_r0, cur_c0, cur_r1, cur_c1))
    return runs


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor the right-side target column-runs with the left palette colors in order.

    Input structure
    - The background surrounds full-height, single-color palette columns on the left
      and a single-color target drawing on the right.
    - Each target column has one contiguous vertical span, and adjacent columns with
      the same span form one rectangular run.
    Rule
    - Remove the palette and target marker color, leaving the background elsewhere.
    - Read the palette columns from left to right.
    - Recolor the target column-runs from left to right with the corresponding
      palette colors.
    Key insight
    - The target drawing is not a connected-component problem: its visible blocks are
      the maximal adjacent column groups that share the same vertical extent.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    palette_cols, palette = _palette_columns(g, bg)
    target = _target_color(g, bg, palette)
    mask = g == target
    target_cols = np.flatnonzero(mask.any(axis=0))
    assert target_cols.size > 0
    assert max(palette_cols) < int(target_cols[0])

    runs = _target_runs(mask)
    assert len(runs) == len(palette)

    out = np.full_like(g, bg)
    for (r0, c0, r1, c1), color in zip(runs, palette):
        assert np.all(mask[r0 : r1 + 1, c0 : c1 + 1])
        out[r0 : r1 + 1, c0 : c1 + 1] = color

    return out

# --------------- generator (o5x) ---------------
# Canvas, palette, and target marker bounds.
HEIGHT_RANGE = (6, 8)
WIDTH_RANGE = (15, 28)
RUN_COUNT_RANGE = (3, 6)
PALETTE_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]
TARGET_COLOR = 8

# Target run geometry, spacing, and rejection bounds.
RUN_WIDTH_RANGE = (2, 4)
RUN_HEIGHT_RANGE = (2, 5)
RUN_GAP_RANGE = (0, 1)
PALETTE_TARGET_GAP_RANGE = (1, 2)
TRAILING_BG_RANGE = (0, 1)
PALETTE_STRIDE = 2
MAX_GENERATION_ATTEMPTS = 300


def _gen_bounded_sum(bounds: list[tuple[int, int]], total: int) -> list[int] | None:
    """Return a shuffled random allocation that sums to total within per-slot bounds."""
    values: list[int | None] = [None] * len(bounds)
    order = list(range(len(bounds)))
    random.shuffle(order)
    remaining = total
    for pos, idx in enumerate(order):
        rest = order[pos + 1 :]
        rest_min = sum(bounds[j][0] for j in rest)
        rest_max = sum(bounds[j][1] for j in rest)
        lo, hi = bounds[idx]
        v_min = max(lo, remaining - rest_max)
        v_max = min(hi, remaining - rest_min)
        if v_min > v_max:
            return None
        value = random.randint(v_min, v_max)
        values[idx] = value
        remaining -= value
    if remaining != 0:
        return None
    return [int(v) for v in values]


def _gen_row_spans(height: int, run_heights: list[int], run_gaps: list[int]) -> list[tuple[int, int]] | None:
    """Choose vertical spans, keeping touching runs distinct by span when gap is zero."""
    spans: list[tuple[int, int]] = []
    for i, run_height in enumerate(run_heights):
        starts = list(range(height - run_height + 1))
        if i > 0 and run_gaps[i - 1] == 0:
            prev_r0, prev_r1 = spans[-1]
            starts = [r0 for r0 in starts if (r0, r0 + run_height - 1) != (prev_r0, prev_r1)]
        if not starts:
            return None
        r0 = random.choice(starts)
        spans.append((r0, r0 + run_height - 1))
    return spans


def _gen_target_layout(height: int, n_runs: int, gap_from_palette: int, trailing_bg: int):
    """Sample target width, per-run geometry, and row spans for one candidate grid."""
    bounds = [RUN_WIDTH_RANGE] * n_runs + [RUN_GAP_RANGE] * (n_runs - 1)
    min_target_width = max(sum(lo for lo, _ in bounds), WIDTH_RANGE[0] - PALETTE_STRIDE * n_runs - gap_from_palette - trailing_bg)
    max_target_width = min(sum(hi for _, hi in bounds), WIDTH_RANGE[1] - PALETTE_STRIDE * n_runs - gap_from_palette - trailing_bg)
    if min_target_width > max_target_width:
        return None
    target_width = random.randint(min_target_width, max_target_width)
    widths_and_gaps = _gen_bounded_sum(bounds, target_width)
    if widths_and_gaps is None:
        return None
    run_widths = widths_and_gaps[:n_runs]
    run_gaps = widths_and_gaps[n_runs:]
    run_heights = [random.randint(*RUN_HEIGHT_RANGE) for _ in range(n_runs)]
    row_spans = _gen_row_spans(height, run_heights, run_gaps)
    if row_spans is None:
        return None
    return target_width, run_widths, run_gaps, run_heights, row_spans


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*HEIGHT_RANGE)
    n_runs = random.randint(*RUN_COUNT_RANGE)
    colors = random.sample(PALETTE_COLORS, n_runs)
    gap_from_palette = random.randint(*PALETTE_TARGET_GAP_RANGE)
    trailing_bg = random.randint(*TRAILING_BG_RANGE)
    layout = _gen_target_layout(height, n_runs, gap_from_palette, trailing_bg)
    if layout is None:
        return None
    target_width, run_widths, run_gaps, run_heights, row_spans = layout
    width = PALETTE_STRIDE * n_runs + gap_from_palette + target_width + trailing_bg
    target_cells = sum(run_width * run_height for run_width, run_height in zip(run_widths, run_heights))
    bg_cells = height * width - height * n_runs - target_cells
    if bg_cells <= max(target_cells, height):
        return None
    inp = np.zeros((height, width), dtype=int)
    out = np.zeros_like(inp)
    for i, color in enumerate(colors):
        inp[:, 1 + PALETTE_STRIDE * i] = color
    col = PALETTE_STRIDE * n_runs + gap_from_palette
    for (r0, r1), run_width, run_gap, color in zip(row_spans, run_widths, run_gaps + [0], colors):
        inp[r0 : r1 + 1, col : col + run_width] = TARGET_COLOR
        out[r0 : r1 + 1, col : col + run_width] = color
        col += run_width + run_gap
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 9b365c51 input")
