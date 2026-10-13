from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
import random

import numpy as np


BG = 0
MARKER = 2
YELLOW = 4
GREY = 5
BAR_COLORS = (1, 3, 6, 7)
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GRID_SIDE_RANGE = (12, 16)
BAR_COUNT_CHOICES = (3, 4)
EDGE_MARGIN = 2
MIN_BAND_GAP = 1
MARKER_COUNT_CHOICES = (2, 2, 3)
GEN_ATTEMPTS = 400


@dataclass(frozen=True)
class Bar:
    orientation: str
    start: int
    end: int
    color: int


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, value in enumerate(mask.astype(bool).tolist()):
        if value and start is None:
            start = i
        elif not value and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _modal_nonzero(values: np.ndarray) -> int:
    counts = np.bincount(values.ravel(), minlength=10)
    counts[BG] = 0
    color = int(np.argmax(counts))
    assert counts[color] > 0, "expected a non-background modal color"
    return color


def _infer_bars(grid: np.ndarray) -> list[Bar]:
    g = np.asarray(grid, dtype=int)
    row_bands = _runs(np.all(g != BG, axis=1))
    col_bands = _runs(np.all(g != BG, axis=0))

    bars: list[Bar] = []
    for r0, r1 in row_bands:
        bars.append(Bar("H", int(r0), int(r1), _modal_nonzero(g[r0 : r1 + 1, :])))
    for c0, c1 in col_bands:
        bars.append(Bar("V", int(c0), int(c1), _modal_nonzero(g[:, c0 : c1 + 1])))

    assert 3 <= len(bars) <= 4, "expected 3 or 4 full-span bars"
    assert any(bar.orientation == "H" for bar in bars), "expected at least one horizontal bar"
    assert any(bar.orientation == "V" for bar in bars), "expected at least one vertical bar"
    assert len({bar.color for bar in bars}) == len(bars), "bar colors must be distinct"
    return bars


def _crossing_constraints(grid: np.ndarray, bars: list[Bar]) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    constraints: list[tuple[int, int]] = []
    horizontals = [(i, bar) for i, bar in enumerate(bars) if bar.orientation == "H"]
    verticals = [(i, bar) for i, bar in enumerate(bars) if bar.orientation == "V"]

    for hi, hbar in horizontals:
        for vi, vbar in verticals:
            patch = g[hbar.start : hbar.end + 1, vbar.start : vbar.end + 1]
            visible = set(map(int, np.unique(patch)))
            assert len(visible) == 1, "each bar crossing must show exactly one visible color"
            color = next(iter(visible))
            if color == hbar.color:
                constraints.append((vi, hi))
            elif color == vbar.color:
                constraints.append((hi, vi))
            else:
                raise AssertionError("crossing color must belong to one of the two crossing bars")
    return constraints


def _infer_layer_order(grid: np.ndarray, bars: list[Bar]) -> list[Bar]:
    constraints = _crossing_constraints(grid, bars)
    candidates: list[tuple[int, ...]] = []
    for order in permutations(range(len(bars))):
        pos = {bar_idx: rank for rank, bar_idx in enumerate(order)}
        if all(pos[lo] < pos[hi] for lo, hi in constraints):
            candidates.append(order)
    assert len(candidates) == 1, "crossing colors must determine a unique total layer order"
    return [bars[i] for i in candidates[0]]


def _paint_trail_cell(out: np.ndarray, r: int, c: int, color: int) -> None:
    current = int(out[r, c])
    if color == YELLOW:
        if current in (BG, GREY):
            out[r, c] = YELLOW
    elif color == GREY:
        if current == BG:
            out[r, c] = GREY
    else:
        raise AssertionError("unexpected trail color")


def _trace_rays(grid: np.ndarray, top_color: int) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    height, width = g.shape

    for r, c in np.argwhere(g == MARKER):
        for dr, dc in DIRS:
            after_top = False
            rr, cc = int(r + dr), int(c + dc)
            while 0 <= rr < height and 0 <= cc < width:
                value = int(g[rr, cc])
                if value == BG:
                    _paint_trail_cell(out, rr, cc, YELLOW if after_top else GREY)
                    rr += dr
                    cc += dc
                    continue
                if value == top_color:
                    after_top = True
                    rr += dr
                    cc += dc
                    continue
                break
    return out


def _trace_ray_events(
    grid: np.ndarray,
    start: tuple[int, int],
    direction: tuple[int, int],
    top_color: int,
) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    dr, dc = direction
    rr, cc = start[0] + dr, start[1] + dc
    after_top = False
    grey_cells: list[tuple[int, int]] = []
    top_cells: list[tuple[int, int]] = []
    yellow_cells: list[tuple[int, int]] = []
    stop_color: int | None = None

    while 0 <= rr < height and 0 <= cc < width:
        value = int(g[rr, cc])
        if value == BG:
            if after_top:
                yellow_cells.append((rr, cc))
            else:
                grey_cells.append((rr, cc))
            rr += dr
            cc += dc
            continue
        if value == top_color:
            top_cells.append((rr, cc))
            after_top = True
            rr += dr
            cc += dc
            continue
        stop_color = value
        break

    return {
        "grey": grey_cells,
        "top": top_cells,
        "yellow": yellow_cells,
        "stop_color": stop_color,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Infer the stripe stacking order from crossing colors, then let only the top stripe be transparent to rays.

    Input structure:
    - Black background contains 3 or 4 full-span bars: some horizontal row bands and some vertical column bands, each with width 1 or 2.
    - Bar colors come from a fixed non-red palette, while 2 or 3 red singleton markers sit on background cells.
    - Each horizontal/vertical crossing shows the visible color of the higher-layer bar.

    Rule:
    - Detect every full-span bar and infer the unique total layer order from the crossing colors.
    - Treat only the single highest bar as transparent to marker rays; every other non-background cell blocks rays.
    - From each red marker, cast rays in the four cardinal directions: paint pre-top background cells grey, keep the top bar unchanged while passing through it, and paint post-top background cells yellow until a lower bar or the border stops the ray.

    Key insight:
    - The crossing color encodes which bar is above, and that inferred top bar is the only one that rays may penetrate.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must stay within 30x30"

    bars = _infer_bars(g)
    ordered = _infer_layer_order(g, bars)
    return _trace_rays(g, ordered[-1].color)


def _sample_bands(length: int, count: int) -> list[tuple[int, int]] | None:
    widths = [random.randint(1, 2) for _ in range(count)]
    required = 2 * EDGE_MARGIN + sum(widths) + MIN_BAND_GAP * max(0, count - 1)
    if required > length:
        return None

    extra = length - required
    slack = [0] * (count + 1)
    for _ in range(extra):
        slack[random.randrange(count + 1)] += 1

    bands: list[tuple[int, int]] = []
    pos = EDGE_MARGIN + slack[0]
    for i, width in enumerate(widths):
        start = pos
        end = pos + width - 1
        bands.append((start, end))
        pos = end + 1
        if i < count - 1:
            pos += MIN_BAND_GAP + slack[i + 1]
    return bands


def _background_axes(shape: tuple[int, int], bars: list[Bar]) -> tuple[list[int], list[int]]:
    height, width = shape
    occupied_rows = {
        r
        for bar in bars
        if bar.orientation == "H"
        for r in range(bar.start, bar.end + 1)
    }
    occupied_cols = {
        c
        for bar in bars
        if bar.orientation == "V"
        for c in range(bar.start, bar.end + 1)
    }
    bg_rows = [r for r in range(height) if r not in occupied_rows]
    bg_cols = [c for c in range(width) if c not in occupied_cols]
    return bg_rows, bg_cols


def _render_bars(shape: tuple[int, int], bars: list[Bar]) -> np.ndarray:
    height, width = shape
    grid = np.full((height, width), BG, dtype=int)
    for bar in bars:
        if bar.orientation == "H":
            grid[bar.start : bar.end + 1, :] = bar.color
        else:
            grid[:, bar.start : bar.end + 1] = bar.color
    return grid


def _orientation_sequence(count: int) -> list[str]:
    first = random.choice(["H", "V"])
    second = "V" if first == "H" else "H"
    return [first if i % 2 == 0 else second for i in range(count)]


def _choose_witness_marker(
    shape: tuple[int, int],
    bars: list[Bar],
) -> tuple[tuple[int, int], tuple[int, int], Bar] | None:
    height, width = shape
    top_bar = bars[-1]
    lower_same = next(bar for bar in reversed(bars[:-1]) if bar.orientation == top_bar.orientation)
    bg_rows, bg_cols = _background_axes(shape, bars)

    if top_bar.orientation == "H":
        col = random.choice(bg_cols)
        if lower_same.end < top_bar.start:
            candidate_rows = [r for r in bg_rows if r >= top_bar.end + 2]
            if not candidate_rows:
                return None
            row = max(candidate_rows)
            direction = (-1, 0)
        else:
            candidate_rows = [r for r in bg_rows if r <= top_bar.start - 2]
            if not candidate_rows:
                return None
            row = min(candidate_rows)
            direction = (1, 0)
        assert 0 <= row < height and 0 <= col < width
        return (row, col), direction, lower_same

    row = random.choice(bg_rows)
    if lower_same.end < top_bar.start:
        candidate_cols = [c for c in bg_cols if c >= top_bar.end + 2]
        if not candidate_cols:
            return None
        col = max(candidate_cols)
        direction = (0, -1)
    else:
        candidate_cols = [c for c in bg_cols if c <= top_bar.start - 2]
        if not candidate_cols:
            return None
        col = min(candidate_cols)
        direction = (0, 1)
    assert 0 <= row < height and 0 <= col < width
    return (row, col), direction, lower_same


def _choose_markers(
    shape: tuple[int, int],
    bars: list[Bar],
    witness: tuple[int, int],
) -> list[tuple[int, int]] | None:
    height, width = shape
    bg_rows, bg_cols = _background_axes(shape, bars)
    all_cells = [(r, c) for r in bg_rows for c in bg_cols]
    edge_cells = [cell for cell in all_cells if min(cell[0], cell[1], height - 1 - cell[0], width - 1 - cell[1]) <= 1]

    target_count = random.choice(MARKER_COUNT_CHOICES)
    markers = [witness]
    used_rows = {witness[0]}
    used_cols = {witness[1]}

    for pool in (edge_cells, all_cells):
        shuffled = pool[:]
        random.shuffle(shuffled)
        for r, c in shuffled:
            if len(markers) >= target_count:
                break
            if (r, c) == witness or r in used_rows or c in used_cols:
                continue
            if any(max(abs(r - rr), abs(c - cc)) < 2 for rr, cc in markers):
                continue
            markers.append((r, c))
            used_rows.add(r)
            used_cols.add(c)
        if len(markers) >= target_count:
            return markers
    return None


def _validate_sample(
    inp: np.ndarray,
    expected: np.ndarray,
    designed_bars: list[Bar],
    witness_marker: tuple[int, int],
    witness_dir: tuple[int, int],
    witness_stop_bar: Bar,
) -> bool:
    try:
        inferred_bars = _infer_bars(inp)
        inferred_order = _infer_layer_order(inp, inferred_bars)
    except AssertionError:
        return False

    if [bar.color for bar in inferred_order] != [bar.color for bar in designed_bars]:
        return False
    if not np.array_equal(expected[inp != BG], inp[inp != BG]):
        return False
    if not np.any(expected == GREY) or not np.any(expected == YELLOW):
        return False

    witness = _trace_ray_events(inp, witness_marker, witness_dir, designed_bars[-1].color)
    if not witness["grey"] or not witness["top"] or not witness["yellow"]:
        return False
    if int(witness["stop_color"]) != witness_stop_bar.color:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_SIDE_RANGE)
    width = random.randint(*GRID_SIDE_RANGE)
    bar_count = random.choice(BAR_COUNT_CHOICES)
    orientations = _orientation_sequence(bar_count)

    horizontal_bands = _sample_bands(height, orientations.count("H"))
    vertical_bands = _sample_bands(width, orientations.count("V"))
    if horizontal_bands is None or vertical_bands is None:
        return None
    random.shuffle(horizontal_bands)
    random.shuffle(vertical_bands)

    colors = random.sample(BAR_COLORS, bar_count)
    bars: list[Bar] = []
    h_idx = 0
    v_idx = 0
    for orientation, color in zip(orientations, colors):
        if orientation == "H":
            start, end = horizontal_bands[h_idx]
            h_idx += 1
        else:
            start, end = vertical_bands[v_idx]
            v_idx += 1
        bars.append(Bar(orientation, int(start), int(end), int(color)))

    base = _render_bars((height, width), bars)
    witness_spec = _choose_witness_marker((height, width), bars)
    if witness_spec is None:
        return None
    witness_marker, witness_dir, witness_stop_bar = witness_spec

    markers = _choose_markers((height, width), bars, witness_marker)
    if markers is None:
        return None

    inp = base.copy()
    for r, c in markers:
        inp[r, c] = MARKER

    expected = _trace_rays(inp, bars[-1].color)
    if not _validate_sample(inp, expected, bars, witness_marker, witness_dir, witness_stop_bar):
        return None

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
