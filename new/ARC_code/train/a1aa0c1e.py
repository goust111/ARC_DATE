import numpy as np

from arc_utils import get_bg
import itertools
import random


def _full_width_bars(g: np.ndarray, bg: int) -> list[tuple[int, int]]:
    bars: list[tuple[int, int]] = []
    for r in range(g.shape[0]):
        vals = np.unique(g[r])
        if vals.size == 1 and int(vals[0]) != bg:
            bars.append((r, int(vals[0])))
    assert len(bars) >= 2, "expected source bars and one base bar"
    return bars


def _singleton_marker(g: np.ndarray, bg: int) -> tuple[int, int, int]:
    vals, counts = np.unique(g, return_counts=True)
    marker_colors = [
        int(v)
        for v, n in zip(vals, counts)
        if int(v) != bg and int(n) == 1
    ]
    assert len(marker_colors) == 1, "expected exactly one singleton marker color"
    marker_color = marker_colors[0]
    coords = np.argwhere(g == marker_color)
    assert coords.shape == (1, 2)
    r, c = map(int, coords[0])
    return marker_color, r, c


def _motif_stats(g: np.ndarray, bar_r: int, next_bar_r: int, color: int) -> tuple[int, int | None]:
    solid_rows = 0
    span: tuple[int, int] | None = None

    for r in range(bar_r + 1, next_bar_r):
        cols = np.flatnonzero(g[r] == color)
        if cols.size == 0:
            continue

        assert cols.size in (2, 3), "motif rows are side-pair or solid triples"
        left = int(cols[0])
        right = int(cols[-1])
        assert right - left == 2, "motif span must be three columns wide"
        if span is None:
            span = (left, right)
        assert span == (left, right), "motif columns must stay aligned"

        if cols.size == 2:
            assert np.array_equal(cols, np.array([left, right]))
        else:
            assert np.array_equal(cols, np.arange(left, right + 1))
            solid_rows += 1

    if span is None:
        return solid_rows, None
    return solid_rows, (span[0] + span[1]) // 2


def _marker_has_clear_neighbors(g: np.ndarray, bg: int, marker_r: int, marker_c: int, base_r: int) -> bool:
    H, W = g.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            r, c = marker_r + dr, marker_c + dc
            if 0 <= r < H and 0 <= c < W and r != base_r and int(g[r, c]) != bg:
                return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress the stacked colored bars into a compact ordered table with the control pixel on the selected row.

    Input structure
    - A background grid contains several full-width colored bars from top to
      bottom; the last bar is the base/separator color.
    - Each non-base bar may have an aligned three-column comb motif below it,
      alternating side-pair rows with solid rows.
    - A singleton control pixel sits away from the colored motifs.

    Rule
    - Make one output row per non-base bar, preserving top-to-bottom order.
    - In that row, paint the bar color once for each solid three-pixel row in
      its motif, then paint the base-bar color in the separator column.
    - The singleton control pixel sits immediately above the base bar. Among
      motifs whose open side-pair parity is compatible with that bottom row,
      put the control pixel in the output row with the fewest solid rows; use
      horizontal distance to the motif center only to break count ties.

    Key insight
    - The vertical comb height is a count, while the control pixel is read by
      projecting it onto the center gap of compatible comb rows rather than by
      its vertical band alone.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    bars = _full_width_bars(g, bg)
    source_bars = bars[:-1]
    base_color = bars[-1][1]
    base_r = bars[-1][0]
    n_rows = len(source_bars)

    marker_color, marker_r, marker_c = _singleton_marker(g, bg)
    assert marker_r == base_r - 1, "control marker must sit immediately above the base bar"
    assert _marker_has_clear_neighbors(g, bg, marker_r, marker_c, base_r), (
        "control marker must not touch non-base colored objects"
    )

    out = np.full((n_rows, n_rows + 2), bg, dtype=int)
    out[:, n_rows] = base_color

    marker_candidates: list[tuple[int, int, int]] = []
    for i, ((bar_r, color), (next_bar_r, _)) in enumerate(zip(source_bars, bars[1:])):
        solid_rows, center_c = _motif_stats(g, bar_r, next_bar_r, color)
        assert solid_rows <= n_rows, "motif count must fit in the output slots"
        out[i, :solid_rows] = color

        if solid_rows > 0 and center_c is not None and marker_r > bar_r and (marker_r - bar_r) % 2 == 1:
            marker_candidates.append((solid_rows, abs(marker_c - center_c), i))

    assert marker_candidates, "control pixel must match at least one open motif row"
    marker_candidates.sort()
    selected_row = marker_candidates[0][2]
    out[selected_row, n_rows + 1] = marker_color
    return out

# --------------- generator (att2) ---------------

_GEN_N = 3
_GEN_SHAPE = (21, 21)
_GEN_BG = 0
_GEN_MARKER = 5
_GEN_BASE = 9
_GEN_BAR_COLORS = [1, 2, 3, 4, 6, 7, 8]

_GEN_FIRST_BAR_ROWS = (1, 2)
_GEN_BASE_ROWS = (19, 20)
_GEN_MIN_REGION_H = 3
_GEN_MAX_REGION_H = 7
_GEN_TOTAL_REGION_LIMIT = 16

_GEN_SOLID_RANGE = range(_GEN_N + 1)
_GEN_CENTER_RANGE = range(3, 19)

_GEN_MAX_ATTEMPTS = 300


def _gen_region_floor(solid_count: int) -> int:
    return max(2 * solid_count + 1, _GEN_MIN_REGION_H)


def _gen_solid_options() -> list[tuple[int, ...]]:
    return [
        counts
        for counts in itertools.product(_GEN_SOLID_RANGE, repeat=_GEN_N)
        if any(counts)
        and sum(_gen_region_floor(s) for s in counts) <= _GEN_TOTAL_REGION_LIMIT
    ]


_GEN_SOLID_OPTIONS = _gen_solid_options()


def _gen_layouts(solid_counts: tuple[int, ...]) -> list[tuple[int, int, tuple[int, ...]]]:
    floors = tuple(_gen_region_floor(s) for s in solid_counts)
    layouts: list[tuple[int, int, tuple[int, ...]]] = []
    for first_bar_r in _GEN_FIRST_BAR_ROWS:
        for base_r in _GEN_BASE_ROWS:
            total_region_h = base_r - first_bar_r - _GEN_N
            extra = total_region_h - sum(floors)
            if extra < 0:
                continue
            caps = [_GEN_MAX_REGION_H - h for h in floors]
            for alloc in itertools.product(*(range(c + 1) for c in caps)):
                if sum(alloc) == extra:
                    heights = tuple(h + a for h, a in zip(floors, alloc))
                    layouts.append((first_bar_r, base_r, heights))
    return layouts


def _gen_bar_rows(first_bar_r: int, region_heights: tuple[int, ...]) -> list[int]:
    rows = [first_bar_r]
    for h in region_heights[:-1]:
        rows.append(rows[-1] + h + 1)
    return rows


def _gen_draw_motif(grid: np.ndarray, bar_r: int, color: int, center_c: int, solid_count: int) -> None:
    left_c = center_c - 1
    for k in range(2 * solid_count + 1):
        r = bar_r + 1 + k
        if k % 2 == 0:
            grid[r, left_c] = color
            grid[r, left_c + 2] = color
        else:
            grid[r, left_c:left_c + 3] = color


def _gen_marker_clear(grid: np.ndarray, marker_r: int, marker_c: int, base_r: int) -> bool:
    if grid[marker_r, marker_c] != _GEN_BG:
        return False
    H, W = grid.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            r, c = marker_r + dr, marker_c + dc
            if 0 <= r < H and 0 <= c < W and r != base_r and grid[r, c] != _GEN_BG:
                return False
    return True


def _gen_selected_row(solid_counts, bar_rows, centers, marker_r, marker_c):
    candidates = []
    for i, solid_count in enumerate(solid_counts):
        if solid_count > 0 and i in centers and marker_r > bar_rows[i] and (marker_r - bar_rows[i]) % 2 == 1:
            candidates.append((solid_count, abs(marker_c - centers[i]), i))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][2]


def _gen_assign_centers(solid_counts: tuple[int, ...], compatible_rows: list[int]) -> tuple[int, dict[int, int]] | None:
    """Choose the intended output row and motif centers that keep it selectable."""
    positive_rows = [i for i, solid_count in enumerate(solid_counts) if solid_count > 0]
    min_count = min(solid_counts[i] for i in compatible_rows)
    min_rows = [i for i in compatible_rows if solid_counts[i] == min_count]
    target_i = random.choice(min_rows)
    target_center = random.randint(_GEN_CENTER_RANGE.start, _GEN_CENTER_RANGE.stop - 1)
    centers: dict[int, int] = {target_i: target_center}

    for i in positive_rows:
        if i == target_i:
            continue
        choices = list(_GEN_CENTER_RANGE)
        if i in min_rows:
            choices = [c for c in choices if c != target_center]
        if not choices:
            return None
        centers[i] = random.choice(choices)
    return target_i, centers


def _gen_attempt() -> np.ndarray | None:
    solid_counts = random.choice(_GEN_SOLID_OPTIONS)
    bar_colors = random.sample(_GEN_BAR_COLORS, _GEN_N)
    expected = np.full((_GEN_N, _GEN_N + 2), _GEN_BG, dtype=int)
    expected[:, _GEN_N] = _GEN_BASE
    for i, solid_count in enumerate(solid_counts):
        expected[i, :solid_count] = bar_colors[i]

    layout_options = _gen_layouts(solid_counts)
    if not layout_options:
        return None
    first_bar_r, base_r, region_heights = random.choice(layout_options)
    bar_rows = _gen_bar_rows(first_bar_r, region_heights)

    marker_r = base_r - 1
    compatible_rows = [
        i
        for i, solid_count in enumerate(solid_counts)
        if solid_count > 0 and marker_r > bar_rows[i] and (marker_r - bar_rows[i]) % 2 == 1
    ]
    if not compatible_rows:
        return None

    center_plan = _gen_assign_centers(solid_counts, compatible_rows)
    if center_plan is None:
        return None
    target_i, centers = center_plan

    grid = np.zeros(_GEN_SHAPE, dtype=int)
    for r, color in zip(bar_rows, bar_colors):
        grid[r, :] = color
    grid[base_r, :] = _GEN_BASE
    for i, solid_count in enumerate(solid_counts):
        if solid_count > 0:
            _gen_draw_motif(grid, bar_rows[i], bar_colors[i], centers[i], solid_count)

    marker_cols = [
        c
        for c in range(_GEN_SHAPE[1])
        if _gen_marker_clear(grid, marker_r, c, base_r)
        and _gen_selected_row(solid_counts, bar_rows, centers, marker_r, c) == target_i
    ]
    if not marker_cols:
        return None
    marker_c = random.choice(marker_cols)
    grid[marker_r, marker_c] = _GEN_MARKER
    expected[target_i, _GEN_N + 1] = _GEN_MARKER

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid a1aa0c1e input")
