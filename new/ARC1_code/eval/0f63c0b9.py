import numpy as np

from arc_utils import find_components, get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each colored marker into a full-width horizontal band frame determined by marker-row proximity.

    Input structure
    - A 2D grid with a dominant background color.
    - The foreground consists of isolated one-pixel colored markers.
    - Marker rows are unique; marker columns only identify the original pixel and do not affect the frame width.

    Rule
    - Sort markers by row and split the grid into horizontal bands at the midpoint between consecutive marker rows.
    - For each marker's band, paint the left and right boundary columns with the marker color.
    - Paint the marker's own row across the full width; also paint the top or bottom grid edge when the band touches it.

    Key insight
    - The single pixels are row anchors for full-width frames: each color owns the rows vertically closest to its marker.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    H, W = map(int, g.shape)
    assert H > 0 and W >= 2, "expected a nonempty grid with side columns"

    bg = get_bg(g)
    marker_coords = np.argwhere(g != bg)
    assert marker_coords.size > 0, "expected at least one colored marker"

    comps = find_components(g, bg=bg, connectivity=8, monochromatic=True)
    assert len(comps) == len(marker_coords), "expected foreground to consist of one-pixel markers"
    assert all(int(comp["area"]) == 1 for comp in comps), "expected every marker to be a singleton"

    markers = [(int(r), int(c), int(g[r, c])) for r, c in marker_coords]
    rows = [r for r, _c, _color in markers]
    assert len(rows) == len(set(rows)), "expected marker rows to be unique"
    markers.sort(key=lambda item: item[0])

    out = np.full_like(g, bg)
    for i, (row, _col, color) in enumerate(markers):
        start = 0 if i == 0 else (markers[i - 1][0] + row) // 2 + 1
        end = H - 1 if i == len(markers) - 1 else (row + markers[i + 1][0]) // 2
        assert start <= row <= end, "marker must lie inside its vertical band"

        out[start : end + 1, 0] = color
        out[start : end + 1, W - 1] = color
        out[row, :] = color
        if start == 0:
            out[0, :] = color
        if end == H - 1:
            out[H - 1, :] = color

    return out

# --------------- generator (o5x) ---------------

# Canvas and palette.
_GEN_H, _GEN_W, _GEN_BG = 15, 15, 0
_GEN_COLOR_CHOICES = np.arange(1, 10)

# Marker, band, and placement bounds.
_GEN_MARKER_COUNT_MIN, _GEN_MARKER_COUNT_MAX_EXCLUSIVE = 3, 5
_GEN_BAND_HEIGHT_MIN, _GEN_BAND_HEIGHT_MAX = 3, 7
_GEN_MARKER_ROW_MIN, _GEN_MARKER_ROW_MAX = 1, _GEN_H - 3
_GEN_MARKER_COL_MIN, _GEN_MARKER_COL_MAX_EXCLUSIVE = 1, _GEN_W - 1

# Row-gap model and retries.
_GEN_GAP_MIN, _GEN_GAP_MAX = 2, 7
_GEN_OFFSET_DELTA_CHOICES = (0, 1)
_GEN_GAP_WEIGHT_BASE, _GEN_GAP_WEIGHT_OFFSET = 2, 3
_GEN_MAX_ATTEMPTS = 300


def _gen_compositions(count: int, total: int, low: int, high: int) -> list[tuple[int, ...]]:
    if count == 0:
        return [()] if total == 0 else []
    out = []
    for value in range(low, high + 1):
        rest = total - value
        if low * (count - 1) <= rest <= high * (count - 1):
            for tail in _gen_compositions(count - 1, rest, low, high):
                out.append((value,) + tail)
    return out


def _gen_rows_from_offsets(heights: tuple[int, ...], offsets: tuple[int, ...]) -> list[int]:
    starts = np.cumsum((0,) + heights[:-1])
    return [int(start) + offset for start, offset in zip(starts, offsets)]


def _gen_offsets_for_heights(heights: tuple[int, ...]) -> list[tuple[int, ...]]:
    offsets = []
    current = []

    def backtrack(i: int) -> None:
        if i == len(heights):
            rows = _gen_rows_from_offsets(heights, tuple(current))
            if all(_GEN_MARKER_ROW_MIN <= row <= _GEN_MARKER_ROW_MAX for row in rows) and all(
                _GEN_GAP_MIN <= rows[j + 1] - rows[j] <= _GEN_GAP_MAX for j in range(len(rows) - 1)
            ):
                offsets.append(tuple(current))
            return

        height = heights[i]
        for offset in range(1, height - 1):
            if i > 0:
                prev_right_gap = heights[i - 1] - 1 - current[-1]
                if prev_right_gap - offset not in _GEN_OFFSET_DELTA_CHOICES:
                    continue
            current.append(offset)
            backtrack(i + 1)
            current.pop()

    backtrack(0)
    return offsets


def _gen_configs(n: int) -> list[tuple[tuple[int, ...], tuple[int, ...], float]]:
    configs = []
    for heights in _gen_compositions(n, _GEN_H, _GEN_BAND_HEIGHT_MIN, _GEN_BAND_HEIGHT_MAX):
        for offsets in _gen_offsets_for_heights(heights):
            rows = _gen_rows_from_offsets(heights, offsets)
            gaps = [rows[i + 1] - rows[i] for i in range(len(rows) - 1)]
            weight = float(np.prod([_GEN_GAP_WEIGHT_BASE ** (gap - _GEN_GAP_WEIGHT_OFFSET) for gap in gaps]))
            configs.append((heights, offsets, weight))
    return configs


_GEN_CONFIGS_BY_N = {
    n: _gen_configs(n)
    for n in range(_GEN_MARKER_COUNT_MIN, _GEN_MARKER_COUNT_MAX_EXCLUSIVE)
}


def _gen_expected_output(rows: list[int], colors: list[int], heights: tuple[int, ...]) -> np.ndarray:
    out = np.full((_GEN_H, _GEN_W), _GEN_BG, dtype=int)
    start = 0
    for row, color, height in zip(rows, colors, heights):
        end = start + height - 1
        out[start : end + 1, 0] = color
        out[start : end + 1, _GEN_W - 1] = color
        out[row, :] = color
        if start == 0:
            out[0, :] = color
        if end == _GEN_H - 1:
            out[_GEN_H - 1, :] = color
        start = end + 1
    return out


def _gen_attempt() -> np.ndarray | None:
    n = int(np.random.randint(_GEN_MARKER_COUNT_MIN, _GEN_MARKER_COUNT_MAX_EXCLUSIVE))
    configs = _GEN_CONFIGS_BY_N[n]
    if not configs:
        return None
    weights = np.array([weight for _heights, _offsets, weight in configs], dtype=float)
    index = int(np.random.choice(len(configs), p=weights / weights.sum()))
    heights, offsets, _weight = configs[index]

    rows = _gen_rows_from_offsets(heights, offsets)
    cols = np.random.randint(
        _GEN_MARKER_COL_MIN, _GEN_MARKER_COL_MAX_EXCLUSIVE, size=n
    ).astype(int).tolist()
    colors = np.random.choice(_GEN_COLOR_CHOICES, size=n, replace=False).astype(int).tolist()

    inp = np.full((_GEN_H, _GEN_W), _GEN_BG, dtype=int)
    for row, col, color in zip(rows, cols, colors):
        inp[row, col] = color

    expected = _gen_expected_output(rows, colors, heights)
    if np.array_equal(main(inp), expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {_GEN_MAX_ATTEMPTS} attempts")
