from __future__ import annotations

import random

import numpy as np


BG = 0
COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]

MIN_SIZE = 9
MAX_SIZE = 21
MIN_SEGMENTS = 2
MAX_SEGMENTS = 4
MIN_POSITION_GAP = 2
GEN_RETRIES = 400


def _legend_colors(g: np.ndarray) -> list[int]:
    legend = [int(x) for x in g[:, -1] if int(x) != BG]
    assert len(legend) == 3, "expected three legend pixels in the rightmost column"
    assert len(set(legend)) == 3, "legend colors must be distinct"
    return legend


def _axis_neighbors(
    core: np.ndarray,
    r: int,
    c: int,
    dirs: tuple[tuple[int, int], tuple[int, int]],
) -> list[int]:
    h, w = core.shape
    colors: list[int] = []
    for dr, dc in dirs:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and int(core[nr, nc]) != BG:
            colors.append(int(core[nr, nc]))
    return colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove bridge-over crossings by restoring the lower-layer path color.

    Input structure:
    - Black is the background and the rightmost column is a vertical legend of
      three colored pixels, read from top to bottom as high to low layer order.
    - The remaining columns contain one-pixel-wide horizontal and vertical
      straight path segments in those three colors.
    - At every spatial crossing between two different colored path segments,
      the input shows only the higher-layer color at the shared pixel.

    Rule:
    - Remove the rightmost legend column from the output.
    - A colored path pixel is a crossing when it has at least one colored
      horizontal neighbor and at least one colored vertical neighbor.
    - For each crossing, infer the horizontal and vertical path colors from the
      adjacent same-axis path pixels, then replace the displayed higher-layer
      color by the lower-layer color.
    - All non-crossing path pixels are copied unchanged.

    Key insight:
    - The legend supplies the hidden layer ordering, while the broken
      continuity of the two straight paths around a crossing reveals which
      lower color was covered by the bridge.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[1] >= 2

    legend = _legend_colors(g)
    rank = {color: idx for idx, color in enumerate(legend)}
    core = g[:, :-1]
    assert set(map(int, np.unique(core))) <= set(legend) | {BG}

    out = core.copy()
    h, w = core.shape
    for r in range(h):
        for c in range(w):
            visible = int(core[r, c])
            if visible == BG:
                continue

            h_neighbors = _axis_neighbors(core, r, c, ((0, -1), (0, 1)))
            v_neighbors = _axis_neighbors(core, r, c, ((-1, 0), (1, 0)))
            if not h_neighbors or not v_neighbors:
                continue

            assert len(set(h_neighbors)) == 1, "horizontal path color must be unambiguous"
            assert len(set(v_neighbors)) == 1, "vertical path color must be unambiguous"
            h_color = h_neighbors[0]
            v_color = v_neighbors[0]

            assert h_color in rank and v_color in rank and visible in rank
            if h_color == v_color:
                continue

            shown = h_color if rank[h_color] < rank[v_color] else v_color
            hidden = v_color if shown == h_color else h_color
            assert visible == shown, "crossing pixel must show the higher layer"
            out[r, c] = hidden

    return out


def _max_spaced_count(lo: int, hi: int, min_gap: int) -> int:
    if hi < lo:
        return 0
    return (hi - lo) // min_gap + 1


def _sample_spaced(
    rng: random.Random,
    count: int,
    lo: int,
    hi: int,
    min_gap: int,
) -> list[int] | None:
    for _ in range(200):
        vals = sorted(rng.sample(range(lo, hi + 1), count))
        if all(vals[i + 1] - vals[i] >= min_gap for i in range(count - 1)):
            return vals
    return None


def _segment_colors(
    rng: random.Random,
    colors: list[int],
    count: int,
) -> list[int]:
    result = list(colors)
    while len(result) < count:
        result.append(rng.choice(colors))
    rng.shuffle(result)
    return result


def _paint_segment(
    covers: list[list[list[tuple[str, int]]]],
    orientation: str,
    color: int,
    fixed: int,
    start: int,
    end: int,
) -> None:
    if orientation == "h":
        r = fixed
        for c in range(start, end + 1):
            covers[r][c].append((orientation, color))
    else:
        c = fixed
        for r in range(start, end + 1):
            covers[r][c].append((orientation, color))


def _render_from_covers(
    covers: list[list[list[tuple[str, int]]]],
    layer_order: list[int],
    legend_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    h = len(covers)
    core_w = len(covers[0])
    rank = {color: idx for idx, color in enumerate(layer_order)}
    inp = np.full((h, core_w + 1), BG, dtype=int)
    expected = np.full((h, core_w), BG, dtype=int)

    for r in range(h):
        for c in range(core_w):
            cell = covers[r][c]
            if not cell:
                continue

            colors = [color for _orientation, color in cell]
            visible = min(colors, key=lambda color: rank[color])
            inp[r, c] = int(visible)
            expected[r, c] = int(visible)

            h_colors = {color for orientation, color in cell if orientation == "h"}
            v_colors = {color for orientation, color in cell if orientation == "v"}
            if h_colors and v_colors:
                assert len(h_colors) == 1 and len(v_colors) == 1
                h_color = next(iter(h_colors))
                v_color = next(iter(v_colors))
                expected[r, c] = h_color if rank[h_color] > rank[v_color] else v_color

    assert legend_width == core_w + 1
    for r, color in enumerate(layer_order):
        inp[r, legend_width - 1] = int(color)
    return inp, expected


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    h = rng.randint(MIN_SIZE, MAX_SIZE)
    core_w = rng.randint(MIN_SIZE, MAX_SIZE)
    input_w = core_w + 1

    row_lo, row_hi = 2, h - 3
    col_lo, col_hi = 2, core_w - 3
    max_h = min(MAX_SEGMENTS, _max_spaced_count(row_lo, row_hi, MIN_POSITION_GAP))
    max_v = min(MAX_SEGMENTS, _max_spaced_count(col_lo, col_hi, MIN_POSITION_GAP))
    if max_h < MIN_SEGMENTS or max_v < MIN_SEGMENTS:
        return None

    colors = rng.sample(COLOR_POOL, 3)
    layer_order = colors[:]
    rng.shuffle(layer_order)

    n_h_colors = rng.choice([1, 2])
    h_colors = rng.sample(colors, n_h_colors)
    v_colors = [color for color in colors if color not in h_colors]

    n_h = rng.randint(max(MIN_SEGMENTS, len(h_colors)), max_h)
    n_v = rng.randint(max(MIN_SEGMENTS, len(v_colors)), max_v)
    rows = _sample_spaced(rng, n_h, row_lo, row_hi, MIN_POSITION_GAP)
    cols = _sample_spaced(rng, n_v, col_lo, col_hi, MIN_POSITION_GAP)
    if rows is None or cols is None:
        return None

    h_segment_colors = _segment_colors(rng, h_colors, n_h)
    v_segment_colors = _segment_colors(rng, v_colors, n_v)

    min_row, max_row = min(rows), max(rows)
    min_col, max_col = min(cols), max(cols)
    if min_col < 1 or max_col + 1 > core_w - 2 or min_row < 1 or max_row + 1 > h - 1:
        return None

    covers: list[list[list[tuple[str, int]]]] = [[[] for _ in range(core_w)] for _ in range(h)]

    for r, color in zip(rows, h_segment_colors):
        c0 = rng.randint(0, min_col - 1)
        c1 = rng.randint(max_col + 1, core_w - 2)
        _paint_segment(covers, "h", int(color), int(r), int(c0), int(c1))

    for c, color in zip(cols, v_segment_colors):
        r0 = rng.randint(0, min_row - 1)
        r1 = rng.randint(max_row + 1, h - 1)
        _paint_segment(covers, "v", int(color), int(c), int(r0), int(r1))

    inp, expected = _render_from_covers(covers, layer_order, input_w)

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid seed807260302/6/3 input")
