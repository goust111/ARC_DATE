import numpy as np


PALETTE = tuple(range(10))
MIN_GRID_SIZE = 10
MAX_GRID_SIZE_EXCLUSIVE = 19
MAX_LINES_PER_AXIS = 4
POSITION_SAMPLE_ATTEMPTS = 200
MAX_GENERATION_ATTEMPTS = 300


def _edge_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _infer_paths(grid: np.ndarray) -> tuple[int, list[dict[str, int]], list[dict[str, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h >= 3 and w >= 3, "grid is too small for edge-to-edge paths"

    edge = _edge_mask(g.shape)
    colors = [int(color) for color in np.unique(g)]

    horizontals: list[dict[str, int]] = []
    verticals: list[dict[str, int]] = []
    leftovers: list[int] = []

    for color in colors:
        pts = np.argwhere((g == color) & edge)
        if pts.shape[0] != 2:
            leftovers.append(color)
            continue

        (r0, c0), (r1, c1) = pts.tolist()
        if r0 == r1 and 0 < r0 < h - 1 and {c0, c1} == {0, w - 1}:
            horizontals.append({"color": color, "row": int(r0)})
        elif c0 == c1 and 0 < c0 < w - 1 and {r0, r1} == {0, h - 1}:
            verticals.append({"color": color, "col": int(c0)})
        else:
            leftovers.append(color)

    assert horizontals and verticals, "expected both horizontal and vertical paths"
    assert len(leftovers) == 1, "expected exactly one non-path color"
    bg = int(leftovers[0])

    rows = [path["row"] for path in horizontals]
    cols = [path["col"] for path in verticals]
    assert len(rows) == len(set(rows)), "horizontal rows must be unique"
    assert len(cols) == len(set(cols)), "vertical columns must be unique"

    horizontal_by_row = {path["row"]: path["color"] for path in horizontals}
    vertical_by_col = {path["col"]: path["color"] for path in verticals}
    horizontal_rows = set(horizontal_by_row)
    vertical_cols = set(vertical_by_col)

    for row, color in horizontal_by_row.items():
        for col in range(w):
            expected = {color}
            if col in vertical_cols:
                expected.add(vertical_by_col[col])
            assert int(g[row, col]) in expected, "horizontal path reconstruction failed"

    for col, color in vertical_by_col.items():
        for row in range(h):
            expected = {color}
            if row in horizontal_rows:
                expected.add(horizontal_by_row[row])
            assert int(g[row, col]) in expected, "vertical path reconstruction failed"

    for row in range(h):
        for col in range(w):
            if row not in horizontal_rows and col not in vertical_cols:
                assert int(g[row, col]) == bg, "unexpected foreground outside inferred paths"

    horizontals.sort(key=lambda path: path["row"])
    verticals.sort(key=lambda path: path["col"])
    return bg, horizontals, verticals


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct edge-to-edge colored paths and flip the visible color at each crossing.
    """
    g = np.asarray(grid, dtype=int)
    _bg, horizontals, verticals = _infer_paths(g)

    out = g.copy()
    for h_path in horizontals:
        row = h_path["row"]
        h_color = h_path["color"]
        for v_path in verticals:
            col = v_path["col"]
            v_color = v_path["color"]
            visible = int(g[row, col])
            if visible == h_color:
                out[row, col] = v_color
            else:
                assert visible == v_color, "crossing must show one of the two intersecting paths"
                out[row, col] = h_color
    return out


def _max_non_adjacent_positions(length: int) -> int:
    return (length - 1) // 2


def _sample_positions(length: int, count: int, rng: np.random.Generator) -> list[int] | None:
    candidates = np.arange(1, length - 1)
    if count > len(candidates):
        return None

    for _ in range(POSITION_SAMPLE_ATTEMPTS):
        picks = np.sort(rng.choice(candidates, size=count, replace=False))
        if np.all(np.diff(picks) >= 2):
            return [int(v) for v in picks]
    return None


def _sample_crossing_visibility(
    h_count: int, v_count: int, rng: np.random.Generator
) -> np.ndarray:
    top_is_horizontal = rng.integers(0, 2, size=(h_count, v_count), dtype=np.int8).astype(bool)
    if h_count * v_count > 1 and (top_is_horizontal.all() or (~top_is_horizontal).all()):
        flip_r = int(rng.integers(0, h_count))
        flip_c = int(rng.integers(0, v_count))
        top_is_horizontal[flip_r, flip_c] = ~top_is_horizontal[flip_r, flip_c]
    return top_is_horizontal


def _build_pair(
    height: int,
    width: int,
    bg: int,
    rows: list[int],
    cols: list[int],
    h_colors: list[int],
    v_colors: list[int],
    top_is_horizontal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((height, width), bg, dtype=int)

    for row, color in zip(rows, h_colors):
        inp[row, :] = color
    for col, color in zip(cols, v_colors):
        inp[:, col] = color

    expected = inp.copy()
    for i, (row, h_color) in enumerate(zip(rows, h_colors)):
        for j, (col, v_color) in enumerate(zip(cols, v_colors)):
            if top_is_horizontal[i, j]:
                inp[row, col] = h_color
                expected[row, col] = v_color
            else:
                inp[row, col] = v_color
                expected[row, col] = h_color

    return inp, expected


def _sample_line_counts(height: int, width: int, rng: np.random.Generator) -> tuple[int, int]:
    max_h = min(MAX_LINES_PER_AXIS, _max_non_adjacent_positions(height))
    max_v = min(MAX_LINES_PER_AXIS, _max_non_adjacent_positions(width))
    assert max_h >= 1 and max_v >= 1

    for _ in range(POSITION_SAMPLE_ATTEMPTS):
        h_count = int(rng.integers(1, max_h + 1))
        v_count = int(rng.integers(1, max_v + 1))
        if h_count + v_count <= len(PALETTE) - 1:
            return h_count, v_count
    raise RuntimeError("failed to sample line counts")


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(MIN_GRID_SIZE, MAX_GRID_SIZE_EXCLUSIVE))
    width = int(rng.integers(MIN_GRID_SIZE, MAX_GRID_SIZE_EXCLUSIVE))
    h_count, v_count = _sample_line_counts(height, width, rng)

    rows = _sample_positions(height, h_count, rng)
    cols = _sample_positions(width, v_count, rng)
    if rows is None or cols is None:
        return None

    bg = int(rng.integers(0, len(PALETTE)))
    colors = [int(color) for color in rng.choice([c for c in PALETTE if c != bg], size=h_count + v_count, replace=False)]
    h_colors = colors[:h_count]
    v_colors = colors[h_count:]
    top_is_horizontal = _sample_crossing_visibility(h_count, v_count, rng)

    inp, expected = _build_pair(height, width, bg, rows, cols, h_colors, v_colors, top_is_horizontal)
    try:
        if np.array_equal(main(inp), expected):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid crossing-path puzzle")
