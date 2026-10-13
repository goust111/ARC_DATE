import random

import numpy as np


BLACK = 0
GREY = 5
STRIPE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

MIN_BODY_HEIGHT = 9
MAX_BODY_HEIGHT = 18
MIN_BODY_WIDTH = 9
MAX_BODY_WIDTH = 18
MIN_STRIPES = 2
MAX_H_STRIPES = 4
MAX_V_STRIPES = 4
MIN_GAP = 2
EDGE_MARGIN = 1
GENERATION_ATTEMPTS = 300


def _is_marker_column(col: np.ndarray) -> bool:
    values = {int(v) for v in np.unique(col)}
    if not values <= {BLACK, GREY} or GREY not in values:
        return False
    return bool(np.all(col[1:] != col[:-1]))


def _find_vertical_stripes(body: np.ndarray) -> list[tuple[int, int]]:
    stripes: list[tuple[int, int]] = []
    for c in range(body.shape[1]):
        column = body[:, c]
        if int(column[0]) != BLACK and bool(np.all(column == column[0])):
            stripes.append((c, int(column[0])))
    assert stripes, "expected at least one vertical stripe"
    assert len({color for _c, color in stripes}) == len(stripes), "vertical stripe colors must be unique"
    return stripes


def _find_horizontal_stripes(body: np.ndarray, vertical_cols: list[int]) -> list[tuple[int, int]]:
    keep = np.ones(body.shape[1], dtype=bool)
    keep[vertical_cols] = False
    assert bool(np.any(keep)), "expected at least one non-vertical column"

    stripes: list[tuple[int, int]] = []
    for r in range(body.shape[0]):
        row_slice = body[r, keep]
        if int(row_slice[0]) != BLACK and bool(np.all(row_slice == row_slice[0])):
            stripes.append((r, int(row_slice[0])))
    assert stripes, "expected at least one horizontal stripe"
    assert len({color for _r, color in stripes}) == len(stripes), "horizontal stripe colors must be unique"
    return stripes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Apply the woven over-under rule encoded by horizontal and vertical stripes.

    The input has a black background, full-height vertical stripes, full-width
    horizontal stripes, and a grey/black alternating marker column on the far
    right.  In the input every crossing shows the vertical color; the output
    removes the marker column and alternates which stripe is on top according to
    the parity of the 1-based horizontal/vertical stripe indices.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] >= 2 and g.shape[1] >= 3, "grid is too small"
    assert _is_marker_column(g[:, -1]), "rightmost column must be the weave marker"

    body = g[:, :-1].copy()
    vertical = _find_vertical_stripes(body)
    vertical_cols = [c for c, _color in vertical]
    vertical_color_by_col = {c: color for c, color in vertical}
    horizontal = _find_horizontal_stripes(body, vertical_cols)
    horizontal_rows = [r for r, _color in horizontal]
    horizontal_color_by_row = {r: color for r, color in horizontal}

    assert len(set(vertical_cols)) == len(vertical_cols)
    assert len(set(horizontal_rows)) == len(horizontal_rows)
    assert not (set(vertical_color_by_col.values()) & set(horizontal_color_by_row.values()))
    assert len(horizontal_rows) < body.shape[0], "expected visible background rows"
    assert len(vertical_cols) < body.shape[1], "expected visible background columns"

    non_vertical_cols = [c for c in range(body.shape[1]) if c not in vertical_color_by_col]
    non_horizontal_rows = [r for r in range(body.shape[0]) if r not in horizontal_color_by_row]
    assert non_vertical_cols and non_horizontal_rows
    background_patch = body[np.ix_(non_horizontal_rows, non_vertical_cols)]
    assert bool(np.all(background_patch == BLACK)), "background region must stay black"

    for r, color in horizontal:
        assert bool(np.all(body[r, non_vertical_cols] == color)), "horizontal stripe row is inconsistent"
    for c, color in vertical:
        assert bool(np.all(body[:, c] == color)), "vertical stripe column is inconsistent"

    out = body.copy()
    for i, (r, h_color) in enumerate(horizontal, start=1):
        for j, (c, v_color) in enumerate(vertical, start=1):
            out[r, c] = v_color if (i + j) % 2 == 1 else h_color

    return out


def _max_spaced_count(length: int, margin: int = EDGE_MARGIN, gap: int = MIN_GAP) -> int:
    usable = length - 2 * margin
    if usable <= 0:
        return 0
    return (usable + gap - 1) // gap


def _choose_spaced_indices(length: int, count: int, margin: int = EDGE_MARGIN, gap: int = MIN_GAP) -> list[int] | None:
    candidates = list(range(margin, length - margin))
    if len(candidates) < count:
        return None
    for _ in range(200):
        picked = sorted(random.sample(candidates, count))
        if all(b - a >= gap for a, b in zip(picked, picked[1:])):
            return picked
    return None


def _build_marker_column(height: int) -> np.ndarray:
    phase = random.randint(0, 1)
    return np.array(
        [GREY if (r + phase) % 2 == 0 else BLACK for r in range(height)],
        dtype=int,
    )


def _gen_attempt() -> np.ndarray | None:
    body_h = random.randint(MIN_BODY_HEIGHT, MAX_BODY_HEIGHT)
    body_w = random.randint(MIN_BODY_WIDTH, MAX_BODY_WIDTH)

    max_h = min(MAX_H_STRIPES, _max_spaced_count(body_h))
    max_v = min(MAX_V_STRIPES, _max_spaced_count(body_w))
    if max_h < MIN_STRIPES or max_v < MIN_STRIPES:
        return None

    n_h = random.randint(MIN_STRIPES, max_h)
    n_v = random.randint(MIN_STRIPES, min(max_v, len(STRIPE_COLORS) - n_h))
    row_positions = _choose_spaced_indices(body_h, n_h)
    col_positions = _choose_spaced_indices(body_w, n_v)
    if row_positions is None or col_positions is None:
        return None

    colors = random.sample(STRIPE_COLORS, n_h + n_v)
    h_colors = colors[:n_h]
    v_colors = colors[n_h:]
    body = np.full((body_h, body_w), BLACK, dtype=int)

    for r, color in zip(row_positions, h_colors):
        body[r, :] = color
    for c, color in zip(col_positions, v_colors):
        body[:, c] = color

    marker_col = _build_marker_column(body_h)[:, None]
    grid = np.concatenate([body, marker_col], axis=1)

    try:
        out = main(grid)
    except AssertionError:
        return None

    odd_crossings = 0
    even_crossings = 0
    for i, (r, h_color) in enumerate(zip(row_positions, h_colors), start=1):
        for j, (c, v_color) in enumerate(zip(col_positions, v_colors), start=1):
            if (i + j) % 2 == 1:
                odd_crossings += 1
                if int(out[r, c]) != v_color:
                    return None
            else:
                even_crossings += 1
                if int(out[r, c]) != h_color:
                    return None

    if odd_crossings == 0 or even_crossings == 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid woven-stripe input")
