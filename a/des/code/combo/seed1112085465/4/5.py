import random

import numpy as np


BG = 0
SEP = 9
FRAME = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8)
GAP_TO_ROT = {"top": 0, "right": 3, "bottom": 2, "left": 1}


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    pts = np.argwhere(mask)
    assert pts.size > 0
    r0 = int(pts[:, 0].min())
    c0 = int(pts[:, 1].min())
    r1 = int(pts[:, 0].max())
    c1 = int(pts[:, 1].max())
    return r0, c0, r1, c1


def _column_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    cols = [int(c) for c in np.flatnonzero(mask)]
    if not cols:
        return []
    runs: list[tuple[int, int]] = []
    start = end = cols[0]
    for col in cols[1:]:
        if col == end + 1:
            end = col
        else:
            runs.append((start, end))
            start = end = col
    runs.append((start, end))
    return runs


def _find_separator(g: np.ndarray) -> tuple[int, int]:
    full_cols = np.all(g == SEP, axis=0)
    runs = _column_runs(full_cols)
    assert len(runs) == 1, "expected exactly one full-height maroon separator"
    return runs[0]


def _decode_template(g: np.ndarray, sep_start: int) -> np.ndarray:
    left_mask = np.zeros_like(g, dtype=bool)
    left_mask[:, :sep_start] = g[:, :sep_start] != BG
    r0, c0, r1, c1 = _bbox(left_mask)
    patch = g[r0 : r1 + 1, c0 : c1 + 1]
    h, w = patch.shape
    assert h % 2 == 0 and w % 2 == 0 and h == w, "template panel must be square 2x2 blocks"

    logical_h = h // 2
    logical_w = w // 2
    logical = np.zeros((logical_h, logical_w), dtype=int)
    for rr in range(logical_h):
        for cc in range(logical_w):
            block = patch[2 * rr : 2 * rr + 2, 2 * cc : 2 * cc + 2]
            vals = np.unique(block)
            assert len(vals) == 1, "every template cell must be a solid 2x2 block"
            logical[rr, cc] = int(vals[0])
    return logical


def _find_frame(g: np.ndarray, sep_end: int) -> tuple[tuple[int, int, int, int], str]:
    right_mask = np.zeros_like(g, dtype=bool)
    right_mask[:, sep_end + 1 :] = g[:, sep_end + 1 :] == FRAME
    r0, c0, r1, c1 = _bbox(right_mask)

    top_missing = [c for c in range(c0, c1 + 1) if int(g[r0, c]) != FRAME]
    bottom_missing = [c for c in range(c0, c1 + 1) if int(g[r1, c]) != FRAME]
    left_missing = [r for r in range(r0, r1 + 1) if int(g[r, c0]) != FRAME]
    right_missing = [r for r in range(r0, r1 + 1) if int(g[r, c1]) != FRAME]

    sides = {
        "top": top_missing,
        "bottom": bottom_missing,
        "left": left_missing,
        "right": right_missing,
    }
    gap_sides = [side for side, miss in sides.items() if len(miss) == 1]
    assert len(gap_sides) == 1, "frame must have exactly one one-pixel border gap"
    gap_side = gap_sides[0]

    for side, miss in sides.items():
        if side == gap_side:
            continue
        assert len(miss) == 0, "all non-gap frame sides must be complete"

    gap_pos = sides[gap_side][0]
    if gap_side in ("top", "bottom"):
        assert c0 < gap_pos < c1, "gap must not be at a corner"
    else:
        assert r0 < gap_pos < r1, "gap must not be at a corner"

    return (r0, c0, r1, c1), gap_side


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read a header-annotated lookup table from the left panel, apply the selected row and
    column keys from the right query panel, then rotate the filled result according to the
    side of the single-pixel gap in the grey frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sep_start, sep_end = _find_separator(g)
    template = _decode_template(g, sep_start)
    assert template.shape[0] >= 3

    row_keys = [int(x) for x in template[1:, 0]]
    col_keys = [int(x) for x in template[0, 1:]]
    assert len(row_keys) == len(set(row_keys))
    assert len(col_keys) == len(set(col_keys))
    row_index = {color: idx for idx, color in enumerate(row_keys)}
    col_index = {color: idx for idx, color in enumerate(col_keys)}
    values = template[1:, 1:]
    assert values.shape == (len(row_keys), len(col_keys))

    (r0, c0, r1, c1), gap_side = _find_frame(g, sep_end)
    interior = g[r0 + 1 : r1, c0 + 1 : c1]
    assert interior.shape[0] >= 2 and interior.shape[1] >= 2

    col_markers = interior[0, 1:]
    row_markers = interior[1:, 0]
    data_h = interior.shape[0] - 1
    data_w = interior.shape[1] - 1
    result = np.full((data_h, data_w), BG, dtype=int)

    for rr, marker in enumerate(row_markers):
        marker = int(marker)
        if marker == BG:
            continue
        assert marker in row_index, "row marker must match a template row header"
        for cc, cmarker in enumerate(col_markers):
            cmarker = int(cmarker)
            if cmarker == BG:
                continue
            assert cmarker in col_index, "column marker must match a template column header"
            result[rr, cc] = int(values[row_index[marker], col_index[cmarker]])

    return np.rot90(result, GAP_TO_ROT[gap_side])


def _sample_value_table(n: int) -> np.ndarray:
    while True:
        table = np.array(
            [[random.choice(COLOR_POOL) for _ in range(n - 1)] for _ in range(n - 1)],
            dtype=int,
        )
        if len(set(map(int, np.unique(table)))) < 2:
            continue
        rows = {tuple(int(x) for x in row) for row in table.tolist()}
        cols = {tuple(int(x) for x in col) for col in table.T.tolist()}
        if len(rows) < min(2, n - 1) or len(cols) < min(2, n - 1):
            continue
        return table


def _build_template_panel(n: int) -> tuple[np.ndarray, list[int], list[int]]:
    row_keys = random.sample(COLOR_POOL, n - 1)
    col_keys = random.sample(COLOR_POOL, n - 1)
    values = _sample_value_table(n)

    logical = np.zeros((n, n), dtype=int)
    logical[0, 0] = random.choice(COLOR_POOL)
    logical[1:, 0] = np.array(row_keys, dtype=int)
    logical[0, 1:] = np.array(col_keys, dtype=int)
    logical[1:, 1:] = values

    panel = np.repeat(np.repeat(logical, 2, axis=0), 2, axis=1)
    return panel, row_keys, col_keys


def _build_query_panel(
    data_h: int,
    data_w: int,
    row_keys: list[int],
    col_keys: list[int],
    gap_side: str,
) -> np.ndarray:
    inner = np.full((data_h + 1, data_w + 1), BG, dtype=int)

    row_count = random.randint(1, min(data_h, len(row_keys)))
    col_count = random.randint(1, min(data_w, len(col_keys)))
    row_slots = random.sample(range(data_h), row_count)
    col_slots = random.sample(range(data_w), col_count)
    row_colors = random.sample(row_keys, row_count)
    col_colors = random.sample(col_keys, col_count)

    for slot, color in zip(row_slots, row_colors, strict=False):
        inner[1 + slot, 0] = int(color)
    for slot, color in zip(col_slots, col_colors, strict=False):
        inner[0, 1 + slot] = int(color)

    panel = np.full((inner.shape[0] + 2, inner.shape[1] + 2), BG, dtype=int)
    panel[0, :] = FRAME
    panel[-1, :] = FRAME
    panel[:, 0] = FRAME
    panel[:, -1] = FRAME
    panel[1:-1, 1:-1] = inner

    if gap_side == "top":
        panel[0, random.randint(1, panel.shape[1] - 2)] = BG
    elif gap_side == "bottom":
        panel[-1, random.randint(1, panel.shape[1] - 2)] = BG
    elif gap_side == "left":
        panel[random.randint(1, panel.shape[0] - 2), 0] = BG
    else:
        panel[random.randint(1, panel.shape[0] - 2), -1] = BG

    return panel


def _place(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    grid[top : top + h, left : left + w] = patch


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(4, 6)
    panel, row_keys, col_keys = _build_template_panel(n)

    data_h = random.randint(2, n + 1)
    data_w = random.randint(2, n + 1)
    gap_side = random.choice(["top", "right", "bottom", "left"])
    query = _build_query_panel(data_h, data_w, row_keys, col_keys, gap_side)

    left_pad = random.randint(1, 3)
    right_pad = random.randint(1, 3)
    top_pad = random.randint(1, 3)
    bottom_pad = random.randint(1, 3)
    gap_left = random.randint(1, 3)
    gap_right = random.randint(2, 4)
    sep_width = random.randint(1, 2)

    panel_top = top_pad + random.randint(0, 2)
    query_top = top_pad + random.randint(0, 2)
    total_h = max(panel_top + panel.shape[0], query_top + query.shape[0]) + bottom_pad
    total_w = (
        left_pad
        + panel.shape[1]
        + gap_left
        + sep_width
        + gap_right
        + query.shape[1]
        + right_pad
    )

    grid = np.full((total_h, total_w), BG, dtype=int)
    panel_left = left_pad
    sep_left = panel_left + panel.shape[1] + gap_left
    query_left = sep_left + sep_width + gap_right

    _place(grid, panel, panel_top, panel_left)
    _place(grid, query, query_top, query_left)
    grid[:, sep_left : sep_left + sep_width] = SEP

    try:
        out = main(grid)
    except Exception:
        return None

    if not isinstance(out, np.ndarray):
        return None
    if out.size == 0 or np.all(out == BG):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
