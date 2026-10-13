from __future__ import annotations

import random

import numpy as np


RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7

SAFE_COLORS = (0, 1, 5, 6, 8, 9)
TEMPLATE_SIZE_RANGE = (3, 8)
TEMPLATE_PALETTE_SIZE_RANGE = (2, 5)
MAX_GENERATION_ATTEMPTS = 300


def _separator_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if bool(np.all(grid[r, :] == RED))]
    assert len(rows) == 1, "expected exactly one full-width red separator row"
    sep_row = int(rows[0])
    assert 0 < sep_row < grid.shape[0] - 1, "separator row must split the grid"
    return sep_row


def _selected_rows(control: np.ndarray, template_h: int) -> list[int]:
    rows: list[int] = []
    for r in range(control.shape[0]):
        marker = int(control[r, 0])
        if marker in (GREEN, ORANGE):
            assert r < template_h, "row marker exceeds template height"
            rows.append(r)
    assert rows, "expected at least one selected row"
    return rows


def _selected_cols(control: np.ndarray, template_w: int) -> list[int]:
    cols: list[int] = []
    for c in range(control.shape[1]):
        marker = int(control[0, c])
        if marker in (YELLOW, ORANGE):
            assert c < template_w, "column marker exceeds template width"
            cols.append(c)
    assert cols, "expected at least one selected column"
    return cols


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split on the red horizontal line, read row/column selectors below it,
    and keep only the chosen template rows and columns.

    The shared control corner may be orange when both template row 0 and
    column 0 are selected simultaneously.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep_row = _separator_row(g)
    template = g[:sep_row, :]
    control = g[sep_row + 1 :, :]

    template_h, template_w = map(int, template.shape)
    control_h, control_w = map(int, control.shape)
    assert template_h > 0 and template_w > 0, "template must be non-empty"
    assert control_h >= template_h, "control area must cover all template rows"
    assert control_w >= template_w, "control area must cover all template columns"

    if control_h > 1 and control_w > 1:
        marker_body = np.isin(control[1:, 1:], (GREEN, YELLOW, ORANGE))
        assert not bool(marker_body.any()), "markers may appear only on the top row or left column"

    rows = _selected_rows(control, template_h)
    cols = _selected_cols(control, template_w)
    return template[np.ix_(rows, cols)].copy()


# --------------- generator ---------------


def _segments(total: int, parts: int) -> list[tuple[int, int]]:
    assert 1 <= parts <= total
    if parts == 1:
        return [(0, total)]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    bounds = [0] + cuts + [total]
    return [(bounds[i], bounds[i + 1]) for i in range(parts)]


def _make_formula_template(h: int, w: int, palette: list[int]) -> np.ndarray:
    rr, cc = np.indices((h, w))
    palette_arr = np.asarray(palette, dtype=int)
    a, b = random.choice(((1, 1), (1, 2), (2, 1), (2, 3), (3, 1)))
    shift = random.randrange(len(palette))
    template = palette_arr[(a * rr + b * cc + shift) % len(palette)]
    if random.choice([False, True]):
        template = np.flipud(template)
    if random.choice([False, True]):
        template = np.fliplr(template)
    return template.astype(int)


def _make_block_template(h: int, w: int, palette: list[int]) -> np.ndarray:
    row_parts = random.randint(2, min(h, 4))
    col_parts = random.randint(2, min(w, 4))
    row_segments = _segments(h, row_parts)
    col_segments = _segments(w, col_parts)
    palette_arr = np.asarray(palette, dtype=int)
    template = np.empty((h, w), dtype=int)
    offset = random.randrange(len(palette))
    for i, (r0, r1) in enumerate(row_segments):
        for j, (c0, c1) in enumerate(col_segments):
            idx = (offset + i + 2 * j + random.randrange(len(palette))) % len(palette)
            template[r0:r1, c0:c1] = int(palette_arr[idx])
    return template


def _make_noise_template(h: int, w: int, palette: list[int]) -> np.ndarray:
    weights = np.random.dirichlet(np.ones(len(palette)))
    return np.random.choice(np.asarray(palette, dtype=int), size=(h, w), p=weights).astype(int)


def _stabilize_template(template: np.ndarray, palette: list[int]) -> np.ndarray:
    out = template.copy()
    h, w = out.shape

    if np.unique(out).size == 1:
        r = random.randrange(h)
        c = random.randrange(w)
        choices = [color for color in palette if color != int(out[r, c])]
        out[r, c] = random.choice(choices)

    edits = random.randint(1, max(1, (h * w) // 6))
    for _ in range(edits):
        r = random.randrange(h)
        c = random.randrange(w)
        choices = [color for color in palette if color != int(out[r, c])]
        if choices:
            out[r, c] = random.choice(choices)
    return out


def _make_template(h: int, w: int, palette: list[int]) -> np.ndarray:
    mode = random.choice(("formula", "blocks", "noise"))
    if mode == "formula":
        template = _make_formula_template(h, w, palette)
    elif mode == "blocks":
        template = _make_block_template(h, w, palette)
    else:
        template = _make_noise_template(h, w, palette)
    return _stabilize_template(template, palette)


def _random_subset(length: int, include_zero: bool) -> list[int]:
    assert length >= 1
    if length == 1:
        return [0]

    rest = list(range(1, length))
    if include_zero:
        extra = random.randint(0, len(rest))
        chosen = {0, *random.sample(rest, extra)}
    else:
        size = random.randint(1, len(rest))
        chosen = set(random.sample(rest, size))
    return sorted(chosen)


def _sample_selections(h: int, w: int) -> tuple[list[int], list[int]]:
    corner_mode = random.choice(("none", "row_only", "col_only", "both"))
    rows = _random_subset(h, include_zero=corner_mode in ("row_only", "both"))
    cols = _random_subset(w, include_zero=corner_mode in ("col_only", "both"))
    return rows, cols


def _build_control(h: int, w: int, bg: int, rows: list[int], cols: list[int]) -> np.ndarray:
    control = np.full((h, w), int(bg), dtype=int)
    control[rows, 0] = GREEN
    control[0, cols] = YELLOW

    if 0 in rows and 0 in cols:
        control[0, 0] = ORANGE
    elif 0 in rows:
        control[0, 0] = GREEN
    elif 0 in cols:
        control[0, 0] = YELLOW
    return control


def _gen_attempt() -> np.ndarray | None:
    template_h = random.randint(*TEMPLATE_SIZE_RANGE)
    template_w = random.randint(*TEMPLATE_SIZE_RANGE)

    bg = random.choice(SAFE_COLORS)
    palette_pool = [color for color in SAFE_COLORS if color != bg]
    palette_size = random.randint(
        TEMPLATE_PALETTE_SIZE_RANGE[0],
        min(TEMPLATE_PALETTE_SIZE_RANGE[1], len(palette_pool)),
    )
    palette = random.sample(palette_pool, palette_size)

    template = _make_template(template_h, template_w, palette)
    rows, cols = _sample_selections(template_h, template_w)
    expected = template[np.ix_(rows, cols)]
    control = _build_control(template_h, template_w, bg, rows, cols)

    grid = np.vstack(
        (
            template,
            np.full((1, template_w), RED, dtype=int),
            control,
        )
    )

    try:
        actual = main(grid)
    except Exception:
        return None

    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
