import random

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 300
PATTERN_MODES = ("solid", "row_stripes", "col_stripes", "diag", "frame", "blocks")


def _nonblack_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid != BG)
    assert len(coords) > 0, "input must contain a non-black rectangle"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _compact_group(patch: np.ndarray, parity: int) -> np.ndarray:
    h, w = patch.shape
    rows: list[np.ndarray] = []
    max_w = 0
    for r in range(h):
        start = (parity - r) % 2
        row = patch[r, start:w:2]
        rows.append(row)
        max_w = max(max_w, int(row.shape[0]))

    out = np.full((h, max_w), BG, dtype=int)
    for r, row in enumerate(rows):
        out[r, : row.shape[0]] = row

    keep_rows = np.any(out != BG, axis=1)
    keep_cols = np.any(out != BG, axis=0)
    return out[np.ix_(keep_rows, keep_cols)]


def _pad_width(arr: np.ndarray, width: int) -> np.ndarray:
    if arr.shape[1] == width:
        return arr
    out = np.full((arr.shape[0], width), BG, dtype=int)
    out[:, : arr.shape[1]] = arr
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Separate a checkerboard-interleaved rectangle into its two compact subpatterns.

    The input contains exactly one full non-black rectangle on a black background.
    Inside that rectangle, local coordinates with even (row + col) belong to the
    upper output pattern and odd coordinates belong to the lower output pattern.
    Each pattern is compacted by removing the alternating blank columns implied by
    the interleaving, then the two recovered patterns are stacked with one black row
    between them.
    """
    g = np.asarray(grid, dtype=int)
    r0, c0, r1, c1 = _nonblack_bbox(g)
    patch = g[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(patch != BG), "foreground must form a full rectangle"

    a = _compact_group(patch, parity=0)
    b = _compact_group(patch, parity=1)
    width = max(a.shape[1], b.shape[1])
    a = _pad_width(a, width)
    b = _pad_width(b, width)
    sep = np.full((1, width), BG, dtype=int)
    return np.vstack([a, sep, b])


def _sample_colors(count: int) -> list[int]:
    return random.sample(COLORS, count)


def _make_solid(h: int, w: int) -> np.ndarray:
    return np.full((h, w), random.choice(COLORS), dtype=int)


def _make_row_stripes(h: int, w: int) -> np.ndarray:
    period = random.choice((2, 3, 4))
    palette = _sample_colors(min(period, 4))
    out = np.empty((h, w), dtype=int)
    for r in range(h):
        out[r, :] = palette[r % len(palette)]
    return out


def _make_col_stripes(h: int, w: int) -> np.ndarray:
    period = random.choice((2, 3, 4))
    palette = _sample_colors(min(period, 4))
    out = np.empty((h, w), dtype=int)
    for c in range(w):
        out[:, c] = palette[c % len(palette)]
    return out


def _make_diag(h: int, w: int) -> np.ndarray:
    period = random.choice((2, 3, 4))
    palette = _sample_colors(min(period, 4))
    rr, cc = np.indices((h, w))
    offset = random.randrange(len(palette))
    return np.asarray([palette[i] for i in ((rr + cc + offset) % len(palette)).ravel()], dtype=int).reshape(h, w)


def _make_frame(h: int, w: int) -> np.ndarray:
    if h < 3 or w < 3:
        return _make_diag(h, w)

    palette = _sample_colors(random.randint(2, 4))
    border = palette[0]
    inner_palette = palette[1:]
    out = np.full((h, w), border, dtype=int)
    inner_h = h - 2
    inner_w = w - 2
    if random.choice((False, True)):
        fill = np.full((inner_h, inner_w), random.choice(inner_palette), dtype=int)
    else:
        fill = _make_blocks(inner_h, inner_w, inner_palette)
    out[1:-1, 1:-1] = fill
    return out


def _segments(length: int) -> list[tuple[int, int]]:
    if length == 1:
        return [(0, 1)]
    max_cuts = min(2, length - 1)
    cut_count = random.randint(1, max_cuts)
    cuts = sorted(random.sample(range(1, length), cut_count))
    bounds = [0, *cuts, length]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def _make_blocks(h: int, w: int, palette: list[int] | None = None) -> np.ndarray:
    colors = palette if palette is not None else _sample_colors(random.randint(2, 4))
    row_spans = _segments(h)
    col_spans = _segments(w)
    out = np.empty((h, w), dtype=int)
    used: list[int] = []
    for rs, re in row_spans:
        for cs, ce in col_spans:
            color = random.choice(colors)
            used.append(color)
            out[rs:re, cs:ce] = color
    if len(set(used)) == 1 and len(colors) > 1:
        rs, re = row_spans[-1]
        cs, ce = col_spans[-1]
        out[rs:re, cs:ce] = colors[1]
    return out


def _make_pattern(h: int, w: int) -> np.ndarray:
    mode = random.choice(PATTERN_MODES)
    if mode == "solid":
        out = _make_solid(h, w)
    elif mode == "row_stripes":
        out = _make_row_stripes(h, w)
    elif mode == "col_stripes":
        out = _make_col_stripes(h, w)
    elif mode == "diag":
        out = _make_diag(h, w)
    elif mode == "frame":
        out = _make_frame(h, w)
    else:
        out = _make_blocks(h, w)

    if random.choice((False, True)):
        out = np.flipud(out)
    if random.choice((False, True)):
        out = np.fliplr(out)
    return out.copy()


def _interleave(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    assert a.shape == b.shape
    h, w = a.shape
    out = np.empty((h, 2 * w), dtype=int)
    for r in range(h):
        out[r, r % 2 :: 2] = a[r]
        out[r, 1 - (r % 2) :: 2] = b[r]
    return out


def _gen_attempt() -> np.ndarray | None:
    pattern_h = random.randint(3, 9)
    pattern_w = random.randint(2, 7)
    a = _make_pattern(pattern_h, pattern_w)
    b = _make_pattern(pattern_h, pattern_w)
    if np.array_equal(a, b):
        return None

    patch = _interleave(a, b)
    if len(np.unique(patch)) < 2:
        return None

    top = random.randint(1, 4)
    left = random.randint(1, 4)
    bottom = random.randint(1, 4)
    right = random.randint(1, 4)
    grid = np.full((pattern_h + top + bottom, patch.shape[1] + left + right), BG, dtype=int)
    grid[top : top + pattern_h, left : left + patch.shape[1]] = patch

    expected = np.vstack([a, np.zeros((1, pattern_w), dtype=int), b])
    try:
        if not np.array_equal(main(grid), expected):
            return None
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid interleaved-pattern input")
