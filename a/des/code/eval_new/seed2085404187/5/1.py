import random

import numpy as np


BG = 0
SIDES = ("left", "right", "top", "bottom")
MAX_GENERATION_ATTEMPTS = 400

SHAPE_TEMPLATES = (
    np.array([[1]], dtype=bool),
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
)

OVERLAP_TEMPLATES = (
    np.array([[1]], dtype=bool),
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
)


def _band_mask(shape: tuple[int, int], side: str, width: int) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    if side == "left":
        mask[:, :width] = True
    elif side == "right":
        mask[:, w - width :] = True
    elif side == "top":
        mask[:width, :] = True
    else:
        assert side == "bottom"
        mask[h - width :, :] = True
    return mask


def _detect_band(grid: np.ndarray) -> tuple[str, int, int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[tuple[str, int, int]] = []

    color = int(g[0, 0])
    if color != BG and np.all(g[:, 0] == color):
        width = 0
        while width < w and np.all(g[:, width] == color):
            width += 1
        if 1 <= width <= 3:
            candidates.append(("left", width, color))

    color = int(g[0, -1])
    if color != BG and np.all(g[:, -1] == color):
        width = 0
        while width < w and np.all(g[:, w - 1 - width] == color):
            width += 1
        if 1 <= width <= 3:
            candidates.append(("right", width, color))

    color = int(g[0, 0])
    if color != BG and np.all(g[0, :] == color):
        width = 0
        while width < h and np.all(g[width, :] == color):
            width += 1
        if 1 <= width <= 3:
            candidates.append(("top", width, color))

    color = int(g[-1, 0])
    if color != BG and np.all(g[-1, :] == color):
        width = 0
        while width < h and np.all(g[h - 1 - width, :] == color):
            width += 1
        if 1 <= width <= 3:
            candidates.append(("bottom", width, color))

    assert len(candidates) == 1, f"expected exactly one edge band, found {candidates}"
    return candidates[0]


def _merge_layers(base: np.ndarray, top: np.ndarray, cover_color: int) -> np.ndarray:
    base_mask = base != BG
    top_mask = top != BG
    out = np.zeros_like(base)
    only_base = base_mask & ~top_mask
    only_top = top_mask & ~base_mask
    overlap = base_mask & top_mask
    out[only_base] = base[only_base]
    out[only_top] = top[only_top]
    out[overlap] = cover_color
    return out


def _fold_left(grid: np.ndarray, band_width: int, cover_color: int) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    half = (w + 1) // 2
    offset = band_width - 1

    base = g[:, :half].copy()
    top = np.fliplr(g[:, w - half :]).copy()
    base[(base == BG) | (base == cover_color)] = BG
    top[(top == BG) | (top == cover_color)] = BG

    out_w = half + offset
    base_pad = np.zeros((h, out_w), dtype=int)
    top_pad = np.zeros((h, out_w), dtype=int)
    base_view = base_pad[:, :half]
    top_view = top_pad[:, offset : offset + half]
    base_view[:] = base
    top_view[:] = top
    return _merge_layers(base_pad, top_pad, cover_color)


def _fold_top(grid: np.ndarray, band_width: int, cover_color: int) -> np.ndarray:
    return _fold_left(grid.T, band_width, cover_color).T


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold the picture toward the band edge and merge the two layers with the band color.

    Input structure:
    - The background is black.
    - Exactly one grid edge carries a full-length solid color band of width 1-3.
    - Other non-black pixels form arbitrary colored patterns, and the band color is reserved for the edge band.

    Rule:
    - The band edge is the hinge side: left/right bands trigger a horizontal fold, top/bottom bands trigger a vertical fold.
    - Let the band width be b and the fold offset be b-1.
    - Along the fold axis, take the hinge-side half and the mirrored far-side half, each of size ceil(span/2); then shift the mirrored layer by the offset in the fold direction.
    - Ignore the band cells themselves. If exactly one layer contributes a non-background pixel, keep that color; if both layers contribute, recolor the overlap with the band color.

    Key insight:
    - The band position chooses the hinge edge, while the band width chooses how far the mirrored layer slides before the two folded layers are merged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    side, width, cover_color = _detect_band(g)
    band_cells = _band_mask(g.shape, side, width)
    assert np.any((g != BG) & ~band_cells), "expected non-band foreground pixels"
    assert not np.any((g == cover_color) & ~band_cells), "band color must be unique to the band"

    if side == "left":
        return _fold_left(g, width, cover_color)
    if side == "right":
        return np.fliplr(_fold_left(np.fliplr(g), width, cover_color))
    if side == "top":
        return _fold_top(g, width, cover_color)
    assert side == "bottom"
    return np.flipud(_fold_top(np.flipud(g), width, cover_color))


def _random_shape() -> np.ndarray:
    shape = random.choice(SHAPE_TEMPLATES).copy()
    if random.random() < 0.5:
        shape = np.fliplr(shape)
    turns = random.randrange(4)
    if turns:
        shape = np.rot90(shape, turns)
    return shape


def _place_shape(
    layer: np.ndarray,
    blocked: np.ndarray,
    color: int,
    row_lo: int,
    row_hi: int,
    col_lo: int,
    col_hi: int,
    shape: np.ndarray | None = None,
    attempts: int = 80,
) -> bool:
    shape = _random_shape() if shape is None else shape
    h, w = shape.shape
    max_row = row_hi - h
    max_col = col_hi - w
    if row_lo > max_row or col_lo > max_col:
        return False

    for _ in range(attempts):
        r = random.randint(row_lo, max_row)
        c = random.randint(col_lo, max_col)
        rr0 = max(0, r - 1)
        rr1 = min(layer.shape[0], r + h + 1)
        cc0 = max(0, c - 1)
        cc1 = min(layer.shape[1], c + w + 1)
        if blocked[rr0:rr1, cc0:cc1].any():
            continue
        target = layer[r : r + h, c : c + w]
        if np.any(target[shape]):
            continue
        target[shape] = color
        blocked[rr0:rr1, cc0:cc1] = True
        return True
    return False


def _orient_from_left(grid: np.ndarray, side: str) -> np.ndarray:
    if side == "left":
        return grid
    if side == "right":
        return np.fliplr(grid)
    if side == "top":
        return grid.T
    assert side == "bottom"
    return np.flipud(grid.T)


def _build_left_input(
    rows: int,
    half: int,
    odd: bool,
    band_width: int,
    band_color: int,
    overlap_branch: bool,
    pattern_colors: list[int],
) -> np.ndarray | None:
    full = 2 * half - int(odd)
    offset = band_width - 1

    base = np.zeros((rows, half), dtype=int)
    top = np.zeros((rows, half), dtype=int)
    blocked_base = np.zeros_like(base, dtype=bool)
    blocked_top = np.zeros_like(top, dtype=bool)
    blocked_base[:, :band_width] = True

    if odd and not overlap_branch:
        blocked_base[:, -1] = True
        blocked_top[:, -1] = True

    if overlap_branch:
        shape = random.choice(OVERLAP_TEMPLATES)
        sh, sw = shape.shape
        base_col_lo = max(band_width, offset)
        if base_col_lo > half - sw:
            return None
        row = random.randint(0, rows - sh)
        base_col = random.randint(base_col_lo, half - sw)
        top_col = base_col - offset
        color_a = random.choice(pattern_colors)
        alt = [color for color in pattern_colors if color != color_a]
        color_b = random.choice(alt or pattern_colors)
        if not _place_shape(base, blocked_base, color_a, row, row + sh, base_col, base_col + sw, shape=shape, attempts=1):
            return None
        if not _place_shape(top, blocked_top, color_b, row, row + sh, top_col, top_col + sw, shape=shape, attempts=1):
            return None

    if overlap_branch:
        base_row_lo, base_row_hi = 0, rows
        top_row_lo, top_row_hi = 0, rows
    else:
        split = rows // 2
        base_row_lo, base_row_hi = 0, split
        top_row_lo, top_row_hi = split, rows

    base_count = random.randint(2, 4)
    top_count = random.randint(2, 4)
    for _ in range(base_count):
        if not _place_shape(base, blocked_base, random.choice(pattern_colors), base_row_lo, base_row_hi, band_width, half):
            return None
    for _ in range(top_count):
        if not _place_shape(top, blocked_top, random.choice(pattern_colors), top_row_lo, top_row_hi, 0, half):
            return None

    if odd:
        if overlap_branch:
            top[:, -1] = base[:, -1]
        else:
            base[:, -1] = BG
            top[:, -1] = BG

    if not np.any(base[:, band_width:] != BG):
        return None
    if not np.any(top != BG):
        return None

    grid = np.zeros((rows, full), dtype=int)
    grid[:, :half] = base
    grid[:, full - half :] = np.fliplr(top)
    grid[:, :band_width] = band_color
    return grid


def _gen_attempt() -> np.ndarray | None:
    side = random.choice(SIDES)
    band_width = random.randint(1, 3)
    overlap_branch = bool(random.getrandbits(1))
    odd = bool(random.getrandbits(1))
    half = random.randint(max(5, band_width + 3), 14)
    rows = random.randint(8, 18)

    band_color = random.randint(1, 9)
    pattern_pool = [color for color in range(1, 10) if color != band_color]
    pattern_colors = random.sample(pattern_pool, random.randint(2, 3))

    left_grid = _build_left_input(rows, half, odd, band_width, band_color, overlap_branch, pattern_colors)
    if left_grid is None:
        return None

    inp = _orient_from_left(left_grid, side)
    try:
        found_side, found_width, found_color = _detect_band(inp)
        out = main(inp)
    except AssertionError:
        return None

    if (found_side, found_width, found_color) != (side, band_width, band_color):
        return None
    if overlap_branch != bool(np.any(out == band_color)):
        return None
    if inp.ndim != 2 or max(inp.shape) > 30 or max(out.shape) > 30:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
