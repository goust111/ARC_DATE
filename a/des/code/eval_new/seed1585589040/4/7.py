from __future__ import annotations

import numpy as np


BLACK = 0
SEPARATOR_COLOR = 5
PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
MAX_GENERATION_ATTEMPTS = 300
_RNG = np.random.default_rng()


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size, "expected a non-empty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_separator_span(grid: np.ndarray) -> tuple[int, int, int]:
    h, w = grid.shape
    candidates: list[tuple[int, int, int]] = []
    c = 0
    while c < w:
        color = int(grid[0, c])
        if color == BLACK or not bool(np.all(grid[:, c] == color)):
            c += 1
            continue
        c1 = c
        while c1 + 1 < w and bool(np.all(grid[:, c1 + 1] == color)):
            assert int(grid[0, c1 + 1]) == color
            c1 += 1
        if c > 0 and c1 < w - 1:
            candidates.append((c, c1, color))
        c = c1 + 1

    assert candidates, "expected a full-height divider"
    return max(candidates, key=lambda item: (item[1] - item[0] + 1, -item[0]))


def _extract_stamp(left_region: np.ndarray, divider_color: int) -> np.ndarray:
    mask = (left_region != BLACK) & (left_region != divider_color)
    r0, c0, r1, c1 = _bbox(mask)
    stamp = left_region[r0 : r1 + 1, c0 : c1 + 1].copy()
    stamp[stamp == divider_color] = BLACK
    return stamp


def _segment_color(segment: np.ndarray) -> int | None:
    colors = np.unique(segment[segment != BLACK])
    assert len(colors) <= 1, "each code-bar slot should use at most one visible color"
    if len(colors) == 0:
        return None
    return int(colors[0])


def _decode_axis(primary: np.ndarray, mirror: np.ndarray, step: int) -> dict[int, int]:
    assert primary.ndim == mirror.ndim == 1
    assert len(primary) == len(mirror)
    assert step > 0 and len(primary) % step == 0

    count = len(primary) // step
    codes: dict[int, int] = {}
    for idx in range(count):
        lo = idx * step
        hi = lo + step
        colors = {color for color in (_segment_color(primary[lo:hi]), _segment_color(mirror[lo:hi])) if color is not None}
        assert len(colors) <= 1, "mirrored code bars must agree per slot"
        if colors:
            codes[idx] = colors.pop()
    return codes


def _paste_nonblack(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = dst.shape
    if r0 >= h or c0 >= w:
        return
    rr = min(h, r0 + patch.shape[0])
    cc = min(w, c0 + patch.shape[1])
    if rr <= r0 or cc <= c0:
        return
    clipped = patch[: rr - r0, : cc - c0]
    mask = clipped != BLACK
    if not bool(mask.any()):
        return
    view = dst[r0:rr, c0:cc]
    view[mask] = clipped[mask]


def _render_canvas(
    stamp: np.ndarray,
    canvas_shape: tuple[int, int],
    row_codes: dict[int, int],
    col_codes: dict[int, int],
) -> np.ndarray:
    out = np.zeros(canvas_shape, dtype=int)
    sh, sw = stamp.shape
    rotated = np.rot90(stamp, -1).copy()

    for row_idx in sorted(row_codes):
        row_color = int(row_codes[row_idx])
        for col_idx in sorted(col_codes):
            col_color = int(col_codes[col_idx])
            patch = stamp if row_color == col_color else rotated
            _paste_nonblack(out, patch, row_idx * sh, col_idx * sw)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the stamp, decode the edge bars, and render the overlapped stamped canvas.

    Input structure:
    - Background is black, and a full-height grey divider splits the grid into a left stamp area and a right coded canvas area.
    - The left area contains one small multicolor stamp template on black.
    - The right area is a rectangular frame: its outermost top/bottom rows and left/right columns are sparse color code bars, and its black interior is the canvas.
    - The code-bar colors are drawn from two colors that also appear in the stamp.

    Rule:
    - Crop the non-black stamp from the left area and use its height and width as the canvas cell size.
    - Divide the right-side black interior into stamp-sized row and column slots.
    - A colored slot on the left bar selects a canvas row, and a colored slot on the top bar selects a canvas column; mirrored right/bottom bars repeat the same selections.
    - Stamp every selected row/column intersection: same row/column color means 0-degree placement, different colors mean a 90-degree clockwise rotation, and later rows overwrite earlier rows where painted cells overlap.

    Key insight:
    - The colored edge pixels do not mark individual canvas pixels; they mark stamp-sized row and column slots, and color agreement versus disagreement chooses the stamp orientation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sep0, sep1, divider_color = _find_separator_span(g)
    left_region = g[:, :sep0]
    right_region = g[:, sep1 + 1 :]
    assert left_region.size and right_region.shape[0] >= 3 and right_region.shape[1] >= 3

    stamp = _extract_stamp(left_region, divider_color)
    sh, sw = map(int, stamp.shape)

    canvas = right_region[1:-1, 1:-1]
    canvas_h, canvas_w = map(int, canvas.shape)
    assert canvas_h % sh == 0 and canvas_w % sw == 0

    top_bar = right_region[0, 1:-1]
    bottom_bar = right_region[-1, 1:-1]
    left_bar = right_region[1:-1, 0]
    right_bar = right_region[1:-1, -1]

    col_codes = _decode_axis(top_bar, bottom_bar, sw)
    row_codes = _decode_axis(left_bar, right_bar, sh)
    assert row_codes and col_codes, "expected at least one selected row and one selected column"

    return _render_canvas(stamp, (canvas_h, canvas_w), row_codes, col_codes)


def _make_stamp(height: int, width: int, colors: np.ndarray) -> np.ndarray:
    rows, cols = np.indices((height, width))
    for _ in range(80):
        mode = int(_RNG.integers(0, 4))
        if mode == 0:
            split = int(_RNG.integers(1, width))
            stamp = np.where(cols < split, colors[0], colors[1]).astype(int)
            accent = int(colors[2] if len(colors) == 3 else colors[0])
            stamp[(rows + cols) % 3 == 0] = accent
        elif mode == 1:
            split = int(_RNG.integers(1, height))
            stamp = np.where(rows < split, colors[0], colors[1]).astype(int)
            accent_col = int(_RNG.integers(0, width))
            accent = int(colors[2] if len(colors) == 3 else colors[0])
            stamp[:, accent_col] = accent
        elif mode == 2:
            threshold = int(_RNG.integers(1, height + width - 1))
            stamp = np.where(rows + cols < threshold, colors[0], colors[1]).astype(int)
            accent = int(colors[2] if len(colors) == 3 else colors[0])
            stamp[rows == (height - 1 - (cols % height))] = accent
        else:
            stamp = np.full((height, width), int(colors[0]), dtype=int)
            split = int(_RNG.integers(1, width))
            stamp[:, split:] = int(colors[1])
            block_h = int(_RNG.integers(1, height + 1))
            block_w = int(_RNG.integers(1, width + 1))
            accent = int(colors[2] if len(colors) == 3 else colors[0])
            stamp[:block_h, :block_w] = accent

        if all(bool(np.any(stamp == int(color))) for color in colors):
            return stamp.astype(int)
    raise RuntimeError("failed to build a sufficiently varied stamp")


def _assign_code_colors(indices: np.ndarray, code_colors: np.ndarray) -> dict[int, int]:
    assert len(indices) >= 2
    order = np.array(indices, dtype=int).copy()
    _RNG.shuffle(order)
    codes = {int(order[0]): int(code_colors[0]), int(order[1]): int(code_colors[1])}
    for idx in order[2:]:
        codes[int(idx)] = int(code_colors[int(_RNG.integers(0, 2))])
    return codes


def _paint_horizontal_bar(width: int, slot_width: int, codes: dict[int, int]) -> np.ndarray:
    bar = np.zeros(width, dtype=int)
    for idx, color in codes.items():
        pos = idx * slot_width + int(_RNG.integers(0, slot_width))
        bar[pos] = int(color)
    return bar


def _paint_vertical_bar(height: int, slot_height: int, codes: dict[int, int]) -> np.ndarray:
    bar = np.zeros(height, dtype=int)
    for idx, color in codes.items():
        pos = idx * slot_height + int(_RNG.integers(0, slot_height))
        bar[pos] = int(color)
    return bar


def _build_input(
    stamp: np.ndarray,
    left_width: int,
    separator_width: int,
    row_count: int,
    col_count: int,
    row_codes: dict[int, int],
    col_codes: dict[int, int],
) -> np.ndarray:
    sh, sw = map(int, stamp.shape)
    canvas_h = row_count * sh
    canvas_w = col_count * sw
    grid_h = canvas_h + 2
    right_width = canvas_w + 2
    grid_w = left_width + separator_width + right_width
    assert max(grid_h, grid_w) <= 30

    grid = np.zeros((grid_h, grid_w), dtype=int)
    grid[:, left_width : left_width + separator_width] = SEPARATOR_COLOR

    stamp_r = int(_RNG.integers(0, grid_h - sh + 1))
    stamp_c = int(_RNG.integers(0, left_width - sw + 1))
    grid[stamp_r : stamp_r + sh, stamp_c : stamp_c + sw] = stamp

    right_region = grid[:, left_width + separator_width :]
    top_bar = _paint_horizontal_bar(canvas_w, sw, col_codes)
    bottom_bar = _paint_horizontal_bar(canvas_w, sw, col_codes)
    left_bar = _paint_vertical_bar(canvas_h, sh, row_codes)
    right_bar = _paint_vertical_bar(canvas_h, sh, row_codes)

    right_region[0, 1:-1] = top_bar
    right_region[-1, 1:-1] = bottom_bar
    right_region[1:-1, 0] = left_bar
    right_region[1:-1, -1] = right_bar
    return grid


def _gen_attempt() -> np.ndarray | None:
    separator_width = int(_RNG.integers(1, 3))
    stamp_h = int(_RNG.choice(np.array([3, 4], dtype=int)))
    stamp_w = int(_RNG.integers(stamp_h + 1, 6))
    row_count = int(_RNG.integers(3, 6))

    left_width_min = stamp_w + 1
    max_col_count = min(4, (30 - separator_width - 2 - left_width_min) // stamp_w)
    if max_col_count < 2:
        return None
    col_count = int(_RNG.integers(2, max_col_count + 1))

    left_width_max = 30 - separator_width - (col_count * stamp_w + 2)
    if left_width_max < left_width_min:
        return None
    left_width = int(_RNG.integers(left_width_min, min(left_width_max, stamp_w + 5) + 1))

    color_count = int(_RNG.integers(2, 4))
    stamp_colors = _RNG.choice(PALETTE, size=color_count, replace=False).astype(int)
    code_colors = _RNG.choice(stamp_colors, size=2, replace=False).astype(int)
    stamp = _make_stamp(stamp_h, stamp_w, stamp_colors)

    active_rows = np.sort(_RNG.choice(np.arange(row_count), size=int(_RNG.integers(2, row_count + 1)), replace=False))
    active_cols = np.sort(_RNG.choice(np.arange(col_count), size=int(_RNG.integers(2, col_count + 1)), replace=False))
    row_codes = _assign_code_colors(active_rows, code_colors)
    col_codes = _assign_code_colors(active_cols, code_colors)

    inp = _build_input(stamp, left_width, separator_width, row_count, col_count, row_codes, col_codes)
    expected = _render_canvas(stamp, (row_count * stamp_h, col_count * stamp_w), row_codes, col_codes)

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid coded-stamp ARC input")
