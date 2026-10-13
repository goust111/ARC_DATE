from __future__ import annotations

import random

import numpy as np


def _infer_panel_colors(left: np.ndarray, right: np.ndarray) -> tuple[int, int, int]:
    left_colors = set(map(int, np.unique(left)))
    right_colors = set(map(int, np.unique(right)))

    common = left_colors & right_colors
    assert len(common) == 1, "left and right masks must share only the background color"
    bg = int(next(iter(common)))

    left_fg = left_colors - {bg}
    right_fg = right_colors - {bg}
    assert len(left_fg) == 1 and len(right_fg) == 1, "each panel must be a binary mask"
    lf = int(next(iter(left_fg)))
    rf = int(next(iter(right_fg)))
    assert lf != rf, "the two panels must use different foreground colors"
    return bg, lf, rf


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the aligned XOR difference of two binary panels by the row color in the center separator.

    Input structure:
    - The grid has odd width; its exact center column is a full-height separator strip.
    - The left and right sides are equally sized binary masks with one shared background color.
    - The left and right foreground colors differ, while the separator gives one color per row.

    Rule:
    - Return a grid with the size of one side panel, initially filled with the shared background.
    - Compare the left and right panels cell by cell; only cells where exactly one side is foreground are differences.
    - Paint each difference cell with the separator color from that same row.
    - If the separator pixel in a row is background, that row is closed and its differences stay background.

    Key insight:
    - The separator is both a row-wise color legend and a row-wise gate; side foreground colors only define masks.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    h, w = g.shape
    assert w % 2 == 1 and w >= 3, "expected an odd-width grid with a center separator"
    mid = w // 2
    left = g[:, :mid]
    right = g[:, mid + 1 :]
    seam = g[:, mid]
    assert left.shape == right.shape and left.shape[0] == h

    bg, _left_fg, _right_fg = _infer_panel_colors(left, right)
    closed = seam == bg
    assert np.any(~closed), "at least one separator row must be colored"
    assert np.any(closed), "at least one separator row must be background-gated"
    assert int(np.count_nonzero(~closed)) > int(np.count_nonzero(closed)), "colored separator rows must be the majority"

    left_mask = left != bg
    right_mask = right != bg
    diff = np.logical_xor(left_mask, right_mask)

    out = np.full(left.shape, bg, dtype=int)
    open_diff = diff & (seam[:, None] != bg)
    rows, cols = np.where(open_diff)
    out[rows, cols] = seam[rows]
    return out


# --------------- generator ---------------

GEN_ATTEMPTS = 300
COLORS = tuple(range(10))


def _expected_from_masks(left_mask: np.ndarray, right_mask: np.ndarray, seam: np.ndarray, bg: int) -> np.ndarray:
    out = np.full(left_mask.shape, int(bg), dtype=int)
    diff = np.logical_xor(left_mask, right_mask) & (seam[:, None] != int(bg))
    rows, cols = np.where(diff)
    out[rows, cols] = seam[rows]
    return out


def _paint_random_features(left_mask: np.ndarray, right_mask: np.ndarray) -> None:
    h, w = left_mask.shape
    feature_count = random.randint(5, 10)
    for _ in range(feature_count):
        fh = random.choice([1, 1, 2, 2, 3])
        fw = random.choice([1, 2, 2, 3, 4])
        if fh > h or fw > w:
            continue
        r0 = random.randint(0, h - fh)
        c0 = random.randint(0, w - fw)
        target = random.choices(["left", "right", "both"], weights=[3, 3, 4], k=1)[0]
        if target in ("left", "both"):
            left_mask[r0 : r0 + fh, c0 : c0 + fw] = True
        if target in ("right", "both"):
            right_mask[r0 : r0 + fh, c0 : c0 + fw] = True


def _choose_distinct_columns(width: int, count: int) -> list[int]:
    return random.sample(range(width), count)


def _build_attempt() -> np.ndarray | None:
    h = random.randint(6, 14)
    panel_w = random.randint(5, 14)

    bg = random.choice(COLORS)
    side_colors = random.sample([c for c in COLORS if c != bg], 2)
    left_color, right_color = side_colors
    seam_pool = [c for c in COLORS if c not in (bg, left_color, right_color)]
    seam_colors = random.sample(seam_pool, random.randint(2, min(4, len(seam_pool))))

    max_closed = max(1, min(h // 3, h - 4))
    closed_count = random.randint(1, max_closed)
    closed_rows = set(random.sample(range(h), closed_count))
    open_rows = [r for r in range(h) if r not in closed_rows]
    if len(open_rows) < 4 or len(open_rows) <= len(closed_rows):
        return None

    left_row, right_row, no_diff_row, mixed_row = random.sample(open_rows, 4)
    closed_diff_row = random.choice(sorted(closed_rows))

    seam = np.full(h, int(bg), dtype=int)
    shuffled_open = open_rows[:]
    random.shuffle(shuffled_open)
    for i, r in enumerate(shuffled_open):
        seam[r] = int(seam_colors[i % len(seam_colors)])
    seam[left_row] = int(seam_colors[0])
    seam[right_row] = int(seam_colors[1])
    seam[mixed_row] = int(seam_colors[random.randrange(len(seam_colors))])

    left_mask = np.zeros((h, panel_w), dtype=bool)
    right_mask = np.zeros((h, panel_w), dtype=bool)
    _paint_random_features(left_mask, right_mask)

    l_only_col, l_both_col = _choose_distinct_columns(panel_w, 2)
    r_only_col, r_both_col = _choose_distinct_columns(panel_w, 2)
    mixed_diff_col, mixed_same_col = _choose_distinct_columns(panel_w, 2)
    closed_col = random.randrange(panel_w)
    no_diff_col = random.randrange(panel_w)

    left_mask[left_row, l_only_col] = True
    right_mask[left_row, l_only_col] = False
    left_mask[left_row, l_both_col] = True
    right_mask[left_row, l_both_col] = True

    left_mask[right_row, r_only_col] = False
    right_mask[right_row, r_only_col] = True
    left_mask[right_row, r_both_col] = True
    right_mask[right_row, r_both_col] = True

    left_mask[mixed_row, mixed_diff_col] = random.choice([True, False])
    right_mask[mixed_row, mixed_diff_col] = not left_mask[mixed_row, mixed_diff_col]
    left_mask[mixed_row, mixed_same_col] = True
    right_mask[mixed_row, mixed_same_col] = True

    right_mask[no_diff_row, :] = left_mask[no_diff_row, :]
    left_mask[no_diff_row, no_diff_col] = True
    right_mask[no_diff_row, no_diff_col] = True

    if random.choice([True, False]):
        left_mask[closed_diff_row, closed_col] = True
        right_mask[closed_diff_row, closed_col] = False
    else:
        left_mask[closed_diff_row, closed_col] = False
        right_mask[closed_diff_row, closed_col] = True

    expected = _expected_from_masks(left_mask, right_mask, seam, bg)
    xor = np.logical_xor(left_mask, right_mask)
    open_mask = seam != bg
    closed_mask = seam == bg

    if not np.any(xor[closed_mask]):
        return None
    if np.any(expected[closed_mask] != bg):
        return None
    if not np.any((left_mask & ~right_mask) & open_mask[:, None]):
        return None
    if not np.any((right_mask & ~left_mask) & open_mask[:, None]):
        return None
    if not np.any((left_mask & right_mask) & open_mask[:, None]):
        return None
    if np.any(xor[no_diff_row]):
        return None
    out_colors = set(map(int, np.unique(expected))) - {int(bg)}
    if len(out_colors) < 2:
        return None

    out_area = int(np.count_nonzero(expected != bg))
    if out_area < 3 or out_area > (h * panel_w) // 2:
        return None

    grid = np.full((h, panel_w * 2 + 1), int(bg), dtype=int)
    grid[:, :panel_w][left_mask] = int(left_color)
    grid[:, panel_w] = seam
    grid[:, panel_w + 1 :][right_mask] = int(right_color)

    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid row-seam XOR task input")
