import random

import numpy as np


BG = 0
RED = 2
GREEN = 3
YELLOW = 4
OBJECT_COLORS = [1, 5, 6, 7, 8, 9]
GEN_RETRIES = 400


def _frame_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros(shape, dtype=bool)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _color_bboxes(grid: np.ndarray) -> dict[int, tuple[int, int, int, int]]:
    bboxes: dict[int, tuple[int, int, int, int]] = {}
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != BG):
        pts = np.argwhere(grid == color)
        if pts.size == 0:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        bboxes[color] = (int(r0), int(c0), int(r1), int(c1))
    return bboxes


def _marker_color(grid: np.ndarray, color: int) -> int:
    cols = np.where(grid[-1] == BG)[0]
    assert len(cols) > 0, "bottom row must contain at least one background cell for the direction marker"
    out_color = RED if color == RED else GREEN
    return int(cols[0]), int(out_color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct two partially overlapping rectangle frames and mark the hidden lower layer.
    """
    g = np.asarray(grid, dtype=int)
    bboxes = _color_bboxes(g)
    assert len(bboxes) == 2, "expected exactly two visible object colors"

    colors = sorted(bboxes)
    frame_masks = {color: _frame_mask(g.shape, bboxes[color]) for color in colors}
    overlap = frame_masks[colors[0]] & frame_masks[colors[1]]
    assert overlap.any(), "the reconstructed frames must overlap"

    visible_overlap = {int(v) for v in np.unique(g[overlap]) if int(v) != BG}
    assert len(visible_overlap) == 1, "overlap cells must show exactly one visible color"
    top_color = next(iter(visible_overlap))
    lower_color = colors[0] if colors[1] == top_color else colors[1]

    out = g.copy()
    out[overlap & (g == top_color)] = YELLOW

    r0a, c0a, r1a, c1a = bboxes[colors[0]]
    r0b, c0b, r1b, c1b = bboxes[colors[1]]
    center_ra = r0a + r1a
    center_rb = r0b + r1b
    center_ca = c0a + c1a
    center_cb = c0b + c1b

    horizontal = center_ra == center_rb
    vertical = center_ca == center_cb
    assert horizontal ^ vertical, "rectangle centers must differ along exactly one axis"

    marker_col, marker_color = _marker_color(g, RED if horizontal else GREEN)
    out[-1, marker_col] = marker_color
    return out


def _horizontal_case(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(12, 20)
    w = rng.randint(14, 24)

    rect_h = rng.randint(4, min(8, h - 4))
    r0 = rng.randint(1, h - rect_h - 3)
    r1 = r0 + rect_h - 1

    wa = rng.randint(5, min(10, w - 4))
    wb = rng.randint(5, min(10, w - 4))
    overlap_len = rng.randint(2, min(wa, wb) - 2)
    left_extra = wa - overlap_len
    right_extra = wb - overlap_len
    total_w = left_extra + overlap_len + right_extra
    if total_w + 2 > w:
        return None

    start = rng.randint(1, w - total_w - 1)
    a = (r0, start, r1, start + wa - 1)
    b = (r0, start + left_extra, r1, start + total_w - 1)
    return _build_case(rng, (h, w), a, b)


def _vertical_case(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(14, 24)
    w = rng.randint(12, 20)

    rect_w = rng.randint(4, min(8, w - 4))
    c0 = rng.randint(1, w - rect_w - 3)
    c1 = c0 + rect_w - 1

    ha = rng.randint(5, min(10, h - 4))
    hb = rng.randint(5, min(10, h - 4))
    overlap_len = rng.randint(2, min(ha, hb) - 2)
    top_extra = ha - overlap_len
    bottom_extra = hb - overlap_len
    total_h = top_extra + overlap_len + bottom_extra
    if total_h + 2 > h:
        return None

    start = rng.randint(1, h - total_h - 1)
    a = (start, c0, start + ha - 1, c1)
    b = (start + top_extra, c0, start + total_h - 1, c1)
    return _build_case(rng, (h, w), a, b)


def _build_case(
    rng: random.Random,
    shape: tuple[int, int],
    bbox_a: tuple[int, int, int, int],
    bbox_b: tuple[int, int, int, int],
) -> np.ndarray | None:
    h, w = shape
    color_a, color_b = rng.sample(OBJECT_COLORS, 2)
    top_color = rng.choice([color_a, color_b])

    frame_a = _frame_mask(shape, bbox_a)
    frame_b = _frame_mask(shape, bbox_b)
    overlap = frame_a & frame_b
    if not overlap.any():
        return None

    if np.any(overlap[-1]):
        return None

    grid = np.full((h, w), BG, dtype=int)
    grid[frame_a] = color_a
    grid[frame_b] = color_b
    grid[overlap] = top_color

    bboxes = _color_bboxes(grid)
    if bboxes.get(color_a) != bbox_a or bboxes.get(color_b) != bbox_b:
        return None

    visible_overlap = {int(v) for v in np.unique(grid[overlap]) if int(v) != BG}
    if visible_overlap != {top_color}:
        return None

    if not np.any(grid[-1] == BG):
        return None

    non_bg_colors = {int(v) for v in np.unique(grid) if int(v) != BG}
    if non_bg_colors != {color_a, color_b}:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        if rng.random() < 0.5:
            grid = _horizontal_case(rng)
        else:
            grid = _vertical_case(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
