from __future__ import annotations

import random

import numpy as np


BG = 0
SEPARATOR = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
GAP_SIDES = ("top", "bottom", "left", "right")
BLOCK_SHAPES = ((1, 1), (1, 2), (2, 1))
GEN_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 160


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.where(mask)
    assert rows.size > 0, "bbox requires a non-empty mask"
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _boundary_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _find_separator_column(grid: np.ndarray, bg: int) -> tuple[int, int]:
    candidates: list[tuple[int, int]] = []
    for c in range(grid.shape[1]):
        uniq = np.unique(grid[:, c])
        if uniq.size == 1 and int(uniq[0]) != bg:
            candidates.append((c, int(uniq[0])))
    assert len(candidates) == 1, "expected a unique full-height separator column"
    return candidates[0]


def _frame_patch(
    color: int,
    height: int,
    width: int,
    gap_side: str,
    gap_start: int,
    gap_len: int,
) -> np.ndarray:
    patch = np.full((height, width), BG, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    if gap_side == "top":
        patch[0, gap_start : gap_start + gap_len] = BG
    elif gap_side == "bottom":
        patch[-1, gap_start : gap_start + gap_len] = BG
    elif gap_side == "left":
        patch[gap_start : gap_start + gap_len, 0] = BG
    else:
        assert gap_side == "right"
        patch[gap_start : gap_start + gap_len, -1] = BG
    return patch


def _complete_frame(color: int, height: int, width: int) -> np.ndarray:
    patch = np.full((height, width), BG, dtype=int)
    patch[_boundary_mask(height, width)] = color
    return patch


def _can_place(
    occupied: np.ndarray,
    top: int,
    left: int,
    height: int,
    width: int,
    pad: int = 1,
) -> bool:
    h, w = occupied.shape
    if top < 0 or left < 0 or top + height > h or left + width > w:
        return False
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    return not occupied[r0:r1, c0:c1].any()


def _place_patch(
    grid: np.ndarray,
    occupied: np.ndarray,
    patch: np.ndarray,
    row_min: int,
    row_max: int,
    col_min: int,
    col_max: int,
) -> tuple[int, int] | None:
    ph, pw = patch.shape
    if row_max - row_min < ph or col_max - col_min < pw:
        return None

    for _ in range(PLACEMENT_ATTEMPTS):
        top = random.randint(row_min, row_max - ph)
        left = random.randint(col_min, col_max - pw)
        if not _can_place(occupied, top, left, ph, pw):
            continue
        grid[top : top + ph, left : left + pw][patch != BG] = patch[patch != BG]
        occupied[top : top + ph, left : left + pw][patch != BG] = True
        return top, left
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair each gapped rectangular frame on the left using its same-colored signal block on the right.

    Input structure:
    - The background is black, and a single full-height grey column splits the grid into left and right halves.
    - The left half contains one-pixel-thick rectangular frames in distinct colors; each frame is missing one contiguous 2-4 cell segment from exactly one side.
    - The right half contains one small solid block per frame color, with size 1x1, 1x2, or 2x1.

    Rule:
    - Identify the separator column and pair left frames with right blocks by color.
    - Erase every right-side signal block back to background.
    - For each left-side color, recover the frame's bounding box and repaint the full rectangular border in that color, filling the missing segment while leaving the frame interior unchanged.

    Key insight:
    - The right-side block is only a color-matching activation cue; the cells to fill are determined entirely by the missing segment on the left frame's own rectangular border.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    sep_col, sep_color = _find_separator_column(g, bg)

    left = g[:, :sep_col]
    right = g[:, sep_col + 1 :]
    left_colors = set(map(int, np.unique(left))) - {bg}
    right_colors = set(map(int, np.unique(right))) - {bg}
    repair_colors = left_colors & right_colors
    assert repair_colors, "expected at least one paired frame/signal color"
    assert sep_color not in repair_colors, "separator color must not be used as a repair color"

    out = g.copy()
    right_out = out[:, sep_col + 1 :]

    for color in sorted(repair_colors):
        left_mask = left == color
        right_mask = right == color
        assert left_mask.any() and right_mask.any(), "each repair color must appear on both sides"

        r0, c0, r1, c1 = _bbox(left_mask)
        local = left_mask[r0 : r1 + 1, c0 : c1 + 1]
        expected = _boundary_mask(r1 - r0 + 1, c1 - c0 + 1)
        assert np.all(~local | expected), "left object must lie on a rectangular border"

        out[r0 : r1 + 1, c0 : c1 + 1][expected] = color
        right_out[right_mask] = bg

    return out


def _gen_attempt() -> np.ndarray | None:
    frame_count = random.randint(2, 4)
    height = random.randint(max(18, 4 * frame_count + 6), 28)
    left_width = random.randint(11, 17)
    right_width = random.randint(6, min(11, 29 - left_width))
    width = left_width + 1 + right_width

    grid = np.full((height, width), BG, dtype=int)
    occupied = np.zeros_like(grid, dtype=bool)
    grid[:, left_width] = SEPARATOR
    occupied[:, left_width] = True

    colors = random.sample(COLOR_POOL, frame_count)
    sides = random.sample(GAP_SIDES, frame_count)

    block_shapes = list(BLOCK_SHAPES)
    if frame_count == 4:
        block_shapes.append(random.choice(BLOCK_SHAPES))
    random.shuffle(block_shapes)
    block_shapes = block_shapes[:frame_count]

    random.shuffle(colors)
    expected = grid.copy()

    for color, gap_side, block_shape in zip(colors, sides, block_shapes):
        frame_h = random.randint(5, 7)
        frame_w = random.randint(5, min(8, left_width - 2))
        side_len = frame_w if gap_side in {"top", "bottom"} else frame_h
        gap_len = random.randint(2, min(4, side_len - 1))
        gap_start = random.randint(0, side_len - gap_len)

        frame = _frame_patch(color, frame_h, frame_w, gap_side, gap_start, gap_len)
        frame_pos = _place_patch(
            grid,
            occupied,
            frame,
            row_min=1,
            row_max=height - 1,
            col_min=1,
            col_max=left_width - 1,
        )
        if frame_pos is None:
            return None

        fr, fc = frame_pos
        full_frame = _complete_frame(color, frame_h, frame_w)
        expected[fr : fr + frame_h, fc : fc + frame_w][full_frame != BG] = full_frame[full_frame != BG]

        bh, bw = block_shape
        block = np.full((bh, bw), color, dtype=int)
        block_pos = _place_patch(
            grid,
            occupied,
            block,
            row_min=1,
            row_max=height - 1,
            col_min=left_width + 2,
            col_max=width - 1,
        )
        if block_pos is None:
            return None

    produced = main(grid)
    if not np.array_equal(produced, expected):
        return None

    if len(set(map(int, np.unique(grid[:, :left_width]))) - {BG}) != frame_count:
        return None
    if len(set(map(int, np.unique(grid[:, left_width + 1 :]))) - {BG}) != frame_count:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC sample")
