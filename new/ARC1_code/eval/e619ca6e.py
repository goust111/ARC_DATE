from __future__ import annotations

import numpy as np

from arc_utils import bbox_h, bbox_w, colors_present, crop_bbox, find_components, get_bg, paste
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy every solid foreground rectangle repeatedly along both downward diagonal directions, clipping copies at the grid edge.

    Input structure
    - The background is the dominant color.
    - There is one foreground color.
    - Each 4-connected foreground object is a filled axis-aligned rectangle.

    Rule
    - Keep the original rectangles.
    - For a rectangle of height h and width w, stamp the same rectangle at offsets (h, -w), (2h, -2w), ... and (h, w), (2h, 2w), ...
    - Drop only the out-of-bounds part of any stamped rectangle and stop once a trajectory is fully outside the grid.

    Key insight
    - Each rectangle is both the sprite and the stride: its height controls the downward step and its width controls the sideways step.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(get_bg(g))
    fg_colors = colors_present(g, exclude={bg})
    assert len(fg_colors) == 1, "expected exactly one foreground color"

    color = int(fg_colors[0])
    components = find_components(g, bg=bg, connectivity=4, monochromatic=True, colors=[color])
    assert components, "expected at least one foreground rectangle"

    H, W = g.shape
    out = g.copy()
    for comp in components:
        b = comp["bbox"]
        h = bbox_h(b)
        w = bbox_w(b)
        assert int(comp["area"]) == h * w, "foreground objects must be solid rectangles"

        r0, c0, _r1, _c1 = b
        sprite = crop_bbox(g, b)
        assert np.all(sprite == color), "rectangle bbox must contain only foreground color"

        for dc in (-w, w):
            r = r0 + h
            c = c0 + dc
            while r < H and c < W and c + w > 0:
                paste(out, sprite, r, c, clip=True)
                r += h
                c += dc

    return out

# --------------- generator (o5x) ---------------

# Canvas and palette.
GRID_HEIGHT_RANGE, GRID_WIDTH_RANGE = (25, 29), (22, 29)
BACKGROUND_COLOR, FOREGROUND_COLOR = 0, 3
# Rectangle counts, dimensions, and placement margins.
RECT_COUNT_RANGE = (1, 3)
RECT_DIM_CHOICES = [(2, 3), (2, 4), (2, 5), (3, 2), (3, 3), (4, 2)]
MIN_ROW_ANCHOR_DELTA, MIN_COL_ANCHOR_DELTA = 2, 4
MIN_VERTICAL_ROOM_STEPS, LEFT_START_STEPS, RIGHT_START_STEPS, COLUMN_PADDING = 4, 1, 2, 1
# Diagonal stamp constraints.
TARGET_STAMP_RANGE, TARGET_FALLBACK_TOLERANCE = (2, 9), 1
MIN_STAMPS_PER_DIRECTION, MAX_STAMPS_PER_DIRECTION = 2, 12
TOTAL_STAMPS_RANGE, CLIPPED_STAMPS_RANGE = (5, 25), (2, 4)
# Retry budget.
MAX_GENERATION_ATTEMPTS = 300


def _gen_interval_gap(a0: int, a1: int, b0: int, b1: int) -> int:
    if a1 < b0:
        return b0 - a1 - 1
    if b1 < a0:
        return a0 - b1 - 1
    return 0


def _gen_boxes_ok(rects: list[tuple[int, int, int, int]], r: int, c: int, h: int, w: int) -> bool:
    r1 = r + h - 1
    c1 = c + w - 1
    for pr, pc, ph, pw in rects:
        pr1 = pr + ph - 1
        pc1 = pc + pw - 1
        row_gap = _gen_interval_gap(r, r1, pr, pr1)
        col_gap = _gen_interval_gap(c, c1, pc, pc1)
        if row_gap == 0 and col_gap == 0:
            return False
        if abs(r - pr) < MIN_ROW_ANCHOR_DELTA or abs(c - pc) < MIN_COL_ANCHOR_DELTA:
            return False
    return True


def _gen_stamp_stats(H: int, W: int, rects: list[tuple[int, int, int, int]]) -> tuple[list[int], int, int]:
    per_direction = []
    total = 0
    clipped = 0
    for r0, c0, h, w in rects:
        k_vert = (H - 1 - r0) // h
        left = min(k_vert, (c0 + w - 1) // w)
        right = min(k_vert, (W - 1 - c0) // w)
        per_direction.extend([left, right])
        total += left + right
        if left > 0:
            lc = c0 - left * w
            if lc < 0 and lc + w > 0:
                clipped += 1
        if right > 0:
            rc = c0 + right * w
            if rc < W and rc + w > W:
                clipped += 1
    return per_direction, total, clipped


def _gen_rect_options(H: int, W: int, rects: list[tuple[int, int, int, int]], target: int):
    """Enumerate viable placements, preferring candidates near the target stamp count."""
    options = []
    fallback_options = []
    for h, w in random.sample(RECT_DIM_CHOICES, len(RECT_DIM_CHOICES)):
        max_r = H - MIN_VERTICAL_ROOM_STEPS * h
        min_c = LEFT_START_STEPS * w + COLUMN_PADDING
        max_c = W - RIGHT_START_STEPS * w - COLUMN_PADDING
        if max_r < 0 or min_c > max_c:
            continue

        for r in range(max_r + 1):
            for c in range(min_c, max_c + 1):
                if not _gen_boxes_ok(rects, r, c, h, w):
                    continue
                per_direction, _total, _clipped = _gen_stamp_stats(H, W, [(r, c, h, w)])
                if not all(MIN_STAMPS_PER_DIRECTION <= x <= MAX_STAMPS_PER_DIRECTION for x in per_direction):
                    continue
                item = (r, c, h, w, per_direction)
                fallback_options.append(item)
                if target in per_direction:
                    options.append(item)

    if options:
        return options

    near_target = [item for item in fallback_options if min(abs(x - target) for x in item[4]) <= TARGET_FALLBACK_TOLERANCE]
    return near_target or fallback_options


def _gen_choose_rect(H: int, W: int, rects: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int] | None:
    target = random.randint(*TARGET_STAMP_RANGE)
    options = _gen_rect_options(H, W, rects, target)
    if not options:
        return None
    r, c, h, w, _per_direction = random.choice(options)
    return r, c, h, w


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_HEIGHT_RANGE)
    W = random.randint(*GRID_WIDTH_RANGE)
    rects: list[tuple[int, int, int, int]] = []
    for _ in range(random.randint(*RECT_COUNT_RANGE)):
        rect = _gen_choose_rect(H, W, rects)
        if rect is None:
            return None
        rects.append(rect)

    per_direction, total_stamps, clipped = _gen_stamp_stats(H, W, rects)
    if not all(MIN_STAMPS_PER_DIRECTION <= x <= MAX_STAMPS_PER_DIRECTION for x in per_direction):
        return None
    if not (TOTAL_STAMPS_RANGE[0] <= total_stamps <= TOTAL_STAMPS_RANGE[1]):
        return None
    if not (CLIPPED_STAMPS_RANGE[0] <= clipped <= CLIPPED_STAMPS_RANGE[1]):
        return None

    grid = np.full((H, W), BACKGROUND_COLOR, dtype=int)
    for r, c, h, w in rects:
        paste(grid, np.full((h, w), FOREGROUND_COLOR, dtype=int), r, c)
    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid e619ca6e input")
