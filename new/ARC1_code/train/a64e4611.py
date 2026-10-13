from __future__ import annotations

import numpy as np

from arc_utils import Color, components_mask, get_bg


def _clear_neighborhood_mask(g: np.ndarray, bg: int) -> np.ndarray:
    height, width = map(int, g.shape)
    mask = np.zeros((height, width), dtype=bool)
    for r in range(height):
        for c in range(width):
            r0 = max(0, r - 1)
            r1 = min(height, r + 2)
            c0 = max(0, c - 1)
            c1 = min(width, c + 2)
            mask[r, c] = np.all(g[r0:r1, c0:c1] == bg)
    return mask


def _maximal_true_rectangles(mask: np.ndarray) -> list[tuple[int, int, int, int, int, int, int]]:
    height, width = map(int, mask.shape)
    rects: set[tuple[int, int, int, int, int, int, int]] = set()
    for r0 in range(height):
        valid_cols = np.ones(width, dtype=bool)
        for r1 in range(r0, height):
            valid_cols &= mask[r1]
            c = 0
            while c < width:
                if not valid_cols[c]:
                    c += 1
                    continue
                c0 = c
                while c < width and valid_cols[c]:
                    c += 1
                c1 = c - 1

                can_extend_up = r0 > 0 and np.all(mask[r0 - 1, c0 : c1 + 1])
                can_extend_down = r1 + 1 < height and np.all(mask[r1 + 1, c0 : c1 + 1])
                can_extend_left = c0 > 0 and np.all(mask[r0 : r1 + 1, c0 - 1])
                can_extend_right = c1 + 1 < width and np.all(mask[r0 : r1 + 1, c1 + 1])
                if can_extend_up or can_extend_down or can_extend_left or can_extend_right:
                    continue

                h = r1 - r0 + 1
                w = c1 - c0 + 1
                rects.add((h * w, h, w, r0, c0, r1, c1))

    return sorted(
        rects,
        key=lambda rect: (-rect[0], -max(rect[1], rect[2]), rect[3], rect[4], rect[5], rect[6]),
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint the long hidden blank corridors green while preserving the colored clutter.

    Input structure:
    - The grid has a dominant background color and one visible clutter color.
    - Green is absent from the input and is used as the output fill color.
    - The intended hidden corridors are long all-background rectangular bars, but
      random background gaps can touch them or create small blank pockets.

    Rule:
    - First keep only background pixels whose whole in-grid 3x3 neighborhood is background.
    - Enumerate maximal rectangles inside that clear-background mask.
    - In descending rectangle area, accept rectangles whose major axis spans at least
      half a grid side and add only newly exposed 4-connected portions at least one
      third of a grid side in area.
    - Recolor the accepted background mask green.

    Key insight:
    - The object to reveal is negative space: true corridors survive a one-pixel
      clearance test and contribute long new rectangular growth, while incidental
      gaps are either short or add only tiny disconnected fragments.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert min(height, width) >= 3, "expected room for 3x3 blank neighborhoods"

    bg = get_bg(g)
    colors = set(map(int, np.unique(g)))
    assert Color.GREEN not in colors, "green is reserved for the revealed corridors"
    assert len(colors - {bg}) == 1, "expected exactly one visible clutter color"

    clear = _clear_neighborhood_mask(g, bg)
    min_major = min(height, width) // 2
    min_new = min(height, width) // 3

    reveal = np.zeros_like(clear, dtype=bool)
    for _area, h, w, r0, c0, r1, c1 in _maximal_true_rectangles(clear):
        if max(h, w) < min_major:
            continue

        rect = np.zeros_like(clear, dtype=bool)
        rect[r0 : r1 + 1, c0 : c1 + 1] = True
        new_cells = rect & ~reveal

        kept_new_cells = np.zeros_like(clear, dtype=bool)
        for coords in components_mask(new_cells, connectivity=4):
            if int(coords.shape[0]) >= min_new:
                kept_new_cells[coords[:, 0], coords[:, 1]] = True

        if np.any(kept_new_cells):
            reveal |= kept_new_cells

    assert np.any(reveal), "expected at least one long hidden corridor"
    assert np.all(g[reveal] == bg), "only background pixels may be revealed"

    out = g.copy()
    out[reveal] = Color.GREEN
    return out

# --------------- generator (o5x) ---------------

# Canvas, colors, and retry budget.
GRID_SIZE, BACKGROUND_COLOR, REVEAL_COLOR = 30, 0, Color.GREEN
CLUTTER_COLORS = np.array([1, 2, 4, 5, 6, 7, 8, 9], dtype=int)
MAX_GENERATION_ATTEMPTS = 300
# Vertical corridor geometry.
V_WIDTH_RANGE, V_PARTIAL_HEIGHT_RANGE = (4, 10), (24, 29)
V_COLUMN_PAD = 1
V_FULL_SPAN, V_TOP_SPAN, V_BOTTOM_SPAN = 0, 1, 2
V_SPAN_KINDS = np.array([V_FULL_SPAN, V_TOP_SPAN, V_BOTTOM_SPAN], dtype=int)
V_SPAN_WEIGHTS = np.array([0.55, 0.225, 0.225], dtype=float)
# Horizontal corridor bands and solver thresholds.
H_SPLIT_PROB, H_MAX_HEIGHT = 0.45, 10
H_MIN_NEW_AREA, H_MIN_MAJOR_AXIS, H_MIN_MID_GAP = 10, 15, 3
# Clutter density and margin support.
CLUTTER_PROB_RANGE, CLUTTER_NEIGHBOR_RADIUS = (0.43, 0.48), 1


def _gen_clear_mask(mask: np.ndarray) -> np.ndarray:
    height, width = map(int, mask.shape)
    padded = np.pad(mask, 1, mode="constant", constant_values=True)
    clear = np.ones_like(mask, dtype=bool)
    for dr in range(3):
        for dc in range(3):
            clear &= padded[dr : dr + height, dc : dc + width]
    return clear


def _gen_vertical_rect(rng: np.random.Generator) -> tuple[int, int, int, int]:
    width = int(rng.integers(V_WIDTH_RANGE[0], V_WIDTH_RANGE[1] + 1))
    c0 = int(rng.integers(V_COLUMN_PAD, GRID_SIZE - width))
    c1 = c0 + width - 1

    span_kind = int(rng.choice(V_SPAN_KINDS, p=V_SPAN_WEIGHTS))
    if span_kind == V_FULL_SPAN:
        return 0, c0, GRID_SIZE - 1, c1

    height = int(rng.integers(V_PARTIAL_HEIGHT_RANGE[0], V_PARTIAL_HEIGHT_RANGE[1] + 1))
    if span_kind == V_TOP_SPAN:
        return 0, c0, height - 1, c1
    return GRID_SIZE - height, c0, GRID_SIZE - 1, c1


def _gen_h_bands(
    rng: np.random.Generator,
    vr0: int,
    vr1: int,
    v_width: int,
    n_bands: int,
) -> list[tuple[int, int]]:
    v_height = vr1 - vr0 + 1
    min_gap = (H_MIN_NEW_AREA + v_width - 1) // v_width
    mid_gap = max(min_gap, H_MIN_MID_GAP)
    if n_bands == 2 and v_height >= 2 * min_gap + mid_gap + 2:
        available = v_height - 2 * min_gap - mid_gap
        h0 = int(rng.integers(1, min(H_MAX_HEIGHT, available - 1) + 1))
        h1 = int(rng.integers(1, min(H_MAX_HEIGHT, available - h0) + 1))
        extra = v_height - h0 - h1 - 2 * min_gap - mid_gap
        gaps = np.array([min_gap, mid_gap, min_gap], dtype=int)
        if extra > 0:
            gaps += rng.multinomial(extra, [1 / 3, 1 / 3, 1 / 3])
        r0 = vr0 + int(gaps[0])
        r1 = r0 + h0 - 1
        r2 = r1 + 1 + int(gaps[1])
        r3 = r2 + h1 - 1
        return [(r0, r1), (r2, r3)]

    max_h = min(H_MAX_HEIGHT, v_height - 2 * min_gap)
    if max_h < 1:
        return []
    h = int(rng.integers(1, max_h + 1))
    lo = vr0 + min_gap
    hi = vr1 - h + 1 - min_gap
    r0 = int(rng.integers(lo, hi + 1))
    return [(r0, r0 + h - 1)]


def _gen_h_columns(
    rng: np.random.Generator,
    h: int,
    vt_c0: int,
    vt_c1: int,
    vc0: int,
    vc1: int,
) -> tuple[int, int] | None:
    left_width = vc1 + 1
    right_width = GRID_SIZE - vc0
    left_arm = vc0
    right_arm = GRID_SIZE - 1 - vc1
    valid: list[str] = []
    if left_width >= H_MIN_MAJOR_AXIS and h * left_arm >= H_MIN_NEW_AREA:
        valid.append("left")
    if right_width >= H_MIN_MAJOR_AXIS and h * right_arm >= H_MIN_NEW_AREA:
        valid.append("right")
    if h * left_arm >= H_MIN_NEW_AREA and h * right_arm >= H_MIN_NEW_AREA:
        valid.append("both")
    if not valid:
        return None

    side = str(rng.choice(valid))
    if side == "left":
        return 0, vt_c1
    if side == "right":
        return vt_c0, GRID_SIZE - 1
    return 0, GRID_SIZE - 1


def _gen_add_horizontal_bars(
    rng: np.random.Generator,
    true_mask: np.ndarray,
    vertical_rect: tuple[int, int, int, int],
) -> bool:
    vt_r0, vt_c0, vt_r1, vt_c1 = vertical_rect
    vc0, vc1 = vt_c0 + 1, vt_c1 - 1
    vr0 = vt_r0 if vt_r0 == 0 else vt_r0 + 1
    vr1 = vt_r1 if vt_r1 == GRID_SIZE - 1 else vt_r1 - 1
    v_width = vc1 - vc0 + 1
    n_bands = 2 if rng.random() < H_SPLIT_PROB else 1
    h_bands = _gen_h_bands(rng, vr0, vr1, v_width, n_bands)
    if not h_bands:
        return False

    for hr0, hr1 in h_bands:
        h = hr1 - hr0 + 1
        h_cols = _gen_h_columns(rng, h, vt_c0, vt_c1, vc0, vc1)
        if h_cols is None:
            return False
        hc0, hc1 = h_cols
        true_mask[hr0 - 1 : hr1 + 2, hc0 : hc1 + 1] = True
    return True


def _gen_support_margin_clutter(
    rng: np.random.Generator,
    true_mask: np.ndarray,
    reveal: np.ndarray,
    outside: np.ndarray,
    clutter_mask: np.ndarray,
) -> bool:
    margin = true_mask & ~reveal
    for r, c in np.argwhere(margin):
        r0 = max(0, int(r) - CLUTTER_NEIGHBOR_RADIUS)
        r1 = min(GRID_SIZE, int(r) + CLUTTER_NEIGHBOR_RADIUS + 1)
        c0 = max(0, int(c) - CLUTTER_NEIGHBOR_RADIUS)
        c1 = min(GRID_SIZE, int(c) + CLUTTER_NEIGHBOR_RADIUS + 1)
        if np.any(clutter_mask[r0:r1, c0:c1]):
            continue
        local = np.argwhere(outside[r0:r1, c0:c1])
        if local.size == 0:
            return False
        rr, cc = local[int(rng.integers(0, local.shape[0]))]
        clutter_mask[r0 + int(rr), c0 + int(cc)] = True
    return True


def _gen_attempt(rng: np.random.Generator | None = None) -> np.ndarray | None:
    if rng is None:
        rng = np.random.default_rng()

    clutter_color = int(rng.choice(CLUTTER_COLORS))
    true_mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)

    vertical_rect = _gen_vertical_rect(rng)
    vt_r0, vt_c0, vt_r1, vt_c1 = vertical_rect
    true_mask[vt_r0 : vt_r1 + 1, vt_c0 : vt_c1 + 1] = True
    if not _gen_add_horizontal_bars(rng, true_mask, vertical_rect):
        return None

    reveal = _gen_clear_mask(true_mask)
    if not np.any(reveal):
        return None

    outside = ~true_mask
    density = float(rng.uniform(*CLUTTER_PROB_RANGE))
    clutter_mask = outside & (rng.random((GRID_SIZE, GRID_SIZE)) < density)
    if not _gen_support_margin_clutter(rng, true_mask, reveal, outside, clutter_mask):
        return None

    out = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    out[clutter_mask] = clutter_color
    out[reveal] = REVEAL_COLOR

    inp = out.copy()
    inp[reveal] = BACKGROUND_COLOR
    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid a64e4611 input")
