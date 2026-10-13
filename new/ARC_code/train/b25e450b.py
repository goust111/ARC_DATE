from __future__ import annotations

import numpy as np

from arc_utils import Color, bbox_coords, components_mask
import random

BBox = tuple[int, int, int, int]


def _edge_side(bbox: BBox, shape: tuple[int, int]) -> str:
    r0, c0, r1, c1 = bbox
    h, w = shape
    sides = []
    if r0 == 0:
        sides.append("top")
    if r1 == h - 1:
        sides.append("bottom")
    if c0 == 0:
        sides.append("left")
    if c1 == w - 1:
        sides.append("right")
    assert len(sides) == 1, "each black rectangle must touch exactly one grid edge"
    return sides[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each black border rectangle to the opposite border, recoloring its swept stripe orange and leaving the rectangle black at the far edge.

    Input structure
    - The grid contains black rectangular 4-connected masks, each touching exactly one outer border.
    - Grey cells form incidental pattern content; orange is the trail color used when a black mask slides.

    Rule
    - Treat all black rectangles as moving simultaneously.
    - A rectangle touching the top or bottom edge moves vertically to the opposite edge in the same columns.
    - A rectangle touching the left or right edge moves horizontally to the opposite edge in the same rows.
    - Recolor the complete swept row or column stripe to orange, then stamp every destination rectangle in black.

    Key insight
    - Edge contact gives the slide direction; the output is the set of opposite-edge black rectangles plus their orange trails.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h > 0 and w > 0

    moving = Color.BLACK
    trail = Color.ORANGE
    assert np.any(g == moving), "expected at least one black rectangle"
    assert np.any(g == trail), "expected orange trail color to be present"

    out = g.copy()
    targets: list[BBox] = []
    for coords in components_mask(g == moving, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(coords)
        rect_h = r1 - r0 + 1
        rect_w = c1 - c0 + 1
        assert int(coords.shape[0]) == rect_h * rect_w, "black mask must be a solid rectangle"

        side = _edge_side((r0, c0, r1, c1), g.shape)
        if side == "top":
            out[:, c0 : c1 + 1] = trail
            targets.append((h - rect_h, c0, h - 1, c1))
        elif side == "bottom":
            out[:, c0 : c1 + 1] = trail
            targets.append((0, c0, rect_h - 1, c1))
        elif side == "left":
            out[r0 : r1 + 1, :] = trail
            targets.append((r0, w - rect_w, r1, w - 1))
        else:
            assert side == "right"
            out[r0 : r1 + 1, :] = trail
            targets.append((r0, 0, r1, rect_w - 1))

    for r0, c0, r1, c1 in targets:
        out[r0 : r1 + 1, c0 : c1 + 1] = moving

    return out

# --------------- generator (att1) ---------------

# Canvas and palette.
MIN_H, MAX_H = 6, 18
MIN_W, MAX_W = 8, 14
BG_GREY, BG_ORANGE = 5, 7
GREY_FRAC_MIN, GREY_FRAC_MAX = 0.2, 0.6

# Rectangle and band constraints.
MIN_RECTS, MAX_RECTS = 1, 8
MIN_BAND_SIZE, MAX_BAND_SIZE = 1, 6
BAND_GAP = 1
MIN_RECT_AREA, MAX_RECT_AREA = 2, 18
MAX_RECT_SIDE = 6
SOURCE_EDGE_OFFSET = 1

# Retry limits.
TARGET_COUNT_ATTEMPTS = 300
ANY_COUNT_ATTEMPTS = 100


def _partition_bands(
    total: int, n: int, min_size: int, max_size: int, min_gap: int = 0
) -> list[tuple[int, int]] | None:
    """Place n non-overlapping inclusive bands inside [0, total - 1]."""
    if n == 0:
        return []

    needed = n * min_size + (n - 1) * min_gap
    if needed > total:
        return None

    effective_max = min(max_size, total - (n - 1) * (min_size + min_gap))
    if effective_max < min_size:
        return None

    widths = [random.randint(min_size, effective_max) for _ in range(n)]
    total_used = sum(widths) + (n - 1) * min_gap
    if total_used > total:
        excess = total_used - total
        indices = list(range(n))
        random.shuffle(indices)
        for i in indices:
            shrink = min(widths[i] - min_size, excess)
            widths[i] -= shrink
            excess -= shrink
            if excess == 0:
                break
        if excess > 0:
            return None

    remaining = total - sum(widths) - (n - 1) * min_gap
    extra_gaps = [0] * (n + 1)
    for _ in range(remaining):
        extra_gaps[random.randint(0, n)] += 1

    bands = []
    pos = 0
    for i in range(n):
        pos += extra_gaps[i]
        if i > 0:
            pos += min_gap
        bands.append((pos, pos + widths[i] - 1))
        pos += widths[i]
    return bands


def _choose_orientation_counts(n_rects: int) -> tuple[int, int] | None:
    max_vert = min(n_rects, (MAX_W - SOURCE_EDGE_OFFSET) // 2)
    max_horiz = min(n_rects, (MAX_H - SOURCE_EDGE_OFFSET) // 2)
    feasible = [(v, n_rects - v) for v in range(n_rects + 1)
                if v <= max_vert and (n_rects - v) <= max_horiz]
    if not feasible:
        return None
    return random.choice(feasible)


def _choose_canvas(n_vert: int, n_horiz: int) -> tuple[int, int] | None:
    min_h = max(MIN_H, 2 * n_horiz + 1) if n_horiz > 0 else MIN_H
    min_w = max(MIN_W, 2 * n_vert + 1) if n_vert > 0 else MIN_W
    if min_h > MAX_H or min_w > MAX_W:
        return None
    return random.randint(min_h, MAX_H), random.randint(min_w, MAX_W)


def _interior_bands(total: int, n: int) -> list[tuple[int, int]] | None:
    raw = _partition_bands(total - 2, n, MIN_BAND_SIZE, MAX_BAND_SIZE, BAND_GAP)
    if raw is None:
        return None
    return [(a + SOURCE_EDGE_OFFSET, b + SOURCE_EDGE_OFFSET) for a, b in raw]


def _rect_span_for_band(band_size: int, travel: int) -> int | None:
    max_len = min(MAX_RECT_SIDE, travel // 2 - 1, MAX_RECT_AREA // band_size)
    min_len = max(MIN_BAND_SIZE, -(-MIN_RECT_AREA // band_size))
    if min_len > max_len:
        return None
    return random.randint(min_len, max_len)


def _build_rects(H: int, W: int, n_vert: int, n_horiz: int) -> list[tuple[int, int, int, int, str]] | None:
    """Return destination rectangles plus their original source edge."""
    vert_edges = [random.choice(["top", "bottom"]) for _ in range(n_vert)]
    horiz_edges = [random.choice(["left", "right"]) for _ in range(n_horiz)]

    col_bands = _interior_bands(W, n_vert)
    if col_bands is None:
        return None
    row_bands = _interior_bands(H, n_horiz)
    if row_bands is None:
        return None

    rects = []
    for i, (c0, c1) in enumerate(col_bands):
        rect_w = c1 - c0 + 1
        rect_h = _rect_span_for_band(rect_w, H)
        if rect_h is None:
            return None
        se = vert_edges[i]
        if se == "top":
            dr0, dr1 = H - rect_h, H - 1
        else:
            dr0, dr1 = 0, rect_h - 1
        rects.append((dr0, c0, dr1, c1, se))

    for i, (r0, r1) in enumerate(row_bands):
        rect_h = r1 - r0 + 1
        rect_w = _rect_span_for_band(rect_h, W)
        if rect_w is None:
            return None
        se = horiz_edges[i]
        if se == "left":
            dc0, dc1 = W - rect_w, W - 1
        else:
            dc0, dc1 = 0, rect_w - 1
        rects.append((r0, dc0, r1, dc1, se))

    return rects


def _random_background(shape: tuple[int, int], grey_frac: float) -> np.ndarray:
    return np.where(np.random.random(shape) < grey_frac, BG_GREY, BG_ORANGE).astype(int)


def _render_output(H: int, W: int, rects: list[tuple[int, int, int, int, str]], grey_frac: float) -> np.ndarray:
    output = _random_background((H, W), grey_frac)

    for (dr0, dc0, dr1, dc1, se) in rects:
        if se in ("top", "bottom"):
            output[:, dc0:dc1 + 1] = BG_ORANGE
        else:
            output[dr0:dr1 + 1, :] = BG_ORANGE

    for (dr0, dc0, dr1, dc1, _) in rects:
        output[dr0:dr1 + 1, dc0:dc1 + 1] = 0
    return output


def _invert_output_to_input(
    output: np.ndarray, rects: list[tuple[int, int, int, int, str]], grey_frac: float
) -> np.ndarray:
    H, W = output.shape
    inp = output.copy()

    for (dr0, dc0, dr1, dc1, se) in rects:
        if se in ("top", "bottom"):
            inp[:, dc0:dc1 + 1] = _random_background((H, dc1 - dc0 + 1), grey_frac)
        else:
            inp[dr0:dr1 + 1, :] = _random_background((dr1 - dr0 + 1, W), grey_frac)

    for (dr0, dc0, dr1, dc1, se) in rects:
        rh, rw = dr1 - dr0 + 1, dc1 - dc0 + 1
        if se == "top":
            inp[0:rh, dc0:dc1 + 1] = 0
        elif se == "bottom":
            inp[H - rh:H, dc0:dc1 + 1] = 0
        elif se == "left":
            inp[dr0:dr1 + 1, 0:rw] = 0
        else:
            inp[dr0:dr1 + 1, W - rw:W] = 0

    return inp


def _is_valid_pair(inp: np.ndarray, output: np.ndarray) -> bool:
    try:
        return bool(np.array_equal(main(inp), output))
    except Exception:
        return False


def _gen_attempt_with_count(n_rects: int) -> np.ndarray | None:
    """Generate a valid input with exactly n_rects black rectangles."""
    counts = _choose_orientation_counts(n_rects)
    if counts is None:
        return None
    n_vert, n_horiz = counts

    canvas = _choose_canvas(n_vert, n_horiz)
    if canvas is None:
        return None
    H, W = canvas

    rects = _build_rects(H, W, n_vert, n_horiz)
    if rects is None:
        return None

    grey_frac = random.uniform(GREY_FRAC_MIN, GREY_FRAC_MAX)
    output = _render_output(H, W, rects, grey_frac)
    inp = _invert_output_to_input(output, rects, grey_frac)
    return inp if _is_valid_pair(inp, output) else None


def generate_input() -> np.ndarray:
    target_count = random.randint(MIN_RECTS, MAX_RECTS)
    for _ in range(TARGET_COUNT_ATTEMPTS):
        result = _gen_attempt_with_count(target_count)
        if result is not None:
            return result
    for _ in range(ANY_COUNT_ATTEMPTS):
        result = _gen_attempt_with_count(random.randint(MIN_RECTS, MAX_RECTS))
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {TARGET_COUNT_ATTEMPTS + ANY_COUNT_ATTEMPTS} attempts")
