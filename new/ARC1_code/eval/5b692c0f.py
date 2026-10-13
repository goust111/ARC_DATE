from __future__ import annotations

from collections import Counter

import numpy as np

from arc_utils import Color, bbox_coords, components_mask, get_bg, paste, random_free_location
import random


def _contiguous(values: list[int]) -> bool:
    return values == list(range(values[0], values[-1] + 1))


def _mirror_axis(g: np.ndarray, coords: np.ndarray) -> tuple[str, int, list[tuple[int, int]]]:
    yellow = [(int(r), int(c)) for r, c in coords if int(g[r, c]) == Color.YELLOW]
    assert yellow, "expected a yellow mirror marker"

    candidates: list[tuple[str, int, int]] = []
    row_counts = Counter(r for r, _ in yellow)
    col_counts = Counter(c for _, c in yellow)

    for row, count in row_counts.items():
        if count < 2:
            continue
        cols = sorted(c for r, c in yellow if r == row)
        if _contiguous(cols):
            candidates.append(("h", int(row), len(cols)))

    for col, count in col_counts.items():
        if count < 2:
            continue
        rows = sorted(r for r, c in yellow if c == col)
        if _contiguous(rows):
            candidates.append(("v", int(col), len(rows)))

    assert candidates, "expected a contiguous yellow mirror axis"
    longest = max(length for _, _, length in candidates)
    best = [candidate for candidate in candidates if candidate[2] == longest]
    assert len(best) == 1, "mirror axis must be uniquely determined"
    orient, axis, _ = best[0]
    return orient, axis, yellow


def _single_side(signs: set[int]) -> int:
    assert signs and 0 not in signs and len(signs) == 1, "yellow control pixels must lie on one side"
    return next(iter(signs))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each marked object by mirroring its trusted side across the yellow axis.

    Input structure:
    - The background is black.
    - Each transformed object has one non-yellow foreground color plus yellow
      pixels.
    - Its yellow pixels contain one contiguous horizontal or vertical mirror
      axis and at least one off-axis yellow control pixel.

    Rule:
    - For every object with a yellow axis, use the side containing the off-axis
      yellow control pixel as the source side.
    - Reflect the whole rectangular source half of the object's bounding box
      across the axis, copying both foreground and background pixels onto the
      opposite side.
    - Leave the source side, the axis, and unrelated pixels unchanged unless
      they fall inside a reflected target half.

    Key insight:
    - The extra yellow pixel is not another object detail; it marks which half
      of the pattern is correct and should overwrite the other half by symmetry.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = get_bg(g)
    assert bg == Color.BLACK, "expected black background"

    out = g.copy()
    processed = 0
    for coords in components_mask(g != bg, connectivity=4):
        if not np.any(g[coords[:, 0], coords[:, 1]] == Color.YELLOW):
            continue

        colors = set(map(int, g[coords[:, 0], coords[:, 1]]))
        object_colors = colors - {int(Color.YELLOW)}
        assert len(object_colors) == 1, "each marked object must have one non-yellow color"

        orient, axis, yellow = _mirror_axis(g, coords)
        r0, c0, r1, c1 = bbox_coords(coords)
        processed += 1

        if orient == "h":
            source_side = _single_side({-1 if r < axis else 1 for r, _ in yellow if r != axis})
            rows = range(r0, axis) if source_side < 0 else range(axis + 1, r1 + 1)
            for src_r in rows:
                dst_r = 2 * axis - src_r
                assert 0 <= dst_r < g.shape[0], "reflected row must stay in bounds"
                out[dst_r, c0 : c1 + 1] = g[src_r, c0 : c1 + 1]
        else:
            source_side = _single_side({-1 if c < axis else 1 for _, c in yellow if c != axis})
            cols = range(c0, axis) if source_side < 0 else range(axis + 1, c1 + 1)
            for src_c in cols:
                dst_c = 2 * axis - src_c
                assert 0 <= dst_c < g.shape[1], "reflected column must stay in bounds"
                out[r0 : r1 + 1, dst_c] = g[r0 : r1 + 1, src_c]

    assert processed > 0, "expected at least one yellow-axis object"
    return out

# --------------- generator (o5x) ---------------

# Canvas and object counts.
OBJECT_COUNT_RANGE = (1, 4)
GRID_MIN_SIDE_BY_OBJECTS = {1: 14, 2: 14, 3: 18, 4: 22}
GRID_SIDE_MAX = 28

# Palette, orientation choices, and object geometry.
OBJECT_COLORS = [Color.BLUE, Color.RED, Color.GREEN, Color.TEAL]
BINARY_CHOICES = [False, True]
H_SOURCE_DEPTH_RANGE = (2, 4)
H_WIDTH_RANGE = (6, 10)
V_SOURCE_DEPTH_RANGE = (3, 5)
V_HEIGHT_RANGE = (3, 9)
V_TARGET_DEPTH_CAP = 4
TARGET_DEPTH_REDUCTION_RANGE = (0, 1)
CONTROL_OFFSET = 2

# Shape fill, yellow-axis span, and target damage rates.
AXIS_SPAN_MIN_LEN = 2
AXIS_SPAN_MIN_FRAC = 0.60
SOURCE_FILL_RANGE = (0.65, 0.80)
DAMAGE_RATE_RANGE = (0.30, 0.45)

# Placement and retry limits.
PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 2
GENERATION_RETRIES = 300


def _gen_axis_span(length):
    min_len = max(AXIS_SPAN_MIN_LEN, int(np.ceil(AXIS_SPAN_MIN_FRAC * length)))
    return random.randint(min_len, length)


def _gen_span_including(length, axis_len, pos):
    lo = max(0, pos - axis_len + 1)
    hi = min(pos, length - axis_len)
    start = random.randint(lo, hi)
    return start, start + axis_len


def _gen_add_path(mask, start, end):
    r, c = start
    er, ec = end
    mask[r, c] = True
    while (r, c) != (er, ec):
        steps = []
        if r < er:
            steps.append((r + 1, c))
        elif r > er:
            steps.append((r - 1, c))
        if c < ec:
            steps.append((r, c + 1))
        elif c > ec:
            steps.append((r, c - 1))
        r, c = random.choice(steps)
        mask[r, c] = True


def _frontier_coords(mask):
    """Return empty cells 4-adjacent to the mask in row-major order."""
    adjacent = np.zeros_like(mask, dtype=bool)
    adjacent[1:, :] |= mask[:-1, :]
    adjacent[:-1, :] |= mask[1:, :]
    adjacent[:, 1:] |= mask[:, :-1]
    adjacent[:, :-1] |= mask[:, 1:]
    return np.argwhere(adjacent & ~mask)


def _gen_source_from_points(h, w, fg, points):
    mask = np.zeros((h, w), dtype=bool)
    cur = points[0]
    mask[cur] = True
    for pt in points[1:]:
        _gen_add_path(mask, cur, pt)
        cur = pt

    area = h * w
    low = int(np.ceil(SOURCE_FILL_RANGE[0] * area))
    high = int(np.floor(SOURCE_FILL_RANGE[1] * area))
    filled = int(mask.sum())
    target = random.randint(max(low, filled), max(high, filled))
    while int(mask.sum()) < target:
        r, c = map(int, random.choice(_frontier_coords(mask)))
        mask[r, c] = True

    source = np.zeros((h, w), dtype=int)
    source[mask] = fg
    return source


def _rect_coords(rows, cols):
    return [(r, c) for r in rows for c in cols]


def _gen_target_depth(source_depth, *, cap=None):
    target_depth = source_depth - random.randint(*TARGET_DEPTH_REDUCTION_RANGE)
    if cap is not None and source_depth > cap:
        return cap
    return target_depth


def _gen_damage(inp, out, full_coords, existing_coords, fg):
    full = list(full_coords)
    existing = set(existing_coords)

    for r, c in full:
        if (r, c) not in existing:
            inp[r, c] = Color.BLACK

    for r, c in existing:
        if int(out[r, c]) == Color.YELLOW:
            inp[r, c] = Color.BLACK

    area = len(full)
    rate = random.uniform(*DAMAGE_RATE_RANGE)
    wanted = int(round(rate * area))
    current = sum(int(inp[r, c]) != int(out[r, c]) for r, c in full)

    candidates = [(r, c) for r, c in existing if int(inp[r, c]) == int(out[r, c])]
    random.shuffle(candidates)
    for r, c in candidates[: max(0, wanted - current)]:
        inp[r, c] = fg if int(out[r, c]) == Color.BLACK else Color.BLACK


def _gen_horizontal_object(fg):
    s = random.randint(*H_SOURCE_DEPTH_RANGE)
    t = _gen_target_depth(s)
    w = random.randint(*H_WIDTH_RANGE)
    ctrl_c = random.randrange(w)
    source = _gen_source_from_points(
        s,
        w,
        fg,
        [(0, ctrl_c), (s - 1, ctrl_c), (s - 1, 0), (s - 1, w - 1)],
    )

    axis_len = _gen_axis_span(w)
    a0, a1 = _gen_span_including(w, axis_len, ctrl_c)

    out = np.zeros((2 * s + 1, w), dtype=int)
    out[:s, :] = source
    out[s, a0:a1] = Color.YELLOW
    out[s - CONTROL_OFFSET, ctrl_c] = Color.YELLOW
    out[s + 1 :, :] = out[:s, :][::-1]

    inp = out.copy()
    full = _rect_coords(range(s + 1, 2 * s + 1), range(w))
    existing = _rect_coords(range(s + 1, s + 1 + t), range(w))
    _gen_damage(inp, out, full, existing, fg)
    inp[s + 1 : s + 1 + t, ctrl_c] = fg

    if random.choice(BINARY_CHOICES):
        inp = np.flipud(inp).copy()
        out = np.flipud(out).copy()
    return inp, out


def _gen_vertical_object(fg):
    s = random.randint(*V_SOURCE_DEPTH_RANGE)
    t = _gen_target_depth(s, cap=V_TARGET_DEPTH_CAP)
    h = random.randint(*V_HEIGHT_RANGE)
    ctrl_r = random.randrange(h)
    source = _gen_source_from_points(
        h,
        s,
        fg,
        [(ctrl_r, 0), (ctrl_r, s - 1), (0, s - 1), (h - 1, s - 1)],
    )

    axis_len = _gen_axis_span(h)
    a0, a1 = _gen_span_including(h, axis_len, ctrl_r)

    out = np.zeros((h, 2 * s + 1), dtype=int)
    out[:, :s] = source
    out[a0:a1, s] = Color.YELLOW
    out[ctrl_r, s - CONTROL_OFFSET] = Color.YELLOW
    out[:, s + 1 :] = out[:, :s][:, ::-1]

    inp = out.copy()
    full = _rect_coords(range(h), range(s + 1, 2 * s + 1))
    existing = _rect_coords(range(h), range(s + 1, s + 1 + t))
    _gen_damage(inp, out, full, existing, fg)
    inp[ctrl_r, s + 1 : s + 1 + t] = fg

    if random.choice(BINARY_CHOICES):
        inp = np.fliplr(inp).copy()
        out = np.fliplr(out).copy()
    return inp, out


def _gen_object(fg):
    if random.choice(BINARY_CHOICES):
        return _gen_horizontal_object(fg)
    return _gen_vertical_object(fg)


def _place_object(grid, expected, reserved, inp_patch, out_patch):
    try:
        r, c = random_free_location(reserved, out_patch, transparent=Color.BLACK, border=PLACEMENT_BORDER, padding=PLACEMENT_PADDING)
    except ValueError:
        return False
    paste(grid, inp_patch, r, c, transparent=Color.BLACK)
    paste(expected, out_patch, r, c, transparent=Color.BLACK)
    paste(reserved, out_patch, r, c, transparent=Color.BLACK)
    return True


def _gen_attempt():
    n_objects = random.randint(*OBJECT_COUNT_RANGE)
    min_side = GRID_MIN_SIDE_BY_OBJECTS[n_objects]
    rows = random.randint(min_side, GRID_SIDE_MAX)
    cols = random.randint(min_side, GRID_SIDE_MAX)
    grid = np.zeros((rows, cols), dtype=int)
    expected = np.zeros_like(grid)
    reserved = np.zeros_like(grid)

    colors = random.sample(OBJECT_COLORS, n_objects)
    for fg in colors:
        inp_patch, out_patch = _gen_object(fg)
        if not _place_object(grid, expected, reserved, inp_patch, out_patch):
            return None
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input():
    for _ in range(GENERATION_RETRIES):
        out = _gen_attempt()
        if out is not None:
            return out
    raise RuntimeError("failed to generate valid input")
