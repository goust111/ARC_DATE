import random

import numpy as np


GRID_MIN = 10
GRID_MAX = 20
SHAPE_INIT_MIN = 3
SHAPE_INIT_MAX = 7
SHAPE_AREA_MIN = 4
SHAPE_AREA_MAX = 14
MIN_NEW_PIXELS = 3
MAX_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False

    seen = {tuple(map(int, coords[0]))}
    stack = [tuple(map(int, coords[0]))]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _mirror_coords(
    coords: np.ndarray,
    axis_kind: str,
    axis2: int,
) -> list[tuple[int, int]]:
    mirrored = []
    if axis_kind == "vertical":
        for r, c in coords:
            mirrored.append((int(r), int(axis2 - int(c))))
    else:
        for r, c in coords:
            mirrored.append((int(axis2 - int(r)), int(c)))
    return mirrored


def _apply_fold(
    grid_shape: tuple[int, int],
    pattern_coords: np.ndarray,
    fg: int,
    bg: int,
    axis_kind: str,
    axis2: int,
) -> np.ndarray:
    out = np.full(grid_shape, bg, dtype=int)
    for r, c in pattern_coords:
        out[int(r), int(c)] = fg
    for r, c in _mirror_coords(pattern_coords, axis_kind, axis2):
        if 0 <= r < grid_shape[0] and 0 <= c < grid_shape[1]:
            out[r, c] = fg
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Mirror the monochrome foreground pattern across the fold axis defined by a two-pixel marker pair.

    Input structure:
    - The background is the most common color.
    - Exactly one non-background color forms a single 4-connected foreground pattern.
    - A second non-background color appears in exactly two pixels outside the pattern, and those two marker pixels lie on one row or one column.

    Rule:
    - Locate the two marker pixels and use their midpoint to define the fold axis perpendicular to the segment between them.
    - Reflect every foreground-pattern pixel across that axis.
    - Return the union of the original pattern and all in-bounds reflected pixels in the original foreground color.
    - Remove the two marker pixels from the output.

    Key insight:
    - The isolated marker pair is only a geometric hint: two points determine an implicit perpendicular-bisector mirror line for the whole pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    assert len(colors) == 2

    counts = {color: int(np.sum(g == color)) for color in colors}
    marker_colors = [color for color, count in counts.items() if count == 2]
    assert len(marker_colors) == 1
    marker = marker_colors[0]

    fg_candidates = [color for color in colors if color != marker]
    assert len(fg_candidates) == 1
    fg = fg_candidates[0]

    pattern_mask = g == fg
    assert int(pattern_mask.sum()) >= 1
    assert _is_connected(pattern_mask)

    marker_coords = np.argwhere(g == marker)
    assert marker_coords.shape == (2, 2)
    (r1, c1), (r2, c2) = marker_coords
    same_row = int(r1) == int(r2)
    same_col = int(c1) == int(c2)
    assert same_row ^ same_col

    axis_kind = "vertical" if same_row else "horizontal"
    axis2 = int(c1 + c2) if same_row else int(r1 + r2)
    return _apply_fold(g.shape, np.argwhere(pattern_mask), fg, bg, axis_kind, axis2)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _random_shape_mask() -> np.ndarray | None:
    for _ in range(80):
        h0 = random.randint(SHAPE_INIT_MIN, SHAPE_INIT_MAX)
        w0 = random.randint(SHAPE_INIT_MIN, SHAPE_INIT_MAX)
        max_area = min(SHAPE_AREA_MAX, h0 * w0 - 1)
        min_area = max(SHAPE_AREA_MIN, min(max_area, max(h0, w0)))
        if min_area > max_area:
            continue

        area = random.randint(min_area, max_area)
        mask = np.zeros((h0, w0), dtype=bool)
        seed = (random.randrange(h0), random.randrange(w0))
        mask[seed] = True

        while int(mask.sum()) < area:
            frontier = set()
            for r, c in np.argwhere(mask):
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = int(r) + dr, int(c) + dc
                    if 0 <= nr < h0 and 0 <= nc < w0 and not mask[nr, nc]:
                        frontier.add((nr, nc))
            if not frontier:
                break
            nr, nc = random.choice(list(frontier))
            mask[nr, nc] = True

        cropped = _crop_mask(mask)
        ch, cw = cropped.shape
        if ch < 2 or cw < 2:
            continue
        if int(cropped.sum()) >= ch * cw:
            continue
        if not _is_connected(cropped):
            continue
        return cropped
    return None


def _axis_candidates(
    grid_shape: tuple[int, int],
    bbox: tuple[int, int, int, int],
    axis_kind: str,
    side: str,
    clipped: bool,
) -> list[int]:
    h, w = grid_shape
    r0, c0, r1, c1 = bbox
    if axis_kind == "vertical":
        lo_bound, hi_bound = 1, 2 * w - 3
        if side == "left":
            if clipped:
                lo = max(2 * c1 + 1, w + c0)
                hi = min(hi_bound, w - 1 + c1)
            else:
                lo = max(lo_bound, 2 * c1 + 1)
                hi = min(hi_bound, w - 1 + c0)
        else:
            if clipped:
                lo = max(lo_bound, c0)
                hi = min(2 * c0 - 1, c1 - 1)
            else:
                lo = max(lo_bound, c1)
                hi = min(hi_bound, 2 * c0 - 1)
    else:
        lo_bound, hi_bound = 1, 2 * h - 3
        if side == "top":
            if clipped:
                lo = max(2 * r1 + 1, h + r0)
                hi = min(hi_bound, h - 1 + r1)
            else:
                lo = max(lo_bound, 2 * r1 + 1)
                hi = min(hi_bound, h - 1 + r0)
        else:
            if clipped:
                lo = max(lo_bound, r0)
                hi = min(2 * r0 - 1, r1 - 1)
            else:
                lo = max(lo_bound, r1)
                hi = min(hi_bound, 2 * r0 - 1)

    if lo > hi:
        return []
    return list(range(lo, hi + 1))


def _place_markers(
    inp: np.ndarray,
    expected: np.ndarray,
    bbox: tuple[int, int, int, int],
    axis_kind: str,
    axis2: int,
    bg: int,
    marker: int,
) -> bool:
    h, w = inp.shape
    r0, c0, r1, c1 = bbox
    clear = (inp == bg) & (expected == bg)

    if axis_kind == "vertical":
        pairs = [(c_left, axis2 - c_left) for c_left in range(w) if c_left < axis2 - c_left < w]
        random.shuffle(pairs)
        pairs.sort(key=lambda item: item[1] - item[0], reverse=True)

        outer_rows = [r for r in range(h) if r < r0 or r > r1]
        inner_rows = [r for r in range(r0, r1 + 1)]
        random.shuffle(outer_rows)
        random.shuffle(inner_rows)
        for r in outer_rows + inner_rows:
            for c_left, c_right in pairs:
                if clear[r, c_left] and clear[r, c_right]:
                    inp[r, c_left] = marker
                    inp[r, c_right] = marker
                    return True
        return False

    pairs = [(r_top, axis2 - r_top) for r_top in range(h) if r_top < axis2 - r_top < h]
    random.shuffle(pairs)
    pairs.sort(key=lambda item: item[1] - item[0], reverse=True)

    outer_cols = [c for c in range(w) if c < c0 or c > c1]
    inner_cols = [c for c in range(c0, c1 + 1)]
    random.shuffle(outer_cols)
    random.shuffle(inner_cols)
    for c in outer_cols + inner_cols:
        for r_top, r_bottom in pairs:
            if clear[r_top, c] and clear[r_bottom, c]:
                inp[r_top, c] = marker
                inp[r_bottom, c] = marker
                return True
    return False


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(GRID_MIN, GRID_MAX)
    grid_w = random.randint(GRID_MIN, GRID_MAX)
    bg, fg, marker = random.sample(range(10), 3)
    axis_kind = random.choice(["vertical", "horizontal"])
    clipped = bool(random.getrandbits(1))

    mask = _random_shape_mask()
    if mask is None:
        return None
    shape_h, shape_w = map(int, mask.shape)

    if axis_kind == "vertical":
        side = random.choice(["left", "right"])
        if side == "left":
            c_hi = grid_w - shape_w - (0 if clipped else shape_w)
            c_lo = 0
        else:
            c_lo = 1 if clipped else shape_w
            c_hi = grid_w - shape_w
        if c_lo > c_hi:
            return None
        c0 = random.randint(c_lo, c_hi)
        r0 = random.randint(0, grid_h - shape_h)
    else:
        side = random.choice(["top", "bottom"])
        if side == "top":
            r_hi = grid_h - shape_h - (0 if clipped else shape_h)
            r_lo = 0
        else:
            r_lo = 1 if clipped else shape_h
            r_hi = grid_h - shape_h
        if r_lo > r_hi:
            return None
        r0 = random.randint(r_lo, r_hi)
        c0 = random.randint(0, grid_w - shape_w)

    bbox = (r0, c0, r0 + shape_h - 1, c0 + shape_w - 1)
    axis_values = _axis_candidates((grid_h, grid_w), bbox, axis_kind, side, clipped)
    if not axis_values:
        return None
    axis2 = random.choice(axis_values)

    inp = np.full((grid_h, grid_w), bg, dtype=int)
    pattern_coords = np.argwhere(mask) + np.array([r0, c0], dtype=int)
    for r, c in pattern_coords:
        inp[int(r), int(c)] = fg

    expected = _apply_fold(inp.shape, pattern_coords, fg, bg, axis_kind, axis2)
    new_pixels = int((expected == fg).sum() - (inp == fg).sum())
    if new_pixels < MIN_NEW_PIXELS:
        return None
    if not _place_markers(inp, expected, bbox, axis_kind, axis2, bg, marker):
        return None

    if _get_bg(inp) != bg:
        return None
    if int(np.sum(inp == marker)) != 2:
        return None
    if not _is_connected(inp == fg):
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fold-hint reflection sample")
