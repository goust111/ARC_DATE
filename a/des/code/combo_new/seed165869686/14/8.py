import numpy as np


BG = 0
YELLOW = 4
GREY = 5
SHAPE_COLORS = (1, 2, 3, 6, 7, 8, 9)
DIRECTIONS = ("top", "bottom", "left", "right")

GRID_MIN = 13
GRID_MAX = 24
SHAPE_COUNT_MIN = 2
SHAPE_COUNT_MAX = 4
LIGHT_LEN_MIN = 3
PLACEMENT_PAD = 1
PAIR_GAP_MIN = 1
PAIR_GAP_MAX = 3
GEN_RETRIES = 400
_GEN_CALL_COUNT = 0


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project solid colored shapes into a grey shadow field using a boundary light-strip direction.

    Input structure:
    - The background is black.
    - One contiguous yellow strip sits on exactly one outer edge and serves only as a light-direction marker.
    - Away from the strip, the grid contains several non-overlapping solid colored shapes.

    Rule:
    - Read the yellow edge strip as the light source direction: top -> down, bottom -> up, left -> right, right -> left.
    - For each scan line parallel to the light rays, only the first non-background, non-yellow cell hit from the light side starts a shadow; all cells behind it on that scan line are shadowed until the far boundary.
    - Paint shadowed background cells grey, but keep every original shape cell unchanged even when it lies inside another shadow.
    - Remove the yellow light strip in the output by turning it back to background.

    Key insight:
    - The yellow border segment is a direction indicator rather than an output object, and shadow occlusion is solved by scanning from the light side so only frontmost pixels cast while hidden pixels do not.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    side = _find_light_side(g)
    out = g.copy()
    out[g == YELLOW] = BG
    solid = (g != BG) & (g != YELLOW)

    h, w = g.shape
    if side == "top":
        for c in range(w):
            rows = np.flatnonzero(solid[:, c])
            if rows.size == 0:
                continue
            segment = out[int(rows[0]) :, c]
            segment[segment == BG] = GREY
    elif side == "bottom":
        for c in range(w):
            rows = np.flatnonzero(solid[:, c])
            if rows.size == 0:
                continue
            segment = out[: int(rows[-1]) + 1, c]
            segment[segment == BG] = GREY
    elif side == "left":
        for r in range(h):
            cols = np.flatnonzero(solid[r, :])
            if cols.size == 0:
                continue
            segment = out[r, int(cols[0]) :]
            segment[segment == BG] = GREY
    else:
        for r in range(h):
            cols = np.flatnonzero(solid[r, :])
            if cols.size == 0:
                continue
            segment = out[r, : int(cols[-1]) + 1]
            segment[segment == BG] = GREY

    return out


def _find_light_side(grid: np.ndarray) -> str:
    h, w = grid.shape
    total_yellow = int(np.count_nonzero(grid == YELLOW))
    assert total_yellow >= LIGHT_LEN_MIN, "expected a boundary light strip"

    candidates = []
    edge_runs = {
        "top": np.flatnonzero(grid[0, :] == YELLOW),
        "bottom": np.flatnonzero(grid[h - 1, :] == YELLOW),
        "left": np.flatnonzero(grid[:, 0] == YELLOW),
        "right": np.flatnonzero(grid[:, w - 1] == YELLOW),
    }
    for side, idxs in edge_runs.items():
        if idxs.size == 0:
            continue
        if int(idxs[-1]) - int(idxs[0]) + 1 != int(idxs.size):
            continue
        if int(idxs.size) != total_yellow:
            continue
        candidates.append(side)

    assert len(candidates) == 1, "light strip must be a unique contiguous boundary segment"
    return candidates[0]


def _trim(mask: np.ndarray) -> np.ndarray:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    return mask[rows][:, cols]


def _random_transform(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    transformed = np.rot90(mask, int(rng.integers(0, 4)))
    if float(rng.random()) < 0.5:
        transformed = np.fliplr(transformed)
    return _trim(transformed)


def _rectangle_shape(h: int, w: int) -> np.ndarray:
    return np.ones((h, w), dtype=bool)


def _l_shape(rng: np.random.Generator) -> np.ndarray:
    h = int(rng.integers(3, 7))
    w = int(rng.integers(3, 7))
    arm_w = int(rng.integers(1, w))
    arm_h = int(rng.integers(1, h))
    if arm_w == w and arm_h == h:
        if w >= h:
            arm_w -= 1
        else:
            arm_h -= 1
    mask = np.zeros((h, w), dtype=bool)
    mask[:, :arm_w] = True
    mask[h - arm_h :, :] = True
    return mask


def _t_shape(rng: np.random.Generator) -> np.ndarray:
    h = int(rng.integers(3, 7))
    w = int(rng.integers(3, 8))
    top_h = int(rng.integers(1, min(3, h) + 1))
    stem_w = int(rng.integers(1, min(3, w) + 1))
    start = int(rng.integers(0, w - stem_w + 1))
    mask = np.zeros((h, w), dtype=bool)
    mask[:top_h, :] = True
    mask[:, start : start + stem_w] = True
    return mask


def _plus_shape(rng: np.random.Generator) -> np.ndarray:
    h = int(rng.integers(3, 8))
    w = int(rng.integers(3, 8))
    bar_h = int(rng.integers(1, min(3, h) + 1))
    bar_w = int(rng.integers(1, min(3, w) + 1))
    row0 = int(rng.integers(0, h - bar_h + 1))
    col0 = int(rng.integers(0, w - bar_w + 1))
    mask = np.zeros((h, w), dtype=bool)
    mask[row0 : row0 + bar_h, :] = True
    mask[:, col0 : col0 + bar_w] = True
    return mask


def _cut_corner_shape(rng: np.random.Generator) -> np.ndarray:
    h = int(rng.integers(3, 7))
    w = int(rng.integers(3, 8))
    mask = np.ones((h, w), dtype=bool)
    max_cut = int(rng.integers(1, w))
    cut_left = bool(rng.integers(0, 2))
    for r in range(h):
        cut = min(int(round(max_cut * r / max(1, h - 1))), w - 1)
        if cut_left:
            mask[r, :cut] = False
        else:
            mask[r, w - cut :] = False
    return mask


def _random_shape(rng: np.random.Generator) -> np.ndarray:
    style = str(rng.choice(("rect", "l", "t", "plus", "cut")))
    if style == "rect":
        h = int(rng.integers(2, 6))
        w = int(rng.integers(2, 7))
        mask = _rectangle_shape(h, w)
    elif style == "l":
        mask = _l_shape(rng)
    elif style == "t":
        mask = _t_shape(rng)
    elif style == "plus":
        mask = _plus_shape(rng)
    else:
        mask = _cut_corner_shape(rng)
    return _random_transform(mask, rng)


def _can_place(reserved: np.ndarray, top: int, left: int, mask: np.ndarray) -> bool:
    h, w = reserved.shape
    mh, mw = mask.shape
    if top < 1 or left < 1 or top + mh > h - 1 or left + mw > w - 1:
        return False
    return not bool(reserved[top : top + mh, left : left + mw].any())


def _mark_reserved(reserved: np.ndarray, top: int, left: int, mask: np.ndarray) -> None:
    mh, mw = mask.shape
    r0 = max(0, top - PLACEMENT_PAD)
    r1 = min(reserved.shape[0], top + mh + PLACEMENT_PAD)
    c0 = max(0, left - PLACEMENT_PAD)
    c1 = min(reserved.shape[1], left + mw + PLACEMENT_PAD)
    reserved[r0:r1, c0:c1] = True


def _paint_shape(grid: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    patch = grid[top : top + mask.shape[0], left : left + mask.shape[1]]
    patch[mask] = int(color)


def _add_light_strip(grid: np.ndarray, direction: str, rng: np.random.Generator) -> None:
    h, w = grid.shape
    if direction in ("top", "bottom"):
        length = int(rng.integers(LIGHT_LEN_MIN, w))
        length = min(length, w - 2)
        start = int(rng.integers(1, w - length))
        row = 0 if direction == "top" else h - 1
        grid[row, start : start + length] = YELLOW
    else:
        length = int(rng.integers(LIGHT_LEN_MIN, h))
        length = min(length, h - 2)
        start = int(rng.integers(1, h - length))
        col = 0 if direction == "left" else w - 1
        grid[start : start + length, col] = YELLOW


def _place_occlusion_pair(
    grid: np.ndarray,
    reserved: np.ndarray,
    direction: str,
    colors: tuple[int, int],
    rng: np.random.Generator,
) -> bool:
    h, w = grid.shape
    color_a, color_b = map(int, colors)

    for _ in range(120):
        mask_a = _random_shape(rng)
        mask_b = _random_shape(rng)
        ha, wa = mask_a.shape
        hb, wb = mask_b.shape

        if direction in ("top", "bottom"):
            if min(wa, wb) < 2:
                continue
            overlap = int(rng.integers(1, min(wa, wb)))
            if overlap >= min(wa, wb):
                continue
            total_span = wa + wb - overlap
            gap = int(rng.integers(PAIR_GAP_MIN, PAIR_GAP_MAX + 1))
            total_height = ha + hb + gap
            if total_span > w - 2 or total_height > h - 2:
                continue

            base_col = int(rng.integers(1, w - total_span))
            if bool(rng.integers(0, 2)):
                left_a = base_col
                left_b = base_col + wa - overlap
            else:
                left_b = base_col
                left_a = base_col + wb - overlap

            if direction == "top":
                top_a = int(rng.integers(1, h - total_height))
                top_b = top_a + ha + gap
            else:
                top_b = int(rng.integers(1, h - total_height))
                top_a = top_b + hb + gap
        else:
            if min(ha, hb) < 2:
                continue
            overlap = int(rng.integers(1, min(ha, hb)))
            if overlap >= min(ha, hb):
                continue
            total_span = ha + hb - overlap
            gap = int(rng.integers(PAIR_GAP_MIN, PAIR_GAP_MAX + 1))
            total_width = wa + wb + gap
            if total_span > h - 2 or total_width > w - 2:
                continue

            base_row = int(rng.integers(1, h - total_span))
            if bool(rng.integers(0, 2)):
                top_a = base_row
                top_b = base_row + ha - overlap
            else:
                top_b = base_row
                top_a = base_row + hb - overlap

            if direction == "left":
                left_a = int(rng.integers(1, w - total_width))
                left_b = left_a + wa + gap
            else:
                left_b = int(rng.integers(1, w - total_width))
                left_a = left_b + wb + gap

        if not _can_place(reserved, top_a, left_a, mask_a):
            continue
        if not _can_place(reserved, top_b, left_b, mask_b):
            continue

        _paint_shape(grid, top_a, left_a, mask_a, color_a)
        _mark_reserved(reserved, top_a, left_a, mask_a)
        _paint_shape(grid, top_b, left_b, mask_b, color_b)
        _mark_reserved(reserved, top_b, left_b, mask_b)
        return True

    return False


def _place_extra_shape(
    grid: np.ndarray,
    reserved: np.ndarray,
    color: int,
    rng: np.random.Generator,
) -> bool:
    h, w = grid.shape
    for _ in range(120):
        mask = _random_shape(rng)
        mh, mw = mask.shape
        if mh >= h - 2 or mw >= w - 2:
            continue
        top = int(rng.integers(1, h - mh))
        left = int(rng.integers(1, w - mw))
        if not _can_place(reserved, top, left, mask):
            continue
        _paint_shape(grid, top, left, mask, int(color))
        _mark_reserved(reserved, top, left, mask)
        return True
    return False


def _gen_attempt(rng: np.random.Generator, direction: str) -> np.ndarray | None:
    h = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    w = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    grid = np.full((h, w), BG, dtype=int)
    _add_light_strip(grid, direction, rng)

    reserved = np.zeros((h, w), dtype=bool)
    shape_count = int(rng.integers(SHAPE_COUNT_MIN, SHAPE_COUNT_MAX + 1))
    colors = tuple(int(x) for x in rng.choice(SHAPE_COLORS, size=shape_count, replace=False))

    if not _place_occlusion_pair(grid, reserved, direction, colors[:2], rng):
        return None

    for color in colors[2:]:
        if not _place_extra_shape(grid, reserved, int(color), rng):
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    if int(np.count_nonzero(out == GREY)) == 0:
        return None
    if int(np.count_nonzero(out == YELLOW)) != 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    direction = DIRECTIONS[_GEN_CALL_COUNT % len(DIRECTIONS)]
    _GEN_CALL_COUNT += 1
    rng = np.random.default_rng()

    for _ in range(GEN_RETRIES):
        result = _gen_attempt(rng, direction)
        if result is not None:
            return result
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
