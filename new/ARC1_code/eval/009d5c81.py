import numpy as np

from arc_utils import Color, bbox, find_components, get_bg, paste
import random


_GLYPH_TO_COLOR = {
    ((0, 1, 0), (1, 1, 1), (0, 1, 0)): Color.RED,
    ((1, 0, 1), (0, 1, 0), (1, 1, 1)): Color.GREEN,
    ((1, 1, 1), (1, 0, 1), (0, 1, 0)): Color.ORANGE,
}


def _normalized_mask(g: np.ndarray, color: int) -> tuple[tuple[int, ...], ...]:
    box = bbox(g == color)
    assert box is not None, "color must be present"
    r0, c0, r1, c1 = box
    mask = (g[r0 : r1 + 1, c0 : c1 + 1] == color).astype(int)
    return tuple(tuple(int(v) for v in row) for row in mask)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the small marker glyph as a color code to recolor the larger mask, then erase the marker.

    Input structure
    - The background is the most common color.
    - There are exactly two foreground colors: one larger mask color and one 3x3 marker-glyph color.
    - The marker glyph is one of the three 3x3 color-code masks shown by the training examples.
    Rule
    - Decode the marker glyph: plus -> red, lower-bar glyph -> green, upper-bar glyph with a center gap -> orange.
    - Recolor every larger-mask pixel to the decoded color.
    - Recolor the marker glyph pixels to the background.
    Key insight
    - The marker's shape, not its position or color, supplies the replacement color for the larger mask.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    foreground_colors = [int(color) for color in np.unique(g) if int(color) != bg]
    assert len(foreground_colors) == 2, "expected exactly two foreground colors"

    symbol_candidates = []
    for color in foreground_colors:
        glyph = _normalized_mask(g, color)
        if glyph in _GLYPH_TO_COLOR:
            symbol_candidates.append((color, glyph))

    assert len(symbol_candidates) == 1, "expected one known 3x3 marker glyph"
    marker_color, marker_glyph = symbol_candidates[0]
    body_colors = [color for color in foreground_colors if color != marker_color]
    assert len(body_colors) == 1, "expected one larger mask color"
    body_color = body_colors[0]
    assert np.count_nonzero(g == body_color) > np.count_nonzero(
        g == marker_color
    ), "mask color must occupy more pixels than the marker glyph"

    out = np.full_like(g, bg)
    out[g == body_color] = _GLYPH_TO_COLOR[marker_glyph]
    return out

# --------------- generator (o5x) ---------------

# Canvas and palette.
_GEN_GRID_SHAPE = (14, 14)
_GEN_BG, _GEN_MARKER_COLOR, _GEN_BODY_COLOR = Color.BLACK, Color.BLUE, Color.TEAL

# Marker glyphs and lower-grid placement.
_GEN_MARKER_SIZE = 3
_GEN_MARKER_ROW_START, _GEN_MARKER_ROW_STOP = 8, 11
_GEN_MARKER_COL_STOP = _GEN_GRID_SHAPE[1] - _GEN_MARKER_SIZE + 1
_GEN_GLYPHS = (
    (np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int), Color.RED),
    (np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1]], dtype=int), Color.GREEN),
    (np.array([[1, 1, 1], [1, 0, 1], [0, 1, 0]], dtype=int), Color.ORANGE),
)

# Body bbox, density, and upper-grid placement.
_GEN_BODY_HEIGHTS = range(3, 9)
_GEN_BODY_WIDTHS = range(7, 11)
_GEN_BODY_TOP_ROWS = range(3)
_GEN_BODY_BOTTOM_MIN, _GEN_BODY_BOTTOM_MAX = 4, 8
_GEN_BODY_AREA_MIN, _GEN_BODY_AREA_MAX = 14, 31
_GEN_BODY_DENSITY_MIN, _GEN_BODY_DENSITY_MAX = 0.38, 0.72

# Irregular body growth and retry policy.
_GEN_SEED_MODE_COUNT = 3
_GEN_RETRY_LIMIT = 300
_GEN_DIRS8 = tuple(
    (dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)
)


def _gen_box(r0, c0, h, w):
    return r0, c0, r0 + h - 1, c0 + w - 1


def _gen_boxes_disjoint(box_a, box_b):
    ar0, ac0, ar1, ac1 = box_a
    br0, bc0, br1, bc1 = box_b
    return ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0


def _gen_path_between(start, end):
    sr, sc = start
    er, ec = end
    step_count = max(abs(er - sr), abs(ec - sc))
    r_sign = 0 if er == sr else (1 if er > sr else -1)
    c_sign = 0 if ec == sc else (1 if ec > sc else -1)
    r_steps = [r_sign] * abs(er - sr) + [0] * (step_count - abs(er - sr))
    c_steps = [c_sign] * abs(ec - sc) + [0] * (step_count - abs(ec - sc))
    random.shuffle(r_steps)
    random.shuffle(c_steps)

    r, c = sr, sc
    coords = [(r, c)]
    for dr, dc in zip(r_steps, c_steps):
        r += dr
        c += dc
        coords.append((r, c))
    return coords


def _gen_add_path(mask, start, end):
    for r, c in _gen_path_between(start, end):
        mask[r, c] = True


def _gen_add_random_diagonal(mask, h, w):
    if random.randrange(2):
        start, end = (0, 0), (h - 1, w - 1)
    else:
        start, end = (0, w - 1), (h - 1, 0)
    _gen_add_path(mask, start, end)


def _gen_nearest_mask_coord(mask, point):
    coords = np.argwhere(mask)
    pr, pc = point
    distances = np.maximum(np.abs(coords[:, 0] - pr), np.abs(coords[:, 1] - pc))
    best = coords[distances == distances.min()]
    r, c = random.choice(best).tolist()
    return int(r), int(c)


def _gen_seed_body_mask(h, w, target_count):
    mask = np.zeros((h, w), dtype=bool)
    mode = random.randrange(_GEN_SEED_MODE_COUNT)

    if mode == 0:
        _gen_add_random_diagonal(mask, h, w)
        return mask

    if mode == 1:
        left = (random.randrange(h), 0)
        right = (random.randrange(h), w - 1)
        top = (0, random.randrange(w))
        bottom = (h - 1, random.randrange(w))
        _gen_add_path(mask, left, right)
        _gen_add_path(mask, top, _gen_nearest_mask_coord(mask, top))
        _gen_add_path(mask, bottom, _gen_nearest_mask_coord(mask, bottom))
    else:
        top = (0, random.randrange(w))
        bottom = (h - 1, random.randrange(w))
        left = (random.randrange(h), 0)
        right = (random.randrange(h), w - 1)
        _gen_add_path(mask, top, bottom)
        _gen_add_path(mask, left, _gen_nearest_mask_coord(mask, left))
        _gen_add_path(mask, right, _gen_nearest_mask_coord(mask, right))

    if int(mask.sum()) <= target_count:
        return mask

    mask[:, :] = False
    _gen_add_random_diagonal(mask, h, w)
    return mask


def _gen_grow_body_mask(h, w, target_count):
    mask = _gen_seed_body_mask(h, w, target_count)
    while int(mask.sum()) < target_count:
        frontier = set()
        for r, c in np.argwhere(mask):
            for dr, dc in _GEN_DIRS8:
                nr, nc = int(r) + dr, int(c) + dc
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        r, c = random.choice(tuple(frontier))
        mask[r, c] = True
    return mask


def _gen_body_choices(marker_box):
    choices_by_size = {}
    for h in _GEN_BODY_HEIGHTS:
        for w in _GEN_BODY_WIDTHS:
            lo = max(
                _GEN_BODY_AREA_MIN, int(np.ceil(_GEN_BODY_DENSITY_MIN * h * w)), max(h, w)
            )
            hi = min(_GEN_BODY_AREA_MAX, int(np.floor(_GEN_BODY_DENSITY_MAX * h * w)))
            if lo > hi:
                continue
            placements = []
            for r0 in _GEN_BODY_TOP_ROWS:
                r1 = r0 + h - 1
                if not (_GEN_BODY_BOTTOM_MIN <= r1 <= _GEN_BODY_BOTTOM_MAX):
                    continue
                for c0 in range(_GEN_GRID_SHAPE[1] - w + 1):
                    body_box = _gen_box(r0, c0, h, w)
                    if _gen_boxes_disjoint(body_box, marker_box):
                        placements.append((r0, c0))
            if placements:
                choices_by_size[(h, w)] = (lo, hi, placements)
    return choices_by_size


def _gen_place_marker(grid):
    glyph_mask, target_color = random.choice(_GEN_GLYPHS)
    marker_r = random.randrange(_GEN_MARKER_ROW_START, _GEN_MARKER_ROW_STOP)
    marker_c = random.randrange(_GEN_MARKER_COL_STOP)
    marker_box = _gen_box(marker_r, marker_c, _GEN_MARKER_SIZE, _GEN_MARKER_SIZE)
    marker = glyph_mask * _GEN_MARKER_COLOR
    paste(grid, marker, marker_r, marker_c, transparent=_GEN_BG)
    return marker_box, target_color


def _gen_place_body(grid, marker_box):
    body_choices = _gen_body_choices(marker_box)
    if not body_choices:
        return None
    h, w = random.choice(tuple(body_choices))
    lo, hi, placements = body_choices[(h, w)]
    target_count = random.randint(lo, hi)
    body_r, body_c = random.choice(placements)
    body_mask = _gen_grow_body_mask(h, w, target_count)
    if body_mask is None:
        return None

    body = body_mask.astype(int) * _GEN_BODY_COLOR
    paste(grid, body, body_r, body_c, transparent=_GEN_BG)
    return target_count


def _gen_is_valid(grid, target_count, target_color):
    if set(np.unique(grid).tolist()) != {_GEN_BG, _GEN_MARKER_COLOR, _GEN_BODY_COLOR}:
        return False
    comps = find_components(
        grid, bg=_GEN_BG, connectivity=8, monochromatic=True, colors=[_GEN_BODY_COLOR]
    )
    if len(comps) != 1 or comps[0]["area"] != target_count:
        return False

    expected = np.full_like(grid, _GEN_BG)
    expected[grid == _GEN_BODY_COLOR] = target_color
    if not np.array_equal(main(grid), expected):
        return False
    return True


def _gen_attempt():
    grid = np.full(_GEN_GRID_SHAPE, _GEN_BG, dtype=int)

    marker_box, target_color = _gen_place_marker(grid)
    target_count = _gen_place_body(grid, marker_box)
    if target_count is None or not _gen_is_valid(grid, target_count, target_color):
        return None
    return grid


def generate_input():
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 009d5c81 input")
