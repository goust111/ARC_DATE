import numpy as np

from arc_utils import bbox, get_bg, is_contiguous, paste, random_free_location


_MARKER_SHAPE = {(0, 2), (1, 2), (2, 0), (2, 2), (3, 1)}
_MARKER_TARGET_REL = (2, 0)


def main(grid: np.ndarray) -> np.ndarray:
    """Move the lower sprite and its pink guide into the upper panel, aligning the sprite to the guide's lower-left branch.

    Input structure
    - The grid has one dominant background color and one full-width separator row.
    - Below the separator are exactly two foreground roles: a five-pixel guide marker and one monochrome sprite.
    - The guide marker has normalized shape {(0,2), (1,2), (2,0), (2,2), (3,1)} and starts immediately below the separator.

    Rule
    - Clear the canvas to the background, preserving the separator row.
    - Translate the guide marker upward so its top row becomes row 0.
    - Move the sprite, preserving its color and shape, so the sprite bbox bottom-right aligns to the translated guide's lower-left branch.

    Key insight
    - The marker supplies the destination anchor: its lower-left branch cell is where the moved sprite's bbox bottom-right must land.
    """
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    bg = int(get_bg(g))

    divider_rows = [
        r
        for r in range(height)
        if len(set(map(int, g[r]))) == 1 and int(g[r, 0]) != bg
    ]
    assert len(divider_rows) == 1, "expected one full-width non-background separator row"
    sep_r = divider_rows[0]
    sep_color = int(g[sep_r, 0])

    role_colors = set(map(int, g.ravel())) - {bg, sep_color}
    marker_candidates = []
    for color in role_colors:
        coords = np.argwhere(g == color)
        color_bbox = bbox(g == color)
        assert color_bbox is not None, "role colors must have pixels"
        cr0, cc0, _cr1, _cc1 = color_bbox
        color_rel = {(int(r - cr0), int(c - cc0)) for r, c in coords}
        if cr0 == sep_r + 1 and color_rel == _MARKER_SHAPE:
            marker_candidates.append(int(color))
    assert len(marker_candidates) == 1, "expected one five-pixel guide marker below separator"
    marker_color = marker_candidates[0]

    sprite_colors = role_colors - {marker_color}
    assert len(sprite_colors) == 1, "expected one movable sprite color"
    sprite_color = next(iter(sprite_colors))

    marker_mask = g == marker_color
    marker_bbox = bbox(marker_mask)
    assert marker_bbox is not None, "expected marker pixels"
    mr0, mc0, _mr1, _mc1 = marker_bbox
    assert mr0 == sep_r + 1, "marker must start immediately below separator"
    marker_coords = np.argwhere(marker_mask)
    marker_rel = {(int(r - mr0), int(c - mc0)) for r, c in marker_coords}
    assert marker_rel == _MARKER_SHAPE, "unexpected guide marker shape"

    out = np.full_like(g, bg)
    out[sep_r, :] = sep_color

    shift_up = sep_r + 1
    for r, c in marker_coords:
        rr = int(r) - shift_up
        cc = int(c)
        assert 0 <= rr < sep_r and 0 <= cc < width, "translated marker must fit in upper panel"
        out[rr, cc] = marker_color

    sprite_mask = g == sprite_color
    assert is_contiguous(sprite_mask, background=False, connectivity=8), "movable sprite must be 8-connected"
    sprite_bbox = bbox(sprite_mask)
    assert sprite_bbox is not None, "expected sprite pixels"
    sr0, sc0, sr1, sc1 = sprite_bbox
    sprite_patch = np.full((sr1 - sr0 + 1, sc1 - sc0 + 1), bg, dtype=int)
    sprite_patch[sprite_mask[sr0 : sr1 + 1, sc0 : sc1 + 1]] = sprite_color

    target_r = mr0 - shift_up + _MARKER_TARGET_REL[0]
    target_c = mc0 + _MARKER_TARGET_REL[1]
    paste_r = target_r - sprite_patch.shape[0] + 1
    paste_c = target_c - sprite_patch.shape[1] + 1
    assert 0 <= paste_r and 0 <= paste_c, "sprite placement must fit upper panel"
    assert paste_r + sprite_patch.shape[0] <= sep_r, "sprite placement must stay above separator"
    assert paste_c + sprite_patch.shape[1] <= width, "sprite placement must fit grid width"

    paste(out, sprite_patch, paste_r, paste_c, transparent=bg)
    return out

# --------------- generator (o5x) ---------------

# Canvas and fixed colors.
_GEN_BG, _GEN_SEP, _GEN_MARKER = 7, 4, 6
_GEN_SIZE, _GEN_SEP_ROW, _GEN_LOWER_ROW0 = 11, 5, 6
# Marker geometry and allowed columns.
_GEN_MARKER_TOP_ROW = 0
_GEN_MARKER_COL_MIN, _GEN_MARKER_COL_STOP = 1, 9
_GEN_MARKER_SHAPE = ((0, 2), (1, 2), (2, 0), (2, 2), (3, 1))
_GEN_MARKER_ANCHOR_REL = (2, 0)
# Sprite shape and color sampling.
_GEN_SPRITE_COLORS = (1, 2, 3, 5, 8, 9)
_GEN_SPRITE_HEIGHTS = (2, 3)
_GEN_SPRITE_WIDTH, _GEN_SPRITE_MAX_PIXELS_STOP = 2, 5
_GEN_SPRITE_COORD_RETRIES = 100
# Retry budget.
_GEN_ATTEMPTS = 300


def _gen_blank_grid() -> np.ndarray:
    grid = np.full((_GEN_SIZE, _GEN_SIZE), _GEN_BG, dtype=int)
    grid[_GEN_SEP_ROW, :] = _GEN_SEP
    return grid


def _gen_valid_sprite_coords(coords: set[tuple[int, int]], h: int, w: int) -> bool:
    rows = {r for r, _c in coords}
    cols = {c for _r, c in coords}
    if len(rows) != h or len(cols) != w:
        return False
    mask = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        mask[r, c] = True
    return is_contiguous(mask, background=False, connectivity=8)


def _gen_fallback_sprite_coords(h: int, k: int) -> set[tuple[int, int]]:
    coords = {(0, 0), (h - 1, 1)}
    if h == 3:
        coords.add((1, 0))
    elif k >= 3:
        coords.add((max(0, h - 2), 0))
    if k >= 4:
        coords.add((h - 1, 0))
    return coords


def _gen_sprite_coords(h: int, w: int, k: int) -> set[tuple[int, int]]:
    all_cells = [(r, c) for r in range(h) for c in range(w)]
    for _ in range(_GEN_SPRITE_COORD_RETRIES):
        picks = np.random.choice(len(all_cells), size=k, replace=False)
        coords = {all_cells[int(idx)] for idx in np.atleast_1d(picks)}
        if _gen_valid_sprite_coords(coords, h, w):
            return coords
    return _gen_fallback_sprite_coords(h, k)


def _gen_sprite_patch() -> np.ndarray:
    h = int(np.random.choice(_GEN_SPRITE_HEIGHTS))
    w = _GEN_SPRITE_WIDTH
    k = int(np.random.randint(h, _GEN_SPRITE_MAX_PIXELS_STOP))
    coords = _gen_sprite_coords(h, w, k)
    color = int(np.random.choice(_GEN_SPRITE_COLORS))
    patch = np.full((h, w), _GEN_BG, dtype=int)
    for r, c in coords:
        patch[r, c] = color
    return patch


def _gen_place_marker(grid: np.ndarray, row0: int, col0: int) -> None:
    for dr, dc in _GEN_MARKER_SHAPE:
        grid[row0 + dr, col0 + dc] = _GEN_MARKER


def _gen_output_grid(marker_col: int, sprite: np.ndarray) -> np.ndarray | None:
    out = _gen_blank_grid()
    _gen_place_marker(out, _GEN_MARKER_TOP_ROW, marker_col)
    sh, sw = sprite.shape
    anchor_r, anchor_c = _GEN_MARKER_ANCHOR_REL
    paste_r, paste_c = anchor_r - sh + 1, marker_col + anchor_c - sw + 1
    if paste_r < 0 or paste_c < 0:
        return None
    paste(out, sprite, paste_r, paste_c, transparent=_GEN_BG)
    return out


def _gen_input_grid(marker_col: int, sprite: np.ndarray) -> np.ndarray | None:
    inp = _gen_blank_grid()
    _gen_place_marker(inp, _GEN_LOWER_ROW0, marker_col)
    lower = inp[_GEN_LOWER_ROW0:, :]
    try:
        sr, sc = random_free_location(lower, sprite, transparent=_GEN_BG)
    except ValueError:
        return None
    paste(inp, sprite, _GEN_LOWER_ROW0 + sr, sc, transparent=_GEN_BG)
    return inp


def _gen_attempt() -> np.ndarray | None:
    mc = int(np.random.randint(_GEN_MARKER_COL_MIN, _GEN_MARKER_COL_STOP))
    sprite = _gen_sprite_patch()

    out = _gen_output_grid(mc, sprite)
    if out is None:
        return None

    inp = _gen_input_grid(mc, sprite)
    if inp is None:
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid dc46ea44 input")
