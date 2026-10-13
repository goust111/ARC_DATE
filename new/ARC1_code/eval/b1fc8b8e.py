import itertools

import numpy as np

from arc_utils import colors_present, components_mask as _gen_components_mask, get_bg, paste as _gen_paste
import itertools as _gen_itertools
import random as _gen_random


def _find_unique_sprite_mask(g: np.ndarray, fg_color: int) -> tuple[tuple[int, int], ...]:
    fg = {tuple(map(int, rc)) for rc in np.argwhere(g == fg_color)}
    assert len(fg) % 4 == 0
    sprite_area = len(fg) // 4
    assert 1 <= sprite_area <= 4

    h, w = g.shape
    offsets = ((0, 0), (0, 1), (1, 0), (1, 1))
    solutions = []
    for mask in itertools.combinations(offsets, sprite_area):
        candidates = []
        for r in range(h - 1):
            for c in range(w - 1):
                cells = frozenset((r + dr, c + dc) for dr, dc in mask)
                if cells <= fg:
                    candidates.append(cells)

        for copies in itertools.combinations(candidates, 4):
            covered = set()
            for cells in copies:
                if covered & cells:
                    break
                covered.update(cells)
            else:
                if covered == fg:
                    solutions.append(mask)

    assert len(solutions) == 1
    return solutions[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Separate four overlapping-looking copies of one 2x2 sprite into the four corners of a 5x5 canvas.

    Input structure
    A 6x6 grid has one background color and one foreground color.  The foreground
    pixels are exactly coverable by four translated copies of one 2x2 sprite mask.

    Rule
    Infer the unique 2x2 sprite mask whose four translated copies cover all
    foreground pixels without overlap, then stamp that mask into the upper-left,
    upper-right, lower-left, and lower-right 2x2 regions of a 5x5 output.

    Key insight
    The tangled input arrangement is only a compact packing of four identical
    small sprites; the output regularizes the packing by inserting a blank middle
    row and column between the copies.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (6, 6)

    bg = get_bg(g)
    fg_colors = colors_present(g, exclude={bg})
    assert len(fg_colors) == 1
    fg_color = fg_colors[0]

    mask = _find_unique_sprite_mask(g, fg_color)
    out = np.full((5, 5), bg, dtype=int)
    for base_r, base_c in ((0, 0), (0, 3), (3, 0), (3, 3)):
        for dr, dc in mask:
            out[base_r + dr, base_c + dc] = fg_color
    return out

# --------------- generator (o5x) ---------------
# Color choices.
_GEN_BG_COLOR = 0
_GEN_FG_COLORS = tuple(range(1, 10))

# Canvas and placement geometry.
_GEN_INPUT_SHAPE = (6, 6)
_GEN_OUTPUT_SHAPE = (5, 5)
_GEN_PATCH_SHAPE = (2, 2)
_GEN_INPUT_ANCHOR_ROWS = range(4)
_GEN_INPUT_ANCHOR_COLS = range(5)
_GEN_OUTPUT_ANCHORS = ((0, 0), (0, 3), (3, 0), (3, 3))

# Sprite masks and packing filters.
_GEN_SPRITE_MASKS = (
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
)
_GEN_COPY_COUNT = 4
_GEN_MIN_MAX_ROW, _GEN_MIN_MAX_COL = 3, 4
_GEN_MAX_COMPONENTS = 2
_GEN_MAX_ATTEMPTS = 300


def _gen_sprite_patch(mask, bg, fg):
    patch = np.full(_GEN_PATCH_SHAPE, bg, dtype=int)
    rows, cols = zip(*mask)
    patch[rows, cols] = fg
    return patch


def _gen_output_from_mask(mask, bg, fg):
    out = np.full(_GEN_OUTPUT_SHAPE, bg, dtype=int)
    patch = _gen_sprite_patch(mask, bg, fg)
    for r, c in _GEN_OUTPUT_ANCHORS:
        _gen_paste(out, patch, r, c, transparent=bg)
    return out


def _gen_is_compact_packing(occupied):
    """Match the original bbox and component-count filter for input packings."""
    coords = np.array(list(occupied), dtype=int)
    min_r, min_c = coords.min(axis=0)
    max_r, max_c = coords.max(axis=0)
    if not (min_r == 0 and min_c == 0 and max_r >= _GEN_MIN_MAX_ROW and max_c >= _GEN_MIN_MAX_COL):
        return False

    bitmask = np.zeros(_GEN_INPUT_SHAPE, dtype=bool)
    rows, cols = zip(*occupied)
    bitmask[rows, cols] = True
    return len(_gen_components_mask(bitmask, connectivity=4)) <= _GEN_MAX_COMPONENTS


def _gen_valid_anchor_sets(mask):
    anchors = []
    for r in _GEN_INPUT_ANCHOR_ROWS:
        for c in _GEN_INPUT_ANCHOR_COLS:
            cells = frozenset((r + dr, c + dc) for dr, dc in mask)
            anchors.append((r, c, cells))

    valid = []
    for combo in _gen_itertools.combinations(anchors, _GEN_COPY_COUNT):
        occupied = set()
        placements = []
        for r, c, cells in combo:
            if occupied & cells:
                break
            occupied.update(cells)
            placements.append((r, c))
        else:
            if _gen_is_compact_packing(occupied):
                valid.append(tuple(placements))
    return valid


def _gen_attempt():
    bg = _GEN_BG_COLOR
    fg = _gen_random.choice(_GEN_FG_COLORS)
    mask = _gen_random.choice(_GEN_SPRITE_MASKS)

    expected = _gen_output_from_mask(mask, bg, fg)
    patch = _gen_sprite_patch(mask, bg, fg)
    anchor_sets = _gen_valid_anchor_sets(mask)
    if not anchor_sets:
        return None

    inp = np.full(_GEN_INPUT_SHAPE, bg, dtype=int)
    for r, c in _gen_random.choice(anchor_sets):
        _gen_paste(inp, patch, r, c, transparent=bg)

    try:
        got = main(inp)
    except Exception:
        return None
    if not np.array_equal(got, expected):
        return None
    return inp


def generate_input():
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError('failed to generate valid b1fc8b8e input')
