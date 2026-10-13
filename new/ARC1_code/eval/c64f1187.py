from collections import Counter

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg
import random


def _unique_most_common_non_bg(g: np.ndarray, bg: int) -> int:
    counts = Counter(map(int, g.ravel()))
    colors = [c for c in counts if c != bg]
    assert colors, "expected at least one non-background color"
    max_count = max(counts[c] for c in colors)
    candidates = [c for c in colors if counts[c] == max_count]
    assert len(candidates) == 1, "slot fill color must be uniquely most common among non-bg colors"
    return int(candidates[0])


def _slot_blocks(g: np.ndarray, slot_color: int) -> list[tuple[int, int, int, int]]:
    blocks: list[tuple[int, int, int, int]] = []
    for coords in components_mask(g == slot_color, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(coords)
        assert (r1 - r0 + 1, c1 - c0 + 1) == (2, 2), "each slot must occupy a 2x2 bbox"
        assert len(coords) in (3, 4), "a slot must be full grey or have one colored control pixel"
        blocks.append((r0, c0, r1, c1))
    assert blocks, "expected a lower array of slot blocks"
    return blocks


def _legend_templates(
    g: np.ndarray, bg: int, slot_color: int, blocks: list[tuple[int, int, int, int]]
) -> dict[int, np.ndarray]:
    first_slot_row = min(r0 for r0, _c0, _r1, _c1 in blocks)
    colors = set(map(int, np.unique(g))) - {bg, slot_color}

    template_candidates: list[int] = []
    for color in colors:
        coords = np.argwhere(g == color)
        if len(coords) > 1 and np.all(coords[:, 0] < first_slot_row):
            template_candidates.append(int(color))
    assert len(template_candidates) == 1, "expected one legend template ink color"
    template_color = template_candidates[0]

    templates: dict[int, np.ndarray] = {}
    covered = np.zeros(g.shape, dtype=bool)
    marker_mask = (g != bg) & (g != slot_color) & (g != template_color)
    for r, c in np.argwhere(marker_mask):
        if r >= first_slot_row:
            continue
        marker_color = int(g[r, c])
        assert marker_color not in templates, "each marker color must have one legend entry"
        assert r + 2 < g.shape[0] and c + 2 < g.shape[1], "legend template must fit below-right of marker"
        mask = g[r + 1 : r + 3, c + 1 : c + 3] == template_color
        assert mask.shape == (2, 2) and mask.any(), "each legend marker must point to a 2x2 template"
        templates[marker_color] = mask
        covered[r + 1 : r + 3, c + 1 : c + 3] |= mask

    assert templates, "expected at least one color-to-template legend entry"
    assert np.array_equal(covered, g == template_color), "legend entries must cover all template ink"
    return templates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Render the lower grey slot array by replacing each colored control pixel with the matching legend template.

    Input structure
    A background grid has a top legend: each colored marker sits one row above and one column left of a
    2x2 template drawn in a shared template ink color.  Below it is a rectangular array of 2x2 grey slots,
    where a slot may contain one colored control pixel in place of grey.

    Rule
    Read the legend as color -> 2x2 template mask.  Convert the lower slot array into a compact array with
    the same row/column layout and one background separator row/column between neighboring tiles.  Empty
    grey slots become blank tiles; a colored slot becomes that color's legend mask recolored with the
    control color.

    Key insight
    The colored pixel in a grey slot names which top legend template to stamp; its position inside the slot
    is irrelevant.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    slot_color = _unique_most_common_non_bg(g, bg)
    blocks = _slot_blocks(g, slot_color)
    templates = _legend_templates(g, bg, slot_color, blocks)

    row_starts = sorted({r0 for r0, _c0, _r1, _c1 in blocks})
    col_starts = sorted({c0 for _r0, c0, _r1, _c1 in blocks})
    block_by_start = {(r0, c0): (r0, c0, r1, c1) for r0, c0, r1, c1 in blocks}
    assert len(block_by_start) == len(row_starts) * len(col_starts), "slot blocks must form a full array"

    out_h = len(row_starts) * 2 + max(0, len(row_starts) - 1)
    out_w = len(col_starts) * 2 + max(0, len(col_starts) - 1)
    out = np.full((out_h, out_w), bg, dtype=int)

    for br, r0 in enumerate(row_starts):
        for bc, c0 in enumerate(col_starts):
            assert (r0, c0) in block_by_start, "every row/column slot position must exist"
            tile = g[r0 : r0 + 2, c0 : c0 + 2]
            control_colors = set(map(int, tile.ravel())) - {bg, slot_color}
            assert len(control_colors) <= 1, "a slot may contain at most one control color"
            if not control_colors:
                continue
            control_color = control_colors.pop()
            assert control_color in templates, "every control color must be defined by the legend"
            rr = br * 3
            cc = bc * 3
            out[rr : rr + 2, cc : cc + 2] = np.where(templates[control_color], control_color, bg)

    return out

# --------------- generator (o5x) ---------------
# Colors.
BG_COLOR, TEMPLATE_INK, SLOT_COLOR, MARKER_PALETTE = 0, 1, 5, [2, 3, 4, 7, 8]

# Legend, slot, and output structure.
SLOT_ROWS_RANGE, SLOT_COLS_RANGE, LEGEND_ENTRY_RANGE = (3, 4), (4, 5), (3, 4)
FILL_RATIO_RANGE, TILE_SIZE, SLOT_STRIDE = (0.58, 0.67), 2, 3

# Canvas and placement bounds.
GRID_HEIGHT_RANGE, GRID_WIDTH_RANGE = (17, 20), (18, 20)
MARKER_ROW_RANGE, GAP_RANGE, LEGEND_SPACING_RANGE = (0, 1), (2, 4), (4, 5)
LEGEND_ENTRY_SPAN = 3

# Retry limit and template mask choices.
GENERATION_RETRIES = 300
FULL_TEMPLATE_MASK = np.ones((2, 2), dtype=bool)
L_SHAPE_MASKS = [np.array(mask, dtype=bool) for mask in (
    [[0, 1], [1, 1]], [[1, 0], [1, 1]], [[1, 1], [0, 1]], [[1, 1], [1, 0]]
)]


def _build_output(rows, cols, marker_colors, mask_by_color):
    """Return the expected output and the control color assigned to each filled slot."""
    out_h = rows * SLOT_STRIDE - 1
    out_w = cols * SLOT_STRIDE - 1
    out = np.full((out_h, out_w), BG_COLOR, dtype=int)
    num_slots = rows * cols
    lo, hi = FILL_RATIO_RANGE
    fill_counts = [k for k in range(num_slots + 1) if lo <= k / num_slots <= hi]
    fill_count = random.choice(fill_counts)
    filled = set(random.sample(range(num_slots), fill_count))
    slot_controls = {}
    for idx in filled:
        sr, sc = divmod(idx, cols)
        color = random.choice(marker_colors)
        slot_controls[(sr, sc)] = color
        rr = sr * SLOT_STRIDE
        cc = sc * SLOT_STRIDE
        out[rr : rr + TILE_SIZE, cc : cc + TILE_SIZE] = np.where(mask_by_color[color], color, BG_COLOR)
    return out, slot_controls


def _render_input(rows, cols, num_entries, out_h, out_w, marker_colors, mask_by_color, slot_controls):
    """Render the top legend and lower slot grid, or return None if they cannot fit."""
    width = random.randint(*GRID_WIDTH_RANGE)
    spacing = random.randint(*LEGEND_SPACING_RANGE)
    marker_span = spacing * (num_entries - 1) + LEGEND_ENTRY_SPAN
    if marker_span > width or out_w > width:
        return None

    marker_row = random.randint(*MARKER_ROW_RANGE)
    gap = random.randint(*GAP_RANGE)
    first_slot_row = marker_row + TILE_SIZE + 1 + gap
    min_height = first_slot_row + out_h
    height = (
        min_height
        if min_height > GRID_HEIGHT_RANGE[1]
        else random.randint(max(GRID_HEIGHT_RANGE[0], min_height), GRID_HEIGHT_RANGE[1])
    )
    if not (GRID_HEIGHT_RANGE[0] <= height <= GRID_HEIGHT_RANGE[1]):
        return None
    inp = np.full((height, width), BG_COLOR, dtype=int)
    marker_col0 = random.randint(0, width - marker_span)
    for i, color in enumerate(marker_colors):
        c = marker_col0 + i * spacing
        inp[marker_row, c] = color
        inp[marker_row + 1 : marker_row + 3, c + 1 : c + 3] = np.where(mask_by_color[color], TEMPLATE_INK, BG_COLOR)
    slot_col0 = random.randint(0, width - out_w)
    for sr in range(rows):
        for sc in range(cols):
            r = first_slot_row + sr * SLOT_STRIDE
            c = slot_col0 + sc * SLOT_STRIDE
            inp[r : r + TILE_SIZE, c : c + TILE_SIZE] = SLOT_COLOR
            control = slot_controls.get((sr, sc))
            if control is not None:
                inp[r + 1, c + 1] = control
    return inp


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(*SLOT_ROWS_RANGE)
    cols = random.randint(*SLOT_COLS_RANGE)
    num_entries = random.randint(*LEGEND_ENTRY_RANGE)

    marker_colors = random.sample(MARKER_PALETTE, num_entries)
    masks = [FULL_TEMPLATE_MASK] + random.sample(L_SHAPE_MASKS, num_entries - 1)
    random.shuffle(masks)
    mask_by_color = dict(zip(marker_colors, masks))

    out, slot_controls = _build_output(rows, cols, marker_colors, mask_by_color)
    inp = _render_input(
        rows, cols, num_entries, out.shape[0], out.shape[1], marker_colors, mask_by_color, slot_controls
    )
    if inp is None:
        return None

    try:
        if np.array_equal(main(inp), out):
            return inp
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid c64f1187 input")
