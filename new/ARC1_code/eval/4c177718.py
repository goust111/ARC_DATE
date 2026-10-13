import numpy as np

from arc_utils import Color, bbox, colors_present, crop_bbox, get_bg, paste
import itertools


def _required_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    b = bbox(mask)
    assert b is not None
    return b


def _single_separator_row(g: np.ndarray, bg: int) -> tuple[int, int]:
    rows = [
        r
        for r in range(g.shape[0])
        if int(g[r, 0]) != bg and np.all(g[r, :] == int(g[r, 0]))
    ]
    assert len(rows) == 1
    r = rows[0]
    return r, int(g[r, 0])


def _red_t_arm_is_top(marker: np.ndarray) -> bool:
    assert marker.shape == (3, 3)
    row_counts = marker.sum(axis=1)
    full_rows = np.flatnonzero(row_counts == marker.shape[1])
    assert len(full_rows) == 1
    arm_row = int(full_rows[0])
    assert arm_row in (0, marker.shape[0] - 1)

    stem_col = marker.shape[1] // 2
    for r in range(marker.shape[0]):
        if r == arm_row:
            continue
        assert bool(marker[r, stem_col])
        assert int(marker[r, :].sum()) == 1

    return arm_row == 0


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the red T marker in the upper demonstration to stamp the colored sprite above or below the lower blue template.

    Input structure
    - A solid non-background separator row divides the grid into an upper demonstration panel and a lower output panel.
    - The lower panel contains one foreground template object; the upper panel contains the same template color, one red T marker, and one differently colored sprite.
    - The marker is a 3x3, 5-cell T whose horizontal arm is either its top row or its bottom row.

    Rule
    - Start the output from the lower panel, excluding the separator and upper demonstration.
    - Crop the upper colored sprite by its bounding box and keep its orientation.
    - If the red T arm is on top, paste the sprite flush below the lower template; if the arm is on bottom, paste it flush above the lower template.
    - Align the sprite's left edge with the lower template's left edge and paste only non-background sprite pixels.

    Key insight
    - The red T does not get copied; its arm marks which horizontal edge of the sprite should touch the lower template.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    sep_r, separator_color = _single_separator_row(g, bg)
    upper = g[:sep_r, :]
    out = g[sep_r + 1 :, :].copy()

    marker_color = Color.RED
    assert np.any(upper == marker_color)

    lower_colors = colors_present(out, exclude={bg})
    assert len(lower_colors) == 1
    template_color = lower_colors[0]
    assert np.any(upper == template_color)

    sprite_colors = colors_present(
        upper,
        exclude={bg, separator_color, marker_color, template_color},
    )
    assert len(sprite_colors) == 1
    sprite_color = sprite_colors[0]

    marker_bbox = _required_bbox(upper == marker_color)
    mr0, mc0, mr1, mc1 = marker_bbox
    marker = upper[mr0 : mr1 + 1, mc0 : mc1 + 1] == marker_color
    arm_is_top = _red_t_arm_is_top(marker)

    sprite_bbox = _required_bbox(upper == sprite_color)
    sprite = crop_bbox(upper, sprite_bbox)
    assert set(map(int, np.unique(sprite))).issubset({bg, sprite_color})

    target_bbox = _required_bbox(out == template_color)
    tr0, tc0, tr1, tc1 = target_bbox
    sprite_h, sprite_w = sprite.shape
    target_h = tr1 - tr0 + 1
    target_w = tc1 - tc0 + 1
    assert (sprite_h, sprite_w) == (target_h, target_w)

    paste_r = tr1 + 1 if arm_is_top else tr0 - sprite_h
    paste_c = tc0
    assert 0 <= paste_r and paste_r + sprite_h <= out.shape[0]
    assert 0 <= paste_c and paste_c + sprite_w <= out.shape[1]
    out = paste(out, sprite, paste_r, paste_c, transparent=bg)
    return out

# --------------- generator (o5x) ---------------

# Canvas, colors, and fixed upper-panel slots.
BACKGROUND_COLOR, TEMPLATE_COLOR, MARKER_COLOR = Color.BLACK, Color.BLUE, Color.RED
GRID_SHAPE, OUTPUT_SHAPE = (15, 15), (9, 15)
SEPARATOR_ROW, SEPARATOR_COLOR, LOWER_PANEL_ROW_OFFSET = 5, Color.GREY, 6
SPRITE_COLORS = (Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL)
UPPER_TEMPLATE_POS, UPPER_MARKER_POS, UPPER_SPRITE_POS = (1, 2), (1, 6), (1, 10)
# 3x3 mask cell-count ranges.
SHAPE_SIZE, TEMPLATE_CELL_RANGE, SPRITE_CELL_RANGE = 3, (5, 8), (4, 8)
# Lower-panel template placement bounds.
TOP_ARM_ROWS, BOTTOM_ARM_ROWS, LOWER_COL_RANGE = (1, 3), (3, 5), (1, 8)
# Generation retry guard.
GENERATION_RETRIES = 300


def _gen_full_bbox_mask(cell_range: tuple[int, int]) -> np.ndarray:
    min_cells, max_cells = cell_range
    k = int(np.random.randint(min_cells, max_cells + 1))
    masks = []
    for cells in itertools.combinations(range(SHAPE_SIZE * SHAPE_SIZE), k):
        mask = np.zeros((SHAPE_SIZE, SHAPE_SIZE), dtype=bool)
        mask.flat[list(cells)] = True
        if mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any():
            masks.append(mask)
    return masks[int(np.random.randint(len(masks)))].copy()


def _gen_t_marker(arm_is_top: bool) -> np.ndarray:
    marker = np.zeros((SHAPE_SIZE, SHAPE_SIZE), dtype=int)
    marker[0 if arm_is_top else -1, :] = MARKER_COLOR
    stem_rows = slice(1, None) if arm_is_top else slice(0, -1)
    marker[stem_rows, SHAPE_SIZE // 2] = MARKER_COLOR
    return marker


def _sample_layout(arm_is_top: bool) -> tuple[int, int, int]:
    """Return lower template row/col and the adjacent sprite row."""
    row_range = TOP_ARM_ROWS if arm_is_top else BOTTOM_ARM_ROWS
    lower_r = int(np.random.randint(row_range[0], row_range[1] + 1))
    lower_c = int(np.random.randint(LOWER_COL_RANGE[0], LOWER_COL_RANGE[1] + 1))
    sprite_r = lower_r + SHAPE_SIZE if arm_is_top else lower_r - SHAPE_SIZE
    return lower_r, lower_c, sprite_r


def _gen_attempt() -> np.ndarray | None:
    template_mask = _gen_full_bbox_mask(TEMPLATE_CELL_RANGE)
    sprite_mask = _gen_full_bbox_mask(SPRITE_CELL_RANGE)
    template = np.where(template_mask, TEMPLATE_COLOR, BACKGROUND_COLOR)

    sprite_color = int(np.random.choice(SPRITE_COLORS))
    sprite = np.where(sprite_mask, sprite_color, BACKGROUND_COLOR)

    arm_is_top = bool(np.random.randint(2))
    lower_r, lower_c, sprite_r = _sample_layout(arm_is_top)

    output = np.zeros(OUTPUT_SHAPE, dtype=int)
    paste(output, template, lower_r, lower_c, transparent=BACKGROUND_COLOR)
    paste(output, sprite, sprite_r, lower_c, transparent=BACKGROUND_COLOR)

    inp = np.zeros(GRID_SHAPE, dtype=int)
    inp[SEPARATOR_ROW, :] = SEPARATOR_COLOR
    paste(inp, template, *UPPER_TEMPLATE_POS, transparent=BACKGROUND_COLOR)
    paste(inp, _gen_t_marker(arm_is_top), *UPPER_MARKER_POS, transparent=BACKGROUND_COLOR)
    paste(inp, sprite, *UPPER_SPRITE_POS, transparent=BACKGROUND_COLOR)
    paste(inp, template, lower_r + LOWER_PANEL_ROW_OFFSET, lower_c, transparent=BACKGROUND_COLOR)

    try:
        return inp if np.array_equal(main(inp), output) else None
    except Exception:
        return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 4c177718 input")
