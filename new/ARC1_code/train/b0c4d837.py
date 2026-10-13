import numpy as np

from arc_utils import bbox, colors_present, get_bg, paste
import random


_SPIRAL = (
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 2),
    (2, 2),
    (2, 1),
    (2, 0),
    (1, 0),
    (1, 1),
)


def _find_frame_color(g: np.ndarray, bg: int, colors: list[int]) -> tuple[int, tuple[int, int, int, int]]:
    candidates = []
    for color in colors:
        mask = g == color
        box = bbox(mask)
        assert box is not None
        r0, c0, r1, c1 = box
        if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
            continue
        is_u_frame = (
            mask[r1, c0 : c1 + 1].all()
            and mask[r0 : r1 + 1, c0].all()
            and mask[r0 : r1 + 1, c1].all()
        )
        if not is_u_frame:
            continue

        other_colors = [other for other in colors if other != color]
        assert len(other_colors) == 1
        other_rows, other_cols = np.where(g == other_colors[0])
        assert len(other_rows) > 0
        inside_frame = (
            (other_rows >= r0)
            & (other_rows < r1)
            & (other_cols > c0)
            & (other_cols < c1)
        )
        if inside_frame.all():
            candidates.append((color, box))

    assert len(candidates) == 1
    frame_color, frame_box = candidates[0]
    assert frame_color != bg
    return int(frame_color), frame_box


def main(grid: np.ndarray) -> np.ndarray:
    """
    Summarize the empty space above liquid in a U-shaped frame as a 3x3 clockwise gauge.
    Input structure
    - The grid has a background, one U-shaped frame color, and one fill color inside the frame.
    - The fill color forms a solid bottom-aligned rectangle in the frame interior.
    Rule
    - Count the empty interior rows above the first filled row.
    - Output a 3x3 background grid and paint that many cells with the fill color along a clockwise spiral from the top-left.
    Key insight
    - The output is not a scaled copy of the liquid; it is a compact gauge of the headspace height.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    colors = colors_present(g, exclude={bg})
    assert len(colors) == 2

    frame_color, (r0, c0, r1, c1) = _find_frame_color(g, bg, colors)
    fill_color = [color for color in colors if color != frame_color]
    assert len(fill_color) == 1
    fill_color = int(fill_color[0])

    interior = g[r0:r1, c0 + 1 : c1]
    assert interior.size > 0
    assert np.isin(interior, [bg, fill_color]).all()

    fill_mask = interior == fill_color
    row_has_fill = fill_mask.any(axis=1)
    assert row_has_fill.any()
    first_filled_row = int(np.argmax(row_has_fill))
    assert not row_has_fill[:first_filled_row].any()
    assert row_has_fill[first_filled_row:].all()
    assert fill_mask[first_filled_row:].all()

    empty_rows = first_filled_row
    assert 0 <= empty_rows <= len(_SPIRAL)

    out = np.full((3, 3), bg, dtype=int)
    for r, c in _SPIRAL[:empty_rows]:
        out[r, c] = fill_color
    return out

# --------------- generator (o5x) ---------------

# Fixed task colors.
_GEN_BG, _GEN_FRAME_COLOR, _GEN_FILL_COLOR = 0, 5, 8
# Canvas and U-frame size bounds.
_GEN_GRID_H_MIN, _GEN_GRID_H_MAX = 5, 9
_GEN_GRID_W_MIN, _GEN_GRID_W_MAX = 6, 9
_GEN_INTERIOR_H_MIN, _GEN_INTERIOR_H_MAX = 3, 7
_GEN_INTERIOR_W_MIN, _GEN_INTERIOR_W_MAX = 2, 5
_GEN_FRAME_H_EXTRA, _GEN_FRAME_W_EXTRA = 1, 2
# Headspace and placement sampling.
_GEN_EMPTY_ROWS_MIN, _GEN_EMPTY_ROWS_MAX = 1, 6
_GEN_TOP_MARGINS = (1, 2)
_GEN_SIDE_MARGINS = (1, 2)
# Retry and oracle-check settings.
_GEN_MAX_ATTEMPTS = 300
_GEN_OUTPUT_SHAPE = (3, 3)


def _gen_expected_output(empty_rows: int) -> np.ndarray:
    out = np.full(_GEN_OUTPUT_SHAPE, _GEN_BG, dtype=int)
    for r, c in _SPIRAL[:empty_rows]:
        out[r, c] = _GEN_FILL_COLOR
    return out


def _gen_sample_layout() -> tuple[int, int, int, int, int, int] | None:
    """Return empty rows, interior size, and margins for one candidate."""
    # empty_rows 7-9 cannot coexist with a top margin, a filled row, and max grid height 9.
    empty_rows = random.randint(_GEN_EMPTY_ROWS_MIN, _GEN_EMPTY_ROWS_MAX)
    min_interior_h = max(_GEN_INTERIOR_H_MIN, empty_rows + 1)
    interior_h = random.randint(min_interior_h, _GEN_INTERIOR_H_MAX)
    interior_w = random.randint(_GEN_INTERIOR_W_MIN, _GEN_INTERIOR_W_MAX)
    frame_h, frame_w = interior_h + _GEN_FRAME_H_EXTRA, interior_w + _GEN_FRAME_W_EXTRA

    top_choices = [
        m for m in _GEN_TOP_MARGINS if _GEN_GRID_H_MIN <= frame_h + m <= _GEN_GRID_H_MAX
    ]
    lr_choices = [
        (left, right) for left in _GEN_SIDE_MARGINS for right in _GEN_SIDE_MARGINS
        if _GEN_GRID_W_MIN <= frame_w + left + right <= _GEN_GRID_W_MAX
    ]
    if not top_choices or not lr_choices:
        return None

    margin_top = random.choice(top_choices)
    margin_left, margin_right = random.choice(lr_choices)
    return empty_rows, interior_h, interior_w, margin_top, margin_left, margin_right


def _gen_render_grid(layout: tuple[int, int, int, int, int, int]) -> np.ndarray:
    """Render a sampled layout tuple into a padded grid."""
    empty_rows, interior_h, interior_w, margin_top, margin_left, margin_right = layout
    frame_h, frame_w = interior_h + _GEN_FRAME_H_EXTRA, interior_w + _GEN_FRAME_W_EXTRA
    height, width = frame_h + margin_top, frame_w + margin_left + margin_right

    grid = np.full((height, width), _GEN_BG, dtype=int)
    frame = np.full((frame_h, frame_w), _GEN_BG, dtype=int)
    frame[:, 0] = frame[:, -1] = _GEN_FRAME_COLOR
    frame[-1, :] = _GEN_FRAME_COLOR

    frame[empty_rows:interior_h, 1:-1] = _GEN_FILL_COLOR
    paste(grid, frame, margin_top, margin_left)
    return grid


def _gen_attempt() -> np.ndarray | None:
    layout = _gen_sample_layout()
    if layout is None:
        return None

    grid = _gen_render_grid(layout)
    expected = _gen_expected_output(layout[0])
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b0c4d837 input")
