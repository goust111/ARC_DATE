import numpy as np

from arc_utils import bbox, colors_present, get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Add a surrounding rectangle of the inner color around a two-color framed rectangle.

    Input structure:
    - The background is the dominant color.
    - The foreground is one solid rectangular region made from exactly two colors.
    - One color forms a rectangular frame; the other forms a solid rectangle strictly inside it.

    Rule:
    - Preserve the input.
    - Enlarge the foreground bbox upward/downward by the inner rectangle height and
      left/right by the inner rectangle width, limited to the largest full symmetric
      enlargement that fits on the grid.
    - Fill background cells in that enlarged bbox with the inner rectangle color.

    Key insight:
    - The small inner rectangle is the next outer layer's color and size cue; the
      output simply adds that next concentric layer outside the existing frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)

    bg = get_bg(g)
    colors = colors_present(g, exclude={bg})
    assert len(colors) == 2, "expected exactly two foreground colors"

    outer_box = bbox(g != bg)
    assert outer_box is not None, "expected a foreground rectangle"
    r0, c0, r1, c1 = outer_box
    foreground_region = g[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(foreground_region != bg), "foreground bbox must be filled"

    inner_candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    for color in colors:
        color_box = bbox(g == color)
        assert color_box is not None, "foreground color must be present"
        cr0, cc0, cr1, cc1 = color_box
        if r0 < cr0 and cr1 < r1 and c0 < cc0 and cc1 < c1:
            inner_candidates.append((color, color_box))
    assert len(inner_candidates) == 1, "expected one color strictly inside the frame"

    inner_color, inner_box = inner_candidates[0]
    frame_color = next(color for color in colors if color != inner_color)
    ir0, ic0, ir1, ic1 = inner_box
    inner_region = g[ir0 : ir1 + 1, ic0 : ic1 + 1]
    assert np.all(inner_region == inner_color), "inner color must form a solid rectangle"

    frame_mask = np.ones_like(foreground_region, dtype=bool)
    frame_mask[ir0 - r0 : ir1 - r0 + 1, ic0 - c0 : ic1 - c0 + 1] = False
    assert np.all(foreground_region[frame_mask] == frame_color), "outer color must form the frame"

    inner_height = ir1 - ir0 + 1
    inner_width = ic1 - ic0 + 1
    vertical_pad = min(inner_height, r0, height - 1 - r1)
    horizontal_pad = min(inner_width, c0, width - 1 - c1)
    assert vertical_pad > 0 and horizontal_pad > 0, "expected room for an outer layer"

    out = g.copy()
    enlarged = out[
        r0 - vertical_pad : r1 + vertical_pad + 1,
        c0 - horizontal_pad : c1 + horizontal_pad + 1,
    ]
    enlarged[enlarged == bg] = inner_color
    return out

# --------------- generator (o5x) ---------------

# Canvas and palette bounds.
GRID_SIZE_MIN = 11
GRID_SIZE_MAX = 21
FOREGROUND_COLORS = np.arange(1, 10)
FOREGROUND_COLOR_COUNT = 2

# Concentric rectangle geometry.
FRAME_THICKNESSES = (1, 2)
INNER_SIZE_MIN = 1
INNER_SIZE_MAX = 4
MIN_EXPANSION_PAD = 1

# Placement bias and retry limit.
EDGE_PLACEMENT_PROB = 0.85
NEAR_START_PLACEMENT_PROB = 0.8
MAX_GENERATION_ATTEMPTS = 300


def _gen_place_axis(size: int, full: int, pad: int, inner: int) -> int:
    slack = size - full
    if slack < 0:
        raise ValueError("full rectangle does not fit")
    if pad < inner:
        return 0 if np.random.random() < EDGE_PLACEMENT_PROB else slack
    if np.random.random() < NEAR_START_PLACEMENT_PROB:
        return int(np.random.randint(0, slack // 2 + 1))
    return int(np.random.randint(0, slack + 1))


def _gen_sample_layout(height: int, width: int) -> tuple[int, ...] | None:
    """Uniformly sample one fitting (thickness, inner, pad, outer, full) layout."""
    choices = []
    for thickness in FRAME_THICKNESSES:
        for inner_h in range(INNER_SIZE_MIN, INNER_SIZE_MAX + 1):
            for inner_w in range(INNER_SIZE_MIN, INNER_SIZE_MAX + 1):
                for vertical_pad in range(MIN_EXPANSION_PAD, inner_h + 1):
                    for horizontal_pad in range(MIN_EXPANSION_PAD, inner_w + 1):
                        outer_h = inner_h + 2 * thickness
                        outer_w = inner_w + 2 * thickness
                        full_h = outer_h + 2 * vertical_pad
                        full_w = outer_w + 2 * horizontal_pad
                        if full_h <= height and full_w <= width:
                            choices.append((
                                thickness, inner_h, inner_w, vertical_pad, horizontal_pad,
                                outer_h, outer_w, full_h, full_w,
                            ))
    if not choices:
        return None
    return choices[int(np.random.randint(0, len(choices)))]


def _gen_render_grids(height, width, layout, top, left, frame_color, inner_color):
    """Return (input, expected_output) for a sampled concentric layout."""
    thickness, inner_h, inner_w, vertical_pad, horizontal_pad, outer_h, outer_w, full_h, full_w = layout
    outer_top = top + vertical_pad
    outer_left = left + horizontal_pad
    inner_top = outer_top + thickness
    inner_left = outer_left + thickness

    expected = np.zeros((height, width), dtype=int)
    expected[top : top + full_h, left : left + full_w] = inner_color
    expected[outer_top : outer_top + outer_h, outer_left : outer_left + outer_w] = frame_color
    expected[inner_top : inner_top + inner_h, inner_left : inner_left + inner_w] = inner_color

    inp = np.zeros((height, width), dtype=int)
    inp[outer_top : outer_top + outer_h, outer_left : outer_left + outer_w] = frame_color
    inp[inner_top : inner_top + inner_h, inner_left : inner_left + inner_w] = inner_color

    return inp, expected


def _gen_attempt() -> np.ndarray | None:
    height = int(np.random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX + 1))
    width = int(np.random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX + 1))

    layout = _gen_sample_layout(height, width)
    if layout is None:
        return None

    _, inner_h, inner_w, vertical_pad, horizontal_pad, _, _, full_h, full_w = layout
    top = _gen_place_axis(height, full_h, vertical_pad, inner_h)
    left = _gen_place_axis(width, full_w, horizontal_pad, inner_w)

    frame_color, inner_color = np.random.choice(
        FOREGROUND_COLORS, size=FOREGROUND_COLOR_COUNT, replace=False
    )
    frame_color = int(frame_color)
    inner_color = int(inner_color)

    inp, expected = _gen_render_grids(height, width, layout, top, left, frame_color, inner_color)
    if np.array_equal(main(inp), expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {MAX_GENERATION_ATTEMPTS} attempts")
