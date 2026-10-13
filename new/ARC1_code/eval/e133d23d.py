import numpy as np

from arc_utils import Color, colors_present, get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay the two side masks split by the separator column and recolor their union red.

    Input structure
    - A 2D grid with odd width and a solid vertical separator column in the middle.
    - The separator divides the remaining columns into equal left and right regions.
    - Each side region contains background plus one foreground mask color.

    Rule
    - Pair cells by their position in the left and right regions.
    - Mark an output cell when either paired input cell is foreground.
    - Use the input background elsewhere and recolor all marked cells red.

    Key insight
    - The separator is only a partition marker; the output is the union of the two aligned side masks.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height > 0 and width >= 3 and width % 2 == 1, "expected an odd-width grid with two side regions"

    bg = get_bg(g)
    mid = width // 2
    separator = int(g[0, mid])
    assert separator != bg, "separator must be non-background"
    assert np.all(g[:, mid] == separator), "expected a solid middle separator column"

    left = g[:, :mid]
    right = g[:, mid + 1 :]
    assert left.shape == right.shape, "side regions must have matching shapes"
    assert separator not in colors_present(left) + colors_present(right), "separator should not appear inside masks"

    left_colors = colors_present(left, exclude={bg})
    right_colors = colors_present(right, exclude={bg})
    assert len(left_colors) == 1, "expected one foreground mask color on the left"
    assert len(right_colors) == 1, "expected one foreground mask color on the right"
    assert left_colors[0] != right_colors[0], "expected distinct left and right mask colors"

    occupied = (left != bg) | (right != bg)
    out = np.full(left.shape, bg, dtype=int)
    out[occupied] = Color.RED
    return out

# --------------- generator (o5x) ---------------
# Canvas and union-mask bounds.
HEIGHT_RANGE = range(3, 7)
WIDTH_RANGE = range(3, 6)
MIN_UNION_FG, MAX_UNION_FG = 4, 8
UNION_FILL_NUM, UNION_FILL_DEN = 2, 5

# Side-mask split constraints.
MIN_OVERLAP, MAX_OVERLAP = 1, 2
LEFT_MIN_FG, LEFT_MAX_FG = 2, 7
RIGHT_MIN_FG, RIGHT_MAX_FG = 3, 5

# Palette and retry policy.
GENERATOR_PALETTE = np.array([0, 1, 3, 4, 5, 6, 7, 8, 9], dtype=int)
GENERATOR_RETRIES = 300


def _render_split(h: int, w: int, left_only: np.ndarray, right_only: np.ndarray, both: np.ndarray):
    """Render the sampled split into an input grid and return (bg, grid)."""
    bg, sep, left_color, right_color = map(int, np.random.choice(GENERATOR_PALETTE, 4, replace=False))
    grid = np.full((h, 2 * w + 1), bg, dtype=int)
    grid[:, w] = sep
    if left_only.size:
        grid[left_only[:, 0], left_only[:, 1]] = left_color
    if right_only.size:
        grid[right_only[:, 0], w + 1 + right_only[:, 1]] = right_color
    if both.size:
        grid[both[:, 0], both[:, 1]] = left_color
        grid[both[:, 0], w + 1 + both[:, 1]] = right_color
    return bg, grid


def _gen_attempt() -> np.ndarray | None:
    valid_shapes = []
    for cand_h in HEIGHT_RANGE:
        for cand_w in WIDTH_RANGE:
            cand_area = cand_h * cand_w
            low_fg = max(MIN_UNION_FG, (UNION_FILL_NUM * cand_area + UNION_FILL_DEN - 1) // UNION_FILL_DEN)
            high_fg = min(MAX_UNION_FG, cand_area - 1)
            if low_fg <= high_fg:
                valid_shapes.append((cand_h, cand_w, low_fg, high_fg))

    h, w, min_fg, max_fg = valid_shapes[int(np.random.randint(len(valid_shapes)))]
    area = h * w
    fg_count = int(np.random.randint(min_fg, max_fg + 1))

    out_mask = np.zeros((h, w), dtype=bool)
    out_mask.flat[np.random.choice(area, fg_count, replace=False)] = True
    coords = np.argwhere(out_mask)

    feasible_splits = []
    for both_count in range(MIN_OVERLAP, min(MAX_OVERLAP, fg_count) + 1):
        remaining = fg_count - both_count
        bg_count = 2 * area - fg_count - both_count
        low_left_only = max(0, LEFT_MIN_FG - both_count, fg_count - RIGHT_MAX_FG, fg_count - bg_count + 1)
        high_left_only = min(
            remaining,
            LEFT_MAX_FG - both_count,
            fg_count - RIGHT_MIN_FG,
            bg_count - both_count - 1,
        )
        if bg_count > h and low_left_only <= high_left_only:
            feasible_splits.append((both_count, low_left_only, high_left_only))

    if not feasible_splits:
        return None

    both_count, low_left_only, high_left_only = feasible_splits[int(np.random.randint(len(feasible_splits)))]
    left_only_count = int(np.random.randint(low_left_only, high_left_only + 1))

    order = np.random.permutation(fg_count)
    both = coords[order[:both_count]]
    remaining_coords = coords[order[both_count:]]
    left_only = remaining_coords[:left_only_count]
    right_only = remaining_coords[left_only_count:]

    bg, grid = _render_split(h, w, left_only, right_only, both)

    out = np.full((h, w), bg, dtype=int)
    out[out_mask] = Color.RED
    if not np.array_equal(main(grid), out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATOR_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
