from __future__ import annotations

import numpy as np

from arc_utils import Color, colors_present, get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay the two equal input panels and mark positions that are empty in both panels.

    Input structure
    - The input is a 2D grid whose width is split into two equal side-by-side panels.
    - The background is the dominant color.
    - Each panel contains exactly one non-background foreground color.

    Rule
    - Align the left and right panels by row and column.
    - A cell in the output is grey iff both aligned panel cells are background.
    - All other output cells are background.

    Key insight
    - The output is not the union or intersection of the colored objects; it is the
      complement of their overlaid occupancy mask.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert width % 2 == 0, "expected two equal-width panels"

    bg = get_bg(g)
    assert bg == Color.BLACK, "expected black background"

    mid = width // 2
    left = g[:, :mid]
    right = g[:, mid:]
    assert left.shape == right.shape

    left_colors = colors_present(left, exclude={bg})
    right_colors = colors_present(right, exclude={bg})
    assert len(left_colors) == 1, "expected one foreground color in the left panel"
    assert len(right_colors) == 1, "expected one foreground color in the right panel"
    assert left_colors[0] != right_colors[0], "expected distinct panel foreground colors"

    empty_in_both = (left == bg) & (right == bg)
    out = np.full((height, mid), bg, dtype=int)
    out[empty_in_both] = Color.GREY
    return out

# --------------- generator (o5x) ---------------

# Canvas dimensions; upper bounds are exclusive for rng.integers().
PANEL_HEIGHT_RANGE, PANEL_WIDTH_RANGE = (3, 7), (5, 10)

# Occupancy constraints for each panel and the grey output mask.
FG_MIN_PCT, FG_MAX_PCT = 35, 55
GREY_MIN_PCT, GREY_MAX_PCT = 15, 40
GREY_MIN_COUNT = 3

# Panel color sampling.
CANONICAL_COLOR_RATE = 0.65
PANEL_COLOR_PALETTE = np.array(
    (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON),
    dtype=int,
)

GENERATION_RETRY_LIMIT = 300  # Rejection sampling budget.


def _gen_ceil_pct(n: int, pct: int) -> int:
    return (n * pct + 99) // 100


def _gen_valid_count_triples(n_cells: int) -> list[tuple[int, int, int]]:
    fg_min = _gen_ceil_pct(n_cells, FG_MIN_PCT)
    fg_max = (n_cells * FG_MAX_PCT) // 100
    grey_min = max(GREY_MIN_COUNT, _gen_ceil_pct(n_cells, GREY_MIN_PCT))
    grey_max = (n_cells * GREY_MAX_PCT) // 100

    triples: list[tuple[int, int, int]] = []
    for left_count in range(fg_min, fg_max + 1):
        for right_count in range(fg_min, fg_max + 1):
            for grey_count in range(grey_min, grey_max + 1):
                active_count = n_cells - grey_count
                if max(left_count, right_count) <= active_count <= left_count + right_count:
                    triples.append((left_count, right_count, grey_count))
    return triples


def _gen_sample_counts(rng: np.random.Generator, n_cells: int) -> tuple[int, int, int, int] | None:
    triples = _gen_valid_count_triples(n_cells)
    if not triples:
        return None

    left_count, right_count, grey_count = triples[int(rng.integers(len(triples)))]
    active_count = n_cells - grey_count
    overlap_count = left_count + right_count - active_count
    left_only_count = left_count - overlap_count
    right_only_count = right_count - overlap_count
    if min(overlap_count, left_only_count, right_only_count) < 0:
        return None
    return grey_count, overlap_count, left_only_count, right_only_count


def _gen_masks_from_counts(rng, shape, counts):
    grey_count, overlap_count, left_only_count, right_only_count = counts
    n_cells = shape[0] * shape[1]
    order = rng.permutation(n_cells)
    grey_idx = order[:grey_count]
    active_idx = order[grey_count:]

    both_end = overlap_count
    left_end = both_end + left_only_count
    both_idx = active_idx[:both_end]
    left_only_idx = active_idx[both_end:left_end]
    right_only_idx = active_idx[left_end : left_end + right_only_count]

    left_mask = np.zeros(n_cells, dtype=bool)
    right_mask = np.zeros(n_cells, dtype=bool)
    expected_grey = np.zeros(n_cells, dtype=bool)
    expected_grey[grey_idx] = True
    left_mask[both_idx] = True
    right_mask[both_idx] = True
    left_mask[left_only_idx] = True
    right_mask[right_only_idx] = True

    return left_mask.reshape(shape), right_mask.reshape(shape), expected_grey.reshape(shape)


def _gen_panel_colors(rng: np.random.Generator) -> tuple[int, int]:
    if rng.random() < CANONICAL_COLOR_RATE:
        return Color.GREEN, Color.RED

    chosen = rng.choice(PANEL_COLOR_PALETTE, size=2, replace=False)
    return int(chosen[0]), int(chosen[1])


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    height = int(rng.integers(*PANEL_HEIGHT_RANGE))
    panel_width = int(rng.integers(*PANEL_WIDTH_RANGE))
    shape = (height, panel_width)
    n_cells = height * panel_width

    counts = _gen_sample_counts(rng, n_cells)
    if counts is None:
        return None

    left_mask, right_mask, expected_grey = _gen_masks_from_counts(rng, shape, counts)
    expected_out = np.full((height, panel_width), Color.BLACK, dtype=int)
    expected_out[expected_grey] = Color.GREY

    left_color, right_color = _gen_panel_colors(rng)
    inp = np.full((height, panel_width * 2), Color.BLACK, dtype=int)
    inp[:, :panel_width][left_mask] = left_color
    inp[:, panel_width:][right_mask] = right_color

    return inp if np.array_equal(main(inp), expected_out) else None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 66f2d22f input")
