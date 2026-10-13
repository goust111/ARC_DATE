from __future__ import annotations

import numpy as np

from arc_utils import Color, colors_present, get_bg


def _divider_row(g: np.ndarray, bg: int) -> int:
    h, w = g.shape
    candidates: list[int] = []
    for r in range(h):
        row_colors = np.unique(g[r])
        if row_colors.size != 1:
            continue
        color = int(row_colors[0])
        if color == bg:
            continue
        if int(np.count_nonzero(g == color)) != w:
            continue
        if r == h - r - 1:
            candidates.append(r)

    assert len(candidates) == 1, "expected one centered full-width divider row"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay the two masks separated by the divider and recolor their union green.

    Input structure:
    - A black-background grid contains one centered full-width divider row.
    - The divider row splits the grid into equal top and bottom rectangular masks.
    - Each mask uses exactly one foreground color on black background.

    Rule:
    - Remove the divider and compare the top and bottom masks cell by cell.
    - Output a grid with the same shape as one mask.
    - A cell is green if either corresponding input mask cell is foreground;
      otherwise it remains black.

    Key insight:
    - The output is not copied from one half; it is the positional union of the
      two same-sized masks on either side of the separator.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    assert bg == Color.BLACK, "expected black background"

    sep = _divider_row(g, bg)
    top = g[:sep]
    bottom = g[sep + 1 :]
    assert top.shape == bottom.shape, "divider must split two equal mask regions"

    top_colors = colors_present(top, exclude={bg})
    bottom_colors = colors_present(bottom, exclude={bg})
    assert len(top_colors) == 1, "top mask must have one foreground color"
    assert len(bottom_colors) == 1, "bottom mask must have one foreground color"
    assert top_colors[0] != bottom_colors[0], "mask foreground colors must differ"

    divider_color = int(g[sep, 0])
    assert divider_color not in top_colors + bottom_colors, "divider color must be separate"

    return np.where((top != bg) | (bottom != bg), Color.GREEN, bg).astype(int)

# --------------- generator (o5x) ---------------

# Canvas dimensions; randint upper bounds are exclusive.
GEN_HEIGHT_RANGE, GEN_WIDTH_RANGE = (3, 7), (4, 8)

# Density constraints as integer percentages.
GEN_MASK_DENSITY_PCT, GEN_UNION_DENSITY_PCT = (30, 70), (65, 90)
GEN_MIN_BOTH_BLACK_PCT, GEN_MAX_OVERLAP_PCT = 10, 40

# Non-black colors and retry budget.
GEN_PALETTE = np.array([1, 2, 4, 5, 6, 7, 8, 9], dtype=int)
GEN_MAX_ATTEMPTS = 300

def _gen_ceil_pct(n: int, pct: int) -> int:
    return (pct * n + 99) // 100


def _gen_attempt() -> np.ndarray | None:
    h = int(np.random.randint(*GEN_HEIGHT_RANGE))
    w = int(np.random.randint(*GEN_WIDTH_RANGE))
    n = h * w

    min_mask = _gen_ceil_pct(n, GEN_MASK_DENSITY_PCT[0])
    max_mask = (GEN_MASK_DENSITY_PCT[1] * n) // 100
    min_union = _gen_ceil_pct(n, GEN_UNION_DENSITY_PCT[0])
    max_union = min(
        (GEN_UNION_DENSITY_PCT[1] * n) // 100,
        n - _gen_ceil_pct(n, GEN_MIN_BOTH_BLACK_PCT),
    )
    union_count = int(np.random.randint(min_union, max_union + 1))

    max_overlap = min((GEN_MAX_OVERLAP_PCT * n) // 100, union_count)
    valid_overlaps = []
    for overlap_count in range(1, max_overlap + 1):
        lo = max(min_mask, overlap_count, union_count + overlap_count - max_mask)
        hi = min(max_mask, union_count, union_count + overlap_count - min_mask)
        if lo <= hi:
            valid_overlaps.append(overlap_count)
    if not valid_overlaps:
        return None

    overlap_count = int(np.random.choice(valid_overlaps))
    top_lo = max(min_mask, overlap_count, union_count + overlap_count - max_mask)
    top_hi = min(max_mask, union_count, union_count + overlap_count - min_mask)
    top_count = int(np.random.randint(top_lo, top_hi + 1))

    cells = np.random.permutation(n)[:union_count]
    both_cells = cells[:overlap_count]
    top_only_count = top_count - overlap_count
    top_only_cells = cells[overlap_count:overlap_count + top_only_count]
    bottom_only_cells = cells[overlap_count + top_only_count:]

    top_color, bottom_color, divider_color = np.random.permutation(GEN_PALETTE)[:3]

    top = np.zeros((h, w), dtype=int)
    bottom = np.zeros((h, w), dtype=int)
    top.flat[both_cells] = int(top_color)
    top.flat[top_only_cells] = int(top_color)
    bottom.flat[both_cells] = int(bottom_color)
    bottom.flat[bottom_only_cells] = int(bottom_color)

    grid = np.zeros((2 * h + 1, w), dtype=int)
    grid[:h] = top
    grid[h] = int(divider_color)
    grid[h + 1:] = bottom

    expected = np.zeros((h, w), dtype=int)
    expected.flat[cells] = Color.GREEN
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_MAX_ATTEMPTS} attempts")
