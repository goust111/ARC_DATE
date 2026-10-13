from __future__ import annotations

import numpy as np

from arc_utils import Color, bbox_h, bbox_w, colors_present, crop_bbox, find_components, get_bg, paste
import random


LEGEND = np.array(
    [
        [Color.BLUE, Color.ORANGE, Color.PINK],
        [Color.YELLOW, Color.TEAL, Color.GREY],
        [Color.RED, Color.MAROON, Color.GREEN],
    ],
    dtype=int,
)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each black 5x5 region's missing interior positions with a fixed 3x3 positional color legend.

    Input structure
    - The grid uses teal as the dominant background and black for one or more
      5x5 regions.
    - Each black region has a black one-pixel border and a 3x3 interior that
      may already contain teal pixels.

    Rule
    - Overlay the legend 1 7 6 / 4 8 5 / 2 9 3 on the 3x3 interior of every
      5x5 black region.
    - Wherever an interior non-center legend position is still black, recolor it
      to the corresponding legend color; leave existing teal pixels, the center,
      and the black border unchanged.

    Key insight
    - The colored output pixels are not derived from the teal motif shape; they
      are the fixed position labels for the missing cells of each 3x3 interior.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    assert bg == Color.TEAL
    fg_colors = colors_present(g, exclude={bg})
    assert fg_colors == [Color.BLACK]

    out = g.copy()
    components = find_components(
        g,
        bg=bg,
        connectivity=8,
        monochromatic=True,
        colors=[Color.BLACK],
    )
    assert components

    for component in components:
        b = component["bbox"]
        assert bbox_h(b) == 5 and bbox_w(b) == 5
        r0, c0, _r1, _c1 = b
        patch = crop_bbox(g, b)
        assert np.all((patch == Color.BLACK) | (patch == bg))
        assert np.all(patch[[0, -1], :] == Color.BLACK)
        assert np.all(patch[:, [0, -1]] == Color.BLACK)
        assert np.any(patch[1:4, 1:4] == bg)

        for rr in range(3):
            for cc in range(3):
                color = int(LEGEND[rr, cc])
                if color == bg:
                    continue
                grid_r = int(r0) + rr + 1
                grid_c = int(c0) + cc + 1
                if g[grid_r, grid_c] == Color.BLACK:
                    out[grid_r, grid_c] = color

    return out

# --------------- generator (att2) ---------------

# Canvas and region geometry.
MIN_GRID_SIZE, MAX_GRID_SIZE = 9, 16
REGION_SIZE, INTERIOR_SIZE, REGION_GAP = 5, 3, 1

# Region count and interior-hole sampling.
MIN_REGIONS, MAX_REGIONS = 1, 4
MIN_TEAL_INTERIOR, MAX_TEAL_INTERIOR = 1, 6
MAX_AREA_PER_REGION = 48

# Generation retry limit and colors.
MAX_GENERATION_ATTEMPTS = 300
GEN_BG, GEN_FG = Color.TEAL, Color.BLACK
GEN_PATCH = np.full((REGION_SIZE, REGION_SIZE), GEN_FG, dtype=int)
GEN_INTERIOR = [
    (rr, cc) for rr in range(INTERIOR_SIZE) for cc in range(INTERIOR_SIZE)
]


def _gen_dim_slots(length):
    return (int(length) + REGION_GAP) // (REGION_SIZE + REGION_GAP)


def _gen_shape_pairs(n_regions):
    pairs = []
    for height in range(MIN_GRID_SIZE, MAX_GRID_SIZE + 1):
        for width in range(MIN_GRID_SIZE, MAX_GRID_SIZE + 1):
            if _gen_dim_slots(height) * _gen_dim_slots(width) < n_regions:
                continue
            if height * width <= MAX_AREA_PER_REGION * n_regions:
                continue
            pairs.append((height, width))
    return pairs


def _gen_separated(a, b):
    ar, ac = a
    br, bc = b
    row_gap = ar + REGION_SIZE < br or br + REGION_SIZE < ar
    col_gap = ac + REGION_SIZE < bc or bc + REGION_SIZE < ac
    return row_gap or col_gap


def _gen_search_regions(candidates, need):
    if need == 0:
        return []
    for i, pos in enumerate(candidates):
        rest = [other for other in candidates[i + 1 :] if _gen_separated(pos, other)]
        if len(rest) < need - 1:
            continue
        tail = _gen_search_regions(rest, need - 1)
        if tail is not None:
            return [pos] + tail
    return None


def _gen_pick_regions(height, width, n_regions):
    candidates = [
        (r, c) for r in range(height - REGION_SIZE + 1) for c in range(width - REGION_SIZE + 1)
    ]
    random.shuffle(candidates)
    return _gen_search_regions(candidates, n_regions)


def _gen_expected_output(grid, regions):
    out = grid.copy()
    for r0, c0 in regions:
        for rr in range(INTERIOR_SIZE):
            for cc in range(INTERIOR_SIZE):
                color = int(LEGEND[rr, cc])
                if color != GEN_BG and grid[r0 + rr + 1, c0 + cc + 1] == GEN_FG:
                    out[r0 + rr + 1, c0 + cc + 1] = color
    return out


def _gen_stamp_region(grid, r0, c0):
    paste(grid, GEN_PATCH, r0, c0)
    teal_count = random.randint(MIN_TEAL_INTERIOR, MAX_TEAL_INTERIOR)
    for rr, cc in random.sample(GEN_INTERIOR, teal_count):
        grid[r0 + rr + 1, c0 + cc + 1] = GEN_BG


def _gen_attempt():
    n_regions = random.randint(MIN_REGIONS, MAX_REGIONS)
    height, width = random.choice(_gen_shape_pairs(n_regions))
    regions = _gen_pick_regions(height, width, n_regions)
    if regions is None:
        return None

    grid = np.full((height, width), GEN_BG, dtype=int)
    for r0, c0 in regions:
        _gen_stamp_region(grid, r0, c0)

    expected = _gen_expected_output(grid, regions)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input():
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid a834deea input")
