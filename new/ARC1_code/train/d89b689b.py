from __future__ import annotations

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the 2x2 marker block with the singleton colors found in matching surrounding quadrants.

    Input structure
    - The grid has a dominant background, one solid 2x2 foreground marker block, and four other foreground singleton pixels.
    - Each singleton lies diagonally outside the marker block: above/below it and left/right of it.

    Rule
    - Clear all foreground pixels from the canvas.
    - Place each singleton color into the cell of the marker block in the same relative quadrant: upper-left, upper-right, lower-left, or lower-right.

    Key insight
    - The 2x2 block is a compact target window; each scattered singleton's side relative to that block determines its destination cell.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    components = find_components(g, bg=bg, connectivity=4, monochromatic=True)

    block_components = [
        comp
        for comp in components
        if int(comp["area"]) == 4
        and bbox_h(comp["bbox"]) == 2
        and bbox_w(comp["bbox"]) == 2
        and np.all(
            g[
                comp["bbox"][0] : comp["bbox"][2] + 1,
                comp["bbox"][1] : comp["bbox"][3] + 1,
            ]
            == int(comp["color"])
        )
    ]
    assert len(block_components) == 1, "expected exactly one solid 2x2 marker block"
    block = block_components[0]
    r0, c0, r1, c1 = map(int, block["bbox"])

    singletons = [comp for comp in components if comp is not block]
    assert len(singletons) == 4, "expected four singleton source pixels"

    out = np.full_like(g, bg)
    destinations: set[tuple[int, int]] = set()
    for comp in singletons:
        assert int(comp["area"]) == 1, "each non-marker foreground object must be a singleton"
        r, c = map(int, comp["coords"][0])

        assert r < r0 or r > r1, "singleton must be above or below the marker block"
        assert c < c0 or c > c1, "singleton must be left or right of the marker block"

        dest_r = r0 if r < r0 else r1
        dest_c = c0 if c < c0 else c1
        dest = (dest_r, dest_c)
        assert dest not in destinations, "expected one singleton per marker-block quadrant"
        destinations.add(dest)
        out[dest] = int(comp["color"])

    assert destinations == {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    return out

# --------------- generator (o5x) ---------------

# Canvas and marker geometry.
GRID_MIN = 0
GRID_SHAPE = (10, 10)
BLOCK_SIZE = 2
BLOCK_MARGIN = 2

# Colors and retry controls.
BACKGROUND_COLOR = 0
BLOCK_COLOR = 8
SOURCE_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]
GENERATION_RETRIES = 300


def _gen_cell(rlo: int, rhi: int, clo: int, chi: int) -> tuple[int, int]:
    return int(np.random.randint(rlo, rhi)), int(np.random.randint(clo, chi))


def _gen_attempt() -> np.ndarray | None:
    h, w = GRID_SHAPE

    r0 = int(np.random.randint(BLOCK_MARGIN, h - BLOCK_MARGIN - BLOCK_SIZE + 1))
    c0 = int(np.random.randint(BLOCK_MARGIN, w - BLOCK_MARGIN - BLOCK_SIZE + 1))
    row_stop = r0 + BLOCK_SIZE
    col_stop = c0 + BLOCK_SIZE

    colors = np.random.choice(SOURCE_COLORS, size=BLOCK_SIZE * BLOCK_SIZE, replace=False)
    out = np.full(GRID_SHAPE, BACKGROUND_COLOR, dtype=int)
    out[r0:row_stop, c0:col_stop] = np.asarray(colors, dtype=int).reshape(BLOCK_SIZE, BLOCK_SIZE)

    inp = out.copy()
    inp[r0:row_stop, c0:col_stop] = BLOCK_COLOR

    singleton_regions = [
        (GRID_MIN, r0, GRID_MIN, c0),
        (GRID_MIN, r0, col_stop, w),
        (row_stop, h, GRID_MIN, c0),
        (row_stop, h, col_stop, w),
    ]
    for color, region in zip(colors, singleton_regions):
        rr, cc = _gen_cell(*region)
        inp[rr, cc] = int(color)

    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid d89b689b input")
