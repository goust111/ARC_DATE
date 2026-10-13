from __future__ import annotations

import numpy as np

from arc_utils import Color, bbox_coords, bbox_h, bbox_w, colors_present, components_mask, get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each 3-cell foreground L by coloring its missing 2x2 corner blue.

    Input structure:
    - The grid has one dominant background color and one foreground color.
    - Each 4-connected foreground object has exactly three pixels inside a 2x2
      bounding box, forming an L shape.

    Rule:
    - For every foreground L object, find the one background pixel in its 2x2
      bounding box.
    - Paint that missing corner blue while preserving all original pixels.

    Key insight:
    - The teal objects are incomplete 2x2 squares; the only changed cells are
      the absent corners of those local 2x2 regions.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    fg_colors = colors_present(g, exclude={bg})
    assert len(fg_colors) == 1, "expected exactly one foreground color"

    out = g.copy()
    components = components_mask(g == int(fg_colors[0]), connectivity=4)
    assert components, "expected at least one foreground L object"

    for coords in components:
        assert coords.shape[0] == 3, "each object must have three pixels"
        box = bbox_coords(coords)
        assert bbox_h(box) == 2 and bbox_w(box) == 2, "each object must fit a 2x2 box"

        r0, c0, r1, c1 = box
        present = {(int(r), int(c)) for r, c in coords}
        box_cells = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
        missing = box_cells - present
        assert len(missing) == 1, "each 2x2 box must have exactly one missing corner"

        r, c = next(iter(missing))
        assert g[r, c] == bg, "missing corner must be background"
        out[r, c] = int(Color.BLUE)

    return out

# --------------- generator (o5x) ---------------

# Canvas and L-object geometry.
GEN_GRID_SHAPE = (7, 7)
GEN_L_SIZE = 2
GEN_L_OFFSETS = ((0, 0), (0, 1), (1, 0), (1, 1))
GEN_L_MISSING_CELLS = 1
GEN_L_FILLED_CELLS = len(GEN_L_OFFSETS) - GEN_L_MISSING_CELLS
GEN_ANCHORS = tuple((r, c) for r in range(GEN_GRID_SHAPE[0] - GEN_L_SIZE + 1)
                    for c in range(GEN_GRID_SHAPE[1] - GEN_L_SIZE + 1))

# Object count, placement spacing, and retry limits.
GEN_COUNT_MIN = 2
GEN_COUNT_MAX = 4
GEN_ADJACENCY_GAP = 1
GEN_RETRY_LIMIT = 300

# Generator colors.
GEN_BG = int(Color.BLACK)
GEN_FG = int(Color.TEAL)
GEN_FILL = int(Color.BLUE)

def _gen_l_box(anchor: tuple[int, int]) -> tuple[int, int, int, int]:
    return anchor[0], anchor[1], anchor[0] + GEN_L_SIZE - 1, anchor[1] + GEN_L_SIZE - 1

def _gen_boxes_compatible(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    """Return whether two inclusive 2x2 boxes avoid overlap and 4-adjacency."""
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    rows_overlap = ar0 <= br1 and br0 <= ar1
    cols_overlap = ac0 <= bc1 and bc0 <= ac1
    side_touch = (
        rows_overlap and (ac1 + GEN_ADJACENCY_GAP == bc0 or bc1 + GEN_ADJACENCY_GAP == ac0)
        or cols_overlap and (ar1 + GEN_ADJACENCY_GAP == br0 or br1 + GEN_ADJACENCY_GAP == ar0)
    )
    return not (rows_overlap and cols_overlap or side_touch)

def _gen_choose_anchors(count: int) -> list[tuple[int, int]] | None:
    """Pick count anchors whose 2x2 boxes do not overlap or touch by edge."""
    anchors = list(GEN_ANCHORS)

    def search(chosen: list[tuple[int, int]], candidates: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
        if len(chosen) == count:
            return chosen
        if len(candidates) < count - len(chosen):
            return None

        for idx in np.random.permutation(len(candidates)):
            anchor = candidates[int(idx)]
            box = _gen_l_box(anchor)
            if not all(_gen_boxes_compatible(box, _gen_l_box(prev)) for prev in chosen):
                continue
            remaining = [
                cand
                for j, cand in enumerate(candidates)
                if j != int(idx) and _gen_boxes_compatible(box, _gen_l_box(cand))
            ]
            result = search(chosen + [anchor], remaining)
            if result is not None:
                return result
        return None

    return search([], anchors)

def _gen_expected_output(grid: np.ndarray) -> np.ndarray:
    """Compute the oracle output by filling each detected L missing corner."""
    out = grid.copy()
    for r, c in GEN_ANCHORS:
        patch = grid[r : r + GEN_L_SIZE, c : c + GEN_L_SIZE]
        if (
            np.count_nonzero(patch == GEN_FG) == GEN_L_FILLED_CELLS
            and np.count_nonzero(patch == GEN_BG) == GEN_L_MISSING_CELLS
        ):
            dr, dc = np.argwhere(patch == GEN_BG)[0]
            out[r + int(dr), c + int(dc)] = GEN_FILL
    return out

def _gen_render_l(grid: np.ndarray, anchor: tuple[int, int], missing: int) -> None:
    """Paint one L object in-place, leaving the sampled missing corner black."""
    r, c = anchor
    for idx, (dr, dc) in enumerate(GEN_L_OFFSETS):
        if idx != missing:
            grid[r + dr, c + dc] = GEN_FG

def _gen_attempt() -> np.ndarray | None:
    """Sample anchors, render L objects, and return one candidate input."""
    grid = np.full(GEN_GRID_SHAPE, GEN_BG, dtype=int)
    count = int(np.random.randint(GEN_COUNT_MIN, GEN_COUNT_MAX + 1))
    anchors = _gen_choose_anchors(count)
    if anchors is None:
        return None

    for anchor in anchors:
        _gen_render_l(grid, anchor, int(np.random.randint(len(GEN_L_OFFSETS))))
    return grid

def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is None:
            continue
        expected = _gen_expected_output(grid)
        try:
            actual = main(grid)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate a valid 3aa6fb7a input")
