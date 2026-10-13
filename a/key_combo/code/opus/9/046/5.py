import numpy as np
from collections import deque


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8

GUIDE_COLORS = (BLUE, RED, GREEN, YELLOW, PINK, ORANGE, TEAL)

GRID_H_RANGE = (11, 17)  # high is exclusive
GRID_W_RANGE = (12, 19)  # high is exclusive
GUIDE_ROW_CHOICES = (0, 1)
GUIDE_COL_CHOICES = (0, 1)
RECT_COUNT_CHOICES = (2, 3)
LARGE_RECT_H_RANGE = (4, 7)
LARGE_RECT_W_RANGE = (4, 8)
THIN_RECT_H_RANGE = (4, 7)
THIN_RECT_W_RANGE = (4, 8)
RANDOM_RECT_H_RANGE = (3, 7)
RANDOM_RECT_W_RANGE = (3, 8)
MIN_GAP = 1
MAX_GENERATION_ATTEMPTS = 300


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = MIN_GAP) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _find_guide(grid: np.ndarray) -> tuple[int, int, tuple[int, int, int]]:
    guide_mask = (grid != BLACK) & (grid != GREY)
    coords = np.argwhere(guide_mask)
    assert coords.shape[0] == 3, "expected exactly three guide cells"
    comps = _components(guide_mask)
    assert len(comps) == 1, "guide cells must form one connected bar"

    r0, c0, r1, c1 = _bbox(coords)
    assert r0 == r1 and c1 - c0 == 2, "guide must be a horizontal 1x3 strip"
    assert r0 <= 2 and c0 <= 2, "guide must sit in the top-left corner region"

    colors = tuple(int(grid[r0, c]) for c in range(c0, c1 + 1))
    assert len(set(colors)) == 3, "guide colors must be pairwise distinct"
    assert set(colors).issubset(set(GUIDE_COLORS)), "unexpected guide color"
    return r0, c0, colors


def _gray_rectangles(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    rects = []
    comps = _components(grid == GREY)
    assert comps, "expected at least one gray rectangle"
    for comp in comps:
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h >= 3 and w >= 3, "gray rectangles must include corner, edge, and interior cells"
        assert comp.shape[0] == h * w, "gray component must fill its bounding box"
        assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == GREY), "gray component must be solid"
        rects.append((r0, c0, r1, c1))
    return rects


def _paint_recolored_rectangles(
    out: np.ndarray, rects: list[tuple[int, int, int, int]], role_colors: tuple[int, int, int]
) -> np.ndarray:
    corner_color, edge_color, interior_color = role_colors
    for r0, c0, r1, c1 in rects:
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                on_h = r == r0 or r == r1
                on_v = c == c0 or c == c1
                if on_h and on_v:
                    out[r, c] = corner_color
                elif on_h or on_v:
                    out[r, c] = edge_color
                else:
                    out[r, c] = interior_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read a 1x3 color guide and use it to recolor each gray rectangle's corners, edges, and interior.

    Input structure:
    - The grid has black background, a horizontal 1x3 color guide in the top-left region, and 2-3 solid gray rectangles elsewhere.
    - The guide contains three distinct non-gray colors ordered from left to right.
    - Every gray component is a non-touching filled rectangle with height and width at least 3.

    Rule:
    - Read the guide from left to right as corner color, edge color, and interior color.
    - For every gray rectangle, classify each cell by its local position inside that rectangle: corner, non-corner border, or strict interior.
    - Recolor each rectangle cell with the guide color assigned to its role.
    - Replace the guide itself with black and leave the remaining black background unchanged.

    Key insight:
    - The guide does not name fixed colors; its left/middle/right positions define a lookup table from rectangle roles to output colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "input must be a non-empty 2D grid"
    assert set(map(int, np.unique(g))).issubset({BLACK, GREY, *GUIDE_COLORS}), "unexpected colors present"

    guide_r, guide_c0, role_colors = _find_guide(g)
    rects = _gray_rectangles(g)

    out = g.copy()
    out[guide_r, guide_c0 : guide_c0 + 3] = BLACK
    _paint_recolored_rectangles(out, rects, role_colors)

    assert not np.any(out == GREY), "all gray rectangle cells should be recolored"
    return out


def _sample_sizes(rng: np.random.Generator, rect_count: int) -> list[tuple[int, int]]:
    sizes = [
        (
            int(rng.integers(*LARGE_RECT_H_RANGE)),
            int(rng.integers(*LARGE_RECT_W_RANGE)),
        )
    ]

    if rng.random() < 0.5:
        sizes.append((3, int(rng.integers(*THIN_RECT_W_RANGE))))
    else:
        sizes.append((int(rng.integers(*THIN_RECT_H_RANGE)), 3))

    for _ in range(rect_count - 2):
        sizes.append(
            (
                int(rng.integers(*RANDOM_RECT_H_RANGE)),
                int(rng.integers(*RANDOM_RECT_W_RANGE)),
            )
        )

    rng.shuffle(sizes)
    return sizes


def _candidate_positions(h: int, w: int, rect_h: int, rect_w: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    positions = [(r, c) for r in range(h - rect_h + 1) for c in range(w - rect_w + 1)]
    rng.shuffle(positions)
    return positions


def _compute_expected_output(
    shape: tuple[int, int],
    rects: list[tuple[int, int, int, int]],
    role_colors: tuple[int, int, int],
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    return _paint_recolored_rectangles(out, rects, role_colors)


def _validate_generated(
    grid: np.ndarray,
    rects: list[tuple[int, int, int, int]],
    guide_box: tuple[int, int, int, int],
    role_colors: tuple[int, int, int],
) -> bool:
    h, w = grid.shape
    assert h <= 30 and w <= 30
    assert len(rects) in RECT_COUNT_CHOICES

    guide_mask = (grid != BLACK) & (grid != GREY)
    assert guide_mask.sum() == 3
    assert len(_components(guide_mask)) == 1
    assert guide_box == (guide_box[0], guide_box[1], guide_box[0], guide_box[1] + 2)

    for i, box_a in enumerate(rects):
        for box_b in rects[i + 1 :]:
            assert _boxes_separated(box_a, box_b)
        assert _boxes_separated(box_a, guide_box)

    expected = _compute_expected_output(grid.shape, rects, role_colors)
    actual = main(grid)
    assert np.array_equal(actual, expected), "solver output must match the semantic construction"

    swapped = (role_colors[1], role_colors[0], role_colors[2])
    swapped_out = _compute_expected_output(grid.shape, rects, swapped)
    assert not np.array_equal(expected, swapped_out), "guide order must affect the result"

    for color in role_colors:
        assert np.any(expected == color), "every guide role color must appear in the output"
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(*GRID_H_RANGE))
    w = int(rng.integers(*GRID_W_RANGE))
    grid = np.zeros((h, w), dtype=int)

    guide_r = int(rng.choice(GUIDE_ROW_CHOICES))
    guide_c = int(rng.choice(GUIDE_COL_CHOICES))
    if guide_c + 2 >= w:
        return None
    guide_box = (guide_r, guide_c, guide_r, guide_c + 2)

    guide_colors = tuple(int(x) for x in rng.choice(GUIDE_COLORS, size=3, replace=False))
    grid[guide_r, guide_c : guide_c + 3] = np.array(guide_colors, dtype=int)

    rect_count = int(rng.choice(RECT_COUNT_CHOICES))
    sizes = _sample_sizes(rng, rect_count)

    rects: list[tuple[int, int, int, int]] = []
    for rect_h, rect_w in sizes:
        placed = False
        for r0, c0 in _candidate_positions(h, w, rect_h, rect_w, rng):
            box = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
            if not _boxes_separated(box, guide_box):
                continue
            if any(not _boxes_separated(box, other) for other in rects):
                continue
            rects.append(box)
            grid[r0 : r0 + rect_h, c0 : c0 + rect_w] = GREY
            placed = True
            break
        if not placed:
            return None

    if not any((r1 - r0 + 1) >= 4 and (c1 - c0 + 1) >= 4 for r0, c0, r1, c1 in rects):
        return None
    if not any((r1 - r0 + 1) == 3 or (c1 - c0 + 1) == 3 for r0, c0, r1, c1 in rects):
        return None

    _validate_generated(grid, rects, guide_box, guide_colors)
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for the guide-driven rectangle recoloring task")
