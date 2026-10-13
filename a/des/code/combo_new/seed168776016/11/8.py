from __future__ import annotations

import numpy as np


BG = 0
RNG = np.random.default_rng()

DIRECTIONS = ("up", "right", "down", "left")
BASE_REL = (
    (-2, 0),
    (-1, 0),
    (1, 0),
    (2, 0),
    (0, -2),
    (0, -1),
    (0, 1),
    (0, 2),
)
TIP_REL = {
    "up": (-3, 0),
    "right": (0, 3),
    "down": (3, 0),
    "left": (0, -3),
}
INPUT_SHAPES = {
    direction: frozenset(BASE_REL + (TIP_REL[direction],))
    for direction in DIRECTIONS
}
SHAPE_TO_DIRECTION = {shape: direction for direction, shape in INPUT_SHAPES.items()}

FULL_STAR = frozenset(BASE_REL + tuple(TIP_REL.values()) + ((0, 0),))
RESERVE_REL = frozenset(
    (dr + ar, dc + ac)
    for dr, dc in FULL_STAR
    for ar in (-1, 0, 1)
    for ac in (-1, 0, 1)
)
NEIGHBORS_8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

GRID_SIDE_RANGE = {3: (18, 23), 4: (20, 23), 5: (21, 23)}
MAX_GEN_ATTEMPTS = 400


def _rotate_clockwise(rel: tuple[int, int], turns: int) -> tuple[int, int]:
    dr, dc = rel
    for _ in range(turns % 4):
        dr, dc = dc, -dr
    return dr, dc


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS_8:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _legend_order(grid: np.ndarray) -> list[int]:
    cols = np.flatnonzero(grid[0] != BG)
    assert 3 <= len(cols) <= 5, "legend must contain 3 to 5 colors"
    if len(cols) > 1:
        assert np.all(np.diff(cols) >= 2), "legend colors must be separated by black gaps"
    colors = [int(grid[0, c]) for c in cols]
    assert len(set(colors)) == len(colors), "legend colors must be distinct"
    return colors


def _match_center(coords: np.ndarray) -> tuple[tuple[int, int], frozenset[tuple[int, int]]]:
    pts = {(int(r), int(c)) for r, c in coords}
    rows = [r for r, _c in pts]
    cols = [c for _r, c in pts]
    matches: list[tuple[tuple[int, int], frozenset[tuple[int, int]]]] = []
    for cr in range(min(rows) - 1, max(rows) + 2):
        for cc in range(min(cols) - 1, max(cols) + 2):
            rel = frozenset((r - cr, c - cc) for r, c in pts)
            if rel in SHAPE_TO_DIRECTION:
                matches.append(((cr, cc), rel))
    assert len(matches) == 1, "each object must match exactly one hollow-cross template"
    return matches[0]


def _object_specs(grid: np.ndarray) -> list[tuple[int, tuple[int, int], frozenset[tuple[int, int]]]]:
    scene = grid != BG
    scene[0, :] = False
    specs: list[tuple[int, tuple[int, int], frozenset[tuple[int, int]]]] = []
    for comp in _components(scene):
        colors = {int(grid[r, c]) for r, c in comp}
        assert len(colors) == 1, "each object must be monochromatic"
        color = next(iter(colors))
        center, rel = _match_center(comp)
        specs.append((color, center, rel))
    assert specs, "expected at least one object"
    return specs


def _paint_object(out: np.ndarray, color: int, center: tuple[int, int], rel: frozenset[tuple[int, int]], turns: int) -> None:
    cr, cc = center
    for dr, dc in rel:
        rr, rc = cr + _rotate_clockwise((dr, dc), turns)[0], cc + _rotate_clockwise((dr, dc), turns)[1]
        assert 0 <= rr < out.shape[0] and 0 <= rc < out.shape[1]
        out[rr, rc] = color
    out[cr, cc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each hollow directional cross by the number of quarter-turns encoded by its color's position in the top legend, then fill the center hole.

    Input structure:
    - Black is the background and the top row contains a left-to-right legend of 3 to 5 distinct single-pixel colors separated by black gaps.
    - Below the legend are several monochrome directional crosses that use legend colors.
    - Each cross is a plus-shape with arms of length 2, its center pixel is black, and exactly one arm has one extra tip pixel that marks the current direction.

    Rule:
    - Read the legend order from left to right and map each legend color to its zero-based index K.
    - For every matching cross, rotate the whole colored shape clockwise by K quarter-turns around its missing center.
    - After rotating, fill the center hole with the same color while keeping the legend row unchanged.

    Key insight:
    - The visible color does not directly encode an angle; the angle comes from the color's position in the edge legend, while the lone longer arm reveals whether the rotation was applied correctly.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    legend = _legend_order(g)
    turns_by_color = {color: idx % 4 for idx, color in enumerate(legend)}
    specs = _object_specs(g)

    out = g.copy()
    rows, cols = np.nonzero(g[1:, :] != BG)
    out[rows + 1, cols] = BG
    for color, center, rel in specs:
        assert color in turns_by_color, "object color must appear in the legend"
        _paint_object(out, color, center, rel, turns_by_color[color])
    return out


def _translated(rel: frozenset[tuple[int, int]], center: tuple[int, int]) -> set[tuple[int, int]]:
    cr, cc = center
    return {(cr + dr, cc + dc) for dr, dc in rel}


def _sample_directions(count: int) -> list[str]:
    perm = list(RNG.permutation(DIRECTIONS))
    chosen = perm[: min(count, len(DIRECTIONS))]
    while len(chosen) < count:
        chosen.append(str(RNG.choice(DIRECTIONS)))
    RNG.shuffle(chosen)
    return chosen


def _place_centers(height: int, width: int, count: int) -> list[tuple[int, int]] | None:
    row_min, row_max = 5, height - 5
    col_min, col_max = 4, width - 5
    if row_min > row_max or col_min > col_max:
        return None

    candidates = [(r, c) for r in range(row_min, row_max + 1) for c in range(col_min, col_max + 1)]
    RNG.shuffle(candidates)

    centers: list[tuple[int, int]] = []
    reserved: set[tuple[int, int]] = set()
    for center in candidates:
        footprint = _translated(RESERVE_REL, center)
        if footprint & reserved:
            continue
        centers.append(center)
        reserved |= footprint
        if len(centers) == count:
            RNG.shuffle(centers)
            return centers
    return None


def _draw_input(height: int, width: int, colors: list[int], centers: list[tuple[int, int]], directions: list[str], start_col: int) -> tuple[np.ndarray, list[tuple[int, tuple[int, int], frozenset[tuple[int, int]]]]]:
    grid = np.zeros((height, width), dtype=int)
    for idx, color in enumerate(colors):
        grid[0, start_col + 2 * idx] = color

    specs: list[tuple[int, tuple[int, int], frozenset[tuple[int, int]]]] = []
    for color, center, direction in zip(colors, centers, directions):
        rel = INPUT_SHAPES[direction]
        for r, c in _translated(rel, center):
            grid[r, c] = color
        specs.append((color, center, rel))
    return grid, specs


def _expected_output(inp: np.ndarray, specs: list[tuple[int, tuple[int, int], frozenset[tuple[int, int]]]], colors: list[int]) -> np.ndarray:
    out = np.zeros_like(inp)
    out[0, :] = inp[0, :]
    turns_by_color = {color: idx % 4 for idx, color in enumerate(colors)}
    for color, center, rel in specs:
        _paint_object(out, color, center, rel, turns_by_color[color])
    return out


def _gen_attempt() -> np.ndarray | None:
    count = int(RNG.integers(3, 6))
    side_low, side_high = GRID_SIDE_RANGE[count]
    height = int(RNG.integers(side_low, side_high))
    width = int(RNG.integers(side_low, side_high))

    legend_span = 2 * count - 1
    if legend_span > width:
        return None
    start_col = int(RNG.integers(0, width - legend_span + 1))

    colors = list(map(int, RNG.choice(np.arange(1, 10), size=count, replace=False)))
    centers = _place_centers(height, width, count)
    if centers is None:
        return None
    directions = _sample_directions(count)

    grid, specs = _draw_input(height, width, colors, centers, directions, start_col)
    expected = _expected_output(grid, specs, colors)

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid directional-cross input")
