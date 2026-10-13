import random
from collections import Counter

import numpy as np


GREY = 5
ARC_COLORS = tuple(range(10))
DIRECTIONS = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
)


def _background_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _is_horizontal_vertical_or_diagonal(dr: int, dc: int) -> bool:
    if dr == 0 and dc == 0:
        return False
    return dr == 0 or dc == 0 or abs(dr) == abs(dc)


def _minimal_period(values: list[int]) -> int:
    n = len(values)
    for period in range(1, n + 1):
        if n % period == 0 and all(values[i] == values[i % period] for i in range(n)):
            return period
    return n


def _recover_palette_line(grid: np.ndarray) -> tuple[int, tuple[int, int], int, list[int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _background_color(g)
    assert bg != GREY, "grey must be a visible anchor, not the background"

    anchors = np.argwhere(g == GREY)
    assert len(anchors) == 1, "expected exactly one grey anchor pixel"
    ar, ac = map(int, anchors[0])

    marker_coords = np.argwhere((g != bg) & (g != GREY))
    assert len(marker_coords) >= 2, "expected at least two colored marker pixels"

    distances = np.maximum(np.abs(marker_coords[:, 0] - ar), np.abs(marker_coords[:, 1] - ac))
    assert np.all(distances > 0), "markers must be away from the anchor"
    order = np.argsort(distances, kind="stable")
    sorted_coords = marker_coords[order]
    sorted_distances = distances[order]
    assert len(set(map(int, sorted_distances))) == len(sorted_distances), (
        "each marker must occupy a distinct ring index"
    )

    first_r, first_c = map(int, sorted_coords[0])
    step_r = first_r - ar
    step_c = first_c - ac
    d = max(abs(step_r), abs(step_c))
    assert d > 0 and _is_horizontal_vertical_or_diagonal(step_r, step_c), (
        "nearest marker must define a horizontal, vertical, or diagonal step"
    )

    marker_set = {tuple(map(int, coord)) for coord in marker_coords}
    expected_positions: list[tuple[int, int]] = []
    for k in range(1, len(marker_coords) + 1):
        rr = ar + k * step_r
        cc = ac + k * step_c
        assert _in_bounds(g.shape, rr, cc), "marker line must stay inside the grid"
        expected_positions.append((rr, cc))

    assert set(expected_positions) == marker_set, (
        "colored markers must be consecutive equally spaced points from the anchor"
    )

    colors = [int(g[r, c]) for r, c in expected_positions]
    assert len(set(colors)) >= 2, "the marker palette must contain at least two colors"
    assert all(color not in (bg, GREY) for color in colors), (
        "marker colors must differ from background and anchor grey"
    )

    return bg, (ar, ac), d, colors


def _paint_concentric_rings(
    grid: np.ndarray,
    bg: int,
    anchor: tuple[int, int],
    spacing: int,
    colors: list[int],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    ar, ac = anchor
    rows, cols = np.indices(out.shape)
    dist = np.maximum(np.abs(rows - ar), np.abs(cols - ac))
    max_k = int(dist.max()) // int(spacing)

    for k in range(1, max_k + 1):
        out[dist == k * spacing] = int(colors[(k - 1) % len(colors)])

    out[ar, ac] = GREY
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Roll an equally spaced colored marker line into concentric square rings around its grey endpoint.

    Input structure:
    - The background is one dominant empty color and grey appears once as a single anchor pixel.
    - From the grey anchor, two or more colored single-pixel markers form one horizontal,
      vertical, or 45-degree diagonal arithmetic line with equal Chebyshev spacing.
    - Marker colors are not all the same; grey is not one of the marker colors.

    Rule:
    - Let d be the Chebyshev distance between adjacent line markers, and read the
      marker colors from the grey anchor outward as colors 1..n.
    - For each k = 1, 2, 3, ... with visible cells in the grid, color every cell
      whose Chebyshev distance from the anchor is exactly k*d with color
      ((k - 1) mod n) + 1 from that line palette.
    - Cells at non-multiple distances stay background, and the grey anchor remains grey.

    Key insight:
    - The straight marker line is a radius-indexed palette: point spacing gives the
      ring step, and the ordered marker colors cycle outward beyond the shown line.
    """
    g = np.asarray(grid, dtype=int)
    bg, anchor, spacing, colors = _recover_palette_line(g)
    return _paint_concentric_rings(g, bg, anchor, spacing, colors)


# --------------- generator ---------------

GEN_RETRIES = 600
GEN_HEIGHT_RANGE = (11, 30)
GEN_WIDTH_RANGE = (11, 30)
GEN_MARKER_COUNT_RANGE = (2, 5)
GEN_SPACING_RANGE = (2, 5)


def _max_visible_ring_index(shape: tuple[int, int], anchor: tuple[int, int], spacing: int) -> int:
    h, w = shape
    r, c = anchor
    max_dist = max(r, h - 1 - r, c, w - 1 - c)
    return max_dist // spacing


def _min_full_ring_index(shape: tuple[int, int], anchor: tuple[int, int], spacing: int) -> int:
    h, w = shape
    r, c = anchor
    max_full = min(r, h - 1 - r, c, w - 1 - c)
    return max_full // spacing


def _visible_ring_mask(shape: tuple[int, int], anchor: tuple[int, int], radius: int) -> np.ndarray:
    rows, cols = np.indices(shape)
    ar, ac = anchor
    return np.maximum(np.abs(rows - ar), np.abs(cols - ac)) == radius


def _choose_colors(count: int, bg: int) -> list[int] | None:
    available = [color for color in ARC_COLORS if color not in (bg, GREY)]
    for _ in range(100):
        if random.random() < 0.65 and count <= len(available):
            colors = random.sample(available, count)
        else:
            palette_size = random.randint(2, min(len(available), max(2, min(4, count))))
            palette = random.sample(available, palette_size)
            colors = [random.choice(palette) for _ in range(count)]
        if len(set(colors)) >= 2 and _minimal_period(colors) == count:
            return colors
    return None


def _candidate_anchors(
    shape: tuple[int, int],
    direction: tuple[int, int],
    spacing: int,
    marker_count: int,
) -> list[tuple[int, int]]:
    h, w = shape
    dr, dc = direction
    anchors: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            end_r = r + marker_count * spacing * dr
            end_c = c + marker_count * spacing * dc
            if not _in_bounds(shape, end_r, end_c):
                continue
            if _max_visible_ring_index(shape, (r, c), spacing) < marker_count + 1:
                continue
            if _min_full_ring_index(shape, (r, c), spacing) < 1:
                continue
            anchors.append((r, c))
    return anchors


def _expected_from_plan(
    shape: tuple[int, int],
    bg: int,
    anchor: tuple[int, int],
    spacing: int,
    colors: list[int],
) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    grid[anchor] = GREY
    return _paint_concentric_rings(grid, bg, anchor, spacing, colors)


def _build_input(
    shape: tuple[int, int],
    bg: int,
    anchor: tuple[int, int],
    direction: tuple[int, int],
    spacing: int,
    colors: list[int],
) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    ar, ac = anchor
    dr, dc = direction
    grid[ar, ac] = GREY
    for k, color in enumerate(colors, start=1):
        grid[ar + k * spacing * dr, ac + k * spacing * dc] = int(color)
    return grid


def _semantic_check(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        bg, anchor, spacing, colors = _recover_palette_line(inp)
        actual = main(inp)
    except AssertionError:
        return False

    if not np.array_equal(actual, expected):
        return False
    if _minimal_period(colors) != len(colors):
        return False
    if _max_visible_ring_index(inp.shape, anchor, spacing) < len(colors) + 1:
        return False

    extrapolated_radius = (len(colors) + 1) * spacing
    extrapolated = _visible_ring_mask(inp.shape, anchor, extrapolated_radius)
    if not np.any(extrapolated & (expected == colors[0])):
        return False

    rows, cols = np.indices(inp.shape)
    ar, ac = anchor
    dist = np.maximum(np.abs(rows - ar), np.abs(cols - ac))
    between = (dist > 0) & (dist % spacing != 0)
    if not np.any(between & (expected == bg)):
        return False

    changed = np.count_nonzero(inp != expected)
    return changed >= max(8, len(colors) * 3)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GEN_HEIGHT_RANGE)
    w = random.randint(*GEN_WIDTH_RANGE)
    shape = (h, w)
    bg = random.choice([color for color in ARC_COLORS if color != GREY])
    marker_count = random.randint(*GEN_MARKER_COUNT_RANGE)
    spacing = random.randint(*GEN_SPACING_RANGE)
    direction = random.choice(DIRECTIONS)

    anchors = _candidate_anchors(shape, direction, spacing, marker_count)
    if not anchors:
        return None
    anchor = random.choice(anchors)

    colors = _choose_colors(marker_count, bg)
    if colors is None:
        return None

    inp = _build_input(shape, bg, anchor, direction, spacing, colors)
    expected = _expected_from_plan(shape, bg, anchor, spacing, colors)
    if not _semantic_check(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid concentric ring palette input")
