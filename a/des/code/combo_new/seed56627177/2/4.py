import random
from collections import Counter

import numpy as np


BG = 0
SEPARATOR = 9
COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8]
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
AXES = ("horizontal", "vertical", "diag_main", "diag_anti")

GRID_COUNT_RANGE = (2, 4)
CELL_SIZE_RANGE = (5, 9)
PATTERN_SIZE_MIN = 3
PATTERN_EXTRA_MAX = 2
REGION_ATTEMPTS = 80
GENERATION_ATTEMPTS = 300


def _neighbors4(r: int, c: int, height: int, width: int):
    for dr, dc in DIRS4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < height and 0 <= cc < width:
            yield rr, cc


def _component_count(mask: np.ndarray) -> int:
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    count = 0
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                for nr, nc in _neighbors4(rr, cc, height, width):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
    return count


def _spans(length: int, cuts: list[int]) -> list[tuple[int, int]]:
    bounds = [-1] + sorted(int(x) for x in cuts) + [int(length)]
    spans = []
    for lo, hi in zip(bounds, bounds[1:]):
        start = lo + 1
        end = hi
        if start < end:
            spans.append((start, end))
    return spans


def _find_separator_color(grid: np.ndarray) -> int:
    g = np.asarray(grid, dtype=int)
    candidates = []
    for color in map(int, np.unique(g)):
        row_mask = np.all(g == color, axis=1)
        col_mask = np.all(g == color, axis=0)
        if not row_mask.any() or not col_mask.any():
            continue
        expected = row_mask[:, None] | col_mask[None, :]
        if np.array_equal(g == color, expected):
            score = (int(row_mask.sum() + col_mask.sum()), int(np.count_nonzero(g == color)))
            candidates.append((score, color))
    assert candidates, "expected a separator color forming full rows and columns"
    candidates.sort()
    return int(candidates[-1][1])


def _reflect_point(r: int, c: int, height: int, width: int, axis: str) -> tuple[int, int]:
    if axis == "horizontal":
        return height - 1 - r, c
    if axis == "vertical":
        return r, width - 1 - c
    if axis == "diag_main":
        return c, r
    if axis == "diag_anti":
        return width - 1 - c, height - 1 - r
    raise AssertionError(f"unknown axis: {axis}")


def _infer_axis(markers: list[tuple[int, int]], height: int, width: int) -> str:
    assert len(markers) == 2, "expected exactly two markers"
    (r1, c1), (r2, c2) = sorted((tuple(map(int, p)) for p in markers))
    assert r1 + r2 == height - 1 and c1 + c2 == width - 1, "markers must be centrally symmetric"

    dr = r2 - r1
    dc = c2 - c1
    if r1 == r2:
        assert height % 2 == 1 and 2 * r1 == height - 1
        return "horizontal"
    if c1 == c2:
        assert width % 2 == 1 and 2 * c1 == width - 1
        return "vertical"

    assert height == width, "diagonal reflections require square regions"
    if dr == dc:
        return "diag_main"
    if dr == -dc:
        return "diag_anti"
    raise AssertionError("marker pair does not encode a supported axis")


def _solve_region(region: np.ndarray, bg: int) -> np.ndarray:
    region = np.asarray(region, dtype=int)
    height, width = region.shape
    colors = [int(color) for color in np.unique(region) if int(color) != int(bg)]
    assert len(colors) == 2, "each region must contain one payload color and one marker color"

    counts = {color: int(np.count_nonzero(region == color)) for color in colors}
    ordered = sorted(colors, key=lambda color: (counts[color], color))
    marker_color, payload_color = ordered
    assert counts[marker_color] == 2, "marker color must occupy exactly two cells"
    assert counts[payload_color] >= PATTERN_SIZE_MIN, "payload must occupy multiple cells"

    marker_mask = region == marker_color
    payload_mask = region == payload_color
    assert _component_count(marker_mask) == 2, "markers must be isolated singletons"
    assert _component_count(payload_mask) == 1, "payload must be one connected component"

    markers = [tuple(map(int, rc)) for rc in np.argwhere(marker_mask)]
    axis = _infer_axis(markers, height, width)
    marker_set = set(markers)

    out = np.full(region.shape, int(bg), dtype=int)
    out[marker_mask] = int(marker_color)
    for r, c in map(tuple, np.argwhere(payload_mask)):
        rr, cc = _reflect_point(int(r), int(c), height, width, axis)
        assert (rr, cc) not in marker_set, "reflected payload may not overwrite markers"
        out[rr, cc] = int(payload_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each region's connected payload using the axis encoded by its centrally symmetric marker pair.

    Input structure:
    - The background is black and a maroon color forms full separator rows and columns that partition the grid into rectangular regions.
    - Each region contains exactly one connected multi-pixel payload in one color plus exactly two single-pixel markers in a second color.
    - The two markers are centrally symmetric within their region, so their connecting direction is horizontal, vertical, or one of the two diagonals through the region center.

    Rule:
    - Detect the separator rows and columns, then process every rectangular region independently.
    - In each region, treat the rarer foreground color as the two markers and the other foreground color as the payload.
    - The marker pair direction selects the reflection axis of the payload: horizontal pair -> horizontal axis, vertical pair -> vertical axis, and diagonal pair -> the matching diagonal.
    - Replace the original payload by its mirrored copy while leaving the markers, separators, and background unchanged.

    Key insight: The two single markers are not targets themselves; their centrally symmetric placement encodes which center-crossing axis should be used to reflect the payload inside that region.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    separator = _find_separator_color(g)
    counts = Counter(map(int, g[g != separator].ravel()))
    assert counts, "grid must contain non-separator cells"
    bg = int(counts.most_common(1)[0][0])

    sep_rows = [int(r) for r in range(g.shape[0]) if np.all(g[r, :] == separator)]
    sep_cols = [int(c) for c in range(g.shape[1]) if np.all(g[:, c] == separator)]
    row_spans = _spans(g.shape[0], sep_rows)
    col_spans = _spans(g.shape[1], sep_cols)
    assert row_spans and col_spans, "expected at least one region in each direction"

    out = g.copy()
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            out[r0:r1, c0:c1] = _solve_region(g[r0:r1, c0:c1], bg)
    return out


def _candidate_cell_sizes(n_rows: int, n_cols: int) -> list[int]:
    max_size = min(
        (30 - (n_rows - 1)) // n_rows,
        (30 - (n_cols - 1)) // n_cols,
        CELL_SIZE_RANGE[1],
    )
    return [size for size in range(CELL_SIZE_RANGE[0], max_size + 1) if size % 2 == 1]


def _marker_positions(size: int, axis: str) -> list[tuple[int, int]]:
    center = size // 2
    delta = random.randint(1, center)
    if axis == "horizontal":
        return [(center, center - delta), (center, center + delta)]
    if axis == "vertical":
        return [(center - delta, center), (center + delta, center)]
    if axis == "diag_main":
        return [(center - delta, center - delta), (center + delta, center + delta)]
    if axis == "diag_anti":
        return [(center - delta, center + delta), (center + delta, center - delta)]
    raise AssertionError(f"unknown axis: {axis}")


def _allowed_half(size: int, axis: str, side: int) -> np.ndarray:
    rr, cc = np.indices((size, size))
    center = size // 2
    if axis == "horizontal":
        return rr < center if side < 0 else rr > center
    if axis == "vertical":
        return cc < center if side < 0 else cc > center
    if axis == "diag_main":
        return rr < cc if side < 0 else rr > cc
    if axis == "diag_anti":
        return rr + cc < size - 1 if side < 0 else rr + cc > size - 1
    raise AssertionError(f"unknown axis: {axis}")


def _grow_connected(allowed: np.ndarray, target_size: int) -> set[tuple[int, int]] | None:
    height, width = allowed.shape
    available = [tuple(map(int, rc)) for rc in np.argwhere(allowed)]
    if len(available) < target_size:
        return None

    seed = random.choice(available)
    cells = {seed}
    frontier = {nb for nb in _neighbors4(seed[0], seed[1], height, width) if allowed[nb]}

    while len(cells) < target_size:
        if not frontier:
            return None
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        for nb in _neighbors4(cell[0], cell[1], height, width):
            if allowed[nb] and nb not in cells:
                frontier.add(nb)
    return cells


def _build_region(size: int, axis: str) -> tuple[np.ndarray, np.ndarray] | None:
    markers = _marker_positions(size, axis)
    marker_set = set(markers)
    side = random.choice((-1, 1))
    allowed = _allowed_half(size, axis, side)
    allowed_count = int(np.count_nonzero(allowed))
    max_payload = min(allowed_count, size + PATTERN_EXTRA_MAX)
    if max_payload < PATTERN_SIZE_MIN:
        return None

    for _ in range(REGION_ATTEMPTS):
        payload_size = random.randint(PATTERN_SIZE_MIN, max_payload)
        payload = _grow_connected(allowed, payload_size)
        if payload is None:
            continue

        reflected = {_reflect_point(r, c, size, size, axis) for r, c in payload}
        if payload == reflected:
            continue
        if reflected & marker_set:
            continue

        payload_color, marker_color = random.sample(COLOR_POOL, 2)
        inp = np.full((size, size), BG, dtype=int)
        out = np.full((size, size), BG, dtype=int)
        for r, c in payload:
            inp[r, c] = int(payload_color)
        for r, c in reflected:
            out[r, c] = int(payload_color)
        for r, c in markers:
            inp[r, c] = int(marker_color)
            out[r, c] = int(marker_color)

        if _component_count(inp == payload_color) != 1:
            continue
        if _component_count(out == payload_color) != 1:
            continue
        if np.array_equal(inp, out):
            continue
        return inp, out
    return None


def _axis_schedule(count: int) -> list[str]:
    axes = list(AXES)
    while len(axes) < count:
        axes.append(random.choice(AXES))
    random.shuffle(axes)
    return axes[:count]


def _layout_origins(n_rows: int, n_cols: int, size: int) -> list[tuple[int, int]]:
    return [
        (row * (size + 1), col * (size + 1))
        for row in range(n_rows)
        for col in range(n_cols)
    ]


def _gen_attempt() -> np.ndarray | None:
    n_rows = random.randint(*GRID_COUNT_RANGE)
    n_cols = random.randint(*GRID_COUNT_RANGE)
    candidate_sizes = _candidate_cell_sizes(n_rows, n_cols)
    if not candidate_sizes:
        return None
    size = random.choice(candidate_sizes)

    height = n_rows * size + (n_rows - 1)
    width = n_cols * size + (n_cols - 1)
    inp = np.full((height, width), BG, dtype=int)
    out = np.full((height, width), BG, dtype=int)

    for r in range(size, height, size + 1):
        inp[r, :] = SEPARATOR
        out[r, :] = SEPARATOR
    for c in range(size, width, size + 1):
        inp[:, c] = SEPARATOR
        out[:, c] = SEPARATOR

    axes = _axis_schedule(n_rows * n_cols)
    for (r0, c0), axis in zip(_layout_origins(n_rows, n_cols, size), axes):
        built = _build_region(size, axis)
        if built is None:
            return None
        region_inp, region_out = built
        inp[r0 : r0 + size, c0 : c0 + size] = region_inp
        out[r0 : r0 + size, c0 : c0 + size] = region_out

    if np.array_equal(inp, out):
        return None
    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
