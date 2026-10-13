import numpy as np


BLACK = 0
COLOR_POOL = tuple(range(1, 10))
GENERATION_ATTEMPTS = 400
LINE_GAP = 4
GRID_SIZE_RANGE = (14, 23)
LINE_COUNT_RANGE = (1, 3)


def _background_color(grid: np.ndarray) -> int:
    if np.any(grid == BLACK):
        return BLACK
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _legend_candidates(grid: np.ndarray, bg: int) -> list[tuple[list[tuple[int, int]], list[int]]]:
    h, w = grid.shape
    candidates: list[tuple[list[tuple[int, int]], list[int]]] = []

    def add_if_valid(coords: list[tuple[int, int]]) -> None:
        colors = [int(grid[r, c]) for r, c in coords]
        if not (2 <= len(colors) <= 4):
            return
        if any(color == bg for color in colors):
            return
        if len(set(colors)) != len(colors):
            return
        candidates.append((coords, colors))

    for r in (0, h - 1):
        c = 0
        while c < w:
            if int(grid[r, c]) == bg:
                c += 1
                continue
            c0 = c
            while c < w and int(grid[r, c]) != bg:
                c += 1
            c1 = c - 1
            inward_r = 1 if r == 0 else h - 2
            if c0 > 0 and int(grid[r, c0 - 1]) != bg:
                continue
            if c1 + 1 < w and int(grid[r, c1 + 1]) != bg:
                continue
            if not np.all(grid[inward_r, c0 : c1 + 1] == bg):
                continue
            add_if_valid([(r, cc) for cc in range(c0, c1 + 1)])

    for c in (0, w - 1):
        r = 0
        while r < h:
            if int(grid[r, c]) == bg:
                r += 1
                continue
            r0 = r
            while r < h and int(grid[r, c]) != bg:
                r += 1
            r1 = r - 1
            inward_c = 1 if c == 0 else w - 2
            if r0 > 0 and int(grid[r0 - 1, c]) != bg:
                continue
            if r1 + 1 < h and int(grid[r1 + 1, c]) != bg:
                continue
            if not np.all(grid[r0 : r1 + 1, inward_c] == bg):
                continue
            add_if_valid([(rr, c) for rr in range(r0, r1 + 1)])

    return candidates


def _extract_legend(grid: np.ndarray, bg: int) -> tuple[dict[int, int], set[tuple[int, int]]]:
    candidates = _legend_candidates(grid, bg)
    assert len(candidates) == 1, "expected exactly one edge legend"
    coords, colors = candidates[0]
    return {color: idx for idx, color in enumerate(colors)}, set(coords)


def _crossings(
    grid: np.ndarray, bg: int, legend_coords: set[tuple[int, int]], rank: dict[int, int]
) -> list[tuple[int, int, int, int, int]]:
    h, w = grid.shape
    found: list[tuple[int, int, int, int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if (r, c) in legend_coords:
                continue
            marker = int(grid[r, c])
            if marker == bg or marker not in rank:
                continue

            nbrs = [(r, c - 1), (r, c + 1), (r - 1, c), (r + 1, c)]
            if any(pos in legend_coords for pos in nbrs):
                continue
            left, right, up, down = (int(grid[p]) for p in nbrs)
            if bg in (left, right, up, down):
                continue
            if left != right or up != down:
                continue
            if left == up:
                continue
            if marker in (left, up):
                continue
            if left not in rank or up not in rank:
                continue
            found.append((r, c, marker, left, up))

    assert found, "expected at least one crossing"
    return found


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _background_color(g)
    rank, legend_coords = _extract_legend(g, bg)
    crossings = _crossings(g, bg, legend_coords, rank)

    out = g.copy()
    for r, c, marker, horiz, vert in crossings:
        horiz_dist = abs(rank[marker] - rank[horiz])
        vert_dist = abs(rank[marker] - rank[vert])
        assert horiz_dist != vert_dist, "legend must resolve the overpass unambiguously"

        if horiz_dist < vert_dist:
            out[r, c] = horiz
            out[r - 1, c] = bg
            out[r + 1, c] = bg
        else:
            out[r, c] = vert
            out[r, c - 1] = bg
            out[r, c + 1] = bg

    return out


def _sample_spaced_positions(
    rng: np.random.Generator, start: int, stop: int, count: int, min_gap: int
) -> list[int] | None:
    values = np.arange(start, stop + 1)
    if count > len(values):
        return None
    for _ in range(300):
        picks = np.sort(rng.choice(values, size=count, replace=False))
        if np.all(np.diff(picks) >= min_gap):
            return [int(v) for v in picks]
    return None


def _max_spaced_count(start: int, stop: int, min_gap: int) -> int:
    length = stop - start + 1
    return 0 if length <= 0 else 1 + (length - 1) // min_gap


def _choose_line_count(rng: np.random.Generator, max_count: int) -> int:
    hi = min(LINE_COUNT_RANGE[1], max_count)
    assert hi >= 1
    lo = LINE_COUNT_RANGE[0]
    if hi >= 2 and rng.random() < 0.7:
        lo = 2
    return int(rng.integers(lo, hi + 1))


def _place_legend(grid: np.ndarray, legend: list[int], side: str, rng: np.random.Generator) -> None:
    h, w = grid.shape
    if side == "top":
        start = int(rng.integers(1, w - len(legend)))
        grid[0, start : start + len(legend)] = legend
        return
    if side == "bottom":
        start = int(rng.integers(1, w - len(legend)))
        grid[h - 1, start : start + len(legend)] = legend
        return
    if side == "left":
        start = int(rng.integers(1, h - len(legend)))
        grid[start : start + len(legend), 0] = legend
        return
    assert side == "right"
    start = int(rng.integers(1, h - len(legend)))
    grid[start : start + len(legend), w - 1] = legend


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(GRID_SIZE_RANGE[0], GRID_SIZE_RANGE[1]))
    w = int(rng.integers(GRID_SIZE_RANGE[0], GRID_SIZE_RANGE[1]))
    grid = np.full((h, w), BLACK, dtype=int)

    row_min, row_max = 2, h - 3
    col_min, col_max = 2, w - 3
    max_h = min(LINE_COUNT_RANGE[1], _max_spaced_count(row_min, row_max, LINE_GAP))
    max_v = min(LINE_COUNT_RANGE[1], _max_spaced_count(col_min, col_max, LINE_GAP))
    if max_h < 1 or max_v < 1:
        return None

    horiz_count = _choose_line_count(rng, max_h)
    vert_count = _choose_line_count(rng, max_v)
    rows = _sample_spaced_positions(rng, row_min, row_max, horiz_count, LINE_GAP)
    cols = _sample_spaced_positions(rng, col_min, col_max, vert_count, LINE_GAP)
    if rows is None or cols is None:
        return None

    legend = list(map(int, rng.choice(COLOR_POOL, size=4, replace=False)))
    if rng.random() < 0.5:
        legend.reverse()

    end_a, marker_a, marker_b, end_b = legend
    if rng.random() < 0.5:
        horiz_color, vert_color = end_a, end_b
        horiz_marker, vert_marker = marker_a, marker_b
    else:
        horiz_color, vert_color = end_b, end_a
        horiz_marker, vert_marker = marker_b, marker_a

    side = str(rng.choice(np.array(["top", "bottom", "left", "right"], dtype=object)))
    _place_legend(grid, legend, side, rng)

    for r in rows:
        grid[r, 1 : w - 1] = horiz_color
    for c in cols:
        grid[1 : h - 1, c] = vert_color

    top_is_horizontal = rng.random((len(rows), len(cols))) < 0.5
    if top_is_horizontal.size > 1 and (top_is_horizontal.all() or (~top_is_horizontal).all()):
        rr = int(rng.integers(0, len(rows)))
        cc = int(rng.integers(0, len(cols)))
        top_is_horizontal[rr, cc] = not top_is_horizontal[rr, cc]

    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            grid[r, c] = horiz_marker if top_is_horizontal[i, j] else vert_marker

    try:
        out = main(grid)
    except AssertionError:
        return None

    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            if top_is_horizontal[i, j]:
                if int(out[r, c]) != horiz_color:
                    return None
                if int(out[r - 1, c]) != BLACK or int(out[r + 1, c]) != BLACK:
                    return None
                if int(out[r, c - 1]) != horiz_color or int(out[r, c + 1]) != horiz_color:
                    return None
            else:
                if int(out[r, c]) != vert_color:
                    return None
                if int(out[r, c - 1]) != BLACK or int(out[r, c + 1]) != BLACK:
                    return None
                if int(out[r - 1, c]) != vert_color or int(out[r + 1, c]) != vert_color:
                    return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid overpass input")
