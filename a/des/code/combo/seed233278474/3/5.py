import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
MIN_BARRIERS = 1
MAX_BARRIERS = 4
MIN_LONG_SIDE = 8
MAX_LONG_SIDE = 18
MIN_SHORT_SIDE = 8
MAX_SHORT_SIDE = 18
MIN_BAND_SIZE = 2
MAX_RETRIES = 300


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _segment(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    if a[0] == b[0]:
        step = 1 if b[1] >= a[1] else -1
        return [(int(a[0]), int(c)) for c in range(int(a[1]), int(b[1]) + step, step)]
    if a[1] == b[1]:
        step = 1 if b[0] >= a[0] else -1
        return [(int(r), int(a[1])) for r in range(int(a[0]), int(b[0]) + step, step)]
    raise AssertionError("segment endpoints must share a row or a column")


def _find_barriers(grid: np.ndarray, bg: int) -> list[dict[str, int | str]]:
    h, w = grid.shape
    barriers: list[dict[str, int | str]] = []

    for r in range(h):
        row = grid[r]
        if int(np.count_nonzero(row == bg)) != 1:
            continue
        non_bg = row[row != bg]
        if non_bg.size == w - 1 and bool(np.all(non_bg == non_bg[0])):
            gap = int(np.flatnonzero(row == bg)[0])
            barriers.append({"axis": "h", "line": int(r), "gap": gap, "color": int(non_bg[0])})

    for c in range(w):
        col = grid[:, c]
        if int(np.count_nonzero(col == bg)) != 1:
            continue
        non_bg = col[col != bg]
        if non_bg.size == h - 1 and bool(np.all(non_bg == non_bg[0])):
            gap = int(np.flatnonzero(col == bg)[0])
            barriers.append({"axis": "v", "line": int(c), "gap": gap, "color": int(non_bg[0])})

    assert barriers, "expected at least one barrier"
    colors = [int(barrier["color"]) for barrier in barriers]
    assert len(colors) == len(set(colors)), "barrier colors must be unique"
    return barriers


def _find_pairs(
    grid: np.ndarray, bg: int, barrier_colors: set[int]
) -> list[tuple[int, tuple[int, int], tuple[int, int]]]:
    pairs: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    for color in np.unique(grid):
        col = int(color)
        if col == bg or col in barrier_colors:
            continue
        coords = np.argwhere(grid == col)
        assert coords.shape[0] == 2, "each endpoint color must appear exactly twice"
        p1 = (int(coords[0, 0]), int(coords[0, 1]))
        p2 = (int(coords[1, 0]), int(coords[1, 1]))
        pairs.append((col, p1, p2))
    assert pairs, "expected at least one endpoint pair"
    return pairs


def _separates(barrier: dict[str, int | str], p1: tuple[int, int], p2: tuple[int, int]) -> bool:
    if barrier["axis"] == "h":
        line = int(barrier["line"])
        return (p1[0] < line < p2[0]) or (p2[0] < line < p1[0])
    line = int(barrier["line"])
    return (p1[1] < line < p2[1]) or (p2[1] < line < p1[1])


def _path_via_gap(
    barrier: dict[str, int | str], p1: tuple[int, int], p2: tuple[int, int]
) -> list[tuple[int, int]]:
    if barrier["axis"] == "h":
        gap_col = int(barrier["gap"])
        align1 = p1[1] == gap_col
        align2 = p2[1] == gap_col
        assert align1 or align2, "one endpoint must align with the horizontal barrier gap"
        if align1 and align2:
            path = _segment(p1, p2)
        elif align1:
            bend = (int(p2[0]), gap_col)
            path = _segment(p1, bend) + _segment(bend, p2)[1:]
        else:
            bend = (int(p1[0]), gap_col)
            path = _segment(p1, bend) + _segment(bend, p2)[1:]
        assert (int(barrier["line"]), gap_col) in path
        return path

    gap_row = int(barrier["gap"])
    align1 = p1[0] == gap_row
    align2 = p2[0] == gap_row
    assert align1 or align2, "one endpoint must align with the vertical barrier gap"
    if align1 and align2:
        path = _segment(p1, p2)
    elif align1:
        bend = (gap_row, int(p2[1]))
        path = _segment(p1, bend) + _segment(bend, p2)[1:]
    else:
        bend = (gap_row, int(p1[1]))
        path = _segment(p1, bend) + _segment(bend, p2)[1:]
    assert (gap_row, int(barrier["line"])) in path
    return path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-color endpoint pair through the unique barrier gap that separates it.

    The new path is drawn in the separating barrier's color, while the original endpoints
    keep their own color and existing barrier cells stay unchanged.
    """
    g = np.asarray(grid, dtype=int)
    bg = _dominant_color(g)

    barriers = _find_barriers(g, bg)
    barrier_colors = {int(barrier["color"]) for barrier in barriers}
    pairs = _find_pairs(g, bg, barrier_colors)

    out = g.copy()
    for _color, p1, p2 in pairs:
        candidates = [barrier for barrier in barriers if _separates(barrier, p1, p2)]
        assert len(candidates) == 1, "each pair must be separated by exactly one barrier"
        barrier = candidates[0]
        path = _path_via_gap(barrier, p1, p2)
        for cell in path[1:-1]:
            assert g[cell] == bg, "path interior must be empty in the input"
            if out[cell] != g[cell]:
                assert out[cell] == int(barrier["color"]), "different paths must not conflict"
            out[cell] = int(barrier["color"])

    return out


def _partition_with_min(total: int, parts: int, minimum: int, rng: np.random.Generator) -> list[int]:
    assert total >= parts * minimum
    values = [minimum] * parts
    slack = total - parts * minimum
    for _ in range(slack):
        values[int(rng.integers(0, parts))] += 1
    return values


def _build_bands(total_span: int, barrier_count: int, rng: np.random.Generator) -> tuple[list[tuple[int, int]], list[int]]:
    band_total = total_span - barrier_count
    band_sizes = _partition_with_min(band_total, barrier_count + 1, MIN_BAND_SIZE, rng)
    bands: list[tuple[int, int]] = []
    lines: list[int] = []
    cursor = 0
    for i, size in enumerate(band_sizes):
        start = cursor
        end = cursor + size - 1
        bands.append((int(start), int(end)))
        cursor = end + 1
        if i < barrier_count:
            lines.append(int(cursor))
            cursor += 1
    assert cursor == total_span
    return bands, lines


def _pick_other_coord(length: int, gap: int, rng: np.random.Generator) -> int:
    choices = [idx for idx in range(length) if idx != gap]
    return int(choices[int(rng.integers(0, len(choices)))])


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    orientation = "h" if float(rng.random()) < 0.5 else "v"
    barrier_count = int(rng.integers(MIN_BARRIERS, MAX_BARRIERS + 1))

    if orientation == "h":
        height = int(rng.integers(3 * barrier_count + 2, MAX_LONG_SIDE + 1))
        width = int(rng.integers(MIN_SHORT_SIDE, MAX_SHORT_SIDE + 1))
        bands, lines = _build_bands(height, barrier_count, rng)
    else:
        height = int(rng.integers(MIN_SHORT_SIDE, MAX_SHORT_SIDE + 1))
        width = int(rng.integers(3 * barrier_count + 2, MAX_LONG_SIDE + 1))
        bands, lines = _build_bands(width, barrier_count, rng)

    colors = rng.choice(np.array(PALETTE, dtype=int), size=2 * barrier_count, replace=False)
    barrier_colors = [int(color) for color in colors[:barrier_count]]
    pair_colors = [int(color) for color in colors[barrier_count:]]

    grid = np.full((height, width), BG, dtype=int)
    barriers: list[dict[str, int | str]] = []
    for line, color in zip(lines, barrier_colors):
        if orientation == "h":
            gap = int(rng.integers(1, width - 1))
            grid[int(line), :] = color
            grid[int(line), gap] = BG
            barriers.append({"axis": "h", "line": int(line), "gap": gap, "color": int(color)})
        else:
            gap = int(rng.integers(1, height - 1))
            grid[:, int(line)] = color
            grid[gap, int(line)] = BG
            barriers.append({"axis": "v", "line": int(line), "gap": gap, "color": int(color)})

    for i, barrier in enumerate(barriers):
        color = pair_colors[i]
        if orientation == "h":
            upper_band = bands[i]
            lower_band = bands[i + 1]
            r_top = int(upper_band[1])
            r_bottom = int(lower_band[0])
            gap = int(barrier["gap"])
            if float(rng.random()) < 0.5:
                p1 = (r_top, gap)
                p2 = (r_bottom, _pick_other_coord(width, gap, rng))
            else:
                p1 = (r_top, _pick_other_coord(width, gap, rng))
                p2 = (r_bottom, gap)
        else:
            left_band = bands[i]
            right_band = bands[i + 1]
            c_left = int(left_band[1])
            c_right = int(right_band[0])
            gap = int(barrier["gap"])
            if float(rng.random()) < 0.5:
                p1 = (gap, c_left)
                p2 = (_pick_other_coord(height, gap, rng), c_right)
            else:
                p1 = (_pick_other_coord(height, gap, rng), c_left)
                p2 = (gap, c_right)

        if grid[p1] != BG or grid[p2] != BG:
            return None
        grid[p1] = color
        grid[p2] = color

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
