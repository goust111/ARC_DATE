import random
from collections import defaultdict

import numpy as np


BG = 0
COLOR_POOL = (1, 2, 3, 4, 5, 6, 7, 8, 9)
GEN_RETRIES = 400


def _legend_info(grid: np.ndarray) -> tuple[list[int], dict[int, int], np.ndarray]:
    h, _ = grid.shape
    rows = [r for r in range(h) if int(grid[r, 0]) != BG]
    assert rows, "legend column must contain at least one color"
    colors = [int(grid[r, 0]) for r in rows]
    assert len(colors) == len(set(colors)), "legend colors must be unique"

    legend_mask = np.zeros(grid.shape, dtype=bool)
    for r in rows:
        legend_mask[r, 0] = True

    rank = {color: i for i, color in enumerate(colors)}
    return colors, rank, legend_mask


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _crossing_data(
    grid: np.ndarray, rank: dict[int, int], legend_mask: np.ndarray
) -> tuple[defaultdict[tuple[int, int], set[int]], list[tuple[int, int, int, str]]]:
    h, w = grid.shape
    owners: defaultdict[tuple[int, int], set[int]] = defaultdict(set)
    for r, c in np.argwhere(grid != BG):
        owners[(int(r), int(c))].add(int(grid[r, c]))

    crossings: list[tuple[int, int, int, str]] = []
    specs = (
        ("h", ((0, -1), (0, 1))),
        ("v", ((-1, 0), (1, 0))),
    )

    for r in range(h):
        for c in range(w):
            if c == 0 or legend_mask[r, c]:
                continue
            top = int(grid[r, c])
            if top == BG:
                continue

            for axis, deltas in specs:
                (dr1, dc1), (dr2, dc2) = deltas
                r1, c1 = r + dr1, c + dc1
                r2, c2 = r + dr2, c + dc2
                if not _in_bounds(grid.shape, r1, c1) or not _in_bounds(grid.shape, r2, c2):
                    continue
                if legend_mask[r1, c1] or legend_mask[r2, c2]:
                    continue

                v1 = int(grid[r1, c1])
                v2 = int(grid[r2, c2])
                if v1 == BG or v1 != v2 or v1 == top:
                    continue
                if top not in rank or v1 not in rank:
                    continue
                if rank[top] >= rank[v1]:
                    continue

                owners[(r, c)].add(v1)
                crossings.append((r, c, v1, axis))

    return owners, crossings


def main(grid: np.ndarray) -> np.ndarray:
    """
    Break the lower-priority path around each over-under crossing.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    _, rank, legend_mask = _legend_info(g)
    off_legend = g[(g != BG) & (~legend_mask)]
    assert all(int(v) in rank for v in off_legend), "every path color must appear in the legend"

    owners, crossings = _crossing_data(g, rank, legend_mask)
    out = g.copy()

    for r, c, low_color, axis in crossings:
        deltas = ((0, -1), (0, 1)) if axis == "h" else ((-1, 0), (1, 0))
        for dr, dc in deltas:
            nr, nc = r + dr, c + dc
            if not _in_bounds(g.shape, nr, nc):
                continue
            if legend_mask[nr, nc]:
                continue
            if int(g[nr, nc]) != low_color:
                continue
            if len(owners[(nr, nc)]) > 1:
                continue
            out[nr, nc] = BG

    return out


def _path_cells(path: dict[str, int | str]) -> list[tuple[int, int]]:
    axis = str(path["axis"])
    fixed = int(path["fixed"])
    start = int(path["start"])
    end = int(path["end"])
    if axis == "h":
        return [(fixed, c) for c in range(start, end + 1)]
    return [(r, fixed) for r in range(start, end + 1)]


def _pick_spaced(
    count: int,
    lo: int,
    hi: int,
    min_gap: int,
    rng: random.Random,
    blocked: list[int] | tuple[int, ...] = (),
) -> list[int] | None:
    if count == 0:
        return []
    values = list(range(lo, hi + 1))
    if len(values) < count:
        return None

    blocked = list(blocked)
    for _ in range(250):
        rng.shuffle(values)
        picked: list[int] = []
        for value in values:
            if any(abs(value - other) < min_gap for other in blocked):
                continue
            if any(abs(value - other) < min_gap for other in picked):
                continue
            picked.append(value)
            if len(picked) == count:
                return sorted(picked)
    return None


def _render_scene(
    height: int,
    width: int,
    priority_colors: list[int],
    paths: list[dict[str, int | str]],
    rng: random.Random,
) -> tuple[np.ndarray, defaultdict[tuple[int, int], set[int]], dict[int, dict[str, int | str]]]:
    grid = np.zeros((height, width), dtype=int)

    legend_span = 2 * len(priority_colors) - 1
    legend_top = rng.randint(0, height - legend_span)
    for i, color in enumerate(priority_colors):
        grid[legend_top + 2 * i, 0] = int(color)

    occupancy: defaultdict[tuple[int, int], set[int]] = defaultdict(set)
    for path in paths:
        color = int(path["color"])
        for cell in _path_cells(path):
            occupancy[cell].add(color)

    rank = {color: i for i, color in enumerate(priority_colors)}
    for (r, c), colors in occupancy.items():
        grid[r, c] = min(colors, key=lambda color: rank[color])

    by_color = {int(path["color"]): path for path in paths}
    return grid, occupancy, by_color


def _expected_output(
    grid: np.ndarray,
    occupancy: defaultdict[tuple[int, int], set[int]],
    by_color: dict[int, dict[str, int | str]],
    priority_colors: list[int],
) -> np.ndarray:
    rank = {color: i for i, color in enumerate(priority_colors)}
    out = grid.copy()

    for (r, c), colors in occupancy.items():
        if len(colors) != 2:
            continue
        high, low = sorted(colors, key=lambda color: rank[color])
        _ = high
        axis = str(by_color[low]["axis"])
        deltas = ((0, -1), (0, 1)) if axis == "h" else ((-1, 0), (1, 0))
        for dr, dc in deltas:
            nbr = (r + dr, c + dc)
            if occupancy.get(nbr) == {low}:
                out[nbr] = BG

    return out


def _normal_scene(rng: random.Random) -> tuple[int, int, list[int], list[dict[str, int | str]]] | None:
    layouts = [(1, 2), (2, 1), (2, 2), (1, 3), (3, 1), (2, 3), (3, 2)]
    for _ in range(100):
        n_h, n_v = rng.choice(layouts)
        total = n_h + n_v
        height = rng.randint(max(12, 2 * total + 1), 22)
        width = rng.randint(14, 24)
        rows = _pick_spaced(n_h, 2, height - 3, rng.choice([3, 4]), rng)
        cols = _pick_spaced(n_v, 4, width - 4, rng.choice([3, 4]), rng)
        if rows is None or cols is None:
            continue

        min_row, max_row = min(rows), max(rows)
        min_col, max_col = min(cols), max(cols)
        if min_row <= 1 or max_row >= height - 2 or min_col <= 3 or max_col >= width - 3:
            continue

        priority_colors = rng.sample(list(COLOR_POOL), total)
        path_colors = priority_colors[:]
        rng.shuffle(path_colors)

        paths: list[dict[str, int | str]] = []
        for i, row in enumerate(rows):
            left = rng.randint(2, min_col - 1)
            right = rng.randint(max_col + 1, width - 2)
            paths.append(
                {"color": path_colors[i], "axis": "h", "fixed": row, "start": left, "end": right}
            )
        for j, col in enumerate(cols):
            top = rng.randint(1, min_row - 1)
            bottom = rng.randint(max_row + 1, height - 2)
            paths.append(
                {"color": path_colors[n_h + j], "axis": "v", "fixed": col, "start": top, "end": bottom}
            )
        return height, width, priority_colors, paths
    return None


def _protected_horizontal_scene(
    rng: random.Random,
) -> tuple[int, int, list[int], list[dict[str, int | str]]] | None:
    for _ in range(100):
        extra_h = rng.randint(0, 2)
        extra_v = rng.randint(0, 1)
        total = 3 + extra_h + extra_v
        height = rng.randint(max(12, 2 * total + 1), 22)
        width = rng.randint(15, 24)

        row = rng.randint(2, height - 3)
        col = rng.randint(5, width - 5)
        extra_rows = _pick_spaced(extra_h, 2, height - 3, 3, rng, blocked=[row])
        extra_cols = _pick_spaced(extra_v, 4, width - 4, 3, rng, blocked=[col, col + 1])
        if extra_rows is None or extra_cols is None:
            continue

        rows = [row] + extra_rows
        cols = [col, col + 1] + extra_cols
        min_row, max_row = min(rows), max(rows)
        min_col, max_col = min(cols), max(cols)
        if min_row <= 1 or max_row >= height - 2 or min_col <= 3 or max_col >= width - 3:
            continue

        priority_colors = rng.sample(list(COLOR_POOL), total)
        high_color, mid_color, low_color = priority_colors[:3]
        extra_colors = priority_colors[3:]
        high_col, low_col = (col, col + 1) if rng.random() < 0.5 else (col + 1, col)

        paths: list[dict[str, int | str]] = []
        main_left = rng.randint(2, min_col - 1)
        main_right = rng.randint(max_col + 1, width - 2)
        paths.append({"color": mid_color, "axis": "h", "fixed": row, "start": main_left, "end": main_right})

        extra_h_colors = extra_colors[:extra_h]
        extra_v_colors = extra_colors[extra_h:]
        for extra_row, color in zip(extra_rows, extra_h_colors):
            left = rng.randint(2, min_col - 1)
            right = rng.randint(max_col + 1, width - 2)
            paths.append({"color": color, "axis": "h", "fixed": extra_row, "start": left, "end": right})

        vertical_specs = [(high_color, high_col), (low_color, low_col)] + list(zip(extra_v_colors, extra_cols))
        for color, fixed_col in vertical_specs:
            top = rng.randint(1, min_row - 1)
            bottom = rng.randint(max_row + 1, height - 2)
            paths.append({"color": color, "axis": "v", "fixed": fixed_col, "start": top, "end": bottom})

        return height, width, priority_colors, paths
    return None


def _protected_vertical_scene(
    rng: random.Random,
) -> tuple[int, int, list[int], list[dict[str, int | str]]] | None:
    for _ in range(100):
        extra_h = rng.randint(0, 1)
        extra_v = rng.randint(0, 2)
        total = 3 + extra_h + extra_v
        height = rng.randint(max(12, 2 * total + 1), 22)
        width = rng.randint(14, 24)

        row = rng.randint(2, height - 4)
        col = rng.randint(4, width - 4)
        extra_rows = _pick_spaced(extra_h, 2, height - 3, 3, rng, blocked=[row, row + 1])
        extra_cols = _pick_spaced(extra_v, 4, width - 4, 3, rng, blocked=[col])
        if extra_rows is None or extra_cols is None:
            continue

        rows = [row, row + 1] + extra_rows
        cols = [col] + extra_cols
        min_row, max_row = min(rows), max(rows)
        min_col, max_col = min(cols), max(cols)
        if min_row <= 1 or max_row >= height - 2 or min_col <= 3 or max_col >= width - 3:
            continue

        priority_colors = rng.sample(list(COLOR_POOL), total)
        high_color, mid_color, low_color = priority_colors[:3]
        extra_colors = priority_colors[3:]
        high_row, low_row = (row, row + 1) if rng.random() < 0.5 else (row + 1, row)

        paths: list[dict[str, int | str]] = []
        vertical_top = rng.randint(1, min_row - 1)
        vertical_bottom = rng.randint(max_row + 1, height - 2)
        paths.append({"color": mid_color, "axis": "v", "fixed": col, "start": vertical_top, "end": vertical_bottom})

        extra_h_colors = extra_colors[:extra_h]
        extra_v_colors = extra_colors[extra_h:]

        horizontal_specs = [(high_color, high_row), (low_color, low_row)] + list(zip(extra_h_colors, extra_rows))
        for color, fixed_row in horizontal_specs:
            left = rng.randint(2, min_col - 1)
            right = rng.randint(max_col + 1, width - 2)
            paths.append({"color": color, "axis": "h", "fixed": fixed_row, "start": left, "end": right})

        for extra_col, color in zip(extra_cols, extra_v_colors):
            top = rng.randint(1, min_row - 1)
            bottom = rng.randint(max_row + 1, height - 2)
            paths.append({"color": color, "axis": "v", "fixed": extra_col, "start": top, "end": bottom})

        return height, width, priority_colors, paths
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    builders = [_normal_scene, _protected_horizontal_scene, _protected_vertical_scene]
    rng.shuffle(builders)

    for builder in builders:
        scene = builder(rng)
        if scene is None:
            continue
        height, width, priority_colors, paths = scene
        grid, occupancy, by_color = _render_scene(height, width, priority_colors, paths, rng)
        expected = _expected_output(grid, occupancy, by_color, priority_colors)

        if not np.any(expected != grid):
            continue
        if sum(len(colors) == 2 for colors in occupancy.values()) == 0:
            continue
        if not np.array_equal(main(grid), expected):
            continue
        return grid
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid crossing-weave input")
