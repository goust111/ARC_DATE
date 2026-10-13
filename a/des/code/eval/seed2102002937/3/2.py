from __future__ import annotations

import random

import numpy as np


BG_COLOR = 0
SOURCE_COLOR = 5
LAMP_COLOR = 9
PATH_COLORS = (1, 2, 3, 4, 6, 8)

GRID_H_RANGE = (12, 22)
GRID_W_RANGE = (12, 22)
MAX_LINES_PER_ORIENTATION = 3
MIN_LINE_GAP = 2
GEN_ATTEMPTS = 400

DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _path_colors(g: np.ndarray) -> set[int]:
    return {
        int(color)
        for color in np.unique(g)
        if int(color) not in (BG_COLOR, SOURCE_COLOR, LAMP_COLOR)
    }


def _source_neighbor(g: np.ndarray, source: tuple[int, int], path_colors: set[int]) -> tuple[tuple[int, int], tuple[int, int], int]:
    sr, sc = source
    neighbors: list[tuple[tuple[int, int], tuple[int, int], int]] = []
    for dr, dc in DIR4:
        nr = sr + dr
        nc = sc + dc
        if not _in_bounds(g.shape, nr, nc):
            continue
        color = int(g[nr, nc])
        if color in path_colors:
            neighbors.append(((nr, nc), (dr, dc), color))
    assert len(neighbors) == 1, f"expected one path neighbor for source at {source}, got {neighbors}"
    return neighbors[0]


def _trace_signal(
    g: np.ndarray,
    source: tuple[int, int],
    path_colors: set[int],
) -> tuple[int, tuple[int, int] | None]:
    (cur_r, cur_c), (dr, dc), path_color = _source_neighbor(g, source, path_colors)

    while True:
        next_r = cur_r + dr
        next_c = cur_c + dc
        assert _in_bounds(g.shape, next_r, next_c), f"path {path_color} ran out of bounds"
        value = int(g[next_r, next_c])

        if value == path_color:
            cur_r, cur_c = next_r, next_c
            continue

        if value == LAMP_COLOR:
            return path_color, (next_r, next_c)

        if value in path_colors:
            after_r = next_r + dr
            after_c = next_c + dc
            assert _in_bounds(g.shape, after_r, after_c), f"blocked path {path_color} must continue past crossing"
            after_value = int(g[after_r, after_c])
            assert after_value in (path_color, LAMP_COLOR), (
                f"blocked path {path_color} must resume after crossing, got {after_value}"
            )
            return path_color, None

        raise AssertionError(f"unexpected value {value} while tracing path {path_color} from {source}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow each straight colored path from its grey source toward its maroon lamp.

    At a crossing, the pixel color names the path that is allowed to continue.
    If a path reaches its lamp, only that lamp changes to the path color.
    """
    g = np.asarray(grid, dtype=int)
    path_colors = _path_colors(g)
    assert path_colors, "expected at least one colored path"

    sources = [tuple(map(int, rc)) for rc in np.argwhere(g == SOURCE_COLOR)]
    assert sources, "expected at least one source"

    out = g.copy()
    seen_colors: set[int] = set()
    lit_lamps: set[tuple[int, int]] = set()
    for source in sources:
        path_color, lamp = _trace_signal(g, source, path_colors)
        assert path_color not in seen_colors, f"duplicate source for path color {path_color}"
        seen_colors.add(path_color)
        if lamp is not None:
            assert lamp not in lit_lamps, f"two paths reached lamp at {lamp}"
            lit_lamps.add(lamp)
            out[lamp] = path_color

    assert seen_colors == path_colors, f"unmatched path colors: {path_colors - seen_colors}"
    return out


def _max_spaced_count(span: int, min_gap: int) -> int:
    if span <= 0:
        return 0
    return (int(span) + int(min_gap) - 1) // int(min_gap)


def _sample_spaced_positions(lo: int, hi: int, count: int, min_gap: int) -> list[int] | None:
    span = hi - lo + 1
    need = count + (count - 1) * (min_gap - 1)
    if span < need:
        return None

    base_span = span - (count - 1) * (min_gap - 1)
    picks = sorted(random.sample(range(base_span), count))
    return [int(lo + pick + i * (min_gap - 1)) for i, pick in enumerate(picks)]


def _render_path(
    g: np.ndarray,
    color: int,
    orientation: str,
    anchor: int,
    start: int,
    end: int,
    source_side: str,
) -> dict:
    if orientation == "h":
        row = int(anchor)
        left = int(start)
        right = int(end)
        g[row, left + 1 : right] = int(color)
        if source_side == "left":
            source = (row, left)
            lamp = (row, right)
        else:
            source = (row, right)
            lamp = (row, left)
    else:
        col = int(anchor)
        top = int(start)
        bottom = int(end)
        g[top + 1 : bottom, col] = int(color)
        if source_side == "top":
            source = (top, col)
            lamp = (bottom, col)
        else:
            source = (bottom, col)
            lamp = (top, col)

    g[source] = SOURCE_COLOR
    g[lamp] = LAMP_COLOR
    return {
        "color": int(color),
        "orientation": orientation,
        "anchor": int(anchor),
        "start": int(start),
        "end": int(end),
        "source_side": source_side,
        "source": source,
        "lamp": lamp,
    }


def _only_lamps_change(inp: np.ndarray, out: np.ndarray, lamps: set[tuple[int, int]]) -> bool:
    diffs = np.argwhere(inp != out)
    if not len(diffs):
        return False
    diff_set = {tuple(map(int, rc)) for rc in diffs}
    return diff_set <= lamps


def _make_scene() -> tuple[np.ndarray, list[dict]] | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)

    row_span = h - 4
    col_span = w - 4
    max_h = min(MAX_LINES_PER_ORIENTATION, _max_spaced_count(row_span, MIN_LINE_GAP))
    max_v = min(MAX_LINES_PER_ORIENTATION, _max_spaced_count(col_span, MIN_LINE_GAP))
    if max_h == 0 or max_v == 0:
        return None

    n_h = random.randint(1, max_h)
    n_v = random.randint(1, max_v)
    rows = _sample_spaced_positions(2, h - 3, n_h, MIN_LINE_GAP)
    cols = _sample_spaced_positions(2, w - 3, n_v, MIN_LINE_GAP)
    if rows is None or cols is None:
        return None

    colors = random.sample(PATH_COLORS, n_h + n_v)
    grid = np.full((h, w), BG_COLOR, dtype=int)

    min_col = min(cols)
    max_col = max(cols)
    min_row = min(rows)
    max_row = max(rows)
    assert min_col >= 2 and max_col <= w - 3
    assert min_row >= 2 and max_row <= h - 3

    paths: list[dict] = []
    horizontal_paths: list[dict] = []
    vertical_paths: list[dict] = []

    for row in rows:
        left = random.randint(0, min_col - 2)
        right = random.randint(max_col + 2, w - 1)
        path = _render_path(
            grid,
            color=colors.pop(),
            orientation="h",
            anchor=row,
            start=left,
            end=right,
            source_side=random.choice(("left", "right")),
        )
        horizontal_paths.append(path)
        paths.append(path)

    for col in cols:
        top = random.randint(0, min_row - 2)
        bottom = random.randint(max_row + 2, h - 1)
        path = _render_path(
            grid,
            color=colors.pop(),
            orientation="v",
            anchor=col,
            start=top,
            end=bottom,
            source_side=random.choice(("top", "bottom")),
        )
        vertical_paths.append(path)
        paths.append(path)

    for h_path in horizontal_paths:
        row = int(h_path["anchor"])
        for v_path in vertical_paths:
            col = int(v_path["anchor"])
            grid[row, col] = random.choice((int(h_path["color"]), int(v_path["color"])))

    return grid, paths


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        built = _make_scene()
        if built is None:
            continue
        grid, paths = built

        try:
            out = main(grid)
        except AssertionError:
            continue

        lamp_cells = {tuple(path["lamp"]) for path in paths}
        if not _only_lamps_change(grid, out, lamp_cells):
            continue

        lit_count = sum(int(out[path["lamp"]]) == int(path["color"]) for path in paths)
        if not (1 <= lit_count < len(paths)):
            continue

        return grid

    raise RuntimeError("failed to generate a valid signal-routing scene")
