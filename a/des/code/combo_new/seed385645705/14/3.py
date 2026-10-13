import random
from collections import defaultdict

import numpy as np


BLACK = 0
GREY = 5
SOURCE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 400


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _line_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    (r0, c0), (r1, c1) = a, b
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    if c0 == c1:
        step = 1 if r1 >= r0 else -1
        return [(r, c0) for r in range(r0, r1 + step, step)]
    raise ValueError("segments must be axis-aligned")


def _corner_candidate(g: np.ndarray, corner: str):
    h, w = g.shape
    if w < 2 or h < 2:
        return None

    if corner.endswith("left"):
        c0, c1 = 0, 1
    else:
        c0, c1 = w - 2, w - 1

    rows = range(h) if corner.startswith("top") else range(h - 1, -1, -1)
    colors = []
    for r in rows:
        a = int(g[r, c0])
        b = int(g[r, c1])
        if a == b and a not in (BLACK, GREY):
            colors.append(a)
        else:
            break

    if len(colors) < 2 or len(set(colors)) != len(colors):
        return None

    if corner.startswith("top"):
        ordered = colors
        r0 = 0
    else:
        ordered = list(reversed(colors))
        r0 = h - len(colors)

    mask = np.zeros_like(g, dtype=bool)
    mask[r0 : r0 + len(ordered), c0 : c1 + 1] = True
    return ordered, mask


def _find_legend(g: np.ndarray):
    candidates = []
    for corner in ("top_left", "top_right", "bottom_left", "bottom_right"):
        candidate = _corner_candidate(g, corner)
        if candidate is not None:
            candidates.append(candidate)
    assert candidates, "expected a corner legend"
    return max(candidates, key=lambda item: len(item[0]))


def _source_direction(pos: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int]:
    r, c = pos
    h, w = shape
    on_edges = (r == 0) + (r == h - 1) + (c == 0) + (c == w - 1)
    assert on_edges == 1, "sources must sit on exactly one border"
    if r == 0:
        return 1, 0
    if r == h - 1:
        return -1, 0
    if c == 0:
        return 0, 1
    return 0, -1


def _extract_sources(g: np.ndarray, legend_mask: np.ndarray, priority: dict[int, int]):
    h, w = g.shape
    sources = []
    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            if legend_mask[r, c]:
                continue
            color = int(g[r, c])
            if color in (BLACK, GREY):
                continue
            assert color in priority, "source color missing from legend"
            sources.append((color, (r, c)))

    assert sources, "expected at least one source"
    colors = [color for color, _pos in sources]
    assert len(set(colors)) == len(colors), "source colors must be unique"
    assert set(colors) == set(priority), "legend and sources must use the same colors"
    return sources


def _trace_path(g: np.ndarray, source_pos: tuple[int, int], legend_mask: np.ndarray):
    h, w = g.shape
    dr, dc = _source_direction(source_pos, g.shape)
    cur = (source_pos[0] + dr, source_pos[1] + dc)
    assert 0 <= cur[0] < h and 0 <= cur[1] < w
    assert not legend_mask[cur], "path cannot enter legend"
    assert int(g[cur]) == GREY, "source must touch a grey path"

    prev = source_pos
    path = []
    seen = set()
    while True:
        assert cur not in seen, "paths must not loop"
        seen.add(cur)
        path.append(cur)

        neighbors = [
            (nr, nc)
            for nr, nc in _neighbors4(cur[0], cur[1], h, w)
            if int(g[nr, nc]) == GREY and not legend_mask[nr, nc]
        ]
        candidates = [cell for cell in neighbors if cell != prev]
        if not candidates:
            break
        if len(candidates) == 1:
            nxt = candidates[0]
        else:
            in_dr = cur[0] - prev[0]
            in_dc = cur[1] - prev[1]
            straight = [
                cell
                for cell in candidates
                if cell[0] - cur[0] == in_dr and cell[1] - cur[1] == in_dc
            ]
            assert len(straight) == 1, "crossings must continue straight through"
            nxt = straight[0]
        prev, cur = cur, nxt

    return path


def _solve(g: np.ndarray) -> np.ndarray:
    priority_colors, legend_mask = _find_legend(g)
    priority = {color: idx for idx, color in enumerate(priority_colors)}
    sources = _extract_sources(g, legend_mask, priority)

    paths: dict[int, list[tuple[int, int]]] = {}
    for color, pos in sources:
        paths[color] = _trace_path(g, pos, legend_mask)

    cell_to_colors: dict[tuple[int, int], list[int]] = defaultdict(list)
    for color, path in paths.items():
        for cell in path:
            cell_to_colors[cell].append(color)

    cut_index = {}
    for color, path in paths.items():
        stop = len(path) - 1
        for idx, cell in enumerate(path):
            if any(priority[other] < priority[color] for other in cell_to_colors[cell] if other != color):
                stop = idx
                break
        cut_index[color] = stop

    active_cells: dict[tuple[int, int], list[int]] = defaultdict(list)
    for color, path in paths.items():
        for cell in path[: cut_index[color] + 1]:
            active_cells[cell].append(color)

    out = g.copy()
    for (r, c), colors in active_cells.items():
        out[r, c] = min(colors, key=priority.get)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate each edge signal along grey paths, keeping only the higher-priority color at crossings.

    Input structure:
    - The background is black.
    - Several single-pixel colored sources sit on the outer border, and each source touches a one-pixel-wide grey path entering the grid.
    - Grey paths run only horizontally or vertically, may bend at right angles, and may cross by sharing one pixel.
    - One corner contains a 2-column legend whose rows list all source colors from top to bottom in priority order.

    Rule:
    - Read the corner legend to get the source-color priority, with earlier rows beating later rows.
    - Trace each source's grey path from the border inward and recolor that path with the source color.
    - If multiple traced paths share a crossing pixel, keep the highest-priority color on that pixel and truncate every lower-priority path there, so the cells after that crossing stay grey.
    - Leave the source pixels and the legend itself unchanged.

    Key insight:
    - The same legend controls both which color wins at a crossing and where a losing path stops propagating.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    return _solve(g)


def _sample_spaced_positions(low: int, high: int, count: int, min_gap: int):
    if count == 0:
        return []
    if low > high:
        return None
    values = list(range(low, high + 1))
    if len(values) < count:
        return None
    for _ in range(120):
        picks = sorted(random.sample(values, count))
        if all(b - a >= min_gap for a, b in zip(picks, picks[1:])):
            return picks
    return None


def _legend_geometry(corner: str, h: int, w: int, n: int):
    if corner == "top_left":
        return (0, n - 1), (0, 1)
    if corner == "top_right":
        return (0, n - 1), (w - 2, w - 1)
    if corner == "bottom_left":
        return (h - n, h - 1), (0, 1)
    return (h - n, h - 1), (w - 2, w - 1)


def _opposite_source_sides(corner: str):
    if corner == "top_left":
        return "bottom", "right"
    if corner == "top_right":
        return "bottom", "left"
    if corner == "bottom_left":
        return "top", "right"
    return "top", "left"


def _terminal_limits(corner: str, h: int, w: int, legend_h: int):
    top_limit = legend_h + 1 if corner.startswith("top") else 1
    bottom_limit = h - legend_h - 2 if corner.startswith("bottom") else h - 2
    left_limit = 2 if corner.endswith("left") else 1
    right_limit = w - 3 if corner.endswith("right") else w - 2
    return top_limit, bottom_limit, left_limit, right_limit


def _place_path(
    grid: np.ndarray,
    cells: list[tuple[int, int]],
    orientation: str,
    allowed_crosses: set[tuple[int, int]],
    occupancy: dict[tuple[int, int], str],
):
    for cell in cells:
        existing = occupancy.get(cell)
        if existing is None:
            occupancy[cell] = orientation
        else:
            if existing == orientation or cell not in allowed_crosses:
                return False
            occupancy[cell] = "cross"
        grid[cell] = GREY
    return True


def _build_vertical_path(
    h: int,
    col: int,
    source_side: str,
    end_row: int,
    turn_col: int | None,
):
    source = (0, col) if source_side == "top" else (h - 1, col)
    start = (1, col) if source_side == "top" else (h - 2, col)
    pivot = (end_row, col)
    cells = _line_cells(start, pivot)
    if turn_col is not None:
        cells.extend(_line_cells(pivot, (end_row, turn_col))[1:])
    return source, cells


def _build_horizontal_path(
    w: int,
    row: int,
    source_side: str,
    end_col: int,
    turn_row: int | None,
):
    source = (row, 0) if source_side == "left" else (row, w - 1)
    start = (row, 1) if source_side == "left" else (row, w - 2)
    pivot = (row, end_col)
    cells = _line_cells(start, pivot)
    if turn_row is not None:
        cells.extend(_line_cells(pivot, (turn_row, end_col))[1:])
    return source, cells


def _gen_attempt():
    total_paths = random.randint(2, 5)
    vertical_count = random.randint(1, total_paths - 1)
    horizontal_count = total_paths - vertical_count

    legend_corner = random.choice(["top_left", "top_right", "bottom_left", "bottom_right"])
    vertical_side, horizontal_side = _opposite_source_sides(legend_corner)

    min_h = max(14, total_paths + 2 * horizontal_count + 9)
    min_w = max(14, total_paths + 2 * vertical_count + 9)
    if min_h > 24 or min_w > 24:
        return None
    h = random.randint(min_h, 24)
    w = random.randint(min_w, 24)

    priority_colors = random.sample(SOURCE_COLORS, total_paths)
    path_colors = priority_colors[:]
    random.shuffle(path_colors)

    legend_rows, legend_cols = _legend_geometry(legend_corner, h, w, total_paths)
    top_limit, bottom_limit, left_limit, right_limit = _terminal_limits(legend_corner, h, w, total_paths)

    row_low = top_limit + 2
    row_high = bottom_limit - 2
    col_low = left_limit + 2
    col_high = right_limit - 2
    cross_rows = _sample_spaced_positions(row_low, row_high, horizontal_count, 2)
    cross_cols = _sample_spaced_positions(col_low, col_high, vertical_count, 2)
    if cross_rows is None or cross_cols is None:
        return None

    grid = np.full((h, w), BLACK, dtype=int)
    r0, r1 = legend_rows
    c0, c1 = legend_cols
    for offset, color in enumerate(priority_colors):
        grid[r0 + offset, c0 : c1 + 1] = color

    allowed_crosses = {(r, c) for r in cross_rows for c in cross_cols}
    occupancy: dict[tuple[int, int], str] = {}

    turn_orientation = random.choice(["none", "vertical", "horizontal"])
    vertical_turn = None
    horizontal_turn = None

    if turn_orientation == "vertical":
        options = []
        if cross_cols[0] > left_limit:
            options.append(("left", 0))
        if cross_cols[-1] < right_limit:
            options.append(("right", vertical_count - 1))
        if options:
            side_name, idx = random.choice(options)
            if vertical_side == "top":
                end_low, end_high = max(cross_rows) + 2, bottom_limit
            else:
                end_low, end_high = top_limit, min(cross_rows) - 2
            if end_low <= end_high:
                end_row = random.randint(end_low, end_high)
                if side_name == "left":
                    turn_col = random.randint(left_limit, cross_cols[0] - 1)
                else:
                    turn_col = random.randint(cross_cols[-1] + 1, right_limit)
                vertical_turn = (idx, end_row, turn_col)

    if turn_orientation == "horizontal":
        options = []
        if cross_rows[0] > top_limit:
            options.append(("up", 0))
        if cross_rows[-1] < bottom_limit:
            options.append(("down", horizontal_count - 1))
        if options:
            side_name, idx = random.choice(options)
            if horizontal_side == "left":
                end_low, end_high = max(cross_cols) + 2, right_limit
            else:
                end_low, end_high = left_limit, min(cross_cols) - 2
            if end_low <= end_high:
                end_col = random.randint(end_low, end_high)
                if side_name == "up":
                    turn_row = random.randint(top_limit, cross_rows[0] - 1)
                else:
                    turn_row = random.randint(cross_rows[-1] + 1, bottom_limit)
                horizontal_turn = (idx, end_col, turn_row)

    specs = []
    color_iter = iter(path_colors)

    for idx, col in enumerate(cross_cols):
        color = next(color_iter)
        if vertical_side == "top":
            end_low, end_high = max(cross_rows) + 2, bottom_limit
        else:
            end_low, end_high = top_limit, min(cross_rows) - 2
        if end_low > end_high:
            return None
        end_row = random.randint(end_low, end_high)
        turn_col = None
        if vertical_turn is not None and vertical_turn[0] == idx:
            end_row = vertical_turn[1]
            turn_col = vertical_turn[2]
        source, cells = _build_vertical_path(h, col, vertical_side, end_row, turn_col)
        specs.append((color, source, cells, "vertical"))

    for idx, row in enumerate(cross_rows):
        color = next(color_iter)
        if horizontal_side == "left":
            end_low, end_high = max(cross_cols) + 2, right_limit
        else:
            end_low, end_high = left_limit, min(cross_cols) - 2
        if end_low > end_high:
            return None
        end_col = random.randint(end_low, end_high)
        turn_row = None
        if horizontal_turn is not None and horizontal_turn[0] == idx:
            end_col = horizontal_turn[1]
            turn_row = horizontal_turn[2]
        source, cells = _build_horizontal_path(w, row, horizontal_side, end_col, turn_row)
        specs.append((color, source, cells, "horizontal"))

    random.shuffle(specs)
    source_positions = set()
    for color, source, cells, orientation in specs:
        if source in source_positions:
            return None
        source_positions.add(source)
        if grid[source] != BLACK:
            return None
        if any(grid[cell] != BLACK and grid[cell] != GREY for cell in cells):
            return None
        if not _place_path(grid, cells, orientation, allowed_crosses, occupancy):
            return None
        grid[source] = color

    if not any(state == "cross" for state in occupancy.values()):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid signal-crossing input")
