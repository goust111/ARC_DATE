import random
from collections import defaultdict

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = tuple(c for c in range(10) if c not in (BLACK, GREY))

DIR_DELTAS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}

PANEL_LAYOUTS = ((2, 2), (2, 3), (3, 2), (2, 4), (4, 2))
PANEL_SIZE_CHOICES = (5, 7, 9)
MAX_GENERATION_ATTEMPTS = 300


def _arrow_cells(height: int, width: int, direction: str) -> set[tuple[int, int]]:
    assert height >= 3 and width >= 3, "panels must fit a centered 3-pixel arrow"
    cr = height // 2
    cc = width // 2
    patterns = {
        "R": {(cr, cc - 1), (cr - 1, cc + 1), (cr + 1, cc + 1)},
        "L": {(cr, cc + 1), (cr - 1, cc - 1), (cr + 1, cc - 1)},
        "U": {(cr + 1, cc), (cr - 1, cc - 1), (cr - 1, cc + 1)},
        "D": {(cr - 1, cc), (cr + 1, cc - 1), (cr + 1, cc + 1)},
    }
    return patterns[direction]


def _find_separator_color(g: np.ndarray) -> int:
    row_colors = {
        int(row[0])
        for row in g
        if row.size and np.all(row == int(row[0]))
    }
    col_colors = {
        int(g[0, c])
        for c in range(g.shape[1])
        if np.all(g[:, c] == int(g[0, c]))
    }

    if GREY in row_colors or GREY in col_colors:
        return GREY

    both = row_colors & col_colors
    if len(both) == 1:
        return both.pop()

    either = row_colors | col_colors
    assert len(either) == 1, "expected a unique full-line separator color"
    return either.pop()


def _line_indices(g: np.ndarray, sep: int, axis: int) -> list[int]:
    if axis == 0:
        return [r for r in range(g.shape[0]) if np.all(g[r, :] == sep)]
    return [c for c in range(g.shape[1]) if np.all(g[:, c] == sep)]


def _spans(length: int, separators: list[int]) -> list[tuple[int, int]]:
    sep_set = set(separators)
    spans = []
    i = 0
    while i < length:
        while i < length and i in sep_set:
            i += 1
        start = i
        while i < length and i not in sep_set:
            i += 1
        if start < i:
            spans.append((start, i - 1))
    return spans


def _detect_arrow(tile: np.ndarray) -> str:
    black = {(int(r), int(c)) for r, c in np.argwhere(tile == BLACK)}
    assert len(black) == 3, "expected exactly one 3-pixel black arrow per panel"
    for direction in DIR_DELTAS:
        if black == _arrow_cells(tile.shape[0], tile.shape[1], direction):
            return direction
    raise AssertionError("unrecognized centered arrow shape")


def _panel_base_color(tile: np.ndarray, sep: int) -> int:
    body = tile[(tile != BLACK) & (tile != sep)]
    colors = np.unique(body)
    assert len(colors) == 1, "each panel must have one non-black, non-separator base color"
    return int(colors[0])


def _panel_infos(g: np.ndarray):
    sep = _find_separator_color(g)
    row_spans = _spans(g.shape[0], _line_indices(g, sep, axis=0))
    col_spans = _spans(g.shape[1], _line_indices(g, sep, axis=1))
    assert row_spans and col_spans, "expected at least one panel row and column"

    heights = {r1 - r0 + 1 for r0, r1 in row_spans}
    widths = {c1 - c0 + 1 for c0, c1 in col_spans}
    assert len(heights) == 1 and len(widths) == 1, "all panels must have the same size"

    infos = {}
    for pr, (r0, r1) in enumerate(row_spans):
        for pc, (c0, c1) in enumerate(col_spans):
            tile = g[r0 : r1 + 1, c0 : c1 + 1]
            infos[(pr, pc)] = {
                "span": (r0, r1, c0, c1),
                "color": _panel_base_color(tile, sep),
                "direction": _detect_arrow(tile),
                "arrow": {(int(r), int(c)) for r, c in np.argwhere(tile == BLACK)},
            }
    return sep, row_spans, col_spans, infos


def main(grid: np.ndarray) -> np.ndarray:
    """Pass each panel's color to the adjacent panel indicated by its black arrow.

    Input structure:
    - Black pixels form a centered 3-pixel arrow inside every colored panel.
    - Full grey separator rows and columns divide the grid into equal rectangular panels.
    - Each panel has one solid non-black, non-grey base color under its arrow.

    Rule:
    - Read every panel's original base color and arrow direction.
    - A panel pointing to an in-bounds cardinal neighbor sends its original base color to that neighbor.
    - Each output panel receives the color from the incoming source with the smallest panel row, then smallest panel column; panels with no incoming source become black.
    - Separator lines and the black arrow pixels remain in their original locations.

    Key insight:
    - The arrows define a directed adjacency graph, and colors are assigned to targets from their incoming sources rather than moved with the arrow-owning panel.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"

    _sep, row_spans, col_spans, infos = _panel_infos(g)
    panel_rows = len(row_spans)
    panel_cols = len(col_spans)

    incoming: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for (pr, pc), info in infos.items():
        dr, dc = DIR_DELTAS[info["direction"]]
        target = (pr + dr, pc + dc)
        if 0 <= target[0] < panel_rows and 0 <= target[1] < panel_cols:
            incoming[target].append((pr, pc))

    out = g.copy()
    for pos, info in infos.items():
        sources = sorted(incoming.get(pos, []))
        new_color = infos[sources[0]]["color"] if sources else BLACK
        r0, r1, c0, c1 = info["span"]
        out[r0 : r1 + 1, c0 : c1 + 1] = new_color
        for ar, ac in info["arrow"]:
            out[r0 + ar, c0 + ac] = BLACK
    return out


def _valid_panel_sizes(count: int) -> list[int]:
    return [size for size in PANEL_SIZE_CHOICES if count * size + count - 1 <= 30]


def _panel_origin(panel_r: int, panel_c: int, panel_h: int, panel_w: int) -> tuple[int, int]:
    return panel_r * (panel_h + 1), panel_c * (panel_w + 1)


def _valid_dirs(panel_r: int, panel_c: int, panel_rows: int, panel_cols: int) -> list[str]:
    dirs = []
    for direction, (dr, dc) in DIR_DELTAS.items():
        rr = panel_r + dr
        cc = panel_c + dc
        if 0 <= rr < panel_rows and 0 <= cc < panel_cols:
            dirs.append(direction)
    return dirs


def _sources_for_target(target: tuple[int, int], panel_rows: int, panel_cols: int) -> list[tuple[int, int, str]]:
    tr, tc = target
    sources = []
    for direction, (dr, dc) in DIR_DELTAS.items():
        sr = tr - dr
        sc = tc - dc
        if 0 <= sr < panel_rows and 0 <= sc < panel_cols:
            sources.append((sr, sc, direction))
    return sources


def _choose_directions(panel_rows: int, panel_cols: int) -> dict[tuple[int, int], str] | None:
    targets = [(r, c) for r in range(panel_rows) for c in range(panel_cols)]
    collision_targets = [
        (target, _sources_for_target(target, panel_rows, panel_cols))
        for target in targets
        if len(_sources_for_target(target, panel_rows, panel_cols)) >= 2
    ]
    if not collision_targets:
        return None

    target, sources = random.choice(collision_targets)
    forced = random.sample(sources, 2)
    directions = {(sr, sc): direction for sr, sc, direction in forced}

    counts = {direction: 0 for direction in DIR_DELTAS}
    for direction in directions.values():
        counts[direction] += 1

    positions = targets[:]
    random.shuffle(positions)
    for panel_r, panel_c in positions:
        if (panel_r, panel_c) in directions:
            continue
        valid = _valid_dirs(panel_r, panel_c, panel_rows, panel_cols)
        low = min(counts[direction] for direction in valid)
        choices = [direction for direction in valid if counts[direction] == low]
        direction = random.choice(choices)
        directions[(panel_r, panel_c)] = direction
        counts[direction] += 1

    assert target in {
        (sr + DIR_DELTAS[direction][0], sc + DIR_DELTAS[direction][1])
        for (sr, sc), direction in directions.items()
    }
    return directions


def _render_grid(
    panel_rows: int,
    panel_cols: int,
    panel_h: int,
    panel_w: int,
    colors: list[int],
    directions: dict[tuple[int, int], str],
) -> np.ndarray:
    height = panel_rows * panel_h + panel_rows - 1
    width = panel_cols * panel_w + panel_cols - 1
    grid = np.full((height, width), GREY, dtype=int)

    color_iter = iter(colors)
    for panel_r in range(panel_rows):
        for panel_c in range(panel_cols):
            r0, c0 = _panel_origin(panel_r, panel_c, panel_h, panel_w)
            grid[r0 : r0 + panel_h, c0 : c0 + panel_w] = next(color_iter)
            for ar, ac in _arrow_cells(panel_h, panel_w, directions[(panel_r, panel_c)]):
                grid[r0 + ar, c0 + ac] = BLACK
    return grid


def _gen_attempt() -> np.ndarray | None:
    panel_rows, panel_cols = random.choice(PANEL_LAYOUTS)
    panel_count = panel_rows * panel_cols
    if panel_count > len(COLOR_POOL):
        return None

    row_sizes = _valid_panel_sizes(panel_rows)
    col_sizes = _valid_panel_sizes(panel_cols)
    if not row_sizes or not col_sizes:
        return None
    panel_h = random.choice(row_sizes)
    panel_w = random.choice(col_sizes)

    colors = random.sample(COLOR_POOL, panel_count)
    directions = _choose_directions(panel_rows, panel_cols)
    if directions is None:
        return None

    grid = _render_grid(panel_rows, panel_cols, panel_h, panel_w, colors, directions)
    try:
        out = main(grid)
    except Exception:
        return None
    if out.ndim == 2 and max(grid.shape) <= 30 and max(out.shape) <= 30:
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arrow relay input")
