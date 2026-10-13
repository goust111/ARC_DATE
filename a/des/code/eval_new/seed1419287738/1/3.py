import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7

SIGNAL_COLORS = (RED, BLUE, GREEN, YELLOW, ORANGE)
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
GEN_MAX_ATTEMPTS = 500


def _in_bounds(shape, r, c):
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _legend_bbox(shape, corner, width):
    h, w = shape
    if corner == "tl":
        return 0, 0, 1, width - 1
    if corner == "tr":
        return 0, w - width, 1, w - 1
    if corner == "bl":
        return h - 2, 0, h - 1, width - 1
    if corner == "br":
        return h - 2, w - width, h - 1, w - 1
    raise ValueError(corner)


def _cells_in_bbox(bbox):
    r0, c0, r1, c1 = bbox
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _expand_cells(shape, cells, radius=1):
    h, w = shape
    out = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _read_legend(grid):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h >= 2 and w >= 2, "legend needs at least two rows and two columns"

    candidates = []
    for r0 in sorted({0, h - 2}):
        if g[r0, 0] == MAGENTA:
            c1 = 0
            while c1 < w and int(g[r0, c1]) == MAGENTA:
                c1 += 1
            width = c1
            colors = [int(v) for v in g[r0 + 1, 0:width]]
            if 2 <= width <= 5 and all(color in SIGNAL_COLORS for color in colors) and len(set(colors)) == width:
                bbox = (r0, 0, r0 + 1, width - 1)
                candidates.append((colors, bbox))

        if g[r0, w - 1] == MAGENTA:
            c0 = w - 1
            while c0 >= 0 and int(g[r0, c0]) == MAGENTA:
                c0 -= 1
            c0 += 1
            width = w - c0
            colors = [int(v) for v in g[r0 + 1, c0:w]]
            if 2 <= width <= 5 and all(color in SIGNAL_COLORS for color in colors) and len(set(colors)) == width:
                bbox = (r0, c0, r0 + 1, w - 1)
                candidates.append((colors, bbox))

    dedup = {}
    for colors, bbox in candidates:
        dedup[bbox] = colors
    assert len(dedup) == 1, "expected exactly one corner legend"

    bbox, colors = next(iter(dedup.items()))
    mask = np.zeros_like(g, dtype=bool)
    r0, c0, r1, c1 = bbox
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return list(colors), mask


def _parse_scene(grid):
    g = np.asarray(grid, dtype=int)
    priority, legend_mask = _read_legend(g)
    priority_set = set(priority)
    towers = []
    walls = set()

    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            value = int(g[r, c])
            if legend_mask[r, c] or value == BLACK:
                continue
            if value == GREY:
                walls.add((r, c))
                continue
            assert value in priority_set, "all non-wall colors must be listed in the legend"
            towers.append((r, c, value))

    assert towers, "expected at least one signal tower"
    for r, c, color in towers:
        same_color_neighbors = 0
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if _in_bounds(g.shape, nr, nc) and not legend_mask[nr, nc] and int(g[nr, nc]) == color:
                same_color_neighbors += 1
        assert same_color_neighbors == 0, "signal towers must be single pixels"
    return priority, towers, walls


def _render_signals(shape, priority, towers, walls):
    rank = {color: i for i, color in enumerate(priority)}
    best_rank = np.full(shape, len(priority), dtype=int)
    best_color = np.full(shape, BLACK, dtype=int)
    contributors = {}
    wall_stop = False

    def claim(r, c, color):
        contributors.setdefault((r, c), set()).add(color)
        if rank[color] < best_rank[r, c]:
            best_rank[r, c] = rank[color]
            best_color[r, c] = color

    for r, c, color in towers:
        claim(r, c, color)
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            while _in_bounds(shape, nr, nc) and (nr, nc) not in walls:
                claim(nr, nc, color)
                nr += dr
                nc += dc
            if _in_bounds(shape, nr, nc) and (nr, nc) in walls:
                wall_stop = True

    out = np.full(shape, BLACK, dtype=int)
    for r, c in walls:
        out[r, c] = GREY
    signal_mask = best_color != BLACK
    out[signal_mask] = best_color[signal_mask]
    for r, c, color in towers:
        out[r, c] = color
    return out, contributors, wall_stop


def main(grid: np.ndarray) -> np.ndarray:
    """
    Broadcast each colored signal tower along rows and columns, resolving crossings by the corner legend priority.

    Input structure:
    - The background is black.
    - The foreground contains several single-pixel signal towers in red, blue, green, yellow, or orange, plus grey wall pixels that may appear as isolated cells or short straight segments.
    - A corner legend occupies a 2xN strip: a horizontal magenta row sits directly above a row of 2-5 colored pixels whose left-to-right order encodes the signal priority.

    Rule:
    - Read the legend's lower row to obtain the color priority from highest to lowest.
    - Each tower emits its own color from its cell in the four cardinal directions and keeps extending until the next step would hit a grey wall or leave the grid.
    - If multiple signal colors cover the same non-wall cell, keep only the highest-priority color from the legend.
    - Produce a fresh output grid: keep walls grey, keep towers at their original colors, color every signaled cell by the winning signal, and leave all other cells black.

    Key insight:
    - The magenta corner strip is not part of the final picture; it is a legend that ranks colors for resolving overlaps between independently broadcast row-and-column signal rays.
    """
    g = np.asarray(grid, dtype=int)
    priority, towers, walls = _parse_scene(g)
    out, _contributors, _wall_stop = _render_signals(g.shape, priority, towers, walls)
    return out


def _sample_spaced(values, count, min_gap):
    values = list(values)
    if len(values) < count:
        return None
    for _ in range(200):
        random.shuffle(values)
        chosen = []
        for value in values:
            if all(abs(value - prev) >= min_gap for prev in chosen):
                chosen.append(value)
                if len(chosen) == count:
                    chosen.sort()
                    return chosen
    return sorted(random.sample(values, count))


def _place_towers(shape, priority, legend_bbox):
    h, w = shape
    legend_forbidden = _expand_cells(shape, _cells_in_bbox(legend_bbox), radius=1)
    forbidden_rows = {r for r, _c in legend_forbidden}
    forbidden_cols = {c for _r, c in legend_forbidden}

    rows = _sample_spaced([r for r in range(1, h - 1) if r not in forbidden_rows], len(priority), 2)
    cols = _sample_spaced([c for c in range(1, w - 1) if c not in forbidden_cols], len(priority), 2)
    if rows is None or cols is None:
        return None

    tower_colors = list(priority)
    random.shuffle(tower_colors)
    random.shuffle(cols)
    towers = [(rows[i], cols[i], tower_colors[i]) for i in range(len(priority))]
    if any((r, c) in legend_forbidden for r, c, _color in towers):
        return None
    return towers


def _ray_to_edge(shape, start, direction):
    r, c = start
    dr, dc = direction
    cells = []
    nr, nc = r + dr, c + dc
    while _in_bounds(shape, nr, nc):
        cells.append((nr, nc))
        nr += dr
        nc += dc
    return cells


def _place_walls(shape, towers, legend_bbox):
    tower_cells = {(r, c) for r, c, _color in towers}
    legend_buffer = _expand_cells(shape, _cells_in_bbox(legend_bbox), radius=1)
    tower_buffer = _expand_cells(shape, tower_cells, radius=1)
    blocked = legend_buffer | tower_cells
    keep_clear = legend_buffer | tower_buffer

    wall_candidates = []
    for r, c, _color in towers:
        for direction in DIRS:
            ray = _ray_to_edge(shape, (r, c), direction)
            if len(ray) < 2:
                continue
            preferred = [cell for cell in ray[1:-1] if cell not in blocked and cell not in keep_clear]
            fallback = [cell for cell in ray[1:] if cell not in blocked and cell not in keep_clear]
            for cell in preferred or fallback:
                wall_candidates.append(cell)
    if not wall_candidates:
        return None

    walls = {random.choice(wall_candidates)}
    segment_count = random.randint(0, 3)
    orientations = [(0, 1), (1, 0)]

    for _ in range(segment_count):
        placed = False
        for _attempt in range(80):
            length = random.randint(1, 4)
            dr, dc = random.choice(orientations)
            r0 = random.randint(0, shape[0] - 1)
            c0 = random.randint(0, shape[1] - 1)
            segment = []
            for step in range(length):
                nr = r0 + dr * step
                nc = c0 + dc * step
                if not _in_bounds(shape, nr, nc):
                    segment = []
                    break
                segment.append((nr, nc))
            if not segment:
                continue
            if any(cell in keep_clear or cell in walls for cell in segment):
                continue
            walls.update(segment)
            placed = True
            break
        if not placed:
            continue
    return walls


def _render_input(shape, priority, corner, towers, walls):
    grid = np.full(shape, BLACK, dtype=int)
    legend_bbox = _legend_bbox(shape, corner, len(priority))
    r0, c0, _r1, c1 = legend_bbox
    grid[r0, c0 : c1 + 1] = MAGENTA
    grid[r0 + 1, c0 : c1 + 1] = np.array(priority, dtype=int)
    for r, c in walls:
        grid[r, c] = GREY
    for r, c, color in towers:
        grid[r, c] = color
    return grid


def _interesting_scene(inp, out, priority, towers, walls):
    _expected, contributors, wall_stop = _render_signals(inp.shape, priority, towers, walls)
    tower_cells = {(r, c) for r, c, _color in towers}
    conflict = any(len(colors) > 1 and cell not in tower_cells for cell, colors in contributors.items())
    signal_only = np.count_nonzero(out != BLACK) - len(walls)
    return wall_stop and conflict and signal_only > len(towers) and not np.array_equal(inp, out)


def _gen_attempt():
    color_count = random.randint(2, 5)
    min_size = 10 + 2 * color_count
    h = random.randint(min_size, 24)
    w = random.randint(min_size, 24)
    shape = (h, w)

    priority = random.sample(list(SIGNAL_COLORS), color_count)
    corner = random.choice(["tl", "tr", "bl", "br"])
    legend_bbox = _legend_bbox(shape, corner, color_count)

    towers = _place_towers(shape, priority, legend_bbox)
    if towers is None:
        return None

    walls = _place_walls(shape, towers, legend_bbox)
    if walls is None:
        return None

    inp = _render_input(shape, priority, corner, towers, walls)
    expected, _contributors, _wall_stop = _render_signals(shape, priority, towers, walls)
    if not _interesting_scene(inp, expected, priority, towers, walls):
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid signal-tower input")
