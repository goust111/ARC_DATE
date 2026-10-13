import random

import numpy as np


BLACK = 0
GREY = 5
PATH_COLORS = (3, 6, 7, 8)

GRID_MIN = 16
GRID_MAX = 22
PATH_COUNT_OPTIONS = (2, 3)
PATH_LEN_RANGE = (8, 15)
LEGEND_K_RANGE = (2, 5)
GEN_ATTEMPTS = 500


def neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        comp: list[tuple[int, int]] = []
        while stack:
            rr, cc = stack.pop()
            comp.append((rr, cc))
            for nr, nc in neighbors4(rr, cc):
                if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(comp)
    return comps


def _is_horizontal_legend_run(comp: list[tuple[int, int]]) -> bool:
    if not (LEGEND_K_RANGE[0] <= len(comp) <= LEGEND_K_RANGE[1]):
        return False
    rows = {r for r, _ in comp}
    if len(rows) != 1:
        return False
    cols = sorted(c for _, c in comp)
    return cols == list(range(cols[0], cols[0] + len(cols)))


def _trace_path(color_comp: list[tuple[int, int]], grey_cell: tuple[int, int]) -> list[tuple[int, int]]:
    comp_set = set(color_comp)
    adjacent = [cell for cell in neighbors4(*grey_cell) if cell in comp_set]
    assert len(adjacent) == 1, "grey marker must touch exactly one colored path cell"
    start = adjacent[0]

    degrees = {cell: sum((nbr in comp_set) for nbr in neighbors4(*cell)) for cell in comp_set}
    assert all(deg <= 2 for deg in degrees.values()), "colored path component must be branchless"
    assert sum(deg == 1 for deg in degrees.values()) == 2, "colored path component must have two endpoints"
    assert degrees[start] == 1, "grey must sit at a path endpoint"

    ordered = [grey_cell, start]
    used = {start}
    cur = start
    while True:
        nxts = [nbr for nbr in neighbors4(*cur) if nbr in comp_set and nbr not in used]
        if not nxts:
            break
        assert len(nxts) == 1, "path traversal must be unambiguous"
        nxt = nxts[0]
        ordered.append(nxt)
        used.add(nxt)
        cur = nxt

    assert len(used) == len(comp_set), "path traversal must cover the whole colored component"
    return ordered


def _extract_structure(grid: np.ndarray) -> tuple[int, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)
    colors = sorted(int(v) for v in np.unique(g) if int(v) not in (bg, GREY))
    assert 2 <= len(colors) <= 3, "expected 2-3 path colors"

    legend_counts: dict[int, int] = {}
    path_components: dict[int, list[tuple[int, int]]] = {}

    for color in colors:
        comps = _components(g == color)
        legend = [comp for comp in comps if _is_horizontal_legend_run(comp)]
        assert len(legend) == 1, "each path color must have exactly one legend row"
        legend_set = set(legend[0])
        others = [comp for comp in comps if set(comp) != legend_set]
        assert len(others) == 1, "each path color must have exactly one path component"
        legend_counts[color] = len(legend[0])
        path_components[color] = others[0]

    grey_components = _components(g == GREY)
    assert all(len(comp) == 1 for comp in grey_components), "grey markers must be single pixels"
    assert len(grey_components) == len(colors), "expected one grey marker per path"

    used_colors: set[int] = set()
    paths: list[dict[str, object]] = []
    for comp in grey_components:
        grey_cell = comp[0]
        adjacent_colors = {
            int(g[nr, nc])
            for nr, nc in neighbors4(*grey_cell)
            if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and int(g[nr, nc]) in path_components
        }
        assert len(adjacent_colors) == 1, "grey marker must match exactly one path color"
        color = adjacent_colors.pop()
        assert color not in used_colors, "each path color must have exactly one grey marker"
        ordered = _trace_path(path_components[color], grey_cell)
        k = int(legend_counts[color])
        assert 1 <= k < len(ordered), "legend count must fit inside its path"
        used_colors.add(color)
        paths.append({"color": color, "k": k, "path": ordered})

    return bg, paths


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg, paths = _extract_structure(g)
    out = g.copy()
    for info in paths:
        k = int(info["k"])
        path = list(info["path"])
        for cell in path[:k]:
            out[cell] = bg
        out[path[k]] = GREY
    return out


def _mark_blocked(blocked: np.ndarray, cells: list[tuple[int, int]], radius: int) -> None:
    h, w = blocked.shape
    for r, c in cells:
        r0 = max(0, r - radius)
        r1 = min(h - 1, r + radius)
        c0 = max(0, c - radius)
        c1 = min(w - 1, c + radius)
        blocked[r0 : r1 + 1, c0 : c1 + 1] = True


def _turn_count(path: list[tuple[int, int]]) -> int:
    turns = 0
    for i in range(2, len(path)):
        dr1 = path[i - 1][0] - path[i - 2][0]
        dc1 = path[i - 1][1] - path[i - 2][1]
        dr2 = path[i][0] - path[i - 1][0]
        dc2 = path[i][1] - path[i - 1][1]
        if (dr1, dc1) != (dr2, dc2):
            turns += 1
    return turns


def _can_step(
    cell: tuple[int, int],
    prev: tuple[int, int],
    used: set[tuple[int, int]],
    blocked: np.ndarray,
) -> bool:
    r, c = cell
    h, w = blocked.shape
    if not (0 <= r < h and 0 <= c < w):
        return False
    if bool(blocked[r, c]) or cell in used:
        return False
    adjacent_used = [nbr for nbr in neighbors4(r, c) if nbr in used]
    return len(adjacent_used) == 1 and adjacent_used[0] == prev


def _future_degree(
    cell: tuple[int, int],
    used: set[tuple[int, int]],
    blocked: np.ndarray,
) -> int:
    expanded = set(used)
    expanded.add(cell)
    return sum(_can_step(nbr, cell, expanded, blocked) for nbr in neighbors4(*cell))


def _extend_path(
    path: list[tuple[int, int]],
    used: set[tuple[int, int]],
    target_len: int,
    blocked: np.ndarray,
    min_turns: int,
) -> bool:
    if len(path) == target_len:
        return _turn_count(path) >= min_turns

    cur = path[-1]
    candidates: list[tuple[int, float, tuple[int, int]]] = []
    for cell in neighbors4(*cur):
        if _can_step(cell, cur, used, blocked):
            score = _future_degree(cell, used, blocked)
            candidates.append((score, random.random(), cell))
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)

    for _score, _tie, cell in candidates:
        path.append(cell)
        used.add(cell)
        if _extend_path(path, used, target_len, blocked, min_turns):
            return True
        used.remove(cell)
        path.pop()
    return False


def _build_path(
    blocked: np.ndarray,
    target_len: int,
    min_turns: int,
) -> list[tuple[int, int]] | None:
    free = [tuple(map(int, cell)) for cell in np.argwhere(~blocked)]
    random.shuffle(free)
    for start in free[: min(len(free), 90)]:
        path = [start]
        used = {start}
        if _extend_path(path, used, target_len, blocked, min_turns):
            return path
    return None


def _legend_rows(height: int, count: int, top: bool) -> list[int]:
    if top:
        return [2 * i for i in range(count)]
    return [height - 1 - 2 * i for i in range(count)]


def _place_legend(
    grid: np.ndarray,
    color_to_k: dict[int, int],
    corner: str,
) -> list[tuple[int, int]]:
    h, w = grid.shape
    colors = list(color_to_k)
    random.shuffle(colors)
    top = corner[0] == "t"
    left = corner[1] == "l"
    rows = _legend_rows(h, len(colors), top=top)

    legend_cells: list[tuple[int, int]] = []
    for row, color in zip(rows, colors):
        k = int(color_to_k[color])
        if left:
            cols = range(0, k)
        else:
            cols = range(w - k, w)
        for col in cols:
            grid[row, col] = color
            legend_cells.append((row, col))
    return legend_cells


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    grid = np.full((h, w), BLACK, dtype=int)

    path_count = random.choice(PATH_COUNT_OPTIONS)
    path_colors = random.sample(list(PATH_COLORS), path_count)
    color_to_k = {color: random.randint(*LEGEND_K_RANGE) for color in path_colors}
    corner = random.choice(("tl", "tr", "bl", "br"))
    legend_cells = _place_legend(grid, color_to_k, corner)

    blocked = np.zeros((h, w), dtype=bool)
    _mark_blocked(blocked, legend_cells, radius=2)

    target_turns = random.choice((0, 1, 2))
    specs = []
    for color in path_colors:
        specs.append(
            {
                "color": color,
                "length": random.randint(*PATH_LEN_RANGE),
                "k": color_to_k[color],
                "min_turns": max(0, target_turns + random.choice((-1, 0, 1))),
            }
        )
    specs.sort(key=lambda item: int(item["length"]), reverse=True)

    for spec in specs:
        path = _build_path(blocked, int(spec["length"]), int(spec["min_turns"]))
        if path is None:
            return None
        if random.random() < 0.5:
            path = list(reversed(path))

        grid[path[0]] = GREY
        for cell in path[1:]:
            grid[cell] = int(spec["color"])
        _mark_blocked(blocked, path, radius=1)

    try:
        bg, paths = _extract_structure(grid)
        assert bg == BLACK
        assert len(paths) == path_count
        _ = main(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
