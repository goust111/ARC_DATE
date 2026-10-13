import numpy as np


BLACK = 0
GREY = 5
ORANGE = 7
PALETTE_COLORS = [1, 2, 3, 4, 6, 8, 9]
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

RNG = np.random.default_rng()
MAX_GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIR4:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _extract_legend(grid: np.ndarray) -> tuple[int, list[int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert w >= 2, "grid must have room for a separator and a legend"

    legend_col = w - 1
    legend_rows = [
        int(r)
        for r in np.flatnonzero((g[:, legend_col] != BLACK) & (g[:, legend_col] != ORANGE))
    ]
    assert legend_rows, "expected a non-empty legend in the rightmost column"

    sep_candidates = [
        c for c in range(w - 1) if all(int(g[r, c]) == ORANGE for r in legend_rows)
    ]
    assert sep_candidates, "expected an orange separator column left of the legend"
    sep_col = int(max(sep_candidates))

    legend = [int(g[r, legend_col]) for r in legend_rows]
    assert len(legend) == len(set(legend)), "legend colors must be distinct"
    return sep_col, legend


def _degrees(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    deg = np.zeros((h, w), dtype=int)
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        deg[rr, cc] = sum(1 for nr, nc in _neighbors4(rr, cc, h, w) if mask[nr, nc])
    return deg


def _trace_dead_end_paths(mask: np.ndarray) -> tuple[list[list[tuple[int, int]]], np.ndarray] | None:
    maze = np.asarray(mask, dtype=bool)
    h, w = maze.shape
    if not maze.any():
        return None

    deg = _degrees(maze)
    dead_ends = [(int(r), int(c)) for r, c in np.argwhere(maze & (deg == 1))]
    if not dead_ends:
        return None

    paths: list[list[tuple[int, int]]] = []
    for start in dead_ends:
        path = [start]
        prev: tuple[int, int] | None = None
        cur = start
        for _ in range(maze.size):
            nxts = [
                (nr, nc)
                for nr, nc in _neighbors4(cur[0], cur[1], h, w)
                if maze[nr, nc] and ((nr, nc) != prev if prev is not None else True)
            ]
            if not nxts:
                return None
            if len(nxts) > 1:
                return None

            nxt = nxts[0]
            if int(deg[nxt]) >= 3:
                break
            path.append(nxt)
            prev, cur = cur, nxt
        else:
            return None
        paths.append(path)

    return paths, deg


def _paint_paths(
    out: np.ndarray,
    paths: list[list[tuple[int, int]]],
    legend: list[int],
    row_offset: int = 0,
    col_offset: int = 0,
) -> np.ndarray:
    painted = np.array(out, copy=True)
    for path in paths:
        color = int(legend[min(len(path), len(legend)) - 1])
        for r, c in path:
            painted[row_offset + r, col_offset + c] = color
    return painted


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color every dead-end branch in the maze by its backtrack depth to the nearest junction.
    """
    g = np.asarray(grid, dtype=int)
    sep_col, legend = _extract_legend(g)
    maze = g[:, :sep_col] == GREY

    traced = _trace_dead_end_paths(maze)
    assert traced is not None, "maze must contain dead ends that reach a junction"
    paths, _deg = traced
    return _paint_paths(g, paths, legend)


def _connected(mask: np.ndarray) -> bool:
    maze = np.asarray(mask, dtype=bool)
    cells = np.argwhere(maze)
    if cells.size == 0:
        return False

    h, w = maze.shape
    start = tuple(map(int, cells[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c, h, w):
            if maze[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(maze.sum())


def _carve_perfect_maze(cell_h: int, cell_w: int) -> np.ndarray:
    maze_h = 2 * cell_h + 1
    maze_w = 2 * cell_w + 1
    mask = np.zeros((maze_h, maze_w), dtype=bool)
    visited = np.zeros((cell_h, cell_w), dtype=bool)

    sr = int(RNG.integers(cell_h))
    sc = int(RNG.integers(cell_w))
    stack = [(sr, sc)]
    visited[sr, sc] = True
    mask[2 * sr + 1, 2 * sc + 1] = True

    while stack:
        r, c = stack[-1]
        nbrs = []
        for dr, dc in DIR4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < cell_h and 0 <= nc < cell_w and not visited[nr, nc]:
                nbrs.append((nr, nc, dr, dc))
        if not nbrs:
            stack.pop()
            continue

        nr, nc, dr, dc = nbrs[int(RNG.integers(len(nbrs)))]
        mask[2 * r + 1 + dr, 2 * c + 1 + dc] = True
        mask[2 * nr + 1, 2 * nc + 1] = True
        visited[nr, nc] = True
        stack.append((nr, nc))

    return mask


def _add_short_branches(mask: np.ndarray) -> np.ndarray:
    maze = np.array(mask, copy=True)
    target_adds = int(RNG.integers(0, 4))
    for _ in range(target_adds):
        deg = _degrees(maze)
        candidates: list[tuple[int, int]] = []
        for r, c in np.argwhere(maze & ((deg == 2) | (deg == 3))):
            rr = int(r)
            cc = int(c)
            for nr, nc in _neighbors4(rr, cc, maze.shape[0], maze.shape[1]):
                if maze[nr, nc]:
                    continue
                if sum(1 for tr, tc in _neighbors4(nr, nc, maze.shape[0], maze.shape[1]) if maze[tr, tc]) == 1:
                    candidates.append((nr, nc))
        if not candidates:
            break
        ar, ac = candidates[int(RNG.integers(len(candidates)))]
        maze[ar, ac] = True
    return maze


def _analyze_maze(mask: np.ndarray) -> tuple[list[list[tuple[int, int]]], list[int]] | None:
    maze = np.asarray(mask, dtype=bool)
    if not _connected(maze):
        return None

    traced = _trace_dead_end_paths(maze)
    if traced is None:
        return None
    paths, deg = traced

    junction_count = int(np.count_nonzero(maze & (deg >= 3)))
    dead_end_count = int(np.count_nonzero(maze & (deg == 1)))
    if junction_count == 0 or dead_end_count < 3:
        return None

    depths = [len(path) for path in paths]
    if len(set(depths)) < 2:
        return None

    covered = [cell for path in paths for cell in path]
    if len(covered) != len(set(covered)):
        return None
    if len(covered) >= int(maze.sum()):
        return None

    return paths, depths


def _choose_maze(target_overflow: bool) -> tuple[np.ndarray, list[list[tuple[int, int]]], list[int]] | None:
    if target_overflow:
        cell_h = int(RNG.integers(4, 8))
        cell_w = int(RNG.integers(4, 8))
    else:
        cell_h = int(RNG.integers(3, 5))
        cell_w = int(RNG.integers(3, 5))

    maze = _carve_perfect_maze(cell_h, cell_w)
    maze = _add_short_branches(maze)
    analyzed = _analyze_maze(maze)
    if analyzed is None:
        return None

    paths, depths = analyzed
    max_depth = max(depths)
    if not target_overflow and max_depth > len(PALETTE_COLORS):
        return None
    if target_overflow and max_depth < 3:
        return None
    return maze, paths, depths


def _choose_legend(depths: list[int], target_overflow: bool) -> list[int] | None:
    max_depth = max(depths)
    max_len = len(PALETTE_COLORS)

    if target_overflow:
        hi = min(max_len, max_depth - 1)
        if hi < 2:
            return None
        legend_len = int(RNG.integers(2, hi + 1))
    else:
        if max_depth > max_len:
            return None
        lo = max(2, max_depth)
        if lo > max_len:
            return None
        legend_len = int(RNG.integers(lo, max_len + 1))

    colors = RNG.choice(PALETTE_COLORS, size=legend_len, replace=False)
    return [int(x) for x in colors]


def _compose_grid(
    maze: np.ndarray,
    paths: list[list[tuple[int, int]]],
    legend: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    maze_h, maze_w = map(int, maze.shape)
    legend_len = len(legend)

    extra_top = int(RNG.integers(0, 4))
    extra_bottom = int(RNG.integers(0, 4))
    height = max(maze_h, legend_len) + extra_top + extra_bottom

    left_margin = int(RNG.integers(0, 5))
    width = left_margin + maze_w + 2
    sep_col = width - 2
    top = int(RNG.integers(0, height - maze_h + 1))
    legend_top = int(RNG.integers(0, height - legend_len + 1))

    grid = np.full((height, width), BLACK, dtype=int)
    grid[:, sep_col] = ORANGE
    grid[top : top + maze_h, left_margin : left_margin + maze_w][maze] = GREY
    for idx, color in enumerate(legend):
        grid[legend_top + idx, width - 1] = int(color)

    expected = _paint_paths(grid, paths, legend, row_offset=top, col_offset=left_margin)
    return grid, expected


def _gen_attempt() -> np.ndarray | None:
    target_overflow = bool(int(RNG.integers(0, 2)))
    chosen = _choose_maze(target_overflow)
    if chosen is None:
        return None
    maze, paths, depths = chosen

    legend = _choose_legend(depths, target_overflow)
    if legend is None:
        return None
    if target_overflow and max(depths) <= len(legend):
        return None
    if not target_overflow and max(depths) > len(legend):
        return None

    grid, expected = _compose_grid(maze, paths, legend)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid dead-end maze sample")
