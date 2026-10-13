from __future__ import annotations

from collections import deque
from math import gcd
from pathlib import Path
import random
import sys

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, bbox_coords, components_mask, get_bg, in_bounds, neighbors4


def _gcd_all(values: list[int]) -> int:
    vals = [abs(int(v)) for v in values if int(v) > 0]
    assert vals, "legend lattice needs at least one positive spacing"
    step = vals[0]
    for v in vals[1:]:
        step = gcd(step, int(v))
    return int(step)


def _fill_block(grid: np.ndarray, top: tuple[int, int], size: int, color: int) -> None:
    r0, c0 = map(int, top)
    side = int(size)
    grid[r0 : r0 + side, c0 : c0 + side] = int(color)


def _bfs(passable: np.ndarray, start: tuple[int, int]) -> dict[tuple[int, int], int]:
    dist = {start: 0}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            nxt = (int(nr), int(nc))
            if in_bounds(passable.shape, nr, nc) and bool(passable[nr, nc]) and nxt not in dist:
                dist[nxt] = dist[(r, c)] + 1
                q.append(nxt)
    return dist


def _ordered_tree_path(
    passable: np.ndarray, start: tuple[int, int], end: tuple[int, int]
) -> list[tuple[int, int]] | None:
    dist = _bfs(passable, start)
    if end not in dist:
        return None
    out = [end]
    cur = end
    while cur != start:
        prevs = []
        for nr, nc in neighbors4(*cur):
            nxt = (int(nr), int(nc))
            if nxt in dist and dist[nxt] == dist[cur] - 1:
                prevs.append(nxt)
        if len(prevs) != 1:
            return None
        cur = prevs[0]
        out.append(cur)
    out.reverse()
    return out


def _tree_path_nodes(
    passable: np.ndarray, start: tuple[int, int], end: tuple[int, int]
) -> tuple[set[tuple[int, int]] | None, set[tuple[int, int]] | None]:
    if not bool(passable[start]) or not bool(passable[end]) or start == end:
        return None, None
    dist_s = _bfs(passable, start)
    if end not in dist_s:
        return None, None
    component = set(dist_s)
    edge_count = 0
    for r, c in component:
        for nr, nc in neighbors4(r, c):
            if (int(nr), int(nc)) in component:
                edge_count += 1
    edge_count //= 2
    if edge_count != len(component) - 1:
        return None, None

    dist_e = _bfs(passable, end)
    path = {
        cell
        for cell in component
        if cell in dist_e and dist_s[cell] + dist_e[cell] == dist_s[end]
    }
    for cell in path:
        deg = sum(((int(nr), int(nc)) in path) for nr, nc in neighbors4(*cell))
        if cell == start or cell == end:
            if deg != 1:
                return None, None
        elif deg != 2:
            return None, None
    return path, component


def _extract_anchor_block(grid: np.ndarray) -> tuple[int, int, int]:
    squares: list[tuple[int, int, int]] = []
    for coords in components_mask(grid == Color.GREY, connectivity=4):
        if int(coords.shape[0]) <= 1:
            continue
        r0, c0, r1, c1 = bbox_coords(coords)
        side_r = int(r1 - r0 + 1)
        side_c = int(c1 - c0 + 1)
        if side_r != side_c:
            continue
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if int(coords.shape[0]) == side_r * side_c and bool(np.all(patch == Color.GREY)):
            squares.append((int(r0), int(c0), int(side_r)))
    assert len(squares) == 1, "input must contain exactly one grey obstacle block"
    return squares[0]


def _extract_legend(
    grid: np.ndarray, bg: int, anchor_block: tuple[int, int, int]
) -> tuple[
    dict[tuple[int, int], int],
    int,
    int,
    tuple[int, int],
    tuple[int, int],
    list[tuple[int, int, int]],
]:
    ar0, ac0, side = anchor_block
    pixels: list[tuple[int, int, int]] = []
    for r, c in np.argwhere(grid != bg):
        rr = int(r)
        cc = int(c)
        if ar0 <= rr < ar0 + side and ac0 <= cc < ac0 + side:
            continue
        pixels.append((rr, cc, int(grid[rr, cc])))
    assert pixels, "legend pixels not found"

    row_vals = sorted({r for r, _, _ in pixels})
    col_vals = sorted({c for _, c, _ in pixels})
    gaps = [row_vals[i + 1] - row_vals[i] for i in range(len(row_vals) - 1)]
    gaps += [col_vals[i + 1] - col_vals[i] for i in range(len(col_vals) - 1)]
    step = _gcd_all(gaps)
    r_min = min(row_vals)
    c_min = min(col_vals)

    legend: dict[tuple[int, int], int] = {}
    for r, c, color in pixels:
        dr = int(r - r_min)
        dc = int(c - c_min)
        assert dr % step == 0 and dc % step == 0, "legend pixels must lie on one lattice"
        cell = (dr // step, dc // step)
        assert cell not in legend, "legend lattice cell reused"
        legend[cell] = int(color)

    height = max(r for r, _ in legend) + 1
    width = max(c for _, c in legend) + 1
    starts = [cell for cell, color in legend.items() if color == Color.RED]
    ends = [cell for cell, color in legend.items() if color == Color.BLUE]
    assert len(starts) == 1 and len(ends) == 1, "legend must contain one red and one blue endpoint"
    return legend, int(height), int(width), starts[0], ends[0], pixels


def _mapping_candidates(
    grid_shape: tuple[int, int],
    anchor_block: tuple[int, int, int],
    board_shape: tuple[int, int],
    obstacle_cells: set[tuple[int, int]],
    legend_pixels: list[tuple[int, int, int]],
) -> list[tuple[int, int]]:
    H, W = map(int, grid_shape)
    ar0, ac0, side = anchor_block
    rows, cols = map(int, board_shape)
    board_h = rows * side
    board_w = cols * side
    out: list[tuple[int, int]] = []
    for rr, cc in sorted(obstacle_cells):
        base_r = int(ar0 - rr * side)
        base_c = int(ac0 - cc * side)
        if base_r < 0 or base_c < 0 or base_r + board_h > H or base_c + board_w > W:
            continue
        overlap = False
        for lr, lc, _ in legend_pixels:
            if base_r <= lr < base_r + board_h and base_c <= lc < base_c + board_w:
                overlap = True
                break
        if not overlap:
            out.append((base_r, base_c))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild a scaled obstacle board from a top-left minimap and color the unique safe route.

    Input structure:
    - Black is the background.
    - The top-left corner contains a sparse legend on a regular dot lattice: grey pixels are obstacles, one red pixel is the start, and one blue pixel is the end.
    - Elsewhere, the main board consists of larger k×k block cells, but only one grey obstacle block is already drawn.

    Rule:
    - Detect the single drawn grey block to recover the block size.
    - Match that block to the only legend alignment whose full scaled board fits the canvas without covering the legend.
    - Project every legend obstacle/start/end cell onto the main block board, then find the unique 4-connected route between the red and blue cells through non-obstacle cells.
    - Paint the non-endpoint cells on that route yellow and keep the legend unchanged.

    Key insight:
    - The small dot pattern is a minimap of the larger block board, and the lone pre-drawn obstacle reveals the scale and offset needed to reconstruct the full maze before tracing its unique path.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    out = g.copy()

    anchor_block = _extract_anchor_block(g)
    legend, rows, cols, start, end, legend_pixels = _extract_legend(g, bg, anchor_block)
    obstacles = {cell for cell, color in legend.items() if color == Color.GREY}
    assert obstacles, "legend must contain obstacles"

    candidates = _mapping_candidates(g.shape, anchor_block, (rows, cols), obstacles, legend_pixels)
    assert len(candidates) == 1, "legend-to-board alignment must be unique"
    base_r, base_c = candidates[0]
    side = int(anchor_block[2])

    passable = np.ones((rows, cols), dtype=bool)
    for cell in obstacles:
        passable[cell] = False
    path_nodes, _component = _tree_path_nodes(passable, start, end)
    assert path_nodes is not None, "main board must contain a unique safe route"

    for cell in obstacles:
        _fill_block(out, (base_r + cell[0] * side, base_c + cell[1] * side), side, Color.GREY)
    _fill_block(out, (base_r + start[0] * side, base_c + start[1] * side), side, Color.RED)
    _fill_block(out, (base_r + end[0] * side, base_c + end[1] * side), side, Color.BLUE)
    for cell in path_nodes:
        if cell == start or cell == end:
            continue
        _fill_block(out, (base_r + cell[0] * side, base_c + cell[1] * side), side, Color.YELLOW)
    return out


_PLACEMENTS = ("left", "right")
_LEGEND_STEPS = (2, 3)
_BLOCK_SIZES = (2, 3, 4)
_GRID_SIDE_RANGE = range(4, 7)
_ORIGIN_CHOICES = (1, 2)
_GAP_RANGE = range(2, 7)
_MAX_ATTEMPTS = 600


def _degree(cell: tuple[int, int], cells: set[tuple[int, int]]) -> int:
    return sum(((int(nr), int(nc)) in cells) for nr, nc in neighbors4(*cell))


def _growth_candidates(
    shape: tuple[int, int],
    open_cells: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    rows, cols = map(int, shape)
    out: set[tuple[int, int]] = set()
    for cell in open_cells:
        for nr, nc in neighbors4(*cell):
            nxt = (int(nr), int(nc))
            if not in_bounds((rows, cols), nr, nc) or nxt in blocked or nxt in open_cells:
                continue
            if _degree(nxt, open_cells) == 1:
                out.add(nxt)
    return sorted(out)


def _grow_tree(
    shape: tuple[int, int],
    blocked: set[tuple[int, int]],
    target_size: int,
) -> set[tuple[int, int]] | None:
    rows, cols = map(int, shape)
    seeds = [
        (r, c)
        for r in range(rows)
        for c in range(cols)
        if (r, c) not in blocked
    ]
    if len(seeds) < target_size:
        return None
    open_cells = {random.choice(seeds)}
    while len(open_cells) < target_size:
        candidates = _growth_candidates((rows, cols), open_cells, blocked)
        if not candidates:
            return None
        open_cells.add(random.choice(candidates))
    return open_cells


def _ordered_path_from_tree(
    open_cells: set[tuple[int, int]], start: tuple[int, int], end: tuple[int, int], shape: tuple[int, int]
) -> list[tuple[int, int]] | None:
    mask = np.zeros(shape, dtype=bool)
    for cell in open_cells:
        mask[cell] = True
    return _ordered_tree_path(mask, start, end)


def _count_turns(path: list[tuple[int, int]]) -> int:
    turns = 0
    for idx in range(1, len(path) - 1):
        dr1 = path[idx][0] - path[idx - 1][0]
        dc1 = path[idx][1] - path[idx - 1][1]
        dr2 = path[idx + 1][0] - path[idx][0]
        dc2 = path[idx + 1][1] - path[idx][1]
        if (dr1, dc1) != (dr2, dc2):
            turns += 1
    return int(turns)


def _choose_endpoints(
    open_cells: set[tuple[int, int]], shape: tuple[int, int]
) -> tuple[tuple[int, int], tuple[int, int], set[tuple[int, int]], list[tuple[int, int]]] | None:
    leaves = [cell for cell in open_cells if _degree(cell, open_cells) == 1]
    if len(leaves) < 3:
        return None
    mask = np.zeros(shape, dtype=bool)
    for cell in open_cells:
        mask[cell] = True

    pairs: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    for idx, start in enumerate(leaves):
        dist = _bfs(mask, start)
        for end in leaves[idx + 1 :]:
            if end in dist:
                pairs.append((int(dist[end]), start, end))
    if not pairs:
        return None
    pairs.sort(reverse=True, key=lambda item: item[0])
    top_band = [item for item in pairs if item[0] >= pairs[0][0] - 2]
    random.shuffle(top_band)
    for _dist, start, end in top_band[: min(8, len(top_band))]:
        path_nodes, component = _tree_path_nodes(mask, start, end)
        if path_nodes is None or component is None:
            continue
        if len(component) - len(path_nodes) < 2:
            continue
        ordered = _ordered_tree_path(mask, start, end)
        if ordered is None or len(ordered) < 5:
            continue
        if _count_turns(ordered) < 2:
            continue
        return start, end, path_nodes, ordered
    return None


def _border_colors_visible(
    rows: int,
    cols: int,
    obstacles: set[tuple[int, int]],
    start: tuple[int, int],
    end: tuple[int, int],
) -> bool:
    colored = set(obstacles)
    colored.add(start)
    colored.add(end)
    return (
        any(r == 0 for r, _ in colored)
        and any(r == rows - 1 for r, _ in colored)
        and any(c == 0 for _, c in colored)
        and any(c == cols - 1 for _, c in colored)
    )


def _coarse_layout() -> tuple[int, int, int, int, int, str] | None:
    rows = random.choice(tuple(_GRID_SIDE_RANGE))
    cols = random.choice(tuple(_GRID_SIDE_RANGE))
    legend_step = random.choice(_LEGEND_STEPS)
    block_size = random.choice(_BLOCK_SIZES)
    origin_r = random.choice(_ORIGIN_CHOICES)
    origin_c = random.choice(_ORIGIN_CHOICES)
    legend_h = 1 + (rows - 1) * legend_step
    legend_w = 1 + (cols - 1) * legend_step
    board_h = rows * block_size
    board_w = cols * block_size
    gap = random.choice(tuple(_GAP_RANGE))
    H = origin_r + legend_h + gap + board_h
    min_W = max(origin_c + legend_w, board_w)
    if H > 30 or min_W > 30:
        return None
    W = random.randint(min_W, 30)
    placement = random.choice(_PLACEMENTS)
    return rows, cols, legend_step, block_size, H, W, origin_r, origin_c, placement


def _build_coarse_board(
    rows: int, cols: int, placement: str
) -> tuple[set[tuple[int, int]], tuple[int, int], tuple[int, int], set[tuple[int, int]], list[tuple[int, int]]] | None:
    anchor = (rows - 1, 0) if placement == "left" else (rows - 1, cols - 1)
    blocked = {(0, 0), (0, cols - 1), anchor}
    total_free = rows * cols - len(blocked)
    if total_free < 10:
        return None

    low = max(8, total_free // 3)
    high = min(total_free - 2, (2 * total_free) // 3 + 1)
    if low > high:
        return None

    for _ in range(150):
        target_size = random.randint(low, high)
        open_cells = _grow_tree((rows, cols), blocked, target_size)
        if open_cells is None:
            continue
        chosen = _choose_endpoints(open_cells, (rows, cols))
        if chosen is None:
            continue
        start, end, path_nodes, ordered = chosen
        obstacles = {
            (r, c)
            for r in range(rows)
            for c in range(cols)
            if (r, c) not in open_cells
        }
        if len(obstacles) < 5 or anchor not in obstacles:
            continue
        if not _border_colors_visible(rows, cols, obstacles, start, end):
            continue
        return obstacles, start, end, path_nodes, ordered
    return None


def _render_legend(
    grid: np.ndarray,
    rows: int,
    cols: int,
    legend_step: int,
    origin_r: int,
    origin_c: int,
    obstacles: set[tuple[int, int]],
    start: tuple[int, int],
    end: tuple[int, int],
) -> None:
    for r, c in obstacles:
        grid[origin_r + r * legend_step, origin_c + c * legend_step] = Color.GREY
    grid[origin_r + start[0] * legend_step, origin_c + start[1] * legend_step] = Color.RED
    grid[origin_r + end[0] * legend_step, origin_c + end[1] * legend_step] = Color.BLUE


def _render_board(
    grid: np.ndarray,
    base_r: int,
    base_c: int,
    block_size: int,
    obstacles: set[tuple[int, int]],
    start: tuple[int, int],
    end: tuple[int, int],
    path_nodes: set[tuple[int, int]],
) -> None:
    for cell in obstacles:
        _fill_block(grid, (base_r + cell[0] * block_size, base_c + cell[1] * block_size), block_size, Color.GREY)
    _fill_block(grid, (base_r + start[0] * block_size, base_c + start[1] * block_size), block_size, Color.RED)
    _fill_block(grid, (base_r + end[0] * block_size, base_c + end[1] * block_size), block_size, Color.BLUE)
    for cell in path_nodes:
        if cell == start or cell == end:
            continue
        _fill_block(grid, (base_r + cell[0] * block_size, base_c + cell[1] * block_size), block_size, Color.YELLOW)


def _validate_example(
    inp: np.ndarray,
    expected: np.ndarray,
    rows: int,
    cols: int,
    block_size: int,
    obstacles: set[tuple[int, int]],
    start: tuple[int, int],
    end: tuple[int, int],
    path_nodes: set[tuple[int, int]],
) -> bool:
    anchor_block = _extract_anchor_block(inp)
    legend, got_rows, got_cols, got_start, got_end, legend_pixels = _extract_legend(inp, Color.BLACK, anchor_block)
    if (got_rows, got_cols) != (rows, cols) or got_start != start or got_end != end:
        return False
    if {cell for cell, color in legend.items() if color == Color.GREY} != obstacles:
        return False
    candidates = _mapping_candidates(inp.shape, anchor_block, (rows, cols), obstacles, legend_pixels)
    if len(candidates) != 1:
        return False

    passable = np.ones((rows, cols), dtype=bool)
    for cell in obstacles:
        passable[cell] = False
    got_path, component = _tree_path_nodes(passable, start, end)
    if got_path != path_nodes or component is None:
        return False
    if len(component) - len(path_nodes) < 2:
        return False
    if not np.array_equal(main(inp), expected):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    layout = _coarse_layout()
    if layout is None:
        return None
    rows, cols, legend_step, block_size, H, W, origin_r, origin_c, placement = layout
    board = _build_coarse_board(rows, cols, placement)
    if board is None:
        return None
    obstacles, start, end, path_nodes, ordered = board
    if len(ordered) < 5 or _count_turns(ordered) < 2:
        return None

    board_h = rows * block_size
    board_w = cols * block_size
    base_r = H - board_h
    base_c = 0 if placement == "left" else W - board_w

    inp = np.zeros((H, W), dtype=int)
    expected = np.zeros((H, W), dtype=int)
    _render_legend(inp, rows, cols, legend_step, origin_r, origin_c, obstacles, start, end)
    expected[:] = inp

    anchor = (rows - 1, 0) if placement == "left" else (rows - 1, cols - 1)
    _fill_block(inp, (base_r + anchor[0] * block_size, base_c + anchor[1] * block_size), block_size, Color.GREY)
    _render_board(expected, base_r, base_c, block_size, obstacles, start, end, path_nodes)

    if not _validate_example(inp, expected, rows, cols, block_size, obstacles, start, end, path_nodes):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid minimap-pathfinding ARC input")
