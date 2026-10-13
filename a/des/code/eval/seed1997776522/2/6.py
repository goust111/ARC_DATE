import random
from collections import deque

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
GREY = 5
MAROON = 9
REGION_COLORS = (1, 4, 6, 7, 8)
PIPE_COLORS = {MAROON, GREEN, RED}
SPECIAL_COLORS = {BLACK, RED, GREEN, GREY, MAROON}

MAX_GEN_ATTEMPTS = 300
MAX_PATH_ATTEMPTS = 300


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_gap_mask(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    gap = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != BLACK:
                continue
            if 0 < c < w - 1:
                vertical_wall = []
                for rr in (r - 1, r + 1):
                    if 0 <= rr < h:
                        vertical_wall.append(int(grid[rr, c]) == GREY)
                if (
                    int(grid[r, c - 1]) in PIPE_COLORS
                    and int(grid[r, c + 1]) in PIPE_COLORS
                    and vertical_wall
                    and all(vertical_wall)
                ):
                    gap[r, c] = True
                    continue
            if 0 < r < h - 1:
                horizontal_wall = []
                for cc in (c - 1, c + 1):
                    if 0 <= cc < w:
                        horizontal_wall.append(int(grid[r, cc]) == GREY)
                if (
                    int(grid[r - 1, c]) in PIPE_COLORS
                    and int(grid[r + 1, c]) in PIPE_COLORS
                    and horizontal_wall
                    and all(horizontal_wall)
                ):
                    gap[r, c] = True
    return gap


def _component_map(mask: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    h, w = mask.shape
    labels = -np.ones((h, w), dtype=int)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or int(labels[r, c]) != -1:
                continue
            idx = len(comps)
            q = deque([(r, c)])
            labels[r, c] = idx
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and int(labels[nr, nc]) == -1:
                        labels[nr, nc] = idx
                        q.append((nr, nc))
            comps.append(pts)
    return labels, comps


def _trace_pipe(grid: np.ndarray, gap_mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    path_pts = [(int(r), int(c)) for r, c in np.argwhere(np.isin(grid, list(PIPE_COLORS)))]
    assert path_pts, "expected pipe pixels"
    adj: dict[tuple[int, int], set[tuple[int, int]]] = {pt: set() for pt in path_pts}
    pt_set = set(path_pts)
    for r, c in path_pts:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if (nr, nc) in pt_set:
                adj[(r, c)].add((nr, nc))
            jr, jc = r + 2 * dr, c + 2 * dc
            mr, mc = r + dr, c + dc
            if 0 <= jr < h and 0 <= jc < w and (jr, jc) in pt_set and bool(gap_mask[mr, mc]):
                adj[(r, c)].add((jr, jc))
    starts = [(int(r), int(c)) for r, c in np.argwhere(grid == GREEN)]
    ends = [(int(r), int(c)) for r, c in np.argwhere(grid == RED)]
    assert len(starts) == 1 and len(ends) == 1, "expected unique entrance and exit"
    start = starts[0]
    end = ends[0]
    path = [start]
    seen = {start}
    prev = None
    cur = start
    while cur != end:
        nxts = [pt for pt in adj[cur] if pt != prev and pt not in seen]
        if not nxts and prev is not None and end in adj[cur]:
            nxts = [end]
        assert len(nxts) == 1, "pipe must be a single simple path"
        nxt = nxts[0]
        path.append(nxt)
        seen.add(nxt)
        prev, cur = cur, nxt
    return path


def _region_colors(grid: np.ndarray, gap_mask: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    barrier = (grid == GREY) | gap_mask
    labels, comps = _component_map(~barrier)
    comp_color: dict[int, int] = {}
    for idx, pts in enumerate(comps):
        vals = [
            int(grid[r, c])
            for r, c in pts
            if int(grid[r, c]) not in SPECIAL_COLORS
        ]
        assert vals, "each pipe region must contain visible region color"
        counts: dict[int, int] = {}
        for v in vals:
            counts[v] = counts.get(v, 0) + 1
        comp_color[idx] = max(counts.items(), key=lambda kv: kv[1])[0]
    return labels, comp_color


def _inward_dir(exit_pt: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int]:
    r, c = exit_pt
    h, w = shape
    on_top = r == 0
    on_bottom = r == h - 1
    on_left = c == 0
    on_right = c == w - 1
    assert sum((on_top, on_bottom, on_left, on_right)) == 1, "exit must be on exactly one edge"
    if on_top:
        return 1, 0
    if on_bottom:
        return -1, 0
    if on_left:
        return 0, 1
    return 0, -1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace the broken pipe through grey-separated regions and emit the visited
    region colors as a bar just inside the red exit.
    """
    g = np.asarray(grid, dtype=int)
    gap_mask = _find_gap_mask(g)
    path = _trace_pipe(g, gap_mask)
    labels, comp_color = _region_colors(g, gap_mask)

    seq: list[int] = []
    prev_comp = None
    for r, c in path:
        comp = int(labels[r, c])
        assert comp != -1
        if comp != prev_comp:
            seq.append(int(comp_color[comp]))
            prev_comp = comp

    out = g.copy()
    exit_pt = path[-1]
    dr, dc = _inward_dir(exit_pt, g.shape)
    for i, color in enumerate(seq, start=1):
        rr = exit_pt[0] + dr * i
        cc = exit_pt[1] + dc * i
        assert 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1], "output bar must fit inside grid"
        out[rr, cc] = int(color)
    return out


def _row_start(row_heights: list[int], r: int) -> int:
    return sum(row_heights[:r]) + r


def _col_start(col_widths: list[int], c: int) -> int:
    return sum(col_widths[:c]) + c


def _cell_bounds(row_heights: list[int], col_widths: list[int], cell: tuple[int, int]) -> tuple[int, int, int, int]:
    r, c = cell
    r0 = _row_start(row_heights, r)
    c0 = _col_start(col_widths, c)
    return r0, c0, r0 + row_heights[r] - 1, c0 + col_widths[c] - 1


def _perimeter_cells(rows: int, cols: int) -> list[tuple[int, int]]:
    pts = []
    for r in range(rows):
        for c in range(cols):
            if r in (0, rows - 1) or c in (0, cols - 1):
                pts.append((r, c))
    return pts


def _step_dir(a: tuple[int, int], b: tuple[int, int]) -> str:
    if b[0] == a[0]:
        return "E" if b[1] > a[1] else "W"
    return "S" if b[0] > a[0] else "N"


def _compatible_border_sides(cell: tuple[int, int], move_dir: str, rows: int, cols: int) -> list[str]:
    r, c = cell
    sides: list[str] = []
    if move_dir in ("E", "W"):
        if r == 0:
            sides.append("top")
        if r == rows - 1:
            sides.append("bottom")
    else:
        if c == 0:
            sides.append("left")
        if c == cols - 1:
            sides.append("right")
    return sides


def _random_cell_path(rng: random.Random, rows: int, cols: int) -> tuple[list[tuple[int, int]], str, str] | None:
    perimeter = _perimeter_cells(rows, cols)
    for _ in range(MAX_PATH_ATTEMPTS):
        target_len = rng.randint(3, min(5, rows * cols))
        require_turn = bool(rng.randint(0, 1))
        start = rng.choice(perimeter)
        stack: list[tuple[int, int]] = [start]
        seen = {start}

        def dfs() -> list[tuple[int, int]] | None:
            if len(stack) == target_len:
                if stack[-1] == stack[0]:
                    return None
                if stack[-1] not in perimeter:
                    return None
                dirs = [_step_dir(stack[i], stack[i + 1]) for i in range(len(stack) - 1)]
                if require_turn and len(set(dirs)) == 1:
                    return None
                if not _compatible_border_sides(stack[0], dirs[0], rows, cols):
                    return None
                if not _compatible_border_sides(stack[-1], dirs[-1], rows, cols):
                    return None
                return stack[:]
            cur = stack[-1]
            nbrs = []
            for nr, nc in neighbors4(*cur):
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in seen:
                    nbrs.append((nr, nc))
            rng.shuffle(nbrs)
            for nxt in nbrs:
                seen.add(nxt)
                stack.append(nxt)
                result = dfs()
                if result is not None:
                    return result
                stack.pop()
                seen.remove(nxt)
            return None

        path = dfs()
        if path is None:
            continue
        first_dir = _step_dir(path[0], path[1])
        last_dir = _step_dir(path[-2], path[-1])
        start_side = rng.choice(_compatible_border_sides(path[0], first_dir, rows, cols))
        end_side = rng.choice(_compatible_border_sides(path[-1], last_dir, rows, cols))
        return path, start_side, end_side
    return None


def _draw_straight(grid: np.ndarray, a: tuple[int, int], b: tuple[int, int], color: int) -> None:
    if a[0] == b[0]:
        r = a[0]
        c0, c1 = sorted((a[1], b[1]))
        grid[r, c0 : c1 + 1] = color
    else:
        c = a[1]
        r0, r1 = sorted((a[0], b[0]))
        grid[r0 : r1 + 1, c] = color


def _draw_connector(grid: np.ndarray, p1: tuple[int, int], p2: tuple[int, int], color: int, rng: random.Random) -> None:
    if p1[0] == p2[0] or p1[1] == p2[1]:
        _draw_straight(grid, p1, p2, color)
        return
    bend_a = (p1[0], p2[1])
    bend_b = (p2[0], p1[1])
    bend = rng.choice((bend_a, bend_b))
    _draw_straight(grid, p1, bend, color)
    _draw_straight(grid, bend, p2, color)


def _endpoint_point(
    rng: random.Random,
    row_heights: list[int],
    col_widths: list[int],
    cell: tuple[int, int],
    side: str,
) -> tuple[int, int]:
    r0, c0, r1, c1 = _cell_bounds(row_heights, col_widths, cell)
    if side == "top":
        return r0, rng.randint(c0 + 1, c1 - 1)
    if side == "bottom":
        return r1, rng.randint(c0 + 1, c1 - 1)
    if side == "left":
        return rng.randint(r0 + 1, r1 - 1), c0
    return rng.randint(r0 + 1, r1 - 1), c1


def _internal_lane(
    rng: random.Random,
    row_heights: list[int],
    col_widths: list[int],
    cell_a: tuple[int, int],
    cell_b: tuple[int, int],
    forced_side: str | None,
) -> int:
    if cell_a[0] == cell_b[0]:
        r0, _, r1, _ = _cell_bounds(row_heights, col_widths, cell_a)
        if forced_side == "top":
            return r0
        if forced_side == "bottom":
            return r1
        return rng.randint(r0 + 1, r1 - 1)
    _, c0, _, c1 = _cell_bounds(row_heights, col_widths, cell_a)
    if forced_side == "left":
        return c0
    if forced_side == "right":
        return c1
    return rng.randint(c0 + 1, c1 - 1)


def _adjacency_ports(
    row_heights: list[int],
    col_widths: list[int],
    cell_a: tuple[int, int],
    cell_b: tuple[int, int],
    lane: int,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    a_r0, a_c0, a_r1, a_c1 = _cell_bounds(row_heights, col_widths, cell_a)
    b_r0, b_c0, b_r1, b_c1 = _cell_bounds(row_heights, col_widths, cell_b)
    if cell_a[0] == cell_b[0]:
        if cell_b[1] > cell_a[1]:
            gap = (lane, a_c1 + 1)
            return (lane, a_c1), gap, (lane, b_c0)
        gap = (lane, b_c1 + 1)
        return (lane, a_c0), gap, (lane, b_c1)
    if cell_b[0] > cell_a[0]:
        gap = (a_r1 + 1, lane)
        return (a_r1, lane), gap, (b_r0, lane)
    gap = (b_r1 + 1, lane)
    return (a_r0, lane), gap, (b_r1, lane)


def _build_grid(rng: random.Random) -> np.ndarray | None:
    rows = rng.randint(3, 4)
    cols = rng.randint(3, 4)
    row_heights = [6] * rows
    col_widths = [6] * cols

    path_info = _random_cell_path(rng, rows, cols)
    if path_info is None:
        return None
    cell_path, start_side, end_side = path_info
    seq_len = len(cell_path)
    end_r, end_c = cell_path[-1]
    if end_side in ("top", "bottom") and row_heights[end_r] - 1 < seq_len:
        return None
    if end_side in ("left", "right") and col_widths[end_c] - 1 < seq_len:
        return None

    h = sum(row_heights) + rows - 1
    w = sum(col_widths) + cols - 1
    grid = np.full((h, w), BLACK, dtype=int)

    sep_rows = set(_row_start(row_heights, r) - 1 for r in range(1, rows))
    sep_cols = set(_col_start(col_widths, c) - 1 for c in range(1, cols))
    for r in sep_rows:
        grid[r, :] = GREY
    for c in sep_cols:
        grid[:, c] = GREY

    visit_colors = list(rng.sample(REGION_COLORS, seq_len))
    cell_colors: dict[tuple[int, int], int] = {}
    for idx, cell in enumerate(cell_path):
        cell_colors[cell] = visit_colors[idx]
    for r in range(rows):
        for c in range(cols):
            cell = (r, c)
            if cell not in cell_colors:
                cell_colors[cell] = rng.choice(REGION_COLORS)
            r0, c0, r1, c1 = _cell_bounds(row_heights, col_widths, cell)
            grid[r0 : r1 + 1, c0 : c1 + 1] = int(cell_colors[cell])

    start_pt = _endpoint_point(rng, row_heights, col_widths, cell_path[0], start_side)
    end_pt = _endpoint_point(rng, row_heights, col_widths, cell_path[-1], end_side)

    ports_prev: dict[int, tuple[int, int]] = {}
    ports_next: dict[int, tuple[int, int]] = {}
    holes: list[tuple[int, int]] = []
    for i in range(len(cell_path) - 1):
        forced = None
        if i == 0:
            forced = start_side
        elif i == len(cell_path) - 2:
            forced = end_side
        lane = _internal_lane(rng, row_heights, col_widths, cell_path[i], cell_path[i + 1], forced)
        port_a, gap, port_b = _adjacency_ports(row_heights, col_widths, cell_path[i], cell_path[i + 1], lane)
        ports_next[i] = port_a
        ports_prev[i + 1] = port_b
        holes.append(gap)

    for hole in holes:
        grid[hole] = BLACK

    for i, cell in enumerate(cell_path):
        if i == 0:
            p1 = start_pt
            p2 = ports_next[i]
        elif i == len(cell_path) - 1:
            p1 = ports_prev[i]
            p2 = end_pt
        else:
            p1 = ports_prev[i]
            p2 = ports_next[i]
        _draw_connector(grid, p1, p2, MAROON, rng)

    grid[start_pt] = GREEN
    grid[end_pt] = RED

    if np.count_nonzero(grid == BLACK) != len(holes):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _build_grid(rng)
        if grid is None:
            continue
        out = main(grid)
        if out.shape != grid.shape:
            continue
        if np.array_equal(out, grid):
            continue
        return grid
    raise RuntimeError("generate_input: exhausted attempts")
