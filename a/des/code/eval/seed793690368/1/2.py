from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
MAROON = 9

PIPE_COLORS = {BLUE, RED}
MARKER_COLORS = {GREEN, YELLOW}
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GRID_H_RANGE = (12, 22)
GRID_W_RANGE = (14, 24)
NODE_COUNT_CHOICES = [1, 2, 2, 3, 3, 4, 5]
MAX_SEG_LEN = 5
MAX_DECOY_LEN = 3
MAX_DISTRACTOR_ARM = 3
MAX_NODE_TRIES = 120
MAX_DISTRACTOR_TRIES = 80
MAX_GEN_ATTEMPTS = 500


Coord = tuple[int, int]
State = tuple[Coord, int, Coord]


def _in_bounds(r: int, c: int, h: int, w: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _add(pt: Coord, d: Coord, steps: int = 1) -> Coord:
    return pt[0] + d[0] * steps, pt[1] + d[1] * steps


def _edge_key(a: Coord, b: Coord) -> tuple[Coord, Coord]:
    return (a, b) if a <= b else (b, a)


def _is_border(pt: Coord, h: int, w: int) -> bool:
    r, c = pt
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _distance_to_border(pt: Coord, d: Coord, h: int, w: int) -> int:
    r, c = pt
    dr, dc = d
    if dr == -1:
        return r
    if dr == 1:
        return h - 1 - r
    if dc == -1:
        return c
    return w - 1 - c


def _perpendicular_dirs(d: Coord) -> list[Coord]:
    dr, dc = d
    return [(dc, dr), (-dc, -dr)]


def _target_color(marker: int) -> int:
    assert marker in MARKER_COLORS
    return BLUE if marker == GREEN else RED


def _marker_for_target(color: int) -> int:
    assert color in PIPE_COLORS
    return GREEN if color == BLUE else YELLOW


def _build_output(grid: np.ndarray, visited_path: Iterable[Coord]) -> np.ndarray:
    out = np.full_like(grid, BG)
    keep_special = np.isin(grid, [GREEN, YELLOW, MAROON])
    out[keep_special] = grid[keep_special]
    for r, c in visited_path:
        if int(grid[r, c]) in PIPE_COLORS:
            out[r, c] = int(grid[r, c])
    return out


def _node_accepts_incoming(grid: np.ndarray, node: Coord, src: Coord, color: int) -> bool:
    h, w = grid.shape
    nr, nc = node
    sr, sc = src
    dr, dc = nr - sr, nc - sc
    far_r, far_c = nr + dr, nc + dc
    return _in_bounds(far_r, far_c, h, w) and int(grid[far_r, far_c]) == color


def _successors(grid: np.ndarray, state: State, visited: set[Coord]) -> list[State]:
    h, w = grid.shape
    (r, c), active_color, prev = state
    val = int(grid[r, c])

    if _is_border((r, c), h, w) and val != MAROON:
        return []

    next_states: list[State] = []
    if val in PIPE_COLORS:
        assert val == active_color
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            nxt = (nr, nc)
            if not _in_bounds(nr, nc, h, w) or nxt == prev or nxt in visited:
                continue
            nval = int(grid[nr, nc])
            if nval == active_color:
                next_states.append((nxt, active_color, (r, c)))
            elif nval in MARKER_COLORS and _node_accepts_incoming(grid, nxt, (r, c), active_color):
                next_states.append((nxt, active_color, (r, c)))
    elif val in MARKER_COLORS:
        pr, pc = prev
        incoming = (r - pr, c - pc)
        target_color = _target_color(val)
        for dr, dc in _perpendicular_dirs(incoming):
            nr, nc = r + dr, c + dc
            nxt = (nr, nc)
            if (
                _in_bounds(nr, nc, h, w)
                and nxt not in visited
                and int(grid[nr, nc]) == target_color
            ):
                next_states.append((nxt, target_color, (r, c)))
    return next_states


def _best_path(grid: np.ndarray, start_state: State) -> list[Coord]:
    best_path: list[Coord] = []
    best_score = (-1, -1, -1)
    h, w = grid.shape

    def dfs(state: State, visited: set[Coord], path: list[Coord]) -> None:
        nonlocal best_path, best_score

        pos, _active_color, _prev = state
        val = int(grid[pos])
        border_hit = int(_is_border(pos, h, w) and val != MAROON)
        pipe_count = sum(int(grid[r, c]) in PIPE_COLORS for r, c in path)

        succs = _successors(grid, state, visited)
        if border_hit or not succs:
            score = (border_hit, pipe_count, len(path))
            if score > best_score:
                best_score = score
                best_path = path[:]
            return

        for nxt in succs:
            nxt_pos = nxt[0]
            if nxt_pos in visited:
                continue
            visited.add(nxt_pos)
            path.append(nxt_pos)
            dfs(nxt, visited, path)
            path.pop()
            visited.remove(nxt_pos)

    start_pos, _color, start_prev = start_state
    dfs(start_state, {start_prev, start_pos}, [start_pos])
    return best_path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace a signal along two crossing pipe layers and keep only the traversed pipes.

    Input structure:
    - bg is black.
    - Blue and red pixels form orthogonal pipe segments.
    - Green and yellow center pixels mark crossing nodes.
    - One maroon pixel on the left edge marks the start.

    Rule:
    - Start from the pipe touching the maroon pixel.
    - Follow the current pipe color until a crossing marker is reached.
    - Green switches the active layer to blue; yellow switches it to red.
    - After switching, continue along the perpendicular branch of that layer.
    - Stop on reaching the grid edge or when no continuation exists.
    - Keep all marker pixels and the maroon start; erase every unvisited pipe.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    start_cells = np.argwhere(g == MAROON)
    assert start_cells.shape == (1, 2), "expected one maroon start pixel"
    sr, sc = map(int, start_cells[0])
    assert sc == 0, "start marker must be on the left edge"

    start_neighbors: list[Coord] = []
    for dr, dc in DIRS4:
        nr, nc = sr + dr, sc + dc
        if _in_bounds(nr, nc, *g.shape) and int(g[nr, nc]) in PIPE_COLORS:
            start_neighbors.append((nr, nc))
    assert len(start_neighbors) == 1, "start marker must touch exactly one pipe"

    first = start_neighbors[0]
    start_color = int(g[first])
    path = _best_path(g, (first, start_color, (sr, sc)))
    return _build_output(g, path)


def _can_merge(
    cells: dict[Coord, int],
    edges: set[tuple[Coord, Coord]],
    add_cells: dict[Coord, int],
    add_edges: list[tuple[Coord, Coord]],
    h: int,
    w: int,
) -> bool:
    merged_cells = dict(cells)
    for pt, color in add_cells.items():
        r, c = pt
        if not _in_bounds(r, c, h, w):
            return False
        if pt in merged_cells and merged_cells[pt] != color:
            return False
        merged_cells[pt] = color

    merged_edges = set(edges)
    for a, b in add_edges:
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1:
            return False
        if a not in merged_cells or b not in merged_cells:
            return False
        merged_edges.add(_edge_key(a, b))

    check_pts = set(add_cells)
    for pt in list(add_cells):
        r, c = pt
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            nb = (nr, nc)
            if _in_bounds(nr, nc, h, w) and nb in merged_cells:
                check_pts.add(nb)

    for pt in check_pts:
        if pt not in merged_cells:
            continue
        r, c = pt
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            nb = (nr, nc)
            if _in_bounds(nr, nc, h, w) and nb in merged_cells:
                if _edge_key(pt, nb) not in merged_edges:
                    return False
    return True


def _commit(
    cells: dict[Coord, int],
    edges: set[tuple[Coord, Coord]],
    add_cells: dict[Coord, int],
    add_edges: list[tuple[Coord, Coord]],
) -> None:
    cells.update(add_cells)
    for a, b in add_edges:
        edges.add(_edge_key(a, b))


def _segment_chain(start: Coord, d: Coord, cell_count: int) -> list[Coord]:
    return [_add(start, d, k) for k in range(cell_count)]


def _try_place_node(
    cells: dict[Coord, int],
    edges: set[tuple[Coord, Coord]],
    visited_path: set[Coord],
    h: int,
    w: int,
    current_pos: Coord,
    current_color: int,
    direction: Coord,
    remaining_nodes: int,
) -> tuple[Coord, int, Coord] | None:
    dist_in = _distance_to_border(current_pos, direction, h, w)
    if dist_in < 3:
        return None

    next_color = RED if current_color == BLUE else BLUE
    marker_color = _marker_for_target(next_color)
    min_seg = 1
    max_seg = min(MAX_SEG_LEN, dist_in - 2)
    if max_seg < min_seg:
        return None

    for _ in range(MAX_NODE_TRIES):
        seg_len = random.randint(min_seg, max_seg)
        node = _add(current_pos, direction, seg_len)
        if _distance_to_border(node, direction, h, w) < 2:
            continue

        out_options = []
        for out_dir in _perpendicular_dirs(direction):
            if _distance_to_border(node, out_dir, h, w) < 2:
                continue
            if _distance_to_border(node, (-out_dir[0], -out_dir[1]), h, w) < 2:
                continue
            next_start = _add(node, out_dir)
            if remaining_nodes > 0 and _distance_to_border(next_start, out_dir, h, w) < 3:
                continue
            out_options.append(out_dir)
        if not out_options:
            continue

        out_dir = random.choice(out_options)
        old_len = random.randint(1, min(MAX_DECOY_LEN, _distance_to_border(node, direction, h, w) - 1))
        decoy_dir = (-out_dir[0], -out_dir[1])
        new_len = random.randint(1, min(MAX_DECOY_LEN, _distance_to_border(node, decoy_dir, h, w) - 1))

        add_cells: dict[Coord, int] = {}
        add_edges: list[tuple[Coord, Coord]] = []

        segment = _segment_chain(current_pos, direction, seg_len)
        for pt in segment:
            add_cells[pt] = current_color
        for a, b in zip(segment, segment[1:]):
            add_edges.append((a, b))

        last_pipe = segment[-1]
        add_cells[node] = marker_color
        add_edges.append((last_pipe, node))

        old_decoy = _segment_chain(_add(node, direction), direction, old_len)
        for pt in old_decoy:
            add_cells[pt] = current_color
        if old_decoy:
            add_edges.append((node, old_decoy[0]))
        for a, b in zip(old_decoy, old_decoy[1:]):
            add_edges.append((a, b))

        actual_out = _add(node, out_dir)
        add_cells[actual_out] = next_color
        add_edges.append((node, actual_out))

        new_decoy = _segment_chain(_add(node, decoy_dir), decoy_dir, new_len)
        for pt in new_decoy:
            add_cells[pt] = next_color
        if new_decoy:
            add_edges.append((node, new_decoy[0]))
        for a, b in zip(new_decoy, new_decoy[1:]):
            add_edges.append((a, b))

        if not _can_merge(cells, edges, add_cells, add_edges, h, w):
            continue

        _commit(cells, edges, add_cells, add_edges)
        visited_path.update(segment)
        visited_path.add(actual_out)
        return actual_out, next_color, out_dir
    return None


def _try_place_terminal(
    cells: dict[Coord, int],
    edges: set[tuple[Coord, Coord]],
    visited_path: set[Coord],
    h: int,
    w: int,
    current_pos: Coord,
    current_color: int,
    direction: Coord,
) -> bool:
    mode = random.choice(["edge", "dead"])
    dist = _distance_to_border(current_pos, direction, h, w)

    for _ in range(MAX_NODE_TRIES):
        if mode == "edge":
            total_len = dist + 1
        else:
            if dist <= 0:
                mode = "edge"
                total_len = dist + 1
            else:
                total_len = random.randint(1, min(MAX_SEG_LEN + 1, dist))

        segment = _segment_chain(current_pos, direction, total_len)
        add_cells = {pt: current_color for pt in segment}
        add_edges = [(a, b) for a, b in zip(segment, segment[1:])]

        if _can_merge(cells, edges, add_cells, add_edges, h, w):
            _commit(cells, edges, add_cells, add_edges)
            visited_path.update(segment)
            return True
        if mode == "edge":
            return False
    return False


def _try_place_distractor(cells: dict[Coord, int], edges: set[tuple[Coord, Coord]], h: int, w: int) -> bool:
    for _ in range(MAX_DISTRACTOR_TRIES):
        center = (random.randint(2, h - 3), random.randint(2, w - 3))
        horiz_color = random.choice([BLUE, RED])
        vert_color = RED if horiz_color == BLUE else BLUE
        marker = random.choice([GREEN, YELLOW])

        left_len = random.randint(1, MAX_DISTRACTOR_ARM)
        right_len = random.randint(1, MAX_DISTRACTOR_ARM)
        up_len = random.randint(1, MAX_DISTRACTOR_ARM)
        down_len = random.randint(1, MAX_DISTRACTOR_ARM)

        if (
            _distance_to_border(center, (0, -1), h, w) <= left_len
            or _distance_to_border(center, (0, 1), h, w) <= right_len
            or _distance_to_border(center, (-1, 0), h, w) <= up_len
            or _distance_to_border(center, (1, 0), h, w) <= down_len
        ):
            continue

        add_cells: dict[Coord, int] = {center: marker}
        add_edges: list[tuple[Coord, Coord]] = []

        for d, length, color in [
            ((0, -1), left_len, horiz_color),
            ((0, 1), right_len, horiz_color),
            ((-1, 0), up_len, vert_color),
            ((1, 0), down_len, vert_color),
        ]:
            arm = _segment_chain(_add(center, d), d, length)
            for pt in arm:
                add_cells[pt] = color
            add_edges.append((center, arm[0]))
            for a, b in zip(arm, arm[1:]):
                add_edges.append((a, b))

        if not _can_merge(cells, edges, add_cells, add_edges, h, w):
            continue
        _commit(cells, edges, add_cells, add_edges)
        return True
    return False


def _render_grid(cells: dict[Coord, int], h: int, w: int) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for (r, c), color in cells.items():
        grid[r, c] = color
    return grid


def _generate_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)

    cells: dict[Coord, int] = {}
    edges: set[tuple[Coord, Coord]] = set()
    visited_path: set[Coord] = set()

    start_row = random.randint(2, h - 3)
    start_color = random.choice([BLUE, RED])
    start = (start_row, 0)
    first = (start_row, 1)
    cells[start] = MAROON
    cells[first] = start_color
    edges.add(_edge_key(start, first))
    visited_path.add(first)

    current_pos = first
    current_color = start_color
    direction = (0, 1)
    node_count = random.choice(NODE_COUNT_CHOICES)

    for node_idx in range(node_count):
        placed = _try_place_node(
            cells,
            edges,
            visited_path,
            h,
            w,
            current_pos,
            current_color,
            direction,
            node_count - node_idx - 1,
        )
        if placed is None:
            return None
        current_pos, current_color, direction = placed

    if not _try_place_terminal(cells, edges, visited_path, h, w, current_pos, current_color, direction):
        return None

    for _ in range(random.randint(0, 2)):
        _try_place_distractor(cells, edges, h, w)

    grid = _render_grid(cells, h, w)
    expected = _build_output(grid, visited_path)

    try:
        got = main(grid)
    except Exception:
        return None
    if not np.array_equal(got, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        sample = _generate_attempt()
        if sample is not None:
            return sample[0]
    raise RuntimeError("failed to generate a valid pipe-switching input")
