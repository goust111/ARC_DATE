"""Path tracing with intersection lookup and synthetic input generator for puzzle 4."""

from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


BLACK = 0
MAROON = 9

Direction = Tuple[int, int]
PathInfo = Tuple[int, int, int, int, Tuple[int, int]]

DIRS: Dict[str, Direction] = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}
DIR_LIST: Sequence[Direction] = list(DIRS.values())
DELTA_TO_DIR: Dict[Direction, str] = {v: k for k, v in DIRS.items()}

ARROW_PATTERNS: Dict[frozenset[Tuple[int, int]], str] = {
    frozenset({(0, 1), (1, 1), (1, 2)}): "right",
    frozenset({(0, 0), (0, 1), (1, 1)}): "down",
    frozenset({(0, 0), (1, 0), (1, 1)}): "left",
    frozenset({(0, 1), (1, 0), (1, 1)}): "up",
}



def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _neighbors(r: int, c: int) -> Iterable[Tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _norm_shape(cells: Iterable[Tuple[int, int]]) -> frozenset[Tuple[int, int]]:
    xs = list(cells)
    r0 = min(r for r, _ in xs)
    c0 = min(c for _, c in xs)
    return frozenset((int(r - r0), int(c - c0)) for r, c in xs)


def _is_l_shape(cells: Iterable[Tuple[int, int]]) -> bool:
    cs = [tuple(map(int, rc)) for rc in cells]
    if len(cs) != 3:
        return False
    s = set(cs)
    degrees: Dict[Tuple[int, int], int] = {p: 0 for p in cs}
    for r, c in cs:
        for nr, nc in _neighbors(r, c):
            if (nr, nc) in s:
                degrees[(r, c)] += 1
    degs = sorted(degrees.values())
    return degs == [1, 1, 2]


def _parse_legend(grid: np.ndarray) -> Dict[int, str]:
    h, w = grid.shape
    r_cap = min(h, 12)
    c_cap = min(w, 12)
    legend: Dict[int, str] = {}

    region = grid[:r_cap, :c_cap]
    colors = [int(c) for c in np.unique(region) if int(c) not in (BLACK, MAROON)]

    for color in colors:
        seen = np.zeros_like(region, dtype=bool)
        for sr in range(r_cap):
            for sc in range(c_cap):
                if seen[sr, sc] or int(region[sr, sc]) != color:
                    continue

                q = deque([(sr, sc)])
                seen[sr, sc] = True
                comp: list[Tuple[int, int]] = []
                while q:
                    r, c = q.popleft()
                    comp.append((int(r), int(c)))
                    for nr, nc in _neighbors(r, c):
                        if 0 <= nr < r_cap and 0 <= nc < c_cap and not seen[nr, nc] and int(region[nr, nc]) == color:
                            seen[nr, nc] = True
                            q.append((nr, nc))

                if len(comp) != 3 or not _is_l_shape(comp):
                    continue

                norm = _norm_shape(comp)
                if norm not in ARROW_PATTERNS:
                    continue

                comp_set = set(comp)
                has_key = False
                for r, c in comp:
                    for nr, nc in _neighbors(r, c):
                        if 0 <= nr < r_cap and 0 <= nc < c_cap and (nr, nc) not in comp_set and int(region[nr, nc]) == color:
                            has_key = True
                            break
                    if has_key:
                        break

                if not has_key:
                    continue

                legend[color] = ARROW_PATTERNS[norm]
                break

    return legend


def _find_start_blocks(grid: np.ndarray) -> List[PathInfo]:
    h, w = grid.shape
    starts: List[PathInfo] = []
    seen_blocks: set[Tuple[int, int]] = set()

    for r in range(h - 1):
        for c in range(w - 1):
            if (r, c) in seen_blocks:
                continue

            color = int(grid[r, c])
            if color in (BLACK, MAROON):
                continue
            block = grid[r : r + 2, c : c + 2]
            if not np.all(block == color):
                continue

            seen_blocks.add((r, c))
            block_cells = [(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)]
            neighbor_colors: Dict[int, int] = {}
            starts_adj: Dict[int, set[Tuple[int, int]]] = {}

            for br, bc in block_cells:
                for nr, nc in _neighbors(br, bc):
                    if not _in_bounds(h, w, nr, nc):
                        continue
                    v = int(grid[nr, nc])
                    if v == BLACK:
                        continue
                    neighbor_colors[v] = neighbor_colors.get(v, 0) + 1
                    starts_adj.setdefault(v, set()).add((int(nr), int(nc)))

            path_candidates = {v: cnt for v, cnt in neighbor_colors.items() if v not in (BLACK, MAROON, color)}
            if not path_candidates:
                continue

            path_color = int(max(path_candidates, key=lambda v: (path_candidates[v], -v)))
            start_options = sorted(starts_adj.get(path_color, set()))
            if not start_options:
                continue

            start_cell = start_options[0]
            starts.append((r, c, path_color, color, start_cell))

    return starts


def _trace_single_path(
    grid: np.ndarray,
    start_cell: Tuple[int, int],
    block_cells: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]],
    path_color: int,
    legend: Dict[int, str],
) -> Optional[Tuple[int, int]]:
    h, w = grid.shape
    block_set = set(block_cells)
    r, c = start_cell

    prev: Optional[Tuple[int, int]] = None
    current = (int(r), int(c))
    visited = set(block_set)
    visited.add(current)

    for _ in range(h * w + 20):
        cur_color = int(grid[current[0], current[1]])
        if cur_color == MAROON:
            return current

        if cur_color in legend:
            dr, dc = DIRS[legend[cur_color]]
            nr, nc = current[0] + dr, current[1] + dc
            if not _in_bounds(h, w, nr, nc):
                return None
            if int(grid[nr, nc]) == BLACK:
                return None
            nxt = (nr, nc)
            if nxt == prev:
                return None
        else:
            neighbors: list[Tuple[int, int]] = []
            for nr, nc in _neighbors(current[0], current[1]):
                if not _in_bounds(h, w, nr, nc):
                    continue
                v = int(grid[nr, nc])
                if v == BLACK:
                    continue
                if (nr, nc) in block_set:
                    continue
                if prev is not None and (nr, nc) == prev:
                    continue
                neighbors.append((nr, nc))

            if not neighbors:
                return None

            same = [p for p in neighbors if int(grid[p[0], p[1]]) == cur_color]
            if same:
                nxt = sorted(same)[0]
            else:
                nxt = sorted(neighbors)[0]

            if nxt == prev:
                return None

        if nxt in visited and int(grid[nxt[0], nxt[1]]) != MAROON:
            return None

        prev, current = current, nxt
        visited.add(current)

    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Task
    ----
    Move each 2x2 cargo block along its path to the maroon terminal, using legend
    instructions when crossing intersection colors.

    Input structure
    --------------
    - Black (0) is the background.
    - Paths are 1-cell-wide vertical/horizontal lines.
    - Each path starts at a 2x2 cargo block (non-black), and ends at a maroon (9)
      pixel.
    - A corner legend stores direction rules as rows of a color key + an L-shaped
      3-pixel arrow.

    Rule
    ----
    1) Parse all legend direction entries in the corner.
    2) Find every 2x2 cargo block by detecting a color square connected to non-black
       path pixels.
    3) Traverse each path cell-by-cell; when hitting an intersection color, move in
       the mapped direction from the legend.
    4) When maroon is reached, clear the original 2x2 block and place it at the
       maroon location with the same color (its top-left as origin).

    Key insight
    -----------
    The route is single-path deterministic except at intersection pixels: those
    pixels encode the next direction, so full-color path tracing reduces to repeated
    local 4-neighbor transitions with a legend lookup.
    """

    g = np.asarray(grid, dtype=int)
    legend = _parse_legend(g)
    starts = _find_start_blocks(g)

    out = g.copy()

    for br, bc, path_color, cargo_color, start_cell in starts:
        block_cells = ((br, bc), (br, bc + 1), (br + 1, bc), (br + 1, bc + 1))
        end_pos = _trace_single_path(g, start_cell, block_cells, path_color, legend)
        if end_pos is None:
            continue

        for r0, c0 in block_cells:
            out[r0, c0] = BLACK

        er, ec = end_pos
        if er + 1 < g.shape[0] and ec + 1 < g.shape[1]:
            out[er : er + 2, ec : ec + 2] = cargo_color

    return out


def _direction_from_delta(dr: int, dc: int) -> str:
    direction = DELTA_TO_DIR[(int(dr), int(dc))]
    return direction


def _paint_legend_entry(grid: np.ndarray, r: int, c: int, color: int, direction: str) -> None:
    grid[r, c] = color
    offsets: Iterable[Tuple[int, int]]
    if direction == "right":
        offsets = {(0, 1), (1, 1), (1, 2)}
    elif direction == "down":
        offsets = {(0, 0), (0, 1), (1, 1)}
    elif direction == "left":
        offsets = {(0, 0), (1, 0), (1, 1)}
    elif direction == "up":
        offsets = {(0, 1), (1, 0), (1, 1)}
    else:
        raise ValueError(f"unknown direction: {direction}")

    h, w = grid.shape
    for dr, dc in offsets:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            grid[rr, cc] = color


def _bfs_shortest_path(
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    blocked: np.ndarray,
    rng: np.random.Generator,
) -> Optional[list[Tuple[int, int]]]:
    h, w = grid.shape
    if start == goal:
        return [start]

    q = deque([start])
    parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
    while q:
        r, c = q.popleft()
        dirs = list(DIR_LIST)
        rng.shuffle(dirs)
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if not _in_bounds(h, w, nr, nc):
                continue
            if blocked[nr, nc] and (nr, nc) != goal:
                continue
            if (nr, nc) in parent:
                continue
            parent[(nr, nc)] = (r, c)
            if (nr, nc) == goal:
                q.clear()
                break
            q.append((nr, nc))

    if goal not in parent:
        return None

    path: list[Tuple[int, int]] = []
    cur: Optional[Tuple[int, int]] = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def _draw_path_segment(grid: np.ndarray, route: list[Tuple[int, int]], color: int) -> None:
    for r, c in route:
        grid[r, c] = color


def _pick_free_block_top_left(grid: np.ndarray, blocked: np.ndarray, rng: np.random.Generator, top_row: int) -> Optional[Tuple[int, int]]:
    h, w = grid.shape
    for _ in range(200):
        r = int(rng.integers(top_row, h - 2))
        c = int(rng.integers(1, w - 2))
        if c >= w - 2:
            c = w - 3
        if blocked[r : r + 2, c : c + 2].any():
            continue
        return int(r), int(c)
    return None


def _candidate_starts_from_block(r: int, c: int) -> List[Tuple[int, int]]:
    cells = [(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)]
    out: list[Tuple[int, int]] = []
    for br, bc in cells:
        out.extend([(br - 1, bc), (br + 1, bc), (br, bc - 1), (br, bc + 1)])
    return out


def generate_input() -> np.ndarray:
    """Generate one random valid input instance with size <= 30 and varied symbols."""
    rng = np.random.default_rng()
    palette = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12]

    for _ in range(120):
        h = int(rng.integers(24, 31))
        w = int(rng.integers(24, 31))

        if h < 24 or w < 24:
            continue

        out = np.zeros((h, w), dtype=int)
        path_row_min = 13

        path_count = int(rng.integers(2, 4))
        path_count = min(path_count, len(palette) // 2)

        if path_count <= 0:
            continue

        blocked = np.zeros((h, w), dtype=bool)
        blocked[:path_row_min, :] = True

        route_data: list[dict[str, object]] = []
        path_colors = rng.choice(np.array(palette, dtype=int), size=path_count, replace=False)
        remaining_colors = [c for c in palette if c not in set(path_colors)]
        if len(remaining_colors) < path_count:
            continue
        cargo_colors = rng.choice(np.array(remaining_colors, dtype=int), size=path_count, replace=False)

        for pi in range(path_count):
            path_color = int(path_colors[pi])
            cargo_color = int(cargo_colors[pi])

            start_block = _pick_free_block_top_left(out, blocked, rng, path_row_min)
            if start_block is None:
                continue

            br, bc = start_block
            start_candidates = [p for p in _candidate_starts_from_block(br, bc) if 0 <= p[0] < h and 0 <= p[1] < w and not blocked[p[0], p[1]] and out[p] == BLACK]
            if not start_candidates:
                continue
            start_exit = start_candidates[int(rng.integers(len(start_candidates)))]

            sr = int(rng.integers(path_row_min, h - 1))
            sc = int(rng.integers(1, w - 1))
            if out[sr, sc] != BLACK or blocked[sr, sc]:
                continue
            goal = (sr, sc)

            route = _bfs_shortest_path(out, start_exit, goal, blocked, rng)
            if route is None or len(route) < 4:
                continue

            route_data.append(
                {
                    "route": route,
                    "path_color": path_color,
                    "cargo_color": cargo_color,
                    "start_block": (br, bc),
                    "goal": goal,
                }
            )

            for rr, cc in route:
                blocked[rr, cc] = True
            blocked[br : br + 2, bc : bc + 2] = True
            blocked[sr, sc] = True

        if len(route_data) < path_count:
            continue

        intersections: list[dict[str, object]] = []
        for idx, item in enumerate(route_data):
            route = item["route"]
            if len(route) < 6:
                continue
            if rng.random() < 0.7:
                k = int(rng.integers(1, min(3, len(route) - 2)))
                ip = int(rng.integers(1, len(route) - 1))
                if ip >= len(route) - 1:
                    ip = len(route) - 2
                rr0, cc0 = route[ip]
                rr1, cc1 = route[ip + 1]
                dr, dc = rr1 - rr0, cc1 - cc0
                direction = _direction_from_delta(dr, dc)
                intersections.append(
                    {
                        "path_idx": idx,
                        "route_i": ip,
                        "direction": direction,
                    }
                )

        used_colors = set(int(c) for c in path_colors) | set(int(c) for c in cargo_colors)
        available_inter = [c for c in palette if c not in used_colors and c != MAROON]
        if not intersections and route_data:
            # force at least one legend entry/path intersection for validity coverage
            route = route_data[0]["route"]
            if len(route) >= 3:
                rr0, cc0 = route[1]
                rr1, cc1 = route[2]
                intersections.append(
                    {
                        "path_idx": 0,
                        "route_i": 1,
                        "direction": _direction_from_delta(rr1 - rr0, cc1 - cc0),
                    }
                )

        if len(intersections) > len(available_inter):
            intersections = intersections[: len(available_inter)]

        rng.shuffle(available_inter)
        legend_map: Dict[int, str] = {}
        for i, item in enumerate(intersections):
            if i >= len(available_inter):
                break
            item["color"] = int(available_inter[i])
            legend_map[int(available_inter[i])] = str(item["direction"])

        # add optional unused legend rows to diversify symbols
        extra_dirs = list(DIRS.keys())
        for color in palette:
            color = int(color)
            if color in legend_map or color in used_colors or len(legend_map) >= 4:
                continue
            if rng.random() < 0.5:
                legend_map[color] = str(rng.choice(extra_dirs))
            if len(legend_map) >= 4:
                break

        if not legend_map:
            continue

        for idx, (color, direction) in enumerate(sorted(legend_map.items(), key=lambda it: it[0])):
            key_r = 1 + idx * 3
            if key_r >= path_row_min:
                break
            _paint_legend_entry(out, key_r, 1, color, direction)

        # draw paths and place cargos
        for item in route_data:
            route = item["route"]
            path_color = int(item["path_color"])
            cargo_color = int(item["cargo_color"])
            br, bc = item["start_block"]
            goal = item["goal"]
            out[br : br + 2, bc : bc + 2] = cargo_color
            _draw_path_segment(out, route, path_color)
            out[goal[0], goal[1]] = MAROON

        # convert chosen route cells to intersection colors and keep routes otherwise
        for item in intersections:
            color = int(item["color"])
            idx = int(item["path_idx"])
            route_i = int(item["route_i"])
            route = route_data[idx]["route"]
            if 0 <= route_i < len(route):
                rr, cc = route[route_i]
                out[rr, cc] = color

        return out

    # deterministic fallback
    h, w = 24, 24
    out = np.zeros((h, w), dtype=int)
    legend_map = {1: "right", 2: "down", 3: "up", 4: "left"}
    for idx, (color, direction) in enumerate(legend_map.items()):
        _paint_legend_entry(out, 1 + 3 * idx, 1, color, direction)

    out[13:15, 12:14] = 6
    for c in range(13, 18):
        out[15, c] = 5
    out[15, 18] = MAROON
    out[16, 17] = 1
    out[17, 18] = 5
    out[16:18, 16:18] = 6
    return out


if __name__ == "__main__":
    inp = generate_input()
    out = main(inp)
    print(inp.shape, "->", out.shape)
