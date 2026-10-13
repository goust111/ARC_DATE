from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


RAIL_COLOR = 5
START_COLOR = 9
STEP = 3
SCREEN_SIZE = 5
SCREEN_INNER = 3
SIDES = ("left", "right", "top", "bottom")
GEN_ATTEMPTS = 500


def _mode(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(counts))


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _node_neighbors(node: tuple[int, int], nodes: set[tuple[int, int]]) -> list[tuple[int, int]]:
    r, c = node
    out: list[tuple[int, int]] = []
    for dr, dc in ((STEP, 0), (-STEP, 0), (0, STEP), (0, -STEP)):
        nxt = (r + dr, c + dc)
        if nxt in nodes:
            out.append(nxt)
    return out


def _graph_dist(a: tuple[int, int], b: tuple[int, int]) -> int:
    return (abs(a[0] - b[0]) + abs(a[1] - b[1])) // STEP


def _screen_from_end(side: str, end_node: tuple[int, int]) -> tuple[tuple[int, int, int, int], tuple[int, int]]:
    r, c = end_node
    if side == "left":
        return (r - 2, c + 1, r + 2, c + 5), (r, c + 1)
    if side == "right":
        return (r - 2, c - 5, r + 2, c - 1), (r, c - 1)
    if side == "top":
        return (r + 1, c - 2, r + 5, c + 2), (r + 1, c)
    assert side == "bottom"
    return (r - 5, c - 2, r - 1, c + 2), (r - 1, c)


def _entry_node(side: str, end_node: tuple[int, int]) -> tuple[int, int]:
    r, c = end_node
    if side == "left":
        return (r, c - STEP)
    if side == "right":
        return (r, c + STEP)
    if side == "top":
        return (r - STEP, c)
    assert side == "bottom"
    return (r + STEP, c)


def _expanded_bbox(bbox: tuple[int, int, int, int], pad: int = 1) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return (r0 - pad, c0 - pad, r1 + pad, c1 + pad)


def _screen_cells(bbox: tuple[int, int, int, int]) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    r0, c0, r1, c1 = bbox
    border: set[tuple[int, int]] = set()
    inner: set[tuple[int, int]] = set()
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if r in (r0, r1) or c in (c0, c1):
                border.add((r, c))
            else:
                inner.add((r, c))
    return border, inner


def _find_screen(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int], tuple[int, int]]:
    h, w = grid.shape
    candidates: list[tuple[int, tuple[int, int, int, int], tuple[int, int]]] = []
    for color in np.unique(grid):
        rail = int(color)
        if rail == bg:
            continue
        for r0 in range(h - SCREEN_SIZE + 1):
            for c0 in range(w - SCREEN_SIZE + 1):
                r1 = r0 + SCREEN_SIZE - 1
                c1 = c0 + SCREEN_SIZE - 1
                border_ok = True
                inner_ok = True
                for r in range(r0, r1 + 1):
                    for c in range(c0, c1 + 1):
                        if r in (r0, r1) or c in (c0, c1):
                            if int(grid[r, c]) != rail:
                                border_ok = False
                                break
                        elif int(grid[r, c]) != bg:
                            inner_ok = False
                            break
                    if not border_ok or not inner_ok:
                        break
                if not border_ok or not inner_ok:
                    continue

                border, _inner = _screen_cells((r0, c0, r1, c1))
                targets: set[tuple[int, int]] = set()
                for br, bc in border:
                    for nr, nc in _neighbors4(br, bc):
                        if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in border and not (r0 < nr < r1 and c0 < nc < c1):
                            if int(grid[nr, nc]) != bg:
                                targets.add((br, bc))
                if len(targets) == 1:
                    candidates.append((rail, (r0, c0, r1, c1), next(iter(targets))))

    assert len(candidates) == 1, "expected exactly one 5x5 display screen"
    return candidates[0]


def _path_through_tree(mask: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = mask.shape
    q: deque[tuple[int, int]] = deque([start])
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            break
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in parent:
                parent[(nr, nc)] = (r, c)
                q.append((nr, nc))
    assert goal in parent, "start must connect to the display screen"
    path: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace the unique rail path from the colored start to the display screen and
    copy the encountered signal colors into the screen row by row.
    """
    g = np.asarray(grid, dtype=int)
    bg = _mode(g)
    rail, screen_bbox, target = _find_screen(g, bg)

    walk = g != bg
    r0, c0, r1, c1 = screen_bbox
    border, _inner = _screen_cells(screen_bbox)
    for cell in border:
        walk[cell] = False
    walk[target] = True

    colored = [(int(r), int(c)) for r, c in np.argwhere((g != bg) & (g != rail))]
    start_candidates: list[tuple[int, int]] = []
    for r, c in colored:
        deg = 0
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and walk[nr, nc]:
                deg += 1
        if deg == 1:
            start_candidates.append((r, c))
    assert len(start_candidates) == 1, "expected a unique colored leaf as the start"
    start = start_candidates[0]
    start_color = int(g[start])

    path = _path_through_tree(walk, start, target)
    signal_colors: list[int] = []
    for r, c in path:
        v = int(g[r, c])
        if v not in (bg, rail, start_color):
            signal_colors.append(v)

    out = g.copy()
    out[r0 + 1 : r0 + 4, c0 + 1 : c0 + 4] = bg
    for i, color in enumerate(signal_colors[:3]):
        out[r0 + 1 + i, c0 + 1 : c0 + 4] = color
    return out


def _draw_segment(grid: np.ndarray, a: tuple[int, int], b: tuple[int, int], color: int) -> None:
    if a[0] == b[0]:
        r = a[0]
        c0, c1 = sorted((a[1], b[1]))
        grid[r, c0 : c1 + 1] = color
    else:
        c = a[1]
        r0, r1 = sorted((a[0], b[0]))
        grid[r0 : r1 + 1, c] = color


def _find_main_path(
    start: tuple[int, int],
    end: tuple[int, int],
    nodes: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
    target_min: int,
    target_max: int,
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    allowed = (nodes - forbidden) | {start, end}
    path = [start]
    seen = {start}

    def dfs(node: tuple[int, int]) -> bool:
        steps = len(path) - 1
        d = _graph_dist(node, end)
        if steps + d > target_max:
            return False
        if node == end:
            return steps >= target_min

        nbrs = [nxt for nxt in _node_neighbors(node, allowed) if nxt not in seen]
        rng.shuffle(nbrs)
        if rng.random() < 0.5:
            nbrs.sort(key=lambda nxt: _graph_dist(nxt, end))
        else:
            nbrs.sort(key=lambda nxt: _graph_dist(nxt, end), reverse=True)

        for nxt in nbrs:
            seen.add(nxt)
            path.append(nxt)
            if dfs(nxt):
                return True
            path.pop()
            seen.remove(nxt)
        return False

    if dfs(start):
        return path.copy()
    return None


def _grow_branch(
    host: tuple[int, int],
    first: tuple[int, int],
    nodes: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    rng: random.Random,
) -> list[tuple[int, int]]:
    branch = [first]
    used = set(blocked)
    used.add(host)
    used.add(first)
    prev = host
    cur = first
    target_len = rng.randint(1, 3)

    for _ in range(target_len - 1):
        options = [nxt for nxt in _node_neighbors(cur, nodes) if nxt not in used]
        if not options:
            break
        if rng.random() < 0.45:
            prev_dir = (cur[0] - prev[0], cur[1] - prev[1])
            turns = [nxt for nxt in options if (nxt[0] - cur[0], nxt[1] - cur[1]) != prev_dir]
            if turns:
                options = turns
        rng.shuffle(options)
        nxt = options[0]
        branch.append(nxt)
        used.add(nxt)
        prev, cur = cur, nxt
    return branch


def _paint_screen(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0] = RAIL_COLOR
    grid[r0 : r1 + 1, c1] = RAIL_COLOR
    grid[r0, c0 : c1 + 1] = RAIL_COLOR
    grid[r1, c0 : c1 + 1] = RAIL_COLOR


def _signal_count(mode: str, max_count: int, has_branches: bool, rng: random.Random) -> int:
    if mode == "few":
        low = 1 if has_branches else 0
        return rng.randint(low, min(2, max_count))
    if mode == "exact":
        return 3
    assert mode == "many"
    lo = 4
    hi = min(6, max_count)
    return rng.randint(lo, hi)


def _build_case(rng: random.Random) -> tuple[np.ndarray, list[int]] | None:
    signal_mode = rng.choice(("few", "exact", "many"))
    h = rng.randint(21, 29)
    w = rng.randint(21, 29)
    bg_choices = [0, 1, 2, 3, 4, 6, 7, 8]
    bg = rng.choice(bg_choices)
    signal_palette = [c for c in bg_choices if c != bg]
    rows = list(range(3, h - 3, STEP))
    cols = list(range(3, w - 3, STEP))
    if len(rows) < 5 or len(cols) < 5:
        return None
    nodes = {(r, c) for r in rows for c in cols}

    side = rng.choice(SIDES)
    end_candidates: list[tuple[int, int]] = []
    for node in nodes:
        screen_bbox, _target = _screen_from_end(side, node)
        er0, ec0, er1, ec1 = _expanded_bbox(screen_bbox, pad=1)
        if er0 < 0 or ec0 < 0 or er1 >= h or ec1 >= w:
            continue
        if _entry_node(side, node) not in nodes:
            continue
        end_candidates.append(node)
    if not end_candidates:
        return None
    end_node = rng.choice(end_candidates)
    screen_bbox, target = _screen_from_end(side, end_node)
    entry = _entry_node(side, end_node)
    sr0, sc0, sr1, sc1 = screen_bbox

    fr0, fc0, fr1, fc1 = _expanded_bbox(screen_bbox, pad=1)
    forbidden = {
        node
        for node in nodes
        if fr0 <= node[0] <= fr1 and fc0 <= node[1] <= fc1 and node not in (end_node, entry)
    }
    start_candidates = [node for node in nodes if node not in forbidden and node != entry and _graph_dist(node, end_node) >= 4]
    if not start_candidates:
        return None
    start_node = rng.choice(start_candidates)

    min_extra = {"few": 2, "exact": 4, "many": 6}[signal_mode]
    dist0 = _graph_dist(start_node, entry) + 1
    main_path = _find_main_path(
        start_node,
        entry,
        nodes,
        forbidden,
        target_min=dist0 + min_extra,
        target_max=dist0 + min_extra + 6,
        rng=rng,
    )
    if main_path is None:
        return None
    main_path.append(end_node)

    turn_indices: list[int] = []
    for i in range(1, len(main_path) - 1):
        prev_node = main_path[i - 1]
        cur = main_path[i]
        nxt = main_path[i + 1]
        if (cur[0] - prev_node[0], cur[1] - prev_node[1]) != (nxt[0] - cur[0], nxt[1] - cur[1]):
            turn_indices.append(i)
    if not turn_indices:
        return None

    used_nodes = set(main_path)
    edges: set[frozenset[tuple[int, int]]] = set()
    for a, b in zip(main_path, main_path[1:]):
        edges.add(frozenset((a, b)))

    branch_hosts_target = rng.randint(1, min(3, max(1, turn_indices[-1] - 1)))
    candidate_host_indices = list(range(1, turn_indices[-1]))
    rng.shuffle(candidate_host_indices)
    branch_host_indices: list[int] = []
    for idx in candidate_host_indices:
        if len(branch_host_indices) >= branch_hosts_target:
            break
        host = main_path[idx]
        prev_node = main_path[idx - 1]
        next_node = main_path[idx + 1]
        blocked = used_nodes | forbidden
        available = [node for node in _node_neighbors(host, nodes) if node not in blocked and node not in (prev_node, next_node)]
        if not available:
            continue
        rng.shuffle(available)
        take = 2 if len(available) >= 2 and rng.random() < 0.35 else 1
        added_any = False
        for first in available[:take]:
            branch = _grow_branch(host, first, nodes, used_nodes | forbidden, rng)
            if not branch:
                continue
            last = host
            for node in branch:
                edges.add(frozenset((last, node)))
                used_nodes.add(node)
                last = node
            added_any = True
        if added_any:
            branch_host_indices.append(idx)
    if not branch_host_indices:
        return None

    degree = Counter()
    for edge in edges:
        a, b = tuple(edge)
        degree[a] += 1
        degree[b] += 1

    eligible_indices: list[int] = []
    for i in range(1, len(main_path) - 1):
        prev_node = main_path[i - 1]
        cur = main_path[i]
        nxt = main_path[i + 1]
        turns = (cur[0] - prev_node[0], cur[1] - prev_node[1]) != (nxt[0] - cur[0], nxt[1] - cur[1])
        if turns or degree[cur] >= 3:
            eligible_indices.append(i)
    if not eligible_indices:
        return None

    last_turn = turn_indices[-1]
    if any(idx >= last_turn for idx in branch_host_indices):
        return None
    max_signals = len(eligible_indices)
    need = {"few": 2, "exact": 3, "many": 4}[signal_mode]
    if max_signals < need:
        return None
    count = _signal_count(signal_mode, max_signals, True, rng)
    if count > max_signals:
        return None

    chosen = {last_turn}
    remaining = [idx for idx in eligible_indices if idx != last_turn]
    rng.shuffle(remaining)
    for idx in remaining:
        if len(chosen) >= count:
            break
        chosen.add(idx)
    if len(chosen) != count:
        return None
    signal_indices = sorted(chosen)

    grid = np.full((h, w), bg, dtype=int)
    _paint_screen(grid, screen_bbox)
    border_cells, _inner = _screen_cells(screen_bbox)

    for edge in edges:
        a, b = tuple(edge)
        _draw_segment(grid, a, b, RAIL_COLOR)
    grid[target] = RAIL_COLOR
    for r, c in border_cells:
        grid[r, c] = RAIL_COLOR

    grid[start_node] = START_COLOR
    colors = [rng.choice(signal_palette) for _ in signal_indices]
    for idx, color in zip(signal_indices, colors):
        grid[main_path[idx]] = color

    if _mode(grid) != bg:
        return None

    out = main(grid)
    expected_rows = colors[:3]
    for i in range(SCREEN_INNER):
        row = out[sr0 + 1 + i, sc0 + 1 : sc0 + 4]
        if i < len(expected_rows):
            if not np.all(row == expected_rows[i]):
                return None
        elif not np.all(row == bg):
            return None
    if not np.array_equal(grid[sr0 + 1 : sr0 + 4, sc0 + 1 : sc0 + 4], np.full((3, 3), bg, dtype=int)):
        return None
    return grid, colors


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        built = _build_case(rng)
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid rail puzzle")
