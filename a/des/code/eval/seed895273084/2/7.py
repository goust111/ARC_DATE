from __future__ import annotations

import itertools
import random
from collections import Counter, defaultdict, deque

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
DOOR_COLORS = (1, 2, 3, 6, 7, 8, 9)
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

GEN_ATTEMPTS = 500
ROOM_ROWS_RANGE = (2, 4)
ROOM_COLS_RANGE = (2, 5)
ROOM_SIZE_RANGE = (3, 6)
PAIR_COUNT_OPTIONS = (2, 3)


def _colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    banned = set() if exclude is None else set(exclude)
    return [int(v) for v in np.unique(grid) if int(v) not in banned]


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for dr, dc in NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _detect_gaps(grid: np.ndarray, wall_color: int) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    gaps: list[dict[str, object]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(g[r, c]) != BLACK:
                continue
            left_open = int(g[r, c - 1]) != wall_color
            right_open = int(g[r, c + 1]) != wall_color
            up_open = int(g[r - 1, c]) != wall_color
            down_open = int(g[r + 1, c]) != wall_color
            if left_open and right_open and not up_open and not down_open:
                gaps.append(
                    {
                        "center": (int(r), int(c)),
                        "orientation": "horizontal",
                        "sides": ((int(r), int(c - 1)), (int(r), int(c + 1))),
                    }
                )
            elif up_open and down_open and not left_open and not right_open:
                gaps.append(
                    {
                        "center": (int(r), int(c)),
                        "orientation": "vertical",
                        "sides": ((int(r - 1), int(c)), (int(r + 1), int(c))),
                    }
                )
    return gaps


def _room_graph(grid: np.ndarray, wall_color: int) -> tuple[list[dict[str, object]], dict[int, list[tuple[int, int]]], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    gaps = _detect_gaps(g, wall_color)
    closed = g.copy()
    for gap in gaps:
        r, c = gap["center"]
        closed[r, c] = wall_color

    room_mask = closed != wall_color
    components = _components_mask(room_mask)
    room_of = np.full(g.shape, -1, dtype=int)
    rooms: list[dict[str, object]] = []
    for rid, comp in enumerate(components):
        rows = [p[0] for p in comp]
        cols = [p[1] for p in comp]
        bbox = (min(rows), min(cols), max(rows), max(cols))
        rooms.append({"cells": comp, "bbox": bbox})
        for r, c in comp:
            room_of[r, c] = int(rid)

    edges: list[dict[str, object]] = []
    adj: dict[int, list[tuple[int, int]]] = defaultdict(list)
    seen_pairs: set[tuple[int, int]] = set()
    for gap in gaps:
        side0, side1 = gap["sides"]
        room0 = int(room_of[side0])
        room1 = int(room_of[side1])
        if room0 < 0 or room1 < 0 or room0 == room1:
            continue
        pair = tuple(sorted((room0, room1)))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        edge_id = len(edges)
        edges.append(
            {
                "rooms": (room0, room1),
                "center": gap["center"],
                "sides": gap["sides"],
                "orientation": gap["orientation"],
            }
        )
        adj[room0].append((room1, edge_id))
        adj[room1].append((room0, edge_id))
    return edges, adj, room_of


def _bfs_room_path(start: int, goal: int, adj: dict[int, list[tuple[int, int]]]) -> tuple[list[int], list[int]]:
    q = deque([start])
    parent: dict[int, tuple[int, int] | None] = {start: None}
    while q and goal not in parent:
        cur = q.popleft()
        for nxt, edge_id in adj.get(cur, []):
            if nxt in parent:
                continue
            parent[nxt] = (cur, edge_id)
            q.append(nxt)
    assert goal in parent, "paired door rooms must be connected"

    rooms = [goal]
    edges = []
    cur = goal
    while parent[cur] is not None:
        prev, edge_id = parent[cur]
        edges.append(int(edge_id))
        rooms.append(int(prev))
        cur = prev
    rooms.reverse()
    edges.reverse()
    return rooms, edges


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect same-colored room markers by coloring the wall gaps on the unique room path.

    Input structure
    - Background is black.
    - Grey one-cell-thick walls partition the canvas into rectangular rooms.
    - Some wall cells are replaced by one-cell black gaps that connect adjacent rooms.
    - Each marker color appears exactly twice, once in each of two different rooms.

    Rule
    - Pair equal-colored markers.
    - For each pair, find the unique path through room-to-room gaps.
    - Color each traversed gap cell with the pair color, plus the adjacent black room-side
      cell on both sides of that gap.
    - If multiple pair paths share one gap, the gap center becomes yellow while each side
      keeps the color chosen from that side's nearest paired endpoint.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = BLACK if np.any(g == BLACK) else _most_common_color(g)
    non_bg = _colors_present(g, exclude={bg})
    assert non_bg, "grid must contain walls"
    wall_color = int(max(non_bg, key=lambda color: int(np.count_nonzero(g == color))))

    edges, adj, room_of = _room_graph(g, wall_color)
    assert edges, "expected at least one room connection"

    marker_colors = _colors_present(g, exclude={bg, wall_color})
    marker_positions: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for color in marker_colors:
        coords = np.argwhere(g == color)
        for r, c in coords:
            marker_positions[int(color)].append((int(r), int(c)))
    assert marker_positions, "expected paired markers"
    for color, coords in marker_positions.items():
        assert len(coords) == 2, f"marker color {color} must appear exactly twice"
        room_ids = {int(room_of[r, c]) for r, c in coords}
        assert len(room_ids) == 2 and -1 not in room_ids, "same-colored markers must occupy two distinct rooms"

    edge_colors: dict[int, set[int]] = defaultdict(set)
    side_votes: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for color, coords in marker_positions.items():
        start = int(room_of[coords[0]])
        goal = int(room_of[coords[1]])
        rooms, path_edges = _bfs_room_path(start, goal, adj)
        k = len(path_edges)
        for i, edge_id in enumerate(path_edges):
            edge_colors[edge_id].add(int(color))
            left_room = int(rooms[i])
            right_room = int(rooms[i + 1])
            side_votes[(edge_id, left_room)].append((int(i), int(color)))
            side_votes[(edge_id, right_room)].append((int(k - 1 - i), int(color)))

    out = g.copy()
    for edge_id, colors in edge_colors.items():
        edge = edges[edge_id]
        center = edge["center"]
        rooms = edge["rooms"]
        sides = edge["sides"]
        out[center] = int(next(iter(colors))) if len(colors) == 1 else YELLOW
        for room_id, cell in zip(rooms, sides):
            if int(g[cell]) != BLACK:
                continue
            votes = side_votes.get((edge_id, int(room_id)), [])
            if not votes:
                continue
            _, chosen = min(votes)
            out[cell] = int(chosen)
    return out


def _build_room_layout(
    room_rows: int, room_cols: int, row_heights: list[int], col_widths: list[int]
) -> tuple[np.ndarray, dict[tuple[int, int], tuple[int, int, int, int]], list[int], list[int]]:
    height = sum(row_heights) + room_rows + 1
    width = sum(col_widths) + room_cols + 1
    grid = np.full((height, width), BLACK, dtype=int)

    row_lines = [0]
    row_starts = []
    cursor = 1
    for h in row_heights:
        row_starts.append(cursor)
        cursor += h
        row_lines.append(cursor)
        cursor += 1

    col_lines = [0]
    col_starts = []
    cursor = 1
    for w in col_widths:
        col_starts.append(cursor)
        cursor += w
        col_lines.append(cursor)
        cursor += 1

    grid[row_lines, :] = GREY
    grid[:, col_lines] = GREY

    rooms: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for rr, h in enumerate(row_heights):
        for cc, w in enumerate(col_widths):
            r0 = row_starts[rr]
            c0 = col_starts[cc]
            rooms[(rr, cc)] = (r0, c0, r0 + h - 1, c0 + w - 1)
    return grid, rooms, row_lines, col_lines


def _tree_edges(room_rows: int, room_cols: int) -> list[tuple[tuple[int, int], tuple[int, int], str]]:
    candidates: list[tuple[tuple[int, int], tuple[int, int], str]] = []
    for rr in range(room_rows):
        for cc in range(room_cols):
            if cc + 1 < room_cols:
                candidates.append(((rr, cc), (rr, cc + 1), "horizontal"))
            if rr + 1 < room_rows:
                candidates.append(((rr, cc), (rr + 1, cc), "vertical"))
    random.shuffle(candidates)

    parent: dict[tuple[int, int], tuple[int, int]] = {}

    def find(x: tuple[int, int]) -> tuple[int, int]:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: tuple[int, int], b: tuple[int, int]) -> bool:
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return False
        parent[rb] = ra
        return True

    chosen = []
    for a, b, orientation in candidates:
        if union(a, b):
            chosen.append((a, b, orientation))
    return chosen


def _open_tree_gaps(
    grid: np.ndarray,
    rooms: dict[tuple[int, int], tuple[int, int, int, int]],
    tree: list[tuple[tuple[int, int], tuple[int, int], str]],
) -> tuple[dict[tuple[tuple[int, int], tuple[int, int]], int], set[tuple[int, int]]]:
    room_adj: dict[tuple[int, int], list[tuple[tuple[int, int], tuple[int, int]]]] = defaultdict(list)
    gap_side_cells: set[tuple[int, int]] = set()
    for a, b, orientation in tree:
        ar0, ac0, ar1, ac1 = rooms[a]
        br0, bc0, br1, bc1 = rooms[b]
        if orientation == "horizontal":
            wall_col = ac1 + 1
            gap_row = random.randint(ar0 + 1, ar1 - 1)
            grid[gap_row, wall_col] = BLACK
            gap_side_cells.add((gap_row, wall_col - 1))
            gap_side_cells.add((gap_row, wall_col + 1))
        else:
            wall_row = ar1 + 1
            gap_col = random.randint(ac0 + 1, ac1 - 1)
            grid[wall_row, gap_col] = BLACK
            gap_side_cells.add((wall_row - 1, gap_col))
            gap_side_cells.add((wall_row + 1, gap_col))
        room_adj[a].append((b, tuple(sorted((a, b)))))
        room_adj[b].append((a, tuple(sorted((a, b)))))
    return room_adj, gap_side_cells


def _room_path_nodes(
    start: tuple[int, int],
    goal: tuple[int, int],
    room_adj: dict[tuple[int, int], list[tuple[tuple[int, int], tuple[tuple[int, int], tuple[int, int]]]]],
) -> tuple[list[tuple[int, int]], list[tuple[tuple[int, int], tuple[int, int]]]]:
    q = deque([start])
    parent: dict[tuple[int, int], tuple[tuple[int, int], tuple[tuple[int, int], tuple[int, int]]] | None] = {start: None}
    while q and goal not in parent:
        cur = q.popleft()
        for nxt, edge_key in room_adj.get(cur, []):
            if nxt in parent:
                continue
            parent[nxt] = (cur, edge_key)
            q.append(nxt)
    if goal not in parent:
        raise ValueError("tree path not found")
    nodes = [goal]
    edges = []
    cur = goal
    while parent[cur] is not None:
        prev, edge_key = parent[cur]
        edges.append(edge_key)
        nodes.append(prev)
        cur = prev
    nodes.reverse()
    edges.reverse()
    return nodes, edges


def _path_tables(
    room_ids: list[tuple[int, int]],
    room_adj: dict[tuple[int, int], list[tuple[tuple[int, int], tuple[tuple[int, int], tuple[int, int]]]]],
) -> dict[tuple[tuple[int, int], tuple[int, int]], tuple[list[tuple[int, int]], list[tuple[tuple[int, int], tuple[int, int]]]]]:
    table = {}
    for a, b in itertools.combinations(room_ids, 2):
        key = tuple(sorted((a, b)))
        table[key] = _room_path_nodes(a, b, room_adj)
    return table


def _edge_side_distances(
    nodes: list[tuple[int, int]], edge_key: tuple[tuple[int, int], tuple[int, int]]
) -> dict[tuple[int, int], int]:
    total_edges = len(nodes) - 1
    for i in range(total_edges):
        a = nodes[i]
        b = nodes[i + 1]
        if tuple(sorted((a, b))) != edge_key:
            continue
        return {a: int(i), b: int(total_edges - 1 - i)}
    raise ValueError("edge not found on path")


def _choose_plain_pairs(
    room_ids: list[tuple[int, int]],
    path_table: dict[tuple[tuple[int, int], tuple[int, int]], tuple[list[tuple[int, int]], list[tuple[tuple[int, int], tuple[int, int]]]]],
) -> list[tuple[tuple[int, int], tuple[int, int]]] | None:
    pair_counts = list(PAIR_COUNT_OPTIONS)
    random.shuffle(pair_counts)
    all_pairs = list(path_table.keys())
    random.shuffle(all_pairs)
    for want in pair_counts:
        picked: list[tuple[tuple[int, int], tuple[int, int]]] = []
        used_rooms: set[tuple[int, int]] = set()
        used_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
        for pair in all_pairs:
            if pair[0] in used_rooms or pair[1] in used_rooms:
                continue
            _, edges = path_table[pair]
            edge_set = set(edges)
            if edge_set & used_edges:
                continue
            picked.append(pair)
            used_rooms.update(pair)
            used_edges |= edge_set
            if len(picked) == want:
                return picked
    return None


def _choose_shared_pairs(
    room_ids: list[tuple[int, int]],
    path_table: dict[tuple[tuple[int, int], tuple[int, int]], tuple[list[tuple[int, int]], list[tuple[tuple[int, int], tuple[int, int]]]]],
) -> list[tuple[tuple[int, int], tuple[int, int]]] | None:
    pairs = list(path_table.keys())
    random.shuffle(pairs)
    for p1, p2 in itertools.combinations(pairs, 2):
        if len({p1[0], p1[1], p2[0], p2[1]}) < 4:
            continue
        nodes1, edges1 = path_table[p1]
        nodes2, edges2 = path_table[p2]
        shared = set(edges1) & set(edges2)
        if not shared:
            continue
        for edge_key in shared:
            dist1 = _edge_side_distances(nodes1, edge_key)
            dist2 = _edge_side_distances(nodes2, edge_key)
            u, v = edge_key
            if (dist1[u] - dist2[u]) * (dist1[v] - dist2[v]) < 0:
                return [p1, p2]
    return None


def _marker_candidates(
    room_bbox: tuple[int, int, int, int], forbidden: set[tuple[int, int]]
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = room_bbox
    candidates = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if (r, c) in forbidden:
                continue
            if r in (r0, r1) or c in (c0, c1):
                candidates.append((int(r), int(c)))
    random.shuffle(candidates)
    return candidates


def _paint_markers(
    grid: np.ndarray,
    rooms: dict[tuple[int, int], tuple[int, int, int, int]],
    pairs: list[tuple[tuple[int, int], tuple[int, int]]],
    forbidden: set[tuple[int, int]],
) -> list[int] | None:
    colors = random.sample(DOOR_COLORS, len(pairs))
    used_cells: set[tuple[int, int]] = set()
    for color, pair in zip(colors, pairs):
        for room_id in pair:
            candidates = [
                cell
                for cell in _marker_candidates(rooms[room_id], forbidden)
                if cell not in used_cells
            ]
            if not candidates:
                return None
            cell = random.choice(candidates[: min(6, len(candidates))])
            used_cells.add(cell)
            grid[cell] = int(color)
    return colors


def _validate_scene(grid: np.ndarray, *, expect_shared: bool) -> bool:
    try:
        out = main(grid)
    except Exception:
        return False
    yellow_count = int(np.count_nonzero(out == YELLOW))
    if expect_shared and yellow_count == 0:
        return False
    if not expect_shared and yellow_count != 0:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    room_rows = random.randint(*ROOM_ROWS_RANGE)
    room_cols = random.randint(*ROOM_COLS_RANGE)
    if room_rows * room_cols < 4:
        return None

    row_heights = [random.randint(*ROOM_SIZE_RANGE) for _ in range(room_rows)]
    col_widths = [random.randint(*ROOM_SIZE_RANGE) for _ in range(room_cols)]
    grid, rooms, _row_lines, _col_lines = _build_room_layout(room_rows, room_cols, row_heights, col_widths)

    tree = _tree_edges(room_rows, room_cols)
    room_adj, gap_side_cells = _open_tree_gaps(grid, rooms, tree)
    room_ids = list(rooms.keys())
    path_table = _path_tables(room_ids, room_adj)

    want_shared = random.random() < 0.5
    if want_shared:
        pairs = _choose_shared_pairs(room_ids, path_table)
        if pairs is None:
            return None
    else:
        pairs = _choose_plain_pairs(room_ids, path_table)
        if pairs is None:
            return None

    if _paint_markers(grid, rooms, pairs, gap_side_cells) is None:
        return None
    if not _validate_scene(grid, expect_shared=want_shared):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-connection puzzle")
