# Variant of 7b0280bc
# Dimension: D6 — 变换操作 (transform)
# Change: 路径上中间站染绿(3)，轨道染灰(5)，其他不变... → 路径上节点计数标记
#
# Design analysis
# ===============
# variant_spec:
#   base_task:
#     type: shortest-path on a square-and-track graph
#     unchanged_parts:
#       - background is still the most common color
#       - there are still exactly 2 terminal squares in one color
#       - all non-terminal stations still share a second color
#       - tracks are still a third color and define graph edges by 8-touching nodes
#       - the shortest terminal-to-terminal path is still unique
#       - path tracks are still recolored to grey(5)
#   d6_change:
#     old_output: every intermediate station on the shortest path becomes solid green
#     new_output: each intermediate station keeps its original station color and receives
#       a green dot tally inside the square; the tally equals that station's ordinal
#       position along the shortest path (first intermediate station = 1 dot, second = 2 dots)
#   concrete_pixel_realization:
#     - To make internal marks readable, generator fixes station size to 3x3.
#     - Path length is fixed to 3 edges, so every valid sample contains exactly
#       2 intermediate stations and therefore both a 1-dot and a 2-dot example.
#     - The 1-dot station uses a single green center pixel.
#     - The 2-dot station uses two green diagonal corner pixels.
#     - Track recoloring remains unchanged.
#   behavior_trace:
#     original:
#       - BFS recovers node_path = [terminal, station_a, station_b, terminal]
#       - solver overwrites every pixel of station_a and station_b with green
#     variant:
#       - the same BFS recovery is used
#       - station_a keeps its station color, but its center cell becomes green
#       - station_b keeps its station color, but two diagonal cells become green
#       - path tracks still become grey
#     difference:
#       - the changed station is no longer a full-square recolor; it is a local transform
#         inside the existing node object, adding an ordinal/counting visual channel
#   complexity_calibration:
#     - Keeping arbitrary 2x2 nodes would make dot layouts cramped and visually weak.
#       Restricting generation to 3x3 nodes is a necessary compatibility change for D6.
#     - Allowing path length 2 would often show only a single 1-dot example, making the
#       counting rule harder to infer from few examples. Fixing path length to 3 guarantees
#       two distinct ordinal markers in every sample without changing the core graph task.
#     - The marker vocabulary is intentionally tiny (1 dot, 2 dots) so the rule is stable.
#   implicit_assumption_review:
#     - Square detection in main() depends on nodes being solid monochrome squares in the input.
#       This still holds because markers are output-only; generator never places marked nodes in input.
#     - Track detection depends on the track color being the only remaining non-node color.
#       This still holds because green(3) only appears in the output.
#     - Old code safely overwrote all station pixels because the station transform was uniform.
#       That assumption is invalid now; solver must only recolor selected interior pixels.
#     - New marker drawing assumes enough room for distinct dots. Generator enforces 3x3 nodes.
#   placement_legality:
#     - Squares are still placed on a cell grid with non-overlap and at least one background gap.
#     - Tracks are still routed as disjoint (N-1)-wide components and may not overlap nodes/tracks.
#     - Because markers are output-only and drawn strictly inside station bboxes, they cannot
#       create new connectivity artifacts that would confuse node/track parsing.
#   visual_observability:
#     - In every sample, the output contains one path station with 1 green dot and one with 2 green dots.
#     - The stations are otherwise left in their original station color, so the transform is visibly
#       different from the base task's "full green block" behavior.
#   visual_cleanliness:
#     - Marks stay inside the 3x3 square and never touch the track pixels.
#     - Using at most 2 dots avoids clutter in a tiny object.
#   color_inference:
#     - No new output color is introduced beyond the original green and grey highlights.
#     - Green still signals "this station belongs to the chosen path", but now the number of
#       green cells inside the station additionally signals station order along that path.
#   main_generator_contract:
#     - main() assumes intermediate-path ordinals fit the chosen marker layout.
#     - generate_input() guarantees exactly 2 intermediate path stations by fixing shortest path length to 3.

from __future__ import annotations

from collections import Counter, defaultdict, deque
from pathlib import Path
import random
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "solvers"))

from arc_utils import Color, components_mask, colors_present, get_bg, neighbors8


def find_solid_squares(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    min_size: int = 2,
    connectivity: int = 8,
) -> list[dict]:
    """
    Local helper (only used by this solver): find solid monochrome NxN squares (N>=min_size).

    Returns list of dicts:
    - color: int
    - size: int (N)
    - bbox: (r0,c0,r1,c1)
    - coords: np.ndarray of (r,c)
    """
    if bg is None:
        bg = get_bg(grid)
    bg = int(bg)

    out: list[dict] = []
    for col in colors_present(grid, exclude={bg}):
        for coords in components_mask(grid == int(col), connectivity=int(connectivity)):
            if coords.size == 0:
                continue
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            if h < int(min_size) or h != w:
                continue
            if int(coords.shape[0]) != int(h * w):
                continue
            out.append(
                {
                    "color": int(col),
                    "size": int(h),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "coords": coords,
                }
            )
    return out


def _marker_offsets(size: int, ordinal: int) -> list[tuple[int, int]]:
    """
    Return marker positions relative to a size x size station square.

    The generator guarantees size == 3 and ordinals in {1, 2}, but the solver keeps
    a small fallback for size == 2 to avoid relying on silent magic constants.
    """
    size = int(size)
    ordinal = int(ordinal)
    if size == 3:
        pip_map = {
            1: [(1, 1)],
            2: [(0, 0), (2, 2)],
            3: [(0, 0), (1, 1), (2, 2)],
            4: [(0, 0), (0, 2), (2, 0), (2, 2)],
            5: [(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)],
        }
        if ordinal in pip_map:
            return pip_map[ordinal]
    if size == 2:
        pip_map = {
            1: [(0, 0)],
            2: [(0, 0), (1, 1)],
            3: [(0, 0), (0, 1), (1, 1)],
            4: [(0, 0), (0, 1), (1, 0), (1, 1)],
        }
        if ordinal in pip_map:
            return pip_map[ordinal]
    raise AssertionError(f"unsupported marker layout: size={size}, ordinal={ordinal}")


def _paint_station_marker(out: np.ndarray, bbox: tuple[int, int, int, int], size: int, ordinal: int):
    r0, c0, _r1, _c1 = map(int, bbox)
    for dr, dc in _marker_offsets(int(size), int(ordinal)):
        out[int(r0 + dr), int(c0 + dc)] = int(Color.GREEN)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark the unique shortest path between the two terminal squares.

    Input structure:
    - bg = most common color.
    - Two non-bg colors form solid NxN squares (N>=2): one color marks exactly 2 terminals, the other marks stations.
    - A third non-bg color forms track segments connecting pairs of squares.

    Rule:
    - Find the shortest path (fewest edges) between the two terminal squares.
    - Recolor every track segment along that path to grey.
    - For each intermediate station on that path, keep the station square intact and draw
      a green dot marker whose dot count equals the station's ordinal position along the path.
    - Everything else is unchanged.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    # --- detect solid squares (nodes) and infer the dominant size N ---
    square_cands = find_solid_squares(g, bg=bg, min_size=2, connectivity=8)
    assert square_cands

    size_counts = Counter([int(d["size"]) for d in square_cands])
    N = int(max(size_counts.items(), key=lambda kv: (int(kv[1]), int(kv[0])))[0])
    nodes_raw = [d for d in square_cands if int(d["size"]) == int(N)]

    nodes: list[dict] = []
    for i, d in enumerate(nodes_raw):
        pts = {(int(r), int(c)) for r, c in d["coords"]}
        nodes.append(
            {
                "id": int(i),
                "color": int(d["color"]),
                "pts": pts,
                "bbox": tuple(map(int, d["bbox"])),
                "size": int(d["size"]),
            }
        )

    nodes_by_color: dict[int, list[int]] = defaultdict(list)
    for nd in nodes:
        nodes_by_color[int(nd["color"])].append(int(nd["id"]))

    node_colors = sorted(nodes_by_color.keys())
    assert len(node_colors) >= 2

    terminal_color = None
    station_color = None
    for col in node_colors:
        if len(nodes_by_color[int(col)]) == 2:
            terminal_color = int(col)
        else:
            station_color = int(col)
    assert terminal_color is not None and station_color is not None
    term_ids = list(map(int, nodes_by_color[int(terminal_color)]))

    # Track color is the remaining non-bg color (not a node color).
    non_bg_colors = colors_present(g, exclude={bg})
    track_candidates = [c for c in non_bg_colors if int(c) not in (int(terminal_color), int(station_color))]
    assert track_candidates
    track_color = int(max(track_candidates, key=lambda c: int((g == int(c)).sum())))

    # --- build node lookup ---
    pix2node: dict[tuple[int, int], int] = {}
    id2bbox: dict[int, tuple[int, int, int, int]] = {}
    id2size: dict[int, int] = {}
    for nd in nodes:
        nid = int(nd["id"])
        id2bbox[nid] = tuple(map(int, nd["bbox"]))
        id2size[nid] = int(nd["size"])
        for p in nd["pts"]:
            pix2node[p] = nid

    # --- build edges from track components that touch exactly 2 nodes (8-adjacency) ---
    track_comps = components_mask(g == int(track_color), connectivity=8)
    track_comp_pts: dict[int, list[tuple[int, int]]] = {}
    adj: dict[int, list[tuple[int, int]]] = defaultdict(list)  # node -> [(neighbor, edge_id)]

    for edge_id, coords in enumerate(track_comps, start=1):
        pts = [(int(r), int(c)) for r, c in coords]
        if not pts:
            continue
        touched: set[int] = set()
        for r, c in pts:
            for rr, cc in neighbors8(int(r), int(c)):
                nid = pix2node.get((int(rr), int(cc)))
                if nid is not None:
                    touched.add(int(nid))
        if len(touched) != 2:
            continue
        a, b = sorted(touched)
        track_comp_pts[int(edge_id)] = pts
        adj[int(a)].append((int(b), int(edge_id)))
        adj[int(b)].append((int(a), int(edge_id)))

    # --- shortest path between terminals (BFS by edge count) ---
    start, goal = int(term_ids[0]), int(term_ids[1])
    q = deque([start])
    prev: dict[int, int | None] = {start: None}
    prev_edge: dict[int, int | None] = {start: None}

    while q:
        u = int(q.popleft())
        if u == goal:
            break
        for v, eid in adj.get(u, []):
            if int(v) in prev:
                continue
            prev[int(v)] = int(u)
            prev_edge[int(v)] = int(eid)
            q.append(int(v))

    assert goal in prev

    node_path: list[int] = []
    edge_path: list[int] = []
    cur = int(goal)
    while cur is not None:
        node_path.append(int(cur))
        e = prev_edge.get(int(cur))
        if e is not None:
            edge_path.append(int(e))
        cur2 = prev.get(int(cur))
        cur = int(cur2) if cur2 is not None else None
    node_path.reverse()
    edge_path.reverse()

    # --- apply path transform in-place ---
    out = g.copy()

    for eid in edge_path:
        for r, c in track_comp_pts.get(int(eid), []):
            out[int(r), int(c)] = int(Color.GREY)

    for ordinal, nid in enumerate(node_path[1:-1], start=1):
        _paint_station_marker(out, id2bbox[int(nid)], id2size[int(nid)], ordinal)

    return out


# --------------- generator ---------------

# --- CFG: tuned to make the D6 marker visually observable ---
NODE_SIZE_CHOICES = [3]                 # fixed to 3 so internal dot markers are legible
NODE_COUNT_RANGE = (7, 10)              # V = total graph nodes (2 terminal + rest station)
CYCLE_COUNT_CHOICES = [1, 2]            # independent cycles in the graph
PATH_LEN_CHOICES = [3]                  # fixed so every sample shows both 1-dot and 2-dot stations
GAP_CHOICES = [2, 3]                    # gap between node square edge and cell boundary
CELL_GRID_DIMS = (3, 4, 5)              # candidate Kr, Kc values for the cell grid
GRID_SIZE_RANGE = (15, 30)              # canvas side length S bounds
COLOR_POOL = [0, 1, 2, 4, 6, 7, 8, 9]   # eligible colors (excludes green=3, grey=5)
CELL_PICK_RETRIES = 50
GEN_ATTEMPT_RETRIES = 1000


def _compute_adjacency(picked: set[tuple[int, int]], use_diagonal: bool, allow_skip: bool):
    """
    Compute valid undirected edges between picked cells on a grid.

    Always includes cardinal distance-1 neighbors. Optionally includes:
    - allow_skip: cardinal distance-2 when the intermediate cell is empty.
    - use_diagonal: diagonal distance-1 when both perpendicular cells are empty.
    """
    edges = set()
    for (i, j) in picked:
        for (di, dj) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            n = (i + di, j + dj)
            if n in picked:
                edges.add(tuple(sorted([(i, j), n])))
        if allow_skip:
            for (di, dj) in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                mid = (i + di // 2, j + dj // 2)
                far = (i + di, j + dj)
                if far in picked and mid not in picked:
                    edges.add(tuple(sorted([(i, j), far])))
        if use_diagonal:
            for (di, dj) in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                n = (i + di, j + dj)
                corner1 = (i + di, j)
                corner2 = (i, j + dj)
                if n in picked and corner1 not in picked and corner2 not in picked:
                    edges.add(tuple(sorted([(i, j), n])))
    return edges


def _pick_cells(
    Kr: int,
    Kc: int,
    V: int,
    rng: random.Random,
    use_diagonal: bool,
    allow_skip: bool,
):
    """Pick V cells on a Kr x Kc grid ensuring the induced adjacency graph is connected."""
    cells = [(i, j) for i in range(Kr) for j in range(Kc)]
    if len(cells) < V:
        return None
    for _ in range(CELL_PICK_RETRIES):
        rng.shuffle(cells)
        picked = set(cells[:V])
        edges = _compute_adjacency(picked, use_diagonal=use_diagonal, allow_skip=allow_skip)
        adj = {c: [] for c in picked}
        for (a, b) in edges:
            adj[a].append(b)
            adj[b].append(a)
        root = next(iter(picked))
        seen = {root}
        stack = [root]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        if len(seen) == V:
            return picked
    return None


def _bfs_with_counts(src: int, allowed: set[tuple[int, int]], V: int):
    """BFS from src over allowed edges. Returns (dist, count, adj_list) dicts."""
    adj = {i: [] for i in range(V)}
    for (a, b) in allowed:
        adj[a].append(b)
        adj[b].append(a)
    dist = {src: 0}
    cnt = {src: 1}
    q = deque([src])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                cnt[v] = cnt[u]
                q.append(v)
            elif dist[v] == dist[u] + 1:
                cnt[v] += cnt[u]
    return dist, cnt, adj


def _build_graph_spatial(
    picked: set[tuple[int, int]],
    cell_to_id: dict[tuple[int, int], int],
    rng: random.Random,
    path_len: int,
    n_cycles: int,
    use_diagonal: bool,
    allow_skip: bool,
):
    """
    Build an edge-subgraph on spatial adjacencies of the picked cells.

    Select two terminals, construct a spanning tree seeded from one shortest path of
    `path_len` edges, then add `n_cycles` extra edges that preserve shortest-path uniqueness.
    """
    V = len(picked)
    cell_edges = _compute_adjacency(picked, use_diagonal=use_diagonal, allow_skip=allow_skip)
    full_edges = {tuple(sorted((cell_to_id[a], cell_to_id[b]))) for (a, b) in cell_edges}

    ids = list(range(V))
    rng.shuffle(ids)
    t0, t1 = ids[0], ids[1]

    dist, cnt, adj = _bfs_with_counts(t0, full_edges, V)
    if t1 not in dist or dist[t1] != path_len or cnt.get(t1) != 1:
        return None

    parent = {t0: None}
    q = deque([t0])
    seen = {t0}
    while q:
        u = q.popleft()
        if u == t1:
            break
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                parent[v] = u
                q.append(v)
    cur = t1
    sp_nodes = []
    while cur is not None:
        sp_nodes.append(cur)
        cur = parent[cur]
    sp_nodes.reverse()
    sp_edges = {tuple(sorted((sp_nodes[i], sp_nodes[i + 1]))) for i in range(len(sp_nodes) - 1)}

    tree_edges = set(sp_edges)
    in_tree = set(sp_nodes)
    frontier = list(sp_nodes)
    while in_tree != set(range(V)):
        next_frontier = []
        for u in frontier:
            for v in adj[u]:
                if v not in in_tree:
                    tree_edges.add(tuple(sorted((u, v))))
                    in_tree.add(v)
                    next_frontier.append(v)
        if not next_frontier:
            return None
        frontier = next_frontier

    extra_candidates = list(full_edges - tree_edges)
    rng.shuffle(extra_candidates)
    final_edges = set(tree_edges)
    added = 0
    for e in extra_candidates:
        if added >= n_cycles:
            break
        cand = final_edges | {e}
        dist2, cnt2, _ = _bfs_with_counts(t0, cand, V)
        if dist2.get(t1) == path_len and cnt2.get(t1) == 1:
            final_edges.add(e)
            added += 1
    if added != n_cycles:
        return None

    dist3, cnt3, _ = _bfs_with_counts(t0, final_edges, V)
    if dist3.get(t1) != path_len or cnt3.get(t1) != 1:
        return None
    if len(final_edges) != V - 1 + n_cycles:
        return None

    id_map = {t0: 0, t1: V - 1}
    for idx, i in enumerate(i for i in range(V) if i != t0 and i != t1):
        id_map[i] = idx + 1
    remapped_edges = {tuple(sorted((id_map[a], id_map[b]))) for (a, b) in final_edges}
    remapped_cell_to_id = {cell: id_map[cell_to_id[cell]] for cell in cell_to_id}
    new_path_nodes = [id_map[n] for n in sp_nodes]
    new_path_edges = [
        tuple(sorted((new_path_nodes[i], new_path_nodes[i + 1])))
        for i in range(len(new_path_nodes) - 1)
    ]

    return list(remapped_edges), new_path_nodes, new_path_edges, remapped_cell_to_id


def _cell_to_square_pos(cell: tuple[int, int], N: int, cell_size: int, origin: tuple[int, int]):
    """Top-left corner of the NxN square centered in the given cell."""
    i, j = cell
    or_r, or_c = origin
    margin = (cell_size - N) // 2
    return (or_r + i * cell_size + margin, or_c + j * cell_size + margin)


def _dock_tl(sq_r: int, sq_c: int, N: int, Wt: int, si: int, sj: int):
    """Top-left of the Wt x Wt footprint docked to one face of an NxN square."""
    if si == -1:
        return sq_r - Wt, sq_c
    if si == 1:
        return sq_r + N, sq_c + N - Wt
    if sj == -1:
        return sq_r + N - Wt, sq_c - Wt
    if sj == 1:
        return sq_r, sq_c + N
    return None


def _walk_footprints(sr: int, sc: int, er: int, ec: int):
    """Generate the sequence of footprint top-left positions from (sr,sc) to (er,ec)."""
    pts = [(sr, sc)]
    cr, cc = sr, sc
    while cr != er and cc != ec:
        cr += 1 if er > cr else -1
        cc += 1 if ec > cc else -1
        pts.append((cr, cc))
    while cr != er:
        cr += 1 if er > cr else -1
        pts.append((cr, cc))
    while cc != ec:
        cc += 1 if ec > cc else -1
        pts.append((cr, cc))
    return pts


def _track_between_cells(
    cell_a: tuple[int, int],
    cell_b: tuple[int, int],
    pos_a: tuple[int, int],
    pos_b: tuple[int, int],
    N: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    """Route a (N-1)-wide track strip between two node squares."""
    del rng  # routing is deterministic; keep signature aligned with the original scaffold
    Wt = N - 1
    ra, ca = pos_a
    rb, cb = pos_b
    dci, dcj = cell_b[0] - cell_a[0], cell_b[1] - cell_a[1]
    si = 0 if dci == 0 else (1 if dci > 0 else -1)
    sj = 0 if dcj == 0 else (1 if dcj > 0 else -1)

    sr, sc = _dock_tl(ra, ca, N, Wt, si, sj)
    er, ec = _dock_tl(rb, cb, N, Wt, -si, -sj)

    pts_tl = _walk_footprints(sr, sc, er, ec)

    pixels = set()
    for r, c in pts_tl:
        for dr in range(Wt):
            for dc in range(Wt):
                pixels.add((r + dr, c + dc))
    return list(pixels)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    """Single attempt to generate a valid input grid. Returns None on failure."""
    N = rng.choice(NODE_SIZE_CHOICES)
    V = rng.randint(*NODE_COUNT_RANGE)
    n_cycles = rng.choice(CYCLE_COUNT_CHOICES)
    path_len = rng.choice(PATH_LEN_CHOICES)

    gap = rng.choice(GAP_CHOICES)
    cell_size = N + gap

    KrKc_options = []
    for Kr in CELL_GRID_DIMS:
        for Kc in CELL_GRID_DIMS:
            if Kr * Kc < V:
                continue
            min_S = max(Kr, Kc) * cell_size + 2
            if min_S > GRID_SIZE_RANGE[1]:
                continue
            KrKc_options.append((Kr, Kc, min_S))
    if not KrKc_options:
        return None
    Kr, Kc, min_S = rng.choice(KrKc_options)
    S = rng.randint(max(GRID_SIZE_RANGE[0], min_S), GRID_SIZE_RANGE[1])

    use_diagonal = False
    allow_skip = True
    picked = _pick_cells(Kr, Kc, V, rng, use_diagonal=use_diagonal, allow_skip=allow_skip)
    if picked is None:
        return None

    picked_list = list(picked)
    rng.shuffle(picked_list)
    cell_to_id = {cell: i for i, cell in enumerate(picked_list)}

    gr = _build_graph_spatial(
        picked,
        cell_to_id,
        rng,
        path_len,
        n_cycles,
        use_diagonal=use_diagonal,
        allow_skip=allow_skip,
    )
    if gr is None:
        return None
    edges, path_nodes, path_edges, cell_to_id = gr
    id_to_cell = {v: k for k, v in cell_to_id.items()}
    V = len(picked)

    pool = list(COLOR_POOL)
    rng.shuffle(pool)
    four = pool[:4]
    bg = rng.choice(four)
    rest = [c for c in four if c != bg]
    terminal_color = rng.choice(rest)
    rest2 = [c for c in rest if c != terminal_color]
    station_color = rng.choice(rest2)
    track_color = [c for c in rest2 if c != station_color][0]

    canvas = np.full((S, S), bg, dtype=int)
    node_mask = np.zeros((S, S), dtype=bool)
    slack_r = S - Kr * cell_size - 2
    slack_c = S - Kc * cell_size - 2
    origin = (1 + rng.randint(0, max(0, slack_r)), 1 + rng.randint(0, max(0, slack_c)))
    id_to_pos = {}
    id_to_bbox = {}
    for i in range(V):
        cell = id_to_cell[i]
        pos = _cell_to_square_pos(cell, N, cell_size, origin)
        id_to_pos[i] = pos
        r0, c0 = pos
        id_to_bbox[i] = (r0, c0, r0 + N - 1, c0 + N - 1)
        color = terminal_color if i in (0, V - 1) else station_color
        canvas[r0 : r0 + N, c0 : c0 + N] = color
        node_mask[r0 : r0 + N, c0 : c0 + N] = True

    track_mask = np.zeros((S, S), dtype=bool)
    edge_track_pixels = {}
    for (a, b) in edges:
        pixels = _track_between_cells(id_to_cell[a], id_to_cell[b], id_to_pos[a], id_to_pos[b], N, rng)
        for (r, c) in pixels:
            if not (0 <= r < S and 0 <= c < S):
                return None
            if node_mask[r, c] or track_mask[r, c]:
                return None
        for (r, c) in pixels:
            canvas[r, c] = track_color
            track_mask[r, c] = True
        edge_track_pixels[tuple(sorted((a, b)))] = pixels

    if np.argmax(np.bincount(canvas.ravel())) != bg:
        return None

    expected = canvas.copy()
    for ordinal, gnode in enumerate(path_nodes[1:-1], start=1):
        _paint_station_marker(expected, id_to_bbox[int(gnode)], N, ordinal)
    for e in path_edges:
        for (r, c) in edge_track_pixels[tuple(sorted(e))]:
            expected[r, c] = int(Color.GREY)

    try:
        actual = main(canvas)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPT_RETRIES):
        out = _gen_attempt(rng)
        if out is not None:
            return out
    raise RuntimeError(f"generate_input: failed after {GEN_ATTEMPT_RETRIES} attempts")
