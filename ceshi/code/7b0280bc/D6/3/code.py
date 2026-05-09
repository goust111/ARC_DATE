# Variant of 7b0280bc
# Dimension: D6 — 变换操作 (transform)
# Change: 路径上中间站染绿(3)，轨道染灰(5)，其他不变... → 源和终端也参与变换
#
# Design analysis
# spec_like:
#   input_objects:
#     - bg: most common color.
#     - source: exactly 1 solid NxN square, fixed input marker color teal(8).
#     - goal: exactly 1 solid NxN square, fixed input marker color maroon(9).
#     - stations: 5-8 solid NxN squares, all same color, distinct from bg/track/source/goal.
#     - tracks: one non-node color, routed exactly as in the base task.
#   output_transform:
#     - shortest-path source square -> blue(1)
#     - shortest-path goal square -> red(2)
#     - shortest-path intermediate station squares -> green(3)
#     - shortest-path track segments -> grey(5)
#     - all other pixels unchanged
#   why_input_changes_are_needed:
#     - The base task keeps both terminals visually identical, so "start" vs "end"
#       is not recoverable from 2-3 ARC examples.
#     - To make D6 observable, the variant introduces two fixed terminal marker
#       colors in the input while preserving the same graph and shortest-path logic.
#   concrete_trace:
#     - Base logic: terminals share one color; BFS finds the unique shortest path;
#       recoloring loop skips both terminals and only paints interior stations green.
#     - Variant logic: BFS still runs from source to goal, but the final recolor loop
#       paints the first node blue, the last node red, and only the interior nodes green.
#     - Difference: the path endpoints now encode directionality in the output.
#   complexity_calibration:
#     - Only one new degree of freedom is added: endpoint role.
#     - Source/goal marker colors are fixed across all samples, so the role mapping
#       is repeated and learnable rather than hidden in arbitrary per-instance colors.
#     - Input avoids output colors 1/2/3/5 so every transform is visibly nontrivial.
#   implicit_assumption_audit:
#     - Old assumption 1: "the color appearing on exactly two squares is the terminal
#       class" no longer holds. New solver instead looks for fixed source/goal marker
#       colors and chooses the most frequent remaining square color as the station class.
#     - Old assumption 2: "terminals must be skipped in recoloring" no longer holds.
#       The recolor pass now assigns node-path positions by role: first/last/interior.
#     - Old assumption 3: "4 total colors are enough" no longer holds once source and
#       goal must be distinguishable in the input. Generator now uses 5 input colors.
#   placement_legality:
#     - Spatial placement and routing stay unchanged from the base generator, so the
#       no-overlap / in-bounds guarantees still come from the original cell-grid layout.
#     - Generator verifies each sample by comparing solver output against the intended
#       output before returning the input grid.
#   visual_observability:
#     - Humans can now see two special endpoint colors in the input and two distinct
#       endpoint colors in the output, while tracks and intermediate stations still
#       behave exactly as in the base task.
#     - This keeps the scene tidy: squares remain solid, tracks remain disjoint strips,
#       and no extra marks are layered inside objects.

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np

from arc_utils import Color, colors_present, components_mask, get_bg, neighbors8


SOURCE_INPUT_COLOR = int(Color.TEAL)
GOAL_INPUT_COLOR = int(Color.MAROON)
SOURCE_OUTPUT_COLOR = int(Color.BLUE)
GOAL_OUTPUT_COLOR = int(Color.RED)
STATION_OUTPUT_COLOR = int(Color.GREEN)
TRACK_OUTPUT_COLOR = int(Color.GREY)


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


def _select_nodes(square_cands: list[dict]) -> tuple[list[dict], int, int, int]:
    """Pick the intended source/goal/station squares among dominant-size solid squares."""
    assert square_cands

    size_counts = Counter(int(d["size"]) for d in square_cands)
    node_size = int(max(size_counts.items(), key=lambda kv: (int(kv[1]), int(kv[0])))[0])
    nodes_raw = [d for d in square_cands if int(d["size"]) == node_size]

    source_raw = [d for d in nodes_raw if int(d["color"]) == SOURCE_INPUT_COLOR]
    goal_raw = [d for d in nodes_raw if int(d["color"]) == GOAL_INPUT_COLOR]
    assert len(source_raw) == 1 and len(goal_raw) == 1

    other_counts = Counter(
        int(d["color"])
        for d in nodes_raw
        if int(d["color"]) not in (SOURCE_INPUT_COLOR, GOAL_INPUT_COLOR)
    )
    assert other_counts
    station_color = int(max(other_counts.items(), key=lambda kv: (int(kv[1]), int(kv[0])))[0])
    assert other_counts[station_color] >= 2

    selected = [source_raw[0], goal_raw[0]]
    selected.extend(d for d in nodes_raw if int(d["color"]) == station_color)

    nodes: list[dict] = []
    start_id = -1
    goal_id = -1
    for i, d in enumerate(selected):
        pts = {(int(r), int(c)) for r, c in d["coords"]}
        color = int(d["color"])
        nodes.append({"id": int(i), "color": color, "pts": pts})
        if color == SOURCE_INPUT_COLOR:
            start_id = int(i)
        elif color == GOAL_INPUT_COLOR:
            goal_id = int(i)

    assert start_id >= 0 and goal_id >= 0
    return nodes, start_id, goal_id, station_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the unique shortest path from a marked source square to a marked goal square.

    Input structure:
    - bg = most common color.
    - Exactly one teal NxN solid square is the source.
    - Exactly one maroon NxN solid square is the goal.
    - Another color forms the station squares.
    - A further color forms 1-cell-wide track components connecting pairs of squares.

    Rule:
    - Find the shortest path (fewest edges) from source to goal in the square-track graph.
    - Recolor the source to blue, the goal to red, interior stations on the path to green,
      and path tracks to grey.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    square_cands = find_solid_squares(g, bg=bg, min_size=2, connectivity=8)
    nodes, start, goal, station_color = _select_nodes(square_cands)

    non_bg_colors = colors_present(g, exclude={bg})
    track_candidates = [
        c
        for c in non_bg_colors
        if int(c) not in (SOURCE_INPUT_COLOR, GOAL_INPUT_COLOR, int(station_color))
    ]
    assert track_candidates
    track_color = int(max(track_candidates, key=lambda c: int((g == int(c)).sum())))

    pix2node: dict[tuple[int, int], int] = {}
    id2pts: dict[int, set[tuple[int, int]]] = {}
    for nd in nodes:
        nid = int(nd["id"])
        pts = set(nd["pts"])
        id2pts[nid] = pts
        for p in pts:
            pix2node[p] = nid

    track_comps = components_mask(g == int(track_color), connectivity=8)
    track_comp_pts: dict[int, list[tuple[int, int]]] = {}
    adj: dict[int, list[tuple[int, int]]] = defaultdict(list)

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

    out = g.copy()

    for idx, nid in enumerate(node_path):
        if idx == 0:
            fill = SOURCE_OUTPUT_COLOR
        elif idx == len(node_path) - 1:
            fill = GOAL_OUTPUT_COLOR
        else:
            fill = STATION_OUTPUT_COLOR
        for r, c in id2pts[int(nid)]:
            out[int(r), int(c)] = int(fill)

    for eid in edge_path:
        for r, c in track_comp_pts.get(int(eid), []):
            out[int(r), int(c)] = TRACK_OUTPUT_COLOR

    return out


# --------------- generator ---------------

NODE_SIZE_CHOICES = [2, 3]
NODE_COUNT_RANGE = (7, 10)
CYCLE_COUNT_CHOICES = [1, 2]
PATH_LEN_CHOICES = [2, 3]
GAP_CHOICES = [2, 3]
CELL_GRID_DIMS = (3, 4, 5)
GRID_SIZE_RANGE = (15, 30)
BASE_COLOR_POOL = [0, 4, 6, 7]
CELL_PICK_RETRIES = 50
GEN_ATTEMPT_RETRIES = 300


def _compute_adjacency(picked: set[tuple[int, int]], use_diagonal: bool, allow_skip: bool):
    """Compute valid undirected edges between picked cells on a grid."""
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


def _bfs_with_counts(src: int, allowed: set, V: int):
    """BFS from src over allowed edges. Returns (dist, count, adj_list)."""
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
    cell_to_id: dict,
    rng: random.Random,
    path_len: int,
    n_cycles: int,
    use_diagonal: bool,
    allow_skip: bool,
):
    """
    Build an edge-subgraph on spatial adjacencies of the picked cells.

    Returns (edges, path_nodes, path_edges, remapped_cell_to_id) or None on failure.
    Node id 0 is the source and node id V-1 is the goal.
    """
    V = len(picked)
    cell_edges = _compute_adjacency(picked, use_diagonal=use_diagonal, allow_skip=allow_skip)
    full_edges = {tuple(sorted((cell_to_id[a], cell_to_id[b]))) for (a, b) in cell_edges}

    ids = list(range(V))
    rng.shuffle(ids)
    t0, t1 = ids[0], ids[1]

    dist, cnt, adj = _bfs_with_counts(t0, full_edges, V)
    if t1 not in dist or dist[t1] != path_len:
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
    del rng
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


def _apply_expected_output(
    canvas: np.ndarray,
    path_nodes: list[int],
    path_edges: list[tuple[int, int]],
    id_to_pos: dict[int, tuple[int, int]],
    edge_track_pixels: dict[tuple[int, int], list[tuple[int, int]]],
    N: int,
) -> np.ndarray:
    """Materialize the intended output grid for oracle-checking the generator."""
    expected = canvas.copy()
    for idx, gnode in enumerate(path_nodes):
        if idx == 0:
            fill = SOURCE_OUTPUT_COLOR
        elif idx == len(path_nodes) - 1:
            fill = GOAL_OUTPUT_COLOR
        else:
            fill = STATION_OUTPUT_COLOR
        r0, c0 = id_to_pos[gnode]
        expected[r0 : r0 + N, c0 : c0 + N] = fill
    for e in path_edges:
        for (r, c) in edge_track_pixels[tuple(sorted(e))]:
            expected[r, c] = TRACK_OUTPUT_COLOR
    return expected


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

    pool = list(BASE_COLOR_POOL)
    rng.shuffle(pool)
    bg, station_color, track_color = pool[:3]

    canvas = np.full((S, S), bg, dtype=int)
    node_mask = np.zeros((S, S), dtype=bool)
    slack_r = S - Kr * cell_size - 2
    slack_c = S - Kc * cell_size - 2
    origin = (1 + rng.randint(0, max(0, slack_r)), 1 + rng.randint(0, max(0, slack_c)))
    id_to_pos = {}
    for i in range(V):
        cell = id_to_cell[i]
        pos = _cell_to_square_pos(cell, N, cell_size, origin)
        id_to_pos[i] = pos
        if i == 0:
            color = SOURCE_INPUT_COLOR
        elif i == V - 1:
            color = GOAL_INPUT_COLOR
        else:
            color = station_color
        r0, c0 = pos
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

    expected = _apply_expected_output(canvas, path_nodes, path_edges, id_to_pos, edge_track_pixels, N)

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
