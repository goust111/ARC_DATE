# Variant of 7b0280bc
# Dimension: D10 — 节点方块属性 (meta)
# Change: 统一尺寸的NxN(N=2或3)实心单色方块，两种颜色区分终端和中间站... → 空心方块节点
#
# Design analysis
# ---------------
# spec_variant:
#   core_change:
#     from: nodes are solid monochrome NxN squares
#     to: nodes are monochrome 1-pixel-thick hollow square frames
#   why_outer_size_changes:
#     reason: a 2x2 "hollow" square is visually identical to a solid 2x2 block, so the
#       original N in {2,3} would partly erase the D10 distinction.
#     adaptation: use outer size N in {3,4}; this keeps the same small-object regime while
#       making the hole observable in every sample.
#   node_rendering:
#     terminals: exactly 2 hollow N x N frames of one color
#     stations: 5-8 hollow N x N frames of another color
#     interior: always background color; no extra semantic color is introduced
#   tracks:
#     graph meaning unchanged: one connected graph, 1-2 cycles, unique terminal shortest path
#     width: N-2 pixels so the realized widths remain {1,2}, matching the original visual
#       density after the node-size adaptation
#   output:
#     path stations recolor only their frame border pixels to green(3)
#     path tracks recolor to grey(5)
#
# a) Concrete behavior trace
#    Original local pattern:
#      input node = 3x3 solid block, track component touches the block exterior, solver labels
#      the block as one monochrome square component and later repaints every pixel of a path
#      station block green.
#    Variant local pattern:
#      input node = 4x4 hollow frame, center is background. The solver must detect the square
#      bbox and confirm "full border in node color, interior not node color". During repaint,
#      only the border cells are recolored; the hole stays background.
#    Difference:
#      the graph logic is identical, but node identification and recoloring can no longer rely
#      on "component area == N*N".
#
# b) Complexity calibration
#    The change is visually strong but still controlled:
#    - only one new geometric property is introduced: hollowness
#    - terminals/stations remain separable by color count exactly as before
#    - interiors are always background, avoiding an extra color-mapping inference burden
#    - N has only two values {3,4}, both repeated across all nodes within a grid
#
# c) Hidden assumptions audit against the original main()
#    - old find_solid_squares assumed node components are filled squares; false now
#    - old repaint loop recolored every node pixel; now that would incorrectly fill holes
#    - old generator overlap check used filled node masks; with hollow nodes that would become
#      unsafely weak if changed to border-only, because tracks could cut through holes
#    Therefore:
#    - detection changes to hollow-frame geometry
#    - repaint targets only frame pixels
#    - generation blocks the entire node bbox for routing, even though only the border is drawn
#
# d) Placement legality
#    The generator still uses graph-first placement on a coarse cell grid, but each node reserves
#    its whole N x N bbox. Tracks are routed only outside these reserved boxes, preventing overlap,
#    hole intrusion, and out-of-bounds placements. Oracle verification via main(input) == expected
#    remains the final acceptance criterion.
#
# e) Visual observability
#    In 2-4 examples a human can now see that graph nodes are ring-like rather than solid.
#    Because every node in a grid shares the same outer size and all station/terminal nodes use
#    the same hollow structure, the D10 change is systematic rather than incidental.
#
# f) Visual cleanliness
#    Node interiors are left empty, and tracks are prevented from entering node bboxes, so no
#    frame/track layering occurs inside a node. This keeps node identity legible.
#
# g) Color-change inferability
#    No new color-role mapping is introduced. The observable rule remains:
#    "the unique shortest route between the two special-colored nodes is highlighted",
#    but now highlighted stations are hollow green frames instead of filled green blocks.
#
# 9) New assumptions introduced by the variant main()
#    - every node is a 1-pixel-thick square frame with side >= 3
#    - all nodes in a grid share one outer size N
#    - the hole inside each frame is not painted with the frame color
#    - track color is distinct from both node colors
#    The generator enforces all of these directly.

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np

from arc_utils import Color, colors_present, components_mask, get_bg, neighbors8


def find_hollow_square_frames(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    min_size: int = 3,
    connectivity: int = 8,
) -> list[dict]:
    """
    Find monochrome 1-pixel-thick hollow square frames.

    Returns list of dicts:
    - color: int
    - size: int (outer side length N)
    - bbox: (r0,c0,r1,c1)
    - coords: np.ndarray of (r,c) frame-border pixels
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

            sub = grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
            border_mask = np.ones((h, w), dtype=bool)
            border_mask[1:-1, 1:-1] = False

            if not np.all(sub[border_mask] == int(col)):
                continue
            if np.any(sub[~border_mask] == int(col)):
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


def _frame_pixels(r0: int, c0: int, size: int) -> list[tuple[int, int]]:
    pts: list[tuple[int, int]] = []
    r1 = int(r0 + size - 1)
    c1 = int(c0 + size - 1)
    for c in range(int(c0), int(c1) + 1):
        pts.append((int(r0), int(c)))
        pts.append((int(r1), int(c)))
    for r in range(int(r0) + 1, int(r1)):
        pts.append((int(r), int(c0)))
        pts.append((int(r), int(c1)))
    return pts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the shortest path between two terminal frames in a frame-and-track graph.

    Input structure:
    - bg = most common color.
    - Two non-bg colors form hollow NxN square frames (N>=3): one color marks exactly
      2 terminals, the other marks stations.
    - A third non-bg color forms track segments connecting pairs of frames.

    Rule:
    - Find the shortest path (fewest edges) between the two terminal frames.
    - Recolor every station frame on that path to green, preserving the hole.
    - Recolor every track segment along that path to grey.
    - Everything else unchanged.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    frame_cands = find_hollow_square_frames(g, bg=bg, min_size=3, connectivity=8)
    assert frame_cands

    size_counts = Counter(int(d["size"]) for d in frame_cands)
    N = int(max(size_counts.items(), key=lambda kv: (int(kv[1]), int(kv[0])))[0])
    nodes_raw = [d for d in frame_cands if int(d["size"]) == int(N)]
    assert nodes_raw

    nodes: list[dict] = []
    for i, d in enumerate(nodes_raw):
        pts = {(int(r), int(c)) for r, c in d["coords"]}
        nodes.append({"id": int(i), "color": int(d["color"]), "pts": pts})

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

    non_bg_colors = colors_present(g, exclude={bg})
    track_candidates = [c for c in non_bg_colors if int(c) not in (int(terminal_color), int(station_color))]
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

    out = g.copy()

    term_set = set(map(int, term_ids))
    for nid in node_path:
        if int(nid) in term_set:
            continue
        for r, c in id2pts[int(nid)]:
            out[int(r), int(c)] = int(Color.GREEN)

    for eid in edge_path:
        for r, c in track_comp_pts.get(int(eid), []):
            out[int(r), int(c)] = int(Color.GREY)

    return out


# --------------- generator ---------------

NODE_SIZE_CHOICES = [3, 4]               # outer side length of hollow frames
NODE_COUNT_RANGE = (7, 10)               # V = total graph nodes
CYCLE_COUNT_CHOICES = [1, 2]             # independent cycles in the graph
PATH_LEN_CHOICES = [2, 3]                # shortest-path edge count between terminals
GAP_CHOICES = [2, 3]                     # gap between frame bbox and cell boundary
CELL_GRID_DIMS = (3, 4, 5)               # candidate Kr, Kc values for the cell grid
GRID_SIZE_RANGE = (15, 30)               # canvas side length S bounds
COLOR_POOL = [0, 1, 2, 4, 6, 7, 8, 9]    # eligible colors (excludes green=3, grey=5)
CELL_PICK_RETRIES = 50                   # retries for _pick_cells connectivity check
GEN_ATTEMPT_RETRIES = 300                # outer retry limit in generate_input


def _compute_adjacency(picked: set[tuple[int, int]], use_diagonal: bool, allow_skip: bool):
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


def _pick_cells(Kr: int, Kc: int, V: int, rng: random.Random, use_diagonal: bool, allow_skip: bool):
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
    i, j = cell
    or_r, or_c = origin
    margin = (cell_size - N) // 2
    return (or_r + i * cell_size + margin, or_c + j * cell_size + margin)


def _dock_tl(sq_r: int, sq_c: int, N: int, track_w: int, si: int, sj: int):
    if si == -1:
        return sq_r - track_w, sq_c
    if si == 1:
        return sq_r + N, sq_c + N - track_w
    if sj == -1:
        return sq_r + N - track_w, sq_c - track_w
    if sj == 1:
        return sq_r, sq_c + N
    return None


def _walk_footprints(sr: int, sc: int, er: int, ec: int):
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
) -> list[tuple[int, int]]:
    track_w = N - 2
    ra, ca = pos_a
    rb, cb = pos_b
    dci, dcj = cell_b[0] - cell_a[0], cell_b[1] - cell_a[1]
    si = 0 if dci == 0 else (1 if dci > 0 else -1)
    sj = 0 if dcj == 0 else (1 if dcj > 0 else -1)

    sr, sc = _dock_tl(ra, ca, N, track_w, si, sj)
    er, ec = _dock_tl(rb, cb, N, track_w, -si, -sj)
    pts_tl = _walk_footprints(sr, sc, er, ec)

    pixels = set()
    for r, c in pts_tl:
        for dr in range(track_w):
            for dc in range(track_w):
                pixels.add((r + dr, c + dc))
    return list(pixels)


def _draw_hollow_node(
    canvas: np.ndarray,
    block_mask: np.ndarray,
    r0: int,
    c0: int,
    N: int,
    color: int,
) -> None:
    for r, c in _frame_pixels(r0, c0, N):
        canvas[int(r), int(c)] = int(color)
    block_mask[int(r0) : int(r0) + int(N), int(c0) : int(c0) + int(N)] = True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
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
    block_mask = np.zeros((S, S), dtype=bool)
    slack_r = S - Kr * cell_size - 2
    slack_c = S - Kc * cell_size - 2
    origin = (1 + rng.randint(0, max(0, slack_r)), 1 + rng.randint(0, max(0, slack_c)))
    id_to_pos = {}
    for i in range(V):
        cell = id_to_cell[i]
        pos = _cell_to_square_pos(cell, N, cell_size, origin)
        id_to_pos[i] = pos
        color = terminal_color if i in (0, V - 1) else station_color
        r0, c0 = pos
        _draw_hollow_node(canvas, block_mask, r0, c0, N, color)

    track_mask = np.zeros((S, S), dtype=bool)
    edge_track_pixels = {}
    for (a, b) in edges:
        pixels = _track_between_cells(id_to_cell[a], id_to_cell[b], id_to_pos[a], id_to_pos[b], N)
        for (r, c) in pixels:
            if not (0 <= r < S and 0 <= c < S):
                return None
            if block_mask[r, c] or track_mask[r, c]:
                return None
        for (r, c) in pixels:
            canvas[r, c] = track_color
            track_mask[r, c] = True
        edge_track_pixels[tuple(sorted((a, b)))] = pixels

    if np.argmax(np.bincount(canvas.ravel())) != bg:
        return None

    expected = canvas.copy()
    for gnode in path_nodes:
        if gnode in (0, V - 1):
            continue
        r0, c0 = id_to_pos[gnode]
        for r, c in _frame_pixels(r0, c0, N):
            expected[r, c] = int(Color.GREEN)
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
