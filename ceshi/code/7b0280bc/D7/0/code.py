# Variant of 7b0280bc
# Dimension: D7 — 输出网格形状 (grid_shape)
# Change: 输出与输入相同尺寸，仅改变颜色... → 紧凑裁剪

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np

from arc_utils import Color, components_mask, colors_present, get_bg, neighbors8


# -----------------------------------------------------------------------------
# Design analysis for the D7 variant
#
# 0. Reference-task note
#    - The prompt cites `e87109e9` as the source of the tight-crop idea.
#    - In this workspace its official JSON exists, but no readable solver `.py`
#      exists, so the "read the source_task solver first" step cannot be
#      executed. The implementation below therefore uses only the original
#      `7b0280bc` solver/generator plus the visual idea from the available JSON:
#      output keeps the solved content but discards irrelevant outer area.
#
# 1. Concretization / behavior trace
#    - Original task: detect the unique shortest path in a graph whose nodes are
#      solid NxN squares and whose edges are 1-component track segments. Recolor
#      path-internal stations to green and path edges to grey, but keep the full
#      canvas.
#    - Variant: keep exactly the same graph reasoning and recoloring, then crop
#      the solved grid to the tight bounding box of all objects on the chosen
#      shortest path:
#        * both terminal squares,
#        * every intermediate station square on that path,
#        * every track component used by that path.
#      Inside that rectangle, background cells remain background; only the outer
#      irrelevant rows/cols are removed.
#    - Example trace:
#      original: recolor path on a 24x24 grid, return 24x24;
#      variant : recolor the same path, compute bbox(path terminals + stations
#      + path tracks), e.g. rows 6..15 / cols 4..18, and return only that
#      10x15 subgrid.
#
# 2. Complexity calibration
#    - Only D7 changes: graph parsing, terminal/station identification, shortest
#      path search, and recolor semantics remain unchanged.
#    - To make the crop visually inferable from 2-4 examples, generated inputs
#      are constrained so that:
#        * the path bbox excludes at least some other graph content;
#        * non-path graph objects do not lie inside the crop rectangle.
#      This avoids ambiguous examples where the crop looks like a random slice.
#
# 3. Implicit-assumption audit of the original solver
#    - `out = g.copy()` originally assumed output shape equals input shape.
#      This is no longer valid; the recolored full canvas must be cropped before
#      returning.
#    - Path detection code itself depends only on node/track topology, so it is
#      still valid under D7.
#    - Any code that indexes output by input coordinates is still safe because
#      recoloring happens before cropping; the crop is a final post-process.
#
# 4. Placement legality requirements for the generator
#    - All squares must remain disjoint and all track components must remain
#      disjoint and non-overlapping with squares.
#    - Every track component must still touch exactly two squares by 8-neighbor
#      contact so the unchanged graph parser can recover the abstract graph.
#    - The path bbox must be strictly smaller than the full canvas.
#
# 5. Visual observability / cleanliness
#    - The generator still uses the original graph-first random construction, so
#      the task keeps the same visual vocabulary as the source task.
#    - To make D7 observable, it rejects any sample where a non-path node or
#      non-path track would remain inside the crop rectangle, and also rejects
#      samples where the crop would remove no graph content.
#      Therefore the output visibly says "solve the path, then keep only the
#      tight rectangle around that solved path".
#
# 6. New solver assumptions and generator guarantees
#    - New solver assumption: the bbox of path-support pixels is non-empty.
#      Guaranteed because terminals are always on the path.
#    - New solver assumption: cropping by that bbox should not include unrelated
#      foreground if we want clean training examples.
#      Guaranteed by generator-side rejection checks.
# -----------------------------------------------------------------------------


def find_solid_squares(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    min_size: int = 2,
    connectivity: int = 8,
) -> list[dict]:
    """Find solid monochrome NxN squares (N>=min_size)."""
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


def _analyze_graph(grid: np.ndarray) -> dict:
    """Recover nodes, track components, and the unique shortest terminal path."""
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    square_cands = find_solid_squares(g, bg=bg, min_size=2, connectivity=8)
    assert square_cands

    size_counts = Counter(int(d["size"]) for d in square_cands)
    N = int(max(size_counts.items(), key=lambda kv: (int(kv[1]), int(kv[0])))[0])
    nodes_raw = [d for d in square_cands if int(d["size"]) == int(N)]

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

    return {
        "grid": g,
        "bg": bg,
        "id2pts": id2pts,
        "term_ids": term_ids,
        "track_comp_pts": track_comp_pts,
        "node_path": node_path,
        "edge_path": edge_path,
    }


def _solve_full_and_bbox(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Apply original recoloring, then return the full solved grid and path bbox."""
    info = _analyze_graph(grid)
    g = info["grid"]
    out = g.copy()

    node_path = list(map(int, info["node_path"]))
    edge_path = list(map(int, info["edge_path"]))
    term_set = set(map(int, info["term_ids"]))
    id2pts = info["id2pts"]
    track_comp_pts = info["track_comp_pts"]

    path_pixels: set[tuple[int, int]] = set()

    for nid in node_path:
        pts = id2pts[int(nid)]
        path_pixels.update(pts)
        if int(nid) in term_set:
            continue
        for r, c in pts:
            out[int(r), int(c)] = int(Color.GREEN)

    for eid in edge_path:
        pts = track_comp_pts.get(int(eid), [])
        path_pixels.update(pts)
        for r, c in pts:
            out[int(r), int(c)] = int(Color.GREY)

    assert path_pixels
    rows = [p[0] for p in path_pixels]
    cols = [p[1] for p in path_pixels]
    bbox = (min(rows), min(cols), max(rows), max(cols))
    return out, bbox


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the shortest terminal-to-terminal path, then tightly crop to the
    bounding box of the path's nodes and track segments.
    """
    solved_full, (r0, c0, r1, c1) = _solve_full_and_bbox(np.array(grid, dtype=int))
    return solved_full[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


# -----------------------------------------------------------------------------
# Generator
# -----------------------------------------------------------------------------

NODE_SIZE_CHOICES = [2, 3]
NODE_COUNT_RANGE = (7, 10)
CYCLE_COUNT_CHOICES = [1, 2]
PATH_LEN_CHOICES = [2, 3]
GAP_CHOICES = [2, 3]
CELL_GRID_DIMS = (3, 4, 5)
GRID_SIZE_RANGE = (15, 30)
COLOR_POOL = [0, 1, 2, 4, 6, 7, 8, 9]
CELL_PICK_RETRIES = 50
GEN_ATTEMPT_RETRIES = 1200


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
    """Generate footprint top-left positions from start to end."""
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
    """Route a (N-1)-wide track strip between two node squares."""
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


def _pick_cells(Kr: int, Kc: int, V: int, rng: random.Random, use_diagonal: bool, allow_skip: bool):
    """Pick V cells whose induced adjacency graph is connected."""
    cells = [(i, j) for i in range(Kr) for j in range(Kc)]
    if len(cells) < V:
        return None
    for _ in range(CELL_PICK_RETRIES):
        rng.shuffle(cells)
        picked = set(cells[:V])
        edges = _compute_adjacency(picked, use_diagonal=use_diagonal, allow_skip=allow_skip)
        adj = {c: [] for c in picked}
        for a, b in edges:
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
    """BFS from src over allowed edges, returning distances and path counts."""
    adj = {i: [] for i in range(V)}
    for a, b in allowed:
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
    """Build a connected graph with a unique shortest path of the requested length."""
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


def _compute_bbox_from_pixels(pixels: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in pixels]
    cols = [c for _, c in pixels]
    return min(rows), min(cols), max(rows), max(cols)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    """Single generation attempt."""
    N = rng.choice(NODE_SIZE_CHOICES)
    V = rng.randint(*NODE_COUNT_RANGE)
    n_cycles = rng.choice(CYCLE_COUNT_CHOICES)
    path_len = rng.choice(PATH_LEN_CHOICES)

    gap = 2 if N == 3 else rng.choice(GAP_CHOICES)
    cell_size = N + gap
    border = max(2, N)

    KrKc_options = []
    for Kr in CELL_GRID_DIMS:
        for Kc in CELL_GRID_DIMS:
            if Kr * Kc < V:
                continue
            min_S = max(Kr, Kc) * cell_size + 2 * border
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
    graph = _build_graph_spatial(
        picked,
        cell_to_id,
        rng,
        path_len,
        n_cycles,
        use_diagonal=use_diagonal,
        allow_skip=allow_skip,
    )
    if graph is None:
        return None
    edges, path_nodes, path_edges, cell_to_id = graph
    id_to_cell = {v: k for k, v in cell_to_id.items()}

    path_cells = [id_to_cell[nid] for nid in path_nodes]
    pr = [r for r, _ in path_cells]
    pc = [c for _, c in path_cells]
    cell_bbox = (min(pr), min(pc), max(pr), max(pc))
    for nid in range(V):
        if nid in path_nodes:
            continue
        rr, cc = id_to_cell[nid]
        if cell_bbox[0] <= rr <= cell_bbox[2] and cell_bbox[1] <= cc <= cell_bbox[3]:
            return None

    pool = list(COLOR_POOL)
    rng.shuffle(pool)
    bg = pool[0]
    terminal_color = pool[1]
    station_color = pool[2]
    track_color = pool[3]

    canvas = np.full((S, S), bg, dtype=int)
    node_mask = np.zeros((S, S), dtype=bool)
    id_to_pos: dict[int, tuple[int, int]] = {}
    id_to_pts: dict[int, set[tuple[int, int]]] = {}
    slack_r = S - Kr * cell_size - 2 * border
    slack_c = S - Kc * cell_size - 2 * border
    origin = (
        border + rng.randint(0, max(0, slack_r)),
        border + rng.randint(0, max(0, slack_c)),
    )

    for nid in range(V):
        cell = id_to_cell[nid]
        pos = _cell_to_square_pos(cell, N, cell_size, origin)
        id_to_pos[nid] = pos
        color = terminal_color if nid in (0, V - 1) else station_color
        r0, c0 = pos
        if not (0 <= r0 and 0 <= c0 and r0 + N <= S and c0 + N <= S):
            return None
        canvas[r0 : r0 + N, c0 : c0 + N] = color
        node_mask[r0 : r0 + N, c0 : c0 + N] = True
        id_to_pts[nid] = {(r, c) for r in range(r0, r0 + N) for c in range(c0, c0 + N)}

    track_mask = np.zeros((S, S), dtype=bool)
    edge_track_pixels: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for a, b in edges:
        pixels = _track_between_cells(id_to_cell[a], id_to_cell[b], id_to_pos[a], id_to_pos[b], N)
        if not pixels:
            return None
        for r, c in pixels:
            if not (0 <= r < S and 0 <= c < S):
                return None
            if node_mask[r, c] or track_mask[r, c]:
                return None
        for r, c in pixels:
            canvas[r, c] = track_color
            track_mask[r, c] = True
        edge_track_pixels[tuple(sorted((a, b)))] = pixels

    if np.argmax(np.bincount(canvas.ravel())) != bg:
        return None

    expected_full = canvas.copy()
    path_pixels: set[tuple[int, int]] = set()
    term_ids = {0, V - 1}

    for nid in path_nodes:
        pts = id_to_pts[nid]
        path_pixels.update(pts)
        if nid in term_ids:
            continue
        for r, c in pts:
            expected_full[r, c] = int(Color.GREEN)

    for e in path_edges:
        pts = edge_track_pixels[tuple(sorted(e))]
        path_pixels.update(pts)
        for r, c in pts:
            expected_full[r, c] = int(Color.GREY)

    bbox = _compute_bbox_from_pixels(path_pixels)
    r0, c0, r1, c1 = bbox
    expected = expected_full[r0 : r1 + 1, c0 : c1 + 1]

    # D7 observability / cleanliness checks.
    if expected.shape[0] >= S or expected.shape[1] >= S:
        return None

    all_foreground = canvas != bg
    inner_foreground = all_foreground[r0 : r1 + 1, c0 : c1 + 1]
    inner_path = np.zeros_like(inner_foreground, dtype=bool)
    for r, c in path_pixels:
        inner_path[r - r0, c - c0] = True
    if np.any(inner_foreground & ~inner_path):
        return None
    crop_window = np.zeros_like(all_foreground, dtype=bool)
    crop_window[r0 : r1 + 1, c0 : c1 + 1] = True
    if not np.any(all_foreground & ~crop_window):
        return None

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
