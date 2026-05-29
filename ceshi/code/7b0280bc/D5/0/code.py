from __future__ import annotations

# Variant of 7b0280bc
# Dimension: D5 — 终点/停止条件 (dst)
# Change: 另一个少数色终端方块为终点，最短路径到达即停止 → 距离编码终点
#
# Design analysis
# ---------------
# spec:
#   input_objects:
#     bg: most common color
#     source_terminal:
#       shape: one solid NxN square in terminal_color
#       size_N: inferred from the dominant solid-square size
#     encoded_terminal:
#       shape: one NxN square whose pixels are terminal_color plus exactly k grey pixels
#       meaning: grey pixel count k = how many graph edges to traverse from the source
#     stations:
#       shape: 5-8 solid NxN station_color squares
#     tracks:
#       shape: one track color, 8-connected components, each touching exactly 2 node squares
#   output_rule:
#     - Build the same graph as the original task.
#     - Find the unique shortest path from source_terminal to encoded_terminal.
#     - Let k be the number of grey pixels inside encoded_terminal.
#     - Recolor only the first k track segments on that path to grey(5).
#     - Recolor stations reached within those k steps to green(3).
#     - Do not recolor either terminal, even if k would land exactly on the encoded terminal.
#
# a) Concrete behavior trace:
#   Original: if shortest path is A - s1 - s2 - B, then edges (A,s1),(s1,s2),(s2,B) all become grey,
#   and s1,s2 become green.
#   Variant: keep the same graph, but B contains k grey pixels. If k=2, only
#   edges (A,s1),(s1,s2) become grey and only s1,s2 become green; B stays unchanged.
#   The new control point is no longer "reach the other terminal", but "consume k edges".
#
# b) Complexity calibration:
#   The distance code uses only 3 values (k in {1,2,3}), repeated across examples.
#   This is enough to create visibly different stop points without exploding the hypothesis space.
#   To make the count legible, generator fixes N=3; with 2x2 terminals the count is much harder to read.
#
# c) Hidden assumptions in the original solver that no longer hold:
#   - "Both terminals are solid squares of the same color": false now; one terminal is mixed terminal_color+grey.
#   - "Grey never appears in the input": false now; grey is a semantic marker inside the encoded terminal.
#   - "Terminal color appears exactly twice among solid squares": false now; it appears exactly once.
#   Therefore terminal detection must be rewritten before graph construction.
#
# d) Placement legality:
#   Generator still uses graph-first placement on a cell grid, with non-overlap checks for squares and tracks.
#   Encoded-terminal greys are painted only inside the already reserved node square, so they cannot collide with tracks.
#
# e) Visual observability:
#   The changed rule is visible in every generated sample because:
#   - one terminal visibly contains grey pixels;
#   - the recolored path stops before that terminal in every sample.
#   This avoids degeneration back to the original "color the full terminal-to-terminal path".
#
# f) Visual cleanliness:
#   Grey is used in exactly two roles: the compact cluster inside one terminal, and the recolored path prefix.
#   These regions are spatially distinct and do not overlap.
#
# g) Color inference:
#   No hidden numeric ordering is used. The only numerical cue is the visibly countable number of grey pixels.
#   The output structure itself shows that the grey count determines how far the recoloring propagates.

from collections import Counter, defaultdict, deque
import random
import numpy as np
from arc_utils import Color, components_mask, colors_present, get_bg, neighbors8


def find_solid_squares(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    min_size: int = 2,
    connectivity: int = 8,
) -> list[dict]:
    """Find solid monochrome NxN squares."""
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


def _find_encoded_terminal_components(grid: np.ndarray, terminal_color: int, size_n: int) -> list[dict]:
    """
    Find NxN components formed by {terminal_color, grey} that contain at least one grey pixel.

    This is the key adaptation for the D5 variant: one terminal is no longer solid.
    """
    mask = (grid == int(terminal_color)) | (grid == int(Color.GREY))
    out: list[dict] = []
    for coords in components_mask(mask, connectivity=8):
        if coords.size == 0:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if h != int(size_n) or w != int(size_n):
            continue
        if int(coords.shape[0]) != int(size_n * size_n):
            continue
        vals = [int(grid[int(r), int(c)]) for r, c in coords]
        if any(v not in (int(terminal_color), int(Color.GREY)) for v in vals):
            continue
        grey_count = sum(1 for v in vals if int(v) == int(Color.GREY))
        if grey_count <= 0:
            continue
        out.append(
            {
                "pts": {(int(r), int(c)) for r, c in coords},
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "grey_count": int(grey_count),
            }
        )
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor only a prefix of the shortest path, where the prefix length is encoded
    by the number of grey pixels inside the encoded terminal.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    # Infer the canonical node size from solid squares, ignoring grey because grey now
    # appears only as the distance code inside one terminal.
    square_cands = [
        d for d in find_solid_squares(g, bg=bg, min_size=2, connectivity=8)
        if int(d["color"]) != int(Color.GREY)
    ]
    assert not (not square_cands)

    size_counts = Counter(int(d["size"]) for d in square_cands)
    size_n = int(max(size_counts.items(), key=lambda kv: (int(kv[1]), int(kv[0])))[0])
    mono_nodes_raw = [d for d in square_cands if int(d["size"]) == int(size_n)]

    by_color: dict[int, list[dict]] = defaultdict(list)
    for d in mono_nodes_raw:
        by_color[int(d["color"])].append(d)

    # Terminal color now has:
    # - exactly one solid square (the source)
    # - exactly one mixed square with grey pixels (the encoded terminal)
    terminal_color = None
    source_square = None
    encoded_square = None
    for col, solids in by_color.items():
        if len(solids) != 1:
            continue
        mixed = _find_encoded_terminal_components(g, int(col), size_n)
        if len(mixed) == 1:
            terminal_color = int(col)
            source_square = solids[0]
            encoded_square = mixed[0]
            break

    assert not (terminal_color is None or source_square is None or encoded_square is None)

    station_color = None
    best_station_count = -1
    for col, solids in by_color.items():
        if int(col) == int(terminal_color):
            continue
        if len(solids) > best_station_count:
            station_color = int(col)
            best_station_count = len(solids)
    assert not (station_color is None or best_station_count <= 0)

    track_candidates = [
        c
        for c in colors_present(g, exclude={bg})
        if int(c) not in (int(terminal_color), int(station_color), int(Color.GREY))
    ]
    assert not (not track_candidates)
    track_color = int(max(track_candidates, key=lambda c: int((g == int(c)).sum())))

    # Build node table: source terminal, stations, encoded terminal.
    nodes: list[dict] = []

    def add_node(kind: str, pts: set[tuple[int, int]], color: int, grey_count: int = 0):
        nodes.append(
            {
                "id": int(len(nodes)),
                "kind": str(kind),
                "color": int(color),
                "pts": set(pts),
                "grey_count": int(grey_count),
            }
        )

    add_node(
        "source",
        {(int(r), int(c)) for r, c in source_square["coords"]},
        int(terminal_color),
        grey_count=0,
    )
    for d in by_color[int(station_color)]:
        add_node(
            "station",
            {(int(r), int(c)) for r, c in d["coords"]},
            int(station_color),
            grey_count=0,
        )
    add_node(
        "encoded",
        set(encoded_square["pts"]),
        int(terminal_color),
        grey_count=int(encoded_square["grey_count"]),
    )

    source_id = next(int(nd["id"]) for nd in nodes if nd["kind"] == "source")
    encoded_id = next(int(nd["id"]) for nd in nodes if nd["kind"] == "encoded")
    stop_len = int(next(nd["grey_count"] for nd in nodes if nd["kind"] == "encoded"))

    pix2node: dict[tuple[int, int], int] = {}
    id2pts: dict[int, set[tuple[int, int]]] = {}
    id2kind: dict[int, str] = {}
    for nd in nodes:
        nid = int(nd["id"])
        id2pts[nid] = set(nd["pts"])
        id2kind[nid] = str(nd["kind"])
        for p in nd["pts"]:
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

    q = deque([source_id])
    prev: dict[int, int | None] = {source_id: None}
    prev_edge: dict[int, int | None] = {source_id: None}

    while q:
        u = int(q.popleft())
        if u == encoded_id:
            break
        for v, eid in adj.get(u, []):
            if int(v) in prev:
                continue
            prev[int(v)] = int(u)
            prev_edge[int(v)] = int(eid)
            q.append(int(v))

    assert not (encoded_id not in prev)

    node_path: list[int] = []
    edge_path: list[int] = []
    cur = int(encoded_id)
    while cur is not None:
        node_path.append(int(cur))
        e = prev_edge.get(int(cur))
        if e is not None:
            edge_path.append(int(e))
        cur2 = prev.get(int(cur))
        cur = int(cur2) if cur2 is not None else None
    node_path.reverse()
    edge_path.reverse()

    assert 1 <= int(stop_len) <= len(edge_path)

    out = g.copy()
    prefix_nodes = node_path[1 : stop_len + 1]
    prefix_edges = edge_path[:stop_len]

    for nid in prefix_nodes:
        if id2kind[int(nid)] != "station":
            continue
        for r, c in id2pts[int(nid)]:
            out[int(r), int(c)] = int(Color.GREEN)

    for eid in prefix_edges:
        for r, c in track_comp_pts.get(int(eid), []):
            out[int(r), int(c)] = int(Color.GREY)

    return out


# --------------- generator ---------------

NODE_SIZE_CHOICES = [3]                  # fixed for count readability
NODE_COUNT_RANGE = (7, 10)               # 2 terminals + 5-8 stations
CYCLE_COUNT_CHOICES = [1, 2]
TOTAL_PATH_LEN_CHOICES = [3, 4]          # terminal-to-encoded-terminal distance
STOP_EDGE_COUNT_CHOICES = [1, 2, 3]      # grey-count encoding inside encoded terminal
GAP_CHOICES = [2, 3]
CELL_GRID_DIMS = (3, 4, 5)
GRID_SIZE_RANGE = (15, 30)
COLOR_POOL = [0, 1, 2, 4, 6, 7, 8, 9]    # excludes green=3 and grey=5
CELL_PICK_RETRIES = 50
GEN_ATTEMPT_RETRIES = 300
ENCODING_ORDER = [(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (1, 2), (2, 0), (2, 1), (2, 2)]


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


def _pick_cells(
    Kr: int,
    Kc: int,
    V: int,
    rng: random.Random,
    use_diagonal: bool,
    allow_skip: bool,
):
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
    total_path_len: int,
    n_cycles: int,
    use_diagonal: bool,
    allow_skip: bool,
):
    """
    Build a connected graph with a unique shortest path between the two terminals.

    The variant still depends on the original terminal-to-terminal shortest path;
    only the stopping condition changes later.
    """
    V = len(picked)
    cell_edges = _compute_adjacency(picked, use_diagonal=use_diagonal, allow_skip=allow_skip)
    full_edges = {tuple(sorted((cell_to_id[a], cell_to_id[b]))) for (a, b) in cell_edges}

    ids = list(range(V))
    rng.shuffle(ids)
    t0, t1 = ids[0], ids[1]

    dist, cnt, adj = _bfs_with_counts(t0, full_edges, V)
    if t1 not in dist or dist[t1] != total_path_len:
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
        if dist2.get(t1) == total_path_len and cnt2.get(t1) == 1:
            final_edges.add(e)
            added += 1
    if added != n_cycles:
        return None

    dist3, cnt3, _ = _bfs_with_counts(t0, final_edges, V)
    if dist3.get(t1) != total_path_len or cnt3.get(t1) != 1:
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


def _dock_tl(sq_r: int, sq_c: int, N: int, Wt: int, si: int, sj: int):
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


def _paint_encoded_terminal(canvas: np.ndarray, top_left: tuple[int, int], terminal_color: int, stop_len: int):
    r0, c0 = top_left
    for dr, dc in ENCODING_ORDER[:stop_len]:
        canvas[int(r0 + dr), int(c0 + dc)] = int(Color.GREY)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    N = rng.choice(NODE_SIZE_CHOICES)
    V = rng.randint(*NODE_COUNT_RANGE)
    n_cycles = rng.choice(CYCLE_COUNT_CHOICES)
    total_path_len = rng.choice(TOTAL_PATH_LEN_CHOICES)
    stop_choices = [k for k in STOP_EDGE_COUNT_CHOICES if int(k) < int(total_path_len)]
    if not stop_choices:
        return None
    stop_len = rng.choice(stop_choices)

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
        total_path_len,
        n_cycles,
        use_diagonal=use_diagonal,
        allow_skip=allow_skip,
    )
    if gr is None:
        return None
    edges, path_nodes, path_edges, cell_to_id = gr
    id_to_cell = {v: k for k, v in cell_to_id.items()}

    pool = list(COLOR_POOL)
    rng.shuffle(pool)
    bg, terminal_color, station_color, track_color = pool[:4]

    canvas = np.full((S, S), bg, dtype=int)
    node_mask = np.zeros((S, S), dtype=bool)
    slack_r = S - Kr * cell_size - 2
    slack_c = S - Kc * cell_size - 2
    origin = (1 + rng.randint(0, max(0, slack_r)), 1 + rng.randint(0, max(0, slack_c)))

    id_to_pos: dict[int, tuple[int, int]] = {}
    for i in range(V):
        cell = id_to_cell[i]
        pos = _cell_to_square_pos(cell, N, cell_size, origin)
        id_to_pos[i] = pos
        color = terminal_color if i in (0, V - 1) else station_color
        r0, c0 = pos
        canvas[r0 : r0 + N, c0 : c0 + N] = color
        node_mask[r0 : r0 + N, c0 : c0 + N] = True

    _paint_encoded_terminal(canvas, id_to_pos[V - 1], terminal_color, stop_len)

    track_mask = np.zeros((S, S), dtype=bool)
    edge_track_pixels: dict[tuple[int, int], list[tuple[int, int]]] = {}
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
    prefix_nodes = path_nodes[1 : stop_len + 1]
    prefix_edges = path_edges[:stop_len]
    for nid in prefix_nodes:
        if int(nid) in (0, V - 1):
            continue
        r0, c0 = id_to_pos[int(nid)]
        expected[r0 : r0 + N, c0 : c0 + N] = int(Color.GREEN)
    for e in prefix_edges:
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
