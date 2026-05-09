# Variant of 7b0280bc
# Dimension: D15 — 符号编码参数 (symbol_instructions)
# Change: 无符号编码，终端通过颜色数量直接识别（恰好2个的颜色）... → 图例区域指定涂色

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np

from arc_utils import colors_present, components_mask, get_bg, neighbors8


# ---------------------------------------------------------------------------
# Design analysis for the D15 variant
# ---------------------------------------------------------------------------
#
# a) Concrete behavior and flow impact
#    - Original input: two square colors (exactly-2 terminals + repeated stations) plus one
#      track color. Output always paints path stations green(3) and path tracks grey(5).
#    - Variant input: the graph stays the same, but a 2-cell legend is added at the top-left
#      corner. The first legend cell encodes the path-track paint color; the second legend
#      cell encodes the path-node paint color.
#      * Horizontal legend: (0,0)=track paint, (0,1)=node paint.
#      * Vertical legend:   (0,0)=track paint, (1,0)=node paint.
#    - Variant output: same shortest path as the original task, but recolored with the two
#      legend colors instead of fixed grey/green.
#    - Small trace:
#      * Original solver: infer terminal/station/track colors -> BFS shortest path ->
#        overwrite path stations with 3 and path tracks with 5.
#      * Variant solver: first read the 2-cell legend, temporarily erase those 2 cells from
#        the working grid, then run the exact same graph inference, but overwrite path
#        stations/tracks with the legend colors.
#      * Main difference: color inference can no longer treat every non-bg color as part of
#        the graph, because two extra colors now exist only as symbolic instructions.
#
# b) Complexity calibration
#    - The symbolic instruction is kept to a single 2-cell domino so the change is visible
#      but not overloaded.
#    - The legend colors are always distinct from bg / terminal / station / track colors and
#      from each other. This prevents ambiguous cases such as "recolor to the existing track
#      color", which would make the rule visually invisible.
#    - Output colors are sampled from a pool that excludes green(3) and grey(5), so the task
#      cannot silently collapse back to the original fixed-color behavior.
#
# c) Old implicit assumptions that break under D15
#    - `track_candidates = non_bg_colors - {terminal_color, station_color}` used to work
#      because the only remaining color was the track color. With a legend, there are now
#      extra non-graph colors, so the legend must be removed before this step.
#    - Recoloring with `Color.GREEN` / `Color.GREY` was harmless only because those values
#      were globally fixed. In the variant that would destroy the legend-specified mapping.
#    - The original solver never needed to reason about border objects. The new solver now
#      assumes the only border-touching non-bg pixels are the 2 legend cells.
#
# d) Placement legality
#    - The generator reserves a 2-cell top-left legend plus a 1-cell moat by forcing the
#      graph to stay at least 1 cell away from every border.
#    - The graph embedding logic is otherwise unchanged: squares do not overlap, tracks do
#      not overlap nodes or other tracks, and the graph remains connected with a unique
#      shortest terminal path.
#    - Because the graph never touches row 0 / col 0 / row H-1 / col W-1, the legend is
#      trivially separable from the graph by the solver.
#
# e) Visual observability
#    - Across examples, the shortest-path geometry is still learned from the graph, while
#      the path colors vary. The only stable clue explaining that variation is the 2-cell
#      legend in the corner, so the new behavior is visible in 2-3 examples.
#    - Using only one fixed corner keeps the instruction location obvious; using both
#      horizontal and vertical dominoes adds mild variation without adding search burden.
#
# f) Visual cleanliness
#    - The legend is isolated in the top-left corner and never touches the graph.
#    - No element layering is introduced: the legend is separate, the graph remains separate,
#      and the output only recolors existing graph elements.
#
# g) Color-change inferability
#    - Only two output colors are symbolic. Their roles are supported structurally:
#      one applies to path edges, the other to interior path nodes.
#    - The mapping is local and explicit; there is no hidden ordering by numeric color id.
#
# h) New solver assumptions and how the generator guarantees them
#    - Assumption: `grid[0,0]` is non-bg and starts the legend.
#      Guarantee: generator always places the legend at the top-left corner.
#    - Assumption: exactly one of `(0,1)` or `(1,0)` is the second legend cell.
#      Guarantee: generator chooses exactly one orientation and keeps the other cell bg.
#    - Assumption: no other non-bg pixels touch the border.
#      Guarantee: generator uses border padding and rejects placements that violate it.
# ---------------------------------------------------------------------------


def find_solid_squares(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    min_size: int = 2,
    connectivity: int = 8,
) -> list[dict]:
    """
    Local helper: find solid monochrome NxN squares (N>=min_size).
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


def _extract_legend(g: np.ndarray, bg: int) -> tuple[np.ndarray, int, int, str, list[tuple[int, int]]]:
    """
    Read the top-left 2-cell legend and return:
      (grid_without_legend, track_paint, node_paint, orientation, legend_cells)
    """
    h, w = g.shape
    assert h >= 2 and w >= 2
    assert int(g[0, 0]) != int(bg), "legend must start at the top-left corner"

    right_live = int(g[0, 1]) != int(bg)
    down_live = int(g[1, 0]) != int(bg)
    assert right_live ^ down_live, "legend must be a 2-cell horizontal or vertical domino"

    if right_live:
        track_paint = int(g[0, 0])
        node_paint = int(g[0, 1])
        legend_cells = [(0, 0), (0, 1)]
        orientation = "h"
    else:
        track_paint = int(g[0, 0])
        node_paint = int(g[1, 0])
        legend_cells = [(0, 0), (1, 0)]
        orientation = "v"

    g_graph = np.array(g, dtype=int, copy=True)
    for r, c in legend_cells:
        g_graph[int(r), int(c)] = int(bg)
    return g_graph, track_paint, node_paint, orientation, legend_cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the unique shortest terminal-to-terminal path in a square-and-track graph.

    Variant rule:
    - The top-left 2-cell legend specifies output colors.
    - First legend slot = path-track color.
    - Second legend slot = path-node color.
    - The graph logic itself is unchanged from the base task.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    g_graph, path_track_color, path_node_color, _legend_orientation, _legend_cells = _extract_legend(g, bg)

    # --- detect solid squares (nodes) and infer the dominant size N ---
    square_cands = find_solid_squares(g_graph, bg=bg, min_size=2, connectivity=8)
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

    non_bg_colors = colors_present(g_graph, exclude={bg})
    track_candidates = [c for c in non_bg_colors if int(c) not in (int(terminal_color), int(station_color))]
    assert track_candidates
    track_color = int(max(track_candidates, key=lambda c: int((g_graph == int(c)).sum())))

    # --- build node lookup ---
    pix2node: dict[tuple[int, int], int] = {}
    id2pts: dict[int, set[tuple[int, int]]] = {}
    for nd in nodes:
        nid = int(nd["id"])
        pts = set(nd["pts"])
        id2pts[nid] = pts
        for p in pts:
            pix2node[p] = nid

    # --- build edges from track components that touch exactly 2 nodes (8-adjacency) ---
    track_comps = components_mask(g_graph == int(track_color), connectivity=8)
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

    out = g.copy()

    term_set = set(map(int, term_ids))
    for nid in node_path:
        if int(nid) in term_set:
            continue
        for r, c in id2pts[int(nid)]:
            out[int(r), int(c)] = int(path_node_color)

    for eid in edge_path:
        for r, c in track_comp_pts.get(int(eid), []):
            out[int(r), int(c)] = int(path_track_color)

    return out


# ---------------------------------------------------------------------------
# Generator configuration
# ---------------------------------------------------------------------------

NODE_SIZE_CHOICES = [2, 3]
NODE_COUNT_RANGE = (7, 10)
CYCLE_COUNT_CHOICES = [1, 2]
PATH_LEN_CHOICES = [2, 3]
GAP_CHOICES = [2, 3]
CELL_GRID_DIMS = (3, 4, 5)
GRID_SIZE_RANGE = (15, 30)
COLOR_POOL = [0, 1, 2, 4, 6, 7, 8, 9]
LEGEND_ORIENT_CHOICES = ["h", "v"]
BORDER_PAD = 2
CELL_PICK_RETRIES = 50
GEN_ATTEMPT_RETRIES = 300


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


def _place_legend(canvas: np.ndarray, track_paint: int, node_paint: int, orient: str):
    canvas[0, 0] = int(track_paint)
    if orient == "h":
        canvas[0, 1] = int(node_paint)
    else:
        canvas[1, 0] = int(node_paint)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    N = rng.choice(NODE_SIZE_CHOICES)
    V = rng.randint(*NODE_COUNT_RANGE)
    n_cycles = rng.choice(CYCLE_COUNT_CHOICES)
    path_len = rng.choice(PATH_LEN_CHOICES)
    legend_orient = rng.choice(LEGEND_ORIENT_CHOICES)

    gap = rng.choice(GAP_CHOICES)
    cell_size = N + gap

    KrKc_options = []
    for Kr in CELL_GRID_DIMS:
        for Kc in CELL_GRID_DIMS:
            if Kr * Kc < V:
                continue
            min_S = max(Kr, Kc) * cell_size + 2 * BORDER_PAD
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
    chosen = pool[:6]
    bg, terminal_color, station_color, track_color, path_track_color, path_node_color = chosen

    canvas = np.full((S, S), bg, dtype=int)
    _place_legend(canvas, path_track_color, path_node_color, legend_orient)

    node_mask = np.zeros((S, S), dtype=bool)
    slack_r = S - Kr * cell_size - 2 * BORDER_PAD
    slack_c = S - Kc * cell_size - 2 * BORDER_PAD
    origin = (
        BORDER_PAD + rng.randint(0, max(0, slack_r)),
        BORDER_PAD + rng.randint(0, max(0, slack_c)),
    )
    id_to_pos = {}
    for i in range(V):
        cell = id_to_cell[i]
        pos = _cell_to_square_pos(cell, N, cell_size, origin)
        id_to_pos[i] = pos
        color = terminal_color if i in (0, V - 1) else station_color
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
            if r in (0, S - 1) or c in (0, S - 1):
                return None
            if node_mask[r, c] or track_mask[r, c]:
                return None
        for (r, c) in pixels:
            canvas[r, c] = track_color
            track_mask[r, c] = True
        edge_track_pixels[tuple(sorted((a, b)))] = pixels

    border_non_bg = np.argwhere(canvas != bg)
    for r, c in border_non_bg:
        if (int(r), int(c)) not in {(0, 0), (0, 1), (1, 0)}:
            if int(r) in (0, S - 1) or int(c) in (0, S - 1):
                return None
    if legend_orient == "h" and int(canvas[1, 0]) != int(bg):
        return None
    if legend_orient == "v" and int(canvas[0, 1]) != int(bg):
        return None

    if np.argmax(np.bincount(canvas.ravel())) != bg:
        return None

    expected = canvas.copy()
    for gnode in path_nodes:
        if gnode in (0, V - 1):
            continue
        r0, c0 = id_to_pos[gnode]
        expected[r0 : r0 + N, c0 : c0 + N] = path_node_color
    for e in path_edges:
        for (r, c) in edge_track_pixels[tuple(sorted(e))]:
            expected[r, c] = path_track_color

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
