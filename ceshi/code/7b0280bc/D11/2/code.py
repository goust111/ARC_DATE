# Variant of 7b0280bc
# Dimension: D11 — 颜色方案 (meta)
# Change: 4种颜色（背景、终端、中间站、轨道），不含绿(3)和灰(5)，输出中绿和灰是固定的涂色... → 五色方案加干扰色

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np

from arc_utils import Color, colors_present, components_mask, get_bg, neighbors8


# ---------------------------------------------------------------------------
# Design analysis for the D11 variant
# ---------------------------------------------------------------------------
#
# spec_variant:
#   base_task: shortest path on a graph embedded as node-squares + track components
#   unchanged:
#     - grid size remains 15..30 square
#     - graph size / path length / recolor rule remain unchanged
#     - output still paints path stations green(3) and path tracks grey(5)
#   changed_core:
#     - input now uses 5 non-output roles: bg, terminal, station, track, distractor
#     - distractor is a fifth non-functional color, scattered as isolated background pixels
#     - distractor never touches nodes, tracks, or other distractor pixels in 8-neighborhood
#   visual_form:
#     - humans still see the same graph of colored squares and rails
#     - additionally, sparse "dust" pixels of another color appear in empty background
#     - these pixels are structurally irrelevant and remain unchanged in the output
#
# a) Concrete behavior trace
#   Original solver step:
#     - infer terminal/station colors from solid NxN squares
#     - among remaining non-bg colors, pick the one with largest area as track color
#   Variant requirement:
#     - after square colors are removed, there are now two remaining colors:
#       true track and distractor
#     - area is no longer a safe discriminator
#     - solver must identify the track color by geometry: its connected components act
#       as graph edges because they touch exactly two node squares
#   Small example:
#     - suppose terminal=2, station=8, track=1, distractor=6, bg=9
#     - if track has 18 pixels total and distractor has 23 isolated pixels, the old
#       solver picks 6 and then fails because those components touch 0 nodes
#     - the new solver tests both remaining colors and chooses 1 because only color 1
#       yields many components with exactly two touched nodes
#
# b) Complexity calibration
#   - distractor is limited to one repeated visual motif: isolated single pixels
#   - this keeps the new rule visible but not ambiguous; humans can infer "ignore the
#     extra dust color" from 2-4 examples
#   - we do not introduce extra path logic, alternate outputs, or mixed distractor shapes
#
# c) Hidden-assumption audit of the original main()
#   - find_solid_squares(... exclude bg ...):
#       Assumption: only node colors produce solid NxN squares. Still valid because the
#       generator forbids distractor adjacency and therefore any distractor NxN block.
#   - node color inference via counts {2 terminals, many stations}:
#       Assumption: exactly two node colors exist. Still valid.
#   - track_candidates = all other non-bg colors:
#       Assumption: there is exactly one non-node, non-bg color. No longer valid.
#   - track_color = max(track_candidates, key=area):
#       Assumption: the track color is the only or largest residual color. Invalid in D11.
#   - components_mask(g == track_color):
#       Assumption: the previously chosen color is truly the edge color. Must be replaced
#       with geometry-based identification before edge extraction.
#
# d) Placement legality
#   - distractor pixels are only placed on background cells
#   - each distractor pixel must be 8-separated from every non-bg structure and from every
#     other distractor pixel, so it cannot merge into a fake edge or touch a node
#   - generator validates final input by running main(input) and comparing against the
#     expected output, preserving the original output-first safety net
#
# e) Visual observability
#   - the only new visual signal is "extra background-colored dust of one additional color"
#   - output ignores that dust and still highlights the same shortest route
#   - across examples, the dust color varies but always behaves as unchanged background noise
#
# f) Visual cleanliness
#   - isolated pixels avoid overlaps and avoid forming mini-paths
#   - a one-cell 8-neighborhood exclusion around structure prevents clutter around rails
#
# g) Color-change inferability
#   - only the original path recolors remain semantically active: station -> green,
#     track -> grey
#   - distractor does not recolor, which gives a clear before/after contrast: some extra
#     color is present, but it never participates in the transformation


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


def _component_touched_nodes(
    pts: list[tuple[int, int]],
    pix2node: dict[tuple[int, int], int],
) -> tuple[int, ...]:
    touched: set[int] = set()
    for r, c in pts:
        for rr, cc in neighbors8(int(r), int(c)):
            nid = pix2node.get((int(rr), int(cc)))
            if nid is not None:
                touched.add(int(nid))
    return tuple(sorted(touched))


def _select_track_color(
    g: np.ndarray,
    candidate_colors: list[int],
    pix2node: dict[tuple[int, int], int],
) -> tuple[int, dict[int, list[tuple[int, int]]], dict[int, list[tuple[int, int]]]]:
    """
    Select the true track color by geometry instead of area.

    For each residual color, keep only connected components that touch exactly two nodes.
    The real track color has many such components; distractor color has none because its
    pixels are isolated in background.
    """
    best_color = None
    best_components: dict[int, list[tuple[int, int]]] = {}
    best_edges: dict[int, list[tuple[int, int]]] = {}
    best_score = (-1, -1)

    for col in candidate_colors:
        comp_pts: dict[int, list[tuple[int, int]]] = {}
        edge_nodes: dict[int, list[tuple[int, int]]] = {}
        for edge_id, coords in enumerate(components_mask(g == int(col), connectivity=8), start=1):
            pts = [(int(r), int(c)) for r, c in coords]
            if not pts:
                continue
            touched = _component_touched_nodes(pts, pix2node)
            if len(touched) != 2:
                continue
            comp_pts[int(edge_id)] = pts
            edge_nodes[int(edge_id)] = [(int(touched[0]), int(touched[1]))]

        score = (
            len(comp_pts),
            sum(len(pts) for pts in comp_pts.values()),
        )
        if score > best_score:
            best_score = score
            best_color = int(col)
            best_components = comp_pts
            best_edges = edge_nodes

    assert best_color is not None
    assert best_score[0] > 0
    return int(best_color), best_components, best_edges


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the shortest path between two terminal squares in a square-and-track graph.

    D11 variant: one extra distractor color may appear as isolated background pixels.
    These pixels must be ignored when inferring the graph.
    """
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

    pix2node: dict[tuple[int, int], int] = {}
    id2pts: dict[int, set[tuple[int, int]]] = {}
    for nd in nodes:
        nid = int(nd["id"])
        pts = set(nd["pts"])
        id2pts[nid] = pts
        for p in pts:
            pix2node[p] = nid

    non_bg_colors = colors_present(g, exclude={bg})
    track_candidates = [
        c
        for c in non_bg_colors
        if int(c) not in (int(terminal_color), int(station_color))
    ]
    assert track_candidates

    _track_color, track_comp_pts, track_comp_nodes = _select_track_color(
        g,
        track_candidates,
        pix2node,
    )

    adj: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for edge_id, pair_list in track_comp_nodes.items():
        a, b = pair_list[0]
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


# ---------------------------------------------------------------------------
# Generator config
# ---------------------------------------------------------------------------

NODE_SIZE_CHOICES = [2, 3]
NODE_COUNT_RANGE = (7, 10)
CYCLE_COUNT_CHOICES = [1, 2]
PATH_LEN_CHOICES = [2, 3]
GAP_CHOICES = [2, 3]
CELL_GRID_DIMS = (3, 4, 5)
GRID_SIZE_RANGE = (15, 30)
COLOR_POOL = [0, 1, 2, 4, 6, 7, 8, 9]
CELL_PICK_RETRIES = 50
GEN_ATTEMPT_RETRIES = 400


def _compute_adjacency(
    picked: set[tuple[int, int]],
    use_diagonal: bool,
    allow_skip: bool,
):
    edges = set()
    for i, j in picked:
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            n = (i + di, j + dj)
            if n in picked:
                edges.add(tuple(sorted([(i, j), n])))
        if allow_skip:
            for di, dj in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                mid = (i + di // 2, j + dj // 2)
                far = (i + di, j + dj)
                if far in picked and mid not in picked:
                    edges.add(tuple(sorted([(i, j), far])))
        if use_diagonal:
            for di, dj in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
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
    V = len(picked)
    cell_edges = _compute_adjacency(picked, use_diagonal=use_diagonal, allow_skip=allow_skip)
    full_edges = {tuple(sorted((cell_to_id[a], cell_to_id[b]))) for a, b in cell_edges}

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
    sp_edges = {
        tuple(sorted((sp_nodes[i], sp_nodes[i + 1])))
        for i in range(len(sp_nodes) - 1)
    }

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
    remapped_edges = {tuple(sorted((id_map[a], id_map[b]))) for a, b in final_edges}
    remapped_cell_to_id = {cell: id_map[cell_to_id[cell]] for cell in cell_to_id}
    new_path_nodes = [id_map[n] for n in sp_nodes]
    new_path_edges = [
        tuple(sorted((new_path_nodes[i], new_path_nodes[i + 1])))
        for i in range(len(new_path_nodes) - 1)
    ]
    return list(remapped_edges), new_path_nodes, new_path_edges, remapped_cell_to_id


def _cell_to_square_pos(
    cell: tuple[int, int],
    N: int,
    cell_size: int,
    origin: tuple[int, int],
):
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
) -> list[tuple[int, int]]:
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


def _safe_distractor_candidates(
    canvas: np.ndarray,
    bg: int,
    structure_mask: np.ndarray,
) -> list[tuple[int, int]]:
    H, W = canvas.shape
    out = []
    for r in range(H):
        for c in range(W):
            if int(canvas[r, c]) != int(bg):
                continue
            ok = True
            for rr in range(max(0, r - 1), min(H, r + 2)):
                for cc in range(max(0, c - 1), min(W, c + 2)):
                    if structure_mask[rr, cc]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                out.append((int(r), int(c)))
    return out


def _pick_isolated_sites(
    candidates: list[tuple[int, int]],
    target: int,
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    cand = list(candidates)
    rng.shuffle(cand)
    chosen: list[tuple[int, int]] = []
    blocked: set[tuple[int, int]] = set()
    for r, c in cand:
        if (int(r), int(c)) in blocked:
            continue
        chosen.append((int(r), int(c)))
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                blocked.add((int(rr), int(cc)))
        if len(chosen) >= target:
            return chosen
    return None


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

    lower_S = max(GRID_SIZE_RANGE[0], min_S)
    if lower_S + 1 <= GRID_SIZE_RANGE[1]:
        S = rng.randint(lower_S + 1, GRID_SIZE_RANGE[1])
    else:
        S = lower_S

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
    bg, terminal_color, station_color, track_color, distractor_color = pool[:5]

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

    track_mask = np.zeros((S, S), dtype=bool)
    edge_track_pixels: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for a, b in edges:
        pixels = _track_between_cells(
            id_to_cell[a],
            id_to_cell[b],
            id_to_pos[a],
            id_to_pos[b],
            N,
        )
        for r, c in pixels:
            if not (0 <= r < S and 0 <= c < S):
                return None
            if node_mask[r, c] or track_mask[r, c]:
                return None
        for r, c in pixels:
            canvas[r, c] = track_color
            track_mask[r, c] = True
        edge_track_pixels[tuple(sorted((a, b)))] = pixels

    structure_mask = node_mask | track_mask
    track_pixel_count = int(track_mask.sum())

    candidates = _safe_distractor_candidates(canvas, bg, structure_mask)
    # Make the distractor visually consequential: old area-based track inference fails
    # whenever distractor pixels outnumber true track pixels.
    target_noise = track_pixel_count + rng.randint(1, 4)
    distractor_sites = _pick_isolated_sites(candidates, target_noise, rng)
    if distractor_sites is None:
        return None

    for r, c in distractor_sites:
        canvas[r, c] = distractor_color

    if np.argmax(np.bincount(canvas.ravel(), minlength=10)) != bg:
        return None

    expected = canvas.copy()
    for gnode in path_nodes:
        if gnode in (0, V - 1):
            continue
        r0, c0 = id_to_pos[gnode]
        expected[r0 : r0 + N, c0 : c0 + N] = int(Color.GREEN)
    for e in path_edges:
        for r, c in edge_track_pixels[tuple(sorted(e))]:
            expected[r, c] = int(Color.GREY)

    try:
        actual = main(canvas)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    # Anti-degeneration check: the old "largest residual color is track" heuristic must
    # disagree with the real track color on this sample.
    residual_counts = {
        int(track_color): int((canvas == int(track_color)).sum()),
        int(distractor_color): int((canvas == int(distractor_color)).sum()),
    }
    if max(residual_counts, key=residual_counts.get) != int(distractor_color):
        return None

    return canvas


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPT_RETRIES):
        out = _gen_attempt(rng)
        if out is not None:
            return out
    raise RuntimeError(f"generate_input: failed after {GEN_ATTEMPT_RETRIES} attempts")
