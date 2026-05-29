# Variant of 7b0280bc
# Dimension: D4 — 源对象定义 (src)
# Change: 数量为2的少数色NxN实心方块作为终端节点对... → 方块内嵌标记
#
# Design analysis
# ===============
# variant_spec:
#   visual_form:
#     - Keep the original "graph of square nodes + track edges" task family.
#     - All graph nodes now share one common base color.
#     - Node size is fixed to 3x3 so the marker can live at a visually obvious interior
#       location: the center cell.
#     - Station nodes are solid 3x3 squares of the base color.
#     - Terminal nodes are also 3x3 squares of the same base color, except their center
#       pixel is replaced by one marker color. Thus the terminal's base-color component
#       is an 8-cell ring with bbox 3x3.
#     - Track segments keep the original role and geometry: 8-connected width-2 strips
#       joining node pairs.
#   behavior_trace:
#     - Original logic: detect solid squares by monochrome color; the color that occurs
#       on exactly two squares defines the terminal pair.
#     - Variant logic: detect 3x3 node-shaped base-color components; exactly two of them
#       have area 8 instead of 9, and their missing center is filled by the marker color.
#       Those two marked nodes are the terminals. After that, graph extraction and BFS
#       shortest-path reasoning stay the same.
#     - Concrete example: a terminal used to be
#           2 2 2
#           2 2 2
#           2 2 2
#       while a station used another full color. In the variant both use the same base:
#           2 2 2      2 2 2
#           2 7 2  vs  2 2 2
#           2 2 2      2 2 2
#       The solver must inspect the center pixel, not just the outline color.
#   complexity_calibration:
#     - Fixing node size to 3 is a necessary compatibility change, not a new conceptual
#       rule: a 2x2 square has no unique center, so the "embedded marker" would become
#       an arbitrary corner and the D4 change would be visually weak.
#     - The marker uses one shared color across both terminals. This keeps the number of
#       colors at four total (bg, node base, track, marker), matching the original task's
#       color complexity and making the new cue repeated exactly twice per grid.
#     - The shortest path is still unique and 2-3 edges long, so the new difficulty is
#       object identification rather than graph ambiguity.
#   implicit_assumption_audit:
#     - Original `find_solid_squares(...)` assumed every node is monochrome and fully
#       filled; that fails immediately on marked terminals because their base-color
#       component has area 8, not 9.
#     - Original terminal detection assumed "one node color appears exactly twice".
#       That is no longer true because all nodes share the same base color.
#     - Original track-color detection excluded the two node colors; now the solver must
#       first infer base color and marker color, then the remaining non-bg color is track.
#     - Recoloring entire intermediate nodes is still safe, because only stations are
#       recolored and stations are solid 3x3 squares without marker pixels.
#   placement_legality:
#     - Generator keeps the graph-first spatial layout from the original task.
#     - Squares do not overlap, tracks do not overlap nodes or other tracks, and the
#       center marker is written only after the full 3x3 node footprint is placed.
#     - The generator validates that solver-side node detection finds exactly the intended
#       node set, preventing accidental fake 3x3 node patterns elsewhere.
#   visual_observability:
#     - Across examples, every node has the same outline color, so a human cannot use the
#       old minority-color heuristic. The only repeated clue identifying the source pair is
#       the center marker inside exactly two nodes.
#     - Because the marker sits at the center of a 3x3 square and appears twice per grid,
#       it is easy to notice in 2-4 examples and clearly tied to the unchanged shortest-
#       path recoloring behavior.
#   cleanliness:
#     - There is only one layer of local detail inside nodes: a single center pixel.
#       Tracks never enter node interiors, so the grid remains readable.
#   color_inference:
#     - Color usage stays structured: one base color for all nodes, one marker color for
#       both terminals, one track color, one background. No hidden ordering by color id.
#   new_solver_assumptions_checked_against_generator:
#     - Solver assumes node size is 3x3. Generator enforces that.
#     - Solver assumes exactly two node bboxes are 3x3 rings (area 8) with one shared
#       non-bg center marker color. Generator enforces that.
#     - Solver assumes the remaining non-bg color is the track color and that each track
#       component touches exactly two nodes. Generator still enforces the original graph
#       constraints and validates against `main(...)`.

from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np

from arc_utils import Color, colors_present, components_mask, get_bg, neighbors8


NODE_SIZE = 3


def _detect_marked_nodes(grid: np.ndarray, *, bg: int | None = None) -> tuple[int, int, list[dict]]:
    """
    Detect 3x3 node squares sharing one base color.

    A station is a solid 3x3 component of the base color (area 9).
    A terminal is a 3x3 ring component of the base color (area 8) whose center cell is a
    non-bg marker color.

    Returns:
      (base_color, marker_color, nodes)

    nodes entries contain:
      - id
      - bbox
      - pts: full 3x3 footprint, including the center marker if present
      - center_color
      - is_terminal
    """
    g = np.asarray(grid, dtype=int)
    if bg is None:
        bg = get_bg(g)
    bg = int(bg)

    by_color: dict[int, list[dict]] = defaultdict(list)
    for col in colors_present(g, exclude={bg}):
        for coords in components_mask(g == int(col), connectivity=8):
            if coords.size == 0:
                continue
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            area = int(coords.shape[0])
            if h != NODE_SIZE or w != NODE_SIZE or area not in (8, 9):
                continue
            center = (int(r0) + 1, int(c0) + 1)
            center_color = int(g[center])
            if area == 8 and center_color in (bg, int(col)):
                continue
            if area == 9 and center_color != int(col):
                continue
            bbox = (int(r0), int(c0), int(r1), int(c1))
            pts = {
                (r, c)
                for r in range(int(r0), int(r0) + NODE_SIZE)
                for c in range(int(c0), int(c0) + NODE_SIZE)
            }
            by_color[int(col)].append(
                {
                    "bbox": bbox,
                    "pts": pts,
                    "area": area,
                    "center_color": center_color,
                    "is_terminal": area == 8,
                }
            )

    assert by_color, "no node-like 3x3 components found"

    base_color = int(max(by_color.items(), key=lambda kv: (len(kv[1]), kv[0]))[0])
    nodes_raw = sorted(by_color[base_color], key=lambda d: d["bbox"])
    assert 7 <= len(nodes_raw) <= 10, "unexpected node count"

    terminals = [d for d in nodes_raw if bool(d["is_terminal"])]
    stations = [d for d in nodes_raw if not bool(d["is_terminal"])]
    assert len(terminals) == 2, "expected exactly two marked terminals"
    assert len(stations) >= 5, "expected at least five stations"

    marker_colors = {int(d["center_color"]) for d in terminals}
    assert len(marker_colors) == 1, "terminal markers must share one color"
    marker_color = int(next(iter(marker_colors)))
    assert marker_color != base_color and marker_color != bg

    nodes: list[dict] = []
    for i, d in enumerate(nodes_raw):
        nodes.append(
            {
                "id": int(i),
                "bbox": d["bbox"],
                "pts": set(d["pts"]),
                "center_color": int(d["center_color"]),
                "is_terminal": bool(d["is_terminal"]),
            }
        )

    return base_color, marker_color, nodes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the shortest path between two marker-bearing terminal squares.

    Input structure:
    - bg = most common color.
    - All nodes are 3x3 squares sharing one base color.
    - Exactly two terminal nodes are identified by a distinct center-pixel marker color.
    - A third non-bg color forms 8-connected track segments connecting node pairs.

    Rule:
    - Find the shortest path (fewest edges) between the two marked terminal nodes.
    - Recolor every station square on that path to green.
    - Recolor every track segment along that path to grey.
    - Leave terminal markers and all unrelated objects unchanged.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    base_color, marker_color, nodes = _detect_marked_nodes(g, bg=bg)
    term_ids = [int(nd["id"]) for nd in nodes if bool(nd["is_terminal"])]

    non_bg_colors = colors_present(g, exclude={bg})
    track_candidates = [
        int(c)
        for c in non_bg_colors
        if int(c) not in {int(base_color), int(marker_color)}
    ]
    assert len(track_candidates) == 1, "expected one track color"
    track_color = int(track_candidates[0])

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

    assert goal in prev, "terminals are not connected by the extracted graph"

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

# Layout and graph parameters intentionally stay close to the original task.
NODE_COUNT_RANGE = (7, 10)
CYCLE_COUNT_CHOICES = [1, 2]
PATH_LEN_CHOICES = [2, 3]
GAP_CHOICES = [2, 3]
CELL_GRID_DIMS = (3, 4, 5)
GRID_SIZE_RANGE = (15, 30)
COLOR_POOL = [0, 1, 2, 4, 6, 7, 8, 9]
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
    remapped_edges = {tuple(sorted((id_map[a], id_map[b]))) for (a, b) in final_edges}
    remapped_cell_to_id = {cell: id_map[cell_to_id[cell]] for cell in cell_to_id}
    new_path_nodes = [id_map[n] for n in sp_nodes]
    new_path_edges = [
        tuple(sorted((new_path_nodes[i], new_path_nodes[i + 1])))
        for i in range(len(new_path_nodes) - 1)
    ]

    return list(remapped_edges), new_path_nodes, new_path_edges, remapped_cell_to_id


def _cell_to_square_pos(cell: tuple[int, int], cell_size: int, origin: tuple[int, int]):
    i, j = cell
    or_r, or_c = origin
    margin = (cell_size - NODE_SIZE) // 2
    return (or_r + i * cell_size + margin, or_c + j * cell_size + margin)


def _dock_tl(sq_r: int, sq_c: int, Wt: int, si: int, sj: int):
    if si == -1:
        return sq_r - Wt, sq_c
    if si == 1:
        return sq_r + NODE_SIZE, sq_c + NODE_SIZE - Wt
    if sj == -1:
        return sq_r + NODE_SIZE - Wt, sq_c - Wt
    if sj == 1:
        return sq_r, sq_c + NODE_SIZE
    raise AssertionError("invalid direction for dock")


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
) -> list[tuple[int, int]]:
    Wt = NODE_SIZE - 1
    ra, ca = pos_a
    rb, cb = pos_b
    dci, dcj = cell_b[0] - cell_a[0], cell_b[1] - cell_a[1]
    si = 0 if dci == 0 else (1 if dci > 0 else -1)
    sj = 0 if dcj == 0 else (1 if dcj > 0 else -1)

    sr, sc = _dock_tl(ra, ca, Wt, si, sj)
    er, ec = _dock_tl(rb, cb, Wt, -si, -sj)

    pts_tl = _walk_footprints(sr, sc, er, ec)
    pixels = set()
    for r, c in pts_tl:
        for dr in range(Wt):
            for dc in range(Wt):
                pixels.add((r + dr, c + dc))
    return list(pixels)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    V = rng.randint(*NODE_COUNT_RANGE)
    n_cycles = rng.choice(CYCLE_COUNT_CHOICES)
    path_len = rng.choice(PATH_LEN_CHOICES)

    gap = rng.choice(GAP_CHOICES)
    cell_size = NODE_SIZE + gap

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

    pool = list(COLOR_POOL)
    rng.shuffle(pool)
    bg, node_color, marker_color, track_color = pool[:4]

    canvas = np.full((S, S), bg, dtype=int)
    node_mask = np.zeros((S, S), dtype=bool)
    slack_r = S - Kr * cell_size - 2
    slack_c = S - Kc * cell_size - 2
    origin = (
        1 + rng.randint(0, max(0, slack_r)),
        1 + rng.randint(0, max(0, slack_c)),
    )
    id_to_pos = {}
    for i in range(V):
        cell = id_to_cell[i]
        pos = _cell_to_square_pos(cell, cell_size, origin)
        id_to_pos[i] = pos
        r0, c0 = pos
        canvas[r0 : r0 + NODE_SIZE, c0 : c0 + NODE_SIZE] = node_color
        node_mask[r0 : r0 + NODE_SIZE, c0 : c0 + NODE_SIZE] = True

    # Mark the two terminals by replacing the center pixel.
    for term_id in (0, V - 1):
        r0, c0 = id_to_pos[term_id]
        canvas[r0 + 1, c0 + 1] = marker_color

    track_mask = np.zeros((S, S), dtype=bool)
    edge_track_pixels = {}
    for (a, b) in edges:
        pixels = _track_between_cells(
            id_to_cell[a],
            id_to_cell[b],
            id_to_pos[a],
            id_to_pos[b],
        )
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

    try:
        det_base, det_marker, det_nodes = _detect_marked_nodes(canvas, bg=bg)
    except Exception:
        return None
    if det_base != int(node_color) or det_marker != int(marker_color):
        return None
    if len(det_nodes) != V:
        return None

    expected = canvas.copy()
    for gnode in path_nodes:
        if gnode in (0, V - 1):
            continue
        r0, c0 = id_to_pos[gnode]
        expected[r0 : r0 + NODE_SIZE, c0 : c0 + NODE_SIZE] = int(Color.GREEN)
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
