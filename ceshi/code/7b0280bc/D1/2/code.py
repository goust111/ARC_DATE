# Variant of 7b0280bc
# Dimension: D1 — 路径选择策略 (path)
# Change: BFS最短路径（边数最少），且最短路径唯一... → 避障路径

from __future__ import annotations

from collections import Counter, defaultdict, deque
from itertools import combinations
import random

import numpy as np

from arc_utils import Color, colors_present, components_mask, get_bg, neighbors8


"""
design_analysis:
  source_task_note:
    requested_source_task: cb2d8a2c
    status: solver file not present in the allowed local files; only the JSON task data exists
    fallback: infer the relevant "blocked path / detour" behavior from the source-task IO pattern and
      implement the obstacle idea conservatively inside the original 7b0280bc visual grammar

  a_concretization_and_trace:
    original_visual:
      - solid NxN terminal squares in one color
      - solid NxN station squares in another color
      - thin track components in one track color
      - output colors the chosen stations green and chosen tracks grey
    variant_visual:
      - keep the same squares and routing style
      - add one extra non-node color used only on fully blocked track components
      - passable track components still use the original track color
      - blocked track components remain unchanged in the output; solver must ignore them during BFS
    concrete_small_example:
      - suppose terminals are connected by a 2-edge shortcut A-B-C and a 4-edge detour A-D-E-F-C
      - in the original task all edges are passable, so BFS chooses A-B-C
      - in this variant one or two edges on that shortcut are drawn in the blocked color
      - solver must first classify those colored components as impassable, then BFS over only the
        remaining edges, producing A-D-E-F-C
      - visible difference: the input still shows the tempting shortcut, but the output highlights
        the longer detour instead of the visually shorter route

  b_complexity_calibration:
    - use exactly 5 colors total: bg, terminal, station, passable_track, blocked_track
    - keep node sizes/counts close to the original task so the only new concept is blocked edges
    - keep blocked edge count small (1-2) but guarantee they matter: the blocked edges are sampled from
      the original unique shortest path, and once removed the new passable shortest path becomes longer
    - keep the passable shortest path unique to avoid ambiguity in the output

  c_hidden_assumption_review_for_main:
    - original code assumed every non-node track component was traversable; that is no longer valid
    - original code inferred only one track color; variant now needs two non-node colors
    - original BFS traversed all detected edges; variant BFS must skip blocked edges
    - original recoloring loop could recolor every chosen edge component; this remains safe because the
      generator guarantees that chosen edges are always passable-only components
    - new implicit assumption: the larger of the two non-node colors is the passable track color
      generator guarantee: there are 7-10 passable edges but only 1-2 blocked edges, and total
      passable-track pixels are enforced to exceed blocked-track pixels

  d_placement_legality:
    - reuse the original graph-first embedding and track routing so squares stay non-overlapping,
      in-bounds, and 8-adjacent only to their incident track components
    - blocked edges are routed exactly like normal edges, only recolored differently
    - generation rejects layouts whose routed pixels overlap existing nodes/tracks or whose oracle
      output disagrees with the solver

  e_visual_observability:
    - each sample visibly contains both normal tracks and blocked tracks
    - the blocked tracks are not cosmetic: they create a shorter forbidden shortcut, so the output
      highlights a longer detour; this is inferable from only a few examples
    - the rule does not rely on numeric color ordering, only on repeated structural role

  f_visual_cleanliness:
    - no extra symbols are placed on top of tracks; the blocked track is simply a full component in a
      separate color, keeping the graph readable
    - node shapes, track widths, and spacing remain the same as the source task

  g_color_change_inference:
    - green and grey keep their original output meanings
    - the new blocked-track color never turns grey, which helps the solver infer that those edges are
      excluded rather than merely "another kind of chosen track"

  generator_guarantees_used_by_main:
    - exactly 2 node colors (one occurs on 2 squares = terminals, the other on the remaining squares)
    - exactly 2 edge colors besides the node colors
    - blocked edges are entire connected components of the blocked color
    - BFS on passable edges yields a unique terminal-to-terminal shortest path
    - BFS on passable+blocked edges yields a strictly shorter distance than the passable-only graph
"""


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


def _bfs_with_counts(src: int, allowed_edges: set[tuple[int, int]], v_count: int):
    adj = {i: [] for i in range(v_count)}
    for a, b in allowed_edges:
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))
    dist = {int(src): 0}
    cnt = {int(src): 1}
    q = deque([int(src)])
    while q:
        u = int(q.popleft())
        for v in adj[int(u)]:
            if int(v) not in dist:
                dist[int(v)] = int(dist[int(u)] + 1)
                cnt[int(v)] = int(cnt[int(u)])
                q.append(int(v))
            elif int(dist[int(v)]) == int(dist[int(u)] + 1):
                cnt[int(v)] = int(cnt[int(v)] + cnt[int(u)])
    return dist, cnt, adj


def _trace_path(start: int, goal: int, adj: dict[int, list[int]]) -> list[int] | None:
    q = deque([int(start)])
    prev: dict[int, int | None] = {int(start): None}
    while q:
        u = int(q.popleft())
        if u == int(goal):
            break
        for v in adj.get(int(u), []):
            if int(v) in prev:
                continue
            prev[int(v)] = int(u)
            q.append(int(v))
    if int(goal) not in prev:
        return None
    out: list[int] = []
    cur: int | None = int(goal)
    while cur is not None:
        out.append(int(cur))
        parent = prev.get(int(cur))
        cur = int(parent) if parent is not None else None
    out.reverse()
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the shortest terminal-to-terminal path while treating one track color as blocked.

    Input structure:
    - bg = most common color
    - one square color appears on exactly 2 solid NxN squares -> terminals
    - another square color appears on the remaining solid NxN squares -> stations
    - two non-node colors are used for edges:
      the more prevalent one is passable track, the other marks blocked track components

    Rule:
    - Build graph edges from connected track components of either edge color
    - Ignore components containing the blocked color
    - Run BFS over only passable edges
    - Recolor path stations green and traversed passable track components grey
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    square_cands = find_solid_squares(g, bg=bg, min_size=2, connectivity=8)
    assert square_cands

    size_counts = Counter(int(d["size"]) for d in square_cands)
    n_size = int(max(size_counts.items(), key=lambda kv: (int(kv[1]), int(kv[0])))[0])
    nodes_raw = [d for d in square_cands if int(d["size"]) == int(n_size)]

    nodes: list[dict] = []
    for i, d in enumerate(nodes_raw):
        pts = {(int(r), int(c)) for r, c in d["coords"]}
        nodes.append({"id": int(i), "color": int(d["color"]), "pts": pts})

    nodes_by_color: dict[int, list[int]] = defaultdict(list)
    for nd in nodes:
        nodes_by_color[int(nd["color"])].append(int(nd["id"]))

    node_colors = sorted(nodes_by_color.keys())
    assert len(node_colors) == 2

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
    edge_colors = [c for c in non_bg_colors if int(c) not in (int(terminal_color), int(station_color))]
    assert len(edge_colors) == 2
    edge_colors = sorted(edge_colors, key=lambda c: int((g == int(c)).sum()), reverse=True)
    track_color = int(edge_colors[0])
    blocked_color = int(edge_colors[1])

    pix2node: dict[tuple[int, int], int] = {}
    id2pts: dict[int, set[tuple[int, int]]] = {}
    for nd in nodes:
        nid = int(nd["id"])
        pts = set(nd["pts"])
        id2pts[nid] = pts
        for p in pts:
            pix2node[p] = nid

    edge_mask = (g == int(track_color)) | (g == int(blocked_color))
    track_comps = components_mask(edge_mask, connectivity=8)
    edge_comp_pts: dict[int, list[tuple[int, int]]] = {}
    edge_blocked: dict[int, bool] = {}
    adj: dict[int, list[tuple[int, int]]] = defaultdict(list)

    for edge_id, coords in enumerate(track_comps, start=1):
        pts = [(int(r), int(c)) for r, c in coords]
        if not pts:
            continue
        touched: set[int] = set()
        has_blocked = False
        for r, c in pts:
            if int(g[int(r), int(c)]) == int(blocked_color):
                has_blocked = True
            for rr, cc in neighbors8(int(r), int(c)):
                nid = pix2node.get((int(rr), int(cc)))
                if nid is not None:
                    touched.add(int(nid))
        if len(touched) != 2:
            continue
        a, b = sorted(touched)
        edge_comp_pts[int(edge_id)] = pts
        edge_blocked[int(edge_id)] = bool(has_blocked)
        if not has_blocked:
            adj[int(a)].append((int(b), int(edge_id)))
            adj[int(b)].append((int(a), int(edge_id)))

    start, goal = int(term_ids[0]), int(term_ids[1])
    q = deque([int(start)])
    prev: dict[int, int | None] = {int(start): None}
    prev_edge: dict[int, int | None] = {int(start): None}

    while q:
        u = int(q.popleft())
        if u == int(goal):
            break
        for v, edge_id in adj.get(int(u), []):
            if int(v) in prev:
                continue
            prev[int(v)] = int(u)
            prev_edge[int(v)] = int(edge_id)
            q.append(int(v))

    assert int(goal) in prev

    node_path: list[int] = []
    edge_path: list[int] = []
    cur: int | None = int(goal)
    while cur is not None:
        node_path.append(int(cur))
        edge_id = prev_edge.get(int(cur))
        if edge_id is not None:
            assert not edge_blocked[int(edge_id)]
            edge_path.append(int(edge_id))
        parent = prev.get(int(cur))
        cur = int(parent) if parent is not None else None
    node_path.reverse()
    edge_path.reverse()

    out = g.copy()
    term_set = set(map(int, term_ids))
    for nid in node_path:
        if int(nid) in term_set:
            continue
        for r, c in id2pts[int(nid)]:
            out[int(r), int(c)] = int(Color.GREEN)

    for edge_id in edge_path:
        for r, c in edge_comp_pts[int(edge_id)]:
            out[int(r), int(c)] = int(Color.GREY)

    return out


NODE_SIZE_CHOICES = [2, 3]
NODE_COUNT_RANGE = (7, 10)
CYCLE_COUNT_CHOICES = [1, 2]
ORIGINAL_PATH_LEN_CHOICES = [2, 3]
BLOCKED_EDGE_COUNT_CHOICES = [1, 2]
GAP_CHOICES = [2, 3]
CELL_GRID_DIMS = (3, 4, 5)
GRID_SIZE_RANGE = (15, 30)
COLOR_POOL = [0, 1, 2, 4, 6, 7, 8, 9]
CELL_PICK_RETRIES = 50
GEN_ATTEMPT_RETRIES = 500


def _compute_adjacency(picked: set[tuple[int, int]], use_diagonal: bool, allow_skip: bool):
    edges = set()
    for i, j in picked:
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            n = (int(i + di), int(j + dj))
            if n in picked:
                edges.add(tuple(sorted(((int(i), int(j)), n))))
        if allow_skip:
            for di, dj in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                mid = (int(i + di // 2), int(j + dj // 2))
                far = (int(i + di), int(j + dj))
                if far in picked and mid not in picked:
                    edges.add(tuple(sorted(((int(i), int(j)), far))))
        if use_diagonal:
            for di, dj in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                n = (int(i + di), int(j + dj))
                corner1 = (int(i + di), int(j))
                corner2 = (int(i), int(j + dj))
                if n in picked and corner1 not in picked and corner2 not in picked:
                    edges.add(tuple(sorted(((int(i), int(j)), n))))
    return edges


def _pick_cells(
    kr: int,
    kc: int,
    v_count: int,
    rng: random.Random,
    use_diagonal: bool,
    allow_skip: bool,
):
    cells = [(i, j) for i in range(int(kr)) for j in range(int(kc))]
    if len(cells) < int(v_count):
        return None
    for _ in range(CELL_PICK_RETRIES):
        rng.shuffle(cells)
        picked = set(cells[: int(v_count)])
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
        if len(seen) == int(v_count):
            return picked
    return None


def _build_passable_graph(
    picked: set[tuple[int, int]],
    cell_to_id: dict[tuple[int, int], int],
    rng: random.Random,
    path_len: int,
    n_cycles: int,
    use_diagonal: bool,
    allow_skip: bool,
):
    v_count = len(picked)
    cell_edges = _compute_adjacency(picked, use_diagonal=use_diagonal, allow_skip=allow_skip)
    full_edges = {tuple(sorted((int(cell_to_id[a]), int(cell_to_id[b])))) for a, b in cell_edges}

    ids = list(range(v_count))
    rng.shuffle(ids)
    t0, t1 = int(ids[0]), int(ids[1])

    dist, cnt, adj = _bfs_with_counts(int(t0), full_edges, v_count)
    if int(t1) not in dist or int(dist[int(t1)]) != int(path_len):
        return None

    sp_nodes = _trace_path(int(t0), int(t1), adj)
    if sp_nodes is None:
        return None
    sp_edges = {
        tuple(sorted((int(sp_nodes[i]), int(sp_nodes[i + 1]))))
        for i in range(len(sp_nodes) - 1)
    }

    tree_edges = set(sp_edges)
    in_tree = set(sp_nodes)
    frontier = list(sp_nodes)
    while in_tree != set(range(v_count)):
        next_frontier = []
        for u in frontier:
            for v in adj[int(u)]:
                if int(v) not in in_tree:
                    tree_edges.add(tuple(sorted((int(u), int(v)))))
                    in_tree.add(int(v))
                    next_frontier.append(int(v))
        if not next_frontier:
            return None
        frontier = next_frontier

    extra_candidates = list(full_edges - tree_edges)
    rng.shuffle(extra_candidates)
    final_edges = set(tree_edges)
    added = 0
    for edge in extra_candidates:
        if int(added) >= int(n_cycles):
            break
        cand = set(final_edges)
        cand.add(tuple(sorted((int(edge[0]), int(edge[1])))))
        dist2, cnt2, _ = _bfs_with_counts(int(t0), cand, v_count)
        if int(dist2.get(int(t1), -1)) == int(path_len) and int(cnt2.get(int(t1), 0)) == 1:
            final_edges = cand
            added += 1
    if int(added) != int(n_cycles):
        return None

    dist3, cnt3, _ = _bfs_with_counts(int(t0), final_edges, v_count)
    if int(dist3.get(int(t1), -1)) != int(path_len) or int(cnt3.get(int(t1), 0)) != 1:
        return None
    if len(final_edges) != int(v_count - 1 + n_cycles):
        return None

    id_map = {int(t0): 0, int(t1): int(v_count - 1)}
    next_id = 1
    for old in range(v_count):
        if old in (int(t0), int(t1)):
            continue
        id_map[int(old)] = int(next_id)
        next_id += 1

    remapped_edges = {tuple(sorted((int(id_map[a]), int(id_map[b])))) for a, b in final_edges}
    remapped_cell_to_id = {cell: int(id_map[cell_to_id[cell]]) for cell in cell_to_id}
    new_path_nodes = [int(id_map[n]) for n in sp_nodes]
    new_path_edges = [
        tuple(sorted((int(new_path_nodes[i]), int(new_path_nodes[i + 1]))))
        for i in range(len(new_path_nodes) - 1)
    ]
    return list(remapped_edges), new_path_nodes, new_path_edges, remapped_cell_to_id


def _select_blocked_edges(
    full_edges: set[tuple[int, int]],
    original_path_edges: list[tuple[int, int]],
    v_count: int,
    original_path_len: int,
    rng: random.Random,
):
    candidates = [tuple(sorted((int(a), int(b)))) for a, b in original_path_edges]
    if not candidates:
        return None
    rng.shuffle(candidates)
    order = list(BLOCKED_EDGE_COUNT_CHOICES)
    rng.shuffle(order)

    for k in order:
        if len(candidates) < int(k):
            continue
        combos = list(combinations(candidates, int(k)))
        rng.shuffle(combos)
        for combo in combos:
            remaining = set(full_edges)
            remaining.difference_update(tuple(sorted((int(a), int(b)))) for a, b in combo)
            dist, cnt, adj = _bfs_with_counts(0, remaining, v_count)
            goal = int(v_count - 1)
            if int(goal) not in dist:
                continue
            if int(dist[int(goal)]) <= int(original_path_len):
                continue
            if int(cnt.get(int(goal), 0)) != 1:
                continue
            node_path = _trace_path(0, int(goal), adj)
            if node_path is None:
                continue
            edge_path = [
                tuple(sorted((int(node_path[i]), int(node_path[i + 1]))))
                for i in range(len(node_path) - 1)
            ]
            return [tuple(sorted((int(a), int(b)))) for a, b in combo], node_path, edge_path
    return None


def _cell_to_square_pos(cell: tuple[int, int], n_size: int, cell_size: int, origin: tuple[int, int]):
    i, j = cell
    or_r, or_c = origin
    margin = (int(cell_size) - int(n_size)) // 2
    return (int(or_r + i * cell_size + margin), int(or_c + j * cell_size + margin))


def _dock_tl(sq_r: int, sq_c: int, n_size: int, track_w: int, si: int, sj: int):
    if si == -1:
        return int(sq_r - track_w), int(sq_c)
    if si == 1:
        return int(sq_r + n_size), int(sq_c + n_size - track_w)
    if sj == -1:
        return int(sq_r + n_size - track_w), int(sq_c - track_w)
    if sj == 1:
        return int(sq_r), int(sq_c + n_size)
    raise ValueError("zero direction is invalid for track docking")


def _walk_footprints(sr: int, sc: int, er: int, ec: int):
    pts = [(int(sr), int(sc))]
    cr, cc = int(sr), int(sc)
    while cr != int(er) and cc != int(ec):
        cr += 1 if int(er) > int(cr) else -1
        cc += 1 if int(ec) > int(cc) else -1
        pts.append((int(cr), int(cc)))
    while cr != int(er):
        cr += 1 if int(er) > int(cr) else -1
        pts.append((int(cr), int(cc)))
    while cc != int(ec):
        cc += 1 if int(ec) > int(cc) else -1
        pts.append((int(cr), int(cc)))
    return pts


def _track_between_cells(
    cell_a: tuple[int, int],
    cell_b: tuple[int, int],
    pos_a: tuple[int, int],
    pos_b: tuple[int, int],
    n_size: int,
) -> list[tuple[int, int]]:
    track_w = int(n_size - 1)
    ra, ca = pos_a
    rb, cb = pos_b
    dci, dcj = int(cell_b[0] - cell_a[0]), int(cell_b[1] - cell_a[1])
    si = 0 if dci == 0 else (1 if dci > 0 else -1)
    sj = 0 if dcj == 0 else (1 if dcj > 0 else -1)

    sr, sc = _dock_tl(int(ra), int(ca), int(n_size), int(track_w), int(si), int(sj))
    er, ec = _dock_tl(int(rb), int(cb), int(n_size), int(track_w), int(-si), int(-sj))
    pts_tl = _walk_footprints(int(sr), int(sc), int(er), int(ec))

    pixels = set()
    for r, c in pts_tl:
        for dr in range(track_w):
            for dc in range(track_w):
                pixels.add((int(r + dr), int(c + dc)))
    return list(pixels)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_size = int(rng.choice(NODE_SIZE_CHOICES))
    v_count = int(rng.randint(*NODE_COUNT_RANGE))
    n_cycles = int(rng.choice(CYCLE_COUNT_CHOICES))
    original_path_len = int(rng.choice(ORIGINAL_PATH_LEN_CHOICES))

    gap = int(rng.choice(GAP_CHOICES))
    cell_size = int(n_size + gap)

    dims = []
    for kr in CELL_GRID_DIMS:
        for kc in CELL_GRID_DIMS:
            if int(kr * kc) < int(v_count):
                continue
            min_side = int(max(kr, kc) * cell_size + 2)
            if int(min_side) > int(GRID_SIZE_RANGE[1]):
                continue
            dims.append((int(kr), int(kc), int(min_side)))
    if not dims:
        return None
    kr, kc, min_side = rng.choice(dims)
    side = int(rng.randint(max(int(GRID_SIZE_RANGE[0]), int(min_side)), int(GRID_SIZE_RANGE[1])))

    use_diagonal = False
    allow_skip = True
    picked = _pick_cells(
        int(kr),
        int(kc),
        int(v_count),
        rng,
        use_diagonal=use_diagonal,
        allow_skip=allow_skip,
    )
    if picked is None:
        return None

    picked_list = list(picked)
    rng.shuffle(picked_list)
    cell_to_id = {cell: int(i) for i, cell in enumerate(picked_list)}

    built = _build_passable_graph(
        picked,
        cell_to_id,
        rng,
        original_path_len,
        n_cycles,
        use_diagonal=use_diagonal,
        allow_skip=allow_skip,
    )
    if built is None:
        return None
    full_edges_list, original_path_nodes, original_path_edges, cell_to_id = built
    id_to_cell = {int(v): k for k, v in cell_to_id.items()}
    full_edges = {tuple(sorted((int(a), int(b)))) for a, b in full_edges_list}

    blocked_result = _select_blocked_edges(
        full_edges,
        original_path_edges,
        v_count,
        original_path_len,
        rng,
    )
    if blocked_result is None:
        return None
    blocked_edges, path_nodes, path_edges = blocked_result
    blocked_edges = [tuple(sorted((int(a), int(b)))) for a, b in blocked_edges]
    passable_edges = set(full_edges)
    passable_edges.difference_update(blocked_edges)

    pool = list(COLOR_POOL)
    rng.shuffle(pool)
    five = pool[:5]
    bg = int(rng.choice(five))
    rest = [int(c) for c in five if int(c) != int(bg)]
    terminal_color = int(rng.choice(rest))
    rest = [int(c) for c in rest if int(c) != int(terminal_color)]
    station_color = int(rng.choice(rest))
    rest = [int(c) for c in rest if int(c) != int(station_color)]
    track_color, blocked_color = map(int, rest[:2])

    canvas = np.full((side, side), int(bg), dtype=int)
    node_mask = np.zeros((side, side), dtype=bool)
    track_mask = np.zeros((side, side), dtype=bool)

    slack_r = int(side - kr * cell_size - 2)
    slack_c = int(side - kc * cell_size - 2)
    origin = (
        int(1 + rng.randint(0, max(0, slack_r))),
        int(1 + rng.randint(0, max(0, slack_c))),
    )
    id_to_pos: dict[int, tuple[int, int]] = {}
    for node_id in range(v_count):
        cell = id_to_cell[int(node_id)]
        pos = _cell_to_square_pos(cell, int(n_size), int(cell_size), origin)
        id_to_pos[int(node_id)] = pos
        color = int(terminal_color) if int(node_id) in (0, int(v_count - 1)) else int(station_color)
        r0, c0 = pos
        canvas[r0 : r0 + n_size, c0 : c0 + n_size] = int(color)
        node_mask[r0 : r0 + n_size, c0 : c0 + n_size] = True

    edge_track_pixels: dict[tuple[int, int], list[tuple[int, int]]] = {}
    blocked_total = 0
    passable_total = 0
    all_edges = [(edge, int(track_color)) for edge in sorted(passable_edges)]
    all_edges += [(edge, int(blocked_color)) for edge in sorted(blocked_edges)]

    for edge, color in all_edges:
        a, b = edge
        pixels = _track_between_cells(
            id_to_cell[int(a)],
            id_to_cell[int(b)],
            id_to_pos[int(a)],
            id_to_pos[int(b)],
            int(n_size),
        )
        for r, c in pixels:
            if not (0 <= int(r) < int(side) and 0 <= int(c) < int(side)):
                return None
            if bool(node_mask[int(r), int(c)]) or bool(track_mask[int(r), int(c)]):
                return None
        for r, c in pixels:
            canvas[int(r), int(c)] = int(color)
            track_mask[int(r), int(c)] = True
        edge_track_pixels[tuple(sorted((int(a), int(b))))] = pixels
        if int(color) == int(track_color):
            passable_total += len(pixels)
        else:
            blocked_total += len(pixels)

    if int(passable_total) <= int(blocked_total):
        return None
    if int(np.argmax(np.bincount(canvas.ravel()))) != int(bg):
        return None

    expected = canvas.copy()
    for node_id in path_nodes:
        if int(node_id) in (0, int(v_count - 1)):
            continue
        r0, c0 = id_to_pos[int(node_id)]
        expected[r0 : r0 + n_size, c0 : c0 + n_size] = int(Color.GREEN)
    for edge in path_edges:
        for r, c in edge_track_pixels[tuple(sorted((int(edge[0]), int(edge[1]))))]:
            expected[int(r), int(c)] = int(Color.GREY)

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
