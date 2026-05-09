# Variant of 7b0280bc
# Dimension: D16 — 障碍物/轨道交互 (obstacle)
# Change: 无显式障碍物，所有轨道段均可通行... → 断裂轨道段

from __future__ import annotations

from collections import Counter, defaultdict, deque
from itertools import combinations
from math import gcd
import random

import numpy as np

from arc_utils import Color, colors_present, components_mask, get_bg


# ---------------------------------------------------------------------------
# Design analysis for the D16 variant
# ---------------------------------------------------------------------------
#
# spec_variant:
#   source_task: none
#   base_task:
#     nodes: two terminal NxN solid squares + 5-8 station NxN solid squares
#     rails: horizontal/vertical strips between aligned node cells
#     output: shortest valid route recolored (stations -> green, rails -> grey)
#   d16_change:
#     visible_change:
#       some non-solution rail candidates are still mostly drawn, but one internal
#       cross-section is erased:
#         - to keep D16 non-degenerate, generation fixes N=3 so rails are 2-pixel
#           thick and can stay 8-connected after a tiny erasure
#         - each broken decoy rail is missing 1-2 interior pixels, leaving a small
#           notch/hole while the color component still spans both endpoint squares
#     semantic_change:
#       only fully filled rails are graph edges.
#       A rail with a tiny internal hole is a visual decoy and must be ignored
#       even if its remaining pixels still form one 8-connected component.
#   concrete_flow_trace:
#     original_main:
#       1. collect all track-color connected components
#       2. if a component touches exactly two nodes, treat it as a usable edge
#       3. BFS over those edges
#     variant_main:
#       1. recover the square nodes exactly as before
#       2. infer the node lattice from square positions
#       3. enumerate only the rule-allowed straight rail candidates between aligned
#          neighboring/skip-one node cells
#       4. for each candidate, reconstruct its full expected footprint and check
#          whether every footprint pixel is still track-colored
#       5. only fully intact candidates become edges; candidates with a 1-2 pixel
#          internal gap are excluded from BFS
#     small_example:
#       if terminals are linked by a 3-edge valid route, and there is another
#       horizontal decoy rail that would give a 2-edge shortcut if intact, the
#       original component-based logic would be unsafe once partial rails exist.
#       The new logic explicitly rejects that shortcut because one middle column or
#       row is missing.
#   complexity_calibration:
#     - Only one new degree of freedom is added: valid vs broken rail continuity.
#     - Broken rails reuse the same track color and same geometry as valid rails,
#       so the visual difference is local and observable instead of introducing
#       extra colors or symbols.
#     - Fixing N=3 is a necessary supporting adjustment: with N=2, deleting one
#       pixel would simply split the rail component and the original solver logic
#       would continue to work, so the D16 change would not actually matter.
#     - At least one broken rail per sample is "critical": if it were intact, the
#       terminal shortest path would become shorter or non-unique.
#   original_main_assumptions_audited:
#     - track connected component == graph edge: no longer valid, replaced
#     - every track-colored pixel belongs to a usable edge: no longer valid
#     - edge validity can be decided without reconstructing intended geometry:
#       no longer valid, replaced by full-footprint checking
#     - BFS / recolor logic itself remains valid once usable edges are rebuilt
#   placement_legality:
#     - valid rails and broken rails are both routed with the same deterministic
#       straight-strip geometry used by the base task
#     - a broken rail is added only if its full footprint is in-bounds and does
#       not overlap nodes or other rails
#     - the erased cross-section is chosen strictly inside the strip so the two
#       fragments stay separated and each fragment remains attached to its own node
#   visual_observability:
#     - examples always contain at least one near-complete rail with a tiny hole
#       in the middle, so humans can see that not every apparent connection works
#     - the output never recolors those broken rails, reinforcing that continuity,
#       not mere alignment, decides passability
#   cleanliness:
#     - no extra overlay colors are introduced
#     - broken rails are non-overlapping straight strips, so the scene remains
#       readable despite the added decoys
#   new_solver_assumptions:
#     - node squares still lie on a common row/column lattice; generator enforces it
#     - usable rail candidates are only horizontal/vertical neighbor or skip-one
#       pairs on that lattice; generator enforces the same rule
#     - each intact rail footprint is fully track-colored; generator guarantees it
#     - each broken rail misses exactly one internal cross-section; generator builds
#       it that way so "all pixels present" is a correct validity test


def find_solid_squares(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    min_size: int = 2,
    connectivity: int = 8,
) -> list[dict]:
    """Find solid monochrome NxN squares used as graph nodes."""
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


def _dock_tl(sq_r: int, sq_c: int, N: int, Wt: int, si: int, sj: int):
    """Top-left of the Wt x Wt rail footprint docked to one square face."""
    if si == -1:
        return sq_r - Wt, sq_c
    if si == 1:
        return sq_r + N, sq_c + N - Wt
    if sj == -1:
        return sq_r + N - Wt, sq_c - Wt
    if sj == 1:
        return sq_r, sq_c + N
    raise ValueError("_dock_tl: zero direction")


def _walk_footprints(sr: int, sc: int, er: int, ec: int):
    """Generate the sequence of footprint top-left positions from start to end."""
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


def _track_between_positions(
    cell_a: tuple[int, int],
    cell_b: tuple[int, int],
    pos_a: tuple[int, int],
    pos_b: tuple[int, int],
    N: int,
) -> list[tuple[int, int]]:
    """Reconstruct the full straight-strip rail footprint between two aligned nodes."""
    Wt = N - 1
    ra, ca = pos_a
    rb, cb = pos_b
    dci, dcj = cell_b[0] - cell_a[0], cell_b[1] - cell_a[1]
    si = 0 if dci == 0 else (1 if dci > 0 else -1)
    sj = 0 if dcj == 0 else (1 if dcj > 0 else -1)

    sr, sc = _dock_tl(ra, ca, N, Wt, si, sj)
    er, ec = _dock_tl(rb, cb, N, Wt, -si, -sj)

    pixels = set()
    for r, c in _walk_footprints(sr, sc, er, ec):
        for dr in range(Wt):
            for dc in range(Wt):
                pixels.add((r + dr, c + dc))
    return sorted(pixels)


def _infer_lattice_step(nodes: list[dict]) -> int:
    """Infer the common lattice step from node square top-left coordinates."""
    diffs: list[int] = []
    row_vals = sorted({int(nd["bbox"][0]) for nd in nodes})
    col_vals = sorted({int(nd["bbox"][1]) for nd in nodes})
    for vals in (row_vals, col_vals):
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                diffs.append(int(vals[j] - vals[i]))
    assert diffs
    step = 0
    for d in diffs:
        step = int(d) if step == 0 else gcd(step, int(d))
    assert step > 0
    return int(step)


def _analyze_grid(grid: np.ndarray) -> dict:
    """Shared parser for solver logic and variant diagnostics."""
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    square_cands = find_solid_squares(g, bg=bg, min_size=2, connectivity=8)
    assert square_cands

    size_counts = Counter(int(d["size"]) for d in square_cands)
    N = int(max(size_counts.items(), key=lambda kv: (int(kv[1]), int(kv[0])))[0])
    nodes_raw = [d for d in square_cands if int(d["size"]) == int(N)]

    nodes: list[dict] = []
    for i, d in enumerate(nodes_raw):
        r0, c0, r1, c1 = d["bbox"]
        pts = {(int(r), int(c)) for r, c in d["coords"]}
        nodes.append(
            {
                "id": int(i),
                "color": int(d["color"]),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "top_left": (int(r0), int(c0)),
                "pts": pts,
            }
        )

    nodes_by_color: dict[int, list[int]] = defaultdict(list)
    for nd in nodes:
        nodes_by_color[int(nd["color"])].append(int(nd["id"]))

    terminal_color = None
    station_color = None
    for col in sorted(nodes_by_color):
        ids = nodes_by_color[int(col)]
        if len(ids) == 2:
            terminal_color = int(col)
        else:
            station_color = int(col)
    assert terminal_color is not None and station_color is not None

    non_bg_colors = colors_present(g, exclude={bg})
    track_candidates = [
        c for c in non_bg_colors if int(c) not in (int(terminal_color), int(station_color))
    ]
    assert track_candidates
    track_color = int(max(track_candidates, key=lambda c: int((g == int(c)).sum())))

    step = _infer_lattice_step(nodes)
    min_r = min(int(nd["top_left"][0]) for nd in nodes)
    min_c = min(int(nd["top_left"][1]) for nd in nodes)

    logical_to_id: dict[tuple[int, int], int] = {}
    id_to_pos: dict[int, tuple[int, int]] = {}
    id_to_cell: dict[int, tuple[int, int]] = {}
    id_to_pts: dict[int, set[tuple[int, int]]] = {}
    for nd in nodes:
        nid = int(nd["id"])
        r0, c0 = nd["top_left"]
        lr = (int(r0) - int(min_r)) // int(step)
        lc = (int(c0) - int(min_c)) // int(step)
        assert int(min_r) + int(lr) * int(step) == int(r0)
        assert int(min_c) + int(lc) * int(step) == int(c0)
        logical_to_id[(int(lr), int(lc))] = nid
        id_to_pos[nid] = (int(r0), int(c0))
        id_to_cell[nid] = (int(lr), int(lc))
        id_to_pts[nid] = set(nd["pts"])

    candidate_edges: list[dict] = []
    for (lr, lc), a in logical_to_id.items():
        for dr, dc in ((0, 1), (1, 0)):
            for dist in (1, 2):
                cell_b = (int(lr + dr * dist), int(lc + dc * dist))
                b = logical_to_id.get(cell_b)
                if b is None:
                    continue
                if dist == 2:
                    mid = (int(lr + dr), int(lc + dc))
                    if mid in logical_to_id:
                        continue
                pixels = _track_between_positions(
                    (int(lr), int(lc)),
                    cell_b,
                    id_to_pos[int(a)],
                    id_to_pos[int(b)],
                    int(N),
                )
                if not pixels:
                    continue
                intact = all(int(g[r, c]) == int(track_color) for r, c in pixels)
                candidate_edges.append(
                    {
                        "nodes": (int(a), int(b)),
                        "cells": ((int(lr), int(lc)), cell_b),
                        "pixels": pixels,
                        "intact": bool(intact),
                    }
                )

    adj: dict[int, list[tuple[int, tuple[int, int]]]] = defaultdict(list)
    intact_pixels: dict[tuple[int, int], list[tuple[int, int]]] = {}
    broken_edges: list[dict] = []
    for item in candidate_edges:
        edge = tuple(sorted(item["nodes"]))
        if item["intact"]:
            intact_pixels[edge] = list(item["pixels"])
            a, b = edge
            adj[int(a)].append((int(b), edge))
            adj[int(b)].append((int(a), edge))
        else:
            broken_edges.append(item)

    start, goal = map(int, nodes_by_color[int(terminal_color)])
    q = deque([start])
    prev: dict[int, int | None] = {start: None}
    prev_edge: dict[int, tuple[int, int] | None] = {start: None}

    while q:
        u = int(q.popleft())
        if u == goal:
            break
        for v, edge in adj.get(u, []):
            if int(v) in prev:
                continue
            prev[int(v)] = int(u)
            prev_edge[int(v)] = edge
            q.append(int(v))

    assert goal in prev

    node_path: list[int] = []
    edge_path: list[tuple[int, int]] = []
    cur = int(goal)
    while cur is not None:
        node_path.append(int(cur))
        edge = prev_edge.get(int(cur))
        if edge is not None:
            edge_path.append(tuple(sorted(edge)))
        parent = prev.get(int(cur))
        cur = int(parent) if parent is not None else None
    node_path.reverse()
    edge_path.reverse()

    return {
        "grid": g,
        "bg": int(bg),
        "N": int(N),
        "track_color": int(track_color),
        "terminal_color": int(terminal_color),
        "station_color": int(station_color),
        "nodes": nodes,
        "id_to_pts": id_to_pts,
        "intact_pixels": intact_pixels,
        "broken_edges": broken_edges,
        "node_path": node_path,
        "edge_path": edge_path,
        "term_ids": tuple(sorted(map(int, nodes_by_color[int(terminal_color)]))),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor the shortest path while treating broken rails as unusable."""
    info = _analyze_grid(grid)
    out = np.array(info["grid"], dtype=int, copy=True)

    term_set = set(info["term_ids"])
    for nid in info["node_path"]:
        if int(nid) in term_set:
            continue
        for r, c in info["id_to_pts"][int(nid)]:
            out[int(r), int(c)] = int(Color.GREEN)

    for edge in info["edge_path"]:
        for r, c in info["intact_pixels"][tuple(sorted(edge))]:
            out[int(r), int(c)] = int(Color.GREY)

    return out


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

NODE_SIZE_CHOICES = [3]
NODE_COUNT_RANGE = (7, 10)
CYCLE_COUNT_CHOICES = [1, 2]
PATH_LEN_CHOICES = [3]
GAP_CHOICES = [2, 3]
CELL_GRID_DIMS = (3, 4, 5)
GRID_SIZE_RANGE = (15, 30)
COLOR_POOL = [0, 1, 2, 4, 6, 7, 8, 9]
CELL_PICK_RETRIES = 50
GEN_ATTEMPT_RETRIES = 200
BROKEN_TRACK_COUNT_CHOICES = [1]

TEMPLATE_LIBRARY = [
    {
        "cells": {
            "T0": (1, 0),
            "T1": (1, 2),
            "A": (0, 0),
            "B": (0, 2),
            "C": (2, 0),
            "D": (2, 1),
            "E": (2, 2),
        },
        "valid_edges": [
            ("T0", "A"),
            ("A", "B"),
            ("B", "T1"),
            ("T0", "C"),
            ("C", "D"),
            ("D", "E"),
            ("E", "T1"),
        ],
        "path_nodes": ["T0", "A", "B", "T1"],
        "broken_edges": [("T0", "T1")],
    },
    {
        "cells": {
            "T0": (1, 0),
            "T1": (1, 2),
            "A": (0, 0),
            "B": (0, 2),
            "C": (2, 0),
            "D": (2, 1),
            "E": (2, 2),
            "H": (3, 2),
        },
        "valid_edges": [
            ("T0", "A"),
            ("A", "B"),
            ("B", "T1"),
            ("T0", "C"),
            ("C", "D"),
            ("D", "E"),
            ("E", "T1"),
            ("E", "H"),
        ],
        "path_nodes": ["T0", "A", "B", "T1"],
        "broken_edges": [("T0", "T1")],
    },
    {
        "cells": {
            "T0": (1, 0),
            "T1": (1, 2),
            "A": (0, 0),
            "B": (0, 2),
            "C": (2, 0),
            "D": (2, 1),
            "E": (2, 2),
            "F": (0, 3),
            "G": (1, 3),
        },
        "valid_edges": [
            ("T0", "A"),
            ("A", "B"),
            ("B", "T1"),
            ("T0", "C"),
            ("C", "D"),
            ("D", "E"),
            ("E", "T1"),
            ("B", "F"),
            ("F", "G"),
            ("G", "T1"),
        ],
        "path_nodes": ["T0", "A", "B", "T1"],
        "broken_edges": [("T0", "T1")],
    },
    {
        "cells": {
            "T0": (1, 0),
            "T1": (1, 2),
            "A": (0, 0),
            "B": (0, 2),
            "C": (2, 0),
            "D": (2, 1),
            "E": (2, 2),
            "F": (0, 3),
            "G": (1, 3),
            "H": (3, 2),
        },
        "valid_edges": [
            ("T0", "A"),
            ("A", "B"),
            ("B", "T1"),
            ("T0", "C"),
            ("C", "D"),
            ("D", "E"),
            ("E", "T1"),
            ("B", "F"),
            ("F", "G"),
            ("G", "T1"),
            ("E", "H"),
        ],
        "path_nodes": ["T0", "A", "B", "T1"],
        "broken_edges": [("T0", "T1")],
    },
]


def _compute_adjacency(picked: set[tuple[int, int]], use_diagonal: bool, allow_skip: bool):
    """Compute rule-allowed undirected adjacencies on the cell lattice."""
    edges = set()
    for i, j in picked:
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            n = (i + di, j + dj)
            if n in picked:
                edges.add(tuple(sorted(((i, j), n))))
        if allow_skip:
            for di, dj in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                mid = (i + di // 2, j + dj // 2)
                far = (i + di, j + dj)
                if far in picked and mid not in picked:
                    edges.add(tuple(sorted(((i, j), far))))
        if use_diagonal:
            for di, dj in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                n = (i + di, j + dj)
                corner1 = (i + di, j)
                corner2 = (i, j + dj)
                if n in picked and corner1 not in picked and corner2 not in picked:
                    edges.add(tuple(sorted(((i, j), n))))
    return edges


def _pick_cells(
    Kr: int,
    Kc: int,
    V: int,
    rng: random.Random,
    use_diagonal: bool,
    allow_skip: bool,
):
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
    """BFS from src; also count how many shortest paths reach each vertex."""
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
    """Build the valid rail graph with a unique terminal shortest path."""
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
    remapped_edges = {tuple(sorted((id_map[a], id_map[b]))) for a, b in final_edges}
    remapped_cell_to_id = {cell: id_map[cell_to_id[cell]] for cell in cell_to_id}
    new_path_nodes = [id_map[n] for n in sp_nodes]
    new_path_edges = [
        tuple(sorted((new_path_nodes[i], new_path_nodes[i + 1])))
        for i in range(len(new_path_nodes) - 1)
    ]

    return list(remapped_edges), new_path_nodes, new_path_edges, remapped_cell_to_id


def _cell_to_square_pos(cell: tuple[int, int], N: int, cell_size: int, origin: tuple[int, int]):
    """Top-left of the NxN square centered in its cell."""
    i, j = cell
    or_r, or_c = origin
    margin = (cell_size - N) // 2
    return (or_r + i * cell_size + margin, or_c + j * cell_size + margin)


def _break_track_pixels(
    pixels: list[tuple[int, int]],
    axis: str,
    N: int,
    rng: random.Random,
) -> set[tuple[int, int]] | None:
    """Erase 1-2 middle pixels while keeping one 8-connected decoy component."""
    rows = sorted({r for r, _ in pixels})
    cols = sorted({c for _, c in pixels})
    pts = sorted(set(pixels))

    if axis == "h":
        interior = [p for p in pts if cols[0] < p[1] < cols[-1]]
        center = (sum(c for _, c in pts) / len(pts), sum(r for r, _ in pts) / len(pts))
    else:
        interior = [p for p in pts if rows[0] < p[0] < rows[-1]]
        center = (sum(r for r, _ in pts) / len(pts), sum(c for _, c in pts) / len(pts))
    if not interior:
        return None

    neigh = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    def stays_connected(removed_subset: set[tuple[int, int]]) -> bool:
        remaining = set(pts) - removed_subset
        if not remaining:
            return False
        seed = next(iter(remaining))
        seen = {seed}
        stack = [seed]
        while stack:
            r, c = stack.pop()
            for dr, dc in neigh:
                npix = (r + dr, c + dc)
                if npix in remaining and npix not in seen:
                    seen.add(npix)
                    stack.append(npix)
        return len(seen) == len(remaining)

    best_subset = None
    best_score = None
    max_remove = min(2, len(interior))
    for k in range(1, max_remove + 1):
        for combo in combinations(interior, k):
            removed = set(combo)
            if not stays_connected(removed):
                continue
            avg_r = sum(r for r, _ in combo) / len(combo)
            avg_c = sum(c for _, c in combo) / len(combo)
            if axis == "h":
                score = (abs(avg_c - center[0]), abs(avg_r - center[1]), k, sorted(combo))
            else:
                score = (abs(avg_r - center[0]), abs(avg_c - center[1]), k, sorted(combo))
            if best_score is None or score < best_score:
                best_score = score
                best_subset = removed
        if best_subset is not None:
            break
    return best_subset


def _find_broken_decoys(
    picked: set[tuple[int, int]],
    valid_edges: set[tuple[int, int]],
    path_len: int,
    V: int,
    cell_to_id: dict[tuple[int, int], int],
    id_to_cell: dict[int, tuple[int, int]],
    id_to_pos: dict[int, tuple[int, int]],
    N: int,
):
    """Find candidate rails to draw as broken decoys."""
    full_cell_edges = _compute_adjacency(picked, use_diagonal=False, allow_skip=True)
    all_edges = {tuple(sorted((cell_to_id[a], cell_to_id[b]))) for a, b in full_cell_edges}
    out = []
    for edge in sorted(all_edges - valid_edges):
        dist2, cnt2, _ = _bfs_with_counts(0, set(valid_edges) | {edge}, V)
        goal_dist = dist2.get(V - 1)
        goal_cnt = cnt2.get(V - 1, 0)
        if goal_dist is None:
            continue
        if goal_dist < path_len:
            criticality = 2
        elif goal_dist == path_len and goal_cnt > 1:
            criticality = 1
        else:
            continue

        a, b = edge
        cell_a = id_to_cell[a]
        cell_b = id_to_cell[b]
        axis = "h" if cell_a[0] == cell_b[0] else "v"
        pixels = _track_between_positions(cell_a, cell_b, id_to_pos[a], id_to_pos[b], N)
        out.append(
            {
                "edge": edge,
                "pixels": pixels,
                "axis": axis,
                "criticality": criticality,
            }
        )
    out.sort(key=lambda item: (-int(item["criticality"]), -len(item["pixels"])))
    return out


def _transform_template_cells(
    cells: dict[str, tuple[int, int]],
    mode: int,
) -> dict[str, tuple[int, int]]:
    """Apply a lattice symmetry and normalize coordinates back to the first quadrant."""
    max_r = max(r for r, _ in cells.values())
    max_c = max(c for _, c in cells.values())

    def xf(r: int, c: int) -> tuple[int, int]:
        if mode == 0:
            return r, c
        if mode == 1:
            return max_r - r, c
        if mode == 2:
            return r, max_c - c
        if mode == 3:
            return max_r - r, max_c - c
        if mode == 4:
            return c, r
        if mode == 5:
            return max_c - c, r
        if mode == 6:
            return c, max_r - r
        return max_c - c, max_r - r

    raw = {label: xf(r, c) for label, (r, c) in cells.items()}
    min_r = min(r for r, _ in raw.values())
    min_c = min(c for _, c in raw.values())
    return {label: (r - min_r, c - min_c) for label, (r, c) in raw.items()}


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    """Single attempt to generate a valid D16 sample with a connected-but-broken decoy rail."""
    N = 3
    gap = rng.choice(GAP_CHOICES)
    cell_size = N + gap
    template = rng.choice(TEMPLATE_LIBRARY)
    cells = _transform_template_cells(template["cells"], rng.randrange(8))
    path_labels = list(template["path_nodes"])
    valid_edges = [tuple(edge) for edge in template["valid_edges"]]
    broken_edges = [tuple(edge) for edge in template["broken_edges"]]

    Kr = 1 + max(r for r, _ in cells.values())
    Kc = 1 + max(c for _, c in cells.values())
    min_S = max(Kr, Kc) * cell_size + 2
    if min_S > GRID_SIZE_RANGE[1]:
        return None
    S = rng.randint(max(GRID_SIZE_RANGE[0], min_S), GRID_SIZE_RANGE[1])

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
    label_to_pos: dict[str, tuple[int, int]] = {}
    for label, cell in cells.items():
        pos = _cell_to_square_pos(cell, N, cell_size, origin)
        label_to_pos[label] = pos
        color = terminal_color if label in ("T0", "T1") else station_color
        r0, c0 = pos
        canvas[r0 : r0 + N, c0 : c0 + N] = color
        node_mask[r0 : r0 + N, c0 : c0 + N] = True

    track_mask = np.zeros((S, S), dtype=bool)
    edge_track_pixels: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for a, b in valid_edges:
        pixels = _track_between_positions(cells[a], cells[b], label_to_pos[a], label_to_pos[b], N)
        for r, c in pixels:
            if not (0 <= r < S and 0 <= c < S):
                return None
            if node_mask[r, c] or track_mask[r, c]:
                return None
        for r, c in pixels:
            canvas[r, c] = track_color
            track_mask[r, c] = True
        edge_track_pixels[tuple(sorted((a, b)))] = pixels

    broken_tracks = []
    for a, b in broken_edges[: rng.choice(BROKEN_TRACK_COUNT_CHOICES)]:
        pixels = _track_between_positions(cells[a], cells[b], label_to_pos[a], label_to_pos[b], N)
        if any(not (0 <= r < S and 0 <= c < S) for r, c in pixels):
            return None
        if any(node_mask[r, c] or track_mask[r, c] for r, c in pixels):
            return None
        axis = "h" if cells[a][0] == cells[b][0] else "v"
        removed = _break_track_pixels(pixels, axis, N, rng)
        if removed is None:
            return None
        drawn = [p for p in pixels if p not in removed]
        if any(track_mask[r, c] for r, c in drawn):
            return None
        for r, c in drawn:
            canvas[r, c] = track_color
            track_mask[r, c] = True
        broken_tracks.append(
            {
                "edge": tuple(sorted((a, b))),
                "pixels": pixels,
                "removed": removed,
                "drawn": drawn,
            }
        )

    if not broken_tracks:
        return None

    if np.argmax(np.bincount(canvas.ravel())) != bg:
        return None

    expected = canvas.copy()
    for label in path_labels:
        if label in ("T0", "T1"):
            continue
        r0, c0 = label_to_pos[label]
        expected[r0 : r0 + N, c0 : c0 + N] = int(Color.GREEN)
    for a, b in zip(path_labels, path_labels[1:]):
        for r, c in edge_track_pixels[tuple(sorted((a, b)))]:
            expected[r, c] = int(Color.GREY)

    try:
        actual = main(canvas)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    # Variant-specific guard: the broken rail must remain one component but be rejected.
    info = _analyze_grid(canvas)
    if len(info["broken_edges"]) < len(broken_tracks):
        return None
    broken_pixels = set(broken_tracks[0]["pixels"])
    if all(canvas[r, c] == track_color for r, c in broken_pixels):
        return None
    return canvas


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPT_RETRIES):
        out = _gen_attempt(rng)
        if out is not None:
            return out
    raise RuntimeError(f"generate_input: failed after {GEN_ATTEMPT_RETRIES} attempts")
