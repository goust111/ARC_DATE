# Variant of 7b0280bc
# Dimension: D1 — 路径选择策略 (path)
# Change: BFS最短路径（边数最少），且最短路径唯一... → 最长简单路径
#
# Design analysis
# - Concrete behavior:
#   The input still shows a graph encoded as solid square nodes plus 8-connected track
#   components. The only semantic change is which terminal-to-terminal route is selected.
#   A typical abstract graph is:
#       0-1-2-3-4-5-6
#       |         /
#       +--------+
#   Original logic recolors the unique shortest path 0-5-6 (2 edges). The variant must
#   recolor the longest simple path 0-1-2-3-4-5-6 (6 edges). In pixels this means the
#   same squares/tracks are detected as before, but many more intermediate stations and
#   track segments change color.
# - Complexity calibration:
#   The generator keeps 7-10 nodes and 1-2 cycles, so the graph is still sparse and easy
#   to read visually. To make the D1 change observable, it enforces a unique shortest
#   path of 2-3 edges and a unique longest simple path that is at least 3 edges longer.
#   This prevents degeneration back to tree cases where shortest == longest.
# - Implicit assumption review:
#   `find_solid_squares`, node-color inference, and track-component extraction all rely on
#   the old object vocabulary, which still holds: exactly two terminal squares share one
#   color, all station squares share another, and every track component touches exactly two
#   nodes. The old BFS block is the only invalid assumption: it hard-codes edge-count
#   optimality. In the variant, a short chord is intentionally present, so BFS would select
#   the wrong route. The solver therefore enumerates all simple terminal-to-terminal paths
#   and keeps the longest one.
# - Placement legality:
#   The generator constructs a snake-like backbone over a small cell grid, then adds 1-2
#   shortcut edges that are spatial adjacencies between nonconsecutive backbone cells.
#   The same square placement and track routing scheme from the original task is reused,
#   and every candidate is rejected if any track would overlap a node, overlap another
#   track, leave the canvas, or fail the oracle check `main(input) == expected_output`.
# - Visual observability:
#   Because the shortcut creates a visibly short route while the backbone winds through
#   most stations, the input/output examples clearly distinguish "take the shortest way"
#   from "take the longest simple way". The changed rule is visible from geometry alone;
#   no color-id ordering or hidden metadata is used.
# - Visual cleanliness / color inference:
#   Colors stay in the original regime: 4 total input colors, green/grey reserved for the
#   output. Nodes never overlap and tracks never touch each other except through their two
#   incident squares, so the graph remains legible. Green marks station squares on the
#   chosen route and grey marks only the track components on that same route.

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
import random
from pathlib import Path
import sys

import numpy as np

for _parent in Path(__file__).resolve().parents:
    _solvers_dir = _parent / "solvers"
    if _solvers_dir.is_dir():
        _solvers_dir_str = str(_solvers_dir)
        if _solvers_dir_str not in sys.path:
            sys.path.append(_solvers_dir_str)
        break

from arc_utils import Color, components_mask, colors_present, get_bg, neighbors8


def find_solid_squares(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    min_size: int = 2,
    connectivity: int = 8,
) -> list[dict]:
    """
    Local helper: find solid monochrome NxN squares (N>=min_size).

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


def _extract_graph(grid: np.ndarray):
    """Recover graph nodes and track edges from the pixel grid."""
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
    term_ids = sorted(map(int, nodes_by_color[int(terminal_color)]))

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

    return g, term_ids, id2pts, track_comp_pts, adj


def _enumerate_simple_paths_with_edges(
    adj: dict[int, list[tuple[int, int]]],
    start: int,
    goal: int,
) -> list[tuple[list[int], list[int]]]:
    """Enumerate all simple node paths and corresponding edge-id paths."""
    out: list[tuple[list[int], list[int]]] = []
    stack: list[tuple[int, list[int], list[int], set[int]]] = [(int(start), [int(start)], [], {int(start)})]

    while stack:
        u, node_path, edge_path, seen = stack.pop()
        if int(u) == int(goal):
            out.append((node_path, edge_path))
            continue
        nbrs = sorted(adj.get(int(u), []), key=lambda x: (int(x[0]), int(x[1])), reverse=True)
        for v, eid in nbrs:
            if int(v) in seen:
                continue
            stack.append(
                (
                    int(v),
                    node_path + [int(v)],
                    edge_path + [int(eid)],
                    seen | {int(v)},
                )
            )
    return out


def _select_longest_simple_path(
    adj: dict[int, list[tuple[int, int]]],
    start: int,
    goal: int,
) -> tuple[list[int], list[int]]:
    """Pick the longest simple path between the two terminals."""
    best_nodes: list[int] | None = None
    best_edges: list[int] | None = None
    best_len = -1

    for node_path, edge_path in _enumerate_simple_paths_with_edges(adj, int(start), int(goal)):
        cur_len = int(len(edge_path))
        if cur_len > best_len or (cur_len == best_len and (best_nodes is None or tuple(node_path) < tuple(best_nodes))):
            best_len = cur_len
            best_nodes = list(map(int, node_path))
            best_edges = list(map(int, edge_path))

    assert best_nodes is not None and best_edges is not None
    return best_nodes, best_edges


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the longest simple path between two terminal squares in a square-and-track graph.

    Input structure and graph extraction are unchanged from the original task. The only
    semantic difference is path selection: choose the longest simple terminal-to-terminal
    route instead of the shortest one.
    """
    g, term_ids, id2pts, track_comp_pts, adj = _extract_graph(grid)

    start, goal = int(term_ids[0]), int(term_ids[1])
    node_path, edge_path = _select_longest_simple_path(adj, start, goal)

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

NODE_SIZE_CHOICES = [2, 3]
NODE_COUNT_RANGE = (7, 10)
CYCLE_COUNT_CHOICES = [1, 2]
GAP_CHOICES = [2, 3]
GRID_SIZE_RANGE = (15, 30)
COLOR_POOL = [0, 1, 2, 4, 6, 7, 8, 9]
SHORT_PATH_LEN_CHOICES = [2, 3]
MIN_LONG_ADVANTAGE = 3
GEN_ATTEMPT_RETRIES = 300
LAYOUT_GRID_CHOICES = [(3, 3), (3, 4), (4, 3), (2, 5), (5, 2)]


def _compute_adjacency(picked: set[tuple[int, int]], use_diagonal: bool, allow_skip: bool):
    """
    Compute valid undirected edges between picked cells on a grid.

    The variant keeps the original spatial vocabulary. In practice the layouts below only
    rely on cardinal contacts, but skip-2 is left enabled so the admissible edge relation
    still matches the original generator.
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


def _snake_cells(Kr: int, Kc: int) -> list[tuple[int, int]]:
    """Snake order over a Kr x Kc cell grid."""
    out: list[tuple[int, int]] = []
    for r in range(int(Kr)):
        cols = range(int(Kc)) if r % 2 == 0 else range(int(Kc) - 1, -1, -1)
        for c in cols:
            out.append((int(r), int(c)))
    return out


def _transform_cells(
    cells: list[tuple[int, int]],
    Kr: int,
    Kc: int,
    mode: int,
) -> tuple[list[tuple[int, int]], int, int]:
    """Apply one of 8 rectangle symmetries to cell coordinates."""
    out: list[tuple[int, int]] = []
    for r, c in cells:
        if mode == 0:
            nr, nc = r, c
        elif mode == 1:
            nr, nc = r, Kc - 1 - c
        elif mode == 2:
            nr, nc = Kr - 1 - r, c
        elif mode == 3:
            nr, nc = Kr - 1 - r, Kc - 1 - c
        elif mode == 4:
            nr, nc = c, r
        elif mode == 5:
            nr, nc = c, Kr - 1 - r
        elif mode == 6:
            nr, nc = Kc - 1 - c, r
        else:
            nr, nc = Kc - 1 - c, Kr - 1 - r
        out.append((int(nr), int(nc)))

    if mode >= 4:
        return out, int(Kc), int(Kr)
    return out, int(Kr), int(Kc)


def _enumerate_simple_node_paths(adj: dict[int, list[int]], start: int, goal: int) -> list[list[int]]:
    """Enumerate all simple node paths in a small undirected graph."""
    out: list[list[int]] = []
    stack: list[tuple[int, list[int], set[int]]] = [(int(start), [int(start)], {int(start)})]

    while stack:
        u, node_path, seen = stack.pop()
        if int(u) == int(goal):
            out.append(node_path)
            continue
        for v in sorted(adj.get(int(u), []), reverse=True):
            if int(v) in seen:
                continue
            stack.append((int(v), node_path + [int(v)], seen | {int(v)}))
    return out


def _analyze_terminal_paths(edges: set[tuple[int, int]], V: int):
    """Return shortest/longest path stats between terminals 0 and V-1."""
    adj: dict[int, list[int]] = {i: [] for i in range(int(V))}
    for a, b in edges:
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))

    paths = _enumerate_simple_node_paths(adj, 0, int(V) - 1)
    if not paths:
        return None

    lengths = [int(len(p) - 1) for p in paths]
    short_len = min(lengths)
    long_len = max(lengths)
    short_paths = [p for p in paths if int(len(p) - 1) == int(short_len)]
    long_paths = [p for p in paths if int(len(p) - 1) == int(long_len)]
    return {
        "short_len": int(short_len),
        "long_len": int(long_len),
        "short_paths": short_paths,
        "long_paths": long_paths,
    }


def _find_layout(V: int, n_cycles: int, cell_size: int, rng: random.Random):
    """
    Build a snake backbone plus 1-2 shortcut edges.

    The backbone is intended to be the unique longest simple path from node 0 to node V-1.
    Shortcut edges create a visibly shorter terminal route, preserving the D1 contrast.
    """
    options = []

    dim_choices = list(LAYOUT_GRID_CHOICES)
    rng.shuffle(dim_choices)
    for Kr, Kc in dim_choices:
        if int(Kr * Kc) < int(V):
            continue
        if int(max(Kr, Kc) * cell_size + 2) > int(GRID_SIZE_RANGE[1]):
            continue

        base_path = _snake_cells(int(Kr), int(Kc))[: int(V)]
        modes = list(range(8))
        rng.shuffle(modes)
        for mode in modes:
            path_cells, new_Kr, new_Kc = _transform_cells(base_path, int(Kr), int(Kc), int(mode))
            picked = set(path_cells)
            cell_to_id = {cell: i for i, cell in enumerate(path_cells)}
            full_edges = _compute_adjacency(picked, use_diagonal=False, allow_skip=True)
            full_id_edges = sorted(
                {
                    tuple(sorted((int(cell_to_id[a]), int(cell_to_id[b]))))
                    for (a, b) in full_edges
                }
            )
            backbone_edges = {tuple(sorted((i, i + 1))) for i in range(int(V) - 1)}
            chord_candidates = [e for e in full_id_edges if e not in backbone_edges]
            if len(chord_candidates) < int(n_cycles):
                continue

            valid_combos = []
            for combo in combinations(chord_candidates, int(n_cycles)):
                final_edges = set(backbone_edges) | set(combo)
                stats = _analyze_terminal_paths(final_edges, int(V))
                if stats is None:
                    continue
                if int(stats["short_len"]) not in SHORT_PATH_LEN_CHOICES:
                    continue
                if len(stats["short_paths"]) != 1:
                    continue
                if int(stats["long_len"]) != int(V) - 1:
                    continue
                if int(stats["long_len"]) < int(stats["short_len"]) + int(MIN_LONG_ADVANTAGE):
                    continue
                if len(stats["long_paths"]) != 1:
                    continue
                if stats["long_paths"][0] != list(range(int(V))):
                    continue
                valid_combos.append(tuple(sorted(combo)))

            if valid_combos:
                options.append(
                    {
                        "Kr": int(new_Kr),
                        "Kc": int(new_Kc),
                        "path_cells": path_cells,
                        "backbone_edges": sorted(backbone_edges),
                        "valid_combos": valid_combos,
                    }
                )

    if not options:
        return None

    layout = rng.choice(options)
    chords = rng.choice(layout["valid_combos"])
    path_edges = [tuple(sorted((i, i + 1))) for i in range(int(V) - 1)]
    return {
        "Kr": int(layout["Kr"]),
        "Kc": int(layout["Kc"]),
        "path_cells": list(layout["path_cells"]),
        "edges": sorted(set(layout["backbone_edges"]) | set(chords)),
        "path_nodes": list(range(int(V))),
        "path_edges": path_edges,
    }


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
    assert sr is not None and er is not None

    pixels = set()
    for r, c in _walk_footprints(sr, sc, er, ec):
        for dr in range(Wt):
            for dc in range(Wt):
                pixels.add((r + dr, c + dc))
    return list(pixels)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    """Single attempt to generate a valid input grid."""
    N = rng.choice(NODE_SIZE_CHOICES)
    V = rng.randint(*NODE_COUNT_RANGE)
    n_cycles = rng.choice(CYCLE_COUNT_CHOICES)
    gap = rng.choice(GAP_CHOICES)
    cell_size = N + gap

    layout = _find_layout(int(V), int(n_cycles), int(cell_size), rng)
    if layout is None:
        return None

    Kr = int(layout["Kr"])
    Kc = int(layout["Kc"])
    path_cells = list(layout["path_cells"])
    edges = list(layout["edges"])
    path_nodes = list(layout["path_nodes"])
    path_edges = list(layout["path_edges"])

    min_S = int(max(Kr, Kc) * cell_size + 2)
    lo = max(int(GRID_SIZE_RANGE[0]), int(min_S))
    hi = int(GRID_SIZE_RANGE[1])
    if lo > hi:
        return None
    S = rng.randint(lo, hi)

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

    id_to_cell = {i: path_cells[i] for i in range(int(V))}
    id_to_pos = {}
    for i in range(int(V)):
        pos = _cell_to_square_pos(id_to_cell[i], int(N), int(cell_size), origin)
        id_to_pos[i] = pos
        color = terminal_color if i in (0, int(V) - 1) else station_color
        r0, c0 = pos
        canvas[r0 : r0 + N, c0 : c0 + N] = color
        node_mask[r0 : r0 + N, c0 : c0 + N] = True

    track_mask = np.zeros((S, S), dtype=bool)
    edge_track_pixels: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for a, b in edges:
        pixels = _track_between_cells(
            id_to_cell[int(a)],
            id_to_cell[int(b)],
            id_to_pos[int(a)],
            id_to_pos[int(b)],
            int(N),
        )
        for r, c in pixels:
            if not (0 <= r < S and 0 <= c < S):
                return None
            if node_mask[int(r), int(c)] or track_mask[int(r), int(c)]:
                return None
        for r, c in pixels:
            canvas[int(r), int(c)] = int(track_color)
            track_mask[int(r), int(c)] = True
        edge_track_pixels[tuple(sorted((int(a), int(b))))] = pixels

    if int(np.argmax(np.bincount(canvas.ravel()))) != int(bg):
        return None

    expected = canvas.copy()
    for nid in path_nodes:
        if int(nid) in (0, int(V) - 1):
            continue
        r0, c0 = id_to_pos[int(nid)]
        expected[r0 : r0 + N, c0 : c0 + N] = int(Color.GREEN)
    for e in path_edges:
        for r, c in edge_track_pixels[tuple(sorted(e))]:
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
