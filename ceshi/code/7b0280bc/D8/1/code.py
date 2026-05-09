# Variant of 7b0280bc
# Dimension: D8 — 多步处理序列 (multi_step)
# Change: 单步处理：找最短路径并一次性涂色... → 先删除再路径

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np

from arc_utils import Color, colors_present, components_mask, get_bg, neighbors8


# ---------------------------------------------------------------------------
# Design analysis for the D8 variant
# ---------------------------------------------------------------------------
#
# a) Concrete behavior and trace
#    Original task: interpret the picture as a graph, then directly recolor the
#    unique terminal-to-terminal shortest path.
#    New task: still parse the same kind of node/track graph, but first delete
#    the visually longest track segment, then compute the shortest path in the
#    remaining graph.
#
#    Pixel-level embodiment in this variant:
#    - Terminals are still two solid 2x2 squares of one color.
#    - Stations are still several solid 2x2 squares of another color.
#    - Tracks are still 1-cell-wide connected components of a third color.
#    - Exactly one track component is visibly longer than all others.
#    - Output removes that longest component (paint it back to background),
#      then paints the new shortest surviving route: intermediate stations green
#      and route tracks grey.
#
#    Small graph trace used by the generator:
#    - Original graph edges:
#      (0-1), (1-6), (0-2), (2-3), (3-6), (1-5), (5-2), (3-4)
#    - Before deletion, shortest terminal path is 0-1-6 (2 edges).
#    - The longest track is edge (0-1), so stage 1 deletes it.
#    - After deletion, the new unique shortest path is 0-2-3-6 (3 edges).
#    Difference from the original logic:
#    - Original solver would grey edge (0-1) and edge (1-6), and green node 1.
#    - Variant solver removes edge (0-1) entirely, then greys
#      (0-2), (2-3), (3-6), and greens nodes 2 and 3.
#
# b) Complexity calibration
#    The rule is not "delete an arbitrary edge". It is "delete the longest track
#    component", which is visible from the picture itself. The generator keeps
#    that longest component unique, so the deletion stage is learnable from a
#    few examples instead of being ambiguous.
#
# c) Hidden-assumption audit against the original solver
#    - Original BFS used the full adjacency graph immediately. That no longer
#      matches the task: we must build the full graph first, then remove some
#      edges, then BFS on the reduced graph.
#    - Original recoloring assumed every path edge remains present in output.
#      Here deleted edges must be painted back to background before path
#      recoloring, otherwise the variant is visually invisible.
#    - Original generator only needed a unique shortest path once. The new
#      generator must guarantee two different facts:
#      1. the deleted edge is the longest visible track component;
#      2. after deletion the reduced graph still has a unique terminal path.
#
# d) Placement legality
#    The generator uses a hand-validated embedded graph instead of a free router.
#    Every node square and every track pixel is explicitly placed, so there is
#    no overlap, no out-of-bounds write, and track components stay separated
#    under 8-connectivity.
#
# e) Visual observability
#    Humans can infer the new rule because the output shows two coupled effects:
#    - the single longest segment disappears;
#    - the highlighted route reroutes around that deletion.
#    This makes the "delete first, then plan" sequence visible in 2-3 examples.
#
# f) Visual cleanliness
#    The graph is sparse, planar-looking, and uses only one deletion target.
#    Tracks do not cross or touch each other accidentally, so each component is
#    readable as one edge.
#
# g) Color-change inferability
#    Colors keep the original semantics: green marks path stations, grey marks
#    path tracks, background restores deleted tracks. There is no hidden use of
#    numeric color ids.


def find_solid_squares(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    min_size: int = 2,
    connectivity: int = 8,
) -> list[dict]:
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


def _parse_graph(grid: np.ndarray) -> dict:
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

    edge_infos: list[dict] = []
    full_adj: dict[int, list[tuple[int, int]]] = defaultdict(list)
    track_comps = components_mask(g == int(track_color), connectivity=8)
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
        info = {
            "id": int(edge_id),
            "pts": pts,
            "nodes": (int(a), int(b)),
            "length": int(len(pts)),
        }
        edge_infos.append(info)
        full_adj[int(a)].append((int(b), int(edge_id)))
        full_adj[int(b)].append((int(a), int(edge_id)))

    assert edge_infos
    return {
        "grid": g,
        "bg": bg,
        "nodes": nodes,
        "id2pts": id2pts,
        "term_ids": term_ids,
        "edge_infos": edge_infos,
        "full_adj": full_adj,
    }


def _shortest_path(
    start: int,
    goal: int,
    adj: dict[int, list[tuple[int, int]]],
) -> tuple[list[int], list[int]]:
    q = deque([int(start)])
    prev: dict[int, int | None] = {int(start): None}
    prev_edge: dict[int, int | None] = {int(start): None}

    while q:
        u = int(q.popleft())
        if u == int(goal):
            break
        for v, eid in adj.get(int(u), []):
            if int(v) in prev:
                continue
            prev[int(v)] = int(u)
            prev_edge[int(v)] = int(eid)
            q.append(int(v))

    assert int(goal) in prev

    node_path: list[int] = []
    edge_path: list[int] = []
    cur = int(goal)
    while cur is not None:
        node_path.append(int(cur))
        e = prev_edge.get(int(cur))
        if e is not None:
            edge_path.append(int(e))
        nxt = prev.get(int(cur))
        cur = int(nxt) if nxt is not None else None
    node_path.reverse()
    edge_path.reverse()
    return node_path, edge_path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule:
    1. Identify the longest track component and delete it (paint to background).
    2. On the remaining graph, find the terminal shortest path by edge count.
    3. Paint intermediate stations green and the surviving path tracks grey.
    """
    parsed = _parse_graph(np.array(grid, dtype=int))
    g = parsed["grid"]
    bg = int(parsed["bg"])
    term_ids = list(map(int, parsed["term_ids"]))
    id2pts = parsed["id2pts"]
    edge_infos = parsed["edge_infos"]

    max_len = int(max(int(e["length"]) for e in edge_infos))
    removed_edge_ids = {int(e["id"]) for e in edge_infos if int(e["length"]) == int(max_len)}

    adj: dict[int, list[tuple[int, int]]] = defaultdict(list)
    edge_pts: dict[int, list[tuple[int, int]]] = {}
    for info in edge_infos:
        eid = int(info["id"])
        edge_pts[eid] = list(info["pts"])
        if eid in removed_edge_ids:
            continue
        a, b = map(int, info["nodes"])
        adj[a].append((b, eid))
        adj[b].append((a, eid))

    start, goal = int(term_ids[0]), int(term_ids[1])
    node_path, edge_path = _shortest_path(start, goal, adj)

    out = g.copy()
    for eid in removed_edge_ids:
        for r, c in edge_pts.get(int(eid), []):
            out[int(r), int(c)] = int(bg)

    term_set = set(map(int, term_ids))
    for nid in node_path:
        if int(nid) in term_set:
            continue
        for r, c in id2pts[int(nid)]:
            out[int(r), int(c)] = int(Color.GREEN)

    for eid in edge_path:
        for r, c in edge_pts.get(int(eid), []):
            out[int(r), int(c)] = int(Color.GREY)

    return out


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

LOCAL_SIZE = 24
GRID_SIZE_RANGE = (24, 30)
COLOR_POOL = [0, 1, 2, 4, 6, 7, 8, 9]
GEN_ATTEMPT_RETRIES = 100

# Fixed abstract graph used by the variant:
# terminals = 0 and 6
# deleted longest edge = (0, 1)
# surviving shortest path = 0-2-3-6
BASE_NODE_PIXELS: dict[int, set[tuple[int, int]]] = {
    0: {(2, 2), (2, 3), (3, 2), (3, 3)},
    1: {(2, 14), (2, 15), (3, 14), (3, 15)},
    2: {(8, 2), (8, 3), (9, 2), (9, 3)},
    5: {(8, 10), (8, 11), (9, 10), (9, 11)},
    6: {(8, 20), (8, 21), (9, 20), (9, 21)},
    3: {(14, 10), (14, 11), (15, 10), (15, 11)},
    4: {(20, 10), (20, 11), (21, 10), (21, 11)},
}

BASE_EDGE_PIXELS: dict[tuple[int, int], set[tuple[int, int]]] = {
    (0, 1): {(2, c) for c in range(4, 14)},
    (1, 6): {(2, 16), (3, 17), (4, 18), (5, 19), (6, 19), (7, 19), (8, 19)},
    (1, 5): {(4, 14), (5, 13), (6, 12), (7, 11)},
    (0, 2): {(4, 3), (5, 2), (6, 2), (7, 2)},
    (5, 2): {(8, c) for c in range(4, 10)},
    (2, 3): {(10, 3), (11, 4), (12, 5), (13, 6), (14, 7), (14, 8), (14, 9)},
    (3, 6): {(14, 12), (13, 13), (12, 14), (11, 15), (10, 16), (10, 17), (10, 18), (10, 19), (10, 20)},
    (3, 4): {(16, 10), (17, 10), (18, 10), (19, 10)},
}

DELETED_EDGE = (0, 1)
PATH_NODES_AFTER_DELETE = {2, 3}
PATH_EDGES_AFTER_DELETE = {(0, 2), (2, 3), (3, 6)}
TERMINAL_IDS = {0, 6}


def _transform_point(pt: tuple[int, int], size: int, rot: int, flip_lr: bool) -> tuple[int, int]:
    r, c = map(int, pt)
    for _ in range(int(rot) % 4):
        r, c = c, int(size - 1 - r)
    if flip_lr:
        c = int(size - 1 - c)
    return int(r), int(c)


def _transform_pixels(
    pixels: set[tuple[int, int]],
    size: int,
    rot: int,
    flip_lr: bool,
    shift: tuple[int, int],
) -> set[tuple[int, int]]:
    dr, dc = map(int, shift)
    out = set()
    for pt in pixels:
        r, c = _transform_point(pt, size, rot, flip_lr)
        out.add((int(r + dr), int(c + dc)))
    return out


def _paint(
    canvas: np.ndarray,
    node_pixels: dict[int, set[tuple[int, int]]],
    edge_pixels: dict[tuple[int, int], set[tuple[int, int]]],
    *,
    bg: int,
    terminal_color: int,
    station_color: int,
    track_color: int,
) -> np.ndarray:
    out = np.full(canvas.shape, int(bg), dtype=int)
    for nid, pts in node_pixels.items():
        col = int(terminal_color) if int(nid) in TERMINAL_IDS else int(station_color)
        for r, c in pts:
            out[int(r), int(c)] = col
    for pts in edge_pixels.values():
        for r, c in pts:
            out[int(r), int(c)] = int(track_color)
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    S = rng.randint(*GRID_SIZE_RANGE)
    rot = rng.randint(0, 3)
    flip_lr = bool(rng.randint(0, 1))
    shift = (
        rng.randint(0, int(S - LOCAL_SIZE)),
        rng.randint(0, int(S - LOCAL_SIZE)),
    )

    node_pixels = {
        int(nid): _transform_pixels(set(pts), LOCAL_SIZE, rot, flip_lr, shift)
        for nid, pts in BASE_NODE_PIXELS.items()
    }
    edge_pixels = {
        tuple(sorted(edge)): _transform_pixels(set(pts), LOCAL_SIZE, rot, flip_lr, shift)
        for edge, pts in BASE_EDGE_PIXELS.items()
    }

    colors = rng.sample(COLOR_POOL, 4)
    bg, terminal_color, station_color, track_color = map(int, colors)

    blank = np.full((S, S), int(bg), dtype=int)
    inp = _paint(
        blank,
        node_pixels,
        edge_pixels,
        bg=bg,
        terminal_color=terminal_color,
        station_color=station_color,
        track_color=track_color,
    )

    expected = inp.copy()
    for r, c in edge_pixels[tuple(sorted(DELETED_EDGE))]:
        expected[int(r), int(c)] = int(bg)
    for nid in PATH_NODES_AFTER_DELETE:
        for r, c in node_pixels[int(nid)]:
            expected[int(r), int(c)] = int(Color.GREEN)
    for edge in PATH_EDGES_AFTER_DELETE:
        for r, c in edge_pixels[tuple(sorted(edge))]:
            expected[int(r), int(c)] = int(Color.GREY)

    # Generator-side assumption check:
    # - longest edge remains unique after symmetry/translation;
    # - solver sees a valid path after deletion;
    # - no hidden merge makes the instance degenerate.
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPT_RETRIES):
        out = _gen_attempt(rng)
        if out is not None:
            return out
    raise RuntimeError(f"generate_input: failed after {GEN_ATTEMPT_RETRIES} attempts")
