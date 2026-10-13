import random
from collections import defaultdict, deque

import numpy as np


BLACK = 0
YELLOW = 4
GRAY = 5
ORANGE = 7
SOURCE_COLORS = (1, 2, 3, 6, 8, 9)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_runs_1d(values: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    n = int(values.size)
    i = 0
    while i < n:
        if not bool(values[i]):
            i += 1
            continue
        j = i
        while j + 1 < n and bool(values[j + 1]):
            j += 1
        runs.append((int(i), int(j)))
        i = j + 1
    return runs


def _detect_segments(mask: np.ndarray, orient: str) -> list[dict]:
    h, w = mask.shape
    segments: list[dict] = []

    if orient == "H":
        active: dict[tuple[int, int], tuple[int, int]] = {}
        for r in range(h + 1):
            runs = _find_runs_1d(mask[r]) if r < h else []
            run_set = set(runs)
            next_active: dict[tuple[int, int], tuple[int, int]] = {}
            for interval in runs:
                if interval in active:
                    r0, _ = active[interval]
                    next_active[interval] = (int(r0), int(r))
                else:
                    next_active[interval] = (int(r), int(r))
            for interval, (r0, r1) in active.items():
                if interval in run_set:
                    continue
                c0, c1 = interval
                thickness = int(r1 - r0 + 1)
                length = int(c1 - c0 + 1)
                if 1 <= thickness <= 3 and length > thickness:
                    segments.append(
                        {
                            "orient": "H",
                            "r0": int(r0),
                            "r1": int(r1),
                            "c0": int(c0),
                            "c1": int(c1),
                            "width": int(thickness),
                        }
                    )
            active = next_active
        return segments

    active_v: dict[tuple[int, int], tuple[int, int]] = {}
    for c in range(w + 1):
        runs = _find_runs_1d(mask[:, c]) if c < w else []
        run_set = set(runs)
        next_active_v: dict[tuple[int, int], tuple[int, int]] = {}
        for interval in runs:
            if interval in active_v:
                c0, _ = active_v[interval]
                next_active_v[interval] = (int(c0), int(c))
            else:
                next_active_v[interval] = (int(c), int(c))
        for interval, (c0, c1) in active_v.items():
            if interval in run_set:
                continue
            r0, r1 = interval
            thickness = int(c1 - c0 + 1)
            length = int(r1 - r0 + 1)
            if 1 <= thickness <= 3 and length > thickness:
                segments.append(
                    {
                        "orient": "V",
                        "r0": int(r0),
                        "r1": int(r1),
                        "c0": int(c0),
                        "c1": int(c1),
                        "width": int(thickness),
                    }
                )
        active_v = next_active_v
    return segments


def _rect_coords(r0: int, r1: int, c0: int, c1: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _extract_graph(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    pipe = g == GRAY

    sources: dict[tuple[int, int], int] = {}
    terminals: set[tuple[int, int]] = set()
    for r in range(h):
        for c in range(w):
            val = int(g[r, c])
            if val == YELLOW:
                terminals.add((int(r), int(c)))
            elif val not in (BLACK, GRAY):
                assert r in (0, h - 1) or c in (0, w - 1), "sources must lie on grid edge"
                sources[(int(r), int(c))] = val

    assert 1 <= len(sources) <= 3, "expected 1-3 edge sources"

    h_segments = _detect_segments(pipe, "H")
    v_segments = _detect_segments(pipe, "V")
    assert h_segments or v_segments, "expected at least one pipe segment"

    nodes: list[dict] = []
    edges: list[dict] = []
    node_incident: defaultdict[int, list[int]] = defaultdict(list)
    h_hits: defaultdict[int, list[tuple[int, int, int]]] = defaultdict(list)
    v_hits: defaultdict[int, list[tuple[int, int, int]]] = defaultdict(list)

    for hi, hs in enumerate(h_segments):
        for vi, vs in enumerate(v_segments):
            ir0 = max(int(hs["r0"]), int(vs["r0"]))
            ir1 = min(int(hs["r1"]), int(vs["r1"]))
            ic0 = max(int(hs["c0"]), int(vs["c0"]))
            ic1 = min(int(hs["c1"]), int(vs["c1"]))
            if ir0 > ir1 or ic0 > ic1:
                continue
            node_id = len(nodes)
            nodes.append(
                {
                    "kind": "branch",
                    "mask": _rect_coords(ir0, ir1, ic0, ic1),
                }
            )
            h_hits[hi].append((int(ic0), int(ic1), int(node_id)))
            v_hits[vi].append((int(ir0), int(ir1), int(node_id)))

    def add_endpoint_node(kind: str, mask: list[tuple[int, int]] | None = None, color: int | None = None) -> int:
        node_id = len(nodes)
        nodes.append({"kind": kind, "mask": mask or [], "color": color})
        return node_id

    def endpoint_from_side(seg: dict, side: str) -> int:
        r0 = int(seg["r0"])
        r1 = int(seg["r1"])
        c0 = int(seg["c0"])
        c1 = int(seg["c1"])
        adjacent: list[tuple[int, int]] = []
        if side == "left" and c0 > 0:
            adjacent = [(r, c0 - 1) for r in range(r0, r1 + 1)]
        elif side == "right" and c1 + 1 < w:
            adjacent = [(r, c1 + 1) for r in range(r0, r1 + 1)]
        elif side == "top" and r0 > 0:
            adjacent = [(r0 - 1, c) for c in range(c0, c1 + 1)]
        elif side == "bottom" and r1 + 1 < h:
            adjacent = [(r1 + 1, c) for c in range(c0, c1 + 1)]

        source_cells = [(r, c, sources[(r, c)]) for r, c in adjacent if (r, c) in sources]
        terminal_cells = [(r, c) for r, c in adjacent if (r, c) in terminals]

        assert len(source_cells) + len(terminal_cells) <= 1, "segment end must have at most one marker"
        if source_cells:
            sr, sc, color = source_cells[0]
            return add_endpoint_node("source", [(int(sr), int(sc))], int(color))
        if terminal_cells:
            tr, tc = terminal_cells[0]
            return add_endpoint_node("terminal", [(int(tr), int(tc))], YELLOW)
        return add_endpoint_node("deadend")

    def add_edge(a: int, b: int, width: int, mask: list[tuple[int, int]]) -> None:
        if not mask:
            return
        edge_id = len(edges)
        edges.append({"a": int(a), "b": int(b), "width": int(width), "mask": mask})
        node_incident[int(a)].append(int(edge_id))
        node_incident[int(b)].append(int(edge_id))

    for hi, seg in enumerate(h_segments):
        c0 = int(seg["c0"])
        c1 = int(seg["c1"])
        r0 = int(seg["r0"])
        r1 = int(seg["r1"])
        markers: list[tuple[int, int, int, str]] = []
        hits = sorted(h_hits.get(hi, []))
        if not hits or hits[0][0] > c0:
            markers.append((int(c0 - 1), int(c0 - 1), endpoint_from_side(seg, "left"), "endpoint"))
        for ic0, ic1, node_id in hits:
            markers.append((int(ic0), int(ic1), int(node_id), "branch"))
        if not hits or hits[-1][1] < c1:
            markers.append((int(c1 + 1), int(c1 + 1), endpoint_from_side(seg, "right"), "endpoint"))
        markers.sort(key=lambda x: (x[0], x[1]))
        for left, right in zip(markers, markers[1:]):
            if left[3] == "endpoint":
                seg_c0 = c0
            else:
                seg_c0 = left[1] + 1
            if right[3] == "endpoint":
                seg_c1 = c1
            else:
                seg_c1 = right[0] - 1
            if seg_c0 > seg_c1:
                continue
            add_edge(
                int(left[2]),
                int(right[2]),
                int(seg["width"]),
                _rect_coords(r0, r1, int(seg_c0), int(seg_c1)),
            )

    for vi, seg in enumerate(v_segments):
        r0 = int(seg["r0"])
        r1 = int(seg["r1"])
        c0 = int(seg["c0"])
        c1 = int(seg["c1"])
        markers_v: list[tuple[int, int, int, str]] = []
        hits = sorted(v_hits.get(vi, []))
        if not hits or hits[0][0] > r0:
            markers_v.append((int(r0 - 1), int(r0 - 1), endpoint_from_side(seg, "top"), "endpoint"))
        for ir0, ir1, node_id in hits:
            markers_v.append((int(ir0), int(ir1), int(node_id), "branch"))
        if not hits or hits[-1][1] < r1:
            markers_v.append((int(r1 + 1), int(r1 + 1), endpoint_from_side(seg, "bottom"), "endpoint"))
        markers_v.sort(key=lambda x: (x[0], x[1]))
        for top, bottom in zip(markers_v, markers_v[1:]):
            if top[3] == "endpoint":
                seg_r0 = r0
            else:
                seg_r0 = top[1] + 1
            if bottom[3] == "endpoint":
                seg_r1 = r1
            else:
                seg_r1 = bottom[0] - 1
            if seg_r0 > seg_r1:
                continue
            add_edge(
                int(top[2]),
                int(bottom[2]),
                int(seg["width"]),
                _rect_coords(int(seg_r0), int(seg_r1), c0, c1),
            )

    covered = np.zeros_like(pipe, dtype=int)
    for node in nodes:
        if node["kind"] != "branch":
            continue
        for r, c in node["mask"]:
            covered[r, c] += 1
    for edge in edges:
        for r, c in edge["mask"]:
            covered[r, c] += 1
    assert np.array_equal(covered > 0, pipe), "pipe pixels must be fully explained by the graph"

    source_nodes = [i for i, node in enumerate(nodes) if node["kind"] == "source"]
    terminal_nodes = [i for i, node in enumerate(nodes) if node["kind"] == "terminal"]
    assert len(source_nodes) == len(sources), "every source marker must attach to exactly one segment end"
    assert terminal_nodes, "expected at least one terminal"
    for node_id in source_nodes:
        assert len(node_incident[node_id]) == 1
    for node_id in terminal_nodes:
        assert len(node_incident[node_id]) == 1

    return {
        "nodes": nodes,
        "edges": edges,
        "incident": {k: list(v) for k, v in node_incident.items()},
        "source_nodes": source_nodes,
    }


def _effective_color(colors: set[int]) -> int | None:
    if not colors:
        return None
    if ORANGE in colors or len(colors) > 1:
        return ORANGE
    return next(iter(colors))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate source colors through a thick pipe graph, choosing widest branches at each fork.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    graph = _extract_graph(g)
    nodes = graph["nodes"]
    edges = graph["edges"]
    incident = graph["incident"]
    out = g.copy()

    node_arrivals: list[defaultdict[object, set[int]]] = [defaultdict(set) for _ in nodes]
    node_effective: list[int | None] = [None] * len(nodes)
    sent_colors: defaultdict[tuple[int, object], set[int]] = defaultdict(set)
    edge_seen: list[set[int]] = [set() for _ in edges]
    edge_effective: list[int | None] = [None] * len(edges)
    queue: deque[tuple[int, object, int]] = deque()

    def other_node(edge_id: int, node_id: int) -> int:
        edge = edges[edge_id]
        return int(edge["b"] if edge["a"] == node_id else edge["a"])

    def paint_node(node_id: int, color: int | None) -> None:
        if color is None:
            return
        node = nodes[node_id]
        if node["kind"] == "source":
            return
        for r, c in node["mask"]:
            out[r, c] = int(color)

    def choose_outgoing(node_id: int, incoming_key: object) -> list[int]:
        node = nodes[node_id]
        inc = incident.get(node_id, [])
        if node["kind"] == "source":
            return list(inc)
        if node["kind"] in {"terminal", "deadend"}:
            return []
        assert node["kind"] == "branch"
        assert isinstance(incoming_key, int)
        candidates = [edge_id for edge_id in inc if edge_id != incoming_key]
        if not candidates:
            return []
        max_width = max(int(edges[edge_id]["width"]) for edge_id in candidates)
        return [edge_id for edge_id in candidates if int(edges[edge_id]["width"]) == max_width]

    def push_from(node_id: int, incoming_key: object, color: int | None) -> None:
        if color is None:
            return
        key = (int(node_id), incoming_key)
        if int(color) in sent_colors[key]:
            return
        sent_colors[key].add(int(color))
        for edge_id in choose_outgoing(node_id, incoming_key):
            edge_seen[edge_id].add(int(color))
            eff = _effective_color(edge_seen[edge_id])
            if edge_effective[edge_id] != eff:
                edge_effective[edge_id] = eff
                for r, c in edges[edge_id]["mask"]:
                    out[r, c] = int(eff)
            queue.append((other_node(edge_id, node_id), int(edge_id), int(eff)))

    for node_id in graph["source_nodes"]:
        source_color = int(nodes[node_id]["color"])
        queue.append((int(node_id), ("source", int(node_id)), source_color))

    while queue:
        node_id, incoming_key, color = queue.popleft()
        if int(color) in node_arrivals[node_id][incoming_key]:
            continue
        node_arrivals[node_id][incoming_key].add(int(color))

        all_colors = set()
        for seen_colors in node_arrivals[node_id].values():
            all_colors.update(seen_colors)
        new_effective = _effective_color(all_colors)
        old_effective = node_effective[node_id]
        node_effective[node_id] = new_effective
        if old_effective != new_effective:
            paint_node(node_id, new_effective)
            for prior_key in list(node_arrivals[node_id].keys()):
                push_from(node_id, prior_key, new_effective)
        else:
            push_from(node_id, incoming_key, new_effective)

    return out


def _paint_rect(grid: np.ndarray, r0: int, r1: int, c0: int, c1: int, color: int) -> None:
    window = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all((window == BLACK) | (window == color))
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _paint_point(grid: np.ndarray, r: int, c: int, color: int) -> None:
    assert int(grid[r, c]) == BLACK
    grid[r, c] = int(color)


def _occupied_mask(grid: np.ndarray) -> np.ndarray:
    return grid != BLACK


def _can_place_component(grid: np.ndarray, cells: set[tuple[int, int]]) -> bool:
    h, w = grid.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if int(grid[r, c]) != BLACK:
            return False
    for r, c in cells:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells and int(grid[nr, nc]) != BLACK:
                return False
    return True


def _add_aux_straight(grid: np.ndarray, rng: random.Random, color: int) -> bool:
    h, w = grid.shape
    sides = ["left", "right", "top", "bottom"]
    rng.shuffle(sides)
    for side in sides:
        for _ in range(50):
            width = rng.randint(1, 3)
            if side in ("left", "right"):
                max_len = min(8, w - 3)
                if max_len < 4 or h < width:
                    continue
                length = rng.randint(4, max_len)
                r0 = rng.randint(0, h - width)
                source_row = r0 + rng.randrange(width)
                term_row = r0 + rng.randrange(width)
                if side == "left":
                    pipe = {(r, c) for r in range(r0, r0 + width) for c in range(1, length + 1)}
                    points = {(source_row, 0, color), (term_row, length + 1, YELLOW)}
                else:
                    pipe = {(r, c) for r in range(r0, r0 + width) for c in range(w - length - 1, w - 1)}
                    points = {(source_row, w - 1, color), (term_row, w - length - 2, YELLOW)}
            else:
                max_len = min(8, h - 3)
                if max_len < 4 or w < width:
                    continue
                length = rng.randint(4, max_len)
                c0 = rng.randint(0, w - width)
                source_col = c0 + rng.randrange(width)
                term_col = c0 + rng.randrange(width)
                if side == "top":
                    pipe = {(r, c) for r in range(1, length + 1) for c in range(c0, c0 + width)}
                    points = {(0, source_col, color), (length + 1, term_col, YELLOW)}
                else:
                    pipe = {(r, c) for r in range(h - length - 1, h - 1) for c in range(c0, c0 + width)}
                    points = {(h - 1, source_col, color), (h - length - 2, term_col, YELLOW)}

            cells = set(pipe) | {(r, c) for r, c, _ in points}
            if not _can_place_component(grid, cells):
                continue
            for r, c in pipe:
                grid[r, c] = GRAY
            for r, c, val in points:
                grid[r, c] = int(val)
            return True
    return False


def _build_single_mode(rng: random.Random, target_sources: int) -> tuple[np.ndarray, dict]:
    wh = rng.choice((2, 3))
    wv = rng.choice(tuple(w for w in (1, 2) if w < wh))
    up_len = rng.randint(3, 5)
    left_len = rng.randint(4, 7)
    right_len = rng.randint(5, 8)
    top_margin = rng.randint(2, 4)
    bottom_margin = rng.randint(3, 6)
    right_margin = rng.randint(2, 5)
    h = top_margin + up_len + wh + bottom_margin
    w = 1 + left_len + wv + right_len + 2 + right_margin
    grid = np.zeros((h, w), dtype=int)

    hr0 = top_margin + up_len
    hr1 = hr0 + wh - 1
    vc0 = 1 + left_len
    vc1 = vc0 + wv - 1
    h_end = vc1 + right_len
    v_top = top_margin

    _paint_rect(grid, hr0, hr1, 1, h_end, GRAY)
    _paint_rect(grid, v_top, hr1, vc0, vc1, GRAY)

    src_row = hr0 + rng.randrange(wh)
    top_term = (v_top - 1, vc0 + rng.randrange(wv))
    right_term = (hr0 + rng.randrange(wh), h_end + 1)
    src_color = rng.choice(SOURCE_COLORS)
    _paint_point(grid, src_row, 0, src_color)
    _paint_point(grid, top_term[0], top_term[1], YELLOW)
    _paint_point(grid, right_term[0], right_term[1], YELLOW)

    used_colors = {src_color}
    for _ in range(target_sources - 1):
        extra_color = rng.choice([c for c in SOURCE_COLORS if c not in used_colors])
        if not _add_aux_straight(grid, rng, extra_color):
            raise RuntimeError("aux placement failed")
        used_colors.add(extra_color)

    meta = {
        "mode": "single",
        "src_color": int(src_color),
        "active_terminal": right_term,
        "inactive_terminal": top_term,
    }
    return grid, meta


def _build_tie_mode(rng: random.Random, target_sources: int) -> tuple[np.ndarray, dict]:
    wh = rng.randint(1, 3)
    wv = rng.randint(1, 3)
    up_len = rng.randint(3, 5)
    down_len = rng.randint(3, 5)
    top_margin = rng.randint(2, 4)
    bottom_margin = rng.randint(2, 4)
    right_margin = rng.randint(5, 8)
    left_len = rng.randint(5, 8)
    h = top_margin + up_len + wh + down_len + bottom_margin
    w = 1 + left_len + wv + right_margin
    grid = np.zeros((h, w), dtype=int)

    hr0 = top_margin + up_len
    hr1 = hr0 + wh - 1
    vc0 = 1 + left_len
    vc1 = vc0 + wv - 1
    v_top = top_margin
    v_bottom = hr1 + down_len

    _paint_rect(grid, hr0, hr1, 1, vc1, GRAY)
    _paint_rect(grid, v_top, v_bottom, vc0, vc1, GRAY)

    src_row = hr0 + rng.randrange(wh)
    top_term = (v_top - 1, vc0 + rng.randrange(wv))
    bottom_term = (v_bottom + 1, vc0 + rng.randrange(wv))
    src_color = rng.choice(SOURCE_COLORS)
    _paint_point(grid, src_row, 0, src_color)
    _paint_point(grid, top_term[0], top_term[1], YELLOW)
    _paint_point(grid, bottom_term[0], bottom_term[1], YELLOW)

    used_colors = {src_color}
    for _ in range(target_sources - 1):
        extra_color = rng.choice([c for c in SOURCE_COLORS if c not in used_colors])
        if not _add_aux_straight(grid, rng, extra_color):
            raise RuntimeError("aux placement failed")
        used_colors.add(extra_color)

    meta = {
        "mode": "tie",
        "src_color": int(src_color),
        "top_terminal": top_term,
        "bottom_terminal": bottom_term,
    }
    return grid, meta


def _build_collision_mode(rng: random.Random, target_sources: int) -> tuple[np.ndarray, dict]:
    wv = rng.randint(1, 2)
    wh = rng.choice(tuple(w for w in (2, 3) if w > wv))
    top_len = rng.randint(3, 5)
    bottom_len = rng.randint(3, 5)
    left_margin = rng.randint(2, 5)
    right_len = rng.randint(5, 8)
    right_margin = rng.randint(2, 5)
    h = top_len + wh + bottom_len + 2
    w = left_margin + wv + right_len + 2 + right_margin
    grid = np.zeros((h, w), dtype=int)

    hr0 = 1 + top_len
    hr1 = hr0 + wh - 1
    vc0 = left_margin
    vc1 = vc0 + wv - 1
    h_end = vc1 + right_len

    _paint_rect(grid, 1, h - 2, vc0, vc1, GRAY)
    _paint_rect(grid, hr0, hr1, vc0, h_end, GRAY)

    top_color, bottom_color = rng.sample(SOURCE_COLORS, 2)
    top_src = (0, vc0 + rng.randrange(wv))
    bottom_src = (h - 1, vc0 + rng.randrange(wv))
    right_term = (hr0 + rng.randrange(wh), h_end + 1)
    _paint_point(grid, top_src[0], top_src[1], top_color)
    _paint_point(grid, bottom_src[0], bottom_src[1], bottom_color)
    _paint_point(grid, right_term[0], right_term[1], YELLOW)

    if target_sources == 3:
        extra_color = rng.choice([c for c in SOURCE_COLORS if c not in {top_color, bottom_color}])
        if not _add_aux_straight(grid, rng, extra_color):
            raise RuntimeError("aux placement failed")

    meta = {
        "mode": "collision",
        "merge_rect": (hr0, hr1, vc0, vc1),
        "right_terminal": right_term,
    }
    return grid, meta


def _build_mixed_mode(rng: random.Random, target_sources: int) -> tuple[np.ndarray, dict]:
    wv1 = 1
    wh = 2
    wv2 = 3
    top_len = rng.randint(4, 6)
    bottom_len = rng.randint(4, 6)
    left_margin = rng.randint(2, 4)
    bridge_len = rng.randint(5, 8)
    right_margin = rng.randint(2, 4)
    h = top_len + wh + bottom_len + 2
    w = left_margin + wv1 + bridge_len + wv2 + right_margin
    grid = np.zeros((h, w), dtype=int)

    hr0 = 1 + top_len
    hr1 = hr0 + wh - 1
    v1c0 = left_margin
    v1c1 = v1c0 + wv1 - 1
    v2c0 = v1c1 + 1 + bridge_len
    v2c1 = v2c0 + wv2 - 1

    available_up = hr0 - 1
    available_down = h - hr1 - 2
    up_len = rng.randint(3, min(5, available_up))
    down_len = rng.randint(3, min(5, available_down))
    v2_top = hr0 - up_len
    v2_bottom = hr1 + down_len

    _paint_rect(grid, 1, h - 2, v1c0, v1c1, GRAY)
    _paint_rect(grid, hr0, hr1, v1c0, v2c1, GRAY)
    _paint_rect(grid, v2_top, v2_bottom, v2c0, v2c1, GRAY)

    top_color, bottom_color = rng.sample(SOURCE_COLORS, 2)
    top_src = (0, v1c0 + rng.randrange(wv1))
    bottom_src = (h - 1, v1c0 + rng.randrange(wv1))
    top_term = (v2_top - 1, v2c0 + rng.randrange(wv2))
    bottom_term = (v2_bottom + 1, v2c0 + rng.randrange(wv2))
    _paint_point(grid, top_src[0], top_src[1], top_color)
    _paint_point(grid, bottom_src[0], bottom_src[1], bottom_color)
    _paint_point(grid, top_term[0], top_term[1], YELLOW)
    _paint_point(grid, bottom_term[0], bottom_term[1], YELLOW)

    if target_sources == 3:
        extra_color = rng.choice([c for c in SOURCE_COLORS if c not in {top_color, bottom_color}])
        if not _add_aux_straight(grid, rng, extra_color):
            raise RuntimeError("aux placement failed")

    meta = {
        "mode": "mixed",
        "merge_rect": (hr0, hr1, v1c0, v1c1),
        "top_terminal": top_term,
        "bottom_terminal": bottom_term,
    }
    return grid, meta


def _validate_case(grid: np.ndarray, meta: dict) -> bool:
    out = main(grid)
    mode = meta["mode"]
    if mode == "single":
        return (
            int(out[meta["active_terminal"]]) == int(meta["src_color"])
            and int(out[meta["inactive_terminal"]]) == YELLOW
        )
    if mode == "tie":
        return (
            int(out[meta["top_terminal"]]) == int(meta["src_color"])
            and int(out[meta["bottom_terminal"]]) == int(meta["src_color"])
        )
    if mode == "collision":
        hr0, hr1, c0, c1 = meta["merge_rect"]
        right_term = meta["right_terminal"]
        return (
            np.all(out[hr0 : hr1 + 1, c0 : c1 + 1] == ORANGE)
            and int(out[right_term]) == ORANGE
        )
    assert mode == "mixed"
    hr0, hr1, c0, c1 = meta["merge_rect"]
    return (
        np.all(out[hr0 : hr1 + 1, c0 : c1 + 1] == ORANGE)
        and int(out[meta["top_terminal"]]) == ORANGE
        and int(out[meta["bottom_terminal"]]) == ORANGE
    )


def generate_input() -> np.ndarray:
    rng = random.Random()
    builders = (
        _build_single_mode,
        _build_tie_mode,
        _build_collision_mode,
        _build_mixed_mode,
    )
    for _ in range(400):
        builder = rng.choice(builders)
        target_sources = rng.choice((2, 3))
        try:
            grid, meta = builder(rng, int(target_sources))
            if _validate_case(grid, meta):
                return grid
        except (AssertionError, RuntimeError, ValueError):
            continue
    raise RuntimeError("failed to generate a valid input")
