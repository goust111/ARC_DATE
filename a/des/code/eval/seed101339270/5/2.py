import random
from collections import deque

import numpy as np


BG = 0
RED = 2
GREY = 5
MAROON = 9
PIECE_COLORS = [1, 3, 4, 6, 7, 8]
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}
SIDE_ORDER = ("up", "down", "left", "right")


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(int(r), int(c))])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _crop(g: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return g[r0 : r1 + 1, c0 : c1 + 1].copy()


def _mask_connected(mask: np.ndarray) -> bool:
    comps = _components(mask)
    return len(comps) == 1


def _non_bg_bbox(g: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(g != BG)
    assert coords.size > 0
    return _bbox(coords)


def _find_separator(g: np.ndarray) -> int:
    cols = [int(c) for c in range(g.shape[1]) if np.all(g[:, c] == GREY)]
    assert len(cols) == 1, "expected exactly one grey separator column"
    return cols[0]


def _parse_graph(left: np.ndarray) -> dict[int, dict[str, int]]:
    mask = (left != BG) & (left != RED)
    comps = _components(mask)
    nodes: dict[int, tuple[int, int, int, int]] = {}
    cell_to_color: dict[tuple[int, int], int] = {}
    for comp in comps:
        colors = {int(left[int(r), int(c)]) for r, c in comp}
        assert len(colors) == 1, "graph nodes must be monochrome"
        color = int(next(iter(colors)))
        bbox = _bbox(comp)
        r0, c0, r1, c1 = bbox
        assert comp.shape[0] == 4 and r1 - r0 == 1 and c1 - c0 == 1, "each graph node must be a 2x2 block"
        assert np.all(left[r0 : r1 + 1, c0 : c1 + 1] == color)
        nodes[color] = bbox
        for r, c in comp:
            cell_to_color[(int(r), int(c))] = color

    h, w = left.shape
    adj: dict[int, dict[str, int]] = {color: {} for color in nodes}

    def _follow(sr: int, sc: int, dr: int, dc: int, src: int) -> int | None:
        r, c = int(sr), int(sc)
        if not (0 <= r < h and 0 <= c < w) or int(left[r, c]) != RED:
            return None
        while 0 <= r < h and 0 <= c < w and int(left[r, c]) == RED:
            r += dr
            c += dc
        if 0 <= r < h and 0 <= c < w and (int(r), int(c)) in cell_to_color:
            dst = int(cell_to_color[(int(r), int(c))])
            if dst != src:
                return dst
        return None

    for color, (r0, c0, r1, c1) in nodes.items():
        probes = {
            "right": [((r0, c1 + 1), (0, 1)), ((r1, c1 + 1), (0, 1))],
            "left": [((r0, c0 - 1), (0, -1)), ((r1, c0 - 1), (0, -1))],
            "down": [((r1 + 1, c0), (1, 0)), ((r1 + 1, c1), (1, 0))],
            "up": [((r0 - 1, c0), (-1, 0)), ((r0 - 1, c1), (-1, 0))],
        }
        for side, starts in probes.items():
            hit = None
            for (sr, sc), (dr, dc) in starts:
                hit = _follow(sr, sc, dr, dc, color)
                if hit is not None:
                    break
            if hit is not None:
                adj[color][side] = hit

    for color, nbrs in list(adj.items()):
        for side, other in list(nbrs.items()):
            rev = OPPOSITE[side]
            prior = adj[other].get(rev)
            if prior is None:
                adj[other][rev] = color
            else:
                assert prior == color

    seen = set()
    root = next(iter(nodes))
    q = deque([root])
    seen.add(root)
    while q:
        cur = q.popleft()
        for other in adj[cur].values():
            if other not in seen:
                seen.add(other)
                q.append(other)
    assert seen == set(nodes), "graph must be connected"
    return adj


def _extract_side_anchors(piece: np.ndarray) -> dict[str, list[tuple[int, int]]]:
    occ = piece != BG
    h, w = piece.shape
    out = {side: [] for side in SIDE_ORDER}
    for r in range(h):
        cols = np.flatnonzero(occ[r, :])
        if cols.size == 0:
            continue
        c0 = int(cols[0])
        c1 = int(cols[-1])
        if int(piece[r, c0]) == MAROON:
            out["left"].append((int(r), c0))
        if int(piece[r, c1]) == MAROON:
            out["right"].append((int(r), c1))
    for c in range(w):
        rows = np.flatnonzero(occ[:, c])
        if rows.size == 0:
            continue
        r0 = int(rows[0])
        r1 = int(rows[-1])
        if int(piece[r0, c]) == MAROON:
            out["up"].append((int(c), r0))
        if int(piece[r1, c]) == MAROON:
            out["down"].append((int(c), r1))
    return out


def _parse_pieces(right: np.ndarray) -> dict[int, dict]:
    pieces: dict[int, dict] = {}
    for comp in _components(right != BG):
        bbox = _bbox(comp)
        patch = _crop(right, bbox)
        colors = {int(v) for v in np.unique(patch) if int(v) != BG}
        base_colors = colors - {MAROON}
        assert len(base_colors) == 1, "each fragment must have one non-maroon color"
        base = int(next(iter(base_colors)))
        pieces[base] = {
            "array": patch,
            "anchors": _extract_side_anchors(patch),
        }
    assert pieces, "expected scattered fragments on the right side"
    return pieces


def _constant(vals: list[int]) -> int:
    uniq = {int(v) for v in vals}
    assert len(uniq) == 1
    return int(next(iter(uniq)))


def _relative_offset(piece_a: dict, piece_b: dict, direction: str) -> tuple[int, int]:
    if direction == "right":
        a = piece_a["anchors"]["right"]
        b = piece_b["anchors"]["left"]
        assert a and len(a) == len(b)
        dr = _constant([ra - rb for (ra, _ca), (rb, _cb) in zip(a, b)])
        dc = _constant([ca + 1 - cb for (_ra, ca), (_rb, cb) in zip(a, b)])
        return int(dr), int(dc)
    if direction == "left":
        a = piece_a["anchors"]["left"]
        b = piece_b["anchors"]["right"]
        assert a and len(a) == len(b)
        dr = _constant([ra - rb for (ra, _ca), (rb, _cb) in zip(a, b)])
        dc = _constant([ca - cb - 1 for (_ra, ca), (_rb, cb) in zip(a, b)])
        return int(dr), int(dc)
    if direction == "down":
        a = piece_a["anchors"]["down"]
        b = piece_b["anchors"]["up"]
        assert a and len(a) == len(b)
        dr = _constant([ra + 1 - rb for (_ca, ra), (_cb, rb) in zip(a, b)])
        dc = _constant([ca - cb for (ca, _ra), (cb, _rb) in zip(a, b)])
        return int(dr), int(dc)
    assert direction == "up"
    a = piece_a["anchors"]["up"]
    b = piece_b["anchors"]["down"]
    assert a and len(a) == len(b)
    dr = _constant([ra - rb - 1 for (_ca, ra), (_cb, rb) in zip(a, b)])
    dc = _constant([ca - cb for (ca, _ra), (cb, _rb) in zip(a, b)])
    return int(dr), int(dc)


def _compose(pieces: dict[int, dict], positions: dict[int, tuple[int, int]]) -> np.ndarray:
    min_r = min(int(pos[0]) for pos in positions.values())
    min_c = min(int(pos[1]) for pos in positions.values())
    max_r = max(int(pos[0]) + pieces[color]["array"].shape[0] for color, pos in positions.items())
    max_c = max(int(pos[1]) + pieces[color]["array"].shape[1] for color, pos in positions.items())
    out = np.full((max_r - min_r, max_c - min_c), BG, dtype=int)
    for color, piece in pieces.items():
        patch = piece["array"]
        r0 = int(positions[color][0] - min_r)
        c0 = int(positions[color][1] - min_c)
        for rr in range(patch.shape[0]):
            for cc in range(patch.shape[1]):
                val = int(patch[rr, cc])
                if val == BG:
                    continue
                tr = r0 + rr
                tc = c0 + cc
                cur = int(out[tr, tc])
                if cur == BG:
                    out[tr, tc] = val
                elif cur == MAROON or val == MAROON:
                    out[tr, tc] = MAROON
                else:
                    assert cur == val, "non-maroon overlaps must agree"
    bbox = _non_bg_bbox(out)
    return _crop(out, bbox)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered fragments according to a left-side pipe graph, aligning
    neighboring fragments by the maroon markers on their touching sides.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sep = _find_separator(g)
    graph_region = g[:, :sep]
    piece_region = g[:, sep + 1 :]

    graph = _parse_graph(graph_region)
    pieces = _parse_pieces(piece_region)
    assert set(graph) == set(pieces), "graph node colors must match fragment colors"

    root = next(iter(graph))
    positions: dict[int, tuple[int, int]] = {root: (0, 0)}
    q = deque([root])
    while q:
        cur = q.popleft()
        base_pos = positions[cur]
        for side, other in graph[cur].items():
            dr, dc = _relative_offset(pieces[cur], pieces[other], side)
            target = (int(base_pos[0] + dr), int(base_pos[1] + dc))
            if other not in positions:
                positions[other] = target
                q.append(other)
            else:
                assert positions[other] == target, "graph constraints must be consistent"

    assert len(positions) == len(pieces)
    return _compose(pieces, positions)


GEN_ATTEMPTS = 400
DIM_RANGE = (4, 7)
GRAPH_STEP = 4


def _dir_between(a: tuple[int, int], b: tuple[int, int]) -> str:
    dr = int(b[0] - a[0])
    dc = int(b[1] - a[1])
    if dr == 1 and dc == 0:
        return "down"
    if dr == -1 and dc == 0:
        return "up"
    if dr == 0 and dc == 1:
        return "right"
    if dr == 0 and dc == -1:
        return "left"
    raise AssertionError("graph edges must be axis-adjacent")


def _generate_coords(mode: str, n_nodes: int) -> tuple[list[tuple[int, int]], list[tuple[tuple[int, int], tuple[int, int]]]] | None:
    if mode == "horizontal":
        coords = [(0, c) for c in range(n_nodes)]
        edges = [(coords[i], coords[i + 1]) for i in range(n_nodes - 1)]
        return coords, edges
    if mode == "vertical":
        coords = [(r, 0) for r in range(n_nodes)]
        edges = [(coords[i], coords[i + 1]) for i in range(n_nodes - 1)]
        return coords, edges

    for _ in range(200):
        coords = [(0, 0)]
        coord_set = {(0, 0)}
        edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
        axes: set[str] = set()
        while len(coords) < n_nodes:
            options = []
            for src in coords:
                for side, (dr, dc) in DIRS.items():
                    dst = (int(src[0] + dr), int(src[1] + dc))
                    if dst in coord_set:
                        continue
                    options.append((src, dst, side))
            if not options:
                break
            needed = {"left", "right"} if "h" not in axes else {"up", "down"}
            if len(axes) < 2:
                preferred = [opt for opt in options if opt[2] in needed]
                if preferred and random.random() < 0.8:
                    options = preferred
            src, dst, side = random.choice(options)
            coords.append(dst)
            coord_set.add(dst)
            edges.append((src, dst))
            axes.add("h" if side in {"left", "right"} else "v")
        if len(coords) == n_nodes and axes == {"h", "v"}:
            return coords, edges
    return None


def _build_adjacency(
    coord_to_color: dict[tuple[int, int], int],
    color_to_coord: dict[int, tuple[int, int]],
    edges: list[tuple[tuple[int, int], tuple[int, int]]],
) -> tuple[dict[int, dict[str, int]], list[tuple[int, str, int]]]:
    adj = {int(color): {} for color in color_to_coord}
    edge_specs: list[tuple[int, str, int]] = []
    for ca, cb in edges:
        color_a = int(coord_to_color[ca])
        color_b = int(coord_to_color[cb])
        direction = _dir_between(color_to_coord[color_a], color_to_coord[color_b])
        adj[color_a][direction] = color_b
        adj[color_b][OPPOSITE[direction]] = color_a
        if direction in {"right", "down"}:
            edge_specs.append((color_a, direction, color_b))
        elif direction == "left":
            edge_specs.append((color_b, "right", color_a))
        else:
            edge_specs.append((color_b, "down", color_a))
    return adj, edge_specs


def _overlap_len(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(int(a1), int(b1)) - max(int(a0), int(b0)) + 1)


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (int(a[2]) < int(b[0]) or int(b[2]) < int(a[0]) or int(a[3]) < int(b[1]) or int(b[3]) < int(a[1]))


def _boxes_touch_edge(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    if int(a[3]) + 1 == int(b[1]) or int(b[3]) + 1 == int(a[1]):
        return _overlap_len(a[0], a[2], b[0], b[2]) > 0
    if int(a[2]) + 1 == int(b[0]) or int(b[2]) + 1 == int(a[0]):
        return _overlap_len(a[1], a[3], b[1], b[3]) > 0
    return False


def _place_boxes(adj: dict[int, dict[str, int]], dims: dict[int, tuple[int, int]]) -> dict[int, tuple[int, int]] | None:
    root = next(iter(adj))
    pos = {root: (0, 0)}
    q = deque([root])
    while q:
        cur = q.popleft()
        cr, cc = pos[cur]
        ch, cw = dims[cur]
        for side, other in adj[cur].items():
            if other in pos:
                continue
            oh, ow = dims[other]
            if side == "right":
                lo = cr - oh + 3
                hi = cr + ch - 3
                if lo > hi:
                    return None
                nr = random.randint(lo, hi)
                nc = cc + cw
            elif side == "left":
                lo = cr - oh + 3
                hi = cr + ch - 3
                if lo > hi:
                    return None
                nr = random.randint(lo, hi)
                nc = cc - ow
            elif side == "down":
                lo = cc - ow + 3
                hi = cc + cw - 3
                if lo > hi:
                    return None
                nr = cr + ch
                nc = random.randint(lo, hi)
            else:
                lo = cc - ow + 3
                hi = cc + cw - 3
                if lo > hi:
                    return None
                nr = cr - oh
                nc = random.randint(lo, hi)
            pos[other] = (int(nr), int(nc))
            q.append(other)

    if len(pos) != len(adj):
        return None

    edge_pairs = {frozenset((a, b)) for a, nbrs in adj.items() for b in nbrs.values()}
    boxes = {}
    for color, (r0, c0) in pos.items():
        h, w = dims[color]
        boxes[color] = (int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1))

    colors = list(adj)
    for i, a in enumerate(colors):
        for b in colors[i + 1 :]:
            ba = boxes[a]
            bb = boxes[b]
            if frozenset((a, b)) in edge_pairs:
                if _boxes_overlap(ba, bb):
                    return None
                vertical_contact = (int(ba[3]) + 1 == int(bb[1])) or (int(bb[3]) + 1 == int(ba[1]))
                horizontal_contact = (int(ba[2]) + 1 == int(bb[0])) or (int(bb[2]) + 1 == int(ba[0]))
                if vertical_contact == horizontal_contact:
                    return None
                if vertical_contact and _overlap_len(ba[0], ba[2], bb[0], bb[2]) < 3:
                    return None
                if horizontal_contact and _overlap_len(ba[1], ba[3], bb[1], bb[3]) < 3:
                    return None
            else:
                if _boxes_overlap(ba, bb) or _boxes_touch_edge(ba, bb):
                    return None
    return pos


def _pick_marker_positions(candidates: list[int]) -> list[int]:
    assert candidates
    cap = min(3, len(candidates))
    if cap == 1:
        k = 1
    elif random.random() < 0.75:
        k = random.randint(2, cap)
    else:
        k = 1
    return sorted(random.sample(candidates, k))


def _assign_markers(
    pos: dict[int, tuple[int, int]],
    dims: dict[int, tuple[int, int]],
    edge_specs: list[tuple[int, str, int]],
) -> dict[int, dict[str, list[int]]]:
    markers = {color: {side: [] for side in SIDE_ORDER} for color in pos}
    for a, direction, b in edge_specs:
        ar, ac = pos[a]
        br, bc = pos[b]
        ah, aw = dims[a]
        bh, bw = dims[b]
        if direction == "right":
            start = max(ar, br)
            end = min(ar + ah - 1, br + bh - 1)
            interior = [r for r in range(start, end + 1) if ar < r < ar + ah - 1 and br < r < br + bh - 1]
            if not interior:
                raise AssertionError("horizontal contact must provide non-corner marker positions")
            rows = _pick_marker_positions(interior)
            markers[a]["right"] = [int(r - ar) for r in rows]
            markers[b]["left"] = [int(r - br) for r in rows]
        else:
            start = max(ac, bc)
            end = min(ac + aw - 1, bc + bw - 1)
            interior = [c for c in range(start, end + 1) if ac < c < ac + aw - 1 and bc < c < bc + bw - 1]
            if not interior:
                raise AssertionError("vertical contact must provide non-corner marker positions")
            cols = _pick_marker_positions(interior)
            markers[a]["down"] = [int(c - ac) for c in cols]
            markers[b]["up"] = [int(c - bc) for c in cols]
    return markers


def _try_carve_notch(arr: np.ndarray, side: str) -> None:
    h, w = arr.shape
    test = arr.copy()
    if side in {"up", "down"}:
        if h < 4 or w < 3:
            return
        depth = random.randint(1, min(2, h - 2))
        width = random.randint(1, min(2, w - 2))
        c0 = random.randint(1, w - width - 1)
        if side == "up":
            test[:depth, c0 : c0 + width] = BG
        else:
            test[h - depth :, c0 : c0 + width] = BG
    else:
        if w < 4 or h < 3:
            return
        depth = random.randint(1, min(2, w - 2))
        height = random.randint(1, min(2, h - 2))
        r0 = random.randint(1, h - height - 1)
        if side == "left":
            test[r0 : r0 + height, :depth] = BG
        else:
            test[r0 : r0 + height, w - depth :] = BG
    mask = test != BG
    if mask.sum() >= max(6, (h * w) // 2) and _mask_connected(mask):
        arr[:, :] = test


def _make_piece(
    color: int,
    shape: tuple[int, int],
    side_markers: dict[str, list[int]],
    used_sides: set[str],
) -> np.ndarray:
    h, w = shape
    arr = np.full((h, w), int(color), dtype=int)
    free_sides = [side for side in SIDE_ORDER if side not in used_sides]
    random.shuffle(free_sides)
    for side in free_sides[: random.randint(0, min(2, len(free_sides)))]:
        _try_carve_notch(arr, side)

    for r in side_markers["left"]:
        arr[int(r), 0] = MAROON
    for r in side_markers["right"]:
        arr[int(r), w - 1] = MAROON
    for c in side_markers["up"]:
        arr[0, int(c)] = MAROON
    for c in side_markers["down"]:
        arr[h - 1, int(c)] = MAROON

    assert _mask_connected(arr != BG)
    return arr


def _draw_graph(
    color_to_coord: dict[int, tuple[int, int]],
    edge_specs: list[tuple[int, str, int]],
) -> np.ndarray:
    rows = [int(rc[0]) for rc in color_to_coord.values()]
    cols = [int(rc[1]) for rc in color_to_coord.values()]
    min_r = min(rows)
    min_c = min(cols)
    max_r = max(rows)
    max_c = max(cols)
    h = (max_r - min_r) * GRAPH_STEP + 4
    w = (max_c - min_c) * GRAPH_STEP + 4
    out = np.full((h, w), BG, dtype=int)

    pix = {}
    for color, (gr, gc) in color_to_coord.items():
        r0 = 1 + (int(gr) - min_r) * GRAPH_STEP
        c0 = 1 + (int(gc) - min_c) * GRAPH_STEP
        pix[color] = (int(r0), int(c0))
        out[r0 : r0 + 2, c0 : c0 + 2] = int(color)

    for a, direction, b in edge_specs:
        ar, ac = pix[a]
        br, bc = pix[b]
        if direction == "right":
            row = ar + random.randint(0, 1)
            c0 = ac + 2
            c1 = bc - 1
            out[row, c0 : c1 + 1] = RED
        else:
            col = ac + random.randint(0, 1)
            r0 = ar + 2
            r1 = br - 1
            out[r0 : r1 + 1, col] = RED
    return out


def _scatter_pieces(
    h: int,
    w: int,
    region_c0: int,
    pieces: dict[int, np.ndarray],
) -> np.ndarray | None:
    grid = np.full((h, w), BG, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)
    order = list(pieces)
    random.shuffle(order)
    min_col = region_c0 + 1
    max_col = w - 2
    if min_col > max_col:
        return None

    for color in order:
        patch = pieces[color]
        ph, pw = patch.shape
        if ph > h - 2 or pw > max_col - min_col + 1:
            return None
        placed = False
        for _ in range(300):
            r0 = random.randint(1, h - ph - 1)
            c0 = random.randint(min_col, max_col - pw + 1)
            rr0 = max(0, r0 - 1)
            rr1 = min(h, r0 + ph + 1)
            cc0 = max(region_c0, c0 - 1)
            cc1 = min(w, c0 + pw + 1)
            if bool(reserved[rr0:rr1, cc0:cc1].any()):
                continue
            mask = patch != BG
            sub = grid[r0 : r0 + ph, c0 : c0 + pw]
            if bool((sub[mask] != BG).any()):
                continue
            sub[mask] = patch[mask]
            grid[r0 : r0 + ph, c0 : c0 + pw] = sub
            reserved[rr0:rr1, cc0:cc1] = True
            placed = True
            break
        if not placed:
            return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        mode = random.choice(["horizontal", "vertical", "mixed"])
        n_nodes = random.randint(3, 5)
        graph_data = _generate_coords(mode, n_nodes)
        if graph_data is None:
            continue
        coords, edge_coords = graph_data
        colors = random.sample(PIECE_COLORS, n_nodes)
        random.shuffle(coords)
        coord_to_color = {coord: int(color) for coord, color in zip(coords, colors)}
        color_to_coord = {int(color): coord for coord, color in coord_to_color.items()}
        adj, edge_specs = _build_adjacency(coord_to_color, color_to_coord, edge_coords)

        dims = {color: (random.randint(*DIM_RANGE), random.randint(*DIM_RANGE)) for color in color_to_coord}
        assembled_pos = _place_boxes(adj, dims)
        if assembled_pos is None:
            continue

        side_markers = _assign_markers(assembled_pos, dims, edge_specs)
        pieces = {}
        piece_info = {}
        for color in color_to_coord:
            used_sides = set(adj[color])
            patch = _make_piece(color, dims[color], side_markers[color], used_sides)
            pieces[color] = patch
            piece_info[color] = {"array": patch, "anchors": _extract_side_anchors(patch)}

        expected = _compose(piece_info, assembled_pos)
        left_graph = _draw_graph(color_to_coord, edge_specs)

        total_piece_area = sum(int((patch != BG).sum()) for patch in pieces.values())
        max_piece_h = max(patch.shape[0] for patch in pieces.values())
        max_piece_w = max(patch.shape[1] for patch in pieces.values())
        full_h = max(left_graph.shape[0] + random.randint(0, 5), max_piece_h + 4, 16 + n_nodes)
        right_w = max(max_piece_w + 6, int(np.ceil(total_piece_area / max(6, full_h - 4))) + 8, 14 + 2 * n_nodes)
        left_c0 = random.randint(0, 2)
        sep_gap = random.randint(1, 3)
        sep_col = left_c0 + left_graph.shape[1] + sep_gap
        full_w = sep_col + 1 + right_w + 2
        if full_h <= left_graph.shape[0] or full_w <= sep_col + 3:
            continue

        scattered = _scatter_pieces(full_h, full_w, sep_col + 1, pieces)
        if scattered is None:
            continue
        grid = scattered
        grid[:, sep_col] = GREY
        top = random.randint(0, full_h - left_graph.shape[0])
        grid[top : top + left_graph.shape[0], left_c0 : left_c0 + left_graph.shape[1]] = np.where(
            left_graph != BG,
            left_graph,
            grid[top : top + left_graph.shape[0], left_c0 : left_c0 + left_graph.shape[1]],
        )

        actual = main(grid)
        if np.array_equal(actual, expected):
            return grid

    raise AssertionError("failed to generate a valid puzzle instance")
