from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


BG = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _order_key(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = _bbox(coords)
    return (r0 + r1, c0 + c1, r0, c0)


def _mix_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a <= b else (b, a)


def _parse_rules(grid: np.ndarray) -> dict[tuple[int, int], int]:
    h, w = grid.shape
    assert h >= 3, "expected a card at the bottom"
    top = grid[h - 2]
    bottom = grid[h - 1]
    rules: dict[tuple[int, int], int] = {}
    for c in range(0, w - 1, 3):
        a = int(top[c])
        b = int(top[c + 1])
        result = int(bottom[c + 1])
        if a == BG and b == BG and result == BG:
            continue
        assert a != BG and b != BG and result != BG, "malformed card entry"
        key = _mix_key(a, b)
        if key in rules:
            assert rules[key] == result, "conflicting card rules"
        else:
            rules[key] = result
    assert rules, "expected at least one color-mix rule"
    return rules


def _lookup_mix(rules: dict[tuple[int, int], int], a: int, b: int) -> int:
    key = _mix_key(int(a), int(b))
    assert key in rules, f"missing mix rule for colors {a} and {b}"
    return int(rules[key])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the bottom color card, recolor each grey connector by the mixed color
    of the two clusters it joins, and recolor any multi-neighbor cluster by
    sequentially mixing it with its neighbors in row-major position order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h >= 3 and w >= 3

    rules = _parse_rules(g)
    scene = g[:-2].copy()
    sh, sw = scene.shape

    cluster_label = np.zeros((sh, sw), dtype=int)
    clusters: dict[int, dict[str, object]] = {}
    next_id = 1
    for color in sorted(int(x) for x in np.unique(scene) if x not in (BG, GREY)):
        for comp in _components(scene == color):
            area = int(comp.shape[0])
            assert 2 <= area <= 6, "cluster area must stay within the described bounds"
            cid = next_id
            next_id += 1
            cluster_label[comp[:, 0], comp[:, 1]] = cid
            clusters[cid] = {
                "color": int(color),
                "coords": comp,
                "order": _order_key(comp),
            }

    out = scene.copy()
    neighbors: dict[int, set[int]] = defaultdict(set)
    for comp in _components(scene == GREY):
        rows = np.unique(comp[:, 0])
        cols = np.unique(comp[:, 1])
        assert rows.size == 1 or cols.size == 1, "grey connectors must be straight"
        touched: set[int] = set()
        for rr, cc in comp:
            for dr, dc in NEIGHBORS4:
                nr, nc = int(rr + dr), int(cc + dc)
                if 0 <= nr < sh and 0 <= nc < sw:
                    cid = int(cluster_label[nr, nc])
                    if cid:
                        touched.add(cid)
        assert len(touched) == 2, "each grey connector must join exactly two clusters"
        a_id, b_id = sorted(touched)
        a_color = int(clusters[a_id]["color"])
        b_color = int(clusters[b_id]["color"])
        line_color = _lookup_mix(rules, a_color, b_color)
        out[comp[:, 0], comp[:, 1]] = line_color
        neighbors[a_id].add(b_id)
        neighbors[b_id].add(a_id)

    for cid, info in clusters.items():
        linked = sorted(neighbors.get(cid, set()), key=lambda nid: clusters[nid]["order"])
        if len(linked) <= 1:
            continue
        current = int(info["color"])
        for nid in linked:
            current = _lookup_mix(rules, current, int(clusters[nid]["color"]))
        coords = np.asarray(info["coords"], dtype=int)
        out[coords[:, 0], coords[:, 1]] = current

    return out


TEMPLATES_NO_MULTI = [
    {
        "nodes": [(0, 0), (0, 2), (2, 1)],
        "edges": [(0, 1)],
    },
    {
        "nodes": [(0, 0), (0, 1), (2, 0), (2, 1)],
        "edges": [(0, 1), (2, 3)],
    },
]


TEMPLATES_MULTI = [
    {
        "nodes": [(0, 1), (1, 1), (1, 2)],
        "edges": [(0, 1), (1, 2)],
    },
    {
        "nodes": [(0, 0), (0, 1), (0, 2), (0, 3)],
        "edges": [(0, 1), (1, 2), (2, 3)],
    },
    {
        "nodes": [(0, 1), (1, 1), (1, 0), (1, 2)],
        "edges": [(1, 0), (1, 2), (1, 3)],
    },
    {
        "nodes": [(0, 1), (1, 1), (2, 1), (1, 0), (1, 2)],
        "edges": [(1, 0), (1, 2), (1, 3), (1, 4)],
    },
]


def _transform_points(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    pts = [(int(r), int(c)) for r, c in points]
    if random.random() < 0.5:
        pts = [(r, -c) for r, c in pts]
    turns = random.randrange(4)
    for _ in range(turns):
        pts = [(c, -r) for r, c in pts]
    min_r = min(r for r, _ in pts)
    min_c = min(c for _, c in pts)
    return [(r - min_r, c - min_c) for r, c in pts]


def _random_shape(area: int) -> np.ndarray:
    assert 2 <= area <= 6
    cells = {(0, 0)}
    while len(cells) < area:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in NEIGHBORS4:
                nr, nc = r + dr, c + dc
                if abs(nr) <= 1 and abs(nc) <= 1 and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        assert frontier, "shape frontier unexpectedly empty"
        cells.add(random.choice(frontier))
    return np.array(sorted(cells), dtype=int)


def _line_cells(coords_a: np.ndarray, coords_b: np.ndarray, anchor_a: tuple[int, int], anchor_b: tuple[int, int]) -> np.ndarray | None:
    ar, ac = anchor_a
    br, bc = anchor_b
    if ar == br:
        row = int(ar)
        if ac < bc:
            left = coords_a
            right = coords_b
        else:
            left = coords_b
            right = coords_a
        left_edge = int(left[left[:, 0] == row, 1].max())
        right_edge = int(right[right[:, 0] == row, 1].min())
        if right_edge - left_edge <= 1:
            return None
        cols = np.arange(left_edge + 1, right_edge, dtype=int)
        rows = np.full(cols.shape, row, dtype=int)
        return np.column_stack([rows, cols])
    if ac == bc:
        col = int(ac)
        if ar < br:
            top = coords_a
            bottom = coords_b
        else:
            top = coords_b
            bottom = coords_a
        top_edge = int(top[top[:, 1] == col, 0].max())
        bottom_edge = int(bottom[bottom[:, 1] == col, 0].min())
        if bottom_edge - top_edge <= 1:
            return None
        rows = np.arange(top_edge + 1, bottom_edge, dtype=int)
        cols = np.full(rows.shape, col, dtype=int)
        return np.column_stack([rows, cols])
    return None


def _make_mixer(palette: list[int]) -> tuple[dict[int, int], callable]:
    ordered = palette[:]
    random.shuffle(ordered)
    index = {color: i for i, color in enumerate(ordered)}
    shift = random.randrange(len(ordered))

    def mix(a: int, b: int) -> int:
        ia = index[int(a)]
        ib = index[int(b)]
        return int(ordered[(ia + ib + shift) % len(ordered)])

    return index, mix


def _scene_rules(
    node_colors: list[int],
    node_coords: list[np.ndarray],
    edges: list[tuple[int, int]],
    mix_fn,
) -> list[tuple[int, int]]:
    needed: set[tuple[int, int]] = set()
    adjacency: dict[int, set[int]] = defaultdict(set)
    for a, b in edges:
        needed.add(_mix_key(node_colors[a], node_colors[b]))
        adjacency[a].add(b)
        adjacency[b].add(a)
    order_keys = [_order_key(coords) for coords in node_coords]
    for idx, color in enumerate(node_colors):
        linked = sorted(adjacency.get(idx, set()), key=lambda nid: order_keys[nid])
        if len(linked) <= 1:
            continue
        current = int(color)
        for nid in linked:
            needed.add(_mix_key(current, node_colors[nid]))
            current = int(mix_fn(current, node_colors[nid]))
    return list(needed)


def _gen_attempt() -> np.ndarray | None:
    if random.random() < 0.5:
        template = random.choice(TEMPLATES_NO_MULTI)
    else:
        template = random.choice(TEMPLATES_MULTI)
    lattice = _transform_points(template["nodes"])
    edges = list(template["edges"])
    node_count = len(lattice)

    palette_size = random.randint(max(4, node_count), min(6, len(PALETTE)))
    palette = random.sample(PALETTE, palette_size)
    node_colors = random.sample(palette, node_count)
    _index, mix_fn = _make_mixer(palette)

    row_step = random.randint(4, 6)
    col_step = random.randint(4, 6)
    top_margin = random.randint(2, 4)
    left_margin = random.randint(2, 4)
    bottom_margin = random.randint(2, 4)
    right_margin = random.randint(2, 4)
    max_lr = max(r for r, _ in lattice)
    max_lc = max(c for _, c in lattice)
    scene_h = top_margin + max_lr * row_step + bottom_margin + 3
    scene_w = left_margin + max_lc * col_step + right_margin + 3

    anchors: list[tuple[int, int]] = []
    node_shapes: list[np.ndarray] = []
    occupancy = np.zeros((scene_h, scene_w), dtype=int)
    for color, (lr, lc) in zip(node_colors, lattice):
        anchor = (top_margin + lr * row_step, left_margin + lc * col_step)
        shape = _random_shape(random.randint(2, 6))
        coords = shape + np.array(anchor, dtype=int)
        if np.any(coords[:, 0] < 0) or np.any(coords[:, 0] >= scene_h):
            return None
        if np.any(coords[:, 1] < 0) or np.any(coords[:, 1] >= scene_w):
            return None
        if np.any(occupancy[coords[:, 0], coords[:, 1]] != BG):
            return None
        occupancy[coords[:, 0], coords[:, 1]] = int(color)
        anchors.append(anchor)
        node_shapes.append(coords)

    for a, b in edges:
        cells = _line_cells(node_shapes[a], node_shapes[b], anchors[a], anchors[b])
        if cells is None or cells.size == 0:
            return None
        if np.any(occupancy[cells[:, 0], cells[:, 1]] != BG):
            return None
        occupancy[cells[:, 0], cells[:, 1]] = GREY

    needed_rules = _scene_rules(node_colors, node_shapes, edges, mix_fn)
    random.shuffle(needed_rules)

    extra_card_blocks = random.randint(0, 2)
    card_width = 3 * len(needed_rules)
    total_w = max(scene_w, card_width + 3 * extra_card_blocks)
    card_slots = [c for c in range(0, total_w - card_width + 1, 3)]
    card_offset = random.choice(card_slots)
    scene_offset = random.randint(0, total_w - scene_w)
    top_pad = random.randint(0, 2)
    final = np.zeros((scene_h + top_pad + 2, total_w), dtype=int)
    final[top_pad : top_pad + scene_h, scene_offset : scene_offset + scene_w] = occupancy

    for idx, (a, b) in enumerate(needed_rules):
        col = card_offset + 3 * idx
        final[-2, col] = int(a)
        final[-2, col + 1] = int(b)
        final[-1, col + 1] = int(mix_fn(a, b))

    try:
        out = main(final)
    except AssertionError:
        return None
    if out.shape[0] != final.shape[0] - 2:
        return None
    if out.shape[1] != final.shape[1]:
        return None
    return final


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input after 400 attempts")
