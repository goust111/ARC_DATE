import random
from collections import defaultdict

import numpy as np


BG = 0
GREY = 5
PATH_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DELTAS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

A = frozenset({"a"})
B = frozenset({"b"})
C = frozenset({"c"})
AB = frozenset({"a", "b"})
AC = frozenset({"a", "c"})
BC = frozenset({"b", "c"})
ABC = frozenset({"a", "b", "c"})
SUBSETS = [A, B, C, AB, AC, BC, ABC]
LEGEND_RULES = [
    (A, B, AB),
    (A, C, AC),
    (B, C, BC),
    (AB, C, ABC),
    (AC, B, ABC),
    (BC, A, ABC),
]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in DELTAS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _legend_mapping(grid: np.ndarray) -> dict[tuple[int, int], int]:
    blocks: list[tuple[int, int, int]] = []
    colors = [int(v) for v in np.unique(grid) if int(v) not in (BG, GREY)]
    for color in colors:
        for pts in _components(grid == color):
            if len(pts) != 4:
                continue
            r0, c0, r1, c1 = _bbox(pts)
            if r1 - r0 != 1 or c1 - c0 != 1:
                continue
            if np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color):
                blocks.append((r0, c0, color))

    assert blocks and len(blocks) % 3 == 0
    by_row: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for r0, c0, color in blocks:
        by_row[r0].append((c0, color))

    mapping: dict[tuple[int, int], int] = {}
    used_blocks = 0
    for row in by_row.values():
        row.sort()
        assert len(row) % 3 == 0
        for i in range(0, len(row), 3):
            a = int(row[i][1])
            b = int(row[i + 1][1])
            out = int(row[i + 2][1])
            prev = mapping.get((a, b))
            assert prev in (None, out)
            mapping[(a, b)] = out
            mapping[(b, a)] = out
            used_blocks += 3

    assert used_blocks == len(blocks)
    return mapping


def _find_nodes(grid: np.ndarray) -> list[tuple[int, int]]:
    nodes: list[tuple[int, int]] = []
    for pts in _components(grid == GREY):
        if len(pts) != 9:
            continue
        r0, c0, r1, c1 = _bbox(pts)
        if r1 - r0 != 2 or c1 - c0 != 2:
            continue
        if np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == GREY):
            nodes.append((r0, c0))
    assert nodes
    return nodes


def _incident_colors(grid: np.ndarray, r0: int, c0: int) -> set[int]:
    h, w = grid.shape
    colors: set[int] = set()
    sides = [
        [(r0 - 1, c) for c in range(c0, c0 + 3)],
        [(r0 + 3, c) for c in range(c0, c0 + 3)],
        [(r, c0 - 1) for r in range(r0, r0 + 3)],
        [(r, c0 + 3) for r in range(r0, r0 + 3)],
    ]
    for cells in sides:
        vals = {
            int(grid[r, c])
            for r, c in cells
            if 0 <= r < h and 0 <= c < w and int(grid[r, c]) not in (BG, GREY)
        }
        if vals:
            assert len(vals) == 1
            colors.add(next(iter(vals)))
    return colors


def _mix_colors(colors: set[int], mapping: dict[tuple[int, int], int]) -> int:
    ordered = sorted(int(c) for c in colors)
    cur = ordered[0]
    for nxt in ordered[1:]:
        key = (cur, nxt)
        if key not in mapping:
            key = (nxt, cur)
        assert key in mapping
        cur = int(mapping[key])
    return cur


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    mapping = _legend_mapping(g)
    nodes = _find_nodes(g)

    out = np.zeros_like(g)
    for r0, c0 in nodes:
        colors = _incident_colors(g, r0, c0)
        assert colors
        final_color = next(iter(colors)) if len(colors) == 1 else _mix_colors(colors, mapping)
        out[r0 : r0 + 3, c0 : c0 + 3] = int(final_color)
    return out


def _neighbors(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    out = []
    for dr, dc in DELTAS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            out.append((nr, nc))
    return out


def _canon_edge(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    return (a, b) if a <= b else (b, a)


def _apply_symmetry(cell: tuple[int, int], mode: int) -> tuple[int, int]:
    r, c = cell
    if mode == 0:
        return r, c
    if mode == 1:
        return c, 2 - r
    if mode == 2:
        return 2 - r, 2 - c
    if mode == 3:
        return 2 - c, r
    if mode == 4:
        return r, 2 - c
    if mode == 5:
        return 2 - r, c
    if mode == 6:
        return c, r
    return 2 - c, 2 - r


def _normalize_graph(
    nodes: set[tuple[int, int]], edges: list[tuple[tuple[int, int], tuple[int, int]]]
) -> tuple[
    list[tuple[int, int]],
    list[tuple[tuple[int, int], tuple[int, int]]],
    dict[tuple[int, int], tuple[int, int]],
]:
    mode = random.randrange(8)
    node_map = {cell: _apply_symmetry(cell, mode) for cell in nodes}
    mapped_nodes = set(node_map.values())
    mapped_edges = [(node_map[a], node_map[b]) for a, b in edges]
    min_r = min(r for r, _ in mapped_nodes)
    min_c = min(c for _, c in mapped_nodes)
    norm_nodes = sorted((r - min_r, c - min_c) for r, c in mapped_nodes)
    norm_edges = [_canon_edge((a[0] - min_r, a[1] - min_c), (b[0] - min_r, b[1] - min_c)) for a, b in mapped_edges]
    norm_map = {old: (new[0] - min_r, new[1] - min_c) for old, new in node_map.items()}
    return norm_nodes, sorted(set(norm_edges)), norm_map


def _make_path_graph() -> tuple[list[tuple[int, int]], list[tuple[tuple[int, int], tuple[int, int]]], dict]:
    for _ in range(200):
        path = [(random.randrange(3), random.randrange(3))]
        used = {path[0]}
        target_len = random.randint(4, 6)
        while len(path) < target_len:
            options = []
            for which, endpoint in (("head", path[0]), ("tail", path[-1])):
                for nb in _neighbors(endpoint):
                    if nb not in used:
                        options.append((which, nb))
            if not options:
                break
            which, nb = random.choice(options)
            used.add(nb)
            if which == "head":
                path.insert(0, nb)
            else:
                path.append(nb)
        if len(path) < 4:
            continue

        nodes = set(path)
        edges = [_canon_edge(path[i], path[i + 1]) for i in range(len(path) - 1)]
        edge_bases = {edge: random.choice(["a", "b", "c"]) for edge in edges}
        pivot = random.randint(1, len(path) - 2)
        e1 = _canon_edge(path[pivot - 1], path[pivot])
        e2 = _canon_edge(path[pivot], path[pivot + 1])
        b1, b2 = random.sample(["a", "b", "c"], 2)
        edge_bases[e1] = b1
        edge_bases[e2] = b2

        norm_nodes, norm_edges, norm_map = _normalize_graph(nodes, edges)
        remapped = {}
        for edge, base in edge_bases.items():
            remapped[_canon_edge(norm_map[edge[0]], norm_map[edge[1]])] = base
        return norm_nodes, norm_edges, remapped
    raise RuntimeError("failed to build path graph")


def _make_hub_graph() -> tuple[list[tuple[int, int]], list[tuple[tuple[int, int], tuple[int, int]]], dict]:
    hubs = [(r, c) for r in range(3) for c in range(3) if len(_neighbors((r, c))) >= 3]
    for _ in range(200):
        hub = random.choice(hubs)
        nbrs = _neighbors(hub)
        spoke_count = 3 if len(nbrs) == 3 or random.random() < 0.7 else 4
        spokes = random.sample(nbrs, spoke_count)

        nodes = {hub, *spokes}
        edges = [_canon_edge(hub, nb) for nb in spokes]
        random.shuffle(spokes)
        for nb in spokes[: random.randint(0, min(2, len(spokes)))]:
            dr = nb[0] - hub[0]
            dc = nb[1] - hub[1]
            ext = (nb[0] + dr, nb[1] + dc)
            if 0 <= ext[0] < 3 and 0 <= ext[1] < 3 and ext not in nodes:
                nodes.add(ext)
                edges.append(_canon_edge(nb, ext))

        edge_bases: dict[tuple[tuple[int, int], tuple[int, int]], str] = {}
        hub_edges = [_canon_edge(hub, nb) for nb in spokes]
        hub_colors = ["a", "b", "c"]
        if len(hub_edges) == 4:
            hub_colors.append(random.choice(["a", "b", "c"]))
        random.shuffle(hub_colors)
        for edge, base in zip(hub_edges, hub_colors):
            edge_bases[edge] = base
        for edge in edges:
            edge_bases.setdefault(edge, random.choice(["a", "b", "c"]))

        norm_nodes, norm_edges, norm_map = _normalize_graph(nodes, edges)
        remapped = {}
        for edge, base in edge_bases.items():
            remapped[_canon_edge(norm_map[edge[0]], norm_map[edge[1]])] = base
        return norm_nodes, norm_edges, remapped
    raise RuntimeError("failed to build hub graph")


def _legend_layout(triples: list[tuple[int, int, int]]) -> tuple[int, int, list[tuple[int, int, int, int]]]:
    placements = []
    for idx, triple in enumerate(triples):
        rr = (idx // 2) * 3
        cc = (idx % 2) * 10
        placements.extend(
            [
                (rr, cc, 2, triple[0]),
                (rr, cc + 3, 2, triple[1]),
                (rr, cc + 6, 2, triple[2]),
            ]
        )
    return 8, 18, placements


def _corner_origin(h: int, w: int, box_h: int, box_w: int, corner: str) -> tuple[int, int]:
    if corner == "tl":
        return 1, 1
    if corner == "tr":
        return 1, w - box_w - 1
    if corner == "bl":
        return h - box_h - 1, 1
    return h - box_h - 1, w - box_w - 1


def _rect_intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _expand_rect(rect: tuple[int, int, int, int], pad: int, h: int, w: int) -> tuple[int, int, int, int]:
    return (
        max(0, rect[0] - pad),
        max(0, rect[1] - pad),
        min(h - 1, rect[2] + pad),
        min(w - 1, rect[3] + pad),
    )


def _draw_case(
    h: int,
    w: int,
    nodes: list[tuple[int, int]],
    edges: list[tuple[tuple[int, int], tuple[int, int]]],
    edge_bases: dict[tuple[tuple[int, int], tuple[int, int]], str],
    color_of_subset: dict[frozenset[str], int],
    corner: str,
    stride: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    triples = [(color_of_subset[x], color_of_subset[y], color_of_subset[z]) for x, y, z in LEGEND_RULES]
    random.shuffle(triples)
    legend_h, legend_w, placements = _legend_layout(triples)
    legend_r0, legend_c0 = _corner_origin(h, w, legend_h, legend_w, corner)
    legend_bbox = (legend_r0, legend_c0, legend_r0 + legend_h - 1, legend_c0 + legend_w - 1)

    max_r = max(r for r, _ in nodes)
    max_c = max(c for _, c in nodes)
    net_h = 3 + max_r * stride
    net_w = 3 + max_c * stride

    candidates = []
    blocked = _expand_rect(legend_bbox, 2, h, w)
    for r0 in range(1, h - net_h):
        for c0 in range(1, w - net_w):
            net_bbox = (r0, c0, r0 + net_h - 1, c0 + net_w - 1)
            if not _rect_intersects(net_bbox, blocked):
                candidates.append((r0, c0))
    if not candidates:
        return None
    origin_r, origin_c = random.choice(candidates)

    grid = np.zeros((h, w), dtype=int)
    for rr, cc, size, color in placements:
        r = legend_r0 + rr
        c = legend_c0 + cc
        if np.any(grid[r : r + size, c : c + size] != BG):
            return None
        grid[r : r + size, c : c + size] = int(color)

    pix_of_node: dict[tuple[int, int], tuple[int, int]] = {}
    for nr, nc in nodes:
        r = origin_r + nr * stride
        c = origin_c + nc * stride
        if np.any(grid[r : r + 3, c : c + 3] != BG):
            return None
        grid[r : r + 3, c : c + 3] = GREY
        pix_of_node[(nr, nc)] = (r, c)

    incident: dict[tuple[int, int], set[str]] = defaultdict(set)
    for a, b in edges:
        base = edge_bases[_canon_edge(a, b)]
        color = color_of_subset[frozenset({base})]
        ra, ca = pix_of_node[a]
        rb, cb = pix_of_node[b]
        incident[a].add(base)
        incident[b].add(base)
        if ra == rb:
            row = ra + 1
            c0 = min(ca, cb) + 3
            c1 = max(ca, cb) - 1
            if c0 > c1 or np.any(grid[row, c0 : c1 + 1] != BG):
                return None
            grid[row, c0 : c1 + 1] = int(color)
        else:
            col = ca + 1
            r0 = min(ra, rb) + 3
            r1 = max(ra, rb) - 1
            if r0 > r1 or np.any(grid[r0 : r1 + 1, col] != BG):
                return None
            grid[r0 : r1 + 1, col] = int(color)

    node_color_sizes = [len(vals) for vals in incident.values()]
    if 1 not in node_color_sizes or max(node_color_sizes) < 2:
        return None

    out = np.zeros_like(grid)
    for node, bases in incident.items():
        r, c = pix_of_node[node]
        out[r : r + 3, c : c + 3] = int(color_of_subset[frozenset(bases)])
    return grid, out


def _gen_attempt() -> np.ndarray | None:
    branch = random.choice(["two", "three"])
    if branch == "two":
        nodes, edges, edge_bases = _make_path_graph()
    else:
        nodes, edges, edge_bases = _make_hub_graph()

    chosen = random.sample(PATH_COLORS, 7)
    color_of_subset = {subset: color for subset, color in zip(SUBSETS, chosen)}
    h = random.randint(28, 36)
    w = random.randint(28, 36)
    corner = random.choice(["tl", "tr", "bl", "br"])
    stride = random.randint(5, 7)

    rendered = _draw_case(h, w, nodes, edges, edge_bases, color_of_subset, corner, stride)
    if rendered is None:
        return None
    grid, expected = rendered
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
