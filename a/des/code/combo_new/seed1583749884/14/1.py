import random
from collections import Counter

import numpy as np


BG = 8
LINE = 5
ARROW = 9
NODE_COLORS = (1, 2, 3, 4, 6, 7)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _monochrome_components(grid: np.ndarray, forbidden: set[int]) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in forbidden or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "area": int(arr.shape[0]),
                    "bbox": (
                        int(arr[:, 0].min()),
                        int(arr[:, 1].min()),
                        int(arr[:, 0].max()),
                        int(arr[:, 1].max()),
                    ),
                }
            )
    return comps


def _mask_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _parse_nodes(grid: np.ndarray) -> list[dict]:
    nodes = _monochrome_components(grid, forbidden={BG, LINE, ARROW})
    parsed: list[dict] = []
    covered = np.zeros_like(grid, dtype=bool)
    for comp in nodes:
        r0, c0, r1, c1 = comp["bbox"]
        color = int(comp["color"])
        assert comp["area"] == 4, "every node must be a 2x2 block"
        assert (r1 - r0 + 1, c1 - c0 + 1) == (2, 2), "every node bbox must be 2x2"
        assert bool(np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color)), "node bbox must be solid"
        covered[r0 : r1 + 1, c0 : c1 + 1] = True
        parsed.append(
            {
                "color": color,
                "bbox": (r0, c0, r1, c1),
                "coords": comp["coords"],
                "area": 4,
            }
        )
    assert parsed, "expected at least one node"
    non_special = ~np.isin(grid, (BG, LINE, ARROW))
    assert bool(np.array_equal(covered, non_special)), "all non-special pixels must belong to 2x2 nodes"
    parsed.sort(key=lambda node: (node["bbox"][0], node["bbox"][1]))
    return parsed


def _component_distances(coords: np.ndarray, start: tuple[int, int]) -> dict[tuple[int, int], int]:
    coord_set = {tuple(map(int, xy)) for xy in coords}
    dist = {start: 0}
    stack = [start]
    head = 0
    while head < len(stack):
        r, c = stack[head]
        head += 1
        base = dist[(r, c)]
        for dr, dc in DIR4:
            nxt = (r + dr, c + dc)
            if nxt in coord_set and nxt not in dist:
                dist[nxt] = base + 1
                stack.append(nxt)
    return dist


def _parse_edges(grid: np.ndarray, nodes: list[dict]) -> list[tuple[int, int]]:
    owner = np.full(grid.shape, -1, dtype=int)
    for idx, node in enumerate(nodes):
        owner[node["coords"][:, 0], node["coords"][:, 1]] = idx

    edges: list[tuple[int, int]] = []
    line_mask = np.isin(grid, (LINE, ARROW))
    for coords in _mask_components(line_mask):
        colors = grid[coords[:, 0], coords[:, 1]]
        arrow_coords = coords[colors == ARROW]
        assert len(arrow_coords) == 1, "each connection must contain exactly one arrow pixel"
        arrow_pos = tuple(map(int, arrow_coords[0]))

        contacts: dict[int, set[tuple[int, int]]] = {}
        for r, c in map(tuple, coords):
            rr, cc = int(r), int(c)
            for dr, dc in DIR4:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                    node_idx = int(owner[nr, nc])
                    if node_idx >= 0:
                        contacts.setdefault(node_idx, set()).add((rr, cc))

        assert len(contacts) == 2, "each connection must link exactly two nodes"
        dist = _component_distances(coords, arrow_pos)
        ranked = []
        for node_idx, touch_cells in contacts.items():
            ranked.append((min(dist[cell] for cell in touch_cells), node_idx))
        ranked.sort()
        assert ranked[0][0] != ranked[1][0], "arrow must be closer to exactly one endpoint"
        target = int(ranked[0][1])
        source = int(ranked[1][1])
        edges.append((source, target))

    assert edges, "expected at least one connection"
    assert len(edges) == len(set(edges)), "duplicate connections are not allowed"
    return edges


def _ancestor_sets(node_count: int, edges: list[tuple[int, int]]) -> list[set[int]]:
    incoming: list[list[int]] = [[] for _ in range(node_count)]
    for src, dst in edges:
        incoming[dst].append(src)

    ancestors: list[set[int]] = []
    for node in range(node_count):
        seen: set[int] = set()
        stack = list(incoming[node])
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(incoming[cur])
        assert node not in seen, "network must be acyclic"
        ancestors.append(seen)
    return ancestors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate node colors along arrow-directed links so each downstream node takes the dominant upstream color.

    Input structure:
    - The background is Azure.
    - Several colored 2x2 solid squares act as nodes.
    - Grey orthogonal connections link pairs of nodes, and each connection contains one Maroon arrow pixel.

    Rule:
    - For each connection, compare the arrow's position to the two touched nodes: the farther endpoint is the source and the nearer endpoint is the target.
    - For every node that has at least one upstream node reachable through directed connections, collect the original colors of all such upstream nodes.
    - Sum source area by color and recolor the node to the unique color with the largest total; nodes with no incoming path keep their original color.
    - Leave all Grey connection pixels and Maroon arrow pixels unchanged.

    Key insight:
    - The arrow does not point by shape; its offset within the line encodes direction, and final node colors come from graph-level upstream aggregation rather than local geometry alone.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    nodes = _parse_nodes(g)
    edges = _parse_edges(g, nodes)
    ancestors = _ancestor_sets(len(nodes), edges)

    out = g.copy()
    for idx, node in enumerate(nodes):
        upstream = ancestors[idx]
        if not upstream:
            continue
        totals = Counter()
        for src in upstream:
            totals[int(nodes[src]["color"])] += int(nodes[src]["area"])
        best = max(totals.values())
        winners = [color for color, area in totals.items() if area == best]
        assert len(winners) == 1, "upstream color majority must be unique"
        r0, c0, r1, c1 = node["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(winners[0])
    return out


TEMPLATES = (
    {
        "coords": ((0, 0), (0, 1), (1, 1)),
        "edges": ((0, 1), (1, 2)),
    },
    {
        "coords": ((0, 0), (0, 1), (1, 1), (1, 2)),
        "edges": ((0, 1), (1, 2), (2, 3)),
    },
    {
        "coords": ((0, 0), (0, 1), (1, 0), (1, 1)),
        "edges": ((0, 1), (0, 2), (1, 3), (2, 3)),
    },
    {
        "coords": ((0, 0), (0, 1), (1, 0), (1, 1), (1, 2)),
        "edges": ((0, 1), (0, 2), (1, 3), (2, 3), (3, 4)),
    },
)
SLOT_STEP_OPTIONS = (6, 7)
MARGIN_RANGE = (0, 2)
EXTRA_NODE_RANGE = (0, 2)
MAX_GENERATION_ATTEMPTS = 400


def _transform_point(point: tuple[int, int], mode: int) -> tuple[int, int]:
    r, c = point
    if mode == 0:
        return r, c
    if mode == 1:
        return r, -c
    if mode == 2:
        return -r, c
    if mode == 3:
        return -r, -c
    if mode == 4:
        return c, r
    if mode == 5:
        return c, -r
    if mode == 6:
        return -c, r
    return -c, -r


def _symmetrize_template(template: dict) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    mode = random.randrange(8)
    coords = [_transform_point(point, mode) for point in template["coords"]]
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    norm_coords = [(r - min_r, c - min_c) for r, c in coords]
    return norm_coords, list(template["edges"])


def _add_extra_nodes(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    used = set(coords)
    max_r = max(r for r, _ in coords)
    max_c = max(c for _, c in coords)
    candidates = [(r, c) for r in range(max_r + 2) for c in range(max_c + 2) if (r, c) not in used]
    random.shuffle(candidates)
    extras: list[tuple[int, int]] = []
    for point in candidates[: random.randint(*EXTRA_NODE_RANGE)]:
        used.add(point)
        extras.append(point)
    return coords + extras


def _ancestor_sets_from_edges(node_count: int, edges: list[tuple[int, int]]) -> list[set[int]]:
    return _ancestor_sets(node_count, edges)


def _sample_node_colors(node_count: int, ancestors: list[set[int]]) -> list[int] | None:
    for _ in range(300):
        colors = [random.choice(NODE_COLORS) for _ in range(node_count)]
        if len(set(colors)) < 2:
            continue
        ok = True
        for upstream in ancestors:
            if not upstream:
                continue
            counts = Counter(colors[idx] for idx in upstream)
            top = max(counts.values())
            if sum(value == top for value in counts.values()) != 1:
                ok = False
                break
        if ok:
            return colors
    return None


def _node_pixels(top: int, left: int) -> list[tuple[int, int]]:
    return [(top, left), (top, left + 1), (top + 1, left), (top + 1, left + 1)]


def _edge_cells(src: tuple[int, int], dst: tuple[int, int], row_step: int, col_step: int) -> list[tuple[int, int]]:
    sr, sc = src
    tr, tc = dst
    if sr == tr:
        lane = random.randint(0, 1)
        row = sr + lane
        if sc < tc:
            return [(row, c) for c in range(sc + 2, tc)]
        return [(row, c) for c in range(sc - 1, tc + 1, -1)]
    assert sc == tc
    lane = random.randint(0, 1)
    col = sc + lane
    if sr < tr:
        return [(r, col) for r in range(sr + 2, tr)]
    return [(r, col) for r in range(sr - 1, tr + 1, -1)]


def _render_input(
    coords: list[tuple[int, int]],
    edges: list[tuple[int, int]],
    colors: list[int],
) -> np.ndarray | None:
    row_step = random.choice(SLOT_STEP_OPTIONS)
    col_step = random.choice(SLOT_STEP_OPTIONS)
    top_margin = random.randint(*MARGIN_RANGE)
    left_margin = random.randint(*MARGIN_RANGE)
    bottom_margin = random.randint(*MARGIN_RANGE)
    right_margin = random.randint(*MARGIN_RANGE)

    pixel_pos = {
        idx: (top_margin + r * row_step, left_margin + c * col_step)
        for idx, (r, c) in enumerate(coords)
    }
    max_bottom = max(top for top, _ in pixel_pos.values()) + 1
    max_right = max(left for _, left in pixel_pos.values()) + 1
    h = max_bottom + 1 + bottom_margin
    w = max_right + 1 + right_margin
    if h > 30 or w > 30:
        return None

    grid = np.full((h, w), BG, dtype=int)
    occupied = set()

    for idx, color in enumerate(colors):
        top, left = pixel_pos[idx]
        for cell in _node_pixels(top, left):
            if cell in occupied:
                return None
            occupied.add(cell)
        grid[top : top + 2, left : left + 2] = int(color)

    for src, dst in edges:
        cells = _edge_cells(pixel_pos[src], pixel_pos[dst], row_step, col_step)
        if len(cells) < 4:
            return None
        for cell in cells:
            if cell in occupied:
                return None
        arrow_index = (len(cells) + 1) // 2
        for i, (r, c) in enumerate(cells):
            grid[r, c] = ARROW if i == arrow_index else LINE
            occupied.add((r, c))

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        template = random.choice(TEMPLATES)
        coords, edges = _symmetrize_template(template)
        coords = _add_extra_nodes(coords)

        ancestors = _ancestor_sets_from_edges(len(coords), list(edges))
        colors = _sample_node_colors(len(coords), ancestors)
        if colors is None:
            continue

        grid = _render_input(coords, list(edges), colors)
        if grid is None:
            continue
        try:
            out = main(grid)
        except AssertionError:
            continue
        if np.array_equal(out, grid):
            continue
        return grid
    raise RuntimeError("failed to generate valid input")
