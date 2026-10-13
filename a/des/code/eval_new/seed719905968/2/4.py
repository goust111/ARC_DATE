import heapq
import random
from collections import Counter

import numpy as np


BLACK = 0
GREY = 5
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
VEC_TO_DIR = {(-1, 0): "U", (1, 0): "D", (0, -1): "L", (0, 1): "R"}
DIR_TO_VEC = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}
PERP_DIRS = {"U": ["L", "R"], "D": ["L", "R"], "L": ["U", "D"], "R": ["U", "D"]}
ARC_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 500


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIR4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _sign(x: int) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def _delta_dir(a: tuple[int, int], b: tuple[int, int]) -> str:
    return VEC_TO_DIR[(_sign(b[0] - a[0]), _sign(b[1] - a[1]))]


def _pipe_degrees(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    deg = np.zeros((h, w), dtype=int)
    for r, c in np.argwhere(mask):
        d = 0
        for rr, cc in _neighbors4(int(r), int(c), h, w):
            if mask[rr, cc]:
                d += 1
        deg[r, c] = d
    return deg


def _read_legend(grid: np.ndarray, bg: int, pipe_color: int) -> list[int]:
    h, w = grid.shape
    border = []
    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            val = int(grid[r, c])
            if val != bg and val != pipe_color:
                border.append((r, c, val))
    assert border, "legend not found on border"

    rows = {r for r, _, _ in border}
    cols = {c for _, c, _ in border}
    assert len(rows) == 1 or len(cols) == 1, "legend must be a straight border line"
    if len(rows) == 1:
        border.sort(key=lambda x: (x[1], x[0]))
    else:
        border.sort(key=lambda x: (x[0], x[1]))
    return [val for _, _, val in border]


def _find_source_blocks(grid: np.ndarray, bg: int, pipe_color: int) -> dict[int, tuple[int, int]]:
    h, w = grid.shape
    blocks = {}
    colors = sorted(set(map(int, np.unique(grid))) - {bg, pipe_color})
    for color in colors:
        found = None
        for r in range(h - 1):
            for c in range(w - 1):
                if np.all(grid[r : r + 2, c : c + 2] == color):
                    found = (r, c)
                    break
            if found is not None:
                break
        if found is not None:
            blocks[color] = found
    assert blocks, "no 2x2 source blocks found"
    return blocks


def _source_endpoints(
    grid: np.ndarray, pipe_mask: np.ndarray, blocks: dict[int, tuple[int, int]]
) -> dict[int, tuple[int, int]]:
    h, w = grid.shape
    endpoints = {}
    for color, (r0, c0) in blocks.items():
        seen = set()
        for r in range(r0, r0 + 2):
            for c in range(c0, c0 + 2):
                for rr, cc in _neighbors4(r, c, h, w):
                    if pipe_mask[rr, cc]:
                        seen.add((rr, cc))
        assert len(seen) == 1, "each source block must touch exactly one pipe endpoint"
        endpoints[color] = next(iter(seen))
    return endpoints


def _propagate(
    pipe_mask: np.ndarray, source_endpoints: dict[int, tuple[int, int]], priority: list[int]
) -> dict[tuple[int, int], int]:
    rank = {color: i for i, color in enumerate(priority)}
    h, w = pipe_mask.shape
    best = {}
    pq = []
    for color, (r, c) in source_endpoints.items():
        state = (0, rank[color], color)
        best[(r, c)] = state
        heapq.heappush(pq, (0, rank[color], color, r, c))

    while pq:
        dist, prio, color, r, c = heapq.heappop(pq)
        if best.get((r, c)) != (dist, prio, color):
            continue
        for rr, cc in _neighbors4(r, c, h, w):
            if not pipe_mask[rr, cc]:
                continue
            cand = (dist + 1, prio, color)
            if cand < best.get((rr, cc), (10**9, 10**9, -1)):
                best[(rr, cc)] = cand
                heapq.heappush(pq, (dist + 1, prio, color, rr, cc))
    return {pos: color for pos, (_, _, color) in best.items()}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate colored signals through a grey pipe network using a corner priority legend.

    Input structure:
    - Background is black. A one-pixel-wide orthogonal network of grey pixels forms the pipes.
    - Several degree-1 pipe endpoints each touch exactly one solid 2x2 colored source block.
    - A corner legend on the border is a straight row or column of colored singleton pixels.

    Rule:
    - Read the legend in visual order: left-to-right for a horizontal legend, top-to-bottom for a vertical legend.
    - Treat that order as the signal priority from high to low.
    - Each source color spreads through the grey pipe network from its touching endpoint.
    - If two colors reach the same pipe location equally, the higher-priority legend color wins that location.
    - Recolor every grey pipe pixel with its winning source color; keep the source blocks and legend unchanged.

    Key insight:
    The pipe color alone only gives the network geometry; the border legend is the visible lookup table
    that resolves otherwise ambiguous collisions between equally plausible signal flows.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(Counter(map(int, g.flat)).most_common(1)[0][0])

    non_bg_counts = Counter(map(int, g[g != bg].flat))
    assert non_bg_counts, "grid must contain non-background pixels"
    pipe_color = int(non_bg_counts.most_common(1)[0][0])
    pipe_mask = g == pipe_color

    priority = _read_legend(g, bg, pipe_color)
    blocks = _find_source_blocks(g, bg, pipe_color)
    assert set(blocks) == set(priority), "legend colors and source colors must match"

    endpoints = _source_endpoints(g, pipe_mask, blocks)
    coloring = _propagate(pipe_mask, endpoints, priority)
    assert int(pipe_mask.sum()) == len(coloring), "all pipe pixels must be reachable from sources"

    out = g.copy()
    for (r, c), color in coloring.items():
        out[r, c] = color
    return out


def _sample_canvas():
    return random.randint(20, 29), random.randint(20, 29)


def _legend_cells(h: int, w: int, colors: list[int]):
    orientation = random.choice(["horizontal", "vertical"])
    corner = random.choice(["tl", "tr", "bl", "br"])
    n = len(colors)
    if orientation == "horizontal":
        row = 0 if corner.startswith("t") else h - 1
        if corner.endswith("l"):
            start_c = 0
        else:
            start_c = w - n
        cells = [(row, start_c + i) for i in range(n)]
    else:
        col = 0 if corner.endswith("l") else w - 1
        if corner.startswith("t"):
            start_r = 0
        else:
            start_r = h - n
        cells = [(start_r + i, col) for i in range(n)]
    return orientation, corner, cells


def _lattice_positions(h: int, w: int, step: int, margin: int):
    rows = list(range(margin, h - margin, step))
    cols = list(range(margin, w - margin, step))
    return rows, cols


def _choose_root(rows: list[int], cols: list[int]):
    r = random.choice(rows[1:-1] if len(rows) > 2 else rows)
    c = random.choice(cols[1:-1] if len(cols) > 2 else cols)
    return (r, c)


def _step_coord(pos: tuple[int, int], direction: str, step: int):
    dr, dc = DIR_TO_VEC[direction]
    return pos[0] + dr * step, pos[1] + dc * step


def _inside_lattice(pos: tuple[int, int], rows: list[int], cols: list[int]) -> bool:
    return pos[0] in set(rows) and pos[1] in set(cols)


def _node_graph(rows: list[int], cols: list[int], step: int):
    rowset = set(rows)
    colset = set(cols)

    def valid(pos):
        return pos[0] in rowset and pos[1] in colset

    return valid


def _add_edge(adjacency: dict, a: tuple[int, int], b: tuple[int, int]):
    adjacency.setdefault(a, set()).add(b)
    adjacency.setdefault(b, set()).add(a)


def _build_tree(h: int, w: int):
    step = random.choice([3, 4])
    margin = 4
    rows, cols = _lattice_positions(h, w, step, margin)
    if len(rows) < 3 or len(cols) < 3:
        return None
    valid = _node_graph(rows, cols, step)

    root = _choose_root(rows, cols)
    hub_dirs = random.sample(["U", "D", "L", "R"], k=random.choice([3, 4]))
    adjacency = {root: set()}
    for d in hub_dirs:
        nxt = _step_coord(root, d, step)
        if not valid(nxt):
            return None
        _add_edge(adjacency, root, nxt)

    extra_steps = random.randint(1, 4)
    for _ in range(extra_steps):
        candidates = [node for node, nbrs in adjacency.items() if len(nbrs) < 4]
        random.shuffle(candidates)
        placed = False
        for node in candidates:
            used_dirs = {_delta_dir(node, nbr) for nbr in adjacency[node]}
            dirs = ["U", "D", "L", "R"]
            random.shuffle(dirs)
            for d in dirs:
                if d in used_dirs:
                    continue
                nxt = _step_coord(node, d, step)
                if not valid(nxt) or nxt in adjacency:
                    continue
                _add_edge(adjacency, node, nxt)
                placed = True
                break
            if placed:
                break

    degrees = {node: len(nbrs) for node, nbrs in adjacency.items()}
    leaves = [node for node, deg in degrees.items() if deg == 1]
    junctions = [node for node, deg in degrees.items() if deg >= 3]
    if len(leaves) < 3 or not junctions:
        return None
    return adjacency, step


def _draw_tree(h: int, w: int, adjacency: dict[tuple[int, int], set[tuple[int, int]]]):
    grid = np.full((h, w), BLACK, dtype=int)
    for a, nbrs in adjacency.items():
        ar, ac = a
        for b in nbrs:
            br, bc = b
            if (br, bc) < (ar, ac):
                continue
            if ar == br:
                c0, c1 = sorted((ac, bc))
                grid[ar, c0 : c1 + 1] = GREY
            else:
                r0, r1 = sorted((ar, br))
                grid[r0 : r1 + 1, ac] = GREY
    return grid


def _endpoint_outward_direction(node: tuple[int, int], parent: tuple[int, int]) -> str:
    return _delta_dir(parent, node)


def _block_cells(endpoint: tuple[int, int], outward: str, variant: int):
    r, c = endpoint
    if outward == "U":
        r0 = r - 2
        c0 = c if variant == 0 else c - 1
    elif outward == "D":
        r0 = r + 1
        c0 = c if variant == 0 else c - 1
    elif outward == "L":
        r0 = r if variant == 0 else r - 1
        c0 = c - 2
    else:
        r0 = r if variant == 0 else r - 1
        c0 = c + 1
    return [(r0 + dr, c0 + dc) for dr in range(2) for dc in range(2)]


def _place_sources(
    grid: np.ndarray, adjacency: dict[tuple[int, int], set[tuple[int, int]]], colors: list[int]
):
    h, w = grid.shape
    degrees = {node: len(nbrs) for node, nbrs in adjacency.items()}
    leaves = [node for node, deg in degrees.items() if deg == 1]
    if len(leaves) != len(colors):
        return None

    random.shuffle(leaves)
    placements = {}
    occupied = set(map(tuple, np.argwhere(grid != BLACK)))
    for color, node in zip(colors, leaves):
        parent = next(iter(adjacency[node]))
        outward = _endpoint_outward_direction(node, parent)
        variants = [0, 1]
        random.shuffle(variants)
        placed = False
        for variant in variants:
            cells = _block_cells(node, outward, variant)
            if any(not (0 <= r < h and 0 <= c < w) for r, c in cells):
                continue
            if any((r, c) in occupied for r, c in cells):
                continue
            adj_pipe = set()
            for r, c in cells:
                for rr, cc in _neighbors4(r, c, h, w):
                    if grid[rr, cc] == GREY:
                        adj_pipe.add((rr, cc))
            if adj_pipe != {node}:
                continue
            placements[color] = cells
            occupied.update(cells)
            placed = True
            break
        if not placed:
            return None
    for color, cells in placements.items():
        for r, c in cells:
            grid[r, c] = color
    return placements


def _place_legend(grid: np.ndarray, colors: list[int]):
    h, w = grid.shape
    orientation, _corner, cells = _legend_cells(h, w, colors)
    for (r, c), color in zip(cells, colors):
        if grid[r, c] != BLACK:
            return None
        grid[r, c] = color
    return orientation, cells


def _priority_sensitive(grid: np.ndarray, colors: list[int]) -> bool:
    out_a = main(grid)
    flipped = list(reversed(colors))
    legend = _read_legend(grid, BLACK, GREY)
    if legend == flipped:
        return False
    g2 = grid.copy()
    border = []
    h, w = g2.shape
    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            if g2[r, c] != BLACK and g2[r, c] != GREY:
                border.append((r, c))
    border.sort()
    if len({r for r, _ in border}) == 1:
        border.sort(key=lambda x: x[1])
    else:
        border.sort(key=lambda x: x[0])
    for (r, c), color in zip(border, flipped):
        g2[r, c] = color
    out_b = main(g2)
    pipe_mask = grid == GREY
    return bool(np.any(out_a[pipe_mask] != out_b[pipe_mask]))


def _valid_structure(grid: np.ndarray, colors: list[int]) -> bool:
    pipe_mask = grid == GREY
    deg = _pipe_degrees(pipe_mask)
    if not np.any(deg >= 3):
        return False
    if sum(np.count_nonzero(grid == color) for color in colors) != 5 * len(colors):
        return False
    try:
        out = main(grid)
    except Exception:
        return False
    if out.shape != grid.shape:
        return False
    return True


def _gen_attempt():
    h, w = _sample_canvas()
    tree = _build_tree(h, w)
    if tree is None:
        return None
    adjacency, _step = tree
    leaves = [node for node, nbrs in adjacency.items() if len(nbrs) == 1]
    n_sources = len(leaves)
    if n_sources < 3 or n_sources > 5:
        return None

    colors = random.sample(ARC_COLORS, n_sources)
    priority = colors[:]
    random.shuffle(priority)

    grid = _draw_tree(h, w, adjacency)
    if _place_sources(grid, adjacency, colors) is None:
        return None
    if _place_legend(grid, priority) is None:
        return None
    if not _valid_structure(grid, colors):
        return None
    if not _priority_sensitive(grid, priority):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pipe-priority ARC input")
