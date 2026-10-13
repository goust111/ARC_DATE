import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
MAROON = 9

PALETTE_PATCH = np.array([[RED, BLUE], [GREEN, YELLOW]], dtype=int)
CYCLE_ORDER = [RED, BLUE, GREEN, YELLOW]
EXTRA_COLORS = [MAGENTA, ORANGE, AZURE, MAROON]

SIDE_TO_DIR = {
    "U": (-1, 0),
    "R": (0, 1),
    "D": (1, 0),
    "L": (0, -1),
}
OPPOSITE = {"U": "D", "R": "L", "D": "U", "L": "R"}
CENTER_PORT = {
    "U": (0, 1),
    "R": (1, 2),
    "D": (2, 1),
    "L": (1, 0),
}

# The visible direction cue used by the generator and parser:
# lines leave the parent from the side center and enter the child at an
# off-center corner on the opposite side.
INCOMING_PORT = {
    ("R", -1): (2, 0),
    ("R", 1): (0, 0),
    ("L", -1): (2, 2),
    ("L", 1): (0, 2),
    ("D", -1): (0, 2),
    ("D", 1): (0, 0),
    ("U", -1): (2, 2),
    ("U", 1): (2, 0),
}
SINK_PORTS_BY_SIDE = {
    "R": {(0, 0), (2, 0)},
    "L": {(0, 2), (2, 2)},
    "D": {(0, 0), (0, 2)},
    "U": {(2, 0), (2, 2)},
}

WORK_SIZE = 44
ROOT_RANGE = (14, 24)
MIN_GAP = 2
MAX_GAP = 8
MAX_ATTEMPTS = 300


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _add(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return a[0] + b[0], a[1] + b[1]


def _components_same_color(grid: np.ndarray, skip: set[int]) -> list[tuple[int, list[tuple[int, int]]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[tuple[int, list[tuple[int, int]]]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in skip or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts = [(r, c)]
            while q:
                cr, cc = q.popleft()
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
                        pts.append((nr, nc))
            comps.append((color, pts))
    return comps


def _find_palette(grid: np.ndarray) -> list[int]:
    h, w = grid.shape
    corners = [
        grid[:2, :2],
        grid[:2, w - 2 : w],
        grid[h - 2 : h, :2],
        grid[h - 2 : h, w - 2 : w],
    ]
    matches = [patch for patch in corners if patch.shape == (2, 2) and np.array_equal(patch, PALETTE_PATCH)]
    assert len(matches) == 1, "expected exactly one 2x2 palette in a corner"
    patch = matches[0]
    return [int(patch[0, 0]), int(patch[0, 1]), int(patch[1, 0]), int(patch[1, 1])]


def _find_blocks(grid: np.ndarray) -> list[dict]:
    blocks: list[dict] = []
    for color, pts in _components_same_color(grid, {BLACK, GREY}):
        if len(pts) != 9:
            continue
        rows = [r for r, _ in pts]
        cols = [c for _, c in pts]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if (r1 - r0, c1 - c0) != (2, 2):
            continue
        cell_set = set(pts)
        if any((r, c) not in cell_set for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)):
            continue
        blocks.append({"color": int(color), "top": (r0, c0)})
    blocks.sort(key=lambda block: block["top"])
    assert len(blocks) >= 2, "expected at least two 3x3 color blocks"
    return blocks


def _segment_cells(src: tuple[int, int], dst: tuple[int, int]) -> list[tuple[int, int]]:
    sr, sc = src
    dr, dc = dst
    cells: list[tuple[int, int]] = []
    if sr == dr:
        step = 1 if dc > sc else -1
        for c in range(sc + step, dc, step):
            cells.append((sr, c))
    elif sc == dc:
        step = 1 if dr > sr else -1
        for r in range(sr + step, dr, step):
            cells.append((r, sc))
    return cells


def _edge_cells(parent_top: tuple[int, int], child_top: tuple[int, int], side: str, skew: int) -> tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]]]:
    src = _add(parent_top, CENTER_PORT[side])
    dst = _add(child_top, INCOMING_PORT[(side, skew)])
    line = _segment_cells(src, dst)
    return src, dst, line


def _trace_line(grid: np.ndarray, start: tuple[int, int], dr: int, dc: int) -> tuple[int, int] | None:
    h, w = grid.shape
    r, c = start[0] + dr, start[1] + dc
    if not (0 <= r < h and 0 <= c < w) or int(grid[r, c]) != GREY:
        return None
    while 0 <= r < h and 0 <= c < w and int(grid[r, c]) == GREY:
        r += dr
        c += dc
    if not (0 <= r < h and 0 <= c < w):
        return None
    return r, c


def _directed_edges(grid: np.ndarray, blocks: list[dict]) -> tuple[list[list[int]], list[int]]:
    cell_to_block: dict[tuple[int, int], int] = {}
    for idx, block in enumerate(blocks):
        r0, c0 = block["top"]
        for r in range(r0, r0 + 3):
            for c in range(c0, c0 + 3):
                cell_to_block[(r, c)] = idx

    adj: list[list[int]] = [[] for _ in blocks]
    indeg = [0] * len(blocks)
    for idx, block in enumerate(blocks):
        r0, c0 = block["top"]
        for side, d in SIDE_TO_DIR.items():
            port = _add((r0, c0), CENTER_PORT[side])
            end = _trace_line(grid, port, d[0], d[1])
            if end is None:
                continue
            assert end in cell_to_block, "line must terminate at another 3x3 block"
            child = cell_to_block[end]
            assert child != idx, "self-loop is not allowed"
            cr, cc = blocks[child]["top"]
            rel = (end[0] - cr, end[1] - cc)
            assert rel in SINK_PORTS_BY_SIDE[side], "line direction must match the child sink port"
            adj[idx].append(child)
            indeg[child] += 1

    return adj, indeg


def _advance_color(color: int, cycle: list[int]) -> int:
    idx = cycle.index(int(color))
    return int(cycle[(idx + 1) % len(cycle)])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate colors through a directed tree of 3x3 blocks using the corner palette
    as a visible cyclic mapping. A parent sends color from its side-center port;
    a child receives from an off-center corner port on the opposite side.
    """
    g = np.asarray(grid, dtype=int)
    cycle = _find_palette(g)
    blocks = _find_blocks(g)
    adj, indeg = _directed_edges(g, blocks)

    n = len(blocks)
    assert sum(indeg) == n - 1, "expected a directed tree with n-1 edges"
    roots = [idx for idx, deg in enumerate(indeg) if deg == 0]
    assert len(roots) == 1, "expected exactly one root block"
    root = roots[0]
    assert all(deg == 1 for idx, deg in enumerate(indeg) if idx != root), "every non-root block must have indegree 1"

    out = g.copy()
    propagated: list[int | None] = [None] * n
    final_colors = [int(block["color"]) for block in blocks]
    root_color = int(blocks[root]["color"])
    assert root_color in cycle, "root color must be in the palette cycle"
    propagated[root] = root_color

    q = deque([root])
    seen = {root}
    while q:
        node = q.popleft()
        cur = int(propagated[node])
        for child in adj[node]:
            assert child not in seen, "graph must be acyclic"
            seen.add(child)
            child_prop = _advance_color(cur, cycle)
            propagated[child] = child_prop
            if int(blocks[child]["color"]) in cycle:
                final_colors[child] = child_prop
            q.append(child)

    assert len(seen) == n, "all blocks must be reachable from the root"

    for block, color in zip(blocks, final_colors):
        r0, c0 = block["top"]
        out[r0 : r0 + 3, c0 : c0 + 3] = int(color)
    return out


def _children_of(parents: list[int]) -> list[list[int]]:
    children = [[] for _ in parents]
    for node in range(1, len(parents)):
        children[parents[node]].append(node)
    return children


def _depths(parents: list[int]) -> list[int]:
    depth = [0] * len(parents)
    for node in range(1, len(parents)):
        depth[node] = depth[parents[node]] + 1
    return depth


def _random_tree(n: int) -> list[int] | None:
    parents = [-1]
    child_counts = [0] * n

    if n >= 4 and random.random() < 0.5:
        parents.extend([0, 1])
        child_counts[0] += 1
        child_counts[1] += 1
        start = 3
    else:
        start = 1

    for node in range(start, n):
        candidates = [
            idx
            for idx in range(node)
            if child_counts[idx] < (4 if idx == 0 else 3)
        ]
        if not candidates:
            return None
        parent = random.choice(candidates)
        parents.append(parent)
        child_counts[parent] += 1

    if len(parents) != n:
        return None
    return parents


def _propagated_by_depth(root_color: int, depths: list[int]) -> list[int]:
    cycle_pos = {color: idx for idx, color in enumerate(CYCLE_ORDER)}
    base = cycle_pos[root_color]
    return [CYCLE_ORDER[(base + d) % 4] for d in depths]


def _assign_colors(parents: list[int]) -> list[int] | None:
    n = len(parents)
    children = _children_of(parents)
    depths = _depths(parents)
    internal_non_root = [idx for idx in range(1, n) if children[idx]]

    for _ in range(200):
        root_color = random.choice(CYCLE_ORDER)
        other_count = n - 1
        max_cycle = min(3, other_count - 1)
        if max_cycle < 1:
            return None
        num_cycle = random.randint(1, max_cycle)
        forced_extra = random.choice(internal_non_root) if internal_non_root and random.random() < 0.75 else None
        cycle_candidates = [idx for idx in range(1, n) if idx != forced_extra]
        if len(cycle_candidates) < num_cycle:
            continue

        cycle_nodes = random.sample(cycle_candidates, num_cycle)
        extra_nodes = [idx for idx in range(1, n) if idx not in cycle_nodes]
        cycle_colors = random.sample([c for c in CYCLE_ORDER if c != root_color], num_cycle)
        extra_colors = random.sample(EXTRA_COLORS, len(extra_nodes))
        random.shuffle(cycle_colors)
        random.shuffle(extra_colors)

        colors = [root_color] + [BLACK] * (n - 1)
        for node, color in zip(cycle_nodes, cycle_colors):
            colors[node] = color
        for node, color in zip(extra_nodes, extra_colors):
            colors[node] = color

        propagated = _propagated_by_depth(root_color, depths)
        if not any(colors[idx] in CYCLE_ORDER and colors[idx] != propagated[idx] for idx in range(1, n)):
            continue
        if not any(colors[idx] not in CYCLE_ORDER for idx in range(1, n)):
            continue
        return colors

    return None


def _root_position() -> tuple[int, int]:
    return random.randint(*ROOT_RANGE), random.randint(*ROOT_RANGE)


def _child_top(parent_top: tuple[int, int], side: str, skew: int, gap: int) -> tuple[int, int]:
    pr, pc = parent_top
    if side == "R":
        return pr + skew, pc + 3 + gap
    if side == "L":
        return pr + skew, pc - 3 - gap
    if side == "D":
        return pr + 3 + gap, pc + skew
    if side == "U":
        return pr - 3 - gap, pc + skew
    raise ValueError(side)


def _block_cells(top: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = top
    return [(r, c) for r in range(r0, r0 + 3) for c in range(c0, c0 + 3)]


def _palette_and_reserved(corner: str) -> tuple[list[tuple[int, int, int]], set[tuple[int, int]]]:
    cells: list[tuple[int, int, int]] = []
    reserved: set[tuple[int, int]] = set()

    if corner == "TL":
        origin = (0, 0)
        rows = range(0, 3)
        cols = range(0, 3)
    elif corner == "TR":
        origin = (0, WORK_SIZE - 2)
        rows = range(0, 3)
        cols = range(WORK_SIZE - 3, WORK_SIZE)
    elif corner == "BL":
        origin = (WORK_SIZE - 2, 0)
        rows = range(WORK_SIZE - 3, WORK_SIZE)
        cols = range(0, 3)
    elif corner == "BR":
        origin = (WORK_SIZE - 2, WORK_SIZE - 2)
        rows = range(WORK_SIZE - 3, WORK_SIZE)
        cols = range(WORK_SIZE - 3, WORK_SIZE)
    else:
        raise ValueError(corner)

    orow, ocol = origin
    for dr in range(2):
        for dc in range(2):
            cells.append((orow + dr, ocol + dc, int(PALETTE_PATCH[dr, dc])))
    for r in rows:
        for c in cols:
            reserved.add((r, c))
    return cells, reserved


def _valid_layout(
    positions: list[tuple[int, int] | None],
    edges: list[tuple[int, int, str, int, int]],
    reserved: set[tuple[int, int]],
) -> bool:
    block_owner: dict[tuple[int, int], int] = {}
    for idx, top in enumerate(positions):
        if top is None:
            continue
        for cell in _block_cells(top):
            r, c = cell
            if not (0 <= r < WORK_SIZE and 0 <= c < WORK_SIZE):
                return False
            if cell in reserved or cell in block_owner:
                return False
            block_owner[cell] = idx

    for cell, owner in block_owner.items():
        for nb in _neighbors4(*cell):
            other = block_owner.get(nb)
            if other is not None and other != owner:
                return False

    line_owner: dict[tuple[int, int], int] = {}
    allowed_touch: dict[int, set[tuple[tuple[int, int], tuple[int, int]]]] = {}
    for edge_id, (parent, child, side, skew, _gap) in enumerate(edges):
        parent_top = positions[parent]
        child_top = positions[child]
        if parent_top is None or child_top is None:
            return False
        src, dst, line = _edge_cells(parent_top, child_top, side, skew)
        if not line:
            return False
        allowed = set()
        allowed.add((line[0], src))
        allowed.add((line[-1], dst))
        allowed_touch[edge_id] = allowed
        for cell in line:
            r, c = cell
            if not (0 <= r < WORK_SIZE and 0 <= c < WORK_SIZE):
                return False
            if cell in reserved or cell in block_owner or cell in line_owner:
                return False
            line_owner[cell] = edge_id

    for cell, edge_id in line_owner.items():
        for nb in _neighbors4(*cell):
            block = block_owner.get(nb)
            if block is not None and (cell, nb) not in allowed_touch[edge_id]:
                return False
            other_edge = line_owner.get(nb)
            if other_edge is not None and other_edge != edge_id:
                return False

    return True


def _place_tree(parents: list[int], corner: str) -> tuple[list[tuple[int, int]], list[tuple[int, int, str, int, int]]] | None:
    _, reserved = _palette_and_reserved(corner)
    positions: list[tuple[int, int] | None] = [None] * len(parents)
    used_sides = [set() for _ in parents]
    edges: list[tuple[int, int, str, int, int]] = []

    for _ in range(40):
        positions[0] = _root_position()
        if _valid_layout(positions, edges, reserved):
            break
    else:
        return None

    for child in range(1, len(parents)):
        parent = parents[child]
        side_options = [side for side in SIDE_TO_DIR if side not in used_sides[parent]]
        random.shuffle(side_options)
        placed = False
        for side in side_options:
            skews = [-1, 1]
            gaps = list(range(MIN_GAP, MAX_GAP + 1))
            random.shuffle(skews)
            random.shuffle(gaps)
            for skew in skews:
                for gap in gaps:
                    positions[child] = _child_top(positions[parent], side, skew, gap)
                    used_sides[parent].add(side)
                    used_sides[child].add(OPPOSITE[side])
                    edges.append((parent, child, side, skew, gap))
                    if _valid_layout(positions, edges, reserved):
                        placed = True
                        break
                    edges.pop()
                    used_sides[parent].remove(side)
                    used_sides[child].remove(OPPOSITE[side])
                    positions[child] = None
                if placed:
                    break
            if placed:
                break
        if not placed:
            return None

    return [tuple(pos) for pos in positions], edges


def _crop_grid(grid: np.ndarray, corner: str) -> np.ndarray:
    pts = np.argwhere(grid != BLACK)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    pad_r = random.randint(1, 3)
    pad_c = random.randint(1, 3)

    if corner == "TL":
        top, left = 0, 0
        bottom = min(WORK_SIZE - 1, int(r1) + pad_r)
        right = min(WORK_SIZE - 1, int(c1) + pad_c)
    elif corner == "TR":
        top, right = 0, WORK_SIZE - 1
        bottom = min(WORK_SIZE - 1, int(r1) + pad_r)
        left = max(0, int(c0) - pad_c)
    elif corner == "BL":
        left, bottom = 0, WORK_SIZE - 1
        top = max(0, int(r0) - pad_r)
        right = min(WORK_SIZE - 1, int(c1) + pad_c)
    elif corner == "BR":
        bottom, right = WORK_SIZE - 1, WORK_SIZE - 1
        top = max(0, int(r0) - pad_r)
        left = max(0, int(c0) - pad_c)
    else:
        raise ValueError(corner)

    return grid[top : bottom + 1, left : right + 1].copy()


def _gen_attempt() -> np.ndarray | None:
    corner = random.choice(["TL", "TR", "BL", "BR"])
    node_count = random.randint(3, 6)
    parents = _random_tree(node_count)
    if parents is None:
        return None
    colors = _assign_colors(parents)
    if colors is None:
        return None
    placed = _place_tree(parents, corner)
    if placed is None:
        return None
    positions, edges = placed

    grid = np.full((WORK_SIZE, WORK_SIZE), BLACK, dtype=int)
    palette_cells, reserved = _palette_and_reserved(corner)
    for r, c, color in palette_cells:
        grid[r, c] = int(color)

    for node, top in enumerate(positions):
        r0, c0 = top
        grid[r0 : r0 + 3, c0 : c0 + 3] = int(colors[node])

    for parent, child, side, skew, _gap in edges:
        _, _, line = _edge_cells(positions[parent], positions[child], side, skew)
        for r, c in line:
            grid[r, c] = GREY

    if not _valid_layout(positions, edges, reserved):
        return None

    cropped = _crop_grid(grid, corner)
    try:
        _ = main(cropped)
    except AssertionError:
        return None
    return cropped


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate a valid input after {MAX_ATTEMPTS} attempts")
