from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5
WHITE = 8

PATH_COLORS = (GREY, RED, BLUE)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GEN_ATTEMPTS = 400
STEP_OPTIONS = (3, 4)
COARSE_ROWS_RANGE = (4, 7)
COARSE_COLS_RANGE = (4, 7)
NODE_COUNT_RANGE = (6, 12)
GRID_OFFSET = 2


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in DIR4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _leaf_square_cells(leaf: tuple[int, int], parent: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = leaf
    pr, pc = parent
    dr, dc = r - pr, c - pc
    if (dr, dc) == (0, 1):
        return [(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)]
    if (dr, dc) == (0, -1):
        return [(r, c - 1), (r, c), (r + 1, c - 1), (r + 1, c)]
    if (dr, dc) == (1, 0):
        return [(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)]
    if (dr, dc) == (-1, 0):
        return [(r - 1, c), (r - 1, c + 1), (r, c), (r, c + 1)]
    raise AssertionError("leaf and parent must be 4-neighbors")


def _root_square_cells(root: tuple[int, int], child: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = root
    cr, cc = child
    dr = 0 if cr == r else (1 if cr > r else -1)
    dc = 0 if cc == c else (1 if cc > c else -1)
    if (dr, dc) == (0, 1):
        return [(r, c - 2), (r, c - 1), (r + 1, c - 2), (r + 1, c - 1)]
    if (dr, dc) == (0, -1):
        return [(r, c + 1), (r, c + 2), (r + 1, c + 1), (r + 1, c + 2)]
    if (dr, dc) == (1, 0):
        return [(r - 2, c), (r - 2, c + 1), (r - 1, c), (r - 1, c + 1)]
    if (dr, dc) == (-1, 0):
        return [(r + 1, c), (r + 1, c + 1), (r + 2, c), (r + 2, c + 1)]
    raise AssertionError("root and child must be 4-neighbors")


def _analyze_tree(grid: np.ndarray, require_markers: bool = True) -> dict:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    white_cells = np.argwhere(g == WHITE)
    assert white_cells.shape[0] == 4, "expected one 2x2 white root block"
    r0, c0 = white_cells.min(axis=0)
    r1, c1 = white_cells.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    assert (r1 - r0, c1 - c0) == (1, 1), "white block must be 2x2"
    assert bool((g[r0 : r1 + 1, c0 : c1 + 1] == WHITE).all())
    root_block = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}

    path_mask = np.isin(g, PATH_COLORS)
    path_pts = [tuple(map(int, rc)) for rc in np.argwhere(path_mask)]
    assert path_pts, "expected a path network"

    adj: dict[tuple[int, int], list[tuple[int, int]]] = {}
    degrees: dict[tuple[int, int], int] = {}
    root_candidates: list[tuple[int, int]] = []
    for r, c in path_pts:
        nbs = []
        touches_root = False
        for rr, cc in _neighbors4(r, c, h, w):
            if path_mask[rr, cc]:
                nbs.append((rr, cc))
            if (rr, cc) in root_block:
                touches_root = True
        adj[(r, c)] = nbs
        degrees[(r, c)] = len(nbs)
        if touches_root:
            root_candidates.append((r, c))

    assert len(root_candidates) == 1, "expected one path cell adjacent to the white root block"
    root = root_candidates[0]
    assert degrees[root] == 1, "root entry must be a path endpoint"

    parent: dict[tuple[int, int], tuple[int, int] | None] = {root: None}
    stack = [root]
    visit_order: list[tuple[int, int]] = []
    while stack:
        cur = stack.pop()
        visit_order.append(cur)
        for nb in adj[cur]:
            if nb == parent[cur]:
                continue
            assert nb not in parent, "path network must be a tree"
            parent[nb] = cur
            stack.append(nb)

    assert len(visit_order) == len(path_pts), "path network must be connected"
    edge_count = sum(degrees.values()) // 2
    assert edge_count == len(path_pts) - 1, "path network must be acyclic"

    leaves = sorted([pt for pt in visit_order if degrees[pt] == 1 and pt != root])
    assert leaves, "expected at least one non-root leaf"

    branches = []
    for leaf in leaves:
        path: list[tuple[int, int]] = []
        cur: tuple[int, int] | None = leaf
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        path.reverse()
        markers = [int(g[r, c]) for r, c in path if int(g[r, c]) in (RED, BLUE)]
        if require_markers:
            assert markers, "every root-to-leaf branch must carry at least one marker"
        marker_set = set(markers)
        active = len(marker_set) == 1
        branches.append(
            {
                "leaf": leaf,
                "path": path,
                "markers": markers,
                "active": active and bool(markers),
                "color": int(markers[0]) if markers and active else None,
            }
        )

    return {
        "root": root,
        "root_block": root_block,
        "parent": parent,
        "degrees": degrees,
        "branches": branches,
        "shape": (h, w),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Filter a tree-shaped path network by branch signal consistency and mark surviving terminals.

    Input structure:
    - The background is black.
    - A one-pixel-wide orthogonal tree network is drawn with grey path pixels, and some path pixels are replaced by red or blue signal markers.
    - A unique 2x2 white block marks the root side of the network and touches exactly one path endpoint.
    - Every non-root leaf is a degree-1 path endpoint, and each root-to-leaf path contains at least one red or blue marker.

    Rule:
    - Traverse every complete root-to-leaf branch of the tree.
    - Collect the colors of all red/blue markers that lie on that branch path.
    - Keep a branch iff all of its markers have the same color; otherwise delete every pixel that is used only by closed branches.
    - For each kept branch, place a 2x2 terminal block of that branch color at the leaf end, while preserving the original white root block.

    Key insight:
    - The path network must be interpreted as a tree, so each leaf is judged by the full set of markers on its root-to-leaf path, and shared prefixes survive whenever any active descendant branch still uses them.
    """
    g = np.asarray(grid, dtype=int)
    info = _analyze_tree(g)
    h, w = g.shape

    out = np.full((h, w), BLACK, dtype=int)
    for r, c in info["root_block"]:
        out[r, c] = WHITE

    keep: set[tuple[int, int]] = set()
    for branch in info["branches"]:
        if branch["active"]:
            keep.update(branch["path"])

    for r, c in keep:
        out[r, c] = int(g[r, c])

    for branch in info["branches"]:
        if not branch["active"]:
            continue
        leaf = branch["leaf"]
        parent = info["parent"][leaf]
        assert parent is not None
        square = _leaf_square_cells(leaf, parent)
        for rr, cc in square:
            assert 0 <= rr < h and 0 <= cc < w
            out[rr, cc] = int(branch["color"])

    return out


def _coarse_neighbors(cell: tuple[int, int], rows: int, cols: int) -> list[tuple[int, int]]:
    r, c = cell
    out: list[tuple[int, int]] = []
    for dr, dc in DIR4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < rows and 0 <= cc < cols:
            out.append((rr, cc))
    return out


def _grow_coarse_tree(rows: int, cols: int, node_count: int) -> dict[tuple[int, int], set[tuple[int, int]]] | None:
    start = (random.randrange(rows), random.randrange(cols))
    adj: dict[tuple[int, int], set[tuple[int, int]]] = {start: set()}
    used = {start}

    while len(used) < node_count:
        frontier: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for cell in used:
            for nb in _coarse_neighbors(cell, rows, cols):
                if nb not in used:
                    used_nbs = sum(1 for x in _coarse_neighbors(nb, rows, cols) if x in used)
                    if used_nbs == 1:
                        frontier.append((cell, nb))
        if not frontier:
            return None
        parent, child = random.choice(frontier)
        used.add(child)
        adj.setdefault(parent, set()).add(child)
        adj[child] = {parent}

    return adj


def _render_tree(
    adj: dict[tuple[int, int], set[tuple[int, int]]],
    rows: int,
    cols: int,
    step: int,
    root_cell: tuple[int, int],
) -> np.ndarray | None:
    h = GRID_OFFSET + step * (rows - 1) + 3
    w = GRID_OFFSET + step * (cols - 1) + 3
    grid = np.full((h, w), BLACK, dtype=int)

    coords = {
        cell: (GRID_OFFSET + step * cell[0], GRID_OFFSET + step * cell[1])
        for cell in adj
    }

    for cell, nbs in adj.items():
        r0, c0 = coords[cell]
        for nb in nbs:
            if cell > nb:
                continue
            r1, c1 = coords[nb]
            if r0 == r1:
                lo, hi = sorted((c0, c1))
                grid[r0, lo : hi + 1] = GREY
            else:
                lo, hi = sorted((r0, r1))
                grid[lo : hi + 1, c0] = GREY

    root_actual = coords[root_cell]
    root_neighbors = list(adj[root_cell])
    assert len(root_neighbors) == 1
    child_actual = coords[root_neighbors[0]]
    white_cells = _root_square_cells(root_actual, child_actual)
    for r, c in white_cells:
        if not (0 <= r < h and 0 <= c < w):
            return None
        if grid[r, c] != BLACK:
            return None
    for r, c in white_cells:
        grid[r, c] = WHITE

    return grid


def _branch_exclusive_pixels(branches: list[dict]) -> dict[tuple[int, int], list[tuple[int, int]]]:
    usage = Counter()
    for branch in branches:
        for cell in branch["path"]:
            usage[cell] += 1
    exclusive: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for branch in branches:
        leaf = branch["leaf"]
        exclusive[leaf] = [cell for cell in branch["path"] if usage[cell] == 1]
    return exclusive


def _paint_markers(grid: np.ndarray, branches: list[dict]) -> bool:
    exclusive = _branch_exclusive_pixels(branches)
    n = len(branches)
    if n < 2:
        return False

    active_count = random.randint(1, n - 1)
    active_indices = set(random.sample(range(n), active_count))

    for idx, branch in enumerate(branches):
        slots = list(exclusive[branch["leaf"]])
        if idx in active_indices:
            if not slots:
                return False
            marker_count = random.randint(1, min(3, len(slots)))
            chosen = random.sample(slots, marker_count)
            color = random.choice([RED, BLUE])
            for r, c in chosen:
                grid[r, c] = color
        else:
            if len(slots) < 2:
                return False
            marker_count = random.randint(2, min(4, len(slots)))
            chosen = random.sample(slots, marker_count)
            colors = [RED, BLUE]
            for _ in range(marker_count - 2):
                colors.append(random.choice([RED, BLUE]))
            random.shuffle(colors)
            for (r, c), color in zip(chosen, colors, strict=True):
                grid[r, c] = color

    parsed = _analyze_tree(grid)
    active = [branch for branch in parsed["branches"] if branch["active"]]
    closed = [branch for branch in parsed["branches"] if not branch["active"]]
    return bool(active) and bool(closed)


def _valid_terminal_space(inp: np.ndarray, branches: list[dict], parent: dict[tuple[int, int], tuple[int, int] | None]) -> bool:
    h, w = inp.shape
    for branch in branches:
        leaf = branch["leaf"]
        p = parent[leaf]
        assert p is not None
        square = _leaf_square_cells(leaf, p)
        for r, c in square:
            if not (0 <= r < h and 0 <= c < w):
                return False
        for r, c in square:
            if (r, c) == leaf:
                continue
            if int(inp[r, c]) != BLACK:
                return False
    return True


def _gen_attempt() -> np.ndarray | None:
    step = random.choice(STEP_OPTIONS)
    rows = random.randint(*COARSE_ROWS_RANGE)
    cols = random.randint(*COARSE_COLS_RANGE)
    max_nodes = min(rows * cols, NODE_COUNT_RANGE[1])
    if max_nodes < NODE_COUNT_RANGE[0]:
        return None
    node_count = random.randint(NODE_COUNT_RANGE[0], max_nodes)

    coarse = _grow_coarse_tree(rows, cols, node_count)
    if coarse is None:
        return None

    leaves = [cell for cell, nbs in coarse.items() if len(nbs) == 1]
    if len(leaves) < 3:
        return None

    root_cell = random.choice(leaves)
    grid = _render_tree(coarse, rows, cols, step, root_cell)
    if grid is None:
        return None

    parsed = _analyze_tree(grid, require_markers=False)
    if len(parsed["branches"]) < 2:
        return None
    if not _valid_terminal_space(grid, parsed["branches"], parsed["parent"]):
        return None

    inp = grid.copy()
    if not _paint_markers(inp, parsed["branches"]):
        return None

    parsed_marked = _analyze_tree(inp)
    if not _valid_terminal_space(inp, parsed_marked["branches"], parsed_marked["parent"]):
        return None

    try:
        main(inp)
    except Exception:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid branch-filter tree input")
