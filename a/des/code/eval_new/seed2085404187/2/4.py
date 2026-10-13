import random
from collections import Counter, deque
from dataclasses import dataclass

import numpy as np


BG = 0
PIPE = 5
ORANGE = 7
TOWER_COLORS = (1, 2, 3, 4, 6, 8, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

MARGIN = 3
SPACING = 3
LATTICE_RANGE = (4, 7)
TARGET_NODE_RANGE = (5, 9)
GEN_ATTEMPTS = 400
COLOR_ATTEMPTS = 200


@dataclass(frozen=True)
class Decision:
    kind: str
    color: int | None
    dist: int | None
    distinct_colors: int


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _is_solid_2x2(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> bool:
    if len(cells) != 4:
        return False
    r0, c0, r1, c1 = _bbox(cells)
    if (r1 - r0 + 1, c1 - c0 + 1) != (2, 2):
        return False
    return bool(np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == int(color)))


def _find_pipe_color(grid: np.ndarray, bg: int) -> int:
    candidates: list[tuple[int, int]] = []
    for color in sorted(set(map(int, np.unique(grid))) - {int(bg)}):
        for comp in _components(grid == int(color)):
            if not _is_solid_2x2(grid, comp, int(color)):
                candidates.append((len(comp), int(color)))
    if not candidates:
        raise AssertionError("failed to infer pipe color")
    return max(candidates)[1]


def _tree_centers(nodes: list[int], neighbors: dict[int, list[int]]) -> list[int]:
    if len(nodes) <= 2:
        return list(nodes)
    degree = {node: len(neighbors[node]) for node in nodes}
    leaves = deque(node for node in nodes if degree[node] <= 1)
    remaining = len(nodes)
    while remaining > 2:
        layer = len(leaves)
        for _ in range(layer):
            leaf = leaves.popleft()
            remaining -= 1
            for nxt in neighbors[leaf]:
                degree[nxt] -= 1
                if degree[nxt] == 1:
                    leaves.append(nxt)
    return list(leaves)


def _resolve(candidates: list[tuple[int, int]]) -> Decision:
    if not candidates:
        return Decision("none", None, None, 0)
    best_by_color: dict[int, int] = {}
    for color, dist in candidates:
        prev = best_by_color.get(int(color), -1)
        if int(dist) > prev:
            best_by_color[int(color)] = int(dist)
    best_dist = max(best_by_color.values())
    top_colors = [color for color, dist in best_by_color.items() if int(dist) == int(best_dist)]
    if len(top_colors) == 1:
        return Decision("win", int(top_colors[0]), int(best_dist), len(best_by_color))
    return Decision("tie", ORANGE, None, len(best_by_color))


def _parse_grid(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]], dict[int, list[int]], dict[int, int], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    pipe_color = _find_pipe_color(g, bg)

    pipe_comps = _components(g == int(pipe_color))
    if not pipe_comps:
        raise AssertionError("pipe network missing")
    pipe_cells = max(pipe_comps, key=len)
    pipe_mask = np.zeros_like(g, dtype=bool)
    for r, c in pipe_cells:
        pipe_mask[r, c] = True

    coords = sorted(pipe_cells)
    node_of = {coord: idx for idx, coord in enumerate(coords)}
    neighbors: dict[int, list[int]] = {idx: [] for idx in range(len(coords))}
    for idx, (r, c) in enumerate(coords):
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if (nr, nc) in node_of:
                neighbors[idx].append(node_of[(nr, nc)])

    tower_cells: list[tuple[int, int]] = []
    sources: dict[int, int] = {}
    for color in sorted(set(map(int, np.unique(g))) - {int(bg), int(pipe_color)}):
        for comp in _components(g == int(color)):
            if not _is_solid_2x2(g, comp, int(color)):
                continue
            tower_cells.extend(comp)
            adjacent_nodes: set[int] = set()
            for r, c in comp:
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in node_of:
                        adjacent_nodes.add(node_of[(nr, nc)])
            leaf_nodes = [node for node in adjacent_nodes if len(neighbors[node]) == 1]
            if len(leaf_nodes) == 1:
                source_node = int(leaf_nodes[0])
            elif len(adjacent_nodes) == 1:
                source_node = int(next(iter(adjacent_nodes)))
            else:
                raise AssertionError("tower must touch exactly one pipe endpoint")
            if source_node in sources:
                raise AssertionError("multiple towers mapped to one endpoint")
            sources[source_node] = int(color)

    return g, coords, neighbors, sources, tower_cells


def _solve_with_meta(grid: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    g, coords, neighbors, sources, _tower_cells = _parse_grid(grid)
    node_ids = list(range(len(coords)))
    centers = _tree_centers(node_ids, neighbors)

    parent: dict[int, int] = {}
    children: dict[int, list[int]] = {node: [] for node in node_ids}
    seen: set[int] = set()
    order: list[int] = []
    q = deque()
    for center in centers:
        q.append(int(center))
        parent[int(center)] = -1
        seen.add(int(center))
    while q:
        node = q.popleft()
        order.append(int(node))
        for nxt in neighbors[node]:
            if nxt in seen:
                continue
            seen.add(int(nxt))
            parent[int(nxt)] = int(node)
            children[int(node)].append(int(nxt))
            q.append(int(nxt))

    if len(seen) != len(node_ids):
        raise AssertionError("pipe network must be connected")

    state: dict[int, tuple[int, int] | None] = {}
    node_decisions: dict[int, Decision] = {}
    for node in reversed(order):
        candidates: list[tuple[int, int]] = []
        if node in sources:
            candidates.append((int(sources[node]), 0))
        for child in children[node]:
            child_state = state[child]
            if child_state is not None:
                candidates.append((int(child_state[0]), int(child_state[1]) + 1))
        decision = _resolve(candidates)
        node_decisions[node] = decision
        if decision.kind == "win":
            state[node] = (int(decision.color), int(decision.dist))
        else:
            state[node] = None

    out = g.copy()
    tie_nodes: list[tuple[int, int]] = []
    competitive_win_nodes: list[tuple[int, int]] = []
    for node, coord in enumerate(coords):
        if node in sources:
            out[coord] = int(sources[node])
            continue
        decision = node_decisions[node]
        if decision.kind == "win":
            out[coord] = int(decision.color)
            if decision.distinct_colors >= 2:
                competitive_win_nodes.append(coord)
        elif decision.kind == "tie":
            out[coord] = ORANGE
            tie_nodes.append(coord)

    meta = {
        "tie_nodes": tie_nodes,
        "competitive_win_nodes": competitive_win_nodes,
        "source_count": len(sources),
        "junction_count": sum(1 for node in node_ids if len(neighbors[node]) >= 3),
    }
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor a grey pipe tree by propagating tower colors inward and resolving branch conflicts by path depth.

    Input structure:
    - The background is black.
    - A single-pixel-wide grey orthogonal pipe network forms one connected tree, with T or cross junctions but no loops.
    - Several degree-1 pipe endpoints are each adjacent to a solid 2x2 tower of one non-grey color.

    Rule:
    - Each tower injects its color into the adjacent pipe endpoint, and that branch keeps the tower color while moving inward.
    - At a junction pixel, compare the incoming colors by how many pipe pixels they have traversed from their source towers.
    - If one color arrives from a strictly longer branch, that color wins at the junction and continues farther inward.
    - If the longest incoming distances tie between different colors, the junction pixel becomes orange and no color continues beyond it; upstream pipe remains grey unless another branch colors it.

    Key insight:
    - The pipe is a tree, so the transformation is solved by rooting the network at its center and reducing leaf-to-center signals with a longest-path winner / equal-path orange-stop rule.
    """
    out, _meta = _solve_with_meta(np.asarray(grid, dtype=int))
    return out


def _abstract_neighbors(node: tuple[int, int], h: int, w: int) -> list[tuple[int, int]]:
    r, c = node
    result: list[tuple[int, int]] = []
    for dr, dc in DIR4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            result.append((int(nr), int(nc)))
    return result


def _grow_abstract_tree() -> tuple[dict[tuple[int, int], set[tuple[int, int]]], tuple[int, int], int, int]:
    ah = random.randint(*LATTICE_RANGE)
    aw = random.randint(*LATTICE_RANGE)
    root = (random.randint(1, ah - 2), random.randint(1, aw - 2))
    adj: dict[tuple[int, int], set[tuple[int, int]]] = {root: set()}

    root_neighbors = _abstract_neighbors(root, ah, aw)
    random.shuffle(root_neighbors)
    initial_deg = min(len(root_neighbors), random.choice((3, 3, 4)))
    if initial_deg < 3:
        raise AssertionError("root needs at least three abstract branches")
    for nbr in root_neighbors[:initial_deg]:
        adj[root].add(nbr)
        adj[nbr] = {root}

    target_nodes = random.randint(max(initial_deg + 1, TARGET_NODE_RANGE[0]), TARGET_NODE_RANGE[1])
    frontier = list(adj.keys())
    while len(adj) < target_nodes:
        expandable = [node for node in frontier if any(nbr not in adj for nbr in _abstract_neighbors(node, ah, aw))]
        if not expandable:
            break
        attach = random.choice(expandable)
        choices = [nbr for nbr in _abstract_neighbors(attach, ah, aw) if nbr not in adj]
        if not choices:
            continue
        new_node = random.choice(choices)
        adj.setdefault(attach, set()).add(new_node)
        adj[new_node] = {attach}
        frontier.append(new_node)

    leaf_count = sum(1 for nbrs in adj.values() if len(nbrs) == 1)
    if len(adj) < 5 or leaf_count < 3 or max(len(nbrs) for nbrs in adj.values()) < 3:
        raise AssertionError("abstract tree too simple")
    return adj, root, ah, aw


def _fine_coord(node: tuple[int, int]) -> tuple[int, int]:
    return MARGIN + node[0] * SPACING, MARGIN + node[1] * SPACING


def _rasterize_tree(adj: dict[tuple[int, int], set[tuple[int, int]]], ah: int, aw: int) -> tuple[np.ndarray, dict[tuple[int, int], tuple[int, int]], set[tuple[int, int]]]:
    h = 2 * MARGIN + (ah - 1) * SPACING + 1
    w = 2 * MARGIN + (aw - 1) * SPACING + 1
    grid = np.full((h, w), BG, dtype=int)
    fine_nodes = {node: _fine_coord(node) for node in adj}
    pipe_cells: set[tuple[int, int]] = set()
    for node, nbrs in adj.items():
        r0, c0 = fine_nodes[node]
        pipe_cells.add((int(r0), int(c0)))
        for nbr in nbrs:
            if node > nbr:
                continue
            r1, c1 = fine_nodes[nbr]
            if r0 == r1:
                step = 1 if c1 > c0 else -1
                for cc in range(c0, c1 + step, step):
                    pipe_cells.add((int(r0), int(cc)))
            else:
                step = 1 if r1 > r0 else -1
                for rr in range(r0, r1 + step, step):
                    pipe_cells.add((int(rr), int(c0)))
    for r, c in pipe_cells:
        grid[r, c] = PIPE
    return grid, fine_nodes, pipe_cells


def _tower_options(leaf: tuple[int, int], outward: tuple[int, int]) -> list[list[tuple[int, int]]]:
    r, c = leaf
    if outward == (0, 1):
        starts = [(r - 1, c + 1), (r, c + 1)]
    elif outward == (0, -1):
        starts = [(r - 1, c - 2), (r, c - 2)]
    elif outward == (1, 0):
        starts = [(r + 1, c - 1), (r + 1, c)]
    elif outward == (-1, 0):
        starts = [(r - 2, c - 1), (r - 2, c)]
    else:
        raise AssertionError("invalid direction")
    return [[(sr + dr, sc + dc) for dr in range(2) for dc in range(2)] for sr, sc in starts]


def _place_towers(
    grid: np.ndarray,
    adj: dict[tuple[int, int], set[tuple[int, int]]],
    fine_nodes: dict[tuple[int, int], tuple[int, int]],
    pipe_cells: set[tuple[int, int]],
) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    towers: list[list[tuple[int, int]]] = []
    occupied = set(pipe_cells)
    for node, nbrs in adj.items():
        if len(nbrs) != 1:
            continue
        leaf = fine_nodes[node]
        nbr = fine_nodes[next(iter(nbrs))]
        outward = (leaf[0] - nbr[0], leaf[1] - nbr[1])
        outward = (0 if outward[0] == 0 else outward[0] // abs(outward[0]), 0 if outward[1] == 0 else outward[1] // abs(outward[1]))
        options = _tower_options(leaf, outward)
        random.shuffle(options)
        chosen = None
        for cells in options:
            if any(not (0 <= r < h and 0 <= c < w) for r, c in cells):
                continue
            if any((r, c) in occupied for r, c in cells):
                continue
            touches = set()
            for r, c in cells:
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in pipe_cells:
                        touches.add((int(nr), int(nc)))
            if touches != {leaf}:
                continue
            chosen = cells
            break
        if chosen is None:
            raise AssertionError("failed to place tower")
        towers.append(chosen)
        occupied.update(chosen)
    if len(towers) < 3:
        raise AssertionError("need at least three towers")
    return towers


def _paint_towers(grid: np.ndarray, towers: list[list[tuple[int, int]]], colors: list[int]) -> np.ndarray:
    out = grid.copy()
    for cells, color in zip(towers, colors):
        for r, c in cells:
            out[r, c] = int(color)
    return out


def _sample_colors(count: int) -> list[int]:
    colors = [random.choice(TOWER_COLORS) for _ in range(count)]
    if len(set(colors)) < 2:
        j = random.randrange(count)
        choices = [color for color in TOWER_COLORS if color != colors[j]]
        colors[j] = random.choice(choices)
    return colors


def _gen_attempt(force_tie: bool) -> np.ndarray:
    adj, _root, ah, aw = _grow_abstract_tree()
    grid, fine_nodes, pipe_cells = _rasterize_tree(adj, ah, aw)
    towers = _place_towers(grid, adj, fine_nodes, pipe_cells)

    for _ in range(COLOR_ATTEMPTS):
        tower_colors = _sample_colors(len(towers))
        candidate = _paint_towers(grid, towers, tower_colors)
        out, meta = _solve_with_meta(candidate)
        has_tie = bool(meta["tie_nodes"])
        has_competitive_win = bool(meta["competitive_win_nodes"])
        if force_tie:
            if has_tie:
                return candidate
        else:
            if (not has_tie) and has_competitive_win:
                return candidate
    raise AssertionError("failed to assign colors with required branch coverage")


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        try:
            return _gen_attempt(force_tie=bool(random.getrandbits(1)))
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid pipe-signal task")
