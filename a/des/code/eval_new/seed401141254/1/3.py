from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import random

import numpy as np


LINE_COLOR = 5
LEAF_TERMINAL_ROW = 2
LEAF_X_STEP = 4
LEAF_COUNT_RANGE = (3, 7)
NODE_GAP_RANGE = (2, 4)
TRUNK_RANGE = (2, 4)
MAX_TREE_DEPTH = 4
GEN_RETRIES = 500
COLOR_POOL = [0, 1, 2, 3, 4, 6, 7, 8, 9]


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            cells: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for rr, cc in _neighbors4(r, c):
                    if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            out.append(np.array(cells, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _is_solid_2x2(coords: np.ndarray) -> bool:
    if int(coords.shape[0]) != 4:
        return False
    r0, c0, r1, c1 = _bbox(coords)
    return (r1 - r0 + 1, c1 - c0 + 1) == (2, 2)


def _pick_subtree_winner(
    counts: Counter[int],
    leftmost: dict[int, tuple[int, int]],
    children_info: list[dict[str, object]],
) -> int:
    best = max(counts.values())
    tied = [color for color, value in counts.items() if int(value) == int(best)]
    if len(tied) == 1:
        return int(tied[0])

    tied_set = set(map(int, tied))
    for child in children_info:
        child_counts_full = child["counts"]
        assert isinstance(child_counts_full, Counter)
        child_counts = Counter(
            {
                int(color): int(child_counts_full[color])
                for color in tied_set
                if int(child_counts_full[color]) > 0
            }
        )
        if not child_counts:
            continue
        return int(
            min(
                [
                    color
                    for color, value in child_counts.items()
                    if int(value) == int(max(child_counts.values()))
                ],
                key=lambda color: tuple(leftmost[int(color)]),
            )
        )

    raise AssertionError("failed to resolve tied subtree winner")


def _find_leaf_blocks(
    grid: np.ndarray, bg: int
) -> tuple[list[dict[str, object]], np.ndarray]:
    leaf_mask = np.zeros(grid.shape, dtype=bool)
    leaves: list[dict[str, object]] = []
    for color0 in np.unique(grid):
        color = int(color0)
        if color == bg:
            continue
        for coords in _components(grid == color):
            if not _is_solid_2x2(coords):
                continue
            leaf_mask[coords[:, 0], coords[:, 1]] = True
            leaves.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                }
            )
    assert leaves, "expected at least one 2x2 leaf block"
    return leaves, leaf_mask


def _edge_path(
    child: tuple[int, int], parent: tuple[int, int]
) -> list[tuple[int, int]]:
    cr, cc = child
    pr, pc = parent
    assert cr <= pr, "generator edges should flow upward to downward"
    path = [(r, cc) for r in range(cr, pr + 1)]
    if cc != pc:
        step = 1 if pc > cc else -1
        path.extend((pr, c) for c in range(cc + step, pc + step, step))
    return path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor a tree network by propagating leaf-square colors upward toward the root.

    Input structure:
    - The background is the most common color.
    - One connected 1-pixel-thick orthogonal tree occupies the grid, and every leaf is a solid 2x2 colored block attached to one terminal pixel of that tree.
    - Internal branch points are single pixels where one parent branch meets two or three child branches.

    Rule:
    - Keep every 2x2 leaf block unchanged and treat its color as the color of the adjacent terminal branch.
    - For each internal branch point, count the descendant leaf colors in each child subtree; the parent branch takes the most frequent color, breaking ties in favor of the leftmost tied subtree.
    - Apply that leaf-to-root propagation to every branch segment until the entire tree has been recolored.

    Key insight:
    - The grey drawing is not just geometry: it encodes an ordered tree, and each internal segment inherits a majority color computed from the leaf blocks below it.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    leaves, leaf_mask = _find_leaf_blocks(g, bg)

    network_mask = (g != bg) & ~leaf_mask
    network_components = _components(network_mask)
    assert len(network_components) == 1, "expected a single connected tree network"

    h, w = g.shape
    network_pixels = [tuple(map(int, rc)) for rc in np.argwhere(network_mask)]
    neighbors: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in network_pixels:
        nbrs: list[tuple[int, int]] = []
        for rr, cc in _neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and network_mask[rr, cc]:
                nbrs.append((rr, cc))
        neighbors[(r, c)] = nbrs

    leaf_info: dict[tuple[int, int], dict[str, object]] = {}
    for leaf in leaves:
        coords = leaf["coords"]
        assert isinstance(coords, np.ndarray)
        touches: set[tuple[int, int]] = set()
        for r, c in coords:
            for rr, cc in _neighbors4(int(r), int(c)):
                if 0 <= rr < h and 0 <= cc < w and network_mask[rr, cc]:
                    touches.add((int(rr), int(cc)))
        assert len(touches) == 1, "each 2x2 leaf must attach to exactly one terminal tree pixel"
        term = next(iter(touches))
        assert term not in leaf_info, "leaf attachments must be unique"
        r0, c0, _r1, _c1 = leaf["bbox"]
        leaf_info[term] = {
            "color": int(leaf["color"]),
            "leftmost": (int(c0), int(r0)),
        }

    significant = {
        pos for pos, nbrs in neighbors.items() if len(nbrs) != 2
    } | set(leaf_info.keys())
    assert significant, "failed to find graph nodes"

    root_candidates = [
        pos
        for pos in significant
        if len(neighbors[pos]) == 1 and pos not in leaf_info
    ]
    assert len(root_candidates) == 1, "expected exactly one non-leaf root endpoint"
    root = root_candidates[0]

    edge_paths: dict[int, list[tuple[int, int]]] = {}
    graph: dict[tuple[int, int], list[tuple[tuple[int, int], int]]] = defaultdict(list)
    visited_pairs: set[frozenset[tuple[int, int]]] = set()
    edge_id = 0

    for start in significant:
        for nxt in neighbors[start]:
            pair = frozenset((start, nxt))
            if pair in visited_pairs:
                continue
            path = [start, nxt]
            visited_pairs.add(pair)
            prev, cur = start, nxt
            while cur not in significant:
                nxts = [cand for cand in neighbors[cur] if cand != prev]
                assert len(nxts) == 1, "tree paths must not branch between graph nodes"
                nxt2 = nxts[0]
                visited_pairs.add(frozenset((cur, nxt2)))
                path.append(nxt2)
                prev, cur = cur, nxt2
            end = cur
            edge_id += 1
            edge_paths[edge_id] = path
            graph[start].append((end, edge_id))
            graph[end].append((start, edge_id))

    assert sum(len(v) for v in graph.values()) // 2 == len(significant) - 1, "compressed graph must be a tree"

    parent: dict[tuple[int, int], tuple[int, int] | None] = {root: None}
    parent_edge: dict[tuple[int, int], int] = {}
    stack = [root]
    while stack:
        node = stack.pop()
        for nxt, eid in graph[node]:
            if nxt in parent:
                continue
            parent[nxt] = node
            parent_edge[nxt] = int(eid)
            stack.append(nxt)

    assert len(parent) == len(significant), "tree must be fully reachable from the root"

    children: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for node, par in parent.items():
        if par is None:
            continue
        children[par].append(node)
    for node in children:
        children[node].sort(key=lambda pos: (int(pos[1]), int(pos[0])))

    summary: dict[tuple[int, int], dict[str, object]] = {}

    def summarize(node: tuple[int, int]) -> dict[str, object]:
        if node in leaf_info:
            info = leaf_info[node]
            color = int(info["color"])
            result = {
                "counts": Counter({color: 1}),
                "leftmost": {color: tuple(info["leftmost"])},
                "winner": color,
            }
            summary[node] = result
            return result

        child_infos = [summarize(child) for child in children[node]]
        counts: Counter[int] = Counter()
        leftmost: dict[int, tuple[int, int]] = {}
        for child_info in child_infos:
            child_counts = child_info["counts"]
            assert isinstance(child_counts, Counter)
            counts.update(child_counts)
            child_leftmost = child_info["leftmost"]
            assert isinstance(child_leftmost, dict)
            for color, pos in child_leftmost.items():
                cur = leftmost.get(int(color))
                if cur is None or tuple(pos) < cur:
                    leftmost[int(color)] = tuple(pos)

        winner = _pick_subtree_winner(counts, leftmost, child_infos)
        result = {
            "counts": counts,
            "leftmost": leftmost,
            "winner": winner,
        }
        summary[node] = result
        return result

    root_info = summarize(root)

    out = g.copy()
    for node, par in parent.items():
        if par is None:
            continue
        winner = int(summary[node]["winner"])
        path = edge_paths[parent_edge[node]]
        if path[0] == par and path[-1] == node:
            path = list(reversed(path))
        assert path[0] == node and path[-1] == par
        for r, c in path[:-1]:
            out[int(r), int(c)] = winner
    out[root[0], root[1]] = int(root_info["winner"])
    return out


@dataclass(eq=False)
class GenNode:
    children: list["GenNode"] = field(default_factory=list)
    leaf_color: int | None = None
    leaf_index: int = -1
    x: int = 0
    row: int = 0
    attach_offset: int = 0
    winner: int | None = None
    is_tie: bool = False

    @property
    def is_leaf(self) -> bool:
        return not self.children


def _compositions(total: int, parts: int) -> list[tuple[int, ...]]:
    if parts == 1:
        return [(total,)]
    out: list[tuple[int, ...]] = []
    for first in range(1, total - parts + 2):
        for rest in _compositions(total - first, parts - 1):
            out.append((first,) + rest)
    return out


def _max_leaves_at_depth(depth: int) -> int:
    return 3 ** max(0, MAX_TREE_DEPTH - depth)


def _build_tree(rng: random.Random, leaves: int, depth: int = 0) -> GenNode:
    node = GenNode()
    if leaves == 1 or depth >= MAX_TREE_DEPTH:
        assert leaves == 1, "depth cap exceeded"
        return node

    arities = [k for k in (2, 3) if k <= leaves]
    rng.shuffle(arities)
    options: list[tuple[int, tuple[int, ...]]] = []
    for arity in arities:
        for parts in _compositions(leaves, arity):
            if all(part <= _max_leaves_at_depth(depth + 1) for part in parts):
                options.append((arity, parts))
    assert options, "failed to find a feasible tree partition"

    _arity, parts = rng.choice(options)
    node.children = [_build_tree(rng, part, depth + 1) for part in parts]
    return node


def _ordered_leaves(node: GenNode) -> list[GenNode]:
    if node.is_leaf:
        return [node]
    out: list[GenNode] = []
    for child in node.children:
        out.extend(_ordered_leaves(child))
    return out


def _internal_nodes(node: GenNode) -> list[GenNode]:
    if node.is_leaf:
        return []
    out = [node]
    for child in node.children:
        out.extend(_internal_nodes(child))
    return out


def _evaluate_tree(node: GenNode) -> tuple[Counter[int], dict[int, int]]:
    if node.is_leaf:
        assert node.leaf_color is not None
        assert node.leaf_index >= 0
        node.winner = int(node.leaf_color)
        node.is_tie = False
        return Counter({int(node.leaf_color): 1}), {int(node.leaf_color): int(node.leaf_index)}

    counts: Counter[int] = Counter()
    leftmost: dict[int, int] = {}
    child_data: list[tuple[GenNode, Counter[int], dict[int, int]]] = []
    for child in node.children:
        child_counts, child_leftmost = _evaluate_tree(child)
        child_data.append((child, child_counts, child_leftmost))
        counts.update(child_counts)
        for color, idx in child_leftmost.items():
            cur = leftmost.get(int(color))
            if cur is None or int(idx) < cur:
                leftmost[int(color)] = int(idx)

    best = max(counts.values())
    tied = [color for color, value in counts.items() if int(value) == int(best)]
    node.is_tie = len(tied) > 1
    if len(tied) == 1:
        node.winner = int(tied[0])
        return counts, leftmost

    tied_set = set(map(int, tied))
    for child, child_counts, child_leftmost in child_data:
        restricted = Counter(
            {
                int(color): int(child_counts[color])
                for color in tied_set
                if int(child_counts[color]) > 0
            }
        )
        if not restricted:
            continue
        best_child = max(restricted.values())
        child_tied = [
            color for color, value in restricted.items() if int(value) == int(best_child)
        ]
        node.winner = int(min(child_tied, key=lambda color: int(child_leftmost[int(color)])))
        return counts, leftmost

    raise AssertionError("failed to resolve generator tie")


def _count_vectors(total: int, parts: int) -> list[tuple[int, ...]]:
    if parts == 1:
        return [(total,)]
    out: list[tuple[int, ...]] = []
    for first in range(total + 1):
        for rest in _count_vectors(total - first, parts - 1):
            out.append((first,) + rest)
    return out


def _assign_leaf_colors(rng: random.Random, root: GenNode, bg: int) -> bool:
    leaves = _ordered_leaves(root)
    for idx, leaf in enumerate(leaves):
        leaf.leaf_index = idx

    internal = _internal_nodes(root)
    want_root_tie = bool(rng.randrange(2))
    available = [color for color in COLOR_POOL if color not in {bg, LINE_COLOR}]
    palette_size = rng.choice((3, 4))

    for _ in range(200):
        palette = rng.sample(available, palette_size)
        vectors = [
            vec
            for vec in _count_vectors(len(leaves), palette_size)
            if sum(v > 0 for v in vec) >= 2
            and ((sum(v == max(vec) for v in vec if v > 0) >= 2) == want_root_tie)
        ]
        if not vectors:
            continue
        vec = rng.choice(vectors)
        bag: list[int] = []
        for color, count in zip(palette, vec):
            bag.extend([int(color)] * int(count))

        for _ in range(60):
            rng.shuffle(bag)
            for leaf, color in zip(leaves, bag):
                leaf.leaf_color = int(color)
            _evaluate_tree(root)
            if root.is_tie != want_root_tie:
                continue
            if len(set(bag)) < 2:
                continue
            if len(internal) >= 2:
                others = [node for node in internal if node is not root]
                if others and not any(node.is_tie != want_root_tie for node in others):
                    continue
            return True
    return False


def _assign_layout(rng: random.Random, root: GenNode) -> int:
    leaves = _ordered_leaves(root)
    for idx, leaf in enumerate(leaves):
        leaf.x = 2 + idx * LEAF_X_STEP
        leaf.row = LEAF_TERMINAL_ROW
        leaf.attach_offset = rng.randint(0, 1)

    def place_x(node: GenNode) -> int:
        if node.is_leaf:
            return int(node.x)
        child_xs = [place_x(child) for child in node.children]
        if len(child_xs) == 3:
            node.x = int(child_xs[1])
        else:
            mode = rng.choice(("left-right", "left-up", "up-right"))
            if mode == "left-right":
                node.x = int((child_xs[0] + child_xs[1]) // 2)
            elif mode == "left-up":
                node.x = int(child_xs[1])
            else:
                node.x = int(child_xs[0])
        return int(node.x)

    def place_rows(node: GenNode) -> int:
        if node.is_leaf:
            return int(node.row)
        max_child_row = max(place_rows(child) for child in node.children)
        node.row = int(max_child_row + rng.randint(*NODE_GAP_RANGE))
        return int(node.row)

    place_x(root)
    place_rows(root)
    return int(root.row + rng.randint(*TRUNK_RANGE))


def _collect_pixels(root: GenNode, base_row: int) -> tuple[set[tuple[int, int]], list[tuple[int, int, int]]]:
    network: set[tuple[int, int]] = set(_edge_path((int(root.row), int(root.x)), (int(base_row), int(root.x))))
    leaf_cells: list[tuple[int, int, int]] = []

    def walk(node: GenNode) -> None:
        if node.is_leaf:
            assert node.leaf_color is not None
            left = int(node.x - node.attach_offset)
            for dr in (0, 1):
                for dc in (0, 1):
                    leaf_cells.append((dr, left + dc, int(node.leaf_color)))
            return
        for child in node.children:
            network.update(_edge_path((int(child.row), int(child.x)), (int(node.row), int(node.x))))
            walk(child)

    walk(root)
    return network, leaf_cells


def _render_grid(
    rng: random.Random, root: GenNode, base_row: int, bg: int
) -> np.ndarray | None:
    network, leaf_cells = _collect_pixels(root, base_row)
    all_cells = [(r, c) for r, c in network] + [(r, c) for r, c, _ in leaf_cells]
    ymin = min(r for r, _ in all_cells)
    ymax = max(r for r, _ in all_cells)
    xmin = min(c for _, c in all_cells)
    xmax = max(c for _, c in all_cells)

    top_pad = rng.randint(0, min(2, 29 - (ymax - ymin)))
    height = int((ymax - ymin + 1) + top_pad)
    if height > 30:
        return None

    min_width = int((xmax - xmin + 1) + 2)
    if min_width > 30:
        return None
    width = int(min_width + rng.randint(0, min(6, 30 - min_width)))

    left_bound = int(1 - xmin)
    right_bound = int(width - 2 - xmax)
    if left_bound > right_bound:
        return None
    target = int(width // 2 - root.x)
    center_shift = min(max(target, left_bound), right_bound)
    jitter_lo = max(left_bound, center_shift - 1)
    jitter_hi = min(right_bound, center_shift + 1)
    shift_c = int(rng.randint(jitter_lo, jitter_hi))
    shift_r = int(top_pad - ymin)

    grid = np.full((height, width), bg, dtype=int)
    for r, c in network:
        grid[int(r + shift_r), int(c + shift_c)] = LINE_COLOR
    for r, c, color in leaf_cells:
        grid[int(r + shift_r), int(c + shift_c)] = int(color)
    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    leaf_count = rng.randint(*LEAF_COUNT_RANGE)
    bg = rng.choice(COLOR_POOL)
    root = _build_tree(rng, leaf_count)
    if not _assign_leaf_colors(rng, root, bg):
        return None

    base_row = _assign_layout(rng, root)
    grid = _render_grid(rng, root, base_row, bg)
    if grid is None:
        return None

    out = main(grid)
    if not np.any(out != grid):
        return None
    if np.any(out == LINE_COLOR):
        return None
    if out.shape != grid.shape or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid tree-color propagation puzzle")
