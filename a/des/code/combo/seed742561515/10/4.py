from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
MAROON = 9

TOWER_COLORS = (BLUE, RED, YELLOW)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MIX_MAP = {
    frozenset({BLUE}): BLUE,
    frozenset({RED}): RED,
    frozenset({YELLOW}): YELLOW,
    frozenset({BLUE, RED}): MAGENTA,
    frozenset({RED, YELLOW}): ORANGE,
    frozenset({BLUE, YELLOW}): GREEN,
    frozenset({BLUE, RED, YELLOW}): MAROON,
}


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    comp_id = 0
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or int(labels[r, c]) != 0:
                continue
            comp_id += 1
            labels[r, c] = comp_id
            q = deque([(r, c)])
            while q:
                cr, cc = q.popleft()
                for dr, dc in DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and int(labels[nr, nc]) == 0:
                        labels[nr, nc] = comp_id
                        q.append((nr, nc))
    return labels, comp_id


def _find_towers(grid: np.ndarray) -> list[dict]:
    towers: list[dict] = []
    for color in TOWER_COLORS:
        labels, ncomp = _label_components(grid == color)
        for comp_id in range(1, ncomp + 1):
            pts = np.argwhere(labels == comp_id)
            if int(len(pts)) != 4:
                continue
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
            if r1 - r0 != 1 or c1 - c0 != 1:
                continue
            if not bool((grid[r0 : r1 + 1, c0 : c1 + 1] == color).all()):
                continue
            cells = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]
            towers.append({"color": int(color), "cells": cells})
    return towers


def _terminal_ring_cells(r0: int, c0: int) -> list[tuple[int, int]]:
    cells = []
    for r in range(r0, r0 + 4):
        for c in range(c0, c0 + 4):
            if r0 + 1 <= r <= r0 + 2 and c0 + 1 <= c <= c0 + 2:
                continue
            cells.append((r, c))
    return cells


def _find_terminals(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    terminals: list[dict] = []
    occupied = np.zeros((h, w), dtype=bool)
    for r0 in range(h - 3):
        for c0 in range(w - 3):
            if bool(occupied[r0 : r0 + 4, c0 : c0 + 4].any()):
                continue
            patch = grid[r0 : r0 + 4, c0 : c0 + 4]
            if not bool((patch[1:3, 1:3] == BLACK).all()):
                continue
            ring = np.ones((4, 4), dtype=bool)
            ring[1:3, 1:3] = False
            if not bool((patch[ring] == GREY).all()):
                continue
            occupied[r0 : r0 + 4, c0 : c0 + 4] = True
            terminals.append(
                {
                    "r0": int(r0),
                    "c0": int(c0),
                    "border": _terminal_ring_cells(r0, c0),
                }
            )
    return terminals


def _adjacent_component_ids(labels: np.ndarray, cells: list[tuple[int, int]]) -> set[int]:
    h, w = labels.shape
    ids: set[int] = set()
    for r, c in cells:
        if 0 <= r < h and 0 <= c < w and int(labels[r, c]) > 0:
            ids.add(int(labels[r, c]))
        for dr, dc in DIR4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and int(labels[nr, nc]) > 0:
                ids.add(int(labels[nr, nc]))
    return ids


def _mix_color(colors: set[int]) -> int:
    key = frozenset(int(color) for color in colors if int(color) in TOWER_COLORS)
    if key not in MIX_MAP:
        raise ValueError(f"Unsupported tower-color set: {sorted(key)}")
    return int(MIX_MAP[key])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Determine each terminal node's final color from the colors of all reachable towers.
    """
    g = np.asarray(grid, dtype=int)
    labels, _ = _label_components(g == GREY)
    towers = _find_towers(g)
    terminals = _find_terminals(g)

    comp_colors: dict[int, set[int]] = defaultdict(set)
    for tower in towers:
        for comp_id in _adjacent_component_ids(labels, tower["cells"]):
            comp_colors[int(comp_id)].add(int(tower["color"]))

    out = g.copy()
    for terminal in terminals:
        seen: set[int] = set()
        for comp_id in _adjacent_component_ids(labels, terminal["border"]):
            seen.update(comp_colors.get(int(comp_id), set()))
        if not seen:
            continue
        fill = _mix_color(seen)
        r0 = int(terminal["r0"])
        c0 = int(terminal["c0"])
        out[r0 + 1 : r0 + 3, c0 + 1 : c0 + 3] = fill
    return out


def _node_center(node: tuple[int, int], step: int, margin: int) -> tuple[int, int]:
    return margin + step * int(node[0]), margin + step * int(node[1])


def _paint_line(grid: np.ndarray, a: tuple[int, int], b: tuple[int, int], color: int) -> bool:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        c_start = min(c0, c1)
        c_end = max(c0, c1)
        for c in range(c_start, c_end + 1):
            if int(grid[r0, c]) not in (BLACK, color):
                return False
        grid[r0, c_start : c_end + 1] = color
        return True
    if c0 == c1:
        r_start = min(r0, r1)
        r_end = max(r0, r1)
        for r in range(r_start, r_end + 1):
            if int(grid[r, c0]) not in (BLACK, color):
                return False
        grid[r_start : r_end + 1, c0] = color
        return True
    return False


def _tower_cells(endpoint: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = endpoint
    dr, dc = direction
    if (dr, dc) == (0, -1):
        return [(r - 1, c - 2), (r - 1, c - 1), (r, c - 2), (r, c - 1)]
    if (dr, dc) == (0, 1):
        return [(r - 1, c + 1), (r - 1, c + 2), (r, c + 1), (r, c + 2)]
    if (dr, dc) == (-1, 0):
        return [(r - 2, c - 1), (r - 2, c), (r - 1, c - 1), (r - 1, c)]
    if (dr, dc) == (1, 0):
        return [(r + 1, c - 1), (r + 1, c), (r + 2, c - 1), (r + 2, c)]
    raise ValueError(f"Bad direction: {(dr, dc)}")


def _terminal_geometry(
    endpoint: tuple[int, int], direction: tuple[int, int]
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    r, c = endpoint
    dr, dc = direction
    if (dr, dc) == (0, -1):
        r0, c0 = r - 2, c - 4
    elif (dr, dc) == (0, 1):
        r0, c0 = r - 2, c + 1
    elif (dr, dc) == (-1, 0):
        r0, c0 = r - 4, c - 2
    elif (dr, dc) == (1, 0):
        r0, c0 = r + 1, c - 2
    else:
        raise ValueError(f"Bad direction: {(dr, dc)}")
    border = _terminal_ring_cells(r0, c0)
    inner = [(r0 + 1, c0 + 1), (r0 + 1, c0 + 2), (r0 + 2, c0 + 1), (r0 + 2, c0 + 2)]
    return border, inner


def _all_in_bounds(shape: tuple[int, int], cells: list[tuple[int, int]]) -> bool:
    h, w = shape
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _cells_are_black(grid: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    return all(int(grid[r, c]) == BLACK for r, c in cells)


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> bool:
    if not _all_in_bounds(grid.shape, cells):
        return False
    if not _cells_are_black(grid, cells):
        return False
    for r, c in cells:
        grid[r, c] = color
    return True


def _bbox_nonblack(grid: np.ndarray) -> tuple[int, int, int, int] | None:
    pts = np.argwhere(grid != BLACK)
    if int(len(pts)) == 0:
        return None
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _trim_pair(inp: np.ndarray, out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    box = _bbox_nonblack(inp)
    if box is None:
        raise ValueError("empty component")
    r0, c0, r1, c1 = box
    return inp[r0 : r1 + 1, c0 : c1 + 1].copy(), out[r0 : r1 + 1, c0 : c1 + 1].copy()


def _random_tree(
    rows: int, cols: int, node_count: int, rng: random.Random
) -> tuple[set[tuple[int, int]], list[tuple[tuple[int, int], tuple[int, int]]]] | None:
    root = (rng.randrange(rows), rng.randrange(cols))
    nodes = {root}
    edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    while len(nodes) < node_count:
        frontier = []
        for r, c in nodes:
            for dr, dc in DIR4:
                nr = r + dr
                nc = c + dc
                nxt = (nr, nc)
                if 0 <= nr < rows and 0 <= nc < cols and nxt not in nodes:
                    frontier.append(((r, c), nxt))
        if not frontier:
            return None
        parent, child = rng.choice(frontier)
        nodes.add(child)
        edges.append((parent, child))
    return nodes, edges


def _component_spec(num_components: int, rng: random.Random) -> tuple[int, int, int]:
    if num_components == 1:
        rows = rng.randint(3, 4)
        cols = rng.randint(3, 4)
        node_count = rng.randint(4, min(rows * cols, 7))
    elif num_components == 2:
        rows = rng.randint(2, 4)
        cols = rng.randint(2, 4)
        node_count = rng.randint(3, min(rows * cols, 6))
    else:
        rows = rng.randint(2, 3)
        cols = rng.randint(2, 3)
        node_count = rng.randint(2, min(rows * cols, 4))
    return rows, cols, node_count


def _generate_component(num_components: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    step = 4
    margin = 4
    rows, cols, node_count = _component_spec(num_components, rng)
    tree = _random_tree(rows, cols, node_count, rng)
    if tree is None:
        return None
    nodes, edges = tree

    adj: dict[tuple[int, int], set[tuple[int, int]]] = {node: set() for node in nodes}
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    leaves = [node for node in nodes if len(adj[node]) == 1]
    if len(leaves) < 2:
        return None
    if node_count >= 4 and rng.random() < 0.7 and len(leaves) < 3:
        return None

    max_towers = min(3, len(leaves) - 1)
    towers_count = rng.randint(1, max_towers)
    tower_leaves = rng.sample(leaves, towers_count)
    tower_colors = rng.sample(list(TOWER_COLORS), towers_count)
    tower_by_leaf = {leaf: color for leaf, color in zip(tower_leaves, tower_colors, strict=False)}

    h = margin * 2 + step * (rows - 1) + 1
    w = margin * 2 + step * (cols - 1) + 1
    inp = np.full((h, w), BLACK, dtype=int)

    for a, b in edges:
        ca = _node_center(a, step, margin)
        cb = _node_center(b, step, margin)
        if not _paint_line(inp, ca, cb, GREY):
            return None

    terminal_inners: list[list[tuple[int, int]]] = []
    for leaf in leaves:
        parent = next(iter(adj[leaf]))
        direction = (int(leaf[0] - parent[0]), int(leaf[1] - parent[1]))
        endpoint = _node_center(leaf, step, margin)
        if leaf in tower_by_leaf:
            if not _paint_cells(inp, _tower_cells(endpoint, direction), int(tower_by_leaf[leaf])):
                return None
        else:
            border, inner = _terminal_geometry(endpoint, direction)
            if not _paint_cells(inp, border, GREY):
                return None
            if not _all_in_bounds(inp.shape, inner) or not _cells_are_black(inp, inner):
                return None
            terminal_inners.append(inner)

    if not terminal_inners:
        return None

    out = inp.copy()
    fill = _mix_color(set(tower_colors))
    for inner in terminal_inners:
        for r, c in inner:
            out[r, c] = fill

    inp_trim, out_trim = _trim_pair(inp, out)
    if len(_find_towers(inp_trim)) != towers_count:
        return None
    if len(_find_terminals(inp_trim)) != len(terminal_inners):
        return None
    try:
        if not np.array_equal(main(inp_trim), out_trim):
            return None
    except Exception:
        return None
    return inp_trim, out_trim


def _place_patch(
    board_in: np.ndarray,
    board_out: np.ndarray,
    occ: np.ndarray,
    patch_in: np.ndarray,
    patch_out: np.ndarray,
    top: int,
    left: int,
) -> None:
    ph, pw = patch_in.shape
    board_in[top : top + ph, left : left + pw] = patch_in
    board_out[top : top + ph, left : left + pw] = patch_out
    occ[max(0, top - 1) : min(board_in.shape[0], top + ph + 1), max(0, left - 1) : min(board_in.shape[1], left + pw + 1)] = True


def _crop_with_padding(grid_in: np.ndarray, grid_out: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    box = _bbox_nonblack(grid_in)
    if box is None:
        raise ValueError("empty board")
    r0, c0, r1, c1 = box
    top_pad = rng.randint(0, min(3, r0))
    left_pad = rng.randint(0, min(3, c0))
    bottom_pad = rng.randint(0, min(3, grid_in.shape[0] - 1 - r1))
    right_pad = rng.randint(0, min(3, grid_in.shape[1] - 1 - c1))
    rr0 = r0 - top_pad
    cc0 = c0 - left_pad
    rr1 = r1 + bottom_pad
    cc1 = c1 + right_pad
    return grid_in[rr0 : rr1 + 1, cc0 : cc1 + 1].copy(), grid_out[rr0 : rr1 + 1, cc0 : cc1 + 1].copy()


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    num_components = rng.randint(1, 3)
    components: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(num_components):
        comp = None
        for _ in range(80):
            comp = _generate_component(num_components, rng)
            if comp is not None:
                break
        if comp is None:
            return None
        components.append(comp)

    max_h = max(comp[0].shape[0] for comp in components)
    max_w = max(comp[0].shape[1] for comp in components)
    board_h = rng.randint(max(18, max_h + 2), 30)
    board_w = rng.randint(max(18, max_w + 2), 30)

    board_in = np.full((board_h, board_w), BLACK, dtype=int)
    board_out = np.full((board_h, board_w), BLACK, dtype=int)
    occ = np.zeros((board_h, board_w), dtype=bool)

    order = list(range(len(components)))
    order.sort(key=lambda idx: components[idx][0].shape[0] * components[idx][0].shape[1], reverse=True)
    for idx in order:
        patch_in, patch_out = components[idx]
        ph, pw = patch_in.shape
        positions = [
            (top, left)
            for top in range(board_h - ph + 1)
            for left in range(board_w - pw + 1)
            if not bool(occ[top : top + ph, left : left + pw].any())
        ]
        if not positions:
            return None
        top, left = rng.choice(positions)
        _place_patch(board_in, board_out, occ, patch_in, patch_out, top, left)

    inp, out = _crop_with_padding(board_in, board_out, rng)
    if np.array_equal(inp, out):
        return None
    if len(_find_towers(inp)) == 0 or len(_find_terminals(inp)) == 0:
        return None
    try:
        if not np.array_equal(main(inp), out):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(300):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
