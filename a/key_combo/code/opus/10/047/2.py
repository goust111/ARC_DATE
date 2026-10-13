from __future__ import annotations

import heapq
import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
BAR = 2
PATH = 3
SQUARE = 8

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
INF = 10**9


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _component_masks(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out = np.zeros_like(mask, dtype=bool)
            for rr, cc in pts:
                out[rr, cc] = True
            comps.append(out)
    return comps


def _all_placements(mask: np.ndarray) -> list[dict]:
    h, w = mask.shape
    placements: dict[tuple[tuple[int, int], ...], dict] = {}
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue

            for r0 in (r - 1, r):
                for c0 in (c - 1, c):
                    if 0 <= r0 < h - 1 and 0 <= c0 < w - 1:
                        cells = [(r0, c0), (r0 + 1, c0), (r0, c0 + 1), (r0 + 1, c0 + 1)]
                        if all(mask[rr, cc] for rr, cc in cells):
                            key = tuple(sorted(cells))
                            placements[key] = {"kind": "square", "cells": key}

            for c0 in range(c - 2, c + 1):
                if 0 <= c0 < w - 2:
                    cells = [(r, c0 + dc) for dc in range(3)]
                    if all(mask[rr, cc] for rr, cc in cells):
                        key = tuple(sorted(cells))
                        placements[key] = {"kind": "bar", "cells": key}

            for r0 in range(r - 2, r + 1):
                if 0 <= r0 < h - 2:
                    cells = [(r0 + dr, c) for dr in range(3)]
                    if all(mask[rr, cc] for rr, cc in cells):
                        key = tuple(sorted(cells))
                        placements[key] = {"kind": "bar", "cells": key}
    return list(placements.values())


def _unique_decomposition(mask: np.ndarray) -> list[dict]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    if not coords:
        return []

    idx_of = {rc: i for i, rc in enumerate(coords)}
    placements = _all_placements(mask)
    for placement in placements:
        bit = 0
        for cell in placement["cells"]:
            bit |= 1 << idx_of[cell]
        placement["bit"] = bit

    cell_to_choices: list[list[int]] = [[] for _ in coords]
    for pi, placement in enumerate(placements):
        for cell in placement["cells"]:
            cell_to_choices[idx_of[cell]].append(pi)

    full = (1 << len(coords)) - 1
    solutions: list[list[int]] = []

    def search(covered: int, picked: list[int]) -> None:
        if len(solutions) >= 2:
            return
        if covered == full:
            solutions.append(picked.copy())
            return

        best_cell = None
        best_options = None
        for i in range(len(coords)):
            if covered >> i & 1:
                continue
            options = [pi for pi in cell_to_choices[i] if placements[pi]["bit"] & covered == 0]
            if not options:
                return
            if best_options is None or len(options) < len(best_options):
                best_cell = i
                best_options = options
                if len(best_options) == 1:
                    break

        assert best_cell is not None and best_options is not None
        for pi in best_options:
            picked.append(pi)
            search(covered | placements[pi]["bit"], picked)
            picked.pop()

    search(0, [])
    assert len(solutions) == 1, "grey pixels must admit a unique primitive decomposition"
    return [placements[pi] for pi in solutions[0]]


def _terminal_accesses(grid: np.ndarray, squares: list[dict]) -> list[set[tuple[int, int]]]:
    accesses: list[set[tuple[int, int]]] = []
    for placement in squares:
        nbrs: set[tuple[int, int]] = set()
        for r, c in placement["cells"]:
            for nr, nc in _neighbors4(r, c):
                if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == BG:
                    nbrs.add((nr, nc))
        assert nbrs, "every terminal square must touch the background"
        accesses.append(nbrs)
    return accesses


def _build_graph(passable: np.ndarray) -> tuple[list[tuple[int, int]], dict[tuple[int, int], int], list[list[int]]]:
    nodes = [tuple(map(int, rc)) for rc in np.argwhere(passable)]
    index = {rc: i for i, rc in enumerate(nodes)}
    adj = [[] for _ in nodes]
    for i, (r, c) in enumerate(nodes):
        for nr, nc in _neighbors4(r, c):
            j = index.get((nr, nc))
            if j is not None:
                adj[i].append(j)
    return nodes, index, adj


def _steiner_tree(
    passable: np.ndarray,
    terminals: list[set[tuple[int, int]]],
) -> tuple[int, set[tuple[int, int]]]:
    nodes, index, adj = _build_graph(passable)
    assert nodes, "background channel must be non-empty"
    n = len(nodes)
    k = len(terminals)
    full_mask = (1 << k) - 1

    node_term_mask = [0] * n
    for t, cells in enumerate(terminals):
        usable = 0
        for cell in cells:
            if cell in index:
                node_term_mask[index[cell]] |= 1 << t
                usable += 1
        assert usable > 0, "each terminal needs at least one reachable access cell"

    dist = [[INF] * n for _ in range(1 << k)]
    parent: list[list[tuple]] = [[("none",)] * n for _ in range(1 << k)]

    for v, mask in enumerate(node_term_mask):
        sub = mask
        while sub:
            if 1 < dist[sub][v]:
                dist[sub][v] = 1
                parent[sub][v] = ("init",)
            sub = (sub - 1) & mask

    for mask in range(1, full_mask + 1):
        sub = (mask - 1) & mask
        while sub:
            other = mask ^ sub
            if sub <= other:
                dv = dist[sub]
                ov = dist[other]
                mv = dist[mask]
                pv = parent[mask]
                for v in range(n):
                    cand = dv[v] + ov[v] - 1
                    if cand < mv[v]:
                        mv[v] = cand
                        pv[v] = ("merge", sub, other)
            sub = (sub - 1) & mask

        heap = []
        for v in range(n):
            if dist[mask][v] < INF:
                heapq.heappush(heap, (dist[mask][v], v))
        while heap:
            cur_d, v = heapq.heappop(heap)
            if cur_d != dist[mask][v]:
                continue
            nd = cur_d + 1
            for nb in adj[v]:
                if nd < dist[mask][nb]:
                    dist[mask][nb] = nd
                    parent[mask][nb] = ("move", v)
                    heapq.heappush(heap, (nd, nb))

    best_v = min(range(n), key=lambda v: dist[full_mask][v])
    best_cost = dist[full_mask][best_v]
    assert best_cost < INF, "terminals must be connectable through background"

    tree_nodes: set[int] = set()

    def collect(mask: int, v: int) -> None:
        tag = parent[mask][v][0]
        tree_nodes.add(v)
        if tag == "init":
            return
        if tag == "move":
            collect(mask, int(parent[mask][v][1]))
            return
        if tag == "merge":
            collect(int(parent[mask][v][1]), v)
            collect(int(parent[mask][v][2]), v)
            return
        raise AssertionError("broken steiner reconstruction")

    collect(full_mask, best_v)
    return best_cost, {nodes[v] for v in tree_nodes}


def _decompose_grid(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    grey_mask = np.asarray(grid == GREY)
    pieces = _unique_decomposition(grey_mask)
    squares = [p for p in pieces if p["kind"] == "square"]
    bars = [p for p in pieces if p["kind"] == "bar"]
    assert 2 <= len(squares) <= 4, "expected 2-4 terminal squares"
    assert bars, "expected at least one bar obstacle"
    return squares, bars


def _solve_structure(grid: np.ndarray) -> tuple[np.ndarray, list[dict], list[dict], set[tuple[int, int]], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert np.all((g == BG) | (g == GREY)), "input must contain only black background and grey primitives"

    squares, bars = _decompose_grid(g)
    terminals = _terminal_accesses(g, squares)
    passable = g == BG
    cost, tree = _steiner_tree(passable, terminals)
    return g, squares, bars, tree, cost


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decompose grey composite pieces into square terminals and bar walls, then fill the unique minimum background tree connecting the squares.

    Input structure:
    - Black background pixels form the only traversable space.
    - Grey foreground pixels are exactly decomposable into non-overlapping 2x2 squares and 1x3 or 3x1 bars, and several primitives may touch inside larger 4-connected grey regions.
    - The total number of 2x2 squares is between two and four.

    Rule:
    - Recover the unique primitive decomposition of every grey pixel.
    - Recolor each 2x2 square cyan to mark it as a terminal, and recolor each length-three bar red to mark it as an obstacle.
    - Treat both recolored primitive types as blocked, and among the remaining black pixels find the minimum-size 4-connected Steiner tree whose pixels touch every cyan square.
    - Recolor exactly those tree pixels green and leave every other pixel unchanged.

    Key insight:
    - Geometry, not the original color, assigns roles: only after decomposing the grey mass can you tell which parts become endpoints and which become walls, and that role assignment then determines the path-filling step.
    """
    g, squares, bars, tree, _cost = _solve_structure(grid)
    out = g.copy()
    for placement in bars:
        for r, c in placement["cells"]:
            out[r, c] = BAR
    for placement in squares:
        for r, c in placement["cells"]:
            out[r, c] = SQUARE
    for r, c in tree:
        out[r, c] = PATH
    return out


def _chain_cells(start: tuple[int, int], orientation: str, n_bars: int) -> list[tuple[int, int]]:
    r, c = start
    cells = []
    if orientation == "h":
        for dc in range(3 * n_bars):
            cells.append((r, c + dc))
    else:
        for dr in range(3 * n_bars):
            cells.append((r + dr, c))
    return cells


def _can_place(cells: list[tuple[int, int]], occ: set[tuple[int, int]], shape: tuple[int, int]) -> bool:
    for r, c in cells:
        if not _in_bounds(shape, r, c) or (r, c) in occ:
            return False
    return True


def _paint_cells(grid: np.ndarray, occ: set[tuple[int, int]], cells: list[tuple[int, int]]) -> None:
    for r, c in cells:
        grid[r, c] = GREY
        occ.add((r, c))


def _edge_square_positions(h: int, w: int, count: int) -> list[tuple[str, tuple[int, int]]]:
    sides = ["top", "bottom", "left", "right"]
    random.shuffle(sides)
    chosen = sides[:count]
    positions = []
    used_ranges: list[tuple[str, int]] = []
    for side in chosen:
        for _ in range(60):
            if side in ("top", "bottom"):
                c = random.randrange(2, w - 4)
                if any(s == side and abs(x - c) < 4 for s, x in used_ranges):
                    continue
                r = 0 if side == "top" else h - 2
                positions.append((side, (r, c)))
                used_ranges.append((side, c))
                break
            else:
                r = random.randrange(2, h - 4)
                if any(s == side and abs(x - r) < 4 for s, x in used_ranges):
                    continue
                c = 0 if side == "left" else w - 2
                positions.append((side, (r, c)))
                used_ranges.append((side, r))
                break
        else:
            raise RuntimeError("failed to place terminal squares on edges")
    return positions


def _square_support_bars(side: str, top_left: tuple[int, int]) -> list[list[tuple[int, int]]]:
    r, c = top_left
    if side == "top":
        return [
            [(rr, c - 1) for rr in range(r, r + 3)],
            [(rr, c + 2) for rr in range(r, r + 3)],
        ]
    if side == "bottom":
        return [
            [(rr, c - 1) for rr in range(r - 1, r + 2)],
            [(rr, c + 2) for rr in range(r - 1, r + 2)],
        ]
    if side == "left":
        return [
            [(r - 1, cc) for cc in range(c, c + 3)],
            [(r + 2, cc) for cc in range(c, c + 3)],
        ]
    return [
        [(r - 1, cc) for cc in range(c - 1, c + 2)],
        [(r + 2, cc) for cc in range(c - 1, c + 2)],
    ]


def _place_terminal_module(
    grid: np.ndarray,
    occ: set[tuple[int, int]],
    side: str,
    top_left: tuple[int, int],
    extra_attach: bool,
) -> bool:
    r, c = top_left
    square = [(r + dr, c + dc) for dr in range(2) for dc in range(2)]
    if not _can_place(square, occ, grid.shape):
        return False

    for bar_cells in _square_support_bars(side, top_left):
        if not _can_place(bar_cells, occ, grid.shape):
            return False

    _paint_cells(grid, occ, square)
    for bar_cells in _square_support_bars(side, top_left):
        _paint_cells(grid, occ, bar_cells)

    if extra_attach:
        options = []
        if side in ("top", "bottom"):
            for rr in range(max(1, r - 3), min(grid.shape[0] - 1, r + 2)):
                left_chain = [(rr, c + dc) for dc in range(3)]
                right_chain = [(rr, c - 1 + dc) for dc in range(3)]
                options.extend([left_chain, right_chain])
        else:
            for cc in range(max(1, c - 3), min(grid.shape[1] - 1, c + 2)):
                up_chain = [(r + dr, cc) for dr in range(3)]
                down_chain = [(r - 1 + dr, cc) for dr in range(3)]
                options.extend([up_chain, down_chain])
        random.shuffle(options)
        for cells in options:
            if _can_place(cells, occ, grid.shape):
                _paint_cells(grid, occ, cells)
                break
    return True


def _place_random_chains(
    grid: np.ndarray,
    occ: set[tuple[int, int]],
    min_chains: int,
    max_chains: int,
) -> None:
    h, w = grid.shape
    n_chains = random.randint(min_chains, max_chains)
    placed = 0
    have_h = False
    have_v = False
    for _ in range(240):
        if placed >= n_chains and have_h and have_v:
            break
        orientation = random.choice(["h", "v"])
        n_bars = random.randint(1, 4)
        if orientation == "h":
            r = random.randrange(1, h - 1)
            c = random.randrange(1, w - 3 * n_bars)
        else:
            r = random.randrange(1, h - 3 * n_bars)
            c = random.randrange(1, w - 1)
        cells = _chain_cells((r, c), orientation, n_bars)
        if not _can_place(cells, occ, grid.shape):
            continue
        if orientation == "h":
            if any((rr, cc) in occ for rr, cc in [(r - 1, c), (r + 1, c + 3 * n_bars - 1)] if _in_bounds(grid.shape, rr, cc)):
                pass
            have_h = True
        else:
            have_v = True
        _paint_cells(grid, occ, cells)
        placed += 1

    assert placed >= min_chains


def _reachable_component(passable: np.ndarray, starts: set[tuple[int, int]]) -> set[tuple[int, int]]:
    q = deque([cell for cell in starts if passable[cell]])
    seen = set(q)
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(passable.shape, nr, nc) and passable[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return seen


def _has_cycle(passable: np.ndarray, starts: list[set[tuple[int, int]]]) -> bool:
    seed = set().union(*starts)
    comp = _reachable_component(passable, seed)
    if not comp:
        return False
    edges = 0
    for r, c in comp:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in comp:
                edges += 1
    return edges // 2 >= len(comp)


def _forbid_and_cost(grid: np.ndarray, forbid: tuple[int, int], squares: list[dict]) -> int:
    passable = np.asarray(grid == BG)
    if not passable[forbid]:
        return INF
    passable[forbid] = False
    terminals = _terminal_accesses(grid, squares)
    try:
        cost, _ = _steiner_tree(passable, terminals)
    except AssertionError:
        return INF
    return cost


def _semantic_ok(grid: np.ndarray) -> bool:
    try:
        g, squares, bars, tree, cost = _solve_structure(grid)
    except Exception:
        return False

    if not (2 <= len(squares) <= 4):
        return False
    if len(bars) < 3:
        return False
    if len(tree) != cost:
        return False
    if cost < 6:
        return False

    accesses = _terminal_accesses(g, squares)
    if not _has_cycle(g == BG, accesses):
        return False

    for cell in tree:
        if _forbid_and_cost(g, cell, squares) <= cost:
            return False

    out = main(g)
    if np.array_equal(out, g):
        return False

    if int(np.count_nonzero(out == SQUARE)) != 4 * len(squares):
        return False
    if int(np.count_nonzero(out == BAR)) != 3 * len(bars):
        return False
    if int(np.count_nonzero(out == PATH)) != cost:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.choice([18, 21, 24])
    w = random.choice([18, 21, 24])
    grid = np.zeros((h, w), dtype=int)
    occ: set[tuple[int, int]] = set()

    square_count = random.choice([2, 3, 4])
    terminal_modules = _edge_square_positions(h, w, square_count)
    random.shuffle(terminal_modules)

    attach_branch = random.choice([False, True])
    for i, (side, pos) in enumerate(terminal_modules):
        if not _place_terminal_module(grid, occ, side, pos, extra_attach=attach_branch and i % 2 == 0):
            return None

    style = random.choice(["sparse", "dense"])
    if style == "sparse":
        _place_random_chains(grid, occ, 4, 7)
    else:
        _place_random_chains(grid, occ, 6, 10)

    if random.random() < 0.6:
        for _ in range(60):
            orientation = random.choice(["h", "v"])
            if orientation == "h":
                r = random.randrange(1, h - 1)
                c = random.randrange(1, w - 5)
                cells = _chain_cells((r, c), "h", 2)
            else:
                r = random.randrange(1, h - 5)
                c = random.randrange(1, w - 1)
                cells = _chain_cells((r, c), "v", 2)
            if _can_place(cells, occ, grid.shape):
                _paint_cells(grid, occ, cells)
                break

    if int(np.count_nonzero(grid == GREY)) > (h * w) // 2:
        return None

    if not _semantic_ok(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
