from __future__ import annotations

from collections import defaultdict
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
CYAN = 8
MAROON = 9

SEED_COLORS = (BLUE, RED, GREEN, MAGENTA, CYAN)
DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
DIR_TO_DIAG = {
    (0, -1): (-1, -1),
    (0, 1): (-1, 1),
    (1, 0): (1, -1),
    (-1, 0): (1, 1),
}
DIAG_TO_DIR = {diag: direction for direction, diag in DIR_TO_DIAG.items()}

GEN_ATTEMPTS = 300
SEED_TRIES = 90
LATTICE_RANGE = (4, 9)
SEED_COUNT_OPTIONS = (2, 3)
MIN_SEED_DISTANCE = 4


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _neighbor_cells(shape: tuple[int, int], r: int, c: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in DIRS:
        rr = int(r + dr)
        cc = int(c + dc)
        if _in_bounds(shape, rr, cc):
            out.append((rr, cc))
    return out


def _seed_positions(grid: np.ndarray) -> list[tuple[int, int, int]]:
    seeds: list[tuple[int, int, int]] = []
    for r, c in np.argwhere(
        (grid != BLACK) & (grid != GREY) & (grid != ORANGE) & (grid != MAROON)
    ):
        seeds.append((int(r), int(c), int(grid[int(r), int(c)])))
    return seeds


def _degree_map(passable: np.ndarray) -> np.ndarray:
    h, w = passable.shape
    degree = np.zeros((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            if not bool(passable[r, c]):
                continue
            degree[r, c] = sum(
                1
                for rr, cc in _neighbor_cells((h, w), r, c)
                if bool(passable[rr, cc])
            )
    return degree


def _marker_map(grid: np.ndarray, passable: np.ndarray, degree: np.ndarray) -> dict[tuple[int, int], tuple[int, int]]:
    h, w = grid.shape
    markers: dict[tuple[int, int], tuple[int, int]] = {}
    for r in range(h):
        for c in range(w):
            if not bool(passable[r, c]) or int(degree[r, c]) < 3:
                continue
            found: list[tuple[int, int]] = []
            for diag, direction in DIAG_TO_DIR.items():
                rr = int(r + diag[0])
                cc = int(c + diag[1])
                if _in_bounds((h, w), rr, cc) and int(grid[rr, cc]) == ORANGE:
                    found.append(direction)
            if found:
                markers[(r, c)] = found[0]
    return markers


def _choose_next(
    prev: tuple[int, int],
    curr: tuple[int, int],
    passable: np.ndarray,
    degree: np.ndarray,
    markers: dict[tuple[int, int], tuple[int, int]],
    seed_blocks: set[tuple[int, int]],
) -> tuple[tuple[int, int] | None, str]:
    h, w = passable.shape
    options: dict[tuple[int, int], tuple[int, int]] = {}
    for dr, dc in DIRS:
        rr = int(curr[0] + dr)
        cc = int(curr[1] + dc)
        if not _in_bounds((h, w), rr, cc):
            continue
        if (rr, cc) == prev:
            continue
        if not bool(passable[rr, cc]):
            continue
        if (rr, cc) in seed_blocks:
            continue
        options[(dr, dc)] = (rr, cc)

    if not options:
        return None, "none"
    if len(options) == 1:
        return next(iter(options.values())), "single"

    incoming = (int(curr[0] - prev[0]), int(curr[1] - prev[1]))
    preferred = markers.get(curr)
    if preferred in options:
        return options[preferred], "preferred"
    if incoming in options:
        return options[incoming], "straight"

    for direction in DIRS:
        if direction in options:
            return options[direction], "remaining"
    return None, "none"


def _simulate_growth(grid: np.ndarray, want_stats: bool = False):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    passable = (g != GREY) & (g != ORANGE)
    degree = _degree_map(passable)
    markers = _marker_map(g, passable, degree)
    seeds = _seed_positions(g)
    seed_set = {(r, c) for r, c, _color in seeds}
    terminals = {tuple(map(int, coord)) for coord in np.argwhere(g == MAROON)}

    out = g.copy()
    occupied: set[tuple[int, int]] = set()
    frontier: list[tuple[tuple[int, int], tuple[int, int] | None, int, tuple[int, int] | None]] = []
    for r, c, color in seeds:
        for dr, dc in DIRS:
            rr = int(r + dr)
            cc = int(c + dc)
            if not _in_bounds((h, w), rr, cc):
                continue
            if not bool(passable[rr, cc]):
                continue
            if (rr, cc) in seed_set:
                continue
            frontier.append(((r, c), None, int(color), (rr, cc)))

    stats = {
        "conflicts": 0,
        "terminal_hits": 0,
        "blocked_by_occupied": 0,
        "dead_end_stops": 0,
        "preferred_used": 0,
        "straight_used": 0,
        "remaining_used": 0,
        "painted": 0,
    }

    while frontier:
        proposals: dict[tuple[int, int], list[tuple[tuple[int, int], int]]] = defaultdict(list)
        next_frontier: list[tuple[tuple[int, int], tuple[int, int], int, None]] = []

        for curr, prev, color, forced in frontier:
            if forced is not None:
                target = forced
                decision = "launch"
            else:
                assert prev is not None
                target, decision = _choose_next(prev, curr, passable, degree, markers, seed_set)
                if decision == "preferred":
                    stats["preferred_used"] += 1
                elif decision == "straight":
                    stats["straight_used"] += 1
                elif decision == "remaining":
                    stats["remaining_used"] += 1

            if target is None:
                continue
            if target in occupied:
                stats["blocked_by_occupied"] += 1
                continue
            if target in seed_set:
                continue
            if target not in terminals and int(degree[target]) == 1:
                stats["dead_end_stops"] += 1
                continue
            proposals[target].append((curr, int(color)))

        if not proposals:
            break

        for target, arrivals in proposals.items():
            colors = {int(color) for _curr, color in arrivals}
            if len(colors) >= 2:
                out[target] = YELLOW
                occupied.add(target)
                stats["conflicts"] += 1
                stats["painted"] += 1
                continue

            color = next(iter(colors))
            out[target] = int(color)
            occupied.add(target)
            stats["painted"] += 1

            if target in terminals:
                stats["terminal_hits"] += 1
                continue

            seen_currents: set[tuple[int, int]] = set()
            for curr, _color in arrivals:
                if curr in seen_currents:
                    continue
                seen_currents.add(curr)
                next_frontier.append((target, curr, int(color), None))

        frontier = next_frontier

    if want_stats:
        stats["decision_total"] = (
            int(stats["preferred_used"]) + int(stats["straight_used"]) + int(stats["remaining_used"])
        )
        stats["changed"] = int(np.count_nonzero(out != g))
        return out, stats
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow colored rays through a one-cell maze, using diagonal markers to resolve turns and yellow to mark simultaneous conflicts.

    Input structure:
    - The background is black, while grey pixels form one-cell-thick maze walls around a connected channel network.
    - Orange single-pixel markers sit on diagonal wall corners next to T-junctions or cross-junctions and encode an absolute preferred direction.
    - Several colored singleton seed pixels lie on channel cells, and some leaf-end channel cells are marked with maroon terminal pixels.

    Rule:
    - Each seed launches one ray into every passable neighboring channel cell, and all rays advance simultaneously one step at a time without crossing grey walls or orange markers.
    - When a ray reaches a branch, it first tries the junction's marker direction, otherwise it tries to continue straight, and otherwise it takes the remaining available direction; unmarked branches default to straight-then-remaining.
    - A ray stops before entering a plain dead end or any previously painted cell; entering a maroon terminal recolors that terminal to the ray color and ends the ray there.
    - If rays of different colors reach the same new cell on the same step, that cell becomes yellow and all colliding rays stop; seeds, walls, and orange markers remain unchanged.

    Key insight:
    - Solving the puzzle requires combining local path following with global synchronous simulation: the orange marker only biases a junction choice, while final colors depend on concurrent arrivals, blocking, and terminal handling.
    """
    return _simulate_growth(np.asarray(grid, dtype=int), want_stats=False)


def _lattice_neighbors(cell: tuple[int, int], rows: int, cols: int) -> list[tuple[int, int]]:
    r, c = cell
    out: list[tuple[int, int]] = []
    for dr, dc in DIRS:
        rr = int(r + dr)
        cc = int(c + dc)
        if 0 <= rr < int(rows) and 0 <= cc < int(cols):
            out.append((rr, cc))
    return out


def _random_tree(rows: int, cols: int) -> set[frozenset[tuple[int, int]]]:
    start = (random.randrange(rows), random.randrange(cols))
    seen = {start}
    stack = [start]
    edges: set[frozenset[tuple[int, int]]] = set()

    while stack:
        cell = stack[-1]
        unseen = [nb for nb in _lattice_neighbors(cell, rows, cols) if nb not in seen]
        if not unseen:
            stack.pop()
            continue
        nxt = random.choice(unseen)
        seen.add(nxt)
        edges.add(frozenset((cell, nxt)))
        stack.append(nxt)

    return edges


def _node_to_grid(cell: tuple[int, int]) -> tuple[int, int]:
    return (2 * int(cell[0]) + 1, 2 * int(cell[1]) + 1)


def _render_base(rows: int, cols: int, edges: set[frozenset[tuple[int, int]]]) -> tuple[np.ndarray, dict[tuple[int, int], int]]:
    grid = np.full((2 * int(rows) + 1, 2 * int(cols) + 1), GREY, dtype=int)
    degrees: dict[tuple[int, int], int] = {(r, c): 0 for r in range(rows) for c in range(cols)}

    for node in degrees:
        rr, cc = _node_to_grid(node)
        grid[rr, cc] = BLACK

    for edge in edges:
        a, b = tuple(edge)
        degrees[a] += 1
        degrees[b] += 1
        ar, ac = _node_to_grid(a)
        br, bc = _node_to_grid(b)
        grid[(ar + br) // 2, (ac + bc) // 2] = BLACK

    return grid, degrees


def _assign_markers(
    grid: np.ndarray,
    degrees: dict[tuple[int, int], int],
) -> bool:
    used: set[tuple[int, int]] = set()
    junctions = [node for node, deg in degrees.items() if int(deg) >= 3]
    random.shuffle(junctions)

    for node in junctions:
        rr, cc = _node_to_grid(node)
        choices = list(DIR_TO_DIAG.items())
        random.shuffle(choices)
        placed = False
        for direction, diag in choices:
            mr = int(rr + diag[0])
            mc = int(cc + diag[1])
            if (mr, mc) in used:
                continue
            if int(grid[mr, mc]) != GREY:
                continue
            grid[mr, mc] = ORANGE
            used.add((mr, mc))
            placed = True
            break
        if not placed:
            return False
    return True


def _place_terminals(
    grid: np.ndarray,
    degrees: dict[tuple[int, int], int],
) -> tuple[bool, list[tuple[int, int]], list[tuple[int, int]]]:
    leaves = [node for node, deg in degrees.items() if int(deg) == 1]
    if len(leaves) < 3:
        return False, [], []

    min_terminals = 1
    max_terminals = min(len(leaves) - 1, max(2, len(leaves) // 2))
    if max_terminals < min_terminals:
        return False, [], []
    terminal_count = random.randint(min_terminals, max_terminals)
    terminal_nodes = random.sample(leaves, terminal_count)
    plain_leaves = [node for node in leaves if node not in terminal_nodes]
    if not plain_leaves:
        return False, [], []

    for node in terminal_nodes:
        rr, cc = _node_to_grid(node)
        grid[rr, cc] = MAROON

    return True, terminal_nodes, plain_leaves


def _seed_candidates(grid: np.ndarray) -> list[tuple[int, int]]:
    passable = (grid != GREY) & (grid != ORANGE)
    degree = _degree_map(passable)
    candidates: list[tuple[int, int]] = []
    for r, c in np.argwhere(grid == BLACK):
        if int(degree[int(r), int(c)]) >= 2:
            candidates.append((int(r), int(c)))
    random.shuffle(candidates)
    return candidates


def _sample_seed_cells(candidates: list[tuple[int, int]], count: int) -> list[tuple[int, int]] | None:
    chosen: list[tuple[int, int]] = []
    for cell in candidates:
        if all(abs(int(cell[0] - rr)) + abs(int(cell[1] - cc)) >= MIN_SEED_DISTANCE for rr, cc in chosen):
            chosen.append(cell)
            if len(chosen) == int(count):
                return chosen
    return None


def _accept_scene(stats: dict[str, int], mode: str) -> bool:
    if int(stats["changed"]) < 8:
        return False
    if int(stats["decision_total"]) < 1:
        return False

    if mode == "conflict":
        return int(stats["conflicts"]) >= 1
    if mode == "blocked":
        return int(stats["blocked_by_occupied"]) >= 1
    if mode == "terminal":
        return int(stats["terminal_hits"]) >= 1 and int(stats["dead_end_stops"]) >= 1
    if mode == "fallback":
        return int(stats["straight_used"]) + int(stats["remaining_used"]) >= 1
    return False


def _build_scene() -> np.ndarray | None:
    lattice_h = random.randint(*LATTICE_RANGE)
    lattice_w = random.randint(*LATTICE_RANGE)
    grid_h = 2 * int(lattice_h) + 1
    grid_w = 2 * int(lattice_w) + 1
    if max(grid_h, grid_w) > 30:
        return None

    edges = _random_tree(lattice_h, lattice_w)
    grid, degrees = _render_base(lattice_h, lattice_w, edges)

    if sum(1 for deg in degrees.values() if int(deg) >= 3) < 2:
        return None
    if not _assign_markers(grid, degrees):
        return None
    ok, _terminal_nodes, _plain_leaves = _place_terminals(grid, degrees)
    if not ok:
        return None

    candidates = _seed_candidates(grid)
    if len(candidates) < 4:
        return None

    modes = ["conflict", "blocked", "terminal", "fallback"]
    random.shuffle(modes)

    for mode in modes:
        for _ in range(SEED_TRIES):
            seed_count = random.choice(SEED_COUNT_OPTIONS)
            cells = _sample_seed_cells(candidates, seed_count)
            if cells is None:
                break

            colors = random.sample(SEED_COLORS, seed_count)
            scene = grid.copy()
            for (r, c), color in zip(cells, colors):
                scene[r, c] = int(color)

            _out, stats = _simulate_growth(scene, want_stats=True)
            if _accept_scene(stats, mode):
                return scene

    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        scene = _build_scene()
        if scene is not None:
            return scene
    raise RuntimeError("failed to generate a valid maze-growth scene")
