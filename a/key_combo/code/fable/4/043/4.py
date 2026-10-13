from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
GREY = 5
TEAL = 8

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GRID_H_RANGE = (12, 22)
GRID_W_RANGE = (12, 22)
CHAIN_LENGTH_CHOICES = [1, 2, 3, 4, 5]
CHAIN_LENGTH_WEIGHTS = [1, 3, 3, 2, 1]
DORMANT_DOMINO_RANGE = (1, 4)
EXTRA_SINGLETON_RANGE = (2, 6)
END_WITH_BLOCKER_PROB = 0.55
MAX_LINK_DISTANCE = 7
MAX_GENERATION_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], cell: tuple[int, int]) -> bool:
    r, c = cell
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _add(cell: tuple[int, int], direction: tuple[int, int], steps: int = 1) -> tuple[int, int]:
    return cell[0] + direction[0] * steps, cell[1] + direction[1] * steps


def _neighbors4(cell: tuple[int, int]):
    r, c = cell
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _ray_distance(
    start: tuple[int, int], direction: tuple[int, int], cell: tuple[int, int]
) -> int | None:
    dr, dc = direction
    sr, sc = start
    r, c = cell
    if dr != 0:
        if c != sc:
            return None
        delta = r - sr
        if delta * dr <= 0:
            return None
        return delta * dr
    if r != sr:
        return None
    delta = c - sc
    if delta * dc <= 0:
        return None
    return delta * dc


def _available_forward_steps(shape: tuple[int, int], head: tuple[int, int], direction: tuple[int, int]) -> int:
    steps = 0
    pos = _add(head, direction)
    while _in_bounds(shape, pos):
        steps += 1
        pos = _add(pos, direction)
    return steps


def _can_place_cells(
    cells: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
    *,
    allowed_neighbors: set[tuple[int, int]] | None = None,
) -> bool:
    allowed_neighbors = allowed_neighbors or set()
    cell_set = set(cells)
    for cell in cell_set:
        if not _in_bounds(shape, cell):
            return False
        if cell in occupied or cell in forbidden:
            return False
    for cell in cell_set:
        for nbr in _neighbors4(cell):
            if nbr in occupied and nbr not in cell_set and nbr not in allowed_neighbors:
                return False
    return True


def _enumerate_domino_placements(
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
):
    placements = []
    for r in range(shape[0]):
        for c in range(shape[1]):
            red = (r, c)
            for direction in DIRS:
                green = _add(red, direction)
                cells = [red, green]
                if not _can_place_cells(cells, shape, occupied, forbidden):
                    continue
                placements.append(
                    {
                        "red": red,
                        "green": green,
                        "dir": direction,
                        "cells": (red, green),
                    }
                )
    return placements


def _spark_candidates(
    shape: tuple[int, int],
    domino: dict,
    occupied: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
):
    cells = set(domino["cells"])
    blocked_front = _add(domino["green"], domino["dir"])
    candidates = []
    seen = set()
    for cell in domino["cells"]:
        for nbr in _neighbors4(cell):
            if nbr in seen or nbr in cells or nbr == blocked_front:
                continue
            seen.add(nbr)
            if not _can_place_cells([nbr], shape, occupied, forbidden, allowed_neighbors=cells):
                continue
            candidates.append(nbr)
    return candidates


def _clear_forward_positions(
    shape: tuple[int, int],
    head: tuple[int, int],
    direction: tuple[int, int],
    occupied: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
):
    positions: list[tuple[int, int]] = []
    pos = _add(head, direction)
    while _in_bounds(shape, pos):
        if pos in occupied or pos in forbidden:
            return positions, pos
        positions.append(pos)
        pos = _add(pos, direction)
    return positions, None


def _ray_link_candidates(
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
    shooter_head: tuple[int, int],
    shooter_dir: tuple[int, int],
):
    candidates = []
    for domino in _enumerate_domino_placements(shape, occupied, forbidden):
        hits = []
        for role in ("red", "green"):
            dist = _ray_distance(shooter_head, shooter_dir, domino[role])
            if dist is not None:
                hits.append((dist, role, domino[role]))
        if not hits:
            continue
        hit_dist, hit_role, hit_cell = min(hits, key=lambda item: item[0])
        if hit_dist < 2 or hit_dist > MAX_LINK_DISTANCE:
            continue
        corridor = [_add(shooter_head, shooter_dir, step) for step in range(1, hit_dist)]
        if any(cell in occupied or cell in forbidden for cell in corridor):
            continue
        candidates.append(
            {
                **domino,
                "hit_dist": hit_dist,
                "hit_role": hit_role,
                "hit_cell": hit_cell,
                "corridor": corridor[:-1],
            }
        )
    return candidates


def _place_domino(grid: np.ndarray, domino: dict, occupied: set[tuple[int, int]]) -> None:
    grid[domino["red"]] = RED
    grid[domino["green"]] = GREEN
    occupied.update(domino["cells"])


def _place_singleton(
    grid: np.ndarray, cell: tuple[int, int], color: int, occupied: set[tuple[int, int]]
) -> None:
    grid[cell] = color
    occupied.add(cell)


def _find_components(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    mask = (g == RED) | (g == GREEN)
    visited = np.zeros(g.shape, dtype=bool)
    dominos = []
    singletons = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if visited[r, c]:
            continue
        queue = deque([(r, c)])
        visited[r, c] = True
        comp = []
        while queue:
            cr, cc = queue.popleft()
            comp.append((cr, cc))
            for nr, nc in _neighbors4((cr, cc)):
                if _in_bounds(g.shape, (nr, nc)) and mask[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    queue.append((nr, nc))

        if len(comp) == 1:
            cell = comp[0]
            singletons.append({"cell": cell, "color": int(g[cell])})
            continue

        assert len(comp) == 2, "red/green objects must be dominos or isolated singletons"
        colors = [int(g[cell]) for cell in comp]
        assert sorted(colors) == [RED, GREEN], "each domino must contain one red tail and one green head"
        a, b = comp
        assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1, "domino cells must be 4-neighbors"
        red = a if int(g[a]) == RED else b
        green = a if int(g[a]) == GREEN else b
        dominos.append(
            {
                "red": red,
                "green": green,
                "dir": (green[0] - red[0], green[1] - red[1]),
                "cells": (red, green),
            }
        )
    return dominos, singletons


def _simulate_chain(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    spark_coords = np.argwhere(g == GREY)
    assert spark_coords.shape == (1, 2), "expected exactly one grey spark"
    spark = tuple(map(int, spark_coords[0]))

    dominos, singletons = _find_components(g)
    assert dominos, "expected at least one red-green domino"

    cell_to_domino: dict[tuple[int, int], int] = {}
    for idx, domino in enumerate(dominos):
        for cell in domino["cells"]:
            cell_to_domino[cell] = idx

    initial = {
        cell_to_domino[nbr]
        for nbr in _neighbors4(spark)
        if _in_bounds(g.shape, nbr) and nbr in cell_to_domino
    }
    assert len(initial) == 1, "spark must touch exactly one domino"

    out = g.copy()
    lit: list[int] = []
    seen = set()
    queue = deque(initial)

    while queue:
        idx = queue.popleft()
        if idx in seen:
            continue
        seen.add(idx)
        lit.append(idx)

        domino = dominos[idx]
        direction = domino["dir"]
        pos = _add(domino["green"], direction)
        while _in_bounds(out.shape, pos) and int(out[pos]) == BLACK:
            out[pos] = TEAL
            pos = _add(pos, direction)

        if _in_bounds(out.shape, pos):
            next_idx = cell_to_domino.get(pos)
            if next_idx is not None and next_idx not in seen:
                queue.append(next_idx)

    return out, lit, dominos, singletons


def main(grid: np.ndarray) -> np.ndarray:
    """
    Ignite the spark-adjacent red-green domino and draw the cyan ray chain through every domino it hits.

    Input structure:
    - The grid has a black background, several 4-connected two-cell dominos made of
      one red tail and one green head, and exactly one grey spark pixel.
    - Extra red or green single pixels may also appear as distractors and blockers.
    - The spark touches exactly one domino in the 4-neighborhood, and each domino's
      direction is the red-to-green arrow.

    Rule:
    - The domino touching the grey spark ignites first.
    - Each ignited domino emits a ray from its green head in its arrow direction,
      repainting black cells cyan until the ray reaches the boundary or any non-black cell.
    - If the first blocking cell belongs to a not-yet-ignited domino, that domino ignites
      next and emits its own ray; sparks, single-pixel distractors, and prior cyan rays block
      but never ignite.
    - Keep every original non-black object unchanged and add only the cyan ray trail.

    Key insight:
    - A red-green domino is a directional trigger, not just a pair: only dominos reached by
      the spark-caused hit chain fire, while untouched dominos stay silent even if they look valid.
    """
    out, _lit, _dominos, _singletons = _simulate_chain(np.asarray(grid, dtype=int))
    return out


def _initial_domino_candidates(shape: tuple[int, int], chain_length: int):
    occupied: set[tuple[int, int]] = set()
    forbidden: set[tuple[int, int]] = set()
    candidates = []
    for domino in _enumerate_domino_placements(shape, occupied, forbidden):
        if _available_forward_steps(shape, domino["green"], domino["dir"]) < 2:
            continue
        if not _spark_candidates(shape, domino, set(domino["cells"]), forbidden):
            continue
        candidates.append(domino)
    return candidates


def _randomize_candidates(candidates: list[dict]) -> list[dict]:
    shuffled = list(candidates)
    random.shuffle(shuffled)
    return shuffled


def _free_singleton_cells(
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
):
    cells = []
    for r in range(shape[0]):
        for c in range(shape[1]):
            cell = (r, c)
            if _can_place_cells([cell], shape, occupied, forbidden):
                cells.append(cell)
    return cells


def _generate_attempt() -> np.ndarray | None:
    shape = (
        random.randint(*GRID_H_RANGE),
        random.randint(*GRID_W_RANGE),
    )
    chain_length = random.choices(CHAIN_LENGTH_CHOICES, weights=CHAIN_LENGTH_WEIGHTS, k=1)[0]
    dormant_count = random.randint(*DORMANT_DOMINO_RANGE)
    extra_singleton_count = random.randint(*EXTRA_SINGLETON_RANGE)

    grid = np.zeros(shape, dtype=int)
    occupied: set[tuple[int, int]] = set()
    forbidden: set[tuple[int, int]] = set()
    chain_dominos = []

    initial_candidates = _randomize_candidates(_initial_domino_candidates(shape, chain_length))
    if not initial_candidates:
        return None

    first = initial_candidates[0]
    _place_domino(grid, first, occupied)
    chain_dominos.append(first)

    spark_options = _spark_candidates(shape, first, occupied, forbidden)
    if not spark_options:
        return None
    spark = random.choice(spark_options)
    _place_singleton(grid, spark, GREY, occupied)

    current = first
    for step_idx in range(chain_length - 1):
        candidates = _randomize_candidates(
            _ray_link_candidates(shape, occupied, forbidden, current["green"], current["dir"])
        )
        candidates = [
            cand
            for cand in candidates
            if _available_forward_steps(shape, cand["green"], cand["dir"]) >= 2
        ]
        if not candidates:
            return None

        chosen = candidates[0]
        forbidden.update(chosen["corridor"])
        _place_domino(grid, chosen, occupied)
        chain_dominos.append(chosen)
        current = chosen

    clear_positions, blocked_cell = _clear_forward_positions(
        shape, current["green"], current["dir"], occupied, forbidden
    )
    if not clear_positions:
        return None

    use_blocker = random.random() < END_WITH_BLOCKER_PROB
    if use_blocker:
        blocker_positions = clear_positions[1 : min(len(clear_positions), MAX_LINK_DISTANCE)]
        if blocker_positions:
            blocker = random.choice(blocker_positions)
            _place_singleton(grid, blocker, random.choice([RED, GREEN]), occupied)
            forbidden.update(clear_positions[: clear_positions.index(blocker)])
        elif blocked_cell is None:
            forbidden.update(clear_positions)
        else:
            return None
    else:
        if blocked_cell is not None:
            return None
        forbidden.update(clear_positions)

    dormant_needed = dormant_count
    while dormant_needed > 0:
        placements = _randomize_candidates(_enumerate_domino_placements(shape, occupied, forbidden))
        if not placements:
            return None
        chosen = placements[0]
        _place_domino(grid, chosen, occupied)
        dormant_needed -= 1

    singleton_colors = [RED, GREEN]
    while len(singleton_colors) < extra_singleton_count:
        singleton_colors.append(random.choice([RED, GREEN]))

    for color in singleton_colors:
        cells = _free_singleton_cells(shape, occupied, forbidden)
        if not cells:
            return None
        chosen = random.choice(cells)
        _place_singleton(grid, chosen, color, occupied)

    try:
        out, lit, dominos, singletons = _simulate_chain(grid)
    except Exception:
        return None

    if len(lit) != chain_length:
        return None
    if len(dominos) <= len(lit):
        return None
    if len(singletons) < 3:
        return None
    if not np.any(out == TEAL):
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid domino-chain input")
