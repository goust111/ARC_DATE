import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIR8 = [
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
]
DIR_TO_INDEX = {direction: idx for idx, direction in enumerate(DIR8)}

GENERATE_ATTEMPTS = 300
GRID_H_RANGE = (10, 18)
GRID_W_RANGE = (10, 18)
EMITTER_COUNT_RANGE = (1, 3)
MAX_PRISMS = 4
MAX_DEPTH_CHOICES = (1, 2, 2, 3)
SPLIT_PROB_ROOT = 0.8
SPLIT_PROB_CHILD = 0.55
WALL_END_PROB = 0.35
EXTRA_WALL_RANGE = (0, 5)


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _edge_cells(shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    cells = {(0, c) for c in range(w)}
    cells |= {(h - 1, c) for c in range(w)}
    cells |= {(r, 0) for r in range(h)}
    cells |= {(r, w - 1) for r in range(h)}
    return sorted(cells)


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            coords = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if (
                        in_bounds(grid.shape, nr, nc)
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            coords.sort()
            comps.append({"color": color, "coords": coords})
    return comps


def _split_directions(direction: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    idx = DIR_TO_INDEX[direction]
    return DIR8[(idx - 1) % 8], DIR8[(idx + 1) % 8]


def _tower_origin_and_direction(
    coords: list[tuple[int, int]], shape: tuple[int, int]
) -> tuple[tuple[int, int], tuple[int, int]]:
    h, w = shape
    rows = sorted(r for r, _ in coords)
    cols = sorted(c for _, c in coords)

    if rows[0] == rows[1]:
        row = rows[0]
        assert cols[1] == cols[0] + 1, "horizontal tower must be contiguous"
        if cols[0] == 0:
            return (row, cols[1]), (0, 1)
        if cols[1] == w - 1:
            return (row, cols[0]), (0, -1)
    elif cols[0] == cols[1]:
        col = cols[0]
        assert rows[1] == rows[0] + 1, "vertical tower must be contiguous"
        if rows[0] == 0:
            return (rows[1], col), (1, 0)
        if rows[1] == h - 1:
            return (rows[0], col), (-1, 0)

    raise AssertionError("tower must be a 2-pixel edge-anchored short bar")


def _parse_scene(
    grid: np.ndarray,
) -> tuple[list[dict[str, object]], dict[tuple[int, int], int], set[tuple[int, int]], set[tuple[int, int]]]:
    emitters: list[dict[str, object]] = []
    prisms: dict[tuple[int, int], int] = {}
    walls: set[tuple[int, int]] = set()
    emitter_cells: set[tuple[int, int]] = set()

    for comp in _components(grid):
        color = int(comp["color"])
        coords = list(comp["coords"])
        if color == GREY:
            assert all(
                r in {0, grid.shape[0] - 1} or c in {0, grid.shape[1] - 1} for r, c in coords
            ), "grey absorber walls must sit on the border"
            walls.update(coords)
            continue

        if len(coords) == 1:
            prisms[coords[0]] = color
            continue

        if len(coords) == 2:
            origin, direction = _tower_origin_and_direction(coords, grid.shape)
            emitters.append(
                {
                    "color": color,
                    "coords": tuple(coords),
                    "origin": origin,
                    "direction": direction,
                }
            )
            emitter_cells.update(coords)
            continue

        raise AssertionError("expected only singleton prisms, edge towers, and grey walls")

    emitters.sort(key=lambda item: (item["origin"], item["direction"], item["color"]))
    assert emitters, "expected at least one emitter"
    assert prisms, "expected at least one prism"
    return emitters, prisms, walls, emitter_cells


def _cast_rays(
    out: np.ndarray,
    emitters: list[dict[str, object]],
    prisms: dict[tuple[int, int], int],
    walls: set[tuple[int, int]],
    emitter_cells: set[tuple[int, int]],
) -> np.ndarray:
    pending = [
        (tuple(emitter["origin"]), tuple(emitter["direction"]), int(emitter["color"]))
        for emitter in emitters
    ]
    seen: set[tuple[int, int, int, int, int]] = set()

    while pending:
        origin, direction, color = pending.pop()
        state = (origin[0], origin[1], direction[0], direction[1], color)
        if state in seen:
            continue
        seen.add(state)

        r, c = origin
        dr, dc = direction
        while True:
            nr, nc = r + dr, c + dc
            if not in_bounds(out.shape, nr, nc):
                break
            if (nr, nc) in walls or (nr, nc) in emitter_cells:
                break
            prism_color = prisms.get((nr, nc))
            if prism_color is not None:
                for split_dir in _split_directions(direction):
                    pending.append(((nr, nc), split_dir, prism_color))
                break
            if int(out[nr, nc]) != BG:
                break
            out[nr, nc] = color
            r, c = nr, nc

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit rays from edge-anchored short towers; prisms split rays by +/-45 degrees and recolor them.

    Input structure
    - Background is black.
    - Each emitter is a 2-pixel short bar that sits on one border, so the missing third
      cell of its implied 3-cell bar points inward and determines the launch direction.
    - Prisms are singleton colored pixels; grey border pixels are absorber walls.

    Rule
    - Every emitter fires a ray in its open direction using the emitter color.
    - When a ray reaches a prism, it stops before the prism and spawns two rays in the
      adjacent compass directions, with the prism's color.
    - Rays stop at the grid boundary, absorber walls, emitters, prisms they do not enter,
      or already-painted non-background pixels.

    Key insight
    - The short edge bar encodes the missing-end direction, and prisms overwrite the ray
      color while rotating the path structure from cardinal to diagonal and back.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    emitters, prisms, walls, emitter_cells = _parse_scene(g)
    out = g.copy()
    return _cast_rays(out, emitters, prisms, walls, emitter_cells)


def _free_run(
    shape: tuple[int, int],
    origin: tuple[int, int],
    direction: tuple[int, int],
    blocked: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    r, c = origin
    dr, dc = direction
    cells = []
    while True:
        r += dr
        c += dc
        if not in_bounds(shape, r, c):
            break
        if (r, c) in blocked:
            break
        cells.append((r, c))
    return cells


def _emitter_candidates(shape: tuple[int, int]) -> list[dict[str, object]]:
    h, w = shape
    candidates = []
    for c in range(1, w - 1):
        candidates.append(
            {"cells": ((0, c), (1, c)), "origin": (1, c), "direction": (1, 0)}
        )
        candidates.append(
            {"cells": ((h - 2, c), (h - 1, c)), "origin": (h - 2, c), "direction": (-1, 0)}
        )
    for r in range(1, h - 1):
        candidates.append(
            {"cells": ((r, 0), (r, 1)), "origin": (r, 1), "direction": (0, 1)}
        )
        candidates.append(
            {"cells": ((r, w - 2), (r, w - 1)), "origin": (r, w - 2), "direction": (0, -1)}
        )
    random.shuffle(candidates)
    return candidates


def _pick_emitters(shape: tuple[int, int], count: int) -> list[dict[str, object]] | None:
    chosen = []
    occupied: set[tuple[int, int]] = set()
    forbidden: set[tuple[int, int]] = set()
    for candidate in _emitter_candidates(shape):
        cells = set(candidate["cells"])
        if cells & forbidden:
            continue
        chosen.append(candidate)
        occupied |= cells
        for r, c in cells:
            forbidden.add((r, c))
            for nr, nc in _neighbors4(r, c):
                if in_bounds(shape, nr, nc):
                    forbidden.add((nr, nc))
        if len(chosen) == count:
            return chosen
    return None


def _weighted_choice(items: list[tuple[int, object]]) -> object:
    total = sum(weight for weight, _ in items)
    pick = random.randint(1, total)
    acc = 0
    for weight, value in items:
        acc += weight
        if pick <= acc:
            return value
    return items[-1][1]


def _plan_scene() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    shape = (h, w)

    emitter_count = random.randint(*EMITTER_COUNT_RANGE)
    max_prisms_here = min(MAX_PRISMS, len(PALETTE) - emitter_count)
    if max_prisms_here < 1:
        return None

    emitters = _pick_emitters(shape, emitter_count)
    if emitters is None:
        return None

    emitter_cells = {cell for emitter in emitters for cell in emitter["cells"]}
    reserved_paint: set[tuple[int, int]] = set()
    object_cells = set(emitter_cells)
    prism_spots: list[tuple[int, int]] = []
    terminal_runs: list[list[tuple[int, int]]] = []

    prism_budget = random.randint(1, max_prisms_here)
    max_depth = random.choice(MAX_DEPTH_CHOICES)
    pending = [
        {"origin": tuple(emitter["origin"]), "direction": tuple(emitter["direction"]), "depth": 0}
        for emitter in emitters
    ]
    random.shuffle(pending)

    while pending:
        branch = pending.pop()
        origin = tuple(branch["origin"])
        direction = tuple(branch["direction"])
        depth = int(branch["depth"])

        blocked = object_cells | reserved_paint
        run = _free_run(shape, origin, direction, blocked)
        if len(run) < 1:
            return None

        allow_split = prism_budget > 0 and depth < max_depth
        split_prob = SPLIT_PROB_ROOT if depth == 0 else SPLIT_PROB_CHILD
        prism_choice = None
        if allow_split and random.random() < split_prob:
            min_pre = 1 if len(run) >= 3 else 0
            candidates: list[tuple[int, tuple[int, list[tuple[int, int]], tuple[int, int], tuple[int, int]]]] = []
            for idx in range(min_pre, len(run)):
                prism = run[idx]
                prefix = run[:idx]
                trial_blocked = blocked | set(prefix) | {prism}
                child_a, child_b = _split_directions(direction)
                run_a = _free_run(shape, prism, child_a, trial_blocked)
                run_b = _free_run(shape, prism, child_b, trial_blocked)
                if min(len(run_a), len(run_b)) < 1:
                    continue
                weight = 1 + idx + min(len(run_a), len(run_b))
                candidates.append((weight, (idx, prefix, child_a, child_b)))
            if candidates:
                prism_choice = _weighted_choice(candidates)

        if prism_choice is not None:
            idx, prefix, child_a, child_b = prism_choice
            prism = run[idx]
            reserved_paint.update(prefix)
            prism_spots.append(prism)
            object_cells.add(prism)
            pending.append({"origin": prism, "direction": child_a, "depth": depth + 1})
            pending.append({"origin": prism, "direction": child_b, "depth": depth + 1})
            prism_budget -= 1
            continue

        reserved_paint.update(run)
        terminal_runs.append(run)

    if not prism_spots or len(reserved_paint) < emitter_count + 3:
        return None

    walls: set[tuple[int, int]] = set()
    for run in terminal_runs:
        if len(run) < 2 or random.random() >= WALL_END_PROB:
            continue
        end = run[-1]
        if end in object_cells or end in walls:
            continue
        if end[0] not in {0, h - 1} and end[1] not in {0, w - 1}:
            continue
        reserved_paint.discard(end)
        walls.add(end)

    free_edge = [
        cell
        for cell in _edge_cells(shape)
        if cell not in object_cells and cell not in walls and cell not in reserved_paint
    ]
    random.shuffle(free_edge)
    extra_wall_count = min(random.randint(*EXTRA_WALL_RANGE), len(free_edge))
    walls.update(free_edge[:extra_wall_count])

    colors = random.sample(PALETTE, emitter_count + len(prism_spots))
    emitter_colors = colors[:emitter_count]
    prism_colors = colors[emitter_count:]

    grid = np.full(shape, BG, dtype=int)
    for emitter, color in zip(emitters, emitter_colors):
        for r, c in emitter["cells"]:
            grid[r, c] = color
    for spot, color in zip(prism_spots, prism_colors):
        grid[spot] = color
    for r, c in walls:
        grid[r, c] = GREY

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    if np.count_nonzero(out == GREY) < np.count_nonzero(grid == GREY):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_ATTEMPTS):
        grid = _plan_scene()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid signal-prism ARC input")
