from __future__ import annotations

import itertools
import random
from collections import deque

import numpy as np


BG = 0
RED = 2
GREEN = 3
GREY = 5
ORANGE = 7

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GRID_MIN = 16
GRID_MAX = 22
SEED_COUNTS = (2, 3)
ORANGE_COUNT_RANGE = (2, 4)
FIXED_RED_COUNT_RANGE = (3, 5)
SEED_GAP = 3
BRANCH_PROBABILITY = 0.55
GEN_RETRIES = 400


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _coords_to_set(coords: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, p)) for p in coords}


def _is_domino(coords: np.ndarray) -> bool:
    if int(coords.shape[0]) != 2:
        return False
    rows = sorted(int(v) for v in coords[:, 0])
    cols = sorted(int(v) for v in coords[:, 1])
    return (rows[0] == rows[1] and cols[1] - cols[0] == 1) or (
        cols[0] == cols[1] and rows[1] - rows[0] == 1
    )


def _can_move_down(coords: np.ndarray, occupied: np.ndarray, h: int) -> bool:
    shifted = coords + np.array([1, 0], dtype=int)
    if int(shifted[:, 0].max()) >= h:
        return False
    for r, c in shifted:
        rr = int(r)
        cc = int(c)
        if occupied[rr, cc]:
            return False
        if rr + 1 < h and occupied[rr + 1, cc]:
            return False
    return True


def _drop_orange_objects(grid: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    h, _ = g.shape
    base = g.copy()
    orange_mask = base == ORANGE
    base[orange_mask] = BG
    occupied = (base == RED) | (base == GREY)

    components = _components_of_color(orange_mask.astype(int), 1)
    components.sort(key=lambda comp: (int(comp[:, 0].max()), int(comp[:, 1].max())), reverse=True)

    settled: list[np.ndarray] = []
    for coords in components:
        cur = coords.copy()
        while _can_move_down(cur, occupied, h):
            cur = cur + np.array([1, 0], dtype=int)
        base[cur[:, 0], cur[:, 1]] = ORANGE
        occupied[cur[:, 0], cur[:, 1]] = True
        settled.append(cur)
    return base, settled


def _simulate_rays(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w

    def is_obstacle(r: int, c: int) -> bool:
        return in_bounds(r, c) and int(g[r, c]) == RED

    def paint(r: int, c: int) -> None:
        if in_bounds(r, c) and int(g[r, c]) == BG:
            out[r, c] = GREEN

    active_obstacles: set[tuple[int, int]] = set()
    local_hits: set[tuple[int, int, tuple[tuple[int, int], ...]]] = set()
    full_hits: set[tuple[int, int, tuple[tuple[int, int], ...]]] = set()
    full_beams: set[tuple[int, int, tuple[tuple[int, int], ...]]] = set()

    def hit_obstacle(
        r: int,
        c: int,
        *,
        full: bool,
        blocked_entries: frozenset[tuple[int, int]] = frozenset(),
    ) -> list[tuple[int, int, frozenset[tuple[int, int]]]]:
        if full:
            key = (r, c, tuple(sorted(blocked_entries)))
            if key in full_hits:
                return []
            full_hits.add(key)
        else:
            key = (r, c, tuple(sorted(blocked_entries)))
            if key in local_hits:
                return []
            local_hits.add(key)

        active_obstacles.add((r, c))
        side_obstacles: list[tuple[int, int]] = []
        side_open = False

        for dr in (-1, 1):
            rr = r + dr
            if not 0 <= rr < h:
                continue
            entry = (rr, c - 1)
            if entry in blocked_entries:
                continue
            if c - 1 >= 0 and is_obstacle(*entry) and entry not in active_obstacles:
                continue
            side_open = True
            for cc in range(max(0, c - 1), min(w, c + 2)):
                paint(rr, cc)
                if is_obstacle(rr, cc) and (rr, cc) != (r, c):
                    side_obstacles.append((rr, cc))

        if not full and side_open:
            paint(r, c + 1)

        if full and is_obstacle(r - 1, c + 1) and is_obstacle(r + 1, c + 1):
            parent = frozenset({(r, c)})
            return [(r - 1, c, parent), (r + 1, c, parent)]

        for rr, cc in side_obstacles:
            hit_obstacle(rr, cc, full=False, blocked_entries=frozenset({(r, c)}))
        return []

    def run_full_beam(
        r: int,
        start_c: int,
        blocked_entries: frozenset[tuple[int, int]] = frozenset(),
    ) -> None:
        if not 0 <= r < h:
            return
        key = (r, start_c, tuple(sorted(blocked_entries)))
        if key in full_beams:
            return
        full_beams.add(key)

        c = start_c + 1
        while c < w:
            cell = int(g[r, c])
            if cell == BG:
                out[r, c] = GREEN
                c += 1
                continue
            if cell == GREEN:
                c += 1
                continue
            if cell == RED:
                branches = hit_obstacle(r, c, full=True, blocked_entries=blocked_entries)
                if branches:
                    for rr, branch_start_c, branch_blocked in branches:
                        run_full_beam(rr, branch_start_c, branch_blocked)
                    return
            c += 1

    seeds = np.argwhere(g == GREEN)
    for r, c in seeds:
        if int(c) == 0:
            run_full_beam(int(r), 0)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop the floating orange dominoes, turn them into red blockers, then fire left-edge green rays through the new obstacle field.

    Input structure:
    - The background is black, the bottom row is a solid grey floor, and 2-3 green seed pixels sit on the left edge.
    - Red single-pixel blockers are fixed in the interior, while orange 1x2 or 2x1 connected pieces float above open space.
    - Orange pieces are not on the floor; after falling they join the red blockers as one unified obstacle set.

    Rule:
    - Drop each orange 4-connected object downward in descending row order, preserving shape and leaving a one-row cushion above the grey floor, any fixed red blocker, or any already settled orange piece.
    - Replace every settled orange pixel with red and erase the original orange positions to black.
    - From every green seed, extend a horizontal green ray rightward through black cells; hitting a red blocker paints the open three-cell side wakes above and below it, and a blocker with red cells at both forward diagonals terminates that beam and starts two new rightward beams on the adjacent rows.

    Key insight:
    - The ray phase cannot be solved from the input obstacle layout alone, because gravity first creates new red blockers whose landing positions decide every later wake and branch.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    settled_grid, _ = _drop_orange_objects(g)
    phase = settled_grid.copy()
    phase[phase == ORANGE] = RED
    return _simulate_rays(phase)


def _pick_seed_rows(h: int, count: int) -> tuple[int, ...] | None:
    rows = list(range(3, h - 5))
    options = [
        combo
        for combo in itertools.combinations(rows, count)
        if all(b - a >= SEED_GAP for a, b in zip(combo, combo[1:]))
    ]
    if not options:
        return None
    return tuple(random.choice(options))


def _well_spaced(
    coords: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    h: int,
    w: int,
    *,
    allow_neighbors: set[tuple[tuple[int, int], tuple[int, int]]] | None = None,
) -> bool:
    allowed = allow_neighbors or set()
    for cell in coords:
        if cell in occupied:
            return False
    for r, c in coords:
        for nr, nc in _neighbors4(r, c):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            pair = tuple(sorted(((r, c), (nr, nc))))
            if (nr, nc) in occupied and pair not in allowed:
                return False
    return True


def _object_coords(shape: str, anchor_r: int, anchor_c: int) -> set[tuple[int, int]]:
    if shape == "H":
        return {(anchor_r, anchor_c), (anchor_r, anchor_c + 1)}
    return {(anchor_r, anchor_c), (anchor_r + 1, anchor_c)}


def _candidate_target_object(
    h: int,
    w: int,
    seed_row: int,
    occupied: set[tuple[int, int]],
    branch_parent: tuple[int, int] | None,
) -> tuple[set[tuple[int, int]], tuple[int, int]] | None:
    shapes = ["H", "V"]
    random.shuffle(shapes)
    for shape in shapes:
        if shape == "H":
            cols = list(range(2, w - 2))
        else:
            cols = list(range(2, w - 1))
        random.shuffle(cols)
        for c in cols:
            if branch_parent is not None and seed_row == branch_parent[0] and c <= branch_parent[1] + 2:
                continue
            if shape == "H":
                coords = _object_coords("H", seed_row, c)
                support_choices = [(seed_row + 2, c), (seed_row + 2, c + 1)]
            else:
                coords = _object_coords("V", seed_row - 1, c)
                support_choices = [(seed_row + 2, c)]
            if max(r for r, _ in coords) >= h - 2:
                continue
            for support in support_choices:
                support_set = {support}
                if not _well_spaced(coords, occupied | support_set, h, w):
                    continue
                if not _well_spaced(support_set, occupied, h, w):
                    continue
                return coords, support
    return None


def _candidate_floor_object(
    h: int,
    w: int,
    occupied: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    candidates: list[set[tuple[int, int]]] = []
    for c in range(2, w - 2):
        candidates.append(_object_coords("H", h - 3, c))
    for c in range(2, w - 1):
        candidates.append(_object_coords("V", h - 4, c))
    random.shuffle(candidates)
    for coords in candidates:
        if _well_spaced(coords, occupied, h, w):
            return coords
    return None


def _lift_options(
    coords: set[tuple[int, int]],
    h: int,
    fixed_occupied: np.ndarray,
    occupied_input: set[tuple[int, int]],
) -> list[set[tuple[int, int]]]:
    upper_limit = h // 2 - 1
    arr = np.array(sorted(coords), dtype=int)
    options: list[set[tuple[int, int]]] = []
    for shift in range(1, h):
        lifted = arr.copy()
        lifted[:, 0] -= shift
        if int(lifted[:, 0].min()) < 1:
            break
        if int(lifted[:, 0].max()) > upper_limit:
            continue
        lifted_set = _coords_to_set(lifted)
        if any(cell in occupied_input for cell in lifted_set):
            continue
        if not _well_spaced(lifted_set, occupied_input, h, fixed_occupied.shape[1]):
            continue
        if not _can_move_down(lifted, fixed_occupied, h):
            continue
        options.append(lifted_set)
    return options


def _build_input_grid(
    h: int,
    w: int,
    seed_rows: tuple[int, ...],
    fixed_reds: set[tuple[int, int]],
    orange_objects: list[set[tuple[int, int]]],
) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    grid[-1, :] = GREY
    for r in seed_rows:
        grid[r, 0] = GREEN
    for r, c in fixed_reds:
        grid[r, c] = RED
    for obj in orange_objects:
        for r, c in obj:
            grid[r, c] = ORANGE
    return grid


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    seed_rows = _pick_seed_rows(h, random.choice(SEED_COUNTS))
    if seed_rows is None:
        return None

    branch_parent: tuple[int, int] | None = None
    fixed_reds: set[tuple[int, int]] = set()
    branch_mode = random.random() < BRANCH_PROBABILITY
    if branch_mode:
        eligible = [r for r in seed_rows if 1 <= r < h - 3]
        if not eligible:
            branch_mode = False
        else:
            r = random.choice(eligible)
            c = random.randint(4, w - 5)
            branch_parent = (r, c)
            fixed_reds |= {(r, c), (r - 1, c + 1), (r + 1, c + 1)}

    interaction_choices = [r for r in seed_rows if r != (branch_parent[0] if branch_parent else None)]
    if not interaction_choices:
        interaction_choices = list(seed_rows)
    interaction_row = random.choice(interaction_choices)

    occupied_final = set(fixed_reds)
    target = _candidate_target_object(h, w, interaction_row, occupied_final, branch_parent)
    if target is None:
        return None
    target_object, target_support = target
    fixed_reds.add(target_support)
    occupied_final = set(fixed_reds) | target_object

    orange_count = random.randint(*ORANGE_COUNT_RANGE)
    final_objects = [target_object]
    for _ in range(orange_count - 1):
        candidate = _candidate_floor_object(h, w, occupied_final)
        if candidate is None:
            return None
        final_objects.append(candidate)
        occupied_final |= candidate

    target_red_count = random.randint(max(len(fixed_reds), FIXED_RED_COUNT_RANGE[0]), FIXED_RED_COUNT_RANGE[1])

    for row in seed_rows:
        if len(fixed_reds) >= target_red_count:
            break
        if any(r == row for r, _ in fixed_reds) or any(r == row for obj in final_objects for r, _ in obj):
            continue
        cols = list(range(3, w - 2))
        random.shuffle(cols)
        for c in cols:
            if branch_parent is not None and row == branch_parent[0] and c < branch_parent[1]:
                continue
            cell = {(row, c)}
            if _well_spaced(cell, fixed_reds | set().union(*final_objects), h, w):
                fixed_reds.add((row, c))
                break

    while len(fixed_reds) < target_red_count:
        row = random.randint(h // 2, h - 4)
        col = random.randint(2, w - 2)
        if branch_parent is not None and row == branch_parent[0] and col < branch_parent[1]:
            continue
        cell = {(row, col)}
        if _well_spaced(cell, fixed_reds | set().union(*final_objects), h, w):
            fixed_reds.add((row, col))

    fixed_only = np.zeros((h, w), dtype=bool)
    fixed_only[-1, :] = True
    for r, c in fixed_reds:
        fixed_only[r, c] = True

    occupied_input = set(fixed_reds) | {(r, 0) for r in seed_rows}
    lifted_objects: list[set[tuple[int, int]]] = []
    for obj in sorted(final_objects, key=lambda s: min(r for r, _ in s)):
        options = _lift_options(obj, h, fixed_only, occupied_input)
        if not options:
            return None
        lifted = random.choice(options)
        lifted_objects.append(lifted)
        occupied_input |= lifted

    grid = _build_input_grid(h, w, seed_rows, fixed_reds, lifted_objects)

    orange_components = _components_of_color(grid, ORANGE)
    if len(orange_components) != orange_count or not all(_is_domino(comp) for comp in orange_components):
        return None
    if any(int(comp[:, 0].max()) > h // 2 - 1 for comp in orange_components):
        return None
    if not (FIXED_RED_COUNT_RANGE[0] <= int(np.sum(grid == RED)) <= FIXED_RED_COUNT_RANGE[1]):
        return None

    settled_grid, settled_objects = _drop_orange_objects(grid)
    settled_union = {(r, c) for obj in settled_objects for r, c in _coords_to_set(obj)}
    planned_union = {(r, c) for obj in final_objects for r, c in obj}
    if settled_union != planned_union:
        return None
    if not any(r in seed_rows for r, _ in settled_union):
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None

    if branch_parent is not None:
        r, c = branch_parent
        if c + 2 >= w or int(out[r - 1, c + 2]) != GREEN or int(out[r + 1, c + 2]) != GREEN:
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gravity-ray ARC input")
