from __future__ import annotations

import numpy as np

from arc_utils import find_components, get_bg
import random


MARKER_COLOR = 9
DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _outward_direction(g: np.ndarray, coords: np.ndarray) -> tuple[int, int]:
    H, W = g.shape
    directions: set[tuple[int, int]] = set()
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        for dr, dc in DIRECTIONS:
            nr = rr + dr
            nc = cc + dc
            if 0 <= nr < H and 0 <= nc < W and int(g[nr, nc]) == MARKER_COLOR:
                directions.add((-dr, -dc))

    assert len(directions) == 1, "each moving object must touch maroon markers on exactly one side"
    return directions.pop()


def _steps_to_edge(shape: tuple[int, int], coords: np.ndarray, direction: tuple[int, int]) -> int:
    H, W = shape
    dr, dc = direction
    assert (dr == 0) != (dc == 0), "movement direction must be cardinal"
    if dr < 0:
        return int(coords[:, 0].min())
    if dr > 0:
        return int(H - 1 - coords[:, 0].max())
    if dc < 0:
        return int(coords[:, 1].min())
    assert dc > 0
    return int(W - 1 - coords[:, 1].max())


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push every colored object touching maroon markers straight outward until it reaches the grid edge.

    Input structure:
    - The background is the dominant color.
    - Maroon pixels are fixed markers.
    - Every other foreground 4-connected object touches the marker pixels on exactly one cardinal side.

    Rule:
    - Clear each non-background, non-maroon object from its original position.
    - Translate it in the direction away from its maroon contact until the object reaches the grid boundary.
    - Keep the maroon marker pixels unchanged.

    Key insight:
    - The maroon contact side is a launcher: it determines a unique outward ray for the whole object, and the edge of the grid determines the translation distance.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    assert bg != MARKER_COLOR
    assert bool(np.any(g == MARKER_COLOR))

    marker_components = find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=[MARKER_COLOR],
        return_format="dict",
    )
    assert 1 <= len(marker_components) <= 3, "maroon markers must form 1-3 moderate components"
    assert sum(int(comp["area"]) for comp in marker_components) <= 17, "maroon marker area must be moderate"
    assert all(int(comp["area"]) <= 9 for comp in marker_components), "each maroon component must be moderate"

    out = g.copy()
    placements: list[tuple[int, np.ndarray]] = []
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        color = int(comp["color"])
        if color == MARKER_COLOR:
            continue

        coords = comp["coords"].astype(int, copy=False)
        direction = _outward_direction(g, coords)
        steps = _steps_to_edge(g.shape, coords, direction)
        assert steps <= 8, "push distance must stay within the observed moderate range"
        offset = np.array(direction, dtype=int) * steps
        shifted = coords + offset
        assert bool(
            np.all(
                (0 <= shifted[:, 0])
                & (shifted[:, 0] < g.shape[0])
                & (0 <= shifted[:, 1])
                & (shifted[:, 1] < g.shape[1])
            )
        )
        own = {tuple(coord) for coord in coords.tolist()}
        for step in range(1, steps + 1):
            swept = coords + np.array(direction, dtype=int) * step
            for rr, cc in swept:
                loc = (int(rr), int(cc))
                assert loc in own or int(g[loc]) == bg, "object movement path must be unobstructed"

        out[coords[:, 0], coords[:, 1]] = bg
        placements.append((color, shifted))

    assert placements
    for color, coords in placements:
        assert bool(np.all(out[coords[:, 0], coords[:, 1]] == bg)), "moved objects must not collide"
        out[coords[:, 0], coords[:, 1]] = color

    return out

# --------------- generator (attempt 1 refined) ---------------

# Generated grid colors.
BG = 7
MAROON = 9
ALL_COLORS = [0, 1, 2, 3, 4, 5, 6, 8]
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Canvas and maroon marker bounds.
GRID_SIZE_MIN = 9
GRID_SIZE_MAX = 16
MAROON_COMPONENT_MIN = 1
MAROON_COMPONENT_MAX = 3
MAROON_TOTAL_AREA_MAX = 17
MAROON_COMPONENT_AREA_MAX = 9
INTERIOR_PADDING = 1
MAROON_PLACEMENT_TRIES = 80

# Moving object bounds and push constraints.
OBJECT_COUNT_MIN = 3
OBJECT_COUNT_MAX = 7
OBJECT_SIZE_MIN = 1
OBJECT_SIZE_MAX = 5
PUSH_DISTANCE_MAX = 8
ZERO_STEP_PROBABILITY = 0.25
ZERO_STEP_ALLOWANCE = 1

# Retry limits for object placement and whole-grid generation.
OBJECT_SHAPE_TRIES = 20
CONTACT_CANDIDATE_LIMIT = 5
GENERATE_RETRIES = 300


def _grow_polyomino(size, rng):
    """Grow a 4-connected polyomino of given size, normalized to (0, 0)."""
    cells = {(0, 0)}
    frontier = [(0, 0)]
    while len(cells) < size:
        r, c = rng.choice(frontier)
        nbrs = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
        rng.shuffle(nbrs)
        grown = False
        for nr, nc in nbrs:
            if (nr, nc) not in cells:
                cells.add((nr, nc))
                frontier.append((nr, nc))
                grown = True
                break
        if not grown:
            frontier.remove((r, c))
            if not frontier:
                break

    min_r = min(r for r, c in cells)
    min_c = min(c for r, c in cells)
    return [(r - min_r, c - min_c) for r, c in cells]


def _gen_steps_to_edge(N, coords, direction):
    arr = np.array(coords, dtype=int)
    dr, dc = direction
    if dr == -1:
        return int(arr[:, 0].min())
    if dr == 1:
        return int(N - 1 - arr[:, 0].max())
    if dc == -1:
        return int(arr[:, 1].min())
    return int(N - 1 - arr[:, 1].max())


def _swept_cells(N, coords, direction):
    dr, dc = direction
    steps = _gen_steps_to_edge(N, coords, direction)
    own = set(coords)
    swept = set()
    for step in range(1, steps + 1):
        for r, c in coords:
            loc = (r + dr * step, c + dc * step)
            if loc not in own:
                swept.add(loc)
    return swept


def _paths_are_clear(inp, objects):
    N = inp.shape[0]
    for _, inp_coords, _, direction in objects:
        for r, c in _swept_cells(N, inp_coords, direction):
            if inp[r, c] != BG:
                return False
    return True


def _paint_cells(grid, coords, color):
    for r, c in coords:
        grid[r, c] = color


def _place_maroon_components(rng, grid, num_maroon):
    """Place separated maroon components; returns (components, occupied) or None."""
    N = grid.shape[0]
    maroon_components = []
    all_maroon = set()
    total_maroon = 0

    for idx in range(num_maroon):
        remaining_components = num_maroon - idx - 1
        max_size = min(
            MAROON_COMPONENT_AREA_MAX,
            MAROON_TOTAL_AREA_MAX - total_maroon - remaining_components,
        )
        if max_size < 1:
            return None

        size = rng.randint(1, max_size)
        cells = _grow_polyomino(size, rng)
        h = max(r for r, c in cells) + 1
        w = max(c for r, c in cells) + 1
        if h > N - 2 or w > N - 2:
            return None

        placed = _try_place_maroon_component(rng, grid, cells, all_maroon)
        if placed is None:
            return None

        _paint_cells(grid, placed, MAROON)
        maroon_components.append(placed)
        all_maroon.update(placed)
        total_maroon += size

    return maroon_components, all_maroon


def _try_place_maroon_component(rng, grid, cells, all_maroon):
    N = grid.shape[0]
    h = max(r for r, c in cells) + 1
    w = max(c for r, c in cells) + 1
    for _ in range(MAROON_PLACEMENT_TRIES):
        r0 = rng.randint(INTERIOR_PADDING, N - INTERIOR_PADDING - h)
        c0 = rng.randint(INTERIOR_PADDING, N - INTERIOR_PADDING - w)
        abs_cells = [(r + r0, c + c0) for r, c in cells]
        if not all(grid[r, c] == BG for r, c in abs_cells):
            continue
        if any((r + dr, c + dc) in all_maroon for r, c in abs_cells for dr, dc in DIRS):
            continue
        return abs_cells
    return None


def _available_slots(N, maroon_components, all_maroon):
    """Return (component index, outward side, contact cells) slots around markers."""
    slots = []
    for mi, m_cells in enumerate(maroon_components):
        for dr, dc in DIRS:
            contacts = []
            for r, c in m_cells:
                nr, nc = r + dr, c + dc
                if 0 <= nr < N and 0 <= nc < N and (nr, nc) not in all_maroon:
                    contacts.append((nr, nc))
            if contacts:
                slots.append((mi, (dr, dc), contacts))
    return slots


def _choose_object_slots(rng, slots):
    target_objects = rng.randint(OBJECT_COUNT_MIN, OBJECT_COUNT_MAX)
    if len(slots) < target_objects:
        target_objects = len(slots)
    if target_objects < OBJECT_COUNT_MIN:
        return None
    rng.shuffle(slots)
    return slots[:target_objects]


def _touching_maroon_sides(coords, all_maroon):
    touching = set()
    for r, c in coords:
        for dr, dc in DIRS:
            if (r + dr, c + dc) in all_maroon:
                touching.add((dr, dc))
    return touching


def _coords_from_contact(obj_cells, contact, rng):
    anchor_idx = rng.randint(0, len(obj_cells) - 1)
    ar, ac = obj_cells[anchor_idx]
    offset_r = contact[0] - ar
    offset_c = contact[1] - ac
    return [(r + offset_r, c + offset_c) for r, c in obj_cells]


def _coords_in_bounds(N, coords):
    return all(0 <= r < N and 0 <= c < N for r, c in coords)


def _translated_coords(coords, direction, steps):
    dr, dc = direction
    return [(r + dr * steps, c + dc * steps) for r, c in coords]


def _pick_unused_color(rng, used_colors):
    remaining = [color for color in ALL_COLORS if color not in used_colors]
    color = rng.choice(remaining)
    used_colors.add(color)
    return color


def _try_place_object(
    rng,
    N,
    slot,
    output,
    input_occupied,
    reserved_swept,
    all_maroon,
    color,
    zero_step_budget,
):
    """Try one side slot; mutates occupancy/output only after accepting a candidate."""
    _, direction, contact_cells = slot
    dr, dc = direction
    obj_size = rng.randint(OBJECT_SIZE_MIN, OBJECT_SIZE_MAX)

    for _ in range(OBJECT_SHAPE_TRIES):
        obj_cells = _grow_polyomino(obj_size, rng)
        rng.shuffle(contact_cells)
        for contact in contact_cells[:CONTACT_CANDIDATE_LIMIT]:
            inp_coords = _coords_from_contact(obj_cells, contact, rng)
            if not _coords_in_bounds(N, inp_coords):
                continue

            inp_set = set(inp_coords)
            if inp_set & input_occupied:
                continue
            if inp_set & reserved_swept:
                continue
            if _touching_maroon_sides(inp_coords, all_maroon) != {(-dr, -dc)}:
                continue

            steps = _gen_steps_to_edge(N, inp_coords, direction)
            if steps > PUSH_DISTANCE_MAX:
                continue
            if steps == 0 and zero_step_budget <= 0:
                continue

            out_coords = _translated_coords(inp_coords, direction, steps)
            swept = _swept_cells(N, inp_coords, direction)
            if swept & input_occupied:
                continue
            if not all(output[r, c] == BG for r, c in out_coords):
                continue

            _paint_cells(output, out_coords, color)
            input_occupied.update(inp_set)
            reserved_swept.update(swept)
            if steps == 0:
                zero_step_budget -= 1
            return (color, inp_coords, out_coords, direction), zero_step_budget

    return None, zero_step_budget


def _build_objects(rng, grid, all_maroon, chosen_slots):
    """Place edge-output objects and retain their derived input coordinates."""
    N = grid.shape[0]
    output = grid.copy()
    used_colors = set()
    objects = []
    input_occupied = set(all_maroon)
    reserved_swept = set()
    zero_step_budget = ZERO_STEP_ALLOWANCE if rng.random() < ZERO_STEP_PROBABILITY else 0

    for slot in chosen_slots:
        if len(used_colors) >= len(ALL_COLORS):
            break
        color = _pick_unused_color(rng, used_colors)
        placed, zero_step_budget = _try_place_object(
            rng,
            N,
            slot,
            output,
            input_occupied,
            reserved_swept,
            all_maroon,
            color,
            zero_step_budget,
        )
        if placed is not None:
            objects.append(placed)

    if len(objects) < OBJECT_COUNT_MIN or len(objects) > OBJECT_COUNT_MAX:
        return None
    return output, objects


def _build_input_grid(grid, objects):
    inp = grid.copy()
    for color, inp_coords, _, _ in objects:
        if not all(inp[r, c] == BG for r, c in inp_coords):
            return None
        _paint_cells(inp, inp_coords, color)
    return inp


def _solver_matches_output(inp, output):
    try:
        return np.array_equal(main(inp), output)
    except Exception:
        return False


def _gen_attempt():
    rng = random.Random()
    N = rng.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    grid = np.full((N, N), BG, dtype=int)

    num_maroon = rng.randint(MAROON_COMPONENT_MIN, MAROON_COMPONENT_MAX)
    placed_markers = _place_maroon_components(rng, grid, num_maroon)
    if placed_markers is None:
        return None
    maroon_components, all_maroon = placed_markers

    slots = _available_slots(N, maroon_components, all_maroon)
    chosen_slots = _choose_object_slots(rng, slots)
    if chosen_slots is None:
        return None

    built = _build_objects(rng, grid, all_maroon, chosen_slots)
    if built is None:
        return None
    output, objects = built

    inp = _build_input_grid(grid, objects)
    if inp is None:
        return None
    if not _paths_are_clear(inp, objects):
        return None
    if not _solver_matches_output(inp, output):
        return None

    return inp


def generate_input():
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {GENERATE_RETRIES} attempts")
