from dataclasses import dataclass
import random

import numpy as np


BG = 0
GEN_RETRIES = 400
GRID_MIN = 18
GRID_MAX = 26
ARROW_COUNT_RANGE = (2, 4)
ARROW_COLORS = list(range(1, 10))

DIR_DATA = {
    "up": {"dir": (-1, 0), "wings": ((-1, -1), (-1, 1)), "trigger_offsets": ((0, -1), (0, 1))},
    "down": {"dir": (1, 0), "wings": ((1, -1), (1, 1)), "trigger_offsets": ((0, -1), (0, 1))},
    "left": {"dir": (0, -1), "wings": ((-1, -1), (1, -1)), "trigger_offsets": ((-1, 0), (1, 0))},
    "right": {"dir": (0, 1), "wings": ((-1, 1), (1, 1)), "trigger_offsets": ((-1, 0), (1, 0))},
}
DIR_NAMES_BY_VEC = {tuple(data["dir"]): name for name, data in DIR_DATA.items()}


@dataclass(frozen=True)
class Arrow:
    color: int
    tail: tuple[int, int]
    wings: tuple[tuple[int, int], tuple[int, int]]
    direction: tuple[int, int]
    direction_name: str


@dataclass(frozen=True)
class Simulation:
    steps: int
    final_tail: tuple[int, int]
    trail: tuple[tuple[tuple[int, int], int], ...]
    stop_reason: str
    trigger_steps: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class Plan:
    color: int
    direction_name: str
    stop_mode: str
    steps: int
    extra_trigger_steps: tuple[int, ...]
    blocker_color: int | None
    event_colors: tuple[int, ...]
    tail: tuple[int, int]
    trigger_landmarks: tuple[tuple[tuple[int, int], int], ...]
    blocker_landmark: tuple[tuple[int, int], int] | None
    zone: frozenset[tuple[int, int]]


def _add(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return a[0] + b[0], a[1] + b[1]


def _mul(v: tuple[int, int], k: int) -> tuple[int, int]:
    return v[0] * k, v[1] * k


def _in_bounds(shape: tuple[int, int], cell: tuple[int, int]) -> bool:
    return 0 <= cell[0] < shape[0] and 0 <= cell[1] < shape[1]


def _neighbors4(cell: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    r, c = cell
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _arrow_cells_from_tail(
    tail: tuple[int, int], wings: tuple[tuple[int, int], tuple[int, int]]
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    return (tail, _add(tail, wings[0]), _add(tail, wings[1]))


def _find_components(g: np.ndarray) -> list[tuple[int, list[tuple[int, int]]]]:
    h, w = g.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[tuple[int, list[tuple[int, int]]]] = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(g[r, c]) == BG:
                continue
            color = int(g[r, c])
            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(g[nr, nc]) == color:
                            visited[nr, nc] = True
                            stack.append((nr, nc))
            components.append((color, coords))
    return components


def _parse_arrow(color: int, coords: list[tuple[int, int]]) -> Arrow:
    assert len(coords) == 3, "arrow sprites must have exactly three pixels"
    for tail in coords:
        wings = [cell for cell in coords if cell != tail]
        rel = [(wr - tail[0], wc - tail[1]) for wr, wc in wings]
        if not all(abs(dr) == 1 and abs(dc) == 1 for dr, dc in rel):
            continue
        dot = rel[0][0] * rel[1][0] + rel[0][1] * rel[1][1]
        if dot != 0:
            continue
        summed = (rel[0][0] + rel[1][0], rel[0][1] + rel[1][1])
        if summed[0] not in (-2, 0, 2) or summed[1] not in (-2, 0, 2):
            continue
        direction = (summed[0] // 2, summed[1] // 2)
        if direction not in DIR_NAMES_BY_VEC:
            continue
        return Arrow(
            color=int(color),
            tail=tail,
            wings=(rel[0], rel[1]),
            direction=direction,
            direction_name=DIR_NAMES_BY_VEC[direction],
        )
    raise AssertionError("invalid arrow geometry")


def _parse_scene(grid: np.ndarray) -> tuple[list[Arrow], dict[tuple[int, int], int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"
    assert set(map(int, np.unique(g))) <= set(range(10)), "ARC colors must be in 0..9"
    assert int(g.min()) >= 0
    assert BG in set(map(int, np.unique(g))), "background must be black"

    arrows: list[Arrow] = []
    landmarks: dict[tuple[int, int], int] = {}
    for color, coords in _find_components(g):
        if len(coords) == 1:
            landmarks[coords[0]] = int(color)
        elif len(coords) == 3:
            arrows.append(_parse_arrow(color, coords))
        else:
            raise AssertionError("every foreground component must be a landmark or a 3-pixel arrow")

    assert arrows, "expected at least one arrow"
    arrow_colors = [arrow.color for arrow in arrows]
    assert len(set(arrow_colors)) == len(arrow_colors), "arrow colors must be distinct"
    assert not (set(arrow_colors) & set(landmarks.values())), "landmark colors must differ from arrow colors"
    return arrows, landmarks


def _adjacent_landmark_color(
    wing_cells: tuple[tuple[int, int], tuple[int, int]],
    landmarks: dict[tuple[int, int], int],
) -> int | None:
    colors = {
        int(landmarks[neighbor])
        for wing in wing_cells
        for neighbor in _neighbors4(wing)
        if neighbor in landmarks
    }
    if not colors:
        return None
    assert len(colors) == 1, "a step cannot be influenced by multiple landmark colors"
    return next(iter(colors))


def _simulate_arrow(
    arrow: Arrow,
    shape: tuple[int, int],
    static_occupied: set[tuple[int, int]],
    landmarks: dict[tuple[int, int], int],
) -> Simulation:
    tail = arrow.tail
    active_color = arrow.color
    trail: list[tuple[tuple[int, int], int]] = []
    trigger_steps: list[tuple[int, int]] = []

    for step in range(1, max(shape) * 2):
        next_tail = _add(tail, arrow.direction)
        next_cells = _arrow_cells_from_tail(next_tail, arrow.wings)
        if not all(_in_bounds(shape, cell) for cell in next_cells):
            return Simulation(step - 1, tail, tuple(trail), "boundary", tuple(trigger_steps))
        if any(cell in static_occupied for cell in next_cells):
            return Simulation(step - 1, tail, tuple(trail), "obstacle", tuple(trigger_steps))

        wing_cells = (_add(next_tail, arrow.wings[0]), _add(next_tail, arrow.wings[1]))
        trigger_color = _adjacent_landmark_color(wing_cells, landmarks)
        if trigger_color is not None:
            if trigger_color != active_color:
                trigger_steps.append((step, trigger_color))
            active_color = trigger_color

        trail.append((tail, active_color))
        tail = next_tail

    raise AssertionError("arrow simulation did not terminate")


def _run_scene(grid: np.ndarray) -> tuple[list[Arrow], dict[tuple[int, int], int], dict[int, Simulation]]:
    g = np.asarray(grid, dtype=int)
    arrows, landmarks = _parse_scene(g)
    occupied = {tuple(cell) for cell in np.argwhere(g != BG)}
    simulations: dict[int, Simulation] = {}
    for arrow in arrows:
        static_occupied = occupied - set(_arrow_cells_from_tail(arrow.tail, arrow.wings))
        simulations[arrow.color] = _simulate_arrow(arrow, g.shape, static_occupied, landmarks)
    return arrows, landmarks, simulations


def main(grid: np.ndarray) -> np.ndarray:
    """Slide each arrow along its pointing direction and color the tail trail with landmark-driven switches.

    Input structure:
    - The background is black.
    - Foreground objects are 3-pixel arrow sprites made of one tail pixel and two diagonal wing pixels, plus isolated single-pixel landmarks.
    - Every arrow has its own color, and landmark colors are different from arrow colors.

    Rule:
    - Move each arrow one cell at a time in the cardinal direction indicated by the opening of its chevron until the next step would leave the grid or hit another non-black input pixel.
    - Each successful step leaves a trail on the cell vacated by the tail; the trail starts in the arrow's own color, but when the moved wings become 4-neighbor adjacent to a landmark, the trail color switches to that landmark's color from that step onward.
    - Clear the original wing positions, keep all landmarks fixed, and redraw every arrow at its final resting position.

    Key insight:
    - The chevron opening determines movement, while the changing trail color depends on wing-to-landmark side adjacency rather than on the tail or on the blocking pixel.
    """
    g = np.asarray(grid, dtype=int)
    arrows, landmarks, simulations = _run_scene(g)

    out = np.full_like(g, BG)
    for cell, color in landmarks.items():
        out[cell] = int(color)

    for arrow in sorted(arrows, key=lambda a: a.color):
        sim = simulations[arrow.color]
        for cell, color in sim.trail:
            assert out[cell] == BG, "trail cells must be unoccupied"
            out[cell] = int(color)

    for arrow in sorted(arrows, key=lambda a: a.color):
        sim = simulations[arrow.color]
        for cell in _arrow_cells_from_tail(sim.final_tail, arrow.wings):
            assert out[cell] == BG, "final arrows must not overlap landmarks or trails"
            out[cell] = int(arrow.color)

    return out


def _dilate(cells: set[tuple[int, int]], shape: tuple[int, int], radius: int = 1) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < shape[0] and 0 <= nc < shape[1]:
                    out.add((nr, nc))
    return out


def _boundary_steps_for_tail(tail: tuple[int, int], direction_name: str, shape: tuple[int, int]) -> int:
    data = DIR_DATA[direction_name]
    cells = _arrow_cells_from_tail(tail, data["wings"])
    if not all(_in_bounds(shape, cell) for cell in cells):
        return -1

    steps = 0
    cur = tail
    while True:
        nxt = _add(cur, data["dir"])
        nxt_cells = _arrow_cells_from_tail(nxt, data["wings"])
        if not all(_in_bounds(shape, cell) for cell in nxt_cells):
            return steps
        steps += 1
        cur = nxt


def _path_footprint(
    tail: tuple[int, int], direction_name: str, steps: int
) -> set[tuple[int, int]]:
    data = DIR_DATA[direction_name]
    cells: set[tuple[int, int]] = set()
    for step in range(steps + 1):
        shifted_tail = _add(tail, _mul(data["dir"], step))
        cells.update(_arrow_cells_from_tail(shifted_tail, data["wings"]))
    return cells


def _trigger_coord(
    tail: tuple[int, int], direction_name: str, step: int, wing_index: int
) -> tuple[int, int]:
    data = DIR_DATA[direction_name]
    shifted_tail = _add(tail, _mul(data["dir"], step))
    wing = _add(shifted_tail, data["wings"][wing_index])
    return _add(wing, data["trigger_offsets"][wing_index])


def _isolated_validation_plan(
    color: int,
    tail: tuple[int, int],
    direction_name: str,
    steps: int,
    extra_trigger_steps: tuple[int, ...],
    event_colors: tuple[int, ...],
    blocker_color: int | None,
    trigger_landmarks: tuple[tuple[tuple[int, int], int], ...],
    blocker_landmark: tuple[tuple[int, int], int] | None,
    shape: tuple[int, int],
) -> bool:
    data = DIR_DATA[direction_name]
    landmarks = {coord: landmark_color for coord, landmark_color in trigger_landmarks}
    if blocker_landmark is not None:
        landmarks[blocker_landmark[0]] = blocker_landmark[1]
    arrow = Arrow(color=color, tail=tail, wings=data["wings"], direction=data["dir"], direction_name=direction_name)
    sim = _simulate_arrow(arrow, shape, set(landmarks), landmarks)
    expected = [(step, event_colors[i]) for i, step in enumerate(extra_trigger_steps)]
    if blocker_color is not None:
        expected.append((steps, blocker_color))
    return sim.steps == steps and sim.trigger_steps == tuple(expected)


def _build_plan(
    shape: tuple[int, int],
    reserved: set[tuple[int, int]],
    *,
    color: int,
    direction_name: str,
    stop_mode: str,
    extra_triggers: int,
    landmark_palette: list[int],
) -> Plan | None:
    min_steps = max(1, extra_triggers + (1 if stop_mode == "obstacle" else 0))
    max_steps = min(6, max(shape) - 2)
    step_choices = list(range(min_steps, max_steps + 1))
    random.shuffle(step_choices)

    tails = [(r, c) for r in range(shape[0]) for c in range(shape[1])]
    random.shuffle(tails)

    for steps in step_choices:
        if stop_mode == "boundary":
            candidate_steps = list(range(1, steps + 1))
        else:
            if steps == 1 and extra_triggers > 0:
                continue
            candidate_steps = list(range(1, steps))
        if len(candidate_steps) < extra_triggers:
            continue

        for _ in range(10):
            extra_trigger_steps = tuple(sorted(random.sample(candidate_steps, extra_triggers)))
            event_count = extra_triggers + (1 if stop_mode == "obstacle" else 0)
            chosen_colors = tuple(random.sample(landmark_palette, event_count))
            blocker_color = chosen_colors[-1] if stop_mode == "obstacle" else None

            for tail in tails:
                boundary_steps = _boundary_steps_for_tail(tail, direction_name, shape)
                if stop_mode == "boundary" and boundary_steps != steps:
                    continue
                if stop_mode == "obstacle" and boundary_steps <= steps:
                    continue

                footprint = _path_footprint(tail, direction_name, steps)
                if footprint & reserved:
                    continue

                trigger_landmarks: list[tuple[tuple[int, int], int]] = []
                used = set(footprint)
                ok = True
                for idx, trigger_step in enumerate(extra_trigger_steps):
                    wing_order = [0, 1]
                    random.shuffle(wing_order)
                    chosen_coord = None
                    for wing_index in wing_order:
                        coord = _trigger_coord(tail, direction_name, trigger_step, wing_index)
                        if not _in_bounds(shape, coord):
                            continue
                        if coord in used or coord in reserved:
                            continue
                        chosen_coord = coord
                        break
                    if chosen_coord is None:
                        ok = False
                        break
                    trigger_landmarks.append((chosen_coord, chosen_colors[idx]))
                    used.add(chosen_coord)
                if not ok:
                    continue

                blocker_landmark = None
                if stop_mode == "obstacle":
                    blocker_coord = _add(tail, _mul(DIR_DATA[direction_name]["dir"], steps + 1))
                    if not _in_bounds(shape, blocker_coord):
                        continue
                    if blocker_coord in used or blocker_coord in reserved:
                        continue
                    blocker_landmark = (blocker_coord, blocker_color)
                    used.add(blocker_coord)

                envelope = set(footprint)
                envelope.update(coord for coord, _color in trigger_landmarks)
                if blocker_landmark is not None:
                    envelope.add(blocker_landmark[0])
                zone = _dilate(envelope, shape, radius=1)
                if zone & reserved:
                    continue

                if not _isolated_validation_plan(
                    color,
                    tail,
                    direction_name,
                    steps,
                    extra_trigger_steps,
                    chosen_colors[:extra_triggers],
                    blocker_color,
                    tuple(trigger_landmarks),
                    blocker_landmark,
                    shape,
                ):
                    continue

                return Plan(
                    color=color,
                    direction_name=direction_name,
                    stop_mode=stop_mode,
                    steps=steps,
                    extra_trigger_steps=extra_trigger_steps,
                    blocker_color=blocker_color,
                    event_colors=chosen_colors,
                    tail=tail,
                    trigger_landmarks=tuple(trigger_landmarks),
                    blocker_landmark=blocker_landmark,
                    zone=frozenset(zone),
                )
    return None


def _render_input(shape: tuple[int, int], plans: list[Plan]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for plan in plans:
        data = DIR_DATA[plan.direction_name]
        for cell in _arrow_cells_from_tail(plan.tail, data["wings"]):
            assert grid[cell] == BG
            grid[cell] = plan.color
        for coord, color in plan.trigger_landmarks:
            assert grid[coord] == BG
            grid[coord] = color
        if plan.blocker_landmark is not None:
            coord, color = plan.blocker_landmark
            assert grid[coord] == BG
            grid[coord] = color
    return grid


def _make_plan_specs(n_arrows: int) -> list[tuple[str, int]]:
    specs: list[tuple[str, int]] = [("boundary", 0), ("obstacle", random.randint(0, 2))]
    while len(specs) < n_arrows:
        stop_mode = random.choice(("boundary", "obstacle"))
        specs.append((stop_mode, random.randint(0, 2)))
    return specs


def _gen_attempt() -> np.ndarray | None:
    shape = (random.randint(GRID_MIN, GRID_MAX), random.randint(GRID_MIN, GRID_MAX))
    n_arrows = random.randint(*ARROW_COUNT_RANGE)
    arrow_colors = random.sample(ARROW_COLORS, n_arrows)
    landmark_palette = [color for color in ARROW_COLORS if color not in arrow_colors]

    specs = _make_plan_specs(n_arrows)
    directions = random.sample(list(DIR_DATA), 2)
    while len(directions) < n_arrows:
        directions.append(random.choice(list(DIR_DATA)))

    plan_jobs = []
    for idx in range(n_arrows):
        plan_jobs.append(
            {
                "color": arrow_colors[idx],
                "direction_name": directions[idx],
                "stop_mode": specs[idx][0],
                "extra_triggers": specs[idx][1],
            }
        )

    random.shuffle(plan_jobs)
    plan_jobs.sort(
        key=lambda item: (
            0 if item["stop_mode"] == "obstacle" else 1,
            -item["extra_triggers"],
        )
    )

    reserved: set[tuple[int, int]] = set()
    plans: list[Plan] = []
    for job in plan_jobs:
        plan = _build_plan(shape, reserved, landmark_palette=landmark_palette, **job)
        if plan is None:
            return None
        reserved.update(plan.zone)
        plans.append(plan)

    grid = _render_input(shape, plans)
    arrows, landmarks, simulations = _run_scene(grid)
    if len(arrows) != n_arrows:
        return None

    planned_by_color = {plan.color: plan for plan in plans}
    if {arrow.color for arrow in arrows} != set(planned_by_color):
        return None

    if len({arrow.direction_name for arrow in arrows}) < 2:
        return None

    actual_stop_reasons = {sim.stop_reason for sim in simulations.values()}
    if actual_stop_reasons != {"boundary", "obstacle"}:
        return None

    switch_counts = []
    for arrow in arrows:
        plan = planned_by_color[arrow.color]
        sim = simulations[arrow.color]
        expected_events = [(step, plan.event_colors[i]) for i, step in enumerate(plan.extra_trigger_steps)]
        if plan.blocker_color is not None:
            expected_events.append((plan.steps, plan.blocker_color))
        if sim.steps != plan.steps or sim.stop_reason != plan.stop_mode:
            return None
        if sim.trigger_steps != tuple(expected_events):
            return None
        switch_counts.append(len(sim.trigger_steps))

    if 0 not in switch_counts or max(switch_counts) < 1:
        return None

    _ = main(grid)
    if np.array_equal(grid, main(grid)):
        return None
    if len(landmarks) < n_arrows:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arrow-landmark ARC input")
