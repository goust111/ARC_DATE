from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


WARM_COLORS = (2, 7, 4)
COOL_COLORS = (1, 3, 8)
VALID_COLORS = frozenset(WARM_COLORS + COOL_COLORS)
DIRS = ((-1, -1), (-1, 1), (1, -1), (1, 1))

GRID_H_RANGE = (12, 15)
GRID_W_RANGE = (12, 15)
SEED_COUNT_WEIGHTS = ((3, 3), (4, 6), (5, 3))
MAX_ATTEMPTS = 800


@dataclass(frozen=True)
class Ray:
    id: int
    seed_id: int
    start: tuple[int, int]
    color: int
    family: str
    direction: tuple[int, int]
    path: tuple[tuple[int, int], ...]
    steps: dict[tuple[int, int], int]


@dataclass(frozen=True)
class CollisionEvent:
    event_time: int
    ray_a: int
    ray_b: int
    cell: tuple[int, int]
    step_a: int
    step_b: int
    color: int


def _family_of(color: int) -> str:
    if color in WARM_COLORS:
        return "warm"
    if color in COOL_COLORS:
        return "cool"
    raise AssertionError(f"unsupported seed color: {color}")


def _is_in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _seed_positions(grid: np.ndarray) -> list[tuple[int, int, int]]:
    coords = np.argwhere(grid != 0)
    seeds = [(int(r), int(c), int(grid[r, c])) for r, c in coords]
    seeds.sort()
    return seeds


def _build_rays(
    seeds: list[tuple[int, int, int]],
    shape: tuple[int, int],
) -> list[Ray]:
    rays: list[Ray] = []
    for seed_id, (r0, c0, color) in enumerate(seeds):
        family = _family_of(color)
        for dr, dc in DIRS:
            rr, cc = r0 + dr, c0 + dc
            step = 1
            path: list[tuple[int, int]] = []
            steps: dict[tuple[int, int], int] = {}
            while _is_in_bounds(shape, rr, cc):
                cell = (rr, cc)
                path.append(cell)
                steps[cell] = step
                rr += dr
                cc += dc
                step += 1
            rays.append(
                Ray(
                    id=len(rays),
                    seed_id=seed_id,
                    start=(r0, c0),
                    color=color,
                    family=family,
                    direction=(dr, dc),
                    path=tuple(path),
                    steps=steps,
                )
            )
    return rays


def _collision_color(ray_a: Ray, ray_b: Ray, step_a: int, step_b: int) -> int:
    if step_a < step_b:
        return ray_a.color
    if step_b < step_a:
        return ray_b.color
    return ray_a.color if ray_a.family == "warm" else ray_b.color


def _best_cross_family_event(
    ray_a: Ray,
    ray_b: Ray,
    *,
    reject_ambiguous: bool,
) -> CollisionEvent | None:
    if ray_a.family == ray_b.family:
        return None
    common = set(ray_a.steps) & set(ray_b.steps)
    if not common:
        return None

    ranked = []
    for cell in common:
        step_a = ray_a.steps[cell]
        step_b = ray_b.steps[cell]
        ranked.append((max(step_a, step_b), cell, step_a, step_b))
    ranked.sort(key=lambda item: (item[0], item[1][0], item[1][1], item[2], item[3]))

    best_time = ranked[0][0]
    best = [item for item in ranked if item[0] == best_time]
    if reject_ambiguous and len(best) != 1:
        return None

    _, cell, step_a, step_b = best[0]
    return CollisionEvent(
        event_time=best_time,
        ray_a=ray_a.id,
        ray_b=ray_b.id,
        cell=cell,
        step_a=step_a,
        step_b=step_b,
        color=_collision_color(ray_a, ray_b, step_a, step_b),
    )


def _analyze(
    grid: np.ndarray,
    *,
    reject_ambiguous: bool,
) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))).issubset(VALID_COLORS | {0}), "unexpected colors"

    seeds = _seed_positions(g)
    assert 3 <= len(seeds) <= 5, "expected 3 to 5 seeds"

    families = {_family_of(color) for _r, _c, color in seeds}
    assert families == {"warm", "cool"}, "both color families must appear"

    seed_cells = [(r, c) for r, c, _color in seeds]
    for i, a in enumerate(seed_cells):
        for b in seed_cells[i + 1 :]:
            assert _chebyshev(a, b) >= 3, "seeds must be isolated"

    rays = _build_rays(seeds, g.shape)
    seed_cell_set = set(seed_cells)
    for ray in rays:
        assert not (seed_cell_set & set(ray.path)), "ray path may not pass through another seed"

    arrivals: dict[tuple[tuple[int, int], int], list[Ray]] = {}
    for ray in rays:
        for cell, step in ray.steps.items():
            arrivals.setdefault((cell, step), []).append(ray)

    for (_cell, _step), arriving in arrivals.items():
        if reject_ambiguous and len(arriving) > 2:
            raise AssertionError("three rays may not arrive at one cell simultaneously")
        if reject_ambiguous and len(arriving) == 2 and arriving[0].family == arriving[1].family:
            raise AssertionError("same-family equal-time overlap is ambiguous")

    pairwise_seed_links = {
        (i, j): False
        for i in range(len(seeds))
        for j in range(i + 1, len(seeds))
        if _family_of(seeds[i][2]) != _family_of(seeds[j][2])
    }
    candidate_events: list[CollisionEvent] = []
    min_event_times_by_ray: dict[int, list[int]] = {}
    for i, ray_a in enumerate(rays):
        for ray_b in rays[i + 1 :]:
            event = _best_cross_family_event(
                ray_a,
                ray_b,
                reject_ambiguous=reject_ambiguous,
            )
            if event is None:
                continue
            pair = tuple(sorted((ray_a.seed_id, ray_b.seed_id)))
            pairwise_seed_links[pair] = True
            candidate_events.append(event)
            min_event_times_by_ray.setdefault(ray_a.id, []).append(event.event_time)
            min_event_times_by_ray.setdefault(ray_b.id, []).append(event.event_time)

    assert pairwise_seed_links and all(pairwise_seed_links.values()), (
        "every warm/cool seed pair needs at least one diagonal collision path"
    )

    if reject_ambiguous:
        for times in min_event_times_by_ray.values():
            earliest = min(times)
            assert times.count(earliest) == 1, "a ray cannot have two earliest collisions"

    candidate_events.sort(
        key=lambda event: (
            event.event_time,
            event.cell[0],
            event.cell[1],
            event.ray_a,
            event.ray_b,
        )
    )

    stop_steps = [len(ray.path) for ray in rays]
    resolved_events: list[CollisionEvent] = []
    alive = [True] * len(rays)
    for event in candidate_events:
        if alive[event.ray_a] and alive[event.ray_b]:
            alive[event.ray_a] = False
            alive[event.ray_b] = False
            stop_steps[event.ray_a] = event.step_a
            stop_steps[event.ray_b] = event.step_b
            resolved_events.append(event)

    features = {
        "equal_collision": any(event.step_a == event.step_b for event in resolved_events),
        "unequal_collision": any(event.step_a != event.step_b for event in resolved_events),
        "boundary_ray": any(
            stop_steps[ray.id] == len(ray.path) and len(ray.path) > 0 for ray in rays
        ),
        "same_family_overlap": False,
    }

    for i, ray_a in enumerate(rays):
        for ray_b in rays[i + 1 :]:
            if ray_a.family != ray_b.family or ray_a.color == ray_b.color:
                continue
            common = set(ray_a.steps) & set(ray_b.steps)
            for cell in common:
                step_a = ray_a.steps[cell]
                step_b = ray_b.steps[cell]
                if (
                    step_a != step_b
                    and step_a <= stop_steps[ray_a.id]
                    and step_b <= stop_steps[ray_b.id]
                ):
                    features["same_family_overlap"] = True
                    break
            if features["same_family_overlap"]:
                break
        if features["same_family_overlap"]:
            break

    return {
        "grid": g,
        "seeds": seeds,
        "rays": rays,
        "stop_steps": stop_steps,
        "events": resolved_events,
        "features": features,
    }


def _render(analysis: dict) -> np.ndarray:
    out = analysis["grid"].copy()
    seed_set = {(r, c) for r, c, _color in analysis["seeds"]}
    paint_events: list[tuple[int, int, int, tuple[int, int], int]] = []

    for ray, stop_step in zip(analysis["rays"], analysis["stop_steps"], strict=False):
        for step in range(1, stop_step + 1):
            cell = ray.path[step - 1]
            paint_events.append((step, 0, ray.id, cell, ray.color))

    for idx, event in enumerate(analysis["events"]):
        paint_events.append((event.event_time, 1, idx, event.cell, event.color))

    paint_events.sort()
    for _time, _kind, _order, cell, color in paint_events:
        if cell not in seed_set:
            out[cell] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend every seed along the four diagonals, stop opposite families at their first crossing, and color each collision by the shorter-travel ray.

    Input structure
    - The grid has a black background with 3 to 5 isolated single-pixel seeds.
    - Every seed color belongs to one of two visible families: warm {red, orange, yellow} or cool {blue, green, teal}.
    - No other foreground objects are present; the puzzle is defined in the grid's standard (row, col) coordinate system.

    Rule
    - From each seed, emit one ray in each diagonal direction and advance all rays in lockstep by step count.
    - A ray that would step out of bounds stops at its current end, same-family overlaps keep moving, and different-family rays stop when their earliest shared cell is reached by both path prefixes.
    - Paint every non-colliding ray cell with its seed color; at a collision cell use the color of the ray with the shorter travel distance, breaking equal-distance ties in favor of the warm-family ray; keep all seed pixels unchanged.

    Key insight
    - Ray interaction depends on color family rather than existing painted pixels: same-family paths pass through each other, while different-family paths truncate at their first diagonal crossing.
    """
    analysis = _analyze(grid, reject_ambiguous=False)
    return _render(analysis)


def _sample_seed_count() -> int:
    counts, weights = zip(*SEED_COUNT_WEIGHTS, strict=False)
    return random.choices(list(counts), weights=list(weights), k=1)[0]


def _sample_family_counts(seed_count: int) -> tuple[int, int]:
    if seed_count == 3:
        warm_count = random.choice((1, 2))
    elif seed_count == 4:
        warm_count = 2
    else:
        warm_count = random.choice((2, 3))
    return warm_count, seed_count - warm_count


def _candidate_cells(h: int, w: int, parity: int) -> list[tuple[int, int]]:
    cells = [
        (r, c)
        for r in range(2, h - 2)
        for c in range(2, w - 2)
        if (r + c) % 2 == parity
    ]
    random.shuffle(cells)
    return cells


def _sample_positions(h: int, w: int, seed_count: int) -> list[tuple[int, int]] | None:
    parity = random.randint(0, 1)
    chosen: list[tuple[int, int]] = []
    for cell in _candidate_cells(h, w, parity):
        if all(_chebyshev(cell, other) >= 3 for other in chosen):
            chosen.append(cell)
            if len(chosen) == seed_count:
                return chosen
    return None


def _make_grid(
    shape: tuple[int, int],
    seeds: list[tuple[int, int, int]],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for r, c, color in seeds:
        grid[r, c] = color
    return grid


def _feature_target_ok(analysis: dict) -> bool:
    features = analysis["features"]
    return all(
        (
            features["equal_collision"],
            features["unequal_collision"],
            features["same_family_overlap"],
            features["boundary_ray"],
        )
    )


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    seed_count = _sample_seed_count()
    positions = _sample_positions(h, w, seed_count)
    if positions is None:
        return None

    warm_count, cool_count = _sample_family_counts(seed_count)
    warm_colors = random.sample(list(WARM_COLORS), warm_count)
    cool_colors = random.sample(list(COOL_COLORS), cool_count)
    colors = warm_colors + cool_colors
    random.shuffle(colors)
    seeds = [(r, c, color) for (r, c), color in zip(positions, colors, strict=False)]
    grid = _make_grid((h, w), seeds)

    try:
        analysis = _analyze(grid, reject_ambiguous=True)
    except AssertionError:
        return None

    if not _feature_target_ok(analysis):
        return None

    out = _render(analysis)
    nonzero = int(np.count_nonzero(out))
    if not (20 <= nonzero <= (h * w) // 2):
        return None
    if np.array_equal(out, grid):
        return None
    if not np.array_equal(main(grid), out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid diagonal-ray collision puzzle")
