import random
from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
COLOR_POOL = [BLUE, RED, YELLOW, PINK, ORANGE, TEAL, MAROON]
GEN_ATTEMPTS = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS:
        yield r + dr, c + dc


def _bbox(coords: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _extract_objects(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    grey = {tuple(map(int, p)) for p in np.argwhere(g == GREY)}
    seeds = np.argwhere(g == GREEN)
    assert seeds.shape == (1, 2), "expected exactly one green seed"
    seed = tuple(map(int, seeds[0]))
    colored = {
        tuple(map(int, p)): int(g[tuple(p)])
        for p in np.argwhere((g != BLACK) & (g != GREY) & (g != GREEN))
    }
    assert 4 <= len(colored) <= 6, "expected 4-6 colored singletons"
    assert len(set(colored.values())) == len(colored), "colored singletons must use distinct colors"
    return g, grey, seed, colored


def _spiral_groups(seed: tuple[int, int], shape: tuple[int, int], targets: set[tuple[int, int]]):
    r, c = seed
    remaining = set(targets)
    leg_length = 1
    direction_index = 0
    steps = 0
    step_limit = (2 * max(shape) + 3) ** 2
    groups: list[list[tuple[tuple[int, int], int]]] = []
    while remaining and steps < step_limit:
        dr, dc = DIRS[direction_index]
        group: list[tuple[tuple[int, int], int]] = []
        for step_along_leg in range(leg_length):
            r += dr
            c += dc
            steps += 1
            pos = (r, c)
            if _in_bounds(shape, r, c) and pos in remaining:
                group.append((pos, step_along_leg))
                remaining.remove(pos)
        groups.append(group)
        direction_index = (direction_index + 1) % 4
        if direction_index in (0, 2):
            leg_length += 1
    assert not remaining, "spiral scan failed to encounter all colored pixels"
    return groups


def _distance_options(
    pos: tuple[int, int],
    mass: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    shape: tuple[int, int],
):
    r, c = pos
    options: list[tuple[int, int, tuple[int, int], tuple[int, int]]] = []
    for direction_index, (dr, dc) in enumerate(DIRS):
        i = 1
        while True:
            rr = r + dr * i
            cc = c + dc * i
            if not _in_bounds(shape, rr, cc):
                break
            if (rr, cc) in mass:
                landing = (r + dr * (i - 1), c + dc * (i - 1))
                options.append((i - 1, direction_index, landing, (dr, dc)))
                break
            if (rr, cc) in blocked:
                break
            i += 1
    return options


def _solve_with_details(grid: np.ndarray):
    g, grey, seed, colored = _extract_objects(grid)
    remaining = dict(colored)
    mass = set(grey)
    shape = g.shape
    groups = _spiral_groups(seed, shape, set(remaining))
    encounter_order: list[tuple[int, int]] = []
    group_sizes: list[int] = []
    same_leg_branch = False
    queue_snapshot_distances: list[tuple[tuple[int, int], int]] = []
    history: list[dict] = []

    out = np.full_like(g, BLACK)
    for cell in mass:
        out[cell] = GREY
    out[seed] = GREEN
    for pos, color in remaining.items():
        out[pos] = color

    for group in groups:
        if not group:
            continue
        group_sizes.append(len(group))
        if len(group) >= 2:
            same_leg_branch = True
        blocked_snapshot = set(remaining) - {pos for pos, _ in group}
        blocked_snapshot.add(seed)
        snapshot_keys = []
        for pos, step_along_leg in group:
            options = _distance_options(pos, mass, blocked_snapshot, shape)
            assert options, "each colored pixel must have an unobstructed cardinal path"
            best_distance = min(option[0] for option in options)
            count_best = sum(option[0] == best_distance for option in options)
            assert count_best == 1, "direction choice must be unique"
            snapshot_keys.append((best_distance, step_along_leg, pos))
            queue_snapshot_distances.append((pos, best_distance))
        order = [pos for _, _, pos in sorted(snapshot_keys)]
        encounter_order.extend(order)
        for pos in order:
            color = remaining.pop(pos)
            out[pos] = BLACK
            blocked_now = set(remaining)
            blocked_now.add(seed)
            options = _distance_options(pos, mass, blocked_now, shape)
            assert options, "current pixel must still have a path when processed"
            best_distance = min(option[0] for option in options)
            best_options = [option for option in options if option[0] == best_distance]
            assert len(best_options) == 1, "current pixel direction must stay unique"
            _, direction_index, landing, direction = best_options[0]
            mass.add(landing)
            out[landing] = GREY
            history.append(
                {
                    "src": pos,
                    "dst": landing,
                    "distance": best_distance,
                    "direction_index": direction_index,
                    "direction": direction,
                    "color": color,
                }
            )

    out[seed] = BLACK
    return {
        "output": out,
        "seed": seed,
        "grey": grey,
        "colored": colored,
        "groups": groups,
        "encounter_order": encounter_order,
        "group_sizes": group_sizes,
        "same_leg_branch": same_leg_branch,
        "queue_snapshot_distances": queue_snapshot_distances,
        "history": history,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use a green-seeded spiral scan to choose the order in which colored singletons slide into a gray mass and become gray.

    Input structure:
    - The grid is mostly black background with exactly one connected gray rectangle-like mass.
    - There is exactly one green single-pixel seed that marks where the spiral scan starts.
    - There are 4-6 other non-black single-pixel colored objects, each with a distinct color.

    Rule:
    - Starting at the green seed, scan the infinite plane by a square spiral using directions up, right, down, left and leg lengths 1, 1, 2, 2, 3, 3, ...
    - The spiral is only a scan order: whenever a leg passes over colored pixels, those pixels enter the processing queue, and pixels hit in the same leg are processed first by smaller current distance to the gray mass.
    - For each queued pixel, look along the four cardinal directions and keep only directions whose ray reaches the current gray mass through black cells only; choose the unique shortest such direction.
    - Slide the pixel straight along that direction until it is 4-neighbor-adjacent to the current mass, recolor it gray, merge it into the mass, and continue with the updated mass.
    - After all colored pixels are processed, erase the green seed back to black.

    Key insight:
    - The visible output contains only the accumulated gray mass, so the hidden spiral scan order must be inferred from how different seed positions change the sequence of mass growth.
    """
    return _solve_with_details(np.asarray(grid, dtype=int))["output"]


def _ray_candidates(mass: set[tuple[int, int]], shape: tuple[int, int]):
    h, w = shape
    r0, c0, r1, c1 = _bbox(mass)
    candidates: set[tuple[int, int]] = set()
    for r in range(r0, r1 + 1):
        for c in range(0, c0 - 1):
            candidates.add((r, c))
        for c in range(c1 + 2, w):
            candidates.add((r, c))
    for c in range(c0, c1 + 1):
        for r in range(0, r0 - 1):
            candidates.add((r, c))
        for r in range(r1 + 2, h):
            candidates.add((r, c))
    return sorted(candidates)


def _is_seed_clear(seed: tuple[int, int], mass: set[tuple[int, int]], occupied: set[tuple[int, int]]):
    if seed in occupied:
        return False
    if min(abs(seed[0] - r) + abs(seed[1] - c) for r, c in mass) < 2:
        return False
    for cell in occupied:
        if abs(seed[0] - cell[0]) + abs(seed[1] - cell[1]) <= 1:
            return False
    return True


def _has_unique_path(
    pos: tuple[int, int],
    mass: set[tuple[int, int]],
    obstacles: set[tuple[int, int]],
    shape: tuple[int, int],
):
    options = _distance_options(pos, mass, obstacles, shape)
    if not options:
        return False
    best = min(option[0] for option in options)
    return sum(option[0] == best for option in options) == 1


def _render_grid(
    shape: tuple[int, int],
    mass: set[tuple[int, int]],
    seed: tuple[int, int],
    colored_positions: list[tuple[int, int]],
    colors: list[int],
):
    grid = np.zeros(shape, dtype=int)
    for cell in mass:
        grid[cell] = GREY
    for pos, color in zip(colored_positions, colors):
        grid[pos] = color
    grid[seed] = GREEN
    return grid


def _queue_signature(details: dict):
    return tuple(details["encounter_order"])


def _count_distinct_outputs_for_seeds(
    shape: tuple[int, int],
    mass: set[tuple[int, int]],
    colored_positions: list[tuple[int, int]],
    colors: list[int],
    seeds: list[tuple[int, int]],
):
    outputs: dict[bytes, list[tuple[int, int]]] = {}
    queue_signatures: dict[tuple[tuple[int, int], ...], list[tuple[int, int]]] = {}
    for seed in seeds:
        grid = _render_grid(shape, mass, seed, colored_positions, colors)
        try:
            details = _solve_with_details(grid)
        except AssertionError:
            continue
        out_key = details["output"].tobytes()
        outputs.setdefault(out_key, []).append(seed)
        qsig = _queue_signature(details)
        queue_signatures.setdefault(qsig, []).append(seed)
    return outputs, queue_signatures


def _distance_sorted_output(grid: np.ndarray):
    g, grey, seed, colored = _extract_objects(grid)
    mass = set(grey)
    remaining = dict(colored)
    out = np.full_like(g, BLACK)
    for cell in mass:
        out[cell] = GREY
    out[seed] = GREEN
    for pos, color in remaining.items():
        out[pos] = color

    while remaining:
        ranking = []
        for pos in remaining:
            blocked = set(remaining) - {pos}
            blocked.add(seed)
            options = _distance_options(pos, mass, blocked, g.shape)
            if not options:
                return None
            best = min(options, key=lambda item: (item[0], item[1], item[2]))
            count_best = sum(option[0] == best[0] for option in options)
            if count_best != 1:
                return None
            ranking.append((best[0], pos))
        ranking.sort()
        pos = ranking[0][1]
        blocked = set(remaining) - {pos}
        blocked.add(seed)
        best = min(_distance_options(pos, mass, blocked, g.shape), key=lambda item: (item[0], item[1], item[2]))
        remaining.pop(pos)
        out[pos] = BLACK
        mass.add(best[2])
        out[best[2]] = GREY
    out[seed] = BLACK
    return out


def _passes_semantic_checks(
    grid: np.ndarray,
    *,
    require_same_leg_branch: bool,
):
    details = _solve_with_details(grid)
    _, grey, seed, colored = _extract_objects(grid)
    positions = list(colored)
    colors = [colored[pos] for pos in positions]
    occupied = set(positions) | grey

    if require_same_leg_branch and not details["same_leg_branch"]:
        return False

    if len(set(details["encounter_order"])) != len(positions):
        return False

    if len(details["history"]) != len(positions):
        return False

    for step in details["history"]:
        if step["src"] == step["dst"]:
            continue
        sr, sc = step["src"]
        dr, dc = step["direction"]
        dist = step["distance"]
        for k in range(1, dist + 1):
            rr = sr + dr * k
            cc = sc + dc * k
            if not _in_bounds(grid.shape, rr, cc):
                return False

    alt_seeds = [
        (r, c)
        for r in range(grid.shape[0])
        for c in range(grid.shape[1])
        if _is_seed_clear((r, c), grey, occupied)
    ]
    random.shuffle(alt_seeds)
    sample_seeds = alt_seeds[: min(40, len(alt_seeds))]
    outputs, queue_signatures = _count_distinct_outputs_for_seeds(
        grid.shape,
        grey,
        positions,
        colors,
        sample_seeds,
    )

    # The hidden clue must at least alter the processing queue across valid seed placements.
    if len(queue_signatures) < 2:
        return False

    distance_sorted = _distance_sorted_output(grid)
    if distance_sorted is not None and np.array_equal(distance_sorted, details["output"]):
        static_ranking = []
        for pos in positions:
            blocked = set(positions) - {pos}
            blocked.add(seed)
            best = min(
                _distance_options(pos, grey, blocked, grid.shape),
                key=lambda item: (item[0], item[1], item[2]),
            )
            static_ranking.append((best[0], pos))
        static_order = tuple(pos for _, pos in sorted(static_ranking))
        if tuple(details["encounter_order"]) == static_order:
            return False

    if len(outputs) == 1 and details["same_leg_branch"]:
        nontrivial_group = any(size >= 2 for size in details["group_sizes"])
        if not nontrivial_group:
            return False

    return True


def _gen_attempt():
    h = random.randint(13, 17)
    w = random.randint(13, 17)
    rect_h = random.randint(2, 3)
    rect_w = random.randint(3, 5)
    r0 = random.randint(max(2, h // 2 - 3), min(h - rect_h - 3, h // 2 + 1))
    c0 = random.randint(max(2, w // 2 - 3), min(w - rect_w - 3, w // 2 + 1))
    grey = {(r, c) for r in range(r0, r0 + rect_h) for c in range(c0, c0 + rect_w)}
    shape = (h, w)

    candidates = _ray_candidates(grey, shape)
    random.shuffle(candidates)
    color_count = random.randint(4, 6)
    require_same_leg_branch = random.random() < 0.45

    chosen: list[tuple[int, int]] = []
    sides = Counter()
    for pos in candidates:
        if any(abs(pos[0] - rr) + abs(pos[1] - cc) <= 1 for rr, cc in chosen):
            continue
        if pos[0] < r0:
            side = "top"
        elif pos[0] > r0 + rect_h - 1:
            side = "bottom"
        elif pos[1] < c0:
            side = "left"
        else:
            side = "right"
        if len(chosen) < color_count:
            chosen.append(pos)
            sides[side] += 1
        if len(chosen) == color_count:
            break
    if len(chosen) != color_count:
        return None
    if len(sides) < 3:
        return None

    color_values = random.sample(COLOR_POOL, color_count)
    occupied = set(chosen) | grey
    seed_candidates = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if _is_seed_clear((r, c), grey, occupied)
    ]
    random.shuffle(seed_candidates)
    if len(seed_candidates) < 3:
        return None

    valid_seeds = []
    for seed in seed_candidates:
        obstacles = set(chosen)
        obstacles.add(seed)
        if all(_has_unique_path(pos, grey, obstacles - {pos}, shape) for pos in chosen):
            valid_seeds.append(seed)
    if len(valid_seeds) < 2:
        return None

    sampled_seeds = valid_seeds[: min(24, len(valid_seeds))]
    scored = []
    for seed in sampled_seeds:
        grid = _render_grid(shape, grey, seed, chosen, color_values)
        try:
            details = _solve_with_details(grid)
        except AssertionError:
            continue
        score = (
            int(details["same_leg_branch"]),
            len(set(details["encounter_order"])),
            sum(details["group_sizes"]),
            -min(abs(seed[0] - r0), abs(seed[1] - c0)),
        )
        scored.append((score, seed, details))
    if not scored:
        return None
    scored.sort(reverse=True)

    if require_same_leg_branch:
        scored = [item for item in scored if item[2]["same_leg_branch"]]
        if not scored:
            return None

    seed = scored[0][1]
    grid = _render_grid(shape, grey, seed, chosen, color_values)
    if not _passes_semantic_checks(grid, require_same_leg_branch=require_same_leg_branch):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid spiral-slide ARC input")
