from __future__ import annotations

import itertools
import random
from collections import Counter

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _colors_present(grid: np.ndarray, bg: int) -> list[int]:
    return [int(color) for color in np.unique(grid) if int(color) != int(bg)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= int(r) < int(h) and 0 <= int(c) < int(w)


def _reflect(center_sum: tuple[int, int], coord: tuple[int, int]) -> tuple[int, int]:
    return int(center_sum[0] - coord[0]), int(center_sum[1] - coord[1])


def _expand_cells(
    cells: set[tuple[int, int]],
    shape: tuple[int, int],
    radius: int = 1,
) -> set[tuple[int, int]]:
    h, w = shape
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _candidate_pairs(
    shape: tuple[int, int],
    center_sum: tuple[int, int],
    radius: int,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    h, w = shape
    sr, sc = center_sum
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r in range(h):
        if abs(2 * r - sr) > 2 * radius:
            continue
        for c in range(w):
            if abs(2 * c - sc) > 2 * radius:
                continue
            other = _reflect(center_sum, (r, c))
            if not _in_bounds(shape, *other) or other == (r, c):
                continue
            if (r, c) < other:
                pairs.append(((r, c), other))
    random.shuffle(pairs)
    return pairs


def _is_symmetric_group(coords: set[tuple[int, int]], center_sum: tuple[int, int]) -> bool:
    for coord in coords:
        if _reflect(center_sum, coord) not in coords:
            return False
    return True


def _analyze_center(
    grid: np.ndarray,
    color: int,
    coords: list[tuple[int, int]],
    center_sum: tuple[int, int],
    bg: int,
) -> dict[str, object] | None:
    coord_set = set(coords)
    paired = 0
    occupied_unmatched: list[tuple[tuple[int, int], tuple[int, int], int]] = []
    empty_unmatched: list[tuple[tuple[int, int], tuple[int, int]]] = []

    for coord in coords:
        other = _reflect(center_sum, coord)
        if other in coord_set:
            if coord < other:
                paired += 1
            continue

        if _in_bounds(grid.shape, *other):
            seen = int(grid[other])
            if seen == int(bg):
                empty_unmatched.append((coord, other))
            else:
                occupied_unmatched.append((coord, other, seen))
        else:
            empty_unmatched.append((coord, other))

    if paired < 2:
        return None
    if len(occupied_unmatched) != 1 or len(empty_unmatched) != 1:
        return None
    if 2 * paired + 2 != len(coords):
        return None

    occupier = int(occupied_unmatched[0][2])
    if occupier in (int(bg), int(color)):
        return None

    return {
        "color": int(color),
        "center": (int(center_sum[0]), int(center_sum[1])),
        "paired": int(paired),
        "source": occupied_unmatched[0][0],
        "target": occupied_unmatched[0][1],
        "occupier_color": occupier,
        "empty_source": empty_unmatched[0][0],
        "empty_target": empty_unmatched[0][1],
    }


def _candidate_infos(grid: np.ndarray, color: int, bg: int) -> list[dict[str, object]]:
    coords = [tuple(map(int, coord)) for coord in np.argwhere(grid == int(color))]
    assert len(coords) >= 6 and len(coords) % 2 == 0, "each color should have an even-sized scattered set"

    pair_sum_counts: Counter[tuple[int, int]] = Counter()
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            pair_sum_counts[(coords[i][0] + coords[j][0], coords[i][1] + coords[j][1])] += 1

    infos: list[dict[str, object]] = []
    for center_sum, support in pair_sum_counts.items():
        info = _analyze_center(grid, color, coords, center_sum, bg)
        if info is None:
            continue
        info["support"] = int(support)
        infos.append(info)

    assert infos, f"no valid hidden center for color {color}"
    best_support = max(int(info["support"]) for info in infos)
    best_infos = [info for info in infos if int(info["support"]) == best_support]
    best_infos.sort(
        key=lambda info: (
            tuple(info["center"]),
            tuple(info["target"]),
            int(info["occupier_color"]),
        )
    )
    return best_infos


def _forms_single_cycle(mapping: dict[int, int], colors: list[int]) -> bool:
    color_set = set(int(color) for color in colors)
    if set(mapping) != color_set or set(mapping.values()) != color_set:
        return False

    start = int(colors[0])
    seen: set[int] = set()
    current = start
    while current not in seen:
        seen.add(current)
        current = int(mapping[current])
    return len(seen) == len(colors) and current == start


def _infer_recolor_plan(grid: np.ndarray) -> dict[int, dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    colors = _colors_present(g, bg)
    assert len(colors) >= 2, "expected at least two foreground colors"

    candidates = {color: _candidate_infos(g, color, bg) for color in colors}
    search_order = sorted(colors, key=lambda color: (len(candidates[color]), color))
    solutions: list[dict[int, dict[str, object]]] = []
    used_occupiers: set[int] = set()
    mapping: dict[int, int] = {}
    selected: dict[int, dict[str, object]] = {}

    def dfs(index: int) -> None:
        if len(solutions) > 1:
            return
        if index == len(search_order):
            if used_occupiers != set(colors):
                return
            if len({tuple(info["target"]) for info in selected.values()}) != len(colors):
                return
            if not _forms_single_cycle(mapping, colors):
                return
            solutions.append({color: dict(info) for color, info in selected.items()})
            return

        color = search_order[index]
        for info in candidates[color]:
            occupier = int(info["occupier_color"])
            if occupier == int(color) or occupier not in candidates or occupier in used_occupiers:
                continue
            used_occupiers.add(occupier)
            mapping[occupier] = int(color)
            selected[int(color)] = info
            dfs(index + 1)
            del selected[int(color)]
            del mapping[occupier]
            used_occupiers.remove(occupier)

    dfs(0)
    assert len(solutions) == 1, f"expected a unique closed recolor cycle, found {len(solutions)}"
    return solutions[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each wrong intruder pixel to the color whose hidden point-symmetry it breaks.

    Input structure:
    - Black background with several sparse foreground colors.
    - For each color, most pixels form same-color pairs under a color-specific hidden 180-degree center.
    - Exactly one pixel of that color is missing, and that location is occupied by another foreground color.
    - The occupation mistakes form one closed cycle across all foreground colors.

    Rule:
    - For each color, infer its hidden center from the repeated midpoint shared by its same-color pairs.
    - Under that center, exactly one visible pixel points to an occupied reflected location of a different color, while the other unpaired pixel points to empty space.
    - Recolor each occupied reflected location back to the current color, without moving, adding, or deleting any pixels.

    Key insight:
    - The apparent noise dots are not random errors: each one is another color's missing symmetric partner, so the puzzle is solved by per-color center recovery plus global cycle consistency.
    """
    g = np.asarray(grid, dtype=int)
    plan = _infer_recolor_plan(g)

    out = g.copy()
    for color, info in plan.items():
        r, c = tuple(info["target"])
        out[int(r), int(c)] = int(color)

    bg = _get_bg(g)
    assert np.array_equal(out != int(bg), g != int(bg)), "the transform must only recolor existing pixels"
    return out


# --------------- generator (output-first, semantic-checked) ---------------

BG_COLOR = 0
PALETTE = (1, 2, 3, 4, 5, 6, 7, 8, 9)

GRID_H_RANGE = (20, 28)
GRID_W_RANGE = (20, 28)
COLOR_COUNT_OPTIONS = (2, 3, 4, 5)
COLOR_COUNT_WEIGHTS = (2, 4, 4, 2)

PAIR_COUNT_RANGE = (4, 6)
PAIR_RADIUS_RANGE = (3, 6)
CENTER_MARGIN_DOUBLED = 4
CENTER_SEPARATION_DOUBLED = 7
CENTER_PLACEMENT_TRIES = 160
PLAN_TRIES_PER_CYCLE = 20
GENERATION_ATTEMPTS = 400


def _build_output_attempt() -> dict[str, object] | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    n_colors = random.choices(COLOR_COUNT_OPTIONS, weights=COLOR_COUNT_WEIGHTS, k=1)[0]
    colors = random.sample(PALETTE, n_colors)

    shape = (h, w)
    occupied: dict[tuple[int, int], int] = {}
    blocked: set[tuple[int, int]] = set()
    centers_in_use: list[tuple[int, int]] = []
    centers: dict[int, tuple[int, int]] = {}
    groups: dict[int, set[tuple[int, int]]] = {}

    for color in colors:
        target_pairs = random.randint(*PAIR_COUNT_RANGE)
        placed = False
        for _ in range(CENTER_PLACEMENT_TRIES):
            center_sum = (
                random.randint(CENTER_MARGIN_DOUBLED, 2 * h - 1 - CENTER_MARGIN_DOUBLED),
                random.randint(CENTER_MARGIN_DOUBLED, 2 * w - 1 - CENTER_MARGIN_DOUBLED),
            )
            if any(
                max(abs(center_sum[0] - old[0]), abs(center_sum[1] - old[1])) < CENTER_SEPARATION_DOUBLED
                for old in centers_in_use
            ):
                continue

            pair_candidates = _candidate_pairs(shape, center_sum, random.randint(*PAIR_RADIUS_RANGE))
            local_blocked = set(blocked)
            chosen_pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
            for p, q in pair_candidates:
                if p in local_blocked or q in local_blocked:
                    continue
                chosen_pairs.append((p, q))
                local_blocked |= _expand_cells({p, q}, shape, radius=1)
                if len(chosen_pairs) >= target_pairs:
                    break

            if len(chosen_pairs) < target_pairs:
                continue

            coords = {cell for pair in chosen_pairs for cell in pair}
            centers[color] = center_sum
            groups[color] = coords
            centers_in_use.append(center_sum)
            blocked = local_blocked
            if center_sum[0] % 2 == 0 and center_sum[1] % 2 == 0:
                blocked.add((center_sum[0] // 2, center_sum[1] // 2))
            for cell in coords:
                occupied[cell] = int(color)
            placed = True
            break

        if not placed:
            return None

    output = np.full(shape, BG_COLOR, dtype=int)
    for cell, color in occupied.items():
        output[cell] = int(color)

    return {
        "output": output,
        "colors": colors,
        "centers": centers,
        "groups": groups,
    }


def _valid_intrusion_spots(
    output: np.ndarray,
    occupier_center: tuple[int, int],
    target_coords: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    spots: list[tuple[int, int]] = []
    for coord in target_coords:
        other = _reflect(occupier_center, coord)
        if not _in_bounds(output.shape, *other) or int(output[other]) == BG_COLOR:
            spots.append(coord)
    return spots


def _cycle_plans(scene: dict[str, object]) -> list[dict[int, tuple[tuple[int, int], int]]]:
    output = np.asarray(scene["output"], dtype=int)
    colors = list(scene["colors"])
    centers = dict(scene["centers"])
    groups = dict(scene["groups"])

    valid: dict[int, dict[int, list[tuple[int, int]]]] = {int(color): {} for color in colors}
    for occupier in colors:
        for target in colors:
            if occupier == target:
                continue
            spots = _valid_intrusion_spots(output, centers[int(occupier)], groups[int(target)])
            if spots:
                valid[int(occupier)][int(target)] = spots

    plans: list[dict[int, tuple[tuple[int, int], int]]] = []
    orders = list(itertools.permutations(colors))
    random.shuffle(orders)
    for order in orders:
        if not all(int(order[(i + 1) % len(order)]) in valid[int(order[i])] for i in range(len(order))):
            continue
        for _ in range(PLAN_TRIES_PER_CYCLE):
            plan: dict[int, tuple[tuple[int, int], int]] = {}
            for i, occupier in enumerate(order):
                target = int(order[(i + 1) % len(order)])
                plan[target] = (random.choice(valid[int(occupier)][target]), int(occupier))
            plans.append(plan)
    return plans


def _apply_plan(output: np.ndarray, plan: dict[int, tuple[tuple[int, int], int]]) -> np.ndarray:
    inp = np.asarray(output, dtype=int).copy()
    for _target_color, (coord, occupier_color) in plan.items():
        inp[int(coord[0]), int(coord[1])] = int(occupier_color)
    return inp


def _validate_generated_pair(
    inp: np.ndarray,
    out: np.ndarray,
    scene: dict[str, object],
    plan: dict[int, tuple[tuple[int, int], int]],
) -> bool:
    colors = list(scene["colors"])
    centers = dict(scene["centers"])
    groups = dict(scene["groups"])

    if np.array_equal(inp, out):
        return False
    if np.count_nonzero(inp != out) != len(colors):
        return False
    if not np.array_equal(inp != BG_COLOR, out != BG_COLOR):
        return False
    if len(set(centers.values())) != len(colors):
        return False

    for color in colors:
        coords = groups[int(color)]
        center_sum = centers[int(color)]
        if not _is_symmetric_group(coords, center_sum):
            return False
        if center_sum[0] % 2 == 0 and center_sum[1] % 2 == 0:
            if (center_sum[0] // 2, center_sum[1] // 2) in coords:
                return False

        missing_spot, occupier = plan[int(color)]
        if missing_spot not in coords or occupier == int(color):
            return False
        if int(out[missing_spot]) != int(color):
            return False
        if int(inp[missing_spot]) != int(occupier):
            return False

    mapping = {int(occupier): int(color) for color, (_coord, occupier) in plan.items()}
    if not _forms_single_cycle(mapping, colors):
        return False

    try:
        inferred = _infer_recolor_plan(inp)
    except Exception:
        return False

    if set(inferred) != set(colors):
        return False
    for color in colors:
        info = inferred[int(color)]
        if tuple(info["center"]) != tuple(centers[int(color)]):
            return False
        if tuple(info["target"]) != tuple(plan[int(color)][0]):
            return False
        if int(info["occupier_color"]) != int(plan[int(color)][1]):
            return False

    return np.array_equal(main(inp), out)


def _gen_attempt() -> np.ndarray | None:
    scene = _build_output_attempt()
    if scene is None:
        return None

    out = np.asarray(scene["output"], dtype=int)
    plans = _cycle_plans(scene)
    random.shuffle(plans)
    for plan in plans:
        inp = _apply_plan(out, plan)
        if _validate_generated_pair(inp, out, scene, plan):
            return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetric-partner recolor puzzle")
