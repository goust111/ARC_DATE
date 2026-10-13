from __future__ import annotations

import itertools
import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
ARROW_SIZE = 3
LEGEND_HEIGHT = 6
SEPARATOR_HEIGHT = 1

DIRS = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}

ARROW_PATTERNS = {
    "up": {(0, 1), (1, 0), (1, 1), (1, 2)},
    "right": {(0, 1), (1, 1), (1, 2), (2, 1)},
    "down": {(1, 0), (1, 1), (1, 2), (2, 1)},
    "left": {(0, 1), (1, 0), (1, 1), (2, 1)},
}

PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    mask = grid == color
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if _in_bounds((h, w), nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(comp)
    return components


def _solid_separator_row(grid: np.ndarray, bg: int) -> int:
    rows = [
        r
        for r in range(grid.shape[0])
        if int(grid[r, 0]) != bg and np.all(grid[r] == int(grid[r, 0]))
    ]
    assert len(rows) == 1, f"expected one separator row, got {rows}"
    return int(rows[0])


def _find_arrow_mapping(top: np.ndarray, bg: int) -> tuple[dict[int, str], int]:
    colors = [int(c) for c in np.unique(top) if int(c) != bg]
    arrow_candidates = []
    for color in colors:
        comps = _components_of_color(top, color)
        if comps and any(len(comp) > 1 for comp in comps):
            if all(len(comp) == 4 for comp in comps):
                arrow_candidates.append(color)
    assert len(arrow_candidates) == 1, f"expected one arrow color, got {arrow_candidates}"
    arrow_color = arrow_candidates[0]

    mapping: dict[int, str] = {}
    marker_coords = [
        (int(r), int(c))
        for r, c in np.argwhere((top != bg) & (top != arrow_color))
    ]
    assert len(marker_coords) >= 3, "legend must contain several colored markers"

    for mr, mc in marker_coords:
        assert 1 <= mc < top.shape[1] - 1, "marker must allow a centered 3x3 arrow slot above"
        matches = []
        for r0 in range(0, mr - 1):
            if r0 + 3 > mr:
                continue
            window = top[r0 : r0 + 3, mc - 1 : mc + 2]
            shape = set(map(tuple, np.argwhere(window == arrow_color)))
            if not shape:
                continue
            dirs = [name for name, pattern in ARROW_PATTERNS.items() if shape == pattern]
            if dirs:
                matches.append(dirs[0])
        assert len(matches) == 1, f"marker at {(mr, mc)} must match one arrow slot"
        marker_color = int(top[mr, mc])
        assert marker_color not in mapping, "legend marker colors must be unique"
        mapping[marker_color] = matches[0]

    assert len(mapping) >= 3
    return mapping, arrow_color


def _extract_field_state(
    field: np.ndarray, mapping: dict[int, str], arrow_color: int, bg: int
) -> tuple[dict[tuple[int, int], int], tuple[int, int], tuple[int, int]]:
    seeds = {
        (int(r), int(c)): int(field[r, c])
        for r, c in np.argwhere((field != bg) & (field != arrow_color))
    }
    assert seeds, "field must contain seeds"
    assert set(seeds.values()) <= set(mapping), "field uses colors outside the legend"

    trigger_candidates = []
    for tr, tc in np.argwhere(field == arrow_color):
        adjacent = [
            (nr, nc)
            for nr, nc in _neighbors4(int(tr), int(tc))
            if (int(nr), int(nc)) in seeds
        ]
        if len(adjacent) == 1:
            trigger_candidates.append(((int(tr), int(tc)), adjacent[0]))
    assert len(trigger_candidates) == 1, f"expected one trigger, got {trigger_candidates}"
    return seeds, trigger_candidates[0][0], trigger_candidates[0][1]


def _trace_ray(
    start: tuple[int, int],
    direction: str,
    seeds: dict[tuple[int, int], int],
    shape: tuple[int, int],
) -> tuple[list[tuple[int, int]], tuple[int, int] | None]:
    dr, dc = DIRS[direction]
    r, c = start
    painted: list[tuple[int, int]] = []
    while True:
        r += dr
        c += dc
        if not _in_bounds(shape, r, c):
            return painted, None
        if (r, c) in seeds:
            return painted, (r, c)
        painted.append((r, c))


def _simulate_chain(
    field: np.ndarray,
    mapping: dict[int, str],
    arrow_color: int,
    *,
    return_order: bool = False,
) -> tuple[np.ndarray, list[tuple[int, int]]] | np.ndarray:
    bg = _get_bg(field)
    seeds, trigger, current = _extract_field_state(field, mapping, arrow_color, bg)

    out = field.copy()
    out[trigger] = bg
    order: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    while True:
        assert current not in seen, "chain must not contain a cycle"
        seen.add(current)
        order.append(current)
        color = seeds[current]
        painted, hit = _trace_ray(current, mapping[color], seeds, field.shape)
        for r, c in painted:
            out[r, c] = color
        if hit is None:
            break
        current = hit

    if return_order:
        return out, order
    return out


def _simulate_all_seeds_fire(field: np.ndarray, mapping: dict[int, str], arrow_color: int) -> np.ndarray:
    bg = _get_bg(field)
    seeds, trigger, _start = _extract_field_state(field, mapping, arrow_color, bg)
    out = field.copy()
    out[trigger] = bg
    for seed in sorted(seeds):
        color = seeds[seed]
        painted, _hit = _trace_ray(seed, mapping[color], seeds, field.shape)
        for r, c in painted:
            out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trigger a color-directed chain of rays by reading the legend arrows at the top.

    Input structure:
    - A top legend contains several grey 3x3 triangle-like arrow sprites, each pointing up, right, down, or left.
    - Directly below each arrow sits one colored marker pixel; together they define that marker color's firing direction.
    - A solid separator row splits the legend from a black playfield that contains colored seed pixels and one grey trigger pixel.

    Rule:
    - Read the legend to map each seed color to one cardinal direction.
    - In the playfield, only the unique seed adjacent to the grey trigger is activated first; it fires a ray in its mapped direction.
    - Starting from the neighboring cell, the ray paints cells with the seed's color until it leaves the grid or reaches the cell just before the next seed; that reached seed then becomes the next activated seed. Remove the trigger and keep only the playfield.

    Key insight:
    - Color meanings are not fixed: the top legend redefines them every example, and non-triggered seeds stay inert unless the activation chain actually reaches them.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    sep = _solid_separator_row(g, bg)
    top = g[:sep]
    field = g[sep + 1 :]
    mapping, arrow_color = _find_arrow_mapping(top, bg)
    return _simulate_chain(field, mapping, arrow_color)


def _path_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    ar, ac = a
    br, bc = b
    if ar == br:
        step = 1 if bc > ac else -1
        return [(ar, c) for c in range(ac + step, bc, step)]
    if ac == bc:
        step = 1 if br > ar else -1
        return [(r, ac) for r in range(ar + step, br, step)]
    raise ValueError("points must be axis-aligned")


def _chebyshev_far_enough(pos: tuple[int, int], others: list[tuple[int, int]], gap: int = 1) -> bool:
    return all(max(abs(pos[0] - r), abs(pos[1] - c)) > gap for r, c in others)


def _choose_legend_layout(n_entries: int) -> tuple[list[int], int]:
    gap = random.choice([1, 2])
    left_margin = random.randint(0, 2)
    right_margin = random.randint(0, 2)
    starts = [left_margin + i * (ARROW_SIZE + gap) for i in range(n_entries)]
    width = starts[-1] + ARROW_SIZE + right_margin
    return starts, width


def _choose_mapping(n_entries: int) -> tuple[dict[int, str], list[int]]:
    directions = random.sample(list(DIRS), n_entries)
    colors = random.sample(PALETTE, n_entries)
    random.shuffle(directions)
    mapping = {color: direction for color, direction in zip(colors, directions)}
    return mapping, colors


def _direction_to_color(mapping: dict[int, str]) -> dict[str, int]:
    return {direction: color for color, direction in mapping.items()}


def _candidate_chain_targets(
    pos: tuple[int, int],
    direction: str,
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    seeds: list[tuple[int, int]],
) -> list[tuple[tuple[int, int], int]]:
    h, w = shape
    dr, dc = DIRS[direction]
    candidates: list[tuple[tuple[int, int], int]] = []
    k = 2
    while True:
        nr = pos[0] + dr * k
        nc = pos[1] + dc * k
        if not (1 <= nr < h - 1 and 1 <= nc < w - 1):
            break
        segment = _path_cells(pos, (nr, nc))
        if any(cell in occupied for cell in segment):
            break
        if (nr, nc) not in occupied and _chebyshev_far_enough((nr, nc), seeds, gap=1):
            candidates.append(((nr, nc), k))
        k += 1
    return candidates


def _candidate_exit_directions(
    pos: tuple[int, int],
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    directions: list[str],
) -> list[str]:
    h, w = shape
    out: list[str] = []
    for direction in directions:
        dr, dc = DIRS[direction]
        r, c = pos
        painted = 0
        ok = True
        while True:
            r += dr
            c += dc
            if not _in_bounds((h, w), r, c):
                break
            if (r, c) in occupied:
                ok = False
                break
            painted += 1
        if ok and painted >= 1:
            out.append(direction)
    return out


def _build_active_chain(
    shape: tuple[int, int],
    directions: list[str],
) -> tuple[list[tuple[int, int]], list[str], set[tuple[int, int]]] | None:
    h, w = shape
    chain_len = random.randint(3, 5)
    for _ in range(160):
        start = (random.randint(1, h - 2), random.randint(1, w - 2))
        positions = [start]
        used = {start}
        steps: list[str] = []
        for step_idx in range(chain_len - 1):
            prev = steps[-1] if steps else None
            order = directions[:]
            random.shuffle(order)
            if prev is not None and random.random() < 0.7:
                order.sort(key=lambda d: d == prev)
            if prev is not None and step_idx == chain_len - 2 and len(set(steps)) == 1:
                non_prev = [d for d in order if d != prev]
                if non_prev:
                    order = non_prev + [prev]

            candidates: list[tuple[str, tuple[int, int], list[tuple[int, int]]]] = []
            for direction in order:
                for target, _dist in _candidate_chain_targets(positions[-1], direction, shape, used, positions):
                    segment = _path_cells(positions[-1], target)
                    candidates.append((direction, target, segment))
            if not candidates:
                break
            direction, target, segment = random.choice(candidates)
            steps.append(direction)
            used.update(segment)
            used.add(target)
            positions.append(target)
        else:
            exits = _candidate_exit_directions(positions[-1], shape, used, directions)
            if exits:
                last_direction = random.choice(exits)
                return positions, steps + [last_direction], used
    return None


def _trace_on_seed_set(
    start: tuple[int, int],
    direction: str,
    seeds: dict[tuple[int, int], int],
    shape: tuple[int, int],
) -> tuple[list[tuple[int, int]], tuple[int, int] | None]:
    return _trace_ray(start, direction, seeds, shape)


def _place_inert_seeds(
    shape: tuple[int, int],
    mapping: dict[int, str],
    active_positions: list[tuple[int, int]],
    active_colors: list[int],
    occupied: set[tuple[int, int]],
) -> dict[tuple[int, int], int] | None:
    h, w = shape
    inert: dict[tuple[int, int], int] = {}
    colors_in_chain = set(active_colors)
    missing = [color for color in mapping if color not in colors_in_chain]
    inert_colors = missing[:]
    shared_color = random.choice(active_colors)
    inert_colors.append(shared_color)
    while len(inert_colors) < random.randint(max(1, len(inert_colors)), max(2, len(inert_colors))):
        inert_colors.append(random.choice(list(mapping)))

    for color in inert_colors:
        placed = False
        for _ in range(120):
            pos = (random.randint(1, h - 2), random.randint(1, w - 2))
            if pos in occupied or pos in inert:
                continue
            if not _chebyshev_far_enough(pos, active_positions + list(inert), gap=1):
                continue

            tentative = {p: c for p, c in zip(active_positions, active_colors)}
            tentative.update(inert)
            tentative[pos] = color
            painted, _hit = _trace_on_seed_set(pos, mapping[color], tentative, shape)
            if len(painted) < 1:
                continue
            if any(cell in occupied for cell in painted):
                continue
            inert[pos] = color
            occupied.add(pos)
            placed = True
            break
        if not placed:
            return None
    return inert


def _choose_trigger(
    start: tuple[int, int],
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    all_seeds: set[tuple[int, int]],
) -> tuple[int, int] | None:
    choices = []
    for nr, nc in _neighbors4(*start):
        if not _in_bounds(shape, nr, nc):
            continue
        if (nr, nc) in occupied or (nr, nc) in all_seeds:
            continue
        adjacent = [(r, c) for r, c in _neighbors4(nr, nc) if (r, c) in all_seeds]
        if adjacent == [start]:
            choices.append((nr, nc))
    return random.choice(choices) if choices else None


def _render_input(
    mapping: dict[int, str],
    legend_order: list[int],
    field: np.ndarray,
    separator_color: int,
    legend_starts: list[int],
) -> np.ndarray:
    width = field.shape[1]
    legend = np.full((LEGEND_HEIGHT, width), BG, dtype=int)
    for color, c0 in zip(legend_order, legend_starts):
        direction = mapping[color]
        for r, c in ARROW_PATTERNS[direction]:
            legend[r, c0 + c] = GREY
        legend[4, c0 + 1] = color
    separator = np.full((SEPARATOR_HEIGHT, width), separator_color, dtype=int)
    return np.vstack([legend, separator, field])


def _simulate_with_mapping(
    field: np.ndarray,
    mapping: dict[int, str],
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    return _simulate_chain(field, mapping, GREY, return_order=True)


def _validate_instance(
    field: np.ndarray,
    mapping: dict[int, str],
    expected_order: list[tuple[int, int]],
) -> bool:
    try:
        expected, order = _simulate_with_mapping(field, mapping)
    except Exception:
        return False
    if order != expected_order:
        return False

    seeds, _trigger, _start = _extract_field_state(field, mapping, GREY, BG)
    if len(seeds) <= len(expected_order):
        return False
    if len({seeds[pos] for pos in expected_order}) < 2:
        return False

    inactive = [pos for pos in seeds if pos not in expected_order]
    if not inactive:
        return False
    if not any(seeds[pos] in {seeds[p] for p in expected_order} for pos in inactive):
        return False

    if np.array_equal(expected, _simulate_all_seeds_fire(field, mapping, GREY)):
        return False

    colors = list(mapping)
    directions = [mapping[c] for c in colors]
    same_count = 0
    for perm in itertools.permutations(directions):
        alt = dict(zip(colors, perm))
        try:
            alt_out = _simulate_chain(field, alt, GREY)
        except Exception:
            continue
        if np.array_equal(alt_out, expected):
            same_count += 1
            if alt != mapping:
                return False
    return same_count == 1


def _gen_attempt() -> np.ndarray | None:
    n_entries = random.choice([3, 4])
    mapping, legend_colors = _choose_mapping(n_entries)
    legend_starts, width = _choose_legend_layout(n_entries)
    field_height = random.randint(9, 17)
    if field_height + LEGEND_HEIGHT + SEPARATOR_HEIGHT > 30 or width > 30:
        return None

    built = _build_active_chain((field_height, width), list(_direction_to_color(mapping)))
    if built is None:
        return None
    active_positions, active_dirs, occupied = built
    color_by_direction = _direction_to_color(mapping)
    active_colors = [color_by_direction[d] for d in active_dirs]

    inert = _place_inert_seeds((field_height, width), mapping, active_positions, active_colors, occupied)
    if inert is None:
        return None

    seed_map = {pos: color for pos, color in zip(active_positions, active_colors)}
    seed_map.update(inert)
    trigger = _choose_trigger(active_positions[0], (field_height, width), occupied, set(seed_map))
    if trigger is None:
        return None

    field = np.full((field_height, width), BG, dtype=int)
    for pos, color in seed_map.items():
        field[pos] = color
    field[trigger] = GREY

    if not _validate_instance(field, mapping, active_positions):
        return None

    separator_color = random.choice([c for c in PALETTE if c not in mapping and c != GREY])
    legend_order = legend_colors[:]
    random.shuffle(legend_order)
    inp = _render_input(mapping, legend_order, field, separator_color, legend_starts)

    expected = main(inp)
    oracle, order = _simulate_with_mapping(field, mapping)
    if order != active_positions:
        return None
    if not np.array_equal(expected, oracle):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
