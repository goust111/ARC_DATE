from __future__ import annotations

import random
import numpy as np


Coord = tuple[int, int]
Direction = tuple[int, int]


BG = 0
COLOR_POOL = (1, 2, 3, 4, 5, 6, 7, 8, 9)
DIRS: tuple[Direction, ...] = ((-1, 0), (0, 1), (1, 0), (0, -1))

GRID_H_RANGE = (10, 18)
GRID_W_RANGE = (10, 18)
PAIR_COUNT_RANGE = (2, 5)
PAIR_DENSITY = 45
MIN_RAY_LEN = 3
MAX_RAY_LEN = 9
AXIS_MIX_PROB = 0.65
CROSSING_PROB = 0.55
SCENE_ATTEMPTS = 300
PAIR_ATTEMPTS = 120
CROSSING_ATTEMPTS = 120


def _perp(direction: Direction) -> Direction:
    dr, _dc = direction
    return (0, 1) if dr else (1, 0)


def _in_bounds(shape: tuple[int, int], cell: Coord) -> bool:
    r, c = cell
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _arrow_cells(tip: Coord, direction: Direction) -> list[Coord]:
    r, c = tip
    dr, dc = direction
    pr, pc = _perp(direction)
    return [
        (r, c),
        (r - dr + pr, c - dc + pc),
        (r - dr - pr, c - dc - pc),
    ]


def _ray_distance(tip: Coord, direction: Direction, cell: Coord) -> int | None:
    tr, tc = tip
    cr, cc = cell
    dr, dc = direction
    if dr != 0:
        if cc != tc:
            return None
        delta = cr - tr
        if delta * dr <= 0:
            return None
        return abs(delta)
    if cr != tr:
        return None
    delta = cc - tc
    if delta * dc <= 0:
        return None
    return abs(delta)


def _ray_cells(tip: Coord, target: Coord, direction: Direction) -> list[Coord]:
    dist = _ray_distance(tip, direction, target)
    assert dist is not None, "target must lie on the forward ray"
    dr, dc = direction
    r, c = tip
    return [(r + dr * step, c + dc * step) for step in range(1, dist)]


def _build_pair(color: int, tip: Coord, direction: Direction, target: Coord) -> dict[str, object]:
    arrow = _arrow_cells(tip, direction)
    return {
        "color": int(color),
        "tip": (int(tip[0]), int(tip[1])),
        "direction": (int(direction[0]), int(direction[1])),
        "target": (int(target[0]), int(target[1])),
        "arrow": tuple(arrow),
        "object_cells": tuple(arrow + [target]),
        "ray": tuple(_ray_cells(tip, target, direction)),
    }


def _decode_pair(color: int, coords: np.ndarray) -> dict[str, object]:
    positions = {tuple(map(int, rc)) for rc in coords}
    assert len(positions) == 4, "each color must contain exactly one emitter and one target"

    candidates: list[dict[str, object]] = []
    for tip in sorted(positions):
        for direction in DIRS:
            arrow = set(_arrow_cells(tip, direction))
            if not arrow.issubset(positions):
                continue
            others = positions - arrow
            if len(others) != 1:
                continue
            target = next(iter(others))
            if _ray_distance(tip, direction, target) is None:
                continue
            candidates.append(_build_pair(color, tip, direction, target))

    assert len(candidates) == 1, "every color must decode to one arrow-target pair"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find each 3-pixel arrow emitter, infer its facing direction, and paint a
    ray from the tip until the same-color target pixel.
    """
    g = np.asarray(grid, dtype=int)
    pairs = []
    for color in np.unique(g):
        color = int(color)
        if color == BG:
            continue
        pairs.append(_decode_pair(color, np.argwhere(g == color)))

    out = g.copy()
    for pair in sorted(pairs, key=lambda item: item["tip"]):
        for r, c in pair["ray"]:
            out[r, c] = int(pair["color"])
    return out


def _cells_clear(shape: tuple[int, int], cells: list[Coord], occupied: set[Coord]) -> bool:
    for cell in cells:
        if not _in_bounds(shape, cell):
            return False
        r, c = cell
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if (rr, cc) in occupied:
                    return False
    return True


def _choose_length(max_len: int, prefer_long: bool) -> int:
    assert max_len >= MIN_RAY_LEN
    if not prefer_long or max_len == MIN_RAY_LEN:
        return random.randint(MIN_RAY_LEN, max_len)
    long_lo = max(MIN_RAY_LEN, (MIN_RAY_LEN + max_len) // 2)
    if random.random() < 0.7:
        return random.randint(long_lo, max_len)
    return random.randint(MIN_RAY_LEN, max_len)


def _sample_directions(count: int, require_axis_mix: bool) -> list[Direction]:
    if count == 0:
        return []
    if not require_axis_mix or count == 1:
        return [random.choice(DIRS) for _ in range(count)]

    vertical = [(-1, 0), (1, 0)]
    horizontal = [(0, -1), (0, 1)]
    dirs = [random.choice(vertical), random.choice(horizontal)]
    while len(dirs) < count:
        dirs.append(random.choice(DIRS))
    random.shuffle(dirs)
    return dirs


def _sample_free_pair(
    shape: tuple[int, int],
    color: int,
    direction: Direction,
    occupied: set[Coord],
    *,
    prefer_long: bool,
) -> dict[str, object] | None:
    H, W = shape
    dr, dc = direction

    for _ in range(PAIR_ATTEMPTS):
        if (dr, dc) == (-1, 0):
            if H < MIN_RAY_LEN + 2 or W < 3:
                return None
            tip = (random.randint(MIN_RAY_LEN, H - 2), random.randint(1, W - 2))
            max_len = min(MAX_RAY_LEN, tip[0])
        elif (dr, dc) == (1, 0):
            if H < MIN_RAY_LEN + 2 or W < 3:
                return None
            tip = (random.randint(1, H - 1 - MIN_RAY_LEN), random.randint(1, W - 2))
            max_len = min(MAX_RAY_LEN, H - 1 - tip[0])
        elif (dr, dc) == (0, 1):
            if W < MIN_RAY_LEN + 2 or H < 3:
                return None
            tip = (random.randint(1, H - 2), random.randint(1, W - 1 - MIN_RAY_LEN))
            max_len = min(MAX_RAY_LEN, W - 1 - tip[1])
        else:
            assert (dr, dc) == (0, -1)
            if W < MIN_RAY_LEN + 2 or H < 3:
                return None
            tip = (random.randint(1, H - 2), random.randint(MIN_RAY_LEN, W - 2))
            max_len = min(MAX_RAY_LEN, tip[1])

        if max_len < MIN_RAY_LEN:
            continue
        length = _choose_length(max_len, prefer_long)
        target = (tip[0] + dr * length, tip[1] + dc * length)
        pair = _build_pair(color, tip, direction, target)
        cells = list(pair["object_cells"])
        if _cells_clear(shape, cells, occupied):
            return pair
    return None


def _split_choices(before_max: int, after_max: int) -> list[tuple[int, int]]:
    choices = []
    for before in range(1, before_max + 1):
        for after in range(1, after_max + 1):
            total = before + after
            if MIN_RAY_LEN <= total <= MAX_RAY_LEN:
                choices.append((before, after))
    return choices


def _sample_crossing_seed(
    shape: tuple[int, int],
    color_a: int,
    color_b: int,
) -> list[dict[str, object]] | None:
    H, W = shape
    if H < MIN_RAY_LEN + 3 or W < MIN_RAY_LEN + 3:
        return None

    for _ in range(CROSSING_ATTEMPTS):
        intersection = (random.randint(1, H - 2), random.randint(1, W - 2))
        hdir = random.choice([(0, -1), (0, 1)])
        vdir = random.choice([(-1, 0), (1, 0)])

        if hdir == (0, 1):
            h_before_max = intersection[1] - 1
            h_after_max = W - 1 - intersection[1]
        else:
            h_before_max = W - 2 - intersection[1]
            h_after_max = intersection[1]

        if vdir == (1, 0):
            v_before_max = intersection[0] - 1
            v_after_max = H - 1 - intersection[0]
        else:
            v_before_max = H - 2 - intersection[0]
            v_after_max = intersection[0]

        h_choices = _split_choices(h_before_max, h_after_max)
        v_choices = _split_choices(v_before_max, v_after_max)
        if not h_choices or not v_choices:
            continue

        h_before, h_after = random.choice(h_choices)
        v_before, v_after = random.choice(v_choices)

        h_tip = (
            intersection[0] - hdir[0] * h_before,
            intersection[1] - hdir[1] * h_before,
        )
        h_target = (
            intersection[0] + hdir[0] * h_after,
            intersection[1] + hdir[1] * h_after,
        )
        v_tip = (
            intersection[0] - vdir[0] * v_before,
            intersection[1] - vdir[1] * v_before,
        )
        v_target = (
            intersection[0] + vdir[0] * v_after,
            intersection[1] + vdir[1] * v_after,
        )

        first = _build_pair(color_a, h_tip, hdir, h_target)
        second = _build_pair(color_b, v_tip, vdir, v_target)
        occupied = set(first["object_cells"])
        if _cells_clear(shape, list(second["object_cells"]), occupied):
            return [first, second]
    return None


def _render_input(shape: tuple[int, int], pairs: list[dict[str, object]]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for pair in pairs:
        for r, c in pair["object_cells"]:
            grid[r, c] = int(pair["color"])
    return grid


def _expected_output(shape: tuple[int, int], pairs: list[dict[str, object]]) -> np.ndarray:
    out = _render_input(shape, pairs)
    for pair in sorted(pairs, key=lambda item: item["tip"]):
        for r, c in pair["ray"]:
            out[r, c] = int(pair["color"])
    return out


def _interaction_count(pairs: list[dict[str, object]]) -> int:
    object_sets = [set(pair["object_cells"]) for pair in pairs]
    path_counts: dict[Coord, int] = {}
    hits = 0

    for i, pair in enumerate(pairs):
        others = set()
        for j, other in enumerate(object_sets):
            if i != j:
                others.update(other)
        for cell in pair["ray"]:
            path_counts[cell] = path_counts.get(cell, 0) + 1
            if cell in others:
                hits += 1

    path_crosses = sum(count > 1 for count in path_counts.values())
    return path_crosses + hits


def _generate_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    shape = (H, W)

    max_pairs = min(PAIR_COUNT_RANGE[1], len(COLOR_POOL), max(PAIR_COUNT_RANGE[0], (H * W) // PAIR_DENSITY))
    pair_count = random.randint(PAIR_COUNT_RANGE[0], max_pairs)
    require_crossing = pair_count >= 2 and random.random() < CROSSING_PROB
    require_axis_mix = pair_count >= 2 and (require_crossing or random.random() < AXIS_MIX_PROB)
    colors = random.sample(COLOR_POOL, pair_count)

    pairs: list[dict[str, object]] = []
    occupied: set[Coord] = set()

    if require_crossing:
        seed = _sample_crossing_seed(shape, colors[0], colors[1])
        if seed is None:
            return None
        pairs.extend(seed)
        for pair in seed:
            occupied.update(pair["object_cells"])
        remaining_colors = colors[2:]
        directions = _sample_directions(len(remaining_colors), False)
    else:
        remaining_colors = colors
        directions = _sample_directions(pair_count, require_axis_mix)

    for color, direction in zip(remaining_colors, directions):
        pair = _sample_free_pair(shape, color, direction, occupied, prefer_long=require_crossing)
        if pair is None:
            return None
        pairs.append(pair)
        occupied.update(pair["object_cells"])

    if require_crossing and _interaction_count(pairs) == 0:
        return None

    grid = _render_input(shape, pairs)
    if not np.array_equal(main(grid), _expected_output(shape, pairs)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(SCENE_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ray-emitter scene")
