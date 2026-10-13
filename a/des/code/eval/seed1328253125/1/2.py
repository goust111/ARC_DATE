from __future__ import annotations

from collections import Counter

import numpy as np


BG = 5
OBSTACLE = 9
PALETTE = (1, 2, 3, 4, 6, 7, 8)

UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRS = (UP, DOWN, LEFT, RIGHT)
OUTCOMES = ("full", "obstacle", "boundary")

GRID_H_MIN, GRID_H_MAX = 14, 24
GRID_W_MIN, GRID_W_MAX = 14, 24
PAIR_COUNT_MIN, PAIR_COUNT_MAX = 2, 4
SEG_LEN_MIN, SEG_LEN_MAX = 2, 5
PAIR_TRIES = 250
GEN_TRIES = 400
EXTRA_OBS_MIN, EXTRA_OBS_MAX = 1, 6
UNCONSTRAINED_OFFSET_MAX = 4
CONSTRAINED_OFFSET_MIN = 3


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if not (dr == 0 and dc == 0)
    ]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _segment_info(cells: list[tuple[int, int]]) -> dict[str, object] | None:
    if not (SEG_LEN_MIN <= len(cells) <= SEG_LEN_MAX):
        return None

    rows = sorted({r for r, _ in cells})
    cols = sorted({c for _, c in cells})

    if len(rows) == 1:
        row = rows[0]
        ordered = sorted(cells, key=lambda rc: rc[1])
        if [c for _, c in ordered] != list(range(cols[0], cols[-1] + 1)):
            return None
        return {
            "cells": ordered,
            "length": len(ordered),
            "endpoints": [((row, cols[0]), LEFT), ((row, cols[-1]), RIGHT)],
        }

    if len(cols) == 1:
        col = cols[0]
        ordered = sorted(cells, key=lambda rc: rc[0])
        if [r for r, _ in ordered] != list(range(rows[0], rows[-1] + 1)):
            return None
        return {
            "cells": ordered,
            "length": len(ordered),
            "endpoints": [((rows[0], col), UP), ((rows[-1], col), DOWN)],
        }

    return None


def _choose_facing_endpoint(
    segment: dict[str, object], other_cells: list[tuple[int, int]]
) -> tuple[tuple[int, int], tuple[int, int]]:
    center_r = sum(r for r, _ in other_cells) / len(other_cells)
    center_c = sum(c for _, c in other_cells) / len(other_cells)
    best_score: tuple[float, float, int, int, int] | None = None
    best_choice: tuple[tuple[int, int], tuple[int, int]] | None = None

    for endpoint, direction in segment["endpoints"]:
        er, ec = endpoint
        dr, dc = direction
        dist = min(abs(er - r) + abs(ec - c) for r, c in other_cells)
        proj = (center_r - er) * dr + (center_c - ec) * dc
        sum_dist = sum(abs(er - r) + abs(ec - c) for r, c in other_cells)
        score = (float(dist), -float(proj), int(sum_dist), int(er), int(ec))
        if best_score is None or score < best_score:
            best_score = score
            best_choice = (endpoint, direction)

    assert best_choice is not None
    return best_choice


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group same-color segments, pick the endpoint facing the partner segment, and
    grow matching rays whose lengths equal the pair's total segment area.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    out = g.copy()

    blocker_mask = np.zeros(g.shape, dtype=bool)
    colored_pairs: list[tuple[int, list[dict[str, object]]]] = []

    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        comps = _components(g == color)
        segments = [_segment_info(comp) for comp in comps]
        if len(comps) == 2 and all(seg is not None for seg in segments):
            colored_pairs.append((color, [seg for seg in segments if seg is not None]))
            continue

        if all(len(comp) == 1 for comp in comps):
            blocker_mask |= g == color
            continue

        raise AssertionError(f"unexpected object structure for color {color}")

    for color, segments in colored_pairs:
        a, b = segments
        total_len = int(a["length"]) + int(b["length"])
        starts = [
            _choose_facing_endpoint(a, b["cells"]),
            _choose_facing_endpoint(b, a["cells"]),
        ]
        for endpoint, direction in starts:
            er, ec = endpoint
            dr, dc = direction
            for step in range(1, total_len + 1):
                rr = er + dr * step
                cc = ec + dc * step
                if not _in_bounds(g.shape, rr, cc):
                    break
                if blocker_mask[rr, cc]:
                    break
                out[rr, cc] = color

    return out


def _pair_signs(
    d1: tuple[int, int], d2: tuple[int, int]
) -> tuple[int | None, int | None]:
    row_signs: set[int] = set()
    col_signs: set[int] = set()

    if d1[0] != 0:
        row_signs.add(int(np.sign(d1[0])))
    if d1[1] != 0:
        col_signs.add(int(np.sign(d1[1])))
    if d2[0] != 0:
        row_signs.add(-int(np.sign(d2[0])))
    if d2[1] != 0:
        col_signs.add(-int(np.sign(d2[1])))

    if len(row_signs) > 1 or len(col_signs) > 1:
        return None, None

    row_sign = next(iter(row_signs)) if row_signs else 0
    col_sign = next(iter(col_signs)) if col_signs else 0
    if row_sign == 0 and col_sign == 0:
        return None, None
    return row_sign, col_sign


COMPATIBLE_DIR_PAIRS = [
    (d1, d2, row_sign, col_sign)
    for d1 in DIRS
    for d2 in DIRS
    for row_sign, col_sign in [_pair_signs(d1, d2)]
    if row_sign is not None
]


def _segment_cells(
    endpoint: tuple[int, int], direction: tuple[int, int], length: int
) -> list[tuple[int, int]]:
    er, ec = endpoint
    dr, dc = direction
    return [(er - dr * k, ec - dc * k) for k in range(length)]


def _forward_space(
    shape: tuple[int, int], endpoint: tuple[int, int], direction: tuple[int, int]
) -> int:
    er, ec = endpoint
    dr, dc = direction
    count = 0
    rr, cc = er, ec
    while True:
        rr += dr
        cc += dc
        if not _in_bounds(shape, rr, cc):
            return count
        count += 1


def _endpoint_candidates(
    shape: tuple[int, int],
    length: int,
    direction: tuple[int, int],
    total_len: int,
    outcome: str,
) -> list[tuple[tuple[int, int], int]]:
    candidates: list[tuple[tuple[int, int], int]] = []
    for r in range(shape[0]):
        for c in range(shape[1]):
            seg = _segment_cells((r, c), direction, length)
            if not all(_in_bounds(shape, rr, cc) for rr, cc in seg):
                continue
            avail = _forward_space(shape, (r, c), direction)
            if outcome == "full" and avail < total_len:
                continue
            if outcome == "boundary" and not (1 <= avail < total_len):
                continue
            if outcome == "obstacle" and min(avail, total_len) < 2:
                continue
            candidates.append(((r, c), avail))
    return candidates


def _ray_plan(
    endpoint: tuple[int, int],
    direction: tuple[int, int],
    total_len: int,
    avail: int,
    outcome: str,
    rng: np.random.Generator,
) -> tuple[list[tuple[int, int]], tuple[int, int] | None]:
    er, ec = endpoint
    dr, dc = direction
    obstacle = None

    if outcome == "full":
        steps = total_len
    elif outcome == "boundary":
        steps = avail
    else:
        stop_dist = int(rng.integers(2, min(avail, total_len) + 1))
        steps = stop_dist - 1
        obstacle = (er + dr * stop_dist, ec + dc * stop_dist)

    cells = [(er + dr * step, ec + dc * step) for step in range(1, steps + 1)]
    return cells, obstacle


def _too_close_to_existing(
    new_cells: set[tuple[int, int]], forbidden: set[tuple[int, int]]
) -> bool:
    return any(cell in forbidden for cell in new_cells)


def _same_color_components_separate(
    seg_a: list[tuple[int, int]], seg_b: list[tuple[int, int]]
) -> bool:
    set_b = set(seg_b)
    return all((nr, nc) not in set_b for r, c in seg_a for nr, nc in _neighbors4(r, c))


def _valid_relative_offset(
    endpoint_a: tuple[int, int],
    endpoint_b: tuple[int, int],
    row_sign: int,
    col_sign: int,
) -> bool:
    dr = endpoint_b[0] - endpoint_a[0]
    dc = endpoint_b[1] - endpoint_a[1]

    if row_sign and int(np.sign(dr)) != row_sign:
        return False
    if col_sign and int(np.sign(dc)) != col_sign:
        return False
    if row_sign and abs(dr) < CONSTRAINED_OFFSET_MIN:
        return False
    if col_sign and abs(dc) < CONSTRAINED_OFFSET_MIN:
        return False
    if row_sign == 0 and abs(dr) > UNCONSTRAINED_OFFSET_MAX:
        return False
    if col_sign == 0 and abs(dc) > UNCONSTRAINED_OFFSET_MAX:
        return False
    return abs(dr) + abs(dc) >= 4


def _place_pair(
    grid: np.ndarray,
    color: int,
    spec: dict[str, object],
    blocked_all: set[tuple[int, int]],
    input_forbidden: set[tuple[int, int]],
    obstacle_cells: set[tuple[int, int]],
    rng: np.random.Generator,
) -> bool:
    shape = grid.shape
    len_a, len_b = spec["lengths"]
    total_len = len_a + len_b
    dir_a, dir_b = spec["dirs"]
    row_sign, col_sign = spec["signs"]
    outcome_a, outcome_b = spec["outcomes"]

    cand_a = _endpoint_candidates(shape, len_a, dir_a, total_len, outcome_a)
    cand_b = _endpoint_candidates(shape, len_b, dir_b, total_len, outcome_b)
    if not cand_a or not cand_b:
        return False

    for _ in range(PAIR_TRIES):
        endpoint_a, avail_a = cand_a[int(rng.integers(len(cand_a)))]
        endpoint_b, avail_b = cand_b[int(rng.integers(len(cand_b)))]

        if not _valid_relative_offset(endpoint_a, endpoint_b, row_sign, col_sign):
            continue

        seg_a = _segment_cells(endpoint_a, dir_a, len_a)
        seg_b = _segment_cells(endpoint_b, dir_b, len_b)
        if not _same_color_components_separate(seg_a, seg_b):
            continue

        info_a = _segment_info(seg_a)
        info_b = _segment_info(seg_b)
        assert info_a is not None and info_b is not None
        if _choose_facing_endpoint(info_a, info_b["cells"]) != (endpoint_a, dir_a):
            continue
        if _choose_facing_endpoint(info_b, info_a["cells"]) != (endpoint_b, dir_b):
            continue

        ray_a, obs_a = _ray_plan(endpoint_a, dir_a, total_len, avail_a, outcome_a, rng)
        ray_b, obs_b = _ray_plan(endpoint_b, dir_b, total_len, avail_b, outcome_b, rng)

        pair_input = set(seg_a) | set(seg_b)
        if obs_a is not None:
            pair_input.add(obs_a)
        if obs_b is not None:
            pair_input.add(obs_b)

        pair_reserved = pair_input | set(ray_a) | set(ray_b)
        if len(pair_reserved) != len(pair_input) + len(set(ray_a)) + len(set(ray_b)):
            continue
        if pair_reserved & blocked_all:
            continue
        if _too_close_to_existing(pair_input, input_forbidden):
            continue

        new_obstacles = [cell for cell in (obs_a, obs_b) if cell is not None]
        bad_obstacle = False
        for cell in new_obstacles:
            if any(nei in obstacle_cells for nei in _neighbors4(*cell)):
                bad_obstacle = True
                break
            if any(other != cell and other in _neighbors4(*cell) for other in new_obstacles):
                bad_obstacle = True
                break
        if bad_obstacle:
            continue

        for r, c in seg_a + seg_b:
            grid[r, c] = color
        for cell in new_obstacles:
            grid[cell] = OBSTACLE

        blocked_all.update(pair_reserved)
        for cell in pair_input:
            input_forbidden.add(cell)
            for nei in _neighbors8(*cell):
                if _in_bounds(shape, *nei):
                    input_forbidden.add(nei)
        obstacle_cells.update(new_obstacles)
        return True

    return False


def _balanced_outcomes(ray_count: int, rng: np.random.Generator) -> list[str]:
    offset = int(rng.integers(len(OUTCOMES)))
    outcomes = [OUTCOMES[(offset + i) % len(OUTCOMES)] for i in range(ray_count)]
    rng.shuffle(outcomes)
    return outcomes


def _add_extra_obstacles(
    grid: np.ndarray,
    blocked_all: set[tuple[int, int]],
    input_forbidden: set[tuple[int, int]],
    obstacle_cells: set[tuple[int, int]],
    rng: np.random.Generator,
) -> None:
    target = int(rng.integers(EXTRA_OBS_MIN, EXTRA_OBS_MAX + 1))
    placed = 0
    attempts = 0
    while placed < target and attempts < 500:
        attempts += 1
        r = int(rng.integers(grid.shape[0]))
        c = int(rng.integers(grid.shape[1]))
        cell = (r, c)
        if cell in blocked_all or cell in input_forbidden:
            continue
        if any(nei in obstacle_cells for nei in _neighbors4(r, c)):
            continue
        grid[cell] = OBSTACLE
        blocked_all.add(cell)
        obstacle_cells.add(cell)
        input_forbidden.add(cell)
        for nei in _neighbors8(r, c):
            if _in_bounds(grid.shape, *nei):
                input_forbidden.add(nei)
        placed += 1


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(GRID_H_MIN, GRID_H_MAX + 1))
    w = int(rng.integers(GRID_W_MIN, GRID_W_MAX + 1))
    pair_count = int(rng.integers(PAIR_COUNT_MIN, PAIR_COUNT_MAX + 1))
    colors = list(rng.choice(PALETTE, size=pair_count, replace=False))
    outcomes = _balanced_outcomes(2 * pair_count, rng)

    specs: list[tuple[int, dict[str, object]]] = []
    for idx, color in enumerate(colors):
        lengths = (
            int(rng.integers(SEG_LEN_MIN, SEG_LEN_MAX + 1)),
            int(rng.integers(SEG_LEN_MIN, SEG_LEN_MAX + 1)),
        )
        dir_a, dir_b, row_sign, col_sign = COMPATIBLE_DIR_PAIRS[int(rng.integers(len(COMPATIBLE_DIR_PAIRS)))]
        specs.append(
            (
                color,
                {
                    "lengths": lengths,
                    "dirs": (dir_a, dir_b),
                    "signs": (row_sign, col_sign),
                    "outcomes": (outcomes[2 * idx], outcomes[2 * idx + 1]),
                    "total": sum(lengths),
                },
            )
        )

    specs.sort(key=lambda item: int(item[1]["total"]), reverse=True)

    grid = np.full((h, w), BG, dtype=int)
    blocked_all: set[tuple[int, int]] = set()
    input_forbidden: set[tuple[int, int]] = set()
    obstacle_cells: set[tuple[int, int]] = set()

    for color, spec in specs:
        if not _place_pair(grid, color, spec, blocked_all, input_forbidden, obstacle_cells, rng):
            return None

    _add_extra_obstacles(grid, blocked_all, input_forbidden, obstacle_cells, rng)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_TRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
