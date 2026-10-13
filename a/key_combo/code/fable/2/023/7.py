from __future__ import annotations

import random

import numpy as np


Seed = tuple[int, int, int]  # (row, col, color)


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(np.asarray(grid, dtype=int))))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def _split_panels(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h > 0 and w >= 3 and w % 2 == 1, "expected two equal-width panels plus a separator column"

    sep_col = w // 2
    sep_color = int(g[0, sep_col])
    assert np.all(g[:, sep_col] == sep_color), "separator must be a full-height uniform line"

    left = g[:, :sep_col]
    right = g[:, sep_col + 1 :]
    assert left.shape == right.shape and left.size > 0

    bg = _get_bg(left)
    assert _get_bg(right) == bg, "both panels must share the same background"
    assert sep_color != bg, "separator color must differ from panel background"
    assert sep_color not in set(map(int, np.unique(left)))
    assert sep_color not in set(map(int, np.unique(right)))
    return left, right, bg, sep_color


def _growth_diagnostics(shape: tuple[int, int], seeds: list[Seed]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h, w = map(int, shape)
    assert seeds, "expected at least one seed"

    rr = np.arange(h, dtype=int)[None, :, None]
    cc = np.arange(w, dtype=int)[None, None, :]
    seed_rows = np.array([r for r, _c, _color in seeds], dtype=int)[:, None, None]
    seed_cols = np.array([c for _r, c, _color in seeds], dtype=int)[:, None, None]
    dists = np.maximum(np.abs(seed_rows - rr), np.abs(seed_cols - cc))

    min_dist = dists.min(axis=0)
    unique = (dists == min_dist).sum(axis=0) == 1
    winner_idx = dists.argmin(axis=0)
    return dists, min_dist, unique, winner_idx


def _simulate_frame(shape: tuple[int, int], seeds: list[Seed], t: int, bg: int) -> np.ndarray:
    dists, min_dist, unique, winner_idx = _growth_diagnostics(shape, seeds)
    del dists
    colors = np.array([color for _r, _c, color in seeds], dtype=int)
    out = np.full(shape, int(bg), dtype=int)
    visible = unique & (min_dist <= int(t))
    out[visible] = colors[winner_idx[visible]]
    return out


def _event_report(shape: tuple[int, int], seeds: list[Seed], step: int) -> dict:
    dists, min_dist, unique, winner_idx = _growth_diagnostics(shape, seeds)
    new_mask = unique & (min_dist == int(step))
    blocked_mask = np.any(dists == int(step), axis=0) & (min_dist < int(step))
    conflict_mask = (min_dist == int(step)) & (~unique)
    new_counts: dict[int, int] = {}
    for idx, (_r, _c, color) in enumerate(seeds):
        new_counts[int(color)] = int(np.count_nonzero(new_mask & (winner_idx == idx)))
    return {
        "new_counts": new_counts,
        "blocked_count": int(np.count_nonzero(blocked_mask)),
        "conflict_count": int(np.count_nonzero(conflict_mask)),
    }


def _candidate_seeds_for_color(
    left_cells: list[tuple[int, int]],
    right_cells: list[tuple[int, int]],
    delta_cells: list[tuple[int, int]],
    common_cells: list[tuple[int, int]],
    color: int,
    t: int,
) -> list[Seed]:
    assert common_cells, "each color must keep its seed in both panels"

    candidates: list[Seed] = []
    for sr, sc in common_cells:
        if not all(max(abs(r - sr), abs(c - sc)) <= int(t) for r, c in left_cells):
            continue
        if not all(max(abs(r - sr), abs(c - sc)) <= int(t) + 1 for r, c in right_cells):
            continue
        if not all(max(abs(r - sr), abs(c - sc)) == int(t) + 1 for r, c in delta_cells):
            continue
        candidates.append((int(sr), int(sc), int(color)))
    return candidates


def _infer_time_and_seeds(left: np.ndarray, right: np.ndarray, bg: int) -> tuple[int, list[Seed]]:
    colors_left = _colors_present(left, exclude={int(bg)})
    colors_right = _colors_present(right, exclude={int(bg)})
    assert colors_left == colors_right, "both frames must contain the same seed colors"
    colors = colors_left
    assert colors, "expected at least one non-background seed color"

    left_cells_by_color = {
        color: [tuple(map(int, rc)) for rc in np.argwhere(left == color)]
        for color in colors
    }
    right_cells_by_color = {
        color: [tuple(map(int, rc)) for rc in np.argwhere(right == color)]
        for color in colors
    }
    delta_cells_by_color = {
        color: [tuple(map(int, rc)) for rc in np.argwhere((right == color) & (left != color))]
        for color in colors
    }
    common_cells_by_color = {
        color: [tuple(map(int, rc)) for rc in np.argwhere((left == color) & (right == color))]
        for color in colors
    }

    shape = left.shape
    max_t = max(map(int, shape))
    solutions: list[tuple[int, list[Seed]]] = []

    for t in range(max_t + 1):
        candidate_map: dict[int, list[Seed]] = {}
        valid = True
        for color in colors:
            candidates = _candidate_seeds_for_color(
                left_cells_by_color[color],
                right_cells_by_color[color],
                delta_cells_by_color[color],
                common_cells_by_color[color],
                color,
                t,
            )
            if not candidates:
                valid = False
                break
            candidate_map[color] = candidates
        if not valid:
            continue

        order = sorted(colors, key=lambda color: len(candidate_map[color]))
        used_positions: set[tuple[int, int]] = set()
        chosen: list[Seed] = []

        def backtrack(index: int) -> None:
            if len(solutions) > 1:
                return
            if index == len(order):
                seeds = chosen.copy()
                if np.array_equal(_simulate_frame(shape, seeds, t, bg), left) and np.array_equal(
                    _simulate_frame(shape, seeds, t + 1, bg), right
                ):
                    solutions.append((int(t), seeds))
                return

            color = order[index]
            for seed in candidate_map[color]:
                pos = (int(seed[0]), int(seed[1]))
                if pos in used_positions:
                    continue
                used_positions.add(pos)
                chosen.append(seed)
                backtrack(index + 1)
                chosen.pop()
                used_positions.remove(pos)

        backtrack(0)
        if len(solutions) > 1:
            break

    assert len(solutions) == 1, "frame pair must imply a unique time and seed layout"
    t, seeds = solutions[0]
    return int(t), sorted(seeds)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extrapolate the next Chebyshev-growth frame from two equal panels that show consecutive times, not overlay layers.

    Input structure
    - The input is split into two equal-width panels by a full-height separator column.
    - The left panel is time T and the right panel is time T+1 on the same background.
    - Each non-background color has exactly one seed pixel, and each seed grows outward by Chebyshev-distance square rings.

    Rule
    - Interpret both panels as the same seed layout observed at two consecutive times.
    - At step s, a cell becomes a seed's color iff that seed is the unique nearest seed in Chebyshev distance and that minimum distance equals or is below s.
    - If two seeds reach a background cell at the same step, that cell stays background; a farther seed also cannot overwrite a cell that a nearer seed filled earlier.
    - Recover the unique time T and seed coordinates that reproduce both input panels, then simulate one more step and return the single-panel frame at time T+2.

    Key insight
    - The separator does not divide transparent layers to combine; it separates adjacent moments of the same growth process, so the task is to infer the growth dynamics from frame differences and push the timeline forward.
    """
    left, right, bg, _sep_color = _split_panels(grid)
    t, seeds = _infer_time_and_seeds(left, right, bg)
    return _simulate_frame(right.shape, seeds, t + 2, bg)


# --------------- generator ---------------

_BG = 0
_PANEL_SIZE_RANGE = (9, 13)
_TIME_RANGE = (1, 3)
_SEED_COUNT_CHOICES = (3, 3, 3, 4)
_PALETTE = tuple(range(1, 10))
_MIN_SEED_GAP = 2
_RIGHT_FILL_LIMIT = 0.58
_OUTPUT_FILL_LIMIT = 0.68
_MAX_GENERATION_ATTEMPTS = 2000


def _well_spaced(existing: list[tuple[int, int]], r: int, c: int) -> bool:
    return all(max(abs(rr - int(r)), abs(cc - int(c))) >= _MIN_SEED_GAP for rr, cc in existing)


def _seed_is_clipped(shape: tuple[int, int], seed: Seed, step: int) -> bool:
    h, w = map(int, shape)
    r, c, _color = seed
    return min(int(r), int(c), h - 1 - int(r), w - 1 - int(c)) < int(step)


def _validate_instance(inp: np.ndarray, expected: np.ndarray, t: int, seeds: list[Seed], sep_color: int) -> None:
    left, right, bg, found_sep = _split_panels(inp)
    assert int(bg) == _BG
    assert int(found_sep) == int(sep_color)

    inferred_t, inferred_seeds = _infer_time_and_seeds(left, right, bg)
    assert inferred_t == int(t)
    assert tuple(inferred_seeds) == tuple(sorted(seeds))

    assert np.array_equal(_simulate_frame(left.shape, seeds, t, bg), left)
    assert np.array_equal(_simulate_frame(left.shape, seeds, t + 1, bg), right)
    assert np.array_equal(_simulate_frame(left.shape, seeds, t + 2, bg), expected)

    report_now = _event_report(left.shape, seeds, t + 1)
    report_next = _event_report(left.shape, seeds, t + 2)
    assert all(count > 0 for count in report_now["new_counts"].values())
    assert all(count > 0 for count in report_next["new_counts"].values())
    assert report_now["conflict_count"] > 0
    assert report_next["conflict_count"] > 0
    assert report_now["blocked_count"] > 0
    assert report_next["blocked_count"] > 0
    assert any(_seed_is_clipped(left.shape, seed, t + 2) for seed in seeds)
    assert np.array_equal(main(inp), expected)


def _generate_input_attempt() -> np.ndarray | None:
    h = random.randint(*_PANEL_SIZE_RANGE)
    w = random.randint(*_PANEL_SIZE_RANGE)
    seed_count = random.choice(_SEED_COUNT_CHOICES)
    t = random.randint(*_TIME_RANGE)

    sampled = random.sample(_PALETTE, seed_count + 1)
    seed_colors = sampled[:seed_count]
    sep_color = sampled[-1]

    edge_positions = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if min(r, c, h - 1 - r, w - 1 - c) < t + 2
    ]
    if not edge_positions:
        return None

    seeds: list[Seed] = []
    used_positions: list[tuple[int, int]] = []

    first_r, first_c = random.choice(edge_positions)
    seeds.append((int(first_r), int(first_c), int(seed_colors[0])))
    used_positions.append((int(first_r), int(first_c)))

    while len(seeds) < seed_count:
        candidates = [
            (r, c)
            for r in range(h)
            for c in range(w)
            if (r, c) not in used_positions and _well_spaced(used_positions, r, c)
        ]
        if not candidates:
            return None
        r, c = random.choice(candidates)
        used_positions.append((int(r), int(c)))
        seeds.append((int(r), int(c), int(seed_colors[len(seeds)])))

    left = _simulate_frame((h, w), seeds, t, _BG)
    right = _simulate_frame((h, w), seeds, t + 1, _BG)
    expected = _simulate_frame((h, w), seeds, t + 2, _BG)

    area = h * w
    if np.count_nonzero(right != _BG) > int(area * _RIGHT_FILL_LIMIT):
        return None
    if np.count_nonzero(expected != _BG) > int(area * _OUTPUT_FILL_LIMIT):
        return None
    if np.array_equal(left, right) or np.array_equal(right, expected) or np.array_equal(left, expected):
        return None

    report_now = _event_report((h, w), seeds, t + 1)
    report_next = _event_report((h, w), seeds, t + 2)
    if not all(count > 0 for count in report_now["new_counts"].values()):
        return None
    if not all(count > 0 for count in report_next["new_counts"].values()):
        return None
    if report_now["conflict_count"] == 0 or report_next["conflict_count"] == 0:
        return None
    if report_now["blocked_count"] == 0 or report_next["blocked_count"] == 0:
        return None
    if not any(_seed_is_clipped((h, w), seed, t + 2) for seed in seeds):
        return None

    inp = np.hstack([left, np.full((h, 1), int(sep_color), dtype=int), right])
    try:
        _validate_instance(inp, expected, t, seeds, sep_color)
    except AssertionError:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _generate_input_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid double-frame growth puzzle")
