from __future__ import annotations

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6

DIR_BY_COLOR = {
    BLUE: (-1, 0),
    RED: (0, 1),
    GREEN: (1, 0),
    YELLOW: (0, -1),
}
DIR_COLORS = np.array([BLUE, RED, GREEN, YELLOW], dtype=int)
ANCHOR_COLORS = np.array([BLUE, RED, GREEN, YELLOW, MAGENTA], dtype=int)

SCENARIOS = ("cascade", "seeded", "simple")
GRID_SIDE_MIN = 11
GRID_SIDE_MAX = 28
STEP_OPTIONS = (3, 4)
MARGIN_OPTIONS = (1, 2)
ROW_COL_COUNT_MIN = 2
ROW_COL_COUNT_MAX = 4
POINT_COUNT_MIN = 5
POINT_COUNT_MAX = 8
LAYOUT_RETRIES = 120
ASSIGNMENT_RETRIES = 80
GENERATE_RETRIES = 500
MIN_GREY_CELLS = 5
MAX_GREY_FRACTION = 0.45

_RNG = np.random.default_rng()


def _pick_one(items, rng: np.random.Generator):
    return items[int(rng.integers(0, len(items)))]


def _sorted_points(mask: np.ndarray) -> list[tuple[int, int]]:
    points = [tuple(map(int, pos)) for pos in np.argwhere(mask)]
    points.sort()
    return points


def _extract_layout(grid: np.ndarray) -> tuple[list[tuple[int, int]], tuple[int, int], list[tuple[int, int]]]:
    mask = np.isin(grid, ANCHOR_COLORS)
    points = _sorted_points(mask)
    seed_points = [pos for pos in points if int(grid[pos]) == MAGENTA]
    assert len(seed_points) == 1, "expected exactly one magenta seed"
    seed = seed_points[0]
    emitters = [pos for pos in points if pos != seed]
    return points, seed, emitters


def _direction_targets(points: list[tuple[int, int]]) -> dict[tuple[int, int], dict[int, tuple[int, int] | None]]:
    targets: dict[tuple[int, int], dict[int, tuple[int, int] | None]] = {}
    for src in points:
        src_r, src_c = src
        color_targets: dict[int, tuple[int, int] | None] = {}
        for color, (dr, dc) in DIR_BY_COLOR.items():
            best = None
            best_dist = None
            for dst in points:
                if dst == src:
                    continue
                dst_r, dst_c = dst
                if dr != 0:
                    if dst_c != src_c:
                        continue
                    delta = dst_r - src_r
                    if delta * dr <= 0:
                        continue
                    dist = abs(delta)
                else:
                    if dst_r != src_r:
                        continue
                    delta = dst_c - src_c
                    if delta * dc <= 0:
                        continue
                    dist = abs(delta)
                if best_dist is None or dist < best_dist:
                    best = dst
                    best_dist = dist
            color_targets[int(color)] = best
        targets[src] = color_targets
    return targets


def _simulate(grid: np.ndarray, *, with_trace: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    out = g.copy()
    points, seed, emitters = _extract_layout(g)
    anchor_set = set(points)
    h, w = map(int, out.shape)

    trace: list[dict[str, object]] | None = [] if with_trace else None
    initial_colors = {pos: int(g[pos]) for pos in points}

    for r, c in emitters:
        used_color = int(out[r, c])
        assert used_color in DIR_BY_COLOR, "non-seed instruction pixels must stay directional"
        dr, dc = DIR_BY_COLOR[used_color]
        rr, cc = int(r + dr), int(c + dc)
        path_len = 0
        hit = None
        target_before = None

        while 0 <= rr < h and 0 <= cc < w:
            if (rr, cc) in anchor_set:
                hit = (int(rr), int(cc))
                target_before = int(out[rr, cc])
                out[rr, cc] = used_color
                break
            out[rr, cc] = GREY
            path_len += 1
            rr += dr
            cc += dc

        if trace is not None:
            trace.append(
                {
                    "emitter": (int(r), int(c)),
                    "initial_color": int(initial_colors[(r, c)]),
                    "used_color": int(used_color),
                    "hit": hit,
                    "target_before": target_before,
                    "hit_seed": bool(hit == seed),
                    "path_len": int(path_len),
                }
            )

    if trace is None:
        return out
    return out, trace


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate color-coded instruction pixels that cast grey rays and recolor later targets in scan order.

    Input structure:
    - The background is black.
    - Foreground objects are isolated 1x1 pixels: blue means shoot upward, red rightward, green downward, yellow leftward.
    - Exactly one magenta seed pixel is also present; it can be recolored by incoming rays but it never emits a ray itself.
    - The instruction pixels are read in top-to-bottom, left-to-right order, and their positions stay fixed.

    Rule:
    - Process every original non-seed instruction pixel in scan order.
    - When a pixel is processed, use its current color at that moment to choose a cardinal direction.
    - Draw a one-cell-wide grey ray from the next cell onward until the ray leaves the grid or reaches another original colored pixel position.
    - If the ray reaches another colored pixel position, recolor that target pixel to the emitter's current color, then stop the ray; otherwise the ray simply runs to the edge.

    Key insight:
    - A pixel's color is both state and control flow: earlier hits can recolor later emitters, so the correct behavior comes from sequentially simulating the fixed scan order while updating colors in place.
    """
    return _simulate(grid, with_trace=False)


def _sample_layout(rng: np.random.Generator) -> tuple[tuple[int, int], list[tuple[int, int]]] | None:
    h = int(rng.integers(GRID_SIDE_MIN, GRID_SIDE_MAX + 1))
    w = int(rng.integers(GRID_SIDE_MIN, GRID_SIDE_MAX + 1))
    step = int(_pick_one(STEP_OPTIONS, rng))
    margin = int(_pick_one(MARGIN_OPTIONS, rng))

    row_candidates = list(range(margin, h - margin, step))
    col_candidates = list(range(margin, w - margin, step))
    if len(row_candidates) < ROW_COL_COUNT_MIN or len(col_candidates) < ROW_COL_COUNT_MIN:
        return None

    nr = int(rng.integers(ROW_COL_COUNT_MIN, min(ROW_COL_COUNT_MAX, len(row_candidates)) + 1))
    nc = int(rng.integers(ROW_COL_COUNT_MIN, min(ROW_COL_COUNT_MAX, len(col_candidates)) + 1))
    rows = sorted(map(int, rng.choice(row_candidates, size=nr, replace=False).tolist()))
    cols = sorted(map(int, rng.choice(col_candidates, size=nc, replace=False).tolist()))
    all_points = [(r, c) for r in rows for c in cols]

    point_lo = max(POINT_COUNT_MIN, nr + nc - 1)
    point_hi = min(POINT_COUNT_MAX, len(all_points))
    if point_lo > point_hi:
        return None

    for _ in range(LAYOUT_RETRIES):
        n_points = int(rng.integers(point_lo, point_hi + 1))
        chosen_ids = rng.choice(len(all_points), size=n_points, replace=False)
        points = sorted(all_points[int(idx)] for idx in chosen_ids.tolist())

        row_counts = {r: 0 for r in rows}
        col_counts = {c: 0 for c in cols}
        for r, c in points:
            row_counts[r] += 1
            col_counts[c] += 1

        if any(count == 0 for count in row_counts.values()):
            continue
        if any(count == 0 for count in col_counts.values()):
            continue
        if sum(count >= 2 for count in row_counts.values()) + sum(count >= 2 for count in col_counts.values()) < 2:
            continue
        if not any(row_counts[r] >= 2 or col_counts[c] >= 2 for r, c in points):
            continue
        return (h, w), points
    return None


def _base_colors(n_emitters: int, rng: np.random.Generator) -> list[int]:
    colors = list(map(int, rng.permutation(DIR_COLORS).tolist()))
    if n_emitters <= len(colors):
        colors = colors[:n_emitters]
    else:
        extras = rng.choice(DIR_COLORS, size=n_emitters - len(colors), replace=True).tolist()
        colors.extend(map(int, extras))
    rng.shuffle(colors)
    return colors


def _collect_choices(
    points: list[tuple[int, int]],
    seed: tuple[int, int],
    targets: dict[tuple[int, int], dict[int, tuple[int, int] | None]],
):
    emitters = [pos for pos in points if pos != seed]
    emitter_set = set(emitters)
    emitter_index = {pos: idx for idx, pos in enumerate(emitters)}

    hit_choices: list[tuple[tuple[int, int], int, tuple[int, int]]] = []
    miss_choices: list[tuple[tuple[int, int], int]] = []
    seed_hit_choices: list[tuple[tuple[int, int], int]] = []
    cascade_choices: list[tuple[tuple[int, int], tuple[int, int], int]] = []

    for src in emitters:
        for color in DIR_COLORS.tolist():
            dst = targets[src][int(color)]
            if dst is None:
                miss_choices.append((src, int(color)))
                continue
            hit_choices.append((src, int(color), dst))
            if dst == seed:
                seed_hit_choices.append((src, int(color)))
            if dst in emitter_set and emitter_index[dst] > emitter_index[src]:
                cascade_choices.append((src, dst, int(color)))

    return emitters, hit_choices, miss_choices, seed_hit_choices, cascade_choices


def _build_grid(
    shape: tuple[int, int],
    seed: tuple[int, int],
    emitters: list[tuple[int, int]],
    colors: dict[tuple[int, int], int],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for pos in emitters:
        grid[pos] = int(colors[pos])
    grid[seed] = MAGENTA
    return grid


def _trace_metrics(trace: list[dict[str, object]], out: np.ndarray) -> dict[str, int]:
    hits = sum(int(item["hit"] is not None) for item in trace)
    seed_hits = sum(int(bool(item["hit_seed"])) for item in trace)
    recolored_before_turn = sum(int(item["used_color"] != item["initial_color"]) for item in trace)
    long_rays = sum(int(int(item["path_len"]) >= 2) for item in trace)
    return {
        "hits": int(hits),
        "misses": int(len(trace) - hits),
        "seed_hits": int(seed_hits),
        "recolored_before_turn": int(recolored_before_turn),
        "long_rays": int(long_rays),
        "grey_cells": int(np.count_nonzero(out == GREY)),
    }


def _valid_sample(
    inp: np.ndarray,
    out: np.ndarray,
    trace: list[dict[str, object]],
    scenario: str,
) -> bool:
    metrics = _trace_metrics(trace, out)
    if metrics["hits"] < 2:
        return False
    if metrics["misses"] < 1:
        return False
    if metrics["long_rays"] < 2:
        return False
    if metrics["grey_cells"] < MIN_GREY_CELLS:
        return False
    if metrics["grey_cells"] > int(np.ceil(inp.size * MAX_GREY_FRACTION)):
        return False

    initial_emitters = [int(inp[tuple(item["emitter"])]) for item in trace]
    if len(set(initial_emitters)) < min(4, len(initial_emitters)):
        return False

    if scenario == "cascade" and metrics["recolored_before_turn"] < 1:
        return False
    if scenario == "seeded" and metrics["seed_hits"] < 1:
        return False
    if scenario == "simple" and (metrics["seed_hits"] != 0 or metrics["recolored_before_turn"] != 0):
        return False

    return True


def _candidate_seeds(
    points: list[tuple[int, int]],
    targets: dict[tuple[int, int], dict[int, tuple[int, int] | None]],
    scenario: str,
):
    candidates = []
    for seed in points:
        emitters, hit_choices, miss_choices, seed_hit_choices, cascade_choices = _collect_choices(points, seed, targets)
        if len(emitters) < 4:
            continue
        if not hit_choices or not miss_choices:
            continue
        if scenario == "seeded" and not seed_hit_choices:
            continue
        if scenario == "cascade" and not cascade_choices:
            continue
        candidates.append((seed, emitters, hit_choices, miss_choices, seed_hit_choices, cascade_choices))
    return candidates


def _assignment_attempt(
    shape: tuple[int, int],
    seed: tuple[int, int],
    emitters: list[tuple[int, int]],
    hit_choices: list[tuple[tuple[int, int], int, tuple[int, int]]],
    miss_choices: list[tuple[tuple[int, int], int]],
    seed_hit_choices: list[tuple[tuple[int, int], int]],
    cascade_choices: list[tuple[tuple[int, int], tuple[int, int], int]],
    scenario: str,
    rng: np.random.Generator,
) -> np.ndarray | None:
    colors = {pos: color for pos, color in zip(emitters, _base_colors(len(emitters), rng))}
    forced: set[tuple[int, int]] = set()

    if scenario == "seeded":
        src, color = _pick_one(seed_hit_choices, rng)
        colors[src] = int(color)
        forced.add(src)
    elif scenario == "cascade":
        src, dst, color = _pick_one(cascade_choices, rng)
        colors[src] = int(color)
        alternatives = [int(c) for c in DIR_COLORS.tolist() if int(c) != int(color)]
        colors[dst] = int(_pick_one(alternatives, rng))
        forced.update({src, dst})

    miss_pool = [item for item in miss_choices if item[0] not in forced] or miss_choices
    miss_src, miss_color = _pick_one(miss_pool, rng)
    colors[miss_src] = int(miss_color)

    if scenario == "simple":
        hit_pool = [item for item in hit_choices if item[2] != seed and item[0] != miss_src]
        if not hit_pool:
            hit_pool = [item for item in hit_choices if item[2] != seed] or hit_choices
    else:
        hit_pool = [item for item in hit_choices if item[0] != miss_src] or hit_choices
    hit_src, hit_color, _hit_dst = _pick_one(hit_pool, rng)
    colors[hit_src] = int(hit_color)

    for pos in emitters:
        if pos in forced:
            continue
        roll = int(rng.integers(0, 4))
        if roll == 1:
            hit_colors = [color for color, dst in _direction_targets(emitters + [seed])[pos].items() if dst is not None]
            if hit_colors:
                colors[pos] = int(_pick_one(hit_colors, rng))
        elif roll == 2:
            miss_colors = [color for color, dst in _direction_targets(emitters + [seed])[pos].items() if dst is None]
            if miss_colors:
                colors[pos] = int(_pick_one(miss_colors, rng))

    return _build_grid(shape, seed, emitters, colors)


def _gen_attempt() -> np.ndarray | None:
    rng = _RNG
    layout = _sample_layout(rng)
    if layout is None:
        return None
    shape, points = layout
    targets = _direction_targets(points)
    scenario = str(_pick_one(SCENARIOS, rng))
    seed_options = _candidate_seeds(points, targets, scenario)
    if not seed_options:
        return None

    seed, emitters, hit_choices, miss_choices, seed_hit_choices, cascade_choices = _pick_one(seed_options, rng)

    for _ in range(ASSIGNMENT_RETRIES):
        inp = _assignment_attempt(
            shape,
            seed,
            emitters,
            hit_choices,
            miss_choices,
            seed_hit_choices,
            cascade_choices,
            scenario,
            rng,
        )
        if inp is None:
            continue
        try:
            out, trace = _simulate(inp, with_trace=True)
        except AssertionError:
            continue
        if not _valid_sample(inp, out, trace, scenario):
            continue
        if not np.array_equal(main(inp), out):
            continue
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ray-propagation input")
