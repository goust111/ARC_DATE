import random
from collections import Counter, defaultdict

import numpy as np


YELLOW = 4
DIRS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
PROFILES = (
    (False, False),
    (False, True),
    (True, False),
    (True, True),
)


def _background_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid, dtype=int).ravel())).most_common(1)[0][0])


def _seed_list(grid: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    seeds: list[tuple[int, int, int]] = []
    for r, c in np.argwhere(grid != bg):
        seeds.append((int(r), int(c), int(grid[r, c])))
    return seeds


def _simulate_growth(
    shape: tuple[int, int], seeds: list[tuple[int, int, int]]
) -> tuple[dict[tuple[int, int], Counter], Counter, Counter]:
    h, w = shape
    row_counts = Counter((color, r) for r, _c, color in seeds)
    col_counts = Counter((color, c) for _r, c, color in seeds)
    seed_positions = {(r, c) for r, c, _color in seeds}
    coverage: dict[tuple[int, int], Counter] = defaultdict(Counter)

    for r, c, color in seeds:
        h_len = int(row_counts[(color, r)])
        v_len = int(col_counts[(color, c)])

        for dc in range(1, h_len + 1):
            for cc in (c - dc, c + dc):
                if 0 <= cc < w and (r, cc) not in seed_positions:
                    coverage[(r, cc)][color] += 1

        for dr in range(1, v_len + 1):
            for rr in (r - dr, r + dr):
                if 0 <= rr < h and (rr, c) not in seed_positions:
                    coverage[(rr, c)][color] += 1

    return coverage, row_counts, col_counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each colored seed into a count-controlled cross and mix intersecting colors on background cells.

    Input structure:
    - The grid has a dominant background color and scattered non-adjacent single-pixel colored seeds.
    - Several seed colors may appear, with at most a few seeds per row or column.
    - No larger foreground objects are present; every non-background pixel is a seed.

    Rule:
    - For each seed, count how many same-color seeds lie on its row (including itself); this row count is the horizontal arm length.
    - Count how many same-color seeds lie on its column (including itself); this column count is the vertical arm length.
    - Extend the seed's color left/right by the horizontal count and up/down by the vertical count.
    - Original seed pixels stay unchanged; only background cells are recolored.
    - A background cell covered by one color becomes that color, while a background cell covered by multiple colors becomes yellow.

    Key insight:
    - Cross-arm lengths are not fixed: each seed reads its own same-color row and column multiplicities independently, then color mixing happens only where differently colored arms overlap on background.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _background_color(g)
    seeds = _seed_list(g, bg)
    if not seeds:
        return g.copy()

    coverage, _row_counts, _col_counts = _simulate_growth(g.shape, seeds)
    out = g.copy()
    for (r, c), hits in coverage.items():
        if int(g[r, c]) != bg:
            continue
        out[r, c] = YELLOW if len(hits) > 1 else int(next(iter(hits)))
    return out


def _relative_region(rel_seeds: list[tuple[int, int, int]]) -> set[tuple[int, int]]:
    row_counts = Counter((slot, r) for r, _c, slot in rel_seeds)
    col_counts = Counter((slot, c) for _r, c, slot in rel_seeds)
    seed_positions = {(r, c) for r, c, _slot in rel_seeds}
    region = set(seed_positions)

    for r, c, slot in rel_seeds:
        h_len = int(row_counts[(slot, r)])
        v_len = int(col_counts[(slot, c)])
        for dc in range(1, h_len + 1):
            region.add((r, c - dc))
            region.add((r, c + dc))
        for dr in range(1, v_len + 1):
            region.add((r - dr, c))
            region.add((r + dr, c))
    return region


def _transform_pattern(
    rel_seeds: list[tuple[int, int, int]], turns: int, flip_r: bool, flip_c: bool
) -> list[tuple[int, int, int]]:
    transformed: list[tuple[int, int, int]] = []
    for r, c, slot in rel_seeds:
        rr, cc = int(r), int(c)
        for _ in range(turns % 4):
            rr, cc = cc, -rr
        if flip_r:
            rr = -rr
        if flip_c:
            cc = -cc
        transformed.append((rr, cc, slot))
    return transformed


def _dilate_cells(shape: tuple[int, int], cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    h, w = shape
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr, dc in DIRS8 + ((0, 0),):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w:
                expanded.add((rr, cc))
    return expanded


def _make_single(_rng: random.Random) -> list[tuple[int, int, int]]:
    return [(0, 0, 0)]


def _make_pair_far(rng: random.Random) -> list[tuple[int, int, int]]:
    gap = rng.choice((4, 5, 6))
    return [(0, 0, 0), (0, gap, 0)]


def _make_pair_overlap(rng: random.Random) -> list[tuple[int, int, int]]:
    gap = rng.choice((2, 3))
    return [(0, 0, 0), (0, gap, 0)]


def _make_triple_overlap(rng: random.Random) -> list[tuple[int, int, int]]:
    right = rng.choice((4, 5))
    return [(0, 0, 0), (0, 2, 0), (0, right, 0)]


def _make_yellow_cross_no_same(rng: random.Random) -> list[tuple[int, int, int]]:
    far_h = rng.choice((3, 4))
    far_v = rng.choice((3, 4))
    return [(0, -2, 0), (0, far_h, 0), (-2, 0, 1), (far_v, 0, 1)]


def _make_yellow_cross_same(_rng: random.Random) -> list[tuple[int, int, int]]:
    return [(0, -2, 0), (0, 2, 0), (-2, 0, 1), (2, 0, 1)]


PATTERN_BUILDERS = {
    "single": (_make_single, 1),
    "pair_far": (_make_pair_far, 1),
    "pair_overlap": (_make_pair_overlap, 1),
    "triple_overlap": (_make_triple_overlap, 1),
    "yellow_cross_no_same": (_make_yellow_cross_no_same, 2),
    "yellow_cross_same": (_make_yellow_cross_same, 2),
}


def _analyze_layout(
    shape: tuple[int, int], seeds: list[tuple[int, int, int]]
) -> dict[str, int | bool]:
    coverage, row_counts, col_counts = _simulate_growth(shape, seeds)
    has_yellow = any(len(hits) > 1 for hits in coverage.values())
    has_same_overlap = any(max(hits.values()) >= 2 for hits in coverage.values())
    return {
        "seed_count": len(seeds),
        "color_count": len({color for _r, _c, color in seeds}),
        "has_yellow": has_yellow,
        "has_same_overlap": has_same_overlap,
        "max_row_count": max((int(v) for v in row_counts.values()), default=0),
        "max_col_count": max((int(v) for v in col_counts.values()), default=0),
        "covered_cells": len(coverage),
    }


def _place_pattern(
    shape: tuple[int, int],
    seeds: list[tuple[int, int, int]],
    row_occ: Counter,
    col_occ: Counter,
    region_occ: set[tuple[int, int]],
    seed_blocked: set[tuple[int, int]],
    rel_seeds: list[tuple[int, int, int]],
    slot_colors: list[int],
    rng: random.Random,
    tries: int = 120,
) -> bool:
    h, w = shape
    for _ in range(tries):
        turned = _transform_pattern(
            rel_seeds,
            turns=rng.randrange(4),
            flip_r=bool(rng.randrange(2)),
            flip_c=bool(rng.randrange(2)),
        )
        rel_region = _relative_region(turned)
        rs = [r for r, _c in rel_region]
        cs = [c for _r, c in rel_region]
        min_r, max_r = min(rs), max(rs)
        min_c, max_c = min(cs), max(cs)
        if min_r > h - 1 or min_c > w - 1:
            continue
        base_r_min = -min_r
        base_r_max = h - 1 - max_r
        base_c_min = -min_c
        base_c_max = w - 1 - max_c
        if base_r_min > base_r_max or base_c_min > base_c_max:
            continue

        base_r = rng.randint(base_r_min, base_r_max)
        base_c = rng.randint(base_c_min, base_c_max)
        global_seeds = [(base_r + r, base_c + c, slot_colors[slot]) for r, c, slot in turned]
        seed_cells = {(r, c) for r, c, _color in global_seeds}
        if seed_cells & seed_blocked:
            continue

        row_add = Counter(r for r, _c, _color in global_seeds)
        col_add = Counter(c for _r, c, _color in global_seeds)
        if any(int(row_occ[r] + add) > 3 for r, add in row_add.items()):
            continue
        if any(int(col_occ[c] + add) > 3 for c, add in col_add.items()):
            continue

        global_region = {(base_r + r, base_c + c) for r, c in rel_region}
        if global_region & region_occ:
            continue

        seeds.extend(global_seeds)
        row_occ.update(row_add)
        col_occ.update(col_add)
        region_occ.update(global_region)
        seed_blocked.update(_dilate_cells(shape, seed_cells))
        return True
    return False


def _profile_specs(rng: random.Random, need_yellow: bool, need_same: bool) -> list[str]:
    specs: list[str] = []
    if need_yellow and need_same:
        if rng.random() < 0.5:
            specs.append("yellow_cross_same")
        else:
            specs.extend(("yellow_cross_no_same", rng.choice(("pair_overlap", "triple_overlap"))))
    elif need_yellow:
        specs.append("yellow_cross_no_same")
    elif need_same:
        specs.append(rng.choice(("pair_overlap", "triple_overlap")))
    else:
        specs.append("pair_far")

    extra_choices = ["single", "pair_far"]
    if need_same:
        extra_choices.append("pair_overlap")
        extra_choices.append("triple_overlap")

    min_extra = 1
    max_extra = 1 if len(specs) >= 2 else 2
    for _ in range(rng.randint(min_extra, max_extra)):
        specs.append(rng.choice(extra_choices))
    return specs


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    need_yellow, need_same = rng.choice(PROFILES)
    height = rng.randint(12, 24)
    width = rng.randint(12, 24)
    bg = rng.choice((0, 1, 2, 3, 5, 6, 7, 8, 9))
    colors = [c for c in range(10) if c not in (bg, YELLOW)]
    rng.shuffle(colors)

    specs = _profile_specs(rng, need_yellow, need_same)
    needed_slots = sum(PATTERN_BUILDERS[name][1] for name in specs)
    if needed_slots > len(colors):
        return None

    seeds: list[tuple[int, int, int]] = []
    row_occ: Counter = Counter()
    col_occ: Counter = Counter()
    region_occ: set[tuple[int, int]] = set()
    seed_blocked: set[tuple[int, int]] = set()

    color_idx = 0
    for name in specs:
        builder, slot_count = PATTERN_BUILDERS[name]
        rel_seeds = builder(rng)
        slot_colors = colors[color_idx : color_idx + slot_count]
        color_idx += slot_count
        if not _place_pattern(
            (height, width),
            seeds,
            row_occ,
            col_occ,
            region_occ,
            seed_blocked,
            rel_seeds,
            slot_colors,
            rng,
        ):
            return None

    grid = np.full((height, width), bg, dtype=int)
    for r, c, color in seeds:
        grid[r, c] = color

    features = _analyze_layout(grid.shape, seeds)
    if features["color_count"] < 2:
        return None
    if features["seed_count"] < 3:
        return None
    if features["covered_cells"] <= 0:
        return None
    if features["max_row_count"] <= 1 and features["max_col_count"] <= 1:
        return None
    if bool(features["has_yellow"]) != bool(need_yellow):
        return None
    if bool(features["has_same_overlap"]) != bool(need_same):
        return None

    out = main(grid)
    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    if bool(np.any(out == YELLOW)) != bool(need_yellow):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input after 400 attempts")
