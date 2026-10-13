from __future__ import annotations

from functools import lru_cache
import random

import numpy as np


BLACK = 0
GREEN = 3
PINK = 6
ORANGE = 7

MAX_GENERATION_ATTEMPTS = 400
SEED_CLUSTER_GAP = 3
WEDGE_GAP_MIN = 3
MAX_SEEDS_PER_WEDGE = 6


def _cluster_seed_columns(cols: list[int]) -> list[tuple[int, ...]]:
    clusters: list[list[int]] = []
    current: list[int] = []
    for col in cols:
        if not current or col - current[-1] <= SEED_CLUSTER_GAP:
            current.append(int(col))
        else:
            clusters.append(current)
            current = [int(col)]
    if current:
        clusters.append(current)
    return [tuple(cluster) for cluster in clusters]


def _seed_clusters(grid: np.ndarray) -> list[tuple[int, tuple[int, ...]]]:
    clusters: list[tuple[int, tuple[int, ...]]] = []
    for row in np.unique(np.argwhere(grid == ORANGE)[:, 0]):
        cols = sorted(int(c) for c in np.where(grid[int(row)] == ORANGE)[0])
        for cluster in _cluster_seed_columns(cols):
            clusters.append((int(row), cluster))
    clusters.sort()
    return clusters


def _detect_wedges(grid: np.ndarray) -> list[dict[str, object]]:
    green_coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == GREEN)]
    wedges: list[dict[str, object]] = []

    for base_row, seeds in _seed_clusters(grid):
        seed_set = set(int(c) for c in seeds)
        row_oranges = set(int(c) for c in np.where(grid[base_row] == ORANGE)[0])
        min_seed = min(seed_set)
        max_seed = max(seed_set)

        left_bases = {
            int(r + c - base_row)
            for r, c in green_coords
            if r <= base_row and r + c - base_row < min_seed
        }
        right_bases = {
            int(base_row - r + c)
            for r, c in green_coords
            if r <= base_row and base_row - r + c > max_seed
        }

        best: tuple[tuple[int, int, int], dict[str, object]] | None = None
        for left_base in left_bases:
            for right_base in right_bases:
                width = int(right_base - left_base)
                if width <= 2 or width % 2 != 0:
                    continue
                height = width // 2
                tip_col = left_base + height
                tip_row = int(base_row - height)
                if tip_row < 0:
                    continue
                if not (left_base < min_seed and max_seed < right_base):
                    continue

                interior_oranges = {c for c in row_oranges if left_base < c < right_base}
                if interior_oranges != seed_set:
                    continue

                left_markers = {
                    (r, c)
                    for r, c in green_coords
                    if tip_row <= r <= base_row and r + c == base_row + left_base
                }
                right_markers = {
                    (r, c)
                    for r, c in green_coords
                    if tip_row <= r <= base_row and r - c == base_row - right_base
                }
                if not left_markers or not right_markers:
                    continue

                bad_marker = False
                for r, c in green_coords:
                    if not (tip_row <= r <= base_row and left_base <= c <= right_base):
                        continue
                    dist = r - tip_row
                    if c != tip_col - dist and c != tip_col + dist:
                        bad_marker = True
                        break
                if bad_marker:
                    continue

                marker_cells = left_markers | right_markers
                score = (
                    len(marker_cells),
                    int((base_row, left_base) in marker_cells) + int((base_row, right_base) in marker_cells),
                    -width,
                )
                spec = {
                    "tip_row": tip_row,
                    "tip_col": tip_col,
                    "height": height,
                    "base_row": int(base_row),
                    "left_base": int(left_base),
                    "right_base": int(right_base),
                    "seed_cols": tuple(sorted(seed_set)),
                }
                if best is None or score > best[0]:
                    best = (score, spec)

        assert best is not None, f"failed to infer wedge for seed row {base_row}, seeds {seeds}"
        wedges.append(best[1])

    return wedges


def _children_from_active(active_cols: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(
        left + 1
        for left, right in zip(active_cols, active_cols[1:])
        if right - left == 2
    )


def _grow_wedge(out: np.ndarray, wedge: dict[str, object]) -> None:
    tip_row = int(wedge["tip_row"])
    tip_col = int(wedge["tip_col"])
    current_row = int(wedge["base_row"])
    active_cols = tuple(int(c) for c in wedge["seed_cols"])
    color = PINK

    while current_row - 2 >= 0:
        target_row = current_row - 2
        raw_children = _children_from_active(active_cols)
        if not raw_children:
            break

        dist = target_row - tip_row
        left_bound = tip_col - dist
        right_bound = tip_col + dist
        children = tuple(c for c in raw_children if left_bound <= c <= right_bound)
        if not children:
            break

        for col in children:
            if out[target_row, col] == BLACK:
                out[target_row, col] = color
        active_cols = children
        current_row = target_row
        color = ORANGE if color == PINK else PINK


def main(grid: np.ndarray) -> np.ndarray:
    """Grow alternating descendants upward inside green-marked triangular wedges.

    Input structure:
    - A black background contains two or three upward-pointing triangular wedges.
    - Each wedge is indicated by sparse green pixels lying on its two 45-degree side boundaries.
    - The wedge base row contains orange seed pixels strictly inside the boundary corners, with one or two blank cells between neighboring seeds.

    Rule:
    - Preserve all green marker pixels and all orange seed pixels.
    - For each wedge independently, treat its orange base row as the first active row and grow upward toward the tip.
    - Whenever an active row contains an active-blank-active triplet, place a child two rows above the blank midpoint; first-generation children are pink, then colors alternate pink and orange on later generations.
    - Discard any child whose column would fall outside that target row's wedge span between the two diagonal boundaries.
    - Stop when no legal child can be created or when the next active row would leave the grid.

    Key insight:
    - The orange growth rule alone is not enough: each generation is additionally clipped by the wedge's narrowing diagonal sides, so the sparse green boundary clues determine how far every branch can survive.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for wedge in _detect_wedges(g):
        _grow_wedge(out, wedge)
    return out


def _simulate_local_growth(seed_cols: tuple[int, ...], height: int) -> dict[str, object]:
    active_cols = tuple(int(c) for c in seed_cols)
    current_dist = int(height)
    generations: list[tuple[int, tuple[int, ...]]] = []
    clipped = False

    while current_dist - 2 >= 0:
        target_dist = current_dist - 2
        raw_children = _children_from_active(active_cols)
        if not raw_children:
            break
        left_bound = height - target_dist
        right_bound = height + target_dist
        kept_children = tuple(c for c in raw_children if left_bound <= c <= right_bound)
        if len(kept_children) < len(raw_children):
            clipped = True
        if not kept_children:
            break
        generations.append((target_dist, kept_children))
        active_cols = kept_children
        current_dist = target_dist

    generated_cells = {
        (dist, col)
        for dist, cols in generations
        for col in cols
    }
    return {
        "levels": len(generations),
        "clipped": clipped,
        "generated_cells": generated_cells,
        "total_children": sum(len(cols) for _dist, cols in generations),
    }


@lru_cache(maxsize=None)
def _pattern_catalog(height: int) -> tuple[dict[str, object], ...]:
    max_col = 2 * int(height) - 1
    records: list[dict[str, object]] = []

    def rec(pattern: tuple[int, ...]) -> None:
        if 3 <= len(pattern) <= MAX_SEEDS_PER_WEDGE:
            sim = _simulate_local_growth(pattern, int(height))
            records.append(
                {
                    "seed_cols": pattern,
                    "levels": int(sim["levels"]),
                    "clipped": bool(sim["clipped"]),
                    "generated_cells": frozenset(sim["generated_cells"]),
                    "total_children": int(sim["total_children"]),
                    "seed_count": len(pattern),
                }
            )
        if len(pattern) == MAX_SEEDS_PER_WEDGE:
            return

        if not pattern:
            next_choices = range(1, max_col + 1)
        else:
            next_choices = (
                pattern[-1] + 2,
                pattern[-1] + 3,
            )

        for col in next_choices:
            if col > max_col:
                continue
            rec(pattern + (int(col),))

    rec(())
    return tuple(records)


def _pick_pattern(height: int, category: str) -> dict[str, object] | None:
    catalog = list(_pattern_catalog(int(height)))
    if category == "deep":
        options = [rec for rec in catalog if int(rec["levels"]) >= 2]
    elif category == "clip":
        options = [rec for rec in catalog if bool(rec["clipped"]) and int(rec["levels"]) >= 1]
    elif category == "shallow":
        options = [
            rec
            for rec in catalog
            if int(rec["levels"]) <= 1 and not bool(rec["clipped"])
        ]
        if not options:
            options = [rec for rec in catalog if int(rec["levels"]) <= 1]
    else:
        options = catalog

    if not options:
        return None

    weights = []
    for rec in options:
        if category == "deep":
            weight = 3 * int(rec["levels"]) + int(rec["total_children"])
        elif category == "clip":
            weight = 2 * int(rec["total_children"]) + int(rec["seed_count"])
        else:
            weight = int(rec["seed_count"]) + 1
        weights.append(weight)
    return random.choices(options, weights=weights, k=1)[0]


def _choose_two_side_by_side(widths: list[int], total_width: int) -> tuple[int, int]:
    max_gap = total_width - (widths[0] + widths[1])
    assert max_gap >= WEDGE_GAP_MIN
    gap = random.randint(WEDGE_GAP_MIN, min(WEDGE_GAP_MIN + 2, max_gap))
    free = total_width - (widths[0] + widths[1] + gap)
    assert free >= 0
    left_margin = random.randint(0, free)
    left0 = left_margin
    left1 = left0 + widths[0] + gap
    return left0, left1


def _band_base_row(top_bound: int, max_height: int, height_limit: int) -> int:
    low = max_height + top_bound
    high = min(height_limit, low + 4)
    assert low <= high
    return random.randint(low, high)


def _choose_layout() -> list[dict[str, object]] | None:
    count = random.choice([2, 3])

    if count == 2:
        categories = ["deep", random.choice(["clip", "shallow"])]
        random.shuffle(categories)
        heights = []
        for category in categories:
            if category == "deep":
                heights.append(random.randint(6, 8))
            else:
                heights.append(random.randint(5, 6))

        widths = [2 * h + 1 for h in heights]
        side_by_side = widths[0] + widths[1] + WEDGE_GAP_MIN <= 30 and random.random() < 0.65
        if side_by_side:
            total_width = random.randint(max(18, widths[0] + widths[1] + WEDGE_GAP_MIN), 30)
            left0, left1 = _choose_two_side_by_side(widths, total_width)
            base_row = random.randint(max(heights) + 1, min(15, max(heights) + 5))
            total_height = random.randint(base_row + 4, min(30, base_row + 10))
            return [
                {
                    "category": categories[0],
                    "height": heights[0],
                    "left_base": left0,
                    "base_row": base_row,
                },
                {
                    "category": categories[1],
                    "height": heights[1],
                    "left_base": left1,
                    "base_row": base_row,
                },
                {
                    "grid_shape": (total_height, total_width),
                },
            ]

        total_width = random.randint(max(widths) + 2, 30)
        first_left = random.randint(0, total_width - widths[0])
        second_left = random.randint(0, total_width - widths[1])
        top_base = random.randint(heights[0] + 1, min(11, heights[0] + 4))
        bottom_base_low = top_base + heights[1] + 4
        bottom_base_high = min(27, bottom_base_low + 4)
        if bottom_base_low > bottom_base_high:
            return None
        bottom_base = random.randint(bottom_base_low, bottom_base_high)
        total_height = random.randint(bottom_base + 2, min(30, bottom_base + 4))
        return [
            {
                "category": categories[0],
                "height": heights[0],
                "left_base": first_left,
                "base_row": top_base,
            },
            {
                "category": categories[1],
                "height": heights[1],
                "left_base": second_left,
                "base_row": bottom_base,
            },
            {
                "grid_shape": (total_height, total_width),
            },
        ]

    small_heights = [random.randint(5, 6), random.randint(5, 6)]
    if 2 * small_heights[0] + 1 + 2 * small_heights[1] + 1 + WEDGE_GAP_MIN > 30:
        return None
    deep_height = random.randint(6, 8)
    categories = ["clip", "shallow"]
    random.shuffle(categories)
    deep_on_top = random.choice([False, True])
    total_width = random.randint(
        max(22, (2 * small_heights[0] + 1) + (2 * small_heights[1] + 1) + WEDGE_GAP_MIN),
        30,
    )
    left0, left1 = _choose_two_side_by_side([2 * h + 1 for h in small_heights], total_width)
    deep_width = 2 * deep_height + 1
    deep_left = random.randint(0, total_width - deep_width)

    if deep_on_top:
        top_base = random.randint(deep_height + 1, min(11, deep_height + 4))
        bottom_low = top_base + max(small_heights) + 4
        bottom_high = min(27, bottom_low + 3)
        if bottom_low > bottom_high:
            return None
        bottom_base = random.randint(bottom_low, bottom_high)
        total_height = random.randint(bottom_base + 2, min(30, bottom_base + 4))
        wedges = [
            {"category": "deep", "height": deep_height, "left_base": deep_left, "base_row": top_base},
            {"category": categories[0], "height": small_heights[0], "left_base": left0, "base_row": bottom_base},
            {"category": categories[1], "height": small_heights[1], "left_base": left1, "base_row": bottom_base},
        ]
    else:
        top_base = random.randint(max(small_heights) + 1, min(12, max(small_heights) + 4))
        bottom_low = top_base + deep_height + 4
        bottom_high = min(27, bottom_low + 3)
        if bottom_low > bottom_high:
            return None
        bottom_base = random.randint(bottom_low, bottom_high)
        total_height = random.randint(bottom_base + 2, min(30, bottom_base + 4))
        wedges = [
            {"category": categories[0], "height": small_heights[0], "left_base": left0, "base_row": top_base},
            {"category": categories[1], "height": small_heights[1], "left_base": left1, "base_row": top_base},
            {"category": "deep", "height": deep_height, "left_base": deep_left, "base_row": bottom_base},
        ]

    wedges.append({"grid_shape": (total_height, total_width)})
    return wedges


def _wedge_generated_cells(spec: dict[str, object], pattern: dict[str, object]) -> set[tuple[int, int]]:
    tip_row = int(spec["base_row"]) - int(spec["height"])
    left_base = int(spec["left_base"])
    cells = set()
    for dist, col in pattern["generated_cells"]:
        row = tip_row + int(dist)
        cells.add((row, left_base + int(col)))
    return cells


def _pick_marker_cells(spec: dict[str, object], pattern: dict[str, object]) -> set[tuple[int, int]]:
    height = int(spec["height"])
    base_row = int(spec["base_row"])
    left_base = int(spec["left_base"])
    right_base = left_base + 2 * height
    tip_row = base_row - height
    tip_col = left_base + height
    generated = _wedge_generated_cells(spec, pattern)

    markers = {(base_row, left_base), (base_row, right_base)}
    candidate_cells = []
    for dist in range(height):
        row = tip_row + dist
        left = (row, tip_col - dist)
        right = (row, tip_col + dist)
        for cell in (left, right):
            if cell not in markers and cell not in generated:
                candidate_cells.append(cell)

    extra_count = random.randint(0, min(2, len(candidate_cells)))
    if extra_count:
        markers.update(random.sample(candidate_cells, extra_count))
    return markers


def _render_input(shape: tuple[int, int], wedge_specs: list[dict[str, object]]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for spec in wedge_specs:
        height = int(spec["height"])
        base_row = int(spec["base_row"])
        left_base = int(spec["left_base"])
        tip_col = left_base + height
        pattern = spec["pattern"]

        for cell in _pick_marker_cells(spec, pattern):
            grid[cell] = GREEN

        for local_col in pattern["seed_cols"]:
            grid[base_row, left_base + int(local_col)] = ORANGE

        assert tip_col - height == left_base
        assert tip_col + height == left_base + 2 * height
    return grid


def _build_attempt() -> np.ndarray | None:
    layout = _choose_layout()
    if layout is None:
        return None
    grid_shape = tuple(layout.pop()["grid_shape"])
    wedges = layout

    placed_specs: list[dict[str, object]] = []
    clipping_present = False
    total_levels = 0
    for spec in wedges:
        pattern = _pick_pattern(int(spec["height"]), str(spec["category"]))
        if pattern is None:
            return None
        spec = dict(spec)
        spec["pattern"] = pattern
        spec["tip_row"] = int(spec["base_row"]) - int(spec["height"])
        spec["tip_col"] = int(spec["left_base"]) + int(spec["height"])
        spec["right_base"] = int(spec["left_base"]) + 2 * int(spec["height"])
        if not (0 <= spec["tip_row"] < spec["base_row"] < grid_shape[0]):
            return None
        if not (0 <= spec["left_base"] and spec["right_base"] < grid_shape[1]):
            return None
        clipping_present |= bool(pattern["clipped"])
        total_levels += int(pattern["levels"])
        placed_specs.append(spec)

    if total_levels < 2:
        return None
    if not clipping_present:
        return None

    grid = _render_input(grid_shape, placed_specs)
    if np.any(grid == PINK):
        return None

    out = main(grid)
    if not np.any(out == PINK):
        return None
    if np.count_nonzero(out != grid) < 4:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid wedge-growth input")
