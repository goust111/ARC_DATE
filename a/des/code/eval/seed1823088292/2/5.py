from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))

# --------------- solver ---------------


def _cross_cells(height: int, width: int, row: int, col: int, arm: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for rr in range(max(1, row - arm), min(height - 1, row + arm) + 1):
        cells.add((rr, col))
    for cc in range(max(1, col - arm), min(width - 1, col + arm) + 1):
        cells.add((row, cc))
    return cells


def _paint_cross(grid: np.ndarray, row: int, col: int, arm: int, color: int) -> None:
    height, width = map(int, grid.shape)
    for rr in range(max(1, row - arm), min(height - 1, row + arm) + 1):
        grid[rr, col] = color
    for cc in range(max(1, col - arm), min(width - 1, col + arm) + 1):
        grid[row, cc] = color


def _extract_specs(grid: np.ndarray) -> list[dict[str, int]]:
    height, width = map(int, grid.shape)
    assert height >= 4 and width >= 4, "grid too small for rulers plus canvas"
    assert int(grid[0, 0]) == BG, "shared ruler corner must stay background"

    top_colors = [int(c) for c in np.unique(grid[0, 1:]) if int(c) != BG]
    left_colors = [int(c) for c in np.unique(grid[1:, 0]) if int(c) != BG]
    assert top_colors, "expected at least one ruler color"
    assert set(top_colors) == set(left_colors), "top and left rulers must use the same colors"

    color_order = sorted(
        top_colors,
        key=lambda color: int(np.flatnonzero(grid[0, 1:] == color)[0]),
    )

    specs: list[dict[str, int]] = []
    for color in color_order:
        cols = np.flatnonzero(grid[0, 1:] == color) + 1
        rows = np.flatnonzero(grid[1:, 0] == color) + 1
        assert len(cols) == 2 and len(rows) == 2, "each ruler color must appear exactly twice on both rulers"

        c1, c2 = map(int, np.sort(cols))
        r1, r2 = map(int, np.sort(rows))
        arm_top = int(c2 - c1 - 1)
        arm_left = int(r2 - r1 - 1)

        assert arm_top == arm_left, "top and left ruler gaps must match"
        assert arm_top >= 1 and arm_top % 2 == 1, "gap must be a positive odd value so the midpoint is integral"
        assert (c1 + c2) % 2 == 0 and (r1 + r2) % 2 == 0, "cross center must land on a grid cell"

        specs.append(
            {
                "color": int(color),
                "row": int((r1 + r2) // 2),
                "col": int((c1 + c2) // 2),
                "arm": int(arm_top),
                "top_start": int(c1),
                "left_start": int(r1),
            }
        )

    return specs


def _paint_order(specs: list[dict[str, int]]) -> list[dict[str, int]]:
    # The prompt states that smaller crosses should remain on top after overlaps,
    # so larger arms are painted first and smaller arms are painted last.
    return sorted(specs, key=lambda spec: (-spec["arm"], spec["top_start"], spec["left_start"]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode paired top/left ruler marks into cross centers and arm lengths,
    then draw the colored crosses on the interior canvas.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    for spec in _paint_order(_extract_specs(g)):
        _paint_cross(out, spec["row"], spec["col"], spec["arm"], spec["color"])
    return out


# --------------- generator ---------------

MODES = ("plain", "trunc_only", "overlap_only", "both")
HEIGHT_RANGE = (12, 22)
WIDTH_RANGE = (12, 22)
MIN_COLOR_COUNT = 2
MAX_COLOR_COUNT = 4
GEN_ATTEMPTS = 500


def _all_arms(height: int, width: int) -> list[int]:
    max_arm = min(height, width) - 3
    return [arm for arm in range(1, max_arm + 1, 2)]


def _safe_arms(height: int, width: int) -> list[int]:
    max_arm = min((height - 2) // 2, (width - 2) // 2)
    return [arm for arm in range(1, max_arm + 1, 2)]


def _pair_from_center(center: int, arm: int) -> tuple[int, int]:
    half_span = (arm + 1) // 2
    return center - half_span, center + half_span


def _centers_for_arm(size: int, arm: int, safe_only: bool = False, edge_only: bool = False) -> list[int]:
    half_span = (arm + 1) // 2
    centers = list(range(1 + half_span, size - half_span))
    if safe_only:
        centers = [center for center in centers if center - arm >= 1 and center + arm <= size - 1]
    if edge_only:
        centers = [center for center in centers if center - arm < 1 or center + arm > size - 1]
    return centers


def _group_options(
    size: int,
    arms: list[int],
    group: tuple[int, ...],
    safe_only: bool,
    edge_index: int | None,
) -> list[dict[int, tuple[int, int, int]]]:
    if len(group) == 1:
        idx = group[0]
        centers = _centers_for_arm(size, arms[idx], safe_only=safe_only, edge_only=(idx == edge_index))
        options: list[dict[int, tuple[int, int, int]]] = []
        for center in centers:
            p1, p2 = _pair_from_center(center, arms[idx])
            options.append({idx: (p1, p2, center)})
        return options

    i, j = group
    centers_i = set(_centers_for_arm(size, arms[i], safe_only=safe_only, edge_only=(i == edge_index)))
    centers_j = set(_centers_for_arm(size, arms[j], safe_only=safe_only, edge_only=(j == edge_index)))
    options = []
    for center in centers_i & centers_j:
        p1_i, p2_i = _pair_from_center(center, arms[i])
        p1_j, p2_j = _pair_from_center(center, arms[j])
        if len({p1_i, p2_i, p1_j, p2_j}) != 4:
            continue
        options.append(
            {
                i: (p1_i, p2_i, center),
                j: (p1_j, p2_j, center),
            }
        )
    return options


def _sample_axis_pairs(
    size: int,
    arms: list[int],
    safe_only: bool,
    share_pair: tuple[int, int] | None,
    edge_index: int | None,
) -> list[tuple[int, int, int]] | None:
    groups: list[tuple[int, ...]] = []
    if share_pair is not None:
        groups.append(tuple(sorted(share_pair)))
    for idx in range(len(arms)):
        if share_pair is not None and idx in share_pair:
            continue
        groups.append((idx,))

    options_by_group = {
        group: _group_options(size, arms, group, safe_only=safe_only, edge_index=edge_index)
        for group in groups
    }
    if any(not options for options in options_by_group.values()):
        return None

    groups.sort(key=lambda group: len(options_by_group[group]))

    def search(
        group_pos: int,
        used_positions: set[int],
        assignment: dict[int, tuple[int, int, int]],
    ) -> dict[int, tuple[int, int, int]] | None:
        if group_pos == len(groups):
            return assignment

        group = groups[group_pos]
        options = options_by_group[group][:]
        random.shuffle(options)
        for option in options:
            new_positions: list[int] = []
            ok = True
            for p1, p2, _center in option.values():
                if p1 in used_positions or p2 in used_positions:
                    ok = False
                    break
                new_positions.extend((p1, p2))
            if not ok:
                continue

            next_assignment = assignment.copy()
            next_assignment.update(option)
            result = search(group_pos + 1, used_positions | set(new_positions), next_assignment)
            if result is not None:
                return result
        return None

    result = search(0, set(), {})
    if result is None:
        return None
    return [result[idx] for idx in range(len(arms))]


def _scene_features(height: int, width: int, specs: list[dict[str, int]]) -> dict[str, int | bool]:
    counts: Counter[tuple[int, int]] = Counter()
    has_truncation = False
    for spec in specs:
        row = spec["row"]
        col = spec["col"]
        arm = spec["arm"]
        if row - arm < 1 or row + arm > height - 1 or col - arm < 1 or col + arm > width - 1:
            has_truncation = True
        for cell in _cross_cells(height, width, row, col, arm):
            counts[cell] += 1

    overlap_cells = sum(count > 1 for count in counts.values())
    max_stack = max(counts.values(), default=0)
    return {
        "has_overlap": overlap_cells > 0,
        "has_truncation": has_truncation,
        "overlap_cells": int(overlap_cells),
        "max_stack": int(max_stack),
        "painted_cells": int(len(counts)),
    }


def _render_from_specs(grid: np.ndarray, specs: list[dict[str, int]]) -> np.ndarray:
    out = grid.copy()
    for spec in _paint_order(specs):
        _paint_cross(out, spec["row"], spec["col"], spec["arm"], spec["color"])
    return out


def _gen_attempt() -> np.ndarray | None:
    mode = random.choice(MODES)
    height = random.randint(*HEIGHT_RANGE)
    width = random.randint(*WIDTH_RANGE)

    safe_arms = _safe_arms(height, width)
    all_arms = _all_arms(height, width)
    if len(all_arms) < MIN_COLOR_COUNT:
        return None

    require_overlap = mode in {"overlap_only", "both"}
    require_truncation = mode in {"trunc_only", "both"}

    if require_truncation:
        trunc_pool = [arm for arm in all_arms if arm >= 3]
        if not trunc_pool:
            return None
        max_count = min(MAX_COLOR_COUNT, len(all_arms))
        color_count = random.randint(MIN_COLOR_COUNT, max_count)
        lead_arm = random.choice(trunc_pool)
        remaining = [arm for arm in all_arms if arm != lead_arm]
        if len(remaining) < color_count - 1:
            return None
        arms = [lead_arm] + random.sample(remaining, color_count - 1)
    else:
        if len(safe_arms) < MIN_COLOR_COUNT:
            return None
        max_count = min(MAX_COLOR_COUNT, len(safe_arms))
        color_count = random.randint(MIN_COLOR_COUNT, max_count)
        arms = random.sample(safe_arms, color_count)

    random.shuffle(arms)
    colors = random.sample(PALETTE, color_count)

    share_pair = tuple(random.sample(range(color_count), 2)) if require_overlap else None
    share_axis = random.choice(("row", "col")) if require_overlap else None

    edge_index = random.choice([idx for idx, arm in enumerate(arms) if arm >= 3]) if require_truncation else None
    edge_axis = random.choice(("row", "col")) if require_truncation else None

    row_pairs = _sample_axis_pairs(
        height,
        arms,
        safe_only=not require_truncation,
        share_pair=share_pair if share_axis == "row" else None,
        edge_index=edge_index if edge_axis == "row" else None,
    )
    if row_pairs is None:
        return None

    col_pairs = _sample_axis_pairs(
        width,
        arms,
        safe_only=not require_truncation,
        share_pair=share_pair if share_axis == "col" else None,
        edge_index=edge_index if edge_axis == "col" else None,
    )
    if col_pairs is None:
        return None

    grid = np.full((height, width), BG, dtype=int)
    specs: list[dict[str, int]] = []
    for color, arm, row_info, col_info in zip(colors, arms, row_pairs, col_pairs):
        r1, r2, row = row_info
        c1, c2, col = col_info
        grid[0, c1] = color
        grid[0, c2] = color
        grid[r1, 0] = color
        grid[r2, 0] = color
        specs.append(
            {
                "color": int(color),
                "row": int(row),
                "col": int(col),
                "arm": int(arm),
                "top_start": int(c1),
                "left_start": int(r1),
            }
        )

    features = _scene_features(height, width, specs)
    if bool(features["has_overlap"]) != require_overlap:
        return None
    if bool(features["has_truncation"]) != require_truncation:
        return None
    if int(features["max_stack"]) > 2:
        return None
    if int(features["overlap_cells"]) > 8:
        return None
    if int(features["painted_cells"]) > int((height - 1) * (width - 1) * 0.45):
        return None

    expected = _render_from_specs(grid, specs)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    extracted = _extract_specs(grid)
    if sorted((spec["color"], spec["row"], spec["col"], spec["arm"]) for spec in extracted) != sorted(
        (spec["color"], spec["row"], spec["col"], spec["arm"]) for spec in specs
    ):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ruler-cross input")
