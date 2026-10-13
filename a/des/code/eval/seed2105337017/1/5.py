from __future__ import annotations

import random

import numpy as np


BG = 0
DIVIDER = 5
PALETTE = (1, 2, 3, 4, 6, 8, 9)


def _find_divider(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == DIVIDER)]
    assert len(cols) == 1, "expected exactly one grey divider column"
    return cols[0]


def _find_crossings(panel: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    h, w = panel.shape
    crossings: list[tuple[int, int, int, int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            nw = int(panel[r - 1, c - 1])
            ne = int(panel[r - 1, c + 1])
            sw = int(panel[r + 1, c - 1])
            se = int(panel[r + 1, c + 1])
            center = int(panel[r, c])
            if nw != se or ne != sw:
                continue
            if nw in (BG, DIVIDER) or ne in (BG, DIVIDER) or nw == ne:
                continue
            if center not in (nw, ne):
                continue
            crossings.append((r, c, nw, ne, center))
    return crossings


def _infer_priority(left_panel: np.ndarray) -> dict[int, int]:
    colors = [int(color) for color in np.unique(left_panel) if int(color) not in (BG, DIVIDER)]
    crossings = _find_crossings(left_panel)
    assert crossings, "expected rule examples in the left panel"

    higher_to_lower = {color: set() for color in colors}
    indegree = {color: 0 for color in colors}
    for _, _, diag_a, diag_b, center in crossings:
        higher, lower = (diag_a, diag_b) if center == diag_a else (diag_b, diag_a)
        if lower not in higher_to_lower[higher]:
            higher_to_lower[higher].add(lower)
            indegree[lower] += 1

    order: list[int] = []
    while len(order) < len(colors):
        available = sorted(color for color in colors if indegree[color] == 0 and color not in order)
        assert len(available) == 1, "left panel must define a unique color order"
        current = available[0]
        order.append(current)
        indegree[current] = -1
        for lower in higher_to_lower[current]:
            indegree[lower] -= 1

    return {color: rank for rank, color in enumerate(order)}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read color-over-color precedence from the left panel's diagonal crossing examples,
    then enforce that precedence on the right panel's diagonal crossings.
    """
    g = np.asarray(grid, dtype=int)
    sep = _find_divider(g)
    left = g[:, :sep]
    right = g[:, sep + 1 :]
    assert left.shape[1] == right.shape[1] and left.shape[1] > 0, "panels must have equal positive width"

    rank = _infer_priority(left)
    out = g.copy()
    for r, c, diag_a, diag_b, _ in _find_crossings(right):
        assert diag_a in rank and diag_b in rank, "right-panel colors must be defined on the left"
        top = diag_a if rank[diag_a] < rank[diag_b] else diag_b
        out[r, sep + 1 + c] = top
    return out


MAX_GENERATION_ATTEMPTS = 300
PANEL_WIDTH_RANGE = (11, 14)
GRID_HEIGHT_RANGE = (18, 26)
COLOR_COUNT_CHOICES = (3, 4)
LEFT_ARM_RANGE = (1, 3)
RIGHT_ARM_RANGE = (1, 2)
RIGHT_CROSS_COUNT_RANGE = (3, 5)
PLACEMENT_ATTEMPTS = 200
PLACEMENT_CLEARANCE = 1


def _overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _place_crosses(height: int, width: int, arms: list[int]) -> list[tuple[int, int, int]] | None:
    order = sorted(range(len(arms)), key=lambda idx: arms[idx], reverse=True)
    placements: list[tuple[int, int, int] | None] = [None] * len(arms)
    used: list[tuple[int, int, int, int]] = []

    for idx in order:
        arm = arms[idx]
        row_lo = arm + PLACEMENT_CLEARANCE
        row_hi = height - arm - PLACEMENT_CLEARANCE - 1
        col_lo = arm + PLACEMENT_CLEARANCE
        col_hi = width - arm - PLACEMENT_CLEARANCE - 1
        if row_lo > row_hi or col_lo > col_hi:
            return None

        for _ in range(PLACEMENT_ATTEMPTS):
            r = random.randint(row_lo, row_hi)
            c = random.randint(col_lo, col_hi)
            rect = (
                r - arm - PLACEMENT_CLEARANCE,
                c - arm - PLACEMENT_CLEARANCE,
                r + arm + PLACEMENT_CLEARANCE,
                c + arm + PLACEMENT_CLEARANCE,
            )
            if any(_overlap(rect, other) for other in used):
                continue
            used.append(rect)
            placements[idx] = (r, c, arm)
            break
        else:
            return None

    return [placement for placement in placements if placement is not None]


def _draw_cross(
    panel: np.ndarray,
    center_r: int,
    center_c: int,
    arm: int,
    main_diag_color: int,
    anti_diag_color: int,
    center_color: int,
) -> None:
    for d in range(-arm, arm + 1):
        panel[center_r + d, center_c + d] = main_diag_color
        panel[center_r + d, center_c - d] = anti_diag_color
    panel[center_r, center_c] = center_color


def _sample_right_pairs(priority: list[int]) -> list[tuple[int, int]]:
    all_pairs = [(priority[i], priority[j]) for i in range(len(priority)) for j in range(i + 1, len(priority))]
    direct_pairs = [(priority[i], priority[i + 1]) for i in range(len(priority) - 1)]
    transitive_pairs = [pair for pair in all_pairs if abs(priority.index(pair[0]) - priority.index(pair[1])) >= 2]

    target_count = min(
        len(all_pairs),
        random.randint(RIGHT_CROSS_COUNT_RANGE[0], RIGHT_CROSS_COUNT_RANGE[1]),
    )

    chosen: list[tuple[int, int]] = []
    required = [random.choice(transitive_pairs)]
    if direct_pairs:
        required.append(random.choice(direct_pairs))
    for pair in required:
        if pair not in chosen:
            chosen.append(pair)

    remaining = [pair for pair in all_pairs if pair not in chosen]
    random.shuffle(remaining)
    chosen.extend(remaining[: max(0, target_count - len(chosen))])
    return chosen


def _expected_output(grid: np.ndarray, sep: int, right_specs: list[tuple[int, int, int, int]]) -> np.ndarray:
    out = grid.copy()
    for r, c, _, higher in right_specs:
        out[r, sep + 1 + c] = higher
    return out


def _gen_attempt() -> np.ndarray | None:
    panel_w = random.randint(*PANEL_WIDTH_RANGE)
    height = random.randint(max(GRID_HEIGHT_RANGE[0], panel_w + 5), GRID_HEIGHT_RANGE[1])
    color_count = random.choice(COLOR_COUNT_CHOICES)
    priority = random.sample(PALETTE, color_count)

    left_pairs = [(priority[i], priority[i + 1]) for i in range(color_count - 1)]
    right_pairs = _sample_right_pairs(priority)
    if len(right_pairs) < 3:
        return None

    left_arm_max = 2 if panel_w <= 12 else LEFT_ARM_RANGE[1]
    left_arms = [random.randint(LEFT_ARM_RANGE[0], left_arm_max) for _ in left_pairs]
    right_arms = [random.randint(*RIGHT_ARM_RANGE) for _ in right_pairs]

    left_places = _place_crosses(height, panel_w, left_arms)
    right_places = _place_crosses(height, panel_w, right_arms)
    if left_places is None or right_places is None:
        return None

    left_panel = np.full((height, panel_w), BG, dtype=int)
    for (higher, lower), (r, c, arm) in zip(left_pairs, left_places):
        if random.random() < 0.5:
            main_diag, anti_diag = higher, lower
        else:
            main_diag, anti_diag = lower, higher
        _draw_cross(left_panel, r, c, arm, main_diag, anti_diag, higher)

    right_panel = np.full((height, panel_w), BG, dtype=int)
    wrong_count = random.randint(1, len(right_pairs) - 1)
    wrong_indices = set(random.sample(range(len(right_pairs)), wrong_count))
    right_specs: list[tuple[int, int, int, int]] = []
    for idx, ((higher, lower), (r, c, arm)) in enumerate(zip(right_pairs, right_places)):
        if random.random() < 0.5:
            main_diag, anti_diag = higher, lower
        else:
            main_diag, anti_diag = lower, higher
        center_color = lower if idx in wrong_indices else higher
        _draw_cross(right_panel, r, c, arm, main_diag, anti_diag, center_color)
        right_specs.append((r, c, lower, higher))

    if len(_find_crossings(left_panel)) != len(left_pairs):
        return None
    if len(_find_crossings(right_panel)) != len(right_pairs):
        return None

    grid = np.full((height, 2 * panel_w + 1), BG, dtype=int)
    grid[:, :panel_w] = left_panel
    grid[:, panel_w] = DIVIDER
    grid[:, panel_w + 1 :] = right_panel

    expected = _expected_output(grid, panel_w, right_specs)
    if np.array_equal(grid, expected):
        return None
    if not np.array_equal(main(grid), expected):
        return None

    inferred = _infer_priority(left_panel)
    known = {color: rank for rank, color in enumerate(priority)}
    if inferred != known:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid weaving-priority input")
