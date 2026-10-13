from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

COLOR_CYCLE = (BLUE, RED, YELLOW, ORANGE)
MAX_GENERATION_ATTEMPTS = 600


def _find_cross(grid: np.ndarray) -> tuple[int, int]:
    h, w = grid.shape
    rows = [r for r in range(h) if np.all(grid[r, :] == GREEN)]
    cols = [c for c in range(w) if np.all(grid[:, c] == GREEN)]
    assert len(rows) == 1, "expected exactly one full green row"
    assert len(cols) == 1, "expected exactly one full green column"
    cr, cc = rows[0], cols[0]
    mask = np.zeros_like(grid, dtype=bool)
    mask[cr, :] = True
    mask[:, cc] = True
    assert np.array_equal(grid == GREEN, mask), "green pixels must form exactly one solid cross"
    assert 0 < cr < h - 1 and 0 < cc < w - 1, "cross intersection must be off the border"
    return int(cr), int(cc)


def _largest_square_from_top_left(black: np.ndarray) -> np.ndarray:
    h, w = black.shape
    dp = np.zeros((h + 1, w + 1), dtype=int)
    for r in range(h - 1, -1, -1):
        for c in range(w - 1, -1, -1):
            if black[r, c]:
                dp[r, c] = 1 + min(dp[r + 1, c], dp[r, c + 1], dp[r + 1, c + 1])
    return dp[:h, :w]


def _contains_square(big: tuple[int, int, int], small: tuple[int, int, int]) -> bool:
    br, bc, bs = big
    sr, sc, ss = small
    return br <= sr and bc <= sc and br + bs - 1 >= sr + ss - 1 and bc + bs - 1 >= sc + ss - 1


def _enumerate_maximal_black_squares(grid: np.ndarray, cr: int, cc: int) -> list[dict[str, int]]:
    black = grid == BLACK
    max_side = _largest_square_from_top_left(black)
    starts: list[tuple[int, int, int]] = []
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            side = int(max_side[r, c])
            if side >= 2:
                starts.append((r, c, side))

    maximal: list[dict[str, int]] = []
    for square in starts:
        if any(_contains_square(other, square) for other in starts if other[2] > square[2]):
            continue
        r, c, side = square
        center_r2 = 2 * r + side - 1
        center_c2 = 2 * c + side - 1
        dist2 = max(abs(center_r2 - 2 * cr), abs(center_c2 - 2 * cc))
        maximal.append(
            {
                "r": int(r),
                "c": int(c),
                "side": int(side),
                "dist2": int(dist2),
            }
        )
    return maximal


def _overlap(a: dict[str, int], b: dict[str, int]) -> bool:
    return not (
        a["r"] + a["side"] <= b["r"]
        or b["r"] + b["side"] <= a["r"]
        or a["c"] + a["side"] <= b["c"]
        or b["c"] + b["side"] <= a["c"]
    )


def _order_key(square: dict[str, int], mode: str) -> tuple[int, ...]:
    if mode == "actual":
        return (-square["side"], square["dist2"], square["r"], square["c"])
    if mode == "no_distance":
        return (-square["side"], square["r"], square["c"], square["dist2"])
    if mode == "distance_first":
        return (square["dist2"], -square["side"], square["r"], square["c"])
    raise ValueError(f"unknown mode: {mode}")


def _select_squares(
    candidates: list[dict[str, int]],
    mode: str = "actual",
) -> tuple[list[dict[str, int]], bool]:
    ordered = sorted(candidates, key=lambda sq: _order_key(sq, mode))
    selected: list[dict[str, int]] = []
    rejected_overlap = False
    for square in ordered:
        if any(_overlap(square, kept) for kept in selected):
            rejected_overlap = True
            continue
        selected.append(square)
    return selected, rejected_overlap


def _paint_selected(grid: np.ndarray, selected: list[dict[str, int]]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for idx, square in enumerate(selected):
        color = COLOR_CYCLE[idx % len(COLOR_CYCLE)]
        r, c, side = square["r"], square["c"], square["side"]
        out[r : r + side, c : c + side] = color
    return out


def _selected_signature(selected: list[dict[str, int]]) -> tuple[tuple[int, int, int], ...]:
    return tuple((sq["r"], sq["c"], sq["side"]) for sq in selected)


def _solve(grid: np.ndarray, mode: str = "actual") -> tuple[np.ndarray, list[dict[str, int]], list[dict[str, int]], bool]:
    cr, cc = _find_cross(grid)
    candidates = _enumerate_maximal_black_squares(grid, cr, cc)
    selected, rejected_overlap = _select_squares(candidates, mode=mode)
    return _paint_selected(grid, selected), candidates, selected, rejected_overlap


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the greedy non-overlapping maximal black squares, ranked by size then cross-center distance.

    Input structure:
    - The background is black.
    - One full green row and one full green column form a single solid cross whose intersection is internal.
    - Extra grey obstacle pixels or short grey segments lie in the black regions and block square growth.

    Rule:
    - Treat only black cells as fillable; green cross cells and grey obstacle cells are blockers.
    - Enumerate every maximal all-black square of side at least 2.
    - Sort candidates by side length descending, then by the square center's Chebyshev distance to the green cross intersection ascending.
    - Greedily keep each candidate iff it does not overlap an earlier kept square, and fill kept squares cyclically with blue, red, yellow, orange.
    - Leave the green cross and grey obstacles unchanged.

    Key insight:
    - The cross intersection is not painted; it is the external reference point that breaks same-size square ties, while greedy non-overlap decides which competing maximal pockets survive.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    out, _candidates, _selected, _rejected_overlap = _solve(g, mode="actual")
    return out


def _quadrant_bounds(shape: tuple[int, int], cr: int, cc: int) -> dict[str, tuple[int, int, int, int]]:
    h, w = shape
    return {
        "tl": (0, cr, 0, cc),
        "tr": (0, cr, cc + 1, w),
        "bl": (cr + 1, h, 0, cc),
        "br": (cr + 1, h, cc + 1, w),
    }


def _opposite_quad(name: str) -> str:
    return {"tl": "br", "tr": "bl", "bl": "tr", "br": "tl"}[name]


def _build_base_grid() -> tuple[np.ndarray, str]:
    short = random.randint(3, 4)
    long_side = random.randint(2 * short, min(11, 2 * short + 5))
    other_h = random.randint(max(short + 2, 5), 10)
    other_w = random.randint(max(short + 2, 5), 10)
    active_quad = random.choice(("tl", "tr", "bl", "br"))
    horizontal = random.random() < 0.5
    active_h = short if horizontal else long_side
    active_w = long_side if horizontal else short

    if active_quad in ("tl", "tr"):
        top, bottom = active_h, other_h
    else:
        top, bottom = other_h, active_h

    if active_quad in ("tl", "bl"):
        left, right = active_w, other_w
    else:
        left, right = other_w, active_w

    h = top + bottom + 1
    w = left + right + 1
    grid = np.full((h, w), BLACK, dtype=int)
    grid[top, :] = GREEN
    grid[:, left] = GREEN
    return grid, active_quad


def _paint_segment(
    grid: np.ndarray,
    bounds: tuple[int, int, int, int],
    length: int,
    orientation: str,
    *,
    interior_only: bool,
) -> bool:
    r0, r1, c0, c1 = bounds
    if interior_only:
        r0 += 1
        c0 += 1
        r1 -= 1
        c1 -= 1
    if r0 >= r1 or c0 >= c1:
        return False

    if orientation == "h":
        if c1 - c0 < length:
            return False
        r = random.randint(r0, r1 - 1)
        c = random.randint(c0, c1 - length)
        cells = [(r, c + dc) for dc in range(length)]
    else:
        if r1 - r0 < length:
            return False
        r = random.randint(r0, r1 - length)
        c = random.randint(c0, c1 - 1)
        cells = [(r + dr, c) for dr in range(length)]

    if any(grid[rr, cc] != BLACK for rr, cc in cells):
        return False
    for rr, cc in cells:
        grid[rr, cc] = GREY
    return True


def _square_quad(square: dict[str, int], cr: int, cc: int) -> str:
    if square["r"] + square["side"] - 1 < cr:
        return "tl" if square["c"] + square["side"] - 1 < cc else "tr"
    return "bl" if square["c"] + square["side"] - 1 < cc else "br"


def _candidate_keys_unique(candidates: list[dict[str, int]]) -> bool:
    seen = set()
    for sq in candidates:
        key = (sq["side"], sq["dist2"])
        if key in seen:
            return False
        seen.add(key)
    return True


def _basic_generator_state_ok(grid: np.ndarray) -> bool:
    try:
        _out, candidates, selected, rejected_overlap = _solve(grid, mode="actual")
    except AssertionError:
        return False
    if len(selected) < 5 or not rejected_overlap:
        return False
    size_counts: dict[int, int] = {}
    for sq in candidates:
        size_counts[sq["side"]] = size_counts.get(sq["side"], 0) + 1
    return _candidate_keys_unique(candidates) and bool(size_counts) and max(size_counts.values()) >= 2


def _break_duplicate_keys(grid: np.ndarray, active_quad: str, max_new_grey: int) -> bool:
    cr, cc = _find_cross(grid)
    start_grey = int(np.count_nonzero(grid == GREY))
    for _ in range(max_new_grey):
        candidates = _enumerate_maximal_black_squares(grid, cr, cc)
        groups: dict[tuple[int, int], list[dict[str, int]]] = {}
        for sq in candidates:
            groups.setdefault((sq["side"], sq["dist2"]), []).append(sq)
        dup_groups = [group for group in groups.values() if len(group) > 1]
        if not dup_groups:
            return True
        dup_groups.sort(key=lambda group: (-len(group), -group[0]["side"], group[0]["dist2"]))
        group = dup_groups[0]
        target = next((sq for sq in group if _square_quad(sq, cr, cc) != active_quad), group[-1])
        mid_r = target["r"] + target["side"] // 2
        mid_c = target["c"] + target["side"] // 2
        cells: list[tuple[int, int, int]] = []
        for rr in range(target["r"], target["r"] + target["side"]):
            for cc2 in range(target["c"], target["c"] + target["side"]):
                score = abs(rr - mid_r) + abs(cc2 - mid_c)
                cells.append((score, rr, cc2))
        cells.sort()
        placed = False
        for _score, rr, cc2 in cells:
            if grid[rr, cc2] != BLACK:
                continue
            trial = np.array(grid, copy=True)
            trial[rr, cc2] = GREY
            if int(np.count_nonzero(trial == GREY)) - start_grey > max_new_grey:
                continue
            try:
                _trial_out, trial_candidates, trial_selected, trial_rejected = _solve(trial, mode="actual")
            except AssertionError:
                continue
            if len(trial_selected) >= 5 and trial_rejected:
                grid[:, :] = trial
                placed = True
                break
        if not placed:
            return False
    return _candidate_keys_unique(_enumerate_maximal_black_squares(grid, cr, cc))


def _add_obstacles(grid: np.ndarray, active_quad: str) -> bool:
    cr, cc = _find_cross(grid)
    quadrants = _quadrant_bounds(grid.shape, cr, cc)
    target = random.randint(10, 14)
    if not _break_duplicate_keys(grid, active_quad, max_new_grey=8):
        return False
    if not _basic_generator_state_ok(grid):
        return False

    attempts = 0
    while int(np.count_nonzero(grid == GREY)) < target and attempts < 400:
        attempts += 1
        remaining = target - int(np.count_nonzero(grid == GREY))
        trial = np.array(grid, copy=True)
        if random.random() < 0.75:
            quad = random.choice([name for name in quadrants if name != active_quad])
        else:
            quad = random.choice(list(quadrants))
        orientation = random.choice(("h", "v"))
        length = random.randint(1, min(3, remaining))
        interior_only = random.random() < 0.35
        if not _paint_segment(trial, quadrants[quad], length, orientation, interior_only=interior_only):
            continue
        if _basic_generator_state_ok(trial):
            grid[:, :] = trial

    return int(np.count_nonzero(grid == GREY)) == target and _basic_generator_state_ok(grid)


def _semantic_sample_ok(grid: np.ndarray) -> bool:
    try:
        actual_out, candidates, selected, rejected_overlap = _solve(grid, mode="actual")
    except AssertionError:
        return False

    if not (10 <= int(np.count_nonzero(grid == GREY)) <= 20):
        return False
    if len(candidates) < 4:
        return False
    if len(selected) < 5:
        return False
    if not rejected_overlap:
        return False

    keys = set()
    for sq in candidates:
        key = (sq["side"], sq["dist2"])
        if key in keys:
            return False
        keys.add(key)

    size_counts: dict[int, int] = {}
    for sq in candidates:
        size_counts[sq["side"]] = size_counts.get(sq["side"], 0) + 1
    if max(size_counts.values()) < 2:
        return False

    no_distance_out, _no_distance_candidates, no_distance_selected, _ = _solve(grid, mode="no_distance")
    if np.array_equal(actual_out, no_distance_out):
        return False

    distance_first_out, _distance_first_candidates, distance_first_selected, _ = _solve(grid, mode="distance_first")
    if np.array_equal(actual_out, distance_first_out):
        return False

    obstacle_ignored = np.array(grid, copy=True)
    obstacle_ignored[obstacle_ignored == GREY] = BLACK
    try:
        _ignored_out, _ignored_candidates, ignored_selected, _ = _solve(obstacle_ignored, mode="actual")
    except AssertionError:
        return False
    if _selected_signature(selected) == _selected_signature(ignored_selected):
        return False

    if _selected_signature(selected) == _selected_signature(no_distance_selected):
        return False
    if _selected_signature(selected) == _selected_signature(distance_first_selected):
        return False

    changed = int(np.count_nonzero(actual_out != grid))
    if changed < 18 or changed > grid.size * 3 // 4:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid, active_quad = _build_base_grid()
        if not _add_obstacles(grid, active_quad):
            continue
        if _semantic_sample_ok(grid):
            return grid
    raise RuntimeError("failed to generate a valid sample for puzzle 2")
