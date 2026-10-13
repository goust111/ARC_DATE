from __future__ import annotations

import random
from collections import Counter

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _evaluate_axis(grid: np.ndarray, axis: int, bg: int) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)
    axis = int(axis)

    if not (0 <= axis < width - 1):
        return {"status": "not_candidate", "axis": axis}
    if int(width - axis - 1) > axis:
        return {"status": "not_candidate", "axis": axis}
    if np.any(g[:, axis] != bg):
        return {"status": "not_candidate", "axis": axis}

    right_width = width - axis - 1
    if right_width <= 0:
        return {"status": "not_candidate", "axis": axis}

    left = g[:, :axis]
    right = np.fliplr(g[:, axis + 1 :])
    start = axis - right_width
    covered_left = left[:, start:axis]
    both = (covered_left != bg) & (right != bg)
    conflict = both & (covered_left != right)
    if np.any(conflict):
        return {
            "status": "conflict",
            "axis": axis,
            "overlap": int(np.count_nonzero(both)),
        }

    overlap = int(np.count_nonzero(both))
    if overlap == 0:
        return {"status": "no_overlap", "axis": axis, "overlap": 0}

    out = left.copy()
    window = out[:, start:axis]
    right_mask = right != bg
    window[right_mask] = right[right_mask]
    return {
        "status": "valid",
        "axis": axis,
        "overlap": overlap,
        "out": out,
    }


def _candidate_axes(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    return [info for c in range(g.shape[1]) if (info := _evaluate_axis(g, c, bg))["status"] != "not_candidate"]


def main(grid: np.ndarray) -> np.ndarray:
    """Find the unmarked vertical fold axis whose mirrored right side agrees with the left side, then output the fold result.

    Input structure:
    - The grid contains only background plus sparse multi-color foreground pixels, with no separator line drawn.
    - Candidate fold axes are full-background columns far enough to the right that their whole right side would fit inside the left side after mirroring.
    - Exactly one such column lets the mirrored right side overlay the left side with no color conflicts and with at least one true foreground overlap.

    Rule:
    - Test every candidate blank column.
    - Keep the unique column whose fold has no conflicting overlap and has at least one overlapping foreground pair.
    - Remove that axis column and output its left side, overlaying the mirrored right side onto the matching rightmost part of the left side.

    Key insight:
    - The fold line is invisible; it must be inferred from self-consistency plus nontrivial overlap, not from any drawn divider.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "input must be a non-empty 2D grid"
    bg = _get_bg(g)

    candidates = _candidate_axes(g, bg)
    valid = [info for info in candidates if info["status"] == "valid"]
    assert len(valid) == 1, "expected exactly one self-consistent fold axis"
    return np.asarray(valid[0]["out"], dtype=int)


BG = 0
PALETTE = tuple(range(1, 10))
HEIGHT_RANGE = (7, 12)
AXIS_RANGE = (6, 12)
RIGHT_WIDTH_MIN = 4
MAX_GENERATION_ATTEMPTS = 500


def _rand_color(rng: random.Random, palette: tuple[int, ...]) -> int:
    return int(rng.choice(palette))


def _rand_distinct_colors(rng: random.Random, palette: tuple[int, ...], count: int) -> tuple[int, ...]:
    return tuple(int(c) for c in rng.sample(list(palette), count))


def _choose_geometry(rng: random.Random) -> tuple[int, int, int, int, int, str]:
    for _ in range(200):
        height = rng.randint(*HEIGHT_RANGE)
        centered = rng.random() < 0.35
        axis = rng.randint(*AXIS_RANGE)
        if centered:
            right_width = axis
        else:
            right_width = rng.randint(RIGHT_WIDTH_MIN, axis - 1)
        width = axis + 1 + right_width
        if width > 30:
            continue

        align_start = axis - right_width
        no_overlap_axis = width - 2

        left_conf = [d for d in range(width // 2, axis - 1) if d - 1 >= 0 and d + 1 < axis]
        right_conf = [d for d in range(axis + 2, width - 3) if d + 1 < width - 1]
        if left_conf and (not centered or rng.random() < 0.7):
            conflict_axis = int(rng.choice(left_conf))
            conflict_side = "left"
        elif right_conf:
            conflict_axis = int(rng.choice(right_conf))
            conflict_side = "right"
        else:
            continue

        if conflict_axis == no_overlap_axis:
            continue
        return height, axis, right_width, width, align_start, conflict_side
    raise RuntimeError("failed to choose workable geometry")


def _right_abs_to_index(width: int, abs_col: int) -> int:
    return int(width - 1 - abs_col)


def _right_abs_to_output(axis: int, abs_col: int) -> int:
    return int(2 * axis - abs_col)


def _safe_rows_for_left(
    left: np.ndarray,
    right: np.ndarray,
    left_blocked: np.ndarray,
    col: int,
    align_start: int,
) -> list[int]:
    rows = []
    for r in range(left.shape[0]):
        if left_blocked[r, col] or left[r, col] != BG:
            continue
        if col >= align_start and right[r, col - align_start] != BG:
            continue
        rows.append(r)
    return rows


def _safe_rows_for_right(
    left: np.ndarray,
    right: np.ndarray,
    left_blocked: np.ndarray,
    right_blocked: np.ndarray,
    out_col: int,
    align_start: int,
) -> list[int]:
    idx = out_col - align_start
    rows = []
    for r in range(left.shape[0]):
        if right_blocked[r, idx] or right[r, idx] != BG:
            continue
        if left[r, out_col] != BG:
            continue
        rows.append(r)
    return rows


def _available_both_positions(
    left: np.ndarray,
    right: np.ndarray,
    left_blocked: np.ndarray,
    right_blocked: np.ndarray,
    axis: int,
    align_start: int,
    blank_cols: set[int],
) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    for out_col in range(align_start, axis):
        right_abs = 2 * axis - out_col
        if out_col in blank_cols or right_abs in blank_cols:
            continue
        idx = out_col - align_start
        for r in range(left.shape[0]):
            if left_blocked[r, out_col] or right_blocked[r, idx]:
                continue
            if left[r, out_col] == BG and right[r, idx] == BG:
                positions.append((r, out_col))
    return positions


def _set_left(left: np.ndarray, r: int, c: int, color: int) -> None:
    left[int(r), int(c)] = int(color)


def _set_right(right: np.ndarray, r: int, out_col: int, align_start: int, color: int) -> None:
    right[int(r), int(out_col - align_start)] = int(color)


def _render_grid(left: np.ndarray, right: np.ndarray, axis: int) -> np.ndarray:
    right_input = np.fliplr(right)
    grid = np.full((left.shape[0], axis + 1 + right.shape[1]), BG, dtype=int)
    grid[:, :axis] = left
    grid[:, axis + 1 :] = right_input
    return grid


def _semantic_counts(left: np.ndarray, right: np.ndarray, align_start: int) -> tuple[int, int, int, int]:
    both = int(np.count_nonzero((left[:, align_start:] != BG) & (right != BG)))
    right_only = int(np.count_nonzero((left[:, align_start:] == BG) & (right != BG)))
    left_only = int(np.count_nonzero(left[:, :align_start] != BG) + np.count_nonzero((left[:, align_start:] != BG) & (right == BG)))
    prefix_fg = int(np.count_nonzero(left[:, :align_start] != BG))
    return both, left_only, right_only, prefix_fg


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height, axis, right_width, width, align_start, conflict_side = _choose_geometry(rng)
    no_overlap_axis = width - 2
    left_conf = [d for d in range(width // 2, axis - 1) if d - 1 >= 0 and d + 1 < axis]
    right_conf = [d for d in range(axis + 2, width - 3) if d + 1 < width - 1]
    if conflict_side == "left":
        conflict_axis = int(rng.choice(left_conf))
    else:
        conflict_axis = int(rng.choice(right_conf))

    blank_cols = {axis, conflict_axis, no_overlap_axis}
    palette = tuple(int(c) for c in rng.sample(PALETTE, rng.randint(3, 4)))

    left = np.full((height, axis), BG, dtype=int)
    right = np.full((height, right_width), BG, dtype=int)
    left_blocked = np.zeros_like(left, dtype=bool)
    right_blocked = np.zeros_like(right, dtype=bool)

    no_rows = rng.sample(range(height), k=rng.randint(1, min(2, height)))
    for r in no_rows:
        left_blocked[r, align_start] = True
        right_blocked[r, 2] = True
        right[r, 0] = _rand_color(rng, palette)

    conflict_rows = [r for r in range(height) if r not in no_rows]
    if not conflict_rows:
        conflict_rows = list(range(height))
    r_conf = int(rng.choice(conflict_rows))

    if conflict_side == "left":
        c1, c2 = _rand_distinct_colors(rng, palette, 2)
        _set_left(left, r_conf, conflict_axis - 1, c1)
        _set_left(left, r_conf, conflict_axis + 1, c2)
    else:
        c1, c2 = _rand_distinct_colors(rng, palette, 2)
        x_left = conflict_axis - 1
        x_right = conflict_axis + 1
        j_left = _right_abs_to_output(axis, x_left)
        j_right = _right_abs_to_output(axis, x_right)
        left_blocked[r_conf, j_left] = True
        left_blocked[r_conf, j_right] = True
        _set_right(right, r_conf, j_left, align_start, c1)
        _set_right(right, r_conf, j_right, align_start, c2)

    both_needed = rng.randint(2, 5)
    both_positions = _available_both_positions(left, right, left_blocked, right_blocked, axis, align_start, blank_cols)
    if len(both_positions) < both_needed:
        return None
    rng.shuffle(both_positions)
    for idx, (r, c) in enumerate(both_positions[:both_needed]):
        color = palette[idx % len(palette)]
        _set_left(left, r, c, color)
        _set_right(right, r, c, align_start, color)

    if align_start > 0:
        prefix_candidates = [(r, c) for c in range(align_start) if c not in blank_cols for r in range(height) if left[r, c] == BG and not left_blocked[r, c]]
        if not prefix_candidates:
            return None
        rng.shuffle(prefix_candidates)
        for r, c in prefix_candidates[: rng.randint(1, min(4, len(prefix_candidates)))]:
            _set_left(left, r, c, _rand_color(rng, palette))

    covered_left_candidates = [
        (r, c)
        for c in range(align_start, axis)
        if c not in blank_cols
        for r in range(height)
        if left[r, c] == BG and not left_blocked[r, c] and right[r, c - align_start] == BG
    ]
    if not covered_left_candidates:
        return None
    rng.shuffle(covered_left_candidates)
    for r, c in covered_left_candidates[: rng.randint(1, min(4, len(covered_left_candidates)))]:
        _set_left(left, r, c, _rand_color(rng, palette))

    right_only_candidates = [
        (r, c)
        for c in range(align_start, axis)
        if (2 * axis - c) not in blank_cols
        for r in range(height)
        if right[r, c - align_start] == BG and not right_blocked[r, c - align_start] and left[r, c] == BG
    ]
    if len(right_only_candidates) < 2:
        return None
    rng.shuffle(right_only_candidates)
    for r, c in right_only_candidates[: rng.randint(2, min(5, len(right_only_candidates)))]:
        _set_right(right, r, c, align_start, _rand_color(rng, palette))

    for abs_col in range(width // 2, width - 1):
        if abs_col in blank_cols:
            continue
        if abs_col < axis:
            if np.all(left[:, abs_col] == BG):
                rows = _safe_rows_for_left(left, right, left_blocked, abs_col, align_start)
                if not rows:
                    return None
                _set_left(left, rng.choice(rows), abs_col, _rand_color(rng, palette))
        elif abs_col > axis:
            out_col = _right_abs_to_output(axis, abs_col)
            if np.all(right[:, out_col - align_start] == BG):
                rows = _safe_rows_for_right(left, right, left_blocked, right_blocked, out_col, align_start)
                if not rows:
                    return None
                _set_right(right, rng.choice(rows), out_col, align_start, _rand_color(rng, palette))

    extra_steps = rng.randint(0, 5)
    for _ in range(extra_steps):
        if rng.random() < 0.5:
            choices = [
                (r, c)
                for c in range(axis)
                if c not in blank_cols
                for r in range(height)
                if left[r, c] == BG and not left_blocked[r, c] and (c < align_start or right[r, c - align_start] == BG)
            ]
            if choices:
                r, c = rng.choice(choices)
                _set_left(left, r, c, _rand_color(rng, palette))
        else:
            choices = [
                (r, c)
                for c in range(align_start, axis)
                if (2 * axis - c) not in blank_cols
                for r in range(height)
                if right[r, c - align_start] == BG and not right_blocked[r, c - align_start] and left[r, c] == BG
            ]
            if choices:
                r, c = rng.choice(choices)
                _set_right(right, r, c, align_start, _rand_color(rng, palette))

    grid = _render_grid(left, right, axis)
    bg = _get_bg(grid)
    statuses = {int(info["axis"]): str(info["status"]) for info in _candidate_axes(grid, bg)}
    if statuses.get(axis) != "valid":
        return None
    if statuses.get(conflict_axis) != "conflict":
        return None
    if statuses.get(no_overlap_axis) != "no_overlap":
        return None
    if set(statuses) != blank_cols:
        return None

    valid = [info for info in _candidate_axes(grid, bg) if info["status"] == "valid"]
    if len(valid) != 1 or int(valid[0]["axis"]) != axis:
        return None

    both, left_only, right_only, prefix_fg = _semantic_counts(left, right, align_start)
    if both < 2 or left_only == 0 or right_only == 0:
        return None
    if align_start > 0 and prefix_fg == 0:
        return None

    out = main(grid)
    if out.shape != (height, axis):
        return None
    if len(set(map(int, np.unique(grid))) - {BG}) < 3:
        return None
    if np.count_nonzero(out != BG) < 8:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid invisible-fold input")
