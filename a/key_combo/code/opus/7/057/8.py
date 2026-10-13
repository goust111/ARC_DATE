from __future__ import annotations

import random

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))

GEN_HEIGHT_RANGE = (14, 18)
GEN_WIDTH_RANGE = (14, 18)
GEN_RETRY_LIMIT = 400
SHAPE_RETRY_LIMIT = 300


def _axis_scores(coords: np.ndarray, width: int) -> dict[int, int]:
    pts = {tuple(map(int, rc)) for rc in coords}
    scores: dict[int, int] = {}
    for k in range(2 * (int(width) - 1) + 1):
        score = 0
        for r, c in pts:
            if (r, k - c) in pts:
                score += 1
        scores[int(k)] = int(score)
    return scores


def _best_axis(coords: np.ndarray, width: int) -> tuple[int, dict[int, int]]:
    scores = _axis_scores(coords, width)
    best_score = max(scores.values())
    best_axes = [k for k, score in scores.items() if score == best_score]
    return int(min(best_axes)), scores


def _noise_pixels(grid: np.ndarray, color: int, axis_k: int) -> list[tuple[int, int]]:
    coords = np.argwhere(grid == int(color))
    noises: list[tuple[int, int]] = []
    width = int(grid.shape[1])
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        mirror_c = int(axis_k) - cc
        if not (0 <= mirror_c < width and int(grid[rr, mirror_c]) == int(color)):
            noises.append((rr, cc))
    return sorted(noises)


def _analyze_colors(grid: np.ndarray) -> tuple[dict[int, int], list[tuple[int, int, int]]]:
    axes: dict[int, int] = {}
    noises: list[tuple[int, int, int]] = []
    width = int(grid.shape[1])
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != BG):
        coords = np.argwhere(grid == int(color))
        axis_k, _scores = _best_axis(coords, width)
        axes[int(color)] = int(axis_k)
        for r, c in _noise_pixels(grid, int(color), axis_k):
            noises.append((int(r), int(c), int(color)))
    return axes, sorted(noises)


def _wedge_cells(shape: tuple[int, int], r: int, c: int) -> set[tuple[int, int]]:
    h, w = map(int, shape)
    max_depth = min(h - 1 - int(r), int(c), w - 1 - int(c))
    cells: set[tuple[int, int]] = set()
    for depth in range(max_depth + 1):
        rr = int(r) + depth
        for cc in range(int(c) - depth, int(c) + depth + 1):
            cells.add((rr, cc))
    return cells


def _apply_wedges(
    clean_grid: np.ndarray,
    noises: list[tuple[int, int, int]],
) -> np.ndarray:
    out = np.array(clean_grid, dtype=int, copy=True)
    for r, c, color in sorted(noises):
        max_depth = min(out.shape[0] - 1 - int(r), int(c), out.shape[1] - 1 - int(c))
        for depth in range(max_depth + 1):
            rr = int(r) + depth
            for cc in range(int(c) - depth, int(c) + depth + 1):
                if int(out[rr, cc]) == BG:
                    out[rr, cc] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect asymmetric noise pixels in each color layer and turn them into downward wedges.

    Input structure:
    - Black background with two non-background colors.
    - Each color forms a nearly vertically symmetric sprite, but each color uses its own vertical symmetry axis.
    - One or two pixels in each color are unmatched noise pixels away from that color's axis.

    Rule:
    - For each non-background color independently, search all integer and half-integer vertical axes and pick the axis with the most same-color mirror matches.
    - Mark every pixel of that color whose reflected position across the chosen axis is not occupied by the same color.
    - Erase those unmatched pixels, then for each erased position grow a downward centered wedge of the same color, stopping when the wedge would leave the grid horizontally or vertically.
    - Wedge pixels only fill black cells; existing non-black pixels remain unchanged.

    Key insight:
    - Vertical symmetry is used as an anomaly detector: the unmatched pixels are not just removed, they become seed points for new wedge-shaped growth.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _axes, noises = _analyze_colors(g)
    clean = g.copy()
    for r, c, _color in noises:
        clean[int(r), int(c)] = BG
    return _apply_wedges(clean, noises)


def _row_interval(width: int, row_width: int, integer_axis: bool) -> tuple[int, int]:
    half = int(width) // 2
    if integer_axis:
        span = (int(row_width) - 1) // 2
        return int(half - span), int(half + span)
    span = int(row_width) // 2
    return int(half - span), int(half + span - 1)


def _build_centered_patch(
    width: int,
    profile: list[int],
    integer_axis: bool,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    patch = np.zeros((len(profile), int(width)), dtype=int)
    intervals: list[tuple[int, int]] = []
    for r, row_width in enumerate(profile):
        left, right = _row_interval(width, row_width, integer_axis)
        patch[int(r), int(left) : int(right) + 1] = 1
        intervals.append((int(left), int(right)))
    return patch, intervals


def _make_object_mask(prefer_side: str) -> tuple[np.ndarray, list[tuple[int, int]], int] | None:
    assert prefer_side in {"left", "right"}
    for _ in range(SHAPE_RETRY_LIMIT):
        integer_axis = random.random() < 0.5
        half_width = random.randint(2, 3)
        width = 2 * half_width + 1 if integer_axis else 2 * half_width
        height = random.randint(4, 6)
        choices = [v for v in range(1 if integer_axis else 2, width + 1, 2)]
        profile = [random.choice(choices) for _ in range(height)]
        clean_count = sum(profile)
        if not (6 <= clean_count <= 12):
            continue
        if len(set(profile)) < 2 or max(profile) == width:
            continue

        patch, intervals = _build_centered_patch(width, profile, integer_axis)
        coords = np.argwhere(patch == 1)
        axis_k, scores = _best_axis(coords, width)
        if axis_k != width - 1:
            continue
        ordered_scores = sorted(scores.values(), reverse=True)
        second_best = ordered_scores[1] if len(ordered_scores) > 1 else -1
        if list(scores.values()).count(scores[axis_k]) != 1 or scores[axis_k] - second_best < 2:
            continue

        row_limit = max(1, (height + 1) // 2)
        candidates: list[tuple[int, int]] = []
        for r in range(row_limit):
            left, right = intervals[r]
            candidate_c = left - 1 if prefer_side == "left" else right + 1
            if not (0 <= candidate_c < width):
                continue
            mirror_c = axis_k - candidate_c
            if patch[r, candidate_c] != 0:
                continue
            if 0 <= mirror_c < width and patch[r, mirror_c] != 0:
                continue
            candidates.append((int(r), int(candidate_c)))

        if not candidates:
            continue
        noise_cap = min(2, len(candidates), 15 - clean_count)
        if clean_count + 1 < 8 or noise_cap < 1:
            continue
        noise_count = 2 if noise_cap >= 2 and random.random() < 0.5 else 1
        rows = random.sample(range(len(candidates)), noise_count)
        noise_pixels = sorted(candidates[i] for i in rows)

        noisy = patch.copy()
        for r, c in noise_pixels:
            noisy[r, c] = 1

        noisy_coords = np.argwhere(noisy == 1)
        noisy_axis, noisy_scores = _best_axis(noisy_coords, width)
        ordered_noisy = sorted(noisy_scores.values(), reverse=True)
        noisy_second = ordered_noisy[1] if len(ordered_noisy) > 1 else -1
        if noisy_axis != axis_k:
            continue
        if list(noisy_scores.values()).count(noisy_scores[noisy_axis]) != 1:
            continue
        if noisy_scores[noisy_axis] - noisy_second < 1:
            continue
        if _noise_pixels(noisy, 1, noisy_axis) != noise_pixels:
            continue

        return noisy, noise_pixels, axis_k
    return None


def _paint_mask(canvas: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    region = canvas[int(top) : int(top) + h, int(left) : int(left) + w]
    assert not np.any((mask != 0) & (region != BG)), "mask overlap"
    region[mask != 0] = int(color)


def _make_attempt() -> np.ndarray | None:
    height = random.randint(*GEN_HEIGHT_RANGE)
    width = random.randint(*GEN_WIDTH_RANGE)
    grid = np.zeros((height, width), dtype=int)
    clean = np.zeros((height, width), dtype=int)

    colors = random.sample(COLORS, 2)
    left_data = _make_object_mask("left")
    right_data = _make_object_mask("right")
    if left_data is None or right_data is None:
        return None
    left_mask, left_noises, left_axis = left_data
    right_mask, right_noises, right_axis = right_data

    left_h, left_w = left_mask.shape
    right_h, right_w = right_mask.shape

    left_min_noise_c = min(c for _r, c in left_noises)
    right_max_noise_c = max(c for _r, c in right_noises)
    left_max_noise_r = max(r for r, _c in left_noises)
    right_max_noise_r = max(r for r, _c in right_noises)

    upper_limit = min(height // 2, height - 5)
    left_top_max = upper_limit - left_max_noise_r
    right_top_max = upper_limit - right_max_noise_r
    if left_top_max < 1 or right_top_max < 1:
        return None
    left_top = random.randint(1, left_top_max)
    right_top = random.randint(1, right_top_max)

    left_target_noise_col = random.randint(3, 4)
    left_left = left_target_noise_col - left_min_noise_c
    if not (0 <= left_left and left_left + left_w <= width):
        return None

    right_target_noise_col = width - 1 - random.randint(3, 4)
    right_left = right_target_noise_col - right_max_noise_c
    if not (0 <= right_left and right_left + right_w <= width):
        return None

    if left_left + left_w + 2 > right_left:
        return None

    left_global_axis = 2 * left_left + left_axis
    right_global_axis = 2 * right_left + right_axis
    if left_global_axis == right_global_axis:
        return None

    left_clean = left_mask.copy()
    for r, c in left_noises:
        left_clean[r, c] = BG
    right_clean = right_mask.copy()
    for r, c in right_noises:
        right_clean[r, c] = BG

    _paint_mask(grid, left_mask, left_top, left_left, colors[0])
    _paint_mask(grid, right_mask, right_top, right_left, colors[1])
    _paint_mask(clean, left_clean, left_top, left_left, colors[0])
    _paint_mask(clean, right_clean, right_top, right_left, colors[1])

    expected_noises = [
        (left_top + r, left_left + c, colors[0]) for r, c in left_noises
    ] + [
        (right_top + r, right_left + c, colors[1]) for r, c in right_noises
    ]
    expected_noises = sorted((int(r), int(c), int(color)) for r, c, color in expected_noises)

    for r, c, _color in expected_noises:
        if not (r <= height // 2 and height - 1 - r >= 4):
            return None
        if min(c, width - 1 - c) < 3:
            return None

    actual_axes, actual_noises = _analyze_colors(grid)
    if sorted(actual_axes.values()) == [left_global_axis, right_global_axis]:
        pass
    elif actual_axes.get(colors[0]) != left_global_axis or actual_axes.get(colors[1]) != right_global_axis:
        return None
    if actual_noises != expected_noises:
        return None

    per_color_counts = {color: int(np.count_nonzero(grid == color)) for color in colors}
    if any(count < 8 or count > 15 for count in per_color_counts.values()):
        return None

    painted_by_wedge: list[set[tuple[int, int]]] = []
    for r, c, color in expected_noises:
        wedge = _wedge_cells(grid.shape, r, c)
        new_cells = {cell for cell in wedge if int(clean[cell]) == BG}
        if len(new_cells - {(r, c)}) < 2:
            return None
        for other in painted_by_wedge:
            if new_cells & other:
                return None
        painted_by_wedge.append(new_cells)

    expected = _apply_wedges(clean, expected_noises)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(actual, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _make_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
