from __future__ import annotations

import random
from collections import Counter
from itertools import combinations

import numpy as np


BACKGROUND = 0
OCCLUDER = 7
PALETTE = (1, 2, 3, 4, 5, 6, 8, 9)
MAX_GENERATION_ATTEMPTS = 600


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = {int(x) for x in grid.flat}
    if exclude:
        colors -= {int(x) for x in exclude}
    return sorted(colors)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _occluder_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    mask = grid == OCCLUDER
    box = _bbox(mask)
    assert box is not None, "expected an orange occluder"
    r0, c0, r1, c1 = box
    assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == OCCLUDER), "occluder must be a solid rectangle"
    assert int(mask.sum()) == (r1 - r0 + 1) * (c1 - c0 + 1), "orange must appear only in the occluder"
    return box


def _axis_score(points: np.ndarray, width: int, k: int) -> int:
    pt_set = {(int(r), int(c)) for r, c in points}
    score = 0
    for r, c in pt_set:
        cc = int(k) - int(c)
        if 0 <= cc < width and (int(r), cc) in pt_set:
            score += 1
    return int(score)


def _best_vertical_axis(points: np.ndarray, width: int) -> tuple[int, int, int]:
    assert points.size > 0
    best_k = 0
    best_score = -1
    second = -1
    for k in range(0, 2 * (width - 1) + 1):
        score = _axis_score(points, width, k)
        if score > best_score:
            second = best_score
            best_score = score
            best_k = k
        elif score > second:
            second = score
    return int(best_k), int(best_score), int(second)


def _outside_occluder(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.ones(shape, dtype=bool)
    r0, c0, r1, c1 = box
    mask[r0 : r1 + 1, c0 : c1 + 1] = False
    return mask


def _denoise_points(
    points: list[tuple[int, int]],
    k: int,
    width: int,
    occluder_mask: np.ndarray,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    point_set = {(int(r), int(c)) for r, c in points}
    kept: list[tuple[int, int]] = []
    removed: list[tuple[int, int]] = []
    for r, c in point_set:
        cc = int(k) - int(c)
        if 0 <= cc < width and bool(occluder_mask[int(r), cc]):
            kept.append((int(r), int(c)))
        elif 0 <= cc < width and (int(r), cc) in point_set:
            kept.append((int(r), int(c)))
        else:
            removed.append((int(r), int(c)))
    kept.sort()
    removed.sort()
    return kept, removed


def _recover_patch(
    kept_by_color: dict[int, list[tuple[int, int]]],
    axes: dict[int, int],
    shape: tuple[int, int],
    box: tuple[int, int, int, int],
) -> np.ndarray:
    r0, c0, r1, c1 = box
    occluder_mask = np.zeros(shape, dtype=bool)
    occluder_mask[r0 : r1 + 1, c0 : c1 + 1] = True
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for color, kept in kept_by_color.items():
        k = int(axes[color])
        for r, c in kept:
            cc = int(k) - int(c)
            if 0 <= cc < shape[1] and bool(occluder_mask[int(r), cc]):
                pr = int(r) - r0
                pc = int(cc) - c0
                previous = int(patch[pr, pc])
                assert previous in (BACKGROUND, int(color)), "recovered colors must not conflict"
                patch[pr, pc] = int(color)
    return patch


def main(grid: np.ndarray) -> np.ndarray:
    """Recover the hidden occluded patch by denoising each color with its own vertical symmetry axis.

    Input structure:
    - A mostly black grid contains exactly one solid orange rectangle, which is the occluder.
    - Three other colors form sparse pixel patterns outside the occluder, and each color is nearly vertically symmetric about its own independent axis.
    - A few same-color noise pixels break that symmetry, while the occluder hides some genuine mirrored pixels.

    Rule:
    - For each non-black, non-orange color, examine only its visible pixels outside the occluder and search all column and half-column vertical axes.
    - Choose the axis with the maximum mirror-match count, then delete only those pixels whose mirror location is outside the occluder and lacks a same-color partner.
    - Using that same per-color axis, reflect every kept pixel into the occluder whenever its mirror lands inside the orange rectangle, then merge all recovered colors.
    - Output only the occluder-sized crop, with the recovered hidden pixels on a black background.

    Key insight:
    - The symmetry axis is not global: each color must be denoised and reflected with its own best vertical axis before the hidden occluded patch can be reconstructed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    box = _occluder_bbox(g)
    occluder_mask = g == OCCLUDER
    visible_mask = ~occluder_mask
    bg = _get_bg(g)
    colors = _colors_present(g, exclude={bg, OCCLUDER})
    assert colors, "expected non-background colors"

    axes: dict[int, int] = {}
    kept_by_color: dict[int, list[tuple[int, int]]] = {}
    for color in colors:
        points = np.argwhere((g == int(color)) & visible_mask)
        assert points.size > 0, "every color must be visible outside the occluder"
        best_k, _best, _second = _best_vertical_axis(points, g.shape[1])
        axes[int(color)] = int(best_k)
        kept, _removed = _denoise_points([tuple(map(int, rc)) for rc in points], best_k, g.shape[1], occluder_mask)
        kept_by_color[int(color)] = kept

    return _recover_patch(kept_by_color, axes, g.shape, box)


def _orbit_cells(k: int, row: int, col: int) -> tuple[tuple[int, int], ...] | None:
    mirror = int(k) - int(col)
    if mirror < 0:
        return None
    if mirror == int(col):
        return ((int(row), int(col)),)
    return ((int(row), int(col)), (int(row), int(mirror)))


def _enumerate_orbits(
    shape: tuple[int, int],
    k: int,
    box: tuple[int, int, int, int],
) -> tuple[list[tuple[tuple[int, int], ...]], list[tuple[tuple[int, int], ...]], list[tuple[tuple[int, int], ...]]]:
    height, width = shape
    r0, c0, r1, c1 = box
    occluder_mask = np.zeros(shape, dtype=bool)
    occluder_mask[r0 : r1 + 1, c0 : c1 + 1] = True
    visible: list[tuple[tuple[int, int], ...]] = []
    crossing: list[tuple[tuple[int, int], ...]] = []
    hidden: list[tuple[tuple[int, int], ...]] = []
    for row in range(height):
        for col in range(width):
            mirror = int(k) - int(col)
            if not (0 <= mirror < width):
                continue
            if col > mirror:
                continue
            cells = _orbit_cells(k, row, col)
            assert cells is not None
            inside = sum(bool(occluder_mask[r, c]) for r, c in cells)
            if inside == 0:
                visible.append(cells)
            elif inside == len(cells):
                hidden.append(cells)
            else:
                crossing.append(cells)
    return visible, crossing, hidden


def _orbit_is_free(cells: tuple[tuple[int, int], ...], occupied: set[tuple[int, int]]) -> bool:
    return all(cell not in occupied for cell in cells)


def _choose_distinct_axes(shape: tuple[int, int], box: tuple[int, int, int, int]) -> list[int] | None:
    _height, width = shape
    candidate_axes = []
    for k in range(2, 2 * (width - 1) - 1):
        visible, crossing, _hidden = _enumerate_orbits(shape, k, box)
        if len(crossing) >= 2 and len(visible) >= 8:
            candidate_axes.append(k)
    if len(candidate_axes) < 3:
        return None

    mixed_parity = [combo for combo in combinations(candidate_axes, 3) if len({k % 2 for k in combo}) > 1]
    if mixed_parity and random.random() < 0.8:
        return list(random.choice(mixed_parity))
    return random.sample(candidate_axes, 3)


def _adjacent_noise_candidates(
    clean_cells: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    k: int,
    shape: tuple[int, int],
    occluder_mask: np.ndarray,
) -> list[tuple[int, int]]:
    height, width = shape
    candidates = set()
    for r, c in clean_cells:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr = int(r) + dr
            nc = int(c) + dc
            if not (0 <= nr < height and 0 <= nc < width):
                continue
            if bool(occluder_mask[nr, nc]) or (nr, nc) in occupied:
                continue
            mirror = int(k) - int(nc)
            if not (0 <= mirror < width):
                continue
            if bool(occluder_mask[nr, mirror]) or (nr, mirror) in occupied:
                continue
            if (nr, mirror) == (nr, nc):
                continue
            candidates.add((nr, nc))
    return sorted(candidates)


def _pick_orbits_for_color(
    shape: tuple[int, int],
    box: tuple[int, int, int, int],
    k: int,
    occupied: set[tuple[int, int]],
    color_index: int,
) -> set[tuple[int, int]] | None:
    height, width = shape
    occluder_mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = box
    occluder_mask[r0 : r1 + 1, c0 : c1 + 1] = True
    visible_orbits, crossing_orbits, _hidden = _enumerate_orbits(shape, k, box)

    crossing_pool = [cells for cells in crossing_orbits if len(cells) == 2 and _orbit_is_free(cells, occupied)]
    visible_pool = [cells for cells in visible_orbits if _orbit_is_free(cells, occupied)]

    if len(crossing_pool) < 2 or len(visible_pool) < 4:
        return None

    crossing_rows = sorted({next(r for r, c in cells if occluder_mask[r, c]) for cells in crossing_pool})
    preferred_hidden_row = crossing_rows[color_index % len(crossing_rows)]
    first_cross_choices = [cells for cells in crossing_pool if any(r == preferred_hidden_row and occluder_mask[r, c] for r, c in cells)]
    if not first_cross_choices:
        first_cross_choices = crossing_pool[:]
    random.shuffle(first_cross_choices)
    first_cross = first_cross_choices[0]

    remaining_cross = [cells for cells in crossing_pool if cells != first_cross]
    random.shuffle(remaining_cross)
    chosen = [first_cross, remaining_cross[0]]

    visible_rows = [
        row
        for row in range(height)
        if row < r0 - 1 or row > r1 + 1
    ]
    visible_pref = [cells for cells in visible_pool if cells[0][0] in visible_rows]
    if len(visible_pref) < 4:
        visible_pref = visible_pool[:]
    random.shuffle(visible_pref)
    needed_visible = random.randint(4, 5)
    seen_rows = {cells[0][0] for cells in chosen}
    for cells in visible_pref:
        if len([x for x in chosen if x[0][0] == cells[0][0]]) >= 1:
            continue
        chosen.append(cells)
        seen_rows.add(cells[0][0])
        if len(chosen) >= 2 + needed_visible:
            break
    if len(chosen) < 2 + needed_visible:
        return None

    if k % 2 == 0 and random.random() < 0.45:
        center_col = k // 2
        center_rows = [row for row in visible_rows if (row, center_col) not in occupied and not occluder_mask[row, center_col]]
        random.shuffle(center_rows)
        if center_rows:
            chosen.append(((center_rows[0], center_col),))

    clean_cells = {cell for orbit in chosen for cell in orbit}
    if any(cell in occupied for cell in clean_cells):
        return None

    noise_candidates = _adjacent_noise_candidates(clean_cells, occupied | clean_cells, k, shape, occluder_mask)
    if len(noise_candidates) < 2:
        return None

    noisy_cells = set(clean_cells)
    noise_count = random.randint(2, min(3, len(noise_candidates)))
    for cell in random.sample(noise_candidates, noise_count):
        noisy_cells.add(cell)
    return noisy_cells


def _solve_color_layers(
    grid: np.ndarray,
) -> tuple[dict[int, int], dict[int, list[tuple[int, int]]], dict[int, list[tuple[int, int]]]]:
    box = _occluder_bbox(grid)
    occluder_mask = grid == OCCLUDER
    colors = _colors_present(grid, exclude={BACKGROUND, OCCLUDER})
    axes: dict[int, int] = {}
    kept: dict[int, list[tuple[int, int]]] = {}
    removed: dict[int, list[tuple[int, int]]] = {}
    for color in colors:
        points = [tuple(map(int, rc)) for rc in np.argwhere((grid == int(color)) & ~occluder_mask)]
        best_k, _best, _second = _best_vertical_axis(np.array(points, dtype=int), grid.shape[1])
        kept_points, removed_points = _denoise_points(points, best_k, grid.shape[1], occluder_mask)
        axes[int(color)] = int(best_k)
        kept[int(color)] = kept_points
        removed[int(color)] = removed_points
    return axes, kept, removed


def _recover_with_axes(grid: np.ndarray, axes: dict[int, int], use_denoise: bool) -> np.ndarray:
    box = _occluder_bbox(grid)
    occluder_mask = grid == OCCLUDER
    patch = np.zeros((box[2] - box[0] + 1, box[3] - box[1] + 1), dtype=int)
    for color in _colors_present(grid, exclude={BACKGROUND, OCCLUDER}):
        points = [tuple(map(int, rc)) for rc in np.argwhere((grid == int(color)) & ~occluder_mask)]
        if use_denoise:
            points, _removed = _denoise_points(points, axes[int(color)], grid.shape[1], occluder_mask)
        for r, c in points:
            cc = int(axes[int(color)]) - int(c)
            if 0 <= cc < grid.shape[1] and bool(occluder_mask[int(r), cc]):
                pr = int(r) - box[0]
                pc = int(cc) - box[1]
                if patch[pr, pc] not in (BACKGROUND, int(color)):
                    return np.full_like(patch, fill_value=9)
                patch[pr, pc] = int(color)
    return patch


def _validate_generated(grid: np.ndarray, clean_scene: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    try:
        box = _occluder_bbox(grid)
    except AssertionError:
        return False

    colors = _colors_present(grid, exclude={BACKGROUND, OCCLUDER})
    if len(colors) != 3:
        return False

    expected = clean_scene[box[0] : box[2] + 1, box[1] : box[3] + 1].copy()
    if np.array_equal(expected, np.zeros_like(expected)):
        return False
    try:
        produced = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(produced, expected):
        return False

    axes, kept, removed = _solve_color_layers(grid)
    if len(set(axes.values())) != 3:
        return False
    occluder_mask = grid == OCCLUDER

    for color in colors:
        visible_points = np.argwhere((grid == int(color)) & ~occluder_mask)
        best_k, best_score, second_score = _best_vertical_axis(visible_points, grid.shape[1])
        if best_k != axes[int(color)] or best_score <= second_score:
            return False
        if len(removed[int(color)]) < 2:
            return False
        recovered_cells = 0
        for r, c in kept[int(color)]:
            cc = axes[int(color)] - c
            if 0 <= cc < grid.shape[1] and bool(occluder_mask[r, cc]):
                recovered_cells += 1
        if recovered_cells == 0:
            return False

    for common_axis in range(0, 2 * (grid.shape[1] - 1) + 1):
        common = {color: common_axis for color in colors}
        if np.array_equal(_recover_with_axes(grid, common, use_denoise=True), expected):
            return False
    return True


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(14, 19)
    width = random.randint(14, 19)
    oc_h = random.randint(4, 6)
    oc_w = random.randint(4, 6)
    r0 = random.randint(2, height - oc_h - 2)
    c0 = random.randint(2, width - oc_w - 2)
    box = (r0, c0, r0 + oc_h - 1, c0 + oc_w - 1)
    axes = _choose_distinct_axes((height, width), box)
    if axes is None:
        return None

    colors = random.sample(PALETTE, 3)
    clean_scene = np.zeros((height, width), dtype=int)
    occupied: set[tuple[int, int]] = set()
    noisy_layers: dict[int, set[tuple[int, int]]] = {}

    axis_order = list(zip(colors, axes))
    random.shuffle(axis_order)
    for color_index, (color, k) in enumerate(axis_order):
        cells = _pick_orbits_for_color((height, width), box, k, occupied, color_index)
        if cells is None:
            return None
        noisy_layers[int(color)] = cells
        clean_points, _removed = _denoise_points(list(cells), k, width, np.zeros((height, width), dtype=bool))
        clean_points_set = set(clean_points)
        if len(clean_points_set) >= len(cells):
            return None
        for r, c in clean_points_set:
            clean_scene[r, c] = int(color)
        occupied |= clean_points_set

    expected = clean_scene[box[0] : box[2] + 1, box[1] : box[3] + 1].copy()
    if any(np.count_nonzero(expected == color) == 0 for color in colors):
        return None

    inp = clean_scene.copy()
    for color, cells in noisy_layers.items():
        inp[list(zip(*cells))[0], list(zip(*cells))[1]] = int(color)
    inp[box[0] : box[2] + 1, box[1] : box[3] + 1] = OCCLUDER

    if not _validate_generated(inp, clean_scene):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid per-color occluder example")
