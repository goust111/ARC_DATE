from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
SEPARATOR = 5
SPRITE_BANK = [
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0], [1, 1, 1]], dtype=int),
    np.array([[1, 1, 1], [1, 1, 1], [1, 0, 1]], dtype=int),
]
GEN_RETRIES = 400


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _find_separator_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == SEPARATOR)]
    assert len(rows) == 1, "expected exactly one separator row"
    return int(rows[0])


def _parse_legend(grid: np.ndarray, sep_row: int) -> tuple[dict[int, np.ndarray], set[int]]:
    legend = grid[:sep_row]
    non_bg_rows = [r for r in range(legend.shape[0]) if np.any(legend[r] != BG)]
    assert non_bg_rows, "legend cannot be empty"
    marker_row = max(non_bg_rows)
    marker_cols = sorted(int(c) for c in np.flatnonzero(legend[marker_row] != BG))
    assert 2 <= len(marker_cols) <= 3, "expected 2-3 legend markers"

    mapping: dict[int, np.ndarray] = {}
    sprite_colors = set()
    for c in marker_cols:
        assert 1 <= c < legend.shape[1] - 1
        assert marker_row >= 3
        marker_color = int(legend[marker_row, c])
        assert marker_color != SEPARATOR
        sprite = legend[marker_row - 3 : marker_row, c - 1 : c + 2]
        assert sprite.shape == (3, 3)
        fg_colors = sorted(set(int(v) for v in sprite.flatten()) - {BG})
        assert len(fg_colors) == 1, "each legend sprite must be monochrome"
        sprite_color = fg_colors[0]
        sprite_colors.add(sprite_color)
        sprite_mask = (sprite == sprite_color).astype(int)
        assert sprite_mask[1, 1] == 1, "legend sprite center must be filled"
        assert np.array_equal(sprite_mask, sprite_mask[:, ::-1]), "legend sprite must be left-right symmetric"
        mapping[marker_color] = sprite_mask * int(sprite_color)

    assert len(mapping) == len(marker_cols), "legend marker colors must be distinct"
    assert len(sprite_colors) == len(marker_cols), "legend sprite colors must be distinct"
    return mapping, set(mapping)


def _iter_explicit_centers(work: np.ndarray, marker_colors: Iterable[int]) -> list[tuple[int, int, int]]:
    out: list[tuple[int, int, int]] = []
    for marker_color in sorted(int(c) for c in marker_colors):
        for r, c in np.argwhere(work == int(marker_color)):
            out.append((int(r), int(c), int(marker_color)))
    out.sort()
    return out


def _axis_score(work: np.ndarray, marker_colors: set[int], axis2: int) -> tuple[int, int]:
    h, w = work.shape
    matches = 0
    mismatches = 0
    for r in range(h):
        for c in range(w):
            mc = axis2 - c
            if mc < c:
                continue
            left_is_solid = int(work[r, c]) not in marker_colors and int(work[r, c]) != BG
            if mc < 0 or mc >= w:
                if left_is_solid:
                    mismatches += 1
                continue
            right_is_solid = int(work[r, mc]) not in marker_colors and int(work[r, mc]) != BG
            if c == mc:
                if left_is_solid:
                    matches += 1
                continue
            if left_is_solid or right_is_solid:
                if left_is_solid and right_is_solid and int(work[r, c]) == int(work[r, mc]):
                    matches += 2
                else:
                    mismatches += 2
    return matches, -mismatches


def _infer_axis2(work: np.ndarray, marker_colors: set[int]) -> tuple[int, dict[int, tuple[int, int]]]:
    scores = {axis2: _axis_score(work, marker_colors, axis2) for axis2 in range(2 * work.shape[1] - 1)}
    best_score = max(scores.values())
    best_axes = [axis2 for axis2, score in scores.items() if score == best_score]
    assert len(best_axes) == 1, "symmetry axis must be unique"
    return best_axes[0], scores


def _stamp(
    out: np.ndarray,
    sprite: np.ndarray,
    center_r: int,
    center_c: int,
    blocked: np.ndarray,
) -> None:
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            if int(sprite[dr + 1, dc + 1]) == BG:
                continue
            rr = int(center_r + dr)
            cc = int(center_c + dc)
            assert 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]
            if not bool(blocked[rr, cc]):
                out[rr, cc] = int(sprite[dr + 1, dc + 1])


def _solve_from_parsed(
    work: np.ndarray,
    legend_map: dict[int, np.ndarray],
    *,
    use_symmetry: bool = True,
) -> np.ndarray:
    marker_colors = set(legend_map)
    blocked = (work != BG) & (~np.isin(work, list(marker_colors)))
    out = work.copy()
    explicit = _iter_explicit_centers(work, marker_colors)
    centers: list[tuple[int, int, int]] = list(explicit)

    if use_symmetry:
        axis2, _ = _infer_axis2(work, marker_colors)
        occupied = {(r, c) for r, c, _ in explicit}
        for r, c, marker_color in explicit:
            mc = axis2 - c
            assert 0 <= mc < work.shape[1]
            if (r, int(mc)) not in occupied:
                occupied.add((r, int(mc)))
                centers.append((r, int(mc), marker_color))

    for r, c, marker_color in centers:
        _stamp(out, legend_map[int(marker_color)], r, c, blocked)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace work-area marker pixels with legend sprites, then mirror missing sprite placements across the inferred vertical symmetry axis.

    Input structure:
    - Black background with a top legend area, one solid grey separator row, and a lower work area.
    - The legend contains 2-3 monochrome 3x3 sprites with holes; each sprite has one differently colored marker pixel directly below its center.
    - The work area contains a large near-bilateral foreground figure plus scattered marker-color pixels placed on background cells.

    Rule:
    - Pair legend items from left to right: the kth marker color maps to the kth 3x3 sprite.
    - Remove the legend and separator; output only the work area.
    - For every work-area marker pixel, stamp its mapped sprite centered there, painting only the sprite foreground cells.
    - Infer the work figure's vertical symmetry axis from all non-marker foreground pixels, then mirror every explicit stamp center across that axis.
    - If a mirrored center has no explicit stamp, place the same sprite there; stamp pixels may cover background or marker cells, but never overwrite the original non-marker foreground figure.

    Key insight:
    - The legend determines which structured sprite each marker color means, while the work figure's symmetry determines which additional unmarked centers must receive mirrored copies of that same sprite.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    sep_row = _find_separator_row(g)
    legend_map, marker_colors = _parse_legend(g, sep_row)
    work = g[sep_row + 1 :].copy()
    return _solve_from_parsed(work, legend_map, use_symmetry=True)


def _draw_work_pattern(
    work_h: int,
    work_w: int,
    axis2: int,
    base_colors: list[int],
) -> tuple[np.ndarray, tuple[int, int, int, int], int]:
    work = np.zeros((work_h, work_w), dtype=int)

    frame_h = random.randint(8, work_h - 2)
    r0 = random.randint(1, work_h - frame_h - 1)
    r1 = r0 + frame_h - 1

    if axis2 % 2 == 0:
        axis_c = axis2 // 2
        span = random.randint(4, min(axis_c - 1, work_w - 2 - axis_c))
        c0 = axis_c - span
        c1 = axis_c + span
    else:
        left_c = axis2 // 2
        span = random.randint(4, min(left_c, work_w - 2 - left_c))
        c0 = left_c - span + 1
        c1 = left_c + span
    assert 1 <= c0 < c1 <= work_w - 2

    band_r = random.randint(r0 + 3, r1 - 3)

    work[r0, c0 : c1 + 1] = base_colors[0]
    work[r1, c0 : c1 + 1] = base_colors[0]
    work[r0 : r1 + 1, c0] = base_colors[1]
    work[r0 : r1 + 1, c1] = base_colors[1]
    work[band_r, c0 + 1 : c1] = base_colors[1]

    if random.random() < 0.8:
        row = random.choice([band_r - 2, band_r + 2])
        for c in range(c0 + 2, c1 - 1):
            mc = axis2 - c
            if c >= mc or not (c0 + 1 <= mc <= c1 - 1):
                continue
            if random.random() < 0.25:
                work[row, c] = base_colors[0]
                work[row, int(mc)] = base_colors[0]

    return work, (r0, c0, r1, c1), band_r


def _candidate_pairs(
    work: np.ndarray,
    axis2: int,
    frame_box: tuple[int, int, int, int],
    band_r: int,
) -> list[tuple[int, int, int]]:
    r0, c0, r1, c1 = frame_box
    rows = list(range(r0 + 2, band_r - 1)) + list(range(band_r + 2, r1 - 1))
    pairs: list[tuple[int, int, int]] = []
    for r in rows:
        for c in range(c0 + 2, c1 - 1):
            mc = axis2 - c
            if not (0 <= mc < work.shape[1]):
                continue
            mc = int(mc)
            if c >= mc or mc > c1 - 2 or mc - c < 4:
                continue
            left_patch = work[r - 1 : r + 2, c - 1 : c + 2]
            right_patch = work[r - 1 : r + 2, mc - 1 : mc + 2]
            if left_patch.shape == (3, 3) and right_patch.shape == (3, 3) and not left_patch.any() and not right_patch.any():
                pairs.append((int(r), int(c), int(mc)))
    random.shuffle(pairs)
    return pairs


def _nonoverlap(center: tuple[int, int], others: list[tuple[int, int]]) -> bool:
    r, c = center
    for orow, ocol in others:
        if abs(r - orow) <= 2 and abs(c - ocol) <= 2:
            return False
    return True


def _choose_pair_groups(
    pairs: list[tuple[int, int, int]],
    num_types: int,
) -> list[tuple[int, int, int, int]]:
    group_count = random.randint(num_types, min(len(pairs), num_types + 2))
    color_ids = list(range(num_types))
    while len(color_ids) < group_count:
        color_ids.append(random.randrange(num_types))
    random.shuffle(color_ids)

    modes = ["single"]
    while len(modes) < group_count:
        modes.append("double" if random.random() < 0.45 else "single")
    random.shuffle(modes)

    chosen: list[tuple[int, int, int, int]] = []
    occupied: list[tuple[int, int]] = []
    for color_id, mode in zip(color_ids, modes):
        for r, c_left, c_right in pairs:
            centers = [(r, c_left), (r, c_right)]
            if all(_nonoverlap(center, occupied) for center in centers):
                side = 0 if random.random() < 0.5 else 1
                chosen.append((r, c_left, c_right, color_id * 10 + (2 if mode == "double" else side)))
                occupied.extend(centers)
                break
        else:
            return []

    color_coverage = {spec[3] // 10 for spec in chosen}
    has_single = any(spec[3] % 10 != 2 for spec in chosen)
    has_double = any(spec[3] % 10 == 2 for spec in chosen)
    if color_coverage != set(range(num_types)) or not has_single:
        return []
    if num_types >= 2 and not has_double and random.random() < 0.6:
        return []
    return chosen


def _legend_columns(width: int, count: int) -> list[int]:
    choices = list(range(2, width - 2))
    random.shuffle(choices)
    cols: list[int] = []
    for c in choices:
        if all(abs(c - other) >= 4 for other in cols):
            cols.append(c)
            if len(cols) == count:
                return sorted(cols)
    return []


def _build_input() -> np.ndarray | None:
    marker_type_count = random.choice([2, 3])
    work_h = random.randint(11, 16)
    work_w = random.randint(13, 18)

    if random.random() < 0.5:
        axis2 = 2 * random.randint(5, work_w - 6)
    else:
        axis2 = 2 * random.randint(4, work_w - 6) + 1

    palette = [1, 2, 3, 4, 6, 7, 8, 9]
    random.shuffle(palette)
    base_colors = palette[:2]
    sprite_colors = palette[2 : 2 + marker_type_count]
    marker_colors = palette[2 + marker_type_count : 2 + 2 * marker_type_count]

    sprite_masks = random.sample(SPRITE_BANK, marker_type_count)
    legend_map = {
        int(marker_color): sprite_mask * int(sprite_color)
        for marker_color, sprite_color, sprite_mask in zip(marker_colors, sprite_colors, sprite_masks)
    }

    work, frame_box, band_r = _draw_work_pattern(work_h, work_w, axis2, base_colors)
    pairs = _candidate_pairs(work, axis2, frame_box, band_r)
    if len(pairs) < marker_type_count:
        return None

    groups = _choose_pair_groups(pairs, marker_type_count)
    if not groups:
        return None

    for r, c_left, c_right, encoded in groups:
        color_id = encoded // 10
        mode = encoded % 10
        marker_color = marker_colors[color_id]
        if mode == 2:
            work[r, c_left] = marker_color
            work[r, c_right] = marker_color
        elif mode == 0:
            work[r, c_left] = marker_color
        else:
            work[r, c_right] = marker_color

    top_pad = random.randint(0, 1)
    marker_row = top_pad + 3
    sep_row = marker_row + 1
    total_h = sep_row + 1 + work_h
    if total_h > 30:
        return None

    legend_cols = _legend_columns(work_w, marker_type_count)
    if not legend_cols:
        return None

    grid = np.zeros((total_h, work_w), dtype=int)
    grid[sep_row, :] = SEPARATOR
    grid[sep_row + 1 :, :] = work

    for c, marker_color, sprite in zip(legend_cols, marker_colors, sprite_masks):
        sprite_color = int(np.max(legend_map[int(marker_color)]))
        patch = sprite * sprite_color
        grid[marker_row - 3 : marker_row, c - 1 : c + 2] = patch
        grid[marker_row, c] = int(marker_color)

    if np.any(grid[:sep_row, :] == SEPARATOR):
        return None
    return grid


def _validate_generated_grid(grid: np.ndarray) -> bool:
    sep_row = _find_separator_row(grid)
    legend_map, marker_colors = _parse_legend(grid, sep_row)
    work = grid[sep_row + 1 :].copy()
    explicit = _iter_explicit_centers(work, marker_colors)
    if len(explicit) < len(marker_colors):
        return False

    counts = {color: 0 for color in marker_colors}
    for _, _, color in explicit:
        counts[color] += 1
    if min(counts.values()) < 1:
        return False

    axis2, scores = _infer_axis2(work, marker_colors)
    sorted_scores = sorted(scores.values(), reverse=True)
    if len(sorted_scores) > 1 and sorted_scores[0] == sorted_scores[1]:
        return False

    occupied = {(r, c) for r, c, _ in explicit}
    if not any((r, axis2 - c) not in occupied for r, c, _ in explicit):
        return False

    blocked = (work != BG) & (~np.isin(work, list(marker_colors)))
    for r, c, color in explicit:
        sprite = legend_map[color]
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if int(sprite[dr + 1, dc + 1]) == BG:
                    continue
                rr = r + dr
                cc = c + dc
                if not (0 <= rr < work.shape[0] and 0 <= cc < work.shape[1]):
                    return False
                if bool(blocked[rr, cc]):
                    return False

    solved = _solve_from_parsed(work, legend_map, use_symmetry=True)
    explicit_only = _solve_from_parsed(work, legend_map, use_symmetry=False)
    if np.array_equal(solved, explicit_only):
        return False

    keys = sorted(legend_map)
    if len(keys) >= 2:
        swapped = dict(legend_map)
        swapped[keys[0]], swapped[keys[1]] = swapped[keys[1]], swapped[keys[0]]
        if np.array_equal(solved, _solve_from_parsed(work, swapped, use_symmetry=True)):
            return False

    return solved.shape == work.shape and max(grid.shape) <= 30 and max(solved.shape) <= 30


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _build_input()
        if grid is not None and _validate_generated_grid(grid):
            return grid
    raise RuntimeError("failed to generate a valid input")
