from __future__ import annotations

import random

import numpy as np


BG = 0
PILLAR = 5
LINE_PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]

MIN_REGIONS = 3
MAX_REGIONS = 5
MAX_GENERATION_ATTEMPTS = 400
MAX_LAYOUT_ATTEMPTS = 300


def _pillar_columns(grid: np.ndarray) -> list[int]:
    cols: list[int] = []
    for c in range(grid.shape[1]):
        value = int(grid[0, c])
        if value == BG:
            continue
        if bool(np.all(grid[:, c] == value)):
            cols.append(int(c))
    assert cols, "expected at least one full-height pillar"
    return cols


def _region_slices(width: int, pillar_cols: list[int]) -> list[tuple[int, int]]:
    regions: list[tuple[int, int]] = []
    start = 0
    for col in pillar_cols:
        assert col > start, "regions must have positive width"
        regions.append((start, col))
        start = col + 1
    assert start < width, "final region must have positive width"
    regions.append((start, width))
    return regions


def _full_width_lines(region: np.ndarray) -> dict[int, int]:
    rows_by_color: dict[int, int] = {}
    for r in range(region.shape[0]):
        value = int(region[r, 0])
        if value == BG:
            continue
        if bool(np.all(region[r] == value)):
            assert value not in rows_by_color, "each color should have at most one line per region"
            rows_by_color[value] = int(r)
    return rows_by_color


def _has_marker(region: np.ndarray, row: int, color: int, offset: int) -> bool:
    top = row + offset
    if top < 0 or top + 2 > region.shape[0] or region.shape[1] < 2:
        return False
    for c in range(region.shape[1] - 1):
        if bool(np.all(region[top : top + 2, c : c + 2] == color)):
            return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore the missing horizontal segments by following each color's per-pillar vertical shift.

    Input structure:
    - The background is black.
    - One-pixel-wide full-height grey pillars split the grid into vertical stripe regions.
    - Each stripe may contain several one-pixel-tall horizontal line segments, and each
      segment spans the full width of its stripe.
    - In the leftmost stripe, every line color also has a touching 2x2 square marker of the
      same color placed directly above or directly below that first segment.

    Rule:
    - Use the full-height pillars to recover the stripe boundaries.
    - Read each leftmost-stripe marker: a square above the first segment means that color
      moves up by one row after every pillar crossing, while a square below means it moves
      down by one row after every crossing.
    - Keep all already visible segments unchanged.
    - For any stripe where a color's segment is missing, draw the full-width segment on the
      row implied by the first segment row plus the repeated per-pillar offset.

    Key insight:
    The marker square is not an extra object to extend locally; it encodes a global
    row-drift rule that must be applied repeatedly from stripe to stripe.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    pillar_cols = _pillar_columns(g)
    regions = _region_slices(g.shape[1], pillar_cols)
    first_c0, first_c1 = regions[0]
    first_region = g[:, first_c0:first_c1]

    first_lines = _full_width_lines(first_region)
    assert first_lines, "expected visible seed lines in the first region"

    directions: dict[int, int] = {}
    for color, row in first_lines.items():
        marker_above = _has_marker(first_region, row, color, -2)
        marker_below = _has_marker(first_region, row, color, 1)
        assert marker_above != marker_below, "each seed line needs exactly one direction marker"
        directions[int(color)] = -1 if marker_above else 1

    out = g.copy()
    for region_idx, (c0, c1) in enumerate(regions):
        region_lines = _full_width_lines(g[:, c0:c1])
        for color, first_row in first_lines.items():
            target_row = int(first_row + directions[int(color)] * region_idx)
            assert 0 <= target_row < g.shape[0], "trajectory must stay inside the grid"
            present_row = region_lines.get(int(color))
            if present_row is None:
                out[target_row, c0:c1] = int(color)
            else:
                assert present_row == target_row, "visible segments must already match the drift rule"

    return out


def _sample_region_widths(region_count: int) -> list[int] | None:
    for _ in range(MAX_LAYOUT_ATTEMPTS):
        widths = [random.randint(4, 7)]
        widths.extend(random.randint(3, 6) for _ in range(region_count - 1))
        if sum(widths) + region_count - 1 <= 30:
            return widths
    return None


def _candidate_rows(height: int, direction: int, region_count: int) -> list[int]:
    if direction < 0:
        low = max(2, region_count - 1)
        high = height - 1
    else:
        low = 0
        high = min(height - 3, height - region_count)
    return list(range(low, high + 1))


def _first_region_occupied(base_row: int, direction: int) -> set[int]:
    if direction < 0:
        return {base_row - 2, base_row - 1, base_row}
    return {base_row, base_row + 1, base_row + 2}


def _trajectory_rows(base_row: int, direction: int, region_count: int) -> list[int]:
    return [int(base_row + direction * region_idx) for region_idx in range(region_count)]


def _assign_base_rows(height: int, directions: list[int], region_count: int) -> list[int] | None:
    color_count = len(directions)
    candidates = [_candidate_rows(height, direction, region_count) for direction in directions]
    if any(not rows for rows in candidates):
        return None

    order = list(range(color_count))
    random.shuffle(order)
    order.sort(key=lambda idx: len(candidates[idx]))

    chosen: dict[int, int] = {}
    occupied_first: dict[int, set[int]] = {}
    trajectories: dict[int, list[int]] = {}

    def backtrack(pos: int) -> bool:
        if pos == len(order):
            return True

        idx = order[pos]
        options = candidates[idx][:]
        random.shuffle(options)
        for base_row in options:
            occ = _first_region_occupied(base_row, directions[idx])
            traj = _trajectory_rows(base_row, directions[idx], region_count)

            ok = True
            for other_idx in chosen:
                if occ & occupied_first[other_idx]:
                    ok = False
                    break
                other_traj = trajectories[other_idx]
                if any(traj[region_idx] == other_traj[region_idx] for region_idx in range(region_count)):
                    ok = False
                    break
            if not ok:
                continue

            chosen[idx] = int(base_row)
            occupied_first[idx] = occ
            trajectories[idx] = traj
            if backtrack(pos + 1):
                return True
            del chosen[idx]
            del occupied_first[idx]
            del trajectories[idx]
        return False

    if not backtrack(0):
        return None
    return [chosen[idx] for idx in range(color_count)]


def _sample_visibility_masks(color_count: int, region_count: int) -> list[list[bool]]:
    later_regions = region_count - 1
    masks: list[list[bool]] = []
    for _ in range(color_count):
        shown_count = random.randint(1, later_regions - 1)
        shown = set(random.sample(range(later_regions), shown_count))
        masks.append([region_idx in shown for region_idx in range(later_regions)])
    return masks


def _render_scene(
    height: int,
    widths: list[int],
    colors: list[int],
    directions: list[int],
    base_rows: list[int],
    visible_masks: list[list[bool]],
) -> tuple[np.ndarray, np.ndarray]:
    region_count = len(widths)
    width = int(sum(widths) + region_count - 1)
    inp = np.full((height, width), BG, dtype=int)
    out = np.full_like(inp, BG)

    regions: list[tuple[int, int]] = []
    cursor = 0
    for region_idx, region_width in enumerate(widths):
        c0 = cursor
        c1 = cursor + region_width
        regions.append((c0, c1))
        cursor = c1
        if region_idx + 1 < region_count:
            inp[:, cursor] = PILLAR
            out[:, cursor] = PILLAR
            cursor += 1
    assert cursor == width

    for color_idx, color in enumerate(colors):
        for region_idx, (c0, c1) in enumerate(regions):
            row = int(base_rows[color_idx] + directions[color_idx] * region_idx)
            out[row, c0:c1] = int(color)
            if region_idx == 0 or visible_masks[color_idx][region_idx - 1]:
                inp[row, c0:c1] = int(color)

        first_c0, first_c1 = regions[0]
        marker_col = random.randint(first_c0, first_c1 - 2)
        base_row = base_rows[color_idx]
        if directions[color_idx] < 0:
            rows = slice(base_row - 2, base_row)
        else:
            rows = slice(base_row + 1, base_row + 3)
        inp[rows, marker_col : marker_col + 2] = int(color)
        out[rows, marker_col : marker_col + 2] = int(color)

    return inp, out


def _build_candidate() -> tuple[np.ndarray, np.ndarray] | None:
    region_count = random.randint(MIN_REGIONS, MAX_REGIONS)
    widths = _sample_region_widths(region_count)
    if widths is None:
        return None

    max_colors = min(5, len(LINE_PALETTE))
    color_count = random.randint(2, max_colors)
    min_height = max(9, 3 * color_count + region_count)
    if min_height > 30:
        return None
    height = random.randint(min_height, min(30, min_height + 6))

    directions = [random.choice([-1, 1]) for _ in range(color_count)]
    base_rows = _assign_base_rows(height, directions, region_count)
    if base_rows is None:
        return None

    colors = random.sample(LINE_PALETTE, color_count)
    visible_masks = _sample_visibility_masks(color_count, region_count)
    inp, out = _render_scene(height, widths, colors, directions, base_rows, visible_masks)

    if bool(np.array_equal(inp, out)):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _build_candidate()
        if built is None:
            continue
        inp, expected = built
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid puzzle instance")
