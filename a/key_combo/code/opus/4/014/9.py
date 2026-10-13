from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
PATTERN_COLORS = (1, 2, 3, 4)
DISTRACTOR_COLORS = (5, 6, 7)

GEN_ATTEMPTS = 400
TILE_ATTEMPTS = 400
POSITION_ATTEMPTS = 400


def _color_counts(grid: np.ndarray) -> Counter[int]:
    return Counter(int(value) for value in grid.ravel() if int(value) != BLACK)


def _infer_distractor_colors(grid: np.ndarray) -> set[int]:
    counts = _color_counts(grid)
    if not counts:
        return set()

    items = sorted(counts.items(), key=lambda item: (item[1], item[0]))
    if len(items) < 2:
        return set()

    strong_splits: list[tuple[float, int, int, int]] = []
    fallback_splits: list[tuple[float, int, int, int]] = []
    for split in range(1, len(items)):
        common_size = len(items) - split
        if common_size < 2:
            continue
        rare_max = max(count for _color, count in items[:split])
        common_min = min(count for _color, count in items[split:])
        ratio = common_min / rare_max
        fallback_splits.append((ratio, common_size, -split, split))
        if rare_max * 2 < common_min:
            strong_splits.append((ratio, common_size, -split, split))

    if strong_splits:
        split = max(strong_splits)[-1]
    elif fallback_splits:
        split = max(fallback_splits)[-1]
    else:
        return set()

    return {color for color, _count in items[:split]}


def _erase_distractors(grid: np.ndarray) -> tuple[np.ndarray, set[int]]:
    distractor_colors = _infer_distractor_colors(grid)
    cleaned = np.asarray(grid, dtype=int).copy()
    for color in distractor_colors:
        cleaned[cleaned == color] = BLACK
    return cleaned, distractor_colors


def _period_score(grid: np.ndarray, dr: int, dc: int) -> tuple[int, int]:
    height, width = map(int, grid.shape)
    r0 = max(0, -dr)
    r1 = min(height, height - dr)
    c0 = max(0, -dc)
    c1 = min(width, width - dc)
    if r0 >= r1 or c0 >= c1:
        return 0, 0

    left = grid[r0:r1, c0:c1]
    right = grid[r0 + dr : r1 + dr, c0 + dc : c1 + dc]
    both_visible = (left != BLACK) & (right != BLACK)
    support = int(np.sum(both_visible))
    if support == 0:
        return 0, 0
    conflicts = int(np.count_nonzero(left[both_visible] != right[both_visible]))
    return support, conflicts


def _best_axis_period(grid: np.ndarray, axis: int) -> int | None:
    height, width = map(int, grid.shape)
    candidates: list[tuple[int, int, int]] = []
    if axis == 0:
        for dr in range(1, height):
            support, conflicts = _period_score(grid, dr, 0)
            if support > 0 and conflicts == 0:
                candidates.append((support, -dr, dr))
    else:
        for dc in range(1, width):
            support, conflicts = _period_score(grid, 0, dc)
            if support > 0 and conflicts == 0:
                candidates.append((support, -dc, dc))

    if not candidates:
        return None
    return max(candidates)[2]


def _orbit_colors(grid: np.ndarray, r: int, c: int, row_period: int | None, col_period: int | None) -> set[int]:
    height, width = map(int, grid.shape)
    rows = [int(r)] if row_period is None else list(range(int(r) % row_period, height, row_period))
    cols = [int(c)] if col_period is None else list(range(int(c) % col_period, width, col_period))
    return {
        int(grid[rr, cc])
        for rr in rows
        for cc in cols
        if int(grid[rr, cc]) != BLACK
    }


def _recover_periodic_pattern(cleaned: np.ndarray) -> np.ndarray:
    row_period = _best_axis_period(cleaned, axis=0)
    col_period = _best_axis_period(cleaned, axis=1)
    assert row_period is not None or col_period is not None, "expected a visible translational period"

    out = cleaned.copy()
    for r, c in np.argwhere(out == BLACK):
        colors = _orbit_colors(cleaned, int(r), int(c), row_period, col_period)
        assert len(colors) == 1, "expected a unique visible orbit color for each missing cell"
        out[r, c] = next(iter(colors))

    assert not np.any(out == BLACK), "all missing cells should be recoverable"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove rare-color distractors and restore the periodic tiling by orbit-consistent translation matches.

    Input structure:
    - The grid is a fully tiled periodic color pattern with no real black background; black cells are missing samples.
    - The true pattern uses only blue/red/green/yellow and repeats with at least one translational period.
    - A few extra pixels in grey/pink/orange overwrite true pattern cells and act as distractors.

    Rule:
    - Count non-black colors and treat the low-frequency color cluster as distractor colors.
    - Replace every distractor pixel with black so those cells become additional missing positions.
    - Detect the smallest visible row/column translation periods from the remaining non-black pixels.
    - For each black cell, collect colors from its translation orbit and fill it with the orbit's unique visible color.

    Key insight:
    - Rare colors do not belong to the periodic palette, so they must be erased before symmetry analysis; once removed, every missing cell shares an orbit with visible copies of exactly one true color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    cleaned, _distractor_colors = _erase_distractors(g)
    return _recover_periodic_pattern(cleaned)


def _tile_axis_period(tile: np.ndarray, axis: int) -> int:
    height, width = map(int, tile.shape)
    if axis == 0:
        for dr in range(1, height):
            if np.array_equal(tile[:-dr, :], tile[dr:, :]):
                return dr
        return height
    for dc in range(1, width):
        if np.array_equal(tile[:, :-dc], tile[:, dc:]):
            return dc
    return width


def _random_partition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    previous = 0
    out = []
    for cut in cuts + [total]:
        out.append(cut - previous)
        previous = cut
    return out


def _make_base_tile(row_period: int, col_period: int, palette: list[int]) -> np.ndarray | None:
    cell_count = row_period * col_period
    min_per_color = 2
    if cell_count < min_per_color * len(palette):
        return None

    for _ in range(TILE_ATTEMPTS):
        values: list[int] = []
        remaining = cell_count
        for idx, color in enumerate(palette):
            colors_left = len(palette) - idx - 1
            if colors_left == 0:
                take = remaining
            else:
                low = min_per_color
                high = remaining - min_per_color * colors_left
                take = random.randint(low, high)
            values.extend([color] * take)
            remaining -= take

        random.shuffle(values)
        tile = np.array(values, dtype=int).reshape(row_period, col_period)
        if set(map(int, np.unique(tile))) != set(palette):
            continue
        if _tile_axis_period(tile, axis=0) != row_period:
            continue
        if _tile_axis_period(tile, axis=1) != col_period:
            continue
        return tile
    return None


def _phase_groups(height: int, width: int, row_period: int, col_period: int) -> dict[tuple[int, int], list[tuple[int, int]]]:
    groups: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r in range(height):
        for c in range(width):
            groups.setdefault((r % row_period, c % col_period), []).append((r, c))
    return groups


def _pick_removed_positions(
    full: np.ndarray,
    row_period: int,
    col_period: int,
    n_occlusions: int,
    n_distractors: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    height, width = map(int, full.shape)
    groups = _phase_groups(height, width, row_period, col_period)
    phase_visible = {phase: len(coords) for phase, coords in groups.items()}

    all_coords = [(r, c) for r in range(height) for c in range(width)]
    random.shuffle(all_coords)

    occlusions: list[tuple[int, int]] = []
    for r, c in all_coords:
        phase = (r % row_period, c % col_period)
        if phase_visible[phase] <= 2:
            continue
        occlusions.append((r, c))
        phase_visible[phase] -= 1
        if len(occlusions) == n_occlusions:
            break
    if len(occlusions) != n_occlusions:
        return None

    occupied = set(occlusions)
    remaining = [coord for coord in all_coords if coord not in occupied]
    random.shuffle(remaining)

    distractors: list[tuple[int, int]] = []
    for r, c in remaining:
        phase = (r % row_period, c % col_period)
        if phase_visible[phase] <= 2:
            continue
        distractors.append((r, c))
        phase_visible[phase] -= 1
        if len(distractors) == n_distractors:
            break
    if len(distractors) != n_distractors:
        return None

    return occlusions, distractors


def _assign_distractor_colors(coords: list[tuple[int, int]]) -> list[tuple[tuple[int, int], int]] | None:
    distinct_color_count = random.randint(1, min(3, len(coords)))
    counts = _random_partition(len(coords), distinct_color_count)
    colors = random.sample(list(DISTRACTOR_COLORS), distinct_color_count)

    available = list(coords)
    assignment: list[tuple[tuple[int, int], int]] = []
    used_positions: set[tuple[int, int]] = set()
    for color, need in zip(colors, counts):
        chosen: list[tuple[int, int]] = []
        for _ in range(need):
            candidates = [
                coord
                for coord in available
                if coord not in used_positions
                and all(abs(coord[0] - rr) + abs(coord[1] - cc) > 1 for rr, cc in chosen)
            ]
            if not candidates:
                return None
            coord = random.choice(candidates)
            chosen.append(coord)
            used_positions.add(coord)
            assignment.append((coord, color))
    return assignment


def _semantic_checks(
    inp: np.ndarray,
    full: np.ndarray,
    distractor_assignment: list[tuple[tuple[int, int], int]],
    expected_periods: tuple[int, int],
) -> bool:
    expected_distractor_colors = {color for _coord, color in distractor_assignment}
    if _infer_distractor_colors(inp) != expected_distractor_colors:
        return False

    cleaned, _ = _erase_distractors(inp)
    row_period = _best_axis_period(cleaned, axis=0)
    col_period = _best_axis_period(cleaned, axis=1)
    if (row_period, col_period) != expected_periods:
        return False

    missing_positions = [tuple(map(int, coord)) for coord in np.argwhere(cleaned == BLACK)]
    if len(missing_positions) < len(distractor_assignment):
        return False

    for r, c in missing_positions:
        colors = _orbit_colors(cleaned, r, c, row_period, col_period)
        if colors != {int(full[r, c])}:
            return False

    ambiguous_distractors = 0
    for (r, c), distractor_color in distractor_assignment:
        raw_orbit = _orbit_colors(inp, r, c, row_period, col_period)
        true_color = int(full[r, c])
        if distractor_color in raw_orbit and true_color in raw_orbit and len(raw_orbit) > 1:
            ambiguous_distractors += 1
    if ambiguous_distractors == 0:
        return False

    try:
        recovered = main(inp)
    except AssertionError:
        return False
    return np.array_equal(recovered, full)


def _gen_attempt() -> np.ndarray | None:
    row_period = random.choice((3, 4))
    col_period = random.choice((3, 4))
    row_repeats = random.randint(3, 5)
    col_repeats = random.randint(3, 5)
    palette = random.sample(list(PATTERN_COLORS), random.randint(3, 4))

    tile = _make_base_tile(row_period, col_period, palette)
    if tile is None:
        return None

    full = np.tile(tile, (row_repeats, col_repeats))
    n_occlusions = random.randint(8, 12)
    n_distractors = random.randint(3, 5)
    picked = _pick_removed_positions(full, row_period, col_period, n_occlusions, n_distractors)
    if picked is None:
        return None
    occlusions, distractor_coords = picked

    distractor_assignment = _assign_distractor_colors(distractor_coords)
    if distractor_assignment is None:
        return None

    inp = full.copy()
    for r, c in occlusions:
        inp[r, c] = BLACK
    for (r, c), color in distractor_assignment:
        inp[r, c] = color

    if not _semantic_checks(inp, full, distractor_assignment, (row_period, col_period)):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid periodic-restoration sample")
