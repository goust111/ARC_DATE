from __future__ import annotations

import itertools
from typing import Iterable

import numpy as np


BLACK = 0


def _split_panels(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert w % 2 == 1, "expected two equal-width panels plus one separator column"
    sep = w // 2
    assert np.all(g[:, sep] == g[0, sep]), "separator must be monochrome"
    left = g[:, :sep].copy()
    right = g[:, sep + 1 :].copy()
    assert left.shape == right.shape
    return left, right, int(g[0, sep])


def _score_shift(left: np.ndarray, right: np.ndarray, dr: int, dc: int) -> tuple[int, int] | None:
    h, w = left.shape
    r0 = max(0, dr)
    r1 = min(h, h + dr)
    c0 = max(0, dc)
    c1 = min(w, w + dc)
    if r0 >= r1 or c0 >= c1:
        return None

    left_view = left[r0:r1, c0:c1]
    right_view = right[r0 - dr : r1 - dr, c0 - dc : c1 - dc]
    both = (left_view != BLACK) & (right_view != BLACK)
    if both.any() and np.any(left_view[both] != right_view[both]):
        return None

    matches = int(both.sum())
    if matches == 0:
        return None
    overlap_area = int((r1 - r0) * (c1 - c0))
    return matches, overlap_area


def _candidate_shifts(left: np.ndarray, right: np.ndarray) -> list[tuple[int, int, int, int, int, int, int]]:
    h, w = left.shape
    candidates: list[tuple[int, int, int, int, int, int, int]] = []
    for dr in range(-h + 1, h):
        for dc in range(-w + 1, w):
            if dr == 0 and dc == 0:
                continue
            score = _score_shift(left, right, dr, dc)
            if score is None:
                continue
            matches, overlap_area = score
            candidates.append(
                (
                    matches,
                    overlap_area,
                    -abs(dr) - abs(dc),
                    -abs(dr),
                    -abs(dc),
                    dr,
                    dc,
                )
            )
    candidates.sort(reverse=True)
    return candidates


def _collect_observations(left: np.ndarray, right: np.ndarray, dr: int, dc: int) -> dict[tuple[int, int], int] | None:
    observations: dict[tuple[int, int], int] = {}

    for r, c in np.argwhere(left != BLACK):
        observations[(int(r), int(c))] = int(left[r, c])

    for r, c in np.argwhere(right != BLACK):
        key = (int(r) + int(dr), int(c) + int(dc))
        color = int(right[r, c])
        prev = observations.get(key)
        if prev is not None and prev != color:
            return None
        observations[key] = color

    return observations


def _period_phase_map(
    observations: dict[tuple[int, int], int], period_r: int, period_c: int
) -> dict[tuple[int, int], int] | None:
    phase_map: dict[tuple[int, int], int] = {}
    for (r, c), color in observations.items():
        key = (r % period_r, c % period_c)
        prev = phase_map.get(key)
        if prev is not None and prev != color:
            return None
        phase_map[key] = color
    return phase_map


def _covers_output(
    phase_map: dict[tuple[int, int], int], out_shape: tuple[int, int], period_r: int, period_c: int
) -> bool:
    h, w = out_shape
    for r in range(h):
        for c in range(w):
            if (r % period_r, c % period_c) not in phase_map:
                return False
    return True


def _infer_period(
    observations: dict[tuple[int, int], int], out_shape: tuple[int, int]
) -> tuple[int, int, dict[tuple[int, int], int]] | None:
    rows = [coord[0] for coord in observations]
    cols = [coord[1] for coord in observations]
    max_period_r = max(rows) - min(rows) + 1
    max_period_c = max(cols) - min(cols) + 1

    pairs = [
        (period_r, period_c)
        for period_r in range(1, max_period_r + 1)
        for period_c in range(1, max_period_c + 1)
    ]
    pairs.sort(key=lambda item: (item[0] * item[1], item[0] + item[1], item[0], item[1]))

    for period_r, period_c in pairs:
        phase_map = _period_phase_map(observations, period_r, period_c)
        if phase_map is None:
            continue
        if not _covers_output(phase_map, out_shape, period_r, period_c):
            continue
        return period_r, period_c, phase_map
    return None


def _render_output(
    out_shape: tuple[int, int], period_r: int, period_c: int, phase_map: dict[tuple[int, int], int]
) -> np.ndarray:
    h, w = out_shape
    out = np.zeros((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            out[r, c] = phase_map[(r % period_r, c % period_c)]
    return out


def _output_fits(output: np.ndarray, left: np.ndarray, right: np.ndarray, dr: int, dc: int) -> bool:
    left_mask = left != BLACK
    if np.any(output[left_mask] != left[left_mask]):
        return False

    h, w = output.shape
    for r, c in np.argwhere(right != BLACK):
        rr = int(r) + int(dr)
        cc = int(c) + int(dc)
        if 0 <= rr < h and 0 <= cc < w and output[rr, cc] != right[r, c]:
            return False
    return True


def _solve_aligned_left(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    candidates = _candidate_shifts(left, right)
    assert candidates, "no valid non-zero alignment shift"

    grouped = itertools.groupby(candidates, key=lambda item: item[:5])
    for _score, group in grouped:
        outputs: list[np.ndarray] = []
        for candidate in group:
            dr, dc = candidate[5], candidate[6]
            observations = _collect_observations(left, right, dr, dc)
            if observations is None:
                continue
            period = _infer_period(observations, left.shape)
            if period is None:
                continue
            period_r, period_c, phase_map = period
            output = _render_output(left.shape, period_r, period_c, phase_map)
            if _output_fits(output, left, right, dr, dc):
                outputs.append(output)
        if outputs:
            first = outputs[0]
            assert all(np.array_equal(first, other) for other in outputs[1:]), "ambiguous reconstruction"
            return first

    raise AssertionError("no reconstruction fits the aligned windows")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align two shifted corrupted wallpaper windows, merge their clues, and complete the left window by periodic phase.

    Input structure:
    - The input contains two equal-size panels separated by a single monochrome separator column.
    - Each panel is a cropped window from the same infinite axis-aligned repeating wallpaper, with black cells marking corruption.
    - The right panel comes from a non-zero row/column shift relative to the left panel.

    Rule:
    - Find the unique non-zero shift whose overlapping non-black cells never disagree in color.
    - Move the right panel back into the left panel's coordinate system with that shift and combine all non-black evidence.
    - Infer the smallest row period and column period consistent with the combined observations, then fill every remaining black cell of the left panel by phase.

    Key insight:
    - The two panels are not a direct before/after pair: the right one must be aligned first, then its visible cells complement the left, and only the remaining gaps require periodic completion.
    """
    left, right, _separator = _split_panels(grid)
    return _solve_aligned_left(left, right)


def _minimal_period_1d(tile: np.ndarray, axis: int) -> int:
    size = tile.shape[axis]
    for period in range(1, size + 1):
        if np.array_equal(tile, np.roll(tile, -period, axis=axis)):
            return period
    raise AssertionError("unreachable")


def _tile_wallpaper(tile: np.ndarray, h: int, w: int, row_offset: int = 0, col_offset: int = 0) -> np.ndarray:
    period_r, period_c = tile.shape
    rows = (np.arange(h)[:, None] + int(row_offset)) % period_r
    cols = (np.arange(w)[None, :] + int(col_offset)) % period_c
    return tile[rows, cols]


def _phase_groups(
    h: int, w: int, period_r: int, period_c: int, row_offset: int = 0, col_offset: int = 0
) -> dict[tuple[int, int], list[tuple[int, int]]]:
    groups = {(pr, pc): [] for pr in range(period_r) for pc in range(period_c)}
    for r in range(h):
        for c in range(w):
            groups[((r + row_offset) % period_r, (c + col_offset) % period_c)].append((r, c))
    return groups


def _visible_phase_sets(
    left: np.ndarray, right: np.ndarray, period_r: int, period_c: int, dr: int, dc: int
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]:
    left_set = {(int(r) % period_r, int(c) % period_c) for r, c in np.argwhere(left != BLACK)}
    right_set = {
        ((int(r) + dr) % period_r, (int(c) + dc) % period_c)
        for r, c in np.argwhere(right != BLACK)
    }
    return left_set, right_set, left_set | right_set


def _merged_left_view(left: np.ndarray, right: np.ndarray, dr: int, dc: int) -> np.ndarray:
    merged = left.copy()
    h, w = merged.shape
    for r, c in np.argwhere(right != BLACK):
        rr = int(r) + int(dr)
        cc = int(c) + int(dc)
        if 0 <= rr < h and 0 <= cc < w and merged[rr, cc] == BLACK:
            merged[rr, cc] = right[r, c]
    return merged


def _alignment_overlap(h: int, w: int, dr: int, dc: int) -> list[tuple[int, int, int, int]]:
    cells: list[tuple[int, int, int, int]] = []
    for r in range(max(0, dr), min(h, h + dr)):
        for c in range(max(0, dc), min(w, w + dc)):
            cells.append((r, c, r - dr, c - dc))
    return cells


def _sample_tile(rng: np.random.Generator, period_r: int, period_c: int) -> tuple[np.ndarray, list[int]]:
    colors = list(range(1, 10))
    while True:
        palette_size = int(rng.integers(3, min(6, period_r * period_c, 8) + 1))
        palette = rng.choice(colors, size=palette_size, replace=False)
        tile = rng.choice(palette, size=(period_r, period_c), replace=True).astype(int)
        if len(np.unique(tile)) < min(3, palette_size):
            continue
        if _minimal_period_1d(tile, axis=0) != period_r:
            continue
        if _minimal_period_1d(tile, axis=1) != period_c:
            continue
        return tile, [int(x) for x in palette]


def _pick_shift(rng: np.random.Generator, period_r: int, period_c: int) -> tuple[int, int]:
    row_choices = [value for value in range(-(period_r - 1) // 2, (period_r - 1) // 2 + 1) if value != 0]
    col_choices = [value for value in range(-(period_c - 1) // 2, (period_c - 1) // 2 + 1) if value != 0]
    branch = int(rng.integers(0, 3))
    if branch == 0:
        return int(rng.choice(row_choices)), 0
    if branch == 1:
        return 0, int(rng.choice(col_choices))
    return int(rng.choice(row_choices)), int(rng.choice(col_choices))


def _apply_corruption(
    rng: np.random.Generator,
    left_full: np.ndarray,
    right_full: np.ndarray,
    period_r: int,
    period_c: int,
    dr: int,
    dc: int,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = left_full.shape
    left = left_full.copy()
    right = right_full.copy()
    all_phases = [(pr, pc) for pr in range(period_r) for pc in range(period_c)]
    left_groups = _phase_groups(h, w, period_r, period_c)
    right_groups = _phase_groups(h, w, period_r, period_c, row_offset=dr, col_offset=dc)

    left_drop_n = 1 + int(period_r * period_c >= 12 and rng.random() < 0.45)
    right_drop_n = 1 + int(period_r * period_c >= 12 and rng.random() < 0.45)

    shuffled = list(all_phases)
    rng.shuffle(shuffled)
    left_drop = set(shuffled[:left_drop_n])
    remaining = [phase for phase in shuffled if phase not in left_drop]
    rng.shuffle(remaining)
    right_drop = set(remaining[:right_drop_n])

    for phase in left_drop:
        for r, c in left_groups[phase]:
            left[r, c] = BLACK
    for phase in right_drop:
        for r, c in right_groups[phase]:
            right[r, c] = BLACK

    overlap = _alignment_overlap(h, w, dr, dc)

    complementary_holes = int(rng.integers(2, 6))
    for _ in range(complementary_holes):
        side = int(rng.integers(0, 2))
        candidates = [
            (r, c, rr, cc)
            for r, c, rr, cc in overlap
            if left[r, c] != BLACK and right[rr, cc] != BLACK
        ]
        if not candidates:
            break
        r, c, rr, cc = candidates[int(rng.integers(0, len(candidates)))]
        if side == 0:
            left[r, c] = BLACK
        else:
            right[rr, cc] = BLACK

    double_holes = int(rng.integers(1, 4))
    for _ in range(double_holes):
        phase_support = {phase: 0 for phase in all_phases}
        for r, c in np.argwhere(left != BLACK):
            phase_support[(int(r) % period_r, int(c) % period_c)] += 1
        for r, c in np.argwhere(right != BLACK):
            phase_support[((int(r) + dr) % period_r, (int(c) + dc) % period_c)] += 1

        candidates = []
        for r, c, rr, cc in overlap:
            phase = (r % period_r, c % period_c)
            if left[r, c] != BLACK and right[rr, cc] != BLACK and phase_support[phase] >= 3:
                candidates.append((r, c, rr, cc))
        if not candidates:
            break
        r, c, rr, cc = candidates[int(rng.integers(0, len(candidates)))]
        left[r, c] = BLACK
        right[rr, cc] = BLACK

    return left, right


def _valid_instance(
    inp: np.ndarray,
    expected: np.ndarray,
    period_r: int,
    period_c: int,
    dr: int,
    dc: int,
) -> bool:
    left, right, _separator = _split_panels(inp)
    left_set, right_set, union_set = _visible_phase_sets(left, right, period_r, period_c, dr, dc)
    all_phases = {(pr, pc) for pr in range(period_r) for pc in range(period_c)}

    if left_set == all_phases or right_set == all_phases:
        return False
    if union_set != all_phases:
        return False

    merged = _merged_left_view(left, right, dr, dc)
    if not np.any(merged == BLACK):
        return False
    if np.count_nonzero(merged != BLACK) <= np.count_nonzero(left != BLACK):
        return False

    candidates = _candidate_shifts(left, right)
    if not candidates:
        return False
    best = candidates[0]
    if (best[5], best[6]) != (dr, dc):
        return False
    if sum(1 for candidate in candidates if candidate[:5] == best[:5]) != 1:
        return False

    try:
        solved = main(inp)
    except AssertionError:
        return False
    return np.array_equal(solved, expected)


def _generate_attempt(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    period_r = int(rng.integers(3, 6))
    period_c = int(rng.integers(3, 6))
    h = int(rng.integers(max(8, period_r + 3), 15))
    w = int(rng.integers(max(8, period_c + 3), 15))

    tile, palette = _sample_tile(rng, period_r, period_c)
    dr, dc = _pick_shift(rng, period_r, period_c)

    left_full = _tile_wallpaper(tile, h, w, row_offset=0, col_offset=0)
    right_full = _tile_wallpaper(tile, h, w, row_offset=dr, col_offset=dc)
    left, right = _apply_corruption(rng, left_full, right_full, period_r, period_c, dr, dc)

    if np.count_nonzero(left != BLACK) < (h * w) // 3:
        return None
    if np.count_nonzero(right != BLACK) < (h * w) // 3:
        return None

    remaining_colors = [color for color in range(1, 10) if color not in set(palette)]
    if not remaining_colors:
        return None
    separator = int(rng.choice(remaining_colors))
    separator_col = np.full((h, 1), separator, dtype=int)
    inp = np.concatenate([left, separator_col, right], axis=1)

    expected = left_full
    if not _valid_instance(inp, expected, period_r, period_c, dr, dc):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(400):
        attempt = _generate_attempt(rng)
        if attempt is not None:
            return attempt[0]
    raise RuntimeError("failed to generate a valid wallpaper alignment instance")
