from __future__ import annotations

import random

import numpy as np


BG = 0
CARDINAL_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
SOLVER_MIN_LEGS = 4
GEN_MIN_LEGS = 8
GEN_MAX_LEGS = 14
GEN_MAX_SIZE = 30
GEN_MAX_PADDING = 8
GEN_MIN_PERIOD = 3
GEN_MAX_PERIOD = 7
GEN_ATTEMPTS = 600


def _orientations() -> tuple[tuple[tuple[int, int], ...], ...]:
    out: list[tuple[tuple[int, int], ...]] = []
    for start in range(4):
        cw = tuple(CARDINAL_DIRS[(start + i) % 4] for i in range(4))
        ccw = tuple(CARDINAL_DIRS[(start - i) % 4] for i in range(4))
        for dirs in (cw, ccw):
            if dirs not in out:
                out.append(dirs)
    return tuple(out)


ORIENTATIONS = _orientations()


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _leg_length(leg_index: int) -> int:
    return (leg_index + 1) // 2


def _spiral_relative_path(dirs: tuple[tuple[int, int], ...], n_legs: int) -> list[tuple[int, int]]:
    path = [(0, 0)]
    r = 0
    c = 0
    used = {(0, 0)}
    for leg in range(1, n_legs + 1):
        dr, dc = dirs[(leg - 1) % 4]
        for _ in range(_leg_length(leg)):
            r += dr
            c += dc
            assert (r, c) not in used
            used.add((r, c))
            path.append((r, c))
    return path


def _spiral_leg_spans(n_legs: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = 1
    for leg in range(1, n_legs + 1):
        length = _leg_length(leg)
        spans.append((cursor, cursor + length))
        cursor += length
    return spans


def _path_bbox(path: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in path]
    cols = [c for _r, c in path]
    return min(rows), min(cols), max(rows), max(cols)


def _translate_path(
    path: list[tuple[int, int]] | tuple[tuple[int, int], ...], dr: int, dc: int
) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for r, c in path]


def _spiral_prefixes(
    center: tuple[int, int], dirs: tuple[tuple[int, int], ...], shape: tuple[int, int]
) -> list[tuple[tuple[int, int], ...]]:
    r, c = center
    path = [center]
    used = {center}
    prefixes: list[tuple[tuple[int, int], ...]] = []
    for leg in range(1, shape[0] * shape[1] + 1):
        dr, dc = dirs[(leg - 1) % 4]
        steps = _leg_length(leg)
        trial = []
        rr, cc = r, c
        ok = True
        for _ in range(steps):
            rr += dr
            cc += dc
            if not _in_bounds(shape, rr, cc) or (rr, cc) in used:
                ok = False
                break
            trial.append((rr, cc))
        if not ok:
            break
        path.extend(trial)
        used.update(trial)
        r, c = rr, cc
        prefixes.append(tuple(path))
    return prefixes


def _enumerate_spiral_candidates(mask: np.ndarray) -> list[tuple[tuple[int, int], ...]]:
    visible = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    assert visible, "expected at least one visible spiral cell"
    visible_set = set(visible)

    seen: set[tuple[tuple[int, int], ...]] = set()
    candidates: list[tuple[tuple[int, int], ...]] = []
    for center in visible:
        for dirs in ORIENTATIONS:
            for leg_count, path in enumerate(_spiral_prefixes(center, dirs, mask.shape), start=1):
                if leg_count < SOLVER_MIN_LEGS:
                    continue
                if path[-1] not in visible_set:
                    continue
                if not visible_set.issubset(path):
                    continue
                if path in seen:
                    continue
                seen.add(path)
                candidates.append(path)
    return candidates


def _recover_unique_spiral_path(grid: np.ndarray) -> tuple[tuple[int, int], ...]:
    candidates = _enumerate_spiral_candidates(grid != BG)
    assert len(candidates) == 1, f"expected exactly one spiral candidate, found {len(candidates)}"
    return candidates[0]


def _infer_min_period(grid: np.ndarray, path: tuple[tuple[int, int], ...]) -> np.ndarray:
    known: list[tuple[int, int]] = []
    for idx, (r, c) in enumerate(path):
        color = int(grid[r, c])
        if color != BG:
            known.append((idx, color))
    assert known, "expected at least one known color on the spiral"

    n = len(path)
    for period in range(1, n + 1):
        pattern = [-1] * period
        seen = [False] * period
        ok = True
        for idx, color in known:
            residue = idx % period
            if pattern[residue] == -1:
                pattern[residue] = color
            elif pattern[residue] != color:
                ok = False
                break
            seen[residue] = True
        if ok and all(seen):
            return np.array(pattern, dtype=int)

    raise AssertionError("no fully determined periodic coloring fits the visible cells")


def _minimal_period_of_sequence(seq: list[int]) -> int:
    for period in range(1, len(seq) + 1):
        if all(seq[i] == seq[i % period] for i in range(len(seq))):
            return period
    raise AssertionError("unreachable")


def _build_output_from_pattern(
    shape: tuple[int, int], path: list[tuple[int, int]] | tuple[tuple[int, int], ...], pattern: list[int] | np.ndarray
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    period = len(pattern)
    for idx, (r, c) in enumerate(path):
        out[r, c] = int(pattern[idx % period])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore a broken square spiral and refill its erased cells from the smallest periodic color cycle.

    Input structure:
    - A black background contains one one-cell-wide square spiral whose leg lengths follow 1,1,2,2,3,3,...
    - One or more interior contiguous path intervals have been erased to black, so the visible path breaks into several pieces while both spiral endpoints remain visible.
    - Every remaining visible spiral cell is colored, and those colors repeat by a smallest cycle along the center-to-outside spiral order rather than by rows, columns, or legs.

    Rule:
    - Reconstruct the unique full spiral path that matches the visible pieces and the leg-length law.
    - Read the visible colors in spiral traversal order from the center endpoint and infer the smallest fully determined repeating cycle consistent with them.
    - Repaint each erased spiral cell with the color dictated by its traversal index, leaving every already visible pixel unchanged.

    Key insight:
    - The spiral is both a geometric object and a one-dimensional signal carrier: geometry restores the missing coordinates, then traversal order restores the missing colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    path = _recover_unique_spiral_path(g)
    pattern = _infer_min_period(g, path)

    out = g.copy()
    period = len(pattern)
    for idx, (r, c) in enumerate(path):
        if out[r, c] == BG:
            out[r, c] = int(pattern[idx % period])
    return out


def _generate_period_pattern(period: int) -> list[int]:
    colors = random.sample(range(1, 10), random.randint(3, min(5, period)))
    for _ in range(300):
        pattern = [random.choice(colors) for _ in range(period)]
        for color in colors:
            pattern[random.randrange(period)] = color
        if _minimal_period_of_sequence(pattern) != period:
            continue
        if len(set(pattern)) < 3:
            continue
        return pattern
    raise RuntimeError("failed to sample a minimal color period")


def _sequence_misaligned_with_legs(pattern: list[int], n_cells: int, leg_spans: list[tuple[int, int]]) -> bool:
    full = [pattern[i % len(pattern)] for i in range(n_cells)]
    varied_legs = 0
    for start, end in leg_spans:
        if end - start >= 2 and len(set(full[start:end])) > 1:
            varied_legs += 1
    return varied_legs >= 2


def _merged_intervals(hidden: np.ndarray) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    i = 1
    while i < len(hidden) - 1:
        if not hidden[i]:
            i += 1
            continue
        j = i
        while j < len(hidden) - 1 and hidden[j]:
            j += 1
        intervals.append((i, j))
        i = j
    return intervals


def _crosses_turn(intervals: list[tuple[int, int]], leg_spans: list[tuple[int, int]]) -> bool:
    boundaries = {end for _start, end in leg_spans[:-1]}
    for start, end in intervals:
        if any(start < boundary < end for boundary in boundaries):
            return True
    return False


def _choose_hidden_intervals(
    n_cells: int, period: int, leg_spans: list[tuple[int, int]], pattern: list[int]
) -> list[tuple[int, int]] | None:
    boundaries = [end for _start, end in leg_spans[:-1] if 2 <= end <= n_cells - 3]
    if not boundaries:
        return None

    max_hidden = max(period + 3, n_cells // 3)
    gap_cap = min(4, max(2, n_cells // 6))
    for _ in range(400):
        hidden = np.zeros(n_cells, dtype=bool)

        boundary = random.choice(boundaries)
        start = random.randint(max(1, boundary - 2), boundary - 1)
        end = random.randint(boundary + 1, min(n_cells - 1, boundary + 3))
        hidden[start:end] = True

        extra_gaps = random.randint(1, 3)
        for _ in range(extra_gaps):
            gap_start = random.randint(1, n_cells - 2)
            gap_len = random.randint(1, gap_cap)
            gap_end = min(n_cells - 1, gap_start + gap_len)
            hidden[gap_start:gap_end] = True

        hidden[0] = False
        hidden[-1] = False
        intervals = _merged_intervals(hidden)
        if not (2 <= len(intervals) <= 4):
            continue
        if not _crosses_turn(intervals, leg_spans):
            continue

        hidden_count = int(hidden.sum())
        if hidden_count < max(period, 4) or hidden_count > max_hidden:
            continue

        visible_indices = [idx for idx in range(n_cells) if not hidden[idx]]
        if {idx % period for idx in visible_indices} != set(range(period)):
            continue

        hidden_colors = {pattern[idx % period] for idx in range(n_cells) if hidden[idx]}
        if len(hidden_colors) < 2:
            continue

        if max(end - start for start, end in intervals) < max(2, period - 1):
            continue

        return intervals
    return None


def _place_relative_path(
    rel_path: list[tuple[int, int]], next_leg: int
) -> list[tuple[int, int]] | None:
    r0, c0, r1, c1 = _path_bbox(rel_path)
    height = r1 - r0 + 1
    width = c1 - c0 + 1

    next_dir = ORIENTATIONS[0][0]
    for dirs in ORIENTATIONS:
        if _spiral_relative_path(dirs, next_leg - 1) == rel_path:
            next_dir = dirs[(next_leg - 1) % 4]
            break

    for _ in range(200):
        pad_top = random.randint(0, GEN_MAX_PADDING)
        pad_bottom = random.randint(0, GEN_MAX_PADDING)
        pad_left = random.randint(0, GEN_MAX_PADDING)
        pad_right = random.randint(0, GEN_MAX_PADDING)
        if height + pad_top + pad_bottom > GEN_MAX_SIZE or width + pad_left + pad_right > GEN_MAX_SIZE:
            continue

        grid_h = height + pad_top + pad_bottom
        grid_w = width + pad_left + pad_right
        dr = pad_top - r0
        dc = pad_left - c0
        path = _translate_path(rel_path, dr, dc)

        endpoint_r, endpoint_c = path[-1]
        leg_room = _leg_length(next_leg)
        rr = endpoint_r
        cc = endpoint_c
        blocked = False
        for _ in range(leg_room):
            rr += next_dir[0]
            cc += next_dir[1]
            if not _in_bounds((grid_h, grid_w), rr, cc):
                blocked = True
                break
        if not blocked:
            continue
        return path
    return None


def _instance_semantically_valid(
    inp: np.ndarray,
    out: np.ndarray,
    path: list[tuple[int, int]],
    pattern: list[int],
    leg_spans: list[tuple[int, int]],
    intervals: list[tuple[int, int]],
) -> bool:
    candidates = _enumerate_spiral_candidates(inp != BG)
    if len(candidates) != 1 or list(candidates[0]) != path:
        return False

    inferred = _infer_min_period(inp, candidates[0]).tolist()
    if inferred != pattern:
        return False

    visible_indices = [idx for idx, (r, c) in enumerate(path) if inp[r, c] != BG]
    if {idx % len(pattern) for idx in visible_indices} != set(range(len(pattern))):
        return False

    if not _crosses_turn(intervals, leg_spans):
        return False

    if not _sequence_misaligned_with_legs(pattern, len(path), leg_spans):
        return False

    if not np.array_equal(main(inp), out):
        return False
    return True


def _generate_instance() -> np.ndarray | None:
    n_legs = random.randint(GEN_MIN_LEGS, GEN_MAX_LEGS)
    dirs = random.choice(ORIENTATIONS)
    rel_path = _spiral_relative_path(dirs, n_legs)
    path = _place_relative_path(rel_path, n_legs + 1)
    if path is None:
        return None

    shape = (
        max(r for r, _c in path) + 1,
        max(c for _r, c in path) + 1,
    )
    leg_spans = _spiral_leg_spans(n_legs)
    n_cells = len(path)

    period_hi = min(GEN_MAX_PERIOD, n_cells // 3)
    if period_hi < GEN_MIN_PERIOD:
        return None
    period = random.randint(GEN_MIN_PERIOD, period_hi)
    pattern = _generate_period_pattern(period)
    if not _sequence_misaligned_with_legs(pattern, n_cells, leg_spans):
        return None

    intervals = _choose_hidden_intervals(n_cells, period, leg_spans, pattern)
    if intervals is None:
        return None

    out = _build_output_from_pattern(shape, path, pattern)
    inp = out.copy()
    for start, end in intervals:
        for idx in range(start, end):
            r, c = path[idx]
            inp[r, c] = BG

    if int(np.count_nonzero(inp)) == int(np.count_nonzero(out)):
        return None

    if not _instance_semantically_valid(inp, out, path, pattern, leg_spans, intervals):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_instance()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid spiral-period repair input")
