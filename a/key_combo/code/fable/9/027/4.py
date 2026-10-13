from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


COLORS = tuple(range(1, 10))
PERIOD_PENALTY = 5
GEN_ATTEMPTS = 400


def _on_diagonal(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a != b and abs(a[0] - b[0]) == abs(a[1] - b[1])


def _diag_segment(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    assert _on_diagonal(a, b)
    dr = 1 if b[0] > a[0] else -1
    dc = 1 if b[1] > a[1] else -1
    steps = abs(b[0] - a[0])
    return [(a[0] + i * dr, a[1] + i * dc) for i in range(steps + 1)]


def _periodic_grid(tile: np.ndarray, height: int, width: int) -> np.ndarray:
    rows = np.arange(height)[:, None] % tile.shape[0]
    cols = np.arange(width)[None, :] % tile.shape[1]
    return tile[rows, cols].astype(int, copy=True)


def _is_minimal_tile(tile: np.ndarray) -> bool:
    pr, pc = tile.shape
    for rr in range(1, pr):
        if pr % rr == 0 and np.array_equal(tile, tile[np.arange(pr)[:, None] % rr, np.arange(pc)[None, :]]):
            return False
    for cc in range(1, pc):
        if pc % cc == 0 and np.array_equal(tile, tile[np.arange(pr)[:, None], np.arange(pc)[None, :] % cc]):
            return False
    return True


def _fit_period(
    grid: np.ndarray, row_period: int, col_period: int
) -> tuple[np.ndarray, int] | None:
    h, w = grid.shape
    tile = np.zeros((row_period, col_period), dtype=int)
    mismatches = 0

    for rr in range(row_period):
        for cc in range(col_period):
            cells = grid[rr:h:row_period, cc:w:col_period].ravel()
            counts = np.bincount(cells, minlength=10)
            best = int(counts.max())
            winners = np.flatnonzero(counts == best)
            if len(winners) != 1:
                return None
            tile[rr, cc] = int(winners[0])
            mismatches += int(cells.size - best)

    pred = _periodic_grid(tile, h, w)
    return pred, mismatches


def _candidate_ranges(height: int, width: int) -> tuple[range, range]:
    row_max = max(1, height // 2)
    col_max = max(1, width // 2)
    return range(1, row_max + 1), range(1, col_max + 1)


def _recover_periodic_base(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[tuple[tuple[int, int, int, int, int], np.ndarray]] = []

    for row_period in _candidate_ranges(h, w)[0]:
        for col_period in _candidate_ranges(h, w)[1]:
            fit = _fit_period(g, row_period, col_period)
            if fit is None:
                continue
            pred, mismatches = fit
            area = row_period * col_period
            cost = area + PERIOD_PENALTY * mismatches
            key = (cost, mismatches, area, row_period, col_period)
            candidates.append((key, pred))

    if not candidates:
        for row_period in range(1, h + 1):
            for col_period in range(1, w + 1):
                fit = _fit_period(g, row_period, col_period)
                if fit is None:
                    continue
                pred, mismatches = fit
                area = row_period * col_period
                cost = area + PERIOD_PENALTY * mismatches
                key = (cost, mismatches, area, row_period, col_period)
                candidates.append((key, pred))

    assert candidates, "no periodic candidate recovered"
    candidates.sort(key=lambda item: item[0])
    best_key, best_pred = candidates[0]
    for key, pred in candidates[1:]:
        if key != best_key:
            break
        assert np.array_equal(pred, best_pred), "best periodic reconstruction must be unique"
    return best_pred, (best_key[3], best_key[4])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the hidden translation pattern, connect diagonal same-color residues, and erase the other defects.

    Input structure:
    - A small translational tile covers the whole board; every cell belongs to that repeated pattern.
    - A few cells deviate from the pattern prediction.
    - Some deviation colors appear exactly twice and those two cells lie on one 45-degree diagonal; other deviation colors appear only once.

    Rule:
    - Reconstruct the periodic background by majority agreement over a minimal repeated row/column period.
    - Mark every cell whose visible color disagrees with that periodic prediction as a deviation.
    - For each color that has exactly one diagonal-aligned pair of deviations, paint the entire diagonal segment between the two endpoints in that color.
    - Any deviation not used as such a pair is repaired back to the periodic prediction.

    Key insight:
    - The visible endpoints are only residues of an otherwise hidden periodic canvas: first infer that canvas, then the surviving off-pattern pixels split into connectable same-color pairs versus singleton decoys that must be erased.
    """
    g = np.asarray(grid, dtype=int)
    base, _period = _recover_periodic_base(g)
    deviations = [(int(r), int(c)) for r, c in np.argwhere(g != base)]

    by_color: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for r, c in deviations:
        by_color[int(g[r, c])].append((r, c))

    out = base.copy()
    for color, coords in by_color.items():
        if len(coords) == 2 and _on_diagonal(coords[0], coords[1]):
            for r, c in _diag_segment(coords[0], coords[1]):
                out[r, c] = int(color)

    return out


def _sample_tile(row_period: int, col_period: int, base_colors: list[int]) -> np.ndarray | None:
    tile = np.random.choice(base_colors, size=(row_period, col_period))
    if len(set(map(int, tile.ravel()))) < min(3, len(base_colors)):
        return None
    if any(len(set(map(int, row))) == 1 for row in tile):
        return None
    if any(len(set(map(int, col))) == 1 for col in tile.T):
        return None
    if not _is_minimal_tile(tile):
        return None
    return tile.astype(int)


def _pick_pair_segment(
    base: np.ndarray, reserved: set[tuple[int, int]], color: int
) -> tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]]] | None:
    h, w = base.shape
    for _ in range(200):
        direction = random.choice((1, -1))
        max_len = min(h, w, 9)
        length = random.randint(4, max_len)
        if direction == 1:
            r0 = random.randint(0, h - length)
            c0 = random.randint(0, w - length)
            segment = [(r0 + i, c0 + i) for i in range(length)]
        else:
            r0 = random.randint(0, h - length)
            c0 = random.randint(length - 1, w - 1)
            segment = [(r0 + i, c0 - i) for i in range(length)]
        if any(p in reserved for p in segment):
            continue
        if any(int(base[p]) == int(color) for p in segment):
            continue
        return segment[0], segment[-1], segment
    return None


def _pick_singleton_positions(
    shape: tuple[int, int],
    reserved: set[tuple[int, int]],
    count: int,
) -> list[tuple[int, int]] | None:
    h, w = shape
    if count < 2:
        return None

    for _ in range(200):
        first = (random.randrange(h), random.randrange(w))
        if first in reserved:
            continue
        choices = [
            (r, c)
            for r in range(h)
            for c in range(w)
            if (r, c) not in reserved and (r, c) != first and _on_diagonal(first, (r, c))
        ]
        choices = [p for p in choices if abs(p[0] - first[0]) >= 3]
        if not choices:
            continue
        second = random.choice(choices)
        positions = [first, second]
        used = set(positions) | set(reserved)
        while len(positions) < count:
            candidate = (random.randrange(h), random.randrange(w))
            if candidate in used:
                continue
            positions.append(candidate)
            used.add(candidate)
        return positions
    return None


def _instance_signature(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    return grid.shape, tuple(map(int, grid.ravel()))


def _validate_instance(
    inp: np.ndarray,
    base: np.ndarray,
    expected: np.ndarray,
    row_period: int,
    col_period: int,
    pair_info: list[tuple[int, tuple[int, int], tuple[int, int], list[tuple[int, int]]]],
    singletons: list[tuple[int, tuple[int, int]]],
) -> bool:
    produced = main(inp)
    if not np.array_equal(produced, expected):
        return False

    recovered, periods = _recover_periodic_base(inp)
    if periods != (row_period, col_period):
        return False
    if not np.array_equal(recovered, base):
        return False

    deviations = {(int(r), int(c)) for r, c in np.argwhere(inp != base)}
    required = {pos for _color, a, b, _segment in pair_info for pos in (a, b)}
    required |= {pos for _color, pos in singletons}
    if deviations != required:
        return False

    if np.array_equal(expected, base):
        return False

    if len(singletons) < 2 or not _on_diagonal(singletons[0][1], singletons[1][1]):
        return False

    for color, _a, _b, segment in pair_info:
        interior = segment[1:-1]
        if not interior:
            return False
        if any(int(base[p]) == int(color) for p in segment):
            return False
        if not any(int(expected[p]) != int(base[p]) for p in interior):
            return False

    for color, pos in singletons:
        if int(inp[pos]) == int(base[pos]) or int(inp[pos]) == int(expected[pos]):
            return False
        if int(expected[pos]) != int(base[pos]):
            return False
        if int(inp[pos]) != int(color):
            return False

    repair_all = base.copy()
    if np.array_equal(repair_all, expected):
        return False

    return _instance_signature(inp) != _instance_signature(expected)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        row_period = random.randint(2, 4)
        col_period = random.randint(2, 4)
        height = random.randint(max(10, row_period * 3), min(22, row_period * 6))
        width = random.randint(max(10, col_period * 3), min(22, col_period * 6))

        pair_count = random.randint(1, 2)
        singleton_count = random.randint(2, 3)
        needed_defect_colors = pair_count + singleton_count
        max_base_colors = min(4, len(COLORS) - needed_defect_colors)
        if max_base_colors < 3:
            continue
        base_color_count = random.randint(3, max_base_colors)
        palette = random.sample(COLORS, base_color_count + needed_defect_colors)
        base_colors = palette[:base_color_count]
        defect_colors = palette[base_color_count:]

        tile = _sample_tile(row_period, col_period, base_colors)
        if tile is None:
            continue
        base = _periodic_grid(tile, height, width)

        reserved: set[tuple[int, int]] = set()
        pair_info: list[tuple[int, tuple[int, int], tuple[int, int], list[tuple[int, int]]]] = []
        success = True
        for color in defect_colors[:pair_count]:
            picked = _pick_pair_segment(base, reserved, color)
            if picked is None:
                success = False
                break
            a, b, segment = picked
            pair_info.append((int(color), a, b, segment))
            reserved.update(segment)
        if not success:
            continue

        singleton_positions = _pick_singleton_positions(base.shape, reserved, singleton_count)
        if singleton_positions is None:
            continue
        if any(int(base[pos]) == int(color) for pos, color in zip(singleton_positions, defect_colors[pair_count:])):
            continue
        singletons = [
            (int(color), pos)
            for color, pos in zip(defect_colors[pair_count:], singleton_positions)
        ]

        inp = base.copy()
        for color, a, b, _segment in pair_info:
            inp[a] = int(color)
            inp[b] = int(color)
        for color, pos in singletons:
            inp[pos] = int(color)

        expected = base.copy()
        for color, _a, _b, segment in pair_info:
            for pos in segment:
                expected[pos] = int(color)

        if not _validate_instance(
            inp, base, expected, row_period, col_period, pair_info, singletons
        ):
            continue
        return inp

    raise RuntimeError("failed to generate a valid input")
