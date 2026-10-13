from __future__ import annotations

from collections import Counter
from itertools import product
import random

import numpy as np


BG = 0
WALL = 5
STRIP_LEN = 4
PAINT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 500


def _most_common(grid: np.ndarray) -> int:
    return Counter(map(int, grid.ravel())).most_common(1)[0][0]


def _rev(pattern: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(reversed(pattern))


def _is_two_color_strip(pattern: tuple[int, ...], bg: int, wall: int) -> bool:
    vals = set(pattern)
    return len(pattern) == STRIP_LEN and len(vals) == 2 and bg not in vals and wall not in vals


def _find_wall(grid: np.ndarray, bg: int) -> tuple[int, int]:
    h, w = grid.shape
    candidates: list[tuple[int, int]] = []
    for c0 in range(w - STRIP_LEN + 1):
        band = grid[:, c0 : c0 + STRIP_LEN]
        if not np.all(band != bg):
            continue

        vals, counts = np.unique(band, return_counts=True)
        wall = int(vals[int(np.argmax(counts))])
        row_ok = True
        for r in range(h):
            row = tuple(map(int, band[r]))
            if all(v == wall for v in row):
                continue
            if not _is_two_color_strip(row, bg, wall):
                row_ok = False
                break
        if row_ok:
            candidates.append((c0, wall))

    assert len(candidates) == 1, "expected one four-column vertical wall band"
    return candidates[0]


def _find_keys(grid: np.ndarray, bg: int, wall: int, wall_c0: int) -> list[tuple[int, tuple[int, ...]]]:
    keys: list[tuple[int, tuple[int, ...]]] = []
    for r in range(grid.shape[0]):
        strip = tuple(map(int, grid[r, :STRIP_LEN]))
        if _is_two_color_strip(strip, bg, wall):
            assert np.all(grid[r, STRIP_LEN:wall_c0] == bg), "key must have clear space before the wall"
            keys.append((r, strip))
    assert keys, "expected at least one left-edge key strip"
    return keys


def _find_locks(grid: np.ndarray, bg: int, wall: int, wall_c0: int) -> list[tuple[int, tuple[int, ...]]]:
    locks: list[tuple[int, tuple[int, ...]]] = []
    for r in range(grid.shape[0]):
        segment = tuple(map(int, grid[r, wall_c0 : wall_c0 + STRIP_LEN]))
        if all(v == wall for v in segment):
            continue
        assert _is_two_color_strip(segment, bg, wall), "lock segment must be a two-color strip"
        locks.append((r, segment))
    assert locks, "expected at least one colored lock segment in the wall"
    return locks


def _analyze(grid: np.ndarray) -> tuple[int, int, int, list[tuple[int, tuple[int, ...]]], list[tuple[int, tuple[int, ...]]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h <= 30 and w <= 30, "ARC grids must fit within 30 by 30"
    assert w >= 2 * STRIP_LEN + 5, "expected room for left keys, wall, and right exits"

    bg = _most_common(g)
    wall_c0, wall = _find_wall(g, bg)
    assert wall_c0 >= STRIP_LEN, "wall must have room for stopped keys on its left"
    assert wall_c0 + STRIP_LEN <= w - STRIP_LEN, "wall must have room for passed keys on its right"

    keys = _find_keys(g, bg, wall, wall_c0)
    locks = _find_locks(g, bg, wall, wall_c0)
    color_pairs = {frozenset(pattern) for _r, pattern in keys + locks}
    assert len(color_pairs) == 1, "all keys and locks should use the same two colors"
    assert len({pattern for _r, pattern in locks}) == len(locks), "lock patterns must be unique"
    assert len({pattern for _r, pattern in keys}) == len(keys), "key patterns must be unique"
    return bg, wall, wall_c0, keys, locks


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair mirror-read keys with wall locks and move each key to the side allowed by that pairing.

    Input structure:
    - A dominant background surrounds one vertical four-column wall band whose base color is grey.
    - Some wall rows contain full-width two-color lock segments instead of grey wall cells.
    - Several two-color key strips of the same length and color pair sit on the left edge; their rows do not name the locks.

    Rule:
    - Read every key strip and find the unique lock segment equal to that key read right-to-left.
    - A matched key is erased from its old row, the matched lock row is restored to wall grey, and the key is translated to the right edge on that lock row.
    - A key with no mirror-matching lock is erased from the left edge and translated only until it touches the wall from the left.
    - Lock segments not opened by a mirror-matched key remain unchanged.

    Key insight:
    - The row destination is inferred from a mirrored pattern match, not from the key's starting row or from direct left-to-right equality.
    """
    g = np.asarray(grid, dtype=int)
    bg, wall, wall_c0, keys, locks = _analyze(g)
    h, w = g.shape

    out = g.copy()
    for key_r, _pattern in keys:
        out[key_r, :STRIP_LEN] = bg

    used_lock_rows: set[int] = set()
    right_c0 = w - STRIP_LEN
    stop_c0 = wall_c0 - STRIP_LEN

    for key_r, key_pattern in keys:
        matches = [lock_r for lock_r, lock_pattern in locks if lock_pattern == _rev(key_pattern)]
        assert len(matches) <= 1, "each key may have at most one mirror-matching lock"

        if matches:
            lock_r = matches[0]
            assert lock_r not in used_lock_rows, "a lock segment may be opened by only one key"
            used_lock_rows.add(lock_r)
            assert np.all(out[lock_r, right_c0:w] == bg), "right-edge landing cells must be empty"
            out[lock_r, wall_c0 : wall_c0 + STRIP_LEN] = wall
            out[lock_r, right_c0:w] = key_pattern
        else:
            assert np.all(out[key_r, stop_c0:wall_c0] == bg), "stopped key path must end in empty cells"
            out[key_r, stop_c0:wall_c0] = key_pattern

    return out


def _all_patterns(a: int, b: int) -> list[tuple[int, ...]]:
    patterns: list[tuple[int, ...]] = []
    for values in product((int(a), int(b)), repeat=STRIP_LEN):
        pattern = tuple(map(int, values))
        if len(set(pattern)) == 2 and pattern != _rev(pattern):
            patterns.append(pattern)
    random.shuffle(patterns)
    return patterns


def _sample_patterns(colors: tuple[int, int]) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    patterns = _all_patterns(*colors)
    first = random.choice(patterns)
    second = _rev(first)

    key_patterns = [first, second]
    lock_patterns = [second, first]
    matched_count = random.randint(2, 4)

    for _ in range(matched_count - 2):
        candidates = [
            p
            for p in patterns
            if p not in key_patterns and _rev(p) not in lock_patterns
        ]
        if not candidates:
            raise ValueError("not enough unique mirror pairs")
        key = random.choice(candidates)
        key_patterns.append(key)
        lock_patterns.append(_rev(key))

    unmatched_candidates = [
        p
        for p in patterns
        if p not in key_patterns
        and p not in lock_patterns
        and _rev(p) not in key_patterns
        and _rev(p) not in lock_patterns
    ]
    if not unmatched_candidates:
        raise ValueError("not enough patterns for an unmatched direct-equality trap")
    unmatched = random.choice(unmatched_candidates)
    key_patterns.append(unmatched)
    lock_patterns.append(unmatched)

    random.shuffle(key_patterns)
    random.shuffle(lock_patterns)
    return key_patterns, lock_patterns


def _sample_spaced_rows(h: int, count: int) -> list[int]:
    rows = list(range(1, h - 1))
    for _ in range(500):
        sample = sorted(random.sample(rows, count))
        if all(b - a >= 2 for a, b in zip(sample, sample[1:])):
            return sample
    raise ValueError("could not place active rows with clear spacing")


def _build_grid(
    h: int,
    w: int,
    wall_c0: int,
    keys: list[tuple[int, tuple[int, ...]]],
    locks: list[tuple[int, tuple[int, ...]]],
) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    grid[:, wall_c0 : wall_c0 + STRIP_LEN] = WALL
    for r, pattern in locks:
        grid[r, wall_c0 : wall_c0 + STRIP_LEN] = pattern
    for r, pattern in keys:
        grid[r, :STRIP_LEN] = pattern
    return grid


def _expected_from_plan(
    grid: np.ndarray,
    wall_c0: int,
    keys: list[tuple[int, tuple[int, ...]]],
    locks: list[tuple[int, tuple[int, ...]]],
    *,
    mirrored: bool,
) -> np.ndarray:
    out = grid.copy()
    h, w = out.shape
    right_c0 = w - STRIP_LEN
    stop_c0 = wall_c0 - STRIP_LEN
    for key_r, _pattern in keys:
        out[key_r, :STRIP_LEN] = BG

    used: set[int] = set()
    for key_r, key_pattern in keys:
        wanted = _rev(key_pattern) if mirrored else key_pattern
        matches = [lock_r for lock_r, lock_pattern in locks if lock_pattern == wanted]
        if len(matches) == 1 and matches[0] not in used:
            lock_r = matches[0]
            used.add(lock_r)
            out[lock_r, wall_c0 : wall_c0 + STRIP_LEN] = WALL
            out[lock_r, right_c0:w] = key_pattern
        else:
            out[key_r, stop_c0:wall_c0] = key_pattern
    assert h <= 30 and w <= 30
    return out


def _semantic_checks(
    grid: np.ndarray,
    wall_c0: int,
    keys: list[tuple[int, tuple[int, ...]]],
    locks: list[tuple[int, tuple[int, ...]]],
) -> bool:
    lock_patterns = [pattern for _r, pattern in locks]
    key_patterns = [pattern for _r, pattern in keys]

    mirror_matches = {
        key: [lock for lock in lock_patterns if lock == _rev(key)]
        for key in key_patterns
    }
    if not any(len(matches) == 0 for matches in mirror_matches.values()):
        return False
    if not all(len(matches) <= 1 for matches in mirror_matches.values()):
        return False

    direct_trap = any(
        key == lock and _rev(key) != lock
        for key in key_patterns
        for lock in lock_patterns
    )
    if not direct_trap:
        return False

    unopened_locks = [
        lock
        for lock in lock_patterns
        if all(_rev(key) != lock for key in key_patterns)
    ]
    if not unopened_locks:
        return False

    expected = _expected_from_plan(grid, wall_c0, keys, locks, mirrored=True)
    direct_wrong = _expected_from_plan(grid, wall_c0, keys, locks, mirrored=False)
    if np.array_equal(expected, direct_wrong):
        return False

    try:
        solved = main(grid)
    except Exception:
        return False
    return bool(np.array_equal(solved, expected))


def _gen_attempt() -> np.ndarray | None:
    colors = tuple(random.sample(PAINT_COLORS, 2))
    try:
        key_patterns, lock_patterns = _sample_patterns(colors)
    except ValueError:
        return None

    active_rows = len(key_patterns) + len(lock_patterns)
    min_h = max(15, 2 * active_rows + 3)
    if min_h > 30:
        return None
    h = random.randint(min_h, 30)
    w = random.randint(18, 30)
    wall_c0 = random.randint(8, w - 9)

    try:
        rows = _sample_spaced_rows(h, active_rows)
    except ValueError:
        return None
    random.shuffle(rows)

    key_rows = rows[: len(key_patterns)]
    lock_rows = rows[len(key_patterns) :]
    keys = list(zip(key_rows, key_patterns))
    locks = list(zip(lock_rows, lock_patterns))
    random.shuffle(keys)
    random.shuffle(locks)

    grid = _build_grid(h, w, wall_c0, keys, locks)
    if not _semantic_checks(grid, wall_c0, keys, locks):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-key gate input")
