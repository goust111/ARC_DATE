from __future__ import annotations

from functools import lru_cache

import numpy as np


BACKGROUND = 0
MASK = 5
OUTPUT_SHAPE = (3, 3)
OUTPUT_CELL_COUNT = OUTPUT_SHAPE[0] * OUTPUT_SHAPE[1]
SNAKE_POSITIONS = (
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 2),
    (1, 1),
    (1, 0),
    (2, 0),
    (2, 1),
    (2, 2),
)

GEN_SIZES = (9, 11)
GEN_FOREGROUND_COLORS = np.array([1, 2, 3, 4], dtype=int)
GEN_COUNTS = (5, 3, 1)
GEN_ATTEMPTS = 400
_RNG = np.random.default_rng()


def _symmetry_orbit(n: int, r: int, c: int) -> tuple[tuple[int, int], ...]:
    last = int(n) - 1
    return tuple(
        sorted(
            {
                (r, c),
                (c, r),
                (r, last - c),
                (last - r, c),
                (last - r, last - c),
                (last - c, last - r),
                (c, last - r),
                (last - c, r),
            }
        )
    )


@lru_cache(maxsize=None)
def _all_orbits(n: int) -> tuple[tuple[tuple[int, int], ...], ...]:
    seen: set[tuple[tuple[int, int], ...]] = set()
    orbits: list[tuple[tuple[int, int], ...]] = []
    for r in range(int(n)):
        for c in range(int(n)):
            orbit = _symmetry_orbit(n, r, c)
            if orbit not in seen:
                seen.add(orbit)
                orbits.append(orbit)
    return tuple(orbits)


def _restore_masked(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    n, w = g.shape
    assert n == w and n in GEN_SIZES, "expected a 9x9 or 11x11 square grid"

    out = g.copy()
    for r, c in np.argwhere(g == MASK):
        orbit = _symmetry_orbit(n, int(r), int(c))
        visible = {
            int(g[rr, cc])
            for rr, cc in orbit
            if int(g[rr, cc]) not in (BACKGROUND, MASK)
        }
        assert len(visible) == 1, "each masked cell must have one visible orbit color"
        out[int(r), int(c)] = visible.pop()

    assert not np.any(out == MASK), "all mask pixels must be restored"
    return out


def _foreground_counts(grid: np.ndarray) -> dict[int, int]:
    colors, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    out = {
        int(color): int(count)
        for color, count in zip(colors.tolist(), counts.tolist())
        if int(color) not in (BACKGROUND, MASK)
    }
    assert len(out) == 3, "expected exactly three foreground colors"
    return out


def _summary_from_counts(counts: dict[int, int]) -> np.ndarray:
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ordered_counts = [count for _color, count in ranked]
    assert ordered_counts[0] > ordered_counts[1] > ordered_counts[2], "foreground counts must be distinct"
    assert sum(ordered_counts) == OUTPUT_CELL_COUNT, "the restored foreground must occupy exactly nine cells"

    out = np.zeros(OUTPUT_SHAPE, dtype=int)
    slot = 0
    for color, count in ranked:
        for _ in range(int(count)):
            r, c = SNAKE_POSITIONS[slot]
            out[r, c] = int(color)
            slot += 1
    assert slot == OUTPUT_CELL_COUNT
    return out


def _visible_summary(grid: np.ndarray) -> np.ndarray:
    visible_counts = _foreground_counts(grid)
    out = np.zeros(OUTPUT_SHAPE, dtype=int)
    ranked = sorted(visible_counts.items(), key=lambda item: (-item[1], item[0]))
    slot = 0
    for color, count in ranked:
        for _ in range(int(count)):
            r, c = SNAKE_POSITIONS[slot]
            out[r, c] = int(color)
            slot += 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Recover gray masked cells from their eight-way orbit, then summarize restored color counts in a 3x3 snake grid.

    Input structure:
    - An odd square grid of size 9 or 11 with black background.
    - Exactly three non-black foreground colors from {1,2,3,4}; gray cells are masked foreground pixels.
    - For every gray cell, its full dihedral orbit in the square contains visible cells of exactly one non-black, non-gray color.

    Rule:
    - For each gray cell, enumerate its deduplicated eight-way square-symmetry orbit (rotations plus reflections) and find the unique visible foreground color in that orbit.
    - Replace the gray cell with that color, then count the restored pixels of the three foreground colors.
    - Sort the colors by descending restored count and write them into a 3x3 black output in snake order: row 0 left-to-right, row 1 right-to-left, row 2 left-to-right.

    Key insight:
    - The ordering depends on hidden pixels, so counting only the visible cells gives the wrong summary; the mask must be resolved first from the square-symmetry orbit.
    """
    restored = _restore_masked(grid)
    counts = _foreground_counts(restored)
    return _summary_from_counts(counts)


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _positions_are_clear(groups: list[set[tuple[int, int]]]) -> bool:
    all_positions = [pos for group in groups for pos in sorted(group)]
    for i, pos_a in enumerate(all_positions):
        for pos_b in all_positions[i + 1 :]:
            if _manhattan(pos_a, pos_b) <= 1:
                return False
    return True


def _pick_distinct_size4_orbits(n: int) -> tuple[tuple[tuple[int, int], ...], ...] | None:
    size4_orbits = [orbit for orbit in _all_orbits(n) if len(orbit) == 4]
    order = _RNG.permutation(len(size4_orbits))
    center = (n // 2, n // 2)
    for i in range(len(order)):
        major_orbit = size4_orbits[int(order[i])]
        major_positions = set(major_orbit) | {center}
        for j in range(i + 1, len(order)):
            mid_orbit = size4_orbits[int(order[j])]
            mid_active = set(_RNG.choice(len(mid_orbit), size=3, replace=False).tolist())
            mid_positions = {mid_orbit[idx] for idx in mid_active}
            for k in range(j + 1, len(order)):
                minor_orbit = size4_orbits[int(order[k])]
                minor_idx = int(_RNG.integers(0, len(minor_orbit)))
                minor_positions = {minor_orbit[minor_idx]}
                if _positions_are_clear([major_positions, mid_positions, minor_positions]):
                    return major_orbit, tuple(sorted(mid_positions)), tuple(sorted(minor_positions))
    return None


def _build_candidate() -> np.ndarray | None:
    n = int(_RNG.choice(GEN_SIZES))
    picked = _pick_distinct_size4_orbits(n)
    if picked is None:
        return None

    major_orbit, mid_positions, minor_positions = picked
    center = (n // 2, n // 2)
    colors = _RNG.choice(GEN_FOREGROUND_COLORS, size=3, replace=False)
    colors = colors[_RNG.permutation(3)]
    major_color, mid_color, minor_color = [int(v) for v in colors.tolist()]

    restored = np.zeros((n, n), dtype=int)
    restored[center] = major_color
    for r, c in major_orbit:
        restored[int(r), int(c)] = major_color
    for r, c in mid_positions:
        restored[int(r), int(c)] = mid_color
    for r, c in minor_positions:
        restored[int(r), int(c)] = minor_color

    visible_major_idx = int(_RNG.integers(0, len(major_orbit)))
    visible_major = major_orbit[visible_major_idx]
    inp = restored.copy()
    for r, c in major_orbit:
        if (r, c) != visible_major:
            inp[int(r), int(c)] = MASK

    return inp


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        restored = _restore_masked(grid)
        restored_counts = _foreground_counts(restored)
        visible_counts = _foreground_counts(grid)
    except AssertionError:
        return False

    if sorted(restored_counts.values(), reverse=True) != list(GEN_COUNTS):
        return False
    if restored_counts == visible_counts:
        return False

    restored_rank = [color for color, _count in sorted(restored_counts.items(), key=lambda item: (-item[1], item[0]))]
    visible_rank = [color for color, _count in sorted(visible_counts.items(), key=lambda item: (-item[1], item[0]))]
    if restored_rank[0] == visible_rank[0]:
        return False

    mask_cells = np.argwhere(grid == MASK)
    if len(mask_cells) != 3:
        return False
    for r, c in mask_cells:
        orbit = _symmetry_orbit(grid.shape[0], int(r), int(c))
        visible = {
            int(grid[rr, cc])
            for rr, cc in orbit
            if int(grid[rr, cc]) not in (BACKGROUND, MASK)
        }
        if len(visible) != 1:
            return False

    if np.array_equal(main(grid), _visible_summary(grid)):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_candidate()
        if grid is None:
            continue
        if not _semantic_checks(grid):
            continue
        return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
