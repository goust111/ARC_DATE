import random
from collections import Counter

import numpy as np


BLACK = 0
PALETTE = tuple(range(1, 10))
GRID_SIZES = (6, 9)
GEN_RETRY_LIMIT = 300


def _d4_orbit(r: int, c: int, n: int) -> tuple[tuple[int, int], ...]:
    last = n - 1
    return tuple(
        sorted(
            {
                (r, c),
                (c, last - r),
                (last - r, last - c),
                (last - c, r),
                (r, last - c),
                (last - r, c),
                (c, r),
                (last - c, last - r),
            }
        )
    )


def _orbit_rep(r: int, c: int, n: int) -> tuple[int, int]:
    return _d4_orbit(r, c, n)[0]


def _iter_orbits(n: int):
    seen: set[tuple[int, int]] = set()
    for r in range(n):
        for c in range(n):
            if (r, c) in seen:
                continue
            orbit = _d4_orbit(r, c, n)
            seen.update(orbit)
            yield orbit


def _repair_and_find_defects(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    n = int(grid.shape[0])
    repaired = grid.copy()
    defects: list[tuple[int, int]] = []
    for orbit in _iter_orbits(n):
        orbit_colors = [int(grid[r, c]) for r, c in orbit]
        counts = Counter(orbit_colors)
        if len(counts) == 1:
            continue
        major_color, major_count = counts.most_common(1)[0]
        assert len(counts) == 2 and major_count > 1, "expected one majority color and one outlier color"
        assert major_count == len(orbit) - 1, "expected exactly one defect in a broken orbit"
        defect = next((r, c) for r, c in orbit if counts[int(grid[r, c])] == 1)
        repaired[defect] = major_color
        defects.append(defect)
    assert defects, "expected at least one defect"
    return repaired, defects


def _macro_positions(coords: list[tuple[int, int]], n: int) -> list[tuple[int, int]]:
    block = n // 3
    positions = [(r // block, c // block) for r, c in coords]
    assert len(set(positions)) == len(positions), "defects must occupy distinct macro-cells"
    return positions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair each broken D4-symmetry orbit, then use the original defect macro-cells as a tiling layout for the repaired pattern.

    Input structure:
    - The input is a square grid whose side length is divisible by 3.
    - Apart from a few altered pixels, the whole pattern is consistent under all
      eight symmetries of the square.
    - No two altered pixels lie in the same cell of the implied 3x3 macro-grid.

    Rule:
    - Partition the square into D4 symmetry orbits formed by rotations and
      reflections.
    - In every broken orbit, the lone disagreeing pixel is the defect; replace it
      with the orbit's majority color to recover the intended symmetric pattern.
    - Create a black output grid with side length three times the input side.
      Split it into a 3x3 grid of blocks the size of the repaired pattern, and
      paste the full repaired pattern into exactly those blocks whose input
      macro-cells contained a defect.

    Key insight:
    - The asymmetric pixels matter twice: locally they are repairs to erase, and
      globally their coarse 3x3 locations are the placement instructions for the
      repaired tile.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    n, m = map(int, g.shape)
    assert n == m and n % 3 == 0, "expected a square grid with side divisible by 3"

    repaired, defects = _repair_and_find_defects(g)
    positions = _macro_positions(defects, n)

    out = np.zeros((3 * n, 3 * n), dtype=int)
    for mr, mc in positions:
        out[mr * n : (mr + 1) * n, mc * n : (mc + 1) * n] = repaired
    return out


def _build_symmetric_pattern(n: int) -> np.ndarray:
    palette = random.sample(PALETTE, random.randint(3, 5))
    mode = random.randrange(4)
    coeffs = random.randint(1, 3), random.randint(1, 3), random.randint(0, len(palette) - 1)
    grid = np.zeros((n, n), dtype=int)

    for orbit in _iter_orbits(n):
        rr, cc = orbit[0]
        border = min(rr, cc, n - 1 - rr, n - 1 - cc)
        diag = cc - rr
        center_dist = max(abs(2 * rr - (n - 1)), abs(2 * cc - (n - 1)))
        a, b, shift = coeffs
        if mode == 0:
            idx = (a * border + b * diag + shift) % len(palette)
        elif mode == 1:
            idx = (a * diag + b * (center_dist // 2) + shift) % len(palette)
        elif mode == 2:
            idx = (a * rr + b * cc + border + shift) % len(palette)
        else:
            idx = (a * (diag % 2) + b * border + (center_dist // 2) + shift) % len(palette)
        color = palette[idx]
        for r, c in orbit:
            grid[r, c] = color

    if len(set(map(int, np.unique(grid)))) < 3:
        accent = random.choice([c for c in PALETTE if c not in set(map(int, np.unique(grid)))])
        for orbit in list(_iter_orbits(n))[::2]:
            for r, c in orbit:
                grid[r, c] = accent
            break
    return grid


def _candidate_cells(grid: np.ndarray) -> dict[tuple[int, int], list[tuple[tuple[int, int], tuple[int, int]]]]:
    n = int(grid.shape[0])
    block = n // 3
    candidates: dict[tuple[int, int], list[tuple[tuple[int, int], tuple[int, int]]]] = {}
    for r in range(n):
        for c in range(n):
            orbit = _d4_orbit(r, c, n)
            if len(orbit) < 4:
                continue
            macro = (r // block, c // block)
            rep = orbit[0]
            candidates.setdefault(macro, []).append(((r, c), rep))
    return candidates


def _choose_defects(grid: np.ndarray) -> list[tuple[int, int]]:
    candidates = _candidate_cells(grid)
    usable_macros = [macro for macro, items in candidates.items() if items]
    all_reps = {rep for items in candidates.values() for _, rep in items}
    max_defects = min(5, len(usable_macros), len(all_reps))
    assert max_defects >= 1, "expected at least one defect candidate"
    min_defects = 1 if max_defects == 1 else 2
    defect_count = random.randint(min_defects, max_defects)

    for _ in range(200):
        chosen: list[tuple[int, int]] = []
        used_reps: set[tuple[int, int]] = set()
        macros = random.sample(usable_macros, defect_count)
        ok = True
        for macro in macros:
            options = candidates[macro][:]
            random.shuffle(options)
            pick = next((((r, c), rep) for (r, c), rep in options if rep not in used_reps), None)
            if pick is None:
                ok = False
                break
            (r, c), rep = pick
            chosen.append((r, c))
            used_reps.add(rep)
        if ok:
            return chosen
    raise RuntimeError("failed to choose valid defect positions")


def _expected_output(repaired: np.ndarray, defects: list[tuple[int, int]]) -> np.ndarray:
    n = int(repaired.shape[0])
    out = np.zeros((3 * n, 3 * n), dtype=int)
    for mr, mc in _macro_positions(defects, n):
        out[mr * n : (mr + 1) * n, mc * n : (mc + 1) * n] = repaired
    return out


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(GRID_SIZES)
    repaired = _build_symmetric_pattern(n)
    defects = _choose_defects(repaired)

    inp = repaired.copy()
    for r, c in defects:
        base_color = int(inp[r, c])
        inp[r, c] = random.choice([color for color in PALETTE if color != base_color])

    expected = _expected_output(repaired, defects)
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
