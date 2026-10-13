from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
ARC_COLORS = tuple(range(10))
FOREGROUND_COLORS = tuple(range(1, 10))
GEN_RETRIES = 500


def _mirror_orbit(n: int, r: int, c: int) -> tuple[tuple[int, int], ...]:
    """Return the four cells related by horizontal and vertical reflection."""
    return tuple(
        sorted(
            {
                (int(r), int(c)),
                (n - 1 - int(r), int(c)),
                (int(r), n - 1 - int(c)),
                (n - 1 - int(r), n - 1 - int(c)),
            }
        )
    )


def _orbits(n: int) -> list[tuple[tuple[int, int], ...]]:
    seen: set[tuple[tuple[int, int], ...]] = set()
    out: list[tuple[tuple[int, int], ...]] = []
    for r in range(n):
        for c in range(n):
            orbit = _mirror_orbit(n, r, c)
            if orbit in seen:
                continue
            assert len(orbit) == 4, "even-sized sprites must have only four-cell mirror orbits"
            seen.add(orbit)
            out.append(orbit)
    return out


def _repair_and_defects(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    n = int(grid.shape[0])
    repaired = grid.copy()
    defects: list[tuple[int, int]] = []

    for orbit in _orbits(n):
        values = [int(grid[r, c]) for r, c in orbit]
        counts = Counter(values)
        majority_color, majority_count = counts.most_common(1)[0]
        assert majority_count in (3, 4), "each mirror orbit must be all equal or have one minority cell"

        if majority_count == 4:
            continue

        bad = [(r, c) for (r, c), value in zip(orbit, values) if value != majority_color]
        assert len(bad) == 1, "a defective orbit must contain exactly one minority cell"
        r, c = bad[0]
        repaired[r, c] = int(majority_color)
        defects.append((r, c))

    return repaired, defects


def _stamp(repaired: np.ndarray, defects: list[tuple[int, int]]) -> np.ndarray:
    n = int(repaired.shape[0])
    out = np.full((n * n, n * n), BLACK, dtype=int)
    for r, c in defects:
        out[r * n : (r + 1) * n, c * n : (c + 1) * n] = repaired
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair one-pixel mirror-orbit defects, then stamp the repaired sprite into the slots named by those defect positions.

    Input structure:
    - The input is an even-sided square sprite on the ARC color palette.
    - The intended sprite is symmetric under both horizontal and vertical reflection.
    - In each four-cell mirror orbit, either all four cells agree or exactly one cell differs from the other three.

    Rule:
    - For every cell, group it with its horizontal, vertical, and double mirror positions.
    - Replace the single minority cell of each defective orbit by the orbit majority color to recover the symmetric sprite.
    - Build an n by n array of n by n black slots, and copy the recovered sprite into slot (r, c) for every repaired defect at input position (r, c).

    Key insight:
    - A defect pixel is erased as image content but kept as an address: its mirror-orbit majority gives the stamp, while its original coordinate chooses where that stamp appears.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] == g.shape[1], "the sprite must be square"
    n = int(g.shape[0])
    assert n > 0 and n % 2 == 0, "the sprite side length must be positive and even"
    assert n * n <= 30, "expanded output must fit the ARC 30x30 limit"

    repaired, defects = _repair_and_defects(g)
    return _stamp(repaired, defects)


def _symmetric_sprite(n: int) -> np.ndarray:
    sprite = np.full((n, n), BLACK, dtype=int)
    orbits = _orbits(n)

    if n == 2:
        color_by_orbit = [random.choice(FOREGROUND_COLORS)]
    else:
        accent_a, accent_b = random.sample(FOREGROUND_COLORS, 2)
        palette = [BLACK, accent_a, accent_b]
        if random.random() < 0.45:
            palette.append(random.choice([c for c in FOREGROUND_COLORS if c not in (accent_a, accent_b)]))

        color_by_orbit = [random.choice(palette) for _ in orbits]
        nonblack_orbits = [i for i, color in enumerate(color_by_orbit) if color != BLACK]
        while len(nonblack_orbits) < 2:
            idx = random.randrange(len(orbits))
            color_by_orbit[idx] = random.choice((accent_a, accent_b))
            nonblack_orbits = [i for i, color in enumerate(color_by_orbit) if color != BLACK]

        if len({color_by_orbit[i] for i in nonblack_orbits}) < 2:
            color_by_orbit[random.choice(nonblack_orbits)] = accent_b

    for orbit, color in zip(orbits, color_by_orbit):
        for r, c in orbit:
            sprite[r, c] = int(color)
    return sprite


def _wrong_color(majority_color: int, n: int, r: int, c: int) -> int:
    choices = [color for color in ARC_COLORS if color != int(majority_color)]
    return int(choices[(r * 7 + c * 11 + n * 3 + int(majority_color)) % len(choices)])


def _slot_positions(out: np.ndarray, repaired: np.ndarray) -> list[tuple[int, int]]:
    n = int(repaired.shape[0])
    positions: list[tuple[int, int]] = []
    for sr in range(n):
        for sc in range(n):
            slot = out[sr * n : (sr + 1) * n, sc * n : (sc + 1) * n]
            if np.array_equal(slot, repaired):
                positions.append((sr, sc))
            else:
                assert np.all(slot == BLACK), "unaddressed slots must stay black"
    return positions


def _is_two_axis_symmetric(sprite: np.ndarray) -> bool:
    return bool(np.array_equal(sprite, np.flipud(sprite)) and np.array_equal(sprite, np.fliplr(sprite)))


def _gen_attempt() -> np.ndarray | None:
    n = 2 if random.random() < 0.18 else 4
    repaired = _symmetric_sprite(n)
    orbits = _orbits(n)

    if n == 2:
        defect_count = 1
    else:
        defect_count = random.choices([1, 2, 3, 4], weights=[1, 3, 4, 2], k=1)[0]

    chosen_orbits = random.sample(orbits, defect_count)
    grid = repaired.copy()
    defects: list[tuple[int, int]] = []
    for orbit in chosen_orbits:
        r, c = random.choice(orbit)
        grid[r, c] = _wrong_color(int(repaired[r, c]), n, r, c)
        defects.append((r, c))

    try:
        out = main(grid)
    except Exception:
        return None

    expected = _stamp(repaired, defects)
    if not np.array_equal(out, expected):
        return None
    if not _is_two_axis_symmetric(repaired):
        return None
    if np.array_equal(grid, repaired):
        return None
    if np.count_nonzero(repaired != BLACK) == 0 or np.count_nonzero(out != BLACK) == 0:
        return None
    if _slot_positions(out, repaired) != sorted(defects):
        return None

    repaired_again, found_defects = _repair_and_defects(grid)
    if not np.array_equal(repaired_again, repaired):
        return None
    if sorted(found_defects) != sorted(defects):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid defect-stamp input after {GEN_RETRIES} attempts")
