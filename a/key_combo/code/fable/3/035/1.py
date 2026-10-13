from __future__ import annotations

from functools import lru_cache

import numpy as np


BLACK = 0
NON_BLACK_COLORS = np.arange(1, 10, dtype=int)
SYMMETRY_GROUPS = ("mirror", "rotation")
GEN_SIDE_CHOICES = np.arange(6, 18, 2, dtype=int)
GEN_PALETTE_SIZES = np.arange(3, 7, dtype=int)
GEN_VISIBLE_COUNT_OPTIONS = np.array([1, 2, 3], dtype=int)
GEN_VISIBLE_COUNT_PROBS = np.array([0.25, 0.5, 0.25], dtype=float)
GEN_MAX_ATTEMPTS = 500


def _mirror_orbit(side: int, row: int, col: int) -> tuple[tuple[int, int], ...]:
    orbit = {
        (int(row), int(col)),
        (int(row), side - 1 - int(col)),
        (side - 1 - int(row), int(col)),
        (side - 1 - int(row), side - 1 - int(col)),
    }
    return tuple(sorted(orbit))


def _rotation_orbit(side: int, row: int, col: int) -> tuple[tuple[int, int], ...]:
    orbit = {
        (int(row), int(col)),
        (int(col), side - 1 - int(row)),
        (side - 1 - int(row), side - 1 - int(col)),
        (side - 1 - int(col), int(row)),
    }
    return tuple(sorted(orbit))


@lru_cache(None)
def _group_orbits(side: int, group: str) -> tuple[tuple[tuple[int, int], ...], ...]:
    if group not in SYMMETRY_GROUPS:
        raise ValueError(f"unknown symmetry group: {group}")

    orbit_fn = _mirror_orbit if group == "mirror" else _rotation_orbit
    seen: set[tuple[int, int]] = set()
    orbits: list[tuple[tuple[int, int], ...]] = []

    for row in range(side):
        for col in range(side):
            cell = (row, col)
            if cell in seen:
                continue
            orbit = orbit_fn(side, row, col)
            seen.update(orbit)
            orbits.append(orbit)

    return tuple(orbits)


def _complete_with_group(grid: np.ndarray, group: str) -> np.ndarray | None:
    side = int(grid.shape[0])
    out = np.array(grid, copy=True)

    for orbit in _group_orbits(side, group):
        visible_colors = {int(grid[row, col]) for row, col in orbit if int(grid[row, col]) != BLACK}
        if not visible_colors or len(visible_colors) != 1:
            return None
        color = next(iter(visible_colors))
        for row, col in orbit:
            out[row, col] = color

    return out


def _other_group(group: str) -> str:
    return "rotation" if group == "mirror" else "mirror"


def _conflicting_orbits(grid: np.ndarray, group: str) -> list[tuple[tuple[int, int], ...]]:
    conflicts: list[tuple[tuple[int, int], ...]] = []
    side = int(grid.shape[0])
    for orbit in _group_orbits(side, group):
        colors = {int(grid[row, col]) for row, col in orbit}
        if len(colors) > 1:
            conflicts.append(orbit)
    return conflicts


def _sample_full_pattern(side: int, group: str, rng: np.random.Generator) -> np.ndarray:
    orbits = list(_group_orbits(side, group))
    palette_size = int(rng.choice(GEN_PALETTE_SIZES))
    palette = [int(color) for color in rng.choice(NON_BLACK_COLORS, size=palette_size, replace=False)]
    orbit_colors = palette[:]
    if len(orbits) > len(orbit_colors):
        extra = rng.choice(np.array(palette, dtype=int), size=len(orbits) - len(orbit_colors), replace=True)
        orbit_colors.extend(int(color) for color in extra)
    rng.shuffle(orbit_colors)

    full = np.zeros((side, side), dtype=int)
    for orbit, color in zip(orbits, orbit_colors):
        for row, col in orbit:
            full[row, col] = int(color)
    return full


def _mask_full_pattern(
    full: np.ndarray,
    group: str,
    forced_visible: set[tuple[int, int]],
    rng: np.random.Generator,
) -> np.ndarray:
    side = int(full.shape[0])
    inp = np.full_like(full, BLACK)

    for orbit in _group_orbits(side, group):
        forced_here = [cell for cell in orbit if cell in forced_visible]
        if len(forced_here) == 4:
            chosen = forced_here
        else:
            visible_count = int(rng.choice(GEN_VISIBLE_COUNT_OPTIONS, p=GEN_VISIBLE_COUNT_PROBS))
            visible_count = max(visible_count, len(forced_here))
            remaining = [cell for cell in orbit if cell not in forced_here]
            rng.shuffle(remaining)
            chosen = forced_here + remaining[: visible_count - len(forced_here)]

        for row, col in chosen:
            inp[row, col] = int(full[row, col])

    return inp


def _quadrants_are_damaged(grid: np.ndarray) -> bool:
    side = int(grid.shape[0])
    half = side // 2
    quadrants = (
        grid[:half, :half],
        grid[:half, half:],
        grid[half:, :half],
        grid[half:, half:],
    )
    return all(np.any(quadrant == BLACK) and np.any(quadrant != BLACK) for quadrant in quadrants)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the black gaps by arbitrating between double-axis reflection and quarter-turn rotation symmetry.

    Input structure:
    - The canvas is a 2n x 2n square.
    - Non-black cells are visible remnants of a complete pattern, while black cells are missing.
    - The fully restored pattern itself contains no black cells.

    Rule:
    - For every location, build its four-cell orbit under double-axis reflection and under 90-degree rotation around the canvas center.
    - Exactly one symmetry group has the property that every orbit contains at least one visible cell and that all visible cells inside the orbit share one color.
    - Choose that unique valid group and fill every black cell with the orbit's visible color.

    Key insight:
    - No quadrant can be trusted as a full template; the correct completion comes from global orbit consistency, because the wrong symmetry group inevitably combines incompatible visible colors somewhere on the board.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height == width and height > 0 and height % 2 == 0, "expected an even square grid"

    mirror = _complete_with_group(g, "mirror")
    rotation = _complete_with_group(g, "rotation")
    valid_count = int(mirror is not None) + int(rotation is not None)
    assert valid_count == 1, "expected exactly one symmetry group to fit the visible evidence"
    return mirror if mirror is not None else rotation


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()

    for _ in range(GEN_MAX_ATTEMPTS):
        side = int(rng.choice(GEN_SIDE_CHOICES))
        true_group = str(rng.choice(np.array(SYMMETRY_GROUPS, dtype=object)))
        wrong_group = _other_group(true_group)

        full = _sample_full_pattern(side, true_group, rng)
        if _complete_with_group(full, wrong_group) is not None:
            continue

        witnesses = _conflicting_orbits(full, wrong_group)
        if not witnesses:
            continue
        witness = witnesses[int(rng.integers(len(witnesses)))]
        inp = _mask_full_pattern(full, true_group, set(witness), rng)

        if not _quadrants_are_damaged(inp):
            continue

        if np.array_equal(inp, full):
            continue

        if _complete_with_group(inp, wrong_group) is not None:
            continue

        try:
            out = main(inp)
        except AssertionError:
            continue
        if np.array_equal(out, full):
            return inp

    raise RuntimeError("failed to generate a valid symmetry-arbitration input")
