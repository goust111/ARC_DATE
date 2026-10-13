from __future__ import annotations

from collections import Counter
import random

import numpy as np


PALETTE = (1, 2, 3, 4, 6)
SIZE_RANGE = (5, 8)
MAX_GENERATION_ATTEMPTS = 400


def _rot_cw_pos(r: int, c: int, n: int) -> tuple[int, int]:
    return c, n - 1 - r


def _rotation_stack(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    return g, np.rot90(g, -1), np.rot90(g, 2), np.rot90(g, 1)


def _first_nonzero(values: list[int]) -> int:
    for value in values:
        if value != 0:
            return value
    return 0


def _solve_with_metadata(grid: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    rots = _rotation_stack(grid)
    h, w = grid.shape
    out = np.zeros((h, w), dtype=int)
    first_overlay = np.zeros((h, w), dtype=int)
    tie_22_mask = np.zeros((h, w), dtype=bool)
    fallback_tie_mask = np.zeros((h, w), dtype=bool)
    strong_majority_mask = np.zeros((h, w), dtype=bool)

    for r in range(h):
        for c in range(w):
            values = [int(rot[r, c]) for rot in rots]
            first_overlay[r, c] = _first_nonzero(values)
            nonzero = [value for value in values if value != 0]
            if not nonzero:
                continue

            counts = Counter(nonzero)
            max_count = max(counts.values())
            winners = [color for color, count in counts.items() if count == max_count]

            if len(winners) == 1:
                out[r, c] = winners[0]
                if max_count >= 3:
                    strong_majority_mask[r, c] = True
            else:
                out[r, c] = _first_nonzero(values)
                if len(counts) == 2 and sorted(counts.values()) == [2, 2]:
                    tie_22_mask[r, c] = True
                if values[0] == 0 and out[r, c] != 0:
                    fallback_tie_mask[r, c] = True

    meta = {
        "first_overlay": first_overlay,
        "tie_22_mask": tie_22_mask,
        "fallback_tie_mask": fallback_tie_mask,
        "strong_majority_mask": strong_majority_mask,
    }
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay the four quarter-turn versions of a square pattern and color each cell by non-black majority vote.

    Input structure:
    - The input is a square grid with black background and a sparse non-black foreground.
    - Foreground cells use 1-3 colors chosen from a small palette.
    - The pattern is not fully 4-fold rotation-symmetric, so different rotations can disagree at the same aligned cell.

    Rule:
    - Generate the 0-degree, 90-degree clockwise, 180-degree, and 270-degree clockwise versions of the whole grid.
    - Align those four versions on the same canvas and inspect the four colors at each cell position.
    - Ignore black votes; if one non-black color has a strict majority, output that color.
    - If all four votes are black, output black.
    - If the non-black votes tie, prefer the 0-degree color at that cell; if it is black, fall through in order to the 90-degree, 180-degree, then 270-degree version and take the first non-black color.

    Key insight:
    - Each output cell is decided by a vote among the four rotational views of one rotation orbit, with the original orientation acting as the tie-breaker.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h == w, "expected a square grid"
    out, _meta = _solve_with_metadata(g)
    return out


def _orbit(start: tuple[int, int], n: int) -> tuple[tuple[int, int], ...]:
    orbit = []
    cur = start
    while cur not in orbit:
        orbit.append(cur)
        cur = _rot_cw_pos(cur[0], cur[1], n)
    return tuple(orbit)


def _all_orbits(n: int) -> list[tuple[tuple[int, int], ...]]:
    seen: set[tuple[int, int]] = set()
    orbits: list[tuple[tuple[int, int], ...]] = []
    for r in range(n):
        for c in range(n):
            if (r, c) in seen:
                continue
            orbit = _orbit((r, c), n)
            seen.update(orbit)
            orbits.append(orbit)
    return orbits


def _assign_orbit(grid: np.ndarray, orbit: tuple[tuple[int, int], ...], values: list[int]) -> None:
    assert len(orbit) == len(values)
    shift = random.randrange(len(orbit))
    ordered_orbit = orbit[shift:] + orbit[:shift]
    for (r, c), value in zip(ordered_orbit, values):
        grid[r, c] = int(value)


def _extra_pattern(colors: list[int]) -> list[int]:
    dominant = random.choice(colors)
    others = [color for color in colors if color != dominant]
    secondary = random.choice(others) if others else dominant

    patterns = [
        [dominant, dominant, dominant, secondary],
        [dominant, secondary, dominant, secondary],
        [dominant, dominant, 0, secondary],
        [dominant, 0, 0, 0],
        [0, dominant, dominant, 0],
    ]
    if len(colors) >= 3:
        a, b, c = random.sample(colors, 3)
        patterns.append([a, b, c, 0])
    return random.choice(patterns)


def _generate_attempt() -> np.ndarray | None:
    n = random.randint(*SIZE_RANGE)
    color_count = random.randint(2, 3)
    colors = random.sample(list(PALETTE), color_count)

    grid = np.zeros((n, n), dtype=int)
    size4_orbits = [orbit for orbit in _all_orbits(n) if len(orbit) == 4]
    if len(size4_orbits) < 3:
        return None
    random.shuffle(size4_orbits)

    c0, c1 = colors[0], colors[1]
    c2 = colors[2] if len(colors) == 3 else random.choice((c0, c1))

    mandatory = [
        [c0, c1, c0, c1],
        [c0, c0, c0, c1],
        random.choice([[0, c1, 0, c2], [0, c1, c2, 0], [c1, 0, c2, 0], [0, c2, 0, c1]]),
    ]
    for orbit, values in zip(size4_orbits[:3], mandatory):
        _assign_orbit(grid, orbit, values)

    remaining = size4_orbits[3:]
    extra_count = random.randint(0, min(3, len(remaining)))
    for orbit in remaining[:extra_count]:
        _assign_orbit(grid, orbit, _extra_pattern(colors))

    used_colors = set(int(color) for color in grid.ravel() if color != 0)
    missing = [color for color in colors if color not in used_colors]
    for orbit, color in zip(remaining[extra_count:], missing):
        _assign_orbit(grid, orbit, [color, color, color, 0])

    out, meta = _solve_with_metadata(grid)
    tie_22_count = int(np.count_nonzero(meta["tie_22_mask"]))
    strong_majority_count = int(np.count_nonzero(meta["strong_majority_mask"]))
    fallback_tie_count = int(np.count_nonzero(meta["fallback_tie_mask"]))

    if np.array_equal(grid, np.rot90(grid, 1)):
        return None
    if tie_22_count < 2:
        return None
    if strong_majority_count < 3:
        return None
    if fallback_tie_count == 0:
        return None
    if np.array_equal(out, grid):
        return None
    if np.array_equal(out, meta["first_overlay"]):
        return None
    if np.array_equal(out, np.rot90(out, 1)):
        return None
    if not np.any((out != grid) & meta["strong_majority_mask"]):
        return None

    tie_colors = set(int(color) for color in out[meta["tie_22_mask"]])
    if len(tie_colors) < 2:
        return None

    input_colors = sorted(set(int(color) for color in grid.ravel() if color != 0))
    if input_colors != sorted(colors):
        return None
    if len(set(int(color) for color in out.ravel() if color != 0)) < 2:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {MAX_GENERATION_ATTEMPTS} attempts")
