import random
from math import isqrt

import numpy as np


BACKGROUND = 0
FOREGROUND_CHOICES = (1, 2, 3, 4, 5, 6, 7, 8, 9)
SIZE_CHOICES = (2, 3, 4, 5)
MAX_GENERATION_ATTEMPTS = 400


def _foreground_color(grid: np.ndarray) -> int:
    colors = sorted(int(v) for v in np.unique(grid) if int(v) != BACKGROUND)
    assert len(colors) == 1, "expected exactly one non-black foreground color"
    return colors[0]


def _half_turn_partner(n: int, r: int, c: int) -> tuple[int, int]:
    return n - 1 - r, n - 1 - c


def _point_symmetry_orbits(n: int) -> list[tuple[tuple[int, int], ...]]:
    seen: set[tuple[int, int]] = set()
    orbits: list[tuple[tuple[int, int], ...]] = []
    for r in range(n):
        for c in range(n):
            if (r, c) in seen:
                continue
            partner = _half_turn_partner(n, r, c)
            if partner == (r, c):
                orbit = ((r, c),)
            else:
                orbit = ((r, c), partner)
            seen.update(orbit)
            orbits.append(orbit)
    return orbits


def _expand_with_rule(grid: np.ndarray, use_partner_orientation: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    n = int(g.shape[0])
    out = np.zeros((n * n, n * n), dtype=int)
    rotated = np.rot90(g, 2)
    mask = g != BACKGROUND
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if use_partner_orientation:
            pr, pc = _half_turn_partner(n, rr, cc)
            patch = g if mask[pr, pc] else rotated
        else:
            patch = g
        out[rr * n : (rr + 1) * n, cc * n : (cc + 1) * n] = patch
    return out


def _expand_all_rotated(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    n = int(g.shape[0])
    out = np.zeros((n * n, n * n), dtype=int)
    rotated = np.rot90(g, 2)
    for r, c in np.argwhere(g != BACKGROUND):
        rr = int(r)
        cc = int(c)
        out[rr * n : (rr + 1) * n, cc * n : (cc + 1) * n] = rotated
    return out


def _recover_layout_mask(output: np.ndarray) -> np.ndarray:
    h, w = map(int, output.shape)
    assert h == w, "output must be square"
    n = isqrt(h)
    assert n * n == h, "output side must be a perfect square"
    mask = np.zeros((n, n), dtype=bool)
    for r in range(n):
        for c in range(n):
            block = output[r * n : (r + 1) * n, c * n : (c + 1) * n]
            mask[r, c] = bool(np.any(block != BACKGROUND))
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """Expand the mask into an n-by-n lattice of self-stamps, flipping stamps for foreground cells whose point-symmetric partner is absent.

    Input structure:
    - The input is an n×n square mask on a black background.
    - All foreground pixels share one non-black color.
    - The mask contains both foreground and background and is not point-symmetric under 180° rotation.

    Rule:
    - Create an n²×n² black canvas and divide it into an n×n lattice of n×n sub-blocks.
    - Each background input cell leaves its matching sub-block black.
    - Each foreground input cell copies the whole input mask into its sub-block:
      use the original mask if its point-symmetric partner is also foreground,
      otherwise use the 180° rotated mask.

    Key insight:
    - The same mask plays three roles at once: it decides which lattice blocks exist,
      it provides the stamped pattern, and its own point-symmetry defects decide each stamp orientation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    n, m = map(int, g.shape)
    assert n == m, "expected a square mask"
    assert 1 < n <= 5 and n * n <= 30, "input size must keep the expanded grid within 30x30"
    fg = _foreground_color(g)
    mask = g == fg
    assert np.any(mask), "expected at least one foreground pixel"
    assert np.any(~mask), "expected at least one background pixel"
    assert not np.array_equal(mask, np.rot90(mask, 2)), "input mask must not be point-symmetric"
    return _expand_with_rule(g, use_partner_orientation=True)


def _sample_mask(n: int) -> np.ndarray:
    orbits = _point_symmetry_orbits(n)
    pair_orbits = [orbit for orbit in orbits if len(orbit) == 2]
    center_orbits = [orbit for orbit in orbits if len(orbit) == 1]
    assert len(pair_orbits) >= 2

    full_orbit = random.choice(pair_orbits)
    remaining = [orbit for orbit in pair_orbits if orbit != full_orbit]
    partial_orbit = random.choice(remaining)

    mask = np.zeros((n, n), dtype=bool)
    for r, c in full_orbit:
        mask[r, c] = True

    chosen_partial = random.choice(partial_orbit)
    mask[chosen_partial] = True

    for orbit in remaining:
        if orbit == partial_orbit:
            continue
        state = random.choices(
            population=("empty", "full", "partial"),
            weights=(2, 2, 3),
            k=1,
        )[0]
        if state == "full":
            for r, c in orbit:
                mask[r, c] = True
        elif state == "partial":
            r, c = random.choice(orbit)
            mask[r, c] = True

    for orbit in center_orbits:
        if random.random() < 0.5:
            r, c = orbit[0]
            mask[r, c] = True

    return mask


def _semantic_check(grid: np.ndarray) -> bool:
    try:
        g = np.asarray(grid, dtype=int)
        n, m = map(int, g.shape)
        if n != m or not (1 < n <= 5 and n * n <= 30):
            return False
        fg = _foreground_color(g)
        mask = g == fg
        if not np.any(mask) or not np.any(~mask):
            return False
        if np.array_equal(mask, np.rot90(mask, 2)):
            return False

        paired = []
        unpaired = []
        for r, c in np.argwhere(mask):
            rr = int(r)
            cc = int(c)
            pr, pc = _half_turn_partner(n, rr, cc)
            if mask[pr, pc]:
                paired.append((rr, cc))
            else:
                unpaired.append((rr, cc))
        if not paired or not unpaired:
            return False

        out = main(g)
        if out.shape != (n * n, n * n):
            return False
        if not np.array_equal(_recover_layout_mask(out), mask):
            return False

        expected = _expand_with_rule(g, use_partner_orientation=True)
        if not np.array_equal(out, expected):
            return False
        if np.array_equal(out, _expand_with_rule(g, use_partner_orientation=False)):
            return False
        if np.array_equal(out, _expand_all_rotated(g)):
            return False

        rotated = np.rot90(g, 2)
        patterns = set()
        for r in range(n):
            for c in range(n):
                block = out[r * n : (r + 1) * n, c * n : (c + 1) * n]
                if not mask[r, c]:
                    if np.any(block != BACKGROUND):
                        return False
                    continue
                pr, pc = _half_turn_partner(n, r, c)
                target = g if mask[pr, pc] else rotated
                if not np.array_equal(block, target):
                    return False
                patterns.add(block.tobytes())
        if len(patterns) != 2:
            return False
    except AssertionError:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(SIZE_CHOICES)
    color = random.choice(FOREGROUND_CHOICES)
    mask = _sample_mask(n)
    grid = np.zeros((n, n), dtype=int)
    grid[mask] = color
    if not _semantic_check(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid point-symmetry fractal mask")
