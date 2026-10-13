from __future__ import annotations

import itertools
import random
from functools import lru_cache

import numpy as np


MASK = 6
PALETTE = (1, 2, 3, 4, 5, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400


def _rotate_cw(pos: tuple[int, int], n: int) -> tuple[int, int]:
    r, c = pos
    return int(c), int(n - 1 - r)


def _apply_sigma(sigma: dict[int, int], color: int, steps: int) -> int:
    out = int(color)
    for _ in range(int(steps) % 4):
        out = int(sigma[out])
    return out


def _cycles_from_sigma(sigma: dict[int, int]) -> list[list[int]]:
    seen: set[int] = set()
    cycles: list[list[int]] = []
    for color in sorted(sigma):
        if color in seen:
            continue
        cur = color
        cycle: list[int] = []
        while cur not in seen:
            seen.add(cur)
            cycle.append(cur)
            cur = int(sigma[cur])
        cycles.append(cycle)
    return cycles


@lru_cache(maxsize=None)
def _orbit_sequence(start: tuple[int, int], n: int) -> tuple[tuple[int, int], ...]:
    seq: list[tuple[int, int]] = []
    cur = (int(start[0]), int(start[1]))
    while cur not in seq:
        seq.append(cur)
        cur = _rotate_cw(cur, n)
    return tuple(seq)


@lru_cache(maxsize=None)
def _all_orbits(n: int) -> tuple[tuple[tuple[int, int], ...], ...]:
    seen: set[tuple[int, int]] = set()
    out: list[tuple[tuple[int, int], ...]] = []
    for r in range(n):
        for c in range(n):
            if (r, c) in seen:
                continue
            orbit = _orbit_sequence((r, c), n)
            seen.update(orbit)
            out.append(orbit)
    return tuple(out)


def _visible_colors(grid: np.ndarray) -> tuple[int, ...]:
    return tuple(sorted(int(color) for color in np.unique(grid) if int(color) != MASK))


def _observed_edges(grid: np.ndarray) -> list[tuple[int, int]]:
    n = int(grid.shape[0])
    edges: list[tuple[int, int]] = []
    for r in range(n):
        for c in range(n):
            a = int(grid[r, c])
            rr, cc = _rotate_cw((r, c), n)
            b = int(grid[rr, cc])
            if a != MASK and b != MASK:
                edges.append((a, b))
    return edges


def _sigma_fits_grid(grid: np.ndarray, sigma: dict[int, int]) -> bool:
    n = int(grid.shape[0])
    for orbit in _all_orbits(n):
        visible = [
            (idx, int(grid[r, c]))
            for idx, (r, c) in enumerate(orbit)
            if int(grid[r, c]) != MASK
        ]
        if not visible:
            return False
        base_idx, base_color = visible[0]
        for idx, color in visible[1:]:
            if _apply_sigma(sigma, base_color, idx - base_idx) != int(color):
                return False
    return True


def _all_consistent_sigmas(grid: np.ndarray) -> list[dict[int, int]]:
    colors = _visible_colors(grid)
    assert colors, "expected at least one visible non-mask color"

    outgoing: dict[int, int] = {}
    incoming: dict[int, int] = {}
    for src, dst in _observed_edges(grid):
        old_dst = outgoing.get(int(src))
        if old_dst is not None and old_dst != int(dst):
            return []
        old_src = incoming.get(int(dst))
        if old_src is not None and old_src != int(src):
            return []
        outgoing[int(src)] = int(dst)
        incoming[int(dst)] = int(src)

    free_sources = [color for color in colors if color not in outgoing]
    free_targets = [color for color in colors if color not in incoming]
    if len(free_sources) != len(free_targets):
        return []

    solutions: list[dict[int, int]] = []
    for perm in itertools.permutations(free_targets):
        sigma = dict(outgoing)
        sigma.update({src: dst for src, dst in zip(free_sources, perm)})
        if any(_apply_sigma(sigma, color, 4) != int(color) for color in colors):
            continue
        if _sigma_fits_grid(grid, sigma):
            solutions.append(sigma)
    return solutions


def _infer_sigma(grid: np.ndarray) -> dict[int, int]:
    solutions = _all_consistent_sigmas(grid)
    assert len(solutions) == 1, f"visible cells must determine one permutation, got {len(solutions)}"
    return solutions[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover pink masks using a 90-degree rotation followed by a fixed color permutation.

    Input structure:
    - The input is a square grid filled with colors, with some cells replaced by pink masks.
    - Ignoring the pink cells, there exists one fixed permutation sigma on the visible colors.
    - Rotating any cell 90 degrees clockwise sends its color to sigma(original_color); pink is only a mask and is not part of sigma.

    Rule:
    - Use every visible pair of cells related by one clockwise quarter-turn to infer sigma.
    - For each pink cell, look at another visible cell in the same rotation orbit and compute how many quarter-turns separate them.
    - Fill the mask with the visible color transformed by sigma to that orbit distance; keep every non-pink cell unchanged.

    Key insight:
    - The pattern is not plain rotationally symmetric in color space; the invariant is rotation composed with a global recoloring.
    - A mask can require sigma squared or sigma cubed, so both the permutation and the orbit step count matter.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1], "input must be a square grid"
    assert np.any(g == MASK), "input must contain pink masks"

    sigma = _infer_sigma(g)
    n = int(g.shape[0])
    out = g.copy()

    for orbit in _all_orbits(n):
        visible = [
            (idx, int(g[r, c]))
            for idx, (r, c) in enumerate(orbit)
            if int(g[r, c]) != MASK
        ]
        assert visible, "every masked orbit must keep at least one visible cell"
        for idx, (r, c) in enumerate(orbit):
            if int(g[r, c]) != MASK:
                continue
            candidates = {
                _apply_sigma(sigma, color, idx - ref_idx)
                for ref_idx, color in visible
            }
            assert len(candidates) == 1, "masked cell must have one orbit-consistent color"
            out[r, c] = candidates.pop()

    assert not np.any(out == MASK), "all masks must be filled"
    assert _sigma_fits_grid(out, sigma), "reconstructed grid must satisfy the twist symmetry"
    return out


def _sample_cycle_lengths(n: int) -> tuple[int, ...]:
    if n % 2 == 1:
        options = [(4, 1), (4, 2, 1), (4, 1, 1)]
        weights = [5, 4, 2]
    else:
        options = [(4,), (4, 2), (4, 1), (4, 2, 1), (4, 1, 1)]
        weights = [4, 4, 2, 2, 1]
    return random.choices(options, weights=weights, k=1)[0]


def _sample_cycles(n: int) -> list[list[int]]:
    lengths = _sample_cycle_lengths(n)
    colors = random.sample(PALETTE, sum(lengths))
    random.shuffle(colors)
    cycles: list[list[int]] = []
    idx = 0
    for length in lengths:
        cycle = colors[idx : idx + length]
        idx += length
        random.shuffle(cycle)
        cycles.append(cycle)
    return cycles


def _sigma_from_cycles(cycles: list[list[int]]) -> dict[int, int]:
    sigma: dict[int, int] = {}
    for cycle in cycles:
        for idx, color in enumerate(cycle):
            sigma[int(color)] = int(cycle[(idx + 1) % len(cycle)])
    return sigma


def _paint_orbit(grid: np.ndarray, orbit: tuple[tuple[int, int], ...], base_color: int, sigma: dict[int, int]) -> None:
    for idx, (r, c) in enumerate(orbit):
        grid[r, c] = _apply_sigma(sigma, int(base_color), idx)


def _mask_orbit(
    grid: np.ndarray,
    orbit: tuple[tuple[int, int], ...],
    visible_indices: set[int],
) -> int:
    masked = 0
    for idx, (r, c) in enumerate(orbit):
        if idx not in visible_indices:
            grid[r, c] = MASK
            masked += 1
    return masked


def _pattern_visible_indices(pattern: str, anchor: int) -> set[int]:
    if pattern == "one_visible":
        return {anchor}
    if pattern == "one_mask":
        return {idx for idx in range(4) if idx != anchor}
    if pattern == "two_adjacent_visible":
        return {anchor, (anchor + 1) % 4}
    if pattern == "two_opposite_visible":
        return {anchor, (anchor + 2) % 4}
    raise ValueError(f"unknown pattern: {pattern}")


def _semantic_checks(inp: np.ndarray, out: np.ndarray, sigma: dict[int, int]) -> None:
    inferred = _infer_sigma(inp)
    assert inferred == sigma, "generator must expose the exact permutation"
    assert np.array_equal(main(inp), out), "main must reconstruct the sampled output"
    assert not np.array_equal(out, np.rot90(out, -1)), "plain rotation should not explain the output"

    visible_colors = set(_visible_colors(inp))
    output_colors = set(int(color) for color in np.unique(out))
    assert output_colors <= visible_colors, "every output color must stay visible somewhere in the input"

    found_noncopy = False
    found_far_power = False
    n = int(inp.shape[0])
    for orbit in _all_orbits(n):
        visible = [
            (idx, int(inp[r, c]))
            for idx, (r, c) in enumerate(orbit)
            if int(inp[r, c]) != MASK
        ]
        masked = [
            idx
            for idx, (r, c) in enumerate(orbit)
            if int(inp[r, c]) == MASK
        ]
        if masked:
            assert visible, "every masked orbit must retain a visible witness"
        for idx in masked:
            target = int(out[orbit[idx][0], orbit[idx][1]])
            if target not in {color for _ref_idx, color in visible}:
                found_noncopy = True
            if len(visible) == 1 and (idx - visible[0][0]) % 4 in {2, 3}:
                found_far_power = True

    assert found_noncopy, "at least one mask must require recoloring rather than direct copying"
    assert found_far_power, "at least one mask must require sigma squared or sigma cubed"


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(8, 20)
    cycles = _sample_cycles(n)
    sigma = _sigma_from_cycles(cycles)
    active_colors = tuple(sorted(sigma))
    four_cycle = next(cycle for cycle in cycles if len(cycle) == 4)
    fixed_colors = [cycle[0] for cycle in cycles if len(cycle) == 1]

    orbits = list(_all_orbits(n))
    size4_orbits = [orbit for orbit in orbits if len(orbit) == 4]
    center_orbit = next((orbit for orbit in orbits if len(orbit) == 1), None)
    random.shuffle(size4_orbits)

    if len(size4_orbits) < len(cycles) + 2:
        return None

    witness_orbits = size4_orbits[: len(cycles)]
    remaining_orbits = size4_orbits[len(cycles) :]
    hard_orbit = remaining_orbits.pop()

    out = np.zeros((n, n), dtype=int)

    for cycle, orbit in zip(cycles, witness_orbits):
        _paint_orbit(out, orbit, random.choice(cycle), sigma)

    _paint_orbit(out, hard_orbit, random.choice(four_cycle), sigma)
    for orbit in remaining_orbits:
        _paint_orbit(out, orbit, random.choice(active_colors), sigma)

    if center_orbit is not None:
        if not fixed_colors:
            return None
        center_color = random.choice(fixed_colors)
        r, c = center_orbit[0]
        out[r, c] = int(center_color)

    inp = out.copy()
    masked_cells = 0

    visible_anchor = random.randrange(4)
    masked_cells += _mask_orbit(inp, hard_orbit, {visible_anchor})

    extra_target = random.randint(max(6, n), min(3 * len(size4_orbits) - 3, 2 * n + 8))
    random.shuffle(remaining_orbits)

    pattern_names = (
        "one_mask",
        "two_adjacent_visible",
        "two_opposite_visible",
        "one_visible",
    )
    pattern_weights = (4, 3, 3, 4)

    used_mask_orbits = {hard_orbit}
    for orbit in remaining_orbits:
        if masked_cells >= extra_target and len(used_mask_orbits) >= 3:
            break
        if orbit in used_mask_orbits:
            continue
        pattern = random.choices(pattern_names, weights=pattern_weights, k=1)[0]
        anchor = random.randrange(4)
        visible_indices = _pattern_visible_indices(pattern, anchor)
        if len(visible_indices) == 4:
            continue
        masked_cells += _mask_orbit(inp, orbit, visible_indices)
        used_mask_orbits.add(orbit)

    if masked_cells < 6 or not np.any(inp == MASK):
        return None

    try:
        _semantic_checks(inp, out, sigma)
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid twist-rotation ARC input")
