from __future__ import annotations

import itertools
import random

import numpy as np


MASK = 6
PALETTE = [1, 2, 3, 4, 5, 7, 8, 9]
SHAPES = [
    np.array([[1]], dtype=int),
    np.array([[1, 1]], dtype=int),
    np.array([[1], [1]], dtype=int),
    np.array([[1, 1], [1, 0]], dtype=int),
    np.array([[1, 1], [0, 1]], dtype=int),
    np.array([[1, 1], [1, 1]], dtype=int),
    np.array([[1, 1, 1]], dtype=int),
    np.array([[1], [1], [1]], dtype=int),
    np.array([[1, 0], [1, 1], [1, 0]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
]
GEN_RETRIES = 300


def _rot90(pos: tuple[int, int], n: int) -> tuple[int, int]:
    r, c = pos
    return c, n - 1 - r


def _canonical_orbit(cells: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    if len(cells) == 1:
        return (cells[0],)
    rotations = [tuple(cells[i:] + cells[:i]) for i in range(len(cells))]
    return min(rotations)


def _all_orbits(n: int) -> list[tuple[tuple[int, int], ...]]:
    seen: set[tuple[int, int]] = set()
    orbits: list[tuple[tuple[int, int], ...]] = []
    for r in range(n):
        for c in range(n):
            if (r, c) in seen:
                continue
            orbit: list[tuple[int, int]] = []
            cur = (r, c)
            for _ in range(4):
                if cur in orbit:
                    break
                orbit.append(cur)
                cur = _rot90(cur, n)
            for cell in orbit:
                seen.add(cell)
            orbits.append(_canonical_orbit(orbit))
    return orbits


def _advance(mapping: dict[int, int], color: int, steps: int) -> int:
    out = int(color)
    for _ in range(steps % 4):
        out = int(mapping[out])
    return out


def _candidate_mappings(colors: list[int]) -> list[dict[int, int]]:
    candidates: list[dict[int, int]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for stable in colors:
        cycle = [c for c in colors if c != stable]
        if len(cycle) not in (2, 4):
            continue
        for perm in itertools.permutations(cycle):
            mapping = {int(stable): int(stable)}
            for i, color in enumerate(perm):
                mapping[int(color)] = int(perm[(i + 1) % len(perm)])
            key = tuple(sorted(mapping.items()))
            if key in seen:
                continue
            seen.add(key)
            if all(_advance(mapping, color, 4) == color for color in mapping):
                candidates.append(mapping)
    return candidates


def _mapping_fits(grid: np.ndarray, orbits: list[tuple[tuple[int, int], ...]], mapping: dict[int, int]) -> bool:
    for orbit in orbits:
        visible: list[tuple[int, int]] = []
        masked_count = 0
        for idx, (r, c) in enumerate(orbit):
            value = int(grid[r, c])
            if value == MASK:
                masked_count += 1
            else:
                if value not in mapping:
                    return False
                visible.append((idx, value))
        if masked_count and not visible:
            return False
        if len(orbit) == 1:
            if visible and mapping[visible[0][1]] != visible[0][1]:
                return False
            continue
        for i, color_i in visible:
            for j, color_j in visible:
                if _advance(mapping, color_i, j - i) != color_j:
                    return False
    return True


def _infer_mapping(grid: np.ndarray) -> tuple[dict[int, int], list[tuple[tuple[int, int], ...]]]:
    n = int(grid.shape[0])
    orbits = _all_orbits(n)
    colors = sorted(int(v) for v in np.unique(grid) if int(v) != MASK)
    candidates = [m for m in _candidate_mappings(colors) if _mapping_fits(grid, orbits, m)]
    assert len(candidates) == 1, f"expected one color-cycle mapping, got {len(candidates)}"
    return candidates[0], orbits


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each pink mask from its 90-degree rotation orbit while advancing colors along one fixed cycle.

    Input structure:
    - The grid is square and contains a mostly complete colored pattern plus some pink masked cells.
    - Rotating the completed pattern by 90 degrees preserves shape positions.
    - Under that same rotation, one special color stays fixed while the other used colors advance by one step in a single cycle.

    Rule:
    - Partition cells into 90-degree rotation orbits around the grid center.
    - Infer the unique stable color and the unique cyclic recoloring order that make all visible orbit cells consistent.
    - For each pink cell, copy a visible color from the same orbit and advance it by the required number of quarter-turn steps.

    Key insight:
    - This is not ordinary rotational symmetry: geometry repeats after a quarter-turn only if colors are transformed by the same global permutation each time.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1], "input must be a square grid"
    assert np.any(g == MASK), "input must contain pink masks"

    mapping, orbits = _infer_mapping(g)
    out = g.copy()
    for orbit in orbits:
        masked_positions = [idx for idx, (r, c) in enumerate(orbit) if g[r, c] == MASK]
        if not masked_positions:
            continue
        visible = [(idx, int(g[r, c])) for idx, (r, c) in enumerate(orbit) if g[r, c] != MASK]
        assert visible, "each masked orbit must retain a visible witness"
        src_idx, src_color = visible[0]
        for idx in masked_positions:
            r, c = orbit[idx]
            out[r, c] = _advance(mapping, src_color, idx - src_idx)

    assert not np.any(out == MASK), "all masks must be resolved"
    return out


def _place_shape(seed: np.ndarray, occupied: np.ndarray, shape: np.ndarray, color: int, padding: int = 1) -> bool:
    q = int(seed.shape[0])
    sh, sw = map(int, shape.shape)
    placements = []
    for r in range(q - sh + 1):
        for c in range(q - sw + 1):
            r0 = max(0, r - padding)
            c0 = max(0, c - padding)
            r1 = min(q, r + sh + padding)
            c1 = min(q, c + sw + padding)
            if np.any(occupied[r0:r1, c0:c1]):
                continue
            placements.append((r, c))
    if not placements:
        return False
    r, c = random.choice(placements)
    mask = shape.astype(bool)
    seed[r : r + sh, c : c + sw][mask] = int(color)
    occupied[r : r + sh, c : c + sw][mask] = True
    return True


def _expand_seed(seed: np.ndarray, n: int, stable: int, mapping: dict[int, int]) -> np.ndarray:
    out = np.full((n, n), int(stable), dtype=int)
    q = int(seed.shape[0])
    for r in range(q):
        for c in range(q):
            base = int(seed[r, c])
            if base == 0:
                continue
            pos = (r, c)
            for step in range(4):
                rr, cc = pos
                out[rr, cc] = _advance(mapping, base, step)
                pos = _rot90(pos, n)
    return out


def _orbit_index_map(orbits: list[tuple[tuple[int, int], ...]]) -> dict[tuple[int, int], int]:
    mapping: dict[tuple[int, int], int] = {}
    for idx, orbit in enumerate(orbits):
        for cell in orbit:
            mapping[cell] = idx
    return mapping


def _gen_seed_pattern(q: int, cycle: list[int]) -> tuple[np.ndarray, set[int]] | None:
    seed = np.zeros((q, q), dtype=int)
    occupied = np.zeros((q, q), dtype=bool)

    placements = random.randint(3, min(6, q + 1))
    anchor_shape = random.choice(SHAPES[1:])
    if not _place_shape(seed, occupied, anchor_shape, cycle[0]):
        return None

    for _ in range(placements - 1):
        shape = random.choice(SHAPES)
        color = random.choice(cycle)
        if not _place_shape(seed, occupied, shape, color):
            return None

    nonzero = set(int(v) for v in seed.flatten() if int(v) != 0)
    if not nonzero:
        return None
    return seed, nonzero


def _choose_mask_positions(full: np.ndarray, orbits: list[tuple[tuple[int, int], ...]], anchor_orbits: set[int]) -> np.ndarray | None:
    inp = full.copy()
    candidates: list[int] = []
    for idx, orbit in enumerate(orbits):
        if idx in anchor_orbits or len(orbit) == 1:
            continue
        candidates.append(idx)
    random.shuffle(candidates)

    target_orbits = random.randint(4, min(12, len(candidates)))
    used = 0
    for orbit_idx in candidates:
        if used >= target_orbits:
            break
        orbit = orbits[orbit_idx]
        values = [int(full[r, c]) for r, c in orbit]
        distinct = len(set(values))
        max_masks = 2 if distinct == 1 else 3
        mask_count = random.randint(1, max_masks)
        mask_count = min(mask_count, len(orbit) - 1)
        if mask_count <= 0:
            continue
        mask_slots = random.sample(range(len(orbit)), k=mask_count)
        for slot in mask_slots:
            r, c = orbit[slot]
            inp[r, c] = MASK
        used += 1

    if not np.any(inp == MASK):
        return None
    return inp


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(10, 18)
    q = n // 2

    colors = random.sample(PALETTE, k=random.choice([3, 5]))
    stable = colors[0]
    cycle = colors[1:]
    if len(cycle) not in (2, 4):
        return None

    cycle_mapping = {stable: stable}
    for i, color in enumerate(cycle):
        cycle_mapping[color] = cycle[(i + 1) % len(cycle)]

    seeded = _gen_seed_pattern(q, cycle)
    if seeded is None:
        return None
    seed, _ = seeded
    full = _expand_seed(seed, n, stable, cycle_mapping)

    orbits = _all_orbits(n)
    orbit_by_cell = _orbit_index_map(orbits)
    anchor_orbits: set[int] = set()
    for r in range(q):
        for c in range(q):
            if int(seed[r, c]) != 0:
                anchor_orbits.add(orbit_by_cell[(r, c)])
                break
        if anchor_orbits:
            break

    inp = _choose_mask_positions(full, orbits, anchor_orbits)
    if inp is None:
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, full):
        return None

    colored = int(np.count_nonzero(full != stable))
    masks = int(np.count_nonzero(inp == MASK))
    if colored < 16 or masks < 4:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
