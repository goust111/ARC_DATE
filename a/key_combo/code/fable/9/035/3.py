from __future__ import annotations

import numpy as np


Coord = tuple[int, int]
Orbit = tuple[Coord, ...]


_ORBIT_CACHE: dict[int, list[Orbit]] = {}


def _rot90_cw(r: int, c: int, n: int) -> Coord:
    return int(c), int(n - 1 - r)


def _rotation_orbits(n: int) -> list[Orbit]:
    cached = _ORBIT_CACHE.get(n)
    if cached is not None:
        return cached

    seen: set[Coord] = set()
    orbits: list[Orbit] = []
    for r in range(n):
        for c in range(n):
            if (r, c) in seen:
                continue
            cycle: list[Coord] = []
            rr, cc = r, c
            while (rr, cc) not in cycle:
                cycle.append((rr, cc))
                rr, cc = _rot90_cw(rr, cc, n)
            start = min(range(len(cycle)), key=lambda i: cycle[i])
            orbit = tuple(cycle[(start + i) % len(cycle)] for i in range(len(cycle)))
            seen.update(orbit)
            orbits.append(orbit)

    _ORBIT_CACHE[n] = orbits
    return orbits


def _apply_mapping(mapping: dict[int, int], color: int, steps: int) -> int:
    out = int(color)
    for _ in range(steps % 4):
        out = int(mapping[out])
    return out


def _collect_visible_edges(grid: np.ndarray, mask_color: int) -> set[tuple[int, int]]:
    n = int(grid.shape[0])
    edges: set[tuple[int, int]] = set()
    for r in range(n):
        for c in range(n):
            nr, nc = _rot90_cw(r, c, n)
            src = int(grid[r, c])
            dst = int(grid[nr, nc])
            if src != mask_color and dst != mask_color:
                edges.add((src, dst))
    return edges


def _infer_candidate(grid: np.ndarray, mask_color: int) -> tuple[dict[int, int], np.ndarray] | None:
    n = int(grid.shape[0])
    colors = {int(color) for color in np.unique(grid) if int(color) != mask_color}
    if not colors:
        return None

    mapping: dict[int, int] = {}
    inverse: dict[int, int] = {}
    for r in range(n):
        for c in range(n):
            nr, nc = _rot90_cw(r, c, n)
            src = int(grid[r, c])
            dst = int(grid[nr, nc])
            if src == mask_color or dst == mask_color:
                continue
            if src in mapping and mapping[src] != dst:
                return None
            if dst in inverse and inverse[dst] != src:
                return None
            mapping[src] = dst
            inverse[dst] = src

    if set(mapping) != colors or set(inverse) != colors:
        return None

    if all(mapping[color] == color for color in colors):
        return None

    for color in colors:
        if _apply_mapping(mapping, color, 4) != color:
            return None

    out = np.asarray(grid, dtype=int).copy()
    for orbit in _rotation_orbits(n):
        visible = [(i, int(grid[r, c])) for i, (r, c) in enumerate(orbit) if int(grid[r, c]) != mask_color]
        if any(int(grid[r, c]) == mask_color for r, c in orbit) and not visible:
            return None
        if not visible:
            continue

        base_i, base_color = visible[0]
        for i, color in visible[1:]:
            if _apply_mapping(mapping, base_color, i - base_i) != color:
                return None

        for i, (r, c) in enumerate(orbit):
            expected = _apply_mapping(mapping, base_color, i - base_i)
            if int(grid[r, c]) == mask_color:
                out[r, c] = expected
            elif int(grid[r, c]) != expected:
                return None

    if np.any(out == mask_color):
        return None
    return mapping, out


def _infer_unique_solution(grid: np.ndarray) -> tuple[int, dict[int, int], np.ndarray]:
    candidates: list[tuple[int, dict[int, int], np.ndarray]] = []
    for color in map(int, np.unique(grid)):
        inferred = _infer_candidate(grid, color)
        if inferred is not None:
            mapping, out = inferred
            candidates.append((color, mapping, out))
    assert len(candidates) == 1, f"expected one mask/permutation pair, got {len(candidates)}"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore the masked cells by combining quarter-turn rotation with a global color permutation.

    Input structure:
    - The grid is square and contains exactly one mask color that does not belong to the hidden pattern.
    - Beneath the mask, the full pattern is invariant under a clockwise 90-degree rotation followed by
      one application of the same global color permutation.
    - Each masked cell belongs to a four-position rotation orbit with at least one visible orbit mate.

    Rule:
    - Infer the unique mask color and the unique color permutation that make every visible clockwise
      rotation pair consistent.
    - For each masked cell, locate a visible cell in the same rotation orbit and apply the permutation
      the required number of times according to the quarter-turn distance between them.
    - Leave all visible cells unchanged.

    Key insight:
    - The pattern is not plain rotational symmetry: rotation changes colors in a fixed cycle, so repairing
      a hidden cell requires both orbit geometry and the correctly oriented color permutation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    rows, cols = map(int, g.shape)
    assert rows == cols, "expected a square grid"

    _mask_color, _mapping, out = _infer_unique_solution(g)
    return out


_GRID_SIDES = (8, 10, 12, 14, 16)
_KIND_WEIGHTS = np.array([0.55, 0.30, 0.15], dtype=float)
_GEN_RETRIES = 400


def _sample_kind() -> str:
    kinds = np.array(["cycle4", "cycle4_swap", "swapswap"], dtype=object)
    pick = int(np.random.choice(len(kinds), p=_KIND_WEIGHTS))
    return str(kinds[pick])


def _sample_permutation() -> tuple[dict[int, int], list[tuple[int, ...]], int]:
    palette = np.random.permutation(10).astype(int).tolist()
    kind = _sample_kind()
    if kind == "cycle4":
        active = palette[:4]
        mask_color = int(palette[4])
        cycles = [tuple(active)]
    elif kind == "cycle4_swap":
        active = palette[:6]
        mask_color = int(palette[6])
        cycles = [tuple(active[:4]), tuple(active[4:6])]
    else:
        active = palette[:4]
        mask_color = int(palette[4])
        cycles = [tuple(active[:2]), tuple(active[2:4])]

    mapping: dict[int, int] = {}
    for cycle in cycles:
        for i, color in enumerate(cycle):
            mapping[int(color)] = int(cycle[(i + 1) % len(cycle)])
    return mapping, cycles, mask_color


def _has_four_cycle(cycles: list[tuple[int, ...]]) -> bool:
    return any(len(cycle) == 4 for cycle in cycles)


def _choose_weighted_start_colors(count: int, cycles: list[tuple[int, ...]]) -> np.ndarray:
    colors = np.array([color for cycle in cycles for color in cycle], dtype=int)
    weights = np.ones(len(colors), dtype=float)
    if _has_four_cycle(cycles):
        for i, color in enumerate(colors):
            if any(color in cycle and len(cycle) == 4 for cycle in cycles):
                weights[i] = 1.6
    weights /= weights.sum()
    return np.random.choice(colors, size=count, replace=True, p=weights)


def _build_output(
    n: int,
    orbits: list[Orbit],
    starts: np.ndarray,
    mapping: dict[int, int],
) -> np.ndarray:
    out = np.zeros((n, n), dtype=int)
    for orbit, start_color in zip(orbits, starts):
        start_color = int(start_color)
        for i, (r, c) in enumerate(orbit):
            out[r, c] = _apply_mapping(mapping, start_color, i)
    return out


def _pick_cycle_color(cycle: tuple[int, ...]) -> int:
    return int(cycle[int(np.random.randint(0, len(cycle)))])


def _mask_positions_in_orbit(inp: np.ndarray, orbit: Orbit, mask_color: int, masked_indices: list[int]) -> None:
    for idx in masked_indices:
        r, c = orbit[idx]
        inp[r, c] = mask_color


def _generate_instance() -> tuple[np.ndarray, np.ndarray, int, dict[int, int], list[tuple[int, ...]], list[Orbit]]:
    n = int(np.random.choice(_GRID_SIDES))
    orbits = _rotation_orbits(n)
    mapping, cycles, mask_color = _sample_permutation()

    orbit_count = len(orbits)
    starts = _choose_weighted_start_colors(orbit_count, cycles)

    active_colors = [int(color) for cycle in cycles for color in cycle]
    force_positions = np.random.permutation(orbit_count)[: len(active_colors)]
    for idx, color in zip(force_positions, active_colors):
        starts[int(idx)] = int(color)

    cycle_to_clue: list[tuple[tuple[int, ...], int]] = []
    available = list(np.random.permutation(orbit_count).astype(int))
    for cycle in cycles:
        clue_orbit = int(available.pop())
        starts[clue_orbit] = int(cycle[0])
        cycle_to_clue.append((cycle, clue_orbit))

    directional_cycle = next((cycle for cycle in cycles if len(cycle) == 4), cycles[0])
    challenge_count = 2 if len(orbits) >= 20 else 1
    challenge_orbits: list[int] = []
    while len(challenge_orbits) < challenge_count:
        candidate = int(available.pop())
        challenge_orbits.append(candidate)
        starts[candidate] = _pick_cycle_color(directional_cycle)

    output = _build_output(n, orbits, starts, mapping)
    assert not np.array_equal(output, np.rot90(output, -1))
    assert not np.array_equal(output, np.rot90(output, 1))

    inp = output.copy()
    reserved = {orbit_idx for _cycle, orbit_idx in cycle_to_clue}

    for orbit_idx in challenge_orbits:
        orbit = orbits[orbit_idx]
        visible_idx = int(np.random.randint(0, len(orbit)))
        masked = [i for i in range(len(orbit)) if i != visible_idx]
        _mask_positions_in_orbit(inp, orbit, mask_color, masked)
        reserved.add(orbit_idx)

    extra_pool = [idx for idx in available if idx not in reserved]
    np.random.shuffle(extra_pool)
    extra_orbit_cap = min(len(extra_pool), max(4, orbit_count // 3))
    extra_orbit_count = int(np.random.randint(max(3, extra_orbit_cap // 2), extra_orbit_cap + 1))
    for orbit_idx in extra_pool[:extra_orbit_count]:
        orbit = orbits[orbit_idx]
        mask_count = int(np.random.choice([1, 1, 2, 2, 3]))
        masked = np.random.choice(len(orbit), size=mask_count, replace=False)
        _mask_positions_in_orbit(inp, orbit, mask_color, masked.astype(int).tolist())

    return inp, output, mask_color, mapping, cycles, orbits


def _validate_semantics(
    inp: np.ndarray,
    out: np.ndarray,
    mask_color: int,
    mapping: dict[int, int],
    cycles: list[tuple[int, ...]],
    orbits: list[Orbit],
) -> bool:
    try:
        inferred_mask, inferred_mapping, solved = _infer_unique_solution(inp)
    except AssertionError:
        return False

    if inferred_mask != mask_color:
        return False
    if inferred_mapping != mapping:
        return False
    if not np.array_equal(solved, out):
        return False
    if np.array_equal(inp, out):
        return False
    if np.any(out == mask_color):
        return False
    if np.array_equal(out, np.rot90(out, -1)) or np.array_equal(out, np.rot90(out, 1)):
        return False

    required_edges = {(color, mapping[color]) for color in mapping}
    if not required_edges.issubset(_collect_visible_edges(inp, mask_color)):
        return False

    masked_orbit_count = 0
    nontrivial_masked_orbit = False
    needs_far_step = False
    has_three_mask_orbit = False
    for orbit in orbits:
        orbit_in = [int(inp[r, c]) for r, c in orbit]
        orbit_out = [int(out[r, c]) for r, c in orbit]
        mask_positions = [i for i, color in enumerate(orbit_in) if color == mask_color]
        if not mask_positions:
            continue
        masked_orbit_count += 1
        if len(set(orbit_out)) > 1:
            nontrivial_masked_orbit = True
        if len(mask_positions) >= 3:
            has_three_mask_orbit = True
        visible_positions = [i for i, color in enumerate(orbit_in) if color != mask_color]
        if len(visible_positions) == 1:
            vis = visible_positions[0]
            if any(((idx - vis) % 4) in (2, 3) for idx in mask_positions):
                needs_far_step = True

    if masked_orbit_count < 4:
        return False
    if not nontrivial_masked_orbit:
        return False
    if not has_three_mask_orbit:
        return False
    if not needs_far_step and _has_four_cycle(cycles):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        inp, out, mask_color, mapping, cycles, orbits = _generate_instance()
        if _validate_semantics(inp, out, mask_color, mapping, cycles, orbits):
            return inp
    raise RuntimeError("failed to generate a valid rotation-composed color-cycle puzzle")
