from __future__ import annotations

from collections import Counter

import numpy as np


BACKGROUND = 0
ARC_COLORS = tuple(range(1, 10))
GENERATION_ATTEMPTS = 700

TRANSFORM_CYCLE = (
    "rot90",
    "mirror_lr",
    "rot180",
    "mirror_ud",
    "rot270",
    "mirror_main_diag",
    "mirror_anti_diag",
)

_GEN_COUNTER = 0


def _d4(tile: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Return the square tile under the eight dihedral orientations."""
    a = np.asarray(tile, dtype=int)
    return [
        ("identity", a.copy()),
        ("rot90", np.rot90(a, 1).copy()),
        ("rot180", np.rot90(a, 2).copy()),
        ("rot270", np.rot90(a, 3).copy()),
        ("mirror_lr", np.fliplr(a).copy()),
        ("mirror_ud", np.flipud(a).copy()),
        ("mirror_main_diag", a.T.copy()),
        ("mirror_anti_diag", np.fliplr(np.flipud(a)).T.copy()),
    ]


def _transform(tile: np.ndarray, name: str) -> np.ndarray:
    for t_name, transformed in _d4(tile):
        if t_name == name:
            return transformed
    raise ValueError(f"unknown transform: {name}")


def _is_primitive_period(tile: np.ndarray) -> bool:
    side = int(tile.shape[0])
    for p in range(1, side):
        if side % p != 0:
            continue
        repeated = np.tile(tile[:p, :p], (side // p, side // p))
        if np.array_equal(repeated, tile):
            return False
    return True


def _has_no_dihedral_symmetry(tile: np.ndarray) -> bool:
    base = np.asarray(tile, dtype=int)
    return all(not np.array_equal(base, t) for name, t in _d4(base) if name != "identity")


def _candidate_sides(height: int, width: int) -> list[int]:
    limit = min(height, width)
    return [
        side
        for side in range(2, limit + 1)
        if height % side == 0 and width % side == 0 and (height // side) * (width // side) >= 3
    ]


def _majority_tile(grid: np.ndarray, side: int) -> np.ndarray | None:
    tile = np.empty((side, side), dtype=int)

    for rr in range(side):
        for cc in range(side):
            values = [int(v) for v in grid[rr::side, cc::side].ravel() if int(v) != BACKGROUND]
            if not values:
                return None

            counts = Counter(values)
            common = counts.most_common()
            if len(common) > 1 and common[0][1] == common[1][1]:
                return None
            tile[rr, cc] = int(common[0][0])

    return tile


def _matching_transforms(tile: np.ndarray, patch: np.ndarray) -> list[tuple[str, np.ndarray]]:
    known = patch != BACKGROUND
    if not np.any(known):
        return []

    matches: list[tuple[str, np.ndarray]] = []
    for name, transformed in _d4(tile):
        if name == "identity":
            continue
        if np.all(patch[known] == transformed[known]):
            matches.append((name, transformed))
    return matches


def _solve_for_side(grid: np.ndarray, side: int) -> np.ndarray | None:
    height, width = map(int, grid.shape)
    tile = _majority_tile(grid, side)
    if tile is None:
        return None
    if not _is_primitive_period(tile):
        return None
    if not _has_no_dihedral_symmetry(tile):
        return None

    conflict_blocks: list[tuple[int, int, np.ndarray]] = []
    for br in range(height // side):
        for bc in range(width // side):
            r0 = br * side
            c0 = bc * side
            patch = grid[r0 : r0 + side, c0 : c0 + side]
            known = patch != BACKGROUND
            conflicts = int(np.count_nonzero(known & (patch != tile)))
            if conflicts:
                conflict_blocks.append((br, bc, patch))

    if len(conflict_blocks) != 1:
        return None

    _br, _bc, patch = conflict_blocks[0]
    matches = _matching_transforms(tile, patch)
    if len(matches) != 1:
        return None

    return matches[0][1].copy().astype(int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the one transformed tile instance in a holed periodic tiling and return it repaired.

    Input structure:
    - A rectangular grid is a seamless repetition of one square nonzero color tile.
    - Black pixels are missing-cell holes and may occur in ordinary instances or in the transformed instance.
    - Exactly one aligned tile instance is a whole rotation or mirror image of the standard tile.

    Rule:
    - Try square periods that divide both grid dimensions and infer the standard tile by nonzero majority
      at each aligned residue position.
    - The correct period has one tile block whose visible nonzero pixels conflict with that majority tile.
    - Match that block's visible pixels against the non-identity rotations and mirrors of the inferred tile,
      then output the complete matched transform with holes repaired.

    Key insight:
    - The black cells are missing data, but the anomaly is a whole tile orientation change; repairing it
      requires first voting through the holes and then identifying which orientation that one instance uses.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must be at most 30x30"
    assert np.any(g == BACKGROUND), "expected at least one black hole"

    solutions: list[tuple[int, np.ndarray]] = []
    for side in _candidate_sides(*map(int, g.shape)):
        solved = _solve_for_side(g, side)
        if solved is not None:
            solutions.append((side, solved))

    assert len(solutions) == 1, "expected one unique recoverable transformed tile period"
    solutions.sort(key=lambda item: item[0])
    return solutions[0][1].astype(int)


def _make_tile(rng: np.random.Generator, side: int) -> np.ndarray | None:
    for _ in range(250):
        color_count = int(rng.integers(4, min(8, len(ARC_COLORS)) + 1))
        palette = list(map(int, rng.choice(ARC_COLORS, size=color_count, replace=False)))
        tile = rng.choice(palette, size=(side, side), replace=True).astype(int)

        for i, color in enumerate(palette):
            tile[(2 * i + 1) % side, (3 * i + 2) % side] = int(color)

        tile[0, 0] = palette[0]
        tile[0, 1 % side] = palette[1]
        tile[1 % side, 0] = palette[2]
        tile[side - 1, side - 1] = palette[3]

        if len(set(map(int, tile.ravel()))) < 4:
            continue
        if not _is_primitive_period(tile):
            continue
        if not _has_no_dihedral_symmetry(tile):
            continue

        diffs = [
            int(np.count_nonzero(tile != transformed))
            for name, transformed in _d4(tile)
            if name != "identity"
        ]
        if min(diffs) < max(4, side):
            continue
        return tile

    return None


def _strict_majority_witnesses(
    grid: np.ndarray,
    tile: np.ndarray,
    anomaly_block: tuple[int, int],
) -> bool:
    side = int(tile.shape[0])
    rows = grid.shape[0] // side
    cols = grid.shape[1] // side

    for rr in range(side):
        for cc in range(side):
            values: list[int] = []
            standard_visible = 0
            for br in range(rows):
                for bc in range(cols):
                    value = int(grid[br * side + rr, bc * side + cc])
                    if value == BACKGROUND:
                        continue
                    values.append(value)
                    if (br, bc) != anomaly_block:
                        if value != int(tile[rr, cc]):
                            return False
                        standard_visible += 1

            if standard_visible < 2:
                return False
            counts = Counter(values).most_common()
            if not counts or counts[0][0] != int(tile[rr, cc]):
                return False
            if len(counts) > 1 and counts[0][1] <= counts[1][1]:
                return False

    return True


def _semantic_requirements(
    grid: np.ndarray,
    tile: np.ndarray,
    expected: np.ndarray,
    anomaly_block: tuple[int, int],
    transform_name: str,
) -> bool:
    side = int(tile.shape[0])
    br, bc = anomaly_block
    patch = grid[br * side : (br + 1) * side, bc * side : (bc + 1) * side]
    known = patch != BACKGROUND

    if not np.any(patch == BACKGROUND):
        return False
    if not np.any(grid == BACKGROUND):
        return False
    outside_holes = int(np.count_nonzero(grid == BACKGROUND) - np.count_nonzero(patch == BACKGROUND))
    if outside_holes <= 0:
        return False
    if int(np.count_nonzero(known)) < max(5, side):
        return False
    if int(np.count_nonzero(known & (patch != tile))) < max(2, side // 2):
        return False
    if not np.any((patch == BACKGROUND) & (expected != BACKGROUND)):
        return False
    if np.array_equal(expected, tile):
        return False
    if np.array_equal(expected, patch):
        return False
    if not _strict_majority_witnesses(grid, tile, anomaly_block):
        return False

    matches = _matching_transforms(tile, patch)
    if len(matches) != 1 or matches[0][0] != transform_name:
        return False

    try:
        solved = main(grid)
    except AssertionError:
        return False
    return bool(np.array_equal(solved, expected))


def _gen_attempt(rng: np.random.Generator, preferred_transform: str) -> np.ndarray | None:
    side = int(rng.integers(4, 8))
    max_repeats = 30 // side
    if max_repeats < 3:
        return None

    rows = int(rng.integers(3, max_repeats + 1))
    cols = int(rng.integers(3, max_repeats + 1))
    if rows * cols < 7:
        return None

    tile = _make_tile(rng, side)
    if tile is None:
        return None

    transform_name = preferred_transform
    expected = _transform(tile, transform_name)
    if np.array_equal(expected, tile):
        return None

    grid = np.tile(tile, (rows, cols)).astype(int)
    br = int(rng.integers(0, rows))
    bc = int(rng.integers(0, cols))
    grid[br * side : (br + 1) * side, bc * side : (bc + 1) * side] = expected

    hole_rate = float(rng.uniform(0.07, 0.18))
    holes = rng.random(grid.shape) < hole_rate

    ar0, ac0 = br * side, bc * side
    anomaly_slice = (slice(ar0, ar0 + side), slice(ac0, ac0 + side))
    if not np.any(holes[anomaly_slice]):
        rr = int(rng.integers(ar0, ar0 + side))
        cc = int(rng.integers(ac0, ac0 + side))
        holes[rr, cc] = True

    outside = np.ones(grid.shape, dtype=bool)
    outside[anomaly_slice] = False
    if not np.any(holes & outside):
        choices = np.argwhere(outside)
        rr, cc = map(int, choices[int(rng.integers(0, len(choices)))])
        holes[rr, cc] = True

    grid_with_holes = grid.copy()
    grid_with_holes[holes] = BACKGROUND

    if not _semantic_requirements(grid_with_holes, tile, expected, (br, bc), transform_name):
        return None
    return grid_with_holes.astype(int)


def generate_input() -> np.ndarray:
    global _GEN_COUNTER

    rng = np.random.default_rng()
    preferred = TRANSFORM_CYCLE[_GEN_COUNTER % len(TRANSFORM_CYCLE)]
    _GEN_COUNTER += 1

    for attempt in range(GENERATION_ATTEMPTS):
        transform_name = preferred
        if attempt > GENERATION_ATTEMPTS // 2:
            transform_name = str(rng.choice(TRANSFORM_CYCLE))
        grid = _gen_attempt(rng, transform_name)
        if grid is not None:
            return grid

    raise RuntimeError("failed to generate a valid transformed-tile tiling input")
