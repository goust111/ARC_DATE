from __future__ import annotations

import random
from itertools import combinations

import numpy as np


Grid = np.ndarray

ALL_COLORS = tuple(range(10))
GEN_QUAD_SIDE_RANGE = (5, 11)
GEN_SHAPE_COUNT_RANGE = (4, 7)
GEN_MAX_DEFECTS = 3
GEN_MAX_ATTEMPTS = 400


def _dihedral_variants(grid: Grid) -> list[Grid]:
    variants: list[Grid] = []
    for k in range(4):
        rotated = np.rot90(grid, k)
        variants.append(rotated.copy())
        variants.append(np.fliplr(rotated).copy())
    return variants


def _freeze(grid: Grid) -> bytes:
    arr = np.ascontiguousarray(grid)
    return arr.shape[0].to_bytes(2, "little") + arr.shape[1].to_bytes(2, "little") + arr.tobytes()


def _hamming(a: Grid, b: Grid) -> int:
    return int(np.count_nonzero(a != b))


def _extract_quadrants(grid: Grid) -> list[Grid]:
    h, w = map(int, grid.shape)
    assert h % 2 == 1 and w % 2 == 1, "expected odd grid dimensions"
    mid_r, mid_c = h // 2, w // 2
    sep = int(grid[mid_r, mid_c])
    assert np.all(grid[mid_r, :] == sep), "expected a separator row"
    assert np.all(grid[:, mid_c] == sep), "expected a separator column"
    quadrants = [
        grid[:mid_r, :mid_c],
        grid[:mid_r, mid_c + 1 :],
        grid[mid_r + 1 :, :mid_c],
        grid[mid_r + 1 :, mid_c + 1 :],
    ]
    assert len({q.shape for q in quadrants}) == 1, "expected equal-sized quadrants"
    return [q.copy() for q in quadrants]


def _min_dihedral_distance(a: Grid, b: Grid) -> int:
    return min(_hamming(a, variant) for variant in _dihedral_variants(b))


def _find_defective_index(quadrants: list[Grid]) -> int:
    zero_neighbors = [0] * len(quadrants)
    for i, j in combinations(range(len(quadrants)), 2):
        dist = _min_dihedral_distance(quadrants[i], quadrants[j])
        if dist == 0:
            zero_neighbors[i] += 1
            zero_neighbors[j] += 1
    candidates = [idx for idx, count in enumerate(zero_neighbors) if count == 0]
    assert len(candidates) == 1, "expected exactly one defective quadrant"
    return int(candidates[0])


def _repair_from_clean_reference(clean: Grid, defective: Grid) -> Grid:
    best_dist: int | None = None
    best_variants: dict[bytes, Grid] = {}
    for variant in _dihedral_variants(clean):
        dist = _hamming(variant, defective)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_variants = {_freeze(variant): variant.copy()}
        elif dist == best_dist:
            best_variants.setdefault(_freeze(variant), variant.copy())
    assert best_dist is not None and best_dist > 0, "defective quadrant must differ from the clean consensus"
    assert len(best_variants) == 1, "repair orientation must be uniquely determined"
    return next(iter(best_variants.values()))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the one defective quadrant by using the dihedral consensus of the other three copies.

    Input structure:
    - The grid has odd height and width, and the middle row and middle column are a background separator cross.
    - The separator cross divides the canvas into four equal square quadrants.
    - Each quadrant is a rotated or reflected copy of the same motif, except that exactly one copy has a few wrong cells.

    Rule:
    - Compare quadrants up to the eight dihedral transforms.
    - The three clean quadrants can be matched perfectly to one another, while the defective quadrant has a small positive mismatch against every clean copy.
    - Identify the defective quadrant, align a clean copy into that quadrant's orientation, and replace the wrong cells with the consensus content.

    Key insight:
    - Raw quadrant arrays can all look different, but under rotation/reflection equivalence three of them collapse to one exact pattern and expose the damaged outlier.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    quadrants = _extract_quadrants(g)
    defective_idx = _find_defective_index(quadrants)
    clean_indices = [idx for idx in range(4) if idx != defective_idx]
    for i, j in combinations(clean_indices, 2):
        assert _min_dihedral_distance(quadrants[i], quadrants[j]) == 0, "clean quadrants must agree up to symmetry"

    clean_reference = quadrants[clean_indices[0]]
    defective = quadrants[defective_idx]
    return _repair_from_clean_reference(clean_reference, defective)


def _paint_shape(canvas: Grid, color: int) -> None:
    n = int(canvas.shape[0])
    kind = random.choice(("rect", "hline", "vline", "ell", "carve"))
    if kind == "rect":
        h = random.randint(1, max(1, n // 2))
        w = random.randint(1, max(1, n // 2))
        r0 = random.randint(0, n - h)
        c0 = random.randint(0, n - w)
        canvas[r0 : r0 + h, c0 : c0 + w] = color
        return
    if kind == "hline":
        r = random.randint(0, n - 1)
        length = random.randint(2, n)
        c0 = random.randint(0, n - length)
        canvas[r, c0 : c0 + length] = color
        return
    if kind == "vline":
        c = random.randint(0, n - 1)
        length = random.randint(2, n)
        r0 = random.randint(0, n - length)
        canvas[r0 : r0 + length, c] = color
        return
    if kind == "ell":
        h = random.randint(2, min(4, n))
        w = random.randint(2, min(4, n))
        r0 = random.randint(0, n - h)
        c0 = random.randint(0, n - w)
        orientation = random.randint(0, 3)
        if orientation == 0:
            canvas[r0, c0 : c0 + w] = color
            canvas[r0 : r0 + h, c0] = color
        elif orientation == 1:
            canvas[r0, c0 : c0 + w] = color
            canvas[r0 : r0 + h, c0 + w - 1] = color
        elif orientation == 2:
            canvas[r0 + h - 1, c0 : c0 + w] = color
            canvas[r0 : r0 + h, c0] = color
        else:
            canvas[r0 + h - 1, c0 : c0 + w] = color
            canvas[r0 : r0 + h, c0 + w - 1] = color
        return

    h = random.randint(1, max(1, n // 3))
    w = random.randint(1, max(1, n // 3))
    r0 = random.randint(0, n - h)
    c0 = random.randint(0, n - w)
    canvas[r0 : r0 + h, c0 : c0 + w] = color


def _make_base_pattern(side: int, bg: int, palette: list[int]) -> Grid | None:
    base = np.full((side, side), bg, dtype=int)
    shape_count = random.randint(*GEN_SHAPE_COUNT_RANGE)
    for _ in range(shape_count):
        if random.random() < 0.2:
            _paint_shape(base, bg)
        _paint_shape(base, random.choice(palette))

    non_bg = int(np.count_nonzero(base != bg))
    if non_bg < side * side // 4 or non_bg > (3 * side * side) // 4:
        return None
    if len(set(map(int, np.unique(base))) - {bg}) < 2:
        return None

    variants = _dihedral_variants(base)
    unique = {_freeze(variant) for variant in variants}
    if len(unique) != 8:
        return None
    min_dist = min(_hamming(a, b) for a, b in combinations(variants, 2))
    if min_dist <= 2 * GEN_MAX_DEFECTS:
        return None
    return base


def _inject_defects(grid: Grid) -> Grid:
    out = grid.copy()
    side = int(grid.shape[0])
    defect_count = random.randint(1, GEN_MAX_DEFECTS)
    cells = random.sample(range(side * side), defect_count)
    for cell in cells:
        r, c = divmod(cell, side)
        old = int(out[r, c])
        choices = [color for color in ALL_COLORS if color != old]
        out[r, c] = random.choice(choices)
    return out


def _build_full_grid(quadrants: list[Grid], bg: int) -> Grid:
    side = int(quadrants[0].shape[0])
    grid = np.full((2 * side + 1, 2 * side + 1), bg, dtype=int)
    grid[:side, :side] = quadrants[0]
    grid[:side, side + 1 :] = quadrants[1]
    grid[side + 1 :, :side] = quadrants[2]
    grid[side + 1 :, side + 1 :] = quadrants[3]
    return grid


def _gen_attempt() -> tuple[Grid, Grid] | None:
    side = random.randint(*GEN_QUAD_SIDE_RANGE)
    bg = random.choice(ALL_COLORS)
    palette_size = random.randint(2, 4)
    palette = random.sample([color for color in ALL_COLORS if color != bg], palette_size)
    base = _make_base_pattern(side, bg, palette)
    if base is None:
        return None

    variant_indices = random.sample(range(8), 4)
    clean_quadrants = [_dihedral_variants(base)[idx] for idx in variant_indices]
    defective_idx = random.randrange(4)
    expected = clean_quadrants[defective_idx].copy()
    corrupted = _inject_defects(expected)
    quadrants = [q.copy() for q in clean_quadrants]
    quadrants[defective_idx] = corrupted

    raw_unique = {_freeze(q) for q in quadrants}
    if len(raw_unique) != 4:
        return None

    grid = _build_full_grid(quadrants, bg)
    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        pair = _gen_attempt()
        if pair is not None:
            return pair[0]
    raise RuntimeError("failed to generate a valid input for puzzle 3")
