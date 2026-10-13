from __future__ import annotations

from collections import Counter

import numpy as np


GRID_SIDE = 15
QUAD_SIDE = 7
SEPARATOR_COLOR = 5
BG = 0
PREFIX_COLORS = (1, 2, 3, 4, 7)
ALLOWED_PREFIX_LENGTHS = (3, 4)
MAX_GENERATION_ATTEMPTS = 500


def _quadrants(grid: np.ndarray) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h % 2 == 1 and w % 2 == 1, "expected odd grid dimensions"
    mid_r, mid_c = h // 2, w // 2
    assert np.all(g[mid_r, :] == SEPARATOR_COLOR), "expected a grey separator row"
    assert np.all(g[:, mid_c] == SEPARATOR_COLOR), "expected a grey separator column"
    return [
        g[:mid_r, :mid_c],
        g[:mid_r, mid_c + 1 :],
        g[mid_r + 1 :, :mid_c],
        g[mid_r + 1 :, mid_c + 1 :],
    ]


def _array_key(arr: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in np.asarray(arr, dtype=int))


def _find_unique_quadrant(quadrants: list[np.ndarray]) -> int:
    keys = [_array_key(q) for q in quadrants]
    counts = Counter(keys)
    unique_indices = [idx for idx, key in enumerate(keys) if counts[key] == 1]
    assert len(unique_indices) == 1, "expected exactly one unique quadrant"
    return int(unique_indices[0])


def _row_prefix_length(row: np.ndarray) -> int:
    row = np.asarray(row, dtype=int)
    assert row.ndim == 1 and row.size > 0
    assert int(row[0]) != BG, "each row must start with a non-black prefix"

    prefix_len = 0
    while prefix_len < row.size and int(row[prefix_len]) != BG:
        prefix_len += 1

    assert prefix_len >= 3, "prefix must be at least 3 cells long"
    assert prefix_len <= (row.size + 1) // 2, "prefix must fit the closure extension"
    assert np.all(row[prefix_len:] == BG), "cells after the prefix must be black"
    assert int(row[0]) == int(row[prefix_len - 1]), "prefix must be closed by the same boundary color"
    return int(prefix_len)


def _extend_patch_rows(patch: np.ndarray) -> np.ndarray:
    patch = np.asarray(patch, dtype=int)
    h, w = patch.shape
    out = patch.copy()
    for r in range(h):
        prefix_len = _row_prefix_length(patch[r])
        prefix = patch[r, :prefix_len].copy()
        suffix_len = prefix_len - 1
        fill_stop = w - suffix_len
        out[r, prefix_len:fill_stop] = int(prefix[0])
        out[r, fill_stop:] = prefix[1:]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the odd quadrant and close-extend every left prefix row inside it.

    Input structure:
    - The input is an odd-sized grid split by a grey separator cross into four equal quadrants.
    - Three quadrants are pixel-identical copies of one pattern.
    - The remaining quadrant is the odd one out, and each of its rows starts with a contiguous non-black prefix followed only by black cells.
    - In every odd-quadrant row, the first and last cells of that prefix share the same boundary color.

    Rule:
    - Remove the separator row and column conceptually and compare the four quadrants as full arrays.
    - Select the unique quadrant that appears once while the other three match each other.
    - For each row of that quadrant, keep the left prefix in place, fill the gap before the copied suffix with the prefix boundary color, and write the prefix without its first cell into the right end of the row.
    - Output the transformed odd quadrant only.

    Key insight:
    - The target is not determined by local row structure alone; it is the quadrant singled out by the three-versus-one comparison, and only then does the row-wise closure extension apply.
    """
    quadrants = _quadrants(np.asarray(grid, dtype=int))
    unique_idx = _find_unique_quadrant(quadrants)
    return _extend_patch_rows(quadrants[unique_idx])


def _make_prefix_row(rng: np.random.Generator) -> tuple[np.ndarray, int]:
    prefix_len = int(rng.choice(ALLOWED_PREFIX_LENGTHS))
    boundary = int(rng.choice(PREFIX_COLORS))
    interior_choices = [color for color in PREFIX_COLORS if color != boundary]
    interior = rng.choice(interior_choices, size=prefix_len - 2, replace=True)
    prefix = np.array([boundary, *map(int, interior), boundary], dtype=int)
    row = np.zeros(QUAD_SIDE, dtype=int)
    row[:prefix_len] = prefix
    return row, prefix_len


def _make_patch(rng: np.random.Generator) -> np.ndarray | None:
    rows: list[np.ndarray] = []
    lengths: list[int] = []
    for _ in range(QUAD_SIDE):
        row, prefix_len = _make_prefix_row(rng)
        rows.append(row)
        lengths.append(prefix_len)

    patch = np.vstack(rows)
    transformed = _extend_patch_rows(patch)

    if set(lengths) != set(ALLOWED_PREFIX_LENGTHS):
        return None
    if len({_array_key(row[np.newaxis, :]) for row in patch}) < 3:
        return None
    if len({_array_key(row[np.newaxis, :]) for row in transformed}) < 3:
        return None
    if len({int(row[0]) for row in patch}) < 2:
        return None

    interior_colors = set()
    for row in patch:
        prefix_len = _row_prefix_length(row)
        interior_colors.update(int(v) for v in row[1 : prefix_len - 1])
        if any(int(v) == int(row[0]) for v in row[1 : prefix_len - 1]):
            return None
    if len(interior_colors) < 2:
        return None
    if np.array_equal(patch, transformed):
        return None
    return patch


def _hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.sum(np.asarray(a, dtype=int) != np.asarray(b, dtype=int)))


def _assemble_grid(majority: np.ndarray, unique: np.ndarray, unique_slot: int) -> np.ndarray:
    grid = np.full((GRID_SIDE, GRID_SIDE), BG, dtype=int)
    mid = QUAD_SIDE
    grid[mid, :] = SEPARATOR_COLOR
    grid[:, mid] = SEPARATOR_COLOR
    slots = [
        (slice(0, QUAD_SIDE), slice(0, QUAD_SIDE)),
        (slice(0, QUAD_SIDE), slice(mid + 1, GRID_SIDE)),
        (slice(mid + 1, GRID_SIDE), slice(0, QUAD_SIDE)),
        (slice(mid + 1, GRID_SIDE), slice(mid + 1, GRID_SIDE)),
    ]
    for idx, (rs, cs) in enumerate(slots):
        grid[rs, cs] = unique if idx == unique_slot else majority
    return grid


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        g = np.asarray(grid, dtype=int)
        assert g.shape == (GRID_SIDE, GRID_SIDE)
        assert np.all(g[QUAD_SIDE, :] == SEPARATOR_COLOR)
        assert np.all(g[:, QUAD_SIDE] == SEPARATOR_COLOR)
        quadrants = _quadrants(g)
        assert all(q.shape == (QUAD_SIDE, QUAD_SIDE) for q in quadrants)
        keys = [_array_key(q) for q in quadrants]
        counts = Counter(keys)
        assert sorted(counts.values()) == [1, 3]
        unique_idx = _find_unique_quadrant(quadrants)
        unique = quadrants[unique_idx]
        majority = quadrants[(unique_idx + 1) % 4]
        assert all(np.array_equal(q, majority) for i, q in enumerate(quadrants) if i != unique_idx)

        for patch in (majority, unique):
            lengths = [_row_prefix_length(row) for row in patch]
            assert set(lengths) == set(ALLOWED_PREFIX_LENGTHS)
            assert len({tuple(int(v) for v in row) for row in patch}) >= 3
            for row, prefix_len in zip(patch, lengths):
                assert all(int(v) in PREFIX_COLORS for v in row[:prefix_len])
                assert np.all(row[prefix_len:] == BG)
                assert all(int(v) != int(row[0]) for v in row[1 : prefix_len - 1])

        transformed = _extend_patch_rows(unique)
        assert np.array_equal(main(g), transformed)
        assert not np.array_equal(unique, majority)
        assert _hamming(unique, majority) >= 10
        assert not np.array_equal(transformed, unique)
        assert not np.array_equal(transformed, majority)
        assert len({tuple(int(v) for v in row) for row in transformed}) >= 3
        return True
    except AssertionError:
        return False


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    majority = _make_patch(rng)
    unique = _make_patch(rng)
    if majority is None or unique is None:
        return None
    if np.array_equal(majority, unique):
        return None
    if _hamming(majority, unique) < 10:
        return None
    if np.array_equal(_extend_patch_rows(unique), majority):
        return None
    unique_slot = int(rng.integers(4))
    grid = _assemble_grid(majority, unique, unique_slot)
    return grid if _validate_generated(grid) else None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
