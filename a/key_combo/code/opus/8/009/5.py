from __future__ import annotations

import numpy as np


MAX_ATTEMPTS = 400
PALETTE = np.arange(1, 10, dtype=int)


def _upper_triangle_pairs(size: int) -> list[tuple[int, int]]:
    rows, cols = np.triu_indices(size, k=1)
    return [(int(r), int(c)) for r, c in zip(rows, cols)]


def _asymmetry_score(tile: np.ndarray) -> int:
    size = int(tile.shape[0])
    rows, cols = np.triu_indices(size, k=1)
    return int(np.count_nonzero(tile[rows, cols] != tile[cols, rows]))


def _stack_tiles(tiles: list[np.ndarray], tile_size: int, dtype: np.dtype) -> np.ndarray:
    if not tiles:
        return np.zeros((0, tile_size), dtype=dtype)
    return np.concatenate(tiles, axis=0)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Filter vertically stacked square tiles by main-diagonal symmetry, then sort the asymmetric ones.

    Input structure:
    - The input is a tall grid made of equal-sized square tiles stacked vertically
      with no separator rows.
    - Each tile uses black background plus one foreground color.
    - Some tiles are perfectly symmetric across the main diagonal and some are not.

    Rule:
    - Split the grid into consecutive square tiles whose side length equals the grid width.
    - For each tile, count how many upper-triangle position pairs (r,c) with r<c
      disagree with their transposed partners (c,r).
    - Discard tiles with zero disagreements.
    - Sort the remaining tiles by that disagreement count in ascending order and
      stack them vertically.

    Key insight:
    - The same diagonal-symmetry test serves two roles at once: zero score means
      the tile is removed, while positive scores determine the output order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    height, width = map(int, g.shape)
    assert height % width == 0, "input must be a vertical stack of square tiles"
    tile_size = width
    n_tiles = height // tile_size
    assert n_tiles >= 1, "expected at least one tile"

    scored_tiles: list[tuple[int, int, np.ndarray]] = []
    for idx in range(n_tiles):
        r0 = idx * tile_size
        tile = g[r0 : r0 + tile_size, :].copy()
        score = _asymmetry_score(tile)
        if score > 0:
            scored_tiles.append((score, idx, tile))

    scored_tiles.sort(key=lambda item: (item[0], item[1]))
    ordered_tiles = [tile for _score, _idx, tile in scored_tiles]
    return _stack_tiles(ordered_tiles, tile_size, g.dtype)


def _make_symmetric_tile(size: int, color: int, rng: np.random.Generator) -> np.ndarray:
    pairs = _upper_triangle_pairs(size)
    diag = np.arange(size)

    for _ in range(200):
        tile = np.zeros((size, size), dtype=int)

        diag_fill = rng.random(size) < rng.uniform(0.35, 0.8)
        if not diag_fill.any():
            diag_fill[int(rng.integers(0, size))] = True
        tile[diag[diag_fill], diag[diag_fill]] = color

        pair_fill = rng.random(len(pairs)) < rng.uniform(0.15, 0.55)
        for filled, (r, c) in zip(pair_fill, pairs):
            if filled:
                tile[r, c] = color
                tile[c, r] = color

        fg = int(np.count_nonzero(tile))
        if 0 < fg < size * size and _asymmetry_score(tile) == 0:
            return tile

    raise RuntimeError("failed to sample a symmetric tile")


def _make_asymmetric_tile(size: int, color: int, score: int, rng: np.random.Generator) -> np.ndarray:
    pairs = _upper_triangle_pairs(size)
    diag = np.arange(size)
    assert 1 <= score <= len(pairs)

    for _ in range(200):
        tile = np.zeros((size, size), dtype=int)

        diag_fill = rng.random(size) < rng.uniform(0.25, 0.75)
        if not diag_fill.any() and rng.random() < 0.6:
            diag_fill[int(rng.integers(0, size))] = True
        tile[diag[diag_fill], diag[diag_fill]] = color

        mismatch_ids = set(int(x) for x in rng.choice(len(pairs), size=score, replace=False))
        remaining_ids = [i for i in range(len(pairs)) if i not in mismatch_ids]
        symmetric_fill = rng.random(len(remaining_ids)) < rng.uniform(0.05, 0.45)
        for filled, pair_id in zip(symmetric_fill, remaining_ids):
            if filled:
                r, c = pairs[pair_id]
                tile[r, c] = color
                tile[c, r] = color

        for pair_id in mismatch_ids:
            r, c = pairs[pair_id]
            if bool(rng.integers(0, 2)):
                tile[r, c] = color
                tile[c, r] = 0
            else:
                tile[r, c] = 0
                tile[c, r] = color

        fg = int(np.count_nonzero(tile))
        if 0 < fg < size * size and _asymmetry_score(tile) == score:
            return tile

    raise RuntimeError("failed to sample an asymmetric tile")


def _build_case(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    tile_size = int(rng.integers(4, 6))
    n_tiles = int(rng.integers(5, 7))
    max_pairs = tile_size * (tile_size - 1) // 2

    n_asym = int(rng.integers(2, n_tiles - 1))
    n_sym = n_tiles - n_asym
    if n_sym < 2:
        return None

    colors = [int(c) for c in rng.choice(PALETTE, size=n_tiles, replace=False)]
    asym_scores = sorted(int(x) for x in rng.choice(np.arange(1, max_pairs + 1), size=n_asym, replace=False))

    if n_asym >= 2:
        perm = rng.permutation(n_asym)
        if np.array_equal(perm, np.arange(n_asym)):
            perm = perm[::-1]
        asym_scores_in_input = [asym_scores[int(i)] for i in perm]
    else:
        asym_scores_in_input = asym_scores[:]

    asym_tiles = [
        _make_asymmetric_tile(tile_size, colors[i], score, rng)
        for i, score in enumerate(asym_scores_in_input)
    ]
    sym_tiles = [
        _make_symmetric_tile(tile_size, colors[n_asym + i], rng)
        for i in range(n_sym)
    ]

    labels = ["A"] * n_asym + ["S"] * n_sym
    rng.shuffle(labels)

    asym_iter = iter(zip(asym_scores_in_input, asym_tiles))
    sym_iter = iter(sym_tiles)
    input_tiles: list[np.ndarray] = []
    input_scores: list[int] = []

    for label in labels:
        if label == "A":
            score, tile = next(asym_iter)
            input_tiles.append(tile)
            input_scores.append(score)
        else:
            input_tiles.append(next(sym_iter))

    if input_scores == sorted(input_scores):
        return None

    inp = np.concatenate(input_tiles, axis=0)
    expected_tiles = [tile for _score, tile in sorted(zip(asym_scores_in_input, asym_tiles), key=lambda x: x[0])]
    out = np.concatenate(expected_tiles, axis=0)

    if inp.shape[0] > 30 or inp.shape[1] > 30 or out.shape[0] > 30 or out.shape[1] > 30:
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        rng = np.random.default_rng()
        case = _build_case(rng)
        if case is None:
            continue
        inp, expected = case
        actual = main(inp)
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid input")
