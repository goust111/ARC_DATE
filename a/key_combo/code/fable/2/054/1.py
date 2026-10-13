from __future__ import annotations

from collections import Counter
import random

import numpy as np


RED = 2
PALETTE = tuple(range(10))
GEN_ORDER_MIN = 3
GEN_ORDER_MAX = 5
GEN_SCENE_ATTEMPTS = 500
GEN_PATTERN_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _infer_order(side: int) -> int:
    candidates = [n for n in range(2, 31) if n * n + n - 1 == side]
    assert len(candidates) == 1, f"grid side {side} is not of the form n*n+n-1"
    return int(candidates[0])


def _tile_origin(tile_r: int, tile_c: int, n: int) -> tuple[int, int]:
    step = n + 1
    return int(tile_r * step), int(tile_c * step)


def _blank_scene(n: int, bg: int) -> np.ndarray:
    side = n * n + n - 1
    return np.full((side, side), int(bg), dtype=int)


def _parse_scene(grid: np.ndarray) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1], "expected a square grid"

    n = _infer_order(int(g.shape[0]))
    bg = _get_bg(g)

    for k in range(1, n):
        sep = k * (n + 1) - 1
        assert np.all(g[sep, :] == bg), "separator rows must be background"
        assert np.all(g[:, sep] == bg), "separator columns must be background"

    patterns = np.empty((n, n, n, n), dtype=int)
    targets = np.empty((n, n, 2), dtype=int)
    input_tiles = np.empty((n, n, n, n), dtype=int)

    for tile_r in range(n):
        for tile_c in range(n):
            r0, c0 = _tile_origin(tile_r, tile_c, n)
            tile = g[r0 : r0 + n, c0 : c0 + n].copy()
            assert tile.shape == (n, n)

            red_coords = np.argwhere(tile == RED)
            assert red_coords.shape == (1, 2), "each tile must contain exactly one red pointer"
            a, b = map(int, red_coords[0])
            targets[tile_r, tile_c] = (a, b)

            tile[a, b] = bg
            assert not np.any(tile == RED), "motifs must not contain extra red pixels"
            patterns[tile_r, tile_c] = tile

            full_tile = tile.copy()
            full_tile[a, b] = RED
            input_tiles[tile_r, tile_c] = full_tile

    return bg, n, patterns, targets, input_tiles


def _build_from_tiles(tiles: np.ndarray, targets: np.ndarray, bg: int) -> np.ndarray:
    n = int(tiles.shape[0])
    out = _blank_scene(n, bg)
    for tile_r in range(n):
        for tile_c in range(n):
            a, b = map(int, targets[tile_r, tile_c])
            r0, c0 = _tile_origin(tile_r, tile_c, n)
            out[r0 : r0 + n, c0 : c0 + n] = tiles[a, b]
    return out


def _build_output(patterns: np.ndarray, targets: np.ndarray, bg: int) -> np.ndarray:
    return _build_from_tiles(patterns, targets, bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each tile by the motif stored at the array address encoded by its red pointer.

    Input structure:
    - A square grid is partitioned by single background rows and columns into an n x n
      array of n x n tiles.
    - Each tile contains one non-red motif and exactly one red pixel placed on a
      background cell inside that tile.
    - The motifs are pairwise distinct, while the separator rows and columns stay
      background.

    Rule:
    - Read the red pixel position inside each tile as the local coordinate (a, b).
    - Use (a, b) as an address into the tile array and fetch the motif from tile row
      a, tile column b.
    - Replace the whole current tile by that fetched motif only, omitting the fetched
      tile's red pixel, and keep the separator rows and columns unchanged.

    Key insight:
    - The red pixel is an indirect pointer, not a label for the current tile; the
      self-pointing tiles are decoys that only make sense under dereferencing.
    """
    bg, _n, patterns, targets, _input_tiles = _parse_scene(grid)
    return _build_output(patterns, targets, bg)


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _make_pointer_targets(n: int, rng: random.Random) -> np.ndarray:
    coords = [(r, c) for r in range(n) for c in range(n)]

    for _ in range(GEN_SCENE_ATTEMPTS):
        shuffled = coords[:]
        rng.shuffle(shuffled)

        fixed = [src == dst for src, dst in zip(coords, shuffled)]
        has_self = any(fixed)
        has_non_self = not all(fixed)
        has_offdiag_self = any(src == dst and src[0] != src[1] for src, dst in zip(coords, shuffled))
        has_cross_move = any(src[0] != dst[0] and src[1] != dst[1] for src, dst in zip(coords, shuffled))
        if not (has_self and has_non_self and has_offdiag_self and has_cross_move):
            continue

        targets = np.empty((n, n, 2), dtype=int)
        for (tile_r, tile_c), (a, b) in zip(coords, shuffled):
            targets[tile_r, tile_c] = (a, b)
        return targets

    raise RuntimeError("failed to sample a non-degenerate pointer permutation")


def _random_connected_mask(n: int, forbidden: tuple[int, int], rng: random.Random) -> np.ndarray:
    available = [(r, c) for r in range(n) for c in range(n) if (r, c) != forbidden]
    max_cells = min(n + 2, n * n - 2)

    for _ in range(GEN_PATTERN_ATTEMPTS):
        size = rng.randint(2, max_cells)
        start = rng.choice(available)
        mask = np.zeros((n, n), dtype=bool)
        mask[start] = True
        filled = {start}

        while len(filled) < size:
            frontier: set[tuple[int, int]] = set()
            for r, c in filled:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < n and 0 <= nc < n and (nr, nc) != forbidden and not mask[nr, nc]:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cell = rng.choice(tuple(frontier))
            mask[cell] = True
            filled.add(cell)

        if len(filled) != size:
            continue

        if mask[forbidden]:
            continue
        return mask

    raise RuntimeError("failed to sample a connected motif mask")


def _generate_pattern_bank(n: int, fg: int, bg: int, targets: np.ndarray, rng: random.Random) -> np.ndarray | None:
    patterns = np.empty((n, n, n, n), dtype=int)
    used: set[bytes] = set()

    coords = [(r, c) for r in range(n) for c in range(n)]
    rng.shuffle(coords)

    for tile_r, tile_c in coords:
        forbidden = tuple(map(int, targets[tile_r, tile_c]))

        for _ in range(GEN_PATTERN_ATTEMPTS):
            mask = _random_connected_mask(n, forbidden, rng)
            key = mask.tobytes()
            if key in used:
                continue

            tile = np.full((n, n), bg, dtype=int)
            tile[mask] = fg
            patterns[tile_r, tile_c] = tile
            used.add(key)
            break
        else:
            return None

    return patterns


def _assemble_input(n: int, bg: int, patterns: np.ndarray, targets: np.ndarray) -> np.ndarray:
    grid = _blank_scene(n, bg)
    for tile_r in range(n):
        for tile_c in range(n):
            tile = patterns[tile_r, tile_c].copy()
            a, b = map(int, targets[tile_r, tile_c])
            assert tile[a, b] == bg, "pointer must land on an empty cell in its own tile"
            tile[a, b] = RED
            r0, c0 = _tile_origin(tile_r, tile_c, n)
            grid[r0 : r0 + n, c0 : c0 + n] = tile
    return grid


def _shortcut_targets_identity(n: int) -> np.ndarray:
    targets = np.empty((n, n, 2), dtype=int)
    for tile_r in range(n):
        for tile_c in range(n):
            targets[tile_r, tile_c] = (tile_r, tile_c)
    return targets


def _shortcut_targets_transposed(targets: np.ndarray) -> np.ndarray:
    out = np.empty_like(targets)
    n = int(targets.shape[0])
    for tile_r in range(n):
        for tile_c in range(n):
            a, b = map(int, targets[tile_r, tile_c])
            out[tile_r, tile_c] = (b, a)
    return out


def _shortcut_targets_row_only(targets: np.ndarray) -> np.ndarray:
    out = np.empty_like(targets)
    n = int(targets.shape[0])
    for tile_r in range(n):
        for tile_c in range(n):
            a = int(targets[tile_r, tile_c, 0])
            out[tile_r, tile_c] = (a, tile_c)
    return out


def _shortcut_targets_col_only(targets: np.ndarray) -> np.ndarray:
    out = np.empty_like(targets)
    n = int(targets.shape[0])
    for tile_r in range(n):
        for tile_c in range(n):
            b = int(targets[tile_r, tile_c, 1])
            out[tile_r, tile_c] = (tile_r, b)
    return out


def _validate_case(grid: np.ndarray, patterns: np.ndarray, targets: np.ndarray, bg: int) -> bool:
    n = int(patterns.shape[0])
    total_tiles = n * n
    coords = [(r, c) for r in range(n) for c in range(n)]

    fixed_points = sum(tuple(map(int, targets[r, c])) == (r, c) for r, c in coords)
    assert 1 <= fixed_points < total_tiles
    assert any(tuple(map(int, targets[r, c])) == (r, c) and r != c for r, c in coords)
    assert any(int(targets[r, c, 0]) != r and int(targets[r, c, 1]) != c for r, c in coords)
    assert len({patterns[r, c].tobytes() for r, c in coords}) == total_tiles

    expected = _build_output(patterns, targets, bg)
    assert np.array_equal(main(grid), expected)
    assert not np.array_equal(grid, expected)
    assert not np.any(expected == RED)

    identity_out = _build_output(patterns, _shortcut_targets_identity(n), bg)
    transposed_out = _build_output(patterns, _shortcut_targets_transposed(targets), bg)
    row_only_out = _build_output(patterns, _shortcut_targets_row_only(targets), bg)
    col_only_out = _build_output(patterns, _shortcut_targets_col_only(targets), bg)

    assert not np.array_equal(expected, identity_out)
    assert not np.array_equal(expected, transposed_out)
    assert not np.array_equal(expected, row_only_out)
    assert not np.array_equal(expected, col_only_out)

    _, _, _patterns2, _targets2, input_tiles = _parse_scene(grid)
    copy_with_red_out = _build_from_tiles(input_tiles, targets, bg)
    assert not np.array_equal(expected, copy_with_red_out)

    separator_rows = {k * (n + 1) - 1 for k in range(1, n)}
    separator_cols = {k * (n + 1) - 1 for k in range(1, n)}
    for r in range(grid.shape[0]):
        if r not in separator_rows:
            assert np.any(grid[r, :] != bg), "non-separator rows must stay visually active"
    for c in range(grid.shape[1]):
        if c not in separator_cols:
            assert np.any(grid[:, c] != bg), "non-separator columns must stay visually active"

    return True


def _try_generate_case(rng: random.Random) -> np.ndarray | None:
    n = rng.randint(GEN_ORDER_MIN, GEN_ORDER_MAX)
    bg, fg = rng.sample([color for color in PALETTE if color != RED], 2)

    targets = _make_pointer_targets(n, rng)
    patterns = _generate_pattern_bank(n, fg, bg, targets, rng)
    if patterns is None:
        return None

    grid = _assemble_input(n, bg, patterns, targets)
    try:
        _validate_case(grid, patterns, targets, bg)
    except AssertionError:
        return None
    return grid


_GEN_RNG = random.Random()


def generate_input() -> np.ndarray:
    for _ in range(GEN_SCENE_ATTEMPTS):
        grid = _try_generate_case(_GEN_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pointer-fetch tile scene")
