from __future__ import annotations

import numpy as np


_RNG = np.random.default_rng()
_BG = 0
_PALETTE = np.arange(1, 10, dtype=int)
_GEN_MAX_ATTEMPTS = 400


def _infer_order(side: int) -> int:
    candidates = [n for n in range(2, 6) if n * n + n - 1 == side]
    assert len(candidates) == 1, f"invalid separated-tile side length: {side}"
    return candidates[0]


def _positions(n: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(n) for c in range(n)]


def _tile_origin(n: int, tile_r: int, tile_c: int) -> tuple[int, int]:
    step = n + 1
    return tile_r * step, tile_c * step


def _separator_indices(n: int) -> list[int]:
    return [k * (n + 1) - 1 for k in range(1, n)]


def _parse_grid(grid: np.ndarray) -> tuple[int, int, dict[tuple[int, int], np.ndarray], dict[tuple[int, int], tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1], "expected a square grid"

    n = _infer_order(int(g.shape[0]))
    sep = _separator_indices(n)
    bg_values = set(map(int, g[sep, :].ravel())) | set(map(int, g[:, sep].ravel()))
    assert len(bg_values) == 1, "separator rows/cols must be a single background color"
    bg = next(iter(bg_values))

    tiles: dict[tuple[int, int], np.ndarray] = {}
    holes: dict[tuple[int, int], tuple[int, int]] = {}
    all_holes = set()
    for tile_r, tile_c in _positions(n):
        r0, c0 = _tile_origin(n, tile_r, tile_c)
        patch = g[r0 : r0 + n, c0 : c0 + n].copy()
        hole_coords = np.argwhere(patch == bg)
        assert hole_coords.shape == (1, 2), "each tile must contain exactly one background hole"
        hole = tuple(map(int, hole_coords[0]))
        non_hole = patch != bg
        assert np.all(non_hole) or np.count_nonzero(~non_hole) == 1, "only the hole may use the background color"
        tiles[(tile_r, tile_c)] = patch
        holes[(tile_r, tile_c)] = hole
        all_holes.add(hole)

    assert all_holes == set(_positions(n)), "hole coordinates must form a permutation of array coordinates"
    return n, bg, tiles, holes


def _compose_grid(n: int, bg: int, tiles: dict[tuple[int, int], np.ndarray]) -> np.ndarray:
    side = n * n + n - 1
    out = np.full((side, side), int(bg), dtype=int)
    for tile_r, tile_c in _positions(n):
        r0, c0 = _tile_origin(n, tile_r, tile_c)
        out[r0 : r0 + n, c0 : c0 + n] = tiles[(tile_r, tile_c)]
    return out


def _pointer_rearrangement(
    n: int,
    bg: int,
    tiles: dict[tuple[int, int], np.ndarray],
    holes: dict[tuple[int, int], tuple[int, int]],
) -> np.ndarray:
    out_tiles = {
        dst: tiles[holes[dst]].copy()
        for dst in _positions(n)
    }
    return _compose_grid(n, bg, out_tiles)


def main(grid: np.ndarray) -> np.ndarray:
    """Reorder a separated tile array by treating each tile's hole as a source address.

    Input structure:
    - The grid is split by single-pixel separator rows and columns into an n x n array of n x n tiles.
    - Every tile contains an arbitrary multicolor pattern with exactly one background-colored hole.
    - The hole positions, read as local (row, col) coordinates, are all distinct and together cover the full set of array coordinates.

    Rule:
    - Keep the separator rows and columns unchanged.
    - For each destination tile position (i, j), read that tile's hole coordinate (r, c).
    - Replace the destination tile with the entire input tile currently located at array position (r, c), copied exactly including the source tile's own hole.

    Key insight:
    - The hole is not a self-label for the tile that contains it; it is a pointer telling you which other tile to fetch and place here.
    """
    n, bg, tiles, holes = _parse_grid(grid)
    return _pointer_rearrangement(n, bg, tiles, holes)


def _cycles_from_permutation(perm: list[int]) -> list[list[int]]:
    seen = [False] * len(perm)
    cycles: list[list[int]] = []
    for start in range(len(perm)):
        if seen[start]:
            continue
        cur = start
        cycle = []
        while not seen[cur]:
            seen[cur] = True
            cycle.append(cur)
            cur = perm[cur]
        cycles.append(cycle)
    return cycles


def _sample_cycle_lengths(total: int) -> list[int]:
    max_fixed = max(1, min(total - 2, total // 2 + 1))
    fixed = int(_RNG.integers(1, max_fixed + 1))
    remain = total - fixed
    if remain == 1:
        fixed -= 1
        remain += 1

    mode = int(_RNG.integers(3))
    lengths = [1] * fixed
    if mode == 0:
        lengths.append(remain)
    else:
        if mode == 2 and remain >= 5:
            long_len = int(_RNG.integers(3, remain))
            if remain - long_len == 1:
                long_len -= 1
            lengths.append(long_len)
            remain -= long_len
        while remain:
            if remain in (2, 3):
                lengths.append(remain)
                remain = 0
            elif remain % 2 == 1:
                lengths.append(3)
                remain -= 3
            else:
                lengths.append(2)
                remain -= 2
    _RNG.shuffle(lengths)
    return lengths


def _sample_permutation(n: int) -> dict[tuple[int, int], tuple[int, int]]:
    pos = _positions(n)
    total = len(pos)
    indices = np.arange(total)
    _RNG.shuffle(indices)
    perm = list(range(total))
    cursor = 0
    for length in _sample_cycle_lengths(total):
        cycle = list(map(int, indices[cursor : cursor + length]))
        cursor += length
        if length == 1:
            perm[cycle[0]] = cycle[0]
            continue
        for idx, src in enumerate(cycle):
            perm[src] = cycle[(idx + 1) % length]

    mapping = {pos[idx]: pos[perm[idx]] for idx in range(total)}
    fixed = [p for p in pos if mapping[p] == p]
    assert fixed and len(fixed) < total
    return mapping


def _sample_tile_patch(n: int, hole: tuple[int, int], used: set[bytes]) -> np.ndarray | None:
    palette_size = int(_RNG.integers(3, min(7, len(_PALETTE)) + 1))
    palette = _RNG.choice(_PALETTE, size=palette_size, replace=False)

    for _ in range(200):
        patch = _RNG.choice(palette, size=(n, n), replace=True)

        # Add a couple of structured accents so tiles are easy to distinguish by eye.
        accent_a = int(palette[_RNG.integers(palette_size)])
        accent_b = int(palette[_RNG.integers(palette_size)])
        patch[np.arange(n), (_RNG.integers(n) + np.arange(n)) % n] = accent_a
        patch[(_RNG.integers(n) + np.arange(n)) % n, np.arange(n)] = accent_b
        if n >= 3 and bool(_RNG.integers(2)):
            patch[:, int(_RNG.integers(n))] = int(palette[_RNG.integers(palette_size)])
        if n >= 3 and bool(_RNG.integers(2)):
            patch[int(_RNG.integers(n)), :] = int(palette[_RNG.integers(palette_size)])

        patch[hole] = _BG
        visible = patch[patch != _BG]
        if len(np.unique(visible)) < 2:
            continue
        sig = patch.tobytes()
        if sig in used:
            continue
        used.add(sig)
        return patch
    return None


def _inverse_rearrangement(
    n: int,
    bg: int,
    tiles: dict[tuple[int, int], np.ndarray],
    holes: dict[tuple[int, int], tuple[int, int]],
) -> np.ndarray:
    out_tiles: dict[tuple[int, int], np.ndarray] = {}
    for src in _positions(n):
        out_tiles[holes[src]] = tiles[src].copy()
    return _compose_grid(n, bg, out_tiles)


def _validate_semantics(grid: np.ndarray) -> bool:
    n, bg, tiles, holes = _parse_grid(grid)
    out = _pointer_rearrangement(n, bg, tiles, holes)
    moved = [pos for pos in _positions(n) if holes[pos] != pos]
    fixed = [pos for pos in _positions(n) if holes[pos] == pos]
    if not moved or not fixed:
        return False
    if np.array_equal(grid, out):
        return False

    tile_sigs = {tile.tobytes() for tile in tiles.values()}
    if len(tile_sigs) != n * n:
        return False

    # The pointer rule must differ from "leave tiles where they are" and from
    # the opposite interpretation "move each tile to the coordinate named by its hole".
    if np.array_equal(out, grid):
        return False
    if np.array_equal(out, _inverse_rearrangement(n, bg, tiles, holes)):
        return False

    pos = _positions(n)
    pos_to_idx = {coord: idx for idx, coord in enumerate(pos)}
    perm = [pos_to_idx[holes[coord]] for coord in pos]
    cycle_lengths = sorted(len(cycle) for cycle in _cycles_from_permutation(perm))
    if cycle_lengths[-1] < 2:
        return False

    # Every moved output tile must come from exactly the addressed source tile.
    for dst in moved:
        src = holes[dst]
        dst_r0, dst_c0 = _tile_origin(n, *dst)
        if not np.array_equal(out[dst_r0 : dst_r0 + n, dst_c0 : dst_c0 + n], tiles[src]):
            return False
    return True


def _generate_attempt() -> np.ndarray | None:
    n = int(_RNG.integers(2, 6))
    holes = _sample_permutation(n)

    tiles: dict[tuple[int, int], np.ndarray] = {}
    used: set[bytes] = set()
    for pos in _positions(n):
        patch = _sample_tile_patch(n, holes[pos], used)
        if patch is None:
            return None
        tiles[pos] = patch

    grid = _compose_grid(n, _BG, tiles)
    if not _validate_semantics(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pointer-tile input")
