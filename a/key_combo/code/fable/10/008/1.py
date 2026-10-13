from __future__ import annotations

from collections import Counter
import math

import numpy as np


_RNG = np.random.default_rng()
_COLOR_POOL = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)
_MAX_GENERATION_ATTEMPTS = 400


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def _minimal_square_period(tile: np.ndarray) -> int:
    k = int(tile.shape[0])
    assert tile.shape == (k, k), "expected a square tile"
    for d in _divisors(k):
        if d == k:
            return k
        if np.array_equal(tile, tile[np.arange(k) % d][:, np.arange(k) % d]):
            return d
    return k


def _candidate_tile(grid: np.ndarray, anomaly: int, k: int) -> np.ndarray | None:
    h, w = map(int, grid.shape)
    if h % k or w % k:
        return None

    tile = np.empty((k, k), dtype=int)
    for r in range(k):
        for c in range(k):
            counts = Counter(map(int, grid[r::k, c::k].ravel()))
            anomaly_hits = counts.pop(int(anomaly), 0)
            if len(counts) != 1:
                return None
            clean_color, clean_hits = next(iter(counts.items()))
            if clean_hits <= anomaly_hits:
                return None
            tile[r, c] = int(clean_color)

    return tile if _minimal_square_period(tile) == k else None


def _all_candidates(grid: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    ks = _divisors(math.gcd(h, w))
    candidates: list[tuple[int, int, np.ndarray]] = []
    for anomaly in map(int, np.unique(g)):
        for k in ks:
            tile = _candidate_tile(g, anomaly, k)
            if tile is not None:
                candidates.append((anomaly, k, tile))
    return candidates


def _infer_unique_solution(grid: np.ndarray) -> tuple[int, int, np.ndarray]:
    candidates = _all_candidates(grid)
    assert len(candidates) == 1, f"expected a unique anomaly/period solution, got {len(candidates)}"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold sparse anomaly pixels back into the minimal periodic tile and keep only odd-hit residues as anomalies.

    Input structure:
    - The whole grid is a perfect tiling of one unknown k x k minimal periodic tile.
    - Some cells have been replaced by one anomaly color that does not belong to the clean tile palette.
    - For every residue position modulo k, clean copies remain in the majority.

    Rule:
    - Infer the unique anomaly color and the smallest square period k that make every residue class agree after ignoring anomalies.
    - Reconstruct the clean k x k tile from the majority color of each residue class.
    - Count how many anomaly pixels fall into each residue position (row mod k, col mod k); write the anomaly color on odd counts and the clean tile color on even counts.

    Key insight:
    - The odd-colored cells are not noise to erase: after periodic alignment, their residue-wise parity encodes which tile positions should flip to the anomaly color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert 1 < min(g.shape) <= 30 and max(g.shape) <= 30, "expected an ARC-sized grid"

    anomaly, k, tile = _infer_unique_solution(g)
    out = tile.copy()
    for r in range(k):
        for c in range(k):
            hits = int(np.count_nonzero(g[r::k, c::k] == anomaly))
            if hits % 2 == 1:
                out[r, c] = anomaly
    return out


def _build_tile(k: int, palette: np.ndarray) -> np.ndarray | None:
    n = k * k
    if n < 2 * len(palette):
        return None

    flat = np.empty(n, dtype=int)
    cells = _RNG.permutation(n)
    idx = 0
    for color in palette:
        flat[cells[idx : idx + 2]] = int(color)
        idx += 2
    if idx < n:
        flat[cells[idx:]] = _RNG.choice(palette, size=n - idx, replace=True)

    tile = flat.reshape(k, k)
    counts = Counter(map(int, tile.ravel()))
    if min(counts.values()) < 2:
        return None
    if _minimal_square_period(tile) != k:
        return None
    return tile


def _copy_positions(h: int, w: int, k: int, r: int, c: int) -> list[tuple[int, int]]:
    return [(rr, cc) for rr in range(r, h, k) for cc in range(c, w, k)]


def _choose_hit_count(parity: int, max_hits: int) -> int:
    options = [n for n in range(1, max_hits + 1) if n % 2 == parity]
    assert options, "expected at least one admissible hit count for the requested parity"
    return int(_RNG.choice(np.array(options, dtype=int)))


def _render_periodic_grid(tile: np.ndarray, repeats_r: int, repeats_c: int) -> np.ndarray:
    return np.tile(tile, (repeats_r, repeats_c))


def _semantic_accepts(
    inp: np.ndarray,
    tile: np.ndarray,
    anomaly: int,
    k: int,
    odd_residue: tuple[int, int],
    even_residue: tuple[int, int],
    affected: dict[tuple[int, int], list[tuple[int, int]]],
) -> bool:
    try:
        pred = main(inp)
    except AssertionError:
        return False

    expected = tile.copy()
    for residue, positions in affected.items():
        if len(positions) % 2 == 1:
            expected[residue] = anomaly

    if not np.array_equal(pred, expected):
        return False
    if np.array_equal(expected, tile):
        return False
    if np.array_equal(inp[:k, :k], expected):
        return False

    odd_positions = set(affected[odd_residue])
    even_positions = set(affected[even_residue])
    if odd_residue in odd_positions:
        return False
    if even_residue not in even_positions:
        return False

    candidates = _all_candidates(inp)
    if len(candidates) != 1:
        return False

    inferred_anomaly, inferred_k, inferred_tile = candidates[0]
    if inferred_anomaly != anomaly or inferred_k != k or not np.array_equal(inferred_tile, tile):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    k = int(_RNG.integers(2, 7))
    repeats_r = int(_RNG.integers(3, min(6, 30 // k + 1)))
    repeats_c = int(_RNG.integers(3, min(6, 30 // k + 1)))
    copies_per_residue = repeats_r * repeats_c
    max_hits = min((copies_per_residue - 1) // 2, 5)
    if max_hits < 2:
        return None

    max_palette = min(4, (k * k) // 2)
    palette_size = int(_RNG.integers(2, max_palette + 1))
    anomaly = int(_RNG.choice(_COLOR_POOL))
    clean_palette = _RNG.choice(_COLOR_POOL[_COLOR_POOL != anomaly], size=palette_size, replace=False)
    tile = _build_tile(k, clean_palette)
    if tile is None:
        return None

    residues_by_color: dict[int, list[tuple[int, int]]] = {}
    for r in range(k):
        for c in range(k):
            residues_by_color.setdefault(int(tile[r, c]), []).append((r, c))
    if any(len(residues) < 2 for residues in residues_by_color.values()):
        return None

    reserved: set[tuple[int, int]] = set()
    for residues in residues_by_color.values():
        reserved.add(tuple(residues[int(_RNG.integers(len(residues)))]))

    affectable = [(r, c) for r in range(k) for c in range(k) if (r, c) not in reserved]
    if len(affectable) < 2:
        return None

    order = list(_RNG.permutation(len(affectable)))
    odd_residue = affectable[order[0]]
    even_residue = affectable[order[1]]
    extra_cap = min(len(affectable) - 2, 4)
    extra_count = int(_RNG.integers(0, extra_cap + 1))
    affected_residues = [odd_residue, even_residue]
    for idx in order[2 : 2 + extra_count]:
        affected_residues.append(affectable[idx])

    inp = _render_periodic_grid(tile, repeats_r, repeats_c)
    affected: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for residue in affected_residues:
        r, c = residue
        positions = _copy_positions(inp.shape[0], inp.shape[1], k, r, c)
        first_copy = (r, c)
        parity = 1 if residue == odd_residue else 0 if residue == even_residue else int(_RNG.integers(0, 2))
        hit_count = _choose_hit_count(parity, max_hits)

        if residue == odd_residue:
            pool = [pos for pos in positions if pos != first_copy]
            chosen_idx = _RNG.choice(len(pool), size=hit_count, replace=False)
            chosen = [pool[int(i)] for i in np.atleast_1d(chosen_idx)]
        elif residue == even_residue:
            pool = [pos for pos in positions if pos != first_copy]
            chosen_idx = _RNG.choice(len(pool), size=hit_count - 1, replace=False)
            chosen = [first_copy] + [pool[int(i)] for i in np.atleast_1d(chosen_idx)]
        else:
            chosen_idx = _RNG.choice(len(positions), size=hit_count, replace=False)
            chosen = [positions[int(i)] for i in np.atleast_1d(chosen_idx)]

        affected[residue] = chosen
        for rr, cc in chosen:
            inp[rr, cc] = anomaly

    if _semantic_accepts(inp, tile, anomaly, k, odd_residue, even_residue, affected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid periodic-residue-folding input")
