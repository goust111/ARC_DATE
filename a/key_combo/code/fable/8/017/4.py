from __future__ import annotations

from collections import Counter

import numpy as np


TILE_COLORS = np.arange(1, 10, dtype=int)
ALL_COLORS = np.arange(10, dtype=int)
GEN_ATTEMPTS = 300
TILE_ATTEMPTS = 200


def _period_options(length: int) -> list[int]:
    return [step for step in range(1, length + 1) if length % step == 0 and length // step >= 3]


def _is_primitive(tile: np.ndarray) -> bool:
    h, w = map(int, tile.shape)
    for dr in range(h):
        for dc in range(w):
            if dr == 0 and dc == 0:
                continue
            if np.array_equal(tile, np.roll(tile, shift=(-dr, -dc), axis=(0, 1))):
                return False
    return True


def _majority_tile(grid: np.ndarray, p: int, q: int) -> np.ndarray | None:
    tile = np.empty((p, q), dtype=int)
    mismatch_total = 0
    for r in range(p):
        for c in range(q):
            values = [int(v) for v in grid[r::p, c::q].ravel()]
            counts = Counter(values).most_common(2)
            if len(counts) > 1 and counts[0][1] == counts[1][1]:
                return None
            color, freq = counts[0]
            tile[r, c] = color
            mismatch_total += len(values) - freq
            if mismatch_total > 1:
                return None
    if mismatch_total != 1:
        return None
    return tile


def _infer_periodic_pointer(grid: np.ndarray) -> tuple[int, int, np.ndarray, tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    h, w = map(int, g.shape)

    candidates: list[tuple[int, int, np.ndarray, tuple[int, int]]] = []
    for p in _period_options(h):
        for q in _period_options(w):
            tile = _majority_tile(g, p, q)
            if tile is None or not _is_primitive(tile):
                continue
            expected = np.tile(tile, (h // p, w // q))
            bad = np.argwhere(g != expected)
            if len(bad) != 1:
                continue
            br, bc = map(int, bad[0])
            candidates.append((p, q, tile, (br % p, bc % q)))

    assert candidates, "failed to recover a unique periodic tile"
    candidates.sort(key=lambda item: (item[0] * item[1], item[0], item[1]))
    return candidates[0]


def _render_output(tile: np.ndarray, phase: tuple[int, int]) -> np.ndarray:
    p, q = map(int, tile.shape)
    slot_r, slot_c = phase
    out = np.zeros((p * p, q * q), dtype=int)
    out[slot_r * p : (slot_r + 1) * p, slot_c * q : (slot_c + 1) * q] = tile
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the repeated clean tile, use the lone defect's phase as an address, and stamp the tile into that output slot.

    Input structure:
    - The whole grid is an exact periodic tiling of one hidden p x q tile, repeated at least three times along both axes.
    - Exactly one cell has been recolored, so each residue class modulo (p, q) still has a strict majority color.
    - The true tile is primitive: no non-zero wrapped translation leaves it unchanged.

    Rule:
    - Search the valid period divisors of the grid size and reconstruct each candidate tile by taking the majority color in every phase class.
    - Keep the primitive candidate whose repeated clean pattern disagrees with the input in exactly one cell; that cell is the unique defect.
    - Let the defect's residue coordinates inside the tile be (r, c), create a black output of size p*p by q*q, and stamp one clean copy of the tile into block slot (r, c).

    Key insight:
    - The wrong pixel is not repaired in place; its phase inside the repeated wallpaper is the pointer that selects where the recovered tile must be placed in the larger block grid.
    """
    p, q, tile, phase = _infer_periodic_pointer(np.asarray(grid, dtype=int))
    return _render_output(tile, phase)


def _sample_tile(rng: np.random.Generator, p: int, q: int) -> np.ndarray:
    area = p * q
    min_colors = 1 if area == 1 else 2
    max_colors = min(5, area, len(TILE_COLORS))
    for _ in range(TILE_ATTEMPTS):
        color_count = int(rng.integers(min_colors, max_colors + 1))
        palette = rng.choice(TILE_COLORS, size=color_count, replace=False)
        tile = rng.choice(palette, size=(p, q), replace=True).astype(int)
        if area > 1 and len(np.unique(tile)) < 2:
            continue
        if not _is_primitive(tile):
            continue
        return tile
    raise RuntimeError("failed to sample a primitive tile")


def _period_pair_options() -> list[tuple[int, int]]:
    options: list[tuple[int, int]] = []
    for p in range(1, 6):
        for q in range(1, 6):
            if p == 1 and q == 1:
                continue
            options.append((p, q))
    return options


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    period_options = _period_pair_options()
    weights = np.array([2.0 if 1 in pair else 5.0 for pair in period_options], dtype=float)
    p, q = period_options[int(rng.choice(len(period_options), p=weights / weights.sum()))]

    reps_r = int(rng.integers(3, min(8, 30 // p) + 1))
    reps_c = int(rng.integers(3, min(8, 30 // q) + 1))
    tile = _sample_tile(rng, p, q)
    clean = np.tile(tile, (reps_r, reps_c))

    bad_r = int(rng.integers(clean.shape[0]))
    bad_c = int(rng.integers(clean.shape[1]))
    expected = int(clean[bad_r, bad_c])
    defect_choices = ALL_COLORS[ALL_COLORS != expected]
    defect_color = int(rng.choice(defect_choices))

    grid = clean.copy()
    grid[bad_r, bad_c] = defect_color

    try:
        got_p, got_q, got_tile, got_phase = _infer_periodic_pointer(grid)
    except AssertionError:
        return None

    if (got_p, got_q) != (p, q):
        return None
    if got_phase != (bad_r % p, bad_c % q):
        return None
    if not np.array_equal(got_tile, tile):
        return None

    out = _render_output(tile, got_phase)
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid periodic-defect input")
