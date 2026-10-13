from __future__ import annotations

import math

import numpy as np


BLACK = 0
GREY = 5
PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
GEN_ATTEMPTS = 500


def _consistent_tile_for_period(grid: np.ndarray, p: int, erased: int = GREY) -> np.ndarray | None:
    """Return the unique p x p tile consistent with visible pixels, or None."""
    h, w = map(int, grid.shape)
    if h % p != 0 or w % p != 0:
        return None

    tile = np.full((p, p), -1, dtype=int)
    for tr in range(p):
        for tc in range(p):
            samples = grid[tr::p, tc::p]
            visible = samples[samples != erased]
            if visible.size == 0:
                return None
            color = int(visible[0])
            if np.any(visible != color):
                return None
            tile[tr, tc] = color
    return tile


def _infer_minimal_tile(grid: np.ndarray, erased: int = GREY) -> tuple[int, np.ndarray]:
    """Infer the smallest square period whose phase classes match all visible pixels."""
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    for p in range(1, min(h, w) + 1):
        tile = _consistent_tile_for_period(g, p, erased=erased)
        if tile is not None:
            return p, tile
    raise ValueError("no valid square period found")


def _paint_tile(tile: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Expand a period tile across a full grid."""
    h, w = map(int, shape)
    p = int(tile.shape[0])
    row_phase = np.arange(h)[:, None] % p
    col_phase = np.arange(w)[None, :] % p
    return tile[row_phase, col_phase]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair a periodic color tile grid, then output only the repaired colors at the original damage sites.

    Input structure:
    - The grid is a rectangle tiled by repeated copies of one square p x p color tile.
    - Tile pixels use non-black non-grey ARC colors, while grey pixels mark damaged cells.
    - Each tile phase class (row mod p, col mod p) keeps at least one visible non-grey sample.

    Rule:
    - Test square periods from small to large and choose the smallest p whose visible non-grey pixels are
      color-consistent within every (row mod p, col mod p) phase class.
    - Reconstruct the full repeated tile coloring from that inferred p x p tile.
    - Output black at every originally visible cell, and output the reconstructed tile color at every
      originally grey damaged cell.

    Key insight:
    - The visible pixels identify the minimal repeating tile, but the output keeps only the complement mask:
      repair colors are needed to fill the damaged sites, while the damage mask decides where any color survives.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _p, tile = _infer_minimal_tile(g, erased=GREY)
    repaired = _paint_tile(tile, g.shape)
    return np.where(g == GREY, repaired, BLACK).astype(int)


def _random_exact_period_tile(rng: np.random.Generator, p: int) -> np.ndarray:
    """Sample a p x p tile whose true minimal square period is exactly p."""
    while True:
        color_count = int(rng.integers(3, 6))
        colors = rng.choice(PALETTE, size=color_count, replace=False)
        tile = rng.choice(colors, size=(p, p), replace=True)
        if np.unique(tile).size < 3:
            continue
        inferred_p, _ = _infer_minimal_tile(tile, erased=GREY)
        if inferred_p != p:
            continue
        return tile.astype(int)


def _all_tile_origins(h: int, w: int, p: int) -> list[tuple[int, int]]:
    return [(br * p, bc * p) for br in range(h // p) for bc in range(w // p)]


def _damage_mask(rng: np.random.Generator, shape: tuple[int, int], p: int) -> np.ndarray:
    """Create a damage mask with 20%-50% coverage, preserving one visible sample per phase."""
    h, w = map(int, shape)
    tile_rows = h // p
    tile_cols = w // p
    tile_instances = tile_rows * tile_cols
    cell_count = h * w

    min_damage = math.ceil(cell_count * 0.20)
    max_damage = math.floor(cell_count * 0.50)
    target = int(rng.integers(max(min_damage, tile_instances), max_damage + 1))

    mask = np.zeros((h, w), dtype=bool)
    phase_visible = np.full((p, p), tile_instances, dtype=int)

    origins = _all_tile_origins(h, w, p)
    rng.shuffle(origins)

    # Force every repeated tile copy to be incomplete, so copying one intact patch cannot solve the task.
    for r0, c0 in origins:
        candidates: list[tuple[int, int, int, int]] = []
        for dr in range(p):
            for dc in range(p):
                if phase_visible[dr, dc] > 1:
                    candidates.append((r0 + dr, c0 + dc, dr, dc))
        assert candidates, "phase visibility constraint should leave at least one removable cell"
        r, c, dr, dc = candidates[int(rng.integers(len(candidates)))]
        mask[r, c] = True
        phase_visible[dr, dc] -= 1

    candidates = [(r, c) for r in range(h) for c in range(w) if not mask[r, c]]
    rng.shuffle(candidates)
    for r, c in candidates:
        if int(mask.sum()) >= target:
            break
        pr = r % p
        pc = c % p
        if phase_visible[pr, pc] <= 1:
            continue
        mask[r, c] = True
        phase_visible[pr, pc] -= 1

    assert int(mask.sum()) == target
    assert np.all(phase_visible >= 1)
    return mask


def _count_distinct_phase_classes(mask: np.ndarray, p: int) -> int:
    phase_hits = np.zeros((p, p), dtype=bool)
    damaged = np.argwhere(mask)
    for r, c in damaged:
        phase_hits[int(r) % p, int(c) % p] = True
    return int(phase_hits.sum())


def _validate_case(inp: np.ndarray, tile: np.ndarray, damage: np.ndarray) -> None:
    """Semantic checks beyond a simple generate->solve roundtrip."""
    h, w = map(int, inp.shape)
    p = int(tile.shape[0])
    full = _paint_tile(tile, inp.shape)
    out = main(inp)
    expected = np.where(damage, full, BLACK)

    inferred_p, inferred_tile = _infer_minimal_tile(inp, erased=GREY)
    assert inferred_p == p
    assert np.array_equal(inferred_tile, tile)

    assert np.array_equal(out, expected)
    assert not np.any(out == GREY)
    assert int(np.count_nonzero(out != BLACK)) == int(damage.sum())

    damaged_colors = set(map(int, out[out != BLACK]))
    assert len(damaged_colors) >= 2, "output must expose multiple repaired colors"

    visible = inp != GREY
    for tr in range(p):
        for tc in range(p):
            assert np.any(visible[tr::p, tc::p]), "every phase needs a visible witness"

    tile_damage_counts = []
    for r0, c0 in _all_tile_origins(h, w, p):
        tile_damage_counts.append(int(damage[r0 : r0 + p, c0 : c0 + p].sum()))
    assert min(tile_damage_counts) >= 1, "every repeated tile copy should be damaged"

    min_phase_classes = min(p * p, max(4, p + 1))
    assert _count_distinct_phase_classes(damage, p) >= min_phase_classes

    smaller_periods = [
        q
        for q in range(1, p)
        if h % q == 0 and w % q == 0 and _consistent_tile_for_period(inp, q, erased=GREY) is not None
    ]
    assert not smaller_periods, "erasures must not accidentally allow a smaller fake period"

    damage_ratio = float(damage.mean())
    assert 0.20 <= damage_ratio <= 0.50


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        rng = np.random.default_rng()
        p = int(rng.choice([3, 4]))

        max_tiles = min(7, 30 // p)
        tile_rows = int(rng.integers(2, max_tiles + 1))
        tile_cols = int(rng.integers(2, max_tiles + 1))
        h = tile_rows * p
        w = tile_cols * p

        tile = _random_exact_period_tile(rng, p)
        full = _paint_tile(tile, (h, w))
        damage = _damage_mask(rng, (h, w), p)
        inp = full.copy()
        inp[damage] = GREY

        try:
            _validate_case(inp, tile, damage)
        except AssertionError:
            continue
        return inp.astype(int)

    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
