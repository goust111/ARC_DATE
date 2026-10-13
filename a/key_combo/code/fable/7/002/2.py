from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(UTILS_DIR) not in sys.path:
    sys.path.append(str(UTILS_DIR))

from arc_utils import crop_bbox, tile_fill


PALETTE = tuple(range(10))
MAX_GENERATION_ATTEMPTS = 400


def _smallest_row_period(tile: np.ndarray) -> int:
    rows = [tuple(map(int, row)) for row in tile]
    for period in range(1, tile.shape[0] + 1):
        if all(rows[r] == rows[r % period] for r in range(tile.shape[0])):
            return period
    raise AssertionError("unreachable")


def _smallest_col_period(tile: np.ndarray) -> int:
    cols = [tuple(map(int, tile[:, c])) for c in range(tile.shape[1])]
    for period in range(1, tile.shape[1] + 1):
        if all(cols[c] == cols[c % period] for c in range(tile.shape[1])):
            return period
    raise AssertionError("unreachable")


def _sample_fundamental_tile(period_r: int, period_c: int) -> np.ndarray:
    for _ in range(300):
        palette_size = random.randint(2, min(6, period_r * period_c, len(PALETTE)))
        palette = random.sample(PALETTE, k=palette_size)
        tile = np.random.choice(palette, size=(period_r, period_c))
        if len(np.unique(tile)) < 2:
            continue
        if _smallest_row_period(tile) != period_r:
            continue
        if _smallest_col_period(tile) != period_c:
            continue
        return tile.astype(int)
    raise RuntimeError("failed to sample a fundamental tile")


def _majority_tile(grid: np.ndarray, period_r: int, period_c: int) -> np.ndarray | None:
    tile = np.empty((period_r, period_c), dtype=int)
    for rr in range(period_r):
        for cc in range(period_c):
            values = grid[rr::period_r, cc::period_c].ravel()
            counts = Counter(map(int, values))
            ranked = counts.most_common()
            if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
                return None
            tile[rr, cc] = int(ranked[0][0])
    return tile


def _pointer_anchor(mismatch_coords: np.ndarray) -> tuple[int, int] | None:
    if mismatch_coords.shape != (6, 2):
        return None
    rows = mismatch_coords[:, 0]
    cols = mismatch_coords[:, 1]
    r0 = int(rows.min())
    c0 = int(cols.min())
    r1 = int(rows.max())
    c1 = int(cols.max())
    if (r1 - r0 + 1, c1 - c0 + 1) != (4, 4):
        return None
    expected = {(r, c1) for r in range(r0, r0 + 3)}
    expected |= {(r1, c) for c in range(c0, c0 + 3)}
    got = {tuple(map(int, rc)) for rc in mismatch_coords}
    if got != expected:
        return None
    return r0, c0


def _solve_candidates(grid: np.ndarray) -> list[dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[dict[str, int]] = []
    for period_r in range(1, h + 1):
        for period_c in range(1, w + 1):
            tile = _majority_tile(g, period_r, period_c)
            if tile is None:
                continue
            predicted = tile_fill(g.shape, tile)
            mismatch_coords = np.argwhere(predicted != g)
            if mismatch_coords.shape[0] != 6:
                continue
            anchor = _pointer_anchor(mismatch_coords)
            if anchor is None:
                continue
            r0, c0 = anchor
            if np.any(predicted[r0 : r0 + 3, c0 : c0 + 3] != g[r0 : r0 + 3, c0 : c0 + 3]):
                continue
            candidates.append(
                {
                    "period_r": period_r,
                    "period_c": period_c,
                    "area": period_r * period_c,
                    "r0": r0,
                    "c0": c0,
                }
            )
    candidates.sort(key=lambda item: (item["area"], item["period_r"], item["period_c"]))
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """Recover the hidden periodic grid, read the L-shaped deviation pointer, and crop the intact 3x3 sample it indicates.

    Input structure:
    - The whole canvas is colored by a 2D repeating palette pattern with a uniquely recoverable fundamental period.
    - Exactly six cells deviate from that global periodic law, but their replacement colors still come from the same palette.
    - Those six deviant cells form the right-side column and bottom-side row adjacent to one intact 3x3 region, creating an L pointer.

    Rule:
    - Enumerate candidate row/column periods and reconstruct each periodic table by majority vote inside every residue class.
    - Keep the candidate whose mismatching cells are exactly the six L-pointer cells around a 4x4 bounding box, preferring the smallest valid fundamental period.
    - Return the untouched 3x3 region immediately above-left of that L pointer.

    Key insight:
    - The pointer is invisible in local color terms; it appears only after rebuilding the global periodic law and spotting the cells that violate it.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    candidates = _solve_candidates(g)
    assert candidates, "expected at least one valid periodic interpretation"

    best_area = candidates[0]["area"]
    best = [cand for cand in candidates if cand["area"] == best_area]
    anchors = {(cand["r0"], cand["c0"]) for cand in best}
    assert len(anchors) == 1, "smallest valid period must identify a unique sample"
    r0, c0 = next(iter(anchors))
    return crop_bbox(g, (r0, c0, r0 + 2, c0 + 2))


def _biased_anchor(limit: int) -> int:
    if limit == 0:
        return 0
    if random.random() < 0.55:
        edgeish = {0, limit}
        edgeish |= {x for x in (1, limit - 1) if 0 <= x <= limit}
        return random.choice(sorted(edgeish))
    return random.randint(0, limit)


def _deviation_positions(r0: int, c0: int) -> list[tuple[int, int]]:
    positions = [(r, c0 + 3) for r in range(r0, r0 + 3)]
    positions.extend((r0 + 3, c) for c in range(c0, c0 + 3))
    return positions


def _generate_case() -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    for _ in range(200):
        period_r = random.randint(2, 5)
        period_c = random.randint(2, 5)
        tile = _sample_fundamental_tile(period_r, period_c)

        min_h = max(8, period_r * 3)
        min_w = max(8, period_c * 3)
        height = random.randint(min_h, min(30, max(min_h, period_r * 6)))
        width = random.randint(min_w, min(30, max(min_w, period_c * 6)))
        offset = (random.randrange(period_r), random.randrange(period_c))
        base = tile_fill((height, width), tile, offset=offset).astype(int)

        r0 = _biased_anchor(height - 4)
        c0 = _biased_anchor(width - 4)
        target = crop_bbox(base, (r0, c0, r0 + 2, c0 + 2))
        if len(np.unique(target)) < 2:
            continue

        palette = sorted(map(int, np.unique(tile)))
        if len(palette) < 2:
            continue

        grid = base.copy()
        for r, c in _deviation_positions(r0, c0):
            original = int(base[r, c])
            choices = [color for color in palette if color != original]
            grid[r, c] = random.choice(choices)

        meta = {"period_r": period_r, "period_c": period_c, "r0": r0, "c0": c0}
        return grid, target, meta
    raise RuntimeError("failed to sample a non-degenerate case")


def _valid_generated_case(grid: np.ndarray, expected: np.ndarray, meta: dict[str, int]) -> bool:
    if grid.ndim != 2 or expected.shape != (3, 3):
        return False
    if max(grid.shape) > 30:
        return False
    if np.array_equal(grid, tile_fill(grid.shape, grid[:1, :1])):
        return False

    output = main(grid)
    if not isinstance(output, np.ndarray) or output.shape != (3, 3):
        return False
    if not np.array_equal(output, expected):
        return False

    candidates = _solve_candidates(grid)
    if not candidates:
        return False
    best_area = candidates[0]["area"]
    best = [cand for cand in candidates if cand["area"] == best_area]
    if len({(cand["r0"], cand["c0"]) for cand in best}) != 1:
        return False
    if (best[0]["period_r"], best[0]["period_c"]) != (meta["period_r"], meta["period_c"]):
        return False
    if (best[0]["r0"], best[0]["c0"]) != (meta["r0"], meta["c0"]):
        return False

    tile = _majority_tile(grid, meta["period_r"], meta["period_c"])
    if tile is None:
        return False
    predicted = tile_fill(grid.shape, tile)
    mismatch_set = {tuple(map(int, rc)) for rc in np.argwhere(predicted != grid)}
    expected_mismatch_set = set(_deviation_positions(meta["r0"], meta["c0"]))
    return mismatch_set == expected_mismatch_set


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            grid, expected, meta = _generate_case()
        except RuntimeError:
            continue
        if _valid_generated_case(grid, expected, meta):
            return grid
    raise RuntimeError("failed to generate a valid ARC instance")
