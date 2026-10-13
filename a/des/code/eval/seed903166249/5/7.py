from __future__ import annotations

from itertools import combinations

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
BOARD_PALETTE = np.array([1, 4, 5, 6, 7, 8, 9], dtype=int)
GEN_RETRIES = 400
SPOT_PICK_RETRIES = 800


def _checker_value(r: int, c: int, color_a: int, color_b: int, phase: int) -> int:
    return int(color_a if ((int(r) + int(c) + int(phase)) & 1) == 0 else color_b)


def _infer_checker_colors(interior: np.ndarray) -> tuple[int, int, int]:
    colors = [int(c) for c in np.unique(interior)]
    assert len(colors) >= 3, "expected two checker colors plus anomaly color"

    best: tuple[tuple[int, int], tuple[int, int, int]] | None = None
    h, w = map(int, interior.shape)

    for color_a, color_b in combinations(colors, 2):
        for phase in (0, 1):
            mismatches = 0
            covered = 0
            for r in range(h):
                for c in range(w):
                    val = int(interior[r, c])
                    if val not in (int(color_a), int(color_b)):
                        continue
                    covered += 1
                    expected = _checker_value(int(r), int(c), int(color_a), int(color_b), int(phase))
                    if int(val) != int(expected):
                        mismatches += 1
            score = (int(mismatches), -int(covered))
            if best is None or score < best[0]:
                best = (score, (int(color_a), int(color_b), int(phase)))

    assert best is not None
    (mismatches, _neg_covered), (color_a, color_b, phase) = best
    assert int(mismatches) == 0, "failed to infer a clean checkerboard phase"
    return int(color_a), int(color_b), int(phase)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair red checkerboard anomalies and mark their row/column projections on the border.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h >= 5 and w >= 5, "expected room for a border and checkerboard interior"

    assert np.all(g[0, :] == BLACK)
    assert np.all(g[h - 1, :] == BLACK)
    assert np.all(g[:, 0] == BLACK)
    assert np.all(g[:, w - 1] == BLACK)

    interior = g[1:-1, 1:-1]
    color_a, color_b, _phase = _infer_checker_colors(interior)
    spot_mask = (interior != int(color_a)) & (interior != int(color_b))
    spot_vals = np.unique(interior[spot_mask])
    assert spot_mask.any(), "expected at least one anomaly"
    assert len(spot_vals) == 1 and int(spot_vals[0]) == RED, "expected a single red anomaly color"

    out = g.copy()
    for rr, cc in np.argwhere(spot_mask):
        r = int(rr + 1)
        c = int(cc + 1)

        count_a = 0
        count_b = 0
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            val = int(g[int(r + dr), int(c + dc)])
            if int(val) == int(color_a):
                count_a += 1
            elif int(val) == int(color_b):
                count_b += 1

        assert count_a != count_b, "an anomaly must have a strict neighbor majority"
        out[r, c] = int(color_b if count_a > count_b else color_a)
        out[0, c] = GREEN
        out[r, 0] = GREEN

    return out


def _spots_are_separated(spots: list[tuple[int, int]], candidate: tuple[int, int]) -> bool:
    r, c = map(int, candidate)
    for r0, c0 in spots:
        if max(abs(int(r) - int(r0)), abs(int(c) - int(c0))) < 2:
            return False
    return True


def _has_shared_row(spots: list[tuple[int, int]]) -> bool:
    rows = [int(r) for r, _c in spots]
    return len(rows) != len(set(rows))


def _has_shared_col(spots: list[tuple[int, int]]) -> bool:
    cols = [int(c) for _r, c in spots]
    return len(cols) != len(set(cols))


def _pick_spots(
    h: int,
    w: int,
    n_spots: int,
    phase: int,
    require_edge: bool,
    share_mode: str,
) -> list[tuple[int, int]]:
    candidates = [(int(r), int(c)) for r in range(1, int(h - 1)) for c in range(1, int(w - 1))]

    for _ in range(SPOT_PICK_RETRIES):
        order = np.random.permutation(len(candidates))
        spots: list[tuple[int, int]] = []

        for idx in order:
            cand = candidates[int(idx)]
            if _spots_are_separated(spots, cand):
                spots.append(cand)
                if len(spots) == int(n_spots):
                    break

        if len(spots) != int(n_spots):
            continue

        parities = {((int(r) + int(c) + int(phase)) & 1) for r, c in spots}
        if len(parities) < 2:
            continue
        if require_edge and not any(r in (1, int(h - 2)) or c in (1, int(w - 2)) for r, c in spots):
            continue
        if share_mode == "row" and not _has_shared_row(spots):
            continue
        if share_mode == "col" and not _has_shared_col(spots):
            continue
        return spots

    raise RuntimeError("failed to place non-degenerate anomaly positions")


def _expected_output(
    h: int,
    w: int,
    color_a: int,
    color_b: int,
    phase: int,
    spots: list[tuple[int, int]],
) -> np.ndarray:
    out = np.full((int(h), int(w)), BLACK, dtype=int)
    for r in range(1, int(h - 1)):
        for c in range(1, int(w - 1)):
            out[r, c] = _checker_value(int(r), int(c), int(color_a), int(color_b), int(phase))
    for r, c in spots:
        out[0, int(c)] = GREEN
        out[int(r), 0] = GREEN
    return out


def _gen_attempt() -> np.ndarray | None:
    h = int(np.random.randint(8, 19))
    w = int(np.random.randint(8, 19))
    phase = int(np.random.randint(0, 2))
    n_spots = int(np.random.randint(3, 7))
    require_edge = bool(np.random.randint(0, 2))
    share_mode = ("none", "row", "col")[int(np.random.randint(0, 3))]
    color_a, color_b = map(int, np.random.choice(BOARD_PALETTE, size=2, replace=False))

    try:
        spots = _pick_spots(h, w, n_spots, phase, require_edge, share_mode)
    except RuntimeError:
        return None

    grid = np.full((int(h), int(w)), BLACK, dtype=int)
    for r in range(1, int(h - 1)):
        for c in range(1, int(w - 1)):
            grid[r, c] = _checker_value(int(r), int(c), int(color_a), int(color_b), int(phase))
    for r, c in spots:
        grid[int(r), int(c)] = RED

    expected = _expected_output(h, w, color_a, color_b, phase, spots)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid checkerboard repair task")
