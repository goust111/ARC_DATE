from __future__ import annotations

from dataclasses import dataclass

import numpy as np


BLACK = 0
ARC_COLORS = tuple(range(10))
NON_BLACK = tuple(range(1, 10))

MAX_GENERATION_ATTEMPTS = 400
MAX_RESIDUAL_FRACTION_DENOM = 8
MIN_SAMPLES_PER_RESIDUE = 3


@dataclass(frozen=True)
class PeriodStats:
    tile: np.ndarray
    mismatch_count: int
    anomaly_by_residue: dict[tuple[int, int], int]


def _period_stats(grid: np.ndarray, pr: int, pc: int) -> PeriodStats | None:
    tile = np.zeros((pr, pc), dtype=int)
    anomalies: dict[tuple[int, int], int] = {}
    mismatch_count = 0

    for rr in range(pr):
        for cc in range(pc):
            cells = grid[rr::pr, cc::pc].reshape(-1)
            if cells.size < MIN_SAMPLES_PER_RESIDUE:
                return None

            values, counts = np.unique(cells, return_counts=True)
            best_i = int(np.argmax(counts))
            best_count = int(counts[best_i])
            if best_count * 2 <= int(cells.size):
                return None

            majority = int(values[best_i])
            tile[rr, cc] = majority

            differing = [int(v) for v in values if int(v) != majority]
            if len(differing) > 1:
                return None
            if differing:
                anomaly_color = differing[0]
                if anomaly_color == BLACK:
                    return None
                anomalies[(rr, cc)] = anomaly_color
                mismatch_count += int(cells.size - best_count)

    return PeriodStats(tile=tile, mismatch_count=mismatch_count, anomaly_by_residue=anomalies)


def _infer_period(grid: np.ndarray) -> tuple[int, int, PeriodStats]:
    H, W = grid.shape
    total = int(H * W)
    candidates: list[tuple[int, int, int, PeriodStats]] = []

    for pr in range(1, H + 1):
        for pc in range(1, W + 1):
            stats = _period_stats(grid, pr, pc)
            if stats is None:
                continue
            if stats.mismatch_count == 0:
                continue
            if stats.mismatch_count * MAX_RESIDUAL_FRACTION_DENOM > total:
                continue
            candidates.append((pr * pc, pr, pc, stats))

    assert candidates, "expected a near-periodic grid with non-black residual anomalies"
    min_area = min(area for area, _pr, _pc, _stats in candidates)
    best = [(pr, pc, stats) for area, pr, pc, stats in candidates if area == min_area]
    assert len(best) == 1, "expected a unique smallest period rectangle"
    return best[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Fold all off-pattern pixels from a periodic image into one period-sized anomaly map.

    Input structure:
    The grid is mostly filled by one two-dimensional periodic color tile, aligned
    to the upper-left corner. For each row/column residue class of the true
    smallest period, a strict majority of samples have the clean tile color, and
    any off-pattern samples in that same residue class share one non-black color.

    Rule:
    Infer the unique smallest period rectangle by testing candidate periods and
    taking the per-residue majority color as the clean tile. Mark every pixel
    whose color differs from that majority as an anomaly. Return a black grid
    with the same shape as the inferred period, painting each anomaly at its
    row/column residue with the anomaly pixel's own color.

    Key insight:
    The repeated pattern is only the carrier. The answer is the projection of
    sparse residual errors into residue space, not the clean periodic tile.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    pr, pc, stats = _infer_period(g)

    out = np.zeros((pr, pc), dtype=int)
    for (rr, cc), color in stats.anomaly_by_residue.items():
        out[rr, cc] = int(color)
    return out


def _tile_grid(tile: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    H, W = shape
    rr = np.arange(H) % tile.shape[0]
    cc = np.arange(W) % tile.shape[1]
    return tile[rr[:, None], cc[None, :]].copy()


def _has_exact_smaller_period(tile: np.ndarray) -> bool:
    h, w = tile.shape
    rows = np.arange(h)
    cols = np.arange(w)
    for pr in range(1, h + 1):
        for pc in range(1, w + 1):
            if pr * pc >= h * w:
                continue
            if np.array_equal(tile, tile[(rows % pr)[:, None], (cols % pc)[None, :]]):
                return True
    return False


def _has_low_residual_smaller_period(clean: np.ndarray, true_area: int) -> bool:
    total = int(clean.size)
    H, W = clean.shape
    for pr in range(1, H + 1):
        for pc in range(1, W + 1):
            if pr * pc >= true_area:
                continue
            stats = _period_stats(clean, pr, pc)
            if stats is None or stats.mismatch_count == 0:
                continue
            if stats.mismatch_count * MAX_RESIDUAL_FRACTION_DENOM <= total:
                return True
    return False


def _balanced_tile(tile: np.ndarray) -> bool:
    _, counts = np.unique(tile, return_counts=True)
    return int(counts.max()) * 5 <= int(tile.size) * 3


def _make_tile(rng: np.random.Generator, pr: int, pc: int) -> np.ndarray | None:
    area = pr * pc
    color_count = int(rng.integers(3, min(7, area) + 1))
    palette = [int(c) for c in rng.choice(np.array(ARC_COLORS), size=color_count, replace=False)]

    for _ in range(80):
        tile = rng.choice(np.array(palette), size=(pr, pc), replace=True).astype(int)
        order = [int(i) for i in rng.permutation(area)]
        for color, idx in zip(palette, order):
            tile[idx // pc, idx % pc] = int(color)
        if _balanced_tile(tile) and not _has_exact_smaller_period(tile):
            return tile
    return None


def _sample_shape(rng: np.random.Generator) -> tuple[int, int, int, int] | None:
    pr = int(rng.integers(2, 7))
    pc = int(rng.integers(2, 7))
    max_row_reps = 30 // pr
    max_col_reps = 30 // pc
    if max_row_reps < 3 or max_col_reps < 3:
        return None

    row_reps = int(rng.integers(3, max_row_reps + 1))
    col_reps = int(rng.integers(3, max_col_reps + 1))
    max_tail_r = min(pr - 1, 30 - pr * row_reps)
    max_tail_c = min(pc - 1, 30 - pc * col_reps)
    tail_r = int(rng.integers(0, max_tail_r + 1))
    tail_c = int(rng.integers(0, max_tail_c + 1))
    H = pr * row_reps + tail_r
    W = pc * col_reps + tail_c
    return H, W, pr, pc


def _positions_for_residue(shape: tuple[int, int], pr: int, pc: int, rr: int, cc: int) -> list[tuple[int, int]]:
    H, W = shape
    return [(r, c) for r in range(rr, H, pr) for c in range(cc, W, pc)]


def _inject_anomalies(
    rng: np.random.Generator,
    clean: np.ndarray,
    tile: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int] | None:
    H, W = clean.shape
    pr, pc = tile.shape
    residues = [(r, c) for r in range(pr) for c in range(pc)]
    rng.shuffle(residues)

    residue_count = int(rng.integers(2, min(7, len(residues)) + 1))
    chosen = residues[:residue_count]
    multi_residue = chosen[0]
    max_total_anomalies = max(3, int(clean.size) // 12)

    inp = clean.copy()
    expected = np.zeros((pr, pc), dtype=int)
    tile_colors = {int(c) for c in np.unique(tile)}
    total_anomalies = 0
    duplicate_fold_seen = False

    for rr, cc in chosen:
        positions = _positions_for_residue(clean.shape, pr, pc, rr, cc)
        max_here = min(3, (len(positions) - 1) // 3)
        if max_here < 1:
            return None
        if (rr, cc) == multi_residue and max_here >= 2 and total_anomalies + 2 <= max_total_anomalies:
            count = 2
        else:
            remaining = max_total_anomalies - total_anomalies
            if remaining <= 0:
                break
            count = int(rng.integers(1, min(max_here, remaining) + 1))

        normal_color = int(tile[rr, cc])
        color_pool = sorted((tile_colors - {normal_color}) & set(NON_BLACK))
        if not color_pool:
            color_pool = [c for c in NON_BLACK if c != normal_color]
        anomaly_color = int(rng.choice(np.array(color_pool)))

        picked = rng.choice(len(positions), size=count, replace=False)
        for idx in picked:
            r, c = positions[int(idx)]
            inp[r, c] = anomaly_color
        expected[rr, cc] = anomaly_color
        total_anomalies += count
        duplicate_fold_seen = duplicate_fold_seen or count > 1

    if total_anomalies < 2 or not duplicate_fold_seen:
        return None
    if total_anomalies * MAX_RESIDUAL_FRACTION_DENOM > int(clean.size):
        return None
    return inp, expected, total_anomalies


def _oracle_accepts(inp: np.ndarray, expected: np.ndarray, period: tuple[int, int]) -> bool:
    try:
        actual = main(inp)
    except Exception:
        return False
    if actual.shape != period:
        return False
    if not np.array_equal(actual, expected):
        return False
    if int(np.count_nonzero(actual)) < 2:
        return False
    if len(set(int(c) for c in actual.ravel()) - {BLACK}) < 1:
        return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    sampled = _sample_shape(rng)
    if sampled is None:
        return None
    H, W, pr, pc = sampled

    tile = _make_tile(rng, pr, pc)
    if tile is None:
        return None

    clean = _tile_grid(tile, (H, W))
    if _has_low_residual_smaller_period(clean, pr * pc):
        return None

    injected = _inject_anomalies(rng, clean, tile)
    if injected is None:
        return None
    inp, expected, _count = injected

    if np.array_equal(expected, tile):
        return None
    if _oracle_accepts(inp, expected, (pr, pc)):
        return inp
    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid periodic residual folding input")
