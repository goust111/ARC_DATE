from __future__ import annotations

from collections import Counter
import random

import numpy as np


MAGENTA = 6
PERIOD_OPTIONS = (2, 3)
INPUT_COLORS = tuple(color for color in range(10) if color != MAGENTA)


def _majority_color(values: np.ndarray) -> int:
    counts = Counter(map(int, values.ravel()))
    ranked = counts.most_common()
    assert ranked, "each tile position must have at least one sample"
    if len(ranked) > 1:
        assert ranked[0][1] != ranked[1][1], "majority vote must be unique"
    return int(ranked[0][0])


def _infer_candidate(grid: np.ndarray, period: int) -> dict[str, np.ndarray | int] | None:
    height, width = grid.shape
    if height % period != 0 or width % period != 0:
        return None

    tile = np.empty((period, period), dtype=int)
    try:
        for r in range(period):
            for c in range(period):
                tile[r, c] = _majority_color(grid[r::period, c::period])
    except AssertionError:
        return None

    recon = np.tile(tile, (height // period, width // period))
    mismatch = recon != grid
    return {
        "period": int(period),
        "tile": tile,
        "recon": recon,
        "mismatch": mismatch,
        "count": int(np.count_nonzero(mismatch)),
    }


def _best_candidate(grid: np.ndarray) -> tuple[dict[str, np.ndarray | int], list[dict[str, np.ndarray | int]]]:
    candidates = [cand for period in PERIOD_OPTIONS if (cand := _infer_candidate(grid, period)) is not None]
    assert candidates, "expected a valid 2x2 or 3x3 periodic explanation"
    candidates.sort(key=lambda cand: (int(cand["count"]), int(cand["period"])))
    return candidates[0], candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark the pixels that violate a hidden periodic tile with magenta.

    Input structure:
    - The whole grid is formed by periodically tiling one small square tile of size 2x2 or 3x3.
    - A few cells have been recolored to other already-used input colors, so color identity alone does not reveal the errors.
    - Magenta is absent from the input and is reserved for the output markers.

    Rule:
    - For each feasible tile size, infer the tile by majority vote over every modulo position in the tiling.
    - Choose the smallest-period explanation with the fewest mismatching cells.
    - Recolor every cell that disagrees with that inferred periodic pattern to magenta.
    - Leave all matching cells unchanged.

    Key insight:
    - The wrong pixels are only detectable through disagreement with a repeated spatial template; the template itself must be reconstructed from the noisy repetitions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    assert not np.any(g == MAGENTA), "magenta must be reserved for the output"

    best, _all = _best_candidate(g)
    out = g.copy()
    out[best["mismatch"]] = MAGENTA
    return out


# --------------- generator ---------------

GEN_RETRIES = 400
AMBIGUOUS_DIM_PROBABILITY = 0.6
MIN_ALT_MARGIN = 2
MIN_REPEAT = {2: 2, 3: 2}


def _has_smaller_axis_period(tile: np.ndarray, axis: int) -> bool:
    size = tile.shape[axis]
    for period in range(1, size):
        if size % period != 0:
            continue
        ok = True
        for idx in range(size):
            if axis == 0:
                if not np.array_equal(tile[idx, :], tile[idx % period, :]):
                    ok = False
                    break
            else:
                if not np.array_equal(tile[:, idx], tile[:, idx % period]):
                    ok = False
                    break
        if ok:
            return True
    return False


def _is_minimal_tile(tile: np.ndarray) -> bool:
    return (
        len(set(map(int, tile.ravel()))) >= 2
        and not _has_smaller_axis_period(tile, axis=0)
        and not _has_smaller_axis_period(tile, axis=1)
    )


def _random_tile(period: int) -> tuple[np.ndarray, list[int]] | None:
    area = period * period
    max_colors = min(5, area)
    color_count = random.randint(2, max_colors)
    palette = random.sample(INPUT_COLORS, color_count)

    flat = palette[:]
    flat.extend(random.choice(palette) for _ in range(area - len(flat)))
    random.shuffle(flat)
    tile = np.array(flat, dtype=int).reshape(period, period)
    if not _is_minimal_tile(tile):
        return None
    return tile, palette


def _choose_shape(period: int) -> tuple[int, int]:
    max_repeat = 30 // period
    repeats = list(range(MIN_REPEAT[period], max_repeat + 1))

    if random.random() < AMBIGUOUS_DIM_PROBABILITY:
        if period == 2:
            allowed = [repeat for repeat in repeats if repeat % 3 == 0]
        else:
            allowed = [repeat for repeat in repeats if repeat % 2 == 0]
        if allowed:
            return period * random.choice(allowed), period * random.choice(allowed)

    while True:
        height_repeat = random.choice(repeats)
        width_repeat = random.choice(repeats)
        if period == 2:
            if height_repeat % 3 != 0 or width_repeat % 3 != 0:
                return period * height_repeat, period * width_repeat
        else:
            if height_repeat % 2 != 0 or width_repeat % 2 != 0:
                return period * height_repeat, period * width_repeat


def _repeat_tile(tile: np.ndarray, height: int, width: int) -> np.ndarray:
    period = tile.shape[0]
    return np.tile(tile, (height // period, width // period))


def _residue_positions(height: int, width: int, period: int) -> dict[tuple[int, int], list[tuple[int, int]]]:
    return {
        (r, c): [(rr, cc) for rr in range(r, height, period) for cc in range(c, width, period)]
        for r in range(period)
        for c in range(period)
    }


def _inject_errors(clean: np.ndarray, palette: list[int], period: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    height, width = clean.shape
    slots = _residue_positions(height, width, period)
    error_count = random.randint(1, min(5, period * period))
    chosen_classes = random.sample(list(slots), error_count)

    noisy = clean.copy()
    changed: list[tuple[int, int]] = []
    for cls in chosen_classes:
        r, c = random.choice(slots[cls])
        original = int(noisy[r, c])
        choices = [color for color in palette if color != original]
        assert choices
        noisy[r, c] = random.choice(choices)
        changed.append((int(r), int(c)))
    return noisy, changed


def _expected_output(grid: np.ndarray, errors: list[tuple[int, int]]) -> np.ndarray:
    out = grid.copy()
    for r, c in errors:
        out[r, c] = MAGENTA
    return out


def _gen_attempt() -> np.ndarray | None:
    period = random.choice(PERIOD_OPTIONS)
    tile_and_palette = _random_tile(period)
    if tile_and_palette is None:
        return None
    tile, palette = tile_and_palette

    height, width = _choose_shape(period)
    clean = _repeat_tile(tile, height, width)
    noisy, errors = _inject_errors(clean, palette, period)
    expected = _expected_output(noisy, errors)

    try:
        best, candidates = _best_candidate(noisy)
        solved = main(noisy)
    except AssertionError:
        return None

    if int(best["period"]) != period:
        return None
    if int(best["count"]) != len(errors):
        return None
    if len(candidates) > 1 and int(candidates[1]["count"]) < len(errors) + MIN_ALT_MARGIN:
        return None
    if not np.array_equal(solved, expected):
        return None
    return noisy


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid periodic-anomaly ARC input")
