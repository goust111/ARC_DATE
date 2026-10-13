import random
from collections import Counter

import numpy as np


GREEN = 3
INPUT_COLORS = (1, 2, 4, 5, 6, 7, 8, 9)
MAX_ANOMALY_RATIO = 0.12
MIN_COMPETITOR_RATIO = 0.18
GEN_ATTEMPTS = 600
TILE_ATTEMPTS = 250
PAIR_ATTEMPTS = 250


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def _majority_value(values: np.ndarray) -> tuple[int, bool]:
    counts = Counter(map(int, values.ravel()))
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    strict = len(ranked) == 1 or ranked[0][1] > ranked[1][1]
    return int(ranked[0][0]), strict


def _infer_tile(grid: np.ndarray, period_r: int, period_c: int) -> tuple[np.ndarray, bool]:
    tile = np.empty((period_r, period_c), dtype=int)
    strict = True
    for r in range(period_r):
        for c in range(period_c):
            tile[r, c], cell_strict = _majority_value(grid[r::period_r, c::period_c])
            strict &= cell_strict
    return tile, strict


def _mismatch_ratio(grid: np.ndarray, period_r: int, period_c: int) -> float:
    tile, _strict = _infer_tile(grid, period_r, period_c)
    expected = np.tile(tile, (grid.shape[0] // period_r, grid.shape[1] // period_c))
    return float(np.count_nonzero(grid != expected)) / float(grid.size)


def _minimal_row_period(tile: np.ndarray) -> int:
    height = tile.shape[0]
    for period in _divisors(height):
        seed = tile[:period, :]
        if np.array_equal(tile, np.tile(seed, (height // period, 1))):
            return period
    return height


def _minimal_col_period(tile: np.ndarray) -> int:
    width = tile.shape[1]
    for period in _divisors(width):
        seed = tile[:, :period]
        if np.array_equal(tile, np.tile(seed, (1, width // period))):
            return period
    return width


def _exact_periods(tile: np.ndarray) -> tuple[int, int]:
    return _minimal_row_period(tile), _minimal_col_period(tile)


def _extract_pairs(coords: np.ndarray) -> list[tuple[tuple[int, int], tuple[int, int]]] | None:
    cells = [tuple(map(int, coord)) for coord in coords]
    if not cells or len(cells) % 2:
        return None

    rows: dict[int, list[tuple[int, int]]] = {}
    cols: dict[int, list[tuple[int, int]]] = {}
    for r, c in cells:
        rows.setdefault(r, []).append((r, c))
        cols.setdefault(c, []).append((r, c))

    if any(len(group) > 2 for group in rows.values()):
        return None
    if any(len(group) > 2 for group in cols.values()):
        return None

    pair_set: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    used: set[tuple[int, int]] = set()
    for cell in cells:
        r, c = cell
        row_pair = len(rows[r]) == 2
        col_pair = len(cols[c]) == 2
        if row_pair == col_pair:
            return None
        pair = tuple(sorted(rows[r] if row_pair else cols[c]))
        pair_set.add(pair)
        used.update(pair)

    if used != set(cells):
        return None
    return sorted(pair_set)


def _candidate_info(grid: np.ndarray, period_r: int, period_c: int) -> dict | None:
    height, width = grid.shape
    if height % period_r or width % period_c:
        return None

    repeat_r = height // period_r
    repeat_c = width // period_c
    if repeat_r < 2 or repeat_c < 2:
        return None

    tile, strict = _infer_tile(grid, period_r, period_c)
    tile_colors = sorted(set(map(int, tile.ravel())))
    if not strict or len(tile_colors) != 2 or GREEN in tile_colors:
        return None

    expected = np.tile(tile, (repeat_r, repeat_c))
    diff = np.argwhere(grid != expected)
    diff_count = int(diff.shape[0])
    if diff_count == 0:
        return None
    if float(diff_count) / float(grid.size) > MAX_ANOMALY_RATIO:
        return None

    pairs = _extract_pairs(diff)
    if pairs is None:
        return None

    return {
        "periods": (period_r, period_c),
        "area": int(period_r * period_c),
        "tile": tile,
        "expected": expected,
        "diff_count": diff_count,
        "pairs": pairs,
    }


def _find_candidates(grid: np.ndarray) -> list[dict]:
    height, width = grid.shape
    candidates = []
    for period_r in _divisors(height):
        for period_c in _divisors(width):
            info = _candidate_info(grid, period_r, period_c)
            if info is not None:
                candidates.append(info)
    candidates.sort(key=lambda info: (info["area"], info["diff_count"], info["periods"]))
    return candidates


def _select_candidate(grid: np.ndarray) -> dict:
    candidates = _find_candidates(grid)
    assert candidates, "no valid periodic explanation found"
    best = candidates[0]
    if len(candidates) > 1:
        runner_up = candidates[1]
        assert (best["area"], best["diff_count"]) != (
            runner_up["area"],
            runner_up["diff_count"],
        ), "ambiguous periodic explanation"
    return best


def _render_output(base: np.ndarray, pairs: list[tuple[tuple[int, int], tuple[int, int]]]) -> np.ndarray:
    out = base.copy()
    for (r0, c0), (r1, c1) in pairs:
        if r0 == r1:
            out[r0, c0 + 1 : c1] = GREEN
        else:
            out[r0 + 1 : r1, c0] = GREEN
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover a corrupted two-color periodic tiling and connect each aligned anomaly pair with green.

    Input structure:
    - The full grid is a rectangular tiling of one unknown minimal 2-color tile.
    - A small even number of cells have been flipped to the other tile color, so no new input color appears.
    - Those flipped cells form disjoint row-pairs or column-pairs; green is absent from the input.

    Rule:
    - Infer the unique smallest repeating tile that still explains almost all cells.
    - Restore every flipped endpoint to its expected periodic color.
    - For each horizontal or vertical endpoint pair, paint the cells strictly between them green.

    Key insight:
    - The endpoints are invisible until the global period is reconstructed; only the periodic mismatches reveal where each connector must start and end.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and 0 < g.shape[0] <= 30 and 0 < g.shape[1] <= 30

    colors = sorted(set(map(int, g.ravel())))
    assert len(colors) == 2, "input must contain exactly two visible colors"
    assert GREEN not in colors, "green is reserved for output connectors"

    best = _select_candidate(g)
    return _render_output(best["expected"], best["pairs"])


_SHAPE_SPECS = [
    (period_r, period_c, repeat_r, repeat_c)
    for period_r in range(1, 7)
    for period_c in range(1, 7)
    for repeat_r in range(2, 6)
    for repeat_c in range(2, 6)
    if period_r * period_c >= 2
    if 8 <= period_r * repeat_r <= 30
    if 8 <= period_c * repeat_c <= 30
]


def _min_competitor_ratio(base: np.ndarray, true_period_r: int, true_period_c: int) -> float:
    true_area = true_period_r * true_period_c
    best = 1.0
    for period_r in _divisors(base.shape[0]):
        for period_c in _divisors(base.shape[1]):
            if (period_r, period_c) == (true_period_r, true_period_c):
                continue
            if base.shape[0] // period_r < 2 or base.shape[1] // period_c < 2:
                continue
            if period_r * period_c > true_area:
                continue
            best = min(best, _mismatch_ratio(base, period_r, period_c))
    return best


def _generate_tile(period_r: int, period_c: int, repeat_r: int, repeat_c: int, colors: tuple[int, int]) -> np.ndarray | None:
    for _ in range(TILE_ATTEMPTS):
        mask = np.random.randint(0, 2, size=(period_r, period_c))
        filled = int(mask.sum())
        if filled == 0 or filled == period_r * period_c:
            continue
        ratio = float(filled) / float(period_r * period_c)
        if ratio < 0.30 or ratio > 0.70:
            continue

        tile = np.where(mask == 0, colors[0], colors[1]).astype(int)
        if _exact_periods(tile) != (period_r, period_c):
            continue

        base = np.tile(tile, (repeat_r, repeat_c))
        if _min_competitor_ratio(base, period_r, period_c) < MIN_COMPETITOR_RATIO:
            continue
        return tile
    return None


def _pick_disjoint_pairs(values: list[int], pair_count: int, min_gap: int) -> list[tuple[int, int]] | None:
    values = sorted(set(values))
    if pair_count == 0:
        return []

    valid_pairs = [(a, b) for i, a in enumerate(values) for b in values[i + 1 :] if b - a >= min_gap]
    if len(valid_pairs) < pair_count:
        return None

    def search(used: set[int], need: int) -> list[tuple[int, int]] | None:
        if need == 0:
            return []
        options = valid_pairs[:]
        random.shuffle(options)
        for a, b in options:
            if a in used or b in used:
                continue
            tail = search(used | {a, b}, need - 1)
            if tail is not None:
                return [(a, b)] + tail
        return None

    return search(set(), pair_count)


def _choose_pair_mix(height: int, width: int, tile_area: int) -> tuple[int, int] | None:
    max_pairs = min(3, tile_area // 2)
    options = []
    weights = []
    for horizontal_count in range(max_pairs + 1):
        for vertical_count in range(max_pairs + 1 - horizontal_count):
            total = horizontal_count + vertical_count
            if total == 0:
                continue
            if 2 * horizontal_count + vertical_count > width:
                continue
            if horizontal_count + 2 * vertical_count > height:
                continue
            if 2 * total > tile_area:
                continue
            options.append((horizontal_count, vertical_count))
            if horizontal_count and vertical_count:
                weights.append(5)
            elif total >= 2:
                weights.append(3)
            else:
                weights.append(2)
    if not options:
        return None
    return random.choices(options, weights=weights, k=1)[0]


def _place_pairs(shape: tuple[int, int], period_r: int, period_c: int, horizontal_count: int, vertical_count: int) -> list[tuple[tuple[int, int], tuple[int, int]]] | None:
    height, width = shape
    for _ in range(PAIR_ATTEMPTS):
        if horizontal_count and vertical_count:
            split = random.randint(horizontal_count, height - 2 * vertical_count)
            horizontal_rows_pool = list(range(split))
            vertical_rows_pool = list(range(split, height))
        else:
            horizontal_rows_pool = list(range(height))
            vertical_rows_pool = list(range(height))

        if len(horizontal_rows_pool) < horizontal_count or len(vertical_rows_pool) < 2 * vertical_count:
            continue

        horizontal_rows = random.sample(horizontal_rows_pool, horizontal_count) if horizontal_count else []
        vertical_cols = random.sample(range(width), vertical_count) if vertical_count else []

        horizontal_cols_pool = [c for c in range(width) if c not in vertical_cols]
        vertical_rows_candidates = [r for r in vertical_rows_pool if r not in horizontal_rows]
        horizontal_col_pairs = _pick_disjoint_pairs(horizontal_cols_pool, horizontal_count, min_gap=3)
        vertical_row_pairs = _pick_disjoint_pairs(vertical_rows_candidates, vertical_count, min_gap=3)
        if horizontal_col_pairs is None or vertical_row_pairs is None:
            continue

        random.shuffle(horizontal_rows)
        random.shuffle(vertical_cols)
        random.shuffle(horizontal_col_pairs)
        random.shuffle(vertical_row_pairs)

        residues: set[tuple[int, int]] = set()
        pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
        ok = True

        for row, (c0, c1) in zip(horizontal_rows, horizontal_col_pairs):
            endpoints = tuple(sorted(((row, c0), (row, c1))))
            for cell in endpoints:
                residue = (cell[0] % period_r, cell[1] % period_c)
                if residue in residues:
                    ok = False
                    break
                residues.add(residue)
            if not ok:
                break
            pairs.append(endpoints)
        if not ok:
            continue

        for col, (r0, r1) in zip(vertical_cols, vertical_row_pairs):
            endpoints = tuple(sorted(((r0, col), (r1, col))))
            for cell in endpoints:
                residue = (cell[0] % period_r, cell[1] % period_c)
                if residue in residues:
                    ok = False
                    break
                residues.add(residue)
            if not ok:
                break
            pairs.append(endpoints)
        if not ok:
            continue

        return sorted(pairs)
    return None


def _flip_to_other(color: int, colors: tuple[int, int]) -> int:
    return colors[1] if int(color) == int(colors[0]) else colors[0]


def _instance_valid(
    inp: np.ndarray,
    base: np.ndarray,
    pairs: list[tuple[tuple[int, int], tuple[int, int]]],
    true_period_r: int,
    true_period_c: int,
) -> bool:
    colors = sorted(set(map(int, base.ravel())))
    if len(colors) != 2 or GREEN in colors:
        return False

    if inp.shape != base.shape or inp.ndim != 2 or max(inp.shape) > 30:
        return False
    if set(map(int, inp.ravel())) != set(colors):
        return False

    tile = base[:true_period_r, :true_period_c]
    if _exact_periods(tile) != (true_period_r, true_period_c):
        return False
    if not np.array_equal(base, np.tile(tile, (base.shape[0] // true_period_r, base.shape[1] // true_period_c))):
        return False
    if _min_competitor_ratio(base, true_period_r, true_period_c) < MIN_COMPETITOR_RATIO:
        return False

    endpoints = {cell for pair in pairs for cell in pair}
    if not endpoints or len(endpoints) != 2 * len(pairs):
        return False
    if set(map(tuple, np.argwhere(inp != base))) != endpoints:
        return False
    if _extract_pairs(np.argwhere(inp != base)) != sorted(pairs):
        return False

    for (r0, c0), (r1, c1) in pairs:
        if r0 == r1:
            if c1 - c0 < 3:
                return False
        elif c0 == c1:
            if r1 - r0 < 3:
                return False
        else:
            return False

    for r, c in endpoints:
        if inp[r, c] == base[r, c]:
            return False
        if {int(inp[r, c]), int(base[r, c])} != set(colors):
            return False

    candidates = _find_candidates(inp)
    if not candidates:
        return False
    if candidates[0]["periods"] != (true_period_r, true_period_c):
        return False
    if not np.array_equal(candidates[0]["expected"], base):
        return False
    if len(candidates) > 1:
        if (candidates[0]["area"], candidates[0]["diff_count"]) == (
            candidates[1]["area"],
            candidates[1]["diff_count"],
        ):
            return False

    expected = _render_output(base, pairs)
    try:
        actual = main(inp)
    except AssertionError:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    period_r, period_c, repeat_r, repeat_c = random.choice(_SHAPE_SPECS)
    colors = tuple(random.sample(INPUT_COLORS, 2))

    tile = _generate_tile(period_r, period_c, repeat_r, repeat_c, colors)
    if tile is None:
        return None

    base = np.tile(tile, (repeat_r, repeat_c))
    pair_mix = _choose_pair_mix(base.shape[0], base.shape[1], tile.size)
    if pair_mix is None:
        return None
    horizontal_count, vertical_count = pair_mix

    pairs = _place_pairs(base.shape, period_r, period_c, horizontal_count, vertical_count)
    if pairs is None:
        return None

    inp = base.copy()
    for pair in pairs:
        for r, c in pair:
            inp[r, c] = _flip_to_other(base[r, c], colors)

    if not _instance_valid(inp, base, pairs, period_r, period_c):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid periodic-endpoint instance")
