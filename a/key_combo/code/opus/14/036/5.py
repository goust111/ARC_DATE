from __future__ import annotations

from collections import Counter
import hashlib

import numpy as np


SEPARATOR = 5
PALETTE = (1, 2, 3)
GEN_ATTEMPTS = 2000
_USED_INPUT_SIGS: set[str] = set()
_USED_OUTPUT_SIGS: set[str] = set()


def _count_bands(separator_mask: np.ndarray) -> int:
    bands = 0
    in_band = False
    for is_separator in map(bool, separator_mask):
        if is_separator:
            in_band = False
        elif not in_band:
            bands += 1
            in_band = True
    return bands


def _majority_color(values: np.ndarray) -> int:
    counts = Counter(map(int, np.asarray(values).flat))
    top = counts.most_common()
    assert len(top) >= 1
    best_count = int(top[0][1])
    return int(min(color for color, count in top if int(count) == best_count))


def _compressed_core(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sep_rows = np.all(grid == SEPARATOR, axis=1)
    sep_cols = np.all(grid == SEPARATOR, axis=0)
    rows = np.flatnonzero(~sep_rows)
    cols = np.flatnonzero(~sep_cols)
    assert rows.size > 0 and cols.size > 0
    core = np.asarray(grid[np.ix_(rows, cols)], dtype=int)
    assert core.size > 0
    assert not np.any(core == SEPARATOR)
    return rows, cols, core


def _tile_for_dims(core: np.ndarray, tile_h: int, tile_w: int) -> tuple[np.ndarray, int]:
    h, w = map(int, core.shape)
    tile = np.zeros((tile_h, tile_w), dtype=int)
    mismatches = 0
    for r0 in range(tile_h):
        for c0 in range(tile_w):
            values = core[r0:h:tile_h, c0:w:tile_w]
            color = _majority_color(values)
            tile[r0, c0] = color
            mismatches += int(np.count_nonzero(values != color))
    return tile, mismatches


def _reconstruct_from_tile(tile: np.ndarray, repeats_y: int, repeats_x: int) -> np.ndarray:
    return np.tile(tile, (int(repeats_y), int(repeats_x)))


def _smallest_period(tile: np.ndarray) -> tuple[int, int]:
    h, w = map(int, tile.shape)
    for ph in range(1, h + 1):
        for pw in range(1, w + 1):
            if ph == h and pw == w:
                return h, w
            ok = True
            for r in range(h):
                for c in range(w):
                    if int(tile[r, c]) != int(tile[r % ph, c % pw]):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return ph, pw
    return h, w


def _raw_coordinate_tile(
    grid: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    tile_h: int,
    tile_w: int,
) -> tuple[np.ndarray | None, int | None]:
    tile = np.zeros((tile_h, tile_w), dtype=int)
    mismatches = 0
    for r0 in range(tile_h):
        for c0 in range(tile_w):
            values = []
            for r in rows:
                if int(r) % tile_h != r0:
                    continue
                for c in cols:
                    if int(c) % tile_w == c0:
                        values.append(int(grid[int(r), int(c)]))
            if not values:
                return None, None
            color = _majority_color(np.array(values, dtype=int))
            tile[r0, c0] = color
            mismatches += sum(int(v != color) for v in values)
    return tile, mismatches


def _nontrivial_alternative(
    core: np.ndarray,
    true_h: int,
    true_w: int,
) -> tuple[tuple[int, int] | None, int]:
    h, w = map(int, core.shape)
    best_dims = None
    best_mismatch = None
    for cand_h in range(1, h // 2 + 1):
        for cand_w in range(1, w // 2 + 1):
            if cand_h == true_h and cand_w == true_w:
                continue
            if cand_h == 1 and cand_w == 1:
                continue
            tile, mismatches = _tile_for_dims(core, cand_h, cand_w)
            if tile.shape == (true_h, true_w) and np.array_equal(tile, _tile_for_dims(core, true_h, true_w)[0]):
                continue
            if best_mismatch is None or mismatches < best_mismatch:
                best_dims = (cand_h, cand_w)
                best_mismatch = mismatches
    return best_dims, -1 if best_mismatch is None else int(best_mismatch)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count separator-created row/column bands to decode the hidden tile size, then repair the noisy periodic pattern and output the clean tile.

    Input structure:
    - Grey (5) full rows and full columns act as separator lines across the whole grid.
    - Every non-separator cell belongs to one global periodic color pattern drawn with colors from the blue/red/green subset and corrupted by sparse wrong-color noise.
    - The number of contiguous non-separator row bands is the tile height, and the number of contiguous non-separator column bands is the tile width.

    Rule:
    - Detect all complete grey rows and complete grey columns.
    - Count contiguous runs of non-grey rows to get R, and contiguous runs of non-grey columns to get C.
    - Remove the separator rows and columns conceptually, so residue positions are measured in the compressed non-separator row/column order.
    - For each residue class (r mod R, c mod C), take the majority color among all mapped cells and place that color in the output tile.

    Key insight:
    - The separators do not describe the output directly; they encode the repair parameter R x C, and the pattern can only be denoised uniquely when residues are computed in the separator-compressed coordinate system.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    sep_rows = np.all(g == SEPARATOR, axis=1)
    sep_cols = np.all(g == SEPARATOR, axis=0)
    assert np.any(sep_rows) and np.any(sep_cols), "expected full grey separator rows and columns"

    tile_h = _count_bands(sep_rows)
    tile_w = _count_bands(sep_cols)
    assert tile_h >= 1 and tile_w >= 1

    rows, cols, core = _compressed_core(g)
    assert core.shape[0] >= 2 * tile_h and core.shape[1] >= 2 * tile_w

    tile, _ = _tile_for_dims(core, tile_h, tile_w)
    return tile


def _random_composition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [int(total)]
    cuts = np.sort(np.random.choice(np.arange(1, total), size=parts - 1, replace=False))
    return np.diff(np.concatenate(([0], cuts, [total]))).astype(int).tolist()


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def _random_base_pattern(h: int, w: int, colors: tuple[int, int]) -> np.ndarray:
    a, b = map(int, colors)
    while True:
        tile = np.random.choice([a, b], size=(h, w)).astype(int)
        if len(set(map(int, tile.flat))) == 2:
            return tile


def _make_deceptive_tile(tile_h: int, tile_w: int, colors: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int], np.ndarray]:
    candidates = []
    for cand_h in _divisors(tile_h):
        for cand_w in _divisors(tile_w):
            if (cand_h, cand_w) == (tile_h, tile_w):
                continue
            if cand_h == 1 and cand_w == 1:
                continue
            candidates.append((cand_h, cand_w))
    np.random.shuffle(candidates)

    for alt_h, alt_w in candidates:
        for _ in range(300):
            base = _random_base_pattern(alt_h, alt_w, colors)
            repeated = np.tile(base, (tile_h // alt_h, tile_w // alt_w))
            max_edits = min(4, tile_h * tile_w - 1)
            order = np.random.permutation(tile_h * tile_w)
            for edits in range(1, max_edits + 1):
                tile = repeated.copy()
                for flat in order[:edits]:
                    r, c = divmod(int(flat), tile_w)
                    current = int(tile[r, c])
                    tile[r, c] = int(colors[0] if current == int(colors[1]) else colors[1])
                if len(set(map(int, tile.flat))) != 2:
                    continue
                if _smallest_period(tile) != (tile_h, tile_w):
                    continue
                return tile, (int(alt_h), int(alt_w)), base
    raise AssertionError("failed to construct a deceptive primitive tile")


def _add_noise(
    clean_core: np.ndarray,
    tile_h: int,
    tile_w: int,
    noise_ratio: float,
) -> np.ndarray:
    noisy = clean_core.copy()
    reps = (clean_core.shape[0] // tile_h) * (clean_core.shape[1] // tile_w)
    max_per_residue = (reps - 1) // 2
    assert max_per_residue >= 1

    residue_positions: list[list[tuple[int, int]]] = []
    for r0 in range(tile_h):
        for c0 in range(tile_w):
            pts = [(r, c) for r in range(r0, clean_core.shape[0], tile_h) for c in range(c0, clean_core.shape[1], tile_w)]
            residue_positions.append(pts)

    total_cells = int(clean_core.size)
    total_noise = int(np.random.randint(max(1, int(np.ceil(total_cells * noise_ratio)) - 1), int(np.floor(total_cells * noise_ratio)) + 2))
    total_noise = max(1, min(total_noise, len(residue_positions) * max_per_residue))

    capacities = np.array([max_per_residue] * len(residue_positions), dtype=int)
    counts = np.zeros(len(residue_positions), dtype=int)
    for _ in range(total_noise):
        choices = np.flatnonzero(capacities > 0)
        assert choices.size > 0
        weights = capacities[choices].astype(float)
        weights /= weights.sum()
        idx = int(np.random.choice(choices, p=weights))
        counts[idx] += 1
        capacities[idx] -= 1

    for residue_idx, count in enumerate(counts.tolist()):
        if count == 0:
            continue
        positions = residue_positions[residue_idx]
        chosen = np.random.choice(len(positions), size=count, replace=False)
        for pos_idx in chosen.tolist():
            r, c = positions[int(pos_idx)]
            current = int(noisy[r, c])
            replacement_pool = [color for color in PALETTE if int(color) != current]
            noisy[r, c] = int(np.random.choice(replacement_pool))

    return noisy


def _separator_layout(non_sep_total: int, bands: int) -> tuple[int, list[int], int]:
    top = int(np.random.choice([0, 1], p=[0.65, 0.35]))
    bottom = int(np.random.choice([0, 1], p=[0.65, 0.35]))
    between = [int(np.random.choice([1, 2], p=[0.65, 0.35])) for _ in range(max(0, bands - 1))]
    if top == 0 and bottom == 0 and all(width == 1 for width in between):
        between[int(np.random.randint(0, len(between)))] = 2
    return top, between, bottom


def _build_grid(core: np.ndarray, row_bands: list[int], col_bands: list[int], row_sep: tuple[int, list[int], int], col_sep: tuple[int, list[int], int]) -> np.ndarray:
    row_top, row_between, row_bottom = row_sep
    col_left, col_between, col_right = col_sep
    h = int(core.shape[0] + row_top + row_bottom + sum(row_between))
    w = int(core.shape[1] + col_left + col_right + sum(col_between))

    sep_rows = np.ones(h, dtype=bool)
    sep_cols = np.ones(w, dtype=bool)

    cursor = row_top
    row_positions = []
    for band_idx, band_size in enumerate(row_bands):
        row_positions.extend(range(cursor, cursor + int(band_size)))
        cursor += int(band_size)
        sep_rows[cursor - int(band_size):cursor] = False
        if band_idx < len(row_between):
            cursor += int(row_between[band_idx])

    cursor = col_left
    col_positions = []
    for band_idx, band_size in enumerate(col_bands):
        col_positions.extend(range(cursor, cursor + int(band_size)))
        cursor += int(band_size)
        sep_cols[cursor - int(band_size):cursor] = False
        if band_idx < len(col_between):
            cursor += int(col_between[band_idx])

    grid = np.full((h, w), SEPARATOR, dtype=int)
    grid[np.ix_(np.array(row_positions, dtype=int), np.array(col_positions, dtype=int))] = core
    return grid


def _validate_instance(
    grid: np.ndarray,
    expected_tile: np.ndarray,
    alt_dims: tuple[int, int],
) -> tuple[str, str]:
    assert grid.ndim == 2 and max(grid.shape) <= 30
    rows, cols, core = _compressed_core(grid)
    tile_h, tile_w = map(int, expected_tile.shape)
    assert _count_bands(np.all(grid == SEPARATOR, axis=1)) == tile_h
    assert _count_bands(np.all(grid == SEPARATOR, axis=0)) == tile_w
    assert core.shape[0] % tile_h == 0 and core.shape[1] % tile_w == 0
    assert core.shape[0] >= 2 * tile_h and core.shape[1] >= 2 * tile_w

    got = main(grid)
    assert np.array_equal(got, expected_tile), "solver did not recover the intended tile"
    assert _smallest_period(expected_tile) == (tile_h, tile_w), "tile must be primitive at the encoded size"

    true_tile, true_mismatch = _tile_for_dims(core, tile_h, tile_w)
    assert np.array_equal(true_tile, expected_tile)

    repeats = (core.shape[0] // tile_h) * (core.shape[1] // tile_w)
    for r0 in range(tile_h):
        for c0 in range(tile_w):
            values = core[r0:core.shape[0]:tile_h, c0:core.shape[1]:tile_w]
            counts = Counter(map(int, values.flat))
            top = counts.most_common(2)
            assert len(top) >= 1
            if len(top) >= 2:
                assert top[0][1] > top[1][1], "true residue class must have a strict majority"
            assert int(top[0][0]) == int(expected_tile[r0, c0])

    raw_tile, raw_mismatch = _raw_coordinate_tile(grid, rows, cols, tile_h, tile_w)
    if raw_tile is not None and raw_mismatch is not None:
        assert raw_mismatch > true_mismatch or not np.array_equal(raw_tile, expected_tile), "must depend on compressed coordinates, not raw grid coordinates"

    alt_h, alt_w = map(int, alt_dims)
    alt_tile, alt_mismatch = _tile_for_dims(core, alt_h, alt_w)
    assert not np.array_equal(alt_tile, expected_tile)
    assert alt_mismatch <= true_mismatch + repeats * 4, "alternative size should remain plausible enough to force counting"

    best_alt_dims, best_alt_mismatch = _nontrivial_alternative(core, tile_h, tile_w)
    assert best_alt_dims is not None
    assert best_alt_mismatch <= true_mismatch + repeats * 5

    inp_sig = hashlib.md5(grid.tobytes() + str(grid.shape).encode()).hexdigest()
    out_sig = hashlib.md5(expected_tile.tobytes() + str(expected_tile.shape).encode()).hexdigest()
    return inp_sig, out_sig


def generate_input() -> np.ndarray:
    global _USED_INPUT_SIGS, _USED_OUTPUT_SIGS

    for _ in range(GEN_ATTEMPTS):
        tile_h = int(np.random.randint(2, 6))
        tile_w = int(np.random.randint(2, 6))
        reps_y = int(np.random.randint(2, 5))
        reps_x = int(np.random.randint(2, 5))
        core_h = tile_h * reps_y
        core_w = tile_w * reps_x

        row_bands = _random_composition(core_h, tile_h)
        col_bands = _random_composition(core_w, tile_w)
        row_sep = _separator_layout(core_h, tile_h)
        col_sep = _separator_layout(core_w, tile_w)

        total_h = core_h + row_sep[0] + row_sep[2] + sum(row_sep[1])
        total_w = core_w + col_sep[0] + col_sep[2] + sum(col_sep[1])
        if total_h > 30 or total_w > 30:
            continue

        colors = tuple(map(int, np.random.choice(PALETTE, size=2, replace=False)))
        clean_tile, alt_dims, _alt_base = _make_deceptive_tile(tile_h, tile_w, colors)
        clean_core = _reconstruct_from_tile(clean_tile, reps_y, reps_x)
        noisy_core = _add_noise(clean_core, tile_h, tile_w, noise_ratio=float(np.random.uniform(0.10, 0.20)))
        grid = _build_grid(noisy_core, row_bands, col_bands, row_sep, col_sep)

        try:
            inp_sig, out_sig = _validate_instance(grid, clean_tile, alt_dims)
        except Exception:
            continue

        if inp_sig in _USED_INPUT_SIGS or out_sig in _USED_OUTPUT_SIGS:
            continue
        _USED_INPUT_SIGS.add(inp_sig)
        _USED_OUTPUT_SIGS.add(out_sig)
        return grid

    raise AssertionError("failed to generate a valid non-degenerate input")
