from __future__ import annotations

import hashlib

import numpy as np


BACKGROUND = 0
MOTIF_COLORS = (1, 2)
NOISE_COLOR = 4

SEED_HEIGHTS = (3, 4)
WIDTH_RANGE = (9, 15)
REPEAT_RANGE = (2, 6)
BAR_WIDTH_RANGE = (2, 4)
BAR_GAP_RANGE = (0, 2)
NOISE_GROUP_RANGE = (2, 4)
MAX_GENERATION_ATTEMPTS = 500

_SEEN_OUTPUT_SIGS: set[str] = set()


def _neighbors4(shape: tuple[int, int], r: int, c: int):
    h, w = shape
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _same_color_support(grid: np.ndarray, r: int, c: int) -> int:
    color = int(grid[r, c])
    return sum(int(grid[nr, nc]) == color for nr, nc in _neighbors4(grid.shape, r, c))


def _bottom_seed_region(grid: np.ndarray) -> tuple[int, np.ndarray]:
    row_has_fg = np.any(grid != BACKGROUND, axis=1)
    assert row_has_fg[-1], "expected a bottom seed region"
    top = grid.shape[0] - 1
    while top >= 0 and row_has_fg[top]:
        top -= 1
    top += 1
    assert np.all(~row_has_fg[:top]), "rows above the seed region must be black"
    assert np.all(row_has_fg[top:]), "seed rows must be contiguous through the bottom"
    return top, grid[top:].copy()


def _filter_seed(seed: np.ndarray) -> np.ndarray:
    keep = np.zeros_like(seed, dtype=bool)
    for r_raw, c_raw in np.argwhere(seed != BACKGROUND):
        r = int(r_raw)
        c = int(c_raw)
        if _same_color_support(seed, r, c) >= 2:
            keep[r, c] = True
    out = np.full_like(seed, BACKGROUND)
    out[keep] = seed[keep]
    return out


def _tile_seed(seed: np.ndarray, total_rows: int, top: int) -> np.ndarray:
    seed_h = int(seed.shape[0])
    out = np.empty((total_rows, seed.shape[1]), dtype=int)
    for r in range(total_rows):
        out[r] = seed[(r - top) % seed_h]
    return out


def _row_period(seed: np.ndarray) -> int:
    h = int(seed.shape[0])
    for period in range(1, h + 1):
        if h % period != 0:
            continue
        if all(np.array_equal(seed[r], seed[r % period]) for r in range(h)):
            return period
    return h


def _all_supported(seed: np.ndarray) -> bool:
    for r_raw, c_raw in np.argwhere(seed != BACKGROUND):
        if _same_color_support(seed, int(r_raw), int(c_raw)) < 2:
            return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """Filter unsupported noise from the bottom seed motif, then repeat the cleaned seed upward to fill the grid.

    Input structure
    - The grid uses black as background.
    - A contiguous block of 3 or 4 non-black rows touches the bottom edge; this block mixes a red/blue seed motif with scattered yellow noise pixels.
    - All rows above that bottom block are black.

    Rule
    - Find the bottom contiguous non-black row block and record its top row and height H.
    - Inside that H-row seed region, erase every non-black pixel whose same-color 4-neighbor count is smaller than 2.
    - Treat the filtered H-row block as the periodic seed and copy its rows upward with the same vertical phase so the original seed rows stay in place.

    Key insight
    The bottom block is not tiled directly: it must first be denoised by a local same-color support test, otherwise the yellow noise would be replicated into every period of the final fill.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    top, seed = _bottom_seed_region(g)
    filtered = _filter_seed(seed)
    return _tile_seed(filtered, g.shape[0], top)


def _nonfull_intervals(seed_h: int) -> list[tuple[int, int]]:
    intervals = []
    for height in range(2, seed_h):
        for r0 in range(0, seed_h - height + 1):
            intervals.append((r0, r0 + height - 1))
    return intervals


def _sample_columns(rng: np.random.Generator, width: int, n_bars: int) -> list[tuple[int, int]] | None:
    for _ in range(100):
        widths = [int(rng.integers(BAR_WIDTH_RANGE[0], BAR_WIDTH_RANGE[1] + 1)) for _ in range(n_bars)]
        gaps = [int(rng.integers(BAR_GAP_RANGE[0], BAR_GAP_RANGE[1] + 1)) for _ in range(n_bars - 1)]
        span = sum(widths) + sum(gaps)
        if span > width - 1:
            continue
        left_margin = int(rng.integers(0, width - span + 1))
        cols = []
        c = left_margin
        for i, bar_w in enumerate(widths):
            cols.append((c, c + bar_w - 1))
            c += bar_w
            if i + 1 < n_bars:
                c += gaps[i]
        return cols
    return None


def _build_clean_seed(rng: np.random.Generator, seed_h: int, width: int) -> np.ndarray | None:
    intervals = _nonfull_intervals(seed_h)
    for _ in range(200):
        max_bars = 4 if width < 12 else 5
        n_bars = int(rng.integers(3, max_bars + 1))
        cols = _sample_columns(rng, width, n_bars)
        if cols is None:
            continue

        base_idx = int(rng.integers(0, n_bars))
        base_color = int(rng.choice(MOTIF_COLORS))
        other_color = MOTIF_COLORS[1] if base_color == MOTIF_COLORS[0] else MOTIF_COLORS[0]

        seed = np.full((seed_h, width), BACKGROUND, dtype=int)
        other_used = False
        for i, (c0, c1) in enumerate(cols):
            if i == base_idx:
                r0, r1 = 0, seed_h - 1
                color = base_color
            else:
                r0, r1 = intervals[int(rng.integers(0, len(intervals)))]
                remaining_partial = sum(j != base_idx for j in range(i + 1, n_bars))
                if not other_used and remaining_partial == 1:
                    color = other_color
                else:
                    color = int(rng.choice(MOTIF_COLORS))
            if color == other_color:
                other_used = True
            seed[r0 : r1 + 1, c0 : c1 + 1] = color

        density = float(np.mean(seed != BACKGROUND))
        if not (0.22 <= density <= 0.78):
            continue
        if not (np.any(seed == MOTIF_COLORS[0]) and np.any(seed == MOTIF_COLORS[1])):
            continue
        if not np.all(np.any(seed != BACKGROUND, axis=1)):
            continue
        if not np.any(seed == BACKGROUND):
            continue
        if _row_period(seed) != seed_h:
            continue
        if not _all_supported(seed):
            continue
        return seed
    return None


def _place_noise_group(
    seed: np.ndarray,
    coords: list[tuple[int, int]],
) -> bool:
    for r, c in coords:
        if int(seed[r, c]) != BACKGROUND:
            return False
        for nr, nc in _neighbors4(seed.shape, r, c):
            if int(seed[nr, nc]) == NOISE_COLOR and (nr, nc) not in coords:
                return False
    for r, c in coords:
        seed[r, c] = NOISE_COLOR
    return True


def _add_noise(rng: np.random.Generator, clean_seed: np.ndarray) -> np.ndarray | None:
    shapes = (
        ((0, 0),),
        ((0, 0), (0, 1)),
        ((0, 0), (1, 0)),
    )
    seed_h, width = clean_seed.shape
    for _ in range(200):
        seed = clean_seed.copy()
        group_count = int(rng.integers(NOISE_GROUP_RANGE[0], NOISE_GROUP_RANGE[1] + 1))
        need_domino = bool(rng.integers(0, 100) < 65)
        group_sizes = [1] * group_count
        if need_domino:
            group_sizes[int(rng.integers(0, group_count))] = 2

        ok = True
        for size in group_sizes:
            placed = False
            for _ in range(80):
                if size == 1:
                    shape = shapes[0]
                else:
                    shape = shapes[int(rng.integers(1, len(shapes)))]
                shape_h = max(dr for dr, _dc in shape) + 1
                shape_w = max(dc for _dr, dc in shape) + 1
                r0 = int(rng.integers(0, seed_h - shape_h + 1))
                c0 = int(rng.integers(0, width - shape_w + 1))
                coords = [(r0 + dr, c0 + dc) for dr, dc in shape]
                if _place_noise_group(seed, coords):
                    placed = True
                    break
            if not placed:
                ok = False
                break

        if not ok:
            continue
        if not np.any(seed == NOISE_COLOR):
            continue
        if any(_same_color_support(seed, int(r), int(c)) >= 2 for r, c in np.argwhere(seed == NOISE_COLOR)):
            continue
        return seed
    return None


def _signature(grid: np.ndarray) -> str:
    return hashlib.md5(grid.tobytes() + str(grid.shape).encode()).hexdigest()


def _validate_candidate(inp: np.ndarray) -> bool:
    h, w = inp.shape
    if max(h, w) > 30:
        return False
    if not set(map(int, np.unique(inp))) <= {BACKGROUND, *MOTIF_COLORS, NOISE_COLOR}:
        return False

    try:
        top, raw_seed = _bottom_seed_region(inp)
    except AssertionError:
        return False

    seed_h = int(raw_seed.shape[0])
    if seed_h not in SEED_HEIGHTS:
        return False
    if top <= 0 or h % seed_h != 0:
        return False
    if not np.all(inp[:top] == BACKGROUND):
        return False
    if not (np.any(raw_seed == MOTIF_COLORS[0]) and np.any(raw_seed == MOTIF_COLORS[1])):
        return False
    if not np.any(raw_seed == NOISE_COLOR):
        return False
    if not np.all(np.any(raw_seed != BACKGROUND, axis=1)):
        return False

    clean_mask = (raw_seed == MOTIF_COLORS[0]) | (raw_seed == MOTIF_COLORS[1])
    for r_raw, c_raw in np.argwhere(clean_mask):
        if _same_color_support(raw_seed, int(r_raw), int(c_raw)) < 2:
            return False
    for r_raw, c_raw in np.argwhere(raw_seed == NOISE_COLOR):
        if _same_color_support(raw_seed, int(r_raw), int(c_raw)) >= 2:
            return False

    filtered = _filter_seed(raw_seed)
    if np.any(filtered == NOISE_COLOR):
        return False
    if not (np.any(filtered == MOTIF_COLORS[0]) and np.any(filtered == MOTIF_COLORS[1])):
        return False
    if not _all_supported(filtered):
        return False
    if not np.all(np.any(filtered != BACKGROUND, axis=1)):
        return False
    if _row_period(filtered) != seed_h:
        return False

    expected = _tile_seed(filtered, h, top)
    raw_tiled = _tile_seed(raw_seed, h, top)
    if np.array_equal(expected, raw_tiled):
        return False
    if np.array_equal(inp, expected):
        return False
    if not np.array_equal(main(inp), expected):
        return False

    if _signature(expected) in _SEEN_OUTPUT_SIGS:
        return False
    return True


def _generate_candidate(rng: np.random.Generator) -> np.ndarray | None:
    seed_h = int(rng.choice(SEED_HEIGHTS))
    max_repeat = min(REPEAT_RANGE[1], 30 // seed_h)
    repeats = int(rng.integers(REPEAT_RANGE[0], max_repeat + 1))
    height = seed_h * repeats
    width = int(rng.integers(WIDTH_RANGE[0], WIDTH_RANGE[1] + 1))

    clean_seed = _build_clean_seed(rng, seed_h, width)
    if clean_seed is None:
        return None
    noisy_seed = _add_noise(rng, clean_seed)
    if noisy_seed is None:
        return None

    inp = np.full((height, width), BACKGROUND, dtype=int)
    inp[-seed_h:] = noisy_seed
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        candidate = _generate_candidate(rng)
        if candidate is None or not _validate_candidate(candidate):
            continue
        out_sig = _signature(main(candidate))
        _SEEN_OUTPUT_SIGS.add(out_sig)
        return candidate
    raise RuntimeError("failed to generate a valid noise-filtered periodic tiling input")
