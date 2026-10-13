import random

import numpy as np


BG = 0
SEPARATOR_COLOR = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
PATTERN_MODES = ("rectangles", "bands", "coarse", "quadrants")
GEN_RETRIES = 400


def _spans(indices: np.ndarray) -> list[tuple[int, int]]:
    if len(indices) == 0:
        return []
    spans: list[tuple[int, int]] = []
    start = int(indices[0])
    prev = start
    for value in indices[1:]:
        value = int(value)
        if value != prev + 1:
            spans.append((start, prev))
            start = value
        prev = value
    spans.append((start, prev))
    return spans


def _find_matches(lock: np.ndarray, patch: np.ndarray, stop_after: int | None = None) -> list[tuple[int, int]]:
    lh, lw = lock.shape
    ph, pw = patch.shape
    matches: list[tuple[int, int]] = []
    for r in range(lh - ph + 1):
        for c in range(lw - pw + 1):
            if np.array_equal(lock[r : r + ph, c : c + pw], patch):
                matches.append((r, c))
                if stop_after is not None and len(matches) >= stop_after:
                    return matches
    return matches


def main(grid: np.ndarray) -> np.ndarray:
    """Replace the unique lock subregion matching the lower-left key block."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    separator_rows = [
        r for r in range(g.shape[0]) if int(g[r, 0]) != BG and np.all(g[r] == g[r, 0])
    ]
    assert len(separator_rows) == 1, "expected a single full-width separator row"
    sep = separator_rows[0]

    upper_mask = g[:sep] != BG
    assert np.any(upper_mask), "upper lock region missing"
    upper_rows = np.where(np.any(upper_mask, axis=1))[0]
    upper_cols = np.where(np.any(upper_mask, axis=0))[0]
    lock_r0, lock_r1 = int(upper_rows[0]), int(upper_rows[-1])
    lock_c0, lock_c1 = int(upper_cols[0]), int(upper_cols[-1])
    lock = g[lock_r0 : lock_r1 + 1, lock_c0 : lock_c1 + 1]

    lower_mask = g[sep + 1 :] != BG
    assert np.any(lower_mask), "lower key/replacement region missing"
    lower_rows = np.where(np.any(lower_mask, axis=1))[0]
    band_r0 = sep + 1 + int(lower_rows[0])
    band_r1 = sep + 1 + int(lower_rows[-1])
    lower_band = g[band_r0 : band_r1 + 1]

    active_cols = np.where(np.any(lower_band != BG, axis=0))[0]
    col_spans = _spans(active_cols)
    assert len(col_spans) == 2, "expected exactly two lower blocks"
    (key_c0, key_c1), (rep_c0, rep_c1) = col_spans

    key = lower_band[:, key_c0 : key_c1 + 1]
    replacement = lower_band[:, rep_c0 : rep_c1 + 1]
    assert key.shape == replacement.shape, "key and replacement must be the same size"

    matches = _find_matches(lock, key)
    assert len(matches) == 1, "key must match exactly one lock subregion"
    match_r, match_c = matches[0]

    out = g.copy()
    kh, kw = key.shape
    out[lock_r0 + match_r : lock_r0 + match_r + kh, lock_c0 + match_c : lock_c0 + match_c + kw] = replacement
    out[sep:] = BG
    return out


def _pattern_rectangles(h: int, w: int, palette: list[int], rng: random.Random) -> np.ndarray:
    arr = np.full((h, w), rng.choice(palette), dtype=int)
    for _ in range(rng.randint(4, 8)):
        color = rng.choice(palette)
        r0 = rng.randint(0, h - 1)
        r1 = rng.randint(r0, h - 1)
        c0 = rng.randint(0, w - 1)
        c1 = rng.randint(c0, w - 1)
        arr[r0 : r1 + 1, c0 : c1 + 1] = color
    return arr


def _pattern_bands(h: int, w: int, palette: list[int], rng: random.Random) -> np.ndarray:
    arr = np.empty((h, w), dtype=int)
    r = 0
    while r < h:
        band_h = rng.randint(1, min(3, h - r))
        arr[r : r + band_h, :] = rng.choice(palette)
        r += band_h
    for _ in range(rng.randint(2, 4)):
        c0 = rng.randint(0, w - 1)
        c1 = rng.randint(c0, w - 1)
        arr[:, c0 : c1 + 1] = rng.choice(palette)
    return arr


def _pattern_coarse(h: int, w: int, palette: list[int], rng: random.Random) -> np.ndarray:
    coarse_h = rng.randint(2, min(4, h))
    coarse_w = rng.randint(2, min(4, w))
    row_edges = np.linspace(0, h, coarse_h + 1, dtype=int)
    col_edges = np.linspace(0, w, coarse_w + 1, dtype=int)
    arr = np.empty((h, w), dtype=int)
    for i in range(coarse_h):
        for j in range(coarse_w):
            arr[row_edges[i] : row_edges[i + 1], col_edges[j] : col_edges[j + 1]] = rng.choice(palette)
    for _ in range(rng.randint(1, 3)):
        if rng.random() < 0.5:
            rr = rng.randint(0, h - 1)
            arr[rr : rr + 1, :] = rng.choice(palette)
        else:
            cc = rng.randint(0, w - 1)
            arr[:, cc : cc + 1] = rng.choice(palette)
    return arr


def _pattern_quadrants(h: int, w: int, palette: list[int], rng: random.Random) -> np.ndarray:
    split_r = rng.randint(1, h - 1)
    split_c = rng.randint(1, w - 1)
    arr = np.empty((h, w), dtype=int)
    colors = [rng.choice(palette) for _ in range(4)]
    arr[:split_r, :split_c] = colors[0]
    arr[:split_r, split_c:] = colors[1]
    arr[split_r:, :split_c] = colors[2]
    arr[split_r:, split_c:] = colors[3]
    if rng.random() < 0.5:
        rr = rng.randint(0, h - 1)
        arr[rr : rr + 1, :] = rng.choice(palette)
    else:
        cc = rng.randint(0, w - 1)
        arr[:, cc : cc + 1] = rng.choice(palette)
    return arr


def _make_pattern(h: int, w: int, palette: list[int], rng: random.Random) -> np.ndarray:
    builders = {
        "rectangles": _pattern_rectangles,
        "bands": _pattern_bands,
        "coarse": _pattern_coarse,
        "quadrants": _pattern_quadrants,
    }
    for _ in range(30):
        mode = rng.choice(PATTERN_MODES)
        arr = builders[mode](h, w, palette, rng)
        if len(np.unique(arr)) >= min(3, len(palette)):
            return arr
    return arr


def _pick_unique_window(lock: np.ndarray, rng: random.Random) -> tuple[int, int, np.ndarray] | None:
    h, w = lock.shape
    for _ in range(120):
        kh = rng.randint(2, min(6, h))
        kw = rng.randint(2, min(6, w))
        if kh == h and kw == w:
            continue
        if kh * kw < 6:
            continue
        r0 = rng.randint(0, h - kh)
        c0 = rng.randint(0, w - kw)
        patch = lock[r0 : r0 + kh, c0 : c0 + kw]
        if len(np.unique(patch)) < 2:
            continue
        if len(_find_matches(lock, patch, stop_after=2)) == 1:
            return r0, c0, patch.copy()
    return None


def _make_replacement(shape: tuple[int, int], palette: list[int], key: np.ndarray, rng: random.Random) -> np.ndarray | None:
    kh, kw = shape
    for _ in range(60):
        replacement = _make_pattern(kh, kw, palette, rng)
        diff = int(np.count_nonzero(replacement != key))
        if diff >= max(1, key.size // 3):
            return replacement
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    palette_size = rng.randint(4, 6)
    palette = rng.sample(COLOR_POOL, palette_size)

    lock_h = rng.randint(6, 12)
    lock_w = rng.randint(7, 14)
    lock = _make_pattern(lock_h, lock_w, palette, rng)

    picked = _pick_unique_window(lock, rng)
    if picked is None:
        return None
    _, _, key = picked
    key_h, key_w = key.shape

    replacement_palette = palette[:]
    extra_colors = [color for color in COLOR_POOL if color not in replacement_palette]
    if extra_colors and rng.random() < 0.5:
        replacement_palette.append(rng.choice(extra_colors))
    replacement = _make_replacement((key_h, key_w), replacement_palette, key, rng)
    if replacement is None:
        return None

    lock_margin_top = rng.randint(1, 3)
    lock_margin_bottom = rng.randint(1, 3)
    lower_margin_top = rng.randint(1, 3)
    lower_margin_bottom = rng.randint(1, 3)

    bottom_w = 2 * key_w + 1
    grid_w = max(lock_w, bottom_w) + rng.randint(4, 8)
    grid_h = (
        lock_margin_top
        + lock_h
        + lock_margin_bottom
        + 1
        + lower_margin_top
        + key_h
        + lower_margin_bottom
    )

    if grid_w <= lock_w + 1 or grid_w <= bottom_w + 1:
        return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    sep_row = lock_margin_top + lock_h + lock_margin_bottom
    grid[sep_row, :] = SEPARATOR_COLOR

    lock_c0 = rng.randint(1, grid_w - lock_w - 1)
    lock_r0 = lock_margin_top
    grid[lock_r0 : lock_r0 + lock_h, lock_c0 : lock_c0 + lock_w] = lock

    lower_r0 = sep_row + 1 + lower_margin_top
    lower_c0 = rng.randint(1, grid_w - bottom_w - 1)
    grid[lower_r0 : lower_r0 + key_h, lower_c0 : lower_c0 + key_w] = key
    grid[lower_r0 : lower_r0 + key_h, lower_c0 + key_w + 1 : lower_c0 + 2 * key_w + 1] = replacement

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
