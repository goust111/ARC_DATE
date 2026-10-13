from __future__ import annotations

from collections import Counter
import random

import numpy as np


def _dominant_period_colors(g: np.ndarray) -> tuple[int, int]:
    counts = Counter(map(int, g.ravel()))
    assert len(counts) >= 3, "expected two periodic colors and at least one anomaly color"
    ordered = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    bg, bg_count = ordered[0]
    pattern, pattern_count = ordered[1]
    assert bg_count > pattern_count, "background must be the most common color"
    assert pattern_count > sum(v for _c, v in ordered[2:]), "pattern color must dominate anomaly colors"
    return int(bg), int(pattern)


def _infer_periodic_model(g: np.ndarray) -> tuple[np.ndarray, int, int]:
    bg, pattern = _dominant_period_colors(g)
    h, w = g.shape
    clean = (g == bg) | (g == pattern)
    assert np.count_nonzero(~clean) >= 1, "expected anomaly pixels"

    best: tuple[tuple[int, int, int, int], np.ndarray] | None = None
    max_r_period = max(1, min(10, h // 2))
    max_c_period = max(1, min(10, w // 2))

    for rp in range(1, max_r_period + 1):
        for cp in range(1, max_c_period + 1):
            tile = np.full((rp, cp), -1, dtype=int)
            conflicts = 0
            missing = 0
            for tr in range(rp):
                rows = np.arange(tr, h, rp)
                for tc in range(cp):
                    cols = np.arange(tc, w, cp)
                    vals = g[np.ix_(rows, cols)]
                    good_vals = vals[(vals == bg) | (vals == pattern)]
                    if good_vals.size == 0:
                        missing += 1
                        continue
                    n_bg = int(np.count_nonzero(good_vals == bg))
                    n_pat = int(np.count_nonzero(good_vals == pattern))
                    conflicts += min(n_bg, n_pat)
                    tile[tr, tc] = pattern if n_pat > n_bg else bg

            if missing:
                continue
            if not np.any(tile == pattern) or not np.any(tile == bg):
                continue

            score = (conflicts, rp * cp, rp + cp, rp)
            if best is None or score < best[0]:
                best = (score, tile)

    assert best is not None, "could not infer a repeated two-color tile"
    score, tile = best
    assert score[0] == 0, "periodic colors must be conflict-free after removing anomalies"

    rr = np.arange(h)[:, None] % tile.shape[0]
    cc = np.arange(w)[None, :] % tile.shape[1]
    expected = tile[rr, cc]
    assert np.all(expected[clean] == g[clean]), "inferred periodic model disagrees with clean pixels"
    return expected.astype(int), bg, pattern


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the periodic pattern bands selected by each anomaly pixel, then repair the anomalies.

    Input structure:
    - The grid is filled by a repeated two-color pattern: a dominant background color and one pattern color.
    - A few single cells use colors outside those two periodic colors; these are anomaly pixels.
    - The repeated clean cells contain enough copies to determine the row and column periods without conflicts.

    Rule:
    - Infer the original two-color periodic grid from the non-anomaly cells.
    - Process anomaly pixels from top to bottom and left to right; for each one, recolor every original pattern-color cell in its row and column to that anomaly's color.
    - Later anomalies overwrite earlier painted band cells, while original background cells are never painted.
    - After all band painting, restore every anomaly pixel itself to its inferred original periodic color.

    Key insight:
    - Each anomaly is a temporary marker whose color claims the pattern cells in its row and column, then the marker disappears back into the periodic pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must be at most 30 by 30"

    expected, bg, pattern = _infer_periodic_model(g)
    anomaly_mask = (g != bg) & (g != pattern)
    anomaly_coords = [tuple(map(int, rc)) for rc in np.argwhere(anomaly_mask)]
    assert anomaly_coords, "expected at least one anomaly pixel"

    h, w = g.shape
    row_index = np.arange(h)[:, None]
    col_index = np.arange(w)[None, :]
    pattern_mask = expected == pattern

    out = expected.copy()
    for r, c in anomaly_coords:
        color = int(g[r, c])
        band = pattern_mask & ((row_index == r) | (col_index == c))
        assert np.any(band), "each anomaly must select at least one pattern-cell band"
        out[band] = color

    for r, c in anomaly_coords:
        out[r, c] = expected[r, c]
    return out.astype(int)


# --------------- generator ---------------

_GRID_ROWS = range(16, 31)
_GRID_COLS = range(16, 31)
_ROW_PERIODS = (2, 3, 4, 5)
_COL_PERIODS = (3, 4, 5, 6)
_ANOMALY_COUNTS = (2, 3, 3, 4)
_COLORS = tuple(range(10))
_MAX_GENERATION_ATTEMPTS = 500


def _choose_colors(rng: np.random.Generator, n_anomalies: int) -> tuple[int, int, list[int]]:
    colors = list(_COLORS)
    bg = int(rng.choice(colors))
    colors.remove(bg)
    pattern = int(rng.choice(colors))
    colors.remove(pattern)
    rng.shuffle(colors)
    return bg, pattern, [int(c) for c in colors[:n_anomalies]]


def _make_tile(rng: np.random.Generator, rp: int, cp: int, bg: int, pattern: int) -> np.ndarray:
    area = rp * cp
    min_pixels = max(2, area // 5)
    max_pixels = max(min_pixels, min(area - 2, (area * 43) // 100))
    count = int(rng.integers(min_pixels, max_pixels + 1))

    tile = np.full((rp, cp), bg, dtype=int)
    coords = [(r, c) for r in range(rp) for c in range(cp)]
    rng.shuffle(coords)
    for r, c in coords[:count]:
        tile[r, c] = pattern

    # Avoid visually flat rows/columns and accidental one-color tiles.
    if len({tuple(row.tolist()) for row in tile}) < 2:
        return _make_tile(rng, rp, cp, bg, pattern)
    if len({tuple(tile[:, c].tolist()) for c in range(cp)}) < 2:
        return _make_tile(rng, rp, cp, bg, pattern)
    return tile


def _repeat_tile(tile: np.ndarray, rows: int, cols: int) -> np.ndarray:
    rr = np.arange(rows)[:, None] % tile.shape[0]
    cc = np.arange(cols)[None, :] % tile.shape[1]
    return tile[rr, cc].astype(int)


def _row_major_key(pos: tuple[int, int]) -> tuple[int, int]:
    return int(pos[0]), int(pos[1])


def _well_spaced(pos: tuple[int, int], chosen: list[tuple[int, int]], min_distance: int = 4) -> bool:
    r, c = pos
    for rr, cc in chosen:
        if abs(r - rr) + abs(c - cc) < min_distance:
            return False
    return True


def _has_cross_pattern(mask: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> bool:
    ar, ac = a
    br, bc = b
    if ar == br or ac == bc:
        return False
    return bool(mask[ar, bc] or mask[br, ac])


def _choose_forced_anomalies(rng: np.random.Generator, expected: np.ndarray, pattern: int) -> list[tuple[int, int]] | None:
    pattern_mask = expected == pattern
    h, w = expected.shape
    pattern_cells = [(int(r), int(c)) for r, c in np.argwhere(pattern_mask)]
    bg_cells = [(int(r), int(c)) for r, c in np.argwhere(~pattern_mask)]
    if not pattern_cells or not bg_cells:
        return None

    for _ in range(5000):
        a = pattern_cells[int(rng.integers(0, len(pattern_cells)))]
        b = bg_cells[int(rng.integers(0, len(bg_cells)))]
        if _row_major_key(b) < _row_major_key(a):
            a, b = b, a
        if not (0 <= a[0] < h and 0 <= b[1] < w):
            continue
        if a[0] == b[0] or a[1] == b[1]:
            continue
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) < 5:
            continue
        if not _has_cross_pattern(pattern_mask, a, b):
            continue
        return [a, b]
    return None


def _choose_extra_anomalies(
    rng: np.random.Generator,
    expected: np.ndarray,
    count: int,
    chosen: list[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    candidates = [(int(r), int(c)) for r in range(expected.shape[0]) for c in range(expected.shape[1])]
    rng.shuffle(candidates)
    for pos in candidates:
        if len(chosen) >= count:
            return chosen
        if pos in chosen:
            continue
        if not _well_spaced(pos, chosen):
            continue
        chosen.append(pos)
    return chosen if len(chosen) == count else None


def _render_input(expected: np.ndarray, anomalies: list[tuple[int, int, int]]) -> np.ndarray:
    grid = expected.copy()
    for r, c, color in anomalies:
        grid[r, c] = color
    return grid


def _render_expected_output(
    expected: np.ndarray,
    pattern: int,
    anomalies: list[tuple[int, int, int]],
) -> np.ndarray:
    h, w = expected.shape
    row_index = np.arange(h)[:, None]
    col_index = np.arange(w)[None, :]
    pattern_mask = expected == pattern

    out = expected.copy()
    for r, c, color in anomalies:
        band = pattern_mask & ((row_index == r) | (col_index == c))
        out[band] = int(color)
    for r, c, _color in anomalies:
        out[r, c] = expected[r, c]
    return out


def _has_visible_order_dependency(expected: np.ndarray, pattern: int, anomalies: list[tuple[int, int, int]]) -> bool:
    ordered = sorted(anomalies, key=lambda item: (item[0], item[1]))
    forward = _render_expected_output(expected, pattern, ordered)
    backward = _render_expected_output(expected, pattern, list(reversed(ordered)))
    return bool(not np.array_equal(forward, backward))


def _visible_legacy_for_each_anomaly(
    out: np.ndarray,
    anomalies: list[tuple[int, int, int]],
) -> bool:
    anomaly_positions = {(r, c) for r, c, _color in anomalies}
    for _r, _c, color in anomalies:
        found = False
        for rr, cc in np.argwhere(out == int(color)):
            if (int(rr), int(cc)) not in anomaly_positions:
                found = True
                break
        if not found:
            return False
    return True


def _generated_pair_is_valid(
    inp: np.ndarray,
    expected: np.ndarray,
    pattern: int,
    anomalies: list[tuple[int, int, int]],
) -> bool:
    if inp.ndim != 2 or max(inp.shape) > 30:
        return False
    if len({color for _r, _c, color in anomalies}) != len(anomalies):
        return False
    if not any(expected[r, c] == pattern for r, c, _color in anomalies):
        return False
    if not any(expected[r, c] != pattern for r, c, _color in anomalies):
        return False
    if not _has_visible_order_dependency(expected, pattern, anomalies):
        return False

    ordered = sorted(anomalies, key=lambda item: (item[0], item[1]))
    oracle = _render_expected_output(expected, pattern, ordered)
    if not _visible_legacy_for_each_anomaly(oracle, anomalies):
        return False
    try:
        inferred_expected, _bg, inferred_pattern = _infer_periodic_model(inp)
        solved = main(inp)
    except Exception:
        return False
    if inferred_pattern != pattern:
        return False
    if not np.array_equal(inferred_expected, expected):
        return False
    if not np.array_equal(solved, oracle):
        return False
    if np.array_equal(inp, oracle):
        return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    rows = int(rng.choice(list(_GRID_ROWS)))
    cols = int(rng.choice(list(_GRID_COLS)))
    rp = int(rng.choice([p for p in _ROW_PERIODS if p <= rows // 3]))
    cp = int(rng.choice([p for p in _COL_PERIODS if p <= cols // 3]))
    n_anomalies = int(rng.choice(_ANOMALY_COUNTS))

    bg, pattern, anomaly_colors = _choose_colors(rng, n_anomalies)
    tile = _make_tile(rng, rp, cp, bg, pattern)
    expected = _repeat_tile(tile, rows, cols)

    # Keep background visibly dominant after partial tiling.
    if np.count_nonzero(expected == bg) <= np.count_nonzero(expected == pattern):
        return None

    chosen = _choose_forced_anomalies(rng, expected, pattern)
    if chosen is None:
        return None
    chosen = _choose_extra_anomalies(rng, expected, n_anomalies, chosen)
    if chosen is None:
        return None
    chosen = sorted(chosen, key=_row_major_key)

    random.shuffle(anomaly_colors)
    anomalies = [(r, c, anomaly_colors[i]) for i, (r, c) in enumerate(chosen)]
    inp = _render_input(expected, anomalies)
    if not _generated_pair_is_valid(inp, expected, pattern, anomalies):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid.astype(int)
    raise RuntimeError("failed to generate a valid anomaly-band periodic input")
