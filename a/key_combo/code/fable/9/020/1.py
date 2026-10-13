from __future__ import annotations

import numpy as np


def _is_contiguous(indices: np.ndarray) -> bool:
    indices = np.asarray(indices, dtype=int)
    return bool(indices.size) and np.array_equal(
        indices, np.arange(int(indices[0]), int(indices[-1]) + 1, dtype=int)
    )


def _d4_orbit(r: int, c: int, n: int) -> tuple[tuple[int, int], ...]:
    last = n - 1
    return tuple(
        sorted(
            {
                (r, c),
                (last - r, c),
                (r, last - c),
                (last - r, last - c),
                (c, r),
                (last - c, r),
                (c, last - r),
                (last - c, last - r),
            }
        )
    )


def _all_d4_orbits(n: int) -> list[tuple[tuple[int, int], ...]]:
    seen = np.zeros((n, n), dtype=bool)
    out: list[tuple[tuple[int, int], ...]] = []
    for r in range(n):
        for c in range(n):
            if seen[r, c]:
                continue
            orbit = _d4_orbit(r, c, n)
            for rr, cc in orbit:
                seen[rr, cc] = True
            out.append(orbit)
    return out


def _line_is_band_candidate(line: np.ndarray) -> bool:
    values, counts = np.unique(line, return_counts=True)
    return bool(values.size <= 2 and int(counts.max()) > line.size // 2)


def _find_band_indices(grid: np.ndarray, axis: int) -> np.ndarray:
    n = grid.shape[axis]
    picks = [
        idx
        for idx in range(n)
        if _line_is_band_candidate(grid[idx, :] if axis == 0 else grid[:, idx])
    ]
    band = np.array(picks, dtype=int)
    assert _is_contiguous(band), "expected one contiguous bar band"
    return band


def _bar_layout(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    g = np.asarray(grid, dtype=int)
    n = g.shape[0]
    rows = _find_band_indices(g, axis=0)
    cols = _find_band_indices(g, axis=1)
    all_rows = np.arange(n, dtype=int)
    all_cols = np.arange(n, dtype=int)
    outer_rows = all_rows[~np.isin(all_rows, rows)]
    outer_cols = all_cols[~np.isin(all_cols, cols)]
    assert outer_rows.size and outer_cols.size, "bars must not cover the whole grid"

    h_strip = g[np.ix_(rows, outer_cols)]
    v_strip = g[np.ix_(outer_rows, cols)]
    assert np.unique(h_strip).size == 1, "horizontal strip exterior must be uniform"
    assert np.unique(v_strip).size == 1, "vertical strip exterior must be uniform"
    h_color = int(h_strip[0, 0])
    v_color = int(v_strip[0, 0])
    assert h_color != v_color, "bars must have different colors"

    cross_colors = np.unique(g[np.ix_(rows, cols)])
    assert cross_colors.size == 1, "crossing must show exactly one top color"
    top_color = int(cross_colors[0])
    assert top_color in {h_color, v_color}, "crossing color must belong to one bar"
    return rows, cols, outer_rows, outer_cols, h_color, v_color, top_color


def _recover_pattern_color(grid: np.ndarray, r: int, c: int, bar_mask: np.ndarray) -> int:
    n = grid.shape[0]
    visible = {
        int(grid[rr, cc])
        for rr, cc in _d4_orbit(int(r), int(c), n)
        if not bool(bar_mask[rr, cc])
    }
    assert visible, "every damaged bar pixel needs at least one visible symmetric twin"
    assert len(visible) == 1, "visible D4-symmetric twins must agree on one color"
    return next(iter(visible))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reveal the lower strip in a crossing by keeping the top strip and restoring the hidden pattern via D4 symmetry.

    Input structure:
    - The input is a square grid containing a multicolor pattern invariant under the
      rotations and reflections of the square.
    - One solid horizontal strip spans the full width and one solid vertical strip spans
      the full height; their colors are distinct and do not belong to the pattern.
    - The rectangular crossing region is uniformly recolored to the color of the strip
      that lies on top.

    Rule:
    - Detect the horizontal strip and the vertical strip, and read the crossing color to
      decide which strip is above the other.
    - Keep the top strip unchanged, including the crossing rectangle.
    - For every visible cell of the lower strip outside the crossing, replace it with the
      unique color shared by all uncovered D4-symmetric copies of that position.
    - Leave the original pattern elsewhere unchanged.

    Key insight:
    - The crossing is not the area to repaint by itself; it is the only cue telling you
      which strip survives and which strip should be erased by copying from visible
      dihedral-symmetric twins of the hidden pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"
    h, w = g.shape
    assert h == w, "grid must be square"

    rows, cols, outer_rows, outer_cols, h_color, v_color, top_color = _bar_layout(g)
    top_is_h = top_color == h_color

    bar_mask = np.zeros_like(g, dtype=bool)
    bar_mask[rows, :] = True
    bar_mask[:, cols] = True

    out = g.copy()
    if top_is_h:
        for r in outer_rows:
            for c in cols:
                out[r, c] = _recover_pattern_color(g, int(r), int(c), bar_mask)
    else:
        for r in rows:
            for c in outer_cols:
                out[r, c] = _recover_pattern_color(g, int(r), int(c), bar_mask)
    return out


GEN_ODD_SIZES = np.arange(13, 24, 2, dtype=int)
GEN_BAR_THICKNESSES = np.array([1, 2, 3], dtype=int)
GEN_RETRIES = 600
GEN_COLORS = np.arange(1, 10, dtype=int)


def _random_inner_interval(n: int, thickness: int) -> np.ndarray:
    start = int(np.random.randint(1, n - thickness))
    return np.arange(start, start + thickness, dtype=int)


def _interval_is_mirror_symmetric(n: int, interval: np.ndarray) -> bool:
    mirrored = np.array(sorted(n - 1 - int(x) for x in interval), dtype=int)
    return np.array_equal(np.asarray(interval, dtype=int), mirrored)


def _random_nonsymmetric_inner_interval(n: int, thickness: int) -> np.ndarray | None:
    choices: list[np.ndarray] = []
    for start in range(1, n - thickness):
        interval = np.arange(start, start + thickness, dtype=int)
        if not _interval_is_mirror_symmetric(n, interval):
            choices.append(interval)
    if not choices:
        return None
    return choices[int(np.random.randint(len(choices)))]


def _build_symmetric_pattern(n: int, palette: np.ndarray) -> np.ndarray:
    orbits = _all_d4_orbits(n)
    grid = np.empty((n, n), dtype=int)
    orbit_order = np.random.permutation(len(orbits))
    for i, orbit_idx in enumerate(orbit_order):
        color = int(palette[i]) if i < len(palette) else int(palette[np.random.randint(len(palette))])
        for r, c in orbits[int(orbit_idx)]:
            grid[r, c] = color
    return grid


def _paint_input(
    pattern: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    h_color: int,
    v_color: int,
    top_is_h: bool,
) -> np.ndarray:
    grid = pattern.copy()
    grid[rows, :] = h_color
    grid[:, cols] = v_color
    grid[np.ix_(rows, cols)] = h_color if top_is_h else v_color
    return grid


def _expected_output(
    pattern: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    h_color: int,
    v_color: int,
    top_is_h: bool,
) -> np.ndarray:
    out = pattern.copy()
    if top_is_h:
        out[rows, :] = h_color
        out[np.ix_(rows, cols)] = h_color
    else:
        out[:, cols] = v_color
        out[np.ix_(rows, cols)] = v_color
    return out


def _validate_discoverability(grid: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> bool:
    n = grid.shape[0]
    found_rows = _find_band_indices(grid, axis=0)
    found_cols = _find_band_indices(grid, axis=1)
    if not np.array_equal(found_rows, rows) or not np.array_equal(found_cols, cols):
        return False

    row_set = set(map(int, rows))
    col_set = set(map(int, cols))
    for r in range(n):
        is_candidate = _line_is_band_candidate(grid[r, :])
        if is_candidate != (r in row_set):
            return False
    for c in range(n):
        is_candidate = _line_is_band_candidate(grid[:, c])
        if is_candidate != (c in col_set):
            return False
    return True


def _validate_recoverability(
    inp: np.ndarray,
    pattern: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    top_is_h: bool,
) -> bool:
    n = inp.shape[0]
    bar_mask = np.zeros_like(inp, dtype=bool)
    bar_mask[rows, :] = True
    bar_mask[:, cols] = True

    restored_colors: list[int] = []
    if top_is_h:
        targets = [(r, c) for r in range(n) if r not in set(map(int, rows)) for c in cols]
    else:
        targets = [(r, c) for r in rows for c in range(n) if c not in set(map(int, cols))]

    for r, c in targets:
        visible_coords = [(rr, cc) for rr, cc in _d4_orbit(int(r), int(c), n) if not bool(bar_mask[rr, cc])]
        if not visible_coords:
            return False
        colors = {int(inp[rr, cc]) for rr, cc in visible_coords}
        if len(colors) != 1:
            return False
        color = next(iter(colors))
        if color != int(pattern[r, c]):
            return False
        restored_colors.append(color)

    return len(set(restored_colors)) >= 3


def _attempt() -> np.ndarray | None:
    n = int(np.random.choice(GEN_ODD_SIZES))
    top_is_h = bool(np.random.randint(2))

    if top_is_h:
        rows = _random_inner_interval(n, int(np.random.choice(GEN_BAR_THICKNESSES)))
        cols = _random_nonsymmetric_inner_interval(
            n, int(np.random.choice(GEN_BAR_THICKNESSES))
        )
    else:
        rows = _random_nonsymmetric_inner_interval(
            n, int(np.random.choice(GEN_BAR_THICKNESSES))
        )
        cols = _random_inner_interval(n, int(np.random.choice(GEN_BAR_THICKNESSES)))
    if rows is None or cols is None:
        return None

    h_color, v_color = map(int, np.random.choice(GEN_COLORS, size=2, replace=False))
    remaining = np.array([c for c in GEN_COLORS if c not in {h_color, v_color}], dtype=int)
    palette_size = int(np.random.randint(5, min(8, len(remaining)) + 1))
    palette = np.random.choice(remaining, size=palette_size, replace=False).astype(int)
    pattern = _build_symmetric_pattern(n, palette)

    inp = _paint_input(pattern, rows, cols, h_color, v_color, top_is_h)
    expected = _expected_output(pattern, rows, cols, h_color, v_color, top_is_h)

    if np.array_equal(inp, expected):
        return None
    if not _validate_discoverability(inp, rows, cols):
        return None
    if not _validate_recoverability(inp, pattern, rows, cols, top_is_h):
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    return inp if np.array_equal(actual, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for crossing-occlusion symmetry task")
