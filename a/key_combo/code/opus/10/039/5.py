from __future__ import annotations

import random

import numpy as np


BG = 0
CONTROL = 7
PALETTE = (1, 2, 3, 4, 5, 6)
MIN_N = 2
MAX_N = 5
MAX_ATTEMPTS = 400


def _rise_fall_sequence(n: int) -> list[int]:
    return list(range(1, n + 1)) + list(range(n - 1, 0, -1))


def _control_length(grid: np.ndarray) -> int:
    top = np.asarray(grid[0], dtype=int)
    n = 0
    while n < top.size and int(top[n]) == CONTROL:
        n += 1
    assert MIN_N <= n <= MAX_N, "control bar length must be in [2, 5]"
    assert np.all(top[n:] == BG), "only the left-aligned control bar may appear in row 0"
    return int(n)


def _parse_bands(grid: np.ndarray, n: int) -> tuple[list[int], list[int]]:
    main = np.asarray(grid[1:], dtype=int)
    assert main.size > 0, "expected a non-empty main region below the control row"

    band_cols: list[int] = []
    band_colors: list[int] = []
    for c in range(main.shape[1]):
        nz = sorted(int(v) for v in np.unique(main[:, c]) if int(v) != BG)
        if not nz:
            continue
        assert len(nz) == 1, "each vertical band column may contain only one foreground color"
        band_cols.append(int(c))
        band_colors.append(int(nz[0]))

    expected_band_count = 2 * n - 1
    assert len(band_cols) == expected_band_count, "band count must be determined by the control bar"
    assert all(
        band_cols[i + 1] - band_cols[i] == 2 for i in range(len(band_cols) - 1)
    ), "bands must be width 1 with 1-column black separators"

    for idx, (c, color) in enumerate(zip(band_cols, band_colors)):
        expected_color = PALETTE[idx % len(PALETTE)]
        assert color == expected_color, "band colors must follow the visible left-to-right palette cycle"
        column = main[:, c]
        assert np.all(np.isin(column, [BG, expected_color])), "a band may contain only its own color or black gaps"
        gap_rows = np.where(column == BG)[0]
        assert len(gap_rows) >= n, "every band must contain enough gaps for the peak fill height"
        deepest_gap = int(gap_rows.max())
        assert np.all(
            column[deepest_gap + 1 :] == expected_color
        ), "each band must have a continuous solid-color lower region"
    return band_cols, band_colors


def _semantic_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    n = _control_length(g)
    band_cols, band_colors = _parse_bands(g, n)
    seq = _rise_fall_sequence(n)

    out = g.copy()
    for idx, c in enumerate(band_cols):
        gap_rows = [int(r) for r in np.where(g[1:, c] == BG)[0] + 1]
        gap_rows.sort(reverse=True)
        fill_rows = gap_rows[: seq[idx]]
        out[fill_rows, c] = band_colors[(idx + 1) % len(band_cols)]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each vertical band's lowest black gaps using the cyclic right-neighbor color, with heights driven by the control-bar rise-fall sequence.

    Input structure:
    - Row 0 contains a left-aligned orange control bar on a black background; its length is n.
    - Rows 1..end contain exactly (2n-1) width-1 vertical color bands, separated by width-1 black columns.
    - Each band uses the visible palette cycle blue, red, green, yellow, grey, magenta, repeating if needed, and each band column contains only its own color or black gaps, with a continuous solid-color lower region.

    Rule:
    - Read the control-bar length n and build the rise-fall sequence 1,2,...,n,...,2,1.
    - Order the bands from left to right; for band k, look at its black gap cells and count upward from the lowest gap.
    - Fill exactly the lowest S[k] gap cells of that band with the color of the cyclic right neighbor band, leaving all original colored pixels, separator columns, and the control row unchanged.

    Key insight:
    - The control bar simultaneously determines both how many bands exist and the per-band fill heights, while the borrowed fill color comes from the next band in the visible cyclic order rather than from the band being filled.
    """
    return _semantic_output(grid)


def _choose_n() -> int:
    return random.choice([2, 3, 4, 4, 5, 5])


def _choose_gap_rows(zone_height: int, gap_count: int) -> list[int]:
    pattern = random.choice(["random", "alternating", "clustered"])
    rows = list(range(zone_height))

    if pattern == "alternating":
        offset = random.randint(0, 1)
        preferred = rows[offset::2]
        if len(preferred) >= gap_count:
            return sorted(random.sample(preferred, gap_count))
        remaining = [r for r in rows if r not in preferred]
        chosen = preferred + random.sample(remaining, gap_count - len(preferred))
        return sorted(chosen)

    if pattern == "clustered":
        start = random.randint(0, zone_height - gap_count)
        chosen = list(range(start, start + gap_count))
        movable = [r for r in rows if r not in chosen]
        swaps = min(len(movable), random.randint(0, max(1, gap_count // 2)))
        for _ in range(swaps):
            drop = random.choice(chosen)
            add = random.choice(movable)
            chosen.remove(drop)
            movable.remove(add)
            movable.append(drop)
            chosen.append(add)
        return sorted(chosen)

    return sorted(random.sample(rows, gap_count))


def _build_band_column(main_height: int, n: int, idx: int, exact_min_band: int, dense_band: int) -> np.ndarray:
    color = PALETTE[idx % len(PALETTE)]
    column = np.full(main_height, color, dtype=int)

    solid_height = random.randint(2, main_height - (n + 1))
    gap_zone_height = main_height - solid_height

    if idx == exact_min_band:
        gap_count = n
    elif idx == dense_band:
        gap_count = gap_zone_height - 1
    else:
        gap_count = random.randint(n, gap_zone_height - 1)

    gap_rows = _choose_gap_rows(gap_zone_height, gap_count)
    column[gap_rows] = BG
    return column


def _oracle_expected(grid: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    n = _control_length(g)
    band_cols, band_colors = _parse_bands(g, n)
    seq = _rise_fall_sequence(n)

    out = g.copy()
    changed: set[tuple[int, int]] = set()
    for idx, c in enumerate(band_cols):
        gap_rows = [int(r) for r in np.where(g[1:, c] == BG)[0] + 1]
        gap_rows.sort(reverse=True)
        target_rows = gap_rows[: seq[idx]]
        fill_color = band_colors[(idx + 1) % len(band_cols)]
        for r in target_rows:
            out[r, c] = fill_color
            changed.add((int(r), int(c)))

    info = {
        "n": n,
        "band_cols": band_cols,
        "band_colors": band_colors,
        "sequence": seq,
        "changed": changed,
    }
    return out, info


def _validate_semantics(inp: np.ndarray, out: np.ndarray) -> bool:
    expected, info = _oracle_expected(inp)
    if not np.array_equal(out, expected):
        return False

    band_cols = info["band_cols"]
    seq = info["sequence"]
    changed = info["changed"]
    n = int(info["n"])

    if not changed:
        return False

    if not np.array_equal(inp[0], out[0]):
        return False

    for c in range(inp.shape[1]):
        if c not in band_cols and not np.array_equal(inp[1:, c], out[1:, c]):
            return False

    saw_wrap = False
    saw_unused_gap = False
    for idx, c in enumerate(band_cols):
        in_col = inp[1:, c]
        out_col = out[1:, c]
        gap_rows = [int(r) for r in np.where(in_col == BG)[0] + 1]
        gap_rows.sort(reverse=True)
        target_rows = set(gap_rows[: seq[idx]])
        untouched_gap_rows = set(gap_rows[seq[idx] :])

        if len(gap_rows) > seq[idx]:
            saw_unused_gap = True
        if idx == len(band_cols) - 1 and target_rows:
            saw_wrap = True

        for r in range(1, inp.shape[0]):
            if r in target_rows:
                if (r, c) not in changed or out[r, c] == inp[r, c]:
                    return False
            elif r in untouched_gap_rows:
                if out[r, c] != BG:
                    return False
            elif out[r, c] != inp[r, c]:
                return False

    return saw_wrap and saw_unused_gap and len(band_cols) == 2 * n - 1


def _generate_once() -> np.ndarray | None:
    n = _choose_n()
    band_count = 2 * n - 1
    start_col = random.randint(0, 1)
    right_pad = random.randint(0, 3)
    width = start_col + 2 * band_count - 1 + right_pad

    main_height = random.randint(n + 5, min(18, n + 10))
    height = main_height + 1
    if max(height, width) > 30:
        return None

    grid = np.zeros((height, width), dtype=int)
    grid[0, :n] = CONTROL

    exact_min_band = random.randrange(band_count)
    dense_choices = [i for i in range(band_count) if i != exact_min_band]
    dense_band = random.choice(dense_choices)

    for idx in range(band_count):
        c = start_col + 2 * idx
        grid[1:, c] = _build_band_column(main_height, n, idx, exact_min_band, dense_band)

    out = main(grid)
    return grid if _validate_semantics(grid, out) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_once()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
