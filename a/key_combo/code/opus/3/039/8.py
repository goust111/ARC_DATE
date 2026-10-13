from __future__ import annotations

from collections import Counter
import random

import numpy as np


PALETTE = [1, 2, 3, 4]
GEN_RETRIES = 400


def _colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(grid)))
    if exclude is not None:
        colors -= {int(v) for v in exclude}
    return list(colors)


def _find_reference_row(grid: np.ndarray) -> tuple[int, dict[int, int], set[int]]:
    all_colors = set(_colors_present(grid, exclude={0}))
    assert 3 <= len(all_colors) <= 4, "expected 3 or 4 band colors"

    candidates: list[tuple[int, dict[int, int]]] = []
    for r in range(grid.shape[0]):
        row = grid[r]
        nz = row[row != 0]
        if nz.size != len(all_colors):
            continue
        vals, counts = np.unique(nz, return_counts=True)
        if set(map(int, vals)) != all_colors:
            continue
        if not np.all(counts == 1):
            continue
        ref_cols = {int(color): int(np.flatnonzero(row == color)[0]) for color in vals}
        candidates.append((r, ref_cols))

    assert len(candidates) == 1, "reference row must be unique"
    ref_row, ref_cols = candidates[0]
    return int(ref_row), ref_cols, all_colors


def _band_majority_col(work: np.ndarray, color: int) -> int:
    coords = np.argwhere(work == int(color))
    assert coords.size > 0, f"missing band color {color}"
    counts = Counter(int(c) for _r, c in coords)
    best = max(counts.values())
    candidates = [c for c, n in counts.items() if n == best]
    return min(candidates)


def _shift_band(work: np.ndarray, color: int, shift: int, out: np.ndarray) -> None:
    h, w = work.shape
    for r, c in np.argwhere(work == int(color)):
        nc = int(c) + int(shift)
        if 0 <= nc < w:
            out[int(r), nc] = int(color)


def _row_majority_color(row: np.ndarray) -> int:
    nz = row[row != 0]
    if nz.size == 0:
        return 0
    vals, counts = np.unique(nz, return_counts=True)
    best = int(counts.max())
    tied = {int(v) for v, n in zip(vals, counts) if int(n) == best}
    for value in row:
        value = int(value)
        if value in tied:
            return value
    raise AssertionError("unreachable")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align rough horizontal color bands to columns specified by a sparse reference row, then compress each remaining row to one majority-color pixel.

    Input structure:
    - The background is black.
    - There are 3 or 4 horizontal monochrome bands, each spanning 2 or 3 nearby rows with ragged left/right edges and occasional black holes.
    - Exactly one sparse reference row contains each band color exactly once, with all other cells black.

    Rule:
    - Find the unique row whose non-black cells are exactly the full band-color set and each appears once; record each color's column there and remove that row.
    - For each color band, count in which columns that color appears across the remaining grid and take the most frequent column as the band's anchor column.
    - Translate every pixel of that color horizontally so the anchor column lands on the reference-row column for that color, clipping off-grid pixels and filling emptied cells with black.
    - For every row of the aligned grid, output one pixel equal to the row's majority non-black color, or black if the row is empty.

    Key insight:
    - The sparse reference row is the only place where all band colors co-occur exactly once, and it tells where each color band must be horizontally aligned before row-wise compression.
    """
    g = np.asarray(grid, dtype=int)
    ref_row, ref_cols, all_colors = _find_reference_row(g)
    work = np.delete(g, ref_row, axis=0)

    aligned = np.zeros_like(work)
    for color in all_colors:
        major_col = _band_majority_col(work, color)
        shift = int(ref_cols[color]) - int(major_col)
        _shift_band(work, color, shift, aligned)

    out = np.array([_row_majority_color(row) for row in aligned], dtype=int)
    return out[:, None]


def _sample_major_col(width: int, rng: random.Random) -> int:
    edge_candidates = [c for c in (1, 2, width - 3, width - 2) if 0 <= c < width]
    if edge_candidates and rng.random() < 0.6:
        return int(rng.choice(edge_candidates))
    return int(rng.randint(1, width - 2))


def _sample_separated_columns(width: int, count: int, rng: random.Random) -> list[int] | None:
    cols = list(range(width))
    for _ in range(120):
        picked = sorted(rng.sample(cols, count))
        if all(b - a >= 2 for a, b in zip(picked, picked[1:])):
            rng.shuffle(picked)
            return picked
    return None


def _sample_major_row(width: int, major: int, rng: random.Random) -> list[int]:
    for _ in range(120):
        left = rng.randint(0, 2)
        right = rng.randint(0, 2)
        if left == 0 and right == 0:
            if major == 0:
                right = 1
            elif major == width - 1:
                left = 1
            else:
                if rng.random() < 0.5:
                    left = 1
                else:
                    right = 1

        cols = list(range(max(0, major - left), min(width - 1, major + right) + 1))
        removable = [c for c in cols if c != major]
        if len(cols) >= 4 and removable and rng.random() < 0.5:
            remove_count = rng.randint(1, min(2, len(removable)))
            removed = set(rng.sample(removable, remove_count))
            cols = [c for c in cols if c not in removed]
        if len(cols) >= 2 and major in cols:
            return cols
    raise RuntimeError("failed to sample a major row")


def _sample_side_row(width: int, major: int, shift: int, rng: random.Random) -> list[int] | None:
    if shift == 0:
        return None
    side = -1 if shift < 0 else 1
    candidates = [major + side * d for d in (1, 2) if 0 <= major + side * d < width]
    if not candidates:
        return None
    if len(candidates) == 1 or rng.random() < 0.5:
        return [int(rng.choice(candidates))]
    return sorted(candidates)


def _sample_band_rows(width: int, height: int, major: int, shift: int, rng: random.Random) -> list[list[int]] | None:
    force_missing = height == 3 and shift != 0 and rng.random() < 0.65
    for _ in range(240):
        rows: list[list[int]] = []
        missing_index = rng.randrange(height) if force_missing else None
        for idx in range(height):
            if idx == missing_index:
                row = _sample_side_row(width, major, shift, rng)
                if row is None:
                    break
            else:
                row = _sample_major_row(width, major, rng)
            rows.append(sorted(set(row)))
        if len(rows) != height:
            continue

        counts = Counter(c for row in rows for c in row)
        best = max(counts.values())
        if counts.get(major, 0) != best:
            continue
        if sum(n == best for n in counts.values()) != 1:
            continue
        if len({tuple(row) for row in rows}) == 1:
            continue

        spans = {(min(row), max(row), len(row)) for row in rows}
        has_gap = any(len(row) != max(row) - min(row) + 1 for row in rows)
        if len(spans) == 1 and not has_gap:
            continue
        return rows
    return None


def _expected_output(
    row_plan: list[dict[str, object] | None],
    ref_row: int,
    width: int,
) -> np.ndarray:
    values: list[int] = []
    for r, info in enumerate(row_plan):
        if r == ref_row:
            continue
        if info is None:
            values.append(0)
            continue
        cols = [int(c) for c in info["cols"]]
        shift = int(info["shift"])
        shifted = [c + shift for c in cols if 0 <= c + shift < width]
        values.append(int(info["color"]) if shifted else 0)
    return np.array(values, dtype=int)[:, None]


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    width = rng.randint(9, 15)
    band_count = rng.randint(3, 4)
    colors = rng.sample(PALETTE, band_count)

    heights = [rng.randint(2, 3) for _ in range(band_count)]
    if 3 not in heights and rng.random() < 0.7:
        heights[rng.randrange(band_count)] = 3

    major_cols = [_sample_major_col(width, rng) for _ in range(band_count)]
    ref_cols = _sample_separated_columns(width, band_count, rng)
    if ref_cols is None:
        return None
    if max(abs(a - b) for a, b in zip(major_cols, ref_cols)) < 2:
        return None

    band_rows: list[list[list[int]]] = []
    for height, major, ref_col in zip(heights, major_cols, ref_cols):
        rows = _sample_band_rows(width, height, major, ref_col - major, rng)
        if rows is None:
            return None
        band_rows.append(rows)

    items: list[tuple[str, int | None]] = []
    ref_slot = rng.randrange(band_count + 1)
    for idx in range(band_count + 1):
        if idx == ref_slot:
            items.append(("ref", None))
        if idx < band_count:
            items.append(("band", idx))

    top_pad = rng.randint(0, 2)
    bottom_pad = rng.randint(0, 2)
    gaps = [rng.randint(0, 1) for _ in range(len(items) - 1)]
    height_total = top_pad + bottom_pad + 1 + sum(heights) + sum(gaps)
    if height_total > 30:
        return None

    grid = np.zeros((height_total, width), dtype=int)
    row_plan: list[dict[str, object] | None] = [None for _ in range(height_total)]

    r = top_pad
    ref_row = -1
    for item_index, (kind, payload) in enumerate(items):
        if kind == "ref":
            ref_row = r
            for color, col in zip(colors, ref_cols):
                grid[r, col] = color
            r += 1
        else:
            band_idx = int(payload)
            color = colors[band_idx]
            shift = ref_cols[band_idx] - major_cols[band_idx]
            for row_cols in band_rows[band_idx]:
                grid[r, row_cols] = color
                row_plan[r] = {"color": color, "cols": tuple(row_cols), "shift": shift}
                r += 1
        if item_index + 1 < len(items):
            r += gaps[item_index]

    r += bottom_pad
    if r != height_total or ref_row < 0:
        return None

    expected = _expected_output(row_plan, ref_row, width)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
