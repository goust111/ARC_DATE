from __future__ import annotations

import random

import numpy as np


BLACK = 0
NON_BLACK_COLORS = (1, 2, 3, 4, 5, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 500


def _segments(row: np.ndarray) -> list[tuple[int, int, int]]:
    """Return contiguous (color, start, end) runs for one row."""
    row = np.asarray(row, dtype=int)
    cuts = np.flatnonzero(row[1:] != row[:-1]) + 1
    starts = np.concatenate(([0], cuts))
    ends = np.concatenate((cuts - 1, [row.size - 1]))
    return [(int(row[s]), int(s), int(e)) for s, e in zip(starts, ends, strict=False)]


def _parse_top_row(row: np.ndarray) -> dict[str, int]:
    segs = _segments(row)
    assert len(segs) == 3, "each upper row must contain exactly three contiguous color segments"

    lengths = [end - start + 1 for _, start, end in segs]
    assert lengths[1] > lengths[0] and lengths[1] > lengths[2], "middle segment must be uniquely longest"

    (a_color, a0, a1), (b_color, b0, b1), (c_color, c0, c1) = segs
    assert a0 == 0 and c1 == row.size - 1, "the three segments must span the full row"
    assert a_color != b_color and b_color != c_color and a_color != c_color, "the three segment colors must differ"
    return {
        "a_color": a_color,
        "b_color": b_color,
        "c_color": c_color,
        "a_width": a1 - a0 + 1,
        "b_width": b1 - b0 + 1,
        "c_width": c1 - c0 + 1,
        "b_end": b1,
    }


def _parse_input(grid: np.ndarray) -> tuple[list[dict[str, int]], int, int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height % 2 == 0, "grid height must split into equal top and bottom halves"

    half = height // 2
    assert np.all(g[half:, :] == BLACK), "lower half must start empty"

    rows = [_parse_top_row(g[r]) for r in range(half)]
    b_widths = {row["b_width"] for row in rows}
    assert len(b_widths) == 1, "all middle segments must share one width"

    right_edges = [row["b_end"] for row in rows]
    assert len(set(right_edges)) == len(right_edges), "middle-segment right edges must be unique"
    return rows, half, width


def _render_output(rows: list[dict[str, int]], half: int, width: int) -> np.ndarray:
    out = np.full((2 * half, width), BLACK, dtype=int)

    b_width = rows[0]["b_width"]
    cmin = min(row["b_end"] for row in rows)
    aligned_b_start = cmin - b_width + 1
    assert aligned_b_start >= 1, "aligned middle segment must leave a visible left side segment"
    assert cmin < width - 1, "aligned middle segment must leave a visible right side segment"

    for r, row in enumerate(rows):
        out[r, :aligned_b_start] = row["a_color"]
        out[r, aligned_b_start : cmin + 1] = row["b_color"]
        out[r, cmin + 1 :] = row["c_color"]

        shift = row["b_end"] - cmin
        assert shift >= 0
        if shift:
            start = (width - shift) // 2
            out[half + r, start : start + shift] = row["a_color"]

    return out


def _parse_rows_without_lower_checks(top: np.ndarray) -> list[dict[str, int]]:
    rows = [_parse_top_row(top[r]) for r in range(top.shape[0])]
    b_widths = {row["b_width"] for row in rows}
    assert len(b_widths) == 1, "all middle segments must share one width"
    return rows


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align each row's longest middle segment by its right edge, then encode the lost shift as a centered line.

    Input structure:
    - The grid has an even height with an upper band of patterned rows and a lower band of black rows.
    - Every upper-row spans the full width with exactly three contiguous color segments: left color A, longest middle color B, and right color C.
    - All middle B segments share one width, but their starting columns differ, so their right edges are all distinct.

    Rule:
    - For each upper row, identify the uniquely longest middle segment B and record its right edge.
    - Find the global minimum of those right-edge columns, then rebuild every upper row so its B segment ends at that shared minimum column while preserving that row's three colors.
    - Let D be the original row's horizontal offset from that minimum right edge.
    - In the corresponding lower row, draw a horizontal segment of length D in color A, centered within the row; leave lower rows black when D = 0.

    Key insight:
    - The visible lower-half lines do not encode object size; they encode the displacement needed to align the upper rows' middle segments.
    """
    rows, half, width = _parse_input(grid)
    return _render_output(rows, half, width)


def _choose_shifts(half: int, rng: random.Random) -> list[int] | None:
    max_shift = rng.randint(max(half, 4), 10)
    positives = list(range(1, max_shift + 1))
    if len(positives) < half - 1:
        return None

    for _ in range(50):
        chosen = sorted(rng.sample(positives, half - 1))
        if half >= 4 and chosen == list(range(1, half)):
            continue
        return [0, *chosen]
    return None


def _feasible_geometry(width: int, shifts: list[int]) -> list[tuple[int, int]]:
    max_shift = max(shifts)
    min_b = (width + 2) // 3
    options: list[tuple[int, int]] = []
    for a_min in range(1, width - 2):
        max_a = a_min + max_shift
        for b_width in range(min_b, width - 1):
            max_c = width - b_width - a_min
            min_c = width - b_width - max_a
            if min_c < 1:
                continue
            if b_width <= max_a or b_width <= max_c:
                continue
            options.append((a_min, b_width))
    return options


def _build_input(
    half: int,
    width: int,
    shifts: list[int],
    a_min: int,
    b_width: int,
    a_colors: list[int],
    b_color: int,
    c_color: int,
    rng: random.Random,
) -> np.ndarray:
    grid = np.full((2 * half, width), BLACK, dtype=int)
    order = list(range(half))
    rng.shuffle(order)
    permuted_shifts = [shifts[i] for i in order]

    for r, shift in enumerate(permuted_shifts):
        a_width = a_min + shift
        c_width = width - a_width - b_width
        assert c_width >= 1
        row = np.empty(width, dtype=int)
        row[:a_width] = a_colors[r]
        row[a_width : a_width + b_width] = b_color
        row[a_width + b_width :] = c_color
        grid[r] = row
    return grid


def _validate_generated_input(grid: np.ndarray) -> None:
    rows, half, width = _parse_input(grid)
    shifts = [row["a_width"] - min(r["a_width"] for r in rows) for row in rows]
    assert sorted(shifts)[0] == 0 and len(set(shifts)) == len(shifts)
    assert max(shifts) <= 10
    assert max(shifts) >= 1

    a_colors = [row["a_color"] for row in rows]
    assert len(set(a_colors)) == len(a_colors), "generator should keep left colors unique per row for clarity"

    out = main(grid)
    out_rows = _parse_rows_without_lower_checks(out[:half])

    out_right_edges = [row["b_end"] for row in out_rows]
    assert len(set(out_right_edges)) == 1, "aligned output must share one common right edge"

    for r, shift in enumerate([row["b_end"] - min(row2["b_end"] for row2 in rows) for row in rows]):
        lower = out[half + r]
        nz = np.flatnonzero(lower != BLACK)
        if shift == 0:
            assert nz.size == 0, "zero displacement should produce an empty lower row"
            continue
        assert nz.size == shift, "encoded line length must match the shift"
        assert np.all(np.diff(nz) == 1), "encoded line must be contiguous"
        start = (width - shift) // 2
        assert int(nz[0]) == start and int(nz[-1]) == start + shift - 1, "encoded line must be centered"
        assert np.all(lower[nz] == rows[r]["a_color"]), "encoded line color must inherit the row's left color"


def generate_input() -> np.ndarray:
    rng = random.Random()

    for _ in range(MAX_GENERATION_ATTEMPTS):
        half = rng.randint(5, 7)
        width = rng.randint(18, 24)
        shifts = _choose_shifts(half, rng)
        if shifts is None:
            continue

        geometry = _feasible_geometry(width, shifts)
        if not geometry:
            continue
        a_min, b_width = rng.choice(geometry)

        b_color = rng.choice(NON_BLACK_COLORS)
        remaining = [color for color in NON_BLACK_COLORS if color != b_color]
        c_color = rng.choice(remaining)
        a_palette = [color for color in remaining if color != c_color]
        if len(a_palette) < half:
            continue
        a_colors = rng.sample(a_palette, half)

        grid = _build_input(half, width, shifts, a_min, b_width, a_colors, b_color, c_color, rng)
        try:
            _validate_generated_input(grid)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid input for puzzle 5")
