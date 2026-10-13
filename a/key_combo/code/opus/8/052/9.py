from __future__ import annotations

from collections import Counter

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _triangular_columns(width: int) -> list[int]:
    cols: list[int] = []
    col = 0
    step = 1
    while col < width:
        cols.append(col)
        col += step
        step += 1
    return cols


def _build_template_row(width: int, bg: int, cycle: tuple[int, int]) -> np.ndarray:
    row = np.full(width, bg, dtype=int)
    for i, c in enumerate(_triangular_columns(width)):
        row[c] = cycle[i % 2]
    return row


def _decode_prefix_row(
    row: np.ndarray,
    template: np.ndarray,
    tri_cols: list[int],
    bg: int,
) -> dict[int, int]:
    nz = np.flatnonzero(row != bg)
    assert len(nz) >= 2, "each prefix row must show at least two active triangular positions"
    prefix_end = int(nz[-1])
    visible_tri = [c for c in tri_cols if c <= prefix_end]
    assert len(visible_tri) >= 2, "prefix must include the first two triangular positions"
    assert np.array_equal(nz, np.array(visible_tri)), "prefix rows must be sparse triangular prefixes"

    mapping: dict[int, int] = {}
    for c in visible_tri:
        key = int(template[c])
        value = int(row[c])
        assert key != bg and value != bg
        if key in mapping:
            assert mapping[key] == value, "template-to-row mapping must be consistent"
        else:
            mapping[key] = value

    assert len(mapping) == 2, "prefix must reveal both template roles"
    values = tuple(mapping[int(template[c])] for c in visible_tri[:2])
    assert values[0] != values[1], "row colors must be distinct"
    return {int(k): int(v) for k, v in mapping.items()}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the sparse triangular template row, then decode and complete each lower sparse prefix row.

    Input structure:
    - The background is the dominant color and row 1 is a full background separator.
    - Row 0 contains only the first few triangular-number columns 0, 1, 3, 6, 10, ...
      colored, alternating between exactly two colors; later triangular columns are still background.
    - Each nonempty row below the separator is a left-aligned sparse prefix: within some prefix width,
      exactly the triangular columns are colored using a row-specific pair of colors, and all other cells are background.

    Rule:
    - Read row 0's visible colors left to right as a 2-color alternating cycle and extend that cycle to every
      triangular-number column within the grid width; this yields the full template row.
    - For each lower sparse prefix row, pair its visible colored cells with the template row at the same columns
      to recover a one-to-one mapping from the template's two colors to that row's two colors.
    - Repaint the whole row by applying that mapping to every colored triangular position of the template,
      while all non-triangular columns remain background; keep the separator row unchanged.

    Key insight:
    - The upper row does not explicitly show the full template; the solver must first recognize triangular spacing
      and extend the hidden alternating cycle, then use column-wise alignment to decode each lower row's color mapping.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h >= 5 and w >= 2

    bg = _get_bg(g)
    assert np.all(g[1] == bg), "row 1 must be the separator row"

    tri_cols = _triangular_columns(w)
    seed_cols = np.flatnonzero(g[0] != bg)
    assert len(seed_cols) >= 2, "template seed must reveal both cycle colors"
    assert len(seed_cols) < len(tri_cols), "template row must need extension"
    assert np.array_equal(seed_cols, np.array(tri_cols[: len(seed_cols)])), "seed must occupy the first triangular columns"

    cycle = (int(g[0, seed_cols[0]]), int(g[0, seed_cols[1]]))
    assert cycle[0] != cycle[1], "template cycle must use two distinct colors"
    for i, c in enumerate(seed_cols):
        assert int(g[0, int(c)]) == cycle[i % 2], "seed colors must alternate on triangular positions"

    template = _build_template_row(w, bg, cycle)
    out = np.full_like(g, bg)
    out[0] = template
    out[1] = bg

    active_rows = 0
    for r in range(2, h):
        row = g[r]
        if not np.any(row != bg):
            continue
        mapping = _decode_prefix_row(row, template, tri_cols, bg)
        filled = np.full(w, bg, dtype=int)
        for c in tri_cols:
            filled[c] = mapping[int(template[c])]
        out[r] = filled
        active_rows += 1

    assert active_rows >= 1, "expected at least one sparse prefix row"
    return out


WIDTH_MIN = 16
WIDTH_MAX = 29
HEIGHT_OPTIONS = (5, 6)
SEED_VISIBLE_MIN = 2
SEED_VISIBLE_MAX = 4
GEN_ATTEMPTS = 400


def _ordered_pair_cycle(rng: np.random.Generator, colors: np.ndarray) -> list[tuple[int, int]]:
    chosen = rng.choice(colors, size=6, replace=False)
    pairs = [(int(chosen[0]), int(chosen[1])), (int(chosen[2]), int(chosen[3])), (int(chosen[4]), int(chosen[5]))]
    rng.shuffle(pairs)
    return pairs


def _visible_counts(tri_count: int, rng: np.random.Generator) -> list[int]:
    if tri_count == 6:
        mid = 3
        long = 5
    else:
        mid_choices = [3, 4]
        long_choices = [tri_count - 2, tri_count - 1]
        mid = int(rng.choice(mid_choices))
        long = int(rng.choice(long_choices))
        if long <= mid:
            long = min(tri_count - 1, mid + 1)
    counts = [2, mid, long]
    rng.shuffle(counts)
    return counts


def _build_input(
    width: int,
    height: int,
    cycle: tuple[int, int],
    seed_visible: int,
    row_orders: list[tuple[int, int]],
    row_visible_counts: list[int],
) -> np.ndarray:
    bg = 0
    grid = np.full((height, width), bg, dtype=int)
    tri_cols = _triangular_columns(width)

    for i, c in enumerate(tri_cols[:seed_visible]):
        grid[0, c] = cycle[i % 2]

    for idx, r in enumerate(range(2, 5)):
        first_color, second_color = row_orders[idx]
        count = row_visible_counts[idx]
        for j, c in enumerate(tri_cols[:count]):
            grid[r, c] = first_color if j % 2 == 0 else second_color
    return grid


def _expected_output(grid: np.ndarray) -> np.ndarray:
    return main(grid)


def _semantic_check(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    out = main(g)
    bg = 0
    h, w = g.shape
    tri_cols = _triangular_columns(w)
    tri_set = set(tri_cols)

    assert h in HEIGHT_OPTIONS
    assert np.all(g[1] == bg)
    assert np.array_equal(out[1], np.full(w, bg, dtype=int))

    seed_cols = np.flatnonzero(g[0] != bg)
    assert 2 <= len(seed_cols) <= min(SEED_VISIBLE_MAX, len(tri_cols) - 1)
    assert len(seed_cols) < len(tri_cols)
    assert np.array_equal(seed_cols, np.array(tri_cols[: len(seed_cols)]))
    assert np.count_nonzero(out[0]) == len(tri_cols)
    assert np.array_equal(np.flatnonzero(out[0] != bg), np.array(tri_cols))

    row_outputs: set[bytes] = set()
    pair_signatures: set[tuple[int, int]] = set()
    active_count = 0
    template_colors = set(map(int, out[0, tri_cols]))
    seed_visible = len(seed_cols)
    visible_lengths = []
    for r in range(2, h):
        row = g[r]
        nz = np.flatnonzero(row != bg)
        if len(nz) == 0:
            assert np.all(out[r] == bg)
            continue
        active_count += 1
        visible_lengths.append(len(nz))
        prefix_end = int(nz[-1])
        visible_tri = [c for c in tri_cols if c <= prefix_end]
        assert np.array_equal(nz, np.array(visible_tri))
        assert all(c in tri_set for c in nz)
        row_colors = tuple(sorted(set(map(int, row[nz]))))
        assert len(row_colors) == 2
        assert set(row_colors).isdisjoint(template_colors)
        pair_signatures.add(row_colors)
        assert np.array_equal(np.flatnonzero(out[r] != bg), np.array(tri_cols))
        out_colors = tuple(sorted(set(map(int, out[r, tri_cols]))))
        assert out_colors == row_colors
        row_outputs.add(out[r].tobytes())

    assert active_count == 3
    assert len(pair_signatures) == 3, "each prefix row should use a distinct color pair"
    assert len(row_outputs) == 3, "completed rows should remain distinct"

    # At least one row should expose only the minimum evidence, and one should expose a longer prefix.
    assert 2 in visible_lengths
    assert max(visible_lengths) >= 4 or (len(tri_cols) == 6 and max(visible_lengths) >= 5)
    assert max(visible_lengths) > seed_visible


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    width = int(rng.integers(WIDTH_MIN, WIDTH_MAX + 1))
    height = int(rng.choice(HEIGHT_OPTIONS))
    tri_cols = _triangular_columns(width)
    tri_count = len(tri_cols)
    if not (6 <= tri_count <= 8):
        return None

    seed_visible_hi = min(SEED_VISIBLE_MAX, tri_count - 2)
    if seed_visible_hi < SEED_VISIBLE_MIN:
        return None
    seed_visible = int(rng.integers(SEED_VISIBLE_MIN, seed_visible_hi + 1))

    palette = np.arange(1, 10, dtype=int)
    cycle_colors = tuple(map(int, rng.choice(palette, size=2, replace=False)))
    remaining = np.array([c for c in palette if int(c) not in cycle_colors], dtype=int)
    if len(remaining) < 6:
        return None

    row_pairs = _ordered_pair_cycle(rng, remaining)
    orders = [
        row_pairs[0],
        (row_pairs[1][1], row_pairs[1][0]),
        row_pairs[2] if int(rng.integers(2)) == 0 else (row_pairs[2][1], row_pairs[2][0]),
    ]
    row_visible_counts = _visible_counts(tri_count, rng)

    grid = _build_input(
        width=width,
        height=height,
        cycle=cycle_colors,
        seed_visible=seed_visible,
        row_orders=orders,
        row_visible_counts=row_visible_counts,
    )
    try:
        expected = _expected_output(grid)
        if not np.array_equal(main(grid), expected):
            return None
        _semantic_check(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid triangular-template decoding input")
