import random
from collections import Counter

import numpy as np


BG = 0
SEP = 7
GEN_RETRIES = 400
GAP_SEARCH_RETRIES = 600


def _corner_bg(grid: np.ndarray) -> int:
    vals = [int(grid[0, 0]), int(grid[0, -1]), int(grid[-1, 0]), int(grid[-1, -1])]
    return int(Counter(vals).most_common(1)[0][0])


def _find_separator_axes(grid: np.ndarray, sep: int) -> tuple[list[int], list[int]]:
    row_counts = np.sum(grid == sep, axis=1)
    col_counts = np.sum(grid == sep, axis=0)

    row_threshold = max(3, int(np.max(row_counts) * 0.6))
    col_threshold = max(3, int(np.max(col_counts) * 0.6))

    rows = [int(r) for r, cnt in enumerate(row_counts) if int(cnt) >= row_threshold]
    cols = [int(c) for c, cnt in enumerate(col_counts) if int(cnt) >= col_threshold]
    assert len(rows) >= 2 and len(cols) >= 2, "failed to detect separator lattice"

    row_step = rows[1] - rows[0]
    col_step = cols[1] - cols[0]
    assert row_step >= 2 and col_step >= 2, "panels must have positive interior"
    assert all(rows[i + 1] - rows[i] == row_step for i in range(len(rows) - 1))
    assert all(cols[i + 1] - cols[i] == col_step for i in range(len(cols) - 1))
    return rows, cols


def _read_legend(grid: np.ndarray, bg: int, panel_bbox: tuple[int, int, int, int]) -> dict[int, int]:
    r0, c0, r1, c1 = panel_bbox
    outside = np.zeros_like(grid, dtype=bool)
    outside[:, :c0] = True
    outside[:, c1 + 1 :] = True
    outside[:r0, :] = True
    outside[r1 + 1 :, :] = True

    mapping: dict[int, int] = {}
    for r in range(grid.shape[0]):
        cols = [int(c) for c in np.where(outside[r] & (grid[r] != bg))[0]]
        if not cols:
            continue
        cols.sort()
        if any(cols[i + 1] != cols[i] + 1 for i in range(len(cols) - 1)):
            continue
        if len(cols) < 2:
            continue
        signature = len(cols) - 1
        color = int(grid[r, cols[-1]])
        mapping[signature] = color

    assert set(mapping) == {1, 2, 3}, "legend must define signatures 1..3"
    return mapping


def _panel_gap_count(grid: np.ndarray, bg: int, r0: int, r1: int, c0: int, c1: int) -> int:
    border = set()
    for c in range(c0, c1 + 1):
        border.add((r0, c))
        border.add((r1, c))
    for r in range(r0, r1 + 1):
        border.add((r, c0))
        border.add((r, c1))
    return sum(int(grid[r, c]) == bg for r, c in border)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract each panel's border-gap signature, map it through the legend, and
    fill the panel interior with the mapped color while erasing the legend.
    """
    g = np.asarray(grid, dtype=int)
    bg = _corner_bg(g)

    counts = Counter(map(int, g.ravel()))
    sep = max((color for color in counts if color != bg), key=lambda color: counts[color])

    sep_rows, sep_cols = _find_separator_axes(g, sep)
    panel_bbox = (sep_rows[0], sep_cols[0], sep_rows[-1], sep_cols[-1])
    mapping = _read_legend(g, bg, panel_bbox)

    out = np.full_like(g, bg)
    r0, c0, r1, c1 = panel_bbox
    out[r0 : r1 + 1, c0 : c1 + 1] = g[r0 : r1 + 1, c0 : c1 + 1]

    for ra, rb in zip(sep_rows, sep_rows[1:]):
        for ca, cb in zip(sep_cols, sep_cols[1:]):
            gaps = _panel_gap_count(g, bg, ra, rb, ca, cb)
            assert 1 <= gaps <= 3, "panel signature must be 1..3"
            fill = mapping[gaps]
            interior = out[ra + 1 : rb, ca + 1 : cb]
            interior[interior == bg] = fill

    return out


def _expected_output(grid: np.ndarray, mapping: dict[int, int], sep_rows: list[int], sep_cols: list[int]) -> np.ndarray:
    out = np.full_like(grid, BG)
    out[sep_rows[0] : sep_rows[-1] + 1, sep_cols[0] : sep_cols[-1] + 1] = grid[
        sep_rows[0] : sep_rows[-1] + 1, sep_cols[0] : sep_cols[-1] + 1
    ]
    for ra, rb in zip(sep_rows, sep_rows[1:]):
        for ca, cb in zip(sep_cols, sep_cols[1:]):
            gaps = _panel_gap_count(grid, BG, ra, rb, ca, cb)
            out[ra + 1 : rb, ca + 1 : cb] = mapping[gaps]
    return out


def _sample_fill_colors(rng: random.Random, dot_color: int) -> dict[int, int]:
    palette = [c for c in range(1, 10) if c not in (SEP, dot_color)]
    rng.shuffle(palette)
    colors = palette[:3]
    return {1: colors[0], 2: colors[1], 3: colors[2]}


def _make_lattice(nr: int, nc: int, ph: int, pw: int) -> tuple[np.ndarray, list[int], list[int]]:
    h = nr * ph + (nr + 1)
    w = nc * pw + (nc + 1)
    grid = np.full((h, w), BG, dtype=int)
    sep_rows = [i * (ph + 1) for i in range(nr + 1)]
    sep_cols = [j * (pw + 1) for j in range(nc + 1)]

    for r in sep_rows:
        grid[r, :] = SEP
    for c in sep_cols:
        grid[:, c] = SEP
    return grid, sep_rows, sep_cols


def _candidate_gap_cells(sep_rows: list[int], sep_cols: list[int]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for r in sep_rows:
        for ca, cb in zip(sep_cols, sep_cols[1:]):
            for c in range(ca + 1, cb):
                cells.append((r, c))
    for c in sep_cols:
        for ra, rb in zip(sep_rows, sep_rows[1:]):
            for r in range(ra + 1, rb):
                cells.append((r, c))
    return cells


def _panel_counts(grid: np.ndarray, sep_rows: list[int], sep_cols: list[int]) -> list[int]:
    result = []
    for ra, rb in zip(sep_rows, sep_rows[1:]):
        for ca, cb in zip(sep_cols, sep_cols[1:]):
            result.append(_panel_gap_count(grid, BG, ra, rb, ca, cb))
    return result


def _place_gaps(rng: random.Random, grid: np.ndarray, sep_rows: list[int], sep_cols: list[int]) -> bool:
    cells = _candidate_gap_cells(sep_rows, sep_cols)
    for _ in range(GAP_SEARCH_RETRIES):
        work = grid.copy()
        p = rng.uniform(0.05, 0.16)
        for r, c in cells:
            if rng.random() < p:
                work[r, c] = BG
        counts = _panel_counts(work, sep_rows, sep_cols)
        if all(1 <= cnt <= 3 for cnt in counts) and set(counts) == {1, 2, 3}:
            grid[:, :] = work
            return True
    return False


def _legend_rows(rng: random.Random, height: int, sep_rows: list[int]) -> list[int]:
    interior_rows = [r for r in range(height) if r not in set(sep_rows)]
    if len(interior_rows) < 3:
        return []
    picks = sorted(rng.sample(interior_rows, 3))
    return picks


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    nr = rng.randint(2, 3)
    nc = rng.randint(2, 4)
    if nr * nc < 3:
        return None
    ph = rng.randint(3, 5)
    pw = rng.randint(3, 6)
    legend_side = rng.choice(["left", "right"])
    gap = rng.randint(2, 3)
    legend_w = 5
    dot_color = rng.choice([1, 2, 3, 4, 5, 6, 8, 9])
    mapping = _sample_fill_colors(rng, dot_color)

    lattice, sep_rows, sep_cols = _make_lattice(nr, nc, ph, pw)
    if not _place_gaps(rng, lattice, sep_rows, sep_cols):
        return None

    h, w = lattice.shape
    full = np.full((h, w + legend_w + gap), BG, dtype=int)
    if legend_side == "left":
        col0 = legend_w + gap
        full[:, col0 : col0 + w] = lattice
        panel_cols = [c + col0 for c in sep_cols]
        legend_x = 0
    else:
        full[:, :w] = lattice
        panel_cols = sep_cols[:]
        legend_x = w + gap

    rows = _legend_rows(rng, h, sep_rows)
    if len(rows) != 3:
        return None
    for signature, r in zip([1, 2, 3], rows):
        full[r, legend_x : legend_x + signature] = dot_color
        full[r, legend_x + signature] = mapping[signature]

    expected = _expected_output(full, mapping, sep_rows, panel_cols)
    try:
        actual = main(full)
    except Exception:
        return None
    if not np.array_equal(expected, actual):
        return None
    return full


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
