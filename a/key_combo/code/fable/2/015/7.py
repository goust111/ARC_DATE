import numpy as np


BG_COLOR = 0
PALETTE = tuple(range(1, 10))
GEN_RETRIES = 500


def _most_common(values: np.ndarray) -> int:
    vals, counts = np.unique(values.astype(int), return_counts=True)
    return int(vals[np.argmax(counts)])


def _border_bg(grid: np.ndarray) -> int:
    border = np.concatenate(
        [
            grid[0, :],
            grid[-1, :],
            grid[1:-1, 0],
            grid[1:-1, -1],
        ]
    )
    return _most_common(border)


def _signature(values: np.ndarray, bg: int) -> tuple[int, ...]:
    return tuple(sorted(int(v) for v in np.unique(values) if int(v) != bg))


def _group_runs(signatures: list[tuple[int, ...]], offset: int = 0) -> list[tuple[int, int, tuple[int, ...]]]:
    groups: list[tuple[int, int, tuple[int, ...]]] = []
    start = 0
    current = signatures[0]
    for idx in range(1, len(signatures) + 1):
        if idx == len(signatures) or signatures[idx] != current:
            groups.append((offset + start, offset + idx, current))
            if idx < len(signatures):
                start = idx
                current = signatures[idx]
    return groups


def _compress_lattice(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _border_bg(g)

    row_signatures = [_signature(g[r, :], bg) for r in range(g.shape[0])]
    col_signatures = [_signature(g[:, c], bg) for c in range(g.shape[1])]

    row_nonempty = [idx for idx, sig in enumerate(row_signatures) if sig]
    col_nonempty = [idx for idx, sig in enumerate(col_signatures) if sig]
    assert row_nonempty and col_nonempty, "expected one foreground lattice"

    row_lo, row_hi = row_nonempty[0], row_nonempty[-1]
    col_lo, col_hi = col_nonempty[0], col_nonempty[-1]
    assert all(row_signatures[idx] for idx in range(row_lo, row_hi + 1))
    assert all(col_signatures[idx] for idx in range(col_lo, col_hi + 1))

    row_bands = _group_runs(row_signatures[row_lo : row_hi + 1], offset=row_lo)
    col_bands = _group_runs(col_signatures[col_lo : col_hi + 1], offset=col_lo)
    assert row_bands and col_bands

    coarse = np.full((len(row_bands), len(col_bands)), bg, dtype=int)
    for rr, (r0, r1, _row_sig) in enumerate(row_bands):
        for cc, (c0, c1, _col_sig) in enumerate(col_bands):
            block = g[r0:r1, c0:c1]
            colors = [int(v) for v in np.unique(block) if int(v) != bg]
            assert len(colors) == 1, "each coarse cell must contain one visible foreground color"
            coarse[rr, cc] = colors[0]
    return coarse


def _repair_coarse(coarse: np.ndarray) -> np.ndarray:
    g = np.asarray(coarse, dtype=int)
    rows, cols = g.shape
    assert rows >= 2 and cols >= 2

    lr_suspects = np.zeros_like(g, dtype=bool)
    ud_suspects = np.zeros_like(g, dtype=bool)

    for r in range(rows):
        for c in range(cols // 2):
            mc = cols - 1 - c
            if int(g[r, c]) != int(g[r, mc]):
                lr_suspects[r, c] = True
                lr_suspects[r, mc] = True

    for r in range(rows // 2):
        mr = rows - 1 - r
        for c in range(cols):
            if int(g[r, c]) != int(g[mr, c]):
                ud_suspects[r, c] = True
                ud_suspects[mr, c] = True

    candidates = np.argwhere(lr_suspects & ud_suspects)
    assert candidates.shape == (1, 2), "expected one cross-axis culprit"
    r, c = map(int, candidates[0])
    mr = rows - 1 - r
    mc = cols - 1 - c
    assert r != mr and c != mc, "the wrong cell must sit off both symmetry axes"

    partner_colors = {
        int(g[r, mc]),
        int(g[mr, c]),
        int(g[mr, mc]),
    }
    assert len(partner_colors) == 1, "the three mirror partners must agree"

    fixed = g.copy()
    fixed[r, c] = partner_colors.pop()
    assert np.array_equal(fixed, fixed[:, ::-1])
    assert np.array_equal(fixed, fixed[::-1, :])
    return fixed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress a noisy rectangular block lattice, find the one coarse cell disproved by both mirror axes, and repair it.

    Input structure:
    - A background border surrounds one rectangular lattice made of touching monochrome blocks.
    - Block heights and widths vary by row band and column band, and background-colored holes may appear inside blocks.
    - All rows inside the same band show the same visible foreground-color set, and all columns inside the same band do too.

    Rule:
    - Segment the lattice into row bands and column bands using those visible color sets.
    - Compress each band intersection to its unique non-background color to obtain a coarse grid.
    - Mark every left-right mirror mismatch pair and every top-bottom mirror mismatch pair; the only cell accused by both axes is the wrong one.
    - Recolor that cell to the shared color of its other three symmetric partners and output the repaired coarse grid.

    Key insight:
    - Pixel-level holes destroy obvious symmetry, so the symmetry must be recovered only after coarse compression, and one mirror axis alone cannot decide which side is wrong.
    """
    coarse = _compress_lattice(np.asarray(grid, dtype=int))
    return _repair_coarse(coarse)


def _coarse_signatures(coarse: np.ndarray) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    row_signatures = [tuple(sorted(set(map(int, row)))) for row in coarse]
    col_signatures = [tuple(sorted(set(map(int, coarse[:, c])))) for c in range(coarse.shape[1])]
    return row_signatures, col_signatures


def _adjacent_signatures_distinct(signatures: list[tuple[int, ...]]) -> bool:
    return all(signatures[idx] != signatures[idx + 1] for idx in range(len(signatures) - 1))


def _random_symmetric_coarse(rng: np.random.Generator) -> np.ndarray:
    rows = int(rng.integers(3, 7))
    cols = int(rng.integers(3, 7))
    palette_size = int(rng.integers(3, 6))
    palette = rng.choice(np.array(PALETTE, dtype=int), size=palette_size, replace=False)

    coarse = np.empty((rows, cols), dtype=int)
    for r in range((rows + 1) // 2):
        for c in range((cols + 1) // 2):
            color = int(rng.choice(palette))
            orbit = {
                (r, c),
                (r, cols - 1 - c),
                (rows - 1 - r, c),
                (rows - 1 - r, cols - 1 - c),
            }
            for rr, cc in orbit:
                coarse[rr, cc] = color
    return coarse


def _off_axis_cells(shape: tuple[int, int]) -> list[tuple[int, int]]:
    rows, cols = shape
    return [
        (r, c)
        for r in range(rows)
        for c in range(cols)
        if r != rows - 1 - r and c != cols - 1 - c
    ]


def _corrupt_symmetric_coarse(base: np.ndarray, rng: np.random.Generator) -> np.ndarray | None:
    colors, counts = np.unique(base, return_counts=True)
    usable_wrong_colors = [int(color) for color, count in zip(colors, counts) if int(count) >= 2]
    off_axis = _off_axis_cells(base.shape)
    if not off_axis:
        return None

    r, c = off_axis[int(rng.integers(len(off_axis)))]
    target = int(base[r, c])
    wrong_choices = [color for color in usable_wrong_colors if color != target]
    if not wrong_choices:
        return None

    corrupted = base.copy()
    corrupted[r, c] = int(rng.choice(np.array(wrong_choices, dtype=int)))
    return corrupted


def _random_band_sizes(rng: np.random.Generator, count: int) -> list[int]:
    for _ in range(100):
        sizes = rng.integers(2, 5, size=count)
        if len(set(map(int, sizes))) > 1:
            return [int(v) for v in sizes]
    sizes = [2] * count
    sizes[0] = 3
    return sizes


def _noisy_block(h: int, w: int, color: int, rng: np.random.Generator, force_hole: bool) -> np.ndarray:
    block = np.full((h, w), int(color), dtype=int)
    row_counts = np.full(h, w, dtype=int)
    col_counts = np.full(w, h, dtype=int)
    max_holes = max(1, (h * w) // 5)
    target_holes = int(rng.integers(0, max_holes + 1))
    if force_hole and target_holes == 0:
        target_holes = 1

    coords = [(r, c) for r in range(h) for c in range(w)]
    holes = 0
    for idx in rng.permutation(len(coords)):
        if holes >= target_holes:
            break
        r, c = coords[int(idx)]
        if row_counts[r] > 1 and col_counts[c] > 1 and rng.random() < 0.7:
            block[r, c] = BG_COLOR
            row_counts[r] -= 1
            col_counts[c] -= 1
            holes += 1

    if force_hole and holes == 0:
        for r, c in coords:
            if row_counts[r] > 1 and col_counts[c] > 1:
                block[r, c] = BG_COLOR
                break
    return block


def _expand_to_lattice(
    coarse: np.ndarray,
    row_sizes: list[int],
    col_sizes: list[int],
    margins: tuple[int, int, int, int],
    rng: np.random.Generator,
) -> np.ndarray:
    top, bottom, left, right = margins
    height = top + sum(row_sizes) + bottom
    width = left + sum(col_sizes) + right
    grid = np.full((height, width), BG_COLOR, dtype=int)

    hole_anchor = (
        int(rng.integers(coarse.shape[0])),
        int(rng.integers(coarse.shape[1])),
    )
    row_start = top
    for r, block_h in enumerate(row_sizes):
        col_start = left
        for c, block_w in enumerate(col_sizes):
            block = _noisy_block(
                block_h,
                block_w,
                int(coarse[r, c]),
                rng,
                force_hole=(r, c) == hole_anchor,
            )
            grid[row_start : row_start + block_h, col_start : col_start + block_w] = block
            col_start += block_w
        row_start += block_h
    return grid


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    base = _random_symmetric_coarse(rng)
    if len(set(map(int, base.flatten()))) < 3:
        return None

    corrupted = _corrupt_symmetric_coarse(base, rng)
    if corrupted is None:
        return None

    row_sigs, col_sigs = _coarse_signatures(corrupted)
    if not _adjacent_signatures_distinct(row_sigs):
        return None
    if not _adjacent_signatures_distinct(col_sigs):
        return None

    try:
        repaired = _repair_coarse(corrupted)
    except AssertionError:
        return None
    if not np.array_equal(repaired, base):
        return None

    row_sizes = _random_band_sizes(rng, corrupted.shape[0])
    col_sizes = _random_band_sizes(rng, corrupted.shape[1])
    top = int(rng.integers(1, 4))
    bottom = int(rng.integers(1, 4))
    left = int(rng.integers(1, 4))
    right = int(rng.integers(1, 4))

    grid = _expand_to_lattice(
        corrupted,
        row_sizes,
        col_sizes,
        (top, bottom, left, right),
        rng,
    )
    if max(grid.shape) > 30:
        return None

    try:
        compressed = _compress_lattice(grid)
        solved = main(grid)
    except AssertionError:
        return None

    if not np.array_equal(compressed, corrupted):
        return None
    if not np.array_equal(solved, base):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid double-mirror lattice input")
