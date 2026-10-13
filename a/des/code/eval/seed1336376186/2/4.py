import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
GEN_RETRIES = 400
MIN_REGION_SIZE = 3
MAX_REGION_SIZE = 6
REGION_COUNT_CHOICES = (2, 3)


def _mode_color(values: np.ndarray) -> int:
    counts = np.bincount(values.astype(int), minlength=10)
    return int(np.argmax(counts))


def _separator_rows(g: np.ndarray) -> list[int]:
    return [int(r) for r in range(g.shape[0]) if np.all(g[r, :] != BG)]


def _separator_cols(g: np.ndarray) -> list[int]:
    return [int(c) for c in range(g.shape[1]) if np.all(g[:, c] != BG)]


def _bands_from_separators(size: int, separators: list[int]) -> list[tuple[int, int, int, int]]:
    assert separators == sorted(separators)
    assert separators and 0 <= separators[0] < size and 0 <= separators[-1] < size
    bands = []
    for left, right in zip(separators, separators[1:]):
        if right - left > 1:
            bands.append((left, left + 1, right - 1, right))
    assert bands
    return bands


def _mirrored_if_needed(pattern: np.ndarray, flip: bool) -> np.ndarray:
    return pattern[::-1].copy() if flip else pattern.copy()


def _fill_region(
    out: np.ndarray,
    region: np.ndarray,
    top_color: int,
    bottom_color: int,
    left_color: int,
    right_color: int,
    top: int,
    left: int,
) -> None:
    h, w = region.shape
    mask = region != BG
    row_hits = np.flatnonzero(mask.any(axis=1))
    col_hits = np.flatnonzero(mask.any(axis=0))

    if row_hits.size == 1 and col_hits.size >= 2:
        sample_r = int(row_hits[0])
        assert sample_r in (0, h - 1), "template row must hug the top or bottom boundary"
        pattern = region[sample_r, :].copy()
        same = top_color == bottom_color
        for local_r in range(h):
            distance = local_r if sample_r == 0 else h - 1 - local_r
            row = _mirrored_if_needed(pattern, (not same) and distance % 2 == 1)
            out[top + local_r, left : left + w] = row
        return

    if col_hits.size == 1 and row_hits.size >= 2:
        sample_c = int(col_hits[0])
        assert sample_c in (0, w - 1), "template column must hug the left or right boundary"
        pattern = region[:, sample_c].copy()
        same = left_color == right_color
        for local_c in range(w):
            distance = local_c if sample_c == 0 else w - 1 - local_c
            col = _mirrored_if_needed(pattern, (not same) and distance % 2 == 1)
            out[top : top + h, left + local_c] = col
        return

    raise AssertionError("each region must contain exactly one row-template or one column-template")


def main(grid: np.ndarray) -> np.ndarray:
    """Tile each bounded subregion from its edge template, alternating mirrors when opposite separators differ."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert np.any(g == BG)

    sep_rows = _separator_rows(g)
    sep_cols = _separator_cols(g)
    assert sep_rows and sep_cols, "expected horizontal and vertical separator lines"

    row_colors = {r: _mode_color(g[r, :]) for r in sep_rows}
    col_colors = {c: _mode_color(g[:, c]) for c in sep_cols}
    row_bands = _bands_from_separators(g.shape[0], sep_rows)
    col_bands = _bands_from_separators(g.shape[1], sep_cols)

    out = g.copy()
    for top_sep, r0, r1, bottom_sep in row_bands:
        for left_sep, c0, c1, right_sep in col_bands:
            region = g[r0 : r1 + 1, c0 : c1 + 1]
            _fill_region(
                out,
                region,
                row_colors[top_sep],
                row_colors[bottom_sep],
                col_colors[left_sep],
                col_colors[right_sep],
                r0,
                c0,
            )
    return out


def _separator_positions(sizes: list[int]) -> list[int]:
    positions = [0]
    cursor = 0
    for size in sizes:
        cursor += size + 1
        positions.append(cursor)
    return positions


def _sample_separator_colors(line_count: int) -> list[int]:
    a, b, c = random.sample(PALETTE, 3)
    templates = {
        3: ([a, a, b], [a, b, b], [a, c, c], [a, a, c]),
        4: ([a, a, b, b], [a, b, b, c], [a, a, c, b], [a, b, c, c]),
    }
    colors = list(random.choice(templates[line_count]))
    pairs = [colors[i] == colors[i + 1] for i in range(line_count - 1)]
    assert any(pairs) and not all(pairs)
    return colors


def _repeat_motif(motif: list[int], length: int) -> np.ndarray:
    reps = (length + len(motif) - 1) // len(motif)
    return np.asarray((motif * reps)[:length], dtype=int)


def _sample_pattern(length: int, color: int) -> np.ndarray:
    assert length >= MIN_REGION_SIZE
    motifs = []
    for motif_len in range(2, min(4, length) + 1):
        for bits in range(1, 1 << motif_len):
            motif = [color if (bits >> i) & 1 else BG for i in range(motif_len)]
            line = _repeat_motif(motif, length)
            if np.count_nonzero(line == color) < 2:
                continue
            if np.count_nonzero(line == BG) < 1:
                continue
            if np.array_equal(line, line[::-1]):
                continue
            motifs.append(line)
    assert motifs, "no usable pattern found"
    return random.choice(motifs).copy()


def _paint_region_input(
    grid: np.ndarray,
    r0: int,
    r1: int,
    c0: int,
    c1: int,
    orientation: str,
    side: str,
    color: int,
) -> None:
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if orientation == "row":
        pattern = _sample_pattern(w, color)
        rr = r0 if side == "top" else r1
        grid[rr, c0 : c1 + 1] = pattern
        return
    pattern = _sample_pattern(h, color)
    cc = c0 if side == "left" else c1
    grid[r0 : r1 + 1, cc] = pattern


def _adjacency_types(colors: list[int]) -> tuple[list[int], list[int]]:
    same = [i for i in range(len(colors) - 1) if colors[i] == colors[i + 1]]
    diff = [i for i in range(len(colors) - 1) if colors[i] != colors[i + 1]]
    assert same and diff
    return same, diff


def _choose_region_color(top: int, bottom: int, left: int, right: int) -> int:
    options = [c for c in PALETTE if c not in {top, bottom, left, right}]
    return random.choice(options)


def _sample_layout() -> tuple[list[int], list[int], list[int], list[int]]:
    row_regions = random.choice(REGION_COUNT_CHOICES)
    col_regions = random.choice(REGION_COUNT_CHOICES)
    row_sizes = [random.randint(MIN_REGION_SIZE, MAX_REGION_SIZE) for _ in range(row_regions)]
    col_sizes = [random.randint(MIN_REGION_SIZE, MAX_REGION_SIZE) for _ in range(col_regions)]
    row_colors = _sample_separator_colors(row_regions + 1)
    col_colors = _sample_separator_colors(col_regions + 1)
    return row_sizes, col_sizes, row_colors, col_colors


def _mandatory_assignments(
    row_same: list[int],
    row_diff: list[int],
    col_same: list[int],
    col_diff: list[int],
) -> dict[tuple[int, int], tuple[str, str]]:
    rs = random.choice(row_same)
    rd = random.choice(row_diff)
    cs = random.choice(col_same)
    cd = random.choice(col_diff)
    return {
        (rs, cs): ("row", random.choice(("top", "bottom"))),
        (rd, cd): ("row", random.choice(("top", "bottom"))),
        (rs, cd): ("col", random.choice(("left", "right"))),
        (rd, cs): ("col", random.choice(("left", "right"))),
    }


def _gen_attempt() -> np.ndarray | None:
    row_sizes, col_sizes, row_colors, col_colors = _sample_layout()
    sep_rows = _separator_positions(row_sizes)
    sep_cols = _separator_positions(col_sizes)
    height = sep_rows[-1] + 1
    width = sep_cols[-1] + 1

    grid = np.zeros((height, width), dtype=int)
    for idx, r in enumerate(sep_rows):
        grid[r, :] = int(row_colors[idx])
    for idx, c in enumerate(sep_cols):
        grid[:, c] = int(col_colors[idx])

    row_same, row_diff = _adjacency_types(row_colors)
    col_same, col_diff = _adjacency_types(col_colors)
    forced = _mandatory_assignments(row_same, row_diff, col_same, col_diff)

    for ri, (top_sep, r0, r1, bottom_sep) in enumerate(_bands_from_separators(height, sep_rows)):
        for ci, (left_sep, c0, c1, right_sep) in enumerate(_bands_from_separators(width, sep_cols)):
            if (ri, ci) in forced:
                orientation, side = forced[(ri, ci)]
            else:
                orientation = random.choice(("row", "col"))
                side = random.choice(("top", "bottom")) if orientation == "row" else random.choice(("left", "right"))
            color = _choose_region_color(
                row_colors[ri],
                row_colors[ri + 1],
                col_colors[ci],
                col_colors[ci + 1],
            )
            _paint_region_input(grid, r0, r1, c0, c1, orientation, side, color)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
