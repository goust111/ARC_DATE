import random
from collections import defaultdict

import numpy as np


BG = 0
COLOR_POOL = tuple(range(1, 10))

# Generator controls.
PAIR_COUNT_RANGE = (2, 4)
EXTRA_SEGMENT_RANGE = (1, 3)
SEGMENT_LENGTH_OPTIONS = (1, 3, 5)
WIDTH_RANGE = (9, 25)
BAND_HEIGHT_RANGE = (2, 6)
TOP_SPACER_ROWS = 1
BAND_GAP_ROWS = 1
BOTTOM_MARGIN_RANGE = (0, 2)
PAIR_MODES = ("vertical", "horizontal", "elbow_left", "elbow_right")
GEN_MAX_ATTEMPTS = 400


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _ruler_segments(ruler: np.ndarray) -> list[tuple[int, int, int, int]]:
    segments: list[tuple[int, int, int, int]] = []
    start = 0
    width = int(ruler.shape[0])
    while start < width:
        color = int(ruler[start])
        end = start
        while end + 1 < width and int(ruler[end + 1]) == color:
            end += 1
        center = (start + end) // 2
        segments.append((color, start, end, center))
        start = end + 1
    return segments


def _ruler_centers(ruler: np.ndarray) -> dict[int, int]:
    centers: dict[int, int] = {}
    for color, _start, _end, center in _ruler_segments(ruler):
        assert color not in centers, "each ruler color must occupy one contiguous segment"
        centers[color] = int(center)
    return centers


def _ordered_pair(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    return (a, b) if a <= b else (b, a)


def _path_between(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    start, end = _ordered_pair(a, b)
    r, c = start
    tr, tc = end
    path = [(r, c)]
    while (r, c) != (tr, tc):
        if r != tr:
            r += 1 if tr > r else -1
        elif c != tc:
            c += 1 if tc > c else -1
        path.append((r, c))
    return path


def _anchor_groups(grid: np.ndarray, bg: int) -> dict[int, list[tuple[int, int]]]:
    groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
    coords = np.argwhere(grid[1:, :] != bg)
    if coords.size == 0:
        return {}
    coords[:, 0] += 1
    for r, c in coords:
        groups[int(grid[r, c])].append((int(r), int(c)))
    for color in groups:
        groups[color].sort()
    return groups


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-color anchor pair with a ruler-driven cyclic color trail.

    Input structure:
    - The dominant color is the black background.
    - The top row is a full-width color ruler made of contiguous non-background color segments.
    - Below the ruler are isolated colored anchor pixels, and each anchor color appears exactly twice.

    Rule:
    - Pair anchors by color and process each pair from the anchor with the smaller row index to the other one.
    - Build the pair's path by repeatedly reducing the row difference first, then the column difference once the rows match.
    - Find the center column of that color's segment in the top ruler; the k-th path pixel takes the ruler color at `(center + k) mod width`.
    - Preserve the ruler and restore both anchor endpoints to their original anchor color.

    Key insight:
    - The top row is not just a legend of anchor colors: it is a cyclic palette strip that controls the full color sequence drawn along each connection.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height >= 3 and width >= 2, "grid is too small"

    bg = _background_color(g)
    ruler = g[0].copy()
    assert np.all(ruler != bg), "the top ruler must not contain background gaps"
    centers = _ruler_centers(ruler)

    out = g.copy()
    anchor_groups = _anchor_groups(g, bg)
    assert anchor_groups, "expected at least one anchor pair"
    items = sorted(anchor_groups.items(), key=lambda item: min(item[1]))
    for color, coords in items:
        assert len(coords) == 2, "each anchor color must occur exactly twice"
        assert color in centers, "every anchor color must appear in the ruler"
        path = _path_between(coords[0], coords[1])
        ref = centers[color]
        for k, (r, c) in enumerate(path):
            out[r, c] = int(ruler[(ref + k) % width])
        for r, c in coords:
            out[r, c] = int(color)

    return out


def _sample_segment_lengths(count: int) -> list[int] | None:
    for _ in range(100):
        lengths = [random.choice(SEGMENT_LENGTH_OPTIONS) for _ in range(count)]
        width = sum(lengths)
        if WIDTH_RANGE[0] <= width <= WIDTH_RANGE[1]:
            return lengths
    return None


def _build_ruler(colors: list[int], lengths: list[int]) -> tuple[np.ndarray, dict[int, int]]:
    width = sum(lengths)
    ruler = np.empty(width, dtype=int)
    centers: dict[int, int] = {}
    c = 0
    for color, length in zip(colors, lengths):
        ruler[c : c + length] = int(color)
        centers[int(color)] = c + length // 2
        c += length
    return ruler, centers


def _sample_band_heights(pair_count: int) -> tuple[list[int], int] | None:
    for _ in range(100):
        heights = [random.randint(*BAND_HEIGHT_RANGE) for _ in range(pair_count)]
        bottom_margin = random.randint(*BOTTOM_MARGIN_RANGE)
        height = (
            1
            + TOP_SPACER_ROWS
            + sum(heights)
            + BAND_GAP_ROWS * (pair_count - 1)
            + bottom_margin
        )
        if height <= 30:
            return heights, height
    return None


def _band_ranges(heights: list[int]) -> list[tuple[int, int]]:
    rows: list[tuple[int, int]] = []
    r = 1 + TOP_SPACER_ROWS
    for height in heights:
        rows.append((r, r + height - 1))
        r += height + BAND_GAP_ROWS
    return rows


def _sample_modes(pair_count: int, want_wrap: bool) -> list[str]:
    modes = random.sample(PAIR_MODES, pair_count)
    if want_wrap and not any(mode.startswith("elbow") for mode in modes):
        replace_index = random.randrange(pair_count)
        modes[replace_index] = random.choice(("elbow_left", "elbow_right"))
    random.shuffle(modes)
    return modes


def _sample_pair_in_band(
    row_lo: int,
    row_hi: int,
    width: int,
    mode: str,
    need_wrap: bool,
) -> tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]]] | None:
    assert row_lo <= row_hi
    for _ in range(200):
        if mode == "vertical":
            if row_hi - row_lo < 2:
                return None
            col = random.randrange(width)
            start_r = random.randint(row_lo, row_hi - 2)
            end_r = random.randint(start_r + 2, row_hi)
            a = (start_r, col)
            b = (end_r, col)
        elif mode == "horizontal":
            if width < 4:
                return None
            row = random.randint(row_lo, row_hi)
            c0 = random.randint(0, width - 4)
            c1 = random.randint(c0 + 3, width - 1)
            a = (row, c0)
            b = (row, c1)
        elif mode == "elbow_left":
            if row_hi == row_lo or width < 3:
                return None
            start_r = random.randint(row_lo, row_hi - 1)
            end_r = random.randint(start_r + 1, row_hi)
            if need_wrap:
                start_c, end_c = width - 1, 0
            else:
                end_c = random.randint(0, width - 3)
                start_c = random.randint(end_c + 2, width - 1)
            a = (start_r, start_c)
            b = (end_r, end_c)
        else:
            assert mode == "elbow_right"
            if row_hi == row_lo or width < 3:
                return None
            start_r = random.randint(row_lo, row_hi - 1)
            end_r = random.randint(start_r + 1, row_hi)
            if need_wrap:
                start_c, end_c = 0, width - 1
            else:
                start_c = random.randint(0, width - 3)
                end_c = random.randint(start_c + 2, width - 1)
            a = (start_r, start_c)
            b = (end_r, end_c)

        path = _path_between(a, b)
        if len(path) < 4:
            continue
        if need_wrap and len(path) <= width:
            continue
        return a, b, path
    return None


def _render_input(
    height: int,
    ruler: np.ndarray,
    placements: dict[int, tuple[tuple[int, int], tuple[int, int]]],
) -> np.ndarray:
    grid = np.full((height, ruler.shape[0]), BG, dtype=int)
    grid[0] = ruler
    for color, (a, b) in placements.items():
        grid[a] = int(color)
        grid[b] = int(color)
    return grid


def _generate_placements(
    band_rows: list[tuple[int, int]],
    width: int,
    pair_colors: list[int],
    modes: list[str],
    want_wrap: bool,
) -> dict[int, tuple[tuple[int, int], tuple[int, int]]] | None:
    placements: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {}
    wrap_index = -1
    if want_wrap:
        elbow_indices = [index for index, mode in enumerate(modes) if mode.startswith("elbow")]
        if not elbow_indices:
            return None
        wrap_index = random.choice(elbow_indices)
    for index, (color, mode, (row_lo, row_hi)) in enumerate(zip(pair_colors, modes, band_rows)):
        sampled = _sample_pair_in_band(
            row_lo=row_lo,
            row_hi=row_hi,
            width=width,
            mode=mode,
            need_wrap=index == wrap_index,
        )
        if sampled is None:
            return None
        a, b, _path = sampled
        placements[int(color)] = (a, b)
    return placements


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    extra_segments = random.randint(*EXTRA_SEGMENT_RANGE)
    segment_count = pair_count + extra_segments
    if segment_count > len(COLOR_POOL):
        return None

    band_sample = _sample_band_heights(pair_count)
    if band_sample is None:
        return None
    band_heights, height = band_sample
    band_rows = _band_ranges(band_heights)

    lengths = _sample_segment_lengths(segment_count)
    if lengths is None:
        return None
    width = sum(lengths)

    colors = random.sample(list(COLOR_POOL), segment_count)
    ruler, _centers = _build_ruler(colors, lengths)

    pair_colors = random.sample(colors, pair_count)
    want_wrap = random.random() < 0.5
    modes = _sample_modes(pair_count, want_wrap)
    placements = _generate_placements(band_rows, width, pair_colors, modes, want_wrap)
    if placements is None:
        return None

    inp = _render_input(height, ruler, placements)
    try:
        out = main(inp)
    except Exception:
        return None
    if out.shape != inp.shape:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        sample = _gen_attempt()
        if sample is not None:
            return sample
    raise RuntimeError("failed to generate a valid ARC input")
