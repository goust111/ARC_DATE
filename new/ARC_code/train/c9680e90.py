from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg, neighbors4
import random


Coord = tuple[int, int]


def _separator_row(g: np.ndarray, bg: int) -> int:
    rows = [r for r in range(g.shape[0]) if np.all(g[r] == g[r, 0]) and int(g[r, 0]) != bg]
    assert len(rows) == 1, "expected exactly one solid non-background separator row"
    sep = rows[0]
    assert g.shape[0] == 2 * sep + 1, "separator must split equal-height top and bottom halves"
    return sep


def _coords_in_bottom(g: np.ndarray, sep: int, color: int) -> list[Coord]:
    return [(int(r + sep + 1), int(c)) for r, c in np.argwhere(g[sep + 1 :] == color)]


def _trail_endpoint(g: np.ndarray, sep: int, trail_color: int, start: Coord) -> tuple[Coord, set[Coord]]:
    H, W = g.shape
    adjacent = [
        (nr, nc)
        for nr, nc in neighbors4(*start)
        if sep < nr < H and 0 <= nc < W and int(g[nr, nc]) == trail_color
    ]
    assert len(adjacent) == 1, "each source pixel must touch exactly one trail"

    prev = start
    cur = adjacent[0]
    trail = {cur}
    while True:
        nxt = [
            (nr, nc)
            for nr, nc in neighbors4(*cur)
            if (nr, nc) != prev and sep < nr < H and 0 <= nc < W and int(g[nr, nc]) == trail_color
        ]
        assert len(nxt) <= 1, "trails must be one-cell-wide unbranched paths"
        if not nxt:
            return cur, trail

        prev, cur = cur, nxt[0]
        assert cur not in trail, "trail cannot loop"
        trail.add(cur)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each mirrored bottom source pixel to the far end of its trail and redraw the top half as its reflection.

    Input structure
    - A solid non-background row separates equal-height top and bottom halves.
    - The top half has one marker color on the background.
    - In the bottom half, exactly one color has positions that mirror the top markers across the separator, and the other color forms trails touching those source pixels.

    Rule
    - Identify the bottom source color by reflecting each bottom color into the top half and matching the top marker set.
    - For every bottom source pixel, follow its unique 4-connected trail of the other bottom color to the trail endpoint.
    - Clear the old top markers, bottom sources, and trails; place the source color at each endpoint and place the top marker color at the reflected endpoint.

    Key insight
    - The top half is only a mirror display of the bottom moving sources; the actual motion is encoded by the trail attached to each bottom source pixel.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    sep = _separator_row(g, bg)

    top_colors = colors_present(g[:sep], exclude={bg})
    assert len(top_colors) == 1, "expected one top marker color"
    top_color = top_colors[0]
    top_coords = {tuple(map(int, rc)) for rc in np.argwhere(g[:sep] == top_color)}
    assert top_coords, "expected at least one top marker"

    bottom_colors = colors_present(g[sep + 1 :], exclude={bg})
    assert len(bottom_colors) == 2, "expected bottom source and trail colors"
    assert top_color not in bottom_colors, "top marker color must be distinct from bottom colors"

    source_candidates = []
    for color in bottom_colors:
        mirrored = {(2 * sep - r, c) for r, c in _coords_in_bottom(g, sep, color)}
        if mirrored == top_coords:
            source_candidates.append(color)
    assert len(source_candidates) == 1, "exactly one bottom color must mirror the top markers"
    source_color = source_candidates[0]
    trail_color = next(color for color in bottom_colors if color != source_color)

    source_coords = _coords_in_bottom(g, sep, source_color)
    all_trail = set(_coords_in_bottom(g, sep, trail_color))
    endpoints: list[Coord] = []
    used_trail: set[Coord] = set()
    for source in source_coords:
        endpoint, trail = _trail_endpoint(g, sep, trail_color, source)
        assert not (used_trail & trail), "trails cannot be shared by source pixels"
        used_trail |= trail
        endpoints.append(endpoint)

    assert used_trail == all_trail, "every trail pixel must belong to a source path"
    assert len(set(endpoints)) == len(source_coords), "source paths cannot collide"

    out = g.copy()
    clear = np.zeros(g.shape, dtype=bool)
    clear[:sep] = g[:sep] == top_color
    clear[sep + 1 :] |= g[sep + 1 :] == source_color
    clear[sep + 1 :] |= g[sep + 1 :] == trail_color
    out[clear] = bg

    for r, c in endpoints:
        out[r, c] = source_color
        out[2 * sep - r, c] = top_color

    return out

# --------------- generator (att2) ---------------

# Fixed task palette.
_GEN_BG = 7
_GEN_SEP = 9
_GEN_TOP = 5
_GEN_SOURCE = 2
_GEN_TRAIL = 6

# Canvas and object-count bounds.
_GEN_GRID_SIZES = (7, 9, 11)
_GEN_MIN_SOURCES = 2
_GEN_MAX_SOURCES = 5
_GEN_MAX_TRAIL_LEN = 4

# Generation retry limits.
_GEN_MAX_ATTEMPTS = 300

# Straight-path directions.
_GEN_DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _gen_neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _gen_in_bottom(r: int, c: int, n: int, sep: int) -> bool:
    return sep < r < n and 0 <= c < n


def _gen_max_trail_total(n: int, k: int) -> int:
    bg_gt_trail = (n * n - n - 2 * k - 1) // 2
    bg_gt_separator = n * n - 2 * n - 2 * k - 1
    return min(bg_gt_trail, bg_gt_separator)


def _gen_segment_options(
    n: int,
    sep: int,
    sources: set[tuple[int, int]],
    trails: set[tuple[int, int]],
    max_len: int,
) -> list[tuple[tuple[int, int], tuple[tuple[int, int], ...], int]]:
    opts: list[tuple[tuple[int, int], tuple[tuple[int, int], ...], int]] = []
    occupied = sources | trails
    for end_r in range(sep + 1, n):
        for end_c in range(n):
            if (end_r, end_c) in occupied:
                continue
            for dr, dc in _GEN_DIRECTIONS:
                for length in range(1, min(_GEN_MAX_TRAIL_LEN, max_len) + 1):
                    src = (end_r - dr * length, end_c - dc * length)
                    if not _gen_in_bottom(src[0], src[1], n, sep):
                        continue
                    trail = tuple((src[0] + dr * i, src[1] + dc * i) for i in range(1, length + 1))
                    if any(not _gen_in_bottom(r, c, n, sep) for r, c in trail):
                        continue
                    if src in occupied or any(cell in occupied for cell in trail):
                        continue
                    trail_set = set(trail)
                    if len(trail_set) != length:
                        continue
                    if sum(nb in trail_set for nb in _gen_neighbors4(*src)) != 1:
                        continue

                    touches_existing = False
                    for cell in trail:
                        for nb in _gen_neighbors4(*cell):
                            if nb in trails or nb in sources:
                                touches_existing = True
                                break
                        if touches_existing:
                            break
                    if touches_existing:
                        continue
                    if any(nb in trails for nb in _gen_neighbors4(*src)):
                        continue
                    opts.append((src, trail, length))
    return opts


def _gen_choose_segment(
    n: int,
    sep: int,
    sources: set[tuple[int, int]],
    trails: set[tuple[int, int]],
    max_len: int,
) -> tuple[tuple[int, int], tuple[tuple[int, int], ...], int] | None:
    opts = _gen_segment_options(n, sep, sources, trails, max_len)
    if not opts:
        return None
    lengths = sorted({length for _, _, length in opts})
    length = random.choice(lengths)
    same_length = [opt for opt in opts if opt[2] == length]
    return random.choice(same_length)


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(_GEN_GRID_SIZES)
    sep = (n - 1) // 2
    k = random.randint(_GEN_MIN_SOURCES, _GEN_MAX_SOURCES)
    max_trail_total = _gen_max_trail_total(n, k)
    if max_trail_total < k:
        return None

    sources: set[tuple[int, int]] = set()
    trails: set[tuple[int, int]] = set()
    paths: list[tuple[tuple[int, int], tuple[tuple[int, int], ...]]] = []
    trail_total = 0

    for i in range(k):
        remaining_paths = k - i - 1
        max_len = max_trail_total - trail_total - remaining_paths
        if max_len < 1:
            return None
        segment = _gen_choose_segment(n, sep, sources, trails, max_len)
        if segment is None:
            return None
        src, trail, length = segment
        sources.add(src)
        trails.update(trail)
        paths.append((src, trail))
        trail_total += length

    out = np.full((n, n), _GEN_BG, dtype=int)
    out[sep, :] = _GEN_SEP
    inp = out.copy()

    for src, trail in paths:
        end = trail[-1]
        inp[src] = _GEN_SOURCE
        for cell in trail:
            inp[cell] = _GEN_TRAIL
        inp[2 * sep - src[0], src[1]] = _GEN_TOP
        out[end] = _GEN_SOURCE
        out[2 * sep - end[0], end[1]] = _GEN_TOP

    counts = np.bincount(inp.ravel(), minlength=10)
    if int(np.argmax(counts)) != _GEN_BG:
        return None
    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid c9680e90 input")
