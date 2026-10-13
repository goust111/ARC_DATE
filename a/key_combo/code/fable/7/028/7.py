from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, components_mask, in_bounds, neighbors4


GLYPH_SIZE = 3
GLYPH_GAP = 1
BACKGROUND = Color.BLACK
START_COLOR = Color.GREY
PATH_COLORS = tuple(color for color in Color.NOT_BLACK if color != START_COLOR)
MAX_GENERATION_ATTEMPTS = 500

DIR_TO_DELTA = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}
DELTA_TO_DIR = {delta: name for name, delta in DIR_TO_DELTA.items()}

CODEBOOK = {
    ("U", 2): np.array([[1, 1, 1], [0, 1, 0], [0, 0, 0]], dtype=int),
    ("U", 4): np.array([[1, 1, 1], [1, 1, 1], [0, 1, 0]], dtype=int),
    ("D", 2): np.array([[0, 0, 0], [0, 1, 0], [1, 1, 1]], dtype=int),
    ("D", 4): np.array([[0, 1, 0], [1, 1, 1], [1, 1, 1]], dtype=int),
    ("L", 2): np.array([[1, 0, 0], [1, 1, 0], [1, 0, 0]], dtype=int),
    ("L", 4): np.array([[1, 1, 0], [1, 1, 1], [1, 1, 0]], dtype=int),
    ("R", 2): np.array([[0, 0, 1], [0, 1, 1], [0, 0, 1]], dtype=int),
    ("R", 4): np.array([[0, 1, 1], [1, 1, 1], [0, 1, 1]], dtype=int),
}
HORIZONTAL_CODES = [key for key in CODEBOOK if key[0] in {"L", "R"}]
VERTICAL_CODES = [key for key in CODEBOOK if key[0] in {"U", "D"}]


def _path_color(grid: np.ndarray, bg: int) -> int:
    colors = sorted(set(map(int, np.unique(grid))) - {int(bg), START_COLOR})
    assert len(colors) == 1, "expected exactly one non-background path color"
    return int(colors[0])


def _path_mask(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int], int]:
    g = np.asarray(grid, dtype=int)
    bg = BACKGROUND
    start_coords = np.argwhere(g == START_COLOR)
    assert start_coords.shape == (1, 2), "expected a unique gray start marker"
    start = tuple(map(int, start_coords[0]))
    path_color = _path_color(g, bg)
    mask = (g == path_color) | (g == START_COLOR)
    assert mask[start], "start must lie on the polyline"
    assert len(components_mask(mask, connectivity=4)) == 1, "polyline must be 4-connected"
    return mask, start, path_color


def _ordered_path(mask: np.ndarray, start: tuple[int, int]) -> list[tuple[int, int]]:
    coords = {tuple(map(int, rc)) for rc in np.argwhere(mask)}
    degrees = {}
    for r, c in coords:
        degree = sum((nr, nc) in coords for nr, nc in neighbors4(r, c))
        degrees[(r, c)] = int(degree)
    assert all(degree in {1, 2} for degree in degrees.values()), "not a simple polyline"
    endpoints = [coord for coord, degree in degrees.items() if degree == 1]
    assert len(endpoints) == 2 and start in endpoints, "gray marker must identify one endpoint"

    ordered = [start]
    prev = None
    cur = start
    while True:
        nxt = [
            (nr, nc)
            for nr, nc in neighbors4(*cur)
            if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and (nr, nc) != prev
        ]
        if not nxt:
            break
        assert len(nxt) == 1, "polyline branches are not allowed"
        prev, cur = cur, nxt[0]
        ordered.append(cur)

    assert len(ordered) == len(coords), "walk must visit the whole polyline exactly once"
    assert ordered[-1] != start and ordered[-1] in endpoints, "walk must terminate at the far endpoint"
    return ordered


def _segments_from_path(coords: list[tuple[int, int]]) -> list[tuple[str, int]]:
    assert len(coords) >= 2, "polyline must have positive length"
    deltas = [
        (int(r1 - r0), int(c1 - c0))
        for (r0, c0), (r1, c1) in zip(coords[:-1], coords[1:])
    ]
    cur_delta = deltas[0]
    cur_len = 2
    segments = []
    for delta in deltas[1:]:
        if delta == cur_delta:
            cur_len += 1
            continue
        segments.append((DELTA_TO_DIR[cur_delta], int(cur_len)))
        cur_delta = delta
        cur_len = 2
    segments.append((DELTA_TO_DIR[cur_delta], int(cur_len)))
    assert all(segment in CODEBOOK for segment in segments), "encountered segment outside the fixed codebook"
    return segments


def _render_program(segments: list[tuple[str, int]], color: int) -> np.ndarray:
    out = np.zeros((len(segments) * (GLYPH_SIZE + GLYPH_GAP) - GLYPH_GAP, GLYPH_SIZE), dtype=int)
    for i, segment in enumerate(segments):
        r0 = i * (GLYPH_SIZE + GLYPH_GAP)
        glyph = CODEBOOK[segment]
        out[r0 : r0 + GLYPH_SIZE, :][glyph.astype(bool)] = int(color)
    return out


def _cells_from_segments(segments: list[tuple[str, int]]) -> list[tuple[int, int]]:
    cells = [(0, 0)]
    r, c = 0, 0
    for direction, length in segments:
        dr, dc = DIR_TO_DELTA[direction]
        for _ in range(length - 1):
            r += dr
            c += dc
            cells.append((int(r), int(c)))
    return cells


def _simple_polyline(cells: list[tuple[int, int]]) -> bool:
    if len(cells) != len(set(cells)):
        return False
    coords = set(cells)
    degrees = {
        coord: sum(neigh in coords for neigh in neighbors4(*coord))
        for coord in coords
    }
    return sorted(degrees.values()) == [1, 1] + [2] * (len(coords) - 2)


def _sample_segments(rng: np.random.Generator) -> list[tuple[str, int]]:
    for _ in range(200):
        count = int(rng.integers(3, 8))
        horizontal_first = bool(rng.integers(0, 2))
        segments = []
        for i in range(count):
            pool = HORIZONTAL_CODES if (i % 2 == 0) == horizontal_first else VERTICAL_CODES
            segments.append(pool[int(rng.integers(0, len(pool)))])

        if len(set(segments)) < 2:
            continue
        lengths = {length for _direction, length in segments}
        if len(lengths) < 2 and rng.random() < 0.7:
            continue
        h_dirs = {direction for direction, _length in segments if direction in {"L", "R"}}
        v_dirs = {direction for direction, _length in segments if direction in {"U", "D"}}
        if len(h_dirs) == 1 and len(v_dirs) == 1 and rng.random() < 0.75:
            continue

        cells = _cells_from_segments(segments)
        if _simple_polyline(cells):
            return segments
    raise RuntimeError("failed to sample a simple segment sequence")


def _render_input(segments: list[tuple[str, int]], path_color: int, rng: np.random.Generator) -> np.ndarray:
    cells = _cells_from_segments(segments)
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    path_h = max_r - min_r + 1
    path_w = max_c - min_c + 1

    extra_h = int(rng.integers(0, min(9, 30 - path_h) + 1))
    extra_w = int(rng.integers(0, min(9, 30 - path_w) + 1))
    if extra_h == 0 and extra_w == 0:
        if path_h < 30:
            extra_h = 1
        elif path_w < 30:
            extra_w = 1
    H = int(path_h + extra_h)
    W = int(path_w + extra_w)
    top = int(rng.integers(0, H - path_h + 1))
    left = int(rng.integers(0, W - path_w + 1))
    shift_r = top - min_r
    shift_c = left - min_c

    grid = np.zeros((H, W), dtype=int)
    for r, c in cells[1:]:
        grid[r + shift_r, c + shift_c] = int(path_color)
    sr, sc = cells[0]
    grid[sr + shift_r, sc + shift_c] = START_COLOR
    return grid


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Decompile the marked polyline into a vertical stack of 3x3 glyph instructions.

    Input structure:
    - The grid has a black background, one simple single-color orthogonal polyline, and a unique gray pixel marking one endpoint as the start.
    - Every non-endpoint path cell has exactly two 4-neighbors on the path, and turns are the only valid segment boundaries.
    - Horizontal and vertical runs alternate, and each run's (direction, length) belongs to a fixed small codebook.

    Rule:
    - Start at the gray endpoint and walk along the polyline to the far endpoint.
    - Split the ordered walk at every turn to obtain the segment list, keeping each segment's direction and cell length.
    - Replace every segment by its fixed 3x3 glyph from the codebook.
    - Output a 3-column grid with the glyphs stacked top-to-bottom in path order, leaving one background row between adjacent glyphs.

    Key insight:
    - The input is the executed drawing, so the hidden instruction sequence is recovered by tracing from the marked endpoint and cutting only where the path changes direction.
    """
    mask, start, path_color = _path_mask(np.asarray(input_grid, dtype=int))
    ordered = _ordered_path(mask, start)
    segments = _segments_from_path(ordered)
    return _render_program(segments, path_color)


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        segments = _sample_segments(rng)
        path_color = int(PATH_COLORS[int(rng.integers(0, len(PATH_COLORS)))])
        grid = _render_input(segments, path_color, rng)
        out = main(grid)
        if grid.ndim != 2 or out.ndim != 2:
            continue
        if max(grid.shape) > 30 or max(out.shape) > 30:
            continue
        if out.shape[1] != GLYPH_SIZE:
            continue
        if out.shape[0] != len(segments) * (GLYPH_SIZE + GLYPH_GAP) - GLYPH_GAP:
            continue
        return grid
    raise RuntimeError("failed to generate a valid trajectory-decompilation input")
