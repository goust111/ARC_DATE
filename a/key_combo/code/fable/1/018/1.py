from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

try:
    _UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
    if str(_UTILS_DIR) not in sys.path:
        sys.path.append(str(_UTILS_DIR))
    from arc_utils import Color, in_bounds  # type: ignore
except Exception:
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        GREEN = 3
        YELLOW = 4
        GREY = 5
        PINK = 6
        ORANGE = 7
        TEAL = 8
        MAROON = 9

    def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
        height, width = map(int, shape)
        return 0 <= int(r) < height and 0 <= int(c) < width


_PALETTE = [
    Color.BLUE,
    Color.RED,
    Color.GREEN,
    Color.YELLOW,
    Color.GREY,
    Color.PINK,
    Color.ORANGE,
    Color.TEAL,
    Color.MAROON,
]

_MIN_SIDE = 7
_MAX_SIDE = 29
_MIN_SEGMENTS = 3
_MAX_SEGMENTS = 7
_MIN_SEGMENT_LEN = 2
_GEN_RETRY_LIMIT = 300


def _turn_clockwise(dr: int, dc: int) -> tuple[int, int]:
    return int(dc), int(-dr)


def _spiral_path(shape: tuple[int, int]) -> list[tuple[int, int]]:
    height, width = map(int, shape)
    path = [(0, 0)]
    used = {(0, 0)}
    r, c = 0, 0
    dr, dc = 0, 1

    while True:
        moved = False
        for _ in range(4):
            nr, nc = r + dr, c + dc
            if not in_bounds(shape, nr, nc):
                dr, dc = _turn_clockwise(dr, dc)
                continue
            if (nr, nc) in used:
                return path
            rr, cc = r + 2 * dr, c + 2 * dc
            if in_bounds(shape, rr, cc) and (rr, cc) in used:
                dr, dc = _turn_clockwise(dr, dc)
                continue

            r, c = nr, nc
            used.add((r, c))
            path.append((r, c))
            moved = True
            break

        if not moved:
            return path


def _path_mask(shape: tuple[int, int], path: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in path:
        mask[r, c] = True
    return mask


def _render_spiral_segments(grid: np.ndarray, path: list[tuple[int, int]]) -> np.ndarray:
    out = np.full(grid.shape, Color.BLACK, dtype=int)
    current = None
    for r, c in path:
        cell = int(grid[r, c])
        if cell != Color.BLACK:
            current = cell
        if current is None:
            raise AssertionError("the spiral start must provide the initial color")
        out[r, c] = current
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the hidden clockwise spiral and color each path segment by the most recent marker.

    Input structure:
    - The input is a black square grid with a sparse set of non-black singleton markers.
    - Every colored pixel lies on the same un-drawn clockwise spiral that starts at the top-left corner.
    - The top-left spiral cell is colored and supplies the initial segment color.

    Rule:
    - Rebuild the standard clockwise spiral for this canvas: start at (0,0) moving right, turn clockwise when the next step would leave the grid or when the cell two steps ahead is already on the spiral, and stop when the next step would revisit the spiral.
    - Traverse the spiral from its start while maintaining the current color.
    - Whenever the traversal reaches a colored input pixel, replace the current color with that pixel's color, then paint that spiral cell and all following cells with the current color until the next marker; keep non-spiral cells black.

    Key insight:
    - The scattered dots are not unordered noise: they are change-points along a single 1D spiral order, so solving the task means recovering that hidden order first and only then propagating colors segment by segment.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert g.shape[0] == g.shape[1], "expected a square grid"

    path = _spiral_path(g.shape)
    mask = _path_mask(g.shape, path)
    assert int(g[path[0]]) != Color.BLACK, "the spiral start must be marked"
    assert not np.any((g != Color.BLACK) & ~mask), "all markers must lie on the spiral"

    return _render_spiral_segments(g, path)


def _sample_segment_lengths(path_len: int, segment_count: int) -> list[int]:
    lengths = [_MIN_SEGMENT_LEN] * segment_count
    remaining = path_len - _MIN_SEGMENT_LEN * segment_count
    if remaining < 0:
        raise ValueError("path is too short for the requested number of segments")
    for _ in range(remaining):
        lengths[random.randrange(segment_count)] += 1
    return lengths


def _has_non_adjacent_repeat(colors: list[int]) -> bool:
    return any(color in colors[: i - 1] for i, color in enumerate(colors[2:], start=2))


def _sample_segment_colors(segment_count: int, *, force_repeat: bool) -> list[int]:
    palette_size = random.randint(2, min(4, segment_count))
    palette = random.sample(_PALETTE, palette_size)

    colors = [random.choice(palette)]
    for i in range(1, segment_count):
        candidates = [color for color in palette if color != colors[-1]]
        reuse = [color for color in dict.fromkeys(colors[:-1]) if color != colors[-1]]
        fresh = [color for color in candidates if color not in colors]

        if force_repeat and i >= 2 and reuse and (not _has_non_adjacent_repeat(colors) or random.random() < 0.7):
            colors.append(random.choice(reuse))
        elif fresh and random.random() < 0.45:
            colors.append(random.choice(fresh))
        else:
            colors.append(random.choice(candidates))

    if force_repeat and not _has_non_adjacent_repeat(colors):
        raise ValueError("failed to create a repeated color branch")
    if len(set(colors)) < 2:
        raise ValueError("need at least two visible segment colors")
    return colors


def _build_input(side: int, marker_indices: list[int], colors: list[int]) -> np.ndarray:
    grid = np.full((side, side), Color.BLACK, dtype=int)
    path = _spiral_path(grid.shape)
    for idx, color in zip(marker_indices, colors):
        r, c = path[idx]
        grid[r, c] = int(color)
    return grid


def _semantic_checks(inp: np.ndarray) -> bool:
    g = np.asarray(inp, dtype=int)
    if g.ndim != 2 or g.shape[0] != g.shape[1] or max(g.shape) > 30:
        return False

    path = _spiral_path(g.shape)
    order = {coord: idx for idx, coord in enumerate(path)}
    markers = [(int(r), int(c)) for r, c in np.argwhere(g != Color.BLACK)]
    if len(markers) < _MIN_SEGMENTS:
        return False
    if int(g[path[0]]) == Color.BLACK:
        return False
    if any(marker not in order for marker in markers):
        return False

    marker_indices = sorted(order[marker] for marker in markers)
    if marker_indices[0] != 0:
        return False
    if marker_indices[-1] < (2 * len(path)) // 3:
        return False

    marker_colors = [int(g[path[idx]]) for idx in marker_indices]
    if len(set(marker_colors)) < 2:
        return False

    path_mask = _path_mask(g.shape, path)
    if np.any(g[~path_mask] != Color.BLACK):
        return False

    out = main(g)
    if out.shape != g.shape:
        return False
    if np.any(out[~path_mask] != Color.BLACK):
        return False

    current = None
    for r, c in path:
        if int(g[r, c]) != Color.BLACK:
            current = int(g[r, c])
        if current is None or int(out[r, c]) != current:
            return False

    return True


def _generate_candidate() -> np.ndarray:
    side = random.randint(_MIN_SIDE, _MAX_SIDE)
    path = _spiral_path((side, side))
    max_segments = min(_MAX_SEGMENTS, len(path) // _MIN_SEGMENT_LEN)
    if max_segments < _MIN_SEGMENTS:
        raise ValueError("spiral too short")

    if max_segments >= 4 and random.random() < 0.75:
        segment_count = random.randint(4, max_segments)
    else:
        segment_count = random.randint(_MIN_SEGMENTS, min(4, max_segments))
    lengths = _sample_segment_lengths(len(path), segment_count)

    marker_indices = [0]
    acc = 0
    for length in lengths[:-1]:
        acc += length
        marker_indices.append(acc)

    force_repeat = segment_count >= 4 and random.random() < 0.7
    colors = _sample_segment_colors(segment_count, force_repeat=force_repeat)

    grid = _build_input(side, marker_indices, colors)
    if not _semantic_checks(grid):
        raise ValueError("candidate failed semantic validation")
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        try:
            return _generate_candidate()
        except ValueError:
            continue
    raise RuntimeError(f"generate_input: failed after {_GEN_RETRY_LIMIT} attempts")
