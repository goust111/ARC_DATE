from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import find_components, get_bg, neighbors4


Coord = tuple[int, int]

MAX_ATTEMPTS = 600
WIDTH_CHOICES = [6, 7, 7, 8, 8, 9]
HEIGHT_RANGE = (8, 18)


def _read_palette_sequence(top_row: np.ndarray, bg: int) -> tuple[list[int], int]:
    gap_cols = np.flatnonzero(top_row == bg)
    assert len(gap_cols) == 1, "expected exactly one palette gap"
    gap = int(gap_cols[0])
    seq = [int(top_row[(gap + offset) % top_row.size]) for offset in range(1, top_row.size)]
    assert all(color != bg for color in seq), "palette may contain only one background gap"
    assert len(set(seq)) >= 2, "palette must be multicolor"
    return seq, gap


def _ordered_path(coords: np.ndarray) -> tuple[list[Coord], tuple[Coord, Coord]]:
    cells = {tuple(map(int, coord)) for coord in coords}
    assert len(cells) == len(coords), "path coordinates must be unique"

    degrees = {
        cell: sum((nbr in cells) for nbr in neighbors4(*cell))
        for cell in cells
    }
    endpoints = sorted(cell for cell, degree in degrees.items() if degree == 1)
    assert len(endpoints) == 2, "path must have exactly two endpoints"
    assert all(degree in (1, 2) for degree in degrees.values()), "path must not branch"
    assert endpoints[0][0] != endpoints[1][0], "one endpoint must be closer to the palette row"

    start = min(endpoints, key=lambda rc: (rc[0], rc[1]))
    path: list[Coord] = []
    prev: Coord | None = None
    cur = start
    while True:
        path.append(cur)
        next_cells = [
            nbr
            for nbr in neighbors4(*cur)
            if nbr in cells and nbr != prev
        ]
        if not next_cells:
            break
        assert len(next_cells) == 1, "path traversal must be unique"
        prev, cur = cur, next_cells[0]

    assert len(path) == len(cells), "path traversal must visit every path cell"
    return path, (endpoints[0], endpoints[1])


def _extract_path(grid: np.ndarray, bg: int) -> tuple[list[Coord], int]:
    body = grid[1:]
    path_colors = sorted(set(map(int, body.flatten())) - {int(bg)})
    assert len(path_colors) == 1, "expected one monochrome path below the palette row"
    components = find_components(body, bg=bg, connectivity=4, monochromatic=False)
    assert len(components) == 1, "expected exactly one lower path component"
    coords = components[0]["coords"] + np.array([1, 0])
    path, _ = _ordered_path(coords)
    return path, path_colors[0]


def _write_colors(path: list[Coord], sequence: list[int]) -> list[int]:
    return [int(sequence[i % len(sequence)]) for i in range(len(path))]


def _render_output(
    grid: np.ndarray,
    path: list[Coord],
    sequence: list[int],
    *,
    reverse_path: bool = False,
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    bg = int(get_bg(out))
    out[0, :] = bg
    ordered_path = list(reversed(path)) if reverse_path else list(path)
    for (r, c), color in zip(ordered_path, _write_colors(ordered_path, sequence)):
        out[r, c] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Write the gap-phased top-row palette cyclically onto the lower one-pixel path, then erase the palette.

    Input structure:
    - The top row is a multicolor palette row with exactly one background-color gap.
    - All non-background cells below the top row form one monochrome 4-connected path.
    - That path is one pixel wide, has no branches, and has exactly two endpoints.

    Rule:
    - Read the palette sequence starting from the cell immediately right of the gap, wrapping around the row and skipping the gap.
    - Choose the endpoint with the smaller row index as the path start, then walk the path cell by cell to the other endpoint.
    - Recolor successive path cells with the palette sequence in order, cycling back to the start of the sequence when needed.
    - Replace the entire top row by the background color and leave every other background cell unchanged.

    Key insight:
    - The gap is a phase marker, not damage: it determines where the cyclic write sequence starts before that sequence is streamed onto the path.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2 and g.shape[1] >= 3
    bg = int(get_bg(g))
    sequence, _gap = _read_palette_sequence(g[0], bg)
    path, _path_color = _extract_path(g, bg)
    return _render_output(g, path, sequence)


def _build_spaced_snake(start_r: int, c0: int, width: int, lanes: int) -> list[Coord]:
    path: list[Coord] = []
    c1 = c0 + width - 1
    for lane in range(lanes):
        row = start_r + 2 * lane
        cols = range(c0, c1 + 1) if lane % 2 == 0 else range(c1, c0 - 1, -1)
        path.extend((row, c) for c in cols)
        if lane < lanes - 1:
            path.append((row + 1, c1 if lane % 2 == 0 else c0))
    return path


def _place_palette(top_row: np.ndarray, sequence: list[int], gap: int, bg: int) -> None:
    top_row[:] = bg
    for i, color in enumerate(sequence):
        top_row[(gap + 1 + i) % top_row.size] = int(color)


def _make_grid(
    height: int,
    width: int,
    bg: int,
    gap: int,
    palette_sequence: list[int],
    path_color: int,
    path: list[Coord],
) -> np.ndarray:
    grid = np.full((height, width), bg, dtype=int)
    _place_palette(grid[0], palette_sequence, gap, bg)
    for r, c in path:
        grid[r, c] = int(path_color)
    return grid


def _wrong_row_start_sequence(top_row: np.ndarray, bg: int) -> list[int]:
    return [int(color) for color in top_row if int(color) != int(bg)]


def _wrong_suffix_only_sequence(top_row: np.ndarray, gap: int) -> list[int]:
    return [int(top_row[c]) for c in range(gap + 1, top_row.size)]


def _check_semantics(grid: np.ndarray) -> bool:
    try:
        g = np.asarray(grid, dtype=int)
        bg = int(get_bg(g))
        sequence, gap = _read_palette_sequence(g[0], bg)
        path, path_color = _extract_path(g, bg)
        assert all(int(g[r, c]) == path_color for r, c in path)

        correct = _render_output(g, path, sequence)
        assert np.array_equal(main(g), correct)
        assert np.all(correct[0] == bg)
        assert len(set(int(correct[r, c]) for r, c in path)) >= 2

        wrong_phase = _render_output(g, path, _wrong_row_start_sequence(g[0], bg))
        assert not np.array_equal(correct, wrong_phase), "gap phase must matter"

        suffix_only = _wrong_suffix_only_sequence(g[0], gap)
        assert suffix_only
        wrong_wrap = _render_output(g, path, suffix_only)
        assert not np.array_equal(correct, wrong_wrap), "wraparound must matter"

        reverse = _render_output(g, path, sequence, reverse_path=True)
        assert not np.array_equal(correct, reverse), "start endpoint choice must matter"

        bg_mask = np.ones_like(g, dtype=bool)
        bg_mask[0, :] = False
        for r, c in path:
            bg_mask[r, c] = False
        assert np.array_equal(correct[bg_mask], g[bg_mask]), "irrelevant background must stay unchanged"
        return True
    except AssertionError:
        return False


def _candidate_paths(height: int, width: int, gap: int, branch: str) -> list[tuple[list[Coord], int, int]]:
    seq_len = width - 1
    suffix_len = width - 1 - gap
    max_path_len = (height * width - seq_len - 1) // 2
    candidates: list[tuple[list[Coord], int, int]] = []

    for start_r in range(1, height - 2):
        max_lanes = 1 + (height - 1 - start_r) // 2
        for lane_width in range(3, width + 1):
            for c0 in range(0, width - lane_width + 1):
                for lanes in range(2, max_lanes + 1):
                    full_path = _build_spaced_snake(start_r, c0, lane_width, lanes)
                    full_len = len(full_path)
                    lower = max(lane_width + 1, suffix_len + 1)
                    upper = min(full_len, max_path_len)
                    if branch == "wrap_only":
                        upper = min(upper, seq_len)
                    else:
                        lower = max(lower, seq_len + 1)
                    if lower <= upper:
                        candidates.append((full_path, lower, upper))
    return candidates


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*HEIGHT_RANGE)
    width = random.choice(WIDTH_CHOICES)
    bg = random.randrange(10)
    gap = random.randint(1, width - 2)

    branch_order = ["cycle", "wrap_only"]
    if random.random() < 0.45:
        branch_order.reverse()

    candidates: list[tuple[str, list[Coord], int, int]] = []
    for branch in branch_order:
        for full_path, lower, upper in _candidate_paths(height, width, gap, branch):
            candidates.append((branch, full_path, lower, upper))
        if candidates:
            break
    if not candidates:
        return None

    branch, full_path, lower, upper = random.choice(candidates)
    path_len = random.randint(lower, upper)
    path = full_path[:path_len]

    colors = list(range(10))
    random.shuffle(colors)
    colors.remove(bg)
    path_color = colors[0]
    palette_sequence = colors[1:width]
    if len(palette_sequence) != width - 1:
        return None
    random.shuffle(palette_sequence)

    grid = _make_grid(height, width, bg, gap, palette_sequence, path_color, path)
    if branch == "cycle" and len(path) <= len(palette_sequence):
        return None
    if branch == "wrap_only":
        suffix_len = width - 1 - gap
        if not (suffix_len < len(path) <= len(palette_sequence)):
            return None
    if not _check_semantics(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid palette-phase ARC input")
