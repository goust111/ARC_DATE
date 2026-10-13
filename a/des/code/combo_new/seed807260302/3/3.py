from __future__ import annotations

import numpy as np


BG = 0
DIRS4 = ((-1, 0), (0, 1), (1, 0), (0, -1))
_RNG = np.random.default_rng()


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _in_bounds(r: int, c: int, h: int, w: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r0, c0 in np.argwhere(mask):
        r0, c0 = int(r0), int(c0)
        if seen[r0, c0]:
            continue
        seen[r0, c0] = True
        stack = [(r0, c0)]
        comp: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            comp.append((r, c))
            for rr, cc in _neighbors4(r, c):
                if _in_bounds(rr, cc, h, w) and bool(mask[rr, cc]) and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        comps.append(comp)
    return comps


def _path_endpoints(cells: set[tuple[int, int]]) -> list[tuple[int, int]] | None:
    if len(cells) < 3:
        return None
    degrees: dict[tuple[int, int], int] = {}
    for r, c in cells:
        degrees[(r, c)] = sum((rr, cc) in cells for rr, cc in _neighbors4(r, c))
    endpoints = [pt for pt, degree in degrees.items() if degree == 1]
    if len(endpoints) != 2:
        return None
    if any(degree not in (1, 2) for degree in degrees.values()):
        return None
    return endpoints


def _trace_path(
    cells: set[tuple[int, int]], start: tuple[int, int], end: tuple[int, int]
) -> list[tuple[int, int]]:
    path = [start]
    prev: tuple[int, int] | None = None
    cur = start
    while cur != end:
        choices = [(rr, cc) for rr, cc in _neighbors4(*cur) if (rr, cc) in cells and (rr, cc) != prev]
        assert len(choices) == 1, "pipe path must be unbranched"
        nxt = choices[0]
        path.append(nxt)
        prev, cur = cur, nxt
        assert len(path) <= len(cells), "pipe path must not loop"
    assert len(path) == len(cells), "pipe path trace must cover the whole component"
    return path


def _find_pipes(g: np.ndarray) -> list[dict]:
    pipes: list[dict] = []
    for color in sorted(int(c) for c in np.unique(g) if int(c) != BG):
        for comp in _components(g == color):
            cells = set(comp)
            endpoints = _path_endpoints(cells)
            if endpoints is None:
                continue
            try:
                _trace_path(cells, endpoints[0], endpoints[1])
            except AssertionError:
                continue
            pipes.append({"color": color, "cells": cells, "endpoints": endpoints})
    return pipes


def _window(center: tuple[int, int]) -> list[tuple[int, int, int, int]]:
    cr, cc = center
    return [(cr + dr, cc + dc, dr + 1, dc + 1) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]


def _signal_at(
    g: np.ndarray,
    endpoint: tuple[int, int],
    inward_dir: tuple[int, int],
    pipe_mask: np.ndarray,
) -> tuple[tuple[int, int], np.ndarray] | None:
    h, w = g.shape
    center = (endpoint[0] - inward_dir[0], endpoint[1] - inward_dir[1])
    cr, cc = center
    if not (1 <= cr < h - 1 and 1 <= cc < w - 1):
        return None
    if int(g[cr, cc]) == BG or bool(pipe_mask[cr, cc]):
        return None

    patch = np.full((3, 3), BG, dtype=int)
    found = False
    for rr, cc, pr, pc in _window(center):
        if bool(pipe_mask[rr, cc]):
            continue
        value = int(g[rr, cc])
        if value != BG:
            patch[pr, pc] = value
            found = True
    if not found:
        return None
    return center, patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each small endpoint signal through its colored pipe and rotate it by the pipe's turn count.

    Input structure:
    - A black background contains one or more one-cell-wide colored orthogonal pipe paths.
    - Each pipe is a single-color 4-connected path with two degree-one endpoints.
    - Exactly one endpoint has a non-pipe 3x3 asymmetric signal centered just outside the pipe; the other endpoint has empty space outside it.

    Rule:
    - Identify the two endpoints of every pipe and use the endpoint with the adjacent 3x3 signal as the entrance.
    - Trace the pipe from entrance to outlet and count the number of 90-degree turns along the path.
    - Clear the entrance signal, rotate that 3x3 signal clockwise by one quarter-turn per pipe turn, and place it just beyond the outlet in the outlet direction.
    - Preserve all pipe cells and the original grid size.

    Key insight:
    - The signal's final orientation is not marked directly; it is determined by counting turns while following the pipe from entrance to outlet.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h <= 30 and w <= 30

    pipes = _find_pipes(g)
    assert pipes, "expected at least one pipe"

    pipe_mask = np.zeros((h, w), dtype=bool)
    for pipe in pipes:
        for r, c in pipe["cells"]:
            pipe_mask[r, c] = True

    moves = []
    for pipe in pipes:
        cells = pipe["cells"]
        candidates = []
        for endpoint in pipe["endpoints"]:
            pipe_neighbors = [(rr, cc) for rr, cc in _neighbors4(*endpoint) if (rr, cc) in cells]
            assert len(pipe_neighbors) == 1
            neighbor = pipe_neighbors[0]
            inward = (neighbor[0] - endpoint[0], neighbor[1] - endpoint[1])
            signal = _signal_at(g, endpoint, inward, pipe_mask)
            if signal is not None:
                candidates.append((endpoint, inward, signal[0], signal[1]))

        assert len(candidates) == 1, "each pipe must have exactly one entrance signal"
        entry, _entry_inward, signal_center, patch = candidates[0]
        exit_endpoint = pipe["endpoints"][0] if pipe["endpoints"][1] == entry else pipe["endpoints"][1]
        path = _trace_path(cells, entry, exit_endpoint)
        step_dirs = [
            (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
            for i in range(len(path) - 1)
        ]
        turns = sum(step_dirs[i] != step_dirs[i - 1] for i in range(1, len(step_dirs)))
        last_dir = step_dirs[-1]
        out_center = (exit_endpoint[0] + last_dir[0], exit_endpoint[1] + last_dir[1])
        assert 1 <= out_center[0] < h - 1 and 1 <= out_center[1] < w - 1
        rotated = np.rot90(patch, -(turns % 4))
        moves.append({"source_center": signal_center, "source": patch, "target_center": out_center, "target": rotated})

    out = g.copy()
    for move in moves:
        for rr, cc, pr, pc in _window(move["source_center"]):
            if int(move["source"][pr, pc]) != BG:
                assert not bool(pipe_mask[rr, cc])
                out[rr, cc] = BG

    for move in moves:
        for rr, cc, pr, pc in _window(move["target_center"]):
            value = int(move["target"][pr, pc])
            if value == BG:
                continue
            assert not bool(pipe_mask[rr, cc]), "signal may not overwrite pipe cells"
            assert int(out[rr, cc]) == BG, "outlet signal locations must be empty"
            out[rr, cc] = value
    return out


def _turn_dir(direction: tuple[int, int], side: int) -> tuple[int, int]:
    dr, dc = direction
    if side < 0:
        return -dc, dr
    return dc, -dr


def _clean_path(cells: list[tuple[int, int]]) -> bool:
    cell_set = set(cells)
    if len(cell_set) != len(cells):
        return False
    endpoints = _path_endpoints(cell_set)
    if endpoints is None:
        return False
    try:
        _trace_path(cell_set, endpoints[0], endpoints[1])
    except AssertionError:
        return False
    return True


def _make_path(
    h: int,
    w: int,
    start: tuple[int, int],
    dirs: list[tuple[int, int]],
    lengths: list[int],
) -> list[tuple[int, int]] | None:
    r, c = start
    cells = [(r, c)]
    for direction, length in zip(dirs, lengths):
        dr, dc = direction
        for _ in range(length):
            r += dr
            c += dc
            if not (2 <= r <= h - 3 and 2 <= c <= w - 3):
                return None
            cells.append((r, c))
    if not _clean_path(cells):
        return None
    return cells


def _patch_cells(center: tuple[int, int], mask: np.ndarray) -> set[tuple[int, int]]:
    cr, cc = center
    cells = set()
    for pr in range(3):
        for pc in range(3):
            if bool(mask[pr, pc]):
                cells.add((cr + pr - 1, cc + pc - 1))
    return cells


def _window_set(center: tuple[int, int]) -> set[tuple[int, int]]:
    return {(r, c) for r, c, _, _ in _window(center)}


def _padded(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    result = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if _in_bounds(rr, cc, h, w):
                    result.add((rr, cc))
    return result


SIGNAL_MASKS = (
    np.array([[1, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=bool),
    np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1]], dtype=bool),
    np.array([[1, 0, 0], [0, 1, 0], [1, 0, 1]], dtype=bool),
    np.array([[0, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
)


def _random_pipe(
    h: int,
    w: int,
    forbidden: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], tuple[int, int], tuple[int, int], np.ndarray, set[tuple[int, int]]] | None:
    rotation_branch = int(_RNG.integers(0, 4))
    turns = rotation_branch
    if rotation_branch == 0 and float(_RNG.random()) < 0.45:
        turns = 4

    for _ in range(300):
        first_dir = DIRS4[int(_RNG.integers(0, 4))]
        dirs = [first_dir]
        for _turn in range(turns):
            dirs.append(_turn_dir(dirs[-1], -1 if float(_RNG.random()) < 0.5 else 1))

        if turns == 0:
            lengths = [int(_RNG.integers(5, 9))]
        else:
            lengths = [int(x) for x in _RNG.integers(3, 7, size=turns + 1)]
        start = (int(_RNG.integers(2, h - 2)), int(_RNG.integers(2, w - 2)))
        path = _make_path(h, w, start, dirs, lengths)
        if path is None:
            continue

        first = dirs[0]
        last = dirs[-1]
        in_center = (path[0][0] - first[0], path[0][1] - first[1])
        out_center = (path[-1][0] + last[0], path[-1][1] + last[1])
        if not (1 <= in_center[0] < h - 1 and 1 <= in_center[1] < w - 1):
            continue
        if not (1 <= out_center[0] < h - 1 and 1 <= out_center[1] < w - 1):
            continue

        in_window = _window_set(in_center)
        out_window = _window_set(out_center)
        if not in_window.isdisjoint(out_window):
            continue

        path_set = set(path)
        if (path_set - {path[0]}) & in_window:
            continue
        if (path_set - {path[-1]}) & out_window:
            continue

        base_mask = SIGNAL_MASKS[int(_RNG.integers(0, len(SIGNAL_MASKS)))]
        mask = np.rot90(base_mask, int(_RNG.integers(0, 4))).astype(bool)
        in_signal = _patch_cells(in_center, mask)
        out_signal = _patch_cells(out_center, np.rot90(mask, -(turns % 4)).astype(bool))
        if in_signal & path_set or out_signal & path_set:
            continue

        reserved = path_set | in_window | out_window
        if reserved & forbidden:
            continue
        return path, in_center, out_center, mask, reserved
    return None


def _gen_attempt() -> np.ndarray | None:
    h = int(_RNG.integers(15, 26))
    w = int(_RNG.integers(15, 26))
    n_pipes = int(_RNG.integers(1, 4))
    palette = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)
    _RNG.shuffle(palette)
    pipe_colors = [int(c) for c in palette[:n_pipes]]
    signal_colors = [int(c) for c in palette[n_pipes : 2 * n_pipes]]

    grid = np.full((h, w), BG, dtype=int)
    forbidden: set[tuple[int, int]] = set()
    for pipe_color, signal_color in zip(pipe_colors, signal_colors):
        item = _random_pipe(h, w, forbidden)
        if item is None:
            return None
        path, in_center, _out_center, mask, reserved = item
        for r, c in path:
            grid[r, c] = pipe_color
        for r, c in _patch_cells(in_center, mask):
            grid[r, c] = signal_color
        forbidden |= _padded(reserved, h, w)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is None:
            continue
        try:
            out = main(grid)
        except AssertionError:
            continue
        if grid.ndim == 2 and out.ndim == 2 and max(grid.shape) <= 30 and max(out.shape) <= 30:
            return grid
    raise RuntimeError("failed to generate a valid pipe-signal input")
