from __future__ import annotations

from collections import Counter

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _window_cells(r0: int, c0: int) -> tuple[tuple[int, int], ...]:
    return ((r0, c0), (r0, c0 + 1), (r0 + 1, c0 + 1), (r0 + 1, c0))


def _recover_lights(grid: np.ndarray, bg: int) -> list[tuple[int, int]]:
    """Recover the unique set of 2x2 light windows covering all non-bg cells."""
    h, w = grid.shape
    fg = {(int(r), int(c)) for r, c in np.argwhere(grid != bg)}
    if not fg:
        return []

    candidates: list[dict[str, object]] = []
    seen: set[tuple[int, int]] = set()
    for r, c in sorted(fg):
        for r0 in (r - 1, r):
            for c0 in (c - 1, c):
                if not (0 <= r0 < h - 1 and 0 <= c0 < w - 1):
                    continue
                if (r0, c0) in seen:
                    continue
                seen.add((r0, c0))
                cells = _window_cells(r0, c0)
                fg_cells = frozenset(cell for cell in cells if cell in fg)
                if fg_cells:
                    candidates.append(
                        {
                            "top_left": (r0, c0),
                            "cells": frozenset(cells),
                            "fg_cells": fg_cells,
                        }
                    )

    cell_to_candidates: dict[tuple[int, int], list[int]] = {cell: [] for cell in fg}
    for idx, cand in enumerate(candidates):
        for cell in cand["fg_cells"]:
            cell_to_candidates[cell].append(idx)

    overlaps: list[set[int]] = []
    for i, cand_i in enumerate(candidates):
        blocked = {i}
        cells_i = cand_i["cells"]
        for j in range(i + 1, len(candidates)):
            if cells_i & candidates[j]["cells"]:
                blocked.add(j)
        overlaps.append(blocked)
    for i in range(len(candidates)):
        for j in list(overlaps[i]):
            overlaps[j].add(i)

    ordered_options = {
        cell: sorted(
            idxs,
            key=lambda idx: (
                -len(candidates[idx]["fg_cells"]),
                candidates[idx]["top_left"][0],
                candidates[idx]["top_left"][1],
            ),
        )
        for cell, idxs in cell_to_candidates.items()
    }

    best_count = [10**9]
    solutions: list[list[int]] = []

    def search(
        uncovered: frozenset[tuple[int, int]],
        available: frozenset[int],
        chosen: tuple[int, ...],
    ) -> None:
        if len(chosen) >= best_count[0]:
            return
        if not uncovered:
            if len(chosen) < best_count[0]:
                best_count[0] = len(chosen)
                solutions.clear()
            if len(solutions) < 2:
                solutions.append(list(chosen))
            return

        target = min(
            uncovered,
            key=lambda cell: sum(idx in available for idx in ordered_options[cell]),
        )
        options = [idx for idx in ordered_options[target] if idx in available]
        if not options:
            return

        for idx in options:
            fg_cells = candidates[idx]["fg_cells"]
            new_uncovered = uncovered - fg_cells
            new_available = available - frozenset(overlaps[idx])
            dead = False
            for cell in new_uncovered:
                if not any(other in new_available for other in ordered_options[cell]):
                    dead = True
                    break
            if dead:
                continue
            search(new_uncovered, new_available, chosen + (idx,))

    search(frozenset(fg), frozenset(range(len(candidates))), ())
    assert len(solutions) == 1, "signal lights must be uniquely recoverable"

    lights = [candidates[idx]["top_left"] for idx in solutions[0]]
    return sorted((int(r0), int(c0)) for r0, c0 in lights)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast axis-aligned rays from 2x2 signal lights.

    Input structure
    - Background-filled grid with several disjoint 2x2 signal lights.
    - Each light may contain background-colored cells.
    - Light positions are uniquely recoverable from the non-background cells.

    Rule
    - For each light, the top-left, top-right, bottom-right, and bottom-left
      cells control left, up, right, and down rays respectively.
    - A non-background controller cell emits a ray in its direction using the
      controller cell's color.
    - Rays stop at the grid edge, another light window, or an already-painted
      ray cell.
    - Lights are processed in row-major reading order and remain unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    lights = _recover_lights(g, bg)

    out = g.copy()
    light_mask = np.zeros_like(g, dtype=bool)
    for r0, c0 in lights:
        light_mask[r0 : r0 + 2, c0 : c0 + 2] = True

    emitters = (
        (0, 0, 0, -1),   # top-left -> left
        (0, 1, -1, 0),   # top-right -> up
        (1, 1, 0, 1),    # bottom-right -> right
        (1, 0, 1, 0),    # bottom-left -> down
    )

    h, w = g.shape
    for r0, c0 in lights:
        for dr0, dc0, step_r, step_c in emitters:
            color = int(g[r0 + dr0, c0 + dc0])
            if color == bg:
                continue
            r = r0 + dr0 + step_r
            c = c0 + dc0 + step_c
            while 0 <= r < h and 0 <= c < w and not light_mask[r, c] and int(out[r, c]) == bg:
                out[r, c] = color
                r += step_r
                c += step_c

    return out


_RNG = np.random.default_rng()
_GRID_H_RANGE = (9, 17)
_GRID_W_RANGE = (9, 17)
_LIGHT_COUNT_RANGE = (2, 5)
_PALETTE_SIZE_RANGE = (3, 6)
_MAX_RETRIES = 300

_VALID_PATTERNS = (
    (1, 0, 1, 0),  # diagonal
    (0, 1, 0, 1),  # diagonal
    (0, 1, 1, 1),
    (1, 0, 1, 1),
    (1, 1, 0, 1),
    (1, 1, 1, 0),
    (1, 1, 1, 1),
)


def _sample_light_positions(h: int, w: int, count: int) -> list[tuple[int, int]] | None:
    candidates = [(r, c) for r in range(1, h - 2) for c in range(1, w - 2)]
    _RNG.shuffle(candidates)

    chosen: list[tuple[int, int]] = []
    reserved = np.zeros((h, w), dtype=bool)
    for r0, c0 in candidates:
        r1 = r0 + 2
        c1 = c0 + 2
        pad_r0 = max(0, r0 - 1)
        pad_r1 = min(h, r1 + 1)
        pad_c0 = max(0, c0 - 1)
        pad_c1 = min(w, c1 + 1)
        if reserved[pad_r0:pad_r1, pad_c0:pad_c1].any():
            continue
        chosen.append((r0, c0))
        reserved[pad_r0:pad_r1, pad_c0:pad_c1] = True
        if len(chosen) == count:
            return sorted(chosen)
    return None


def _build_light(pattern: tuple[int, int, int, int], colors: list[int], bg: int) -> np.ndarray:
    light = np.full((2, 2), bg, dtype=int)
    cells = ((0, 0), (0, 1), (1, 1), (1, 0))
    for active, (r, c) in zip(pattern, cells):
        if active:
            light[r, c] = int(_RNG.choice(colors))
    return light


def _active_directions(light: np.ndarray, bg: int) -> set[int]:
    return {idx for idx, value in enumerate(light[(0, 0, 1, 1), (0, 1, 1, 0)]) if int(value) != bg}


def _gen_attempt() -> np.ndarray | None:
    h = int(_RNG.integers(_GRID_H_RANGE[0], _GRID_H_RANGE[1] + 1))
    w = int(_RNG.integers(_GRID_W_RANGE[0], _GRID_W_RANGE[1] + 1))
    bg = int(_RNG.integers(0, 10))

    capacity_hint = ((h - 3) // 4 + 1) * ((w - 3) // 4 + 1)
    max_lights = min(_LIGHT_COUNT_RANGE[1], capacity_hint)
    if max_lights < _LIGHT_COUNT_RANGE[0]:
        return None
    count = int(_RNG.integers(_LIGHT_COUNT_RANGE[0], max_lights + 1))
    positions = _sample_light_positions(h, w, count)
    if positions is None:
        return None

    palette_size = int(_RNG.integers(_PALETTE_SIZE_RANGE[0], _PALETTE_SIZE_RANGE[1] + 1))
    colors = [c for c in range(10) if c != bg]
    _RNG.shuffle(colors)
    palette = colors[:palette_size]

    grid = np.full((h, w), bg, dtype=int)
    placed_windows: list[tuple[int, int, tuple[int, ...]]] = []
    used_dirs: set[int] = set()
    active_emitters = 0
    for r0, c0 in positions:
        pattern = _VALID_PATTERNS[int(_RNG.integers(len(_VALID_PATTERNS)))]
        light = _build_light(pattern, palette, bg)
        grid[r0 : r0 + 2, c0 : c0 + 2] = light
        placed_windows.append((r0, c0, tuple(int(v) for v in light.ravel())))
        used_dirs |= _active_directions(light, bg)
        active_emitters += int(np.count_nonzero(light != bg))

    if len(used_dirs) < 3 or active_emitters < count * 2:
        return None
    if np.count_nonzero(grid != bg) * 2 >= grid.size:
        return None

    recovered = _recover_lights(grid, bg)
    if recovered != positions:
        return None
    recovered_windows = [
        (r0, c0, tuple(int(v) for v in grid[r0 : r0 + 2, c0 : c0 + 2].ravel()))
        for r0, c0 in recovered
    ]
    if Counter(placed_windows) != Counter(recovered_windows):
        return None

    out = main(grid)
    added = int(np.count_nonzero((grid == bg) & (out != bg)))
    if added < max(2, count - 1):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid signal-light puzzle input")
