from __future__ import annotations

from collections import deque

import numpy as np


BLACK = 0
BAR_COLORS = (1, 2, 3)
CONFLICT_COLOR = 4


def _find_bar_components(grid: np.ndarray) -> list[dict]:
    h, w = map(int, grid.shape)
    visited = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BLACK or visited[r, c]:
                continue

            q: deque[tuple[int, int]] = deque([(r, c)])
            visited[r, c] = True
            coords: list[tuple[int, int]] = []

            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        q.append((nr, nc))

            components.append({"color": color, "coords": coords})

    return components


def _make_bar(color: int, orientation: str, start_r: int, start_c: int, length: int) -> dict:
    radius = (int(length) - 1) // 2
    if orientation == "H":
        cells = [(int(start_r), int(start_c + offset)) for offset in range(int(length))]
        return {
            "color": int(color),
            "orientation": "H",
            "length": int(length),
            "radius": int(radius),
            "row": int(start_r),
            "c0": int(start_c),
            "c1": int(start_c + length - 1),
            "cells": cells,
        }

    cells = [(int(start_r + offset), int(start_c)) for offset in range(int(length))]
    return {
        "color": int(color),
        "orientation": "V",
        "length": int(length),
        "radius": int(radius),
        "col": int(start_c),
        "r0": int(start_r),
        "r1": int(start_r + length - 1),
        "cells": cells,
    }


def _component_to_bar(component: dict) -> dict:
    color = int(component["color"])
    coords = sorted((int(r), int(c)) for r, c in component["coords"])
    rows = sorted({r for r, _c in coords})
    cols = sorted({c for _r, c in coords})

    if len(rows) == 1:
        row = rows[0]
        c0, c1 = cols[0], cols[-1]
        expected = {(row, c) for c in range(c0, c1 + 1)}
        assert expected == set(coords), "horizontal bar must be contiguous"
        return _make_bar(color, "H", row, c0, c1 - c0 + 1)

    if len(cols) == 1:
        col = cols[0]
        r0, r1 = rows[0], rows[-1]
        expected = {(r, col) for r in range(r0, r1 + 1)}
        assert expected == set(coords), "vertical bar must be contiguous"
        return _make_bar(color, "V", r0, col, r1 - r0 + 1)

    raise AssertionError("foreground component must be a one-cell-thick straight bar")


def _bar_cell_mask(shape: tuple[int, int], bar: dict) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in bar["cells"]:
        mask[int(r), int(c)] = True
    return mask


def _diamond_mask(shape: tuple[int, int], bar: dict) -> np.ndarray:
    h, w = map(int, shape)
    mask = np.zeros((h, w), dtype=bool)
    radius = int(bar["radius"])

    if bar["orientation"] == "H":
        row = int(bar["row"])
        c0 = int(bar["c0"])
        c1 = int(bar["c1"])
        for r in range(h):
            row_dist = abs(r - row)
            if row_dist > radius:
                continue
            for c in range(w):
                if c < c0:
                    dist = row_dist + (c0 - c)
                elif c > c1:
                    dist = row_dist + (c - c1)
                else:
                    dist = row_dist
                if dist <= radius:
                    mask[r, c] = True
        return mask

    col = int(bar["col"])
    r0 = int(bar["r0"])
    r1 = int(bar["r1"])
    for r in range(h):
        for c in range(w):
            col_dist = abs(c - col)
            if col_dist > radius:
                continue
            if r < r0:
                dist = col_dist + (r0 - r)
            elif r > r1:
                dist = col_dist + (r - r1)
            else:
                dist = col_dist
            if dist <= radius:
                mask[r, c] = True
    return mask


def _coverage_info(shape: tuple[int, int], bars: list[dict]) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    coverage = np.zeros(shape, dtype=int)
    masks: list[np.ndarray] = []
    bar_mask = np.zeros(shape, dtype=bool)

    for bar in bars:
        mask = _diamond_mask(shape, bar)
        masks.append(mask)
        coverage += mask.astype(int)
        bar_mask |= _bar_cell_mask(shape, bar)

    return coverage, masks, bar_mask


def _render_from_bars(shape: tuple[int, int], bars: list[dict]) -> np.ndarray:
    coverage, masks, _bar_mask = _coverage_info(shape, bars)
    out = np.full(shape, BLACK, dtype=int)

    for bar, mask in zip(bars, masks):
        out[mask & (coverage == 1)] = int(bar["color"])

    out[coverage >= 2] = CONFLICT_COLOR
    for bar in bars:
        for r, c in bar["cells"]:
            out[int(r), int(c)] = int(bar["color"])

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each colored bar into its filled diamond envelope and mark non-bar overlaps yellow.

    Input structure:
    - The grid has a black background and 2-4 non-touching one-cell-thick solid bars.
    - Each bar is either horizontal or vertical, uses one of the source colors, and has length at least 3.
    - No other foreground objects are present.

    Rule:
    - For each bar of length L, build the filled diamond region consisting of all cells whose Manhattan distance to that bar is at most floor((L-1)/2).
    - Cells covered by exactly one bar region take that bar's color, while cells covered by two or more regions become yellow.
    - Original bar pixels always keep their source color even if another bar's region also covers them, and uncovered cells stay black.

    Key insight:
    - The input only gives thin line segments, so the real task is to infer each segment's Manhattan-radius diamond first and only then do a coverage-count recoloring pass.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    components = _find_bar_components(g)
    assert len(components) >= 2, "expected at least two bars"
    bars = [_component_to_bar(component) for component in components]

    for bar in bars:
        assert int(bar["length"]) >= 3, "bar length must be at least 3"
        assert int(bar["color"]) != CONFLICT_COLOR, "yellow is reserved for overlap cells"

    return _render_from_bars(g.shape, bars)


_RNG = np.random.default_rng()

GRID_ROWS_RANGE = (18, 25)
GRID_COLS_RANGE = (18, 25)
BAR_COUNT_OPTIONS = (2, 3)
BAR_LENGTH_VALUES = np.array([3, 4, 5, 6, 7, 8, 9], dtype=int)
BAR_LENGTH_WEIGHTS = np.array([0.06, 0.10, 0.18, 0.22, 0.19, 0.15, 0.10], dtype=float)
MAX_GENERATION_ATTEMPTS = 500


def _bar_distance(a: dict, b: dict) -> int:
    return min(abs(int(ar) - int(br)) + abs(int(ac) - int(bc)) for ar, ac in a["cells"] for br, bc in b["cells"])


def _bar_is_legal(bar: dict, placed: list[dict], shape: tuple[int, int]) -> bool:
    h, w = map(int, shape)
    if any(not (0 <= int(r) < h and 0 <= int(c) < w) for r, c in bar["cells"]):
        return False
    return all(_bar_distance(bar, other) > 1 for other in placed)


def _sample_orientations(count: int) -> list[str]:
    if count == 2:
        if float(_RNG.random()) < 0.55:
            return ["H", "V"]
        axis = str(_RNG.choice(np.array(["H", "V"], dtype=object)))
        return [axis, axis]

    patterns = np.array(
        [
            ["H", "H", "V"],
            ["H", "V", "V"],
            ["H", "H", "H"],
            ["V", "V", "V"],
        ],
        dtype=object,
    )
    return list(patterns[int(_RNG.integers(0, len(patterns)))])


def _enumerate_positions(shape: tuple[int, int], color: int, orientation: str, length: int, placed: list[dict]) -> list[dict]:
    h, w = map(int, shape)
    candidates: list[dict] = []

    if orientation == "H":
        for r in range(h):
            for c in range(w - int(length) + 1):
                bar = _make_bar(color, "H", r, c, length)
                if _bar_is_legal(bar, placed, shape):
                    candidates.append(bar)
    else:
        for r in range(h - int(length) + 1):
            for c in range(w):
                bar = _make_bar(color, "V", r, c, length)
                if _bar_is_legal(bar, placed, shape):
                    candidates.append(bar)

    return candidates


def _sample_specs() -> list[tuple[int, str, int]]:
    count = int(_RNG.choice(np.array(BAR_COUNT_OPTIONS, dtype=int)))
    colors = list(_RNG.permutation(np.array(BAR_COLORS, dtype=int))[:count])
    orientations = _sample_orientations(count)
    lengths = [int(_RNG.choice(BAR_LENGTH_VALUES, p=BAR_LENGTH_WEIGHTS)) for _ in range(count)]
    specs = list(zip(colors, orientations, lengths))
    specs.sort(key=lambda item: item[2], reverse=True)
    return specs


def _validate_layout(shape: tuple[int, int], bars: list[dict], require_protected_overlap: bool) -> bool:
    coverage, masks, bar_mask = _coverage_info(shape, bars)
    out = _render_from_bars(shape, bars)

    yellow_cells = int(np.count_nonzero((coverage >= 2) & ~bar_mask))
    total_colored = int(np.count_nonzero(out != BLACK))
    protected_overlap = bool(np.any((coverage >= 2) & bar_mask))

    if yellow_cells == 0:
        return False
    if yellow_cells > max(6, total_colored // 3):
        return False
    if total_colored < 18 or total_colored > (shape[0] * shape[1]) // 2:
        return False
    if require_protected_overlap and not protected_overlap:
        return False

    for bar, mask in zip(bars, masks):
        visible_unique = mask & (coverage == 1) & ~_bar_cell_mask(shape, bar)
        if int(np.count_nonzero(visible_unique)) == 0:
            return False

    return True


def _attempt_generate() -> np.ndarray | None:
    shape = (
        int(_RNG.integers(GRID_ROWS_RANGE[0], GRID_ROWS_RANGE[1])),
        int(_RNG.integers(GRID_COLS_RANGE[0], GRID_COLS_RANGE[1])),
    )
    specs = _sample_specs()
    require_protected_overlap = bool(float(_RNG.random()) < 0.35)

    bars: list[dict] = []
    for color, orientation, length in specs:
        candidates = _enumerate_positions(shape, color, orientation, length, bars)
        if not candidates:
            return None
        _RNG.shuffle(candidates)

        chosen = None
        for candidate in candidates:
            trial = bars + [candidate]
            if len(trial) < len(specs):
                chosen = candidate
                break
            if _validate_layout(shape, trial, require_protected_overlap):
                chosen = candidate
                break
        if chosen is None:
            return None
        bars.append(chosen)

    if not _validate_layout(shape, bars, require_protected_overlap):
        return None

    grid = np.full(shape, BLACK, dtype=int)
    for bar in bars:
        for r, c in bar["cells"]:
            grid[int(r), int(c)] = int(bar["color"])

    expected = _render_from_bars(shape, bars)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid diamond-envelope collision input")
