from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
BAR = 5
ONE_LAYER = 3
TWO_LAYERS = 2
THREE_PLUS_LAYERS = 4

GRID_ROWS = (16, 21)
GRID_COLS = (16, 21)
BAR_COUNT_CHOICES = (2, 2, 3, 3, 4)
BAR_LENGTHS = (3, 4, 5, 6, 7)
MAX_GENERATION_ATTEMPTS = 400
SEARCH_CANDIDATE_LIMIT = 96
MAX_FILL_FRACTION = 0.34


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    height, width = map(int, mask.shape)
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []

    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _make_bar(orientation: str, anchor0: int, anchor1: int, length: int) -> dict:
    radius = (int(length) - 1) // 2
    if orientation == "H":
        row = int(anchor0)
        c0 = int(anchor1)
        cells = frozenset((row, c0 + dc) for dc in range(int(length)))
        envelope = set()
        for d in range(1, radius + 1):
            left = c0 + d
            right = c0 + int(length) - 1 - d
            for rr in (row - d, row + d):
                for cc in range(left, right + 1):
                    envelope.add((rr, cc))
        return {
            "orientation": "H",
            "length": int(length),
            "radius": radius,
            "row": row,
            "c0": c0,
            "c1": c0 + int(length) - 1,
            "cells": cells,
            "envelope": frozenset(envelope),
        }

    r0 = int(anchor0)
    col = int(anchor1)
    cells = frozenset((r0 + dr, col) for dr in range(int(length)))
    envelope = set()
    for d in range(1, radius + 1):
        top = r0 + d
        bottom = r0 + int(length) - 1 - d
        for cc in (col - d, col + d):
            for rr in range(top, bottom + 1):
                envelope.add((rr, cc))
    return {
        "orientation": "V",
        "length": int(length),
        "radius": radius,
        "r0": r0,
        "r1": r0 + int(length) - 1,
        "col": col,
        "cells": cells,
        "envelope": frozenset(envelope),
    }


def _extract_bar(coords: np.ndarray) -> dict:
    rows = sorted({int(r) for r in coords[:, 0]})
    cols = sorted({int(c) for c in coords[:, 1]})
    if len(rows) == 1:
        row = rows[0]
        c0, c1 = cols[0], cols[-1]
        assert len(cols) == c1 - c0 + 1
        assert int(coords.shape[0]) == len(cols)
        return _make_bar("H", row, c0, len(cols))
    if len(cols) == 1:
        col = cols[0]
        r0, r1 = rows[0], rows[-1]
        assert len(rows) == r1 - r0 + 1
        assert int(coords.shape[0]) == len(rows)
        return _make_bar("V", r0, col, len(rows))
    raise AssertionError("foreground component must be a straight one-cell-thick bar")


def _bars_from_grid(grid: np.ndarray) -> list[dict]:
    comps = _components(grid == BAR)
    bars = [_extract_bar(comp) for comp in comps]
    assert 2 <= len(bars) <= 4, "expected 2-4 bars"
    return bars


def _coverage_counts(shape: tuple[int, int], bars: list[dict]) -> np.ndarray:
    height, width = map(int, shape)
    counts = np.zeros((height, width), dtype=int)
    for bar in bars:
        for r, c in bar["envelope"]:
            assert 0 <= r < height and 0 <= c < width
            counts[r, c] += 1
    return counts


def _render_output_from_bars(shape: tuple[int, int], bars: list[dict]) -> np.ndarray:
    out = np.full(shape, BACKGROUND, dtype=int)
    counts = _coverage_counts(shape, bars)
    bar_mask = np.zeros(shape, dtype=bool)
    for bar in bars:
        for r, c in bar["cells"]:
            bar_mask[r, c] = True

    bg_mask = ~bar_mask
    out[bg_mask & (counts == 1)] = ONE_LAYER
    out[bg_mask & (counts == 2)] = TWO_LAYERS
    out[bg_mask & (counts >= 3)] = THREE_PLUS_LAYERS
    out[bar_mask] = BAR
    return out


def _bars_conflict(a: dict, b: dict) -> bool:
    if a["cells"] & b["cells"]:
        return True
    b_cells = b["cells"]
    for r, c in a["cells"]:
        for nr, nc in _neighbors4(int(r), int(c)):
            if (nr, nc) in b_cells:
                return True
    return False


def _shadow_present(bars: list[dict]) -> bool:
    for i, bar in enumerate(bars):
        for j, other in enumerate(bars):
            if i == j:
                continue
            if bar["cells"] & other["envelope"]:
                return True
    return False


def _semantic_profile(shape: tuple[int, int], bars: list[dict]) -> dict:
    counts = _coverage_counts(shape, bars)
    bar_mask = np.zeros(shape, dtype=bool)
    for bar in bars:
        for r, c in bar["cells"]:
            bar_mask[r, c] = True
    bg_mask = ~bar_mask
    painted = bg_mask & (counts > 0)
    return {
        "has_single": bool(np.any(bg_mask & (counts == 1))),
        "has_double": bool(np.any(bg_mask & (counts == 2))),
        "has_triple": bool(np.any(bg_mask & (counts >= 3))),
        "painted_cells": int(np.count_nonzero(painted)),
        "fill_fraction": float(np.count_nonzero(painted)) / float(counts.size),
        "shadow": _shadow_present(bars),
        "mixed_orientation": len({bar["orientation"] for bar in bars}) > 1,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color each background cell by how many bar-generated diamond envelopes cover it.

    Input structure:
    - A black background contains 2-4 gray foreground objects.
    - Each foreground object is a one-cell-thick straight horizontal or vertical bar.
    - Bars are separate 4-connected components; only black and gray appear in the input.

    Rule:
    - Keep every gray bar pixel unchanged.
    - Around each bar, form the same shrinking diamond envelope used in bar-envelope ARC tasks: every step away from the bar shortens the parallel segment by one cell at each end, and the bar itself is excluded from its own envelope.
    - For every background cell, count how many bar envelopes cover it, then recolor count 1 to green, count 2 to red, and count 3 or more to yellow.
    - Leave uncovered background cells black, and if a bar cell lies inside another bar's envelope it still stays gray.

    Key insight:
    - The hidden object is not a single envelope but the overlap multiplicity of several envelopes, so solving requires reconstructing each diamond and then reading color from coverage depth rather than from bar order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))) <= {BACKGROUND, BAR}, "input must contain only black background and gray bars"

    bars = _bars_from_grid(g)
    out = _render_output_from_bars(g.shape, bars)
    return out


def _cover_candidates(shape: tuple[int, int], target: tuple[int, int]) -> list[dict]:
    height, width = map(int, shape)
    tr, tc = map(int, target)
    seen: set[tuple[str, int, int, int]] = set()
    candidates: list[dict] = []

    for length in BAR_LENGTHS:
        radius = (int(length) - 1) // 2

        for row in range(radius, height - radius):
            d = abs(row - tr)
            if not (1 <= d <= radius):
                continue
            left_min = tc - (int(length) - 1 - d)
            left_max = tc - d
            for c0 in range(max(0, left_min), min(width - int(length), left_max) + 1):
                key = ("H", row, c0, int(length))
                if key not in seen:
                    seen.add(key)
                    candidates.append(_make_bar("H", row, c0, int(length)))

        for col in range(radius, width - radius):
            d = abs(col - tc)
            if not (1 <= d <= radius):
                continue
            top_min = tr - (int(length) - 1 - d)
            top_max = tr - d
            for r0 in range(max(0, top_min), min(height - int(length), top_max) + 1):
                key = ("V", r0, col, int(length))
                if key not in seen:
                    seen.add(key)
                    candidates.append(_make_bar("V", r0, col, int(length)))

    random.shuffle(candidates)
    candidates.sort(key=lambda bar: (-bar["length"], random.random()))
    return candidates[:SEARCH_CANDIDATE_LIMIT]


def _choose_bars_for_target(
    shape: tuple[int, int],
    target: tuple[int, int],
    n_bars: int,
    require_mixed: bool,
    require_shadow: bool,
    require_triple: bool,
) -> list[dict] | None:
    candidates = _cover_candidates(shape, target)
    if len(candidates) < n_bars:
        return None

    chosen: list[dict] = []

    def dfs(start: int) -> list[dict] | None:
        if len(chosen) == n_bars:
            profile = _semantic_profile(shape, chosen)
            if not profile["has_single"] or not profile["has_double"]:
                return None
            if require_triple and not profile["has_triple"]:
                return None
            if not require_triple and profile["has_triple"]:
                return None
            if require_mixed and not profile["mixed_orientation"]:
                return None
            if require_shadow and not profile["shadow"]:
                return None
            if profile["painted_cells"] < 6:
                return None
            if profile["fill_fraction"] > MAX_FILL_FRACTION:
                return None
            return [dict(bar) for bar in chosen]

        remaining = n_bars - len(chosen)
        if len(candidates) - start < remaining:
            return None

        for idx in range(start, len(candidates)):
            cand = candidates[idx]
            if any(_bars_conflict(cand, other) for other in chosen):
                continue
            chosen.append(cand)
            result = dfs(idx + 1)
            if result is not None:
                return result
            chosen.pop()
        return None

    return dfs(0)


def _render_input(shape: tuple[int, int], bars: list[dict]) -> np.ndarray:
    grid = np.full(shape, BACKGROUND, dtype=int)
    for bar in bars:
        for r, c in bar["cells"]:
            grid[r, c] = BAR
    return grid


def _valid_generated_grid(grid: np.ndarray, require_triple: bool, require_shadow: bool) -> bool:
    try:
        bars = _bars_from_grid(grid)
    except AssertionError:
        return False

    profile = _semantic_profile(grid.shape, bars)
    if not profile["has_single"] or not profile["has_double"]:
        return False
    if require_triple != profile["has_triple"]:
        return False
    if require_shadow and not profile["shadow"]:
        return False
    if profile["fill_fraction"] > MAX_FILL_FRACTION:
        return False

    expected = _render_output_from_bars(grid.shape, bars)
    return bool(np.array_equal(main(grid), expected))


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(*GRID_ROWS)
    width = random.randint(*GRID_COLS)
    shape = (height, width)

    n_bars = random.choice(BAR_COUNT_CHOICES)
    require_triple = n_bars >= 3 and random.random() < 0.8
    if n_bars == 2:
        require_triple = False
    require_mixed = random.random() < (0.7 if n_bars >= 3 else 0.55)
    require_shadow = random.random() < 0.45

    margin = 4
    if height <= 2 * margin or width <= 2 * margin:
        return None
    target = (random.randint(margin, height - margin - 1), random.randint(margin, width - margin - 1))

    bars = _choose_bars_for_target(shape, target, n_bars, require_mixed, require_shadow, require_triple)
    if bars is None:
        return None

    grid = _render_input(shape, bars)
    if not _valid_generated_grid(grid, require_triple, require_shadow):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid overlapping-envelope ARC input")
