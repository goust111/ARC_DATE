from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
RAY_COLOR = 5
CONFLICT_COLOR = 2
ENVELOPE_COLOR = 8
BAR_COLORS = (1, 3, 4, 6, 7, 9)

GRID_H_RANGE = (21, 29)
GRID_W_RANGE = (17, 25)
BAR_COUNT_CHOICES = (2, 3)
MARKER_COUNT_CHOICES = (1, 2)
BAR_LENGTH_RANGE = (5, 9)
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    visited = np.zeros(mask.shape, dtype=bool)
    comps: list[np.ndarray] = []
    h, w = mask.shape

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _parse_input(grid: np.ndarray) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert not np.any(g == CONFLICT_COLOR), "red is reserved for output"
    assert not np.any(g == ENVELOPE_COLOR), "teal is reserved for output"

    markers = [tuple(map(int, rc)) for rc in np.argwhere(g == RAY_COLOR)]
    assert markers, "expected at least one grey marker"
    marker_set = set(markers)
    for r, c in markers:
        assert sum((nr, nc) in marker_set for nr, nc in _neighbors4(r, c)) == 0, "markers must be isolated"

    bars: list[tuple[int, int, int, int]] = []
    for color in sorted(set(map(int, g.flatten())) - {BG, RAY_COLOR}):
        for comp in _components(g == color):
            r0, c0 = map(int, comp.min(axis=0))
            r1, c1 = map(int, comp.max(axis=0))
            assert r0 == r1, "every non-marker component must be horizontal"
            assert int(comp.shape[0]) == c1 - c0 + 1, "bars must be solid contiguous segments"
            bars.append((r0, c0, c1, int(color)))

    assert bars, "expected at least one bar"
    return sorted(bars), markers


def _envelope_masks(shape: tuple[int, int], bars: list[tuple[int, int, int, int]]) -> list[np.ndarray]:
    h, w = shape
    masks: list[np.ndarray] = []
    for row, c0, c1, _color in bars:
        mask = np.zeros(shape, dtype=bool)
        for offset in range(1, c1 - c0 + 1):
            left = c0 + offset
            right = c1 - offset
            if left > right:
                break
            for rr in (row - offset, row + offset):
                assert 0 <= rr < h, "bar envelope must stay in bounds"
                assert 0 <= left <= right < w, "bar envelope must stay in bounds"
                mask[rr, left : right + 1] = True
        masks.append(mask)
    return masks


def _ray_mask(shape: tuple[int, int], markers: list[tuple[int, int]]) -> np.ndarray:
    h, _w = shape
    mask = np.zeros(shape, dtype=bool)
    for r, c in markers:
        mask[:, c] = True
        mask[r, c] = False
    return mask


def _render_output(
    grid: np.ndarray, envelope_union: np.ndarray, ray_union: np.ndarray
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    bg_mask = out == BG
    conflict = bg_mask & envelope_union & ray_union
    teal_only = bg_mask & envelope_union & ~ray_union
    grey_only = bg_mask & ray_union & ~envelope_union
    out[teal_only] = ENVELOPE_COLOR
    out[grey_only] = RAY_COLOR
    out[conflict] = CONFLICT_COLOR
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build teal shrinking envelopes around horizontal bars and mark ray-envelope intersections in red.

    Input structure:
    - The background is black.
    - Foreground consists of several separated one-cell-thick horizontal solid bars, each in its own non-grey color.
    - One or more isolated grey pixels act as ray sources; red and teal are absent from the input.

    Rule:
    - Around each horizontal bar, generate a teal envelope by drawing matching rows above and below it; at offset k, the row is shortened by k cells from both ends.
    - From each grey source pixel, extend a vertical ray upward and downward through the same column.
    - Original bars and original grey source pixels stay unchanged.
    - On background cells, teal marks envelope-only positions, grey marks ray-only positions, and red marks positions that belong to both the ray column and an envelope position.

    Key insight:
    - Red pixels are not produced by ordinary overwrite order; they reveal cells that two independent constructions would both claim, so the solver must infer both the shrinking envelope and the vertical ray.
    """
    g = np.asarray(grid, dtype=int)
    bars, markers = _parse_input(g)
    envelope_union = np.any(np.stack(_envelope_masks(g.shape, bars), axis=0), axis=0)
    ray_union = _ray_mask(g.shape, markers)
    return _render_output(g, envelope_union, ray_union)


def _sample_rows(radii: list[int], height: int) -> list[int] | None:
    rows: list[int | None] = [None] * len(radii)
    order = sorted(range(len(radii)), key=lambda i: (-radii[i], random.random()))
    for idx in order:
        radius = radii[idx]
        candidates = []
        for row in range(radius, height - radius):
            ok = True
            for j, other in enumerate(rows):
                if other is None:
                    continue
                min_gap = max(6, radius + radii[j] + 1)
                if abs(row - other) < min_gap:
                    ok = False
                    break
            if ok:
                candidates.append(row)
        if not candidates:
            return None
        rows[idx] = random.choice(candidates)
    return [int(r) for r in rows]


def _sample_columns(lengths: list[int], width: int) -> list[int] | None:
    use_shared_core = len(lengths) >= 2 and random.random() < 0.7
    if use_shared_core:
        max_core = max(1, min(length - 2 for length in lengths))
        core_w = random.randint(1, min(3, max_core))
        core_start = random.randint(1, width - core_w - 2)
        core_end = core_start + core_w - 1
        cols = []
        for length in lengths:
            lo = max(0, core_end - length + 1)
            hi = min(core_start, width - length)
            if lo > hi:
                return None
            cols.append(random.randint(lo, hi))
        return cols
    return [random.randint(0, width - length) for length in lengths]


def _segment_choices(free_rows: list[int]) -> list[list[int]]:
    if not free_rows:
        return []
    segments = [[free_rows[0]]]
    for row in free_rows[1:]:
        if row == segments[-1][-1] + 1:
            segments[-1].append(row)
        else:
            segments.append([row])
    return segments


def _pick_marker_columns(per_bar_masks: list[np.ndarray], count: int) -> list[int] | None:
    width = per_bar_masks[0].shape[1]
    hit_counts = [sum(int(mask[:, c].any()) for mask in per_bar_masks) for c in range(width)]
    multi = [c for c, hits in enumerate(hit_counts) if hits >= 2]
    single = [c for c, hits in enumerate(hit_counts) if hits == 1]
    any_hit = [c for c, hits in enumerate(hit_counts) if hits > 0]
    if len(any_hit) < count:
        return None

    chosen: list[int] = []
    if count == 2:
        if multi and single:
            chosen = [random.choice(multi), random.choice(single)]
        elif len(multi) >= 2 and random.random() < 0.6:
            chosen = random.sample(multi, 2)

    if not chosen:
        pool = multi if multi and random.random() < 0.5 else any_hit
        chosen = [random.choice(pool)]
        if count == 2:
            remaining = [c for c in any_hit if c != chosen[0] and abs(c - chosen[0]) >= 2]
            if not remaining:
                remaining = [c for c in any_hit if c != chosen[0]]
            if not remaining:
                return None
            chosen.append(random.choice(remaining))

    random.shuffle(chosen)
    return chosen


def _pick_marker_rows(
    grid: np.ndarray,
    envelope_union: np.ndarray,
    marker_cols: list[int],
) -> list[int] | None:
    h, _w = grid.shape
    chosen_rows: list[int] = []
    used_positions: set[tuple[int, int]] = set()

    for col in marker_cols:
        free_rows = [r for r in range(h) if grid[r, col] == BG and not envelope_union[r, col]]
        segments = _segment_choices(free_rows)
        if not segments:
            return None
        segments.sort(key=lambda seg: (len(seg), random.random()), reverse=True)
        picked = None
        for seg in segments:
            candidates = seg[:]
            random.shuffle(candidates)
            for row in candidates:
                if (row, col) not in used_positions:
                    picked = row
                    break
            if picked is not None:
                break
        if picked is None:
            return None
        chosen_rows.append(int(picked))
        used_positions.add((int(picked), int(col)))
    return chosen_rows


def _manual_expected(grid: np.ndarray) -> np.ndarray:
    bars, markers = _parse_input(grid)
    per_bar_masks = _envelope_masks(grid.shape, bars)
    envelope_union = np.any(np.stack(per_bar_masks, axis=0), axis=0)
    ray_union = _ray_mask(grid.shape, markers)
    return _render_output(grid, envelope_union, ray_union)


def _validate_instance(grid: np.ndarray) -> None:
    bars, markers = _parse_input(grid)
    assert 2 <= len(bars) <= 3, "generator must produce 2-3 bars"
    assert 1 <= len(markers) <= 2, "generator must produce 1-2 markers"
    assert len({color for _r, _c0, _c1, color in bars}) == len(bars), "bar colors should be distinct"
    assert len({c for _r, c in markers}) == len(markers), "marker columns must be distinct"

    lengths = []
    for row, c0, c1, _color in bars:
        lengths.append(c1 - c0 + 1)
        assert 5 <= lengths[-1] <= 9, "bar length must stay in the target range"
        for mr, _mc in markers:
            assert mr != row, "marker may not sit on a bar row"

    per_bar_masks = _envelope_masks(grid.shape, bars)
    envelope_union = np.any(np.stack(per_bar_masks, axis=0), axis=0)
    for i in range(len(per_bar_masks)):
        for j in range(i + 1, len(per_bar_masks)):
            assert not np.any(per_bar_masks[i] & per_bar_masks[j]), "envelopes must not overlap"

    marker_set = set(markers)
    for r, c in marker_set:
        assert not envelope_union[r, c], "marker should stay off the future envelope for visual clarity"
        assert per_bar_masks[0].shape[0] <= 30 and per_bar_masks[0].shape[1] <= 30

    ray_union = _ray_mask(grid.shape, markers)
    actual = main(grid)
    expected = _manual_expected(grid)
    assert np.array_equal(actual, expected), "manual semantics and solver disagree"

    for _r, c in markers:
        conflict_count = int(np.count_nonzero(envelope_union[:, c]))
        assert conflict_count > 0, "every marker column must cross at least one envelope"
        assert np.any(actual[:, c] == CONFLICT_COLOR), "every marker must create red conflict pixels"
        assert np.count_nonzero(actual[:, c] == RAY_COLOR) >= 2, "every marker ray should show visible grey travel"

    assert np.any(actual == CONFLICT_COLOR), "output must contain red conflicts"
    assert np.any(actual == ENVELOPE_COLOR), "output must contain teal non-conflicts"
    assert np.count_nonzero(actual == RAY_COLOR) > len(markers), "output must contain grey beyond the source pixels"

    env_only = np.asarray(grid, dtype=int).copy()
    env_only[(grid == BG) & envelope_union] = ENVELOPE_COLOR
    ray_only = np.asarray(grid, dtype=int).copy()
    ray_only[(grid == BG) & ray_union] = RAY_COLOR
    assert not np.array_equal(actual, env_only), "ray step must matter"
    assert not np.array_equal(actual, ray_only), "envelope step must matter"


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    bar_count = random.choice(BAR_COUNT_CHOICES)
    marker_count = random.choice(MARKER_COUNT_CHOICES)
    lengths = [random.randint(*BAR_LENGTH_RANGE) for _ in range(bar_count)]
    radii = [(length - 1) // 2 for length in lengths]
    rows = _sample_rows(radii, height)
    if rows is None:
        return None
    cols = _sample_columns(lengths, width)
    if cols is None:
        return None

    grid = np.full((height, width), BG, dtype=int)
    colors = random.sample(BAR_COLORS, bar_count)
    for row, col, length, color in zip(rows, cols, lengths, colors):
        grid[row, col : col + length] = int(color)

    bars = [(row, col, col + length - 1, color) for row, col, length, color in zip(rows, cols, lengths, colors)]
    per_bar_masks = _envelope_masks(grid.shape, bars)
    envelope_union = np.any(np.stack(per_bar_masks, axis=0), axis=0)
    marker_cols = _pick_marker_columns(per_bar_masks, marker_count)
    if marker_cols is None:
        return None
    marker_rows = _pick_marker_rows(grid, envelope_union, marker_cols)
    if marker_rows is None:
        return None

    for row, col in zip(marker_rows, marker_cols):
        grid[row, col] = RAY_COLOR

    try:
        _validate_instance(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
