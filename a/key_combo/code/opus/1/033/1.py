import random
from typing import Dict, List, Sequence, Tuple

import numpy as np


BG = 0
MARKER = 6
PERIOD_VALUES = (2, 3, 4)
GRID_HEIGHT_RANGE = (12, 18)
GRID_WIDTH_RANGE = (15, 22)
MARKER_COUNT_RANGE = (3, 8)
GEN_ATTEMPTS = 400
COLOR_POOL = tuple(c for c in range(1, 10) if c != MARKER)

_RNG = random.Random()
_PERIOD_BAG: List[Tuple[int, int]] = []
_COLOR_COUNT_BAG: List[int] = []


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore the pink defects inside a right-sheared periodic tiling.

    Input structure:
    - The background is black.
    - Every row has a contiguous visible suffix of non-black cells, and the suffix start moves one column right per row, forming a descending black staircase edge.
    - The visible non-black region comes from a rectangular color tile with height and width between 2 and 4, repeated on an unsheared grid and then sheared row by row.
    - Some visible cells are replaced by pink marker pixels that hide the original tile colors.

    Rule:
    - Infer the per-row horizontal shear offsets from the staircase edge.
    - Convert each visible cell at (r, c) back to unsheared coordinates (r, c - offset[r]).
    - Among all 2..4 by 2..4 tile sizes, find the unique smallest period consistent with every non-pink visible cell.
    - Recolor each pink marker from that recovered tile phase, leaving the background staircase and all non-pink cells unchanged.

    Key insight:
    The periodic pattern only aligns after undoing the row-wise shear revealed by the diagonal black edge, so the marker colors must be recovered in unsheared phase coordinates rather than screen coordinates.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert np.any(g == MARKER), "expected at least one marker"

    offsets = _infer_offsets(g)
    candidates = _valid_period_candidates(g, offsets)
    assert candidates, "expected at least one consistent tile period"

    best_area = min(h_t * w_t for h_t, w_t, _phase_map in candidates)
    best = [cand for cand in candidates if cand[0] * cand[1] == best_area]
    assert len(best) == 1, "tile period must be uniquely determined"
    h_t, w_t, phase_map = best[0]

    out = g.copy()
    marker_rows, marker_cols = np.where(g == MARKER)
    for r0, c0 in zip(marker_rows, marker_cols):
        r = int(r0)
        c = int(c0)
        phase = (r % h_t, (c - offsets[r]) % w_t)
        assert phase in phase_map, "marker phase must be visible elsewhere"
        out[r, c] = phase_map[phase]

    assert not np.any(out == MARKER), "all markers must be repaired"
    return out


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid sheared periodic repair input")


def _infer_offsets(grid: np.ndarray) -> List[int]:
    h, w = grid.shape
    offsets: List[int] = []
    for r in range(h):
        row = grid[r]
        non_bg = np.flatnonzero(row != BG)
        assert non_bg.size > 0, "every row must contain visible pattern cells"
        start = int(non_bg[0])
        assert start < w
        assert np.all(row[:start] == BG), "cells before the visible suffix must be background"
        assert np.all(row[start:] != BG), "visible suffix must be contiguous and non-black"
        offsets.append(start)

    base = offsets[0]
    expected = [base + r for r in range(h)]
    assert offsets == expected, "row offsets must follow a unit right shear"
    return offsets


def _valid_period_candidates(
    grid: np.ndarray, offsets: Sequence[int]
) -> List[Tuple[int, int, Dict[Tuple[int, int], int]]]:
    h, w = grid.shape
    max_period_w = min(4, max(w - start for start in offsets))
    candidates: List[Tuple[int, int, Dict[Tuple[int, int], int]]] = []

    for h_t in range(2, min(4, h) + 1):
        for w_t in range(2, max_period_w + 1):
            phase_map: Dict[Tuple[int, int], int] = {}
            valid = True
            for r in range(h):
                start = offsets[r]
                for c in range(start, w):
                    val = int(grid[r, c])
                    if val == MARKER:
                        continue
                    phase = (r % h_t, (c - start) % w_t)
                    prev = phase_map.get(phase)
                    if prev is None:
                        phase_map[phase] = val
                    elif prev != val:
                        valid = False
                        break
                if not valid:
                    break
            if not valid or len(phase_map) != h_t * w_t:
                continue

            marker_rows, marker_cols = np.where(grid == MARKER)
            for r0, c0 in zip(marker_rows, marker_cols):
                phase = (int(r0) % h_t, (int(c0) - offsets[int(r0)]) % w_t)
                if phase not in phase_map:
                    valid = False
                    break
            if valid:
                candidates.append((h_t, w_t, phase_map))

    return candidates


def _gen_attempt() -> np.ndarray | None:
    h_t, w_t = _draw_period_pair()
    color_count = _draw_color_count()
    colors = _RNG.sample(COLOR_POOL, color_count)
    tile = _make_tile(h_t, w_t, colors)
    if tile is None:
        return None

    height = _RNG.randint(GRID_HEIGHT_RANGE[0], GRID_HEIGHT_RANGE[1])
    min_width = max(GRID_WIDTH_RANGE[0], height + 1, w_t + 6)
    max_width = GRID_WIDTH_RANGE[1]
    if min_width > max_width:
        return None
    width = _RNG.randint(min_width, max_width)

    base = _build_unsheared_pattern(tile, height, width)
    grid = _shear_pattern(base)

    marker_cells = _choose_marker_cells(grid, tile)
    if marker_cells is None:
        return None
    for r, c in marker_cells:
        grid[r, c] = MARKER

    expected = _expected_output(grid, tile)
    if not _generator_checks(grid, expected, tile, marker_cells):
        return None
    return grid


def _draw_period_pair() -> Tuple[int, int]:
    if not _PERIOD_BAG:
        _PERIOD_BAG.extend((h_t, w_t) for h_t in PERIOD_VALUES for w_t in PERIOD_VALUES)
        _RNG.shuffle(_PERIOD_BAG)
    return _PERIOD_BAG.pop()


def _draw_color_count() -> int:
    if not _COLOR_COUNT_BAG:
        _COLOR_COUNT_BAG.extend([2, 2, 3, 3, 4, 4])
        _RNG.shuffle(_COLOR_COUNT_BAG)
    return _COLOR_COUNT_BAG.pop()


def _make_tile(h_t: int, w_t: int, colors: Sequence[int]) -> np.ndarray | None:
    for _ in range(200):
        tile = np.array([_RNG.choices(colors, k=w_t) for _ in range(h_t)], dtype=int)
        if not _tile_uses_all_colors(tile, colors):
            continue
        if _minimal_period(tile) != (h_t, w_t):
            continue
        if not _tile_needs_both_axes(tile):
            continue
        return tile
    return None


def _tile_uses_all_colors(tile: np.ndarray, colors: Sequence[int]) -> bool:
    return set(map(int, tile.flatten())) == set(map(int, colors))


def _minimal_period(tile: np.ndarray) -> Tuple[int, int]:
    h, w = tile.shape
    best_h = h
    best_w = w

    for cand_h in range(1, h + 1):
        if all(np.array_equal(tile[r], tile[r % cand_h]) for r in range(h)):
            best_h = cand_h
            break
    for cand_w in range(1, w + 1):
        if all(np.array_equal(tile[:, c], tile[:, c % cand_w]) for c in range(w)):
            best_w = cand_w
            break
    return best_h, best_w


def _tile_needs_both_axes(tile: np.ndarray) -> bool:
    h, w = tile.shape
    row_patterns = {tuple(map(int, tile[r])) for r in range(h)}
    col_patterns = {tuple(map(int, tile[:, c])) for c in range(w)}
    if len(row_patterns) < 2 or len(col_patterns) < 2:
        return False
    if not any(len(set(map(int, tile[r]))) > 1 for r in range(h)):
        return False
    if not any(len(set(map(int, tile[:, c]))) > 1 for c in range(w)):
        return False
    return True


def _build_unsheared_pattern(tile: np.ndarray, height: int, width: int) -> np.ndarray:
    h_t, w_t = tile.shape
    rows = np.arange(height)[:, None] % h_t
    cols = np.arange(width)[None, :] % w_t
    return tile[rows, cols]


def _shear_pattern(base: np.ndarray) -> np.ndarray:
    h, w = base.shape
    out = np.zeros((h, w), dtype=int)
    for r in range(h):
        out[r, r:] = base[r, : w - r]
    return out


def _choose_marker_cells(
    sheared: np.ndarray, tile: np.ndarray
) -> List[Tuple[int, int]] | None:
    h, w = sheared.shape
    h_t, w_t = tile.shape
    phase_to_cells: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for r in range(h):
        for c in range(r, w):
            phase = (r % h_t, (c - r) % w_t)
            phase_to_cells.setdefault(phase, []).append((r, c))

    eligible_phases = [phase for phase, cells in phase_to_cells.items() if len(cells) >= 2]
    if len(eligible_phases) < 3:
        return None

    max_markers = min(MARKER_COUNT_RANGE[1], sum(len(cells) - 1 for cells in phase_to_cells.values()))
    if max_markers < MARKER_COUNT_RANGE[0]:
        return None
    marker_count = _RNG.randint(MARKER_COUNT_RANGE[0], max_markers)

    forced_marker = _find_shear_sensitive_marker(sheared, tile, phase_to_cells)
    if forced_marker is None:
        return None

    markers = [forced_marker]
    marked = {forced_marker}
    phase_remaining = {phase: len(cells) for phase, cells in phase_to_cells.items()}
    phase = (forced_marker[0] % h_t, (forced_marker[1] - forced_marker[0]) % w_t)
    phase_remaining[phase] -= 1

    candidates = [cell for cells in phase_to_cells.values() for cell in cells if cell != forced_marker]
    _RNG.shuffle(candidates)
    for cell in candidates:
        if len(markers) >= marker_count:
            break
        if cell in marked:
            continue
        phase = (cell[0] % h_t, (cell[1] - cell[0]) % w_t)
        if phase_remaining[phase] <= 1:
            continue
        markers.append(cell)
        marked.add(cell)
        phase_remaining[phase] -= 1

    if len(markers) < MARKER_COUNT_RANGE[0]:
        return None
    if len(markers) < marker_count:
        marker_count = len(markers)
        markers = markers[:marker_count]

    marker_phases = {(r % h_t, (c - r) % w_t) for r, c in markers}
    row_phases = {phase[0] for phase in marker_phases}
    col_phases = {phase[1] for phase in marker_phases}
    if len(marker_phases) < 3 or len(row_phases) < 2 or len(col_phases) < 2:
        return None
    if not any(r > 0 for r, _c in markers):
        return None
    return markers


def _find_shear_sensitive_marker(
    sheared: np.ndarray,
    tile: np.ndarray,
    phase_to_cells: Dict[Tuple[int, int], List[Tuple[int, int]]],
) -> Tuple[int, int] | None:
    h_t, w_t = tile.shape
    candidates: List[Tuple[int, int]] = []
    for phase, cells in phase_to_cells.items():
        if len(cells) < 2:
            continue
        for r, c in cells:
            if r == 0:
                continue
            true_color = int(tile[r % h_t, (c - r) % w_t])
            direct_color = int(tile[r % h_t, c % w_t])
            if true_color != direct_color:
                candidates.append((r, c))
    if not candidates:
        return None
    return _RNG.choice(candidates)


def _expected_output(masked: np.ndarray, tile: np.ndarray) -> np.ndarray:
    out = masked.copy()
    h_t, w_t = tile.shape
    marker_rows, marker_cols = np.where(masked == MARKER)
    for r0, c0 in zip(marker_rows, marker_cols):
        r = int(r0)
        c = int(c0)
        out[r, c] = tile[r % h_t, (c - r) % w_t]
    return out


def _generator_checks(
    grid: np.ndarray,
    expected: np.ndarray,
    tile: np.ndarray,
    marker_cells: Sequence[Tuple[int, int]],
) -> bool:
    try:
        offsets = _infer_offsets(grid)
    except AssertionError:
        return False

    if offsets != list(range(grid.shape[0])):
        return False

    candidates = _valid_period_candidates(grid, offsets)
    if not candidates:
        return False

    best_area = min(h_t * w_t for h_t, w_t, _phase_map in candidates)
    best = [cand for cand in candidates if cand[0] * cand[1] == best_area]
    truth_dims = tuple(map(int, tile.shape))
    if len(best) != 1:
        return False
    if best[0][:2] != truth_dims:
        return False

    truth_phases = {
        (r % truth_dims[0], c % truth_dims[1]): int(tile[r, c])
        for r in range(truth_dims[0])
        for c in range(truth_dims[1])
    }
    if best[0][2] != truth_phases:
        return False

    for r, c in marker_cells:
        if expected[r, c] == MARKER or expected[r, c] == BG:
            return False

    if not _all_tile_phases_visible(grid, truth_dims):
        return False
    if not _markers_force_unshear(tile, marker_cells):
        return False
    if not _markers_not_local_copies(grid, expected):
        return False

    try:
        actual = main(grid)
    except AssertionError:
        return False
    return np.array_equal(actual, expected)


def _all_tile_phases_visible(grid: np.ndarray, dims: Tuple[int, int]) -> bool:
    h_t, w_t = dims
    seen = set()
    for r in range(grid.shape[0]):
        for c in range(r, grid.shape[1]):
            if int(grid[r, c]) == MARKER:
                continue
            seen.add((r % h_t, (c - r) % w_t))
    return len(seen) == h_t * w_t


def _markers_force_unshear(
    tile: np.ndarray, marker_cells: Sequence[Tuple[int, int]]
) -> bool:
    h_t, w_t = tile.shape
    for r, c in marker_cells:
        if r == 0:
            continue
        true_color = int(tile[r % h_t, (c - r) % w_t])
        direct_color = int(tile[r % h_t, c % w_t])
        if true_color != direct_color:
            return True
    return False


def _markers_not_local_copies(grid: np.ndarray, expected: np.ndarray) -> bool:
    marker_rows, marker_cols = np.where(grid == MARKER)
    for r0, c0 in zip(marker_rows, marker_cols):
        r = int(r0)
        c = int(c0)
        target = int(expected[r, c])
        neighbor_colors = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr = r + dr
            cc = c + dc
            if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]:
                val = int(grid[rr, cc])
                if val not in (BG, MARKER):
                    neighbor_colors.append(val)
        if neighbor_colors and all(color != target for color in neighbor_colors):
            return True
    return False
