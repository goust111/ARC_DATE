from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, bbox_coords, components_mask


Cell = tuple[int, int, int]
BBox = tuple[int, int, int, int]

RECT_SIZE = 3
MAX_GENERATION_ATTEMPTS = 400
AMBIGUOUS_PERIODS = (6, 8, 9)
OTHER_PERIODS = (5, 6, 7, 8, 9)
REPEAT_MIN = 3
REPEAT_MAX = 4
PALETTE = tuple(Color.NOT_BLACK)


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, int(n) + 1) if int(n) % d == 0]


def _minimal_exact_period(tile: np.ndarray) -> tuple[int, int]:
    h, w = map(int, tile.shape)
    best = (h, w)
    best_area = h * w
    for pr in _divisors(h):
        for pc in _divisors(w):
            candidate = np.tile(tile[:pr, :pc], (h // pr, w // pc))
            if not np.array_equal(candidate, tile):
                continue
            area = pr * pc
            if area < best_area or (area == best_area and (pr, pc) < best):
                best = (pr, pc)
                best_area = area
    return best


def _build_phase_table(cells: list[Cell], row_period: int, col_period: int) -> np.ndarray | None:
    table = np.full((int(row_period), int(col_period)), -1, dtype=int)
    for r, c, color in cells:
        tr = int(r) % int(row_period)
        tc = int(c) % int(col_period)
        known = int(table[tr, tc])
        if known == -1:
            table[tr, tc] = int(color)
        elif known != int(color):
            return None
    return table


def _apply_cells(table: np.ndarray, cells: list[Cell]) -> np.ndarray | None:
    filled = np.array(table, copy=True)
    pr, pc = map(int, filled.shape)
    for r, c, color in cells:
        tr = int(r) % pr
        tc = int(c) % pc
        known = int(filled[tr, tc])
        if known == -1:
            filled[tr, tc] = int(color)
        elif known != int(color):
            return None
    return filled


def _render_from_tile(tile: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = map(int, shape)
    rr = np.arange(h)[:, None] % int(tile.shape[0])
    cc = np.arange(w)[None, :] % int(tile.shape[1])
    return tile[rr, cc]


def _detect_rectangles(grid: np.ndarray) -> list[BBox]:
    black_components = components_mask(grid == Color.BLACK, connectivity=4)
    rectangles: list[BBox] = []
    for coords in black_components:
        r0, c0, r1, c1 = map(int, bbox_coords(coords))
        rectangles.append((r0, c0, r1, c1))
    rectangles.sort()
    return rectangles


def _partition_evidence(grid: np.ndarray, rectangles: list[BBox]) -> tuple[list[Cell], list[Cell], np.ndarray]:
    inside = np.zeros_like(grid, dtype=bool)
    for r0, c0, r1, c1 in rectangles:
        inside[r0 : r1 + 1, c0 : c1 + 1] = True

    outer_cells: list[Cell] = []
    survivor_cells: list[Cell] = []
    for r, c in np.argwhere(grid != Color.BLACK):
        cell = (int(r), int(c), int(grid[int(r), int(c)]))
        if inside[int(r), int(c)]:
            survivor_cells.append(cell)
        else:
            outer_cells.append(cell)
    return outer_cells, survivor_cells, inside


def _enumerate_visible_candidates(cells: list[Cell], shape: tuple[int, int]) -> list[tuple[int, int, np.ndarray]]:
    h, w = map(int, shape)
    candidates: list[tuple[int, int, np.ndarray]] = []
    for row_period in range(1, h + 1):
        for col_period in range(1, w + 1):
            table = _build_phase_table(cells, row_period, col_period)
            if table is not None:
                candidates.append((int(row_period), int(col_period), table))
    return candidates


def _enumerate_full_candidates(
    visible_candidates: list[tuple[int, int, np.ndarray]],
    survivor_cells: list[Cell],
) -> list[tuple[int, int, np.ndarray]]:
    full: list[tuple[int, int, np.ndarray]] = []
    for row_period, col_period, partial in visible_candidates:
        table = _apply_cells(partial, survivor_cells)
        if table is None or np.any(table == -1):
            continue
        if _minimal_exact_period(table) != (int(row_period), int(col_period)):
            continue
        full.append((int(row_period), int(col_period), table))
    return full


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the full wallpaper by using survivor pixels inside black rectangles to disambiguate the true period.

    Input structure:
    - The grid is a periodic tiling of one rectangular multicolor base tile.
    - Several solid black rectangles occlude parts of that wallpaper.
    - Some rectangle interiors contain isolated non-black survivor pixels from the hidden original pattern.

    Rule:
    - Treat black cells as unknown and detect the occluder rectangles from their black connected components.
    - Enumerate every row/column period compatible with the non-black pixels outside those rectangles.
    - Keep only the candidate whose phase table also matches the survivor pixels inside the rectangles, then tile it back across the whole grid.

    Key insight:
    - The outside-visible wallpaper is deliberately compatible with a smaller fake period; only the colored survivors trapped inside the black rectangles reveal which candidate period is the real one.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert np.any(g == Color.BLACK), "expected black occluders"

    rectangles = _detect_rectangles(g)
    assert rectangles, "expected at least one black rectangle"

    outer_cells, survivor_cells, _inside_mask = _partition_evidence(g, rectangles)
    assert outer_cells, "expected visible non-black pixels outside the occluders"
    assert survivor_cells, "expected survivor pixels inside the occluders"

    visible_candidates = _enumerate_visible_candidates(outer_cells, g.shape)
    assert len(visible_candidates) > 1, "visible region should admit multiple candidate periods"

    full_candidates = _enumerate_full_candidates(visible_candidates, survivor_cells)
    assert len(full_candidates) == 1, "survivor pixels should identify one unique true period"

    row_period, col_period, tile = full_candidates[0]
    assert np.all(tile != Color.BLACK), "black is reserved for occlusion only"

    out = _render_from_tile(tile, g.shape)
    assert np.all((g == Color.BLACK) | (g == out)), "every visible input pixel must agree with the restored wallpaper"
    assert (row_period, col_period) == _minimal_exact_period(tile)
    return out


def _sample_primitive_tile(height: int, width: int, palette: list[int]) -> np.ndarray:
    for _ in range(600):
        tile = np.random.choice(palette, size=(int(height), int(width))).astype(int)
        if len(set(map(int, tile.ravel()))) < 2:
            continue
        if _minimal_exact_period(tile) == (int(height), int(width)):
            return tile
    raise RuntimeError("failed to sample a primitive false-period tile")


def _sample_periods() -> tuple[tuple[int, int], tuple[int, int], str]:
    branch = random.choice(("row", "col", "both"))
    if branch == "row":
        true_period = (random.choice(AMBIGUOUS_PERIODS), random.choice(OTHER_PERIODS))
        false_period = (
            random.choice([d for d in _divisors(true_period[0]) if 1 < d < true_period[0]]),
            true_period[1],
        )
    elif branch == "col":
        true_period = (random.choice(OTHER_PERIODS), random.choice(AMBIGUOUS_PERIODS))
        false_period = (
            true_period[0],
            random.choice([d for d in _divisors(true_period[1]) if 1 < d < true_period[1]]),
        )
    else:
        true_period = (random.choice(AMBIGUOUS_PERIODS), random.choice(AMBIGUOUS_PERIODS))
        false_period = (
            random.choice([d for d in _divisors(true_period[0]) if 1 < d < true_period[0]]),
            random.choice([d for d in _divisors(true_period[1]) if 1 < d < true_period[1]]),
        )
    return true_period, false_period, branch


def _true_tile_and_patch() -> tuple[np.ndarray, tuple[int, int], tuple[int, int], str] | None:
    true_period, false_period, branch = _sample_periods()
    true_h, true_w = true_period
    false_h, false_w = false_period
    if true_h < 5 or true_w < 5:
        return None

    palette = random.sample(list(PALETTE), random.randint(3, 5))
    false_tile = _sample_primitive_tile(false_h, false_w, palette)
    true_tile = np.tile(false_tile, (true_h // false_h, true_w // false_w))

    rect_r0 = random.randint(1, true_h - RECT_SIZE - 1)
    rect_c0 = random.randint(1, true_w - RECT_SIZE - 1)
    for r in range(rect_r0, rect_r0 + RECT_SIZE):
        for c in range(rect_c0, rect_c0 + RECT_SIZE):
            choices = [color for color in PALETTE if int(color) != int(true_tile[r, c])]
            true_tile[r, c] = random.choice(choices)

    if _minimal_exact_period(true_tile) != true_period:
        return None
    return true_tile, false_period, (rect_r0, rect_c0), branch


def _validate_generated(
    inp: np.ndarray,
    out: np.ndarray,
    true_period: tuple[int, int],
    false_period: tuple[int, int],
    expected_rectangles: int,
) -> bool:
    try:
        rectangles = _detect_rectangles(inp)
        if len(rectangles) != int(expected_rectangles):
            return False
        if any((r1 - r0 + 1, c1 - c0 + 1) != (RECT_SIZE, RECT_SIZE) for r0, c0, r1, c1 in rectangles):
            return False

        outer_cells, survivor_cells, _inside = _partition_evidence(inp, rectangles)
        if not outer_cells or not survivor_cells:
            return False

        visible_candidates = _enumerate_visible_candidates(outer_cells, inp.shape)
        visible_map = {(pr, pc): table for pr, pc, table in visible_candidates}
        if len(visible_map) <= 1:
            return False
        if true_period not in visible_map or false_period not in visible_map:
            return False

        false_tile = visible_map[false_period]
        if np.any(false_tile == -1):
            return False
        if _minimal_exact_period(false_tile) != false_period:
            return False
        if np.array_equal(_render_from_tile(false_tile, inp.shape), out):
            return False

        full_candidates = _enumerate_full_candidates(visible_candidates, survivor_cells)
        if len(full_candidates) != 1 or full_candidates[0][:2] != true_period:
            return False
        if not np.array_equal(_render_from_tile(full_candidates[0][2], inp.shape), out):
            return False

        survivor_offsets = {
            (int(r) - r0, int(c) - c0)
            for r0, c0, r1, c1 in rectangles
            for r, c, _color in survivor_cells
            if r0 <= int(r) <= r1 and c0 <= int(c) <= c1
        }
        if survivor_offsets != {(dr, dc) for dr in range(RECT_SIZE) for dc in range(RECT_SIZE)}:
            return False

        return np.array_equal(main(inp), out)
    except Exception:
        return False


def _gen_attempt() -> np.ndarray | None:
    sampled = _true_tile_and_patch()
    if sampled is None:
        return None
    true_tile, false_period, (rect_r0, rect_c0), _branch = sampled
    true_period = tuple(map(int, true_tile.shape))

    max_repeat_h = min(REPEAT_MAX, 30 // true_period[0])
    max_repeat_w = min(REPEAT_MAX, 30 // true_period[1])
    if max_repeat_h < REPEAT_MIN or max_repeat_w < REPEAT_MIN:
        return None

    repeat_h = random.randint(REPEAT_MIN, max_repeat_h)
    repeat_w = random.randint(REPEAT_MIN, max_repeat_w)
    rectangle_count = repeat_h * repeat_w
    if rectangle_count < RECT_SIZE * RECT_SIZE:
        return None

    out = np.tile(true_tile, (repeat_h, repeat_w))
    inp = out.copy()

    tile_origins = [(tr * true_period[0], tc * true_period[1]) for tr in range(repeat_h) for tc in range(repeat_w)]
    for tile_r0, tile_c0 in tile_origins:
        rr = tile_r0 + rect_r0
        cc = tile_c0 + rect_c0
        inp[rr : rr + RECT_SIZE, cc : cc + RECT_SIZE] = Color.BLACK

    hidden_offsets = [(dr, dc) for dr in range(RECT_SIZE) for dc in range(RECT_SIZE)]
    random.shuffle(tile_origins)
    random.shuffle(hidden_offsets)
    for (tile_r0, tile_c0), (dr, dc) in zip(tile_origins, hidden_offsets):
        rr = tile_r0 + rect_r0 + dr
        cc = tile_c0 + rect_c0 + dc
        inp[rr, cc] = out[rr, cc]

    if not _validate_generated(inp, out, true_period, false_period, rectangle_count):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
