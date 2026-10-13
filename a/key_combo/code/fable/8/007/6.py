from __future__ import annotations

from collections import deque
from pathlib import Path
import sys

import numpy as np


ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import Color, bbox_coords, find_components


BLACK = Color.BLACK
RED = Color.RED
YELLOW = Color.YELLOW
TEAL = Color.TEAL

GEN_MAX_ATTEMPTS = 400
GEN_SIDE_RANGE = (20, 29)
GEN_MARKER_MARGIN = 7


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _yellow_marker(grid: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(grid == YELLOW)
    assert coords.shape == (4, 2), "expected exactly one 2x2 yellow marker"
    r0, c0, r1, c1 = bbox_coords(coords)
    assert (r1 - r0, c1 - c0) == (1, 1), "yellow marker must be a solid 2x2 block"
    assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == YELLOW)
    return int(r0), int(c0)


def _cross_erased(grid: np.ndarray, marker_r: int, marker_c: int) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    selected = np.zeros(out.shape, dtype=bool)
    selected[marker_r : marker_r + 2, :] = True
    selected[:, marker_c : marker_c + 2] = True
    out[selected & (out == TEAL)] = BLACK
    return out


def _nonyellow_components(grid: np.ndarray) -> list[np.ndarray]:
    work = np.asarray(grid, dtype=int).copy()
    work[work == YELLOW] = BLACK
    return find_components(work, bg=BLACK, connectivity=4, monochromatic=False, return_format="objects")


def _red_square_blocks(grid: np.ndarray, comp_mask: np.ndarray) -> list[set[tuple[int, int]]]:
    coords = np.argwhere(comp_mask)
    r0, c0, r1, c1 = bbox_coords(coords)
    found: set[frozenset[tuple[int, int]]] = set()
    for r in range(int(r0), int(r1)):
        for c in range(int(c0), int(c1)):
            block = {(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)}
            if all(comp_mask[rr, cc] and grid[rr, cc] == RED for rr, cc in block):
                found.add(frozenset(block))
    blocks = [set(block) for block in found]
    assert len(blocks) == 2, "each dumbbell must contain exactly two red 2x2 squares"
    blocks.sort(key=lambda block: min(block))
    assert blocks[0].isdisjoint(blocks[1]), "the two red squares must be distinct"
    return blocks


def _blocks_connected(grid: np.ndarray, comp_mask: np.ndarray, start_block: set[tuple[int, int]], goal_block: set[tuple[int, int]]) -> bool:
    passable = (grid != BLACK) & comp_mask
    q = deque()
    seen = np.zeros(grid.shape, dtype=bool)
    for r, c in start_block:
        if passable[r, c]:
            q.append((r, c))
            seen[r, c] = True
    goal = set(goal_block)
    while q:
        r, c = q.popleft()
        if (r, c) in goal:
            return True
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(grid.shape, nr, nc) and passable[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase teal bridge pixels on the yellow marker's row-column cross, then keep only dumbbells whose red end blocks remain connected.

    Input structure:
    - The grid has a black background and exactly one isolated 2x2 yellow marker.
    - Each other object is a dumbbell: two 2x2 red squares joined by a 4-connected one-cell-wide bridge made mostly of teal, with occasional red bridge cells.
    - Different dumbbells are separated from each other and from the yellow marker.

    Rule:
    - Take the two rows and two columns occupied by the yellow marker and treat their union as a cross-shaped erase zone.
    - Recolor every teal pixel in that cross to black, while leaving red pixels and the yellow marker unchanged.
    - For each original dumbbell, test after this erasure whether its two red 2x2 squares still lie in the same non-black 4-connected component; keep the modified dumbbell if they do, otherwise erase that entire dumbbell.

    Key insight:
    - The yellow block is not a filled rectangular eraser: only its two rows and two columns matter, and red bridge pixels can survive inside that cross, so connectivity must be checked after the selective erase.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    marker_r, marker_c = _yellow_marker(g)
    out = _cross_erased(g, marker_r, marker_c)

    for comp in _nonyellow_components(g):
        comp_mask = comp != BLACK
        left_block, right_block = _red_square_blocks(g, comp_mask)
        if not _blocks_connected(out, comp_mask, left_block, right_block):
            out[comp_mask] = BLACK

    return out


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if _in_bounds((h, w), nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _place_cells(grid: np.ndarray, blocked: np.ndarray, cells: dict[tuple[int, int], int]) -> bool:
    h, w = grid.shape
    coords = list(cells)
    if len(coords) != len(set(coords)):
        return False
    if not coords:
        return False
    for r, c in coords:
        if not _in_bounds((h, w), r, c) or blocked[r, c]:
            return False
    for (r, c), color in cells.items():
        grid[r, c] = int(color)
    for r, c in coords:
        for nr, nc in _neighbors8(r, c):
            if _in_bounds((h, w), nr, nc):
                blocked[nr, nc] = True
        blocked[r, c] = True
    return True


def _build_horizontal(
    sr: int,
    left_c: int,
    right_c: int,
    bridge_row: int,
    marker_cols: tuple[int, int],
    survive_cross: bool,
    rng: np.random.Generator,
    *,
    must_cross: bool,
) -> dict:
    cells: dict[tuple[int, int], int] = {}
    for rr in (sr, sr + 1):
        for cc in (left_c, left_c + 1, right_c, right_c + 1):
            cells[(rr, cc)] = RED

    bridge_cols = list(range(left_c + 2, right_c))
    crossing_cols = [c for c in bridge_cols if c in marker_cols]
    if must_cross and len(crossing_cols) != 2:
        return {}
    if not must_cross and crossing_cols:
        return {}
    red_bridge_cols = set()
    if crossing_cols:
        if survive_cross:
            red_bridge_cols.update(crossing_cols)
        else:
            keep_count = int(rng.integers(0, len(crossing_cols)))
            if keep_count:
                for c in rng.choice(crossing_cols, size=keep_count, replace=False):
                    red_bridge_cols.add(int(c))
    extra_candidates = [c for c in bridge_cols if c not in crossing_cols]
    extra_red_target = int(rng.integers(0, min(3, len(extra_candidates)) + 1))
    if extra_red_target:
        for c in rng.choice(extra_candidates, size=extra_red_target, replace=False):
            red_bridge_cols.add(int(c))
    for c in bridge_cols:
        cells[(bridge_row, c)] = RED if c in red_bridge_cols else TEAL
    if all(color == RED for (r, _c), color in cells.items() if r == bridge_row and (r, _c) not in {(sr, left_c), (sr, left_c + 1), (sr, right_c), (sr, right_c + 1), (sr + 1, left_c), (sr + 1, left_c + 1), (sr + 1, right_c), (sr + 1, right_c + 1)}):
        mid = bridge_cols[len(bridge_cols) // 2]
        cells[(bridge_row, mid)] = TEAL
    return {
        "cells": cells,
        "kind": "horizontal-cross" if must_cross else "horizontal-safe",
        "survive": bool(survive_cross if must_cross else True),
    }


def _build_vertical(
    sc: int,
    top_r: int,
    bottom_r: int,
    bridge_col: int,
    marker_rows: tuple[int, int],
    survive_cross: bool,
    rng: np.random.Generator,
    *,
    must_cross: bool,
) -> dict:
    cells: dict[tuple[int, int], int] = {}
    for rr in (top_r, top_r + 1, bottom_r, bottom_r + 1):
        for cc in (sc, sc + 1):
            cells[(rr, cc)] = RED

    bridge_rows = list(range(top_r + 2, bottom_r))
    crossing_rows = [r for r in bridge_rows if r in marker_rows]
    if must_cross and len(crossing_rows) != 2:
        return {}
    if not must_cross and crossing_rows:
        return {}
    red_bridge_rows = set()
    if crossing_rows:
        if survive_cross:
            red_bridge_rows.update(crossing_rows)
        else:
            keep_count = int(rng.integers(0, len(crossing_rows)))
            if keep_count:
                for r in rng.choice(crossing_rows, size=keep_count, replace=False):
                    red_bridge_rows.add(int(r))
    extra_candidates = [r for r in bridge_rows if r not in crossing_rows]
    extra_red_target = int(rng.integers(0, min(3, len(extra_candidates)) + 1))
    if extra_red_target:
        for r in rng.choice(extra_candidates, size=extra_red_target, replace=False):
            red_bridge_rows.add(int(r))
    for r in bridge_rows:
        cells[(r, bridge_col)] = RED if r in red_bridge_rows else TEAL
    if all(color == RED for (_r, c), color in cells.items() if c == bridge_col and (_r, c) not in {(top_r, sc), (top_r, sc + 1), (top_r + 1, sc), (top_r + 1, sc + 1), (bottom_r, sc), (bottom_r, sc + 1), (bottom_r + 1, sc), (bottom_r + 1, sc + 1)}):
        mid = bridge_rows[len(bridge_rows) // 2]
        cells[(mid, bridge_col)] = TEAL
    return {
        "cells": cells,
        "kind": "vertical-cross" if must_cross else "vertical-safe",
        "survive": bool(survive_cross if must_cross else True),
    }


def _try_horizontal_cross(
    h: int,
    w: int,
    marker_r: int,
    marker_c: int,
    blocked: np.ndarray,
    rng: np.random.Generator,
    survive: bool,
) -> dict | None:
    marker_cols = (marker_c, marker_c + 1)
    row_bands = []
    if marker_r >= 5:
        row_bands.append(range(2, marker_r - 1))
    if marker_r + 4 <= h - 3:
        row_bands.append(range(marker_r + 3, h - 3))
    if not row_bands:
        return None
    for _ in range(120):
        band = row_bands[int(rng.integers(len(row_bands)))]
        sr = int(rng.choice(list(band)))
        bridge_row = sr + int(rng.integers(0, 2))
        left_choices = list(range(marker_c - 6, marker_c - 3))
        right_choices = list(range(marker_c + 4, marker_c + 7))
        left_choices = [c for c in left_choices if c >= 1]
        right_choices = [c for c in right_choices if c + 1 < w]
        if not left_choices or not right_choices:
            return None
        left_c = int(rng.choice(left_choices))
        right_c = int(rng.choice(right_choices))
        obj = _build_horizontal(sr, left_c, right_c, bridge_row, marker_cols, survive, rng, must_cross=True)
        if obj and all(not blocked[r, c] for r, c in obj["cells"]):
            return obj
    return None


def _try_vertical_cross(
    h: int,
    w: int,
    marker_r: int,
    marker_c: int,
    blocked: np.ndarray,
    rng: np.random.Generator,
    survive: bool,
) -> dict | None:
    marker_rows = (marker_r, marker_r + 1)
    col_bands = []
    if marker_c >= 9:
        col_bands.append(range(1, marker_c - 7))
    if marker_c + 9 <= w - 2:
        col_bands.append(range(marker_c + 8, w - 2))
    if not col_bands:
        return None
    for _ in range(120):
        band = col_bands[int(rng.integers(len(col_bands)))]
        sc = int(rng.choice(list(band)))
        bridge_col = sc + int(rng.integers(0, 2))
        top_choices = [r for r in range(marker_r - 6, marker_r - 3) if r >= 1]
        bottom_choices = [r for r in range(marker_r + 4, marker_r + 7) if r + 1 < h]
        if not top_choices or not bottom_choices:
            return None
        top_r = int(rng.choice(top_choices))
        bottom_r = int(rng.choice(bottom_choices))
        obj = _build_vertical(sc, top_r, bottom_r, bridge_col, marker_rows, survive, rng, must_cross=True)
        if obj and all(not blocked[r, c] for r, c in obj["cells"]):
            return obj
    return None


def _try_safe_object(
    h: int,
    w: int,
    marker_r: int,
    marker_c: int,
    blocked: np.ndarray,
    rng: np.random.Generator,
) -> dict | None:
    marker_rows = (marker_r, marker_r + 1)
    marker_cols = (marker_c, marker_c + 1)
    for _ in range(200):
        if rng.random() < 0.5:
            top_band = [r for r in range(1, marker_r - 4)]
            bottom_band = [r for r in range(marker_r + 4, h - 2)]
            if not top_band and not bottom_band:
                continue
            sr = int(rng.choice(top_band if (not bottom_band or rng.random() < 0.5) else bottom_band))
            bridge_row = sr + int(rng.integers(0, 2))
            on_left = bool(rng.integers(0, 2))
            if on_left:
                left_candidates = [c for c in range(1, marker_c - 9)]
                if not left_candidates:
                    continue
                left_c = int(rng.choice(left_candidates))
                right_limit = min(marker_c - 4, left_c + 8)
                right_candidates = [c for c in range(left_c + 4, right_limit + 1) if c + 1 < w]
            else:
                left_candidates = [c for c in range(marker_c + 4, w - 9)]
                if not left_candidates:
                    continue
                left_c = int(rng.choice(left_candidates))
                right_limit = min(w - 3, left_c + 8)
                right_candidates = [c for c in range(left_c + 4, right_limit + 1) if c + 1 < w]
            if not right_candidates:
                continue
            right_c = int(rng.choice(right_candidates))
            obj = _build_horizontal(sr, left_c, right_c, bridge_row, marker_cols, True, rng, must_cross=False)
        else:
            left_band = [c for c in range(1, marker_c - 4)]
            right_band = [c for c in range(marker_c + 4, w - 2)]
            if not left_band and not right_band:
                continue
            sc = int(rng.choice(left_band if (not right_band or rng.random() < 0.5) else right_band))
            bridge_col = sc + int(rng.integers(0, 2))
            on_top = bool(rng.integers(0, 2))
            if on_top:
                top_candidates = [r for r in range(1, marker_r - 9)]
                if not top_candidates:
                    continue
                top_r = int(rng.choice(top_candidates))
                bottom_limit = min(marker_r - 4, top_r + 8)
                bottom_candidates = [r for r in range(top_r + 4, bottom_limit + 1) if r + 1 < h]
            else:
                top_candidates = [r for r in range(marker_r + 4, h - 9)]
                if not top_candidates:
                    continue
                top_r = int(rng.choice(top_candidates))
                bottom_limit = min(h - 3, top_r + 8)
                bottom_candidates = [r for r in range(top_r + 4, bottom_limit + 1) if r + 1 < h]
            if not bottom_candidates:
                continue
            bottom_r = int(rng.choice(bottom_candidates))
            obj = _build_vertical(sc, top_r, bottom_r, bridge_col, marker_rows, True, rng, must_cross=False)
        if not obj:
            continue
        if any(r in marker_rows or c in marker_cols for r, c in obj["cells"]):
            continue
        if all(not blocked[r, c] for r, c in obj["cells"]):
            return obj
    return None


def _oracle(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    marker_r, marker_c = _yellow_marker(g)
    out = _cross_erased(g, marker_r, marker_c)
    nonyellow = (g != BLACK) & (g != YELLOW)
    for coords in _component_coords(nonyellow):
        mask = np.zeros(g.shape, dtype=bool)
        mask[coords[:, 0], coords[:, 1]] = True
        blocks = _red_square_blocks(g, mask)
        if not _blocks_connected(out, mask, blocks[0], blocks[1]):
            out[mask] = BLACK
    return out


def _validate_generated(grid: np.ndarray, specs: list[dict]) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False
    try:
        marker_r, marker_c = _yellow_marker(grid)
    except Exception:
        return False

    cross_rows = {marker_r, marker_r + 1}
    cross_cols = {marker_c, marker_c + 1}
    nonyellow = (grid != BLACK) & (grid != YELLOW)
    comps = _component_coords(nonyellow)
    if len(comps) != 3:
        return False

    crossing_survive = 0
    crossing_broken = 0
    horizontal_count = 0
    vertical_count = 0
    erased = _cross_erased(grid, marker_r, marker_c)

    for coords in comps:
        mask = np.zeros(grid.shape, dtype=bool)
        mask[coords[:, 0], coords[:, 1]] = True
        try:
            blocks = _red_square_blocks(grid, mask)
        except Exception:
            return False

        all_twobytwo = 0
        r0, c0, r1, c1 = bbox_coords(coords)
        for r in range(int(r0), int(r1)):
            for c in range(int(c0), int(c1)):
                block = {(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)}
                if all(mask[rr, cc] for rr, cc in block):
                    all_twobytwo += 1
        if all_twobytwo != 2:
            return False

        block_a, block_b = blocks
        a_r0, a_c0, a_r1, a_c1 = bbox_coords(np.array(sorted(block_a), dtype=int))
        b_r0, b_c0, b_r1, b_c1 = bbox_coords(np.array(sorted(block_b), dtype=int))
        if a_r0 == b_r0:
            horizontal_count += 1
        elif a_c0 == b_c0:
            vertical_count += 1
        else:
            return False

        bridge_mask = mask.copy()
        for r, c in block_a | block_b:
            bridge_mask[r, c] = False
        bridge_coords = np.argwhere(bridge_mask)
        if bridge_coords.size == 0:
            return False
        if any(
            bridge_mask[r, c]
            and bridge_mask[r + 1, c]
            and bridge_mask[r, c + 1]
            and bridge_mask[r + 1, c + 1]
            for r in range(grid.shape[0] - 1)
            for c in range(grid.shape[1] - 1)
        ):
            return False

        crosses = any(r in cross_rows or c in cross_cols for r, c in bridge_coords)
        connected = _blocks_connected(erased, mask, block_a, block_b)
        if crosses and connected:
            crossing_survive += 1
        if crosses and not connected:
            crossing_broken += 1

    if horizontal_count < 1 or vertical_count < 1:
        return False
    if crossing_survive < 1 or crossing_broken < 1:
        return False
    kinds = {spec["kind"] for spec in specs}
    if kinds != {"horizontal-cross", "vertical-cross", "horizontal-safe"} and kinds != {"horizontal-cross", "vertical-cross", "vertical-safe"}:
        return False

    try:
        actual = main(grid)
        expected = _oracle(grid)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _attempt_generate(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(*GEN_SIDE_RANGE))
    w = int(rng.integers(*GEN_SIDE_RANGE))
    grid = np.zeros((h, w), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    marker_r = int(rng.integers(GEN_MARKER_MARGIN, h - GEN_MARKER_MARGIN - 1))
    marker_c = int(rng.integers(GEN_MARKER_MARGIN, w - GEN_MARKER_MARGIN - 1))
    marker_cells = {
        (marker_r, marker_c): YELLOW,
        (marker_r + 1, marker_c): YELLOW,
        (marker_r, marker_c + 1): YELLOW,
        (marker_r + 1, marker_c + 1): YELLOW,
    }
    if not _place_cells(grid, blocked, marker_cells):
        return None

    h_survive = bool(rng.integers(0, 2))
    v_survive = not h_survive

    specs: list[dict] = []
    horizontal = _try_horizontal_cross(h, w, marker_r, marker_c, blocked, rng, h_survive)
    if horizontal is None:
        return None
    if not _place_cells(grid, blocked, horizontal["cells"]):
        return None
    specs.append(horizontal)

    vertical = _try_vertical_cross(h, w, marker_r, marker_c, blocked, rng, v_survive)
    if vertical is None:
        return None
    if not _place_cells(grid, blocked, vertical["cells"]):
        return None
    specs.append(vertical)

    safe = _try_safe_object(h, w, marker_r, marker_c, blocked, rng)
    if safe is None:
        return None
    if not _place_cells(grid, blocked, safe["cells"]):
        return None
    specs.append(safe)

    if not _validate_generated(grid, specs):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cross-bridge puzzle")
