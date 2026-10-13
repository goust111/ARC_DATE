from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _UTILS_DIR.exists():
    utils_path = str(_UTILS_DIR)
    if utils_path not in sys.path:
        sys.path.insert(0, utils_path)

try:
    from arc_utils import Color, bbox, crop_bbox
except Exception:
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        YELLOW = 4
        GREY = 5

    def bbox(mask: np.ndarray):
        coords = np.argwhere(mask)
        if coords.size == 0:
            return None
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        return int(r0), int(c0), int(r1), int(c1)

    def crop_bbox(grid: np.ndarray, box):
        r0, c0, r1, c1 = map(int, box)
        return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


PALETTE = (Color.BLUE, Color.RED, Color.YELLOW)
LEFT_SIZES = (7, 9, 11)
TILE_DIMS = ((2, 2), (2, 3), (3, 2), (3, 3))
RIGHT_WIDTH_RANGE = (6, 9)
HOLE_FRACTION = (0.20, 0.35)
GEN_RETRY_LIMIT = 600


def _d4_orbit(r: int, c: int, n: int) -> set[tuple[int, int]]:
    last = n - 1
    return {
        (r, c),
        (r, last - c),
        (last - r, c),
        (last - r, last - c),
        (c, r),
        (c, last - r),
        (last - c, r),
        (last - c, last - r),
    }


def _find_separator(g: np.ndarray) -> int:
    grey_cols = np.flatnonzero(np.all(g == Color.GREY, axis=0))
    assert grey_cols.shape == (1,), "expected exactly one full grey separator column"
    sep = int(grey_cols[0])
    assert 0 < sep < g.shape[1] - 1
    return sep


def _repair_left_square(left: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    n = int(left.shape[0])
    assert left.shape == (n, n), "left region must be square"
    assert n % 2 == 1, "left square must have odd side length"

    missing = left == Color.BLACK
    missing_bbox = bbox(missing)
    assert missing_bbox is not None, "expected one black missing rectangle in the left square"
    r0, c0, r1, c1 = missing_bbox
    assert np.all(left[r0 : r1 + 1, c0 : c1 + 1] == Color.BLACK), "missing region must be solid"
    assert int(missing.sum()) == (r1 - r0 + 1) * (c1 - c0 + 1)

    repaired = left.copy()
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            values = {
                int(left[rr, cc])
                for rr, cc in _d4_orbit(r, c, n)
                if int(left[rr, cc]) != Color.BLACK
            }
            assert len(values) == 1, "each missing cell must have one visible D4 value"
            repaired[r, c] = values.pop()
    return repaired, crop_bbox(repaired, missing_bbox), missing_bbox


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair a periodic field by extracting its seed tile from a D4-symmetric patch completed on the left.

    Input structure:
    - A single full grey column splits the grid into a left odd-sized square and a right rectangular field.
    - The left square uses blue, red, and yellow and contains one solid black rectangular hole inside a D4-symmetric pattern.
    - The right field is a blue/red/yellow periodic region of the same height, but some cells are black holes.

    Rule:
    - Find the black rectangle in the left square and complete each missing cell from the unique non-black value in its D4 symmetry orbit.
    - Crop the repaired rectangle; this cropped patch is the seed tile.
    - In the right field, keep every non-black cell and replace each black hole by the seed-tile color at the same row/column residue modulo the seed-tile height and width.
    - Output only the repaired right field, excluding the left square and the grey separator.

    Key insight:
    - The left puzzle is not an independent output target: after D4 completion, its cropped missing rectangle becomes the exact periodic tile needed to repair the separate right-side pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep = _find_separator(g)
    left = g[:, :sep]
    right = g[:, sep + 1 :]
    assert left.shape[0] == left.shape[1], "left side must be a square"
    assert right.shape[0] == left.shape[0] and right.shape[1] > 0

    _repaired_left, tile, _missing_bbox = _repair_left_square(left)
    tile_h, tile_w = tile.shape

    out = right.copy()
    for r in range(right.shape[0]):
        for c in range(right.shape[1]):
            expected = int(tile[r % tile_h, c % tile_w])
            if int(right[r, c]) == Color.BLACK:
                out[r, c] = expected
            else:
                assert int(right[r, c]) == expected, "visible right pixels must already match the hidden tile"
    return out


def _tile_has_min_period(tile: np.ndarray) -> bool:
    h, w = tile.shape
    for ph in range(1, h + 1):
        if h % ph != 0:
            continue
        for pw in range(1, w + 1):
            if w % pw != 0:
                continue
            if ph == h and pw == w:
                continue
            base = tile[:ph, :pw]
            if np.array_equal(np.tile(base, (h // ph, w // pw)), tile):
                return False
    return True


def _repeat_tile(tile: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    tile_h, tile_w = tile.shape
    reps_h = (out_h + tile_h - 1) // tile_h
    reps_w = (out_w + tile_w - 1) // tile_w
    return np.tile(tile, (reps_h, reps_w))[:out_h, :out_w].copy()


def _rect_orbits_are_unique(n: int, r0: int, c0: int, rect_h: int, rect_w: int) -> bool:
    rect_cells = {(r, c) for r in range(r0, r0 + rect_h) for c in range(c0, c0 + rect_w)}
    for r, c in rect_cells:
        if len(_d4_orbit(r, c, n) & rect_cells) != 1:
            return False
    return True


def _rect_candidates(n: int, rect_h: int, rect_w: int) -> list[tuple[int, int]]:
    mid = n // 2
    candidates = []
    row_ranges = (
        range(0, mid - rect_h + 1),
        range(mid + 1, n - rect_h + 1),
    )
    col_ranges = (
        range(0, mid - rect_w + 1),
        range(mid + 1, n - rect_w + 1),
    )
    for rows in row_ranges:
        for cols in col_ranges:
            for r0 in rows:
                for c0 in cols:
                    if _rect_orbits_are_unique(n, r0, c0, rect_h, rect_w):
                        candidates.append((r0, c0))

    return candidates


def _sample_tile(rect_h: int, rect_w: int) -> np.ndarray | None:
    for _ in range(120):
        tile = np.array(random.choices(PALETTE, k=rect_h * rect_w), dtype=int).reshape(rect_h, rect_w)
        if len(set(map(int, tile.ravel()))) < 2:
            continue
        if not _tile_has_min_period(tile):
            continue
        return tile
    return None


def _build_left_pattern(n: int, r0: int, c0: int, tile: np.ndarray) -> np.ndarray:
    full = np.full((n, n), -1, dtype=int)
    tile_h, tile_w = tile.shape
    forced = {(r0 + r, c0 + c): int(tile[r, c]) for r in range(tile_h) for c in range(tile_w)}

    for r in range(n):
        for c in range(n):
            if full[r, c] != -1:
                continue
            orbit = _d4_orbit(r, c, n)
            orbit_forced = {forced[pos] for pos in orbit if pos in forced}
            if orbit_forced:
                assert len(orbit_forced) == 1
                color = orbit_forced.pop()
            else:
                color = random.choice(PALETTE)
            for rr, cc in orbit:
                full[rr, cc] = color

    assert np.all(full >= 0)
    return full


def _right_cells_by_residue(out_h: int, out_w: int, tile_h: int, tile_w: int):
    cells: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r in range(out_h):
        for c in range(out_w):
            cells.setdefault((r % tile_h, c % tile_w), []).append((r, c))
    return cells


def _sample_hole_mask(out_h: int, out_w: int, tile_h: int, tile_w: int) -> np.ndarray | None:
    cells_by_residue = _right_cells_by_residue(out_h, out_w, tile_h, tile_w)
    if any(len(cells) < 2 for cells in cells_by_residue.values()):
        return None

    total = out_h * out_w
    low = math.ceil(total * HOLE_FRACTION[0])
    high = math.floor(total * HOLE_FRACTION[1])
    if low > high:
        return None

    holes: set[tuple[int, int]] = set()
    hole_count_by_residue = {residue: 0 for residue in cells_by_residue}

    for residue, cells in cells_by_residue.items():
        cell = random.choice(cells)
        holes.add(cell)
        hole_count_by_residue[residue] += 1

    target = random.randint(max(low, len(holes)), high)
    available = [(r, c) for r in range(out_h) for c in range(out_w) if (r, c) not in holes]
    random.shuffle(available)

    for r, c in available:
        if len(holes) >= target:
            break
        residue = (r % tile_h, c % tile_w)
        class_size = len(cells_by_residue[residue])
        if hole_count_by_residue[residue] + 1 >= class_size:
            continue
        holes.add((r, c))
        hole_count_by_residue[residue] += 1

    if len(holes) != target:
        return None

    mask = np.zeros((out_h, out_w), dtype=bool)
    for r, c in holes:
        mask[r, c] = True

    for residue, cells in cells_by_residue.items():
        visible = sum(not mask[r, c] for r, c in cells)
        if visible < 1:
            return None
    return mask


def _full_is_d4_symmetric(full: np.ndarray) -> bool:
    n = full.shape[0]
    for r in range(n):
        for c in range(n):
            values = {int(full[rr, cc]) for rr, cc in _d4_orbit(r, c, n)}
            if len(values) != 1:
                return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(LEFT_SIZES)
    rect_h, rect_w = random.choice(TILE_DIMS)
    positions = _rect_candidates(n, rect_h, rect_w)
    if not positions:
        return None

    tile = _sample_tile(rect_h, rect_w)
    if tile is None:
        return None

    r0, c0 = random.choice(positions)
    left_full = _build_left_pattern(n, r0, c0, tile)
    if not _full_is_d4_symmetric(left_full):
        return None

    if not np.array_equal(left_full[r0 : r0 + rect_h, c0 : c0 + rect_w], tile):
        return None

    right_w = random.randint(max(RIGHT_WIDTH_RANGE[0], rect_w * 2), RIGHT_WIDTH_RANGE[1])
    right_full = _repeat_tile(tile, n, right_w)
    hole_mask = _sample_hole_mask(n, right_w, rect_h, rect_w)
    if hole_mask is None:
        return None

    right_in = right_full.copy()
    right_in[hole_mask] = Color.BLACK

    left_in = left_full.copy()
    left_in[r0 : r0 + rect_h, c0 : c0 + rect_w] = Color.BLACK
    grid = np.concatenate(
        [left_in, np.full((n, 1), Color.GREY, dtype=int), right_in],
        axis=1,
    )

    try:
        out = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(out, right_full):
        return None
    if out.shape != right_in.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle-10 input")
