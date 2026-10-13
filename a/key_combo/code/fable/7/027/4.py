from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, components_mask, get_bg


MAX_GENERATION_ATTEMPTS = 400
PALETTE = [color for color in Color.NOT_BLACK]
BLOCK_SHAPES = ((1, 1), (1, 2), (2, 1), (2, 2), (1, 3))


def _row_color(row: np.ndarray, bg: int) -> int:
    colors = {int(value) for value in row if int(value) != bg}
    assert len(colors) == 1, "each row must expose exactly one band color"
    return colors.pop()


def _split_bands(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(get_bg(g))
    assert bg == Color.BLACK, "expected black background"

    row_colors = [_row_color(g[r], bg) for r in range(g.shape[0])]
    runs: list[tuple[int, int, int]] = []
    start = 0
    current = row_colors[0]
    for r in range(1, g.shape[0]):
        if row_colors[r] != current:
            runs.append((start, r, current))
            start = r
            current = row_colors[r]
    runs.append((start, g.shape[0], current))

    heights = {r1 - r0 for r0, r1, _color in runs}
    assert len(runs) >= 2, "expected multiple horizontal bands"
    assert len(heights) == 1, "expected equal-height bands"

    band_colors = [color for _r0, _r1, color in runs]
    assert len(set(band_colors)) == len(band_colors), "band colors must be distinct"

    bands = []
    for r0, r1, color in runs:
        band = g[r0:r1].copy()
        assert np.all((band == bg) | (band == color)), "each band must be a monochrome mask"
        bands.append({"color": int(color), "grid": band})
    return bands


def _count_holes(band: np.ndarray, color: int) -> int:
    mask = np.asarray(band == int(color), dtype=bool)
    assert mask.any(), "band must contain its foreground color"
    holes = 0
    height, width = mask.shape
    for coords in components_mask(~mask, connectivity=4):
        touches_border = np.any(
            (coords[:, 0] == 0)
            | (coords[:, 0] == height - 1)
            | (coords[:, 1] == 0)
            | (coords[:, 1] == width - 1)
        )
        if not touches_border:
            holes += 1
    return int(holes)


def _overlay_by_order(bands: list[np.ndarray], order: list[int]) -> np.ndarray:
    out = np.full_like(bands[0], Color.BLACK)
    for index in order:
        band = bands[index]
        mask = (band != Color.BLACK) & (out == Color.BLACK)
        out[mask] = band[mask]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Collapse stacked monochrome mask bands by ordering them with their enclosed-hole counts.

    Input structure:
    - The input is split into equal-height horizontal bands on a black background.
    - Each band is a same-size single-color mask, and every band color is distinct.
    - A band may enclose zero or more 4-connected black holes inside its mask.

    Rule:
    - Count, for each band, how many black background components are fully enclosed by that band's mask.
    - Sort the bands from more holes to fewer holes; the larger hole count gets higher overlay priority.
    - At each local coordinate, scan bands from top priority to low priority and output the first non-black color; if all are black, output black.

    Key insight:
    - The layer order is not the vertical input order: it is encoded by each mask's topology, so the solver must count enclosed holes before compositing corresponding pixels.
    """
    parsed = _split_bands(np.asarray(grid, dtype=int))
    bands = [entry["grid"] for entry in parsed]
    hole_counts = [_count_holes(entry["grid"], entry["color"]) for entry in parsed]
    assert len(set(hole_counts)) == len(hole_counts), "hole counts must be unique"
    priority = sorted(range(len(bands)), key=lambda idx: hole_counts[idx], reverse=True)
    return _overlay_by_order(bands, priority)


def _paint_ring(band: np.ndarray, top: int, left: int, color: int) -> None:
    band[top : top + 3, left : left + 3] = color
    band[top + 1, left + 1] = Color.BLACK


def _place_hole_layout(
    band: np.ndarray,
    color: int,
    holes: int,
    side: str,
    rng: np.random.Generator,
) -> None:
    if holes == 0:
        return

    height, width = band.shape
    feasible_cols = []
    for cols in (1, 2):
        rows = math.ceil(holes / cols)
        req_h = rows * 3 + (rows - 1)
        req_w = cols * 3 + (cols - 1)
        usable_left = 2 if side == "left" else 0
        usable_right = width - 2 if side == "right" else width
        if req_h <= height and usable_left + req_w <= usable_right:
            feasible_cols.append(cols)
    assert feasible_cols, "band too small for requested hole count"

    cols = int(rng.choice(feasible_cols))
    rows = math.ceil(holes / cols)
    req_h = rows * 3 + (rows - 1)
    req_w = cols * 3 + (cols - 1)
    top = int(rng.integers(0, height - req_h + 1))

    if side == "left":
        min_left = 2
        max_left = width - req_w
    else:
        min_left = 0
        max_left = width - 2 - req_w
    left = int(rng.integers(min_left, max_left + 1))

    for index in range(holes):
        rr = top + (index // cols) * 4
        cc = left + (index % cols) * 4
        _paint_ring(band, rr, cc, color)

    for index in range(holes):
        rr = top + (index // cols) * 4
        cc = left + (index % cols) * 4
        outward: list[tuple[int, int]] = []
        if rr > 0 and index // cols == 0:
            outward.append((rr - 1, cc + 1))
        if rr + 3 < height and index // cols == rows - 1:
            outward.append((rr + 3, cc + 1))
        if cc > 0 and index % cols == 0:
            outward.append((rr + 1, cc - 1))
        if cc + 3 < width and index % cols == cols - 1:
            outward.append((rr + 1, cc + 3))
        if outward and rng.random() < 0.65:
            r, c = outward[int(rng.integers(0, len(outward)))]
            band[r, c] = color


def _place_zero_hole_shape(
    band: np.ndarray,
    color: int,
    side: str,
    rng: np.random.Generator,
) -> None:
    height, width = band.shape
    rows = rng.choice(np.arange(height), size=int(rng.integers(2, min(5, height) + 1)), replace=False)
    for row in rows:
        length = int(rng.integers(2, max(3, width)))
        if side == "left":
            band[int(row), 0:min(width, length)] = color
        else:
            band[int(row), max(0, width - length):width] = color

    rect_h = int(rng.integers(2, min(4, height) + 1))
    rect_w = int(rng.integers(2, min(5, width) + 1))
    top = int(rng.integers(0, height - rect_h + 1))
    if side == "left":
        band[top : top + rect_h, :rect_w] = color
    else:
        band[top : top + rect_h, width - rect_w : width] = color


def _try_add_extras(
    band: np.ndarray,
    color: int,
    target_holes: int,
    side: str,
    rng: np.random.Generator,
) -> None:
    height, width = band.shape
    attempts = int(rng.integers(4, 10))
    for _ in range(attempts):
        block_h, block_w = BLOCK_SHAPES[int(rng.integers(0, len(BLOCK_SHAPES)))]
        if block_h > height or block_w > width:
            continue
        r0 = int(rng.integers(0, height - block_h + 1))
        c_min = 1 if side == "left" else 0
        c_max = width - block_w - (1 if side == "right" else 0)
        if c_max < c_min:
            continue
        c0 = int(rng.integers(c_min, c_max + 1))
        patch = band[r0 : r0 + block_h, c0 : c0 + block_w]
        if np.all(patch != Color.BLACK):
            continue
        candidate = band.copy()
        candidate[r0 : r0 + block_h, c0 : c0 + block_w] = color
        if _count_holes(candidate, color) == target_holes:
            band[:, :] = candidate


def _make_band(
    height: int,
    width: int,
    color: int,
    holes: int,
    side: str,
    rng: np.random.Generator,
) -> np.ndarray:
    band = np.full((height, width), Color.BLACK, dtype=int)
    spine_col = 0 if side == "left" else width - 1
    band[:, spine_col] = color

    if holes == 0:
        _place_zero_hole_shape(band, color, side, rng)
    else:
        _place_hole_layout(band, color, holes, side, rng)

    _try_add_extras(band, color, holes, side, rng)
    assert _count_holes(band, color) == holes
    assert np.all(np.any(band != Color.BLACK, axis=1)), "every row must expose the band color"
    return band


def _priority_order(hole_counts: list[int]) -> list[int]:
    return sorted(range(len(hole_counts)), key=lambda idx: hole_counts[idx], reverse=True)


def _sample_hole_counts(band_count: int, rng: np.random.Generator) -> list[int]:
    while True:
        counts = list(map(int, rng.choice(np.arange(5), size=band_count, replace=False)))
        if max(counts) > 0:
            return counts


def _choose_band_height(band_count: int, max_holes: int, rng: np.random.Generator) -> int:
    min_height = 7 if max_holes >= 3 else 6
    max_height = min(10, 30 // band_count)
    if max_holes == 4:
        min_height = max(min_height, 7)
    assert min_height <= max_height
    return int(rng.integers(min_height, max_height + 1))


def _choose_width(rng: np.random.Generator) -> int:
    return int(rng.integers(11, 19))


def _semantic_checks(grid: np.ndarray, source_counts: list[int]) -> bool:
    parsed = _split_bands(grid)
    bands = [entry["grid"] for entry in parsed]
    hole_counts = [_count_holes(entry["grid"], entry["color"]) for entry in parsed]
    if sorted(hole_counts) != sorted(source_counts):
        return False
    if len(set(hole_counts)) != len(hole_counts):
        return False

    priority = _priority_order(hole_counts)
    if priority == list(range(len(bands))):
        return False

    expected = _overlay_by_order(bands, priority)
    naive = _overlay_by_order(bands, list(range(len(bands))))
    if np.array_equal(expected, naive):
        return False

    areas = [int(np.count_nonzero(band != Color.BLACK)) for band in bands]
    area_priority = sorted(range(len(bands)), key=lambda idx: areas[idx], reverse=True)
    if area_priority == priority:
        return False

    if np.array_equal(expected, _overlay_by_order(bands, area_priority)):
        return False

    visible_colors = {int(value) for value in expected.ravel() if int(value) != Color.BLACK}
    if len(visible_colors) < 2:
        return False

    return np.array_equal(expected, main(grid))


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    band_count = int(rng.integers(3, 5))
    hole_counts = _sample_hole_counts(band_count, rng)
    band_height = _choose_band_height(band_count, max(hole_counts), rng)
    width = _choose_width(rng)
    side = "left" if rng.random() < 0.5 else "right"
    colors = list(map(int, rng.choice(PALETTE, size=band_count, replace=False)))

    bands = [
        _make_band(band_height, width, colors[index], hole_counts[index], side, rng)
        for index in range(band_count)
    ]

    priority = _priority_order(hole_counts)
    order = list(map(int, rng.permutation(band_count)))
    if order[0] == priority[0]:
        return None

    stacked = np.vstack([bands[index] for index in order]).astype(int)
    if max(stacked.shape) > 30:
        return None
    if not _semantic_checks(stacked, [hole_counts[index] for index in order]):
        return None
    return stacked


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hole-priority overlay sample")
