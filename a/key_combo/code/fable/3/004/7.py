from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


_ARC_UTILS = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS) not in sys.path:
    sys.path.append(str(_ARC_UTILS))

from arc_utils import get_bg


TRANSFORM_NAMES = ("rot90", "rot180", "rot270", "flip_h", "flip_v")
MAX_GENERATION_ATTEMPTS = 1000
PLAN_CYCLE = ((2, 4), (2, 5), (3, 5), (3, 6))
_GENERATE_CALLS = 0


def _apply_transform(tile: np.ndarray, name: str) -> np.ndarray:
    if name == "rot90":
        return np.rot90(tile, -1).copy()
    if name == "rot180":
        return np.rot90(tile, 2).copy()
    if name == "rot270":
        return np.rot90(tile, 1).copy()
    if name == "flip_h":
        return np.fliplr(tile).copy()
    if name == "flip_v":
        return np.flipud(tile).copy()
    raise AssertionError(f"unknown transform: {name}")


def _runs(rows: list[int]) -> list[tuple[int, int]]:
    if not rows:
        return []
    runs: list[tuple[int, int]] = []
    start = prev = int(rows[0])
    for r in rows[1:]:
        r = int(r)
        if r == prev + 1:
            prev = r
        else:
            runs.append((start, prev + 1))
            start = prev = r
    runs.append((start, prev + 1))
    return runs


def _candidate_bands_for_color(grid: np.ndarray, bg: int, color: int) -> list[tuple[int, int]] | None:
    h, _w = grid.shape
    sep_rows = [
        r
        for r in range(h)
        if int(grid[r, 0]) == int(color) and bool(np.all(grid[r] == int(color)))
    ]
    sep_runs = _runs(sep_rows)
    if not sep_runs:
        return None

    thicknesses = {b - a for a, b in sep_runs}
    if len(thicknesses) != 1:
        return None

    bands: list[tuple[int, int]] = []
    top = 0
    for a, b in sep_runs:
        if a <= top:
            return None
        bands.append((top, a))
        top = b
    if top >= h:
        return None
    bands.append((top, h))

    heights = {b - a for a, b in bands}
    if len(heights) != 1:
        return None
    if next(iter(heights)) <= 0 or len(bands) < 4:
        return None
    return bands


def _layout_is_valid(grid: np.ndarray, bands: list[tuple[int, int]], bg: int) -> bool:
    side = int(bands[0][1] - bands[0][0])
    width = int(grid.shape[1])
    if width < 2 * side:
        return False

    for i, (r0, r1) in enumerate(bands):
        band = grid[r0:r1, :]
        left = band[:, :side]
        gap = band[:, side : width - side]
        right = band[:, width - side :]

        if bool(np.all(left == int(bg))):
            return False
        if gap.size and not bool(np.all(gap == int(bg))):
            return False
        if i == len(bands) - 1:
            if not bool(np.all(right == int(bg))):
                return False
        elif bool(np.all(right == int(bg))):
            return False
    return True


def _split_bands(grid: np.ndarray, bg: int) -> list[tuple[int, int]]:
    full_row_colors = {
        int(grid[r, 0])
        for r in range(grid.shape[0])
        if int(grid[r, 0]) != int(bg) and bool(np.all(grid[r] == int(grid[r, 0])))
    }
    candidates: list[list[tuple[int, int]]] = []
    for color in full_row_colors:
        bands = _candidate_bands_for_color(grid, bg, color)
        if bands is not None and _layout_is_valid(grid, bands, bg):
            candidates.append(bands)

    assert candidates, "expected full-width separator rows that define equal-height bands"
    max_count = max(len(bands) for bands in candidates)
    candidates = [bands for bands in candidates if len(bands) == max_count]
    assert len(candidates) == 1, "separator rows must determine a unique band layout"
    return candidates[0]


def _tiles_from_bands(grid: np.ndarray, bands: list[tuple[int, int]]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    side = int(bands[0][1] - bands[0][0])
    width = int(grid.shape[1])
    lefts: list[np.ndarray] = []
    rights: list[np.ndarray] = []
    for r0, r1 in bands:
        band = grid[r0:r1, :]
        lefts.append(band[:, :side].copy())
        rights.append(band[:, width - side :].copy())
    return lefts, rights


def _detect_transform(left: np.ndarray, right: np.ndarray) -> str:
    matches = [
        name
        for name in TRANSFORM_NAMES
        if np.array_equal(_apply_transform(left, name), right)
    ]
    assert len(matches) == 1, "each completed tile pair must reveal exactly one transform"
    return matches[0]


def _infer_next_transform(seq: list[str]) -> str:
    assert len(seq) >= 3, "expected enough completed bands to reveal a cycle"
    periods: list[int] = []
    for period in range(1, len(seq)):
        if all(seq[i] == seq[i % period] for i in range(len(seq))):
            periods.append(period)
    assert periods, "completed transforms must form a repeated cycle"

    predictions = {seq[len(seq) % period] for period in periods}
    assert len(predictions) == 1, "observed cycle must predict a unique next transform"
    return predictions.pop()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the missing right tile by extrapolating the cycle of left-to-right tile transforms.

    Input structure:
    - A dominant background surrounds several horizontal bands separated by full-width non-background rows.
    - Each band contains a left square tile and a same-size right square tile, optionally with a background spacer between them.
    - In every band except the last, the right tile is exactly one rigid transform of the left tile: rotate 90/180/270 degrees, mirror horizontally, or mirror vertically.
    - The last band's right tile is all background, while its left tile is present.

    Rule:
    - Split the grid into bands using the separator rows and read the bands from top to bottom.
    - For each completed band, compare the left and right tiles and record the unique rigid transform symbol.
    - Infer the shortest repeated cycle in this transform-symbol sequence and take the transform that comes next in the cycle.
    - Apply that transform to the last band's left tile and return only the missing right tile.

    Key insight:
    - The extrapolation is not in the pixels of any one tile; it is in the meta-level sequence of transforms between left and right tiles across the ordered bands.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(get_bg(g))
    bands = _split_bands(g, bg)
    lefts, rights = _tiles_from_bands(g, bands)
    assert len(lefts) == len(rights) and len(lefts) >= 4

    observed: list[str] = []
    for left, right in zip(lefts[:-1], rights[:-1]):
        assert not bool(np.all(right == bg)), "completed bands must have non-empty right tiles"
        observed.append(_detect_transform(left, right))

    assert bool(np.all(rights[-1] == bg)), "last right tile must be empty"
    next_name = _infer_next_transform(observed)
    return _apply_transform(lefts[-1], next_name)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _is_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, pts[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _all_transform_forms_unique(tile: np.ndarray) -> bool:
    forms = [tile] + [_apply_transform(tile, name) for name in TRANSFORM_NAMES]
    return len({form.tobytes() for form in forms}) == len(forms)


def _has_constant_foreground_row(tile: np.ndarray, bg: int) -> bool:
    for row in tile:
        if int(row[0]) != int(bg) and bool(np.all(row == int(row[0]))):
            return True
    return False


def _random_tile(side: int, bg: int, color_pool: list[int]) -> np.ndarray | None:
    cells = [(r, c) for r in range(side) for c in range(side)]
    min_area = max(4, side + 1)
    max_area = min(side * side - 2, max(min_area, int(round(side * side * 0.62))))

    for _ in range(300):
        area = random.randint(min_area, max_area)
        coords = random.sample(cells, area)
        mask = np.zeros((side, side), dtype=bool)
        for r, c in coords:
            mask[r, c] = True
        if not _is_connected(mask):
            continue

        k = random.randint(2, min(4, len(color_pool), area))
        palette = random.sample(color_pool, k)
        assignments = palette[:] + [random.choice(palette) for _ in range(area - k)]
        random.shuffle(assignments)

        tile = np.full((side, side), int(bg), dtype=int)
        for (r, c), color in zip(coords, assignments):
            tile[r, c] = int(color)

        if _has_constant_foreground_row(tile, bg):
            continue
        if _all_transform_forms_unique(tile):
            return tile
    return None


def _sample_dimensions(period: int, observed_count: int) -> tuple[int, int, int] | None:
    band_count = observed_count + 1
    options: list[tuple[int, int, int]] = []
    for side in (3, 4, 5):
        for sep_thickness in (1, 2):
            for gap in (0, 1, 2):
                height = band_count * side + observed_count * sep_thickness
                width = 2 * side + gap
                if height <= 30 and width <= 30:
                    options.append((side, sep_thickness, gap))
    if not options:
        return None
    return random.choice(options)


def _render_case(
    side: int,
    sep_thickness: int,
    gap: int,
    bg: int,
    sep_color: int,
    cycle: list[str],
    observed_count: int,
    color_pool: list[int],
) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    band_count = observed_count + 1
    height = band_count * side + observed_count * sep_thickness
    width = 2 * side + gap
    grid = np.full((height, width), int(bg), dtype=int)

    left_tiles: list[np.ndarray] = []
    observed_rights: list[np.ndarray] = []
    row = 0
    for i in range(band_count):
        left = _random_tile(side, bg, color_pool)
        if left is None:
            return None
        left_tiles.append(left)

        grid[row : row + side, :side] = left
        transform = cycle[i % len(cycle)]
        right = _apply_transform(left, transform)
        if i < observed_count:
            grid[row : row + side, side + gap : side + gap + side] = right
            observed_rights.append(right)

        row += side
        if i < observed_count:
            grid[row : row + sep_thickness, :] = int(sep_color)
            row += sep_thickness

    expected_transform = cycle[observed_count % len(cycle)]
    expected = _apply_transform(left_tiles[-1], expected_transform)

    for previous in left_tiles[:-1] + observed_rights:
        if np.array_equal(previous, expected):
            return None

    expected_sequence = [cycle[i % len(cycle)] for i in range(observed_count)]
    return grid, expected, expected_sequence


def _gen_attempt(period: int, observed_count: int) -> np.ndarray | None:
    dims = _sample_dimensions(period, observed_count)
    if dims is None:
        return None
    side, sep_thickness, gap = dims

    colors = list(range(10))
    bg = random.choice(colors)
    sep_color = random.choice([c for c in colors if c != bg])
    color_pool = [c for c in colors if c not in (bg, sep_color)]

    cycle = random.sample(list(TRANSFORM_NAMES), period)
    rendered = _render_case(side, sep_thickness, gap, bg, sep_color, cycle, observed_count, color_pool)
    if rendered is None:
        return None
    grid, expected, expected_sequence = rendered

    if int(get_bg(grid)) != int(bg):
        return None
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return None

    try:
        bands = _split_bands(grid, bg)
        lefts, rights = _tiles_from_bands(grid, bands)
        observed = [_detect_transform(left, right) for left, right in zip(lefts[:-1], rights[:-1])]
        next_transform = _infer_next_transform(observed)
        solved = main(grid)
    except Exception:
        return None

    if observed != expected_sequence:
        return None
    if next_transform != cycle[observed_count % period]:
        return None
    if len(set(observed)) != period:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GENERATE_CALLS
    preferred = PLAN_CYCLE[_GENERATE_CALLS % len(PLAN_CYCLE)]
    _GENERATE_CALLS += 1

    for attempt in range(MAX_GENERATION_ATTEMPTS):
        period, observed_count = preferred if attempt < MAX_GENERATION_ATTEMPTS // 2 else random.choice(PLAN_CYCLE)
        grid = _gen_attempt(period, observed_count)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid transform-cycle tile puzzle")
