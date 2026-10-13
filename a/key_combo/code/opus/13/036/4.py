from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
BLUE = 1
RED = 2
INPUT_COLORS = (4, 5, 6, 7, 8, 9)
GEN_ATTEMPTS = 800
INNER_HEIGHT_ATTEMPTS = 60


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_to_foreground(grid: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(grid != BG)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _single_foreground_color(grid: np.ndarray) -> int:
    colors = sorted(set(map(int, grid.flat)) - {BG})
    assert len(colors) == 1
    return int(colors[0])


def _stack_wave(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = np.full((4 * h - 3, w), BG, dtype=int)
    flipped = np.flipud(g)
    for k, patch in enumerate((g, flipped, g, flipped)):
        r0 = k * (h - 1)
        block = out[r0 : r0 + h]
        block[patch != BG] = patch[patch != BG]
    return out


def _enclosed_black_regions(grid: np.ndarray) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    regions: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if g[r, c] != BG or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            touches_edge = False
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                if cr == 0 or cr == h - 1 or cc == 0 or cc == w - 1:
                    touches_edge = True
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and g[nr, nc] == BG and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            if not touches_edge:
                regions.append(np.array(pts, dtype=int))
    return regions


def _paint_pockets_by_parity(stacked: np.ndarray) -> np.ndarray:
    out = np.array(stacked, copy=True)
    for region in _enclosed_black_regions(stacked):
        fill = RED if int(region.shape[0]) % 2 == 1 else BLUE
        out[region[:, 0], region[:, 1]] = fill
    return out


def _column_intervals(grid: np.ndarray) -> tuple[list[int], list[int]] | None:
    color = _single_foreground_color(grid)
    top: list[int] = []
    bottom: list[int] = []
    for c in range(grid.shape[1]):
        rows = np.flatnonzero(grid[:, c] == color)
        if rows.size == 0:
            return None
        if rows[-1] - rows[0] + 1 != rows.size:
            return None
        top.append(int(rows[0]))
        bottom.append(int(rows[-1]))
    return top, bottom


def main(grid: np.ndarray) -> np.ndarray:
    """Stack the wave four times, then recolor each emergent enclosed black pocket by area parity.

    Input structure:
    - A black background contains exactly one monochrome 4-connected wave-shaped foreground object.
    - The wave spans the full effective width of its cropped bounding box and has no pre-existing enclosed black holes.
    - Its upper and lower boundaries both undulate so that alternating vertical flips can interlock.

    Rule:
    - Crop to the wave's minimal bounding box and call that pattern P.
    - Build a taller canvas by pasting P, flipud(P), P, flipud(P) from top to bottom, with each adjacent pair sharing exactly one row.
    - In the stacked result, find every 4-connected black region that cannot reach any canvas edge through black cells; fill odd-area pockets red and even-area pockets blue.

    Key insight:
    - The colored pockets do not exist in the input at all; they emerge only after alternating flipped copies of the same wave are overlapped by one row.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert np.any(g != BG)
    pattern = _crop_to_foreground(g)
    _single_foreground_color(pattern)
    stacked = _stack_wave(pattern)
    return _paint_pockets_by_parity(stacked)


def _random_walk_wave(height: int | None = None) -> np.ndarray | None:
    h = random.randint(5, 8) if height is None else int(height)
    w = random.randint(8, 14)

    top = [random.randint(0, h - 3)]
    thickness = [random.randint(2, min(5, h - top[0]))]
    for _ in range(1, w):
        next_top = max(0, min(h - 3, top[-1] + random.choice((-1, 0, 1))))
        lo = max(2, thickness[-1] - 1)
        hi = min(5, thickness[-1] + 1, h - next_top)
        if lo > hi:
            lo, hi = 2, min(5, h - next_top)
        if lo > hi:
            return None
        top.append(next_top)
        thickness.append(random.randint(lo, hi))

    bottom = [top[c] + thickness[c] - 1 for c in range(w)]
    if min(top) != 0 or max(bottom) != h - 1:
        return None
    if max(top) - min(top) < 2 or max(bottom) - min(bottom) < 2:
        return None

    color = random.choice(INPUT_COLORS)
    grid = np.full((h, w), BG, dtype=int)
    for c in range(w):
        grid[top[c] : bottom[c] + 1, c] = color
    return grid


def _bouncing_wave(height: int | None = None) -> np.ndarray | None:
    h = random.randint(5, 8) if height is None else int(height)
    w = random.randint(12, 14) if h >= 7 else random.randint(8, 14)

    top = [0]
    top_dir = 1
    for _ in range(1, w):
        if top[-1] <= 0:
            top_dir = 1
        elif top[-1] >= h - 3:
            top_dir = -1
        step = top_dir if random.random() < 0.75 else 0
        top.append(max(0, min(h - 3, top[-1] + step)))

    if max(top) - min(top) < 2:
        return None

    thickness = [random.randint(2, min(5, h - top[0]))]
    thick_dir = 1 if random.random() < 0.5 else -1
    for c in range(1, w):
        upper = min(5, h - top[c])
        if thickness[-1] <= 2:
            thick_dir = 1
        elif thickness[-1] >= upper:
            thick_dir = -1
        step = thick_dir if random.random() < 0.7 else 0
        candidate = max(2, min(upper, thickness[-1] + step))
        thickness.append(candidate)

    bottom = [top[c] + thickness[c] - 1 for c in range(w)]
    if max(bottom) != h - 1 or max(bottom) - min(bottom) < 2:
        return None

    color = random.choice(INPUT_COLORS)
    grid = np.full((h, w), BG, dtype=int)
    for c in range(w):
        grid[top[c] : bottom[c] + 1, c] = color
    return grid


def _random_orientation(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, copy=True)
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    return out


def _wave_has_required_input_structure(grid: np.ndarray) -> bool:
    if grid.ndim != 2:
        return False
    h, w = grid.shape
    if not (5 <= h <= 8 and 8 <= w <= 14):
        return False

    fg_mask = grid != BG
    if not fg_mask.any():
        return False

    colors = sorted(set(map(int, grid.flat)) - {BG})
    if len(colors) != 1 or colors[0] in (BLUE, RED):
        return False

    if len(_component_coords(fg_mask)) != 1:
        return False

    intervals = _column_intervals(grid)
    if intervals is None:
        return False
    top, bottom = intervals
    if max(top) - min(top) < 2 or max(bottom) - min(bottom) < 2:
        return False

    if _enclosed_black_regions(grid):
        return False
    return True


def _stacked_wave_is_semantically_valid(grid: np.ndarray) -> bool:
    stacked = _stack_wave(grid)
    pockets = _enclosed_black_regions(stacked)
    if len(pockets) < 2:
        return False

    areas = [int(region.shape[0]) for region in pockets]
    if len(set(areas)) < 2:
        return False
    if not any(area % 2 == 1 for area in areas):
        return False
    if not any(area % 2 == 0 for area in areas):
        return False

    out = main(grid)
    if out.shape != stacked.shape:
        return False
    if np.array_equal(out, stacked):
        return False

    for region in pockets:
        area = int(region.shape[0])
        expected = RED if area % 2 == 1 else BLUE
        vals = out[region[:, 0], region[:, 1]]
        if not np.all(vals == expected):
            return False

    keep_mask = stacked != BG
    if not np.array_equal(out[keep_mask], stacked[keep_mask]):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    target_height = random.randint(5, 8)
    for _ in range(INNER_HEIGHT_ATTEMPTS):
        builder = _random_walk_wave if random.random() < 0.5 else _bouncing_wave
        grid = builder(target_height)
        if grid is None:
            continue
        grid = _random_orientation(grid)
        if not _wave_has_required_input_structure(grid):
            continue
        if not _stacked_wave_is_semantically_valid(grid):
            continue
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_ATTEMPTS} attempts")
