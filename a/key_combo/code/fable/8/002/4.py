from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import Color


BG = int(Color.BLACK)
PALETTE = [int(c) for c in Color.NOT_BLACK]
MAX_ATTEMPTS = 400
SHAPE_OPTIONS = [
    (2, 3),
    (2, 4),
    (2, 5),
    (3, 2),
    (3, 3),
    (3, 4),
    (4, 2),
    (4, 3),
    (4, 4),
]


def _sample_points(grid: np.ndarray) -> list[tuple[int, int, int]]:
    coords = np.argwhere(grid != BG)
    return [(int(r), int(c), int(grid[r, c])) for r, c in coords]


def _candidate_tile(
    samples: list[tuple[int, int, int]], h: int, w: int
) -> np.ndarray | None:
    if h <= 0 or w <= 0 or h * w > len(samples):
        return None

    tile = np.zeros((h, w), dtype=int)
    seen = np.zeros((h, w), dtype=bool)

    for r, c, color in samples:
        rr = r % h
        cc = c % w
        if seen[rr, cc]:
            if int(tile[rr, cc]) != int(color):
                return None
        else:
            tile[rr, cc] = int(color)
            seen[rr, cc] = True

    if not bool(seen.all()):
        return None
    return tile


def _find_minimal_period(grid: np.ndarray) -> tuple[int, int, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    samples = _sample_points(g)
    assert samples, "grid must contain sample pixels"

    H, W = g.shape
    for h in range(1, H + 1):
        for w in range(1, W + 1):
            tile = _candidate_tile(samples, h, w)
            if tile is not None:
                return h, w, tile
    raise AssertionError("no valid fully covered period found")


def _tile_canvas(tile: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = tile.shape
    H, W = shape
    rows = np.arange(H) % h
    cols = np.arange(W) % w
    return tile[rows[:, None], cols[None, :]].copy()


def _minimal_tile_period(tile: np.ndarray) -> tuple[int, int]:
    h, w = tile.shape
    rows = np.arange(h)
    cols = np.arange(w)
    for hh in range(1, h + 1):
        for ww in range(1, w + 1):
            reduced = tile[rows % hh][:, cols % ww]
            if np.array_equal(tile, reduced):
                return hh, ww
    raise AssertionError("tile period search failed")


def _is_isolated(mask: np.ndarray, r: int, c: int) -> bool:
    r0 = max(0, r - 1)
    r1 = min(mask.shape[0], r + 2)
    c0 = max(0, c - 1)
    c1 = min(mask.shape[1], c + 2)
    return not bool(mask[r0:r1, c0:c1].any())


def _choose_covering_samples(H: int, W: int, h: int, w: int) -> list[tuple[int, int]] | None:
    classes = [(rr, cc) for rr in range(h) for cc in range(w)]
    candidates: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for rr, cc in classes:
        coords = [(r, c) for r in range(rr, H, h) for c in range(cc, W, w)]
        random.shuffle(coords)
        candidates[(rr, cc)] = coords

    order = sorted(classes, key=lambda key: len(candidates[key]))
    blocked = np.zeros((H, W), dtype=bool)
    chosen: list[tuple[int, int]] = []

    def backtrack(idx: int) -> bool:
        if idx == len(order):
            return True

        rr, cc = order[idx]
        for r, c in candidates[(rr, cc)]:
            if not _is_isolated(blocked, r, c):
                continue

            prev = blocked.copy()
            blocked[r, c] = True
            chosen.append((r, c))
            if backtrack(idx + 1):
                return True
            chosen.pop()
            blocked[:, :] = prev
        return False

    if not backtrack(0):
        return None
    return chosen


def _add_extra_samples(
    chosen: list[tuple[int, int]], H: int, W: int, h: int, w: int
) -> list[tuple[int, int]]:
    blocked = np.zeros((H, W), dtype=bool)
    for r, c in chosen:
        blocked[r, c] = True

    coords = [(r, c) for r in range(H) for c in range(W)]
    random.shuffle(coords)
    extra_budget = random.randint(max(1, len(chosen) // 4), max(1, len(chosen) // 2))

    for r, c in coords:
        if extra_budget == 0:
            break
        if blocked[r, c]:
            continue
        if not _is_isolated(blocked, r, c):
            continue
        chosen.append((r, c))
        blocked[r, c] = True
        extra_budget -= 1
    return chosen


def _random_primitive_tile(h: int, w: int) -> np.ndarray | None:
    palette_size = random.randint(2, min(4, h * w, len(PALETTE)))
    colors = random.sample(PALETTE, palette_size)

    for _ in range(80):
        flat = np.empty(h * w, dtype=int)
        flat[:palette_size] = np.array(colors, dtype=int)
        flat[palette_size:] = np.random.choice(colors, size=h * w - palette_size, replace=True)
        np.random.shuffle(flat)
        tile = flat.reshape(h, w)
        if _minimal_tile_period(tile) == (h, w):
            return tile
    return None


def _build_input_from_tile(tile: np.ndarray, shape: tuple[int, int]) -> np.ndarray | None:
    H, W = shape
    h, w = tile.shape
    points = _choose_covering_samples(H, W, h, w)
    if points is None:
        return None
    points = _add_extra_samples(points, H, W, h, w)

    full = _tile_canvas(tile, shape)
    inp = np.full((H, W), BG, dtype=int)
    for r, c in points:
        inp[r, c] = int(full[r, c])
    return inp


def _valid_periods(grid: np.ndarray) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    samples = _sample_points(g)
    H, W = g.shape
    out: list[tuple[int, int]] = []
    for h in range(1, H + 1):
        for w in range(1, W + 1):
            if _candidate_tile(samples, h, w) is not None:
                out.append((h, w))
    return out


def _samples_cover_period(grid: np.ndarray, h: int, w: int) -> bool:
    seen = np.zeros((h, w), dtype=bool)
    for r, c, _color in _sample_points(grid):
        seen[r % h, c % w] = True
    return bool(seen.all())


def _is_valid_instance(grid: np.ndarray, tile: np.ndarray) -> bool:
    H, W = grid.shape
    h, w = tile.shape
    samples = _sample_points(grid)
    if len(samples) < h * w:
        return False
    if len(samples) > (H * W) // 4:
        return False
    if not _samples_cover_period(grid, h, w):
        return False

    for r, c, _ in samples:
        r0 = max(0, r - 1)
        r1 = min(H, r + 2)
        c0 = max(0, c - 1)
        c1 = min(W, c + 2)
        neighborhood = (grid[r0:r1, c0:c1] != BG).sum()
        if neighborhood != 1:
            return False

    valid_periods = _valid_periods(grid)
    if not valid_periods or valid_periods[0] != (h, w):
        return False

    recovered_h, recovered_w, recovered_tile = _find_minimal_period(grid)
    if (recovered_h, recovered_w) != (h, w):
        return False
    if not np.array_equal(recovered_tile, tile):
        return False

    expected = _tile_canvas(tile, grid.shape)
    if not np.array_equal(main(grid), expected):
        return False
    if BG in np.unique(expected):
        return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the smallest hidden 2D period from sparse sample pixels and tile it over the full canvas.

    Input structure:
    - The grid is mostly black, with isolated non-black pixels scattered across the canvas.
    - Each colored sample pixel comes from one position of a hidden non-black periodic tile.
    - For the true tile, every residue class modulo its height and width is sampled at least once.

    Rule:
    - Search for the smallest row period and column period whose modulo classes assign one consistent color to every sampled pixel.
    - Reject candidate periods unless all of their residue classes are actually covered by samples.
    - Reconstruct the hidden tile from those residue-class colors, then repeat that tile across the whole canvas.

    Key insight:
    - The colored pixels are not separate objects; they are sparse observations of one unseen periodic pattern, so the task is solved by congruence-class alignment rather than local shape completion.
    """
    g = np.asarray(grid, dtype=int)
    h, w, tile = _find_minimal_period(g)
    return _tile_canvas(tile, g.shape)


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        h, w = random.choice(SHAPE_OPTIONS)
        tile = _random_primitive_tile(h, w)
        if tile is None:
            continue

        min_H = max(10, h * 3)
        min_W = max(10, w * 3)
        H = random.randint(min_H, 30)
        W = random.randint(min_W, 30)
        if H < h or W < w:
            continue

        inp = _build_input_from_tile(tile, (H, W))
        if inp is None:
            continue
        if _is_valid_instance(inp, tile):
            return inp
    raise RuntimeError("failed to generate a valid periodic-sampling ARC input")
