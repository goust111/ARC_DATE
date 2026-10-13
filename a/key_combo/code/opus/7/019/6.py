from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.exists():
    sys.path.append(str(_ARC_UTILS_DIR))

try:
    from arc_utils import Color, components_mask  # type: ignore
except Exception:
    class Color:
        BLACK = 0
        BLUE = 1
        ORANGE = 7
        TEAL = 8

    def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
        if int(connectivity) != 4:
            raise ValueError("fallback components_mask only supports 4-connectivity")
        m = np.asarray(mask, dtype=bool)
        h, w = m.shape
        seen = np.zeros((h, w), dtype=bool)
        parts: list[np.ndarray] = []
        for r in range(h):
            for c in range(w):
                if not m[r, c] or seen[r, c]:
                    continue
                stack = [(r, c)]
                seen[r, c] = True
                coords: list[tuple[int, int]] = []
                while stack:
                    rr, cc = stack.pop()
                    coords.append((rr, cc))
                    for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                        if 0 <= nr < h and 0 <= nc < w and m[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
                parts.append(np.array(coords, dtype=int))
        return parts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress the sparse separator grid, then recolor the unique smallest and largest blue regions in the compressed domain.

    Input structure:
    - The grid has even height and width, with black background.
    - Every odd-index row and every odd-index column is entirely black, so only even/even intersections may contain blue pixels.
    - All foreground pixels are blue and appear visually isolated in the sparse original grid.

    Rule:
    - Delete all odd-index rows and columns, equivalently keeping only cells at even/even coordinates.
    - In the compressed grid, find all 4-connected blue regions and measure their areas there.
    - Recolor the unique smallest compressed blue region orange and the unique largest compressed blue region teal.
    - Leave every other blue region blue and keep black cells black.

    Key insight:
    - Connectivity must be recomputed after compression: pixels that are isolated in the original sparse grid can become adjacent only in the compressed coordinate system.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h % 2 == 0 and w % 2 == 0

    compact = g[::2, ::2].copy()
    regions = components_mask(compact == Color.BLUE, connectivity=4)
    assert len(regions) >= 3

    areas = np.array([int(region.shape[0]) for region in regions], dtype=int)
    smallest = np.flatnonzero(areas == areas.min())
    largest = np.flatnonzero(areas == areas.max())
    assert smallest.size == 1 and largest.size == 1

    out = compact.copy()
    small_region = regions[int(smallest[0])]
    large_region = regions[int(largest[0])]
    out[small_region[:, 0], small_region[:, 1]] = Color.ORANGE
    out[large_region[:, 0], large_region[:, 1]] = Color.TEAL
    return out


def _neighbors4(shape: tuple[int, int], r: int, c: int) -> list[tuple[int, int]]:
    h, w = shape
    out = []
    for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _guard_mask(shape: tuple[int, int], coords: set[tuple[int, int]]) -> np.ndarray:
    blocked = np.zeros(shape, dtype=bool)
    for r, c in coords:
        blocked[r, c] = True
        for nr, nc in _neighbors4(shape, r, c):
            blocked[nr, nc] = True
    return blocked


def _grow_component(
    shape: tuple[int, int],
    blocked: np.ndarray,
    area: int,
    rng: np.random.Generator,
) -> set[tuple[int, int]] | None:
    free = np.argwhere(~blocked)
    if free.size == 0:
        return None

    seed_r, seed_c = map(int, free[int(rng.integers(len(free)))])
    cells = {(seed_r, seed_c)}
    frontier = set(_neighbors4(shape, seed_r, seed_c))
    frontier = {cell for cell in frontier if not blocked[cell]}

    while len(cells) < int(area):
        if not frontier:
            return None
        options = list(frontier)
        weights = []
        for r, c in options:
            degree = sum((nbr in frontier) or (nbr in cells) for nbr in _neighbors4(shape, r, c))
            weights.append(max(1, degree))
        pick = int(rng.choice(len(options), p=np.array(weights, dtype=float) / np.sum(weights)))
        cell = options[pick]
        frontier.remove(cell)
        cells.add(cell)
        for nbr in _neighbors4(shape, cell[0], cell[1]):
            if nbr not in cells and not blocked[nbr]:
                frontier.add(nbr)

    return cells


def _build_compact_grid(
    h: int,
    w: int,
    areas: list[int],
    rng: np.random.Generator,
) -> np.ndarray | None:
    compact = np.zeros((h, w), dtype=int)
    occupied: set[tuple[int, int]] = set()

    for area in sorted(areas, reverse=True):
        blocked = _guard_mask((h, w), occupied)
        cells = _grow_component((h, w), blocked, int(area), rng)
        if cells is None:
            return None
        for cell in cells:
            occupied.add(cell)
            compact[cell] = Color.BLUE

    return compact


def _expand_sparse(compact: np.ndarray) -> np.ndarray:
    h, w = compact.shape
    sparse = np.zeros((2 * h, 2 * w), dtype=int)
    sparse[::2, ::2] = compact
    return sparse


def _valid_structure(inp: np.ndarray) -> bool:
    if inp.ndim != 2:
        return False
    h, w = inp.shape
    if h % 2 or w % 2 or max(inp.shape) > 30:
        return False
    if not np.all(inp[1::2, :] == Color.BLACK):
        return False
    if not np.all(inp[:, 1::2] == Color.BLACK):
        return False
    return set(map(int, np.unique(inp))) <= {Color.BLACK, Color.BLUE}


def _semantic_checks(inp: np.ndarray) -> bool:
    if not _valid_structure(inp):
        return False

    compact = inp[::2, ::2]
    regions = components_mask(compact == Color.BLUE, connectivity=4)
    if len(regions) < 3:
        return False

    areas = [int(region.shape[0]) for region in regions]
    if len(set(areas)) != len(areas):
        return False
    if sum(area > 1 for area in areas) < 2:
        return False

    out = main(inp)
    if out.shape != compact.shape:
        return False

    small_area = min(areas)
    large_area = max(areas)
    orange_regions = components_mask(out == Color.ORANGE, connectivity=4)
    teal_regions = components_mask(out == Color.TEAL, connectivity=4)
    blue_regions = components_mask(out == Color.BLUE, connectivity=4)
    if len(orange_regions) != 1 or len(teal_regions) != 1:
        return False
    if int(orange_regions[0].shape[0]) != small_area:
        return False
    if int(teal_regions[0].shape[0]) != large_area:
        return False
    if len(blue_regions) != len(regions) - 2:
        return False
    if not set(map(int, np.unique(out))) <= {Color.BLACK, Color.BLUE, Color.ORANGE, Color.TEAL}:
        return False
    return True


def _sample_areas(k: int, rng: np.random.Generator) -> list[int]:
    while True:
        upper = 6 if k >= 5 else 8
        areas = sorted(map(int, rng.choice(np.arange(1, upper + 1), size=k, replace=False).tolist()))
        if sum(area > 1 for area in areas) >= 2 and max(areas) - min(areas) >= 2:
            rng.shuffle(areas)
            return list(map(int, areas))


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(500):
        compact_h = int(rng.integers(6, 13))
        compact_w = int(rng.integers(6, 13))
        n_components = int(rng.integers(3, 6))
        areas = _sample_areas(n_components, rng)

        compact = _build_compact_grid(compact_h, compact_w, areas, rng)
        if compact is None:
            continue
        inp = _expand_sparse(compact)
        if _semantic_checks(inp):
            return inp
    raise RuntimeError("failed to generate a valid compressed-connectivity ARC input")
