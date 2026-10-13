from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


def _ensure_arc_utils_path() -> None:
    here = Path(__file__).resolve()
    candidates: list[Path] = [Path("/home/lixinyu/snap/new/solvers")]
    if len(here.parents) > 6:
        candidates.append(here.parents[6] / "new" / "solvers")
    for candidate in candidates:
        if candidate.is_dir():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.append(candidate_str)
            return
    raise ImportError("unable to locate /home/lixinyu/snap/new/solvers for arc_utils import")


_ensure_arc_utils_path()

from arc_utils import Color, components_mask


GEN_ATTEMPTS = 400
FRAME_HEIGHT_CHOICES = (3, 4, 5)
GRID_HEIGHT_RANGE = (7, 9)
GRID_WIDTH_RANGE = (5, 8)
INPUT_COLORS = (1, 2, 3, 5, 6, 8, 9)


def _mirror_tile(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    return np.block([[g, np.fliplr(g)], [np.flipud(g), np.flipud(np.fliplr(g))]])


def _touches_outer_edge(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = map(int, shape)
    return bool(
        np.any(
            (coords[:, 0] == 0)
            | (coords[:, 0] == h - 1)
            | (coords[:, 1] == 0)
            | (coords[:, 1] == w - 1)
        )
    )


def _touches_tiling_seam(coords: np.ndarray, input_shape: tuple[int, int]) -> bool:
    h, w = map(int, input_shape)
    return bool(
        np.any(
            (coords[:, 0] == h - 1)
            | (coords[:, 0] == h)
            | (coords[:, 1] == w - 1)
            | (coords[:, 1] == w)
        )
    )


def _enclosed_black_regions(grid: np.ndarray) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    regions: list[np.ndarray] = []
    for coords in components_mask(g == Color.BLACK, connectivity=4):
        if not _touches_outer_edge(coords, g.shape):
            regions.append(coords)
    return regions


def _paint_region(grid: np.ndarray, coords: np.ndarray, color: int) -> None:
    grid[coords[:, 0], coords[:, 1]] = int(color)


def _u_right_coords(height: int, width: int, r0: int, r1: int, c0: int) -> set[tuple[int, int]]:
    coords: set[tuple[int, int]] = set()
    for c in range(c0, width):
        coords.add((r0, c))
        coords.add((r1, c))
    for r in range(r0, r1 + 1):
        coords.add((r, c0))
    return coords


def _edge_decorator_coords(height: int, width: int) -> list[tuple[int, int]]:
    if random.random() < 0.5:
        start = random.randint(0, height - 2)
        length = random.randint(2, min(height - start, 4))
        coords = [(start + i, 0) for i in range(length)]
        if width >= 3:
            arm_len = random.randint(1, 2)
            arm_row = random.choice((start, start + length - 1))
            coords.extend((arm_row, c) for c in range(1, arm_len + 1))
        return coords
    start = random.randint(0, width - 2)
    length = random.randint(2, min(width - start, 4))
    coords = [(0, start + i) for i in range(length)]
    if height >= 3:
        arm_len = random.randint(1, 2)
        arm_col = random.choice((start, start + length - 1))
        coords.extend((r, arm_col) for r in range(1, arm_len + 1))
    return coords


def _can_place_isolated(grid: np.ndarray, coords: set[tuple[int, int]] | list[tuple[int, int]]) -> bool:
    h, w = grid.shape
    coord_set = set(coords)
    for r, c in coord_set:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if grid[r, c] != Color.BLACK:
            return False
    for r, c in coord_set:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in coord_set and grid[nr, nc] != Color.BLACK:
                    return False
    return True


def _place_coords(grid: np.ndarray, coords: set[tuple[int, int]] | list[tuple[int, int]], color: int) -> None:
    for r, c in set(coords):
        grid[r, c] = int(color)


def _has_edge_foreground(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    return bool(
        np.any(g[0, :] != Color.BLACK)
        or np.any(g[-1, :] != Color.BLACK)
        or np.any(g[:, 0] != Color.BLACK)
        or np.any(g[:, -1] != Color.BLACK)
    )


def _validate_generated_input(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        return False
    colors = sorted(set(map(int, g.flatten())) - {Color.BLACK})
    if len(colors) not in (2, 3):
        return False
    if not _has_edge_foreground(g):
        return False
    if _enclosed_black_regions(g):
        return False

    tiled = _mirror_tile(g)
    regions = _enclosed_black_regions(tiled)
    if len(regions) < 2:
        return False
    if not all(_touches_tiling_seam(region, g.shape) for region in regions):
        return False

    areas = [int(region.shape[0]) for region in regions]
    if len(set(areas)) < 2:
        return False

    # With 2x2 seam mirroring and 4-connected black regions, every emergent region that
    # becomes closed through a seam is duplicated across that seam. The prompt's request
    # for mixed odd/even emergent areas is therefore unsatisfiable under the stated rules.
    if any(area % 2 for area in areas):
        return False

    expected = tiled.copy()
    for region in regions:
        fill = Color.YELLOW if int(region.shape[0]) % 2 == 0 else Color.ORANGE
        _paint_region(expected, region, fill)
    return np.array_equal(main(g), expected)


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    grid = np.full((height, width), Color.BLACK, dtype=int)

    palette_size = random.choice((2, 3))
    palette = random.sample(INPUT_COLORS, palette_size)

    placed = False
    for _ in range(80):
        h1 = random.choice(FRAME_HEIGHT_CHOICES)
        h2 = random.choice(FRAME_HEIGHT_CHOICES)
        gap = random.randint(1, 2)
        total = h1 + h2 + gap
        if total > height:
            continue

        top = random.randint(0, height - total)
        r0_a, r1_a = top, top + h1 - 1
        r0_b, r1_b = r1_a + gap + 1, r1_a + gap + h2

        if width < 5:
            continue
        c0_a = random.randint(0, width - 3)
        c0_b = random.randint(0, width - 3)

        area_a = 2 * (width - c0_a - 1) * (h1 - 2)
        area_b = 2 * (width - c0_b - 1) * (h2 - 2)
        if area_a == area_b:
            continue

        candidate = np.full((height, width), Color.BLACK, dtype=int)
        coords_a = _u_right_coords(height, width, r0_a, r1_a, c0_a)
        coords_b = _u_right_coords(height, width, r0_b, r1_b, c0_b)
        if not _can_place_isolated(candidate, coords_a):
            continue
        _place_coords(candidate, coords_a, palette[0])
        if not _can_place_isolated(candidate, coords_b):
            continue
        _place_coords(candidate, coords_b, palette[1])
        grid = candidate
        placed = True
        break

    if not placed:
        return None

    if palette_size == 3:
        for _ in range(20):
            deco = _edge_decorator_coords(height, width)
            if _can_place_isolated(grid, deco):
                _place_coords(grid, deco, palette[2])
                break

    grid = np.rot90(grid, random.randrange(4)).copy()
    return grid if _validate_generated_input(grid) else None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror-tile the input into a 2x2 reflection mosaic, then fill each emergent enclosed black region by area parity.

    Input structure:
    - The input is a small black-background tile whose non-black pixels form one or more open-edged motifs.
    - The tile itself contains no 4-connected enclosed black hole.
    - Some foreground pixels touch the tile boundary so mirror seams can change background connectivity.

    Rule:
    - Build a 2x2 mirror tiling: top-left = original, top-right = left-right flip, bottom-left = up-down flip, bottom-right = both flips.
    - In the tiled grid, find every 4-connected black component that does not touch the outer border.
    - Fill each enclosed black component yellow when its area is even, or orange when its area is odd.
    - Keep all original non-black foreground pixels unchanged, and leave border-connected black background black.

    Key insight:
    - The black regions to fill are not properties of the source tile; they emerge only after the mirror copies meet along the tiling seams.
    """
    tiled = _mirror_tile(np.asarray(grid, dtype=int))
    out = tiled.copy()
    for region in _enclosed_black_regions(tiled):
        fill = Color.YELLOW if int(region.shape[0]) % 2 == 0 else Color.ORANGE
        _paint_region(out, region, fill)
    return out


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 10")
