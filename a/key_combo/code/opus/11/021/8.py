from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
SWAP_COLORS = (BLUE, RED)
BRANCHES = ("central", "vertical", "horizontal", "central_vertical", "central_horizontal")
TARGET_REGION_COUNT = {
    "central": 1,
    "vertical": 2,
    "horizontal": 2,
    "central_vertical": 3,
    "central_horizontal": 3,
}
DECOR_SHAPES = (
    ((0, 0), (0, 1), (1, 1), (2, 1)),
    ((0, 0), (1, 0), (1, 1), (1, 2)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _tile_2x2(grid: np.ndarray) -> np.ndarray:
    return np.tile(grid, (2, 2))


def _black_regions(grid: np.ndarray) -> list[set[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    regions: list[set[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] != BLACK or seen[r, c]:
                continue
            region: set[tuple[int, int]] = set()
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                region.add((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == BLACK and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            regions.append(region)
    return regions


def _enclosed_black_regions(grid: np.ndarray) -> list[set[tuple[int, int]]]:
    h, w = grid.shape
    enclosed: list[set[tuple[int, int]]] = []
    for region in _black_regions(grid):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in region):
            continue
        enclosed.append(region)
    return enclosed


def _swap_on_enclosed_boundaries(canvas: np.ndarray) -> np.ndarray:
    out = canvas.copy()
    swap_mask = np.zeros(canvas.shape, dtype=bool)
    h, w = canvas.shape
    for region in _enclosed_black_regions(canvas):
        for r, c in region:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and canvas[nr, nc] in SWAP_COLORS:
                    swap_mask[nr, nc] = True
    blue_mask = swap_mask & (canvas == BLUE)
    red_mask = swap_mask & (canvas == RED)
    out[blue_mask] = RED
    out[red_mask] = BLUE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Tile the motif into a 2x2 block and swap red/blue only around emergent enclosed black regions.

    Input structure:
    - The input is a small black-background motif using only blue (1), red (2), and black (0).
    - The single motif itself contains no enclosed black 4-connected region.
    - Open foreground fragments are arranged so that repeating the motif in a 2x2 block creates new black cavities at the seams.

    Rule:
    - Build a 2H x 2W canvas by repeating the input motif in all four quadrants of a 2x2 layout.
    - Detect every 4-connected black region in that tiled canvas that cannot reach the outer canvas boundary.
    - For each such enclosed black region, find every blue/red foreground pixel that is 4-neighbor-adjacent to the region.
    - Swap only those adjacent foreground pixels: blue becomes red and red becomes blue.
    - Leave every other pixel unchanged.

    Key insight:
    - The swap scope is not marked in the input; it emerges only after tiling creates seam-closed black cavities that were impossible in the single motif.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(map(int, np.unique(g))) <= {BLACK, BLUE, RED}
    tiled = _tile_2x2(g)
    return _swap_on_enclosed_boundaries(tiled)


def _paint_projected_ring(tile: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    h, w = tile.shape
    phase = random.randint(0, 1)
    painted: list[tuple[int, int, int]] = []
    for i, (r, c) in enumerate(cells):
        tr, tc = r % h, c % w
        color = BLUE if (r + c + i + phase) % 2 == 0 else RED
        if tile[tr, tc] not in (BLACK, color):
            return False
        painted.append((tr, tc, color))
    for tr, tc, color in painted:
        tile[tr, tc] = color
    return True


def _add_central_feature(tile: np.ndarray) -> bool:
    h, w = tile.shape
    cells = [
        (r, c)
        for r in range(h - 2, h + 2)
        for c in range(w - 2, w + 2)
        if r in (h - 2, h + 1) or c in (w - 2, w + 1)
    ]
    return _paint_projected_ring(tile, cells)


def _add_vertical_feature(tile: np.ndarray, *, reserve_for_center: bool) -> bool:
    h, w = tile.shape
    limit = h - 4 if reserve_for_center else h - 3
    candidates: list[tuple[int, int]] = []
    for r0 in range(2, limit):
        max_r1 = min(r0 + 3, limit)
        for r1 in range(r0 + 1, max_r1 + 1):
            candidates.append((r0, r1))
    if not candidates:
        return False
    r0, r1 = random.choice(candidates)
    cells = [
        (r, c)
        for r in range(r0 - 1, r1 + 2)
        for c in range(w - 2, w + 2)
        if r in (r0 - 1, r1 + 1) or c in (w - 2, w + 1)
    ]
    return _paint_projected_ring(tile, cells)


def _add_horizontal_feature(tile: np.ndarray, *, reserve_for_center: bool) -> bool:
    h, w = tile.shape
    limit = w - 4 if reserve_for_center else w - 3
    candidates: list[tuple[int, int]] = []
    for c0 in range(2, limit):
        max_c1 = min(c0 + 3, limit)
        for c1 in range(c0 + 1, max_c1 + 1):
            candidates.append((c0, c1))
    if not candidates:
        return False
    c0, c1 = random.choice(candidates)
    cells = [
        (r, c)
        for r in range(h - 2, h + 2)
        for c in range(c0 - 1, c1 + 2)
        if r in (h - 2, h + 1) or c in (c0 - 1, c1 + 1)
    ]
    return _paint_projected_ring(tile, cells)


def _add_decoration(tile: np.ndarray) -> bool:
    h, w = tile.shape
    for _ in range(60):
        shape = random.choice(DECOR_SHAPES)
        sh = max(r for r, _ in shape) + 1
        sw = max(c for _, c in shape) + 1
        if h - sh - 2 < 1 or w - sw - 2 < 1:
            return False
        r0 = random.randint(1, h - sh - 2)
        c0 = random.randint(1, w - sw - 2)
        cells = [(r0 + dr, c0 + dc) for dr, dc in shape]
        if any(tile[r, c] != BLACK for r, c in cells):
            continue
        if any(r in (0, 1, h - 2, h - 1) or c in (0, 1, w - 2, w - 1) for r, c in cells):
            continue
        phase = random.randint(0, 1)
        for idx, (r, c) in enumerate(cells):
            tile[r, c] = BLUE if (idx + phase) % 2 == 0 else RED
        return True
    return False


def _region_boundary_cells(grid: np.ndarray, region: set[tuple[int, int]]) -> set[tuple[int, int]]:
    h, w = grid.shape
    boundary: set[tuple[int, int]] = set()
    for r, c in region:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] in SWAP_COLORS:
                boundary.add((nr, nc))
    return boundary


def _has_adjacent_opposite_colors(grid: np.ndarray, cells: set[tuple[int, int]]) -> bool:
    for r, c in cells:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells and grid[nr, nc] != grid[r, c]:
                return True
    return False


def _validate_generated(tile: np.ndarray, branch: str) -> bool:
    if np.count_nonzero(tile) < 15 or np.count_nonzero(tile) > 25:
        return False
    if _enclosed_black_regions(tile):
        return False

    tiled = _tile_2x2(tile)
    regions = _enclosed_black_regions(tiled)
    if len(regions) != TARGET_REGION_COUNT[branch]:
        return False

    swap_mask = np.zeros(tiled.shape, dtype=bool)
    for region in regions:
        if len(region) < 2:
            return False
        boundary = _region_boundary_cells(tiled, region)
        colors = {int(tiled[r, c]) for r, c in boundary}
        if colors != {BLUE, RED}:
            return False
        if not _has_adjacent_opposite_colors(tiled, boundary):
            return False
        for r, c in boundary:
            swap_mask[r, c] = True

    changed = int(np.count_nonzero(swap_mask))
    unchanged = int(np.count_nonzero((tiled > BLACK) & ~swap_mask))
    if changed < 4 or unchanged < 4:
        return False

    out = main(tile)
    if np.array_equal(out, tiled):
        return False
    return True


def _build_candidate(branch: str) -> np.ndarray | None:
    min_side = 8 if branch.startswith("central_") else 7
    h = random.randint(min_side, 10)
    w = random.randint(min_side, 10)
    tile = np.zeros((h, w), dtype=int)

    if branch == "central":
        ok = _add_central_feature(tile)
    elif branch == "vertical":
        ok = _add_vertical_feature(tile, reserve_for_center=False)
    elif branch == "horizontal":
        ok = _add_horizontal_feature(tile, reserve_for_center=False)
    elif branch == "central_vertical":
        ok = _add_central_feature(tile) and _add_vertical_feature(tile, reserve_for_center=True)
    else:
        ok = _add_central_feature(tile) and _add_horizontal_feature(tile, reserve_for_center=True)
    if not ok:
        return None

    for _ in range(8):
        if np.count_nonzero(tile) >= 15:
            break
        _add_decoration(tile)

    if not _validate_generated(tile, branch):
        return None
    return tile


def generate_input() -> np.ndarray:
    for _ in range(400):
        branch = random.choice(BRANCHES)
        tile = _build_candidate(branch)
        if tile is not None:
            return tile
    raise RuntimeError("failed to generate a valid tiled-cavity input")
