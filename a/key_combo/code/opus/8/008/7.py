from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import Color, components_mask, neighbors4


BG = Color.BLACK
WALL = Color.BLUE
OBJ = Color.GREEN
GLYPH = Color.GREY
BORDER = Color.ORANGE

GLYPH_PATTERNS = {
    "h": np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    "v": np.array([[0, 1, 0], [0, 1, 0], [1, 1, 1]], dtype=bool),
    "hv": np.array([[1, 0, 1], [1, 1, 1], [0, 1, 0]], dtype=bool),
}


def _find_glyph(grid: np.ndarray) -> tuple[str, tuple[int, int]]:
    h, w = grid.shape
    corners = [(0, 0), (0, w - 3), (h - 3, 0), (h - 3, w - 3)]
    for r0, c0 in corners:
        if r0 < 0 or c0 < 0:
            continue
        window = grid[r0 : r0 + 3, c0 : c0 + 3]
        if window.shape != (3, 3):
            continue
        mask = window == GLYPH
        for mode, pattern in GLYPH_PATTERNS.items():
            if np.array_equal(mask, pattern):
                return mode, (r0, c0)
    raise AssertionError("no valid 3x3 corner glyph found")


def _mirror_coords(coords: np.ndarray, shape: tuple[int, int], mode: str) -> np.ndarray:
    h, w = shape
    mirrored = coords.copy()
    if "h" in mode:
        mirrored[:, 0] = h - 1 - mirrored[:, 0]
    if "v" in mode:
        mirrored[:, 1] = w - 1 - mirrored[:, 1]
    return mirrored


def _structural_wall_mask(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    structural = np.zeros_like(grid, dtype=bool)
    for comp in components_mask(grid == WALL, connectivity=4):
        cells = [(int(r), int(c)) for r, c in comp]
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in cells):
            for r, c in cells:
                structural[r, c] = True
    return structural


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Decode a corner glyph into reflection axes, mirror the green objects, erase decorative walls, and draw an inner border where objects face structural walls.

    Input structure:
    - Black background contains blue wall fragments, where only border-touching 4-connected components are structural.
    - One to three irregular green 4-connected objects sit on one side of the grid midpoint.
    - A single 3x3 grey glyph sits in a corner and encodes horizontal, vertical, or double reflection.

    Rule:
    - Read the grey glyph shape to decide whether to reflect the green pixels across the horizontal midline, vertical midline, or both.
    - Keep the original green objects and add their mirrored copies in green.
    - Remove blue wall components that do not touch the grid border, then keep only the structural blue walls.
    - For every green pixel, paint an adjacent background cell orange when that background cell also touches a structural wall or lies on the grid edge.
    - Erase the grey glyph to black.

    Key insight:
    - The glyph shape does not map to a color; it implicitly specifies the geometric reflection, and that reflected placement determines where orange border cells can appear relative to the surviving structural walls.
    """
    grid = np.asarray(input_grid, dtype=int)
    assert grid.ndim == 2
    h, w = grid.shape

    mode, (gr, gc) = _find_glyph(grid)
    structural = _structural_wall_mask(grid)

    out = np.where(structural, WALL, BG).astype(int)

    green_coords = np.argwhere(grid == OBJ)
    assert len(green_coords) > 0, "expected at least one green object"
    out[green_coords[:, 0], green_coords[:, 1]] = OBJ

    mirrored = _mirror_coords(green_coords, grid.shape, mode)
    out[mirrored[:, 0], mirrored[:, 1]] = OBJ

    out[gr : gr + 3, gc : gc + 3][grid[gr : gr + 3, gc : gc + 3] == GLYPH] = BG

    object_mask = out == OBJ
    border_cells: set[tuple[int, int]] = set()
    for r, c in np.argwhere(object_mask):
        rr = int(r)
        cc = int(c)
        for nr, nc in neighbors4(rr, cc):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if out[nr, nc] != BG:
                continue
            touches_edge = nr == 0 or nc == 0 or nr == h - 1 or nc == w - 1
            touches_wall = any(
                0 <= wr < h and 0 <= wc < w and structural[wr, wc]
                for wr, wc in neighbors4(nr, nc)
            )
            if touches_edge or touches_wall:
                border_cells.add((nr, nc))

    for r, c in border_cells:
        if out[r, c] == BG:
            out[r, c] = BORDER

    return out


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _random_connected_mask(min_cells: int, max_cells: int, max_dim: int = 4) -> np.ndarray:
    for _ in range(200):
        h = random.randint(2, max_dim)
        w = random.randint(2, max_dim)
        limit = min(max_cells, h * w)
        if limit < min_cells:
            continue
        area = random.randint(min_cells, limit)
        mask = np.zeros((h, w), dtype=bool)
        seed = (random.randrange(h), random.randrange(w))
        mask[seed] = True
        frontier = set()
        for nr, nc in neighbors4(*seed):
            if 0 <= nr < h and 0 <= nc < w:
                frontier.add((nr, nc))
        while int(mask.sum()) < area and frontier:
            nr, nc = random.choice(tuple(frontier))
            frontier.remove((nr, nc))
            mask[nr, nc] = True
            for rr, cc in neighbors4(nr, nc):
                if 0 <= rr < h and 0 <= cc < w and not mask[rr, cc]:
                    frontier.add((rr, cc))
        if int(mask.sum()) != area:
            continue
        cropped = _crop_mask(mask)
        if cropped.shape[0] == 1 or cropped.shape[1] == 1:
            continue
        if np.all(cropped):
            continue
        return cropped
    raise RuntimeError("failed to sample connected mask")


def _can_place(mask: np.ndarray, occupied: np.ndarray, r0: int, c0: int, pad: int = 1) -> bool:
    h, w = occupied.shape
    mh, mw = mask.shape
    if r0 < 0 or c0 < 0 or r0 + mh > h or c0 + mw > w:
        return False
    rs0 = max(0, r0 - pad)
    cs0 = max(0, c0 - pad)
    rs1 = min(h, r0 + mh + pad)
    cs1 = min(w, c0 + mw + pad)
    if np.any(occupied[rs0:rs1, cs0:cs1]):
        return False
    return True


def _paint_mask(grid: np.ndarray, occupied: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    mh, mw = mask.shape
    for r in range(mh):
        for c in range(mw):
            if mask[r, c]:
                grid[r0 + r, c0 + c] = color
                occupied[r0 + r, c0 + c] = True


def _source_limits(shape: tuple[int, int], mode: str) -> tuple[int, int]:
    h, w = shape
    top_max = h // 2 - 2
    left_max = w // 2 - 2
    if mode == "h":
        return top_max, w - 2
    if mode == "v":
        return h - 2, left_max
    return top_max, left_max


def _extra_object_position(
    mask: np.ndarray,
    occupied: np.ndarray,
    mode: str,
) -> tuple[int, int] | None:
    h, w = occupied.shape
    mh, mw = mask.shape
    max_r, max_c = _source_limits((h, w), mode)
    min_r = 3
    min_c = 3
    if max_r - mh + 1 < min_r or max_c - mw + 1 < min_c:
        return None
    positions = [
        (r0, c0)
        for r0 in range(min_r, max_r - mh + 2)
        for c0 in range(min_c, max_c - mw + 2)
    ]
    random.shuffle(positions)
    for r0, c0 in positions:
        if _can_place(mask, occupied, r0, c0, pad=1):
            return r0, c0
    return None


def _add_decorations(grid: np.ndarray, occupied: np.ndarray, count: int) -> None:
    h, w = grid.shape
    for _ in range(count):
        placed = False
        for _ in range(80):
            mask = _random_connected_mask(1, 3, max_dim=3)
            mh, mw = mask.shape
            if h - mh - 2 <= 1 or w - mw - 2 <= 1:
                continue
            r0 = random.randint(1, h - mh - 2)
            c0 = random.randint(1, w - mw - 2)
            if not _can_place(mask, occupied, r0, c0, pad=1):
                continue
            _paint_mask(grid, occupied, mask, r0, c0, WALL)
            placed = True
            break
        if not placed:
            return


def _build_input(mode: str) -> np.ndarray:
    h = random.randint(16, 20)
    w = random.randint(16, 20)
    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    if mode == "h":
        wall_r = random.choice([h - 2, h - 3])
        grid[wall_r, :] = WALL
        occupied[wall_r, :] = True
    elif mode == "v":
        wall_c = random.choice([w - 2, w - 3])
        grid[:, wall_c] = WALL
        occupied[:, wall_c] = True
    else:
        wall_r = random.choice([h - 2, h - 3])
        wall_c = random.choice([w - 2, w - 3])
        grid[wall_r, :] = WALL
        grid[:, wall_c] = WALL
        occupied[wall_r, :] = True
        occupied[:, wall_c] = True

    n_objects = random.randint(1, 3)
    anchor_mask = _random_connected_mask(4, 7, max_dim=4)
    ah, aw = anchor_mask.shape

    if mode == "h":
        target_r = wall_r - 1 - ah
        target_c = random.randint(4, w - aw - 2)
        source_r = h - target_r - ah
        source_c = target_c
    elif mode == "v":
        target_r = random.randint(4, h - ah - 2)
        target_c = wall_c - 1 - aw
        source_r = target_r
        source_c = w - target_c - aw
    else:
        target_r = wall_r - 1 - ah
        target_c = wall_c - 1 - aw
        source_r = h - target_r - ah
        source_c = w - target_c - aw

    assert source_r >= 3 and source_c >= 3
    if not _can_place(anchor_mask, occupied, source_r, source_c, pad=1):
        raise AssertionError("anchor placement unexpectedly blocked")
    _paint_mask(grid, occupied, anchor_mask, source_r, source_c, OBJ)

    for _ in range(n_objects - 1):
        placed = False
        for _ in range(120):
            mask = _random_connected_mask(4, 10, max_dim=4)
            pos = _extra_object_position(mask, occupied, mode)
            if pos is None:
                continue
            _paint_mask(grid, occupied, mask, pos[0], pos[1], OBJ)
            placed = True
            break
        if not placed:
            break

    _add_decorations(grid, occupied, random.randint(0, 2))

    glyph = GLYPH_PATTERNS[mode]
    for r in range(3):
        for c in range(3):
            if glyph[r, c]:
                grid[r, c] = GLYPH

    return grid


def _is_valid_generated(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if grid.shape[0] < 8 or grid.shape[1] < 8:
        return False
    try:
        out = main(grid)
    except Exception:
        return False
    if np.any(out == GLYPH):
        return False
    if np.count_nonzero(out == OBJ) != 2 * np.count_nonzero(grid == OBJ):
        return False
    if np.count_nonzero(out == BORDER) == 0:
        return False
    green_components = components_mask(grid == OBJ, connectivity=4)
    if not (1 <= len(green_components) <= 3):
        return False
    blue_components = components_mask(out == WALL, connectivity=4)
    h, w = out.shape
    for comp in blue_components:
        if not any(
            int(r) == 0 or int(c) == 0 or int(r) == h - 1 or int(c) == w - 1 for r, c in comp
        ):
            return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(400):
        mode = random.choice(["h", "v", "hv"])
        grid = _build_input(mode)
        if _is_valid_generated(grid):
            return grid
    raise RuntimeError("failed to generate a valid input")
