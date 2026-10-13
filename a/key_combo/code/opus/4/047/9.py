from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import components_mask, get_bg


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4

HEIGHT_RANGE = (11, 16)
WIDTH_RANGE = (13, 18)
GEN_ATTEMPTS = 600


def _component_order(coords: np.ndarray, direction: int) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    if direction == -1:
        return (int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max()))
    return (-int(cols.max()), int(rows.min()), -int(cols.min()), int(rows.max()))


def _slide_side(grid: np.ndarray, color: int, side_col: int, direction: int) -> list[np.ndarray]:
    movable = (grid == color)
    movable[:, side_col] = False
    comps = components_mask(movable, connectivity=4)
    comps.sort(key=lambda coords: _component_order(coords, direction))

    h, w = grid.shape
    settled = np.zeros((h, w), dtype=bool)
    settled[:, side_col] = True
    placed_components: list[np.ndarray] = []

    for coords in comps:
        shift = 0
        while True:
            candidate = coords + np.array([0, shift + direction], dtype=int)
            cols = candidate[:, 1]
            if int(cols.min()) < 0 or int(cols.max()) >= w:
                break
            if np.any(settled[candidate[:, 0], candidate[:, 1]]):
                break
            shift += direction

        placed = coords + np.array([0, shift], dtype=int)
        assert not np.any(settled[placed[:, 0], placed[:, 1]])
        settled[placed[:, 0], placed[:, 1]] = True
        placed_components.append(placed)

    return placed_components


def _closed_background_regions(grid: np.ndarray, bg: int) -> list[np.ndarray]:
    regions = components_mask(grid == bg, connectivity=4)
    h = grid.shape[0]
    closed = []
    for coords in regions:
        rows = coords[:, 0]
        if not np.any((rows == 0) | (rows == h - 1)):
            closed.append(coords)
    return closed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide both wall-colored object groups sideways, then flood-fill the enclosed background pockets that only appear after compression.

    Input structure:
    - The background is black, with a full-height blue wall on the left edge and a full-height red wall on the right edge.
    - Away from the walls, blue and red 4-connected objects of the matching wall colors are scattered in the interior.
    - The movable objects do not start on the wall columns; their row positions encode the barriers that will exist after sliding.

    Rule:
    - Split the non-wall blue pixels into 4-connected objects and slide them left, processing nearer objects first so each one stops at the wall or at already-settled blue pixels.
    - Split the non-wall red pixels into 4-connected objects and slide them right, again processing nearer objects first against the red wall or already-settled red pixels.
    - Erase the original movable objects, redraw them only at their settled positions, and keep both wall columns unchanged.
    - In the slid grid, find every 4-connected black region; if a region touches neither the top row nor the bottom row, fill that entire region yellow.

    Key insight:
    - The target yellow areas are not present in the input at all: they are the black pockets that become topologically enclosed only after the wall-colored objects compress against their matching walls.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = int(get_bg(g))
    h, w = g.shape
    left_color = int(g[0, 0])
    right_color = int(g[0, w - 1])

    out = np.full_like(g, bg)
    out[:, 0] = left_color
    out[:, w - 1] = right_color

    placements: list[tuple[int, np.ndarray]] = []
    placements.extend((left_color, coords) for coords in _slide_side(g, left_color, 0, -1))
    placements.extend((right_color, coords) for coords in _slide_side(g, right_color, w - 1, 1))

    occupied = out != bg
    for color, coords in placements:
        assert not np.any(occupied[coords[:, 0], coords[:, 1]])
        out[coords[:, 0], coords[:, 1]] = int(color)
        occupied[coords[:, 0], coords[:, 1]] = True

    for region in _closed_background_regions(out, bg):
        out[region[:, 0], region[:, 1]] = YELLOW

    return out


def _coords_from_mask(mask: np.ndarray, r0: int, c0: int) -> np.ndarray:
    return np.argwhere(mask) + np.array([int(r0), int(c0)], dtype=int)


def _paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    coords = _coords_from_mask(mask, r0, c0)
    grid[coords[:, 0], coords[:, 1]] = int(color)


def _touches_existing(coords: np.ndarray, existing: list[np.ndarray], shape: tuple[int, int]) -> bool:
    h, w = shape
    if np.any(coords[:, 0] <= 0) or np.any(coords[:, 0] >= h - 1):
        return True
    if np.any(coords[:, 1] <= 1) or np.any(coords[:, 1] >= w - 2):
        return True
    for other in existing:
        dr = np.abs(coords[:, None, 0] - other[None, :, 0])
        dc = np.abs(coords[:, None, 1] - other[None, :, 1])
        if bool(np.any((dr <= 1) & (dc <= 1))):
            return True
    return False


def _random_connected_mask(size: int, max_h: int = 3, max_w: int = 4) -> np.ndarray:
    dims = [
        (h, w)
        for h in range(1, max_h + 1)
        for w in range(1, max_w + 1)
        if h * w >= size and max(h, w) <= size
    ]
    h, w = random.choice(dims)
    coords = {(random.randrange(h), random.randrange(w))}

    while len(coords) < size:
        frontier = []
        for r, c in coords:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in coords:
                    frontier.append((nr, nc))
        coords.add(random.choice(frontier))

    mask = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        mask[r, c] = True

    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _hook_mask(side: str, width: int) -> np.ndarray:
    mask = np.zeros((3, width), dtype=bool)
    mask[0, :] = True
    mask[2, :] = True
    if side == "left":
        mask[1, width - 1] = True
    else:
        mask[1, 0] = True
    return mask


def _place_patch(
    grid: np.ndarray,
    existing: list[np.ndarray],
    color: int,
    mask: np.ndarray,
    r0: int,
    c0: int,
) -> tuple[int, int, np.ndarray] | None:
    coords = _coords_from_mask(mask, r0, c0)
    if _touches_existing(coords, existing, grid.shape):
        return None
    _paint_mask(grid, mask, r0, c0, color)
    existing.append(coords)
    return r0, c0, coords


def _row_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(a[0], b[0]) <= min(a[1], b[1])


def _place_extra(
    grid: np.ndarray,
    existing: list[np.ndarray],
    color: int,
    target_rows: tuple[int, int],
    wall_near_col: int,
) -> bool:
    h, w = grid.shape
    size = random.randint(3, 5)
    mask = _random_connected_mask(size)
    ph, pw = mask.shape

    row_candidates = list(range(1, h - ph))
    random.shuffle(row_candidates)

    if color == BLUE:
        col_lo = wall_near_col + 3
        col_hi = w - pw - 2
    else:
        col_lo = 2
        col_hi = wall_near_col - pw - 3
    if col_lo > col_hi:
        return False

    col_candidates = list(range(col_lo, col_hi + 1))
    random.shuffle(col_candidates)

    for r0 in row_candidates:
        rows = (r0, r0 + ph - 1)
        if not _row_overlap(rows, target_rows):
            continue
        for c0 in col_candidates:
            coords = _coords_from_mask(mask, r0, c0)
            if _touches_existing(coords, existing, grid.shape):
                continue
            _paint_mask(grid, mask, r0, c0, color)
            existing.append(coords)
            return True
    return False


def _hook_start_rows(height: int, count: int) -> list[int] | None:
    starts = list(range(1, height - 3))
    random.shuffle(starts)
    chosen: list[int] = []
    for r0 in starts:
        if all(abs(r0 - prev) >= 4 for prev in chosen):
            chosen.append(r0)
            if len(chosen) == count:
                chosen.sort()
                return chosen
    return None


def _movable_components(grid: np.ndarray, color: int, side_col: int) -> list[np.ndarray]:
    mask = (grid == color)
    mask[:, side_col] = False
    return components_mask(mask, connectivity=4)


def _validate_input(grid: np.ndarray) -> bool:
    h, w = grid.shape
    if not np.all(grid[:, 0] == BLUE) or not np.all(grid[:, w - 1] == RED):
        return False

    all_objects = _movable_components(grid, BLUE, 0) + _movable_components(grid, RED, w - 1)
    if len(_movable_components(grid, BLUE, 0)) < 2 or len(_movable_components(grid, RED, w - 1)) < 2:
        return False
    for coords in all_objects:
        if not (3 <= int(coords.shape[0]) <= 7):
            return False
        if np.any(coords[:, 0] <= 0) or np.any(coords[:, 0] >= h - 1):
            return False
        if np.any(coords[:, 1] <= 1) or np.any(coords[:, 1] >= w - 2):
            return False

    for i, a in enumerate(all_objects):
        for b in all_objects[i + 1 :]:
            dr = np.abs(a[:, None, 0] - b[None, :, 0])
            dc = np.abs(a[:, None, 1] - b[None, :, 1])
            if bool(np.any((dr <= 1) & (dc <= 1))):
                return False

    if _closed_background_regions(grid, BLACK):
        return False
    return True


def _validate_output(inp: np.ndarray, out: np.ndarray) -> bool:
    if not np.any(out == YELLOW):
        return False
    if not np.any(out == BLACK):
        return False
    if np.array_equal(inp, out):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(*HEIGHT_RANGE)
    w = random.randint(*WIDTH_RANGE)

    grid = np.zeros((h, w), dtype=int)
    grid[:, 0] = BLUE
    grid[:, w - 1] = RED

    existing: list[np.ndarray] = []

    blue_count = random.choice([2, 3])
    red_count = random.choice([2, 3])
    blue_rows = _hook_start_rows(h, blue_count)
    red_rows = _hook_start_rows(h, red_count)
    if blue_rows is None or red_rows is None:
        return None

    blue_specs = [(r0, _hook_mask("left", random.choice([2, 3]))) for r0 in blue_rows]
    red_specs = [(r0, _hook_mask("right", random.choice([2, 3]))) for r0 in red_rows]

    blue_hooks: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    red_hooks: list[tuple[int, int, np.ndarray, np.ndarray]] = []

    for r0, mask in blue_specs:
        pw = mask.shape[1]
        candidates = list(range(2, min(7, w - pw - 3) + 1))
        random.shuffle(candidates)
        placed = None
        for c0 in candidates:
            placed = _place_patch(grid, existing, BLUE, mask, r0, c0)
            if placed is not None:
                blue_hooks.append((r0, c0, mask, placed[2]))
                break
        if placed is None:
            return None

    for r0, mask in red_specs:
        pw = mask.shape[1]
        gap_choices = list(range(2, min(7, w - pw - 3) + 1))
        random.shuffle(gap_choices)
        placed = None
        for gap in gap_choices:
            c0 = w - pw - gap
            placed = _place_patch(grid, existing, RED, mask, r0, c0)
            if placed is not None:
                red_hooks.append((r0, c0, mask, placed[2]))
                break
        if placed is None:
            return None

    if random.random() < 0.45 and blue_hooks:
        target_r0, target_c0, target_mask, _coords = random.choice(blue_hooks)
        target_rows = (target_r0, target_r0 + target_mask.shape[0] - 1)
        if not _place_extra(grid, existing, BLUE, target_rows, target_c0 + target_mask.shape[1] - 1):
            return None

    if random.random() < 0.45 and red_hooks:
        target_r0, target_c0, target_mask, _coords = random.choice(red_hooks)
        target_rows = (target_r0, target_r0 + target_mask.shape[0] - 1)
        if not _place_extra(grid, existing, RED, target_rows, target_c0):
            return None

    if not _validate_input(grid):
        return None

    out = main(grid)
    if not _validate_output(grid, out):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gravity-compression enclosure input")
