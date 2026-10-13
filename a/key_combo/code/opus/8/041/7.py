from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
GREEN = 3
YELLOW = 4

_DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIR8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _tile_four(grid: np.ndarray) -> np.ndarray:
    top = np.concatenate((grid, np.fliplr(grid)), axis=1)
    bottom = np.concatenate((np.flipud(grid), np.rot90(grid, 2)), axis=1)
    return np.concatenate((top, bottom), axis=0)


def _exterior_background(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    exterior = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    def add(r: int, c: int) -> None:
        if grid[r, c] == BLACK and not exterior[r, c]:
            exterior[r, c] = True
            q.append((r, c))

    for r in range(h):
        add(r, 0)
        add(r, w - 1)
    for c in range(w):
        add(0, c)
        add(h - 1, c)

    while q:
        r, c = q.popleft()
        for dr, dc in _DIR4:
            nr, nc = r + dr, c + dc
            if _in_bounds(grid.shape, nr, nc) and grid[nr, nc] == BLACK and not exterior[nr, nc]:
                exterior[nr, nc] = True
                q.append((nr, nc))

    return exterior


def _outline_mask(grid: np.ndarray, exterior: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    outline = np.zeros((h, w), dtype=bool)
    for r, c in np.argwhere(exterior):
        r = int(r)
        c = int(c)
        for dr, dc in _DIR8:
            nr, nc = r + dr, c + dc
            if _in_bounds(grid.shape, nr, nc) and grid[nr, nc] == BLUE:
                outline[r, c] = True
                break
    return outline


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        q: deque[tuple[int, int]] = deque([(r, c)])
        seen[r, c] = True
        comp: list[tuple[int, int]] = []
        while q:
            cr, cc = q.popleft()
            comp.append((cr, cc))
            for dr, dc in _DIR4:
                nr, nc = cr + dr, cc + dc
                if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _analyze(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    exterior = _exterior_background(grid)
    holes = (grid == BLACK) & ~exterior
    outline = _outline_mask(grid, exterior)
    return exterior, holes, outline


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror the input into four reflected quadrants, then mark exterior outlines and newly enclosed holes.

    Input structure:
    - The grid is a small black background with blue foreground pixels forming one or more sparse wall-like shapes.
    - Some blue shapes create pockets that are open to the source tile's right edge and/or bottom edge.
    - Optional interior blue rings may already contain closed holes before tiling.

    Rule:
    - First reflect-tile the whole input into a 2x2 arrangement:
      upper-left = original, upper-right = left-right mirror,
      lower-left = up-down mirror, lower-right = 180-degree rotation.
    - On that tiled grid, flood-fill black background from the outer border using 4-connectivity.
    - Recolor every unreachable black region yellow and every border-reachable black cell that 8-neighbors blue foreground green.
    - Keep every blue foreground cell unchanged and leave all other background black.

    Key insight:
    - The open pockets must be analyzed only after reflection tiling, because the seam between mirrored copies can close them and create holes that do not exist in the original tile.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 15, "the reflected output must stay within 30x30"
    assert np.any(g == BLUE), "expected blue foreground"
    assert set(map(int, np.unique(g))).issubset({BLACK, BLUE}), "expected only black and blue in the input"

    tiled = _tile_four(g)
    _exterior, holes, outline = _analyze(tiled)

    out = tiled.copy()
    out[outline] = GREEN
    out[holes] = YELLOW
    return out


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _boxes_separated(box: tuple[int, int, int, int], boxes: list[tuple[int, int, int, int]], pad: int = 1) -> bool:
    r0, c0, r1, c1 = box
    for or0, oc0, or1, oc1 in boxes:
        if not (r1 + pad < or0 or or1 + pad < r0 or c1 + pad < oc0 or oc1 + pad < c0):
            return False
    return True


def _paint_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    grid[r0 : r0 + h, c0 : c0 + w][patch] = BLUE


def _horizontal_pocket(height: int, width: int) -> np.ndarray:
    patch = np.zeros((height, width), dtype=bool)
    patch[0, :] = True
    patch[-1, :] = True
    patch[:, 0] = True
    return patch


def _vertical_pocket(height: int, width: int) -> np.ndarray:
    patch = np.zeros((height, width), dtype=bool)
    patch[:, 0] = True
    patch[:, -1] = True
    patch[0, :] = True
    return patch


def _ring(height: int, width: int) -> np.ndarray:
    patch = np.ones((height, width), dtype=bool)
    patch[1:-1, 1:-1] = False
    return patch


def _place_horizontal_pocket(grid: np.ndarray, boxes: list[tuple[int, int, int, int]], keep_top_clear: bool) -> bool:
    h, w = grid.shape
    candidates: list[tuple[np.ndarray, int, int, tuple[int, int, int, int]]] = []
    max_height = min(6, h)
    max_width = min(6, w)
    for ph in range(3, max_height + 1):
        for pw in range(3, max_width + 1):
            c0 = w - pw
            max_r0 = h - ph
            if keep_top_clear:
                max_r0 = min(max_r0, h - ph - 2)
            for r0 in range(max(0, 0), max_r0 + 1):
                box = (r0, c0, r0 + ph - 1, c0 + pw - 1)
                if _boxes_separated(box, boxes):
                    candidates.append((_horizontal_pocket(ph, pw), r0, c0, box))
    random.shuffle(candidates)
    if not candidates:
        return False
    patch, r0, c0, box = candidates[0]
    _paint_patch(grid, patch, r0, c0)
    boxes.append(box)
    return True


def _place_vertical_pocket(grid: np.ndarray, boxes: list[tuple[int, int, int, int]], keep_left_clear: bool) -> bool:
    h, w = grid.shape
    candidates: list[tuple[np.ndarray, int, int, tuple[int, int, int, int]]] = []
    max_height = min(6, h)
    max_width = min(6, w)
    for ph in range(3, max_height + 1):
        for pw in range(3, max_width + 1):
            r0 = h - ph
            max_c0 = w - pw
            if keep_left_clear:
                max_c0 = min(max_c0, w - pw - 2)
            for c0 in range(0, max_c0 + 1):
                box = (r0, c0, r0 + ph - 1, c0 + pw - 1)
                if _boxes_separated(box, boxes):
                    candidates.append((_vertical_pocket(ph, pw), r0, c0, box))
    random.shuffle(candidates)
    if not candidates:
        return False
    patch, r0, c0, box = candidates[0]
    _paint_patch(grid, patch, r0, c0)
    boxes.append(box)
    return True


def _place_ring(grid: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> bool:
    h, w = grid.shape
    candidates: list[tuple[np.ndarray, int, int, tuple[int, int, int, int]]] = []
    max_height = min(5, h - 2)
    max_width = min(5, w - 2)
    for ph in range(3, max_height + 1):
        for pw in range(3, max_width + 1):
            for r0 in range(1, h - ph):
                for c0 in range(1, w - pw):
                    box = (r0, c0, r0 + ph - 1, c0 + pw - 1)
                    if _boxes_separated(box, boxes):
                        candidates.append((_ring(ph, pw), r0, c0, box))
    random.shuffle(candidates)
    if not candidates:
        return False
    patch, r0, c0, box = candidates[0]
    _paint_patch(grid, patch, r0, c0)
    boxes.append(box)
    return True


def _hole_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    _exterior, holes, _outline = _analyze(grid)
    return _components(holes)


def _crosses_vertical_seam(comp: list[tuple[int, int]], seam_col: int) -> bool:
    cols = {c for _, c in comp}
    return seam_col - 1 in cols and seam_col in cols


def _crosses_horizontal_seam(comp: list[tuple[int, int]], seam_row: int) -> bool:
    rows = {r for r, _ in comp}
    return seam_row - 1 in rows and seam_row in rows


def _semantic_checks(grid: np.ndarray, mode: tuple[str, ...], with_ring: bool) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 15:
        return False
    if not np.any(grid == BLUE):
        return False
    if np.sum(grid == BLUE) >= (grid.size // 2):
        return False

    original_holes = _hole_components(grid)
    if with_ring != (len(original_holes) > 0):
        return False

    tiled = _tile_four(grid)
    exterior, holes, outline = _analyze(tiled)
    expected = tiled.copy()
    expected[outline] = GREEN
    expected[holes] = YELLOW
    if not np.array_equal(main(grid), expected):
        return False

    if not np.any(expected == GREEN) or not np.any(expected == YELLOW):
        return False
    if np.any(outline & holes):
        return False
    if not np.array_equal(expected[tiled == BLUE], np.full(np.sum(tiled == BLUE), BLUE, dtype=int)):
        return False
    if not np.all(exterior | holes | (tiled == BLUE)):
        return False

    tiled_holes = _components(holes)
    if len(tiled_holes) <= 4 * len(original_holes):
        return False

    h, w = grid.shape
    if "h" in mode and not any(_crosses_vertical_seam(comp, w) for comp in tiled_holes):
        return False
    if "v" in mode and not any(_crosses_horizontal_seam(comp, h) for comp in tiled_holes):
        return False
    return True


def _build_candidate() -> np.ndarray | None:
    mode = random.choices(
        [("h",), ("v",), ("h", "v")],
        weights=[3, 3, 4],
        k=1,
    )[0]
    with_ring = random.random() < 0.55

    min_side = 6
    if len(mode) == 2 or with_ring:
        min_side = 7
    max_side = 10
    h = random.randint(min_side, max_side)
    w = random.randint(min_side, max_side)

    grid = np.zeros((h, w), dtype=int)
    boxes: list[tuple[int, int, int, int]] = []

    if "h" in mode and not _place_horizontal_pocket(grid, boxes, keep_top_clear=("v" in mode)):
        return None
    if "v" in mode and not _place_vertical_pocket(grid, boxes, keep_left_clear=("h" in mode)):
        return None
    if with_ring and not _place_ring(grid, boxes):
        return None

    if not _semantic_checks(grid, mode, with_ring):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _build_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to construct a valid mirror-closure example")
