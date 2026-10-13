from __future__ import annotations

import random
from collections import deque

import numpy as np


YELLOW = 4
GREY = 5
DIR_BY_CORNER = {
    "TL": (1, 1),
    "TR": (1, -1),
    "BL": (-1, 1),
    "BR": (-1, -1),
}
_CORNER_BAG: list[str] = []
_BLOCK_BAG: list[bool] = []

GEN_ATTEMPTS = 400
PLACE_ATTEMPTS = 160
GRID_H_RANGE = (10, 18)
GRID_W_RANGE = (10, 18)
OBJECT_COUNT_RANGE = (2, 5)


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _corners(shape: tuple[int, int]) -> dict[str, tuple[int, int]]:
    h, w = shape
    return {
        "TL": (0, 0),
        "TR": (0, w - 1),
        "BL": (h - 1, 0),
        "BR": (h - 1, w - 1),
    }


def _find_light_corner(grid: np.ndarray) -> tuple[str, tuple[int, int]]:
    marked = [(name, rc) for name, rc in _corners(grid.shape).items() if int(grid[rc]) == YELLOW]
    assert len(marked) == 1, "expected exactly one yellow corner marker"
    return marked[0]


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(grid: np.ndarray, bg: int, ignore: tuple[int, int]) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    seen[ignore] = True
    out: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == int(bg):
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(int(rr), int(cc), h, w):
                    if seen[nr, nc] or (nr, nc) == ignore:
                        continue
                    if int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            out.append(np.array(cells, dtype=int))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each foreground object by a 2-pixel diagonal grey shadow set by a yellow corner light.

    Input structure:
    - The background is the dominant color.
    - Exactly one grid corner contains a yellow singleton light marker.
    - Other foreground cells form 4-connected monochrome objects.

    Rule:
    - Remove all original object pixels.
    - For every object pixel, paint grey at one and two steps along the corner's
      inward diagonal direction.
    - Shadow pixels do not overwrite the yellow marker or any other object's
      original pixels.
    - Keep the yellow corner marker unchanged.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    corner, marker = _find_light_corner(g)
    dr, dc = DIR_BY_CORNER[corner]
    h, w = g.shape

    comps = _components(g, bg, marker)
    occupied = g != int(bg)
    occupied[marker] = False

    out = np.full_like(g, bg)
    out[marker] = YELLOW

    for comp in comps:
        own = {tuple(map(int, cell)) for cell in comp}
        for rr, cc in comp:
            r = int(rr)
            c = int(cc)
            for step in (1, 2):
                nr = r + step * dr
                nc = c + step * dc
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if (nr, nc) == marker:
                    continue
                if bool(occupied[nr, nc]) and (nr, nc) not in own:
                    continue
                out[nr, nc] = GREY

    return out


def _rect_mask(height: int, width: int) -> np.ndarray:
    return np.ones((height, width), dtype=bool)


def _diamond_mask(radius: int) -> np.ndarray:
    size = 2 * radius + 1
    rr, cc = np.indices((size, size))
    return np.abs(rr - radius) + np.abs(cc - radius) <= radius


def _random_shape(allow_singleton: bool = True) -> np.ndarray:
    kind = random.choice(["rect", "diamond"])
    if kind == "diamond":
        radius = random.choice([1, 2])
        return _diamond_mask(radius)

    while True:
        height = random.randint(1, 4)
        width = random.randint(1, 4)
        if allow_singleton or height * width > 1:
            return _rect_mask(height, width)


def _shape_cells(mask: np.ndarray, top: int, left: int) -> list[tuple[int, int]]:
    coords = np.argwhere(mask)
    return [(int(top + r), int(left + c)) for r, c in coords]


def _adjacent4(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1])) == 1


def _can_place(
    cells: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
    marker: tuple[int, int],
    shape: tuple[int, int],
    dr: int,
    dc: int,
    *,
    full_shadow: bool,
) -> bool:
    h, w = shape
    marker_neighbors = set(_neighbors4(marker[0], marker[1], h, w))
    for cell in cells:
        r, c = cell
        if not (0 <= r < h and 0 <= c < w):
            return False
        if cell == marker or cell in occupied or cell in marker_neighbors:
            return False
        if full_shadow and not (0 <= r + 2 * dr < h and 0 <= c + 2 * dc < w):
            return False
        for nr, nc in _neighbors4(r, c, h, w):
            if (nr, nc) in occupied:
                return False
    return True


def _placement_ranges(
    grid_shape: tuple[int, int],
    mask_shape: tuple[int, int],
    dr: int,
    dc: int,
    *,
    full_shadow: bool,
) -> tuple[int, int, int, int] | None:
    h, w = grid_shape
    mh, mw = mask_shape
    row_lo = 0
    row_hi = h - mh
    col_lo = 0
    col_hi = w - mw

    if full_shadow:
        if dr > 0:
            row_hi -= 2
        elif dr < 0:
            row_lo += 2
        if dc > 0:
            col_hi -= 2
        elif dc < 0:
            col_lo += 2

    if row_lo > row_hi or col_lo > col_hi:
        return None
    return int(row_lo), int(row_hi), int(col_lo), int(col_hi)


def _place_random_object(
    mask: np.ndarray,
    occupied: set[tuple[int, int]],
    marker: tuple[int, int],
    grid_shape: tuple[int, int],
    dr: int,
    dc: int,
    *,
    full_shadow: bool,
) -> list[tuple[int, int]] | None:
    bounds = _placement_ranges(grid_shape, mask.shape, dr, dc, full_shadow=full_shadow)
    if bounds is None:
        return None
    row_lo, row_hi, col_lo, col_hi = bounds
    for _ in range(PLACE_ATTEMPTS):
        top = random.randint(row_lo, row_hi)
        left = random.randint(col_lo, col_hi)
        cells = _shape_cells(mask, top, left)
        if _can_place(cells, occupied, marker, grid_shape, dr, dc, full_shadow=full_shadow):
            return cells
    return None


def _place_blocker(
    anchor: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
    marker: tuple[int, int],
    grid_shape: tuple[int, int],
    dr: int,
    dc: int,
) -> list[tuple[int, int]] | None:
    anchor_set = set(anchor)
    candidates: list[list[tuple[int, int]]] = []
    for r, c in anchor:
        for step in (1, 2):
            target = (int(r + step * dr), int(c + step * dc))
            if target in anchor_set:
                continue
            cells = [target]
            if _can_place(cells, occupied, marker, grid_shape, dr, dc, full_shadow=False):
                candidates.append(cells)
    if not candidates:
        return None
    return random.choice(candidates)


def _has_inter_object_block(objects: list[list[tuple[int, int]]], dr: int, dc: int) -> bool:
    occupied = set()
    for cells in objects:
        occupied.update(cells)
    for cells in objects:
        own = set(cells)
        for r, c in cells:
            for step in (1, 2):
                target = (int(r + step * dr), int(c + step * dc))
                if target in occupied and target not in own:
                    return True
    return False


def _build_grid(
    shape: tuple[int, int],
    bg: int,
    corner: str,
    objects: list[list[tuple[int, int]]],
    colors: list[int],
) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    marker = _corners(shape)[corner]
    grid[marker] = YELLOW
    for cells, color in zip(objects, colors):
        for r, c in cells:
            grid[r, c] = int(color)
    return grid


def _draw_corner_choice() -> str:
    global _CORNER_BAG
    if not _CORNER_BAG:
        _CORNER_BAG = list(DIR_BY_CORNER)
        random.shuffle(_CORNER_BAG)
    return _CORNER_BAG.pop()


def _draw_block_choice() -> bool:
    global _BLOCK_BAG
    if not _BLOCK_BAG:
        _BLOCK_BAG = [False, True]
        random.shuffle(_BLOCK_BAG)
    return _BLOCK_BAG.pop()


def _gen_attempt(corner: str, want_block: bool) -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    bg = random.choice([c for c in range(10) if c not in (YELLOW, GREY)])
    dr, dc = DIR_BY_CORNER[corner]
    marker = _corners((h, w))[corner]

    n_objects = random.randint(*OBJECT_COUNT_RANGE)
    palette = [c for c in range(10) if c not in (bg, YELLOW, GREY)]
    if len(palette) < n_objects:
        return None
    colors = random.sample(palette, n_objects)

    occupied: set[tuple[int, int]] = set()
    objects: list[list[tuple[int, int]]] = []

    if want_block:
        anchor = _place_random_object(
            _random_shape(allow_singleton=False),
            occupied,
            marker,
            (h, w),
            dr,
            dc,
            full_shadow=True,
        )
        if anchor is None:
            return None
        objects.append(anchor)
        occupied.update(anchor)

        blocker = _place_blocker(anchor, occupied, marker, (h, w), dr, dc)
        if blocker is None:
            return None
        objects.append(blocker)
        occupied.update(blocker)

    while len(objects) < n_objects:
        full_shadow = True
        mask = _random_shape(allow_singleton=True)
        cells = _place_random_object(mask, occupied, marker, (h, w), dr, dc, full_shadow=full_shadow)
        if cells is None:
            return None
        objects.append(cells)
        occupied.update(cells)

    has_block = _has_inter_object_block(objects, dr, dc)
    if has_block != want_block:
        return None

    grid = _build_grid((h, w), bg, corner, objects, colors)
    out = main(grid)

    if int(out[marker]) != YELLOW:
        return None
    if int(np.count_nonzero(out == GREY)) == 0:
        return None
    if not want_block and _has_inter_object_block(objects, dr, dc):
        return None
    if want_block and not _has_inter_object_block(objects, dr, dc):
        return None
    return grid


def generate_input() -> np.ndarray:
    want_block = _draw_block_choice()
    corner = _draw_corner_choice()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(corner, want_block)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
