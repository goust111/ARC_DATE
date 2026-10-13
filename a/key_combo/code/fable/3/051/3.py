from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _mask_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _straight_segment_info(coords: np.ndarray) -> tuple[str, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    if np.all(rows == rows[0]):
        c0 = int(cols.min())
        c1 = int(cols.max())
        length = c1 - c0 + 1
        assert len(coords) == length, "horizontal segment must be contiguous"
        return "h", length
    if np.all(cols == cols[0]):
        r0 = int(rows.min())
        r1 = int(rows.max())
        length = r1 - r0 + 1
        assert len(coords) == length, "vertical segment must be contiguous"
        return "v", length
    raise AssertionError("component must be a straight segment")


def _find_key(grid: np.ndarray) -> tuple[set[tuple[int, int]], int]:
    key_mask = grid == GREY
    key_components = _mask_components(key_mask)
    assert len(key_components) == 1, "expected exactly one grey key segment"
    key_coords = key_components[0]
    _orientation, key_len = _straight_segment_info(key_coords)

    key_cells = {tuple(map(int, coord)) for coord in key_coords}
    for r, c in key_cells:
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(grid.shape, nr, nc):
                assert grid[nr, nc] != RED, "key must be isolated from the red fences"
    return key_cells, key_len


def _extract_wall_segments(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    red = grid == RED
    h, w = grid.shape
    segments: list[list[tuple[int, int]]] = []

    for r in range(h):
        c = 0
        while c < w:
            if not red[r, c]:
                c += 1
                continue
            c1 = c
            while c1 + 1 < w and red[r, c1 + 1]:
                c1 += 1
            run_len = c1 - c + 1
            if run_len > 1:
                segments.append([(r, cc) for cc in range(c, c1 + 1)])
            elif (r == 0 or not red[r - 1, c]) and (r == h - 1 or not red[r + 1, c]):
                segments.append([(r, c)])
            c = c1 + 1

    for c in range(w):
        r = 0
        while r < h:
            if not red[r, c]:
                r += 1
                continue
            r1 = r
            while r1 + 1 < h and red[r1 + 1, c]:
                r1 += 1
            run_len = r1 - r + 1
            if run_len > 1:
                segments.append([(rr, c) for rr in range(r, r1 + 1)])
            r = r1 + 1

    covered = {cell for segment in segments for cell in segment}
    red_cells = {tuple(map(int, coord)) for coord in np.argwhere(red)}
    assert covered == red_cells, "every red cell must belong to a wall segment"
    return segments


def _exterior_black_mask(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    black = grid == BLACK
    exterior = np.zeros_like(black, dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if black[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if black[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(grid.shape, nr, nc) and black[nr, nc] and not exterior[nr, nc]:
                exterior[nr, nc] = True
                q.append((nr, nc))

    return exterior


def main(grid: np.ndarray) -> np.ndarray:
    """
    Measure the isolated grey key segment, erase every equally long red wall segment, then fill the still-enclosed black regions blue.

    Input structure:
    - The grid has a black background with one or more one-cell-thick red fence shapes.
    - A single isolated grey straight segment sits in a corner and does not touch any red fence.
    - Each fence can be decomposed into maximal horizontal or vertical red wall segments joined at corners.

    Rule:
    - Read the grey key length k.
    - Remove every maximal red wall segment whose length is exactly k, turning those cells black while leaving the other red fence cells in place.
    - In the modified grid, recolor every black cell that cannot reach the outer boundary by 4-neighbor steps to blue; keep the grey key unchanged.

    Key insight:
    - Enclosure is judged only after the matching wall segments are erased, so the key first changes the fence topology and only then determines which black regions are true holes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    assert bg == BLACK, "expected black to be the background"
    colors = set(map(int, np.unique(g)))
    assert colors <= {BLACK, RED, GREY}, "input should only use black, red, and grey"
    assert RED in colors and GREY in colors, "expected both red fences and one grey key"

    _key_cells, key_len = _find_key(g)
    wall_segments = _extract_wall_segments(g)

    removed = g.copy()
    to_remove = {
        cell
        for segment in wall_segments
        if len(segment) == key_len
        for cell in segment
    }
    for r, c in to_remove:
        removed[r, c] = BLACK

    exterior = _exterior_black_mask(removed)
    enclosed = (removed == BLACK) & ~exterior

    out = removed.copy()
    out[enclosed] = BLUE
    return out


MIN_GRID = 18
MAX_GRID = 30
MIN_KEY = 3
MAX_KEY = 8
RECT_MIN = 4
RECT_MAX = 10
PLACEMENT_RETRIES = 300
GENERATION_RETRIES = 300


def _rect_outline_cells(rect: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def _draw_rect_outline(grid: np.ndarray, rect: tuple[int, int, int, int], color: int) -> None:
    for r, c in _rect_outline_cells(rect):
        grid[r, c] = color


def _boxes_too_close(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    gap: int,
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0)


def _key_cells_for_corner(
    h: int,
    w: int,
    key_len: int,
    corner: str,
    orientation: str,
) -> list[tuple[int, int]]:
    if corner == "tl":
        return [(0, c) for c in range(key_len)] if orientation == "h" else [(r, 0) for r in range(key_len)]
    if corner == "tr":
        return (
            [(0, c) for c in range(w - key_len, w)]
            if orientation == "h"
            else [(r, w - 1) for r in range(key_len)]
        )
    if corner == "bl":
        return (
            [(h - 1, c) for c in range(key_len)]
            if orientation == "h"
            else [(r, 0) for r in range(h - key_len, h)]
        )
    if corner == "br":
        return (
            [(h - 1, c) for c in range(w - key_len, w)]
            if orientation == "h"
            else [(r, w - 1) for r in range(h - key_len, h)]
        )
    raise ValueError(f"unknown corner: {corner}")


def _cells_bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _sample_open_dims(key_len: int, side_max: int) -> tuple[int, int]:
    if random.random() < 0.5:
        width = key_len
        height_choices = [v for v in range(RECT_MIN, side_max + 1) if v != key_len]
        height = random.choice(height_choices)
        return height, width
    height = key_len
    width_choices = [v for v in range(RECT_MIN, side_max + 1) if v != key_len]
    width = random.choice(width_choices)
    return height, width


def _sample_closed_dims(key_len: int, side_max: int) -> tuple[int, int]:
    choices = [
        (h, w)
        for h in range(RECT_MIN, side_max + 1)
        for w in range(RECT_MIN, side_max + 1)
        if h != key_len and w != key_len
    ]
    return random.choice(choices)


def _place_rectangles(
    h: int,
    w: int,
    specs: list[tuple[str, int, int]],
    forbidden: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]] | None:
    placed: list[tuple[int, int, int, int]] = []
    for _kind, rect_h, rect_w in sorted(specs, key=lambda x: x[1] * x[2], reverse=True):
        candidates = [
            (r0, c0)
            for r0 in range(1, h - rect_h)
            for c0 in range(1, w - rect_w)
        ]
        random.shuffle(candidates)
        rect = None
        for r0, c0 in candidates:
            candidate = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
            if any(_boxes_too_close(candidate, other, gap=1) for other in forbidden + placed):
                continue
            rect = candidate
            break
        if rect is None:
            return None
        placed.append(rect)
    return placed


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(MIN_GRID, MAX_GRID)
    w = random.randint(MIN_GRID, MAX_GRID)
    key_len = random.randint(MIN_KEY, min(MAX_KEY, h - 10, w - 10))
    side_max = min(RECT_MAX, h - 2, w - 2)
    if side_max < RECT_MIN or side_max == key_len:
        return None

    corner = random.choice(["tl", "tr", "bl", "br"])
    orientation = random.choice(["h", "v"])
    key_cells = _key_cells_for_corner(h, w, key_len, corner, orientation)
    key_box = _cells_bbox(key_cells)

    n_total = random.randint(2, 4)
    specs: list[tuple[str, int, int]] = [("open", *_sample_open_dims(key_len, side_max)), ("closed", *_sample_closed_dims(key_len, side_max))]
    while len(specs) < n_total:
        kind = random.choice(["open", "closed"])
        dims = _sample_open_dims(key_len, side_max) if kind == "open" else _sample_closed_dims(key_len, side_max)
        specs.append((kind, *dims))

    rects = _place_rectangles(h, w, specs, [key_box])
    if rects is None:
        return None

    grid = np.full((h, w), BLACK, dtype=int)
    for rect in rects:
        _draw_rect_outline(grid, rect, RED)
    for r, c in key_cells:
        grid[r, c] = GREY

    out = main(grid)
    if not np.any((grid == RED) & (out == BLACK)):
        return None
    if not np.any(out == BLUE):
        return None
    if not np.any(out == RED):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid key-wall ARC input")
