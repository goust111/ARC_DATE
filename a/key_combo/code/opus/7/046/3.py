from __future__ import annotations

from collections import deque
import random

import numpy as np


BLUE = 1
RED = 2
GREY = 5
BACKGROUND = 0

DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIRS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

BASE_H = 9
BASE_W = 12

BASE_SHAPES_THREE = (
    ((0, 3, 4), (1, 2, 4), (2, 2, 4), (3, 1, 4), (4, 0, 5)),
    ((5, 0, 3), (6, 0, 3), (7, 0, 4), (8, 1, 4)),
    ((3, 8, 11), (4, 7, 11), (5, 6, 11), (6, 6, 10), (7, 5, 9)),
)

BASE_SHAPES_TWO = (
    ((0, 3, 4), (1, 2, 4), (2, 2, 4), (3, 1, 4), (4, 0, 5), (5, 0, 3), (6, 0, 3), (7, 0, 4), (8, 1, 4)),
    BASE_SHAPES_THREE[2],
)

MAX_GENERATION_ATTEMPTS = 400


def _components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    h, w = mask.shape
    dirs = DIRS4 if connectivity == 4 else DIRS8
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in dirs:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _edge_reachable(passable: np.ndarray) -> np.ndarray:
    h, w = passable.shape
    seen = np.zeros_like(passable, dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if passable[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if passable[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in DIRS4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return seen


def _fill_outlines(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for r in range(out.shape[0]):
        cols = np.flatnonzero(out[r] == BLUE)
        if cols.size == 0:
            continue
        assert cols.size % 2 == 0, "each occupied row must contain paired blue boundary pixels"
        for i in range(0, cols.size, 2):
            c0 = int(cols[i])
            c1 = int(cols[i + 1])
            out[r, c0 : c1 + 1] = BLUE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each blue outline row-span, then recolor markers trapped by the new solid barriers.

    Input structure
    - The grid has a black background, 2-3 blue hollow outlines, and several red single-pixel markers.
    - Every outline occupies consecutive rows and has exactly two blue pixels on each occupied row.
    - The red markers sit on background cells rather than on the blue outlines.

    Rule
    - For each occupied row, pair the blue boundary pixels from left to right and fill every full horizontal span between each pair.
    - After filling, treat black and red cells as passable space, find which passable cells still reach the grid edge by 4-connectivity, and recolor every non-edge-reachable red marker to grey.
    - Leave the blue pixels and all black background cells unchanged.

    Key insight
    - The sparse blue outlines do not fully block the background until their row spans are filled, and that fill step creates new solid barriers whose topology determines which markers become enclosed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    filled = _fill_outlines(g)
    passable = (filled == BACKGROUND) | (filled == RED)
    reachable = _edge_reachable(passable)

    out = filled.copy()
    trapped_red = (filled == RED) & ~reachable
    out[trapped_red] = GREY
    return out


def _transform_shape(
    shape: tuple[tuple[int, int, int], ...],
    *,
    flip_h: bool,
    flip_v: bool,
) -> list[tuple[int, int, int]]:
    rows: list[tuple[int, int, int]] = []
    for r, c0, c1 in shape:
        rr = BASE_H - 1 - r if flip_v else r
        if flip_h:
            nc0 = BASE_W - 1 - c1
            nc1 = BASE_W - 1 - c0
        else:
            nc0, nc1 = c0, c1
        rows.append((rr, nc0, nc1))
    rows.sort()
    return rows


def _place_family(
    base_shapes: tuple[tuple[tuple[int, int, int], ...], ...],
    h: int,
    w: int,
) -> list[list[tuple[int, int, int]]]:
    flip_h = random.choice((False, True))
    flip_v = random.choice((False, True))
    top = random.randint(1, h - BASE_H - 1)
    left = random.randint(1, w - BASE_W - 1)

    placed: list[list[tuple[int, int, int]]] = []
    for shape in base_shapes:
        transformed = _transform_shape(shape, flip_h=flip_h, flip_v=flip_v)
        placed.append([(r + top, c0 + left, c1 + left) for r, c0, c1 in transformed])
    return placed


def _render_outline_input(h: int, w: int, shapes: list[list[tuple[int, int, int]]]) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for shape in shapes:
        for r, c0, c1 in shape:
            grid[r, c0] = BLUE
            grid[r, c1] = BLUE
    return grid


def _render_filled_blue(h: int, w: int, shapes: list[list[tuple[int, int, int]]]) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for shape in shapes:
        for r, c0, c1 in shape:
            grid[r, c0 : c1 + 1] = BLUE
    return grid


def _split_zero_regions(filled_blue: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    open_regions: list[np.ndarray] = []
    enclosed_regions: list[np.ndarray] = []
    h, w = filled_blue.shape
    for coords in _components(filled_blue == BACKGROUND, connectivity=4):
        touches_edge = bool(
            np.any(
                (coords[:, 0] == 0)
                | (coords[:, 0] == h - 1)
                | (coords[:, 1] == 0)
                | (coords[:, 1] == w - 1)
            )
        )
        if touches_edge:
            open_regions.append(coords)
        else:
            enclosed_regions.append(coords)
    return open_regions, enclosed_regions


def _safe_marker_cells(region: np.ndarray, filled_blue: np.ndarray) -> list[tuple[int, int]]:
    h, w = filled_blue.shape
    region_mask = np.zeros_like(filled_blue, dtype=bool)
    region_mask[region[:, 0], region[:, 1]] = True
    candidates: list[tuple[int, int]] = []
    for r, c in map(tuple, region):
        if r in (0, h - 1) or c in (0, w - 1):
            continue
        degree = 0
        for dr, dc in DIRS4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and region_mask[nr, nc]:
                degree += 1
        if degree >= 2:
            candidates.append((int(r), int(c)))
    if candidates:
        return candidates
    return [tuple(map(int, rc)) for rc in region]


def _pick_markers(filled_blue: np.ndarray) -> list[tuple[int, int]]:
    open_regions, enclosed_regions = _split_zero_regions(filled_blue)
    assert open_regions and enclosed_regions

    enclosed_regions.sort(key=len, reverse=True)
    open_regions.sort(key=len, reverse=True)

    enclosed_candidates = _safe_marker_cells(enclosed_regions[0], filled_blue)
    open_candidates = _safe_marker_cells(open_regions[0], filled_blue)
    assert enclosed_candidates and open_candidates

    total_markers = random.randint(4, 6)
    enclosed_count = min(len(enclosed_candidates), random.randint(1, min(2, total_markers - 1)))
    open_count = total_markers - enclosed_count
    if open_count > len(open_candidates):
        open_count = len(open_candidates)
        enclosed_count = min(enclosed_count, total_markers - open_count)
    assert enclosed_count >= 1 and open_count >= 1

    markers = random.sample(enclosed_candidates, enclosed_count)
    remaining_open = [cell for cell in open_candidates if cell not in markers]
    markers.extend(random.sample(remaining_open, open_count))
    return markers
def _gen_attempt() -> np.ndarray | None:
    h = random.randint(14, 18)
    w = random.randint(14, 18)
    want_two = random.random() < 0.45
    base_shapes = BASE_SHAPES_TWO if want_two else BASE_SHAPES_THREE
    shapes = _place_family(base_shapes, h, w)

    filled_blue = _render_filled_blue(h, w, shapes)
    open_regions, enclosed_regions = _split_zero_regions(filled_blue)
    if not open_regions or not enclosed_regions:
        return None

    inp = _render_outline_input(h, w, shapes)
    for r in range(h):
        count = int(np.count_nonzero(inp[r] == BLUE))
        if count % 2 != 0:
            return None

    markers = _pick_markers(filled_blue)
    for r, c in markers:
        inp[r, c] = RED

    passable_before = (inp == BACKGROUND) | (inp == RED)
    if not np.array_equal(_edge_reachable(passable_before), passable_before):
        return None

    out = main(inp)
    if np.count_nonzero(out == GREY) < 1:
        return None
    if np.count_nonzero(out == RED) < 1:
        return None
    if out.shape != inp.shape or max(out.shape) > 30:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
