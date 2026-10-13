import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_ATTEMPTS = 300
OBJECT_COUNT_RANGE = (3, 7)
GRID_HEIGHT_RANGE = (12, 24)
GRID_WIDTH_RANGE = (12, 24)
AREA_POOL = tuple(range(1, 12))
SHAPE_ATTEMPTS = 80
PLACEMENT_ATTEMPTS = 120


def _split_runs(indices: np.ndarray) -> list[np.ndarray]:
    if indices.size == 0:
        return []
    cuts = np.where(np.diff(indices) > 1)[0] + 1
    return [part for part in np.split(indices, cuts) if part.size]


def _extract_palette(line: np.ndarray, expected_len: int) -> list[int] | None:
    values = np.asarray(line, dtype=int)
    positions = np.flatnonzero(values != BG)
    candidates: list[tuple[int, list[int]]] = []
    for run in _split_runs(positions):
        colors = [int(values[idx]) for idx in run]
        if (
            len(colors) == expected_len
            and all(color not in (BG, GREY) for color in colors)
            and len(set(colors)) == expected_len
        ):
            candidates.append((int(run[0]), colors))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _grey_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != GREY:
                continue
            seen[r, c] = True
            queue: deque[tuple[int, int]] = deque([(r, c)])
            cells: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == GREY:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(cells)
    return comps


def _shape_key(cells: list[tuple[int, int]]) -> tuple[int, int, int]:
    top = min(r for r, _c in cells)
    left = min(c for _r, c in cells)
    return (len(cells), top, left)


def _detect_reference(grid: np.ndarray, object_count: int) -> tuple[str, list[int]]:
    top_palette = _extract_palette(grid[0, :], object_count)
    left_palette = _extract_palette(grid[:, 0], object_count)
    if top_palette is not None and left_palette is None:
        return "top", top_palette
    if left_palette is not None and top_palette is None:
        return "left", left_palette
    if top_palette is not None and left_palette is not None:
        top_start = int(np.flatnonzero(grid[0, :] != BG)[0])
        left_start = int(np.flatnonzero(grid[:, 0] != BG)[0])
        if top_start <= left_start:
            return "top", top_palette
        return "left", left_palette
    raise AssertionError("expected a reference color ribbon on the top row or left column")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor scattered grey objects by sorting them by area and reading the edge color ribbon in order.

    Input structure:
    - Background is black.
    - The top row or the left column contains one contiguous ribbon of distinct non-black colors,
      with one pixel per color and one ribbon color for each grey object.
    - Elsewhere, several 4-connected grey objects are scattered on the background, with distinct areas.

    Rule:
    - Detect all grey 4-connected objects and measure each object's area in pixels.
    - Sort the grey objects from smallest area to largest area.
    - Read the ribbon colors left-to-right if it is on the top row, or top-to-bottom if it is on the left column.
    - Recolor each sorted grey object with the correspondingly ordered ribbon color, keeping the ribbon unchanged.

    Key insight:
    - The edge ribbon is an ordered legend, and object identity comes only from relative area, not shape or position.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    comps = _grey_components(g)
    if not comps:
        return out

    _orientation, palette = _detect_reference(g, len(comps))
    ordered = sorted(comps, key=_shape_key)
    for color, cells in zip(palette, ordered):
        for r, c in cells:
            out[r, c] = int(color)
    return out


def _normalize(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _c in cells)
    min_c = min(c for _r, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _transform_shape(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    cells = list(shape)
    for _ in range(random.randrange(4)):
        cells = [(c, -r) for r, c in cells]
    if random.random() < 0.5:
        cells = [(r, -c) for r, c in cells]
    return _normalize(set(cells))


def _make_polyomino(area: int) -> tuple[tuple[int, int], ...]:
    if area == 1:
        return ((0, 0),)

    best: tuple[tuple[int, int], ...] | None = None
    for _ in range(SHAPE_ATTEMPTS):
        mode = random.choice(("compact", "mixed", "branchy"))
        cells = {(0, 0)}
        frontier = {(1, 0), (-1, 0), (0, 1), (0, -1)}
        while len(cells) < area:
            options = list(frontier)
            scored: list[tuple[float, tuple[int, int]]] = []
            for cand in options:
                adj = sum((cand[0] + dr, cand[1] + dc) in cells for dr, dc in NEIGHBORS4)
                manhattan = abs(cand[0]) + abs(cand[1])
                if mode == "compact":
                    score = 3.0 * adj - 0.7 * manhattan + random.random()
                elif mode == "branchy":
                    score = 1.5 * adj + 0.8 * manhattan + random.random()
                else:
                    score = 2.0 * adj - 0.2 * manhattan + 2.0 * random.random()
                scored.append((score, cand))
            scored.sort(key=lambda item: item[0], reverse=True)
            top_k = min(4, len(scored))
            chosen = random.choice([cand for _score, cand in scored[:top_k]])
            cells.add(chosen)
            frontier.discard(chosen)
            for dr, dc in NEIGHBORS4:
                nxt = (chosen[0] + dr, chosen[1] + dc)
                if nxt not in cells:
                    frontier.add(nxt)

        shape = _transform_shape(_normalize(cells))
        h = 1 + max(r for r, _c in shape)
        w = 1 + max(c for _r, c in shape)
        if h <= 6 and w <= 6 and (min(h, w) > 1 or area <= 4):
            return shape
        best = shape

    assert best is not None
    return best


def _mark_blocked(blocked: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for nr in range(max(0, r - 1), min(h, r + 2)):
            for nc in range(max(0, c - 1), min(w, c + 2)):
                blocked[nr, nc] = True


def _place_objects(
    grid: np.ndarray,
    shapes: list[tuple[tuple[int, int], ...]],
    orientation: str,
    start: int,
    count: int,
) -> bool:
    h, w = grid.shape
    blocked = np.zeros((h, w), dtype=bool)

    if orientation == "top":
        blocked[0, :] = True
        blocked[1, :] = True
        for idx in range(count):
            blocked[0, start + idx] = True
    else:
        blocked[:, 0] = True
        blocked[:, 1] = True
        for idx in range(count):
            blocked[start + idx, 0] = True

    for shape in sorted(shapes, key=len, reverse=True):
        sh = 1 + max(r for r, _c in shape)
        sw = 1 + max(c for _r, c in shape)
        row_min = 2 if orientation == "top" else 0
        col_min = 0 if orientation == "top" else 2
        row_max = h - sh
        col_max = w - sw
        if row_min > row_max or col_min > col_max:
            return False

        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            base_r = random.randint(row_min, row_max)
            base_c = random.randint(col_min, col_max)
            cells = [(base_r + r, base_c + c) for r, c in shape]
            if any(blocked[r, c] for r, c in cells):
                continue
            for r, c in cells:
                grid[r, c] = GREY
            _mark_blocked(blocked, cells)
            placed = True
            break
        if not placed:
            return False
    return True


def _build_input() -> np.ndarray | None:
    orientation = random.choice(("top", "left"))
    count = random.randint(*OBJECT_COUNT_RANGE)
    colors = random.sample(PALETTE, count)
    areas = random.sample(AREA_POOL, count)

    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    if orientation == "top" and w < count + 2:
        w = count + 2
    if orientation == "left" and h < count + 2:
        h = count + 2
    if h > 30 or w > 30:
        return None

    grid = np.full((h, w), BG, dtype=int)

    if orientation == "top":
        if w - count < 2:
            return None
        start = random.randint(1, w - count - 1)
        for idx, color in enumerate(colors):
            grid[0, start + idx] = int(color)
    else:
        if h - count < 2:
            return None
        start = random.randint(1, h - count - 1)
        for idx, color in enumerate(colors):
            grid[start + idx, 0] = int(color)

    shapes = [_make_polyomino(area) for area in areas]
    if not _place_objects(grid, shapes, orientation, start, count):
        return None

    comps = _grey_components(grid)
    if len(comps) != count:
        return None
    comp_areas = sorted(len(comp) for comp in comps)
    if len(set(comp_areas)) != count:
        return None
    detected_orientation, detected_colors = _detect_reference(grid, count)
    if detected_orientation != orientation or detected_colors != colors:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_input()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after repeated attempts")
