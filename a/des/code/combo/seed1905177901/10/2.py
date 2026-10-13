from __future__ import annotations

import random
from collections import deque

import numpy as np


AZURE = 8
GREY = 5
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _find_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue

            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []

            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))

            coords = np.array(cells, dtype=int)
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                }
            )

    components.sort(key=lambda comp: tuple(comp["bbox"]))  # deterministic order
    return components


def _trend(coords: np.ndarray) -> int:
    cols = coords[:, 1]
    min_c = int(cols.min())
    max_c = int(cols.max())
    assert min_c < max_c, "stair object must span multiple columns"

    left_rows = coords[cols == min_c, 0]
    right_rows = coords[cols == max_c, 0]
    left_row = float(left_rows.mean())
    right_row = float(right_rows.mean())
    assert left_row != right_row, "stair direction must be unambiguous"
    return -1 if right_row < left_row else 1


def _normalize(coords: list[tuple[int, int]]) -> np.ndarray:
    arr = np.array(coords, dtype=int)
    arr[:, 0] -= int(arr[:, 0].min())
    arr[:, 1] -= int(arr[:, 1].min())
    return arr


def _make_stair(widths: list[int], trend: int) -> np.ndarray:
    cells: list[tuple[int, int]] = []
    col = 0
    for i, width in enumerate(widths):
        row = trend * i
        for dc in range(width):
            cells.append((row, col + dc))
        col += width - 1
    return _normalize(cells)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each stair-shaped object vertically by its global trend and leave a grey trail.

    Rising stairs (their right side sits above their left side) move up by 2.
    Descending stairs move down by 2. The original and intermediate positions
    are recolored grey, and the final translated object keeps its original color.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = AZURE if np.any(g == AZURE) else int(np.bincount(g.ravel()).argmax())
    objects = _find_components(g, bg=bg)
    if not objects:
        return g.copy()

    out = np.full_like(g, bg)
    for obj in objects:
        color = int(obj["color"])
        coords = np.asarray(obj["coords"], dtype=int)
        trend = _trend(coords)

        shifted_rows = coords[:, 0] + 2 * trend
        assert shifted_rows.min() >= 0 and shifted_rows.max() < h, "moved object must stay in bounds"
        assert coords[:, 1].min() >= 0 and coords[:, 1].max() < w

        for step in range(2):
            dr = trend * step
            for r, c in coords:
                out[int(r + dr), int(c)] = GREY

        for r, c in coords:
            out[int(r + 2 * trend), int(c)] = color

    return out


# --------------- generator ---------------

PALETTE = [0, 1, 2, 3, 4, 6, 7, 9]
GRID_H_RANGE = (10, 17)
GRID_W_RANGE = (12, 21)
OBJECT_COUNT_RANGE = (2, 5)
STEP_COUNT_RANGE = (3, 6)
STEP_WIDTH_RANGE = (2, 5)
PAD = 1
MAX_RETRIES = 400


def _reserved_cells(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    reserved: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    reserved.add((rr, cc))
    return reserved


def _footprint(coords: np.ndarray, top: int, left: int, trend: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for step in range(3):
        dr = trend * step
        for r, c in coords:
            cells.add((int(top + r + dr), int(left + c)))
    return cells


def _sample_shape(rng: random.Random, trend: int) -> np.ndarray:
    step_count = rng.randint(*STEP_COUNT_RANGE)
    widths = [rng.randint(*STEP_WIDTH_RANGE) for _ in range(step_count)]
    return _make_stair(widths, trend)


def _place_object(
    grid: np.ndarray,
    reserved: np.ndarray,
    coords: np.ndarray,
    trend: int,
    color: int,
    rng: random.Random,
) -> bool:
    h, w = grid.shape
    shape_h = int(coords[:, 0].max()) + 1
    shape_w = int(coords[:, 1].max()) + 1
    min_shift = min(0, trend, 2 * trend)
    max_shift = max(0, trend, 2 * trend)

    row_lo = PAD - min_shift
    row_hi = h - PAD - shape_h - max_shift
    col_lo = PAD
    col_hi = w - PAD - shape_w
    if row_lo > row_hi or col_lo > col_hi:
        return False

    candidates = [(r, c) for r in range(row_lo, row_hi + 1) for c in range(col_lo, col_hi + 1)]
    rng.shuffle(candidates)

    for top, left in candidates:
        footprint = _footprint(coords, top, left, trend)
        reserved_cells = _reserved_cells(footprint, h, w)
        if any(reserved[r, c] for r, c in reserved_cells):
            continue

        for r, c in coords:
            grid[int(top + r), int(left + c)] = color
        for r, c in reserved_cells:
            reserved[r, c] = True
        return True

    return False


def _balanced_trends(count: int, rng: random.Random) -> list[int]:
    trends = [-1, 1] * (count // 2)
    if count % 2 == 1:
        trends.append(rng.choice([-1, 1]))
    rng.shuffle(trends)
    return trends


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    grid = np.full((h, w), AZURE, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    max_objects = min(OBJECT_COUNT_RANGE[1], len(PALETTE), max(OBJECT_COUNT_RANGE[0], (h * w) // 28))
    count = rng.randint(OBJECT_COUNT_RANGE[0], max_objects)
    trends = _balanced_trends(count, rng)
    colors = rng.sample(PALETTE, count)

    specs: list[tuple[int, int, int, np.ndarray]] = []
    for trend, color in zip(trends, colors):
        coords = _sample_shape(rng, trend)
        specs.append((coords.shape[0], trend, color, coords))
    specs.sort(key=lambda item: -item[0])

    for _area, trend, color, coords in specs:
        if not _place_object(grid, reserved, coords, trend, color, rng):
            return None

    components = _find_components(grid, bg=AZURE)
    if len(components) != count:
        return None

    seen_trends = {_trend(np.asarray(comp["coords"], dtype=int)) for comp in components}
    if seen_trends != {-1, 1} and count >= 2:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stair-trail input")
