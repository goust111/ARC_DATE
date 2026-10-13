import numpy as np


BG = 0
FRAME = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRECTIONS = ("left", "right", "up", "down")

GRID_MIN = 14
GRID_MAX = 30
SEED_COUNT_RANGE = (1, 5)
FRAME_SIZES = (7, 9, 11)
GENERATION_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 200
SEED_SAMPLING_ATTEMPTS = 120


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _border_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def _rectangular_frame_bbox(comp: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    rows = [r for r, _c in comp]
    cols = [c for _r, c in comp]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None
    expected = set(_border_cells(r0, c0, r1, c1))
    if set(comp) != expected:
        return None
    return r0, c0, r1, c1


def _direction_from_centroid(
    rows: np.ndarray,
    cols: np.ndarray,
    center_r: float,
    center_c: float,
) -> str:
    dr = float(rows.mean()) - float(center_r)
    dc = float(cols.mean()) - float(center_c)
    if abs(dc) > abs(dr):
        return "left" if dc < 0 else "right"
    if abs(dr) > abs(dc):
        return "up" if dr < 0 else "down"
    if abs(dc) > 0:
        return "left" if dc < 0 else "right"
    return "up" if dr <= 0 else "down"


def _spike_cells(bbox: tuple[int, int, int, int], direction: str, length: int) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    mid_r = (r0 + r1) // 2
    mid_c = (c0 + c1) // 2
    if direction == "left":
        return [(mid_r, c0 - step) for step in range(1, length + 1)]
    if direction == "right":
        return [(mid_r, c1 + step) for step in range(1, length + 1)]
    if direction == "up":
        return [(r0 - step, mid_c) for step in range(1, length + 1)]
    return [(r1 + step, mid_c) for step in range(1, length + 1)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow an outward spike from each grey rectangular frame using the enclosed seed count and centroid bias.

    Input structure:
    - The background is black.
    - One or more one-pixel-thick grey rectangular frames enclose separate regions.
    - Each frame contains 1 to 5 interior seed pixels of a single non-grey color.
    - Different frames use different seed colors, and the grey borders stay unchanged.

    Rule:
    - For each grey rectangular frame, count its interior seed pixels to get N.
    - Compare the seed centroid with the frame center to find the dominant offset
      direction: left, right, up, or down.
    - From the midpoint of the matching frame side, extend a same-color spike of
      length N outward beyond the grey border.
    - Keep the original frame pixels and seed pixels unchanged.

    Key insight:
    - The same seed set encodes two independent attributes: cardinality gives the
      spike length, while its spatial bias relative to the frame center gives the
      side from which the spike emerges.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    h, w = g.shape

    for comp in _components4(g == FRAME):
        bbox = _rectangular_frame_bbox(comp)
        if bbox is None:
            continue

        r0, c0, r1, c1 = bbox
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        seed_mask = (interior != BG) & (interior != FRAME)
        if not np.any(seed_mask):
            continue

        seed_colors = np.unique(interior[seed_mask])
        if seed_colors.size != 1:
            continue

        seed_rows, seed_cols = np.argwhere(seed_mask).T
        seed_rows = seed_rows + r0 + 1
        seed_cols = seed_cols + c0 + 1
        direction = _direction_from_centroid(seed_rows, seed_cols, (r0 + r1) / 2.0, (c0 + c1) / 2.0)
        length = int(seed_mask.sum())
        color = int(seed_colors[0])

        for rr, cc in _spike_cells(bbox, direction, length):
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = color

    return out


def _max_region_count(height: int, width: int) -> int:
    area = height * width
    if area >= 500:
        return 3
    if area >= 260:
        return 2
    return 1


def _semantic_cells(bbox: tuple[int, int, int, int], direction: str, length: int) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
    cells.update(_spike_cells(bbox, direction, length))
    return cells


def _reserved_cells(cells: set[tuple[int, int]], height: int, width: int) -> set[tuple[int, int]]:
    reserved: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width:
                    reserved.add((nr, nc))
    return reserved


def _ranked_interior_cells(inner_h: int, inner_w: int, direction: str) -> list[tuple[int, int]]:
    center_r = (inner_h - 1) / 2.0
    center_c = (inner_w - 1) / 2.0
    cells = [(r, c) for r in range(inner_h) for c in range(inner_w)]
    if direction == "left":
        key = lambda rc: (rc[1], abs(rc[0] - center_r), rc[0])
    elif direction == "right":
        key = lambda rc: (-rc[1], abs(rc[0] - center_r), rc[0])
    elif direction == "up":
        key = lambda rc: (rc[0], abs(rc[1] - center_c), rc[1])
    else:
        key = lambda rc: (-rc[0], abs(rc[1] - center_c), rc[1])
    return sorted(cells, key=key)


def _sample_seed_offsets(
    rng: np.random.Generator,
    outer_h: int,
    outer_w: int,
    count: int,
    direction: str,
) -> list[tuple[int, int]]:
    inner_h = outer_h - 2
    inner_w = outer_w - 2
    ranked = _ranked_interior_cells(inner_h, inner_w, direction)
    center_r = (inner_h - 1) / 2.0
    center_c = (inner_w - 1) / 2.0
    pool_size = min(len(ranked), max(count + 4, count * 2))
    pool = ranked[:pool_size]

    for _ in range(SEED_SAMPLING_ATTEMPTS):
        picks = rng.choice(len(pool), size=count, replace=False)
        offsets = [pool[int(i)] for i in picks]
        rows = np.array([r for r, _c in offsets], dtype=float)
        cols = np.array([c for _r, c in offsets], dtype=float)
        if _direction_from_centroid(rows, cols, center_r, center_c) == direction:
            return offsets

    fallback = ranked[:count]
    rows = np.array([r for r, _c in fallback], dtype=float)
    cols = np.array([c for _r, c in fallback], dtype=float)
    assert _direction_from_centroid(rows, cols, center_r, center_c) == direction
    return fallback


def _paint_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    width = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    region_count = int(rng.integers(1, _max_region_count(height, width) + 1))
    colors = [int(c) for c in rng.choice(np.array(SEED_COLORS), size=region_count, replace=False)]

    size_choices = FRAME_SIZES if region_count == 1 else FRAME_SIZES[:2]
    inp = np.full((height, width), BG, dtype=int)
    out = np.full((height, width), BG, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    for color in colors:
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            outer_h = int(rng.choice(size_choices))
            outer_w = int(rng.choice(size_choices))
            direction = DIRECTIONS[int(rng.integers(0, len(DIRECTIONS)))]
            count = int(rng.integers(SEED_COUNT_RANGE[0], SEED_COUNT_RANGE[1] + 1))

            if outer_h > height - 2 or outer_w > width - 2:
                continue

            r0 = int(rng.integers(1, height - outer_h))
            c0 = int(rng.integers(1, width - outer_w))
            bbox = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
            semantic = _semantic_cells(bbox, direction, count)

            if any(r <= 0 or r >= height - 1 or c <= 0 or c >= width - 1 for r, c in semantic):
                continue

            reserved = _reserved_cells(semantic, height, width)
            if any(occupied[r, c] for r, c in reserved):
                continue

            offsets = _sample_seed_offsets(rng, outer_h, outer_w, count, direction)

            _paint_frame(inp, bbox, FRAME)
            _paint_frame(out, bbox, FRAME)
            for sr, sc in offsets:
                rr = r0 + 1 + sr
                cc = c0 + 1 + sc
                inp[rr, cc] = color
                out[rr, cc] = color
            for rr, cc in _spike_cells(bbox, direction, count):
                out[rr, cc] = color

            for r, c in reserved:
                occupied[r, c] = True
            placed = True
            break

        if not placed:
            return None

    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
