import random

import numpy as np


BG = 0
RED = 2
GREEN = 3
PATTERN_COLORS = (1, 4, 5, 6, 7, 8, 9)

GRID_H_RANGE = (18, 31)
GRID_W_RANGE = (18, 31)
OUTER_SIZE_MIN = 7
OUTER_SIZE_MAX_BY_COUNT = {
    1: 13,
    2: 13,
    3: 11,
    4: 9,
}
OBJECT_GAP = 1
PLACEMENT_TRIES = 200
MAX_GENERATION_ATTEMPTS = 300


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)

    return comps


def _is_pattern(value: int) -> bool:
    return value not in (BG, RED, GREEN)


def _frame_info(grid: np.ndarray, pts: list[tuple[int, int]]) -> tuple[int, int, int, int, tuple[int, int]]:
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    assert r1 - r0 >= 4 and c1 - c0 >= 4, "frame must enclose a non-trivial interior"

    border = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }
    assert set(pts) == border, "special-color component must be a complete rectangular border"

    reds = [(r, c) for r, c in border if int(grid[r, c]) == RED]
    assert len(reds) == 1, "each frame must contain exactly one red edge marker"
    marker = reds[0]
    assert marker not in {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}, "marker cannot sit on a corner"

    for r, c in border:
        assert int(grid[r, c]) in (GREEN, RED), "frame border must use only green plus one red marker"

    return r0, c0, r1, c1, marker


def _count_pattern(grid: np.ndarray, rows: list[int], cols: list[int]) -> int:
    if not rows or not cols:
        return 0
    sub = grid[np.ix_(rows, cols)]
    return int(np.count_nonzero((sub != BG) & (sub != GREEN) & (sub != RED)))


def _mirror_vertical(out: np.ndarray, grid: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    rows = list(range(r0 + 1, r1))
    cols = list(range(c0 + 1, c1))
    half = len(cols) // 2
    left_cols = cols[:half]
    right_cols = cols[-half:]
    assert left_cols and right_cols, "vertical fold needs cells on both sides of the axis"

    left_count = _count_pattern(grid, rows, left_cols)
    right_count = _count_pattern(grid, rows, right_cols)
    assert left_count or right_count, "one side of the frame must contain a pattern"

    source_cols = left_cols if left_count >= right_count else right_cols
    edge_sum = cols[0] + cols[-1]

    for r in rows:
        for c in source_cols:
            mc = edge_sum - c
            value = int(grid[r, c])
            out[r, mc] = value if _is_pattern(value) else BG


def _mirror_horizontal(out: np.ndarray, grid: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    rows = list(range(r0 + 1, r1))
    cols = list(range(c0 + 1, c1))
    half = len(rows) // 2
    top_rows = rows[:half]
    bottom_rows = rows[-half:]
    assert top_rows and bottom_rows, "horizontal fold needs cells on both sides of the axis"

    top_count = _count_pattern(grid, top_rows, cols)
    bottom_count = _count_pattern(grid, bottom_rows, cols)
    assert top_count or bottom_count, "one side of the frame must contain a pattern"

    source_rows = top_rows if top_count >= bottom_count else bottom_rows
    edge_sum = rows[0] + rows[-1]

    for r in source_rows:
        mr = edge_sum - r
        for c in cols:
            value = int(grid[r, c])
            out[mr, c] = value if _is_pattern(value) else BG


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer each frame's fold axis from its red border marker, then mirror the colored
    half-pattern across the corresponding center line to fill the blank half.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    special_mask = (g == GREEN) | (g == RED)
    for pts in _components(special_mask):
        r0, c0, r1, c1, marker = _frame_info(g, pts)
        mr, mc = marker
        if mr in (r0, r1):
            _mirror_horizontal(out, g, r0, c0, r1, c1)
        else:
            _mirror_vertical(out, g, r0, c0, r1, c1)
        out[mr, mc] = GREEN

    return out


def _choose_object_count(area: int, rng: random.Random) -> int:
    if area < 450:
        return rng.choice((1, 2, 2, 3))
    if area < 650:
        return rng.choice((1, 2, 2, 3, 3, 4))
    return rng.choice((2, 2, 3, 3, 4))


def _sample_pattern_mask(height: int, width: int, rng: random.Random) -> np.ndarray:
    area = height * width
    mask = np.zeros((height, width), dtype=bool)
    mode = rng.randrange(4)

    if mode == 0:
        target = rng.randint(1, max(1, min(area - 1, (3 * area) // 5))) if area > 1 else 1
        start = (rng.randrange(height), rng.randrange(width))
        mask[start] = True
        frontier = [start]
        while int(mask.sum()) < target:
            br, bc = frontier[rng.randrange(len(frontier))]
            neighbors = []
            for nr, nc in ((br - 1, bc), (br + 1, bc), (br, bc - 1), (br, bc + 1)):
                if 0 <= nr < height and 0 <= nc < width:
                    neighbors.append((nr, nc))
            nr, nc = neighbors[rng.randrange(len(neighbors))]
            mask[nr, nc] = True
            frontier.append((nr, nc))
    elif mode == 1:
        rect_count = rng.randint(1, 3)
        for _ in range(rect_count):
            rh = rng.randint(1, height)
            rw = rng.randint(1, width)
            rr = rng.randint(0, height - rh)
            cc = rng.randint(0, width - rw)
            mask[rr : rr + rh, cc : cc + rw] = True
    elif mode == 2:
        if width >= height:
            row = rng.randrange(height)
            step = rng.choice((-1, 0, 1))
            for c in range(width):
                row = min(height - 1, max(0, row + (step if c and rng.random() < 0.7 else 0)))
                mask[row, c] = True
                if height > 1 and rng.random() < 0.35:
                    mask[min(height - 1, row + 1), c] = True
        else:
            col = rng.randrange(width)
            step = rng.choice((-1, 0, 1))
            for r in range(height):
                col = min(width - 1, max(0, col + (step if r and rng.random() < 0.7 else 0)))
                mask[r, col] = True
                if width > 1 and rng.random() < 0.35:
                    mask[r, min(width - 1, col + 1)] = True
    else:
        points = rng.randint(1, max(1, min(area, height + width)))
        for _ in range(points):
            rr = rng.randrange(height)
            cc = rng.randrange(width)
            mask[rr, cc] = True
            if rng.random() < 0.5 and height > 1:
                mask[max(0, rr - 1), cc] = True
            if rng.random() < 0.5 and width > 1:
                mask[rr, min(width - 1, cc + 1)] = True

    if not np.any(mask):
        mask[rng.randrange(height), rng.randrange(width)] = True
    if area > 1 and np.all(mask):
        rr = rng.randrange(height)
        cc = rng.randrange(width)
        mask[rr, cc] = False
    if area > 2 and int(mask.sum()) == 1 and rng.random() < 0.6:
        rr, cc = map(int, np.argwhere(mask)[0])
        if cc + 1 < width:
            mask[rr, cc + 1] = True
        elif rr + 1 < height:
            mask[rr + 1, cc] = True

    return mask


def _can_place(occupied: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> bool:
    rr0 = max(0, r0 - OBJECT_GAP)
    cc0 = max(0, c0 - OBJECT_GAP)
    rr1 = min(occupied.shape[0] - 1, r1 + OBJECT_GAP)
    cc1 = min(occupied.shape[1] - 1, c1 + OBJECT_GAP)
    return not np.any(occupied[rr0 : rr1 + 1, cc0 : cc1 + 1])


def _reserve(occupied: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    rr0 = max(0, r0 - OBJECT_GAP)
    cc0 = max(0, c0 - OBJECT_GAP)
    rr1 = min(occupied.shape[0] - 1, r1 + OBJECT_GAP)
    cc1 = min(occupied.shape[1] - 1, c1 + OBJECT_GAP)
    occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _paint_frame(
    grid: np.ndarray,
    r0: int,
    c0: int,
    outer_h: int,
    outer_w: int,
    axis: str,
    marker_side: str,
    source_side: str,
    color: int,
    rng: random.Random,
) -> None:
    r1 = r0 + outer_h - 1
    c1 = c0 + outer_w - 1

    grid[r0, c0 : c1 + 1] = GREEN
    grid[r1, c0 : c1 + 1] = GREEN
    grid[r0 : r1 + 1, c0] = GREEN
    grid[r0 : r1 + 1, c1] = GREEN

    inner_h = outer_h - 2
    inner_w = outer_w - 2

    if axis == "horizontal":
        half_h = inner_h // 2
        source_r = 0 if source_side == "top" else inner_h - half_h
        source_c = 0
        pattern_h = half_h
        pattern_w = inner_w
    else:
        half_w = inner_w // 2
        source_r = 0
        source_c = 0 if source_side == "left" else inner_w - half_w
        pattern_h = inner_h
        pattern_w = half_w

    mask = _sample_pattern_mask(pattern_h, pattern_w, rng)
    patch = grid[r0 + 1 + source_r : r0 + 1 + source_r + pattern_h, c0 + 1 + source_c : c0 + 1 + source_c + pattern_w]
    patch[mask] = color

    if marker_side == "top":
        grid[r0, c0 + outer_w // 2] = RED
    elif marker_side == "bottom":
        grid[r1, c0 + outer_w // 2] = RED
    elif marker_side == "left":
        grid[r0 + outer_h // 2, c0] = RED
    else:
        grid[r0 + outer_h // 2, c1] = RED


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    area = h * w
    object_count = _choose_object_count(area, rng)
    max_size = OUTER_SIZE_MAX_BY_COUNT[object_count]
    grid = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    colors = list(PATTERN_COLORS)
    rng.shuffle(colors)

    for idx in range(object_count):
        outer_h = rng.randrange(OUTER_SIZE_MIN, max_size + 1, 2)
        outer_w = rng.randrange(OUTER_SIZE_MIN, max_size + 1, 2)
        if outer_h + 2 > h or outer_w + 2 > w:
            return None

        placed = False
        for _ in range(PLACEMENT_TRIES):
            r0 = rng.randint(1, h - outer_h - 1)
            c0 = rng.randint(1, w - outer_w - 1)
            r1 = r0 + outer_h - 1
            c1 = c0 + outer_w - 1
            if not _can_place(occupied, r0, c0, r1, c1):
                continue

            axis = rng.choice(("horizontal", "vertical"))
            if axis == "horizontal":
                marker_side = rng.choice(("top", "bottom"))
                source_side = rng.choice(("top", "bottom"))
            else:
                marker_side = rng.choice(("left", "right"))
                source_side = rng.choice(("left", "right"))

            _paint_frame(grid, r0, c0, outer_h, outer_w, axis, marker_side, source_side, colors[idx], rng)
            _reserve(occupied, r0, c0, r1, c1)
            placed = True
            break

        if not placed:
            return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if np.any(out == RED):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        result = _gen_attempt(rng)
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid input")
