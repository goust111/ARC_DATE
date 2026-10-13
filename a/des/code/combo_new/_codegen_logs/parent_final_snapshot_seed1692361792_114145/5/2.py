import random

import numpy as np


BG = 0
ARC_MAX = 30
PALETTE = tuple(range(1, 10))
GEN_RETRIES = 400
RECT_COUNT_RANGE = (2, 5)
SIDE_RANGE = (1, 7)
MAX_FRAGMENT_PATCHES = 3
MAX_GROW_STEPS = 10
PLACEMENT_RETRIES = 250


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _color_rectangles(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    rectangles: list[tuple[int, int, int, int]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != BG):
        coords = np.argwhere(grid == color)
        assert coords.size > 0, f"color {color} must appear"
        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        rectangles.append((h * w, h, w, color))
    return rectangles


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each color's bounding box and repack the resulting solid rectangles along the diagonal.

    Input structure:
    - The background is black.
    - Several non-black colors appear as sparse fragments, and one color may have multiple disconnected pieces.
    - For each color, all of its pixels fit inside one axis-aligned bounding box, and these color boxes do not overlap each other.

    Rule:
    - For every non-black color, compute the minimal axis-aligned bounding box covering all pixels of that color.
    - Fill that entire box with the color, turning each fragment collection into one solid rectangle.
    - Sort the rectangles by area from smallest to largest.
    - On a new black grid, place the sorted rectangles so the first starts at the top-left corner and each later rectangle starts one row below and one column to the right of the previous rectangle's bottom-right corner.

    Key insight:
    - The sparse pixels are only hints for each color's full bounding rectangle; the output depends on those recovered rectangles and their area order, not on the original fragment layout.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    rectangles = _color_rectangles(g)
    assert rectangles, "expected at least one foreground color"

    areas = [area for area, _h, _w, _color in rectangles]
    assert len(set(areas)) == len(areas), "rectangle areas must be unique"

    rectangles.sort(key=lambda item: item[0])
    out_h = sum(h for _area, h, _w, _color in rectangles)
    out_w = sum(w for _area, _h, w, _color in rectangles)
    assert out_h <= ARC_MAX and out_w <= ARC_MAX, "output exceeds ARC bounds"

    out = np.full((out_h, out_w), BG, dtype=int)
    row = 0
    col = 0
    for _area, h, w, color in rectangles:
        out[row : row + h, col : col + w] = color
        row += h
        col += w
    return out


def _sample_rectangles() -> list[tuple[int, int, int]] | None:
    count = random.randint(*RECT_COUNT_RANGE)
    for _ in range(200):
        rectangles: list[tuple[int, int, int]] = []
        used_areas: set[int] = set()
        sum_h = 0
        sum_w = 0
        for _ in range(count):
            for _ in range(200):
                h = random.randint(*SIDE_RANGE)
                w = random.randint(*SIDE_RANGE)
                area = h * w
                if area in used_areas:
                    continue
                if sum_h + h > ARC_MAX or sum_w + w > ARC_MAX:
                    continue
                rectangles.append((h, w, area))
                used_areas.add(area)
                sum_h += h
                sum_w += w
                break
            else:
                rectangles = []
                break
        if not rectangles:
            continue
        if max(area for _h, _w, area in rectangles) < 4:
            continue
        return rectangles
    return None


def _fragment_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    anchors = {
        (0, random.randrange(w)),
        (h - 1, random.randrange(w)),
        (random.randrange(h), 0),
        (random.randrange(h), w - 1),
    }
    for r, c in anchors:
        mask[r, c] = True

    for _ in range(random.randint(0, MAX_FRAGMENT_PATCHES)):
        ph = random.randint(1, max(1, min(h, 3)))
        pw = random.randint(1, max(1, min(w, 3)))
        r0 = random.randint(0, h - ph)
        c0 = random.randint(0, w - pw)
        mask[r0 : r0 + ph, c0 : c0 + pw] = True

    steps = random.randint(0, min(MAX_GROW_STEPS, max(0, h * w - int(mask.sum()))))
    for _ in range(steps):
        cells = np.argwhere(mask)
        if cells.size == 0:
            break
        r, c = map(int, cells[random.randrange(len(cells))])
        neighbors = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w and not mask[rr, cc]:
                neighbors.append((rr, cc))
        if neighbors:
            rr, cc = random.choice(neighbors)
            mask[rr, cc] = True

    if h * w >= 4 and mask.all():
        removable = [(r, c) for r in range(h) for c in range(w) if (r, c) not in anchors]
        if removable:
            r, c = random.choice(removable)
            mask[r, c] = False

    return mask


def _place_rectangles(rectangles: list[tuple[int, int, int]], height: int, width: int) -> list[tuple[int, int]] | None:
    occupied = np.zeros((height, width), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(rectangles)

    order = sorted(range(len(rectangles)), key=lambda idx: rectangles[idx][0] * rectangles[idx][1], reverse=True)
    for idx in order:
        h, w, _area = rectangles[idx]
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            r0 = random.randint(0, height - h)
            c0 = random.randint(0, width - w)
            r1 = r0 + h
            c1 = c0 + w
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(height, r1 + 1)
            cc1 = min(width, c1 + 1)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            occupied[r0:r1, c0:c1] = True
            placements[idx] = (r0, c0)
            placed = True
            break
        if not placed:
            return None

    return [(int(r), int(c)) for r, c in placements if r is not None and c is not None]


def _build_input(rectangles: list[tuple[int, int, int]]) -> np.ndarray | None:
    total_area = sum(area for _h, _w, area in rectangles)
    max_h = max(h for h, _w, _area in rectangles)
    max_w = max(w for _h, w, _area in rectangles)
    base_span = max(int(np.ceil(np.sqrt(total_area * 3))), max_h + 3, max_w + 3, 10)
    min_h = min(ARC_MAX, base_span)
    min_w = min(ARC_MAX, base_span)
    max_h_grid = min(ARC_MAX, base_span + 8)
    max_w_grid = min(ARC_MAX, base_span + 8)
    if min_h > max_h_grid or min_w > max_w_grid:
        return None

    grid_h = random.randint(min_h, max_h_grid)
    grid_w = random.randint(min_w, max_w_grid)
    placements = _place_rectangles(rectangles, grid_h, grid_w)
    if placements is None:
        return None

    colors = random.sample(PALETTE, len(rectangles))
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    had_sparse_piece = False
    for (h, w, _area), (r0, c0), color in zip(rectangles, placements, colors):
        mask = _fragment_mask(h, w)
        had_sparse_piece |= int(mask.sum()) < h * w
        grid[r0 : r0 + h, c0 : c0 + w][mask] = color

    if not had_sparse_piece:
        return None

    if grid.shape[0] > ARC_MAX or grid.shape[1] > ARC_MAX:
        return None

    out = main(grid)
    if out.shape[0] > ARC_MAX or out.shape[1] > ARC_MAX:
        return None

    observed = _color_rectangles(grid)
    if sorted(area for area, _h, _w, _color in observed) != sorted(area for _h, _w, area in rectangles):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rectangles = _sample_rectangles()
        if rectangles is None:
            continue
        grid = _build_input(rectangles)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
