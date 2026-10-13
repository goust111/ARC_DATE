import random
from collections import deque

import numpy as np


BLACK = 0
COLOR_POOL = tuple(range(1, 10))
GRID_SIZE_RANGE = (10, 30)
OBJECT_COUNT_RANGE = (2, 5)
BBOX_HEIGHT_RANGE = (3, 8)
BBOX_WIDTH_RANGE = (3, 8)
OBJECT_DENSITY_RANGE = (0.45, 0.78)
PLACEMENT_PADDING = 1
PLACE_ATTEMPTS = 200
GENERATION_RETRIES = 500


def _spiral_coords(height: int, width: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    top, left = 0, 0
    bottom, right = height - 1, width - 1

    while top <= bottom and left <= right:
        for c in range(left, right + 1):
            coords.append((top, c))
        for r in range(top + 1, bottom + 1):
            coords.append((r, right))
        if top < bottom:
            for c in range(right - 1, left - 1, -1):
                coords.append((bottom, c))
        if left < right:
            for r in range(bottom - 1, top, -1):
                coords.append((r, left))
        top += 1
        bottom -= 1
        left += 1
        right -= 1

    return coords


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for sr in range(height):
        for sc in range(width):
            if seen[sr, sc] or not bool(mask[sr, sc]):
                continue
            seen[sr, sc] = True
            q: deque[tuple[int, int]] = deque([(sr, sc)])
            coords: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                coords.append((r, c))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width:
                        if not seen[nr, nc] and bool(mask[nr, nc]):
                            seen[nr, nc] = True
                            q.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def main(grid: np.ndarray) -> np.ndarray:
    """Repack every foreground object's colors into its bounding box by spiral order.

    Input structure:
    - The background is black.
    - Several separated 4-connected foreground objects sit on the canvas.
    - Each object may use multiple non-black colors and may leave black holes or
      gaps inside its rectangular bounding box.

    Rule:
    - For each foreground object, crop its bounding box and follow a clockwise
      spiral path from the box's top-left cell.
    - Read only the non-black cells encountered on that spiral path, preserving
      their color sequence.
    - Clear the whole bounding box to black, then write that color sequence back
      from the same top-left spiral path; any remaining cells stay black.

    Key insight:
    - The object shape is discarded while its visible color sequence is conserved:
      sparse irregular pixels are read through the spiral and compacted into the
      front of the same rectangular spiral.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"

    comps = _components(g != BLACK)
    out = g.copy()

    for coords in sorted(comps, key=lambda comp: _bbox(comp)):
        r0, c0, r1, c1 = _bbox(coords)
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        path = _spiral_coords(patch.shape[0], patch.shape[1])
        colors = [int(patch[rr, cc]) for rr, cc in path if int(patch[rr, cc]) != BLACK]

        out[r0 : r1 + 1, c0 : c1 + 1] = BLACK
        for color, (rr, cc) in zip(colors, path):
            out[r0 + rr, c0 + cc] = color

    return out


def _path_between(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = start
    er, ec = end
    cells = [(r, c)]
    while (r, c) != (er, ec):
        choices: list[tuple[int, int]] = []
        if r < er:
            choices.append((r + 1, c))
        elif r > er:
            choices.append((r - 1, c))
        if c < ec:
            choices.append((r, c + 1))
        elif c > ec:
            choices.append((r, c - 1))
        r, c = random.choice(choices)
        cells.append((r, c))
    return cells


def _make_connected_mask(height: int, width: int) -> np.ndarray | None:
    center = (random.randrange(height), random.randrange(width))
    targets = [
        (0, random.randrange(width)),
        (height - 1, random.randrange(width)),
        (random.randrange(height), 0),
        (random.randrange(height), width - 1),
    ]

    cells: set[tuple[int, int]] = set()
    for target in targets:
        cells.update(_path_between(center, target))

    total = height * width
    if len(cells) >= total:
        return None
    min_cells = len(cells)
    low = max(min_cells, int(total * OBJECT_DENSITY_RANGE[0]))
    high = min(total - 1, int(total * OBJECT_DENSITY_RANGE[1]))
    if low > high:
        return None
    target_count = random.randint(low, high)

    while len(cells) < target_count:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(frontier))

    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    if len(_components(mask)) != 1:
        return None
    if bool(np.all(mask)):
        return None
    if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
        return None
    return mask


def _paint_random_colors(mask: np.ndarray) -> np.ndarray:
    coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    palette_size = random.randint(2, min(4, len(coords), len(COLOR_POOL)))
    palette = random.sample(COLOR_POOL, palette_size)
    values = palette[:]
    values.extend(random.choice(palette) for _ in range(len(coords) - len(values)))
    random.shuffle(values)

    patch = np.full(mask.shape, BLACK, dtype=int)
    random.shuffle(coords)
    for (r, c), color in zip(coords, values):
        patch[r, c] = int(color)
    return patch


def _free_with_padding(reserved: np.ndarray, top: int, left: int, height: int, width: int) -> bool:
    h, w = reserved.shape
    r0 = max(0, top - PLACEMENT_PADDING)
    r1 = min(h, top + height + PLACEMENT_PADDING)
    c0 = max(0, left - PLACEMENT_PADDING)
    c1 = min(w, left + width + PLACEMENT_PADDING)
    return not bool(np.any(reserved[r0:r1, c0:c1]))


def _reserve_with_padding(reserved: np.ndarray, top: int, left: int, height: int, width: int) -> None:
    h, w = reserved.shape
    r0 = max(0, top - PLACEMENT_PADDING)
    r1 = min(h, top + height + PLACEMENT_PADDING)
    c0 = max(0, left - PLACEMENT_PADDING)
    c1 = min(w, left + width + PLACEMENT_PADDING)
    reserved[r0:r1, c0:c1] = True


def _try_place_object(grid: np.ndarray, reserved: np.ndarray) -> bool:
    h, w = grid.shape
    max_obj_h = min(BBOX_HEIGHT_RANGE[1], h - 2)
    max_obj_w = min(BBOX_WIDTH_RANGE[1], w - 2)
    if max_obj_h < BBOX_HEIGHT_RANGE[0] or max_obj_w < BBOX_WIDTH_RANGE[0]:
        return False

    for _ in range(PLACE_ATTEMPTS):
        obj_h = random.randint(BBOX_HEIGHT_RANGE[0], max_obj_h)
        obj_w = random.randint(BBOX_WIDTH_RANGE[0], max_obj_w)
        mask = _make_connected_mask(obj_h, obj_w)
        if mask is None:
            continue
        patch = _paint_random_colors(mask)

        top = random.randint(0, h - obj_h)
        left = random.randint(0, w - obj_w)
        if not _free_with_padding(reserved, top, left, obj_h, obj_w):
            continue

        grid[top : top + obj_h, left : left + obj_w][mask] = patch[mask]
        _reserve_with_padding(reserved, top, left, obj_h, obj_w)
        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_SIZE_RANGE)
    width = random.randint(*GRID_SIZE_RANGE)
    count = random.randint(*OBJECT_COUNT_RANGE)
    grid = np.full((height, width), BLACK, dtype=int)
    reserved = np.zeros((height, width), dtype=bool)

    for _ in range(count):
        if not _try_place_object(grid, reserved):
            return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid spiral-repacking input")
