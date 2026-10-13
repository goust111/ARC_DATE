import random
from collections import deque

import numpy as np


FRAME_COLOR = 5
SIDES = ("top", "bottom", "left", "right")
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

# Generator configuration.
GRID_LIMIT = 30
MIN_MARGIN = 1
MAX_MARGIN = 6
MIN_INTERIOR = 8
MAX_INTERIOR = 18
OBJECT_COUNT_CHOICES = (4, 5, 6)
OBJECT_SIZE_CHOICES = (3, 4, 5)
GEN_RETRIES = 400

# Rooted half-plane fallback shapes for the top-contact case.
FALLBACK_SHAPES = (
    {(0, 0), (1, 0), (1, 1)},
    {(0, 0), (1, 0), (2, 0), (2, 1)},
    {(0, 0), (1, 0), (1, 1), (2, 1)},
    {(0, 0), (1, 0), (2, 0), (1, 1)},
    {(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)},
    {(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)},
)


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _find_frame(grid: np.ndarray, bg: int) -> tuple[int, int, int, int, int]:
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        pts = np.argwhere(grid == color)
        if pts.size == 0:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        r0 = int(r0)
        c0 = int(c0)
        r1 = int(r1)
        c1 = int(c1)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            continue
        perimeter = 2 * (h + w) - 4
        if int(pts.shape[0]) != perimeter:
            continue
        if not np.all(grid[r0, c0 : c1 + 1] == color):
            continue
        if not np.all(grid[r1, c0 : c1 + 1] == color):
            continue
        if not np.all(grid[r0 : r1 + 1, c0] == color):
            continue
        if not np.all(grid[r0 : r1 + 1, c1] == color):
            continue
        if np.any(grid[r0 + 1 : r1, c0 + 1 : c1] == color):
            continue
        return color, r0, c0, r1, c1
    raise AssertionError("expected one hollow rectangular frame")


def _collect_objects(
    grid: np.ndarray, bg: int, frame_color: int, frame_bbox: tuple[int, int, int, int]
) -> list[tuple[int, np.ndarray]]:
    r0, c0, r1, c1 = frame_bbox
    visited = np.zeros(grid.shape, dtype=bool)
    objects: list[tuple[int, np.ndarray]] = []
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if visited[r, c]:
                continue
            color = int(grid[r, c])
            if color in (bg, frame_color):
                continue
            q = deque([(r, c)])
            visited[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if not (r0 + 1 <= nr < r1 and c0 + 1 <= nc < c1):
                        continue
                    if visited[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    q.append((nr, nc))
            objects.append((color, np.array(coords, dtype=int)))
    return objects


def _contact_side(coords: np.ndarray, frame_bbox: tuple[int, int, int, int]) -> str:
    r0, c0, r1, c1 = frame_bbox
    counts = {
        "top": int(np.sum(coords[:, 0] == r0 + 1)),
        "bottom": int(np.sum(coords[:, 0] == r1 - 1)),
        "left": int(np.sum(coords[:, 1] == c0 + 1)),
        "right": int(np.sum(coords[:, 1] == c1 - 1)),
    }
    touched = [side for side, count in counts.items() if count > 0]
    assert len(touched) == 1, "each object must touch exactly one frame side"
    return touched[0]


def _mirror_cell(
    r: int, c: int, side: str, frame_bbox: tuple[int, int, int, int]
) -> tuple[int, int]:
    r0, c0, r1, c1 = frame_bbox
    if side == "top":
        return 2 * r0 - r, c
    if side == "bottom":
        return 2 * r1 - r, c
    if side == "left":
        return r, 2 * c0 - c
    return r, 2 * c1 - c


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each frame-touching interior object across the contacted frame side onto the outside.

    Input structure:
    - The background is the dominant color.
    - A single-pixel-thick hollow rectangular frame sits near the center of the grid.
    - Several small monochrome 4-connected objects lie inside the frame, and each object has exactly one pixel 4-adjacent to exactly one frame side.
    - The exterior of the frame starts as background only.

    Rule:
    - Keep the original objects and the frame unchanged.
    - For each interior object, detect which frame side it uniquely touches.
    - Reflect every pixel of that object across the touched frame side and paint the reflected copy outside the frame in the same color.
    - If reflected pixels fall outside the grid, draw only the in-bounds part.

    Key insight:
    - The single touching pixel identifies the mirror axis: the contacted frame side determines both the flip direction and where the outside copy must appear.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    frame_color, r0, c0, r1, c1 = _find_frame(g, bg)
    out = g.copy()

    for color, coords in _collect_objects(g, bg, frame_color, (r0, c0, r1, c1)):
        side = _contact_side(coords, (r0, c0, r1, c1))
        for r, c in coords:
            rr, cc = _mirror_cell(int(r), int(c), side, (r0, c0, r1, c1))
            if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]:
                out[rr, cc] = color
    return out


def _cells_to_mask(cells: set[tuple[int, int]]) -> np.ndarray:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0 = min(rows)
    c0 = min(cols)
    arr = np.zeros((max(rows) - r0 + 1, max(cols) - c0 + 1), dtype=int)
    for r, c in cells:
        arr[r - r0, c - c0] = 1
    return arr


def _is_asymmetric(cells: set[tuple[int, int]]) -> bool:
    arr = _cells_to_mask(cells)
    return not np.array_equal(arr, np.flipud(arr)) and not np.array_equal(arr, np.fliplr(arr))


def _random_rooted_shape(size: int) -> set[tuple[int, int]]:
    for _ in range(120):
        cells = {(0, 0)}
        while len(cells) < size:
            frontier = set()
            for r, c in cells:
                for dr, dc in DIR4:
                    nr = r + dr
                    nc = c + dc
                    if nr < 0 or nr > 4 or abs(nc) > 3:
                        continue
                    if nr == 0 and nc != 0:
                        continue
                    if (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))
        if len(cells) != size:
            continue
        if sum(r == 0 for r, _ in cells) != 1:
            continue
        if random.random() < 0.5:
            cells = {(r, -c) for r, c in cells}
        if _is_asymmetric(cells):
            return cells
    fallback = random.choice([shape for shape in FALLBACK_SHAPES if len(shape) == size])
    if random.random() < 0.5:
        fallback = {(r, -c) for r, c in fallback}
    return fallback


def _orient_shape(cells: set[tuple[int, int]], side: str) -> list[tuple[int, int]]:
    if side == "top":
        out = list(cells)
    elif side == "bottom":
        out = [(-r, -c) for r, c in cells]
    elif side == "left":
        out = [(-c, r) for r, c in cells]
    else:
        out = [(c, -r) for r, c in cells]
    return [(int(r), int(c)) for r, c in out]


def _neighbor8(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    out = []
    for rr in range(r - 1, r + 2):
        for cc in range(c - 1, c + 2):
            out.append((rr, cc))
    return out


def _touches_occupied(cells: set[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    for cell in cells:
        for near in _neighbor8(cell):
            if near in occupied:
                return True
    return False


def _visible_mirror_cells(
    cells: set[tuple[int, int]], side: str, frame_bbox: tuple[int, int, int, int], shape: tuple[int, int]
) -> set[tuple[int, int]]:
    h, w = shape
    visible = set()
    for r, c in cells:
        rr, cc = _mirror_cell(r, c, side, frame_bbox)
        if 0 <= rr < h and 0 <= cc < w:
            visible.add((rr, cc))
    return visible


def _balanced_sides(n_objects: int) -> list[str]:
    sides = []
    while len(sides) < n_objects:
        block = list(SIDES)
        random.shuffle(block)
        sides.extend(block)
    return sides[:n_objects]


def _sample_dimensions(n_objects: int) -> tuple[int, int, int, int, int, int]:
    min_interior = MIN_INTERIOR + (1 if n_objects >= 6 else 0)
    interior_h = random.randint(min_interior, MAX_INTERIOR)
    interior_w = random.randint(min_interior, MAX_INTERIOR)
    margins = [random.randint(MIN_MARGIN, MAX_MARGIN) for _ in range(4)]
    top, bottom, left, right = margins
    h = top + interior_h + 2 + bottom
    w = left + interior_w + 2 + right
    if h > GRID_LIMIT or w > GRID_LIMIT:
        raise ValueError("sampled grid exceeds ARC limit")
    return h, w, top, left, top + interior_h + 1, left + interior_w + 1


def _candidate_roots(
    side: str,
    offsets: list[tuple[int, int]],
    frame_bbox: tuple[int, int, int, int],
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = frame_bbox
    drs = [r for r, _ in offsets]
    dcs = [c for _, c in offsets]
    min_r = min(drs)
    max_r = max(drs)
    min_c = min(dcs)
    max_c = max(dcs)

    roots = []
    if side == "top":
        root_r = r0 + 1
        if root_r + max_r > r1 - 2:
            return []
        start = c0 + 2 - min_c
        end = c1 - 2 - max_c
        roots = [(root_r, cc) for cc in range(start, end + 1)]
    elif side == "bottom":
        root_r = r1 - 1
        if root_r + min_r < r0 + 2:
            return []
        start = c0 + 2 - min_c
        end = c1 - 2 - max_c
        roots = [(root_r, cc) for cc in range(start, end + 1)]
    elif side == "left":
        root_c = c0 + 1
        if root_c + max_c > c1 - 2:
            return []
        start = r0 + 2 - min_r
        end = r1 - 2 - max_r
        roots = [(rr, root_c) for rr in range(start, end + 1)]
    else:
        root_c = c1 - 1
        if root_c + min_c < c0 + 2:
            return []
        start = r0 + 2 - min_r
        end = r1 - 2 - max_r
        roots = [(rr, root_c) for rr in range(start, end + 1)]

    random.shuffle(roots)
    return roots


def _place_object(
    grid: np.ndarray,
    color: int,
    side: str,
    frame_bbox: tuple[int, int, int, int],
    occupied_input: set[tuple[int, int]],
    occupied_output: set[tuple[int, int]],
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    size = random.choice(OBJECT_SIZE_CHOICES)
    rooted = _random_rooted_shape(size)
    offsets = _orient_shape(rooted, side)
    for root_r, root_c in _candidate_roots(side, offsets, frame_bbox):
        cells = {(root_r + dr, root_c + dc) for dr, dc in offsets}
        if _touches_occupied(cells, occupied_input):
            continue
        if any(grid[r, c] != _get_bg(grid) for r, c in cells):
            continue
        mirrors = _visible_mirror_cells(cells, side, frame_bbox, grid.shape)
        if not mirrors:
            continue
        if _touches_occupied(mirrors, occupied_output):
            continue
        for r, c in cells:
            grid[r, c] = color
        return cells, mirrors
    return None


def _build_case() -> np.ndarray | None:
    n_objects = random.choice(OBJECT_COUNT_CHOICES)
    try:
        h, w, r0, c0, r1, c1 = _sample_dimensions(n_objects)
    except ValueError:
        return None

    bg_choices = [color for color in range(10) if color != FRAME_COLOR]
    bg = random.choice(bg_choices)
    obj_colors = random.sample([color for color in range(10) if color not in (bg, FRAME_COLOR)], n_objects)

    grid = np.full((h, w), bg, dtype=int)
    grid[r0, c0 : c1 + 1] = FRAME_COLOR
    grid[r1, c0 : c1 + 1] = FRAME_COLOR
    grid[r0 : r1 + 1, c0] = FRAME_COLOR
    grid[r0 : r1 + 1, c1] = FRAME_COLOR

    occupied_input = set()
    occupied_output = set()
    frame_bbox = (r0, c0, r1, c1)
    for color, side in zip(obj_colors, _balanced_sides(n_objects), strict=False):
        placed = None
        for _ in range(40):
            placed = _place_object(grid, color, side, frame_bbox, occupied_input, occupied_output)
            if placed is not None:
                break
        if placed is None:
            return None
        cells, mirrors = placed
        occupied_input |= cells
        occupied_output |= mirrors

    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None

    objects = _collect_objects(grid, bg, FRAME_COLOR, frame_bbox)
    if len(objects) != n_objects:
        return None
    if len({int(color) for color, _coords in objects}) != n_objects:
        return None
    for _color, coords in objects:
        side = _contact_side(coords, frame_bbox)
        counts = {
            "top": int(np.sum(coords[:, 0] == r0 + 1)),
            "bottom": int(np.sum(coords[:, 0] == r1 - 1)),
            "left": int(np.sum(coords[:, 1] == c0 + 1)),
            "right": int(np.sum(coords[:, 1] == c1 - 1)),
        }
        if counts[side] != 1:
            return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _build_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
