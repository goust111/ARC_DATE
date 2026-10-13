import random
from collections import Counter, deque

import numpy as np


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def find_components(grid: np.ndarray, *, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            color = int(g[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if not in_bounds(g.shape, nr, nc) or seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "area": len(coords),
                }
            )
    return comps


def _detect_component_centers(grid: np.ndarray, comp: dict) -> list[tuple[int, int]]:
    color = int(comp["color"])
    centers = []
    for r, c in comp["coords"]:
        if all(in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == color for nr, nc in neighbors4(r, c)):
            centers.append((int(r), int(c)))
    return centers


def _detect_cross_centers(grid: np.ndarray) -> list[tuple[int, int, int]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    centers = []
    for comp in find_components(g, bg=bg):
        for r, c in _detect_component_centers(g, comp):
            centers.append((r, c, int(comp["color"])))
    return sorted(centers)


def _paint_segment(
    out: np.ndarray, src: np.ndarray, p0: tuple[int, int], p1: tuple[int, int], color: int, bg: int
) -> None:
    r0, c0 = p0
    r1, c1 = p1
    color = int(color)
    bg = int(bg)
    if r0 == r1:
        for cc in range(min(c0, c1) + 1, max(c0, c1)):
            if int(src[r0, cc]) == bg:
                if int(out[r0, cc]) not in (bg, color):
                    raise AssertionError("conflicting painted row segment")
                out[r0, cc] = color
    else:
        for rr in range(min(r0, r1) + 1, max(r0, r1)):
            if int(src[rr, c0]) == bg:
                if int(out[rr, c0]) not in (bg, color):
                    raise AssertionError("conflicting painted column segment")
                out[rr, c0] = color


def main(grid: np.ndarray) -> np.ndarray:
    """Reveal hidden cross centers inside monochrome blobs and connect same-color aligned pairs through background.

    Input structure:
    - A background canvas contains several separated monochromatic 4-connected blobs; colors may repeat across blobs.
    - Some blobs hide a cross center: one interior pixel whose four orthogonal neighbors belong to the same blob.
    - Other blob protrusions and centerless blobs act as distractors but stay unchanged.

    Rule:
    - Detect every blob pixel whose four orthogonal neighbors have the blob's color.
    - For each pair of detected centers that share a row or a column and whose parent blobs have the same color, paint the strictly intermediate background cells on that segment with that color.
    - Existing non-background blob pixels are never overwritten, and all original blobs remain in place.

    Key insight:
    - The endpoints are implicit, not explicitly marked: only a local four-neighbor test reveals which irregular blobs contribute line anchors, while aligned centers of different colors must be ignored.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    out = g.copy()
    centers = _detect_cross_centers(g)
    for i, (r0, c0, color0) in enumerate(centers):
        for r1, c1, color1 in centers[i + 1 :]:
            if color0 != color1:
                continue
            if r0 == r1 or c0 == c1:
                _paint_segment(out, g, (r0, c0), (r1, c1), color0, bg)
    return out


# --------------- generator ---------------

_BG = 0
_PALETTE = tuple(range(1, 10))
_GRID_MIN = 24
_GRID_MAX = 29
_MARGIN = 3
_SLOT_STEP = 7
_LOCAL_RADIUS = 2
_GEN_ATTEMPTS = 500


def _local_centers(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    return {
        (r, c)
        for r, c in cells
        if all((nr, nc) in cells for nr, nc in neighbors4(r, c))
    }


def _shape_frontier(cells: set[tuple[int, int]], radius: int) -> list[tuple[int, int]]:
    frontier = set()
    for r, c in cells:
        for nr, nc in neighbors4(r, c):
            if (nr, nc) in cells:
                continue
            if max(abs(nr), abs(nc)) <= radius:
                frontier.add((nr, nc))
    return sorted(frontier)


def _build_cross_blob(rng: random.Random) -> set[tuple[int, int]]:
    base = {(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)}
    for _ in range(200):
        cells = set(base)
        target = rng.randint(7, 10)
        while len(cells) < target:
            frontier = _shape_frontier(cells, _LOCAL_RADIUS)
            if not frontier:
                break
            cells.add(rng.choice(frontier))
        if _local_centers(cells) != {(0, 0)}:
            continue
        if len(cells) <= len(base) + 1:
            continue
        if not any(abs(r) == _LOCAL_RADIUS or abs(c) == _LOCAL_RADIUS or (abs(r) == 1 and abs(c) == 1) for r, c in cells):
            continue
        bbox_area = (max(r for r, _ in cells) - min(r for r, _ in cells) + 1) * (
            max(c for _, c in cells) - min(c for _, c in cells) + 1
        )
        if bbox_area == len(cells):
            continue
        return cells
    raise AssertionError("failed to build a single-center blob")


def _build_centerless_blob(rng: random.Random) -> set[tuple[int, int]]:
    for _ in range(200):
        cells = {(0, 0), rng.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])}
        target = rng.randint(4, 7)
        while len(cells) < target:
            frontier = _shape_frontier(cells, _LOCAL_RADIUS)
            if not frontier:
                break
            cells.add(rng.choice(frontier))
        if _local_centers(cells):
            continue
        bbox_area = (max(r for r, _ in cells) - min(r for r, _ in cells) + 1) * (
            max(c for _, c in cells) - min(c for _, c in cells) + 1
        )
        if bbox_area <= len(cells):
            continue
        return cells
    raise AssertionError("failed to build a centerless blob")


def _translate(cells: set[tuple[int, int]], origin: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0 = origin
    return {(r0 + r, c0 + c) for r, c in cells}


def _paint_blob(grid: np.ndarray, cells: set[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)


def _cells_with_border(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    out = set(cells)
    for r, c in list(cells):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                out.add((r + dr, c + dc))
    return out


def _oracle_output(grid: np.ndarray, centers: list[tuple[int, int, int]]) -> np.ndarray:
    out = grid.copy()
    bg = get_bg(grid)
    for i, (r0, c0, color0) in enumerate(centers):
        for r1, c1, color1 in centers[i + 1 :]:
            if color0 != color1:
                continue
            if r0 == r1 or c0 == c1:
                _paint_segment(out, grid, (r0, c0), (r1, c1), color0, bg)
    return out


def _aligned_counts(centers: list[tuple[int, int, int]]) -> tuple[int, int, int]:
    same_color_pairs = 0
    diff_color_pairs = 0
    axes = 0
    for i, (r0, c0, color0) in enumerate(centers):
        for r1, c1, color1 in centers[i + 1 :]:
            if r0 != r1 and c0 != c1:
                continue
            if color0 == color1:
                same_color_pairs += 1
                axes |= 1 if r0 == r1 else 2
            else:
                diff_color_pairs += 1
    return same_color_pairs, diff_color_pairs, axes


def _slot_coords(size: int) -> list[int]:
    coords = list(range(_MARGIN, size - _MARGIN, _SLOT_STEP))
    return [value for value in coords if _MARGIN <= value < size - _MARGIN]


def _attempt_generate(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(_GRID_MIN, _GRID_MAX)
    w = rng.randint(_GRID_MIN, _GRID_MAX)
    rows = _slot_coords(h)
    cols = _slot_coords(w)
    if len(rows) < 3 or len(cols) < 3:
        return None

    row_triplet = sorted(rng.sample(rows, 3))
    col_triplet = sorted(rng.sample(cols, 3))
    r_top, r_mid, r_bot = row_triplet
    c_left, c_mid, c_right = col_triplet

    color_row, color_mid, color_col = rng.sample(_PALETTE, 3)
    center_specs = [
        ((r_mid, c_left), color_row),
        ((r_mid, c_mid), color_mid),
        ((r_mid, c_right), color_row),
        ((r_top, c_mid), color_col),
        ((r_bot, c_mid), color_col),
    ]

    grid = np.zeros((h, w), dtype=int)
    reserved = set()
    designed_centers = []
    used_slots = {origin for origin, _ in center_specs}

    for origin, color in center_specs:
        cells = _translate(_build_cross_blob(rng), origin)
        if any(not in_bounds(grid.shape, r, c) for r, c in cells):
            return None
        if reserved & cells:
            return None
        if reserved & _cells_with_border(cells):
            return None
        _paint_blob(grid, cells, color)
        reserved |= _cells_with_border(cells)
        designed_centers.append((origin[0], origin[1], color))

    free_slots = [(r, c) for r in rows for c in cols if (r, c) not in used_slots]
    rng.shuffle(free_slots)
    distractor_count = min(len(free_slots), rng.randint(1, 3))
    for origin in free_slots[:distractor_count]:
        color = rng.choice(_PALETTE)
        cells = _translate(_build_centerless_blob(rng), origin)
        if any(not in_bounds(grid.shape, r, c) for r, c in cells):
            return None
        if reserved & cells or reserved & _cells_with_border(cells):
            return None
        _paint_blob(grid, cells, color)
        reserved |= _cells_with_border(cells)

    detected = _detect_cross_centers(grid)
    if detected != sorted(designed_centers):
        return None

    same_color_pairs, diff_color_pairs, axes = _aligned_counts(detected)
    if same_color_pairs < 2 or diff_color_pairs < 2 or axes != 3:
        return None

    out = _oracle_output(grid, detected)
    if np.array_equal(out, grid):
        return None

    painted = np.count_nonzero((grid == _BG) & (out != _BG))
    if painted < 6:
        return None

    solver_out = main(grid)
    if not np.array_equal(solver_out, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(_GEN_ATTEMPTS):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC sample")
