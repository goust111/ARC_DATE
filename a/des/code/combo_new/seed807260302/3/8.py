from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _legend_candidates(corner: str, h: int, w: int) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    if corner == "tl":
        return (0, 0), [(0, 1), (1, 0)]
    if corner == "tr":
        return (0, w - 1), [(0, w - 4), (1, w - 3)]
    if corner == "bl":
        return (h - 1, 0), [(h - 3, 1), (h - 4, 0)]
    if corner == "br":
        return (h - 1, w - 1), [(h - 3, w - 4), (h - 4, w - 3)]
    raise ValueError(corner)


def _read_legends(g: np.ndarray) -> tuple[dict[int, np.ndarray], np.ndarray]:
    h, w = g.shape
    assert h >= 6 and w >= 6, "grid must be large enough for four corner legends"

    legends: dict[int, np.ndarray] = {}
    legend_mask = np.zeros(g.shape, dtype=bool)

    for corner in ("tl", "tr", "bl", "br"):
        (mr, mc), origins = _legend_candidates(corner, h, w)
        marker = int(g[mr, mc])
        assert marker != BG, "legend marker must be a non-black wall color"

        valid = []
        for r0, c0 in origins:
            if r0 < 0 or c0 < 0 or r0 + 3 > h or c0 + 3 > w:
                continue
            patch = g[r0 : r0 + 3, c0 : c0 + 3]
            valid.append((int(np.count_nonzero(patch != BG)), r0, c0, patch.copy()))
        assert valid, "legend must have an adjacent 3x3 stamp patch"
        valid.sort(key=lambda item: item[0], reverse=True)
        non_black, r0, c0, patch = valid[0]
        assert non_black > 0, "stamp patch cannot be all black"
        assert marker not in legends, "wall-color legend keys must be unique"

        legends[marker] = patch
        legend_mask[mr, mc] = True
        legend_mask[r0 : r0 + 3, c0 : c0 + 3] = True

    return legends, legend_mask


def _same_color_components(g: np.ndarray, mask: np.ndarray) -> list[np.ndarray]:
    h, w = g.shape
    seen = np.zeros(g.shape, dtype=bool)
    comps: list[np.ndarray] = []

    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        color = int(g[r, c])
        q: deque[tuple[int, int]] = deque([(r, c)])
        seen[r, c] = True
        pts: list[tuple[int, int]] = []

        while q:
            cr, cc = q.popleft()
            pts.append((cr, cc))
            for dr, dc in NEIGHBORS4:
                nr, nc = cr + dr, cc + dc
                if not _in_bounds(g.shape, nr, nc):
                    continue
                if seen[nr, nc] or not mask[nr, nc] or int(g[nr, nc]) != color:
                    continue
                seen[nr, nc] = True
                q.append((nr, nc))

        comps.append(np.array(pts, dtype=int))

    return comps


def _wall_mask(g: np.ndarray, wall_colors: set[int], legend_mask: np.ndarray) -> np.ndarray:
    wall = np.zeros(g.shape, dtype=bool)
    for color in wall_colors:
        mask = (g == color) & (~legend_mask)
        for pts in _same_color_components(g, mask):
            if pts.shape[0] < 3:
                continue
            rows = set(map(int, pts[:, 0]))
            cols = set(map(int, pts[:, 1]))
            if len(rows) == 1 or len(cols) == 1:
                wall[pts[:, 0], pts[:, 1]] = True
    return wall


def _object_components(g: np.ndarray, legend_mask: np.ndarray, wall: np.ndarray) -> list[np.ndarray]:
    mask = (g != BG) & (~legend_mask) & (~wall)
    return [pts for pts in _same_color_components(g, mask) if pts.shape[0] >= 4]


def _touching_wall_color(g: np.ndarray, pts: np.ndarray, wall: np.ndarray) -> int:
    colors: set[int] = set()
    for r_raw, c_raw in pts:
        r = int(r_raw)
        c = int(c_raw)
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if _in_bounds(g.shape, nr, nc) and wall[nr, nc]:
                colors.add(int(g[nr, nc]))
    assert len(colors) == 1, "each object must 4-touch exactly one wall color"
    return next(iter(colors))


def _stamp_center(pts: np.ndarray) -> tuple[int, int]:
    return int(np.floor(float(np.mean(pts[:, 0])))), int(np.floor(float(np.mean(pts[:, 1]))))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp each colored object with the 3x3 pattern keyed by the wall color it touches.

    Input structure:
    - Black is the background, with several isolated 4-connected monochrome objects in the central area.
    - Colored one-cell-thick horizontal or vertical wall segments separate the objects; each object 4-touches exactly one wall.
    - Each grid corner contains a legend entry: a one-pixel wall-color marker and an adjacent 3x3 stamp pattern.

    Rule:
    - Read the corner legends as a mapping from wall color to 3x3 stamp.
    - For each central object, find the color of the single wall it 4-touches.
    - Place the matching 3x3 stamp with its center at the floor of the object's pixel-coordinate centroid.
    - Non-black stamp cells overwrite the scene, while black stamp cells are transparent; walls and legends remain unchanged.

    Key insight:
    - The object does not choose its stamp by its own color; the adjacent wall color is the lookup key into the corner legend.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    legends, legend_mask = _read_legends(g)
    walls = _wall_mask(g, set(legends), legend_mask)
    assert np.any(walls), "expected at least one colored wall segment"

    out = g.copy()
    protected = legend_mask | walls
    for pts in _object_components(g, legend_mask, walls):
        wall_color = _touching_wall_color(g, pts, walls)
        stamp = legends[wall_color]
        center_r, center_c = _stamp_center(pts)

        for sr in range(3):
            for sc in range(3):
                color = int(stamp[sr, sc])
                if color == BG:
                    continue
                rr = center_r + sr - 1
                cc = center_c + sc - 1
                if _in_bounds(g.shape, rr, cc) and not protected[rr, cc]:
                    out[rr, cc] = color

    return out


# --------------- generator ---------------

GRID_MIN, GRID_MAX = 18, 25
ALL_COLORS = tuple(range(1, 10))
MAX_GENERATE_ATTEMPTS = 400


def _stamp_templates(colors: list[int]) -> list[np.ndarray]:
    a, b, c, d, e = colors[:5]
    return [
        np.array([[BG, a, BG], [a, a, a], [BG, a, BG]], dtype=int),
        np.array([[b, BG, b], [BG, b, BG], [b, BG, b]], dtype=int),
        np.array([[c, c, c], [c, BG, c], [c, c, c]], dtype=int),
        np.array([[d, d, d], [BG, e, BG], [BG, e, BG]], dtype=int),
    ]


def _place_legends(grid: np.ndarray, wall_colors: list[int], stamps: list[np.ndarray]) -> np.ndarray:
    h, w = grid.shape
    legend_mask = np.zeros(grid.shape, dtype=bool)
    specs = [
        ((0, 0), (0, 1)),
        ((0, w - 1), (0, w - 4)),
        ((h - 1, 0), (h - 3, 1)),
        ((h - 1, w - 1), (h - 3, w - 4)),
    ]

    for color, stamp, ((mr, mc), (pr, pc)) in zip(wall_colors, stamps, specs, strict=False):
        grid[mr, mc] = color
        grid[pr : pr + 3, pc : pc + 3] = stamp
        legend_mask[mr, mc] = True
        legend_mask[pr : pr + 3, pc : pc + 3] = True
    return legend_mask


def _base_shape_masks() -> list[np.ndarray]:
    return [
        np.ones((3, 3), dtype=bool),
        np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
        np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
        np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=bool),
        np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
        np.ones((3, 4), dtype=bool),
        np.ones((4, 3), dtype=bool),
    ]


def _shape_variants() -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[int, int, bytes]] = set()
    for mask in _base_shape_masks():
        for k in range(4):
            rot = np.rot90(mask, k)
            for variant in (rot, np.fliplr(rot), np.flipud(rot)):
                arr = variant.copy()
                key = (arr.shape[0], arr.shape[1], arr.tobytes())
                if key not in seen:
                    seen.add(key)
                    variants.append(arr)
    return variants


SHAPES = _shape_variants()


def _zones(h: int, w: int) -> list[tuple[int, int, int, int]]:
    top = 4
    bottom = h - 5
    left = 4
    right = w - 5
    mid_r = (top + bottom) // 2
    mid_c = (left + right) // 2
    return [
        (top, mid_r - 1, left, mid_c - 1),
        (top, mid_r - 1, mid_c + 1, right),
        (mid_r + 1, bottom, left, mid_c - 1),
        (mid_r + 1, bottom, mid_c + 1, right),
    ]


def _valid_sides(mask: np.ndarray) -> list[str]:
    pts = np.argwhere(mask)
    center_r = int(np.floor(float(np.mean(pts[:, 0]))))
    center_c = int(np.floor(float(np.mean(pts[:, 1]))))
    h, w = mask.shape
    sides = []
    if np.any(mask[0, :]) and center_r >= 1:
        sides.append("top")
    if np.any(mask[h - 1, :]) and center_r <= h - 2:
        sides.append("bottom")
    if np.any(mask[:, 0]) and center_c >= 1:
        sides.append("left")
    if np.any(mask[:, w - 1]) and center_c <= w - 2:
        sides.append("right")
    return sides


def _randint(lo: int, hi: int) -> int | None:
    if lo > hi:
        return None
    return random.randint(lo, hi)


def _feature_candidate(
    zone: tuple[int, int, int, int],
    mask: np.ndarray,
    side: str,
    ext: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    zr0, zr1, zc0, zc1 = zone
    sh, sw = mask.shape

    if side == "top":
        r0 = _randint(zr0 + 1, zr1 - sh + 1)
        c0 = _randint(zc0 + ext, zc1 - sw + 1 - ext)
        if r0 is None or c0 is None:
            return None
        wall = [(r0 - 1, c) for c in range(c0 - ext, c0 + sw + ext)]
    elif side == "bottom":
        r0 = _randint(zr0, zr1 - sh)
        c0 = _randint(zc0 + ext, zc1 - sw + 1 - ext)
        if r0 is None or c0 is None:
            return None
        wall = [(r0 + sh, c) for c in range(c0 - ext, c0 + sw + ext)]
    elif side == "left":
        r0 = _randint(zr0 + ext, zr1 - sh + 1 - ext)
        c0 = _randint(zc0 + 1, zc1 - sw + 1)
        if r0 is None or c0 is None:
            return None
        wall = [(r, c0 - 1) for r in range(r0 - ext, r0 + sh + ext)]
    else:
        assert side == "right"
        r0 = _randint(zr0 + ext, zr1 - sh + 1 - ext)
        c0 = _randint(zc0, zc1 - sw)
        if r0 is None or c0 is None:
            return None
        wall = [(r, c0 + sw) for r in range(r0 - ext, r0 + sh + ext)]

    obj = [(r0 + int(r), c0 + int(c)) for r, c in np.argwhere(mask)]
    all_cells = obj + wall
    if any(not (zr0 <= r <= zr1 and zc0 <= c <= zc1) for r, c in all_cells):
        return None
    return obj, wall


def _place_object_and_wall(
    grid: np.ndarray,
    zone: tuple[int, int, int, int],
    obj_color: int,
    wall_color: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    random.shuffle(SHAPES)
    for mask in SHAPES:
        sides = _valid_sides(mask)
        random.shuffle(sides)
        for side in sides:
            for ext in random.sample([0, 1], 2):
                result = _feature_candidate(zone, mask, side, ext)
                if result is None:
                    continue
                obj, wall = result
                if any(grid[r, c] != BG for r, c in obj + wall):
                    continue
                for r, c in obj:
                    grid[r, c] = obj_color
                for r, c in wall:
                    grid[r, c] = wall_color
                return obj, wall
    return None


def _expected_output(
    inp: np.ndarray,
    legend_mask: np.ndarray,
    tasks: list[tuple[list[tuple[int, int]], list[tuple[int, int]], np.ndarray]],
) -> np.ndarray:
    out = inp.copy()
    wall_mask = np.zeros(inp.shape, dtype=bool)
    for _obj, wall, _stamp in tasks:
        for r, c in wall:
            wall_mask[r, c] = True
    protected = legend_mask | wall_mask

    for obj, _wall, stamp in tasks:
        pts = np.array(obj, dtype=int)
        center_r, center_c = _stamp_center(pts)
        for sr in range(3):
            for sc in range(3):
                color = int(stamp[sr, sc])
                if color == BG:
                    continue
                rr = center_r + sr - 1
                cc = center_c + sc - 1
                if _in_bounds(inp.shape, rr, cc) and not protected[rr, cc]:
                    out[rr, cc] = color
    return out


def _valid_generated(inp: np.ndarray, expected: np.ndarray, task_count: int) -> bool:
    if max(inp.shape) > 30:
        return False
    if task_count < 3:
        return False
    try:
        actual = main(inp)
    except Exception:
        return False
    return bool(np.array_equal(actual, expected))


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)

    colors = list(ALL_COLORS)
    random.shuffle(colors)
    wall_colors = colors[:4]
    object_palette = colors[4:]
    stamp_colors = random.sample(list(ALL_COLORS), 5)
    stamps = _stamp_templates(stamp_colors)

    grid = np.full((h, w), BG, dtype=int)
    legend_mask = _place_legends(grid, wall_colors, stamps)

    zones = _zones(h, w)
    zone_order = list(range(4))
    random.shuffle(zone_order)
    task_count = random.randint(3, 4)
    use_indices = zone_order[:task_count]

    tasks: list[tuple[list[tuple[int, int]], list[tuple[int, int]], np.ndarray]] = []
    for i in use_indices:
        obj_color = random.choice(object_palette)
        placed = _place_object_and_wall(grid, zones[i], obj_color, wall_colors[i])
        if placed is None:
            return None
        obj, wall = placed
        tasks.append((obj, wall, stamps[i]))

    expected = _expected_output(grid, legend_mask, tasks)
    if not _valid_generated(grid, expected, task_count):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stamp-by-wall-color input")
