import random

import numpy as np


BG = 0
CROSS = 5
RED = 2
GREEN = 3
OBJECT_COLORS = (1, 4, 6, 7, 8, 9)
GRID_HEIGHTS = (15, 17, 19, 21)
GRID_WIDTHS = (15, 17, 19, 21)
GENERATION_RETRIES = 400
PLACEMENT_RETRIES = 120

BASE_MASKS = [
    np.array([[1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=bool),
]


def _most_common_color(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel())
    return int(np.argmax(counts))


def _find_cross(grid: np.ndarray, bg: int) -> tuple[int, int, int]:
    colors = [int(color) for color in np.unique(grid) if int(color) != bg]
    candidates: list[tuple[int, int, int]] = []
    for color in colors:
        rows = np.flatnonzero(np.all(grid == color, axis=1))
        cols = np.flatnonzero(np.all(grid == color, axis=0))
        if len(rows) == 1 and len(cols) == 1:
            r = int(rows[0])
            c = int(cols[0])
            if int(grid[r, c]) == color:
                candidates.append((color, r, c))
    assert len(candidates) == 1, "expected a unique full-row/full-column cross"
    return candidates[0]


def _find_components(grid: np.ndarray, ignore_colors: set[int]) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in ignore_colors or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            coords = np.array(cells, dtype=int)
            coord_set = {tuple(map(int, cell)) for cell in coords}
            objects.append({"color": color, "coords": coords, "coord_set": coord_set})

    return objects


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _center_from_bbox(coords: np.ndarray) -> tuple[int, int]:
    r0, c0, r1, c1 = _bbox(coords)
    assert (r0 + r1) % 2 == 0 and (c0 + c1) % 2 == 0, "object center must be a cell"
    return (r0 + r1) // 2, (c0 + c1) // 2


def _quadrant(center: tuple[int, int], cross_r: int, cross_c: int) -> str | None:
    r, c = center
    if r < cross_r and c < cross_c:
        return "UL"
    if r < cross_r and c > cross_c:
        return "UR"
    if r > cross_r and c < cross_c:
        return "LL"
    if r > cross_r and c > cross_c:
        return "LR"
    return None


def _line_points(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc

    points = []
    r, c = r0, c0
    while True:
        points.append((r, c))
        if (r, c) == (r1, c1):
            return points
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect same-color objects that are point-symmetric across the grey cross.

    Objects mirrored between the upper-left and lower-right quadrants receive a
    red line; objects mirrored between the upper-right and lower-left quadrants
    receive a green line. Lines are only drawn onto original background cells.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _most_common_color(g)
    cross_color, cross_r, cross_c = _find_cross(g, bg)
    objects = _find_components(g, {bg, cross_color})

    by_color: dict[int, list[dict[str, object]]] = {}
    for idx, obj in enumerate(objects):
        center = _center_from_bbox(obj["coords"])
        quadrant = _quadrant(center, cross_r, cross_c)
        assert quadrant is not None, "objects must lie entirely within one quadrant"
        assert all(r != cross_r and c != cross_c for r, c in obj["coord_set"])
        obj["id"] = idx
        obj["center"] = center
        obj["quadrant"] = quadrant
        by_color.setdefault(int(obj["color"]), []).append(obj)

    out = g.copy()
    matched: set[int] = set()
    for color, comps in by_color.items():
        center_map = {tuple(comp["center"]): comp for comp in comps}
        for comp in comps:
            if int(comp["id"]) in matched:
                continue

            center = tuple(comp["center"])
            if center[0] >= cross_r:
                continue

            partner_center = (2 * cross_r - center[0], 2 * cross_c - center[1])
            partner = center_map.get(partner_center)
            if partner is None or int(partner["id"]) in matched:
                continue

            mirrored = {(2 * cross_r - r, 2 * cross_c - c) for r, c in comp["coord_set"]}
            if mirrored != partner["coord_set"]:
                continue

            pair = (str(comp["quadrant"]), str(partner["quadrant"]))
            if pair == ("UL", "LR"):
                line_color = RED
            elif pair == ("UR", "LL"):
                line_color = GREEN
            else:
                continue

            for r, c in _line_points(center, partner_center):
                if int(g[r, c]) == bg:
                    prev = int(out[r, c])
                    assert prev in {bg, line_color}, "different connector lines must stay disjoint"
                    out[r, c] = line_color

            matched.add(int(comp["id"]))
            matched.add(int(partner["id"]))

    return out


def _rotated_mask(mask: np.ndarray) -> np.ndarray:
    return np.rot90(mask, random.randrange(4)).copy()


def _mask_coords(mask: np.ndarray, center: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = mask.shape
    r0 = center[0] - h // 2
    c0 = center[1] - w // 2
    coords = set()
    for r in range(h):
        for c in range(w):
            if bool(mask[r, c]):
                coords.add((r0 + r, c0 + c))
    return coords


def _reserve_halo(blocked: np.ndarray, coords: set[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in coords:
        for rr in range(max(0, r - 1), min(h - 1, r + 1) + 1):
            for cc in range(max(0, c - 1), min(w - 1, c + 1) + 1):
                blocked[rr, cc] = True


def _paint_object(grid: np.ndarray, coords: set[tuple[int, int]], color: int) -> None:
    for r, c in coords:
        grid[r, c] = color


def _candidate_centers(
    shape: tuple[int, int],
    quadrant: str,
    cross_r: int,
    cross_c: int,
    height: int,
    width: int,
) -> list[tuple[int, int]]:
    h, w = shape
    hr = h // 2
    hc = w // 2
    if quadrant == "UL":
        rows = range(hr, cross_r - 1 - hr + 1)
        cols = range(hc, cross_c - 1 - hc + 1)
    elif quadrant == "UR":
        rows = range(hr, cross_r - 1 - hr + 1)
        cols = range(cross_c + 1 + hc, width - hc)
    else:
        raise AssertionError(f"unsupported quadrant: {quadrant}")
    return [(r, c) for r in rows for c in cols]


def _pair_line_cells(
    a: tuple[int, int],
    b: tuple[int, int],
    cross_r: int,
    cross_c: int,
    own_cells: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    cells = set()
    for cell in _line_points(a, b):
        if cell in own_cells:
            continue
        if cell[0] == cross_r or cell[1] == cross_c:
            continue
        cells.add(cell)
    return cells


def _try_place_pair(
    grid: np.ndarray,
    blocked: np.ndarray,
    occupied: set[tuple[int, int]],
    line_reserved: set[tuple[int, int]],
    quadrant: str,
    color: int,
    mask: np.ndarray,
    cross_r: int,
    cross_c: int,
) -> bool:
    height, width = grid.shape
    candidates = _candidate_centers(mask.shape, quadrant, cross_r, cross_c, height, width)
    random.shuffle(candidates)

    for center in candidates[:PLACEMENT_RETRIES]:
        coords_a = _mask_coords(mask, center)
        coords_b = {(2 * cross_r - r, 2 * cross_c - c) for r, c in coords_a}
        pair_cells = coords_a | coords_b
        if any(blocked[r, c] or (r, c) in line_reserved for r, c in pair_cells):
            continue

        partner_center = (2 * cross_r - center[0], 2 * cross_c - center[1])
        line_cells = _pair_line_cells(center, partner_center, cross_r, cross_c, pair_cells)
        if any(cell in occupied or cell in line_reserved for cell in line_cells):
            continue

        _paint_object(grid, coords_a, color)
        _paint_object(grid, coords_b, color)
        occupied.update(pair_cells)
        line_reserved.update(line_cells)
        _reserve_halo(blocked, coords_a)
        _reserve_halo(blocked, coords_b)
        return True

    return False


def _try_place_noise(
    grid: np.ndarray,
    blocked: np.ndarray,
    line_reserved: set[tuple[int, int]],
    color: int,
    mask: np.ndarray,
    quadrant: str,
    cross_r: int,
    cross_c: int,
) -> bool:
    height, width = grid.shape
    candidates = _candidate_centers(mask.shape, quadrant, cross_r, cross_c, height, width)
    random.shuffle(candidates)
    for center in candidates[:PLACEMENT_RETRIES]:
        coords = _mask_coords(mask, center)
        if any(blocked[r, c] or (r, c) in line_reserved for r, c in coords):
            continue
        _paint_object(grid, coords, color)
        _reserve_halo(blocked, coords)
        return True
    return False


def _blank_grid(height: int, width: int) -> tuple[np.ndarray, np.ndarray, int, int]:
    grid = np.full((height, width), BG, dtype=int)
    cross_r = height // 2
    cross_c = width // 2
    grid[cross_r, :] = CROSS
    grid[:, cross_c] = CROSS

    blocked = np.zeros((height, width), dtype=bool)
    for r in range(height):
        blocked[r, cross_c] = True
        if cross_c > 0:
            blocked[r, cross_c - 1] = True
        if cross_c + 1 < width:
            blocked[r, cross_c + 1] = True
    for c in range(width):
        blocked[cross_r, c] = True
        if cross_r > 0:
            blocked[cross_r - 1, c] = True
        if cross_r + 1 < height:
            blocked[cross_r + 1, c] = True

    return grid, blocked, cross_r, cross_c


def _gen_attempt() -> np.ndarray | None:
    height = random.choice(GRID_HEIGHTS)
    width = random.choice(GRID_WIDTHS)
    grid, blocked, cross_r, cross_c = _blank_grid(height, width)
    occupied: set[tuple[int, int]] = set()
    line_reserved: set[tuple[int, int]] = set()

    colors = list(OBJECT_COLORS)
    random.shuffle(colors)

    pair_quadrants = ["UL", "UR"]
    if random.random() < 0.45:
        pair_quadrants.append(random.choice(["UL", "UR"]))
    if len(pair_quadrants) > len(colors):
        return None

    pair_colors = [colors.pop() for _ in range(len(pair_quadrants))]
    pair_specs = list(zip(pair_quadrants, pair_colors))
    random.shuffle(pair_specs)
    for quadrant, color in pair_specs:
        mask = _rotated_mask(random.choice(BASE_MASKS))
        if not _try_place_pair(
            grid,
            blocked,
            occupied,
            line_reserved,
            quadrant,
            color,
            mask,
            cross_r,
            cross_c,
        ):
            return None

    max_noise = min(len(colors), 3)
    noise_count = random.randint(0, max_noise)
    random.shuffle(colors)
    for color in colors[:noise_count]:
        mask = _rotated_mask(random.choice(BASE_MASKS))
        quadrant = random.choice(["UL", "UR"])
        if not _try_place_noise(grid, blocked, line_reserved, color, mask, quadrant, cross_r, cross_c):
            return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    if not np.any(out == RED) or not np.any(out == GREEN):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
