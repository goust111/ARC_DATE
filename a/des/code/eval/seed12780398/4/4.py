import random
from collections import deque

import numpy as np


BLACK = 0
GREEN = 3
GRAY = 5
PALETTE = [1, 2, 4, 6, 7, 8, 9]

PATTERN_LIBRARY = [
    np.array([[1, 1, 1], [1, 0, 0], [1, 1, 1]], dtype=int),
    np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 1], [1, 1, 0]], dtype=int),
    np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=int),
    np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]], dtype=int),
    np.array([[0, 1, 1], [1, 1, 0], [1, 0, 1]], dtype=int),
    np.array([[1, 1, 0], [1, 0, 1], [1, 1, 0]], dtype=int),
]

GEN_RETRIES = 300
PLACEMENT_RETRIES = 100


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _intervals(mask_1d: np.ndarray) -> list[tuple[int, int]]:
    groups = []
    start = None
    for idx, flag in enumerate(mask_1d.tolist()):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            groups.append((start, idx))
            start = None
    if start is not None:
        groups.append((start, len(mask_1d)))
    return groups


def _is_rect_frame(points: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(points)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 5 or w < 5:
        return False
    expected = 2 * (h + w) - 4
    if len(points) != expected:
        return False
    frame = set(points)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_border = r in (r0, r1) or c in (c0, c1)
            if on_border != ((r, c) in frame):
                return False
    return True


def _find_legend_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    candidates = []
    for pts in _components(grid == GRAY):
        if _is_rect_frame(pts):
            bbox = _bbox(pts)
            candidates.append((bbox[2], bbox[1], bbox))
    if not candidates:
        raise ValueError("legend frame not found")
    return max(candidates)[2]


def _extract_textures(grid: np.ndarray, legend_bbox: tuple[int, int, int, int]) -> dict[int, np.ndarray]:
    r0, c0, r1, c1 = legend_bbox
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    used_rows = np.where(np.any(inner != BLACK, axis=1))[0]
    if len(used_rows) == 0:
        raise ValueError("legend is empty")
    band = inner[used_rows.min() : used_rows.max() + 1, :]
    if band.shape[0] != 3:
        raise ValueError("legend samples must be 3 rows tall")
    used_cols = np.any(band != BLACK, axis=0)
    groups = _intervals(used_cols)
    textures: dict[int, np.ndarray] = {}
    for start, end in groups:
        if end - start != 3:
            raise ValueError("legend sample width must be 3")
        tile = band[:, start:end]
        colors = [int(v) for v in np.unique(tile) if int(v) != BLACK]
        if len(colors) != 1:
            raise ValueError("legend sample must use exactly one color")
        textures[colors[0]] = tile.copy()
    if not textures:
        raise ValueError("no textures parsed from legend")
    return textures


def _scaled_tile(tile: np.ndarray, scale: int) -> np.ndarray:
    return np.kron(tile, np.ones((scale, scale), dtype=int))


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    h, w = g.shape

    legend_bbox = _find_legend_bbox(g)
    textures = _extract_textures(g, legend_bbox)

    blocked = g == GREEN
    lr0, lc0, lr1, lc1 = legend_bbox
    blocked[lr0 : lr1 + 1, lc0 : lc1 + 1] = True

    for pts in _components(~blocked):
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in pts):
            continue

        colors = [int(g[r, c]) for r, c in pts if int(g[r, c]) not in (BLACK,)]
        if not colors:
            continue
        color_set = set(colors)
        if len(color_set) != 1:
            raise ValueError("each region must contain exactly one seed color")
        color = colors[0]
        seed_count = len(colors)
        if seed_count not in (1, 2, 3):
            raise ValueError("seed count must be 1..3")
        if color not in textures:
            raise ValueError("missing matching legend texture")

        r0, c0, r1, c1 = _bbox(pts)
        tile = _scaled_tile(textures[color], seed_count)
        th, tw = tile.shape
        for r, c in pts:
            out[r, c] = int(tile[(r - r0) % th, (c - c0) % tw])

    return out


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rs0 = max(0, -dr)
            rs1 = min(h, h - dr)
            cs0 = max(0, -dc)
            cs1 = min(w, w - dc)
            rd0 = max(0, dr)
            rd1 = rd0 + (rs1 - rs0)
            cd0 = max(0, dc)
            cd1 = cd0 + (cs1 - cs0)
            out[rd0:rd1, cd0:cd1] |= mask[rs0:rs1, cs0:cs1]
    return out


def _random_polyomino(rng: random.Random) -> set[tuple[int, int]] | None:
    rows = rng.randint(2, 4)
    cols = rng.randint(2, 5)
    max_cells = rows * cols - 1
    if max_cells < 4:
        return None
    target = rng.randint(4, max_cells)
    start = (rng.randrange(rows), rng.randrange(cols))
    cells = {start}
    while len(cells) < target:
        frontier = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        cells.add(rng.choice(frontier))

    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    norm = {(r - min_r, c - min_c) for r, c in cells}
    height = max(r for r, _ in norm) + 1
    width = max(c for _, c in norm) + 1
    if height < 2 or width < 2:
        return None
    if len(norm) == height * width:
        return None
    return norm


def _build_region_shape(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    cells = _random_polyomino(rng)
    if cells is None:
        return None
    cell_size = rng.randint(2, 3)
    height = (max(r for r, _ in cells) + 1) * cell_size
    width = (max(c for _, c in cells) + 1) * cell_size
    interior = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        rr = r * cell_size
        cc = c * cell_size
        interior[rr : rr + cell_size, cc : cc + cell_size] = True
    padded_inside = np.pad(interior, 1, constant_values=False)
    border = _dilate8(padded_inside) & ~padded_inside
    return border, padded_inside


def _choose_seed_positions(
    rng: random.Random, interior_coords: list[tuple[int, int]], count: int
) -> list[tuple[int, int]]:
    for _ in range(40):
        choice = rng.sample(interior_coords, count)
        ok = True
        for i in range(len(choice)):
            for j in range(i + 1, len(choice)):
                if max(abs(choice[i][0] - choice[j][0]), abs(choice[i][1] - choice[j][1])) <= 1:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return choice
    return rng.sample(interior_coords, count)


def _region_specs(rng: random.Random, colors: list[int], n_regions: int) -> list[tuple[int, int]]:
    color_a, color_b = rng.sample(colors, 2)
    counts = rng.sample([1, 2, 3], 3)
    specs = [
        (color_a, counts[0]),
        (color_a, counts[1]),
        (color_b, counts[0]),
    ]
    if n_regions >= 4:
        specs.append((color_b, counts[2]))
    while len(specs) < n_regions:
        specs.append((rng.choice(colors), rng.randint(1, 3)))
    rng.shuffle(specs)
    return specs


def _paint_legend(
    grid: np.ndarray,
    legend_top: int,
    legend_left: int,
    legend_width: int,
    legend_height: int,
    textures: dict[int, np.ndarray],
    order: list[int],
):
    r0 = legend_top
    c0 = legend_left
    r1 = legend_top + legend_height - 1
    c1 = legend_left + legend_width - 1
    grid[r0, c0 : c1 + 1] = GRAY
    grid[r1, c0 : c1 + 1] = GRAY
    grid[r0 : r1 + 1, c0] = GRAY
    grid[r0 : r1 + 1, c1] = GRAY
    band = grid[r0 + 1 : r0 + 4, c0 + 1 : c1]
    for idx, color in enumerate(order):
        start = idx * 4
        tile = textures[color]
        band[:, start : start + 3] = tile


def _place_region(
    grid: np.ndarray,
    forbidden: np.ndarray,
    legend_top: int,
    color: int,
    seed_count: int,
    rng: random.Random,
) -> bool:
    h, w = grid.shape
    for _ in range(PLACEMENT_RETRIES):
        built = _build_region_shape(rng)
        if built is None:
            continue
        border, interior = built
        rh, rw = border.shape
        max_top = legend_top - rh - 2
        max_left = w - rw - 2
        if max_top < 1 or max_left < 1:
            continue
        top = rng.randint(1, max_top)
        left = rng.randint(1, max_left)
        footprint = border | interior
        area = forbidden[top : top + rh, left : left + rw]
        if np.any(area & footprint):
            continue

        inside_coords = list(map(tuple, np.argwhere(interior)))
        if len(inside_coords) < seed_count:
            continue
        seeds = _choose_seed_positions(rng, inside_coords, seed_count)

        grid[top : top + rh, left : left + rw][border] = GREEN
        for sr, sc in seeds:
            grid[top + sr, left + sc] = color

        reserved = np.zeros_like(grid, dtype=bool)
        reserved[top : top + rh, left : left + rw] = footprint
        forbidden |= _dilate8(reserved)
        return True
    return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_textures = rng.randint(2, 4)
    n_regions = rng.randint(3, 5)

    colors = rng.sample(PALETTE, n_textures)
    patterns = rng.sample(PATTERN_LIBRARY, n_textures)
    texture_map = {color: pattern * color for color, pattern in zip(colors, patterns, strict=False)}
    legend_order = colors[:]
    rng.shuffle(legend_order)

    legend_inner_width = 4 * n_textures - 1
    legend_width = legend_inner_width + 2
    legend_height = 5

    width = rng.randint(max(24, legend_width + 4), 34)
    height = rng.randint(22, 30)
    legend_top = height - legend_height
    legend_left = rng.randint(1, width - legend_width - 1)

    if legend_top < 10:
        return None

    grid = np.zeros((height, width), dtype=int)
    _paint_legend(grid, legend_top, legend_left, legend_width, legend_height, texture_map, legend_order)

    forbidden = np.zeros_like(grid, dtype=bool)
    forbidden[legend_top : legend_top + legend_height, legend_left : legend_left + legend_width] = True
    forbidden = _dilate8(forbidden)

    for color, seed_count in _region_specs(rng, colors, n_regions):
        if not _place_region(grid, forbidden, legend_top, color, seed_count, rng):
            return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
