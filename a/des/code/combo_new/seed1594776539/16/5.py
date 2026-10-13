import random
from collections import defaultdict

import numpy as np


BG = 0
WALL = 5
CELL_H = 2
CELL_W = 3
MAX_GEN_RETRIES = 400
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
PATTERN_LIBRARY = [
    np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
]
PHASE_BY_COUNT = {1: (0, 0), 2: (0, 1), 3: (1, 1)}


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
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
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _parse_legend(legend: np.ndarray, wall_color: int) -> dict[int, np.ndarray]:
    content = np.any((legend != BG) & (legend != wall_color), axis=0)
    runs: list[tuple[int, int]] = []
    c = 0
    while c < legend.shape[1]:
        if not content[c]:
            c += 1
            continue
        start = c
        while c < legend.shape[1] and content[c]:
            c += 1
        runs.append((start, c))
    mapping: dict[int, np.ndarray] = {}
    for start, end in runs:
        assert end - start == 3, "legend blocks must be 3 columns wide"
        block = legend[:, start:end]
        colors = set(map(int, block.flatten())) - {BG, wall_color}
        assert len(colors) == 1, "each legend block must use exactly one main color"
        color = int(next(iter(colors)))
        mapping[color] = block == color
    assert 2 <= len(mapping) <= 4, "legend must contain 2-4 textures"
    return mapping


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h >= 7 and w >= 7
    sep = h - 4
    wall_color = int(g[sep, 0])
    assert wall_color != BG and np.all(g[sep] == wall_color), "expected a full-width separator row"

    textures = _parse_legend(g[sep + 1 : sep + 4], wall_color)
    work = g[: sep + 1]
    out = g.copy()

    for comp in _components(work != wall_color):
        vals = [int(work[r, c]) for r, c in comp if int(work[r, c]) not in (BG, wall_color)]
        assert vals, "each region must contain at least one seed"
        seed_color = vals[0]
        assert all(v == seed_color for v in vals), "all seeds in a region must share one color"
        seed_count = len(vals)
        assert seed_count in PHASE_BY_COUNT, "seed count must be 1, 2, or 3"
        assert seed_color in textures, "seed color must appear in the legend"
        pattern = textures[seed_color]
        pr, pc = PHASE_BY_COUNT[seed_count]
        for r, c in comp:
            out[r, c] = seed_color if pattern[(r - pr) % 3, (c - pc) % 3] else BG
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed region with the legend texture selected by its seed color and shifted by its seed count.

    Input structure:
    - Background is black.
    - Grey wall pixels partition the upper part of the grid into enclosed orthogonal regions, and a full-width grey separator sits above the bottom legend.
    - The bottom legend has 2-4 side-by-side 3x3 binary textures, each using black plus one distinct main color.
    - Every enclosed region contains 1-3 same-color seed pixels whose color matches one legend texture.

    Rule:
    - Read the bottom legend as a mapping from main color to a 3x3 texture mask.
    - Split the upper field into enclosed regions using the grey wall network.
    - For each region, choose the texture whose main color matches that region's seed color, tile it across the whole grid with origin at the top-left, and keep only the pixels that fall inside the region.
    - Use no phase shift for 1 seed, a 1-cell horizontal shift for 2 seeds, and a 1-cell horizontal-plus-vertical shift for 3 seeds; wall pixels and the legend stay unchanged, and black texture cells stay black.

    Key insight:
    - Seed color selects which texture to use, while seed count controls the tiling phase, so identical colors can produce different placements in different regions.
    """
    return _solve(grid)


def _wall_mask(room_rows: int, room_cols: int) -> np.ndarray:
    h = room_rows * (CELL_H + 1)
    w = room_cols * (CELL_W + 1) + 1
    mask = np.zeros((h, w), dtype=bool)
    mask[:, :: CELL_W + 1] = True
    mask[:: CELL_H + 1, :] = True
    return mask


def _room_id(rr: int, cc: int, room_cols: int) -> int:
    return rr * room_cols + cc


def _room_cells(rr: int, cc: int) -> tuple[slice, slice]:
    r0 = rr * (CELL_H + 1) + 1
    c0 = cc * (CELL_W + 1) + 1
    return slice(r0, r0 + CELL_H), slice(c0, c0 + CELL_W)


def _segment_cells(kind: str, rr: int, cc: int) -> tuple[slice, slice]:
    if kind == "v":
        col = (cc + 1) * (CELL_W + 1)
        r0 = rr * (CELL_H + 1) + 1
        return slice(r0, r0 + CELL_H), slice(col, col + 1)
    row = (rr + 1) * (CELL_H + 1)
    c0 = cc * (CELL_W + 1) + 1
    return slice(row, row + 1), slice(c0, c0 + CELL_W)


def _mask_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return True
    start = tuple(map(int, pts[0]))
    seen = {start}
    stack = [start]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


class _DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return True


def _coarse_groups(dsu: _DSU, room_rows: int, room_cols: int) -> dict[int, list[tuple[int, int]]]:
    groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for rr in range(room_rows):
        for cc in range(room_cols):
            groups[dsu.find(_room_id(rr, cc, room_cols))].append((rr, cc))
    return dict(groups)


def _build_partition(room_rows: int, room_cols: int, rng: random.Random) -> np.ndarray | None:
    nrooms = room_rows * room_cols
    target_regions = rng.randint(max(4, nrooms // 3), min(10, max(4, (nrooms + 1) // 2)))
    target_merges = nrooms - target_regions

    wall = _wall_mask(room_rows, room_cols)
    dsu = _DSU(nrooms)
    candidates: list[tuple[str, int, int]] = []
    for rr in range(room_rows):
        for cc in range(room_cols - 1):
            candidates.append(("v", rr, cc))
    for rr in range(room_rows - 1):
        for cc in range(room_cols):
            candidates.append(("h", rr, cc))

    merges = 0
    for _ in range(12):
        if merges >= target_merges:
            break
        rng.shuffle(candidates)
        progressed = False
        for kind, rr, cc in candidates:
            if merges >= target_merges:
                break
            a = _room_id(rr, cc, room_cols)
            b = _room_id(rr, cc + 1, room_cols) if kind == "v" else _room_id(rr + 1, cc, room_cols)
            ra = dsu.find(a)
            rb = dsu.find(b)
            if ra == rb:
                continue
            if dsu.size[ra] + dsu.size[rb] > 3:
                continue
            rs, cs = _segment_cells(kind, rr, cc)
            trial = wall.copy()
            trial[rs, cs] = False
            if not _mask_connected(trial):
                continue
            wall = trial
            dsu.union(ra, rb)
            merges += 1
            progressed = True
        if not progressed:
            break

    if merges != target_merges:
        return None

    groups = _coarse_groups(dsu, room_rows, room_cols)
    if len(groups) < 4:
        return None

    irregular = False
    for cells in groups.values():
        if len(cells) < 3:
            continue
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        if (max(rs) - min(rs) + 1) * (max(cs) - min(cs) + 1) > len(cells):
            irregular = True
            break
    if not irregular:
        return None
    return wall


def _grid_from_wall(wall: np.ndarray) -> np.ndarray:
    h, w = wall.shape
    grid = np.full((h + 4, w), BG, dtype=int)
    grid[:h][wall] = WALL
    grid[h] = WALL
    return grid


def _regions(grid: np.ndarray) -> list[np.ndarray]:
    sep = grid.shape[0] - 4
    work = grid[: sep + 1]
    return _components(work != WALL)


def _place_legend(grid: np.ndarray, color_to_pattern: dict[int, np.ndarray], rng: random.Random) -> None:
    sep = grid.shape[0] - 4
    legend = grid[sep + 1 : sep + 4]
    w = grid.shape[1]
    colors = list(color_to_pattern)
    span = 4 * len(colors) - 1
    start = rng.randint(0, w - span)
    for i, color in enumerate(colors):
        c0 = start + 4 * i
        legend[:, c0 : c0 + 3] = np.where(color_to_pattern[color], color, BG)
        if i + 1 < len(colors):
            legend[:, c0 + 3] = WALL


def _assign_region_specs(
    regions: list[np.ndarray], colors: list[int], rng: random.Random
) -> tuple[dict[int, int], dict[int, int]]:
    order = list(range(len(regions)))
    rng.shuffle(order)
    demo_color = colors[0]
    region_color: dict[int, int] = {}
    region_count: dict[int, int] = {}

    for idx, count in zip(order[:3], [1, 2, 3]):
        region_color[idx] = demo_color
        region_count[idx] = count

    cursor = 3
    for color in colors[1:]:
        idx = order[cursor]
        cursor += 1
        region_color[idx] = color
        region_count[idx] = rng.choice([1, 2, 3])

    for idx in order[cursor:]:
        region_color[idx] = rng.choice(colors)
        region_count[idx] = rng.choice([1, 2, 3])
    return region_color, region_count


def _paint_seeds(
    grid: np.ndarray,
    regions: list[np.ndarray],
    region_color: dict[int, int],
    region_count: dict[int, int],
    rng: random.Random,
) -> None:
    for idx, comp in enumerate(regions):
        color = region_color[idx]
        count = region_count[idx]
        choices = [tuple(map(int, pt)) for pt in comp]
        picks = rng.sample(choices, count)
        for r, c in picks:
            grid[r, c] = color


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    room_rows = rng.randint(3, 5)
    room_cols = rng.randint(4, 6)
    wall = _build_partition(room_rows, room_cols, rng)
    if wall is None:
        return None

    grid = _grid_from_wall(wall)
    regions = _regions(grid)
    nregions = len(regions)
    legend_cap = min(4, nregions - 2, (grid.shape[1] + 1) // 4)
    if legend_cap < 2:
        return None

    ntextures = rng.randint(2, legend_cap)
    colors = rng.sample(COLOR_POOL, ntextures)
    patterns = rng.sample(PATTERN_LIBRARY, ntextures)
    color_to_pattern = {color: pattern.copy() for color, pattern in zip(colors, patterns)}

    region_color, region_count = _assign_region_specs(regions, colors, rng)
    _paint_seeds(grid, regions, region_color, region_count, rng)
    _place_legend(grid, color_to_pattern, rng)

    # Recompute via the solver once to reject malformed instances early.
    _ = _solve(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        rng = random.Random()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid textured-region puzzle")
