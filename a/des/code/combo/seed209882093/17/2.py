import random

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR_BY_NAME = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}


GRID_MIN = 12
GRID_MAX = 20
LEGEND_LEN_MIN = 3
LEGEND_LEN_MAX = 7
MIN_ARM_LEN = 2
MAX_ARM_LEN = 6
MAX_TOWERS = 6
GEN_RETRIES = 500
COLOR_POOL = tuple(range(1, 10))
CORE_OPTIONS = [
    ("L", "U"),
    ("L", "D"),
    ("R", "U"),
    ("R", "D"),
    ("L", "R", "U"),
    ("L", "R", "D"),
    ("L", "U", "D"),
    ("R", "U", "D"),
    ("L", "R", "U", "D"),
]
CORE_WEIGHTS = [3, 3, 3, 3, 2, 2, 2, 2, 1]
LEGEND_CORNERS = ("tl", "tr", "bl", "br")


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _find_legend(grid: np.ndarray, bg: int) -> tuple[np.ndarray, dict[int, int]]:
    h, w = grid.shape
    candidates: list[tuple[list[int], int]] = []

    for coords in _components4(grid != bg):
        if coords.shape[0] < LEGEND_LEN_MIN:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))

        if c0 != c1:
            continue
        if c0 not in (0, w - 1):
            continue
        if coords.shape[0] != (r1 - r0 + 1):
            continue
        if not (r0 == 0 or r1 == h - 1):
            continue

        inner_c = 1 if c0 == 0 else w - 2
        if w > 1 and np.any(grid[r0 : r1 + 1, inner_c] != bg):
            continue

        rows = list(range(r0, r1 + 1))
        colors = [int(grid[r, c0]) for r in rows]
        if len(set(colors)) != len(colors):
            continue
        candidates.append((rows, c0))

    assert len(candidates) == 1, "expected a unique corner legend"
    rows, col = candidates[0]

    legend_mask = np.zeros(grid.shape, dtype=bool)
    legend_mask[rows, col] = True
    priority = {int(grid[r, col]): i for i, r in enumerate(rows)}
    return legend_mask, priority


def _trace_rays(
    grid: np.ndarray, bg: int, legend_mask: np.ndarray
) -> tuple[list[tuple[int, int, int]], list[list[set[int]]]]:
    h, w = grid.shape
    towers = [
        (int(r), int(c), int(grid[r, c]))
        for r, c in np.argwhere((grid != bg) & (~legend_mask))
    ]
    hits = [[set() for _ in range(w)] for _ in range(h)]

    for r, c, color in towers:
        for dr, dc in DIR4:
            rr, cc = r + dr, c + dc
            while 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == bg:
                hits[rr][cc].add(color)
                rr += dr
                cc += dc

    return towers, hits


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each single-pixel tower as cardinal rays, using the corner legend to
    resolve crossing colors by priority.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _bg_color(g)
    legend_mask, priority = _find_legend(g, bg)
    towers, hits = _trace_rays(g, bg, legend_mask)

    assert towers, "expected at least one signal tower"
    assert all(color in priority for _, _, color in towers), "tower color missing from legend"

    out = g.copy()
    h, w = g.shape
    for r in range(h):
        for c in range(w):
            if int(g[r, c]) != bg:
                continue
            if not hits[r][c]:
                continue
            out[r, c] = min(hits[r][c], key=lambda color: priority[color])
    return out


def _legend_cells(h: int, w: int, corner: str, length: int) -> list[tuple[int, int]]:
    if corner == "tl":
        rows = range(0, length)
        col = 0
    elif corner == "tr":
        rows = range(0, length)
        col = w - 1
    elif corner == "bl":
        rows = range(h - length, h)
        col = 0
    else:
        rows = range(h - length, h)
        col = w - 1
    return [(int(r), int(col)) for r in rows]


def _mark_chebyshev(mask: np.ndarray, r: int, c: int, radius: int = 1) -> None:
    h, w = mask.shape
    r0 = max(0, r - radius)
    r1 = min(h, r + radius + 1)
    c0 = max(0, c - radius)
    c1 = min(w, c + radius + 1)
    mask[r0:r1, c0:c1] = True


def _available_steps(mask: np.ndarray, r: int, c: int, dr: int, dc: int) -> int:
    h, w = mask.shape
    steps = 0
    rr, cc = r + dr, c + dc
    while 0 <= rr < h and 0 <= cc < w and not mask[rr, cc]:
        steps += 1
        rr += dr
        cc += dc
    return steps


def _mark_segment(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> None:
    r0, c0 = start
    r1, c1 = end
    if r0 == r1:
        a, b = sorted((c0, c1))
        mask[r0, a : b + 1] = True
    else:
        a, b = sorted((r0, r1))
        mask[a : b + 1, c0] = True


def _count_cells(hits: list[list[set[int]]]) -> tuple[int, int]:
    mixed = 0
    painted = 0
    for row in hits:
        for colors in row:
            if not colors:
                continue
            painted += 1
            if len(colors) >= 2:
                mixed += 1
    return mixed, painted


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    core_dirs = random.choices(CORE_OPTIONS, weights=CORE_WEIGHTS, k=1)[0]

    legend_len_hi = min(LEGEND_LEN_MAX, h - 4, len(COLOR_POOL))
    min_towers = max(3, len(core_dirs))
    max_towers = min(MAX_TOWERS, legend_len_hi)
    if legend_len_hi < LEGEND_LEN_MIN or min_towers > max_towers:
        return None

    tower_count = random.randint(min_towers, max_towers)
    legend_len = random.randint(max(LEGEND_LEN_MIN, tower_count), legend_len_hi)
    legend_colors = random.sample(COLOR_POOL, legend_len)
    tower_colors = random.sample(legend_colors, tower_count)
    corner = random.choice(LEGEND_CORNERS)

    grid = np.zeros((h, w), dtype=int)
    legend_cells = sorted(_legend_cells(h, w, corner, legend_len))
    for color, (r, c) in zip(legend_colors, legend_cells):
        grid[r, c] = color

    forbidden = np.zeros((h, w), dtype=bool)
    forbidden[0, :] = True
    forbidden[-1, :] = True
    forbidden[:, 0] = True
    forbidden[:, -1] = True
    for r, c in legend_cells:
        _mark_chebyshev(forbidden, r, c, radius=1)

    centers: list[tuple[int, int, dict[str, int]]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if forbidden[r, c]:
                continue
            avail = {
                d: _available_steps(forbidden, r, c, DIR_BY_NAME[d][0], DIR_BY_NAME[d][1])
                for d in core_dirs
            }
            if all(avail[d] >= MIN_ARM_LEN for d in core_dirs):
                centers.append((r, c, avail))
    if not centers:
        return None

    center_r, center_c, avail = random.choice(centers)
    core_towers: list[tuple[int, int, int]] = []
    for i, d in enumerate(core_dirs):
        dr, dc = DIR_BY_NAME[d]
        max_len = min(avail[d], MAX_ARM_LEN)
        dist = random.randint(MIN_ARM_LEN, max_len)
        rr = center_r + dr * dist
        cc = center_c + dc * dist
        core_towers.append((int(rr), int(cc), int(tower_colors[i])))

    occupied = set(legend_cells)
    placement_mask = forbidden.copy()
    _mark_chebyshev(placement_mask, center_r, center_c, radius=0)
    for r, c, _color in core_towers:
        if (r, c) in occupied or placement_mask[r, c]:
            return None
        occupied.add((r, c))
        _mark_chebyshev(placement_mask, r, c, radius=1)
        _mark_segment(placement_mask, (center_r, center_c), (r, c))
        grid[r, c] = _color

    extra_colors = tower_colors[len(core_towers) :]
    random.shuffle(extra_colors)
    for color in extra_colors:
        candidates = [
            (r, c)
            for r in range(1, h - 1)
            for c in range(1, w - 1)
            if not placement_mask[r, c]
        ]
        if not candidates:
            return None
        r, c = random.choice(candidates)
        grid[r, c] = color
        _mark_chebyshev(placement_mask, r, c, radius=1)

    try:
        bg = _bg_color(grid)
        legend_mask, _priority = _find_legend(grid, bg)
        towers, hits = _trace_rays(grid, bg, legend_mask)
        mixed, painted = _count_cells(hits)
        out = main(grid)
    except AssertionError:
        return None

    if len(towers) != tower_count:
        return None
    if mixed < 1 or painted < tower_count + 2:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
