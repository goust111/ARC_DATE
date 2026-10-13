from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
MAROON = 9
SEED_COLORS = (1, 2, 3, 6, 7, 8)
DIRS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
PATTERN_BANK: tuple[tuple[tuple[int, int], ...], ...] = (
    ((-1, 0),),
    ((1, 0),),
    ((0, -1),),
    ((0, 1),),
    ((-1, -1),),
    ((1, 1),),
    ((-1, 1),),
    ((1, -1),),
    ((-1, 0), (0, 1)),
    ((1, 0), (0, -1)),
    ((-1, 0), (1, 0)),
    ((0, -1), (0, 1)),
    ((-1, -1), (1, 1)),
    ((-1, 1), (1, -1)),
    ((-1, 0), (0, 1), (1, 0), (0, -1)),
    ((-1, -1), (-1, 1), (1, -1), (1, 1)),
    ((-1, 0), (0, 1), (1, -1)),
    ((-1, 1), (0, -1), (1, 0)),
)
GEN_RETRIES = 400
GRID_MIN = 16
GRID_MAX = 30


def _components(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))
    else:
        dirs = DIRS8
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            stack = [(r, c)]
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for dr, dc in dirs:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _pattern_to_directions(pattern: np.ndarray) -> tuple[tuple[int, int], ...]:
    dirs: list[tuple[int, int]] = []
    for r in range(3):
        for c in range(3):
            if r == 1 and c == 1:
                continue
            if int(pattern[r, c]) == YELLOW:
                dirs.append((r - 1, c - 1))
    return tuple(dirs)


def _directions_to_pattern(directions: tuple[tuple[int, int], ...]) -> np.ndarray:
    pattern = np.full((3, 3), BLACK, dtype=int)
    pattern[1, 1] = YELLOW
    for dr, dc in directions:
        pattern[1 + dr, 1 + dc] = YELLOW
    return pattern


def _simulate_growth(
    inner_h: int,
    inner_w: int,
    seed_r: int,
    seed_c: int,
    directions: tuple[tuple[int, int], ...],
) -> set[tuple[int, int]]:
    filled = {(int(seed_r), int(seed_c))}
    q: deque[tuple[int, int]] = deque([(int(seed_r), int(seed_c))])
    while q:
        r, c = q.popleft()
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < inner_h and 0 <= nc < inner_w and (nr, nc) not in filled:
                filled.add((nr, nc))
                q.append((nr, nc))
    return filled


def _parse_legend(grid: np.ndarray) -> dict[int, tuple[tuple[int, int], ...]]:
    candidates: list[tuple[int, dict[int, tuple[tuple[int, int], ...]]]] = []
    for comp in _components(grid == MAROON, connectivity=4):
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 5 or w < 6:
            continue
        if not np.all(grid[r0, c0 : c1 + 1] == MAROON):
            continue
        if not np.all(grid[r1, c0 : c1 + 1] == MAROON):
            continue
        if not np.all(grid[r0 : r1 + 1, c0] == MAROON):
            continue
        if not np.all(grid[r0 : r1 + 1, c1] == MAROON):
            continue

        row_lines = [r for r in range(r0, r1 + 1) if np.all(grid[r, c0 : c1 + 1] == MAROON)]
        if len(row_lines) < 2 or row_lines[0] != r0 or row_lines[-1] != r1:
            continue
        if any(row_lines[i + 1] - row_lines[i] != 4 for i in range(len(row_lines) - 1)):
            continue

        mapping: dict[int, tuple[tuple[int, int], ...]] = {}
        ok = True
        for top, bottom in zip(row_lines, row_lines[1:], strict=False):
            if bottom - top != 4:
                ok = False
                break
            strip = grid[top + 1 : bottom, c0 + 1 : c1]
            if strip.shape[0] != 3 or strip.shape[1] < 4:
                ok = False
                break

            left_part = strip[:, : strip.shape[1] - 3]
            pattern = strip[:, -3:]
            swatch_colors = set(map(int, np.unique(left_part))) - {BLACK, YELLOW, MAROON}
            if len(swatch_colors) != 1:
                ok = False
                break
            swatch_color = int(next(iter(swatch_colors)))
            if int(np.sum(left_part == swatch_color)) != 1:
                ok = False
                break
            if not set(map(int, np.unique(pattern))) <= {BLACK, YELLOW}:
                ok = False
                break
            if int(pattern[1, 1]) != YELLOW:
                ok = False
                break
            if swatch_color in mapping:
                ok = False
                break
            mapping[swatch_color] = _pattern_to_directions(pattern)

        if ok and mapping:
            candidates.append((len(comp), mapping))

    assert candidates, "expected one maroon legend block"
    return max(candidates, key=lambda item: item[0])[1]


def _find_fences(grid: np.ndarray) -> list[tuple[int, int, int, int, int, int, int]]:
    fences: list[tuple[int, int, int, int, int, int, int]] = []
    for comp in _components(grid == GREY, connectivity=4):
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 5 or w < 5:
            continue
        if len(comp) != 2 * (h + w) - 4:
            continue
        if not np.all(grid[r0, c0 : c1 + 1] == GREY):
            continue
        if not np.all(grid[r1, c0 : c1 + 1] == GREY):
            continue
        if not np.all(grid[r0 : r1 + 1, c0] == GREY):
            continue
        if not np.all(grid[r0 : r1 + 1, c1] == GREY):
            continue
        if np.any(grid[r0 + 1 : r1, c0 + 1 : c1] == GREY):
            continue

        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        colors = set(map(int, np.unique(inner))) - {BLACK}
        if len(colors) != 1:
            continue
        seed_color = int(next(iter(colors)))
        seed_pts = np.argwhere(inner == seed_color)
        if seed_pts.shape[0] != 1:
            continue
        sr, sc = map(int, seed_pts[0])
        fences.append((r0, c0, r1, c1, seed_color, sr, sc))
    return fences


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each seed inside its grey fence according to the 3x3 yellow/black pattern
    associated with the seed color in the maroon legend.
    """
    g = np.asarray(grid, dtype=int)
    mapping = _parse_legend(g)
    out = g.copy()

    for r0, c0, r1, c1, seed_color, sr, sc in _find_fences(g):
        directions = mapping[seed_color]
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        filled = _simulate_growth(inner_h, inner_w, sr, sc, directions)
        for rr, cc in filled:
            out[r0 + 1 + rr, c0 + 1 + cc] = seed_color

    return out


def _choose_seed(inner_h: int, inner_w: int, directions: tuple[tuple[int, int], ...]) -> tuple[int, int] | None:
    choices = [(r, c) for r in range(inner_h) for c in range(inner_w)]
    random.shuffle(choices)
    for seed_r, seed_c in choices:
        area = len(_simulate_growth(inner_h, inner_w, seed_r, seed_c, directions))
        if area > 1:
            return int(seed_r), int(seed_c)
    return None


def _build_fence(spec: dict[str, object]) -> np.ndarray:
    inner_h = int(spec["inner_h"])
    inner_w = int(spec["inner_w"])
    seed_r = int(spec["seed_r"])
    seed_c = int(spec["seed_c"])
    color = int(spec["color"])
    tile = np.full((inner_h + 2, inner_w + 2), BLACK, dtype=int)
    tile[0, :] = GREY
    tile[-1, :] = GREY
    tile[:, 0] = GREY
    tile[:, -1] = GREY
    tile[1 + seed_r, 1 + seed_c] = color
    return tile


def _partition_indices(n: int, groups: int) -> list[list[int]]:
    if groups == 1:
        return [list(range(n))]
    cuts = sorted(random.sample(range(1, n), groups - 1))
    starts = [0] + cuts
    ends = cuts + [n]
    return [list(range(s, e)) for s, e in zip(starts, ends, strict=False)]


def _layout_fences(tiles: list[np.ndarray]) -> tuple[list[tuple[int, int]], int, int]:
    order = list(range(len(tiles)))
    random.shuffle(order)
    max_rows = min(len(tiles), 3)
    min_rows = 1 if len(tiles) <= 2 else 2
    row_count = random.randint(min_rows, max_rows)
    groups = _partition_indices(len(order), row_count)

    placements: list[tuple[int, int] | None] = [None] * len(tiles)
    cur_y = 0
    max_x = 0
    for group in groups:
        row_ids = [order[idx] for idx in group]
        row_h = max(int(tiles[i].shape[0]) for i in row_ids)
        cur_x = random.randint(0, 3)
        for j, idx in enumerate(row_ids):
            tile = tiles[idx]
            h, w = map(int, tile.shape)
            top = cur_y + random.randint(0, row_h - h)
            placements[idx] = (int(top), int(cur_x))
            cur_x += w
            if j != len(row_ids) - 1:
                cur_x += random.randint(1, 3)
        max_x = max(max_x, cur_x)
        cur_y += row_h
        if group is not groups[-1]:
            cur_y += random.randint(1, 3)

    height = cur_y + random.randint(0, 2)
    width = max_x + random.randint(0, 2)
    return [p for p in placements if p is not None], int(height), int(width)


def _build_legend(entries: list[tuple[int, tuple[tuple[int, int], ...]]]) -> np.ndarray:
    left_width = random.randint(1, 3)
    width = left_width + 3 + 2
    height = 4 * len(entries) + 1
    legend = np.full((height, width), BLACK, dtype=int)
    legend[0, :] = MAROON
    legend[-1, :] = MAROON
    legend[:, 0] = MAROON
    legend[:, -1] = MAROON
    for i, (color, directions) in enumerate(entries):
        sep = 4 * i
        legend[sep, :] = MAROON
        row0 = sep + 1
        legend[row0 + 1, 1 + random.randrange(left_width)] = int(color)
        legend[row0 : row0 + 3, -4:-1] = _directions_to_pattern(directions)
    legend[-1, :] = MAROON
    return legend


def _expected_output(grid: np.ndarray, specs: list[dict[str, object]], placements: list[tuple[int, int]]) -> np.ndarray:
    out = np.array(grid, copy=True)
    for spec, (r0, c0) in zip(specs, placements, strict=False):
        color = int(spec["color"])
        inner_h = int(spec["inner_h"])
        inner_w = int(spec["inner_w"])
        seed_r = int(spec["seed_r"])
        seed_c = int(spec["seed_c"])
        directions = spec["directions"]
        assert isinstance(directions, tuple)
        filled = _simulate_growth(inner_h, inner_w, seed_r, seed_c, directions)
        for rr, cc in filled:
            out[r0 + 1 + rr, c0 + 1 + cc] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    n_fences = random.randint(2, 4)
    colors = random.sample(SEED_COLORS, n_fences)
    patterns = random.sample(PATTERN_BANK, n_fences)
    specs: list[dict[str, object]] = []

    grew_big = False
    for color, directions in zip(colors, patterns, strict=False):
        inner_h = random.randint(3, 8)
        inner_w = random.randint(3, 8)
        seed = _choose_seed(inner_h, inner_w, directions)
        if seed is None:
            return None
        seed_r, seed_c = seed
        area = len(_simulate_growth(inner_h, inner_w, seed_r, seed_c, directions))
        grew_big |= area >= 3
        specs.append(
            {
                "color": int(color),
                "directions": directions,
                "inner_h": int(inner_h),
                "inner_w": int(inner_w),
                "seed_r": int(seed_r),
                "seed_c": int(seed_c),
            }
        )
    if not grew_big:
        return None

    tiles = [_build_fence(spec) for spec in specs]
    placements, pack_h, pack_w = _layout_fences(tiles)
    legend_entries = list(zip(colors, patterns, strict=False))
    random.shuffle(legend_entries)
    legend = _build_legend(legend_entries)
    side = random.choice(["top", "left"])

    if side == "top":
        gap = random.randint(1, 3)
        left_margin = random.randint(1, 4)
        right_margin = random.randint(1, 4)
        bottom_margin = random.randint(1, 4)
        pack_r0 = legend.shape[0] + gap
        pack_c0 = left_margin
        grid_h = pack_r0 + pack_h + bottom_margin
        grid_w = max(left_margin + pack_w + right_margin, int(legend.shape[1]) + 1)
        legend_r0 = 0
        legend_c0 = random.randint(0, grid_w - int(legend.shape[1]))
    else:
        gap = random.randint(1, 3)
        top_margin = random.randint(1, 4)
        right_margin = random.randint(1, 4)
        bottom_margin = random.randint(1, 4)
        pack_r0 = top_margin
        pack_c0 = legend.shape[1] + gap
        grid_h = max(top_margin + pack_h + bottom_margin, int(legend.shape[0]) + 1)
        grid_w = pack_c0 + pack_w + right_margin
        legend_r0 = random.randint(0, grid_h - int(legend.shape[0]))
        legend_c0 = 0

    if not (GRID_MIN <= grid_h <= GRID_MAX and GRID_MIN <= grid_w <= GRID_MAX):
        return None

    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    lr1 = legend_r0 + legend.shape[0]
    lc1 = legend_c0 + legend.shape[1]
    grid[legend_r0:lr1, legend_c0:lc1] = legend

    global_placements: list[tuple[int, int]] = []
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    occupied[legend_r0:lr1, legend_c0:lc1] = legend != BLACK
    for tile, (pr, pc) in zip(tiles, placements, strict=False):
        r0 = pack_r0 + pr
        c0 = pack_c0 + pc
        r1 = r0 + tile.shape[0]
        c1 = c0 + tile.shape[1]
        if r1 > grid_h or c1 > grid_w:
            return None
        mask = tile != BLACK
        if np.any(occupied[r0:r1, c0:c1] & mask):
            return None
        grid[r0:r1, c0:c1][mask] = tile[mask]
        occupied[r0:r1, c0:c1] |= mask
        global_placements.append((int(r0), int(c0)))

    if len(_find_fences(grid)) != n_fences:
        return None
    if set(_parse_legend(grid)) != set(colors):
        return None

    expected = _expected_output(grid, specs, global_placements)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
