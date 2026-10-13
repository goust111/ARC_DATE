import random
from collections import Counter, deque

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_GENERATION_ATTEMPTS = 400
ALL_COLORS = tuple(range(10))


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
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
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _reachable(passable: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    h, w = passable.shape
    reach = np.zeros((h, w), dtype=bool)
    q = deque()
    seed_pts = np.argwhere(passable & seeds)
    for r, c in seed_pts:
        rr, cc = int(r), int(c)
        reach[rr, cc] = True
        q.append((rr, cc))
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and not reach[nr, nc]:
                reach[nr, nc] = True
                q.append((nr, nc))
    return reach


def _parse_legend_strip(primary: np.ndarray, secondary: np.ndarray, bg: int):
    n = len(primary)
    groups = []
    i = 0
    while i < n:
        if int(primary[i]) == bg and int(secondary[i]) == bg:
            i += 1
            continue
        if int(primary[i]) == bg or int(secondary[i]) == bg:
            return None
        j = i
        while j < n and int(primary[j]) != bg and int(secondary[j]) != bg:
            j += 1
        pseg = primary[i:j]
        sseg = secondary[i:j]
        if len(set(map(int, pseg))) != 1 or len(set(map(int, sseg))) != 1:
            return None
        groups.append((i, j, j - i, int(pseg[0]), int(sseg[0])))
        i = j
    if len(groups) < 2:
        return None
    lengths = [length for _start, _end, length, _pc, _sc in groups]
    if len(set(lengths)) != len(lengths):
        return None
    max_len = max(lengths)
    if set(lengths) != set(range(1, max_len + 1)):
        return None
    mapping = {length: fill for _start, _end, length, _pc, fill in groups}
    return groups, mapping


def _find_legend(grid: np.ndarray, bg: int):
    h, w = grid.shape
    candidates = []

    def add_horizontal(r0: int, r1: int):
        parsed = _parse_legend_strip(grid[r0], grid[r1], bg)
        if parsed is None:
            return
        groups, mapping = parsed
        mask = np.zeros((h, w), dtype=bool)
        for start, end, _length, _pc, _sc in groups:
            mask[r0, start:end] = True
            mask[r1, start:end] = True
        candidates.append(
            {
                "mapping": mapping,
                "mask": mask,
                "score": (len(mapping), int(mask.sum())),
            }
        )

    def add_vertical(c0: int, c1: int):
        parsed = _parse_legend_strip(grid[:, c0], grid[:, c1], bg)
        if parsed is None:
            return
        groups, mapping = parsed
        mask = np.zeros((h, w), dtype=bool)
        for start, end, _length, _pc, _sc in groups:
            mask[start:end, c0] = True
            mask[start:end, c1] = True
        candidates.append(
            {
                "mapping": mapping,
                "mask": mask,
                "score": (len(mapping), int(mask.sum())),
            }
        )

    if h >= 2:
        add_horizontal(0, 1)
        add_horizontal(h - 2, h - 1)
    if w >= 2:
        add_vertical(0, 1)
        add_vertical(w - 2, w - 1)

    if not candidates:
        raise AssertionError("legend not found")
    best = max(candidates, key=lambda item: item["score"])
    return best["mapping"], best["mask"]


def _outside_seeds(passable: np.ndarray, legend_mask: np.ndarray) -> np.ndarray:
    h, w = passable.shape
    seeds = np.zeros((h, w), dtype=bool)
    seeds[0, :] |= passable[0, :]
    seeds[h - 1, :] |= passable[h - 1, :]
    seeds[:, 0] |= passable[:, 0]
    seeds[:, w - 1] |= passable[:, w - 1]
    for r, c in np.argwhere(legend_mask):
        rr, cc = int(r), int(c)
        for dr, dc in DIR4:
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and passable[nr, nc]:
                seeds[nr, nc] = True
    return seeds


def _regions_for_wall_color(
    grid: np.ndarray, bg: int, legend_mask: np.ndarray, wall_color: int, max_k: int
) -> list[dict]:
    h, w = grid.shape
    passable = (~legend_mask) & (grid != wall_color)
    outside = _reachable(passable, _outside_seeds(passable, legend_mask))
    enclosed = passable & (~outside)
    regions = []
    for cells in _components(enclosed):
        colors = {
            int(grid[r, c])
            for r, c in cells
            if int(grid[r, c]) != bg and int(grid[r, c]) != wall_color
        }
        if not (1 <= len(colors) <= max_k):
            continue
        wall_cells = set()
        for r, c in cells:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == wall_color:
                    wall_cells.add((nr, nc))
        if not wall_cells:
            continue
        regions.append(
            {
                "cells": cells,
                "color_count": len(colors),
                "wall_cells": sorted(wall_cells),
            }
        )
    return regions


def _find_wall_color(grid: np.ndarray, bg: int, legend_mask: np.ndarray, max_k: int):
    colors = {int(v) for v in np.unique(grid) if int(v) != bg}
    best = None
    for color in colors:
        regions = _regions_for_wall_color(grid, bg, legend_mask, color, max_k)
        score = (
            len(regions),
            sum(len(region["wall_cells"]) for region in regions),
            int(np.count_nonzero(grid == color)),
        )
        if best is None or score > best[0]:
            best = (score, color, regions)
    if best is None or best[0][0] == 0:
        raise AssertionError("wall color not found")
    return best[1], best[2]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select the border color from the edge legend using the count of distinct colors
    inside each closed region, then paint one outer halo layer around the grey border.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    mapping, legend_mask = _find_legend(g, bg)
    wall_color, regions = _find_wall_color(g, bg, legend_mask, max(mapping))

    enclosed_mask = np.zeros_like(g, dtype=bool)
    for region in regions:
        for r, c in region["cells"]:
            enclosed_mask[r, c] = True

    out = g.copy()
    regions.sort(key=lambda region: min(region["cells"]))
    for region in regions:
        fill = int(mapping[region["color_count"]])
        for wr, wc in region["wall_cells"]:
            for dr, dc in DIR4:
                nr, nc = wr + dr, wc + dc
                if not (0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]):
                    continue
                if legend_mask[nr, nc] or enclosed_mask[nr, nc]:
                    continue
                if int(out[nr, nc]) == bg:
                    out[nr, nc] = fill
    return out


def _draw_polyline(shape: tuple[int, int], points: list[tuple[int, int]]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    for (r0, c0), (r1, c1) in zip(points, points[1:]):
        if r0 == r1:
            step = 1 if c1 >= c0 else -1
            for c in range(c0, c1 + step, step):
                mask[r0, c] = True
        elif c0 == c1:
            step = 1 if r1 >= r0 else -1
            for r in range(r0, r1 + step, step):
                mask[r, c0] = True
        else:
            raise ValueError("polyline must be axis-aligned")
    return mask


def _make_loop_patch(rng: random.Random):
    family = rng.choice(("rect", "bevel", "notch"))
    h = rng.randint(5, 9)
    w = rng.randint(5, 11)

    if family == "rect":
        wall = np.zeros((h, w), dtype=bool)
        wall[0, :] = True
        wall[h - 1, :] = True
        wall[:, 0] = True
        wall[:, w - 1] = True
    elif family == "bevel":
        points = [
            (0, 1),
            (0, w - 2),
            (1, w - 2),
            (1, w - 1),
            (h - 2, w - 1),
            (h - 2, w - 2),
            (h - 1, w - 2),
            (h - 1, 1),
            (h - 2, 1),
            (h - 2, 0),
            (1, 0),
            (1, 1),
            (0, 1),
        ]
        wall = _draw_polyline((h, w), points)
    else:
        side = rng.choice(("left", "right"))
        notch_r = rng.randint(1, h - 3)
        if side == "right":
            points = [
                (0, 0),
                (0, w - 1),
                (notch_r, w - 1),
                (notch_r, w - 2),
                (notch_r + 1, w - 2),
                (notch_r + 1, w - 1),
                (h - 1, w - 1),
                (h - 1, 0),
                (0, 0),
            ]
        else:
            points = [
                (0, w - 1),
                (0, 0),
                (notch_r, 0),
                (notch_r, 1),
                (notch_r + 1, 1),
                (notch_r + 1, 0),
                (h - 1, 0),
                (h - 1, w - 1),
                (0, w - 1),
            ]
        wall = _draw_polyline((h, w), points)

    passable = ~wall
    seeds = np.zeros((h, w), dtype=bool)
    seeds[0, :] = passable[0, :]
    seeds[h - 1, :] = passable[h - 1, :]
    seeds[:, 0] |= passable[:, 0]
    seeds[:, w - 1] |= passable[:, w - 1]
    outside = _reachable(passable, seeds)
    interior = passable & (~outside)
    cells = [tuple(map(int, pt)) for pt in np.argwhere(interior)]
    if len(cells) < 1:
        return None
    return wall, cells


def _fill_patch(
    wall: np.ndarray,
    interior_cells: list[tuple[int, int]],
    grey: int,
    bg: int,
    fill_palette: list[int],
    k: int,
    rng: random.Random,
) -> np.ndarray:
    patch = np.full(wall.shape, bg, dtype=int)
    patch[wall] = grey
    colors = rng.sample(fill_palette, k)
    cells = interior_cells[:]
    rng.shuffle(cells)
    for idx, color in enumerate(colors):
        r, c = cells[idx]
        patch[r, c] = color
    extras = cells[k:]
    rng.shuffle(extras)
    extra_count = rng.randint(0, min(len(extras), k + 2))
    for r, c in extras[:extra_count]:
        patch[r, c] = rng.choice(colors)
    return patch


def _build_legend(
    width: int, bg: int, fill_palette: list[int], rng: random.Random
) -> tuple[np.ndarray, dict[int, int], np.ndarray]:
    lengths = [1, 2, 3, 4, 5]
    order = lengths[:]
    rng.shuffle(order)
    total = sum(order) + len(order) - 1
    start = rng.randint(0, width - total)
    legend = np.full((2, width), bg, dtype=int)
    mapping = {}
    count_palette = fill_palette[:]
    pos = start
    for idx, length in enumerate(order):
        fill = fill_palette[length - 1]
        top_color_choices = [c for c in count_palette if c != fill]
        top_color = rng.choice(top_color_choices or count_palette)
        legend[0, pos : pos + length] = top_color
        legend[1, pos : pos + length] = fill
        mapping[length] = fill
        pos += length
        if idx != len(order) - 1:
            pos += 1
    mask = legend != bg
    return legend, mapping, mask


def _try_place_patch(
    grid: np.ndarray, blocked: np.ndarray, patch: np.ndarray, rng: random.Random
) -> tuple[int, int] | None:
    h, w = grid.shape
    ph, pw = patch.shape
    row_choices = list(range(1, h - ph - 1))
    col_choices = list(range(1, w - pw - 1))
    rng.shuffle(row_choices)
    rng.shuffle(col_choices)
    for r0 in row_choices:
        for c0 in col_choices:
            if blocked[r0 - 1 : r0 + ph + 1, c0 - 1 : c0 + pw + 1].any():
                continue
            return r0, c0
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(16, 24)
    width = rng.randint(24, 32)
    bg = rng.choice(ALL_COLORS)
    grey = rng.choice([c for c in ALL_COLORS if c != bg])
    fill_palette = rng.sample([c for c in ALL_COLORS if c not in (bg, grey)], 5)

    grid = np.full((height, width), bg, dtype=int)
    blocked = np.zeros((height, width), dtype=bool)
    legend, mapping, legend_local_mask = _build_legend(width, bg, fill_palette, rng)
    legend_on_top = rng.choice((True, False))
    if legend_on_top:
        grid[0:2, :] = legend
        blocked[0:3, :] = True
        legend_mask = np.zeros_like(grid, dtype=bool)
        legend_mask[0:2, :] = legend_local_mask
    else:
        grid[height - 2 : height, :] = legend
        blocked[height - 3 : height, :] = True
        legend_mask = np.zeros_like(grid, dtype=bool)
        legend_mask[height - 2 : height, :] = legend_local_mask

    blocked[:, 0] = True
    blocked[:, width - 1] = True
    region_count = rng.randint(2, 5)
    ks = rng.sample([1, 2, 3, 4, 5], region_count)
    rng.shuffle(ks)

    for k in ks:
        placed = False
        for _ in range(80):
            made = _make_loop_patch(rng)
            if made is None:
                continue
            wall, interior_cells = made
            if len(interior_cells) < k:
                continue
            patch = _fill_patch(wall, interior_cells, grey, bg, fill_palette, k, rng)
            pos = _try_place_patch(grid, blocked, patch, rng)
            if pos is None:
                continue
            r0, c0 = pos
            ph, pw = patch.shape
            grid[r0 : r0 + ph, c0 : c0 + pw] = patch
            blocked[r0 - 1 : r0 + ph + 1, c0 - 1 : c0 + pw + 1] = True
            placed = True
            break
        if not placed:
            return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None

    halo_colors = {int(v) for v in np.unique(out[(grid == bg) & (out != bg)])}
    if not halo_colors or not halo_colors.issubset(set(mapping.values())):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
