from __future__ import annotations

import random
from collections import Counter

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _is_rect_frame(points: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(points)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    local = np.zeros((h, w), dtype=bool)
    for r, c in points:
        local[r - r0, c - c0] = True
    perimeter = np.zeros((h, w), dtype=bool)
    perimeter[0, :] = True
    perimeter[-1, :] = True
    perimeter[:, 0] = True
    perimeter[:, -1] = True
    return bool(np.array_equal(local, perimeter))


def _find_regions(grid: np.ndarray, bg: int) -> list[tuple[int, int, int, int, int]]:
    regions: list[tuple[int, int, int, int, int]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != int(bg)):
        for comp in _components(grid == int(color)):
            if not _is_rect_frame(comp):
                continue
            r0, c0, r1, c1 = _bbox(comp)
            regions.append((int(r0), int(c0), int(r1), int(c1), int(color)))
    regions.sort()
    return regions


def _infer_seed_side(grid: np.ndarray, regions: list[tuple[int, int, int, int, int]], bg: int) -> int:
    estimates: list[int] = []
    for r0, c0, r1, c1, _color in regions:
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        mark_r = interior.shape[0] - 1
        mark_c = interior.shape[1] - 1
        mask = interior != int(bg)
        mask[mark_r, mark_c] = False
        pts = np.argwhere(mask)
        if pts.size == 0:
            continue
        side = int(max(int(pts[:, 0].max()), int(pts[:, 1].max())) + 1)
        estimates.append(side)
    assert estimates, "at least one seed pattern is required"
    return int(max(estimates))


def _tile_seed(seed: np.ndarray, interior_h: int, interior_w: int, bg: int, normal: bool) -> np.ndarray:
    side = int(seed.shape[0])
    reps_r = interior_h // side
    reps_c = interior_w // side
    tiled_h = reps_r * side
    tiled_w = reps_c * side
    start_r = 0 if normal else interior_h - tiled_h
    start_c = 0 if normal else interior_w - tiled_w
    pattern = seed if normal else np.rot90(seed, 2)

    out = np.full((interior_h, interior_w), int(bg), dtype=int)
    for br in range(reps_r):
        for bc in range(reps_c):
            rr = start_r + br * side
            cc = start_c + bc * side
            out[rr : rr + side, cc : cc + side] = pattern
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the seed square inside each framed region and tile it across the region interior.

    Input structure:
    - The most common color is the background.
    - One or more single-color rectangular frames partition the grid into closed regions.
    - Inside each frame, the top-left corner of the interior contains a small square seed pattern, and the bottom-right interior cell is a one-pixel color marker.

    Rule:
    - Detect every rectangular frame and measure its interior height and width, excluding the border.
    - Read the common seed-square side length from the occupied extent of the top-left pattern, then extract that square from each region.
    - If the bottom-right marker color appears inside that seed, tile the seed from the top-left using floor(H / S) by floor(W / S) copies.
    - Otherwise rotate the seed by 180 degrees and tile it from the bottom-right, leaving any leftover rows or columns as background.
    - Replace the whole region interior with the tiled result, which removes the original seed square and the marker.

    Key insight:
    - The marker does not name a direction directly; it only says whether the seed's own colors are reused, which selects between normal top-left tiling and a 180-degree bottom-right tiling.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    regions = _find_regions(g, bg)
    assert regions, "no framed regions found"

    side = _infer_seed_side(g, regions, bg)
    out = g.copy()

    for r0, c0, r1, c1, _border_color in regions:
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        ih, iw = interior.shape
        assert side <= ih and side <= iw, "seed must fit inside each region interior"

        seed = interior[:side, :side].copy()
        marker = int(interior[ih - 1, iw - 1])
        seed_colors = {int(v) for v in np.unique(seed) if int(v) != int(bg)}
        normal = marker in seed_colors

        out[r0 + 1 : r1, c0 + 1 : c1] = _tile_seed(seed, ih, iw, bg, normal)

    return out


SEED_MASKS: dict[int, list[np.ndarray]] = {
    2: [
        np.array([[1, 1], [1, 0]], dtype=int),
        np.array([[1, 0], [1, 1]], dtype=int),
        np.array([[1, 1], [0, 1]], dtype=int),
    ],
    3: [
        np.array([[1, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=int),
        np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=int),
        np.array([[1, 0, 1], [1, 1, 0], [0, 1, 1]], dtype=int),
    ],
}


def _rand_choice(rng: random.Random, values: list[int]) -> int:
    return int(values[rng.randrange(len(values))])


def _paint_seed(mask: np.ndarray, bg: int, colors: list[int], rng: random.Random) -> np.ndarray:
    seed = np.full(mask.shape, int(bg), dtype=int)
    occ = [(int(r), int(c)) for r, c in np.argwhere(mask > 0)]
    if len(colors) == 1:
        for r, c in occ:
            seed[r, c] = int(colors[0])
        return seed

    rng.shuffle(occ)
    split = rng.randint(1, len(occ) - 1)
    group_a = set(occ[:split])
    used: set[int] = set()
    for r, c in occ:
        color = int(colors[0] if (r, c) in group_a else colors[1])
        seed[r, c] = color
        used.add(color)
    if len(used) < 2:
        rr, cc = occ[-1]
        seed[rr, cc] = int(colors[1])
    return seed


def _can_place(boxes: list[tuple[int, int, int, int]], top: int, left: int, height: int, width: int) -> bool:
    bottom = top + height - 1
    right = left + width - 1
    for r0, c0, r1, c1 in boxes:
        if not (bottom + 1 < r0 or r1 + 1 < top or right + 1 < c0 or c1 + 1 < left):
            return False
    return True


def _branch_flags(rng: random.Random, count: int) -> list[bool]:
    if count == 1:
        return [bool(rng.randrange(2))]
    normal_count = count // 2
    flags = [True] * normal_count + [False] * (count - normal_count)
    rng.shuffle(flags)
    if all(flags):
        flags[-1] = False
    if not any(flags):
        flags[0] = True
    return flags


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    bg = rng.randrange(10)
    colors = [c for c in range(10) if c != bg]
    border = colors.pop(rng.randrange(len(colors)))

    side = _rand_choice(rng, [2, 3])
    seed_mask = SEED_MASKS[side][rng.randrange(len(SEED_MASKS[side]))]

    max_regions = 4 if side == 2 else 3
    region_count = rng.randint(2, max_regions)
    flags = _branch_flags(rng, region_count)

    boxes: list[tuple[int, int, int, int]] = []
    specs: list[dict[str, object]] = []
    for normal in flags:
        inner_h = 2 * side + rng.randint(0, 2)
        inner_w = 2 * side + rng.randint(0, 2)
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        specs.append(
            {
                "normal": bool(normal),
                "inner_h": int(inner_h),
                "inner_w": int(inner_w),
                "outer_h": int(outer_h),
                "outer_w": int(outer_w),
            }
        )

    min_h = max(max(int(spec["outer_h"]) for spec in specs) + 2, 16)
    min_w = max(max(int(spec["outer_w"]) for spec in specs) + 2, 16)
    max_h = 30
    max_w = 30
    grid_h = rng.randint(min_h, max_h)
    grid_w = rng.randint(min_w, max_w)
    grid = np.full((grid_h, grid_w), int(bg), dtype=int)

    rng.shuffle(specs)
    for spec in specs:
        placed = False
        outer_h = int(spec["outer_h"])
        outer_w = int(spec["outer_w"])
        for _ in range(250):
            top = rng.randint(0, grid_h - outer_h)
            left = rng.randint(0, grid_w - outer_w)
            if not _can_place(boxes, top, left, outer_h, outer_w):
                continue
            boxes.append((top, left, top + outer_h - 1, left + outer_w - 1))
            spec["top"] = int(top)
            spec["left"] = int(left)
            placed = True
            break
        if not placed:
            return None

    for spec in specs:
        top = int(spec["top"])
        left = int(spec["left"])
        inner_h = int(spec["inner_h"])
        inner_w = int(spec["inner_w"])
        bottom = top + inner_h + 1
        right = left + inner_w + 1

        grid[top, left : right + 1] = int(border)
        grid[bottom, left : right + 1] = int(border)
        grid[top : bottom + 1, left] = int(border)
        grid[top : bottom + 1, right] = int(border)

        palette = [c for c in range(10) if c not in {bg, border}]
        rng.shuffle(palette)
        if side == 2 or rng.random() < 0.5:
            used_colors = [int(palette[0])]
        else:
            used_colors = [int(palette[0]), int(palette[1])]

        seed = _paint_seed(seed_mask, bg, used_colors, rng)
        grid[top + 1 : top + 1 + side, left + 1 : left + 1 + side] = seed

        if bool(spec["normal"]):
            marker = int(used_colors[rng.randrange(len(used_colors))])
        else:
            marker_choices = [c for c in palette if c not in set(used_colors)]
            if not marker_choices:
                return None
            marker = int(marker_choices[0])
        grid[bottom - 1, right - 1] = int(marker)

    counts = Counter(int(v) for v in grid.ravel())
    if counts[int(bg)] != max(counts.values()):
        return None

    try:
        solved = main(grid)
    except Exception:
        return None

    if solved.shape != grid.shape:
        return None
    if bool(np.array_equal(solved, grid)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        rng = random.Random()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
