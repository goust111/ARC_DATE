from __future__ import annotations

from collections import deque
import random

import numpy as np


COLORS = tuple(range(10))
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR_NAMES = ("up", "down", "left", "right")
DIR_BY_NAME = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

GEN_ATTEMPTS = 500
GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
OBJECT_COUNT_RANGE = (1, 4)
PATCH_H_RANGE = (5, 9)
PATCH_W_RANGE = (6, 10)
SPAN_RADIUS = 2
OBJECT_TRIES = 120
PLACEMENT_TRIES = 160


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _neighbors8(r: int, c: int, h: int, w: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                yield nr, nc


def _color_components(
    grid: np.ndarray, color: int, conn8: bool
) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    neighbor_fn = _neighbors8 if conn8 else _neighbors4

    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) != int(color):
                continue
            q = deque([(r, c)])
            visited[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in neighbor_fn(rr, cc, h, w):
                    if visited[nr, nc] or int(grid[nr, nc]) != int(color):
                        continue
                    visited[nr, nc] = True
                    q.append((nr, nc))
            comps.append(comp)
    return comps


def _singleton_seed_pixels(grid: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    seeds: list[tuple[int, int, int]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != int(bg)):
        for comp in _color_components(grid, color, conn8=True):
            if len(comp) == 1:
                r, c = comp[0]
                seeds.append((r, c, color))
    return seeds


def _ray_cells(
    start: tuple[int, int], step: tuple[int, int], h: int, w: int
) -> list[tuple[int, int]]:
    r = start[0] + step[0]
    c = start[1] + step[1]
    cells: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w:
        cells.append((r, c))
        r += step[0]
        c += step[1]
    return cells


def _flood_seed_region(
    grid: np.ndarray,
    seed: tuple[int, int],
    blocked: tuple[int, int],
    bg: int,
) -> tuple[bool, int]:
    h, w = grid.shape
    q = deque([seed])
    seen = {seed}
    touches_border = seed[0] in (0, h - 1) or seed[1] in (0, w - 1)
    bg_count = 0

    while q:
        r, c = q.popleft()
        if int(grid[r, c]) == int(bg):
            bg_count += 1
        for nr, nc in _neighbors4(r, c, h, w):
            if (nr, nc) == blocked or (nr, nc) in seen:
                continue
            if (nr, nc) != seed and int(grid[nr, nc]) != int(bg):
                continue
            seen.add((nr, nc))
            if nr in (0, h - 1) or nc in (0, w - 1):
                touches_border = True
            q.append((nr, nc))
    return touches_border, bg_count


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find each singleton seed, detect the unique open direction through the outline
    gap, count the enclosed blank area behind that gap, then extend a seed-colored
    ray outward from the gap by that many cells.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)
    out = g.copy()

    for sr, sc, seed_color in _singleton_seed_pixels(g, bg):
        open_rays: list[tuple[tuple[int, int], list[tuple[int, int]]]] = []
        for step in DIRS4:
            ray = _ray_cells((sr, sc), step, h, w)
            if ray and all(int(g[r, c]) == int(bg) for r, c in ray):
                open_rays.append((step, ray))

        assert len(open_rays) == 1, "expected exactly one open gap direction per seed"
        step, ray = open_rays[0]

        gap: tuple[int, int] | None = None
        length = -1
        for cell in ray:
            touches_border, bg_count = _flood_seed_region(g, (sr, sc), cell, bg)
            if not touches_border:
                gap = cell
                length = bg_count

        assert gap is not None, "failed to localize the contour gap"

        r, c = gap
        for _ in range(length):
            if not (0 <= r < h and 0 <= c < w):
                break
            if int(out[r, c]) != int(bg):
                break
            out[r, c] = int(seed_color)
            r += step[0]
            c += step[1]

    return out


def _build_horizontal_inside(
    rng: random.Random, patch_h: int, patch_w: int, seed_row: int
) -> np.ndarray:
    inside = np.zeros((patch_h, patch_w), dtype=bool)
    top_lo = max(1, seed_row - SPAN_RADIUS)
    bot_hi = min(patch_h - 2, seed_row + SPAN_RADIUS)
    for c in range(1, patch_w - 1):
        top = rng.randint(top_lo, seed_row)
        bottom = rng.randint(seed_row, bot_hi)
        inside[top : bottom + 1, c] = True
    return inside


def _build_vertical_inside(
    rng: random.Random, patch_h: int, patch_w: int, seed_col: int
) -> np.ndarray:
    inside = np.zeros((patch_h, patch_w), dtype=bool)
    left_lo = max(1, seed_col - SPAN_RADIUS)
    right_hi = min(patch_w - 2, seed_col + SPAN_RADIUS)
    for r in range(1, patch_h - 1):
        left = rng.randint(left_lo, seed_col)
        right = rng.randint(seed_col, right_hi)
        inside[r, left : right + 1] = True
    return inside


def _build_patch(
    rng: random.Random,
    bg: int,
    outline_color: int,
    seed_color: int,
    direction_name: str,
) -> tuple[np.ndarray, dict[str, object]] | None:
    patch_h = rng.randint(*PATCH_H_RANGE)
    patch_w = rng.randint(*PATCH_W_RANGE)
    direction = DIR_BY_NAME[direction_name]

    if direction_name in ("left", "right"):
        seed_row = rng.randint(1, patch_h - 2)
        if direction_name == "right":
            seed_col = rng.randint(1, patch_w - 3)
            gap = (seed_row, patch_w - 1)
        else:
            seed_col = rng.randint(2, patch_w - 2)
            gap = (seed_row, 0)
        inside = _build_horizontal_inside(rng, patch_h, patch_w, seed_row)
    else:
        seed_col = rng.randint(1, patch_w - 2)
        if direction_name == "down":
            seed_row = rng.randint(1, patch_h - 3)
            gap = (patch_h - 1, seed_col)
        else:
            seed_row = rng.randint(2, patch_h - 2)
            gap = (0, seed_col)
        inside = _build_vertical_inside(rng, patch_h, patch_w, seed_col)

    if not bool(inside[seed_row, seed_col]):
        return None

    inside_area = int(inside.sum())
    if inside_area < 4:
        return None

    boundary: set[tuple[int, int]] = set()
    for r, c in np.argwhere(inside):
        rr = int(r)
        cc = int(c)
        for nr, nc in _neighbors4(rr, cc, patch_h, patch_w):
            if not bool(inside[nr, nc]):
                boundary.add((nr, nc))

    if gap not in boundary:
        return None
    boundary.remove(gap)

    patch = np.full((patch_h, patch_w), int(bg), dtype=int)
    for r, c in boundary:
        patch[r, c] = int(outline_color)
    patch[seed_row, seed_col] = int(seed_color)

    meta: dict[str, object] = {
        "patch": patch,
        "gap_local": gap,
        "direction": direction,
        "direction_name": direction_name,
        "seed_color": int(seed_color),
        "length": inside_area - 1,
    }
    return patch, meta


def _box_with_margin(
    top: int, left: int, height: int, width: int, grid_h: int, grid_w: int
) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    r0 = max(0, top - 1)
    r1 = min(grid_h - 1, top + height)
    c0 = max(0, left - 1)
    c1 = min(grid_w - 1, left + width)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            cells.add((r, c))
    return cells


def _sample_position(
    rng: random.Random,
    grid_h: int,
    grid_w: int,
    patch_h: int,
    patch_w: int,
    direction_name: str,
    want_truncation: bool,
) -> tuple[int, int] | None:
    if direction_name == "right":
        row_low, row_high = 0, grid_h - patch_h
        col_low, col_high = 0, grid_w - patch_w - 1
        if col_low > col_high or row_low > row_high:
            return None
        split = (col_low + col_high) // 2
        col = rng.randint(split if want_truncation else col_low, col_high if want_truncation else split)
        row = rng.randint(row_low, row_high)
        return row, col
    if direction_name == "left":
        row_low, row_high = 0, grid_h - patch_h
        col_low, col_high = 1, grid_w - patch_w
        if col_low > col_high or row_low > row_high:
            return None
        split = (col_low + col_high) // 2
        col = rng.randint(col_low if want_truncation else split, split if want_truncation else col_high)
        row = rng.randint(row_low, row_high)
        return row, col
    if direction_name == "down":
        row_low, row_high = 0, grid_h - patch_h - 1
        col_low, col_high = 0, grid_w - patch_w
        if row_low > row_high or col_low > col_high:
            return None
        split = (row_low + row_high) // 2
        row = rng.randint(split if want_truncation else row_low, row_high if want_truncation else split)
        col = rng.randint(col_low, col_high)
        return row, col

    row_low, row_high = 1, grid_h - patch_h
    col_low, col_high = 0, grid_w - patch_w
    if row_low > row_high or col_low > col_high:
        return None
    split = (row_low + row_high) // 2
    row = rng.randint(row_low if want_truncation else split, split if want_truncation else row_high)
    col = rng.randint(col_low, col_high)
    return row, col


def _ray_from_gap(
    gap: tuple[int, int], step: tuple[int, int], grid_h: int, grid_w: int
) -> list[tuple[int, int]]:
    r, c = gap
    cells: list[tuple[int, int]] = []
    while 0 <= r < grid_h and 0 <= c < grid_w:
        cells.append((r, c))
        r += step[0]
        c += step[1]
    return cells


def _expected_output(grid: np.ndarray, specs: list[dict[str, object]]) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        r, c = spec["gap_abs"]
        dr, dc = spec["direction"]
        for _ in range(int(spec["length"])):
            if not (0 <= r < out.shape[0] and 0 <= c < out.shape[1]):
                break
            if int(out[r, c]) != int(spec["bg"]):
                break
            out[r, c] = int(spec["seed_color"])
            r += dr
            c += dc
    return out


def _try_build_instance(rng: random.Random) -> np.ndarray | None:
    grid_h = rng.randint(*GRID_H_RANGE)
    grid_w = rng.randint(*GRID_W_RANGE)
    bg = rng.choice(COLORS)
    grid = np.full((grid_h, grid_w), int(bg), dtype=int)

    max_objects = min(OBJECT_COUNT_RANGE[1], (len(COLORS) - 1) // 2)
    n_objects = rng.randint(OBJECT_COUNT_RANGE[0], max_objects)

    palette = [c for c in COLORS if c != bg]
    rng.shuffle(palette)
    pairs = [(palette[2 * i], palette[2 * i + 1]) for i in range(n_objects)]

    reserved_patch: set[tuple[int, int]] = set()
    reserved_ray: set[tuple[int, int]] = set()
    specs: list[dict[str, object]] = []

    for outline_color, seed_color in pairs:
        want_truncation = bool(rng.getrandbits(1))
        placed = False
        for _ in range(OBJECT_TRIES):
            direction_name = rng.choice(DIR_NAMES)
            built = _build_patch(rng, bg, outline_color, seed_color, direction_name)
            if built is None:
                continue
            patch, meta = built
            patch_h, patch_w = patch.shape

            for _ in range(PLACEMENT_TRIES):
                pos = _sample_position(
                    rng,
                    grid_h,
                    grid_w,
                    patch_h,
                    patch_w,
                    direction_name,
                    want_truncation,
                )
                if pos is None:
                    break
                top, left = pos
                box = _box_with_margin(top, left, patch_h, patch_w, grid_h, grid_w)
                if box & reserved_patch or box & reserved_ray:
                    continue

                gap_local = meta["gap_local"]
                gap_abs = (top + int(gap_local[0]), left + int(gap_local[1]))
                ray_cells = _ray_from_gap(gap_abs, meta["direction"], grid_h, grid_w)
                if ray_cells and (set(ray_cells) & reserved_patch or set(ray_cells) & reserved_ray):
                    continue

                truncated = int(meta["length"]) > len(ray_cells)
                if truncated != want_truncation:
                    continue

                patch_view = grid[top : top + patch_h, left : left + patch_w]
                if np.any(patch_view != int(bg)):
                    continue

                grid[top : top + patch_h, left : left + patch_w] = patch
                spec = {
                    "gap_abs": gap_abs,
                    "direction": meta["direction"],
                    "length": int(meta["length"]),
                    "seed_color": int(seed_color),
                    "bg": int(bg),
                }
                specs.append(spec)
                reserved_patch.update(box)
                reserved_ray.update(ray_cells)
                placed = True
                break
            if placed:
                break
        if not placed:
            return None

    expected = _expected_output(grid, specs)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _try_build_instance(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
