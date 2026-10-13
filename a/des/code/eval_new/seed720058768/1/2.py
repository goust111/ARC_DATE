from __future__ import annotations

import itertools
import random
from collections import deque

import numpy as np


BG = 0
WALL = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIR_NAMES = ("up", "down", "left", "right")
DIR_VECS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

GRID_SIZE_BY_COUNT = {
    1: ((16, 22), (16, 22)),
    2: ((20, 27), (20, 27)),
    3: ((24, 30), (24, 30)),
}
OBJECT_COUNT_CHOICES = (1, 2, 2, 3)
BASE_H_RANGE = (6, 9)
BASE_W_RANGE = (6, 9)
MAX_BUMPS = 2
BUMP_DEPTH_RANGE = (1, 2)
BUMP_SPAN_MIN = 2
COLOR_COUNT_CHOICES = (1, 3)
MAX_BRANCH_LEN = 5
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _outside_mask(blocked: np.ndarray) -> np.ndarray:
    h, w = blocked.shape
    outside = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if not blocked[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not blocked[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not blocked[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))
    return outside


def _direction_from_gap(closed_wall: list[tuple[int, int]], gap: tuple[int, int]) -> str:
    mean_r = sum(r for r, _c in closed_wall) / len(closed_wall)
    mean_c = sum(c for _r, c in closed_wall) / len(closed_wall)
    dr = gap[0] - mean_r
    dc = gap[1] - mean_c
    if abs(dr) >= abs(dc):
        return "down" if dr > 0 else "up"
    return "right" if dc > 0 else "left"


def _analyze_wall(grid: np.ndarray, comp: list[tuple[int, int]]) -> dict | None:
    r0, c0, r1, c1 = _bbox(comp)
    pr0 = max(0, r0 - 1)
    pc0 = max(0, c0 - 1)
    pr1 = min(grid.shape[0] - 1, r1 + 1)
    pc1 = min(grid.shape[1] - 1, c1 + 1)
    patch = grid[pr0 : pr1 + 1, pc0 : pc1 + 1]
    wall_mask = patch == WALL
    if len(_components(wall_mask)) != 1:
        return None

    candidates: list[dict] = []
    ph, pw = patch.shape
    for lr in range(1, ph - 1):
        for lc in range(1, pw - 1):
            if patch[lr, lc] != BG:
                continue
            wall_neighbors = sum(int(wall_mask[nr, nc]) for nr, nc in _neighbors4(lr, lc))
            if wall_neighbors < 2:
                continue

            closed = wall_mask.copy()
            closed[lr, lc] = True
            if len(_components(closed)) != 1:
                continue

            outside = _outside_mask(closed)
            interior = (~closed) & (~outside)
            if not interior.any():
                continue

            interior_comps = _components(interior)
            if len(interior_comps) != 1:
                continue
            interior_coords = interior_comps[0]

            seed_pixels = [
                (rr, cc)
                for rr, cc in interior_coords
                if int(patch[rr, cc]) not in (BG, WALL)
            ]
            if not seed_pixels:
                continue

            inside_adj = any(
                0 <= nr < ph and 0 <= nc < pw and interior[nr, nc]
                for nr, nc in _neighbors4(lr, lc)
            )
            outside_adj = any(
                0 <= nr < ph and 0 <= nc < pw and outside[nr, nc]
                for nr, nc in _neighbors4(lr, lc)
            )
            if not (inside_adj and outside_adj):
                continue

            candidates.append(
                {
                    "gap_local": (lr, lc),
                    "interior_mask": interior,
                    "interior_area": int(interior.sum()),
                    "seed_count": len(seed_pixels),
                }
            )

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item["interior_area"], item["seed_count"]), reverse=True)
    best = candidates[0]
    gap = (pr0 + best["gap_local"][0], pc0 + best["gap_local"][1])
    direction = _direction_from_gap(comp + [gap], gap)
    vertical = direction in ("up", "down")

    groups = []
    interior_mask = best["interior_mask"]
    for color in np.unique(patch[interior_mask]):
        color = int(color)
        if color in (BG, WALL):
            continue
        coords = np.argwhere(interior_mask & (patch == color))
        if vertical:
            axis_key = float(coords[:, 1].mean())
            tie_key = float(coords[:, 0].mean())
        else:
            axis_key = float(coords[:, 0].mean())
            tie_key = float(coords[:, 1].mean())
        groups.append((axis_key, tie_key, color, int(coords.shape[0])))

    groups.sort()
    return {
        "gap": gap,
        "direction": direction,
        "groups": [(color, count) for _a, _b, color, count in groups],
    }


def _wall_infos(grid: np.ndarray) -> list[dict]:
    infos: list[dict] = []
    for comp in _components(grid == WALL):
        info = _analyze_wall(grid, comp)
        if info is not None:
            infos.append(info)
    return infos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow outward color branches from each broken gray enclosure, with branch lengths set by the enclosed seed counts.

    Input structure:
    - The background is black.
    - One or more gray 4-connected wall loops appear as 1-pixel-thick irregular enclosures, each with exactly one missing wall cell.
    - Each enclosure's interior contains scattered seed pixels in one or more non-black, non-gray colors.

    Rule:
    - For each gray wall component, find the unique missing wall cell whose restoration recreates one enclosed interior region.
    - The missing cell's position relative to the wall centroid determines the outward growth direction: up, down, left, or right.
    - Count the interior seed pixels of each color. For upward or downward growth, order the colors by the seeds' left-to-right positions; for leftward or rightward growth, order them by top-to-bottom positions.
    - Starting one cell outside the gap, draw one ray per color. The ray launch cells occupy consecutive positions perpendicular to the growth direction, centered on the gap, and each ray extends outward by exactly that color's seed count.

    Key insight:
    - The broken wall encodes direction, the seed multiplicities encode lengths, and the seed layout encodes the side-by-side ordering, so the transformation requires combining three separate spatial cues.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for info in _wall_infos(g):
        gap_r, gap_c = info["gap"]
        direction = info["direction"]
        dr, dc = DIR_VECS[direction]
        perp_r, perp_c = ((0, 1) if direction in ("up", "down") else (1, 0))
        offset0 = -(len(info["groups"]) // 2)
        start_r = gap_r + dr
        start_c = gap_c + dc

        for idx, (color, length) in enumerate(info["groups"]):
            shift = offset0 + idx
            ray_r = start_r + perp_r * shift
            ray_c = start_c + perp_c * shift
            for step in range(length):
                rr = ray_r + dr * step
                cc = ray_c + dc * step
                if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
                    out[rr, cc] = int(color)

    return out


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _boundary_from_fill(fill: np.ndarray) -> np.ndarray:
    h, w = fill.shape
    boundary = np.zeros((h, w), dtype=bool)
    for r, c in np.argwhere(fill):
        rr = int(r)
        cc = int(c)
        for nr, nc in _neighbors4(rr, cc):
            if not (0 <= nr < h and 0 <= nc < w and fill[nr, nc]):
                boundary[rr, cc] = True
                break
    return boundary


def _make_fill_shape(direction: str) -> np.ndarray:
    base_h = random.randint(*BASE_H_RANGE)
    base_w = random.randint(*BASE_W_RANGE)
    canvas = np.zeros((base_h + 6, base_w + 6), dtype=bool)
    r0 = c0 = 3
    canvas[r0 : r0 + base_h, c0 : c0 + base_w] = True

    sides = [side for side in DIR_NAMES if side != direction]
    random.shuffle(sides)
    for side in sides[: random.randint(0, MAX_BUMPS)]:
        depth = random.randint(*BUMP_DEPTH_RANGE)
        if side in ("left", "right"):
            span_hi = base_h - 2
            if span_hi < BUMP_SPAN_MIN:
                continue
            span = random.randint(BUMP_SPAN_MIN, span_hi)
            top = random.randint(r0 + 1, r0 + base_h - span - 1)
            if side == "left":
                canvas[top : top + span, c0 - depth : c0] = True
            else:
                canvas[top : top + span, c0 + base_w : c0 + base_w + depth] = True
        else:
            span_hi = base_w - 2
            if span_hi < BUMP_SPAN_MIN:
                continue
            span = random.randint(BUMP_SPAN_MIN, span_hi)
            left = random.randint(c0 + 1, c0 + base_w - span - 1)
            if side == "up":
                canvas[r0 - depth : r0, left : left + span] = True
            else:
                canvas[r0 + base_h : r0 + base_h + depth, left : left + span] = True

    return _crop_mask(canvas)


def _choose_gap(boundary: np.ndarray, fill: np.ndarray, direction: str, n_colors: int) -> tuple[int, int] | None:
    h, w = fill.shape
    radius = n_colors // 2
    candidates: list[tuple[int, int]] = []

    if direction == "up":
        row = min(r for r, _c in np.argwhere(boundary))
        for c in range(radius, w - radius):
            if (
                boundary[row, c]
                and 0 < c < w - 1
                and boundary[row, c - 1]
                and boundary[row, c + 1]
                and row + 1 < h
                and fill[row + 1, c]
            ):
                candidates.append((row, c))
    elif direction == "down":
        row = max(r for r, _c in np.argwhere(boundary))
        for c in range(radius, w - radius):
            if (
                boundary[row, c]
                and 0 < c < w - 1
                and boundary[row, c - 1]
                and boundary[row, c + 1]
                and row - 1 >= 0
                and fill[row - 1, c]
            ):
                candidates.append((row, c))
    elif direction == "left":
        col = min(c for _r, c in np.argwhere(boundary))
        for r in range(radius, h - radius):
            if (
                boundary[r, col]
                and 0 < r < h - 1
                and boundary[r - 1, col]
                and boundary[r + 1, col]
                and col + 1 < w
                and fill[r, col + 1]
            ):
                candidates.append((r, col))
    else:
        col = max(c for _r, c in np.argwhere(boundary))
        for r in range(radius, h - radius):
            if (
                boundary[r, col]
                and 0 < r < h - 1
                and boundary[r - 1, col]
                and boundary[r + 1, col]
                and col - 1 >= 0
                and fill[r, col - 1]
            ):
                candidates.append((r, col))

    if not candidates:
        return None

    if direction in ("up", "down"):
        target = w / 2
        candidates.sort(key=lambda rc: (abs(rc[1] - target), abs(rc[0] - h / 2), random.random()))
    else:
        target = h / 2
        candidates.sort(key=lambda rc: (abs(rc[0] - target), abs(rc[1] - w / 2), random.random()))
    return candidates[0]


def _sample_spaced(values: list[int], n: int) -> list[int] | None:
    values = sorted(values)
    for min_gap in (2, 1, 0):
        combos = [
            combo
            for combo in itertools.combinations(values, n)
            if all(combo[i + 1] - combo[i] >= min_gap for i in range(n - 1))
        ]
        if combos:
            return list(random.choice(combos))
    return None


def _build_local_object() -> dict | None:
    direction = random.choice(DIR_NAMES)
    n_colors = random.choice(COLOR_COUNT_CHOICES)
    fill = _make_fill_shape(direction)
    boundary = _boundary_from_fill(fill)
    interior = fill & ~boundary
    if int(interior.sum()) < n_colors:
        return None

    gap = _choose_gap(boundary, fill, direction, n_colors)
    if gap is None:
        return None

    h, w = fill.shape
    if direction in ("up", "down"):
        buckets = {c: [r for r in range(h) if interior[r, c]] for c in range(w)}
    else:
        buckets = {r: [c for c in range(w) if interior[r, c]] for r in range(h)}
    axes = [axis for axis, cells in buckets.items() if cells]
    picked_axes = _sample_spaced(axes, n_colors)
    if picked_axes is None:
        return None

    colors = random.sample(SEED_COLORS, n_colors)
    local = np.full((h, w), BG, dtype=int)
    local[boundary] = WALL
    local[gap] = BG

    counts: list[int] = []
    for color, axis in zip(colors, sorted(picked_axes)):
        available = buckets[axis]
        count = random.randint(1, min(MAX_BRANCH_LEN, len(available)))
        counts.append(count)
        for pos in sorted(random.sample(available, count)):
            if direction in ("up", "down"):
                local[pos, axis] = int(color)
            else:
                local[axis, pos] = int(color)

    if n_colors == 1 and counts[0] == 1 and random.random() < 0.75:
        return None

    return {
        "grid": local,
        "direction": direction,
        "max_len": max(counts),
    }


def _reserve_box(r0: int, c0: int, h: int, w: int, direction: str, max_len: int) -> tuple[int, int, int, int]:
    rr0, cc0 = r0, c0
    rr1, cc1 = r0 + h - 1, c0 + w - 1
    if direction == "up":
        rr0 -= max_len
    elif direction == "down":
        rr1 += max_len
    elif direction == "left":
        cc0 -= max_len
    else:
        cc1 += max_len
    return rr0, cc0, rr1, cc1


def _place_object(grid: np.ndarray, occupied: np.ndarray, obj: dict) -> bool:
    h, w = obj["grid"].shape
    direction = obj["direction"]
    max_len = obj["max_len"]
    H, W = grid.shape

    min_r = 1 + (max_len if direction == "up" else 0)
    max_r = H - h - 1 - (max_len if direction == "down" else 0)
    min_c = 1 + (max_len if direction == "left" else 0)
    max_c = W - w - 1 - (max_len if direction == "right" else 0)
    if min_r > max_r or min_c > max_c:
        return False

    for _ in range(200):
        r0 = random.randint(min_r, max_r)
        c0 = random.randint(min_c, max_c)
        rr0, cc0, rr1, cc1 = _reserve_box(r0, c0, h, w, direction, max_len)
        rr0 = max(0, rr0 - 1)
        cc0 = max(0, cc0 - 1)
        rr1 = min(H - 1, rr1 + 1)
        cc1 = min(W - 1, cc1 + 1)
        if occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
            continue

        patch = obj["grid"]
        mask = patch != BG
        view = grid[r0 : r0 + h, c0 : c0 + w]
        if bool((view[mask] != BG).any()):
            continue
        view[mask] = patch[mask]
        occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
        return True

    return False


def _grid_shape_for_objects(objects: list[dict]) -> tuple[int, int]:
    count = len(objects)
    (h_lo, h_hi), (w_lo, w_hi) = GRID_SIZE_BY_COUNT[count]
    max_res_h = 0
    max_res_w = 0
    total_area = 0
    for obj in objects:
        h, w = obj["grid"].shape
        if obj["direction"] in ("up", "down"):
            res_h = h + obj["max_len"]
            res_w = w
        else:
            res_h = h
            res_w = w + obj["max_len"]
        max_res_h = max(max_res_h, res_h + 2)
        max_res_w = max(max_res_w, res_w + 2)
        total_area += (res_h + 2) * (res_w + 2)

    side_hint = min(30, max(int(total_area ** 0.5) + 4, max_res_h, max_res_w))
    h_low = max(h_lo, max_res_h, side_hint - 3)
    h_high = min(30, max(h_low, min(h_hi, side_hint + 3)))
    w_low = max(w_lo, max_res_w, side_hint - 3)
    w_high = min(30, max(w_low, min(w_hi, side_hint + 3)))
    return random.randint(h_low, h_high), random.randint(w_low, w_high)


def _gen_attempt() -> np.ndarray | None:
    n_objects = random.choice(OBJECT_COUNT_CHOICES)
    objects: list[dict] = []
    for _ in range(n_objects):
        obj = _build_local_object()
        if obj is None:
            return None
        objects.append(obj)

    if not any(obj["max_len"] > 1 or (obj["grid"] != BG).sum() > (obj["grid"] == WALL).sum() + 1 for obj in objects):
        return None

    h, w = _grid_shape_for_objects(objects)
    grid = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    random.shuffle(objects)
    for obj in objects:
        if not _place_object(grid, occupied, obj):
            return None

    infos = _wall_infos(grid)
    if len(infos) != n_objects:
        return None
    if int((grid == BG).sum()) <= (h * w) // 2:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if out.shape != grid.shape or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
