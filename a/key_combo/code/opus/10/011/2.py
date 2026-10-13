from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BACKGROUND = 0
ANCHOR = 1
RAY = 2
OBSTACLE = 5

DIAGONALS = [(-1, -1), (-1, 1), (1, 1), (1, -1)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _bbox(coords: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    pts = list(coords)
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _find_components(mask: np.ndarray) -> tuple[list[dict], np.ndarray]:
    h, w = mask.shape
    labels = np.full((h, w), -1, dtype=int)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != -1:
                continue
            stack = [(r, c)]
            labels[r, c] = len(comps)
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and labels[nr, nc] == -1:
                        labels[nr, nc] = len(comps)
                        stack.append((nr, nc))
            r0, c0, r1, c1 = _bbox(coords)
            comps.append({"coords": coords, "bbox": (r0, c0, r1, c1)})
    return comps, labels


def _find_anchor_origin(grid: np.ndarray) -> tuple[int, int]:
    anchors: list[tuple[int, int]] = []
    h, w = grid.shape
    for r in range(h - 1):
        for c in range(w - 1):
            if np.all(grid[r : r + 2, c : c + 2] == ANCHOR):
                anchors.append((r, c))
    assert len(anchors) == 1, "expected a unique blue 2x2 anchor square"
    return anchors[0]


def _collect_seeds(grid: np.ndarray, anchor_r: int, anchor_c: int) -> list[tuple[int, int, int, int]]:
    seeds: list[tuple[int, int, int, int]] = []
    square_cells = {
        (anchor_r, anchor_c),
        (anchor_r, anchor_c + 1),
        (anchor_r + 1, anchor_c),
        (anchor_r + 1, anchor_c + 1),
    }
    for dr, dc in DIAGONALS:
        corner_r = anchor_r if dr < 0 else anchor_r + 1
        corner_c = anchor_c if dc < 0 else anchor_c + 1
        seed_r = corner_r + dr
        seed_c = corner_c + dc
        if _in_bounds(grid.shape, seed_r, seed_c) and int(grid[seed_r, seed_c]) == ANCHOR:
            seeds.append((seed_r, seed_c, dr, dc))

    blue_coords = {tuple(map(int, rc)) for rc in np.argwhere(grid == ANCHOR)}
    expected = square_cells | {(r, c) for r, c, _dr, _dc in seeds}
    assert blue_coords == expected, "blue cells must be exactly the anchor plus diagonal seeds"
    assert 1 <= len(seeds) <= 2, "expected one or two ray seeds"
    return seeds


def _split_obstacle(
    out: np.ndarray,
    coords: list[tuple[int, int]],
    hit_r: int,
    hit_c: int,
    dr: int,
    dc: int,
) -> None:
    h, w = out.shape
    for r, c in coords:
        out[r, c] = BACKGROUND

    pdr, pdc = dr, -dc
    for r, c in coords:
        side = dr * (r - hit_r) - dc * (c - hit_c)
        if side == 0:
            out[r, c] = RAY
            continue
        nr = r + (pdr if side > 0 else -pdr)
        nc = c + (pdc if side > 0 else -pdc)
        if _in_bounds((h, w), nr, nc):
            out[nr, nc] = OBSTACLE


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast each blue diagonal seed ray through gray rectangles, splitting them open and painting the slit red.

    Input structure:
    - The background is black.
    - A unique blue 2x2 square sits near the center and acts as the anchor.
    - One or two extra blue pixels sit on diagonal corner-neighbor cells of that square; these are ray seeds.
    - Gray solid rectangular obstacles lie farther out on those diagonal rays.

    Rule:
    - Preserve the blue anchor square and the blue seed pixels.
    - From each seed, continue along the same diagonal direction away from the anchor.
    - Background cells on the ray become red.
    - When the ray hits a gray obstacle, split that whole obstacle along the ray's diagonal line:
      cells on one side shift one perpendicular-diagonal step outward, cells on the other side shift the opposite way,
      and the diagonal slit itself becomes red.
    - Continue each ray until it leaves the grid.

    Key insight:
    - The blue corner pixels define outward diagonal rays, and gray blocks do not stop those rays; the rays carve a red diagonal slit through each block and push the two halves apart.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    anchor_r, anchor_c = _find_anchor_origin(g)
    seeds = _collect_seeds(g, anchor_r, anchor_c)

    obstacle_components, obstacle_labels = _find_components(g == OBSTACLE)
    out = g.copy()
    processed: set[int] = set()

    for seed_r, seed_c, dr, dc in seeds:
        r = seed_r + dr
        c = seed_c + dc
        while _in_bounds(g.shape, r, c):
            cell = int(g[r, c])
            if cell == BACKGROUND:
                out[r, c] = RAY
            elif cell == OBSTACLE:
                comp_id = int(obstacle_labels[r, c])
                assert comp_id >= 0, "obstacle lookup failed"
                if comp_id not in processed:
                    _split_obstacle(out, obstacle_components[comp_id]["coords"], r, c, dr, dc)
                    processed.add(comp_id)
            else:
                raise AssertionError("ray path may only meet background or gray obstacles")
            r += dr
            c += dc

    return out


def _ray_cells_from_seed(seed_r: int, seed_c: int, dr: int, dc: int, shape: tuple[int, int]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    r = seed_r + dr
    c = seed_c + dc
    while _in_bounds(shape, r, c):
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _central_offsets(extra: int) -> list[int]:
    return sorted({extra // 2, (extra + 1) // 2})


def _build_rect_cells(
    segment: list[tuple[int, int]],
    dr: int,
    dc: int,
    thickness: int,
    length: int,
    offset: int,
) -> set[tuple[int, int]]:
    if dr == dc:
        r_min = min(r for r, _ in segment)
        c_min = min(c for _, c in segment)
        r0 = r_min
        c0 = c_min - offset
        return {
            (r0 + rr, c0 + cc)
            for rr in range(thickness)
            for cc in range(length)
        }

    r_min = min(r for r, _ in segment)
    c_max = max(c for _, c in segment)
    r0 = r_min - offset
    c0 = c_max - (thickness - 1)
    return {
        (r0 + rr, c0 + cc)
        for rr in range(length)
        for cc in range(thickness)
    }


def _place_obstacles_for_ray(
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    ray_cells: list[tuple[int, int]],
    dr: int,
    dc: int,
    count: int,
) -> bool:
    placed: list[tuple[int, int]] = []
    if len(ray_cells) < 5:
        return False

    for _ in range(count):
        success = False
        for _attempt in range(200):
            thickness = random.randint(1, 2)
            length = random.randint(2, 4)
            if len(ray_cells) - thickness <= 2:
                break
            start = random.randint(2, len(ray_cells) - thickness)
            end = start + thickness - 1
            if any(not (end + 2 < s or e + 2 < start) for s, e in placed):
                continue

            segment = ray_cells[start : start + thickness]
            offsets = _central_offsets(length - thickness)
            random.shuffle(offsets)
            for offset in offsets:
                cells = _build_rect_cells(segment, dr, dc, thickness, length, offset)
                if any(not _in_bounds(grid.shape, r, c) for r, c in cells):
                    continue
                if any((r, c) in occupied or int(grid[r, c]) != BACKGROUND for r, c in cells):
                    continue
                for r, c in cells:
                    grid[r, c] = OBSTACLE
                    occupied.add((r, c))
                placed.append((start, end))
                success = True
                break
            if success:
                break
        if not success:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(18, 24)
    w = random.randint(18, 24)
    grid = np.zeros((h, w), dtype=int)

    row_choices = list(range(max(4, h // 2 - 2), min(h - 5, h // 2 + 1) + 1))
    col_choices = list(range(max(4, w // 2 - 2), min(w - 5, w // 2 + 1) + 1))
    if not row_choices or not col_choices:
        return None

    anchor_r = random.choice(row_choices)
    anchor_c = random.choice(col_choices)
    grid[anchor_r : anchor_r + 2, anchor_c : anchor_c + 2] = ANCHOR
    occupied = {
        (anchor_r, anchor_c),
        (anchor_r, anchor_c + 1),
        (anchor_r + 1, anchor_c),
        (anchor_r + 1, anchor_c + 1),
    }

    seed_count = 1 if random.random() < 0.45 else 2
    dirs = random.sample(DIAGONALS, seed_count)

    rays: list[tuple[int, int, int, int, list[tuple[int, int]]]] = []
    for dr, dc in dirs:
        corner_r = anchor_r if dr < 0 else anchor_r + 1
        corner_c = anchor_c if dc < 0 else anchor_c + 1
        seed_r = corner_r + dr
        seed_c = corner_c + dc
        if not _in_bounds(grid.shape, seed_r, seed_c):
            return None
        grid[seed_r, seed_c] = ANCHOR
        occupied.add((seed_r, seed_c))
        ray_cells = _ray_cells_from_seed(seed_r, seed_c, dr, dc, grid.shape)
        if len(ray_cells) < 5:
            return None
        rays.append((seed_r, seed_c, dr, dc, ray_cells))

    for _seed_r, _seed_c, dr, dc, ray_cells in rays:
        max_obs = min(3, max(1, (len(ray_cells) - 1) // 3))
        count = random.randint(1, max_obs)
        if not _place_obstacles_for_ray(grid, occupied, ray_cells, dr, dc, count):
            return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    if not np.any(out == RAY):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
