from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
WALL = 5
ORANGE = 7
INDICATOR_COLORS = (1, 2, 3, 4, 7)

GRID_MIN = 15
GRID_MAX = 22
MIN_QUADRANT = 5
TARGET_REGION_COUNTS = (4, 5)
STUB_PROBABILITY = 0.65
MAX_GENERATION_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _component_masks(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            comp = np.zeros((h, w), dtype=bool)
            queue = deque([(r, c)])
            seen[r, c] = True
            while queue:
                rr, cc = queue.popleft()
                comp[rr, cc] = True
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(comp)
    return components


def _boundary_mask(region_mask: np.ndarray) -> np.ndarray:
    boundary = np.zeros_like(region_mask, dtype=bool)
    for r, c in np.argwhere(region_mask):
        rr = int(r)
        cc = int(c)
        for nr, nc in _neighbors4(rr, cc):
            if not _in_bounds(region_mask.shape, nr, nc) or not region_mask[nr, nc]:
                boundary[rr, cc] = True
                break
    return boundary


def _layer_depths(region_mask: np.ndarray) -> np.ndarray:
    depths = np.full(region_mask.shape, -1, dtype=int)
    boundary = _boundary_mask(region_mask)
    queue = deque()

    for r, c in np.argwhere(boundary):
        rr = int(r)
        cc = int(c)
        depths[rr, cc] = 0
        queue.append((rr, cc))

    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors8(r, c):
            if not _in_bounds(region_mask.shape, nr, nc):
                continue
            if not region_mask[nr, nc] or depths[nr, nc] != -1:
                continue
            depths[nr, nc] = depths[r, c] + 1
            queue.append((nr, nc))

    assert np.all(depths[region_mask] >= 0)
    return depths


def _analyze_regions(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    passable = g != WALL
    regions: list[dict] = []

    for region_mask in _component_masks(passable):
        seed_mask = region_mask & (g != BG) & (g != WALL)
        seed_coords = np.argwhere(seed_mask)
        assert len(seed_coords) == 1, "each region must contain exactly one indicator pixel"
        seed_r, seed_c = map(int, seed_coords[0])
        depths = _layer_depths(region_mask)
        regions.append(
            {
                "mask": region_mask,
                "area": int(region_mask.sum()),
                "seed": (seed_r, seed_c),
                "color": int(g[seed_r, seed_c]),
                "depths": depths,
            }
        )

    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor wall-separated regions by flooding from their indicators, ranking areas, and applying three distinct fills.

    Input structure:
    - A black background grid contains a connected gray wall network that splits the
      canvas into four or five 4-connected non-wall regions.
    - Each region contains exactly one indicator pixel, and indicator colors are
      distinct members of {blue, red, green, yellow, orange}.
    - The walls remain fixed and the rest of each region is initially black.

    Rule:
    - Treat every non-wall connected component as the flood-filled region of its
      single indicator pixel and measure that region's area.
    - Rank the regions by area from largest to smallest.
    - Fill the largest region by concentric 8-neighbor layers: layer 0 keeps the
      indicator color, odd depths become orange, and even depths return to the
      indicator color; fill the smallest region solidly with its indicator color;
      erase every middle-ranked region to black.
    - Keep all gray wall pixels unchanged.

    Key insight:
    The flood-filled region does double duty: it reveals both the shape whose
    boundary drives the concentric layering and the area order that selects which
    of the three region-level transformations is applied.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    wall_mask = g == WALL
    regions = _analyze_regions(g)
    ranked = sorted(regions, key=lambda reg: (-reg["area"], reg["seed"][0], reg["seed"][1]))

    out = np.full(g.shape, BG, dtype=int)
    out[wall_mask] = WALL

    largest = ranked[0]
    smallest = ranked[-1]

    for reg in ranked:
        if reg is largest:
            fill = np.where(reg["depths"] % 2 == 0, reg["color"], ORANGE)
            out[reg["mask"]] = fill[reg["mask"]]
        elif reg is smallest and reg is not largest:
            out[reg["mask"]] = reg["color"]

    return out


def _room_rects(h: int, w: int, split_r: int, split_c: int) -> list[tuple[int, int, int, int]]:
    return [
        (0, 0, split_r - 1, split_c - 1),
        (0, split_c + 1, split_r - 1, w - 1),
        (split_r + 1, 0, h - 1, split_c - 1),
        (split_r + 1, split_c + 1, h - 1, w - 1),
    ]


def _draw_base_walls(h: int, w: int, split_r: int, split_c: int) -> np.ndarray:
    walls = np.zeros((h, w), dtype=bool)
    walls[split_r, :] = True
    walls[:, split_c] = True
    return walls


def _split_room(
    walls: np.ndarray,
    room: tuple[int, int, int, int],
    orientation: str,
    position: int,
) -> list[tuple[int, int, int, int]]:
    r0, c0, r1, c1 = room
    if orientation == "V":
        walls[r0 : r1 + 1, position] = True
        return [(r0, c0, r1, position - 1), (r0, position + 1, r1, c1)]
    walls[position, c0 : c1 + 1] = True
    return [(r0, c0, position - 1, c1), (position + 1, c0, r1, c1)]


def _room_side_has_wall(walls: np.ndarray, room: tuple[int, int, int, int], side: str) -> bool:
    r0, c0, r1, c1 = room
    if side == "top":
        return r0 > 0 and bool(walls[r0 - 1, c0 : c1 + 1].any())
    if side == "bottom":
        return r1 + 1 < walls.shape[0] and bool(walls[r1 + 1, c0 : c1 + 1].any())
    if side == "left":
        return c0 > 0 and bool(walls[r0 : r1 + 1, c0 - 1].any())
    return c1 + 1 < walls.shape[1] and bool(walls[r0 : r1 + 1, c1 + 1].any())


def _try_add_stub(
    walls: np.ndarray,
    room: tuple[int, int, int, int],
    target_region_count: int,
    rng: random.Random,
) -> bool:
    r0, c0, r1, c1 = room
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    options = []

    if height >= 4 and width >= 3:
        if _room_side_has_wall(walls, room, "top"):
            options.append("top")
        if _room_side_has_wall(walls, room, "bottom"):
            options.append("bottom")
    if width >= 4 and height >= 3:
        if _room_side_has_wall(walls, room, "left"):
            options.append("left")
        if _room_side_has_wall(walls, room, "right"):
            options.append("right")

    rng.shuffle(options)
    baseline_wall_count = int(walls.sum())

    for side in options:
        added: list[tuple[int, int]] = []
        if side in {"top", "bottom"}:
            col = rng.randint(c0 + 1, c1 - 1)
            length = rng.randint(1, height - 2)
            if side == "top":
                added = [(r0 + k, col) for k in range(length)]
            else:
                added = [(r1 - k, col) for k in range(length)]
        else:
            row = rng.randint(r0 + 1, r1 - 1)
            length = rng.randint(1, width - 2)
            if side == "left":
                added = [(row, c0 + k) for k in range(length)]
            else:
                added = [(row, c1 - k) for k in range(length)]

        if not added or any(walls[r, c] for r, c in added):
            continue

        trial = walls.copy()
        for r, c in added:
            trial[r, c] = True

        if int(trial.sum()) <= baseline_wall_count:
            continue
        if len(_component_masks(trial)) != 1:
            continue
        if len(_component_masks(~trial)) != target_region_count:
            continue

        walls[:, :] = trial
        return True

    return False


def _indicator_candidates(region_mask: np.ndarray, depths: np.ndarray, wall_mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = region_mask.shape
    candidates = []
    better = []
    best = []

    for r, c in np.argwhere(region_mask):
        rr = int(r)
        cc = int(c)
        if any(_in_bounds(region_mask.shape, nr, nc) and wall_mask[nr, nc] for nr, nc in _neighbors4(rr, cc)):
            continue
        candidates.append((rr, cc))
        if 0 < rr < h - 1 and 0 < cc < w - 1:
            better.append((rr, cc))
            if depths[rr, cc] >= 1:
                best.append((rr, cc))

    return best or better or candidates


def _assign_colors(region_count: int, rng: random.Random) -> list[int]:
    largest_color = rng.choice([1, 2, 3, 4])
    remaining = [color for color in INDICATOR_COLORS if color != largest_color]

    if region_count == 4 and 7 in remaining and rng.random() < 0.75:
        picked = [7]
        picked.extend(rng.sample([color for color in remaining if color != 7], 2))
        rng.shuffle(picked)
        return [largest_color, *picked]

    return [largest_color, *rng.sample(remaining, region_count - 1)]


def _build_candidate(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(GRID_MIN, GRID_MAX)
    w = rng.randint(GRID_MIN, GRID_MAX)
    split_r = rng.randint(MIN_QUADRANT, h - MIN_QUADRANT - 1)
    split_c = rng.randint(MIN_QUADRANT, w - MIN_QUADRANT - 1)

    walls = _draw_base_walls(h, w, split_r, split_c)
    rooms = _room_rects(h, w, split_r, split_c)
    target_region_count = rng.choice(TARGET_REGION_COUNTS)

    if target_region_count == 5:
        feasible: list[tuple[int, str]] = []
        for idx, room in enumerate(rooms):
            r0, c0, r1, c1 = room
            height = r1 - r0 + 1
            width = c1 - c0 + 1
            if width >= 7:
                feasible.append((idx, "V"))
            if height >= 7:
                feasible.append((idx, "H"))
        if not feasible:
            return None

        room_idx, orientation = rng.choice(feasible)
        room = rooms.pop(room_idx)
        r0, c0, r1, c1 = room
        if orientation == "V":
            position = rng.randint(c0 + 3, c1 - 3)
        else:
            position = rng.randint(r0 + 3, r1 - 3)
        rooms.extend(_split_room(walls, room, orientation, position))

    if len(_component_masks(walls)) != 1:
        return None

    if rng.random() < STUB_PROBABILITY:
        room_order = list(range(len(rooms)))
        rng.shuffle(room_order)
        room_order.sort(key=lambda idx: (rooms[idx][2] - rooms[idx][0] + 1) * (rooms[idx][3] - rooms[idx][1] + 1), reverse=True)
        for idx in room_order[:3]:
            if _try_add_stub(walls, rooms[idx], target_region_count, rng):
                break

    regions = []
    for region_mask in _component_masks(~walls):
        depths = _layer_depths(region_mask)
        regions.append(
            {
                "mask": region_mask,
                "area": int(region_mask.sum()),
                "depths": depths,
            }
        )

    if len(regions) != target_region_count:
        return None

    areas = [reg["area"] for reg in regions]
    if len(set(areas)) != len(areas):
        return None

    ranked = sorted(regions, key=lambda reg: reg["area"], reverse=True)
    if ranked[0]["depths"][ranked[0]["mask"]].max() < 2:
        return None

    indicator_cells = []
    for reg in regions:
        candidates = _indicator_candidates(reg["mask"], reg["depths"], walls)
        if not candidates:
            return None
        indicator_cells.append(rng.choice(candidates))

    grid = np.full((h, w), BG, dtype=int)
    grid[walls] = WALL

    colors = _assign_colors(len(regions), rng)
    sorted_regions = sorted(regions, key=lambda reg: (-reg["area"], rng.random()))
    region_to_color = {id(reg): color for reg, color in zip(sorted_regions, colors)}

    for reg, cell in zip(regions, indicator_cells):
        r, c = cell
        grid[r, c] = region_to_color[id(reg)]

    analyzed = _analyze_regions(grid)
    if len(analyzed) != target_region_count:
        return None

    analyzed_ranked = sorted(analyzed, key=lambda reg: (-reg["area"], reg["seed"][0], reg["seed"][1]))
    if len({reg["area"] for reg in analyzed_ranked}) != target_region_count:
        return None
    if analyzed_ranked[0]["color"] == ORANGE:
        return None
    if analyzed_ranked[0]["depths"][analyzed_ranked[0]["mask"]].max() < 2:
        return None

    out = main(grid)
    if out.shape != grid.shape or np.array_equal(out, grid):
        return None

    largest = analyzed_ranked[0]
    smallest = analyzed_ranked[-1]
    if largest["color"] != ORANGE and not np.any(out[largest["mask"]] == ORANGE):
        return None
    if not np.all(out[smallest["mask"]] == smallest["color"]):
        return None
    for reg in analyzed_ranked[1:-1]:
        if not np.all(out[reg["mask"]] == BG):
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        rng = random.Random()
        grid = _build_candidate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
