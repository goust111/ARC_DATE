from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


BG = 0
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left
SIDE_TO_OPPOSITE_DIR = {"top": 2, "right": 3, "bottom": 0, "left": 1}
CW_GAP_SIDES = {"top", "right"}

# Generator palette / bounds.
WALL = 5
SEED_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
SIDES = ["top", "right", "bottom", "left"]
GRID_SIDE_RANGE = (18, 31)
INNER_SIDE_RANGE = (5, 10)
REGION_PAD = 1
MAX_REGIONS = 4
MAX_ATTEMPTS = 400


@dataclass(frozen=True)
class Region:
    bbox: tuple[int, int, int, int]  # r0, c0, r1, c1 inclusive border bbox
    gap: tuple[int, int]
    gap_side: str
    seed: tuple[int, int]
    color: int


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in DIRS:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _perimeter_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _gap_side_from_cell(gap: tuple[int, int], bbox: tuple[int, int, int, int]) -> str:
    r0, c0, r1, c1 = bbox
    gr, gc = gap
    if gr == r0 and c0 < gc < c1:
        return "top"
    if gc == c1 and r0 < gr < r1:
        return "right"
    if gr == r1 and c0 < gc < c1:
        return "bottom"
    if gc == c0 and r0 < gr < r1:
        return "left"
    raise AssertionError("gap must lie on exactly one non-corner border cell")


def _distance_map(seed: tuple[int, int], bbox: tuple[int, int, int, int]) -> dict[str, int]:
    r0, c0, r1, c1 = bbox
    sr, sc = seed
    return {
        "top": sr - (r0 + 1),
        "right": (c1 - 1) - sc,
        "bottom": (r1 - 1) - sr,
        "left": sc - (c0 + 1),
    }


def _unique_min_side(distances: dict[str, int]) -> str:
    best = min(distances.values())
    sides = [side for side, dist in distances.items() if dist == best]
    assert len(sides) == 1, "seed must be biased toward exactly one border side"
    return sides[0]


def _unique_min_side_or_none(distances: dict[str, int]) -> str | None:
    best = min(distances.values())
    sides = [side for side, dist in distances.items() if dist == best]
    if len(sides) != 1:
        return None
    return sides[0]


def _extract_regions(grid: np.ndarray) -> list[Region]:
    colors = [int(color) for color in np.unique(grid) if int(color) != BG]
    assert colors, "expected at least one non-background color"

    counts = {color: int(np.count_nonzero(grid == color)) for color in colors}
    wall_colors = [color for color, count in counts.items() if count > 1]
    assert len(wall_colors) == 1, "expected one repeated wall color and singleton seed colors"
    wall = wall_colors[0]

    regions: list[Region] = []
    for comp in _components4(grid == wall):
        bbox = _bbox(comp)
        r0, c0, r1, c1 = bbox
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "rectangle must have an interior"

        border = _perimeter_cells(bbox)
        comp_set = set(comp)
        assert comp_set <= border, "wall pixels must lie on the rectangle perimeter"

        missing = list(border - comp_set)
        assert len(missing) == 1, "each rectangle border must have exactly one gap"
        gap = missing[0]
        gap_side = _gap_side_from_cell(gap, bbox)

        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        seed_rel = np.argwhere((interior != BG) & (interior != wall))
        assert seed_rel.shape[0] == 1, "each region must contain exactly one seed"
        sr = int(seed_rel[0, 0] + r0 + 1)
        sc = int(seed_rel[0, 1] + c0 + 1)
        color = int(grid[sr, sc])
        assert counts[color] == 1, "each seed color must be unique"

        regions.append(Region(bbox=bbox, gap=gap, gap_side=gap_side, seed=(sr, sc), color=color))

    assert regions, "expected at least one rectangle region"
    return sorted(regions, key=lambda region: region.bbox)


def _paint_spiral(out: np.ndarray, region: Region) -> None:
    r0, c0, r1, c1 = region.bbox
    sr, sc = region.seed
    assert r0 < sr < r1 and c0 < sc < c1

    distances = _distance_map(region.seed, region.bbox)
    nearest_side = _unique_min_side(distances)
    dir_idx = SIDE_TO_OPPOSITE_DIR[nearest_side]
    turn = 1 if region.gap_side in CW_GAP_SIDES else -1

    cur_r, cur_c = sr, sc
    stuck_turns = 0
    out[cur_r, cur_c] = region.color
    while stuck_turns < 4:
        dr, dc = DIRS[dir_idx]
        nr = cur_r + dr
        nc = cur_c + dc
        if r0 < nr < r1 and c0 < nc < c1 and int(out[nr, nc]) == BG:
            out[nr, nc] = region.color
            cur_r, cur_c = nr, nc
            stuck_turns = 0
        else:
            dir_idx = (dir_idx + turn) % 4
            stuck_turns += 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow a seed-colored spiral inside each grey rectangle.

    The border gap only encodes spiral handedness:
      - top/right gap -> clockwise turns
      - bottom/left gap -> counterclockwise turns

    The seed's nearest border determines the first segment: the spiral starts by
    moving away from that nearest border into the larger interior space.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    for region in _extract_regions(g):
        _paint_spiral(out, region)
    return out


def _max_regions_for_shape(h: int, w: int) -> int:
    area = h * w
    if area < 430:
        return 2
    if area < 620:
        return 3
    return MAX_REGIONS


def _sample_gap(side: str, bbox_h: int, bbox_w: int) -> tuple[int, int]:
    if side == "top":
        return 0, random.randint(1, bbox_w - 2)
    if side == "right":
        return random.randint(1, bbox_h - 2), bbox_w - 1
    if side == "bottom":
        return bbox_h - 1, random.randint(1, bbox_w - 2)
    return random.randint(1, bbox_h - 2), 0


def _sample_seed_cell(inner_h: int, inner_w: int, nearest_side: str) -> tuple[int, int]:
    if nearest_side == "top":
        rows = range(0, min(2, inner_h))
        cols = range(1, inner_w - 1)
    elif nearest_side == "right":
        rows = range(1, inner_h - 1)
        cols = range(max(0, inner_w - 2), inner_w)
    elif nearest_side == "bottom":
        rows = range(max(0, inner_h - 2), inner_h)
        cols = range(1, inner_w - 1)
    else:
        rows = range(1, inner_h - 1)
        cols = range(0, min(2, inner_w))

    candidates = []
    bbox = (0, 0, inner_h + 1, inner_w + 1)
    for sr in rows:
        for sc in cols:
            distances = _distance_map((sr + 1, sc + 1), bbox)
            if _unique_min_side_or_none(distances) == nearest_side:
                candidates.append((sr, sc))
    assert candidates, "generator must be able to place a biased seed"
    return random.choice(candidates)


def _can_place(occupied: np.ndarray, top: int, left: int, bbox_h: int, bbox_w: int) -> bool:
    h, w = occupied.shape
    bottom = top + bbox_h
    right = left + bbox_w
    if top < 1 or left < 1 or bottom > h - 1 or right > w - 1:
        return False
    r0 = max(0, top - REGION_PAD)
    c0 = max(0, left - REGION_PAD)
    r1 = min(h, bottom + REGION_PAD)
    c1 = min(w, right + REGION_PAD)
    return not bool(occupied[r0:r1, c0:c1].any())


def _mark_occupied(occupied: np.ndarray, top: int, left: int, bbox_h: int, bbox_w: int) -> None:
    h, w = occupied.shape
    bottom = top + bbox_h
    right = left + bbox_w
    r0 = max(0, top - REGION_PAD)
    c0 = max(0, left - REGION_PAD)
    r1 = min(h, bottom + REGION_PAD)
    c1 = min(w, right + REGION_PAD)
    occupied[r0:r1, c0:c1] = True


def _draw_region(grid: np.ndarray, region: Region, wall: int) -> None:
    r0, c0, r1, c1 = region.bbox
    grid[r0, c0 : c1 + 1] = wall
    grid[r1, c0 : c1 + 1] = wall
    grid[r0 : r1 + 1, c0] = wall
    grid[r0 : r1 + 1, c1] = wall
    grid[region.gap] = BG
    grid[region.seed] = region.color


def _sample_layout() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(*GRID_SIDE_RANGE)
    w = random.randint(*GRID_SIDE_RANGE)
    n_regions = random.randint(1, _max_regions_for_shape(h, w))
    colors = random.sample(SEED_COLORS, n_regions)

    specs = []
    for color in colors:
        inner_h = random.randint(*INNER_SIDE_RANGE)
        inner_w = random.randint(*INNER_SIDE_RANGE)
        bbox_h = inner_h + 2
        bbox_w = inner_w + 2

        clockwise = bool(random.getrandbits(1))
        gap_side = random.choice(["top", "right"] if clockwise else ["bottom", "left"])
        seed_side = random.choice(SIDES)
        specs.append(
            {
                "color": color,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "bbox_h": bbox_h,
                "bbox_w": bbox_w,
                "gap_side": gap_side,
                "seed_side": seed_side,
            }
        )

    specs.sort(key=lambda spec: spec["bbox_h"] * spec["bbox_w"], reverse=True)

    grid = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    regions: list[Region] = []

    for spec in specs:
        bbox_h = int(spec["bbox_h"])
        bbox_w = int(spec["bbox_w"])
        placements = [
            (top, left)
            for top in range(1, h - bbox_h)
            for left in range(1, w - bbox_w)
            if _can_place(occupied, top, left, bbox_h, bbox_w)
        ]
        if not placements:
            return None

        top, left = random.choice(placements)
        _mark_occupied(occupied, top, left, bbox_h, bbox_w)

        gap_r, gap_c = _sample_gap(str(spec["gap_side"]), bbox_h, bbox_w)
        seed_r, seed_c = _sample_seed_cell(int(spec["inner_h"]), int(spec["inner_w"]), str(spec["seed_side"]))

        region = Region(
            bbox=(top, left, top + bbox_h - 1, left + bbox_w - 1),
            gap=(top + gap_r, left + gap_c),
            gap_side=str(spec["gap_side"]),
            seed=(top + 1 + seed_r, left + 1 + seed_c),
            color=int(spec["color"]),
        )
        _draw_region(grid, region, WALL)
        regions.append(region)

    expected = grid.copy()
    for region in regions:
        _paint_spiral(expected, region)

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    recovered = _extract_regions(grid)
    if len(recovered) != len(regions):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        sample = _sample_layout()
        if sample is None:
            continue
        grid, expected = sample
        if np.array_equal(grid, expected):
            continue
        return grid
    raise RuntimeError("failed to generate a valid spiral-frame input")
