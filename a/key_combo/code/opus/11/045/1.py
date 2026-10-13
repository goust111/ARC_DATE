from __future__ import annotations

from collections import Counter, deque

import numpy as np


BG = 0
WALL = 1
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7
MARKER_COLORS = (RED, YELLOW, ORANGE)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

_RNG = np.random.default_rng()
_GEN_ATTEMPTS = 400
_GRID_MIN = 20
_GRID_MAX = 27
_SLOTS = ("tl", "tr", "bl", "br")


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _neighbors8(r: int, c: int):
    for dr, dc in DIR8:
        yield r + dr, c + dc


def _is_barrier(value: int) -> bool:
    return value == WALL or value in MARKER_COLORS


def _flood_regions(grid: np.ndarray):
    h, w = grid.shape
    region_id = np.full((h, w), -1, dtype=int)
    regions: list[dict] = []
    next_id = 0

    for sr in range(h):
        for sc in range(w):
            if region_id[sr, sc] >= 0 or _is_barrier(int(grid[sr, sc])):
                continue
            q = deque([(sr, sc)])
            region_id[sr, sc] = next_id
            cells: list[tuple[int, int]] = []
            greens: list[tuple[int, int]] = []
            touches_border = False
            while q:
                r, c = q.popleft()
                cells.append((r, c))
                if int(grid[r, c]) == GREEN:
                    greens.append((r, c))
                if r in (0, h - 1) or c in (0, w - 1):
                    touches_border = True
                for nr, nc in _neighbors4(r, c):
                    if not _in_bounds(grid.shape, nr, nc):
                        continue
                    if region_id[nr, nc] >= 0 or _is_barrier(int(grid[nr, nc])):
                        continue
                    region_id[nr, nc] = next_id
                    q.append((nr, nc))
            regions.append(
                {
                    "id": next_id,
                    "cells": cells,
                    "greens": greens,
                    "touches_border": touches_border,
                    "marker_cells": [],
                    "marker_counts": Counter(),
                    "winner": None,
                    "runner_up": None,
                }
            )
            next_id += 1
    return region_id, regions


def _analyze(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    region_id, regions = _flood_regions(g)
    marker_adjacency: dict[tuple[int, int], tuple[int, ...]] = {}

    for r, c in np.argwhere(np.isin(g, np.array(MARKER_COLORS, dtype=int))):
        rr = int(r)
        cc = int(c)
        adjacent_regions = sorted(
            {
                int(region_id[nr, nc])
                for nr, nc in _neighbors8(rr, cc)
                if _in_bounds(g.shape, nr, nc) and region_id[nr, nc] >= 0
            }
        )
        if len(adjacent_regions) < 2:
            continue
        marker_adjacency[(rr, cc)] = tuple(adjacent_regions)
        color = int(g[rr, cc])
        for rid in adjacent_regions:
            regions[rid]["marker_cells"].append((rr, cc))
            regions[rid]["marker_counts"][color] += 1

    for region in regions:
        counts = region["marker_counts"]
        if not counts:
            continue
        max_count = max(counts.values())
        winners = [color for color, cnt in counts.items() if cnt == max_count]
        region["winner"] = winners[0] if len(winners) == 1 else None
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        if len(ranked) >= 2:
            region["runner_up"] = ranked[1][0]

    return g, region_id, regions, marker_adjacency


def _ray_lengths(grid: np.ndarray, r: int, c: int) -> list[int]:
    lengths: list[int] = []
    for dr, dc in DIR4:
        rr = r + dr
        cc = c + dc
        n = 0
        while _in_bounds(grid.shape, rr, cc) and not _is_barrier(int(grid[rr, cc])):
            n += 1
            rr += dr
            cc += dc
        lengths.append(n)
    return lengths


def _paint_regions(
    grid: np.ndarray,
    regions: list[dict],
    *,
    step_override: dict[int, int] | None = None,
    color_override: dict[int, int] | None = None,
    return_painted_counts: bool = False,
):
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    painted_counts = [0] * len(regions)

    for region in regions:
        rid = int(region["id"])
        greens = region["greens"]
        if not greens:
            continue
        step = int(step_override.get(rid, len(greens))) if step_override else len(greens)
        color = int(color_override.get(rid, region["winner"])) if color_override else int(region["winner"])
        for r, c in greens:
            for dr, dc in DIR4:
                d = 1
                rr = r + dr
                cc = c + dc
                while _in_bounds(g.shape, rr, cc) and not _is_barrier(int(g[rr, cc])):
                    if d % step == 0 and int(g[rr, cc]) == BG and int(out[rr, cc]) == BG:
                        out[rr, cc] = color
                        painted_counts[rid] += 1
                    rr += dr
                    cc += dc
                    d += 1

    if return_painted_counts:
        return out, painted_counts
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Shoot region-bounded periodic rays from green dots using the neighboring regions' majority marker color.

    Input structure:
    - Black is the background.
    - Blue cells form wall loops; red, yellow, and orange marker pixels sit on wall corners and act as barriers too.
    - The remaining black plus isolated green pixels form 4-connected open regions separated by those barriers.

    Rule:
    - Flood-fill the non-barrier cells into 4-connected regions.
    - For each region, collect every marker whose 8-neighborhood touches that region and at least one different region; the region's ray color is the unique majority marker color among those markers.
    - For every green pixel in the region, let step be the number of green pixels in that region, then extend rays in the four cardinal directions until a wall, marker, or the grid boundary is reached; only cells at distances divisible by step are recolored to the region's voted color.
    - Keep the original walls, markers, and green pixels unchanged.

    Key insight:
    - The color of a region's rays is determined externally by cross-region marker votes, while the spacing of those rays is determined internally by how many isolated green seeds the region contains.
    """
    g, _region_id, regions, marker_adjacency = _analyze(grid)
    assert g.ndim == 2
    assert np.all((0 <= g) & (g <= 9))
    assert marker_adjacency, "expected at least one marker"

    for marker, adjacent_regions in marker_adjacency.items():
        assert len(adjacent_regions) == 2, f"marker {marker} must touch exactly two regions"

    for region in regions:
        assert region["greens"], "every region must contain at least one green seed"
        assert len(region["marker_counts"]) >= 2, "every region must see at least two marker colors"
        assert region["winner"] is not None, "each region needs a unique majority marker color"

    return _paint_regions(g, regions)


def _slot_start_ranges(shape: tuple[int, int], slot: str, box_h: int, box_w: int):
    h, w = shape
    mid_r = h // 2
    mid_c = w // 2
    if slot in ("tl", "tr"):
        r_lo, r_hi = 2, mid_r - box_h - 1
    else:
        r_lo, r_hi = mid_r + 1, h - box_h - 2
    if slot in ("tl", "bl"):
        c_lo, c_hi = 2, mid_c - box_w - 1
    else:
        c_lo, c_hi = mid_c + 1, w - box_w - 2
    if r_lo > r_hi or c_lo > c_hi:
        return None
    return (r_lo, r_hi), (c_lo, c_hi)


def _choose_loops(shape: tuple[int, int]):
    slots = list(_SLOTS)
    missing = slots[int(_RNG.integers(len(slots)))]
    chosen_slots = [slot for slot in slots if slot != missing]
    _RNG.shuffle(chosen_slots)

    loops = []
    for slot in chosen_slots:
        box_h = 7
        box_w = 7
        ranges = _slot_start_ranges(shape, slot, box_h, box_w)
        if ranges is None:
            return None
        (r_lo, r_hi), (c_lo, c_hi) = ranges
        r0 = int(_RNG.integers(r_lo, r_hi + 1))
        c0 = int(_RNG.integers(c_lo, c_hi + 1))
        loops.append({"slot": slot, "r0": r0, "c0": c0, "r1": r0 + box_h - 1, "c1": c0 + box_w - 1})
    return loops


def _loop_corners(loop: dict) -> list[tuple[int, int]]:
    return [
        (int(loop["r0"]), int(loop["c0"])),
        (int(loop["r0"]), int(loop["c1"])),
        (int(loop["r1"]), int(loop["c1"])),
        (int(loop["r1"]), int(loop["c0"])),
    ]


def _draw_loop_walls(grid: np.ndarray, loops: list[dict]) -> None:
    for loop in loops:
        r0 = int(loop["r0"])
        c0 = int(loop["c0"])
        r1 = int(loop["r1"])
        c1 = int(loop["c1"])
        grid[r0 : r1 + 1, c0] = WALL
        grid[r0 : r1 + 1, c1] = WALL
        grid[r0, c0 : c1 + 1] = WALL
        grid[r1, c0 : c1 + 1] = WALL


def _assign_marker_colors(loops: list[dict]):
    for _ in range(300):
        majorities = [int(_RNG.choice(MARKER_COLORS)) for _ in loops]
        if len(set(majorities)) < 2:
            continue
        all_counts = Counter()
        assignments = []
        for majority in majorities:
            secondary_choices = [color for color in MARKER_COLORS if color != majority]
            secondary = int(_RNG.choice(secondary_choices))
            minority_index = int(_RNG.integers(4))
            colors = [majority, majority, majority, majority]
            colors[minority_index] = secondary
            assignments.append(colors)
            all_counts.update(colors)
        if len(all_counts) < 3:
            continue
        max_count = max(all_counts.values())
        winners = [color for color, cnt in all_counts.items() if cnt == max_count]
        if len(winners) != 1:
            continue
        outside_winner = winners[0]
        if all(majority == outside_winner for majority in majorities):
            continue
        return assignments
    return None


def _paint_markers(grid: np.ndarray, loops: list[dict], assignments: list[list[int]]) -> None:
    for loop, colors in zip(loops, assignments):
        for (r, c), color in zip(_loop_corners(loop), colors):
            grid[r, c] = int(color)


def _safe_dot_candidates(grid: np.ndarray, region_cells: list[tuple[int, int]], step: int) -> list[tuple[int, int]]:
    candidates = []
    region_set = set(region_cells)
    for r, c in region_cells:
        if int(grid[r, c]) != BG:
            continue
        if any((nr, nc) not in region_set for nr, nc in _neighbors4(r, c)):
            continue
        lengths = _ray_lengths(grid, r, c)
        if max(lengths) < step:
            continue
        candidates.append((r, c))
    return candidates


def _far_enough(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1])) > 1


def _choose_dots_for_region(grid: np.ndarray, region_cells: list[tuple[int, int]], count: int):
    candidates = _safe_dot_candidates(grid, region_cells, count)
    if len(candidates) < count:
        return None
    for _ in range(300):
        order = list(candidates)
        _RNG.shuffle(order)
        chosen: list[tuple[int, int]] = []
        for cell in order:
            if all(_far_enough(cell, other) for other in chosen):
                chosen.append(cell)
                if len(chosen) == count:
                    return chosen
    return None


def _place_greens(grid: np.ndarray, regions: list[dict]) -> bool:
    border_regions = [region for region in regions if region["touches_border"]]
    if len(border_regions) != 1 or len(regions) != 4:
        return False
    outside_id = int(border_regions[0]["id"])
    interior_ids = [int(region["id"]) for region in regions if int(region["id"]) != outside_id]
    counts = [1, 2, 3]
    _RNG.shuffle(counts)
    target_counts = {outside_id: 4}
    for rid, count in zip(interior_ids, counts):
        target_counts[rid] = count

    for region in regions:
        rid = int(region["id"])
        chosen = _choose_dots_for_region(grid, region["cells"], target_counts[rid])
        if chosen is None:
            return False
        for r, c in chosen:
            grid[r, c] = GREEN
    return True


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        g, _region_id, regions, marker_adjacency = _analyze(grid)
        if len(regions) != 4:
            return False
        border_regions = [region for region in regions if region["touches_border"]]
        if len(border_regions) != 1:
            return False
        if sorted(len(region["greens"]) for region in regions) != [1, 2, 3, 4]:
            return False
        for adjacent_regions in marker_adjacency.values():
            if len(adjacent_regions) != 2:
                return False
        for region in regions:
            if len(region["greens"]) == 0:
                return False
            if len(region["marker_counts"]) < 2:
                return False
            if region["winner"] is None or region["runner_up"] is None:
                return False

        out, painted_counts = _paint_regions(g, regions, return_painted_counts=True)
        if any(count == 0 for count in painted_counts):
            return False

        for region in regions:
            rid = int(region["id"])
            if len(region["greens"]) > 1:
                alt_step = _paint_regions(g, regions, step_override={rid: 1})
                if np.array_equal(out, alt_step):
                    return False
            alt_color = _paint_regions(g, regions, color_override={rid: int(region["runner_up"])})
            if np.array_equal(out, alt_color):
                return False

        winner_by_region = [int(region["winner"]) for region in regions]
        if len(set(winner_by_region)) == 1:
            return False

        return np.array_equal(main(g), out)
    except Exception:
        return False


def _generate_attempt() -> np.ndarray | None:
    h = int(_RNG.integers(_GRID_MIN, _GRID_MAX))
    w = int(_RNG.integers(_GRID_MIN, _GRID_MAX))
    grid = np.full((h, w), BG, dtype=int)

    loops = _choose_loops(grid.shape)
    if loops is None:
        return None
    _draw_loop_walls(grid, loops)

    assignments = _assign_marker_colors(loops)
    if assignments is None:
        return None
    _paint_markers(grid, loops, assignments)

    _g0, _region_id0, regions0, marker_adjacency0 = _analyze(grid)
    if len(regions0) != 4 or len(marker_adjacency0) != 12:
        return None
    if not _place_greens(grid, regions0):
        return None
    if not _validate_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for opus/11/045/1")
