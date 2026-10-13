from __future__ import annotations

from collections import deque
import random

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
COLORS = tuple(range(1, 10))
GEN_RETRIES = 400
PLACEMENT_TRIES = 200


def _same_color_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == 0 or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: set[tuple[int, int]] = set()
            while q:
                rr, cc = q.popleft()
                cells.add((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            anchor = min(cells)
            components.append(
                {
                    "color": color,
                    "cells": cells,
                    "area": len(cells),
                    "anchor": anchor,
                }
            )

    return components


def _find_single_pixel_holes(grid: np.ndarray, component_cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    holes: set[tuple[int, int]] = set()
    for r, c in component_cells:
        for dr, dc in DIR4:
            nr = r + dr
            nc = c + dc
            if (nr, nc) in component_cells:
                continue
            if not (0 < nr < grid.shape[0] - 1 and 0 < nc < grid.shape[1] - 1):
                continue
            if int(grid[nr, nc]) != 0:
                continue
            if all((nr + rr, nc + cc) in component_cells for rr, cc in DIR4):
                holes.add((nr, nc))
    return holes


def _region_infos(grid: np.ndarray) -> list[dict]:
    infos = _same_color_components(grid)
    for info in infos:
        holes = _find_single_pixel_holes(grid, info["cells"])
        info["holes"] = holes
        info["hole_count"] = len(holes)
        info["priority"] = (info["hole_count"], info["area"], -info["anchor"][0], -info["anchor"][1])
    return infos


def _simulate(grid: np.ndarray, collect_trace: bool = False) -> tuple[np.ndarray, list[dict], dict]:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    regions = _region_infos(g)
    immutable_holes = set().union(*(region["holes"] for region in regions)) if regions else set()
    max_layers = max((region["hole_count"] for region in regions), default=0)
    frontiers = [set(region["cells"]) for region in regions]
    trace = {
        "contests": [],
        "has_hole_priority": False,
        "has_area_priority": False,
    }

    for layer in range(1, max_layers + 1):
        proposals: dict[tuple[int, int], list[int]] = {}
        next_frontiers = [set() for _ in regions]

        for idx, region in enumerate(regions):
            if region["hole_count"] < layer or not frontiers[idx]:
                continue
            candidates: set[tuple[int, int]] = set()
            for r, c in frontiers[idx]:
                for dr, dc in DIR4:
                    nr = r + dr
                    nc = c + dc
                    if not (0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]):
                        continue
                    if int(out[nr, nc]) != 0 or (nr, nc) in immutable_holes:
                        continue
                    candidates.add((nr, nc))
            for pos in candidates:
                proposals.setdefault(pos, []).append(idx)

        for pos, idxs in proposals.items():
            winner = max(idxs, key=lambda idx: regions[idx]["priority"])
            out[pos] = int(regions[winner]["color"])
            next_frontiers[winner].add(pos)

            if collect_trace and len(idxs) > 1:
                max_holes = max(regions[idx]["hole_count"] for idx in idxs)
                hole_tied = [idx for idx in idxs if regions[idx]["hole_count"] == max_holes]
                reason = "fallback"
                if len(hole_tied) == 1:
                    reason = "holes"
                    trace["has_hole_priority"] = True
                else:
                    max_area = max(regions[idx]["area"] for idx in hole_tied)
                    area_tied = [idx for idx in hole_tied if regions[idx]["area"] == max_area]
                    if len(area_tied) == 1:
                        reason = "area"
                        trace["has_area_priority"] = True
                trace["contests"].append(
                    {
                        "cell": pos,
                        "layer": layer,
                        "proposers": idxs,
                        "winner": winner,
                        "reason": reason,
                    }
                )

        frontiers = next_frontiers

    return out, regions, trace


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each colored region outward by as many layers as the number of black single-pixel holes inside it.

    Input structure:
    - Black (0) background fills the canvas.
    - Several disconnected monochrome regions of distinct colors sit in the background.
    - Some regions contain fully enclosed single black pixels; these are the region's holes.
    - Regions are already separated in the input, and holes stay black throughout the task.

    Rule:
    - For each colored 4-connected region, count how many enclosed single black holes it contains.
    - Let that count be N; perform N rounds of synchronous 4-neighbor dilation from that region into black cells.
    - During a round, if multiple regions try to claim the same black cell, give it to the region with more holes; if hole counts tie, give it to the region with larger original area.
    - Do not fill the original holes themselves.

    Key insight:
    - The hidden control signal is topological rather than geometric: each region's number of internal holes determines its dilation depth, and collisions are resolved by comparing those hole counts before comparing region size.
    """
    out, _regions, _trace = _simulate(np.asarray(grid, dtype=int), collect_trace=False)
    return out


def _normalize(cells: set[tuple[int, int]], holes: set[tuple[int, int]]) -> dict:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    norm_cells = {(r - min_r, c - min_c) for r, c in cells}
    norm_holes = {(r - min_r, c - min_c) for r, c in holes}
    height = max(r for r, _ in norm_cells) + 1
    width = max(c for _, c in norm_cells) + 1
    return {
        "cells": norm_cells,
        "holes": norm_holes,
        "h": height,
        "w": width,
        "area": len(norm_cells),
    }


def _make_region(hole_count: int, size_bias: int = 0, forbid_sides: tuple[str, ...] = (), base_h: int | None = None) -> dict:
    for _ in range(100):
        min_h = 3 if hole_count == 0 else 5
        min_w = 3 if hole_count == 0 else 5
        height = base_h if base_h is not None else random.randint(min_h, min(8 + size_bias, 9))
        width = random.randint(min_w, min(8 + size_bias, 9))

        candidates = [(r, c) for r in range(1, height - 1, 2) for c in range(1, width - 1, 2)]
        if len(candidates) < hole_count:
            continue

        holes = set(random.sample(candidates, hole_count))
        cells = {(r, c) for r in range(height) for c in range(width)} - holes

        sides = [side for side in ("top", "bottom", "left", "right") if side not in forbid_sides]
        max_protrusions = min(3, len(sides))
        protrusions = random.randint(0, max_protrusions)
        if max_protrusions and random.random() < 0.7:
            protrusions = max(protrusions, 1)
        random.shuffle(sides)

        for side in sides[:protrusions]:
            if side in ("top", "bottom"):
                seg_len = random.randint(1, width)
                start = random.randint(0, width - seg_len)
                row = -1 if side == "top" else height
                for c in range(start, start + seg_len):
                    cells.add((row, c))
            else:
                seg_len = random.randint(1, height)
                start = random.randint(0, height - seg_len)
                col = -1 if side == "left" else width
                for r in range(start, start + seg_len):
                    cells.add((r, col))

        return _normalize(cells, holes)

    raise RuntimeError("failed to build region")


def _translate(cells: set[tuple[int, int]], top: int, left: int) -> set[tuple[int, int]]:
    return {(r + top, c + left) for r, c in cells}


def _min_manhattan(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> int:
    best = 10**9
    for r1, c1 in a:
        for r2, c2 in b:
            dist = abs(r1 - r2) + abs(c1 - c2)
            if dist < best:
                best = dist
    return best


def _can_place(shape: dict, top: int, left: int, height: int, width: int, placed: list[dict], avoid_conflicts: bool) -> bool:
    if top < 0 or left < 0 or top + shape["h"] > height or left + shape["w"] > width:
        return False

    abs_cells = _translate(shape["cells"], top, left)
    for other in placed:
        gap = _min_manhattan(abs_cells, other["abs_cells"])
        if gap <= 2:
            return False
        if avoid_conflicts and gap <= shape["hole_count"] + other["hole_count"] + 1:
            return False
    return True


def _draw_shape(grid: np.ndarray, shape: dict, color: int, top: int, left: int, placed: list[dict]) -> None:
    abs_cells = _translate(shape["cells"], top, left)
    for r, c in abs_cells:
        grid[r, c] = color
    placed.append(
        {
            "abs_cells": abs_cells,
            "hole_count": shape["hole_count"],
            "area": shape["area"],
        }
    )


def _place_random_shape(grid: np.ndarray, shape: dict, color: int, placed: list[dict], avoid_conflicts: bool) -> bool:
    h, w = grid.shape
    top_low = 1
    top_high = h - shape["h"] - 1
    left_low = 1
    left_high = w - shape["w"] - 1
    if top_high < top_low or left_high < left_low:
        return False

    for _ in range(PLACEMENT_TRIES):
        top = random.randint(top_low, top_high)
        left = random.randint(left_low, left_high)
        if _can_place(shape, top, left, h, w, placed, avoid_conflicts=avoid_conflicts):
            _draw_shape(grid, shape, color, top, left, placed)
            return True
    return False


def _scene_summary(grid: np.ndarray) -> dict:
    out, regions, trace = _simulate(grid, collect_trace=True)
    return {
        "output": out,
        "regions": regions,
        "contest_count": len(trace["contests"]),
        "has_hole_priority": trace["has_hole_priority"],
        "has_area_priority": trace["has_area_priority"],
        "zero_hole_regions": sum(1 for region in regions if region["hole_count"] == 0),
    }


def _build_conflict_scene(kind: str) -> np.ndarray | None:
    pair_layer = random.choice((2, 3))
    if kind == "hole_priority":
        low_holes = pair_layer
        high_holes = random.randint(pair_layer + 1, 4)
        pair_holes = [low_holes, high_holes]
        random.shuffle(pair_holes)
        left_holes, right_holes = pair_holes
        left_shape = _make_region(left_holes, size_bias=0, forbid_sides=("right",), base_h=random.randint(5, 7))
        right_shape = _make_region(right_holes, size_bias=1, forbid_sides=("left",), base_h=left_shape["h"] if random.random() < 0.5 else random.randint(5, 7))
    else:
        shared_holes = random.randint(pair_layer, 4)
        pair_h = random.randint(5, 7)
        left_shape = _make_region(shared_holes, size_bias=0, forbid_sides=("right",), base_h=pair_h)
        right_shape = _make_region(shared_holes, size_bias=2, forbid_sides=("left",), base_h=pair_h)
        if left_shape["area"] == right_shape["area"]:
            return None

    gap = 2 * pair_layer - 1
    min_h = max(left_shape["h"], right_shape["h"]) + 8
    min_w = left_shape["w"] + right_shape["w"] + gap + 8
    if min_h > 30 or min_w > 30:
        return None

    height = random.randint(max(18, min_h), 30)
    width = random.randint(max(20, min_w), 30)
    grid = np.zeros((height, width), dtype=int)
    placed: list[dict] = []

    region_count = 3 + random.randint(0, 2)
    colors = random.sample(COLORS, region_count)

    pair_top = random.randint(1, height - max(left_shape["h"], right_shape["h"]) - 2)
    max_left = width - right_shape["w"] - left_shape["w"] - gap - 2
    if max_left < 1:
        return None
    left_left = random.randint(1, max_left)
    right_left = left_left + left_shape["w"] + gap

    left_shape["hole_count"] = len(left_shape["holes"])
    right_shape["hole_count"] = len(right_shape["holes"])
    if not _can_place(left_shape, pair_top, left_left, height, width, placed, avoid_conflicts=False):
        return None
    _draw_shape(grid, left_shape, colors[0], pair_top, left_left, placed)
    if not _can_place(right_shape, pair_top, right_left, height, width, placed, avoid_conflicts=False):
        return None
    _draw_shape(grid, right_shape, colors[1], pair_top, right_left, placed)

    zero_shape = _make_region(0, size_bias=random.randint(0, 1))
    zero_shape["hole_count"] = 0
    if not _place_random_shape(grid, zero_shape, colors[2], placed, avoid_conflicts=True):
        return None

    for color in colors[3:]:
        holes = random.randint(0, 4)
        extra_shape = _make_region(holes, size_bias=random.randint(0, 2))
        extra_shape["hole_count"] = holes
        if not _place_random_shape(grid, extra_shape, color, placed, avoid_conflicts=True):
            return None

    summary = _scene_summary(grid)
    if len(summary["regions"]) != region_count or summary["zero_hole_regions"] == 0:
        return None
    if kind == "hole_priority" and not summary["has_hole_priority"]:
        return None
    if kind == "area_priority" and not summary["has_area_priority"]:
        return None
    return grid


def _build_independent_scene() -> np.ndarray | None:
    height = random.randint(20, 30)
    width = random.randint(20, 30)
    region_count = random.randint(3, 5)
    colors = random.sample(COLORS, region_count)
    grid = np.zeros((height, width), dtype=int)
    placed: list[dict] = []

    zero_shape = _make_region(0, size_bias=random.randint(0, 1))
    zero_shape["hole_count"] = 0
    if not _place_random_shape(grid, zero_shape, colors[0], placed, avoid_conflicts=True):
        return None

    hole_options = [1, 2, 3, 4]
    random.shuffle(hole_options)
    for color in colors[1:]:
        holes = hole_options.pop() if hole_options and random.random() < 0.6 else random.randint(0, 4)
        shape = _make_region(holes, size_bias=random.randint(0, 2))
        shape["hole_count"] = holes
        if not _place_random_shape(grid, shape, color, placed, avoid_conflicts=True):
            return None

    summary = _scene_summary(grid)
    if summary["contest_count"] != 0 or summary["zero_hole_regions"] == 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    scene_types = ("independent", "hole_priority", "area_priority")
    scene_type = random.choice(scene_types)
    for _ in range(GEN_RETRIES):
        if scene_type == "independent":
            grid = _build_independent_scene()
        elif scene_type == "hole_priority":
            grid = _build_conflict_scene("hole_priority")
        else:
            grid = _build_conflict_scene("area_priority")

        if grid is None:
            continue

        summary = _scene_summary(grid)
        if any(region["hole_count"] > 4 for region in summary["regions"]):
            continue
        if scene_type == "hole_priority" and not summary["has_hole_priority"]:
            continue
        if scene_type == "area_priority" and not summary["has_area_priority"]:
            continue
        if scene_type == "independent" and summary["contest_count"] != 0:
            continue
        if summary["zero_hole_regions"] == 0:
            continue

        return grid

    raise RuntimeError("failed to generate a valid ARC input")
