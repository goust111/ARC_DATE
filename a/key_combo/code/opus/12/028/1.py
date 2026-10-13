from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


BACKGROUND = 0
STAIR = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _extract_anchors(grid: np.ndarray) -> list[tuple[int, int, int]]:
    coords = np.argwhere((grid != BACKGROUND) & (grid != STAIR))
    anchors = [(int(r), int(c), int(grid[r, c])) for r, c in coords]
    anchors.sort()
    return anchors


def _simulate_staircases(grid: np.ndarray):
    g = np.asarray(grid, dtype=int).copy()
    H, W = g.shape
    anchors = _extract_anchors(g)
    owner = np.full((H, W), -1, dtype=int)
    for idx, (r, c, _color) in enumerate(anchors):
        owner[r, c] = idx

    arms = []
    for idx, (r, c, _color) in enumerate(anchors):
        arms.append(
            {
                "anchor_idx": idx,
                "r": r,
                "c": c,
                "dirs": ((-1, 0), (0, 1)),
                "dir_index": 0,
                "steps_in_segment": 0,
                "live": True,
            }
        )
        arms.append(
            {
                "anchor_idx": idx,
                "r": r,
                "c": c,
                "dirs": ((1, 0), (0, -1)),
                "dir_index": 0,
                "steps_in_segment": 0,
                "live": True,
            }
        )

    collision_count = 0
    for _ in range(H * W * 2):
        proposals: dict[tuple[int, int], list[int]] = defaultdict(list)
        any_live = False
        for arm_id, arm in enumerate(arms):
            if not arm["live"]:
                continue
            any_live = True
            dr, dc = arm["dirs"][arm["dir_index"]]
            nr = arm["r"] + dr
            nc = arm["c"] + dc
            if not (0 <= nr < H and 0 <= nc < W):
                arm["live"] = False
                continue

            cell = int(g[nr, nc])
            cell_owner = int(owner[nr, nc])
            if cell == STAIR and cell_owner != arm["anchor_idx"]:
                arm["live"] = False
                collision_count += 1
                continue
            if cell not in (BACKGROUND, STAIR) and cell_owner != arm["anchor_idx"]:
                arm["live"] = False
                collision_count += 1
                continue
            proposals[(nr, nc)].append(arm_id)

        if not any_live or not proposals:
            break

        for (nr, nc), arm_ids in proposals.items():
            proposer_owners = {arms[arm_id]["anchor_idx"] for arm_id in arm_ids}
            if len(proposer_owners) > 1:
                collision_count += len(arm_ids)
                for arm_id in arm_ids:
                    arms[arm_id]["live"] = False
                continue

            anchor_idx = arms[arm_ids[0]]["anchor_idx"]
            if g[nr, nc] == BACKGROUND:
                g[nr, nc] = STAIR
                owner[nr, nc] = anchor_idx

            for arm_id in arm_ids:
                arm = arms[arm_id]
                arm["r"] = nr
                arm["c"] = nc
                arm["steps_in_segment"] += 1
                if arm["steps_in_segment"] == 2:
                    arm["steps_in_segment"] = 0
                    arm["dir_index"] = 1 - arm["dir_index"]

    for r, c, color in anchors:
        g[r, c] = color

    return g, owner, anchors, collision_count


def _enclosed_black_regions(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    H, W = grid.shape
    seen = np.zeros((H, W), dtype=bool)
    regions: list[list[tuple[int, int]]] = []

    for r in range(H):
        for c in range(W):
            if grid[r, c] != BACKGROUND or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            region = []
            touches_border = False
            while stack:
                cr, cc = stack.pop()
                region.append((cr, cc))
                if cr in (0, H - 1) or cc in (0, W - 1):
                    touches_border = True
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and grid[nr, nc] == BACKGROUND:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            if not touches_border:
                regions.append(region)

    return regions


def _region_boundary_counts(
    region: list[tuple[int, int]], stair_grid: np.ndarray, owner: np.ndarray
) -> Counter[int]:
    H, W = stair_grid.shape
    boundary_pixels = set()
    for r, c in region:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < H and 0 <= nc < W and stair_grid[nr, nc] == STAIR:
                boundary_pixels.add((nr, nc))
    return Counter(int(owner[r, c]) for r, c in boundary_pixels)


def _winner_anchor_idx(counts: Counter[int], anchors: list[tuple[int, int, int]]) -> int:
    return min(counts, key=lambda idx: (-counts[idx], anchors[idx][0], anchors[idx][1]))


def _analyze(grid: np.ndarray):
    stair_grid, owner, anchors, collision_count = _simulate_staircases(grid)
    out = stair_grid.copy()
    region_infos = []
    for region in _enclosed_black_regions(stair_grid):
        counts = _region_boundary_counts(region, stair_grid, owner)
        winner_idx = _winner_anchor_idx(counts, anchors)
        fill_color = anchors[winner_idx][2]
        for r, c in region:
            out[r, c] = fill_color

        top_count = max(counts.values())
        top_owners = sorted(idx for idx, count in counts.items() if count == top_count)
        region_infos.append(
            {
                "cells": region,
                "size": len(region),
                "boundary_counts": counts,
                "winner_idx": winner_idx,
                "top_owners": top_owners,
            }
        )

    return {
        "stair_grid": stair_grid,
        "owner": owner,
        "anchors": anchors,
        "collision_count": collision_count,
        "regions": region_infos,
        "output": out,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend colliding staircase arms from multiple anchors, then fill each enclosed pocket by its dominant boundary owner.

    Input structure:
    - The grid is black except for 2-4 distinct non-black anchor pixels.
    - Each anchor sits away from the border and spawns a pair of opposite gray staircases.
    - The anchors are the only colored control pixels in the input.

    Rule:
    - For every anchor, grow one staircase upward then rightward in 2-pixel segments, and a point-symmetric staircase downward then leftward.
    - Each arm advances pixel by pixel and stops before leaving the grid or entering a gray pixel already owned by another anchor; own staircase pixels may be reused.
    - After all gray staircases stop, find 4-connected black regions that do not touch the outer border.
    - Fill each enclosed region with the color of the anchor that owns the most adjacent gray boundary pixels; ties go to the higher anchor (smaller row, then smaller column for determinism).

    Key insight:
    - The fill color is not readable from the anchors alone: the staircase collision simulation must run first, because the collisions create both the enclosed regions and the boundary-ownership counts that choose each region's color.
    """
    return _analyze(np.asarray(grid, dtype=int))["output"]


def _anchors_respect_spacing(anchors: list[tuple[int, int]]) -> bool:
    for i, (r1, c1) in enumerate(anchors):
        for r2, c2 in anchors[i + 1 :]:
            if abs(r1 - r2) + abs(c1 - c2) < 5:
                return False
    return True


def _choose_anchor_positions(H: int, W: int, n_anchors: int) -> list[tuple[int, int]] | None:
    candidates = [(r, c) for r in range(3, H - 3) for c in range(3, W - 3)]
    if len(candidates) < n_anchors:
        return None

    for _ in range(120):
        anchors = random.sample(candidates, n_anchors)
        if _anchors_respect_spacing(anchors):
            return anchors
    return None


def _tie_region_ok(info, anchors: list[tuple[int, int, int]]) -> bool:
    if info["size"] < 2 or len(info["boundary_counts"]) < 2:
        return False
    top_owners = info["top_owners"]
    if len(top_owners) < 2:
        return False
    top_rows = {anchors[idx][0] for idx in top_owners}
    return len(top_rows) == len(top_owners)


def _unique_region_ok(info) -> bool:
    return info["size"] >= 2 and len(info["boundary_counts"]) >= 2 and len(info["top_owners"]) == 1


def generate_input() -> np.ndarray:
    mode = random.choice(("tie", "unique"))

    for _ in range(800):
        H = random.randint(14, 18)
        W = random.randint(14, 18)
        n_anchors = random.choices([2, 3, 4], weights=[4, 3, 2])[0]

        anchor_positions = _choose_anchor_positions(H, W, n_anchors)
        if anchor_positions is None:
            continue

        colors = random.sample(PALETTE, n_anchors)
        grid = np.zeros((H, W), dtype=int)
        for (r, c), color in zip(anchor_positions, colors):
            grid[r, c] = color

        analysis = _analyze(grid)
        regions = analysis["regions"]
        if not regions:
            continue
        if analysis["collision_count"] == 0:
            continue

        total_fill = sum(info["size"] for info in regions)
        if total_fill < 2 or total_fill > (H * W) // 3:
            continue
        if np.count_nonzero(analysis["stair_grid"] == STAIR) < 10:
            continue
        if np.array_equal(grid, analysis["output"]):
            continue

        has_unique = any(_unique_region_ok(info) for info in regions)
        has_tie = any(_tie_region_ok(info, analysis["anchors"]) for info in regions)
        if mode == "tie" and not has_tie:
            continue
        if mode == "unique" and not has_unique:
            continue

        if not any(len(info["boundary_counts"]) >= 2 for info in regions):
            continue

        return grid

    raise RuntimeError("failed to generate a valid staircase-collision puzzle instance")
