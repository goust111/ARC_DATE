from __future__ import annotations

import itertools
import math
from collections import deque

import numpy as np


BG = 0
MARKER_COLORS = (1, 2)
RECT_COLORS = (3, 4, 5, 6, 7, 8, 9)
RECT_DIMS = (
    (2, 3),
    (2, 4),
    (2, 5),
    (3, 4),
    (3, 5),
    (4, 3),
    (4, 5),
)
GROUP_STEPS = (2, 3, 4)
GEN_MAX_ATTEMPTS = 600


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _component_coords(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        if _in_bounds(grid.shape, r, c):
            grid[r, c] = color


def _bbox_cells(bbox: tuple[int, int, int, int], shape: tuple[int, int] | None = None) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    out = set()
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if shape is None or _in_bounds(shape, r, c):
                out.add((r, c))
    return out


def _expand_cells(cells: set[tuple[int, int]], shape: tuple[int, int], radius: int) -> set[tuple[int, int]]:
    if radius <= 0:
        return set(cells)
    out = set(cells)
    h, w = shape
    for r, c in list(cells):
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _marker_candidates(grid: np.ndarray) -> list[dict]:
    candidates: list[dict] = []
    for color in MARKER_COLORS:
        coords = [tuple(map(int, p)) for p in np.argwhere(grid == color)]
        for combo in itertools.combinations(coords, 3):
            pts = sorted(combo)
            same_row = pts[0][0] == pts[1][0] == pts[2][0]
            same_col = pts[0][1] == pts[1][1] == pts[2][1]
            if same_row == same_col:
                continue
            dr1 = pts[1][0] - pts[0][0]
            dc1 = pts[1][1] - pts[0][1]
            dr2 = pts[2][0] - pts[1][0]
            dc2 = pts[2][1] - pts[1][1]
            if (dr1, dc1) != (dr2, dc2) or (dr1 == 0 and dc1 == 0):
                continue
            path: list[tuple[int, int]] = []
            rr = pts[2][0] + dr1
            cc = pts[2][1] + dc1
            while _in_bounds(grid.shape, rr, cc):
                path.append((rr, cc))
                rr += dr1
                cc += dc1
            candidates.append(
                {
                    "color": color,
                    "points": pts,
                    "cells": frozenset(pts),
                    "step": (dr1, dc1),
                    "path": path,
                    "length": len(path),
                    "orientation": "h" if dr1 == 0 else "v",
                }
            )
    return candidates


def _extract_groups(grid: np.ndarray) -> list[dict]:
    marker_cells = {
        tuple(map(int, p))
        for p in np.argwhere(np.isin(grid, np.array(MARKER_COLORS, dtype=int)))
    }
    candidates = _marker_candidates(grid)
    cell_to_cands: dict[tuple[int, int], list[int]] = {cell: [] for cell in marker_cells}
    for idx, cand in enumerate(candidates):
        for cell in cand["cells"]:
            if cell in cell_to_cands:
                cell_to_cands[cell].append(idx)
    if not marker_cells or any(not idxs for idxs in cell_to_cands.values()):
        raise ValueError("marker triples are not uniquely recoverable")

    solutions: list[list[int]] = []

    def rec(uncovered: set[tuple[int, int]], chosen: list[int]) -> None:
        if len(solutions) > 1:
            return
        if not uncovered:
            if 2 <= len(chosen) <= 3:
                solutions.append(chosen.copy())
            return
        if len(chosen) >= 3:
            return
        pivot = min(
            uncovered,
            key=lambda cell: len([i for i in cell_to_cands[cell] if candidates[i]["cells"] <= uncovered]),
        )
        options = [i for i in cell_to_cands[pivot] if candidates[i]["cells"] <= uncovered]
        for idx in options:
            rec(uncovered - set(candidates[idx]["cells"]), chosen + [idx])

    rec(set(marker_cells), [])
    if len(solutions) != 1:
        raise ValueError("marker triples are ambiguous")
    groups = [dict(candidates[idx]) for idx in solutions[0]]
    groups.sort(key=lambda g: g["points"][0])
    return groups


def _notched_rect_patch(color: int, h: int, w: int) -> np.ndarray:
    patch = np.full((h, w), color, dtype=int)
    patch[0, 0] = BG
    return patch


def _extract_rectangles(grid: np.ndarray) -> list[dict]:
    rects: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in (BG, *MARKER_COLORS)):
        for comp in _component_coords(grid == color):
            r0, c0, r1, c1 = _bbox_from_coords(comp)
            patch = np.array(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
            h, w = patch.shape
            if not (2 <= h <= 4 and 3 <= w <= 5 and h != w):
                raise ValueError("unexpected rectangle bbox")
            if set(map(int, np.unique(patch))) - {BG, color}:
                raise ValueError("rectangle contains unexpected colors")
            corner_positions = ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1))
            bg_corners = [(rr, cc) for rr, cc in corner_positions if int(patch[rr, cc]) == BG]
            if len(bg_corners) != 1:
                raise ValueError("rectangle anchor notch missing")
            expected = np.full((h, w), color, dtype=int)
            expected[bg_corners[0]] = BG
            if not np.array_equal(patch, expected):
                raise ValueError("rectangle is not a single-notch box")
            rects.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "patch": patch,
                }
            )
    rects.sort(key=lambda r: r["bbox"])
    return rects


def _governing_group(
    bbox: tuple[int, int, int, int],
    groups: list[dict],
) -> tuple[int, int, int]:
    r0, c0, r1, c1 = bbox
    center2 = (r0 + r1, c0 + c1)
    dists = []
    for idx, group in enumerate(groups):
        best = min(abs(center2[0] - 2 * r) + abs(center2[1] - 2 * c) for r, c in group["points"])
        dists.append((best, idx))
    dists.sort()
    if len(dists) > 1 and dists[0][0] == dists[1][0]:
        raise ValueError("rectangle-governor tie")
    margin = 999 if len(dists) == 1 else dists[1][0] - dists[0][0]
    return dists[0][1], dists[0][0], margin


def _centered_top_left(
    bbox: tuple[int, int, int, int],
    new_shape: tuple[int, int],
) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    new_h, new_w = new_shape
    nr0 = math.floor(((r0 + r1) - (new_h - 1)) / 2)
    nc0 = math.floor(((c0 + c1) - (new_w - 1)) / 2)
    return nr0, nc0


def _paste_clip(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    for rr in range(h):
        for cc in range(w):
            dr = r0 + rr
            dc = c0 + cc
            if _in_bounds(dst.shape, dr, dc):
                dst[dr, dc] = int(patch[rr, cc])


def _analyze_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")
    groups = _extract_groups(g)
    rects = _extract_rectangles(g)
    if not (2 <= len(groups) <= 3 and 2 <= len(rects) <= 3):
        raise ValueError("unexpected group or rectangle count")
    for rect in rects:
        group_idx, dist, margin = _governing_group(rect["bbox"], groups)
        rect["group_index"] = group_idx
        rect["group_distance"] = dist
        rect["group_margin"] = margin
        clockwise = bool(groups[group_idx]["length"] % 2)
        rot_patch = np.rot90(rect["patch"], k=3 if clockwise else 1)
        nr0, nc0 = _centered_top_left(rect["bbox"], rot_patch.shape)
        rect["clockwise"] = clockwise
        rect["rot_patch"] = rot_patch
        rect["out_bbox"] = (nr0, nc0, nr0 + rot_patch.shape[0] - 1, nc0 + rot_patch.shape[1] - 1)
        rect["out_clipped"] = not (
            0 <= nr0
            and 0 <= nc0
            and nr0 + rot_patch.shape[0] <= g.shape[0]
            and nc0 + rot_patch.shape[1] <= g.shape[1]
        )
    return {"groups": groups, "rects": rects}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each marker ray to the border, then rotate each anchored rectangle according to the ray-length parity of its nearest marker triple.

    Input structure:
    - Black background with 2-3 marker triples and 2-3 rectangular box objects.
    - Each marker triple uses color 1 or 2, lies on one horizontal or vertical arithmetic line, and consists of exactly three equally spaced pixels.
    - Each rectangle is a non-square monochrome box of size 2-4 by 3-5 with one black corner notch; rectangle colors avoid 0, 1, and 2.

    Rule:
    - Order each triple in row-major order, take the step from the first point to the second, and continue that ray from the third point until the next step would leave the grid.
    - Paint every extrapolated ray cell with the triple color and keep the original markers.
    - For each rectangle, find the governing triple by the unique minimum Manhattan distance from the rectangle center to the nearest marker of each triple.
    - If that triple's extrapolation length is odd, rotate the whole notched rectangle 90 degrees clockwise; if even, rotate it 90 degrees counterclockwise; place the rotated rectangle back around the same center, clipping anything outside the grid.

    Key insight:
    - The geometric quantity that matters is not the raw ray length but only its parity, which becomes a binary control signal for the downstream rectangle rotation.
    """
    g = np.asarray(grid, dtype=int)
    scene = _analyze_scene(g)
    out = np.array(g, copy=True)

    for rect in scene["rects"]:
        r0, c0, r1, c1 = rect["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = BG

    for group in scene["groups"]:
        _paint_cells(out, group["path"], group["color"])

    for rect in scene["rects"]:
        r0, c0, _r1, _c1 = rect["out_bbox"]
        _paste_clip(out, rect["rot_patch"], r0, c0)

    return out


def _enumerate_group_slots(h: int, w: int) -> list[dict]:
    slots: list[dict] = []
    for step in GROUP_STEPS:
        for r in range(h):
            for c0 in range(w - 2 * step):
                pts = [(r, c0), (r, c0 + step), (r, c0 + 2 * step)]
                path = []
                cc = c0 + 3 * step
                while cc < w:
                    path.append((r, cc))
                    cc += step
                if not path:
                    continue
                occ = set(pts) | set(path)
                slots.append(
                    {
                        "points": pts,
                        "path": path,
                        "length": len(path),
                        "orientation": "h",
                        "occupied": occ,
                    }
                )
        for c in range(w):
            for r0 in range(h - 2 * step):
                pts = [(r0, c), (r0 + step, c), (r0 + 2 * step, c)]
                path = []
                rr = r0 + 3 * step
                while rr < h:
                    path.append((rr, c))
                    rr += step
                if not path:
                    continue
                occ = set(pts) | set(path)
                slots.append(
                    {
                        "points": pts,
                        "path": path,
                        "length": len(path),
                        "orientation": "v",
                        "occupied": occ,
                    }
                )
    return slots


def _pick_groups(h: int, w: int) -> list[dict] | None:
    slots = _enumerate_group_slots(h, w)
    if not slots:
        return None
    for _ in range(160):
        n_groups = 2 if np.random.random() < 0.55 else 3
        required = [0, 1] + ([int(np.random.randint(2))] if n_groups == 3 else [])
        np.random.shuffle(required)
        chosen: list[dict] = []
        blocked: set[tuple[int, int]] = set()
        for parity in required:
            candidates = [
                slot
                for slot in slots
                if slot["length"] % 2 == parity
                and not (slot["occupied"] & blocked)
            ]
            if not candidates:
                chosen = []
                break
            slot = candidates[int(np.random.randint(len(candidates)))]
            chosen.append(slot)
            blocked |= _expand_cells(slot["occupied"], (h, w), 1)
        if not chosen:
            continue

        colors = np.random.choice(np.array(MARKER_COLORS, dtype=int), size=len(chosen), replace=True)
        grid = np.zeros((h, w), dtype=int)
        signature = set()
        for slot, color in zip(chosen, colors):
            for r, c in slot["points"]:
                grid[r, c] = int(color)
            signature.add(frozenset(slot["points"]))
        try:
            groups = _extract_groups(grid)
        except ValueError:
            continue
        if {frozenset(group["points"]) for group in groups} != signature:
            continue
        return groups
    return None


def _rect_candidate_positions(
    shape: tuple[int, int],
    groups: list[dict],
    target_idx: int,
    input_forbidden: set[tuple[int, int]],
    output_forbidden: set[tuple[int, int]],
    clip_mode: str,
) -> list[dict]:
    h, w = shape
    group = groups[target_idx]
    clockwise = bool(group["length"] % 2)
    candidates: list[dict] = []
    for rect_h, rect_w in np.random.permutation(np.array(RECT_DIMS, dtype=int)):
        bbox_area = rect_h * rect_w
        patch = _notched_rect_patch(9, int(rect_h), int(rect_w))
        rot_patch = np.rot90(patch, k=3 if clockwise else 1)
        for r0 in range(h - int(rect_h) + 1):
            for c0 in range(w - int(rect_w) + 1):
                bbox = (r0, c0, r0 + int(rect_h) - 1, c0 + int(rect_w) - 1)
                bbox_cells = _bbox_cells(bbox)
                if bbox_cells & input_forbidden:
                    continue
                try:
                    governor, best, margin = _governing_group(bbox, groups)
                except ValueError:
                    continue
                if governor != target_idx or margin < 2:
                    continue
                nr0, nc0 = _centered_top_left(bbox, rot_patch.shape)
                out_bbox = (nr0, nc0, nr0 + rot_patch.shape[0] - 1, nc0 + rot_patch.shape[1] - 1)
                full_in = (
                    0 <= nr0
                    and 0 <= nc0
                    and nr0 + rot_patch.shape[0] <= h
                    and nc0 + rot_patch.shape[1] <= w
                )
                clipped = not full_in
                if clip_mode == "must" and not clipped:
                    continue
                if clip_mode == "forbid" and clipped:
                    continue
                in_bounds_out = _bbox_cells(out_bbox, (h, w))
                if len(in_bounds_out) < max(3, int(math.ceil(0.6 * bbox_area))):
                    continue
                if in_bounds_out & output_forbidden:
                    continue
                target_mid = group["points"][1]
                anchor_dist = abs((bbox[0] + bbox[2]) - 2 * target_mid[0]) + abs((bbox[1] + bbox[3]) - 2 * target_mid[1])
                candidates.append(
                    {
                        "bbox": bbox,
                        "out_bbox": out_bbox,
                        "patch_shape": (int(rect_h), int(rect_w)),
                        "clipped": clipped,
                        "margin": margin,
                        "best": best,
                        "anchor_dist": anchor_dist,
                    }
                )
    candidates.sort(key=lambda c: (-c["margin"], c["best"], c["anchor_dist"], c["bbox"]))
    return candidates


def _generate_candidate() -> np.ndarray | None:
    h = int(np.random.randint(15, 19))
    w = int(np.random.randint(15, 19))
    groups = _pick_groups(h, w)
    if groups is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    group_occ: set[tuple[int, int]] = set()
    for group in groups:
        for r, c in group["points"]:
            grid[r, c] = int(group["color"])
        group_occ |= set(group["points"])
        group_occ |= set(group["path"])

    odd_targets = [idx for idx, group in enumerate(groups) if group["length"] % 2 == 1]
    even_targets = [idx for idx, group in enumerate(groups) if group["length"] % 2 == 0]
    if not odd_targets or not even_targets:
        return None

    n_rects = 2 if np.random.random() < 0.55 else 3
    targets = [int(np.random.choice(odd_targets)), int(np.random.choice(even_targets))]
    while len(targets) < n_rects:
        targets.append(int(np.random.randint(len(groups))))
    np.random.shuffle(targets)

    clip_rect = int(np.random.randint(n_rects)) if np.random.random() < 0.35 else -1
    rect_colors = np.random.choice(np.array(RECT_COLORS, dtype=int), size=n_rects, replace=False)
    input_forbidden = _expand_cells(group_occ, (h, w), 1)
    output_forbidden = set(group_occ)

    for idx, (target_idx, color) in enumerate(zip(targets, rect_colors)):
        clip_mode = "must" if idx == clip_rect else "any"
        if clip_mode == "must":
            cands = _rect_candidate_positions((h, w), groups, target_idx, input_forbidden, output_forbidden, "must")
            if not cands:
                return None
        else:
            strict = _rect_candidate_positions((h, w), groups, target_idx, input_forbidden, output_forbidden, "forbid")
            loose = _rect_candidate_positions((h, w), groups, target_idx, input_forbidden, output_forbidden, "any")
            cands = strict if strict else loose
            if not cands:
                return None

        pick_pool = cands[: min(12, len(cands))]
        chosen = pick_pool[int(np.random.randint(len(pick_pool)))]
        bbox = chosen["bbox"]
        rect_h = bbox[2] - bbox[0] + 1
        rect_w = bbox[3] - bbox[1] + 1
        patch = _notched_rect_patch(int(color), rect_h, rect_w)
        r0, c0, r1, c1 = bbox
        grid[r0 : r1 + 1, c0 : c1 + 1] = patch
        input_forbidden |= _expand_cells(_bbox_cells(bbox), (h, w), 1)
        output_forbidden |= _bbox_cells(chosen["out_bbox"], (h, w))

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _generate_candidate()
        if grid is None:
            continue
        try:
            scene = _analyze_scene(grid)
        except ValueError:
            continue
        group_parities = {group["length"] % 2 for group in scene["groups"]}
        rect_parities = {scene["groups"][rect["group_index"]]["length"] % 2 for rect in scene["rects"]}
        if group_parities != {0, 1} or rect_parities != {0, 1}:
            continue
        if any(rect["group_margin"] < 2 for rect in scene["rects"]):
            continue
        if any(group["length"] < 1 for group in scene["groups"]):
            continue
        out = main(grid)
        if np.array_equal(out, grid):
            continue
        if np.count_nonzero(out != grid) < 10:
            continue
        if not any(rect["out_clipped"] for rect in scene["rects"]) and np.random.random() < 0.15:
            continue
        return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
