from __future__ import annotations

import random

import numpy as np


BG = 8
ARROW = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 9]
DIR_VECTORS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

ARROW_VARIANTS = {
    "right": [
        frozenset({(0, 2), (1, 0), (1, 1), (1, 2)}),
        frozenset({(0, 0), (0, 1), (0, 2), (1, 2)}),
    ],
    "left": [
        frozenset({(0, 0), (1, 0), (1, 1), (1, 2)}),
        frozenset({(0, 0), (0, 1), (0, 2), (1, 0)}),
    ],
    "up": [
        frozenset({(0, 0), (0, 1), (1, 1), (2, 1)}),
        frozenset({(0, 0), (0, 1), (1, 0), (2, 0)}),
    ],
    "down": [
        frozenset({(0, 1), (1, 1), (2, 0), (2, 1)}),
        frozenset({(0, 0), (1, 0), (2, 0), (2, 1)}),
    ],
}
TEMPLATE_TO_DIRECTION = {
    variant: direction
    for direction, variants in ARROW_VARIANTS.items()
    for variant in variants
}

GEN_ATTEMPTS = 500
SPEC_ATTEMPTS = 80


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(points: list[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    min_r = min(r for r, _ in points)
    min_c = min(c for _, c in points)
    return frozenset((r - min_r, c - min_c) for r, c in points)


def _detect_direction(block: np.ndarray) -> str:
    black_points = [(int(r), int(c)) for r, c in np.argwhere(block == ARROW)]
    if len(black_points) != 4:
        raise ValueError("each region must contain exactly one 4-pixel black arrow")
    shape = _normalize(black_points)
    if shape not in TEMPLATE_TO_DIRECTION:
        raise ValueError("unrecognized arrow shape")
    return TEMPLATE_TO_DIRECTION[shape]


def _build_region(spec: dict) -> np.ndarray:
    arr = np.full((spec["h"], spec["w"]), int(spec["color"]), dtype=int)
    ar, ac = spec["arrow_pos"]
    for rr, cc in spec["arrow_cells"]:
        arr[ar + rr, ac + cc] = ARROW
    return arr


def _shifted_box(
    top: int, left: int, height: int, width: int, direction: str
) -> tuple[int, int, int, int]:
    dr, dc = DIR_VECTORS[direction]
    nr = top + dr * height
    nc = left + dc * height
    return nr, nc, nr + height - 1, nc + width - 1


def _clip_box(
    box: tuple[int, int, int, int], grid_h: int, grid_w: int
) -> tuple[int, int, int, int] | None:
    r0 = max(0, int(box[0]))
    c0 = max(0, int(box[1]))
    r1 = min(grid_h - 1, int(box[2]))
    c1 = min(grid_w - 1, int(box[3]))
    if r0 > r1 or c0 > c1:
        return None
    return r0, c0, r1, c1


def _boxes_intersect(
    a: tuple[int, int, int, int] | None, b: tuple[int, int, int, int] | None
) -> bool:
    if a is None or b is None:
        return False
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _box_area(box: tuple[int, int, int, int] | None) -> int:
    if box is None:
        return 0
    return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)


def _intersection_area(
    a: tuple[int, int, int, int] | None, b: tuple[int, int, int, int] | None
) -> int:
    if not _boxes_intersect(a, b):
        return 0
    assert a is not None and b is not None
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each colored rectangle by its own height in the direction encoded by its black arrow.
    """
    g = np.asarray(grid, dtype=int)
    out = np.full(g.shape, BG, dtype=int)
    objects = []

    for pts in _components(g != BG):
        r0, c0, r1, c1 = _bbox(pts)
        block = g[r0 : r1 + 1, c0 : c1 + 1]
        if np.any(block == BG):
            raise ValueError("each foreground object must fill its bounding box")
        direction = _detect_direction(block)
        height, width = map(int, block.shape)
        target_r0, target_c0, _, _ = _shifted_box(r0, c0, height, width, direction)
        objects.append(
            {
                "block": block.copy(),
                "target_r0": int(target_r0),
                "target_c0": int(target_c0),
                "source_r0": int(r0),
                "source_c0": int(c0),
            }
        )

    # Smaller final row index is on top, so draw from larger rows to smaller rows.
    objects.sort(
        key=lambda obj: (
            int(obj["target_r0"]),
            int(obj["target_c0"]),
            int(obj["source_r0"]),
            int(obj["source_c0"]),
        ),
        reverse=True,
    )

    for obj in objects:
        block = obj["block"]
        h, w = map(int, block.shape)
        for rr in range(h):
            gr = int(obj["target_r0"]) + rr
            if not (0 <= gr < g.shape[0]):
                continue
            for cc in range(w):
                gc = int(obj["target_c0"]) + cc
                if 0 <= gc < g.shape[1]:
                    out[gr, gc] = int(block[rr, cc])

    return out


def _expand_box(box: tuple[int, int, int, int], gap: int) -> tuple[int, int, int, int]:
    return box[0] - gap, box[1] - gap, box[2] + gap, box[3] + gap


def _input_box_clear(
    box: tuple[int, int, int, int], existing: list[tuple[int, int, int, int]], gap: int = 1
) -> bool:
    padded = _expand_box(box, gap)
    return not any(_boxes_intersect(padded, other) for other in existing)


def _sample_spec(color: int, grid_h: int, grid_w: int) -> dict | None:
    direction = random.choice(list(DIR_VECTORS))
    if direction in ("left", "right"):
        max_h = min(6, grid_h - 1)
        max_w = min(8, grid_w - 1)
        if max_h < 2 or max_w < 3:
            return None
        h = random.randint(2, max_h)
        w = random.randint(3, max_w)
    else:
        max_h = min(8, grid_h - 1)
        max_w = min(6, grid_w - 1)
        if max_h < 3 or max_w < 2:
            return None
        h = random.randint(3, max_h)
        w = random.randint(2, max_w)

    arrow_cells = random.choice(list(ARROW_VARIANTS[direction]))
    arrow_h = 1 + max(r for r, _ in arrow_cells)
    arrow_w = 1 + max(c for _, c in arrow_cells)
    arrow_r = random.randint(0, h - arrow_h)
    arrow_c = random.randint(0, w - arrow_w)
    return {
        "color": int(color),
        "direction": direction,
        "h": int(h),
        "w": int(w),
        "arrow_cells": arrow_cells,
        "arrow_pos": (int(arrow_r), int(arrow_c)),
    }


def _candidate_positions(
    spec: dict, grid_h: int, grid_w: int, existing: list[tuple[int, int, int, int]]
) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    h = int(spec["h"])
    w = int(spec["w"])
    for r0 in range(grid_h - h + 1):
        for c0 in range(grid_w - w + 1):
            input_box = (r0, c0, r0 + h - 1, c0 + w - 1)
            if not _input_box_clear(input_box, existing, gap=1):
                continue
            final_box = _shifted_box(r0, c0, h, w, spec["direction"])
            if _clip_box(final_box, grid_h, grid_w) is None:
                continue
            positions.append((r0, c0))
    return positions


def _placed_object(spec: dict, pos: tuple[int, int], grid_h: int, grid_w: int) -> dict:
    r0, c0 = pos
    input_box = (r0, c0, r0 + spec["h"] - 1, c0 + spec["w"] - 1)
    final_box = _shifted_box(r0, c0, spec["h"], spec["w"], spec["direction"])
    visible_final_box = _clip_box(final_box, grid_h, grid_w)
    return {
        "spec": spec,
        "pos": (int(r0), int(c0)),
        "input_box": input_box,
        "final_box": final_box,
        "visible_final_box": visible_final_box,
        "clipped": visible_final_box != final_box,
    }


def _choose_object(
    color: int,
    grid_h: int,
    grid_w: int,
    existing: list[tuple[int, int, int, int]],
    require_clip: bool = False,
    prefer_clip: bool = False,
    overlap_with: tuple[int, int, int, int] | None = None,
) -> dict | None:
    for _ in range(SPEC_ATTEMPTS):
        spec = _sample_spec(color, grid_h, grid_w)
        if spec is None:
            continue
        positions = _candidate_positions(spec, grid_h, grid_w, existing)
        if not positions:
            continue

        if overlap_with is not None:
            filtered = []
            ref_area = _box_area(overlap_with)
            for pos in positions:
                placed = _placed_object(spec, pos, grid_h, grid_w)
                vis = placed["visible_final_box"]
                area = _intersection_area(vis, overlap_with)
                if area == 0:
                    continue
                max_area = max(1, min(_box_area(vis), ref_area) // 2)
                if area > max_area:
                    continue
                if vis is not None and overlap_with is not None and vis[0] == overlap_with[0]:
                    continue
                filtered.append(pos)
            positions = filtered
            if not positions:
                continue

        clipped_positions = []
        unclipped_positions = []
        for pos in positions:
            placed = _placed_object(spec, pos, grid_h, grid_w)
            if placed["clipped"]:
                clipped_positions.append(pos)
            else:
                unclipped_positions.append(pos)

        if require_clip:
            if not clipped_positions:
                continue
            chosen = random.choice(clipped_positions)
        elif prefer_clip and clipped_positions and random.random() < 0.75:
            chosen = random.choice(clipped_positions)
        elif unclipped_positions and random.random() < 0.65:
            chosen = random.choice(unclipped_positions)
        else:
            chosen = random.choice(positions)

        return _placed_object(spec, chosen, grid_h, grid_w)
    return None


def _has_output_overlap(placed: list[dict]) -> bool:
    for i in range(len(placed)):
        for j in range(i + 1, len(placed)):
            if _boxes_intersect(placed[i]["visible_final_box"], placed[j]["visible_final_box"]):
                return True
    return False


def _build_input_grid(placed: list[dict], grid_h: int, grid_w: int) -> np.ndarray:
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    for obj in placed:
        region = _build_region(obj["spec"])
        r0, c0 = obj["pos"]
        h, w = map(int, region.shape)
        grid[r0 : r0 + h, c0 : c0 + w] = region
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid_h = random.randint(14, 24)
        grid_w = random.randint(14, 24)
        area = grid_h * grid_w
        max_objects = min(5, max(2, 2 + area // 140))
        obj_count = random.randint(2, max_objects)
        colors = random.sample(PALETTE, obj_count)
        need_clip = random.random() < 0.4
        need_overlap = obj_count >= 2 and random.random() < 0.4

        placed: list[dict] = []
        existing: list[tuple[int, int, int, int]] = []
        clip_satisfied = False

        first = _choose_object(
            colors[0],
            grid_h,
            grid_w,
            existing,
            prefer_clip=need_clip,
        )
        if first is None:
            continue
        placed.append(first)
        existing.append(first["input_box"])
        clip_satisfied = clip_satisfied or bool(first["clipped"])

        next_index = 1
        if need_overlap:
            second = _choose_object(
                colors[1],
                grid_h,
                grid_w,
                existing,
                require_clip=False,
                prefer_clip=need_clip and not clip_satisfied,
                overlap_with=first["visible_final_box"],
            )
            if second is None:
                continue
            placed.append(second)
            existing.append(second["input_box"])
            clip_satisfied = clip_satisfied or bool(second["clipped"])
            next_index = 2

        failed = False
        for idx in range(next_index, obj_count):
            remaining = obj_count - idx
            require_clip = need_clip and not clip_satisfied and remaining == 1
            prefer_clip = need_clip and not clip_satisfied and not require_clip
            obj = _choose_object(
                colors[idx],
                grid_h,
                grid_w,
                existing,
                require_clip=require_clip,
                prefer_clip=prefer_clip,
            )
            if obj is None:
                failed = True
                break
            placed.append(obj)
            existing.append(obj["input_box"])
            clip_satisfied = clip_satisfied or bool(obj["clipped"])
        if failed:
            continue

        if need_clip and not clip_satisfied:
            continue
        if need_overlap and not _has_output_overlap(placed):
            continue

        grid = _build_input_grid(placed, grid_h, grid_w)
        comps = _components(grid != BG)
        if len(comps) != obj_count:
            continue
        if any(np.any(grid[r0 : r1 + 1, c0 : c1 + 1] == BG) for r0, c0, r1, c1 in map(_bbox, comps)):
            continue
        return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
