from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
WALL = 5
TRAP_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

GRID_H_RANGE = (18, 28)
GRID_W_RANGE = (18, 28)
RECT_H_RANGE = (5, 10)
RECT_W_RANGE = (5, 10)
RECT_COUNT_CHOICES = (1, 2, 3, 4)
RECT_COUNT_WEIGHTS = (2, 4, 3, 1)
MAX_TRAPS_PER_RECT = 3
GEN_ATTEMPTS = 400
PLACE_ATTEMPTS = 200
GEN_MODES = ("open", "boundary", "wall_block")


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _rect_frame_bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = _bbox(coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return None
    if len(coords) != 2 * (height + width) - 4:
        return None
    for r, c in coords:
        if r not in (r0, r1) and c not in (c0, c1):
            return None
    return r0, c0, r1, c1


def _background_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _find_scene(grid: np.ndarray) -> tuple[int, int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    assert colors, "expected at least one foreground color"

    wall_candidates: list[tuple[int, int, list[tuple[int, int, int, int]]]] = []
    for color in colors:
        comps = _connected_components(g == color)
        bboxes: list[tuple[int, int, int, int]] = []
        ok = True
        total = 0
        for comp in comps:
            bbox = _rect_frame_bbox(comp)
            if bbox is None:
                ok = False
                break
            total += len(comp)
            bboxes.append(bbox)
        if ok and bboxes:
            wall_candidates.append((total, color, bboxes))

    assert wall_candidates, "could not identify wall color"
    _wall_area, wall, rect_bboxes = max(wall_candidates, key=lambda item: item[0])
    rect_bboxes = sorted(rect_bboxes)

    trap_coords: set[tuple[int, int]] = set()
    rects: list[dict] = []
    for bbox in rect_bboxes:
        r0, c0, r1, c1 = bbox
        traps: list[dict] = []
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                color = int(g[r, c])
                if color in (bg, wall):
                    continue
                sides: list[str] = []
                if int(g[r - 1, c]) == wall:
                    sides.append("top")
                if int(g[r + 1, c]) == wall:
                    sides.append("bottom")
                if int(g[r, c - 1]) == wall:
                    sides.append("left")
                if int(g[r, c + 1]) == wall:
                    sides.append("right")
                assert len(sides) == 1, "each trap must encode exactly one wall direction"
                traps.append({"pos": (r, c), "color": color, "side": sides[0]})
                trap_coords.add((r, c))
        assert traps, "each rectangle must contain at least one trap"
        traps.sort(key=lambda trap: trap["pos"])
        rects.append({"bbox": bbox, "traps": traps})

    all_traps = {
        (int(r), int(c))
        for r, c in np.argwhere((g != bg) & (g != wall))
    }
    assert trap_coords == all_traps, "all non-wall foreground pixels must be traps inside rectangles"
    return bg, wall, rects


def _tail_spec(bbox: tuple[int, int, int, int], trap: dict) -> tuple[tuple[int, int], tuple[int, int], int]:
    r0, c0, r1, c1 = bbox
    r, c = trap["pos"]
    side = trap["side"]
    if side == "top":
        return (r0 - 1, c), DIRS[side], c1 - c0 + 1
    if side == "bottom":
        return (r1 + 1, c), DIRS[side], c1 - c0 + 1
    if side == "left":
        return (r, c0 - 1), DIRS[side], r1 - r0 + 1
    return (r, c1 + 1), DIRS[side], r1 - r0 + 1


def _apply_tails(grid: np.ndarray, return_meta: bool = False):
    g = np.asarray(grid, dtype=int)
    bg, _wall, rects = _find_scene(g)
    out = g.copy()
    h, w = out.shape
    metas: list[dict] = []

    for rect in rects:
        bbox = rect["bbox"]
        for trap in rect["traps"]:
            (sr, sc), (dr, dc), length = _tail_spec(bbox, trap)
            cells: list[tuple[int, int]] = []
            stop = "full"
            blocked_by_input = False
            for step in range(length):
                rr = sr + step * dr
                cc = sc + step * dc
                if not (0 <= rr < h and 0 <= cc < w):
                    stop = "boundary"
                    break
                if int(out[rr, cc]) != bg:
                    stop = "pixel"
                    blocked_by_input = int(g[rr, cc]) != bg
                    break
                out[rr, cc] = int(trap["color"])
                cells.append((rr, cc))
            metas.append(
                {
                    "bbox": bbox,
                    "trap": trap,
                    "cells": cells,
                    "length": length,
                    "stop": stop,
                    "blocked_by_input": blocked_by_input,
                }
            )

    if return_meta:
        return out, metas
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw outward tails from rectangular rooms according to the side each trap touches.

    Input structure:
    - Black background fills the grid.
    - One wall color forms one-pixel-thick hollow rectangles.
    - Each rectangle contains one or more single-pixel traps whose color is not the wall color.

    Rule:
    - A trap touching the top, bottom, left, or right inner wall grows a tail in that outward direction.
    - The tail starts one cell outside the opposite wall position of the trap.
    - Tail length equals the full length of the touched wall side.
    - Tail drawing stops early at the grid boundary or just before any existing colored pixel.
    """
    return _apply_tails(np.asarray(grid, dtype=int))


def _boxes_too_close(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _sample_free_rect(
    rng: random.Random,
    height: int,
    width: int,
    existing: list[dict],
) -> tuple[int, int, int, int] | None:
    hmax = min(RECT_H_RANGE[1], height - 2)
    wmax = min(RECT_W_RANGE[1], width - 2)
    if hmax < RECT_H_RANGE[0] or wmax < RECT_W_RANGE[0]:
        return None

    for _ in range(PLACE_ATTEMPTS):
        rect_h = rng.randint(RECT_H_RANGE[0], hmax)
        rect_w = rng.randint(RECT_W_RANGE[0], wmax)
        max_r0 = height - rect_h - 1
        max_c0 = width - rect_w - 1
        if max_r0 < 1 or max_c0 < 1:
            continue
        r0 = rng.randint(1, max_r0)
        c0 = rng.randint(1, max_c0)
        bbox = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
        if any(_boxes_too_close(bbox, other["bbox"], gap=1) for other in existing):
            continue
        return bbox
    return None


def _sample_boundary_rect(
    rng: random.Random,
    height: int,
    width: int,
    existing: list[dict],
) -> dict | None:
    for _ in range(PLACE_ATTEMPTS):
        side = rng.choice(("top", "bottom", "left", "right"))
        rect_h = rng.randint(RECT_H_RANGE[0], min(RECT_H_RANGE[1], height - 2))
        rect_w = rng.randint(RECT_W_RANGE[0], min(RECT_W_RANGE[1], width - 2))
        bbox: tuple[int, int, int, int] | None = None

        if side == "top":
            length = rect_w
            upper = min(height - rect_h - 1, length - 1)
            if upper >= 1:
                r0 = rng.randint(1, upper)
                c0 = rng.randint(1, width - rect_w - 1)
                bbox = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
        elif side == "bottom":
            length = rect_w
            margin = rng.randint(1, min(length - 1, height - rect_h - 1))
            r0 = height - rect_h - margin
            if 1 <= r0 <= height - rect_h - 1:
                c0 = rng.randint(1, width - rect_w - 1)
                bbox = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
        elif side == "left":
            length = rect_h
            upper = min(width - rect_w - 1, length - 1)
            if upper >= 1:
                c0 = rng.randint(1, upper)
                r0 = rng.randint(1, height - rect_h - 1)
                bbox = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
        else:
            length = rect_h
            margin = rng.randint(1, min(length - 1, width - rect_w - 1))
            c0 = width - rect_w - margin
            if 1 <= c0 <= width - rect_w - 1:
                r0 = rng.randint(1, height - rect_h - 1)
                bbox = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)

        if bbox is None:
            continue
        if any(_boxes_too_close(bbox, other["bbox"], gap=1) for other in existing):
            continue
        return {"bbox": bbox, "forced": [{"side": side}], "traps": []}
    return None


def _sample_wall_block_pair(
    rng: random.Random,
    height: int,
    width: int,
) -> list[dict] | None:
    for _ in range(PLACE_ATTEMPTS):
        side = rng.choice(("top", "bottom", "left", "right"))
        src_h = rng.randint(6, min(RECT_H_RANGE[1], height - 2))
        src_w = rng.randint(6, min(RECT_W_RANGE[1], width - 2))
        blk_h = rng.randint(5, min(RECT_H_RANGE[1], height - 2))
        blk_w = rng.randint(5, min(RECT_W_RANGE[1], width - 2))

        if side in ("left", "right"):
            max_gap = min(src_h - 1, width - src_w - blk_w - 2)
        else:
            max_gap = min(src_w - 1, height - src_h - blk_h - 2)
        if max_gap < 1:
            continue
        gap = rng.randint(1, max_gap)

        if side == "right":
            max_src_c0 = width - (src_w + gap + blk_w) - 1
            if max_src_c0 < 1:
                continue
            src_c0 = rng.randint(1, max_src_c0)
            blk_c0 = src_c0 + src_w + gap
            src_r0 = rng.randint(1, height - src_h - 1)
            src_row = rng.randint(src_r0 + 2, src_r0 + src_h - 3)
            blk_r0_min = max(1, src_row - blk_h + 1)
            blk_r0_max = min(src_row, height - blk_h - 1)
            if blk_r0_min > blk_r0_max:
                continue
            blk_r0 = rng.randint(blk_r0_min, blk_r0_max)
            src_bbox = (src_r0, src_c0, src_r0 + src_h - 1, src_c0 + src_w - 1)
            blk_bbox = (blk_r0, blk_c0, blk_r0 + blk_h - 1, blk_c0 + blk_w - 1)
            forced = {"side": "right", "pos": (src_row, src_bbox[3] - 1)}
        elif side == "left":
            min_src_c0 = blk_w + gap + 1
            max_src_c0 = width - src_w - 1
            if min_src_c0 > max_src_c0:
                continue
            src_c0 = rng.randint(min_src_c0, max_src_c0)
            blk_c0 = src_c0 - gap - blk_w
            src_r0 = rng.randint(1, height - src_h - 1)
            src_row = rng.randint(src_r0 + 2, src_r0 + src_h - 3)
            blk_r0_min = max(1, src_row - blk_h + 1)
            blk_r0_max = min(src_row, height - blk_h - 1)
            if blk_r0_min > blk_r0_max:
                continue
            blk_r0 = rng.randint(blk_r0_min, blk_r0_max)
            src_bbox = (src_r0, src_c0, src_r0 + src_h - 1, src_c0 + src_w - 1)
            blk_bbox = (blk_r0, blk_c0, blk_r0 + blk_h - 1, blk_c0 + blk_w - 1)
            forced = {"side": "left", "pos": (src_row, src_bbox[1] + 1)}
        elif side == "bottom":
            max_src_r0 = height - (src_h + gap + blk_h) - 1
            if max_src_r0 < 1:
                continue
            src_r0 = rng.randint(1, max_src_r0)
            blk_r0 = src_r0 + src_h + gap
            src_c0 = rng.randint(1, width - src_w - 1)
            src_col = rng.randint(src_c0 + 2, src_c0 + src_w - 3)
            blk_c0_min = max(1, src_col - blk_w + 1)
            blk_c0_max = min(src_col, width - blk_w - 1)
            if blk_c0_min > blk_c0_max:
                continue
            blk_c0 = rng.randint(blk_c0_min, blk_c0_max)
            src_bbox = (src_r0, src_c0, src_r0 + src_h - 1, src_c0 + src_w - 1)
            blk_bbox = (blk_r0, blk_c0, blk_r0 + blk_h - 1, blk_c0 + blk_w - 1)
            forced = {"side": "bottom", "pos": (src_bbox[2] - 1, src_col)}
        else:
            min_src_r0 = blk_h + gap + 1
            max_src_r0 = height - src_h - 1
            if min_src_r0 > max_src_r0:
                continue
            src_r0 = rng.randint(min_src_r0, max_src_r0)
            blk_r0 = src_r0 - gap - blk_h
            src_c0 = rng.randint(1, width - src_w - 1)
            src_col = rng.randint(src_c0 + 2, src_c0 + src_w - 3)
            blk_c0_min = max(1, src_col - blk_w + 1)
            blk_c0_max = min(src_col, width - blk_w - 1)
            if blk_c0_min > blk_c0_max:
                continue
            blk_c0 = rng.randint(blk_c0_min, blk_c0_max)
            src_bbox = (src_r0, src_c0, src_r0 + src_h - 1, src_c0 + src_w - 1)
            blk_bbox = (blk_r0, blk_c0, blk_r0 + blk_h - 1, blk_c0 + blk_w - 1)
            forced = {"side": "top", "pos": (src_bbox[0] + 1, src_col)}

        if _boxes_too_close(src_bbox, blk_bbox, gap=1):
            continue
        return [
            {"bbox": src_bbox, "forced": [forced], "traps": []},
            {"bbox": blk_bbox, "forced": [], "traps": []},
        ]
    return None


def _slot_map(bbox: tuple[int, int, int, int]) -> dict[str, list[tuple[int, int]]]:
    r0, c0, r1, c1 = bbox
    return {
        "top": [(r0 + 1, c) for c in range(c0 + 2, c1 - 1)],
        "bottom": [(r1 - 1, c) for c in range(c0 + 2, c1 - 1)],
        "left": [(r, c0 + 1) for r in range(r0 + 2, r1 - 1)],
        "right": [(r, c1 - 1) for r in range(r0 + 2, r1 - 1)],
    }


def _assign_traps(rects: list[dict], rng: random.Random) -> bool:
    colors = list(TRAP_COLORS)
    rng.shuffle(colors)

    for idx, rect in enumerate(rects):
        slots = _slot_map(rect["bbox"])
        used: set[tuple[int, int]] = set()
        chosen: list[tuple[str, tuple[int, int]]] = []

        for forced in rect["forced"]:
            side = forced["side"]
            side_slots = slots[side]
            if not side_slots:
                return False
            pos = forced.get("pos")
            if pos is None:
                candidates = [slot for slot in side_slots if slot not in used]
                if not candidates:
                    return False
                pos = rng.choice(candidates)
            if pos not in side_slots or pos in used:
                return False
            used.add(pos)
            chosen.append((side, pos))

        remaining_rects = len(rects) - idx - 1
        colors_left = len(colors)
        min_traps = max(1, len(chosen))
        max_traps = min(MAX_TRAPS_PER_RECT, sum(len(v) for v in slots.values()), colors_left - remaining_rects)
        if max_traps < min_traps:
            return False
        target_count = rng.randint(min_traps, max_traps)

        while len(chosen) < target_count:
            available_sides = [
                side
                for side, positions in slots.items()
                if any(pos not in used for pos in positions)
            ]
            if not available_sides:
                return False
            side = rng.choice(available_sides)
            candidates = [pos for pos in slots[side] if pos not in used]
            pos = rng.choice(candidates)
            used.add(pos)
            chosen.append((side, pos))

        rect["traps"] = []
        for side, pos in chosen:
            if not colors:
                return False
            rect["traps"].append({"side": side, "pos": pos, "color": colors.pop()})
        rect["traps"].sort(key=lambda trap: trap["pos"])

    return True


def _render_grid(height: int, width: int, rects: list[dict]) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    for rect in rects:
        r0, c0, r1, c1 = rect["bbox"]
        grid[r0, c0 : c1 + 1] = WALL
        grid[r1, c0 : c1 + 1] = WALL
        grid[r0 : r1 + 1, c0] = WALL
        grid[r0 : r1 + 1, c1] = WALL
    for rect in rects:
        for trap in rect["traps"]:
            r, c = trap["pos"]
            grid[r, c] = int(trap["color"])
    return grid


def _validate_generated(grid: np.ndarray, mode: str) -> bool:
    try:
        out, metas = _apply_tails(grid, return_meta=True)
    except AssertionError:
        return False

    if not metas or not np.any(out != grid):
        return False
    if any(not meta["cells"] for meta in metas):
        return False
    if any(meta["stop"] == "pixel" and not meta["blocked_by_input"] for meta in metas):
        return False

    if mode == "open" and not any(meta["stop"] == "full" for meta in metas):
        return False
    if mode == "boundary" and not any(meta["stop"] == "boundary" for meta in metas):
        return False
    if mode == "wall_block" and not any(
        meta["stop"] == "pixel" and meta["blocked_by_input"] for meta in metas
    ):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    height = rng.randint(*GRID_H_RANGE)
    width = rng.randint(*GRID_W_RANGE)
    mode = rng.choice(GEN_MODES)

    min_rects = 2 if mode == "wall_block" else 1
    rect_count = max(
        min_rects,
        rng.choices(RECT_COUNT_CHOICES, weights=RECT_COUNT_WEIGHTS, k=1)[0],
    )
    rect_count = min(rect_count, len(TRAP_COLORS))

    rects: list[dict] = []
    if mode == "boundary":
        first = _sample_boundary_rect(rng, height, width, rects)
        if first is None:
            return None
        rects.append(first)
    elif mode == "wall_block":
        pair = _sample_wall_block_pair(rng, height, width)
        if pair is None:
            return None
        rects.extend(pair)

    while len(rects) < rect_count:
        bbox = _sample_free_rect(rng, height, width, rects)
        if bbox is None:
            return None
        rects.append({"bbox": bbox, "forced": [], "traps": []})

    if not _assign_traps(rects, rng):
        return None

    grid = _render_grid(height, width, rects)
    if not _validate_generated(grid, mode):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
