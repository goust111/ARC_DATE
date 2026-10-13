from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BG = 0
RED = 2
ORTH_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

FRAME_COLORS = [1, 3, 6, 8, 9]
INNER_COLORS = [4, 5, 7, 1, 3, 6, 8, 9]

GRID_MIN = 18
GRID_MAX = 24
FRAME_COUNT_CHOICES = [2, 3]
FRAME_SIZE_RANGE = (5, 8)
INNER_COUNT_RANGE = (2, 4)
RED_LEN_CHOICES = [2, 3]
SLIDE_DIST_RANGE = (4, 7)
GEN_RETRIES = 500
FRAME_PADDING = 3
RAY_PAD = 1

NOTCH_GAPS = {
    "TL": [("top", (0, 1)), ("left", (1, 0))],
    "TR": [("top", (0, -1)), ("right", (1, 0))],
    "BL": [("bottom", (0, 1)), ("left", (-1, 0))],
    "BR": [("bottom", (0, -1)), ("right", (-1, 0))],
}

SIDE_DIR = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ORTH_DIRS:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _frame_boundary_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _frame_side_of_gap(frame: dict, gap: tuple[int, int]) -> str | None:
    r0, c0, r1, c1 = frame["bbox"]
    r, c = gap
    sides = []
    if r == r0:
        sides.append("top")
    if r == r1:
        sides.append("bottom")
    if c == c0:
        sides.append("left")
    if c == c1:
        sides.append("right")
    if len(sides) != 1:
        return None
    return sides[0]


def _detect_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    colors = sorted(int(v) for v in np.unique(g) if int(v) not in (BG, RED))
    frames: list[dict] = []
    for color in colors:
        coords = [(int(r), int(c)) for r, c in np.argwhere(g == color)]
        if len(coords) < 12:
            continue
        r0, c0, r1, c1 = _bbox(coords)
        if r1 - r0 + 1 < 5 or c1 - c0 + 1 < 5:
            continue
        boundary = _frame_boundary_cells(r0, c0, r1, c1)
        coord_set = set(coords)
        if not coord_set <= boundary:
            continue
        gaps = sorted(boundary - coord_set)
        if not (1 <= len(gaps) <= 2):
            continue
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        inner_colors = sorted(int(v) for v in np.unique(inner) if int(v) not in (BG, RED, color))
        if len(inner_colors) != 1:
            continue
        frames.append(
            {
                "frame_color": color,
                "inner_color": inner_colors[0],
                "bbox": (r0, c0, r1, c1),
                "frame_pixels": coord_set,
                "gaps": gaps,
            }
        )
    frames.sort(key=lambda fr: (fr["bbox"][0], fr["bbox"][1], fr["frame_color"]))
    return frames


def _diagonal_fill_cells(grid: np.ndarray, frame: dict) -> set[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    frame_color = int(frame["frame_color"])
    new_cells: set[tuple[int, int]] = set()
    for r in range(h):
        for c in range(w):
            if int(g[r, c]) != BG:
                continue
            diag = False
            orth = False
            for dr, dc in DIAG_DIRS:
                rr, cc = r + dr, c + dc
                if _in_bounds(g.shape, rr, cc) and int(g[rr, cc]) == frame_color:
                    diag = True
                    break
            if not diag:
                continue
            for dr, dc in ORTH_DIRS:
                rr, cc = r + dr, c + dc
                if _in_bounds(g.shape, rr, cc) and int(g[rr, cc]) == frame_color:
                    orth = True
                    break
            if not orth:
                new_cells.add((r, c))
    return new_cells


def _red_components(grid: np.ndarray) -> tuple[list[list[tuple[int, int]]], dict[tuple[int, int], int]]:
    comps = _components(np.asarray(grid, dtype=int) == RED)
    coord_to_id: dict[tuple[int, int], int] = {}
    for idx, comp in enumerate(comps):
        for coord in comp:
            coord_to_id[coord] = idx
    return comps, coord_to_id


def _ray_hit(
    grid: np.ndarray,
    gap: tuple[int, int],
    direction: tuple[int, int],
    red_at: dict[tuple[int, int], int],
) -> tuple[int, tuple[int, int], int] | None:
    g = np.asarray(grid, dtype=int)
    dr, dc = direction
    r, c = gap
    dist = 0
    while True:
        r += dr
        c += dc
        dist += 1
        if not _in_bounds(g.shape, r, c):
            return None
        val = int(g[r, c])
        if val == BG:
            continue
        if val == RED:
            return red_at[(r, c)], (r, c), dist
        return None


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    frames = _detect_frames(g)
    red_comps, red_at = _red_components(g)

    additions_by_frame: list[set[tuple[int, int]]] = []
    slots: list[dict] = []
    candidates: list[dict] = []

    for frame_idx, frame in enumerate(frames):
        added = _diagonal_fill_cells(g, frame)
        additions_by_frame.append(added)
        r0, c0, r1, c1 = frame["bbox"]
        for gap in frame["gaps"]:
            side = _frame_side_of_gap(frame, gap)
            if side is None:
                continue
            cue_r, cue_c = gap[0] - SIDE_DIR[side][0], gap[1] - SIDE_DIR[side][1]
            if (cue_r, cue_c) not in added:
                continue
            if not (r0 < cue_r < r1 and c0 < cue_c < c1):
                continue
            direction = SIDE_DIR[side]
            slot = {"frame_idx": frame_idx, "gap": gap, "dir": direction}
            slots.append(slot)
            hit = _ray_hit(g, gap, direction, red_at)
            if hit is not None:
                red_id, hit_coord, dist = hit
                candidates.append(
                    {
                        "frame_idx": frame_idx,
                        "gap": gap,
                        "dir": direction,
                        "red_id": red_id,
                        "hit": hit_coord,
                        "dist": dist,
                    }
                )

    candidates.sort(
        key=lambda item: (
            item["dist"],
            item["frame_idx"],
            item["gap"][0],
            item["gap"][1],
            item["dir"][0],
            item["dir"][1],
        )
    )
    assigned_frames: set[int] = set()
    assigned_reds: set[int] = set()
    assignments: list[dict] = []
    for cand in candidates:
        if cand["frame_idx"] in assigned_frames or cand["red_id"] in assigned_reds:
            continue
        assigned_frames.add(cand["frame_idx"])
        assigned_reds.add(cand["red_id"])
        assignments.append(cand)

    return {
        "frames": frames,
        "additions_by_frame": additions_by_frame,
        "slots": slots,
        "candidates": candidates,
        "assignments": assignments,
        "red_comps": red_comps,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor diagonal-only frame neighbors to create docking slots, then slide aligned red components into those slots.

    Input structure:
    - A black background contains 2-3 colored rectangular one-pixel frames with 1-2 boundary gaps.
    - Each frame encloses a few pixels of a second color that marks the frame's inner foreground color.
    - Small red 4-connected components sit outside the frames.

    Rule:
    - For each frame, recolor every black cell that is diagonally adjacent to the frame color but not orthogonally adjacent to it, using that frame's inner color.
    - A boundary gap becomes a docking slot when one of those new inner-color pixels appears immediately behind the gap on the inside of the frame.
    - From each such slot, cast a ray outward from the gap; if the ray first reaches a red component, slide that whole component straight back until the struck front pixel lands on the gap.
    - Cells covered by intermediate component positions are painted black, and every red component that is never reached by any valid slot ray disappears.

    Key insight:
    - The docking geometry is not present in the input by itself: the diagonal-only recolor step creates the slot-defining corner cue, and only then can the correct outward ray be inferred.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    info = _analyze(g)

    for frame, added in zip(info["frames"], info["additions_by_frame"]):
        for r, c in added:
            out[r, c] = int(frame["inner_color"])

    out[out == RED] = BG

    for assign in info["assignments"]:
        comp = info["red_comps"][assign["red_id"]]
        hit_r, hit_c = assign["hit"]
        gap_r, gap_c = assign["gap"]
        shift_r = gap_r - hit_r
        shift_c = gap_c - hit_c
        step_r = 0 if shift_r == 0 else (1 if shift_r > 0 else -1)
        step_c = 0 if shift_c == 0 else (1 if shift_c > 0 else -1)
        steps = abs(shift_r) + abs(shift_c)

        for step in range(steps):
            off_r = step * step_r
            off_c = step * step_c
            for r, c in comp:
                rr = r + off_r
                cc = c + off_c
                if _in_bounds(out.shape, rr, cc):
                    out[rr, cc] = BG

        for r, c in comp:
            rr = r + shift_r
            cc = c + shift_c
            if _in_bounds(out.shape, rr, cc):
                out[rr, cc] = RED

    return out


def _reserve_rect(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int) -> bool:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(mask.shape[0] - 1, r1 + pad)
    cc1 = min(mask.shape[1] - 1, c1 + pad)
    if mask[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
        return False
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
    return True


def _place_frame(
    grid: np.ndarray,
    reserved: np.ndarray,
    frame_color: int,
    inner_color: int,
) -> dict | None:
    h, w = grid.shape
    for _ in range(300):
        fh = random.randint(*FRAME_SIZE_RANGE)
        fw = random.randint(*FRAME_SIZE_RANGE)
        notch = random.choice(sorted(NOTCH_GAPS))
        gap_defs = NOTCH_GAPS[notch]
        active_side, active_offset = random.choice(gap_defs)
        inactive_side, inactive_offset = [item for item in gap_defs if item[0] != active_side][0]
        active_dir = SIDE_DIR[active_side]
        red_len = random.choice(RED_LEN_CHOICES)
        slide_dist = random.randint(*SLIDE_DIST_RANGE)

        if active_dir == (-1, 0):
            r0_min = slide_dist + red_len
            r0_max = h - fh - 2
            c0_min = 1
            c0_max = w - fw - 2
        elif active_dir == (1, 0):
            r0_min = 1
            r0_max = h - fh - slide_dist - red_len - 1
            c0_min = 1
            c0_max = w - fw - 2
        elif active_dir == (0, -1):
            r0_min = 1
            r0_max = h - fh - 2
            c0_min = slide_dist + red_len
            c0_max = w - fw - 2
        else:
            r0_min = 1
            r0_max = h - fh - 2
            c0_min = 1
            c0_max = w - fw - slide_dist - red_len - 1

        if r0_min > r0_max or c0_min > c0_max:
            continue

        r0 = random.randint(r0_min, r0_max)
        c0 = random.randint(c0_min, c0_max)
        r1 = r0 + fh - 1
        c1 = c0 + fw - 1
        if not _reserve_rect(reserved.copy(), r0, c0, r1, c1, FRAME_PADDING):
            continue

        boundary = _frame_boundary_cells(r0, c0, r1, c1)
        gaps = []
        for side, (dr, dc) in gap_defs:
            if side == "top":
                gap = (r0, c0 + (1 if dc > 0 else fw - 2))
            elif side == "bottom":
                gap = (r1, c0 + (1 if dc > 0 else fw - 2))
            elif side == "left":
                gap = (r0 + (1 if dr > 0 else fh - 2), c0)
            else:
                gap = (r0 + (1 if dr > 0 else fh - 2), c1)
            gaps.append(gap)

        frame_pixels = sorted(boundary - set(gaps))
        if any(grid[r, c] != BG for r, c in frame_pixels):
            continue

        inner_positions = [
            (r, c)
            for r in range(r0 + 1, r1)
            for c in range(c0 + 1, c1)
        ]
        notch_new = {
            "TL": (r0 + 1, c0 + 1),
            "TR": (r0 + 1, c1 - 1),
            "BL": (r1 - 1, c0 + 1),
            "BR": (r1 - 1, c1 - 1),
        }[notch]
        inner_positions = [pos for pos in inner_positions if pos != notch_new]
        if len(inner_positions) < INNER_COUNT_RANGE[0]:
            continue
        inner_count = random.randint(*INNER_COUNT_RANGE)
        inner_cells = random.sample(inner_positions, inner_count)

        active_gap = None
        for gap in gaps:
            if _frame_side_of_gap(
                {"bbox": (r0, c0, r1, c1)},
                gap,
            ) == active_side:
                active_gap = gap
                break
        if active_gap is None:
            continue

        if active_dir == (-1, 0):
            hit = (active_gap[0] - slide_dist, active_gap[1])
            red_cells = [(hit[0] - k, hit[1]) for k in range(red_len)]
        elif active_dir == (1, 0):
            hit = (active_gap[0] + slide_dist, active_gap[1])
            red_cells = [(hit[0] + k, hit[1]) for k in range(red_len)]
        elif active_dir == (0, -1):
            hit = (active_gap[0], active_gap[1] - slide_dist)
            red_cells = [(hit[0], hit[1] - k) for k in range(red_len)]
        else:
            hit = (active_gap[0], active_gap[1] + slide_dist)
            red_cells = [(hit[0], hit[1] + k) for k in range(red_len)]

        if not all(_in_bounds(grid.shape, r, c) and grid[r, c] == BG for r, c in red_cells):
            continue

        clear = True
        rr, cc = active_gap
        dr, dc = active_dir
        for step in range(1, slide_dist):
            rr += dr
            cc += dc
            if not _in_bounds(grid.shape, rr, cc) or grid[rr, cc] != BG:
                clear = False
                break
        if not clear:
            continue

        if any(reserved[max(0, r - RAY_PAD) : min(h, r + RAY_PAD + 1), max(0, c - RAY_PAD) : min(w, c + RAY_PAD + 1)].any() for r, c in red_cells):
            continue

        if not _reserve_rect(reserved, r0, c0, r1, c1, FRAME_PADDING):
            continue
        for r, c in red_cells:
            reserved[max(0, r - RAY_PAD) : min(h, r + RAY_PAD + 1), max(0, c - RAY_PAD) : min(w, c + RAY_PAD + 1)] = True

        for r, c in frame_pixels:
            grid[r, c] = frame_color
        for r, c in inner_cells:
            grid[r, c] = inner_color
        for r, c in red_cells:
            grid[r, c] = RED

        return {
            "bbox": (r0, c0, r1, c1),
            "notch": notch,
            "gaps": gaps,
            "active_gap": active_gap,
            "active_dir": active_dir,
            "red_cells": red_cells,
        }
    return None


def _random_red_shape() -> list[tuple[int, int]]:
    shapes = [
        [(0, 0), (1, 0)],
        [(0, 0), (0, 1)],
        [(0, 0), (1, 0), (2, 0)],
        [(0, 0), (0, 1), (0, 2)],
        [(0, 0), (1, 0), (1, 1)],
        [(0, 1), (1, 0), (1, 1)],
    ]
    return random.choice(shapes)


def _place_distractor(grid: np.ndarray, reserved: np.ndarray) -> bool:
    h, w = grid.shape
    shape = _random_red_shape()
    sh = max(r for r, _ in shape) + 1
    sw = max(c for _, c in shape) + 1
    for _ in range(200):
        r0 = random.randint(1, h - sh - 2)
        c0 = random.randint(1, w - sw - 2)
        cells = [(r0 + r, c0 + c) for r, c in shape]
        if any(grid[r, c] != BG for r, c in cells):
            continue
        rr0 = max(0, r0 - 2)
        cc0 = max(0, c0 - 2)
        rr1 = min(h - 1, r0 + sh + 1)
        cc1 = min(w - 1, c0 + sw + 1)
        if reserved[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
            continue
        reserved[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
        for r, c in cells:
            grid[r, c] = RED
        return True
    return False


def _validate_semantics(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    info = _analyze(g)
    frame_count = len(info["frames"])
    red_count = len(info["red_comps"])
    if frame_count not in (2, 3):
        return False
    if red_count not in (2, 3):
        return False
    if len(info["assignments"]) != frame_count:
        return False

    hits_per_frame = defaultdict(int)
    for cand in info["candidates"]:
        hits_per_frame[cand["frame_idx"]] += 1
    for idx in range(frame_count):
        if hits_per_frame[idx] != 1:
            return False

    for frame, added in zip(info["frames"], info["additions_by_frame"]):
        r0, c0, r1, c1 = frame["bbox"]
        inside_added = [(r, c) for r, c in added if r0 < r < r1 and c0 < c < c1]
        if len(inside_added) != 1:
            return False
        if len(frame["gaps"]) != 2:
            return False

    out = main(g)
    out_reds, _ = _red_components(out)
    if len(out_reds) != frame_count:
        return False
    for assign in info["assignments"]:
        if int(out[assign["gap"]]) != RED:
            return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        h = random.randint(GRID_MIN, GRID_MAX)
        w = random.randint(GRID_MIN, GRID_MAX)
        frame_count = random.choice(FRAME_COUNT_CHOICES)
        if frame_count == 3 and min(h, w) < 20:
            frame_count = 2

        grid = np.zeros((h, w), dtype=int)
        reserved = np.zeros((h, w), dtype=bool)

        frame_colors = random.sample(FRAME_COLORS, frame_count)
        inner_palette = [c for c in INNER_COLORS if c not in frame_colors and c != RED]
        if len(set(inner_palette)) < frame_count:
            continue
        inner_colors = random.sample(sorted(set(inner_palette)), frame_count)

        ok = True
        for fc, ic in zip(frame_colors, inner_colors):
            spec = _place_frame(grid, reserved, fc, ic)
            if spec is None:
                ok = False
                break
        if not ok:
            continue

        if frame_count == 2 and random.random() < 0.65:
            if not _place_distractor(grid, reserved):
                continue

        if _validate_semantics(grid):
            return grid
    raise RuntimeError("failed to generate valid docking-slot ARC input")
