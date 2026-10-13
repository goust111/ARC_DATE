from __future__ import annotations

import random

import numpy as np


BG = 0
FENCE = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
SIDES = ("top", "bottom", "left", "right")

GRID_H_RANGE = (12, 24)
GRID_W_RANGE = (12, 24)
OUTER_H_RANGE = (4, 8)
OUTER_W_RANGE = (4, 9)
FRAME_COUNT_OPTIONS = (1, 2, 2, 3, 3, 4)
GAP_PROB = 0.55
EDGE_BIAS_PROB = 0.45
GEN_RETRIES = 400
FRAME_PLACEMENT_RETRIES = 240


def _analyze_side(line: np.ndarray) -> tuple[bool, int | None]:
    if int(line[0]) != FENCE or int(line[-1]) != FENCE:
        return False, None
    inner = line[1:-1]
    if inner.size == 0:
        return False, None
    if np.any((inner != BG) & (inner != FENCE)):
        return False, None
    gaps = np.flatnonzero(inner == BG)
    if gaps.size > 1:
        return False, None
    if gaps.size == 1:
        return True, int(gaps[0]) + 1
    return True, None


def _rays_from_frame(frame: dict, shape: tuple[int, int]) -> list[dict]:
    h, w = shape
    r0 = int(frame["r0"])
    c0 = int(frame["c0"])
    r1 = int(frame["r1"])
    c1 = int(frame["c1"])
    width = c1 - c0 + 1
    height = r1 - r0 + 1
    rays = []
    gaps = frame["gaps"]

    for side, offset in gaps.items():
        if side == "top":
            start = (r0, c0 + int(offset))
            delta = (-1, 0)
            length = width
        elif side == "bottom":
            start = (r1, c0 + int(offset))
            delta = (1, 0)
            length = width
        elif side == "left":
            start = (r0 + int(offset), c0)
            delta = (0, -1)
            length = height
        else:
            start = (r0 + int(offset), c1)
            delta = (0, 1)
            length = height

        cells = []
        for step in range(length):
            rr = start[0] + delta[0] * step
            cc = start[1] + delta[1] * step
            if not (0 <= rr < h and 0 <= cc < w):
                break
            cells.append((rr, cc))
        rays.append(
            {
                "side": side,
                "start": start,
                "delta": delta,
                "length": length,
                "cells": cells,
            }
        )
    return rays


def _analyze_candidate(
    g: np.ndarray,
    seed: tuple[int, int],
    r0: int,
    c0: int,
    r1: int,
    c1: int,
) -> dict | None:
    sr, sc = seed
    if not (r0 < sr < r1 and c0 < sc < c1):
        return None
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    inner = g[r0 + 1 : r1, c0 + 1 : c1]
    if np.any(inner == FENCE):
        return None
    inner_colors = np.argwhere((inner != BG) & (inner != FENCE))
    if inner_colors.shape[0] != 1:
        return None
    only_r = int(inner_colors[0, 0]) + r0 + 1
    only_c = int(inner_colors[0, 1]) + c0 + 1
    if (only_r, only_c) != seed:
        return None

    top = g[r0, c0 : c1 + 1]
    bottom = g[r1, c0 : c1 + 1]
    left = g[r0 : r1 + 1, c0]
    right = g[r0 : r1 + 1, c1]

    ok, top_gap = _analyze_side(top)
    if not ok:
        return None
    ok, bottom_gap = _analyze_side(bottom)
    if not ok:
        return None
    ok, left_gap = _analyze_side(left)
    if not ok:
        return None
    ok, right_gap = _analyze_side(right)
    if not ok:
        return None

    gaps: dict[str, int] = {}
    if top_gap is not None:
        gaps["top"] = top_gap
    if bottom_gap is not None:
        gaps["bottom"] = bottom_gap
    if left_gap is not None:
        gaps["left"] = left_gap
    if right_gap is not None:
        gaps["right"] = right_gap

    gray_count = int(
        np.count_nonzero(top == FENCE)
        + np.count_nonzero(bottom == FENCE)
        + np.count_nonzero(left[1:-1] == FENCE)
        + np.count_nonzero(right[1:-1] == FENCE)
    )
    frame = {
        "seed": seed,
        "color": int(g[seed]),
        "r0": r0,
        "c0": c0,
        "r1": r1,
        "c1": c1,
        "gaps": gaps,
        "bbox": (r0, c0, r1, c1),
        "area": (r1 - r0 + 1) * (c1 - c0 + 1),
        "gray_count": gray_count,
    }
    frame["rays"] = _rays_from_frame(frame, g.shape)
    return frame


def _find_frame_for_seed(g: np.ndarray, seed: tuple[int, int]) -> dict:
    sr, sc = seed
    h, w = g.shape
    gray = g == FENCE
    candidates: dict[tuple[int, int, int, int], dict] = {}

    for r0 in range(sr):
        if not gray[r0, :sc].any() or not gray[r0, sc + 1 :].any():
            continue
        for r1 in range(sr + 1, h):
            left_cols = np.flatnonzero(gray[r0, :sc] & gray[r1, :sc])
            if left_cols.size == 0:
                continue
            right_cols = np.flatnonzero(gray[r0, sc + 1 :] & gray[r1, sc + 1 :]) + sc + 1
            if right_cols.size == 0:
                continue

            for c0 in left_cols.tolist():
                for c1 in right_cols.tolist():
                    info = _analyze_candidate(g, seed, r0, int(c0), r1, int(c1))
                    if info is not None:
                        candidates[(r0, int(c0), r1, int(c1))] = info

    if not candidates:
        raise AssertionError(f"no fence found for seed at {seed}")
    infos = sorted(candidates.values(), key=lambda item: (int(item["area"]), -int(item["gray_count"])))
    return infos[0]


def _draw_ray(out: np.ndarray, ray: dict, color: int) -> None:
    h, w = out.shape
    start_r, start_c = ray["start"]
    dr, dc = ray["delta"]
    length = int(ray["length"])
    for step in range(length):
        rr = start_r + dr * step
        cc = start_c + dc * step
        if not (0 <= rr < h and 0 <= cc < w):
            break
        if int(out[rr, cc]) == BG:
            out[rr, cc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow colored rays outward from the gaps of gray rectangular fences.

    Each fence encloses exactly one non-black, non-gray seed pixel.
    Every gap emits a ray in the outward normal direction of its side.
    The ray length equals the full length of that side, including corners.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    seeds = sorted(
        (int(r), int(c))
        for r, c in np.argwhere((g != BG) & (g != FENCE))
    )
    frames = [_find_frame_for_seed(g, seed) for seed in seeds]

    for frame in sorted(frames, key=lambda item: item["seed"]):
        for ray in frame["rays"]:
            _draw_ray(out, ray, int(frame["color"]))
    return out


def _sample_gaps(rng: random.Random, outer_h: int, outer_w: int) -> dict[str, int]:
    chosen = [side for side in SIDES if rng.random() < GAP_PROB]
    if not chosen:
        chosen = [rng.choice(SIDES)]

    gaps: dict[str, int] = {}
    for side in chosen:
        if side in ("top", "bottom"):
            gaps[side] = rng.randint(1, outer_w - 2)
        else:
            gaps[side] = rng.randint(1, outer_h - 2)
    return gaps


def _expand_bbox(bbox: tuple[int, int, int, int], pad: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    r0, c0, r1, c1 = bbox
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _point_in_bbox(point: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    r, c = point
    return bbox[0] <= r <= bbox[2] and bbox[1] <= c <= bbox[3]


def _layout_is_clear(frames: list[dict], shape: tuple[int, int]) -> bool:
    expanded = [_expand_bbox(tuple(frame["bbox"]), 1, shape) for frame in frames]
    for i in range(len(expanded)):
        for j in range(i + 1, len(expanded)):
            if _boxes_overlap(expanded[i], expanded[j]):
                return False

    ray_sets: list[set[tuple[int, int]]] = []
    bboxes = [tuple(frame["bbox"]) for frame in frames]
    for i, frame in enumerate(frames):
        for ray in _rays_from_frame(frame, shape):
            ray_cells = set(ray["cells"])
            for other_i, other_bbox in enumerate(bboxes):
                if other_i == i:
                    continue
                if any(_point_in_bbox(cell, other_bbox) for cell in ray_cells):
                    return False
            for prev in ray_sets:
                if ray_cells & prev:
                    return False
            ray_sets.append(ray_cells)
    return True


def _render_input(shape: tuple[int, int], frames: list[dict]) -> np.ndarray:
    h, w = shape
    grid = np.full((h, w), BG, dtype=int)

    for frame in frames:
        r0 = int(frame["r0"])
        c0 = int(frame["c0"])
        r1 = int(frame["r1"])
        c1 = int(frame["c1"])
        gaps = frame["gaps"]

        grid[r0, c0 : c1 + 1] = FENCE
        grid[r1, c0 : c1 + 1] = FENCE
        grid[r0 : r1 + 1, c0] = FENCE
        grid[r0 : r1 + 1, c1] = FENCE

        if "top" in gaps:
            grid[r0, c0 + int(gaps["top"])] = BG
        if "bottom" in gaps:
            grid[r1, c0 + int(gaps["bottom"])] = BG
        if "left" in gaps:
            grid[r0 + int(gaps["left"]), c0] = BG
        if "right" in gaps:
            grid[r0 + int(gaps["right"]), c1] = BG

        sr, sc = frame["seed"]
        grid[int(sr), int(sc)] = int(frame["color"])
    return grid


def _expected_output(grid: np.ndarray, frames: list[dict]) -> np.ndarray:
    out = grid.copy()
    for frame in sorted(frames, key=lambda item: item["seed"]):
        for ray in _rays_from_frame(frame, grid.shape):
            _draw_ray(out, ray, int(frame["color"]))
    return out


def _sample_frame(rng: random.Random, shape: tuple[int, int], color: int) -> dict:
    h, w = shape
    outer_h = rng.randint(OUTER_H_RANGE[0], min(OUTER_H_RANGE[1], h))
    outer_w = rng.randint(OUTER_W_RANGE[0], min(OUTER_W_RANGE[1], w))
    gaps = _sample_gaps(rng, outer_h, outer_w)

    edge_side = rng.choice(list(gaps)) if rng.random() < EDGE_BIAS_PROB else None
    if edge_side == "top":
        r0 = rng.randint(0, min(2, h - outer_h))
        c0 = rng.randint(0, w - outer_w)
    elif edge_side == "bottom":
        low = max(0, h - outer_h - 2)
        r0 = rng.randint(low, h - outer_h)
        c0 = rng.randint(0, w - outer_w)
    elif edge_side == "left":
        r0 = rng.randint(0, h - outer_h)
        c0 = rng.randint(0, min(2, w - outer_w))
    elif edge_side == "right":
        r0 = rng.randint(0, h - outer_h)
        low = max(0, w - outer_w - 2)
        c0 = rng.randint(low, w - outer_w)
    else:
        r0 = rng.randint(0, h - outer_h)
        c0 = rng.randint(0, w - outer_w)

    seed_r = r0 + rng.randint(1, outer_h - 2)
    seed_c = c0 + rng.randint(1, outer_w - 2)
    return {
        "r0": r0,
        "c0": c0,
        "r1": r0 + outer_h - 1,
        "c1": c0 + outer_w - 1,
        "seed": (seed_r, seed_c),
        "color": int(color),
        "gaps": gaps,
        "bbox": (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1),
    }


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    shape = (h, w)

    n_frames = rng.choice(FRAME_COUNT_OPTIONS)
    if h * w < 180:
        n_frames = min(n_frames, 2)
    elif h * w < 280:
        n_frames = min(n_frames, 3)

    colors = rng.sample(SEED_COLORS, n_frames)
    frames: list[dict] = []

    for color in colors:
        placed = False
        for _ in range(FRAME_PLACEMENT_RETRIES):
            frame = _sample_frame(rng, shape, int(color))
            trial = frames + [frame]
            if _layout_is_clear(trial, shape):
                frames.append(frame)
                placed = True
                break
        if not placed:
            return None

    grid = _render_input(shape, frames)
    expected = _expected_output(grid, frames)
    if np.array_equal(expected, grid):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted retries")
