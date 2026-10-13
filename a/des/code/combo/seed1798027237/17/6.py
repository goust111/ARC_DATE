from __future__ import annotations

import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
SIDES = ("top", "bottom", "left", "right")

GRID_MIN = 22
GRID_MAX = 34
FRAME_SIZE_MIN = 5
FRAME_SIZE_MAX = 9
GAP_MIN = 2
GAP_MAX = 5
PATTERN_DIST_CHOICES = (2, 3)
GEN_RETRIES = 300
ZONE_RETRIES = 80


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(sorted(pts))
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _border_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    border: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        border.add((r0, c))
        border.add((r1, c))
    for r in range(r0, r1 + 1):
        border.add((r, c0))
        border.add((r, c1))
    return border


def _is_contiguous(values: list[int]) -> bool:
    if not values:
        return False
    return values == list(range(values[0], values[0] + len(values)))


def _frame_info(grid: np.ndarray, cells: list[tuple[int, int]], bg: int):
    colors = {int(grid[r, c]) for r, c in cells}
    if len(colors) != 1:
        return None

    r0, c0, r1, c1 = _bbox(cells)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    pts = set(cells)
    corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    if not corners.issubset(pts):
        return None

    border = _border_cells(r0, c0, r1, c1)
    if not pts.issubset(border):
        return None

    top_missing = [c for c in range(c0 + 1, c1) if (r0, c) not in pts]
    bottom_missing = [c for c in range(c0 + 1, c1) if (r1, c) not in pts]
    left_missing = [r for r in range(r0 + 1, r1) if (r, c0) not in pts]
    right_missing = [r for r in range(r0 + 1, r1) if (r, c1) not in pts]

    gap_options: list[tuple[str, list[tuple[int, int]]]] = []
    if top_missing:
        if not _is_contiguous(top_missing):
            return None
        gap_options.append(("top", [(r0, c) for c in top_missing]))
    if bottom_missing:
        if not _is_contiguous(bottom_missing):
            return None
        gap_options.append(("bottom", [(r1, c) for c in bottom_missing]))
    if left_missing:
        if not _is_contiguous(left_missing):
            return None
        gap_options.append(("left", [(r, c0) for r in left_missing]))
    if right_missing:
        if not _is_contiguous(right_missing):
            return None
        gap_options.append(("right", [(r, c1) for r in right_missing]))

    if len(gap_options) != 1:
        return None

    side, gap_cells = gap_options[0]
    missing_total = len(border) - len(pts)
    if missing_total != len(gap_cells):
        return None
    if pts != border - set(gap_cells):
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "cells": cells,
        "color": int(next(iter(colors))),
        "side": side,
        "gap_cells": gap_cells,
        "gap_len": len(gap_cells),
    }


def _pattern_info(grid: np.ndarray, cells: list[tuple[int, int]], bg: int):
    r0, c0, r1, c1 = _bbox(cells)
    if r0 != r1:
        return None

    pts = set(cells)
    length = c1 - c0 + 1
    if length != len(cells):
        return None
    if any((r0, c) not in pts for c in range(c0, c1 + 1)):
        return None

    colors = [int(grid[r0, c]) for c in range(c0, c1 + 1)]
    return {
        "bbox": (r0, c0, r1, c1),
        "cells": cells,
        "length": length,
        "colors": colors,
    }


def _scene_objects(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    comps = _components(g != bg)
    frames = []
    patterns = []
    others = []
    for cells in comps:
        frame = _frame_info(g, cells, bg)
        if frame is not None:
            frames.append(frame)
            continue
        pattern = _pattern_info(g, cells, bg)
        if pattern is not None:
            patterns.append(pattern)
            continue
        others.append(cells)
    frames.sort(key=lambda item: item["bbox"])
    patterns.sort(key=lambda item: item["bbox"])
    return bg, frames, patterns, others


def _min_manhattan(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
    best = 10**9
    for ar, ac in a:
        for br, bc in b:
            dist = abs(ar - br) + abs(ac - bc)
            if dist < best:
                best = dist
    return best


def _pair_frames_and_patterns(frames, patterns) -> list[int]:
    assert frames, "expected at least one frame"
    assert len(frames) == len(patterns), "frame and pattern counts must match"

    dists = [[_min_manhattan(frame["cells"], pattern["cells"]) for pattern in patterns] for frame in frames]
    mapping: list[int] = []
    for row in dists:
        best = min(row)
        best_ids = [j for j, dist in enumerate(row) if dist == best]
        assert len(best_ids) == 1, "each frame must have a unique nearest pattern"
        mapping.append(best_ids[0])

    assert len(set(mapping)) == len(frames), "nearest matching must be one-to-one"

    for j in range(len(patterns)):
        column = [dists[i][j] for i in range(len(frames))]
        best = min(column)
        assert sum(dist == best for dist in column) == 1, "each pattern must have a unique nearest frame"

    for i, j in enumerate(mapping):
        assert frames[i]["gap_len"] == patterns[j]["length"], "pattern length must match gap length"

    return mapping


def _rotated_fill_values(colors: list[int], side: str) -> list[int]:
    arr = np.array([colors], dtype=int)
    if side == "left":
        return np.rot90(arr, -1).reshape(-1).tolist()
    if side == "right":
        return np.rot90(arr, 1).reshape(-1).tolist()
    return arr.reshape(-1).tolist()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each gapped rectangular frame by moving the nearest horizontal color strip into the gap.

    - Top/bottom gaps keep the strip horizontal.
    - Left gaps use a clockwise 90-degree rotation.
    - Right gaps use a counter-clockwise 90-degree rotation.
    """
    g = np.asarray(grid, dtype=int)
    bg, frames, patterns, others = _scene_objects(g)
    assert not others, "all foreground components must be either frames or patterns"

    mapping = _pair_frames_and_patterns(frames, patterns)

    out = g.copy()
    for pattern in patterns:
        for r, c in pattern["cells"]:
            out[r, c] = bg

    for frame_idx, pattern_idx in enumerate(mapping):
        frame = frames[frame_idx]
        pattern = patterns[pattern_idx]
        fill_values = _rotated_fill_values(pattern["colors"], frame["side"])
        assert len(fill_values) == len(frame["gap_cells"])
        for (r, c), value in zip(frame["gap_cells"], fill_values):
            out[r, c] = value

    return out


def _choose_gap_start(span: int, gap_len: int, rng: np.random.Generator) -> int:
    low = 1
    high = span - gap_len - 1
    assert low <= high
    center = (span - gap_len) // 2
    near_center = [s for s in range(low, high + 1) if abs(s - center) <= 1]
    choices = near_center if near_center else list(range(low, high + 1))
    return int(rng.choice(choices))


def _pattern_sequence(length: int, palette: np.ndarray, rng: np.random.Generator) -> list[int]:
    if length == 1:
        return [int(rng.choice(palette))]
    ends = rng.choice(palette, size=2, replace=False).astype(int).tolist()
    seq = rng.choice(palette, size=length, replace=True).astype(int).tolist()
    seq[0], seq[-1] = ends[0], ends[1]
    if seq == seq[::-1]:
        alternatives = [int(c) for c in palette if int(c) != seq[-2]]
        seq[-1] = int(rng.choice(alternatives))
    return seq


def _zones(h: int, w: int) -> list[tuple[int, int, int, int]]:
    mid_r = h // 2
    mid_c = w // 2
    return [
        (1, mid_r - 2, 1, mid_c - 2),
        (1, mid_r - 2, mid_c + 1, w - 2),
        (mid_r + 1, h - 2, 1, mid_c - 2),
        (mid_r + 1, h - 2, mid_c + 1, w - 2),
    ]


def _draw_bundle(
    grid: np.ndarray,
    zone: tuple[int, int, int, int],
    side: str,
    frame_color: int,
    palette: np.ndarray,
    rng: np.random.Generator,
) -> bool:
    zr0, zr1, zc0, zc1 = zone
    zone_h = zr1 - zr0 + 1
    zone_w = zc1 - zc0 + 1
    if zone_h < FRAME_SIZE_MIN or zone_w < FRAME_SIZE_MIN:
        return False

    for _ in range(ZONE_RETRIES):
        frame_h = int(rng.integers(FRAME_SIZE_MIN, min(FRAME_SIZE_MAX, zone_h) + 1))
        frame_w = int(rng.integers(FRAME_SIZE_MIN, min(FRAME_SIZE_MAX, zone_w) + 1))
        span = frame_w if side in ("top", "bottom") else frame_h
        max_gap = min(GAP_MAX, span - 2)
        if max_gap < GAP_MIN:
            continue
        gap_len = int(rng.integers(GAP_MIN, max_gap + 1))
        dist = int(rng.choice(PATTERN_DIST_CHOICES))

        r0_lo = zr0
        r0_hi = zr1 - frame_h + 1
        c0_lo = zc0
        c0_hi = zc1 - frame_w + 1
        if side == "top":
            r0_lo += dist
        elif side == "bottom":
            r0_hi -= dist
        elif side == "left":
            c0_lo += dist + gap_len - 1
        else:
            c0_hi -= dist + gap_len - 1
        if r0_lo > r0_hi or c0_lo > c0_hi:
            continue

        r0 = int(rng.integers(r0_lo, r0_hi + 1))
        c0 = int(rng.integers(c0_lo, c0_hi + 1))
        r1 = r0 + frame_h - 1
        c1 = c0 + frame_w - 1

        if side in ("top", "bottom"):
            gap_start = _choose_gap_start(frame_w, gap_len, rng)
            gap_cells = [(r0 if side == "top" else r1, c0 + gap_start + i) for i in range(gap_len)]
            gap_c0 = gap_cells[0][1]
            shift = int(rng.integers(-1, 2))
            p_c0 = int(np.clip(gap_c0 + shift, zc0, zc1 - gap_len + 1))
            p_r = r0 - dist if side == "top" else r1 + dist
        else:
            gap_start = _choose_gap_start(frame_h, gap_len, rng)
            gap_cells = [(r0 + gap_start + i, c0 if side == "left" else c1) for i in range(gap_len)]
            gap_r0 = gap_cells[0][0]
            p_r_lo = max(zr0, gap_r0 - 1)
            p_r_hi = min(zr1, gap_r0 + gap_len)
            if p_r_lo > p_r_hi:
                continue
            p_r = int(rng.integers(p_r_lo, p_r_hi + 1))
            p_c0 = c0 - dist - gap_len + 1 if side == "left" else c1 + dist

        frame_cells = sorted(_border_cells(r0, c0, r1, c1) - set(gap_cells))
        pattern_cells = [(p_r, p_c0 + i) for i in range(gap_len)]

        all_cells = frame_cells + pattern_cells
        if any(not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]) for r, c in all_cells):
            continue
        if any(grid[r, c] != grid[0, 0] for r, c in all_cells):
            continue

        pattern_colors = _pattern_sequence(gap_len, palette, rng)
        for r, c in frame_cells:
            grid[r, c] = frame_color
        for (r, c), color in zip(pattern_cells, pattern_colors):
            grid[r, c] = color
        return True

    return False


def _valid_generated_scene(grid: np.ndarray) -> bool:
    try:
        bg, frames, patterns, others = _scene_objects(grid)
        if others or len(frames) < 2 or len(frames) != len(patterns):
            return False
        if _bg_color(grid) != bg:
            return False
        mapping = _pair_frames_and_patterns(frames, patterns)
        if len(set(mapping)) != len(frames):
            return False
        out = main(grid)
        if np.array_equal(out, grid):
            return False
    except AssertionError:
        return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    w = int(rng.integers(GRID_MIN, GRID_MAX + 1))

    bg = int(rng.integers(0, 10))
    remaining = [color for color in range(10) if color != bg]
    frame_color = int(rng.choice(remaining))
    palette = np.array([color for color in remaining if color != frame_color], dtype=int)

    grid = np.full((h, w), bg, dtype=int)
    zones = _zones(h, w)
    count = int(rng.integers(2, 5))
    chosen_zone_ids = rng.choice(len(zones), size=count, replace=False)
    sides = rng.permutation(np.array(SIDES, dtype=object))[:count].tolist()

    for zone_id, side in zip(chosen_zone_ids.tolist(), sides):
        ok = _draw_bundle(grid, zones[zone_id], side, frame_color, palette, rng)
        if not ok:
            return None

    if not _valid_generated_scene(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
