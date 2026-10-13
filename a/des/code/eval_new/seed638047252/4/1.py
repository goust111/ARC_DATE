from __future__ import annotations

from collections import defaultdict

import numpy as np


BG = 0
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
PALETTE = tuple(range(1, 10))
SLOT = 15
BRANCH_MAX = 3
OUTER_SIDES = (7, 9)
GEN_ATTEMPTS = 300


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
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _rect_border(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    border: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        border.add((r0, c))
        border.add((r1, c))
    for r in range(r0 + 1, r1):
        border.add((r, c0))
        border.add((r, c1))
    return border


def _find_frames(grid: np.ndarray) -> list[dict[str, int | tuple[int, int, int, int]]]:
    frames: list[dict[str, int | tuple[int, int, int, int]]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != BG):
        for comp in _components(grid == color):
            r0, c0, r1, c1 = _bbox(comp)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            if set(comp) == _rect_border(r0, c0, r1, c1):
                frames.append({"color": color, "bbox": (r0, c0, r1, c1)})
    frames.sort(key=lambda frame: (frame["bbox"][0], frame["bbox"][1], frame["color"]))  # type: ignore[index]
    return frames


def _centroid_direction(
    points: list[tuple[int, int]], center_r: float, center_c: float
) -> str | None:
    if not points:
        return None
    mean_r = float(np.mean([r for r, _ in points]))
    mean_c = float(np.mean([c for _, c in points]))
    dr = mean_r - center_r
    dc = mean_c - center_c
    if abs(dr) > abs(dc):
        return "down" if dr > 0 else "up"
    if abs(dc) > abs(dr):
        return "right" if dc > 0 else "left"
    if abs(dr) > 1e-9:
        return "down" if dr > 0 else "up"
    if abs(dc) > 1e-9:
        return "right" if dc > 0 else "left"
    return None


def _paint_branch(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    direction: str,
    length: int,
    color: int,
) -> None:
    r0, c0, r1, c1 = bbox
    mid_r = (r0 + r1) // 2
    mid_c = (c0 + c1) // 2
    dr, dc = DIRS[direction]
    if direction == "up":
        start_r, start_c = r0 - 1, mid_c
    elif direction == "down":
        start_r, start_c = r1 + 1, mid_c
    elif direction == "left":
        start_r, start_c = mid_r, c0 - 1
    else:
        start_r, start_c = mid_r, c1 + 1

    h, w = grid.shape
    for step in range(length):
        rr = start_r + dr * step
        cc = start_c + dc * step
        if 0 <= rr < h and 0 <= cc < w:
            grid[rr, cc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow color-matched branches outward from each frame based on interior marker groups.

    Input structure:
    - bg = black. The grid contains several single-color hollow rectangular frames, each with a distinct frame color.
    - Inside each frame are a few single-pixel markers whose colors are borrowed from other frames in the same grid.
    - Frames do not overlap and leave open space around themselves for outward growth.

    Rule:
    - Detect each hollow frame and group its interior marker pixels by marker color.
    - For one marker color group, compare that group's centroid with the frame center; the side it leans toward (up, down, left, or right) is the growth direction.
    - From the midpoint just outside that frame side, grow a 1-pixel-wide branch in the frame's own color, with length equal to the number of markers in the group.

    Key insight:
    - A marker group's color identifies another frame, but its position inside the current frame determines direction while its multiplicity determines branch length.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    frames = _find_frames(g)
    frame_colors = {int(frame["color"]) for frame in frames}

    for frame in frames:
        color = int(frame["color"])
        r0, c0, r1, c1 = frame["bbox"]  # type: ignore[index]
        center_r = (r0 + r1) / 2.0
        center_c = (c0 + c1) / 2.0

        groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                marker = int(g[r, c])
                if marker != BG and marker != color and marker in frame_colors:
                    groups[marker].append((r, c))

        for marker_color in sorted(groups):
            points = groups[marker_color]
            direction = _centroid_direction(points, center_r, center_c)
            if direction is None:
                continue
            _paint_branch(out, (r0, c0, r1, c1), direction, len(points), color)

    return out


def _zone_cells(inner_h: int, inner_w: int, direction: str) -> list[tuple[int, int]]:
    center_r = (inner_h - 1) / 2.0
    center_c = (inner_w - 1) / 2.0
    cells: list[tuple[int, int]] = []
    for r in range(inner_h):
        for c in range(inner_w):
            if _centroid_direction([(r, c)], center_r, center_c) == direction:
                cells.append((r, c))
    return cells


def _sample_group_cells(
    zone: list[tuple[int, int]],
    count: int,
    used: set[tuple[int, int]],
    rng: np.random.Generator,
) -> list[tuple[int, int]] | None:
    for _ in range(80):
        order = list(zone)
        rng.shuffle(order)
        chosen: list[tuple[int, int]] = []
        for cell in order:
            if cell in used:
                continue
            if any(abs(cell[0] - rr) + abs(cell[1] - cc) == 1 for rr, cc in chosen):
                continue
            chosen.append(cell)
            if len(chosen) == count:
                return chosen
    return None


def _build_marker_groups(
    inner_h: int,
    inner_w: int,
    group_specs: list[tuple[int, str]],
    rng: np.random.Generator,
) -> dict[int, dict[str, object]] | None:
    zones = {direction: _zone_cells(inner_h, inner_w, direction) for _, direction in group_specs}
    for _ in range(120):
        used: set[tuple[int, int]] = set()
        groups: dict[int, dict[str, object]] = {}
        ok = True
        for marker_color, direction in group_specs:
            max_count = min(BRANCH_MAX, len(zones[direction]))
            if max_count < 1:
                ok = False
                break
            count = int(rng.integers(1, max_count + 1))
            cells = _sample_group_cells(zones[direction], count, used, rng)
            if cells is None:
                ok = False
                break
            groups[marker_color] = {"direction": direction, "cells": cells}
            used.update(cells)
        if ok:
            return groups
    return None


def _normalized_slots(n_frames: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    all_slots = [(0, 0), (0, 1), (1, 0), (1, 1)]
    chosen_ids = rng.choice(len(all_slots), size=n_frames, replace=False)
    chosen = [all_slots[int(i)] for i in chosen_ids]
    min_r = min(r for r, _ in chosen)
    min_c = min(c for _, c in chosen)
    return [(r - min_r, c - min_c) for r, c in chosen]


def _crop_to_content(
    inp: np.ndarray, out: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(out != BG)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    top_keep = int(rng.integers(0, min(3, int(r0)) + 1))
    left_keep = int(rng.integers(0, min(3, int(c0)) + 1))
    bottom_keep = int(rng.integers(0, min(3, int(out.shape[0] - 1 - r1)) + 1))
    right_keep = int(rng.integers(0, min(3, int(out.shape[1] - 1 - c1)) + 1))
    top = int(r0) - top_keep
    left = int(c0) - left_keep
    bottom = int(r1) + 1 + bottom_keep
    right = int(c1) + 1 + right_keep
    return inp[top:bottom, left:right], out[top:bottom, left:right]


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_frames = int(rng.integers(2, 5))
    slots = _normalized_slots(n_frames, rng)
    slot_rows = max(r for r, _ in slots) + 1
    slot_cols = max(c for _, c in slots) + 1
    h0 = slot_rows * SLOT
    w0 = slot_cols * SLOT

    colors = [int(v) for v in rng.choice(PALETTE, size=n_frames, replace=False)]
    specs: list[dict[str, object]] = []

    for slot_index, (slot_r, slot_c) in enumerate(slots):
        color = colors[slot_index]
        height = int(rng.choice(OUTER_SIDES))
        width = int(rng.choice(OUTER_SIDES))
        free_h = SLOT - height - 2 * BRANCH_MAX
        free_w = SLOT - width - 2 * BRANCH_MAX
        top = slot_r * SLOT + BRANCH_MAX + int(rng.integers(0, free_h + 1))
        left = slot_c * SLOT + BRANCH_MAX + int(rng.integers(0, free_w + 1))

        other_colors = [c for c in colors if c != color]
        max_groups = min(2, len(other_colors))
        if max_groups < 1:
            return None
        n_groups = int(rng.integers(1, max_groups + 1))
        marker_colors = [int(v) for v in rng.choice(other_colors, size=n_groups, replace=False)]
        directions = list(rng.choice(list(DIRS.keys()), size=n_groups, replace=False))

        groups = _build_marker_groups(height - 2, width - 2, list(zip(marker_colors, directions)), rng)
        if groups is None:
            return None

        specs.append(
            {
                "color": color,
                "bbox": (top, left, top + height - 1, left + width - 1),
                "groups": groups,
            }
        )

    inp = np.full((h0, w0), BG, dtype=int)
    expected = np.full((h0, w0), BG, dtype=int)

    for spec in specs:
        color = int(spec["color"])
        r0, c0, r1, c1 = spec["bbox"]  # type: ignore[index]
        for r, c in _rect_border(r0, c0, r1, c1):
            inp[r, c] = color
            expected[r, c] = color

        groups: dict[int, dict[str, object]] = spec["groups"]  # type: ignore[assignment]
        for marker_color, group in groups.items():
            for ir, ic in group["cells"]:  # type: ignore[index]
                inp[r0 + 1 + ir, c0 + 1 + ic] = marker_color
                expected[r0 + 1 + ir, c0 + 1 + ic] = marker_color

        for group in groups.values():
            _paint_branch(expected, (r0, c0, r1, c1), group["direction"], len(group["cells"]), color)  # type: ignore[index]

    if np.array_equal(inp, expected):
        return None

    inp, expected = _crop_to_content(inp, expected, rng)
    if inp.ndim != 2 or max(inp.shape) > 30 or max(expected.shape) > 30:
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_ATTEMPTS} attempts")
