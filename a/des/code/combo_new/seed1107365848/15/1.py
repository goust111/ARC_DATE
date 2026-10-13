from __future__ import annotations

from collections import Counter
from itertools import product
import random

import numpy as np


BG = 8
SIDES = ("top", "bottom", "left", "right")
GEN_RETRIES = 400
PLACEMENT_RETRIES = 120


def _corner_bg(grid: np.ndarray) -> int:
    corners = [
        int(grid[0, 0]),
        int(grid[0, -1]),
        int(grid[-1, 0]),
        int(grid[-1, -1]),
    ]
    return int(Counter(corners).most_common(1)[0][0])


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _color_components(grid: np.ndarray, bg: int) -> list[tuple[int, list[tuple[int, int]]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[tuple[int, list[tuple[int, int]]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((int(rr), int(cc)))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append((color, pts))
    return comps


def _marker_cell(r0: int, c0: int, r1: int, c1: int, side: str) -> tuple[int, int]:
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    mid_r = r0 + 1 + (inner_h - 1) // 2
    mid_c = c0 + 1 + (inner_w - 1) // 2
    if side == "top":
        return r0 + 1, mid_c
    if side == "bottom":
        return r1 - 1, mid_c
    if side == "left":
        return mid_r, c0 + 1
    if side == "right":
        return mid_r, c1 - 1
    raise ValueError(f"unknown side: {side}")


def _erase_gate(arr: np.ndarray, frame: dict, value: int | bool) -> None:
    r0, c0, r1, c1 = frame["r0"], frame["c0"], frame["r1"], frame["c1"]
    side = frame["side"]
    if side == "top":
        arr[r0, c0 + 1 : c1] = value
    elif side == "bottom":
        arr[r1, c0 + 1 : c1] = value
    elif side == "left":
        arr[r0 + 1 : r1, c0] = value
    elif side == "right":
        arr[r0 + 1 : r1, c1] = value
    else:
        raise ValueError(f"unknown side: {side}")


def _spill_cells(occupied: np.ndarray, frame: dict) -> tuple[list[tuple[int, int]], str]:
    h, w = occupied.shape
    r0, c0, r1, c1 = frame["r0"], frame["c0"], frame["r1"], frame["c1"]
    side = frame["side"]
    cells: list[tuple[int, int]] = []
    mode = "boundary"

    if side == "top":
        cols = slice(c0 + 1, c1)
        for r in range(r0 - 1, -1, -1):
            if bool(occupied[r, cols].any()):
                mode = "fence"
                break
            cells.extend((r, c) for c in range(c0 + 1, c1))
    elif side == "bottom":
        cols = slice(c0 + 1, c1)
        for r in range(r1 + 1, h):
            if bool(occupied[r, cols].any()):
                mode = "fence"
                break
            cells.extend((r, c) for c in range(c0 + 1, c1))
    elif side == "left":
        rows = slice(r0 + 1, r1)
        for c in range(c0 - 1, -1, -1):
            if bool(occupied[rows, c].any()):
                mode = "fence"
                break
            cells.extend((r, c) for r in range(r0 + 1, r1))
    elif side == "right":
        rows = slice(r0 + 1, r1)
        for c in range(c1 + 1, w):
            if bool(occupied[rows, c].any()):
                mode = "fence"
                break
            cells.extend((r, c) for r in range(r0 + 1, r1))
    else:
        raise ValueError(f"unknown side: {side}")

    return cells, mode


def _find_fences(grid: np.ndarray, bg: int) -> list[dict]:
    fences: list[dict] = []
    for fence_color, pts in _color_components(grid, bg):
        r0, c0, r1, c1 = _bbox(pts)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if height < 4 or width < 4:
            continue

        expected_size = 2 * (height + width) - 4
        if len(pts) != expected_size:
            continue

        ptset = set(pts)
        ring_ok = True
        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                on_border = rr in (r0, r1) or cc in (c0, c1)
                if ((rr, cc) in ptset) != on_border:
                    ring_ok = False
                    break
            if not ring_ok:
                break
        if not ring_ok:
            continue

        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        inner_h, inner_w = inner.shape
        if inner_h < 2 or inner_w < 2:
            continue

        vals, cnts = np.unique(inner, return_counts=True)
        fill_color = int(vals[int(np.argmax(cnts))])
        if fill_color in (bg, fence_color):
            continue

        marker_locs = np.argwhere(inner != fill_color)
        if marker_locs.shape[0] != 1:
            continue
        mr, mc = map(int, marker_locs[0])
        mr += r0 + 1
        mc += c0 + 1
        marker_color = int(grid[mr, mc])
        if marker_color in (bg, fence_color, fill_color):
            continue

        candidate_sides: list[str] = []
        if mr == r0 + 1:
            candidate_sides.append("top")
        if mr == r1 - 1:
            candidate_sides.append("bottom")
        if mc == c0 + 1:
            candidate_sides.append("left")
        if mc == c1 - 1:
            candidate_sides.append("right")
        if len(candidate_sides) != 1:
            continue
        side = candidate_sides[0]

        if (mr, mc) != _marker_cell(r0, c0, r1, c1, side):
            continue

        fences.append(
            {
                "r0": int(r0),
                "c0": int(c0),
                "r1": int(r1),
                "c1": int(c1),
                "fence": int(fence_color),
                "fill": int(fill_color),
                "marker": int(marker_color),
                "marker_pos": (int(mr), int(mc)),
                "side": side,
            }
        )

    fences.sort(key=lambda frame: (frame["r0"], frame["c0"], frame["fence"]))
    return fences


def main(grid: np.ndarray) -> np.ndarray:
    """
    Open each marked rectangular fence on the marked side and spill its fill outward.

    Input structure:
    - The background is azure and reaches the grid border.
    - Several one-pixel-thick monochrome rectangular fence outlines sit on the background.
    - Each fence encloses a solid interior color plus one single-pixel marker of a third color
      placed just inside exactly one side at that side's midpoint.

    Rule:
    - Identify every hollow rectangle and read the marker location as the gate side
      (top, bottom, left, or right).
    - Remove the marker pixel and erase the non-corner fence segment on that side,
      creating an opening whose width matches the fence interior span along that side.
    - Extend the interior fill color straight outward through the opening as a constant-width
      strip until the next strip step would hit any non-background pixel or leave the grid.

    Key insight:
    - The marker does not supply a color mapping; it encodes which side of the rectangle
      becomes the gate, and that inferred direction controls the outward spill.
    """
    g = np.asarray(grid, dtype=int)
    bg = _corner_bg(g)
    fences = _find_fences(g, bg)
    assert fences, "expected at least one marked rectangular fence"

    base = g.copy()
    for frame in fences:
        mr, mc = frame["marker_pos"]
        base[mr, mc] = bg
        _erase_gate(base, frame, bg)

    occupied = base != bg
    out = base.copy()
    for frame in fences:
        cells, _mode = _spill_cells(occupied, frame)
        for r, c in cells:
            out[r, c] = frame["fill"]

    return out


def _boxes_too_close(box_a: dict, box_b: dict, pad: int = 1) -> bool:
    return not (
        box_a["r1"] + pad < box_b["r0"] - pad
        or box_b["r1"] + pad < box_a["r0"] - pad
        or box_a["c1"] + pad < box_b["c0"] - pad
        or box_b["c1"] + pad < box_a["c0"] - pad
    )


def _place_boxes(rng: random.Random, h: int, w: int, n_frames: int) -> list[dict] | None:
    boxes: list[dict] = []
    size_hi = 8 if n_frames <= 3 else 7
    for _ in range(n_frames):
        placed = False
        for _attempt in range(PLACEMENT_RETRIES):
            box_h = rng.randint(4, min(size_hi, h - 2))
            box_w = rng.randint(4, min(size_hi, w - 2))
            if h - box_h - 1 < 1 or w - box_w - 1 < 1:
                continue
            r0 = rng.randint(1, h - box_h - 1)
            c0 = rng.randint(1, w - box_w - 1)
            cand = {"r0": r0, "c0": c0, "r1": r0 + box_h - 1, "c1": c0 + box_w - 1}
            if all(not _boxes_too_close(cand, other, pad=1) for other in boxes):
                boxes.append(cand)
                placed = True
                break
        if not placed:
            return None
    boxes.sort(key=lambda frame: (frame["r0"], frame["c0"]))
    return boxes


def _build_base_mask(boxes: list[dict], sides: tuple[str, ...], h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for box in boxes:
        mask[box["r0"] : box["r1"] + 1, box["c0"] : box["c1"] + 1] = True
    for box, side in zip(boxes, sides):
        frame = {**box, "side": side}
        _erase_gate(mask, frame, False)
        mr, mc = _marker_cell(box["r0"], box["c0"], box["r1"], box["c1"], side)
        mask[mr, mc] = False
    return mask


def _evaluate_assignment(
    boxes: list[dict], sides: tuple[str, ...], h: int, w: int
) -> tuple[tuple[int, int, int, int], list[dict]] | None:
    base_mask = _build_base_mask(boxes, sides, h, w)
    used: set[tuple[int, int]] = set()
    plan: list[dict] = []
    modes: list[str] = []
    side_counts = Counter()
    total_area = 0

    for box, side in zip(boxes, sides):
        frame = {**box, "side": side}
        cells, mode = _spill_cells(base_mask, frame)
        if not cells:
            return None
        for cell in cells:
            if cell in used:
                return None
        used.update(cells)
        plan.append({**box, "side": side, "spill_cells": cells, "mode": mode})
        modes.append(mode)
        side_counts[side] += 1
        total_area += len(cells)

    if "fence" not in modes or "boundary" not in modes:
        return None

    score = (
        -abs(modes.count("fence") - modes.count("boundary")),
        len(side_counts),
        -max(side_counts.values()),
        total_area,
    )
    return score, plan


def _choose_assignment(
    rng: random.Random, boxes: list[dict], h: int, w: int
) -> list[dict] | None:
    combos = list(product(SIDES, repeat=len(boxes)))
    rng.shuffle(combos)

    best_score: tuple[int, int, int, int] | None = None
    best_plans: list[list[dict]] = []
    for sides in combos:
        evaluated = _evaluate_assignment(boxes, sides, h, w)
        if evaluated is None:
            continue
        score, plan = evaluated
        if best_score is None or score > best_score:
            best_score = score
            best_plans = [plan]
        elif score == best_score:
            best_plans.append(plan)

    if not best_plans:
        return None
    return rng.choice(best_plans)


def _assign_colors(rng: random.Random, n_frames: int) -> tuple[list[int], list[int], list[int]]:
    palette = [color for color in range(10) if color != BG]
    fence_colors = rng.sample(palette, n_frames)
    fill_pool = [color for color in palette if color not in fence_colors]
    fill_colors = rng.sample(fill_pool, n_frames)
    marker_colors: list[int] = []
    for fence, fill in zip(fence_colors, fill_colors):
        marker_pool = [color for color in palette if color not in (fence, fill)]
        marker_colors.append(rng.choice(marker_pool))
    return fence_colors, fill_colors, marker_colors


def _draw_frame(
    grid: np.ndarray,
    box: dict,
    side: str,
    fence_color: int,
    fill_color: int,
    marker_color: int,
) -> None:
    r0, c0, r1, c1 = box["r0"], box["c0"], box["r1"], box["c1"]
    grid[r0 + 1 : r1, c0 + 1 : c1] = fill_color
    grid[r0, c0 : c1 + 1] = fence_color
    grid[r1, c0 : c1 + 1] = fence_color
    grid[r0 : r1 + 1, c0] = fence_color
    grid[r0 : r1 + 1, c1] = fence_color
    mr, mc = _marker_cell(r0, c0, r1, c1, side)
    grid[mr, mc] = marker_color


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_frames = rng.choice([2, 3, 4])
    if n_frames == 2:
        h = rng.randint(14, 24)
        w = rng.randint(14, 24)
    elif n_frames == 3:
        h = rng.randint(16, 28)
        w = rng.randint(16, 28)
    else:
        h = rng.randint(18, 30)
        w = rng.randint(18, 30)

    boxes = _place_boxes(rng, h, w, n_frames)
    if boxes is None:
        return None

    assignment = _choose_assignment(rng, boxes, h, w)
    if assignment is None:
        return None

    fence_colors, fill_colors, marker_colors = _assign_colors(rng, n_frames)
    grid = np.full((h, w), BG, dtype=int)
    for idx, plan in enumerate(assignment):
        _draw_frame(
            grid,
            plan,
            plan["side"],
            fence_colors[idx],
            fill_colors[idx],
            marker_colors[idx],
        )

    try:
        parsed = _find_fences(grid, BG)
        if len(parsed) != n_frames:
            return None
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid marked-fence puzzle")
