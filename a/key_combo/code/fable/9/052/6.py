from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
WALL = 5
OBJECT_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_ATTEMPTS = 400
LANE_GAP = 1
ZONE_GAP = 2


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _paint_rect(grid: np.ndarray, r: int, c: int, h: int, w: int, color: int) -> None:
    grid[r : r + h, c : c + w] = int(color)


def _rect_in_bounds(shape: tuple[int, int], r: int, c: int, h: int, w: int) -> bool:
    rows, cols = shape
    return 0 <= r and 0 <= c and r + h <= rows and c + w <= cols


def _dominant_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _extract_objects(grid: np.ndarray) -> tuple[int, np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = _dominant_color(g)
    wall_mask = g == WALL
    objects: list[dict] = []

    for color in sorted(int(v) for v in np.unique(g) if int(v) not in {bg, WALL}):
        for coords in _components(g == color):
            r0, c0, r1, c1 = _bbox(coords)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if coords.shape[0] != h * w:
                raise AssertionError("foreground objects must be solid rectangles")
            if h == w:
                raise AssertionError("foreground rectangles must have a unique long side")
            objects.append(
                {
                    "color": color,
                    "top_left": (r0, c0),
                    "h": h,
                    "w": w,
                    "step": max(h, w),
                    "dir": (0, 1) if w > h else (1, 0),
                    "coords": coords,
                }
            )

    objects.sort(key=lambda obj: (obj["top_left"][0], obj["top_left"][1], obj["color"]))
    return bg, wall_mask, objects


def _simulate_quantum(obj: dict, wall_mask: np.ndarray, shape: tuple[int, int]) -> dict:
    h = obj["h"]
    w = obj["w"]
    step = obj["step"]
    dr, dc = obj["dir"]
    r, c = obj["top_left"]
    steps = 0

    while True:
        nr = r + dr * step
        nc = c + dc * step
        if not _rect_in_bounds(shape, nr, nc, h, w):
            return {"top_left": (r, c), "n_steps": steps, "stop_reason": "boundary"}
        if bool(wall_mask[nr : nr + h, nc : nc + w].any()):
            return {"top_left": (r, c), "n_steps": steps, "stop_reason": "wall"}
        r, c = nr, nc
        steps += 1


def _simulate_glide(obj: dict, wall_mask: np.ndarray, shape: tuple[int, int]) -> tuple[int, int]:
    h = obj["h"]
    w = obj["w"]
    dr, dc = obj["dir"]
    r, c = obj["top_left"]
    while True:
        nr = r + dr
        nc = c + dc
        if not _rect_in_bounds(shape, nr, nc, h, w):
            return (r, c)
        if bool(wall_mask[nr : nr + h, nc : nc + w].any()):
            return (r, c)
        r, c = nr, nc


def _corridor_cells(obj: dict, steps: int) -> set[tuple[int, int]]:
    h = obj["h"]
    w = obj["w"]
    step = obj["step"]
    dr, dc = obj["dir"]
    r0, c0 = obj["top_left"]
    cells: set[tuple[int, int]] = set()
    for t in range(steps + 1):
        rr = r0 + dr * step * t
        cc = c0 + dc * step * t
        for r in range(rr, rr + h):
            for c in range(cc, cc + w):
                cells.add((r, c))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Translate each solid rectangle by repeated whole-length jumps along its long side.

    Input structure:
    - The background contains several non-grey solid foreground rectangles and several grey wall segments.
    - Every rectangle is non-square: wider rectangles move right, taller rectangles move down.
    - Wall segments are fixed obstacles, and the rectangles' motion corridors do not intersect each other.

    Rule:
    - Detect each solid foreground rectangle and infer its direction from its long side.
    - Repeatedly try to move that rectangle by one full step, where the step length equals its own long-side length.
    - A jump is allowed only if the destination stays inside the grid and does not overlap any grey wall cell; otherwise the rectangle stops there.
    - Keep all grey walls unchanged, clear every old rectangle position to background, and redraw each rectangle at its final position.

    Key insight:
    - The motion is quantized by each rectangle's own long side, so objects usually stop with a leftover gap instead of sliding flush against the obstacle.
    """
    g = np.asarray(grid, dtype=int)
    bg, wall_mask, objects = _extract_objects(g)
    out = np.full_like(g, bg)
    out[wall_mask] = WALL

    for obj in objects:
        sim = _simulate_quantum(obj, wall_mask, g.shape)
        r, c = sim["top_left"]
        _paint_rect(out, r, c, obj["h"], obj["w"], obj["color"])

    return out


def _choose_mode(total: int) -> tuple[int, int]:
    roll = random.random()
    if total == 2 and roll < 0.65:
        return 1, 1
    if roll < 0.65:
        n_h = random.randint(1, total - 1)
        return n_h, total - n_h
    if roll < 0.825:
        return total, 0
    return 0, total


def _cause_list(total: int) -> list[str]:
    boundary_count = random.randint(1, total - 1)
    causes = ["boundary"] * boundary_count + ["wall"] * (total - boundary_count)
    random.shuffle(causes)
    return causes


def _random_gap(step: int) -> int:
    if step <= 2:
        return 1 if random.random() < 0.75 else 0
    if random.random() < 0.75:
        return random.randint(1, step - 1)
    return random.randint(0, step - 1)


def _pick_horizontal_position(width: int, grid_w: int, cause: str) -> dict | None:
    if cause == "boundary":
        lo = max(width, grid_w - 2 * width + 1)
        hi = grid_w - width
        if lo > hi:
            return None
        final_c = random.randint(lo, hi)
        max_jumps = min(3, final_c // width)
        if max_jumps < 1:
            return None
        jumps = random.randint(1, max_jumps)
        return {
            "input_c": final_c - jumps * width,
            "final_c": final_c,
            "jumps": jumps,
        }

    lo = width
    hi = grid_w - 2 * width
    if lo > hi:
        return None
    final_c = random.randint(lo, hi)
    max_jumps = min(3, final_c // width)
    if max_jumps < 1:
        return None
    jumps = random.randint(1, max_jumps)
    gap = _random_gap(width)
    return {
        "input_c": final_c - jumps * width,
        "final_c": final_c,
        "wall_c": final_c + width + gap,
        "jumps": jumps,
    }


def _pick_vertical_position(height: int, grid_h: int, base_r: int, cause: str) -> dict | None:
    if cause == "boundary":
        lo = max(base_r + height, grid_h - 2 * height + 1)
        hi = grid_h - height
        if lo > hi:
            return None
        final_r = random.randint(lo, hi)
        max_jumps = min(3, (final_r - base_r) // height)
        if max_jumps < 1:
            return None
        jumps = random.randint(1, max_jumps)
        return {
            "input_r": final_r - jumps * height,
            "final_r": final_r,
            "jumps": jumps,
        }

    lo = base_r + height
    hi = grid_h - 2 * height
    if lo > hi:
        return None
    final_r = random.randint(lo, hi)
    max_jumps = min(3, (final_r - base_r) // height)
    if max_jumps < 1:
        return None
    jumps = random.randint(1, max_jumps)
    gap = _random_gap(height)
    return {
        "input_r": final_r - jumps * height,
        "final_r": final_r,
        "wall_r": final_r + height + gap,
        "jumps": jumps,
    }


def _expected_from_objects(shape: tuple[int, int], objects: list[dict]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for obj in objects:
        if obj["kind"] == "H" and obj["cause"] == "wall":
            out[obj["r"] : obj["r"] + obj["h"], obj["wall_c"]] = WALL
        if obj["kind"] == "V" and obj["cause"] == "wall":
            out[obj["wall_r"], obj["c"] : obj["c"] + obj["w"]] = WALL
    for obj in objects:
        if obj["kind"] == "H":
            _paint_rect(out, obj["r"], obj["input_c"], obj["h"], obj["w"], obj["color"])
        else:
            _paint_rect(out, obj["input_r"], obj["c"], obj["h"], obj["w"], obj["color"])
    return out


def _build_attempt() -> np.ndarray | None:
    total = random.randint(2, 4)
    n_h, n_v = _choose_mode(total)
    causes = _cause_list(total)
    steps = random.sample([2, 3, 4, 5, 6], total)
    colors = random.sample(OBJECT_COLORS, total)

    horiz: list[dict] = []
    vert: list[dict] = []
    for idx in range(n_h):
        step = steps[idx]
        horiz.append(
            {
                "kind": "H",
                "step": step,
                "h": random.randint(1, min(3, step - 1)),
                "w": step,
                "cause": causes[idx],
                "color": colors[idx],
            }
        )
    for idx in range(n_h, total):
        step = steps[idx]
        vert.append(
            {
                "kind": "V",
                "step": step,
                "h": step,
                "w": random.randint(1, min(3, step - 1)),
                "cause": causes[idx],
                "color": colors[idx],
            }
        )

    required_top = sum(obj["h"] for obj in horiz) + max(0, len(horiz) - 1) * LANE_GAP
    separator = ZONE_GAP if horiz and vert else 0
    base0 = required_top + separator
    motion_h = max((base0 + (3 if obj["cause"] == "wall" else 2) * obj["h"] for obj in vert), default=0)
    h_min = max(10, required_top, base0, motion_h)

    band_w = sum(obj["w"] for obj in vert) + max(0, len(vert) - 1) * LANE_GAP
    motion_w = max(((3 if obj["cause"] == "wall" else 2) * obj["w"] for obj in horiz), default=0)
    w_min = max(10, band_w, motion_w)
    if h_min > 30 or w_min > 30:
        return None

    grid_h = random.randint(h_min, 30)
    grid_w = random.randint(w_min, 30)

    if horiz and vert:
        top_shift = random.randint(0, grid_h - max(base0, motion_h))
        base_r = top_shift + base0
    elif horiz:
        top_shift = random.randint(0, grid_h - required_top)
        base_r = 0
    else:
        top_shift = 0
        base_r = 0

    row = top_shift
    for obj in horiz:
        obj["r"] = row
        row += obj["h"] + LANE_GAP

    if vert:
        left_shift = random.randint(0, grid_w - band_w)
        col = left_shift
        for obj in vert:
            obj["c"] = col
            col += obj["w"] + LANE_GAP

    for obj in horiz:
        pos = _pick_horizontal_position(obj["w"], grid_w, obj["cause"])
        if pos is None:
            return None
        obj.update(pos)

    for obj in vert:
        pos = _pick_vertical_position(obj["h"], grid_h, base_r, obj["cause"])
        if pos is None:
            return None
        obj.update(pos)

    all_objects = horiz + vert
    grid = _expected_from_objects((grid_h, grid_w), all_objects)
    if not _validate_instance(grid):
        return None
    return grid


def _validate_instance(grid: np.ndarray) -> bool:
    try:
        bg, wall_mask, objects = _extract_objects(grid)
        if bg != BG or len(objects) < 2:
            return False
        if len({obj["step"] for obj in objects}) < 2:
            return False

        wall_comps = _components(wall_mask)
        if not wall_comps:
            return False
        for comp in wall_comps:
            r0, c0, r1, c1 = _bbox(comp)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if comp.shape[0] != h * w or (h != 1 and w != 1):
                return False

        sims = [_simulate_quantum(obj, wall_mask, grid.shape) for obj in objects]
        if {sim["stop_reason"] for sim in sims} != {"wall", "boundary"}:
            return False
        if any(sim["n_steps"] < 1 for sim in sims):
            return False

        corridors: list[set[tuple[int, int]]] = []
        for obj, sim in zip(objects, sims):
            corridor = _corridor_cells(obj, sim["n_steps"])
            for prev in corridors:
                if corridor & prev:
                    return False
            corridors.append(corridor)

        if not any(_simulate_glide(obj, wall_mask, grid.shape) != sim["top_left"] for obj, sim in zip(objects, sims)):
            return False

        expected = np.full_like(grid, bg)
        expected[wall_mask] = WALL
        for obj, sim in zip(objects, sims):
            r, c = sim["top_left"]
            _paint_rect(expected, r, c, obj["h"], obj["w"], obj["color"])

        if np.array_equal(expected, grid):
            return False
        if not np.array_equal(main(grid), expected):
            return False
        return True
    except AssertionError:
        return False


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_ATTEMPTS} attempts")
