from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
SIDES = ("left", "right", "top", "bottom")

# --------------- shared geometry ---------------


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[np.argmax(counts)])


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_for_color(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _monochromatic_components(grid: np.ndarray, bg: int) -> list[dict]:
    comps: list[dict] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for coords in _components_for_color(grid == color):
            r0, c0, r1, c1 = _bbox(coords)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(len(coords)),
                    "pos": min(coords),
                }
            )
    comps.sort(key=lambda comp: comp["pos"])
    return comps


def _decode_arrow(coords: list[tuple[int, int]]) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    rel = {(r - r0, c - c0) for r, c in coords}
    candidates: list[dict] = []

    for row in range(r1 - r0 + 1):
        cols = sorted(c for r, c in rel if r == row)
        if len(cols) == 3 and cols == list(range(cols[0], cols[0] + 3)):
            line = {(row, cc) for cc in cols}
            extra = list(rel - line)
            if len(extra) == 1:
                er, ec = extra[0]
                if abs(er - row) + abs(ec - cols[0]) == 1:
                    candidates.append(
                        {
                            "direction": "left",
                            "front": (int(r0 + row), int(c0 + cols[0])),
                        }
                    )
                if abs(er - row) + abs(ec - cols[2]) == 1:
                    candidates.append(
                        {
                            "direction": "right",
                            "front": (int(r0 + row), int(c0 + cols[2])),
                        }
                    )

    for col in range(c1 - c0 + 1):
        rows = sorted(r for r, c in rel if c == col)
        if len(rows) == 3 and rows == list(range(rows[0], rows[0] + 3)):
            line = {(rr, col) for rr in rows}
            extra = list(rel - line)
            if len(extra) == 1:
                er, ec = extra[0]
                if abs(er - rows[0]) + abs(ec - col) == 1:
                    candidates.append(
                        {
                            "direction": "up",
                            "front": (int(r0 + rows[0]), int(c0 + col)),
                        }
                    )
                if abs(er - rows[2]) + abs(ec - col) == 1:
                    candidates.append(
                        {
                            "direction": "down",
                            "front": (int(r0 + rows[2]), int(c0 + col)),
                        }
                    )

    unique = {
        (cand["direction"], cand["front"][0], cand["front"][1]) for cand in candidates
    }
    if len(unique) != 1:
        return None
    direction, fr, fc = next(iter(unique))
    return {"direction": direction, "front": (int(fr), int(fc))}


def _extract_scene(grid: np.ndarray) -> tuple[int, list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    rects: list[dict] = []
    arrows: list[dict] = []

    for comp in _monochromatic_components(g, bg):
        r0, c0, r1, c1 = comp["bbox"]
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if int(comp["area"]) == h * w:
            rects.append(
                {
                    "color": int(comp["color"]),
                    "coords": comp["coords"],
                    "bbox": comp["bbox"],
                    "pos": comp["pos"],
                }
            )
            continue

        arrow = _decode_arrow(comp["coords"])
        assert arrow is not None, "every non-rectangle component must be a valid arrow"
        arrows.append(
            {
                "color": int(comp["color"]),
                "coords": comp["coords"],
                "bbox": comp["bbox"],
                "pos": comp["pos"],
                "direction": arrow["direction"],
                "front": arrow["front"],
            }
        )

    rects.sort(key=lambda rect: rect["pos"])
    arrows.sort(key=lambda arrow: arrow["pos"])
    assert rects, "expected at least one solid rectangle"
    assert arrows, "expected at least one arrow"
    return bg, rects, arrows


def _arrow_target_index(arrow: dict, rects: list[dict]) -> int:
    fr, fc = arrow["front"]
    direction = arrow["direction"]
    hits: list[tuple[int, int]] = []
    for idx, rect in enumerate(rects):
        r0, c0, r1, c1 = rect["bbox"]
        dist: int | None = None
        if direction == "right" and r0 <= fr <= r1 and fc < c0:
            dist = int(c0 - fc - 1)
        elif direction == "left" and r0 <= fr <= r1 and c1 < fc:
            dist = int(fc - c1 - 1)
        elif direction == "down" and c0 <= fc <= c1 and fr < r0:
            dist = int(r0 - fr - 1)
        elif direction == "up" and c0 <= fc <= c1 and r1 < fr:
            dist = int(fr - r1 - 1)
        if dist is not None and dist >= 0:
            hits.append((dist, idx))
    assert hits, "each arrow must point to a rectangle"
    hits.sort()
    best_dist = hits[0][0]
    best = [idx for dist, idx in hits if dist == best_dist]
    assert len(best) == 1, "arrow-to-rectangle match must be unambiguous"
    return int(best[0])


def _group_layers(arrows: list[dict]) -> list[dict]:
    by_color: dict[int, list[dict]] = defaultdict(list)
    for arrow in arrows:
        by_color[int(arrow["color"])].append(arrow)

    groups: list[dict] = []
    for color, items in by_color.items():
        groups.append(
            {
                "color": int(color),
                "count": int(len(items)),
                "first_pos": min(item["pos"] for item in items),
            }
        )

    groups.sort(key=lambda item: (item["count"], item["first_pos"][0], item["first_pos"][1], item["color"]))

    # Equal counts need a stable inner/outer order. Push tied layers outward by the
    # smallest possible amount so every visible ring remains distinct.
    prev_gap = -1
    for group in groups:
        gap = int(group["count"])
        if gap <= prev_gap:
            gap = prev_gap + 1
        group["gap"] = int(gap)
        prev_gap = int(gap)
    return groups


def _draw_ring(out: np.ndarray, bbox: tuple[int, int, int, int], gap: int, color: int) -> None:
    r0, c0, r1, c1 = bbox
    top = int(r0 - gap - 1)
    bottom = int(r1 + gap + 1)
    left = int(c0 - gap - 1)
    right = int(c1 + gap + 1)
    assert 0 <= top < out.shape[0] and 0 <= bottom < out.shape[0]
    assert 0 <= left < out.shape[1] and 0 <= right < out.shape[1]
    out[top, left : right + 1] = int(color)
    out[bottom, left : right + 1] = int(color)
    out[top : bottom + 1, left] = int(color)
    out[top : bottom + 1, right] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Colored arrows point at solid rectangles. For each rectangle, count incoming
    arrows by color and draw colored rectangular rings around the rectangle.

    The ring order is determined by arrow count first, then by the earliest arrow
    position for tie-breaking. All arrows are removed in the output.
    """
    g = np.asarray(grid, dtype=int)
    bg, rects, arrows = _extract_scene(g)

    arrows_by_rect: dict[int, list[dict]] = defaultdict(list)
    for arrow in arrows:
        arrows_by_rect[_arrow_target_index(arrow, rects)].append(arrow)

    out = g.copy()
    for arrow in arrows:
        for r, c in arrow["coords"]:
            out[r, c] = int(bg)

    for idx, rect in enumerate(rects):
        layers = _group_layers(arrows_by_rect.get(idx, []))
        for layer in layers:
            _draw_ring(out, rect["bbox"], int(layer["gap"]), int(layer["color"]))

    return out


# --------------- generator ---------------

GRID_HEIGHT_RANGE = (20, 30)
GRID_WIDTH_RANGE = (20, 30)
OBJECT_COUNT_CHOICES = (2, 3)
RECT_HEIGHT_RANGE = (2, 4)
RECT_WIDTH_RANGE = (2, 5)
ARROW_GAP_RANGE = (1, 3)
GENERATION_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 200

SINGLE_PATTERNS = ((1,), (2,), (3,))
DISTINCT_PATTERNS = ((1, 2), (1, 3))
TIE_PATTERNS = ((1, 1), (2, 2), (1, 1, 2))
ALL_PATTERNS = SINGLE_PATTERNS + DISTINCT_PATTERNS + TIE_PATTERNS


def _arrow_cells(direction: str, front_r: int, front_c: int, bend: int) -> list[tuple[int, int]]:
    if direction == "right":
        return [(front_r, front_c - 2), (front_r, front_c - 1), (front_r, front_c), (front_r + bend, front_c)]
    if direction == "left":
        return [(front_r, front_c), (front_r, front_c + 1), (front_r, front_c + 2), (front_r + bend, front_c)]
    if direction == "down":
        return [(front_r - 2, front_c), (front_r - 1, front_c), (front_r, front_c), (front_r, front_c + bend)]
    assert direction == "up"
    return [(front_r, front_c), (front_r + 1, front_c), (front_r + 2, front_c), (front_r, front_c + bend)]


def _pattern_for_mode(mode: str) -> tuple[int, ...]:
    if mode == "single":
        return random.choice(SINGLE_PATTERNS)
    if mode == "distinct":
        return random.choice(DISTINCT_PATTERNS)
    assert mode == "tie"
    return random.choice(TIE_PATTERNS)


def _build_plan(
    object_color: int,
    arrow_colors: list[int],
    mode: str | None = None,
) -> dict:
    h = random.randint(*RECT_HEIGHT_RANGE)
    w = random.randint(*RECT_WIDTH_RANGE)
    pattern = _pattern_for_mode(mode) if mode is not None else random.choice(ALL_PATTERNS)
    total_arrows = int(sum(pattern))
    chosen_sides = random.sample(SIDES, total_arrows)
    random.shuffle(chosen_sides)

    group_colors = random.sample(arrow_colors, len(pattern))
    rect_cells = [(r, c) for r in range(h) for c in range(w)]
    arrows: list[dict] = []

    side_iter = iter(chosen_sides)
    for color, count in zip(group_colors, pattern):
        for _ in range(count):
            side = next(side_iter)
            gap = random.randint(*ARROW_GAP_RANGE)
            bend = random.choice((-1, 1))
            if side == "left":
                row = random.randrange(h)
                cells = _arrow_cells("right", row, -gap - 1, bend)
            elif side == "right":
                row = random.randrange(h)
                cells = _arrow_cells("left", row, w + gap, bend)
            elif side == "top":
                col = random.randrange(w)
                cells = _arrow_cells("down", -gap - 1, col, bend)
            else:
                col = random.randrange(w)
                cells = _arrow_cells("up", h + gap, col, bend)
            arrows.append({"color": int(color), "coords": cells, "pos": min(cells)})

    used = set(rect_cells)
    for arrow in arrows:
        for cell in arrow["coords"]:
            assert cell not in used, "arrow/object overlap is not allowed"
            used.add(cell)

    outer_gap = max(layer["gap"] for layer in _group_layers(arrows))
    future_box = (-outer_gap - 1, -outer_gap - 1, h + outer_gap, w + outer_gap)
    all_cells = rect_cells[:]
    for arrow in arrows:
        all_cells.extend(arrow["coords"])
    ar0, ac0, ar1, ac1 = _bbox(all_cells)
    pr0 = min(future_box[0], ar0) - 1
    pc0 = min(future_box[1], ac0) - 1
    pr1 = max(future_box[2], ar1) + 1
    pc1 = max(future_box[3], ac1) + 1

    paint_cells = [(r, c, int(object_color)) for r, c in rect_cells]
    for arrow in arrows:
        paint_cells.extend((r, c, int(arrow["color"])) for r, c in arrow["coords"])

    return {
        "paint_cells": paint_cells,
        "pack_bbox": (int(pr0), int(pc0), int(pr1), int(pc1)),
    }


def _place_plan(grid: np.ndarray, taken: np.ndarray, plan: dict) -> bool:
    h, w = grid.shape
    r0, c0, r1, c1 = plan["pack_bbox"]
    row_lo = int(-r0)
    row_hi = int(h - 1 - r1)
    col_lo = int(-c0)
    col_hi = int(w - 1 - c1)
    if row_lo > row_hi or col_lo > col_hi:
        return False

    for _ in range(PLACEMENT_ATTEMPTS):
        base_r = random.randint(row_lo, row_hi)
        base_c = random.randint(col_lo, col_hi)
        rr0 = base_r + r0
        cc0 = base_c + c0
        rr1 = base_r + r1
        cc1 = base_c + c1
        if np.any(taken[rr0 : rr1 + 1, cc0 : cc1 + 1]):
            continue

        for r, c, color in plan["paint_cells"]:
            grid[base_r + r, base_c + c] = int(color)
        taken[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
        return True
    return False


def _attempt_input() -> np.ndarray | None:
    bg = random.randrange(10)
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    n_objects = random.choice(OBJECT_COUNT_CHOICES)
    mode = random.choice(("single", "distinct", "tie"))

    available = [color for color in range(10) if color != bg]
    random.shuffle(available)
    object_colors = available[:n_objects]
    arrow_pool = available[n_objects:]
    if len(arrow_pool) < 3:
        return None

    plans: list[dict] = []
    for i in range(n_objects):
        forced_mode = mode if i == 0 else None
        try:
            plans.append(_build_plan(int(object_colors[i]), arrow_pool, forced_mode))
        except ValueError:
            return None

    random.shuffle(plans)
    grid = np.full((h, w), int(bg), dtype=int)
    taken = np.zeros((h, w), dtype=bool)
    for plan in plans:
        if not _place_plan(grid, taken, plan):
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        try:
            grid = _attempt_input()
            if grid is None:
                continue
            out = main(grid)
            if out.shape != grid.shape:
                continue
            if np.array_equal(out, grid):
                continue
            return grid
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid input")
