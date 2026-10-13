from __future__ import annotations

import random

import numpy as np


BG = 0
MAGENTA = 6
COLOR_POOL = (1, 2, 3, 4, 5, 7, 8, 9)

GRID_MIN = 12
GRID_MAX = 30
RECT_COUNT_MIN = 2
RECT_COUNT_MAX = 5
RECT_SIZE_MIN = 3
MARKED_RECT_SIZE_MIN = 4
RECT_SIZE_MAX = 9
PLACEMENT_TRIES = 120
GEN_RETRIES = 300


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    arr = np.asarray(points, dtype=int)
    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    return int(mins[0]), int(mins[1]), int(maxs[0]), int(maxs[1])


def _find_rectangles(grid: np.ndarray) -> list[dict[str, object]]:
    rects: list[dict[str, object]] = []
    colors = sorted(set(map(int, grid.flat)) - {BG, MAGENTA})
    for color in colors:
        for pts in _components(grid == int(color)):
            r0, c0, r1, c1 = _bbox(pts)
            if r1 - r0 + 1 < RECT_SIZE_MIN or c1 - c0 + 1 < RECT_SIZE_MIN:
                continue
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            if np.all((patch == int(color)) | (patch == MAGENTA)):
                rects.append({"color": int(color), "bbox": (int(r0), int(c0), int(r1), int(c1))})
    rects.sort(key=lambda rect: tuple(rect["bbox"]))  # stable ownership for repeated markers
    return rects


def _hollow_rectangle(out: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    out[r0 : r1 + 1, c0 : c1 + 1] = BG
    out[r0, c0 : c1 + 1] = int(color)
    out[r1, c0 : c1 + 1] = int(color)
    out[r0 : r1 + 1, c0] = int(color)
    out[r0 : r1 + 1, c1] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Hollow the colored rectangles whose row/column projection contains a magenta marker.

    Input structure:
    - Background is Black.
    - The grid contains several non-overlapping solid rectangles, each with a unique non-black, non-magenta color.
    - Some rectangles are punctured by one or more isolated Magenta single cells inside their bounding boxes; unmarked rectangles remain fully solid.

    Rule:
    - Detect every colored rectangle by its monochrome component and rectangular bounding box.
    - For each Magenta marker, find the unique rectangle whose row span and column span both contain that marker.
    - Remove all Magenta markers.
    - Every rectangle claimed by at least one marker is transformed from solid to hollow: keep its outer one-pixel border and turn its interior Black.
    - Rectangles not claimed by any marker remain solid.

    Key insight:
    - The marker does not match by color or adjacency; it selects the rectangle whose row and column ranges intersect at the marker cell, and that selected rectangle is the one that gets hollowed.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    out[out == MAGENTA] = BG

    rects = _find_rectangles(g)
    claimed: set[int] = set()
    for mr, mc in np.argwhere(g == MAGENTA):
        matches = []
        for idx, rect in enumerate(rects):
            r0, c0, r1, c1 = rect["bbox"]
            if int(r0) <= int(mr) <= int(r1) and int(c0) <= int(mc) <= int(c1):
                matches.append(int(idx))
        assert len(matches) == 1, "each marker must belong to exactly one rectangle"
        claimed.add(matches[0])

    for idx in claimed:
        rect = rects[idx]
        _hollow_rectangle(out, rect["bbox"], int(rect["color"]))

    return out


def _can_place(mask: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    rr0 = max(0, int(r0) - 1)
    cc0 = max(0, int(c0) - 1)
    rr1 = min(mask.shape[0], int(r0) + int(h) + 1)
    cc1 = min(mask.shape[1], int(c0) + int(w) + 1)
    return not bool(mask[rr0:rr1, cc0:cc1].any())


def _sample_marker_cells(
    bbox: tuple[int, int, int, int],
    rng: random.Random,
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    interior = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    if not interior:
        return []

    h = r1 - r0 + 1
    w = c1 - c0 + 1
    target = 1
    if min(h, w) >= 5 and rng.random() < 0.45:
        target = 2
    if min(h, w) >= 7 and rng.random() < 0.25:
        target = 3

    candidates = interior[:]
    rng.shuffle(candidates)
    chosen: list[tuple[int, int]] = []
    for cell in candidates:
        if all(max(abs(cell[0] - rr), abs(cell[1] - cc)) > 1 for rr, cc in chosen):
            chosen.append(cell)
            if len(chosen) == target:
                break
    return chosen[:1] if not chosen else chosen


def _grid_rect_count(h: int, w: int) -> int:
    area = h * w
    if area < 190:
        hi = 3
    elif area < 320:
        hi = 4
    else:
        hi = RECT_COUNT_MAX
    return hi


def _validate_generated(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False

    rects = _find_rectangles(grid)
    if len(rects) < 2:
        return False

    markers = [tuple(map(int, pt)) for pt in np.argwhere(grid == MAGENTA)]
    if not markers:
        return False

    claimed: set[int] = set()
    for mr, mc in markers:
        matches = []
        for idx, rect in enumerate(rects):
            r0, c0, r1, c1 = rect["bbox"]
            if int(r0) < mr < int(r1) and int(c0) < mc < int(c1):
                matches.append(int(idx))
        if len(matches) != 1:
            return False
        idx = matches[0]
        claimed.add(idx)
        r0, c0, r1, c1 = rects[idx]["bbox"]
        interior_area = (int(r1) - int(r0) - 1) * (int(c1) - int(c0) - 1)
        marker_count = int(np.sum(grid[r0 : r1 + 1, c0 : c1 + 1] == MAGENTA))
        if interior_area <= marker_count:
            return False

    if not (1 <= len(claimed) < len(rects)):
        return False

    try:
        out = main(grid)
    except Exception:
        return False
    return isinstance(out, np.ndarray) and out.shape == grid.shape and not np.array_equal(out, grid)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(GRID_MIN, GRID_MAX)
    w = rng.randint(GRID_MIN, GRID_MAX)
    rect_count = rng.randint(RECT_COUNT_MIN, _grid_rect_count(h, w))
    marked_count = rng.randint(1, rect_count - 1)
    marked_slots = set(rng.sample(range(rect_count), marked_count))
    colors = rng.sample(COLOR_POOL, rect_count)

    specs: list[dict[str, int | bool]] = []
    for idx in range(rect_count):
        marked = idx in marked_slots
        min_side = MARKED_RECT_SIZE_MIN if marked else RECT_SIZE_MIN
        max_h = min(RECT_SIZE_MAX, h - 2)
        max_w = min(RECT_SIZE_MAX, w - 2)
        if max_h < min_side or max_w < min_side:
            return None
        rect_h = rng.randint(min_side, max_h)
        rect_w = rng.randint(min_side, max_w)
        specs.append(
            {
                "marked": marked,
                "color": int(colors[idx]),
                "h": int(rect_h),
                "w": int(rect_w),
            }
        )
    specs.sort(key=lambda spec: int(spec["h"]) * int(spec["w"]), reverse=True)

    grid = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    rects: list[dict[str, object]] = []

    for spec in specs:
        rect_h = int(spec["h"])
        rect_w = int(spec["w"])
        placed = False
        for _ in range(PLACEMENT_TRIES):
            r0 = rng.randint(1, h - rect_h - 1)
            c0 = rng.randint(1, w - rect_w - 1)
            if not _can_place(occupied, r0, c0, rect_h, rect_w):
                continue
            grid[r0 : r0 + rect_h, c0 : c0 + rect_w] = int(spec["color"])
            occupied[max(0, r0 - 1) : min(h, r0 + rect_h + 1), max(0, c0 - 1) : min(w, c0 + rect_w + 1)] = True
            rects.append(
                {
                    "marked": bool(spec["marked"]),
                    "color": int(spec["color"]),
                    "bbox": (int(r0), int(c0), int(r0 + rect_h - 1), int(c0 + rect_w - 1)),
                }
            )
            placed = True
            break
        if not placed:
            return None

    multi_marker_used = False
    for rect in rects:
        if not bool(rect["marked"]):
            continue
        markers = _sample_marker_cells(rect["bbox"], rng)
        if not markers:
            return None
        if len(markers) > 1:
            multi_marker_used = True
        for r, c in markers:
            grid[r, c] = MAGENTA

    if not multi_marker_used and any(bool(rect["marked"]) for rect in rects):
        big_marked = [rect for rect in rects if bool(rect["marked"]) and min(rect["bbox"][2] - rect["bbox"][0] + 1, rect["bbox"][3] - rect["bbox"][1] + 1) >= 5]
        if big_marked and rng.random() < 0.35:
            rect = rng.choice(big_marked)
            existing = {(r, c) for r, c in np.argwhere(grid == MAGENTA) if rect["bbox"][0] <= r <= rect["bbox"][2] and rect["bbox"][1] <= c <= rect["bbox"][3]}
            extra = _sample_marker_cells(rect["bbox"], rng)
            for cell in extra:
                if cell not in existing:
                    grid[cell] = MAGENTA
                    break

    return grid if _validate_generated(grid) else None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to produce a valid sample")
