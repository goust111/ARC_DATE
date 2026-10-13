from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5

BBox = tuple[int, int, int, int]

STRUCT4_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

PATTERN_SIDES = {
    "TL": ("top", "left"),
    "TR": ("top", "right"),
    "BL": ("bottom", "left"),
    "BR": ("bottom", "right"),
}

# Consecutive members expand by one cell while keeping the opposite missing corner fixed.
PATTERN_STEPS = {
    "TL": (-1, -1, 0, 0),
    "TR": (-1, 0, 0, 1),
    "BL": (0, -1, 1, 0),
    "BR": (0, 0, 1, 1),
}

GEN_H_RANGE = (13, 19)
GEN_W_RANGE = (13, 19)
GEN_BASE_LONG_SIDE = (2, 5)
GEN_RED_COUNT = (2, 3)
GEN_RED_SIDE = (2, 3)
GEN_SINGLETON_COLORS = [BLUE, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 500

_CATALOG_CACHE: dict[tuple[int, int], list[dict]] = {}


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    for dr, dc in STRUCT4_DIRS:
        yield r + dr, c + dc


def _bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(grid: np.ndarray, color: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if (
                        _in_bounds((h, w), nr, nc)
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": _bbox_coords(arr),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _box_hw(box: BBox) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _add_box(box: BBox, delta: BBox, mul: int = 1) -> BBox:
    return tuple(int(box[i] + mul * delta[i]) for i in range(4))


def _stroke_mask(shape: tuple[int, int], box: BBox, pattern: str) -> np.ndarray:
    h, w = shape
    r0, c0, r1, c1 = box
    if r1 - r0 + 1 < 2 or c1 - c0 + 1 < 2:
        return np.zeros(shape, dtype=bool)
    out = np.zeros(shape, dtype=bool)
    cc0 = max(0, c0)
    cc1 = min(w - 1, c1)
    rr0 = max(0, r0)
    rr1 = min(h - 1, r1)
    if cc0 <= cc1:
        if "top" in PATTERN_SIDES[pattern] and 0 <= r0 < h:
            out[r0, cc0 : cc1 + 1] = True
        if "bottom" in PATTERN_SIDES[pattern] and 0 <= r1 < h:
            out[r1, cc0 : cc1 + 1] = True
    if rr0 <= rr1:
        if "left" in PATTERN_SIDES[pattern] and 0 <= c0 < w:
            out[rr0 : rr1 + 1, c0] = True
        if "right" in PATTERN_SIDES[pattern] and 0 <= c1 < w:
            out[rr0 : rr1 + 1, c1] = True
    return out


def _sequence_boxes(shape: tuple[int, int], base_box: BBox, pattern: str) -> list[BBox]:
    step = PATTERN_STEPS[pattern]
    boxes: list[BBox] = []
    cur = base_box
    limit = 2 * (shape[0] + shape[1]) + 10
    for _ in range(limit):
        mask = _stroke_mask(shape, cur, pattern)
        if not np.any(mask):
            break
        boxes.append(cur)
        cur = _add_box(cur, step)
    return boxes


def _candidate_union(shape: tuple[int, int], boxes: list[BBox], pattern: str) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    for box in boxes:
        out |= _stroke_mask(shape, box, pattern)
    return out


def _base_candidates(shape: tuple[int, int], pattern: str):
    h, w = shape
    step = PATTERN_STEPS[pattern]
    for hh in range(2, GEN_BASE_LONG_SIDE[1] + 1):
        for ww in range(2, GEN_BASE_LONG_SIDE[1] + 1):
            if min(hh, ww) != 2:
                continue
            for r0 in range(h - hh + 1):
                for c0 in range(w - ww + 1):
                    box = (r0, c0, r0 + hh - 1, c0 + ww - 1)
                    prev_box = _add_box(box, step, -1)
                    if np.any(_stroke_mask(shape, prev_box, pattern)):
                        continue
                    yield box


def _repeat_color_mask(grid: np.ndarray) -> np.ndarray:
    repeated = set()
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != BLACK):
        if len(_find_components(grid, color)) >= 2:
            repeated.add(color)
    return np.isin(grid, list(repeated))


def _family_catalog(shape: tuple[int, int]) -> list[dict]:
    if shape in _CATALOG_CACHE:
        return _CATALOG_CACHE[shape]

    catalog: list[dict] = []
    for pattern in PATTERN_SIDES:
        for base_box in _base_candidates(shape, pattern):
            boxes = _sequence_boxes(shape, base_box, pattern)
            if len(boxes) < 4:
                continue
            catalog.append(
                {
                    "pattern": pattern,
                    "base_box": base_box,
                    "boxes": boxes,
                    "union": _candidate_union(shape, boxes, pattern),
                }
            )
    _CATALOG_CACHE[shape] = catalog
    return catalog


def _find_family_candidates(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    green_mask = g == GREEN
    repeated_mask = _repeat_color_mask(g)
    allowed = green_mask | repeated_mask
    blocked = (~allowed) & (g != BLACK)
    candidates = []
    for item in _family_catalog(g.shape):
        pattern = item["pattern"]
        boxes = item["boxes"]
        union = item["union"]
        if not np.any(union & repeated_mask):
            continue
        if np.any(union & blocked):
            continue
        if np.any(union & ~allowed):
            continue
        if not np.all(green_mask <= union):
            continue
        full_visible = sum(
            1
            for box in boxes
            if 0 <= box[0] <= box[2] < g.shape[0] and 0 <= box[1] <= box[3] < g.shape[1]
        )
        visible_members = sum(int(np.any(_stroke_mask(g.shape, box, pattern) & green_mask)) for box in boxes)
        hidden_members = sum(int(np.any(_stroke_mask(g.shape, box, pattern) & repeated_mask)) for box in boxes)
        candidates.append(
            {
                "pattern": pattern,
                "base_box": item["base_box"],
                "boxes": boxes,
                "union": union,
                "score": (
                    int(visible_members),
                    int(hidden_members),
                    int(full_visible),
                    -int(union.sum()),
                ),
            }
        )
    if not candidates:
        return []
    best = max(c["score"] for c in candidates)
    return [c for c in candidates if c["score"] == best]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase the repeated-color occluders, infer the hidden arithmetic family of rectangular L-strokes, and grey-fill the remaining background.

    Input structure:
    - A black grid contains green rectangular side-strokes that all use the same adjacent pair of bbox sides.
    - Red solid 4-connected objects appear at least twice and cover parts of some green strokes.
    - There may also be one singleton-color object that is not part of the stroke family.

    Rule:
    - Count 4-connected components for each non-black color and erase every color that appears in two or more components.
    - Treat the surviving green pixels together with the erased repeated-color footprint as evidence for one arithmetic progression of rectangular L-stroke bboxes.
    - Reconstruct the unique maximal progression, draw every visible clipped member in green, keep any singleton-color object unchanged, and recolor all remaining black cells grey.

    Key insight:
    - The repeated color is not decoration: its footprint marks exactly where green pixels may be missing, so the correct stroke progression is the only family whose visible cells fit entirely inside the union of observed green cells and erased occluder cells.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    repeated_mask = _repeat_color_mask(g)
    candidates = _find_family_candidates(g)
    assert candidates, "expected at least one valid stroke family"
    family = candidates[0]

    out = np.full(g.shape, GREY, dtype=int)
    singleton_mask = (g != BLACK) & (g != GREEN) & (~repeated_mask)
    out[singleton_mask] = g[singleton_mask]
    out[family["union"]] = GREEN
    return out


def _connected_count(mask: np.ndarray) -> int:
    g = mask.astype(int)
    return len(_find_components(g, 1))


def _full_output(shape: tuple[int, int], boxes: list[BBox], pattern: str, singleton: list[tuple[int, int, int]]) -> np.ndarray:
    out = np.full(shape, GREY, dtype=int)
    out[_candidate_union(shape, boxes, pattern)] = GREEN
    for r, c, color in singleton:
        out[r, c] = color
    return out


def _boxes_full_visible(shape: tuple[int, int], boxes: list[BBox]) -> int:
    return sum(1 for r0, c0, r1, c1 in boxes if 0 <= r0 <= r1 < shape[0] and 0 <= c0 <= c1 < shape[1])


def _corner_rect(box: BBox, pattern: str, rh: int, rw: int) -> BBox:
    r0, c0, r1, c1 = box
    if pattern == "TL":
        return r0, c0, r0 + rh - 1, c0 + rw - 1
    if pattern == "TR":
        return r0, c1 - rw + 1, r0 + rh - 1, c1
    if pattern == "BL":
        return r1 - rh + 1, c0, r1, c0 + rw - 1
    return r1 - rh + 1, c1 - rw + 1, r1, c1


def _rect_cells(box: BBox) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _rect_valid(shape: tuple[int, int], box: BBox) -> bool:
    r0, c0, r1, c1 = box
    return 0 <= r0 <= r1 < shape[0] and 0 <= c0 <= c1 < shape[1]


def _boxes_touch_or_overlap(a: BBox, b: BBox) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + 1 < br0 or br1 + 1 < ar0 or ac1 + 1 < bc0 or bc1 + 1 < ac0)


def _sample_singleton(shape: tuple[int, int], forbidden: np.ndarray) -> list[tuple[int, int, int]]:
    if random.random() < 0.45:
        return []
    h, w = shape
    border = []
    for r in range(h):
        for c in range(w):
            if r in (0, h - 1) or c in (0, w - 1):
                if not forbidden[r, c]:
                    border.append((r, c))
    if not border:
        return []
    r, c = random.choice(border)
    return [(r, c, random.choice(GEN_SINGLETON_COLORS))]


def _render_input(shape: tuple[int, int], boxes: list[BBox], pattern: str, red_boxes: list[BBox], singleton: list[tuple[int, int, int]]) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    grid[_candidate_union(shape, boxes, pattern)] = GREEN
    for r, c, color in singleton:
        grid[r, c] = color
    for box in red_boxes:
        for r, c in _rect_cells(box):
            grid[r, c] = RED
    return grid


def _hidden_member_stats(shape: tuple[int, int], boxes: list[BBox], pattern: str, grid: np.ndarray) -> tuple[int, int]:
    hidden_partial = 0
    fully_visible = 0
    for box in boxes:
        mask = _stroke_mask(shape, box, pattern)
        green_here = int(np.sum(mask & (grid == GREEN)))
        red_here = int(np.sum(mask & (grid == RED)))
        if red_here > 0 and green_here > 0:
            hidden_partial += 1
        if red_here == 0:
            fully_visible += 1
    return hidden_partial, fully_visible


def _gen_attempt() -> np.ndarray | None:
    shape = (
        random.randint(*GEN_H_RANGE),
        random.randint(*GEN_W_RANGE),
    )
    pattern = random.choice(list(PATTERN_SIDES))
    long_side = random.randint(*GEN_BASE_LONG_SIDE)
    if random.random() < 0.5:
        hh, ww = 2, long_side
    else:
        hh, ww = long_side, 2
    h, w = shape
    if hh > h or ww > w:
        return None

    candidates = [c for c in _family_catalog(shape) if c["pattern"] == pattern]
    random.shuffle(candidates)
    for item in candidates[:160]:
        base_box = item["base_box"]
        bh, bw = _box_hw(base_box)
        if (bh, bw) != (hh, ww):
            continue
        boxes = item["boxes"]
        if not 4 <= len(boxes) <= 7:
            continue
        if _boxes_full_visible(shape, boxes) < 2:
            continue

        full_union = item["union"]
        if _connected_count(full_union) != 1:
            continue

        singleton = _sample_singleton(shape, full_union)
        singleton_mask = np.zeros(shape, dtype=bool)
        for r, c, _color in singleton:
            singleton_mask[r, c] = True

        candidate_reds = []
        middle = list(range(1, max(2, len(boxes) - 1)))
        random.shuffle(middle)
        for idx in middle:
            for _ in range(6):
                rh = random.randint(*GEN_RED_SIDE)
                rw = random.randint(*GEN_RED_SIDE)
                red_box = _corner_rect(boxes[idx], pattern, rh, rw)
                if not _rect_valid(shape, red_box):
                    continue
                cells = _rect_cells(red_box)
                if not any(full_union[r, c] for r, c in cells):
                    continue
                if any(singleton_mask[r, c] for r, c in cells):
                    continue
                if any(_boxes_touch_or_overlap(red_box, other) for other in candidate_reds):
                    continue
                candidate_reds.append(red_box)
                break
            if len(candidate_reds) >= random.randint(*GEN_RED_COUNT):
                break
        if len(candidate_reds) < 2:
            continue

        inp = _render_input(shape, boxes, pattern, candidate_reds, singleton)
        if len(_find_components(inp, RED)) < 2:
            continue
        if len(_find_components(inp, GREEN)) != 1:
            continue

        hidden_partial, fully_visible = _hidden_member_stats(shape, boxes, pattern, inp)
        if hidden_partial < 1 or fully_visible < 2:
            continue

        candidates = _find_family_candidates(inp)
        if len(candidates) != 1:
            continue

        out = _full_output(shape, boxes, pattern, singleton)
        try:
            pred = main(inp)
        except Exception:
            continue
        if not np.array_equal(pred, out):
            continue
        if np.array_equal(inp, out):
            continue
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
