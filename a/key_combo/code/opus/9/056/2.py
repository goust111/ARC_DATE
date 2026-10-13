from __future__ import annotations

import itertools
from collections import deque

import numpy as np


BACKGROUND = 0
INDICATOR = 2
RECTANGLE = 3
DIRS = ("up", "down", "left", "right")


def _overlap_1d(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(np.array(pts, dtype=int))
    return components


def _middle_values(start: int, length: int) -> list[int]:
    if length % 2 == 1:
        return [start + length // 2]
    mid = start + length // 2
    return [mid - 1, mid]


def _direction_from_indicator(
    red_r: int,
    red_c: int,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
) -> str:
    candidates = []
    if red_r == r0:
        candidates.append("up")
    if red_r == r1:
        candidates.append("down")
    if red_c == c0:
        candidates.append("left")
    if red_c == c1:
        candidates.append("right")
    assert len(candidates) == 1, "indicator must lie on exactly one side of its rectangle"
    return candidates[0]


def _indicator_coord(rect: dict[str, int], direction: str, rng: np.random.Generator) -> tuple[int, int]:
    r0, c0, r1, c1 = rect["r0"], rect["c0"], rect["r1"], rect["c1"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if direction in ("up", "down"):
        c = int(rng.choice(_middle_values(c0, w)))
        return (r0 if direction == "up" else r1), c
    r = int(rng.choice(_middle_values(r0, h)))
    return r, (c0 if direction == "left" else c1)


def _direction_meta(
    rects: list[dict[str, int]],
    idx: int,
    direction: str,
    shape: tuple[int, int],
) -> dict[str, object]:
    h, w = shape
    rect = rects[idx]
    r0, c0, r1, c1 = rect["r0"], rect["c0"], rect["r1"], rect["c1"]
    blocker = None
    edge_distance = None

    if direction == "up":
        edge_distance = r0
        for j, other in enumerate(rects):
            if j == idx:
                continue
            if other["r1"] < r0 and _overlap_1d(c0, c1, other["c0"], other["c1"]):
                dist = r0 - other["r1"] - 1
                if blocker is None or dist < blocker[0]:
                    blocker = (dist, j)
        length = blocker[0] if blocker is not None else edge_distance
        rows = range(r0 - 1, r0 - length - 1, -1)
        band = {(r, c) for r in rows for c in range(c0, c1 + 1)}
    elif direction == "down":
        edge_distance = h - 1 - r1
        for j, other in enumerate(rects):
            if j == idx:
                continue
            if other["r0"] > r1 and _overlap_1d(c0, c1, other["c0"], other["c1"]):
                dist = other["r0"] - r1 - 1
                if blocker is None or dist < blocker[0]:
                    blocker = (dist, j)
        length = blocker[0] if blocker is not None else edge_distance
        rows = range(r1 + 1, r1 + length + 1)
        band = {(r, c) for r in rows for c in range(c0, c1 + 1)}
    elif direction == "left":
        edge_distance = c0
        for j, other in enumerate(rects):
            if j == idx:
                continue
            if other["c1"] < c0 and _overlap_1d(r0, r1, other["r0"], other["r1"]):
                dist = c0 - other["c1"] - 1
                if blocker is None or dist < blocker[0]:
                    blocker = (dist, j)
        length = blocker[0] if blocker is not None else edge_distance
        cols = range(c0 - 1, c0 - length - 1, -1)
        band = {(r, c) for r in range(r0, r1 + 1) for c in cols}
    else:
        edge_distance = w - 1 - c1
        for j, other in enumerate(rects):
            if j == idx:
                continue
            if other["c0"] > c1 and _overlap_1d(r0, r1, other["r0"], other["r1"]):
                dist = other["c0"] - c1 - 1
                if blocker is None or dist < blocker[0]:
                    blocker = (dist, j)
        length = blocker[0] if blocker is not None else edge_distance
        cols = range(c1 + 1, c1 + length + 1)
        band = {(r, c) for r in range(r0, r1 + 1) for c in cols}

    return {
        "direction": direction,
        "length": int(length),
        "blocked": blocker is not None,
        "blocker": None if blocker is None else int(blocker[1]),
        "band": band,
    }


def _find_rectangles(grid: np.ndarray) -> list[dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    components = _component_coords(g != BACKGROUND)
    rects: list[dict[str, int]] = []
    for pts in components:
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(sub != BACKGROUND), "each object must fill its bbox"
        colors = set(map(int, np.unique(sub)))
        assert colors <= {INDICATOR, RECTANGLE}, "unexpected foreground color"
        red_pts = np.argwhere(sub == INDICATOR)
        assert len(red_pts) == 1, "each rectangle must contain exactly one indicator"
        red_r, red_c = map(int, red_pts[0])
        abs_red = (r0 + red_r, c0 + red_c)
        direction = _direction_from_indicator(abs_red[0], abs_red[1], r0, c0, r1, c1)
        rects.append(
            {
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
                "red_r": abs_red[0],
                "red_c": abs_red[1],
                "direction": direction,
            }
        )
    rects.sort(key=lambda rect: (rect["r0"], rect["c0"], rect["r1"], rect["c1"]))
    return rects


def _solve_from_rectangles(grid: np.ndarray, rects: list[dict[str, int]]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for idx, rect in enumerate(rects):
        meta = _direction_meta(rects, idx, rect["direction"], out.shape)
        length = int(meta["length"])
        if length <= 0:
            continue
        r0, c0, r1, c1 = rect["r0"], rect["c0"], rect["r1"], rect["c1"]
        if rect["direction"] == "up":
            out[r0 - length : r0, c0 : c1 + 1] = RECTANGLE
            for step in range(1, length + 1):
                out[rect["red_r"] - step, rect["red_c"]] = INDICATOR
        elif rect["direction"] == "down":
            out[r1 + 1 : r1 + length + 1, c0 : c1 + 1] = RECTANGLE
            for step in range(1, length + 1):
                out[rect["red_r"] + step, rect["red_c"]] = INDICATOR
        elif rect["direction"] == "left":
            out[r0 : r1 + 1, c0 - length : c0] = RECTANGLE
            for step in range(1, length + 1):
                out[rect["red_r"], rect["red_c"] - step] = INDICATOR
        else:
            out[r0 : r1 + 1, c1 + 1 : c1 + length + 1] = RECTANGLE
            for step in range(1, length + 1):
                out[rect["red_r"], rect["red_c"] + step] = INDICATOR
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each indicated green rectangle until another rectangle blocks the sightline or the canvas edge.

    Input structure:
    - The grid has a black background with 3-5 non-touching solid green rectangles.
    - Each rectangle contains exactly one red border pixel that replaces one green pixel on a single side.
    - The red pixel is the only direction marker; other rectangles act as potential blockers.

    Rule:
    - Detect every rectangle-and-indicator object and recover its full bounding box.
    - The side containing the red pixel determines the extension direction: top=up, bottom=down, left=left, right=right.
    - Extend a green band with the rectangle's full orthogonal thickness until the next step would leave the canvas or enter another rectangle's bbox, and do not overwrite the blocking rectangle.
    - Draw a red line from the indicator pixel outward along the same direction for exactly the same length.

    Key insight:
    - The answer needs both pieces of information at once: the indicator chooses the ray direction, while the first aligned rectangle in that ray limits how far the extension can go.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    rects = _find_rectangles(g)
    return _solve_from_rectangles(g, rects)


GEN_ATTEMPTS = 400
GRID_MIN = 14
GRID_MAX = 19
RECT_COUNT_CHOICES = (3, 4, 5)


def _can_place(r0: int, c0: int, h: int, w: int, rects: list[dict[str, int]], shape: tuple[int, int]) -> bool:
    rows, cols = shape
    if r0 < 0 or c0 < 0 or r0 + h > rows or c0 + w > cols:
        return False
    nr0, nc0, nr1, nc1 = r0, c0, r0 + h - 1, c0 + w - 1
    for rect in rects:
        if not (nr1 < rect["r0"] - 1 or nr0 > rect["r1"] + 1 or nc1 < rect["c0"] - 1 or nc0 > rect["c1"] + 1):
            return False
    return True


def _random_size(rng: np.random.Generator) -> tuple[int, int]:
    while True:
        h = int(rng.integers(2, 5))
        w = int(rng.integers(2, 5))
        if h == 2 and w == 2:
            continue
        return h, w


def _place_random_rect(
    rng: np.random.Generator,
    rects: list[dict[str, int]],
    shape: tuple[int, int],
) -> dict[str, int] | None:
    h, w = _random_size(rng)
    rows, cols = shape
    for _ in range(60):
        r0 = int(rng.integers(0, rows - h + 1))
        c0 = int(rng.integers(0, cols - w + 1))
        if _can_place(r0, c0, h, w, rects, shape):
            return {"r0": r0, "c0": c0, "r1": r0 + h - 1, "c1": c0 + w - 1}
    return None


def _place_aligned_rect(
    rng: np.random.Generator,
    anchor: dict[str, int],
    rects: list[dict[str, int]],
    shape: tuple[int, int],
) -> dict[str, int] | None:
    rows, cols = shape
    for _ in range(120):
        h, w = _random_size(rng)
        vertical = bool(rng.integers(0, 2))
        if vertical:
            if rng.integers(0, 2) == 0:
                r0 = anchor["r0"] - int(rng.integers(1, 6)) - h
            else:
                r0 = anchor["r1"] + int(rng.integers(2, 7))
            low = max(0, anchor["c0"] - w + 1)
            high = min(cols - w, anchor["c1"])
            if low > high:
                continue
            c0 = int(rng.integers(low, high + 1))
        else:
            if rng.integers(0, 2) == 0:
                c0 = anchor["c0"] - int(rng.integers(1, 6)) - w
            else:
                c0 = anchor["c1"] + int(rng.integers(2, 7))
            low = max(0, anchor["r0"] - h + 1)
            high = min(rows - h, anchor["r1"])
            if low > high:
                continue
            r0 = int(rng.integers(low, high + 1))
        if _can_place(r0, c0, h, w, rects, shape):
            return {"r0": r0, "c0": c0, "r1": r0 + h - 1, "c1": c0 + w - 1}
    return None


def _place_rectangles(
    rng: np.random.Generator,
    shape: tuple[int, int],
    count: int,
) -> list[dict[str, int]] | None:
    rects: list[dict[str, int]] = []
    first = _place_random_rect(rng, rects, shape)
    if first is None:
        return None
    rects.append(first)

    second = _place_aligned_rect(rng, rects[0], rects, shape)
    if second is None:
        return None
    rects.append(second)

    while len(rects) < count:
        if rng.random() < 0.65:
            anchor = rects[int(rng.integers(0, len(rects)))]
            rect = _place_aligned_rect(rng, anchor, rects, shape)
        else:
            rect = _place_random_rect(rng, rects, shape)
        if rect is None:
            return None
        rects.append(rect)
    return rects


def _valid_assignment(
    rects: list[dict[str, int]],
    directions: tuple[str, ...],
    metas_by_rect: list[dict[str, dict[str, object]]],
) -> tuple[bool, tuple]:
    used_cells: set[tuple[int, int]] = set()
    blocked = 0
    max_length = 0
    for idx, direction in enumerate(directions):
        meta = metas_by_rect[idx][direction]
        if int(meta["length"]) <= 0:
            return False, ()
        band = meta["band"]
        if used_cells & band:
            return False, ()
        used_cells |= band
        blocked += int(bool(meta["blocked"]))
        max_length = max(max_length, int(meta["length"]))
    if blocked == 0 or blocked == len(rects):
        return False, ()
    if max_length < 2:
        return False, ()
    axes = {("v" if direction in ("up", "down") else "h") for direction in directions}
    score = (
        len(axes),
        len(set(directions)),
        min(blocked, len(rects) - blocked),
        sum(int(metas_by_rect[i][direction]["length"]) for i, direction in enumerate(directions)),
    )
    return True, score


def _choose_directions(
    rng: np.random.Generator,
    rects: list[dict[str, int]],
    shape: tuple[int, int],
) -> tuple[tuple[str, ...], list[dict[str, object]]] | None:
    metas_by_rect: list[dict[str, dict[str, object]]] = []
    direction_options: list[list[str]] = []
    for idx, rect in enumerate(rects):
        h = rect["r1"] - rect["r0"] + 1
        w = rect["c1"] - rect["c0"] + 1
        options: list[str] = []
        meta_map: dict[str, dict[str, object]] = {}
        for direction in DIRS:
            if direction in ("up", "down") and w < 3:
                continue
            if direction in ("left", "right") and h < 3:
                continue
            meta = _direction_meta(rects, idx, direction, shape)
            if int(meta["length"]) <= 0:
                continue
            options.append(direction)
            meta_map[direction] = meta
        if not options:
            return None
        metas_by_rect.append(meta_map)
        direction_options.append(options)

    valid: list[tuple[tuple, tuple[str, ...]]] = []
    for directions in itertools.product(*direction_options):
        ok, score = _valid_assignment(rects, directions, metas_by_rect)
        if ok:
            valid.append((score, tuple(directions)))
    if not valid:
        return None

    valid.sort(key=lambda item: item[0], reverse=True)
    top_score = valid[0][0]
    finalists = [directions for score, directions in valid if score == top_score]
    chosen = finalists[int(rng.integers(0, len(finalists)))]
    metas = [metas_by_rect[i][direction] for i, direction in enumerate(chosen)]
    return chosen, metas


def _build_grid(
    rng: np.random.Generator,
    rects: list[dict[str, int]],
    directions: tuple[str, ...],
    shape: tuple[int, int],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for rect, direction in zip(rects, directions):
        grid[rect["r0"] : rect["r1"] + 1, rect["c0"] : rect["c1"] + 1] = RECTANGLE
        red_r, red_c = _indicator_coord(rect, direction, rng)
        grid[red_r, red_c] = INDICATOR
    return grid


def _validate_generated(grid: np.ndarray) -> bool:
    rects = _find_rectangles(grid)
    if not (3 <= len(rects) <= 5):
        return False
    out = main(grid)
    if np.array_equal(grid, out):
        return False
    metas = [_direction_meta(rects, i, rect["direction"], grid.shape) for i, rect in enumerate(rects)]
    blocked = sum(int(bool(meta["blocked"])) for meta in metas)
    if blocked == 0 or blocked == len(rects):
        return False
    used: set[tuple[int, int]] = set()
    for meta in metas:
        band = meta["band"]
        if used & band:
            return False
        used |= band
    if len({rect["direction"] for rect in rects}) < 2:
        return False
    if max(int(meta["length"]) for meta in metas) < 2:
        return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    rows = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    cols = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    count = int(rng.choice(RECT_COUNT_CHOICES))
    rects = _place_rectangles(rng, (rows, cols), count)
    if rects is None:
        return None
    picked = _choose_directions(rng, rects, (rows, cols))
    if picked is None:
        return None
    directions, _metas = picked
    grid = _build_grid(rng, rects, directions, (rows, cols))
    return grid if _validate_generated(grid) else None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
