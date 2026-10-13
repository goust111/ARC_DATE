from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
PATH_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


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
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _grey_rectangles(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    rects: list[tuple[int, int, int, int]] = []
    for comp in _components(grid == GREY):
        r0, c0, r1, c1 = _bbox(comp)
        if len(comp) != (r1 - r0 + 1) * (c1 - c0 + 1):
            continue
        if np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == GREY):
            rects.append((r0, c0, r1, c1))
    rects.sort()
    return rects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct straight path segments that pass through grey occluders.

    Input structure:
    - The background is black.
    - One or more solid grey rectangles act as occluders and do not overlap.
    - Colored paths are one-pixel-wide orthogonal chains that stop at an occluder boundary and reappear on the opposite side.

    Rule:
    - For each grey rectangle, inspect the colored pixels immediately outside its four sides.
    - Pair left/right boundary pixels that share a color and row, and pair top/bottom boundary pixels that share a color and column.
    - Fill the grey cells inside the rectangle along each paired straight row or column with that path color; leave every other cell unchanged.

    Key insight:
    - The visible path fragments already reveal both the matching color and whether the hidden transit is horizontal or vertical, so each missing piece is just a straight opposite-side connection.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for r0, c0, r1, c1 in _grey_rectangles(g):
        left: dict[tuple[int, int], tuple[int, int]] = {}
        right: dict[tuple[int, int], tuple[int, int]] = {}
        top: dict[tuple[int, int], tuple[int, int]] = {}
        bottom: dict[tuple[int, int], tuple[int, int]] = {}

        if c0 > 0:
            for r in range(r0, r1 + 1):
                color = int(g[r, c0 - 1])
                if color not in (BLACK, GREY):
                    left[(color, r)] = (r, c0 - 1)
        if c1 + 1 < g.shape[1]:
            for r in range(r0, r1 + 1):
                color = int(g[r, c1 + 1])
                if color not in (BLACK, GREY):
                    right[(color, r)] = (r, c1 + 1)
        if r0 > 0:
            for c in range(c0, c1 + 1):
                color = int(g[r0 - 1, c])
                if color not in (BLACK, GREY):
                    top[(color, c)] = (r0 - 1, c)
        if r1 + 1 < g.shape[0]:
            for c in range(c0, c1 + 1):
                color = int(g[r1 + 1, c])
                if color not in (BLACK, GREY):
                    bottom[(color, c)] = (r1 + 1, c)

        for color, row in left.keys() & right.keys():
            out[row, c0 : c1 + 1] = color
        for color, col in top.keys() & bottom.keys():
            out[r0 : r1 + 1, col] = color

    return out


# --------------- generator ---------------

GRID_H_RANGE = (14, 26)
GRID_W_RANGE = (14, 26)
RECT_H_RANGE = (3, 7)
RECT_W_RANGE = (3, 7)
RECT_MARGIN = 3
RECT_PAD = 2

GEN_RETRIES = 400
RECT_COUNT_CHOICES = (1, 2, 2, 3)
RECT_PLACE_RETRIES = 80
ENDPOINT_TRIES = 24


def _place_rectangles(h: int, w: int) -> list[tuple[int, int, int, int]] | None:
    max_rect_h = min(RECT_H_RANGE[1], h - 2 * RECT_MARGIN)
    max_rect_w = min(RECT_W_RANGE[1], w - 2 * RECT_MARGIN)
    if max_rect_h < RECT_H_RANGE[0] or max_rect_w < RECT_W_RANGE[0]:
        return None

    count = random.choice(RECT_COUNT_CHOICES)
    rects: list[tuple[int, int, int, int]] = []
    occupied = np.zeros((h, w), dtype=bool)

    for _ in range(count):
        placed = False
        for _ in range(RECT_PLACE_RETRIES):
            rh = random.randint(RECT_H_RANGE[0], max_rect_h)
            rw = random.randint(RECT_W_RANGE[0], max_rect_w)
            r0_hi = h - rh - RECT_MARGIN
            c0_hi = w - rw - RECT_MARGIN
            if r0_hi < RECT_MARGIN or c0_hi < RECT_MARGIN:
                continue
            r0 = random.randint(RECT_MARGIN, r0_hi)
            c0 = random.randint(RECT_MARGIN, c0_hi)
            r1 = r0 + rh - 1
            c1 = c0 + rw - 1

            pr0 = max(0, r0 - RECT_PAD)
            pc0 = max(0, c0 - RECT_PAD)
            pr1 = min(h - 1, r1 + RECT_PAD)
            pc1 = min(w - 1, c1 + RECT_PAD)
            if occupied[pr0 : pr1 + 1, pc0 : pc1 + 1].any():
                continue

            occupied[pr0 : pr1 + 1, pc0 : pc1 + 1] = True
            rects.append((r0, c0, r1, c1))
            placed = True
            break
        if not placed:
            return None

    rects.sort()
    return rects


def _pick_slots(rects: list[tuple[int, int, int, int]]) -> list[tuple[int, str, int, int]] | None:
    slots: list[tuple[int, str, int]] = []
    for idx, (r0, c0, r1, c1) in enumerate(rects):
        orientation = random.choice(("H", "V"))
        values = list(range(r0, r1 + 1)) if orientation == "H" else list(range(c0, c1 + 1))
        random.shuffle(values)
        chosen: list[int] = []
        for value in values:
            if all(abs(value - prev) >= 2 for prev in chosen):
                chosen.append(value)
            if len(chosen) == 2:
                break
        if not chosen:
            continue
        keep = 1 if len(chosen) == 1 else random.choice((1, 2))
        random.shuffle(chosen)
        for value in chosen[:keep]:
            slots.append((idx, orientation, value))

    if len(slots) < 2:
        return None

    random.shuffle(slots)
    total = random.randint(2, min(5, len(slots), len(PATH_COLORS)))
    colors = random.sample(PATH_COLORS, total)
    return [(idx, orientation, value, color) for (idx, orientation, value), color in zip(slots[:total], colors)]


def _touch_mask(rects: list[tuple[int, int, int, int]], h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for r0, c0, r1, c1 in rects:
        if c0 > 0:
            mask[r0 : r1 + 1, c0 - 1] = True
        if c1 + 1 < w:
            mask[r0 : r1 + 1, c1 + 1] = True
        if r0 > 0:
            mask[r0 - 1, c0 : c1 + 1] = True
        if r1 + 1 < h:
            mask[r1 + 1, c0 : c1 + 1] = True
    return mask


def _region_mask(shape: tuple[int, int], side: str, rect: tuple[int, int, int, int]) -> np.ndarray:
    h, w = shape
    r0, c0, r1, c1 = rect
    mask = np.zeros((h, w), dtype=bool)
    if side == "L":
        mask[:, : c0 - 1] = True
    elif side == "R":
        mask[:, c1 + 2 :] = True
    elif side == "T":
        mask[: r0 - 1, :] = True
    else:
        mask[r1 + 2 :, :] = True
    return mask


def _border_candidates(side: str, h: int, w: int) -> list[tuple[int, int]]:
    if side == "L":
        cells = [(r, 0) for r in range(1, h - 1)]
    elif side == "R":
        cells = [(r, w - 1) for r in range(1, h - 1)]
    elif side == "T":
        cells = [(0, c) for c in range(1, w - 1)]
    else:
        cells = [(h - 1, c) for c in range(1, w - 1)]
    random.shuffle(cells)
    return cells


def _reconstruct_path(
    parents: dict[tuple[int, int], tuple[int, int] | None],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = goal
    while cur is not None:
        path.append(cur)
        cur = parents[cur]
    path.reverse()
    return path


def _bfs_path(
    allowed: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]] | None:
    if not allowed[start] or not allowed[goal]:
        return None

    h, w = allowed.shape
    queue: deque[tuple[int, int]] = deque([start])
    parents: dict[tuple[int, int], tuple[int, int] | None] = {start: None}

    while queue:
        r, c = queue.popleft()
        nbrs = list(_neighbors4(r, c))
        random.shuffle(nbrs)
        for nr, nc in nbrs:
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if not allowed[nr, nc] or (nr, nc) in parents:
                continue
            parents[(nr, nc)] = (r, c)
            if (nr, nc) == goal:
                return _reconstruct_path(parents, goal)
            queue.append((nr, nc))
    return None


def _is_simple_path(cells: list[tuple[int, int]]) -> bool:
    unique = set(cells)
    if len(unique) != len(cells) or len(unique) < 2:
        return False
    degrees = []
    for r, c in unique:
        degree = sum((nr, nc) in unique for nr, nc in _neighbors4(r, c))
        degrees.append(degree)
    return degrees.count(1) == 2 and all(degree in (1, 2) for degree in degrees)


def _mark_reserved(reserved: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = reserved.shape
    for r, c in cells:
        reserved[r, c] = True
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w:
                reserved[nr, nc] = True


def _route_component(
    grid: np.ndarray,
    reserved: np.ndarray,
    touch: np.ndarray,
    rect: tuple[int, int, int, int],
    side: str,
    breakpoint: tuple[int, int],
    anchor: tuple[int, int],
) -> list[tuple[int, int]] | None:
    h, w = grid.shape
    ar, ac = anchor
    if not (0 <= ar < h and 0 <= ac < w):
        return None
    if int(grid[ar, ac]) != BLACK or touch[ar, ac] or reserved[ar, ac]:
        return None

    region = _region_mask(grid.shape, side, rect)
    if not region[anchor]:
        return None

    candidates = []
    for endpoint in _border_candidates(side, h, w):
        er, ec = endpoint
        if region[endpoint] and int(grid[er, ec]) == BLACK and not touch[er, ec] and not reserved[er, ec]:
            candidates.append(endpoint)
    if not candidates:
        return None

    base_allowed = (grid == BLACK) & (~touch) & (~reserved) & region
    base_allowed[anchor] = True

    for endpoint in candidates[:ENDPOINT_TRIES]:
        allowed = base_allowed.copy()
        allowed[endpoint] = True
        route = _bfs_path(allowed, anchor, endpoint)
        if route is None:
            continue
        full = [breakpoint] + route
        if _is_simple_path(full):
            return full
    return None


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    return _components(grid == color)


def _valid_sample(
    inp: np.ndarray,
    expected: np.ndarray,
    specs: list[tuple[int, str, int, int]],
    touch: np.ndarray,
) -> bool:
    if not np.array_equal(main(inp), expected):
        return False

    touch_count = int(np.count_nonzero(touch & (inp != BLACK) & (inp != GREY)))
    if touch_count != 2 * len(specs):
        return False

    for _rect_idx, _orientation, _value, color in specs:
        input_comps = _components_of_color(inp, color)
        output_comps = _components_of_color(expected, color)
        if len(input_comps) != 2 or len(output_comps) != 1:
            return False
        if not all(_is_simple_path(comp) for comp in input_comps):
            return False
        if not _is_simple_path(output_comps[0]):
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)

    rects = _place_rectangles(h, w)
    if rects is None:
        return None

    specs = _pick_slots(rects)
    if specs is None:
        return None

    inp = np.full((h, w), BLACK, dtype=int)
    for r0, c0, r1, c1 in rects:
        inp[r0 : r1 + 1, c0 : c1 + 1] = GREY

    reserved = np.zeros((h, w), dtype=bool)
    touch = _touch_mask(rects, h, w)

    for rect_idx, orientation, value, color in specs:
        rect = rects[rect_idx]
        r0, c0, r1, c1 = rect
        if orientation == "H":
            left = _route_component(
                inp,
                reserved,
                touch,
                rect,
                "L",
                (value, c0 - 1),
                (value, c0 - 2),
            )
            if left is None:
                return None
            for r, c in left:
                inp[r, c] = color
            _mark_reserved(reserved, left)

            right = _route_component(
                inp,
                reserved,
                touch,
                rect,
                "R",
                (value, c1 + 1),
                (value, c1 + 2),
            )
            if right is None:
                return None
            for r, c in right:
                inp[r, c] = color
            _mark_reserved(reserved, right)
        else:
            top = _route_component(
                inp,
                reserved,
                touch,
                rect,
                "T",
                (r0 - 1, value),
                (r0 - 2, value),
            )
            if top is None:
                return None
            for r, c in top:
                inp[r, c] = color
            _mark_reserved(reserved, top)

            bottom = _route_component(
                inp,
                reserved,
                touch,
                rect,
                "B",
                (r1 + 1, value),
                (r1 + 2, value),
            )
            if bottom is None:
                return None
            for r, c in bottom:
                inp[r, c] = color
            _mark_reserved(reserved, bottom)

    expected = inp.copy()
    for rect_idx, orientation, value, color in specs:
        r0, c0, r1, c1 = rects[rect_idx]
        if orientation == "H":
            expected[value, c0 : c1 + 1] = color
        else:
            expected[r0 : r1 + 1, value] = color

    if not _valid_sample(inp, expected, specs, touch):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid occluded-path sample")
