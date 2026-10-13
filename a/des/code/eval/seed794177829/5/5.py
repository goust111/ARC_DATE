from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
AZURE = 8
MAROON = 9
STAMP_COLORS = (1, 2, 3, 4, 5, 6, 7)


def _find_separator(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == MAROON)]
    assert len(cols) == 1, "expected exactly one maroon separator column"
    return int(cols[0])


def _find_anchor(grid: np.ndarray, sep: int) -> tuple[int, int]:
    right = grid[:, sep + 1 :]
    anchor = np.argwhere(right == BLACK)
    assert len(anchor) == 1, "expected exactly one black anchor in the right region"
    ar, ac = map(int, anchor[0])
    return ar, ac + sep + 1


def _run_len(mask: np.ndarray, r: int, c: int, dr: int, dc: int) -> int:
    h, w = mask.shape
    n = 0
    rr, cc = int(r), int(c)
    while 0 <= rr < h and 0 <= cc < w and bool(mask[rr, cc]):
        n += 1
        rr += dr
        cc += dc
    return n


def _inside(rect: tuple[int, int, int, int], r: int, c: int) -> bool:
    r0, c0, r1, c1 = rect
    return r0 <= r <= r1 and c0 <= c <= c1


def _candidate_ok(
    left: np.ndarray,
    coords: np.ndarray,
    rect: tuple[int, int, int, int],
) -> bool:
    h, w = left.shape
    r0, c0, r1, c1 = rect
    if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
        return False
    rows = coords[:, 0]
    cols = coords[:, 1]
    if np.any(rows < r0) or np.any(rows > r1) or np.any(cols < c0) or np.any(cols > c1):
        return False
    window = left[r0 : r1 + 1, c0 : c1 + 1]
    if np.any(window == AZURE):
        return False
    return True


def _candidate_rectangles(left: np.ndarray, color: int) -> list[tuple[int, int, int, int]]:
    mask = left == int(color)
    coords = np.argwhere(mask)
    assert coords.size > 0, f"missing visible pixels for color {color}"
    rects: set[tuple[int, int, int, int]] = set()

    for r_raw, c_raw in coords:
        r = int(r_raw)
        c = int(c_raw)

        w = _run_len(mask, r, c, 0, 1)
        h = _run_len(mask, r, c, 1, 0)
        rect = (r, c, r + h - 1, c + w - 1)
        if _candidate_ok(left, coords, rect):
            rects.add(rect)

        w = _run_len(mask, r, c, 0, -1)
        h = _run_len(mask, r, c, 1, 0)
        rect = (r, c - w + 1, r + h - 1, c)
        if _candidate_ok(left, coords, rect):
            rects.add(rect)

        w = _run_len(mask, r, c, 0, 1)
        h = _run_len(mask, r, c, -1, 0)
        rect = (r - h + 1, c, r, c + w - 1)
        if _candidate_ok(left, coords, rect):
            rects.add(rect)

        w = _run_len(mask, r, c, 0, -1)
        h = _run_len(mask, r, c, -1, 0)
        rect = (r - h + 1, c - w + 1, r, c)
        if _candidate_ok(left, coords, rect):
            rects.add(rect)

    ordered = sorted(rects)
    assert ordered, f"no rectangle candidates for color {color}"
    return ordered


def _topological_order(
    colors: list[int],
    below_to_above: dict[int, set[int]],
) -> list[int] | None:
    indeg = {color: 0 for color in colors}
    for color in colors:
        for nxt in below_to_above[color]:
            indeg[nxt] += 1

    ready = sorted(color for color in colors if indeg[color] == 0)
    order: list[int] = []
    while ready:
        color = ready.pop(0)
        order.append(int(color))
        for nxt in sorted(below_to_above[color]):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                ready.append(int(nxt))
        ready.sort()

    if len(order) != len(colors):
        return None
    return order


def _render_left(
    shape: tuple[int, int],
    assignment: dict[int, tuple[int, int, int, int]],
    order: list[int],
) -> np.ndarray:
    out = np.full(shape, AZURE, dtype=int)
    for color in order:
        r0, c0, r1, c1 = assignment[int(color)]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    return out


def _validate_assignment(
    left: np.ndarray,
    colors: list[int],
    assignment: dict[int, tuple[int, int, int, int]],
) -> list[int] | None:
    below_to_above = {color: set() for color in colors}
    h, w = left.shape
    for r in range(h):
        for c in range(w):
            observed = int(left[r, c])
            covering = [
                color
                for color in colors
                if _inside(assignment[int(color)], int(r), int(c))
            ]
            if observed == AZURE:
                if covering:
                    return None
                continue
            if observed not in assignment:
                return None
            if observed not in covering:
                return None
            for lower in covering:
                if lower != observed:
                    below_to_above[int(lower)].add(int(observed))

    order = _topological_order(colors, below_to_above)
    if order is None:
        return None
    if not np.array_equal(_render_left(left.shape, assignment, order), left):
        return None
    return order


def _collect_solutions(
    left: np.ndarray,
    colors: list[int],
    limit: int = 1,
) -> list[tuple[dict[int, tuple[int, int, int, int]], list[int]]]:
    candidates = {color: _candidate_rectangles(left, int(color)) for color in colors}
    search_order = sorted(colors, key=lambda color: (len(candidates[color]), color))
    solutions: list[tuple[dict[int, tuple[int, int, int, int]], list[int]]] = []
    chosen: dict[int, tuple[int, int, int, int]] = {}

    def rec(i: int) -> None:
        if len(solutions) >= limit:
            return
        if i == len(search_order):
            order = _validate_assignment(left, colors, chosen)
            if order is not None:
                solutions.append((chosen.copy(), order))
            return

        color = int(search_order[i])
        for rect in candidates[color]:
            chosen[color] = rect
            rec(i + 1)
            if len(solutions) >= limit:
                return
        chosen.pop(color, None)

    rec(0)
    return solutions


def _reconstruct_rectangles(
    left: np.ndarray,
) -> tuple[dict[int, tuple[int, int, int, int]], list[int]]:
    colors = sorted(int(c) for c in np.unique(left) if int(c) != AZURE)
    assert colors, "expected at least one stamp color"
    solutions = _collect_solutions(left, colors, limit=1)
    assert solutions, "failed to reconstruct the left-hand rectangles"
    return solutions[0]


def _center_start(anchor_coord: int, length: int) -> int:
    return int(anchor_coord) - (int(length) - 1) // 2


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct overlapped solid rectangles from the left scene and re-stack them on the right anchor by area.

    Input structure:
    - Azure background.
    - A full maroon separator column splits the grid into left and right regions.
    - The left region contains distinct-color solid rectangles that partially occlude each other.
    - The right region is blank azure except for one black anchor pixel.

    Rule:
    - Infer each rectangle's full bounding box from a visible adjacent horizontal/vertical edge pair.
    - Use overlap constraints to recover a consistent front/back layering in the left region.
    - Repaint the right region by centering every reconstructed rectangle on the black anchor.
    - Place larger-area rectangles first (bottom) and smaller-area rectangles last (top).
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    sep = _find_separator(g)
    anchor_r, anchor_c = _find_anchor(g, sep)

    left = g[:, :sep]
    rects, layer_order = _reconstruct_rectangles(left)

    out = g.copy()
    out[:, sep + 1 :] = AZURE
    out[anchor_r, anchor_c] = BLACK

    layer_rank = {color: i for i, color in enumerate(layer_order)}
    paint_order = sorted(
        rects,
        key=lambda color: (
            -(
                (rects[color][2] - rects[color][0] + 1)
                * (rects[color][3] - rects[color][1] + 1)
            ),
            layer_rank[color],
            color,
        ),
    )

    for color in paint_order:
        r0, c0, r1, c1 = rects[color]
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        rr0 = _center_start(anchor_r, h)
        cc0 = _center_start(anchor_c, w)
        rr1 = rr0 + h - 1
        cc1 = cc0 + w - 1
        assert 0 <= rr0 and rr1 < g.shape[0]
        assert sep + 1 <= cc0 and cc1 < g.shape[1]
        out[rr0 : rr1 + 1, cc0 : cc1 + 1] = int(color)

    return out


# --------------- generator ---------------

GEN_RETRIES = 400
PLACEMENT_RETRIES = 200
GRID_H_RANGE = (12, 20)
LEFT_W_RANGE = (12, 16)
RECT_COUNT_RANGE = (3, 5)
RECT_H_RANGE = (3, 8)
RECT_W_RANGE = (3, 8)
RIGHT_MARGIN_RANGE = (4, 8)
MAX_TOTAL_WIDTH = 30


def _rect_area(rect: dict[str, int]) -> int:
    return int(rect["h"] * rect["w"])


def _protected_cells(rect: dict[str, int]) -> list[tuple[int, int]]:
    r0 = int(rect["r0"])
    c0 = int(rect["c0"])
    r1 = r0 + int(rect["h"]) - 1
    c1 = c0 + int(rect["w"]) - 1
    corner = rect["corner"]

    cells: list[tuple[int, int]] = []
    if corner == "TL":
        cells.extend((r0, c) for c in range(c0, c1 + 1))
        cells.extend((r, c0) for r in range(r0 + 1, r1 + 1))
    elif corner == "TR":
        cells.extend((r0, c) for c in range(c0, c1 + 1))
        cells.extend((r, c1) for r in range(r0 + 1, r1 + 1))
    elif corner == "BL":
        cells.extend((r1, c) for c in range(c0, c1 + 1))
        cells.extend((r, c0) for r in range(r0, r1))
    else:
        assert corner == "BR"
        cells.extend((r1, c) for c in range(c0, c1 + 1))
        cells.extend((r, c1) for r in range(r0, r1))
    return cells


def _intersects(a: dict[str, int], b: dict[str, int]) -> int:
    ar0, ac0 = int(a["r0"]), int(a["c0"])
    ar1 = ar0 + int(a["h"]) - 1
    ac1 = ac0 + int(a["w"]) - 1
    br0, bc0 = int(b["r0"]), int(b["c0"])
    br1 = br0 + int(b["h"]) - 1
    bc1 = bc0 + int(b["w"]) - 1

    hr = min(ar1, br1) - max(ar0, br0) + 1
    wc = min(ac1, bc1) - max(ac0, bc0) + 1
    if hr <= 0 or wc <= 0:
        return 0
    return int(hr * wc)


def _render_left_from_rects(
    h: int,
    w: int,
    rects: list[dict[str, int]],
) -> np.ndarray:
    left = np.full((h, w), AZURE, dtype=int)
    for rect in rects:
        r0 = int(rect["r0"])
        c0 = int(rect["c0"])
        left[r0 : r0 + int(rect["h"]), c0 : c0 + int(rect["w"])] = int(rect["color"])
    return left


def _component_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            count += 1
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
    return int(count)


def _sample_dimensions(n: int, h: int, w: int) -> list[tuple[int, int]] | None:
    max_h = min(RECT_H_RANGE[1], h - 2)
    max_w = min(RECT_W_RANGE[1], w - 2)
    if max_h < RECT_H_RANGE[0] or max_w < RECT_W_RANGE[0]:
        return None

    choices = [
        (hh, ww)
        for hh in range(RECT_H_RANGE[0], max_h + 1)
        for ww in range(RECT_W_RANGE[0], max_w + 1)
    ]
    random.shuffle(choices)
    dims: list[tuple[int, int]] = []
    used_areas: set[int] = set()
    for hh, ww in choices:
        area = hh * ww
        if area in used_areas:
            continue
        dims.append((hh, ww))
        used_areas.add(area)
        if len(dims) == n:
            break
    return dims if len(dims) == n else None


def _place_rectangles(
    grid_h: int,
    left_w: int,
    colors: list[int],
    dims: list[tuple[int, int]],
) -> list[dict[str, int]] | None:
    rects: list[dict[str, int]] = []
    reserved = np.zeros((grid_h, left_w), dtype=bool)
    corners = ("TL", "TR", "BL", "BR")

    for i, (color, (hh, ww)) in enumerate(zip(colors, dims, strict=False)):
        must_overlap = i > 0 and (i == 1 or random.random() < 0.8)
        for _ in range(PLACEMENT_RETRIES):
            r0 = random.randint(0, grid_h - hh)
            c0 = random.randint(0, left_w - ww)
            if bool(reserved[r0 : r0 + hh, c0 : c0 + ww].any()):
                continue

            rect = {
                "color": int(color),
                "r0": int(r0),
                "c0": int(c0),
                "h": int(hh),
                "w": int(ww),
                "corner": random.choice(corners),
            }
            overlaps = [_intersects(rect, prev) for prev in rects]
            if must_overlap and max(overlaps or [0]) < 2:
                continue
            if overlaps and sum(overlaps) > _rect_area(rect) - min(rect["h"], rect["w"]):
                continue

            cells = _protected_cells(rect)
            if any(bool(reserved[r, c]) for r, c in cells):
                continue

            rects.append(rect)
            for r, c in cells:
                reserved[r, c] = True
            break
        else:
            return None

    return rects


def _expected_output(
    inp: np.ndarray,
    sep: int,
    anchor: tuple[int, int],
    rects: list[dict[str, int]],
) -> np.ndarray:
    out = inp.copy()
    out[:, sep + 1 :] = AZURE
    out[anchor] = BLACK
    order = sorted(rects, key=lambda rect: (-_rect_area(rect), rect["color"]))
    for rect in order:
        rr0 = _center_start(anchor[0], int(rect["h"]))
        cc0 = _center_start(anchor[1], int(rect["w"]))
        out[rr0 : rr0 + int(rect["h"]), cc0 : cc0 + int(rect["w"])] = int(rect["color"])
    return out


def _scene_is_clear(left: np.ndarray, rects: list[dict[str, int]]) -> bool:
    overlap_pairs = 0
    hidden_rects = 0
    for i, rect in enumerate(rects):
        if any(_intersects(rect, other) > 0 for other in rects[i + 1 :]):
            hidden_rects += 1
        visible = left == int(rect["color"])
        if int(visible.sum()) == 0:
            return False
        if _component_count(visible) > 3:
            return False
        if any(left[r, c] != int(rect["color"]) for r, c in _protected_cells(rect)):
            return False
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            if _intersects(rects[i], rects[j]) > 0:
                overlap_pairs += 1
    return overlap_pairs >= 2 and hidden_rects >= 1


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_H_RANGE)
    left_w = random.randint(*LEFT_W_RANGE)
    n_rects = random.randint(*RECT_COUNT_RANGE)
    dims = _sample_dimensions(n_rects, grid_h, left_w)
    if dims is None:
        return None
    random.shuffle(dims)

    max_w = max(w for _h, w in dims)
    min_right_w = max_w + random.randint(*RIGHT_MARGIN_RANGE)
    max_right_w = min(29 - left_w, min_right_w + 2)
    if max_right_w < min_right_w or min_right_w < 8:
        return None
    right_w = random.randint(min_right_w, max_right_w)

    colors = random.sample(STAMP_COLORS, n_rects)
    rects = _place_rectangles(grid_h, left_w, colors, dims)
    if rects is None:
        return None

    left = _render_left_from_rects(grid_h, left_w, rects)
    if not _scene_is_clear(left, rects):
        return None

    grid_w = left_w + 1 + right_w
    inp = np.full((grid_h, grid_w), AZURE, dtype=int)
    inp[:, left_w] = MAROON
    inp[:, :left_w] = left

    max_h = max(h for h, _w in dims)
    anchor_r_min = (max_h - 1) // 2
    anchor_r_max = grid_h - 1 - max_h // 2
    if anchor_r_min > anchor_r_max:
        return None

    anchor_c_min = left_w + 1 + (max_w - 1) // 2
    anchor_c_max = grid_w - 1 - max_w // 2
    if anchor_c_min > anchor_c_max:
        return None

    anchor = (
        random.randint(anchor_r_min, anchor_r_max),
        random.randint(anchor_c_min, anchor_c_max),
    )
    inp[anchor] = BLACK

    solutions = _collect_solutions(inp[:, :left_w], colors, limit=2)
    if not solutions:
        return None

    expected = _expected_output(inp, left_w, anchor, rects)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid overlapped-rectangle puzzle")
