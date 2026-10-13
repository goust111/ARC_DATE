from __future__ import annotations

import random

import numpy as np


BG = 0
YELLOW = 4
WALL_COLORS = (1, 2, 3, 5, 6, 7, 8, 9)
SIDES = ("top", "right", "bottom", "left")
SIDE_DIRS = {
    "top": (-1, 0),
    "right": (0, 1),
    "bottom": (1, 0),
    "left": (0, -1),
}
MAX_TRACE_STEPS = 400
GEN_RETRIES = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _boundary_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _gap_cell(rect: dict, side: str, offset: int) -> tuple[int, int]:
    r0, c0, r1, c1 = rect["bbox"]
    if side == "top":
        return r0, c0 + offset
    if side == "bottom":
        return r1, c0 + offset
    if side == "left":
        return r0 + offset, c0
    return r0 + offset, c1


def _contour_cycle(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    pts: list[tuple[int, int]] = []
    for c in range(c0 - 1, c1 + 2):
        pts.append((r0 - 1, c))
    for r in range(r0, r1 + 2):
        pts.append((r, c1 + 1))
    for c in range(c1, c0 - 2, -1):
        pts.append((r1 + 1, c))
    for r in range(r1, r0 - 1, -1):
        pts.append((r, c0 - 1))
    return pts


def _make_rect(r0: int, c0: int, h: int, w: int, color: int) -> dict:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    return {
        "bbox": (r0, c0, r1, c1),
        "color": int(color),
        "h": h,
        "w": w,
        "interior_cells": [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)],
    }


def _assign_gap(rect: dict, side: str, offset: int) -> dict:
    out = dict(rect)
    gap = _gap_cell(out, side, offset)
    walls = _boundary_cells(*out["bbox"])
    walls.remove(gap)
    cycle = _contour_cycle(*out["bbox"])
    out["gap_side"] = side
    out["gap_offset"] = offset
    out["gap"] = gap
    out["dir"] = SIDE_DIRS[side]
    out["wall_cells"] = walls
    out["contour_cycle"] = cycle
    out["contour_index"] = {pt: i for i, pt in enumerate(cycle)}
    return out


def _all_gap_options(rect: dict) -> list[tuple[str, int]]:
    opts: list[tuple[str, int]] = []
    _, _, _, _ = rect["bbox"]
    for side in ("top", "bottom"):
        for offset in range(1, rect["w"] - 1):
            opts.append((side, offset))
    for side in ("left", "right"):
        for offset in range(1, rect["h"] - 1):
            opts.append((side, offset))
    return opts


def _wall_owner(rects: list[dict]) -> dict[tuple[int, int], int]:
    owner: dict[tuple[int, int], int] = {}
    for idx, rect in enumerate(rects):
        for cell in rect["wall_cells"]:
            owner[cell] = idx
    return owner


def _trace_path(
    rect: dict,
    rects: list[dict],
    owner: dict[tuple[int, int], int],
    shape: tuple[int, int],
    max_steps: int = MAX_TRACE_STEPS,
) -> tuple[list[tuple[int, int]], int | None]:
    dr, dc = rect["dir"]
    current = (rect["gap"][0] + dr, rect["gap"][1] + dc)
    path: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    collision_at: int | None = None
    mode = "ray"
    contour_id = -1
    contour_idx = -1

    for _ in range(max_steps):
        if not _in_bounds(shape, current[0], current[1]):
            break
        if current in owner or current in seen:
            break
        path.append(current)
        seen.add(current)

        if mode == "ray":
            nr = current[0] + dr
            nc = current[1] + dc
            if not _in_bounds(shape, nr, nc):
                break
            hit_id = owner.get((nr, nc))
            if hit_id is None:
                current = (nr, nc)
                continue
            if current not in rects[hit_id]["contour_index"]:
                break
            collision_at = len(path)
            contour_id = hit_id
            contour = rects[contour_id]["contour_cycle"]
            contour_idx = (rects[contour_id]["contour_index"][current] + 1) % len(contour)
            current = contour[contour_idx]
            mode = "contour"
            continue

        contour = rects[contour_id]["contour_cycle"]
        contour_idx = (contour_idx + 1) % len(contour)
        current = contour[contour_idx]

    return path, collision_at


def _parse_rectangles(grid: np.ndarray) -> list[dict]:
    wall_mask = (grid != BG) & (grid != YELLOW)
    rects: list[dict] = []
    for comp in _components(wall_mask):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h >= 3 and w >= 3

        boundary = _boundary_cells(r0, c0, r1, c1)
        comp_set = set(comp)
        assert comp_set.issubset(boundary)
        missing = boundary - comp_set
        assert len(missing) == 1
        gap = next(iter(missing))
        gr, gc = gap
        if gr == r0 and c0 < gc < c1:
            side = "top"
            offset = gc - c0
        elif gr == r1 and c0 < gc < c1:
            side = "bottom"
            offset = gc - c0
        elif gc == c0 and r0 < gr < r1:
            side = "left"
            offset = gr - r0
        elif gc == c1 and r0 < gr < r1:
            side = "right"
            offset = gr - r0
        else:
            raise AssertionError("gap must be a non-corner side cell")

        color = int(grid[comp[0]])
        rect = _assign_gap(_make_rect(r0, c0, h, w, color), side, offset)
        rect["yellow_count"] = int(np.sum(grid[r0 + 1 : r1, c0 + 1 : c1] == YELLOW))
        rects.append(rect)

    rects.sort(key=lambda rect: rect["bbox"])
    return rects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pull each rectangle's interior yellow pixels out through its unique gap and lay them outside in order.

    Input structure:
    - The background is black and the foreground walls are monochrome hollow rectangles.
    - Each wall is one cell thick and has exactly one missing side cell that acts as a gap.
    - Yellow pixels appear only inside the rectangles, and different rectangles do not touch.

    Rule:
    - Clear the yellow pixels from every rectangle interior while keeping each wall and gap unchanged.
    - For a rectangle with k interior yellow pixels, start at the first cell just outside the gap and place k yellow cells along the gap's outward normal.
    - If the path would step into another rectangle wall, switch to that wall's outside contour and continue clockwise around its exterior until all k cells are placed.

    Key insight:
    - The missing wall cell simultaneously identifies the exit point and the initial direction, and a wall collision changes the motion from straight ejection to clockwise wall-following on the obstacle's outer ring.
    """
    g = np.asarray(grid, dtype=int)
    rects = _parse_rectangles(g)
    owner = _wall_owner(rects)

    out = g.copy()
    for rect in rects:
        r0, c0, r1, c1 = rect["bbox"]
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[interior == YELLOW] = BG

    for rect in rects:
        k = rect["yellow_count"]
        if k == 0:
            continue
        path, _ = _trace_path(rect, rects, owner, g.shape)
        assert len(path) >= k
        for r, c in path[:k]:
            out[r, c] = YELLOW

    return out


def _bboxes_separated(a: dict, b: dict, margin: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a["bbox"]
    br0, bc0, br1, bc1 = b["bbox"]
    return (
        ar1 + margin < br0
        or br1 + margin < ar0
        or ac1 + margin < bc0
        or bc1 + margin < ac0
    )


def _render_input(shape: tuple[int, int], rects: list[dict], counts: list[int], rng: random.Random) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for rect in rects:
        for r, c in rect["wall_cells"]:
            grid[r, c] = rect["color"]
    for rect, k in zip(rects, counts):
        picks = rng.sample(rect["interior_cells"], k)
        for r, c in picks:
            grid[r, c] = YELLOW
    return grid


def _expected_output(inp: np.ndarray, rects: list[dict], counts: list[int]) -> np.ndarray:
    out = inp.copy()
    owner = _wall_owner(rects)
    for rect in rects:
        r0, c0, r1, c1 = rect["bbox"]
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[interior == YELLOW] = BG
    for rect, k in zip(rects, counts):
        path, _ = _trace_path(rect, rects, owner, inp.shape)
        for r, c in path[:k]:
            out[r, c] = YELLOW
    return out


def _collision_pair(shape: tuple[int, int], rng: random.Random) -> tuple[dict, dict] | None:
    h, w = shape
    for _ in range(200):
        side = rng.choice(SIDES)
        ah = rng.randint(4, 7)
        aw = rng.randint(4, 8)
        bh = rng.randint(4, 7)
        bw = rng.randint(4, 8)
        free_steps = rng.randint(2, 5)
        color_a, color_b = rng.sample(WALL_COLORS, 2)

        if side in ("top", "bottom"):
            hit_col = rng.randint(2, w - 3)
            a_gap = rng.randint(1, aw - 2)
            b_hit = rng.randint(1, bw - 2)
            ca = hit_col - a_gap
            cb = hit_col - b_hit
            if not (1 <= ca <= w - aw - 1 and 1 <= cb <= w - bw - 1):
                continue
            if side == "top":
                max_rb = h - ah - bh - free_steps - 1
                if max_rb < 1:
                    continue
                rb = rng.randint(1, max_rb)
                ra = rb + bh + free_steps
            else:
                min_rb = ah + free_steps + 1
                max_rb = h - bh - 1
                if min_rb > max_rb:
                    continue
                rb = rng.randint(min_rb, max_rb)
                ra = rb - free_steps - ah
            rect_a = _assign_gap(_make_rect(ra, ca, ah, aw, color_a), side, a_gap)
            rect_b = _make_rect(rb, cb, bh, bw, color_b)
        else:
            hit_row = rng.randint(2, h - 3)
            a_gap = rng.randint(1, ah - 2)
            b_hit = rng.randint(1, bh - 2)
            ra = hit_row - a_gap
            rb = hit_row - b_hit
            if not (1 <= ra <= h - ah - 1 and 1 <= rb <= h - bh - 1):
                continue
            if side == "left":
                max_cb = w - aw - bw - free_steps - 1
                if max_cb < 1:
                    continue
                cb = rng.randint(1, max_cb)
                ca = cb + bw + free_steps
            else:
                min_cb = aw + free_steps + 1
                max_cb = w - bw - 1
                if min_cb > max_cb:
                    continue
                cb = rng.randint(min_cb, max_cb)
                ca = cb - free_steps - aw
            rect_a = _assign_gap(_make_rect(ra, ca, ah, aw, color_a), side, a_gap)
            rect_b = _make_rect(rb, cb, bh, bw, color_b)

        if not _bboxes_separated(rect_a, rect_b, margin=0):
            continue
        return rect_a, rect_b
    return None


def _try_collision_input(rng: random.Random) -> np.ndarray | None:
    shape = (rng.randint(18, 28), rng.randint(18, 28))
    pair = _collision_pair(shape, rng)
    if pair is None:
        return None
    rect_a, rect_b_base = pair

    options = _all_gap_options(rect_b_base)
    rng.shuffle(options)
    for side_b, offset_b in options:
        rect_b = _assign_gap(rect_b_base, side_b, offset_b)
        rects = [rect_a, rect_b]
        owner = _wall_owner(rects)
        path_a, coll_a = _trace_path(rect_a, rects, owner, shape)
        path_b, coll_b = _trace_path(rect_b, rects, owner, shape)
        if coll_a is None or coll_b is not None:
            continue

        max_a = min(len(path_a), len(rect_a["interior_cells"]), 8)
        max_b = min(len(path_b), len(rect_b["interior_cells"]), 6)
        if max_a <= coll_a or max_b < 1:
            continue

        k_a_choices = list(range(coll_a + 1, max_a + 1))
        k_b_choices = list(range(1, max_b + 1))
        rng.shuffle(k_a_choices)
        rng.shuffle(k_b_choices)
        for k_a in k_a_choices:
            prefix_a = set(path_a[:k_a])
            for k_b in k_b_choices:
                prefix_b = set(path_b[:k_b])
                if prefix_a & prefix_b:
                    continue
                counts = [k_a, k_b]
                inp = _render_input(shape, rects, counts, rng)
                expected = _expected_output(inp, rects, counts)
                if np.array_equal(main(inp), expected):
                    return inp
    return None


def _zone_ranges(shape: tuple[int, int], zone: str, h: int, w: int) -> tuple[range, range] | None:
    rows, cols = shape
    mid_r = rows // 2
    mid_c = cols // 2
    if zone == "tl":
        r_range = range(1, max(1, mid_r - h))
        c_range = range(1, max(1, mid_c - w))
    elif zone == "tr":
        r_range = range(1, max(1, mid_r - h))
        c_range = range(mid_c, max(mid_c, cols - w - 1) + 1)
    elif zone == "bl":
        r_range = range(mid_r, max(mid_r, rows - h - 1) + 1)
        c_range = range(1, max(1, mid_c - w))
    else:
        r_range = range(mid_r, max(mid_r, rows - h - 1) + 1)
        c_range = range(mid_c, max(mid_c, cols - w - 1) + 1)
    if not r_range or not c_range:
        return None
    return r_range, c_range


def _outward_sides(zone: str) -> tuple[str, str]:
    if zone == "tl":
        return "top", "left"
    if zone == "tr":
        return "top", "right"
    if zone == "bl":
        return "bottom", "left"
    return "bottom", "right"


def _try_open_input(rng: random.Random) -> np.ndarray | None:
    shape = (rng.randint(16, 28), rng.randint(16, 28))
    zones = ["tl", "tr", "bl", "br"]
    rng.shuffle(zones)
    count = rng.randint(1, 3)
    colors = rng.sample(WALL_COLORS, count)
    rects: list[dict] = []

    for zone, color in zip(zones[:count], colors):
        placed = False
        for _ in range(80):
            rh = rng.randint(4, 7)
            rw = rng.randint(4, 8)
            rc = _zone_ranges(shape, zone, rh, rw)
            if rc is None:
                continue
            r_range, c_range = rc
            r0 = rng.choice(list(r_range))
            c0 = rng.choice(list(c_range))
            side = rng.choice(_outward_sides(zone))
            if side in ("top", "bottom"):
                offset = rng.randint(1, rw - 2)
            else:
                offset = rng.randint(1, rh - 2)
            rect = _assign_gap(_make_rect(r0, c0, rh, rw, color), side, offset)
            if all(_bboxes_separated(rect, other, margin=2) for other in rects):
                rects.append(rect)
                placed = True
                break
        if not placed:
            return None

    owner = _wall_owner(rects)
    used: set[tuple[int, int]] = set()
    counts: list[int] = []
    for rect in rects:
        path, coll = _trace_path(rect, rects, owner, shape)
        if coll is not None:
            return None
        max_k = min(len(path), len(rect["interior_cells"]), 6)
        if max_k < 1:
            return None
        choices = list(range(1, max_k + 1))
        rng.shuffle(choices)
        chosen = None
        for k in choices:
            prefix = set(path[:k])
            if prefix & used:
                continue
            chosen = k
            used |= prefix
            break
        if chosen is None:
            return None
        counts.append(chosen)

    inp = _render_input(shape, rects, counts, rng)
    expected = _expected_output(inp, rects, counts)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        if rng.random() < 0.55:
            grid = _try_collision_input(rng)
        else:
            grid = _try_open_input(rng)
        if grid is None:
            continue
        out = main(grid)
        if (
            isinstance(grid, np.ndarray)
            and isinstance(out, np.ndarray)
            and grid.ndim == 2
            and out.ndim == 2
            and max(grid.shape) <= 30
            and max(out.shape) <= 30
            and int(np.sum(grid == YELLOW)) == int(np.sum(out == YELLOW))
        ):
            return grid
    raise RuntimeError("failed to generate a valid gap-outlet input")
