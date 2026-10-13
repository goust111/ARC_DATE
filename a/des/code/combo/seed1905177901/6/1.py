from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

SIDE_COLORS = {
    "top": RED,
    "right": BLUE,
    "bottom": GREEN,
    "left": YELLOW,
}
SIDE_ORDER = ("top", "right", "bottom", "left")
CORNER_SIDES = {
    "top_left": ("top", "left"),
    "top_right": ("top", "right"),
    "bottom_left": ("bottom", "left"),
    "bottom_right": ("bottom", "right"),
}
LEGEND_POOL = (6, 7, 8, 9)


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            comp: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                comp.append((r, c))
                for rr, cc in _neighbors4(r, c, h, w):
                    if mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _hole_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    holes: list[list[tuple[int, int]]] = []
    for r0 in range(h):
        for c0 in range(w):
            if mask[r0, c0] or seen[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            comp: list[tuple[int, int]] = []
            touches_bbox = False
            while stack:
                r, c = stack.pop()
                comp.append((r, c))
                if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                    touches_bbox = True
                for rr, cc in _neighbors4(r, c, h, w):
                    if not mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            if not touches_bbox:
                holes.append(comp)
    return holes


def _find_frame(g: np.ndarray) -> tuple[int, int, int, int]:
    top_rows = np.unique(np.argwhere(g == RED)[:, 0])
    bottom_rows = np.unique(np.argwhere(g == GREEN)[:, 0])
    left_cols = np.unique(np.argwhere(g == YELLOW)[:, 1])
    right_cols = np.unique(np.argwhere(g == BLUE)[:, 1])
    assert len(top_rows) == len(bottom_rows) == len(left_cols) == len(right_cols) == 1

    r_top = int(top_rows[0])
    r_bottom = int(bottom_rows[0])
    c_left = int(left_cols[0])
    c_right = int(right_cols[0])
    assert r_top < r_bottom and c_left < c_right
    assert np.all(g[r_top, c_left + 1 : c_right] == RED)
    assert np.all(g[r_bottom, c_left + 1 : c_right] == GREEN)
    assert np.all(g[r_top + 1 : r_bottom, c_left] == YELLOW)
    assert np.all(g[r_top + 1 : r_bottom, c_right] == BLUE)
    assert g[r_top, c_left] == BG and g[r_top, c_right] == BG
    assert g[r_bottom, c_left] == BG and g[r_bottom, c_right] == BG
    return r_top, c_left, r_bottom, c_right


def _find_legend(g: np.ndarray, r_top: int, r_bottom: int) -> tuple[int, dict[int, int]]:
    rows = [
        r
        for r in range(g.shape[0])
        if (r < r_top or r > r_bottom) and np.any(g[r] != BG)
    ]
    assert len(rows) == 1
    row = int(rows[0])
    cols = [int(c) for c in np.flatnonzero(g[row] != BG)]
    assert len(cols) == 4
    colors = [int(g[row, c]) for c in cols]
    return row, {i: color for i, color in enumerate(colors)}


def _contact_counts(
    points: list[tuple[int, int]],
    r_top: int,
    c_left: int,
    r_bottom: int,
    c_right: int,
) -> dict[str, int]:
    counts = {side: 0 for side in SIDE_ORDER}
    for r, c in points:
        if r == r_top + 1 and c_left < c < c_right:
            counts["top"] += 1
        if r == r_bottom - 1 and c_left < c < c_right:
            counts["bottom"] += 1
        if c == c_left + 1 and r_top < r < r_bottom:
            counts["left"] += 1
        if c == c_right - 1 and r_top < r < r_bottom:
            counts["right"] += 1
    return counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the holes of border-touching grey objects with the contacted frame-side color,
    recolor the object by its hole count via the legend row, and erase the legend row.
    """
    g = np.asarray(grid, dtype=int)
    r_top, c_left, r_bottom, c_right = _find_frame(g)
    legend_row, legend = _find_legend(g, r_top, r_bottom)

    out = g.copy()
    out[legend_row, :] = BG

    for comp in _components(g == GREY):
        counts = _contact_counts(comp, r_top, c_left, r_bottom, c_right)
        if max(counts.values()) == 0:
            continue

        side = max(SIDE_ORDER, key=lambda name: counts[name])
        r0, c0, r1, c1 = _bbox(comp)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in comp:
            mask[r - r0, c - c0] = True

        holes = _hole_components(mask)
        hole_count = len(holes)
        assert hole_count in legend
        fill_color = SIDE_COLORS[side]
        body_color = int(legend[hole_count])

        for hole in holes:
            for rr, cc in hole:
                out[r0 + rr, c0 + cc] = fill_color
        for r, c in comp:
            out[r, c] = body_color

    return out


def _pick_holes(h: int, w: int, hole_count: int) -> list[tuple[int, int]] | None:
    if hole_count == 0:
        return []

    layouts: list[list[tuple[int, int]]] = []
    for r_start in (1, 2):
        rows = list(range(r_start, h - 1, 2))
        if not rows:
            continue
        for c_start in (1, 2):
            cols = list(range(c_start, w - 1, 2))
            if not cols:
                continue
            slots = [(r, c) for r in rows for c in cols]
            if len(slots) >= hole_count:
                layouts.append(slots)

    if not layouts:
        return None
    slots = random.choice(layouts)
    return random.sample(slots, hole_count)


def _carve_notch(mask: np.ndarray) -> None:
    h, w = mask.shape
    side = random.choice(SIDE_ORDER)
    if side in ("top", "bottom"):
        if w <= 3:
            return
        length = random.randint(1, min(2, w - 2))
        start = random.randint(1, w - 1 - length)
        row = 0 if side == "top" else h - 1
        mask[row, start : start + length] = False
    else:
        if h <= 3:
            return
        length = random.randint(1, min(2, h - 2))
        start = random.randint(1, h - 1 - length)
        col = 0 if side == "left" else w - 1
        mask[start : start + length, col] = False


def _make_object_mask(height: int, width: int, hole_count: int) -> np.ndarray | None:
    holes = _pick_holes(height, width, hole_count)
    if holes is None:
        return None

    mask = np.ones((height, width), dtype=bool)
    for r, c in holes:
        mask[r, c] = False

    if random.random() < 0.75:
        _carve_notch(mask)
    if random.random() < 0.25:
        _carve_notch(mask)
    return mask


def _dimension_choices(hole_count: int, mode: str) -> list[tuple[int, int]]:
    choices: list[tuple[int, int]] = []
    if hole_count == 0:
        h_range = range(3, 7)
        w_range = range(3, 7)
    elif hole_count == 1:
        h_range = range(3, 7)
        w_range = range(3, 7)
    elif hole_count == 2:
        h_range = range(4, 7)
        w_range = range(4, 7)
    else:
        h_range = range(5, 7)
        w_range = range(5, 7)

    for h in h_range:
        for w in w_range:
            if mode in CORNER_SIDES and h == w:
                continue
            if _pick_holes(h, w, hole_count) is None:
                continue
            choices.append((h, w))
    return choices


def _candidate_positions(mode: str, inner_h: int, inner_w: int, h: int, w: int) -> list[tuple[int, int]]:
    if h > inner_h or w > inner_w:
        return []
    if mode == "free":
        if inner_h - h < 2 or inner_w - w < 2:
            return []
        return [(r, c) for r in range(1, inner_h - h) for c in range(1, inner_w - w)]
    if mode == "top":
        if inner_w - w < 2:
            return []
        return [(0, c) for c in range(1, inner_w - w)]
    if mode == "bottom":
        if inner_w - w < 2:
            return []
        return [(inner_h - h, c) for c in range(1, inner_w - w)]
    if mode == "left":
        if inner_h - h < 2:
            return []
        return [(r, 0) for r in range(1, inner_h - h)]
    if mode == "right":
        if inner_h - h < 2:
            return []
        return [(r, inner_w - w) for r in range(1, inner_h - h)]
    if mode == "top_left":
        return [(0, 0)]
    if mode == "top_right":
        return [(0, inner_w - w)]
    if mode == "bottom_left":
        return [(inner_h - h, 0)]
    if mode == "bottom_right":
        return [(inner_h - h, inner_w - w)]
    raise AssertionError("unknown placement mode")


def _bbox_free(blocked: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(blocked.shape[0], r0 + h + 1)
    cc1 = min(blocked.shape[1], c0 + w + 1)
    return not bool(blocked[rr0:rr1, cc0:cc1].any())


def _mark_blocked(blocked: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(blocked.shape[0], r0 + h + 1)
    cc1 = min(blocked.shape[1], c0 + w + 1)
    blocked[rr0:rr1, cc0:cc1] = True


def _contact_counts_interior(mask: np.ndarray, r0: int, c0: int, inner_h: int, inner_w: int) -> dict[str, int]:
    counts = {side: 0 for side in SIDE_ORDER}
    for rr, cc in np.argwhere(mask):
        rr = int(rr) + r0
        cc = int(cc) + c0
        if rr == 0:
            counts["top"] += 1
        if rr == inner_h - 1:
            counts["bottom"] += 1
        if cc == 0:
            counts["left"] += 1
        if cc == inner_w - 1:
            counts["right"] += 1
    return counts


def _sample_modes(n_objects: int) -> list[str]:
    touch_count = random.randint(1, min(2, n_objects - 1))
    touch_modes: list[str] = []
    pool = ["top", "bottom", "left", "right", "top_left", "top_right", "bottom_left", "bottom_right"]
    if random.random() < 0.5:
        corner = random.choice(list(CORNER_SIDES))
        touch_modes.append(corner)
        pool.remove(corner)
    extra = touch_count - len(touch_modes)
    if extra > 0:
        touch_modes.extend(random.sample(pool, extra))
    modes = touch_modes + ["free"] * (n_objects - touch_count)
    random.shuffle(modes)
    return modes


def _sample_hole_counts(modes: list[str]) -> list[int]:
    hole_counts = [random.randint(0, 3) for _ in modes]
    touch_idx = [i for i, mode in enumerate(modes) if mode != "free"]
    free_idx = [i for i, mode in enumerate(modes) if mode == "free"]

    if not any(hole_counts[i] > 0 for i in touch_idx):
        hole_counts[random.choice(touch_idx)] = random.randint(1, 3)
    if len(touch_idx) > 1 and len({hole_counts[i] for i in touch_idx}) == 1:
        idx = random.choice(touch_idx)
        options = [v for v in range(4) if v != hole_counts[idx]]
        hole_counts[idx] = random.choice(options)
    if free_idx and not any(hole_counts[i] > 0 for i in free_idx) and random.random() < 0.75:
        hole_counts[random.choice(free_idx)] = random.randint(1, 3)
    return hole_counts


def _draw_frame(grid: np.ndarray, top: int, left: int, inner_h: int, inner_w: int) -> None:
    bottom = top + inner_h + 1
    right = left + inner_w + 1
    grid[top, left + 1 : right] = RED
    grid[bottom, left + 1 : right] = GREEN
    grid[top + 1 : bottom, left] = YELLOW
    grid[top + 1 : bottom, right] = BLUE


def _draw_legend(grid: np.ndarray, row: int, frame_left: int, frame_w: int, colors: list[int]) -> None:
    gaps = [random.randint(0, 2) for _ in range(3)]
    total = 4 + sum(gaps)
    start = random.randint(frame_left, frame_left + frame_w - total)
    c = start
    for i, color in enumerate(colors):
        grid[row, c] = int(color)
        c += 1
        if i < 3:
            c += gaps[i]


def _place_objects(
    grid: np.ndarray,
    frame_top: int,
    frame_left: int,
    inner_h: int,
    inner_w: int,
    modes: list[str],
    hole_counts: list[int],
) -> bool:
    specs = []
    for mode, hole_count in zip(modes, hole_counts, strict=False):
        dims = _dimension_choices(hole_count, mode)
        random.shuffle(dims)
        mask = None
        for h, w in dims:
            mask = _make_object_mask(h, w, hole_count)
            if mask is not None:
                specs.append({"mode": mode, "holes": hole_count, "mask": mask})
                break
        if mask is None:
            return False

    priority = {**{name: 0 for name in CORNER_SIDES}, "top": 1, "right": 1, "bottom": 1, "left": 1, "free": 2}
    specs.sort(key=lambda spec: (priority[spec["mode"]], -int(spec["mask"].sum())))

    blocked = np.zeros((inner_h, inner_w), dtype=bool)
    corner_present = any(spec["mode"] in CORNER_SIDES for spec in specs)
    found_corner_contact = False

    for spec in specs:
        mode = spec["mode"]
        mask = spec["mask"]
        h, w = map(int, mask.shape)
        positions = _candidate_positions(mode, inner_h, inner_w, h, w)
        random.shuffle(positions)
        placed = False
        for r0, c0 in positions:
            if not _bbox_free(blocked, r0, c0, h, w):
                continue

            counts = _contact_counts_interior(mask, r0, c0, inner_h, inner_w)
            touched = [side for side in SIDE_ORDER if counts[side] > 0]
            if mode == "free":
                if touched:
                    continue
            elif mode in CORNER_SIDES:
                expected = CORNER_SIDES[mode]
                if set(touched) != set(expected):
                    continue
                top_two = sorted(counts.values(), reverse=True)[:2]
                if len(top_two) < 2 or top_two[0] == top_two[1]:
                    continue
                found_corner_contact = True
            else:
                if touched != [mode]:
                    continue

            for rr, cc in np.argwhere(mask):
                grid[frame_top + 1 + r0 + int(rr), frame_left + 1 + c0 + int(cc)] = GREY
            _mark_blocked(blocked, r0, c0, h, w)
            placed = True
            break

        if not placed:
            return False

    if corner_present and not found_corner_contact:
        return False
    return True


def _build_input() -> np.ndarray | None:
    inner_h = random.randint(11, 18)
    inner_w = random.randint(11, 18)
    frame_h = inner_h + 2
    frame_w = inner_w + 2

    top_margin = random.randint(1, 3)
    bottom_margin = random.randint(1, 3)
    left_margin = random.randint(1, 4)
    right_margin = random.randint(1, 4)
    legend_gap = random.randint(1, 2)
    legend_above = random.choice([True, False])

    if legend_above:
        legend_row = top_margin
        frame_top = legend_row + legend_gap + 1
        grid_h = frame_top + frame_h + bottom_margin
    else:
        frame_top = top_margin
        legend_row = frame_top + frame_h + legend_gap
        grid_h = legend_row + 1 + bottom_margin

    frame_left = left_margin
    grid_w = frame_left + frame_w + right_margin
    grid = np.zeros((grid_h, grid_w), dtype=int)

    _draw_frame(grid, frame_top, frame_left, inner_h, inner_w)
    legend_colors = random.sample(LEGEND_POOL, 4)
    _draw_legend(grid, legend_row, frame_left, frame_w, legend_colors)

    n_objects = random.choice([3, 3, 4])
    modes = _sample_modes(n_objects)
    hole_counts = _sample_hole_counts(modes)
    if not _place_objects(grid, frame_top, frame_left, inner_h, inner_w, modes, hole_counts):
        return None

    grey_comps = _components(grid == GREY)
    if len(grey_comps) != n_objects:
        return None

    touch_count = 0
    free_count = 0
    for comp in grey_comps:
        counts = _contact_counts(comp, frame_top, frame_left, frame_top + frame_h - 1, frame_left + frame_w - 1)
        if max(counts.values()) == 0:
            free_count += 1
        else:
            touch_count += 1
    if touch_count == 0 or free_count == 0:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _build_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
