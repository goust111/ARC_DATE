from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
WALL_COLORS = (1, 2, 3)
INDICATOR = 4
TARGET = 7

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

GEN_RETRIES = 600


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for dr, dc in DIR4]


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for dr, dc in DIR8]


def _component_list(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]] | np.ndarray) -> tuple[int, int, int, int]:
    if isinstance(coords, np.ndarray):
        rows = coords[:, 0]
        cols = coords[:, 1]
        return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _parse_rectangles(grid: np.ndarray) -> tuple[list[dict], np.ndarray]:
    wall_colors = [int(c) for c in np.unique(grid) if int(c) not in {BG, INDICATOR, TARGET}]
    assert 2 <= len(wall_colors) <= 3, "expected 2-3 wall colors"

    rects: list[dict] = []
    occupied = np.zeros(grid.shape, dtype=bool)
    wall_mask = np.zeros(grid.shape, dtype=bool)
    for color in wall_colors:
        coords = np.argwhere(grid == color)
        assert coords.shape == (2, 2), "each wall color must appear exactly twice"
        r0, c0, r1, c1 = _bbox(coords)
        assert r0 < r1 and c0 < c1, "corner markers must be diagonal"
        rect_mask = np.zeros(grid.shape, dtype=bool)
        rect_mask[r0 : r1 + 1, c0 : c1 + 1] = True
        assert not np.any(occupied & rect_mask), "filled wall rectangles must not overlap"
        occupied |= rect_mask
        wall_mask |= rect_mask
        rects.append({"color": color, "bbox": (r0, c0, r1, c1)})
    rects.sort(key=lambda item: item["color"])
    return rects, wall_mask


def _indicator_pos(grid: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(grid == INDICATOR)
    assert coords.shape == (1, 2), "expected one yellow indicator"
    return tuple(map(int, coords[0]))


def _flood_reachable(shape: tuple[int, int], wall_mask: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    h, w = shape
    out = np.zeros(shape, dtype=bool)
    if wall_mask[start]:
        return out
    q = deque([start])
    out[start] = True
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not wall_mask[nr, nc] and not out[nr, nc]:
                out[nr, nc] = True
                q.append((nr, nc))
    return out


def _orange_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    return _component_list(grid == TARGET)


def _paint_walls(grid: np.ndarray, rects: list[dict]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for rect in rects:
        color = int(rect["color"])
        r0, c0, r1, c1 = rect["bbox"]
        region = out[r0 : r1 + 1, c0 : c1 + 1]
        fill_mask = (region != INDICATOR) & (region != TARGET)
        region[fill_mask] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each same-color corner pair into a wall rectangle, then erase only the orange objects that the yellow seed can flood-reach.

    Input structure:
    - The background is black.
    - There are 2-3 non-black/non-yellow/non-orange colors, each appearing exactly twice as isolated diagonal corner markers.
    - One yellow singleton marks the flood source outside the wall system, and the orange pixels form several separate 4-connected target objects.

    Rule:
    - For each paired wall color, take the inclusive bounding box of its two markers and treat that whole rectangle as a wall region.
    - Paint those wall rectangles in their own colors, but keep yellow and orange cells visible while computing reachability through only non-wall cells.
    - Flood-fill 4-connectedly from the yellow pixel, blocked by all wall-rectangle cells.
    - Erase an orange object to black iff every pixel of that object lies in the flood-reachable region; otherwise keep the whole object unchanged.

    Key insight:
    - The colored pairs are not tiny markers to ignore: each pair expands into a hidden blocking rectangle, and the yellow pixel matters only because flood reachability, not raw position, decides which orange objects disappear.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    rects, wall_mask = _parse_rectangles(g)
    indicator = _indicator_pos(g)
    assert not wall_mask[indicator], "indicator must be outside all wall rectangles"
    reachable = _flood_reachable(g.shape, wall_mask, indicator)

    out = _paint_walls(g, rects)
    for comp in _orange_components(g):
        if all(reachable[r, c] for r, c in comp):
            for r, c in comp:
                out[r, c] = BG
    return out


def _mark_halo(mask: np.ndarray, cells: list[tuple[int, int]] | set[tuple[int, int]]) -> None:
    h, w = mask.shape
    for r, c in cells:
        for nr, nc in _neighbors8(r, c) + [(r, c)]:
            if 0 <= nr < h and 0 <= nc < w:
                mask[nr, nc] = True


def _rect_mask(shape: tuple[int, int], rects: list[dict]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for rect in rects:
        r0, c0, r1, c1 = rect["bbox"]
        mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _boundary_cells(shape: tuple[int, int], orient: str, side: str) -> list[tuple[int, int]]:
    h, w = shape
    if orient == "vertical":
        col = 0 if side == "left" else w - 1
        return [(r, col) for r in range(h)]
    row = 0 if side == "top" else h - 1
    return [(row, c) for c in range(w)]


def _random_marker_pair(r0: int, c0: int, r1: int, c1: int) -> tuple[tuple[int, int], tuple[int, int]]:
    if random.random() < 0.5:
        return (r0, c0), (r1, c1)
    return (r0, c1), (r1, c0)


def _make_rect(color: int, bbox: tuple[int, int, int, int], role: str) -> dict:
    r0, c0, r1, c1 = bbox
    return {
        "color": color,
        "bbox": bbox,
        "markers": _random_marker_pair(r0, c0, r1, c1),
        "role": role,
    }


def _sample_layout() -> tuple[tuple[int, int], list[dict], str, str] | None:
    h = random.randint(13, 16)
    w = random.randint(13, 16)
    orient = random.choice(["vertical", "horizontal"])
    n_rects = random.choice([2, 3])
    colors = random.sample(list(WALL_COLORS), n_rects)

    if orient == "vertical":
        barrier_w = random.randint(2, 3)
        candidates = [
            c0
            for c0 in range(4, w - barrier_w - 3)
            if c0 >= 5 and w - (c0 + barrier_w) >= 5
        ]
        if not candidates:
            return None
        c0 = random.choice(candidates)
        barrier = _make_rect(colors[0], (0, c0, h - 1, c0 + barrier_w - 1), "barrier")
        left_w = c0 - 2
        right_w = w - (c0 + barrier_w) - 1
        if left_w < 4 or right_w < 4:
            return None

        rects = [barrier]
        if n_rects == 2:
            side = random.choice(["left", "right"])
            side_c0 = 1 if side == "left" else c0 + barrier_w + 1
            side_c1 = c0 - 2 if side == "left" else w - 2
            max_w = side_c1 - side_c0 + 1
            if max_w < 4:
                return None
            rh = random.randint(4, min(6, h - 2))
            rw = random.randint(4, min(6, max_w))
            rr0 = random.randint(1, h - rh - 1)
            cc0 = random.randint(side_c0, side_c1 - rw + 1)
            rects.append(_make_rect(colors[1], (rr0, cc0, rr0 + rh - 1, cc0 + rw - 1), "host"))
        else:
            for idx, side in enumerate(["left", "right"], start=1):
                side_c0 = 1 if side == "left" else c0 + barrier_w + 1
                side_c1 = c0 - 2 if side == "left" else w - 2
                max_w = side_c1 - side_c0 + 1
                if max_w < 4:
                    return None
                rh = random.randint(4, min(6, h - 2))
                rw = random.randint(4, min(6, max_w))
                rr0 = random.randint(1, h - rh - 1)
                cc0 = random.randint(side_c0, side_c1 - rw + 1)
                rects.append(_make_rect(colors[idx], (rr0, cc0, rr0 + rh - 1, cc0 + rw - 1), "host"))
        return (h, w), rects, orient, random.choice(["left", "right"])

    barrier_h = random.randint(2, 3)
    candidates = [
        r0
        for r0 in range(4, h - barrier_h - 3)
        if r0 >= 5 and h - (r0 + barrier_h) >= 5
    ]
    if not candidates:
        return None
    r0 = random.choice(candidates)
    barrier = _make_rect(colors[0], (r0, 0, r0 + barrier_h - 1, w - 1), "barrier")
    top_h = r0 - 2
    bottom_h = h - (r0 + barrier_h) - 1
    if top_h < 4 or bottom_h < 4:
        return None

    rects = [barrier]
    if n_rects == 2:
        side = random.choice(["top", "bottom"])
        side_r0 = 1 if side == "top" else r0 + barrier_h + 1
        side_r1 = r0 - 2 if side == "top" else h - 2
        max_h = side_r1 - side_r0 + 1
        if max_h < 4:
            return None
        rh = random.randint(4, min(6, max_h))
        rw = random.randint(4, min(6, w - 2))
        rr0 = random.randint(side_r0, side_r1 - rh + 1)
        cc0 = random.randint(1, w - rw - 1)
        rects.append(_make_rect(colors[1], (rr0, cc0, rr0 + rh - 1, cc0 + rw - 1), "host"))
    else:
        for idx, side in enumerate(["top", "bottom"], start=1):
            side_r0 = 1 if side == "top" else r0 + barrier_h + 1
            side_r1 = r0 - 2 if side == "top" else h - 2
            max_h = side_r1 - side_r0 + 1
            if max_h < 4:
                return None
            rh = random.randint(4, min(6, max_h))
            rw = random.randint(4, min(6, w - 2))
            rr0 = random.randint(side_r0, side_r1 - rh + 1)
            cc0 = random.randint(1, w - rw - 1)
            rects.append(_make_rect(colors[idx], (rr0, cc0, rr0 + rh - 1, cc0 + rw - 1), "host"))
    return (h, w), rects, orient, random.choice(["top", "bottom"])


def _choose_indicator(
    shape: tuple[int, int],
    wall_mask: np.ndarray,
    marker_cells: set[tuple[int, int]],
    orient: str,
    side: str,
) -> tuple[int, int] | None:
    forbidden = np.zeros(shape, dtype=bool)
    _mark_halo(forbidden, marker_cells)
    candidates = [
        cell
        for cell in _boundary_cells(shape, orient, side)
        if not wall_mask[cell] and not forbidden[cell]
    ]
    if not candidates:
        return None
    return random.choice(candidates)


def _place_object(
    allowed: np.ndarray,
    blocked: np.ndarray,
    size_low: int = 2,
    size_high: int = 5,
) -> list[tuple[int, int]] | None:
    h, w = allowed.shape
    free = [(r, c) for r in range(h) for c in range(w) if allowed[r, c] and not blocked[r, c]]
    if len(free) < size_low:
        return None

    max_size = min(size_high, len(free))
    for _ in range(120):
        size = random.randint(size_low, max_size)
        seed = random.choice(free)
        obj = {seed}
        while len(obj) < size:
            frontier = sorted(
                {
                    nbr
                    for r, c in obj
                    for nbr in _neighbors4(r, c)
                    if 0 <= nbr[0] < h
                    and 0 <= nbr[1] < w
                    and allowed[nbr]
                    and not blocked[nbr]
                    and nbr not in obj
                }
            )
            if not frontier:
                break
            obj.add(random.choice(frontier))
        if len(obj) == size:
            return sorted(obj)
    return None


def _strict_interior_mask(shape: tuple[int, int], rect: dict) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = rect["bbox"]
    if r1 - r0 >= 2 and c1 - c0 >= 2:
        mask[r0 + 1 : r1, c0 + 1 : c1] = True
    return mask


def _outside_reachable_masks(
    shape: tuple[int, int],
    rects: list[dict],
    indicator: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wall_mask = _rect_mask(shape, rects)
    reachable = _flood_reachable(shape, wall_mask, indicator)
    exterior = ~wall_mask
    return reachable, reachable & exterior, (~reachable) & exterior


def _render_input(
    shape: tuple[int, int],
    rects: list[dict],
    indicator: tuple[int, int],
    orange_objects: list[list[tuple[int, int]]],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for rect in rects:
        for r, c in rect["markers"]:
            grid[r, c] = int(rect["color"])
    grid[indicator] = INDICATOR
    for obj in orange_objects:
        for r, c in obj:
            grid[r, c] = TARGET
    return grid


def _validate_generation(
    grid: np.ndarray,
    orient: str,
    indicator_side: str,
    opposite_side: str,
) -> bool:
    rects, wall_mask = _parse_rectangles(grid)
    indicator = _indicator_pos(grid)
    reachable = _flood_reachable(grid.shape, wall_mask, indicator)
    corner_only = np.zeros_like(wall_mask)
    for rect in rects:
        coords = np.argwhere(grid == int(rect["color"]))
        for r, c in coords:
            corner_only[r, c] = True
    reachable_corner_only = _flood_reachable(grid.shape, corner_only, indicator)

    orange = _orange_components(grid)
    if len(orange) < 3:
        return False

    inside_count = 0
    outside_reach = 0
    outside_unreach = 0
    corner_only_changes = 0
    for comp in orange:
        inside = any(
            all(r0 < r < r1 and c0 < c < c1 for r, c in comp)
            for (_color, r0, c0, r1, c1) in [
                (int(rect["color"]), *rect["bbox"]) for rect in rects
            ]
        )
        if inside:
            inside_count += 1
            if any(reachable_corner_only[r, c] for r, c in comp):
                corner_only_changes += 1
            continue
        if not all(not wall_mask[r, c] for r, c in comp):
            return False
        if all(reachable[r, c] for r, c in comp):
            outside_reach += 1
        elif all(not reachable[r, c] for r, c in comp):
            outside_unreach += 1
        else:
            return False

    if not (inside_count >= 1 and outside_reach >= 1 and outside_unreach >= 1):
        return False
    if corner_only_changes == 0:
        return False

    opposite_candidates = [
        cell
        for cell in _boundary_cells(grid.shape, orient, opposite_side)
        if not wall_mask[cell] and grid[cell] == BG
    ]
    if not opposite_candidates:
        return False
    alt_indicator = random.choice(opposite_candidates)
    alt_reachable = _flood_reachable(grid.shape, wall_mask, alt_indicator)
    if not any(
        all(reachable[r, c] for r, c in comp) != all(alt_reachable[r, c] for r, c in comp)
        for comp in orange
        if all(not wall_mask[r, c] for r, c in comp)
    ):
        return False

    out = main(grid)
    erased = 0
    kept = 0
    for comp in orange:
        erased_now = all(out[r, c] == BG for r, c in comp)
        if erased_now:
            erased += 1
        else:
            kept += 1
    if erased == 0 or kept == 0:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    layout = _sample_layout()
    if layout is None:
        return None
    shape, rects, orient, indicator_side = layout
    opposite_side = {
        ("vertical", "left"): "right",
        ("vertical", "right"): "left",
        ("horizontal", "top"): "bottom",
        ("horizontal", "bottom"): "top",
    }[(orient, indicator_side)]

    wall_mask = _rect_mask(shape, rects)
    marker_cells = {cell for rect in rects for cell in rect["markers"]}
    indicator = _choose_indicator(shape, wall_mask, marker_cells, orient, indicator_side)
    if indicator is None:
        return None

    reachable, outside_reachable, outside_unreachable = _outside_reachable_masks(shape, rects, indicator)
    blocked = np.zeros(shape, dtype=bool)
    _mark_halo(blocked, marker_cells | {indicator})

    orange_objects: list[list[tuple[int, int]]] = []

    inside_hosts = [rect for rect in rects if rect["role"] == "host"]
    inside_mask = _strict_interior_mask(shape, random.choice(inside_hosts))
    inside_obj = _place_object(inside_mask, blocked)
    if inside_obj is None:
        return None
    orange_objects.append(inside_obj)
    _mark_halo(blocked, inside_obj)

    outside_obj_reach = _place_object(outside_reachable, blocked)
    if outside_obj_reach is None:
        return None
    orange_objects.append(outside_obj_reach)
    _mark_halo(blocked, outside_obj_reach)

    outside_obj_unreach = _place_object(outside_unreachable, blocked)
    if outside_obj_unreach is None:
        return None
    orange_objects.append(outside_obj_unreach)
    _mark_halo(blocked, outside_obj_unreach)

    extra_masks: list[np.ndarray] = []
    if len(inside_hosts) > 1:
        for rect in inside_hosts:
            mask = _strict_interior_mask(shape, rect)
            if np.any(mask):
                extra_masks.append(mask)
    extra_masks.extend([outside_reachable, outside_unreachable])
    random.shuffle(extra_masks)
    for mask in extra_masks:
        if len(orange_objects) >= 5 or random.random() < 0.45:
            continue
        obj = _place_object(mask, blocked)
        if obj is None:
            continue
        orange_objects.append(obj)
        _mark_halo(blocked, obj)

    grid = _render_input(shape, rects, indicator, orange_objects)
    if not _validate_generation(grid, orient, indicator_side, opposite_side):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid paired-wall flood-erase puzzle")
