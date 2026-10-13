from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
WALL = 2
ARROW_COLORS = (1, 3, 4, 7)
SLOTS = ("TL", "TR", "BL", "BR")
ROOM_SIZE_POOL = (
    (5, 5),
    (5, 6),
    (6, 5),
    (6, 6),
    (5, 7),
    (7, 5),
    (6, 7),
    (7, 6),
)
MAX_GEN_ATTEMPTS = 400

# Arrow cells are encoded relative to the tip pixel.
ARROW_REL = {
    "right": ((0, -3), (0, -2), (0, -1), (0, 0), (-1, -2), (1, -2)),
    "left": ((0, 0), (0, 1), (0, 2), (0, 3), (-1, 2), (1, 2)),
    "down": ((-3, 0), (-2, 0), (-1, 0), (0, 0), (-2, -1), (-2, 1)),
    "up": ((0, 0), (1, 0), (2, 0), (3, 0), (2, -1), (2, 1)),
}
DIR_STEP = {
    "right": (0, 1),
    "left": (0, -1),
    "down": (1, 0),
    "up": (-1, 0),
}


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _components_mask(mask: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    components: list[list[tuple[int, int]]] = []
    current = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            current += 1
            labels[r, c] = current
            queue = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = current
                        queue.append((nr, nc))
            components.append(coords)
    return labels, components


def _exterior_background(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    exterior = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    def add(r: int, c: int) -> None:
        if grid[r, c] == BG and not exterior[r, c]:
            exterior[r, c] = True
            queue.append((r, c))

    for r in range(h):
        add(r, 0)
        add(r, w - 1)
    for c in range(w):
        add(0, c)
        add(h - 1, c)

    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(grid.shape, nr, nc):
                add(nr, nc)
    return exterior


def _arrow_from_coords(grid: np.ndarray, coords: list[tuple[int, int]]) -> dict:
    pts = [(int(r), int(c)) for r, c in coords]
    color = int(grid[pts[0]])
    row_counts = Counter(r for r, _c in pts)
    col_counts = Counter(c for _r, c in pts)

    long_rows = [r for r, n in row_counts.items() if n == 4]
    long_cols = [c for c, n in col_counts.items() if n == 4]
    cross_rows = [r for r, n in row_counts.items() if n == 3]
    cross_cols = [c for c, n in col_counts.items() if n == 3]

    horizontal = len(long_rows) == 1 and len(cross_cols) == 1 and len(long_cols) == 0
    vertical = len(long_cols) == 1 and len(cross_rows) == 1 and len(long_rows) == 0
    assert horizontal ^ vertical, "expected a six-pixel asymmetric T arrow"

    if horizontal:
        bar_r = long_rows[0]
        cross_c = cross_cols[0]
        line_cols = sorted(c for r, c in pts if r == bar_r)
        cross_rs = sorted(r for r, c in pts if c == cross_c)
        assert line_cols == list(range(line_cols[0], line_cols[0] + 4))
        assert cross_rs == [bar_r - 1, bar_r, bar_r + 1]
        left_dist = cross_c - line_cols[0]
        right_dist = line_cols[-1] - cross_c
        assert {left_dist, right_dist} == {1, 2}
        if left_dist < right_dist:
            return {"color": color, "tip": (bar_r, line_cols[-1]), "dir": (0, 1), "coords": pts}
        return {"color": color, "tip": (bar_r, line_cols[0]), "dir": (0, -1), "coords": pts}

    bar_c = long_cols[0]
    cross_r = cross_rows[0]
    line_rows = sorted(r for r, c in pts if c == bar_c)
    cross_cs = sorted(c for r, c in pts if r == cross_r)
    assert line_rows == list(range(line_rows[0], line_rows[0] + 4))
    assert cross_cs == [bar_c - 1, bar_c, bar_c + 1]
    up_dist = cross_r - line_rows[0]
    down_dist = line_rows[-1] - cross_r
    assert {up_dist, down_dist} == {1, 2}
    if up_dist < down_dist:
        return {"color": color, "tip": (line_rows[-1], bar_c), "dir": (1, 0), "coords": pts}
    return {"color": color, "tip": (line_rows[0], bar_c), "dir": (-1, 0), "coords": pts}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire each enclosed-region arrow outward with a same-color ray whose length equals that region's free background area.

    Input structure:
    - The grid has black background, red one-cell-thick closed walls, and 2 to 4 enclosed regions.
    - Each enclosed region contains exactly one monochrome six-pixel asymmetric T arrow in blue, green, yellow, or orange.
    - The arrow's long bar is horizontal or vertical, and the crossbar is offset toward the tail end.

    Rule:
    - Flood-fill black cells from the grid boundary to mark the exterior; the remaining black cells form the enclosed regions.
    - For each arrow, the crossbar-near end is the tail, so the opposite bar end is the tip and defines the firing direction.
    - Count the enclosed region's black cells to get the ray length, then paint that many cells starting at the tip and extending straight through walls and exterior space, truncating at the grid edge.

    Key insight:
    - The ray length is hidden in the amount of free background inside the arrow's own sealed room, so the solver must first recover region ownership before extending the ray.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    exterior = _exterior_background(g)
    enclosed_mask = (g == BG) & ~exterior
    region_labels, regions = _components_mask(enclosed_mask)
    assert 2 <= len(regions) <= 4, "expected 2 to 4 enclosed regions"
    region_sizes = {index + 1: len(coords) for index, coords in enumerate(regions)}

    arrow_mask = (g != BG) & (g != WALL)
    _, arrow_components = _components_mask(arrow_mask)
    assert len(arrow_components) == len(regions), "expected one arrow per enclosed region"

    out = g.copy()
    used_regions: set[int] = set()
    for coords in arrow_components:
        assert len(coords) == 6, "each arrow must have exactly 6 pixels"
        colors = {int(g[r, c]) for r, c in coords}
        assert len(colors) == 1 and next(iter(colors)) in ARROW_COLORS
        arrow = _arrow_from_coords(g, coords)

        adjacent_regions = {
            int(region_labels[nr, nc])
            for r, c in arrow["coords"]
            for nr, nc in _neighbors4(r, c)
            if _in_bounds(g.shape, nr, nc) and region_labels[nr, nc] > 0
        }
        assert len(adjacent_regions) == 1, "each arrow must belong to exactly one enclosed region"
        region_id = adjacent_regions.pop()
        assert region_id not in used_regions, "multiple arrows mapped to the same region"
        used_regions.add(region_id)

        length = region_sizes[region_id]
        tip_r, tip_c = arrow["tip"]
        dr, dc = arrow["dir"]
        for step in range(length):
            rr = tip_r + dr * step
            cc = tip_c + dc * step
            if not _in_bounds(g.shape, rr, cc):
                break
            out[rr, cc] = arrow["color"]

    return out


def _inside_bbox(cell: tuple[int, int], room: dict) -> bool:
    r, c = cell
    return (
        room["top"] <= r < room["top"] + room["outer_h"]
        and room["left"] <= c < room["left"] + room["outer_w"]
    )


def _slot_selection(count: int) -> tuple[str, ...]:
    if count == 2:
        return random.choice((("TL", "BR"), ("TR", "BL")))
    if count == 3:
        omitted = random.choice(SLOTS)
        return tuple(slot for slot in SLOTS if slot != omitted)
    return SLOTS


def _layout_rooms() -> dict | None:
    count = random.randint(2, 4)
    slots = _slot_selection(count)
    sizes = {slot: random.choice(ROOM_SIZE_POOL) for slot in slots}

    top_slots = [slot for slot in slots if slot.startswith("T")]
    bottom_slots = [slot for slot in slots if slot.startswith("B")]
    left_slots = [slot for slot in slots if slot.endswith("L")]
    right_slots = [slot for slot in slots if slot.endswith("R")]

    top_band_h = max((sizes[slot][0] + 2 for slot in top_slots), default=0)
    bottom_band_h = max((sizes[slot][0] + 2 for slot in bottom_slots), default=0)
    left_band_w = max((sizes[slot][1] + 2 for slot in left_slots), default=0)
    right_band_w = max((sizes[slot][1] + 2 for slot in right_slots), default=0)

    margin_top = random.randint(1, 3)
    margin_bottom = random.randint(1, 3)
    margin_left = random.randint(1, 3)
    margin_right = random.randint(1, 3)
    gap_row = random.randint(4, 8)
    gap_col = random.randint(4, 8)

    height = margin_top + top_band_h + gap_row + bottom_band_h + margin_bottom
    width = margin_left + left_band_w + gap_col + right_band_w + margin_right
    if max(height, width) > 30:
        return None

    top_band = (margin_top, margin_top + top_band_h - 1)
    bottom_band = (margin_top + top_band_h + gap_row, margin_top + top_band_h + gap_row + bottom_band_h - 1)
    left_band = (margin_left, margin_left + left_band_w - 1)
    right_band = (margin_left + left_band_w + gap_col, margin_left + left_band_w + gap_col + right_band_w - 1)

    rooms: list[dict] = []
    for slot in slots:
        inner_h, inner_w = sizes[slot]
        outer_h, outer_w = inner_h + 2, inner_w + 2
        band_r0, band_r1 = top_band if slot.startswith("T") else bottom_band
        band_c0, band_c1 = left_band if slot.endswith("L") else right_band
        max_r_jitter = (band_r1 - band_r0 + 1) - outer_h
        max_c_jitter = (band_c1 - band_c0 + 1) - outer_w
        if max_r_jitter < 0 or max_c_jitter < 0:
            return None
        top = band_r0 + random.randint(0, max_r_jitter)
        left = band_c0 + random.randint(0, max_c_jitter)
        rooms.append(
            {
                "slot": slot,
                "top": top,
                "left": left,
                "outer_h": outer_h,
                "outer_w": outer_w,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "area": inner_h * inner_w - 6,
            }
        )
    return {"height": height, "width": width, "rooms": rooms}


def _arrow_candidates(room: dict, rooms: list[dict], shape: tuple[int, int]) -> list[dict]:
    inner_r0 = room["top"] + 1
    inner_r1 = room["top"] + room["outer_h"] - 2
    inner_c0 = room["left"] + 1
    inner_c1 = room["left"] + room["outer_w"] - 2
    others = [other for other in rooms if other is not room]
    candidates: list[dict] = []

    for direction, rels in ARROW_REL.items():
        drs = [dr for dr, _dc in rels]
        dcs = [dc for _dr, dc in rels]
        tip_r_min = inner_r0 - min(drs)
        tip_r_max = inner_r1 - max(drs)
        tip_c_min = inner_c0 - min(dcs)
        tip_c_max = inner_c1 - max(dcs)

        if direction == "up":
            tip_r_min = max(tip_r_min, inner_r0 + 1)
        elif direction == "down":
            tip_r_max = min(tip_r_max, inner_r1 - 1)
        elif direction == "left":
            tip_c_min = max(tip_c_min, inner_c0 + 1)
        else:
            tip_c_max = min(tip_c_max, inner_c1 - 1)

        if tip_r_min > tip_r_max or tip_c_min > tip_c_max:
            continue

        step_r, step_c = DIR_STEP[direction]
        for tip_r in range(tip_r_min, tip_r_max + 1):
            for tip_c in range(tip_c_min, tip_c_max + 1):
                arrow_cells = {(tip_r + dr, tip_c + dc) for dr, dc in rels}
                ray_cells: list[tuple[int, int]] = []
                for step in range(room["area"]):
                    rr = tip_r + step_r * step
                    cc = tip_c + step_c * step
                    if not _in_bounds(shape, rr, cc):
                        break
                    ray_cells.append((rr, cc))
                if not ray_cells:
                    continue

                ray_outside = [cell for cell in ray_cells if not _inside_bbox(cell, room)]
                if any(_inside_bbox(cell, other) for cell in ray_outside for other in others):
                    continue

                candidates.append(
                    {
                        "direction": direction,
                        "tip": (tip_r, tip_c),
                        "arrow_cells": arrow_cells,
                        "ray_cells": tuple(ray_cells),
                        "ray_set": set(ray_cells),
                        "truncated": len(ray_cells) < room["area"],
                    }
                )
    random.shuffle(candidates)
    return candidates


def _select_candidates(rooms: list[dict], room_candidates: dict[int, list[dict]], require_non_truncated: bool) -> list[dict] | None:
    order = sorted(range(len(rooms)), key=lambda idx: len(room_candidates[idx]))
    chosen: list[dict] = []
    used_rays: set[tuple[int, int]] = set()

    def backtrack(pos: int) -> bool:
        if pos == len(order):
            return (not require_non_truncated) or any(not entry["candidate"]["truncated"] for entry in chosen)

        room_index = order[pos]
        candidates = room_candidates[room_index][:]
        random.shuffle(candidates)
        for candidate in candidates:
            if candidate["ray_set"] & used_rays:
                continue
            chosen.append({"room_index": room_index, "candidate": candidate})
            used_rays.update(candidate["ray_set"])
            if backtrack(pos + 1):
                return True
            used_rays.difference_update(candidate["ray_set"])
            chosen.pop()
        return False

    if backtrack(0):
        return chosen
    return None


def _draw_room(grid: np.ndarray, room: dict) -> None:
    top, left = room["top"], room["left"]
    bottom = top + room["outer_h"] - 1
    right = left + room["outer_w"] - 1
    grid[top, left : right + 1] = WALL
    grid[bottom, left : right + 1] = WALL
    grid[top : bottom + 1, left] = WALL
    grid[top : bottom + 1, right] = WALL


def _generate_attempt() -> np.ndarray | None:
    layout = _layout_rooms()
    if layout is None:
        return None
    height = int(layout["height"])
    width = int(layout["width"])
    rooms = list(layout["rooms"])

    room_candidates = {idx: _arrow_candidates(room, rooms, (height, width)) for idx, room in enumerate(rooms)}
    if any(not room_candidates[idx] for idx in range(len(rooms))):
        return None

    require_non_truncated = random.random() < 0.55
    chosen = _select_candidates(rooms, room_candidates, require_non_truncated=require_non_truncated)
    if chosen is None:
        return None

    by_index = {entry["room_index"]: entry["candidate"] for entry in chosen}
    grid = np.full((height, width), BG, dtype=int)

    colors = random.sample(list(ARROW_COLORS), len(rooms))
    for room in rooms:
        _draw_room(grid, room)

    for idx, room in enumerate(rooms):
        color = colors[idx]
        candidate = by_index[idx]
        for r, c in candidate["arrow_cells"]:
            grid[r, c] = color

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    if len({entry["candidate"]["direction"] for entry in chosen}) < 2 and len(rooms) >= 3:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid enclosed-arrow-ray puzzle")
