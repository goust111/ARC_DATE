from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 400
PLACE_RETRIES = 250


def _prefix(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(int), ((1, 0), (1, 0)))
    return padded.cumsum(axis=0).cumsum(axis=1)


def _rect_sum(prefix: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    return int(prefix[r1, c1] - prefix[r0, c1] - prefix[r1, c0] + prefix[r0, c0])


def _window_offsets(side: np.ndarray) -> list[int] | None:
    if side.size == 0:
        return []
    if np.any((side != BG) & (side != GREY)):
        return None
    windows = side == BG
    if np.any(windows[:-1] & windows[1:]):
        return None
    return [int(idx) for idx in np.flatnonzero(windows)]


def _room_candidate(
    grid: np.ndarray,
    grey_prefix: np.ndarray,
    marker_prefix: np.ndarray,
    marker: tuple[int, int],
    top: int,
    bottom: int,
    left: int,
    right: int,
):
    if bottom - top < 2 or right - left < 2:
        return None
    if (
        int(grid[top, left]) != GREY
        or int(grid[top, right]) != GREY
        or int(grid[bottom, left]) != GREY
        or int(grid[bottom, right]) != GREY
    ):
        return None

    if _rect_sum(grey_prefix, top + 1, left + 1, bottom, right) != 0:
        return None
    if _rect_sum(marker_prefix, top + 1, left + 1, bottom, right) != 1:
        return None
    mr, mc = marker
    if not (top < mr < bottom and left < mc < right):
        return None

    top_side = _window_offsets(grid[top, left + 1 : right])
    bottom_side = _window_offsets(grid[bottom, left + 1 : right])
    left_side = _window_offsets(grid[top + 1 : bottom, left])
    right_side = _window_offsets(grid[top + 1 : bottom, right])
    if None in (top_side, bottom_side, left_side, right_side):
        return None

    windows: list[tuple[int, int, int, int]] = []
    for offset in top_side:
        windows.append((top, left + 1 + offset, 1, 0))
    for offset in bottom_side:
        windows.append((bottom, left + 1 + offset, -1, 0))
    for offset in left_side:
        windows.append((top + 1 + offset, left, 0, 1))
    for offset in right_side:
        windows.append((top + 1 + offset, right, 0, -1))

    if not windows:
        return None

    return {
        "top": top,
        "bottom": bottom,
        "left": left,
        "right": right,
        "marker": marker,
        "color": int(grid[mr, mc]),
        "windows": windows,
        "window_count": len(windows),
        "area": (bottom - top + 1) * (right - left + 1),
    }


def _find_room_for_marker(
    grid: np.ndarray, grey_prefix: np.ndarray, marker_prefix: np.ndarray, marker: tuple[int, int]
):
    h, w = grid.shape
    mr, mc = marker
    best = None
    for top in range(mr - 1, -1, -1):
        for bottom in range(mr + 1, h):
            for left in range(mc - 1, -1, -1):
                for right in range(mc + 1, w):
                    candidate = _room_candidate(grid, grey_prefix, marker_prefix, marker, top, bottom, left, right)
                    if candidate is None:
                        continue
                    if best is None or (
                        candidate["area"],
                        candidate["top"],
                        candidate["left"],
                        candidate["bottom"],
                        candidate["right"],
                    ) < (
                        best["area"],
                        best["top"],
                        best["left"],
                        best["bottom"],
                        best["right"],
                    ):
                        best = candidate
    if best is None:
        raise AssertionError(f"main: failed to locate a valid room for marker at {marker}")
    return best


def _find_rooms(grid: np.ndarray) -> list[dict[str, object]]:
    markers = [
        (int(r), int(c))
        for r, c in np.argwhere((grid != BG) & (grid != GREY))
    ]
    if not markers:
        raise AssertionError("main: expected at least one colored room marker")

    grey_prefix = _prefix(grid == GREY)
    marker_prefix = _prefix((grid != BG) & (grid != GREY))
    rooms = [_find_room_for_marker(grid, grey_prefix, marker_prefix, marker) for marker in markers]

    bounds = set()
    for room in rooms:
        key = (room["top"], room["bottom"], room["left"], room["right"])
        if key in bounds:
            raise AssertionError("main: duplicate room detection")
        bounds.add(key)

    rooms.sort(key=lambda room: (room["top"], room["left"], room["marker"]))
    return rooms


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast colored interior rays from each room window, with ray length set by that room's window count.

    Input structure:
    - The background is black.
    - Several axis-aligned rectangular rooms are outlined by one-pixel-thick grey walls, and each wall may contain isolated one-pixel black gaps that act as windows.
    - Every room interior contains exactly one non-grey, non-black single-pixel marker color, and room interiors otherwise stay black.

    Rule:
    - Identify each room by the unique colored marker inside its rectangle, then count all one-pixel wall gaps on that room's boundary.
    - For every window, cast a ray from the gap inward, using the room marker's color, for exactly N steps where N is that room's total window count.
    - A ray paints only black interior cells; it stops before a wall, marker, or previously painted ray. Windows and walls themselves remain unchanged.

    Key insight:
    - The same hidden quantity, the number of windows on a room, simultaneously determines both which boundary cells are active emitters and how long every emitted ray should be.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for room in _find_rooms(g):
        top = int(room["top"])
        bottom = int(room["bottom"])
        left = int(room["left"])
        right = int(room["right"])
        color = int(room["color"])
        ray_len = int(room["window_count"])
        for wr, wc, dr, dc in room["windows"]:
            cr, cc = int(wr), int(wc)
            for _ in range(ray_len):
                nr, nc = cr + int(dr), cc + int(dc)
                if not (top < nr < bottom and left < nc < right):
                    break
                if int(out[nr, nc]) != BG:
                    break
                out[nr, nc] = color
                cr, cc = nr, nc

    return out


def _all_boundary_positions(inner_h: int, inner_w: int, side: str) -> list[int]:
    return list(range(1, inner_w + 1)) if side in ("top", "bottom") else list(range(1, inner_h + 1))


def _pick_offset(rng: random.Random, inner_h: int, inner_w: int, side: str, banned: set[int] | None = None) -> int:
    choices = [value for value in _all_boundary_positions(inner_h, inner_w, side) if banned is None or value not in banned]
    if not choices:
        raise ValueError("no valid boundary position left")
    return rng.choice(choices)


def _sample_room_template(rng: random.Random, template: str) -> dict[str, object]:
    if template == "open1":
        inner_h = rng.randint(3, 7)
        inner_w = rng.randint(3, 7)
        side = rng.choice(["top", "bottom", "left", "right"])
        windows = [(side, _pick_offset(rng, inner_h, inner_w, side))]
    elif template == "open2":
        inner_h = rng.randint(4, 7)
        inner_w = rng.randint(4, 7)
        side_a, side_b = rng.sample(["top", "bottom", "left", "right"], 2)
        windows = [
            (side_a, _pick_offset(rng, inner_h, inner_w, side_a)),
            (side_b, _pick_offset(rng, inner_h, inner_w, side_b)),
        ]
    elif template == "wall3":
        if rng.random() < 0.5:
            inner_h = 2
            inner_w = rng.randint(4, 6)
            top_col = _pick_offset(rng, inner_h, inner_w, "top")
            bottom_col = _pick_offset(rng, inner_h, inner_w, "bottom", {top_col})
            side = rng.choice(["left", "right"])
            side_row = _pick_offset(rng, inner_h, inner_w, side)
            windows = [("top", top_col), ("bottom", bottom_col), (side, side_row)]
        else:
            inner_h = rng.randint(4, 6)
            inner_w = 2
            left_row = _pick_offset(rng, inner_h, inner_w, "left")
            right_row = _pick_offset(rng, inner_h, inner_w, "right", {left_row})
            side = rng.choice(["top", "bottom"])
            side_col = _pick_offset(rng, inner_h, inner_w, side)
            windows = [("left", left_row), ("right", right_row), (side, side_col)]
    elif template == "collision2":
        inner_h = rng.randint(3, 6)
        inner_w = 3
        row = _pick_offset(rng, inner_h, inner_w, "left")
        windows = [("left", row), ("right", row)]
    elif template == "collision4":
        inner_h = rng.choice([5, 7])
        inner_w = rng.choice([5, 7])
        row = inner_h // 2 + 1
        col = inner_w // 2 + 1
        windows = [("top", col), ("bottom", col), ("left", row), ("right", row)]
    else:
        raise ValueError(f"unknown template {template}")

    return {
        "inner_h": inner_h,
        "inner_w": inner_w,
        "outer_h": inner_h + 2,
        "outer_w": inner_w + 2,
        "windows": windows,
        "template": template,
    }


def _local_window_info(room: dict[str, object]) -> list[tuple[int, int, int, int]]:
    inner_h = int(room["inner_h"])
    inner_w = int(room["inner_w"])
    local = []
    for side, offset in room["windows"]:
        offset = int(offset)
        if side == "top":
            local.append((0, offset, 1, 0))
        elif side == "bottom":
            local.append((inner_h + 1, offset, -1, 0))
        elif side == "left":
            local.append((offset, 0, 0, 1))
        else:
            local.append((offset, inner_w + 1, 0, -1))
    return local


def _local_ray_cells(room: dict[str, object]) -> list[tuple[int, int]]:
    outer_h = int(room["outer_h"])
    outer_w = int(room["outer_w"])
    ray_len = len(room["windows"])
    canvas = np.full((outer_h, outer_w), BG, dtype=int)
    canvas[0, :] = GREY
    canvas[-1, :] = GREY
    canvas[:, 0] = GREY
    canvas[:, -1] = GREY
    for wr, wc, _dr, _dc in _local_window_info(room):
        canvas[wr, wc] = BG

    cells: list[tuple[int, int]] = []
    for wr, wc, dr, dc in _local_window_info(room):
        cr, cc = wr, wc
        for _ in range(ray_len):
            nr, nc = cr + dr, cc + dc
            if not (0 < nr < outer_h - 1 and 0 < nc < outer_w - 1):
                break
            if int(canvas[nr, nc]) != BG:
                break
            canvas[nr, nc] = 1
            cells.append((nr, nc))
            cr, cc = nr, nc
    return cells


def _choose_marker_cell(rng: random.Random, room: dict[str, object]) -> tuple[int, int] | None:
    forbidden = set(_local_ray_cells(room))
    inner_h = int(room["inner_h"])
    inner_w = int(room["inner_w"])
    choices = [
        (r, c)
        for r in range(1, inner_h + 1)
        for c in range(1, inner_w + 1)
        if (r, c) not in forbidden
    ]
    if not choices:
        return None
    return rng.choice(choices)


def _classify_room(room: dict[str, object]) -> str:
    outer_h = int(room["outer_h"])
    outer_w = int(room["outer_w"])
    ray_len = len(room["windows"])
    canvas = np.full((outer_h, outer_w), BG, dtype=int)
    canvas[0, :] = GREY
    canvas[-1, :] = GREY
    canvas[:, 0] = GREY
    canvas[:, -1] = GREY
    for wr, wc, _dr, _dc in _local_window_info(room):
        canvas[wr, wc] = BG

    hit_ray = False
    hit_wall = False
    for wr, wc, dr, dc in _local_window_info(room):
        cr, cc = wr, wc
        for _ in range(ray_len):
            nr, nc = cr + dr, cc + dc
            if not (0 < nr < outer_h - 1 and 0 < nc < outer_w - 1):
                hit_wall = True
                break
            cell = int(canvas[nr, nc])
            if cell == GREY:
                hit_wall = True
                break
            if cell != BG:
                hit_ray = True
                break
            canvas[nr, nc] = 1
            cr, cc = nr, nc
    if hit_ray:
        return "ray"
    if hit_wall:
        return "wall"
    return "open"


def _paint_room(
    grid: np.ndarray,
    top: int,
    left: int,
    room: dict[str, object],
    color: int,
    marker_local: tuple[int, int],
) -> None:
    outer_h = int(room["outer_h"])
    outer_w = int(room["outer_w"])
    bottom = top + outer_h - 1
    right = left + outer_w - 1

    grid[top : bottom + 1, left : right + 1] = BG
    grid[top, left : right + 1] = GREY
    grid[bottom, left : right + 1] = GREY
    grid[top : bottom + 1, left] = GREY
    grid[top : bottom + 1, right] = GREY

    for side, offset in room["windows"]:
        offset = int(offset)
        if side == "top":
            grid[top, left + offset] = BG
        elif side == "bottom":
            grid[bottom, left + offset] = BG
        elif side == "left":
            grid[top + offset, left] = BG
        else:
            grid[top + offset, right] = BG

    mr, mc = marker_local
    grid[top + mr, left + mc] = color


def _reserve(blocked: np.ndarray, top: int, left: int, bottom: int, right: int, pad: int = 1) -> None:
    h, w = blocked.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h - 1, bottom + pad)
    c1 = min(w - 1, right + pad)
    blocked[r0 : r1 + 1, c0 : c1 + 1] = True


def _template_plan(rng: random.Random, room_count: int) -> list[str]:
    open_templates = ["open1", "open2"]
    blocked_templates = ["wall3", "collision2", "collision4"]

    if room_count == 1:
        return [rng.choice(open_templates + blocked_templates)]

    plan = [rng.choice(open_templates), rng.choice(blocked_templates)]
    while len(plan) < room_count:
        plan.append(rng.choice(open_templates + blocked_templates))
    rng.shuffle(plan)
    return plan


def _grid_shape_for_rooms(rng: random.Random, room_count: int) -> tuple[int, int]:
    if room_count == 1:
        return rng.randint(10, 18), rng.randint(10, 18)
    if room_count == 2:
        return rng.randint(14, 24), rng.randint(14, 24)
    return rng.randint(20, 30), rng.randint(20, 30)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    room_count = rng.randint(1, 4)
    templates = _template_plan(rng, room_count)
    rooms = [_sample_room_template(rng, template) for template in templates]

    marker_locals = []
    for room in rooms:
        marker_local = _choose_marker_cell(rng, room)
        if marker_local is None:
            return None
        marker_locals.append(marker_local)

    h, w = _grid_shape_for_rooms(rng, room_count)
    if any(int(room["outer_h"]) + 2 > h or int(room["outer_w"]) + 2 > w for room in rooms):
        return None

    grid = np.full((h, w), BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)
    colors = rng.sample(MARKER_COLORS, room_count)

    placement_order = list(range(room_count))
    rng.shuffle(placement_order)
    placed = [None] * room_count
    for idx in placement_order:
        room = rooms[idx]
        outer_h = int(room["outer_h"])
        outer_w = int(room["outer_w"])
        candidates = []
        for _ in range(PLACE_RETRIES):
            top = rng.randint(1, h - outer_h - 1)
            left = rng.randint(1, w - outer_w - 1)
            bottom = top + outer_h - 1
            right = left + outer_w - 1
            if blocked[top - 1 : bottom + 2, left - 1 : right + 2].any():
                continue
            candidates.append((top, left, bottom, right))
        if not candidates:
            return None
        top, left, bottom, right = rng.choice(candidates)
        _paint_room(grid, top, left, room, colors[idx], marker_locals[idx])
        _reserve(blocked, top, left, bottom, right, pad=1)
        placed[idx] = (top, left, bottom, right)

    classifications = {_classify_room(room) for room in rooms}
    if room_count >= 2 and "open" not in classifications:
        return None
    if room_count >= 2 and not ({"wall", "ray"} & classifications):
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    if len(_find_rooms(grid)) != room_count:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid room-and-window puzzle")
