from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np


YELLOW = 4
GREY = 5
BG_CHOICES = [0, 1, 2, 3, 6, 7, 8, 9]
GEN_RETRIES = 400
SIDES = ("top", "bottom", "left", "right")


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _most_common_non_bg(grid: np.ndarray, bg: int) -> int | None:
    values = grid[grid != bg]
    if values.size == 0:
        return None
    cols, counts = np.unique(values, return_counts=True)
    return int(cols[np.argmax(counts)])


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = map(int, mask.shape)
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    pts = np.array(list(coords), dtype=int)
    return (
        int(pts[:, 0].min()),
        int(pts[:, 1].min()),
        int(pts[:, 0].max()),
        int(pts[:, 1].max()),
    )


def _touches_edge(coords: Iterable[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or r == h - 1 or c == 0 or c == w - 1 for r, c in coords)


def _room_border_closed(
    grid: np.ndarray,
    bg: int,
    wall: int,
    room_bbox: tuple[int, int, int, int],
) -> bool:
    h, w = map(int, grid.shape)
    r0, c0, r1, c1 = room_bbox
    if r0 <= 0 or c0 <= 0 or r1 >= h - 1 or c1 >= w - 1:
        return False

    for c in range(c0 - 1, c1 + 2):
        if int(grid[r0 - 1, c]) == bg or int(grid[r1 + 1, c]) == bg:
            return False
    for r in range(r0, r1 + 1):
        if int(grid[r, c0 - 1]) == bg or int(grid[r, c1 + 1]) == bg:
            return False

    corners = ((r0 - 1, c0 - 1), (r0 - 1, c1 + 1), (r1 + 1, c0 - 1), (r1 + 1, c1 + 1))
    return all(int(grid[r, c]) == wall for r, c in corners)


def _collect_windows(
    grid: np.ndarray,
    bg: int,
    wall: int,
    room_bbox: tuple[int, int, int, int],
) -> list[tuple[str, int, int, int]]:
    r0, c0, r1, c1 = room_bbox
    windows: list[tuple[str, int, int, int]] = []

    for c in range(c0, c1 + 1):
        value = int(grid[r0 - 1, c])
        if value not in (bg, wall):
            windows.append(("top", r0 - 1, c, value))
        value = int(grid[r1 + 1, c])
        if value not in (bg, wall):
            windows.append(("bottom", r1 + 1, c, value))

    for r in range(r0, r1 + 1):
        value = int(grid[r, c0 - 1])
        if value not in (bg, wall):
            windows.append(("left", r, c0 - 1, value))
        value = int(grid[r, c1 + 1])
        if value not in (bg, wall):
            windows.append(("right", r, c1 + 1, value))

    return windows


def _cone_cells(
    room_bbox: tuple[int, int, int, int],
    side: str,
    win_r: int,
    win_c: int,
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = room_bbox
    cells: list[tuple[int, int]] = []

    if side == "top":
        depth = min(r1 - r0, win_c - c0, c1 - win_c)
        for d in range(depth + 1):
            rr = r0 + d
            for cc in range(win_c - d, win_c + d + 1):
                cells.append((rr, cc))
    elif side == "bottom":
        depth = min(r1 - r0, win_c - c0, c1 - win_c)
        for d in range(depth + 1):
            rr = r1 - d
            for cc in range(win_c - d, win_c + d + 1):
                cells.append((rr, cc))
    elif side == "left":
        depth = min(c1 - c0, win_r - r0, r1 - win_r)
        for d in range(depth + 1):
            cc = c0 + d
            for rr in range(win_r - d, win_r + d + 1):
                cells.append((rr, cc))
    else:
        depth = min(c1 - c0, win_r - r0, r1 - win_r)
        for d in range(depth + 1):
            cc = c1 - d
            for rr in range(win_r - d, win_r + d + 1):
                cells.append((rr, cc))

    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project colored triangular light cones from border windows into each room.

    Input structure:
    - bg is the most common color.
    - Grey 1-pixel frames enclose rectangular rooms whose interiors are bg.
    - Some border pixels on those frames are recolored as windows.

    Rule:
    - Top/bottom windows project vertical isosceles triangles inward.
    - Left/right windows project horizontal isosceles triangles inward.
    - If two or more cones cover the same interior pixel, that pixel becomes yellow.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    h, w = map(int, g.shape)
    bg = _most_common_color(g)
    wall = _most_common_non_bg(g, bg)
    if wall is None:
        return out

    for comp in _components_mask(g == bg):
        if _touches_edge(comp, h, w):
            continue

        room_bbox = _bbox(comp)
        r0, c0, r1, c1 = room_bbox
        if (r1 - r0 + 1) * (c1 - c0 + 1) != len(comp):
            continue
        if not _room_border_closed(g, bg, wall, room_bbox):
            continue

        windows = _collect_windows(g, bg, wall, room_bbox)
        if not windows:
            continue

        room_h = r1 - r0 + 1
        room_w = c1 - c0 + 1
        cover_count = np.zeros((room_h, room_w), dtype=int)
        fill = np.full((room_h, room_w), bg, dtype=int)

        for side, win_r, win_c, color in windows:
            for rr, cc in _cone_cells(room_bbox, side, win_r, win_c):
                ir = rr - r0
                ic = cc - c0
                if cover_count[ir, ic] == 0:
                    fill[ir, ic] = int(color)
                else:
                    fill[ir, ic] = YELLOW
                cover_count[ir, ic] += 1

        mask = cover_count > 0
        region = out[r0 : r1 + 1, c0 : c1 + 1]
        region[mask] = fill[mask]

    return out


def _safe_span(length: int) -> tuple[int, int]:
    if length >= 4:
        return 1, length - 2
    return 0, length - 1


def _pick_local(length: int, mode: str, rng: np.random.Generator) -> int:
    lo, hi = _safe_span(length)
    if mode == "near_start":
        return int(lo)
    if mode == "near_end":
        return int(hi)
    if mode == "center":
        target = (lo + hi) / 2.0
        candidates = [i for i in range(lo, hi + 1) if abs(i - target) == min(abs(j - target) for j in range(lo, hi + 1))]
        return int(rng.choice(candidates))
    return int(rng.integers(lo, hi + 1))


def _overlap_axis(room: dict[str, int], rng: np.random.Generator) -> str:
    height = int(room["height"])
    width = int(room["width"])
    can_vertical = (width - 1) // 2 >= (height - 1 + 1) // 2
    can_horizontal = (height - 1) // 2 >= (width - 1 + 1) // 2
    options = []
    if can_vertical:
        options.append("vertical")
    if can_horizontal:
        options.append("horizontal")
    if not options:
        options.append("vertical" if width >= height else "horizontal")
    return str(rng.choice(options))


def _fill_extra_sides(existing: list[str], target_count: int, rng: np.random.Generator) -> list[str]:
    sides = list(existing)
    remaining = [side for side in SIDES if side not in sides]
    rng.shuffle(remaining)
    while len(sides) < target_count and remaining:
        sides.append(remaining.pop())
    return sides


def _assign_side_plans(
    rooms: list[dict[str, int]],
    rng: np.random.Generator,
) -> tuple[list[list[str]], dict[str, int | str | bool]]:
    n = len(rooms)
    order = list(map(int, rng.permutation(n)))
    empty_room, single_room, pair_room, rich_room = order[:4]

    counts = [int(rng.choice([0, 1, 2], p=[0.25, 0.45, 0.30])) for _ in range(n)]
    counts[empty_room] = 0
    counts[single_room] = 1
    counts[pair_room] = 2 + int(rng.random() < 0.35)
    counts[rich_room] = 2 + int(rng.random() < 0.50)

    plans: list[list[str]] = [[] for _ in range(n)]
    force_overlap = bool(rng.random() < 0.60)
    overlap_room = pair_room
    overlap_mode: str | None = None

    if force_overlap:
        overlap_mode = _overlap_axis(rooms[overlap_room], rng)
        if overlap_mode == "vertical":
            plans[overlap_room] = ["top", "bottom"]
            lateral = ["left", "right"]
            rng.shuffle(lateral)
            plans[single_room] = [lateral[0]]
            plans[rich_room] = [lateral[1]]
        else:
            plans[overlap_room] = ["left", "right"]
            vertical = ["top", "bottom"]
            rng.shuffle(vertical)
            plans[single_room] = [vertical[0]]
            plans[rich_room] = [vertical[1]]
    else:
        adjacent_pairs = [
            ("top", "left"),
            ("top", "right"),
            ("bottom", "left"),
            ("bottom", "right"),
        ]
        first, second = adjacent_pairs[int(rng.integers(len(adjacent_pairs)))]
        plans[overlap_room] = [first, second]
        remaining = [side for side in SIDES if side not in plans[overlap_room]]
        rng.shuffle(remaining)
        plans[single_room] = [remaining[0]]
        plans[rich_room] = [remaining[1]]

    plans[overlap_room] = _fill_extra_sides(plans[overlap_room], counts[overlap_room], rng)
    plans[rich_room] = _fill_extra_sides(plans[rich_room], counts[rich_room], rng)

    for idx in range(n):
        if idx in (empty_room, single_room, pair_room, rich_room):
            continue
        k = counts[idx]
        if k == 0:
            continue
        side_list = list(SIDES)
        rng.shuffle(side_list)
        plans[idx] = side_list[:k]

    meta = {
        "force_overlap": force_overlap,
        "overlap_room": overlap_room,
        "overlap_mode": overlap_mode if overlap_mode is not None else "",
        "empty_room": empty_room,
        "single_room": single_room,
        "rich_room": rich_room,
    }
    return plans, meta


def _window_positions_for_room(
    room: dict[str, int],
    sides: list[str],
    rng: np.random.Generator,
    overlap_mode: str | None,
) -> dict[str, int]:
    height = int(room["height"])
    width = int(room["width"])
    positions: dict[str, int] = {}

    if overlap_mode == "vertical" and {"top", "bottom"}.issubset(sides):
        local = _pick_local(width, "center", rng)
        positions["top"] = local
        positions["bottom"] = local
    if overlap_mode == "horizontal" and {"left", "right"}.issubset(sides):
        local = _pick_local(height, "center", rng)
        positions["left"] = local
        positions["right"] = local

    if "top" in sides and "bottom" in sides and overlap_mode != "vertical":
        if rng.random() < 0.5:
            positions.setdefault("top", _pick_local(width, "near_start", rng))
            positions.setdefault("bottom", _pick_local(width, "near_end", rng))
        else:
            positions.setdefault("top", _pick_local(width, "near_end", rng))
            positions.setdefault("bottom", _pick_local(width, "near_start", rng))

    if "left" in sides and "right" in sides and overlap_mode != "horizontal":
        if rng.random() < 0.5:
            positions.setdefault("left", _pick_local(height, "near_start", rng))
            positions.setdefault("right", _pick_local(height, "near_end", rng))
        else:
            positions.setdefault("left", _pick_local(height, "near_end", rng))
            positions.setdefault("right", _pick_local(height, "near_start", rng))

    for side in sides:
        if side in positions:
            continue
        length = width if side in ("top", "bottom") else height
        positions[side] = _pick_local(length, "random", rng)

    return positions


def _border_cell(room: dict[str, int], side: str, local: int) -> tuple[int, int]:
    r0 = int(room["r0"])
    c0 = int(room["c0"])
    r1 = int(room["r1"])
    c1 = int(room["c1"])
    if side == "top":
        return r0 - 1, c0 + local
    if side == "bottom":
        return r1 + 1, c0 + local
    if side == "left":
        return r0 + local, c0 - 1
    return r0 + local, c1 + 1


def _draw_room(grid: np.ndarray, room: dict[str, int], wall: int) -> None:
    r0 = int(room["r0"])
    c0 = int(room["c0"])
    r1 = int(room["r1"])
    c1 = int(room["c1"])
    grid[r0 - 1, c0 - 1 : c1 + 2] = wall
    grid[r1 + 1, c0 - 1 : c1 + 2] = wall
    grid[r0 : r1 + 1, c0 - 1] = wall
    grid[r0 : r1 + 1, c1 + 1] = wall


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    bg = int(rng.choice(BG_CHOICES))
    palette = [color for color in range(10) if color not in (bg, GREY, YELLOW)]
    if len(palette) < 3:
        return None

    n_rows = int(rng.integers(2, 4))
    n_cols = int(rng.integers(2, 4))
    row_heights = [int(rng.integers(4, 9)) for _ in range(n_rows)]
    col_widths = [int(rng.integers(4, 9)) for _ in range(n_cols)]
    row_gaps = [int(rng.integers(1, 4)) for _ in range(max(0, n_rows - 1))]
    col_gaps = [int(rng.integers(1, 4)) for _ in range(max(0, n_cols - 1))]
    margin_top = int(rng.integers(1, 4))
    margin_left = int(rng.integers(1, 4))
    margin_bottom = int(rng.integers(1, 4))
    margin_right = int(rng.integers(1, 4))

    height = margin_top + margin_bottom + sum(h + 2 for h in row_heights) + sum(row_gaps)
    width = margin_left + margin_right + sum(w + 2 for w in col_widths) + sum(col_gaps)
    grid = np.full((height, width), bg, dtype=int)

    rooms: list[dict[str, int]] = []
    top = margin_top
    for i, room_h in enumerate(row_heights):
        left = margin_left
        for j, room_w in enumerate(col_widths):
            room = {
                "r0": int(top + 1),
                "c0": int(left + 1),
                "r1": int(top + room_h),
                "c1": int(left + room_w),
                "height": int(room_h),
                "width": int(room_w),
            }
            rooms.append(room)
            _draw_room(grid, room, GREY)
            if j < n_cols - 1:
                left += room_w + 2 + col_gaps[j]
        if i < n_rows - 1:
            top += room_h + 2 + row_gaps[i]

    plans, meta = _assign_side_plans(rooms, rng)
    total_windows = sum(len(sides) for sides in plans)
    if total_windows < 4:
        return None

    for room_idx, sides in enumerate(plans):
        if not sides:
            continue

        overlap_mode = None
        if room_idx == int(meta["overlap_room"]) and bool(meta["force_overlap"]):
            overlap_mode = str(meta["overlap_mode"])

        positions = _window_positions_for_room(rooms[room_idx], sides, rng, overlap_mode)
        colors = rng.choice(palette, size=len(sides), replace=False)
        for side, color in zip(sides, colors):
            rr, cc = _border_cell(rooms[room_idx], side, positions[side])
            grid[rr, cc] = int(color)

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    if bool(meta["force_overlap"]) and not bool(np.any(out == YELLOW)):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
