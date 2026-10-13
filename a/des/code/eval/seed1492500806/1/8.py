from __future__ import annotations

from collections import Counter, deque

import numpy as np


BG = 0
BORDER = 5
DOOR_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

GRID_MAX = 30
ROOM_ROWS_RANGE = (2, 4)
ROOM_COLS_RANGE = (2, 4)
MARGIN_RANGE = (1, 3)
ROOM_OUTER_MIN = 5
ROOM_OUTER_MAX = 8
GAP = 2
GEN_RETRIES = 300


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
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
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bg_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[cnts.argmax()])


def _border_color(grid: np.ndarray, bg: int) -> int:
    vals = grid[grid != bg]
    assert vals.size > 0, "expected non-background room borders"
    colors, counts = np.unique(vals, return_counts=True)
    return int(colors[counts.argmax()])


def _find_rooms(grid: np.ndarray, bg: int) -> list[dict]:
    rooms: list[dict] = []
    for coords in _components(grid != bg):
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        window = grid[r0 : r1 + 1, c0 : c1 + 1]
        hh, ww = window.shape
        if hh < 3 or ww < 3:
            continue
        rim = np.zeros_like(window, dtype=bool)
        rim[0, :] = True
        rim[-1, :] = True
        rim[:, 0] = True
        rim[:, -1] = True
        if not np.all(window[rim] != bg):
            continue
        if not np.all(window[~rim] == bg):
            continue
        if int(coords.shape[0]) != int(rim.sum()):
            continue
        rooms.append({"bbox": (r0, c0, r1, c1)})
    assert rooms, "expected at least one framed room"
    rooms.sort(key=lambda room: (room["bbox"][0], room["bbox"][1]))
    return rooms


def _room_doors(grid: np.ndarray, room: dict, border: int, bg: int) -> dict[str, dict[int, int]]:
    r0, c0, r1, c1 = room["bbox"]
    doors = {"top": {}, "bottom": {}, "left": {}, "right": {}}
    for c in range(c0 + 1, c1):
        v = int(grid[r0, c])
        if v not in (bg, border):
            doors["top"][int(c)] = v
        v = int(grid[r1, c])
        if v not in (bg, border):
            doors["bottom"][int(c)] = v
    for r in range(r0 + 1, r1):
        v = int(grid[r, c0])
        if v not in (bg, border):
            doors["left"][int(r)] = v
        v = int(grid[r, c1])
        if v not in (bg, border):
            doors["right"][int(r)] = v
    return doors


def _corridor_color(a: int, b: int) -> int:
    # ARC grids do not expose human-readable color names.
    # Generated examples therefore only connect equal-colored aligned door pairs.
    assert int(a) == int(b), "aligned door colors must match in generated instances"
    return int(a)


def main(grid: np.ndarray) -> np.ndarray:
    """
    根据房间边框上的彩色单像素门，在相邻房间的对齐门之间铺设走廊。

    说明：
    - 房间边框是单像素厚的矩形框，背景为 Black。
    - 为避免使用 ARC 中不可见的“颜色名称字母序”语义，生成器只产生同色对齐门；
      因此走廊颜色就是该对齐门的公共颜色。
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _bg_color(g)
    border = _border_color(g, bg)
    rooms = _find_rooms(g, bg)
    for room in rooms:
        room["doors"] = _room_doors(g, room, border, bg)

    out = g.copy()

    by_row: dict[int, list[dict]] = {}
    by_col: dict[int, list[dict]] = {}
    for room in rooms:
        r0, c0, _r1, _c1 = room["bbox"]
        by_row.setdefault(int(r0), []).append(room)
        by_col.setdefault(int(c0), []).append(room)

    for row_rooms in by_row.values():
        row_rooms.sort(key=lambda room: room["bbox"][1])
        for left_room, right_room in zip(row_rooms, row_rooms[1:]):
            lr0, lc0, lr1, lc1 = left_room["bbox"]
            rr0, rc0, rr1, rc1 = right_room["bbox"]
            if lr0 != rr0 or lr1 != rr1 or rc0 <= lc1:
                continue
            common_rows = sorted(
                set(left_room["doors"]["right"]).intersection(right_room["doors"]["left"])
            )
            for row in common_rows:
                color = _corridor_color(
                    left_room["doors"]["right"][row], right_room["doors"]["left"][row]
                )
                out[row, lc1 + 1 : rc0] = int(color)

    for col_rooms in by_col.values():
        col_rooms.sort(key=lambda room: room["bbox"][0])
        for top_room, bottom_room in zip(col_rooms, col_rooms[1:]):
            tr0, tc0, tr1, tc1 = top_room["bbox"]
            br0, bc0, br1, bc1 = bottom_room["bbox"]
            if tc0 != bc0 or tc1 != bc1 or br0 <= tr1:
                continue
            common_cols = sorted(
                set(top_room["doors"]["bottom"]).intersection(bottom_room["doors"]["top"])
            )
            for col in common_cols:
                color = _corridor_color(
                    top_room["doors"]["bottom"][col], bottom_room["doors"]["top"][col]
                )
                out[tr1 + 1 : br0, col] = int(color)

    return out


def _sample_layout(
    rng: np.random.Generator,
) -> tuple[int, int, int, int, int, int, int, int]:
    for _ in range(200):
        room_rows = int(rng.integers(ROOM_ROWS_RANGE[0], ROOM_ROWS_RANGE[1] + 1))
        room_cols = int(rng.integers(ROOM_COLS_RANGE[0], ROOM_COLS_RANGE[1] + 1))
        pad_top, pad_bottom, pad_left, pad_right = map(
            int, rng.integers(MARGIN_RANGE[0], MARGIN_RANGE[1] + 1, size=4)
        )
        max_outer_h = min(
            ROOM_OUTER_MAX,
            (GRID_MAX - pad_top - pad_bottom - GAP * (room_rows - 1)) // room_rows,
        )
        max_outer_w = min(
            ROOM_OUTER_MAX,
            (GRID_MAX - pad_left - pad_right - GAP * (room_cols - 1)) // room_cols,
        )
        if max_outer_h < ROOM_OUTER_MIN or max_outer_w < ROOM_OUTER_MIN:
            continue
        room_h = int(rng.integers(ROOM_OUTER_MIN, max_outer_h + 1))
        room_w = int(rng.integers(ROOM_OUTER_MIN, max_outer_w + 1))
        height = pad_top + pad_bottom + room_rows * room_h + GAP * (room_rows - 1)
        width = pad_left + pad_right + room_cols * room_w + GAP * (room_cols - 1)
        return room_rows, room_cols, room_h, room_w, pad_top, pad_left, height, width
    raise RuntimeError("failed to sample a valid room layout")


def _room_bbox(
    room_r: int,
    room_c: int,
    room_h: int,
    room_w: int,
    pad_top: int,
    pad_left: int,
) -> tuple[int, int, int, int]:
    r0 = pad_top + room_r * (room_h + GAP)
    c0 = pad_left + room_c * (room_w + GAP)
    return int(r0), int(c0), int(r0 + room_h - 1), int(c0 + room_w - 1)


def _door_cell(
    bbox: tuple[int, int, int, int], side: str, offset: int
) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    if side == "top":
        return int(r0), int(c0 + offset)
    if side == "bottom":
        return int(r1), int(c0 + offset)
    if side == "left":
        return int(r0 + offset), int(c0)
    assert side == "right"
    return int(r0 + offset), int(c1)


def _add_door(
    doors: dict[tuple[int, int], dict[str, dict[int, int]]],
    room_key: tuple[int, int],
    side: str,
    offset: int,
    color: int,
) -> None:
    side_map = doors[room_key][side]
    assert int(offset) not in side_map, "door offsets must stay unique per side"
    side_map[int(offset)] = int(color)


def _choice_without_replacement(
    rng: np.random.Generator, seq: list[tuple[int, int]], count: int
) -> list[tuple[int, int]]:
    idx = rng.permutation(len(seq))[:count]
    return [seq[int(i)] for i in idx]


def _make_expected(
    grid: np.ndarray,
    connections: list[tuple[str, tuple[int, int], int, int]],
    bboxes: dict[tuple[int, int], tuple[int, int, int, int]],
) -> np.ndarray:
    out = grid.copy()
    for axis, room_key, offset, color in connections:
        rr, cc = room_key
        bbox = bboxes[(rr, cc)]
        if axis == "h":
            left_bbox = bbox
            right_bbox = bboxes[(rr, cc + 1)]
            row = left_bbox[0] + offset
            out[row, left_bbox[3] + 1 : right_bbox[1]] = int(color)
        else:
            top_bbox = bbox
            bottom_bbox = bboxes[(rr + 1, cc)]
            col = top_bbox[1] + offset
            out[top_bbox[2] + 1 : bottom_bbox[0], col] = int(color)
    return out


def _attempt_generate(rng: np.random.Generator) -> np.ndarray | None:
    room_rows, room_cols, room_h, room_w, pad_top, pad_left, height, width = _sample_layout(rng)

    bboxes = {
        (rr, cc): _room_bbox(rr, cc, room_h, room_w, pad_top, pad_left)
        for rr in range(room_rows)
        for cc in range(room_cols)
    }
    doors = {
        (rr, cc): {"top": {}, "bottom": {}, "left": {}, "right": {}}
        for rr in range(room_rows)
        for cc in range(room_cols)
    }

    horizontal_pairs = [(rr, cc) for rr in range(room_rows) for cc in range(room_cols - 1)]
    vertical_pairs = [(rr, cc) for rr in range(room_rows - 1) for cc in range(room_cols)]
    mode = int(rng.integers(0, 3))  # 0: horizontal, 1: vertical, 2: both

    want_h = mode in (0, 2)
    want_v = mode in (1, 2)
    chosen_h: list[tuple[int, int]] = []
    chosen_v: list[tuple[int, int]] = []

    if want_h:
        count_h = int(rng.integers(1, min(3, len(horizontal_pairs)) + 1))
        chosen_h = _choice_without_replacement(rng, horizontal_pairs, count_h)
    if want_v:
        count_v = int(rng.integers(1, min(3, len(vertical_pairs)) + 1))
        chosen_v = _choice_without_replacement(rng, vertical_pairs, count_v)

    connections: list[tuple[str, tuple[int, int], int, int]] = []

    for rr, cc in chosen_h:
        offset = int(rng.integers(1, room_h - 1))
        color = int(rng.choice(DOOR_COLORS))
        _add_door(doors, (rr, cc), "right", offset, color)
        _add_door(doors, (rr, cc + 1), "left", offset, color)
        connections.append(("h", (rr, cc), offset, color))

    for rr, cc in chosen_v:
        offset = int(rng.integers(1, room_w - 1))
        color = int(rng.choice(DOOR_COLORS))
        _add_door(doors, (rr, cc), "bottom", offset, color)
        _add_door(doors, (rr + 1, cc), "top", offset, color)
        connections.append(("v", (rr, cc), offset, color))

    if not connections:
        return None

    internal_candidates: list[tuple[tuple[int, int], str, int]] = []
    for rr, cc in horizontal_pairs:
        used_left = set(doors[(rr, cc)]["right"])
        used_right = set(doors[(rr, cc + 1)]["left"])
        available = [off for off in range(1, room_h - 1) if off not in used_left and off not in used_right]
        if available:
            offset = int(rng.choice(available))
            room_key = (rr, cc) if rng.random() < 0.5 else (rr, cc + 1)
            side = "right" if room_key == (rr, cc) else "left"
            internal_candidates.append((room_key, side, offset))
    for rr, cc in vertical_pairs:
        used_top = set(doors[(rr, cc)]["bottom"])
        used_bottom = set(doors[(rr + 1, cc)]["top"])
        available = [off for off in range(1, room_w - 1) if off not in used_top and off not in used_bottom]
        if available:
            offset = int(rng.choice(available))
            room_key = (rr, cc) if rng.random() < 0.5 else (rr + 1, cc)
            side = "bottom" if room_key == (rr, cc) else "top"
            internal_candidates.append((room_key, side, offset))
    if internal_candidates:
        room_key, side, offset = internal_candidates[int(rng.integers(len(internal_candidates)))]
        color = int(rng.choice(DOOR_COLORS))
        _add_door(doors, room_key, side, offset, color)

    boundary_added = False
    for rr in range(room_rows):
        for cc in range(room_cols):
            room_key = (rr, cc)
            boundary_sides: list[str] = []
            if rr == 0:
                boundary_sides.append("top")
            if rr == room_rows - 1:
                boundary_sides.append("bottom")
            if cc == 0:
                boundary_sides.append("left")
            if cc == room_cols - 1:
                boundary_sides.append("right")
            for side in boundary_sides:
                if rng.random() >= 0.35:
                    continue
                side_len = room_w if side in ("top", "bottom") else room_h
                available = [off for off in range(1, side_len - 1) if off not in doors[room_key][side]]
                if not available:
                    continue
                offset = int(rng.choice(available))
                color = int(rng.choice(DOOR_COLORS))
                _add_door(doors, room_key, side, offset, color)
                boundary_added = True

    if not boundary_added:
        boundary_options: list[tuple[tuple[int, int], str, list[int]]] = []
        for rr in range(room_rows):
            for cc in range(room_cols):
                room_key = (rr, cc)
                if rr == 0:
                    available = [off for off in range(1, room_w - 1) if off not in doors[room_key]["top"]]
                    if available:
                        boundary_options.append((room_key, "top", available))
                if rr == room_rows - 1:
                    available = [
                        off for off in range(1, room_w - 1) if off not in doors[room_key]["bottom"]
                    ]
                    if available:
                        boundary_options.append((room_key, "bottom", available))
                if cc == 0:
                    available = [off for off in range(1, room_h - 1) if off not in doors[room_key]["left"]]
                    if available:
                        boundary_options.append((room_key, "left", available))
                if cc == room_cols - 1:
                    available = [off for off in range(1, room_h - 1) if off not in doors[room_key]["right"]]
                    if available:
                        boundary_options.append((room_key, "right", available))
        if boundary_options:
            room_key, side, available = boundary_options[int(rng.integers(len(boundary_options)))]
            offset = int(rng.choice(available))
            color = int(rng.choice(DOOR_COLORS))
            _add_door(doors, room_key, side, offset, color)

    grid = np.full((height, width), BG, dtype=int)
    for bbox in bboxes.values():
        r0, c0, r1, c1 = bbox
        grid[r0, c0 : c1 + 1] = BORDER
        grid[r1, c0 : c1 + 1] = BORDER
        grid[r0 : r1 + 1, c0] = BORDER
        grid[r0 : r1 + 1, c1] = BORDER

    for room_key, room_doors in doors.items():
        bbox = bboxes[room_key]
        for side, side_doors in room_doors.items():
            for offset, color in side_doors.items():
                r, c = _door_cell(bbox, side, offset)
                grid[r, c] = int(color)

    expected = _make_expected(grid, connections, bboxes)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    border = _border_color(grid, BG)
    if border != BORDER:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
