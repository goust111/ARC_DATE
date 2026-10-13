from __future__ import annotations

import random

import numpy as np


BG = 0
WALL = 1
WINDOW_COLORS = (2, 3, 4, 5, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400


def _most_common_color(arr: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(arr, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for sr in range(h):
        for sc in range(w):
            if not mask[sr, sc] or seen[sr, sc]:
                continue
            stack = [(sr, sc)]
            seen[sr, sc] = True
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _extract_rooms(grid: np.ndarray, bg: int, wall: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    rooms: list[dict] = []

    for comp in _components4(g == bg):
        if np.any(comp[:, 0] == 0) or np.any(comp[:, 0] == h - 1) or np.any(comp[:, 1] == 0) or np.any(comp[:, 1] == w - 1):
            continue

        r0, c0, r1, c1 = _bbox(comp)
        windows: list[dict] = []
        side_specs = [
            ("top", [(r0 - 1, c) for c in range(c0, c1 + 1)]),
            ("bottom", [(r1 + 1, c) for c in range(c0, c1 + 1)]),
            ("left", [(r, c0 - 1) for r in range(r0, r1 + 1)]),
            ("right", [(r, c1 + 1) for r in range(r0, r1 + 1)]),
        ]
        for side, cells in side_specs:
            for wr, wc in cells:
                color = int(g[wr, wc])
                if color == wall:
                    continue
                if color == bg:
                    raise AssertionError("room boundary must stay closed")
                windows.append({"side": side, "pos": (wr, wc), "color": color})

        pillars: list[tuple[int, int]] = []
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if int(g[r, c]) == wall:
                    pillars.append((r, c))

        rooms.append(
            {
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
                "windows": windows,
                "pillars": pillars,
            }
        )

    rooms.sort(key=lambda room: (room["r0"], room["c0"]))
    return rooms


def _room_beams(room: dict) -> list[dict]:
    pillar_rows = sorted({r for r, _ in room["pillars"]})
    pillar_cols = sorted({c for _, c in room["pillars"]})
    beams: list[dict] = []

    for priority, window in enumerate(room["windows"]):
        side = window["side"]
        beam = dict(window)
        beam["priority"] = priority
        if side == "top":
            beam["limit"] = pillar_rows[0] if pillar_rows else room["r1"] + 1
        elif side == "bottom":
            beam["limit"] = pillar_rows[-1] if pillar_rows else room["r0"] - 1
        elif side == "left":
            beam["limit"] = pillar_cols[0] if pillar_cols else room["c1"] + 1
        else:
            beam["limit"] = pillar_cols[-1] if pillar_cols else room["c0"] - 1
        beams.append(beam)

    return beams


def _beam_distance(beam: dict, room: dict, r: int, c: int) -> int | None:
    side = beam["side"]
    if side == "top":
        if r >= beam["limit"]:
            return None
        return int(r - (room["r0"] - 1))
    if side == "bottom":
        if r <= beam["limit"]:
            return None
        return int((room["r1"] + 1) - r)
    if side == "left":
        if c >= beam["limit"]:
            return None
        return int(c - (room["c0"] - 1))
    if c <= beam["limit"]:
        return None
    return int((room["c1"] + 1) - c)


def _room_has_tie(room: dict, grid: np.ndarray, bg: int) -> bool:
    beams = _room_beams(room)
    if len(beams) < 2:
        return False

    for r in range(room["r0"], room["r1"] + 1):
        for c in range(room["c0"], room["c1"] + 1):
            if int(grid[r, c]) != bg:
                continue
            best = None
            count = 0
            for beam in beams:
                dist = _beam_distance(beam, room, r, c)
                if dist is None:
                    continue
                if best is None or dist < best:
                    best = dist
                    count = 1
                elif dist == best:
                    count += 1
            if count > 1:
                return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed room by projecting same-color light inward from its wall windows.

    Non-background pixels form a blue wall structure with colored one-cell windows
    on room walls. A window colors whole interior rows or columns while moving away
    from its wall, and any internal blue pillar stops that propagation. If multiple
    beams reach a cell, the closer window wins.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    non_bg = g[g != bg]
    if non_bg.size == 0:
        return g.copy()
    wall = _most_common_color(non_bg)

    out = g.copy()
    for room in _extract_rooms(g, bg, wall):
        beams = _room_beams(room)
        if not beams:
            continue
        for r in range(room["r0"], room["r1"] + 1):
            for c in range(room["c0"], room["c1"] + 1):
                if int(g[r, c]) != bg:
                    continue
                best_key = None
                best_color = None
                for beam in beams:
                    dist = _beam_distance(beam, room, r, c)
                    if dist is None:
                        continue
                    key = (dist, beam["priority"])
                    if best_key is None or key < best_key:
                        best_key = key
                        best_color = int(beam["color"])
                if best_color is not None:
                    out[r, c] = best_color
    return out


def _build_layout(row_heights: list[int], col_widths: list[int]) -> tuple[np.ndarray, list[dict]]:
    top_margin = random.randint(1, 2)
    bottom_margin = random.randint(1, 2)
    left_margin = random.randint(1, 2)
    right_margin = random.randint(1, 2)

    wall_rows = [top_margin]
    row_starts: list[int] = []
    cursor = top_margin + 1
    for height in row_heights:
        row_starts.append(cursor)
        cursor += height
        wall_rows.append(cursor)
        cursor += 1

    wall_cols = [left_margin]
    col_starts: list[int] = []
    cursor = left_margin + 1
    for width in col_widths:
        col_starts.append(cursor)
        cursor += width
        wall_cols.append(cursor)
        cursor += 1

    h = wall_rows[-1] + 1 + bottom_margin
    w = wall_cols[-1] + 1 + right_margin
    grid = np.full((h, w), BG, dtype=int)

    top_wall = wall_rows[0]
    bottom_wall = wall_rows[-1]
    left_wall = wall_cols[0]
    right_wall = wall_cols[-1]
    for wr in wall_rows:
        grid[wr, left_wall : right_wall + 1] = WALL
    for wc in wall_cols:
        grid[top_wall : bottom_wall + 1, wc] = WALL

    rooms: list[dict] = []
    for i, row_start in enumerate(row_starts):
        for j, col_start in enumerate(col_starts):
            rooms.append(
                {
                    "grid_row": i,
                    "grid_col": j,
                    "r0": row_start,
                    "r1": row_start + row_heights[i] - 1,
                    "c0": col_start,
                    "c1": col_start + col_widths[j] - 1,
                    "rows": len(row_heights),
                    "cols": len(col_widths),
                }
            )
    return grid, rooms


def _outer_sides(room: dict) -> list[str]:
    sides: list[str] = []
    if room["grid_row"] == 0:
        sides.append("top")
    if room["grid_row"] == room["rows"] - 1:
        sides.append("bottom")
    if room["grid_col"] == 0:
        sides.append("left")
    if room["grid_col"] == room["cols"] - 1:
        sides.append("right")
    return sides


def _window_cell(room: dict, side: str) -> tuple[int, int]:
    if side == "top":
        return room["r0"] - 1, random.randint(room["c0"], room["c1"])
    if side == "bottom":
        return room["r1"] + 1, random.randint(room["c0"], room["c1"])
    if side == "left":
        return random.randint(room["r0"], room["r1"]), room["c0"] - 1
    return random.randint(room["r0"], room["r1"]), room["c1"] + 1


def _pick_window_sides(rooms: list[dict], rows: int, cols: int) -> list[list[str]]:
    pair_indices: set[int] = set()
    if rows == 1:
        for idx in range(len(rooms)):
            if random.random() < 0.5:
                pair_indices.add(idx)
        if not pair_indices:
            pair_indices.add(random.randrange(len(rooms)))
    elif cols == 1:
        for idx in range(len(rooms)):
            if random.random() < 0.5:
                pair_indices.add(idx)
        if not pair_indices:
            pair_indices.add(random.randrange(len(rooms)))

    side_lists: list[list[str]] = []
    for idx, room in enumerate(rooms):
        available = _outer_sides(room)
        if idx in pair_indices and rows == 1:
            side_lists.append(["top", "bottom"])
        elif idx in pair_indices and cols == 1:
            side_lists.append(["left", "right"])
        else:
            side_lists.append([random.choice(available)])
    return side_lists


def _allowed_pillar_cells(room: dict, sides: list[str]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for r in range(room["r0"], room["r1"] + 1):
        for c in range(room["c0"], room["c1"] + 1):
            if "top" in sides and r == room["r0"]:
                continue
            if "bottom" in sides and r == room["r1"]:
                continue
            if "left" in sides and c == room["c0"]:
                continue
            if "right" in sides and c == room["c1"]:
                continue
            cells.append((r, c))
    return cells


def _decorate_rooms(grid: np.ndarray, rooms: list[dict], side_lists: list[list[str]]) -> None:
    for room, sides in zip(rooms, side_lists):
        colors = random.sample(WINDOW_COLORS, len(sides))
        for side, color in zip(sides, colors):
            wr, wc = _window_cell(room, side)
            grid[wr, wc] = color

        candidates = _allowed_pillar_cells(room, sides)
        if len(sides) == 2:
            pillar_count = 0 if random.random() < 0.5 else 1
        else:
            pillar_count = random.choices([0, 1, 2], weights=[3, 4, 2])[0]
        pillar_count = min(pillar_count, len(candidates))
        if pillar_count > 0:
            for r, c in random.sample(candidates, pillar_count):
                grid[r, c] = WALL


def _gen_attempt() -> np.ndarray | None:
    family = random.choice(("h_strip", "v_strip", "h_table", "v_table"))
    if family == "h_strip":
        row_heights = [random.choice((4, 6))]
        col_widths = [random.randint(3, 5) for _ in range(random.randint(2, 4))]
    elif family == "v_strip":
        row_heights = [random.randint(3, 5) for _ in range(random.randint(2, 4))]
        col_widths = [random.choice((4, 6))]
    elif family == "h_table":
        row_heights = [random.randint(3, 5) for _ in range(2)]
        col_widths = [random.randint(3, 5) for _ in range(random.randint(2, 4))]
    else:
        row_heights = [random.randint(3, 5) for _ in range(random.randint(2, 4))]
        col_widths = [random.randint(3, 5) for _ in range(2)]

    grid, rooms = _build_layout(row_heights, col_widths)
    if grid.shape[0] > 28 or grid.shape[1] > 28:
        return None

    side_lists = _pick_window_sides(rooms, len(row_heights), len(col_widths))
    _decorate_rooms(grid, rooms, side_lists)

    parsed_rooms = _extract_rooms(grid, BG, WALL)
    if len(parsed_rooms) != len(rooms):
        return None
    if any(not (1 <= len(room["windows"]) <= 2) for room in parsed_rooms):
        return None
    if any(_room_has_tie(room, grid, BG) for room in parsed_rooms):
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
