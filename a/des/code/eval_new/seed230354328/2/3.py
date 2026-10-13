import random
from typing import Iterable

import numpy as np


_KEY_TEMPLATES = (
    (1, 1, 1),
    (1, 2),
    (2, 1),
    (2, 2),
    (1, 2, 1),
    (2, 1, 2),
    (1, 2, 2),
    (2, 2, 1),
)

_MAX_RETRIES = 300


def _wall_indices(grid: np.ndarray) -> tuple[list[int], list[int]]:
    h, w = grid.shape
    wall_rows = [r for r in range(h) if int(np.count_nonzero(grid[r])) > w // 2]
    wall_cols = [c for c in range(w) if int(np.count_nonzero(grid[:, c])) > h // 2]
    return wall_rows, wall_cols


def _room_spans(bounds: list[int], limit: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    extended = [-1, *bounds, limit]
    for i in range(len(extended) - 1):
        start = extended[i] + 1
        end = extended[i + 1] - 1
        if start <= end:
            spans.append((start, end))
    return spans


def _segment_color(values: Iterable[int]) -> int | None:
    colors = {int(v) for v in values if int(v) != 0}
    if len(colors) != 1:
        return None
    return next(iter(colors))


def _room_sides(
    grid: np.ndarray,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    wall_rows: list[int],
    wall_cols: list[int],
    room_r: int,
    room_c: int,
) -> list[dict]:
    r0, r1 = row_spans[room_r]
    c0, c1 = col_spans[room_c]
    sides: list[dict] = []

    if room_r > 0:
        wall_row = wall_rows[room_r - 1]
        color = _segment_color(grid[wall_row, c0 : c1 + 1])
        if color is not None:
            sides.append(
                {
                    "name": "top",
                    "orientation": "horizontal",
                    "color": color,
                    "fixed": wall_row,
                    "start": c0,
                    "end": c1,
                    "id": ("h", wall_row, c0, c1),
                }
            )
    if room_r < len(row_spans) - 1:
        wall_row = wall_rows[room_r]
        color = _segment_color(grid[wall_row, c0 : c1 + 1])
        if color is not None:
            sides.append(
                {
                    "name": "bottom",
                    "orientation": "horizontal",
                    "color": color,
                    "fixed": wall_row,
                    "start": c0,
                    "end": c1,
                    "id": ("h", wall_row, c0, c1),
                }
            )
    if room_c > 0:
        wall_col = wall_cols[room_c - 1]
        color = _segment_color(grid[r0 : r1 + 1, wall_col])
        if color is not None:
            sides.append(
                {
                    "name": "left",
                    "orientation": "vertical",
                    "color": color,
                    "fixed": wall_col,
                    "start": r0,
                    "end": r1,
                    "id": ("v", wall_col, r0, r1),
                }
            )
    if room_c < len(col_spans) - 1:
        wall_col = wall_cols[room_c]
        color = _segment_color(grid[r0 : r1 + 1, wall_col])
        if color is not None:
            sides.append(
                {
                    "name": "right",
                    "orientation": "vertical",
                    "color": color,
                    "fixed": wall_col,
                    "start": r0,
                    "end": r1,
                    "id": ("v", wall_col, r0, r1),
                }
            )
    return sides


def main(grid: np.ndarray) -> np.ndarray:
    """Open room-dividing wall segments when a room's key projects exactly onto that wall's lockhole.

    Input structure:
    - bg is black.
    - Single-pixel colored horizontal and vertical divider lines split the grid into rectangular rooms.
    - Some rooms contain one small key shape whose color matches one of that room's colored walls.
    - Some wall segments contain black missing-cell lockholes along the wall line itself.

    Rule:
    - Detect the rooms from the full-width/full-height divider rows and columns.
    - For each room that contains a key, use the key color to choose the corresponding wall.
    - Project the key onto that wall's axis; if the projected positions exactly equal the wall's black lockhole positions, erase the whole wall segment for that room and erase the key itself.
    - Rooms without a key, or keys whose projection does not match the wall lockhole, remain unchanged.

    Key insight:
    - The key's color identifies which wall to inspect, but the opening test uses only the key's one-dimensional projection along the wall, not the key's full 2D shape.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    wall_rows, wall_cols = _wall_indices(g)
    row_spans = _room_spans(wall_rows, h)
    col_spans = _room_spans(wall_cols, w)

    out = g.copy()
    keys_to_erase: set[tuple[int, int]] = set()
    walls_to_open: set[tuple] = set()

    for room_r, (r0, r1) in enumerate(row_spans):
        for room_c, (c0, c1) in enumerate(col_spans):
            room = g[r0 : r1 + 1, c0 : c1 + 1]
            room_colors = [int(v) for v in np.unique(room) if int(v) != 0]
            if not room_colors:
                continue
            if len(room_colors) != 1:
                continue

            key_color = room_colors[0]
            key_coords_local = np.argwhere(room == key_color)
            if key_coords_local.size == 0:
                continue
            key_coords = [(r0 + int(r), c0 + int(c)) for r, c in key_coords_local]

            sides = _room_sides(g, row_spans, col_spans, wall_rows, wall_cols, room_r, room_c)
            matches = [side for side in sides if int(side["color"]) == key_color]
            if len(matches) != 1:
                continue
            side = matches[0]

            if side["orientation"] == "horizontal":
                key_projection = tuple(sorted({c for _r, c in key_coords}))
                lockhole = tuple(
                    c
                    for c in range(int(side["start"]), int(side["end"]) + 1)
                    if int(g[int(side["fixed"]), c]) == 0
                )
            else:
                key_projection = tuple(sorted({r for r, _c in key_coords}))
                lockhole = tuple(
                    r
                    for r in range(int(side["start"]), int(side["end"]) + 1)
                    if int(g[r, int(side["fixed"])]) == 0
                )

            if key_projection != lockhole:
                continue

            keys_to_erase.update(key_coords)
            walls_to_open.add(tuple(side["id"]))

    for r, c in keys_to_erase:
        out[r, c] = 0

    for orientation, fixed, start, end in walls_to_open:
        if orientation == "h":
            out[int(fixed), int(start) : int(end) + 1] = 0
        else:
            out[int(start) : int(end) + 1, int(fixed)] = 0

    return out


def _random_sizes(count: int) -> list[int]:
    return [random.randint(4, 8) for _ in range(count)]


def _draw_partitions(
    heights: list[int], widths: list[int], h_colors: list[int], v_colors: list[int], vertical_on_top: bool
) -> tuple[np.ndarray, list[int], list[int]]:
    h = sum(heights) + len(h_colors)
    w = sum(widths) + len(v_colors)
    grid = np.zeros((h, w), dtype=int)

    wall_rows: list[int] = []
    wall_cols: list[int] = []

    pos = heights[0]
    for color in h_colors:
        wall_rows.append(pos)
        pos += 1
        pos += heights[len(wall_rows)]

    pos = widths[0]
    for color in v_colors:
        wall_cols.append(pos)
        pos += 1
        pos += widths[len(wall_cols)]

    if vertical_on_top:
        for row, color in zip(wall_rows, h_colors, strict=True):
            grid[row, :] = int(color)
        for col, color in zip(wall_cols, v_colors, strict=True):
            grid[:, col] = int(color)
    else:
        for col, color in zip(wall_cols, v_colors, strict=True):
            grid[:, col] = int(color)
        for row, color in zip(wall_rows, h_colors, strict=True):
            grid[row, :] = int(color)

    return grid, wall_rows, wall_cols


def _candidate_sides(
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    wall_rows: list[int],
    wall_cols: list[int],
    h_colors: list[int],
    v_colors: list[int],
) -> list[dict]:
    candidates: list[dict] = []
    for room_r, (r0, r1) in enumerate(row_spans):
        for room_c, (c0, c1) in enumerate(col_spans):
            room_id = (room_r, room_c)
            if room_r > 0:
                candidates.append(
                    {
                        "room_id": room_id,
                        "room_bounds": (r0, r1, c0, c1),
                        "side": "top",
                        "orientation": "horizontal",
                        "fixed": wall_rows[room_r - 1],
                        "start": c0,
                        "end": c1,
                        "color": h_colors[room_r - 1],
                        "segment_id": ("h", wall_rows[room_r - 1], c0, c1),
                    }
                )
            if room_r < len(row_spans) - 1:
                candidates.append(
                    {
                        "room_id": room_id,
                        "room_bounds": (r0, r1, c0, c1),
                        "side": "bottom",
                        "orientation": "horizontal",
                        "fixed": wall_rows[room_r],
                        "start": c0,
                        "end": c1,
                        "color": h_colors[room_r],
                        "segment_id": ("h", wall_rows[room_r], c0, c1),
                    }
                )
            if room_c > 0:
                candidates.append(
                    {
                        "room_id": room_id,
                        "room_bounds": (r0, r1, c0, c1),
                        "side": "left",
                        "orientation": "vertical",
                        "fixed": wall_cols[room_c - 1],
                        "start": r0,
                        "end": r1,
                        "color": v_colors[room_c - 1],
                        "segment_id": ("v", wall_cols[room_c - 1], r0, r1),
                    }
                )
            if room_c < len(col_spans) - 1:
                candidates.append(
                    {
                        "room_id": room_id,
                        "room_bounds": (r0, r1, c0, c1),
                        "side": "right",
                        "orientation": "vertical",
                        "fixed": wall_cols[room_c],
                        "start": r0,
                        "end": r1,
                        "color": v_colors[room_c],
                        "segment_id": ("v", wall_cols[room_c], r0, r1),
                    }
                )
    return candidates


def _choose_projection_start(segment_len: int, proj_len: int) -> int:
    slack = segment_len - proj_len
    if slack >= 2:
        return random.randint(1, slack - 1)
    return random.randint(0, slack)


def _pick_holes(segment_len: int, key_projection: tuple[int, ...]) -> tuple[int, ...] | None:
    options: list[tuple[int, ...]] = []
    max_len = min(3, segment_len - 1 if segment_len > 1 else 1)
    for length in range(1, max_len + 1):
        for offset in range(segment_len - length + 1):
            cand = tuple(range(offset, offset + length))
            if cand != key_projection:
                options.append(cand)
    if not options:
        return None
    return random.choice(options)


def _place_key(grid: np.ndarray, candidate: dict) -> tuple[tuple[int, ...], list[tuple[int, int]]]:
    r0, r1, c0, c1 = candidate["room_bounds"]
    segment_len = int(candidate["end"]) - int(candidate["start"]) + 1
    depth_cap = (r1 - r0 + 1) if candidate["orientation"] == "horizontal" else (c1 - c0 + 1)
    templates = [tpl for tpl in _KEY_TEMPLATES if len(tpl) <= segment_len and max(tpl) <= depth_cap]
    depths = list(random.choice(templates))
    proj_len = len(depths)
    offset = _choose_projection_start(segment_len, proj_len)
    projection = tuple(range(offset, offset + proj_len))

    cells: list[tuple[int, int]] = []
    if candidate["orientation"] == "horizontal":
        wall_row = int(candidate["fixed"])
        row_step = 1 if candidate["side"] == "top" else -1
        for col, depth in zip(range(int(candidate["start"]) + offset, int(candidate["start"]) + offset + proj_len), depths, strict=True):
            for k in range(1, depth + 1):
                cells.append((wall_row + row_step * k, int(col)))
    else:
        wall_col = int(candidate["fixed"])
        col_step = 1 if candidate["side"] == "left" else -1
        for row, depth in zip(range(int(candidate["start"]) + offset, int(candidate["start"]) + offset + proj_len), depths, strict=True):
            for k in range(1, depth + 1):
                cells.append((int(row), wall_col + col_step * k))

    for r, c in cells:
        grid[int(r), int(c)] = int(candidate["color"])

    return projection, cells


def _carve_lockhole(grid: np.ndarray, candidate: dict, projection: tuple[int, ...], success: bool) -> bool:
    segment_len = int(candidate["end"]) - int(candidate["start"]) + 1
    relative = projection if success else _pick_holes(segment_len, projection)
    if relative is None:
        return False

    if candidate["orientation"] == "horizontal":
        row = int(candidate["fixed"])
        for offset in relative:
            grid[row, int(candidate["start"]) + int(offset)] = 0
    else:
        col = int(candidate["fixed"])
        for offset in relative:
            grid[int(candidate["start"]) + int(offset), col] = 0
    return True


def _gen_attempt() -> np.ndarray | None:
    n_h = random.randint(1, 2)
    n_v = random.randint(1, 2)
    heights = _random_sizes(n_h + 1)
    widths = _random_sizes(n_v + 1)

    h = sum(heights) + n_h
    w = sum(widths) + n_v
    if h > 30 or w > 30:
        return None

    colors = random.sample(range(1, 10), n_h + n_v)
    h_colors = colors[:n_h]
    v_colors = colors[n_h:]
    grid, wall_rows, wall_cols = _draw_partitions(heights, widths, h_colors, v_colors, vertical_on_top=bool(random.getrandbits(1)))
    row_spans = _room_spans(wall_rows, h)
    col_spans = _room_spans(wall_cols, w)
    rooms = [(ri, ci) for ri in range(len(row_spans)) for ci in range(len(col_spans))]

    if len(rooms) < 4:
        return None

    candidates = _candidate_sides(row_spans, col_spans, wall_rows, wall_cols, h_colors, v_colors)
    random.shuffle(candidates)
    max_keys = min(4, len(rooms) - 1)
    if max_keys < 2:
        return None
    target_count = random.randint(2, max_keys)

    selected: list[dict] = []
    used_rooms: set[tuple[int, int]] = set()
    used_segments: set[tuple] = set()
    for candidate in candidates:
        if candidate["room_id"] in used_rooms or tuple(candidate["segment_id"]) in used_segments:
            continue
        selected.append(candidate)
        used_rooms.add(tuple(candidate["room_id"]))
        used_segments.add(tuple(candidate["segment_id"]))
        if len(selected) == target_count:
            break
    if len(selected) < 2:
        return None

    success_count = len(selected) // 2
    if len(selected) % 2 == 1 and random.random() < 0.5:
        success_count += 1
    success_flags = [True] * success_count + [False] * (len(selected) - success_count)
    random.shuffle(success_flags)

    for candidate, success in zip(selected, success_flags, strict=True):
        projection, cells = _place_key(grid, candidate)
        if not cells:
            return None
        if not _carve_lockhole(grid, candidate, projection, success):
            return None

    out = main(grid)
    if np.array_equal(grid, out):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {_MAX_RETRIES} attempts")
