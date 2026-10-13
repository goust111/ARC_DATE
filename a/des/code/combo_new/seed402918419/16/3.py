import random

import numpy as np


BLACK = 0
MARKER = 4
LEGEND_BG = 5
LEGEND_COUNT = 7
ROOM_COLOR_POOL = [1, 2, 3, 6, 8, 9]
GEN_ATTEMPTS = 400
PLACE_ATTEMPTS = 200


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _border_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0 + 1, r1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _find_rooms(grid: np.ndarray, bg: int) -> tuple[list[dict], dict[int, set[tuple[int, int]]]]:
    rooms: list[dict] = []
    room_pixels_by_color: dict[int, set[tuple[int, int]]] = {}
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        for comp in _components(grid == color):
            if len(comp) < 7:
                continue
            r0, c0, r1, c1 = _bbox(comp)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            comp_set = set(comp)
            border = _border_cells(r0, c0, r1, c1)
            if not comp_set.issubset(border):
                continue
            missing = list(border - comp_set)
            if len(missing) != 1:
                continue
            rooms.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "door": missing[0],
                    "pixels": comp_set,
                }
            )
            room_pixels_by_color.setdefault(color, set()).update(comp_set)
    return rooms, room_pixels_by_color


def _extract_thresholds(
    grid: np.ndarray,
    bg: int,
    rooms: list[dict],
    room_pixels_by_color: dict[int, set[tuple[int, int]]],
) -> dict[int, int]:
    _h, w = grid.shape
    thresholds: dict[int, int] = {}
    for color in sorted({int(room["color"]) for room in rooms}):
        extras = [
            (int(r), int(c))
            for r, c in np.argwhere(grid == color)
            if (int(r), int(c)) not in room_pixels_by_color.get(color, set())
        ]
        best = None
        for r, c in extras:
            if c + 2 >= w:
                continue
            spacer = int(grid[r, c + 1])
            run_color = int(grid[r, c + 2])
            if spacer in (bg, color) or run_color in (bg, color, spacer):
                continue
            length = 0
            j = c + 2
            while j < w and int(grid[r, j]) == run_color:
                length += 1
                j += 1
            if length > 0 and (best is None or length > best):
                best = length
        assert best is not None, f"missing legend row for room color {color}"
        thresholds[color] = best
    return thresholds


def main(grid: np.ndarray) -> np.ndarray:
    """
    Close each room's 1-pixel door when its interior marker count reaches the legend threshold.

    Input structure:
    - The background is black.
    - Several colored 1-pixel-thick rectangular room borders appear on the grid, and each border has exactly one missing border pixel that acts as a door.
    - Each room interior contains scattered single-pixel markers, while a separate small legend rectangle contains one color key pixel per room color and a rightward run of count pixels encoding that color's threshold.

    Rule:
    - Detect every rectangular room border and count the non-background marker pixels inside that room.
    - Read the legend row whose key pixel has the same color as the room border, and use the length of the count run on that row as the room's threshold.
    - If the room's interior marker count is at least the threshold, fill the room's missing border pixel with the room border color; otherwise leave that door open.

    Key insight:
    - The output depends on matching each room color to its legend row, then comparing an interior count with the row's encoded threshold rather than analyzing rooms independently.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    rooms, room_pixels_by_color = _find_rooms(g, bg)
    thresholds = _extract_thresholds(g, bg, rooms, room_pixels_by_color)

    out = g.copy()
    for room in rooms:
        r0, c0, r1, c1 = room["bbox"]
        color = int(room["color"])
        inside = g[r0 + 1 : r1, c0 + 1 : c1]
        marker_count = int(np.count_nonzero(inside != bg))
        if marker_count >= thresholds[color]:
            dr, dc = room["door"]
            out[dr, dc] = color
    return out


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _place_rectangles(
    shape: tuple[int, int], sizes: list[tuple[int, int]]
) -> list[tuple[int, int, int, int]] | None:
    h, w = shape
    placed: list[tuple[int, int, int, int]] = []
    for rh, rw in sizes:
        if rh + 2 > h or rw + 2 > w:
            return None
        success = False
        for _ in range(PLACE_ATTEMPTS):
            r0 = random.randint(1, h - rh - 1)
            c0 = random.randint(1, w - rw - 1)
            rect = (r0, c0, r0 + rh - 1, c0 + rw - 1)
            padded = (r0 - 1, c0 - 1, r0 + rh, c0 + rw)
            if any(_rects_overlap(padded, other) for other in placed):
                continue
            placed.append(rect)
            success = True
            break
        if not success:
            return None
    return placed


def _choose_statuses(room_count: int) -> list[bool]:
    closed_count = random.choice([room_count // 2, (room_count + 1) // 2])
    closed_count = min(max(closed_count, 1), room_count - 1)
    statuses = [True] * closed_count + [False] * (room_count - closed_count)
    random.shuffle(statuses)
    return statuses


def _room_spec(status_closed: bool) -> dict:
    threshold = random.randint(1, 5)
    if status_closed:
        if random.random() < 0.5:
            marker_count = threshold
        else:
            marker_count = random.randint(threshold, 5)
    else:
        marker_count = random.randint(0, threshold - 1)

    while True:
        outer_h = random.randint(4, 9)
        outer_w = random.randint(4, 9)
        if (outer_h - 2) * (outer_w - 2) >= marker_count:
            break

    side = random.choice(["top", "bottom", "left", "right"])
    if side in {"top", "bottom"}:
        gap_offset = random.randint(1, outer_w - 2)
    else:
        gap_offset = random.randint(1, outer_h - 2)

    return {
        "threshold": threshold,
        "marker_count": marker_count,
        "closed": status_closed,
        "outer_h": outer_h,
        "outer_w": outer_w,
        "door_side": side,
        "gap_offset": gap_offset,
    }


def _legend_patch(rows: list[tuple[int, int]]) -> np.ndarray:
    top_pad = random.randint(1, 2)
    left_pad = random.randint(1, 2)
    bottom_pad = random.randint(1, 2)
    right_pad = random.randint(1, 3)
    row_step = random.choice([1, 2])
    height = top_pad + bottom_pad + 1 + (len(rows) - 1) * row_step
    width = left_pad + 2 + max(threshold for _, threshold in rows) + right_pad
    patch = np.full((height, width), LEGEND_BG, dtype=int)
    for idx, (color, threshold) in enumerate(rows):
        rr = top_pad + idx * row_step
        cc = left_pad
        patch[rr, cc] = color
        patch[rr, cc + 2 : cc + 2 + threshold] = LEGEND_COUNT
    return patch


def _door_cell(room: dict, r0: int, c0: int) -> tuple[int, int]:
    if room["door_side"] == "top":
        return r0, c0 + room["gap_offset"]
    if room["door_side"] == "bottom":
        return r0 + room["outer_h"] - 1, c0 + room["gap_offset"]
    if room["door_side"] == "left":
        return r0 + room["gap_offset"], c0
    return r0 + room["gap_offset"], c0 + room["outer_w"] - 1


def _draw_room(grid: np.ndarray, room: dict, color: int, r0: int, c0: int) -> tuple[int, int]:
    r1 = r0 + room["outer_h"] - 1
    c1 = c0 + room["outer_w"] - 1
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    door = _door_cell(room, r0, c0)
    grid[door] = BLACK

    interior = [
        (rr, cc)
        for rr in range(r0 + 1, r1)
        for cc in range(c0 + 1, c1)
    ]
    for rr, cc in random.sample(interior, room["marker_count"]):
        grid[rr, cc] = MARKER
    return door


def _expected_output(inp: np.ndarray, rooms: list[dict], doors: list[tuple[int, int]], colors: list[int]) -> np.ndarray:
    out = inp.copy()
    for room, door, color in zip(rooms, doors, colors, strict=False):
        if room["closed"]:
            out[door] = color
    return out


def _generate_attempt() -> np.ndarray | None:
    room_count = random.randint(2, 4)
    colors = random.sample(ROOM_COLOR_POOL, room_count)
    statuses = _choose_statuses(room_count)
    rooms = [_room_spec(status) for status in statuses]

    legend_rows = list(zip(colors, [room["threshold"] for room in rooms], strict=False))
    random.shuffle(legend_rows)
    legend = _legend_patch(legend_rows)

    grid_h = random.randint(max(18, legend.shape[0] + 4), 30)
    grid_w = random.randint(max(18, legend.shape[1] + 4), 30)
    sizes = [(legend.shape[0], legend.shape[1])] + [(room["outer_h"], room["outer_w"]) for room in rooms]
    order = sorted(range(len(sizes)), key=lambda idx: sizes[idx][0] * sizes[idx][1], reverse=True)
    ordered_sizes = [sizes[idx] for idx in order]
    ordered_rects = _place_rectangles((grid_h, grid_w), ordered_sizes)
    if ordered_rects is None:
        return None

    rects: list[tuple[int, int, int, int] | None] = [None] * len(sizes)
    for idx, rect in zip(order, ordered_rects, strict=False):
        rects[idx] = rect

    grid = np.full((grid_h, grid_w), BLACK, dtype=int)

    legend_r0, legend_c0, legend_r1, legend_c1 = rects[0]
    grid[legend_r0 : legend_r1 + 1, legend_c0 : legend_c1 + 1] = legend

    doors: list[tuple[int, int]] = []
    for idx, room in enumerate(rooms, start=1):
        r0, c0, _r1, _c1 = rects[idx]
        doors.append(_draw_room(grid, room, colors[idx - 1], r0, c0))

    expected = _expected_output(grid, rooms, doors, colors)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-threshold puzzle")
