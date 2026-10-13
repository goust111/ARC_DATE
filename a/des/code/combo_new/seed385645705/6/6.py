from __future__ import annotations

import random

import numpy as np


BG = 0
SIDES = ("top", "bottom", "left", "right")
SIDE_DIRS = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
ADJACENT_PAIRS = (
    ("top", "left"),
    ("top", "right"),
    ("bottom", "left"),
    ("bottom", "right"),
)
MAIN_DIAGONAL_PAIRS = {
    frozenset(("top", "left")),
    frozenset(("bottom", "right")),
}

GRID_MIN = 15
GRID_MAX = 30
ROOM_COUNT_CHOICES = (1, 2, 3)
ROOM_SIZE_MIN = 5
ROOM_SIZE_MAX = 10
ROOM_PLACEMENT_TRIES = 200
GENERATION_TRIES = 500
PALETTE = tuple(range(1, 10))


def _background_color(grid: np.ndarray) -> int:
    return BG if np.any(grid == BG) else int(np.bincount(grid.ravel()).argmax())


def _perimeter_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def _window_side(cell: tuple[int, int], bbox: tuple[int, int, int, int]) -> str:
    r, c = cell
    r0, c0, r1, c1 = bbox
    if r == r0 and c0 < c < c1:
        return "top"
    if r == r1 and c0 < c < c1:
        return "bottom"
    if c == c0 and r0 < r < r1:
        return "left"
    if c == c1 and r0 < r < r1:
        return "right"
    raise AssertionError("window must be a non-corner perimeter gap")


def _sample_point(
    bbox: tuple[int, int, int, int],
    side: str,
    cell: tuple[int, int],
) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    hi = r1 - r0 - 1
    wi = c1 - c0 - 1
    r, c = cell
    if side == "top":
        return 0, c - c0 - 1
    if side == "bottom":
        return hi - 1, c - c0 - 1
    if side == "left":
        return r - r0 - 1, 0
    return r - r0 - 1, wi - 1


def _split_scores(hi: int, wi: int, mode: str) -> np.ndarray:
    rr = np.arange(hi, dtype=int)[:, None]
    cc = np.arange(wi, dtype=int)[None, :]
    if mode == "horizontal":
        return 2 * rr + 1 - hi
    if mode == "vertical":
        return 2 * cc + 1 - wi
    if mode == "main":
        return (2 * rr + 1) * wi - (2 * cc + 1) * hi
    if mode == "anti":
        return (2 * rr + 1) * wi + (2 * cc + 1) * hi - 2 * hi * wi
    raise ValueError(mode)


def _score_at(rr: int, cc: int, hi: int, wi: int, mode: str) -> int:
    if mode == "horizontal":
        return 2 * rr + 1 - hi
    if mode == "vertical":
        return 2 * cc + 1 - wi
    if mode == "main":
        return (2 * rr + 1) * wi - (2 * cc + 1) * hi
    if mode == "anti":
        return (2 * rr + 1) * wi + (2 * cc + 1) * hi - 2 * hi * wi
    raise ValueError(mode)


def _paint_room(
    out: np.ndarray,
    original: np.ndarray,
    bbox: tuple[int, int, int, int],
    windows: list[dict[str, object]],
    bg: int,
) -> None:
    if not windows:
        return

    r0, c0, r1, c1 = bbox
    patch = out[r0 + 1 : r1, c0 + 1 : c1]
    original_bg = original[r0 + 1 : r1, c0 + 1 : c1] == bg
    if not np.any(original_bg):
        return

    if len(windows) == 1:
        patch[original_bg] = int(windows[0]["source_color"])
        return

    assert len(windows) == 2, "rooms may have at most two windows"
    by_side = {str(w["side"]): w for w in windows}
    sides = frozenset(by_side)
    hi = int(r1 - r0 - 1)
    wi = int(c1 - c0 - 1)

    if sides == frozenset(("top", "bottom")):
        primary = by_side["top"]
        secondary = by_side["bottom"]
        mode = "horizontal"
    elif sides == frozenset(("left", "right")):
        primary = by_side["left"]
        secondary = by_side["right"]
        mode = "vertical"
    else:
        assert sides in {frozenset(pair) for pair in ADJACENT_PAIRS}
        mode = "main" if sides in MAIN_DIAGONAL_PAIRS else "anti"
        primary = next(w for w in windows if str(w["side"]) in {"top", "bottom"})
        secondary = next(w for w in windows if str(w["side"]) in {"left", "right"})

    scores = _split_scores(hi, wi, mode)
    primary_negative_side = str(primary["side"]) in {"top", "left"}
    if primary_negative_side:
        primary_mask = original_bg & (scores <= 0)
    else:
        primary_mask = original_bg & (scores >= 0)
    secondary_mask = original_bg & ~primary_mask
    patch[primary_mask] = int(primary["source_color"])
    patch[secondary_mask] = int(secondary["source_color"])


def _detect_rooms(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    rooms: list[dict[str, object]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        coords = np.argwhere(grid == color)
        if coords.size == 0:
            continue

        r0, c0 = map(int, coords.min(axis=0))
        r1, c1 = map(int, coords.max(axis=0))
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue

        bbox = (r0, c0, r1, c1)
        perimeter = _perimeter_cells(r0, c0, r1, c1)
        perimeter_set = set(perimeter)
        color_cells = {tuple(map(int, rc)) for rc in coords}
        if not color_cells <= perimeter_set:
            continue

        windows: list[dict[str, object]] = []
        seen_sides: set[str] = set()
        ok = True
        for cell in perimeter:
            if cell in color_cells:
                continue
            if int(grid[cell]) != bg:
                ok = False
                break
            side = _window_side(cell, bbox)
            if side in seen_sides:
                ok = False
                break
            seen_sides.add(side)
            windows.append({"cell": cell, "side": side})
        if not ok or len(windows) > 2:
            continue

        rooms.append({"color": color, "bbox": bbox, "windows": windows})

    assert rooms, "expected at least one room frame"
    return rooms


def _ray_source_color(grid: np.ndarray, window: dict[str, object], bg: int) -> int:
    h, w = grid.shape
    r, c = map(int, window["cell"])
    dr, dc = SIDE_DIRS[str(window["side"])]
    r += dr
    c += dc
    while 0 <= r < h and 0 <= c < w:
        value = int(grid[r, c])
        if value != bg:
            assert r in {0, h - 1} or c in {0, w - 1}, "window ray must terminate on an outer marker"
            return value
        r += dr
        c += dc
    raise AssertionError("window ray did not hit an outer marker")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each hollow room interior with the color seen through its border windows, splitting two-window rooms by a center line or diagonal.

    Input structure:
    - The background is black.
    - The grid contains multiple one-pixel-thick colored rectangular hollow frames, each defining one closed room.
    - A frame may have zero, one, or two single-pixel gaps on different sides, and single-pixel color markers sit on the outer border of the grid.

    Rule:
    - Detect every rectangular room frame and treat each gap on its perimeter as a window.
    - From each window, cast a ray straight outward, perpendicular to the frame side, and read the first non-background marker on the grid boundary.
    - A one-window room fills its whole interior background with that sampled color; a two-window room splits by a horizontal or vertical midline for opposite windows, or by the matching corner-to-corner diagonal for adjacent windows, using the horizontal-side window color on its side of the split and the vertical-side window color on the other.
    - Rooms with no windows stay black inside, and all frame pixels, window gaps, and outer markers remain unchanged.

    Key insight:
    - The missing border pixels serve two roles at once: each one samples an external color by ray casting, and the arrangement of one or two windows determines how the room interior is partitioned before filling.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    rooms = _detect_rooms(g, bg)

    out = g.copy()
    for room in rooms:
        windows = []
        for window in room["windows"]:
            window_with_color = dict(window)
            window_with_color["source_color"] = _ray_source_color(g, window, bg)
            windows.append(window_with_color)
        _paint_room(out, g, tuple(room["bbox"]), windows, bg)
    return out


def _boxes_separated(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    pad: int = 1,
) -> bool:
    return (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _draw_frame(canvas: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    canvas[r0, c0 : c1 + 1] = color
    canvas[r1, c0 : c1 + 1] = color
    canvas[r0 : r1 + 1, c0] = color
    canvas[r0 : r1 + 1, c1] = color


def _draw_solid_mask(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    mask[r0, c0 : c1 + 1] = 1
    mask[r1, c0 : c1 + 1] = 1
    mask[r0 : r1 + 1, c0] = 1
    mask[r0 : r1 + 1, c1] = 1


def _window_positions(bbox: tuple[int, int, int, int], side: str) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    if side == "top":
        return [(r0, c) for c in range(c0 + 1, c1)]
    if side == "bottom":
        return [(r1, c) for c in range(c0 + 1, c1)]
    if side == "left":
        return [(r, c0) for r in range(r0 + 1, r1)]
    return [(r, c1) for r in range(r0 + 1, r1)]


def _marker_cell(cell: tuple[int, int], side: str, shape: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    r, c = cell
    if side == "top":
        return 0, c
    if side == "bottom":
        return h - 1, c
    if side == "left":
        return r, 0
    return r, w - 1


def _corridor_clear(mask: np.ndarray, cell: tuple[int, int], side: str) -> bool:
    h, w = mask.shape
    r, c = cell
    dr, dc = SIDE_DIRS[side]
    r += dr
    c += dc
    while 0 <= r < h and 0 <= c < w:
        if mask[r, c] != 0:
            return False
        r += dr
        c += dc
    return True


def _side_candidates(
    solid_mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    used_markers: set[tuple[int, int]],
) -> dict[str, list[tuple[int, int]]]:
    candidates: dict[str, list[tuple[int, int]]] = {}
    for side in SIDES:
        side_cells = _window_positions(bbox, side)
        positions = []
        for cell in side_cells:
            marker = _marker_cell(cell, side, solid_mask.shape)
            if marker in used_markers:
                continue
            if _corridor_clear(solid_mask, cell, side):
                positions.append(cell)
        candidates[side] = positions
    return candidates


def _choose_windows_for_room(
    solid_mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    used_markers: set[tuple[int, int]],
    force_nonempty: bool,
) -> list[dict[str, object]] | None:
    candidates = _side_candidates(solid_mask, bbox, used_markers)
    categories: list[str] = []
    if not force_nonempty:
        categories.append("none")
    single_sides = [side for side in SIDES if candidates[side]]
    if single_sides:
        categories.append("single")

    opposite_pairs = []
    if candidates["top"] and candidates["bottom"]:
        opposite_pairs.append(("top", "bottom"))
    if candidates["left"] and candidates["right"]:
        opposite_pairs.append(("left", "right"))
    if opposite_pairs:
        categories.append("opposite")

    adjacent_pairs = [
        pair for pair in ADJACENT_PAIRS if candidates[pair[0]] and candidates[pair[1]]
    ]
    if adjacent_pairs:
        categories.append("adjacent")

    if not categories:
        return None

    category = random.choice(categories)
    if category == "none":
        return []
    if category == "single":
        side = random.choice(single_sides)
        return [{"cell": random.choice(candidates[side]), "side": side}]
    if category == "opposite":
        side_a, side_b = random.choice(opposite_pairs)
        return [
            {"cell": random.choice(candidates[side_a]), "side": side_a},
            {"cell": random.choice(candidates[side_b]), "side": side_b},
        ]

    side_a, side_b = random.choice(adjacent_pairs)
    return [
        {"cell": random.choice(candidates[side_a]), "side": side_a},
        {"cell": random.choice(candidates[side_b]), "side": side_b},
    ]


def _sample_boxes(h: int, w: int, room_count: int) -> list[tuple[int, int, int, int]] | None:
    boxes: list[tuple[int, int, int, int]] = []
    for _ in range(room_count):
        placed = False
        for _ in range(ROOM_PLACEMENT_TRIES):
            room_h = random.randint(ROOM_SIZE_MIN, min(ROOM_SIZE_MAX, h - 2))
            room_w = random.randint(ROOM_SIZE_MIN, min(ROOM_SIZE_MAX, w - 2))
            r0 = random.randint(1, h - room_h - 1)
            c0 = random.randint(1, w - room_w - 1)
            bbox = (r0, c0, r0 + room_h - 1, c0 + room_w - 1)
            if all(_boxes_separated(bbox, other, pad=1) for other in boxes):
                boxes.append(bbox)
                placed = True
                break
        if not placed:
            return None
    return boxes


def _render_expected(inp: np.ndarray, rooms: list[dict[str, object]]) -> np.ndarray:
    expected = inp.copy()
    for room in rooms:
        _paint_room(expected, inp, tuple(room["bbox"]), room["windows"], BG)
    return expected


def _generate_case() -> np.ndarray | None:
    room_count = random.choice(ROOM_COUNT_CHOICES)
    min_size = max(GRID_MIN, 10 + 3 * room_count)
    h = random.randint(min_size, GRID_MAX)
    w = random.randint(min_size, GRID_MAX)

    boxes = _sample_boxes(h, w, room_count)
    if boxes is None:
        return None

    solid_mask = np.zeros((h, w), dtype=int)
    for bbox in boxes:
        _draw_solid_mask(solid_mask, bbox)

    rooms = [{"bbox": bbox, "windows": []} for bbox in boxes]
    order = list(range(room_count))
    random.shuffle(order)
    used_markers: set[tuple[int, int]] = set()
    nonempty_rooms = 0
    for i, room_index in enumerate(order):
        force_nonempty = i == len(order) - 1 and nonempty_rooms == 0
        windows = _choose_windows_for_room(
            solid_mask,
            tuple(rooms[room_index]["bbox"]),
            used_markers,
            force_nonempty=force_nonempty,
        )
        if windows is None:
            return None
        rooms[room_index]["windows"] = windows
        if windows:
            nonempty_rooms += 1
        for window in windows:
            used_markers.add(_marker_cell(tuple(window["cell"]), str(window["side"]), (h, w)))

    total_colors = room_count + sum(len(room["windows"]) for room in rooms)
    if total_colors > len(PALETTE):
        return None
    colors = list(PALETTE)
    random.shuffle(colors)

    grid = np.full((h, w), BG, dtype=int)
    room_colors = colors[:room_count]
    source_colors = colors[room_count:total_colors]

    for room, color in zip(rooms, room_colors, strict=False):
        room["color"] = color
        _draw_frame(grid, tuple(room["bbox"]), color)

    source_index = 0
    for room in rooms:
        room_windows = []
        for window in room["windows"]:
            cell = tuple(window["cell"])
            side = str(window["side"])
            grid[cell] = BG
            marker = _marker_cell(cell, side, grid.shape)
            source_color = source_colors[source_index]
            source_index += 1
            grid[marker] = source_color
            room_windows.append({"cell": cell, "side": side, "source_color": source_color})
        room["windows"] = room_windows

    expected = _render_expected(grid, rooms)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_TRIES):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-window puzzle")
