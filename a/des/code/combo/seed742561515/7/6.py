from __future__ import annotations

import itertools
import random
from collections import defaultdict

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
RING_COLORS = (1, 2, 3, 4, 6, 7, 8)
SIDES = ("top", "bottom", "left", "right")


def _ring_positions(height: int, width: int) -> list[tuple[int, int]]:
    pos = [(0, c) for c in range(width)]
    pos.extend((r, width - 1) for r in range(1, height))
    pos.extend((height - 1, c) for c in range(width - 2, -1, -1))
    pos.extend((r, 0) for r in range(height - 2, 0, -1))
    return pos


def _ring_segment_colors(grid: np.ndarray) -> list[int]:
    ring = [int(grid[r, c]) for r, c in _ring_positions(*grid.shape)]
    assert ring, "outer ring must be non-empty"
    segments = [ring[0]]
    for color in ring[1:]:
        if color != segments[-1]:
            segments.append(color)
    return segments


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(height):
        for c in range(width):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < height and 0 <= nc < width and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _parse_room(grid: np.ndarray, comp: list[tuple[int, int]]) -> dict | None:
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    coords = set(comp)
    for r, c in coords:
        if r not in (r0, r1) and c not in (c0, c1):
            return None

    corners = ((r0, c0), (r0, c1), (r1, c0), (r1, c1))
    if any(int(grid[r, c]) != GREY for r, c in corners):
        return None

    windows: list[tuple[int, int, str]] = []
    for c in range(c0 + 1, c1):
        if int(grid[r0, c]) != GREY:
            windows.append((r0, c, "top"))
        if int(grid[r1, c]) != GREY:
            windows.append((r1, c, "bottom"))
    for r in range(r0 + 1, r1):
        if int(grid[r, c0]) != GREY:
            windows.append((r, c0, "left"))
        if int(grid[r, c1]) != GREY:
            windows.append((r, c1, "right"))

    return {
        "bbox": (r0, c0, r1, c1),
        "windows": windows,
    }


def _extract_rooms(grid: np.ndarray) -> list[dict]:
    rooms = []
    for comp in _components4(grid == GREY):
        room = _parse_room(grid, comp)
        if room is not None:
            rooms.append(room)
    rooms.sort(key=lambda room: room["bbox"])
    return rooms


def _border_color_facing_window(grid: np.ndarray, window: tuple[int, int, str]) -> int:
    r, c, side = window
    if side == "top":
        return int(grid[0, c])
    if side == "bottom":
        return int(grid[grid.shape[0] - 1, c])
    if side == "left":
        return int(grid[r, 0])
    return int(grid[r, grid.shape[1] - 1])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each grey-walled room from the outer-ring segment indexed by its window count,
    then close only those windows whose outward-facing border color matches the fill.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    segments = _ring_segment_colors(g)
    rooms = _extract_rooms(g)
    assert rooms, "expected at least one room"

    out = g.copy()
    for room in rooms:
        r0, c0, r1, c1 = room["bbox"]
        windows = room["windows"]
        window_count = len(windows)
        assert 1 <= window_count <= len(segments), "window count must index a ring segment"
        fill_color = int(segments[window_count - 1])

        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[interior == BLACK] = fill_color

        for window in windows:
            wr, wc, _ = window
            if _border_color_facing_window(g, window) == fill_color:
                out[wr, wc] = MAROON

    return out


# --------------- generator ---------------

GRID_MIN_BY_ROOMS = {1: 15, 2: 18, 3: 20}
GRID_MAX = 24
SEGMENT_COUNT_OPTIONS = (4, 5, 6)
ROOM_COUNT_OPTIONS = (1, 2, 2, 3)
ROOM_SIDE_MIN = 4
ROOM_SIDE_MAX = 7
ROOM_PADDING = 1
RING_SEGMENT_MIN_LEN = 4
ROOM_PLACEMENT_TRIES = 120
GENERATION_RETRIES = 400


def _sample_segment_lengths(total: int, count: int) -> list[int] | None:
    min_total = count * RING_SEGMENT_MIN_LEN
    if total < min_total:
        return None
    extras = np.random.multinomial(total - min_total, np.ones(count) / count)
    lengths = [RING_SEGMENT_MIN_LEN + int(x) for x in extras]
    random.shuffle(lengths)
    return lengths


def _paint_ring(grid: np.ndarray, colors: list[int], lengths: list[int]) -> None:
    ring = _ring_positions(*grid.shape)
    assert sum(lengths) == len(ring)
    idx = 0
    for color, length in zip(colors, lengths, strict=False):
        for _ in range(length):
            r, c = ring[idx]
            grid[r, c] = int(color)
            idx += 1


def _side_slots(r0: int, c0: int, r1: int, c1: int, grid: np.ndarray) -> dict[str, list[dict]]:
    slots: dict[str, list[dict]] = {side: [] for side in SIDES}
    for idx, c in enumerate(range(c0 + 1, c1)):
        slots["top"].append({"r": r0, "c": c, "side": "top", "idx": idx, "border_color": int(grid[0, c])})
        slots["bottom"].append(
            {"r": r1, "c": c, "side": "bottom", "idx": idx, "border_color": int(grid[grid.shape[0] - 1, c])}
        )
    for idx, r in enumerate(range(r0 + 1, r1)):
        slots["left"].append({"r": r, "c": c0, "side": "left", "idx": idx, "border_color": int(grid[r, 0])})
        slots["right"].append(
            {"r": r, "c": c1, "side": "right", "idx": idx, "border_color": int(grid[r, grid.shape[1] - 1])}
        )
    return slots


def _no_adjacent_subsets(slots: list[dict]) -> list[list[dict]]:
    count = len(slots)
    subsets: list[list[dict]] = []
    for mask in range(1 << count):
        if mask & (mask << 1):
            continue
        subsets.append([slots[i] for i in range(count) if (mask >> i) & 1])
    return subsets


def _windows_keep_wall_connected(bbox: tuple[int, int, int, int], windows: list[tuple[int, int, str]]) -> bool:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    for r, c, _side in windows:
        mask[r - r0, c - c0] = False
    comps = _components4(mask)
    return len(comps) == 1


def _window_plan_groups(grid: np.ndarray, bbox: tuple[int, int, int, int], segments: list[int]) -> dict[tuple[int, str], list[dict]]:
    r0, c0, r1, c1 = bbox
    slots_by_side = _side_slots(r0, c0, r1, c1, grid)
    subset_options = [_no_adjacent_subsets(slots_by_side[side]) for side in SIDES]
    grouped: dict[tuple[int, str], list[dict]] = defaultdict(list)

    for combo in itertools.product(*subset_options):
        chosen = [slot for subset in combo for slot in subset]
        window_count = len(chosen)
        if not (1 <= window_count <= len(segments)):
            continue
        windows = [(int(slot["r"]), int(slot["c"]), str(slot["side"])) for slot in chosen]
        if not _windows_keep_wall_connected(bbox, windows):
            continue
        fill_color = int(segments[window_count - 1])
        match_count = sum(int(slot["border_color"] == fill_color) for slot in chosen)
        if match_count == window_count:
            mode = "all_close"
        elif match_count == 0:
            mode = "all_open"
        else:
            mode = "mixed"
        grouped[(window_count, mode)].append(
            {
                "fill_color": fill_color,
                "match_count": int(match_count),
                "windows": windows,
            }
        )

    return grouped


def _mode_has_match(mode: str) -> bool:
    return mode in {"all_close", "mixed"}


def _mode_has_open(mode: str) -> bool:
    return mode in {"all_open", "mixed"}


def _choose_plan(grouped: dict[tuple[int, str], list[dict]], need_match: bool, need_open: bool) -> dict | None:
    keys = list(grouped)
    if not keys:
        return None

    strict = [
        key
        for key in keys
        if (not need_match or _mode_has_match(key[1])) and (not need_open or _mode_has_open(key[1]))
    ]
    if strict:
        key_pool = strict
    else:
        soft = [
            key
            for key in keys
            if (need_match and _mode_has_match(key[1])) or (need_open and _mode_has_open(key[1]))
        ]
        key_pool = soft if soft else keys

    key = random.choice(key_pool)
    return random.choice(grouped[key])


def _bbox_is_free(occupied: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int) -> bool:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(occupied.shape[0], r1 + pad + 1)
    cc1 = min(occupied.shape[1], c1 + pad + 1)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _mark_bbox(occupied: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    occupied[r0 : r1 + 1, c0 : c1 + 1] = True


def _draw_room(grid: np.ndarray, room: dict) -> None:
    r0, c0, r1, c1 = room["bbox"]
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY
    for r, c, _side in room["windows"]:
        grid[r, c] = BLACK


def _build_expected(grid: np.ndarray, rooms: list[dict]) -> np.ndarray:
    out = grid.copy()
    for room in rooms:
        r0, c0, r1, c1 = room["bbox"]
        fill_color = int(room["fill_color"])
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[interior == BLACK] = fill_color
        for window in room["windows"]:
            wr, wc, _ = window
            if _border_color_facing_window(grid, window) == fill_color:
                out[wr, wc] = MAROON
    return out


def _sample_rooms(grid: np.ndarray, segments: list[int], room_count: int) -> list[dict] | None:
    height, width = grid.shape
    occupied = np.zeros((height, width), dtype=bool)
    rooms: list[dict] = []
    has_match = False
    has_open = False

    for _room_idx in range(room_count):
        placed = False
        for _ in range(ROOM_PLACEMENT_TRIES):
            room_h = random.randint(ROOM_SIDE_MIN, min(ROOM_SIDE_MAX, height - 4))
            room_w = random.randint(ROOM_SIDE_MIN, min(ROOM_SIDE_MAX, width - 4))
            max_r0 = height - room_h - 2
            max_c0 = width - room_w - 2
            if max_r0 < 2 or max_c0 < 2:
                continue
            r0 = random.randint(2, max_r0)
            c0 = random.randint(2, max_c0)
            r1 = r0 + room_h - 1
            c1 = c0 + room_w - 1
            if not _bbox_is_free(occupied, r0, c0, r1, c1, ROOM_PADDING):
                continue

            grouped = _window_plan_groups(grid, (r0, c0, r1, c1), segments)
            plan = _choose_plan(grouped, need_match=not has_match, need_open=not has_open)
            if plan is None:
                continue

            room = {
                "bbox": (r0, c0, r1, c1),
                "windows": plan["windows"],
                "fill_color": int(plan["fill_color"]),
                "match_count": int(plan["match_count"]),
            }
            rooms.append(room)
            _mark_bbox(occupied, r0, c0, r1, c1)
            has_match |= room["match_count"] > 0
            has_open |= room["match_count"] < len(room["windows"])
            placed = True
            break

        if not placed:
            return None

    total_windows = sum(len(room["windows"]) for room in rooms)
    if total_windows >= 2 and not (has_match and has_open):
        return None
    return rooms


def _generate_attempt() -> np.ndarray | None:
    room_count = random.choice(ROOM_COUNT_OPTIONS)
    min_side = GRID_MIN_BY_ROOMS[int(room_count)]
    height = random.randint(min_side, GRID_MAX)
    width = random.randint(min_side, GRID_MAX)
    grid = np.zeros((height, width), dtype=int)

    segment_count = random.choice(SEGMENT_COUNT_OPTIONS)
    colors = random.sample(RING_COLORS, segment_count)
    lengths = _sample_segment_lengths(2 * height + 2 * width - 4, segment_count)
    if lengths is None:
        return None
    _paint_ring(grid, colors, lengths)

    segments = _ring_segment_colors(grid)
    if segments != colors:
        return None

    rooms = _sample_rooms(grid, segments, room_count)
    if rooms is None:
        return None

    for room in rooms:
        _draw_room(grid, room)

    expected = _build_expected(grid, rooms)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-and-ring puzzle")
