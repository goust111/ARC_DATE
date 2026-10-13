from __future__ import annotations

from collections import deque
from pathlib import Path
import random
import sys

import numpy as np


_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if _SOLVER_DIR.exists() and str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

try:
    from arc_utils import bbox_coords as _arc_bbox_coords
    from arc_utils import find_components as _arc_find_components
except Exception:  # pragma: no cover - local fallback keeps SourceFileLoader robust.
    _arc_bbox_coords = None
    _arc_find_components = None


BLACK = 0
Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _bbox_coords(coords: np.ndarray) -> BBox:
    if _arc_bbox_coords is not None:
        return _arc_bbox_coords(coords)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _neighbors4(r: int, c: int, shape: tuple[int, int]) -> list[Coord]:
    h, w = shape
    out: list[Coord] = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _components(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[np.ndarray] = []
    for sr0, sc0 in np.argwhere(mask):
        sr, sc = int(sr0), int(sc0)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        q: deque[Coord] = deque([(sr, sc)])
        pts: list[Coord] = []
        while q:
            r, c = q.popleft()
            pts.append((r, c))
            for nr, nc in _neighbors4(r, c, mask.shape):
                if mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(np.array(pts, dtype=int))
    return comps


def _mono_components(grid: np.ndarray, color: int) -> list[dict]:
    if _arc_find_components is not None:
        return list(
            _arc_find_components(
                grid,
                bg=BLACK,
                connectivity=4,
                monochromatic=True,
                colors=[int(color)],
                return_format="dict",
            )
        )

    out = []
    for coords in _components(grid == int(color)):
        out.append(
            {
                "color": int(color),
                "coords": coords,
                "bbox": _bbox_coords(coords),
                "area": int(coords.shape[0]),
            }
        )
    return out


def _border_cells(box: BBox) -> list[Coord]:
    r0, c0, r1, c1 = box
    cells: list[Coord] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        if r1 != r0:
            cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        if c1 != c0:
            cells.append((r, c1))
    return cells


def _interior_cells(box: BBox) -> list[Coord]:
    r0, c0, r1, c1 = box
    return [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]


def _component_as_room(grid: np.ndarray, comp: dict) -> dict | None:
    frame_color = int(comp["color"])
    r0, c0, r1, c1 = comp["bbox"]
    if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
        return None

    coords = comp["coords"]
    if any(not (r in (r0, r1) or c in (c0, c1)) for r, c in coords):
        return None

    border = _border_cells((r0, c0, r1, c1))
    if int(coords.shape[0]) != len(border) - 1:
        return None

    missing = [pos for pos in border if int(grid[pos]) != frame_color]
    if len(missing) != 1:
        return None

    lock_pos = missing[0]
    lock_color = int(grid[lock_pos])
    if lock_color in (BLACK, frame_color):
        return None

    for pos in border:
        if pos != lock_pos and int(grid[pos]) != frame_color:
            return None

    if np.any(grid[r0 + 1 : r1, c0 + 1 : c1] == frame_color):
        return None

    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "frame_color": frame_color,
        "lock_pos": lock_pos,
        "lock_color": lock_color,
    }


def _find_rooms(grid: np.ndarray) -> tuple[int, list[dict]]:
    colors = sorted(set(map(int, grid.ravel())) - {BLACK})
    by_frame: dict[int, list[dict]] = {}
    for color in colors:
        rooms = []
        for comp in _mono_components(grid, color):
            room = _component_as_room(grid, comp)
            if room is not None:
                rooms.append(room)
        if rooms:
            by_frame[int(color)] = rooms

    assert len(by_frame) == 1, "expected one shared rectangular frame color"
    frame_color, rooms = next(iter(by_frame.items()))
    rooms = sorted(rooms, key=lambda room: room["bbox"])

    seen_boxes: set[BBox] = set()
    for room in rooms:
        box = room["bbox"]
        assert box not in seen_boxes, "room bboxes must be unique"
        seen_boxes.add(box)
    return int(frame_color), rooms


def _room_area_mask(shape: tuple[int, int], rooms: list[dict]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for room in rooms:
        r0, c0, r1, c1 = room["bbox"]
        mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _annotate_room_keys(grid: np.ndarray, frame_color: int, rooms: list[dict]) -> None:
    for room in rooms:
        key_colors: set[int] = set()
        key_positions: list[Coord] = []
        for r, c in _interior_cells(room["bbox"]):
            color = int(grid[r, c])
            assert color != frame_color, "room interior cannot contain frame-color pixels"
            if color != BLACK:
                key_colors.add(color)
                key_positions.append((r, c))
        room["key_colors"] = key_colors
        room["key_positions"] = key_positions


def _opened_rooms_from_records(rooms: list[dict], initial_keys: set[int]) -> list[bool]:
    held = set(map(int, initial_keys))
    opened = [False] * len(rooms)
    changed = True
    while changed:
        changed = False
        for idx, room in enumerate(rooms):
            if opened[idx] or int(room["lock_color"]) not in held:
                continue
            opened[idx] = True
            held.update(map(int, room["key_colors"]))
            changed = True
    return opened


def main(grid: np.ndarray) -> np.ndarray:
    """
    Open color-locked rectangular rooms to a reachability fixpoint and light the rooms that open.

    Input structure:
    - A black background contains several separated one-cell-thick rectangular room frames in one shared frame color.
    - Each room border has exactly one colored lock cell replacing a frame pixel.
    - Room interiors may contain zero, one, or several single-pixel colored keys.
    - Additional colored key pixels outside all rooms form the initial key set.

    Rule:
    - Start with the colors of the outside key pixels.
    - Repeatedly open every still-closed room whose lock color is currently held, adding that room's interior key colors.
    - Fill only original background cells inside opened rooms with the room's lock color.
    - Preserve frame pixels, lock cells, outside keys, interior key pixels, and every room that never opens.

    Key insight:
    - A room is opened by the whole dependency chain of keys, not only by keys visible outside the rooms at the start.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert np.any(g == BLACK), "expected black background"

    frame_color, rooms = _find_rooms(g)
    assert rooms, "expected at least one room"
    _annotate_room_keys(g, frame_color, rooms)

    room_area = _room_area_mask(g.shape, rooms)
    outside_keys = (g != BLACK) & (g != frame_color) & ~room_area
    initial_keys = set(map(int, g[outside_keys]))

    opened = _opened_rooms_from_records(rooms, initial_keys)

    out = g.copy()
    for is_open, room in zip(opened, rooms, strict=False):
        if not is_open:
            continue
        r0, c0, r1, c1 = room["bbox"]
        interior_in = g[r0 + 1 : r1, c0 + 1 : c1]
        interior_out = out[r0 + 1 : r1, c0 + 1 : c1]
        interior_out[interior_in == BLACK] = int(room["lock_color"])
    return out


# ------------------------------ generator ------------------------------

_MAX_ATTEMPTS = 500
_PALETTE = list(range(1, 10))


def _side_positions(box: BBox, side: str) -> list[Coord]:
    r0, c0, r1, c1 = box
    if side == "top":
        return [(r0, c) for c in range(c0 + 1, c1)]
    if side == "bottom":
        return [(r1, c) for c in range(c0 + 1, c1)]
    if side == "left":
        return [(r, c0) for r in range(r0 + 1, r1)]
    if side == "right":
        return [(r, c1) for r in range(r0 + 1, r1)]
    raise ValueError("bad side")


def _draw_room(grid: np.ndarray, box: BBox, frame_color: int, lock_pos: Coord, lock_color: int) -> None:
    for r, c in _border_cells(box):
        grid[r, c] = int(frame_color)
    grid[lock_pos] = int(lock_color)


def _separated(a: BBox, b: BBox, gap: int = 1) -> bool:
    return (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _place_boxes(rows: int, cols: int, dims: list[tuple[int, int]]) -> list[BBox] | None:
    order = sorted(range(len(dims)), key=lambda i: dims[i][0] * dims[i][1], reverse=True)
    boxes: list[BBox | None] = [None] * len(dims)
    placed: list[BBox] = []
    for idx in order:
        h, w = dims[idx]
        if h + 2 > rows or w + 2 > cols:
            return None
        for _ in range(400):
            r0 = random.randint(1, rows - h - 1)
            c0 = random.randint(1, cols - w - 1)
            box = (r0, c0, r0 + h - 1, c0 + w - 1)
            if all(_separated(box, other, gap=1) for other in placed):
                boxes[idx] = box
                placed.append(box)
                break
        if boxes[idx] is None:
            return None
    return [box for box in boxes if box is not None]


def _choose_key_cells(box: BBox, count: int) -> list[Coord] | None:
    if count == 0:
        return []
    candidates = _interior_cells(box)
    random.shuffle(candidates)
    chosen: list[Coord] = []
    for cell in candidates:
        if all(max(abs(cell[0] - r), abs(cell[1] - c)) > 1 for r, c in chosen):
            chosen.append(cell)
            if len(chosen) == count:
                return chosen
    return None


def _place_outside_key(grid: np.ndarray, boxes: list[BBox], color: int) -> bool:
    rows, cols = grid.shape
    candidates: list[Coord] = []
    for r in range(rows):
        for c in range(cols):
            if int(grid[r, c]) != BLACK:
                continue
            if any(b[0] - 1 <= r <= b[2] + 1 and b[1] - 1 <= c <= b[3] + 1 for b in boxes):
                continue
            candidates.append((r, c))
    if not candidates:
        return False
    r, c = random.choice(candidates)
    grid[r, c] = int(color)
    return True


def _opened_from_specs(room_specs: list[dict], initial_keys: set[int]) -> list[bool]:
    held = set(map(int, initial_keys))
    opened = [False] * len(room_specs)
    changed = True
    while changed:
        changed = False
        for idx, room in enumerate(room_specs):
            if opened[idx] or int(room["lock_color"]) not in held:
                continue
            opened[idx] = True
            held.update(map(int, room["key_colors"]))
            changed = True
    return opened


def _expected_from_specs(grid: np.ndarray, room_specs: list[dict], initial_keys: set[int]) -> tuple[np.ndarray, list[bool]]:
    opened = _opened_from_specs(room_specs, initial_keys)
    out = grid.copy()
    for is_open, room in zip(opened, room_specs, strict=False):
        if not is_open:
            continue
        r0, c0, r1, c1 = room["bbox"]
        src = grid[r0 + 1 : r1, c0 + 1 : c1]
        dst = out[r0 + 1 : r1, c0 + 1 : c1]
        dst[src == BLACK] = int(room["lock_color"])
    return out, opened


def _semantic_checks(inp: np.ndarray, out: np.ndarray, rooms: list[dict], opened: list[bool], initial_keys: set[int]) -> bool:
    if not any(opened) or all(opened):
        return False
    direct = [int(room["lock_color"]) in initial_keys for room in rooms]
    if not any(is_open and not is_direct for is_open, is_direct in zip(opened, direct, strict=False)):
        return False
    if not any(len(room["key_colors"]) == 0 and is_open for room, is_open in zip(rooms, opened, strict=False)):
        return False
    if not any(len(room["key_colors"]) >= 2 for room in rooms):
        return False
    if np.array_equal(inp, out):
        return False

    for room, is_open in zip(rooms, opened, strict=False):
        r0, c0, r1, c1 = room["bbox"]
        if not is_open:
            if not np.array_equal(inp[r0 : r1 + 1, c0 : c1 + 1], out[r0 : r1 + 1, c0 : c1 + 1]):
                return False
            continue

        if out[room["lock_pos"]] != inp[room["lock_pos"]]:
            return False
        for r, c in _border_cells(room["bbox"]):
            if out[r, c] != inp[r, c]:
                return False
        for r, c in room["key_positions"]:
            if out[r, c] != inp[r, c]:
                return False
        if not np.any((inp[r0 + 1 : r1, c0 + 1 : c1] == BLACK) & (out[r0 + 1 : r1, c0 + 1 : c1] == room["lock_color"])):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(24, 30)
    cols = random.randint(24, 30)
    room_count = random.randint(5, 7)
    open_count = random.randint(4, room_count - 1)

    frame_color = random.choice(_PALETTE)
    lock_colors = random.sample([c for c in _PALETTE if c != frame_color], room_count)

    dims = [(random.randint(5, 7), random.randint(5, 8)) for _ in range(room_count)]
    boxes = _place_boxes(rows, cols, dims)
    if boxes is None:
        return None

    indices = list(range(room_count))
    random.shuffle(indices)
    open_order = indices[:open_count]
    closed = indices[open_count:]

    key_colors_by_room: list[list[int]] = [[] for _ in range(room_count)]
    key_colors_by_room[open_order[0]].extend([lock_colors[open_order[1]], lock_colors[open_order[2]]])
    key_colors_by_room[open_order[1]].append(lock_colors[open_order[3]])
    for pos in range(3, open_count - 1):
        key_colors_by_room[open_order[pos]].append(lock_colors[open_order[pos + 1]])

    if len(closed) == 1:
        key_colors_by_room[closed[0]].append(lock_colors[closed[0]])
    else:
        for idx, nxt in zip(closed, closed[1:] + closed[:1], strict=False):
            key_colors_by_room[idx].append(lock_colors[nxt])

    grid = np.full((rows, cols), BLACK, dtype=int)
    specs: list[dict] = []
    for idx, box in enumerate(boxes):
        side = random.choice(("top", "bottom", "left", "right"))
        lock_pos = random.choice(_side_positions(box, side))
        _draw_room(grid, box, frame_color, lock_pos, lock_colors[idx])
        specs.append(
            {
                "bbox": box,
                "lock_pos": lock_pos,
                "lock_color": int(lock_colors[idx]),
                "key_colors": set(map(int, key_colors_by_room[idx])),
                "key_positions": [],
            }
        )

    for idx, colors in enumerate(key_colors_by_room):
        positions = _choose_key_cells(boxes[idx], len(colors))
        if positions is None:
            return None
        for (r, c), color in zip(positions, colors, strict=False):
            grid[r, c] = int(color)
        specs[idx]["key_positions"] = positions

    initial_keys = {int(lock_colors[open_order[0]])}
    if not _place_outside_key(grid, boxes, next(iter(initial_keys))):
        return None

    expected, opened = _expected_from_specs(grid, specs, initial_keys)
    if not _semantic_checks(grid, expected, specs, opened, initial_keys):
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid locked-room key-chain input")
