from __future__ import annotations

import random
from collections import deque

import numpy as np


_DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_PALETTE = tuple(range(10))
_MAX_ATTEMPTS = 400
_ROOM_COUNT_CHOICES = (3, 4)
_GRID_SIDE_RANGE = (18, 30)
_ROOM_OUTER_SIDE_RANGE = (7, 11)
_ROOM_PADDING = 1
_PATTERNS = (
    np.array(
        [
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 0],
            [1, 0, 0],
            [1, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [0, 1, 0],
            [1, 1, 0],
            [1, 0, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 0, 0, 0],
            [1, 1, 1, 0],
            [0, 0, 1, 0],
            [0, 1, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [0, 1, 0, 0],
            [1, 1, 1, 0],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
        ],
        dtype=bool,
    ),
)


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in _DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.asarray(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _border_coords(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
    for r in range(r0 + 1, r1):
        coords.append((r, c1))
    for c in range(c1, c0 - 1, -1):
        coords.append((r1, c))
    for r in range(r1 - 1, r0, -1):
        coords.append((r, c0))
    return coords


def _analyze_room(
    grid: np.ndarray,
    bg: int,
    frame_color: int,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
) -> dict | None:
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 5 or w < 5:
        return None

    border = _border_coords(r0, c0, r1, c1)
    values = [int(grid[r, c]) for r, c in border]
    if any(value not in (bg, frame_color) for value in values):
        return None

    corner_ids = {0, w - 1, w + h - 2, 2 * w + h - 3}
    gap_ids = [i for i, value in enumerate(values) if value == bg]
    if not (1 <= len(gap_ids) <= 3):
        return None
    if any(i in corner_ids for i in gap_ids):
        return None
    for idx in gap_ids:
        if values[idx - 1] == bg or values[(idx + 1) % len(values)] == bg:
            return None

    inside = grid[r0 + 1 : r1, c0 + 1 : c1]
    if inside.shape[0] < 3 or inside.shape[1] < 3:
        return None
    if np.any(inside == frame_color):
        return None

    object_mask = inside != bg
    if not object_mask.any():
        return None
    if object_mask[0].any() or object_mask[-1].any() or object_mask[:, 0].any() or object_mask[:, -1].any():
        return None

    parts = _components(object_mask)
    if len(parts) != 1:
        return None

    local_bbox = _bbox(parts[0])
    object_bbox = (
        int(r0 + 1 + local_bbox[0]),
        int(c0 + 1 + local_bbox[1]),
        int(r0 + 1 + local_bbox[2]),
        int(c0 + 1 + local_bbox[3]),
    )
    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "frame_color": int(frame_color),
        "gaps": int(len(gap_ids)),
        "object_bbox": object_bbox,
    }


def _find_rooms(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    rooms: list[dict] = []
    seen = set()

    for frame_color in sorted(set(map(int, grid.ravel())) - {int(bg)}):
        mask = grid == int(frame_color)
        coords = [tuple(map(int, pos)) for pos in np.argwhere(mask)]
        if len(coords) < 8:
            continue
        coord_set = set(coords)
        row_cols = {
            int(r): [int(c) for c in np.flatnonzero(mask[r])]
            for r in np.flatnonzero(mask.any(axis=1))
        }
        col_rows = {
            int(c): [int(r) for r in np.flatnonzero(mask[:, c])]
            for c in np.flatnonzero(mask.any(axis=0))
        }

        for r0, c0 in coords:
            right_options = [c1 for c1 in row_cols[r0] if c1 >= c0 + 4]
            if not right_options:
                continue
            down_options = [r1 for r1 in col_rows[c0] if r1 >= r0 + 4]
            if not down_options:
                continue

            for c1 in right_options:
                for r1 in down_options:
                    if (r1, c1) not in coord_set:
                        continue
                    key = (frame_color, r0, c0, r1, c1)
                    if key in seen:
                        continue
                    seen.add(key)
                    room = _analyze_room(grid, bg, frame_color, r0, c0, r1, c1)
                    if room is not None:
                        rooms.append(room)

    rooms.sort(
        key=lambda room: (
            -((room["bbox"][2] - room["bbox"][0] + 1) * (room["bbox"][3] - room["bbox"][1] + 1)),
            room["bbox"][0],
            room["bbox"][1],
        )
    )

    selected: list[dict] = []
    occupied = np.zeros((h, w), dtype=bool)
    for room in rooms:
        r0, c0, r1, c1 = room["bbox"]
        footprint = occupied[r0 : r1 + 1, c0 : c1 + 1]
        if footprint.any():
            continue
        occupied[r0 : r1 + 1, c0 : c1 + 1] = True
        selected.append(room)
    return selected


def _rotate_object_in_place(
    out: np.ndarray,
    object_bbox: tuple[int, int, int, int],
    turns_cw: int,
    bg: int,
    room_bbox: tuple[int, int, int, int] | None = None,
) -> None:
    turns_cw %= 4
    if turns_cw == 0:
        return

    r0, c0, r1, c1 = object_bbox
    patch = out[r0 : r1 + 1, c0 : c1 + 1].copy()
    object_mask = patch != int(bg)
    out[r0 : r1 + 1, c0 : c1 + 1][object_mask] = int(bg)

    rotated = np.rot90(patch, -turns_cw)
    new_h, new_w = map(int, rotated.shape)
    nr0 = (int(r0) + int(r1) - (new_h - 1)) // 2
    nc0 = (int(c0) + int(c1) - (new_w - 1)) // 2
    nr1 = nr0 + new_h - 1
    nc1 = nc0 + new_w - 1

    if room_bbox is not None:
        rr0, cc0, rr1, cc1 = room_bbox
        assert rr0 + 1 <= nr0 <= nr1 <= rr1 - 1
        assert cc0 + 1 <= nc0 <= nc1 <= cc1 - 1

    target = out[nr0 : nr1 + 1, nc0 : nc1 + 1]
    rot_mask = rotated != int(bg)
    target[rot_mask] = rotated[rot_mask]


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate each room's interior object according to how many window gaps its frame has.

    Input structure:
    - The most common color is the background.
    - Several one-pixel-thick rectangular frames form separate rooms.
    - Each room border is drawn in one color but contains 1 to 3 single-cell
      gaps on its edges, and the room interior holds one non-background object
      that stays at least one cell away from the border.

    Rule:
    - Detect each rectangular frame and count the number N of border gaps.
    - Rotate the interior object clockwise by N * 90 degrees around its own
      center, then place it back with the same center position.
    - Keep every frame, every gap, the background, and the grid size unchanged.

    Key insight:
    - The small breaks in the frame are not damage; they encode the rotation
      amount for the object inside that room.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    rooms = _find_rooms(g, bg)
    out = g.copy()
    for room in rooms:
        _rotate_object_in_place(out, room["object_bbox"], room["gaps"], bg, room["bbox"])
    return out


def _pattern_variants() -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    for pattern in _PATTERNS:
        keys = {np.rot90(pattern, k).tobytes() for k in range(4)}
        assert len(keys) == 4
        variants.append(pattern.copy())
    return variants


_GEN_PATTERNS = _pattern_variants()


def _sample_gap_cells(room_bbox: tuple[int, int, int, int], count: int) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = room_bbox
    border = _border_coords(r0, c0, r1, c1)
    corner_ids = {0, c1 - c0, c1 - c0 + r1 - r0, 2 * (c1 - c0) + r1 - r0}
    eligible = [idx for idx in range(len(border)) if idx not in corner_ids]
    for _ in range(200):
        picks = sorted(random.sample(eligible, count))
        good = True
        for i in range(len(picks)):
            a = picks[i]
            b = picks[(i + 1) % len(picks)]
            if i + 1 < len(picks) and b == a + 1:
                good = False
                break
        if good:
            return [border[idx] for idx in picks]
    raise RuntimeError("failed to place isolated window gaps")


def _can_place(room_bbox: tuple[int, int, int, int], occupied: np.ndarray) -> bool:
    h, w = occupied.shape
    r0, c0, r1, c1 = room_bbox
    rr0 = max(0, r0 - _ROOM_PADDING)
    cc0 = max(0, c0 - _ROOM_PADDING)
    rr1 = min(h - 1, r1 + _ROOM_PADDING)
    cc1 = min(w - 1, c1 + _ROOM_PADDING)
    return not occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any()


def _paint_room(
    grid: np.ndarray,
    bg: int,
    room_bbox: tuple[int, int, int, int],
    frame_color: int,
    object_color: int,
    pattern: np.ndarray,
    turns_cw: int,
    object_top_left: tuple[int, int],
) -> dict:
    r0, c0, r1, c1 = room_bbox
    grid[r0 : r1 + 1, c0] = int(frame_color)
    grid[r0 : r1 + 1, c1] = int(frame_color)
    grid[r0, c0 : c1 + 1] = int(frame_color)
    grid[r1, c0 : c1 + 1] = int(frame_color)

    for gr, gc in _sample_gap_cells(room_bbox, turns_cw):
        grid[gr, gc] = int(bg)

    pr, pc = object_top_left
    patch = np.where(pattern, int(object_color), int(bg))
    ph, pw = patch.shape
    grid[pr : pr + ph, pc : pc + pw][patch != int(bg)] = patch[patch != int(bg)]
    return {
        "bbox": room_bbox,
        "object_bbox": (int(pr), int(pc), int(pr + ph - 1), int(pc + pw - 1)),
        "turns": int(turns_cw),
    }


def _gen_attempt() -> np.ndarray | None:
    bg, frame_color = random.sample(_PALETTE, 2)
    object_colors = [color for color in _PALETTE if color not in {bg, frame_color}]
    h = random.randint(_GRID_SIDE_RANGE[0], _GRID_SIDE_RANGE[1])
    w = random.randint(_GRID_SIDE_RANGE[0], _GRID_SIDE_RANGE[1])
    grid = np.full((h, w), int(bg), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    room_count = random.choice(_ROOM_COUNT_CHOICES)
    gap_counts = [1, 2, 3]
    if room_count == 4:
        gap_counts.append(random.choice((1, 2, 3)))
    random.shuffle(gap_counts)

    room_specs = []
    for turns in gap_counts:
        base = random.choice(_GEN_PATTERNS)
        pattern = np.rot90(base, random.randrange(4))
        ph, pw = map(int, pattern.shape)
        room_h = random.randint(max(_ROOM_OUTER_SIDE_RANGE[0], ph + 4), max(_ROOM_OUTER_SIDE_RANGE[0], ph + 6))
        room_w = random.randint(max(_ROOM_OUTER_SIDE_RANGE[0], pw + 4), max(_ROOM_OUTER_SIDE_RANGE[0], pw + 6))
        room_h = min(room_h, _ROOM_OUTER_SIDE_RANGE[1])
        room_w = min(room_w, _ROOM_OUTER_SIDE_RANGE[1])
        if room_h < ph + 4 or room_w < pw + 4:
            return None
        room_specs.append((pattern, turns, room_h, room_w))

    rooms: list[dict] = []
    for pattern, turns, room_h, room_w in room_specs:
        placements = []
        for r0 in range(0, h - room_h + 1):
            for c0 in range(0, w - room_w + 1):
                bbox = (r0, c0, r0 + room_h - 1, c0 + room_w - 1)
                if _can_place(bbox, occupied):
                    placements.append(bbox)
        if not placements:
            return None
        room_bbox = random.choice(placements)
        r0, c0, r1, c1 = room_bbox
        pr = random.randint(r0 + 2, r1 - pattern.shape[0] - 1)
        pc = random.randint(c0 + 2, c1 - pattern.shape[1] - 1)
        object_color = random.choice(object_colors)
        room = _paint_room(grid, bg, room_bbox, frame_color, object_color, pattern, turns, (pr, pc))
        rooms.append(room)

        rr0 = max(0, r0 - _ROOM_PADDING)
        cc0 = max(0, c0 - _ROOM_PADDING)
        rr1 = min(h - 1, r1 + _ROOM_PADDING)
        cc1 = min(w - 1, c1 + _ROOM_PADDING)
        occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True

    expected = grid.copy()
    for room in rooms:
        _rotate_object_in_place(expected, room["object_bbox"], room["turns"], bg, room["bbox"])

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
    raise RuntimeError("failed to generate a valid room-rotation puzzle")
