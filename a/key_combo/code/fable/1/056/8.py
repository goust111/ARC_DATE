from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
BLUE = 1
RED = 2
WALL = 3
YELLOW = 4
SIGN = 5

ROOM_FILL_COLORS = (BLUE, RED, YELLOW)

SYMBOL_BY_COLOR = {
    BLUE: np.array(
        [
            [1, 1, 1],
            [0, 1, 0],
            [0, 1, 0],
        ],
        dtype=int,
    ),
    RED: np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ],
        dtype=int,
    ),
    YELLOW: np.array(
        [
            [1, 0, 0],
            [1, 0, 0],
            [1, 1, 1],
        ],
        dtype=int,
    ),
}
PATTERN_TO_COLOR = {
    tuple(tuple(int(v) for v in row) for row in pattern.tolist()): int(color)
    for color, pattern in SYMBOL_BY_COLOR.items()
}

GRID_SIDE_RANGE = (21, 29)
ROOM_COUNT_CHOICES = (4, 5)
INTERIOR_SIDE_RANGE = (5, 9)
ROOM_PAD = 1
ROOM_PLACEMENT_ATTEMPTS = 80
GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _touches_border(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    rows = coords[:, 0]
    cols = coords[:, 1]
    return bool((rows == 0).any() or (rows == h - 1).any() or (cols == 0).any() or (cols == w - 1).any())


def _paste(dst: np.ndarray, patch: np.ndarray, top: int, left: int, *, transparent: int = BACKGROUND) -> None:
    h, w = patch.shape
    view = dst[top : top + h, left : left + w]
    mask = patch != int(transparent)
    view[mask] = patch[mask]


def _symbol_patch(color: int) -> np.ndarray:
    return np.where(SYMBOL_BY_COLOR[int(color)] == 1, SIGN, BACKGROUND).astype(int)


def _symbol_offset(inner_h: int, inner_w: int) -> tuple[int, int]:
    return 1 + (int(inner_h) - 5) // 2, 1 + (int(inner_w) - 5) // 2


def _room_component_masks(grid: np.ndarray) -> list[np.ndarray]:
    return [
        comp
        for comp in _components_mask(grid != WALL)
        if not _touches_border(comp, grid.shape)
    ]


def _room_sign_color(grid: np.ndarray, room_coords: np.ndarray) -> int | None:
    mask = np.zeros(grid.shape, dtype=bool)
    mask[room_coords[:, 0], room_coords[:, 1]] = True
    sign_comps = _components_mask((grid == SIGN) & mask)
    if len(sign_comps) != 1:
        return None
    bbox = _bbox_coords(sign_comps[0])
    patch = _crop_bbox((grid == SIGN).astype(int), bbox)
    if patch.shape != (3, 3):
        return None
    return PATTERN_TO_COLOR.get(tuple(tuple(int(v) for v in row) for row in patch.tolist()))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode each enclosed room's gray sign into a fill color, paint that room, and erase all gray signs.

    Input structure:
    - Black background with one or more green closed wall loops; each loop surrounds a rectangular room interior.
    - Some room interiors contain a single 3x3 gray connected symbol, while other rooms stay empty and black.
    - Matching gray 3x3 symbols may also appear outside every enclosed room as distractors.

    Rule:
    - Treat every non-green connected region that does not touch the grid border as an enclosed room interior.
    - Erase every gray symbol everywhere first.
    - If an enclosed room contains exactly one recognized 3x3 gray symbol, fill the whole room interior, including the former sign cells, with the fixed color encoded by that symbol; if the room has no sign, leave it black.
    - Keep all green walls unchanged.

    Key insight:
    - The gray 3x3 shapes are not decorations to preserve; they are reusable lookup-table nameplates whose meaning only activates inside topologically enclosed regions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    out[out == SIGN] = BACKGROUND

    for room_coords in _room_component_masks(g):
        fill_color = _room_sign_color(g, room_coords)
        if fill_color is None:
            continue
        out[room_coords[:, 0], room_coords[:, 1]] = int(fill_color)

    return out


def _separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int) -> bool:
    return (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _sample_room_layout() -> tuple[int, int, list[dict]] | None:
    room_count = random.choice(ROOM_COUNT_CHOICES)
    min_side = 24 if room_count == 5 else GRID_SIDE_RANGE[0]
    h = random.randint(min_side, GRID_SIDE_RANGE[1])
    w = random.randint(min_side, GRID_SIDE_RANGE[1])

    room_specs = [
        (
            random.randint(*INTERIOR_SIDE_RANGE),
            random.randint(*INTERIOR_SIDE_RANGE),
        )
        for _ in range(room_count)
    ]
    room_specs.sort(key=lambda item: item[0] * item[1], reverse=True)

    rooms: list[dict] = []
    for inner_h, inner_w in room_specs:
        outer_h = int(inner_h) + 2
        outer_w = int(inner_w) + 2
        placed = False
        for _ in range(ROOM_PLACEMENT_ATTEMPTS):
            top = random.randint(1, h - outer_h - 1)
            left = random.randint(1, w - outer_w - 1)
            bbox = (top, left, top + outer_h - 1, left + outer_w - 1)
            if all(_separated(bbox, room["bbox"], ROOM_PAD) for room in rooms):
                rooms.append(
                    {
                        "bbox": bbox,
                        "inner_h": int(inner_h),
                        "inner_w": int(inner_w),
                    }
                )
                placed = True
                break
        if not placed:
            return None

    fills = [BLUE, RED, YELLOW, BACKGROUND]
    if room_count == 5:
        fills.append(random.choice([BACKGROUND, random.choice(ROOM_FILL_COLORS)]))
    random.shuffle(fills)
    for room, fill_color in zip(rooms, fills):
        room["fill_color"] = int(fill_color)
    return h, w, rooms


def _room_interior_bbox(room: dict) -> tuple[int, int, int, int]:
    top, left, bottom, right = room["bbox"]
    return top + 1, left + 1, bottom - 1, right - 1


def _build_expected(h: int, w: int, rooms: list[dict]) -> np.ndarray:
    grid = np.full((h, w), BACKGROUND, dtype=int)
    for room in rooms:
        top, left, bottom, right = room["bbox"]
        grid[top : bottom + 1, left : right + 1] = WALL
        r0, c0, r1, c1 = _room_interior_bbox(room)
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(room["fill_color"])
    return grid


def _decorate_input(expected: np.ndarray, rooms: list[dict]) -> tuple[np.ndarray, list[dict]]:
    inp = expected.copy()
    distractors: list[dict] = []

    for room in rooms:
        r0, c0, r1, c1 = _room_interior_bbox(room)
        inp[r0 : r1 + 1, c0 : c1 + 1] = BACKGROUND
        fill_color = int(room["fill_color"])
        if fill_color == BACKGROUND:
            continue
        dr, dc = _symbol_offset(room["inner_h"], room["inner_w"])
        sign_top = r0 + dr
        sign_left = c0 + dc
        room["sign_top_left"] = (int(sign_top), int(sign_left))
        _paste(inp, _symbol_patch(fill_color), sign_top, sign_left)

    reserved = np.zeros(inp.shape, dtype=bool)
    for room in rooms:
        top, left, bottom, right = room["bbox"]
        r0 = max(0, top - 1)
        c0 = max(0, left - 1)
        r1 = min(inp.shape[0] - 1, bottom + 1)
        c1 = min(inp.shape[1] - 1, right + 1)
        reserved[r0 : r1 + 1, c0 : c1 + 1] = True

    room_order = sorted(rooms, key=lambda room: (room["bbox"][0], room["bbox"][1]))
    distractor_colors = [room["fill_color"] for room in room_order if room["fill_color"] != BACKGROUND]
    target_count = 2 if len(rooms) == 4 else 3
    placed = 0
    for color in distractor_colors:
        if placed >= target_count:
            break
        found = False
        for top in range(1, inp.shape[0] - 3):
            if found:
                break
            for left in range(1, inp.shape[1] - 3):
                if reserved[top - 1 : top + 4, left - 1 : left + 4].any():
                    continue
                patch = _symbol_patch(int(color))
                _paste(inp, patch, top, left)
                reserved[top - 1 : top + 4, left - 1 : left + 4] = True
                distractors.append(
                    {
                        "color": int(color),
                        "top_left": (int(top), int(left)),
                    }
                )
                placed += 1
                found = True
                break
        if not found:
            return inp, []

    return inp, distractors


def _validate_semantics(inp: np.ndarray, expected: np.ndarray, rooms: list[dict], distractors: list[dict]) -> bool:
    if not distractors:
        return False
    if not np.array_equal(main(inp), expected):
        return False

    room_components = _room_component_masks(inp)
    if len(room_components) != len(rooms):
        return False

    fills = [int(room["fill_color"]) for room in rooms]
    if set(ROOM_FILL_COLORS) - set(fills):
        return False
    if BACKGROUND not in fills:
        return False

    for room in rooms:
        r0, c0, r1, c1 = _room_interior_bbox(room)
        interior = expected[r0 : r1 + 1, c0 : c1 + 1]
        fill_color = int(room["fill_color"])
        if fill_color == BACKGROUND:
            if np.any(inp[r0 : r1 + 1, c0 : c1 + 1] == SIGN):
                return False
            if not np.all(interior == BACKGROUND):
                return False
        else:
            sign_top, sign_left = room["sign_top_left"]
            patch = _symbol_patch(fill_color)
            observed = inp[sign_top : sign_top + 3, sign_left : sign_left + 3]
            if not np.array_equal(observed, patch):
                return False
            if not np.all(interior == fill_color):
                return False

    for distractor in distractors:
        top, left = distractor["top_left"]
        patch = _symbol_patch(int(distractor["color"]))
        observed = inp[top : top + 3, left : left + 3]
        if not np.array_equal(observed, patch):
            return False
        if not np.all(expected[top : top + 3, left : left + 3] == BACKGROUND):
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    layout = _sample_room_layout()
    if layout is None:
        return None
    h, w, rooms = layout
    expected = _build_expected(h, w, rooms)
    inp, distractors = _decorate_input(expected, rooms)
    if not _validate_semantics(inp, expected, rooms, distractors):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-nameplate input")
