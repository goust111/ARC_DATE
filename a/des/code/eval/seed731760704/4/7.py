from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
AZURE = 8
SPRITE_COLORS = (1, 2, 3, 4, 5, 6, 7, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

GEN_RETRIES = 300
SHAPE_RETRIES = 500
MODULE_RETRIES = 120
PLACEMENT_RETRIES = 120


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
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
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    r0, c0, _, _ = _bbox(coords)
    return sorted((r - r0, c - c0) for r, c in coords)


def _shape_size(coords: list[tuple[int, int]]) -> tuple[int, int]:
    r0, c0, r1, c1 = _bbox(coords)
    return r1 - r0 + 1, c1 - c0 + 1


def _shape_escape_direction(
    coords: list[tuple[int, int]], axis: str
) -> str | None:
    rel = _normalize(coords)
    h, w = _shape_size(rel)
    if axis == "horizontal":
        left = sum(2 * c + 1 < w for _, c in rel)
        right = sum(2 * c + 1 > w for _, c in rel)
        if left == right:
            return None
        return "right" if left > right else "left"
    top = sum(2 * r + 1 < h for r, _ in rel)
    bottom = sum(2 * r + 1 > h for r, _ in rel)
    if top == bottom:
        return None
    return "down" if top > bottom else "up"


def _translate(
    coords: list[tuple[int, int]], dr: int, dc: int
) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for r, c in coords]


def _random_connected_shape(box_h: int, box_w: int, area: int) -> list[tuple[int, int]]:
    start = (random.randrange(box_h), random.randrange(box_w))
    cells = {start}
    frontier = set(_neighbors4(start[0], start[1], box_h, box_w))
    while len(cells) < area:
        if not frontier:
            return []
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        for nxt in _neighbors4(cell[0], cell[1], box_h, box_w):
            if nxt not in cells:
                frontier.add(nxt)
    return _normalize(sorted(cells))


def _sample_shape(max_h: int, max_w: int, direction: str) -> list[tuple[int, int]] | None:
    axis = "horizontal" if direction in ("left", "right") else "vertical"
    min_h = 1 if axis == "horizontal" else 2
    min_w = 2 if axis == "horizontal" else 1
    if max_h < min_h or max_w < min_w:
        return None

    for _ in range(SHAPE_RETRIES):
        area = random.randint(3, 6)
        box_h = random.randint(min_h, min(max_h, 4))
        box_w = random.randint(min_w, min(max_w, 4))
        if box_h * box_w < area:
            continue
        shape = _random_connected_shape(box_h, box_w, area)
        if not shape:
            continue
        h, w = _shape_size(shape)
        if h > max_h or w > max_w:
            continue
        if _shape_escape_direction(shape, axis) != direction:
            continue
        return shape
    return None


def _place_shape_in_room(
    room_h: int,
    room_w: int,
    shape: list[tuple[int, int]],
    gap_index: int,
    orientation: str,
) -> tuple[int, int] | None:
    sh, sw = _shape_size(shape)
    if orientation == "vertical":
        tops = [
            top
            for top in range(1, room_h - sh)
            if top <= gap_index <= top + sh - 1
        ]
        lefts = list(range(1, room_w - sw))
    else:
        tops = list(range(1, room_h - sh))
        lefts = [
            left
            for left in range(1, room_w - sw)
            if left <= gap_index <= left + sw - 1
        ]
    if not tops or not lefts:
        return None
    return random.choice(tops), random.choice(lefts)


def _room_sprite(
    grid: np.ndarray, room: tuple[int, int, int, int]
) -> dict[str, object]:
    r0, r1, c0, c1 = room
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert not np.any(sub == AZURE)
    mask = sub != BLACK
    comps = _components(mask)
    assert len(comps) == 1
    coords = [(r0 + r, c0 + c) for r, c in comps[0]]
    colors = {int(grid[r, c]) for r, c in coords}
    assert len(colors) == 1
    return {"coords": coords, "color": next(iter(colors))}


def _parse_module(
    grid: np.ndarray, coords: list[tuple[int, int]]
) -> dict[str, object]:
    r0, c0, r1, c1 = _bbox(coords)
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert sub.shape[0] >= 5 and sub.shape[1] >= 5
    assert np.all(sub[0, :] == AZURE)
    assert np.all(sub[-1, :] == AZURE)
    assert np.all(sub[:, 0] == AZURE)
    assert np.all(sub[:, -1] == AZURE)

    vertical_candidates: list[tuple[int, int]] = []
    for local_c in range(1, sub.shape[1] - 1):
        col = sub[1:-1, local_c]
        if np.count_nonzero(col != AZURE) == 1 and np.count_nonzero(col == BLACK) == 1:
            gap_r = int(np.where(col == BLACK)[0][0]) + 1
            vertical_candidates.append((local_c, gap_r))

    horizontal_candidates: list[tuple[int, int]] = []
    for local_r in range(1, sub.shape[0] - 1):
        row = sub[local_r, 1:-1]
        if np.count_nonzero(row != AZURE) == 1 and np.count_nonzero(row == BLACK) == 1:
            gap_c = int(np.where(row == BLACK)[0][0]) + 1
            horizontal_candidates.append((local_r, gap_c))

    assert len(vertical_candidates) + len(horizontal_candidates) == 1

    if vertical_candidates:
        wall_c, gap_r = vertical_candidates[0]
        left_room = (r0 + 1, r1 - 1, c0 + 1, c0 + wall_c - 1)
        right_room = (r0 + 1, r1 - 1, c0 + wall_c + 1, c1 - 1)
        assert left_room[2] <= left_room[3] and right_room[2] <= right_room[3]
        return {
            "orientation": "vertical",
            "gap": (r0 + gap_r, c0 + wall_c),
            "moves": (
                (left_room, right_room, "right"),
                (right_room, left_room, "left"),
            ),
        }

    wall_r, gap_c = horizontal_candidates[0]
    top_room = (r0 + 1, r0 + wall_r - 1, c0 + 1, c1 - 1)
    bottom_room = (r0 + wall_r + 1, r1 - 1, c0 + 1, c1 - 1)
    assert top_room[0] <= top_room[1] and bottom_room[0] <= bottom_room[1]
    return {
        "orientation": "horizontal",
        "gap": (r0 + wall_r, c0 + gap_c),
        "moves": (
            (top_room, bottom_room, "down"),
            (bottom_room, top_room, "up"),
        ),
    }


def _move_into_room(
    coords: list[tuple[int, int]],
    direction: str,
    room: tuple[int, int, int, int],
) -> list[tuple[int, int]]:
    r0, r1, c0, c1 = room
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    if direction == "right":
        shift = c1 - max(cols)
        moved = _translate(coords, 0, shift)
    elif direction == "left":
        shift = c0 - min(cols)
        moved = _translate(coords, 0, shift)
    elif direction == "down":
        shift = r1 - max(rows)
        moved = _translate(coords, shift, 0)
    else:
        assert direction == "up"
        shift = r0 - min(rows)
        moved = _translate(coords, shift, 0)

    for r, c in moved:
        assert r0 <= r <= r1 and c0 <= c <= c1
    return moved


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each sprite through the unique wall gap of its room.

    This implementation treats each azure structure as an independent two-room
    module: an outer rectangle plus one inner wall with a single-cell gap.
    The gap picks the movement axis, and the sprite's imbalance on that axis
    picks the concrete direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    sprite_mask = (g != BLACK) & (g != AZURE)
    out[sprite_mask] = BLACK

    for comp in _components(g == AZURE):
        module = _parse_module(g, comp)
        axis = "horizontal" if module["orientation"] == "vertical" else "vertical"
        for src_room, dst_room, required_direction in module["moves"]:
            sprite = _room_sprite(g, src_room)
            actual_direction = _shape_escape_direction(sprite["coords"], axis)
            assert actual_direction == required_direction
            moved = _move_into_room(sprite["coords"], required_direction, dst_room)
            for r, c in moved:
                out[r, c] = int(sprite["color"])
    return out


def _build_vertical_module() -> np.ndarray | None:
    room_h = random.randint(5, 8)
    left_w = random.randint(4, 8)
    right_w = random.randint(4, 8)
    gap_row = random.randint(1, room_h - 2)

    left_shape = _sample_shape(
        max_h=room_h - 2,
        max_w=min(left_w - 2, right_w),
        direction="right",
    )
    right_shape = _sample_shape(
        max_h=room_h - 2,
        max_w=min(right_w - 2, left_w),
        direction="left",
    )
    if left_shape is None or right_shape is None:
        return None

    left_pos = _place_shape_in_room(room_h, left_w, left_shape, gap_row, "vertical")
    right_pos = _place_shape_in_room(room_h, right_w, right_shape, gap_row, "vertical")
    if left_pos is None or right_pos is None:
        return None

    h = room_h + 2
    w = left_w + right_w + 3
    wall_c = left_w + 1
    grid = np.full((h, w), BLACK, dtype=int)
    grid[0, :] = AZURE
    grid[-1, :] = AZURE
    grid[:, 0] = AZURE
    grid[:, -1] = AZURE
    grid[1:-1, wall_c] = AZURE
    grid[1 + gap_row, wall_c] = BLACK

    left_color, right_color = random.sample(SPRITE_COLORS, 2)
    top, left = left_pos
    for dr, dc in left_shape:
        grid[1 + top + dr, 1 + left + dc] = left_color
    top, left = right_pos
    base_c = wall_c + 1
    for dr, dc in right_shape:
        grid[1 + top + dr, base_c + left + dc] = right_color
    return grid


def _build_horizontal_module() -> np.ndarray | None:
    room_w = random.randint(5, 8)
    top_h = random.randint(4, 8)
    bottom_h = random.randint(4, 8)
    gap_col = random.randint(1, room_w - 2)

    top_shape = _sample_shape(
        max_h=min(top_h - 2, bottom_h),
        max_w=room_w - 2,
        direction="down",
    )
    bottom_shape = _sample_shape(
        max_h=min(bottom_h - 2, top_h),
        max_w=room_w - 2,
        direction="up",
    )
    if top_shape is None or bottom_shape is None:
        return None

    top_pos = _place_shape_in_room(top_h, room_w, top_shape, gap_col, "horizontal")
    bottom_pos = _place_shape_in_room(bottom_h, room_w, bottom_shape, gap_col, "horizontal")
    if top_pos is None or bottom_pos is None:
        return None

    h = top_h + bottom_h + 3
    w = room_w + 2
    wall_r = top_h + 1
    grid = np.full((h, w), BLACK, dtype=int)
    grid[0, :] = AZURE
    grid[-1, :] = AZURE
    grid[:, 0] = AZURE
    grid[:, -1] = AZURE
    grid[wall_r, 1:-1] = AZURE
    grid[wall_r, 1 + gap_col] = BLACK

    top_color, bottom_color = random.sample(SPRITE_COLORS, 2)
    top, left = top_pos
    for dr, dc in top_shape:
        grid[1 + top + dr, 1 + left + dc] = top_color
    top, left = bottom_pos
    base_r = wall_r + 1
    for dr, dc in bottom_shape:
        grid[base_r + top + dr, 1 + left + dc] = bottom_color
    return grid


def _sample_module() -> np.ndarray | None:
    builder = random.choice((_build_vertical_module, _build_horizontal_module))
    return builder()


def _can_place(
    occupied: np.ndarray, top: int, left: int, h: int, w: int, pad: int = 1
) -> bool:
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(occupied.shape[0], top + h + pad)
    c1 = min(occupied.shape[1], left + w + pad)
    return not occupied[r0:r1, c0:c1].any()


def _pack_modules(modules: list[np.ndarray]) -> np.ndarray | None:
    max_h = max(m.shape[0] for m in modules)
    max_w = max(m.shape[1] for m in modules)
    sum_h = sum(m.shape[0] for m in modules)
    sum_w = sum(m.shape[1] for m in modules)
    canvas_h = max(max_h + 2, min(30, sum_h + random.randint(2, 8)))
    canvas_w = max(max_w + 2, min(30, sum_w + random.randint(2, 8)))
    if canvas_h < max_h or canvas_w < max_w:
        return None

    for _ in range(PLACEMENT_RETRIES):
        grid = np.full((canvas_h, canvas_w), BLACK, dtype=int)
        occupied = np.zeros((canvas_h, canvas_w), dtype=bool)
        placed = True
        for module in random.sample(modules, len(modules)):
            mh, mw = module.shape
            spots = [
                (r, c)
                for r in range(canvas_h - mh + 1)
                for c in range(canvas_w - mw + 1)
                if _can_place(occupied, r, c, mh, mw, pad=1)
            ]
            if not spots:
                placed = False
                break
            top, left = random.choice(spots)
            grid[top : top + mh, left : left + mw] = np.where(
                module != BLACK,
                module,
                grid[top : top + mh, left : left + mw],
            )
            occupied[top : top + mh, left : left + mw] |= module != BLACK
        if placed:
            return grid
    return None


def _gen_attempt() -> np.ndarray | None:
    module_count = random.randint(1, 3)
    modules: list[np.ndarray] = []
    for _ in range(module_count):
        module = None
        for _ in range(MODULE_RETRIES):
            module = _sample_module()
            if module is not None:
                break
        if module is None:
            return None
        modules.append(module)

    grid = _pack_modules(modules)
    if grid is None:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
