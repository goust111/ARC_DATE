from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
GREY = 5
SEED_COLORS = (2, 3, 4, 6, 7, 8, 9)

ROOM_LAYOUTS = ((2, 2), (2, 3), (3, 2))
ROOM_H_MIN = 4
ROOM_H_MAX = 7
ROOM_W_MIN = 4
ROOM_W_MAX = 7
GEN_RETRIES = 500


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _solve_room_patch(room: np.ndarray) -> np.ndarray:
    h, w = room.shape
    seeds = np.argwhere((room != BLACK) & (room != GREY))
    assert len(seeds) == 1, "each room must contain exactly one seed"
    sr, sc = map(int, seeds[0])
    fill_color = int(room[sr, sc])

    out = room.copy()
    seen = np.zeros((h, w), dtype=bool)
    stack = [(sr, sc)]
    seen[sr, sc] = True

    while stack:
        r, c = stack.pop()
        out[r, c] = fill_color
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if seen[nr, nc] or room[nr, nc] == GREY:
                continue
            seen[nr, nc] = True
            stack.append((int(nr), int(nc)))

    return out


def _room_boxes(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    for comp in _components4(grid != BLUE):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        assert not np.any(patch == BLUE), "room interior must not contain blue separators"
        boxes.append((int(r0), int(c0), int(r1), int(c1)))
    boxes.sort()
    return boxes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Flood-fill each blue-bounded room from its seed while grey wall bricks remain as directional barriers.

    Input structure:
    - The background is black and blue single-pixel lines partition the grid into rectangular rooms.
    - Each room contains exactly one single-cell colored seed and zero or more grey wall bricks.
    - Every grey wall brick is either horizontal (1x2) or vertical (2x1), and the brick cells themselves stay grey.

    Rule:
    - Process each room independently, never crossing the blue separators.
    - Starting from the room's seed, flood-fill orthogonally through the room's non-grey cells.
    - Grey wall bricks are not filled; their orientation matters because a horizontal brick blocks top-to-bottom crossing while a vertical brick blocks left-to-right crossing, though fill may still go around exposed ends.
    - Any room cells unreachable from the seed remain black.

    Key insight:
    - The grey pieces are small directional barriers inside otherwise open rectangular rooms, so the task is room-wise flood fill with obstacles that can often be bypassed around their ends.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for r0, c0, r1, c1 in _room_boxes(g):
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        out[r0 : r1 + 1, c0 : c1 + 1] = _solve_room_patch(patch)
    return out


def _sample_sizes(count: int, low: int, high: int, require_even: bool) -> list[int]:
    vals = [random.randint(low, high) for _ in range(count)]
    if require_even and all(v % 2 == 1 for v in vals):
        even_choices = [v for v in range(low, high + 1) if v % 2 == 0]
        vals[random.randrange(count)] = random.choice(even_choices)
    return vals


def _build_rooms() -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    room_rows, room_cols = random.choice(ROOM_LAYOUTS)
    heights = _sample_sizes(room_rows, ROOM_H_MIN, ROOM_H_MAX, require_even=True)
    widths = _sample_sizes(room_cols, ROOM_W_MIN, ROOM_W_MAX, require_even=True)

    h = sum(heights) + room_rows + 1
    w = sum(widths) + room_cols + 1
    grid = np.full((h, w), BLUE, dtype=int)

    rooms: list[tuple[int, int, int, int]] = []
    r0 = 1
    for rh in heights:
        c0 = 1
        for rw in widths:
            grid[r0 : r0 + rh, c0 : c0 + rw] = BLACK
            rooms.append((int(r0), int(c0), int(rh), int(rw)))
            c0 += rw + 1
        r0 += rh + 1
    return grid, rooms


def _seed_candidates(room: np.ndarray, rows: range | None = None, cols: range | None = None) -> list[tuple[int, int]]:
    row_vals = rows if rows is not None else range(room.shape[0])
    col_vals = cols if cols is not None else range(room.shape[1])
    return [(r, c) for r in row_vals for c in col_vals if room[r, c] == BLACK]


def _place_seed(room: np.ndarray, color: int, rows: range | None = None, cols: range | None = None) -> None:
    candidates = _seed_candidates(room, rows, cols)
    assert candidates, "seed placement requires at least one empty room cell"
    sr, sc = random.choice(candidates)
    room[sr, sc] = color


def _paint_horizontal_chain(room: np.ndarray, row: int, start: int, length: int) -> None:
    assert length % 2 == 0
    for c in range(start, start + length, 2):
        room[row, c : c + 2] = GREY


def _paint_vertical_chain(room: np.ndarray, col: int, start: int, length: int) -> None:
    assert length % 2 == 0
    for r in range(start, start + length, 2):
        room[r : r + 2, col] = GREY


def _build_open_room(h: int, w: int, color: int) -> np.ndarray:
    room = np.full((h, w), BLACK, dtype=int)
    _place_seed(room, color)
    return room


def _build_sealed_h_room(h: int, w: int, color: int) -> np.ndarray:
    assert w % 2 == 0
    room = np.full((h, w), BLACK, dtype=int)
    barrier_row = random.randint(1, h - 2)
    _paint_horizontal_chain(room, barrier_row, 0, w)
    if random.random() < 0.5:
        _place_seed(room, color, rows=range(0, barrier_row))
    else:
        _place_seed(room, color, rows=range(barrier_row + 1, h))
    return room


def _build_sealed_v_room(h: int, w: int, color: int) -> np.ndarray:
    assert h % 2 == 0
    room = np.full((h, w), BLACK, dtype=int)
    barrier_col = random.randint(1, w - 2)
    _paint_vertical_chain(room, barrier_col, 0, h)
    if random.random() < 0.5:
        _place_seed(room, color, cols=range(0, barrier_col))
    else:
        _place_seed(room, color, cols=range(barrier_col + 1, w))
    return room


def _build_detour_h_room(h: int, w: int, color: int) -> np.ndarray:
    room = np.full((h, w), BLACK, dtype=int)
    barrier_row = random.randint(1, h - 2)
    gap = 1 if w % 2 == 1 else 2
    length = w - gap
    if random.random() < 0.5:
        start = 0
        seed_cols = range(0, length)
    else:
        start = gap
        seed_cols = range(start, w)
    _paint_horizontal_chain(room, barrier_row, start, length)
    if random.random() < 0.5:
        _place_seed(room, color, rows=range(0, barrier_row), cols=seed_cols)
    else:
        _place_seed(room, color, rows=range(barrier_row + 1, h), cols=seed_cols)
    return room


def _build_detour_v_room(h: int, w: int, color: int) -> np.ndarray:
    room = np.full((h, w), BLACK, dtype=int)
    barrier_col = random.randint(1, w - 2)
    gap = 1 if h % 2 == 1 else 2
    length = h - gap
    if random.random() < 0.5:
        start = 0
        seed_rows = range(0, length)
    else:
        start = gap
        seed_rows = range(start, h)
    _paint_vertical_chain(room, barrier_col, start, length)
    if random.random() < 0.5:
        _place_seed(room, color, rows=seed_rows, cols=range(0, barrier_col))
    else:
        _place_seed(room, color, rows=seed_rows, cols=range(barrier_col + 1, w))
    return room


def _build_mixed_room(h: int, w: int, color: int) -> np.ndarray:
    if h < 5 or w < 5:
        return _build_open_room(h, w, color)

    room = np.full((h, w), BLACK, dtype=int)
    tries = 0
    while tries < 50:
        tries += 1
        hr = random.randint(1, h - 2)
        hc = random.randint(1, w - 3)
        vr = random.randint(1, h - 3)
        vc = random.randint(1, w - 2)
        wall_cells = {(hr, hc), (hr, hc + 1), (vr, vc), (vr + 1, vc)}
        if len(wall_cells) != 4:
            continue
        for r, c in wall_cells:
            room[r, c] = GREY
        break
    else:
        return _build_open_room(h, w, color)

    _place_seed(room, color)
    return room


def _assign_roles(rooms: list[tuple[int, int, int, int]]) -> list[str]:
    roles = ["open"] * len(rooms)
    indices = list(range(len(rooms)))

    sealed_h = [i for i, (_r, _c, h, w) in enumerate(rooms) if w % 2 == 0]
    sealed_v = [i for i, (_r, _c, h, _w) in enumerate(rooms) if h % 2 == 0]
    seal_orients = []
    if sealed_h:
        seal_orients.append("h")
    if sealed_v:
        seal_orients.append("v")
    assert seal_orients, "at least one sealed orientation must be available"

    seal_orient = random.choice(seal_orients)
    seal_pool = sealed_h if seal_orient == "h" else sealed_v
    seal_idx = random.choice(seal_pool)
    roles[seal_idx] = f"sealed_{seal_orient}"

    remaining = [i for i in indices if i != seal_idx]
    open_idx = random.choice(remaining)
    roles[open_idx] = "open"
    remaining.remove(open_idx)

    detour_orient = "v" if seal_orient == "h" else "h"
    detour_idx = random.choice(remaining)
    roles[detour_idx] = f"detour_{detour_orient}"
    remaining.remove(detour_idx)

    extra_roles = ("open", "detour_h", "detour_v", "mixed")
    for idx in remaining:
        roles[idx] = random.choice(extra_roles)
    return roles


def _room_expected_blocked(role: str) -> bool:
    return role.startswith("sealed_")


def _render_room(h: int, w: int, color: int, role: str) -> np.ndarray:
    if role == "open":
        room = _build_open_room(h, w, color)
    elif role == "sealed_h":
        room = _build_sealed_h_room(h, w, color)
    elif role == "sealed_v":
        room = _build_sealed_v_room(h, w, color)
    elif role == "detour_h":
        room = _build_detour_h_room(h, w, color)
    elif role == "detour_v":
        room = _build_detour_v_room(h, w, color)
    else:
        room = _build_mixed_room(h, w, color)

    solved = _solve_room_patch(room)
    has_black = bool(np.any(solved == BLACK))
    if has_black != _room_expected_blocked(role):
        raise AssertionError("generated room does not match its intended reachability pattern")
    return room


def _gen_attempt() -> np.ndarray | None:
    grid, rooms = _build_rooms()
    room_count = len(rooms)
    colors = list(SEED_COLORS)
    random.shuffle(colors)
    if room_count > len(colors):
        return None

    roles = _assign_roles(rooms)

    for (r0, c0, h, w), color, role in zip(rooms, colors[:room_count], roles):
        try:
            room = _render_room(h, w, color, role)
        except AssertionError:
            return None
        grid[r0 : r0 + h, c0 : c0 + w] = room

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    if not np.any(grid == GREY):
        return None
    if not np.any(out == BLACK):
        return None
    grey = grid == GREY
    if not np.any(grey[:, :-1] & grey[:, 1:]):
        return None
    if not np.any(grey[:-1, :] & grey[1:, :]):
        return None
    if not np.array_equal(out == BLUE, grid == BLUE):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-fill sample")
