from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

SEED_COLORS = (RED, BLUE, GREEN, YELLOW)
ORTHO_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIAG_DIRS = ((-1, -1), (-1, 1), (1, -1), (1, 1))


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
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for dr, dc in ORTHO_DIRS:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_rooms(grid: np.ndarray) -> list[tuple[int, int, int, int, int, int, int]]:
    rooms: list[tuple[int, int, int, int, int, int, int]] = []
    for comp in _components(grid == GREY):
        r0, c0, r1, c1 = _bbox(comp)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        assert height >= 3 and width >= 3, "grey wall must enclose a nonempty interior"

        ring_area = 2 * (height + width) - 4
        assert len(comp) == ring_area, "grey component must be a 1px rectangular frame"

        comp_set = set(comp)
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                on_border = r in (r0, r1) or c in (c0, c1)
                assert ((r, c) in comp_set) == on_border, "grey component must match its bbox border exactly"

        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        assert interior.size > 0, "room interior must exist"
        assert not np.any(interior == GREY), "room interior cannot contain grey cells"

        seed_pos = np.argwhere((interior != BLACK) & (interior != GREY))
        assert seed_pos.shape == (1, 2), "each room must contain exactly one colored seed"
        sr = int(r0 + 1 + seed_pos[0, 0])
        sc = int(c0 + 1 + seed_pos[0, 1])
        seed_color = int(grid[sr, sc])
        assert seed_color in SEED_COLORS, "seed color must be red, blue, green, or yellow"

        seed_mask = np.zeros_like(interior, dtype=bool)
        seed_mask[seed_pos[0, 0], seed_pos[0, 1]] = True
        assert np.all((interior == BLACK) | seed_mask), "all non-seed interior pixels must be black"

        rooms.append((r0, c0, r1, c1, sr, sc, seed_color))

    assert rooms, "expected at least one grey rectangular room"
    rooms.sort()
    return rooms


def _paint_room(out: np.ndarray, room: tuple[int, int, int, int, int, int, int]) -> None:
    r0, c0, r1, c1, sr, sc, seed_color = room
    if seed_color == RED:
        out[sr, c0 + 1 : c1] = RED
        out[r0 + 1 : r1, sc] = RED
        return

    if seed_color == BLUE:
        for dr, dc in DIAG_DIRS:
            r, c = sr + dr, sc + dc
            while r0 < r < r1 and c0 < c < c1:
                out[r, c] = BLUE
                r += dr
                c += dc
        return

    if seed_color == GREEN:
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[interior == BLACK] = GREEN
        return

    assert seed_color == YELLOW, "unexpected seed color"
    for dr, dc in ORTHO_DIRS:
        nr, nc = sr + dr, sc + dc
        if r0 < nr < r1 and c0 < nc < c1 and out[nr, nc] == BLACK:
            out[nr, nc] = YELLOW


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each walled room by the growth pattern encoded in its seed color.

    Input structure:
    - The background is black.
    - Several grey 1-pixel-thick rectangular frames form closed rooms with black interiors.
    - Each room contains exactly one non-black, non-grey seed pixel.
    - Seed colors are restricted to red, blue, green, and yellow.

    Rule:
    - Keep every grey wall and every seed pixel in place.
    - Red seed: paint the full interior row and column through the seed, forming a plus.
    - Blue seed: paint the four interior diagonals through the seed, forming an X.
    - Green seed: fill every black interior pixel in that room with green.
    - Yellow seed: paint only the four orthogonal interior neighbors of the seed that exist.

    Key insight:
    - The seed color itself is the hidden instruction: identical-looking grey rooms expand in different ways depending only on that color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    for room in _find_rooms(g):
        _paint_room(out, room)
    return out


ROOM_COUNT_CHOICES = [2, 3, 4, 5]
ROOM_COUNT_WEIGHTS = [1, 2, 3, 1]
PLACEMENT_RETRIES = 150
GENERATION_RETRIES = 400
MIN_MARGIN = 1
ROOM_GAP = 1


def _sample_canvas_size(room_count: int) -> tuple[int, int]:
    min_side = min(28, 8 + 4 * room_count)
    return random.randint(min_side, 30), random.randint(min_side, 30)


def _sample_room_colors(room_count: int) -> list[int]:
    colors = list(SEED_COLORS)
    random.shuffle(colors)
    picked = colors[: min(room_count, len(colors))]
    while len(picked) < room_count:
        picked.append(random.choice(SEED_COLORS))
    random.shuffle(picked)
    return picked


def _sample_room_size(seed_color: int, room_count: int, h: int, w: int) -> tuple[int, int] | None:
    min_side = 6 if seed_color in (RED, BLUE) else 5
    max_side = 9 if room_count <= 3 else 8 if room_count == 4 else 7
    max_h = min(max_side, h - 2 * MIN_MARGIN)
    max_w = min(max_side, w - 2 * MIN_MARGIN)
    if max_h < min_side or max_w < min_side:
        return None
    return random.randint(min_side, max_h), random.randint(min_side, max_w)


def _sample_seed_position(
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    seed_color: int,
) -> tuple[int, int]:
    all_cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    deep_cells = [(r, c) for r in range(r0 + 2, r1 - 1) for c in range(c0 + 2, c1 - 1)]

    if seed_color in (RED, BLUE) and deep_cells and random.random() < 0.8:
        return random.choice(deep_cells)
    if seed_color == YELLOW and deep_cells and random.random() < 0.5:
        return random.choice(deep_cells)
    return random.choice(all_cells)


def _place_room_box(occupied: np.ndarray, room_h: int, room_w: int) -> tuple[int, int, int, int] | None:
    h, w = occupied.shape
    if room_h > h - 2 * MIN_MARGIN or room_w > w - 2 * MIN_MARGIN:
        return None

    row_lo = MIN_MARGIN
    row_hi = h - MIN_MARGIN - room_h
    col_lo = MIN_MARGIN
    col_hi = w - MIN_MARGIN - room_w
    if row_hi < row_lo or col_hi < col_lo:
        return None

    for _ in range(PLACEMENT_RETRIES):
        r0 = random.randint(row_lo, row_hi)
        c0 = random.randint(col_lo, col_hi)
        r1 = r0 + room_h - 1
        c1 = c0 + room_w - 1

        pr0 = max(0, r0 - ROOM_GAP)
        pc0 = max(0, c0 - ROOM_GAP)
        pr1 = min(h - 1, r1 + ROOM_GAP)
        pc1 = min(w - 1, c1 + ROOM_GAP)
        if occupied[pr0 : pr1 + 1, pc0 : pc1 + 1].any():
            continue

        occupied[pr0 : pr1 + 1, pc0 : pc1 + 1] = True
        return r0, c0, r1, c1
    return None


def _draw_input_room(grid: np.ndarray, room: tuple[int, int, int, int, int, int, int]) -> None:
    r0, c0, r1, c1, sr, sc, seed_color = room
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[sr, sc] = seed_color


def _render_expected(inp: np.ndarray, rooms: list[tuple[int, int, int, int, int, int, int]]) -> np.ndarray:
    out = inp.copy()
    for room in rooms:
        _paint_room(out, room)
    return out


def _gen_attempt() -> np.ndarray | None:
    room_count = random.choices(ROOM_COUNT_CHOICES, weights=ROOM_COUNT_WEIGHTS, k=1)[0]
    h, w = _sample_canvas_size(room_count)
    colors = _sample_room_colors(room_count)

    blueprints: list[tuple[int, int, int, int]] = []
    for color in colors:
        size = _sample_room_size(color, room_count, h, w)
        if size is None:
            return None
        room_h, room_w = size
        blueprints.append((room_h * room_w, room_h, room_w, color))
    blueprints.sort(reverse=True)

    occupied = np.zeros((h, w), dtype=bool)
    rooms: list[tuple[int, int, int, int, int, int, int]] = []
    for _area, room_h, room_w, color in blueprints:
        box = _place_room_box(occupied, room_h, room_w)
        if box is None:
            return None
        r0, c0, r1, c1 = box
        sr, sc = _sample_seed_position(r0, c0, r1, c1, color)
        rooms.append((r0, c0, r1, c1, sr, sc, color))

    grid = np.full((h, w), BLACK, dtype=int)
    for room in rooms:
        _draw_input_room(grid, room)

    expected = _render_expected(grid, rooms)
    try:
        actual = main(grid)
    except Exception:
        return None
    if np.array_equal(actual, expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"failed to generate a valid sample after {GENERATION_RETRIES} attempts")
