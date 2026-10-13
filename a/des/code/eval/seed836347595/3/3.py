from __future__ import annotations

import random

import numpy as np


BLACK = 0
MAROON = 9
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8]

UP = "up"
DOWN = "down"
LEFT = "left"
RIGHT = "right"
DIRECTIONS = (UP, DOWN, LEFT, RIGHT)
DIR_STEP = {
    UP: (-1, 0),
    DOWN: (1, 0),
    LEFT: (0, -1),
    RIGHT: (0, 1),
}

ARROW_MASKS = {
    UP: np.array([[0, 1, 0], [1, 1, 1]], dtype=bool),
    DOWN: np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    LEFT: np.array([[0, 1], [1, 1], [0, 1]], dtype=bool),
    RIGHT: np.array([[1, 0], [1, 1], [1, 0]], dtype=bool),
}

TABLE_ROW_STRIDE = 4
TABLE_HEIGHT = 15
TABLE_WIDTH = 7
TABLE_ARROW_COL = 4
TABLE_CORNERS = ("tl", "tr", "bl", "br")

GRID_HEIGHT_RANGE = (22, 30)
GRID_WIDTH_RANGE = (22, 30)
ROOM_COUNT_RANGE = (2, 4)
INNER_HEIGHT_RANGE = (2, 4)
INNER_WIDTH_RANGE = (2, 4)
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _signature(coords: list[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    r0 = min(r for r, _ in coords)
    c0 = min(c for _, c in coords)
    return frozenset((r - r0, c - c0) for r, c in coords)


def _touches_edge(coords: list[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords)


BLOCK_SIG = frozenset({(0, 0), (0, 1), (1, 0), (1, 1)})
ARROW_SIG_TO_DIR = {
    frozenset(map(tuple, np.argwhere(mask))): direction
    for direction, mask in ARROW_MASKS.items()
}


def _build_color_to_direction(grid: np.ndarray) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for color in map(int, np.unique(grid)):
        if color in (BLACK, MAROON):
            continue
        comps = _components(grid == color)
        big = [comp for comp in comps if len(comp) > 1]
        if len(big) != 2:
            continue

        swatches = [comp for comp in big if _signature(comp) == BLOCK_SIG]
        arrows = [comp for comp in big if _signature(comp) in ARROW_SIG_TO_DIR]
        if len(swatches) == 1 and len(arrows) == 1:
            mapping[color] = ARROW_SIG_TO_DIR[_signature(arrows[0])]

    assert mapping, "expected at least one color-to-direction entry"
    return mapping


def _room_specs(grid: np.ndarray, color_to_direction: dict[int, str]) -> list[tuple[int, int, int, int, str]]:
    specs: list[tuple[int, int, int, int, str]] = []
    for comp in _components(grid != MAROON):
        if _touches_edge(comp, grid.shape):
            continue
        seeds = [(r, c) for r, c in comp if int(grid[r, c]) not in (BLACK, MAROON)]
        if not seeds:
            continue
        assert len(seeds) == 1, "each enclosed region must contain exactly one seed"
        sr, sc = seeds[0]
        color = int(grid[sr, sc])
        assert color in color_to_direction, "seed color must appear in the corner code table"
        black_count = sum(int(grid[r, c]) == BLACK for r, c in comp)
        assert black_count > 0, "room interior must contain black pixels besides the seed"
        specs.append((sr, sc, color, black_count, color_to_direction[color]))

    assert specs, "expected enclosed seed rooms"
    return specs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the corner color-direction code table, measure each seed room's black area,
    and grow a seed-colored ray in the decoded direction with that exact length.
    """
    g = np.asarray(grid, dtype=int)
    color_to_direction = _build_color_to_direction(g)
    rooms = _room_specs(g, color_to_direction)

    out = g.copy()
    h, w = g.shape
    for sr, sc, color, length, direction in rooms:
        dr, dc = DIR_STEP[direction]
        for step in range(1, length + 1):
            rr = sr + dr * step
            cc = sc + dc * step
            assert 0 <= rr < h and 0 <= cc < w, "generator must keep rays in bounds"
            out[rr, cc] = color
    return out


def _draw_mask(grid: np.ndarray, r0: int, c0: int, mask: np.ndarray, color: int) -> None:
    window = grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]]
    window[mask] = int(color)


def _table_anchor(shape: tuple[int, int], corner: str) -> tuple[int, int]:
    h, w = shape
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, w - TABLE_WIDTH
    if corner == "bl":
        return h - TABLE_HEIGHT, 0
    return h - TABLE_HEIGHT, w - TABLE_WIDTH


def _paint_table(
    grid: np.ndarray,
    color_to_direction: dict[int, str],
    row_colors: list[int],
    corner: str,
) -> tuple[int, int, int, int]:
    base_r, base_c = _table_anchor(grid.shape, corner)
    for row_idx, color in enumerate(row_colors):
        row_r = base_r + row_idx * TABLE_ROW_STRIDE
        grid[row_r : row_r + 2, base_c : base_c + 2] = color
        _draw_mask(grid, row_r, base_c + TABLE_ARROW_COL, ARROW_MASKS[color_to_direction[color]], color)
    return (base_r, base_c, base_r + TABLE_HEIGHT - 1, base_c + TABLE_WIDTH - 1)


def _reserve_rect(blocked: np.ndarray, bbox: tuple[int, int, int, int], pad: int) -> None:
    h, w = blocked.shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _choose_seed_offset(inner_h: int, inner_w: int, direction: str) -> tuple[int, int]:
    if direction == UP:
        return inner_h - 1, random.randrange(inner_w)
    if direction == DOWN:
        return 0, random.randrange(inner_w)
    if direction == LEFT:
        return random.randrange(inner_h), inner_w - 1
    return random.randrange(inner_h), 0


def _candidate_range(
    size: int,
    outer: int,
    seed_offset: int,
    length: int,
    positive: bool,
) -> tuple[int, int] | None:
    low = 1
    high = size - outer - 1
    if positive:
        high = min(high, size - 2 - seed_offset - length)
    else:
        low = max(low, length - 1 - seed_offset)
    if low > high:
        return None
    return low, high


def _plan_room(
    grid: np.ndarray,
    blocked: np.ndarray,
    ray_blocked: np.ndarray,
    color: int,
    direction: str,
) -> dict[str, object] | None:
    h, w = grid.shape
    for _ in range(120):
        inner_h = random.randint(*INNER_HEIGHT_RANGE)
        inner_w = random.randint(*INNER_WIDTH_RANGE)
        length = inner_h * inner_w - 1
        seed_dr, seed_dc = _choose_seed_offset(inner_h, inner_w, direction)
        outer_h = inner_h + 2
        outer_w = inner_w + 2

        row_range = _candidate_range(
            h,
            outer_h,
            seed_dr,
            length,
            direction == DOWN,
        )
        col_range = _candidate_range(
            w,
            outer_w,
            seed_dc,
            length,
            direction == RIGHT,
        )
        if row_range is None or col_range is None:
            continue

        rows = list(range(row_range[0], row_range[1] + 1))
        cols = list(range(col_range[0], col_range[1] + 1))
        random.shuffle(rows)
        random.shuffle(cols)
        positions = [(r0, c0) for r0 in rows for c0 in cols]
        random.shuffle(positions)

        for r0, c0 in positions:
            r1 = r0 + outer_h - 1
            c1 = c0 + outer_w - 1
            if blocked[r0 : r1 + 1, c0 : c1 + 1].any():
                continue
            if np.any(grid[r0 : r1 + 1, c0 : c1 + 1] != BLACK):
                continue

            seed_r = r0 + 1 + seed_dr
            seed_c = c0 + 1 + seed_dc
            dr, dc = DIR_STEP[direction]
            ray = [(seed_r + dr * step, seed_c + dc * step) for step in range(1, length + 1)]
            if any(not (0 <= rr < h and 0 <= cc < w) for rr, cc in ray):
                continue

            bad = False
            for rr, cc in ray:
                inside_room = r0 <= rr <= r1 and c0 <= cc <= c1
                if ray_blocked[rr, cc]:
                    bad = True
                    break
                if blocked[rr, cc] and not inside_room:
                    bad = True
                    break
            if bad:
                continue

            return {
                "bbox": (r0, c0, r1, c1),
                "seed": (seed_r, seed_c),
                "color": color,
                "direction": direction,
                "length": length,
                "ray": ray,
            }
    return None


def _paint_room(grid: np.ndarray, room: dict[str, object]) -> None:
    r0, c0, r1, c1 = room["bbox"]
    grid[r0, c0 : c1 + 1] = MAROON
    grid[r1, c0 : c1 + 1] = MAROON
    grid[r0 : r1 + 1, c0] = MAROON
    grid[r0 : r1 + 1, c1] = MAROON
    sr, sc = room["seed"]
    grid[sr, sc] = int(room["color"])


def _expected_output(grid: np.ndarray, rooms: list[dict[str, object]]) -> np.ndarray:
    out = grid.copy()
    for room in rooms:
        for rr, cc in room["ray"]:
            out[rr, cc] = int(room["color"])
    return out


def _attempt() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    if h < TABLE_HEIGHT + 4 or w < TABLE_WIDTH + 8:
        return None

    grid = np.full((h, w), BLACK, dtype=int)
    table_colors = random.sample(PALETTE, 4)
    directions = random.sample(list(DIRECTIONS), 4)
    color_to_direction = dict(zip(table_colors, directions))
    row_colors = random.sample(table_colors, len(table_colors))
    table_corner = random.choice(TABLE_CORNERS)
    table_bbox = _paint_table(grid, color_to_direction, row_colors, table_corner)

    blocked = np.zeros((h, w), dtype=bool)
    ray_blocked = np.zeros((h, w), dtype=bool)
    _reserve_rect(blocked, table_bbox, pad=1)

    room_count = random.randint(*ROOM_COUNT_RANGE)
    room_colors = random.sample(table_colors, room_count)
    rooms: list[dict[str, object]] = []

    for color in room_colors:
        direction = color_to_direction[color]
        room = _plan_room(grid, blocked, ray_blocked, color, direction)
        if room is None:
            return None
        _paint_room(grid, room)
        _reserve_rect(blocked, room["bbox"], pad=1)
        for rr, cc in room["ray"]:
            ray_blocked[rr, cc] = True
        rooms.append(room)

    expected = _expected_output(grid, rooms)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
