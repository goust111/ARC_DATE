import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

DIR_BY_ARROW = {
    RED: (-1, 0),
    BLUE: (1, 0),
    GREEN: (0, -1),
    YELLOW: (0, 1),
}
LOOKUP_COLORS = (6, 7, 8, 9)

GEN_RETRIES = 300
PLACEMENT_TRIES = 200


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
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


def _legend_mapping(grid: np.ndarray) -> dict[int, tuple[int, int]]:
    h, w = grid.shape
    best: dict[int, tuple[int, int]] = {}
    search_cols = min(w - 1, max(1, w // 3))
    for c in range(search_cols):
        mapping: dict[int, tuple[int, int]] = {}
        ok = True
        for r in range(h):
            key = int(grid[r, c])
            arrow = int(grid[r, c + 1])
            if key in (BG, GREY) or arrow not in DIR_BY_ARROW:
                continue
            direction = DIR_BY_ARROW[arrow]
            if key in mapping and mapping[key] != direction:
                ok = False
                break
            mapping[key] = direction
        if ok and len(mapping) > len(best):
            best = mapping
    if not best:
        raise AssertionError("failed to find the left-side legend")
    return best


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stretch each room's marker toward the wall using a legend-defined direction lookup.

    Input structure:
    - The background is black.
    - On the left side, a vertical legend is made of 1x2 entries: a lookup color pixel on the left and a direction-code pixel on the right.
    - The rest of the grid contains several grey rectangular enclosures whose interiors are black except for one colored marker pixel each.

    Rule:
    - Keep the legend and every grey border unchanged.
    - Decode the direction code colors as up, down, left, or right, and build a lookup table from legend color to direction.
    - For each enclosed room, find its unique interior marker color, look up that direction, and repaint black cells from the marker outward until the next grey border cell.

    Key insight:
    - A room's marker color is not its movement rule by itself; it is only a key that must be matched against the external legend before the directional stretch can be applied.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    mapping = _legend_mapping(g)

    out = g.copy()
    for comp in _components(g != GREY):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        markers = [(r, c, int(g[r, c])) for r, c in comp if int(g[r, c]) != BG]
        if len(markers) != 1:
            continue
        r, c, color = markers[0]
        if color not in mapping:
            continue
        dr, dc = mapping[color]
        rr, cc = r + dr, c + dc
        while 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) != GREY:
            if int(out[rr, cc]) == BG:
                out[rr, cc] = color
            rr += dr
            cc += dc
    return out


def _legend_rows(h: int, count: int) -> list[int]:
    for _ in range(100):
        rows = sorted(random.sample(range(1, h - 1), count))
        if all(b - a >= 2 for a, b in zip(rows, rows[1:])):
            return rows
    return sorted(random.sample(range(1, h - 1), count))


def _room_marker(r0: int, c0: int, inner_h: int, inner_w: int, direction: tuple[int, int]) -> tuple[int, int]:
    top = r0 + 1
    bottom = r0 + inner_h
    left = c0 + 1
    right = c0 + inner_w
    dr, dc = direction
    if dr == -1:
        return random.randint(top + 1, bottom), random.randint(left, right)
    if dr == 1:
        return random.randint(top, bottom - 1), random.randint(left, right)
    if dc == -1:
        return random.randint(top, bottom), random.randint(left + 1, right)
    return random.randint(top, bottom), random.randint(left, right - 1)


def _draw_room(
    grid: np.ndarray,
    r0: int,
    c0: int,
    inner_h: int,
    inner_w: int,
    color: int,
    direction: tuple[int, int],
) -> None:
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    grid[r0, c0 : c0 + outer_w] = GREY
    grid[r0 + outer_h - 1, c0 : c0 + outer_w] = GREY
    grid[r0 : r0 + outer_h, c0] = GREY
    grid[r0 : r0 + outer_h, c0 + outer_w - 1] = GREY
    mr, mc = _room_marker(r0, c0, inner_h, inner_w, direction)
    grid[mr, mc] = color


def _valid_grid(grid: np.ndarray, room_count: int) -> bool:
    try:
        mapping = _legend_mapping(grid)
    except AssertionError:
        return False
    if len(mapping) != room_count:
        return False
    enclosed = 0
    h, w = grid.shape
    for comp in _components(grid != GREY):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        markers = [int(grid[r, c]) for r, c in comp if int(grid[r, c]) != BG]
        if len(markers) == 1 and markers[0] in mapping:
            enclosed += 1
    if enclosed != room_count:
        return False
    return bool(np.any(main(grid) != grid))


def _gen_attempt() -> np.ndarray | None:
    room_count = random.choice((2, 3, 4))
    h = random.randint(max(14, room_count * 3 + 6), 30)
    w = random.randint(16, 30)

    grid = np.full((h, w), BG, dtype=int)
    legend_rows = _legend_rows(h, room_count)
    colors = random.sample(list(LOOKUP_COLORS), room_count)
    arrow_colors = random.sample(list(DIR_BY_ARROW), room_count)

    for row, color, arrow in zip(legend_rows, colors, arrow_colors):
        grid[row, 0] = color
        grid[row, 1] = arrow

    occupied = np.zeros((h, w), dtype=bool)
    occupied[:, :4] = True

    items = list(zip(colors, arrow_colors))
    random.shuffle(items)
    for color, arrow in items:
        direction = DIR_BY_ARROW[arrow]
        inner_h = random.randint(3, 6)
        inner_w = random.randint(3, 6)
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        if h - outer_h - 1 < 1 or w - outer_w - 1 < 4:
            return None

        placed = False
        for _ in range(PLACEMENT_TRIES):
            r0 = random.randint(1, h - outer_h - 1)
            c0 = random.randint(4, w - outer_w - 1)
            r1 = max(0, r0 - 1)
            r2 = min(h, r0 + outer_h + 1)
            c1 = max(0, c0 - 1)
            c2 = min(w, c0 + outer_w + 1)
            if occupied[r1:r2, c1:c2].any():
                continue
            occupied[r1:r2, c1:c2] = True
            _draw_room(grid, r0, c0, inner_h, inner_w, color, direction)
            placed = True
            break
        if not placed:
            return None

    if not _valid_grid(grid, room_count):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
