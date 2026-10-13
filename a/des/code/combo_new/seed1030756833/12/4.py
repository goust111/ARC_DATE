from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
WINDOW_COLORS = (2, 3, 4, 5, 6, 7, 8, 9)
SIDES = ("top", "bottom", "left", "right")
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_GENERATION_ATTEMPTS = 500


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((int(rr), int(cc)))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _rect_boundary(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    pts = {(r0, c) for c in range(c0, c1 + 1)}
    pts.update((r1, c) for c in range(c0, c1 + 1))
    pts.update((r, c0) for r in range(r0 + 1, r1))
    pts.update((r, c1) for r in range(r0 + 1, r1))
    return pts


def _find_rooms(g: np.ndarray) -> list[tuple[int, int, int, int, list[tuple[int, int, int]]]]:
    rooms: list[tuple[int, int, int, int, list[tuple[int, int, int]]]] = []
    for comp in _components(g != BLACK):
        rows = [r for r, _c in comp]
        cols = [c for _r, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue

        boundary = _rect_boundary(r0, c0, r1, c1)
        if set(comp) != boundary:
            continue
        if np.any(g[r0 + 1 : r1, c0 + 1 : c1] != BLACK):
            continue

        windows: list[tuple[int, int, int]] = []
        blue_count = 0
        for r, c in sorted(boundary):
            color = int(g[r, c])
            if color == BLUE:
                blue_count += 1
            elif color != BLACK:
                windows.append((r, c, color))
        if blue_count == 0 or not windows:
            continue
        rooms.append((r0, c0, r1, c1, windows))
    return sorted(rooms, key=lambda item: (item[0], item[1], item[2], item[3]))


def _paint_window(
    out: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    r: int,
    c: int,
    color: int,
) -> bool:
    if r == r0 and c0 < c < c1:
        out[r0 + 1 : r1, c] = color
        return True
    if r == r1 and c0 < c < c1:
        out[r0 + 1 : r1, c] = color
        return True
    if c == c0 and r0 < r < r1:
        out[r, c0 + 1 : c1] = color
        return True
    if c == c1 and r0 < r < r1:
        out[r, c0 + 1 : c1] = color
        return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint interior stripes from colored window pixels on blue rectangular room borders.

    Input structure:
    - The background is black.
    - Several separated rooms are one-pixel-thick blue rectangular frames.
    - Each frame has one to three single-pixel non-blue, non-black window markers on its border, and each room interior is black.

    Rule:
    - For every rectangular room, find all colored window pixels on its border.
    - Determine whether each window is on the top, bottom, left, or right side of the room.
    - Draw a one-pixel-wide stripe of the window color through the room interior, perpendicular to that side, all the way to the opposite inner wall.
    - When stripes cross, later windows in top-to-bottom, left-to-right scan order overwrite earlier stripe colors.

    Key insight:
    The colored border gaps are not damage; each window encodes the stripe color, starting side, and perpendicular direction into the enclosed black room.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    out = g.copy()

    for r0, c0, r1, c1, windows in _find_rooms(g):
        for r, c, color in sorted(windows, key=lambda item: (item[0], item[1])):
            _paint_window(out, r0, c0, r1, c1, r, c, color)

    return out


def _window_candidates(side: str, r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    if side == "top":
        return [(r0, c) for c in range(c0 + 1, c1)]
    if side == "bottom":
        return [(r1, c) for c in range(c0 + 1, c1)]
    if side == "left":
        return [(r, c0) for r in range(r0 + 1, r1)]
    if side == "right":
        return [(r, c1) for r in range(r0 + 1, r1)]
    raise ValueError(side)


def _choose_window_sides(count: int, rng: random.Random) -> list[str]:
    if count >= 2 and rng.random() < 0.65:
        sides = [rng.choice(("top", "bottom")), rng.choice(("left", "right"))]
        while len(sides) < count:
            sides.append(rng.choice(SIDES))
        rng.shuffle(sides)
        return sides
    return [rng.choice(SIDES) for _ in range(count)]


def _draw_room(
    grid: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    rng: random.Random,
) -> list[tuple[int, int, int]]:
    grid[r0, c0 : c1 + 1] = BLUE
    grid[r1, c0 : c1 + 1] = BLUE
    grid[r0 : r1 + 1, c0] = BLUE
    grid[r0 : r1 + 1, c1] = BLUE

    window_count = rng.randint(1, 3)
    sides = _choose_window_sides(window_count, rng)
    colors = rng.sample(list(WINDOW_COLORS), window_count)
    used: set[tuple[int, int]] = set()
    windows: list[tuple[int, int, int]] = []
    for side, color in zip(sides, colors):
        choices = [cell for cell in _window_candidates(side, r0, c0, r1, c1) if cell not in used]
        if not choices:
            raise ValueError("room side has no free window cells")
        r, c = rng.choice(choices)
        used.add((r, c))
        grid[r, c] = int(color)
        windows.append((r, c, int(color)))
    return windows


def _apply_specs(
    grid: np.ndarray,
    specs: list[tuple[int, int, int, int, list[tuple[int, int, int]]]],
) -> np.ndarray:
    out = grid.copy()
    for r0, c0, r1, c1, windows in specs:
        for r, c, color in sorted(windows, key=lambda item: (item[0], item[1])):
            _paint_window(out, r0, c0, r1, c1, r, c, color)
    return out


def _place_room(
    occupied: np.ndarray,
    outer_h: int,
    outer_w: int,
    rng: random.Random,
) -> tuple[int, int, int, int] | None:
    h, w = occupied.shape
    if outer_h > h - 2 or outer_w > w - 2:
        return None
    for _ in range(150):
        r0 = rng.randint(1, h - outer_h - 1)
        c0 = rng.randint(1, w - outer_w - 1)
        r1 = r0 + outer_h - 1
        c1 = c0 + outer_w - 1
        rr0, rr1 = max(0, r0 - 1), min(h, r1 + 2)
        cc0, cc1 = max(0, c0 - 1), min(w, c1 + 2)
        if not occupied[rr0:rr1, cc0:cc1].any():
            occupied[rr0:rr1, cc0:cc1] = True
            return r0, c0, r1, c1
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(9, 30)
    w = rng.randint(9, 30)
    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    max_rooms = min(5, max(1, (h * w) // 85))
    room_count = rng.randint(1, max_rooms)
    specs: list[tuple[int, int, int, int, list[tuple[int, int, int]]]] = []

    for _ in range(room_count):
        max_oh = min(10, h - 2)
        max_ow = min(10, w - 2)
        if max_oh < 5 or max_ow < 5:
            return None
        outer_h = rng.randint(5, max_oh)
        outer_w = rng.randint(5, max_ow)
        box = _place_room(occupied, outer_h, outer_w, rng)
        if box is None:
            return None
        r0, c0, r1, c1 = box
        try:
            windows = _draw_room(grid, r0, c0, r1, c1, rng)
        except ValueError:
            return None
        specs.append((r0, c0, r1, c1, windows))

    expected = _apply_specs(grid, specs)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(actual, grid):
        return None
    if max(grid.shape) > 30 or max(actual.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid window-stripe room input")
