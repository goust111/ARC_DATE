from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
BROWN = 9

PALETTE_COLORS = (RED, GREEN, MAGENTA, ORANGE, AZURE, BROWN)
PALETTE_H = 3
PALETTE_W = 11
MAX_GENERATION_ATTEMPTS = 400
MAX_ROOM_PLACEMENT_TRIES = 150


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _border_positions(
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    include_corners: bool = True,
) -> list[tuple[int, int]]:
    pts: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        pts.append((r0, c))
        pts.append((r1, c))
    for r in range(r0 + 1, r1):
        pts.append((r, c0))
        pts.append((r, c1))
    if not include_corners:
        corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
        pts = [p for p in pts if p not in corners]
    return pts


def _find_palette(grid: np.ndarray) -> tuple[tuple[int, int, int, int], list[int]]:
    candidates: list[tuple[tuple[int, int, int, int], list[int]]] = []
    for comp in _components(grid == GREY):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 + 1 != PALETTE_H or c1 - c0 + 1 != PALETTE_W:
            continue
        if len(comp) != 2 * (PALETTE_H + PALETTE_W) - 4:
            continue
        if not np.all(grid[r0, c0 : c1 + 1] == GREY):
            continue
        if not np.all(grid[r1, c0 : c1 + 1] == GREY):
            continue
        if not np.all(grid[r0:r1 + 1, c0] == GREY):
            continue
        if not np.all(grid[r0:r1 + 1, c1] == GREY):
            continue

        inner = grid[r0 + 1, c0 + 1 : c1]
        if inner.shape[0] != 9:
            continue

        colors: list[int] = []
        ok = True
        for idx, value in enumerate(inner):
            value = int(value)
            if idx % 2 == 0:
                if value in (BLACK, GREY):
                    ok = False
                    break
                colors.append(value)
            elif value != BLACK:
                ok = False
                break
        if ok and len(colors) == 5:
            candidates.append(((r0, c0, r1, c1), colors))

    assert candidates, "expected a bottom palette frame"
    candidates.sort(key=lambda item: (item[0][2], item[0][1]))
    return candidates[-1]


def _room_bbox_if_valid(grid: np.ndarray, comp: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = _bbox(comp)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    expected = _border_positions(r0, c0, r1, c1, include_corners=True)
    if len(comp) != len(expected):
        return None

    expected_set = set(expected)
    if set(comp) != expected_set:
        return None

    for r, c in expected:
        if int(grid[r, c]) not in (BLUE, YELLOW):
            return None

    if r1 - r0 > 1 and c1 - c0 > 1:
        if not np.all(grid[r0 + 1 : r1, c0 + 1 : c1] == BLACK):
            return None
    return r0, c0, r1, c1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each closed room with the palette color indexed by its window count.

    The palette is a grey-framed strip near the bottom. Its five interior colors
    correspond to rooms whose blue/yellow rectangular borders contain 1..5 yellow
    window pixels.
    """
    g = np.asarray(grid, dtype=int)
    _palette_bbox, palette = _find_palette(g)

    out = g.copy()
    room_mask = (g == BLUE) | (g == YELLOW)
    for comp in _components(room_mask):
        bbox = _room_bbox_if_valid(g, comp)
        if bbox is None:
            continue
        r0, c0, r1, c1 = bbox
        window_count = sum(int(g[r, c] == YELLOW) for r, c in _border_positions(r0, c0, r1, c1))
        assert 1 <= window_count <= 5, "room window count must index the palette"
        fill_color = int(palette[window_count - 1])
        if r1 - r0 > 1 and c1 - c0 > 1:
            interior = out[r0 + 1 : r1, c0 + 1 : c1]
            interior[interior == BLACK] = fill_color
    return out


def _draw_palette(grid: np.ndarray, r0: int, c0: int, colors: list[int]) -> None:
    grid[r0 : r0 + PALETTE_H, c0 : c0 + PALETTE_W] = BLACK
    grid[r0, c0 : c0 + PALETTE_W] = GREY
    grid[r0 + PALETTE_H - 1, c0 : c0 + PALETTE_W] = GREY
    grid[r0 : r0 + PALETTE_H, c0] = GREY
    grid[r0 : r0 + PALETTE_H, c0 + PALETTE_W - 1] = GREY
    for idx, color in enumerate(colors):
        grid[r0 + 1, c0 + 1 + 2 * idx] = int(color)


def _reserve(occ: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1) -> None:
    h, w = occ.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    occ[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _can_place(occ: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1) -> bool:
    h, w = occ.shape
    if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    return not bool(occ[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _draw_room(
    grid: np.ndarray,
    room: tuple[int, int, int, int],
    window_count: int,
    rng: random.Random,
) -> None:
    r0, c0, r1, c1 = room
    border = _border_positions(r0, c0, r1, c1, include_corners=True)
    for r, c in border:
        grid[r, c] = BLUE

    allow_corners = rng.random() < 0.5
    window_positions = _border_positions(r0, c0, r1, c1, include_corners=allow_corners)
    chosen = rng.sample(window_positions, window_count)
    for r, c in chosen:
        grid[r, c] = YELLOW


def _manual_output(
    grid: np.ndarray,
    rooms: list[tuple[int, int, int, int, int]],
    palette_colors: list[int],
) -> np.ndarray:
    out = grid.copy()
    for r0, c0, r1, c1, window_count in rooms:
        out[r0 + 1 : r1, c0 + 1 : c1] = int(palette_colors[window_count - 1])
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(20, 30)
    w = rng.randint(20, 30)
    if w < PALETTE_W + 2 or h < PALETTE_H + 8:
        return None

    grid = np.full((h, w), BLACK, dtype=int)
    occ = np.zeros((h, w), dtype=bool)

    palette_r0 = h - PALETTE_H
    palette_c0 = rng.randint(0, w - PALETTE_W)
    palette_colors = rng.sample(list(PALETTE_COLORS), 5)
    _draw_palette(grid, palette_r0, palette_c0, palette_colors)
    _reserve(occ, palette_r0, palette_c0, palette_r0 + PALETTE_H - 1, palette_c0 + PALETTE_W - 1, pad=1)

    room_count = rng.choice((3, 4, 5))
    window_counts = rng.sample([1, 2, 3, 4, 5], room_count)
    specs: list[tuple[int, int, int, int]] = []
    for count in window_counts:
        min_side = 4 if count <= 3 else 5
        room_h = rng.randint(min_side, 9)
        room_w = rng.randint(min_side, 9)
        specs.append((room_h * room_w, count, room_h, room_w))
    specs.sort(reverse=True)

    rooms: list[tuple[int, int, int, int, int]] = []
    usable_bottom = palette_r0 - 2
    if usable_bottom < 4:
        return None

    for _area, count, room_h, room_w in specs:
        placed = False
        max_r0 = usable_bottom - room_h + 1
        max_c0 = w - room_w - 1
        if max_r0 < 1 or max_c0 < 1:
            return None
        for _ in range(MAX_ROOM_PLACEMENT_TRIES):
            r0 = rng.randint(1, max_r0)
            c0 = rng.randint(1, max_c0)
            r1 = r0 + room_h - 1
            c1 = c0 + room_w - 1
            if not _can_place(occ, r0, c0, r1, c1, pad=1):
                continue
            _reserve(occ, r0, c0, r1, c1, pad=1)
            _draw_room(grid, (r0, c0, r1, c1), count, rng)
            rooms.append((r0, c0, r1, c1, count))
            placed = True
            break
        if not placed:
            return None

    expected = _manual_output(grid, rooms, palette_colors)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
