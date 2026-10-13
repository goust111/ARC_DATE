from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
WALL = 5
LOCK = 9
KEY_COLORS = (1, 2, 3, 4, 6, 7, 8)
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGHBORS8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(cells)
    return out


def _normalize(cells: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    cells = list(cells)
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _rot90(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    h = max(r for r, _ in cells) + 1
    return _normalize((c, h - 1 - r) for r, c in cells)


def _flip_h(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    w = max(c for _, c in cells) + 1
    return _normalize((r, w - 1 - c) for r, c in cells)


def _dihedral_variants(cells: Iterable[tuple[int, int]]) -> list[tuple[tuple[int, int], ...]]:
    base = _normalize(cells)
    out: list[tuple[tuple[int, int], ...]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    cur = base
    for _ in range(4):
        if cur not in seen:
            seen.add(cur)
            out.append(cur)
        cur = _rot90(cur)
    cur = _flip_h(base)
    for _ in range(4):
        if cur not in seen:
            seen.add(cur)
            out.append(cur)
        cur = _rot90(cur)
    return out


def _canonical_shape(cells: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    return min(_dihedral_variants(cells))


def _bbox(cells: Iterable[tuple[int, int]]) -> tuple[int, int]:
    cells = list(cells)
    return max(r for r, _ in cells) + 1, max(c for _, c in cells) + 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    In each wall-enclosed room, recolor maroon lockholes whose shape matches the
    room's colored key up to rotation/reflection, then erase the key.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    rooms = _components(g != WALL)
    assert rooms, "expected at least one room"

    for room in rooms:
        room_mask = np.zeros(g.shape, dtype=bool)
        for r, c in room:
            room_mask[r, c] = True

        key_components: list[tuple[int, list[tuple[int, int]]]] = []
        lock_components: list[list[tuple[int, int]]] = []

        used = np.zeros(g.shape, dtype=bool)
        for r, c in room:
            color = int(g[r, c])
            if color in (BG, WALL) or used[r, c]:
                continue
            comp_cells: list[tuple[int, int]] = []
            stack = [(r, c)]
            used[r, c] = True
            while stack:
                rr, cc = stack.pop()
                comp_cells.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if (
                        0 <= nr < g.shape[0]
                        and 0 <= nc < g.shape[1]
                        and room_mask[nr, nc]
                        and not used[nr, nc]
                        and int(g[nr, nc]) == color
                    ):
                        used[nr, nc] = True
                        stack.append((nr, nc))

            if color == LOCK:
                lock_components.append(comp_cells)
            else:
                key_components.append((color, comp_cells))

        assert len(key_components) == 1, "each room must contain exactly one key"
        key_color, key_cells = key_components[0]
        key_shape = _canonical_shape(key_cells)

        for lock_cells in lock_components:
            if _canonical_shape(lock_cells) == key_shape:
                for r, c in lock_cells:
                    out[r, c] = key_color

        for r, c in key_cells:
            out[r, c] = BG

    return out


def _random_connected_shape(size: int) -> tuple[tuple[int, int], ...]:
    cells = {(0, 0)}
    while len(cells) < size:
        frontier = []
        for r, c in cells:
            for dr, dc in NEIGHBORS4:
                pos = (r + dr, c + dc)
                if pos not in cells:
                    frontier.append(pos)
        cells.add(random.choice(frontier))
    return _normalize(cells)


def _random_shape_excluding(
    size_range: tuple[int, int],
    banned: set[tuple[tuple[int, int], ...]],
) -> tuple[tuple[int, int], ...]:
    for _ in range(500):
        shape = _random_connected_shape(random.randint(*size_range))
        if _canonical_shape(shape) not in banned:
            return shape
    raise AssertionError("failed to generate a distinct shape")


def _build_room_layout(
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    room = np.full((height, width), BG, dtype=int)
    expected = room.copy()

    key_shape = _random_connected_shape(random.randint(3, 6))
    key_sig = _canonical_shape(key_shape)
    key_color = random.choice(KEY_COLORS)

    n_match = random.randint(1, 2)
    n_nonmatch = random.randint(1, 2)

    objects: list[tuple[int, tuple[tuple[int, int], ...], int | None, bool]] = []
    objects.append((key_color, key_shape, None, True))

    key_variants = _dihedral_variants(key_shape)
    for _ in range(n_match):
        objects.append((LOCK, random.choice(key_variants), key_color, False))

    banned = {key_sig}
    for _ in range(n_nonmatch):
        shape = _random_shape_excluding((3, 6), banned)
        banned.add(_canonical_shape(shape))
        objects.append((LOCK, random.choice(_dihedral_variants(shape)), None, False))

    random.shuffle(objects)

    blocked = np.zeros((height, width), dtype=bool)
    for color, shape, opened_color, is_key in objects:
        sh, sw = _bbox(shape)
        placed = False
        for _ in range(300):
            r0 = random.randint(0, height - sh)
            c0 = random.randint(0, width - sw)
            cells = [(r0 + dr, c0 + dc) for dr, dc in shape]
            if any(blocked[r, c] for r, c in cells):
                continue
            for r, c in cells:
                room[r, c] = color
                expected[r, c] = color
                if opened_color is not None:
                    expected[r, c] = opened_color
            if is_key:
                for r, c in cells:
                    expected[r, c] = BG
            for r, c in cells:
                for dr, dc in NEIGHBORS8:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width:
                        blocked[nr, nc] = True
            placed = True
            break
        if not placed:
            raise AssertionError("room placement failed")

    return room, expected


def _split_lengths(parts: int, min_size: int, max_size: int) -> list[int]:
    return [random.randint(min_size, max_size) for _ in range(parts)]


def _make_puzzle() -> tuple[np.ndarray, np.ndarray]:
    n_rows = random.randint(2, 3)
    n_cols = random.randint(2, 3)
    room_heights = _split_lengths(n_rows, 8, 12)
    room_widths = _split_lengths(n_cols, 8, 12)

    h = sum(room_heights) + n_rows + 1
    w = sum(room_widths) + n_cols + 1
    grid = np.full((h, w), WALL, dtype=int)
    expected = grid.copy()

    r = 1
    for rh in room_heights:
        c = 1
        for cw in room_widths:
            room, room_expected = _build_room_layout(rh, cw)
            grid[r : r + rh, c : c + cw] = room
            expected[r : r + rh, c : c + cw] = room_expected
            c += cw + 1
        r += rh + 1

    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(500):
        try:
            grid, expected = _make_puzzle()
        except AssertionError:
            continue
        try:
            actual = main(grid)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate a valid sample")
