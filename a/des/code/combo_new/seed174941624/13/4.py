from __future__ import annotations

import random

import numpy as np


BG = 0
WALL = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 300


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            components.append(comp)
    return components


def _room_boundary(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for c in range(c0 - 1, c1 + 2):
        cells.append((r0 - 1, c))
        cells.append((r1 + 1, c))
    for r in range(r0, r1 + 1):
        cells.append((r, c0 - 1))
        cells.append((r, c1 + 1))
    return cells


def _extract_rooms(grid: np.ndarray, wall: int = WALL) -> list[tuple[int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    rooms: list[tuple[int, int, int, int]] = []
    for comp in _components4(g != int(wall)):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if len(comp) != (r1 - r0 + 1) * (c1 - c0 + 1):
            continue
        boundary = _room_boundary(r0, c0, r1, c1)
        if not all(g[br, bc] == int(wall) for br, bc in boundary):
            continue
        rooms.append((int(r0), int(c0), int(r1), int(c1)))
    rooms.sort()
    return rooms


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill or clear each grey-walled room according to the parity of its interior colored-pixel count.

    Input structure:
    - The background is black.
    - Thin grey orthogonal walls partition part of the grid into closed rectangular rooms, with walls possibly shared by neighboring rooms.
    - Inside each room, the interior is otherwise black or contains scattered pixels of exactly one non-black, non-grey color.

    Rule:
    - Treat every non-grey 4-connected component that is fully enclosed by grey walls as one room interior.
    - Count the non-black, non-grey pixels inside that room.
    - If the count is odd, fill the entire room interior with that color and recolor the room's surrounding grey border to the same color.
    - If the count is positive and even, erase those interior colored pixels back to black while leaving the room otherwise unchanged.
    - Empty rooms stay unchanged.

    Key insight:
    - The important objects are the enclosed rooms induced by the grey wall network; each room is processed independently based on the parity of its own colored-pixel count, and odd rooms expand onto both the interior and the bordering wall ring.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    out = g.copy()
    wall_requests: dict[tuple[int, int], set[int]] = {}

    for r0, c0, r1, c1 in _extract_rooms(g, wall=WALL):
        interior = g[r0 : r1 + 1, c0 : c1 + 1]
        colored_mask = (interior != BG) & (interior != WALL)
        count = int(np.count_nonzero(colored_mask))
        if count == 0:
            continue

        colors = np.unique(interior[colored_mask])
        assert colors.size == 1, "each non-empty room must use exactly one interior color"
        color = int(colors[0])

        if count % 2 == 1:
            out[r0 : r1 + 1, c0 : c1 + 1] = color
            for cell in _room_boundary(r0, c0, r1, c1):
                wall_requests.setdefault(cell, set()).add(color)
        else:
            room_view = out[r0 : r1 + 1, c0 : c1 + 1]
            room_view[colored_mask] = BG

    for (r, c), colors in wall_requests.items():
        if len(colors) == 1:
            out[r, c] = next(iter(colors))

    return out


def _sample_segments(count: int, min_size: int, max_size: int) -> list[int]:
    return [random.randint(min_size, max_size) for _ in range(count)]


def _build_partition(
    room_heights: list[int],
    room_widths: list[int],
    top: int,
    left: int,
) -> tuple[list[int], list[int], list[tuple[int, int, int, int]]]:
    wall_rows = [int(top)]
    for size in room_heights:
        wall_rows.append(int(wall_rows[-1] + size + 1))
    wall_cols = [int(left)]
    for size in room_widths:
        wall_cols.append(int(wall_cols[-1] + size + 1))

    rooms: list[tuple[int, int, int, int]] = []
    for i, height in enumerate(room_heights):
        for j, width in enumerate(room_widths):
            r0 = wall_rows[i] + 1
            c0 = wall_cols[j] + 1
            rooms.append((int(r0), int(c0), int(r0 + height - 1), int(c0 + width - 1)))
    return wall_rows, wall_cols, rooms


def _random_layout() -> tuple[np.ndarray, list[tuple[int, int, int, int]]] | None:
    room_rows = random.randint(2, 4)
    room_cols = random.randint(2, 4)
    room_heights = _sample_segments(room_rows, 2, 6)
    room_widths = _sample_segments(room_cols, 2, 6)

    layout_h = sum(room_heights) + room_rows + 1
    layout_w = sum(room_widths) + room_cols + 1
    if layout_h > 28 or layout_w > 28:
        return None

    max_top = 30 - layout_h
    max_left = 30 - layout_w
    if max_top < 0 or max_left < 0:
        return None

    top = random.randint(0, min(2, max_top))
    left = random.randint(0, min(2, max_left))
    if max_top > 2 and random.random() < 0.5:
        top = random.randint(0, max_top)
    if max_left > 2 and random.random() < 0.5:
        left = random.randint(0, max_left)

    bottom_extra = random.randint(0, max_top - top)
    right_extra = random.randint(0, max_left - left)
    height = layout_h + top + bottom_extra
    width = layout_w + left + right_extra
    if height > 30 or width > 30:
        return None

    grid = np.full((height, width), BG, dtype=int)
    wall_rows, wall_cols, rooms = _build_partition(room_heights, room_widths, top, left)

    for r in wall_rows:
        grid[r, left : left + layout_w] = WALL
    for c in wall_cols:
        grid[top : top + layout_h, c] = WALL

    return grid, rooms


def _choose_count(area: int, parity: str) -> int:
    if parity == "odd":
        options = [n for n in range(1, area + 1) if n % 2 == 1]
    else:
        options = [n for n in range(2, area + 1) if n % 2 == 0]
    assert options, "room area must support requested parity"
    trimmed = [n for n in options if n < area]
    return random.choice(trimmed or options)


def _populate_rooms(grid: np.ndarray, rooms: list[tuple[int, int, int, int]]) -> tuple[np.ndarray, list[str]]:
    out = grid.copy()
    branches = ["empty", "odd", "even"]
    assignments = branches.copy()
    while len(assignments) < len(rooms):
        assignments.append(random.choice(branches))
    random.shuffle(assignments)

    odd_color = random.choice(PALETTE)
    even_palette = [color for color in PALETTE if color != odd_color]
    if not even_palette:
        even_palette = list(PALETTE)

    for (r0, c0, r1, c1), branch in zip(rooms, assignments):
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if branch == "empty":
            continue

        color = odd_color if branch == "odd" else random.choice(even_palette)
        count = _choose_count(area, branch)
        cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
        for r, c in random.sample(cells, count):
            out[r, c] = color

    return out, assignments


def _valid_generated_grid(grid: np.ndarray, assignments: list[str]) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    rooms = _extract_rooms(grid, wall=WALL)
    if len(rooms) != len(assignments):
        return False
    if set(assignments) != {"empty", "odd", "even"}:
        return False

    for r0, c0, r1, c1 in rooms:
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        colors = set(int(v) for v in np.unique(patch) if v not in {BG, WALL})
        if len(colors) > 1:
            return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        sampled = _random_layout()
        if sampled is None:
            continue
        base, rooms = sampled
        if len(rooms) < 3:
            continue
        grid, assignments = _populate_rooms(base, rooms)
        if not _valid_generated_grid(grid, assignments):
            continue
        return grid
    raise RuntimeError("failed to generate a valid room-parity puzzle")
