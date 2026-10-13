from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
RED = 2
GRAY = 5
PALETTE_POOL = (1, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 400
ARC_LIMIT = 30


def _neighbors4(r: int, c: int, height: int, width: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < height:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < width:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = map(int, mask.shape)
    seen = np.zeros((height, width), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                comp.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(int(cr), int(cc), height, width):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _touches_border(cells: list[tuple[int, int]], height: int, width: int) -> bool:
    return any(r == 0 or c == 0 or r == height - 1 or c == width - 1 for r, c in cells)


def _parse_palette_row(row: np.ndarray) -> tuple[list[int], list[int]] | None:
    positions = [int(c) for c, value in enumerate(row) if int(value) != BLACK]
    if not positions:
        return None

    colors = [int(row[c]) for c in positions]
    if any(color == GRAY for color in colors):
        return None
    if len(set(colors)) != len(colors):
        return None
    if any(positions[i + 1] - positions[i] != 2 for i in range(len(positions) - 1)):
        return None
    return positions, colors


def _find_palette(grid: np.ndarray) -> tuple[int, list[int]]:
    candidates: list[tuple[int, list[int]]] = []
    for row_idx in (0, int(grid.shape[0] - 1)):
        parsed = _parse_palette_row(grid[row_idx])
        if parsed is not None:
            _positions, colors = parsed
            candidates.append((int(row_idx), colors))

    assert candidates, "expected a palette row on the top or bottom edge"
    if len(candidates) == 1:
        return candidates[0]

    counts = [len(colors) for _, colors in candidates]
    best = max(counts)
    winners = [cand for cand in candidates if len(cand[1]) == best]
    assert len(winners) == 1, "palette row must be uniquely identifiable"
    return winners[0]


def _room_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = map(int, grid.shape)
    passable = grid != GRAY
    return [
        comp
        for comp in _components(passable)
        if not _touches_border(comp, height, width)
    ]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repaint each enclosed room with the palette color indexed by its red marker count.

    Input structure:
    - Black is the background floor color.
    - Gray wall pixels partition the grid into 4-connected regions, including border-touching outside space and enclosed rooms.
    - Each enclosed room contains only black floor plus one or more red marker pixels.
    - The top row or bottom row is a palette row: distinct single-pixel colors laid out left-to-right with one black gap between neighbors.

    Rule:
    - Detect the palette row and record its left-to-right colors as indices 1..N.
    - Find every non-wall 4-connected component that does not touch the grid border; each such component is one room.
    - Count the red markers inside a room, let that count be k, then replace every room cell (both black floor and red markers) with palette color k.
    - Keep the gray walls and the palette row unchanged.

    Key insight:
    - The puzzle combines two hidden structures: an edge palette whose order defines a lookup table, and enclosed rooms whose red-marker counts query that table.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    _palette_row, palette = _find_palette(g)
    out = g.copy()

    for room in _room_components(g):
        values = {int(g[r, c]) for r, c in room}
        assert values <= {BLACK, RED}, "room interiors must contain only black floor and red markers"
        marker_count = sum(int(g[r, c]) == RED for r, c in room)
        assert 1 <= marker_count <= len(palette), "room marker count must index the palette"
        fill_color = int(palette[marker_count - 1])
        for r, c in room:
            out[r, c] = fill_color

    return out


def _sample_marker_counts(room_count: int, palette_len: int) -> list[int]:
    distinct_count = random.randint(2, min(room_count, palette_len))
    distinct = random.sample(range(1, palette_len + 1), distinct_count)
    counts = list(distinct)
    while len(counts) < room_count:
        if random.random() < 0.5:
            counts.append(random.choice(distinct))
        else:
            counts.append(random.randint(1, palette_len))
    random.shuffle(counts)
    return counts


def _sample_room_size(marker_count: int) -> tuple[int, int]:
    for _ in range(100):
        inner_h = random.randint(2, 5)
        inner_w = random.randint(2, 6)
        if inner_h * inner_w >= marker_count + 1:
            return inner_h, inner_w
    raise RuntimeError("failed to sample a room with enough marker capacity")


def _split_shelves(count: int, shelf_count: int) -> list[int]:
    if shelf_count == 1:
        return [count]
    split = random.randint(1, count - 1)
    return [split, count - split]


def _layout_rooms(specs: list[dict[str, int]]):
    order = list(range(len(specs)))
    random.shuffle(order)

    shelf_count = random.randint(1, min(2, len(specs)))
    shelf_sizes = _split_shelves(len(specs), shelf_count)

    shelves: list[list[int]] = []
    cursor = 0
    for size in shelf_sizes:
        shelves.append(order[cursor : cursor + size])
        cursor += size

    room_positions: dict[int, tuple[int, int]] = {}
    layout_width = 0
    current_y = 0

    for shelf in shelves:
        shelf_height = max(specs[idx]["outer_h"] for idx in shelf)
        x = 0
        for pos, idx in enumerate(shelf):
            if pos > 0:
                x += random.randint(2, 4)
            y = current_y + random.randint(0, shelf_height - specs[idx]["outer_h"])
            room_positions[idx] = (y, x)
            x += specs[idx]["outer_w"]
        layout_width = max(layout_width, x)
        current_y += shelf_height
        if shelf is not shelves[-1]:
            current_y += random.randint(2, 4)

    layout_height = current_y
    return room_positions, layout_height, layout_width


def _palette_start(width: int, span: int) -> int:
    if width - span >= 2:
        return random.randint(1, width - span - 1)
    return random.randint(0, width - span)


def _place_markers(grid: np.ndarray, top: int, left: int, inner_h: int, inner_w: int, marker_count: int) -> None:
    cells = [(top + 1 + r, left + 1 + c) for r in range(inner_h) for c in range(inner_w)]
    for r, c in random.sample(cells, marker_count):
        grid[r, c] = RED


def _generated_marker_counts(grid: np.ndarray) -> list[int]:
    return sorted(sum(int(grid[r, c]) == RED for r, c in room) for room in _room_components(grid))


def _generate_attempt() -> np.ndarray | None:
    palette_side = random.choice(["top", "bottom"])
    palette_len = random.randint(2, 6)
    palette_colors = random.sample(PALETTE_POOL, palette_len)

    room_count = random.randint(2, 4)
    marker_counts = _sample_marker_counts(room_count, palette_len)

    specs: list[dict[str, int]] = []
    for marker_count in marker_counts:
        inner_h, inner_w = _sample_room_size(marker_count)
        specs.append(
            {
                "markers": int(marker_count),
                "inner_h": int(inner_h),
                "inner_w": int(inner_w),
                "outer_h": int(inner_h + 2),
                "outer_w": int(inner_w + 2),
            }
        )

    room_positions, layout_h, layout_w = _layout_rooms(specs)
    palette_gap = random.randint(1, 3)
    other_margin = random.randint(1, 3)

    if palette_side == "top":
        layout_y = 1 + palette_gap
        grid_h = 1 + palette_gap + layout_h + other_margin
    else:
        layout_y = other_margin
        grid_h = other_margin + layout_h + palette_gap + 1

    base_width = max(layout_w + 2, 2 * palette_len + 1)
    extra_width = random.randint(0, 4)
    grid_w = base_width + extra_width
    if grid_h > ARC_LIMIT or grid_w > ARC_LIMIT:
        return None

    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    span = 2 * palette_len - 1
    start = _palette_start(grid_w, span)
    palette_row = 0 if palette_side == "top" else grid_h - 1
    for i, color in enumerate(palette_colors):
        grid[palette_row, start + 2 * i] = int(color)

    x_shift = random.randint(1, grid_w - layout_w - 1)
    for idx, spec in enumerate(specs):
        y0, x0 = room_positions[idx]
        top = layout_y + y0
        left = x_shift + x0
        bottom = top + spec["outer_h"] - 1
        right = left + spec["outer_w"] - 1

        if bottom >= grid_h or right >= grid_w:
            return None

        grid[top : bottom + 1, left] = GRAY
        grid[top : bottom + 1, right] = GRAY
        grid[top, left : right + 1] = GRAY
        grid[bottom, left : right + 1] = GRAY
        _place_markers(grid, top, left, spec["inner_h"], spec["inner_w"], spec["markers"])

    try:
        palette_row_found, colors_found = _find_palette(grid)
        if palette_row_found != palette_row or colors_found != palette_colors:
            return None
        rooms = _room_components(grid)
        if len(rooms) != room_count:
            return None
        if _generated_marker_counts(grid) != sorted(marker_counts):
            return None
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    if out.shape != grid.shape or max(out.shape) > ARC_LIMIT:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
