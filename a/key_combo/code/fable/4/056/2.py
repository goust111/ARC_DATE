from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
PALETTE = tuple(range(3, 10))

GRID_MIN = 20
GRID_MAX = 28
ROOM_SIZE_MIN = 5
ROOM_SIZE_MAX = 9
ROOM_PAD = 2
GEN_ATTEMPTS = 500


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox_from_coords(coords),
                    "area": len(coords),
                }
            )
    return components


def _is_corner_bracket(component: dict) -> bool:
    coords = set(component["coords"])
    r0, c0, r1, c1 = component["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3 or component["area"] != h + w - 1:
        return False
    candidates = [
        {(r0, c) for c in range(c0, c1 + 1)} | {(r, c0) for r in range(r0, r1 + 1)},
        {(r0, c) for c in range(c0, c1 + 1)} | {(r, c1) for r in range(r0, r1 + 1)},
        {(r1, c) for c in range(c0, c1 + 1)} | {(r, c0) for r in range(r0, r1 + 1)},
        {(r1, c) for c in range(c0, c1 + 1)} | {(r, c1) for r in range(r0, r1 + 1)},
    ]
    return any(coords == candidate for candidate in candidates)


def _find_bracket_and_marker(grid: np.ndarray, bg: int) -> tuple[set[tuple[int, int]], tuple[int, int], int]:
    components = _find_monochrome_components(grid, bg)
    h, w = grid.shape
    brackets = []
    for component in components:
        if not _is_corner_bracket(component):
            continue
        r0, c0, r1, c1 = component["bbox"]
        touches_canvas_corner = (
            (r0 == 0 and c0 == 0)
            or (r0 == 0 and c1 == w - 1)
            or (r1 == h - 1 and c0 == 0)
            or (r1 == h - 1 and c1 == w - 1)
        )
        if touches_canvas_corner:
            brackets.append(component)
    assert len(brackets) == 1, "expected exactly one corner bracket"

    bracket = brackets[0]
    bracket_coords = set(bracket["coords"])
    r0, c0, r1, c1 = bracket["bbox"]
    markers = [
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if (r, c) not in bracket_coords and int(grid[r, c]) != bg
    ]
    assert len(markers) == 1, "expected exactly one marker inside the bracket"
    marker = markers[0]
    marker_color = int(grid[marker])
    assert marker_color != bg and marker_color != int(bracket["color"])
    return bracket_coords, marker, marker_color


def _exterior_background_mask(grid: np.ndarray, bg: int) -> np.ndarray:
    h, w = grid.shape
    exterior = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    def add(r: int, c: int) -> None:
        if int(grid[r, c]) == bg and not exterior[r, c]:
            exterior[r, c] = True
            queue.append((r, c))

    for r in range(h):
        add(r, 0)
        add(r, w - 1)
    for c in range(w):
        add(0, c)
        add(h - 1, c)

    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w:
                add(nr, nc)

    return exterior


def _erase_selected_color(grid: np.ndarray, bg: int, marker: tuple[int, int], marker_color: int) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    out[out == marker_color] = bg
    out[marker] = marker_color
    return out


def _fill_enclosed_background(grid: np.ndarray, bg: int) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    exterior = _exterior_background_mask(out, bg)
    enclosed = (out == bg) & ~exterior
    out[enclosed] = BLUE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase the wall color named by the bracket marker, then fill only the black regions that remain sealed afterward.

    Input structure:
    - The background is black, with red wall outlines forming one or more cavities.
    - Some wall cells are recolored into one or two non-red wall-segment colors.
    - Exactly one monochrome corner bracket sits in a canvas corner, and exactly one non-background marker pixel lies inside its bbox outside the bracket cells.

    Rule:
    - Detect the marker color inside the bracket and erase every other pixel of that color to black.
    - Treat every remaining non-black pixel as a wall and flood from the canvas border through black cells.
    - Recolor the black cells that are still unreachable from the border to blue, leaving the bracket, the marker, and all surviving wall cells unchanged.

    Key insight:
    - The marker does not directly name the regions to fill; it names wall segments to remove first, and enclosure must be recomputed on the modified wall layout.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert not np.any(g == BLUE), "blue is reserved for the output"

    bg = _most_common_color(g)
    bracket_coords, marker, marker_color = _find_bracket_and_marker(g, bg)
    assert marker not in bracket_coords

    erased = _erase_selected_color(g, bg, marker, marker_color)
    return _fill_enclosed_background(erased, bg)


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int) -> bool:
    ar0, ac0, ah, aw = a
    br0, bc0, bh, bw = b
    ar1, ac1 = ar0 + ah - 1, ac0 + aw - 1
    br1, bc1 = br0 + bh - 1, bc0 + bw - 1
    return not (ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0)


def _perimeter_cells(box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, h, w = box
    r1, c1 = r0 + h - 1, c0 + w - 1
    return (
        {(r0, c) for c in range(c0, c1 + 1)}
        | {(r1, c) for c in range(c0, c1 + 1)}
        | {(r, c0) for r in range(r0, r1 + 1)}
        | {(r, c1) for r in range(r0, r1 + 1)}
    )


def _room_corners(box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, h, w = box
    r1, c1 = r0 + h - 1, c0 + w - 1
    return {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}


def _side_cells(box: tuple[int, int, int, int], side: str, *, exclude_corners: bool = True) -> list[tuple[int, int]]:
    r0, c0, h, w = box
    r1, c1 = r0 + h - 1, c0 + w - 1
    if side == "top":
        cells = [(r0, c) for c in range(c0, c1 + 1)]
    elif side == "bottom":
        cells = [(r1, c) for c in range(c0, c1 + 1)]
    elif side == "left":
        cells = [(r, c0) for r in range(r0, r1 + 1)]
    else:
        cells = [(r, c1) for r in range(r0, r1 + 1)]
    if exclude_corners:
        corners = _room_corners(box)
        cells = [cell for cell in cells if cell not in corners]
    return cells


def _side_segment(box: tuple[int, int, int, int], side: str, length: int) -> list[tuple[int, int]] | None:
    cells = _side_cells(box, side, exclude_corners=True)
    if len(cells) < length:
        return None
    start = random.randint(0, len(cells) - length)
    return cells[start : start + length]


def _sample_box(
    height: int,
    width: int,
    forbidden: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    for _ in range(200):
        h = random.randint(ROOM_SIZE_MIN, ROOM_SIZE_MAX)
        w = random.randint(ROOM_SIZE_MIN, ROOM_SIZE_MAX)
        if h > height - 2 or w > width - 2:
            continue
        r0 = random.randint(1, height - h - 1)
        c0 = random.randint(1, width - w - 1)
        box = (r0, c0, h, w)
        if all(not _boxes_too_close(box, other, ROOM_PAD) for other in forbidden):
            return box
    return None


def _sample_chain_pair(
    height: int,
    width: int,
    forbidden: list[tuple[int, int, int, int]],
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], str] | None:
    for _ in range(250):
        attach = random.choice(("left", "right", "up", "down"))
        if attach in {"left", "right"}:
            h = random.randint(ROOM_SIZE_MIN, ROOM_SIZE_MAX)
            wa = random.randint(ROOM_SIZE_MIN, ROOM_SIZE_MAX)
            wb = random.randint(ROOM_SIZE_MIN, ROOM_SIZE_MAX)
            total_h, total_w = h, wa + wb - 1
            if total_h > height - 2 or total_w > width - 2:
                continue
            r0 = random.randint(1, height - total_h - 1)
            c0 = random.randint(1, width - total_w - 1)
            if attach == "right":
                a = (r0, c0, h, wa)
                b = (r0, c0 + wa - 1, h, wb)
            else:
                b = (r0, c0, h, wb)
                a = (r0, c0 + wb - 1, h, wa)
            combined = (r0, c0, total_h, total_w)
        else:
            w = random.randint(ROOM_SIZE_MIN, ROOM_SIZE_MAX)
            ha = random.randint(ROOM_SIZE_MIN, ROOM_SIZE_MAX)
            hb = random.randint(ROOM_SIZE_MIN, ROOM_SIZE_MAX)
            total_h, total_w = ha + hb - 1, w
            if total_h > height - 2 or total_w > width - 2:
                continue
            r0 = random.randint(1, height - total_h - 1)
            c0 = random.randint(1, width - total_w - 1)
            if attach == "down":
                a = (r0, c0, ha, w)
                b = (r0 + ha - 1, c0, hb, w)
            else:
                b = (r0, c0, hb, w)
                a = (r0 + hb - 1, c0, ha, w)
            combined = (r0, c0, total_h, total_w)
        if all(not _boxes_too_close(combined, other, ROOM_PAD) for other in forbidden):
            return a, b, attach
    return None


def _corner_bracket_spec(
    height: int,
    width: int,
    size: int,
    corner: str,
) -> tuple[set[tuple[int, int]], tuple[int, int], tuple[int, int, int, int]]:
    if corner == "tl":
        cells = {(size - 1, c) for c in range(size)} | {(r, size - 1) for r in range(size)}
        marker = (random.randint(1, size - 2), random.randint(1, size - 2))
        box = (0, 0, size, size)
    elif corner == "tr":
        c0 = width - size
        cells = {(size - 1, c) for c in range(c0, width)} | {(r, c0) for r in range(size)}
        marker = (random.randint(1, size - 2), random.randint(c0 + 1, width - 2))
        box = (0, c0, size, size)
    elif corner == "bl":
        r0 = height - size
        cells = {(r0, c) for c in range(size)} | {(r, size - 1) for r in range(r0, height)}
        marker = (random.randint(r0 + 1, height - 2), random.randint(1, size - 2))
        box = (r0, 0, size, size)
    else:
        r0 = height - size
        c0 = width - size
        cells = {(r0, c) for c in range(c0, width)} | {(r, c0) for r in range(r0, height)}
        marker = (random.randint(r0 + 1, height - 2), random.randint(c0 + 1, width - 2))
        box = (r0, c0, size, size)
    return cells, marker, box


def _enclosed_background_mask(grid: np.ndarray, bg: int) -> np.ndarray:
    exterior = _exterior_background_mask(grid, bg)
    return (grid == bg) & ~exterior


def _decorate_walls(
    wall_colors: dict[tuple[int, int], int],
    rooms: list[tuple[int, int, int, int]],
    forbidden_cells: set[tuple[int, int]],
    colors: list[int],
) -> None:
    if not colors:
        return
    eligible: list[tuple[int, int]] = []
    for room in rooms:
        corners = _room_corners(room)
        for cell in _perimeter_cells(room):
            if cell in corners or cell in forbidden_cells:
                continue
            if wall_colors.get(cell) == RED:
                eligible.append(cell)
    if len(eligible) < len(colors):
        return
    random.shuffle(eligible)
    total = random.randint(len(colors), min(len(eligible), len(colors) + 4))
    chosen = eligible[:total]
    for index, cell in enumerate(chosen):
        color = colors[index % len(colors)]
        wall_colors[cell] = color


def _build_candidate() -> np.ndarray | None:
    height = random.randint(GRID_MIN, GRID_MAX)
    width = random.randint(GRID_MIN, GRID_MAX)
    grid = np.zeros((height, width), dtype=int)

    bracket_color = random.choice(PALETTE)
    marker_color = random.choice([color for color in PALETTE if color != bracket_color])
    extra_wall_colors = []
    if random.random() < 0.65:
        extra_wall_colors.append(random.choice([color for color in PALETTE if color not in {bracket_color, marker_color}]))

    bracket_size = random.randint(4, 5)
    bracket_corner = random.choice(("tl", "tr", "bl", "br"))
    bracket_cells, marker_cell, bracket_box = _corner_bracket_spec(height, width, bracket_size, bracket_corner)

    forbidden_boxes = [bracket_box]
    chain_reaction = random.random() < 0.5

    if chain_reaction:
        chain = _sample_chain_pair(height, width, forbidden_boxes)
        if chain is None:
            return None
        leak_box, chained_box, attach = chain
        if attach == "right":
            shared_side, outer_side = "right", "left"
        elif attach == "left":
            shared_side, outer_side = "left", "right"
        elif attach == "down":
            shared_side, outer_side = "bottom", "top"
        else:
            shared_side, outer_side = "top", "bottom"
        rooms = [leak_box, chained_box]
        r0 = min(leak_box[0], chained_box[0])
        c0 = min(leak_box[1], chained_box[1])
        r1 = max(leak_box[0] + leak_box[2] - 1, chained_box[0] + chained_box[2] - 1)
        c1 = max(leak_box[1] + leak_box[3] - 1, chained_box[1] + chained_box[3] - 1)
        forbidden_boxes.append((r0, c0, r1 - r0 + 1, c1 - c0 + 1))
    else:
        leak_box = _sample_box(height, width, forbidden_boxes)
        if leak_box is None:
            return None
        chained_box = None
        shared_side = outer_side = ""
        rooms = [leak_box]
        forbidden_boxes.append(leak_box)

    keep_count = random.randint(1, 2)
    for _ in range(keep_count):
        keep_box = _sample_box(height, width, forbidden_boxes)
        if keep_box is None:
            return None
        rooms.append(keep_box)
        forbidden_boxes.append(keep_box)

    wall_colors: dict[tuple[int, int], int] = {}
    for room in rooms:
        for cell in _perimeter_cells(room):
            wall_colors[cell] = RED

    selected_cells: set[tuple[int, int]] = set()
    outer_segment = _side_segment(leak_box, outer_side or random.choice(("top", "bottom", "left", "right")), random.randint(1, 2))
    if outer_segment is None:
        return None
    selected_cells.update(outer_segment)

    if chained_box is not None:
        shared_cells = _side_segment(leak_box, shared_side, 1)
        if shared_cells is None:
            return None
        selected_cells.update(shared_cells)

    for cell in selected_cells:
        wall_colors[cell] = marker_color

    _decorate_walls(wall_colors, rooms, selected_cells, extra_wall_colors)

    for r, c in bracket_cells:
        if (r, c) in wall_colors:
            return None
        grid[r, c] = bracket_color
    if marker_cell in bracket_cells or marker_cell in wall_colors:
        return None
    grid[marker_cell] = marker_color

    for (r, c), color in wall_colors.items():
        if grid[r, c] != BLACK:
            return None
        grid[r, c] = color

    erased = _erase_selected_color(grid, BLACK, marker_cell, marker_color)
    before = _enclosed_background_mask(grid, BLACK)
    after = _enclosed_background_mask(erased, BLACK)
    if not np.any(after):
        return None
    if int(np.sum(before)) <= int(np.sum(after)):
        return None

    out = main(grid)
    if not np.any(out == BLUE):
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
