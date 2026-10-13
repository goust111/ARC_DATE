from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
ORANGE = 7

OFFSETS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PLUS_OFFSETS = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]

ROOM_SIZE_OPTIONS = (4, 5)
GRID_SHAPES = ((2, 2), (2, 3), (3, 2), (3, 3))
MARGIN_OPTIONS = (1, 2, 3)
MAX_GENERATION_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                comp.append((cr, cc))
                for dr, dc in OFFSETS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _touches_border(coords: list[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords)


def _enclosed_chambers(grid: np.ndarray) -> list[set[tuple[int, int]]]:
    """Return enclosed 4-connected chambers over cells that are black or red."""
    passable = (grid == BLACK) | (grid == RED)
    chambers: list[set[tuple[int, int]]] = []
    for comp in _components(passable):
        if not _touches_border(comp, grid.shape):
            chambers.append(set(comp))
    return chambers


def _plus_centers_in_chamber(grid: np.ndarray, chamber: set[tuple[int, int]]) -> list[tuple[int, int]]:
    h, w = grid.shape
    centers: list[tuple[int, int]] = []
    for r, c in chamber:
        if grid[r, c] != RED:
            continue
        ok = True
        for dr, dc in OFFSETS4:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                ok = False
                break
            if grid[nr, nc] != RED or (nr, nc) not in chamber:
                ok = False
                break
        if ok:
            centers.append((r, c))
    return centers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill only those contour-enclosed chambers whose interiors contain a red plus marker.

    Input structure:
    - The grid uses black background, green contour walls, red 5-cell plus markers, and a few orange distractor pixels.
    - Green and orange cells act as barriers that partition the black/red interior into 4-connected chambers.
    - Some enclosed chambers contain one red orthogonal plus fully inside the chamber, while other enclosed chambers contain only black cells.

    Rule:
    - Find every 4-connected chamber of black/red cells that does not touch the grid border.
    - In each enclosed chamber, detect whether there is a red plus: one red center whose four orthogonal neighbors are also red and remain inside the same chamber.
    - If a chamber contains such a plus, erase that marker implicitly by recoloring the entire chamber green; otherwise leave the chamber unchanged.
    - Preserve all green contour pixels and all orange distractor pixels.

    Key insight:
    - The red plus is a fill permission token, not an output object: the task is chamber selection by local template match inside a contour-defined enclosure.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    for chamber in _enclosed_chambers(g):
        if _plus_centers_in_chamber(g, chamber):
            for r, c in chamber:
                out[r, c] = GREEN
    return out


def _draw_room_grid(
    room_rows: int,
    room_cols: int,
    row_heights: list[int],
    col_widths: list[int],
    top_margin: int,
    left_margin: int,
) -> tuple[np.ndarray, list[dict[str, tuple[int, int, int, int] | int]]]:
    wall_rows = [top_margin]
    for height in row_heights:
        wall_rows.append(wall_rows[-1] + height + 1)

    wall_cols = [left_margin]
    for width in col_widths:
        wall_cols.append(wall_cols[-1] + width + 1)

    height = wall_rows[-1] + top_margin + 1
    width = wall_cols[-1] + left_margin + 1
    grid = np.full((height, width), BLACK, dtype=int)

    for wr in wall_rows:
        grid[wr, wall_cols[0] : wall_cols[-1] + 1] = GREEN
    for wc in wall_cols:
        grid[wall_rows[0] : wall_rows[-1] + 1, wc] = GREEN

    rooms: list[dict[str, tuple[int, int, int, int] | int]] = []
    for rr in range(room_rows):
        for cc in range(room_cols):
            r0 = wall_rows[rr] + 1
            r1 = wall_rows[rr + 1] - 1
            c0 = wall_cols[cc] + 1
            c1 = wall_cols[cc + 1] - 1
            rooms.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "height": r1 - r0 + 1,
                    "width": c1 - c0 + 1,
                }
            )
    return grid, rooms


def _valid_room_layout(room_rows: int, room_cols: int) -> tuple[list[int], list[int]] | None:
    for _ in range(100):
        row_heights = [random.choice(ROOM_SIZE_OPTIONS) for _ in range(room_rows)]
        col_widths = [random.choice(ROOM_SIZE_OPTIONS) for _ in range(room_cols)]
        plus_candidates = sum(h == 5 for h in row_heights) * sum(w == 5 for w in col_widths)
        if plus_candidates >= 2:
            return row_heights, col_widths
    return None


def _pick_marked_rooms(rooms: list[dict[str, tuple[int, int, int, int] | int]]) -> set[int] | None:
    plus_candidates = [
        idx
        for idx, room in enumerate(rooms)
        if int(room["height"]) >= 5 and int(room["width"]) >= 5
    ]
    if len(plus_candidates) < 2 or len(rooms) - 2 < 2:
        return None

    max_marked = min(len(plus_candidates), len(rooms) - 2)
    marked_count = random.randint(2, max_marked)
    return set(random.sample(plus_candidates, marked_count))


def _place_plus(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    centers = [
        (r, c)
        for r in range(r0 + 2, r1 - 1)
        for c in range(c0 + 2, c1 - 1)
    ]
    assert centers, bbox
    center = random.choice(centers)
    for dr, dc in PLUS_OFFSETS:
        grid[center[0] + dr, center[1] + dc] = RED
    return center


def _room_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _scatter_orange(grid: np.ndarray, contour_bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = contour_bbox
    exterior = [
        (r, c)
        for r in range(grid.shape[0])
        for c in range(grid.shape[1])
        if grid[r, c] == BLACK and not (r0 <= r <= r1 and c0 <= c <= c1)
    ]
    if not exterior:
        return
    random.shuffle(exterior)
    orange_count = min(len(exterior), random.randint(1, 3))
    for r, c in exterior[:orange_count]:
        grid[r, c] = ORANGE


def _scatter_green_distractors(grid: np.ndarray, contour_bbox: tuple[int, int, int, int]) -> None:
    if random.random() < 0.45:
        return
    r0, c0, r1, c1 = contour_bbox
    candidates = [
        (r, c)
        for r in range(grid.shape[0])
        for c in range(grid.shape[1])
        if grid[r, c] == BLACK and not (r0 - 1 <= r <= r1 + 1 and c0 - 1 <= c <= c1 + 1)
    ]
    random.shuffle(candidates)
    for r, c in candidates[: random.randint(1, 3)]:
        grid[r, c] = GREEN


def _expected_output(inp: np.ndarray, rooms: list[dict[str, tuple[int, int, int, int] | int]], marked: set[int]) -> np.ndarray:
    out = inp.copy()
    for idx in marked:
        r0, c0, r1, c1 = rooms[idx]["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = GREEN
    return out


def _verify_generated(
    inp: np.ndarray,
    rooms: list[dict[str, tuple[int, int, int, int] | int]],
    marked: set[int],
    expected: np.ndarray,
) -> bool:
    if inp.ndim != 2 or max(inp.shape) > 30:
        return False

    out = main(inp)
    if not np.array_equal(out, expected):
        return False

    chambers = _enclosed_chambers(inp)
    if len(chambers) != len(rooms):
        return False
    if len(marked) < 2 or len(rooms) - len(marked) < 2:
        return False

    room_sets = [_room_cells(room["bbox"]) for room in rooms]
    chamber_sets = {frozenset(chamber) for chamber in chambers}
    if {frozenset(room_cells) for room_cells in room_sets} != chamber_sets:
        return False

    for idx, room in enumerate(rooms):
        r0, c0, r1, c1 = room["bbox"]
        room_in = inp[r0 : r1 + 1, c0 : c1 + 1]
        room_out = out[r0 : r1 + 1, c0 : c1 + 1]
        centers = _plus_centers_in_chamber(inp, room_sets[idx])
        if idx in marked:
            if len(centers) != 1:
                return False
            if np.any(room_out != GREEN):
                return False
            if np.any(room_out == RED):
                return False
        else:
            if centers:
                return False
            if not np.array_equal(room_in, room_out):
                return False

    if np.any((inp == ORANGE) != (out == ORANGE)):
        return False
    if np.any((inp == GREEN) & (out != GREEN)):
        return False
    return True


def _generate_once() -> np.ndarray | None:
    room_rows, room_cols = random.choice(GRID_SHAPES)
    layout = _valid_room_layout(room_rows, room_cols)
    if layout is None:
        return None
    row_heights, col_widths = layout
    top_margin = random.choice(MARGIN_OPTIONS)
    left_margin = random.choice(MARGIN_OPTIONS)
    grid, rooms = _draw_room_grid(room_rows, room_cols, row_heights, col_widths, top_margin, left_margin)

    marked = _pick_marked_rooms(rooms)
    if marked is None:
        return None

    for idx in marked:
        _place_plus(grid, rooms[idx]["bbox"])

    contour_bbox = (
        top_margin,
        left_margin,
        grid.shape[0] - top_margin - 1,
        grid.shape[1] - left_margin - 1,
    )
    _scatter_orange(grid, contour_bbox)
    _scatter_green_distractors(grid, contour_bbox)

    expected = _expected_output(grid, rooms, marked)
    if not _verify_generated(grid, rooms, marked, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_once()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid enclosed-plus puzzle")
