from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
ARROW_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_SIDE = 30
GEN_ATTEMPTS = 400

ARROW_PATTERNS: dict[str, tuple[tuple[int, int], ...]] = {
    "up": ((0, 0), (0, 2), (1, 1)),
    "down": ((0, 1), (1, 0), (1, 2)),
    "left": ((0, 0), (1, 1), (2, 0)),
    "right": ((0, 1), (1, 0), (2, 1)),
}


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
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(cells)
    return out


def _arrow_direction(cells: list[tuple[int, int]]) -> str:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0 = min(rows)
    c0 = min(cols)
    norm = {(r - r0, c - c0) for r, c in cells}
    for direction, pattern in ARROW_PATTERNS.items():
        if norm == set(pattern):
            return direction
    raise AssertionError("expected one 3-pixel V-shaped arrow per room")


def _wall_cells(bbox: tuple[int, int, int, int], direction: str) -> list[tuple[int, int]]:
    r0, r1, c0, c1 = bbox
    if direction == "up":
        return [(r0 - 1, c) for c in range(c0, c1 + 1)]
    if direction == "down":
        return [(r1 + 1, c) for c in range(c0, c1 + 1)]
    if direction == "left":
        return [(r, c0 - 1) for r in range(r0, r1 + 1)]
    if direction == "right":
        return [(r, c1 + 1) for r in range(r0, r1 + 1)]
    raise AssertionError(f"unknown direction: {direction}")


def _parse_room(grid: np.ndarray, cells: list[tuple[int, int]]) -> dict:
    h, w = grid.shape
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0 = min(rows)
    r1 = max(rows)
    c0 = min(cols)
    c1 = max(cols)
    assert 0 < r0 and r1 < h - 1 and 0 < c0 and c1 < w - 1
    assert len(cells) == (r1 - r0 + 1) * (c1 - c0 + 1), "room interior must be rectangular"

    assert np.all(grid[r0 - 1, c0 - 1 : c1 + 2] == GREY)
    assert np.all(grid[r1 + 1, c0 - 1 : c1 + 2] == GREY)
    assert np.all(grid[r0 - 1 : r1 + 2, c0 - 1] == GREY)
    assert np.all(grid[r0 - 1 : r1 + 2, c1 + 1] == GREY)

    arrow_cells = [(r, c) for r, c in cells if int(grid[r, c]) != BLACK]
    assert len(arrow_cells) == 3, "room must contain exactly one 3-pixel arrow"
    arrow_colors = {int(grid[r, c]) for r, c in arrow_cells}
    assert len(arrow_colors) == 1 and GREY not in arrow_colors
    direction = _arrow_direction(arrow_cells)
    return {
        "bbox": (r0, r1, c0, c1),
        "arrow_cells": arrow_cells,
        "direction": direction,
        "color": arrow_colors.pop(),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint the instructed wall of each rectangular room and erase the arrow.

    Input structure:
    - The background is black.
    - Grey pixels form one-pixel-thick closed rectangular room walls; rooms may be isolated or share full wall segments.
    - Each room interior is a solid black rectangle except for exactly one monochrome 3-pixel V-shaped arrow.

    Rule:
    - Find every enclosed non-grey interior component, treat its bounding box as one room, and read the arrow's color and pointing direction.
    - Recolor the wall segment on that side of the room to the arrow color.
    - Erase the arrow back to black and leave every other wall pixel unchanged.

    Key insight:
    - The tiny arrow carries two independent instructions at once: its color says what to paint, and its orientation says which wall to paint.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    rooms: list[dict] = []
    for cells in _components(g != GREY):
        if any(r == 0 or c == 0 or r == g.shape[0] - 1 or c == g.shape[1] - 1 for r, c in cells):
            continue
        rooms.append(_parse_room(g, cells))
    assert rooms, "expected at least one enclosed room"

    out = g.copy()
    for room in rooms:
        for r, c in room["arrow_cells"]:
            out[r, c] = BLACK
    for room in rooms:
        for r, c in _wall_cells(room["bbox"], room["direction"]):
            current = int(out[r, c])
            assert current in (GREY, room["color"]), "shared walls must not receive conflicting colors"
            out[r, c] = room["color"]
    return out


def _render_complex(row_heights: list[int], col_widths: list[int]) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    ph = sum(row_heights) + len(row_heights) + 1
    pw = sum(col_widths) + len(col_widths) + 1
    patch = np.full((ph, pw), GREY, dtype=int)
    rooms: list[tuple[int, int, int, int]] = []

    top = 1
    for height in row_heights:
        left = 1
        for width in col_widths:
            r0 = top
            r1 = top + height - 1
            c0 = left
            c1 = left + width - 1
            patch[r0 : r1 + 1, c0 : c1 + 1] = BLACK
            rooms.append((r0, r1, c0, c1))
            left = c1 + 2
        top = r1 + 2

    return patch, rooms


def _sample_complex(mode: str) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    if mode == "standalone":
        rows = [random.randint(3, 8)]
        cols = [random.randint(3, 8)]
        return _render_complex(rows, cols)

    shared_shapes = [(1, 2), (2, 1), (1, 3), (3, 1), (2, 2), (2, 3), (3, 2), (3, 3)]
    n_rows, n_cols = random.choice(shared_shapes)
    if n_rows == 3 and n_cols == 3:
        lo, hi = 3, 4
    elif n_rows * n_cols >= 5:
        lo, hi = 3, 5
    else:
        lo, hi = 3, 6
    row_heights = [random.randint(lo, hi) for _ in range(n_rows)]
    col_widths = [random.randint(lo, hi) for _ in range(n_cols)]
    return _render_complex(row_heights, col_widths)


def _place_patch(blocked: np.ndarray, patch: np.ndarray) -> tuple[int, int] | None:
    ph, pw = patch.shape
    h, w = blocked.shape
    candidates: list[tuple[int, int]] = []
    for top in range(1, h - ph):
        for left in range(1, w - pw):
            if blocked[top - 1 : top + ph + 1, left - 1 : left + pw + 1].any():
                continue
            candidates.append((top, left))
    if not candidates:
        return None
    return random.choice(candidates)


def _sample_directions(rooms: list[dict]) -> bool:
    for _ in range(200):
        used: set[tuple[int, int]] = set()
        ok = True
        for room in rooms:
            room["direction"] = random.choice(tuple(ARROW_PATTERNS))
        for room in rooms:
            for cell in _wall_cells(room["bbox"], room["direction"]):
                if cell in used:
                    ok = False
                    break
                used.add(cell)
            if not ok:
                break
        if ok:
            return True
    return False


def _sample_arrow_cells(bbox: tuple[int, int, int, int], direction: str) -> list[tuple[int, int]]:
    r0, r1, c0, c1 = bbox
    pattern = ARROW_PATTERNS[direction]
    ph = 1 + max(r for r, _ in pattern)
    pw = 1 + max(c for _, c in pattern)
    top = random.randint(r0, r1 - ph + 1)
    left = random.randint(c0, c1 - pw + 1)
    return [(top + dr, left + dc) for dr, dc in pattern]


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    master = np.full((MAX_SIDE, MAX_SIDE), BLACK, dtype=int)
    blocked = np.zeros_like(master, dtype=bool)
    room_specs: list[dict] = []

    layout_mode = random.choice(("shared", "mixed", "separate"))
    complex_modes: list[str]
    if layout_mode == "shared":
        complex_modes = ["shared"]
    elif layout_mode == "mixed":
        complex_modes = ["shared"] + ["standalone"] * random.randint(1, 2)
    else:
        complex_modes = ["standalone"] * random.randint(2, 4)

    for mode in complex_modes:
        patch, local_rooms = _sample_complex(mode)
        placed = _place_patch(blocked, patch)
        if placed is None:
            return None
        top, left = placed
        ph, pw = patch.shape
        master[top : top + ph, left : left + pw] = patch
        blocked[top - 1 : top + ph + 1, left - 1 : left + pw + 1] = True
        for r0, r1, c0, c1 in local_rooms:
            room_specs.append({"bbox": (top + r0, top + r1, left + c0, left + c1)})

    if not room_specs or not _sample_directions(room_specs):
        return None

    inp = master.copy()
    out = master.copy()
    for room in room_specs:
        room["color"] = random.choice(ARROW_COLORS)
        room["arrow_cells"] = _sample_arrow_cells(room["bbox"], room["direction"])
        for r, c in room["arrow_cells"]:
            inp[r, c] = room["color"]
        for r, c in _wall_cells(room["bbox"], room["direction"]):
            out[r, c] = room["color"]

    non_black = np.argwhere(inp != BLACK)
    if non_black.size == 0:
        return None

    r0, c0 = non_black.min(axis=0)
    r1, c1 = non_black.max(axis=0)
    top_margin = random.randint(1, min(3, int(r0)))
    left_margin = random.randint(1, min(3, int(c0)))
    bottom_margin = random.randint(1, min(3, int(MAX_SIDE - 1 - r1)))
    right_margin = random.randint(1, min(3, int(MAX_SIDE - 1 - c1)))

    rs = slice(int(r0 - top_margin), int(r1 + bottom_margin + 1))
    cs = slice(int(c0 - left_margin), int(c1 + right_margin + 1))
    cropped_inp = inp[rs, cs].copy()
    cropped_out = out[rs, cs].copy()
    if max(cropped_inp.shape) > MAX_SIDE or max(cropped_out.shape) > MAX_SIDE:
        return None
    return cropped_inp, cropped_out


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        built = _build_case()
        if built is None:
            continue
        inp, expected = built
        try:
            actual = main(inp)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid room-arrow puzzle")
