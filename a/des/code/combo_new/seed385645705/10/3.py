from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
FRAME_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GRID_SIZE = 30
GEN_RETRY_LIMIT = 400
FRAME_COUNT_RANGE = (3, 5)
INNER_HEIGHT_RANGE = (3, 10)
INNER_WIDTH_RANGE = (3, 10)
EXTRA_CANVAS_RANGE = (0, 6)
CORNERS = ("tl", "tr", "bl", "br")
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _draw_frame(
    grid: np.ndarray, top: int, left: int, height: int, width: int, color: int
) -> None:
    grid[top, left : left + width] = color
    grid[top + height - 1, left : left + width] = color
    grid[top : top + height, left] = color
    grid[top : top + height, left + width - 1] = color


def _frame_cells(top: int, left: int, height: int, width: int) -> set[tuple[int, int]]:
    cells = set()
    bottom = top + height - 1
    right = left + width - 1
    for c in range(left, right + 1):
        cells.add((top, c))
        cells.add((bottom, c))
    for r in range(top, bottom + 1):
        cells.add((r, left))
        cells.add((r, right))
    return cells


def _side_cells(
    top: int, left: int, height: int, width: int, side: str
) -> set[tuple[int, int]]:
    bottom = top + height - 1
    right = left + width - 1
    if side == "top":
        return {(top, c) for c in range(left, right + 1)}
    if side == "bottom":
        return {(bottom, c) for c in range(left, right + 1)}
    if side == "left":
        return {(r, left) for r in range(top, bottom + 1)}
    if side == "right":
        return {(r, right) for r in range(top, bottom + 1)}
    raise ValueError(f"unknown side: {side}")


def _corner_cell(frame: dict, corner: str) -> tuple[int, int]:
    row = int(frame["r0"]) if corner[0] == "t" else int(frame["r1"])
    col = int(frame["c0"]) if corner[1] == "l" else int(frame["c1"])
    return row, col


def _opposite_corner(corner: str) -> str:
    return ("b" if corner[0] == "t" else "t") + ("r" if corner[1] == "l" else "l")


def _frame_spec(
    outer_top: int, outer_left: int, outer_h: int, outer_w: int, idx: int, color: int
) -> dict:
    inset = 2 * idx
    r0 = outer_top + inset
    c0 = outer_left + inset
    height = outer_h - 4 * idx
    width = outer_w - 4 * idx
    return {
        "color": int(color),
        "r0": int(r0),
        "c0": int(c0),
        "r1": int(r0 + height - 1),
        "c1": int(c0 + width - 1),
        "height": int(height),
        "width": int(width),
        "area": int(height * width),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the full concentric rectangular frames hidden by a grey occluder and redraw them without the occlusion.

    Input structure:
    - The background is black.
    - Three to five differently colored 1-pixel rectangular frames are concentric, with one black cell of gap between neighboring layers.
    - A single connected grey blob covers parts of the colored borders, so one frame color may appear as several disconnected visible segments.

    Rule:
    - Ignore black and grey, and treat each remaining color as one partially visible rectangular frame.
    - For each color, use the minimum and maximum visible row and column coordinates to recover the full rectangle that frame belongs to.
    - Sort the recovered rectangles from largest area to smallest, crop the output to the outermost rectangle's size, and redraw the complete frames from outer to inner on a black canvas.

    Key insight:
    - Grey pixels are pure occlusion: each frame's visible extremes still reveal its true bounding box, so the task is to complete the hidden rectangles and then recrop to the outer frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    items = []
    for color in np.unique(g):
        color = int(color)
        if color in (BLACK, GREY):
            continue
        coords = np.argwhere(g == color)
        assert coords.size, "each frame color must remain visible"
        r0, c0 = map(int, coords.min(axis=0))
        r1, c1 = map(int, coords.max(axis=0))
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)
        items.append(
            {
                "color": color,
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
                "height": height,
                "width": width,
                "area": int(height * width),
            }
        )

    assert 3 <= len(items) <= 5, "expected 3-5 frame colors"
    areas = [item["area"] for item in items]
    assert len(areas) == len(set(areas)), "nested frame sizes must be unique"
    items.sort(key=lambda item: int(item["area"]), reverse=True)

    outer = items[0]
    out = np.full((int(outer["height"]), int(outer["width"])), BLACK, dtype=int)
    base_r0 = int(outer["r0"])
    base_c0 = int(outer["c0"])
    for item in items:
        top = int(item["r0"]) - base_r0
        left = int(item["c0"]) - base_c0
        _draw_frame(
            out,
            top,
            left,
            int(item["height"]),
            int(item["width"]),
            int(item["color"]),
        )
    return out


def _render_frames(frames: list[dict], shape: tuple[int, int]) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    for frame in frames:
        _draw_frame(
            grid,
            int(frame["r0"]),
            int(frame["c0"]),
            int(frame["height"]),
            int(frame["width"]),
            int(frame["color"]),
        )
    return grid


def _protected_cells(frames: list[dict], protected_corner: str) -> set[tuple[int, int]]:
    row_side = "top" if protected_corner[0] == "t" else "bottom"
    col_side = "left" if protected_corner[1] == "l" else "right"
    cells: set[tuple[int, int]] = set()
    for frame in frames:
        cells |= _side_cells(
            int(frame["r0"]),
            int(frame["c0"]),
            int(frame["height"]),
            int(frame["width"]),
            row_side,
        )
        cells |= _side_cells(
            int(frame["r0"]),
            int(frame["c0"]),
            int(frame["height"]),
            int(frame["width"]),
            col_side,
        )
    return cells


def _base_blob(frames: list[dict], protected_corner: str) -> set[tuple[int, int]]:
    opposite = _opposite_corner(protected_corner)
    step_r = -1 if opposite[0] == "b" else 1
    step_c = -1 if opposite[1] == "r" else 1
    corners = [_corner_cell(frame, opposite) for frame in frames]
    blob = {corners[0], corners[-1]}
    for (r, c), (nr, nc) in zip(corners, corners[1:]):
        path = [
            (r, c),
            (r + step_r, c),
            (r + step_r, c + step_c),
            (r + 2 * step_r, c + step_c),
            (nr, nc),
        ]
        blob.update(path)
    return blob


def _growth_window(
    outer: dict, protected_corner: str, rows: int, cols: int
) -> tuple[int, int, int, int]:
    r0 = int(outer["r0"])
    c0 = int(outer["c0"])
    r1 = int(outer["r1"])
    c1 = int(outer["c1"])
    mid_r = (r0 + r1) // 2
    mid_c = (c0 + c1) // 2
    opposite = _opposite_corner(protected_corner)

    if opposite[0] == "b":
        row_lo = max(0, mid_r - 1)
        row_hi = min(rows - 1, r1 + 1)
    else:
        row_lo = max(0, r0 - 1)
        row_hi = min(rows - 1, mid_r + 1)

    if opposite[1] == "r":
        col_lo = max(0, mid_c - 1)
        col_hi = min(cols - 1, c1 + 1)
    else:
        col_lo = max(0, c0 - 1)
        col_hi = min(cols - 1, mid_c + 1)

    return row_lo, row_hi, col_lo, col_hi


def _grow_connected_blob(
    base_blob: set[tuple[int, int]],
    allowed: set[tuple[int, int]],
    min_extra: int = 2,
    max_extra: int = 12,
) -> set[tuple[int, int]]:
    candidates_left = len(allowed) - len(base_blob)
    assert candidates_left >= 0
    if candidates_left == 0:
        return set(base_blob)

    lo = min(min_extra, candidates_left)
    hi = min(max_extra, candidates_left)
    extra = random.randint(lo, hi) if hi >= lo else 0
    target = len(base_blob) + extra

    blob = set(base_blob)
    while len(blob) < target:
        frontier = set()
        for r, c in blob:
            for dr, dc in DIR4:
                rr = r + dr
                cc = c + dc
                cell = (rr, cc)
                if cell in allowed and cell not in blob:
                    frontier.add(cell)
        if not frontier:
            break
        blob.add(random.choice(tuple(frontier)))
    return blob


def _connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    seen = set()
    stack = [next(iter(cells))]
    while stack:
        cell = stack.pop()
        if cell in seen:
            continue
        seen.add(cell)
        r, c = cell
        for dr, dc in DIR4:
            nxt = (r + dr, c + dc)
            if nxt in cells and nxt not in seen:
                stack.append(nxt)
    return len(seen) == len(cells)


def _frame_is_partially_occluded(inp: np.ndarray, frame: dict) -> bool:
    visible = int(np.count_nonzero(inp == int(frame["color"])))
    total = len(
        _frame_cells(
            int(frame["r0"]),
            int(frame["c0"]),
            int(frame["height"]),
            int(frame["width"]),
        )
    )
    return 0 < visible < total


def _visible_extremes_match(inp: np.ndarray, frame: dict) -> bool:
    coords = np.argwhere(inp == int(frame["color"]))
    if coords.size == 0:
        return False
    r0, c0 = map(int, coords.min(axis=0))
    r1, c1 = map(int, coords.max(axis=0))
    return (
        r0 == int(frame["r0"])
        and c0 == int(frame["c0"])
        and r1 == int(frame["r1"])
        and c1 == int(frame["c1"])
    )


def _sample_canvas(n_frames: int) -> tuple[int, int, int, int, int, int] | None:
    max_inner_h = min(INNER_HEIGHT_RANGE[1], MAX_GRID_SIZE - 4 * (n_frames - 1))
    max_inner_w = min(INNER_WIDTH_RANGE[1], MAX_GRID_SIZE - 4 * (n_frames - 1))
    if max_inner_h < INNER_HEIGHT_RANGE[0] or max_inner_w < INNER_WIDTH_RANGE[0]:
        return None

    inner_h = random.randint(INNER_HEIGHT_RANGE[0], max_inner_h)
    inner_w = random.randint(INNER_WIDTH_RANGE[0], max_inner_w)
    outer_h = inner_h + 4 * (n_frames - 1)
    outer_w = inner_w + 4 * (n_frames - 1)

    extra_h = random.randint(EXTRA_CANVAS_RANGE[0], min(EXTRA_CANVAS_RANGE[1], MAX_GRID_SIZE - outer_h))
    extra_w = random.randint(EXTRA_CANVAS_RANGE[0], min(EXTRA_CANVAS_RANGE[1], MAX_GRID_SIZE - outer_w))
    top = random.randint(0, extra_h)
    left = random.randint(0, extra_w)
    return outer_h, outer_w, outer_h + extra_h, outer_w + extra_w, top, left


def _expected_output(frames: list[dict]) -> np.ndarray:
    outer = frames[0]
    out = np.full((int(outer["height"]), int(outer["width"])), BLACK, dtype=int)
    base_r0 = int(outer["r0"])
    base_c0 = int(outer["c0"])
    for frame in frames:
        _draw_frame(
            out,
            int(frame["r0"]) - base_r0,
            int(frame["c0"]) - base_c0,
            int(frame["height"]),
            int(frame["width"]),
            int(frame["color"]),
        )
    return out


def _gen_attempt() -> np.ndarray | None:
    n_frames = random.randint(*FRAME_COUNT_RANGE)
    sampled = _sample_canvas(n_frames)
    if sampled is None:
        return None
    outer_h, outer_w, rows, cols, outer_top, outer_left = sampled
    colors = random.sample(FRAME_COLORS, n_frames)
    frames = [
        _frame_spec(outer_top, outer_left, outer_h, outer_w, idx, color)
        for idx, color in enumerate(colors)
    ]
    protected_corner = random.choice(CORNERS)

    clean = _render_frames(frames, (rows, cols))
    expected = _expected_output(frames)
    protected = _protected_cells(frames, protected_corner)
    base_blob = _base_blob(frames, protected_corner)
    if base_blob & protected:
        return None

    window = _growth_window(frames[0], protected_corner, rows, cols)
    row_lo, row_hi, col_lo, col_hi = window
    allowed = {
        (r, c)
        for r in range(row_lo, row_hi + 1)
        for c in range(col_lo, col_hi + 1)
        if (r, c) not in protected
    }
    if not base_blob <= allowed:
        return None

    blob = _grow_connected_blob(base_blob, allowed)
    if not _connected(blob):
        return None

    inp = clean.copy()
    for r, c in blob:
        inp[r, c] = GREY

    if int(np.count_nonzero(inp == GREY)) != len(blob):
        return None
    if not all(_frame_is_partially_occluded(inp, frame) for frame in frames):
        return None
    if not all(_visible_extremes_match(inp, frame) for frame in frames):
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    return inp if np.array_equal(actual, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
