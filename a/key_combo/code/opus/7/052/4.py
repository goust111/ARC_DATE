from __future__ import annotations

import math
import random

import numpy as np


BG = 0
YELLOW = 4
VERTEX_COLORS = (1, 2, 3, 6, 7, 8, 9)
TOP_BAND_MAX_ROW = 4
GRID_SIDE_RANGE = (18, 23)
GROUP_COUNT_RANGE = (2, 3)
YELLOW_COUNT_RANGE = (4, 8)
PLACEMENT_RETRIES = 80
GENERATION_RETRIES = 400


def _angle_key(point: tuple[int, int], center_r: float, center_c: float) -> float:
    dr = point[0] - center_r
    dc = point[1] - center_c
    return math.atan2(dc, -dr) % (2 * math.pi)


def _ordered_vertices(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    center_r = sum(r for r, _ in points) / len(points)
    center_c = sum(c for _, c in points) / len(points)
    return sorted(points, key=lambda p: (_angle_key(p, center_r, center_c), p[0], p[1]))


def _aligned(a: tuple[int, int], b: tuple[int, int]) -> bool:
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    return (dr == 0 and dc != 0) or (dc == 0 and dr != 0) or (abs(dr) == abs(dc) and dr != 0)


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    if not _aligned(a, b):
        return []
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    step_r = 0 if dr == 0 else (1 if dr > 0 else -1)
    step_c = 0 if dc == 0 else (1 if dc > 0 else -1)
    steps = max(abs(dr), abs(dc))
    return [(a[0] + step_r * k, a[1] + step_c * k) for k in range(steps + 1)]


def _draw_outlines(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    vertex_colors = sorted(int(v) for v in np.unique(g) if int(v) not in (BG, YELLOW))

    for color in vertex_colors:
        vertices = [tuple(map(int, p)) for p in np.argwhere(g == color)]
        if len(vertices) < 2:
            continue
        ordered = _ordered_vertices(vertices)
        edges = list(zip(ordered, ordered[1:] + ordered[:1])) if len(ordered) > 2 else [(ordered[0], ordered[1])]
        for a, b in edges:
            cells = _segment_cells(a, b)
            if not cells:
                continue
            if any(g[r, c] != BG for r, c in cells[1:-1]):
                continue
            for r, c in cells:
                out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each colored vertex set into an outline, then let the yellow pixels fall onto that outline or the bottom.

    Input structure:
    - The background is black.
    - Each non-yellow foreground color marks one set of 3-5 single-pixel vertices in the lower half.
    - Several yellow single pixels sit in the upper rows, each in its own column.

    Rule:
    - For each non-yellow color, sort its vertices by polar angle around their centroid, using positive-up as angle zero and increasing clockwise.
    - Connect consecutive vertices, plus the last back to the first, but only for horizontal, vertical, or 45-degree diagonal segments whose interior cells are black in the input.
    - After all outlines are drawn, move every yellow pixel straight downward until the next cell in that column is non-black or the pixel reaches the bottom row.
    - Keep all original vertices and all drawn outline cells unchanged apart from clearing each yellow pixel's starting cell before repainting it at its stopping position.

    Key insight:
    - The constructed outline is both the geometric result of the vertex-ordering step and the terrain that stops the later column-wise gravity step.
    """
    g = np.asarray(grid, dtype=int)
    out = _draw_outlines(g)

    yellows = sorted((tuple(map(int, p)) for p in np.argwhere(g == YELLOW)), reverse=True)
    for r, c in yellows:
        out[r, c] = BG
        rr = r
        while rr + 1 < out.shape[0] and out[rr + 1, c] == BG:
            rr += 1
        out[rr, c] = YELLOW

    return out


def _transform_points(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    arr = np.array(points, dtype=int)
    if random.random() < 0.5:
        arr = arr[:, ::-1]
    if random.random() < 0.5:
        arr[:, 0] *= -1
    if random.random() < 0.5:
        arr[:, 1] *= -1
    arr -= arr.min(axis=0)
    return [tuple(map(int, p)) for p in arr]


def _triangle() -> list[tuple[int, int]]:
    side = random.randint(3, 6)
    return [(0, 0), (0, side), (side, side)]


def _rectangle() -> list[tuple[int, int]]:
    height = random.randint(3, 6)
    width = random.randint(4, 7)
    return [(0, 0), (0, width), (height, width), (height, 0)]


def _diamond() -> list[tuple[int, int]]:
    arm = random.randint(2, 4)
    return [(0, arm), (arm, 2 * arm), (2 * arm, arm), (arm, 0)]


def _house() -> list[tuple[int, int]]:
    roof = random.randint(2, 4)
    wall = random.randint(2, 4)
    bottom = roof + wall
    return [(0, roof), (roof, 2 * roof), (bottom, 2 * roof), (bottom, 0), (roof, 0)]


SHAPE_BUILDERS = (_triangle, _rectangle, _diamond, _house)


def _make_group_patch(color: int) -> tuple[np.ndarray, np.ndarray] | None:
    vertices = _transform_points(random.choice(SHAPE_BUILDERS)())
    height = max(r for r, _ in vertices) + 1
    width = max(c for _, c in vertices) + 1
    inp = np.zeros((height, width), dtype=int)
    for r, c in vertices:
        inp[r, c] = color
    out = _draw_outlines(inp)
    if np.count_nonzero(inp == color) < 3 or np.count_nonzero(inp == color) > 5:
        return None
    if np.count_nonzero(out == color) <= np.count_nonzero(inp == color):
        return None
    return inp, out


def _can_place(reserved: np.ndarray, top: int, left: int, height: int, width: int, pad: int = 1) -> bool:
    row0 = max(0, top - pad)
    col0 = max(0, left - pad)
    row1 = min(reserved.shape[0], top + height + pad)
    col1 = min(reserved.shape[1], left + width + pad)
    return not reserved[row0:row1, col0:col1].any()


def _paint_patch(dst: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    height, width = patch.shape
    view = dst[top : top + height, left : left + width]
    mask = patch != BG
    view[mask] = patch[mask]


def _place_patch(
    inp: np.ndarray,
    out: np.ndarray,
    reserved: np.ndarray,
    patch_in: np.ndarray,
    patch_out: np.ndarray,
) -> bool:
    grid_h, grid_w = inp.shape
    patch_h, patch_w = patch_in.shape
    if patch_h > grid_h - 5 or patch_w > grid_w:
        return False

    positions = [(r, c) for r in range(5, grid_h - patch_h + 1) for c in range(grid_w - patch_w + 1)]
    random.shuffle(positions)
    for top, left in positions[:PLACEMENT_RETRIES]:
        if not _can_place(reserved, top, left, patch_h, patch_w):
            continue
        _paint_patch(inp, patch_in, top, left)
        _paint_patch(out, patch_out, top, left)
        row0 = max(0, top - 1)
        col0 = max(0, left - 1)
        row1 = min(grid_h, top + patch_h + 1)
        col1 = min(grid_w, left + patch_w + 1)
        reserved[row0:row1, col0:col1] = True
        return True
    return False


def _outline_columns(grid: np.ndarray) -> dict[int, int]:
    cols: dict[int, int] = {}
    for c in range(grid.shape[1]):
        rows = np.flatnonzero(grid[:, c] != BG)
        if len(rows):
            cols[c] = int(rows[0])
    return cols


def _build_instance() -> np.ndarray | None:
    height = random.randint(*GRID_SIDE_RANGE)
    width = random.randint(*GRID_SIDE_RANGE)
    inp = np.zeros((height, width), dtype=int)
    base_out = np.zeros((height, width), dtype=int)
    reserved = np.zeros((height, width), dtype=bool)

    group_count = random.randint(*GROUP_COUNT_RANGE)
    group_colors = random.sample(VERTEX_COLORS, group_count)

    for color in group_colors:
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            patch = _make_group_patch(color)
            if patch is None:
                continue
            if _place_patch(inp, base_out, reserved, patch[0], patch[1]):
                placed = True
                break
        if not placed:
            return None

    column_to_first_outline = _outline_columns(base_out)
    candidate_cols = [c for c, first_row in column_to_first_outline.items() if first_row >= 5]
    if len(candidate_cols) < YELLOW_COUNT_RANGE[0]:
        return None

    yellow_count = random.randint(YELLOW_COUNT_RANGE[0], min(YELLOW_COUNT_RANGE[1], len(candidate_cols)))
    chosen_cols = random.sample(candidate_cols, yellow_count)
    moved_any = False

    for i, c in enumerate(chosen_cols):
        stop_row = column_to_first_outline[c] - 1
        if i == 0:
            start_row = random.randint(0, min(3, stop_row))
        else:
            start_row = random.randint(0, min(TOP_BAND_MAX_ROW, stop_row))
        inp[start_row, c] = YELLOW
        moved_any |= start_row != stop_row

    if not moved_any:
        return None

    out = main(inp)
    if out.shape != inp.shape or np.array_equal(out, inp):
        return None
    outline_input = inp.copy()
    outline_input[outline_input == YELLOW] = BG
    outline_only = out.copy()
    outline_only[outline_only == YELLOW] = BG
    if not np.array_equal(_draw_outlines(outline_input), outline_only):
        return None
    if len(set(np.flatnonzero(np.any(inp == YELLOW, axis=0)).tolist())) != yellow_count:
        return None
    if np.any(inp[5:, :] == YELLOW):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _build_instance()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
