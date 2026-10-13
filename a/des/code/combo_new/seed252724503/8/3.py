import random
from typing import Optional

import numpy as np


BG = 0
GRAY = 5
OBJECT = 7
MARKER = 9
CONTROL_COLORS = (1, 2, 3, 4)
SIDES = ("top", "right", "bottom", "left")
MARKER_MODES = (None, "up", "down", "left", "right")
LAYOUTS = ((2, 3), (3, 2), (3, 3))
GEN_RETRIES = 400
_GEN_COUNTER = 0


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _frame_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells = set()
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    return cells


def _find_frames(grid: np.ndarray, bg: int) -> list[tuple[int, int, int, int]]:
    mask = (grid != bg) & (grid != OBJECT) & (grid != MARKER)
    frames: list[tuple[int, int, int, int]] = []
    for comp in _components(mask):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 < 4 or c1 - c0 < 4:
            continue
        if set(comp) == _frame_cells(r0, c0, r1, c1):
            frames.append((r0, c0, r1, c1))
    assert frames, "expected at least one framed compartment"
    return sorted(frames)


def _instruction_side(grid: np.ndarray, frame: tuple[int, int, int, int]) -> str:
    r0, c0, r1, c1 = frame
    sides = {
        "top": grid[r0, c0 + 1 : c1],
        "bottom": grid[r1, c0 + 1 : c1],
        "left": grid[r0 + 1 : r1, c0],
        "right": grid[r0 + 1 : r1, c1],
    }
    hits = [name for name, arr in sides.items() if bool(np.any(arr != GRAY))]
    assert len(hits) == 1, "each frame must have exactly one colored border segment"
    return hits[0]


def _rotation_k(side: str) -> int:
    return {"top": 0, "right": 3, "bottom": 2, "left": 1}[side]


def _transform_inner(inner: np.ndarray, side: str, bg: int) -> np.ndarray:
    obj = np.argwhere(inner == OBJECT)
    assert obj.size > 0, "each frame must contain an orange object"
    r0, c0 = obj.min(axis=0)
    r1, c1 = obj.max(axis=0)
    sprite = inner[r0 : r1 + 1, c0 : c1 + 1] == OBJECT

    marker_coords = np.argwhere(inner == MARKER)
    assert marker_coords.shape[0] <= 1, "each frame may contain at most one marker"
    mirror = None
    if marker_coords.shape[0] == 1:
        mr, mc = map(int, marker_coords[0])
        if mr < int(r0) or mr > int(r1):
            mirror = "horizontal"
        elif mc < int(c0) or mc > int(c1):
            mirror = "vertical"
        else:
            raise AssertionError("marker must lie outside the object bbox")

    transformed = np.rot90(sprite, _rotation_k(side))
    if mirror == "horizontal":
        transformed = np.flipud(transformed)
    elif mirror == "vertical":
        transformed = np.fliplr(transformed)

    out = np.full(inner.shape, bg, dtype=int)
    h, w = transformed.shape
    top = (inner.shape[0] - h) // 2
    left = (inner.shape[1] - w) // 2
    out[top : top + h, left : left + w][transformed] = OBJECT
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each orange sprite by its frame-side cue, then optionally mirror it by the marker cue.

    Input structure:
    - The background is black.
    - Gray rectangular frames are arranged on a row/column grid, forming separate compartments of varying sizes.
    - Every frame has exactly one non-gray colored segment on one side of its border.
    - Inside each frame sits one orange sprite, plus an optional single maroon marker pixel.

    Rule:
    - Read the colored border segment by its side only: top = no rotation, right = clockwise 90 degrees,
      bottom = 180 degrees, left = counterclockwise 90 degrees.
    - If a maroon marker exists, apply a second transform after the rotation: markers above or below the
      sprite's bbox mean reflect across a horizontal axis, while markers left or right of the bbox mean
      reflect across a vertical axis.
    - Clear the original orange sprite and the marker, then place the transformed orange sprite centered
      back inside the same frame interior.

    Key insight:
    - The puzzle composes two independent instruction channels inside each compartment: border-side
      position chooses the rotation, and marker position chooses whether the rotated sprite is flipped.
    """
    g = np.asarray(grid, dtype=int)
    bg = _dominant_color(g)
    frames = _find_frames(g, bg)

    out = g.copy()
    for r0, c0, r1, c1 in frames:
        side = _instruction_side(g, (r0, c0, r1, c1))
        out[r0 + 1 : r1, c0 + 1 : c1] = _transform_inner(g[r0 + 1 : r1, c0 + 1 : c1], side, bg)
    return out


def _mask(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


BASE_SHAPES = (
    _mask(["11", "10"]),
    _mask(["110", "011"]),
    _mask(["100", "111"]),
    _mask(["111", "001"]),
    _mask(["110", "010", "011"]),
    _mask(["100", "110", "011"]),
    _mask(["1110", "0011"]),
    _mask(["1100", "0111"]),
    _mask(["1010", "1110"]),
)


def _normalize(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _random_shape(max_box: int) -> np.ndarray:
    candidates = [shape for shape in BASE_SHAPES if max(shape.shape) <= max_box]
    assert candidates, "need at least one shape candidate"
    shape = candidates[random.randrange(len(candidates))].copy()
    if random.random() < 0.5:
        shape = np.fliplr(shape)
    shape = np.rot90(shape, random.randrange(4))
    return _normalize(shape)


def _sample_sizes(count: int, max_total: int) -> Optional[list[int]]:
    for _ in range(200):
        sizes = [random.randint(6, 8) for _ in range(count)]
        if sum(sizes) <= max_total:
            if count == 1 or len(set(sizes)) > 1 or random.random() < 0.4:
                return sizes
            sizes[random.randrange(count)] += 1
            if sum(sizes) <= max_total:
                return sizes
    return None


def _frame_positions(n_rows: int, n_cols: int, gutter: int) -> Optional[tuple[list[int], list[int], list[int], list[int]]]:
    max_row_total = 30 - gutter * (n_rows - 1)
    max_col_total = 30 - gutter * (n_cols - 1)
    heights = _sample_sizes(n_rows, max_row_total)
    widths = _sample_sizes(n_cols, max_col_total)
    if heights is None or widths is None:
        return None

    row_span = sum(heights) + gutter * (n_rows - 1)
    col_span = sum(widths) + gutter * (n_cols - 1)
    top_margin = random.randint(0, 30 - row_span)
    left_margin = random.randint(0, 30 - col_span)

    rows: list[int] = []
    cols: list[int] = []
    cur = top_margin
    for h in heights:
        rows.append(cur)
        cur += h + gutter
    cur = left_margin
    for w in widths:
        cols.append(cur)
        cur += w + gutter
    return rows, cols, heights, widths


def _paint_frame(grid: np.ndarray, frame: tuple[int, int, int, int], side: str, color: int) -> None:
    r0, c0, r1, c1 = frame
    grid[r0 : r1 + 1, c0] = GRAY
    grid[r0 : r1 + 1, c1] = GRAY
    grid[r0, c0 : c1 + 1] = GRAY
    grid[r1, c0 : c1 + 1] = GRAY

    if side in ("top", "bottom"):
        available = c1 - c0 - 1
        seg_len = random.randint(2, available)
        start = random.randint(c0 + 1, c1 - seg_len)
        row = r0 if side == "top" else r1
        grid[row, start : start + seg_len] = color
    else:
        available = r1 - r0 - 1
        seg_len = random.randint(2, available)
        start = random.randint(r0 + 1, r1 - seg_len)
        col = c0 if side == "left" else c1
        grid[start : start + seg_len, col] = color


def _centered_bbox(inner_h: int, inner_w: int, mask: np.ndarray) -> tuple[int, int, int, int]:
    h, w = mask.shape
    top = (inner_h - h) // 2
    left = (inner_w - w) // 2
    return top, left, top + h - 1, left + w - 1


def _paint_object(inner: np.ndarray, shape: np.ndarray, marker_mode: Optional[str]) -> None:
    inner_h, inner_w = inner.shape
    r0, c0, r1, c1 = _centered_bbox(inner_h, inner_w, shape)
    inner[r0 : r1 + 1, c0 : c1 + 1][shape] = OBJECT

    if marker_mode == "up":
        inner[r0 - 1, random.randint(c0, c1)] = MARKER
    elif marker_mode == "down":
        inner[r1 + 1, random.randint(c0, c1)] = MARKER
    elif marker_mode == "left":
        inner[random.randint(r0, r1), c0 - 1] = MARKER
    elif marker_mode == "right":
        inner[random.randint(r0, r1), c1 + 1] = MARKER


def _build_expected(inp: np.ndarray, frames: list[tuple[int, int, int, int]], sides: dict[tuple[int, int, int, int], str]) -> np.ndarray:
    out = inp.copy()
    for frame in frames:
        r0, c0, r1, c1 = frame
        out[r0 + 1 : r1, c0 + 1 : c1] = _transform_inner(inp[r0 + 1 : r1, c0 + 1 : c1], sides[frame], BG)
    return out


def _attempt_generate(layout: tuple[int, int]) -> Optional[np.ndarray]:
    n_rows, n_cols = layout
    gutter = random.choice((1, 2))
    placed = _frame_positions(n_rows, n_cols, gutter)
    if placed is None:
        return None
    row_starts, col_starts, heights, widths = placed

    grid_h = row_starts[-1] + heights[-1]
    grid_w = col_starts[-1] + widths[-1]
    grid = np.full((grid_h, grid_w), BG, dtype=int)

    frames: list[tuple[int, int, int, int]] = []
    for i, r0 in enumerate(row_starts):
        for j, c0 in enumerate(col_starts):
            h = heights[i]
            w = widths[j]
            frames.append((r0, c0, r0 + h - 1, c0 + w - 1))

    sides_list = list(SIDES)
    while len(sides_list) < len(frames):
        sides_list.append(random.choice(SIDES))
    random.shuffle(sides_list)

    marker_list = list(MARKER_MODES)
    while len(marker_list) < len(frames):
        marker_list.append(random.choice(MARKER_MODES))
    random.shuffle(marker_list)

    side_map: dict[tuple[int, int, int, int], str] = {}
    changed_count = 0

    for idx, frame in enumerate(frames):
        side = sides_list[idx]
        marker_mode = marker_list[idx]
        side_map[frame] = side
        _paint_frame(grid, frame, side, random.choice(CONTROL_COLORS))

        r0, c0, r1, c1 = frame
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        limit = min(inner.shape[0] - 2, inner.shape[1] - 2)
        if limit < 2:
            return None
        shape = _random_shape(limit)
        _paint_object(inner, shape, marker_mode)

        transformed = _transform_inner(inner, side, BG)
        if not np.array_equal(inner == OBJECT, transformed == OBJECT):
            changed_count += 1

    if changed_count < 3:
        return None

    expected = _build_expected(grid, frames, side_map)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_COUNTER
    for attempt in range(GEN_RETRIES):
        layout = LAYOUTS[(_GEN_COUNTER + attempt) % len(LAYOUTS)]
        grid = _attempt_generate(layout)
        if grid is not None:
            _GEN_COUNTER += 1
            return grid
    raise RuntimeError("failed to generate a valid input")
