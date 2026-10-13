import numpy as np


BG = 0
GREY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400

ROW_TEMPLATES = [
    np.array([[1, 1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1, 1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
]
COL_TEMPLATES = [template.T.copy() for template in ROW_TEMPLATES]


def _most_common_color(values: np.ndarray) -> int:
    values = np.asarray(values, dtype=int)
    values = values[values >= 0]
    if values.size == 0:
        raise ValueError("expected at least one non-negative color")
    return int(np.bincount(values).argmax())


def _nonzero_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid != BG)
    if coords.size == 0:
        return 0, 0, -1, -1
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _candidate_wall_lines(sub: np.ndarray) -> tuple[list[int], list[int]]:
    wall_mask = (sub != BG) & (sub != GREY)
    h, w = sub.shape
    row_lines = [r for r in range(h) if int(np.count_nonzero(wall_mask[r])) >= w - 1]
    col_lines = [c for c in range(w) if int(np.count_nonzero(wall_mask[:, c])) >= h - 1]
    return row_lines, col_lines


def _room_color(sub: np.ndarray, r0: int, r1: int, c0: int, c1: int, mode: str) -> int:
    samples: list[np.ndarray] = []
    if mode == "row_strip":
        samples.append(sub[r0, c0 + 1 : c1])
        samples.append(sub[r1, c0 + 1 : c1])
    elif mode == "col_strip":
        samples.append(sub[r0 + 1 : r1, c0])
        samples.append(sub[r0 + 1 : r1, c1])
    else:
        samples.append(sub[r0, c0 + 1 : c1])
        samples.append(sub[r1, c0 + 1 : c1])
        samples.append(sub[r0 + 1 : r1, c0])
        samples.append(sub[r0 + 1 : r1, c1])
    values = np.concatenate([arr.ravel() for arr in samples if arr.size > 0])
    values = values[(values != BG) & (values != GREY)]
    if values.size == 0:
        edge = sub[r0 : r1 + 1, c0 : c1 + 1]
        values = edge[(edge != BG) & (edge != GREY)]
    return _most_common_color(values)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each room's grey furniture onto shared door walls using the neighboring room colors.

    Input structure:
    - The background is black.
    - One-pixel-thick colored horizontal and vertical wall segments form a floor plan of rectangular rooms.
    - Every shared wall between two adjacent rooms has exactly one one-pixel door opening.
    - Each room contains one grey connected furniture pattern fully inside the room.

    Rule:
    - Detect the wall-line rows and columns that define the rectangular room grid.
    - For every room, recover its furniture footprint and its visible wall color.
    - For every horizontal shared wall, project the upper room's furniture columns and the lower room's furniture columns onto the door row, coloring each projection with that room's wall color.
    - For every vertical shared wall, project the left room's furniture rows and the right room's furniture rows onto the door column, again using the corresponding room wall colors.
    - Keep the grey furniture and all unchanged walls as they are.

    Key insight:
    The doors identify which two rooms interact, and each room contributes only a one-dimensional projection of its furniture footprint, colored by the room's own walls rather than by the grey furniture itself.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    br0, bc0, br1, bc1 = _nonzero_bbox(g)
    if br1 < br0 or not np.any(g == GREY):
        return g.copy()

    sub = g[br0 : br1 + 1, bc0 : bc1 + 1]
    row_lines, col_lines = _candidate_wall_lines(sub)
    assert len(row_lines) >= 2 and len(col_lines) >= 2

    if len(row_lines) == 2 and len(col_lines) > 2:
        mode = "row_strip"
    elif len(col_lines) == 2 and len(row_lines) > 2:
        mode = "col_strip"
    else:
        mode = "generic"

    rooms: dict[tuple[int, int], dict[str, object]] = {}
    for ri in range(len(row_lines) - 1):
        for ci in range(len(col_lines) - 1):
            r0, r1 = row_lines[ri], row_lines[ri + 1]
            c0, c1 = col_lines[ci], col_lines[ci + 1]
            if r1 - r0 <= 1 or c1 - c0 <= 1:
                continue
            interior = sub[r0 + 1 : r1, c0 + 1 : c1]
            furniture = np.argwhere(interior == GREY)
            if furniture.size == 0:
                continue
            rooms[(ri, ci)] = {
                "color": _room_color(sub, r0, r1, c0, c1, mode),
                "rows": sorted({int(pt[0] + r0 + 1) for pt in furniture}),
                "cols": sorted({int(pt[1] + c0 + 1) for pt in furniture}),
            }

    out = g.copy()

    for ri in range(len(row_lines) - 1):
        for ci in range(len(col_lines) - 2):
            left = rooms.get((ri, ci))
            right = rooms.get((ri, ci + 1))
            if left is None or right is None:
                continue
            wall_col = col_lines[ci + 1]
            span_r0 = row_lines[ri] + 1
            span_r1 = row_lines[ri + 1]
            door_rows = [r for r in range(span_r0, span_r1) if sub[r, wall_col] == BG]
            assert len(door_rows) == 1
            for r in left["rows"]:
                out[br0 + int(r), bc0 + wall_col] = int(left["color"])
            for r in right["rows"]:
                out[br0 + int(r), bc0 + wall_col] = int(right["color"])

    for ri in range(len(row_lines) - 2):
        for ci in range(len(col_lines) - 1):
            top = rooms.get((ri, ci))
            bottom = rooms.get((ri + 1, ci))
            if top is None or bottom is None:
                continue
            wall_row = row_lines[ri + 1]
            span_c0 = col_lines[ci] + 1
            span_c1 = col_lines[ci + 1]
            door_cols = [c for c in range(span_c0, span_c1) if sub[wall_row, c] == BG]
            assert len(door_cols) == 1
            for c in top["cols"]:
                out[br0 + wall_row, bc0 + int(c)] = int(top["color"])
            for c in bottom["cols"]:
                out[br0 + wall_row, bc0 + int(c)] = int(bottom["color"])

    return out


def _sample_template(templates: list[np.ndarray], h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    valid = [template for template in templates if template.shape[0] <= h and template.shape[1] <= w]
    assert valid
    template = valid[int(rng.integers(len(valid)))].copy()
    if rng.random() < 0.5:
        template = np.fliplr(template)
    if rng.random() < 0.5:
        template = np.flipud(template)
    return template


def _row_furniture(room_h: int, room_w: int, band: str, rng: np.random.Generator) -> np.ndarray:
    template = _sample_template(ROW_TEMPLATES, 2, room_w, rng)
    mask = np.zeros((room_h, room_w), dtype=bool)
    row0 = 0 if band == "top" else room_h - template.shape[0]
    col0 = int(rng.integers(0, room_w - template.shape[1] + 1))
    mask[row0 : row0 + template.shape[0], col0 : col0 + template.shape[1]] = template
    return mask


def _col_furniture(room_h: int, room_w: int, band: str, rng: np.random.Generator) -> np.ndarray:
    template = _sample_template(COL_TEMPLATES, room_h, 2, rng)
    mask = np.zeros((room_h, room_w), dtype=bool)
    row0 = int(rng.integers(0, room_h - template.shape[0] + 1))
    col0 = 0 if band == "left" else room_w - template.shape[1]
    mask[row0 : row0 + template.shape[0], col0 : col0 + template.shape[1]] = template
    return mask


def _pad_canvas(local: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = local.shape
    free_h = 30 - h
    free_w = 30 - w
    top = int(rng.integers(0, min(3, free_h) + 1))
    bottom = int(rng.integers(0, min(3, free_h - top) + 1))
    left = int(rng.integers(0, min(3, free_w) + 1))
    right = int(rng.integers(0, min(3, free_w - left) + 1))
    out = np.zeros((h + top + bottom, w + left + right), dtype=int)
    out[top : top + h, left : left + w] = local
    return out


def _gen_row_strip(rng: np.random.Generator) -> np.ndarray:
    room_count = int(rng.integers(2, 5))
    room_h = int(rng.integers(5, 8))
    room_ws = [int(rng.integers(4, 7)) for _ in range(room_count)]
    local_h = room_h + 2
    local_w = sum(room_ws) + room_count + 1
    if local_w > 30:
        raise ValueError("row strip too wide")

    colors = [int(c) for c in rng.choice(COLOR_POOL, size=room_count, replace=False)]
    grid = np.zeros((local_h, local_w), dtype=int)

    boundaries = [0]
    for width in room_ws:
        boundaries.append(boundaries[-1] + width + 1)

    for idx, color in enumerate(colors):
        c0, c1 = boundaries[idx], boundaries[idx + 1]
        grid[0, c0 : c1 + 1] = color
        grid[-1, c0 : c1 + 1] = color
    grid[:, 0] = colors[0]
    grid[:, -1] = colors[-1]

    top_first = bool(rng.integers(0, 2))
    for idx, width in enumerate(room_ws):
        band = "top" if ((idx % 2 == 0) == top_first) else "bottom"
        furniture = _row_furniture(room_h, width, band, rng)
        c0, c1 = boundaries[idx], boundaries[idx + 1]
        interior = grid[1:-1, c0 + 1 : c1]
        interior[furniture] = GREY

    for idx in range(room_count - 1):
        wall_col = boundaries[idx + 1]
        wall_color = colors[idx + int(rng.integers(0, 2))]
        grid[1:-1, wall_col] = wall_color
        door_row = int(rng.integers(1, local_h - 1))
        grid[door_row, wall_col] = BG

    return _pad_canvas(grid, rng)


def _gen_col_strip(rng: np.random.Generator) -> np.ndarray:
    room_count = int(rng.integers(2, 5))
    room_w = int(rng.integers(5, 8))
    room_hs = [int(rng.integers(4, 7)) for _ in range(room_count)]
    local_h = sum(room_hs) + room_count + 1
    local_w = room_w + 2
    if local_h > 30:
        raise ValueError("column strip too tall")

    colors = [int(c) for c in rng.choice(COLOR_POOL, size=room_count, replace=False)]
    grid = np.zeros((local_h, local_w), dtype=int)

    boundaries = [0]
    for height in room_hs:
        boundaries.append(boundaries[-1] + height + 1)

    for idx, color in enumerate(colors):
        r0, r1 = boundaries[idx], boundaries[idx + 1]
        grid[r0 : r1 + 1, 0] = color
        grid[r0 : r1 + 1, -1] = color
    grid[0, :] = colors[0]
    grid[-1, :] = colors[-1]

    left_first = bool(rng.integers(0, 2))
    for idx, height in enumerate(room_hs):
        band = "left" if ((idx % 2 == 0) == left_first) else "right"
        furniture = _col_furniture(height, room_w, band, rng)
        r0, r1 = boundaries[idx], boundaries[idx + 1]
        interior = grid[r0 + 1 : r1, 1:-1]
        interior[furniture] = GREY

    for idx in range(room_count - 1):
        wall_row = boundaries[idx + 1]
        wall_color = colors[idx + int(rng.integers(0, 2))]
        grid[wall_row, 1:-1] = wall_color
        door_col = int(rng.integers(1, local_w - 1))
        grid[wall_row, door_col] = BG

    return _pad_canvas(grid, rng)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    try:
        grid = _gen_row_strip(rng) if rng.random() < 0.5 else _gen_col_strip(rng)
    except ValueError:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if not np.any(grid == GREY):
        return None
    if np.array_equal(grid, out):
        return None
    if grid.ndim != 2 or out.ndim != 2 or max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-projection puzzle")
