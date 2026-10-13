import random

import numpy as np


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            stack = [(int(r0), int(c0))]
            seen[r0, c0] = True
            comp: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                comp.append((r, c))
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _edge_runs(line: np.ndarray, bg: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for idx, value in enumerate(line.tolist() + [bg], start=0):
        if start is None:
            if int(value) != bg:
                start = idx
            continue
        if int(value) == bg:
            runs.append((start, idx))
            start = None
    return runs


def _detect_palette(grid: np.ndarray, bg: int) -> tuple[list[int], np.ndarray]:
    """
    Detect a 1-cell-thick border palette strip.

    Horizontal palettes are read left-to-right; vertical palettes are read top-to-bottom.
    """
    h, w = grid.shape
    candidates: list[tuple[int, str, int, int, list[int]]] = []

    def add_horizontal(side: str, border_row: int, inner_row: int) -> None:
        line = grid[border_row, :]
        inner = grid[inner_row, :]
        for start, end in _edge_runs(line, bg):
            values = [int(v) for v in line[start:end]]
            if len(values) != len(set(values)):
                continue
            if np.any(inner[start:end] != bg):
                continue
            candidates.append((end - start, side, start, end, values))

    def add_vertical(side: str, border_col: int, inner_col: int) -> None:
        line = grid[:, border_col]
        inner = grid[:, inner_col]
        for start, end in _edge_runs(line, bg):
            values = [int(v) for v in line[start:end]]
            if len(values) != len(set(values)):
                continue
            if np.any(inner[start:end] != bg):
                continue
            candidates.append((end - start, side, start, end, values))

    if h >= 2:
        add_horizontal("top", 0, 1)
        add_horizontal("bottom", h - 1, h - 2)
    if w >= 2:
        add_vertical("left", 0, 1)
        add_vertical("right", w - 1, w - 2)

    assert candidates, "expected a detectable border palette strip"
    candidates.sort(key=lambda item: (item[0], item[1], -item[2]), reverse=True)
    _, side, start, end, values = candidates[0]

    mask = np.zeros_like(grid, dtype=bool)
    if side == "top":
        mask[0, start:end] = True
    elif side == "bottom":
        mask[h - 1, start:end] = True
    elif side == "left":
        mask[start:end, 0] = True
    else:
        mask[start:end, w - 1] = True

    return values, mask


def _frame_boundary(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    return cells


def _rectangular_frame_info(cells: list[tuple[int, int]], grid: np.ndarray) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = _bbox(cells)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None
    color = int(grid[cells[0]])
    if any(int(grid[r, c]) != color for r, c in cells):
        return None
    if set(cells) != _frame_boundary(r0, c0, r1, c1):
        return None
    return r0, c0, r1, c1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each closed rectangular frame by counting its interior marker blobs and indexing the edge palette.

    Input structure:
    - The background is the dominant color, and one grid edge contains a contiguous 1-cell-thick palette strip of distinct colors.
    - Away from the edge palette sit one or more monochrome 1-cell-thick rectangular frames.
    - Each frame encloses one or more separate non-background marker blobs inside its interior.

    Rule:
    - Detect the palette strip on the border; read it left-to-right if horizontal, or top-to-bottom if vertical.
    - For every rectangular frame, count the number K of 4-connected non-background blobs strictly inside the frame.
    - Recolor every pixel on that frame's boundary with the K-th palette color, keeping the interior marker blobs and the palette unchanged.

    Key insight:
    - The enclosed blobs are counters, and the border palette is a lookup table, so the task is solved by counting enclosed components and using that count as a visible palette index.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _background_color(g)
    palette, palette_mask = _detect_palette(g, bg)

    out = g.copy()
    foreground = (g != bg) & (~palette_mask)
    for comp in _components(foreground):
        frame = _rectangular_frame_info(comp, g)
        if frame is None:
            continue
        r0, c0, r1, c1 = frame
        interior = g[r0 + 1 : r1, c0 + 1 : c1] != bg
        count = len(_components(interior))
        if count == 0:
            continue
        assert count <= len(palette), "palette must be long enough for every frame count"
        color = int(palette[count - 1])
        for r, c in comp:
            out[r, c] = color

    return out


# --------------- generator ---------------

GEN_RETRIES = 400
PLACEMENT_RETRIES = 120
FRAME_COLORS = tuple(range(1, 10))


def _paint_palette(
    grid: np.ndarray, side: str, start: int, colors: list[int]
) -> np.ndarray:
    mask = np.zeros_like(grid, dtype=bool)
    if side == "top":
        for idx, color in enumerate(colors):
            grid[0, start + idx] = color
            mask[0, start + idx] = True
    elif side == "bottom":
        for idx, color in enumerate(colors):
            grid[-1, start + idx] = color
            mask[-1, start + idx] = True
    elif side == "left":
        for idx, color in enumerate(colors):
            grid[start + idx, 0] = color
            mask[start + idx, 0] = True
    else:
        for idx, color in enumerate(colors):
            grid[start + idx, -1] = color
            mask[start + idx, -1] = True
    return mask


def _reserve_edge_band(occupied: np.ndarray, side: str) -> None:
    if side == "top":
        occupied[0:2, :] = True
    elif side == "bottom":
        occupied[-2:, :] = True
    elif side == "left":
        occupied[:, 0:2] = True
    else:
        occupied[:, -2:] = True


def _sample_frame_spec(palette_len: int) -> tuple[int, int, int, list[list[tuple[int, int]]]] | None:
    count = random.randint(1, palette_len)
    candidates = [
        (slot_rows, slot_cols)
        for slot_rows in range(1, 4)
        for slot_cols in range(1, 5)
        if slot_rows * slot_cols >= count
    ]
    if not candidates:
        return None

    slot_rows, slot_cols = random.choice(candidates)
    base_inner_h = 3 * slot_rows - 1
    base_inner_w = 3 * slot_cols - 1
    extra_h = random.randint(0, 2)
    extra_w = random.randint(0, 2)
    inner_h = base_inner_h + extra_h
    inner_w = base_inner_w + extra_w
    offset_r = random.randint(0, extra_h)
    offset_c = random.randint(0, extra_w)

    slots = [(offset_r + 3 * rr, offset_c + 3 * cc) for rr in range(slot_rows) for cc in range(slot_cols)]
    chosen = random.sample(slots, count)
    blobs: list[list[tuple[int, int]]] = []
    for base_r, base_c in chosen:
        shape_type = random.choice(["dot", "h2", "v2", "box"])
        if shape_type == "dot":
            blobs.append([(base_r + random.randint(0, 1), base_c + random.randint(0, 1))])
        elif shape_type == "h2":
            row_offset = random.randint(0, 1)
            blobs.append([(base_r + row_offset, base_c), (base_r + row_offset, base_c + 1)])
        elif shape_type == "v2":
            col_offset = random.randint(0, 1)
            blobs.append([(base_r, base_c + col_offset), (base_r + 1, base_c + col_offset)])
        else:
            blobs.append(
                [(base_r, base_c), (base_r, base_c + 1), (base_r + 1, base_c), (base_r + 1, base_c + 1)]
            )

    return inner_h + 2, inner_w + 2, count, blobs


def _paint_frame(
    grid: np.ndarray,
    top: int,
    left: int,
    height: int,
    width: int,
    frame_color: int,
    marker_color: int,
    blobs: list[list[tuple[int, int]]],
) -> list[tuple[int, int]]:
    bottom = top + height - 1
    right = left + width - 1
    border = sorted(_frame_boundary(top, left, bottom, right))
    for r, c in border:
        grid[r, c] = frame_color
    for blob in blobs:
        for dr, dc in blob:
            grid[top + 1 + dr, left + 1 + dc] = marker_color
    return border


def _build_expected(
    grid: np.ndarray, frame_records: list[tuple[list[tuple[int, int]], int]], palette: list[int]
) -> np.ndarray:
    out = grid.copy()
    for border_cells, count in frame_records:
        color = int(palette[count - 1])
        for r, c in border_cells:
            out[r, c] = color
    return out


def _place_frames(
    grid: np.ndarray,
    occupied: np.ndarray,
    frame_color: int,
    marker_color: int,
    palette_len: int,
    frame_count: int,
) -> list[tuple[list[tuple[int, int]], int]] | None:
    h, w = grid.shape
    records: list[tuple[list[tuple[int, int]], int]] = []

    for _ in range(frame_count):
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            spec = _sample_frame_spec(palette_len)
            if spec is None:
                continue
            frame_h, frame_w, count, blobs = spec
            if frame_h >= h - 1 or frame_w >= w - 1:
                continue
            top = random.randint(1, h - frame_h - 1)
            left = random.randint(1, w - frame_w - 1)
            bottom = top + frame_h - 1
            right = left + frame_w - 1
            rmin = max(0, top - 1)
            rmax = min(h - 1, bottom + 1)
            cmin = max(0, left - 1)
            cmax = min(w - 1, right + 1)
            if np.any(occupied[rmin : rmax + 1, cmin : cmax + 1]):
                continue
            occupied[rmin : rmax + 1, cmin : cmax + 1] = True
            border = _paint_frame(grid, top, left, frame_h, frame_w, frame_color, marker_color, blobs)
            records.append((border, count))
            placed = True
            break
        if not placed:
            return None

    return records


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(14, 28)
    width = random.randint(14, 28)
    side = random.choice(["top", "bottom", "left", "right"])

    span = width if side in ("top", "bottom") else height
    palette_len = random.randint(3, min(6, span - 2))

    colors = list(FRAME_COLORS)
    random.shuffle(colors)
    frame_color = colors.pop()
    marker_color = colors.pop()
    palette = colors[:palette_len]

    grid = np.zeros((height, width), dtype=int)
    occupied = np.zeros((height, width), dtype=bool)
    _reserve_edge_band(occupied, side)

    start = random.randint(0, span - palette_len)
    _paint_palette(grid, side, start, palette)

    max_frames = 4
    frame_count = random.randint(1, max_frames)
    records = _place_frames(grid, occupied, frame_color, marker_color, palette_len, frame_count)
    if records is None:
        return None

    expected = _build_expected(grid, records, palette)
    try:
        actual = main(grid)
    except Exception:
        return None
    if np.array_equal(actual, expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for the palette-frame counting task")
