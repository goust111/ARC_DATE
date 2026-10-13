from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
AZURE = 8
BAND = 2

DIR8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

INTERIOR_COLOR_POOL = (1, 2, 3, 4, 6, 7, 9)
MOTIF_SHAPES = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 1)),
)

RIGHT_SHAPE = frozenset({(1, 0), (0, 1), (2, 1)})
LEFT_SHAPE = frozenset({(0, 0), (2, 0), (1, 1)})
DOWN_SHAPE = frozenset({(0, 1), (1, 0), (1, 2)})
UP_SHAPE = frozenset({(0, 0), (0, 2), (1, 1)})

GEN_RETRIES = 400
INNER_HEIGHT_RANGE = (8, 18)
INNER_WIDTH_RANGE = (8, 18)


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in DIR8:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _shift_foreground(line: np.ndarray, delta: int) -> np.ndarray:
    length = len(line)
    if length == 0:
        return line.copy()
    idx = np.flatnonzero(line != BG)
    if idx.size == 0:
        return line.copy()
    out = np.full_like(line, BG)
    out[(idx + delta) % length] = line[idx]
    return out


def _apply_controls(
    inner: np.ndarray,
    row_ops: dict[int, int],
    col_ops: dict[int, int],
) -> np.ndarray:
    out = inner.copy()
    for row, delta in sorted(row_ops.items()):
        if 0 <= row < out.shape[0]:
            out[row] = _shift_foreground(out[row], delta)
    for col, delta in sorted(col_ops.items()):
        if 0 <= col < out.shape[1]:
            out[:, col] = _shift_foreground(out[:, col], delta)
    return out


def _parse_arrow_component(
    coords: list[tuple[int, int]],
    full_h: int,
    full_w: int,
) -> dict[str, int | str] | None:
    if len(coords) != 3:
        return None
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    norm = frozenset((r - r0, c - c0) for r, c in coords)

    if norm == RIGHT_SHAPE:
        axis = "row"
        direction = "right"
        line = r0 + 1
    elif norm == LEFT_SHAPE:
        axis = "row"
        direction = "left"
        line = r0 + 1
    elif norm == DOWN_SHAPE:
        axis = "col"
        direction = "down"
        line = c0 + 1
    elif norm == UP_SHAPE:
        axis = "col"
        direction = "up"
        line = c0 + 1
    else:
        return None

    if axis == "row":
        if c0 < BAND:
            side = "left"
        elif c1 >= full_w - BAND:
            side = "right"
        else:
            return None
    else:
        if r0 < BAND:
            side = "top"
        elif r1 >= full_h - BAND:
            side = "bottom"
        else:
            return None

    return {
        "axis": axis,
        "direction": direction,
        "line": line,
        "side": side,
    }


def _opposite_marker_cells(info: dict[str, int | str], full_h: int, full_w: int) -> list[tuple[int, int]]:
    axis = str(info["axis"])
    side = str(info["side"])
    line = int(info["line"])
    if axis == "row":
        if side == "left":
            return [(line - 1, full_w - 2), (line, full_w - 1), (line + 1, full_w - 2)]
        return [(line - 1, 1), (line, 0), (line + 1, 1)]
    if side == "top":
        return [(full_h - 2, line - 1), (full_h - 1, line), (full_h - 2, line + 1)]
    return [(0, line), (1, line - 1), (1, line + 1)]


def _control_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    mask[:BAND, :] = True
    mask[-BAND:, :] = True
    mask[:, :BAND] = True
    mask[:, -BAND:] = True
    return mask


def _arrow_cells(direction: str, index: int, full_h: int, full_w: int) -> list[tuple[int, int]]:
    if direction == "right":
        row = index + BAND
        return [(row, 0), (row - 1, 1), (row + 1, 1)]
    if direction == "left":
        row = index + BAND
        return [(row - 1, full_w - 2), (row + 1, full_w - 2), (row, full_w - 1)]
    if direction == "down":
        col = index + BAND
        return [(0, col), (1, col - 1), (1, col + 1)]
    col = index + BAND
    return [(full_h - 2, col - 1), (full_h - 2, col + 1), (full_h - 1, col)]


def _marker_cells(direction: str, index: int, full_h: int, full_w: int) -> list[tuple[int, int]]:
    if direction == "right":
        row = index + BAND
        return [(row - 1, full_w - 2), (row, full_w - 1), (row + 1, full_w - 2)]
    if direction == "left":
        row = index + BAND
        return [(row - 1, 1), (row, 0), (row + 1, 1)]
    if direction == "down":
        col = index + BAND
        return [(full_h - 2, col - 1), (full_h - 1, col), (full_h - 2, col + 1)]
    col = index + BAND
    return [(0, col), (1, col - 1), (1, col + 1)]


def _pick_spaced_indices(rng: random.Random, limit: int, count: int, gap: int) -> list[int] | None:
    values = list(range(limit))
    for _ in range(60):
        rng.shuffle(values)
        chosen: list[int] = []
        for value in values:
            if all(abs(value - prev) >= gap for prev in chosen):
                chosen.append(value)
                if len(chosen) == count:
                    return sorted(chosen)
    return None


def _fits(inner: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    h, w = inner.shape
    return all(0 <= r < h and 0 <= c < w and inner[r, c] == BG for r, c in cells)


def _place_motifs(rng: random.Random, inner: np.ndarray, palette: list[int], count: int) -> None:
    h, w = inner.shape
    for _ in range(count):
        shape = rng.choice(MOTIF_SHAPES)
        color = rng.choice(palette)
        for _ in range(30):
            anchor_r = rng.randrange(h)
            anchor_c = rng.randrange(w)
            cells = [(anchor_r + dr, anchor_c + dc) for dr, dc in shape]
            if _fits(inner, cells):
                for r, c in cells:
                    inner[r, c] = color
                break


def _ensure_palette_variety(rng: random.Random, inner: np.ndarray, palette: list[int]) -> None:
    used = {int(v) for v in inner.flatten() if int(v) != BG}
    if len(used) >= min(2, len(palette)):
        return
    free = list(map(tuple, np.argwhere(inner == BG)))
    if not free:
        return
    rng.shuffle(free)
    missing = [color for color in palette if color not in used]
    for color, (r, c) in zip(missing, free):
        inner[r, c] = color


def _build_inner(
    rng: random.Random,
    inner_h: int,
    inner_w: int,
    axis_mode: str,
    controlled: list[int],
    palette: list[int],
) -> np.ndarray:
    inner = np.full((inner_h, inner_w), BG, dtype=int)

    if axis_mode == "row":
        for row in controlled:
            count = rng.randint(2, min(5, inner_w))
            for col in rng.sample(range(inner_w), count):
                inner[row, col] = rng.choice(palette)
    else:
        for col in controlled:
            count = rng.randint(2, min(5, inner_h))
            for row in rng.sample(range(inner_h), count):
                inner[row, col] = rng.choice(palette)

    motif_count = rng.randint(4, max(5, (inner_h * inner_w) // 12))
    _place_motifs(rng, inner, palette, motif_count)
    _ensure_palette_variety(rng, inner, palette)
    return inner


def main(grid: np.ndarray) -> np.ndarray:
    """Use border arrowheads and opposite-side azure counts to cyclically shift foreground pixels by row or column.

    Input structure:
    - The background is black and the picture sits inside a grey control frame.
    - The frame carries 3-pixel colored arrowheads: one offset shape on the left or right side controls an interior row, and one offset shape on the top or bottom controls an interior column.
    - On the opposite side of that same row or column, some of the mirrored cue cells may be azure single-pixel markers.

    Rule:
    - Read each 3-pixel arrowhead by geometry: right means one pixel on the left with two on the right above and below; down means one on top with two below left and right; left and up are the mirrored cases.
    - Count the azure markers on the opposite end of the same row or column to get K.
    - Remove the control frame and cyclically shift every non-background pixel in the controlled interior row or column by K cells along the arrow direction; background cells stay background and do not move.
    - Rows and columns with no arrow stay unchanged.

    Key insight:
    - The arrowhead provides only the direction, while the shift amount is encoded separately by the azure count on the far side of the same line.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    if h <= 2 * BAND or w <= 2 * BAND:
        return np.zeros((max(0, h - 2 * BAND), max(0, w - 2 * BAND)), dtype=int)

    inner = g[BAND:-BAND, BAND:-BAND].copy()
    control_mask = _control_mask(g.shape)

    row_ops: dict[int, int] = {}
    col_ops: dict[int, int] = {}

    colors = sorted(
        int(color)
        for color in np.unique(g[control_mask])
        if int(color) not in {BG, GREY, AZURE}
    )

    for color in colors:
        mask = (g == color) & control_mask
        comps = _components_mask(mask)
        comps.sort(key=lambda comp: min(comp))
        for comp in comps:
            info = _parse_arrow_component(comp, h, w)
            if info is None:
                continue
            index = int(info["line"]) - BAND
            marker_cells = _opposite_marker_cells(info, h, w)
            amount = sum(int(0 <= r < h and 0 <= c < w and g[r, c] == AZURE) for r, c in marker_cells)
            delta = amount if str(info["direction"]) in {"right", "down"} else -amount
            if str(info["axis"]) == "row":
                row_ops[index] = delta
            else:
                col_ops[index] = delta

    return _apply_controls(inner, row_ops, col_ops)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    inner_h = rng.randint(*INNER_HEIGHT_RANGE)
    inner_w = rng.randint(*INNER_WIDTH_RANGE)
    axis_mode = rng.choice(("row", "col"))
    limit = inner_h if axis_mode == "row" else inner_w
    max_controls = min(3, (limit + 2) // 3)
    control_count = rng.randint(1, max_controls)
    controlled = _pick_spaced_indices(rng, limit, control_count, gap=3)
    if controlled is None:
        return None

    max_pattern_colors = min(4, len(INTERIOR_COLOR_POOL) - control_count)
    pattern_color_count = rng.randint(2, max_pattern_colors)
    palette = rng.sample(list(INTERIOR_COLOR_POOL), pattern_color_count)
    arrow_color_pool = [color for color in INTERIOR_COLOR_POOL if color not in palette]
    if len(arrow_color_pool) < control_count:
        return None
    arrow_colors = rng.sample(arrow_color_pool, control_count)

    inner = _build_inner(rng, inner_h, inner_w, axis_mode, controlled, palette)
    full_h = inner_h + 2 * BAND
    full_w = inner_w + 2 * BAND
    grid = np.full((full_h, full_w), GREY, dtype=int)
    grid[BAND:-BAND, BAND:-BAND] = inner

    row_ops: dict[int, int] = {}
    col_ops: dict[int, int] = {}
    used: set[tuple[int, int]] = set()

    for index, color in zip(controlled, arrow_colors):
        direction = rng.choice(("left", "right")) if axis_mode == "row" else rng.choice(("up", "down"))
        arrow_cells = _arrow_cells(direction, index, full_h, full_w)
        marker_cells = _marker_cells(direction, index, full_h, full_w)
        if any(cell in used for cell in arrow_cells + marker_cells):
            return None

        for r, c in arrow_cells:
            grid[r, c] = color
            used.add((r, c))

        amount = rng.choice((1, 2, 3))
        chosen_markers = rng.sample(marker_cells, amount)
        for r, c in chosen_markers:
            grid[r, c] = AZURE
            used.add((r, c))

        delta = amount if direction in {"right", "down"} else -amount
        if axis_mode == "row":
            row_ops[index] = delta
        else:
            col_ops[index] = delta

    expected = _apply_controls(inner, row_ops, col_ops)
    if np.array_equal(expected, inner):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted retries")
