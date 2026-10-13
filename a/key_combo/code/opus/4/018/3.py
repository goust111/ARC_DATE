import random
from typing import Any

import numpy as np


BACKGROUND = 0
YELLOW = 4
GREY = 5
COLOR_POOL = (1, 2, 3, 6, 7, 8, 9)

GRID_HEIGHT_RANGE = (11, 17)
GRID_WIDTH_RANGE = (14, 20)
RUN_LEN_RANGE = (2, 4)
PALETTE_LEN_CHOICES = (3, 4)
BODY_HEIGHT_RANGE = (6, 10)
SIDE_WIDTH_RANGE = (3, 5)
MIN_GAP_TO_PALETTE = 2
GEN_ATTEMPTS = 400


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _read_run_and_palette(grid: np.ndarray) -> tuple[int, list[int]]:
    h, w = grid.shape
    edge = grid[:, w - 1]

    run_len = 0
    while run_len < h and int(edge[run_len]) == GREY:
        run_len += 1
    assert 2 <= run_len <= 4, "expected a top-aligned grey run of height 2..4"

    palette: list[int] = []
    r = run_len
    while r < h and int(edge[r]) != BACKGROUND:
        color = int(edge[r])
        assert color not in (BACKGROUND, GREY), "palette must be below the grey run"
        palette.append(color)
        r += 1

    assert 3 <= len(palette) <= 4, "expected a palette of 3..4 colors"
    assert len(set(palette)) == len(palette), "palette colors must be distinct"
    return run_len, palette


def _find_axis_and_control(grid: np.ndarray) -> tuple[int, int, int, tuple[int, int]]:
    h, w = grid.shape
    yellow_coords = np.argwhere(grid[:, : w - 1] == YELLOW)
    assert yellow_coords.size > 0, "expected yellow axis/control pixels"

    axis_col = None
    axis_rows: list[int] | None = None
    best_len = -1
    for col in sorted(set(map(int, yellow_coords[:, 1]))):
        rows = sorted(map(int, yellow_coords[yellow_coords[:, 1] == col, 0]))
        if len(rows) < 2:
            continue
        if rows != list(range(rows[0], rows[-1] + 1)):
            continue
        if len(rows) > best_len:
            best_len = len(rows)
            axis_col = col
            axis_rows = rows

    assert axis_col is not None and axis_rows is not None, "failed to find the vertical yellow axis"

    control_coords = yellow_coords[yellow_coords[:, 1] != axis_col]
    assert control_coords.shape[0] == 1, "expected exactly one off-axis yellow control marker"
    control = tuple(map(int, control_coords[0]))
    return int(axis_col), axis_rows[0], axis_rows[-1], control


def _mirrored_fill_positions(
    grid: np.ndarray,
    axis_col: int,
    body_r0: int,
    body_r1: int,
    source_side: int,
) -> list[tuple[int, int]]:
    h, w = grid.shape
    fill_positions: list[tuple[int, int]] = []

    for r in range(body_r0, body_r1 + 1):
        if source_side < 0:
            source_cols = np.where(grid[r, :axis_col] != BACKGROUND)[0]
        else:
            source_cols = np.where(grid[r, axis_col + 1 : w - 1] != BACKGROUND)[0] + axis_col + 1

        for src_c in source_cols:
            dst_c = 2 * axis_col - int(src_c)
            assert 0 <= dst_c < w - 1, "mirrored pixels must stay left of the palette column"
            fill_positions.append((int(r), int(dst_c)))

    fill_positions.sort()
    return fill_positions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror the trusted half of the sprite, then color the generated half by a palette cycle with fixed run length.

    Input structure:
    - The background is black.
    - A multicolor sprite on the left contains a 1-cell-wide vertical yellow axis spanning the sprite's body rows.
    - One off-axis yellow control pixel sits above or below the body on the trusted side, and the rightmost column contains a top-aligned grey run followed by a vertical palette.

    Rule:
    - Use the control pixel's side relative to the yellow axis as the trusted source side.
    - Mirror every non-background source-side body pixel across the yellow axis to obtain the target-side positions that must be generated.
    - Read the grey run height as run length R, read the palette colors from top to bottom, order the generated target positions in row-major order, and color position i with palette[(i // R) mod k].

    Key insight:
    - The mirror determines geometry only; the generated side does not copy source colors and is instead repainted by the independent palette/run-length legend.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    run_len, palette = _read_run_and_palette(g)
    axis_col, body_r0, body_r1, control = _find_axis_and_control(g)
    source_side = -1 if control[1] < axis_col else 1
    fill_positions = _mirrored_fill_positions(g, axis_col, body_r0, body_r1, source_side)

    out = g.copy()
    for i, (r, c) in enumerate(fill_positions):
        out[r, c] = palette[(i // run_len) % len(palette)]
    return out


def _oracle_from_meta(grid: np.ndarray, meta: dict[str, Any]) -> np.ndarray:
    out = grid.copy()
    palette = list(meta["palette"])
    run_len = int(meta["run_len"])
    for i, (r, c) in enumerate(meta["fill_positions"]):
        out[r, c] = palette[(i // run_len) % len(palette)]
    return out


def _sample_widths(rng: random.Random, body_h: int, side_w: int, min_area: int) -> list[int] | None:
    widths = [side_w] * body_h
    taper_steps = rng.randint(1, max(2, body_h // 2))
    for _ in range(taper_steps):
        idx = rng.randrange(body_h)
        if widths[idx] > 1:
            widths[idx] -= rng.randint(1, min(2, widths[idx] - 1))

    if all(width == side_w for width in widths):
        idx = rng.randrange(body_h)
        widths[idx] = max(1, side_w - 1)

    if sum(widths) < min_area:
        return None
    return widths


def _build_case(rng: random.Random) -> tuple[np.ndarray, dict[str, Any]] | None:
    run_len = rng.randint(*RUN_LEN_RANGE)
    palette_len = rng.choice(PALETTE_LEN_CHOICES)
    min_source_area = 2 * run_len * palette_len

    viable_shapes = [
        (body_h, side_w)
        for body_h in range(BODY_HEIGHT_RANGE[0], BODY_HEIGHT_RANGE[1] + 1)
        for side_w in range(SIDE_WIDTH_RANGE[0], SIDE_WIDTH_RANGE[1] + 1)
        if body_h * side_w >= min_source_area
    ]
    if not viable_shapes:
        return None

    body_h, side_w = viable_shapes[rng.randrange(len(viable_shapes))]
    widths = _sample_widths(rng, body_h, side_w, min_source_area)
    if widths is None:
        return None

    min_h = max(GRID_HEIGHT_RANGE[0], body_h + 3)
    max_h = GRID_HEIGHT_RANGE[1]
    if min_h > max_h:
        return None
    h = rng.randint(min_h, max_h)

    min_w = max(GRID_WIDTH_RANGE[0], 2 * side_w + MIN_GAP_TO_PALETTE + 3)
    max_w = GRID_WIDTH_RANGE[1]
    if min_w > max_w:
        return None
    w = rng.randint(min_w, max_w)

    grid = np.zeros((h, w), dtype=int)

    palette = rng.sample(COLOR_POOL, palette_len)
    source_color_pool = [color for color in COLOR_POOL if color not in palette]
    source_color_count = rng.randint(2, min(3, len(source_color_pool)))
    source_colors = rng.sample(source_color_pool, source_color_count)

    body_r0 = rng.randint(1, h - body_h - 2)
    body_r1 = body_r0 + body_h - 1
    axis_col = rng.randint(side_w + 1, w - side_w - MIN_GAP_TO_PALETTE - 2)
    source_left = bool(rng.randrange(2))
    marker_above = bool(rng.randrange(2))
    color_phase = rng.randrange(len(source_colors))

    for i, width in enumerate(widths):
        r = body_r0 + i
        if source_left:
            cols = range(axis_col - width, axis_col)
        else:
            cols = range(axis_col + 1, axis_col + width + 1)

        for c in cols:
            dist = abs(c - axis_col)
            color = source_colors[(color_phase + i + dist) % len(source_colors)]
            grid[r, c] = color

    grid[body_r0 : body_r1 + 1, axis_col] = YELLOW

    marker_r = body_r0 - 1 if marker_above else body_r1 + 1
    edge_width = widths[0] if marker_above else widths[-1]
    marker_offset = rng.randint(1, edge_width)
    marker_c = axis_col - marker_offset if source_left else axis_col + marker_offset
    grid[marker_r, marker_c] = YELLOW

    grid[:run_len, w - 1] = GREY
    for i, color in enumerate(palette):
        grid[run_len + i, w - 1] = color

    fill_positions = []
    for i, width in enumerate(widths):
        r = body_r0 + i
        if source_left:
            src_cols = range(axis_col - width, axis_col)
        else:
            src_cols = range(axis_col + 1, axis_col + width + 1)
        for src_c in src_cols:
            fill_positions.append((r, 2 * axis_col - src_c))
    fill_positions.sort()

    if any(c == w - 1 for _, c in fill_positions):
        return None
    if any(grid[r, c] != BACKGROUND for r, c in fill_positions):
        return None

    yellow_coords = np.argwhere(grid == YELLOW)
    y_bbox = _bbox_coords(yellow_coords)
    if y_bbox[1] == w - 1:
        return None

    meta = {
        "run_len": run_len,
        "palette": palette,
        "fill_positions": fill_positions,
        "axis_col": axis_col,
        "body_bbox": (body_r0, axis_col - side_w, body_r1, axis_col + side_w),
        "source_left": source_left,
        "marker": (marker_r, marker_c),
    }

    expected = _oracle_from_meta(grid, meta)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid, meta


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        case = _build_case(random.Random())
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid puzzle instance")
