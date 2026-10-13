import numpy as np


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a non-empty mask"
    (r0, c0), (r1, c1) = coords.min(axis=0), coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _get_bg(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[counts.argmax()])


def _find_target_frame(grid: np.ndarray, bg: int) -> dict[str, object]:
    candidates: list[dict[str, object]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue

        mask = grid == color
        r0, c0, r1, c1 = _bbox(mask)
        box = grid[r0 : r1 + 1, c0 : c1 + 1]
        box_mask = mask[r0 : r1 + 1, c0 : c1 + 1]

        line_rows = np.flatnonzero(np.all(box_mask, axis=1))
        line_cols = np.flatnonzero(np.all(box_mask, axis=0))
        if line_rows.size < 2 or line_cols.size < 2:
            continue
        if line_rows[0] != 0 or line_rows[-1] != box_mask.shape[0] - 1:
            continue
        if line_cols[0] != 0 or line_cols[-1] != box_mask.shape[1] - 1:
            continue

        row_steps = np.diff(line_rows)
        col_steps = np.diff(line_cols)
        if np.any(row_steps <= 1) or np.any(col_steps <= 1):
            continue
        if len(set(map(int, row_steps))) != 1 or len(set(map(int, col_steps))) != 1:
            continue

        expected = np.zeros_like(box_mask, dtype=bool)
        expected[line_rows, :] = True
        expected[:, line_cols] = True
        if not np.array_equal(box_mask, expected):
            continue
        if not np.all(box[~expected] == bg):
            continue

        candidates.append(
            {
                "color": color,
                "bbox": (r0, c0, r1, c1),
                "line_rows": tuple(int(r0 + x) for x in line_rows),
                "line_cols": tuple(int(c0 + x) for x in line_cols),
                "rows": int(line_rows.size - 1),
                "cols": int(line_cols.size - 1),
            }
        )

    assert len(candidates) == 1, "expected exactly one target frame"
    return candidates[0]


def _extract_source_matrix(grid: np.ndarray, bg: int, frame: dict[str, object]) -> np.ndarray:
    frame_color = int(frame["color"])
    frame_r0, frame_c0, _frame_r1, _frame_c1 = frame["bbox"]
    rows = int(frame["rows"])
    cols = int(frame["cols"])

    source_mask = (grid != bg) & (grid != frame_color)
    assert np.any(source_mask), "expected a source region"
    r0, c0, r1, c1 = _bbox(source_mask)
    assert c1 < frame_c0, "source region must be left of the target frame"
    assert r0 >= 0 and r1 < grid.shape[0]
    assert frame_r0 >= 0

    source_box = source_mask[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(source_box), "source region must be a solid tiled rectangle"

    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    assert height % rows == 0 and width % cols == 0, "source size must match target grid dimensions"

    cell_h = height // rows
    cell_w = width // cols
    matrix = np.empty((rows, cols), dtype=int)

    for rr in range(rows):
        for cc in range(cols):
            patch = grid[
                r0 + rr * cell_h : r0 + (rr + 1) * cell_h,
                c0 + cc * cell_w : c0 + (cc + 1) * cell_w,
            ]
            color = int(patch[0, 0])
            assert color != bg and color != frame_color
            assert np.all(patch == color), "each source tile must be monochrome"
            matrix[rr, cc] = color

    return matrix


def main(grid: np.ndarray) -> np.ndarray:
    """Map the source color matrix into the target frame after reversing row and column order."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    frame = _find_target_frame(g, bg)
    source = _extract_source_matrix(g, bg, frame)
    rotated = source[::-1, ::-1]

    out = g.copy()
    line_rows = frame["line_rows"]
    line_cols = frame["line_cols"]

    for rr in range(int(frame["rows"])):
        for cc in range(int(frame["cols"])):
            r0 = int(line_rows[rr]) + 1
            r1 = int(line_rows[rr + 1])
            c0 = int(line_cols[cc]) + 1
            c1 = int(line_cols[cc + 1])
            assert np.all(g[r0:r1, c0:c1] == bg), "target cells must be empty in the input"
            out[r0:r1, c0:c1] = int(rotated[rr, cc])

    return out


# --------------- generator ---------------

_RNG = np.random.default_rng()

GRID_ROWS_RANGE = (2, 6)
GRID_COLS_RANGE = (2, 6)
SOURCE_CELL_SIZE_RANGE = (1, 5)
TARGET_CELL_SIZE_RANGE = (1, 4)
VERTICAL_OFFSET_RANGE = (1, 6)
SIDE_MARGIN_RANGE = (1, 6)
TOP_BOTTOM_MARGIN_RANGE = (1, 6)
GAP_RANGE = (2, 7)
EXTRA_CANVAS_RANGE = (0, 7)
SOURCE_PALETTE_SIZE_RANGE = (3, 7)
MAX_GENERATION_ATTEMPTS = 300


def _sample_source_matrix(rows: int, cols: int, colors: np.ndarray) -> np.ndarray:
    for _ in range(50):
        matrix = np.empty((rows, cols), dtype=int)
        for rr in range(rows):
            for cc in range(cols):
                blocked = set()
                if rr > 0:
                    blocked.add(int(matrix[rr - 1, cc]))
                if cc > 0:
                    blocked.add(int(matrix[rr, cc - 1]))
                choices = [int(color) for color in colors if int(color) not in blocked]
                assert choices, "need at least one visible color choice per tile"
                matrix[rr, cc] = int(_RNG.choice(choices))
        if not np.array_equal(matrix, matrix[::-1, ::-1]):
            return matrix
    raise RuntimeError("failed to sample a non-symmetric source matrix")


def _paint_source(
    grid: np.ndarray,
    top: int,
    left: int,
    matrix: np.ndarray,
    cell_h: int,
    cell_w: int,
) -> None:
    rows, cols = matrix.shape
    for rr in range(rows):
        for cc in range(cols):
            color = int(matrix[rr, cc])
            r0 = top + rr * cell_h
            c0 = left + cc * cell_w
            grid[r0 : r0 + cell_h, c0 : c0 + cell_w] = color


def _paint_target_frame(
    grid: np.ndarray,
    top: int,
    left: int,
    rows: int,
    cols: int,
    cell_h: int,
    cell_w: int,
    frame_color: int,
) -> None:
    height = rows * cell_h + rows + 1
    width = cols * cell_w + cols + 1

    for rr in range(rows + 1):
        r = top + rr * (cell_h + 1)
        grid[r, left : left + width] = frame_color
    for cc in range(cols + 1):
        c = left + cc * (cell_w + 1)
        grid[top : top + height, c] = frame_color


def _build_expected_output(
    grid: np.ndarray,
    target_top: int,
    target_left: int,
    matrix: np.ndarray,
    target_cell_h: int,
    target_cell_w: int,
) -> np.ndarray:
    out = grid.copy()
    rotated = matrix[::-1, ::-1]
    rows, cols = matrix.shape
    for rr in range(rows):
        for cc in range(cols):
            r0 = target_top + 1 + rr * (target_cell_h + 1)
            c0 = target_left + 1 + cc * (target_cell_w + 1)
            out[r0 : r0 + target_cell_h, c0 : c0 + target_cell_w] = int(rotated[rr, cc])
    return out


def _gen_attempt() -> np.ndarray | None:
    rows = int(_RNG.integers(*GRID_ROWS_RANGE))
    cols = int(_RNG.integers(*GRID_COLS_RANGE))
    source_cell_h = int(_RNG.integers(*SOURCE_CELL_SIZE_RANGE))
    source_cell_w = int(_RNG.integers(*SOURCE_CELL_SIZE_RANGE))
    target_cell_h = int(_RNG.integers(*TARGET_CELL_SIZE_RANGE))
    target_cell_w = int(_RNG.integers(*TARGET_CELL_SIZE_RANGE))

    all_colors = np.arange(10, dtype=int)
    bg = int(_RNG.choice(all_colors))
    remaining = np.array([color for color in all_colors if color != bg], dtype=int)
    frame_color = int(_RNG.choice(remaining))
    tile_candidates = np.array([color for color in remaining if color != frame_color], dtype=int)

    palette_size = int(
        _RNG.integers(
            SOURCE_PALETTE_SIZE_RANGE[0],
            min(SOURCE_PALETTE_SIZE_RANGE[1], len(tile_candidates)) + 1,
        )
    )
    tile_colors = _RNG.choice(tile_candidates, size=palette_size, replace=False)
    matrix = _sample_source_matrix(rows, cols, tile_colors)

    source_h = rows * source_cell_h
    source_w = cols * source_cell_w
    target_h = rows * target_cell_h + rows + 1
    target_w = cols * target_cell_w + cols + 1

    source_top = int(_RNG.integers(*VERTICAL_OFFSET_RANGE))
    target_top = int(_RNG.integers(*VERTICAL_OFFSET_RANGE))
    left_margin = int(_RNG.integers(*SIDE_MARGIN_RANGE))
    right_margin = int(_RNG.integers(*SIDE_MARGIN_RANGE))
    gap = int(_RNG.integers(*GAP_RANGE))
    bottom_margin = int(_RNG.integers(*TOP_BOTTOM_MARGIN_RANGE))
    extra_h = int(_RNG.integers(*EXTRA_CANVAS_RANGE))
    extra_w = int(_RNG.integers(*EXTRA_CANVAS_RANGE))

    source_left = left_margin
    target_left = source_left + source_w + gap
    height = max(source_top + source_h, target_top + target_h) + bottom_margin + extra_h
    width = target_left + target_w + right_margin + extra_w

    grid = np.full((height, width), bg, dtype=int)
    _paint_source(grid, source_top, source_left, matrix, source_cell_h, source_cell_w)
    _paint_target_frame(grid, target_top, target_left, rows, cols, target_cell_h, target_cell_w, frame_color)

    if _get_bg(grid) != bg:
        return None

    expected = _build_expected_output(grid, target_top, target_left, matrix, target_cell_h, target_cell_w)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
