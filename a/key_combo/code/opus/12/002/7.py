import numpy as np


BACKGROUND = 0
BLUE = 1
ENCODER_COLORS = (2, 3, 4, 7)
GEN_ATTEMPTS = 500


def _nearest_center_index(indices: list[int], length: int) -> int:
    center = (length - 1) / 2.0
    return min(indices, key=lambda idx: (abs(idx - center), idx))


def _blank_lines(grid: np.ndarray) -> tuple[list[int], list[int]]:
    rows = [int(r) for r in range(grid.shape[0]) if np.all(grid[r] == BACKGROUND)]
    cols = [int(c) for c in range(grid.shape[1]) if np.all(grid[:, c] == BACKGROUND)]
    return rows, cols


def _row_candidates(grid: np.ndarray, row: int) -> list[int]:
    top = grid[row - 1]
    bottom = grid[row + 1]
    return [
        int(c)
        for c in range(grid.shape[1])
        if top[c] == bottom[c] and int(top[c]) not in (BACKGROUND, BLUE)
    ]


def _col_candidates(grid: np.ndarray, col: int) -> list[int]:
    left = grid[:, col - 1]
    right = grid[:, col + 1]
    return [
        int(r)
        for r in range(grid.shape[0])
        if left[r] == right[r] and int(left[r]) not in (BACKGROUND, BLUE)
    ]


def _decode_row_color(grid: np.ndarray, row: int) -> int:
    candidates = _row_candidates(grid, row)
    assert candidates, f"blank row {row} has no encoding pair"
    col = _nearest_center_index(candidates, grid.shape[1])
    return int(grid[row - 1, col])


def _decode_col_color(grid: np.ndarray, col: int) -> int:
    candidates = _col_candidates(grid, col)
    assert candidates, f"blank col {col} has no encoding pair"
    row = _nearest_center_index(candidates, grid.shape[0])
    return int(grid[row, col - 1])


def _expected_output(
    grid: np.ndarray, row_colors: dict[int, int], col_colors: dict[int, int]
) -> np.ndarray:
    out = grid.copy()
    for col, color in col_colors.items():
        out[:, col] = color
    for row, color in row_colors.items():
        out[row, :] = color
    return out


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components_count(mask: np.ndarray) -> int:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return 0
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    count = 0
    for r, c in coords:
        if seen[r, c]:
            continue
        count += 1
        stack = [(int(r), int(c))]
        seen[r, c] = True
        while stack:
            cr, cc = stack.pop()
            for nr, nc in _neighbors4(cr, cc, h, w):
                if mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
    return count


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each fully blank row and column with the color encoded by the nearest matching border pair.

    Input structure:
    - Black background contains a blue irregular pattern interrupted by fully blank rows and/or columns.
    - Each blank row has one same-color non-black/non-blue pixel pair on the rows immediately above and below it.
    - Each blank column has one same-color non-black/non-blue pixel pair on the columns immediately left and right of it.

    Rule:
    - Detect all rows and columns made entirely of black.
    - For each blank row, find the matching same-color pair shared by row-1 and row+1; if several exist, use the column nearest the horizontal center.
    - For each blank column, find the matching same-color pair shared by col-1 and col+1; if several exist, use the row nearest the vertical center.
    - Paint each blank column with its decoded color, then paint each blank row with its decoded color so row/column intersections use the row color.

    Key insight:
    - The blank lines say where to paint, while nearby paired pixels say what color to paint; the encoding is read locally from the two sides of each missing line.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    blank_rows, blank_cols = _blank_lines(g)
    assert blank_rows or blank_cols, "expected at least one blank row or column"
    assert all(0 < r < h - 1 for r in blank_rows), "blank rows must be away from edges"
    assert all(0 < c < w - 1 for c in blank_cols), "blank cols must be away from edges"

    row_colors = {row: _decode_row_color(g, row) for row in blank_rows}
    col_colors = {col: _decode_col_color(g, col) for col in blank_cols}
    return _expected_output(g, row_colors, col_colors)


def _sample_spaced_positions(
    rng: np.random.Generator, length: int, count: int, border: int = 1, min_gap: int = 3
) -> list[int] | None:
    if count == 0:
        return []
    candidates = np.arange(border, length - border, dtype=int)
    if len(candidates) < count:
        return None
    for _ in range(200):
        picks = sorted(int(v) for v in rng.choice(candidates, size=count, replace=False))
        if all(b - a >= min_gap for a, b in zip(picks, picks[1:])):
            return picks
    return None


def _make_covering_backbone(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)

    row = int(rng.integers(0, h))
    mask[row, 0] = True
    for col in range(1, w):
        prev_row = row
        row = int(np.clip(prev_row + rng.integers(-1, 2), 0, h - 1))
        mask[prev_row, col] = True
        mask[row, col] = True

    col = int(rng.integers(0, w))
    mask[0, col] = True
    for row in range(1, h):
        prev_col = col
        col = int(np.clip(prev_col + rng.integers(-1, 2), 0, w - 1))
        mask[row, prev_col] = True
        mask[row, col] = True

    if _components_count(mask) > 1:
        a = np.argwhere(mask[:, 0] | mask[:, -1] | mask[0, :] | mask[-1, :])[0]
        points = np.argwhere(mask)
        source = tuple(int(v) for v in points[0])
        target = tuple(int(v) for v in points[-1])
        if rng.random() < 0.5:
            for cc in range(min(source[1], target[1]), max(source[1], target[1]) + 1):
                mask[source[0], cc] = True
            for rr in range(min(source[0], target[0]), max(source[0], target[0]) + 1):
                mask[rr, target[1]] = True
        else:
            for rr in range(min(source[0], target[0]), max(source[0], target[0]) + 1):
                mask[rr, source[1]] = True
            for cc in range(min(source[1], target[1]), max(source[1], target[1]) + 1):
                mask[target[0], cc] = True

    assert _components_count(mask) == 1
    assert np.all(mask.sum(axis=1) >= 1)
    assert np.all(mask.sum(axis=0) >= 1)
    return mask


def _grow_blue_mask(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    mask = _make_covering_backbone(h, w, rng)
    total = h * w
    target_min = max(int(np.ceil(0.32 * total)), int(mask.sum()))
    target_max = max(target_min, int(np.floor(0.48 * total)))
    target = int(rng.integers(target_min, target_max + 1))

    while int(mask.sum()) < target:
        frontier = set()
        for r, c in np.argwhere(mask):
            for nr, nc in _neighbors4(int(r), int(c), h, w):
                if not mask[nr, nc]:
                    frontier.add((nr, nc))
        if not frontier:
            break
        nr, nc = list(frontier)[int(rng.integers(0, len(frontier)))]
        mask[nr, nc] = True

    assert _components_count(mask) == 1
    assert np.all(mask.sum(axis=1) >= 1)
    assert np.all(mask.sum(axis=0) >= 1)
    return mask


def _expand_with_blank_lines(
    mask: np.ndarray, blank_rows: list[int], blank_cols: list[int]
) -> np.ndarray:
    h = mask.shape[0] + len(blank_rows)
    w = mask.shape[1] + len(blank_cols)
    grid = np.zeros((h, w), dtype=int)
    active_rows = [r for r in range(h) if r not in set(blank_rows)]
    active_cols = [c for c in range(w) if c not in set(blank_cols)]
    row_idx = np.array(active_rows, dtype=int)
    col_idx = np.array(active_cols, dtype=int)
    grid[np.ix_(row_idx, col_idx)] = mask.astype(int) * BLUE
    return grid


def _safe_to_replace(
    grid: np.ndarray,
    row_blue_counts: np.ndarray,
    col_blue_counts: np.ndarray,
    r: int,
    c: int,
    reserved: set[tuple[int, int]],
) -> bool:
    if (r, c) in reserved:
        return False
    value = int(grid[r, c])
    if value not in (BACKGROUND, BLUE):
        return False
    if value == BLUE and (row_blue_counts[r] <= 1 or col_blue_counts[c] <= 1):
        return False
    return True


def _choose_from_center_band(
    values: list[int], length: int, rng: np.random.Generator, band: int = 3
) -> int:
    ordered = sorted(values, key=lambda idx: (abs(idx - (length - 1) / 2.0), idx))
    return int(ordered[int(rng.integers(0, min(len(ordered), band)))])


def _place_encoders(
    grid: np.ndarray,
    blank_rows: list[int],
    blank_cols: list[int],
    row_colors: dict[int, int],
    col_colors: dict[int, int],
    rng: np.random.Generator,
) -> bool:
    reserved: set[tuple[int, int]] = set()
    row_blue_counts = np.sum(grid == BLUE, axis=1).astype(int)
    col_blue_counts = np.sum(grid == BLUE, axis=0).astype(int)

    for row in blank_rows:
        zero_candidates = []
        fallback_candidates = []
        for col in range(grid.shape[1]):
            if col in blank_cols:
                continue
            coords = [(row - 1, col), (row + 1, col)]
            if all(
                _safe_to_replace(grid, row_blue_counts, col_blue_counts, r, c, reserved)
                for r, c in coords
            ):
                if all(int(grid[r, c]) == BACKGROUND for r, c in coords):
                    zero_candidates.append(col)
                else:
                    fallback_candidates.append(col)
        candidates = zero_candidates or fallback_candidates
        if not candidates:
            return False
        col = _choose_from_center_band(candidates, grid.shape[1], rng)
        color = row_colors[row]
        for rr in (row - 1, row + 1):
            if int(grid[rr, col]) == BLUE:
                row_blue_counts[rr] -= 1
                col_blue_counts[col] -= 1
            grid[rr, col] = color
            reserved.add((rr, col))

    for col in blank_cols:
        zero_candidates = []
        fallback_candidates = []
        for row in range(grid.shape[0]):
            if row in blank_rows:
                continue
            coords = [(row, col - 1), (row, col + 1)]
            if all(
                _safe_to_replace(grid, row_blue_counts, col_blue_counts, r, c, reserved)
                for r, c in coords
            ):
                if all(int(grid[r, c]) == BACKGROUND for r, c in coords):
                    zero_candidates.append(row)
                else:
                    fallback_candidates.append(row)
        candidates = zero_candidates or fallback_candidates
        if not candidates:
            return False
        row = _choose_from_center_band(candidates, grid.shape[0], rng)
        color = col_colors[col]
        for cc in (col - 1, col + 1):
            if int(grid[row, cc]) == BLUE:
                row_blue_counts[row] -= 1
                col_blue_counts[cc] -= 1
            grid[row, cc] = color
            reserved.add((row, cc))

    return True


def _sample_line_colors(
    rng: np.random.Generator, n_rows: int, n_cols: int
) -> tuple[dict[int, int], dict[int, int]]:
    total = n_rows + n_cols
    while True:
        colors = [int(rng.choice(ENCODER_COLORS)) for _ in range(total)]
        if total >= 2 and len(set(colors)) < 2:
            continue
        row_vals = colors[:n_rows]
        col_vals = colors[n_rows:]
        if n_rows and n_cols and len(set(row_vals + col_vals)) == 1:
            continue
        return (
            {idx: color for idx, color in enumerate(row_vals)},
            {idx: color for idx, color in enumerate(col_vals)},
        )


def _compressed_blue_is_valid(grid: np.ndarray) -> bool:
    blank_rows, blank_cols = _blank_lines(grid)
    active_rows = [r for r in range(grid.shape[0]) if r not in blank_rows]
    active_cols = [c for c in range(grid.shape[1]) if c not in blank_cols]
    cropped = grid[np.ix_(active_rows, active_cols)]
    blue_mask = cropped == BLUE
    if not np.all(blue_mask.sum(axis=1) >= 1):
        return False
    if not np.all(blue_mask.sum(axis=0) >= 1):
        return False
    return _components_count(blue_mask) == 1


def _validate_semantics(
    grid: np.ndarray,
    blank_rows: list[int],
    blank_cols: list[int],
    expected_row_colors: dict[int, int],
    expected_col_colors: dict[int, int],
) -> bool:
    found_rows, found_cols = _blank_lines(grid)
    if found_rows != blank_rows or found_cols != blank_cols:
        return False
    if not _compressed_blue_is_valid(grid):
        return False

    for row in blank_rows:
        candidates = _row_candidates(grid, row)
        if len(candidates) != 1:
            return False
        if _decode_row_color(grid, row) != expected_row_colors[row]:
            return False
    for col in blank_cols:
        candidates = _col_candidates(grid, col)
        if len(candidates) != 1:
            return False
        if _decode_col_color(grid, col) != expected_col_colors[col]:
            return False

    out = main(grid)
    expected = _expected_output(grid, expected_row_colors, expected_col_colors)
    if not np.array_equal(out, expected):
        return False

    for row, color in expected_row_colors.items():
        if not np.all(out[row] == color):
            return False
    for col, color in expected_col_colors.items():
        for row in blank_rows:
            if out[row, col] != expected_row_colors[row]:
                return False
        for row in range(grid.shape[0]):
            if row not in blank_rows and out[row, col] != color:
                return False

    active_area = (grid.shape[0] - len(blank_rows)) * (grid.shape[1] - len(blank_cols))
    blue_density = float(np.sum(grid == BLUE)) / float(active_area)
    if not (0.22 <= blue_density <= 0.50):
        return False
    if len(blank_rows) + len(blank_cols) >= 2:
        used_colors = set(expected_row_colors.values()) | set(expected_col_colors.values())
        if len(used_colors) < 2:
            return False
    return True


def _sample_branch_counts(
    rng: np.random.Generator,
) -> tuple[int, int]:
    branch = int(rng.integers(0, 3))
    if branch == 0:
        return int(rng.integers(2, 5)), 0
    if branch == 1:
        return 0, int(rng.integers(1, 4))
    return int(rng.integers(2, 5)), int(rng.integers(1, 4))


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_blank_rows, n_blank_cols = _sample_branch_counts(rng)
    h = int(rng.integers(max(12, n_blank_rows * 3 + 3), 19))
    w = int(rng.integers(max(12, n_blank_cols * 3 + 4), 19))

    blank_rows = _sample_spaced_positions(rng, h, n_blank_rows, border=1, min_gap=3)
    blank_cols = _sample_spaced_positions(rng, w, n_blank_cols, border=1, min_gap=3)
    if blank_rows is None or blank_cols is None:
        return None

    active_h = h - n_blank_rows
    active_w = w - n_blank_cols
    if active_h < 7 or active_w < 7:
        return None

    blue_mask = _grow_blue_mask(active_h, active_w, rng)
    grid = _expand_with_blank_lines(blue_mask, blank_rows, blank_cols)

    row_color_seed, col_color_seed = _sample_line_colors(rng, n_blank_rows, n_blank_cols)
    row_colors = {row: row_color_seed[i] for i, row in enumerate(blank_rows)}
    col_colors = {col: col_color_seed[i] for i, col in enumerate(blank_cols)}

    if not _place_encoders(grid, blank_rows, blank_cols, row_colors, col_colors, rng):
        return None
    if not _validate_semantics(grid, blank_rows, blank_cols, row_colors, col_colors):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
