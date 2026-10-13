from collections import Counter

import numpy as np

from arc_utils import components_mask, neighbors4
import random


def _unique_most_common(counts: Counter[int], candidates: list[int], role: str) -> int:
    assert candidates, f"expected at least one {role} color"
    top_count = max(counts[color] for color in candidates)
    winners = [color for color in candidates if counts[color] == top_count]
    assert len(winners) == 1, f"expected a unique {role} color"
    return winners[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each background chamber with the majority color of the colored marker pixels touching it.

    Input structure
    - A 2-D grid has one dominant background color, one wall color, and marker colors.
    - Background pixels form 4-connected chambers separated by wall and marker pixels.
    Rule
    - Infer the background as the unique most frequent color and the wall as the unique most frequent remaining color.
    - For each 4-connected background chamber, collect the distinct orthogonally adjacent marker pixels.
    - Recolor that whole chamber to the unique majority marker color, preserving all non-background pixels.
    Key insight
    - The colored pixels are boundary votes for adjacent empty regions; the grey wall structure only separates the regions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    flat_counts = Counter(map(int, g.ravel()))
    colors = list(flat_counts)
    bg = _unique_most_common(flat_counts, colors, "background")
    wall = _unique_most_common(flat_counts, [color for color in colors if color != bg], "wall")
    marker_colors = set(colors) - {bg, wall}
    assert marker_colors, "expected marker colors besides background and wall"

    out = g.copy()
    height, width = g.shape
    chambers = components_mask(g == bg, connectivity=4)
    assert chambers, "expected at least one background chamber"

    for coords in chambers:
        adjacent_markers: dict[tuple[int, int], int] = {}
        for r0, c0 in coords:
            for r, c in neighbors4(int(r0), int(c0)):
                if 0 <= r < height and 0 <= c < width and int(g[r, c]) in marker_colors:
                    adjacent_markers[(r, c)] = int(g[r, c])
        assert adjacent_markers, "each chamber must touch at least one marker"

        marker_counts = Counter(adjacent_markers.values())
        fill = _unique_most_common(marker_counts, list(marker_counts), "majority marker")
        out[coords[:, 0], coords[:, 1]] = fill

    return out

# --------------- generator (o5x) ---------------

# Grid colors and fixed canvas.
GRID_SIZE, GRID_LAST = 10, 9
BG_COLOR, WALL_COLOR = 0, 5
TARGET_COLOR_OFFSET = 1
NUM_COL_BANDS = 3

# Wall row/column layout choices.
THREE_WALL_ROW_PROB = 0.5
THREE_WALL_ROW_MIDDLES = (4, 5)
FOUR_WALL_ROWS = (0, 3, 6, 9)
WALL_COL_MIN = 1
WALL_COL_MAX = 8
WALL_COL_GAP_MIN = 3
WALL_COL_GAP_MAX = 5
WALL_COL_PAIRS = tuple(
    (c0, c0 + gap)
    for c0 in range(WALL_COL_MIN, WALL_COL_MAX + 1)
    for gap in range(WALL_COL_GAP_MIN, WALL_COL_GAP_MAX + 1)
    if c0 + gap <= WALL_COL_MAX
)

# Marker count constraints.
ROW_MARKER_MIN = 2
ROW_MARKER_MAX = 4
COL_MARKER_MIN = 1
COL_MARKER_MAX = 3
TOTAL_MARKER_MIN = 11
TOTAL_MARKER_MAX = 18

# Retry budget.
MAX_GENERATION_ATTEMPTS = 300


def _gen_wall_rows():
    if random.random() < THREE_WALL_ROW_PROB:
        return [0, random.choice(THREE_WALL_ROW_MIDDLES), GRID_LAST]
    return list(FOUR_WALL_ROWS)


def _gen_wall_cols():
    return list(random.choice(WALL_COL_PAIRS))


def _gen_bands(wall_rows, wall_cols):
    row_bands = [list(range(wall_rows[i] + 1, wall_rows[i + 1])) for i in range(len(wall_rows) - 1)]
    c0, c1 = wall_cols
    col_bands = [list(range(0, c0)), list(range(c0 + 1, c1)), list(range(c1 + 1, GRID_SIZE))]
    return row_bands, col_bands


def _gen_target_ok(target, row_bands, col_bands):
    band_rows = len(row_bands)
    for i in range(1, band_rows):
        free = sum(len(col_bands[j]) for j in range(NUM_COL_BANDS) if target[i - 1][j] == target[i][j])
        if free < 2:
            return False
    if band_rows == len(FOUR_WALL_ROWS) - 1:
        for j in range(NUM_COL_BANDS):
            if target[1][j] != target[0][j] and target[1][j] != target[2][j]:
                return False
    for j in range(NUM_COL_BANDS - 1):
        free = sum(len(row_bands[i]) for i in range(band_rows) if target[i][j] == target[i][j + 1])
        if free < 1:
            return False
    return True


def _gen_targets(row_bands, col_bands):
    band_rows = len(row_bands)
    valid = []
    for bits in range(1, (1 << (band_rows * NUM_COL_BANDS)) - 1):
        target = []
        for i in range(band_rows):
            row = []
            for j in range(NUM_COL_BANDS):
                row.append(TARGET_COLOR_OFFSET + ((bits >> (i * NUM_COL_BANDS + j)) & 1))
            target.append(row)
        if _gen_target_ok(target, row_bands, col_bands):
            valid.append(target)
    if not valid:
        return None
    return random.choice(valid)


def _gen_add_marker(grid, r, c, color, wall_rows_set, wall_cols_set, row_counts, col_counts):
    if int(grid[r, c]) != WALL_COLOR:
        return False
    if r in wall_rows_set and c in wall_cols_set:
        return False
    grid[r, c] = color
    if r in wall_rows_set:
        row_counts[r] += 1
    elif c in wall_cols_set:
        col_counts[c] += 1
    else:
        return False
    return True


def _gen_row_candidates(row_idx, wall_rows, col_bands, target):
    out = []
    for j, cols in enumerate(col_bands):
        if row_idx == 0:
            color = target[0][j]
        elif row_idx == len(wall_rows) - 1:
            color = target[-1][j]
        elif target[row_idx - 1][j] == target[row_idx][j]:
            color = target[row_idx][j]
        else:
            continue
        for c in cols:
            out.append((wall_rows[row_idx], c, color))
    random.shuffle(out)
    return out


def _gen_col_candidates(col_idx, wall_cols, row_bands, target):
    out = []
    for i, rows in enumerate(row_bands):
        if target[i][col_idx] != target[i][col_idx + 1]:
            continue
        color = target[i][col_idx]
        for r in rows:
            out.append((r, wall_cols[col_idx], color))
    random.shuffle(out)
    return out


def _gen_place_from_candidates(grid, candidates, count, wall_rows_set, wall_cols_set, row_counts, col_counts):
    placed = 0
    for r, c, color in candidates:
        if _gen_add_marker(grid, r, c, color, wall_rows_set, wall_cols_set, row_counts, col_counts):
            placed += 1
            if placed == count:
                return True
    return placed == count


def _gen_chamber_has_marker(grid, rows, cols, color, wall_rows_set, wall_cols_set):
    for c in cols:
        for r in (rows[0] - 1, rows[-1] + 1):
            if (r in wall_rows_set) ^ (c in wall_cols_set):
                if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE and int(grid[r, c]) == color:
                    return True
    for r in rows:
        for c in (cols[0] - 1, cols[-1] + 1):
            if (r in wall_rows_set) ^ (c in wall_cols_set):
                if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE and int(grid[r, c]) == color:
                    return True
    return False


def _gen_cover_options(wall_rows, wall_cols, row_bands, col_bands, target, row_counts, col_counts, i, j):
    color = target[i][j]
    rows = row_bands[i]
    cols = col_bands[j]
    options = []
    if row_counts[wall_rows[i]] < ROW_MARKER_MAX:
        options.extend((wall_rows[i], c, color) for c in cols)
    if row_counts[wall_rows[i + 1]] < ROW_MARKER_MAX:
        options.extend((wall_rows[i + 1], c, color) for c in cols)
    if j > 0 and target[i][j - 1] == color and col_counts[wall_cols[j - 1]] < COL_MARKER_MAX:
        options.extend((r, wall_cols[j - 1], color) for r in rows)
    if j < NUM_COL_BANDS - 1 and target[i][j + 1] == color and col_counts[wall_cols[j]] < COL_MARKER_MAX:
        options.extend((r, wall_cols[j], color) for r in rows)
    return options


def _gen_cover_chambers(grid, wall_rows, wall_cols, row_bands, col_bands, target, row_counts, col_counts):
    wall_rows_set = set(wall_rows)
    wall_cols_set = set(wall_cols)
    for i, rows in enumerate(row_bands):
        for j, cols in enumerate(col_bands):
            color = target[i][j]
            if _gen_chamber_has_marker(grid, rows, cols, color, wall_rows_set, wall_cols_set):
                continue
            options = _gen_cover_options(wall_rows, wall_cols, row_bands, col_bands, target, row_counts, col_counts, i, j)
            random.shuffle(options)
            ok = False
            for r, c, marker_color in options:
                if _gen_add_marker(grid, r, c, marker_color, wall_rows_set, wall_cols_set, row_counts, col_counts):
                    ok = True
                    break
            if not ok:
                return False
    return True


def _gen_fill_chambers(grid, row_bands, col_bands, target, value=None):
    for i, rows in enumerate(row_bands):
        for j, cols in enumerate(col_bands):
            fill = target[i][j] if value is None else value
            grid[np.ix_(rows, cols)] = fill


def _gen_output_grid(wall_rows, wall_cols, row_bands, col_bands, target):
    out = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    for r in wall_rows:
        out[r, :] = WALL_COLOR
    for c in wall_cols:
        out[:, c] = WALL_COLOR
    _gen_fill_chambers(out, row_bands, col_bands, target)
    return out


def _gen_seed_row_idx(i, j, wall_rows, row_bands, target, row_counts):
    if i == 0:
        return 0
    if i == len(row_bands) - 1:
        return len(wall_rows) - 1
    matches = []
    if target[i - 1][j] == target[i][j]:
        matches.append(i)
    if target[i + 1][j] == target[i][j]:
        matches.append(i + 1)
    if not matches:
        return None
    return min(matches, key=lambda idx: (row_counts[wall_rows[idx]], random.random()))


def _gen_place_seed_markers(out, wall_rows, row_bands, col_bands, target, wall_rows_set, wall_cols_set, row_counts, col_counts):
    for i, _ in enumerate(row_bands):
        for j, cols in enumerate(col_bands):
            row_idx = _gen_seed_row_idx(i, j, wall_rows, row_bands, target, row_counts)
            if row_idx is None:
                return False
            c = random.choice(cols)
            if not _gen_add_marker(out, wall_rows[row_idx], c, target[i][j], wall_rows_set, wall_cols_set, row_counts, col_counts):
                return False
    return True


def _gen_top_up_rows(out, wall_rows, col_bands, target, wall_rows_set, wall_cols_set, row_counts, col_counts):
    for row_idx, r in enumerate(wall_rows):
        need = max(0, ROW_MARKER_MIN - row_counts[r])
        if not need:
            continue
        candidates = _gen_row_candidates(row_idx, wall_rows, col_bands, target)
        if not _gen_place_from_candidates(out, candidates, need, wall_rows_set, wall_cols_set, row_counts, col_counts):
            return False
    return True


def _gen_top_up_cols(out, wall_cols, row_bands, target, wall_rows_set, wall_cols_set, row_counts, col_counts):
    for col_idx, c in enumerate(wall_cols):
        need = max(0, COL_MARKER_MIN - col_counts[c])
        if not need:
            continue
        candidates = _gen_col_candidates(col_idx, wall_cols, row_bands, target)
        if not _gen_place_from_candidates(out, candidates, need, wall_rows_set, wall_cols_set, row_counts, col_counts):
            return False
    return True


def _gen_extra_candidates(kind, idx, wall_rows, wall_cols, row_bands, col_bands, target, row_counts, col_counts):
    if kind == "row":
        r = wall_rows[idx]
        if row_counts[r] >= ROW_MARKER_MAX:
            return None
        return _gen_row_candidates(idx, wall_rows, col_bands, target)
    c = wall_cols[idx]
    if col_counts[c] >= COL_MARKER_MAX:
        return None
    return _gen_col_candidates(idx, wall_cols, row_bands, target)


def _gen_add_extra_markers(out, wall_rows, wall_cols, row_bands, col_bands, target, wall_rows_set, wall_cols_set, row_counts, col_counts):
    total = sum(row_counts.values()) + sum(col_counts.values())
    final_total = random.randint(max(TOTAL_MARKER_MIN, total), TOTAL_MARKER_MAX)
    all_lines = [("row", i) for i in range(len(wall_rows))] + [("col", i) for i in range(len(wall_cols))]
    while total < final_total:
        random.shuffle(all_lines)
        placed = False
        for kind, idx in all_lines:
            candidates = _gen_extra_candidates(kind, idx, wall_rows, wall_cols, row_bands, col_bands, target, row_counts, col_counts)
            if candidates is None:
                continue
            if _gen_place_from_candidates(out, candidates, 1, wall_rows_set, wall_cols_set, row_counts, col_counts):
                total += 1
                placed = True
                break
        if not placed:
            return None
    return total


def _gen_counts_ok(row_counts, col_counts, total):
    if not all(ROW_MARKER_MIN <= count <= ROW_MARKER_MAX for count in row_counts.values()):
        return False
    if not all(COL_MARKER_MIN <= count <= COL_MARKER_MAX for count in col_counts.values()):
        return False
    return TOTAL_MARKER_MIN <= total <= TOTAL_MARKER_MAX


def _gen_attempt():
    wall_rows = _gen_wall_rows()
    wall_cols = _gen_wall_cols()
    row_bands, col_bands = _gen_bands(wall_rows, wall_cols)
    target = _gen_targets(row_bands, col_bands)
    if target is None:
        return None

    out = _gen_output_grid(wall_rows, wall_cols, row_bands, col_bands, target)
    wall_rows_set = set(wall_rows)
    wall_cols_set = set(wall_cols)
    row_counts = {r: 0 for r in wall_rows}
    col_counts = {c: 0 for c in wall_cols}

    if not _gen_place_seed_markers(out, wall_rows, row_bands, col_bands, target, wall_rows_set, wall_cols_set, row_counts, col_counts):
        return None
    if not _gen_top_up_rows(out, wall_rows, col_bands, target, wall_rows_set, wall_cols_set, row_counts, col_counts):
        return None
    if not _gen_top_up_cols(out, wall_cols, row_bands, target, wall_rows_set, wall_cols_set, row_counts, col_counts):
        return None
    if not _gen_cover_chambers(out, wall_rows, wall_cols, row_bands, col_bands, target, row_counts, col_counts):
        return None

    total = _gen_add_extra_markers(out, wall_rows, wall_cols, row_bands, col_bands, target, wall_rows_set, wall_cols_set, row_counts, col_counts)
    if total is None or not _gen_counts_ok(row_counts, col_counts, total):
        return None

    inp = out.copy()
    _gen_fill_chambers(inp, row_bands, col_bands, target, value=BG_COLOR)
    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input():
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate valid input for 7c8af763")
