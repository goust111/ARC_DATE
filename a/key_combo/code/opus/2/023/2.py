from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
_ARC_UTILS_SPEC = importlib.util.spec_from_file_location("_arc_utils_shared", _ARC_UTILS_PATH)
if _ARC_UTILS_SPEC is None or _ARC_UTILS_SPEC.loader is None:
    raise ImportError(f"unable to load arc_utils from {_ARC_UTILS_PATH}")
_ARC_UTILS = importlib.util.module_from_spec(_ARC_UTILS_SPEC)
_ARC_UTILS_SPEC.loader.exec_module(_ARC_UTILS)

Color = _ARC_UTILS.Color
in_bounds = _ARC_UTILS.in_bounds
neighbors8 = _ARC_UTILS.neighbors8


MARKER_COLOR = Color.RED
OBSTACLE_COLOR = Color.GREY
BACKGROUND_COLOR = Color.BLACK
HORIZONTAL_STAMP = Color.GREEN
VERTICAL_STAMP = Color.BLUE
BOTH_STAMP = Color.YELLOW

STATUS_TO_FLAGS = {
    "horizontal": (True, False),
    "vertical": (False, True),
    "both": (True, True),
    "none": (False, False),
}
FLAGS_TO_STATUS = {flags: status for status, flags in STATUS_TO_FLAGS.items()}
FLAGS_TO_STAMP = {
    (True, False): HORIZONTAL_STAMP,
    (False, True): VERTICAL_STAMP,
    (True, True): BOTH_STAMP,
}

# Generator bounds.
HEIGHT_RANGE = (13, 18)
WIDTH_RANGE = (17, 24)
MARKER_COUNT_RANGE = (4, 8)
NOISE_COUNT_RANGE = (4, 12)
GENERATION_ATTEMPTS = 400
POSITION_ATTEMPTS = 400


def _row_is_clear(grid: np.ndarray, r: int, c: int) -> bool:
    row = grid[int(r), :]
    return bool(np.all(np.concatenate((row[:c], row[c + 1 :])) == BACKGROUND_COLOR))


def _col_is_clear(grid: np.ndarray, r: int, c: int) -> bool:
    col = grid[:, int(c)]
    return bool(np.all(np.concatenate((col[:r], col[r + 1 :])) == BACKGROUND_COLOR))


def _corridor_flags(grid: np.ndarray, r: int, c: int) -> tuple[bool, bool]:
    return _row_is_clear(grid, r, c), _col_is_clear(grid, r, c)


def _iter_neighborhood(shape: tuple[int, int], r: int, c: int):
    if in_bounds(shape, r, c):
        yield r, c
    for rr, cc in neighbors8(r, c):
        if in_bounds(shape, rr, cc):
            yield rr, cc


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp color-coded 3x3 neighborhoods around red markers whose rows and columns form unobstructed edge-to-edge corridors.

    Input structure:
    - The grid has a black background, scattered grey obstacle pixels, and several isolated red marker pixels.
    - Each marker can test a full horizontal corridor along its row and a full vertical corridor along its column.
    - Grey obstacles remain fixed and may block those row or column corridors without forming complete separator lines.

    Rule:
    - For each red marker, the horizontal corridor exists iff every other cell in that marker's row is black, and the vertical corridor exists iff every other cell in that marker's column is black.
    - If only the horizontal corridor exists, recolor black cells in the marker's in-bounds 3x3 neighborhood green; if only the vertical corridor exists, recolor them blue; if both exist, recolor them yellow.
    - Leave the red marker itself unchanged, never overwrite grey obstacles or other non-black cells, and interpret all stamps against the original input grid.
    - If neither corridor exists, erase that red marker to black.

    Key insight:
    - A global edge-to-edge corridor test on the marker's full row and column decides both whether the marker survives and which local 3x3 stamp color it emits.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))) <= {BACKGROUND_COLOR, MARKER_COLOR, OBSTACLE_COLOR}

    markers = [tuple(map(int, coord)) for coord in np.argwhere(g == MARKER_COLOR)]
    assert markers, "expected at least one red marker"

    out = g.copy()
    desired_colors: dict[tuple[int, int], set[int]] = {}

    for r, c in markers:
        horizontal, vertical = _corridor_flags(g, r, c)
        if not horizontal and not vertical:
            out[r, c] = BACKGROUND_COLOR
            continue

        stamp_color = FLAGS_TO_STAMP[(horizontal, vertical)]
        for rr, cc in _iter_neighborhood(g.shape, r, c):
            if (rr, cc) == (r, c):
                continue
            if int(g[rr, cc]) == BACKGROUND_COLOR:
                desired_colors.setdefault((rr, cc), set()).add(stamp_color)

    for coord, colors in desired_colors.items():
        assert len(colors) == 1, "overlapping stamps with different colors would be ambiguous"
        out[coord] = next(iter(colors))

    return out


def _markers_well_spaced(markers: list[tuple[int, int]]) -> bool:
    rows = [r for r, _c in markers]
    cols = [c for _r, c in markers]
    if len(set(rows)) != len(rows) or len(set(cols)) != len(cols):
        return False
    for i, (r1, c1) in enumerate(markers):
        for r2, c2 in markers[i + 1 :]:
            if abs(r1 - r2) + abs(c1 - c2) < 3:
                return False
            if max(abs(r1 - r2), abs(c1 - c2)) < 3:
                return False
    return True


def _safe_diagonal_cells(
    shape: tuple[int, int], marker: tuple[int, int], marker_rows: set[int], marker_cols: set[int]
) -> list[tuple[int, int]]:
    r, c = marker
    cells: list[tuple[int, int]] = []
    for dr in (-1, 1):
        for dc in (-1, 1):
            rr, cc = r + dr, c + dc
            if in_bounds(shape, rr, cc) and rr not in marker_rows and cc not in marker_cols:
                cells.append((rr, cc))
    return cells


def _pick_marker_positions(h: int, w: int, count: int) -> list[tuple[int, int]] | None:
    row_candidates = list(range(1, h - 1))
    col_candidates = list(range(1, w - 1))
    for _ in range(POSITION_ATTEMPTS):
        rows = random.sample(row_candidates, count)
        cols = random.sample(col_candidates, count)
        random.shuffle(rows)
        random.shuffle(cols)
        markers = list(zip(rows, cols))
        if _markers_well_spaced(markers):
            return markers
    return None


def _pick_block_coordinate(values: list[int], anchor: int) -> int:
    far = [value for value in values if abs(value - anchor) >= 3]
    return random.choice(far or values)


def _clear_row_greys(grid: np.ndarray, r: int) -> np.ndarray:
    out = grid.copy()
    row = out[int(r), :]
    row[row == OBSTACLE_COLOR] = BACKGROUND_COLOR
    return out


def _clear_col_greys(grid: np.ndarray, c: int) -> np.ndarray:
    out = grid.copy()
    col = out[:, int(c)]
    col[col == OBSTACLE_COLOR] = BACKGROUND_COLOR
    return out


def _validate_generated(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if set(map(int, np.unique(grid))) - {BACKGROUND_COLOR, MARKER_COLOR, OBSTACLE_COLOR}:
        return False

    markers = [tuple(map(int, coord)) for coord in np.argwhere(grid == MARKER_COLOR)]
    if not (4 <= len(markers) <= 8) or not _markers_well_spaced(markers):
        return False

    classifications = {marker: _corridor_flags(grid, *marker) for marker in markers}
    counts = {status: 0 for status in STATUS_TO_FLAGS}
    for flags in classifications.values():
        counts[FLAGS_TO_STATUS[flags]] += 1
    if not all(counts[status] >= 1 for status in ("horizontal", "vertical", "both", "none")):
        return False

    out = main(grid)
    if set(map(int, np.unique(out))) - {
        BACKGROUND_COLOR,
        VERTICAL_STAMP,
        MARKER_COLOR,
        HORIZONTAL_STAMP,
        BOTH_STAMP,
        OBSTACLE_COLOR,
    }:
        return False
    if not np.array_equal(out[grid == OBSTACLE_COLOR], grid[grid == OBSTACLE_COLOR]):
        return False

    claims: dict[tuple[int, int], set[int]] = {}
    seen_active_diagonal_obstacle = False

    for marker, flags in classifications.items():
        r, c = marker
        status = FLAGS_TO_STATUS[flags]
        horizontal, vertical = flags

        if status == "horizontal":
            if not np.any(grid[:, c] == OBSTACLE_COLOR):
                return False
            if _corridor_flags(_clear_col_greys(grid, c), r, c) != (True, True):
                return False
        elif status == "vertical":
            if not np.any(grid[r, :] == OBSTACLE_COLOR):
                return False
            if _corridor_flags(_clear_row_greys(grid, r), r, c) != (True, True):
                return False
        elif status == "both":
            if np.any(grid[r, :] == OBSTACLE_COLOR) or np.any(grid[:, c] == OBSTACLE_COLOR):
                return False
        else:
            if not np.any(grid[r, :] == OBSTACLE_COLOR) or not np.any(grid[:, c] == OBSTACLE_COLOR):
                return False
            if _corridor_flags(_clear_row_greys(grid, r), r, c) != (True, False):
                return False
            if _corridor_flags(_clear_col_greys(grid, c), r, c) != (False, True):
                return False

        if status == "none":
            if out[r, c] != BACKGROUND_COLOR:
                return False
            continue

        if out[r, c] != MARKER_COLOR:
            return False

        stamp_color = FLAGS_TO_STAMP[(horizontal, vertical)]
        stamp_black_count = 0
        for rr, cc in _iter_neighborhood(grid.shape, r, c):
            if (rr, cc) == (r, c):
                continue
            if int(grid[rr, cc]) == BACKGROUND_COLOR:
                stamp_black_count += 1
                claims.setdefault((rr, cc), set()).add(stamp_color)
            elif int(grid[rr, cc]) == OBSTACLE_COLOR and abs(rr - r) == 1 and abs(cc - c) == 1:
                seen_active_diagonal_obstacle = True
        if stamp_black_count < 5:
            return False

    if not seen_active_diagonal_obstacle:
        return False
    if any(len(colors) != 1 for colors in claims.values()):
        return False
    if any(int(out[r, c]) != next(iter(colors)) for (r, c), colors in claims.items()):
        return False

    return True


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(*HEIGHT_RANGE)
    w = random.randint(*WIDTH_RANGE)
    marker_count = random.randint(*MARKER_COUNT_RANGE)

    markers = _pick_marker_positions(h, w, marker_count)
    if markers is None:
        return None

    statuses = ["horizontal", "vertical", "both", "none"]
    statuses.extend(random.choice(list(STATUS_TO_FLAGS)) for _ in range(marker_count - 4))
    random.shuffle(statuses)

    marker_rows = {r for r, _c in markers}
    marker_cols = {c for _r, c in markers}
    active_markers = [
        marker for marker, status in zip(markers, statuses, strict=True) if status != "none"
    ]
    if not any(_safe_diagonal_cells((h, w), marker, marker_rows, marker_cols) for marker in active_markers):
        return None

    grid = np.full((h, w), BACKGROUND_COLOR, dtype=int)
    for r, c in markers:
        grid[r, c] = MARKER_COLOR

    spare_rows = [r for r in range(h) if r not in marker_rows]
    spare_cols = [c for c in range(w) if c not in marker_cols]
    if not spare_rows or not spare_cols:
        return None

    row_block_pool = spare_cols.copy()
    col_block_pool = spare_rows.copy()
    random.shuffle(row_block_pool)
    random.shuffle(col_block_pool)

    for (r, c), status in zip(markers, statuses, strict=True):
        horizontal, vertical = STATUS_TO_FLAGS[status]
        if not horizontal:
            choices = row_block_pool or spare_cols
            block_c = _pick_block_coordinate(choices, c)
            grid[r, block_c] = OBSTACLE_COLOR
            if block_c in row_block_pool:
                row_block_pool.remove(block_c)
        if not vertical:
            choices = col_block_pool or spare_rows
            block_r = _pick_block_coordinate(choices, r)
            grid[block_r, c] = OBSTACLE_COLOR
            if block_r in col_block_pool:
                col_block_pool.remove(block_r)

    diagonal_hosts = [
        marker
        for marker, status in zip(markers, statuses, strict=True)
        if status != "none" and _safe_diagonal_cells((h, w), marker, marker_rows, marker_cols)
    ]
    random.shuffle(diagonal_hosts)
    diagonal_target = random.randint(1, min(3, len(diagonal_hosts)))
    diagonal_placed = 0
    for marker in diagonal_hosts:
        candidates = [
            cell
            for cell in _safe_diagonal_cells((h, w), marker, marker_rows, marker_cols)
            if int(grid[cell]) == BACKGROUND_COLOR
        ]
        if not candidates:
            continue
        grid[random.choice(candidates)] = OBSTACLE_COLOR
        diagonal_placed += 1
        if diagonal_placed == diagonal_target:
            break
    if diagonal_placed == 0:
        return None

    safe_noise_cells = [
        (r, c)
        for r in spare_rows
        for c in spare_cols
        if int(grid[r, c]) == BACKGROUND_COLOR
        and all(max(abs(r - mr), abs(c - mc)) >= 2 for mr, mc in markers)
    ]
    random.shuffle(safe_noise_cells)
    noise_count = min(random.randint(*NOISE_COUNT_RANGE), len(safe_noise_cells))
    for r, c in safe_noise_cells[:noise_count]:
        grid[r, c] = OBSTACLE_COLOR

    if not _validate_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid corridor-gated stamp input")
