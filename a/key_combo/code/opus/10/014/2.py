from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

MAX_GENERATION_ATTEMPTS = 400
LINE_COUNT_CHOICES = (3, 4)
CELL_SPAN_RANGE = (2, 4)
PADDING_RANGE = (1, 3)
GRID_SIZE_RANGE = (10, 16)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(cells, dtype=int))
    return out


def _has_adjacent_pair(values: np.ndarray, color: int = BLUE) -> bool:
    return bool(np.any((values[:-1] == color) & (values[1:] == color)))


def _flood_exterior(walls: np.ndarray) -> np.ndarray:
    h, w = walls.shape
    exterior = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    def add(r: int, c: int) -> None:
        if not walls[r, c] and not exterior[r, c]:
            exterior[r, c] = True
            q.append((r, c))

    for r in range(h):
        add(r, 0)
        add(r, w - 1)
    for c in range(w):
        add(0, c)
        add(h - 1, c)

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w:
                add(nr, nc)

    return exterior


def _detect_marked_axes(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[list[int], list[int]]:
    r0, c0, r1, c1 = bbox
    line_rows = [r for r in range(r0, r1 + 1) if _has_adjacent_pair(grid[r, c0 : c1 + 1], BLUE)]
    line_cols = [c for c in range(c0, c1 + 1) if _has_adjacent_pair(grid[r0 : r1 + 1, c], BLUE)]
    return line_rows, line_cols


def _complete_lines(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    line_rows: list[int],
    line_cols: list[int],
) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    out = np.asarray(grid, dtype=int).copy()
    for r in line_rows:
        row = out[r, c0 : c1 + 1]
        row[row == BLACK] = RED
    for c in line_cols:
        col = out[r0 : r1 + 1, c]
        col[col == BLACK] = RED
    return out


def _enclosed_regions(completed: np.ndarray) -> list[np.ndarray]:
    walls = (completed == BLUE) | (completed == RED)
    exterior = _flood_exterior(walls)
    enclosed = (~walls) & (~exterior)
    return _components(enclosed)


def _region_boundary_touches(completed: np.ndarray, region: np.ndarray, color: int) -> bool:
    h, w = completed.shape
    region_set = {tuple(map(int, rc)) for rc in region}
    for r, c in region_set:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in region_set:
                if int(completed[nr, nc]) == color:
                    return True
    return False


def _expected_output_from_plan(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    line_rows: list[int],
    line_cols: list[int],
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    completed = _complete_lines(grid, bbox, line_rows, line_cols)
    out = completed.copy()
    regions = _enclosed_regions(completed)
    for region in regions:
        fill = YELLOW if _region_boundary_touches(completed, region, RED) else GREEN
        out[region[:, 0], region[:, 1]] = fill
    return out, completed, regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the broken blue grid lines, then fill naturally closed chambers green and repair-closed chambers yellow.

    Input structure:
    - The grid has a black background and a foreground bbox containing interrupted blue horizontal and vertical grid lines.
    - A valid horizontal line row contains at least one horizontally adjacent blue pair inside the foreground bbox, and a valid vertical line column contains at least one vertically adjacent blue pair there.
    - Non-valid rows or columns may still contain isolated blue pixels coming from the opposite line orientation, but those singletons do not define full lines.

    Rule:
    - Find the foreground bbox of all blue pixels.
    - Mark every bbox row with a horizontal blue adjacency and every bbox column with a vertical blue adjacency as valid lines.
    - On each valid line, recolor every black gap cell inside the bbox to red while preserving all original blue pixels.
    - Treat the union of blue and red pixels as walls, flood-fill background from the grid boundary, and then recolor each remaining enclosed background region green if its boundary uses only original blue walls, or yellow if any boundary wall cell is a red repair pixel.

    Key insight:
    - Adjacent blue pairs identify which rows and columns are true grid lines; once those lines are repaired, red repair pixels reveal exactly which enclosed chambers depended on the completion step to become sealed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))).issubset({BLACK, BLUE}), "input must contain only black and blue"

    bbox = _bbox(g == BLUE)
    assert bbox is not None, "expected blue foreground lines"

    line_rows, line_cols = _detect_marked_axes(g, bbox)
    assert line_rows and line_cols, "expected at least one valid horizontal line and one valid vertical line"

    completed = _complete_lines(g, bbox, line_rows, line_cols)
    out = completed.copy()
    regions = _enclosed_regions(completed)
    assert regions, "expected at least one enclosed region after completion"
    for region in regions:
        fill = YELLOW if _region_boundary_touches(completed, region, RED) else GREEN
        out[region[:, 0], region[:, 1]] = fill
    return out


def _relative_line_positions(cell_spans: list[int]) -> list[int]:
    positions = [0]
    for span in cell_spans:
        positions.append(positions[-1] + span + 1)
    return positions


def _boundary_segments(cell_rows: int, cell_cols: int) -> list[tuple[str, int, int]]:
    segments: list[tuple[str, int, int]] = []
    for j in range(cell_cols):
        segments.append(("top", 0, j))
        segments.append(("bottom", cell_rows - 1, j))
    for i in range(cell_rows):
        segments.append(("left", i, 0))
        segments.append(("right", i, cell_cols - 1))
    return segments


def _green_protected_segments(green_cell: tuple[int, int], cell_rows: int, cell_cols: int) -> set[tuple[str, int, int]]:
    i, j = green_cell
    protected: set[tuple[str, int, int]] = set()
    if i == 0:
        protected.add(("top", i, j))
    if i == cell_rows - 1:
        protected.add(("bottom", i, j))
    if j == 0:
        protected.add(("left", i, j))
    if j == cell_cols - 1:
        protected.add(("right", i, j))
    return protected


def _segment_cells(
    line_rows: list[int],
    line_cols: list[int],
    segment: tuple[str, int, int],
) -> list[tuple[int, int]]:
    side, i, j = segment
    if side == "top":
        r = line_rows[i]
        return [(r, c) for c in range(line_cols[j] + 1, line_cols[j + 1])]
    if side == "bottom":
        r = line_rows[i + 1]
        return [(r, c) for c in range(line_cols[j] + 1, line_cols[j + 1])]
    if side == "left":
        c = line_cols[j]
        return [(r, c) for r in range(line_rows[i] + 1, line_rows[i + 1])]
    assert side == "right"
    c = line_cols[j + 1]
    return [(r, c) for r in range(line_rows[i] + 1, line_rows[i + 1])]


def _choose_gap_slice(segment_cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    assert segment_cells, "gap segment must have at least one non-corner cell"
    width = random.randint(1, min(2, len(segment_cells)))
    start = random.randint(0, len(segment_cells) - width)
    return segment_cells[start : start + width]


def _cell_center(line_rows: list[int], line_cols: list[int], cell: tuple[int, int]) -> tuple[int, int]:
    i, j = cell
    rr = (line_rows[i] + line_rows[i + 1]) // 2
    cc = (line_cols[j] + line_cols[j + 1]) // 2
    return int(rr), int(cc)


def _pick_yellow_cell(side: str, cell_rows: int, cell_cols: int) -> tuple[int, int]:
    if side == "top":
        return 0, random.randrange(cell_cols)
    if side == "bottom":
        return cell_rows - 1, random.randrange(cell_cols)
    if side == "left":
        return random.randrange(cell_rows), 0
    return random.randrange(cell_rows), cell_cols - 1


def _green_candidates(side: str, cell_rows: int, cell_cols: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for i in range(cell_rows):
        for j in range(cell_cols):
            if side == "top" and i == 0:
                continue
            if side == "bottom" and i == cell_rows - 1:
                continue
            if side == "left" and j == 0:
                continue
            if side == "right" and j == cell_cols - 1:
                continue
            out.append((i, j))
    return out


def _build_candidate():
    row_line_count = random.choice(LINE_COUNT_CHOICES)
    col_line_count = random.choice(LINE_COUNT_CHOICES)
    row_spans = [random.randint(*CELL_SPAN_RANGE) for _ in range(row_line_count - 1)]
    col_spans = [random.randint(*CELL_SPAN_RANGE) for _ in range(col_line_count - 1)]

    row_rel = _relative_line_positions(row_spans)
    col_rel = _relative_line_positions(col_spans)
    bbox_h = row_rel[-1] + 1
    bbox_w = col_rel[-1] + 1

    top_pad = random.randint(*PADDING_RANGE)
    bottom_pad = random.randint(*PADDING_RANGE)
    left_pad = random.randint(*PADDING_RANGE)
    right_pad = random.randint(*PADDING_RANGE)
    h = bbox_h + top_pad + bottom_pad
    w = bbox_w + left_pad + right_pad
    if not (GRID_SIZE_RANGE[0] <= h <= GRID_SIZE_RANGE[1]):
        return None
    if not (GRID_SIZE_RANGE[0] <= w <= GRID_SIZE_RANGE[1]):
        return None

    line_rows = [top_pad + r for r in row_rel]
    line_cols = [left_pad + c for c in col_rel]
    bbox = (line_rows[0], line_cols[0], line_rows[-1], line_cols[-1])
    grid = np.zeros((h, w), dtype=int)

    r0, c0, r1, c1 = bbox
    for r in line_rows:
        grid[r, c0 : c1 + 1] = BLUE
    for c in line_cols:
        grid[r0 : r1 + 1, c] = BLUE

    cell_rows = len(line_rows) - 1
    cell_cols = len(line_cols) - 1
    side = random.choice(("top", "bottom", "left", "right"))
    yellow_cell = _pick_yellow_cell(side, cell_rows, cell_cols)
    green_cell = random.choice(_green_candidates(side, cell_rows, cell_cols))

    protected = _green_protected_segments(green_cell, cell_rows, cell_cols)
    primary_segment = (side, yellow_cell[0], yellow_cell[1])
    segments = [primary_segment]

    extras = [
        segment
        for segment in _boundary_segments(cell_rows, cell_cols)
        if segment not in protected and segment != primary_segment
    ]
    random.shuffle(extras)
    for segment in extras[: random.randint(0, min(2, len(extras)))]:
        segments.append(segment)

    used: set[tuple[int, int]] = set()
    for segment in segments:
        cells = _segment_cells(line_rows, line_cols, segment)
        gap_cells = _choose_gap_slice(cells)
        if any(cell in used for cell in gap_cells):
            continue
        used.update(gap_cells)
        for r, c in gap_cells:
            grid[r, c] = BLACK

    return grid, bbox, line_rows, line_cols, green_cell, yellow_cell


def _validate_candidate(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    line_rows: list[int],
    line_cols: list[int],
    green_cell: tuple[int, int],
    yellow_cell: tuple[int, int],
) -> bool:
    actual_bbox = _bbox(grid == BLUE)
    if actual_bbox != bbox:
        return False

    detected_rows, detected_cols = _detect_marked_axes(grid, bbox)
    if detected_rows != line_rows or detected_cols != line_cols:
        return False

    expected, completed, regions = _expected_output_from_plan(grid, bbox, line_rows, line_cols)
    if not regions:
        return False

    green_rep = _cell_center(line_rows, line_cols, green_cell)
    yellow_rep = _cell_center(line_rows, line_cols, yellow_cell)
    if int(expected[green_rep]) != GREEN or int(expected[yellow_rep]) != YELLOW:
        return False

    pre_exterior = _flood_exterior(grid == BLUE)
    if pre_exterior[green_rep]:
        return False
    if not pre_exterior[yellow_rep]:
        return False

    if not np.any(expected == GREEN) or not np.any(expected == YELLOW):
        return False
    if not np.any(completed == RED):
        return False

    non_marked_rows = [r for r in range(bbox[0], bbox[2] + 1) if r not in set(line_rows)]
    non_marked_cols = [c for c in range(bbox[1], bbox[3] + 1) if c not in set(line_cols)]
    if not any(np.any(grid[r, bbox[1] : bbox[3] + 1] == BLUE) and not _has_adjacent_pair(grid[r, bbox[1] : bbox[3] + 1]) for r in non_marked_rows):
        return False
    if not any(np.any(grid[bbox[0] : bbox[2] + 1, c] == BLUE) and not _has_adjacent_pair(grid[bbox[0] : bbox[2] + 1, c]) for c in non_marked_cols):
        return False

    for region in regions:
        coords = (region[:, 0], region[:, 1])
        touches_red = _region_boundary_touches(completed, region, RED)
        if touches_red:
            if not np.all(pre_exterior[coords]):
                return False
        else:
            if np.any(pre_exterior[coords]):
                return False

    try:
        actual = main(grid)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        candidate = _build_candidate()
        if candidate is None:
            continue
        grid, bbox, line_rows, line_cols, green_cell, yellow_cell = candidate
        if _validate_candidate(grid, bbox, line_rows, line_cols, green_cell, yellow_cell):
            return grid
    raise RuntimeError("failed to generate a valid input for ARC task 014/2")
