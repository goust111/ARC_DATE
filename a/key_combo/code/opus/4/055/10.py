from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, bbox_h, bbox_w, find_components, get_bg


MARKER_COLORS = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW)
GEN_ATTEMPTS = 500
INTERIOR_ROW_RANGE = (3, 7)
INTERIOR_COL_RANGE = (3, 8)
OFFSET_RANGE = (1, 6)
SLACK_RANGE = (0, 6)


def _is_rect_frame(component: dict) -> bool:
    r0, c0, r1, c1 = component["bbox"]
    h = bbox_h(component["bbox"])
    w = bbox_w(component["bbox"])
    if h < 3 or w < 3:
        return False

    rel = component["coords"] - np.array([[r0, c0]])
    mask = np.zeros((h, w), dtype=bool)
    mask[rel[:, 0], rel[:, 1]] = True

    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(np.array_equal(mask, border))


def _find_frame(grid: np.ndarray) -> dict:
    bg = get_bg(grid)
    components = find_components(
        grid,
        bg=bg,
        connectivity=4,
        colors=[Color.GREY],
        monochromatic=True,
        return_format="dict",
    )
    frames = [comp for comp in components if _is_rect_frame(comp)]
    assert len(frames) == 1, "expected exactly one hollow grey frame"
    return frames[0]


def _extract_markers(grid: np.ndarray, frame_bbox: tuple[int, int, int, int]) -> tuple[list[int], list[int]]:
    r0, c0, r1, c1 = frame_bbox
    assert r0 > 0 and c0 > 0, "frame needs room for top and left markers"
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame interior must be non-empty"

    row_markers = [int(x) for x in grid[r0 + 1 : r1, c0 - 1]]
    col_markers = [int(x) for x in grid[r0 - 1, c0 + 1 : c1]]
    assert row_markers and col_markers, "expected row and column marker pixels"
    assert all(color in MARKER_COLORS for color in row_markers + col_markers), "unexpected marker color"
    return row_markers, col_markers


def _marker_counts(row_markers: list[int], col_markers: list[int]) -> Counter:
    counts = Counter(row_markers + col_markers)
    assert len(counts) in (2, 3), "markers must use 2-3 colors"
    assert len(set(counts.values())) == len(counts), "marker color frequencies must be unique"
    return counts


def _resolved_fill(row_markers: list[int], col_markers: list[int], counts: Counter) -> np.ndarray:
    out = np.zeros((len(row_markers), len(col_markers)), dtype=int)
    for r, row_color in enumerate(row_markers):
        for c, col_color in enumerate(col_markers):
            if row_color == col_color:
                out[r, c] = row_color
            else:
                out[r, c] = row_color if counts[row_color] > counts[col_color] else col_color
    return out


def _priority_fill(row_markers: list[int], col_markers: list[int], prefer: str) -> np.ndarray:
    out = np.zeros((len(row_markers), len(col_markers)), dtype=int)
    for r, row_color in enumerate(row_markers):
        for c, col_color in enumerate(col_markers):
            if row_color == col_color:
                out[r, c] = row_color
            elif prefer == "row":
                out[r, c] = row_color
            else:
                out[r, c] = col_color
    return out


def _dominant_conflict_fill(row_markers: list[int], col_markers: list[int], counts: Counter) -> np.ndarray:
    dominant = max(counts, key=counts.get)
    out = np.zeros((len(row_markers), len(col_markers)), dtype=int)
    for r, row_color in enumerate(row_markers):
        for c, col_color in enumerate(col_markers):
            out[r, c] = row_color if row_color == col_color else dominant
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Fill the inside of the hollow grey frame by combining aligned row and column marker constraints.

    Input structure:
    - The background is black.
    - Exactly one grey one-pixel-thick hollow rectangle encloses a black interior.
    - One colored marker pixel sits above each interior column and one colored marker pixel sits left of each interior row.

    Rule:
    - Keep the grey frame and all marker pixels unchanged.
    - For each interior cell, read the color of its aligned row marker and aligned column marker.
    - If the two marker colors match, fill the cell with that color; otherwise compare their global marker frequencies and use the more frequent color.

    Key insight:
    - Every interior cell is determined by the intersection of a row-local and a column-local constraint, while conflicts are resolved only by the global frequency distribution of all markers.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    assert bg == Color.BLACK

    frame = _find_frame(g)
    r0, c0, r1, c1 = frame["bbox"]
    row_markers, col_markers = _extract_markers(g, frame["bbox"])
    counts = _marker_counts(row_markers, col_markers)

    interior = g[r0 + 1 : r1, c0 + 1 : c1]
    assert interior.shape == (len(row_markers), len(col_markers))
    assert np.all(interior == bg), "frame interior must be empty in the input"

    allowed = np.zeros_like(g, dtype=bool)
    allowed[r0, c0 : c1 + 1] = True
    allowed[r1, c0 : c1 + 1] = True
    allowed[r0 : r1 + 1, c0] = True
    allowed[r0 : r1 + 1, c1] = True
    allowed[r0 + 1 : r1, c0 - 1] = True
    allowed[r0 - 1, c0 + 1 : c1] = True
    assert np.all((g == bg) | allowed), "unexpected extra foreground objects"

    out = g.copy()
    out[r0 + 1 : r1, c0 + 1 : c1] = _resolved_fill(row_markers, col_markers, counts)
    return out


def _marker_sequences_ok(row_markers: list[int], col_markers: list[int]) -> bool:
    counts = Counter(row_markers + col_markers)
    if len(counts) not in (2, 3):
        return False
    if len(set(counts.values())) != len(counts):
        return False

    row_counts = Counter(row_markers)
    col_counts = Counter(col_markers)
    equal_cells = sum(row_counts[color] * col_counts[color] for color in counts)
    total_cells = len(row_markers) * len(col_markers)
    if equal_cells < 2 or total_cells - equal_cells < 2:
        return False

    row_wins = False
    col_wins = False
    top_color = max(counts, key=counts.get)
    non_top_conflict = False
    for row_color in row_counts:
        for col_color in col_counts:
            if row_color == col_color:
                continue
            if counts[row_color] > counts[col_color]:
                row_wins = True
            if counts[col_color] > counts[row_color]:
                col_wins = True
            if row_color != top_color and col_color != top_color:
                non_top_conflict = True
    if not (row_wins and col_wins):
        return False
    if len(counts) == 3 and not non_top_conflict:
        return False
    return True


def _sample_markers(n_rows: int, n_cols: int) -> tuple[list[int], list[int]] | None:
    color_count = 2 if random.random() < 0.3 else 3
    palette = random.sample(MARKER_COLORS, color_count)
    for _ in range(300):
        row_markers = [random.choice(palette) for _ in range(n_rows)]
        col_markers = [random.choice(palette) for _ in range(n_cols)]
        if _marker_sequences_ok(row_markers, col_markers):
            return row_markers, col_markers
    return None


def _render_input(
    grid_shape: tuple[int, int],
    frame_top: int,
    frame_left: int,
    row_markers: list[int],
    col_markers: list[int],
) -> np.ndarray:
    n_rows = len(row_markers)
    n_cols = len(col_markers)
    grid = np.zeros(grid_shape, dtype=int)

    r0 = frame_top
    c0 = frame_left
    r1 = r0 + n_rows + 1
    c1 = c0 + n_cols + 1
    grid[r0, c0 : c1 + 1] = Color.GREY
    grid[r1, c0 : c1 + 1] = Color.GREY
    grid[r0 : r1 + 1, c0] = Color.GREY
    grid[r0 : r1 + 1, c1] = Color.GREY

    grid[r0 + 1 : r1, c0 - 1] = np.array(row_markers, dtype=int)
    grid[r0 - 1, c0 + 1 : c1] = np.array(col_markers, dtype=int)
    return grid


def _semantic_validation(grid: np.ndarray) -> bool:
    frame = _find_frame(grid)
    r0, c0, r1, c1 = frame["bbox"]
    row_markers, col_markers = _extract_markers(grid, frame["bbox"])
    counts = _marker_counts(row_markers, col_markers)
    fill = _resolved_fill(row_markers, col_markers, counts)

    solved = main(grid)
    if not np.array_equal(solved[r0 + 1 : r1, c0 + 1 : c1], fill):
        return False
    if np.array_equal(fill, _priority_fill(row_markers, col_markers, "row")):
        return False
    if np.array_equal(fill, _priority_fill(row_markers, col_markers, "col")):
        return False
    if len(counts) == 3 and np.array_equal(fill, _dominant_conflict_fill(row_markers, col_markers, counts)):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n_rows = random.randint(*INTERIOR_ROW_RANGE)
    n_cols = random.randint(*INTERIOR_COL_RANGE)

    frame_top = random.randint(*OFFSET_RANGE)
    frame_left = random.randint(*OFFSET_RANGE)
    bottom_slack = random.randint(*SLACK_RANGE)
    right_slack = random.randint(*SLACK_RANGE)

    grid_rows = frame_top + (n_rows + 2) + bottom_slack
    grid_cols = frame_left + (n_cols + 2) + right_slack
    if max(grid_rows, grid_cols) > 30:
        return None

    markers = _sample_markers(n_rows, n_cols)
    if markers is None:
        return None
    row_markers, col_markers = markers

    grid = _render_input((grid_rows, grid_cols), frame_top, frame_left, row_markers, col_markers)
    if not _semantic_validation(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
