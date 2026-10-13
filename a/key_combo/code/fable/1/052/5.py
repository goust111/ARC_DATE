from __future__ import annotations

import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import Color, bbox, crop_bbox, get_bg, paste


BLACK = Color.BLACK
RED = Color.RED
GREEN = Color.GREEN
MARKER_COLORS = [
    Color.BLUE,
    Color.YELLOW,
    Color.GREY,
    Color.PINK,
    Color.ORANGE,
    Color.TEAL,
    Color.MAROON,
]

DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
ORIENTATION = {"up": "v", "down": "v", "left": "h", "right": "h"}
H_FLIP_DIR = {"up": "up", "down": "down", "left": "right", "right": "left"}
V_FLIP_DIR = {"up": "down", "down": "up", "left": "left", "right": "right"}
TRANSPOSE_DIR = {"up": "left", "down": "right", "left": "up", "right": "down"}

BASE_GLYPH = np.array(
    [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
    ],
    dtype=int,
)
GAP_POS = {
    "up": (0, 1),
    "down": (2, 1),
    "left": (1, 0),
    "right": (1, 2),
}
GLYPHS: dict[str, np.ndarray] = {}
GLYPH_TO_DIR: dict[tuple[int, ...], str] = {}
for _direction, (_gr, _gc) in GAP_POS.items():
    _glyph = BASE_GLYPH.copy()
    _glyph[_gr, _gc] = 0
    GLYPHS[_direction] = _glyph
    GLYPH_TO_DIR[tuple(int(v) for v in _glyph.ravel())] = _direction


GEN_ATTEMPTS = 400
CELL_SIZE_OPTIONS = (4, 5)
SCENARIOS = ("cross_and_block", "block_only", "two_crosses")
_GEN_CALL_COUNT = 0


def _regions_between(lines: list[int], limit: int) -> list[tuple[int, int]]:
    regions: list[tuple[int, int]] = []
    prev = -1
    for line in sorted(int(v) for v in lines) + [int(limit)]:
        start = prev + 1
        end = line - 1
        if start <= end:
            regions.append((start, end))
        prev = line
    return regions


def _find_separator_grid(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    candidates = []
    for color in np.unique(g):
        color = int(color)
        if color == bg:
            continue
        full_rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == color)]
        full_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == color)]
        if not full_rows or not full_cols:
            continue
        mask = np.zeros_like(g, dtype=bool)
        mask[full_rows, :] = True
        mask[:, full_cols] = True
        if np.array_equal(g == color, mask):
            candidates.append((color, full_rows, full_cols))

    assert len(candidates) == 1, "expected one separator-grid color"
    line_color, full_rows, full_cols = candidates[0]
    row_regions = _regions_between(full_rows, g.shape[0])
    col_regions = _regions_between(full_cols, g.shape[1])
    assert len(row_regions) >= 2 and len(col_regions) >= 2, "expected a multi-cell table"
    return int(line_color), row_regions, col_regions


def _cell_bbox(row: int, col: int, cell_h: int, cell_w: int) -> tuple[int, int, int, int]:
    r0 = row * (cell_h + 1)
    c0 = col * (cell_w + 1)
    return r0, c0, r0 + cell_h - 1, c0 + cell_w - 1


def _decode_marker(cell: np.ndarray, bg: int) -> dict | None:
    fg_mask = cell != bg
    if not fg_mask.any():
        return None

    fg_colors = {int(v) for v in cell[fg_mask]}
    assert len(fg_colors) == 1, "marker cell must contain exactly one foreground color"
    color = next(iter(fg_colors))

    marker_bbox = bbox(fg_mask)
    assert marker_bbox is not None, "non-empty cell must have a bbox"
    r0, c0, r1, c1 = marker_bbox
    assert (r1 - r0 + 1, c1 - c0 + 1) == (3, 3), "marker glyph must fit in a 3x3 bbox"

    glyph_mask = crop_bbox((cell == color).astype(int), marker_bbox)
    signature = tuple(int(v) for v in glyph_mask.ravel())
    assert signature in GLYPH_TO_DIR, "unknown marker glyph"

    return {
        "color": int(color),
        "direction": GLYPH_TO_DIR[signature],
        "bbox": marker_bbox,
    }


def _extract_markers(
    grid: np.ndarray,
) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]], dict[tuple[int, int], dict]]:
    g = np.asarray(grid, dtype=int)
    line_color, row_regions, col_regions = _find_separator_grid(g)
    bg = int(get_bg(g))
    markers: dict[tuple[int, int], dict] = {}

    for row_idx, (r0, r1) in enumerate(row_regions):
        for col_idx, (c0, c1) in enumerate(col_regions):
            assert not np.any(g[r0 : r1 + 1, c0 : c1 + 1] == line_color), "separator leaked into a cell"
            cell = g[r0 : r1 + 1, c0 : c1 + 1]
            marker = _decode_marker(cell, bg)
            if marker is None:
                continue
            markers[(row_idx, col_idx)] = marker

    assert len(markers) >= 2, "expected at least two marker glyphs"
    colors = [int(info["color"]) for info in markers.values()]
    assert len(set(colors)) == len(colors), "marker colors must be distinct"
    return line_color, row_regions, col_regions, markers


def _simulate_rays(
    rows: int,
    cols: int,
    markers: dict[tuple[int, int], dict],
) -> tuple[
    dict[tuple[int, int], list[tuple[int, int]]],
    dict[tuple[int, int], list[tuple[int, int]]],
    dict[tuple[int, int], bool],
    dict[tuple[int, int], set[tuple[int, int]]],
]:
    marker_cells = set(markers)
    visits: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    paths: dict[tuple[int, int], list[tuple[int, int]]] = {}
    stopped_by_marker: dict[tuple[int, int], bool] = {}
    blocked_targets: dict[tuple[int, int], set[tuple[int, int]]] = defaultdict(set)

    for marker_cell, info in markers.items():
        dr, dc = DIRS[info["direction"]]
        rr = marker_cell[0] + dr
        cc = marker_cell[1] + dc
        ray_cells: list[tuple[int, int]] = []
        hit_marker = False

        while 0 <= rr < rows and 0 <= cc < cols:
            target = (rr, cc)
            if target in marker_cells:
                hit_marker = True
                blocked_targets[target].add(marker_cell)
                break
            visits[target].append(marker_cell)
            ray_cells.append(target)
            rr += dr
            cc += dc

        paths[marker_cell] = ray_cells
        stopped_by_marker[marker_cell] = hit_marker

    return visits, paths, stopped_by_marker, blocked_targets


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each marker glyph through table cells and paint orthogonal ray crossings red.

    Input structure:
    - Green full-row and full-column separators partition the canvas into a rectangular table.
    - Some cells contain a single 3x3 monochrome ring glyph with the center empty and exactly one missing edge midpoint.
    - All other cells are black, and each glyph color is unique.

    Rule:
    - Read the glyph's missing edge midpoint as a direction in table-cell coordinates.
    - From each glyph cell, march cell by cell in that row or column and fill every traversed non-glyph cell with the glyph color.
    - Stop before the table boundary or before the next glyph cell; if two rays traverse the same cell, fill that whole cell red instead.
    - Preserve separator lines and glyph cells exactly as they are.

    Key insight:
    - The 3x3 pixel glyph is only a direction code; the real motion happens at the coarser unit of whole table cells.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _line_color, row_regions, col_regions, markers = _extract_markers(g)
    visits, _paths, _stopped_by_marker, _blocked_targets = _simulate_rays(
        len(row_regions), len(col_regions), markers
    )

    out = g.copy()
    for row_idx, (r0, r1) in enumerate(row_regions):
        for col_idx, (c0, c1) in enumerate(col_regions):
            cell = (row_idx, col_idx)
            if cell in markers:
                continue
            sources = visits.get(cell, [])
            if not sources:
                continue
            fill_color = RED if len(sources) >= 2 else int(markers[sources[0]]["color"])
            out[r0 : r1 + 1, c0 : c1 + 1] = fill_color

    return out


def _layout_flip_h(rows: int, cols: int, markers: dict[tuple[int, int], dict]):
    flipped = {}
    for (r, c), info in markers.items():
        new_info = dict(info)
        new_info["direction"] = H_FLIP_DIR[info["direction"]]
        flipped[(r, cols - 1 - c)] = new_info
    return rows, cols, flipped


def _layout_flip_v(rows: int, cols: int, markers: dict[tuple[int, int], dict]):
    flipped = {}
    for (r, c), info in markers.items():
        new_info = dict(info)
        new_info["direction"] = V_FLIP_DIR[info["direction"]]
        flipped[(rows - 1 - r, c)] = new_info
    return rows, cols, flipped


def _layout_transpose(rows: int, cols: int, markers: dict[tuple[int, int], dict]):
    transposed = {}
    for (r, c), info in markers.items():
        new_info = dict(info)
        new_info["direction"] = TRANSPOSE_DIR[info["direction"]]
        transposed[(c, r)] = new_info
    return cols, rows, transposed


def _apply_random_symmetry(rows: int, cols: int, markers: dict[tuple[int, int], dict]):
    if random.random() < 0.5:
        rows, cols, markers = _layout_flip_h(rows, cols, markers)
    if random.random() < 0.5:
        rows, cols, markers = _layout_flip_v(rows, cols, markers)
    if random.random() < 0.5:
        rows, cols, markers = _layout_transpose(rows, cols, markers)
    return rows, cols, markers


def _base_cross_and_block() -> tuple[int, int, dict[tuple[int, int], dict]]:
    rows = random.randint(3, 5)
    cols = random.randint(4, 5)
    ray_row = random.randint(1, rows - 2)
    cross_col = random.randint(1, cols - 2)
    left_col = random.randint(0, cross_col - 1)
    right_blocker_col = random.randint(cross_col + 1, cols - 1)
    top_row = random.randint(0, ray_row - 1)

    markers = {
        (ray_row, left_col): {"direction": "right"},
        (top_row, cross_col): {"direction": "down"},
        (ray_row, right_blocker_col): {"direction": random.choice(("up", "down"))},
    }
    return rows, cols, markers


def _base_block_only() -> tuple[int, int, dict[tuple[int, int], dict]]:
    rows = random.randint(3, 5)
    cols = random.randint(3, 5)
    ray_row = random.randint(1, rows - 2)
    left_col = random.randint(0, cols - 3)
    right_blocker_col = random.randint(left_col + 2, cols - 1)

    markers = {
        (ray_row, left_col): {"direction": "right"},
        (ray_row, right_blocker_col): {"direction": random.choice(("up", "down"))},
    }
    return rows, cols, markers


def _base_two_crosses() -> tuple[int, int, dict[tuple[int, int], dict]]:
    rows = random.randint(3, 5)
    cols = 5
    ray_row = random.randint(1, rows - 2)
    cross_cols = sorted(random.sample([1, 2, 3], 2))
    left_col = random.randint(0, cross_cols[0] - 1)
    right_blocker_col = random.randint(cross_cols[1] + 1, cols - 1)
    top_row_a = random.randint(0, ray_row - 1)
    top_row_b = random.randint(0, ray_row - 1)

    markers = {
        (ray_row, left_col): {"direction": "right"},
        (top_row_a, cross_cols[0]): {"direction": "down"},
        (top_row_b, cross_cols[1]): {"direction": "down"},
        (ray_row, right_blocker_col): {"direction": random.choice(("up", "down"))},
    }
    return rows, cols, markers


def _scenario_layout(scenario: str) -> tuple[int, int, dict[tuple[int, int], dict]]:
    if scenario == "cross_and_block":
        return _base_cross_and_block()
    if scenario == "block_only":
        return _base_block_only()
    if scenario == "two_crosses":
        return _base_two_crosses()
    raise ValueError(f"unknown scenario: {scenario}")


def _assign_marker_colors(markers: dict[tuple[int, int], dict]) -> dict[tuple[int, int], dict]:
    colors = random.sample(MARKER_COLORS, len(markers))
    colored: dict[tuple[int, int], dict] = {}
    for cell, color in zip(sorted(markers), colors):
        info = markers[cell]
        colored[cell] = {
            "color": int(color),
            "direction": info["direction"],
        }
    return colored


def _layout_is_semantically_valid(
    rows: int,
    cols: int,
    markers: dict[tuple[int, int], dict],
    scenario: str,
) -> bool:
    visits, paths, stopped_by_marker, blocked_targets = _simulate_rays(rows, cols, markers)
    if not visits:
        return False

    for sources in visits.values():
        if len(sources) > 2:
            return False
        if len(sources) == 2:
            orientations = {ORIENTATION[markers[cell]["direction"]] for cell in sources}
            if len(orientations) != 2:
                return False

    red_count = sum(1 for sources in visits.values() if len(sources) == 2)
    single_count = sum(1 for sources in visits.values() if len(sources) == 1)
    if single_count == 0:
        return False
    if rows * cols - len(markers) <= len(visits):
        return False

    blocker_cells = set(blocked_targets)
    if not blocker_cells:
        return False
    if not any(path and not stopped_by_marker[cell] for cell, path in paths.items()):
        return False
    if any((not paths[cell]) and cell not in blocker_cells for cell in markers):
        return False

    if scenario == "cross_and_block":
        return red_count >= 1
    if scenario == "block_only":
        return red_count == 0
    if scenario == "two_crosses":
        return red_count >= 2
    return False


def _render_layout(
    rows: int,
    cols: int,
    cell_h: int,
    cell_w: int,
    markers: dict[tuple[int, int], dict],
) -> tuple[np.ndarray, np.ndarray]:
    grid_h = rows * cell_h + (rows - 1)
    grid_w = cols * cell_w + (cols - 1)
    inp = np.full((grid_h, grid_w), BLACK, dtype=int)

    for split_row in range(rows - 1):
        rr = (split_row + 1) * cell_h + split_row
        inp[rr, :] = GREEN
    for split_col in range(cols - 1):
        cc = (split_col + 1) * cell_w + split_col
        inp[:, cc] = GREEN

    for (row, col), info in markers.items():
        glyph = GLYPHS[info["direction"]] * int(info["color"])
        r0, c0, r1, c1 = _cell_bbox(row, col, cell_h, cell_w)
        glyph_r = r0 + random.randint(0, cell_h - 3)
        glyph_c = c0 + random.randint(0, cell_w - 3)
        paste(inp, glyph, glyph_r, glyph_c, transparent=BLACK)

    out = inp.copy()
    visits, _paths, _stopped_by_marker, _blocked_targets = _simulate_rays(rows, cols, markers)
    for (row, col), sources in visits.items():
        if (row, col) in markers:
            continue
        fill = RED if len(sources) >= 2 else int(markers[sources[0]]["color"])
        r0, c0, r1, c1 = _cell_bbox(row, col, cell_h, cell_w)
        out[r0 : r1 + 1, c0 : c1 + 1] = fill

    return inp, out


def _render_is_consistent(
    inp: np.ndarray,
    expected: np.ndarray,
    rows: int,
    cols: int,
    markers: dict[tuple[int, int], dict],
) -> bool:
    try:
        _line_color, row_regions, col_regions, parsed = _extract_markers(inp)
    except Exception:
        return False

    if len(row_regions) != rows or len(col_regions) != cols:
        return False
    parsed_simple = {
        cell: {"color": int(info["color"]), "direction": info["direction"]}
        for cell, info in parsed.items()
    }
    if parsed_simple != markers:
        return False
    if np.array_equal(inp, expected):
        return False

    predicted = main(inp)
    if not np.array_equal(predicted, expected):
        return False

    changed_cells = 0
    marker_cells = set(markers)
    for row_idx, (r0, r1) in enumerate(row_regions):
        for col_idx, (c0, c1) in enumerate(col_regions):
            if (row_idx, col_idx) in marker_cells:
                if not np.array_equal(inp[r0 : r1 + 1, c0 : c1 + 1], expected[r0 : r1 + 1, c0 : c1 + 1]):
                    return False
                continue
            if not np.array_equal(inp[r0 : r1 + 1, c0 : c1 + 1], expected[r0 : r1 + 1, c0 : c1 + 1]):
                changed_cells += 1
    return changed_cells > 0


def _gen_attempt(scenario: str) -> np.ndarray | None:
    rows, cols, markers = _scenario_layout(scenario)
    rows, cols, markers = _apply_random_symmetry(rows, cols, markers)
    markers = _assign_marker_colors(markers)

    if not _layout_is_semantically_valid(rows, cols, markers, scenario):
        return None

    cell_h = random.choice(CELL_SIZE_OPTIONS)
    cell_w = random.choice(CELL_SIZE_OPTIONS)
    inp, expected = _render_layout(rows, cols, cell_h, cell_w, markers)

    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return None
    if not _render_is_consistent(inp, expected, rows, cols, markers):
        return None
    return inp


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    scenario = SCENARIOS[_GEN_CALL_COUNT % len(SCENARIOS)]
    _GEN_CALL_COUNT += 1

    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(scenario)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input for scenario {scenario}")
