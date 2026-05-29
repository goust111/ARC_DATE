# Combo variant of 136b0064 (index 13)
# combo_spec: D4/1+D7/2
# Name: 多起点多折线分组构建

from __future__ import annotations

import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[4]
_SOLVER_DIR = _ROOT / "solvers"
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import bbox_h, bbox_w, find_components, get_bg


# Design analysis for the fused variant
# a) Behavior trace:
#    Original behavior uses one gray start and one global glyph stream in column-major order.
#    Here, the left glyph field is still arranged in two visible tile columns, but each tile
#    column is now its own instruction group. The right side therefore contains two gray starts,
#    and the solver must pair left glyph columns with starts in left-to-right order before drawing.
#    Example with N=2:
#      left glyph column 0: [area 5, area 4] and start at shifted col 1 -> one polyline.
#      left glyph column 4: [area 7, area 5] and start at shifted col 5 -> another polyline.
#    The old solver would incorrectly concatenate all four glyphs behind a single cursor.
# b) Complexity calibration:
#    The grouping dimension is intentionally small and repeated: exactly two visible glyph
#    columns and exactly two gray starts. This is visually obvious in 2-4 examples and avoids
#    ambiguous many-to-many matching.
# c) Broken hidden assumptions from the original solver:
#    - "there is exactly one gray start" is no longer valid.
#    - "all glyphs share one cursor" is no longer valid.
#    - "global column-major order gives one instruction stream" becomes "per-column top-down
#      order gives one instruction stream per group".
#    The new solver therefore first extracts starts, then groups glyphs by c0, then draws each
#    group independently.
# d) Placement legality:
#    The generator uses output-first sampling with collision checks. Each path must stay within
#    the 7-column canvas, below the bottom boundary, and avoid cells already occupied by the
#    other path. Start columns are distinct, so gray markers never overlap either.
# e) Visual observability:
#    A human can see two gray markers on the right and two vertical stacks of glyphs on the left.
#    Sorting both by left-to-right order is a visible relation; no hidden color ordering or
#    numeric-color trick is used.
# f) Visual neatness:
#    The generator rejects path pairs whose painted cells intersect, so the output remains as two
#    clean polylines instead of an unreadable overwrite soup.
# g) Inferability of changes:
#    Colors remain bound to glyph shapes exactly as in the base task. The only new inference is
#    grouping: one start per glyph column, one polyline per group. Because both groups reuse the
#    same small instruction alphabet, examples contain repeated support for the rule.


def find_full_height_constant_col(grid: np.ndarray, *, bg: int | None = None) -> tuple[int | None, int | None]:
    """Find a full-height constant non-background column."""
    H, W = grid.shape
    if bg is None:
        bg = get_bg(grid)
    bg = int(bg)

    for c in range(int(W)):
        col = grid[:, int(c)]
        if (col == col[0]).all() and int(col[0]) != bg:
            return int(c), int(col[0])
    return None, None


def _extract_groups_and_starts(
    g: np.ndarray, div_c: int, div_color: int | None, bg: int
) -> tuple[list[tuple[int, int]], list[list[dict[str, int]]]]:
    """Parse gray starts on the right and 3x3 glyph groups on the left."""
    starts: list[tuple[int, int]] = []
    for r, c in np.argwhere(g == 5):
        shifted_c = int(c) - (int(div_c) + 1)
        if 0 <= shifted_c < int(div_c):
            starts.append((int(r), shifted_c))
    starts.sort(key=lambda rc: (rc[1], rc[0]))

    left = g[:, : int(div_c)]
    ignore = {int(bg), 5}
    if div_color is not None:
        ignore.add(int(div_color))

    grouped: dict[int, list[dict[str, int]]] = defaultdict(list)
    for comp in find_components(left, bg=bg, connectivity=8, monochromatic=True, return_format="dict"):
        color = int(comp["color"])
        if color in ignore:
            continue
        b = comp["bbox"]
        if bbox_h(b) != 3 or bbox_w(b) != 3:
            continue
        r0, c0, _r1, _c1 = map(int, b)
        grouped[int(c0)].append(
            {"color": color, "r0": int(r0), "c0": int(c0), "area": int(comp["area"])}
        )

    glyph_groups: list[list[dict[str, int]]] = []
    for c0 in sorted(grouped):
        glyphs = grouped[int(c0)]
        glyphs.sort(key=lambda d: int(d["r0"]))
        glyph_groups.append(glyphs)

    return starts, glyph_groups


def _draw_group(
    out: np.ndarray,
    start: tuple[int, int],
    glyphs: list[dict[str, int]],
    area_to_len: dict[int, int],
    area_to_dir: dict[int, int],
) -> None:
    """Draw one grouped polyline from one start."""
    H, W = out.shape
    r_cur, c_cur = map(int, start)
    out[r_cur, c_cur] = 5

    for glyph in glyphs:
        color = int(glyph["color"])
        area = int(glyph["area"])
        seg_len = int(area_to_len.get(area, 2))

        r_cur += 1
        assert r_cur < H, f"segment exits bottom: r_cur={r_cur}, H={H}"
        out[r_cur, c_cur] = color

        if area == 4:
            end_r = r_cur + (seg_len - 1)
            assert end_r < H, f"vertical segment exits bottom: end_r={end_r}, H={H}"
            for _ in range(seg_len - 1):
                r_cur += 1
                out[r_cur, c_cur] = color
        else:
            dir_h = int(area_to_dir.get(area, 1))
            end_c = c_cur + dir_h * (seg_len - 1)
            assert 0 <= end_c < W, (
                f"horizontal segment exits canvas: c_cur={c_cur}, dir={dir_h}, "
                f"seg_len={seg_len}, W={W}"
            )
            for _ in range(seg_len - 1):
                c_cur += dir_h
                out[r_cur, c_cur] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw multiple grouped polylines by decoding 3x3 glyph columns against multiple starts.

    Input structure:
    - Left of a full-height vertical divider: two visible columns of 3x3 colored glyphs on bg.
    - Right of divider: multiple gray (5) start pixels, one for each glyph column group.

    Rule:
    - Remove the divider and shift every gray start into the left canvas.
    - Group glyphs by their tile column; within each group, read glyphs top-to-bottom.
    - Pair glyph columns with gray starts in left-to-right order, then draw one independent
      polyline per pair using the base instruction mapping:
      area 4 -> down 2, area 5 -> right 3, area 6 -> left 4, area 7 -> left 2.
    - Each polyline still steps down one row before executing each segment, but groups do not
      share a cursor; they coexist on the same output canvas.

    Key insight:
    The shape-to-segment code is unchanged, so the new difficulty is not decoding a glyph but
    discovering that the glyph stream has been split into parallel column groups, each launched
    from its own gray start.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    div_c, div_color = find_full_height_constant_col(g, bg=bg)
    assert div_c is not None

    out = np.zeros((H, int(div_c)), dtype=int)
    starts, glyph_groups = _extract_groups_and_starts(g, int(div_c), div_color, bg)
    assert starts, "expected at least one gray start"
    assert len(starts) == len(glyph_groups), (
        f"start/group mismatch: starts={len(starts)}, groups={len(glyph_groups)}"
    )

    area_to_len = {4: 2, 5: 3, 6: 4, 7: 2}
    area_to_dir = {5: 1, 6: -1, 7: -1}

    for start, glyphs in zip(starts, glyph_groups):
        _draw_group(out, start, glyphs, area_to_len, area_to_dir)

    return out


# --- canvas geometry ---
N_MIN, N_MAX = 2, 5
GRID_W = 15
DIVIDER_COL = 7
DIVIDER_COLOR = 4
LEFT_W = 7
TILE_COLS = (0, 4)
TILE_ROW_STRIDE = 4
GRAY = 5
GRAY_START_ROW = 0
NUM_GROUPS = 2

# --- shape grammar ---
_TEMPLATES = {
    4: np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0]], dtype=int),
    5: np.array([[1, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=int),
    6: np.array([[1, 1, 1], [0, 1, 0], [1, 0, 1]], dtype=int),
    7: np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=int),
}
_AREA_TO_COLOR = {4: 6, 5: 1, 6: 3, 7: 2}
_AREA_TO_LEN = {4: 2, 5: 3, 6: 4, 7: 2}
_AREA_TO_DIR = {4: 0, 5: 1, 6: -1, 7: -1}
_AREAS = (4, 5, 6, 7)

# --- generation constraints ---
MIN_DIRECTION_CHANGES_PER_PATH = 1
MAX_ATTEMPTS = 600


def _segment_cells(r: int, c: int, area: int) -> tuple[list[tuple[int, int]], tuple[int, int]] | None:
    """Return painted cells for one instruction and the new cursor position."""
    seg_len = int(_AREA_TO_LEN[area])
    dir_h = int(_AREA_TO_DIR[area])
    r1 = int(r + 1)
    cells = [(r1, int(c))]
    if area == 4:
        for delta in range(1, seg_len):
            cells.append((r1 + delta, int(c)))
    else:
        for delta in range(1, seg_len):
            cells.append((r1, int(c + dir_h * delta)))
    return cells, cells[-1]


def _path_direction_changes(areas: list[int]) -> int:
    is_vertical = [int(area == 4) for area in areas]
    return sum(1 for i in range(1, len(is_vertical)) if is_vertical[i] != is_vertical[i - 1])


def _has_required_instruction_mix(path_specs: list[dict[str, object]]) -> bool:
    """Keep every sample visibly multi-branch instead of degenerating to a weak subset."""
    grouped_areas = [tuple(int(area) for area in spec["areas"]) for spec in path_specs]
    if len(set(grouped_areas)) != len(grouped_areas):
        return False

    combined = {area for areas in grouped_areas for area in areas}
    return combined.issuperset({4, 5, 6, 7})


def _expand_forbidden(cells: set[tuple[int, int]], H: int, W: int) -> set[tuple[int, int]]:
    """Reserve occupied cells and their 4-neighbors for other groups."""
    out = set(cells)
    for r, c in cells:
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= rr < H and 0 <= cc < W:
                out.add((int(rr), int(cc)))
    return out


def _sample_path(
    N: int,
    H: int,
    start_col: int,
    forbidden: set[tuple[int, int]],
) -> tuple[list[int], set[tuple[int, int]]] | None:
    """Backtracking sampler for one non-overlapping grouped polyline."""

    start_cell = (GRAY_START_ROW, int(start_col))
    if start_cell in forbidden:
        return None

    target_changes = min(MIN_DIRECTION_CHANGES_PER_PATH, max(0, N - 1))

    def dfs(
        idx: int,
        r: int,
        c: int,
        areas: list[int],
        used: set[tuple[int, int]],
        changes: int,
    ) -> tuple[list[int], set[tuple[int, int]]] | None:
        if idx == N:
            if changes >= target_changes:
                return areas[:], set(used)
            return None

        remaining = N - idx - 1
        prev_vertical = areas[-1] == 4 if areas else None
        options = list(_AREAS)
        random.shuffle(options)
        for area in options:
            seg = _segment_cells(r, c, int(area))
            if seg is None:
                continue
            cells, end_pos = seg
            if any(not (0 <= rr < H and 0 <= cc < LEFT_W) for rr, cc in cells):
                continue
            cell_set = set(cells)
            if cell_set & forbidden or cell_set & used:
                continue

            new_changes = changes
            if prev_vertical is not None and prev_vertical != (area == 4):
                new_changes += 1
            if new_changes + remaining < target_changes:
                continue

            areas.append(int(area))
            used.update(cell_set)
            found = dfs(idx + 1, int(end_pos[0]), int(end_pos[1]), areas, used, new_changes)
            if found is not None:
                return found
            for cell in cell_set:
                used.remove(cell)
            areas.pop()
        return None

    return dfs(0, GRAY_START_ROW, int(start_col), [], {start_cell}, 0)


def _render_output(path_specs: list[dict[str, object]], H: int) -> np.ndarray:
    out = np.zeros((H, LEFT_W), dtype=int)
    for spec in path_specs:
        r, c = GRAY_START_ROW, int(spec["start_col"])
        out[r, c] = GRAY
        for area in spec["areas"]:
            color = int(_AREA_TO_COLOR[int(area)])
            cells, end_pos = _segment_cells(r, c, int(area))
            for rr, cc in cells:
                out[int(rr), int(cc)] = color
            r, c = map(int, end_pos)
    return out


def _render_grid(N: int, path_specs: list[dict[str, object]]) -> np.ndarray:
    H = 4 * N - 1
    grid = np.zeros((H, GRID_W), dtype=int)
    grid[:, DIVIDER_COL] = DIVIDER_COLOR

    for spec in path_specs:
        start_col = int(spec["start_col"])
        tile_col = int(spec["tile_col"])
        grid[GRAY_START_ROW, DIVIDER_COL + 1 + start_col] = GRAY
        for row_idx, area in enumerate(spec["areas"]):
            r0 = TILE_ROW_STRIDE * row_idx
            grid[r0 : r0 + 3, tile_col : tile_col + 3] = _TEMPLATES[int(area)] * _AREA_TO_COLOR[int(area)]
    return grid


def _gen_attempt() -> np.ndarray | None:
    N = random.randint(N_MIN, N_MAX)
    H = 4 * N - 1

    # Keep the starts in two visible bands so their left-to-right pairing with glyph columns
    # is easy to read from examples.
    start_pairs = [(a, b) for a in range(0, 3) for b in range(4, LEFT_W)]
    start_cols = list(random.choice(start_pairs))
    occupied: set[tuple[int, int]] = set()
    forbidden: set[tuple[int, int]] = set()
    path_specs: list[dict[str, object]] = []

    for tile_col, start_col in zip(TILE_COLS, start_cols):
        sampled = _sample_path(N, H, int(start_col), forbidden)
        if sampled is None:
            return None
        areas, used_cells = sampled
        path_specs.append({"tile_col": int(tile_col), "start_col": int(start_col), "areas": areas})
        occupied |= used_cells
        forbidden = _expand_forbidden(occupied, H, LEFT_W)

    total_changes = sum(_path_direction_changes(spec["areas"]) for spec in path_specs)
    if total_changes < 2:
        return None
    if not _has_required_instruction_mix(path_specs):
        return None

    grid = _render_grid(N, path_specs)
    expected = _render_output(path_specs, H)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError(f"generate_input: failed after {MAX_ATTEMPTS} attempts")
