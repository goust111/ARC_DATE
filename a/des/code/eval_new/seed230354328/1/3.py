from __future__ import annotations

import numpy as np


BG = 0
GRID = 5
CELL_H = 5
CELL_W = 3
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 400


def _mask(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


SOURCE_MASKS: dict[int, list[np.ndarray]] = {
    0: [
        _mask(["111", "100", "100", "100", "111"]),
        _mask(["111", "010", "010", "010", "010"]),
        _mask(["111", "001", "111", "100", "111"]),
        _mask(["100", "110", "111", "001", "001"]),
    ],
    1: [_mask(["111", "101", "101", "101", "111"])],
    2: [_mask(["111", "101", "111", "101", "111"])],
}

LEGEND_OUTPUT_MASKS = [
    _mask(["111", "111", "111", "111", "111"]),
    _mask(["010", "111", "010", "111", "010"]),
    _mask(["100", "110", "111", "110", "100"]),
    _mask(["001", "011", "111", "011", "001"]),
    _mask(["110", "110", "000", "011", "011"]),
]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
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
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(int(rr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _border_bg(grid: np.ndarray) -> int:
    corners = np.array([grid[0, 0], grid[0, -1], grid[-1, 0], grid[-1, -1]], dtype=int)
    vals, cnts = np.unique(corners, return_counts=True)
    return int(vals[cnts.argmax()])


def _most_common_nonbg(grid: np.ndarray, bg: int) -> int:
    vals, cnts = np.unique(grid[grid != bg], return_counts=True)
    assert len(vals) > 0
    return int(vals[cnts.argmax()])


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in points]
    cs = [c for _, c in points]
    return min(rs), min(cs), max(rs), max(cs)


def _parse_grid_component(
    grid: np.ndarray, color: int, points: list[tuple[int, int]]
) -> dict[str, int | list[int]] | None:
    r0, c0, r1, c1 = _bbox(points)
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    rows = [r for r in range(sub.shape[0]) if bool(np.all(sub[r, :] == color))]
    cols = [c for c in range(sub.shape[1]) if bool(np.all(sub[:, c] == color))]
    if len(rows) < 2 or len(cols) < 2:
        return None
    if rows[0] != 0 or rows[-1] != sub.shape[0] - 1 or cols[0] != 0 or cols[-1] != sub.shape[1] - 1:
        return None
    cell_hs = [rows[i + 1] - rows[i] - 1 for i in range(len(rows) - 1)]
    cell_ws = [cols[i + 1] - cols[i] - 1 for i in range(len(cols) - 1)]
    if min(cell_hs) <= 0 or min(cell_ws) <= 0:
        return None
    if len(set(cell_hs)) != 1 or len(set(cell_ws)) != 1:
        return None
    return {
        "r0": int(r0),
        "c0": int(c0),
        "r1": int(r1),
        "c1": int(c1),
        "rows": [int(x) for x in rows],
        "cols": [int(x) for x in cols],
        "nrows": int(len(rows) - 1),
        "ncols": int(len(cols) - 1),
        "cell_h": int(cell_hs[0]),
        "cell_w": int(cell_ws[0]),
    }


def _cell_bounds(spec: dict[str, int | list[int]], row: int, col: int) -> tuple[int, int, int, int]:
    rows = spec["rows"]
    cols = spec["cols"]
    assert isinstance(rows, list)
    assert isinstance(cols, list)
    r0 = int(spec["r0"]) + int(rows[row]) + 1
    r1 = int(spec["r0"]) + int(rows[row + 1]) - 1
    c0 = int(spec["c0"]) + int(cols[col]) + 1
    c1 = int(spec["c0"]) + int(cols[col + 1]) - 1
    return r0, r1, c0, c1


def _count_holes(patch: np.ndarray, bg: int) -> int:
    hole_mask = patch == bg
    holes = 0
    for comp in _components(hole_mask):
        if any(r == 0 or c == 0 or r == patch.shape[0] - 1 or c == patch.shape[1] - 1 for r, c in comp):
            continue
        holes += 1
    return int(holes)


def _parse_markers(
    grid: np.ndarray, spec: dict[str, int | list[int]], bg: int, horizontal: bool
) -> dict[int, int]:
    out: dict[int, int] = {}
    count = int(spec["ncols"] if horizontal else spec["nrows"])
    for idx in range(count):
        r0, r1, c0, c1 = _cell_bounds(spec, idx if not horizontal else 0, idx if horizontal else 0)
        if horizontal:
            marker_vals = [int(v) for v in grid[int(spec["r0"]) - 1, c0 : c1 + 1] if int(v) != bg]
        else:
            marker_vals = [int(v) for v in grid[r0 : r1 + 1, int(spec["c0"]) - 1] if int(v) != bg]
        if not marker_vals:
            continue
        uniq = set(marker_vals)
        assert len(uniq) == 1
        out[marker_vals[0]] = idx
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match top/left border colors to board cells, count holes there, and replace each hit using the legend pattern for that hole count.

    Input structure:
    - Background is black, with one grey square-cell board and a separate grey 2x3 legend grid on the right.
    - Each board cell contains one colored sprite whose black interior may contain 0, 1, or 2 enclosed holes.
    - Single-pixel colored markers sit just above some board columns and just left of some board rows.
    - In the legend, the top row shows example sprites with 0/1/2 holes, and the bottom row shows the replacement pattern for each example column.

    Rule:
    - Parse the grey board to recover its cells, and pair colors that appear in both the top markers and the left markers.
    - For each paired color, locate the row/column intersection cell and count the number of enclosed black holes in that cell's sprite.
    - Read the legend by counting holes in each top legend sample, then use the pattern in the corresponding bottom legend cell as the replacement mask.
    - Replace every selected board cell with that legend pattern drawn in the paired marker color; all other cells stay unchanged.

    Key insight:
    The transformation requires composing three cues: border-color pairing chooses a specific cell, the chosen cell's hole count chooses a legend column, and the legend's lower sample supplies the shape to draw.
    """
    g = np.asarray(grid, dtype=int)
    bg = _border_bg(g)
    grid_color = _most_common_nonbg(g, bg)

    specs: list[dict[str, int | list[int]]] = []
    for comp in _components(g == grid_color):
        spec = _parse_grid_component(g, grid_color, comp)
        if spec is not None:
            specs.append(spec)

    board = next(spec for spec in specs if int(spec["nrows"]) == int(spec["ncols"]) and int(spec["nrows"]) >= 2)
    legend = next(spec for spec in specs if int(spec["nrows"]) == 2 and int(spec["ncols"]) == 3)

    top_markers = _parse_markers(g, board, bg, horizontal=True)
    left_markers = _parse_markers(g, board, bg, horizontal=False)

    hole_to_pattern: dict[int, np.ndarray] = {}
    for col in range(3):
        tr0, tr1, tc0, tc1 = _cell_bounds(legend, 0, col)
        br0, br1, bc0, bc1 = _cell_bounds(legend, 1, col)
        top_patch = g[tr0 : tr1 + 1, tc0 : tc1 + 1]
        bottom_patch = g[br0 : br1 + 1, bc0 : bc1 + 1]
        hole_to_pattern[_count_holes(top_patch, bg)] = bottom_patch != bg

    assert set(hole_to_pattern) == {0, 1, 2}

    out = g.copy()
    for color in sorted(set(top_markers) & set(left_markers)):
        row = int(left_markers[color])
        col = int(top_markers[color])
        r0, r1, c0, c1 = _cell_bounds(board, row, col)
        hole_count = _count_holes(g[r0 : r1 + 1, c0 : c1 + 1], bg)
        pattern = hole_to_pattern[hole_count]
        patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
        patch[pattern] = int(color)
        out[r0 : r1 + 1, c0 : c1 + 1] = patch

    return out


def _empty_grid(rows: int, cols: int, bg: int, line_color: int) -> np.ndarray:
    patch = np.full((rows * (CELL_H + 1) + 1, cols * (CELL_W + 1) + 1), bg, dtype=int)
    patch[:: CELL_H + 1, :] = line_color
    patch[:, :: CELL_W + 1] = line_color
    return patch


def _patch_cell_bounds(row: int, col: int) -> tuple[int, int, int, int]:
    r0 = 1 + row * (CELL_H + 1)
    c0 = 1 + col * (CELL_W + 1)
    return r0, r0 + CELL_H, c0, c0 + CELL_W


def _draw_mask(canvas: np.ndarray, top: int, left: int, mask: np.ndarray, color: int, bg: int) -> None:
    patch = np.full(mask.shape, bg, dtype=int)
    patch[mask] = int(color)
    canvas[top : top + mask.shape[0], left : left + mask.shape[1]] = patch


def _center_on_board_col(col: int) -> int:
    return 1 + 1 + col * (CELL_W + 1) + CELL_W // 2


def _center_on_board_row(row: int) -> int:
    return 1 + 1 + row * (CELL_H + 1) + CELL_H // 2


def _choice_without(rng: np.random.Generator, values: list[int], forbidden: set[int]) -> int:
    options = [v for v in values if v not in forbidden]
    assert options
    return int(options[int(rng.integers(len(options)))])


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    n = int(rng.integers(2, 4))
    board = _empty_grid(n, n, BG, GRID)
    legend = _empty_grid(2, 3, BG, GRID)

    board_h, board_w = board.shape
    legend_h, legend_w = legend.shape
    height = 1 + board_h
    width = 1 + board_w + 1 + legend_w
    legend_r0 = height - legend_h
    legend_c0 = 1 + board_w + 1

    canvas = np.full((height, width), BG, dtype=int)
    canvas[1 : 1 + board_h, 1 : 1 + board_w] = board
    canvas[legend_r0 : legend_r0 + legend_h, legend_c0 : legend_c0 + legend_w] = legend

    legend_color = int(rng.choice(PALETTE))
    marker_palette = [c for c in PALETTE if c != legend_color]

    matched_count = int(rng.integers(1, n + 1))
    top_count = int(rng.integers(matched_count, n + 1))
    left_count = int(rng.integers(matched_count, n + 1))

    top_slots = [int(x) for x in rng.choice(n, size=top_count, replace=False)]
    left_slots = [int(x) for x in rng.choice(n, size=left_count, replace=False)]
    top_match_slots = [int(x) for x in rng.choice(top_slots, size=matched_count, replace=False)]
    left_match_slots = [int(x) for x in rng.choice(left_slots, size=matched_count, replace=False)]
    rng.shuffle(top_match_slots)
    rng.shuffle(left_match_slots)

    marker_colors = [int(c) for c in rng.choice(marker_palette, size=top_count + left_count - matched_count, replace=False)]
    matched_colors = marker_colors[:matched_count]
    extra_top_colors = marker_colors[matched_count:top_count]
    extra_left_colors = marker_colors[top_count:]

    top_markers: dict[int, int] = {color: col for color, col in zip(matched_colors, top_match_slots, strict=False)}
    left_markers: dict[int, int] = {color: row for color, row in zip(matched_colors, left_match_slots, strict=False)}

    free_top_slots = [slot for slot in top_slots if slot not in top_match_slots]
    free_left_slots = [slot for slot in left_slots if slot not in left_match_slots]
    for color, col in zip(extra_top_colors, free_top_slots, strict=False):
        top_markers[int(color)] = int(col)
    for color, row in zip(extra_left_colors, free_left_slots, strict=False):
        left_markers[int(color)] = int(row)

    hole_assignments = np.array(rng.integers(0, 3, size=(n, n)), dtype=int)
    if matched_count == 1:
        matched_holes = [int(rng.integers(0, 3))]
    elif matched_count == 2:
        matched_holes = [int(x) for x in rng.choice(3, size=2, replace=False)]
    else:
        matched_holes = [0, 1, 2]
        rng.shuffle(matched_holes)

    color_to_hole: dict[int, int] = {}
    for color, hole in zip(matched_colors, matched_holes, strict=False):
        row = int(left_markers[color])
        col = int(top_markers[color])
        hole_assignments[row, col] = int(hole)
        color_to_hole[int(color)] = int(hole)

    cell_colors = np.zeros((n, n), dtype=int)
    cell_masks: list[list[np.ndarray]] = []
    for r in range(n):
        mask_row: list[np.ndarray] = []
        for c in range(n):
            hole = int(hole_assignments[r, c])
            mask = SOURCE_MASKS[hole][int(rng.integers(len(SOURCE_MASKS[hole])))]
            forbidden = set()
            for color, row in left_markers.items():
                if row == r and top_markers.get(color) == c:
                    forbidden.add(int(color))
            cell_color = _choice_without(rng, PALETTE, forbidden)
            cell_colors[r, c] = cell_color
            mask_row.append(mask)
            rr0, rr1, cc0, cc1 = _patch_cell_bounds(r, c)
            _draw_mask(canvas, 1 + rr0, 1 + cc0, mask, cell_color, BG)
        cell_masks.append(mask_row)

    legend_holes = [0, 1, 2]
    rng.shuffle(legend_holes)
    output_patterns = [LEGEND_OUTPUT_MASKS[int(i)] for i in rng.choice(len(LEGEND_OUTPUT_MASKS), size=3, replace=False)]
    hole_to_output: dict[int, np.ndarray] = {}
    for col, hole in enumerate(legend_holes):
        top_mask = SOURCE_MASKS[int(hole)][int(rng.integers(len(SOURCE_MASKS[int(hole)])))]
        bottom_mask = output_patterns[col]
        hole_to_output[int(hole)] = bottom_mask
        tr0, _tr1, tc0, _tc1 = _patch_cell_bounds(0, col)
        br0, _br1, bc0, _bc1 = _patch_cell_bounds(1, col)
        _draw_mask(canvas, legend_r0 + tr0, legend_c0 + tc0, top_mask, legend_color, BG)
        _draw_mask(canvas, legend_r0 + br0, legend_c0 + bc0, bottom_mask, legend_color, BG)

    for color, col in top_markers.items():
        canvas[0, _center_on_board_col(int(col))] = int(color)
    for color, row in left_markers.items():
        canvas[_center_on_board_row(int(row)), 0] = int(color)

    expected = canvas.copy()
    for color in set(top_markers) & set(left_markers):
        row = int(left_markers[color])
        col = int(top_markers[color])
        rr0, rr1, cc0, cc1 = _patch_cell_bounds(row, col)
        patch = np.full((CELL_H, CELL_W), BG, dtype=int)
        patch[hole_to_output[int(color_to_hole[color])]] = int(color)
        expected[1 + rr0 : 1 + rr1, 1 + cc0 : 1 + cc1] = patch

    try:
        produced = main(canvas)
    except Exception:
        return None
    if not np.array_equal(produced, expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
