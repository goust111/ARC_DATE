from __future__ import annotations

from collections import Counter
import random

import numpy as np


BRIDGE_COLOR = 7


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _tight_mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0 = int(coords[:, 0].min())
    r1 = int(coords[:, 0].max())
    c0 = int(coords[:, 1].min())
    c1 = int(coords[:, 1].max())
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out


def _shape_dims(mask: np.ndarray) -> tuple[int, int]:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    r0 = int(np.argmax(rows))
    r1 = int(len(rows) - 1 - np.argmax(rows[::-1]))
    c0 = int(np.argmax(cols))
    c1 = int(len(cols) - 1 - np.argmax(cols[::-1]))
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _detect_grid(g: np.ndarray, bg: int) -> tuple[int, list[int], list[int], int]:
    counts = Counter(map(int, g.ravel()))
    candidates = [color for color, _count in counts.most_common() if color != bg]
    for line_color in candidates:
        mask = g == line_color
        row_counts = mask.sum(axis=1)
        col_counts = mask.sum(axis=0)
        if row_counts.max() == 0 or col_counts.max() == 0:
            continue
        h_lines = np.flatnonzero(row_counts == row_counts.max()).tolist()
        v_lines = np.flatnonzero(col_counts == col_counts.max()).tolist()
        if len(h_lines) < 3 or len(v_lines) < 3:
            continue
        row_steps = np.diff(h_lines)
        col_steps = np.diff(v_lines)
        if len(set(map(int, row_steps))) != 1 or len(set(map(int, col_steps))) != 1:
            continue
        if int(row_steps[0]) <= 1 or int(col_steps[0]) <= 1 or int(row_steps[0]) != int(col_steps[0]):
            continue
        r0, r1 = int(h_lines[0]), int(h_lines[-1])
        c0, c1 = int(v_lines[0]), int(v_lines[-1])
        ok = True
        for r in h_lines:
            if not np.all(mask[r, c0 : c1 + 1]):
                ok = False
                break
        if not ok:
            continue
        for c in v_lines:
            if not np.all(mask[r0 : r1 + 1, c]):
                ok = False
                break
        if not ok:
            continue
        cell_size = int(row_steps[0] - 1)
        return int(line_color), [int(x) for x in h_lines], [int(x) for x in v_lines], cell_size
    raise AssertionError("failed to detect grid geometry")


def _legend_shapes(
    g: np.ndarray,
    bg: int,
    line_color: int,
    grid_bbox: tuple[int, int, int, int],
) -> dict[int, np.ndarray]:
    r0, c0, r1, c1 = grid_bbox
    outside = np.ones_like(g, dtype=bool)
    outside[r0 : r1 + 1, c0 : c1 + 1] = False

    shapes: dict[int, np.ndarray] = {}
    for color in sorted(int(x) for x in np.unique(g) if int(x) not in (bg, line_color)):
        mask = (g == color) & outside
        if not mask.any():
            continue
        comps = _components(mask)
        assert len(comps) == 1, "each legend color must form one outside component"
        shape = _tight_mask_from_coords(comps[0])
        assert shape.shape[0] % 2 == 1 and shape.shape[1] % 2 == 1, "legend bboxes must be odd-sized"
        ar, ac = shape.shape[0] // 2, shape.shape[1] // 2
        assert shape[ar, ac], "legend anchor pixel must be present at the center"
        shapes[color] = shape
    assert shapes, "expected outside legend shapes"
    return shapes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow same-color legend shapes from seed pixels inside a regular grey cell grid,
    then bridge adjacent non-empty cells on their shared separators.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    line_color, h_lines, v_lines, cell_size = _detect_grid(g, bg)
    grid_bbox = (h_lines[0], v_lines[0], h_lines[-1], v_lines[-1])
    shapes = _legend_shapes(g, bg, line_color, grid_bbox)

    out = g.copy()
    n_rows = len(h_lines) - 1
    n_cols = len(v_lines) - 1
    cell_info: dict[tuple[int, int], dict[str, object]] = {}

    for gr in range(n_rows):
        for gc in range(n_cols):
            top = h_lines[gr] + 1
            left = v_lines[gc] + 1
            window = g[top : top + cell_size, left : left + cell_size]
            seed_coords = np.argwhere((window != bg) & (window != line_color))
            assert seed_coords.shape[0] <= 1, "each cell may contain at most one seed"
            if seed_coords.shape[0] == 0:
                cell_info[(gr, gc)] = {"occupied": False}
                continue

            sr, sc = map(int, seed_coords[0])
            color = int(window[sr, sc])
            assert color in shapes, "seed color must have a matching legend"
            shape = shapes[color]
            ah, aw = shape.shape[0] // 2, shape.shape[1] // 2

            out[top : top + cell_size, left : left + cell_size] = bg
            for rr, cc in np.argwhere(shape):
                ar = top + sr + int(rr) - ah
                ac = left + sc + int(cc) - aw
                assert top <= ar < top + cell_size and left <= ac < left + cell_size, "grown shape must stay inside its cell"
                out[ar, ac] = color

            shape_h, shape_w = _shape_dims(shape)
            cell_info[(gr, gc)] = {
                "occupied": True,
                "shape_h": int(shape_h),
                "shape_w": int(shape_w),
            }

    for gr in range(n_rows):
        for gc in range(n_cols - 1):
            left_cell = cell_info[(gr, gc)]
            right_cell = cell_info[(gr, gc + 1)]
            if not left_cell["occupied"] or not right_cell["occupied"]:
                continue
            bridge_len = min(int(left_cell["shape_w"]), int(right_cell["shape_w"]))
            top = h_lines[gr] + 1
            start = top + (cell_size - bridge_len) // 2
            col = v_lines[gc + 1]
            out[start : start + bridge_len, col] = BRIDGE_COLOR

    for gr in range(n_rows - 1):
        for gc in range(n_cols):
            upper_cell = cell_info[(gr, gc)]
            lower_cell = cell_info[(gr + 1, gc)]
            if not upper_cell["occupied"] or not lower_cell["occupied"]:
                continue
            bridge_len = min(int(upper_cell["shape_h"]), int(lower_cell["shape_h"]))
            left = v_lines[gc] + 1
            start = left + (cell_size - bridge_len) // 2
            row = h_lines[gr + 1]
            out[row, start : start + bridge_len] = BRIDGE_COLOR

    return out


# --------------- generator ---------------

BG_COLOR = 0
GRID_COLOR = 5
PALETTE = [1, 2, 3, 4, 6, 8, 9]
GEN_ATTEMPTS = 400

SHAPE_LIBRARY = [
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[0, 1, 1], [0, 1, 0], [1, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1], [0, 1, 0]], dtype=bool),
    np.array(
        [[0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]],
        dtype=bool,
    ),
    np.array(
        [[0, 0, 1, 0, 0], [0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0], [0, 0, 1, 0, 0]],
        dtype=bool,
    ),
    np.array(
        [[1, 1, 1, 1, 1], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]],
        dtype=bool,
    ),
    np.array(
        [[1, 1, 0, 0, 0], [0, 1, 0, 0, 0], [1, 1, 1, 1, 1], [0, 0, 0, 1, 0], [0, 0, 0, 1, 1]],
        dtype=bool,
    ),
]


def _transform_shape(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    if random.random() < 0.5:
        out = np.fliplr(out)
    k = random.randrange(4)
    return np.rot90(out, k)


def _adjacency_flags(occ: set[tuple[int, int]]) -> tuple[bool, bool]:
    has_h = any((r, c + 1) in occ for r, c in occ)
    has_v = any((r + 1, c) in occ for r, c in occ)
    return has_h, has_v


def _sample_occupancy(n_rows: int, n_cols: int, target: tuple[bool, bool]) -> list[tuple[int, int]] | None:
    cells = [(r, c) for r in range(n_rows) for c in range(n_cols)]
    min_count = 3 if target == (True, True) else 2
    max_count = min(len(cells) - 1, len(PALETTE))
    if max_count < min_count:
        return None
    for _ in range(200):
        count = random.randint(min_count, max_count)
        occ = set(random.sample(cells, count))
        if _adjacency_flags(occ) == target:
            return sorted(occ)
    return None


def _grid_layout(
    n_rows: int,
    n_cols: int,
    cell_size: int,
    legend_shapes: list[np.ndarray],
    side: str,
) -> tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]]]:
    grid_h = n_rows * cell_size + (n_rows + 1)
    grid_w = n_cols * cell_size + (n_cols + 1)
    gap = 2
    legend_dims = [(int(shape.shape[0]), int(shape.shape[1])) for shape in legend_shapes]

    if side in ("top", "bottom"):
        strip_h = max(h for h, _w in legend_dims)
        strip_w = sum(w for _h, w in legend_dims) + gap * (len(legend_dims) - 1)
        width = max(grid_w + random.randint(2, 6), strip_w + random.randint(2, 6))
        grid_c0 = random.randint(1, width - grid_w - 1)
        strip_c0 = random.randint(1, width - strip_w - 1)
        if side == "top":
            legend_r0 = random.randint(1, 2)
            grid_r0 = legend_r0 + strip_h + random.randint(2, 3)
            height = grid_r0 + grid_h + random.randint(1, 3)
        else:
            grid_r0 = random.randint(1, 3)
            legend_r0 = grid_r0 + grid_h + random.randint(2, 3)
            height = legend_r0 + strip_h + random.randint(1, 2)
        positions = []
        cursor = strip_c0
        for shape in legend_shapes:
            r = legend_r0 + (strip_h - shape.shape[0]) // 2
            positions.append((int(r), int(cursor)))
            cursor += int(shape.shape[1]) + gap
        return (int(height), int(width)), (int(grid_r0), int(grid_c0)), positions

    strip_w = max(w for _h, w in legend_dims)
    strip_h = sum(h for h, _w in legend_dims) + gap * (len(legend_dims) - 1)
    height = max(grid_h + random.randint(2, 6), strip_h + random.randint(2, 6))
    grid_r0 = random.randint(1, height - grid_h - 1)
    strip_r0 = random.randint(1, height - strip_h - 1)
    if side == "left":
        legend_c0 = random.randint(1, 2)
        grid_c0 = legend_c0 + strip_w + random.randint(2, 3)
        width = grid_c0 + grid_w + random.randint(1, 3)
    else:
        grid_c0 = random.randint(1, 3)
        legend_c0 = grid_c0 + grid_w + random.randint(2, 3)
        width = legend_c0 + strip_w + random.randint(1, 2)

    positions = []
    cursor = strip_r0
    for shape in legend_shapes:
        c = legend_c0 + (strip_w - shape.shape[1]) // 2
        positions.append((int(cursor), int(c)))
        cursor += int(shape.shape[0]) + gap
    return (int(height), int(width)), (int(grid_r0), int(grid_c0)), positions


def _paste_mask(canvas: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    coords = np.argwhere(mask)
    for r, c in coords:
        canvas[r0 + int(r), c0 + int(c)] = int(color)


def _build_input() -> np.ndarray | None:
    branch = random.choice([(True, False), (False, True), (True, True)])
    n_rows = random.randint(2, 4)
    n_cols = random.randint(2, 4)
    cell_size = random.choice([5, 7])

    occupied = _sample_occupancy(n_rows, n_cols, branch)
    if occupied is None:
        return None

    colors = random.sample(PALETTE, len(occupied))
    base_shapes = random.sample(SHAPE_LIBRARY, len(occupied))
    legend_shapes = [_transform_shape(shape) for shape in base_shapes]
    random.shuffle(legend_shapes)
    side = random.choice(["top", "bottom", "left", "right"])

    canvas_shape, (grid_r0, grid_c0), legend_positions = _grid_layout(
        n_rows, n_cols, cell_size, legend_shapes, side
    )
    grid = np.full(canvas_shape, BG_COLOR, dtype=int)
    grid_h = n_rows * cell_size + (n_rows + 1)
    grid_w = n_cols * cell_size + (n_cols + 1)

    for i in range(n_rows + 1):
        rr = grid_r0 + i * (cell_size + 1)
        grid[rr, grid_c0 : grid_c0 + grid_w] = GRID_COLOR
    for j in range(n_cols + 1):
        cc = grid_c0 + j * (cell_size + 1)
        grid[grid_r0 : grid_r0 + grid_h, cc] = GRID_COLOR

    assignments: dict[tuple[int, int], tuple[int, np.ndarray]] = {}
    for (cell_r, cell_c), color, shape in zip(occupied, colors, legend_shapes):
        assignments[(cell_r, cell_c)] = (int(color), shape)

    for (cell_r, cell_c), (color, shape) in assignments.items():
        rad = shape.shape[0] // 2
        local_r = random.randint(rad, cell_size - 1 - rad)
        local_c = random.randint(rad, cell_size - 1 - rad)
        abs_r = grid_r0 + 1 + cell_r * (cell_size + 1) + local_r
        abs_c = grid_c0 + 1 + cell_c * (cell_size + 1) + local_c
        grid[abs_r, abs_c] = color

    for (_cell, (color, shape)), (lr, lc) in zip(assignments.items(), legend_positions):
        _paste_mask(grid, shape, int(lr), int(lc), int(color))

    if Counter(map(int, grid.ravel())).most_common(2)[1][0] != GRID_COLOR:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape != grid.shape:
        return None
    if not np.any(out == BRIDGE_COLOR):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
