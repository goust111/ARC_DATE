import itertools
import random
from collections import Counter

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4


def get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def colors_present(grid: np.ndarray, *, exclude=()) -> list[int]:
    excluded = {int(x) for x in exclude}
    return [int(v) for v in np.unique(grid) if int(v) not in excluded]


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _find_components(grid: np.ndarray, color: int) -> list[dict]:
    comps = _components_from_mask(grid == int(color))
    out = []
    for coords in comps:
        out.append({"coords": coords, "bbox": bbox_coords(coords), "area": int(coords.shape[0])})
    return out


def _regular_lattice(
    grid: np.ndarray, bg: int
) -> tuple[int, list[int], list[int], int, int]:
    candidates: list[tuple[int, list[int], list[int], int, int]] = []
    for color in colors_present(grid, exclude={bg}):
        coords = np.argwhere(grid == int(color))
        rows = sorted({int(r) for r, _ in coords})
        cols = sorted({int(c) for _, c in coords})
        if len(rows) < 2 or len(cols) < 2:
            continue
        if int(coords.shape[0]) != len(rows) * len(cols):
            continue
        if any(int(grid[r, c]) != int(color) for r in rows for c in cols):
            continue
        row_diffs = np.diff(rows)
        col_diffs = np.diff(cols)
        if len(set(map(int, row_diffs))) != 1 or len(set(map(int, col_diffs))) != 1:
            continue
        row_pitch = int(row_diffs[0])
        col_pitch = int(col_diffs[0])
        if row_pitch < 3 or col_pitch < 3:
            continue
        candidates.append((int(color), rows, cols, row_pitch, col_pitch))
    assert len(candidates) == 1, "expected exactly one regular marker lattice"
    return candidates[0]


def _fill_enclosed_regions(grid: np.ndarray, *, bg: int = BLACK, fill_color: int = YELLOW) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    bg_comps = _components_from_mask(out == int(bg))
    if len(bg_comps) <= 1:
        return out
    outside_idx = max(range(len(bg_comps)), key=lambda i: int(bg_comps[i].shape[0]))
    for idx, comp in enumerate(bg_comps):
        if idx == outside_idx:
            continue
        out[comp[:, 0], comp[:, 1]] = int(fill_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Center-align each green fragment within its marker-lattice cell so the fragments close a contour, then fill the enclosed background.

    Input structure:
    - The background is black.
    - One foreground color forms a complete regular Cartesian lattice of isolated marker pixels.
    - The other foreground color forms one small 4-connected open fragment in each lattice cell, offset away from that cell's center.

    Rule:
    - Keep the marker lattice fixed and treat consecutive marker rows and columns as the cell boundaries.
    - For each green fragment, find the unique lattice cell containing its bbox, compare the fragment bbox center to that cell center, and translate the fragment so those centers coincide.
    - After all fragments are centered, treat the red markers and green pixels as walls and split the black background into 4-connected regions.
    - Leave the largest black region as the exterior and fill every other black region yellow.

    Key insight:
    - The green pieces are not meaningful in their offset positions; only after cell-wise center alignment do their open arcs line up with the red lattice corners to create a closed boundary whose enclosed background can be filled.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = get_bg(g)
    fg_colors = colors_present(g, exclude={bg})
    assert len(fg_colors) == 2, "expected marker and fragment colors"

    marker_color, marker_rows, marker_cols, _row_pitch, _col_pitch = _regular_lattice(g, bg)
    piece_colors = [color for color in fg_colors if int(color) != int(marker_color)]
    assert len(piece_colors) == 1, "expected exactly one fragment color"
    piece_color = int(piece_colors[0])

    pieces = _find_components(g, piece_color)
    expected_cells = (len(marker_rows) - 1) * (len(marker_cols) - 1)
    assert len(pieces) == expected_cells, "expected one fragment per lattice cell"

    out = g.copy()
    out[out == piece_color] = bg

    used_cells: set[tuple[int, int]] = set()
    h, w = g.shape
    for piece in pieces:
        coords = piece["coords"]
        r0, c0, r1, c1 = map(int, piece["bbox"])

        cell_rows = [i for i in range(len(marker_rows) - 1) if marker_rows[i] <= r0 and r1 <= marker_rows[i + 1]]
        cell_cols = [j for j in range(len(marker_cols) - 1) if marker_cols[j] <= c0 and c1 <= marker_cols[j + 1]]
        assert len(cell_rows) == 1 and len(cell_cols) == 1, "expected each fragment bbox inside one lattice cell"
        cell = (int(cell_rows[0]), int(cell_cols[0]))
        assert cell not in used_cells, "expected at most one fragment per cell"
        used_cells.add(cell)

        shift_r2 = (marker_rows[cell[0]] + marker_rows[cell[0] + 1]) - (r0 + r1)
        shift_c2 = (marker_cols[cell[1]] + marker_cols[cell[1] + 1]) - (c0 + c1)
        assert shift_r2 % 2 == 0 and shift_c2 % 2 == 0, "expected integer center-alignment shift"
        dr = shift_r2 // 2
        dc = shift_c2 // 2

        moved = coords + np.array([[dr, dc]], dtype=int)
        assert np.all((0 <= moved[:, 0]) & (moved[:, 0] < h))
        assert np.all((0 <= moved[:, 1]) & (moved[:, 1] < w))
        assert np.all(g[moved[:, 0], moved[:, 1]] != marker_color), "translated fragments must not overwrite markers"
        out[moved[:, 0], moved[:, 1]] = piece_color

    return _fill_enclosed_regions(out, bg=bg, fill_color=YELLOW)


PITCH = 4
BASE_MARKER_ROWS = (0, 4, 8)
BASE_MARKER_COLS = (0, 4, 8, 12)
BASE_CELL_SHAPES = (
    ((-1, -1), (0, -1), (0, 0), (0, 1), (1, 1)),
    ((0, -2), (0, -1), (0, 0), (0, 1), (0, 2)),
    ((0, -1), (0, 0), (0, 1)),
    ((0, -1), (0, 0), (0, 1)),
    ((0, -2), (0, -1), (0, 0), (0, 1), (0, 2)),
    ((-1, -1), (-1, 0), (-1, 1), (0, -1), (1, -1)),
)
LOCAL_CORNERS = {(-2, -2), (-2, 2), (2, -2), (2, 2)}
BASE_CENTERS = tuple((BASE_MARKER_ROWS[i] + 2, BASE_MARKER_COLS[j] + 2) for i in range(2) for j in range(3))


def _base_grid() -> np.ndarray:
    grid = np.zeros((BASE_MARKER_ROWS[-1] + 1, BASE_MARKER_COLS[-1] + 1), dtype=int)
    for r in BASE_MARKER_ROWS:
        for c in BASE_MARKER_COLS:
            grid[int(r), int(c)] = RED
    for (cr, cc), shape in zip(BASE_CENTERS, BASE_CELL_SHAPES):
        for dr, dc in shape:
            grid[int(cr + dr), int(cc + dc)] = GREEN
    return grid


def _local_shift_options(shape: tuple[tuple[int, int], ...]) -> list[tuple[int, int]]:
    options = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            moved = {(r + dr, c + dc) for r, c in shape}
            if not all(-2 <= r <= 2 and -2 <= c <= 2 for r, c in moved):
                continue
            if any((r, c) in LOCAL_CORNERS for r, c in moved):
                continue
            options.append((int(dr), int(dc)))
    return options


def _build_shift_solutions() -> tuple[tuple[tuple[int, int], ...], ...]:
    marker_set = {(int(r), int(c)) for r in BASE_MARKER_ROWS for c in BASE_MARKER_COLS}
    per_cell: list[list[tuple[int, int]]] = []
    for idx, ((cr, cc), shape) in enumerate(zip(BASE_CENTERS, BASE_CELL_SHAPES)):
        i = idx // 3
        j = idx % 3
        r0, r1 = BASE_MARKER_ROWS[i], BASE_MARKER_ROWS[i + 1]
        c0, c1 = BASE_MARKER_COLS[j], BASE_MARKER_COLS[j + 1]
        options = []
        for dr, dc in _local_shift_options(shape):
            pts = [(cr + r + dr, cc + c + dc) for r, c in shape]
            if any((int(r), int(c)) in marker_set for r, c in pts):
                continue
            if not all(r0 <= int(r) <= r1 and c0 <= int(c) <= c1 for r, c in pts):
                continue
            options.append((int(dr), int(dc)))
        assert options, "each base fragment needs at least one non-zero legal shift"
        per_cell.append(options)

    solutions: list[tuple[tuple[int, int], ...]] = []
    for combo in itertools.product(*per_cell):
        absolute: list[set[tuple[int, int]]] = []
        for (cr, cc), shape, (dr, dc) in zip(BASE_CENTERS, BASE_CELL_SHAPES, combo):
            pts = {(int(cr + r + dr), int(cc + c + dc)) for r, c in shape}
            absolute.append(pts)
        ok = True
        for i in range(len(absolute)):
            for j in range(i + 1, len(absolute)):
                if any(abs(r1 - r2) <= 1 and abs(c1 - c2) <= 1 for r1, c1 in absolute[i] for r2, c2 in absolute[j]):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            solutions.append(tuple((int(dr), int(dc)) for dr, dc in combo))
    assert solutions, "expected at least one valid shift assignment"
    return tuple(solutions)


BASE_CENTERED = _base_grid()
BASE_OUTPUT = _fill_enclosed_regions(BASE_CENTERED, bg=BLACK, fill_color=YELLOW)
BASE_SHIFT_SOLUTIONS = _build_shift_solutions()


def _build_base_input(shifts: tuple[tuple[int, int], ...]) -> np.ndarray:
    grid = np.zeros_like(BASE_CENTERED)
    for r in BASE_MARKER_ROWS:
        for c in BASE_MARKER_COLS:
            grid[int(r), int(c)] = RED
    for (cr, cc), shape, (dr, dc) in zip(BASE_CENTERS, BASE_CELL_SHAPES, shifts):
        for sr, sc in shape:
            grid[int(cr + sr + dr), int(cc + sc + dc)] = GREEN
    return grid


def _transform_grid(grid: np.ndarray, k: int, flip_lr: bool) -> np.ndarray:
    out = np.rot90(grid, int(k)).copy()
    if bool(flip_lr):
        out = np.fliplr(out).copy()
    return out


def _pad_grid(grid: np.ndarray, top: int, bottom: int, left: int, right: int) -> np.ndarray:
    h, w = grid.shape
    out = np.zeros((h + top + bottom, w + left + right), dtype=int)
    out[top : top + h, left : left + w] = grid
    return out


def _random_split(total_extra: int) -> tuple[int, int]:
    top_or_left = random.randint(1, int(total_extra) - 1)
    return int(top_or_left), int(total_extra - top_or_left)


def _essential_corner_after_transform(k: int, flip_lr: bool) -> tuple[int, int]:
    probe = np.zeros_like(BASE_CENTERED)
    probe[-1, -1] = 1
    transformed = _transform_grid(probe, k, flip_lr)
    rr, cc = np.argwhere(transformed == 1)[0]
    return int(rr), int(cc)


def generate_input() -> np.ndarray:
    for _ in range(300):
        base_input = _build_base_input(random.choice(BASE_SHIFT_SOLUTIONS))
        rot_k = random.randint(0, 3)
        flip_lr = bool(random.randint(0, 1))

        transformed_input = _transform_grid(base_input, rot_k, flip_lr)
        expected_output = _transform_grid(BASE_OUTPUT, rot_k, flip_lr)

        h, w = transformed_input.shape
        corner_r, corner_c = _essential_corner_after_transform(rot_k, flip_lr)

        vertical_pad = random.randint(1, min(14, 30 - h))
        horizontal_pad = random.randint(1, min(14, 30 - w))
        if corner_r == 0:
            top, bottom = 0, vertical_pad
        else:
            top, bottom = vertical_pad, 0
        if corner_c == 0:
            left, right = 0, horizontal_pad
        else:
            left, right = horizontal_pad, 0

        candidate = _pad_grid(transformed_input, top, bottom, left, right)
        expected = _pad_grid(expected_output, top, bottom, left, right)

        try:
            if np.array_equal(main(candidate), expected):
                return candidate
        except Exception:
            continue

    raise RuntimeError("failed to generate a valid centered-contour input")
