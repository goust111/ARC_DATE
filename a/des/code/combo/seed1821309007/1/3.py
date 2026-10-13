import random
from collections import Counter

import numpy as np


BG_COLOR = 7
SEP_COLOR = 9
MARKER_COLOR = 8
TEMPLATE_SIZE_OPTIONS = (3, 5)
CELL_SIZE_OPTIONS = (5, 7)
GRID_SIZE_OPTIONS = (5, 7, 9)
MAX_GEN_ATTEMPTS = 300


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _full_line_color(line: np.ndarray) -> int | None:
    if np.all(line == line[0]):
        return int(line[0])
    return None


def _find_separator_color(grid: np.ndarray, bg: int) -> int:
    candidate_counts = Counter()
    for r in range(grid.shape[0]):
        color = _full_line_color(grid[r, :])
        if color is not None and color != bg:
            candidate_counts[color] += 1
    for c in range(grid.shape[1]):
        color = _full_line_color(grid[:, c])
        if color is not None and color != bg:
            candidate_counts[color] += 1
    assert candidate_counts, "no separator color found"
    return int(candidate_counts.most_common(1)[0][0])


def _separator_positions(grid: np.ndarray, sep: int) -> tuple[list[int], list[int]]:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == sep)]
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == sep)]
    assert len(rows) >= 2 and len(cols) >= 2
    return rows, cols


def _cell_slices(seps: list[int]) -> list[tuple[int, int]]:
    spans = [(seps[i] + 1, seps[i + 1]) for i in range(len(seps) - 1)]
    sizes = {b - a for a, b in spans}
    assert len(sizes) == 1
    return spans


def _extract_template(cell: np.ndarray, bg: int, sep: int) -> np.ndarray:
    mask = (cell != bg) & (cell != sep)
    pts = np.argwhere(mask)
    assert len(pts) > 1
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return cell[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste_center(cell: np.ndarray, patch: np.ndarray, bg: int, sep: int) -> np.ndarray:
    out = np.full_like(cell, bg)
    h, w = patch.shape
    ch, cw = cell.shape
    r0 = (ch - h) // 2
    c0 = (cw - w) // 2
    mask = (patch != bg) & (patch != sep)
    out[r0 : r0 + h, c0 : c0 + w][mask] = patch[mask]
    return out


def _marker_direction(cell: np.ndarray, bg: int, sep: int) -> str | None:
    pts = np.argwhere((cell != bg) & (cell != sep))
    if len(pts) == 0:
        return None
    assert len(pts) == 1
    r, c = map(int, pts[0])
    cr = (cell.shape[0] - 1) / 2.0
    cc = (cell.shape[1] - 1) / 2.0
    dr = r - cr
    dc = c - cc
    if abs(dr) > abs(dc):
        return "down" if dr > 0 else "up"
    return "right" if dc > 0 else "left"


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _dominant_color(g)
    sep = _find_separator_color(g, bg)
    sep_rows, sep_cols = _separator_positions(g, sep)
    row_spans = _cell_slices(sep_rows)
    col_spans = _cell_slices(sep_cols)
    assert len(row_spans) == len(col_spans)

    n = len(row_spans)
    mid = n // 2
    template_cells = {
        "up": (0, mid),
        "down": (n - 1, mid),
        "left": (mid, 0),
        "right": (mid, n - 1),
    }

    templates: dict[str, np.ndarray] = {}
    for side, (ri, ci) in template_cells.items():
        r0, r1 = row_spans[ri]
        c0, c1 = col_spans[ci]
        templates[side] = _extract_template(g[r0:r1, c0:c1], bg, sep)

    out = g.copy()
    for ri, (r0, r1) in enumerate(row_spans):
        for ci, (c0, c1) in enumerate(col_spans):
            if (ri, ci) in template_cells.values():
                continue
            cell = g[r0:r1, c0:c1]
            direction = _marker_direction(cell, bg, sep)
            if direction is None:
                continue
            patch = np.rot90(templates[direction], 2)
            out[r0:r1, c0:c1] = _paste_center(cell, patch, bg, sep)
    return out


def _random_template(shape_size: int, color: int, rng: random.Random) -> np.ndarray:
    if shape_size == 3:
        patterns = [
            np.array([[0, color, 0], [color, color, 0], [0, 0, 0]], dtype=int),
            np.array([[color, 0, 0], [color, color, 0], [0, color, 0]], dtype=int),
            np.array([[0, color, color], [0, color, 0], [0, 0, 0]], dtype=int),
            np.array([[0, color, 0], [0, color, color], [0, 0, color]], dtype=int),
        ]
    else:
        patterns = [
            np.array(
                [
                    [0, 0, color, 0, 0],
                    [0, color, color, 0, 0],
                    [0, 0, color, 0, 0],
                    [0, 0, color, color, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=int,
            ),
            np.array(
                [
                    [0, color, 0, 0, 0],
                    [0, color, color, 0, 0],
                    [0, 0, color, 0, 0],
                    [0, 0, color, color, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=int,
            ),
            np.array(
                [
                    [0, 0, color, color, 0],
                    [0, 0, color, 0, 0],
                    [0, color, color, 0, 0],
                    [0, 0, color, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=int,
            ),
            np.array(
                [
                    [0, color, 0, 0, 0],
                    [0, color, color, 0, 0],
                    [0, 0, color, 0, 0],
                    [0, color, color, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=int,
            ),
        ]
    return patterns[rng.randrange(len(patterns))].copy()


def _marker_positions(cell_size: int) -> dict[str, tuple[int, int]]:
    m = cell_size // 2
    return {
        "up": (1, m),
        "down": (cell_size - 2, m),
        "left": (m, 1),
        "right": (m, cell_size - 2),
    }


def _paste_patch(grid: np.ndarray, r0: int, c0: int, patch: np.ndarray) -> None:
    mask = patch != 0
    grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]][mask] = patch[mask]


def _cell_origin(index: int, cell_size: int) -> int:
    return 1 + index * (cell_size + 1)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n = rng.choice(GRID_SIZE_OPTIONS)
    cell_size = rng.choice(CELL_SIZE_OPTIONS)
    template_size = rng.choice([s for s in TEMPLATE_SIZE_OPTIONS if s <= cell_size])
    side_order = ["up", "down", "left", "right"]
    palette = [1, 2, 3, 4, 5, 6]
    rng.shuffle(palette)
    template_colors = palette[:4]

    grid_side = n * cell_size + (n + 1)
    grid = np.full((grid_side, grid_side), BG_COLOR, dtype=int)
    for k in range(n + 1):
        pos = k * (cell_size + 1)
        grid[pos, :] = SEP_COLOR
        grid[:, pos] = SEP_COLOR

    mid = n // 2
    template_cells = {
        "up": (0, mid),
        "down": (n - 1, mid),
        "left": (mid, 0),
        "right": (mid, n - 1),
    }

    used_signatures = set()
    for side, color in zip(side_order, template_colors):
        patch = _random_template(template_size, color, rng)
        if np.array_equal(patch, np.rot90(patch, 2)):
            return None
        sig = tuple(map(tuple, patch))
        if sig in used_signatures:
            return None
        used_signatures.add(sig)
        ri, ci = template_cells[side]
        cell_r0 = _cell_origin(ri, cell_size)
        cell_c0 = _cell_origin(ci, cell_size)
        offset = (cell_size - template_size) // 2
        _paste_patch(grid, cell_r0 + offset, cell_c0 + offset, patch)

    candidate_cells = [
        (ri, ci)
        for ri in range(n)
        for ci in range(n)
        if (ri, ci) not in template_cells.values()
    ]
    rng.shuffle(candidate_cells)

    min_markers = 4
    max_markers = max(min_markers, len(candidate_cells) - rng.randint(0, max(0, len(candidate_cells) - min_markers)))
    marker_count = rng.randint(min_markers, max_markers)
    chosen = candidate_cells[:marker_count]

    # Ensure all four branches appear.
    directions = side_order.copy()
    while len(directions) < marker_count:
        directions.append(rng.choice(side_order))
    rng.shuffle(directions)

    marker_pos = _marker_positions(cell_size)
    for (ri, ci), direction in zip(chosen, directions):
        cell_r0 = _cell_origin(ri, cell_size)
        cell_c0 = _cell_origin(ci, cell_size)
        mr, mc = marker_pos[direction]
        grid[cell_r0 + mr, cell_c0 + mc] = MARKER_COLOR

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
