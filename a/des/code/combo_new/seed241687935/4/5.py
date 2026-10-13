from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


BG = 0
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
GEN_RETRIES = 400
PATTERN_RETRIES = 120


def _rotate_clockwise(coords: set[tuple[int, int]]) -> set[tuple[int, int]]:
    height = max(r for r, _ in coords) + 1
    return {(c, height - 1 - r) for r, c in coords}


def _normalize(coords: set[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return frozenset((r - min_r, c - min_c) for r, c in coords)


def _arrow_shapes() -> dict[str, frozenset[tuple[int, int]]]:
    right = {(1, 0), (1, 1), (1, 2), (0, 2)}
    down = _rotate_clockwise(right)
    left = _rotate_clockwise(down)
    up = _rotate_clockwise(left)
    return {
        "right": _normalize(right),
        "down": _normalize(down),
        "left": _normalize(left),
        "up": _normalize(up),
    }


ARROW_SHAPES = _arrow_shapes()
ARROW_BY_SHAPE = {shape: direction for direction, shape in ARROW_SHAPES.items()}


def _majority_color(values: np.ndarray) -> int:
    colors, counts = np.unique(values, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _bands_from_separators(indices: list[int], limit: int) -> list[tuple[int, int]] | None:
    if not indices:
        return None
    bands: list[tuple[int, int]] = []
    start = 0
    for idx in indices:
        if idx < start:
            return None
        if idx > start:
            bands.append((start, idx))
        start = idx + 1
    if start < limit:
        bands.append((start, limit))
    if len(bands) < 2:
        return None
    size = bands[0][1] - bands[0][0]
    if size <= 0:
        return None
    if any((r1 - r0) != size for r0, r1 in bands):
        return None
    return bands


def _parse_board(g: np.ndarray) -> tuple[int, int, list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = g.shape
    candidates: list[tuple[int, int, list[tuple[int, int]], list[tuple[int, int]]]] = []
    for color0 in np.unique(g):
        color = int(color0)
        rows = [r for r in range(h) if np.all(g[r, :] == color)]
        cols = [c for c in range(w) if np.all(g[:, c] == color)]
        if len(rows) < 2 or len(cols) < 2:
            continue
        if any(b - a == 1 for a, b in zip(rows, rows[1:])):
            continue
        if any(b - a == 1 for a, b in zip(cols, cols[1:])):
            continue
        row_bands = _bands_from_separators(rows, h)
        col_bands = _bands_from_separators(cols, w)
        if row_bands is None or col_bands is None:
            continue
        candidates.append((len(rows) + len(cols), color, row_bands, col_bands))

    assert candidates, "failed to identify separator lines"
    candidates.sort(key=lambda item: item[0], reverse=True)
    best_score = candidates[0][0]
    best = [item for item in candidates if item[0] == best_score]
    assert len(best) == 1, "separator color must be unique"
    _score, sep, row_bands, col_bands = best[0]
    bg = _majority_color(g[g != sep])
    return int(sep), int(bg), row_bands, col_bands


def _cell_patch(
    g: np.ndarray,
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
    cell: tuple[int, int],
) -> np.ndarray:
    r0, r1 = row_bands[cell[0]]
    c0, c1 = col_bands[cell[1]]
    return g[r0:r1, c0:c1]


def _detect_arrow(patch: np.ndarray, bg: int) -> tuple[str, int] | None:
    coords = np.argwhere(patch != bg)
    if len(coords) != 4:
        return None
    colors = {int(v) for v in patch[patch != bg]}
    if len(colors) != 1:
        return None
    raw = {(int(r), int(c)) for r, c in coords}
    direction = ARROW_BY_SHAPE.get(_normalize(raw))
    if direction is None:
        return None
    return direction, next(iter(colors))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy each distant source stamp into the arrow's adjacent target cell and erase the arrows.

    Input structure:
    - The background is black, and one separator color forms full 1-pixel grid lines that split the board into equal rectangular cells.
    - Some cells contain small colored stamp patterns; other cells contain only one monochrome 4-pixel L-shaped arrow marker.
    - Arrow cells contain no stamp pixels, and empty cells contain only background.

    Rule:
    - Detect the separator lattice, extract every cell, and identify the unique arrow color by the cells whose monochrome 4-pixel content matches one of the four arrow orientations.
    - For each arrow, the adjacent cell in its pointing direction is the paste target; scanning farther in that same direction, the first non-arrow non-empty cell is the source stamp.
    - Erase every arrow cell to background and copy each source cell's full interior pattern into its target cell, preserving all colors and relative offsets inside the cell.

    Key insight:
    - The arrow tells you where to paste, not where the stamp comes from; the actual source is the first later patterned cell found along the same ray.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    _sep, bg, row_bands, col_bands = _parse_board(g)
    n_rows = len(row_bands)
    n_cols = len(col_bands)

    cell_patches: dict[tuple[int, int], np.ndarray] = {}
    cells_by_color: dict[int, set[tuple[int, int]]] = defaultdict(set)
    arrow_candidates: list[tuple[tuple[int, int], str, int]] = []
    for rr in range(n_rows):
        for cc in range(n_cols):
            cell = (rr, cc)
            patch = _cell_patch(g, row_bands, col_bands, cell)
            cell_patches[cell] = patch.copy()
            for color in {int(v) for v in np.unique(patch) if int(v) != bg}:
                cells_by_color[color].add(cell)
            arrow = _detect_arrow(patch, bg)
            if arrow is not None:
                direction, color = arrow
                arrow_candidates.append((cell, direction, color))

    possible_arrow_colors: list[tuple[int, int]] = []
    for color, cells in cells_by_color.items():
        candidate_cells = {cell for cell, _direction, c in arrow_candidates if c == color}
        if cells and cells == candidate_cells:
            possible_arrow_colors.append((len(candidate_cells), color))
    assert possible_arrow_colors, "failed to isolate the arrow color"

    max_count = max(count for count, _color in possible_arrow_colors)
    best_colors = [color for count, color in possible_arrow_colors if count == max_count]
    assert len(best_colors) == 1, "arrow color must be unique"
    arrow_color = best_colors[0]

    arrows: dict[tuple[int, int], str] = {}
    for cell, direction, color in arrow_candidates:
        if color == arrow_color:
            arrows[cell] = direction
    assert arrows, "expected at least one arrow"

    cell_kind: dict[tuple[int, int], str] = {}
    for cell, patch in cell_patches.items():
        if cell in arrows:
            cell_kind[cell] = "arrow"
        elif np.any(patch != bg):
            cell_kind[cell] = "pattern"
        else:
            cell_kind[cell] = "empty"

    planned_stamps: dict[tuple[int, int], np.ndarray] = {}
    for (rr, cc), direction in arrows.items():
        dr, dc = DIRS[direction]
        target = (rr + dr, cc + dc)
        assert 0 <= target[0] < n_rows and 0 <= target[1] < n_cols, "arrow target out of bounds"

        scan = target
        source: tuple[int, int] | None = None
        while 0 <= scan[0] < n_rows and 0 <= scan[1] < n_cols:
            if cell_kind[scan] == "pattern":
                source = scan
                break
            scan = (scan[0] + dr, scan[1] + dc)
        assert source is not None, "every arrow must see a source stamp"

        source_patch = cell_patches[source]
        if target in planned_stamps:
            assert np.array_equal(planned_stamps[target], source_patch), "conflicting stamps for one target"
        planned_stamps[target] = source_patch.copy()

    out = g.copy()
    for cell in arrows:
        r0, r1 = row_bands[cell[0]]
        c0, c1 = col_bands[cell[1]]
        out[r0:r1, c0:c1] = bg
    for cell, patch in planned_stamps.items():
        r0, r1 = row_bands[cell[0]]
        c0, c1 = col_bands[cell[1]]
        out[r0:r1, c0:c1] = patch
    return out


def _layout_options() -> list[tuple[int, int, int, int]]:
    options: list[tuple[int, int, int, int]] = []
    for n_rows in range(3, 6):
        for n_cols in range(3, 6):
            for cell_h in range(4, 7):
                for cell_w in range(4, 7):
                    height = n_rows * cell_h + n_rows + 1
                    width = n_cols * cell_w + n_cols + 1
                    if height <= 30 and width <= 30:
                        options.append((n_rows, n_cols, cell_h, cell_w))
    return options


LAYOUT_OPTIONS = _layout_options()
PREFERRED_LAYOUTS = [opt for opt in LAYOUT_OPTIONS if max(opt[0], opt[1]) >= 5]


def _in_bounds(cell: tuple[int, int], n_rows: int, n_cols: int) -> bool:
    return 0 <= cell[0] < n_rows and 0 <= cell[1] < n_cols


def _chain_cells(
    source: tuple[int, int], direction: str, length: int
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    dr, dc = DIRS[direction]
    arrows: list[tuple[int, int]] = []
    gaps: list[tuple[int, int]] = []
    for step in range(1, 2 * length + 1):
        cell = (source[0] - step * dr, source[1] - step * dc)
        if step % 2 == 0:
            arrows.append(cell)
        else:
            gaps.append(cell)
    return arrows, gaps


def _chain_candidates(
    state: np.ndarray,
    n_rows: int,
    n_cols: int,
) -> list[tuple[tuple[int, int], str, int]]:
    candidates: list[tuple[tuple[int, int], str, int]] = []
    for rr in range(n_rows):
        for cc in range(n_cols):
            if state[rr, cc] != 0:
                continue
            source = (rr, cc)
            for direction, (dr, dc) in DIRS.items():
                max_steps = 0
                probe = source
                while True:
                    probe = (probe[0] - dr, probe[1] - dc)
                    if not _in_bounds(probe, n_rows, n_cols):
                        break
                    max_steps += 1
                max_length = min(3, max_steps // 2)
                for length in range(1, max_length + 1):
                    arrows, gaps = _chain_cells(source, direction, length)
                    cells = [source] + arrows + gaps
                    if all(_in_bounds(cell, n_rows, n_cols) and state[cell] == 0 for cell in cells):
                        candidates.append((source, direction, length))
    return candidates


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in DIRS.values():
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _make_pattern_patch(cell_h: int, cell_w: int, palette: list[int]) -> np.ndarray | None:
    inner_h_max = min(4, cell_h)
    inner_w_max = min(4, cell_w)
    dims = [(h, w) for h in range(2, inner_h_max + 1) for w in range(2, inner_w_max + 1)]
    if not dims:
        return None

    for _ in range(PATTERN_RETRIES):
        shape_h, shape_w = random.choice(dims)
        area = random.randint(2, min(6, shape_h * shape_w))

        seed = (random.randrange(shape_h), random.randrange(shape_w))
        cells = {seed}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for rr, cc in _neighbors4(r, c, shape_h, shape_w):
                    if (rr, cc) not in cells:
                        frontier.append((rr, cc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != area:
            continue

        patch = np.full((cell_h, cell_w), BG, dtype=int)
        offset_r = random.randint(0, cell_h - shape_h)
        offset_c = random.randint(0, cell_w - shape_w)

        color_count = 1 if area <= 2 else random.choice([1, 1, 2])
        colors = random.sample(palette, color_count)
        assigned: dict[tuple[int, int], int] = {}
        for cell in cells:
            assigned[cell] = random.choice(colors)
        if color_count == 2 and len(set(assigned.values())) == 1:
            assigned[random.choice(tuple(cells))] = colors[1]

        for (r, c), color in assigned.items():
            patch[offset_r + r, offset_c + c] = color

        non_bg = np.argwhere(patch != BG)
        if len(non_bg) == 4:
            colors_used = {int(v) for v in patch[patch != BG]}
            if len(colors_used) == 1:
                norm = _normalize({(int(r), int(c)) for r, c in non_bg})
                if norm in ARROW_BY_SHAPE:
                    continue
        return patch
    return None


def _place_chain(
    state: np.ndarray,
    arrows_by_cell: dict[tuple[int, int], str],
    source_cells: set[tuple[int, int]],
    source: tuple[int, int],
    direction: str,
    length: int,
) -> int:
    arrows, gaps = _chain_cells(source, direction, length)
    state[source] = 2
    source_cells.add(source)
    for gap in gaps:
        state[gap] = 1
    for arrow in arrows:
        state[arrow] = 3
        arrows_by_cell[arrow] = direction
    return len(arrows)


def _render_grid(
    n_rows: int,
    n_cols: int,
    cell_h: int,
    cell_w: int,
    sep: int,
    arrow_color: int,
    arrows_by_cell: dict[tuple[int, int], str],
    pattern_cells: dict[tuple[int, int], np.ndarray],
) -> np.ndarray:
    height = n_rows * cell_h + n_rows + 1
    width = n_cols * cell_w + n_cols + 1
    grid = np.full((height, width), sep, dtype=int)

    row_bands = [(1 + rr * (cell_h + 1), 1 + rr * (cell_h + 1) + cell_h) for rr in range(n_rows)]
    col_bands = [(1 + cc * (cell_w + 1), 1 + cc * (cell_w + 1) + cell_w) for cc in range(n_cols)]
    for rr in range(n_rows):
        for cc in range(n_cols):
            r0, r1 = row_bands[rr]
            c0, c1 = col_bands[cc]
            grid[r0:r1, c0:c1] = BG

    for cell, patch in pattern_cells.items():
        r0, r1 = row_bands[cell[0]]
        c0, c1 = col_bands[cell[1]]
        grid[r0:r1, c0:c1] = patch

    for cell, direction in arrows_by_cell.items():
        shape = ARROW_SHAPES[direction]
        shape_h = max(r for r, _ in shape) + 1
        shape_w = max(c for _, c in shape) + 1
        offset_r = random.randint(0, cell_h - shape_h)
        offset_c = random.randint(0, cell_w - shape_w)
        r0, r1 = row_bands[cell[0]]
        c0, c1 = col_bands[cell[1]]
        for dr, dc in shape:
            grid[r0 + offset_r + dr, c0 + offset_c + dc] = arrow_color

    return grid


def _gen_attempt() -> np.ndarray | None:
    if PREFERRED_LAYOUTS and random.random() < 0.65:
        n_rows, n_cols, cell_h, cell_w = random.choice(PREFERRED_LAYOUTS)
    else:
        n_rows, n_cols, cell_h, cell_w = random.choice(LAYOUT_OPTIONS)

    sep = random.randint(1, 9)
    arrow_choices = [color for color in range(1, 10) if color != sep]
    arrow_color = random.choice(arrow_choices)
    pattern_palette = [color for color in range(1, 10) if color not in (sep, arrow_color)]
    if len(pattern_palette) < 2:
        return None

    state = np.zeros((n_rows, n_cols), dtype=int)
    arrows_by_cell: dict[tuple[int, int], str] = {}
    source_cells: set[tuple[int, int]] = set()

    target_arrows = random.randint(2, min(6, max(2, (n_rows * n_cols) // 3)))
    want_long_chain = random.random() < 0.5
    total_arrows = 0

    for _ in range(6):
        candidates = _chain_candidates(state, n_rows, n_cols)
        if not candidates:
            break
        chosen_pool = candidates
        if want_long_chain:
            long_candidates = [cand for cand in candidates if cand[2] >= 2]
            if long_candidates:
                chosen_pool = long_candidates
            want_long_chain = False
        else:
            shortfall = target_arrows - total_arrows
            fitting = [cand for cand in candidates if cand[2] <= max(1, shortfall + 1)]
            if fitting:
                chosen_pool = fitting

        source, direction, length = random.choice(chosen_pool)
        total_arrows += _place_chain(state, arrows_by_cell, source_cells, source, direction, length)
        if total_arrows >= target_arrows and len(source_cells) >= 1:
            break

    if total_arrows < 2:
        return None

    empty_cells = [(rr, cc) for rr in range(n_rows) for cc in range(n_cols) if state[rr, cc] == 0]
    random.shuffle(empty_cells)
    extra_patterns = random.randint(0, min(3, len(empty_cells)))
    extra_cells = empty_cells[:extra_patterns]

    pattern_cells: dict[tuple[int, int], np.ndarray] = {}
    for cell in list(source_cells) + extra_cells:
        patch = _make_pattern_patch(cell_h, cell_w, pattern_palette)
        if patch is None:
            return None
        pattern_cells[cell] = patch

    inp = _render_grid(n_rows, n_cols, cell_h, cell_w, sep, arrow_color, arrows_by_cell, pattern_cells)
    try:
        out = main(inp)
    except AssertionError:
        return None

    expected = inp.copy()
    _sep, _bg, row_bands, col_bands = _parse_board(inp)
    for cell in arrows_by_cell:
        r0, r1 = row_bands[cell[0]]
        c0, c1 = col_bands[cell[1]]
        expected[r0:r1, c0:c1] = BG
    for arrow_cell, direction in arrows_by_cell.items():
        dr, dc = DIRS[direction]
        target = (arrow_cell[0] + dr, arrow_cell[1] + dc)
        scan = target
        while scan not in pattern_cells:
            scan = (scan[0] + dr, scan[1] + dc)
        r0, r1 = row_bands[target[0]]
        c0, c1 = col_bands[target[1]]
        expected[r0:r1, c0:c1] = pattern_cells[scan]

    if np.array_equal(out, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stamp-and-arrow board")
