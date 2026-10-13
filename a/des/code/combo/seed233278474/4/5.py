from __future__ import annotations

import random

import numpy as np


def _constant_step(indices: list[int]) -> int | None:
    if len(indices) < 2:
        return None
    diffs = [b - a for a, b in zip(indices, indices[1:], strict=False)]
    if len(set(diffs)) != 1:
        return None
    step = diffs[0]
    return step if step > 1 else None


def _find_lattice(grid: np.ndarray) -> tuple[int, list[int], list[int], int]:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    candidates: list[tuple[int, int, int, list[int], list[int]]] = []

    for color in np.unique(g):
        rows = [r for r in range(height) if np.all(g[r, :] == int(color))]
        cols = [c for c in range(width) if np.all(g[:, c] == int(color))]
        if len(rows) < 3 or len(cols) < 3:
            continue
        row_step = _constant_step(rows)
        col_step = _constant_step(cols)
        if row_step is None or col_step is None or row_step != col_step:
            continue
        if rows[0] != 0 or rows[-1] != height - 1:
            continue
        if cols[0] != 0 or cols[-1] != width - 1:
            continue
        if (height - 1) % row_step != 0 or (width - 1) % col_step != 0:
            continue
        cell_size = row_step - 1
        if cell_size < 1:
            continue
        score = len(rows) * len(cols)
        candidates.append((score, cell_size, int(color), rows, cols))

    assert candidates, "expected a regular full-width lattice"
    candidates.sort(reverse=True)
    _score, cell_size, line_color, row_lines, col_lines = candidates[0]
    return line_color, row_lines, col_lines, cell_size


def _infer_background(grid: np.ndarray, line_color: int) -> int:
    g = np.asarray(grid, dtype=int)
    vals, counts = np.unique(g[g != line_color], return_counts=True)
    assert vals.size > 0, "expected non-lattice colors"
    return int(vals[counts.argmax()])


def _extract_cells(
    grid: np.ndarray,
    line_color: int,
    bg: int,
    row_lines: list[int],
    col_lines: list[int],
    cell_size: int,
) -> tuple[list[list[int]], list[list[np.ndarray]]]:
    g = np.asarray(grid, dtype=int)
    cell_rows = len(row_lines) - 1
    cell_cols = len(col_lines) - 1
    colors: list[list[int]] = [[0] * cell_cols for _ in range(cell_rows)]
    masks: list[list[np.ndarray]] = [[np.zeros((cell_size, cell_size), dtype=bool) for _ in range(cell_cols)] for _ in range(cell_rows)]

    for i in range(cell_rows):
        for j in range(cell_cols):
            r0 = row_lines[i] + 1
            r1 = row_lines[i + 1]
            c0 = col_lines[j] + 1
            c1 = col_lines[j + 1]
            cell = g[r0:r1, c0:c1]
            assert cell.shape == (cell_size, cell_size), "cell size mismatch"
            present = set(map(int, np.unique(cell)))
            present.discard(line_color)
            present.discard(bg)
            assert len(present) == 1, "each cell must contain exactly one shape color"
            color = present.pop()
            mask = cell == color
            assert mask.any(), "each cell must contain a nonempty shape"
            colors[i][j] = color
            masks[i][j] = mask
    return colors, masks


def main(grid: np.ndarray) -> np.ndarray:
    """
    Open channels across gray grid lines wherever same-color neighboring shapes
    touch the shared boundary at matching positions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    line_color, row_lines, col_lines, cell_size = _find_lattice(g)
    bg = _infer_background(g, line_color)
    colors, masks = _extract_cells(g, line_color, bg, row_lines, col_lines, cell_size)

    out = g.copy()
    cell_rows = len(colors)
    cell_cols = len(colors[0])

    for i in range(cell_rows):
        base_rows = row_lines[i] + 1
        for j in range(cell_cols - 1):
            if colors[i][j] != colors[i][j + 1]:
                continue
            overlap = masks[i][j][:, -1] & masks[i][j + 1][:, 0]
            if np.any(overlap):
                rows = base_rows + np.flatnonzero(overlap)
                out[rows, col_lines[j + 1]] = colors[i][j]

    for i in range(cell_rows - 1):
        boundary_row = row_lines[i + 1]
        for j in range(cell_cols):
            if colors[i][j] != colors[i + 1][j]:
                continue
            overlap = masks[i][j][-1, :] & masks[i + 1][j][0, :]
            if np.any(overlap):
                cols = col_lines[j] + 1 + np.flatnonzero(overlap)
                out[boundary_row, cols] = colors[i][j]

    return out


BLACK = 0
GRAY = 5
SHAPE_COLORS = (1, 2, 3, 4, 6, 8, 9)

MIN_CELL_ROWS = 3
MAX_CELL_ROWS = 5
MIN_CELL_COLS = 3
MAX_CELL_COLS = 5
MIN_CELL_SIZE = 4
MAX_CELL_SIZE = 7
MAX_SIDE_CONTACTS = 2
MAX_GENERATE_ATTEMPTS = 400


def _random_side_positions(pool: list[int], allow_empty: bool = True) -> set[int]:
    limit = min(MAX_SIDE_CONTACTS, len(pool))
    if allow_empty:
        count = random.choices(
            list(range(limit + 1)),
            weights=[4, 4, 2][: limit + 1],
            k=1,
        )[0]
    else:
        count = random.randint(1, limit)
    if count == 0:
        return set()
    return set(random.sample(pool, count))


def _random_disjoint_pair(pool: list[int], require_visible: bool) -> tuple[set[int], set[int]]:
    while True:
        left = _random_side_positions(pool, allow_empty=True)
        remaining = [p for p in pool if p not in left]
        right = _random_side_positions(remaining, allow_empty=True) if remaining else set()
        if not require_visible or left or right:
            return left, right


def _anchor_path(anchor: tuple[int, int], seed: tuple[int, int], cell_size: int) -> list[tuple[int, int]]:
    r, c = anchor
    if r == 0:
        cur_r, cur_c = 1, c
    elif r == cell_size - 1:
        cur_r, cur_c = cell_size - 2, c
    elif c == 0:
        cur_r, cur_c = r, 1
    else:
        cur_r, cur_c = r, cell_size - 2

    path = [anchor, (cur_r, cur_c)]
    seed_r, seed_c = seed
    first_horizontal = bool(random.getrandbits(1))

    if first_horizontal:
        while cur_c != seed_c:
            cur_c += 1 if seed_c > cur_c else -1
            path.append((cur_r, cur_c))
        while cur_r != seed_r:
            cur_r += 1 if seed_r > cur_r else -1
            path.append((cur_r, cur_c))
    else:
        while cur_r != seed_r:
            cur_r += 1 if seed_r > cur_r else -1
            path.append((cur_r, cur_c))
        while cur_c != seed_c:
            cur_c += 1 if seed_c > cur_c else -1
            path.append((cur_r, cur_c))
    return path


def _interior_neighbors(cell: tuple[int, int], cell_size: int) -> list[tuple[int, int]]:
    r, c = cell
    out: list[tuple[int, int]] = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 1 <= nr < cell_size - 1 and 1 <= nc < cell_size - 1:
            out.append((nr, nc))
    return out


def _build_shape_mask(cell_size: int, sides: dict[str, set[int]]) -> np.ndarray:
    interior_cells = [
        (r, c)
        for r in range(1, cell_size - 1)
        for c in range(1, cell_size - 1)
    ]
    assert interior_cells, "cell must have an interior"
    seed = random.choice(interior_cells)

    anchors: list[tuple[int, int]] = []
    anchors.extend((0, c) for c in sorted(sides["top"]))
    anchors.extend((cell_size - 1, c) for c in sorted(sides["bottom"]))
    anchors.extend((r, 0) for r in sorted(sides["left"]))
    anchors.extend((r, cell_size - 1) for r in sorted(sides["right"]))
    random.shuffle(anchors)

    mask_cells = {seed}
    for anchor in anchors:
        mask_cells.update(_anchor_path(anchor, seed, cell_size))

    extra_budget = random.randint(0, cell_size)
    frontier = set()
    for cell in tuple(mask_cells):
        frontier.update(_interior_neighbors(cell, cell_size))
    frontier.difference_update(mask_cells)

    while frontier and extra_budget > 0:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        mask_cells.add(cell)
        for nbr in _interior_neighbors(cell, cell_size):
            if nbr not in mask_cells:
                frontier.add(nbr)
        extra_budget -= 1

    mask = np.zeros((cell_size, cell_size), dtype=bool)
    for r, c in mask_cells:
        mask[r, c] = True

    top = {c for c in range(cell_size) if mask[0, c]}
    bottom = {c for c in range(cell_size) if mask[cell_size - 1, c]}
    left = {r for r in range(cell_size) if mask[r, 0]}
    right = {r for r in range(cell_size) if mask[r, cell_size - 1]}
    assert top == sides["top"]
    assert bottom == sides["bottom"]
    assert left == sides["left"]
    assert right == sides["right"]
    assert mask.any()
    return mask


def _render_grid(
    cell_colors: list[list[int]],
    cell_masks: list[list[np.ndarray]],
    cell_size: int,
) -> tuple[np.ndarray, list[int], list[int]]:
    cell_rows = len(cell_colors)
    cell_cols = len(cell_colors[0])
    row_lines = [i * (cell_size + 1) for i in range(cell_rows + 1)]
    col_lines = [j * (cell_size + 1) for j in range(cell_cols + 1)]
    height = row_lines[-1] + 1
    width = col_lines[-1] + 1

    grid = np.full((height, width), BLACK, dtype=int)
    grid[row_lines, :] = GRAY
    grid[:, col_lines] = GRAY

    for i in range(cell_rows):
        for j in range(cell_cols):
            patch = np.where(cell_masks[i][j], cell_colors[i][j], BLACK)
            r0 = row_lines[i] + 1
            c0 = col_lines[j] + 1
            grid[r0 : r0 + cell_size, c0 : c0 + cell_size] = patch
    return grid, row_lines, col_lines


def _expected_output(
    grid: np.ndarray,
    cell_colors: list[list[int]],
    side_maps: list[list[dict[str, set[int]]]],
    row_lines: list[int],
    col_lines: list[int],
) -> np.ndarray:
    out = grid.copy()
    cell_rows = len(cell_colors)
    cell_cols = len(cell_colors[0])

    for i in range(cell_rows):
        for j in range(cell_cols - 1):
            if cell_colors[i][j] != cell_colors[i][j + 1]:
                continue
            overlap = side_maps[i][j]["right"] & side_maps[i][j + 1]["left"]
            for pos in overlap:
                out[row_lines[i] + 1 + pos, col_lines[j + 1]] = cell_colors[i][j]

    for i in range(cell_rows - 1):
        for j in range(cell_cols):
            if cell_colors[i][j] != cell_colors[i + 1][j]:
                continue
            overlap = side_maps[i][j]["bottom"] & side_maps[i + 1][j]["top"]
            for pos in overlap:
                out[row_lines[i + 1], col_lines[j] + 1 + pos] = cell_colors[i][j]

    return out


def _stats(
    cell_colors: list[list[int]],
    side_maps: list[list[dict[str, set[int]]]],
) -> tuple[int, int, int, int]:
    cell_rows = len(cell_colors)
    cell_cols = len(cell_colors[0])
    h_connect = 0
    v_connect = 0
    same_color_blocked = 0
    diff_color_pairs = 0

    for i in range(cell_rows):
        for j in range(cell_cols - 1):
            same = cell_colors[i][j] == cell_colors[i][j + 1]
            overlap = side_maps[i][j]["right"] & side_maps[i][j + 1]["left"]
            if same and overlap:
                h_connect += 1
            elif same:
                same_color_blocked += 1
            else:
                diff_color_pairs += 1

    for i in range(cell_rows - 1):
        for j in range(cell_cols):
            same = cell_colors[i][j] == cell_colors[i + 1][j]
            overlap = side_maps[i][j]["bottom"] & side_maps[i + 1][j]["top"]
            if same and overlap:
                v_connect += 1
            elif same:
                same_color_blocked += 1
            else:
                diff_color_pairs += 1

    return h_connect, v_connect, same_color_blocked, diff_color_pairs


def _sample_case() -> tuple[list[list[int]], list[list[dict[str, set[int]]]], int] | None:
    cell_rows = random.randint(MIN_CELL_ROWS, MAX_CELL_ROWS)
    cell_cols = random.randint(MIN_CELL_COLS, MAX_CELL_COLS)
    cell_size = random.randint(MIN_CELL_SIZE, MAX_CELL_SIZE)
    pool = list(range(1, cell_size - 1))
    if not pool:
        return None

    palette_size = random.randint(2, min(4, len(SHAPE_COLORS)))
    palette = random.sample(list(SHAPE_COLORS), palette_size)
    cell_colors = [
        [random.choice(palette) for _ in range(cell_cols)]
        for _ in range(cell_rows)
    ]
    if len({color for row in cell_colors for color in row}) < 2:
        return None

    side_maps = [
        [
            {"top": set(), "bottom": set(), "left": set(), "right": set()}
            for _ in range(cell_cols)
        ]
        for _ in range(cell_rows)
    ]

    for i in range(cell_rows):
        for j in range(cell_cols - 1):
            same = cell_colors[i][j] == cell_colors[i][j + 1]
            if same and random.random() < 0.5:
                shared = _random_side_positions(pool, allow_empty=False)
                side_maps[i][j]["right"] = set(shared)
                side_maps[i][j + 1]["left"] = set(shared)
            elif same:
                left, right = _random_disjoint_pair(pool, require_visible=True)
                side_maps[i][j]["right"] = left
                side_maps[i][j + 1]["left"] = right
            else:
                side_maps[i][j]["right"] = _random_side_positions(pool, allow_empty=True)
                side_maps[i][j + 1]["left"] = _random_side_positions(pool, allow_empty=True)

    for i in range(cell_rows - 1):
        for j in range(cell_cols):
            same = cell_colors[i][j] == cell_colors[i + 1][j]
            if same and random.random() < 0.5:
                shared = _random_side_positions(pool, allow_empty=False)
                side_maps[i][j]["bottom"] = set(shared)
                side_maps[i + 1][j]["top"] = set(shared)
            elif same:
                top, bottom = _random_disjoint_pair(pool, require_visible=True)
                side_maps[i][j]["bottom"] = top
                side_maps[i + 1][j]["top"] = bottom
            else:
                side_maps[i][j]["bottom"] = _random_side_positions(pool, allow_empty=True)
                side_maps[i + 1][j]["top"] = _random_side_positions(pool, allow_empty=True)

    for i in range(cell_rows):
        side_maps[i][0]["left"] = _random_side_positions(pool, allow_empty=True)
        side_maps[i][cell_cols - 1]["right"] = _random_side_positions(pool, allow_empty=True)
    for j in range(cell_cols):
        side_maps[0][j]["top"] = _random_side_positions(pool, allow_empty=True)
        side_maps[cell_rows - 1][j]["bottom"] = _random_side_positions(pool, allow_empty=True)

    h_connect, v_connect, same_color_blocked, diff_color_pairs = _stats(cell_colors, side_maps)
    if h_connect == 0 or v_connect == 0:
        return None
    if same_color_blocked == 0 or diff_color_pairs == 0:
        return None

    return cell_colors, side_maps, cell_size


def _gen_attempt() -> np.ndarray | None:
    sampled = _sample_case()
    if sampled is None:
        return None
    cell_colors, side_maps, cell_size = sampled

    cell_rows = len(cell_colors)
    cell_cols = len(cell_colors[0])
    cell_masks: list[list[np.ndarray]] = [[np.zeros((cell_size, cell_size), dtype=bool) for _ in range(cell_cols)] for _ in range(cell_rows)]
    for i in range(cell_rows):
        for j in range(cell_cols):
            cell_masks[i][j] = _build_shape_mask(cell_size, side_maps[i][j])

    grid, row_lines, col_lines = _render_grid(cell_colors, cell_masks, cell_size)
    expected = _expected_output(grid, cell_colors, side_maps, row_lines, col_lines)

    try:
        actual = main(grid)
    except Exception:
        return None
    if np.array_equal(actual, expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
