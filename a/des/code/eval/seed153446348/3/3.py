from collections import Counter

import numpy as np


BLACK = 0
COLOR_POOL = tuple(range(1, 10))
CELL_SIZE_CHOICES = (2, 3)
GRID_SIDE_RANGE = (18, 28)
OCCLUDER_COLOR_COUNT_RANGE = (2, 4)
OCCLUDER_COUNT_RANGE = (3, 6)
RECT_CELLS_MIN = 1
RECT_CELLS_MAX = 3
MAX_GENERATION_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.where(mask)
    assert rows.size > 0
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _parse_legend(grid: np.ndarray) -> tuple[tuple[int, int, int, int], dict[int, int]]:
    black_mask = grid == BLACK
    assert black_mask.any(), "expected a black legend background"
    r0, c0, r1, c1 = _bbox(black_mask)
    legend = grid[r0 : r1 + 1, c0 : c1 + 1]

    mapping: dict[int, int] = {}
    for row in legend:
        nz = np.where(row != BLACK)[0]
        if nz.size == 0:
            continue
        start = int(nz.min())
        end = int(nz.max())
        run = row[start : end + 1]
        color = int(run[0])
        assert color != BLACK
        assert np.all(run == color), "legend rows must contain a single-color run"
        assert np.array_equal(nz, np.arange(start, end + 1)), "legend marker must be contiguous"
        marker_len = int(end - start + 1)
        assert marker_len in (1, 2), "legend marker length must be 1 or 2"
        assert color not in mapping, "each occluder color must have exactly one legend row"
        mapping[color] = marker_len

    assert mapping, "legend must define at least one occluder color"
    return (r0, c0, r1, c1), mapping


def _checker_parity(shape: tuple[int, int], cell_size: int, row_phase: int, col_phase: int) -> np.ndarray:
    rows = np.arange(shape[0], dtype=int)[:, None]
    cols = np.arange(shape[1], dtype=int)[None, :]
    return (((rows - row_phase) // cell_size) + ((cols - col_phase) // cell_size)) & 1


def _infer_checkerboard(
    grid: np.ndarray, board_mask: np.ndarray, colors: tuple[int, int]
) -> tuple[int, int, int, int]:
    color_a, color_b = colors
    best_score = None
    best_params = None
    max_cell = min(6, grid.shape[0], grid.shape[1])
    visible = board_mask & ((grid == color_a) | (grid == color_b))
    coverage = int(visible.sum())
    assert coverage > 0, "expected visible checkerboard pixels"
    for cell_size in range(1, max_cell + 1):
        for row_phase in range(cell_size):
            for col_phase in range(cell_size):
                parity = _checker_parity(grid.shape, cell_size, row_phase, col_phase)
                for flip in (0, 1):
                    expect_a = parity == flip
                    mismatch = int(
                        np.count_nonzero(
                            visible
                            & (
                                ((grid == color_a) & ~expect_a)
                                | ((grid == color_b) & expect_a)
                            )
                        )
                    )
                    matches = coverage - mismatch
                    score = (mismatch == 0, matches, -mismatch, cell_size)
                    if best_score is None or score > best_score:
                        best_score = score
                        best_params = (cell_size, row_phase, col_phase, flip, mismatch)

    assert best_params is not None, "failed to infer the checkerboard colors"
    cell_size, row_phase, col_phase, flip, mismatch = best_params
    assert mismatch == 0, "checkerboard fit must exactly match all visible board pixels"
    return cell_size, row_phase, col_phase, flip


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
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
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the hidden checkerboard phase, then use the black-corner legend to decide
    whether each solid occluder should be removed or permanently printed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    legend_bbox, legend_mapping = _parse_legend(g)
    board_mask = np.ones(g.shape, dtype=bool)
    r0, c0, r1, c1 = legend_bbox
    board_mask[r0 : r1 + 1, c0 : c1 + 1] = False

    board_colors = sorted(
        int(color)
        for color in np.unique(g[board_mask])
        if int(color) not in legend_mapping and int(color) != BLACK
    )
    assert len(board_colors) == 2, "expected exactly two visible checkerboard colors"
    color_a, color_b = board_colors

    cell_size, row_phase, col_phase, flip = _infer_checkerboard(g, board_mask, (color_a, color_b))
    parity = _checker_parity(g.shape, cell_size, row_phase, col_phase)
    expected = np.where(parity == flip, color_a, color_b)

    occluder_mask = board_mask & (g != color_a) & (g != color_b)
    out = g.copy()
    for comp in _components(occluder_mask):
        rows = np.array([r for r, _ in comp], dtype=int)
        cols = np.array([c for _, c in comp], dtype=int)
        color = int(g[rows[0], cols[0]])
        assert np.all(g[rows, cols] == color), "occluders must be monochrome"
        rr0, cc0, rr1, cc1 = int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())
        patch = g[rr0 : rr1 + 1, cc0 : cc1 + 1]
        assert np.all(patch == color), "each occluder must be a solid rectangle"
        mode = legend_mapping[color]
        if mode == 1:
            out[rr0 : rr1 + 1, cc0 : cc1 + 1] = expected[rr0 : rr1 + 1, cc0 : cc1 + 1]
        else:
            assert mode == 2

    out[r0 : r1 + 1, c0 : c1 + 1] = g[r0 : r1 + 1, c0 : c1 + 1]
    return out


def _checkerboard(shape: tuple[int, int], cell_size: int, colors: tuple[int, int], flip: int) -> np.ndarray:
    parity = _checker_parity(shape, cell_size, 0, 0)
    return np.where(parity == flip, colors[0], colors[1]).astype(int)


def _legend_placement(height: int, width: int, legend_h: int, legend_w: int, corner: str) -> tuple[int, int]:
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, width - legend_w
    if corner == "bl":
        return height - legend_h, 0
    assert corner == "br"
    return height - legend_h, width - legend_w


def _distribute_modes(count: int, rng: np.random.Generator) -> list[int]:
    assert count >= 2
    modes = [1, 2]
    while len(modes) < count:
        modes.append(int(rng.integers(1, 3)))
    rng.shuffle(modes)
    return modes


def _build_legend(
    grid: np.ndarray,
    legend_top: int,
    legend_left: int,
    colors: list[int],
    modes: list[int],
    rng: np.random.Generator,
) -> None:
    legend_h = 2 * len(colors) + 1
    legend_w = 4
    grid[legend_top : legend_top + legend_h, legend_left : legend_left + legend_w] = BLACK
    order = rng.permutation(len(colors))
    for idx, order_idx in enumerate(order):
        row = legend_top + 1 + 2 * idx
        color = colors[int(order_idx)]
        mode = modes[int(order_idx)]
        grid[row, legend_left + 1 : legend_left + 1 + mode] = color


def _rectangles_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    cell_size = int(rng.choice(CELL_SIZE_CHOICES))
    height = int(rng.integers(GRID_SIDE_RANGE[0], GRID_SIDE_RANGE[1] + 1))
    width = int(rng.integers(GRID_SIDE_RANGE[0], GRID_SIDE_RANGE[1] + 1))

    color_perm = list(rng.permutation(COLOR_POOL))
    board_colors = (int(color_perm[0]), int(color_perm[1]))
    occluder_color_count = int(
        rng.integers(OCCLUDER_COLOR_COUNT_RANGE[0], OCCLUDER_COLOR_COUNT_RANGE[1] + 1)
    )
    occluder_colors = [int(color) for color in color_perm[2 : 2 + occluder_color_count]]
    modes = _distribute_modes(occluder_color_count, rng)
    flip = int(rng.integers(0, 2))

    legend_h = 2 * occluder_color_count + 1
    legend_w = 4
    if height <= legend_h + cell_size * 2 or width <= legend_w + cell_size * 2:
        return None
    corner = str(rng.choice(np.array(["tl", "tr", "bl", "br"], dtype=object)))
    legend_top, legend_left = _legend_placement(height, width, legend_h, legend_w, corner)
    legend_rect = (
        legend_top,
        legend_left,
        legend_top + legend_h - 1,
        legend_left + legend_w - 1,
    )

    grid = _checkerboard((height, width), cell_size, board_colors, flip)
    _build_legend(grid, legend_top, legend_left, occluder_colors, modes, rng)

    occluder_count = int(
        rng.integers(max(OCCLUDER_COUNT_RANGE[0], occluder_color_count), OCCLUDER_COUNT_RANGE[1] + 1)
    )
    chosen_colors = list(occluder_colors)
    while len(chosen_colors) < occluder_count:
        chosen_colors.append(int(rng.choice(occluder_colors)))
    rng.shuffle(chosen_colors)

    placed: list[tuple[int, int, int, int]] = []
    metadata: list[tuple[tuple[int, int, int, int], int]] = []
    n_cell_rows = height // cell_size
    n_cell_cols = width // cell_size
    if n_cell_rows < 4 or n_cell_cols < 4:
        return None

    for color in chosen_colors:
        success = False
        for _ in range(200):
            rect_h_cells = int(rng.integers(RECT_CELLS_MIN, RECT_CELLS_MAX + 1))
            rect_w_cells = int(rng.integers(RECT_CELLS_MIN, RECT_CELLS_MAX + 1))
            if rect_h_cells >= n_cell_rows or rect_w_cells >= n_cell_cols:
                continue
            cell_r = int(rng.integers(0, n_cell_rows - rect_h_cells + 1))
            cell_c = int(rng.integers(0, n_cell_cols - rect_w_cells + 1))
            r0 = cell_r * cell_size
            c0 = cell_c * cell_size
            r1 = r0 + rect_h_cells * cell_size - 1
            c1 = c0 + rect_w_cells * cell_size - 1
            rect = (r0, c0, r1, c1)
            if _rectangles_overlap(rect, legend_rect):
                continue
            if any(_rectangles_overlap(rect, other, pad=1) for other in placed):
                continue
            placed.append(rect)
            metadata.append((rect, color))
            grid[r0 : r1 + 1, c0 : c1 + 1] = color
            success = True
            break
        if not success:
            return None

    legend_mapping = dict(zip(occluder_colors, modes))
    assert 1 in legend_mapping.values() and 2 in legend_mapping.values()

    board_visible = (grid != BLACK).copy()
    board_visible[legend_top : legend_top + legend_h, legend_left : legend_left + legend_w] = False
    visible_counts = Counter(map(int, grid[board_visible].reshape(-1)))
    if visible_counts[board_colors[0]] < cell_size * cell_size * 2:
        return None
    if visible_counts[board_colors[1]] < cell_size * cell_size * 2:
        return None

    out = main(grid)
    restored_area = 0
    permanent_area = 0
    for rect, color in metadata:
        r0, c0, r1, c1 = rect
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if legend_mapping[color] == 1:
            restored_area += area
            if np.all(out[r0 : r1 + 1, c0 : c1 + 1] == color):
                return None
        else:
            permanent_area += area
            if not np.all(out[r0 : r1 + 1, c0 : c1 + 1] == color):
                return None
    if restored_area == 0 or permanent_area == 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid checkerboard-occluder input")
