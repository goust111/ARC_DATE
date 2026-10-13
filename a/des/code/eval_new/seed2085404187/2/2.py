import random

import numpy as np


BG = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
KINDS = ("both", "horizontal", "vertical", "none")
RANK = {"both": 2, "horizontal": 1, "vertical": 1, "none": 0}
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

CELL_SIZE_CHOICES = (4, 5)
GRID_DIM_CHOICES = (2, 3, 4)
LEGEND_COUNT_CHOICES = (3, 4)
LEGEND_SIDE_CHOICES = ("bottom", "right")
LEGEND_BLOCK = 5
GEN_RETRIES = 300


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        coords: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            coords.append((cr, cc))
            for dr, dc in DIRS4:
                nr = cr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert len(coords) > 0, "expected a non-empty mask"
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _connected(mask: np.ndarray) -> bool:
    return mask.any() and len(_components(mask)) == 1


def _classify_mask(mask: np.ndarray) -> str:
    tight = _tight_mask(mask)
    horizontal = np.array_equal(tight, np.flipud(tight))
    vertical = np.array_equal(tight, np.fliplr(tight))
    if horizontal and vertical:
        return "both"
    if horizontal:
        return "horizontal"
    if vertical:
        return "vertical"
    return "none"


def _regular_step(indices: list[int]) -> int | None:
    diffs = [b - a for a, b in zip(indices, indices[1:])]
    if not diffs or min(diffs) <= 1 or len(set(diffs)) != 1:
        return None
    return diffs[0]


def _locate_main_grid(
    grid: np.ndarray,
) -> tuple[list[np.ndarray], int, tuple[int, int, int, int], list[int], list[int]]:
    grey_components = _components(grid == GREY)
    best = None
    for idx, coords in enumerate(grey_components):
        r0, c0, r1, c1 = _bbox(coords)
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        row_seps = [r0 + r for r in range(sub.shape[0]) if np.all(sub[r, :] == GREY)]
        col_seps = [c0 + c for c in range(sub.shape[1]) if np.all(sub[:, c] == GREY)]
        row_step = _regular_step(row_seps)
        col_step = _regular_step(col_seps)
        if row_step is None or col_step is None:
            continue
        if len(row_seps) < 3 or len(col_seps) < 3:
            continue
        score = len(coords)
        if best is None or score > best[0]:
            best = (score, idx, (r0, c0, r1, c1), row_seps, col_seps)
    assert best is not None, "expected one regular grey partition grid"
    return grey_components, best[1], best[2], best[3], best[4]


def _is_legend_component(
    grid: np.ndarray, coords: np.ndarray, main_bbox: tuple[int, int, int, int]
) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    if not (r0 > main_bbox[2] or c0 > main_bbox[3]):
        return False
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h != w or h < 4:
        return False
    if len(coords) != 2 * h + 2 * w - 4:
        return False
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    if not np.all(sub[0, :] == GREY):
        return False
    if not np.all(sub[-1, :] == GREY):
        return False
    if not np.all(sub[:, 0] == GREY):
        return False
    if not np.all(sub[:, -1] == GREY):
        return False
    interior = sub[1:-1, 1:-1]
    if interior.size == 0 or np.any(interior == GREY):
        return False
    fill_colors = [int(v) for v in np.unique(interior) if int(v) not in (BG, GREY)]
    if len(fill_colors) != 1:
        return False
    return bool(np.any(interior == BG))


def _legend_mapping(
    grid: np.ndarray,
    grey_components: list[np.ndarray],
    main_idx: int,
    main_bbox: tuple[int, int, int, int],
) -> dict[str, int]:
    boxes = []
    for idx, coords in enumerate(grey_components):
        if idx == main_idx:
            continue
        if _is_legend_component(grid, coords, main_bbox):
            boxes.append(_bbox(coords))
    assert 3 <= len(boxes) <= 4, "expected three or four legend squares"

    mapping: dict[str, int] = {}
    for r0, c0, r1, c1 in boxes:
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        fill_colors = [int(v) for v in np.unique(interior) if int(v) not in (BG, GREY)]
        assert len(fill_colors) == 1, "legend square must have exactly one fill color"
        kind = _classify_mask(interior == BG)
        assert kind not in mapping, "legend symmetry kinds must be unique"
        mapping[kind] = fill_colors[0]
    return mapping


def _cell_kinds(grid: np.ndarray, row_seps: list[int], col_seps: list[int]) -> list[list[str]]:
    object_colors = set()
    kinds: list[list[str]] = []
    for i in range(len(row_seps) - 1):
        row: list[str] = []
        for j in range(len(col_seps) - 1):
            interior = grid[row_seps[i] + 1 : row_seps[i + 1], col_seps[j] + 1 : col_seps[j + 1]]
            colors = [int(v) for v in np.unique(interior) if int(v) not in (BG, GREY)]
            assert len(colors) == 1, "each subgrid must contain one pattern color"
            color = colors[0]
            object_colors.add(color)
            row.append(_classify_mask(interior == color))
        kinds.append(row)
    assert len(object_colors) == 1, "all main-cell patterns must share one drawing color"
    return kinds


def _shared_edge_color(kind_a: str, kind_b: str | None, mapping: dict[str, int]) -> int:
    if kind_b is None:
        return int(mapping[kind_a])
    rank_a = RANK[kind_a]
    rank_b = RANK[kind_b]
    if rank_a > rank_b:
        return int(mapping[kind_a])
    if rank_b > rank_a:
        return int(mapping[kind_b])
    return GREY


def _paint_main_borders(
    out: np.ndarray, row_seps: list[int], col_seps: list[int], kinds: list[list[str]], mapping: dict[str, int]
) -> None:
    n_rows = len(kinds)
    n_cols = len(kinds[0])

    for boundary_idx, row in enumerate(row_seps):
        for j in range(n_cols):
            c0 = col_seps[j] + 1
            c1 = col_seps[j + 1]
            if boundary_idx == 0:
                color = _shared_edge_color(kinds[0][j], None, mapping)
            elif boundary_idx == n_rows:
                color = _shared_edge_color(kinds[-1][j], None, mapping)
            else:
                color = _shared_edge_color(kinds[boundary_idx - 1][j], kinds[boundary_idx][j], mapping)
            out[row, c0:c1] = color

    for boundary_idx, col in enumerate(col_seps):
        for i in range(n_rows):
            r0 = row_seps[i] + 1
            r1 = row_seps[i + 1]
            if boundary_idx == 0:
                color = _shared_edge_color(kinds[i][0], None, mapping)
            elif boundary_idx == n_cols:
                color = _shared_edge_color(kinds[i][-1], None, mapping)
            else:
                color = _shared_edge_color(kinds[i][boundary_idx - 1], kinds[i][boundary_idx], mapping)
            out[r0:r1, col] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each subgrid border using the legend color for that subgrid's symmetry class.

    Input structure:
    - Background is black.
    - A large grey partition grid divides the canvas into equal-size subgrids, each
      containing one small monochrome pattern on black.
    - Separate grey-framed legend squares sit below or to the right of the main grid.
      Each legend square has one fill color and a black example pattern inside it.

    Rule:
    - Read the legend by classifying each black example pattern as horizontal-only,
      vertical-only, both horizontal and vertical, or none; the square's fill color
      is the color assigned to that symmetry class.
    - Classify every main subgrid pattern by the same symmetry test.
    - Repaint the grey border segments around each subgrid with the legend color for
      that class, while keeping the inner pattern unchanged.
    - For a border segment shared by two neighboring subgrids, use the color of the
      higher symmetry rank (both > horizontal = vertical > none); if the two ranks
      are equal, keep that shared segment grey. Separator junction pixels stay grey.

    Key insight:
    - The legend does not name the classes directly: the solver must infer the
      symmetry-to-color mapping from the legend examples, then apply it to the main grid.
    """
    g = np.asarray(grid, dtype=int)
    grey_components, main_idx, main_bbox, row_seps, col_seps = _locate_main_grid(g)
    mapping = _legend_mapping(g, grey_components, main_idx, main_bbox)
    kinds = _cell_kinds(g, row_seps, col_seps)
    for row in kinds:
        for kind in row:
            assert kind in mapping, "every main pattern class must appear in the legend"

    out = g.copy()
    _paint_main_borders(out, row_seps, col_seps, kinds, mapping)
    return out


def _random_connected_mask(
    h: int,
    w: int,
    min_cells: int,
    max_cells: int,
    rng: random.Random,
    start: tuple[int, int] | None = None,
) -> np.ndarray | None:
    total = h * w
    max_cells = min(max_cells, total)
    if min_cells > max_cells:
        return None
    target = rng.randint(min_cells, max_cells)
    if start is None:
        start = (rng.randrange(h), rng.randrange(w))
    mask = np.zeros((h, w), dtype=bool)
    mask[start] = True
    filled = {start}
    while len(filled) < target:
        frontier: list[tuple[int, int]] = []
        frontier_seen = set()
        for r, c in filled:
            for dr, dc in DIRS4:
                nr = r + dr
                nc = c + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if mask[nr, nc] or (nr, nc) in frontier_seen:
                    continue
                frontier_seen.add((nr, nc))
                frontier.append((nr, nc))
        if not frontier:
            return None
        cell = rng.choice(frontier)
        mask[cell] = True
        filled.add(cell)
    return mask


def _odd_choices(lo: int, hi: int) -> list[int]:
    return [n for n in range(lo, hi + 1) if n % 2 == 1]


def _make_symmetry_mask(kind: str, max_h: int, max_w: int, rng: random.Random) -> np.ndarray | None:
    for _ in range(200):
        if kind == "both":
            h_choices = _odd_choices(3, max_h)
            w_choices = _odd_choices(3, max_w)
            if not h_choices or not w_choices:
                return None
            h = rng.choice(h_choices)
            w = rng.choice(w_choices)
            hh = h // 2 + 1
            ww = w // 2 + 1
            base = _random_connected_mask(hh, ww, 2, max(2, hh * ww - 1), rng, start=(hh - 1, ww - 1))
            if base is None:
                continue
            mask = np.zeros((h, w), dtype=bool)
            mask[:hh, :ww] = base
            mask |= np.fliplr(mask)
            mask |= np.flipud(mask)
        elif kind == "horizontal":
            h_choices = _odd_choices(3, max_h)
            if not h_choices or max_w < 2:
                return None
            h = rng.choice(h_choices)
            w = rng.randint(2, max_w)
            hh = h // 2 + 1
            base = _random_connected_mask(
                hh, w, 2, max(2, hh * w - 1), rng, start=(hh - 1, rng.randrange(w))
            )
            if base is None:
                continue
            mask = np.zeros((h, w), dtype=bool)
            mask[:hh, :] = base
            mask |= np.flipud(mask)
        elif kind == "vertical":
            w_choices = _odd_choices(3, max_w)
            if not w_choices or max_h < 2:
                return None
            h = rng.randint(2, max_h)
            w = rng.choice(w_choices)
            ww = w // 2 + 1
            base = _random_connected_mask(
                h, ww, 2, max(2, h * ww - 1), rng, start=(rng.randrange(h), ww - 1)
            )
            if base is None:
                continue
            mask = np.zeros((h, w), dtype=bool)
            mask[:, :ww] = base
            mask |= np.fliplr(mask)
        else:
            if max_h < 2 or max_w < 2:
                return None
            h = rng.randint(2, max_h)
            w = rng.randint(2, max_w)
            mask = _random_connected_mask(h, w, 2, max(2, h * w - 1), rng)
            if mask is None:
                continue

        if not mask.any() or int(mask.sum()) == mask.size:
            continue
        if not _connected(mask):
            continue
        if _classify_mask(mask) == kind:
            return mask
    return None


def _stamp_mask(panel: np.ndarray, mask: np.ndarray, color: int, rng: random.Random) -> None:
    h, w = panel.shape
    mh, mw = mask.shape
    top = rng.randint(0, h - mh)
    left = rng.randint(0, w - mw)
    window = panel[top : top + mh, left : left + mw]
    window[mask] = color


def _has_equal_rank_neighbor(kinds: np.ndarray) -> bool:
    h, w = kinds.shape
    for r in range(h):
        for c in range(w):
            if r + 1 < h and RANK[kinds[r, c]] == RANK[kinds[r + 1, c]]:
                return True
            if c + 1 < w and RANK[kinds[r, c]] == RANK[kinds[r, c + 1]]:
                return True
    return False


def _has_diff_rank_neighbor(kinds: np.ndarray) -> bool:
    h, w = kinds.shape
    for r in range(h):
        for c in range(w):
            if r + 1 < h and RANK[kinds[r, c]] != RANK[kinds[r + 1, c]]:
                return True
            if c + 1 < w and RANK[kinds[r, c]] != RANK[kinds[r, c + 1]]:
                return True
    return False


def _draw_main_grid(
    grid: np.ndarray,
    top: int,
    left: int,
    cell_size: int,
    kind_grid: np.ndarray,
    object_color: int,
    rng: random.Random,
) -> bool:
    n_rows, n_cols = kind_grid.shape
    main_h = n_rows * cell_size + n_rows + 1
    main_w = n_cols * cell_size + n_cols + 1
    for i in range(n_rows + 1):
        r = top + i * (cell_size + 1)
        grid[r, left : left + main_w] = GREY
    for j in range(n_cols + 1):
        c = left + j * (cell_size + 1)
        grid[top : top + main_h, c] = GREY

    for i in range(n_rows):
        for j in range(n_cols):
            panel = grid[
                top + i * (cell_size + 1) + 1 : top + i * (cell_size + 1) + 1 + cell_size,
                left + j * (cell_size + 1) + 1 : left + j * (cell_size + 1) + 1 + cell_size,
            ]
            mask = _make_symmetry_mask(kind_grid[i, j], cell_size - 1, cell_size - 1, rng)
            if mask is None:
                return False
            _stamp_mask(panel, mask, object_color, rng)
    return True


def _draw_legend_block(
    grid: np.ndarray, top: int, left: int, fill_color: int, kind: str, rng: random.Random
) -> bool:
    grid[top : top + LEGEND_BLOCK, left : left + LEGEND_BLOCK] = BG
    grid[top, left : left + LEGEND_BLOCK] = GREY
    grid[top + LEGEND_BLOCK - 1, left : left + LEGEND_BLOCK] = GREY
    grid[top : top + LEGEND_BLOCK, left] = GREY
    grid[top : top + LEGEND_BLOCK, left + LEGEND_BLOCK - 1] = GREY

    interior = grid[top + 1 : top + LEGEND_BLOCK - 1, left + 1 : left + LEGEND_BLOCK - 1]
    interior[:] = fill_color
    mask = _make_symmetry_mask(kind, interior.shape[0], interior.shape[1], rng)
    if mask is None:
        return False
    _stamp_mask(interior, mask, BG, rng)
    return True


def _build_instance(rng: random.Random) -> np.ndarray | None:
    for _ in range(80):
        legend_count = rng.choice(LEGEND_COUNT_CHOICES)
        side = rng.choice(LEGEND_SIDE_CHOICES)
        cell_size = rng.choice(CELL_SIZE_CHOICES)
        n_rows = rng.choice(GRID_DIM_CHOICES)
        n_cols = rng.choice(GRID_DIM_CHOICES)

        main_h = n_rows * cell_size + n_rows + 1
        main_w = n_cols * cell_size + n_cols + 1
        legend_span = legend_count * LEGEND_BLOCK + (legend_count - 1)
        if side == "bottom":
            total_h = main_h + 1 + LEGEND_BLOCK
            total_w = max(main_w, legend_span)
        else:
            total_h = max(main_h, legend_span)
            total_w = main_w + 1 + LEGEND_BLOCK
        if total_h <= 30 and total_w <= 30:
            break
    else:
        return None

    if legend_count == 4:
        used_kinds = list(KINDS)
    else:
        omitted = rng.choice(KINDS)
        used_kinds = [kind for kind in KINDS if kind != omitted]

    cell_count = n_rows * n_cols
    cell_kinds = used_kinds[:] + [rng.choice(used_kinds) for _ in range(cell_count - len(used_kinds))]
    kind_grid = None
    for _ in range(60):
        rng.shuffle(cell_kinds)
        candidate = np.array(cell_kinds, dtype=object).reshape(n_rows, n_cols)
        if _has_equal_rank_neighbor(candidate) and _has_diff_rank_neighbor(candidate):
            kind_grid = candidate
            break
    if kind_grid is None:
        kind_grid = np.array(cell_kinds, dtype=object).reshape(n_rows, n_cols)

    chosen_colors = rng.sample(PALETTE, 1 + legend_count)
    object_color = chosen_colors[0]
    legend_colors = chosen_colors[1:]
    kind_order = used_kinds[:]
    rng.shuffle(kind_order)
    mapping = dict(zip(kind_order, legend_colors, strict=False))

    grid = np.full((total_h, total_w), BG, dtype=int)
    if side == "bottom":
        main_top = 0
        legend_top = main_h + 1
        main_left = rng.randint(0, total_w - main_w)
        legend_left = rng.randint(0, total_w - legend_span)
    else:
        main_top = rng.randint(0, total_h - main_h)
        legend_top = rng.randint(0, total_h - legend_span)
        main_left = 0
        legend_left = main_w + 1

    if not _draw_main_grid(grid, main_top, main_left, cell_size, kind_grid, object_color, rng):
        return None

    legend_items = list(mapping.items())
    rng.shuffle(legend_items)
    for idx, (kind, color) in enumerate(legend_items):
        if side == "bottom":
            block_top = legend_top
            block_left = legend_left + idx * (LEGEND_BLOCK + 1)
        else:
            block_top = legend_top + idx * (LEGEND_BLOCK + 1)
            block_left = legend_left
        if not _draw_legend_block(grid, block_top, block_left, color, kind, rng):
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != grid.shape or not np.any(out != grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _build_instance(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
