import random

import numpy as np


BLACK = 0
GREY = 5
CARDINALS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
]
TEMPLATE_LIBRARY = [
    ((-1, 0), (1, 0)),
    ((0, -1), (0, 1)),
    ((-1, 0), (1, 0), (0, -1), (0, 1)),
    ((-1, -1), (-1, 1), (1, -1), (1, 1)),
    tuple(DIR8),
]
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]

GEN_ATTEMPTS = 400
SEED_PLACE_ATTEMPTS = 250
LEGEND_LAYOUTS = {
    2: [(1, 2), (2, 1)],
    3: [(1, 3), (3, 1)],
    4: [(2, 2)],
}


def _corner_points(shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    return [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]


def _sign(x: int) -> int:
    if x < 0:
        return -1
    if x > 0:
        return 1
    return 0


def _component_from_start(mask: np.ndarray, start: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = mask.shape
    sr, sc = start
    if not mask[sr, sc]:
        return []
    stack = [(sr, sc)]
    seen = {start}
    points: list[tuple[int, int]] = []
    while stack:
        r, c = stack.pop()
        points.append((r, c))
        for dr, dc in CARDINALS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return points


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), max(rows), min(cols), max(cols)


def _inside_bbox(box: tuple[int, int, int, int], r: int, c: int) -> bool:
    r0, r1, c0, c1 = box
    return r0 <= r <= r1 and c0 <= c <= c1


def _full_grey_lines(arr: np.ndarray, axis: int) -> list[int]:
    if axis == 0:
        return [i for i in range(arr.shape[0]) if np.all(arr[i] == GREY)]
    return [i for i in range(arr.shape[1]) if np.all(arr[:, i] == GREY)]


def _spans_from_lines(lines: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for a, b in zip(lines, lines[1:], strict=False):
        if a + 1 <= b - 1:
            spans.append((a + 1, b - 1))
    return spans


def _legend_bbox_and_templates(grid: np.ndarray) -> tuple[tuple[int, int, int, int], dict[int, tuple[tuple[int, int], ...]]]:
    corners = [pt for pt in _corner_points(grid.shape) if int(grid[pt]) == GREY]
    assert len(corners) == 1, "expected exactly one grey-framed corner legend"
    legend_points = _component_from_start(grid == GREY, corners[0])
    assert legend_points, "corner grey component must exist"
    box = _bbox(legend_points)
    r0, r1, c0, c1 = box
    legend = grid[r0 : r1 + 1, c0 : c1 + 1]

    row_lines = _full_grey_lines(legend, axis=0)
    col_lines = _full_grey_lines(legend, axis=1)
    assert row_lines[0] == 0 and row_lines[-1] == legend.shape[0] - 1
    assert col_lines[0] == 0 and col_lines[-1] == legend.shape[1] - 1

    row_spans = _spans_from_lines(row_lines)
    col_spans = _spans_from_lines(col_lines)
    assert row_spans and col_spans, "legend must contain at least one cell"

    heights = {r1s - r0s + 1 for r0s, r1s in row_spans}
    widths = {c1s - c0s + 1 for c0s, c1s in col_spans}
    assert len(heights) == 1 and len(widths) == 1
    cell_h = heights.pop()
    cell_w = widths.pop()
    assert cell_h == cell_w and cell_h % 2 == 1, "legend cells must be odd square regions"

    templates: dict[int, tuple[tuple[int, int], ...]] = {}
    for rr0, rr1 in row_spans:
        for cc0, cc1 in col_spans:
            cell = legend[rr0 : rr1 + 1, cc0 : cc1 + 1]
            cr, cc = cell.shape[0] // 2, cell.shape[1] // 2
            color = int(cell[cr, cc])
            assert color not in {BLACK, GREY}, "legend cell center must be a colored seed"

            dirs: set[tuple[int, int]] = set()
            for ar, ac in np.argwhere(cell == color):
                dr = int(ar) - cr
                dc = int(ac) - cc
                if dr == 0 and dc == 0:
                    continue
                dirs.add((_sign(dr), _sign(dc)))
            assert dirs, "legend template must show at least one growth direction"
            assert color not in templates, "legend seed colors must be unique"
            templates[color] = tuple(sorted(dirs))

    return box, templates


def _open_steps_to_obstacle(grid: np.ndarray, r: int, c: int, dr: int, dc: int) -> int:
    h, w = grid.shape
    steps = 0
    nr, nc = r + dr, c + dc
    while 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) != GREY:
        steps += 1
        nr += dr
        nc += dc
    return steps


def _growth_layers(grid: np.ndarray, r: int, c: int) -> int:
    return min(_open_steps_to_obstacle(grid, r, c, dr, dc) for dr, dc in CARDINALS)


def _grown_output(
    grid: np.ndarray,
    legend_box: tuple[int, int, int, int],
    templates: dict[int, tuple[tuple[int, int], ...]],
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    seeds = []
    for r, c in np.argwhere((g != BLACK) & (g != GREY)):
        rr, cc = int(r), int(c)
        if _inside_bbox(legend_box, rr, cc):
            continue
        color = int(g[rr, cc])
        if color in templates:
            seeds.append((rr, cc, color))

    for r, c, color in sorted(seeds):
        layers = _growth_layers(g, r, c)
        for step in range(1, layers + 1):
            for dr, dc in templates[color]:
                nr, nc = r + dr * step, c + dc * step
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if _inside_bbox(legend_box, nr, nc):
                    continue
                if int(g[nr, nc]) == GREY:
                    continue
                out[nr, nc] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each colored seed by copying the legend's ray template until a grey wall or the grid edge blocks it.

    Input structure:
    - Background is black.
    - One corner contains a grey framed legend made of equal odd-sized cells; each cell has a colored center seed and same-colored pixels extending from the center.
    - Outside the legend, matching colors reappear as isolated single-pixel seeds, and some grey obstacle pixels or wall segments sit near them.

    Rule:
    - Read each legend cell as an eight-direction growth template: every same-colored pixel around the center marks one active compass ray.
    - For every seed outside the legend, find the matching legend color and count how many orthogonal steps remain open before the first grey obstacle or the grid edge.
    - Extend the seed along all active rays for exactly that many layers, skipping any target pixel already occupied by grey.
    - Leave the legend and all grey obstacle pixels unchanged.

    Key insight:
    - The corner legend is a color-to-template lookup table, while the actual growth size is determined independently by the nearest wall or boundary in the seed's four cardinal directions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D ARC grid"
    legend_box, templates = _legend_bbox_and_templates(g)
    return _grown_output(g, legend_box, templates)


def _legend_size(layout: tuple[int, int], cell_size: int) -> tuple[int, int]:
    rows, cols = layout
    return rows * cell_size + rows + 1, cols * cell_size + cols + 1


def _legend_bbox_for_corner(
    shape: tuple[int, int],
    legend_h: int,
    legend_w: int,
    corner: str,
) -> tuple[int, int, int, int]:
    h, w = shape
    if corner == "tl":
        return 0, legend_h - 1, 0, legend_w - 1
    if corner == "tr":
        return 0, legend_h - 1, w - legend_w, w - 1
    if corner == "bl":
        return h - legend_h, h - 1, 0, legend_w - 1
    if corner == "br":
        return h - legend_h, h - 1, w - legend_w, w - 1
    raise ValueError(f"unknown corner: {corner}")


def _mark_blocked(mask: np.ndarray, box: tuple[int, int, int, int], pad: int) -> None:
    h, w = mask.shape
    r0, r1, c0, c1 = box
    rr0 = max(0, r0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc0 = max(0, c0 - pad)
    cc1 = min(w - 1, c1 + pad)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _draw_legend(
    grid: np.ndarray,
    corner: str,
    layout: tuple[int, int],
    cell_size: int,
    items: list[tuple[int, tuple[tuple[int, int], ...]]],
) -> tuple[int, int, int, int]:
    rows, cols = layout
    legend_h, legend_w = _legend_size(layout, cell_size)
    box = _legend_bbox_for_corner(grid.shape, legend_h, legend_w, corner)
    r0, r1, c0, c1 = box

    for i in range(rows + 1):
        rr = r0 + i * (cell_size + 1)
        grid[rr, c0 : c1 + 1] = GREY
    for j in range(cols + 1):
        cc = c0 + j * (cell_size + 1)
        grid[r0 : r1 + 1, cc] = GREY

    radius = cell_size // 2
    idx = 0
    for row in range(rows):
        for col in range(cols):
            color, dirs = items[idx]
            idx += 1
            sr0 = r0 + 1 + row * (cell_size + 1)
            sc0 = c0 + 1 + col * (cell_size + 1)
            cr = sr0 + radius
            cc = sc0 + radius
            grid[cr, cc] = color
            for dr, dc in dirs:
                for step in range(1, radius + 1):
                    grid[cr + dr * step, cc + dc * step] = color
    return box


def _choose_templates(n_templates: int) -> list[tuple[tuple[int, int], ...]]:
    diagonal_templates = [
        tpl for tpl in TEMPLATE_LIBRARY if any(dr != 0 and dc != 0 for dr, dc in tpl)
    ]
    first = random.choice(diagonal_templates)
    remaining = [tpl for tpl in TEMPLATE_LIBRARY if tpl != first]
    chosen = [first] + random.sample(remaining, n_templates - 1)
    random.shuffle(chosen)
    return chosen


def _candidate_edge_position(
    shape: tuple[int, int],
    side: str,
    layers: int,
) -> tuple[int, int] | None:
    h, w = shape
    if layers < 1:
        return None
    if side == "up":
        r = layers
        if r >= h:
            return None
        lo, hi = layers, w - 1 - layers
        if lo > hi:
            return None
        return r, random.randint(lo, hi)
    if side == "down":
        r = h - 1 - layers
        if r < 0:
            return None
        lo, hi = layers, w - 1 - layers
        if lo > hi:
            return None
        return r, random.randint(lo, hi)
    if side == "left":
        c = layers
        if c >= w:
            return None
        lo, hi = layers, h - 1 - layers
        if lo > hi:
            return None
        return random.randint(lo, hi), c
    c = w - 1 - layers
    if c < 0:
        return None
    lo, hi = layers, h - 1 - layers
    if lo > hi:
        return None
    return random.randint(lo, hi), c


def _candidate_wall_position(
    shape: tuple[int, int],
    side: str,
    layers: int,
) -> tuple[tuple[int, int], set[tuple[int, int]]] | None:
    h, w = shape
    pad = layers + 1
    row_lo, row_hi = pad, h - 1 - pad
    col_lo, col_hi = pad, w - 1 - pad
    if row_lo > row_hi or col_lo > col_hi:
        return None

    r = random.randint(row_lo, row_hi)
    c = random.randint(col_lo, col_hi)
    length = 3 if random.random() < 0.75 else 1
    delta_span = [-1, 0, 1] if length == 3 else [0]
    wall_cells: set[tuple[int, int]] = set()
    if side == "up":
        wr = r - layers - 1
        for dc in delta_span:
            wc = c + dc
            if 0 <= wc < w:
                wall_cells.add((wr, wc))
    elif side == "down":
        wr = r + layers + 1
        for dc in delta_span:
            wc = c + dc
            if 0 <= wc < w:
                wall_cells.add((wr, wc))
    elif side == "left":
        wc = c - layers - 1
        for dr in delta_span:
            wr = r + dr
            if 0 <= wr < h:
                wall_cells.add((wr, wc))
    else:
        wc = c + layers + 1
        for dr in delta_span:
            wr = r + dr
            if 0 <= wr < h:
                wall_cells.add((wr, wc))
    return (r, c), wall_cells


def _free_steps_with_extra(
    grid: np.ndarray,
    r: int,
    c: int,
    dr: int,
    dc: int,
    extra_grey: set[tuple[int, int]],
) -> int:
    h, w = grid.shape
    steps = 0
    nr, nc = r + dr, c + dc
    while 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) != GREY and (nr, nc) not in extra_grey:
        steps += 1
        nr += dr
        nc += dc
    return steps


def _proposed_growth_cells(
    grid: np.ndarray,
    seed: tuple[int, int],
    dirs: tuple[tuple[int, int], ...],
    layers: int,
    extra_grey: set[tuple[int, int]],
    legend_box: tuple[int, int, int, int],
) -> set[tuple[int, int]] | None:
    h, w = grid.shape
    r, c = seed
    cells = {(r, c)}
    for step in range(1, layers + 1):
        for dr, dc in dirs:
            nr, nc = r + dr * step, c + dc * step
            if not (0 <= nr < h and 0 <= nc < w):
                return None
            if _inside_bbox(legend_box, nr, nc):
                return None
            if int(grid[nr, nc]) == GREY or (nr, nc) in extra_grey:
                continue
            if int(grid[nr, nc]) != BLACK:
                return None
            cells.add((nr, nc))
    return cells


def _reserve_box(shape: tuple[int, int], seed: tuple[int, int], layers: int) -> tuple[int, int, int, int]:
    h, w = shape
    r, c = seed
    pad = layers + 2
    return max(0, r - pad), min(h - 1, r + pad), max(0, c - pad), min(w - 1, c + pad)


def _cells_all_free(
    blocked: np.ndarray,
    cells: set[tuple[int, int]],
    legend_box: tuple[int, int, int, int],
) -> bool:
    h, w = blocked.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if blocked[r, c]:
            return False
        if _inside_bbox(legend_box, r, c):
            return False
    return True


def _place_seed(
    grid: np.ndarray,
    blocked: np.ndarray,
    legend_box: tuple[int, int, int, int],
    color: int,
    dirs: tuple[tuple[int, int], ...],
    mode: str,
    force_blocker: bool,
) -> bool:
    diagonal_dirs = [d for d in dirs if d[0] != 0 and d[1] != 0]
    for _ in range(SEED_PLACE_ATTEMPTS):
        if mode == "edge":
            layers = random.choice([1, 1, 2, 2, 3, 3, 4])
        else:
            layers = random.choice([0, 1, 1, 2, 2, 3, 4])
        side = random.choice(["up", "down", "left", "right"])

        if mode == "edge":
            seed = _candidate_edge_position(grid.shape, side, layers)
            if seed is None:
                continue
            extra_grey: set[tuple[int, int]] = set()
        else:
            wall_candidate = _candidate_wall_position(grid.shape, side, layers)
            if wall_candidate is None:
                continue
            seed, extra_grey = wall_candidate

        reserve = _reserve_box(grid.shape, seed, layers)
        rr0, rr1, cc0, cc1 = reserve
        if np.any(blocked[rr0 : rr1 + 1, cc0 : cc1 + 1]):
            continue

        blocker_cells: set[tuple[int, int]] = set()
        if diagonal_dirs and layers >= 1 and (force_blocker or random.random() < 0.35):
            k = random.randint(1, layers)
            dr, dc = random.choice(diagonal_dirs)
            blocker_cells.add((seed[0] + dr * k, seed[1] + dc * k))
        if force_blocker and diagonal_dirs and not blocker_cells:
            continue

        proposed_grey = set(extra_grey) | blocker_cells
        if not _cells_all_free(blocked, proposed_grey | {seed}, legend_box):
            continue
        if any(int(grid[r, c]) != BLACK for r, c in proposed_grey):
            continue
        if int(grid[seed]) != BLACK:
            continue

        actual_layers = min(
            _free_steps_with_extra(grid, seed[0], seed[1], dr, dc, proposed_grey)
            for dr, dc in CARDINALS
        )
        if actual_layers != layers:
            continue

        growth_cells = _proposed_growth_cells(grid, seed, dirs, layers, proposed_grey, legend_box)
        if growth_cells is None:
            continue

        grid[seed] = color
        for r, c in proposed_grey:
            grid[r, c] = GREY
        blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    n_templates = random.randint(2, 4)
    cell_size = random.choice([3, 5])
    layout = random.choice(LEGEND_LAYOUTS[n_templates])
    legend_h, legend_w = _legend_size(layout, cell_size)

    h_lo = max(legend_h + 10, 22)
    w_lo = max(legend_w + 10, 22)
    if h_lo > 30 or w_lo > 30:
        return None

    h = random.randint(h_lo, 30)
    w = random.randint(w_lo, 30)
    grid = np.full((h, w), BLACK, dtype=int)

    colors = random.sample(COLOR_POOL, n_templates)
    templates = _choose_templates(n_templates)
    legend_items = list(zip(colors, templates, strict=False))

    corner = random.choice(["tl", "tr", "bl", "br"])
    legend_box = _draw_legend(grid, corner, layout, cell_size, [(c, t) for c, t in legend_items])

    blocked = np.zeros((h, w), dtype=bool)
    _mark_blocked(blocked, legend_box, pad=1)

    seed_count = random.randint(n_templates + 1, min(n_templates + 2, 6))
    seed_specs: list[tuple[int, tuple[tuple[int, int], ...], str, bool]] = []
    base_colors = legend_items.copy()
    random.shuffle(base_colors)
    for color, dirs in base_colors:
        seed_specs.append((color, dirs, "wall", False))

    extra_count = seed_count - n_templates
    mode_pool = ["edge", "wall"] * 3
    for _ in range(extra_count):
        color, dirs = random.choice(legend_items)
        seed_specs.append((color, dirs, random.choice(mode_pool), False))

    if len(seed_specs) >= 2:
        seed_specs[0] = (seed_specs[0][0], seed_specs[0][1], "edge", seed_specs[0][3])
        seed_specs[1] = (seed_specs[1][0], seed_specs[1][1], "wall", seed_specs[1][3])

    diagonal_indices = [
        i for i, (_, dirs, _, _) in enumerate(seed_specs) if any(dr != 0 and dc != 0 for dr, dc in dirs)
    ]
    if diagonal_indices:
        idx = random.choice(diagonal_indices)
        color, dirs, mode, _ = seed_specs[idx]
        seed_specs[idx] = (color, dirs, mode, True)

    random.shuffle(seed_specs)
    for color, dirs, mode, force_blocker in seed_specs:
        if not _place_seed(grid, blocked, legend_box, color, dirs, mode, force_blocker):
            return None

    template_map = {color: tuple(sorted(dirs)) for color, dirs in legend_items}
    expected = _grown_output(grid, legend_box, template_map)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid growth-template ARC input")
