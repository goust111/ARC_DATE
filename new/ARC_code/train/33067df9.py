import numpy as np

from arc_utils import get_bg


TARGET_SIZE = 26
SPACER_SIZE = 2


def _axis_spans(active_indices: list[int], axis_len: int) -> list[tuple[int, int]]:
    active = set(active_indices)
    assert active, "expected at least one logical row/column"

    spacer_count = axis_len - len(active)
    band_total = TARGET_SIZE - SPACER_SIZE * spacer_count
    assert band_total > 0, "logical bands must have positive size"
    assert band_total % len(active) == 0, "logical bands must divide the 26x26 canvas"
    band_size = band_total // len(active)

    spans = []
    pos = 0
    for idx in range(axis_len):
        size = band_size if idx in active else SPACER_SIZE
        spans.append((pos, pos + size))
        pos += size

    assert pos == TARGET_SIZE, "expanded spans must cover the 26x26 canvas"
    return spans


def main(grid: np.ndarray) -> np.ndarray:
    """Expand the sparse dot lattice into a 26x26 drawing with thick bars between adjacent matching dots.

    Input structure
    - The background is the dominant color.
    - Foreground pixels lie on logical rows and columns separated by background spacer rows/columns.
    - A logical row or column is present when it contains at least one foreground pixel.

    Rule
    - Expand every background spacer row/column to thickness 2.
    - Expand the logical rows/columns equally so the full output is 26x26.
    - Paint each foreground logical pixel as a solid rectangle of its color.
    - Connect adjacent same-color logical pixels horizontally through the intervening spacer.
    - Connect adjacent same-color logical pixels vertically only when neither endpoint is part of a horizontal same-color run.

    Key insight
    - The input is a sampled line drawing: horizontal same-color runs define the bars first, and only unclaimed same-color vertical neighbors are joined.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    active_rows = [int(r) for r in np.flatnonzero(np.any(g != bg, axis=1))]
    active_cols = [int(c) for c in np.flatnonzero(np.any(g != bg, axis=0))]
    assert active_rows and active_cols, "expected foreground logical lattice"

    row_spans = _axis_spans(active_rows, g.shape[0])
    col_spans = _axis_spans(active_cols, g.shape[1])
    col_position = {c: i for i, c in enumerate(active_cols)}

    def has_horizontal_peer(r: int, c: int, color: int) -> bool:
        j = col_position[c]
        for jj in (j - 1, j + 1):
            if 0 <= jj < len(active_cols) and int(g[r, active_cols[jj]]) == color:
                return True
        return False

    out = np.full((TARGET_SIZE, TARGET_SIZE), bg, dtype=int)

    for r in active_rows:
        r0, r1 = row_spans[r]
        for c in active_cols:
            color = int(g[r, c])
            if color == bg:
                continue
            c0, c1 = col_spans[c]
            out[r0:r1, c0:c1] = color

    for r in active_rows:
        r0, r1 = row_spans[r]
        for c_left, c_right in zip(active_cols, active_cols[1:]):
            color = int(g[r, c_left])
            if color != bg and int(g[r, c_right]) == color:
                c0 = col_spans[c_left][0]
                c1 = col_spans[c_right][1]
                out[r0:r1, c0:c1] = color

    for r_top, r_bottom in zip(active_rows, active_rows[1:]):
        r0 = row_spans[r_top][0]
        r1 = row_spans[r_bottom][1]
        for c in active_cols:
            color = int(g[r_top, c])
            if color == bg or int(g[r_bottom, c]) != color:
                continue
            if has_horizontal_peer(r_top, c, color) or has_horizontal_peer(r_bottom, c, color):
                continue
            c0, c1 = col_spans[c]
            out[r0:r1, c0:c1] = color

    return out

# --------------- generator (o5x) ---------------

# CFG: lattice dimensions and sparse foreground coverage.
LATTICE_COUNT_CHOICES = (2, 3, 4)
MAX_HOLES = 2
MIN_FILL_NUMERATOR = 3
MIN_FILL_DENOMINATOR = 4

# CFG: same-color adjacency structure.
MIN_HORIZONTAL_EDGES = 1
MAX_HORIZONTAL_EDGES = 4
VERTICAL_EDGE_PROBABILITY = 0.45
NEIGHBOR_OFFSETS = ((0, -1), (0, 1), (-1, 0), (1, 0))

# CFG: palette and validation limits.
MIN_DISTINCT_COLORS = 2
MAX_DISTINCT_COLORS = 6
COLOR_VALUES = np.arange(1, 10)
EXPECTED_OUTPUT_SHAPE = (26, 26)
GENERATION_RETRIES = 300


def _pick(seq):
    return seq[int(np.random.randint(len(seq)))]


def _neighbors(cell, r_count, c_count):
    r, c = cell
    for dr, dc in NEIGHBOR_OFFSETS:
        rr, cc = r + dr, c + dc
        if 0 <= rr < r_count and 0 <= cc < c_count:
            yield rr, cc


def _edge(a, b):
    return frozenset((a, b))


def _component_duplicate_slots(foreground, equal_edges):
    parent = {cell: cell for cell in foreground}

    def find(cell):
        while parent[cell] != cell:
            parent[cell] = parent[parent[cell]]
            cell = parent[cell]
        return cell

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for edge in equal_edges:
        a, b = tuple(edge)
        if a in foreground and b in foreground:
            union(a, b)

    sizes = {}
    for cell in foreground:
        root = find(cell)
        sizes[root] = sizes.get(root, 0) + 1
    return sum(size - 1 for size in sizes.values())


def _horizontal_edges(r_count, c_count):
    """Choose shuffled same-color horizontal edge targets."""
    all_h_edges = [
        ((r, c), (r, c + 1))
        for r in range(r_count)
        for c in range(c_count - 1)
    ]
    np.random.shuffle(all_h_edges)
    h_limit = min(MAX_HORIZONTAL_EDGES, len(all_h_edges))
    h_target = int(np.random.randint(MIN_HORIZONTAL_EDGES, h_limit + 1))
    return all_h_edges[:h_target]


def _choose_holes(r_count, c_count, protected, min_foreground):
    total = r_count * c_count
    max_holes = min(MAX_HOLES, total - min_foreground)
    if max_holes <= 0:
        return set()

    target = int(np.random.randint(max_holes + 1))
    rows_left = [c_count] * r_count
    cols_left = [r_count] * c_count
    holes = set()

    while len(holes) < target:
        safe = []
        for r in range(r_count):
            for c in range(c_count):
                cell = (r, c)
                if cell in protected or cell in holes:
                    continue
                if rows_left[r] > 1 and cols_left[c] > 1:
                    safe.append(cell)
        if not safe:
            break
        cell = _pick(safe)
        holes.add(cell)
        rows_left[cell[0]] -= 1
        cols_left[cell[1]] -= 1

    return holes


def _foreground_cells(r_count, c_count, protected, equal_edges):
    """Choose foreground lattice cells while protecting required edge endpoints."""
    total = r_count * c_count
    all_cells = {(r, c) for r in range(r_count) for c in range(c_count)}
    min_foreground = _component_duplicate_slots(all_cells, equal_edges)
    min_foreground += MIN_DISTINCT_COLORS
    fill_floor = (MIN_FILL_NUMERATOR * total + MIN_FILL_DENOMINATOR - 1) // MIN_FILL_DENOMINATOR
    min_foreground = max(min_foreground, fill_floor)
    holes = _choose_holes(r_count, c_count, protected, min_foreground)
    return {(r, c) for r in range(r_count) for c in range(c_count) if (r, c) not in holes}


def _choose_vertical_edge(r_count, c_count, foreground, h_claimed, h_equal_edges):
    """Choose and validate the optional unclaimed same-color vertical edge."""
    if np.random.random() >= VERTICAL_EDGE_PROBABILITY:
        return None, set(h_equal_edges)

    candidates = []
    for r in range(r_count - 1):
        for c in range(c_count):
            pair = ((r, c), (r + 1, c))
            if pair[0] in foreground and pair[1] in foreground:
                if pair[0] not in h_claimed and pair[1] not in h_claimed:
                    candidates.append(pair)

    if not candidates:
        return None, set(h_equal_edges)
    v_pair = _pick(candidates)
    equal_edges = set(h_equal_edges)
    trial_edges = set(equal_edges)
    trial_edges.add(_edge(*v_pair))
    distinct_slots = len(foreground) - _component_duplicate_slots(foreground, trial_edges)
    if distinct_slots < MIN_DISTINCT_COLORS:
        return None, equal_edges
    return v_pair, trial_edges


def _choose_palette(foreground, equal_edges):
    """Pick distinct colors constrained by merged component slots."""
    distinct_slots = len(foreground) - _component_duplicate_slots(foreground, equal_edges)
    max_distinct = min(MAX_DISTINCT_COLORS, distinct_slots)
    if max_distinct < MIN_DISTINCT_COLORS:
        return None

    distinct_count = int(np.random.randint(MIN_DISTINCT_COLORS, max_distinct + 1))
    return list(np.random.choice(COLOR_VALUES, size=distinct_count, replace=False))


def _color_lattice(r_count, c_count, foreground, h_edges, v_pair, palette):
    """Backtracking color assignment honoring required equal and unequal neighbors."""
    h_edge_set = {_edge(*pair) for pair in h_edges}
    v_edge = _edge(*v_pair) if v_pair is not None else None
    equal_edges = set(h_edge_set)
    if v_edge is not None:
        equal_edges.add(v_edge)

    def valid_color(cell, color, current):
        for nb in _neighbors(cell, r_count, c_count):
            if nb not in foreground or nb not in current:
                continue
            same_required = _edge(cell, nb) in equal_edges
            if same_required and color != current[nb]:
                return False
            if not same_required and color == current[nb]:
                return False
        return True

    def search(current):
        remaining_cells = [cell for cell in foreground if cell not in current]
        missing = [color for color in palette if color not in current.values()]
        if len(missing) > len(remaining_cells):
            return None
        if not remaining_cells:
            return current if not missing else None

        best_cell = None
        best_domain = None
        force_missing = len(missing) == len(remaining_cells)
        for cell in remaining_cells:
            domain = missing[:] if force_missing else palette[:]
            domain = [color for color in domain if valid_color(cell, color, current)]
            if best_domain is None or len(domain) < len(best_domain):
                best_cell = cell
                best_domain = domain
            if best_domain == []:
                break
        if not best_domain:
            return None

        order = best_domain[:]
        np.random.shuffle(order)
        if missing and not force_missing:
            order.sort(key=lambda color: color not in missing)

        for color in order:
            current[best_cell] = color
            solved = search(current)
            if solved is not None:
                return solved
            del current[best_cell]
        return None

    solved = search({})
    if solved is None:
        return None

    lattice = np.zeros((r_count, c_count), dtype=int)
    for cell, color in solved.items():
        lattice[cell] = color
    return lattice


def _build_sparse_grid(lattice):
    grid = np.zeros((2 * lattice.shape[0] + 1, 2 * lattice.shape[1] + 1), dtype=int)
    grid[1::2, 1::2] = lattice
    return grid


def _gen_attempt():
    r_count = _pick(LATTICE_COUNT_CHOICES)
    c_count = _pick(LATTICE_COUNT_CHOICES)

    h_edges = _horizontal_edges(r_count, c_count)
    h_equal_edges = {_edge(*pair) for pair in h_edges}
    protected = {cell for pair in h_edges for cell in pair}
    foreground = _foreground_cells(r_count, c_count, protected, h_equal_edges)

    v_pair, equal_edges = _choose_vertical_edge(r_count, c_count, foreground, protected, h_equal_edges)

    palette = _choose_palette(foreground, equal_edges)
    if palette is None:
        return None

    lattice = _color_lattice(r_count, c_count, foreground, h_edges, v_pair, palette)
    if lattice is None:
        return None
    if np.any(np.all(lattice == 0, axis=0)) or np.any(np.all(lattice == 0, axis=1)):
        return None

    grid = _build_sparse_grid(lattice)
    out = main(grid)
    if out.shape != EXPECTED_OUTPUT_SHAPE:
        return None
    return grid


def generate_input():
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 33067df9 input")
