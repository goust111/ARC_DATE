import numpy as np


DIRECTIONS_8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

GENERATOR_COUNTS = (3, 4, 5, 6)
GENERATOR_TEMPLATE_DIM_MAX = 7
GENERATOR_PAIR_COUNTS = (2, 3, 4)
GENERATOR_MARGIN = 1
GENERATOR_MIN_GAP = 1
GENERATOR_RETRIES = 400
GENERATOR_TEMPLATE_RETRIES = 160
GENERATOR_CONSTELLATION_RETRIES = 160

BG = 0
SEPARATOR = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
RNG = np.random.default_rng()


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _sort_coords(coords) -> list[tuple[int, int]]:
    return sorted((int(r), int(c)) for r, c in coords)


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _normalize_points(points) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in points)
    min_c = min(c for _, c in points)
    return [(int(r - min_r), int(c - min_c)) for r, c in points]


def _aligned(a: tuple[int, int], b: tuple[int, int]) -> bool:
    dr = abs(int(a[0]) - int(b[0]))
    dc = abs(int(a[1]) - int(b[1]))
    return dr == 0 or dc == 0 or dr == dc


def _line_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = map(int, a)
    r1, c1 = map(int, b)
    dr = r1 - r0
    dc = c1 - c0
    step_r = _sign(dr)
    step_c = _sign(dc)
    if dr == 0:
        length = abs(dc)
    elif dc == 0:
        length = abs(dr)
    else:
        assert abs(dr) == abs(dc), "template edges must be horizontal, vertical, or diagonal"
        length = abs(dr)
    return [(r0 + k * step_r, c0 + k * step_c) for k in range(length + 1)]


def _bresenham_line(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = map(int, a)
    r1, c1 = map(int, b)
    points: list[tuple[int, int]] = []

    dc = abs(c1 - c0)
    dr = abs(r1 - r0)
    sc = 1 if c0 < c1 else -1
    sr = 1 if r0 < r1 else -1
    err = dc - dr

    c, r = c0, r0
    while True:
        points.append((r, c))
        if c == c1 and r == r1:
            break
        e2 = 2 * err
        if e2 > -dr:
            err -= dr
            c += sc
        if e2 < dc:
            err += dc
            r += sr
    return points


def _same_color_dirs(mask: np.ndarray, r: int, c: int) -> list[tuple[int, int]]:
    h, w = mask.shape
    dirs = []
    for dr, dc in DIRECTIONS_8:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
            dirs.append((dr, dc))
    return dirs


def _is_line_interior(mask: np.ndarray, r: int, c: int) -> bool:
    dirs = _same_color_dirs(mask, r, c)
    if len(dirs) != 2:
        return False
    d0, d1 = dirs
    return d0[0] == -d1[0] and d0[1] == -d1[1]


def _extract_template_graph(mask: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    coords = np.argwhere(mask)
    assert len(coords) > 0
    node_coords = []
    for r, c in coords:
        if not _is_line_interior(mask, int(r), int(c)):
            node_coords.append((int(r), int(c)))
    nodes = _sort_coords(node_coords)
    assert len(nodes) > 0

    node_set = set(nodes)
    node_index = {pt: idx for idx, pt in enumerate(nodes)}
    edges = set()

    for node in nodes:
        r, c = node
        for dr, dc in _same_color_dirs(mask, r, c):
            cur_r = r + dr
            cur_c = c + dc
            while (cur_r, cur_c) not in node_set:
                assert mask[cur_r, cur_c], "edge trace left template pixels"
                dirs = _same_color_dirs(mask, cur_r, cur_c)
                assert len(dirs) == 2, "non-node template pixel must have degree 2"
                assert (dr, dc) in dirs and (-dr, -dc) in dirs, "template edge must stay straight"
                cur_r += dr
                cur_c += dc
            a = node_index[node]
            b = node_index[(cur_r, cur_c)]
            if a != b:
                edges.add(tuple(sorted((a, b))))

    return nodes, sorted(edges)


def _components(mask: np.ndarray, connectivity: int = 8) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    if connectivity == 8:
        steps = DIRECTIONS_8
    else:
        steps = [(-1, 0), (0, -1), (0, 1), (1, 0)]

    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in steps:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _find_separator(grid: np.ndarray, bg: int) -> tuple[int, int]:
    h, w = grid.shape
    candidates = []
    for c in range(w):
        col = grid[:, c]
        color = int(col[0])
        if color != bg and np.all(col == color):
            candidates.append((c, color))
    assert len(candidates) == 1, "expected one full-height separator column"
    sep_col, sep_color = candidates[0]
    assert 0 < sep_col < w - 1
    return int(sep_col), int(sep_color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each right-side pixel constellation to a left-side template by point count,
    then redraw the template's point-to-point connections between the ordered
    constellation pixels.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 6 and w >= 10

    bg = _bg_color(g)
    sep_col, sep_color = _find_separator(g, bg)

    left = g[:, :sep_col]
    right = g[:, sep_col + 1 :]
    assert np.any(left != bg)
    assert np.any(right != bg)

    template_data: dict[int, dict[str, object]] = {}
    for color in sorted(int(c) for c in np.unique(left) if c not in (bg, sep_color)):
        mask = left == color
        nodes, edges = _extract_template_graph(mask)
        count = len(nodes)
        assert count not in template_data, "template counts must be unique"
        template_data[count] = {"nodes": nodes, "edges": edges, "color": color}

    constellation_data: dict[int, dict[str, object]] = {}
    for color in sorted(int(c) for c in np.unique(right) if c != bg):
        mask = right == color
        comps = _components(mask, connectivity=8)
        assert all(len(comp) == 1 for comp in comps), "constellation pixels must stay isolated"
        coords = _sort_coords((r, c + sep_col + 1) for r, c in np.argwhere(mask))
        count = len(coords)
        assert count not in constellation_data, "constellation counts must be unique"
        constellation_data[count] = {"coords": coords, "color": color}

    assert template_data and constellation_data
    assert set(template_data) == set(constellation_data), "point counts must pair templates and constellations"

    out = g.copy()
    for count, template in template_data.items():
        stars = constellation_data[count]
        color = int(stars["color"])
        coords = stars["coords"]
        for i, j in template["edges"]:
            line = _bresenham_line(coords[i], coords[j])
            for r, c in line:
                assert sep_col < c < w, "drawn lines must stay on the right half"
                assert out[r, c] in (bg, color), "constellation outputs must not overlap other colors"
                out[r, c] = color
    return out


def _render_graph_mask(nodes: list[tuple[int, int]], edges: list[tuple[int, int]]) -> np.ndarray:
    max_r = max(r for r, _ in nodes)
    max_c = max(c for _, c in nodes)
    mask = np.zeros((max_r + 1, max_c + 1), dtype=bool)
    for r, c in nodes:
        mask[r, c] = True
    for i, j in edges:
        for r, c in _line_cells(nodes[i], nodes[j]):
            mask[r, c] = True
    return mask


def _finalize_graph(points, edges):
    normalized = _normalize_points(points)
    order = sorted(range(len(normalized)), key=lambda idx: normalized[idx])
    nodes = [normalized[idx] for idx in order]
    remap = {old: new for new, old in enumerate(order)}
    edge_pairs = sorted({tuple(sorted((remap[a], remap[b]))) for a, b in edges})

    span_r = max(r for r, _ in nodes) + 1
    span_c = max(c for _, c in nodes) + 1
    if span_r > GENERATOR_TEMPLATE_DIM_MAX or span_c > GENERATOR_TEMPLATE_DIM_MAX:
        return None

    mask = _render_graph_mask(nodes, edge_pairs)
    try:
        extracted_nodes, extracted_edges = _extract_template_graph(mask)
    except AssertionError:
        return None
    if extracted_nodes != nodes or extracted_edges != edge_pairs:
        return None
    return {"nodes": nodes, "edges": edge_pairs, "mask": mask, "shape": mask.shape}


def _generate_template_graph(count: int):
    families = [
        [(-1, 0), (1, 0)],
        [(0, -1), (0, 1)],
        [(-1, -1), (1, 1)],
        [(-1, 1), (1, -1)],
    ]
    for _ in range(GENERATOR_TEMPLATE_RETRIES):
        points = [(0, 0)]
        edges = []
        ok = True
        while len(points) < count:
            placed = False
            for _ in range(80):
                parent = int(RNG.integers(len(points)))
                step = families[int(RNG.integers(len(families)))][int(RNG.integers(2))]
                length = int(RNG.integers(1, 4))
                new_point = (points[parent][0] + step[0] * length, points[parent][1] + step[1] * length)
                if new_point in points:
                    continue
                trial_points = points + [new_point]
                trial_edges = edges + [(parent, len(points))]
                graph = _finalize_graph(trial_points, trial_edges)
                if graph is None:
                    continue
                points = trial_points
                edges = trial_edges
                placed = True
                break
            if not placed:
                ok = False
                break
        if not ok:
            continue

        extra_budget = int(RNG.integers(0, min(3, count - 1)))
        pairs = [
            (i, j)
            for i in range(count)
            for j in range(i + 1, count)
            if (i, j) not in edges and _aligned(points[i], points[j])
        ]
        RNG.shuffle(pairs)
        for i, j in pairs:
            if extra_budget <= 0:
                break
            graph = _finalize_graph(points, edges + [(i, j)])
            if graph is None:
                continue
            edges.append((i, j))
            extra_budget -= 1

        graph = _finalize_graph(points, edges)
        if graph is None:
            continue
        if len(graph["edges"]) >= count - 1:
            return graph
    return None


def _stack_positions(heights: list[int], total: int) -> list[int] | None:
    required = 2 * GENERATOR_MARGIN + sum(heights) + GENERATOR_MIN_GAP * max(0, len(heights) - 1)
    if required > total:
        return None
    slack = total - required
    slots = [0] * (len(heights) + 1)
    for _ in range(slack):
        slots[int(RNG.integers(len(slots)))] += 1
    positions = []
    cur = GENERATOR_MARGIN + slots[0]
    for idx, height in enumerate(heights):
        positions.append(cur)
        cur += height
        if idx < len(heights) - 1:
            cur += GENERATOR_MIN_GAP + slots[idx + 1]
    return positions


def _constellation_ok(points: list[tuple[int, int]], edges: list[tuple[int, int]]) -> bool:
    sorted_points = _sort_coords(points)
    if max(r for r, _ in sorted_points) - min(r for r, _ in sorted_points) < 2:
        return False
    if max(c for _, c in sorted_points) - min(c for _, c in sorted_points) < 2:
        return False
    for i in range(len(sorted_points)):
        for j in range(i + 1, len(sorted_points)):
            if max(abs(sorted_points[i][0] - sorted_points[j][0]), abs(sorted_points[i][1] - sorted_points[j][1])) <= 1:
                return False

    star_set = set(sorted_points)
    seen: dict[tuple[int, int], set[tuple[int, int]]] = {}
    added_cells = set()
    for i, j in edges:
        endpoints = {sorted_points[i], sorted_points[j]}
        cells = _bresenham_line(sorted_points[i], sorted_points[j])
        if len(cells) < 3:
            return False
        for cell in cells[1:-1]:
            if cell in star_set:
                return False
        for cell in cells:
            if cell in seen and not (cell in endpoints and cell in seen[cell]):
                return False
            seen[cell] = endpoints
        added_cells.update(cells)
    return len(added_cells - star_set) > 0


def _generate_constellation(count: int, edges: list[tuple[int, int]], row0: int, row1: int, col0: int, col1: int):
    cells = [(r, c) for r in range(row0, row1) for c in range(col0, col1)]
    if len(cells) < count:
        return None
    for _ in range(GENERATOR_CONSTELLATION_RETRIES):
        chosen = RNG.choice(len(cells), size=count, replace=False)
        points = [cells[int(idx)] for idx in chosen]
        if _constellation_ok(points, edges):
            return _sort_coords(points)
    return None


def _expected_output(
    grid: np.ndarray,
    graphs: dict[int, dict[str, object]],
    constellations: dict[int, dict[str, object]],
    sep_col: int,
) -> np.ndarray:
    out = grid.copy()
    h, w = grid.shape
    for count, graph in graphs.items():
        color = int(constellations[count]["color"])
        points = constellations[count]["coords"]
        for i, j in graph["edges"]:
            for r, c in _bresenham_line(points[i], points[j]):
                assert 0 <= r < h and sep_col < c < w
                assert out[r, c] in (BG, color)
                out[r, c] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    pair_count = int(RNG.choice(GENERATOR_PAIR_COUNTS))
    counts = sorted(RNG.choice(np.array(GENERATOR_COUNTS), size=pair_count, replace=False).tolist())

    graphs = {}
    for count in counts:
        graph = _generate_template_graph(int(count))
        if graph is None:
            return None
        graphs[int(count)] = graph

    template_order = counts.copy()
    constellation_order = counts.copy()
    RNG.shuffle(template_order)
    while True:
        RNG.shuffle(constellation_order)
        if pair_count == 1 or constellation_order != template_order:
            break

    template_heights = [graphs[count]["shape"][0] for count in template_order]
    right_band_heights = [int(RNG.integers(5 + (count >= 5), 8 + (count == 6))) for count in constellation_order]

    min_h_left = 2 * GENERATOR_MARGIN + sum(template_heights) + GENERATOR_MIN_GAP * (pair_count - 1)
    min_h_right = 2 * GENERATOR_MARGIN + sum(right_band_heights) + GENERATOR_MIN_GAP * (pair_count - 1)
    height = max(min_h_left, min_h_right) + int(RNG.integers(0, 5))

    template_rows = _stack_positions(template_heights, height)
    constellation_rows = _stack_positions(right_band_heights, height)
    if template_rows is None or constellation_rows is None:
        return None

    left_w = max(graphs[count]["shape"][1] for count in counts) + int(RNG.integers(4, 7))
    right_w = int(RNG.integers(12, 17))
    width = left_w + 1 + right_w

    if pair_count * 2 > len(PALETTE):
        return None
    colors = RNG.choice(np.array(PALETTE), size=pair_count * 2, replace=False).tolist()
    template_colors = {count: int(colors[idx]) for idx, count in enumerate(template_order)}
    constellation_colors = {count: int(colors[pair_count + idx]) for idx, count in enumerate(constellation_order)}

    grid = np.full((height, width), BG, dtype=int)
    grid[:, left_w] = SEPARATOR

    for top, count in zip(template_rows, template_order):
        mask = graphs[count]["mask"]
        th, tw = mask.shape
        max_left = left_w - tw - 1
        if max_left < 1:
            return None
        left = int(RNG.integers(1, max_left + 1))
        color = template_colors[count]
        grid[top : top + th, left : left + tw][mask] = color

    constellations: dict[int, dict[str, object]] = {}
    for top, band_h, count in zip(constellation_rows, right_band_heights, constellation_order):
        points = _generate_constellation(
            count,
            graphs[count]["edges"],
            top,
            top + band_h,
            left_w + 2,
            width - 1,
        )
        if points is None:
            return None
        color = constellation_colors[count]
        for r, c in points:
            grid[r, c] = color
        constellations[count] = {"coords": points, "color": color}

    try:
        expected = _expected_output(grid, graphs, constellations, left_w)
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(expected, actual):
        return None
    if np.array_equal(actual, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATOR_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
