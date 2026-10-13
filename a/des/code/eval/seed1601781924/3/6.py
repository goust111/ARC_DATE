import random

import numpy as np


BLACK = 0
GREY = 5
MAGENTA = 6
AZURE = 8

PALETTE_COLOR_POOL = (1, 2, 3, 4, 7, 9)
MAX_GENERATION_ATTEMPTS = 400

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _is_full_rect_frame(points: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(points)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    point_set = set(points)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_border = r in (r0, r1) or c in (c0, c1)
            if on_border != ((r, c) in point_set):
                return False
    return True


def _extract_palette(grid: np.ndarray) -> tuple[list[int], tuple[int, int, int, int]]:
    candidates: list[tuple[tuple[int, int], list[int], tuple[int, int, int, int]]] = []
    for comp in _components(grid == GREY):
        if not _is_full_rect_frame(comp):
            continue
        r0, c0, r1, c1 = _bbox(comp)
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        if interior.size == 0:
            continue
        row_counts = [
            int(np.count_nonzero((row != BLACK) & (row != GREY))) for row in interior
        ]
        if not row_counts or max(row_counts) < 2:
            continue
        row_idx = int(np.argmax(row_counts))
        seq = [
            int(v)
            for v in interior[row_idx]
            if int(v) not in (BLACK, GREY)
        ]
        if len(seq) < 2 or len(set(seq)) != len(seq):
            continue
        candidates.append(((r0, c0), seq, (r0, c0, r1, c1)))
    if not candidates:
        raise AssertionError("palette frame not found")
    candidates.sort(key=lambda item: item[0])
    _, palette, bbox = candidates[0]
    return palette, bbox


def _build_visibility_graph(
    markers: list[tuple[int, int]], shape: tuple[int, int]
) -> dict[tuple[int, int], set[tuple[int, int]]]:
    h, w = shape
    marker_set = set(markers)
    graph = {marker: set() for marker in markers}
    for r, c in markers:
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            while 0 <= nr < h and 0 <= nc < w:
                if (nr, nc) in marker_set:
                    graph[(r, c)].add((nr, nc))
                    graph[(nr, nc)].add((r, c))
                    break
                nr += dr
                nc += dc
    return graph


def _trace_marker_path(
    markers: list[tuple[int, int]], shape: tuple[int, int]
) -> list[tuple[int, int]]:
    if len(markers) < 2:
        raise AssertionError("need at least two markers")
    graph = _build_visibility_graph(markers, shape)
    endpoints = [marker for marker, nbrs in graph.items() if len(nbrs) == 1]
    if len(endpoints) != 2:
        raise AssertionError("markers must form one visible polyline chain")
    start = min(endpoints, key=lambda pt: (pt[0], pt[1]))
    order = [start]
    prev = None
    curr = start
    while True:
        nxts = [pt for pt in graph[curr] if pt != prev]
        if not nxts:
            break
        if len(nxts) != 1:
            raise AssertionError("polyline branch is ambiguous")
        nxt = nxts[0]
        order.append(nxt)
        prev, curr = curr, nxt
    if len(order) != len(markers):
        raise AssertionError("polyline trace did not visit every marker")
    return order


def _segment_pixels(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    ar, ac = a
    br, bc = b
    if ar == br:
        step = 1 if bc >= ac else -1
        return [(ar, c) for c in range(ac, bc + step, step)]
    if ac == bc:
        step = 1 if br >= ar else -1
        return [(r, ac) for r in range(ar, br + step, step)]
    raise AssertionError("consecutive markers must align horizontally or vertically")


def _ordered_path_pixels(vertices: list[tuple[int, int]]) -> list[tuple[int, int]]:
    pixels: list[tuple[int, int]] = []
    for idx in range(len(vertices) - 1):
        seg = _segment_pixels(vertices[idx], vertices[idx + 1])
        if idx > 0:
            seg = seg[1:]
        pixels.extend(seg)
    return pixels


def _render_output(
    grid: np.ndarray, palette: list[int], vertices: list[tuple[int, int]]
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for idx, (r, c) in enumerate(_ordered_path_pixels(vertices)):
        out[r, c] = int(palette[idx % len(palette)])
    for r, c in vertices:
        out[r, c] = AZURE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild an orthogonal polyline from magenta markers and color it cyclically.
    """
    g = np.asarray(grid, dtype=int)
    palette, _ = _extract_palette(g)
    markers = [tuple(map(int, pt)) for pt in np.argwhere(g == MAGENTA)]
    vertices = _trace_marker_path(markers, g.shape)
    return _render_output(g, palette, vertices)


def _normalize_vertices(vertices: list[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in vertices)
    min_c = min(c for _, c in vertices)
    return [(r - min_r, c - min_c) for r, c in vertices]


def _same_row_vertices(rng: random.Random, palette_len: int) -> list[tuple[int, int]]:
    if rng.random() < 0.35:
        hlen = rng.randint(palette_len + 2, palette_len + 8)
        hdir = rng.choice([-1, 1])
        return _normalize_vertices([(0, 0), (0, hdir * hlen)])

    vlen = rng.randint(2, 7)
    hlen = rng.randint(max(3, palette_len - 1), palette_len + 6)
    vdir = rng.choice([-1, 1])
    hdir = rng.choice([-1, 1])
    vertices = [
        (0, 0),
        (vdir * vlen, 0),
        (vdir * vlen, hdir * hlen),
        (0, hdir * hlen),
    ]
    return _normalize_vertices(vertices)


def _diff_row_vertices(rng: random.Random, palette_len: int) -> list[tuple[int, int]]:
    if rng.random() < 0.2:
        vlen = rng.randint(palette_len + 2, palette_len + 8)
        return [(0, 0), (vlen, 0)]

    start_horizontal = rng.random() < 0.5
    n_segments = rng.randint(2, 5)
    hdir = rng.choice([-1, 1])
    vdir = rng.choice([-1, 1])
    lengths = [rng.randint(2, 6) for _ in range(n_segments)]
    if sum(lengths) < palette_len + 3:
        lengths[-1] += palette_len + 3 - sum(lengths)

    r = 0
    c = 0
    vertices = [(r, c)]
    horizontal = start_horizontal
    for length in lengths:
        if horizontal:
            c += hdir * length
        else:
            r += vdir * length
        vertices.append((r, c))
        horizontal = not horizontal
    return _normalize_vertices(vertices)


def _place_vertices(
    rng: random.Random,
    vertices: list[tuple[int, int]],
    palette_len: int,
) -> tuple[np.ndarray, list[tuple[int, int]], list[int]]:
    palette = rng.sample(PALETTE_COLOR_POOL, palette_len)
    palette_w = palette_len + 2
    row_offset = 4 + rng.randint(0, 3)
    col_offset = rng.randint(0, 3)
    placed = [(r + row_offset, c + col_offset) for r, c in vertices]

    max_r = max(r for r, _ in placed)
    max_c = max(c for _, c in placed)
    h = max_r + 1 + rng.randint(2, 5)
    w = max(max_c + 1 + rng.randint(2, 5), palette_w + rng.randint(0, 4))

    grid = np.full((h, w), BLACK, dtype=int)
    grid[0, :palette_w] = GREY
    grid[2, :palette_w] = GREY
    grid[1, 0] = GREY
    grid[1, palette_w - 1] = GREY
    grid[1, 1 : palette_w - 1] = np.array(palette, dtype=int)

    for r, c in placed:
        grid[r, c] = MAGENTA
    return grid, placed, palette


def _valid_generated_case(
    grid: np.ndarray,
    placed_vertices: list[tuple[int, int]],
    same_row_endpoints: bool,
    palette_len: int,
) -> bool:
    try:
        traced = _trace_marker_path(placed_vertices, grid.shape)
        extracted_palette, _ = _extract_palette(grid)
    except AssertionError:
        return False

    if len(extracted_palette) != palette_len:
        return False
    if (traced[0][0] == traced[-1][0]) != same_row_endpoints:
        return False
    if len(_ordered_path_pixels(traced)) < palette_len + 3:
        return False
    try:
        out = main(grid)
    except AssertionError:
        return False
    return isinstance(out, np.ndarray) and out.shape == grid.shape


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    palette_len = rng.randint(2, 5)
    same_row_endpoints = rng.random() < 0.5
    if same_row_endpoints:
        vertices = _same_row_vertices(rng, palette_len)
    else:
        vertices = _diff_row_vertices(rng, palette_len)

    grid, placed_vertices, _ = _place_vertices(rng, vertices, palette_len)
    if not _valid_generated_case(grid, placed_vertices, same_row_endpoints, palette_len):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
