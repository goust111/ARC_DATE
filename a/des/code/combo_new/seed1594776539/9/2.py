import random
from collections import Counter, deque

import numpy as np


BG = 7       # orange
BLUE = 1     # output path color
GREY = 5     # legend separators
MAROON = 9   # legend/fill pattern color
PATH_COLORS = (0, 2, 3, 4, 6, 8)

PATTERNS = [
    np.array(
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 0, 1],
            [0, 0, 0],
            [1, 0, 1],
        ],
        dtype=bool,
    ),
]

GEN_ATTEMPTS = 400
SCENE_H_RANGE = (14, 21)
GRID_W_RANGE = (25, 30)


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _edge_points(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    (r0, c0), (r1, c1) = a, b
    if r0 == r1:
        step = 1 if c1 > c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    if c0 == c1:
        step = 1 if r1 > r0 else -1
        return [(r, c0) for r in range(r0, r1 + step, step)]
    raise AssertionError("edges must be axis-aligned")


def _group_edges(vertices: list[tuple[int, int]], groups: list[list[int]]) -> list[list[tuple[int, int]]]:
    edges = [_edge_points(vertices[i], vertices[i + 1]) for i in range(len(vertices) - 1)]
    out: list[list[tuple[int, int]]] = []
    for group in groups:
        pts = list(edges[group[0]])
        used = set(pts)
        for edge_idx in group[1:]:
            for pt in edges[edge_idx][1:]:
                assert pt not in used, "fragment pieces must not self-overlap"
                pts.append(pt)
                used.add(pt)
        out.append(pts)
    return out


def _concat_fragments(fragments: list[list[tuple[int, int]]]) -> list[tuple[int, int]]:
    walk = list(fragments[0])
    for frag in fragments[1:]:
        assert walk[-1] == frag[0]
        walk.extend(frag[1:])
    return walk


def _transform_point(pt: tuple[int, int], op: int) -> tuple[int, int]:
    r, c = pt
    if op == 0:
        return r, c
    if op == 1:
        return c, -r
    if op == 2:
        return -r, -c
    if op == 3:
        return -c, r
    if op == 4:
        return r, -c
    if op == 5:
        return -r, c
    if op == 6:
        return c, r
    if op == 7:
        return -c, -r
    raise AssertionError("unknown transform")


def _normalize_groups(groups: list[list[tuple[int, int]]]) -> list[list[tuple[int, int]]]:
    all_pts = [pt for group in groups for pt in group]
    min_r = min(r for r, _ in all_pts)
    min_c = min(c for _, c in all_pts)
    return [[(r - min_r, c - min_c) for r, c in group] for group in groups]


def _transformed_fragments(fragments: list[list[tuple[int, int]]]) -> list[list[tuple[int, int]]]:
    op = random.randrange(8)
    transformed = [[_transform_point(pt, op) for pt in frag] for frag in fragments]
    return _normalize_groups(transformed)


def _count_turns(walk: list[tuple[int, int]]) -> int:
    dirs = [(r1 - r0, c1 - c0) for (r0, c0), (r1, c1) in zip(walk, walk[1:])]
    turns = 0
    for i, cur in enumerate(dirs):
        prev = dirs[i - 1]
        if prev[0] * cur[0] + prev[1] * cur[1] == 0:
            turns += 1
    return turns


def _ordered_component_path(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cell_set = set(cells)
    deg = {pt: sum((nr, nc) in cell_set for nr, nc in _neighbors4(*pt)) for pt in cell_set}
    endpoints = [pt for pt, d in deg.items() if d == 1]
    assert len(endpoints) == 2, "each fragment must be a simple path"
    start = endpoints[0]
    end = endpoints[1]
    ordered = [start]
    prev = None
    cur = start
    while cur != end:
        nxts = [pt for pt in _neighbors4(*cur) if pt in cell_set and pt != prev]
        assert len(nxts) == 1, "fragment path must be non-branching"
        nxt = nxts[0]
        ordered.append(nxt)
        prev, cur = cur, nxt
    return ordered


def _find_legend(grid: np.ndarray, bg: int) -> tuple[int, int, list[np.ndarray]]:
    h, w = grid.shape
    for top in range(h - 3, -1, -1):
        band = grid[top : top + 3, :]
        colors = {int(v) for v in np.unique(band) if int(v) != bg}
        if not colors or not colors <= {GREY, MAROON}:
            continue
        cols = np.flatnonzero(np.any(band != bg, axis=0))
        if cols.size == 0:
            continue
        c0 = int(cols[0])
        c1 = int(cols[-1]) + 1
        sub = band[:, c0:c1]
        if sub.shape[1] < 9 or (sub.shape[1] - 1) % 4 != 0:
            continue
        sep_cols = [c for c in range(sub.shape[1]) if bool(np.all(sub[:, c] == GREY))]
        if sep_cols != list(range(0, sub.shape[1], 4)):
            continue
        patterns = [sub[:, 1 + 4 * i : 4 + 4 * i] == MAROON for i in range((sub.shape[1] - 1) // 4)]
        if len(patterns) >= 6:
            return top, c0, patterns
    raise AssertionError("legend not found")


def _extract_fragments(grid: np.ndarray, scene_bottom: int, bg: int) -> list[dict]:
    scene = grid[:scene_bottom, :]
    mask = (scene != bg) & (scene != GREY) & (scene != MAROON)
    fragments = []
    for comp in _components(mask):
        ordered = _ordered_component_path(comp)
        assert len(ordered) >= 3, "fragments must be at least 3 cells long"
        interior_colors = {int(scene[r, c]) for r, c in ordered[1:-1]}
        assert len(interior_colors) == 1, "fragment interior must have one base color"
        own_color = next(iter(interior_colors))
        end_colors = (int(scene[ordered[0]]), int(scene[ordered[-1]]))
        assert end_colors[0] != own_color and end_colors[1] != own_color, "endpoints must be recolored"
        fragments.append(
            {
                "own": own_color,
                "markers": end_colors,
                "points": ordered,
            }
        )
    assert len(fragments) >= 3, "expected several fragments"
    return fragments


def _orient_fragment(fragment: dict, enter_color: int) -> tuple[list[tuple[int, int]], int]:
    p0, p1 = fragment["markers"]
    if p0 == enter_color and p1 != enter_color:
        return list(fragment["points"]), p1
    if p1 == enter_color and p0 != enter_color:
        return list(reversed(fragment["points"])), p0
    raise AssertionError("fragment orientation is ambiguous")


def _reconstruct_walk(fragments: list[dict]) -> list[tuple[int, int]]:
    by_color = {frag["own"]: frag for frag in fragments}
    assert len(by_color) == len(fragments), "fragment colors must be unique"
    start = fragments[0]
    for start_end in (0, 1):
        sequence = []
        used = set()
        current = start
        enter_color = current["markers"][start_end]
        ok = True
        while True:
            oriented, next_frag_color = _orient_fragment(current, enter_color)
            sequence.append(oriented)
            used.add(current["own"])
            nxt = by_color.get(next_frag_color)
            if nxt is None:
                ok = False
                break
            if nxt["own"] == start["own"]:
                if len(used) != len(fragments) or int(start["markers"][start_end]) != int(current["own"]):
                    ok = False
                break
            if nxt["own"] in used:
                ok = False
                break
            enter_color = current["own"]
            current = nxt
        if not ok:
            continue

        placed_walk = [(r - sequence[0][0][0], c - sequence[0][0][1]) for r, c in sequence[0]]
        for frag in sequence[1:]:
            dr = placed_walk[-1][0] - frag[0][0]
            dc = placed_walk[-1][1] - frag[0][1]
            shifted = [(r + dr, c + dc) for r, c in frag]
            placed_walk.extend(shifted[1:])
        assert placed_walk[0] == placed_walk[-1], "assembled path must close"
        return placed_walk
    raise AssertionError("failed to reconstruct fragment order")


def _interior_mask(path_cells: set[tuple[int, int]]) -> tuple[np.ndarray, int, int]:
    min_r = min(r for r, _ in path_cells)
    max_r = max(r for r, _ in path_cells)
    min_c = min(c for _, c in path_cells)
    max_c = max(c for _, c in path_cells)
    h = max_r - min_r + 1
    w = max_c - min_c + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in path_cells:
        mask[r - min_r, c - min_c] = True

    pad = np.zeros((h + 2, w + 2), dtype=bool)
    pad[1:-1, 1:-1] = mask
    reach = np.zeros_like(pad, dtype=bool)
    q = deque([(0, 0)])
    reach[0, 0] = True
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h + 2 and 0 <= nc < w + 2 and not pad[nr, nc] and not reach[nr, nc]:
                reach[nr, nc] = True
                q.append((nr, nc))
    inside = (~reach & ~pad)[1:-1, 1:-1]
    return inside, min_r, min_c


def _render_output(walk: list[tuple[int, int]], pattern: np.ndarray, bg: int) -> np.ndarray:
    path_cells = set(walk[:-1])
    inside, min_r, min_c = _interior_mask(path_cells)
    h, w = inside.shape
    out = np.full((h, w), bg, dtype=int)
    for r in range(h):
        for c in range(w):
            if inside[r, c] and bool(pattern[r % 3, c % 3]):
                out[r, c] = MAROON
    for r, c in path_cells:
        out[r - min_r, c - min_c] = BLUE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble color-coded path fragments, count the turns of the closed walk, and fill its enclosed area with the indexed legend pattern.

    Input structure:
    - The background is orange, and the bottom of the grid contains a 3-row legend of 3x3 maroon patterns separated by grey columns.
    - Above the legend are several scattered width-1 polyline fragments; each fragment is 4-connected and uses one interior color plus two recolored endpoints.
    - Every endpoint pixel takes the color of the fragment it should connect to, so each fragment locally encodes its predecessor and successor.

    Rule:
    - Ignore the bottom legend, extract every non-background fragment, and recover each fragment's base color from its non-endpoint cells.
    - Follow the endpoint-color links to order the fragments around one closed walk, then translate each fragment so consecutive labeled endpoints overlap.
    - Count the number N of 90-degree turns in the assembled walk, take the Nth 3x3 legend pattern, crop to the reassembled path's tight bounding box, draw the full path in blue, and tile the chosen maroon pattern across every enclosed cell.

    Key insight:
    - The endpoint recolors are connection instructions rather than noise, so the puzzle first requires rebuilding a global path topology and only then using its turn count as a legend lookup key.
    """
    g = np.asarray(grid, dtype=int)
    bg = Counter(map(int, g.ravel())).most_common(1)[0][0]
    legend_top, _legend_left, patterns = _find_legend(g, bg)
    fragments = _extract_fragments(g, legend_top, bg)
    walk = _reconstruct_walk(fragments)
    turn_count = _count_turns(walk)
    assert 1 <= turn_count <= len(patterns), "turn count must map to a legend entry"
    return _render_output(walk, patterns[turn_count - 1], bg)


def _rectangle_template() -> list[list[tuple[int, int]]]:
    w = random.randint(3, 6)
    h = random.randint(3, 6)
    vertices = [(0, 0), (0, w), (h, w), (h, 0), (0, 0)]
    return _group_edges(vertices, [[0], [1], [2], [3]])


def _hexagon_template() -> list[list[tuple[int, int]]]:
    a = random.randint(2, 4)
    b = random.randint(2, 4)
    h1 = random.randint(2, 3)
    h2 = random.randint(2, 3)
    while a + b > 7 or h1 + h2 > 6:
        a = random.randint(2, 4)
        b = random.randint(2, 4)
        h1 = random.randint(2, 3)
        h2 = random.randint(2, 3)
    vertices = [(0, 0), (0, a + b), (h1, a + b), (h1, b), (h1 + h2, b), (h1 + h2, 0), (0, 0)]
    return _group_edges(vertices, [[0], [1], [2], [3], [4], [5]])


def _stem_template() -> list[list[tuple[int, int]]]:
    a = random.randint(4, 7)
    d = random.randint(2, 4)
    vertices = [(0, 0), (0, a), (d, a), (0, a), (0, 0)]
    return _group_edges(vertices, [[0], [1], [2], [3]])


def _double_stem_template() -> list[list[tuple[int, int]]]:
    a = random.randint(5, 7)
    b = random.randint(2, 4)
    while not (2 <= a - b <= 3):
        a = random.randint(5, 7)
        b = random.randint(2, 4)
    d = random.randint(2, 3)
    e = random.randint(2, 3)
    vertices = [(0, 0), (0, a), (d, a), (0, a), (0, b), (e, b), (0, b), (0, 0)]
    return _group_edges(vertices, [[0], [1], [2, 3], [4], [5, 6]])


def _build_template() -> list[list[tuple[int, int]]]:
    kind = random.choice(["rect", "hex", "stem", "double_stem"])
    if kind == "rect":
        return _rectangle_template()
    if kind == "hex":
        return _hexagon_template()
    if kind == "stem":
        return _stem_template()
    return _double_stem_template()


def _paint_fragment(points: list[tuple[int, int]], prev_color: int, own_color: int, next_color: int) -> dict:
    colors = {pt: own_color for pt in points}
    colors[points[0]] = prev_color
    colors[points[-1]] = next_color
    return {"points": points, "colors": colors, "own": own_color}


def _translated(points: list[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for r, c in points]


def _can_place(points: list[tuple[int, int]], occupied: np.ndarray) -> bool:
    h, w = occupied.shape
    for r, c in points:
        if not (0 <= r < h and 0 <= c < w):
            return False
        r0 = max(0, r - 1)
        r1 = min(h, r + 2)
        c0 = max(0, c - 1)
        c1 = min(w, c + 2)
        if bool(occupied[r0:r1, c0:c1].any()):
            return False
    return True


def _mark_occupied(points: list[tuple[int, int]], occupied: np.ndarray) -> None:
    h, w = occupied.shape
    for r, c in points:
        r0 = max(0, r - 1)
        r1 = min(h, r + 2)
        c0 = max(0, c - 1)
        c1 = min(w, c + 2)
        occupied[r0:r1, c0:c1] = True


def _place_fragments(scene_h: int, scene_w: int, fragments: list[dict]) -> np.ndarray | None:
    grid = np.full((scene_h, scene_w), BG, dtype=int)
    occupied = np.zeros((scene_h, scene_w), dtype=bool)
    order = list(range(len(fragments)))
    random.shuffle(order)
    for idx in order:
        frag = fragments[idx]
        pts = frag["points"]
        min_r = min(r for r, _ in pts)
        min_c = min(c for _, c in pts)
        norm = [(r - min_r, c - min_c) for r, c in pts]
        frag_h = max(r for r, _ in norm) + 1
        frag_w = max(c for _, c in norm) + 1
        positions = [
            (r, c)
            for r in range(scene_h - frag_h + 1)
            for c in range(scene_w - frag_w + 1)
        ]
        random.shuffle(positions)
        placed = False
        for r0, c0 in positions:
            translated = _translated(norm, r0, c0)
            if not _can_place(translated, occupied):
                continue
            for pt in translated:
                grid[pt] = frag["colors"][(pt[0] - r0 + min_r, pt[1] - c0 + min_c)]
            _mark_occupied(translated, occupied)
            placed = True
            break
        if not placed:
            return None
    return grid


def _legend_band(width: int) -> np.ndarray:
    band = np.full((3, width), BG, dtype=int)
    legend_w = 4 * len(PATTERNS) + 1
    left = random.randint(0, width - legend_w)
    for i in range(len(PATTERNS) + 1):
        band[:, left + 4 * i] = GREY
    for i, pat in enumerate(PATTERNS):
        c0 = left + 1 + 4 * i
        band[:, c0 : c0 + 3][pat] = MAROON
    return band


def _oracle_from_fragments(fragments: list[list[tuple[int, int]]]) -> np.ndarray:
    walk = _concat_fragments(fragments)
    assert walk[0] == walk[-1]
    turns = _count_turns(walk)
    assert 1 <= turns <= len(PATTERNS)
    return _render_output(walk, PATTERNS[turns - 1], BG)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        canonical = _transformed_fragments(_build_template())
        walk = _concat_fragments(canonical)
        turns = _count_turns(walk)
        if not (1 <= turns <= len(PATTERNS)):
            continue

        scene_h = random.randint(*SCENE_H_RANGE)
        grid_w = random.randint(*GRID_W_RANGE)
        fragments_n = len(canonical)
        if fragments_n > len(PATH_COLORS):
            continue

        colors = list(PATH_COLORS)
        random.shuffle(colors)
        frag_colors = colors[:fragments_n]
        painted = []
        for i, pts in enumerate(canonical):
            prev_color = frag_colors[i - 1]
            own_color = frag_colors[i]
            next_color = frag_colors[(i + 1) % fragments_n]
            if len(pts) < 3 or len(pts) > 8:
                break
            painted.append(_paint_fragment(pts, prev_color, own_color, next_color))
        else:
            scene = _place_fragments(scene_h, grid_w, painted)
            if scene is None:
                continue
            grid = np.full((scene_h + 4, grid_w), BG, dtype=int)
            grid[:scene_h, :] = scene
            grid[scene_h + 1 :, :] = _legend_band(grid_w)

            oracle = _oracle_from_fragments(canonical)
            if oracle.ndim != 2 or max(oracle.shape) > 30:
                continue
            try:
                solved = main(grid)
            except AssertionError:
                continue
            if not np.array_equal(solved, oracle):
                continue
            return grid
    raise RuntimeError("failed to generate a valid sample")
