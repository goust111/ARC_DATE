from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(int(rr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in points]
    cols = [c for _r, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[counts.argmax()])


def _frame_shape(
    grid: np.ndarray,
    points: list[tuple[int, int]],
    color: int,
    bg: int,
) -> tuple[int, int] | None:
    r0, c0, r1, c1 = _bbox(points)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 4 or w < 4:
        return None
    if h - 2 not in (2, 3) or w - 2 not in (2, 3):
        return None
    if len(points) != 2 * (h + w) - 4:
        return None

    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if not np.all(sub[border] == color):
        return None
    if not np.all(sub[~border] == bg):
        return None
    return h - 2, w - 2


def _is_simple_path_component(points: list[tuple[int, int]]) -> bool:
    if len(points) == 1:
        return True
    point_set = set(points)
    degs = []
    for r, c in points:
        deg = sum((nr, nc) in point_set for nr, nc in _neighbors4(r, c))
        degs.append(deg)
    return degs.count(1) == 2 and all(deg in (1, 2) for deg in degs)


def _find_template_and_grey(
    grid: np.ndarray,
    bg: int,
) -> tuple[int, tuple[int, int], tuple[int, int, int, int]]:
    candidates: list[tuple[int, tuple[int, int], tuple[int, int, int, int]]] = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        for comp in _components(grid == color):
            shape = _frame_shape(grid, comp, color, bg)
            if shape is None:
                continue
            candidates.append((color, shape, _bbox(comp)))
    assert len(candidates) == 1, "expected exactly one grey template frame"
    grey, shape, bbox = candidates[0]
    return int(grey), (int(shape[0]), int(shape[1])), bbox


def _find_blocks(
    grid: np.ndarray,
    bg: int,
    grey: int,
) -> tuple[dict[int, tuple[int, int, int, int]], int]:
    blocks: dict[int, tuple[int, int, int, int]] = {}
    sizes: set[int] = set()
    for color in map(int, np.unique(grid)):
        if color in (bg, grey):
            continue
        comps = _components(grid == color)
        assert len(comps) == 1, "each color block must be a single component"
        comp = comps[0]
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h == w and h in (2, 3), "color blocks must be 2x2 or 3x3 squares"
        assert len(comp) == h * w, "color blocks must be solid"
        blocks[color] = (int(r0), int(c0), int(r1), int(c1))
        sizes.add(int(h))
    assert len(blocks) in (4, 6), "task uses 4 or 6 distinct colors"
    assert len(sizes) == 1, "all source squares must have the same size"
    return blocks, int(next(iter(sizes)))


def _path_edges(
    grid: np.ndarray,
    bg: int,
    grey: int,
    frame_bbox: tuple[int, int, int, int],
) -> list[tuple[int, int]]:
    r0f, c0f, r1f, c1f = frame_bbox
    edges: list[tuple[int, int]] = []
    for comp in _components(grid == grey):
        r0, c0, r1, c1 = _bbox(comp)
        if (r0, c0, r1, c1) == (r0f, c0f, r1f, c1f):
            continue
        assert _is_simple_path_component(comp), "grey connectors must be simple paths"
        adj_colors: set[int] = set()
        for r, c in comp:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                    val = int(grid[nr, nc])
                    if val not in (bg, grey):
                        adj_colors.add(val)
        assert len(adj_colors) == 2, "each path must connect exactly two colors"
        a, b = sorted(adj_colors)
        edges.append((int(a), int(b)))
    assert len(edges) == len(set(edges)), "duplicate connector edges are not allowed"
    return edges


def _solve_layout(
    colors: list[int],
    edges: list[tuple[int, int]],
    shape: tuple[int, int],
    limit: int | None = 2,
) -> list[np.ndarray]:
    rows, cols = shape
    positions = [(r, c) for r in range(rows) for c in range(cols)]
    pos_neighbors = {
        pos: {
            (nr, nc)
            for nr, nc in _neighbors4(*pos)
            if 0 <= nr < rows and 0 <= nc < cols
        }
        for pos in positions
    }
    graph = {int(color): set() for color in colors}
    for a, b in edges:
        graph[int(a)].add(int(b))
        graph[int(b)].add(int(a))

    assigned: dict[int, tuple[int, int]] = {}
    used: set[tuple[int, int]] = set()
    solutions: list[np.ndarray] = []

    def candidates_for(color: int) -> list[tuple[int, int]]:
        taken_out = [pos for pos in positions if pos not in used]
        placed_neighbors = [n for n in graph[color] if n in assigned]
        if not placed_neighbors:
            return taken_out
        out = []
        for pos in taken_out:
            if all(assigned[n] in pos_neighbors[pos] for n in placed_neighbors):
                out.append(pos)
        return out

    def partial_ok() -> bool:
        for color, pos in assigned.items():
            needed = sum(1 for n in graph[color] if n not in assigned)
            free_adj = sum(1 for p in pos_neighbors[pos] if p not in used)
            if free_adj < needed:
                return False
        return True

    def search() -> None:
        if limit is not None and len(solutions) >= limit:
            return
        if len(assigned) == len(colors):
            layout = np.zeros(shape, dtype=int)
            for color, (r, c) in assigned.items():
                layout[r, c] = int(color)
            solutions.append(layout)
            return

        remaining = [color for color in colors if color not in assigned]
        remaining.sort(key=lambda color: (len(candidates_for(color)), -len(graph[color]), color))
        color = remaining[0]
        for pos in candidates_for(color):
            assigned[color] = pos
            used.add(pos)
            if partial_ok():
                search()
            used.remove(pos)
            del assigned[color]

    search()
    return solutions


def _block_centers(block_boxes: dict[int, tuple[int, int, int, int]]) -> dict[int, tuple[float, float]]:
    return {
        color: ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
        for color, box in block_boxes.items()
    }


def _layout_alignment_score(
    layout: np.ndarray,
    centers: dict[int, tuple[float, float]],
) -> tuple[int, float]:
    colors = [int(x) for x in layout.flat]
    pos = {int(layout[r, c]): (r, c) for r in range(layout.shape[0]) for c in range(layout.shape[1])}

    pairwise = 0
    for i, a in enumerate(colors):
        ya, xa = centers[a]
        ra, ca = pos[a]
        for b in colors[i + 1 :]:
            yb, xb = centers[b]
            rb, cb = pos[b]
            if xa < xb:
                pairwise += 2 if ca < cb else (1 if ca == cb else 0)
            elif xa > xb:
                pairwise += 2 if ca > cb else (1 if ca == cb else 0)
            if ya < yb:
                pairwise += 2 if ra < rb else (1 if ra == rb else 0)
            elif ya > yb:
                pairwise += 2 if ra > rb else (1 if ra == rb else 0)

    ys = [centers[color][0] for color in colors]
    xs = [centers[color][1] for color in colors]
    y0 = min(ys)
    y1 = max(ys)
    x0 = min(xs)
    x1 = max(xs)
    residual = 0.0
    for color in colors:
        row, col = pos[color]
        if y1 == y0:
            target_row = (layout.shape[0] - 1) / 2.0
        else:
            target_row = (centers[color][0] - y0) * (layout.shape[0] - 1) / (y1 - y0)
        if x1 == x0:
            target_col = (layout.shape[1] - 1) / 2.0
        else:
            target_col = (centers[color][1] - x0) * (layout.shape[1] - 1) / (x1 - x0)
        residual += (row - target_row) ** 2 + (col - target_col) ** 2
    return pairwise, -residual


def _choose_best_layout(
    solutions: list[np.ndarray],
    centers: dict[int, tuple[float, float]],
) -> np.ndarray:
    assert solutions, "expected at least one valid layout"
    scored = [(_layout_alignment_score(layout, centers), layout) for layout in solutions]
    best_key = max(score for score, _layout in scored)
    best = [layout for score, layout in scored if score == best_key]
    assert len(best) == 1, "spatial tie-break must select a unique layout"
    return best[0]


def _render_layout(layout: np.ndarray, block_size: int) -> np.ndarray:
    rows, cols = layout.shape
    out = np.zeros((rows * block_size, cols * block_size), dtype=int)
    for r in range(rows):
        for c in range(cols):
            out[r * block_size : (r + 1) * block_size, c * block_size : (c + 1) * block_size] = int(layout[r, c])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build the compact color matrix encoded by grey path adjacencies.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    grey, shape, frame_bbox = _find_template_and_grey(g, bg)
    blocks, block_size = _find_blocks(g, bg, grey)
    assert len(blocks) == shape[0] * shape[1]

    edges = _path_edges(g, bg, grey, frame_bbox)
    solutions = _solve_layout(sorted(blocks), edges, shape, limit=None)
    chosen = _choose_best_layout(solutions, _block_centers(blocks))
    return _render_layout(chosen, block_size)


def _grid_edges(layout: np.ndarray) -> list[tuple[int, int]]:
    rows, cols = layout.shape
    edges: list[tuple[int, int]] = []
    for r in range(rows):
        for c in range(cols):
            if r + 1 < rows:
                a = int(layout[r, c])
                b = int(layout[r + 1, c])
                edges.append(tuple(sorted((a, b))))
            if c + 1 < cols:
                a = int(layout[r, c])
                b = int(layout[r, c + 1])
                edges.append(tuple(sorted((a, b))))
    return edges


def _is_connected(colors: list[int], edges: list[tuple[int, int]]) -> bool:
    graph = {int(color): set() for color in colors}
    for a, b in edges:
        graph[int(a)].add(int(b))
        graph[int(b)].add(int(a))
    seen = {int(colors[0])}
    q = deque([int(colors[0])])
    while q:
        cur = q.popleft()
        for nxt in graph[cur]:
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return len(seen) == len(colors)


def _choose_constraint_edges(layout: np.ndarray, rng: random.Random) -> list[tuple[int, int]]:
    colors = [int(x) for x in layout.flat]
    all_edges = _grid_edges(layout)
    candidates: list[list[tuple[int, int]]] = []
    for mask in range(1, 1 << len(all_edges)):
        subset = [all_edges[i] for i in range(len(all_edges)) if (mask >> i) & 1]
        if not _is_connected(colors, subset):
            continue
        candidates.append(subset)
    assert candidates, "expected at least one connected subset"

    sizes = sorted({len(edges) for edges in candidates})
    chosen_size = rng.choice(sizes)
    same_size = [edges for edges in candidates if len(edges) == chosen_size]
    return list(rng.choice(same_size))


def _boxes_overlap(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _expand_box(box: tuple[int, int, int, int], pad: int) -> tuple[int, int, int, int]:
    return box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _adjacent_color_set(grid: np.ndarray, pos: tuple[int, int], bg: int, grey: int) -> set[int]:
    r, c = pos
    out: set[int] = set()
    for nr, nc in _neighbors4(r, c):
        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
            val = int(grid[nr, nc])
            if val not in (bg, grey):
                out.add(val)
    return out


def _candidate_ports(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    color: int,
    blocked_by_grey: np.ndarray,
    bg: int,
    grey: int,
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    ports: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        ports.append((r0 - 1, c))
        ports.append((r1 + 1, c))
    for r in range(r0, r1 + 1):
        ports.append((r, c0 - 1))
        ports.append((r, c1 + 1))

    valid = []
    for r, c in ports:
        if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]):
            continue
        if blocked_by_grey[r, c] or int(grid[r, c]) != bg:
            continue
        if _adjacent_color_set(grid, (r, c), bg, grey) == {int(color)}:
            valid.append((int(r), int(c)))
    return list(dict.fromkeys(valid))


def _bfs_path(
    allowed: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    q: deque[tuple[int, int]] = deque([start])
    prev = {start: None}
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        dirs = DIR4[:]
        rng.shuffle(dirs)
        for dr, dc in dirs:
            nr = cur[0] + dr
            nc = cur[1] + dc
            nxt = (int(nr), int(nc))
            if not (0 <= nr < allowed.shape[0] and 0 <= nc < allowed.shape[1]):
                continue
            if not allowed[nr, nc] or nxt in prev:
                continue
            prev[nxt] = cur
            q.append(nxt)
    if goal not in prev:
        return None

    path: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def _valid_new_path(
    grid: np.ndarray,
    path: list[tuple[int, int]],
    start_color: int,
    end_color: int,
    bg: int,
    grey: int,
) -> bool:
    if not path:
        return False
    path_set = set(path)
    if len(path) > 1:
        degs = []
        for cell in path:
            deg = sum(nei in path_set for nei in _neighbors4(*cell))
            degs.append(deg)
        if degs.count(1) != 2 or any(deg not in (1, 2) for deg in degs):
            return False

    for i, cell in enumerate(path):
        adj = _adjacent_color_set(grid, cell, bg, grey)
        if i == 0:
            if adj != {int(start_color)}:
                return False
        elif i == len(path) - 1:
            if adj != {int(end_color)}:
                return False
        elif adj:
            return False
    return True


def _route_edge(
    grid: np.ndarray,
    block_boxes: dict[int, tuple[int, int, int, int]],
    edge: tuple[int, int],
    bg: int,
    grey: int,
    rng: random.Random,
) -> bool:
    a, b = edge
    block_mask = np.zeros_like(grid, dtype=bool)
    for box in block_boxes.values():
        r0, c0, r1, c1 = box
        block_mask[r0 : r1 + 1, c0 : c1 + 1] = True
    near_blocks = _dilate4(block_mask)

    grey_mask = grid == grey
    blocked_by_grey = _dilate4(grey_mask)
    ports_a = _candidate_ports(grid, block_boxes[a], a, blocked_by_grey, bg, grey)
    ports_b = _candidate_ports(grid, block_boxes[b], b, blocked_by_grey, bg, grey)
    rng.shuffle(ports_a)
    rng.shuffle(ports_b)
    port_pairs = [(pa, pb) for pa in ports_a for pb in ports_b]
    rng.shuffle(port_pairs)

    for start, goal in port_pairs[: min(len(port_pairs), 80)]:
        allowed = (grid == bg) & (~near_blocks) & (~blocked_by_grey)
        allowed[start] = True
        allowed[goal] = True
        path = _bfs_path(allowed, start, goal, rng)
        if path is None or not _valid_new_path(grid, path, a, b, bg, grey):
            continue
        for r, c in path:
            grid[r, c] = grey
        return True
    return False


def _sample_block_boxes(
    height: int,
    width: int,
    block_size: int,
    count: int,
    frame_bbox: tuple[int, int, int, int],
    rng: random.Random,
) -> list[tuple[int, int, int, int]] | None:
    boxes: list[tuple[int, int, int, int]] = []
    candidates = [
        (r, c)
        for r in range(2, height - block_size - 2)
        for c in range(2, width - block_size - 2)
    ]
    rng.shuffle(candidates)
    pad = 3 if block_size == 3 else 2
    frame_guard = _expand_box(frame_bbox, 2)
    for r, c in candidates:
        box = (r, c, r + block_size - 1, c + block_size - 1)
        if _boxes_overlap(_expand_box(box, pad), frame_guard):
            continue
        if any(r == prev[0] or c == prev[1] for prev in boxes):
            continue
        if any(_boxes_overlap(_expand_box(box, pad), _expand_box(prev, pad)) for prev in boxes):
            continue
        boxes.append(box)
        if len(boxes) == count:
            break
    if len(boxes) != count:
        return None
    return boxes


def _paint_box(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _paint_frame(
    grid: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    color: int,
) -> None:
    r0, c0, r1, c1 = frame_bbox
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)


def _make_attempt(rng: random.Random) -> np.ndarray | None:
    shape = rng.choice([(2, 2), (2, 3), (3, 2)])
    rows, cols = shape
    count = rows * cols
    block_size = rng.choice([2, 3])

    colors = rng.sample(PALETTE, count)
    layout = np.array(rng.sample(colors, count), dtype=int).reshape(rows, cols)
    edges = _choose_constraint_edges(layout, rng)
    all_layouts = _solve_layout([int(x) for x in layout.flat], edges, layout.shape, limit=None)
    expected = _render_layout(layout, block_size)

    height = rng.randint(22, 30)
    width = rng.randint(22, 30)
    frame_h = rows + 2
    frame_w = cols + 2
    frame_r0 = height - frame_h - rng.randint(0, 2)
    frame_c0 = rng.randint(0, 2)
    frame_bbox = (frame_r0, frame_c0, frame_r0 + frame_h - 1, frame_c0 + frame_w - 1)

    boxes = _sample_block_boxes(height, width, block_size, count, frame_bbox, rng)
    if boxes is None:
        return None

    shuffled_colors = colors[:]
    rng.shuffle(shuffled_colors)
    block_boxes = {color: box for color, box in zip(shuffled_colors, boxes, strict=False)}

    grid = np.full((height, width), BG, dtype=int)
    _paint_frame(grid, frame_bbox, GREY)
    for color, box in block_boxes.items():
        _paint_box(grid, box, color)

    centers = _block_centers(block_boxes)
    try:
        chosen = _choose_best_layout(all_layouts, centers)
    except AssertionError:
        return None
    if not np.array_equal(chosen, layout):
        return None

    route_order = list(edges)
    route_order.sort(
        key=lambda edge: (
            -abs(centers[edge[0]][0] - centers[edge[1]][0]) - abs(centers[edge[0]][1] - centers[edge[1]][1]),
            rng.random(),
        )
    )
    for edge in route_order:
        if not _route_edge(grid, block_boxes, edge, BG, GREY, rng):
            return None

    try:
        out = main(grid)
    except Exception:
        return None
    if not np.array_equal(out, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    """
    Generate scattered square blocks, grey connector paths, and a lower-left template.
    """
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _make_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
