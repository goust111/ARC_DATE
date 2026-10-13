from collections import Counter, defaultdict, deque

import numpy as np


ORTH_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
CATEGORY_ORDER = ("straight", "corner", "tee", "cross")
_RNG = np.random.default_rng()


def _background_color(grid: np.ndarray) -> int:
    h, w = grid.shape
    if h == 1 or w == 1:
        border = grid.ravel()
    else:
        border = np.concatenate(
            [grid[0, :], grid[-1, :], grid[1:-1, 0], grid[1:-1, -1]]
        )
    counts = np.bincount(border.astype(int), minlength=max(10, int(grid.max()) + 1))
    return int(np.argmax(counts))


def _classify_open_dirs(open_dirs: tuple[bool, bool, bool, bool]) -> str | None:
    up, right, down, left = open_dirs
    degree = int(sum(open_dirs))
    if degree == 1:
        return "endpoint"
    if degree == 2:
        if (up and down) or (left and right):
            return "straight"
        return "corner"
    if degree == 3:
        return "tee"
    if degree == 4:
        return "cross"
    return None


def _value_components(grid: np.ndarray, value: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != value:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords = []
            touches_border = r in (0, h - 1) or c in (0, w - 1)
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in ORTH_DIRS:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != value:
                        continue
                    seen[nr, nc] = True
                    if nr in (0, h - 1) or nc in (0, w - 1):
                        touches_border = True
                    q.append((nr, nc))
            components.append(
                {
                    "coords": coords,
                    "area": len(coords),
                    "touches_border": touches_border,
                }
            )
    return components


def _legend_candidates(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    candidates = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            center = int(grid[r, c])
            if center == bg:
                continue
            patch = grid[r - 1 : r + 2, c - 1 : c + 2]
            surround = np.delete(patch.reshape(-1), 4)
            if np.any(surround == center):
                continue
            other = sorted({int(v) for v in surround if int(v) != bg})
            if len(other) != 1:
                continue
            wall = other[0]
            open_dirs = (
                int(patch[0, 1]) == bg,
                int(patch[1, 2]) == bg,
                int(patch[2, 1]) == bg,
                int(patch[1, 0]) == bg,
            )
            category = _classify_open_dirs(open_dirs)
            if category not in CATEGORY_ORDER:
                continue
            candidates.append(
                {
                    "r": r,
                    "c": c,
                    "color": center,
                    "wall": wall,
                    "category": category,
                }
            )
    return candidates


def _extract_legend_mapping(grid: np.ndarray, bg: int) -> tuple[dict[str, int], int]:
    h, w = grid.shape
    groups = defaultdict(list)
    for cand in _legend_candidates(grid, bg):
        groups[int(cand["wall"])].append(cand)

    best = None
    best_key = None
    for wall, group in groups.items():
        by_category = {cat: [d for d in group if d["category"] == cat] for cat in CATEGORY_ORDER}
        if any(not by_category[cat] for cat in CATEGORY_ORDER):
            continue

        rows = [int(d["r"]) for d in group]
        cols = [int(d["c"]) for d in group]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        corner_costs = [
            r0 + c0,
            r0 + (w - 1 - c1),
            (h - 1 - r1) + c0,
            (h - 1 - r1) + (w - 1 - c1),
        ]
        corner_idx = int(np.argmin(corner_costs))

        def dist_to_corner(item: dict) -> int:
            rr, cc = int(item["r"]), int(item["c"])
            if corner_idx == 0:
                return rr + cc
            if corner_idx == 1:
                return rr + (w - 1 - cc)
            if corner_idx == 2:
                return (h - 1 - rr) + cc
            return (h - 1 - rr) + (w - 1 - cc)

        mapping = {}
        for category in CATEGORY_ORDER:
            pick = min(by_category[category], key=dist_to_corner)
            mapping[category] = int(pick["color"])

        key = (corner_costs[corner_idx], -len(group))
        if best is None or key < best_key:
            best = (mapping, int(wall))
            best_key = key

    if best is None:
        raise AssertionError("failed to extract legend mapping")
    return best


def _maze_mask(grid: np.ndarray, bg: int) -> np.ndarray:
    enclosed = [comp for comp in _value_components(grid, bg) if not comp["touches_border"]]
    if not enclosed:
        raise AssertionError("no enclosed background component found")
    maze = max(enclosed, key=lambda comp: int(comp["area"]))
    mask = np.zeros(grid.shape, dtype=bool)
    for r, c in maze["coords"]:
        mask[r, c] = True
    return mask


def _mask_category(mask: np.ndarray, r: int, c: int) -> str | None:
    h, w = mask.shape
    open_dirs = []
    for dr, dc in ORTH_DIRS:
        nr, nc = r + dr, c + dc
        open_dirs.append(0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]))
    return _classify_open_dirs(tuple(open_dirs))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color a maze's path pixels from a corner legend that encodes local path topology.

    Input structure:
    - A dominant background surrounds one grey-walled maze and one corner legend.
    - The maze interior is a single-pixel-wide 4-connected path network of background cells.
    - The legend contains four 3x3 diagrams whose colored centers define the colors for
      straight, corner, tee, and cross local path configurations.

    Rule:
    - Read the legend by classifying each 3x3 diagram from its open orthogonal neighbors.
    - Find the maze's enclosed background component.
    - For each maze path pixel, classify its open orthogonal neighbors:
      opposite pair -> straight, adjacent pair -> corner, three -> tee, four -> cross.
    - Leave endpoints unchanged and recolor every other maze path pixel with the legend color
      for its topology class.
    - Keep the maze walls and the legend unchanged.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    mapping, _wall = _extract_legend_mapping(g, bg)
    maze = _maze_mask(g, bg)

    out = g.copy()
    rows, cols = np.where(maze)
    for r, c in zip(rows.tolist(), cols.tolist()):
        category = _mask_category(maze, int(r), int(c))
        if category in mapping:
            out[r, c] = int(mapping[category])
    return out


# --------------- generator ---------------

GRID_H_RANGE = (24, 35)
GRID_W_RANGE = (24, 35)
LEGEND_SIZE = 7
LEGEND_OFFSETS = [(0, 0), (0, 4), (4, 0), (4, 4)]
LEGEND_CORNERS = ("top_left", "top_right", "bottom_left", "bottom_right")
CELL_SIDE_RANGE = (4, 8)
PLACEMENT_PADDING = 2
GEN_RETRIES = 300

OPENINGS_BY_CATEGORY = {
    "straight": [(0, 2), (1, 3)],
    "corner": [(0, 1), (1, 2), (2, 3), (3, 0)],
    "tee": [(0, 1, 2), (1, 2, 3), (2, 3, 0), (3, 0, 1)],
    "cross": [(0, 1, 2, 3)],
}

BASE_NODES = {
    "cross": (1, 1),
    "tee": (1, 2),
    "corner": (2, 1),
    "endpoint": (0, 1),
    "leaf_a": (1, 0),
    "leaf_b": (1, 3),
    "junction_b": (2, 2),
    "leaf_c": (3, 2),
}

BASE_EDGES = [
    ("cross", "endpoint"),
    ("cross", "corner"),
    ("cross", "leaf_a"),
    ("cross", "tee"),
    ("tee", "leaf_b"),
    ("tee", "junction_b"),
    ("corner", "junction_b"),
    ("junction_b", "leaf_c"),
]


def _norm_edge(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    return (a, b) if a <= b else (b, a)


def _transform_square_point(
    point: tuple[int, int], size: int, rot: int, flip_lr: bool
) -> tuple[int, int]:
    r, c = point
    if flip_lr:
        c = size - 1 - c
    for _ in range(rot % 4):
        r, c = c, size - 1 - r
    return r, c


def _coarse_neighbors(cell: tuple[int, int], cell_h: int, cell_w: int) -> list[tuple[int, int]]:
    r, c = cell
    out = []
    for dr, dc in ORTH_DIRS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < cell_h and 0 <= nc < cell_w:
            out.append((nr, nc))
    return out


def _build_coarse_graph(cell_h: int, cell_w: int) -> tuple[set, set]:
    rot = int(_RNG.integers(0, 4))
    flip_lr = bool(int(_RNG.integers(0, 2)))
    row_off = int(_RNG.integers(0, cell_h - 3))
    col_off = int(_RNG.integers(0, cell_w - 3))

    placed = {}
    for label, point in BASE_NODES.items():
        rr, cc = _transform_square_point(point, 4, rot, flip_lr)
        placed[label] = (rr + row_off, cc + col_off)

    active = set(placed.values())
    edges = {_norm_edge(placed[a], placed[b]) for a, b in BASE_EDGES}
    protected = {
        placed["cross"],
        placed["tee"],
        placed["corner"],
        placed["endpoint"],
    }

    max_new = max(0, cell_h * cell_w - len(active))
    grow_steps = int(_RNG.integers(0, min(7, max_new) + 1))
    for _ in range(grow_steps):
        frontier = []
        for cell in active:
            if cell in protected:
                continue
            for nb in _coarse_neighbors(cell, cell_h, cell_w):
                if nb not in active:
                    frontier.append((cell, nb))
        if not frontier:
            break
        parent, child = frontier[int(_RNG.integers(len(frontier)))]
        active.add(child)
        edges.add(_norm_edge(parent, child))

    extra_edges = int(_RNG.integers(0, max(2, len(active) // 4 + 1)))
    for _ in range(extra_edges):
        candidates = []
        for cell in active:
            if cell in protected:
                continue
            for nb in _coarse_neighbors(cell, cell_h, cell_w):
                if nb not in active or nb in protected:
                    continue
                edge = _norm_edge(cell, nb)
                if edge not in edges:
                    candidates.append(edge)
        if not candidates:
            break
        edge = candidates[int(_RNG.integers(len(candidates)))]
        edges.add(edge)

    return active, edges


def _random_dihedral(arr: np.ndarray) -> np.ndarray:
    out = arr
    if bool(int(_RNG.integers(0, 2))):
        out = np.fliplr(out)
    out = np.rot90(out, int(_RNG.integers(0, 4)))
    return out.copy()


def _maze_patch(bg: int, wall: int) -> np.ndarray:
    cell_h = int(_RNG.integers(*CELL_SIDE_RANGE))
    cell_w = int(_RNG.integers(*CELL_SIDE_RANGE))
    active, edges = _build_coarse_graph(cell_h, cell_w)

    patch = np.full((2 * cell_h + 1, 2 * cell_w + 1), wall, dtype=int)
    for r, c in active:
        patch[2 * r + 1, 2 * c + 1] = bg
    for (r0, c0), (r1, c1) in edges:
        pr0, pc0 = 2 * r0 + 1, 2 * c0 + 1
        pr1, pc1 = 2 * r1 + 1, 2 * c1 + 1
        patch[(pr0 + pr1) // 2, (pc0 + pc1) // 2] = bg
    return _random_dihedral(patch)


def _legend_tile(category: str, color: int, bg: int, wall: int) -> np.ndarray:
    tile = np.full((3, 3), wall, dtype=int)
    tile[1, 1] = int(color)
    options = OPENINGS_BY_CATEGORY[category]
    openings = options[int(_RNG.integers(len(options)))]
    for idx in openings:
        dr, dc = ORTH_DIRS[idx]
        tile[1 + dr, 1 + dc] = bg
    return tile


def _legend_patch(mapping: dict[str, int], bg: int, wall: int) -> np.ndarray:
    patch = np.full((LEGEND_SIZE, LEGEND_SIZE), wall, dtype=int)
    positions = LEGEND_OFFSETS.copy()
    _RNG.shuffle(positions)
    for (r0, c0), category in zip(positions, CATEGORY_ORDER):
        patch[r0 : r0 + 3, c0 : c0 + 3] = _legend_tile(category, mapping[category], bg, wall)
    return patch


def _legend_rect(corner: str, h: int, w: int) -> tuple[int, int, int, int]:
    if corner == "top_left":
        return 0, 0, LEGEND_SIZE, LEGEND_SIZE
    if corner == "top_right":
        return 0, w - LEGEND_SIZE, LEGEND_SIZE, w
    if corner == "bottom_left":
        return h - LEGEND_SIZE, 0, h, LEGEND_SIZE
    return h - LEGEND_SIZE, w - LEGEND_SIZE, h, w


def _rects_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int) -> bool:
    return not (
        a[2] + pad <= b[0]
        or b[2] + pad <= a[0]
        or a[3] + pad <= b[1]
        or b[3] + pad <= a[1]
    )


def _sample_maze_position(
    h: int, w: int, maze_shape: tuple[int, int], legend_rect: tuple[int, int, int, int]
) -> tuple[int, int] | None:
    mh, mw = maze_shape
    if h - mh <= 1 or w - mw <= 1:
        return None
    for _ in range(200):
        r0 = int(_RNG.integers(1, h - mh))
        c0 = int(_RNG.integers(1, w - mw))
        maze_rect = (r0, c0, r0 + mh, c0 + mw)
        if _rects_too_close(maze_rect, legend_rect, PLACEMENT_PADDING):
            continue
        return r0, c0
    return None


def _validate_generated_grid(grid: np.ndarray) -> bool:
    bg = _background_color(grid)
    try:
        mapping, _wall = _extract_legend_mapping(grid, bg)
        maze = _maze_mask(grid, bg)
    except AssertionError:
        return False

    counts = Counter()
    rows, cols = np.where(maze)
    for r, c in zip(rows.tolist(), cols.tolist()):
        category = _mask_category(maze, int(r), int(c))
        if category is None:
            return False
        counts[category] += 1
    for category in ("endpoint",) + CATEGORY_ORDER:
        if counts[category] == 0:
            return False

    out = main(grid)
    if out.shape != grid.shape:
        return False
    if not np.any(out != grid):
        return False

    stable = ~maze
    if np.any(out[stable] != grid[stable]):
        return False

    for r, c in zip(rows.tolist(), cols.tolist()):
        category = _mask_category(maze, int(r), int(c))
        expected = bg if category == "endpoint" else int(mapping[category])
        if int(out[r, c]) != expected:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = int(_RNG.integers(*GRID_H_RANGE))
    w = int(_RNG.integers(*GRID_W_RANGE))

    colors = [int(v) for v in _RNG.choice(10, size=6, replace=False)]
    bg, wall = colors[:2]
    mapping = {category: color for category, color in zip(CATEGORY_ORDER, colors[2:])}

    legend = _legend_patch(mapping, bg, wall)
    maze = _maze_patch(bg, wall)
    corner = LEGEND_CORNERS[int(_RNG.integers(len(LEGEND_CORNERS)))]
    legend_rect = _legend_rect(corner, h, w)
    maze_pos = _sample_maze_position(h, w, maze.shape, legend_rect)
    if maze_pos is None:
        return None

    grid = np.full((h, w), bg, dtype=int)
    lr0, lc0, lr1, lc1 = legend_rect
    grid[lr0:lr1, lc0:lc1] = legend

    mr0, mc0 = maze_pos
    mh, mw = maze.shape
    grid[mr0 : mr0 + mh, mc0 : mc0 + mw] = maze

    if not _validate_generated_grid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
