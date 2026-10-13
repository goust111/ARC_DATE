import random
from collections import Counter, deque

import numpy as np


BG = 0
COLOR_POOL = tuple(range(1, 10))
GEN_RETRIES = 400
EDGES = ("top", "bottom", "left", "right")
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
NEIGHBORS8 = (
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
)


def _bg_color(grid: np.ndarray) -> int:
    return int(Counter(np.asarray(grid, dtype=int).ravel().tolist()).most_common(1)[0][0])


def _connected_components(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    dirs = NEIGHBORS4 if connectivity == 4 else NEIGHBORS8
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        comp = [(r, c)]
        q = deque([(r, c)])
        while q:
            cr, cc = q.popleft()
            for dr, dc in dirs:
                nr = cr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
                    comp.append((nr, nc))
        comps.append(comp)
    return comps


def _edge_line_positions(h: int, w: int, edge: str) -> list[tuple[int, int]]:
    if edge == "top":
        return [(0, c) for c in range(w)]
    if edge == "bottom":
        return [(h - 1, c) for c in range(w)]
    if edge == "left":
        return [(r, 0) for r in range(h)]
    return [(r, w - 1) for r in range(h)]


def _legend_info(grid: np.ndarray, bg: int) -> tuple[list[int], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    best = None
    for edge in EDGES:
        line = _edge_line_positions(h, w, edge)
        i = 0
        while i < len(line):
            r, c = line[i]
            if int(g[r, c]) == bg:
                i += 1
                continue
            j = i
            cells: list[tuple[int, int]] = []
            colors: list[int] = []
            while j < len(line):
                rr, cc = line[j]
                v = int(g[rr, cc])
                if v == bg:
                    break
                cells.append((rr, cc))
                colors.append(v)
                j += 1
            if len(cells) >= 2:
                distinct = len(set(colors)) == len(colors)
                repeated = sum(int(np.count_nonzero(g == color) > 1) for color in colors)
                score = (int(distinct), repeated, len(cells))
                cand = (score, colors, cells)
                if best is None or cand[0] > best[0]:
                    best = cand
            i = j + 1
    if best is None:
        raise ValueError("failed to detect edge legend")
    return list(best[1]), list(best[2])


def _largest_contours_and_seeds(
    grid: np.ndarray, legend_colors: list[int], legend_positions: list[tuple[int, int]]
) -> tuple[dict[int, set[tuple[int, int]]], dict[int, set[tuple[int, int]]]]:
    g = np.asarray(grid, dtype=int)
    legend_by_color: dict[int, set[tuple[int, int]]] = {}
    for pos, color in zip(legend_positions, legend_colors):
        legend_by_color.setdefault(int(color), set()).add(tuple(pos))

    contours: dict[int, set[tuple[int, int]]] = {}
    seeds: dict[int, set[tuple[int, int]]] = {}
    for color in legend_colors:
        mask = g == int(color)
        for r, c in legend_by_color.get(int(color), set()):
            mask[r, c] = False
        comps = _connected_components(mask, connectivity=8)
        if not comps:
            raise ValueError("legend color has no contour component")
        contour = set(max(comps, key=len))
        contours[int(color)] = contour
        all_cells = {tuple(map(int, coord)) for coord in np.argwhere(mask)}
        seeds[int(color)] = all_cells - contour
    return contours, seeds


def _inside_of_wall(shape: tuple[int, int], wall_cells: set[tuple[int, int]]) -> np.ndarray:
    h, w = shape
    wall = np.zeros((h, w), dtype=bool)
    for r, c in wall_cells:
        wall[r, c] = True

    outside = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if not wall[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not wall[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in NEIGHBORS4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and not wall[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))
    return (~outside) & (~wall)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill contour-map rings selected by same-color seed pixels and remove the seeds.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)

    legend_colors, legend_positions = _legend_info(g, bg)
    contours, seeds = _largest_contours_and_seeds(g, legend_colors, legend_positions)

    out = g.copy()
    for cells in seeds.values():
        for r, c in cells:
            out[r, c] = bg

    inside_masks = [_inside_of_wall((h, w), contours[int(color)]) for color in legend_colors]
    empty = np.zeros((h, w), dtype=bool)
    for i, color in enumerate(legend_colors):
        if not seeds[int(color)]:
            continue
        inner_next = inside_masks[i + 1] if i + 1 < len(inside_masks) else empty
        fill_mask = inside_masks[i] & ~inner_next & (out == bg)
        out[fill_mask] = int(color)
    return out


def _draw_rect_border(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def _legend_positions_for(edge: str, start: int, length: int, h: int, w: int) -> list[tuple[int, int]]:
    if edge == "top":
        return [(0, start + i) for i in range(length)]
    if edge == "bottom":
        return [(h - 1, start + i) for i in range(length)]
    if edge == "left":
        return [(start + i, 0) for i in range(length)]
    return [(start + i, w - 1) for i in range(length)]


def _seed_candidates(
    grid: np.ndarray,
    terrain_bbox: tuple[int, int, int, int],
    legend_positions: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    h, w = grid.shape
    r0, c0, r1, c1 = terrain_bbox
    candidates: list[tuple[int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if (r, c) in legend_positions or int(grid[r, c]) != BG:
                continue
            if r0 - 1 <= r <= r1 + 1 and c0 - 1 <= c <= c1 + 1:
                continue
            if all(int(grid[r + dr, c + dc]) == BG for dr, dc in NEIGHBORS8):
                candidates.append((r, c))
    return candidates


def _place_seeds(
    grid: np.ndarray,
    colors: list[int],
    terrain_bbox: tuple[int, int, int, int],
    legend_positions: list[tuple[int, int]],
    rng: random.Random,
) -> bool:
    active = [color for color in colors if rng.random() < 0.5]
    if not active:
        active = [rng.choice(colors)]

    legend_set = set(legend_positions)
    used: set[tuple[int, int]] = set()
    for color in active:
        options = _seed_candidates(grid, terrain_bbox, legend_set)
        rng.shuffle(options)
        placed = False
        for r, c in options:
            if any(max(abs(r - rr), abs(c - cc)) <= 1 for rr, cc in used):
                continue
            grid[r, c] = int(color)
            used.add((r, c))
            placed = True
            break
        if not placed:
            return False
    return True


def _build_bboxes(
    h: int, w: int, n_layers: int, edge: str, rng: random.Random
) -> list[tuple[int, int, int, int]] | None:
    inner_h = rng.randint(5, 9)
    inner_w = rng.randint(5, 11)
    top_steps = [rng.randint(2, 4) for _ in range(n_layers - 1)]
    bottom_steps = [rng.randint(2, 4) for _ in range(n_layers - 1)]
    left_steps = [rng.randint(2, 4) for _ in range(n_layers - 1)]
    right_steps = [rng.randint(2, 4) for _ in range(n_layers - 1)]

    outer_h = inner_h + sum(top_steps) + sum(bottom_steps)
    outer_w = inner_w + sum(left_steps) + sum(right_steps)

    top_margin = 4 if edge == "top" else 2
    bottom_margin = 4 if edge == "bottom" else 2
    left_margin = 4 if edge == "left" else 2
    right_margin = 4 if edge == "right" else 2

    if outer_h > h - top_margin - bottom_margin:
        return None
    if outer_w > w - left_margin - right_margin:
        return None

    r0 = rng.randint(top_margin, h - bottom_margin - outer_h)
    c0 = rng.randint(left_margin, w - right_margin - outer_w)
    r1 = r0 + outer_h - 1
    c1 = c0 + outer_w - 1

    bboxes = [(r0, c0, r1, c1)]
    for i in range(n_layers - 1):
        r0 += top_steps[i]
        c0 += left_steps[i]
        r1 -= bottom_steps[i]
        c1 -= right_steps[i]
        if r1 - r0 + 1 < 5 or c1 - c0 + 1 < 5:
            return None
        bboxes.append((r0, c0, r1, c1))
    return bboxes


def _place_legend(grid: np.ndarray, colors: list[int], edge: str, rng: random.Random) -> list[tuple[int, int]] | None:
    h, w = grid.shape
    n = len(colors)
    if edge in ("top", "bottom"):
        if w < n + 2:
            return None
        start = rng.randint(1, w - n - 1)
    else:
        if h < n + 2:
            return None
        start = rng.randint(1, h - n - 1)
    positions = _legend_positions_for(edge, start, n, h, w)
    for (r, c), color in zip(positions, colors):
        grid[r, c] = int(color)
    return positions


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(18, 28)
    w = rng.randint(18, 30)
    n_layers = rng.randint(2, 4)
    colors = rng.sample(COLOR_POOL, n_layers)
    edge = rng.choice(EDGES)
    bboxes = _build_bboxes(h, w, n_layers, edge, rng)
    if bboxes is None:
        return None

    grid = np.full((h, w), BG, dtype=int)
    for color, bbox in zip(colors, bboxes):
        _draw_rect_border(grid, bbox, color)

    legend_positions = _place_legend(grid, colors, edge, rng)
    if legend_positions is None:
        return None

    if not _place_seeds(grid, colors, bboxes[0], legend_positions, rng):
        return None

    # Validate the generated instance matches the solver's structural assumptions.
    out = main(grid)
    if np.array_equal(out, grid):
        return None
    legend_colors, found_positions = _legend_info(grid, BG)
    if legend_colors != colors or found_positions != legend_positions:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted retries")
