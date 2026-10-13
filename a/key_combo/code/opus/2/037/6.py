from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 5
MARKER = 2
WALL = 6
RECT_COLORS = [1, 3, 4, 7, 8, 9]
CORNERS = ("TL", "TR", "BL", "BR")
POCKET_SIZES = [(3, 4), (4, 3)]  # (height, width); wall length is always 8.
RECT_SHAPES = [(2, 2), (2, 3), (3, 2), (3, 3), (2, 4)]
SELECTED_SHAPES = [(2, 2), (3, 2)]
GEN_HEIGHT_RANGE = (14, 19)
GEN_WIDTH_RANGE = (14, 19)
GEN_ATTEMPTS = 500


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = grid == int(color)
    H, W = grid.shape
    seen = np.zeros((H, W), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _find_rectangles(grid: np.ndarray) -> list[dict]:
    bg = get_bg(grid)
    rects: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in {bg, MARKER, WALL}):
        comps = _components_of_color(grid, color)
        for coords in comps:
            r0, c0, r1, c1 = bbox_coords(coords)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            assert int(coords.shape[0]) == h * w, "foreground objects must be solid rectangles"
            assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color), "rectangle crop must be monochrome"
            rects.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "area": h * w,
                    "width": w,
                    "height": h,
                }
            )
    assert 3 <= len(rects) <= 4, "expected 3 or 4 rectangles"
    return rects


def _reachable_marker_count(grid: np.ndarray, rect: dict) -> int:
    H, W = grid.shape
    seen = np.zeros((H, W), dtype=bool)
    q = deque()
    for r, c in rect["coords"]:
        rr = int(r)
        cc = int(c)
        seen[rr, cc] = True
        q.append((rr, cc))

    markers: set[tuple[int, int]] = set()
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if not (0 <= nr < H and 0 <= nc < W) or seen[nr, nc]:
                continue
            val = int(grid[nr, nc])
            if val in {BG, MARKER}:
                seen[nr, nc] = True
                q.append((nr, nc))
                if val == MARKER:
                    markers.add((nr, nc))
    return len(markers)


def _choose_target(rects: list[dict], counts: list[int]) -> dict:
    scored = list(zip(rects, counts, strict=True))
    scored.sort(key=lambda item: (item[1], item[0]["area"], item[0]["bbox"][0], item[0]["bbox"][1]))
    return scored[0][0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Filter marker counts by wall-constrained reachability, keep the rectangle with the fewest reachable markers, and shift it right by its own width.

    Input structure:
    - The background is grey, magenta pixels form thin blocking wall segments, and red singleton pixels act as markers.
    - The remaining foreground colors each form one solid monochrome rectangle, with 3 or 4 rectangles total.
    - Walls and the grid boundary restrict which background and marker cells each rectangle can reach by 4-connected movement.

    Rule:
    - For each rectangle, start from all of its pixels and flood through only grey background cells and red marker cells.
    - Count how many red markers are reachable for each rectangle, treating walls and every other rectangle as impassable.
    - Select the rectangle with the smallest reachable-marker count; if counts tie, prefer the smaller bounding-box area.
    - Erase every marker and every non-selected rectangle, keep the walls unchanged, and move the selected rectangle right by its own width.

    Key insight:
    - The deciding signal is not the total number of nearby markers but the number that remain reachable once the wall topology blocks off parts of the grid.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"

    bg = get_bg(g)
    rects = _find_rectangles(g)
    counts = [_reachable_marker_count(g, rect) for rect in rects]
    target = _choose_target(rects, counts)

    out = np.full_like(g, bg)
    out[g == WALL] = WALL
    shift = int(target["width"])
    color = int(target["color"])
    for r, c in target["coords"]:
        rr = int(r)
        cc = int(c) + shift
        assert 0 <= cc < g.shape[1], "translated rectangle must stay in bounds"
        assert out[rr, cc] == bg, "translated rectangle must not overlap a wall"
        out[rr, cc] = color
    return out


def _pocket_geometry(corner: str, pocket_h: int, pocket_w: int, H: int, W: int) -> dict:
    if corner == "TL":
        interior = (0, 0, pocket_h - 1, pocket_w - 1)
        wall = {(r, pocket_w) for r in range(pocket_h + 1)}
        wall |= {(pocket_h, c) for c in range(pocket_w + 1)}
        near = [(r, pocket_w + 1) for r in range(pocket_h) if pocket_w + 1 < W]
        near += [(pocket_h + 1, c) for c in range(pocket_w) if pocket_h + 1 < H]
    elif corner == "TR":
        c0 = W - pocket_w
        wall_c = c0 - 1
        interior = (0, c0, pocket_h - 1, W - 1)
        wall = {(r, wall_c) for r in range(pocket_h + 1)}
        wall |= {(pocket_h, c) for c in range(wall_c, W)}
        near = [(r, wall_c - 1) for r in range(pocket_h) if wall_c - 1 >= 0]
        near += [(pocket_h + 1, c) for c in range(c0, W) if pocket_h + 1 < H]
    elif corner == "BL":
        r0 = H - pocket_h
        wall_r = r0 - 1
        interior = (r0, 0, H - 1, pocket_w - 1)
        wall = {(r, pocket_w) for r in range(wall_r, H)}
        wall |= {(wall_r, c) for c in range(pocket_w + 1)}
        near = [(r, pocket_w + 1) for r in range(r0, H) if pocket_w + 1 < W]
        near += [(wall_r - 1, c) for c in range(pocket_w) if wall_r - 1 >= 0]
    elif corner == "BR":
        r0 = H - pocket_h
        c0 = W - pocket_w
        wall_r = r0 - 1
        wall_c = c0 - 1
        interior = (r0, c0, H - 1, W - 1)
        wall = {(r, wall_c) for r in range(wall_r, H)}
        wall |= {(wall_r, c) for c in range(wall_c, W)}
        near = [(r, wall_c - 1) for r in range(r0, H) if wall_c - 1 >= 0]
        near += [(wall_r - 1, c) for c in range(c0, W) if wall_r - 1 >= 0]
    else:
        raise ValueError(f"unknown corner {corner}")

    return {
        "corner": corner,
        "pocket_h": pocket_h,
        "pocket_w": pocket_w,
        "interior": interior,
        "wall": wall,
        "near": near,
    }


def _cells_in_bbox(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _rect_candidates(geom: dict, count: int, selected: bool) -> list[tuple[int, int, int, int]]:
    r0, c0, r1, c1 = geom["interior"]
    pocket_area = geom["pocket_h"] * geom["pocket_w"]
    shapes = SELECTED_SHAPES if selected else RECT_SHAPES
    candidates: list[tuple[int, int, int, int]] = []
    for h, w in shapes:
        if r0 + h - 1 > r1 or c0 + w - 1 > c1:
            continue
        if pocket_area - h * w < count:
            continue
        for rr in range(r0, r1 - h + 2):
            for cc in range(c0, c1 - w + 2):
                if selected and cc + 2 * w - 1 > c1:
                    continue
                candidates.append((rr, cc, h, w))
    return candidates


def _paint_rect(grid: np.ndarray, top: int, left: int, h: int, w: int, color: int) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(top, top + h) for c in range(left, left + w)]
    for r, c in cells:
        grid[r, c] = color
    return cells


def _min_rect_distance(rect_cells: list[tuple[int, int]], cell: tuple[int, int]) -> int:
    return min(abs(r - cell[0]) + abs(c - cell[1]) for r, c in rect_cells)


def _count_combos(n_rect: int) -> list[tuple[int, ...]]:
    values = [0, 1, 2, 3, 4]
    combos: list[tuple[int, ...]] = []
    stack: list[int] = []

    def rec(start: int):
        if len(stack) == n_rect:
            total = sum(stack)
            if 5 <= total <= 9:
                combos.append(tuple(stack))
            return
        for idx in range(start, len(values)):
            stack.append(values[idx])
            rec(idx + 1)
            stack.pop()

    rec(0)
    return combos


COUNT_COMBOS = {
    3: _count_combos(3),
    4: _count_combos(4),
}


def _extra_marker_candidates(grid: np.ndarray, geoms: dict[str, dict]) -> list[tuple[int, int]]:
    blocked = np.zeros_like(grid, dtype=bool)
    for geom in geoms.values():
        for r, c in geom["wall"]:
            blocked[r, c] = True
        r0, c0, r1, c1 = geom["interior"]
        blocked[r0 : r1 + 1, c0 : c1 + 1] = True

    cells: list[tuple[int, int]] = []
    H, W = grid.shape
    for r in range(H):
        for c in range(W):
            if not blocked[r, c] and int(grid[r, c]) == BG:
                cells.append((r, c))
    return cells


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GEN_HEIGHT_RANGE)
    W = random.randint(*GEN_WIDTH_RANGE)
    n_rect = random.choice([3, 4])
    used_corners = random.sample(CORNERS, n_rect)
    selected_corner = random.choice(used_corners)

    geoms: dict[str, dict] = {}
    for corner in used_corners:
        choices = POCKET_SIZES[:]
        if corner == selected_corner:
            choices = [(3, 4)]
        pocket_h, pocket_w = random.choice(choices)
        geoms[corner] = _pocket_geometry(corner, pocket_h, pocket_w, H, W)

    counts = list(random.choice(COUNT_COMBOS[n_rect]))
    counts.sort()
    per_corner_count: dict[str, int] = {selected_corner: counts[0]}
    others = [corner for corner in used_corners if corner != selected_corner]
    other_counts = counts[1:]
    random.shuffle(others)
    random.shuffle(other_counts)
    for corner, count in zip(others, other_counts, strict=True):
        per_corner_count[corner] = count

    total_reachable = sum(per_corner_count.values())
    min_unreachable = max(1, 6 - total_reachable)
    max_unreachable = min(3, 10 - total_reachable)
    if min_unreachable > max_unreachable:
        return None
    unreachable_total = random.randint(min_unreachable, max_unreachable)

    grid = np.full((H, W), BG, dtype=int)
    for geom in geoms.values():
        for r, c in geom["wall"]:
            grid[r, c] = WALL

    colors = random.sample(RECT_COLORS, n_rect)
    per_corner_color = dict(zip(used_corners, colors, strict=True))
    selected_color = per_corner_color[selected_corner]
    rect_cells_by_corner: dict[str, list[tuple[int, int]]] = {}

    for corner in used_corners:
        geom = geoms[corner]
        count = per_corner_count[corner]
        candidates = _rect_candidates(geom, count, selected=corner == selected_corner)
        if not candidates:
            return None
        top, left, h, w = random.choice(candidates)
        rect_cells = _paint_rect(grid, top, left, h, w, per_corner_color[corner])
        rect_cells_by_corner[corner] = rect_cells

        interior_cells = _cells_in_bbox(geom["interior"])
        free_cells = [cell for cell in interior_cells if cell not in set(rect_cells)]
        if len(free_cells) < count:
            return None
        for r, c in random.sample(free_cells, count):
            grid[r, c] = MARKER

    near_candidates = [
        cell
        for cell in geoms[selected_corner]["near"]
        if 0 <= cell[0] < H and 0 <= cell[1] < W and int(grid[cell]) == BG
    ]
    if not near_candidates:
        return None
    near_candidates.sort(key=lambda cell: _min_rect_distance(rect_cells_by_corner[selected_corner], cell))

    chosen_extra: list[tuple[int, int]] = [near_candidates[0]]
    remaining_need = unreachable_total - 1
    extra_pool = [cell for cell in _extra_marker_candidates(grid, geoms) if cell not in chosen_extra]
    random.shuffle(extra_pool)
    for cell in near_candidates[1:]:
        if remaining_need == 0:
            break
        if cell not in chosen_extra:
            chosen_extra.append(cell)
            remaining_need -= 1
    while remaining_need > 0 and extra_pool:
        chosen_extra.append(extra_pool.pop())
        remaining_need -= 1
    if remaining_need > 0:
        return None
    for r, c in chosen_extra:
        grid[r, c] = MARKER

    total_markers = int(np.count_nonzero(grid == MARKER))
    if not (6 <= total_markers <= 10):
        return None

    try:
        rects = _find_rectangles(grid)
        scores = {rect["color"]: _reachable_marker_count(grid, rect) for rect in rects}
        if len(set(scores.values())) != len(scores):
            return None
        target = _choose_target(rects, [scores[rect["color"]] for rect in rects])
        if int(target["color"]) != int(selected_color):
            return None
        shift = int(target["width"])
        for r, c in target["coords"]:
            nr = int(r)
            nc = int(c) + shift
            if not (0 <= nc < W) or int(grid[nr, nc]) == WALL:
                return None
        if _min_rect_distance(rect_cells_by_corner[selected_corner], chosen_extra[0]) > 3:
            return None
        out = main(grid)
        if out.shape != grid.shape:
            return None
        if np.count_nonzero(out == MARKER) != 0:
            return None
    except Exception:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid reachability-filter input")
