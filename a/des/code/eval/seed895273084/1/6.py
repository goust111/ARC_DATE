import random
from collections import Counter, deque

import numpy as np


BG = 0
WALL_COLOR = 5
MARK_COLOR = 9
OBJECT_PALETTE = (1, 2, 3, 4, 6, 7, 8)


GEN_ATTEMPTS = 400
HEIGHT_RANGE = (10, 16)
LEFT_WIDTH_RANGE = (9, 13)
BAND_COUNT_RANGE = (2, 3)
OBJECT_COUNT_RANGE = (3, 4)
OBJECT_BBOX_H_RANGE = (2, 5)
OBJECT_BBOX_W_RANGE = (2, 5)
OBJECT_CELL_RANGE = (4, 10)
PLACEMENT_MARGIN = 1
POLYOMINO_ATTEMPTS = 80


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    comps.sort(key=lambda comp: min(comp))
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _find_corridor(grid: np.ndarray, bg: int) -> tuple[int, int, int]:
    h, w = grid.shape
    candidates: list[tuple[int, int, int]] = []
    for c0 in range(w - 3):
        c1 = c0 + 3
        wall_value = int(grid[0, c0])
        if wall_value == bg:
            continue
        if not np.all(grid[:, c0] == wall_value):
            continue
        if not np.all(grid[:, c1] == wall_value):
            continue
        if w - (c1 + 1) != c0:
            continue
        candidates.append((c0, c1, wall_value))
    assert len(candidates) == 1, "expected a unique symmetric corridor"
    left_wall, right_wall, wall_color = candidates[0]
    assert wall_color == WALL_COLOR
    assert grid.shape[1] == 2 * left_wall + 4
    assert right_wall == left_wall + 3
    return left_wall, right_wall, wall_color


def _collect_filter_colors(
    grid: np.ndarray,
    left_wall: int,
    wall_color: int,
    bg: int,
) -> set[int]:
    colors: set[int] = set()
    for r in range(grid.shape[0]):
        left_band = int(grid[r, left_wall + 1])
        right_band = int(grid[r, left_wall + 2])
        if left_band == bg and right_band == bg:
            continue
        assert left_band == right_band, "corridor bands must span both corridor cells"
        assert left_band not in {bg, wall_color}, "band rows must use distinct filter colors"
        colors.add(left_band)
    assert colors, "expected at least one corridor filter color"
    return colors


def _mirror_col(left_wall: int, col: int) -> int:
    return 2 * left_wall + 3 - col


def _scene_info(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    left_wall, right_wall, wall_color = _find_corridor(g, bg)
    filter_colors = _collect_filter_colors(g, left_wall, wall_color, bg)
    left_mask = g[:, :left_wall] != bg
    components = _components(left_mask)
    assert components, "expected at least one source component on the left"
    return {
        "grid": g,
        "bg": bg,
        "left_wall": left_wall,
        "right_wall": right_wall,
        "wall_color": wall_color,
        "filter_colors": filter_colors,
        "components": components,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each left-side component across the corridor, then remove mirrored
    pixels whose colors appear in the corridor bands. If a component loses all
    mirrored pixels, mark the center of its mirrored bbox with maroon.
    """
    info = _scene_info(grid)
    g = info["grid"]
    out = g.copy()
    left_wall = int(info["left_wall"])
    filter_colors: set[int] = info["filter_colors"]

    for comp in info["components"]:
        kept: list[tuple[int, int, int]] = []
        for r, c in comp:
            color = int(g[r, c])
            if color in filter_colors:
                continue
            kept.append((r, _mirror_col(left_wall, c), color))

        if kept:
            for r, c, color in kept:
                out[r, c] = color
            continue

        r0, c0, r1, c1 = _bbox(comp)
        mc0 = _mirror_col(left_wall, c1)
        mc1 = _mirror_col(left_wall, c0)
        center_r = (r0 + r1 + 1) // 2
        center_c = (mc0 + mc1 + 1) // 2
        out[center_r, center_c] = MARK_COLOR

    return out


def _sample_rows(height: int, count: int) -> list[int] | None:
    for _ in range(80):
        rows = sorted(random.sample(range(height), count))
        if all(b - a >= 2 for a, b in zip(rows, rows[1:])):
            return rows
    return None


def _random_connected_cells(height: int, width: int, target: int) -> list[tuple[int, int]] | None:
    start = (random.randrange(height), random.randrange(width))
    cells = {start}
    frontier = set()
    for nr, nc in _neighbors4(*start):
        if 0 <= nr < height and 0 <= nc < width:
            frontier.add((nr, nc))

    while len(cells) < target and frontier:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        for nr, nc in _neighbors4(*cell):
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                frontier.add((nr, nc))

    if len(cells) != target:
        return None

    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    if len(set(rows)) < 2 or len(set(cols)) < 2:
        return None

    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    cropped = sorted((r - r0, c - c0) for r, c in cells)
    return cropped


def _choose_object_colors(branch: str, filter_colors: list[int], other_colors: list[int]) -> list[int]:
    if branch == "full":
        n_colors = random.randint(2, min(4, len(filter_colors)))
        return random.sample(filter_colors, n_colors)
    if branch == "none":
        n_colors = random.randint(2, min(4, len(other_colors)))
        return random.sample(other_colors, n_colors)

    n_filtered = random.randint(1, min(2, len(filter_colors)))
    n_other = random.randint(1, min(2, len(other_colors)))
    colors = random.sample(filter_colors, n_filtered) + random.sample(other_colors, n_other)
    random.shuffle(colors)
    return colors[: min(4, len(colors))]


def _make_object(branch: str, filter_colors: list[int], other_colors: list[int]) -> np.ndarray | None:
    for _ in range(POLYOMINO_ATTEMPTS):
        box_h = random.randint(*OBJECT_BBOX_H_RANGE)
        box_w = random.randint(*OBJECT_BBOX_W_RANGE)
        colors = _choose_object_colors(branch, filter_colors, other_colors)
        max_cells = min(box_h * box_w, OBJECT_CELL_RANGE[1])
        min_cells = max(OBJECT_CELL_RANGE[0], len(colors))
        if min_cells > max_cells:
            continue
        target = random.randint(min_cells, max_cells)
        cells = _random_connected_cells(box_h, box_w, target)
        if cells is None:
            continue

        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        height = max(rows) + 1
        width = max(cols) + 1
        arr = np.full((height, width), BG, dtype=int)

        shuffled_cells = cells[:]
        random.shuffle(shuffled_cells)
        for idx, color in enumerate(colors):
            r, c = shuffled_cells[idx]
            arr[r, c] = color
        for r, c in shuffled_cells[len(colors) :]:
            arr[r, c] = random.choice(colors)

        used_colors = {int(arr[r, c]) for r, c in shuffled_cells}
        if not (2 <= len(used_colors) <= 4):
            continue
        if branch == "full" and not used_colors.issubset(filter_colors):
            continue
        if branch == "none" and not used_colors.isdisjoint(filter_colors):
            continue
        if branch == "partial":
            if used_colors.isdisjoint(filter_colors) or used_colors.issubset(filter_colors):
                continue
        return arr
    return None


def _placement_candidates(blocked: np.ndarray, obj_h: int, obj_w: int) -> list[tuple[int, int]]:
    h, left_w = blocked.shape
    candidates: list[tuple[int, int]] = []
    for top in range(h - obj_h + 1):
        for left in range(left_w - obj_w + 1):
            r0 = max(0, top - PLACEMENT_MARGIN)
            r1 = min(h, top + obj_h + PLACEMENT_MARGIN)
            c0 = max(0, left - PLACEMENT_MARGIN)
            c1 = min(left_w, left + obj_w + PLACEMENT_MARGIN)
            if not blocked[r0:r1, c0:c1].any():
                candidates.append((top, left))
    return candidates


def _place_object(grid: np.ndarray, blocked: np.ndarray, obj: np.ndarray, left_w: int) -> bool:
    obj_h, obj_w = obj.shape
    candidates = _placement_candidates(blocked, obj_h, obj_w)
    if not candidates:
        return False
    top, left = random.choice(candidates)
    mask = obj != BG
    patch = grid[top : top + obj_h, left : left + obj_w]
    patch[mask] = obj[mask]

    r0 = max(0, top - PLACEMENT_MARGIN)
    r1 = min(blocked.shape[0], top + obj_h + PLACEMENT_MARGIN)
    c0 = max(0, left - PLACEMENT_MARGIN)
    c1 = min(left_w, left + obj_w + PLACEMENT_MARGIN)
    blocked[r0:r1, c0:c1] = True
    return True


def _component_branch(grid: np.ndarray, comp: list[tuple[int, int]], filter_colors: set[int]) -> str:
    colors = {int(grid[r, c]) for r, c in comp}
    if colors.issubset(filter_colors):
        return "full"
    if colors.isdisjoint(filter_colors):
        return "none"
    return "partial"


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*HEIGHT_RANGE)
    left_w = random.randint(*LEFT_WIDTH_RANGE)
    width = 2 * left_w + 4
    grid = np.full((height, width), BG, dtype=int)
    grid[:, left_w] = WALL_COLOR
    grid[:, left_w + 3] = WALL_COLOR

    band_count = random.randint(*BAND_COUNT_RANGE)
    band_rows = _sample_rows(height, band_count)
    if band_rows is None:
        return None
    filter_colors = random.sample(list(OBJECT_PALETTE), band_count)
    for row, color in zip(band_rows, filter_colors):
        grid[row, left_w + 1 : left_w + 3] = color

    other_colors = [color for color in OBJECT_PALETTE if color not in filter_colors]
    n_objects = random.randint(*OBJECT_COUNT_RANGE)
    branches = ["full", "partial", "none"]
    while len(branches) < n_objects:
        branches.append(random.choice(branches))
    random.shuffle(branches)

    blocked = np.zeros((height, left_w), dtype=bool)
    for branch in branches:
        obj = _make_object(branch, filter_colors, other_colors)
        if obj is None:
            return None
        if not _place_object(grid, blocked, obj, left_w):
            return None

    info = _scene_info(grid)
    if info["bg"] != BG:
        return None
    if info["left_wall"] != left_w or info["right_wall"] != left_w + 3:
        return None
    if info["filter_colors"] != set(filter_colors):
        return None
    if np.any(grid[:, info["right_wall"] + 1 :] != BG):
        return None

    components = info["components"]
    if len(components) != n_objects:
        return None
    branch_set = {_component_branch(grid, comp, info["filter_colors"]) for comp in components}
    if branch_set != {"full", "partial", "none"}:
        return None
    for comp in components:
        colors = {int(grid[r, c]) for r, c in comp}
        if not (2 <= len(colors) <= 4):
            return None

    out = main(grid)
    right = out[:, info["right_wall"] + 1 :]
    if not np.any(right == MARK_COLOR):
        return None
    if not np.any((right != BG) & (right != MARK_COLOR)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
