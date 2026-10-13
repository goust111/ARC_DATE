from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
AZURE = 8
MAROON = 9
TOWER_COLORS = (1, 2, 3, 4, 6, 7)

DIRECTIONS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

TEMPLATE_POOL = (
    ((-1, 0), (0, -1), (0, 1), (1, 0)),
    ((-1, -1), (-1, 1), (1, -1), (1, 1)),
    ((-1, 0), (0, 1)),
    ((0, 1), (1, 0)),
    ((1, 0), (0, -1)),
    ((0, -1), (-1, 0)),
    ((-1, 0), (0, -1), (0, 1)),
    ((-1, 0), (0, 1), (1, 0)),
    ((0, -1), (0, 1), (1, 0)),
    ((-1, 0), (0, -1), (1, 0)),
)

GRID_H_RANGE = (22, 30)
GRID_W_RANGE = (28, 36)
LEGEND_W = 7
ENTRY_GAP = 1
ENTRY_H = 3
OBSTACLE_COUNT_RANGE = (2, 4)
OBSTACLE_H_RANGE = (2, 5)
OBSTACLE_W_RANGE = (2, 6)
EXTRA_TOWERS_RANGE = (0, 1)
GEN_RETRIES = 400
PLACEMENT_RETRIES = 500


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            comp: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                comp.append((r, c))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _legend_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    candidates: list[tuple[int, int, int, int]] = []
    for comp in _components(grid == MAROON):
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 5 or w < 7:
            continue
        if len(comp) != 2 * (h + w) - 4:
            continue
        if not np.all(grid[r0, c0 : c1 + 1] == MAROON):
            continue
        if not np.all(grid[r1, c0 : c1 + 1] == MAROON):
            continue
        if not np.all(grid[r0 : r1 + 1, c0] == MAROON):
            continue
        if not np.all(grid[r0 : r1 + 1, c1] == MAROON):
            continue
        if np.any(grid[r0 + 1 : r1, c0 + 1 : c1] == MAROON):
            continue
        candidates.append((r0, c0, r1, c1))
    assert len(candidates) == 1, "expected one maroon legend frame"
    return candidates[0]


def _legend_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = box
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _parse_legend(grid: np.ndarray, box: tuple[int, int, int, int]) -> dict[int, tuple[tuple[int, int], ...]]:
    r0, c0, r1, c1 = box
    inside = grid[r0 + 1 : r1, c0 + 1 : c1]
    used = np.zeros_like(inside, dtype=bool)
    mapping: dict[int, tuple[tuple[int, int], ...]] = {}
    key_mask = (inside != BLACK) & (inside != AZURE)

    for ir, ic in np.argwhere(key_mask):
        ir = int(ir)
        ic = int(ic)
        color = int(inside[ir, ic])
        assert color not in (GREY, MAROON), "legend keys must use tower colors"
        gr = r0 + 1 + ir
        gc = c0 + 1 + ic
        assert gr - 1 >= r0 + 1 and gr + 1 <= r1 - 1, "legend key must align to a 3x3 template"
        assert gc + 3 <= c1 - 1, "legend template must fit to the key's right"
        template = grid[gr - 1 : gr + 2, gc + 1 : gc + 4]
        assert template[1, 1] == AZURE, "template center must be marked"
        assert set(map(int, np.unique(template))) <= {BLACK, AZURE}, "templates use only black and azure"

        dirs = []
        for tr in range(3):
            for tc in range(3):
                if template[tr, tc] == AZURE and (tr, tc) != (1, 1):
                    dirs.append((tr - 1, tc - 1))
        assert dirs, "each legend entry must activate at least one direction"
        assert color not in mapping, "legend key colors must be unique"
        mapping[color] = tuple(dirs)

        used[ir, ic] = True
        used[ir - 1 : ir + 2, ic + 1 : ic + 4] = True

    assert mapping, "expected at least one legend entry"
    assert np.all(inside[~used] == BLACK), "legend interior should contain only entries on black"
    return mapping


def _find_towers(
    grid: np.ndarray,
    legend: np.ndarray,
) -> list[tuple[int, int, int]]:
    h, w = grid.shape
    towers: list[tuple[int, int, int]] = []
    for r in range(h):
        for c in range(w):
            if legend[r, c]:
                continue
            color = int(grid[r, c])
            if color in (BLACK, GREY, AZURE, MAROON):
                continue
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w:
                    assert grid[nr, nc] == BLACK, "towers must be isolated in 4-neighborhood"
            towers.append((r, c, color))
    assert towers, "expected at least one signal tower"
    return towers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each signal tower color to its legend template, then cast rays in the
    template's marked directions until a grey obstacle or the grid edge.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    legend_box = _legend_bbox(g)
    legend = _legend_mask(g.shape, legend_box)
    templates = _parse_legend(g, legend_box)
    towers = _find_towers(g, legend)

    out = g.copy()
    out[legend] = BLACK

    h, w = g.shape
    for tr, tc, color in towers:
        assert color in templates, "every tower color must appear in the legend"
        for dr, dc in templates[color]:
            r, c = tr, tc
            while True:
                nr = r + dr
                nc = c + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    break
                if legend[nr, nc] or g[nr, nc] == GREY:
                    break
                out[nr, nc] = color
                r, c = nr, nc

    return out


def _entry_centers(top: int, n_entries: int) -> list[int]:
    return [top + 2 + i * (ENTRY_H + ENTRY_GAP) for i in range(n_entries)]


def _template_patch(dirs: tuple[tuple[int, int], ...]) -> np.ndarray:
    patch = np.zeros((3, 3), dtype=int)
    patch[1, 1] = AZURE
    for dr, dc in dirs:
        patch[1 + dr, 1 + dc] = AZURE
    return patch


def _draw_legend(
    grid: np.ndarray,
    colors: list[int],
    templates: list[tuple[tuple[int, int], ...]],
    top: int,
) -> tuple[int, int, int, int]:
    frame_h = 4 * len(colors) + 1
    r0, c0 = top, 0
    r1, c1 = r0 + frame_h - 1, LEGEND_W - 1
    grid[r0 : r1 + 1, c0] = MAROON
    grid[r0 : r1 + 1, c1] = MAROON
    grid[r0, c0 : c1 + 1] = MAROON
    grid[r1, c0 : c1 + 1] = MAROON
    for center_r, color, dirs in zip(_entry_centers(top, len(colors)), colors, templates):
        grid[center_r, c0 + 1] = color
        patch = _template_patch(dirs)
        grid[center_r - 1 : center_r + 2, c0 + 2 : c0 + 5] = patch
    return (r0, c0, r1, c1)


def _box_with_pad(
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    h: int,
    w: int,
    pad: int = 1,
) -> tuple[int, int, int, int]:
    return (
        max(0, r0 - pad),
        max(0, c0 - pad),
        min(h - 1, r1 + pad),
        min(w - 1, c1 + pad),
    )


def _place_obstacles(
    shape: tuple[int, int],
    legend_box: tuple[int, int, int, int],
) -> list[tuple[int, int, int, int]]:
    h, w = shape
    occupied = np.zeros(shape, dtype=bool)
    lr0, lc0, lr1, lc1 = legend_box
    pr0, pc0, pr1, pc1 = _box_with_pad(lr0, lc0, lr1, lc1, h, w, pad=1)
    occupied[pr0 : pr1 + 1, pc0 : pc1 + 1] = True

    obstacles: list[tuple[int, int, int, int]] = []
    target = random.randint(*OBSTACLE_COUNT_RANGE)
    min_col = lc1 + 3
    if min_col >= w - 2:
        return []

    for _ in range(PLACEMENT_RETRIES):
        if len(obstacles) >= target:
            break
        oh = random.randint(*OBSTACLE_H_RANGE)
        ow = random.randint(*OBSTACLE_W_RANGE)
        if oh >= h - 2 or ow >= w - min_col - 1:
            continue
        r0 = random.randint(1, h - oh - 1)
        c0 = random.randint(min_col, w - ow - 1)
        r1 = r0 + oh - 1
        c1 = c0 + ow - 1
        pr0, pc0, pr1, pc1 = _box_with_pad(r0, c0, r1, c1, h, w, pad=1)
        if occupied[pr0 : pr1 + 1, pc0 : pc1 + 1].any():
            continue
        occupied[pr0 : pr1 + 1, pc0 : pc1 + 1] = True
        obstacles.append((r0, c0, r1, c1))

    return obstacles if len(obstacles) == target else []


def _paint_obstacles(grid: np.ndarray, obstacles: list[tuple[int, int, int, int]]) -> None:
    for r0, c0, r1, c1 in obstacles:
        grid[r0 : r1 + 1, c0 : c1 + 1] = GREY


def _isolated_background_cell(grid: np.ndarray, r: int, c: int) -> bool:
    if grid[r, c] != BLACK:
        return False
    h, w = grid.shape
    for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
        if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] != BLACK:
            return False
    return True


def _ray_trace(
    grid: np.ndarray,
    legend: np.ndarray,
    origin: tuple[int, int],
    dirs: tuple[tuple[int, int], ...],
) -> tuple[list[tuple[int, int]], list[str]] | None:
    h, w = grid.shape
    all_cells: list[tuple[int, int]] = []
    endings: list[str] = []
    for dr, dc in dirs:
        r, c = origin
        ray_cells: list[tuple[int, int]] = []
        while True:
            nr = r + dr
            nc = c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                if not ray_cells:
                    return None
                endings.append("boundary")
                break
            if legend[nr, nc]:
                return None
            value = int(grid[nr, nc])
            if value == GREY:
                if not ray_cells:
                    return None
                endings.append("obstacle")
                break
            if value != BLACK:
                return None
            ray_cells.append((nr, nc))
            r, c = nr, nc
        all_cells.extend(ray_cells)
    return all_cells, endings


def _place_towers(
    grid: np.ndarray,
    legend_box: tuple[int, int, int, int],
    mapping: dict[int, tuple[tuple[int, int], ...]],
) -> bool:
    h, w = grid.shape
    legend = _legend_mask(grid.shape, legend_box)
    reserved = np.zeros((h, w), dtype=bool)

    colors = list(mapping)
    extra = random.randint(*EXTRA_TOWERS_RANGE)
    tower_colors = colors + random.choices(colors, k=extra)
    random.shuffle(tower_colors)

    modes = ["obstacle", "boundary"] * ((len(tower_colors) + 1) // 2)
    modes = modes[: len(tower_colors)]
    random.shuffle(modes)

    candidate_cells = [
        (r, c)
        for r in range(h)
        for c in range(legend_box[3] + 2, w)
        if not legend[r, c]
    ]

    for color, mode in zip(tower_colors, modes):
        dirs = mapping[color]
        random.shuffle(candidate_cells)
        placed = False
        for r, c in candidate_cells:
            if reserved[r, c] or not _isolated_background_cell(grid, r, c):
                continue
            traced = _ray_trace(grid, legend, (r, c), dirs)
            if traced is None:
                continue
            ray_cells, endings = traced
            if mode not in endings:
                continue
            if any(reserved[rr, cc] for rr, cc in ray_cells):
                continue
            grid[r, c] = color
            reserved[r, c] = True
            for rr, cc in ray_cells:
                reserved[rr, cc] = True
            placed = True
            break
        if not placed:
            return False

    return True


def _generate_case() -> np.ndarray | None:
    n_entries = random.randint(3, min(5, len(TOWER_COLORS)))
    frame_h = 4 * n_entries + 1
    h = random.randint(max(GRID_H_RANGE[0], frame_h + 1), GRID_H_RANGE[1])
    w = random.randint(GRID_W_RANGE[0], GRID_W_RANGE[1])

    grid = np.zeros((h, w), dtype=int)
    colors = random.sample(list(TOWER_COLORS), n_entries)
    templates = random.sample(list(TEMPLATE_POOL), n_entries)
    legend_top = random.randint(0, h - frame_h)
    legend_box = _draw_legend(grid, colors, templates, legend_top)

    obstacles = _place_obstacles(grid.shape, legend_box)
    if not obstacles:
        return None
    _paint_obstacles(grid, obstacles)

    mapping = {color: template for color, template in zip(colors, templates)}
    if not _place_towers(grid, legend_box, mapping):
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    legend = _legend_mask(grid.shape, legend_box)
    if not np.all(out[legend] == BLACK):
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        case = _generate_case()
        if case is not None:
            return case
    raise RuntimeError("failed to generate a valid signal-tower puzzle")
