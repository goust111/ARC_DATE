from __future__ import annotations

from collections import deque
import itertools
import random

import numpy as np


BG = 0
FRAME = 5
MARKER = 9
DOUBLE_OVERLAP = 7
TRIPLE_OVERLAP = 6
SHAPE_COLOR_POOL = (1, 2, 3, 4, 8)

GEN_MAX_ATTEMPTS = 400
GRID_HEIGHT_RANGE = (22, 30)
GRID_WIDTH_RANGE = (24, 32)
SHAPE_COUNT_RANGE = (3, 5)
FRAME_MARGIN_RANGE = (1, 3)
PLACEMENT_PADDING = 1
BRANCH_RADIUS = 6
BRANCH_LEN_RANGE = (2, 4)
EXTRA_UNIQUE_BRANCH_PROB = 0.45

CORE_CROSS = frozenset({(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)})
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _neighbors4(r: int, c: int, height: int, width: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < height:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < width:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    seen = np.zeros((height, width), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, height, width):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _frame_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    best = None
    best_key = None
    for coords in _components(grid == FRAME):
        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            continue
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros_like(sub, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if not np.all(sub[border] == FRAME):
            continue
        if np.any(sub[1:-1, 1:-1] != BG):
            continue
        key = (h * w, r0 + c0)
        if best is None or key > best_key:
            best = (r0, c0, r1, c1)
            best_key = key
    assert best is not None, "target grey frame not found"
    return best


def _template_order(grid: np.ndarray) -> tuple[list[int], set[tuple[int, int]]]:
    candidates: list[tuple[int, int, list[int], set[tuple[int, int]]]] = []
    for coords in _components((grid != BG) & (grid != FRAME)):
        r0, c0, r1, c1 = _bbox(coords)
        if c0 != c1:
            continue
        if int(coords.shape[0]) != r1 - r0 + 1:
            continue
        rows = sorted(int(r) for r in coords[:, 0])
        if rows != list(range(r0, r1 + 1)):
            continue
        colors = [int(grid[r, c0]) for r in rows]
        if MARKER in colors or len(set(colors)) != len(colors):
            continue
        coords_set = {(int(r), int(c0)) for r in rows}
        candidates.append((c0, -len(colors), colors, coords_set))
    assert candidates, "template strip not found"
    candidates.sort()
    _, _, order, coords_set = candidates[0]
    return order, coords_set


def _marker_center(coords: np.ndarray, grid: np.ndarray) -> tuple[int, int]:
    marker_cells = {(int(r), int(c)) for r, c in coords if int(grid[int(r), int(c)]) == MARKER}
    assert marker_cells, "shape marker not found"
    for r, c in marker_cells:
        if all((r + dr, c + dc) in marker_cells for dr, dc in DIRS4):
            return r, c
    rows = sorted(r for r, _ in marker_cells)
    cols = sorted(c for _, c in marker_cells)
    return rows[len(rows) // 2], cols[len(cols) // 2]


def _shape_specs(grid: np.ndarray, template_cells: set[tuple[int, int]]) -> dict[int, tuple[tuple[int, int], ...]]:
    specs: dict[int, tuple[tuple[int, int], ...]] = {}
    for coords in _components((grid != BG) & (grid != FRAME)):
        cell_set = {tuple(map(int, rc)) for rc in coords.tolist()}
        if cell_set == template_cells:
            continue
        colors = {int(grid[r, c]) for r, c in cell_set}
        non_marker = [color for color in colors if color != MARKER]
        if MARKER not in colors or len(non_marker) != 1:
            continue
        color = int(non_marker[0])
        center_r, center_c = _marker_center(coords, grid)
        rel = tuple(
            sorted((int(r) - center_r, int(c) - center_c) for r, c in cell_set)
        )
        specs[color] = rel
    assert specs, "no anchored shapes found"
    return specs


def _render(order: list[int], shapes: dict[int, tuple[tuple[int, int], ...]], shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    out = np.full((height, width), BG, dtype=int)
    out[0, :] = FRAME
    out[-1, :] = FRAME
    out[:, 0] = FRAME
    out[:, -1] = FRAME

    center_r = height // 2
    center_c = width // 2
    counts = np.zeros((height, width), dtype=int)
    single_color = np.zeros((height, width), dtype=int)

    for color in order:
        rel = shapes.get(int(color))
        if rel is None:
            continue
        for dr, dc in rel:
            rr = center_r + int(dr)
            cc = center_c + int(dc)
            if 1 <= rr < height - 1 and 1 <= cc < width - 1:
                counts[rr, cc] += 1
                single_color[rr, cc] = int(color)

    out[counts == 1] = single_color[counts == 1]
    out[counts == 2] = DOUBLE_OVERLAP
    out[counts >= 3] = TRIPLE_OVERLAP
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align each colored shape's maroon cross with the grey frame center, then
    render the stacked result inside the frame crop. A covered cell keeps its
    original color for one layer, becomes orange for exactly two layers, and
    becomes magenta for three or more layers.
    """
    g = np.asarray(grid, dtype=int)
    r0, c0, r1, c1 = _frame_bbox(g)
    order, template_cells = _template_order(g)
    shapes = _shape_specs(g, template_cells)
    return _render(order, shapes, (r1 - r0 + 1, c1 - c0 + 1))


def _frontier(seed: set[tuple[int, int]], forbidden: set[tuple[int, int]]) -> list[tuple[int, int]]:
    frontier: set[tuple[int, int]] = set()
    for r, c in seed:
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            cell = (nr, nc)
            if max(abs(nr), abs(nc)) > BRANCH_RADIUS:
                continue
            if cell in seed or cell in forbidden or cell in CORE_CROSS:
                continue
            frontier.add(cell)
    return list(frontier)


def _grow_motif(forbidden: set[tuple[int, int]]) -> set[tuple[int, int]] | None:
    for _ in range(120):
        target_len = random.randint(*BRANCH_LEN_RANGE)
        cells: set[tuple[int, int]] = set()
        first_choices = _frontier(set(CORE_CROSS), forbidden)
        if not first_choices:
            return None
        cells.add(random.choice(first_choices))
        while len(cells) < target_len:
            choices = _frontier(set(CORE_CROSS) | cells, forbidden | cells)
            if not choices:
                break
            cells.add(random.choice(choices))
        if len(cells) == target_len:
            return cells
    return None


def _build_relative_shapes(n_shapes: int) -> dict[int, tuple[tuple[int, int], ...]] | None:
    colors = random.sample(list(SHAPE_COLOR_POOL), n_shapes)
    rel_shapes: list[set[tuple[int, int]]] = [set(CORE_CROSS) for _ in range(n_shapes)]
    used = set(CORE_CROSS)

    pairs = list(itertools.combinations(range(n_shapes), 2))
    random.shuffle(pairs)
    pair_count = random.randint(1, min(3, len(pairs)))
    for i, j in pairs[:pair_count]:
        motif = _grow_motif(used)
        if motif is None:
            return None
        used |= motif
        rel_shapes[i] |= motif
        rel_shapes[j] |= motif

    for i in range(n_shapes):
        branch_count = 1 + int(random.random() < EXTRA_UNIQUE_BRANCH_PROB)
        for _ in range(branch_count):
            motif = _grow_motif(used)
            if motif is None:
                return None
            used |= motif
            rel_shapes[i] |= motif

    out: dict[int, tuple[tuple[int, int], ...]] = {}
    for color, cells in zip(colors, rel_shapes):
        out[int(color)] = tuple(sorted(cells))
    return out


def _frame_shape(rel_shapes: dict[int, tuple[tuple[int, int], ...]]) -> tuple[int, int]:
    max_r = max(max(abs(r) for r, _ in cells) for cells in rel_shapes.values())
    max_c = max(max(abs(c) for _, c in cells) for cells in rel_shapes.values())
    half_h = max_r + random.randint(1, 3)
    half_w = max_c + random.randint(1, 4)
    return 2 * half_h + 3, 2 * half_w + 3


def _shape_patch(rel_cells: tuple[tuple[int, int], ...], color: int) -> tuple[np.ndarray, tuple[int, int]]:
    rows = [r for r, _ in rel_cells]
    cols = [c for _, c in rel_cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
    for r, c in rel_cells:
        patch[r - r0, c - c0] = MARKER if (r, c) in CORE_CROSS else int(color)
    return patch, (-r0, -c0)


def _can_place_rect(occ: np.ndarray, r0: int, c0: int, h: int, w: int, padding: int) -> bool:
    height, width = occ.shape
    if r0 < 0 or c0 < 0 or r0 + h > height or c0 + w > width:
        return False
    rr0 = max(0, r0 - padding)
    cc0 = max(0, c0 - padding)
    rr1 = min(height, r0 + h + padding)
    cc1 = min(width, c0 + w + padding)
    return not bool(occ[rr0:rr1, cc0:cc1].any())


def _mark_rect(occ: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    occ[r0 : r0 + h, c0 : c0 + w] = True


def _paste_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    mask = patch != BG
    view = grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
    view[mask] = patch[mask]


def _draw_blank_frame(grid: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    grid[r0 : r0 + h, c0 : c0 + w] = BG
    grid[r0, c0 : c0 + w] = FRAME
    grid[r0 + h - 1, c0 : c0 + w] = FRAME
    grid[r0 : r0 + h, c0] = FRAME
    grid[r0 : r0 + h, c0 + w - 1] = FRAME


def _coverage_classes(frame: np.ndarray) -> set[int]:
    interior = frame[1:-1, 1:-1]
    values = set(int(v) for v in np.unique(interior))
    out: set[int] = set()
    if DOUBLE_OVERLAP in values:
        out.add(2)
    if TRIPLE_OVERLAP in values:
        out.add(3)
    if any(v in SHAPE_COLOR_POOL for v in values):
        out.add(1)
    return out


def _assemble_grid(
    order: list[int],
    rel_shapes: dict[int, tuple[tuple[int, int], ...]],
    expected: np.ndarray,
) -> np.ndarray | None:
    frame_h, frame_w = expected.shape
    patches = []
    for color in order:
        patch, _ = _shape_patch(rel_shapes[color], color)
        patches.append((color, patch))

    min_h = max(GRID_HEIGHT_RANGE[0], frame_h + 4)
    min_w = max(GRID_WIDTH_RANGE[0], frame_w + 8)
    if min_h > GRID_HEIGHT_RANGE[1] or min_w > GRID_WIDTH_RANGE[1]:
        return None

    for _ in range(120):
        height = random.randint(min_h, GRID_HEIGHT_RANGE[1])
        width = random.randint(min_w, GRID_WIDTH_RANGE[1])
        grid = np.full((height, width), BG, dtype=int)
        occ = np.zeros((height, width), dtype=bool)

        bottom_margin = random.randint(*FRAME_MARGIN_RANGE)
        right_margin = random.randint(*FRAME_MARGIN_RANGE)
        frame_r = height - frame_h - bottom_margin
        frame_c = width - frame_w - right_margin
        if frame_r < 4 or frame_c < 8:
            continue

        _draw_blank_frame(grid, frame_r, frame_c, frame_h, frame_w)
        _mark_rect(occ, frame_r, frame_c, frame_h, frame_w)

        strip_r = random.randint(1, height - len(order) - 2)
        strip_c = random.randint(0, min(2, frame_c - 3))
        if not _can_place_rect(occ, strip_r, strip_c, len(order), 1, padding=2):
            continue
        for i, color in enumerate(order):
            grid[strip_r + i, strip_c] = int(color)
        _mark_rect(occ, strip_r, strip_c, len(order), 1)

        placements_ok = True
        random.shuffle(patches)
        for _color, patch in sorted(patches, key=lambda item: -int(np.count_nonzero(item[1]))):
            ph, pw = patch.shape
            candidates: list[tuple[int, int]] = []
            for r0 in range(1, height - ph):
                for c0 in range(1, width - pw):
                    if not _can_place_rect(occ, r0, c0, ph, pw, padding=PLACEMENT_PADDING):
                        continue
                    if r0 + ph > frame_r - 1 and c0 + pw > frame_c - 1:
                        continue
                    candidates.append((r0, c0))
            if not candidates:
                placements_ok = False
                break
            r0, c0 = random.choice(candidates)
            _paste_patch(grid, patch, r0, c0)
            _mark_rect(occ, r0, c0, ph, pw)
        if placements_ok:
            return grid
    return None


def _gen_attempt() -> np.ndarray | None:
    n_shapes = random.randint(*SHAPE_COUNT_RANGE)
    rel_shapes = _build_relative_shapes(n_shapes)
    if rel_shapes is None:
        return None

    order = random.sample(list(rel_shapes), len(rel_shapes))
    frame_shape = _frame_shape(rel_shapes)
    expected = _render(order, rel_shapes, frame_shape)
    if _coverage_classes(expected) != {1, 2, 3}:
        return None

    grid = _assemble_grid(order, rel_shapes, expected)
    if grid is None:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
