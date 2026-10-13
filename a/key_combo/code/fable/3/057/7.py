from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
NON_BLACK_COLORS = tuple(range(1, 10))
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_palette(grid: np.ndarray) -> list[int]:
    width = int(grid.shape[1])
    length = 0
    while length < width and int(grid[0, length]) != BG:
        length += 1
    assert length > 0, "expected a non-empty palette row starting at the top-left corner"
    return [int(color) for color in grid[0, :length]]


def _boundary_gap(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    gaps: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        if int(grid[r0, c]) == BG:
            gaps.append((r0, c))
        if int(grid[r1, c]) == BG:
            gaps.append((r1, c))
    for r in range(r0 + 1, r1):
        if int(grid[r, c0]) == BG:
            gaps.append((r, c0))
        if int(grid[r, c1]) == BG:
            gaps.append((r, c1))
    assert len(gaps) == 1, "expected exactly one wall opening on the container boundary"
    return gaps[0]


def _distance_map(mask: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    dist = np.full(mask.shape, -1, dtype=int)
    queue: deque[tuple[int, int]] = deque([start])
    dist[start] = 0
    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and dist[nr, nc] < 0:
                dist[nr, nc] = dist[r, c] + 1
                queue.append((nr, nc))
    return dist


def _distance_map_on_cells(cells: set[tuple[int, int]], start: tuple[int, int]) -> dict[tuple[int, int], int]:
    queue: deque[tuple[int, int]] = deque([start])
    dist = {start: 0}
    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            nxt = (nr, nc)
            if nxt in cells and nxt not in dist:
                dist[nxt] = dist[(r, c)] + 1
                queue.append(nxt)
    return dist


def main(grid: np.ndarray) -> np.ndarray:
    """
    Flood the open container interior with palette colors according to geodesic layers from the opening.

    Input structure:
    - The top row starts with a contiguous non-black palette strip whose length equals the number of fill layers.
    - Below it sits one monochrome container wall on a black background, with exactly one single-cell opening on the wall boundary.
    - Black cells inside the wall form one connected interior cavity whose shape may bend or branch.

    Rule:
    - Keep the palette row, the wall cells, the opening cell, and the exterior background unchanged.
    - Identify the cell immediately inside the opening and compute 4-connected shortest-path distances through the interior black cells.
    - Recolor every interior cell at distance d with the d-th palette color from the top row.

    Key insight:
    - The coloring layers are wavefronts that flow from the unique opening through the cavity, not rings based on distance to the wall.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    palette = _extract_palette(g)

    wall_mask = g != BG
    wall_mask[0, :] = False
    wall_bbox = _bbox(wall_mask)
    assert wall_bbox is not None, "expected a non-black container below the palette row"

    opening = _boundary_gap(g, wall_bbox)
    r0, c0, r1, c1 = wall_bbox

    cavity_mask = np.zeros_like(g, dtype=bool)
    cavity_mask[r0 : r1 + 1, c0 : c1 + 1] = (g[r0 : r1 + 1, c0 : c1 + 1] == BG)
    cavity_mask[opening] = False

    start_candidates = []
    for nr, nc in _neighbors4(*opening):
        if r0 <= nr <= r1 and c0 <= nc <= c1 and cavity_mask[nr, nc]:
            start_candidates.append((nr, nc))
    assert len(start_candidates) == 1, "expected a single interior seed next to the opening"
    start = start_candidates[0]

    dist = _distance_map(cavity_mask, start)
    assert np.all(dist[cavity_mask] >= 0), "expected one connected interior cavity"
    max_dist = int(dist[cavity_mask].max())
    assert len(palette) == max_dist + 1, "palette length must match the number of geodesic layers"

    out = g.copy()
    for depth, color in enumerate(palette):
        out[dist == depth] = int(color)
    return out


def _interior_neighbors(cell: tuple[int, int], block_h: int, block_w: int) -> list[tuple[int, int]]:
    r, c = cell
    out = []
    for nr, nc in _neighbors4(r, c):
        if 1 <= nr < block_h - 1 and 1 <= nc < block_w - 1:
            out.append((nr, nc))
    return out


def _free_degree(
    cell: tuple[int, int], used: set[tuple[int, int]], block_h: int, block_w: int
) -> int:
    return sum(1 for nbr in _interior_neighbors(cell, block_h, block_w) if nbr not in used)


def _path_turns(path: list[tuple[int, int]]) -> int:
    turns = 0
    for idx in range(2, len(path)):
        dr0 = path[idx - 1][0] - path[idx - 2][0]
        dc0 = path[idx - 1][1] - path[idx - 2][1]
        dr1 = path[idx][0] - path[idx - 1][0]
        dc1 = path[idx][1] - path[idx - 1][1]
        if (dr0, dc0) != (dr1, dc1):
            turns += 1
    return turns


def _extend_path(
    path: list[tuple[int, int]],
    used: set[tuple[int, int]],
    block_h: int,
    block_w: int,
    target_steps: int,
) -> bool:
    if len(path) - 1 == target_steps:
        return True

    current = path[-1]
    candidates = [nbr for nbr in _interior_neighbors(current, block_h, block_w) if nbr not in used]
    random.shuffle(candidates)
    candidates.sort(key=lambda cell: _free_degree(cell, used, block_h, block_w), reverse=True)
    for nbr in candidates:
        used.add(nbr)
        path.append(nbr)
        if _extend_path(path, used, block_h, block_w, target_steps):
            return True
        path.pop()
        used.remove(nbr)
    return False


def _random_path(
    start: tuple[int, int], block_h: int, block_w: int, target_steps: int
) -> list[tuple[int, int]] | None:
    for _ in range(60):
        path = [start]
        used = {start}
        if _extend_path(path, used, block_h, block_w, target_steps) and _path_turns(path) >= 1:
            return path
    return None


def _branch_path(
    anchor: tuple[int, int],
    occupied: set[tuple[int, int]],
    block_h: int,
    block_w: int,
    length: int,
) -> list[tuple[int, int]] | None:
    for _ in range(40):
        branch: list[tuple[int, int]] = []
        used = set(occupied)
        current = anchor
        success = True
        for _step in range(length):
            options = [nbr for nbr in _interior_neighbors(current, block_h, block_w) if nbr not in used]
            random.shuffle(options)
            if not options:
                success = False
                break
            nxt = options[0]
            branch.append(nxt)
            used.add(nxt)
            current = nxt
        if success:
            return branch
    return None


def _grow_cavity(
    path: list[tuple[int, int]], block_h: int, block_w: int, max_layers: int
) -> set[tuple[int, int]] | None:
    cavity = set(path)
    path_dist = {cell: idx for idx, cell in enumerate(path)}
    anchor_indices = list(range(1, len(path) - 1))
    random.shuffle(anchor_indices)

    branch_added = False
    for idx in anchor_indices:
        anchor = path[idx]
        max_branch_len = min(2, max_layers - 1 - path_dist[anchor])
        if max_branch_len < 1:
            continue
        branch_len = random.randint(1, max_branch_len)
        branch = _branch_path(anchor, cavity, block_h, block_w, branch_len)
        if branch is not None:
            cavity.update(branch)
            branch_added = True
            break

    if not branch_added:
        return None

    extra_cells = random.randint(0, 4)
    for _ in range(extra_cells):
        dist = _distance_map_on_cells(cavity, path[0])
        candidates = []
        for cell in list(cavity):
            for nbr in _interior_neighbors(cell, block_h, block_w):
                if nbr in cavity:
                    continue
                projected = dist[cell] + 1
                if projected <= max_layers - 1:
                    touch_count = sum(1 for adj in _interior_neighbors(nbr, block_h, block_w) if adj in cavity)
                    candidates.append((touch_count, nbr))
        if not candidates:
            break
        best_touch = max(score for score, _ in candidates)
        filtered = [nbr for score, nbr in candidates if score == best_touch]
        cavity.add(random.choice(filtered))

    return cavity


def _start_and_opening(side: str, block_h: int, block_w: int) -> tuple[tuple[int, int], tuple[int, int]]:
    if side == "top":
        c = random.randint(1, block_w - 2)
        return (1, c), (0, c)
    if side == "bottom":
        c = random.randint(1, block_w - 2)
        return (block_h - 2, c), (block_h - 1, c)
    if side == "left":
        r = random.randint(1, block_h - 2)
        return (r, 1), (r, 0)
    assert side == "right"
    r = random.randint(1, block_h - 2)
    return (r, block_w - 2), (r, block_w - 1)


def _build_candidate() -> np.ndarray | None:
    max_layers = random.randint(4, 7)
    target_steps = random.randint(max(3, max_layers - 2), max_layers - 1)

    block_h = random.randint(max(7, target_steps + 3), 13)
    block_w = random.randint(max(7, target_steps + 3), 13)
    side = random.choice(["top", "bottom", "left", "right"])
    start, opening = _start_and_opening(side, block_h, block_w)

    path = _random_path(start, block_h, block_w, target_steps)
    if path is None:
        return None

    cavity = _grow_cavity(path, block_h, block_w, max_layers)
    if cavity is None:
        return None

    dist = _distance_map_on_cells(cavity, start)
    actual_layers = max(dist.values()) + 1
    if not (4 <= actual_layers <= 7):
        return None
    if len(cavity) < actual_layers + 2:
        return None

    wall_color = random.choice(NON_BLACK_COLORS)
    palette_choices = [color for color in NON_BLACK_COLORS if color != wall_color]
    palette = random.sample(palette_choices, actual_layers)

    top_margin = random.randint(2, 5)
    left_margin = random.randint(1, 4)
    bottom_margin = random.randint(1, 4)
    right_margin = random.randint(1, 4)

    height = top_margin + block_h + bottom_margin
    width = max(left_margin + block_w + right_margin, actual_layers + random.randint(1, 4))
    if height > 30 or width > 30:
        return None

    grid = np.zeros((height, width), dtype=int)
    grid[0, :actual_layers] = np.array(palette, dtype=int)
    top = top_margin
    left = left_margin
    grid[top : top + block_h, left : left + block_w] = int(wall_color)
    grid[top + opening[0], left + opening[1]] = BG
    for r, c in cavity:
        grid[top + r, left + c] = BG

    try:
        out = main(grid)
    except AssertionError:
        return None

    changed = out != grid
    if int(changed.sum()) != len(cavity):
        return None
    if not np.array_equal(out[0], grid[0]):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
