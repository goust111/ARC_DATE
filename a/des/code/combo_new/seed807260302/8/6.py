from __future__ import annotations

import random
from collections import deque

import numpy as np


PLUS_OFFSETS = (
    (-2, 0),
    (-1, 0),
    (0, -2),
    (0, -1),
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 0),
    (2, 0),
)

BACKGROUND = 0
FRAME_COLOR = 5
CROSS_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
OBJECT_COUNT_RANGE = (2, 4)
GRID_HEIGHT_RANGE = (12, 22)
GRID_WIDTH_RANGE = (18, 28)
FRAME_GAP_RANGE = (2, 5)
MARGIN_RANGE = (1, 4)
MAX_RELATIVE_SPAN = 12
GENERATION_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            comp = []
            while queue:
                rr, cc = queue.popleft()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _cross_cells(center: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    r, c = center
    return tuple((r + dr, c + dc) for dr, dc in PLUS_OFFSETS)


def _frame_candidates(grid: np.ndarray, bg: int) -> list[dict]:
    candidates = []
    for color in np.unique(grid):
        color = int(color)
        if color == int(bg):
            continue
        for comp in _components(grid == color):
            rows = [r for r, _c in comp]
            cols = [c for _r, c in comp]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            sub = grid[r0 : r1 + 1, c0 : c1 + 1]
            if sub.shape[1] != 3 or sub.shape[0] < 4:
                continue
            if not np.all(sub[0, :] == color):
                continue
            if not np.all(sub[-1, :] == color):
                continue
            if not np.all(sub[:, 0] == color):
                continue
            if not np.all(sub[:, -1] == color):
                continue
            if not np.all(sub[1:-1, 1:-1] == bg):
                continue
            candidates.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "height": int(sub.shape[0]),
                    "width": int(sub.shape[1]),
                }
            )
    candidates.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    return candidates


def _find_output_frame(grid: np.ndarray, bg: int) -> dict:
    candidates = _frame_candidates(grid, bg)
    if not candidates:
        raise ValueError("no valid output frame found")
    return candidates[-1]


def _candidate_centers(
    grid: np.ndarray,
    color: int,
    bg: int,
    frame_color: int,
) -> list[tuple[int, int]]:
    visible = np.argwhere(grid == color)
    if visible.size == 0:
        return []

    centers: set[tuple[int, int]] | None = None
    for vr, vc in visible:
        local = {(int(vr - dr), int(vc - dc)) for dr, dc in PLUS_OFFSETS}
        centers = local if centers is None else centers & local

    h, w = grid.shape
    valid = []
    for center in sorted(centers or ()):
        cells = _cross_cells(center)
        ok = True
        for rr, cc in cells:
            if not (0 <= rr < h and 0 <= cc < w):
                ok = False
                break
            value = int(grid[rr, cc])
            if value == int(bg) or value == int(frame_color):
                ok = False
                break
        if ok:
            valid.append(center)
    return valid


def _has_cycle(colors: list[int], edges: dict[int, set[int]]) -> bool:
    indeg = {color: 0 for color in colors}
    for src in colors:
        for dst in edges[src]:
            indeg[dst] += 1
    queue = deque([color for color in colors if indeg[color] == 0])
    seen = 0
    while queue:
        cur = queue.popleft()
        seen += 1
        for nxt in edges[cur]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue.append(nxt)
    return seen != len(colors)


def _unique_topological_order(colors: list[int], edges: dict[int, set[int]]) -> list[int] | None:
    indeg = {color: 0 for color in colors}
    for src in colors:
        for dst in edges[src]:
            indeg[dst] += 1

    remaining = set(colors)
    order = []
    while remaining:
        zeros = [color for color in remaining if indeg[color] == 0]
        if len(zeros) != 1:
            return None
        cur = zeros[0]
        order.append(cur)
        remaining.remove(cur)
        for nxt in edges[cur]:
            indeg[nxt] -= 1

    return order


def _order_for_assignment(
    grid: np.ndarray,
    colors: list[int],
    assignment: dict[int, tuple[int, int]],
) -> list[int] | None:
    edges = {color: set() for color in colors}
    for color in colors:
        for rr, cc in _cross_cells(assignment[color]):
            top = int(grid[rr, cc])
            if top == color:
                continue
            if top not in edges:
                return None
            edges[color].add(top)
    return _unique_topological_order(colors, edges)


def _solve_scene(grid: np.ndarray) -> tuple[dict, list[int]]:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    frame = _find_output_frame(g, bg)
    frame_color = int(frame["color"])

    colors = [int(color) for color in np.unique(g) if color not in (bg, frame_color)]
    inner_height = int(frame["height"]) - 2
    if len(colors) != inner_height:
        raise ValueError("frame height does not match number of visible cross colors")

    candidates = {
        color: _candidate_centers(g, color, bg, frame_color)
        for color in colors
    }
    if any(not options for options in candidates.values()):
        raise ValueError("at least one color cannot be matched to a cross template")

    search_order = sorted(colors, key=lambda color: (len(candidates[color]), color))
    assignments: dict[int, tuple[int, int]] = {}
    recovered_orders: set[tuple[int, ...]] = set()

    def partial_ok() -> bool:
        assigned_colors = list(assignments)
        edges = {color: set() for color in assigned_colors}
        for color in assigned_colors:
            for rr, cc in _cross_cells(assignments[color]):
                top = int(g[rr, cc])
                if top in edges and top != color:
                    edges[color].add(top)
        return not _has_cycle(assigned_colors, edges)

    def dfs(idx: int) -> None:
        if len(recovered_orders) > 1:
            return
        if idx == len(search_order):
            order = _order_for_assignment(g, colors, assignments)
            if order is not None:
                recovered_orders.add(tuple(order))
            return

        color = search_order[idx]
        for center in candidates[color]:
            assignments[color] = center
            if partial_ok():
                dfs(idx + 1)
            del assignments[color]

    dfs(0)
    if len(recovered_orders) != 1:
        raise ValueError("scene does not admit a unique recoverable layer order")

    return frame, list(next(iter(recovered_orders)))


def _render_output(frame: dict, bg: int, order: list[int]) -> np.ndarray:
    height = int(frame["height"])
    width = int(frame["width"])
    frame_color = int(frame["color"])
    out = np.full((height, width), int(bg), dtype=int)
    out[0, :] = frame_color
    out[-1, :] = frame_color
    out[:, 0] = frame_color
    out[:, -1] = frame_color
    for idx, color in enumerate(order, start=1):
        out[idx, 1] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover overlapped fixed-size cross layers and write their colors into the right-hand frame from bottom to top.

    Input structure:
    - The background is black, and the right side contains a grey rectangular frame with a 1-cell-wide empty interior column.
    - Each non-grey foreground color belongs to one 5x5 plus-sign template: a center pixel with arms of length 2 in the four cardinal directions.
    - Different crosses overlap, and visible pixels show only the topmost color at each occupied cell.

    Rule:
    - Detect the grey frame and use its interior height as the number of crosses to recover.
    - For each visible foreground color, infer the translated cross template whose nine cells are all occupied by cross colors, even if some of those cells are hidden by higher layers.
    - Whenever one reconstructed template contains a cell currently showing another color, the shown color is above it; use these covering relations to recover the unique bottom-to-top layer order, then fill the frame's interior column from top to bottom with that order.

    Key insight:
    - A visible color fragment is only a partial view of a fixed plus template, so the puzzle is solved by completing each hidden template first and then reading the occlusion graph as a layer-order graph.
    """
    frame, order = _solve_scene(np.asarray(grid, dtype=int))
    return _render_output(frame, BACKGROUND, order)


def _cross_footprint(center: tuple[int, int]) -> set[tuple[int, int]]:
    return set(_cross_cells(center))


def _cluster_span(centers: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    cells = [cell for center in centers for cell in _cross_cells(center)]
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _overlap_offsets() -> tuple[tuple[int, int], ...]:
    base = _cross_footprint((0, 0))
    offsets = []
    for dr in range(-4, 5):
        for dc in range(-4, 5):
            if dr == 0 and dc == 0:
                continue
            shifted = {(r + dr, c + dc) for r, c in base}
            overlap = len(base & shifted)
            if 1 <= overlap <= 6:
                offsets.append((dr, dc))
    return tuple(offsets)


OVERLAP_OFFSETS = _overlap_offsets()


def _draw_cross(grid: np.ndarray, center: tuple[int, int], color: int) -> None:
    for rr, cc in _cross_cells(center):
        grid[rr, cc] = int(color)


def _draw_frame(grid: np.ndarray, top: int, left: int, inner_height: int, color: int) -> None:
    bottom = top + inner_height + 1
    right = left + 2
    grid[top, left : right + 1] = int(color)
    grid[bottom, left : right + 1] = int(color)
    grid[top : bottom + 1, left] = int(color)
    grid[top : bottom + 1, right] = int(color)


def _random_centers(n: int) -> list[tuple[int, int]] | None:
    centers = [(0, 0)]
    for idx in range(1, n):
        placed = False
        for _ in range(60):
            anchor = centers[idx - 1] if random.random() < 0.7 else random.choice(centers)
            dr, dc = random.choice(OVERLAP_OFFSETS)
            cand = (anchor[0] + dr, anchor[1] + dc)
            if cand in centers:
                continue
            trial = centers + [cand]
            r0, c0, r1, c1 = _cluster_span(trial)
            if r1 - r0 + 1 > MAX_RELATIVE_SPAN or c1 - c0 + 1 > MAX_RELATIVE_SPAN:
                continue
            centers.append(cand)
            placed = True
            break
        if not placed:
            return None
    return centers


def _generate_attempt() -> np.ndarray | None:
    n = random.randint(*OBJECT_COUNT_RANGE)
    order = random.sample(CROSS_COLORS, n)
    rel_centers = _random_centers(n)
    if rel_centers is None:
        return None

    rel_r0, rel_c0, rel_r1, rel_c1 = _cluster_span(rel_centers)
    cluster_h = rel_r1 - rel_r0 + 1
    cluster_w = rel_c1 - rel_c0 + 1
    frame_h = n + 2
    frame_w = 3

    grid_h = random.randint(max(GRID_HEIGHT_RANGE[0], max(cluster_h, frame_h) + 2), GRID_HEIGHT_RANGE[1])
    left_margin = random.randint(*MARGIN_RANGE)
    gap = random.randint(*FRAME_GAP_RANGE)
    right_margin = random.randint(*MARGIN_RANGE)
    min_w = left_margin + cluster_w + gap + frame_w + right_margin
    if min_w > GRID_WIDTH_RANGE[1]:
        return None
    grid_w = random.randint(max(GRID_WIDTH_RANGE[0], min_w), GRID_WIDTH_RANGE[1])

    max_cluster_top = grid_h - cluster_h - 1
    max_frame_top = grid_h - frame_h - 1
    if max_cluster_top < 1 or max_frame_top < 1:
        return None

    cluster_top = random.randint(1, max_cluster_top)
    frame_top = random.randint(1, max_frame_top)
    frame_left = grid_w - frame_w - right_margin
    max_cluster_left = frame_left - gap - cluster_w
    if max_cluster_left < left_margin:
        return None
    cluster_left = random.randint(left_margin, max_cluster_left)

    shift_r = cluster_top - rel_r0
    shift_c = cluster_left - rel_c0
    centers = [(r + shift_r, c + shift_c) for r, c in rel_centers]

    grid = np.full((grid_h, grid_w), BACKGROUND, dtype=int)
    _draw_frame(grid, frame_top, frame_left, n, FRAME_COLOR)
    for color, center in zip(order, centers):
        _draw_cross(grid, center, color)

    visible_counts = {color: int(np.sum(grid == color)) for color in order}
    if min(visible_counts.values()) < 2:
        return None

    try:
        frame, recovered_order = _solve_scene(grid)
    except ValueError:
        return None

    if recovered_order != order:
        return None

    expected = _render_output(frame, BACKGROUND, order)
    if not np.array_equal(main(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid overlapped-cross input")
