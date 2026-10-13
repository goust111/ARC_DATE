from __future__ import annotations

from dataclasses import dataclass

import numpy as np


PALETTE = tuple(range(10))
GEN_RETRIES = 500
GRID_MIN = 18
GRID_MAX = 30
RECT_COUNT_OPTIONS = (3, 4, 5)
AREA_OPTIONS = (25, 35, 45, 49, 63, 81)
AREA_TO_SHAPES = {
    25: ((5, 5),),
    35: ((5, 7), (7, 5)),
    45: ((5, 9), (9, 5)),
    49: ((7, 7),),
    63: ((7, 9), (9, 7)),
    81: ((9, 9),),
}
MIN_RECT_GAP = 1
MIN_VISIBLE_INTERSECTIONS = 4


@dataclass(frozen=True)
class Rectangle:
    color: int
    marker_color: int
    bbox: tuple[int, int, int, int]
    center: tuple[int, int]
    marker_pos: tuple[int, int]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _is_hollow_rectangle(cells: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = _bbox(cells)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3 or height % 2 == 0 or width % 2 == 0:
        return None
    expected = 2 * height + 2 * width - 4
    if len(cells) != expected:
        return None
    cell_set = set(cells)
    for rr in range(r0, r1 + 1):
        for cc in range(c0, c1 + 1):
            on_edge = rr in (r0, r1) or cc in (c0, c1)
            if on_edge != ((rr, cc) in cell_set):
                return None
    return r0, c0, r1, c1


def _extract_rectangles(grid: np.ndarray) -> list[Rectangle]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    rectangles: list[Rectangle] = []
    seen_colors: set[int] = set()
    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        for comp in _components_of_color(g, color):
            box = _is_hollow_rectangle(comp)
            if box is None:
                continue
            if color in seen_colors:
                raise AssertionError("rectangle border colors must be unique")
            r0, c0, r1, c1 = box
            interior = g[r0 + 1 : r1, c0 + 1 : c1]
            marker_cells = np.argwhere(interior != bg)
            if marker_cells.shape != (1, 2):
                raise AssertionError("each rectangle must contain exactly one interior marker pixel")
            mr = int(marker_cells[0, 0]) + r0 + 1
            mc = int(marker_cells[0, 1]) + c0 + 1
            center = ((r0 + r1) // 2, (c0 + c1) // 2)
            rectangles.append(
                Rectangle(
                    color=color,
                    marker_color=int(g[mr, mc]),
                    bbox=box,
                    center=center,
                    marker_pos=(mr, mc),
                )
            )
            seen_colors.add(color)
    if len(rectangles) < 2:
        raise AssertionError("expected at least two hollow rectangles")
    rectangles.sort(key=lambda rect: rect.bbox)
    return rectangles


def _decode_chain(rectangles: list[Rectangle]) -> list[int]:
    border_to_idx = {rect.color: idx for idx, rect in enumerate(rectangles)}
    next_idx: list[int | None] = []
    indegree = [0] * len(rectangles)
    tail_count = 0
    for idx, rect in enumerate(rectangles):
        dst = border_to_idx.get(rect.marker_color)
        if dst is None:
            next_idx.append(None)
            tail_count += 1
            continue
        if dst == idx:
            raise AssertionError("a rectangle cannot point to itself")
        indegree[dst] += 1
        next_idx.append(dst)
    if tail_count != 1:
        raise AssertionError("expected exactly one chain tail")
    heads = [idx for idx, deg in enumerate(indegree) if deg == 0]
    if len(heads) != 1:
        raise AssertionError("expected exactly one chain head")
    order: list[int] = []
    current = heads[0]
    visited: set[int] = set()
    while current is not None:
        if current in visited:
            raise AssertionError("pointer chain must be acyclic")
        visited.add(current)
        order.append(current)
        current = next_idx[current]
    if len(order) != len(rectangles):
        raise AssertionError("pointer chain must include all rectangles")
    if any(deg != 1 for idx, deg in enumerate(indegree) if idx != heads[0]):
        raise AssertionError("pointer chain must be unbranched")
    return order


def _rank_map(order: list[int]) -> dict[int, int]:
    return {idx: rank for rank, idx in enumerate(order)}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a marker-color pointer chain between hollow rectangles and stack full-canvas crosshairs by that chain order.

    Input structure:
    - A dominant background surrounds several separated hollow rectangles whose border colors are all distinct.
    - Each rectangle contains exactly one interior marker pixel; every non-tail marker color matches the border color of another rectangle.
    - The rectangles therefore form one unique branch-free, cycle-free pointer chain whose order does not follow size or placement.

    Rule:
    - Read the total chain by treating each rectangle's marker color as a pointer to the rectangle with that border color; the one marker color that matches no border identifies the tail.
    - For every rectangle, draw a full-width horizontal line and a full-height vertical line through its own center using its border color.
    - Draw those crosshairs from chain tail toward chain head so later rectangles in that tail-to-head process overwrite earlier crosshair intersections, then redraw the original rectangle borders and marker pixels on top.

    Key insight:
    - The interior marker is not a repaint instruction but a link to another rectangle, and that decoded chain is the only valid z-order for the overlapping centerlines.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    rectangles = _extract_rectangles(g)
    order = _decode_chain(rectangles)

    out = np.full_like(g, bg)
    for idx in reversed(order):
        rect = rectangles[idx]
        rr, cc = rect.center
        out[rr, :] = rect.color
        out[:, cc] = rect.color

    out[g != bg] = g[g != bg]
    return out


def _expanded_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = MIN_RECT_GAP) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _paint_rectangle(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _visible_intersections(grid: np.ndarray, rectangles: list[Rectangle]) -> list[tuple[int, int, int, int]]:
    bg = _get_bg(grid)
    points: list[tuple[int, int, int, int]] = []
    for i, rect_i in enumerate(rectangles):
        for j, rect_j in enumerate(rectangles):
            if i == j:
                continue
            r = rect_i.center[0]
            c = rect_j.center[1]
            if int(grid[r, c]) == bg:
                points.append((i, j, r, c))
    return points


def _shortcut_orders(rectangles: list[Rectangle]) -> list[list[int]]:
    orders: list[list[int]] = []
    orders.append(sorted(range(len(rectangles)), key=lambda idx: rectangles[idx].bbox))
    orders.append(sorted(range(len(rectangles)), key=lambda idx: rectangles[idx].bbox[::-1]))
    orders.append(sorted(range(len(rectangles)), key=lambda idx: rectangles[idx].center[0]))
    orders.append(sorted(range(len(rectangles)), key=lambda idx: rectangles[idx].center[1]))
    orders.append(sorted(range(len(rectangles)), key=lambda idx: (rectangles[idx].bbox[2] - rectangles[idx].bbox[0] + 1) * (rectangles[idx].bbox[3] - rectangles[idx].bbox[1] + 1)))
    orders.append(sorted(range(len(rectangles)), key=lambda idx: rectangles[idx].color))
    return orders


def _semantic_checks(grid: np.ndarray) -> None:
    rectangles = _extract_rectangles(grid)
    if len(rectangles) < 3:
        raise AssertionError("generator must create at least three rectangles")

    centers = [rect.center for rect in rectangles]
    if len({r for r, _ in centers}) != len(centers) or len({c for _, c in centers}) != len(centers):
        raise AssertionError("center rows and columns must be unique")

    order = _decode_chain(rectangles)
    order_rev = list(reversed(order))
    for candidate in _shortcut_orders(rectangles):
        if order == candidate or order_rev == candidate:
            raise AssertionError("chain order must not match a simple spatial, size, or color ordering")

    visible = _visible_intersections(grid, rectangles)
    if len(visible) < MIN_VISIBLE_INTERSECTIONS:
        raise AssertionError("too few visible crosshair intersections")

    out = main(grid)
    bg = _get_bg(grid)
    ranks = _rank_map(order)
    winning_colors: set[int] = set()
    for i, j, r, c in visible:
        winner = rectangles[i] if ranks[i] < ranks[j] else rectangles[j]
        if int(out[r, c]) != winner.color:
            raise AssertionError("intersection color does not follow the decoded chain order")
        winning_colors.add(winner.color)
        if int(grid[r, c]) != bg:
            raise AssertionError("visible intersection unexpectedly occupied in the input")
    if len(winning_colors) < 2:
        raise AssertionError("intersections should exhibit multiple chain layers")

    for rect in rectangles:
        mr, mc = rect.marker_pos
        if int(out[mr, mc]) != rect.marker_color:
            raise AssertionError("marker pixel must remain on the topmost layer")
        r0, c0, r1, c1 = rect.bbox
        if not np.all(out[r0, c0 : c1 + 1] == rect.color):
            raise AssertionError("top border must be restored above crosshairs")
        if not np.all(out[r1, c0 : c1 + 1] == rect.color):
            raise AssertionError("bottom border must be restored above crosshairs")
        if not np.all(out[r0 : r1 + 1, c0] == rect.color):
            raise AssertionError("left border must be restored above crosshairs")
        if not np.all(out[r0 : r1 + 1, c1] == rect.color):
            raise AssertionError("right border must be restored above crosshairs")


def _choose_shapes(rect_count: int) -> list[tuple[int, int]]:
    areas = list(np.random.choice(AREA_OPTIONS, size=rect_count, replace=False))
    np.random.shuffle(areas)
    shapes = [AREA_TO_SHAPES[int(area)][np.random.randint(len(AREA_TO_SHAPES[int(area)]))] for area in areas]
    return [(int(h), int(w)) for h, w in shapes]


def _sample_marker_pos(box: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    return (
        int(np.random.randint(r0 + 1, r1)),
        int(np.random.randint(c0 + 1, c1)),
    )


def _try_place_boxes(grid_shape: tuple[int, int], shapes: list[tuple[int, int]]) -> list[tuple[int, int, int, int]] | None:
    h, w = grid_shape
    placed: list[tuple[int, int, int, int]] = []
    used_center_rows: set[int] = set()
    used_center_cols: set[int] = set()
    order = list(np.argsort([shape[0] * shape[1] for shape in shapes]))[::-1]
    boxes: list[tuple[int, int, int, int] | None] = [None] * len(shapes)
    for shape_idx in order:
        rh, rw = shapes[shape_idx]
        candidates: list[tuple[int, int, int, int]] = []
        for r0 in range(1, h - rh):
            for c0 in range(1, w - rw):
                box = (r0, c0, r0 + rh - 1, c0 + rw - 1)
                center = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
                if center[0] in used_center_rows or center[1] in used_center_cols:
                    continue
                if any(_expanded_overlap(box, other) for other in placed):
                    continue
                candidates.append(box)
        if not candidates:
            return None
        choice = candidates[int(np.random.randint(len(candidates)))]
        boxes[shape_idx] = choice
        placed.append(choice)
        used_center_rows.add((choice[0] + choice[2]) // 2)
        used_center_cols.add((choice[1] + choice[3]) // 2)
    return [box for box in boxes if box is not None]


def _build_input() -> np.ndarray:
    bg = 0
    rect_count = int(np.random.choice(RECT_COUNT_OPTIONS))
    shapes = _choose_shapes(rect_count)
    min_h = max(GRID_MIN, max(shape[0] for shape in shapes) + rect_count * 2)
    min_w = max(GRID_MIN, max(shape[1] for shape in shapes) + rect_count * 2)
    grid_h = int(np.random.randint(min_h, GRID_MAX + 1))
    grid_w = int(np.random.randint(min_w, GRID_MAX + 1))
    boxes = _try_place_boxes((grid_h, grid_w), shapes)
    if boxes is None:
        raise AssertionError("failed to place rectangles")

    usable_colors = [color for color in PALETTE if color != bg]
    chosen = list(np.random.choice(usable_colors, size=rect_count + 1, replace=False))
    border_colors = [int(v) for v in chosen[:rect_count]]
    tail_marker_color = int(chosen[-1])

    chain_order = list(range(rect_count))
    np.random.shuffle(chain_order)

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    marker_positions = [_sample_marker_pos(box) for box in boxes]
    marker_colors = [tail_marker_color] * rect_count
    for pos, idx in enumerate(chain_order[:-1]):
        nxt = chain_order[pos + 1]
        marker_colors[idx] = border_colors[nxt]

    for idx, box in enumerate(boxes):
        _paint_rectangle(grid, box, border_colors[idx])
        mr, mc = marker_positions[idx]
        grid[mr, mc] = marker_colors[idx]

    _semantic_checks(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            return _build_input()
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a non-degenerate pointer-chain rectangle input")
