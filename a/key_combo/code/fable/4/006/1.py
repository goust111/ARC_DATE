from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
WALL = 1
DIVIDER = 5
START = 6
PAINT_COLORS = [2, 3, 4, 7, 8, 9]
GEN_RETRIES = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    components: list[list[tuple[int, int]]] = []
    label = 0
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if labels[sr, sc]:
            continue
        label += 1
        q = deque([(sr, sc)])
        labels[sr, sc] = label
        coords: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            coords.append((r, c))
            for nr, nc in _neighbors4(r, c):
                if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and labels[nr, nc] == 0:
                    labels[nr, nc] = label
                    q.append((nr, nc))
        components.append(coords)
    return labels, components


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return _label_components(mask)[1]


def _count_components(mask: np.ndarray) -> int:
    return len(_components(mask))


def _detect_layout(g: np.ndarray) -> tuple[list[int], np.ndarray, np.ndarray]:
    h, w = g.shape
    divider_cols = [c for c in range(w) if np.all(g[:, c] == DIVIDER)]
    assert len(divider_cols) == 1, "expected one full-height gray divider column"
    div_c = divider_cols[0]

    candidates = []
    for side in (-1, 1):
        legend_c = div_c + side
        if not (0 <= legend_c < w):
            continue
        rows = [
            r
            for r in range(h)
            if int(g[r, legend_c]) not in (BG, WALL, DIVIDER, START)
        ]
        if not rows:
            continue
        r0, r1 = min(rows), max(rows)
        if rows != list(range(r0, r1 + 1)):
            continue
        if side == -1:
            field_cols = range(div_c + 1, w)
        else:
            field_cols = range(0, div_c)
        if not field_cols:
            continue
        legend = [int(g[r, legend_c]) for r in rows]
        candidates.append((side, legend, field_cols))

    assert len(candidates) == 1, "expected one single-column legend beside the divider"
    _side, legend, field_cols = candidates[0]

    field_mask = np.zeros_like(g, dtype=bool)
    field_mask[:, list(field_cols)] = True
    protected = ~field_mask
    return legend, field_mask, protected


def _structural_walls(g: np.ndarray, field_mask: np.ndarray) -> np.ndarray:
    blue = field_mask & (g == WALL)
    blue_components = _components(blue)
    base_passable = field_mask & ~blue
    base_count = _count_components(base_passable)

    structural = np.zeros_like(g, dtype=bool)
    for comp in blue_components:
        comp_mask = np.zeros_like(g, dtype=bool)
        for r, c in comp:
            comp_mask[r, c] = True
        if _count_components(base_passable | comp_mask) < base_count:
            structural |= comp_mask
    return structural


def _room_graph(labels: np.ndarray, room_count: int, structural: np.ndarray) -> list[set[int]]:
    graph = [set() for _ in range(room_count + 1)]
    for wall in _components(structural):
        adjacent_rooms: set[int] = set()
        for r, c in wall:
            for nr, nc in _neighbors4(r, c):
                if _in_bounds(labels.shape, nr, nc) and labels[nr, nc] > 0:
                    adjacent_rooms.add(int(labels[nr, nc]))
        rooms = sorted(adjacent_rooms)
        for i, a in enumerate(rooms):
            for b in rooms[i + 1 :]:
                graph[a].add(b)
                graph[b].add(a)
    return graph


def _depths_from_room(
    labels: np.ndarray,
    room_count: int,
    structural: np.ndarray,
    start_room: int,
) -> dict[int, int]:
    graph = _room_graph(labels, room_count, structural)
    depths = {int(start_room): 0}
    q = deque([int(start_room)])
    while q:
        room = q.popleft()
        for nxt in graph[room]:
            if nxt not in depths:
                depths[nxt] = depths[room] + 1
                q.append(nxt)
    return depths


def _analyze(g: np.ndarray, field_mask: np.ndarray):
    structural = _structural_walls(g, field_mask)
    passable = field_mask & ~structural
    labels, rooms = _label_components(passable)
    return structural, labels, rooms


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor every room by its minimum number of structural wall crossings from the start room.

    Input structure:
    - A black background field is split into rooms by blue wall components.
    - One magenta singleton lies inside the start room.
    - A corner single-column legend is separated from the room field by a full gray divider column.
    - Extra blue fragments may sit inside rooms without changing room connectivity.

    Rule:
    - Treat a blue connected component as a structural wall only when removing that component
      would merge two or more room components; all other blue components are erased.
    - Build the room adjacency graph by connecting rooms that touch the same structural wall.
    - Starting from the magenta room, assign each room the fewest crossed structural walls.
    - Fill background cells and erased-decoration cells in each room with the legend color at
      that depth, while preserving structural walls, the legend, the divider, and the start pixel.

    Key insight:
    - The legend indexes topological crossing depth through real separating walls, not geometric
      direction or distance; decorative blue fragments must be ignored before the room depths are read.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    legend, field_mask, _protected = _detect_layout(g)

    starts = np.argwhere(g == START)
    assert starts.shape == (1, 2), "expected one magenta start pixel"
    start = (int(starts[0, 0]), int(starts[0, 1]))
    assert field_mask[start], "start must be in the room field"

    structural, labels, rooms = _analyze(g, field_mask)
    assert structural.any(), "expected at least one structural wall"
    room_count = len(rooms)
    start_room = int(labels[start])
    assert start_room > 0, "start must lie in a room"

    depths = _depths_from_room(labels, room_count, structural, start_room)
    assert len(depths) == room_count, "all rooms must be reachable by crossing structural walls"
    assert max(depths.values()) < len(legend), "legend must cover every crossing depth"

    out = g.copy()
    decoration = field_mask & (g == WALL) & ~structural
    for room_id, depth in depths.items():
        room_mask = labels == room_id
        fill_mask = room_mask & ((g == BG) | decoration)
        out[fill_mask] = int(legend[depth])

    out[start] = START
    out[structural] = WALL
    return out


def _draw_frame(mask: np.ndarray, box: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = box
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True


def _box_h(box: tuple[int, int, int, int]) -> int:
    return box[2] - box[0] + 1


def _box_w(box: tuple[int, int, int, int]) -> int:
    return box[3] - box[1] + 1


def _expand_box(box: tuple[int, int, int, int], pad: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return r0 - pad, c0 - pad, r1 + pad, c1 + pad


def _boxes_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _gen_chain_boxes(fh: int, fw: int) -> list[tuple[int, int, int, int]] | None:
    max_depth = min(5, (fh - 2) // 4, (fw - 2) // 4)
    if max_depth < 2:
        return None
    depth = random.randint(2, max_depth)

    cur_h = random.randint(4, 6)
    cur_w = random.randint(4, 6)
    margins: list[tuple[int, int, int, int]] = []
    for _ in range(depth - 1):
        mt, mb = random.randint(2, 4), random.randint(2, 4)
        ml, mr = random.randint(2, 4), random.randint(2, 4)
        margins.append((mt, mb, ml, mr))
        cur_h += mt + mb
        cur_w += ml + mr

    if cur_h > fh - 2 or cur_w > fw - 2:
        return None
    r0 = random.randint(1, fh - cur_h - 1)
    c0 = random.randint(1, fw - cur_w - 1)
    current = (r0, c0, r0 + cur_h - 1, c0 + cur_w - 1)
    boxes = [current]
    for mt, mb, ml, mr in reversed(margins):
        pr0, pc0, pr1, pc1 = current
        current = (pr0 + mt, pc0 + ml, pr1 - mb, pc1 - mr)
        boxes.append(current)
    return boxes


def _gen_branch_boxes(fh: int, fw: int) -> list[tuple[int, int, int, int]] | None:
    root_count = random.randint(2, 3)
    boxes: list[tuple[int, int, int, int]] = []
    for _ in range(root_count):
        placed = False
        for _try in range(120):
            max_h = min(11, fh - 3)
            max_w = min(11, fw - 3)
            if max_h < 5 or max_w < 5:
                return None
            bh = random.randint(5, max_h)
            bw = random.randint(5, max_w)
            r0 = random.randint(1, fh - bh - 1)
            c0 = random.randint(1, fw - bw - 1)
            box = (r0, c0, r0 + bh - 1, c0 + bw - 1)
            padded = _expand_box(box, 2)
            if all(not _boxes_intersect(padded, _expand_box(other, 1)) for other in boxes):
                boxes.append(box)
                placed = True
                break
        if not placed:
            return None

    nestable = [box for box in boxes if _box_h(box) >= 9 and _box_w(box) >= 9]
    random.shuffle(nestable)
    nested_count = random.randint(1, min(2, len(nestable))) if nestable else 0
    for parent in nestable[:nested_count]:
        pr0, pc0, pr1, pc1 = parent
        max_h = _box_h(parent) - 4
        max_w = _box_w(parent) - 4
        if max_h < 4 or max_w < 4:
            continue
        ch = random.randint(4, min(7, max_h))
        cw = random.randint(4, min(7, max_w))
        r0 = random.randint(pr0 + 2, pr1 - ch - 1)
        c0 = random.randint(pc0 + 2, pc1 - cw - 1)
        boxes.append((r0, c0, r0 + ch - 1, c0 + cw - 1))
    return boxes


def _make_structural_field(fh: int, fw: int) -> tuple[np.ndarray, list[tuple[int, int, int, int]]] | None:
    mode = random.choice(["chain", "branch", "branch", "chain"])
    boxes = _gen_chain_boxes(fh, fw) if mode == "chain" else _gen_branch_boxes(fh, fw)
    if not boxes or len(boxes) < 2:
        return None
    structural = np.zeros((fh, fw), dtype=bool)
    for box in boxes:
        _draw_frame(structural, box)
    return structural, boxes


def _graph_for_structural(structural: np.ndarray):
    passable = ~structural
    labels, rooms = _label_components(passable)
    return labels, rooms


def _choose_start(structural: np.ndarray) -> tuple[tuple[int, int], dict[int, int], np.ndarray] | None:
    labels, rooms = _graph_for_structural(structural)
    room_ids = list(range(1, len(rooms) + 1))
    random.shuffle(room_ids)
    for room_id in room_ids:
        depths = _depths_from_room(labels, len(rooms), structural, room_id)
        if len(depths) != len(rooms):
            continue
        max_depth = max(depths.values())
        if not (2 <= max_depth < len(PAINT_COLORS)):
            continue
        cells = [(r, c) for r, c in rooms[room_id - 1] if len(rooms[room_id - 1]) >= 4]
        if not cells:
            continue
        return random.choice(cells), depths, labels
    return None


def _far_from_mask(mask: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    h, w = mask.shape
    for r, c in cells:
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                if mask[rr, cc]:
                    return False
    return True


def _add_decorations(
    field: np.ndarray,
    structural: np.ndarray,
    labels: np.ndarray,
    start: tuple[int, int],
) -> np.ndarray | None:
    fh, fw = field.shape
    base_count = _count_components(~structural)
    decor = np.zeros((fh, fw), dtype=bool)
    target = random.randint(1, 4)
    shapes = [
        [(0, 0)],
        [(0, 0), (0, 1)],
        [(0, 0), (1, 0)],
        [(0, 0), (0, 1), (1, 0)],
        [(0, 0), (1, 0), (1, 1)],
    ]

    for _ in range(target):
        placed = False
        random.shuffle(shapes)
        for _try in range(160):
            shape = random.choice(shapes)
            r0 = random.randrange(fh)
            c0 = random.randrange(fw)
            cells = [(r0 + dr, c0 + dc) for dr, dc in shape]
            if any(not _in_bounds(field.shape, r, c) for r, c in cells):
                continue
            if any((r, c) == start for r, c in cells):
                continue
            if any(structural[r, c] or decor[r, c] or labels[r, c] == 0 for r, c in cells):
                continue
            if not _far_from_mask(structural | decor, cells):
                continue
            trial = decor.copy()
            for r, c in cells:
                trial[r, c] = True
            if _count_components(~(structural | trial)) != base_count:
                continue
            for r, c in cells:
                decor[r, c] = True
                field[r, c] = WALL
            placed = True
            break
        if not placed and not decor.any():
            return None
    return decor


def _embed_field(
    field: np.ndarray,
    legend: list[int],
    legend_at_left: bool,
    legend_at_top: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fh, fw = field.shape
    grid = np.full((fh, fw + 2), BG, dtype=int)
    if legend_at_left:
        grid[:, 1] = DIVIDER
        grid[:, 2:] = field
        field_mask = np.zeros_like(grid, dtype=bool)
        field_mask[:, 2:] = True
        legend_c = 0
    else:
        grid[:, fw] = DIVIDER
        grid[:, :fw] = field
        field_mask = np.zeros_like(grid, dtype=bool)
        field_mask[:, :fw] = True
        legend_c = fw + 1

    r0 = 0 if legend_at_top else fh - len(legend)
    for i, color in enumerate(legend):
        grid[r0 + i, legend_c] = int(color)
    protected = ~field_mask
    return grid, field_mask, protected


def _expected_grid(
    grid: np.ndarray,
    field_mask: np.ndarray,
    structural_field: np.ndarray,
    labels_field: np.ndarray,
    depths: dict[int, int],
    legend: list[int],
    legend_at_left: bool,
) -> np.ndarray:
    out = grid.copy()
    if legend_at_left:
        c_slice = slice(2, None)
    else:
        c_slice = slice(0, structural_field.shape[1])
    structural_full = np.zeros_like(grid, dtype=bool)
    structural_full[:, c_slice] = structural_field
    labels_full = np.zeros_like(grid, dtype=int)
    labels_full[:, c_slice] = labels_field

    decoration = field_mask & (grid == WALL) & ~structural_full
    for room_id, depth in depths.items():
        room = labels_full == room_id
        out[room & ((grid == BG) | decoration)] = int(legend[depth])
    out[grid == START] = START
    out[structural_full] = WALL
    return out


def _semantic_check(
    inp: np.ndarray,
    expected: np.ndarray,
    structural_field: np.ndarray,
    decor_field: np.ndarray,
    legend: list[int],
    depths: dict[int, int],
    legend_at_left: bool,
) -> bool:
    try:
        out = main(inp)
        detected_legend, field_mask, _protected = _detect_layout(inp)
        detected_structural, _labels, _rooms = _analyze(inp, field_mask)
    except Exception:
        return False
    if detected_legend != legend or not np.array_equal(out, expected):
        return False
    if max(inp.shape) > 30 or max(out.shape) > 30:
        return False
    if legend_at_left:
        struct_slice = detected_structural[:, 2:]
        decor_full = np.zeros_like(inp, dtype=bool)
        decor_full[:, 2:] = decor_field
    else:
        fw = structural_field.shape[1]
        struct_slice = detected_structural[:, :fw]
        decor_full = np.zeros_like(inp, dtype=bool)
        decor_full[:, :fw] = decor_field
    if not np.array_equal(struct_slice, structural_field):
        return False
    if not decor_field.any() or np.any(out[decor_full] == WALL):
        return False
    if set(depths.values()) != set(range(max(depths.values()) + 1)):
        return False
    if len(set(legend)) != len(legend):
        return False
    if len(legend) < 3:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    fh = random.randint(15, 28)
    fw = random.randint(15, 28)
    made = _make_structural_field(fh, fw)
    if made is None:
        return None
    structural, _boxes = made
    chosen = _choose_start(structural)
    if chosen is None:
        return None
    start, depths, labels = chosen
    max_depth = max(depths.values())
    legend = random.sample(PAINT_COLORS, max_depth + 1)

    field = np.full((fh, fw), BG, dtype=int)
    field[structural] = WALL
    field[start] = START
    decor = _add_decorations(field, structural, labels, start)
    if decor is None:
        return None

    legend_at_left = random.random() < 0.5
    legend_at_top = random.random() < 0.5
    inp, field_mask, _protected = _embed_field(field, legend, legend_at_left, legend_at_top)
    expected = _expected_grid(
        inp,
        field_mask,
        structural,
        labels,
        depths,
        legend,
        legend_at_left,
    )
    if _semantic_check(inp, expected, structural, decor, legend, depths, legend_at_left):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid crossing-depth legend input")
