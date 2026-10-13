from __future__ import annotations

from collections import deque
import random

import numpy as np


_BG = 0
_WALL = 5
_PATTERN_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
_GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _most_common_color(values: np.ndarray) -> int:
    flat = np.asarray(values, dtype=int).ravel()
    uniq, cnt = np.unique(flat, return_counts=True)
    return int(uniq[int(np.argmax(cnt))])


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    h, w = mask.shape
    labels = np.full((h, w), -1, dtype=int)
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != -1:
                continue
            idx = len(components)
            q: deque[tuple[int, int]] = deque([(r, c)])
            labels[r, c] = idx
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == -1:
                        labels[nr, nc] = idx
                        q.append((int(nr), int(nc)))
            components.append(coords)

    return labels, components


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _door_kind(g: np.ndarray, r: int, c: int, wall: int) -> str | None:
    if int(g[r, c]) == wall:
        return None

    vertical_wall = (
        int(g[r - 1, c]) == wall
        and int(g[r + 1, c]) == wall
        and int(g[r, c - 1]) != wall
        and int(g[r, c + 1]) != wall
    )
    horizontal_wall = (
        int(g[r, c - 1]) == wall
        and int(g[r, c + 1]) == wall
        and int(g[r - 1, c]) != wall
        and int(g[r + 1, c]) != wall
    )

    if vertical_wall == horizontal_wall:
        return None
    return "vertical" if vertical_wall else "horizontal"


def _transform_patch(patch: np.ndarray, flip_ud: bool, flip_lr: bool) -> np.ndarray:
    out = patch
    if flip_ud:
        out = np.flipud(out)
    if flip_lr:
        out = np.fliplr(out)
    return np.array(out, copy=True)


def _paste_center(out: np.ndarray, room_bbox: tuple[int, int, int, int], patch: np.ndarray, bg: int) -> None:
    r0, c0, r1, c1 = room_bbox
    room_h = r1 - r0 + 1
    room_w = c1 - c0 + 1
    ph, pw = patch.shape
    assert ph <= room_h and pw <= room_w, "pattern must fit inside every room"

    top = r0 + (room_h - ph) // 2
    left = c0 + (room_w - pw) // 2
    mask = patch != bg
    out[top : top + ph, left : left + pw][mask] = patch[mask]


def _analyze_rooms(
    g: np.ndarray,
) -> tuple[
    int,
    int,
    list[tuple[int, int, int, int]],
    int,
    np.ndarray,
    list[list[tuple[int, str]]],
]:
    bg = _most_common_color(g)
    non_bg = g[g != bg]
    assert non_bg.size > 0, "expected wall pixels"
    wall = _most_common_color(non_bg)

    h, w = g.shape
    doors: list[tuple[int, int, str]] = []
    closed = g.copy()
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            kind = _door_kind(g, r, c, wall)
            if kind is None:
                continue
            doors.append((r, c, kind))
            closed[r, c] = wall

    labels, comps = _label_components(closed != wall)
    room_label_to_id: dict[int, int] = {}
    rooms: list[tuple[int, int, int, int]] = []
    for label, coords in enumerate(comps):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords):
            continue
        box = _bbox(coords)
        r0, c0, r1, c1 = box
        if (r1 - r0 + 1) * (c1 - c0 + 1) != len(coords):
            continue
        room_label_to_id[label] = len(rooms)
        rooms.append(box)

    assert rooms, "expected enclosed rectangular rooms"

    pattern_mask = (g != bg) & (g != wall)
    pattern_coords = np.argwhere(pattern_mask)
    assert pattern_coords.size > 0, "expected one source pattern"

    source_ids = {
        room_label_to_id[int(labels[r, c])]
        for r, c in pattern_coords
        if int(labels[r, c]) in room_label_to_id
    }
    assert len(source_ids) == 1, "pattern must lie in exactly one room"
    source_room = next(iter(source_ids))

    pr0, pc0 = map(int, pattern_coords.min(axis=0))
    pr1, pc1 = map(int, pattern_coords.max(axis=0))
    pattern = np.array(g[pr0 : pr1 + 1, pc0 : pc1 + 1], copy=True)

    adjacency: list[list[tuple[int, str]]] = [[] for _ in rooms]
    seen_pairs: set[tuple[int, int]] = set()
    for r, c, kind in doors:
        if kind == "vertical":
            left_label = int(labels[r, c - 1])
            right_label = int(labels[r, c + 1])
            flip = "lr"
        else:
            left_label = int(labels[r - 1, c])
            right_label = int(labels[r + 1, c])
            flip = "ud"
        if left_label not in room_label_to_id or right_label not in room_label_to_id:
            continue
        a = room_label_to_id[left_label]
        b = room_label_to_id[right_label]
        if a == b:
            continue
        key = (min(a, b), max(a, b))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        adjacency[a].append((b, flip))
        adjacency[b].append((a, flip))

    return bg, wall, rooms, source_room, pattern, adjacency


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate the source-room motif through the room graph.

    Crossing a door in a horizontal wall flips the motif vertically.
    Crossing a door in a vertical wall flips the motif horizontally.
    """
    g = np.asarray(grid, dtype=int)
    bg, _wall, rooms, source_room, pattern, adjacency = _analyze_rooms(g)

    transforms: dict[int, tuple[bool, bool]] = {source_room: (False, False)}
    distance: dict[int, int] = {source_room: 0}
    q: deque[int] = deque([source_room])

    while q:
        room = q.popleft()
        flip_ud, flip_lr = transforms[room]
        for nxt, axis in adjacency[room]:
            next_state = (
                flip_ud ^ (axis == "ud"),
                flip_lr ^ (axis == "lr"),
            )
            next_dist = distance[room] + 1
            if nxt not in transforms:
                transforms[nxt] = next_state
                distance[nxt] = next_dist
                q.append(nxt)
            elif distance[nxt] == next_dist:
                assert transforms[nxt] == next_state, "shortest-path transforms must agree"

    out = np.array(g, copy=True)
    for room_id, state in transforms.items():
        motif = _transform_patch(pattern, state[0], state[1])
        _paste_center(out, rooms[room_id], motif, bg)
    return out


def _normalize_cells(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return [(r - min_r, c - min_c) for r, c in cells]


def _grow_tree_layout(rows: int, cols: int, room_count: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    start = (random.randrange(rows), random.randrange(cols))
    occupied = {start}
    edges_by_cell: list[tuple[tuple[int, int], tuple[int, int]]] = []

    while len(occupied) < room_count:
        candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for r in range(rows):
            for c in range(cols):
                cell = (r, c)
                if cell in occupied:
                    continue
                nbrs = [nbr for nbr in _neighbors4(r, c) if nbr in occupied]
                if len(nbrs) == 1:
                    candidates.append((cell, nbrs[0]))
        if not candidates:
            return None
        cell, parent = random.choice(candidates)
        occupied.add(cell)
        edges_by_cell.append((cell, parent))

    normalized_cells = _normalize_cells(sorted(occupied))
    cell_map = {old: new for old, new in zip(sorted(occupied), normalized_cells, strict=False)}
    normalized_edges = [(cell_map[a], cell_map[b]) for a, b in edges_by_cell]

    cells = sorted(set(normalized_cells))
    index = {cell: i for i, cell in enumerate(cells)}
    edges = [(index[a], index[b]) for a, b in normalized_edges]
    return cells, edges


def _sample_layout() -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    mode = random.choice(("mixed", "hchain", "vchain"))
    if mode == "hchain":
        room_count = random.randint(3, 5)
        cells = [(0, c) for c in range(room_count)]
        edges = [(i - 1, i) for i in range(1, room_count)]
        return cells, edges
    if mode == "vchain":
        room_count = random.randint(3, 5)
        cells = [(r, 0) for r in range(room_count)]
        edges = [(i - 1, i) for i in range(1, room_count)]
        return cells, edges

    for _ in range(100):
        rows = random.randint(2, 4)
        cols = random.randint(2, 4)
        room_count = random.randint(4, min(7, rows * cols))
        grown = _grow_tree_layout(rows, cols, room_count)
        if grown is None:
            continue
        cells, edges = grown
        has_horizontal = False
        has_vertical = False
        for a, b in edges:
            ar, ac = cells[a]
            br, bc = cells[b]
            if ar == br:
                has_horizontal = True
            else:
                has_vertical = True
        if has_horizontal and has_vertical:
            return cells, edges
    return None


def _random_pattern(max_h: int, max_w: int) -> np.ndarray | None:
    if max_h < 2 or max_w < 2:
        return None

    for _ in range(200):
        h = random.randint(2, max_h)
        w = random.randint(2, max_w)
        area = random.randint(3, min(h * w - 1, 6))
        start = (random.randrange(h), random.randrange(w))
        cells = {start}

        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))

        if len(cells) != area:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        mask = mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if np.array_equal(mask, np.flipud(mask)):
            continue
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        return mask
    return None


def _draw_room_walls(grid: np.ndarray, bbox: tuple[int, int, int, int], wall: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 - 1, c0 - 1 : c1 + 2] = wall
    grid[r1 + 1, c0 - 1 : c1 + 2] = wall
    grid[r0 - 1 : r1 + 2, c0 - 1] = wall
    grid[r0 - 1 : r1 + 2, c1 + 1] = wall


def _room_bboxes(
    cells: list[tuple[int, int]],
    room_h: int,
    room_w: int,
    top_margin: int,
    left_margin: int,
) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    for r, c in cells:
        top = top_margin + 1 + r * (room_h + 1)
        left = left_margin + 1 + c * (room_w + 1)
        boxes.append((top, left, top + room_h - 1, left + room_w - 1))
    return boxes


def _edge_transforms(
    room_count: int,
    edges: list[tuple[int, int]],
    cells: list[tuple[int, int]],
    source_room: int,
) -> dict[int, tuple[bool, bool]]:
    adjacency: list[list[tuple[int, str]]] = [[] for _ in range(room_count)]
    for a, b in edges:
        ar, ac = cells[a]
        br, bc = cells[b]
        flip = "lr" if ar == br else "ud"
        adjacency[a].append((b, flip))
        adjacency[b].append((a, flip))

    state: dict[int, tuple[bool, bool]] = {source_room: (False, False)}
    q: deque[int] = deque([source_room])
    while q:
        cur = q.popleft()
        cur_ud, cur_lr = state[cur]
        for nxt, axis in adjacency[cur]:
            nxt_state = (cur_ud ^ (axis == "ud"), cur_lr ^ (axis == "lr"))
            if nxt not in state:
                state[nxt] = nxt_state
                q.append(nxt)
    return state


def _render_case(
    cells: list[tuple[int, int]],
    edges: list[tuple[int, int]],
    room_h: int,
    room_w: int,
    pattern: np.ndarray,
    color: int,
    source_room: int,
) -> tuple[np.ndarray, np.ndarray]:
    top_margin = random.randint(1, 2)
    bottom_margin = random.randint(1, 2)
    left_margin = random.randint(1, 2)
    right_margin = random.randint(1, 2)

    max_r = max(r for r, _ in cells) + 1
    max_c = max(c for _, c in cells) + 1
    height = top_margin + max_r * (room_h + 1) + 1 + bottom_margin
    width = left_margin + max_c * (room_w + 1) + 1 + right_margin

    base = np.full((height, width), _BG, dtype=int)
    boxes = _room_bboxes(cells, room_h, room_w, top_margin, left_margin)
    for box in boxes:
        _draw_room_walls(base, box, _WALL)

    for a, b in edges:
        ar, ac = cells[a]
        br, bc = cells[b]
        ra0, ca0, ra1, ca1 = boxes[a]
        rb0, cb0, rb1, cb1 = boxes[b]
        if ar == br:
            wall_col = ca1 + 1 if ac < bc else cb1 + 1
            top = max(ra0, rb0)
            bottom = min(ra1, rb1)
            door_row = random.randint(top, bottom)
            base[door_row, wall_col] = _BG
        else:
            wall_row = ra1 + 1 if ar < br else rb1 + 1
            left = max(ca0, cb0)
            right = min(ca1, cb1)
            door_col = random.randint(left, right)
            base[wall_row, door_col] = _BG

    input_grid = np.array(base, copy=True)
    output_grid = np.array(base, copy=True)

    source_patch = np.full(pattern.shape, _BG, dtype=int)
    source_patch[pattern] = color
    _paste_center(input_grid, boxes[source_room], source_patch, _BG)

    transforms = _edge_transforms(len(cells), edges, cells, source_room)
    for room_id, (flip_ud, flip_lr) in transforms.items():
        motif = _transform_patch(source_patch, flip_ud, flip_lr)
        _paste_center(output_grid, boxes[room_id], motif, _BG)

    return input_grid, output_grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        layout = _sample_layout()
        if layout is None:
            continue
        cells, edges = layout
        room_h = random.randint(4, 5)
        room_w = random.randint(4, 5)
        pattern = _random_pattern(min(4, room_h - 2), min(4, room_w - 2))
        if pattern is None:
            continue
        color = random.choice(_PATTERN_COLORS)
        source_room = random.randrange(len(cells))
        inp, expected = _render_case(cells, edges, room_h, room_w, pattern, color, source_room)
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid room-propagation puzzle")
