import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREY = 5
MAROON = 9
SEQ_COLORS = (3, 4, 6, 7, 8)
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _rotate_right(seq: list[int], steps: int) -> list[int]:
    if not seq:
        return []
    k = steps % len(seq)
    if k == 0:
        return list(seq)
    return list(seq[-k:] + seq[:-k])


def _node_border_coords(top: int, left: int, height: int, width: int) -> set[tuple[int, int]]:
    coords: set[tuple[int, int]] = set()
    for r in range(top, top + height):
        for c in range(left, left + width):
            if r in (top, top + height - 1) or c in (left, left + width - 1):
                coords.add((r, c))
    return coords


def _node_interior_coords(top: int, left: int, orient: str, length: int) -> list[tuple[int, int]]:
    if orient == "h":
        return [(top + 1, left + 1 + i) for i in range(length)]
    return [(top + 1 + i, left + 1) for i in range(length)]


def _detect_nodes(grid: np.ndarray) -> list[dict]:
    height, width = map(int, grid.shape)
    used = np.zeros_like(grid, dtype=bool)
    nodes: list[dict] = []

    def add_node(top: int, left: int, orient: str, length: int, marker_pos: tuple[int, int]) -> None:
        node_height, node_width = (3, length + 2) if orient == "h" else (length + 2, 3)
        interior = _node_interior_coords(top, left, orient, length)
        seq = [int(grid[r, c]) for r, c in interior]
        nodes.append(
            {
                "top": int(top),
                "left": int(left),
                "height": int(node_height),
                "width": int(node_width),
                "orient": orient,
                "length": int(length),
                "marker": int(grid[marker_pos]),
                "marker_pos": tuple(map(int, marker_pos)),
                "seq": seq,
                "border": _node_border_coords(top, left, node_height, node_width),
                "interior": interior,
            }
        )
        used[top : top + node_height, left : left + node_width] = True

    def valid_horizontal(top: int, left: int, length: int) -> tuple[bool, tuple[int, int] | None]:
        patch = grid[top : top + 3, left : left + length + 2]
        if patch.shape != (3, length + 2):
            return False, None
        corners = [(0, 0), (0, length + 1), (2, 0), (2, length + 1)]
        marker_cells = [(r, c) for r, c in corners if int(patch[r, c]) in (RED, BLUE)]
        if len(marker_cells) != 1:
            return False, None
        for r, c in corners:
            v = int(patch[r, c])
            if (r, c) == marker_cells[0]:
                continue
            if v != GREY:
                return False, None
        for c in range(1, length + 1):
            if int(patch[0, c]) != GREY or int(patch[2, c]) != GREY:
                return False, None
        if int(patch[1, 0]) != GREY or int(patch[1, length + 1]) != GREY:
            return False, None
        seq = [int(v) for v in patch[1, 1 : length + 1]]
        if any(v in (BG, BLUE, RED, GREY, MAROON) for v in seq):
            return False, None
        if len(set(seq)) != length:
            return False, None
        return True, (top + marker_cells[0][0], left + marker_cells[0][1])

    def valid_vertical(top: int, left: int, length: int) -> tuple[bool, tuple[int, int] | None]:
        patch = grid[top : top + length + 2, left : left + 3]
        if patch.shape != (length + 2, 3):
            return False, None
        corners = [(0, 0), (0, 2), (length + 1, 0), (length + 1, 2)]
        marker_cells = [(r, c) for r, c in corners if int(patch[r, c]) in (RED, BLUE)]
        if len(marker_cells) != 1:
            return False, None
        for r, c in corners:
            v = int(patch[r, c])
            if (r, c) == marker_cells[0]:
                continue
            if v != GREY:
                return False, None
        for r in range(1, length + 1):
            if int(patch[r, 0]) != GREY or int(patch[r, 2]) != GREY:
                return False, None
        if int(patch[0, 1]) != GREY or int(patch[length + 1, 1]) != GREY:
            return False, None
        seq = [int(v) for v in patch[1 : length + 1, 1]]
        if any(v in (BG, BLUE, RED, GREY, MAROON) for v in seq):
            return False, None
        if len(set(seq)) != length:
            return False, None
        return True, (top + marker_cells[0][0], left + marker_cells[0][1])

    for top in range(height):
        for left in range(width):
            if used[top, left]:
                continue
            matched = False
            for length in range(5, 2, -1):
                ok, marker_pos = valid_horizontal(top, left, length)
                if ok:
                    add_node(top, left, "h", length, marker_pos)
                    matched = True
                    break
                ok, marker_pos = valid_vertical(top, left, length)
                if ok:
                    add_node(top, left, "v", length, marker_pos)
                    matched = True
                    break
            if matched:
                continue
    return nodes


def _collect_component(mask: np.ndarray, start: tuple[int, int], seen: set[tuple[int, int]]) -> list[tuple[int, int]]:
    height, width = map(int, mask.shape)
    queue = deque([start])
    seen.add(start)
    comp: list[tuple[int, int]] = []
    while queue:
        r, c = queue.popleft()
        comp.append((r, c))
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < height and 0 <= nc < width and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return comp


def _trace_path(component: set[tuple[int, int]], start: tuple[int, int]) -> list[tuple[int, int]]:
    order = [start]
    prev: tuple[int, int] | None = None
    cur = start
    while True:
        nxt = [p for p in _neighbors4(cur[0], cur[1]) if p in component and p != prev]
        if not nxt:
            break
        if len(nxt) != 1:
            raise AssertionError("path component must be a simple orthogonal chain")
        prev, cur = cur, nxt[0]
        order.append(cur)
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy each source color sequence into its connected target, rotating by the path's turn count.

    Input structure:
    - The background is black.
    - Small grey one-cell-thick rectangular nodes contain a horizontal or vertical sequence of 3-5 distinct colors.
    - Source nodes have one red corner on the frame; target nodes have one blue corner.
    - Each target is connected to exactly one source by a one-cell-wide maroon orthogonal path.

    Rule:
    - For each maroon path, identify the blue-corner target node and the red-corner source node it connects.
    - Trace the path and count how many times its direction changes by 90 degrees.
    - Copy the source node's internal color sequence into the target node after cyclically rotating that sequence to the right by the turn count.
    - Keep the node frames, source contents, and maroon paths unchanged.

    Key insight:
    - The path length is irrelevant; only the number of orthogonal bends determines the sequence shift.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    nodes = _detect_nodes(g)

    border_owner: dict[tuple[int, int], int] = {}
    for idx, node in enumerate(nodes):
        for cell in node["border"]:
            border_owner[cell] = idx

    path_mask = g == MAROON
    seen: set[tuple[int, int]] = set()
    components: list[list[tuple[int, int]]] = []
    for r, c in zip(*np.where(path_mask)):
        cell = (int(r), int(c))
        if cell not in seen:
            components.append(_collect_component(path_mask, cell, seen))

    for comp in components:
        comp_set = set(comp)
        adjacent: dict[int, set[tuple[int, int]]] = {}
        for r, c in comp:
            for nr, nc in _neighbors4(r, c):
                owner = border_owner.get((nr, nc))
                if owner is not None:
                    adjacent.setdefault(owner, set()).add((r, c))

        if len(adjacent) != 2:
            continue
        source_ids = [idx for idx in adjacent if nodes[idx]["marker"] == RED]
        target_ids = [idx for idx in adjacent if nodes[idx]["marker"] == BLUE]
        if len(source_ids) != 1 or len(target_ids) != 1:
            continue

        source = nodes[source_ids[0]]
        target = nodes[target_ids[0]]
        if source["orient"] != target["orient"] or source["length"] != target["length"]:
            continue

        target_path_cells = adjacent[target_ids[0]]
        if len(target_path_cells) != 1:
            raise AssertionError("target must touch the path at exactly one point")
        ordered = _trace_path(comp_set, next(iter(target_path_cells)))
        directions = [
            (ordered[i + 1][0] - ordered[i][0], ordered[i + 1][1] - ordered[i][1])
            for i in range(len(ordered) - 1)
        ]
        turns = sum(directions[i] != directions[i - 1] for i in range(1, len(directions)))
        rotated = _rotate_right(source["seq"], turns)
        for (r, c), color in zip(target["interior"], rotated):
            out[r, c] = int(color)

    return out


def _render_horizontal_node(length: int, seq: list[int], marker_color: int, marker_corner: tuple[int, int]) -> np.ndarray:
    node = np.full((3, length + 2), GREY, dtype=int)
    node[1, 1 : length + 1] = np.array(seq, dtype=int)
    node[marker_corner] = int(marker_color)
    return node


def _random_sequence(length: int) -> list[int]:
    return random.sample(SEQ_COLORS, length)


def _random_target_sequence(length: int, forbidden: list[int]) -> list[int]:
    while True:
        seq = _random_sequence(length)
        if seq != forbidden:
            return seq


def _path_cells_from_points(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for idx, (a, b) in enumerate(zip(points, points[1:])):
        (r0, c0), (r1, c1) = a, b
        if r0 == r1:
            step = 1 if c1 > c0 else -1
            segment = [(r0, c) for c in range(c0, c1 + step, step)]
        else:
            step = 1 if r1 > r0 else -1
            segment = [(r, c0) for r in range(r0, r1 + step, step)]
        if idx:
            segment = segment[1:]
        cells.extend(segment)
    return cells


def _build_path(turns: int) -> list[tuple[int, int]]:
    if turns == 0:
        a = random.randint(3, 6)
        points = [(0, 0), (0, a)]
    elif turns == 1:
        a = random.randint(2, 5)
        b = random.randint(2, 5)
        points = [(0, 0), (0, a), (b, a)]
    elif turns == 2:
        a = random.randint(2, 5)
        b = random.randint(2, 4)
        c = random.randint(2, 5)
        points = [(0, 0), (0, a), (b, a), (b, a + c)]
    elif turns == 3:
        a = random.randint(2, 4)
        b = random.randint(2, 4)
        c = random.randint(2, 4)
        d = random.randint(2, 4)
        points = [(0, 0), (0, a), (b, a), (b, a + c), (b + d, a + c)]
    else:
        a = random.randint(2, 4)
        b = random.randint(2, 4)
        c = random.randint(2, 4)
        d = random.randint(2, 4)
        e = random.randint(2, 4)
        points = [(0, 0), (0, a), (b, a), (b, a + c), (b + d, a + c), (b + d, a + c + e)]
    return _path_cells_from_points(points)


def _nonzero_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.where(grid != BG)
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _crop_nonzero(grid: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _nonzero_bbox(grid)
    return grid[r0 : r1 + 1, c0 : c1 + 1]


def _build_pair_patch() -> np.ndarray:
    length = random.randint(3, 5)
    turns = random.randint(0, 4)
    source_seq = _random_sequence(length)
    shifted = _rotate_right(source_seq, turns)
    target_seq = _random_target_sequence(length, shifted)
    source_corner = random.choice(((0, 0), (0, length + 1), (2, 0), (2, length + 1)))
    target_corner = random.choice(((0, 0), (0, length + 1), (2, 0), (2, length + 1)))

    source_node = _render_horizontal_node(length, source_seq, RED, source_corner)
    target_node = _render_horizontal_node(length, target_seq, BLUE, target_corner)
    path_cells = _build_path(turns)
    node_width = length + 2
    end_r, end_c = path_cells[-1]

    placements: list[tuple[np.ndarray, int, int]] = []
    placements.append((source_node, -1, -node_width))
    if turns % 2 == 0:
        placements.append((target_node, end_r - 1, end_c + 1))
    else:
        attach_offset = random.randint(1, node_width - 2)
        placements.append((target_node, end_r + 1, end_c - attach_offset))

    min_row = min([r for r, _c in path_cells] + [top for _arr, top, _left in placements])
    min_col = min([c for _r, c in path_cells] + [left for _arr, _top, left in placements])
    max_row = max(
        [r for r, _c in path_cells]
        + [top + arr.shape[0] - 1 for arr, top, _left in placements]
    )
    max_col = max(
        [c for _r, c in path_cells]
        + [left + arr.shape[1] - 1 for arr, _top, left in placements]
    )

    shift_r = 1 - min_row
    shift_c = 1 - min_col
    patch = np.zeros((max_row - min_row + 3, max_col - min_col + 3), dtype=int)

    for arr, top, left in placements:
        rr = top + shift_r
        cc = left + shift_c
        patch[rr : rr + arr.shape[0], cc : cc + arr.shape[1]] = arr

    for r, c in path_cells:
        patch[r + shift_r, c + shift_c] = MAROON

    if random.random() < 0.5:
        patch = np.fliplr(patch)
    patch = np.rot90(patch, random.randint(0, 3))
    return _crop_nonzero(patch)


def _can_place(occupied: np.ndarray, top: int, left: int, patch_shape: tuple[int, int], gap: int) -> bool:
    height, width = map(int, occupied.shape)
    ph, pw = map(int, patch_shape)
    r0 = max(0, top - gap)
    c0 = max(0, left - gap)
    r1 = min(height, top + ph + gap)
    c1 = min(width, left + pw + gap)
    return not bool(occupied[r0:r1, c0:c1].any())


def _place_patch(canvas: np.ndarray, occupied: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    ph, pw = map(int, patch.shape)
    mask = patch != BG
    target = canvas[top : top + ph, left : left + pw]
    target[mask] = patch[mask]
    occupied[top : top + ph, left : left + pw] = True


def _generate_attempt() -> np.ndarray | None:
    patch_count = random.randint(1, 3)
    patches = [_build_pair_patch() for _ in range(patch_count)]
    if max(p.shape[0] for p in patches) > 30 or max(p.shape[1] for p in patches) > 30:
        return None

    canvas = np.zeros((30, 30), dtype=int)
    occupied = np.zeros((30, 30), dtype=bool)
    for patch in random.sample(patches, len(patches)):
        ph, pw = map(int, patch.shape)
        placed = False
        for _ in range(120):
            top = random.randint(0, 30 - ph)
            left = random.randint(0, 30 - pw)
            if _can_place(occupied, top, left, patch.shape, gap=1):
                _place_patch(canvas, occupied, patch, top, left)
                placed = True
                break
        if not placed:
            return None

    r0, c0, r1, c1 = _nonzero_bbox(canvas)
    extra_top = random.randint(0, min(2, r0))
    extra_left = random.randint(0, min(2, c0))
    extra_bottom = random.randint(0, min(2, 29 - r1))
    extra_right = random.randint(0, min(2, 29 - c1))
    inp = canvas[
        r0 - extra_top : r1 + extra_bottom + 1,
        c0 - extra_left : c1 + extra_right + 1,
    ]

    out = main(inp)
    if inp.shape != out.shape:
        return None
    if max(inp.shape) > 30 or max(out.shape) > 30:
        return None
    if np.array_equal(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
