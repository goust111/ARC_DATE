from __future__ import annotations

from collections import Counter, deque
from itertools import product
import random

import numpy as np


BG = 0
WALL = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
LAYOUTS = ((2, 2), (2, 3), (3, 2))
ROOM_SIZE_RANGE = (4, 6)
MARGIN_RANGE = (4, 6)
DOOR_LEN_RANGE = (2, 4)
PASSAGE_LEN_RANGE = (2, 3)
KEY_SHAPES = (
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1], [0, 0, 1]], dtype=bool),
)
GEN_RETRIES = 400


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _most_common_color(g: np.ndarray) -> int:
    values, counts = np.unique(g, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _component_infos(mask: np.ndarray) -> list[dict]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            patch[arr[:, 0] - r0, arr[:, 1] - c0] = True
            comps.append(
                {
                    "coords": arr,
                    "bbox": (r0, c0, r1, c1),
                    "area": int(arr.shape[0]),
                    "mask": patch,
                }
            )
    return comps


def _flood_from_boundary(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if mask[r, c] and not seen[r, c]:
                seen[r, c] = True
                queue.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if mask[r, c] and not seen[r, c]:
                seen[r, c] = True
                queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c, h, w):
            if mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                queue.append((nr, nc))
    return seen


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, dict[int, tuple[int, int, int, int]]]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    bboxes: dict[int, tuple[int, int, int, int]] = {}
    next_label = 1

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            queue = deque([(r, c)])
            labels[r, c] = next_label
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = next_label
                        queue.append((nr, nc))
            arr = np.array(coords, dtype=int)
            bboxes[next_label] = (
                int(arr[:, 0].min()),
                int(arr[:, 1].min()),
                int(arr[:, 0].max()),
                int(arr[:, 1].max()),
            )
            next_label += 1
    return labels, bboxes


def _find_wall_color(g: np.ndarray, bg: int) -> int:
    counts = Counter(int(v) for v in g.ravel() if int(v) != bg)
    assert counts, "expected at least one non-background color"
    return counts.most_common(1)[0][0]


def _is_door_component(comp: dict, building_bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    br0, bc0, br1, bc1 = building_bbox
    if not (br0 < r0 <= r1 < br1 and bc0 < c0 <= c1 < bc1):
        return False
    if not (2 <= int(comp["area"]) <= 4):
        return False
    return (r0 == r1) ^ (c0 == c1)


def _collect_objects(g: np.ndarray, bg: int, wall: int, building_bbox: tuple[int, int, int, int]) -> tuple[dict[int, dict], dict[int, dict]]:
    door_by_color: dict[int, dict] = {}
    key_by_color: dict[int, dict] = {}
    for color in np.unique(g):
        color = int(color)
        if color in (bg, wall):
            continue
        comps = _component_infos(g == color)
        is_door = [_is_door_component(comp, building_bbox) for comp in comps]
        doors = [comp for comp, flag in zip(comps, is_door) if flag]
        keys = [comp for comp, flag in zip(comps, is_door) if not flag]
        assert len(doors) == 1, "each color must define exactly one door"
        assert len(keys) == 1, "each color must define exactly one key"
        door_by_color[color] = doors[0]
        key_by_color[color] = keys[0]
    assert door_by_color, "expected at least one door"
    return door_by_color, key_by_color


def _side_samples(door: dict, orientation: str) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    r0, c0, r1, c1 = door["bbox"]
    if orientation == "horizontal":
        return ([(r0 - 1, c) for c in range(c0, c1 + 1)], [(r0 + 1, c) for c in range(c0, c1 + 1)])
    return ([(r, c0 - 1) for r in range(r0, r1 + 1)], [(r, c0 + 1) for r in range(r0, r1 + 1)])


def _place_mask(
    out: np.ndarray,
    mask: np.ndarray,
    color: int,
    start_r: int,
    start_c: int,
    hidden_labels: np.ndarray,
    label: int,
) -> None:
    coords = np.argwhere(mask)
    for dr_raw, dc_raw in coords:
        dr = int(dr_raw)
        dc = int(dc_raw)
        rr = start_r + dr
        cc = start_c + dc
        assert 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1], "mirrored key leaves the grid"
        assert int(hidden_labels[rr, cc]) == int(label), "mirrored key must stay inside the hidden room"
        out[rr, cc] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each colored key to the same-colored door, mirror the key across the door wall axis,
    and place the mirrored key inside the hidden room behind that door.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _most_common_color(g)
    wall = _find_wall_color(g, bg)
    wall_coords = np.argwhere(g == wall)
    assert wall_coords.size > 0, "expected wall pixels"
    br0 = int(wall_coords[:, 0].min())
    bc0 = int(wall_coords[:, 1].min())
    br1 = int(wall_coords[:, 0].max())
    bc1 = int(wall_coords[:, 1].max())
    building_bbox = (br0, bc0, br1, bc1)

    door_by_color, key_by_color = _collect_objects(g, bg, wall, building_bbox)

    black_mask = g == bg
    outside_visible = _flood_from_boundary(black_mask)
    hidden_labels, _hidden_bboxes = _label_components(black_mask & ~outside_visible)

    out = g.copy()
    for color, door in door_by_color.items():
        key = key_by_color[color]
        r0, c0, r1, c1 = door["bbox"]
        orientation = "horizontal" if r0 == r1 else "vertical"
        side_a, side_b = _side_samples(door, orientation)

        def hidden_ids(samples: list[tuple[int, int]]) -> set[int]:
            vals = set()
            for rr, cc in samples:
                if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]:
                    label = int(hidden_labels[rr, cc])
                    if label > 0:
                        vals.add(label)
            return vals

        def visible_hits(samples: list[tuple[int, int]]) -> int:
            hits = 0
            for rr, cc in samples:
                if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1] and bool(outside_visible[rr, cc]):
                    hits += 1
            return hits

        hidden_a = hidden_ids(side_a)
        hidden_b = hidden_ids(side_b)
        visible_a = visible_hits(side_a)
        visible_b = visible_hits(side_b)

        if hidden_a and visible_b and not hidden_b:
            hidden_label = next(iter(hidden_a))
            hidden_side = "negative"
        elif hidden_b and visible_a and not hidden_a:
            hidden_label = next(iter(hidden_b))
            hidden_side = "positive"
        else:
            raise AssertionError("door must separate one visible side and one hidden side")

        assert len(hidden_a or hidden_b) == 1, "door must border exactly one hidden room"
        mirrored = np.flipud(key["mask"]) if orientation == "horizontal" else np.fliplr(key["mask"])
        ph, pw = mirrored.shape

        if orientation == "horizontal":
            start_r = r0 - ph if hidden_side == "negative" else r0 + 1
            start_c = (c0 + c1 - (pw - 1)) // 2
        else:
            start_r = (r0 + r1 - (ph - 1)) // 2
            start_c = c0 - pw if hidden_side == "negative" else c0 + 1

        out[door["coords"][:, 0], door["coords"][:, 1]] = wall
        _place_mask(out, mirrored, color, start_r, start_c, hidden_labels, hidden_label)

    return out


def _partition(total: int, parts: int, min_size: int) -> list[int]:
    sizes = [min_size] * parts
    extra = total - min_size * parts
    for _ in range(extra):
        sizes[random.randrange(parts)] += 1
    random.shuffle(sizes)
    return sizes


def _interior_ranges(start: int, sizes: list[int]) -> tuple[list[tuple[int, int]], list[int]]:
    rooms: list[tuple[int, int]] = []
    walls: list[int] = []
    pos = start + 1
    for i, size in enumerate(sizes):
        rooms.append((pos, pos + size - 1))
        pos += size
        if i + 1 < len(sizes):
            walls.append(pos)
            pos += 1
    return rooms, walls


def _cell_neighbors(cell: tuple[int, int], n_rows: int, n_cols: int) -> list[tuple[int, int]]:
    r, c = cell
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < n_rows:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < n_cols:
        out.append((r, c + 1))
    return out


def _is_connected(cells: set[tuple[int, int]], n_rows: int, n_cols: int) -> bool:
    start = next(iter(cells))
    queue = deque([start])
    seen = {start}
    while queue:
        cell = queue.popleft()
        for nb in _cell_neighbors(cell, n_rows, n_cols):
            if nb in cells and nb not in seen:
                seen.add(nb)
                queue.append(nb)
    return len(seen) == len(cells)


def _choose_visible_cells(n_rows: int, n_cols: int) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    cells = [(r, c) for r in range(n_rows) for c in range(n_cols)]
    boundary = {cell for cell in cells if cell[0] in (0, n_rows - 1) or cell[1] in (0, n_cols - 1)}
    candidates: list[tuple[set[tuple[int, int]], set[tuple[int, int]]]] = []

    for bits in range(1, (1 << len(cells)) - 1):
        visible = {cells[i] for i in range(len(cells)) if (bits >> i) & 1}
        hidden = set(cells) - visible
        if not (1 <= len(hidden) <= 4):
            continue
        if not visible & boundary:
            continue
        if not _is_connected(visible, n_rows, n_cols):
            continue
        if any(not (set(_cell_neighbors(cell, n_rows, n_cols)) & visible) for cell in hidden):
            continue
        candidates.append((visible, hidden))

    assert candidates, "expected at least one valid room split"
    return random.choice(candidates)


def _door_orientation(a: tuple[int, int], b: tuple[int, int]) -> str:
    return "horizontal" if a[1] == b[1] else "vertical"


def _choose_door_assignments(
    visible: set[tuple[int, int]],
    hidden: set[tuple[int, int]],
    n_rows: int,
    n_cols: int,
) -> dict[tuple[int, int], tuple[int, int]]:
    hidden_list = sorted(hidden)
    choices = [sorted(set(_cell_neighbors(cell, n_rows, n_cols)) & visible) for cell in hidden_list]
    all_assignments = [dict(zip(hidden_list, picks)) for picks in product(*choices)]
    profiles: dict[str, list[dict[tuple[int, int], tuple[int, int]]]] = {"mixed": [], "horizontal": [], "vertical": []}

    for assignment in all_assignments:
        orientations = {_door_orientation(cell, nb) for cell, nb in assignment.items()}
        if orientations == {"horizontal", "vertical"}:
            profiles["mixed"].append(assignment)
        elif orientations == {"horizontal"}:
            profiles["horizontal"].append(assignment)
        else:
            profiles["vertical"].append(assignment)

    target = random.choice(("mixed", "horizontal", "vertical"))
    options = profiles[target]
    if not options:
        options = profiles["mixed"] or profiles["horizontal"] or profiles["vertical"]
    assert options, "expected at least one hidden-to-visible assignment"
    return random.choice(options)


def _centered_segment(lo: int, hi: int, length_choices: tuple[int, int]) -> tuple[int, int]:
    span = hi - lo + 1
    max_len = min(length_choices[1], span - 2)
    min_len = length_choices[0]
    assert min_len <= max_len, "wall span is too short for a readable opening"
    length = random.randint(min_len, max_len)
    start = lo + (span - length) // 2
    return start, start + length - 1


def _draw_building(
    room_rows: list[tuple[int, int]],
    room_cols: list[tuple[int, int]],
    top: int,
    left: int,
) -> np.ndarray:
    bottom = room_rows[-1][1] + 1
    right = room_cols[-1][1] + 1
    grid = np.full((bottom + top + 1, right + left + 1), BG, dtype=int)
    grid[top : bottom + 1, left] = WALL
    grid[top : bottom + 1, right] = WALL
    grid[top, left : right + 1] = WALL
    grid[bottom, left : right + 1] = WALL

    for idx in range(len(room_rows) - 1):
        wall_r = room_rows[idx][1] + 1
        grid[wall_r, left : right + 1] = WALL
    for idx in range(len(room_cols) - 1):
        wall_c = room_cols[idx][1] + 1
        grid[top : bottom + 1, wall_c] = WALL
    return grid


def _segment_in_shared_wall(
    room_rows: list[tuple[int, int]],
    room_cols: list[tuple[int, int]],
    cell_a: tuple[int, int],
    cell_b: tuple[int, int],
    length_choices: tuple[int, int],
) -> tuple[np.ndarray, str]:
    ar, ac = cell_a
    br, bc = cell_b
    if ar != br:
        wall_r = room_rows[min(ar, br)][1] + 1
        c0, c1 = room_cols[ac]
        seg0, seg1 = _centered_segment(c0, c1, length_choices)
        coords = np.array([(wall_r, c) for c in range(seg0, seg1 + 1)], dtype=int)
        return coords, "horizontal"

    wall_c = room_cols[min(ac, bc)][1] + 1
    r0, r1 = room_rows[ar]
    seg0, seg1 = _centered_segment(r0, r1, length_choices)
    coords = np.array([(r, wall_c) for r in range(seg0, seg1 + 1)], dtype=int)
    return coords, "vertical"


def _outer_opening_coords(
    room_rows: list[tuple[int, int]],
    room_cols: list[tuple[int, int]],
    cell: tuple[int, int],
    top: int,
    left: int,
) -> np.ndarray:
    r, c = cell
    sides: list[str] = []
    if r == 0:
        sides.append("top")
    if r == len(room_rows) - 1:
        sides.append("bottom")
    if c == 0:
        sides.append("left")
    if c == len(room_cols) - 1:
        sides.append("right")
    side = random.choice(sides)

    rr0, rr1 = room_rows[r]
    cc0, cc1 = room_cols[c]
    if side == "top":
        seg0, seg1 = _centered_segment(cc0, cc1, PASSAGE_LEN_RANGE)
        return np.array([(top, col) for col in range(seg0, seg1 + 1)], dtype=int)
    if side == "bottom":
        outer_r = room_rows[-1][1] + 1
        seg0, seg1 = _centered_segment(cc0, cc1, PASSAGE_LEN_RANGE)
        return np.array([(outer_r, col) for col in range(seg0, seg1 + 1)], dtype=int)
    if side == "left":
        seg0, seg1 = _centered_segment(rr0, rr1, PASSAGE_LEN_RANGE)
        return np.array([(row, left) for row in range(seg0, seg1 + 1)], dtype=int)
    outer_c = room_cols[-1][1] + 1
    seg0, seg1 = _centered_segment(rr0, rr1, PASSAGE_LEN_RANGE)
    return np.array([(row, outer_c) for row in range(seg0, seg1 + 1)], dtype=int)


def _transform_shape(mask: np.ndarray) -> np.ndarray:
    out = np.array(mask, copy=True)
    out = np.rot90(out, random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    return out


def _dilate(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            for rr in range(max(0, r - 1), min(h, r + 2)):
                for cc in range(max(0, c - 1), min(w, c + 2)):
                    out[rr, cc] = True
    return out


def _place_key(grid: np.ndarray, shape: np.ndarray, color: int, forbidden: np.ndarray) -> bool:
    h, w = grid.shape
    ph, pw = shape.shape
    occupied = _dilate(grid != BG) | forbidden
    candidates: list[tuple[int, int]] = []

    for r0 in range(h - ph + 1):
        for c0 in range(w - pw + 1):
            ok = True
            for dr_raw, dc_raw in np.argwhere(shape):
                rr = r0 + int(dr_raw)
                cc = c0 + int(dc_raw)
                if occupied[rr, cc]:
                    ok = False
                    break
            if ok:
                candidates.append((r0, c0))

    if not candidates:
        return False
    r0, c0 = random.choice(candidates)
    for dr_raw, dc_raw in np.argwhere(shape):
        grid[r0 + int(dr_raw), c0 + int(dc_raw)] = int(color)
    return True


def _make_attempt() -> np.ndarray | None:
    n_rows, n_cols = random.choice(LAYOUTS)
    room_heights = [random.randint(*ROOM_SIZE_RANGE) for _ in range(n_rows)]
    room_widths = [random.randint(*ROOM_SIZE_RANGE) for _ in range(n_cols)]
    top = random.randint(*MARGIN_RANGE)
    left = random.randint(*MARGIN_RANGE)
    bottom_margin = random.randint(*MARGIN_RANGE)
    right_margin = random.randint(*MARGIN_RANGE)

    building_h = sum(room_heights) + n_rows + 1
    building_w = sum(room_widths) + n_cols + 1
    total_h = top + building_h + bottom_margin
    total_w = left + building_w + right_margin

    room_rows, _row_walls = _interior_ranges(top, room_heights)
    room_cols, _col_walls = _interior_ranges(left, room_widths)
    grid = np.full((total_h, total_w), BG, dtype=int)
    bottom = top + building_h - 1
    right = left + building_w - 1

    grid[top : bottom + 1, left] = WALL
    grid[top : bottom + 1, right] = WALL
    grid[top, left : right + 1] = WALL
    grid[bottom, left : right + 1] = WALL
    for idx in range(n_rows - 1):
        grid[room_rows[idx][1] + 1, left : right + 1] = WALL
    for idx in range(n_cols - 1):
        grid[top : bottom + 1, room_cols[idx][1] + 1] = WALL

    visible, hidden = _choose_visible_cells(n_rows, n_cols)
    assignment = _choose_door_assignments(visible, hidden, n_rows, n_cols)
    door_colors = random.sample(PALETTE, len(hidden))
    hidden_list = sorted(hidden)

    for cell_a in visible:
        for cell_b in _cell_neighbors(cell_a, n_rows, n_cols):
            if cell_b not in visible or cell_a > cell_b:
                continue
            coords, _orientation = _segment_in_shared_wall(room_rows, room_cols, cell_a, cell_b, PASSAGE_LEN_RANGE)
            grid[coords[:, 0], coords[:, 1]] = BG

    for color, cell in zip(door_colors, hidden_list):
        neighbor = assignment[cell]
        coords, _orientation = _segment_in_shared_wall(room_rows, room_cols, cell, neighbor, DOOR_LEN_RANGE)
        grid[coords[:, 0], coords[:, 1]] = int(color)

    entry_cell = random.choice(sorted(visible))
    opening = _outer_opening_coords(room_rows, room_cols, entry_cell, top, left)
    grid[opening[:, 0], opening[:, 1]] = BG

    forbidden = np.zeros_like(grid, dtype=bool)
    fr0 = max(0, top - 1)
    fc0 = max(0, left - 1)
    fr1 = min(grid.shape[0] - 1, bottom + 1)
    fc1 = min(grid.shape[1] - 1, right + 1)
    forbidden[fr0 : fr1 + 1, fc0 : fc1 + 1] = True

    for color, cell in zip(door_colors, hidden_list):
        room_h = room_heights[cell[0]]
        room_w = room_widths[cell[1]]
        fitting = []
        for base in KEY_SHAPES:
            for _ in range(4):
                transformed = _transform_shape(base)
                if transformed.shape[0] <= room_h and transformed.shape[1] <= room_w:
                    fitting.append(transformed)
        if not fitting:
            return None
        shape = random.choice(fitting)
        if not _place_key(grid, shape, color, forbidden):
            return None

    if np.array_equal(main(grid), grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _make_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
