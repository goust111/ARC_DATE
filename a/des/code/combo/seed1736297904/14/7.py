from __future__ import annotations

import random

import numpy as np


BG = 0
WALL = 8
PALETTE = (1, 2, 3, 4, 5, 6, 7, 9)

# Rooms are at least 4x4, so every library shape below fits somewhere.
_BASE_SHAPES = (
    np.array([[1, 1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 1, 0]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
)


def _all_orientations(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[bytes] = set()
    for k in range(4):
        rotated = np.rot90(mask, k)
        for candidate in (rotated, np.fliplr(rotated)):
            key = candidate.tobytes() + bytes(candidate.shape)
            if key in seen:
                continue
            seen.add(key)
            variants.append(candidate.copy())
    return variants


SHAPES = [variant for base in _BASE_SHAPES for variant in _all_orientations(base)]


def _edge_key(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    return (a, b) if a <= b else (b, a)


def _neighbors(room: tuple[int, int], n_rows: int, n_cols: int):
    r, c = room
    if r > 0:
        yield (r - 1, c)
    if r + 1 < n_rows:
        yield (r + 1, c)
    if c > 0:
        yield (r, c - 1)
    if c + 1 < n_cols:
        yield (r, c + 1)


def _is_four_connected(coords: np.ndarray) -> bool:
    cells = {tuple(map(int, coord)) for coord in coords}
    stack = [next(iter(cells))]
    seen = set()
    while stack:
        r, c = stack.pop()
        if (r, c) in seen:
            continue
        seen.add((r, c))
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = (r + dr, c + dc)
            if nxt in cells and nxt not in seen:
                stack.append(nxt)
    return seen == cells


def _wall_lines(g: np.ndarray, axis: int) -> list[int]:
    wall_mask = g == WALL
    other = g.shape[1 - axis]
    counts = np.count_nonzero(wall_mask, axis=1 - axis)
    lines = np.flatnonzero(counts > other // 2).tolist()
    assert lines and lines[0] == 0 and lines[-1] == g.shape[axis] - 1
    assert all(b - a >= 2 for a, b in zip(lines, lines[1:])), "rooms must have nonempty interiors"
    return [int(x) for x in lines]


def _spans(lines: list[int]) -> list[tuple[int, int]]:
    spans = []
    for a, b in zip(lines, lines[1:]):
        assert b - a >= 2
        spans.append((a + 1, b - 1))
    return spans


def _room_boxes(row_spans: list[tuple[int, int]], col_spans: list[tuple[int, int]]):
    boxes = {}
    for ri, (r0, r1) in enumerate(row_spans):
        for ci, (c0, c1) in enumerate(col_spans):
            boxes[(ri, ci)] = (r0, c0, r1, c1)
    return boxes


def _classify_room(patch: np.ndarray) -> dict:
    coords = np.argwhere((patch != BG) & (patch != WALL))
    if len(coords) == 0:
        return {"kind": "empty"}

    colors = {int(patch[r, c]) for r, c in coords}
    assert len(colors) == 1, "room content should be monochrome"
    color = next(iter(colors))

    if len(coords) == 1:
        rr, cc = map(int, coords[0])
        h, w = patch.shape
        assert (rr, cc) in {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}, "marker must sit in a corner"
        return {"kind": "marker", "color": color, "local": (rr, cc)}

    assert 2 <= len(coords) <= 5, "furniture area must stay small"
    assert _is_four_connected(coords), "furniture must be 4-connected"
    r0, c0 = map(int, coords.min(axis=0))
    r1, c1 = map(int, coords.max(axis=0))
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return {
        "kind": "furniture",
        "color": color,
        "coords": coords.copy(),
        "mask": mask,
    }


def _collect_rooms(g: np.ndarray):
    row_lines = _wall_lines(g, axis=0)
    col_lines = _wall_lines(g, axis=1)
    row_spans = _spans(row_lines)
    col_spans = _spans(col_lines)
    boxes = _room_boxes(row_spans, col_spans)

    contents = {}
    for room, (r0, c0, r1, c1) in boxes.items():
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        assert not np.any(patch == WALL), "room interiors cannot contain wall pixels"
        contents[room] = _classify_room(patch)
    return row_lines, col_lines, row_spans, col_spans, boxes, contents


def _collect_doors(
    g: np.ndarray,
    row_lines: list[int],
    col_lines: list[int],
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
) -> dict[tuple[tuple[int, int], tuple[int, int]], tuple[str, int, int]]:
    doors = {}

    for ri, (r0, r1) in enumerate(row_spans):
        for ci in range(len(col_spans) - 1):
            wall_c = col_lines[ci + 1]
            segment = g[r0 : r1 + 1, wall_c]
            openings = np.flatnonzero(segment != WALL)
            assert len(openings) == 1, "each vertical wall segment must have exactly one door"
            door_r = r0 + int(openings[0])
            assert int(g[door_r, wall_c]) == BG, "door must be a black gap"
            doors[_edge_key((ri, ci), (ri, ci + 1))] = ("V", door_r, wall_c)

    for ri in range(len(row_spans) - 1):
        wall_r = row_lines[ri + 1]
        for ci, (c0, c1) in enumerate(col_spans):
            segment = g[wall_r, c0 : c1 + 1]
            openings = np.flatnonzero(segment != WALL)
            assert len(openings) == 1, "each horizontal wall segment must have exactly one door"
            door_c = c0 + int(openings[0])
            assert int(g[wall_r, door_c]) == BG, "door must be a black gap"
            doors[_edge_key((ri, ci), (ri + 1, ci))] = ("H", wall_r, door_c)

    return doors


def _placement_for_target(
    source_room: tuple[int, int],
    target_room: tuple[int, int],
    target_box: tuple[int, int, int, int],
    door: tuple[str, int, int],
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    r0, c0, r1, c1 = map(int, target_box)
    h, w = mask.shape
    orient, first, second = door

    if orient == "V":
        door_r, wall_c = first, second
        top = door_r - h // 2
        left = wall_c + 1 if target_room[1] > source_room[1] else wall_c - w
    else:
        wall_r, door_c = first, second
        top = wall_r + 1 if target_room[0] > source_room[0] else wall_r - h
        left = door_c - w // 2

    bottom = top + h - 1
    right = left + w - 1
    assert r0 <= top <= bottom <= r1
    assert c0 <= left <= right <= c1

    local = np.argwhere(mask)
    rows = local[:, 0] + top
    cols = local[:, 1] + left
    return rows, cols


def main(grid: np.ndarray) -> np.ndarray:
    """Move each furniture object through its door into the adjacent marked room and recolor it."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    row_lines, col_lines, row_spans, col_spans, boxes, contents = _collect_rooms(g)
    doors = _collect_doors(g, row_lines, col_lines, row_spans, col_spans)

    out = g.copy()
    for room, info in contents.items():
        r0, c0, _r1, _c1 = boxes[room]
        if info["kind"] == "furniture":
            coords = info["coords"]
            out[coords[:, 0] + r0, coords[:, 1] + c0] = BG
        elif info["kind"] == "marker":
            rr, cc = info["local"]
            out[r0 + rr, c0 + cc] = BG

    n_rows = len(row_spans)
    n_cols = len(col_spans)
    for room, info in contents.items():
        if info["kind"] != "furniture":
            continue

        matches = []
        for nbr in _neighbors(room, n_rows, n_cols):
            nbr_info = contents[nbr]
            if nbr_info["kind"] != "marker":
                continue
            door = doors[_edge_key(room, nbr)]
            matches.append((nbr, nbr_info["color"], door))

        assert len(matches) == 1, "each furniture room must match exactly one marked neighbor"
        target_room, target_color, door = matches[0]
        rows, cols = _placement_for_target(room, target_room, boxes[target_room], door, info["mask"])
        out[rows, cols] = target_color

    return out


ROOM_ROWS = (2, 4)
ROOM_COLS = (2, 4)
ROOM_SIDE = (4, 7)
GEN_ATTEMPTS = 400


def _greedy_pairs(n_rows: int, n_cols: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    edges = []
    for r in range(n_rows):
        for c in range(n_cols):
            if c + 1 < n_cols:
                edges.append(((r, c), (r, c + 1)))
            if r + 1 < n_rows:
                edges.append(((r, c), (r + 1, c)))
    random.shuffle(edges)

    blocked: set[tuple[int, int]] = set()
    pairs = []
    for a, b in edges:
        if a in blocked or b in blocked:
            continue
        pairs.append((a, b))
        shadow = {a, b}
        shadow.update(_neighbors(a, n_rows, n_cols))
        shadow.update(_neighbors(b, n_rows, n_cols))
        blocked.update(shadow)

    random.shuffle(pairs)
    keep = random.randint(1, len(pairs)) if pairs else 0
    return pairs[:keep]


def _room_layout():
    n_rows = random.randint(*ROOM_ROWS)
    n_cols = random.randint(*ROOM_COLS)
    heights = [random.randint(*ROOM_SIDE) for _ in range(n_rows)]
    widths = [random.randint(*ROOM_SIDE) for _ in range(n_cols)]

    row_lines = [0]
    cursor = 0
    for height in heights:
        cursor += height + 1
        row_lines.append(cursor)

    col_lines = [0]
    cursor = 0
    for width in widths:
        cursor += width + 1
        col_lines.append(cursor)

    row_spans = _spans(row_lines)
    col_spans = _spans(col_lines)
    boxes = _room_boxes(row_spans, col_spans)
    grid = np.full((row_lines[-1] + 1, col_lines[-1] + 1), BG, dtype=int)
    grid[row_lines, :] = WALL
    grid[:, col_lines] = WALL
    return grid, row_lines, col_lines, row_spans, col_spans, boxes


def _shape_that_fits(max_h: int, max_w: int) -> np.ndarray:
    choices = [shape for shape in SHAPES if shape.shape[0] <= max_h and shape.shape[1] <= max_w]
    assert choices
    return random.choice(choices).copy()


def _door_slot(span: tuple[int, int], extent: int) -> int:
    start, end = map(int, span)
    low = start + extent // 2
    high = end - (extent - 1 - extent // 2)
    assert low <= high
    return random.randint(low, high)


def _paint_mask(grid: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    local = np.argwhere(mask)
    rows = local[:, 0] + int(top)
    cols = local[:, 1] + int(left)
    assert np.all(grid[rows, cols] == BG)
    grid[rows, cols] = int(color)


def _corner_cell(box: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, box)
    return random.choice(((r0, c0), (r0, c1), (r1, c0), (r1, c1)))


def _place_pair(
    grid: np.ndarray,
    a: tuple[int, int],
    b: tuple[int, int],
    boxes: dict[tuple[int, int], tuple[int, int, int, int]],
) -> tuple[tuple[tuple[int, int], tuple[int, int]], tuple[str, int]]:
    furniture_room, marker_room = random.choice(((a, b), (b, a)))
    fa = boxes[furniture_room]
    fb = boxes[marker_room]
    fh = fa[2] - fa[0] + 1
    fw = fa[3] - fa[1] + 1
    mh = fb[2] - fb[0] + 1
    mw = fb[3] - fb[1] + 1

    mask = _shape_that_fits(min(fh, mh), min(fw, mw))
    furniture_color = random.choice(PALETTE)
    marker_choices = [color for color in PALETTE if color != furniture_color]
    marker_color = random.choice(marker_choices)

    fr0, fc0, fr1, fc1 = fa
    top = random.randint(fr0, fr1 - mask.shape[0] + 1)
    left = random.randint(fc0, fc1 - mask.shape[1] + 1)
    _paint_mask(grid, top, left, mask, furniture_color)

    mr, mc = _corner_cell(fb)
    assert grid[mr, mc] == BG
    grid[mr, mc] = marker_color

    if furniture_room[0] == marker_room[0]:
        door = ("V", _door_slot((fr0, fr1), mask.shape[0]))
    else:
        door = ("H", _door_slot((fc0, fc1), mask.shape[1]))
    return (furniture_room, marker_room), door


def _apply_doors(
    grid: np.ndarray,
    row_lines: list[int],
    col_lines: list[int],
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    pair_doors: dict[tuple[tuple[int, int], tuple[int, int]], tuple[str, int]],
) -> None:
    for ri, (r0, r1) in enumerate(row_spans):
        for ci in range(len(col_spans) - 1):
            a = (ri, ci)
            b = (ri, ci + 1)
            wall_c = col_lines[ci + 1]
            key = _edge_key(a, b)
            if key in pair_doors:
                orient, coord = pair_doors[key]
                assert orient == "V"
                door_r = coord
            else:
                door_r = random.randint(r0, r1)
            grid[door_r, wall_c] = BG

    for ri in range(len(row_spans) - 1):
        wall_r = row_lines[ri + 1]
        for ci, (c0, c1) in enumerate(col_spans):
            a = (ri, ci)
            b = (ri + 1, ci)
            key = _edge_key(a, b)
            if key in pair_doors:
                orient, coord = pair_doors[key]
                assert orient == "H"
                door_c = coord
            else:
                door_c = random.randint(c0, c1)
            grid[wall_r, door_c] = BG


def _gen_attempt() -> np.ndarray | None:
    grid, row_lines, col_lines, row_spans, col_spans, boxes = _room_layout()
    pairs = _greedy_pairs(len(row_spans), len(col_spans))
    if not pairs:
        return None

    pair_doors: dict[tuple[tuple[int, int], tuple[int, int]], tuple[str, int]] = {}
    for a, b in pairs:
        key, door = _place_pair(grid, a, b, boxes)
        pair_doors[_edge_key(*key)] = door

    _apply_doors(grid, row_lines, col_lines, row_spans, col_spans, pair_doors)

    try:
        solved = main(grid)
    except Exception:
        return None

    # The task should be nontrivial: at least one furniture object must move to a new color.
    if np.array_equal(solved, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-transfer input")
