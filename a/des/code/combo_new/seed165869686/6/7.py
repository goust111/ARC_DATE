import random

import numpy as np


BG = 0
GREEN = 3
GREY = 5
PATH_COLORS = [1, 2, 4, 6, 7, 8, 9]
DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]
LEFT_OF = {DIRS4[i]: DIRS4[(i - 1) % 4] for i in range(4)}
RIGHT_OF = {DIRS4[i]: DIRS4[(i + 1) % 4] for i in range(4)}
SIDE_TO_INWARD = {
    "top": (1, 0),
    "bottom": (-1, 0),
    "left": (0, 1),
    "right": (0, -1),
}
SIDE_TO_OUTWARD = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
OPPOSITE_SIDE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
GEN_GRID_H_RANGE = (18, 29)
GEN_GRID_W_RANGE = (18, 29)
GEN_SEGMENT_COUNT_RANGE = (2, 5)
GEN_SEGMENT_STEP_RANGE = (1, 4)
GEN_TOTAL_STEPS_MAX = 16
GEN_PATH_BOUND = 8
GEN_MAX_OUTPUT_SPAN = 14
GEN_MAX_ATTEMPTS = 300


def _add(a, b):
    return (int(a[0] + b[0]), int(a[1] + b[1]))


def _neg(v):
    return (-int(v[0]), -int(v[1]))


def _rot_coord(pt, k):
    r, c = int(pt[0]), int(pt[1])
    k %= 4
    if k == 0:
        return (r, c)
    if k == 1:
        return (-c, r)
    if k == 2:
        return (-r, -c)
    return (c, -r)


def _rot_dir(vec, k):
    return _rot_coord(vec, k)


def _neighbors4(pt):
    r, c = int(pt[0]), int(pt[1])
    return [(r + dr, c + dc) for dr, dc in DIRS4]


def _in_bounds(shape, pt):
    return 0 <= int(pt[0]) < int(shape[0]) and 0 <= int(pt[1]) < int(shape[1])


def _border_side(pt, shape):
    r, c = map(int, pt)
    h, w = map(int, shape)
    if r == 0:
        return "top"
    if r == h - 1:
        return "bottom"
    if c == 0:
        return "left"
    if c == w - 1:
        return "right"
    raise AssertionError("port must lie on the border")


def _color_components(grid, color):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for nr, nc in _neighbors4((cr, cc)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _normalize_coords(coords):
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return [(int(r - min_r), int(c - min_c)) for r, c in coords]


def _path_output(path_cells):
    rows = [r for r, _ in path_cells]
    cols = [c for _, c in path_cells]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    out = np.full((max_r - min_r + 1, max_c - min_c + 1), BG, dtype=int)
    for r, c in path_cells:
        out[int(r - min_r), int(c - min_c)] = GREEN
    return out


def _extract_segments(grid):
    colors = [int(v) for v in np.unique(grid) if int(v) not in (BG, GREY)]
    segments = {}
    for color in colors:
        comps = _color_components(grid, color)
        path_comps = [comp for comp in comps if len(comp) >= 2]
        if not path_comps:
            continue
        assert len(path_comps) == 1, "each path color must have exactly one segment"
        coords = [tuple(map(int, pt)) for pt in path_comps[0]]
        coord_set = set(coords)
        endpoints = []
        inward = {}
        for pt in coords:
            same = [nb for nb in _neighbors4(pt) if nb in coord_set]
            assert len(same) in (1, 2), "path segment must be a thin polyline"
            if len(same) == 1:
                endpoints.append(pt)
                inward[pt] = (int(same[0][0] - pt[0]), int(same[0][1] - pt[1]))
        assert len(endpoints) == 2, "each segment must have exactly two endpoints"

        marker_endpoint = None
        next_color = None
        grey_endpoints = []
        for ep in endpoints:
            marker_colors = set()
            has_grey = False
            for nb in _neighbors4(ep):
                if not _in_bounds(grid.shape, nb):
                    continue
                value = int(grid[nb])
                if value == GREY:
                    has_grey = True
                elif value not in (BG, color):
                    marker_colors.add(value)
            assert len(marker_colors) <= 1, "endpoint marker must be unambiguous"
            if marker_colors:
                assert marker_endpoint is None, "segment may expose only one next-color marker"
                marker_endpoint = ep
                next_color = int(next(iter(marker_colors)))
            if has_grey:
                grey_endpoints.append(ep)

        segments[color] = {
            "color": color,
            "coords": coords,
            "coord_set": coord_set,
            "endpoints": endpoints,
            "inward": inward,
            "marker_endpoint": marker_endpoint,
            "next_color": next_color,
            "grey_endpoints": grey_endpoints,
        }
    return segments


def _transform_segment(segment, entry_endpoint, rotation, anchor):
    rotated_entry = _rot_coord(entry_endpoint, rotation)
    delta = (int(anchor[0] - rotated_entry[0]), int(anchor[1] - rotated_entry[1]))
    coords = [_add(_rot_coord(pt, rotation), delta) for pt in segment["coords"]]
    endpoints = {ep: _add(_rot_coord(ep, rotation), delta) for ep in segment["endpoints"]}
    inward = {ep: _rot_dir(segment["inward"][ep], rotation) for ep in segment["endpoints"]}
    return {
        "coords": coords,
        "endpoints": endpoints,
        "inward": inward,
    }


def _assemble_path(segments, order, start_dir, end_dir):
    occupied = set()
    current_pos = (0, 0)
    current_dir = start_dir

    for idx, color in enumerate(order):
        segment = segments[color]
        candidates = []

        if idx < len(order) - 1:
            assert segment["marker_endpoint"] is not None, "non-terminal segment must point to the next color"
            exit_ep = segment["marker_endpoint"]
            entry_ep = segment["endpoints"][0] if segment["endpoints"][1] == exit_ep else segment["endpoints"][1]
            for rot in range(4):
                if _rot_dir(segment["inward"][entry_ep], rot) == current_dir:
                    candidates.append((entry_ep, exit_ep, rot))
        else:
            assert segment["next_color"] is None, "terminal segment must not have a next-color marker"
            for entry_ep in segment["endpoints"]:
                exit_ep = segment["endpoints"][0] if segment["endpoints"][1] == entry_ep else segment["endpoints"][1]
                for rot in range(4):
                    if _rot_dir(segment["inward"][entry_ep], rot) != current_dir:
                        continue
                    out_dir = _neg(_rot_dir(segment["inward"][exit_ep], rot))
                    if out_dir == end_dir:
                        candidates.append((entry_ep, exit_ep, rot))

        assert candidates, "segment orientation must be solvable"
        placed = None
        allowed_overlap = {current_pos} if idx > 0 else set()
        for entry_ep, exit_ep, rot in candidates:
            transformed = _transform_segment(segment, entry_ep, rot, current_pos)
            coords = transformed["coords"]
            if any(pt in occupied and pt not in allowed_overlap for pt in coords):
                continue
            placed = (transformed, exit_ep)
            break
        assert placed is not None, "segment placement must not create extra overlaps"

        transformed, exit_ep = placed
        for pt in transformed["coords"]:
            occupied.add(pt)
        current_pos = transformed["endpoints"][exit_ep]
        current_dir = _neg(transformed["inward"][exit_ep])

    return occupied


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered colored path segments into one continuous path by following markers and rotating each piece.

    Input structure:
    - The background is black.
    - Several colors each form one 4-connected one-pixel-thick path segment, and each such segment has exactly two endpoints.
    - Two grey single-pixel ports lie on opposite borders; the start port touches the first segment's color, while endpoint-adjacent colored singleton markers indicate the next segment color.

    Rule:
    - Find the start port as the grey border pixel adjacent to a non-black non-grey path color, and use that adjacent color as the first segment.
    - Follow the chain of endpoint markers to order the remaining segments.
    - Starting from the start-port side direction, rotate each segment by 0/90/180/270 degrees and translate it so its entry endpoint coincides with the previous tail and its local tangent continues the path.
    - Choose the last segment orientation so its final tangent points toward the opposite grey port side.
    - Output the assembled path on its minimal bounding box, recolored entirely green.

    Key insight:
    - The puzzle factorizes into two independent inferences: marker colors give the linked-list order of segments, and endpoint tangents determine which rotation makes each segment connect continuously.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    ports = [tuple(map(int, pt)) for pt in np.argwhere(g == GREY)]
    assert len(ports) == 2, "expected exactly two grey ports"

    start_info = []
    for port in ports:
        adj_colors = {
            int(g[nb])
            for nb in _neighbors4(port)
            if _in_bounds(g.shape, nb) and int(g[nb]) not in (BG, GREY)
        }
        if adj_colors:
            start_info.append((port, adj_colors))
    assert len(start_info) == 1, "exactly one grey port must indicate the first segment"
    start_port, start_colors = start_info[0]
    assert len(start_colors) == 1, "start port must identify one first-segment color"
    start_color = int(next(iter(start_colors)))
    end_port = ports[0] if ports[1] == start_port else ports[1]

    start_side = _border_side(start_port, g.shape)
    end_side = _border_side(end_port, g.shape)
    assert OPPOSITE_SIDE[start_side] == end_side, "ports must lie on opposite sides"

    segments = _extract_segments(g)
    assert start_color in segments, "the start color must correspond to a path segment"

    order = []
    seen = set()
    color = start_color
    while True:
        assert color not in seen, "segment order must be acyclic"
        assert color in segments, "every referenced marker color must name a real segment"
        seen.add(color)
        order.append(color)
        nxt = segments[color]["next_color"]
        if nxt is None:
            break
        color = int(nxt)

    assert seen == set(segments), "all segments must belong to the single marker chain"

    occupied = _assemble_path(
        segments=segments,
        order=order,
        start_dir=SIDE_TO_INWARD[start_side],
        end_dir=SIDE_TO_OUTWARD[end_side],
    )
    assert occupied, "assembled path must be non-empty"

    rows = [r for r, _ in occupied]
    cols = [c for _, c in occupied]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    out = np.full((max_r - min_r + 1, max_c - min_c + 1), BG, dtype=int)
    for r, c in occupied:
        out[int(r - min_r), int(c - min_c)] = GREEN
    return out


def _path_legal(new_pt, prev_pt, occupied):
    if abs(int(new_pt[0])) > GEN_PATH_BOUND or abs(int(new_pt[1])) > GEN_PATH_BOUND:
        return False
    if new_pt in occupied:
        return False
    neighbors = 0
    for nb in _neighbors4(new_pt):
        if nb in occupied:
            if nb != prev_pt:
                return False
            neighbors += 1
    return neighbors == 1


def _generate_path(total_steps, start_dir, end_dir):
    path = [(0, 0)]
    occupied = {(0, 0)}

    def dfs(pos, prev_dir, steps_done):
        if steps_done == total_steps:
            return prev_dir == end_dir
        if steps_done == 0:
            candidates = [start_dir]
        elif steps_done == total_steps - 1:
            candidates = [end_dir]
        else:
            candidates = [prev_dir, LEFT_OF[prev_dir], RIGHT_OF[prev_dir]]
            random.shuffle(candidates)
        for direction in candidates:
            nxt = _add(pos, direction)
            if not _path_legal(nxt, pos, occupied):
                continue
            occupied.add(nxt)
            path.append(nxt)
            if dfs(nxt, direction, steps_done + 1):
                return True
            path.pop()
            occupied.remove(nxt)
        return False

    ok = dfs((0, 0), None, 0)
    return path if ok else None


def _step_dirs(path):
    return [(int(path[i + 1][0] - path[i][0]), int(path[i + 1][1] - path[i][1])) for i in range(len(path) - 1)]


def _segment_sequences(path, step_counts):
    out = []
    offset = 0
    for count in step_counts:
        out.append(path[offset : offset + count + 1])
        offset += count
    return out


def _rotate_sequence(seq, rotation):
    rotated = [_rot_coord(pt, rotation) for pt in seq]
    return _normalize_coords(rotated)


def _is_inner(pt, shape):
    h, w = map(int, shape)
    r, c = map(int, pt)
    return 1 <= r < h - 1 and 1 <= c < w - 1


def _adjacent_count(cell, cell_set):
    return sum(nb in cell_set for nb in _neighbors4(cell))


def _marker_candidates(seq):
    path_set = set(seq)
    exit_cell = seq[-1]
    options = []
    for nb in _neighbors4(exit_cell):
        if nb in path_set:
            continue
        if _adjacent_count(nb, path_set) == 1 and exit_cell in _neighbors4(nb):
            options.append(nb)
    random.shuffle(options)
    return options


def _can_place_object(shape, occupied, cells):
    h, w = map(int, shape)
    cell_set = set(cells)
    for r, c in cell_set:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if occupied[r, c]:
            return False
    for r, c in cell_set:
        for nr in range(max(0, r - 1), min(h, r + 2)):
            for nc in range(max(0, c - 1), min(w, c + 2)):
                if (nr, nc) in cell_set:
                    continue
                if occupied[nr, nc]:
                    return False
    return True


def _stamp_object(grid, occupied, color_map):
    for (r, c), color in color_map.items():
        grid[r, c] = int(color)
        occupied[r, c] = True


def _segment_color_map(seq, color):
    return {tuple(map(int, pt)): int(color) for pt in seq}


def _try_place_first(grid, occupied, seq, color, next_color, start_side):
    h, w = grid.shape
    entry = seq[0]
    inward = SIDE_TO_INWARD[start_side]
    port_offset = _neg(inward)
    marker_options = [None] if next_color is None else _marker_candidates(seq)
    if next_color is not None and not marker_options:
        return False

    if start_side in ("left", "right"):
        targets = [(r, 1 if start_side == "left" else w - 2) for r in range(1, h - 1)]
    else:
        targets = [(1 if start_side == "top" else h - 2, c) for c in range(1, w - 1)]
    random.shuffle(targets)

    for target in targets:
        shift = (int(target[0] - entry[0]), int(target[1] - entry[1]))
        placed_seq = [_add(pt, shift) for pt in seq]
        placed_set = set(placed_seq)
        if not all(_is_inner(pt, grid.shape) for pt in placed_seq):
            continue

        port = _add(target, port_offset)
        if _border_side(port, grid.shape) != start_side:
            continue
        if _adjacent_count(port, placed_set) != 1 or target not in _neighbors4(port):
            continue

        for marker_local in marker_options:
            color_map = _segment_color_map(placed_seq, color)
            if marker_local is not None:
                marker = _add(marker_local, shift)
                if not _is_inner(marker, grid.shape):
                    continue
                color_map[marker] = int(next_color)
            color_map[port] = GREY
            if _can_place_object(grid.shape, occupied, color_map.keys()):
                _stamp_object(grid, occupied, color_map)
                return True
    return False


def _try_place_free(grid, occupied, seq, color, next_color):
    h, w = grid.shape
    path_rows = [r for r, _ in seq]
    path_cols = [c for _, c in seq]
    path_h = max(path_rows) + 1
    path_w = max(path_cols) + 1
    marker_options = [None] if next_color is None else _marker_candidates(seq)
    if next_color is not None and not marker_options:
        return False

    positions = [(r0, c0) for r0 in range(1, h - path_h) for c0 in range(1, w - path_w)]
    random.shuffle(positions)
    for r0, c0 in positions:
        placed_seq = [(int(r + r0), int(c + c0)) for r, c in seq]
        color_map = _segment_color_map(placed_seq, color)
        for marker_local in marker_options:
            local_map = dict(color_map)
            if marker_local is not None:
                marker = (int(marker_local[0] + r0), int(marker_local[1] + c0))
                if not _is_inner(marker, grid.shape):
                    continue
                local_map[marker] = int(next_color)
            if _can_place_object(grid.shape, occupied, local_map.keys()):
                _stamp_object(grid, occupied, local_map)
                return True
    return False


def _try_place_end_port(grid, occupied, end_side):
    h, w = grid.shape
    if end_side in ("left", "right"):
        cells = [(r, 0 if end_side == "left" else w - 1) for r in range(1, h - 1)]
    else:
        cells = [(0 if end_side == "top" else h - 1, c) for c in range(1, w - 1)]
    random.shuffle(cells)
    for cell in cells:
        if _can_place_object(grid.shape, occupied, [cell]):
            _stamp_object(grid, occupied, {cell: GREY})
            return True
    return False


def _gen_attempt():
    start_side = random.choice(["left", "right", "top", "bottom"])
    end_side = OPPOSITE_SIDE[start_side]
    start_dir = SIDE_TO_INWARD[start_side]
    end_dir = SIDE_TO_OUTWARD[end_side]

    n_segments = random.randint(*GEN_SEGMENT_COUNT_RANGE)
    step_counts = [random.randint(*GEN_SEGMENT_STEP_RANGE) for _ in range(n_segments)]
    total_steps = sum(step_counts)
    if total_steps > GEN_TOTAL_STEPS_MAX:
        return None

    path = _generate_path(total_steps, start_dir, end_dir)
    if path is None:
        return None

    dirs = _step_dirs(path)
    if sum(dirs[i] != dirs[i - 1] for i in range(1, len(dirs))) == 0:
        return None

    rows = [r for r, _ in path]
    cols = [c for _, c in path]
    if max(rows) - min(rows) + 1 > GEN_MAX_OUTPUT_SPAN or max(cols) - min(cols) + 1 > GEN_MAX_OUTPUT_SPAN:
        return None

    segments = _segment_sequences(path, step_counts)
    colors = random.sample(PATH_COLORS, n_segments)
    expected = _path_output(path)

    h = random.randint(*GEN_GRID_H_RANGE)
    w = random.randint(*GEN_GRID_W_RANGE)
    inp = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    rotated = {}
    for idx, seq in enumerate(segments):
        rotations = [0, 1, 2, 3]
        random.shuffle(rotations)
        rotated[idx] = [_rotate_sequence(seq, rot) for rot in rotations]

    first_ok = False
    for seq in rotated[0]:
        if _try_place_first(inp, occupied, seq, colors[0], colors[1] if n_segments > 1 else None, start_side):
            first_ok = True
            break
    if not first_ok:
        return None

    middle_indices = list(range(1, n_segments))
    random.shuffle(middle_indices)
    middle_indices.sort(key=lambda i: -len(segments[i]))
    for idx in middle_indices:
        next_color = colors[idx + 1] if idx + 1 < n_segments else None
        placed = False
        for seq in rotated[idx]:
            if _try_place_free(inp, occupied, seq, colors[idx], next_color):
                placed = True
                break
        if not placed:
            return None

    if not _try_place_end_port(inp, occupied, end_side):
        return None

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample for task 7")
