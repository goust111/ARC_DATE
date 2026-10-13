import random
from collections import Counter, deque

import numpy as np


DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]
DIRS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]
DIR_TO_INDEX = {d: i for i, d in enumerate(DIRS4)}

PATH_COLOR = 5
BG_CHOICES = [0, 8, 9]
PITCH_CHOICES = [3, 4]
MARGIN_CHOICES = [3, 4]
COARSE_SIZE_RANGE = (4, 6)
SEGMENT_COUNT_RANGE = (3, 5)
MAX_COARSE_SEG_LEN = 3
STAMP_ATTEMPTS = 200
PATH_ATTEMPTS = 200
GEN_ATTEMPTS = 400


def _in_bounds(shape, r, c):
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors(r, c, connectivity=4):
    dirs = DIRS4 if connectivity == 4 else DIRS8
    for dr, dc in dirs:
        yield r + dr, c + dc


def _dominant_color(grid):
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask, connectivity=4):
    mask = np.asarray(mask, dtype=bool)
    visited = np.zeros_like(mask, dtype=bool)
    comps = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if visited[r, c]:
            continue
        queue = deque([(r, c)])
        visited[r, c] = True
        cells = []
        while queue:
            cr, cc = queue.popleft()
            cells.append((cr, cc))
            for nr, nc in _neighbors(cr, cc, connectivity):
                if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    queue.append((nr, nc))
        coords = np.array(cells, dtype=int)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        comps.append(
            {
                "coords": coords,
                "size": int(len(cells)),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
            }
        )
    return comps


def _path_endpoints(path_mask):
    comps = _components(path_mask, connectivity=4)
    if len(comps) != 1:
        return None
    endpoints = []
    for r, c in comps[0]["coords"]:
        deg = 0
        for nr, nc in _neighbors(int(r), int(c), 4):
            if _in_bounds(path_mask.shape, nr, nc) and path_mask[nr, nc]:
                deg += 1
        if deg == 1:
            endpoints.append((int(r), int(c)))
        elif deg != 2:
            return None
    if len(endpoints) != 2:
        return None
    return endpoints


def _trace_path(path_mask, start):
    total = int(path_mask.sum())
    path = [start]
    prev = None
    curr = start
    while True:
        options = []
        for nr, nc in _neighbors(curr[0], curr[1], 4):
            if not _in_bounds(path_mask.shape, nr, nc) or not path_mask[nr, nc]:
                continue
            if prev is not None and (nr, nc) == prev:
                continue
            options.append((nr, nc))
        if not options:
            break
        assert len(options) == 1, "path must not branch"
        nxt = options[0]
        path.append(nxt)
        prev, curr = curr, nxt
    assert len(path) == total, "path must be fully traversable from the stamp endpoint"
    return path


def _touches(mask, cell):
    r, c = cell
    for nr, nc in _neighbors(r, c, 4):
        if _in_bounds(mask.shape, nr, nc) and mask[nr, nc]:
            return True
    return False


def _find_structures(grid):
    bg = _dominant_color(grid)
    fg_colors = [int(c) for c in np.unique(grid) if int(c) != bg]
    counts = Counter(map(int, grid[grid != bg].flat))
    ordered_colors = sorted(fg_colors, key=lambda color: (-counts[color], color))

    for path_color in ordered_colors:
        colored_mask = (grid != bg) & (grid != path_color)
        colored_components = _components(colored_mask, connectivity=8)
        for idx, comp in enumerate(colored_components):
            if comp["size"] < 2:
                continue
            r0, c0, r1, c1 = comp["bbox"]
            if (r1 - r0 + 1, c1 - c0 + 1) != (3, 3):
                continue
            if any(other["size"] != 1 for j, other in enumerate(colored_components) if j != idx):
                continue

            stamp_mask = np.zeros_like(colored_mask, dtype=bool)
            rows = comp["coords"][:, 0]
            cols = comp["coords"][:, 1]
            stamp_mask[rows, cols] = True

            path_mask = (grid != bg) & ~stamp_mask
            endpoints = _path_endpoints(path_mask)
            if endpoints is None:
                continue
            touching = [pt for pt in endpoints if _touches(stamp_mask, pt)]
            if len(touching) != 1:
                continue
            return {
                "bg": bg,
                "path_color": path_color,
                "stamp_bbox": (r0, c0, r1, c1),
                "stamp_mask": stamp_mask,
                "path_mask": path_mask,
                "start": touching[0],
            }

    raise AssertionError("failed to infer stamp and path structure")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the endpoint stamp onto marked bends of the orthogonal path, rotating it cumulatively at every turn.

    Input structure:
    - The grid has a dominant background, one single-pixel-wide orthogonal polyline, and one adjacent 3x3 colored stamp.
    - Most path cells share one base path color, while some bend cells are recolored as singleton markers.
    - The stamp uses non-background, non-path colors and is the only multi-pixel colored cluster outside the path base color.

    Rule:
    - Find the path endpoint touching the stamp and traverse the whole polyline from that endpoint.
    - Whenever the path turns left, rotate the current stamp state 90 degrees counterclockwise; whenever it turns right, rotate it 90 degrees clockwise.
    - Only at marked bends, place a copy of the current rotated stamp centered on that bend and recolor every non-background stamp cell to the marker color.
    - Keep the original stamp and the path unchanged except where a placed copy covers them.

    Key insight:
    - Marker colors say where to print and what color to use, but the orientation at each print depends on the full history of earlier bends, including unmarked ones.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    info = _find_structures(g)
    r0, c0, r1, c1 = info["stamp_bbox"]
    stamp = g[r0 : r1 + 1, c0 : c1 + 1]
    stamp_ink = stamp != info["bg"]
    path = _trace_path(info["path_mask"], info["start"])
    assert int(g[path[0]]) == info["path_color"]
    assert int(g[path[-1]]) == info["path_color"]

    out = g.copy()
    rotation = 0
    for i in range(1, len(path) - 1):
        prev_r, prev_c = path[i - 1]
        cur_r, cur_c = path[i]
        next_r, next_c = path[i + 1]
        d1 = (cur_r - prev_r, cur_c - prev_c)
        d2 = (next_r - cur_r, next_c - cur_c)
        assert d1 in DIR_TO_INDEX and d2 in DIR_TO_INDEX

        cell_color = int(g[cur_r, cur_c])
        if d1 == d2:
            assert cell_color == info["path_color"], "only bends may carry markers"
            continue

        turn = d1[0] * d2[1] - d1[1] * d2[0]
        assert turn != 0, "bend must be a 90-degree turn"
        rotation = (rotation + (1 if turn > 0 else -1)) % 4
        if cell_color == info["path_color"]:
            continue

        patch = np.rot90(stamp_ink, rotation)
        top = cur_r - 1
        left = cur_c - 1
        assert 0 <= top and top + 3 <= g.shape[0]
        assert 0 <= left and left + 3 <= g.shape[1]
        for rr in range(3):
            for cc in range(3):
                if patch[rr, cc]:
                    out[top + rr, left + cc] = cell_color
    return out


def _connected(mask):
    return len(_components(mask, connectivity=4)) == 1


def _make_stamp(bg):
    palette = [c for c in range(10) if c not in (bg, PATH_COLOR)]
    for _ in range(STAMP_ATTEMPTS):
        target_size = random.randint(4, 6)
        cells = {(1, 1)}
        while len(cells) < target_size:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors(r, c, 4):
                    if 0 <= nr < 3 and 0 <= nc < 3 and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != target_size:
            continue

        mask = np.zeros((3, 3), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if not _connected(mask):
            continue
        if not mask[0].any() or not mask[2].any() or not mask[:, 0].any() or not mask[:, 2].any():
            continue
        if any(np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3)):
            continue

        side_dirs = []
        if mask[0, 1]:
            side_dirs.append((-1, 0))
        if mask[1, 2]:
            side_dirs.append((0, 1))
        if mask[2, 1]:
            side_dirs.append((1, 0))
        if mask[1, 0]:
            side_dirs.append((0, -1))
        if not side_dirs:
            continue

        color_count = random.randint(2, min(3, int(mask.sum())))
        chosen = random.sample(palette, color_count)
        assignments = chosen + [random.choice(chosen) for _ in range(int(mask.sum()) - color_count)]
        random.shuffle(assignments)

        stamp = np.full((3, 3), bg, dtype=int)
        coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
        random.shuffle(coords)
        for (r, c), color in zip(coords, assignments):
            stamp[r, c] = color
        return stamp, random.choice(side_dirs)
    return None


def _segment_cells(curr, direction, distance):
    dr, dc = direction
    return [(curr[0] + dr * step, curr[1] + dc * step) for step in range(1, distance + 1)]


def _try_segment(curr, direction, distance, occupied, forbidden, shape):
    cells = _segment_cells(curr, direction, distance)
    prev = curr
    for cell in cells:
        r, c = cell
        if not _in_bounds(shape, r, c):
            return None
        if cell in occupied or cell in forbidden:
            return None
        for nr, nc in _neighbors(r, c, 4):
            if (nr, nc) in occupied and (nr, nc) != prev:
                return None
        prev = cell
    return cells


def _max_coarse_len(curr, direction, shape, pitch):
    dr, dc = direction
    if dr == -1:
        return min(MAX_COARSE_SEG_LEN, curr[0] // pitch)
    if dr == 1:
        return min(MAX_COARSE_SEG_LEN, (shape[0] - 1 - curr[0]) // pitch)
    if dc == -1:
        return min(MAX_COARSE_SEG_LEN, curr[1] // pitch)
    return min(MAX_COARSE_SEG_LEN, (shape[1] - 1 - curr[1]) // pitch)


def _build_path(shape, start, dir0, forbidden, pitch, n_segments):
    occupied = {start}
    ordered = [start]
    vertices = [start]

    def backtrack(curr, dir_idx, segments_left):
        max_len = _max_coarse_len(curr, DIRS4[dir_idx], shape, pitch)
        lengths = list(range(1, max_len + 1))
        random.shuffle(lengths)
        for coarse_len in lengths:
            distance = coarse_len * pitch
            cells = _try_segment(curr, DIRS4[dir_idx], distance, occupied, forbidden, shape)
            if cells is None:
                continue

            for cell in cells:
                occupied.add(cell)
                ordered.append(cell)
            vertices.append(cells[-1])

            if segments_left == 1:
                return True

            turns = [-1, 1]
            random.shuffle(turns)
            for turn in turns:
                next_dir_idx = (dir_idx + turn) % 4
                if backtrack(cells[-1], next_dir_idx, segments_left - 1):
                    return True

            vertices.pop()
            for _ in cells:
                occupied.remove(ordered.pop())
        return False

    if backtrack(start, DIR_TO_INDEX[dir0], n_segments):
        return ordered, vertices
    return None


def _stamp_forbidden(center, start):
    forbidden = set()
    box = set()
    for rr in range(center[0] - 1, center[0] + 2):
        for cc in range(center[1] - 1, center[1] + 2):
            box.add((rr, cc))
            forbidden.add((rr, cc))
    for r, c in list(box):
        for nr, nc in _neighbors(r, c, 4):
            if (nr, nc) != start:
                forbidden.add((nr, nc))
    forbidden.discard(start)
    return forbidden


def _choose_marked_indices(n_corners):
    if n_corners == 1:
        return {0}
    mode = random.choice(["all", "mixed", "hidden"])
    all_indices = list(range(n_corners))
    if mode == "all":
        return set(all_indices)
    if mode == "mixed":
        count = random.randint(1, n_corners - 1)
        return set(random.sample(all_indices, count))

    for _ in range(50):
        count = random.randint(1, n_corners - 1)
        marked = set(random.sample(all_indices, count))
        if any(any(i < j and i not in marked for i in all_indices) for j in marked):
            return marked
    return {n_corners - 1}


def _embed(grid, bg):
    h, w = grid.shape
    max_top = 30 - h
    max_left = 30 - w
    top = random.randint(0, max_top)
    bottom = random.randint(0, 30 - h - top)
    left = random.randint(0, max_left)
    right = random.randint(0, 30 - w - left)
    out = np.full((h + top + bottom, w + left + right), bg, dtype=int)
    out[top : top + h, left : left + w] = grid
    return out


def _gen_attempt():
    bg = random.choice(BG_CHOICES)
    stamp_result = _make_stamp(bg)
    if stamp_result is None:
        return None
    stamp, dir0 = stamp_result

    pitch = random.choice(PITCH_CHOICES)
    margin = random.choice(MARGIN_CHOICES)
    coarse_h = random.randint(*COARSE_SIZE_RANGE)
    coarse_w = random.randint(*COARSE_SIZE_RANGE)
    base_h = 2 * margin + pitch * (coarse_h - 1) + 1
    base_w = 2 * margin + pitch * (coarse_w - 1) + 1
    shape = (base_h, base_w)

    valid_starts = []
    for sr in range(coarse_h):
        for sc in range(coarse_w):
            nr = sr + dir0[0]
            nc = sc + dir0[1]
            if 0 <= nr < coarse_h and 0 <= nc < coarse_w:
                valid_starts.append((sr, sc))
    if not valid_starts:
        return None

    for _ in range(PATH_ATTEMPTS):
        start_coarse = random.choice(valid_starts)
        start = (margin + pitch * start_coarse[0], margin + pitch * start_coarse[1])
        stamp_center = (start[0] - 2 * dir0[0], start[1] - 2 * dir0[1])
        forbidden = _stamp_forbidden(stamp_center, start)
        if any(not _in_bounds(shape, r, c) for r, c in forbidden):
            continue

        n_segments = random.randint(*SEGMENT_COUNT_RANGE)
        path_result = _build_path(shape, start, dir0, forbidden, pitch, n_segments)
        if path_result is None:
            continue
        ordered, vertices = path_result
        corners = vertices[1:-1]
        if not corners:
            continue

        marked_indices = _choose_marked_indices(len(corners))
        marker_palette = [c for c in range(10) if c not in (bg, PATH_COLOR)]
        marker_colors = {idx: random.choice(marker_palette) for idx in marked_indices}

        grid = np.full(shape, bg, dtype=int)
        for r, c in ordered:
            grid[r, c] = PATH_COLOR
        for idx, (r, c) in enumerate(corners):
            if idx in marker_colors:
                grid[r, c] = marker_colors[idx]

        top = stamp_center[0] - 1
        left = stamp_center[1] - 1
        if top < 0 or left < 0 or top + 3 > shape[0] or left + 3 > shape[1]:
            continue
        for rr in range(3):
            for cc in range(3):
                if stamp[rr, cc] != bg:
                    if grid[top + rr, left + cc] != bg:
                        break
                    grid[top + rr, left + cc] = stamp[rr, cc]
            else:
                continue
            break
        else:
            grid = _embed(grid, bg)
            try:
                out = main(grid)
            except AssertionError:
                continue
            if np.array_equal(out, grid):
                continue
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
