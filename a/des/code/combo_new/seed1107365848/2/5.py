import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
RIGHT = (0, 1)
DOWN = (1, 0)
LEFT = (0, -1)
UP = (-1, 0)
DIRS4 = (UP, DOWN, LEFT, RIGHT)
DIR_TO_ROT = {
    RIGHT: 0,
    DOWN: 3,  # clockwise 90
    LEFT: 2,
    UP: 1,    # clockwise 270
}


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIRS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _turn_left(direction):
    dr, dc = direction
    return -dc, dr


def _turn_right(direction):
    dr, dc = direction
    return dc, -dr


def _parse_legend(grid: np.ndarray):
    grey_pts = np.argwhere(grid == GREY)
    assert grey_pts.size > 0, "legend must contain grey grid lines"
    r0, c0 = map(int, grey_pts.min(axis=0))
    r1, c1 = map(int, grey_pts.max(axis=0))
    box = grid[r0 : r1 + 1, c0 : c1 + 1]

    hlines = [r for r in range(box.shape[0]) if np.all(box[r, :] == GREY)]
    vlines = [c for c in range(box.shape[1]) if np.all(box[:, c] == GREY)]
    assert len(hlines) >= 2 and len(vlines) >= 2, "legend must be a grey-framed grid"

    row_sizes = [hlines[i + 1] - hlines[i] - 1 for i in range(len(hlines) - 1)]
    col_sizes = [vlines[i + 1] - vlines[i] - 1 for i in range(len(vlines) - 1)]
    assert row_sizes and col_sizes
    cell_size = int(row_sizes[0])
    assert cell_size > 0
    assert all(size == cell_size for size in row_sizes), "legend rows must be equally sized"
    assert all(size == cell_size for size in col_sizes), "legend cols must be equally sized"

    templates = {}
    for i in range(len(hlines) - 1):
        for j in range(len(vlines) - 1):
            cell = box[hlines[i] + 1 : hlines[i + 1], vlines[j] + 1 : vlines[j + 1]].copy()
            colors = sorted(set(map(int, cell.ravel())) - {BLACK})
            assert len(colors) == 1, "each legend cell must contain exactly one colored template"
            color = int(colors[0])
            assert color != GREY
            assert color not in templates, "legend colors must be unique"
            templates[color] = cell
    return templates, (r0, c0, r1, c1), cell_size


def _find_path_components(grid: np.ndarray, legend_bbox):
    h, w = grid.shape
    mask = (grid != BLACK) & (grid != GREY)
    r0, c0, r1, c1 = legend_bbox
    mask[r0 : r1 + 1, c0 : c1 + 1] = False

    visited = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            visited[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if visited[nr, nc] or not mask[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    q.append((nr, nc))
            comps.append((color, coords))
    comps.sort(key=lambda item: min(item[1]))
    return comps


def _order_path(coords):
    points = sorted({(int(r), int(c)) for r, c in coords})
    neigh = {}
    for r, c in points:
        cur = []
        for dr, dc in (RIGHT, DOWN, LEFT, UP):
            nxt = (r + dr, c + dc)
            if nxt in points:
                cur.append(nxt)
        neigh[(r, c)] = cur

    start = min(points)
    endpoints = [pt for pt, ns in neigh.items() if len(ns) == 1]
    if len(points) > 1 and len(neigh[start]) != 1 and endpoints:
        start = min(endpoints)

    order = [start]
    prev = None
    cur = start
    while True:
        nxts = [pt for pt in neigh[cur] if pt != prev]
        if not nxts:
            break
        nxt = nxts[0] if len(nxts) == 1 else min(nxts)
        order.append(nxt)
        prev, cur = cur, nxt
        if len(order) > len(points):
            break
    assert len(order) == len(points), "path must be a simple non-branching polyline"
    return order


def _point_directions(path):
    if len(path) == 1:
        return [RIGHT]
    dirs = []
    for i in range(len(path) - 1):
        dr = int(path[i + 1][0] - path[i][0])
        dc = int(path[i + 1][1] - path[i][1])
        dirs.append((dr, dc))
    return dirs + [dirs[-1]]


def _paint_patch(out: np.ndarray, patch: np.ndarray, center):
    side = int(patch.shape[0])
    radius = side // 2
    cr, cc = map(int, center)
    for r in range(side):
        rr = cr - radius + r
        if not (0 <= rr < out.shape[0]):
            continue
        for c in range(side):
            cc2 = cc - radius + c
            if not (0 <= cc2 < out.shape[1]):
                continue
            val = int(patch[r, c])
            if val != BLACK:
                out[rr, cc2] = val


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each colored polyline with rotated copies of its matching legend template.

    Input structure:
    - Background is black.
    - A grey-framed legend block on the left is subdivided into equal square cells by grey lines.
    - Each legend cell contains a small asymmetric one-color template.
    - Outside the legend are several 4-connected single-pixel-wide colored polylines whose colors match legend templates.

    Rule:
    - Read the legend as a color -> square template mapping; the cell side length is the template spacing.
    - For each path color, find every same-color path component and choose its start as the topmost path pixel, breaking ties by leftmost column.
    - The path's first segment sets the initial template rotation; then every template copy follows the path direction, updating at each 90-degree turn.
    - Clear the original path pixels and stamp the rotated template at path indices 0, s, 2s, ... where s is the legend cell side length, centering each copy on the sampled path pixel and clipping anything outside the grid.

    Key insight:
    - The colored path is only a control trace: its color selects the template, its first segment selects the initial orientation, and later left/right turns rotate subsequent template copies.
    """
    g = np.asarray(grid, dtype=int)
    templates, legend_bbox, cell_size = _parse_legend(g)

    out = g.copy()
    for color, coords in _find_path_components(g, legend_bbox):
        assert color in templates, "every path color must appear in the legend"
        for r, c in coords:
            out[r, c] = BLACK

        path = _order_path(coords)
        directions = _point_directions(path)
        template = templates[color]
        for idx in range(0, len(path), cell_size):
            direction = directions[min(idx, len(directions) - 1)]
            patch = np.rot90(template, DIR_TO_ROT[direction])
            _paint_patch(out, patch, path[idx])
    return out


# --------------- generator (codex) ---------------

SIDE_CHOICES = (3, 5)
TEMPLATE_COUNT_BY_SIDE = {
    3: (2, 3),
    5: (2, 3),
}
TURN_COUNT_CHOICES = (0, 1, 2)
GEN_RETRIES = 300
PATH_BUILD_RETRIES = 120
PATH_PLACE_RETRIES = 120
USABLE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def _same_up_to_rotation(mask_a: np.ndarray, mask_b: np.ndarray) -> bool:
    return any(np.array_equal(mask_a, np.rot90(mask_b, k)) for k in range(4))


def _make_template_mask(side: int, rng: random.Random) -> np.ndarray:
    center = (side // 2, side // 2)
    for _ in range(200):
        target = rng.randint(side, min(side * side - 1, side + 4))
        mask = np.zeros((side, side), dtype=bool)
        mask[center] = True
        cells = {center}
        frontier = set()
        frontier.update(
            (nr, nc)
            for nr, nc in _neighbors4(center[0], center[1], side, side)
        )

        while len(cells) < target and frontier:
            nr, nc = rng.choice(tuple(frontier))
            frontier.remove((nr, nc))
            if mask[nr, nc]:
                continue
            mask[nr, nc] = True
            cells.add((nr, nc))
            for ar, ac in _neighbors4(nr, nc, side, side):
                if not mask[ar, ac]:
                    frontier.add((ar, ac))

        rots = [np.rot90(mask, k) for k in range(4)]
        if len({rot.tobytes() for rot in rots}) < 4:
            continue
        if any(np.array_equal(mask, np.fliplr(rot)) for rot in rots):
            continue
        if int(mask.sum()) < 3:
            continue
        return mask
    raise RuntimeError("failed to build an asymmetric template")


def _build_templates(side: int, colors, rng: random.Random):
    templates = {}
    masks = []
    for color in colors:
        for _ in range(200):
            mask = _make_template_mask(side, rng)
            if any(_same_up_to_rotation(mask, other) for other in masks):
                continue
            cell = np.zeros((side, side), dtype=int)
            cell[mask] = int(color)
            templates[int(color)] = cell
            masks.append(mask)
            break
        else:
            raise RuntimeError("failed to build distinct templates")
    return templates


def _legend_layout(num_templates: int, side: int, rng: random.Random):
    if num_templates == 2:
        return 2, 1
    if num_templates == 3:
        return 3, 1
    return 4, 1


def _draw_legend(grid: np.ndarray, templates: dict[int, np.ndarray], r0: int, c0: int, rows: int, cols: int, order):
    side = next(iter(templates.values())).shape[0]
    legend_h = rows * side + rows + 1
    legend_w = cols * side + cols + 1
    for r in range(legend_h):
        for c in range(legend_w):
            if r % (side + 1) == 0 or c % (side + 1) == 0:
                grid[r0 + r, c0 + c] = GREY
    idx = 0
    for i in range(rows):
        for j in range(cols):
            color = order[idx]
            idx += 1
            rr = r0 + 1 + i * (side + 1)
            cc = c0 + 1 + j * (side + 1)
            grid[rr : rr + side, cc : cc + side] = templates[color]


def _build_relative_path(side: int, turns: int, initial_dir, rng: random.Random, forced_signs=None):
    max_units = 2 if side == 5 or turns >= 2 else 3
    for _ in range(PATH_BUILD_RETRIES):
        directions = [initial_dir]
        lengths = [rng.randint(min(2, max_units) if turns == 0 else 1, max_units)]
        signs = []
        cur_r = initial_dir[0] * lengths[0] * side
        cur_c = initial_dir[1] * lengths[0] * side
        cur_dir = initial_dir

        valid = True
        for turn_idx in range(turns):
            units = rng.randint(1, max_units)
            forced = None
            if forced_signs is not None and turn_idx < len(forced_signs):
                forced = forced_signs[turn_idx]

            candidates = [forced] if forced is not None else ["L", "R"]
            if forced is None:
                rng.shuffle(candidates)

            chosen = None
            for sign in candidates:
                nxt = _turn_left(cur_dir) if sign == "L" else _turn_right(cur_dir)
                end_r = cur_r + nxt[0] * units * side
                end_c = cur_c + nxt[1] * units * side
                if end_r < 0 or end_c < 0:
                    continue
                chosen = (sign, nxt, units, end_r, end_c)
                break
            if chosen is None:
                valid = False
                break

            sign, nxt, units, cur_r, cur_c = chosen
            signs.append(sign)
            directions.append(nxt)
            lengths.append(units)
            cur_dir = nxt

        if not valid:
            continue

        pts = [(0, 0)]
        seen = {(0, 0)}
        ok = True
        for direction, units in zip(directions, lengths):
            for _ in range(units * side):
                nr = pts[-1][0] + direction[0]
                nc = pts[-1][1] + direction[1]
                if nr < 0 or nc < 0 or (nr, nc) in seen:
                    ok = False
                    break
                pts.append((nr, nc))
                seen.add((nr, nc))
            if not ok:
                break
        if not ok:
            continue
        max_r = max(r for r, _ in pts)
        max_c = max(c for _, c in pts)
        limit = 8 if side == 3 else 9
        if max_r == 0 and max_c < side:
            continue
        if max_r > limit or max_c > limit:
            continue
        centers = [pts[i] for i in range(0, len(pts), side)]
        return {
            "points": pts,
            "centers": centers,
            "turns": turns,
            "signs": signs,
            "initial_dir": initial_dir,
        }
    return None


def _reserved_cells(centers, side: int, h: int, w: int, pad: int = 0):
    radius = side // 2
    cells = set()
    for cr, cc in centers:
        for r in range(cr - radius - pad, cr + radius + pad + 1):
            if not (0 <= r < h):
                continue
            for c in range(cc - radius - pad, cc + radius + pad + 1):
                if 0 <= c < w:
                    cells.add((r, c))
    return cells


def _place_path(path, side: int, h: int, w: int, main_c0: int, occupied, rng: random.Random):
    pts = path["points"]
    centers = path["centers"]
    max_r = max(r for r, _ in pts)
    max_c = max(c for _, c in pts)
    min_box_left = min(c - side // 2 for _, c in centers)
    sr_low, sr_high = 0, h - 1 - max_r
    sc_low = max(main_c0, main_c0 - min_box_left)
    sc_high = w - 1 - max_c
    if sr_low > sr_high or sc_low > sc_high:
        return None

    row_choices = list(range(sr_low, sr_high + 1))
    col_choices = list(range(sc_low, sc_high + 1))
    for _ in range(PATH_PLACE_RETRIES):
        if len(row_choices) == 1:
            sr = row_choices[0]
        elif rng.random() < 0.35:
            sr = row_choices[0] if rng.random() < 0.5 else row_choices[-1]
        else:
            sr = rng.choice(row_choices)

        if len(col_choices) == 1:
            sc = col_choices[0]
        elif rng.random() < 0.25:
            sc = col_choices[-1]
        else:
            sc = rng.choice(col_choices)

        t_points = [(sr + r, sc + c) for r, c in pts]
        if any(c < main_c0 for _, c in t_points):
            continue
        t_centers = [(sr + r, sc + c) for r, c in centers]
        reserved = _reserved_cells(t_centers, side, h, w, pad=0)
        if reserved & occupied:
            continue
        clipped = any(
            cr - side // 2 < 0 or cr + side // 2 >= h or cc + side // 2 >= w
            for cr, cc in t_centers
        )
        return {
            "points": t_points,
            "centers": t_centers,
            "turns": path["turns"],
            "signs": path["signs"],
            "initial_dir": path["initial_dir"],
            "clipped": clipped,
            "reserved": reserved,
        }
    return None


def _scatter_slack(parts: int, slack: int, rng: random.Random):
    gaps = [0] * parts
    for _ in range(slack):
        gaps[rng.randrange(parts)] += 1
    return gaps


def _gen_attempt(rng: random.Random):
    side = rng.choice(SIDE_CHOICES)
    num_templates = rng.choice(TEMPLATE_COUNT_BY_SIDE[side])
    colors = rng.sample(USABLE_COLORS, num_templates)
    templates = _build_templates(side, colors, rng)

    legend_rows, legend_cols = _legend_layout(num_templates, side, rng)
    legend_h = legend_rows * side + legend_rows + 1
    legend_w = legend_cols * side + legend_cols + 1

    min_h = max(legend_h + 1, 16 if side == 3 else 18)
    min_w = max(legend_w + side + 7, 20 if side == 3 else 22)
    if min_h > 30 or min_w > 30:
        return None

    h = rng.randint(min_h, 30)
    w = rng.randint(min_w, 30)
    legend_r0 = rng.randint(0, h - legend_h)
    legend_c0 = rng.choice([0, 1]) if w - legend_w >= 2 else 0
    main_c0 = legend_c0 + legend_w + 2
    if main_c0 + side >= w:
        return None

    num_paths = min(num_templates, 3 if side == 3 else 2)
    if side == 3 and rng.random() < 0.25:
        num_paths += 1
    turn_counts = [rng.choice(TURN_COUNT_CHOICES) for _ in range(num_paths)]
    if all(t == 0 for t in turn_counts):
        turn_counts[rng.randrange(num_paths)] = 1

    initial_dirs = [rng.choice((RIGHT, DOWN)) for _ in range(num_paths)]
    if num_paths >= 2 and len(set(initial_dirs)) == 1:
        initial_dirs[0] = RIGHT
        initial_dirs[1] = DOWN

    forced_signs = [None for _ in range(num_paths)]
    turning = [i for i, t in enumerate(turn_counts) if t > 0]
    if len(turning) >= 2:
        right_idx = next((i for i in turning if initial_dirs[i] == RIGHT), turning[0])
        left_idx = next((i for i in turning if initial_dirs[i] == DOWN and i != right_idx), None)
        if left_idx is None:
            left_idx = next((i for i in turning if i != right_idx), turning[0])
        forced_signs[right_idx] = ["R"] + [None] * (turn_counts[right_idx] - 1)
        forced_signs[left_idx] = ["L"] + [None] * (turn_counts[left_idx] - 1)
    elif len(turning) == 1 and turn_counts[turning[0]] >= 2:
        if initial_dirs[turning[0]] == RIGHT:
            forced_signs[turning[0]] = ["R", "L"] + [None] * (turn_counts[turning[0]] - 2)
        else:
            forced_signs[turning[0]] = ["L", "R"] + [None] * (turn_counts[turning[0]] - 2)

    path_colors = colors[:]
    while len(path_colors) < num_paths:
        path_colors.append(rng.choice(colors))
    rng.shuffle(path_colors)

    rel_paths = []
    for idx, (color, turns, initial_dir) in enumerate(zip(path_colors, turn_counts, initial_dirs)):
        rel = _build_relative_path(side, turns, initial_dir, rng, forced_signs=forced_signs[idx])
        if rel is None:
            return None
        max_center_r = max(r for r, _ in rel["centers"])
        max_center_c = max(c for _, c in rel["centers"])
        max_path_c = max(c for _, c in rel["points"])
        rel["color"] = int(color)
        rel["need_h"] = max_center_r + side
        rel["max_center_c"] = max_center_c
        rel["max_path_c"] = max_path_c
        rel_paths.append(rel)

    lane_gap = 1
    required_h = sum(path["need_h"] for path in rel_paths) + lane_gap * (len(rel_paths) - 1)
    required_w = max(path["max_path_c"] + side for path in rel_paths)
    required_h = max(required_h, legend_h)
    required_w = max(required_w + legend_w + side + 4, legend_w + side + 7)
    if required_h > 30 or required_w > 30:
        return None

    h = rng.randint(required_h, 30)
    w = rng.randint(required_w, 30)
    legend_r0 = rng.randint(0, h - legend_h)
    legend_c0 = rng.choice([0, 1]) if w - legend_w >= 2 else 0
    main_c0 = legend_c0 + legend_w + 2
    if main_c0 + side >= w:
        return None

    slack_h = h - (sum(path["need_h"] for path in rel_paths) + lane_gap * (len(rel_paths) - 1))
    extra_gaps = _scatter_slack(len(rel_paths) + 1, slack_h, rng)

    placed = []
    cursor = extra_gaps[0]
    for idx, rel in enumerate(rel_paths):
        sr = cursor + side // 2
        sc_low = main_c0 + side // 2
        sc_high_inside = w - 1 - (rel["max_center_c"] + side // 2)
        sc_high_path = w - 1 - rel["max_path_c"]
        if sc_low > sc_high_path:
            return None

        allow_clip = rng.random() < 0.3 and sc_high_inside < sc_high_path
        if allow_clip:
            clip_low = max(sc_low, sc_high_inside + 1)
            sc = rng.randint(clip_low, sc_high_path)
        else:
            if sc_low > sc_high_inside:
                return None
            sc = rng.randint(sc_low, sc_high_inside)

        points = [(sr + r, sc + c) for r, c in rel["points"]]
        centers = [(sr + r, sc + c) for r, c in rel["centers"]]
        placed.append(
            {
                "color": rel["color"],
                "points": points,
                "centers": centers,
                "turns": rel["turns"],
                "signs": rel["signs"],
                "initial_dir": rel["initial_dir"],
            }
        )
        cursor += rel["need_h"] + lane_gap + extra_gaps[idx + 1]

    grid = np.zeros((h, w), dtype=int)
    legend_order = colors[:]
    rng.shuffle(legend_order)
    _draw_legend(grid, templates, legend_r0, legend_c0, legend_rows, legend_cols, legend_order)
    for item in placed:
        for r, c in item["points"]:
            grid[r, c] = item["color"]
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
