from collections import Counter

import numpy as np

from arc_utils import colors_present, components_mask, get_bg, neighbors4, paste
import random


YELLOW = 4
RED = 2


def _trace_cycle(coords: np.ndarray) -> list[tuple[int, int]]:
    pts = {tuple(map(int, p)) for p in coords}
    for r, c in pts:
        degree = sum((nr, nc) in pts for nr, nc in neighbors4(r, c))
        assert degree == 2, (r, c, degree)

    start = min(pts)
    neighbors = sorted((nr, nc) for nr, nc in neighbors4(*start) if (nr, nc) in pts)
    assert len(neighbors) == 2

    path = [start]
    prev = start
    cur = neighbors[0]
    while cur != start:
        path.append(cur)
        next_cells = [
            (nr, nc)
            for nr, nc in neighbors4(*cur)
            if (nr, nc) in pts and (nr, nc) != prev
        ]
        assert len(next_cells) == 1, (cur, next_cells)
        prev, cur = cur, next_cells[0]
        assert len(path) <= len(pts)

    assert len(path) == len(pts)
    return path


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each closed orthogonal line loop so convex bend pixels are yellow and concave bend pixels are red.

    Input structure
    - The grid has a dominant background and exactly one foreground color.
    - The foreground forms one or more 4-connected one-pixel-wide closed cycles; every foreground pixel has degree 2.
    Rule
    - Trace each cycle and inspect the turn at each foreground pixel.
    - Straight pixels keep the original foreground color.
    - Bend pixels whose turn direction is the majority direction for that cycle are convex corners and become yellow.
    - Bend pixels whose turn direction is the minority direction for that cycle are concave corners and become red.
    Key insight
    - A simple rectilinear closed boundary has exactly four more convex bends than concave bends, so convexity can be inferred structurally from the majority turn sign within each traced cycle.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    colors = colors_present(g, exclude={bg})
    assert len(colors) == 1, colors
    foreground = colors[0]

    out = g.copy()
    for coords in components_mask(g == foreground, connectivity=4):
        path = _trace_cycle(coords)
        turn_signs: dict[tuple[int, int], int] = {}
        counts: Counter[int] = Counter()
        n = len(path)
        for i, (r, c) in enumerate(path):
            pr, pc = path[i - 1]
            nr, nc = path[(i + 1) % n]
            incoming = (r - pr, c - pc)
            outgoing = (nr - r, nc - c)
            cross = incoming[0] * outgoing[1] - incoming[1] * outgoing[0]
            if cross == 0:
                continue
            sign = 1 if cross > 0 else -1
            turn_signs[(r, c)] = sign
            counts[sign] += 1

        assert turn_signs
        assert abs(counts[1] - counts[-1]) == 4, counts
        convex_sign = 1 if counts[1] > counts[-1] else -1
        for (r, c), sign in turn_signs.items():
            out[r, c] = YELLOW if sign == convex_sign else RED

    return out

# --------------- generator (o5x) ---------------

# Canvas and palette sampling.
GEN_BACKGROUND_COLOR = 0
GEN_FOREGROUND_COLORS, GEN_CYCLE_COUNTS = [1, 3, 5, 6, 7, 8, 9], [2, 3]
GEN_CANVAS_ROW_RANGE, GEN_CANVAS_COL_RANGE = (12, 15), (13, 16)
GEN_THREE_CYCLE_COUNT = 3

# Cycle geometry bounds.
GEN_PATCH_HEIGHT_RANGE, GEN_PATCH_WIDTH_RANGE = (3, 10), (3, 12)
GEN_THREE_CYCLE_PATCH_HEIGHT_RANGE, GEN_THREE_CYCLE_PATCH_WIDTH_RANGE = (3, 7), (3, 9)
GEN_PERIMETER_RANGE, GEN_CONVEX_RANGE, GEN_CONCAVE_RANGE = (8, 42), (4, 12), (0, 8)
GEN_CONVEX_CONCAVE_DELTA = 4

# Notch count and spacing controls.
GEN_NOTCH_COUNTS, GEN_NOTCH_WEIGHTS = [0, 1, 2, 3, 4], [1, 1, 2, 3, 4]
GEN_THREE_CYCLE_NOTCH_COUNTS, GEN_THREE_CYCLE_NOTCH_WEIGHTS = [0, 1, 2], [2, 3, 3]
GEN_MAX_NOTCHES, GEN_MIN_NOTCH_DEPTH = 4, 1
GEN_MIN_NOTCH_CANVAS_SIDE, GEN_MIN_MULTI_NOTCH_HEIGHT = 4, 5
GEN_MULTI_NOTCH_THRESHOLD = 2
GEN_NOTCH_INTERVAL_GAP, GEN_NOTCH_FLAT_PAD, GEN_NOTCH_CLEARANCE_PAD = 2, 2, 1
GEN_PROFILE_CLEARANCE = 2
GEN_NOTCH_CAPACITY_WIDTH_OFFSET, GEN_NOTCH_CAPACITY_STRIDE = 3, 4
GEN_NOTCH_INTERVAL_RULES = {
    "top": {"start_min": 2, "start_stop_margin": 4, "end_stop_margin": 3},
    "bottom": {"start_min": 3, "start_stop_margin": 3, "end_stop_margin": 2},
}

# Placement and retry limits.
GEN_PLACEMENT_MARGIN = 1
GEN_RETRY_LIMIT = 300
GEN_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_OCCUPIED_OFFSETS = [(0, 0), *GEN_DIRS4]
GEN_ROTATION_COUNT, GEN_FLIP_CHOICES = 4, [False, True]


def _gen_append_segment(path, r0, c0, r1, c1):
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        cells = [(r0, c) for c in range(c0, c1 + step, step)]
    else:
        step = 1 if r1 >= r0 else -1
        cells = [(r, c0) for r in range(r0, r1 + step, step)]
    if path and cells and path[-1] == cells[0]:
        cells = cells[1:]
    path.extend(cells)


def _gen_path_from_profiles(top, bottom):
    width = len(top)
    path = []
    _gen_append_segment(path, top[0], 0, top[0], 0)
    for c in range(width - 1):
        _gen_append_segment(path, top[c], c, top[c], c + 1)
        _gen_append_segment(path, top[c], c + 1, top[c + 1], c + 1)
    _gen_append_segment(path, top[-1], width - 1, bottom[-1], width - 1)
    for c in range(width - 1, 0, -1):
        _gen_append_segment(path, bottom[c], c, bottom[c], c - 1)
        _gen_append_segment(path, bottom[c], c - 1, bottom[c - 1], c - 1)
    _gen_append_segment(path, bottom[0], 0, top[0], 0)
    if path and path[-1] == path[0]:
        path.pop()
    return path


def _gen_trace_cycle(mask):
    pts = {tuple(map(int, p)) for p in np.argwhere(mask)}
    if len(pts) < GEN_PERIMETER_RANGE[0]:
        return None
    for r, c in pts:
        degree = sum((r + dr, c + dc) in pts for dr, dc in GEN_DIRS4)
        if degree != 2:
            return None

    start = min(pts)
    neigh = sorted((start[0] + dr, start[1] + dc) for dr, dc in GEN_DIRS4 if (start[0] + dr, start[1] + dc) in pts)
    if len(neigh) != 2:
        return None
    path = [start]
    prev, cur = start, neigh[0]
    while cur != start:
        path.append(cur)
        nxt = [
            (cur[0] + dr, cur[1] + dc)
            for dr, dc in GEN_DIRS4
            if (cur[0] + dr, cur[1] + dc) in pts
            and (cur[0] + dr, cur[1] + dc) != prev
        ]
        if len(nxt) != 1 or len(path) > len(pts):
            return None
        prev, cur = cur, nxt[0]
    return path if len(path) == len(pts) else None


def _gen_cycle_metrics(mask):
    path = _gen_trace_cycle(mask)
    if path is None:
        return None
    counts = {1: 0, -1: 0}
    n = len(path)
    for i, (r, c) in enumerate(path):
        pr, pc = path[i - 1]
        nr, nc = path[(i + 1) % n]
        incoming = (r - pr, c - pc)
        outgoing = (nr - r, nc - c)
        cross = incoming[0] * outgoing[1] - incoming[1] * outgoing[0]
        if cross:
            counts[1 if cross > 0 else -1] += 1
    convex = max(counts.values())
    concave = min(counts.values())
    return len(path), convex, concave


def _gen_notch_options(top, bottom, side, max_depth):
    height = max(bottom) + 1
    width = len(top)
    base = 0 if side == "top" else height - 1
    profile = top if side == "top" else bottom
    options = []
    if height < GEN_MIN_NOTCH_CANVAS_SIDE or width < GEN_MIN_NOTCH_CANVAS_SIDE or max_depth < GEN_MIN_NOTCH_DEPTH:
        return options

    rule = GEN_NOTCH_INTERVAL_RULES[side]
    start_range = range(rule["start_min"], width - rule["start_stop_margin"])
    end_limit = width - rule["end_stop_margin"]

    for start in start_range:
        for end in range(start + 1, end_limit):
            lo = max(0, start - GEN_NOTCH_FLAT_PAD)
            hi = min(width, end + GEN_NOTCH_FLAT_PAD + 1)
            if any(v != base for v in profile[lo:hi]):
                continue

            sep_cols = range(max(0, start - GEN_NOTCH_CLEARANCE_PAD), min(width, end + GEN_NOTCH_CLEARANCE_PAD + 1))
            if side == "top":
                limit = min(bottom[c] - GEN_PROFILE_CLEARANCE for c in sep_cols)
            else:
                limit = min((height - 1) - top[c] - GEN_PROFILE_CLEARANCE for c in sep_cols)
            for depth in range(GEN_MIN_NOTCH_DEPTH, min(max_depth, limit) + 1):
                options.append((side, start, end, depth))
    return options


def _gen_apply_notch(top, bottom, option):
    side, start, end, depth = option
    if side == "top":
        top[start : end + 1] = [depth] * (end - start + 1)
    else:
        height = max(bottom) + 1
        bottom[start : end + 1] = [height - 1 - depth] * (end - start + 1)


def _gen_interval_sets(width, count, side):
    if count == 0:
        return [[]]
    intervals = []
    rule = GEN_NOTCH_INTERVAL_RULES[side]
    start_range = range(rule["start_min"], width - rule["start_stop_margin"])
    end_limit = width - rule["end_stop_margin"]
    for start in start_range:
        for end in range(start + 1, end_limit):
            intervals.append((start, end))

    out = []

    def rec(chosen, pos):
        if len(chosen) == count:
            out.append(chosen[:])
            return
        for i in range(pos, len(intervals)):
            start, end = intervals[i]
            if chosen and start - (chosen[-1][1] + 1) < GEN_NOTCH_INTERVAL_GAP:
                continue
            chosen.append((start, end))
            rec(chosen, i + 1)
            chosen.pop()

    rec([], 0)
    return out


def _gen_add_notches(top, bottom, notch_count, depth_budget):
    if notch_count == 0:
        return True
    width = len(top)
    splits = []
    for top_count in range(notch_count + 1):
        bottom_count = notch_count - top_count
        top_sets = _gen_interval_sets(width, top_count, "top")
        bottom_sets = _gen_interval_sets(width, bottom_count, "bottom")
        if top_sets and bottom_sets:
            splits.append((top_sets, bottom_sets))
    if not splits:
        return False

    top_sets, bottom_sets = random.choice(splits)
    planned = [("top", start, end) for start, end in random.choice(top_sets)]
    planned += [("bottom", start, end) for start, end in random.choice(bottom_sets)]
    random.shuffle(planned)

    remaining_budget = depth_budget
    remaining_notches = len(planned)
    for side, start, end in planned:
        max_depth = remaining_budget - (remaining_notches - 1)
        options = [
            opt
            for opt in _gen_notch_options(top, bottom, side, max_depth)
            if opt[1] == start and opt[2] == end
        ]
        if not options:
            return False
        option = random.choice(options)
        _gen_apply_notch(top, bottom, option)
        remaining_budget -= option[3]
        remaining_notches -= 1
    return True


def _gen_patch_from_profiles(top, bottom, color):
    height = max(bottom) + 1
    width = len(top)
    patch = np.zeros((height, width), dtype=int)
    path = _gen_path_from_profiles(top, bottom)
    if len(set(path)) != len(path):
        return None
    coords = np.asarray(path, dtype=int)
    rows = coords[:, 0]
    cols = coords[:, 1]
    if np.any((rows < 0) | (rows >= height) | (cols < 0) | (cols >= width)):
        return None
    patch[rows, cols] = color

    mask = patch != GEN_BACKGROUND_COLOR
    metrics = _gen_cycle_metrics(mask)
    if metrics is None:
        return None
    perimeter, convex, concave = metrics
    metrics_valid = (
        GEN_PERIMETER_RANGE[0] <= perimeter <= GEN_PERIMETER_RANGE[1]
        and GEN_CONVEX_RANGE[0] <= convex <= GEN_CONVEX_RANGE[1]
        and GEN_CONCAVE_RANGE[0] <= concave <= GEN_CONCAVE_RANGE[1]
        and convex - concave == GEN_CONVEX_CONCAVE_DELTA
    )
    if not metrics_valid:
        return None
    if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
        return None
    return patch


def _gen_feasible_dimensions(notch_count, max_height, max_width):
    feasible_dims = []
    for height in range(GEN_PATCH_HEIGHT_RANGE[0], max_height + 1):
        for width in range(GEN_PATCH_WIDTH_RANGE[0], max_width + 1):
            base_perimeter = 2 * height + 2 * width - 4
            depth_budget = max(0, (GEN_PERIMETER_RANGE[1] - base_perimeter) // 2)
            side_capacity = 0
            if height >= GEN_MIN_NOTCH_CANVAS_SIDE:
                side_capacity = 2 * max(0, (width - GEN_NOTCH_CAPACITY_WIDTH_OFFSET) // GEN_NOTCH_CAPACITY_STRIDE)
            if notch_count >= GEN_MULTI_NOTCH_THRESHOLD and height < GEN_MIN_MULTI_NOTCH_HEIGHT:
                continue
            if notch_count <= min(GEN_MAX_NOTCHES, depth_budget, side_capacity):
                feasible_dims.append((height, width, depth_budget))
    return feasible_dims


def _gen_oriented_patch(patch):
    variants = []
    for k in range(GEN_ROTATION_COUNT):
        candidate = np.rot90(patch, k)
        h, w = candidate.shape
        if (
            GEN_PATCH_HEIGHT_RANGE[0] <= h <= GEN_PATCH_HEIGHT_RANGE[1]
            and GEN_PATCH_WIDTH_RANGE[0] <= w <= GEN_PATCH_WIDTH_RANGE[1]
        ):
            variants.append(candidate)
    if not variants:
        return None

    patch = random.choice(variants).copy()
    if random.choice(GEN_FLIP_CHOICES):
        patch = np.fliplr(patch).copy()
    if random.choice(GEN_FLIP_CHOICES):
        patch = np.flipud(patch).copy()
    return patch


def _gen_cycle_patch(
    color,
    desired_notches=None,
    max_height=GEN_PATCH_HEIGHT_RANGE[1],
    max_width=GEN_PATCH_WIDTH_RANGE[1],
):
    if desired_notches is None:
        notch_count = random.choices(GEN_NOTCH_COUNTS, weights=GEN_NOTCH_WEIGHTS)[0]
    else:
        notch_count = int(desired_notches)
    feasible_dims = _gen_feasible_dimensions(notch_count, max_height, max_width)
    if not feasible_dims:
        return None

    height, width, depth_budget = random.choice(feasible_dims)
    top = [0] * width
    bottom = [height - 1] * width
    if not _gen_add_notches(top, bottom, notch_count, depth_budget):
        return None

    patch = _gen_patch_from_profiles(top, bottom, color)
    if patch is None:
        return None

    return _gen_oriented_patch(patch)


def _gen_boxes_nested(box_a, box_b):
    ar0, ac0, ar1, ac1 = box_a
    br0, bc0, br1, bc1 = box_b
    a_in_b = ar0 >= br0 and ac0 >= bc0 and ar1 <= br1 and ac1 <= bc1
    b_in_a = br0 >= ar0 and bc0 >= ac0 and br1 <= ar1 and bc1 <= ac1
    return a_in_b or b_in_a


def _gen_can_place(coords, r0, c0, occupied):
    for pr, pc in coords:
        r = r0 + int(pr)
        c = c0 + int(pc)
        for dr, dc in GEN_OCCUPIED_OFFSETS:
            if (r + dr, c + dc) in occupied:
                return False
    return True


def _gen_place_cycles(canvas, patches):
    occupied = set()
    boxes = []
    for patch in sorted(patches, key=lambda p: int(np.count_nonzero(p)), reverse=True):
        h, w = patch.shape
        coords = np.argwhere(patch != GEN_BACKGROUND_COLOR)
        candidates = []
        for r0 in range(GEN_PLACEMENT_MARGIN, canvas.shape[0] - h):
            for c0 in range(GEN_PLACEMENT_MARGIN, canvas.shape[1] - w):
                box = (r0, c0, r0 + h - 1, c0 + w - 1)
                if any(_gen_boxes_nested(box, other) for other in boxes):
                    continue
                if _gen_can_place(coords, r0, c0, occupied):
                    candidates.append((r0, c0, box))
        if not candidates:
            return False
        r0, c0, box = random.choice(candidates)
        paste(canvas, patch, r0, c0, transparent=GEN_BACKGROUND_COLOR)
        boxes.append(box)
        occupied.update((r0 + int(pr), c0 + int(pc)) for pr, pc in coords)
    return True


def _gen_make_patches(color, cycle_count):
    patches = []
    for _ in range(cycle_count):
        if cycle_count == GEN_THREE_CYCLE_COUNT:
            desired = random.choices(GEN_THREE_CYCLE_NOTCH_COUNTS, weights=GEN_THREE_CYCLE_NOTCH_WEIGHTS)[0]
            height_range = GEN_THREE_CYCLE_PATCH_HEIGHT_RANGE
            width_range = GEN_THREE_CYCLE_PATCH_WIDTH_RANGE
        else:
            desired = random.choices(GEN_NOTCH_COUNTS, weights=GEN_NOTCH_WEIGHTS)[0]
            height_range = GEN_PATCH_HEIGHT_RANGE
            width_range = GEN_PATCH_WIDTH_RANGE
        patch = _gen_cycle_patch(color, desired, max_height=height_range[1], max_width=width_range[1])
        if patch is None:
            return None
        patches.append(patch)
    return patches


def _gen_attempt():
    height = random.randint(*GEN_CANVAS_ROW_RANGE)
    width = random.randint(*GEN_CANVAS_COL_RANGE)
    color = random.choice(GEN_FOREGROUND_COLORS)
    count = random.choice(GEN_CYCLE_COUNTS)
    patches = _gen_make_patches(color, count)
    if patches is None:
        return None

    canvas = np.zeros((height, width), dtype=int)
    if not _gen_place_cycles(canvas, patches):
        return None

    foreground = int(np.count_nonzero(canvas))
    if foreground >= canvas.size - foreground:
        return None
    try:
        main(canvas)
    except Exception:
        return None
    return canvas


def generate_input():
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed")
