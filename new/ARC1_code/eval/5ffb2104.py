import numpy as np

from arc_utils import find_components, get_bg, is_contiguous, paste
import itertools
import random


def _right_slide_order(component: dict) -> tuple[int, int, int, int, int]:
    r0, c0, r1, c1 = component["bbox"]
    return (-c1, -c0, r0, r1, -int(component["area"]))


def main(grid: np.ndarray) -> np.ndarray:
    """Slide every colored object rightward until it is blocked by the edge or by an object that started farther right.

    Input structure
    A black-background grid contains several monochromatic foreground objects, each a 4-connected component.

    Rule
    Clear the foreground, then process objects from right to left by their original bounding boxes.
    Each object keeps its shape and color while repeatedly moving one cell to the right until its next step would leave the grid or overlap an already placed object.

    Key insight
    The objects behave like horizontal gravity: rightmost objects settle first and become blockers, so later objects cannot pass through them even if there is empty space beyond.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    components = find_components(g, bg=bg, connectivity=4, monochromatic=True)
    assert components

    h, w = g.shape
    out = np.full_like(g, bg)
    occupied = np.zeros((h, w), dtype=bool)

    for component in sorted(components, key=_right_slide_order):
        color = int(component["color"])
        coords = np.asarray(component["coords"], dtype=int)
        assert coords.ndim == 2 and coords.shape[1] == 2
        assert color != bg

        shift = 0
        rows = coords[:, 0]
        cols = coords[:, 1]
        while True:
            next_cols = cols + shift + 1
            if np.any(next_cols >= w):
                break
            if np.any(occupied[rows, next_cols]):
                break
            shift += 1

        dest_cols = cols + shift
        assert not np.any(occupied[rows, dest_cols])
        out[rows, dest_cols] = color
        occupied[rows, dest_cols] = True

    assert int(np.sum(out != bg)) == int(np.sum(g != bg))
    return out

# --------------- generator (o5x) ---------------

# Canvas and object count bounds.
_GEN_H_RANGE = (6, 10)
_GEN_W_RANGE = (6, 13)
_GEN_OBJECT_COUNT_RANGE = (5, 10)

# Foreground shape and density constraints.
_GEN_DENSITY_RANGE = (0.10, 0.35)
_GEN_TALL_GROUP_PROB = 0.55
_GEN_AREA_BOUNDS_BY_HEIGHT = {1: (1, 3), 2: (2, 5)}
_GEN_WIDTH_OPTIONS = {(1, 1): (1,), (1, 2): (2,), (1, 3): (3,), (2, 2): (1,), (2, 3): (2,), (2, 4): (2, 3), (2, 5): (3,)}

# Color, blocking, and retry constraints.
_GEN_COLORS = tuple(range(1, 10))
_GEN_DISTINCT_COLOR_RANGE = (4, 6)
_GEN_MIN_BLOCKED_OBJECTS = 2
_GEN_MIN_WALL_OBJECTS = 2
_GEN_RETRY_LIMIT = 300
_GEN_BG = 0
_GEN_CONNECTIVITY = 4


def _gen_partition_sum(total, lows, highs):
    vals = []
    remaining = int(total)
    for i, (lo, hi) in enumerate(zip(lows, highs)):
        rest_l = sum(lows[i + 1 :])
        rest_h = sum(highs[i + 1 :])
        a = max(int(lo), remaining - rest_h)
        b = min(int(hi), remaining - rest_l)
        if a > b:
            return None
        v = random.randint(a, b)
        vals.append(v)
        remaining -= v
    return vals if remaining == 0 else None


def _gen_width_options(height, area):
    return list(_GEN_WIDTH_OPTIONS.get((int(height), int(area)), ()))


def _gen_mask(height, width, area, required=()):
    required = {tuple(p) for p in required}
    cells = [(r, c) for r in range(height) for c in range(width)]
    if len(required) > area:
        return None
    choices = []
    for combo in itertools.combinations(cells, area):
        coords = set(combo)
        if not required <= coords:
            continue
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        if min(rows) != 0 or max(rows) != height - 1:
            continue
        if min(cols) != 0 or max(cols) != width - 1:
            continue
        mask = np.zeros((height, width), dtype=int)
        for r, c in coords:
            mask[r, c] = 1
        if is_contiguous(mask, background=_GEN_BG, connectivity=_GEN_CONNECTIVITY):
            choices.append(mask)
    if not choices:
        return None
    return random.choice(choices)


def _gen_build_group(height, length, areas, grid_width):
    if length == 1:
        widths = _gen_width_options(height, areas[0])
        random.shuffle(widths)
        for width in widths:
            mask = _gen_mask(height, width, areas[0])
            if mask is not None:
                return [(mask, grid_width - width)]
        return None

    left_area, right_area = areas
    left_widths = _gen_width_options(height, left_area)
    right_widths = _gen_width_options(height, right_area)
    pairs = [(lw, rw) for lw in left_widths for rw in right_widths if lw + rw <= grid_width - 1]
    random.shuffle(pairs)
    contact_rows = list(range(height))
    random.shuffle(contact_rows)
    for left_width, right_width in pairs:
        for contact_row in contact_rows:
            left = _gen_mask(height, left_width, left_area, [(contact_row, left_width - 1)])
            right = _gen_mask(height, right_width, right_area, [(contact_row, 0)])
            if left is None or right is None:
                continue
            right_c0 = grid_width - right_width
            left_c0 = right_c0 - left_width
            return [(left, left_c0), (right, right_c0)]
    return None


def _gen_abs_coords(mask, r0, c0):
    locs = np.argwhere(mask != _GEN_BG)
    return {(int(r0 + r), int(c0 + c)) for r, c in locs}


def _gen_are_adjacent(coords_a, coords_b):
    for r, c in coords_a:
        if (r - 1, c) in coords_b or (r + 1, c) in coords_b or (r, c - 1) in coords_b or (r, c + 1) in coords_b:
            return True
    return False


def _gen_group_lengths(k, h):
    min_pairs = max(_GEN_MIN_BLOCKED_OBJECTS, k - h)
    max_pairs = min(k // 2, h)
    if min_pairs > max_pairs:
        return None
    n_pairs = random.randint(min_pairs, max_pairs)
    lengths = [2] * n_pairs + [1] * (k - 2 * n_pairs)
    random.shuffle(lengths)
    return lengths


def _gen_group_heights(group_count, h):
    heights = [1] * group_count
    extra_rows = h - group_count
    for idx in random.sample(range(group_count), group_count):
        if extra_rows <= 0:
            break
        if random.random() < _GEN_TALL_GROUP_PROB:
            heights[idx] = 2
            extra_rows -= 1
    return heights


def _gen_group_areas(h, w, lengths, heights):
    """Split a sampled foreground area into per-object areas grouped by row."""
    lows = []
    highs = []
    object_groups = []
    for gi, (length, height) in enumerate(zip(lengths, heights)):
        lo, hi = _GEN_AREA_BOUNDS_BY_HEIGHT[height]
        for _ in range(length):
            lows.append(lo)
            highs.append(hi)
            object_groups.append(gi)

    min_density, max_density = _GEN_DENSITY_RANGE
    min_total = int(np.ceil(min_density * h * w))
    max_total = int(np.floor(max_density * h * w))
    low = max(min_total, sum(lows))
    high = min(max_total, sum(highs))
    if low > high:
        return None

    areas = _gen_partition_sum(random.randint(low, high), lows, highs)
    if areas is None:
        return None

    per_group_areas = [[] for _ in lengths]
    for area, gi in zip(areas, object_groups):
        per_group_areas[gi].append(area)
    return per_group_areas


def _gen_build_objects(h, w, lengths, heights, per_group_areas):
    """Build output-position objects with masks, anchors, and absolute coords."""
    group_count = len(heights)
    gaps = [0] * (group_count + 1)
    for _ in range(h - sum(heights)):
        gaps[random.randrange(group_count + 1)] += 1
    row = gaps[0]
    objects = []
    for gi, (length, height) in enumerate(zip(lengths, heights)):
        built = _gen_build_group(height, length, per_group_areas[gi], w)
        if built is None:
            return None
        for mask, c0 in built:
            coords = _gen_abs_coords(mask, row, c0)
            objects.append({"mask": mask, "r0": row, "c0": c0, "coords": coords, "group": gi})
        row += height + gaps[gi + 1]
    return objects


def _gen_apply_input_shifts(objects, group_count):
    """Choose a shared left shift per group and attach input coords in place."""
    group_shifts = {}
    for gi in range(group_count):
        min_col = min(c for obj in objects if obj["group"] == gi for _, c in obj["coords"])
        if min_col < 1:
            return False
        group_shifts[gi] = random.randint(1, min_col)
    for obj in objects:
        shift = group_shifts[obj["group"]]
        obj["shift"] = shift
        obj["in_coords"] = {(r, c - shift) for r, c in obj["coords"]}
    return True


def _gen_assign_colors(objects, distinct):
    n = len(objects)
    palette = random.sample(_GEN_COLORS, distinct)
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if _gen_are_adjacent(objects[i]["coords"], objects[j]["coords"]) or _gen_are_adjacent(
                objects[i]["in_coords"], objects[j]["in_coords"]
            ):
                adj[i].add(j)
                adj[j].add(i)

    order = sorted(range(n), key=lambda idx: len(adj[idx]), reverse=True)
    colors = [None] * n

    def rec(pos, used):
        if pos == n:
            return len(used) == distinct
        idx = order[pos]
        forbidden = {colors[j] for j in adj[idx] if colors[j] is not None}
        candidates = list(palette)
        random.shuffle(candidates)
        for color in candidates:
            if color in forbidden:
                continue
            new_used = used | {color}
            if distinct - len(new_used) > n - pos - 1:
                continue
            colors[idx] = color
            if rec(pos + 1, new_used):
                return True
            colors[idx] = None
        return False

    if not rec(0, set()):
        return None
    return colors


def _gen_render_grids(h, w, objects, colors):
    inp = np.zeros((h, w), dtype=int)
    out = np.zeros((h, w), dtype=int)
    for obj, color in zip(objects, colors):
        patch = obj["mask"] * int(color)
        paste(out, patch, obj["r0"], obj["c0"], transparent=_GEN_BG)
        paste(inp, patch, obj["r0"], obj["c0"] - obj["shift"], transparent=_GEN_BG)
    return inp, out


def _gen_count_stops(objects, w):
    blocked = 0
    wall = 0
    for obj in objects:
        coords = obj["coords"]
        if any(c == w - 1 for _, c in coords):
            wall += 1
        blockers = set().union(*(o["coords"] for o in objects if o is not obj))
        if any((r, c + 1) in blockers for r, c in coords):
            blocked += 1
    return blocked, wall


def _gen_is_valid_candidate(inp, out, objects, k, w):
    comps = find_components(inp, bg=_GEN_BG, connectivity=_GEN_CONNECTIVITY, monochromatic=True)
    if len(comps) != k:
        return False
    blocked, wall = _gen_count_stops(objects, w)
    return blocked >= _GEN_MIN_BLOCKED_OBJECTS and wall >= _GEN_MIN_WALL_OBJECTS and np.array_equal(main(inp), out)


def _gen_attempt():
    h = random.randint(*_GEN_H_RANGE)
    w = random.randint(*_GEN_W_RANGE)
    k = random.randint(*_GEN_OBJECT_COUNT_RANGE)
    lengths = _gen_group_lengths(k, h)
    if lengths is None:
        return None
    group_count = len(lengths)
    heights = _gen_group_heights(group_count, h)
    per_group_areas = _gen_group_areas(h, w, lengths, heights)
    if per_group_areas is None:
        return None
    objects = _gen_build_objects(h, w, lengths, heights, per_group_areas)
    if objects is None:
        return None
    if not _gen_apply_input_shifts(objects, group_count):
        return None
    distinct = random.randint(_GEN_DISTINCT_COLOR_RANGE[0], min(_GEN_DISTINCT_COLOR_RANGE[1], k))
    colors = _gen_assign_colors(objects, distinct)
    if colors is None:
        return None
    inp, out = _gen_render_grids(h, w, objects, colors)
    if not _gen_is_valid_candidate(inp, out, objects, k, w):
        return None
    return inp


def generate_input():
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
