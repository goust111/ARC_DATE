from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _trace_cycle(coords: np.ndarray) -> list[tuple[int, int]]:
    pts = {tuple(map(int, p)) for p in coords}
    for r, c in pts:
        degree = sum((nr, nc) in pts for nr, nc in neighbors4(r, c))
        assert degree == 2, (r, c, degree)

    start = min(pts)
    nxts = sorted((nr, nc) for nr, nc in neighbors4(*start) if (nr, nc) in pts)
    assert len(nxts) == 2

    path = [start]
    prev = start
    cur = nxts[0]
    while cur != start:
        path.append(cur)
        candidates = [
            (nr, nc)
            for nr, nc in neighbors4(*cur)
            if (nr, nc) in pts and (nr, nc) != prev
        ]
        assert len(candidates) == 1, (cur, candidates)
        prev, cur = cur, candidates[0]
        assert len(path) <= len(pts)

    assert len(path) == len(pts)
    return path


def _classify_cycle(path: list[tuple[int, int]]) -> dict[str, object]:
    counts: Counter[int] = Counter()
    turn_signs: dict[tuple[int, int], int] = {}
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

    assert turn_signs and counts[1] != counts[-1]
    convex_sign = 1 if counts[1] > counts[-1] else -1
    convex = {pos for pos, sign in turn_signs.items() if sign == convex_sign}
    concave = set(turn_signs) - convex
    straight = set(path) - set(turn_signs)
    assert len(convex) - len(concave) == 4
    return {
        "convex": convex,
        "concave": concave,
        "straight": straight,
        "convex_count": len(convex),
    }


def _interior_coords(grid: np.ndarray, coords: np.ndarray) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox_coords(coords)
    local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        local[int(r) - r0, int(c) - c0] = True

    bg_comps = components_mask((grid[r0 : r1 + 1, c0 : c1 + 1] == BLACK) & (~local))
    interior = []
    for comp in bg_comps:
        if any(
            rr == 0 or cc == 0 or rr == local.shape[0] - 1 or cc == local.shape[1] - 1
            for rr, cc in comp
        ):
            continue
        interior.append(comp)
    assert len(interior) == 1
    return [(r0 + int(rr), c0 + int(cc)) for rr, cc in interior[0]]


def _read_legend(grid: np.ndarray) -> tuple[list[int], tuple[int, int]]:
    col = grid[:, -1]
    rows = [r for r, value in enumerate(col.tolist()) if int(value) != BLACK]
    assert rows
    start, end = rows[0], rows[-1]
    assert rows == list(range(start, end + 1))
    colors = [int(grid[r, -1]) for r in range(start, end + 1)]
    assert len(colors) == len(set(colors))
    return colors, (start, end)


def _analyze_loops(grid: np.ndarray) -> list[dict[str, object]]:
    loops = []
    for coords in components_mask(grid == GREEN):
        path = _trace_cycle(coords)
        info = _classify_cycle(path)
        interior = _interior_coords(grid, coords)
        assert interior
        loops.append(
            {
                "coords": {tuple(map(int, p)) for p in coords},
                "convex": info["convex"],
                "concave": info["concave"],
                "straight": info["straight"],
                "convex_count": info["convex_count"],
                "interior": interior,
            }
        )
    counts = [int(loop["convex_count"]) for loop in loops]
    assert len(counts) == len(set(counts))
    return loops


def main(grid: np.ndarray) -> np.ndarray:
    """Classify each green loop's bends, rank loops by convex-corner count, and fill them from a right-edge legend.

    Input structure:
    - Black background with 3-4 disjoint green one-pixel-wide closed orthogonal loops.
    - Every loop pixel has exactly two 4-neighbors inside its own loop, and each loop encloses a black interior that stays away from the grid boundary.
    - The rightmost column contains a contiguous vertical legend of distinct colors, one entry per loop.

    Rule:
    - Trace every green loop and inspect the turn at each path pixel.
    - Bend pixels whose turn direction is the majority sign on that loop are convex and become yellow; minority-sign bends are concave and become red; straight pixels stay green.
    - Count the yellow convex pixels on each loop, sort loops from fewest convex corners to most, then assign legend colors from top to bottom in that order.
    - Fill each loop's enclosed black interior with its assigned legend color and erase the legend back to black.

    Key insight:
    - The legend does not map by spatial position; it maps by the structural ranking of loops after convex-versus-concave corner classification.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    legend_colors, (legend_start, legend_end) = _read_legend(g)
    loops = _analyze_loops(g)
    assert len(legend_colors) == len(loops)

    for loop in loops:
        for r, c in loop["convex"]:
            out[r, c] = YELLOW
        for r, c in loop["concave"]:
            out[r, c] = RED

    ranked = sorted(loops, key=lambda loop: int(loop["convex_count"]))
    for fill_color, loop in zip(legend_colors, ranked, strict=True):
        for r, c in loop["interior"]:
            out[r, c] = int(fill_color)

    out[legend_start : legend_end + 1, -1] = BLACK
    return out


# Generator controls.
GEN_GRID_ROWS = (20, 28)
GEN_GRID_COLS = (22, 30)
GEN_LOOP_COUNT_OPTIONS = (3, 4)
GEN_LEGEND_COLORS = [BLUE, RED, YELLOW, ORANGE]
GEN_ATTEMPTS = 400
GEN_OCCUPIED_OFFSETS = [(0, 0), *DIRS4]

GEN_PATCH_HEIGHT_RANGE = (3, 10)
GEN_PATCH_WIDTH_RANGE = (4, 12)
GEN_PERIMETER_RANGE = (8, 42)
GEN_CONVEX_RANGE = (4, 12)
GEN_CONCAVE_RANGE = (0, 8)

GEN_MIN_NOTCH_CANVAS_SIDE = 4
GEN_MIN_NOTCH_DEPTH = 1
GEN_MIN_MULTI_NOTCH_HEIGHT = 5
GEN_MULTI_NOTCH_THRESHOLD = 2
GEN_MAX_NOTCHES = 4
GEN_NOTCH_INTERVAL_GAP = 2
GEN_NOTCH_FLAT_PAD = 2
GEN_NOTCH_CLEARANCE_PAD = 1
GEN_PROFILE_CLEARANCE = 2
GEN_NOTCH_CAPACITY_WIDTH_OFFSET = 3
GEN_NOTCH_CAPACITY_STRIDE = 4
GEN_NOTCH_INTERVAL_RULES = {
    "top": {"start_min": 2, "start_stop_margin": 4, "end_stop_margin": 3},
    "bottom": {"start_min": 3, "start_stop_margin": 3, "end_stop_margin": 2},
}


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


def _gen_cycle_metrics(mask: np.ndarray) -> tuple[int, int, int] | None:
    try:
        path = _trace_cycle(np.argwhere(mask))
    except AssertionError:
        return None
    counts = Counter({1: 0, -1: 0})
    n = len(path)
    for i, (r, c) in enumerate(path):
        pr, pc = path[i - 1]
        nr, nc = path[(i + 1) % n]
        incoming = (r - pr, c - pc)
        outgoing = (nr - r, nc - c)
        cross = incoming[0] * outgoing[1] - incoming[1] * outgoing[0]
        if cross:
            counts[1 if cross > 0 else -1] += 1
    return len(path), max(counts.values()), min(counts.values())


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
            option
            for option in _gen_notch_options(top, bottom, side, max_depth)
            if option[1] == start and option[2] == end
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
    rows = np.array([r for r, _ in path], dtype=int)
    cols = np.array([c for _, c in path], dtype=int)
    if np.any((rows < 0) | (rows >= height) | (cols < 0) | (cols >= width)):
        return None
    patch[rows, cols] = color

    metrics = _gen_cycle_metrics(patch == color)
    if metrics is None:
        return None
    perimeter, convex, concave = metrics
    if not (
        GEN_PERIMETER_RANGE[0] <= perimeter <= GEN_PERIMETER_RANGE[1]
        and GEN_CONVEX_RANGE[0] <= convex <= GEN_CONVEX_RANGE[1]
        and GEN_CONCAVE_RANGE[0] <= concave <= GEN_CONCAVE_RANGE[1]
        and convex - concave == 4
    ):
        return None
    if not (np.any(patch[0]) and np.any(patch[-1]) and np.any(patch[:, 0]) and np.any(patch[:, -1])):
        return None
    return patch


def _gen_feasible_dimensions(notch_count):
    dims = []
    for height in range(GEN_PATCH_HEIGHT_RANGE[0], GEN_PATCH_HEIGHT_RANGE[1] + 1):
        for width in range(GEN_PATCH_WIDTH_RANGE[0], GEN_PATCH_WIDTH_RANGE[1] + 1):
            base_perimeter = 2 * height + 2 * width - 4
            depth_budget = max(0, (GEN_PERIMETER_RANGE[1] - base_perimeter) // 2)
            side_capacity = 0
            if height >= GEN_MIN_NOTCH_CANVAS_SIDE:
                side_capacity = 2 * max(0, (width - GEN_NOTCH_CAPACITY_WIDTH_OFFSET) // GEN_NOTCH_CAPACITY_STRIDE)
            if notch_count >= GEN_MULTI_NOTCH_THRESHOLD and height < GEN_MIN_MULTI_NOTCH_HEIGHT:
                continue
            if notch_count <= min(GEN_MAX_NOTCHES, depth_budget, side_capacity):
                dims.append((height, width, depth_budget))
    return dims


def _gen_oriented_patch(patch: np.ndarray) -> np.ndarray:
    patch = np.rot90(patch, random.randrange(4)).copy()
    if random.choice([False, True]):
        patch = np.fliplr(patch).copy()
    if random.choice([False, True]):
        patch = np.flipud(patch).copy()
    return patch


def _gen_cycle_patch(color: int, notch_count: int) -> np.ndarray | None:
    dims = _gen_feasible_dimensions(notch_count)
    if not dims:
        return None
    height, width, depth_budget = random.choice(dims)
    top = [0] * width
    bottom = [height - 1] * width
    if not _gen_add_notches(top, bottom, notch_count, depth_budget):
        return None
    patch = _gen_patch_from_profiles(top, bottom, color)
    if patch is None:
        return None
    return _gen_oriented_patch(patch)


def _boxes_nested(a, b) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    a_in_b = ar0 >= br0 and ac0 >= bc0 and ar1 <= br1 and ac1 <= bc1
    b_in_a = br0 >= ar0 and bc0 >= ac0 and br1 <= ar1 and bc1 <= ac1
    return a_in_b or b_in_a


def _can_place_patch(coords: np.ndarray, r0: int, c0: int, occupied: set[tuple[int, int]]) -> bool:
    for pr, pc in coords:
        r = r0 + int(pr)
        c = c0 + int(pc)
        for dr, dc in GEN_OCCUPIED_OFFSETS:
            if (r + dr, c + dc) in occupied:
                return False
    return True


def _place_patches(canvas: np.ndarray, patches: list[np.ndarray]) -> bool:
    occupied: set[tuple[int, int]] = set()
    boxes: list[tuple[int, int, int, int]] = []
    for patch in sorted(patches, key=lambda p: int(np.count_nonzero(p)), reverse=True):
        h, w = patch.shape
        coords = np.argwhere(patch == GREEN)
        candidates = []
        for r0 in range(1, canvas.shape[0] - h):
            for c0 in range(1, canvas.shape[1] - w - 1):
                box = (r0, c0, r0 + h - 1, c0 + w - 1)
                if any(_boxes_nested(box, other) for other in boxes):
                    continue
                if _can_place_patch(coords, r0, c0, occupied):
                    candidates.append((r0, c0, box))
        if not candidates:
            return False
        r0, c0, box = random.choice(candidates)
        rr, cc = np.where(patch == GREEN)
        canvas[r0 + rr, c0 + cc] = GREEN
        boxes.append(box)
        occupied.update((r0 + int(pr), c0 + int(pc)) for pr, pc in coords)
    return True


def _choose_notch_counts(loop_count: int) -> list[int]:
    pool = [0, 1, 2, 3]
    chosen = sorted(random.sample(pool, loop_count))
    return chosen


def _make_patches(loop_count: int) -> tuple[list[np.ndarray], list[int]] | None:
    patches = []
    convex_counts = []
    for notch_count in _choose_notch_counts(loop_count):
        patch = _gen_cycle_patch(GREEN, notch_count)
        if patch is None:
            return None
        metrics = _gen_cycle_metrics(patch == GREEN)
        if metrics is None:
            return None
        _, convex, _ = metrics
        try:
            interior = _interior_coords(patch, np.argwhere(patch == GREEN))
        except AssertionError:
            return None
        if len(interior) < 2:
            return None
        patches.append(patch)
        convex_counts.append(convex)
    if len(set(convex_counts)) != loop_count:
        return None
    return patches, convex_counts


def _place_legend(grid: np.ndarray, loop_count: int) -> list[int]:
    colors = random.sample(GEN_LEGEND_COLORS, loop_count)
    if grid.shape[0] - loop_count > 2:
        start = random.randint(1, grid.shape[0] - loop_count - 1)
    else:
        start = 0
    grid[start : start + loop_count, -1] = np.array(colors, dtype=int)
    return colors


def _validate_pair(inp: np.ndarray, out: np.ndarray) -> None:
    legend_colors, (start, end) = _read_legend(inp)
    loops = _analyze_loops(inp)
    assert 3 <= len(loops) <= 4
    assert len(legend_colors) == len(loops)
    ranked = sorted(loops, key=lambda loop: int(loop["convex_count"]))
    expected_boundary = np.full(inp.shape, BLACK, dtype=int)
    for loop in loops:
        for r, c in loop["straight"]:
            expected_boundary[r, c] = GREEN
        for r, c in loop["convex"]:
            expected_boundary[r, c] = YELLOW
        for r, c in loop["concave"]:
            expected_boundary[r, c] = RED
    for fill_color, loop in zip(legend_colors, ranked, strict=True):
        for r, c in loop["interior"]:
            assert out[r, c] == fill_color
    for r in range(inp.shape[0]):
        for c in range(inp.shape[1] - 1):
            if expected_boundary[r, c] != BLACK:
                assert out[r, c] == expected_boundary[r, c]
    assert np.all(out[start : end + 1, -1] == BLACK)


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(*GEN_GRID_ROWS)
    w = random.randint(*GEN_GRID_COLS)
    loop_count = random.choice(GEN_LOOP_COUNT_OPTIONS)

    built = _make_patches(loop_count)
    if built is None:
        return None
    patches, convex_counts = built
    if len(set(convex_counts)) != loop_count:
        return None

    grid = np.zeros((h, w), dtype=int)
    if not _place_patches(grid[:, :-1], patches):
        return None
    _place_legend(grid, loop_count)

    try:
        out = main(grid)
        _validate_pair(grid, out)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
