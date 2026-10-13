import random
from collections import defaultdict, deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

DEPTH_TO_COLOR = {
    1: RED,
    2: BLUE,
    3: GREEN,
    4: YELLOW,
}

MAX_GENERATION_ATTEMPTS = 400
LEAF_SIZES = (
    (4, 4),
    (4, 5),
    (5, 4),
    (4, 6),
    (6, 4),
    (5, 5),
    (5, 6),
    (6, 5),
)


class FrameNode:
    def __init__(self, h, w, children=None, layout=None, target_depth=1):
        self.h = int(h)
        self.w = int(w)
        self.children = list(children or [])
        self.layout = layout
        self.target_depth = int(target_depth)
        self.box = None


def _bbox(coords):
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(coords)
    return out


def _min_grey_crossings(grid):
    h, w = grid.shape
    inf = 10 ** 9
    dist = np.full((h, w), inf, dtype=int)
    dq = deque()
    border = set()
    for r in range(h):
        border.add((r, 0))
        border.add((r, w - 1))
    for c in range(w):
        border.add((0, c))
        border.add((h - 1, c))
    for r, c in border:
        cost = 1 if int(grid[r, c]) == GREY else 0
        if cost < dist[r, c]:
            dist[r, c] = cost
            if cost == 0:
                dq.appendleft((r, c))
            else:
                dq.append((r, c))

    while dq:
        r, c = dq.popleft()
        base = int(dist[r, c])
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            step = 1 if int(grid[nr, nc]) == GREY else 0
            nd = base + step
            if nd < dist[nr, nc]:
                dist[nr, nc] = nd
                if step == 0:
                    dq.appendleft((nr, nc))
                else:
                    dq.append((nr, nc))
    return dist


def _analyze_regions(grid):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    colors = set(map(int, np.unique(g)))
    assert colors <= {BLACK, GREY}, "input must contain only black background and grey walls"

    dist = _min_grey_crossings(g)
    regions = []
    for coords in _components(g == BLACK):
        depth_values = {int(dist[r, c]) for r, c in coords}
        depth = min(depth_values)
        regions.append(
            {
                "coords": coords,
                "area": len(coords),
                "bbox": _bbox(coords),
                "depth": depth,
            }
        )
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill every other enclosed black region, using nested-wall depth to choose the fill color.

    Input structure:
    - The input uses black as background and grey as one-cell walls that form nested rectangular or near-rectangular boundaries.
    - Black cells split into 4-connected regions, including outer background pockets and interior pockets enclosed by one or more grey walls.
    - Multiple disjoint enclosed regions can occur at the same nesting depth.

    Rule:
    - Identify every 4-connected black region and compute its depth as the minimum number of grey wall layers that must be crossed to reach the grid edge.
    - Group regions by depth, then sort each depth group by descending area; break ties by the region bbox top row, then left column.
    - Within each depth group, fill regions at even indices only (0, 2, 4, ...).
    - Use depth-dependent colors for selected enclosed regions: depth 1 -> red, depth 2 -> blue, depth 3 -> green, depth 4 -> yellow; depth 0 stays black and grey walls stay unchanged.

    Key insight:
    - The target regions are chosen by a double index: nesting depth determines the color, while area ranking inside the same depth determines which regions are filled.
    """
    g = np.asarray(grid, dtype=int)
    regions = _analyze_regions(g)
    by_depth = defaultdict(list)
    for region in regions:
        by_depth[int(region["depth"])].append(region)

    out = g.copy()
    for depth, items in by_depth.items():
        items.sort(key=lambda item: (-int(item["area"]), int(item["bbox"][0]), int(item["bbox"][1])))
        fill_color = DEPTH_TO_COLOR.get(int(depth))
        if fill_color is None:
            continue
        for idx, item in enumerate(items):
            if idx % 2 != 0:
                continue
            for r, c in item["coords"]:
                out[r, c] = fill_color
    return out


def _split_nonnegative(total, parts, rng):
    if parts == 1:
        return [total]
    cuts = sorted(rng.sample(range(total + parts - 1), parts - 1))
    out = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        out.append(cut - prev - 1)
        prev = cut
    return out


def _make_leaf(rng, depth):
    h, w = rng.choice(LEAF_SIZES)
    return FrameNode(h, w, target_depth=depth)


def _wrap_single(child, rng):
    extra_h = rng.randint(0, 2)
    extra_w = rng.randint(0, 2)
    return FrameNode(child.h + 4 + extra_h, child.w + 4 + extra_w, [child], "single", child.target_depth - 1)


def _build_chain(depth_start, depth_end, rng):
    node = _make_leaf(rng, depth_end)
    for _ in range(depth_end - depth_start):
        node = _wrap_single(node, rng)
    return node


def _pack_children(children, rng, depth):
    if len(children) == 1:
        child = children[0]
        extra_h = rng.randint(0, 2)
        extra_w = rng.randint(0, 2)
        return FrameNode(child.h + 4 + extra_h, child.w + 4 + extra_w, [child], "single", depth)

    horizontal_h = max(child.h for child in children) + 4 + rng.randint(0, 2)
    horizontal_w = sum(child.w for child in children) + (len(children) - 1) + 4 + rng.randint(0, 2)
    vertical_h = sum(child.h for child in children) + (len(children) - 1) + 4 + rng.randint(0, 2)
    vertical_w = max(child.w for child in children) + 4 + rng.randint(0, 2)

    layouts = [
        ("horizontal", horizontal_h, horizontal_w),
        ("vertical", vertical_h, vertical_w),
    ]
    rng.shuffle(layouts)
    layout, h, w = min(layouts, key=lambda item: (item[1] * item[2], max(item[1], item[2])))
    return FrameNode(h, w, children, layout, depth)


def _assign_boxes(node, r0, c0, rng):
    node.box = (int(r0), int(c0), int(r0 + node.h - 1), int(c0 + node.w - 1))
    if not node.children:
        return

    if node.layout == "single":
        child = node.children[0]
        extra_h = node.h - 4 - child.h
        extra_w = node.w - 4 - child.w
        child_r = r0 + 2 + rng.randint(0, extra_h)
        child_c = c0 + 2 + rng.randint(0, extra_w)
        _assign_boxes(child, child_r, child_c, rng)
        return

    if node.layout == "horizontal":
        extra_w = node.w - 4 - sum(child.w for child in node.children) - (len(node.children) - 1)
        gaps = _split_nonnegative(extra_w, len(node.children) + 1, rng)
        x = c0 + 2 + gaps[0]
        base_r = r0 + 2
        free_h = node.h - 4 - max(child.h for child in node.children)
        for idx, child in enumerate(node.children):
            child_r = base_r + rng.randint(0, free_h)
            _assign_boxes(child, child_r, x, rng)
            x += child.w
            if idx < len(node.children) - 1:
                x += 1 + gaps[idx + 1]
        return

    extra_h = node.h - 4 - sum(child.h for child in node.children) - (len(node.children) - 1)
    gaps = _split_nonnegative(extra_h, len(node.children) + 1, rng)
    y = r0 + 2 + gaps[0]
    base_c = c0 + 2
    free_w = node.w - 4 - max(child.w for child in node.children)
    for idx, child in enumerate(node.children):
        child_c = base_c + rng.randint(0, free_w)
        _assign_boxes(child, y, child_c, rng)
        y += child.h
        if idx < len(node.children) - 1:
            y += 1 + gaps[idx + 1]


def _draw_frame(grid, box):
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY


def _collect_boxes(node):
    boxes = [node.box]
    for child in node.children:
        boxes.extend(_collect_boxes(child))
    return boxes


def _build_depth3_roots(rng):
    chain_a1 = _build_chain(2, 3, rng)
    child_a2 = _make_leaf(rng, 2)
    root_a = _pack_children([chain_a1, child_a2], rng, 1)

    chain_b1 = _build_chain(2, 3, rng)
    root_b = _pack_children([chain_b1], rng, 1)
    return root_a, root_b


def _build_depth4_roots(rng):
    chain_a1 = _build_chain(2, 4, rng)
    root_a = _pack_children([chain_a1], rng, 1)

    chain_b1 = _build_chain(2, 3, rng)
    root_b = _pack_children([chain_b1], rng, 1)
    return root_a, root_b


def _place_roots(root_a, root_b, rng):
    candidates = []
    for order in ((root_a, root_b), (root_b, root_a)):
        a, b = order
        for layout in ("horizontal", "vertical"):
            gap = rng.randint(1, 3)
            if layout == "horizontal":
                left = rng.randint(0, 1)
                right = rng.randint(0, 2)
                top = rng.randint(0, 1)
                bottom = rng.randint(0, 2)
                h = max(a.h, b.h) + top + bottom
                w = a.w + b.w + gap + left + right
                if max(h, w) <= 30:
                    candidates.append((h, w, layout, order, (top, left, gap)))
            else:
                top = rng.randint(0, 1)
                bottom = rng.randint(0, 2)
                left = rng.randint(0, 1)
                right = rng.randint(0, 2)
                h = a.h + b.h + gap + top + bottom
                w = max(a.w, b.w) + left + right
                if max(h, w) <= 30:
                    candidates.append((h, w, layout, order, (top, left, gap)))
    if not candidates:
        return None
    h, w, layout, order, params = min(candidates, key=lambda item: (max(item[0], item[1]), item[0] * item[1]))
    grid = np.zeros((h, w), dtype=int)
    top, left, gap = params
    first, second = order
    if layout == "horizontal":
        first_r = top + rng.randint(0, h - max(first.h, second.h) - top)
        second_r = top + rng.randint(0, h - max(first.h, second.h) - top)
        _assign_boxes(first, first_r, left, rng)
        _assign_boxes(second, second_r, left + first.w + gap, rng)
    else:
        first_c = left + rng.randint(0, w - max(first.w, second.w) - left)
        second_c = left + rng.randint(0, w - max(first.w, second.w) - left)
        _assign_boxes(first, top, first_c, rng)
        _assign_boxes(second, top + first.h + gap, second_c, rng)
    for box in _collect_boxes(first) + _collect_boxes(second):
        _draw_frame(grid, box)
    return grid


def _depth_histogram(regions):
    hist = defaultdict(int)
    for region in regions:
        hist[int(region["depth"])] += 1
    return hist


def _intervals_overlap(a, b):
    return max(min(a), min(b)) <= min(max(a), max(b))


def _valid_generated_grid(grid, target_depth):
    regions = _analyze_regions(grid)
    hist = _depth_histogram(regions)
    min_required = {depth: 2 for depth in range(1, target_depth + 1)}
    if target_depth == 4:
        min_required[4] = 1
    if any(hist.get(depth, 0) < need for depth, need in min_required.items()):
        return False
    if max((region["depth"] for region in regions), default=0) != target_depth:
        return False
    if target_depth == 3 and max(hist.get(depth, 0) for depth in range(1, target_depth + 1)) < 3:
        return False

    areas_by_depth = defaultdict(list)
    for region in regions:
        if region["depth"] > 0:
            areas_by_depth[int(region["depth"])].append(int(region["area"]))
    overlap_found = False
    for depth_a in range(1, target_depth + 1):
        for depth_b in range(depth_a + 1, target_depth + 1):
            if _intervals_overlap(areas_by_depth[depth_a], areas_by_depth[depth_b]):
                overlap_found = True
                break
        if overlap_found:
            break
    if not overlap_found:
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    used_colors = set(map(int, np.unique(out))) - {BLACK, GREY}
    expected = {DEPTH_TO_COLOR[d] for d in range(1, target_depth + 1)}
    return expected <= used_colors


def _generate_attempt(rng, target_depth):
    if target_depth == 4:
        root_a, root_b = _build_depth4_roots(rng)
    else:
        root_a, root_b = _build_depth3_roots(rng)
    grid = _place_roots(root_a, root_b, rng)
    if grid is None:
        return None
    if not _valid_generated_grid(grid, target_depth):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    preferred_depth = 4 if rng.random() < 0.35 else 3
    primary_budget = MAX_GENERATION_ATTEMPTS // 2 if preferred_depth == 4 else MAX_GENERATION_ATTEMPTS
    for _ in range(primary_budget):
        grid = _generate_attempt(rng, preferred_depth)
        if grid is not None:
            return grid
    fallback_depth = 3 if preferred_depth == 4 else 4
    for _ in range(MAX_GENERATION_ATTEMPTS - primary_budget):
        grid = _generate_attempt(rng, fallback_depth)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid nested-wall example")
