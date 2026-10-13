from __future__ import annotations

import random
from collections import Counter, defaultdict, deque

import numpy as np


BG = 0
FRAME = 5
PALETTE = (1, 2, 3, 4)
MAX_SIDE = 30
GEN_ATTEMPTS = 600


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _is_rectangular_frame(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    pts = {tuple(map(int, pt)) for pt in coords}
    expected = set()
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    for r in range(r0, r1 + 1):
        expected.add((r, c0))
        expected.add((r, c1))
    return pts == expected


def _extract_frames(grid: np.ndarray) -> list[dict]:
    frames = []
    for coords in _components_of_color(grid, FRAME):
        bbox = _bbox(coords)
        if _is_rectangular_frame(coords, bbox):
            frames.append(
                {
                    "coords": coords,
                    "bbox": bbox,
                    "bbox_area": (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1),
                }
            )
    assert frames, "expected at least one rectangular frame"
    return frames


def _strictly_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return outer[0] < inner[0] and outer[1] < inner[1] and outer[2] > inner[2] and outer[3] > inner[3]


def _frame_tree(frames: list[dict]) -> tuple[list[int | None], dict[int, list[int]], dict[int, int]]:
    parents: list[int | None] = [None] * len(frames)
    children: dict[int, list[int]] = defaultdict(list)
    for i, frame in enumerate(frames):
        box = frame["bbox"]
        parent_candidates = [
            j
            for j, other in enumerate(frames)
            if i != j and _strictly_contains(other["bbox"], box)
        ]
        if parent_candidates:
            parents[i] = min(parent_candidates, key=lambda j: frames[j]["bbox_area"])
            children[parents[i]].append(i)

    depths: dict[int, int] = {}
    roots = [i for i, parent in enumerate(parents) if parent is None]
    stack = [(root, 1) for root in roots]
    while stack:
        node, depth = stack.pop()
        depths[node] = depth
        for child in children[node]:
            stack.append((child, depth + 1))
    return parents, children, depths


def _find_cycle(grid: np.ndarray) -> tuple[int, list[int]]:
    candidates = []
    for row in (0, grid.shape[0] - 1):
        vals = [int(v) for v in grid[row] if int(v) not in (BG, FRAME)]
        if 3 <= len(vals) <= 4 and len(set(vals)) == len(vals):
            candidates.append((row, vals))
    assert len(candidates) == 1, "expected exactly one legend row on the top or bottom border"
    return candidates[0]


def _paint_between_frames(grid: np.ndarray, cycle: list[int], frames: list[dict], children: dict[int, list[int]], depths: dict[int, int]) -> np.ndarray:
    out = grid.copy()
    for idx, frame in enumerate(frames):
        r0, c0, r1, c1 = frame["bbox"]
        mask = np.zeros(grid.shape, dtype=bool)
        mask[r0 + 1 : r1, c0 + 1 : c1] = True
        direct_children = children.get(idx, [])
        assert len(direct_children) <= 1, "each nested-frame group must form a simple chain"
        if direct_children:
            cr0, cc0, cr1, cc1 = frames[direct_children[0]]["bbox"]
            mask[cr0 : cr1 + 1, cc0 : cc1 + 1] = False
        out[mask & (grid == BG)] = cycle[(depths[idx] - 1) % len(cycle)]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the gaps between nested gray rectangular frames using legend colors indexed by nesting depth.

    Input structure:
    - The background is black, and every frame is a 1-pixel-thick gray rectangle.
    - There are 2-3 independent groups of strictly nested rectangular frames; each group forms a single containment chain.
    - The innermost area of each group contains one non-black seed pixel, and the top or bottom border row carries a left-to-right legend color cycle outside all frames.

    Rule:
    - Detect every gray rectangular frame and build the containment tree to recover each frame's depth inside its own nested group.
    - Read the legend row from left to right as a cyclic color sequence.
    - For each frame at depth d, fill the black cells between that frame and its direct child frame; if it has no child frame, fill the black cells inside it except for the seed pixel.
    - Use legend color cycle[(d - 1) mod n] for that filled band, and keep the legend, seeds, and gray frames unchanged.

    Key insight:
    - The fill color is not determined by a frame's size or position; it comes from the frame's topological depth in the nesting tree, which must then be mapped through a separate legend cycle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= MAX_SIDE
    legend_row, cycle = _find_cycle(g)
    assert legend_row in (0, g.shape[0] - 1)
    frames = _extract_frames(g)
    _, children, depths = _frame_tree(frames)
    return _paint_between_frames(g, cycle, frames, children, depths)


def _group_boxes(top: int, left: int, layers: int, gap: int, inner_h: int, inner_w: int) -> list[tuple[int, int, int, int]]:
    outer_h = inner_h + 2 * (layers - 1) * (gap + 1)
    outer_w = inner_w + 2 * (layers - 1) * (gap + 1)
    bottom = top + outer_h - 1
    right = left + outer_w - 1
    return [
        (top + inset, left + inset, bottom - inset, right - inset)
        for inset in ((gap + 1) * depth for depth in range(layers))
    ]


def _draw_frame(grid: np.ndarray, box: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = FRAME
    grid[r1, c0 : c1 + 1] = FRAME
    grid[r0 : r1 + 1, c0] = FRAME
    grid[r0 : r1 + 1, c1] = FRAME


def _layout_positions(dims: list[tuple[int, int]], legend_top: bool) -> tuple[int, int, list[tuple[int, int]]] | None:
    order = list(range(len(dims)))
    random.shuffle(order)
    shuffled = [dims[i] for i in order]
    layout_types = ["row", "col"]
    if len(dims) == 3:
        layout_types.append("lshape")
    random.shuffle(layout_types)

    reserved_top = 2 if legend_top else 1
    reserved_bottom = 1 if legend_top else 2
    for layout in layout_types:
        if layout == "row":
            gap_x = [random.randint(1, 3) for _ in range(len(dims) - 1)]
            content_h = max(h for h, _ in shuffled)
            content_w = sum(w for _, w in shuffled) + sum(gap_x)
            ys = [random.randint(0, content_h - h) for h, _ in shuffled]
            xs = []
            cursor = 0
            for idx, (_, w) in enumerate(shuffled):
                xs.append(cursor)
                if idx < len(shuffled) - 1:
                    cursor += w + gap_x[idx]
            rel_positions = list(zip(ys, xs))
        elif layout == "col":
            gap_y = [random.randint(1, 3) for _ in range(len(dims) - 1)]
            content_h = sum(h for h, _ in shuffled) + sum(gap_y)
            content_w = max(w for _, w in shuffled)
            xs = [random.randint(0, content_w - w) for _, w in shuffled]
            ys = []
            cursor = 0
            for idx, (h, _) in enumerate(shuffled):
                ys.append(cursor)
                if idx < len(shuffled) - 1:
                    cursor += h + gap_y[idx]
            rel_positions = list(zip(ys, xs))
        else:
            (h0, w0), (h1, w1), (h2, w2) = shuffled
            gap_x = random.randint(1, 3)
            gap_y = random.randint(1, 3)
            top_h = max(h0, h1)
            top_w = w0 + gap_x + w1
            x2 = random.randint(0, max(0, top_w - w2))
            content_h = top_h + gap_y + h2
            content_w = max(top_w, x2 + w2)
            rel_positions = [
                (random.randint(0, top_h - h0), 0),
                (random.randint(0, top_h - h1), w0 + gap_x),
                (top_h + gap_y, x2),
            ]

        extra_top = random.randint(0, 1)
        extra_bottom = random.randint(0, 1)
        extra_left = random.randint(0, 1)
        extra_right = random.randint(0, 1)
        height = reserved_top + reserved_bottom + extra_top + extra_bottom + content_h
        width = extra_left + extra_right + content_w
        if height > MAX_SIDE or width > MAX_SIDE:
            continue

        top_shift = reserved_top + extra_top
        left_shift = extra_left
        positions_shuffled = [(top_shift + r, left_shift + c) for r, c in rel_positions]
        positions = [None] * len(dims)
        for shuffled_idx, original_idx in enumerate(order):
            positions[original_idx] = positions_shuffled[shuffled_idx]
        assert all(pos is not None for pos in positions)
        return height, width, positions
    return None


def _sample_groups(num_groups: int) -> list[dict]:
    if num_groups == 2:
        layer_counts = [3, 2]
    else:
        layer_counts = [3, 2, 2]
    random.shuffle(layer_counts)

    groups = []
    compact_two_layer_assigned = False
    for idx, layers in enumerate(layer_counts):
        if layers == 3:
            gap = 2
            inner_h = random.choice([4, 4, 5])
            inner_w = random.choice([4, 4, 5])
        else:
            gap = 2
            if not compact_two_layer_assigned:
                inner_h = random.choice([3, 4])
                inner_w = random.choice([3, 4])
                compact_two_layer_assigned = True
            else:
                inner_h = random.randint(3, 5)
                inner_w = random.randint(3, 5)

        outer_h = inner_h + 2 * (layers - 1) * (gap + 1)
        outer_w = inner_w + 2 * (layers - 1) * (gap + 1)
        groups.append(
            {
                "layers": layers,
                "gap": gap,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "outer_h": outer_h,
                "outer_w": outer_w,
                "seed_color": random.choice(PALETTE),
                "index": idx,
            }
        )
    return groups


def _place_legend(grid: np.ndarray, cycle: list[int], row: int) -> None:
    gaps = [random.randint(1, 2) for _ in range(len(cycle) - 1)]
    span = len(cycle) + sum(gaps)
    start = random.randint(0, grid.shape[1] - span)
    col = start
    for idx, color in enumerate(cycle):
        grid[row, col] = color
        if idx < len(cycle) - 1:
            col += 1 + gaps[idx]


def _expected_from_blueprint(grid: np.ndarray, cycle: list[int], groups: list[dict]) -> np.ndarray:
    out = grid.copy()
    for group in groups:
        for depth, box in enumerate(group["boxes"], start=1):
            r0, c0, r1, c1 = box
            color = cycle[(depth - 1) % len(cycle)]
            mask = np.zeros(grid.shape, dtype=bool)
            mask[r0 + 1 : r1, c0 + 1 : c1] = True
            if depth < len(group["boxes"]):
                cr0, cc0, cr1, cc1 = group["boxes"][depth]
                mask[cr0 : cr1 + 1, cc0 : cc1 + 1] = False
            out[mask & (grid == BG)] = color
    return out


def _depth_counterexample_exists(groups: list[dict]) -> bool:
    frames = []
    for group_id, group in enumerate(groups):
        for depth, box in enumerate(group["boxes"], start=1):
            r0, c0, r1, c1 = box
            area = (r1 - r0 + 1) * (c1 - c0 + 1)
            frames.append((group_id, depth, area))
    for group_a, depth_a, area_a in frames:
        if depth_a != 1:
            continue
        for group_b, depth_b, area_b in frames:
            if group_a != group_b and depth_b >= 2 and area_a < area_b:
                return True
    return False


def _validate_generated(grid: np.ndarray, cycle: list[int], groups: list[dict], legend_row: int) -> bool:
    if grid.ndim != 2 or max(grid.shape) > MAX_SIDE:
        return False
    if legend_row not in (0, grid.shape[0] - 1):
        return False
    if len(groups) not in (2, 3):
        return False
    if not _depth_counterexample_exists(groups):
        return False

    expected = _expected_from_blueprint(grid, cycle, groups)
    if np.array_equal(expected, grid):
        return False

    frames = _extract_frames(grid)
    _, children, depths = _frame_tree(frames)
    if len([idx for idx, depth in depths.items() if depth == 1]) != len(groups):
        return False
    if sorted(len(children.get(i, [])) for i in range(len(frames)))[-1] > 1:
        return False

    recovered_cycle_row, recovered_cycle = _find_cycle(grid)
    if recovered_cycle_row != legend_row or recovered_cycle != cycle:
        return False

    expected_boxes = sorted(box for group in groups for box in group["boxes"])
    recovered_boxes = sorted(frame["bbox"] for frame in frames)
    if expected_boxes != recovered_boxes:
        return False

    if not np.array_equal(main(grid), expected):
        return False

    filled_colors = set(map(int, expected[(expected != grid) & (expected != FRAME)]))
    if len(filled_colors) < 2:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    num_groups = random.choice([2, 2, 3])
    legend_top = random.choice([True, False])
    cycle_len = random.choice([3, 4])
    cycle = random.sample(PALETTE, cycle_len)
    groups = _sample_groups(num_groups)
    dims = [(group["outer_h"], group["outer_w"]) for group in groups]
    layout = _layout_positions(dims, legend_top)
    if layout is None:
        return None
    height, width, positions = layout

    grid = np.zeros((height, width), dtype=int)
    legend_row = 0 if legend_top else height - 1
    _place_legend(grid, cycle, legend_row)

    for group, (top, left) in zip(groups, positions):
        boxes = _group_boxes(top, left, group["layers"], group["gap"], group["inner_h"], group["inner_w"])
        group["boxes"] = boxes
        for box in boxes:
            _draw_frame(grid, box)
        r0, c0, r1, c1 = boxes[-1]
        seed_r = random.randint(r0 + 1, r1 - 1)
        seed_c = random.randint(c0 + 1, c1 - 1)
        grid[seed_r, seed_c] = group["seed_color"]
        group["seed"] = (seed_r, seed_c)

    if not _validate_generated(grid, cycle, groups, legend_row):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-frame depth-cycle input")
