import numpy as np


BG = 0
FRAME = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]


def _component_rectangles(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    rects = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rows = [p[0] for p in cells]
            cols = [p[1] for p in cells]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            rects.append((r0, c0, r1, c1))
    return rects


def _draw_frame(grid: np.ndarray, rect, color: int):
    r0, c0, r1, c1 = rect
    grid[r0, c0:c1 + 1] = color
    grid[r1, c0:c1 + 1] = color
    grid[r0:r1 + 1, c0] = color
    grid[r0:r1 + 1, c1] = color


def _contains(outer, inner) -> bool:
    return outer[0] < inner[0] and outer[1] < inner[1] and outer[2] > inner[2] and outer[3] > inner[3]


def _depth_data(rects):
    n = len(rects)
    parents = [-1] * n
    areas = [(r1 - r0 + 1) * (c1 - c0 + 1) for r0, c0, r1, c1 in rects]
    for i, rect in enumerate(rects):
        best = -1
        best_area = None
        for j, maybe_parent in enumerate(rects):
            if i == j or not _contains(maybe_parent, rect):
                continue
            if best == -1 or areas[j] < best_area:
                best = j
                best_area = areas[j]
        parents[i] = best
    depths = [0] * n
    for i in range(n):
        d = 0
        cur = parents[i]
        while cur != -1:
            d += 1
            cur = parents[cur]
        depths[i] = d
    children = [[] for _ in range(n)]
    for child, parent in enumerate(parents):
        if parent != -1:
            children[parent].append(child)
    return parents, depths, children


def _connector_side(parent, child):
    gaps = [
        ("top", child[0] - parent[0]),
        ("bottom", parent[2] - child[2]),
        ("left", child[1] - parent[1]),
        ("right", parent[3] - child[3]),
    ]
    return min(gaps, key=lambda item: (item[1], ["top", "bottom", "left", "right"].index(item[0])))[0]


def _draw_connector(grid: np.ndarray, parent, child, color: int):
    side = _connector_side(parent, child)
    if side == "top":
        col = (child[1] + child[3]) // 2
        grid[parent[0]:child[0] + 1, col] = color
    elif side == "bottom":
        col = (child[1] + child[3]) // 2
        grid[child[2]:parent[2] + 1, col] = color
    elif side == "left":
        row = (child[0] + child[2]) // 2
        grid[row, parent[1]:child[1] + 1] = color
    else:
        row = (child[0] + child[2]) // 2
        grid[row, child[3]:parent[3] + 1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor nested grey frames by containment depth and connect each child frame to its parent.

    Input structure:
    - Black background contains several one-pixel-thick grey rectangular frames with empty interiors.
    - Frames are either disjoint or fully nested, so they define a containment forest with 1-3 outer roots.
    - The bottom legend row contains isolated colored pixels separated by one black cell, ordered from left to right.

    Rule:
    - Read the legend colors left to right as the colors for depth 0, 1, 2, ...
    - Build the frame containment hierarchy: outer frames have depth 0, their direct children depth 1, and so on.
    - Recolor every grey frame border with the color assigned to its depth.
    - For each direct parent-child pair, find the child side whose matching parent side is closest, then draw a one-pixel straight connector from that child-side midpoint to the parent border using the child color.

    Key insight:
    - The grey rectangles are not independent shapes: their nesting depth determines border color, and each direct parent-child relation adds a connector colored by the child rather than the parent.
    """
    out = grid.copy()
    nonempty_rows = np.where(np.any(grid != BG, axis=1))[0]
    if len(nonempty_rows) == 0:
        return out
    legend_row = nonempty_rows[-1]
    legend_colors = [int(grid[legend_row, c]) for c in range(grid.shape[1]) if grid[legend_row, c] != BG]
    body = grid[:legend_row, :]
    frame_colors = sorted(int(v) for v in np.unique(body) if v != BG)
    if not frame_colors:
        return out
    frame_color = frame_colors[0]
    rects = _component_rectangles(body == frame_color)
    parents, depths, _ = _depth_data(rects)

    out[:legend_row, :][out[:legend_row, :] == frame_color] = BG
    rect_colors = {}
    for i, rect in enumerate(rects):
        color = legend_colors[depths[i]]
        rect_colors[i] = color
        _draw_frame(out, rect, color)
    for i, parent in enumerate(parents):
        if parent != -1:
            _draw_connector(out, rects[parent], rects[i], rect_colors[i])
    return out


def _odd_values(low: int, high: int):
    return [v for v in range(low, high + 1) if v % 2 == 1]


def _even_values(low: int, high: int):
    return [v for v in range(low, high + 1) if v % 2 == 0]


def _min_size(depth_remaining: int) -> int:
    return 3 + 8 * depth_remaining


def _middle_options(parent_len: int, child_min: int):
    options = []
    for g1 in _even_values(4, parent_len - child_min - 4):
        for g2 in _even_values(4, parent_len - child_min - g1):
            child_len = parent_len - g1 - g2
            if child_len >= child_min and child_len % 2 == 1:
                options.append((g1, g2, child_len))
    return options


def _single_side_lengths(parent_len: int, child_min: int):
    options = []
    for far_gap in _even_values(4, parent_len - child_min - 2):
        child_len = parent_len - 2 - far_gap
        if child_len >= child_min and child_len % 2 == 1:
            options.append((far_gap, child_len))
    return options


def _pair_lengths(parent_len: int, first_min: int, second_min: int):
    options = []
    for first_len in _odd_values(first_min, parent_len):
        for second_len in _odd_values(second_min, parent_len):
            if first_len + second_len <= parent_len - 5:
                options.append((first_len, second_len))
    return options


def _make_single_child(rect, side: str, child_min: int, rng):
    r0, c0, r1, c1 = rect
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if side in ("top", "bottom"):
        v_options = _single_side_lengths(h, child_min)
        h_options = _middle_options(w, child_min)
        far_gap, child_h = v_options[rng.integers(len(v_options))]
        left_gap, right_gap, child_w = h_options[rng.integers(len(h_options))]
        child_c0 = c0 + left_gap
        child_c1 = child_c0 + child_w - 1
        if side == "top":
            child_r0 = r0 + 2
            child_r1 = child_r0 + child_h - 1
        else:
            child_r1 = r1 - 2
            child_r0 = child_r1 - child_h + 1
        return (child_r0, child_c0, child_r1, child_c1)
    v_options = _middle_options(h, child_min)
    h_options = _single_side_lengths(w, child_min)
    top_gap, bottom_gap, child_h = v_options[rng.integers(len(v_options))]
    far_gap, child_w = h_options[rng.integers(len(h_options))]
    child_r0 = r0 + top_gap
    child_r1 = child_r0 + child_h - 1
    if side == "left":
        child_c0 = c0 + 2
        child_c1 = child_c0 + child_w - 1
    else:
        child_c1 = c1 - 2
        child_c0 = child_c1 - child_w + 1
    return (child_r0, child_c0, child_r1, child_c1)


def _make_tb_pair(rect, top_min: int, bottom_min: int, rng):
    r0, c0, r1, c1 = rect
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    length_options = _pair_lengths(h, top_min, bottom_min)
    top_h, bottom_h = length_options[rng.integers(len(length_options))]
    top_widths = _middle_options(w, top_min)
    bottom_widths = _middle_options(w, bottom_min)
    top_left_gap, _, top_w = top_widths[rng.integers(len(top_widths))]
    bottom_left_gap, _, bottom_w = bottom_widths[rng.integers(len(bottom_widths))]
    top_rect = (r0 + 2, c0 + top_left_gap, r0 + 1 + top_h, c0 + top_left_gap + top_w - 1)
    bottom_rect = (r1 - 1 - bottom_h, c0 + bottom_left_gap, r1 - 2, c0 + bottom_left_gap + bottom_w - 1)
    return top_rect, bottom_rect


def _make_lr_pair(rect, left_min: int, right_min: int, rng):
    r0, c0, r1, c1 = rect
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    length_options = _pair_lengths(w, left_min, right_min)
    left_w, right_w = length_options[rng.integers(len(length_options))]
    left_heights = _middle_options(h, left_min)
    right_heights = _middle_options(h, right_min)
    left_top_gap, _, left_h = left_heights[rng.integers(len(left_heights))]
    right_top_gap, _, right_h = right_heights[rng.integers(len(right_heights))]
    left_rect = (r0 + left_top_gap, c0 + 2, r0 + left_top_gap + left_h - 1, c0 + 1 + left_w)
    right_rect = (r0 + right_top_gap, c1 - 1 - right_w, r0 + right_top_gap + right_h - 1, c1 - 2)
    return left_rect, right_rect


def _single_feasible(rect, side: str, child_min: int) -> bool:
    r0, c0, r1, c1 = rect
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if side in ("top", "bottom"):
        return bool(_single_side_lengths(h, child_min) and _middle_options(w, child_min))
    return bool(_middle_options(h, child_min) and _single_side_lengths(w, child_min))


def _pair_feasible(rect, axis: str, first_min: int, second_min: int) -> bool:
    r0, c0, r1, c1 = rect
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if axis == "tb":
        return bool(_pair_lengths(h, first_min, second_min) and _middle_options(w, first_min) and _middle_options(w, second_min))
    return bool(_pair_lengths(w, first_min, second_min) and _middle_options(h, first_min) and _middle_options(h, second_min))


def _build_node(rect, depth_remaining: int, rng):
    node = {"rect": rect, "children": []}
    if depth_remaining == 0:
        return node

    if depth_remaining == 1:
        modes = []
        single_sides = [side for side in ("top", "bottom", "left", "right") if _single_feasible(rect, side, 3)]
        if single_sides:
            modes.append("single")
        pair_axes = [axis for axis in ("tb", "lr") if _pair_feasible(rect, axis, 3, 3)]
        if pair_axes:
            modes.append("pair")
        mode = modes[rng.integers(len(modes))]
        if mode == "single":
            side = single_sides[rng.integers(len(single_sides))]
            child = _make_single_child(rect, side, 3, rng)
            node["children"].append(_build_node(child, 0, rng))
            return node
        axis = pair_axes[rng.integers(len(pair_axes))]
        if axis == "tb":
            top_rect, bottom_rect = _make_tb_pair(rect, 3, 3, rng)
            node["children"] = [_build_node(top_rect, 0, rng), _build_node(bottom_rect, 0, rng)]
        else:
            left_rect, right_rect = _make_lr_pair(rect, 3, 3, rng)
            node["children"] = [_build_node(left_rect, 0, rng), _build_node(right_rect, 0, rng)]
        return node

    child_min = _min_size(depth_remaining - 1)
    deep_sides = [side for side in ("top", "bottom", "left", "right") if _single_feasible(rect, side, child_min)]
    deep_side = deep_sides[rng.integers(len(deep_sides))]

    extra_modes = ["deep_only"]
    if deep_side in ("top", "bottom") and _pair_feasible(rect, "tb", child_min if deep_side == "top" else 3, 3 if deep_side == "top" else child_min):
        extra_modes.append("deep_plus_leaf")
    if deep_side in ("left", "right") and _pair_feasible(rect, "lr", child_min if deep_side == "left" else 3, 3 if deep_side == "left" else child_min):
        extra_modes.append("deep_plus_leaf")

    mode = extra_modes[rng.integers(len(extra_modes))]
    if mode == "deep_only":
        child = _make_single_child(rect, deep_side, child_min, rng)
        node["children"].append(_build_node(child, depth_remaining - 1, rng))
        return node

    if deep_side == "top":
        top_rect, bottom_rect = _make_tb_pair(rect, child_min, 3, rng)
        node["children"] = [_build_node(top_rect, depth_remaining - 1, rng), _build_node(bottom_rect, 0, rng)]
    elif deep_side == "bottom":
        top_rect, bottom_rect = _make_tb_pair(rect, 3, child_min, rng)
        node["children"] = [_build_node(top_rect, 0, rng), _build_node(bottom_rect, depth_remaining - 1, rng)]
    elif deep_side == "left":
        left_rect, right_rect = _make_lr_pair(rect, child_min, 3, rng)
        node["children"] = [_build_node(left_rect, depth_remaining - 1, rng), _build_node(right_rect, 0, rng)]
    else:
        left_rect, right_rect = _make_lr_pair(rect, 3, child_min, rng)
        node["children"] = [_build_node(left_rect, 0, rng), _build_node(right_rect, depth_remaining - 1, rng)]
    return node


def _flatten(node):
    rects = [node["rect"]]
    for child in node["children"]:
        rects.extend(_flatten(child))
    return rects


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    while True:
        h = int(rng.integers(18, 31))
        w = int(rng.integers(18, 31))
        body_h = h - 2
        gap = 2
        feasible_root_counts = []
        for root_count in (1, 2, 3):
            avail = w - gap * (root_count + 1)
            if avail >= 11 + 5 * (root_count - 1):
                feasible_root_counts.append(root_count)
        if not feasible_root_counts:
            continue
        root_count = feasible_root_counts[rng.integers(len(feasible_root_counts))]
        avail = w - gap * (root_count + 1)

        widths = [5] * root_count
        primary = int(rng.integers(root_count))
        widths[primary] = 11
        remaining = avail - sum(widths)
        for _ in range(remaining):
            widths[int(rng.integers(root_count))] += 1

        lane_starts = []
        cur = gap
        for lane_w in widths:
            lane_starts.append(cur)
            cur += lane_w + gap

        max_depths = []
        for lane_w in widths:
            lane_max = 0
            while _min_size(lane_max + 1) <= lane_w and _min_size(lane_max + 1) <= body_h:
                lane_max += 1
            max_depths.append(lane_max)
        if max(max_depths) == 0:
            continue

        depths = [0] * root_count
        primary_options = [i for i, d in enumerate(max_depths) if d >= 1]
        chosen_primary = primary_options[rng.integers(len(primary_options))]
        depths[chosen_primary] = int(rng.integers(1, max_depths[chosen_primary] + 1))
        for i in range(root_count):
            if i == chosen_primary:
                continue
            depths[i] = int(rng.integers(0, max_depths[i] + 1))

        roots = []
        for i in range(root_count):
            lane_w = widths[i]
            min_dim = max(5, _min_size(depths[i]))
            root_w_choices = _odd_values(min_dim, lane_w)
            root_h_choices = _odd_values(min_dim, body_h)
            if not root_w_choices or not root_h_choices:
                break
            root_w = int(root_w_choices[rng.integers(len(root_w_choices))])
            root_h = int(root_h_choices[rng.integers(len(root_h_choices))])
            lane_left = lane_starts[i]
            col = int(rng.integers(lane_left, lane_left + lane_w - root_w + 1))
            row = int(rng.integers(0, body_h - root_h + 1))
            root_rect = (row, col, row + root_h - 1, col + root_w - 1)
            roots.append(_build_node(root_rect, depths[i], rng))
        if len(roots) != root_count:
            continue

        rects = []
        for node in roots:
            rects.extend(_flatten(node))

        max_depth = 0
        stack = [(node, 0) for node in roots]
        while stack:
            node, depth = stack.pop()
            max_depth = max(max_depth, depth)
            for child in node["children"]:
                stack.append((child, depth + 1))

        colors = PALETTE.copy()
        rng.shuffle(colors)
        legend = colors[:max_depth + 1]
        legend_len = 2 * len(legend) - 1
        if legend_len > w:
            continue
        legend_start = int(rng.integers(0, w - legend_len + 1))

        grid = np.zeros((h, w), dtype=int)
        for rect in rects:
            _draw_frame(grid, rect, FRAME)
        for idx, color in enumerate(legend):
            grid[h - 1, legend_start + 2 * idx] = color

        if np.any(grid[h - 2, :] != BG):
            continue
        return grid
