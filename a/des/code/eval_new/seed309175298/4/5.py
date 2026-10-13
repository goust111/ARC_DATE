from __future__ import annotations

import random

import numpy as np


BLACK = 0
GRAY = 5
MAROON = 9
BASE_COLORS = (1, 2, 3, 4, 6, 7, 8)

GRID_H_RANGE = (16, 30)
MAX_LEVELS = 5
LEFT_PANEL_MIN = 10
LEFT_PANEL_MAX = 18
RIGHT_PANEL_EXTRA_RANGE = (2, 8)
RECT_DIM_RANGE = (3, 7)
PLACEMENT_RETRIES = 200
GEN_RETRIES = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
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
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _find_separator(grid: np.ndarray, bg: int) -> tuple[int, int]:
    h, w = grid.shape
    candidates: list[tuple[int, int]] = []
    for c in range(1, w - 1):
        color = int(grid[0, c])
        if color == bg:
            continue
        if not np.all(grid[:, c] == color):
            continue
        if not np.any(grid[:, :c] != bg):
            continue
        if not np.all(grid[:, c + 1 :] == bg):
            continue
        candidates.append((c, color))
    assert len(candidates) == 1, "expected one full-height separator before a blank right panel"
    return candidates[0]


def _extract_nodes(left_panel: np.ndarray, bg: int) -> list[dict[str, int | tuple[int, int, int, int]]]:
    nodes: list[dict[str, int | tuple[int, int, int, int]]] = []
    for coords in _components(left_panel != bg):
        r0, c0 = map(int, coords.min(axis=0))
        r1, c1 = map(int, coords.max(axis=0))
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert len(coords) == h * w, "each object must be a dense rectangle"

        sub = left_panel[r0 : r1 + 1, c0 : c1 + 1]
        marker_color = int(sub[0, 0])
        colors = {int(v) for v in sub.reshape(-1)}
        assert colors != {marker_color}, "rectangle must retain a base color after the corner marker swap"
        base_colors = colors - {marker_color}
        assert len(base_colors) == 1, "each rectangle must have one base color plus one corner marker"
        assert int(np.count_nonzero(sub == marker_color)) == 1, "corner marker must be a single pixel"
        base_color = int(next(iter(base_colors)))

        nodes.append(
            {
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
                "h": h,
                "w": w,
                "area": h * w,
                "base_color": base_color,
                "marker_color": marker_color,
            }
        )
    assert nodes, "left panel must contain rectangles"
    return nodes


def _depth_groups(nodes: list[dict[str, int | tuple[int, int, int, int]]]) -> tuple[dict[int, list[int]], int]:
    base_to_idx = {int(node["base_color"]): idx for idx, node in enumerate(nodes)}
    root_indices = [idx for idx, node in enumerate(nodes) if int(node["marker_color"]) not in base_to_idx]
    assert len(root_indices) == 1, "expected exactly one root marker color outside the block palette"
    root = root_indices[0]

    children: dict[int, list[int]] = {idx: [] for idx in range(len(nodes))}
    for idx, node in enumerate(nodes):
        if idx == root:
            continue
        parent = base_to_idx.get(int(node["marker_color"]))
        assert parent is not None, "every non-root marker color must match a parent block color"
        children[parent].append(idx)

    depths = {root: 0}
    queue = [root]
    while queue:
        cur = queue.pop(0)
        for child in children[cur]:
            assert child not in depths, "marker pointers must form a tree"
            depths[child] = depths[cur] + 1
            queue.append(child)
    assert len(depths) == len(nodes), "all rectangles must belong to the root tree"

    grouped: dict[int, list[int]] = {}
    for idx, depth in depths.items():
        grouped.setdefault(depth, []).append(idx)
    return grouped, root


def _paint_split_frame(
    out: np.ndarray,
    top: int,
    left: int,
    height: int,
    width: int,
    colors_top_to_bottom: list[int],
) -> None:
    if height < 1 or width < 1:
        return

    rows = np.arange(top, top + height, dtype=int)
    row_groups = np.array_split(rows, len(colors_top_to_bottom))
    bottom = top + height - 1
    right = left + width - 1

    for row_group, color in zip(row_groups, colors_top_to_bottom):
        if row_group.size == 0:
            continue
        out[row_group, left] = int(color)
        out[row_group, right] = int(color)
        if int(top) in row_group:
            out[top, left : right + 1] = int(color)
        if int(bottom) in row_group:
            out[bottom, left : right + 1] = int(color)


def _render_with_nodes(
    grid: np.ndarray,
    separator_col: int,
    nodes: list[dict[str, int | tuple[int, int, int, int]]],
) -> np.ndarray:
    h, w = grid.shape
    grouped, root = _depth_groups(nodes)
    root_h = int(nodes[root]["h"])
    root_w = int(nodes[root]["w"])
    right_width = w - separator_col - 1
    assert root_h <= h and root_w <= right_width, "root rectangle must fit inside the right panel"

    frame_top = (h - root_h) // 2
    frame_left = separator_col + 1 + (right_width - root_w) // 2
    out = grid.copy()

    depth = 0
    while depth in grouped:
        frame_h = root_h - 2 * depth
        frame_w = root_w - 2 * depth
        if frame_h < 1 or frame_w < 1:
            break

        layer = grouped[depth]
        ordered = sorted(
            layer,
            key=lambda idx: (
                -int(nodes[idx]["area"]),
                int(nodes[idx]["r0"]),
                int(nodes[idx]["c0"]),
            ),
        )
        colors = [int(nodes[idx]["base_color"]) for idx in ordered]
        _paint_split_frame(out, frame_top + depth, frame_left + depth, frame_h, frame_w, colors)
        depth += 1

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the corner-color parent pointers on the left and draw their depth layers as concentric frames on the right.

    Input structure:
    - The dominant background color fills the canvas, and one full-height non-background separator column splits it into a left object panel and a blank right panel.
    - The left panel contains separated dense rectangles; each rectangle has one base color, except that its top-left pixel is replaced by a corner-marker color.
    - Exactly one rectangle has a corner-marker color that does not match any rectangle base color; the right panel contains only background.

    Rule:
    - Treat each corner-marker color as a pointer to the parent rectangle whose base color matches it; the unique unmatched marker defines the root at depth 0.
    - The root rectangle's height and width define the outer frame size, and each deeper layer shrinks both dimensions by 2 while staying centered in the right panel.
    - Draw one-pixel rectangular rings from outer to inner using the base colors of rectangles at each depth; when several rectangles share a depth, sort them by area descending and split that ring horizontally from top to bottom in that order.

    Key insight:
    - The altered top-left pixels are not decoration but color pointers, so the task is to decode a tree from color references and then map tree depth to nested geometry.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    separator_col, _separator_color = _find_separator(g, bg)
    left_panel = g[:, :separator_col]
    nodes = _extract_nodes(left_panel, bg)
    return _render_with_nodes(g, separator_col, nodes)


def _sample_depth_counts() -> tuple[list[int], int]:
    level_count = random.randint(2, MAX_LEVELS)
    counts = [1] * level_count
    extra_budget = min(len(BASE_COLORS) - level_count, level_count - 1)
    branching = extra_budget > 0 and random.random() < 0.5
    if branching:
        branch_count = random.randint(1, extra_budget)
        for depth in random.sample(range(1, level_count), branch_count):
            counts[depth] = 2

    if level_count >= 3 and random.random() < 0.5:
        drawable_levels = random.randint(2, level_count - 1)
    else:
        drawable_levels = level_count

    return counts, drawable_levels


def _root_min_size(counts: list[int], drawable_levels: int) -> int:
    min_size = 2 * (drawable_levels - 1) + 1
    for depth, count in enumerate(counts[:drawable_levels]):
        if count > 1:
            min_size = max(min_size, 2 * depth + 2)
    return min_size


def _sample_nodes() -> tuple[list[dict[str, int]], int]:
    counts, drawable_levels = _sample_depth_counts()
    min_root = _root_min_size(counts, drawable_levels)
    max_root = min(12, min_root + 4)
    if min_root > max_root:
        raise RuntimeError("no valid root size range")

    node_count = sum(counts)
    colors = random.sample(list(BASE_COLORS), node_count)
    color_iter = iter(colors)

    nodes: list[dict[str, int]] = []
    depth_nodes: list[list[int]] = []
    for depth, count in enumerate(counts):
        current: list[int] = []
        sibling_areas: set[int] = set()
        for _ in range(count):
            if depth == 0:
                rect_h = random.randint(min_root, max_root)
                rect_w = random.randint(min_root, max_root)
            else:
                while True:
                    rect_h = random.randint(*RECT_DIM_RANGE)
                    rect_w = random.randint(*RECT_DIM_RANGE)
                    area = rect_h * rect_w
                    if area not in sibling_areas:
                        sibling_areas.add(area)
                        break

            base_color = next(color_iter)
            node = {
                "depth": depth,
                "h": rect_h,
                "w": rect_w,
                "area": rect_h * rect_w,
                "base_color": base_color,
                "marker_color": MAROON,
                "parent_idx": -1,
                "r0": -1,
                "c0": -1,
                "r1": -1,
                "c1": -1,
            }
            nodes.append(node)
            current.append(len(nodes) - 1)
        depth_nodes.append(current)

    for depth in range(1, len(depth_nodes)):
        for idx in depth_nodes[depth]:
            parent_idx = random.choice(depth_nodes[depth - 1])
            nodes[idx]["parent_idx"] = parent_idx
            nodes[idx]["marker_color"] = nodes[parent_idx]["base_color"]

    return nodes, drawable_levels


def _place_nodes(nodes: list[dict[str, int]], height: int, left_width: int) -> bool:
    occupied = np.zeros((height, left_width), dtype=bool)
    order = sorted(range(len(nodes)), key=lambda idx: -int(nodes[idx]["area"]))
    for idx in order:
        node = nodes[idx]
        rect_h = int(node["h"])
        rect_w = int(node["w"])
        if rect_h > height - 2 or rect_w > left_width - 2:
            return False

        placed = False
        for _ in range(PLACEMENT_RETRIES):
            r0 = random.randint(1, height - rect_h - 1)
            c0 = random.randint(1, left_width - rect_w - 1)
            rs = slice(max(0, r0 - 1), min(height, r0 + rect_h + 1))
            cs = slice(max(0, c0 - 1), min(left_width, c0 + rect_w + 1))
            if occupied[rs, cs].any():
                continue
            occupied[r0 : r0 + rect_h, c0 : c0 + rect_w] = True
            node["r0"] = r0
            node["c0"] = c0
            node["r1"] = r0 + rect_h - 1
            node["c1"] = c0 + rect_w - 1
            placed = True
            break
        if not placed:
            return False
    return True


def _build_input(
    nodes: list[dict[str, int]],
    height: int,
    left_width: int,
    right_width: int,
) -> np.ndarray:
    grid = np.full((height, left_width + 1 + right_width), BLACK, dtype=int)
    grid[:, left_width] = GRAY
    for node in nodes:
        r0 = int(node["r0"])
        c0 = int(node["c0"])
        r1 = int(node["r1"])
        c1 = int(node["c1"])
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(node["base_color"])
        grid[r0, c0] = int(node["marker_color"])
    return grid


def _expected_output(grid: np.ndarray, left_width: int, nodes: list[dict[str, int]]) -> np.ndarray:
    solver_nodes = [{k: v for k, v in node.items()} for node in nodes]
    return _render_with_nodes(grid, left_width, solver_nodes)


def _gen_attempt() -> np.ndarray | None:
    try:
        nodes, drawable_levels = _sample_nodes()
    except RuntimeError:
        return None

    root = nodes[0]
    root_h = int(root["h"])
    root_w = int(root["w"])
    node_count = len(nodes)

    height_min = max(GRID_H_RANGE[0], root_h + 4, 10 + node_count)
    if height_min > GRID_H_RANGE[1]:
        return None
    height = random.randint(height_min, GRID_H_RANGE[1])

    left_min = max(LEFT_PANEL_MIN + (1 if node_count >= 5 else 0), root_w + 2)
    left_max = min(LEFT_PANEL_MAX, 27 - root_w)
    if left_min > left_max:
        return None
    left_width = random.randint(left_min, left_max)

    right_min = root_w + RIGHT_PANEL_EXTRA_RANGE[0]
    right_max = min(root_w + RIGHT_PANEL_EXTRA_RANGE[1], 30 - left_width - 1)
    if right_min > right_max:
        return None
    right_width = random.randint(right_min, right_max)

    if drawable_levels > 1 and min(root_h, root_w) <= 3:
        return None
    if not _place_nodes(nodes, height, left_width):
        return None

    grid = _build_input(nodes, height, left_width, right_width)
    expected = _expected_output(grid, left_width, nodes)
    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid concentric-tree ARC input")
