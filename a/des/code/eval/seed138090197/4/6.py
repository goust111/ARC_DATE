from __future__ import annotations

import random
from typing import Any

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _border_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _find_legend(grid: np.ndarray, bg: int) -> tuple[int, list[int]]:
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)
    candidates: list[tuple[int, int, list[int]]] = []
    for row in (0, height - 1):
        cols = [int(c) for c in range(width) if int(g[row, c]) != int(bg)]
        if len(cols) < 2:
            continue
        colors = [int(g[row, c]) for c in cols]
        if len(set(colors)) != len(colors):
            continue
        clear = True
        probe_row = 1 if row == 0 else height - 2
        for c in cols:
            if 0 <= probe_row < height and int(g[probe_row, c]) != int(bg):
                clear = False
                break
        if clear:
            candidates.append((len(cols), int(row), colors))
    assert candidates, "expected a top or bottom legend row"
    candidates.sort(key=lambda item: (item[0], item[1]))
    _count, row, colors = candidates[-1]
    return int(row), list(colors)


def _find_frames(grid: np.ndarray, bg: int, legend_row: int) -> list[dict[str, Any]]:
    g = np.asarray(grid, dtype=int).copy()
    g[int(legend_row), :] = int(bg)
    frames: list[dict[str, Any]] = []
    for color in np.unique(g):
        color = int(color)
        if color == int(bg):
            continue
        pts = np.argwhere(g == int(color))
        if int(pts.shape[0]) == 0:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)
        if height < 3 or width < 3:
            continue
        border = _border_mask(height, width)
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        if int(pts.shape[0]) != int(border.sum()):
            continue
        if not bool(np.all(sub[border] == int(color))):
            continue
        if not bool(np.all(sub[~border] != int(color))):
            continue
        frames.append(
            {
                "color": int(color),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "height": int(height),
                "width": int(width),
            }
        )
    assert frames, "expected at least one rectangular frame"
    return frames


def _build_tree(frames: list[dict[str, Any]]) -> tuple[dict[int, int | None], dict[int, list[int]]]:
    parents: dict[int, int | None] = {idx: None for idx in range(len(frames))}
    children: dict[int, list[int]] = {idx: [] for idx in range(len(frames))}
    for idx, frame in enumerate(frames):
        r0, c0, r1, c1 = frame["bbox"]
        parent_choice: tuple[int, int] | None = None
        for jdx, other in enumerate(frames):
            if idx == jdx:
                continue
            pr0, pc0, pr1, pc1 = other["bbox"]
            if pr0 < r0 and pc0 < c0 and pr1 > r1 and pc1 > c1:
                area = int((pr1 - pr0 + 1) * (pc1 - pc0 + 1))
                cand = (area, int(jdx))
                if parent_choice is None or cand < parent_choice:
                    parent_choice = cand
        if parent_choice is not None:
            parent_idx = int(parent_choice[1])
            parents[idx] = int(parent_idx)
            children[parent_idx].append(int(idx))
    for idx in children:
        children[idx].sort(key=lambda child_idx: frames[child_idx]["bbox"])
    return parents, children


def _extract_patch(
    structure: np.ndarray,
    bg: int,
    frame: dict[str, Any],
    child_indices: list[int],
    frames: list[dict[str, Any]],
) -> np.ndarray:
    r0, c0, r1, c1 = frame["bbox"]
    patch = np.asarray(structure[r0 : r1 + 1, c0 : c1 + 1], dtype=int).copy()
    for child_idx in child_indices:
        cr0, cc0, cr1, cc1 = frames[int(child_idx)]["bbox"]
        patch[int(cr0 - r0) : int(cr1 - r0) + 1, int(cc0 - c0) : int(cc1 - c0) + 1] = int(bg)
    return patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Unpack nested colored frames and lay each frame out horizontally using the legend order.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    legend_row, legend_colors = _find_legend(g, bg)

    structure = g.copy()
    structure[int(legend_row), :] = int(bg)
    frames = _find_frames(g, bg, legend_row)
    _parents, children = _build_tree(frames)

    by_color = {int(frame["color"]): idx for idx, frame in enumerate(frames)}
    ordered_indices = [int(by_color[color]) for color in legend_colors if int(color) in by_color]
    assert len(ordered_indices) == len(frames), "legend colors must cover every frame exactly once"

    patches = [
        _extract_patch(structure, bg, frames[idx], children[idx], frames)
        for idx in ordered_indices
    ]
    max_height = max(int(patch.shape[0]) for patch in patches)
    total_width = sum(int(patch.shape[1]) for patch in patches) + max(0, len(patches) - 1)
    out = np.full((max_height, total_width), int(bg), dtype=int)

    cursor = 0
    for patch in patches:
        ph, pw = map(int, patch.shape)
        top = int((max_height - ph) // 2)
        out[top : top + ph, cursor : cursor + pw] = patch
        cursor += pw + 1

    return out


COLOR_POOL = tuple(range(10))
MAX_GENERATION_ATTEMPTS = 400
TOPOLOGIES = (
    ((1, 2), (), ()),
    ((1, 2, 3), (), (), ()),
    ((1, 2, 3, 4), (), (), (), ()),
    ((1,), (2,), ()),
    ((1, 2), (), (3,), ()),
    ((1, 2), (3,), (4,), (), ()),
)


def _make_leaf(node_id: int, color: int, content_colors: list[int]) -> dict[str, Any]:
    height = random.randint(4, 7)
    width = random.randint(4, 7)
    cells = [(r, c) for r in range(1, height - 1) for c in range(1, width - 1)]
    dot_count = random.randint(1, min(4, len(cells)))
    content = []
    for r, c in random.sample(cells, dot_count):
        content.append((int(r), int(c), int(random.choice(content_colors))))
    return {
        "id": int(node_id),
        "color": int(color),
        "height": int(height),
        "width": int(width),
        "children": [],
        "content": content,
    }


def _layout_children(children: list[dict[str, Any]]) -> tuple[int, int, list[tuple[dict[str, Any], int, int]]]:
    shuffled = list(children)
    random.shuffle(shuffled)
    if len(shuffled) == 1:
        child = shuffled[0]
        top_gap = random.randint(1, 3)
        bottom_gap = random.randint(1, 3)
        left_gap = random.randint(1, 3)
        right_gap = random.randint(1, 3)
        height = int(child["height"] + top_gap + bottom_gap + 2)
        width = int(child["width"] + left_gap + right_gap + 2)
        return height, width, [(child, int(1 + top_gap), int(1 + left_gap))]

    heights = [int(child["height"]) for child in shuffled]
    widths = [int(child["width"]) for child in shuffled]
    min_h_span = max(heights)
    min_w_span = sum(widths) + (len(shuffled) - 1)
    min_h_stack = sum(heights) + (len(shuffled) - 1)
    min_w_stack = max(widths)
    if max(min_h_span, min_w_span) < max(min_h_stack, min_w_stack):
        orientation = "horizontal"
    elif max(min_h_span, min_w_span) > max(min_h_stack, min_w_stack):
        orientation = "vertical"
    else:
        orientation = random.choice(("horizontal", "vertical"))

    top_gap = random.randint(1, 2)
    bottom_gap = random.randint(1, 2)
    left_gap = random.randint(1, 2)
    right_gap = random.randint(1, 2)
    between = [random.randint(1, 2) for _ in range(len(shuffled) - 1)]

    placements: list[tuple[dict[str, Any], int, int]] = []
    if orientation == "horizontal":
        band_height = max(heights) + random.randint(0, 2)
        cursor = left_gap
        for idx, child in enumerate(shuffled):
            slack = int(band_height - child["height"])
            y = top_gap + (0 if slack == 0 else random.randint(0, slack))
            placements.append((child, int(1 + y), int(1 + cursor)))
            cursor += int(child["width"])
            if idx < len(shuffled) - 1:
                cursor += between[idx]
        height = int(2 + top_gap + band_height + bottom_gap)
        width = int(2 + left_gap + sum(widths) + sum(between) + right_gap)
    else:
        band_width = max(widths) + random.randint(0, 2)
        cursor = top_gap
        for idx, child in enumerate(shuffled):
            slack = int(band_width - child["width"])
            x = left_gap + (0 if slack == 0 else random.randint(0, slack))
            placements.append((child, int(1 + cursor), int(1 + x)))
            cursor += int(child["height"])
            if idx < len(shuffled) - 1:
                cursor += between[idx]
        height = int(2 + top_gap + sum(heights) + sum(between) + bottom_gap)
        width = int(2 + left_gap + band_width + right_gap)

    return height, width, placements


def _make_internal(node_id: int, color: int, children: list[dict[str, Any]]) -> dict[str, Any]:
    height, width, placements = _layout_children(children)
    return {
        "id": int(node_id),
        "color": int(color),
        "height": int(height),
        "width": int(width),
        "children": placements,
        "content": [],
    }


def _build_node(
    node_id: int,
    topology: tuple[tuple[int, ...], ...],
    colors: list[int],
    content_colors: list[int],
) -> dict[str, Any]:
    child_ids = [int(child_id) for child_id in topology[node_id]]
    if not child_ids:
        return _make_leaf(node_id, int(colors[node_id]), content_colors)
    children = [
        _build_node(int(child_id), topology, colors, content_colors)
        for child_id in child_ids
    ]
    return _make_internal(node_id, int(colors[node_id]), children)


def _render_node(node: dict[str, Any], bg: int) -> np.ndarray:
    patch = np.full((int(node["height"]), int(node["width"])), int(bg), dtype=int)
    color = int(node["color"])
    patch[0, :] = int(color)
    patch[-1, :] = int(color)
    patch[:, 0] = int(color)
    patch[:, -1] = int(color)
    for r, c, value in node["content"]:
        patch[int(r), int(c)] = int(value)
    for child, row, col in node["children"]:
        child_patch = _render_node(child, bg)
        patch[int(row) : int(row) + int(child["height"]), int(col) : int(col) + int(child["width"])] = child_patch
    return patch


def _extract_expected_patch(node: dict[str, Any], bg: int) -> np.ndarray:
    patch = _render_node(node, bg)
    for child, row, col in node["children"]:
        patch[int(row) : int(row) + int(child["height"]), int(col) : int(col) + int(child["width"])] = int(bg)
    return patch


def _collect_nodes(node: dict[str, Any], depth: int = 0) -> list[tuple[dict[str, Any], int]]:
    out = [(node, int(depth))]
    for child, _row, _col in node["children"]:
        out.extend(_collect_nodes(child, int(depth + 1)))
    return out


def _is_monotone_depths(depths: list[int]) -> bool:
    nondecreasing = all(depths[idx] <= depths[idx + 1] for idx in range(len(depths) - 1))
    nonincreasing = all(depths[idx] >= depths[idx + 1] for idx in range(len(depths) - 1))
    return bool(nondecreasing or nonincreasing)


def _sample_legend_order(nodes_with_depth: list[tuple[dict[str, Any], int]]) -> list[int]:
    node_ids = [int(node["id"]) for node, _depth in nodes_with_depth]
    depth_by_id = {int(node["id"]): int(depth) for node, depth in nodes_with_depth}
    for _ in range(100):
        order = list(node_ids)
        random.shuffle(order)
        depths = [int(depth_by_id[node_id]) for node_id in order]
        if not _is_monotone_depths(depths):
            return order
    return order


def _render_expected(root: dict[str, Any], bg: int, legend_order: list[int]) -> np.ndarray:
    node_by_id = {int(node["id"]): node for node, _depth in _collect_nodes(root)}
    patches = [_extract_expected_patch(node_by_id[int(node_id)], bg) for node_id in legend_order]
    max_height = max(int(patch.shape[0]) for patch in patches)
    total_width = sum(int(patch.shape[1]) for patch in patches) + len(patches) - 1
    out = np.full((max_height, total_width), int(bg), dtype=int)
    cursor = 0
    for patch in patches:
        ph, pw = map(int, patch.shape)
        top = int((max_height - ph) // 2)
        out[top : top + ph, cursor : cursor + pw] = patch
        cursor += pw + 1
    return out


def _make_legend_row(colors: list[int], bg: int) -> np.ndarray:
    left_pad = random.randint(1, 3)
    right_pad = random.randint(1, 3)
    gaps = [random.randint(1, 3) for _ in range(len(colors) - 1)]
    width = left_pad + right_pad + len(colors) + sum(gaps)
    row = np.full(width, int(bg), dtype=int)
    cursor = left_pad
    for idx, color in enumerate(colors):
        row[int(cursor)] = int(color)
        cursor += 1
        if idx < len(colors) - 1:
            cursor += gaps[idx]
    return row


def _assemble_input(root: dict[str, Any], bg: int, legend_colors: list[int]) -> np.ndarray:
    structure = _render_node(root, bg)
    legend_row = _make_legend_row(legend_colors, bg)

    width = max(int(structure.shape[1]) + 2, int(legend_row.shape[0]) + 2) + random.randint(0, 4)
    structure_col = random.randint(1, width - int(structure.shape[1]) - 1)
    legend_col = random.randint(1, width - int(legend_row.shape[0]) - 1)

    if random.random() < 0.5:
        sep = random.randint(1, 3)
        bottom_pad = random.randint(1, 3)
        height = 1 + sep + int(structure.shape[0]) + bottom_pad
        grid = np.full((height, width), int(bg), dtype=int)
        grid[0, legend_col : legend_col + int(legend_row.shape[0])] = legend_row
        grid[1 + sep : 1 + sep + int(structure.shape[0]), structure_col : structure_col + int(structure.shape[1])] = structure
    else:
        top_pad = random.randint(1, 3)
        sep = random.randint(1, 3)
        height = top_pad + int(structure.shape[0]) + sep + 1
        grid = np.full((height, width), int(bg), dtype=int)
        grid[top_pad : top_pad + int(structure.shape[0]), structure_col : structure_col + int(structure.shape[1])] = structure
        grid[-1, legend_col : legend_col + int(legend_row.shape[0])] = legend_row
    return grid


def _generate_attempt() -> np.ndarray | None:
    topology = random.choice(TOPOLOGIES)
    frame_count = len(topology)
    palette = list(COLOR_POOL)
    bg = int(random.choice(palette))
    palette.remove(bg)
    frame_colors = random.sample(palette, frame_count)
    content_pool = [int(color) for color in palette if int(color) not in frame_colors]
    if not content_pool:
        return None

    root = _build_node(0, topology, frame_colors, content_pool)
    if max(int(root["height"]), int(root["width"])) > 28:
        return None

    nodes_with_depth = _collect_nodes(root)
    legend_order = _sample_legend_order(nodes_with_depth)
    expected = _render_expected(root, bg, legend_order)
    legend_colors = [int(frame_colors[node_id]) for node_id in legend_order]
    grid = _assemble_input(root, bg, legend_colors)

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
