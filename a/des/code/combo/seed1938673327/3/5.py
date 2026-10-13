import random
from typing import Any

import numpy as np


BG_COLOR = 7  # orange
COLOR_POOL = [c for c in range(10) if c != BG_COLOR]
GENERATION_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
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
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _is_frame_component(grid: np.ndarray, color: int, coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if int(coords.shape[0]) != 2 * (h + w) - 4:
        return False

    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if not np.all(sub[border] == color):
        return False
    if np.any(sub[~border] == color):
        return False
    return True


def _find_frames(grid: np.ndarray, bg: int) -> tuple[list[dict[str, Any]], np.ndarray]:
    frames: list[dict[str, Any]] = []
    frame_mask = np.zeros(grid.shape, dtype=bool)
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for coords in _components(grid == color):
            if not _is_frame_component(grid, color, coords):
                continue
            r0, c0, r1, c1 = _bbox(coords)
            frame_mask[coords[:, 0], coords[:, 1]] = True
            frames.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "coords": coords,
                }
            )
    assert frames, "expected at least one rectangular frame"
    return frames, frame_mask


def _find_legend(grid: np.ndarray, bg: int, frame_mask: np.ndarray) -> list[int]:
    residual = (grid != bg) & (~frame_mask)
    comps = _components(residual)
    assert len(comps) == 1, f"expected one legend component, found {len(comps)}"
    coords = comps[0]
    r0, c0, r1, c1 = _bbox(coords)
    assert r0 == r1, "expected a one-row legend strip"
    assert int(coords.shape[0]) == c1 - c0 + 1, "expected a fully filled legend strip"
    legend = [int(v) for v in grid[r0, c0 : c1 + 1]]
    assert len(legend) >= 1
    return legend


def _contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return or0 < ir0 and oc0 < ic0 and or1 > ir1 and oc1 > ic1


def _frame_depths(frames: list[dict[str, Any]]) -> list[int]:
    parents: list[int | None] = [None] * len(frames)
    areas = []
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        areas.append((r1 - r0 + 1) * (c1 - c0 + 1))

    for child_idx, child in enumerate(frames):
        containers = []
        for parent_idx, parent in enumerate(frames):
            if child_idx == parent_idx:
                continue
            if _contains(parent["bbox"], child["bbox"]):
                containers.append(parent_idx)
        if containers:
            parents[child_idx] = min(containers, key=lambda idx: areas[idx])

    depths = [-1] * len(frames)

    def depth(idx: int) -> int:
        if depths[idx] >= 0:
            return depths[idx]
        parent = parents[idx]
        depths[idx] = 0 if parent is None else depth(parent) + 1
        return depths[idx]

    for idx in range(len(frames)):
        depth(idx)
    return depths


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each rectangular frame according to its nesting depth using the outside legend strip.

    Input structure:
    - Orange background.
    - Several 1-pixel-thick closed rectangular frames of arbitrary colors.
    - A horizontal legend strip outside all outer frames.

    Rule:
    - Build the containment tree induced by frame interiors.
    - Depth 0 frames take the first legend color, depth 1 frames take the second, etc.
    - Keep the background, legend, and all frame shapes/positions unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    frames, frame_mask = _find_frames(g, bg)
    legend = _find_legend(g, bg, frame_mask)
    depths = _frame_depths(frames)

    assert max(depths) < len(legend), "legend is shorter than the maximum nesting depth"

    out = g.copy()
    for frame, depth in zip(frames, depths):
        color = int(legend[depth])
        coords = frame["coords"]
        out[coords[:, 0], coords[:, 1]] = color
    return out


def _leaf() -> dict[str, Any]:
    return {"children": []}


def _chain(levels: int) -> dict[str, Any]:
    assert levels >= 1
    if levels == 1:
        return _leaf()
    return {"children": [_chain(levels - 1)]}


def _branch(levels: int) -> dict[str, Any]:
    assert levels >= 2
    if levels == 2:
        child_count = random.randint(2, 3)
        return {"children": [_leaf() for _ in range(child_count)]}

    deep_child = _branch(levels - 1) if random.random() < 0.6 else _chain(levels - 1)
    children = [deep_child]
    extra_children = 1 if levels >= 4 else random.randint(1, 2)
    for _ in range(extra_children):
        children.append(_leaf())
    random.shuffle(children)
    return {"children": children}


def _count_nodes(node: dict[str, Any]) -> int:
    return 1 + sum(_count_nodes(child) for child in node["children"])


def _assign_colors(roots: list[dict[str, Any]]) -> None:
    nodes: list[dict[str, Any]] = []

    def walk(node: dict[str, Any]) -> None:
        nodes.append(node)
        for child in node["children"]:
            walk(child)

    for root in roots:
        walk(root)
    colors = random.sample(COLOR_POOL, len(nodes))
    for node, color in zip(nodes, colors):
        node["color"] = int(color)


def _draw_frame(patch: np.ndarray, color: int) -> None:
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color


def _compose_children(children: list[np.ndarray], margin_low: int, margin_high: int) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    if len(children) == 1:
        child = children[0]
        top = random.randint(margin_low, margin_high)
        bottom = random.randint(margin_low, margin_high)
        left = random.randint(margin_low, margin_high)
        right = random.randint(margin_low, margin_high)
        h = child.shape[0] + top + bottom
        w = child.shape[1] + left + right
        return (h, w), [(top, left)]

    orient = random.choice(["horizontal", "vertical"])
    gaps = [random.randint(2, 4) for _ in range(len(children) - 1)]
    top = random.randint(margin_low, margin_high)
    bottom = random.randint(margin_low, margin_high)
    left = random.randint(margin_low, margin_high)
    right = random.randint(margin_low, margin_high)
    extra = random.randint(0, 2)

    if orient == "horizontal":
        max_h = max(child.shape[0] for child in children)
        h = top + max_h + bottom + extra
        w = left + sum(child.shape[1] for child in children) + sum(gaps) + right
        positions = []
        col = left
        for idx, child in enumerate(children):
            max_row = h - bottom - child.shape[0]
            row = random.randint(top, max_row)
            positions.append((row, col))
            col += child.shape[1]
            if idx < len(gaps):
                col += gaps[idx]
        return (h, w), positions

    max_w = max(child.shape[1] for child in children)
    h = top + sum(child.shape[0] for child in children) + sum(gaps) + bottom
    w = left + max_w + right + extra
    positions = []
    row = top
    for idx, child in enumerate(children):
        max_col = w - right - child.shape[1]
        col = random.randint(left, max_col)
        positions.append((row, col))
        row += child.shape[0]
        if idx < len(gaps):
            row += gaps[idx]
    return (h, w), positions


def _render_node(node: dict[str, Any], depth: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
    children = node["children"]
    if not children:
        h = random.randint(3, 6)
        w = random.randint(3, 8)
        patch = np.full((h, w), BG_COLOR, dtype=int)
        _draw_frame(patch, int(node["color"]))
        return patch, [{"bbox": (0, 0, h - 1, w - 1), "depth": depth}]

    child_results = [_render_node(child, depth + 1) for child in children]
    child_patches = [result[0] for result in child_results]
    (h, w), positions = _compose_children(child_patches, margin_low=2, margin_high=4)

    patch = np.full((h, w), BG_COLOR, dtype=int)
    frames = [{"bbox": (0, 0, h - 1, w - 1), "depth": depth}]
    for (child_patch, child_frames), (r0, c0) in zip(child_results, positions):
        ch, cw = child_patch.shape
        patch[r0 : r0 + ch, c0 : c0 + cw] = child_patch
        for frame in child_frames:
            fr0, fc0, fr1, fc1 = frame["bbox"]
            frames.append(
                {
                    "bbox": (fr0 + r0, fc0 + c0, fr1 + r0, fc1 + c0),
                    "depth": frame["depth"],
                }
            )
    _draw_frame(patch, int(node["color"]))
    return patch, frames


def _compose_forest(roots: list[dict[str, Any]]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    root_results = [_render_node(root, depth=0) for root in roots]
    root_patches = [result[0] for result in root_results]
    if len(root_patches) == 1:
        return root_results[0]

    (h, w), positions = _compose_children(root_patches, margin_low=0, margin_high=2)
    patch = np.full((h, w), BG_COLOR, dtype=int)
    frames: list[dict[str, Any]] = []
    for (root_patch, root_frames), (r0, c0) in zip(root_results, positions):
        rh, rw = root_patch.shape
        patch[r0 : r0 + rh, c0 : c0 + rw] = root_patch
        for frame in root_frames:
            fr0, fc0, fr1, fc1 = frame["bbox"]
            frames.append(
                {
                    "bbox": (fr0 + r0, fc0 + c0, fr1 + r0, fc1 + c0),
                    "depth": frame["depth"],
                }
            )
    return patch, frames


def _build_scene() -> tuple[np.ndarray, np.ndarray]:
    mode = random.choice(["flat", "chain", "branch"])
    if mode == "flat":
        roots = [_leaf() for _ in range(random.randint(2, 4))]
        max_depth = 0
    elif mode == "chain":
        levels = random.randint(2, 4)
        roots = [_chain(levels)]
        if random.random() < 0.6:
            for _ in range(random.randint(1, 2)):
                roots.append(_leaf())
        max_depth = levels - 1
    else:
        levels = random.randint(2, 4)
        roots = [_branch(levels)]
        if random.random() < 0.5:
            roots.append(_leaf())
        max_depth = levels - 1

    total_frames = sum(_count_nodes(root) for root in roots)
    if total_frames > len(COLOR_POOL):
        raise ValueError("too many frames for distinct colors")
    _assign_colors(roots)

    forest_patch, local_frames = _compose_forest(roots)
    legend_len = max(2, max_depth + 1)
    legend_colors = random.sample(COLOR_POOL, legend_len)

    fh, fw = forest_patch.shape
    left_margin = random.randint(1, 4)
    right_margin = random.randint(1, 4)
    forest_col_slack = random.randint(0, 3)
    width = max(fw + left_margin + right_margin + forest_col_slack, legend_len + left_margin + right_margin + 2)

    legend_on_top = random.random() < 0.5
    gap = random.randint(1, 3)
    top_padding = random.randint(1, 3)
    bottom_padding = random.randint(1, 4)
    height = fh + top_padding + bottom_padding + gap + 1

    grid = np.full((height, width), BG_COLOR, dtype=int)
    max_forest_col = width - fw - right_margin
    forest_col = random.randint(left_margin, max_forest_col)

    if legend_on_top:
        legend_row = top_padding
        forest_row = legend_row + 1 + gap
    else:
        forest_row = top_padding
        legend_row = forest_row + fh + gap

    grid[forest_row : forest_row + fh, forest_col : forest_col + fw] = forest_patch

    legend_start_min = 1
    legend_start_max = width - legend_len - 1
    legend_col = random.randint(legend_start_min, legend_start_max)
    grid[legend_row, legend_col : legend_col + legend_len] = np.array(legend_colors, dtype=int)

    global_frames = []
    for frame in local_frames:
        r0, c0, r1, c1 = frame["bbox"]
        global_frames.append(
            {
                "bbox": (r0 + forest_row, c0 + forest_col, r1 + forest_row, c1 + forest_col),
                "depth": frame["depth"],
            }
        )

    expected = grid.copy()
    for frame in global_frames:
        r0, c0, r1, c1 = frame["bbox"]
        color = int(legend_colors[frame["depth"]])
        expected[r0, c0 : c1 + 1] = color
        expected[r1, c0 : c1 + 1] = color
        expected[r0 : r1 + 1, c0] = color
        expected[r0 : r1 + 1, c1] = color

    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        try:
            grid, expected = _build_scene()
        except Exception:
            continue
        try:
            actual = main(grid)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate a valid input")
