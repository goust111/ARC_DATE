from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
LINE = 1
RED = 2
GREEN = 3
YELLOW = 4

PAD = 1
MAX_GENERATION_ATTEMPTS = 300
MAIN_CHAIN_DEPTHS = (3, 4)
SIDE_BRANCH_PROB = 0.45
EXTRA_ROOT_BRANCH_PROB = 0.55
EXTRA_BRANCH_DEPTHS = (1, 2, 3)
SIZE_SLACK_CAP = 6

NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


class _TreeNode:
    def __init__(self, children: list["_TreeNode"] | None = None, layout: str | None = None):
        self.children = children or []
        self.layout = layout or random.choice(("h", "v"))
        self.min_region_h = 1
        self.min_region_w = 1
        self.min_outer_h = 3
        self.min_outer_w = 3


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for dr, dc in NEIGHBORS4]


def _label_mask(mask: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    components: list[list[tuple[int, int]]] = []
    label = 0

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            label += 1
            coords: list[tuple[int, int]] = []
            q = deque([(r, c)])
            labels[r, c] = label
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = label
                        q.append((nr, nc))
            components.append(coords)

    return labels, components


def _analyze_regions(grid: np.ndarray) -> tuple[int, np.ndarray, list[list[tuple[int, int]]], dict[int, int], dict[int, set[int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert np.any(g == BG), "expected black background cells"

    colors = {int(v) for v in np.unique(g)}
    colors.discard(BG)
    assert len(colors) == 1, "expected exactly one contour color"
    line_color = next(iter(colors))

    bg_labels, bg_components = _label_mask(g == BG)
    assert bg_components, "expected at least one background region"

    outer_ids = {
        int(bg_labels[r, c])
        for r in range(g.shape[0])
        for c in (0, g.shape[1] - 1)
        if bg_labels[r, c] != 0
    }
    outer_ids |= {
        int(bg_labels[r, c])
        for c in range(g.shape[1])
        for r in (0, g.shape[0] - 1)
        if bg_labels[r, c] != 0
    }
    assert len(outer_ids) == 1, "expected a unique border-connected outer background region"
    outer_id = next(iter(outer_ids))

    _, line_components = _label_mask(g == line_color)
    assert line_components, "expected at least one closed contour"

    region_graph = {idx: set() for idx in range(1, len(bg_components) + 1)}
    for comp in line_components:
        adjacent_regions: set[int] = set()
        for r, c in comp:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and bg_labels[nr, nc] != 0:
                    adjacent_regions.add(int(bg_labels[nr, nc]))
        assert len(adjacent_regions) == 2, "each contour must separate exactly two background regions"
        a, b = tuple(adjacent_regions)
        region_graph[a].add(b)
        region_graph[b].add(a)

    depths = {outer_id: 0}
    q = deque([outer_id])
    while q:
        region_id = q.popleft()
        for neighbor in region_graph[region_id]:
            if neighbor in depths:
                continue
            depths[neighbor] = depths[region_id] + 1
            q.append(neighbor)

    assert len(depths) == len(bg_components), "all background regions must be reachable in the containment graph"
    return line_color, bg_labels, bg_components, depths, region_graph


def main(grid: np.ndarray) -> np.ndarray:
    """Fill nested enclosed background regions by depth parity, then override leaf regions in red.

    Input structure:
    - bg = black.
    - A single foreground color (blue) forms separated 1-pixel-thick closed contours.
    - The contours do not touch and may be nested, so the black 4-connected regions form a containment tree.

    Rule:
    - Find every black 4-connected region and treat the border-touching one as depth 0.
    - Across each blue contour, the adjacent black region on the inside is one level deeper than the one on the outside.
    - Fill odd-depth enclosed regions yellow and even-depth enclosed regions green, then override every non-root region with no deeper child regions to red.
    - Keep the blue contour pixels and the depth-0 outer background unchanged.

    Key insight:
    - The visible nesting of closed contours induces a region tree; parity colors the interior levels, while the deepest leaves ignore parity and become red.
    """
    g = np.asarray(grid, dtype=int)
    line_color, _, bg_components, depths, region_graph = _analyze_regions(g)

    out = g.copy()
    for region_id, coords in enumerate(bg_components, start=1):
        depth = depths[region_id]
        if depth == 0:
            continue
        has_child = any(depths[neighbor] == depth + 1 for neighbor in region_graph[region_id])
        fill_color = RED if not has_child else (YELLOW if depth % 2 == 1 else GREEN)
        for r, c in coords:
            out[r, c] = fill_color

    out[g == line_color] = line_color
    return out


def _build_chain(depth: int) -> _TreeNode:
    if depth <= 1:
        return _TreeNode([])
    return _TreeNode([_build_chain(depth - 1)])


def _shuffle_tree(node: _TreeNode) -> None:
    random.shuffle(node.children)
    for child in node.children:
        _shuffle_tree(child)


def _annotate_sizes(node: _TreeNode) -> None:
    for child in node.children:
        _annotate_sizes(child)

    if not node.children:
        node.min_region_h = 1
        node.min_region_w = 1
        node.min_outer_h = 3
        node.min_outer_w = 3
        return

    hs = [child.min_outer_h for child in node.children]
    ws = [child.min_outer_w for child in node.children]
    if node.layout == "h":
        node.min_region_h = max(hs) + 2 * PAD
        node.min_region_w = sum(ws) + (len(node.children) + 1) * PAD
    else:
        node.min_region_h = sum(hs) + (len(node.children) + 1) * PAD
        node.min_region_w = max(ws) + 2 * PAD

    node.min_outer_h = node.min_region_h + 2
    node.min_outer_w = node.min_region_w + 2


def _split_extra(total: int, parts: int) -> list[int]:
    buckets = [0] * parts
    for _ in range(total):
        buckets[random.randrange(parts)] += 1
    return buckets


def _draw_frame(grid: np.ndarray, rect: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = rect
    grid[r0, c0 : c1 + 1] = LINE
    grid[r1, c0 : c1 + 1] = LINE
    grid[r0 : r1 + 1, c0] = LINE
    grid[r0 : r1 + 1, c1] = LINE


def _place_children(grid: np.ndarray, region: tuple[int, int, int, int], node: _TreeNode) -> bool:
    if not node.children:
        return True

    r0, c0, r1, c1 = region
    region_h = r1 - r0 + 1
    region_w = c1 - c0 + 1
    children = node.children

    if node.layout == "h":
        if region_h < node.min_region_h or region_w < node.min_region_w:
            return False
        child_mins_h = [child.min_outer_h for child in children]
        child_mins_w = [child.min_outer_w for child in children]
        extra_w = region_w - node.min_region_w
        gap_and_width = _split_extra(extra_w, 2 * len(children) + 1)
        width_extras = gap_and_width[: len(children)]
        gap_extras = gap_and_width[len(children) :]

        current_c = c0 + PAD + gap_extras[0]
        max_child_h = region_h - 2 * PAD
        for idx, child in enumerate(children):
            width = child_mins_w[idx] + width_extras[idx]
            height = random.randint(child_mins_h[idx], max_child_h)
            top = random.randint(r0 + PAD, r1 - PAD - height + 1)
            rect = (top, current_c, top + height - 1, current_c + width - 1)
            _draw_frame(grid, rect)
            if not _place_children(grid, (rect[0] + 1, rect[1] + 1, rect[2] - 1, rect[3] - 1), child):
                return False
            current_c = rect[3] + 1 + PAD + gap_extras[idx + 1]
    else:
        if region_h < node.min_region_h or region_w < node.min_region_w:
            return False
        child_mins_h = [child.min_outer_h for child in children]
        child_mins_w = [child.min_outer_w for child in children]
        extra_h = region_h - node.min_region_h
        gap_and_height = _split_extra(extra_h, 2 * len(children) + 1)
        height_extras = gap_and_height[: len(children)]
        gap_extras = gap_and_height[len(children) :]

        current_r = r0 + PAD + gap_extras[0]
        max_child_w = region_w - 2 * PAD
        for idx, child in enumerate(children):
            height = child_mins_h[idx] + height_extras[idx]
            width = random.randint(child_mins_w[idx], max_child_w)
            left = random.randint(c0 + PAD, c1 - PAD - width + 1)
            rect = (current_r, left, current_r + height - 1, left + width - 1)
            _draw_frame(grid, rect)
            if not _place_children(grid, (rect[0] + 1, rect[1] + 1, rect[2] - 1, rect[3] - 1), child):
                return False
            current_r = rect[2] + 1 + PAD + gap_extras[idx + 1]
    return True


def _build_tree() -> _TreeNode:
    main_depth = random.choice(MAIN_CHAIN_DEPTHS)
    main_branch = _build_chain(main_depth)

    cursor = main_branch
    while cursor.children:
        main_child = cursor.children[0]
        if random.random() < SIDE_BRANCH_PROB:
            cursor.children.append(_build_chain(random.choice((1, 2))))
        cursor.layout = random.choice(("h", "v"))
        cursor = main_child

    root_children = [main_branch]
    if random.random() < EXTRA_ROOT_BRANCH_PROB:
        root_children.append(_build_chain(random.choice(EXTRA_BRANCH_DEPTHS)))
    root = _TreeNode(root_children, random.choice(("h", "v")))
    _shuffle_tree(root)
    _annotate_sizes(root)
    return root


def _gen_attempt() -> np.ndarray | None:
    root = _build_tree()
    min_h, min_w = root.min_region_h, root.min_region_w
    if min_h > 30 or min_w > 30:
        return None

    extra_h = random.randint(0, min(SIZE_SLACK_CAP, 30 - min_h))
    extra_w = random.randint(0, min(SIZE_SLACK_CAP, 30 - min_w))
    grid = np.full((min_h + extra_h, min_w + extra_w), BG, dtype=int)

    if not _place_children(grid, (0, 0, grid.shape[0] - 1, grid.shape[1] - 1), root):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if not (np.any(out == RED) and np.any(out == GREEN) and np.any(out == YELLOW)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-contour ARC input")
