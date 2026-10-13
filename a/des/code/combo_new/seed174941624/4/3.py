from __future__ import annotations

import random

import numpy as np


_DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_COLORS = tuple(range(1, 10))
_MAX_RETRIES = 400


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _bbox_height(bbox: tuple[int, int, int, int]) -> int:
    return bbox[2] - bbox[0] + 1


def _bbox_width(bbox: tuple[int, int, int, int]) -> int:
    return bbox[3] - bbox[1] + 1


def _bbox_contains(
    outer: tuple[int, int, int, int], inner: tuple[int, int, int, int], margin: int = 0
) -> bool:
    return (
        outer[0] + margin <= inner[0]
        and outer[1] + margin <= inner[1]
        and inner[2] <= outer[2] - margin
        and inner[3] <= outer[3] - margin
    )


def _bbox_separated(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1
) -> bool:
    return (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _find_components(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == 0 or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            r0 = r1 = r
            c0 = c1 = c
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                r0 = min(r0, cr)
                c0 = min(c0, cc)
                r1 = max(r1, cr)
                c1 = max(c1, cc)
                for dr, dc in _DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            comps.append(
                {
                    "color": color,
                    "coords": tuple(coords),
                    "bbox": (r0, c0, r1, c1),
                    "area": len(coords),
                }
            )

    return comps


def _is_rectlike_component(comp: dict[str, object]) -> bool:
    bbox = comp["bbox"]
    assert isinstance(bbox, tuple)
    r0, c0, r1, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    mask = np.zeros((h, w), dtype=bool)
    coords = comp["coords"]
    assert isinstance(coords, tuple)
    for r, c in coords:
        mask[r - r0, c - c0] = True

    if mask.all():
        return True
    if h >= 3 and w >= 3:
        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if np.array_equal(mask, border):
            return True
    return False


def _find_legend(
    singleton_comps: list[dict[str, object]], shape: tuple[int, int]
) -> tuple[list[int], set[tuple[int, int]]]:
    if not singleton_comps:
        raise AssertionError("expected at least one singleton legend pixel")

    h, w = shape
    cols = sorted({int(comp["bbox"][1]) for comp in singleton_comps})
    runs: list[tuple[tuple[int, int, int, int], list[tuple[int, int]]]] = []
    for col in cols:
        items = sorted((int(comp["bbox"][0]), int(comp["color"])) for comp in singleton_comps if int(comp["bbox"][1]) == col)
        start = 0
        while start < len(items):
            run = [items[start]]
            end = start + 1
            while end < len(items) and items[end][0] == run[-1][0] + 1:
                run.append(items[end])
                end += 1
            top = run[0][0]
            bottom = run[-1][0]
            preferred = int(col >= w // 2 and bottom >= h // 2)
            runs.append(((preferred, len(run), col, bottom), run))
            start = end

    if not runs:
        raise AssertionError("failed to find legend run")

    best_score, best_run = max(runs, key=lambda item: item[0])
    colors = [color for _row, color in best_run]
    legend_col = int(best_score[2])
    cells = {(row, legend_col) for row, _color in best_run}
    return colors, cells


def _analyze_input(grid: np.ndarray) -> tuple[list[int], list[dict[str, object]], list[int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    comps = _find_components(g)

    singletons = [comp for comp in comps if int(comp["area"]) == 1]
    legend_colors, legend_cells = _find_legend(singletons, g.shape)

    rects: list[dict[str, object]] = []
    for comp in comps:
        if int(comp["area"]) == 1:
            cell = comp["coords"][0]
            if cell in legend_cells:
                continue
            continue
        if _is_rectlike_component(comp):
            rects.append(comp)

    assert rects, "expected at least one rectangular object"

    parents: list[int | None] = [None] * len(rects)
    for i, child in enumerate(rects):
        child_bbox = child["bbox"]
        assert isinstance(child_bbox, tuple)
        candidates: list[tuple[int, int]] = []
        for j, outer in enumerate(rects):
            if i == j:
                continue
            outer_bbox = outer["bbox"]
            assert isinstance(outer_bbox, tuple)
            if _bbox_contains(outer_bbox, child_bbox, margin=2):
                candidates.append((_bbox_area(outer_bbox), j))
        if candidates:
            parents[i] = min(candidates)[1]

    roots = [i for i, parent in enumerate(parents) if parent is None]
    assert len(roots) == 1, f"expected one containment root, found {len(roots)}"

    depths: list[int | None] = [None] * len(rects)

    def depth_of(idx: int) -> int:
        cached = depths[idx]
        if cached is not None:
            return cached
        parent = parents[idx]
        value = 0 if parent is None else depth_of(parent) + 1
        depths[idx] = value
        return value

    resolved = [depth_of(i) for i in range(len(rects))]
    max_depth = max(resolved)
    assert len(legend_colors) >= max_depth + 1, "legend does not cover every nesting depth"
    return legend_colors, rects, resolved


def _draw_frame(out: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    out[r0, c0 : c1 + 1] = color
    out[r1, c0 : c1 + 1] = color
    out[r0 : r1 + 1, c0] = color
    out[r0 : r1 + 1, c1] = color


def _draw_object(out: np.ndarray, bbox: tuple[int, int, int, int], color: int, solid: bool) -> None:
    r0, c0, r1, c1 = bbox
    if solid:
        out[r0 : r1 + 1, c0 : c1 + 1] = color
    else:
        _draw_frame(out, bbox, color)


def main(grid: np.ndarray) -> np.ndarray:
    """Compress a containment hierarchy of rectangles into centered concentric frames recolored by a depth legend.

    Input structure:
    - The background is black.
    - Several monochrome rectangular objects appear on the canvas; some rectangles lie strictly inside others, with black gaps separating parent and child.
    - These containment relations form a single rooted tree.
    - The lower-right corner contains a vertical single-pixel legend whose top-to-bottom colors encode depth 0, depth 1, depth 2, and so on.

    Rule:
    - Detect the rectangular objects and build their containment tree to determine each object's nesting depth.
    - Ignore the objects' original colors and the legend's original position.
    - Produce a black output grid of the same size, then draw one hollow rectangular frame per depth level, centered on the grid.
    - The depth-0 frame is the outermost frame, each deeper frame is inset by two cells so adjacent frames are separated by a one-pixel black gap, and each frame uses the legend color for that depth.

    Key insight:
    - The input geometry matters only for recovering depth, while the lower-right legend remaps depth to color and the output canonicalizes the tree into evenly spaced concentric frames.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    legend_colors, _rects, depths = _analyze_input(g)
    levels = max(depths) + 1

    assert h >= 4 * levels - 1 and w >= 4 * levels - 1, "grid is too small for the requested concentric frames"

    out = np.zeros_like(g)
    for depth in range(levels):
        inset = 2 * depth
        bbox = (inset, inset, h - 1 - inset, w - 1 - inset)
        _draw_frame(out, bbox, legend_colors[depth])
    return out


def _ancestor_chain(nodes: list[dict[str, object]], idx: int) -> set[int]:
    out: set[int] = {idx}
    cur = nodes[idx]
    while cur["parent"] is not None:
        parent = int(cur["parent"])
        out.add(parent)
        cur = nodes[parent]
    return out


def _sample_bbox_within(
    parent_bbox: tuple[int, int, int, int],
    min_h: int,
    min_w: int,
    max_h: int | None = None,
    max_w: int | None = None,
) -> tuple[int, int, int, int] | None:
    pr0, pc0, pr1, pc1 = parent_bbox
    top = pr0 + 2
    left = pc0 + 2
    max_bottom = pr1 - 2
    max_right = pc1 - 2
    cap_h = max_bottom - top + 1
    cap_w = max_right - left + 1
    if cap_h < min_h or cap_w < min_w:
        return None

    high_h = cap_h if max_h is None else min(cap_h, max_h)
    high_w = cap_w if max_w is None else min(cap_w, max_w)
    if high_h < min_h or high_w < min_w:
        return None

    child_h = random.randint(min_h, high_h)
    child_w = random.randint(min_w, high_w)
    r0 = random.randint(top, max_bottom - child_h + 1)
    c0 = random.randint(left, max_right - child_w + 1)
    return (r0, c0, r0 + child_h - 1, c0 + child_w - 1)


def _can_place_extra(
    candidate: tuple[int, int, int, int], parent_idx: int, nodes: list[dict[str, object]]
) -> bool:
    allowed = _ancestor_chain(nodes, parent_idx)
    for idx, node in enumerate(nodes):
        if idx in allowed:
            continue
        bbox = node["bbox"]
        assert isinstance(bbox, tuple)
        if not _bbox_separated(candidate, bbox, gap=1):
            return False
    return True


def _generate_nodes(
    h: int, w: int, legend_col: int, levels: int
) -> list[dict[str, object]] | None:
    min_root = 4 * levels - 1
    max_root_h = h - 2
    max_root_w = legend_col - 1
    if max_root_h < min_root or max_root_w < min_root:
        return None

    root_h = random.randint(min_root, max_root_h)
    root_w = random.randint(min_root, max_root_w)
    r0 = random.randint(0, h - root_h)
    c0 = random.randint(0, legend_col - root_w)
    root_bbox = (r0, c0, r0 + root_h - 1, c0 + root_w - 1)

    nodes: list[dict[str, object]] = [
        {"bbox": root_bbox, "depth": 0, "parent": None, "solid": levels == 1 and random.random() < 0.5}
    ]

    chain_parent = 0
    for depth in range(1, levels):
        parent_bbox = nodes[chain_parent]["bbox"]
        assert isinstance(parent_bbox, tuple)
        min_dim = 4 * (levels - depth) - 1
        child_bbox = _sample_bbox_within(parent_bbox, min_dim, min_dim)
        if child_bbox is None:
            return None
        nodes.append(
            {
                "bbox": child_bbox,
                "depth": depth,
                "parent": chain_parent,
                "solid": depth == levels - 1 and random.random() < 0.65,
            }
        )
        chain_parent = len(nodes) - 1

    extra_budget = 0 if levels == 1 else random.randint(0, 2)
    for _ in range(extra_budget):
        parent_options = [idx for idx, node in enumerate(nodes) if int(node["depth"]) < levels - 1]
        random.shuffle(parent_options)
        placed = False
        for parent_idx in parent_options:
            parent_depth = int(nodes[parent_idx]["depth"])
            parent_bbox = nodes[parent_idx]["bbox"]
            assert isinstance(parent_bbox, tuple)
            max_h = max(3, min(_bbox_height(parent_bbox) - 4, 9))
            max_w = max(3, min(_bbox_width(parent_bbox) - 4, 9))
            for _attempt in range(60):
                candidate = _sample_bbox_within(parent_bbox, 3, 3, max_h=max_h, max_w=max_w)
                if candidate is None:
                    break
                if _can_place_extra(candidate, parent_idx, nodes):
                    nodes.append(
                        {
                            "bbox": candidate,
                            "depth": parent_depth + 1,
                            "parent": parent_idx,
                            "solid": random.random() < 0.7,
                        }
                    )
                    placed = True
                    break
            if placed:
                break

    return nodes


def _build_input() -> np.ndarray | None:
    levels = random.randint(1, 4)
    min_side = max(4 * levels - 1, 14)
    h = random.randint(min_side, 30)
    w = random.randint(min_side, 30)
    legend_col = w - random.choice((1, 2)) if w >= min_side + 2 else w - 1
    legend_start = h - levels

    nodes = _generate_nodes(h, w, legend_col, levels)
    if nodes is None:
        return None

    obj_colors = random.sample(_COLORS, len(nodes))
    legend_colors = random.sample(_COLORS, levels)

    grid = np.zeros((h, w), dtype=int)
    for node, color in zip(nodes, obj_colors):
        bbox = node["bbox"]
        assert isinstance(bbox, tuple)
        solid = bool(node["solid"])
        _draw_object(grid, bbox, color, solid)

    for depth, color in enumerate(legend_colors):
        grid[legend_start + depth, legend_col] = color

    try:
        parsed_legend, rects, depths = _analyze_input(grid)
        if len(rects) != len(nodes):
            return None
        if max(depths) + 1 != levels:
            return None
        if parsed_legend[:levels] != legend_colors:
            return None
        out = main(grid)
        if out.shape != grid.shape:
            return None
    except AssertionError:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        grid = _build_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
