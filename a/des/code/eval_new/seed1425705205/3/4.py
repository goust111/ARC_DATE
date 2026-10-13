from collections import deque
import random

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ARC_COLORS = tuple(range(1, 10))
CANVAS_SIDE = 30
GEN_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _touches_edge(coords: np.ndarray, h: int, w: int) -> bool:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return bool(
        np.any(rows == 0)
        or np.any(rows == h - 1)
        or np.any(cols == 0)
        or np.any(cols == w - 1)
    )


def _label_regions(mask: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    labels = np.zeros(mask.shape, dtype=int)
    comps = _components(mask)
    for idx, coords in enumerate(comps, start=1):
        labels[coords[:, 0], coords[:, 1]] = idx
    return labels, comps


def _color_components(grid: np.ndarray) -> list[dict]:
    comps: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != 0):
        for coords in _components(grid == color):
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                }
            )
    return comps


def _analyze_rings(grid: np.ndarray) -> tuple[list[dict], list[int], int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    ring_components = _color_components(g)
    assert ring_components, "expected at least one ring component"

    ring_mask = g != 0
    region_labels, region_coords = _label_regions(~ring_mask)
    assert region_coords, "expected at least one background region"

    exterior_ids = set()
    for r in range(h):
        if not ring_mask[r, 0]:
            exterior_ids.add(int(region_labels[r, 0]))
        if not ring_mask[r, w - 1]:
            exterior_ids.add(int(region_labels[r, w - 1]))
    for c in range(w):
        if not ring_mask[0, c]:
            exterior_ids.add(int(region_labels[0, c]))
        if not ring_mask[h - 1, c]:
            exterior_ids.add(int(region_labels[h - 1, c]))
    exterior_ids.discard(0)
    assert len(exterior_ids) == 1, "expected a unique exterior background region"
    exterior_id = next(iter(exterior_ids))

    graph = {idx: set() for idx in range(1, len(region_coords) + 1)}
    component_regions: list[tuple[int, int]] = []
    for comp in ring_components:
        coords = comp["coords"]
        assert not _touches_edge(coords, h, w), "rings must stay away from the outer border"
        adjacent_regions: set[int] = set()
        for r, c in coords:
            for dr, dc in DIR4:
                nr, nc = int(r + dr), int(c + dc)
                if 0 <= nr < h and 0 <= nc < w and not ring_mask[nr, nc]:
                    adjacent_regions.add(int(region_labels[nr, nc]))
        assert len(adjacent_regions) == 2, "each ring must separate exactly two background regions"
        assert 0 not in adjacent_regions
        a, b = sorted(adjacent_regions)
        graph[a].add(b)
        graph[b].add(a)
        component_regions.append((a, b))

    region_depth = {exterior_id: 0}
    queue = deque([exterior_id])
    while queue:
        cur = queue.popleft()
        for nxt in graph[cur]:
            if nxt not in region_depth:
                region_depth[nxt] = region_depth[cur] + 1
                queue.append(nxt)
    assert len(region_depth) == len(region_coords), "background dual graph must be connected"

    ring_depths: list[int] = []
    for a, b in component_regions:
        da = region_depth[a]
        db = region_depth[b]
        assert abs(da - db) == 1, "crossing a single ring must change nesting depth by exactly one"
        ring_depths.append(max(da, db))

    return ring_components, ring_depths, max(ring_depths)


def _palette_candidates(grid: np.ndarray) -> list[list[tuple[int, int, int]]]:
    by_col: dict[int, list[tuple[int, int]]] = {}
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != 0):
        for coords in _components(grid == color):
            if len(coords) != 1:
                continue
            r, c = map(int, coords[0])
            by_col.setdefault(c, []).append((r, color))

    candidates: list[list[tuple[int, int, int]]] = []
    for col, entries in by_col.items():
        entries.sort()
        run: list[tuple[int, int]] = []
        for item in entries:
            if run and item[0] != run[-1][0] + 1:
                colors = {color for _row, color in run}
                if 1 <= len(run) <= 4 and len(colors) == len(run):
                    candidates.append([(row, col, color) for row, color in run])
                run = [item]
            else:
                run.append(item)
        colors = {color for _row, color in run}
        if 1 <= len(run) <= 4 and len(colors) == len(run):
            candidates.append([(row, col, color) for row, color in run])
    return candidates


def _solve(grid: np.ndarray) -> tuple[list[tuple[int, int, int]], list[dict], list[int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    valid: list[tuple[list[tuple[int, int, int]], list[dict], list[int]]] = []
    for palette in _palette_candidates(g):
        trimmed = g.copy()
        for r, c, _color in palette:
            trimmed[r, c] = 0
        try:
            ring_components, ring_depths, max_depth = _analyze_rings(trimmed)
        except AssertionError:
            continue
        if max_depth != len(palette):
            continue
        valid.append((palette, ring_components, ring_depths))

    assert valid, "failed to identify the palette and nested rings"
    max_len = max(len(palette) for palette, _comps, _depths in valid)
    valid = [item for item in valid if len(item[0]) == max_len]
    assert len(valid) == 1, "expected a unique palette interpretation"
    return valid[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor nested ring regions by the depth-indexed palette and erase the palette itself.

    Input structure:
    - The background is black.
    - Several differently colored 4-connected ring regions appear on the grid; a ring
      may contain black interior space and smaller rings nested inside it.
    - A separate palette is present as a vertical top-to-bottom stack of K single pixels,
      where K equals the maximum nesting depth.

    Rule:
    - Identify the palette as the unique vertical sequence of singleton pixels.
    - Remove the palette and build the nesting structure of the remaining rings by
      considering how each ring separates background regions from the exterior inward.
    - Recolor every ring with the palette color at its depth, keeping the black spaces
      inside the rings unchanged, and turn the palette cells to black.

    Key insight:
    - The transformation is driven by topological containment depth, not by ring size,
      absolute position, or original color.
    """
    g = np.asarray(grid, dtype=int)
    palette, ring_components, ring_depths = _solve(g)
    palette_colors = [color for _r, _c, color in palette]

    out = g.copy()
    for r, c, _color in palette:
        out[r, c] = 0

    for comp, depth in zip(ring_components, ring_depths):
        coords = comp["coords"]
        out[coords[:, 0], coords[:, 1]] = palette_colors[depth - 1]
    return out


def _new_node() -> dict:
    return {"children": [], "color": None}


def _make_chain(length: int) -> tuple[dict, list[dict]]:
    root = _new_node()
    path = [root]
    cur = root
    for _ in range(length - 1):
        child = _new_node()
        cur["children"].append(child)
        cur = child
        path.append(cur)
    return root, path


def _iter_nodes(node: dict):
    yield node
    for child in node["children"]:
        yield from _iter_nodes(child)


def _forest_nodes(forest: list[dict]) -> list[dict]:
    nodes: list[dict] = []
    for root in forest:
        nodes.extend(list(_iter_nodes(root)))
    return nodes


def _make_forest(max_depth: int) -> list[dict]:
    root, path = _make_chain(max_depth)
    forest = [root]
    max_nodes = random.randint(max_depth, min(7, max_depth + 3))
    node_count = max_depth

    for idx, node in enumerate(path[:-1]):
        remaining = max_depth - idx - 1
        if node_count >= max_nodes:
            break
        if remaining >= 1 and random.random() < 0.55:
            extra_depth = random.randint(1, remaining)
            extra_root, _extra_path = _make_chain(extra_depth)
            node["children"].append(extra_root)
            node_count += extra_depth

    while node_count < max_nodes and len(forest) < 3 and random.random() < 0.55:
        extra_depth = random.randint(1, max_depth)
        extra_root, _extra_path = _make_chain(extra_depth)
        forest.append(extra_root)
        node_count += extra_depth

    return forest


def _assign_ring_colors(forest: list[dict]) -> None:
    nodes = _forest_nodes(forest)
    assert len(nodes) <= len(ARC_COLORS)
    colors = random.sample(ARC_COLORS, len(nodes))
    random.shuffle(colors)
    for node, color in zip(nodes, colors):
        node["color"] = color


def _layout_children(children: list[np.ndarray]) -> tuple[int, int, list[tuple[int, int]]]:
    if not children:
        return random.randint(3, 6), random.randint(3, 6), []

    if len(children) == 1:
        child = children[0]
        ch, cw = child.shape
        top = random.randint(1, 2)
        bottom = random.randint(1, 2)
        left = random.randint(1, 2)
        right = random.randint(1, 2)
        slack_h = random.randint(0, 2)
        slack_w = random.randint(0, 2)
        hole_h = ch + top + bottom + slack_h
        hole_w = cw + left + right + slack_w
        pos_r = top + random.randint(0, slack_h)
        pos_c = left + random.randint(0, slack_w)
        return hole_h, hole_w, [(pos_r, pos_c)]

    assert len(children) == 2
    first, second = children
    h1, w1 = first.shape
    h2, w2 = second.shape
    if random.random() < 0.5:
        top = random.randint(1, 2)
        bottom = random.randint(1, 2)
        left = random.randint(1, 2)
        right = random.randint(1, 2)
        gap = random.randint(1, 2)
        slack_h = random.randint(0, 2)
        hole_h = max(h1, h2) + top + bottom + slack_h
        hole_w = left + w1 + gap + w2 + right
        pos1 = (top + random.randint(0, slack_h), left)
        pos2 = (top + random.randint(0, slack_h), left + w1 + gap)
        return hole_h, hole_w, [pos1, pos2]

    top = random.randint(1, 2)
    bottom = random.randint(1, 2)
    left = random.randint(1, 2)
    right = random.randint(1, 2)
    gap = random.randint(1, 2)
    slack_w = random.randint(0, 2)
    hole_h = top + h1 + gap + h2 + bottom
    hole_w = max(w1, w2) + left + right + slack_w
    pos1 = (top, left + random.randint(0, slack_w))
    pos2 = (top + h1 + gap, left + random.randint(0, slack_w))
    return hole_h, hole_w, [pos1, pos2]


def _tab_count() -> int:
    pick = random.random()
    if pick < 0.45:
        return 0
    if pick < 0.85:
        return 1
    return 2


def _render_subtree(node: dict) -> np.ndarray:
    child_canvases = [_render_subtree(child) for child in node["children"]]
    hole_h, hole_w, placements = _layout_children(child_canvases)

    top_thick = random.randint(1, 2)
    bottom_thick = random.randint(1, 2)
    left_thick = random.randint(1, 2)
    right_thick = random.randint(1, 2)

    outer_h = hole_h + top_thick + bottom_thick
    outer_w = hole_w + left_thick + right_thick

    extents = {"top": 0, "bottom": 0, "left": 0, "right": 0}
    tabs: list[tuple[str, int, int]] = []
    for side in random.sample(["top", "bottom", "left", "right"], _tab_count()):
        extent = random.randint(1, 2)
        extents[side] = extent
        if side in ("top", "bottom"):
            span = random.randint(2, max(2, outer_w - 1))
        else:
            span = random.randint(2, max(2, outer_h - 1))
        start = random.randint(0, (outer_w if side in ("top", "bottom") else outer_h) - span)
        tabs.append((side, start, span))

    canvas = np.zeros(
        (outer_h + extents["top"] + extents["bottom"], outer_w + extents["left"] + extents["right"]),
        dtype=int,
    )
    main_r0 = extents["top"]
    main_c0 = extents["left"]
    color = int(node["color"])

    canvas[main_r0 : main_r0 + outer_h, main_c0 : main_c0 + outer_w] = color
    hole_r0 = main_r0 + top_thick
    hole_c0 = main_c0 + left_thick
    canvas[hole_r0 : hole_r0 + hole_h, hole_c0 : hole_c0 + hole_w] = 0

    for side, start, span in tabs:
        if side == "top":
            canvas[main_r0 - extents["top"] : main_r0, main_c0 + start : main_c0 + start + span] = color
        elif side == "bottom":
            canvas[
                main_r0 + outer_h : main_r0 + outer_h + extents["bottom"],
                main_c0 + start : main_c0 + start + span,
            ] = color
        elif side == "left":
            canvas[main_r0 + start : main_r0 + start + span, main_c0 - extents["left"] : main_c0] = color
        else:
            canvas[
                main_r0 + start : main_r0 + start + span,
                main_c0 + outer_w : main_c0 + outer_w + extents["right"],
            ] = color

    for child, (pr, pc) in zip(child_canvases, placements):
        ch, cw = child.shape
        view = canvas[hole_r0 + pr : hole_r0 + pr + ch, hole_c0 + pc : hole_c0 + pc + cw]
        mask = child != 0
        view[mask] = child[mask]

    return canvas


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            src_r0 = max(0, -dr)
            src_r1 = min(h, h - dr)
            src_c0 = max(0, -dc)
            src_c1 = min(w, w - dc)
            dst_r0 = max(0, dr)
            dst_r1 = min(h, h + dr)
            dst_c0 = max(0, dc)
            dst_c1 = min(w, w + dc)
            out[dst_r0:dst_r1, dst_c0:dst_c1] |= mask[src_r0:src_r1, src_c0:src_c1]
    return out


def _place_objects(objects: list[np.ndarray]) -> np.ndarray | None:
    objects = sorted(objects, key=lambda arr: int(np.count_nonzero(arr)), reverse=True)
    for _ in range(120):
        canvas = np.zeros((CANVAS_SIDE, CANVAS_SIDE), dtype=int)
        occupied = np.zeros((CANVAS_SIDE, CANVAS_SIDE), dtype=bool)
        placed_ok = True
        for obj in objects:
            h, w = obj.shape
            if h >= CANVAS_SIDE - 1 or w >= CANVAS_SIDE - 1:
                return None
            mask = obj != 0
            slots = [(r, c) for r in range(1, CANVAS_SIDE - h) for c in range(1, CANVAS_SIDE - w)]
            random.shuffle(slots)
            buffer = _dilate8(occupied)
            placed = False
            for r0, c0 in slots:
                if np.any(buffer[r0 : r0 + h, c0 : c0 + w] & mask):
                    continue
                patch = canvas[r0 : r0 + h, c0 : c0 + w]
                patch[mask] = obj[mask]
                occupied[r0 : r0 + h, c0 : c0 + w] |= mask
                placed = True
                break
            if not placed:
                placed_ok = False
                break
        if not placed_ok:
            continue

        coords = np.argwhere(occupied)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        top = random.randint(1, min(3, int(r0)))
        left = random.randint(1, min(3, int(c0)))
        bottom = random.randint(1, min(3, CANVAS_SIDE - 1 - int(r1)))
        right = random.randint(1, min(3, CANVAS_SIDE - 1 - int(c1)))
        return canvas[r0 - top : r1 + bottom + 1, c0 - left : c1 + right + 1]
    return None


def _palette_colors(ring_colors: set[int], depth: int) -> list[int]:
    unused = [color for color in ARC_COLORS if color not in ring_colors]
    if len(unused) >= depth:
        return random.sample(unused, depth)
    colors = list(ARC_COLORS)
    random.shuffle(colors)
    return colors[:depth]


def _gen_attempt() -> np.ndarray | None:
    max_depth = random.randint(1, 4)
    forest = _make_forest(max_depth)
    ring_nodes = _forest_nodes(forest)
    if len(ring_nodes) > len(ARC_COLORS):
        return None

    _assign_ring_colors(forest)
    roots = [_render_subtree(root) for root in forest]
    if any(max(arr.shape) > 22 for arr in roots):
        return None

    ring_colors = {int(node["color"]) for node in ring_nodes}
    palette = np.zeros((max_depth, 1), dtype=int)
    for r, color in enumerate(_palette_colors(ring_colors, max_depth)):
        palette[r, 0] = color

    grid = _place_objects(roots + [palette])
    if grid is None:
        return None
    if max(grid.shape) > 30:
        return None

    try:
        palette_cells, _ring_components, ring_depths = _solve(grid)
    except AssertionError:
        return None
    if len(palette_cells) != max_depth:
        return None
    if max(ring_depths) != max_depth:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-ring palette puzzle")
