from __future__ import annotations

from collections import deque
import random

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(int(r), int(c))])
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _ring_mask_from_bbox(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros(shape, dtype=bool)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _is_rect_ring(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    expected = 2 * h + 2 * w - 4
    if len(coords) != expected:
        return False
    local = np.zeros((h, w), dtype=bool)
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(np.array_equal(local, border))


def _draw_ring(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _extract_legend(grid: np.ndarray, bg: int) -> list[int]:
    coords = np.argwhere(grid != bg)
    assert len(coords) > 0, "expected at least one foreground cell"
    legend_col = int(coords[:, 1].min())
    legend_mask = np.zeros_like(grid, dtype=bool)
    legend_mask[:, legend_col] = grid[:, legend_col] != bg

    comps = _components(legend_mask)
    assert comps, "expected a left-side depth legend"
    colors: list[int] = []
    for comp in sorted(comps, key=lambda x: int(x[:, 0].min())):
        vals = set(map(int, grid[comp[:, 0], comp[:, 1]]))
        vals.discard(bg)
        assert len(vals) == 1, "each legend block must be monochrome"
        colors.append(vals.pop())
    assert len(colors) >= 2, "legend must encode at least depths 0 and 1"
    assert len(colors) == len(set(colors)), "legend colors must be unique by depth"
    return colors


def _find_rings(grid: np.ndarray, bg: int) -> list[dict]:
    rings: list[dict] = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        for comp in _components(grid == color):
            if not _is_rect_ring(comp):
                continue
            bbox = _bbox(comp)
            rings.append(
                {
                    "bbox": bbox,
                    "color": color,
                    "parent": None,
                    "depth": None,
                }
            )
    assert rings, "expected at least one rectangular ring"
    return rings


def _contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return or0 < ir0 and oc0 < ic0 and or1 > ir1 and oc1 > ic1


def _area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _annotate_hierarchy(nodes: list[dict]) -> None:
    for i, node in enumerate(nodes):
        parents = [
            j
            for j, other in enumerate(nodes)
            if i != j and _contains(other["bbox"], node["bbox"])
        ]
        if parents:
            node["parent"] = min(parents, key=lambda j: _area(nodes[j]["bbox"]))

    def assign_depth(idx: int) -> int:
        cached = nodes[idx]["depth"]
        if cached is not None:
            return int(cached)
        parent = nodes[idx]["parent"]
        depth = 0 if parent is None else assign_depth(int(parent)) + 1
        nodes[idx]["depth"] = depth
        return depth

    for i in range(len(nodes)):
        assign_depth(i)


def _project_to_root(nodes: list[dict], idx: int) -> tuple[int, int, int, int]:
    node = nodes[idx]
    cur = idx
    offset_r = 0
    offset_c = 0
    while nodes[cur]["parent"] is not None:
        parent_idx = int(nodes[cur]["parent"])
        parent = nodes[parent_idx]
        child_bbox = nodes[cur]["bbox"]
        parent_bbox = parent["bbox"]
        offset_r += child_bbox[0] - (parent_bbox[0] + 1)
        offset_c += child_bbox[1] - (parent_bbox[1] + 1)
        cur = parent_idx

    root_bbox = nodes[cur]["bbox"]
    h = node["bbox"][2] - node["bbox"][0]
    w = node["bbox"][3] - node["bbox"][1]
    r0 = root_bbox[0] + 1 + offset_r
    c0 = root_bbox[1] + 1 + offset_c
    return int(r0), int(c0), int(r0 + h), int(c0 + w)


def _paint_output(base: np.ndarray, nodes: list[dict], legend_colors: list[int]) -> np.ndarray:
    out = np.array(base, copy=True)
    max_depth = max(int(node["depth"]) for node in nodes)
    assert max_depth < len(legend_colors), "legend is shorter than the nesting depth"

    for node in nodes:
        _draw_ring(out, node["bbox"], int(legend_colors[int(node["depth"])]))

    deepest_color = int(legend_colors[max_depth])
    for idx, node in enumerate(nodes):
        if int(node["depth"]) != max_depth:
            continue
        projected_bbox = _project_to_root(nodes, idx)
        _draw_ring(out, projected_bbox, deepest_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build a nesting tree from rectangular rings, recolor each ring by depth using
    the left legend, then project the deepest rings into the root layer.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _most_common_color(g)

    legend_colors = _extract_legend(g, bg)
    nodes = _find_rings(g, bg)
    _annotate_hierarchy(nodes)
    return _paint_output(g, nodes, legend_colors)


def _leaf() -> tuple:
    return ()


def _node(*children) -> tuple:
    return tuple(children)


T_CHAIN2 = _node(_node(_leaf()))
T_CHAIN3 = _node(_node(_node(_leaf())))
T_BRANCH2 = _node(_node(_leaf()), _leaf())
T_FORK2 = _node(_node(_leaf()), _node(_leaf()))
T_BRANCH3 = _node(_node(_node(_leaf())), _leaf())
T_DEPTH1 = _node(_leaf())

SCENARIOS = [
    (T_CHAIN2,),
    (T_BRANCH2,),
    (T_FORK2,),
    (T_CHAIN3,),
    (T_BRANCH3,),
    (T_CHAIN2, _leaf()),
    (T_BRANCH2, _leaf()),
    (T_CHAIN2, T_DEPTH1),
]

LEAF_SIDE_MIN = 3
LEAF_SIDE_MAX = 5
MARGIN_MIN = 1
MARGIN_MAX = 2
SIBLING_GAP_MIN = 2
SIBLING_GAP_MAX = 4
GRID_MIN = 18
GRID_EXTRA = 6
ROOT_GAP = 2
GEN_RETRIES = 400


def _template_depth(template: tuple) -> int:
    if not template:
        return 0
    return 1 + max(_template_depth(child) for child in template)


def _count_nodes(template: tuple) -> int:
    return 1 + sum(_count_nodes(child) for child in template)


def _build_geometry(template: tuple) -> dict:
    children = [_build_geometry(child) for child in template]
    if not children:
        return {
            "h": random.randint(LEAF_SIDE_MIN, LEAF_SIDE_MAX),
            "w": random.randint(LEAF_SIDE_MIN, LEAF_SIDE_MAX),
            "children": [],
            "offsets": [],
        }

    if len(children) == 1:
        child = children[0]
        top = random.randint(MARGIN_MIN, MARGIN_MAX)
        bottom = random.randint(MARGIN_MIN, MARGIN_MAX)
        left = random.randint(MARGIN_MIN, MARGIN_MAX)
        right = random.randint(MARGIN_MIN, MARGIN_MAX)
        return {
            "h": child["h"] + top + bottom + 2,
            "w": child["w"] + left + right + 2,
            "children": children,
            "offsets": [(1 + top, 1 + left)],
        }

    assert len(children) == 2, "generator only supports binary branching"
    if random.random() < 0.5:
        children = [children[1], children[0]]

    first, second = children
    orientation = random.choice(("horizontal", "vertical"))
    if orientation == "horizontal":
        top = random.randint(MARGIN_MIN, MARGIN_MAX)
        bottom = random.randint(MARGIN_MIN, MARGIN_MAX)
        left = random.randint(MARGIN_MIN, MARGIN_MAX)
        right = random.randint(MARGIN_MIN, MARGIN_MAX)
        gap = random.randint(SIBLING_GAP_MIN, SIBLING_GAP_MAX)
        band_h = max(first["h"], second["h"]) + random.randint(0, 1)
        r1 = 1 + top + random.randint(0, band_h - first["h"])
        r2 = 1 + top + random.randint(0, band_h - second["h"])
        c1 = 1 + left
        c2 = 1 + left + first["w"] + gap
        return {
            "h": band_h + top + bottom + 2,
            "w": left + first["w"] + gap + second["w"] + right + 2,
            "children": children,
            "offsets": [(r1, c1), (r2, c2)],
        }

    top = random.randint(MARGIN_MIN, MARGIN_MAX)
    bottom = random.randint(MARGIN_MIN, MARGIN_MAX)
    left = random.randint(MARGIN_MIN, MARGIN_MAX)
    right = random.randint(MARGIN_MIN, MARGIN_MAX)
    gap = random.randint(SIBLING_GAP_MIN, SIBLING_GAP_MAX)
    band_w = max(first["w"], second["w"]) + random.randint(0, 1)
    c1 = 1 + left + random.randint(0, band_w - first["w"])
    c2 = 1 + left + random.randint(0, band_w - second["w"])
    r1 = 1 + top
    r2 = 1 + top + first["h"] + gap
    return {
        "h": top + first["h"] + gap + second["h"] + bottom + 2,
        "w": band_w + left + right + 2,
        "children": children,
        "offsets": [(r1, c1), (r2, c2)],
    }


def _instantiate_nodes(
    geom: dict,
    top: int,
    left: int,
    depth: int,
    parent: int | None,
    nodes: list[dict],
) -> int:
    idx = len(nodes)
    node = {
        "bbox": (top, left, top + geom["h"] - 1, left + geom["w"] - 1),
        "parent": parent,
        "depth": depth,
    }
    nodes.append(node)
    for child_geom, (dr, dc) in zip(geom["children"], geom["offsets"], strict=False):
        _instantiate_nodes(child_geom, top + dr, left + dc, depth + 1, idx, nodes)
    return idx


def _fits(occupied: np.ndarray, top: int, left: int, h: int, w: int, pad: int) -> bool:
    H, W = occupied.shape
    if top < 1 or left < 3 or top + h >= H or left + w >= W:
        return False
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(H, top + h + pad)
    c1 = min(W, left + w + pad)
    return not occupied[r0:r1, c0:c1].any()


def _occupy(occupied: np.ndarray, top: int, left: int, h: int, w: int, pad: int) -> None:
    H, W = occupied.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(H, top + h + pad)
    c1 = min(W, left + w + pad)
    occupied[r0:r1, c0:c1] = True


def _place_roots(root_geoms: list[dict], max_depth: int) -> tuple[int, int, list[tuple[int, int]]] | None:
    max_h = max(geom["h"] for geom in root_geoms)
    # Reserve enough height for the tallest possible legend: 2-row blocks with 1-row gaps.
    legend_h = 3 * (max_depth + 1) - 1
    total_w = sum(geom["w"] for geom in root_geoms)
    H = max(GRID_MIN, max_h + 4, legend_h + 4) + random.randint(0, GRID_EXTRA)
    W = max(16, total_w + 7 + ROOT_GAP * len(root_geoms)) + random.randint(0, GRID_EXTRA)

    occupied = np.zeros((H, W), dtype=bool)
    occupied[:, :3] = True
    positions: list[tuple[int, int] | None] = [None] * len(root_geoms)
    order = sorted(range(len(root_geoms)), key=lambda i: root_geoms[i]["h"] * root_geoms[i]["w"], reverse=True)

    for idx in order:
        geom = root_geoms[idx]
        candidates = [
            (r, c)
            for r in range(1, H - geom["h"])
            for c in range(3, W - geom["w"])
            if _fits(occupied, r, c, geom["h"], geom["w"], ROOT_GAP)
        ]
        if not candidates:
            return None
        top, left = random.choice(candidates)
        positions[idx] = (top, left)
        _occupy(occupied, top, left, geom["h"], geom["w"], ROOT_GAP)

    return H, W, [pos for pos in positions if pos is not None]


def _render_input(shape: tuple[int, int], legend_colors: list[int], nodes: list[dict], ring_colors: list[int], block_h: int, start_row: int) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for depth, color in enumerate(legend_colors):
        row0 = start_row + depth * (block_h + 1)
        grid[row0 : row0 + block_h, 0] = color
    for node, color in zip(nodes, ring_colors, strict=False):
        _draw_ring(grid, node["bbox"], int(color))
    return grid


def _projection_is_clean(nodes: list[dict], shape: tuple[int, int], max_depth: int) -> bool:
    all_rings = np.zeros(shape, dtype=bool)
    ring_masks = []
    for node in nodes:
        mask = _ring_mask_from_bbox(shape, node["bbox"])
        ring_masks.append(mask)
        all_rings |= mask

    projected_union = np.zeros(shape, dtype=bool)
    for idx, node in enumerate(nodes):
        if int(node["depth"]) != max_depth:
            continue
        bbox = _project_to_root(nodes, idx)
        root_idx = idx
        while nodes[root_idx]["parent"] is not None:
            root_idx = int(nodes[root_idx]["parent"])
        root_bbox = nodes[root_idx]["bbox"]
        if not (
            root_bbox[0] < bbox[0]
            and root_bbox[1] < bbox[1]
            and root_bbox[2] > bbox[2]
            and root_bbox[3] > bbox[3]
        ):
            return False
        mask = _ring_mask_from_bbox(shape, bbox)
        if projected_union.any() and np.any(projected_union & mask):
            return False
        other_rings = all_rings & ~ring_masks[idx]
        if np.any(mask & other_rings):
            return False
        projected_union |= mask

    return bool(np.any(projected_union & ~all_rings))


def _sample_scene() -> tuple[np.ndarray, np.ndarray] | None:
    templates = random.choice(SCENARIOS)
    max_depth = max(_template_depth(template) for template in templates)
    assert max_depth >= 2

    root_geoms = [_build_geometry(template) for template in templates]
    if max(geom["h"] for geom in root_geoms) > 26 or max(geom["w"] for geom in root_geoms) > 26:
        return None

    placement = _place_roots(root_geoms, max_depth)
    if placement is None:
        return None
    H, W, positions = placement

    nodes: list[dict] = []
    for geom, (top, left) in zip(root_geoms, positions, strict=False):
        _instantiate_nodes(geom, top, left, 0, None, nodes)

    if not _projection_is_clean(nodes, (H, W), max_depth):
        return None

    ring_count = len(nodes)
    palette = list(range(1, 10))
    random.shuffle(palette)
    ring_colors = palette[:ring_count]
    legend_colors = random.sample(range(1, 10), max_depth + 1)

    block_h = random.randint(1, 2)
    legend_h = block_h * len(legend_colors) + (len(legend_colors) - 1)
    max_start = H - legend_h - 1
    if max_start < 1:
        return None
    start_row = random.randint(1, max_start)

    inp = _render_input((H, W), legend_colors, nodes, ring_colors, block_h, start_row)
    expected = _paint_output(inp, nodes, legend_colors)

    if not np.any((inp == 0) & (expected != 0)):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _sample_scene()
        if result is not None:
            inp, _expected = result
            return inp
    raise RuntimeError(f"failed to generate a valid puzzle after {GEN_RETRIES} attempts")
