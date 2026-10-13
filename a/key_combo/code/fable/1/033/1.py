from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 300


TEMPLATES = [
    {
        "name": "line",
        "vertices": [(0, 0), (0, 1)],
        "edges": [(0, 1)],
    },
    {
        "name": "elbow",
        "vertices": [(0, 0), (0, 1), (1, 1)],
        "edges": [(0, 1), (1, 2)],
    },
    {
        "name": "zigzag",
        "vertices": [(0, 0), (0, 1), (1, 1), (1, 2)],
        "edges": [(0, 1), (1, 2), (2, 3)],
    },
    {
        "name": "tee",
        "vertices": [(0, 1), (1, 0), (1, 1), (1, 2)],
        "edges": [(0, 2), (1, 2), (2, 3)],
    },
    {
        "name": "plus",
        "vertices": [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)],
        "edges": [(0, 2), (1, 2), (2, 3), (2, 4)],
    },
    {
        "name": "frame",
        "vertices": [(0, 0), (0, 1), (1, 0), (1, 1)],
        "edges": [(0, 1), (0, 2), (1, 3), (2, 3)],
    },
]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        seen[rr, cc] = True
        queue = deque([(rr, cc)])
        cells: list[tuple[int, int]] = []
        while queue:
            cr, cc = queue.popleft()
            cells.append((cr, cc))
            for dr, dc in DIRS4:
                nr = cr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        components.append(np.array(cells, dtype=int))
    return components


def _edge_segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        return [(r0, c) for c in range(lo, hi + 1)]
    if c0 == c1:
        lo, hi = sorted((r0, r1))
        return [(r, c0) for r in range(lo, hi + 1)]
    raise AssertionError("prototype edges must be axis-aligned")


def _draw_segment(grid: np.ndarray, a: tuple[int, int], b: tuple[int, int], color: int) -> None:
    for r, c in _edge_segment_cells(a, b):
        grid[r, c] = int(color)


def _neighbor_dirs(cell: tuple[int, int], cell_set: set[tuple[int, int]]) -> list[tuple[int, int]]:
    r, c = cell
    dirs = []
    for dr, dc in DIRS4:
        if (r + dr, c + dc) in cell_set:
            dirs.append((dr, dc))
    return dirs


def _is_vertex(cell: tuple[int, int], cell_set: set[tuple[int, int]]) -> bool:
    dirs = _neighbor_dirs(cell, cell_set)
    if len(dirs) != 2:
        return True
    (dr0, dc0), (dr1, dc1) = dirs
    return not (dr0 == -dr1 and dc0 == -dc1)


def _extract_graph(grid: np.ndarray, coords: np.ndarray) -> dict:
    colors = {int(grid[r, c]) for r, c in coords}
    assert len(colors) == 1, "instruction prototype must be monochromatic"
    color = colors.pop()

    cell_set = {tuple(map(int, rc)) for rc in coords}
    vertices = sorted(cell for cell in cell_set if _is_vertex(cell, cell_set))
    assert len(vertices) >= 2, "prototype must have at least two vertices"

    vertex_set = set(vertices)
    edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for vr, vc in vertices:
        for dr, dc in _neighbor_dirs((vr, vc), cell_set):
            cur = (vr + dr, vc + dc)
            while True:
                if cur in vertex_set:
                    if cur != (vr, vc):
                        a, b = sorted(((vr, vc), cur))
                        edges.add((a, b))
                    break
                nxt = (cur[0] + dr, cur[1] + dc)
                assert nxt in cell_set, "segment must continue straight until the next vertex"
                cur = nxt

    assert edges, "prototype must contain at least one segment"
    rows = sorted({r for r, _ in vertices})
    cols = sorted({c for _, c in vertices})
    return {
        "color": int(color),
        "vertices": vertices,
        "edges": sorted(edges),
        "rows": rows,
        "cols": cols,
    }


def _parse_container(grid: np.ndarray, coords: np.ndarray, bg: int) -> dict:
    r0, c0, r1, c1 = _bbox_coords(coords)
    crop = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(crop != bg), "rectangle container must fill its bounding box"

    border = np.concatenate(
        [
            crop[0, :],
            crop[-1, :],
            crop[1:-1, 0],
            crop[1:-1, -1],
        ]
    )
    border_colors = {int(v) for v in border}
    assert len(border_colors) == 1, "rectangle border must be monochromatic"
    rect_color = border_colors.pop()

    anchors = np.argwhere(crop != rect_color)
    assert anchors.size > 0, "each rectangle must contain anchor pixels"
    anchor_set = {tuple(map(int, rc)) for rc in anchors}
    for ar, ac in anchor_set:
        assert 0 < ar < crop.shape[0] - 1 and 0 < ac < crop.shape[1] - 1, "anchors must stay off the border"

    return {
        "bbox": (r0, c0, r1, c1),
        "rect_color": int(rect_color),
        "anchor_set": anchor_set,
        "anchor_rows": sorted({r for r, _ in anchor_set}),
        "anchor_cols": sorted({c for _, c in anchor_set}),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stretch the outside orthogonal prototype to each rectangle's anchor rows and columns, then stamp it inside.

    Input structure:
    - The background is black and the smallest non-background 4-connected component is a monochromatic prototype made of horizontal and vertical line segments.
    - Every other foreground component is a solid rectangle whose border uses one color and whose interior contains anchor pixels of another color.
    - The anchor pixels inside a rectangle occupy as many distinct rows and columns as the prototype's graph vertices do.

    Rule:
    - Extract the prototype's graph vertices: endpoints, corners, and intersections.
    - For each rectangle, sort the prototype vertex rows and the anchor rows, and map them one-to-one; do the same independently for columns.
    - Apply those row and column mappings to every prototype vertex, then redraw every prototype segment between its mapped endpoints using the prototype color, overwriting the rectangle interior and anchors while leaving the prototype and rectangle borders unchanged.

    Key insight:
    - The anchor pixels are stretched projections of the prototype's vertices, so the whole drawing is recovered by matching sorted vertex rows/columns to sorted anchor rows/columns independently.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)

    components = _components_mask(g != bg)
    assert len(components) >= 2, "expected one prototype and at least one rectangle"
    components.sort(key=lambda comp: int(comp.shape[0]))

    prototype = _extract_graph(g, components[0])
    out = g.copy()

    for coords in components[1:]:
        container = _parse_container(g, coords, bg)
        assert len(container["anchor_rows"]) == len(prototype["rows"])
        assert len(container["anchor_cols"]) == len(prototype["cols"])

        row_map = dict(zip(prototype["rows"], container["anchor_rows"]))
        col_map = dict(zip(prototype["cols"], container["anchor_cols"]))
        mapped_vertices = {
            vertex: (row_map[vertex[0]], col_map[vertex[1]])
            for vertex in prototype["vertices"]
        }
        assert set(mapped_vertices.values()) == container["anchor_set"], "anchors must be the mapped prototype vertices"

        r0, c0, _r1, _c1 = container["bbox"]
        for a, b in prototype["edges"]:
            pa = mapped_vertices[a]
            pb = mapped_vertices[b]
            _draw_segment(
                out,
                (r0 + pa[0], c0 + pa[1]),
                (r0 + pb[0], c0 + pb[1]),
                prototype["color"],
            )

    return out


def _template_variants(vertices: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    transforms = []
    for swap in (False, True):
        for sr in (-1, 1):
            for sc in (-1, 1):
                pts = []
                for r, c in vertices:
                    rr, cc = (c, r) if swap else (r, c)
                    pts.append((sr * rr, sc * cc))
                min_r = min(r for r, _ in pts)
                min_c = min(c for _, c in pts)
                norm = [(r - min_r, c - min_c) for r, c in pts]
                transforms.append(norm)

    unique: list[list[tuple[int, int]]] = []
    seen = set()
    for pts in transforms:
        key = tuple(pts)
        if key not in seen:
            seen.add(key)
            unique.append(pts)
    return unique


def _sample_axis_positions(count: int, gap_lo: int, gap_hi: int) -> list[int]:
    pos = [0]
    for _ in range(1, count):
        pos.append(pos[-1] + random.randint(gap_lo, gap_hi))
    return pos


def _build_prototype() -> dict:
    template = random.choice(TEMPLATES)
    abstract_vertices = random.choice(_template_variants(template["vertices"]))

    row_keys = sorted({r for r, _ in abstract_vertices})
    col_keys = sorted({c for _, c in abstract_vertices})
    row_values = _sample_axis_positions(len(row_keys), 1, 3)
    col_values = _sample_axis_positions(len(col_keys), 1, 3)
    row_map = dict(zip(row_keys, row_values))
    col_map = dict(zip(col_keys, col_values))

    vertices = [(row_map[r], col_map[c]) for r, c in abstract_vertices]
    edges = [(vertices[i], vertices[j]) for i, j in template["edges"]]
    return {
        "vertices": vertices,
        "edges": edges,
        "rows": sorted({r for r, _ in vertices}),
        "cols": sorted({c for _, c in vertices}),
    }


def _render_prototype(prototype: dict, color: int) -> np.ndarray:
    h = max(r for r, _ in prototype["vertices"]) + 1
    w = max(c for _, c in prototype["vertices"]) + 1
    sprite = np.full((h, w), BG, dtype=int)
    for a, b in prototype["edges"]:
        _draw_segment(sprite, a, b, color)
    return sprite


def _sample_container_patch(prototype: dict, rect_color: int, anchor_color: int) -> tuple[np.ndarray, dict]:
    row_count = len(prototype["rows"])
    col_count = len(prototype["cols"])

    h = random.randint(max(5, row_count + 2), min(14, row_count + 9))
    w = random.randint(max(5, col_count + 2), min(14, col_count + 9))
    anchor_rows = sorted(random.sample(range(1, h - 1), row_count))
    anchor_cols = sorted(random.sample(range(1, w - 1), col_count))

    row_map = dict(zip(prototype["rows"], anchor_rows))
    col_map = dict(zip(prototype["cols"], anchor_cols))
    patch = np.full((h, w), rect_color, dtype=int)
    anchors = set()
    for vr, vc in prototype["vertices"]:
        ar = row_map[vr]
        ac = col_map[vc]
        patch[ar, ac] = anchor_color
        anchors.add((ar, ac))

    return patch, {"anchor_rows": anchor_rows, "anchor_cols": anchor_cols, "anchors": anchors}


def _boxes_separate(a: tuple[int, int, int, int], b: tuple[int, int, int, int], padding: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 < br0 - padding or br1 < ar0 - padding or ac1 < bc0 - padding or bc1 < ac0 - padding


def _choose_location(
    shape: tuple[int, int],
    patch_shape: tuple[int, int],
    used_boxes: list[tuple[int, int, int, int]],
    *,
    border: int,
    padding: int,
) -> tuple[int, int] | None:
    h, w = patch_shape
    H, W = shape
    candidates = []
    for r0 in range(border, H - h - border + 1):
        for c0 in range(border, W - w - border + 1):
            box = (r0, c0, r0 + h - 1, c0 + w - 1)
            if all(_boxes_separate(box, other, padding) for other in used_boxes):
                candidates.append((r0, c0))
    if not candidates:
        return None
    return random.choice(candidates)


def _paste_non_bg(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    mask = patch != BG
    view = dst[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
    view[mask] = patch[mask]


def _expected_output(grid: np.ndarray, prototype: dict, proto_color: int, placements: list[dict]) -> np.ndarray:
    out = grid.copy()
    for info in placements:
        row_map = dict(zip(prototype["rows"], info["anchor_rows"]))
        col_map = dict(zip(prototype["cols"], info["anchor_cols"]))
        for a, b in prototype["edges"]:
            pa = (row_map[a[0]], col_map[a[1]])
            pb = (row_map[b[0]], col_map[b[1]])
            _draw_segment(
                out,
                (info["r0"] + pa[0], info["c0"] + pa[1]),
                (info["r0"] + pb[0], info["c0"] + pb[1]),
                proto_color,
            )
    return out


def _gen_attempt() -> np.ndarray | None:
    proto_color, rect_color, anchor_color = random.sample(range(1, 10), 3)
    prototype = _build_prototype()
    proto_patch = _render_prototype(prototype, proto_color)
    container_count = random.randint(2, 3)

    patches = [proto_patch]
    placements_meta: list[dict] = []
    for _ in range(container_count):
        patch, meta = _sample_container_patch(prototype, rect_color, anchor_color)
        patches.append(patch)
        placements_meta.append(meta)

    heights = [patch.shape[0] for patch in patches]
    widths = [patch.shape[1] for patch in patches]
    min_h = max(max(heights) + 2, sum(heights) // 2 + 5, 15)
    min_w = max(max(widths) + 2, sum(widths) // 2 + 5, 15)
    if min_h > 30 or min_w > 30:
        return None

    H = random.randint(min_h, 30)
    W = random.randint(min_w, 30)
    grid = np.full((H, W), BG, dtype=int)

    used_boxes: list[tuple[int, int, int, int]] = []
    proto_loc = _choose_location((H, W), proto_patch.shape, used_boxes, border=1, padding=1)
    if proto_loc is None:
        return None
    _paste_non_bg(grid, proto_patch, proto_loc[0], proto_loc[1])
    used_boxes.append((proto_loc[0], proto_loc[1], proto_loc[0] + proto_patch.shape[0] - 1, proto_loc[1] + proto_patch.shape[1] - 1))

    placed_containers: list[dict] = []
    for patch, meta in zip(patches[1:], placements_meta):
        loc = _choose_location((H, W), patch.shape, used_boxes, border=1, padding=1)
        if loc is None:
            return None
        _paste_non_bg(grid, patch, loc[0], loc[1])
        used_boxes.append((loc[0], loc[1], loc[0] + patch.shape[0] - 1, loc[1] + patch.shape[1] - 1))
        placed_containers.append(
            {
                "r0": loc[0],
                "c0": loc[1],
                "anchor_rows": meta["anchor_rows"],
                "anchor_cols": meta["anchor_cols"],
            }
        )

    expected = _expected_output(grid, prototype, proto_color, placed_containers)
    if np.array_equal(expected, grid):
        return None

    added_proto_pixels = int(np.count_nonzero((expected == proto_color) & (grid != proto_color)))
    if added_proto_pixels < len(prototype["vertices"]):
        return None

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid container-stretch input")
