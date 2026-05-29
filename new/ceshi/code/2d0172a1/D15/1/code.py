# Variant of 2d0172a1
# Dimension: D15 — 符号编码：控制像素指示变换参数 (symbol_instructions)
# Change: 无符号编码——变换规则固定，不依赖任何控制像素或图例... → 角落图例指示输出样式

from __future__ import annotations

import random

import numpy as np

from arc_utils import find_components, get_bg, label_mask, paste


# Design analysis
# ---
# variant:
#   idea:
#     keep the original task's semantic core unchanged: recover the containment
#     tree of rough frames/blobs and canonicalize it. The D15 change is that the
#     canonical rendering style is no longer fixed; it is read from a small corner
#     legend drawn in the same foreground color as the main objects.
#   concrete_encoding:
#     legend_window: fixed 5x9 patch placed flush to one of the four corners
#     frame_example:
#       location: legend cols 0..4, rows 0..4
#       values:
#         thickness_1: a 5x5 hollow square border of width 1
#         thickness_2: a 5x5 hollow square border of width 2
#     dot_example:
#       location: legend cols 6..8, rows 1..3
#       values:
#         size_1: one foreground pixel at local coord (2,7)
#         size_2: one 2x2 filled block at local rows 1..2, cols 6..7
#     sampled_parameters:
#       frame_thickness: 1 or 2
#       dot_size: 1 or 2
#   behavior_trace:
#     original:
#       every blob becomes 1 pixel, every frame becomes a 1-pixel rectangle, and
#       the inside offset from border to child group is always 2 (= border 1 + gap 1).
#     variant:
#       the legend is read first, then the same containment/layout logic runs, but
#       blobs become dot_size x dot_size filled squares and frames become rectangles
#       with border width frame_thickness. The empty gap between a frame's inner edge
#       and its children stays 1 cell, so the child offset becomes frame_thickness + 1.
#     example:
#       if a frame contains two sibling blobs:
#       - old output: thin rectangle with two 1x1 dots.
#       - new output under legend(thickness=2, dot=2): thick rectangle with two 2x2
#         blocks placed in the same relative row/col order.
#   complexity_calibration:
#     - Only two binary parameters are introduced, keeping the rule learnable from
#       2-4 examples.
#     - The legend explicitly shows both parameters in separate subregions, so the
#       mapping is observable inside a single sample rather than requiring hidden
#       color-to-style lookup across samples.
#     - Output differences are visually strong: width-2 borders and 2x2 dots are
#       clearly different from the original style.
#   implicit_assumptions_review:
#     old_main_dependencies:
#       - assumes all foreground components belong to the scene; no longer true,
#         because the legend adds extra foreground components.
#       - assumes blob canonical mask is always 1x1; no longer true.
#       - assumes frame canonical border width is always 1 and child inset is always 2;
#         no longer true.
#       - assumes outside blobs occupy a 3x3 slot around a single pixel; this must be
#         generalized to a (dot_size+2) slot.
#     fixes:
#       - detect and erase the unique corner legend before component analysis.
#       - parameterize canonical masks by a _Style object.
#       - keep sibling spacing logic unchanged, because it already works for arbitrary
#         mask sizes once anchors/masks are updated consistently.
#   placement_legality:
#     - The generator reserves a full top or bottom band for the legend, so main
#       objects never touch the legend and the solver sees a unique exact-match patch.
#     - Main objects still use the original non-overlap strategy inside the scene.
#     - The legend uses the same fg color but is disconnected from the scene by at
#       least one background row.
#   visual_observability:
#     - Humans can read the legend directly: the left sample is the target frame
#       border style; the right sample is the target blob style.
#     - The main drawing remains visually identical in semantics to the base task,
#       so the only new concept to infer is "copy the style shown in the corner".
#   cleanliness:
#     - The legend is confined to a dedicated corner patch with no overdraw.
#     - No color-id ordering or hidden numeric dependence is introduced.


LEGEND_H = 5
LEGEND_W = 9
LEGEND_DOT_COL0 = 6
LEGEND_CLEARANCE = 1
LEGEND_CORNERS = ("top_left", "top_right", "bottom_left", "bottom_right")


class _Style:
    def __init__(self, frame_thickness: int, dot_size: int):
        self.frame_thickness = int(frame_thickness)
        self.dot_size = int(dot_size)


class _Node:
    def __init__(self, coords: np.ndarray, bbox: tuple[int, int, int, int], center: np.ndarray):
        self.coords = coords
        self.bbox = bbox
        self.center = center
        self.is_frame = False
        self.parent: _Node | None = None
        self.children: list[_Node] = []
        self.mask: np.ndarray | None = None
        self.anchor: tuple[int, int] | None = None
        self.axis_map: dict[str, list[tuple[float, int]]] = {"row": [], "col": []}


def _legend_patch(bg: int, fg: int, style: _Style) -> np.ndarray:
    patch = np.full((LEGEND_H, LEGEND_W), int(bg), dtype=int)
    t = style.frame_thickness
    patch[:t, :5] = int(fg)
    patch[-t:, :5] = int(fg)
    patch[:, :t] = int(fg)
    patch[:, 5 - t : 5] = int(fg)
    if style.dot_size == 1:
        patch[2, 7] = int(fg)
    else:
        patch[1:3, LEGEND_DOT_COL0 : LEGEND_DOT_COL0 + 2] = int(fg)
    return patch


def _corner_bbox(shape: tuple[int, int], corner: str) -> tuple[int, int, int, int]:
    h, w = map(int, shape)
    if corner == "top_left":
        return 0, 0, LEGEND_H, LEGEND_W
    if corner == "top_right":
        return 0, w - LEGEND_W, LEGEND_H, w
    if corner == "bottom_left":
        return h - LEGEND_H, 0, h, LEGEND_W
    return h - LEGEND_H, w - LEGEND_W, h, w


def _find_legend(grid: np.ndarray, bg: int, fg: int) -> tuple[_Style, tuple[int, int, int, int]]:
    matches: list[tuple[_Style, tuple[int, int, int, int]]] = []
    for corner in LEGEND_CORNERS:
        r0, c0, r1, c1 = _corner_bbox(grid.shape, corner)
        patch = grid[r0:r1, c0:c1]
        for frame_thickness in (1, 2):
            for dot_size in (1, 2):
                style = _Style(frame_thickness, dot_size)
                if np.array_equal(patch, _legend_patch(bg, fg, style)):
                    matches.append((style, (r0, c0, r1, c1)))
    assert len(matches) == 1
    return matches[0]


def _component_holes(shape: tuple[int, int], coords: np.ndarray) -> list[np.ndarray]:
    mask = np.zeros(shape, dtype=bool)
    mask[tuple(coords.T)] = True
    lab, n = label_mask(~mask, connectivity=4)
    boundary = set(lab[0, :]) | set(lab[-1, :]) | set(lab[:, 0]) | set(lab[:, -1])
    return [(lab == k) for k in range(1, n + 1) if k not in boundary]


def _intervals_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(a[0], b[0]) <= min(a[1], b[1])


def _build_layout(node: _Node, style: _Style) -> None:
    if node.mask is not None:
        return

    if not node.is_frame:
        node.mask = np.ones((style.dot_size, style.dot_size), dtype=bool)
        anchor = (style.dot_size // 2, style.dot_size // 2)
        node.anchor = anchor
        node.axis_map = {
            "row": [(float(node.center[0]), int(anchor[0]))],
            "col": [(float(node.center[1]), int(anchor[1]))],
        }
        return

    assert node.children
    child_mask, _placements, axis_map = _layout_group(node.children, style)
    h, w = child_mask.shape
    pad = style.frame_thickness + 1
    t = style.frame_thickness
    out = np.zeros((h + 2 * pad, w + 2 * pad), dtype=bool)
    out[:t, :] = True
    out[-t:, :] = True
    out[:, :t] = True
    out[:, -t:] = True
    out[pad : pad + h, pad : pad + w] |= child_mask

    node.mask = out
    node.anchor = (out.shape[0] // 2, out.shape[1] // 2)
    node.axis_map = {
        "row": [(orig, pos + pad) for orig, pos in axis_map["row"]],
        "col": [(orig, pos + pad) for orig, pos in axis_map["col"]],
    }


def _axis_coordinates(nodes: list[_Node], axis: int) -> dict[_Node, int]:
    items = []
    for node in nodes:
        assert node.mask is not None and node.anchor is not None
        r0, c0, r1, c1 = node.bbox
        interval = (r0, r1) if axis == 0 else (c0, c1)
        mid = (interval[0] + interval[1]) / 2
        items.append((node, interval, mid, node.anchor[axis], node.mask.shape[axis]))

    bands: list[dict] = []
    for item in sorted(items, key=lambda x: x[2]):
        interval = item[1]
        for band in bands:
            if any(_intervals_overlap(interval, other[1]) for other in band["items"]):
                band["items"].append(item)
                band["interval"] = (
                    min(band["interval"][0], interval[0]),
                    max(band["interval"][1], interval[1]),
                )
                break
        else:
            bands.append({"items": [item], "interval": interval})

    coord_by_node: dict[_Node, int] = {}
    coord = 0
    for i, band in enumerate(bands):
        if i:
            prev = bands[i - 1]
            prev_down = max(item[4] - 1 - item[3] for item in prev["items"])
            curr_up = max(item[3] for item in band["items"])
            coord += prev_down + 2 + curr_up
        for item in band["items"]:
            coord_by_node[item[0]] = int(coord)

    return coord_by_node


def _layout_group(
    nodes: list[_Node], style: _Style
) -> tuple[np.ndarray, dict[_Node, tuple[int, int]], dict[str, list[tuple[float, int]]]]:
    assert nodes
    for node in nodes:
        _build_layout(node, style)

    row_coord = _axis_coordinates(nodes, axis=0)
    col_coord = _axis_coordinates(nodes, axis=1)

    min_r = min(row_coord[n] - n.anchor[0] for n in nodes if n.anchor is not None)
    min_c = min(col_coord[n] - n.anchor[1] for n in nodes if n.anchor is not None)
    max_r = max(
        row_coord[n] - n.anchor[0] - min_r + n.mask.shape[0]
        for n in nodes
        if n.anchor is not None and n.mask is not None
    )
    max_c = max(
        col_coord[n] - n.anchor[1] - min_c + n.mask.shape[1]
        for n in nodes
        if n.anchor is not None and n.mask is not None
    )

    out = np.zeros((max_r, max_c), dtype=bool)
    placements: dict[_Node, tuple[int, int]] = {}
    axis_map = {"row": [], "col": []}
    for node in nodes:
        assert node.mask is not None and node.anchor is not None
        r = row_coord[node] - node.anchor[0] - min_r
        c = col_coord[node] - node.anchor[1] - min_c
        placements[node] = (int(r), int(c))
        out[r : r + node.mask.shape[0], c : c + node.mask.shape[1]] |= node.mask
        axis_map["row"].append((float(node.center[0]), int(r + node.anchor[0])))
        axis_map["col"].append((float(node.center[1]), int(c + node.anchor[1])))

    return out, placements, axis_map


def _outside_side(dot: _Node, frame: _Node) -> str:
    r0, c0, r1, c1 = dot.bbox
    fr0, fc0, fr1, fc1 = frame.bbox
    if c0 > fc1:
        return "right"
    if c1 < fc0:
        return "left"
    if r0 > fr1:
        return "bottom"
    if r1 < fr0:
        return "top"

    dr = float(dot.center[0] - (fr0 + fr1) / 2)
    dc = float(dot.center[1] - (fc0 + fc1) / 2)
    if abs(dc) >= abs(dr):
        return "right" if dc > 0 else "left"
    return "bottom" if dr > 0 else "top"


def _map_frame_axis(frame: _Node, axis: int, value: float) -> int:
    assert frame.anchor is not None
    key = "row" if axis == 0 else "col"
    candidates = frame.axis_map[key]
    if not candidates:
        return int(frame.anchor[axis])
    return int(min(candidates, key=lambda item: abs(item[0] - value))[1])


def _layout_root(nodes: list[_Node], style: _Style) -> np.ndarray:
    assert nodes
    for node in nodes:
        _build_layout(node, style)

    frames = [node for node in nodes if node.is_frame]
    dots = [node for node in nodes if not node.is_frame]
    if len(frames) == 1 and dots:
        frame = frames[0]
        assert frame.mask is not None
        placements: dict[_Node, tuple[int, int]] = {frame: (0, 0)}
        extents = [(0, 0, frame.mask.shape[0], frame.mask.shape[1])]

        for dot in dots:
            assert dot.anchor is not None and dot.mask is not None
            side = _outside_side(dot, frame)
            if side in {"right", "left"}:
                r = _map_frame_axis(frame, 0, float(dot.center[0])) - dot.anchor[0]
                c = frame.mask.shape[1] + 1 if side == "right" else -style.dot_size - 1
            else:
                r = frame.mask.shape[0] + 1 if side == "bottom" else -style.dot_size - 1
                c = _map_frame_axis(frame, 1, float(dot.center[1])) - dot.anchor[1]
            placements[dot] = (int(r), int(c))
            extents.append((r - 1, c - 1, r + dot.mask.shape[0] + 1, c + dot.mask.shape[1] + 1))

        min_r = min(e[0] for e in extents)
        min_c = min(e[1] for e in extents)
        max_r = max(e[2] for e in extents)
        max_c = max(e[3] for e in extents)
        out = np.zeros((max_r - min_r, max_c - min_c), dtype=bool)
        for node, (r, c) in placements.items():
            rr = r - min_r
            cc = c - min_c
            if node is frame:
                out[rr : rr + frame.mask.shape[0], cc : cc + frame.mask.shape[1]] |= frame.mask
            else:
                assert node.mask is not None
                out[rr : rr + node.mask.shape[0], cc : cc + node.mask.shape[1]] |= node.mask
        return out

    mask, _placements, _axis_map = _layout_group(nodes, style)
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Canonicalize a rough nested frame-and-blob scene using the style shown in a corner legend.

    Input structure:
    - The grid still uses one background color and one foreground color for the main scene.
    - A dedicated corner patch contains a tiny legend: a sample frame on the left and a sample blob on the right.
    - Outside that legend, foreground 4-connected objects are rough closed frames or filled blobs with the same containment semantics as the base task.

    Rule:
    - Read the legend first to recover the target rectangular border thickness and canonical blob size.
    - Ignore the legend when detecting the foreground containment tree of the main scene.
    - Replace each blob by a filled square of the legend-indicated size, and replace each frame by a rectangle whose border width matches the legend while preserving one background cell of inner clearance.
    - Keep the original relative sibling row/column ordering and the outside-blob side relation exactly as in the base task.

    Key insight: The geometry logic is unchanged; the only new information is a local visual style key in the corner, and that key must be applied uniformly to the whole canonical redraw.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    fg_colors = sorted(set(map(int, g.flatten())) - {bg})
    assert len(fg_colors) == 1
    fg = int(fg_colors[0])

    style, (lr0, lc0, lr1, lc1) = _find_legend(g, bg, fg)
    scene = g.copy()
    scene[lr0:lr1, lc0:lc1] = bg

    comps = find_components(scene, bg=bg, connectivity=4, monochromatic=True, colors=[fg], return_format="dict")
    assert comps

    nodes: list[_Node] = []
    holes: list[list[np.ndarray]] = []
    for comp in comps:
        coords = np.asarray(comp["coords"], dtype=int)
        node = _Node(
            coords=coords,
            bbox=tuple(map(int, comp["bbox"])),
            center=coords.mean(axis=0),
        )
        node_holes = _component_holes(scene.shape, coords)
        node.is_frame = bool(node_holes)
        nodes.append(node)
        holes.append(node_holes)

    assert any(node.is_frame for node in nodes)
    for node_i, node in enumerate(nodes):
        containers = []
        for frame_i, frame in enumerate(nodes):
            if node_i == frame_i or not frame.is_frame:
                continue
            if any(np.all(hole[tuple(node.coords.T)]) for hole in holes[frame_i]):
                containers.append(frame)
        if containers:
            node.parent = min(containers, key=lambda item: int(item.coords.shape[0]))
            node.parent.children.append(node)

    roots = [node for node in nodes if node.parent is None]
    mask = _layout_root(roots, style)
    out = np.full(mask.shape, bg, dtype=int)
    out[mask] = fg
    return out


# --------------- generator (o5x) ---------------

# Grid / canvas bounds
GRID_MIN = 16
GRID_MAX = 25
MAX_EXTRA_PAD = 12
MAX_PAD_PER_SIDE = 6

# Frame geometry
FRAME_H_RANGE = (7, 16)
FRAME_W_RANGE = (7, 24)
FRAME_CORNER_MAX = 5
FRAME_MIN_EDGE = 2
FRAME_TURN_RATIO = (0.34, 0.43)
FRAME_RETRIES = 60

# Blob geometry
BLOB_SIZE_RANGE = (4, 11)
BLOB_BBOX_RANGE = (2, 5)

# Layout / arrangement
ARRANGE_GAP_COMPACT = (2, 3)
ARRANGE_GAP_NORMAL = (2, 5)
ARRANGE_EXTRA_SPAN = (0, 4)
FRAME_MARGIN_RANGE = (3, 5)
COMPACT_MARGIN = 3

# Outside blob placement
OUTSIDE_GAP_RANGE = (2, 5)

# Tree structure probabilities
DEPTH1_PROB = 0.35
OUTSIDE_BLOB_PROB = 0.35
DEPTH1_INSIDE_RANGE = (1, 3)
DEPTH2_INNER_BLOBS = (1, 2)
DEPTH2_OUTER_BLOBS = (0, 2)
MAX_TOTAL_CHILDREN = 4

# Step-mix alternation bias
STEP_MIX_ALT_PROB = 0.7

# Generation retry budget
GEN_RETRIES = 400


class _Obj:
    def __init__(self, kind, children=None, outside_side=None):
        self.kind = kind
        self.children = list(children or [])
        self.mask = None
        self.rel = (0, 0)
        self.pos = (0, 0)
        self.outside_side = outside_side


def _loop_ok(mask):
    _, n = label_mask(mask, connectivity=4)
    if n != 1:
        return False
    deg = np.zeros_like(mask, dtype=int)
    deg[1:, :] += mask[:-1, :]
    deg[:-1, :] += mask[1:, :]
    deg[:, 1:] += mask[:, :-1]
    deg[:, :-1] += mask[:, 1:]
    if not np.all(deg[mask] == 2):
        return False
    inv_lab, inv_n = label_mask(~mask, connectivity=4)
    boundary = set(inv_lab[0, :]) | set(inv_lab[-1, :]) | set(inv_lab[:, 0]) | set(inv_lab[:, -1])
    return sum(1 for k in range(1, inv_n + 1) if k not in boundary) == 1


def _step_mix(n_a, step_a, n_b, step_b):
    out = []
    last = None
    while n_a or n_b:
        if n_a and n_b:
            if last is not None and random.random() < STEP_MIX_ALT_PROB:
                pick = "b" if last == "a" else "a"
            else:
                pick = "a" if random.random() < n_a / (n_a + n_b) else "b"
        elif n_a:
            pick = "a"
        else:
            pick = "b"
        if pick == "a":
            out.append(step_a)
            n_a -= 1
        else:
            out.append(step_b)
            n_b -= 1
        last = pick
    return out


def _add_line(path, end):
    r, c = path[-1]
    er, ec = end
    dr = 0 if er == r else (1 if er > r else -1)
    dc = 0 if ec == c else (1 if ec > c else -1)
    while (r, c) != (er, ec):
        r += dr
        c += dc
        path.append((r, c))


def _add_steps(path, steps):
    r, c = path[-1]
    for dr, dc in steps:
        r += dr
        c += dc
        path.append((r, c))


def _turn_ratio(path):
    dirs = []
    cyc = path + [path[0]]
    for (r0, c0), (r1, c1) in zip(cyc, cyc[1:]):
        dirs.append((r1 - r0, c1 - c0))
    turns = sum(1 for a, b in zip(dirs, dirs[1:] + [dirs[0]]) if a != b)
    return turns / len(path)


def _frame_mask(h, w):
    max_dx = min(FRAME_CORNER_MAX, max(1, (w - 5) // 2))
    max_dy = min(FRAME_CORNER_MAX, max(1, (h - 5) // 2))
    lo, hi = FRAME_TURN_RATIO
    for _ in range(FRAME_RETRIES):
        tl_dx = random.randint(1, max_dx)
        tr_dx = random.randint(1, max_dx)
        br_dx = random.randint(1, max_dx)
        bl_dx = random.randint(1, max_dx)
        tl_dy = random.randint(1, max_dy)
        tr_dy = random.randint(1, max_dy)
        br_dy = random.randint(1, max_dy)
        bl_dy = random.randint(1, max_dy)

        p0 = (0, tl_dx)
        p1 = (0, w - 1 - tr_dx)
        p2 = (tr_dy, w - 1)
        p3 = (h - 1 - br_dy, w - 1)
        p4 = (h - 1, w - 1 - br_dx)
        p5 = (h - 1, bl_dx)
        p6 = (h - 1 - bl_dy, 0)
        p7 = (tl_dy, 0)
        if not (
            p1[1] - p0[1] >= FRAME_MIN_EDGE
            and p3[0] - p2[0] >= FRAME_MIN_EDGE
            and p4[1] - p5[1] >= FRAME_MIN_EDGE
            and p6[0] - p7[0] >= FRAME_MIN_EDGE
        ):
            continue

        path = [p0]
        _add_line(path, p1)
        _add_steps(path, _step_mix(tr_dy, (1, 0), tr_dx, (0, 1)))
        _add_line(path, p3)
        _add_steps(path, _step_mix(br_dy, (1, 0), br_dx, (0, -1)))
        _add_line(path, p5)
        _add_steps(path, _step_mix(bl_dy, (-1, 0), bl_dx, (0, -1)))
        _add_line(path, p7)
        _add_steps(path, _step_mix(tl_dy, (-1, 0), tl_dx, (0, 1)))
        if path[-1] == path[0]:
            path.pop()

        coords = np.array(path, dtype=int)
        mask = np.zeros((h, w), dtype=bool)
        mask[coords[:, 0], coords[:, 1]] = True
        ratio = _turn_ratio(path)
        if lo <= ratio <= hi and _loop_ok(mask):
            return mask
    return None


def _blob_mask():
    size = random.randint(*BLOB_SIZE_RANGE)
    feasible = [
        (h, w)
        for h in range(*BLOB_BBOX_RANGE)
        for w in range(*BLOB_BBOX_RANGE)
        if h * w >= size and h + w - 1 <= size
    ]
    h, w = random.choice(feasible)
    rr = random.randrange(h)
    cc = random.randrange(w)
    cells = {(r, cc) for r in range(h)} | {(rr, c) for c in range(w)}
    while len(cells) < size:
        frontier = []
        for r, c in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                p = (r + dr, c + dc)
                if 0 <= p[0] < h and 0 <= p[1] < w and p not in cells:
                    frontier.append(p)
        cells.add(random.choice(frontier))
    mask = np.zeros((h, w), dtype=bool)
    pts = np.array(list(cells), dtype=int)
    mask[pts[:, 0], pts[:, 1]] = True
    return mask


def _arrange(children, compact=False):
    random.shuffle(children)
    dims = [child.mask.shape for child in children]
    k = len(children)
    if k == 1:
        return {children[0]: (0, 0)}, dims[0][0], dims[0][1]

    mode = random.choice(("vertical", "horizontal", "diagonal"))
    gap_lo, gap_hi = ARRANGE_GAP_COMPACT if compact else ARRANGE_GAP_NORMAL
    gaps = [random.randint(gap_lo, gap_hi) for _ in range(k - 1)]
    placements = {}

    if mode == "vertical":
        group_h = sum(h for h, _ in dims) + sum(gaps)
        group_w = max(w for _, w in dims) + random.randint(*ARRANGE_EXTRA_SPAN)
        r = 0
        for child, (h, w), gap in zip(children, dims, gaps + [0]):
            placements[child] = (r, random.randint(0, group_w - w))
            r += h + gap
        return placements, group_h, group_w

    if mode == "horizontal":
        group_h = max(h for h, _ in dims) + random.randint(*ARRANGE_EXTRA_SPAN)
        group_w = sum(w for _, w in dims) + sum(gaps)
        c = 0
        for child, (h, w), gap in zip(children, dims, gaps + [0]):
            placements[child] = (random.randint(0, group_h - h), c)
            c += w + gap
        return placements, group_h, group_w

    group_h = sum(h for h, _ in dims) + sum(gaps)
    group_w = sum(w for _, w in dims) + sum(gaps)
    r = c = 0
    for child, (h, w), gap in zip(children, dims, gaps + [0]):
        placements[child] = (r, c)
        r += h + gap
        c += w + gap
    return placements, group_h, group_w


def _prepare(node, compact=False):
    if node.kind == "blob":
        node.mask = _blob_mask()
        return True

    for child in node.children:
        if not _prepare(child, compact=child.kind == "frame"):
            return False

    has_frame_child = any(child.kind == "frame" for child in node.children)
    tight = compact or has_frame_child
    placements, group_h, group_w = _arrange(node.children, compact=tight)
    margin = COMPACT_MARGIN if tight else random.randint(*FRAME_MARGIN_RANGE)
    min_h = max(FRAME_H_RANGE[0], group_h + 2 * margin)
    min_w = max(FRAME_W_RANGE[0], group_w + 2 * margin)
    if min_h > FRAME_H_RANGE[1] or min_w > FRAME_W_RANGE[1]:
        return False
    h_hi = min(FRAME_H_RANGE[1], min_h + 2) if compact else FRAME_H_RANGE[1]
    w_hi = min(FRAME_W_RANGE[1], min_w + 3) if compact else FRAME_W_RANGE[1]
    h = random.randint(min_h, h_hi)
    w = random.randint(min_w, w_hi)
    top = margin + random.randint(0, h - 2 * margin - group_h)
    left = margin + random.randint(0, w - 2 * margin - group_w)
    for child, (r, c) in placements.items():
        child.rel = (top + r, left + c)
    node.mask = _frame_mask(h, w)
    return node.mask is not None


def _walk(node, base=(0, 0)):
    br, bc = base
    rr, cc = node.rel
    node.pos = (br + rr, bc + cc)
    out = [node]
    for child in node.children:
        out.extend(_walk(child, node.pos))
    return out


def _place_outside(root, outside):
    assert root.mask is not None and outside.mask is not None
    h, w = root.mask.shape
    bh, bw = outside.mask.shape
    gap = random.randint(*OUTSIDE_GAP_RANGE)
    side = outside.outside_side
    if side == "right":
        outside.rel = (random.randint(1, h - bh - 1), w + gap)
    elif side == "left":
        outside.rel = (random.randint(1, h - bh - 1), -gap - bw)
    elif side == "bottom":
        outside.rel = (h + gap, random.randint(1, w - bw - 1))
    else:
        outside.rel = (-gap - bh, random.randint(1, w - bw - 1))


def _node_from_obj(obj):
    assert obj.mask is not None
    r0, c0 = obj.pos
    coords = np.argwhere(obj.mask) + np.array([r0, c0])
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    node = _Node(
        coords=coords.astype(int),
        bbox=(int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])),
        center=coords.mean(axis=0),
    )
    node.is_frame = obj.kind == "frame"
    return node


def _expected_output(root, outside, bg, fg, style):
    def add(obj):
        node = _node_from_obj(obj)
        for child in obj.children:
            child_node = add(child)
            child_node.parent = node
            node.children.append(child_node)
        return node

    roots = [add(root)]
    if outside is not None:
        roots.append(add(outside))
    mask = _layout_root(roots, style)
    out = np.full(mask.shape, bg, dtype=int)
    out[mask] = fg
    return out


def _make_tree():
    root = _Obj("frame")
    outside = None
    sides = ("left", "right", "top", "bottom")
    if random.random() < DEPTH1_PROB:
        n_inside = random.randint(*DEPTH1_INSIDE_RANGE)
        root.children = [_Obj("blob") for _ in range(n_inside)]
        if n_inside < MAX_TOTAL_CHILDREN and random.random() < OUTSIDE_BLOB_PROB:
            outside = _Obj("blob", outside_side=random.choice(sides))
    else:
        inner = _Obj("frame")
        inner.children = [_Obj("blob") for _ in range(random.randint(*DEPTH2_INNER_BLOBS))]
        outer_blobs = [_Obj("blob") for _ in range(random.randint(*DEPTH2_OUTER_BLOBS))]
        root.children = [inner] + outer_blobs
        if len(inner.children) + len(outer_blobs) < MAX_TOTAL_CHILDREN and random.random() < OUTSIDE_BLOB_PROB:
            outside = _Obj("blob", outside_side=random.choice(sides))
    return root, outside


def _style_sample():
    return _Style(frame_thickness=random.choice((1, 2)), dot_size=random.choice((1, 2)))


def _draw(objects, bg, fg, style, corner):
    min_r = min(obj.pos[0] for obj in objects)
    min_c = min(obj.pos[1] for obj in objects)
    max_r = max(obj.pos[0] + obj.mask.shape[0] for obj in objects)
    max_c = max(obj.pos[1] + obj.mask.shape[1] for obj in objects)
    draw_h = max_r - min_r
    draw_w = max_c - min_c

    reserved_h = LEGEND_H + LEGEND_CLEARANCE
    min_h = max(GRID_MIN, draw_h + reserved_h)
    min_w = max(GRID_MIN, max(draw_w, LEGEND_W))
    max_h = min(GRID_MAX, draw_h + reserved_h + MAX_EXTRA_PAD)
    max_w = min(GRID_MAX, draw_w + MAX_EXTRA_PAD)
    if min_h > max_h or min_w > max_w:
        return None

    H = random.randint(min_h, max_h)
    W = random.randint(min_w, max_w)

    if corner.startswith("top"):
        row_lo = reserved_h
        row_hi = H - draw_h
    else:
        row_lo = 0
        row_hi = H - reserved_h - draw_h
    if row_lo > row_hi:
        return None
    col_lo = 0
    col_hi = W - draw_w
    if col_lo > col_hi:
        return None

    start_r = random.randint(row_lo, row_hi)
    start_c = random.randint(col_lo, col_hi)
    shift_r = start_r - min_r
    shift_c = start_c - min_c

    grid = np.full((H, W), bg, dtype=int)
    for obj in objects:
        assert obj.mask is not None
        r = obj.pos[0] + shift_r
        c = obj.pos[1] + shift_c
        obj.pos = (r, c)
        patch = np.full(obj.mask.shape, bg, dtype=int)
        patch[obj.mask] = fg
        paste(grid, patch, r, c, transparent=bg)

    legend = _legend_patch(bg, fg, style)
    lr0, lc0, lr1, lc1 = _corner_bbox(grid.shape, corner)
    if np.any(grid[lr0:lr1, lc0:lc1] != bg):
        return None
    grid[lr0:lr1, lc0:lc1] = legend

    if np.count_nonzero(grid == bg) <= np.count_nonzero(grid == fg):
        return None
    return grid


def _gen_attempt():
    bg, fg = random.sample(range(10), 2)
    style = _style_sample()
    corner = random.choice(LEGEND_CORNERS)
    root, outside = _make_tree()
    if not _prepare(root):
        return None
    if outside is not None:
        if not _prepare(outside):
            return None
        _place_outside(root, outside)

    objects = _walk(root)
    if outside is not None:
        outside.pos = outside.rel
        objects.append(outside)

    grid = _draw(objects, bg, fg, style, corner)
    if grid is None:
        return None
    expected = _expected_output(root, outside, bg, fg, style)
    try:
        got = main(grid)
    except Exception:
        return None
    if not np.array_equal(got, expected):
        return None
    return grid


def generate_input():
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 2d0172a1 D15 input")
