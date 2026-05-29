# Variant of 2d0172a1
# Dimension: D3 — 源对象：框架的几何属性 (src)
# Change: 框架为4-连通闭合简单曲线，每个像素恰好有2个4-邻域前景邻居，外接矩形7x7到20x24，带有不规则弯折（turn_ratio 0.34-0.43）→ 框架为圆形/椭圆形闭合曲线

from __future__ import annotations

import random

import numpy as np

from arc_utils import find_components, get_bg, label_mask, paste

# Design analysis for the D3 variant
# ----------------------------------
# spec:
#   variant_goal:
#     keep the original containment/tree-to-canonical-output rule intact
#     while changing only the source-frame geometry from rough rectangles
#     to rough circles / ellipses. Output frames remain rectangular.
#   a_concretization:
#     input frames are now one-pixel-thick closed oval loops. Their bbox is
#     still 7x7 to 20x24, but the top/bottom/left/right touches concentrate
#     near midpoints instead of spanning long straight sides. A small example:
#     an original 11x15 frame might look like a shaky rectangle whose children
#     fit because the interior is box-shaped; in the variant the same children
#     sit inside an 11x15 oval loop. Solver execution difference:
#       original generator: place children inside a rectangular margin, then
#       draw a wobbly rectangle around them.
#       variant generator: place children in a layout group first, then search
#       for a translation where all child pixels lie inside the actual oval hole.
#     main() itself still only needs hole detection + sibling ordering, so the
#     solver rule is unchanged apart from documentation.
#   b_complexity_calibration:
#     the change is visually obvious because frame pixels no longer follow the
#     bbox edges. To avoid the task collapsing back to rectangle-like behavior,
#     each generated frame uses midpoint/Bresenham ellipse profiles plus small
#     per-quadrant jitter, leaving empty bbox corners and short side-contact runs.
#     Freedom is controlled: only 1-2 frames, 1-4 blobs, depth 1-2, same as the
#     original task, so the new geometric abstraction remains learnable.
#   c_implicit_assumption_audit:
#     - main() classifies frames via "has a hole", not via rectangularity:
#       still valid.
#     - containment uses "all child pixels lie inside a frame hole":
#       generator must enforce true geometric fit, not just bbox fit.
#     - sibling layout uses bbox overlap/order only:
#       generator must keep children separated in row-band or col-band as before.
#     - outside-blob side detection prefers bbox separation:
#       generator therefore keeps outside blobs fully beyond one bbox side.
#   d_placement_legality:
#     every frame is generated first as a valid 4-connected simple loop with one
#     hole. Child placements are accepted only if all child foreground pixels are
#     contained in that hole mask. This prevents overlaps and guarantees main()
#     reconstructs the intended tree.
#   e_visual_observability:
#     from a few examples, humans can see that freehand ovals become rectangles
#     in the output. The rule does not depend on hidden ids, only on visible
#     enclosure and relative placement.
#   f_visual_tidiness:
#     children are arranged with gaps exactly as in the original task, then
#     translated into a central feasible region of the oval hole. This avoids
#     frame-child tangencies and crowded corners.
#   g_color_inference:
#     colors remain "one bg + one fg"; no new color logic is introduced.


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


def _component_holes(shape: tuple[int, int], coords: np.ndarray) -> list[np.ndarray]:
    mask = np.zeros(shape, dtype=bool)
    mask[tuple(coords.T)] = True
    lab, n = label_mask(~mask, connectivity=4)
    boundary = set(lab[0, :]) | set(lab[-1, :]) | set(lab[:, 0]) | set(lab[:, -1])
    return [(lab == k) for k in range(1, n + 1) if k not in boundary]


def _intervals_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(a[0], b[0]) <= min(a[1], b[1])


def _build_layout(node: _Node) -> None:
    if node.mask is not None:
        return

    if not node.is_frame:
        node.mask = np.ones((1, 1), dtype=bool)
        node.anchor = (0, 0)
        node.axis_map = {
            "row": [(float(node.center[0]), 0)],
            "col": [(float(node.center[1]), 0)],
        }
        return

    assert node.children
    child_mask, _placements, axis_map = _layout_group(node.children)
    h, w = child_mask.shape
    out = np.zeros((h + 4, w + 4), dtype=bool)
    out[0, :] = True
    out[-1, :] = True
    out[:, 0] = True
    out[:, -1] = True
    out[2 : 2 + h, 2 : 2 + w] |= child_mask

    node.mask = out
    node.anchor = (out.shape[0] // 2, out.shape[1] // 2)
    node.axis_map = {
        "row": [(orig, pos + 2) for orig, pos in axis_map["row"]],
        "col": [(orig, pos + 2) for orig, pos in axis_map["col"]],
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


def _layout_group(nodes: list[_Node]) -> tuple[np.ndarray, dict[_Node, tuple[int, int]], dict[str, list[tuple[float, int]]]]:
    assert nodes
    for node in nodes:
        _build_layout(node)

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


def _layout_root(nodes: list[_Node]) -> np.ndarray:
    assert nodes
    for node in nodes:
        _build_layout(node)

    frames = [node for node in nodes if node.is_frame]
    dots = [node for node in nodes if not node.is_frame]
    if len(frames) == 1 and dots:
        frame = frames[0]
        assert frame.mask is not None
        placements: dict[_Node, tuple[int, int]] = {frame: (0, 0)}
        extents = [(0, 0, frame.mask.shape[0], frame.mask.shape[1])]

        for dot in dots:
            side = _outside_side(dot, frame)
            if side in {"right", "left"}:
                r = _map_frame_axis(frame, 0, float(dot.center[0]))
                c = frame.mask.shape[1] + 1 if side == "right" else -2
            else:
                r = frame.mask.shape[0] + 1 if side == "bottom" else -2
                c = _map_frame_axis(frame, 1, float(dot.center[1]))
            placements[dot] = (int(r), int(c))
            extents.append((r - 1, c - 1, r + 2, c + 2))

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
                out[rr, cc] = True
        return out

    mask, _placements, _axis_map = _layout_group(nodes)
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert a nested sketch of oval loops and blobs into a clean canonical frame-and-dot diagram.

    Input structure:
    - The grid uses exactly one background color and one foreground color.
    - Foreground components are either small filled blobs or one-pixel-thick closed oval/elliptic loops.
    - Loops may enclose other loops/blobs; a blob outside the outer loop stays outside.

    Rule:
    - Recover the containment tree by finding which components lie in the hole of each loop.
    - Collapse every blob to one foreground pixel.
    - Replace every enclosing loop by a rectangular border with a two-cell inset around its direct children.
    - Preserve sibling row/column ordering from the input when laying out the canonical objects, with one empty cell between them.

    Key insight: The input frame shape is no longer rectangular, so the solver must ignore curve geometry and rely only on enclosure plus relative placement.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    fg_colors = sorted(set(map(int, g.flatten())) - {bg})
    assert len(fg_colors) == 1
    fg = int(fg_colors[0])

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, colors=[fg], return_format="dict")
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
        node_holes = _component_holes(g.shape, coords)
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
    mask = _layout_root(roots)
    out = np.full(mask.shape, bg, dtype=int)
    out[mask] = fg
    return out


# --------------- generator (ellipse variant) ---------------

# Grid / canvas bounds
GRID_MIN = 16
GRID_MAX = 25
MAX_EXTRA_PAD = 12
MAX_PAD_PER_SIDE = 6

# Frame geometry
FRAME_H_RANGE = (7, 20)
FRAME_W_RANGE = (7, 24)
FRAME_RETRIES = 80
FRAME_SIZE_SLACK = (1, 3)
ELLIPSE_JITTER = 1
ELLIPSE_JITTER_PROB = 0.45

# Blob geometry
BLOB_SIZE_RANGE = (4, 11)
BLOB_BBOX_RANGE = (2, 5)

# Layout / arrangement
ARRANGE_GAP_COMPACT = (2, 3)
ARRANGE_GAP_NORMAL = (2, 5)
ARRANGE_EXTRA_SPAN = (0, 4)
FRAME_MARGIN_RANGE = (3, 5)
COMPACT_MARGIN = 2

# Outside blob placement
OUTSIDE_GAP_RANGE = (2, 5)

# Tree structure probabilities
DEPTH1_PROB = 0.35
OUTSIDE_BLOB_PROB = 0.35
DEPTH1_INSIDE_RANGE = (1, 3)
DEPTH2_INNER_BLOBS = (1, 2)
DEPTH2_OUTER_BLOBS = (0, 2)
MAX_TOTAL_CHILDREN = 4

# Generation retry budget
GEN_RETRIES = 300


class _Obj:
    def __init__(self, kind, children=None, outside_side=None):
        self.kind = kind
        self.children = list(children or [])
        self.mask = None
        self.rel = (0, 0)
        self.pos = (0, 0)
        self.outside_side = outside_side


def _loop_ok(mask: np.ndarray) -> bool:
    """Check that mask is a single 4-connected simple closed curve with exactly one hole."""
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


def _midpoint_profile(a: int, b: int) -> list[int]:
    """Quarter-ellipse profile y[x] from top point to right/left point using midpoint/Bresenham ellipse logic."""
    x = 0
    y = b
    a2 = a * a
    b2 = b * b
    dx = 0
    dy = 2 * a2 * y
    d1 = b2 - a2 * b + 0.25 * a2
    pts: list[tuple[int, int]] = []

    while dx < dy:
        pts.append((x, y))
        if d1 < 0:
            x += 1
            dx = 2 * b2 * x
            d1 += dx + b2
        else:
            x += 1
            y -= 1
            dx = 2 * b2 * x
            dy = 2 * a2 * y
            d1 += dx - dy + b2

    d2 = b2 * (x + 0.5) ** 2 + a2 * (y - 1) ** 2 - a2 * b2
    while y >= 0:
        pts.append((x, y))
        if d2 > 0:
            y -= 1
            dy = 2 * a2 * y
            d2 += a2 - dy
        else:
            x += 1
            y -= 1
            dx = 2 * b2 * x
            dy = 2 * a2 * y
            d2 += dx - dy + a2

    profile = {}
    for xx, yy in pts:
        profile[xx] = max(profile.get(xx, -1), yy)

    out = []
    last = b
    for xx in range(a + 1):
        if xx in profile:
            last = profile[xx]
        out.append(last)
    out[-1] = 0
    for xx in range(a - 1, -1, -1):
        out[xx] = max(out[xx], out[xx + 1])
    return out


def _jitter_profile(base: list[int]) -> list[int]:
    """Small monotone perturbation so the loop reads as hand-drawn rather than perfectly analytic."""
    out = list(base)
    top = base[0]
    for i in range(1, len(out) - 1):
        if random.random() < ELLIPSE_JITTER_PROB:
            out[i] += random.choice((-ELLIPSE_JITTER, ELLIPSE_JITTER))
    out[0] = top
    out[-1] = 0
    for i in range(1, len(out) - 1):
        out[i] = max(0, min(top, out[i]))
    for i in range(1, len(out)):
        out[i] = min(out[i], out[i - 1])
    out[-1] = 0
    for i in range(len(out) - 2, -1, -1):
        out[i] = max(out[i], out[i + 1])
    return out


def _move_h(path: list[tuple[int, int]], target_c: int) -> None:
    r, c = path[-1]
    step = 1 if target_c > c else -1
    while c != target_c:
        c += step
        path.append((r, c))


def _move_v(path: list[tuple[int, int]], target_r: int) -> None:
    r, c = path[-1]
    step = 1 if target_r > r else -1
    while r != target_r:
        r += step
        path.append((r, c))


def _ellipse_path(a_left: int, a_right: int, b_top: int, b_bottom: int) -> list[tuple[int, int]]:
    ur = _jitter_profile(_midpoint_profile(a_right, b_top))
    dr = _jitter_profile(_midpoint_profile(a_right, b_bottom))
    dl = _jitter_profile(_midpoint_profile(a_left, b_bottom))
    ul = _jitter_profile(_midpoint_profile(a_left, b_top))

    path = [(-ur[0], 0)]

    for x in range(1, a_right + 1):
        _move_h(path, x)
        _move_v(path, -ur[x])

    for x in range(a_right - 1, -1, -1):
        _move_v(path, dr[x])
        _move_h(path, x)

    for x in range(1, a_left + 1):
        _move_h(path, -x)
        _move_v(path, dl[x])

    for x in range(a_left - 1, -1, -1):
        _move_v(path, -ul[x])
        _move_h(path, -x)

    if path[-1] == path[0]:
        path.pop()
    return path


def _frame_mask(h: int, w: int) -> np.ndarray | None:
    """Generate a rough circular/elliptic loop with the requested bbox."""
    a_left = (w - 1) // 2
    a_right = (w - 1) - a_left
    b_top = (h - 1) // 2
    b_bottom = (h - 1) - b_top

    if min(a_left, a_right, b_top, b_bottom) < 2:
        return None

    for _ in range(FRAME_RETRIES):
        path = _ellipse_path(a_left, a_right, b_top, b_bottom)
        coords = np.asarray(path, dtype=int)
        lo = coords.min(axis=0)
        hi = coords.max(axis=0)
        if int(hi[0] - lo[0] + 1) != h or int(hi[1] - lo[1] + 1) != w:
            continue
        mask = np.zeros((h, w), dtype=bool)
        shifted = coords - lo
        mask[shifted[:, 0], shifted[:, 1]] = True
        if mask.sum() != len(path):
            continue
        if _loop_ok(mask):
            return mask
    return None


def _hole_mask(mask: np.ndarray) -> np.ndarray | None:
    holes = _component_holes(mask.shape, np.argwhere(mask))
    if len(holes) != 1:
        return None
    return holes[0]


def _blob_mask() -> np.ndarray:
    """Generate a small filled blob mask (size 4-11, bbox 2x2 to 4x4)."""
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
    """Arrange children in a random layout (vertical, horizontal, or diagonal)."""
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
        group_h = sum(hh for hh, _ in dims) + sum(gaps)
        group_w = max(ww for _, ww in dims) + random.randint(*ARRANGE_EXTRA_SPAN)
        r = 0
        for child, (hh, ww), gap in zip(children, dims, gaps + [0]):
            placements[child] = (r, random.randint(0, group_w - ww))
            r += hh + gap
        return placements, group_h, group_w

    if mode == "horizontal":
        group_h = max(hh for hh, _ in dims) + random.randint(*ARRANGE_EXTRA_SPAN)
        group_w = sum(ww for _, ww in dims) + sum(gaps)
        c = 0
        for child, (hh, ww), gap in zip(children, dims, gaps + [0]):
            placements[child] = (random.randint(0, group_h - hh), c)
            c += ww + gap
        return placements, group_h, group_w

    group_h = sum(hh for hh, _ in dims) + sum(gaps)
    group_w = sum(ww for _, ww in dims) + sum(gaps)
    r = c = 0
    for child, (hh, ww), gap in zip(children, dims, gaps + [0]):
        placements[child] = (r, c)
        r += hh + gap
        c += ww + gap
    return placements, group_h, group_w


def _group_cells(placements: dict[_Obj, tuple[int, int]]) -> np.ndarray:
    coords = []
    for child, (r0, c0) in placements.items():
        child_coords = np.argwhere(child.mask) + np.array([r0, c0])
        coords.append(child_coords)
    return np.vstack(coords).astype(int)


def _fit_group_in_hole(hole: np.ndarray, placements: dict[_Obj, tuple[int, int]], group_h: int, group_w: int) -> tuple[int, int] | None:
    frame_h, frame_w = hole.shape
    if group_h > frame_h or group_w > frame_w:
        return None
    cells = _group_cells(placements)
    candidates = []
    for top in range(frame_h - group_h + 1):
        for left in range(frame_w - group_w + 1):
            rr = cells[:, 0] + top
            cc = cells[:, 1] + left
            if np.all(hole[rr, cc]):
                candidates.append((top, left))
    if not candidates:
        return None
    return random.choice(candidates)


def _prepare(node, compact=False):
    """Recursively build masks and place children within frames."""
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
    extra = random.randint(1, 2) if tight else random.randint(*FRAME_SIZE_SLACK)
    min_h = max(FRAME_H_RANGE[0], group_h + 2 * margin + extra)
    min_w = max(FRAME_W_RANGE[0], group_w + 2 * margin + extra)
    if min_h > FRAME_H_RANGE[1] or min_w > FRAME_W_RANGE[1]:
        return False

    h_hi = min(FRAME_H_RANGE[1], min_h + 3) if tight else FRAME_H_RANGE[1]
    w_hi = min(FRAME_W_RANGE[1], min_w + 4) if tight else FRAME_W_RANGE[1]

    for _ in range(FRAME_RETRIES):
        h = random.randint(min_h, h_hi)
        w = random.randint(min_w, w_hi)
        mask = _frame_mask(h, w)
        if mask is None:
            continue
        hole = _hole_mask(mask)
        if hole is None:
            continue
        offset = _fit_group_in_hole(hole, placements, group_h, group_w)
        if offset is None:
            continue
        top, left = offset
        for child, (r, c) in placements.items():
            child.rel = (top + r, left + c)
        node.mask = mask
        return True
    return False


def _walk(node, base=(0, 0)):
    """Flatten tree into list, computing absolute positions."""
    br, bc = base
    rr, cc = node.rel
    node.pos = (br + rr, bc + cc)
    out = [node]
    for child in node.children:
        out.extend(_walk(child, node.pos))
    return out


def _place_outside(root, outside):
    """Position an outside blob adjacent to the root frame."""
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
    """Convert a generator _Obj (with absolute pos) to a solver _Node."""
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


def _expected_output(root, outside, bg, fg):
    """Compute expected solver output for the generated tree."""

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
    mask = _layout_root(roots)
    out = np.full(mask.shape, bg, dtype=int)
    out[mask] = fg
    return out


def _make_tree():
    """Build a random containment tree (1-2 frames, 1-4 blobs, optional outside blob)."""
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


def _draw(objects, bg, fg):
    """Render all objects onto a padded grid. Returns None if size constraints fail."""
    min_r = min(obj.pos[0] for obj in objects)
    min_c = min(obj.pos[1] for obj in objects)
    max_r = max(obj.pos[0] + obj.mask.shape[0] for obj in objects)
    max_c = max(obj.pos[1] + obj.mask.shape[1] for obj in objects)
    draw_h = max_r - min_r
    draw_w = max_c - min_c
    if draw_h > GRID_MAX or draw_w > GRID_MAX:
        return None

    extra_h_min = max(0, GRID_MIN - draw_h)
    extra_w_min = max(0, GRID_MIN - draw_w)
    extra_h_max = min(MAX_EXTRA_PAD, GRID_MAX - draw_h)
    extra_w_max = min(MAX_EXTRA_PAD, GRID_MAX - draw_w)
    if extra_h_min > extra_h_max or extra_w_min > extra_w_max:
        return None
    extra_h = random.randint(extra_h_min, extra_h_max)
    extra_w = random.randint(extra_w_min, extra_w_max)
    top_pad = random.randint(max(0, extra_h - MAX_PAD_PER_SIDE), min(MAX_PAD_PER_SIDE, extra_h))
    left_pad = random.randint(max(0, extra_w - MAX_PAD_PER_SIDE), min(MAX_PAD_PER_SIDE, extra_w))
    shift_r = top_pad - min_r
    shift_c = left_pad - min_c

    grid = np.full((draw_h + extra_h, draw_w + extra_w), bg, dtype=int)
    for obj in objects:
        assert obj.mask is not None
        r = obj.pos[0] + shift_r
        c = obj.pos[1] + shift_c
        obj.pos = (r, c)
        patch = np.full(obj.mask.shape, bg, dtype=int)
        patch[obj.mask] = fg
        paste(grid, patch, r, c, transparent=bg)
    if np.count_nonzero(grid == bg) <= np.count_nonzero(grid == fg):
        return None
    return grid


def _gen_attempt():
    """Single generation attempt: build tree, render input, verify against solver."""
    bg, fg = random.sample(range(10), 2)
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

    grid = _draw(objects, bg, fg)
    if grid is None:
        return None
    expected = _expected_output(root, outside, bg, fg)
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
    raise RuntimeError("failed to generate valid 2d0172a1 input")
