# Combo variant of 2d0172a1 (index 27)
# combo_spec: D3/1+D17/0
# Name: 圆形/椭圆框架+保留宽高比

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import find_components, get_bg, label_mask, paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concretization and behavior trace
#    - Original task:
#      input frames are rough rectangular loops; output frame size is determined
#      only by the canonical child layout `(child_h + 4, child_w + 4)`.
#    - Combo task:
#      input frames are one-pixel-thick circular / elliptic loops, while output
#      frames are still rectangles. However, the output rectangle now inherits
#      the input frame bbox aspect ratio by integer scaling.
#    - Concrete trace:
#      suppose a frame encloses a child layout whose canonical mask is 5x3, so
#      the old minimum output frame would be 9x7. If the input oval bbox is
#      12x18, the reduced ratio is 2:3, therefore the combo output frame becomes
#      the smallest `(2k, 3k)` that can contain 9x7, namely 10x15. If the same
#      children sit inside an 18x12 oval, output becomes 15x10 instead. The
#      containment tree and sibling order stay the same; only source geometry
#      and frame-size transfer change.
# b) Complexity calibration
#    - To keep the aspect-ratio rule visible from 2-4 examples, the generator
#      draws frame bboxes from a small repeated ratio family:
#      `(1:1, 2:3, 3:2, 3:4, 4:3)`.
#    - Non-square ratios are preferred, so outputs often remain visibly wide or
#      tall even when children themselves would fit in a nearly square box.
#    - The ellipse geometry is also visually obvious because bbox corners are
#      typically empty and the loop follows midpoint-ellipse arcs with jitter.
# c) Implicit-assumption audit
#    - `main()` detects frames by "has at least one hole", not by rectangular
#      edges; this still holds for ellipses.
#    - Containment testing assumes every child component lies fully inside a
#      frame hole; the generator therefore fits children against the true hole
#      mask, not just against the frame bbox.
#    - The original `_build_layout()` assumed `frame_shape = child_shape + 4`;
#      this no longer holds. It is replaced by `_scaled_frame_shape()`.
#    - Outside-blob placement relies on bbox-side separation and frame axis-map
#      alignment. That remains valid, but `axis_map` must shift by the actual
#      centered inset, not by a hard-coded `2`.
# d) Placement legality
#    - Every generated frame is validated as a single 4-connected simple loop
#      with exactly one hole.
#    - Child placements are accepted only when every child foreground cell lands
#      inside that hole mask, preventing overlap with the frame and guaranteeing
#      the intended containment tree.
#    - Outside blobs are placed fully beyond one side of the outer frame bbox so
#      side recovery in the solver is stable.
# e) Visual observability
#    - A human can see two linked regularities:
#      ellipse-like input loops become rectangles, and tall/wide ellipses become
#      correspondingly tall/wide output rectangles.
#    - No hidden feature such as color ids or generation order is used.
# f) Visual tidiness
#    - Child groups are arranged with the original band-order constraints, then
#      translated toward the center of the ellipse hole to avoid awkward edge
#      crowding. This keeps the grids readable.
# g) Inferability of changes
#    - Colors remain binary (one bg, one fg), so the fusion does not introduce
#      extra color ambiguity. The new signal is geometric: enclosure + measured
#      bbox ratio.
# New solver assumptions and generator support
#    - Solver assumption: each frame bbox aspect ratio is meaningful.
#      Generator support: most frames come from a small ratio family with many
#      non-square cases.
#    - Solver assumption: each frame has at least one child so ratio transfer is
#      exercised on actual canonical content.
#      Generator support: tree construction preserves the original non-empty
#      frame constraint.


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


def _bbox_hw(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _scaled_frame_shape(bbox: tuple[int, int, int, int], min_h: int, min_w: int) -> tuple[int, int]:
    src_h, src_w = _bbox_hw(bbox)
    g = math.gcd(src_h, src_w)
    unit_h = src_h // g
    unit_w = src_w // g
    k = max(_ceil_div(min_h, unit_h), _ceil_div(min_w, unit_w), 1)
    return unit_h * k, unit_w * k


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
    child_h, child_w = child_mask.shape
    min_h = child_h + 4
    min_w = child_w + 4
    out_h, out_w = _scaled_frame_shape(node.bbox, min_h, min_w)
    inset_r = 2 + (out_h - min_h) // 2
    inset_c = 2 + (out_w - min_w) // 2

    out = np.zeros((out_h, out_w), dtype=bool)
    out[0, :] = True
    out[-1, :] = True
    out[:, 0] = True
    out[:, -1] = True
    out[inset_r : inset_r + child_h, inset_c : inset_c + child_w] |= child_mask

    node.mask = out
    node.anchor = (out_h // 2, out_w // 2)
    node.axis_map = {
        "row": [(orig, pos + inset_r) for orig, pos in axis_map["row"]],
        "col": [(orig, pos + inset_c) for orig, pos in axis_map["col"]],
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
    Convert a nested sketch of oval-like loops and blobs into a clean rectangle-and-dot diagram that preserves frame aspect ratios.

    Input structure:
    - The grid contains exactly one background color and one foreground color.
    - Foreground components are either small filled blobs or one-pixel-thick closed circular/elliptic loops.
    - Loops may enclose other loops or blobs, and one blob may remain outside the outermost loop.

    Rule:
    - Recover the containment tree by detecting which components lie inside each loop hole.
    - Collapse every blob to a single foreground pixel.
    - Replace every loop by a 1-pixel rectangular border whose size is the smallest integer-scaled copy of the input loop bbox ratio that can still contain its direct children with the usual border gap.
    - Preserve sibling row/column ordering, keep one empty cell between canonical objects, and keep any outside blob attached to the same side of the root frame.

    Key insight: The hand-drawn curve itself is normalized away, but two geometric facts survive cleanup: who contains whom, and whether each enclosing loop was visibly tall, wide, or square.
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


# --------------- generator (combo) ---------------

# Grid / canvas bounds
GRID_MIN = 16
GRID_MAX = 25
MAX_EXTRA_PAD = 12
MAX_PAD_PER_SIDE = 6

# Frame geometry
FRAME_H_RANGE = (7, 20)
FRAME_W_RANGE = (7, 24)
FRAME_RETRIES = 100
ELLIPSE_JITTER = 1
ELLIPSE_JITTER_PROB = 0.45
FRAME_RATIO_UNITS = ((1, 1), (2, 3), (3, 2), (3, 4), (4, 3))
FRAME_SLACK_CAP = 9
COMPACT_FRAME_SLACK_CAP = 5
FRAME_SIZE_EXTRA = (1, 3)
COMPACT_FRAME_SIZE_EXTRA = (1, 2)

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
STRICT_TARGETED_RETRIES = 120
TARGETED_RETRIES = 220

SCENARIO_SEQUENCE = (
    ("depth1", False),
    ("depth1", True),
    ("depth2", False),
    ("depth2", True),
)
SIDE_SEQUENCE = ("left", "top", "right", "bottom")
ROOT_RATIO_SEQUENCE = ((2, 3), (3, 2), (3, 4), (4, 3), (1, 1))
INNER_RATIO_SEQUENCE = ((3, 4), (4, 3), (2, 3), (3, 2), (1, 1))
_GENERATE_COUNTER = 0


class _Obj:
    def __init__(self, kind: str, children=None, outside_side: str | None = None):
        self.kind = kind
        self.children = list(children or [])
        self.mask: np.ndarray | None = None
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
    """Quarter-ellipse profile generated by midpoint/Bresenham ellipse logic."""
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
    """Small monotone perturbation so the loop is hand-drawn rather than analytic."""
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
        shifted = coords - lo
        mask = np.zeros((h, w), dtype=bool)
        mask[shifted[:, 0], shifted[:, 1]] = True
        if int(mask.sum()) != len(path):
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


def _arrange(children: list[_Obj], compact: bool = False):
    """Arrange children in a vertical, horizontal, or diagonal layout."""
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


def _fit_group_in_hole(
    hole: np.ndarray,
    placements: dict[_Obj, tuple[int, int]],
    group_h: int,
    group_w: int,
) -> tuple[int, int] | None:
    frame_h, frame_w = hole.shape
    if group_h > frame_h or group_w > frame_w:
        return None
    cells = _group_cells(placements)
    hole_center = ((frame_h - 1) / 2.0, (frame_w - 1) / 2.0)
    group_center = ((group_h - 1) / 2.0, (group_w - 1) / 2.0)
    candidates: list[tuple[float, int, int]] = []
    for top in range(frame_h - group_h + 1):
        for left in range(frame_w - group_w + 1):
            rr = cells[:, 0] + top
            cc = cells[:, 1] + left
            if not np.all(hole[rr, cc]):
                continue
            center_r = top + group_center[0]
            center_c = left + group_center[1]
            dist = abs(center_r - hole_center[0]) + abs(center_c - hole_center[1])
            candidates.append((dist + random.random() * 0.05, top, left))
    if not candidates:
        return None
    candidates.sort()
    topk = candidates[: min(10, len(candidates))]
    _, top, left = random.choice(topk)
    return int(top), int(left)


def _candidate_frame_sizes(
    min_h: int,
    min_w: int,
    group_h: int,
    group_w: int,
    compact: bool,
    preferred_units: tuple[int, int] | None = None,
) -> list[tuple[int, int]]:
    child_aspect = group_w / group_h
    slack_cap = COMPACT_FRAME_SLACK_CAP if compact else FRAME_SLACK_CAP
    scored: list[tuple[float, int, int]] = []
    for unit_h, unit_w in FRAME_RATIO_UNITS:
        if preferred_units is not None and (unit_h, unit_w) != preferred_units:
            continue
        q_min = max(_ceil_div(min_h, unit_h), _ceil_div(min_w, unit_w))
        q_max = min(FRAME_H_RANGE[1] // unit_h, FRAME_W_RANGE[1] // unit_w)
        if q_min > q_max:
            continue
        for q in range(q_min, q_max + 1):
            h = unit_h * q
            w = unit_w * q
            slack_h = h - min_h
            slack_w = w - min_w
            slack_total = slack_h + slack_w
            if slack_total > slack_cap:
                continue
            frame_aspect = w / h
            mismatch = abs(math.log(frame_aspect / child_aspect))
            non_square_bonus = 0.2 if unit_h != unit_w else -0.15
            slack_bonus = 0.08 * min(slack_total, 6)
            imbalance_penalty = 0.03 * abs(slack_h - slack_w)
            score = mismatch + non_square_bonus + slack_bonus - imbalance_penalty
            score += random.random() * 0.02
            scored.append((score, h, w))
    scored.sort(reverse=True)
    return [(h, w) for _, h, w in scored]


def _prepare(
    node: _Obj,
    compact: bool = False,
    preferred_units: tuple[int, int] | None = None,
    child_frame_units: tuple[int, int] | None = None,
    strict_units: bool = False,
) -> bool:
    """Recursively build masks and place children within frame holes."""
    if node.kind == "blob":
        node.mask = _blob_mask()
        return True

    for child in node.children:
        child_units = child_frame_units if child.kind == "frame" else None
        if not _prepare(child, compact=child.kind == "frame", preferred_units=child_units):
            return False

    has_frame_child = any(child.kind == "frame" for child in node.children)
    tight = compact or has_frame_child
    placements, group_h, group_w = _arrange(node.children, compact=tight)
    margin = COMPACT_MARGIN if tight else random.randint(*FRAME_MARGIN_RANGE)
    extra_lo, extra_hi = COMPACT_FRAME_SIZE_EXTRA if tight else FRAME_SIZE_EXTRA
    extra = random.randint(extra_lo, extra_hi)
    min_h = max(FRAME_H_RANGE[0], group_h + 2 * margin + extra)
    min_w = max(FRAME_W_RANGE[0], group_w + 2 * margin + extra)
    if min_h > FRAME_H_RANGE[1] or min_w > FRAME_W_RANGE[1]:
        return False

    candidates = _candidate_frame_sizes(
        min_h,
        min_w,
        group_h,
        group_w,
        compact=tight,
        preferred_units=preferred_units,
    )
    if not candidates and preferred_units is not None:
        if strict_units:
            return False
        candidates = _candidate_frame_sizes(min_h, min_w, group_h, group_w, compact=tight)
    if not candidates:
        return False

    for h, w in candidates[: min(20, len(candidates))]:
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


def _walk(node: _Obj, base=(0, 0)) -> list[_Obj]:
    """Flatten the tree while computing absolute positions."""
    br, bc = base
    rr, cc = node.rel
    node.pos = (br + rr, bc + cc)
    out = [node]
    for child in node.children:
        out.extend(_walk(child, node.pos))
    return out


def _place_outside(root: _Obj, outside: _Obj) -> None:
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


def _node_from_obj(obj: _Obj) -> _Node:
    """Convert a generator object into a solver node."""
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


def _expected_output(root: _Obj, outside: _Obj | None, bg: int, fg: int) -> np.ndarray:
    """Compute the expected combo output from the generated tree."""

    def add(obj: _Obj) -> _Node:
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


def _make_tree(
    mode: str | None = None,
    want_outside: bool | None = None,
    outside_side: str | None = None,
) -> tuple[_Obj, _Obj | None]:
    """Build a random containment tree with 1-2 frames and 1-4 blobs."""
    root = _Obj("frame")
    outside = None
    sides = ("left", "right", "top", "bottom")
    if mode is None:
        mode = "depth1" if random.random() < DEPTH1_PROB else "depth2"
    if want_outside is None:
        want_outside = random.random() < OUTSIDE_BLOB_PROB
    if outside_side is None:
        outside_side = random.choice(sides)

    if mode == "depth1":
        n_inside = random.randint(*DEPTH1_INSIDE_RANGE)
        root.children = [_Obj("blob") for _ in range(n_inside)]
        if want_outside and n_inside < MAX_TOTAL_CHILDREN:
            outside = _Obj("blob", outside_side=outside_side)
    else:
        inner = _Obj("frame")
        inner.children = [_Obj("blob") for _ in range(random.randint(*DEPTH2_INNER_BLOBS))]
        outer_blobs = [_Obj("blob") for _ in range(random.randint(*DEPTH2_OUTER_BLOBS))]
        root.children = [inner] + outer_blobs
        if want_outside and len(inner.children) + len(outer_blobs) < MAX_TOTAL_CHILDREN:
            outside = _Obj("blob", outside_side=outside_side)
    return root, outside


def _draw(objects: list[_Obj], bg: int, fg: int) -> np.ndarray | None:
    """Render all objects onto a padded grid."""
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


def _gen_attempt(spec: dict | None = None) -> np.ndarray | None:
    """Single generation attempt: build a valid combo input and self-check it."""
    bg, fg = random.sample(range(10), 2)
    spec = spec or {}
    root, outside = _make_tree(
        mode=spec.get("mode"),
        want_outside=spec.get("want_outside"),
        outside_side=spec.get("outside_side"),
    )
    if not _prepare(
        root,
        preferred_units=spec.get("root_units"),
        child_frame_units=spec.get("inner_units"),
        strict_units=bool(spec.get("strict_root_units")),
    ):
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


def generate_input() -> np.ndarray:
    global _GENERATE_COUNTER
    idx = _GENERATE_COUNTER
    _GENERATE_COUNTER += 1
    scenario_mode, want_outside = SCENARIO_SEQUENCE[idx % len(SCENARIO_SEQUENCE)]
    outside_case_idx = idx // 2
    preferred = {
        "mode": scenario_mode,
        "want_outside": want_outside,
        "outside_side": SIDE_SEQUENCE[outside_case_idx % len(SIDE_SEQUENCE)],
        "root_units": ROOT_RATIO_SEQUENCE[idx % len(ROOT_RATIO_SEQUENCE)],
        "inner_units": INNER_RATIO_SEQUENCE[(idx + 1) % len(INNER_RATIO_SEQUENCE)],
        "strict_root_units": True,
    }

    for attempt in range(GEN_RETRIES):
        if attempt < STRICT_TARGETED_RETRIES:
            spec = preferred
        elif attempt < TARGETED_RETRIES:
            spec = dict(preferred)
            spec["strict_root_units"] = False
        else:
            spec = None
        grid = _gen_attempt(spec)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 2d0172a1 combo input")
