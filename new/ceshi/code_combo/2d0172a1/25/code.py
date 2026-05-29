# Combo variant of 2d0172a1 (index 25)
# combo_spec: D7/2+D11/0+D14/2
# Name: 深度3层+环形排列+行列取反

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import find_components, get_bg, label_mask, paste


# Design analysis for the fused variant
# -------------------------------------
# a) Concretization / behavior trace:
#    The base task rebuilt each frame by packing direct children with independent
#    row-band and col-band coordinates. D14 replaces that with a ring: children
#    are ordered around the parent and mapped to cardinal slots. D7 then has to
#    act on that ring direction rather than on independent axes, so the fused
#    canonical slot cycle is reversed to counterclockwise: top -> left -> bottom
#    -> right. D11 fixes the tree to outer -> middle -> inner, so this same
#    counterclockwise ring rebuild happens recursively at three frame levels.
#    Example: outer contains [middle, blob_A], middle contains [inner, blob_B],
#    inner contains [blob_C, blob_D]. In the base task, each sibling group would
#    become a row/column/diagonal packing. In the fused task, each sibling group
#    is read by angular order around its parent; the child nearest 12 o'clock is
#    assigned to the top slot, then later children go left, bottom, right. The
#    result is a three-level rectangular nesting whose side children spiral in a
#    visibly counterclockwise manner.
# b) Complexity calibration:
#    To keep the ring behavior observable, the generator forces every frame to
#    have at least two direct children: outer has middle plus 1-2 blobs, middle
#    has inner plus 1-2 blobs, inner has 2-3 blobs. Direct children stay <= 4 so
#    they fit the four slots cleanly. Blob counts are small and frame margins are
#    tight, preventing size blow-ups on a 25-30 grid while still making the ring
#    visible at each level.
# c) Implicit-assumption review:
#    - `_component_holes()` and `node.is_frame` still rely on a frame being one
#      4-connected simple loop with exactly one hole; the generator preserves it.
#    - `_build_layout()` can no longer assume row/col decomposition inside a
#      frame; it must use a joint ring placement routine.
#    - `_layout_root()` still assumes any extra root object is a blob outside the
#      single outer frame. The generator keeps exactly one root frame and at most
#      one outside blob, so the special root branch remains valid.
#    - `frame.axis_map` is still needed for outside-blob alignment. The fused
#      solver rebuilds that map from ring placements, so `_map_frame_axis()`
#      continues to work without knowing about the ring directly.
#    - The direct-parent selection via smallest containing frame becomes more
#      important with depth 3, but remains correct because the middle frame is
#      always the smallest container of the inner frame and of its descendants.
# d) Placement legality:
#    Input generation uses a ring-sector placer with one child per cardinal slot
#    and random jitter only along the slot's tangent axis. That keeps siblings
#    separated, inside their parent, and recoverable by angular order. Outside
#    blobs are attached only if the sampled side interval is legal.
# e) Visual observability:
#    The transformation depends only on geometry humans can see: containment and
#    angular position. No color ids or hidden metadata affect ordering. Because
#    all three frames have multi-child rings, a solver can infer the repeated
#    counterclockwise pattern from a few examples.
# f) Visual cleanliness:
#    The canonical output leaves a central empty box in every ring, keeps one
#    background cell between canonical objects, and caps child count at four, so
#    the nested output stays readable instead of collapsing into overlap.
# g) Inferability of changes:
#    There is no color remapping. The only changed control variables are nesting
#    depth and sibling placement. The generator rejects samples whose output would
#    match the clockwise-ring alternative, preventing silent regression into a
#    visually ambiguous or effectively original behavior.


SLOT_ORDER_CCW = ("top", "left", "bottom", "right")
SLOT_ORDER_CW = ("top", "right", "bottom", "left")


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


def _polar_angle(parent_center: np.ndarray, point: np.ndarray, counterclockwise: bool) -> float:
    dr = float(point[0] - parent_center[0])
    dc = float(point[1] - parent_center[1])
    angle = math.atan2(-dc, -dr) if counterclockwise else math.atan2(dc, -dr)
    if angle < 0:
        angle += 2 * math.pi
    return angle


def _ring_order(children: list[_Node], parent_center: np.ndarray, counterclockwise: bool) -> list[_Node]:
    assert 1 <= len(children) <= 4
    items = [(child, _polar_angle(parent_center, child.center, counterclockwise)) for child in children]
    items.sort(key=lambda item: item[1])
    start = min(
        range(len(items)),
        key=lambda i: (min(items[i][1], 2 * math.pi - items[i][1]), items[i][1]),
    )
    return [items[(start + i) % len(items)][0] for i in range(len(items))]


def _canonical_ring_group(
    children: list[_Node], parent_center: np.ndarray, counterclockwise: bool
) -> tuple[np.ndarray, dict[_Node, tuple[int, int]], dict[str, list[tuple[float, int]]]]:
    for child in children:
        _build_layout(child, counterclockwise=counterclockwise)

    ordered = _ring_order(children, parent_center, counterclockwise=counterclockwise)
    slot_order = SLOT_ORDER_CCW if counterclockwise else SLOT_ORDER_CW
    slot_map = {slot: ordered[i] for i, slot in enumerate(slot_order[: len(ordered)])}

    def shape(side: str) -> tuple[int, int]:
        child = slot_map.get(side)
        if child is None:
            return 0, 0
        assert child.mask is not None
        return child.mask.shape

    top_h, top_w = shape("top")
    left_h, left_w = shape("left")
    bottom_h, bottom_w = shape("bottom")
    right_h, right_w = shape("right")

    center_h = max(1, left_h, right_h)
    center_w = max(1, top_w, bottom_w)
    top_gap = 1 if top_h else 0
    left_gap = 1 if left_w else 0
    bottom_gap = 1 if bottom_h else 0
    right_gap = 1 if right_w else 0

    center_r = top_h + top_gap
    center_c = left_w + left_gap
    group_h = top_h + top_gap + center_h + bottom_gap + bottom_h
    group_w = left_w + left_gap + center_w + right_gap + right_w
    out = np.zeros((group_h, group_w), dtype=bool)
    placements: dict[_Node, tuple[int, int]] = {}
    axis_map = {"row": [], "col": []}

    for side, child in slot_map.items():
        assert child.mask is not None and child.anchor is not None
        h, w = child.mask.shape
        if side == "top":
            r = 0
            c = center_c + (center_w - w) // 2
        elif side == "left":
            r = center_r + (center_h - h) // 2
            c = 0
        elif side == "bottom":
            r = center_r + center_h + bottom_gap
            c = center_c + (center_w - w) // 2
        else:
            r = center_r + (center_h - h) // 2
            c = center_c + center_w + right_gap
        placements[child] = (int(r), int(c))
        out[r : r + h, c : c + w] |= child.mask
        axis_map["row"].append((float(child.center[0]), int(r + child.anchor[0])))
        axis_map["col"].append((float(child.center[1]), int(c + child.anchor[1])))

    return out, placements, axis_map


def _build_layout(node: _Node, counterclockwise: bool) -> None:
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

    assert 1 <= len(node.children) <= 4
    child_mask, _placements, axis_map = _canonical_ring_group(
        node.children,
        node.center,
        counterclockwise=counterclockwise,
    )
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


def _axis_coordinates(nodes: list[_Node], axis: int, reverse_bands: bool) -> dict[_Node, int]:
    items = []
    for node in nodes:
        assert node.mask is not None and node.anchor is not None
        r0, c0, r1, c1 = node.bbox
        interval = (r0, r1) if axis == 0 else (c0, c1)
        mid = (interval[0] + interval[1]) / 2
        items.append((node, interval, mid, node.anchor[axis], node.mask.shape[axis]))

    bands: list[dict[str, object]] = []
    for item in sorted(items, key=lambda x: x[2]):
        interval = item[1]
        for band in bands:
            band_items = band["items"]
            assert isinstance(band_items, list)
            if any(_intervals_overlap(interval, other[1]) for other in band_items):
                band_items.append(item)
                band_interval = band["interval"]
                assert isinstance(band_interval, tuple)
                band["interval"] = (
                    min(band_interval[0], interval[0]),
                    max(band_interval[1], interval[1]),
                )
                break
        else:
            bands.append({"items": [item], "interval": interval})

    ordered_bands = list(reversed(bands)) if reverse_bands else bands
    coord_by_node: dict[_Node, int] = {}
    coord = 0
    for i, band in enumerate(ordered_bands):
        band_items = band["items"]
        assert isinstance(band_items, list)
        if i:
            prev_items = ordered_bands[i - 1]["items"]
            assert isinstance(prev_items, list)
            prev_down = max(item[4] - 1 - item[3] for item in prev_items)
            curr_up = max(item[3] for item in band_items)
            coord += prev_down + 2 + curr_up
        for item in band_items:
            coord_by_node[item[0]] = int(coord)

    return coord_by_node


def _layout_group(
    nodes: list[_Node], counterclockwise: bool, reverse_bands: bool
) -> tuple[np.ndarray, dict[_Node, tuple[int, int]], dict[str, list[tuple[float, int]]]]:
    assert nodes
    for node in nodes:
        _build_layout(node, counterclockwise=counterclockwise)

    row_coord = _axis_coordinates(nodes, axis=0, reverse_bands=reverse_bands)
    col_coord = _axis_coordinates(nodes, axis=1, reverse_bands=reverse_bands)

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


def _layout_root(nodes: list[_Node], counterclockwise: bool, reverse_bands: bool) -> np.ndarray:
    assert nodes
    for node in nodes:
        _build_layout(node, counterclockwise=counterclockwise)

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

    mask, _placements, _axis_map = _layout_group(
        nodes,
        counterclockwise=counterclockwise,
        reverse_bands=reverse_bands,
    )
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert a rough three-level nested frame sketch into a clean counterclockwise ring diagram.

    Input structure:
    - The grid contains exactly one background color and one foreground color.
    - Foreground components are either rough single-loop frames with one hole or small solid blobs.
    - The containment tree is always a depth-3 frame chain (outer contains middle contains inner), with a few blob siblings at each level and at most one blob outside the outer frame.

    Rule:
    - Recover the containment tree by testing which components lie inside each frame hole.
    - Replace every blob with one foreground pixel and every frame with a 1-pixel rectangular border padded two cells around its direct children.
    - For each frame, read its direct children by angular order around the parent center, starting from the child nearest 12 o'clock and then moving counterclockwise.
    - Rebuild those children onto the fixed counterclockwise ring slots top, left, bottom, right, while keeping one background cell between neighboring canonical objects; any outside blob stays attached to the corresponding side of the outer frame.

    Key insight:
    - The hand-drawn stroke wiggles do not matter. The stable information is the three-step enclosure chain plus the counterclockwise order in which siblings orbit each parent.
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
    mask = _layout_root(roots, counterclockwise=True, reverse_bands=True)
    out = np.full(mask.shape, bg, dtype=int)
    out[mask] = fg
    return out


# --------------- generator (o5x) ---------------

# Grid / canvas bounds
GRID_MIN = 25
GRID_MAX = 30
MAX_EXTRA_PAD = 8
MAX_PAD_PER_SIDE = 4

# Frame geometry
FRAME_H_RANGE = (7, 28)
FRAME_W_RANGE = (7, 30)
FRAME_CORNER_MAX = 3
FRAME_MIN_EDGE = 2
FRAME_TURN_RATIO = (0.34, 0.43)
FRAME_RETRIES = 120

# Blob geometry
BLOB_SIZE_RANGE = (3, 6)
BLOB_BBOX_RANGE = (2, 4)

# Ring placement inside frames
RING_GAP_COMPACT = (1, 1)
RING_GAP_NORMAL = (1, 2)
RING_CENTER_EXTRA = (0, 1)
FRAME_MARGIN_RANGE = (2, 3)
COMPACT_MARGIN = 2

# Outside blob placement
OUTSIDE_GAP_RANGE = (1, 2)

# Tree structure patterns
OUTSIDE_BLOB_PROB = 0.35
BLOB_DISTRIBUTIONS = [
    (1, 1, 2),
    (1, 1, 3),
    (1, 2, 2),
    (2, 1, 2),
]
MAX_DIRECT_CHILDREN = 4
# Rotate successful samples across the main structural branches instead of
# letting rejection bias collapse everything to the easiest case.
BRANCH_SPECS = [
    ((1, 1, 2), False),
    ((1, 1, 2), True),
    ((1, 1, 3), False),
    ((1, 2, 2), False),
    ((2, 1, 2), False),
]
BRANCH_RETRIES = 80
_BRANCH_CURSOR = 0

# Step-mix alternation bias
STEP_MIX_ALT_PROB = 0.7

# Generation retry budget
GEN_RETRIES = 500


class _Obj:
    def __init__(self, kind, children=None, outside_side=None):
        self.kind = kind
        self.children = list(children or [])
        self.mask: np.ndarray | None = None
        self.rel = (0, 0)
        self.pos = (0, 0)
        self.outside_side = outside_side


def _loop_ok(mask: np.ndarray) -> bool:
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


def _step_mix(n_a: int, step_a: tuple[int, int], n_b: int, step_b: tuple[int, int]) -> list[tuple[int, int]]:
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


def _add_line(path: list[tuple[int, int]], end: tuple[int, int]) -> None:
    r, c = path[-1]
    er, ec = end
    dr = 0 if er == r else (1 if er > r else -1)
    dc = 0 if ec == c else (1 if ec > c else -1)
    while (r, c) != (er, ec):
        r += dr
        c += dc
        path.append((r, c))


def _add_steps(path: list[tuple[int, int]], steps: list[tuple[int, int]]) -> None:
    r, c = path[-1]
    for dr, dc in steps:
        r += dr
        c += dc
        path.append((r, c))


def _turn_ratio(path: list[tuple[int, int]]) -> float:
    dirs = []
    cyc = path + [path[0]]
    for (r0, c0), (r1, c1) in zip(cyc, cyc[1:]):
        dirs.append((r1 - r0, c1 - c0))
    turns = sum(1 for a, b in zip(dirs, dirs[1:] + [dirs[0]]) if a != b)
    return turns / len(path)


def _frame_mask(h: int, w: int) -> np.ndarray | None:
    max_dx = min(FRAME_CORNER_MAX, max(1, (w - 5) // 2))
    max_dy = min(FRAME_CORNER_MAX, max(1, (h - 5) // 2))
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
        if _loop_ok(mask):
            return mask
    return None


def _blob_mask() -> np.ndarray:
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


def _frame_hole_mask(mask: np.ndarray) -> np.ndarray | None:
    lab, n = label_mask(~mask, connectivity=4)
    boundary = set(lab[0, :]) | set(lab[-1, :]) | set(lab[:, 0]) | set(lab[:, -1])
    holes = [(lab == k) for k in range(1, n + 1) if k not in boundary]
    if len(holes) != 1:
        return None
    return holes[0]


def _children_embed_ok(
    frame_mask: np.ndarray,
    placements: dict[_Obj, tuple[int, int]],
    children: list[_Obj],
) -> bool:
    hole = _frame_hole_mask(frame_mask)
    if hole is None:
        return False
    canvas = frame_mask.copy()
    for child in children:
        assert child.mask is not None
        r0, c0 = placements[child]
        coords = np.argwhere(child.mask) + np.array([r0, c0])
        if not np.all(hole[coords[:, 0], coords[:, 1]]):
            return False
        if np.any(canvas[coords[:, 0], coords[:, 1]]):
            return False
        canvas[coords[:, 0], coords[:, 1]] = True
    _labels, n = label_mask(canvas, connectivity=4)
    return n == 1 + len(children)


def _arrange_ring(children: list[_Obj], compact: bool) -> tuple[dict[_Obj, tuple[int, int]], int, int] | None:
    k = len(children)
    if not 1 <= k <= MAX_DIRECT_CHILDREN:
        return None

    order = children[:]
    random.shuffle(order)
    gap_lo, gap_hi = RING_GAP_COMPACT if compact else RING_GAP_NORMAL

    candidates = []
    for start in range(4):
        slot_cycle = [SLOT_ORDER_CCW[(start + i) % 4] for i in range(k)]
        slot_map = {slot: child for slot, child in zip(slot_cycle, order)}

        def shape(side: str) -> tuple[int, int]:
            child = slot_map.get(side)
            if child is None:
                return 0, 0
            assert child.mask is not None
            return child.mask.shape

        top_h, top_w = shape("top")
        left_h, left_w = shape("left")
        bottom_h, bottom_w = shape("bottom")
        right_h, right_w = shape("right")

        extra = 0 if compact else random.randint(*RING_CENTER_EXTRA)
        center_h = max(1, left_h, right_h) + extra
        center_w = max(1, top_w, bottom_w) + extra
        top_gap = (gap_lo if compact else random.randint(gap_lo, gap_hi)) if top_h else 0
        left_gap = (gap_lo if compact else random.randint(gap_lo, gap_hi)) if left_w else 0
        bottom_gap = (gap_lo if compact else random.randint(gap_lo, gap_hi)) if bottom_h else 0
        right_gap = (gap_lo if compact else random.randint(gap_lo, gap_hi)) if right_w else 0

        group_h = top_h + top_gap + center_h + bottom_gap + bottom_h
        group_w = left_w + left_gap + center_w + right_gap + right_w
        score = (max(group_h, group_w), group_h * group_w)
        candidates.append(
            (
                score,
                slot_map,
                center_h,
                center_w,
                top_gap,
                left_gap,
                bottom_gap,
                right_gap,
                group_h,
                group_w,
            )
        )

    best_score = min(item[0] for item in candidates)
    score_matches = [item for item in candidates if item[0] == best_score]
    (
        _score,
        slot_map,
        center_h,
        center_w,
        top_gap,
        left_gap,
        bottom_gap,
        right_gap,
        group_h,
        group_w,
    ) = random.choice(score_matches)

    center_r = top_gap + (slot_map["top"].mask.shape[0] if "top" in slot_map else 0)
    center_c = left_gap + (slot_map["left"].mask.shape[1] if "left" in slot_map else 0)
    placements: dict[_Obj, tuple[int, int]] = {}

    for side, child in slot_map.items():
        assert child.mask is not None
        h, w = child.mask.shape
        if side == "top":
            r = 0
            c = center_c + random.randint(0, center_w - w)
        elif side == "left":
            r = center_r + random.randint(0, center_h - h)
            c = 0
        elif side == "bottom":
            r = center_r + center_h + bottom_gap
            c = center_c + random.randint(0, center_w - w)
        else:
            r = center_r + random.randint(0, center_h - h)
            c = center_c + center_w + right_gap
        placements[child] = (r, c)

    return placements, group_h, group_w


def _prepare(node: _Obj, compact: bool = False) -> bool:
    if node.kind == "blob":
        node.mask = _blob_mask()
        return True

    if not 1 <= len(node.children) <= MAX_DIRECT_CHILDREN:
        return False
    for child in node.children:
        if not _prepare(child, compact=child.kind == "frame"):
            return False

    has_frame_child = any(child.kind == "frame" for child in node.children)
    tight = compact or has_frame_child
    arranged = _arrange_ring(node.children, compact=tight)
    if arranged is None:
        return False
    placements, group_h, group_w = arranged

    margin = COMPACT_MARGIN if tight else random.randint(*FRAME_MARGIN_RANGE)
    min_h = max(FRAME_H_RANGE[0], group_h + 2 * margin)
    min_w = max(FRAME_W_RANGE[0], group_w + 2 * margin)
    if min_h > FRAME_H_RANGE[1] or min_w > FRAME_W_RANGE[1]:
        return False
    if tight:
        h_hi = min(FRAME_H_RANGE[1], min_h + 3)
        w_hi = min(FRAME_W_RANGE[1], min_w + 4)
    else:
        h_hi = FRAME_H_RANGE[1]
        w_hi = FRAME_W_RANGE[1]
    for _ in range(20):
        h = random.randint(min_h, h_hi)
        w = random.randint(min_w, w_hi)
        mask = _frame_mask(h, w)
        if mask is None:
            continue
        top = margin + random.randint(0, h - 2 * margin - group_h)
        left = margin + random.randint(0, w - 2 * margin - group_w)
        shifted = {child: (top + r, left + c) for child, (r, c) in placements.items()}
        if not _children_embed_ok(mask, shifted, node.children):
            continue
        for child, rel in shifted.items():
            child.rel = rel
        node.mask = mask
        return True
    return False


def _walk(node: _Obj, base: tuple[int, int] = (0, 0)) -> list[_Obj]:
    br, bc = base
    rr, cc = node.rel
    node.pos = (br + rr, bc + cc)
    out = [node]
    for child in node.children:
        out.extend(_walk(child, node.pos))
    return out


def _place_outside(root: _Obj, outside: _Obj) -> bool:
    assert root.mask is not None and outside.mask is not None
    h, w = root.mask.shape
    bh, bw = outside.mask.shape
    gap = random.randint(*OUTSIDE_GAP_RANGE)
    side = outside.outside_side

    if side in {"right", "left"}:
        row_hi = h - bh - 1
        if row_hi < 1:
            return False
        row = random.randint(1, row_hi)
        col = w + gap if side == "right" else -gap - bw
    else:
        col_hi = w - bw - 1
        if col_hi < 1:
            return False
        row = h + gap if side == "bottom" else -gap - bh
        col = random.randint(1, col_hi)

    outside.rel = (row, col)
    return True


def _node_from_obj(obj: _Obj) -> _Node:
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


def _expected_output(
    root: _Obj,
    outside: _Obj | None,
    bg: int,
    fg: int,
    *,
    counterclockwise: bool,
    reverse_bands: bool,
) -> np.ndarray:
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
    mask = _layout_root(roots, counterclockwise=counterclockwise, reverse_bands=reverse_bands)
    out = np.full(mask.shape, bg, dtype=int)
    out[mask] = fg
    return out


def _make_tree(
    pattern: tuple[int, int, int] | None = None,
    want_outside: bool | None = None,
) -> tuple[_Obj, _Obj | None]:
    outer = _Obj("frame")
    middle = _Obj("frame")
    inner = _Obj("frame")

    if pattern is None:
        pattern = random.choice(BLOB_DISTRIBUTIONS)
    outer_extra, middle_extra, inner_count = pattern
    inner.children = [_Obj("blob") for _ in range(inner_count)]
    middle.children = [inner] + [_Obj("blob") for _ in range(middle_extra)]
    outer.children = [middle] + [_Obj("blob") for _ in range(outer_extra)]

    outside = None
    total_inside_blobs = outer_extra + middle_extra + inner_count
    if want_outside is None:
        want_outside = random.random() < OUTSIDE_BLOB_PROB
    # Keep the outside-root branch visible in training examples instead of
    # making it a near-dead case, while still capping total clutter.
    if want_outside and total_inside_blobs <= 5:
        outside = _Obj("blob", outside_side=random.choice(("left", "right", "top", "bottom")))
    return outer, outside


def _draw(objects: list[_Obj], bg: int, fg: int) -> np.ndarray | None:
    min_r = min(obj.pos[0] for obj in objects)
    min_c = min(obj.pos[1] for obj in objects)
    max_r = max(obj.pos[0] + obj.mask.shape[0] for obj in objects if obj.mask is not None)
    max_c = max(obj.pos[1] + obj.mask.shape[1] for obj in objects if obj.mask is not None)
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


def _gen_attempt(
    pattern: tuple[int, int, int] | None = None,
    want_outside: bool | None = None,
) -> np.ndarray | None:
    bg, fg = random.sample(range(10), 2)
    root, outside = _make_tree(pattern=pattern, want_outside=want_outside)
    if not _prepare(root):
        return None
    if outside is not None:
        if not _prepare(outside):
            return None
        if not _place_outside(root, outside):
            return None

    objects = _walk(root)
    if outside is not None:
        outside.pos = outside.rel
        objects.append(outside)

    grid = _draw(objects, bg, fg)
    if grid is None:
        return None

    expected = _expected_output(root, outside, bg, fg, counterclockwise=True, reverse_bands=True)
    clockwise = _expected_output(root, outside, bg, fg, counterclockwise=False, reverse_bands=False)
    if np.array_equal(expected, clockwise):
        return None

    try:
        got = main(grid)
    except Exception:
        return None
    if not np.array_equal(got, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _BRANCH_CURSOR

    for branch_offset in range(len(BRANCH_SPECS)):
        branch_index = (_BRANCH_CURSOR + branch_offset) % len(BRANCH_SPECS)
        pattern, want_outside = BRANCH_SPECS[branch_index]
        for _ in range(BRANCH_RETRIES):
            grid = _gen_attempt(pattern=pattern, want_outside=want_outside)
            if grid is not None:
                _BRANCH_CURSOR = (branch_index + 1) % len(BRANCH_SPECS)
                return grid

    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 2d0172a1 combo input")
