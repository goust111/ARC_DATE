# Combo variant of 2d0172a1 (index 13)
# combo_spec: D1/2+D3/0
# Name: 双线边框框架+矩形框架

from __future__ import annotations

import random

import numpy as np

from arc_utils import find_components, get_bg, label_mask, paste


# Design analysis for the combo variant
# =====================================
# combo_summary:
#   semantic_change:
#     - Each semantic frame is no longer a single rough one-pixel loop.
#     - Instead, it is encoded by two directly nested perfect rectangular rings.
#     - Visually this looks like two nested rectangles, but semantically it is one double-line frame.
#     - Output still collapses every semantic frame to one canonical 1px rectangular border.
#   concrete_behavior_trace:
#     - Small example:
#         original raw input:
#           one outer wobbly frame containing one blob and one inner frame.
#         combo raw input:
#           the outer semantic frame becomes outer_rect + inner_rect inset by 2 cells;
#           the inner semantic frame also becomes its own outer_rect + inner_rect pair;
#           blobs remain small filled clusters inside the innermost hole or outside the root frame.
#         original solver behavior:
#           treat each one-hole component as one frame and directly canonicalize it.
#         combo solver behavior:
#           first build the raw containment tree of all fg components;
#           then detect direct frame-child pairs whose bboxes differ by exactly (2,2,2,2) and whose
#           members are both rectangular rings;
#           merge each such pair into one semantic frame, using the inner raw ring as the carrier for
#           real children; finally reuse the original canonical layout on the semantic tree.
#         key difference:
#           the raw tree contains twice as many frame components as the semantic tree, and rectangle
#           geometry makes the pairing visually obvious.
#   complexity_calibration:
#     - The change is clearly visible in every sample: all frames are double-line and perfectly straight.
#     - The pairing cue is low-entropy and repeated across the whole task: direct containment + exact
#       bbox inset of 2 on all four sides + both raw members are rectangular rings.
#     - True nested semantic frames remain distinguishable because they are placed deeper inside the
#       cavity, never at the decorative inset distance.
#   implicit_assumption_audit_main:
#     - `node.is_frame = bool(holes)` still correctly recognizes raw one-hole components.
#     - The original assumption "one raw frame component = one semantic frame" is broken and must be
#       replaced by an explicit collapse stage.
#     - The original layout code depends only on semantic bbox/center/children ordering, so it remains
#       valid after collapse.
#     - Pair detection now additionally assumes rectangular raw rings; generator guarantees this.
#     - Using the inner raw ring as the carrier is necessary because real children lie inside its hole,
#       not in the annulus between the two raw rings.
#   placement_legality:
#     - Generator always places the inner decorative ring at offset (2,2) inside the outer ring bbox.
#     - Children are placed strictly inside the inner ring hole, so they never overlap either border.
#     - Nested semantic frames are inset by at least 4 cells from the parent's outer bbox, preventing
#       accidental confusion with the decorative partner.
#     - Outside blobs are positioned adjacent to the outer semantic frame with a clear positive gap.
#   visual_observability:
#     - From a few examples, a human can see that every input frame has a two-pixel-thick rectangular
#       border while every output frame is single-line and still preserves containment and sibling order.
#     - No invisible properties such as color ids or component enumeration order are used.
#   visual_cleanliness:
#     - Only one foreground color is used, but the double-line rectangle remains legible because the two
#       raw rings are separated by a one-cell blank band.
#     - Object counts stay small and spacing is constrained so the grid does not become cluttered.
#   change_inferability:
#     - There is no color remapping; all inferable structure comes from visible geometry and enclosure.
#     - The only special correspondence the solver needs is the repeated exact inset-2 rectangle pair,
#       which is observable across samples and therefore learnable.
#   solver_generator_contract:
#     - `main()` assumes decorative frame pairs are direct parent-child frame pairs, both rectangular,
#       with bbox inset exactly (2,2,2,2).
#     - `generate_input()` enforces those conditions and rejects any sample that fails the rectangle/pair
#       metrics.


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


class _RawComp:
    def __init__(self, coords: np.ndarray, bbox: tuple[int, int, int, int], center: np.ndarray):
        self.coords = coords
        self.bbox = bbox
        self.center = center
        self.is_frame = False
        self.is_rectangle = False
        self.holes: list[np.ndarray] = []
        self.parent: _RawComp | None = None
        self.children: list[_RawComp] = []
        self.partner: _RawComp | None = None
        self.semantic: _Node | None = None
        self.carrier = False


def _component_holes(shape: tuple[int, int], coords: np.ndarray) -> list[np.ndarray]:
    mask = np.zeros(shape, dtype=bool)
    mask[tuple(coords.T)] = True
    lab, n = label_mask(~mask, connectivity=4)
    boundary = set(lab[0, :]) | set(lab[-1, :]) | set(lab[:, 0]) | set(lab[:, -1])
    return [(lab == k) for k in range(1, n + 1) if k not in boundary]


def _is_rectangular_ring(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if coords.shape[0] != 2 * (h + w) - 4:
        return False
    mask = np.zeros((h, w), dtype=bool)
    shifted = coords - np.array([r0, c0])
    mask[shifted[:, 0], shifted[:, 1]] = True
    if not np.all(mask[0, :]) or not np.all(mask[-1, :]) or not np.all(mask[:, 0]) or not np.all(mask[:, -1]):
        return False
    return not np.any(mask[1:-1, 1:-1])


def _build_raw_components(g: np.ndarray, bg: int, fg: int) -> list[_RawComp]:
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, colors=[fg], return_format="dict")
    assert comps

    raw_nodes: list[_RawComp] = []
    for comp in comps:
        coords = np.asarray(comp["coords"], dtype=int)
        raw = _RawComp(
            coords=coords,
            bbox=tuple(map(int, comp["bbox"])),
            center=coords.mean(axis=0),
        )
        raw.holes = _component_holes(g.shape, coords)
        raw.is_frame = bool(raw.holes)
        raw.is_rectangle = raw.is_frame and _is_rectangular_ring(coords, raw.bbox)
        raw_nodes.append(raw)

    assert any(raw.is_frame for raw in raw_nodes)
    for node_i, node in enumerate(raw_nodes):
        containers = []
        for frame_i, frame in enumerate(raw_nodes):
            if node_i == frame_i or not frame.is_frame:
                continue
            if any(np.all(hole[tuple(node.coords.T)]) for hole in frame.holes):
                containers.append(frame)
        if containers:
            node.parent = min(containers, key=lambda item: int(item.coords.shape[0]))
            node.parent.children.append(node)
    return raw_nodes


def _bbox_inset(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return ir0 - or0, ic0 - oc0, or1 - ir1, oc1 - ic1


def _is_decorative_pair(outer: _RawComp, inner: _RawComp) -> bool:
    if not (outer.is_frame and inner.is_frame and inner.parent is outer):
        return False
    if not (outer.is_rectangle and inner.is_rectangle):
        return False
    return _bbox_inset(outer.bbox, inner.bbox) == (2, 2, 2, 2)


def _raw_components_to_semantic(raw_nodes: list[_RawComp]) -> list[_Node]:
    for raw in raw_nodes:
        raw.partner = None
        raw.semantic = None
        raw.carrier = False

    for outer in raw_nodes:
        if not outer.is_frame:
            continue
        candidates = [child for child in outer.children if _is_decorative_pair(outer, child)]
        assert len(candidates) <= 1
        if candidates:
            inner = candidates[0]
            outer.partner = inner
            inner.partner = outer

    semantic_nodes: list[_Node] = []
    for raw in raw_nodes:
        if raw.partner is not None and raw.parent is raw.partner:
            continue

        if raw.partner is not None:
            coords = np.vstack([raw.coords, raw.partner.coords])
            node = _Node(coords=coords, bbox=raw.bbox, center=coords.mean(axis=0))
            node.is_frame = True
            raw.semantic = node
            raw.carrier = False
            raw.partner.semantic = node
            raw.partner.carrier = True
        else:
            node = _Node(coords=raw.coords, bbox=raw.bbox, center=raw.center)
            node.is_frame = raw.is_frame
            raw.semantic = node
            raw.carrier = raw.is_frame
        semantic_nodes.append(node)

    for raw in raw_nodes:
        if raw.partner is not None and raw.parent is raw.partner:
            continue
        assert raw.semantic is not None

    for raw in raw_nodes:
        if not raw.carrier:
            continue
        parent_sem = raw.semantic
        assert parent_sem is not None
        for child in raw.children:
            child_sem = child.semantic
            assert child_sem is not None
            if child_sem is parent_sem:
                continue
            if child_sem.parent is None:
                child_sem.parent = parent_sem
                parent_sem.children.append(child_sem)

    return [node for node in semantic_nodes if node.parent is None]


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
    Convert a double-line rectangular frame-and-blob drawing into a clean canonical frame-and-dot diagram.

    Input structure:
    - The grid uses exactly one background color and one foreground color.
    - Foreground objects are small filled blobs or perfect rectangular rings.
    - Each semantic frame is drawn as two directly nested rectangular rings with a one-cell gap between them.
    - Real children lie inside the inner ring; an optional blob may also sit outside the outermost frame.

    Rule:
    - Build the raw containment tree from frame holes.
    - Detect decorative rectangle pairs: direct frame-child pairs whose bounding boxes are inset by exactly two cells on every side.
    - Collapse each such pair into one semantic frame, keeping the inner raw ring as the carrier of the actual children.
    - Canonicalize blobs to single pixels and semantic frames to 1px rectangular borders padded two cells around their direct children, while preserving sibling row/column order and outside-side relations.

    Key insight:
    The apparent "two nested rectangles" at one location are not two semantic levels; they are one thick rectangular frame encoded by a repeated inset-2 rectangle pair.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    fg_colors = sorted(set(map(int, g.flatten())) - {bg})
    assert len(fg_colors) == 1
    fg = int(fg_colors[0])

    raw_nodes = _build_raw_components(g, bg=bg, fg=fg)
    roots = _raw_components_to_semantic(raw_nodes)
    assert roots and any(node.is_frame for node in roots)
    mask = _layout_root(roots)
    out = np.full(mask.shape, bg, dtype=int)
    out[mask] = fg
    return out


# --------------- generator (o5x) ---------------

# Grid / canvas bounds
GRID_MIN = 16
GRID_MAX = 25
MAX_EXTRA_PAD = 12
MAX_PAD_PER_SIDE = 6

# Raw rectangle geometry:
# semantic frame = outer rectangular ring + inner rectangular ring inset by 2 cells
FRAME_INNER_H_RANGE = (7, 19)
FRAME_INNER_W_RANGE = (7, 21)
FRAME_INNER_H_COMPACT_RANGE = (7, 10)
FRAME_INNER_W_COMPACT_RANGE = (7, 12)

# Blob geometry
BLOB_SIZE_RANGE = (4, 11)
BLOB_BBOX_RANGE = (2, 5)

# Layout / arrangement
ARRANGE_GAP_COMPACT = (1, 2)
ARRANGE_GAP_NORMAL = (2, 4)
ARRANGE_EXTRA_SPAN = (0, 3)
FRAME_MARGIN_RANGE = (1, 3)
COMPACT_MARGIN = 1

# Outside blob placement
OUTSIDE_GAP_RANGE = (2, 5)

# Tree structure coverage
STRUCTURE_MODES = (
    ("shallow", False),
    ("shallow", True),
    ("nested", False),
    ("nested", True),
)
_STRUCTURE_MODE_CURSOR = 0
DEPTH1_INSIDE_RANGE = (1, 3)
DEPTH2_INNER_BLOBS = (1, 2)
DEPTH2_OUTER_BLOBS = (0, 2)
MAX_TOTAL_CHILDREN = 4

# Generation retry budget
GEN_RETRIES = 400


class _Obj:
    def __init__(self, kind, children=None, outside_side=None):
        self.kind = kind
        self.children = list(children or [])
        self.mask: np.ndarray | None = None
        self.outer_mask: np.ndarray | None = None
        self.inner_mask: np.ndarray | None = None
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


def _frame_mask(h: int, w: int) -> np.ndarray | None:
    if h < 3 or w < 3:
        return None
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask if _loop_ok(mask) else None


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


def _arrange(children: list[_Obj], compact: bool = False) -> tuple[dict[_Obj, tuple[int, int]], int, int]:
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


def _prepare(node: _Obj, compact: bool = False) -> bool:
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

    h_range = FRAME_INNER_H_COMPACT_RANGE if compact else FRAME_INNER_H_RANGE
    w_range = FRAME_INNER_W_COMPACT_RANGE if compact else FRAME_INNER_W_RANGE

    min_inner_h = max(h_range[0], group_h + 2 * margin + 2)
    min_inner_w = max(w_range[0], group_w + 2 * margin + 2)
    if min_inner_h > h_range[1] or min_inner_w > w_range[1]:
        return False

    h_hi = min(h_range[1], min_inner_h + (1 if tight else 3))
    w_hi = min(w_range[1], min_inner_w + (2 if tight else 4))
    inner_h = random.randint(min_inner_h, h_hi)
    inner_w = random.randint(min_inner_w, w_hi)

    avail_h = inner_h - 2
    avail_w = inner_w - 2
    top = 1 + margin + random.randint(0, avail_h - group_h - 2 * margin)
    left = 1 + margin + random.randint(0, avail_w - group_w - 2 * margin)
    for child, (r, c) in placements.items():
        child.rel = (2 + top + r, 2 + left + c)

    outer_h = inner_h + 4
    outer_w = inner_w + 4
    outer_mask = _frame_mask(outer_h, outer_w)
    inner_mask = _frame_mask(inner_h, inner_w)
    if outer_mask is None or inner_mask is None:
        return False

    combined = np.zeros((outer_h, outer_w), dtype=bool)
    combined |= outer_mask
    combined[2 : 2 + inner_h, 2 : 2 + inner_w] |= inner_mask
    node.mask = combined
    node.outer_mask = outer_mask
    node.inner_mask = inner_mask
    return True


def _walk(node: _Obj, base=(0, 0)) -> list[_Obj]:
    br, bc = base
    rr, cc = node.rel
    node.pos = (br + rr, bc + cc)
    out = [node]
    for child in node.children:
        out.extend(_walk(child, node.pos))
    return out


def _place_outside(root: _Obj, outside: _Obj) -> None:
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


def _make_tree(mode: tuple[str, bool] | None = None) -> tuple[_Obj, _Obj | None]:
    root = _Obj("frame")
    outside = None
    sides = ("left", "right", "top", "bottom")

    depth_mode, want_outside = mode if mode is not None else random.choice(STRUCTURE_MODES)
    if depth_mode == "shallow":
        n_inside = random.randint(*DEPTH1_INSIDE_RANGE)
        root.children = [_Obj("blob") for _ in range(n_inside)]
        if want_outside:
            outside = _Obj("blob", outside_side=random.choice(sides))
    else:
        feasible = [
            (n_inner, n_outer)
            for n_inner in range(DEPTH2_INNER_BLOBS[0], DEPTH2_INNER_BLOBS[1] + 1)
            for n_outer in range(DEPTH2_OUTER_BLOBS[0], DEPTH2_OUTER_BLOBS[1] + 1)
            if (n_inner + n_outer < MAX_TOTAL_CHILDREN) or not want_outside
        ]
        n_inner, n_outer = random.choice(feasible)
        inner = _Obj("frame")
        inner.children = [_Obj("blob") for _ in range(n_inner)]
        outer_blobs = [_Obj("blob") for _ in range(n_outer)]
        root.children = [inner] + outer_blobs
        if want_outside:
            outside = _Obj("blob", outside_side=random.choice(sides))
    return root, outside


def _collect_renderables(node: _Obj, out: list[tuple[tuple[int, int], np.ndarray]]) -> None:
    if node.kind == "blob":
        assert node.mask is not None
        out.append((node.pos, node.mask))
        return
    assert node.outer_mask is not None and node.inner_mask is not None
    out.append((node.pos, node.outer_mask))
    out.append(((node.pos[0] + 2, node.pos[1] + 2), node.inner_mask))


def _draw(objects: list[_Obj], bg: int, fg: int) -> np.ndarray | None:
    renderables: list[tuple[tuple[int, int], np.ndarray]] = []
    for obj in objects:
        _collect_renderables(obj, renderables)

    min_r = min(pos[0] for pos, _ in renderables)
    min_c = min(pos[1] for pos, _ in renderables)
    max_r = max(pos[0] + mask.shape[0] for pos, mask in renderables)
    max_c = max(pos[1] + mask.shape[1] for pos, mask in renderables)
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
        obj.pos = (obj.pos[0] + shift_r, obj.pos[1] + shift_c)

    renderables = []
    for obj in objects:
        _collect_renderables(obj, renderables)
    for (r, c), mask in renderables:
        patch = np.full(mask.shape, bg, dtype=int)
        patch[mask] = fg
        paste(grid, patch, r, c, transparent=bg)

    if np.count_nonzero(grid == bg) <= np.count_nonzero(grid == fg):
        return None
    return grid


def _fusion_metrics(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    fg_colors = sorted(set(map(int, g.flatten())) - {bg})
    assert len(fg_colors) == 1
    fg = int(fg_colors[0])

    raw_nodes = _build_raw_components(g, bg=bg, fg=fg)
    raw_frame_count = sum(raw.is_frame for raw in raw_nodes)
    rectangular_frame_count = sum(raw.is_rectangle for raw in raw_nodes)
    decorative_pair_count = sum(
        1
        for raw in raw_nodes
        if any(_is_decorative_pair(raw, child) for child in raw.children)
    )
    inset_signatures = sorted(
        {
            _bbox_inset(raw.bbox, child.bbox)
            for raw in raw_nodes
            for child in raw.children
            if _is_decorative_pair(raw, child)
        }
    )
    return {
        "raw_frame_count": int(raw_frame_count),
        "rectangular_frame_count": int(rectangular_frame_count),
        "decorative_pair_count": int(decorative_pair_count),
        "inset_signatures": inset_signatures,
    }


def _gen_attempt(mode: tuple[str, bool] | None = None) -> np.ndarray | None:
    bg, fg = random.sample(range(10), 2)
    root, outside = _make_tree(mode=mode)
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

    semantic_frame_count = sum(obj.kind == "frame" for obj in objects)
    metrics = _fusion_metrics(grid)
    if metrics["raw_frame_count"] != 2 * semantic_frame_count:
        return None
    if metrics["rectangular_frame_count"] != metrics["raw_frame_count"]:
        return None
    if metrics["decorative_pair_count"] != semantic_frame_count:
        return None
    if metrics["inset_signatures"] != [(2, 2, 2, 2)]:
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
    global _STRUCTURE_MODE_CURSOR
    mode = STRUCTURE_MODES[_STRUCTURE_MODE_CURSOR % len(STRUCTURE_MODES)]
    _STRUCTURE_MODE_CURSOR += 1
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(mode=mode)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 2d0172a1 combo input")
