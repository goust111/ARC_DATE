# Combo variant of 2d0172a1 (index 21)
# combo_spec: D5/2+D7/1+D17/0
# Name: 距离排序+保留框架宽高比+色块保留尺寸

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import find_components, get_bg, label_mask, paste


# Design analysis
# a) concretization and behavior trace:
#    - Base task: detect the containment tree of rough foreground components, then redraw each frame as a
#      clean rectangle and each blob as a canonical mark.
#    - Combo change 1 (D5/2): blobs no longer collapse to 1x1. If an input blob has bbox 3x4, its output
#      object is a filled 3x4 block.
#    - Combo change 2 (D7/1): direct siblings inside the same frame are no longer positioned by input
#      row/column bands. They are ranked by centroid distance to the parent frame center, near to far,
#      then displayed as a down-right diagonal chain.
#    - Combo change 3 (D17/0): a frame still becomes a rectangle, but its output rectangle must inherit the
#      input frame bbox aspect ratio. Concretely, if the direct-child canonical layout needs at least 9x7
#      cells and the input rough frame bbox reduces to ratio 2:3, the output frame becomes the smallest
#      integer 2:3 rectangle that fits 9x7, namely 10x15.
#    - Small trace:
#      old: two blobs with bboxes 2x3 and 3x3 inside a tall rough frame would become two dots, sorted by
#      band order, and the frame size would be child-layout-plus-4 only.
#      new: those blobs become a 2x3 block and a 3x3 block; whichever blob is closer to the frame center
#      becomes the first diagonal child; the enclosing rectangle is then expanded to the smallest scaled
#      copy of the input frame ratio that can still hold the diagonal child layout with a 1-cell moat.
# b) complexity calibration:
#    - Blob sizes already vary in the source task, so preserving bbox size produces visible non-dot outputs
#      without adding colors.
#    - Distance sorting is only informative when a frame has at least two children and the distance order is
#      not the same as both row order and column order. The generator enforces this.
#    - Aspect-ratio preservation is weak if all frames are nearly square or if the minimal child box already
#      matches the frame ratio. The generator samples from a small non-square ratio family and rejects cases
#      where the preserved-ratio frame collapses back to the original child-driven size.
# c) implicit assumption audit:
#    - Original `_build_layout` assumed `blob -> 1x1`; invalid now, so blob masks derive from measured bbox.
#    - Original `_build_layout(frame)` assumed output frame size is exactly `(child_h+4, child_w+4)` and the
#      child inset is fixed at 2. Invalid now: the frame may gain extra slack from aspect-ratio scaling, so
#      the inset must be recomputed and `axis_map` must shift by that recomputed inset.
#    - Original `_layout_group` assumed sibling placement is recoverable from row/column overlap bands.
#      Invalid now: sibling order depends only on radial distance, so layout becomes a diagonal rank display.
#    - Original root outside-blob logic reserved a fixed 3x3 slot around a dot. Invalid now because outside
#      blobs keep their bbox-sized rectangles and need full extents.
# d) placement legality:
#    - Canonical children are laid out as non-overlapping rectangles with a 1-cell gap between consecutive
#      diagonal items.
#    - Generator-side input placements use rectangle packing with at least one background cell of clearance.
#    - Chosen frame bboxes always satisfy `frame bbox >= child group bbox + 2*margin`, and `_draw` rejects
#      any scene that exceeds the 16..25 canvas bound.
# e) visual observability:
#    - All three controls are human-visible: blob block sizes, frame width/height bias, and "near center
#      first" sibling ranking. No color-id arithmetic or hidden metadata is used.
#    - The generator centers child groups inside frames so radial order relative to the parent frame is
#      readable from a few examples.
# f) cleanliness:
#    - Still only one foreground color. Structure is separated by containment, rectangular borders, and
#      1-cell gaps, so the outputs remain legible despite the extra metric information.
# g) inferability:
#    - No color remapping is introduced. The only inference burden is geometric: measure blob bbox size,
#      compare sibling distances to the frame center, and preserve frame aspect ratio when redrawing.
# new solver assumptions and generator support:
#    - Every frame has at least one child; tree construction preserves the source task's non-empty frames.
#    - Sortable sibling sets need distinct enough distances; generator retries until at least one frame
#      exhibits a radial order that disagrees with plain row/column order.
#    - Aspect-ratio preservation must matter on at least one frame; generator rejects trees where every
#      frame's scaled rectangle equals the minimal child-driven rectangle.


class _Node:
    def __init__(self, coords: np.ndarray, bbox: tuple[int, int, int, int], center: np.ndarray):
        self.coords = coords
        self.bbox = bbox
        self.center = center
        self.rect_shape = (int(bbox[2] - bbox[0] + 1), int(bbox[3] - bbox[1] + 1))
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


def _distance_key(node: _Node, parent_center: np.ndarray) -> tuple[float, float, float]:
    dr = float(node.center[0] - parent_center[0])
    dc = float(node.center[1] - parent_center[1])
    return (dr * dr + dc * dc, float(node.center[0]), float(node.center[1]))


def _build_layout(node: _Node) -> None:
    if node.mask is not None:
        return

    if not node.is_frame:
        h, w = node.rect_shape
        node.mask = np.ones((h, w), dtype=bool)
        node.anchor = (h // 2, w // 2)
        node.axis_map = {
            "row": [(float(node.center[0]), int(node.anchor[0]))],
            "col": [(float(node.center[1]), int(node.anchor[1]))],
        }
        return

    assert node.children
    child_mask, _placements, axis_map = _layout_group(node.children, parent_center=node.center)
    h, w = child_mask.shape
    min_h = h + 4
    min_w = w + 4
    out_h, out_w = _scaled_frame_shape(node.bbox, min_h, min_w)
    inset_r = 2 + (out_h - min_h) // 2
    inset_c = 2 + (out_w - min_w) // 2

    out = np.zeros((out_h, out_w), dtype=bool)
    out[0, :] = True
    out[-1, :] = True
    out[:, 0] = True
    out[:, -1] = True
    out[inset_r : inset_r + h, inset_c : inset_c + w] |= child_mask

    node.mask = out
    node.anchor = (out_h // 2, out_w // 2)
    node.axis_map = {
        "row": [(orig, pos + inset_r) for orig, pos in axis_map["row"]],
        "col": [(orig, pos + inset_c) for orig, pos in axis_map["col"]],
    }


def _layout_group(
    nodes: list[_Node], parent_center: np.ndarray
) -> tuple[np.ndarray, dict[_Node, tuple[int, int]], dict[str, list[tuple[float, int]]]]:
    assert nodes
    for node in nodes:
        _build_layout(node)

    ordered = sorted(nodes, key=lambda item: _distance_key(item, parent_center))
    placements: dict[_Node, tuple[int, int]] = {}
    axis_map = {"row": [], "col": []}
    r = c = 0
    out_h = out_w = 0
    for i, node in enumerate(ordered):
        assert node.mask is not None and node.anchor is not None
        placements[node] = (int(r), int(c))
        axis_map["row"].append((float(node.center[0]), int(r + node.anchor[0])))
        axis_map["col"].append((float(node.center[1]), int(c + node.anchor[1])))
        out_h = max(out_h, r + node.mask.shape[0])
        out_w = max(out_w, c + node.mask.shape[1])
        if i < len(ordered) - 1:
            r += node.mask.shape[0] + 1
            c += node.mask.shape[1] + 1

    out = np.zeros((out_h, out_w), dtype=bool)
    for node, (r, c) in placements.items():
        assert node.mask is not None
        out[r : r + node.mask.shape[0], c : c + node.mask.shape[1]] |= node.mask
    return out, placements, axis_map


def _outside_side(blob: _Node, frame: _Node) -> str:
    r0, c0, r1, c1 = blob.bbox
    fr0, fc0, fr1, fc1 = frame.bbox
    if c0 > fc1:
        return "right"
    if c1 < fc0:
        return "left"
    if r0 > fr1:
        return "bottom"
    if r1 < fr0:
        return "top"

    dr = float(blob.center[0] - (fr0 + fr1) / 2)
    dc = float(blob.center[1] - (fc0 + fc1) / 2)
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
    blobs = [node for node in nodes if not node.is_frame]
    if len(frames) == 1 and blobs:
        frame = frames[0]
        assert frame.mask is not None and frame.anchor is not None
        placements: dict[_Node, tuple[int, int]] = {frame: (0, 0)}
        extents = [(0, 0, frame.mask.shape[0], frame.mask.shape[1])]

        for blob in blobs:
            assert blob.mask is not None and blob.anchor is not None
            bh, bw = blob.mask.shape
            side = _outside_side(blob, frame)
            if side in {"right", "left"}:
                r = _map_frame_axis(frame, 0, float(blob.center[0])) - blob.anchor[0]
                c = frame.mask.shape[1] + 1 if side == "right" else -bw - 1
            else:
                r = frame.mask.shape[0] + 1 if side == "bottom" else -bh - 1
                c = _map_frame_axis(frame, 1, float(blob.center[1])) - blob.anchor[1]
            placements[blob] = (int(r), int(c))
            extents.append((int(r), int(c), int(r + bh), int(c + bw)))

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

    parent_center = np.mean(np.stack([node.center for node in nodes]), axis=0)
    mask, _placements, _axis_map = _layout_group(nodes, parent_center=parent_center)
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert a rough nested frame-and-blob sketch into a clean rectangle-and-block diagram that keeps more metric information.

    Input structure:
    - The grid uses exactly one background color and one foreground color.
    - Foreground components are rough closed frames with one hole or compact filled blobs.
    - Frames may contain other frames/blobs, and at most one blob may sit outside the outer frame.

    Rule:
    - Reconstruct the containment tree by checking which components lie inside each frame hole.
    - Replace each blob with a filled rectangle whose height and width match the blob's input bbox.
    - Replace each frame with a 1-pixel rectangular border whose height/width ratio matches the input frame bbox, scaled up just enough to contain the canonical layout of its direct children with a 1-cell moat.
    - Inside each frame, sort direct children by increasing centroid distance to the frame center and place them as a down-right diagonal chain with one background cell between consecutive canonical objects.

    Key insight: the sketchy contours are disposable, but three geometric measurements survive the cleanup: blob bbox size, child distance rank relative to each frame center, and each frame's bbox aspect ratio.
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


# --------------- generator (o5x) ---------------

# ── CFG: tunable generation parameters ──────────────────────────────────────
# Grid / canvas bounds
GRID_MIN = 16
GRID_MAX = 25
MAX_EXTRA_PAD = 12
MAX_PAD_PER_SIDE = 6

# Frame geometry
FRAME_H_RANGE = (7, 20)
FRAME_W_RANGE = (7, 24)
FRAME_CORNER_MAX = 5
FRAME_MIN_EDGE = 2
FRAME_TURN_RATIO = (0.34, 0.43)
FRAME_RETRIES = 60
FRAME_RATIO_UNITS = ((2, 3), (3, 2), (3, 4), (4, 3))
FRAME_SLACK_CAP = 10
COMPACT_FRAME_SLACK_CAP = 6

# Blob geometry
BLOB_SIZE_RANGE = (4, 11)
BLOB_BBOX_RANGE = (2, 5)

# Layout / arrangement
ARRANGE_GAP_COMPACT = (2, 3)
ARRANGE_GAP_NORMAL = (2, 5)
ARRANGE_EXTRA_SPAN = (0, 4)
FRAME_MARGIN_RANGE = (3, 5)
COMPACT_MARGIN = 3
ARRANGE_RETRIES = 180
MIN_DIST_DELTA = 2.0

# Outside blob placement
OUTSIDE_GAP_RANGE = (2, 5)

# Tree structure probabilities
DEPTH1_PROB = 0.35
OUTSIDE_BLOB_PROB = 0.35
DEPTH1_INSIDE_RANGE = (2, 3)
DEPTH2_INNER_BLOBS = (2, 2)
DEPTH2_OUTER_BLOBS = (0, 0)
MAX_TOTAL_CHILDREN = 4

# Step-mix alternation bias
STEP_MIX_ALT_PROB = 0.7

# Generation retry budget
GEN_RETRIES = 400
# ────────────────────────────────────────────────────────────────────────────

_NEXT_BRANCH_MODE = 0


class _Obj:
    def __init__(self, kind, children=None, outside_side=None):
        self.kind = kind
        self.children = list(children or [])
        self.mask = None
        self.rel = (0, 0)
        self.pos = (0, 0)
        self.outside_side = outside_side


def _has_hole(mask: np.ndarray) -> bool:
    lab, n = label_mask(~mask, connectivity=4)
    boundary = set(lab[0, :]) | set(lab[-1, :]) | set(lab[:, 0]) | set(lab[:, -1])
    return any(k not in boundary for k in range(1, n + 1))


def _loop_ok(mask):
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


def _step_mix(n_a, step_a, n_b, step_b):
    """Interleave n_a copies of step_a with n_b copies of step_b, biased toward alternation."""
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
    """Extend path with a straight line from its last point to end."""
    r, c = path[-1]
    er, ec = end
    dr = 0 if er == r else (1 if er > r else -1)
    dc = 0 if ec == c else (1 if ec > c else -1)
    while (r, c) != (er, ec):
        r += dr
        c += dc
        path.append((r, c))


def _add_steps(path, steps):
    """Extend path by applying a sequence of (dr, dc) deltas."""
    r, c = path[-1]
    for dr, dc in steps:
        r += dr
        c += dc
        path.append((r, c))


def _turn_ratio(path):
    """Fraction of direction changes along a cyclic path."""
    dirs = []
    cyc = path + [path[0]]
    for (r0, c0), (r1, c1) in zip(cyc, cyc[1:]):
        dirs.append((r1 - r0, c1 - c0))
    turns = sum(1 for a, b in zip(dirs, dirs[1:] + [dirs[0]]) if a != b)
    return turns / len(path)


def _frame_mask(h, w):
    """Generate a wobbly simple closed curve mask of size (h, w)."""
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
    """Generate a small filled blob mask (size 4-11, bbox 2x2 to 4x4)."""
    for _ in range(80):
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
        if not _has_hole(mask):
            return mask
    return np.ones((2, 2), dtype=bool)


def _rects_overlap(a, b, pad=1):
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0)


def _distance_orders(children, placements, group_h, group_w):
    center = np.array([(group_h - 1) / 2, (group_w - 1) / 2], dtype=float)
    child_data = []
    for idx, child in enumerate(children):
        r, c = placements[child]
        h, w = child.mask.shape
        child_center = np.array([r + (h - 1) / 2, c + (w - 1) / 2], dtype=float)
        dr = float(child_center[0] - center[0])
        dc = float(child_center[1] - center[1])
        child_data.append((idx, child_center[0], child_center[1], dr * dr + dc * dc))
    dist_values = sorted(item[3] for item in child_data)
    dist_order = tuple(item[0] for item in sorted(child_data, key=lambda item: (item[3], item[1], item[2])))
    row_order = tuple(item[0] for item in sorted(child_data, key=lambda item: (item[1], item[2])))
    col_order = tuple(item[0] for item in sorted(child_data, key=lambda item: (item[2], item[1])))
    distinct = all(b - a >= MIN_DIST_DELTA for a, b in zip(dist_values, dist_values[1:]))
    return dist_order, row_order, col_order, distinct


def _arrange(children, compact=False):
    """Arrange children so distance order from the frame center is visible and non-band-based."""
    dims = [child.mask.shape for child in children]
    k = len(children)
    if k == 1:
        return {children[0]: (0, 0)}, dims[0][0], dims[0][1]

    if compact:
        max_h = max(h for h, _ in dims)
        max_w = max(w for _, w in dims)
        sum_h = sum(h for h, _ in dims)
        sum_w = sum(w for _, w in dims)
        lo_h = max(max_h + 2, (sum_h + 1) // 2 + 1)
        hi_h = min(sum_h + 4, max_h + 8)
        lo_w = max(max_w + 2, (sum_w + 1) // 2 + 1)
        hi_w = min(sum_w + 4, max_w + 8)
        if lo_h > hi_h or lo_w > hi_w:
            return None

        for _ in range(ARRANGE_RETRIES * 2):
            group_h = random.randint(lo_h, hi_h)
            group_w = random.randint(lo_w, hi_w)
            order = sorted(children, key=lambda child: child.mask.shape[0] * child.mask.shape[1], reverse=True)
            placements = {}
            rects = []
            for child in order:
                h, w = child.mask.shape
                placed = False
                for _ in range(100):
                    r = random.randint(0, group_h - h)
                    c = random.randint(0, group_w - w)
                    rect = (r, c, r + h - 1, c + w - 1)
                    if any(_rects_overlap(rect, other, pad=1) for other in rects):
                        continue
                    placements[child] = (r, c)
                    rects.append(rect)
                    placed = True
                    break
                if not placed:
                    break
            if len(placements) != k:
                continue

            dist_order, row_order, col_order, distinct = _distance_orders(children, placements, group_h, group_w)
            if distinct and (dist_order != row_order or dist_order != col_order):
                return placements, group_h, group_w

        return None

    gap_lo, gap_hi = ARRANGE_GAP_COMPACT if compact else ARRANGE_GAP_NORMAL
    max_h = max(h for h, _ in dims)
    max_w = max(w for _, w in dims)
    sum_h = sum(h for h, _ in dims)
    sum_w = sum(w for _, w in dims)

    for _ in range(ARRANGE_RETRIES):
        mode = random.choice(("vertical", "horizontal", "diagonal", "scatter"))
        gaps = [random.randint(gap_lo, gap_hi) for _ in range(k - 1)]
        if mode == "vertical":
            group_h = sum_h + sum(gaps)
            group_w = max_w + gap_hi + 1 + random.randint(*ARRANGE_EXTRA_SPAN)
        elif mode == "horizontal":
            group_h = max_h + gap_hi + 1 + random.randint(*ARRANGE_EXTRA_SPAN)
            group_w = sum_w + sum(gaps)
        elif mode == "diagonal":
            group_h = sum_h + sum(gaps)
            group_w = sum_w + sum(gaps)
        else:
            group_h = max(max_h + gap_hi + 2, (sum_h + sum(gaps) + 1) // 2 + gap_hi + 1)
            group_w = max(max_w + gap_hi + 2, (sum_w + sum(gaps) + 1) // 2 + gap_hi + 1)

        order = sorted(children, key=lambda child: child.mask.shape[0] * child.mask.shape[1], reverse=True)
        placements = {}
        rects = []
        for child in order:
            h, w = child.mask.shape
            placed = False
            for _ in range(80):
                r = random.randint(0, group_h - h)
                c = random.randint(0, group_w - w)
                rect = (r, c, r + h - 1, c + w - 1)
                if any(_rects_overlap(rect, other, pad=1) for other in rects):
                    continue
                placements[child] = (r, c)
                rects.append(rect)
                placed = True
                break
            if not placed:
                break
        if len(placements) != k:
            continue

        dist_order, row_order, col_order, distinct = _distance_orders(children, placements, group_h, group_w)
        if distinct and (dist_order != row_order or dist_order != col_order):
            return placements, group_h, group_w

    return None


def _pick_frame_size(min_h: int, min_w: int, group_h: int, group_w: int, compact: bool) -> tuple[int, int] | None:
    child_aspect = group_w / group_h
    slack_cap = COMPACT_FRAME_SLACK_CAP if compact else FRAME_SLACK_CAP
    candidates: list[tuple[float, int, int]] = []
    for unit_h, unit_w in FRAME_RATIO_UNITS:
        q_min = max(_ceil_div(min_h, unit_h), _ceil_div(min_w, unit_w))
        q_max = min(FRAME_H_RANGE[1] // unit_h, FRAME_W_RANGE[1] // unit_w)
        if q_min > q_max:
            continue
        for q in range(q_min, q_max + 1):
            h = unit_h * q
            w = unit_w * q
            slack_h = h - min_h
            slack_w = w - min_w
            if slack_h + slack_w > slack_cap:
                continue
            frame_aspect = w / h
            mismatch = abs(math.log(frame_aspect / child_aspect))
            one_axis_slack = int((slack_h == 0) != (slack_w == 0))
            score = mismatch + 0.12 * min(slack_h + slack_w, 6) + 0.2 * one_axis_slack
            score += random.random() * 0.03
            candidates.append((score, h, w))
    if not candidates:
        return None
    candidates.sort()
    top = candidates[: min(8, len(candidates))]
    _, h, w = random.choice(top)
    return h, w


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
    arranged = _arrange(node.children, compact=tight)
    if arranged is None:
        return False
    placements, group_h, group_w = arranged
    margin = COMPACT_MARGIN if tight else random.randint(*FRAME_MARGIN_RANGE)
    min_h = max(FRAME_H_RANGE[0], group_h + 2 * margin)
    min_w = max(FRAME_W_RANGE[0], group_w + 2 * margin)
    if min_h > FRAME_H_RANGE[1] or min_w > FRAME_W_RANGE[1]:
        return False
    picked = _pick_frame_size(min_h, min_w, group_h, group_w, compact=tight)
    if picked is None:
        return False
    h, w = picked
    top = margin + (h - 2 * margin - group_h) // 2
    left = margin + (w - 2 * margin - group_w) // 2
    for child, (r, c) in placements.items():
        child.rel = (top + r, left + c)
    node.mask = _frame_mask(h, w)
    return node.mask is not None


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


def _build_generated_roots(root, outside):
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
    return roots


def _expected_output(root, outside, bg, fg):
    """Compute expected solver output for the generated tree."""
    roots = _build_generated_roots(root, outside)
    mask = _layout_root(roots)
    out = np.full(mask.shape, bg, dtype=int)
    out[mask] = fg
    return out


def _tree_has_sortable_siblings(node):
    return (node.kind == "frame" and len(node.children) >= 2) or any(
        _tree_has_sortable_siblings(child) for child in node.children if child.kind == "frame"
    )


def _distance_visible(frame: _Node) -> bool:
    if len(frame.children) >= 2:
        child_data = []
        for idx, child in enumerate(frame.children):
            dr = float(child.center[0] - frame.center[0])
            dc = float(child.center[1] - frame.center[1])
            child_data.append((idx, float(child.center[0]), float(child.center[1]), dr * dr + dc * dc))
        dist_order = tuple(item[0] for item in sorted(child_data, key=lambda item: (item[3], item[1], item[2])))
        row_order = tuple(item[0] for item in sorted(child_data, key=lambda item: (item[1], item[2])))
        col_order = tuple(item[0] for item in sorted(child_data, key=lambda item: (item[2], item[1])))
        if dist_order != row_order or dist_order != col_order:
            return True
    return any(_distance_visible(child) for child in frame.children if child.is_frame)


def _aspect_visible(frame: _Node) -> bool:
    assert frame.children
    child_mask, _placements, _axis_map = _layout_group(frame.children, parent_center=frame.center)
    min_shape = (child_mask.shape[0] + 4, child_mask.shape[1] + 4)
    scaled_shape = _scaled_frame_shape(frame.bbox, min_shape[0], min_shape[1])
    if scaled_shape != min_shape:
        return True
    return any(_aspect_visible(child) for child in frame.children if child.is_frame)


def measure_variant_features(grid: np.ndarray) -> dict[str, int]:
    """Quantify whether a grid instance exhibits the combo's radial-sort and aspect-ratio signals."""
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    fg_colors = sorted(set(map(int, g.flatten())) - {bg})
    assert len(fg_colors) == 1
    fg = int(fg_colors[0])
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, colors=[fg], return_format="dict")
    nodes: list[_Node] = []
    holes: list[list[np.ndarray]] = []
    for comp in comps:
        coords = np.asarray(comp["coords"], dtype=int)
        node = _Node(coords=coords, bbox=tuple(map(int, comp["bbox"])), center=coords.mean(axis=0))
        node.is_frame = bool(_component_holes(g.shape, coords))
        nodes.append(node)
        holes.append(_component_holes(g.shape, coords))
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
    frames = [node for node in nodes if node.is_frame]
    return {
        "frame_count": len(frames),
        "blob_count": sum(1 for node in nodes if not node.is_frame),
        "distance_visible": int(any(_distance_visible(frame) for frame in frames)),
        "aspect_visible": int(any(_aspect_visible(frame) for frame in frames)),
        "large_blob_count": sum(1 for node in nodes if not node.is_frame and node.rect_shape != (1, 1)),
    }


def _variant_visible_in_tree(root, outside):
    roots = _build_generated_roots(root, outside)
    frames = [node for node in roots if node.is_frame]
    distance_visible = any(_distance_visible(frame) for frame in frames)
    aspect_visible = any(_aspect_visible(frame) for frame in frames)
    return {"distance_visible": distance_visible, "aspect_visible": aspect_visible}


def _make_tree(mode=None):
    """Build a random containment tree with at least one sortable sibling set."""
    sides = ("left", "right", "top", "bottom")
    while True:
        root = _Obj("frame")
        outside = None
        if mode is None:
            mode = "depth1" if random.random() < DEPTH1_PROB else "depth2"
        if mode == "depth1":
            n_inside = random.randint(*DEPTH1_INSIDE_RANGE)
            root.children = [_Obj("blob") for _ in range(n_inside)]
            if n_inside < MAX_TOTAL_CHILDREN and random.random() < OUTSIDE_BLOB_PROB:
                outside = _Obj("blob", outside_side=random.choice(sides))
        else:
            inner = _Obj("frame")
            inner.children = [_Obj("blob") for _ in range(random.randint(*DEPTH2_INNER_BLOBS))]
            root.children = [inner]
        if _tree_has_sortable_siblings(root):
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


def _gen_attempt(mode=None):
    """Single generation attempt: build tree, render input, verify against solver."""
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
    visibility = _variant_visible_in_tree(root, outside)
    if not visibility["distance_visible"] or not visibility["aspect_visible"]:
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
    global _NEXT_BRANCH_MODE
    mode = "depth1" if _NEXT_BRANCH_MODE % 2 == 0 else "depth2"
    _NEXT_BRANCH_MODE += 1
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(mode=mode)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid combo 2d0172a1 input for {mode}")
