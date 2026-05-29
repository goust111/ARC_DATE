# Combo variant of 2d0172a1 (index 3)
# combo_spec: D6/2+D12/0
# Name: 多色层级+颜色编码父子关系

from __future__ import annotations

import random

import numpy as np

from arc_utils import find_components, get_bg, label_mask, paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concretization and behavior trace
# - Original 2d0172a1 used one foreground color. A solver had to inspect holes:
#   if a blob or inner frame sat inside a rough loop's enclosed background hole,
#   it became that frame's child; otherwise it stayed outside. Output colors were
#   then collapsed to the single foreground color.
# - This combo changes both the grouping cue and the palette. A sample now uses
#   2-3 foreground colors, one color per semantic frame family:
#   the root frame and its direct blob children share color A, the inner frame
#   and its own blob children share color B, and an optional outside blob may
#   use color C.
# - The crucial D6 effect is that semantic children are no longer required to
#   lie in the parent's actual hole. Example: a red outer frame may have a red
#   blob child and a blue inner frame parked beside it, not inside it; that blue
#   inner frame may in turn have blue blobs parked beside it. The output still
#   canonicalizes the red child blob inside the red rectangle and the blue blobs
#   inside the blue rectangle, because color families define ownership.
# - Relative ordering is still geometric: among siblings of the same semantic
#   parent, their original row/column arrangement determines the cleaned layout.
#
# b) Complexity calibration
# - Keep the original object grammar: 1-2 frames total, 1-4 blobs total, depth 1-2.
# - Limit colors to 2 or 3 foreground colors, with repeated support in each
#   sample. Every sample has two frame families: root color repeated on the root
#   frame plus 1-2 root blobs, inner color repeated on the inner frame plus 2-3
#   inner blobs, and sometimes a third outside-blob color.
# - At least one semantic child of each frame is always geometrically outside its
#   parent's hole, so the task does not silently collapse back to the original
#   hole-based puzzle.
#
# c) Implicit-assumption audit for main()
# - `assert len(fg_colors) == 1` is invalid; the fused task requires 2-3 colors.
# - `find_components(... colors=[fg])` is invalid; the solver must scan every
#   non-background color.
# - The original parent assignment loop depended on hole membership. That is the
#   main D6 assumption that breaks here: semantic children may be fully outside
#   the parent frame. Parent inference is therefore replaced with a color-family
#   tree recovered from frame-size order plus frame/blob color matching.
# - Geometry is still used for two things that remain valid: (1) deciding whether
#   a component is a frame by checking if it has a hole, and (2) reading sibling
#   row/column order from component bboxes and centers.
# - The new solver assumes there are at most two frames and at most one frame per
#   frame-family color. The generator guarantees that by construction.
#
# d) Placement legality
# - Input objects remain disjoint monochrome connected components. Child subtrees
#   are arranged as non-overlapping groups beside their parent frame with explicit
#   gaps, so no accidental merging occurs.
# - Outside root blobs use a distinct non-frame color and are placed on a
#   different side from the semantic child group when possible, keeping
#   root-family children and outside siblings separable.
#
# e) Visual observability
# - Across examples, humans can see that a frame and its same-colored blob
#   children move together into the same cleaned container even when the raw
#   geometry does not show containment.
# - Because at least one child blob is visibly outside each parent loop in every
#   sample, the color rule is not a hidden tiebreaker; it visibly overrides the
#   old topological cue.
#
# f) Visual cleanliness
# - No overlays are introduced. Each rough component stays monochrome and disjoint.
# - Child clusters are compact side-groups rather than arbitrary scatter, so the
#   input remains readable and the output remains the same clean rectangle/dot form.
#
# g) Change inferability
# - The fused rule is not "random recolor". It is: recover frame families by
#   color grouping, then regularize geometry while preserving each object's color.
# - With only 2-3 colors and repeated objects per family, that mapping is
#   inferable from a few ARC examples without relying on numeric color ids.


TRANSPARENT = -1


class _Node:
    def __init__(self, coords: np.ndarray, bbox: tuple[int, int, int, int], center: np.ndarray, color: int):
        self.coords = coords
        self.bbox = bbox
        self.center = center
        self.color = int(color)
        self.is_frame = False
        self.parent: _Node | None = None
        self.children: list[_Node] = []
        self.mask: np.ndarray | None = None
        self.canvas: np.ndarray | None = None
        self.anchor: tuple[int, int] | None = None
        self.axis_map: dict[str, list[tuple[float, int]]] = {"row": [], "col": []}


def _component_holes(shape: tuple[int, int], coords: np.ndarray) -> list[np.ndarray]:
    mask = np.zeros(shape, dtype=bool)
    mask[tuple(coords.T)] = True
    lab, n = label_mask(~mask, connectivity=4)
    boundary = set(lab[0, :]) | set(lab[-1, :]) | set(lab[:, 0]) | set(lab[:, -1])
    return [(lab == k) for k in range(1, n + 1) if k not in boundary]


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _intervals_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(a[0], b[0]) <= min(a[1], b[1])


def _build_layout(node: _Node) -> None:
    if node.mask is not None and node.canvas is not None:
        return

    if not node.is_frame:
        node.mask = np.ones((1, 1), dtype=bool)
        node.canvas = np.array([[node.color]], dtype=int)
        node.anchor = (0, 0)
        node.axis_map = {
            "row": [(float(node.center[0]), 0)],
            "col": [(float(node.center[1]), 0)],
        }
        return

    assert node.children
    child_canvas, _placements, axis_map = _layout_group(node.children)
    h, w = child_canvas.shape
    out = np.full((h + 4, w + 4), TRANSPARENT, dtype=int)
    out[0, :] = node.color
    out[-1, :] = node.color
    out[:, 0] = node.color
    out[:, -1] = node.color
    paste(out, child_canvas, 2, 2, transparent=TRANSPARENT)

    node.canvas = out
    node.mask = out != TRANSPARENT
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

    out = np.full((max_r, max_c), TRANSPARENT, dtype=int)
    placements: dict[_Node, tuple[int, int]] = {}
    axis_map = {"row": [], "col": []}
    for node in nodes:
        assert node.mask is not None and node.anchor is not None and node.canvas is not None
        r = row_coord[node] - node.anchor[0] - min_r
        c = col_coord[node] - node.anchor[1] - min_c
        placements[node] = (int(r), int(c))
        paste(out, node.canvas, r, c, transparent=TRANSPARENT)
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
        assert frame.mask is not None and frame.canvas is not None
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
        out = np.full((max_r - min_r, max_c - min_c), TRANSPARENT, dtype=int)
        for node, (r, c) in placements.items():
            rr = r - min_r
            cc = c - min_c
            assert node.canvas is not None
            paste(out, node.canvas, rr, cc, transparent=TRANSPARENT)
        return out

    canvas, _placements, _axis_map = _layout_group(nodes)
    return canvas


def _assign_color_tree(nodes: list[_Node]) -> list[_Node]:
    for node in nodes:
        node.parent = None
        node.children = []

    frames = [node for node in nodes if node.is_frame]
    assert 1 <= len(frames) <= 2
    frames.sort(key=lambda node: (_bbox_area(node.bbox), int(node.coords.shape[0])), reverse=True)
    root = frames[0]
    frame_by_color = {}
    for frame in frames:
        assert frame.color not in frame_by_color
        frame_by_color[frame.color] = frame

    if len(frames) == 2:
        inner = frames[1]
        inner.parent = root
        root.children.append(inner)

    outside_colors: set[int] = set()
    for node in nodes:
        if node.is_frame:
            continue
        owner = frame_by_color.get(node.color)
        if owner is None:
            outside_colors.add(node.color)
            continue
        node.parent = owner
        owner.children.append(node)

    assert len(outside_colors) <= 1
    if len(frames) == 1:
        assert any(not child.is_frame for child in root.children)
    else:
        inner = frames[1]
        assert any((not child.is_frame) and child.color == root.color for child in root.children)
        assert any((not child.is_frame) and child.color == inner.color for child in inner.children)

    return [node for node in nodes if node.parent is None]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert a rough multi-color frame-family sketch into a clean canonical frame-and-dot diagram.

    Input structure:
    - The grid has one background color and 2-3 foreground colors.
    - Each foreground 4-connected component is monochrome and is either a rough closed frame or a small filled blob.
    - Foreground colors encode frame families rather than geometric enclosure:
      the root frame shares its color with its direct blob children, an inner
      frame shares a second color with its own blob children, and an optional
      outside blob may use a third color.

    Rule:
    - Recover every monochrome component and detect frames by whether they contain a hole.
    - Infer the containment tree from frame colors rather than from geometric enclosure:
      larger frames determine the nesting of frames, and blobs attach to the frame
      that shares their color even if they are not physically inside the rough loop.
    - Replace each blob by a single pixel of its original color.
    - Replace each frame by a rectangular border of its original color, padded two cells around its direct children, while preserving sibling row/column ordering from the input arrangement.

    Key insight: the rough loops are no longer trustworthy containers; color
    grouping tells which blobs belong to which frame family, while geometry only
    supplies frame-vs-blob identity, frame-size nesting, and sibling ordering.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    fg_colors = sorted(set(map(int, g.flatten())) - {bg})
    assert 2 <= len(fg_colors) <= 3

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, colors=fg_colors, return_format="dict")
    assert comps

    nodes: list[_Node] = []
    for comp in comps:
        coords = np.asarray(comp["coords"], dtype=int)
        node = _Node(
            coords=coords,
            bbox=tuple(map(int, comp["bbox"])),
            center=coords.mean(axis=0),
            color=int(comp["color"]),
        )
        node.is_frame = bool(_component_holes(g.shape, coords))
        nodes.append(node)

    roots = _assign_color_tree(nodes)
    canvas = _layout_root(roots)
    out = np.full(canvas.shape, bg, dtype=int)
    mask = canvas != TRANSPARENT
    out[mask] = canvas[mask]
    return out


# --------------- generator (o5x) ---------------

# Grid / canvas bounds
GRID_MIN = 16
GRID_MAX = 25
MAX_EXTRA_PAD = 10
MAX_PAD_PER_SIDE = 5

# Frame geometry
FRAME_H_RANGE = (7, 13)
FRAME_W_RANGE = (7, 15)
ROOT_FRAME_H_RANGE = (9, 13)
ROOT_FRAME_W_RANGE = (10, 15)
INNER_FRAME_H_RANGE = (7, 10)
INNER_FRAME_W_RANGE = (7, 11)
FRAME_CORNER_MAX = 5
FRAME_MIN_EDGE = 2
FRAME_TURN_RATIO = (0.34, 0.43)
FRAME_RETRIES = 60

# Blob geometry
BLOB_SIZE_RANGE = (4, 11)
BLOB_BBOX_RANGE = (2, 5)

# Input-side group arrangement
ARRANGE_GAP_RANGE = (1, 2)
ARRANGE_EXTRA_SPAN = (0, 1)
CHILD_CLUSTER_GAP_RANGE = (1, 2)
ROOT_OUTSIDE_GAP_RANGE = (1, 3)

# Tree structure
OUTSIDE_BLOB_PROB = 0.5
OUTER_BLOB_RANGE = (1, 2)
INNER_BLOB_RANGE = (2, 3)

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
        self.color: int | None = None
        self.depth: int | None = None
        self.subtree_bbox = (0, 0, 0, 0)
        self.cluster_side: str | None = None


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


def _blob_mask() -> np.ndarray:
    while True:
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
        if not _component_holes(mask.shape, pts):
            return mask


def _subtree_shape(obj: _Obj) -> tuple[int, int]:
    r0, c0, r1, c1 = obj.subtree_bbox
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _arrange_subtrees(children: list[_Obj]) -> tuple[dict[_Obj, tuple[int, int]], int, int]:
    random.shuffle(children)
    dims = [_subtree_shape(child) for child in children]
    if len(children) == 1:
        h, w = dims[0]
        return {children[0]: (0, 0)}, h, w

    mode = random.choice(("vertical", "horizontal", "diagonal"))
    gaps = [random.randint(*ARRANGE_GAP_RANGE) for _ in range(len(children) - 1)]
    placements: dict[_Obj, tuple[int, int]] = {}

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


def _attach_group(frame_shape: tuple[int, int], group_h: int, group_w: int, side: str, gap: int) -> tuple[int, int]:
    h, w = frame_shape
    if side == "right":
        base_r = random.randint(0, h - group_h) if group_h <= h else 0
        return base_r, w + gap
    if side == "left":
        base_r = random.randint(0, h - group_h) if group_h <= h else 0
        return base_r, -gap - group_w
    if side == "bottom":
        base_c = random.randint(0, w - group_w) if group_w <= w else 0
        return h + gap, base_c
    base_c = random.randint(0, w - group_w) if group_w <= w else 0
    return -gap - group_h, base_c


def _union_bbox(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return (
        min(a[0], b[0]),
        min(a[1], b[1]),
        max(a[2], b[2]),
        max(a[3], b[3]),
    )


def _prepare(node: _Obj, depth: int = 0) -> bool:
    if node.kind == "blob":
        node.mask = _blob_mask()
        h, w = node.mask.shape
        node.subtree_bbox = (0, 0, h - 1, w - 1)
        return True

    for child in node.children:
        if not _prepare(child, depth + 1):
            return False

    if depth == 0:
        h = random.randint(*ROOT_FRAME_H_RANGE)
        w = random.randint(*ROOT_FRAME_W_RANGE)
    else:
        h = random.randint(*INNER_FRAME_H_RANGE)
        w = random.randint(*INNER_FRAME_W_RANGE)
    node.mask = _frame_mask(h, w)
    if node.mask is None:
        return False

    bbox = (0, 0, h - 1, w - 1)
    if not node.children:
        node.subtree_bbox = bbox
        return True

    placements, group_h, group_w = _arrange_subtrees(node.children)
    node.cluster_side = random.choice(("right", "bottom"))
    gap = random.randint(*CHILD_CLUSTER_GAP_RANGE)
    base_r, base_c = _attach_group(node.mask.shape, group_h, group_w, node.cluster_side, gap)

    for child, (r, c) in placements.items():
        cr0, cc0, cr1, cc1 = child.subtree_bbox
        child.rel = (base_r + r - cr0, base_c + c - cc0)
        shifted = (
            child.rel[0] + cr0,
            child.rel[1] + cc0,
            child.rel[0] + cr1,
            child.rel[1] + cc1,
        )
        bbox = _union_bbox(bbox, shifted)

    node.subtree_bbox = bbox
    return True


def _walk(node: _Obj, base: tuple[int, int] = (0, 0)) -> list[_Obj]:
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
    options = [side for side in ("left", "right", "top", "bottom") if side != root.cluster_side]
    side = random.choice(options or ["right"])
    outside.outside_side = side
    gap = random.randint(*ROOT_OUTSIDE_GAP_RANGE)
    if side == "right":
        outside.rel = ((h - bh) // 2, w + gap)
    elif side == "left":
        outside.rel = ((h - bh) // 2, -gap - bw)
    elif side == "bottom":
        outside.rel = (h + gap, (w - bw) // 2)
    else:
        outside.rel = (-gap - bh, (w - bw) // 2)


def _assign_family_colors(root: _Obj, outside: _Obj | None, bg: int) -> None:
    frame_depths: list[int] = []

    def scan_frames(obj: _Obj, frame_depth: int) -> None:
        frame_depths.append(frame_depth)
        for child in obj.children:
            if child.kind == "frame":
                scan_frames(child, frame_depth + 1)

    scan_frames(root, 0)
    n_frame_colors = max(frame_depths) + 1
    n_colors = n_frame_colors + (1 if outside is not None else 0)
    palette = random.sample([c for c in range(10) if c != bg], n_colors)

    def paint_family(obj: _Obj, frame_depth: int) -> None:
        obj.depth = frame_depth
        obj.color = palette[frame_depth]
        for child in obj.children:
            if child.kind == "frame":
                paint_family(child, frame_depth + 1)
            else:
                child.depth = frame_depth
                child.color = palette[frame_depth]

    paint_family(root, 0)
    if outside is not None:
        outside.depth = -1
        outside.color = palette[-1]


def _node_from_obj(obj: _Obj) -> _Node:
    assert obj.mask is not None and obj.color is not None
    r0, c0 = obj.pos
    coords = np.argwhere(obj.mask) + np.array([r0, c0])
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    node = _Node(
        coords=coords.astype(int),
        bbox=(int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])),
        center=coords.mean(axis=0),
        color=int(obj.color),
    )
    node.is_frame = obj.kind == "frame"
    return node


def _expected_output(root: _Obj, outside: _Obj | None, bg: int) -> np.ndarray:
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
    canvas = _layout_root(roots)
    out = np.full(canvas.shape, bg, dtype=int)
    mask = canvas != TRANSPARENT
    out[mask] = canvas[mask]
    return out


def _make_tree() -> tuple[_Obj, _Obj | None]:
    root = _Obj("frame")
    inner = _Obj("frame")
    inner.children = [_Obj("blob") for _ in range(random.randint(*INNER_BLOB_RANGE))]
    outer_blobs = [_Obj("blob") for _ in range(random.randint(*OUTER_BLOB_RANGE))]
    root.children = [inner] + outer_blobs
    outside = _Obj("blob") if random.random() < OUTSIDE_BLOB_PROB else None
    return root, outside


def _draw(objects: list[_Obj], bg: int) -> np.ndarray | None:
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
        assert obj.mask is not None and obj.color is not None
        r = obj.pos[0] + shift_r
        c = obj.pos[1] + shift_c
        obj.pos = (r, c)
        patch = np.full(obj.mask.shape, bg, dtype=int)
        patch[obj.mask] = int(obj.color)
        paste(grid, patch, r, c, transparent=bg)

    fg_colors = sorted({int(obj.color) for obj in objects})
    bg_count = int(np.count_nonzero(grid == bg))
    if bg_count <= max(int(np.count_nonzero(grid == col)) for col in fg_colors):
        return None
    return grid


def _family_signature_ok(root: _Obj, outside: _Obj | None) -> bool:
    inner_frames = [child for child in root.children if child.kind == "frame"]
    outer_blobs = [child for child in root.children if child.kind == "blob"]
    if len(inner_frames) != 1 or not outer_blobs:
        return False
    inner = inner_frames[0]
    inner_blobs = [child for child in inner.children if child.kind == "blob"]
    if len(inner_blobs) < 2:
        return False
    if not all(child.kind == "blob" for child in inner.children):
        return False
    if root.color is None or inner.color is None or root.color == inner.color:
        return False
    if any(blob.color != root.color for blob in outer_blobs):
        return False
    if any(blob.color != inner.color for blob in inner_blobs):
        return False
    if outside is not None and (outside.color in {root.color, inner.color}):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    bg = random.randrange(10)
    root, outside = _make_tree()
    if not _prepare(root):
        return None
    if outside is not None:
        if not _prepare(outside):
            return None
        _place_outside(root, outside)
    _assign_family_colors(root, outside, bg)
    if not _family_signature_ok(root, outside):
        return None

    objects = _walk(root)
    if outside is not None:
        outside.pos = outside.rel
        objects.append(outside)

    grid = _draw(objects, bg)
    if grid is None:
        return None
    expected = _expected_output(root, outside, bg)
    try:
        got = main(grid)
    except Exception:
        return None
    if not np.array_equal(got, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 2d0172a1 combo input")
