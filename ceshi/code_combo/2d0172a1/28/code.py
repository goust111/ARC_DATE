# Combo variant of 2d0172a1 (index 28)
# combo_spec: D1/3+D7/3
# Name: 方向性色块+按宽高比排序

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT / "solvers") not in sys.path:
    sys.path.append(str(ROOT / "solvers"))

from arc_utils import find_components, get_bg, label_mask, paste


# Design analysis for the fused variant:
# a) Concrete behavior:
#    Original blobs were undirected filled clusters and collapsed to 1 pixel; siblings
#    kept their input row/column-band order. Here each non-frame component is a
#    directional blob with a non-square bbox: horizontal examples are 2x3 or 2x4,
#    vertical examples are 3x2 or 4x2. In the output, horizontal blobs become 1x2
#    dominoes and vertical blobs become 2x1 dominoes. Siblings are no longer placed
#    from their spatial bands; they are sorted by input bbox aspect ratio (width/height)
#    from wide to tall and then laid out left-to-right. Example: if a frame contains a
#    2x4 horizontal arrow, a near-square inner frame, and a 4x2 vertical arrow, the
#    original task would preserve whichever left/right or top/bottom order the sketch
#    used, while this variant outputs [horizontal blob][inner frame][vertical blob].
# b) Complexity calibration:
#    Only two visible blob axes are used in the output (1x2 or 2x1), which keeps the
#    rule human-readable, but the input still includes multiple template sizes so the
#    aspect-ratio key is visible and repeated rather than collapsing to a single tie.
#    Frames stay in the same object pool, so sorting can mix frames with blobs.
# c) Hidden-assumption review:
#    - `_build_layout` can no longer assume every blob reduces to 1x1.
#    - Any layout code that relied on row/col overlap bands is invalid; it is replaced
#      by a pure ratio sort with only deterministic tie-breakers as fallback.
#    - Root-side placement for outside blobs must use each blob mask's real width/height.
#    - Blob-axis detection must not depend on color ids or non-visual metadata; it uses
#      only geometric spread and bbox shape.
# d) Placement legality:
#    Generator uses fixed connected templates, random frame sizes, and explicit gaps.
#    Children are arranged without overlap inside the hole area; optional outside blobs
#    are placed one side away from the root frame with at least one bg cell gap.
# e/f/g) Observability, cleanliness, and inferability:
#    The fused rule is visible in a few examples because input blobs are visibly wide or
#    tall, outputs preserve that as 1x2 vs 2x1, and the left-to-right order follows the
#    same wide-to-tall ranking. No hidden use of palette values is involved. The output
#    stays clean because each canonical sibling is separated by one bg column and each
#    frame keeps a two-cell margin around its children.


class _Node:
    def __init__(self, coords: np.ndarray, bbox: tuple[int, int, int, int], center: np.ndarray):
        self.coords = coords
        self.bbox = bbox
        self.center = center
        self.ratio = _bbox_ratio(bbox)
        self.is_frame = False
        self.orient: str | None = None
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


def _bbox_ratio(bbox: tuple[int, int, int, int]) -> float:
    h, w = _bbox_hw(bbox)
    return float(w) / float(h)


def _detect_blob_axis(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> str:
    h, w = _bbox_hw(bbox)
    if w > h:
        return "h"
    if h > w:
        return "v"
    centered = coords.astype(float) - coords.mean(axis=0, keepdims=True)
    var_r, var_c = centered.var(axis=0)
    return "h" if var_c >= var_r else "v"


def _canonical_blob_mask(orient: str) -> np.ndarray:
    if orient == "h":
        return np.ones((1, 2), dtype=bool)
    return np.ones((2, 1), dtype=bool)


def _node_sort_key(node: _Node) -> tuple[float, int, int, float, float]:
    h, w = _bbox_hw(node.bbox)
    return (-node.ratio, -w, h, float(node.center[0]), float(node.center[1]))


def _build_layout(node: _Node) -> None:
    if node.mask is not None:
        return

    if not node.is_frame:
        node.orient = _detect_blob_axis(node.coords, node.bbox)
        node.mask = _canonical_blob_mask(node.orient)
        node.anchor = (node.mask.shape[0] // 2, node.mask.shape[1] // 2)
        node.axis_map = {
            "row": [(float(node.center[0]), int(node.anchor[0]))],
            "col": [(float(node.center[1]), int(node.anchor[1]))],
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


def _layout_group(nodes: list[_Node]) -> tuple[np.ndarray, dict[_Node, tuple[int, int]], dict[str, list[tuple[float, int]]]]:
    assert nodes
    for node in nodes:
        _build_layout(node)

    ordered = sorted(nodes, key=_node_sort_key)
    group_h = max(node.mask.shape[0] for node in ordered if node.mask is not None)
    group_w = sum(node.mask.shape[1] for node in ordered if node.mask is not None) + max(0, len(ordered) - 1)

    out = np.zeros((group_h, group_w), dtype=bool)
    placements: dict[_Node, tuple[int, int]] = {}
    axis_map = {"row": [], "col": []}

    c = 0
    for node in ordered:
        assert node.mask is not None and node.anchor is not None
        h, w = node.mask.shape
        r = (group_h - h) // 2
        placements[node] = (int(r), int(c))
        out[r : r + h, c : c + w] |= node.mask
        axis_map["row"].append((float(node.center[0]), int(r + node.anchor[0])))
        axis_map["col"].append((float(node.center[1]), int(c + node.anchor[1])))
        c += w + 1

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
    blobs = [node for node in nodes if not node.is_frame]
    if len(frames) == 1 and blobs:
        frame = frames[0]
        assert frame.mask is not None
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
            extents.append((r - 1, c - 1, r + bh + 1, c + bw + 1))

        min_r = min(e[0] for e in extents)
        min_c = min(e[1] for e in extents)
        max_r = max(e[2] for e in extents)
        max_c = max(e[3] for e in extents)
        out = np.zeros((max_r - min_r, max_c - min_c), dtype=bool)
        for node, (r, c) in placements.items():
            rr = r - min_r
            cc = c - min_c
            assert node.mask is not None
            out[rr : rr + node.mask.shape[0], cc : cc + node.mask.shape[1]] |= node.mask
        return out

    mask, _placements, _axis_map = _layout_group(nodes)
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert a rough nested frame-and-directional-blob sketch into a clean canonical diagram.

    Input structure:
    - The grid has one background color and one foreground color.
    - Foreground 4-connected objects are rough simple closed frames or small directional blobs.
    - Directional blobs are non-frame connected shapes whose bbox is visibly wider or taller; frames may contain frames/blobs, and one blob may sit outside the outer frame.

    Rule:
    - Recover the containment tree from frame holes exactly as in the base task.
    - Replace each directional blob by a 1x2 horizontal block or a 2x1 vertical block according to the blob's geometric axis.
    - Replace each frame by a rectangular border with a two-cell interior margin around its direct children.
    - For every sibling group, ignore the original row/column sketch order and instead sort siblings by input bbox aspect ratio (width/height), from wider to taller, then place them left-to-right with one bg cell between canonical objects.

    Key insight: the sketch's exact wobble still does not matter, but blob direction now changes both the blob's canonical shape and its aspect-ratio rank, so orientation drives order through geometry rather than through raw position.
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


# --------------- generator ---------------

GRID_MIN = 16
GRID_MAX = 25
MAX_EXTRA_PAD = 12
MAX_PAD_PER_SIDE = 6

FRAME_H_RANGE = (8, 20)
FRAME_W_RANGE = (8, 24)
FRAME_CORNER_MAX = 5
FRAME_MIN_EDGE = 2
FRAME_TURN_RATIO = (0.34, 0.43)
FRAME_RETRIES = 60

ARRANGE_GAP_COMPACT = (2, 3)
ARRANGE_GAP_NORMAL = (2, 5)
ARRANGE_EXTRA_SPAN = (0, 3)
FRAME_MARGIN_RANGE = (3, 5)
COMPACT_MARGIN = 3
OUTSIDE_GAP_RANGE = (2, 5)

DEPTH1_PROB = 0.4
OUTSIDE_BLOB_PROB = 0.35
DEPTH1_INSIDE_RANGE = (2, 4)
DEPTH2_INNER_BLOBS = (1, 3)
DEPTH2_OUTER_BLOBS = (1, 2)
MAX_TOTAL_CHILDREN = 5

STEP_MIX_ALT_PROB = 0.7
GEN_RETRIES = 400

H_BLOB_TEMPLATES = [
    np.array([[0, 1, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[0, 1, 1, 0], [1, 1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1, 1], [0, 1, 1, 0]], dtype=bool),
]
V_BLOB_TEMPLATES = [arr.T.copy() for arr in H_BLOB_TEMPLATES]


class _Obj:
    def __init__(self, kind, children=None, outside_side=None, orient=None):
        self.kind = kind
        self.children = list(children or [])
        self.outside_side = outside_side
        self.orient = orient
        self.mask: np.ndarray | None = None
        self.rel = (0, 0)
        self.pos = (0, 0)


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
    out: list[tuple[int, int]] = []
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


def _blob_mask(orient: str) -> np.ndarray:
    if orient == "h":
        return random.choice(H_BLOB_TEMPLATES).copy()
    return random.choice(V_BLOB_TEMPLATES).copy()


def _arrange(children: list[_Obj], compact: bool = False) -> tuple[dict[_Obj, tuple[int, int]], int, int]:
    random.shuffle(children)
    dims = [child.mask.shape for child in children if child.mask is not None]
    k = len(children)
    if k == 1:
        h, w = dims[0]
        return {children[0]: (0, 0)}, h, w

    mode = random.choice(("vertical", "horizontal", "diagonal"))
    gap_lo, gap_hi = ARRANGE_GAP_COMPACT if compact else ARRANGE_GAP_NORMAL
    gaps = [random.randint(gap_lo, gap_hi) for _ in range(k - 1)]
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


def _prepare(node: _Obj, compact: bool = False) -> bool:
    if node.kind == "blob":
        assert node.orient in {"h", "v"}
        node.mask = _blob_mask(node.orient)
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


def _random_blob_orient(force: str | None = None) -> str:
    if force is not None:
        return force
    return random.choice(("h", "v"))


def _make_tree(force_depth2: bool | None = None, force_outside: bool | None = None) -> tuple[_Obj, _Obj | None]:
    root = _Obj("frame")
    outside = None
    sides = ("left", "right", "top", "bottom")
    use_depth1 = random.random() < DEPTH1_PROB if force_depth2 is None else not force_depth2
    if use_depth1:
        n_inside = random.randint(*DEPTH1_INSIDE_RANGE)
        orients = [_random_blob_orient("h"), _random_blob_orient("v")]
        while len(orients) < n_inside:
            orients.append(_random_blob_orient())
        random.shuffle(orients)
        root.children = [_Obj("blob", orient=o) for o in orients[:n_inside]]
        add_outside = random.random() < OUTSIDE_BLOB_PROB if force_outside is None else force_outside
        if add_outside and n_inside < MAX_TOTAL_CHILDREN:
            outside = _Obj("blob", outside_side=random.choice(sides), orient=_random_blob_orient())
    else:
        inner = _Obj("frame")
        n_inner_lo, n_inner_hi = DEPTH2_INNER_BLOBS
        n_outer_lo, n_outer_hi = DEPTH2_OUTER_BLOBS
        if force_outside is True:
            allowed: list[tuple[int, int]] = []
            for n_inner in range(n_inner_lo, n_inner_hi + 1):
                for n_outer in range(n_outer_lo, n_outer_hi + 1):
                    if n_inner + n_outer < MAX_TOTAL_CHILDREN:
                        allowed.append((n_inner, n_outer))
            assert allowed
            n_inner, n_outer = random.choice(allowed)
        else:
            n_inner = random.randint(n_inner_lo, n_inner_hi)
        inner_orients = [_random_blob_orient("h")]
        while len(inner_orients) < n_inner:
            inner_orients.append(_random_blob_orient())
        random.shuffle(inner_orients)
        inner.children = [_Obj("blob", orient=o) for o in inner_orients[:n_inner]]

        if force_outside is not True:
            n_outer = random.randint(n_outer_lo, n_outer_hi)
        outer_orients = [_random_blob_orient("v")]
        while len(outer_orients) < n_outer:
            outer_orients.append(_random_blob_orient())
        random.shuffle(outer_orients)
        outer_blobs = [_Obj("blob", orient=o) for o in outer_orients[:n_outer]]
        root.children = [inner] + outer_blobs
        add_outside = random.random() < OUTSIDE_BLOB_PROB if force_outside is None else force_outside
        if add_outside and len(inner.children) + len(outer_blobs) < MAX_TOTAL_CHILDREN:
            outside = _Obj("blob", outside_side=random.choice(sides), orient=_random_blob_orient())
    return root, outside


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


def _ratio_order_signature(frame_obj: _Obj) -> tuple[list[int], list[int]]:
    children = frame_obj.children
    if len(children) < 2:
        return [], []

    def ratio(obj: _Obj) -> float:
        assert obj.mask is not None
        h, w = obj.mask.shape
        return float(w) / float(h)

    by_ratio = sorted(range(len(children)), key=lambda i: (-ratio(children[i]), children[i].pos[1], children[i].pos[0]))
    by_x = sorted(range(len(children)), key=lambda i: (children[i].pos[1], children[i].pos[0]))
    return by_ratio, by_x


def _has_fusion_signal(root: _Obj) -> bool:
    stack = [root]
    saw_h = False
    saw_v = False
    saw_ratio_vs_position_mismatch = False
    while stack:
        node = stack.pop()
        if node.kind == "blob":
            saw_h |= node.orient == "h"
            saw_v |= node.orient == "v"
        else:
            by_ratio, by_x = _ratio_order_signature(node)
            if by_ratio and by_ratio != by_x:
                saw_ratio_vs_position_mismatch = True
            stack.extend(node.children)
    return saw_h and saw_v and saw_ratio_vs_position_mismatch


def _children_have_distinct_ratios(children: list[_Obj]) -> bool:
    if len(children) < 2:
        return True
    ratios = []
    for child in children:
        assert child.mask is not None
        h, w = child.mask.shape
        ratios.append(round(float(w) / float(h), 6))
    return len(ratios) == len(set(ratios))


def _has_distinct_ratio_groups(root: _Obj) -> bool:
    stack = [root]
    while stack:
        node = stack.pop()
        if node.kind != "frame":
            continue
        if not _children_have_distinct_ratios(node.children):
            return False
        stack.extend(node.children)
    return True


def _gen_attempt(force_depth2: bool | None = None, force_outside: bool | None = None) -> np.ndarray | None:
    bg, fg = random.sample(range(10), 2)
    root, outside = _make_tree(force_depth2=force_depth2, force_outside=force_outside)
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
    if not _has_distinct_ratio_groups(root):
        return None
    if not _has_fusion_signal(root):
        return None

    expected = _expected_output(root, outside, bg, fg)
    try:
        got = main(grid)
    except Exception:
        return None
    if not np.array_equal(got, expected):
        return None
    return grid


_BRANCH_MODES = (
    (False, False),
    (False, True),
    (True, False),
    (True, True),
)
_branch_mode_idx = 0


def generate_input() -> np.ndarray:
    global _branch_mode_idx

    force_depth2, force_outside = _BRANCH_MODES[_branch_mode_idx % len(_BRANCH_MODES)]
    _branch_mode_idx += 1
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(force_depth2=force_depth2, force_outside=force_outside)
        if grid is not None:
            return grid
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(force_depth2=force_depth2, force_outside=None)
        if grid is not None:
            return grid
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(force_depth2=None, force_outside=None)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid fused 2d0172a1 input")
