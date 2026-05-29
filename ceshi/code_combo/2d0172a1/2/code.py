# Combo variant of 2d0172a1 (index 2)
# combo_spec: D2/2+D7/0
# Name: 色块面积编码+按面积排序

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import find_components, get_bg, label_mask, paste


# ---------------------------------------------------------------------------
# Design analysis for combo D2/2 + D7/0
#
# fused_rule:
#   - Input stays a 2-color rough nesting task: each foreground component is
#     either a one-hole frame or a compact filled blob.
#   - Blob area now has two simultaneous jobs:
#       4-6  -> output as 1x1 dot
#       7-8  -> output as 1x2 horizontal dash
#       9-11 -> output as 2x1 vertical dash
#     and siblings inside the same parent are ordered by original area, largest
#     first, instead of preserving row/column band order from the sketch.
#
# behavior_trace:
#   - Example sibling blobs with areas [5, 8, 10].
#   - Original task:
#       input top-to-bottom order = [5, 8, 10]
#       output markers            = [dot, dot, dot]
#       output order              = still [5, 8, 10] because band order is kept
#   - Fused task:
#       input top-to-bottom order may still be [5, 8, 10]
#       output markers become     = [dot, 1x2, 2x1]
#       output order becomes      = [10, 8, 5] because area also drives layout
#     So the same measured quantity changes both glyph shape and sibling slot.
#
# complexity_calibration:
#   - Only three blob glyphs are used, keeping the count-to-shape mapping
#     learnable from a few examples.
#   - Generator forces at least one non-small blob in every sample and at least
#     one sibling group with >=2 distinct-area children, so the combo does not
#     collapse into either the original all-dot task or a pure size-encoding task.
#   - Blob areas are drawn from a small visible set with repeated bucket support,
#     which keeps the concept observable without exploding the hypothesis space.
#
# implicit_assumption_audit:
#   - Original `_build_layout()` assumed every blob canonicalized to (1, 1);
#     that must be replaced with area-dependent masks.
#   - Original `_axis_coordinates()` assumed input spatial bands define sibling
#     order; that premise is invalid here, so both axes must instead consume the
#     same area-based ordering.
#   - Original `_layout_root()` reserved a hard-coded 3x3 slot for outside dots;
#     this must expand to the true blob glyph bbox so 1x2 and 2x1 markers fit.
#   - The new solver assumes blob areas lie in 4..11 and therefore map cleanly to
#     one of the three glyph buckets; the generator constructs blobs exactly under
#     that constraint.
#
# placement_legality:
#   - `_layout_group()` still spaces canonical masks using true mask extents, so
#     mixed 1x1 / 1x2 / 2x1 blob markers keep at least one background cell between
#     siblings and from enclosing frame borders.
#   - The generator arranges input siblings with the original non-overlap margins
#     and still verifies every sample by executing `main()`.
#   - Outside blobs are placed with a positive gap from the outer frame and their
#     rendered bbox is used when padding the grid, so no clipping occurs.
#
# visual_observability:
#   - Humans can count rough blob pixels and see both consequences in the output:
#     larger blobs move toward the top-left of each sibling group and also change
#     from dot to horizontal/vertical dash.
#   - No hidden value such as color id or generation order affects the rule.
#   - The generator biases input placement to ascending area order, making the
#     contrast with the descending output order visible in 2-4 examples.
#
# cleanliness_and_inference:
#   - Output remains a sparse blackboard-style diagram of rectangular borders and
#     small marks; only size and relative placement change, so the image stays
#     readable.
#   - Because the same scalar feature controls two visible effects, the combo is
#     easier to infer than two unrelated simultaneous modifications.
# ---------------------------------------------------------------------------


SMALL_RANGE = (4, 6)
MEDIUM_RANGE = (7, 8)
LARGE_RANGE = (9, 11)
BUCKET_NAMES = ("small", "medium", "large")
AREA_BY_BUCKET = {
    "small": [4, 5, 6],
    "medium": [7, 8],
    "large": [9, 10, 11],
}
VISIBLE_AREA_POOL = [5, 6, 7, 8, 10, 11]


class _Node:
    def __init__(self, coords: np.ndarray, bbox: tuple[int, int, int, int], center: np.ndarray):
        self.coords = coords
        self.bbox = bbox
        self.center = center
        self.area = int(coords.shape[0])
        self.is_frame = False
        self.parent: _Node | None = None
        self.children: list[_Node] = []
        self.mask: np.ndarray | None = None
        self.anchor: tuple[int, int] | None = None
        self.axis_map: dict[str, list[tuple[float, int]]] = {"row": [], "col": []}


def _blob_bucket(area: int) -> str:
    area = int(area)
    if SMALL_RANGE[0] <= area <= SMALL_RANGE[1]:
        return "small"
    if MEDIUM_RANGE[0] <= area <= MEDIUM_RANGE[1]:
        return "medium"
    if LARGE_RANGE[0] <= area <= LARGE_RANGE[1]:
        return "large"
    raise AssertionError(f"unexpected blob area: {area}")


def _blob_output_mask(area: int) -> np.ndarray:
    bucket = _blob_bucket(int(area))
    if bucket == "small":
        return np.ones((1, 1), dtype=bool)
    if bucket == "medium":
        return np.ones((1, 2), dtype=bool)
    return np.ones((2, 1), dtype=bool)


def _component_holes(shape: tuple[int, int], coords: np.ndarray) -> list[np.ndarray]:
    mask = np.zeros(shape, dtype=bool)
    mask[tuple(coords.T)] = True
    lab, n = label_mask(~mask, connectivity=4)
    boundary = set(lab[0, :]) | set(lab[-1, :]) | set(lab[:, 0]) | set(lab[:, -1])
    return [(lab == k) for k in range(1, n + 1) if k not in boundary]


def _node_sort_key(node: _Node) -> tuple[float, float, float, int, int]:
    return (-float(node.area), float(node.center[0]), float(node.center[1]), int(node.bbox[0]), int(node.bbox[1]))


def _build_layout(node: _Node) -> None:
    if node.mask is not None:
        return

    if not node.is_frame:
        node.mask = _blob_output_mask(node.area)
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
    ordered = sorted(nodes, key=_node_sort_key)
    coord_by_node: dict[_Node, int] = {}
    coord = 0
    for i, node in enumerate(ordered):
        assert node.mask is not None and node.anchor is not None
        if i:
            prev = ordered[i - 1]
            assert prev.mask is not None and prev.anchor is not None
            prev_down = prev.mask.shape[axis] - 1 - prev.anchor[axis]
            curr_up = node.anchor[axis]
            coord += prev_down + 2 + curr_up
        coord_by_node[node] = int(coord)
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
            assert dot.mask is not None and dot.anchor is not None
            side = _outside_side(dot, frame)
            if side in {"right", "left"}:
                r = _map_frame_axis(frame, 0, float(dot.center[0])) - dot.anchor[0]
                c = frame.mask.shape[1] + 1 if side == "right" else -dot.mask.shape[1] - 1
            else:
                r = frame.mask.shape[0] + 1 if side == "bottom" else -dot.mask.shape[0] - 1
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
            assert node.mask is not None
            out[rr : rr + node.mask.shape[0], cc : cc + node.mask.shape[1]] |= node.mask
        return out

    mask, _placements, _axis_map = _layout_group(nodes)
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert a rough nested frame-and-blob sketch into a clean canonical frame diagram where blob area controls both mark shape and sibling order.

    Input structure:
    - The grid uses exactly one background color and one foreground color.
    - Foreground components are either rough one-hole frames or compact filled blobs with area 4-11.
    - Frames may contain frames/blobs, and at most one blob may sit outside the outer frame.

    Rule:
    - Recover the containment tree by checking which frame-hole fully contains each component.
    - Replace each frame with a 1-pixel rectangular border padded two cells around its direct children.
    - Replace each blob by an area-coded glyph: 4-6 -> 1x1, 7-8 -> 1x2 horizontal, 9-11 -> 2x1 vertical.
    - Inside every sibling group, ignore the original row/column bands and sort children by original component area, largest first, leaving one background cell between canonical objects.
    - Keep an outside blob on the side of the root frame where it originally appears, using its area-coded glyph there as well.

    Key insight: The sketch still only matters for containment and outside-vs-inside placement, but blob pixel count now becomes the single latent variable that determines both how a blob is drawn and where siblings appear.
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
        if not node.is_frame:
            _blob_bucket(node.area)
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

FRAME_H_RANGE = (7, 20)
FRAME_W_RANGE = (7, 24)
FRAME_CORNER_MAX = 5
FRAME_MIN_EDGE = 2
FRAME_TURN_RATIO = (0.34, 0.43)
FRAME_RETRIES = 60

BLOB_SIZE_RANGE = (4, 11)
BLOB_BBOX_RANGE = (2, 5)

ARRANGE_GAP_COMPACT = (2, 3)
ARRANGE_GAP_NORMAL = (2, 5)
ARRANGE_EXTRA_SPAN = (0, 4)
FRAME_MARGIN_RANGE = (3, 5)
COMPACT_MARGIN = 3
ANTI_AREA_ORDER_PROB = 0.9

OUTSIDE_GAP_RANGE = (2, 5)

DEPTH1_PROB = 0.25
OUTSIDE_BLOB_PROB = 0.55
DEPTH1_INSIDE_RANGE = (2, 3)
DEPTH2_INNER_BLOBS = (2, 2)
DEPTH2_OUTER_BLOBS = (0, 1)
MAX_TOTAL_BLOBS = 4

STEP_MIX_ALT_PROB = 0.7
BLOB_SHAPE_ATTEMPTS = 12

GEN_RETRIES = 500


class _Obj:
    def __init__(self, kind, children=None, outside_side=None, blob_area=None):
        self.kind = kind
        self.children = list(children or [])
        self.mask: np.ndarray | None = None
        self.rel = (0, 0)
        self.pos = (0, 0)
        self.outside_side = outside_side
        self.blob_area = blob_area
        self.layout_mode: str | None = None
        self.layout_order: list[_Obj] = []


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


def _blob_mask(size: int) -> np.ndarray:
    size = int(size)
    assert BLOB_SIZE_RANGE[0] <= size <= BLOB_SIZE_RANGE[1]
    feasible = [
        (h, w)
        for h in range(*BLOB_BBOX_RANGE)
        for w in range(*BLOB_BBOX_RANGE)
        if h * w >= size and h + w - 1 <= size
    ]
    random.shuffle(feasible)
    for h, w in feasible:
        for _ in range(BLOB_SHAPE_ATTEMPTS):
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
                if not frontier:
                    break
                cells.add(random.choice(frontier))
            if len(cells) != size:
                continue
            mask = np.zeros((h, w), dtype=bool)
            pts = np.array(list(cells), dtype=int)
            mask[pts[:, 0], pts[:, 1]] = True
            if not _component_holes(mask.shape, pts):
                return mask
    raise RuntimeError(f"failed to build blob of area {size}")


def _obj_area(obj: _Obj) -> int:
    assert obj.mask is not None
    return int(np.count_nonzero(obj.mask))


def _sort_key_for_input(child: _Obj) -> tuple[int, int, int]:
    assert child.mask is not None
    return (_obj_area(child), child.mask.shape[0], child.mask.shape[1])


def _sort_key_for_output(child: _Obj) -> tuple[int, float, float]:
    assert child.mask is not None
    center_r = child.pos[0] + (child.mask.shape[0] - 1) / 2
    center_c = child.pos[1] + (child.mask.shape[1] - 1) / 2
    return (-_obj_area(child), float(center_r), float(center_c))


def _arrange(children, compact=False):
    ordered = list(children)
    dims = [child.mask.shape for child in ordered]
    k = len(ordered)
    assert k >= 1
    if k > 1:
        if random.random() < ANTI_AREA_ORDER_PROB:
            ordered.sort(key=_sort_key_for_input)
        else:
            random.shuffle(ordered)
        dims = [child.mask.shape for child in ordered]
    if k == 1:
        return {ordered[0]: (0, 0)}, dims[0][0], dims[0][1], "single", ordered

    mode = random.choice(("vertical", "horizontal", "diagonal"))
    gap_lo, gap_hi = ARRANGE_GAP_COMPACT if compact else ARRANGE_GAP_NORMAL
    gaps = [random.randint(gap_lo, gap_hi) for _ in range(k - 1)]
    placements = {}

    if mode == "vertical":
        group_h = sum(h for h, _ in dims) + sum(gaps)
        group_w = max(w for _, w in dims) + random.randint(*ARRANGE_EXTRA_SPAN)
        r = 0
        for child, (h, w), gap in zip(ordered, dims, gaps + [0]):
            placements[child] = (r, random.randint(0, group_w - w))
            r += h + gap
        return placements, group_h, group_w, mode, ordered

    if mode == "horizontal":
        group_h = max(h for h, _ in dims) + random.randint(*ARRANGE_EXTRA_SPAN)
        group_w = sum(w for _, w in dims) + sum(gaps)
        c = 0
        for child, (h, w), gap in zip(ordered, dims, gaps + [0]):
            placements[child] = (random.randint(0, group_h - h), c)
            c += w + gap
        return placements, group_h, group_w, mode, ordered

    group_h = sum(h for h, _ in dims) + sum(gaps)
    group_w = sum(w for _, w in dims) + sum(gaps)
    r = c = 0
    for child, (h, w), gap in zip(ordered, dims, gaps + [0]):
        placements[child] = (r, c)
        r += h + gap
        c += w + gap
    return placements, group_h, group_w, mode, ordered


def _prepare(node, compact=False):
    if node.kind == "blob":
        assert node.blob_area is not None
        try:
            node.mask = _blob_mask(int(node.blob_area))
        except RuntimeError:
            return False
        return True

    for child in node.children:
        if not _prepare(child, compact=child.kind == "frame"):
            return False

    has_frame_child = any(child.kind == "frame" for child in node.children)
    tight = compact or has_frame_child
    placements, group_h, group_w, mode, ordered = _arrange(node.children, compact=tight)
    node.layout_mode = mode
    node.layout_order = list(ordered)
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


def _expected_output(root, outside, bg, fg):
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


def _sample_blob_areas(total_blobs: int) -> list[int]:
    total_blobs = int(total_blobs)
    assert 1 <= total_blobs <= MAX_TOTAL_BLOBS
    if total_blobs == 1:
        return [random.choice(AREA_BY_BUCKET[random.choice(("medium", "large"))])]
    if total_blobs == 2:
        chosen = [random.choice(AREA_BY_BUCKET["small"]), random.choice(AREA_BY_BUCKET[random.choice(("medium", "large"))])]
        random.shuffle(chosen)
        return chosen
    picked = [random.choice(AREA_BY_BUCKET["small"]), random.choice(AREA_BY_BUCKET["medium"]), random.choice(AREA_BY_BUCKET["large"])]
    remaining = total_blobs - 3
    pool = [area for area in VISIBLE_AREA_POOL if area not in picked]
    random.shuffle(pool)
    picked.extend(pool[:remaining])
    random.shuffle(picked)
    return picked


def _make_tree():
    root = _Obj("frame")
    outside = None
    sides = ("left", "right", "top", "bottom")

    if random.random() < DEPTH1_PROB:
        n_inside = random.randint(*DEPTH1_INSIDE_RANGE)
        want_outside = n_inside < MAX_TOTAL_BLOBS and random.random() < OUTSIDE_BLOB_PROB
        total_blobs = n_inside + int(want_outside)
        areas = _sample_blob_areas(total_blobs)
        area_iter = iter(areas)
        root.children = [_Obj("blob", blob_area=next(area_iter)) for _ in range(n_inside)]
        if want_outside:
            outside = _Obj("blob", outside_side=random.choice(sides), blob_area=next(area_iter))
    else:
        n_inner = random.randint(*DEPTH2_INNER_BLOBS)
        n_outer = random.randint(*DEPTH2_OUTER_BLOBS)
        want_outside = n_inner + n_outer < MAX_TOTAL_BLOBS and random.random() < OUTSIDE_BLOB_PROB
        total_blobs = n_inner + n_outer + int(want_outside)
        areas = _sample_blob_areas(total_blobs)
        area_iter = iter(areas)
        inner = _Obj("frame")
        inner.children = [_Obj("blob", blob_area=next(area_iter)) for _ in range(n_inner)]
        outer_blobs = [_Obj("blob", blob_area=next(area_iter)) for _ in range(n_outer)]
        root.children = [inner] + outer_blobs
        if want_outside:
            outside = _Obj("blob", outside_side=random.choice(sides), blob_area=next(area_iter))
    return root, outside


def _draw(objects, bg, fg):
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


def _blob_areas(node: _Obj) -> list[int]:
    out = []
    if node.kind == "blob":
        assert node.blob_area is not None
        out.append(int(node.blob_area))
    for child in node.children:
        out.extend(_blob_areas(child))
    return out


def _has_size_signal(root: _Obj, outside: _Obj | None) -> bool:
    areas = _blob_areas(root)
    if outside is not None:
        areas.extend(_blob_areas(outside))
    return any(area >= MEDIUM_RANGE[0] for area in areas)


def _has_sort_candidate(node: _Obj) -> bool:
    if len(node.children) >= 2:
        areas = [child.blob_area if child.kind == "blob" else -1 for child in node.children]
        if len(set(areas)) > 1 or any(child.kind == "frame" for child in node.children):
            return True
    return any(child.kind == "frame" and _has_sort_candidate(child) for child in node.children)


def _count_sort_conflicts(node: _Obj) -> int:
    count = 0
    if len(node.children) >= 2:
        expected = sorted(node.children, key=_sort_key_for_output)
        if node.layout_order and node.layout_order != expected:
            count += 1
    for child in node.children:
        if child.kind == "frame":
            count += _count_sort_conflicts(child)
    return count


def _gen_attempt():
    bg, fg = random.sample(range(10), 2)
    root, outside = _make_tree()
    if not _has_size_signal(root, outside):
        return None
    if not _has_sort_candidate(root):
        return None
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
    if _count_sort_conflicts(root) == 0:
        return None

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
    raise RuntimeError("failed to generate valid 2d0172a1 combo input")
