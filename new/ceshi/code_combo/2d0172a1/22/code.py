# Combo variant of 2d0172a1 (index 22)
# combo_spec: D4/0+D5/1+D15/1
# Name: 角落图例指示输出样式+填充矩形框架+十字形色块

from __future__ import annotations

import random

import numpy as np

from arc_utils import find_components, get_bg, label_mask, paste

_CROSS_MASK = np.array(
    [
        [False, True, False],
        [True, True, True],
        [False, True, False],
    ],
    dtype=bool,
)


# Design analysis for combo D4/0 + D5/1 + D15/1
# 1. Concrete behavior and trace
#    - Source family stays the same: one fg color, rough simple closed frames, and small filled blobs.
#      The new part is a corner legend made from two adjacent 3x3 exemplars: a solid 3x3 block and a
#      3x3 cross. A literal single 3x3 patch cannot show both target shapes legibly at once, so the
#      corner strip is 3x7 while each exemplar itself remains 3x3, which preserves the intended
#      “small corner legend” behavior and keeps the hint human-readable.
#    - Output change:
#      * frame: no longer a hollow border. It becomes a filled rectangle.
#      * blob: no longer a single pixel. It becomes a 3x3 cross.
#      * legend: does not appear in the output; it only tells the solver which canonical shapes to use.
#    - Important trace on a tiny example:
#      * original task: one outer frame containing one blob -> output is a hollow 5x5 rectangle with one
#        dot inside.
#      * naive fused version would try to make the outer frame a solid 5x5 block and paste a same-color
#        cross inside. That fails visually because the cross merges into the solid block.
#      * implemented fused version: build the parent as a solid rectangle, but when a child is pasted,
#        first clear a 1-cell bg moat around the child bbox and then paste the child tile. The visible
#        result is still a filled parent rectangle, while the child cross or child filled-rectangle stays
#        readable as a cut-in overlay.
# 2. Complexity calibration
#    - Only one legend vocabulary is used: solid block for frames, cross for blobs. No hidden color table,
#      no per-color branching, and no large parameter grid.
#    - Every generated tree contains at least one blob, so the cross rule is always visible.
#    - Every generated sample contains a frame with at least one child, so the fill-rectangle rule is
#      always visible and never collapses to the original hollow-border behavior.
# 3. Implicit-assumption audit
#    - Old assumption: every fg component belongs to the scene. Broken by D15/1, so `main()` must detect
#      and erase the corner legend before component analysis.
#    - Old assumption: blob canonical footprint is 1x1. Broken by D5/1, so anchors, outside placement,
#      and spacing must all use a 3x3 cross footprint.
#    - Old assumption: frame canonical object is a boolean border mask that can be OR-composed with
#      children. Broken by D4/0 in the monochrome setting. The fused solver therefore switches from pure
#      mask union to tile composition on an integer canvas, where child tiles can overwrite parent fill
#      with background cutouts.
#    - Old assumption: `_layout_root()` can reserve a 3x3 slot around an outside dot. Broken because the
#      outside object is now a tight 3x3 cross tile.
# 4. Placement legality
#    - Sibling layout still uses the original band-order logic, so relative row/column ordering remains
#      recoverable from the input geometry.
#    - Direct children inside a filled frame are placed at the same +2 inset as before; this leaves one
#      fg rim between the outer boundary and the cleared moat around child tiles.
#    - The generator reserves a top or bottom band for the legend so the legend never touches the scene.
# 5. Visual observability
#    - The corner legend explicitly shows the two target shapes the human should look for in the output.
#    - Filled rectangles and crosses are much more distinct than the source task’s borders and dots.
#    - No sorting or branching uses hidden numeric color ids; only visible enclosure and position matter.
# 6. Visual cleanliness
#    - The only overlap mechanism is parent/child layering. Siblings never overlap.
#    - Child overlays carve local bg cutouts before drawing themselves, so nested monochrome objects stay
#      separable instead of collapsing into one solid mass.
# 7. Fusion-behavior checks and new assumptions
#    - The non-degenerate signal is: every sample has the legend, every output differs from the base
#      border+dot canonicalization, and every output contains at least one exact 3x3 cross.
#    - New solver assumption: the legend is an exact corner patch and is unique among the four corners.
#      The generator enforces that by reserving the legend band and rejecting any accidental second match.


LEGEND_H = 3
LEGEND_W = 7
LEGEND_CLEARANCE = 1
LEGEND_CORNERS = ("top_left", "top_right", "bottom_left", "bottom_right")
OUTSIDE_SIDES = ("left", "right", "top", "bottom")

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

# Tree structure schedule
DEPTH1_INSIDE_RANGE = (1, 3)
MAX_TOTAL_CHILDREN = 4
GEN_CASES = (
    ("depth1", False),
    ("depth1", True),
    ("depth2", False),
    ("depth2", True),
)
_GEN_CALL_COUNT = 0

# Step-mix alternation bias
STEP_MIX_ALT_PROB = 0.7

# Generation retry budget
GEN_RETRIES = 500


class _Node:
    def __init__(self, coords: np.ndarray, bbox: tuple[int, int, int, int], center: np.ndarray):
        self.coords = coords
        self.bbox = bbox
        self.center = center
        self.is_frame = False
        self.parent: _Node | None = None
        self.children: list[_Node] = []
        self.tile: np.ndarray | None = None
        self.anchor: tuple[int, int] | None = None
        self.axis_map: dict[str, list[tuple[float, int]]] = {"row": [], "col": []}


class _Obj:
    def __init__(self, kind, children=None, outside_side=None):
        self.kind = kind
        self.children = list(children or [])
        self.mask = None
        self.rel = (0, 0)
        self.pos = (0, 0)
        self.outside_side = outside_side


def _legend_patch(bg: int, fg: int) -> np.ndarray:
    patch = np.full((LEGEND_H, LEGEND_W), int(bg), dtype=int)
    patch[:, :3] = int(fg)
    patch[:, 4:7][_CROSS_MASK] = int(fg)
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


def _find_legend(grid: np.ndarray, bg: int, fg: int) -> tuple[int, int, int, int]:
    legend_bbox, _frame_fill, _blob_tile = _read_legend(grid, bg, fg)
    return legend_bbox


def _parse_legend_patch(patch: np.ndarray, bg: int, fg: int) -> tuple[bool, np.ndarray] | None:
    if patch.shape != (LEGEND_H, LEGEND_W):
        return None
    if not np.all(patch[:, 3] == bg):
        return None

    frame_sample = patch[:, :3]
    blob_sample = patch[:, 4:7]
    if np.any((frame_sample != bg) & (frame_sample != fg)):
        return None
    if np.any((blob_sample != bg) & (blob_sample != fg)):
        return None

    frame_fill = bool(np.all(frame_sample == fg))
    if not frame_fill:
        return None
    blob_mask = blob_sample == fg
    if not blob_mask[1, 1]:
        return None
    lab, n = label_mask(blob_mask, connectivity=4)
    if n != 1 or np.count_nonzero(blob_mask) != 5 or np.array_equal(blob_sample, frame_sample):
        return None
    return frame_fill, blob_sample.copy()


def _read_legend(grid: np.ndarray, bg: int, fg: int) -> tuple[tuple[int, int, int, int], bool, np.ndarray]:
    matches = []
    for corner in LEGEND_CORNERS:
        r0, c0, r1, c1 = _corner_bbox(grid.shape, corner)
        parsed = _parse_legend_patch(grid[r0:r1, c0:c1], bg, fg)
        if parsed is not None:
            frame_fill, blob_tile = parsed
            matches.append(((r0, c0, r1, c1), frame_fill, blob_tile))
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


def _cross_tile(bg: int, fg: int) -> np.ndarray:
    out = np.full((3, 3), int(bg), dtype=int)
    out[_CROSS_MASK] = int(fg)
    return out


def _tile_anchor(tile: np.ndarray, bg: int) -> tuple[int, int]:
    coords = np.argwhere(tile != bg)
    center = coords.mean(axis=0)
    idx = int(np.argmin(np.sum((coords - center) ** 2, axis=1)))
    return tuple(map(int, coords[idx]))


def _build_layout(node: _Node, bg: int, fg: int, frame_fill: bool, blob_tile: np.ndarray) -> None:
    if node.tile is not None:
        return

    if not node.is_frame:
        node.tile = blob_tile.copy()
        node.anchor = _tile_anchor(node.tile, bg)
        node.axis_map = {
            "row": [(float(node.center[0]), node.anchor[0])],
            "col": [(float(node.center[1]), node.anchor[1])],
        }
        return

    assert node.children
    child_canvas, placements, axis_map = _layout_group(node.children, bg, fg, frame_fill, blob_tile)
    h, w = child_canvas.shape
    if frame_fill:
        out = np.full((h + 4, w + 4), int(fg), dtype=int)
        for child, (r, c) in placements.items():
            assert child.tile is not None
            ch, cw = child.tile.shape
            sr = r + 2
            sc = c + 2
            out[sr - 1 : sr + ch + 1, sc - 1 : sc + cw + 1] = int(bg)
            out[sr : sr + ch, sc : sc + cw] = child.tile
    else:
        out = np.full((h + 4, w + 4), int(bg), dtype=int)
        out[0, :] = int(fg)
        out[-1, :] = int(fg)
        out[:, 0] = int(fg)
        out[:, -1] = int(fg)
        out[2 : 2 + h, 2 : 2 + w] = child_canvas

    node.tile = out
    node.anchor = (out.shape[0] // 2, out.shape[1] // 2)
    node.axis_map = {
        "row": [(orig, pos + 2) for orig, pos in axis_map["row"]],
        "col": [(orig, pos + 2) for orig, pos in axis_map["col"]],
    }


def _axis_coordinates(nodes: list[_Node], axis: int) -> dict[_Node, int]:
    items = []
    for node in nodes:
        assert node.tile is not None and node.anchor is not None
        r0, c0, r1, c1 = node.bbox
        interval = (r0, r1) if axis == 0 else (c0, c1)
        mid = (interval[0] + interval[1]) / 2
        items.append((node, interval, mid, node.anchor[axis], node.tile.shape[axis]))

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
    nodes: list[_Node], bg: int, fg: int, frame_fill: bool, blob_tile: np.ndarray
) -> tuple[np.ndarray, dict[_Node, tuple[int, int]], dict[str, list[tuple[float, int]]]]:
    assert nodes
    for node in nodes:
        _build_layout(node, bg, fg, frame_fill, blob_tile)

    row_coord = _axis_coordinates(nodes, axis=0)
    col_coord = _axis_coordinates(nodes, axis=1)

    min_r = min(row_coord[n] - n.anchor[0] for n in nodes if n.anchor is not None)
    min_c = min(col_coord[n] - n.anchor[1] for n in nodes if n.anchor is not None)
    max_r = max(
        row_coord[n] - n.anchor[0] - min_r + n.tile.shape[0]
        for n in nodes
        if n.anchor is not None and n.tile is not None
    )
    max_c = max(
        col_coord[n] - n.anchor[1] - min_c + n.tile.shape[1]
        for n in nodes
        if n.anchor is not None and n.tile is not None
    )

    out = np.full((max_r, max_c), int(bg), dtype=int)
    placements: dict[_Node, tuple[int, int]] = {}
    axis_map = {"row": [], "col": []}
    for node in nodes:
        assert node.tile is not None and node.anchor is not None
        r = row_coord[node] - node.anchor[0] - min_r
        c = col_coord[node] - node.anchor[1] - min_c
        placements[node] = (int(r), int(c))
        out[r : r + node.tile.shape[0], c : c + node.tile.shape[1]] = node.tile
        axis_map["row"].append((float(node.center[0]), int(r + node.anchor[0])))
        axis_map["col"].append((float(node.center[1]), int(c + node.anchor[1])))
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


def _layout_root(
    nodes: list[_Node], bg: int, fg: int, frame_fill: bool, blob_tile: np.ndarray
) -> np.ndarray:
    assert nodes
    for node in nodes:
        _build_layout(node, bg, fg, frame_fill, blob_tile)

    frames = [node for node in nodes if node.is_frame]
    blobs = [node for node in nodes if not node.is_frame]
    if len(frames) == 1 and blobs:
        frame = frames[0]
        assert frame.tile is not None
        placements: dict[_Node, tuple[int, int]] = {frame: (0, 0)}
        extents = [(0, 0, frame.tile.shape[0], frame.tile.shape[1])]

        for blob in blobs:
            assert blob.tile is not None and blob.anchor is not None
            side = _outside_side(blob, frame)
            if side in {"right", "left"}:
                r = _map_frame_axis(frame, 0, float(blob.center[0])) - blob.anchor[0]
                c = frame.tile.shape[1] + 1 if side == "right" else -blob.tile.shape[1] - 1
            else:
                r = frame.tile.shape[0] + 1 if side == "bottom" else -blob.tile.shape[0] - 1
                c = _map_frame_axis(frame, 1, float(blob.center[1])) - blob.anchor[1]
            placements[blob] = (int(r), int(c))
            extents.append((r, c, r + blob.tile.shape[0], c + blob.tile.shape[1]))

        min_r = min(e[0] for e in extents)
        min_c = min(e[1] for e in extents)
        max_r = max(e[2] for e in extents)
        max_c = max(e[3] for e in extents)
        out = np.full((max_r - min_r, max_c - min_c), int(bg), dtype=int)
        for node, (r, c) in placements.items():
            rr = r - min_r
            cc = c - min_c
            assert node.tile is not None
            out[rr : rr + node.tile.shape[0], cc : cc + node.tile.shape[1]] = node.tile
        return out

    canvas, _placements, _axis_map = _layout_group(nodes, bg, fg, frame_fill, blob_tile)
    return canvas


def main(grid: np.ndarray) -> np.ndarray:
    """
    Redraw a rough nested frame-and-blob sketch using the two target shapes shown in the corner legend.

    Input structure:
    - The grid uses one background color and one foreground color for the scene.
    - One corner contains a tiny legend strip made from two 3x3 samples: a solid block and a cross.
    - Outside that legend, foreground components are rough closed frames or compact filled blobs, with the
      same nesting and outside-blob semantics as the source task.

    Rule:
    - Read and erase the corner legend before analyzing the main scene.
    - Recover the containment tree from frame holes exactly as in the source task.
    - Replace each blob with a 3x3 cross, and replace each frame with a filled rectangle that preserves the
      same child ordering; direct children are overlaid by carving a local bg moat and then drawing the child.
    - Keep sibling row/column order and the side of any outside blob relative to the outermost frame.

    Key insight: The legend changes the canonical drawing language, but the structural reasoning is still
    “ignore wobble, keep enclosure and sibling order”; the only extra care is that monochrome filled frames
    must be composed as layered tiles rather than simple mask unions.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    fg_colors = sorted(set(map(int, g.flatten())) - {bg})
    assert len(fg_colors) == 1
    fg = int(fg_colors[0])

    (lr0, lc0, lr1, lc1), frame_fill, blob_tile = _read_legend(g, bg, fg)
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
    return _layout_root(roots, bg, fg, frame_fill, blob_tile)


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
    return _layout_root(roots, bg, fg, True, _cross_tile(bg, fg))


def _make_tree(branch: str, with_outside: bool, token: int):
    root = _Obj("frame")
    outside = None
    if branch == "depth1":
        max_inside = min(DEPTH1_INSIDE_RANGE[1], MAX_TOTAL_CHILDREN - int(with_outside))
        n_inside = 1 + (token % max_inside)
        root.children = [_Obj("blob") for _ in range(n_inside)]
    else:
        inner = _Obj("frame")
        inner.children = [_Obj("blob")]
        root.children = [inner]
    if with_outside:
        outside = _Obj("blob", outside_side=OUTSIDE_SIDES[token % len(OUTSIDE_SIDES)])
    return root, outside


def _draw(objects, bg, fg, corner):
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

    legend = _legend_patch(bg, fg)
    lr0, lc0, lr1, lc1 = _corner_bbox(grid.shape, corner)
    if np.any(grid[lr0:lr1, lc0:lc1] != bg):
        return None
    grid[lr0:lr1, lc0:lc1] = legend

    if sum(np.array_equal(grid[a:b, c:d], legend) for a, c, b, d in (_corner_bbox(grid.shape, x) for x in LEGEND_CORNERS)) != 1:
        return None
    if np.count_nonzero(grid == bg) <= np.count_nonzero(grid == fg):
        return None
    return grid


def _gen_attempt(branch: str, with_outside: bool, corner: str, token: int):
    bg, fg = random.sample(range(10), 2)
    root, outside = _make_tree(branch, with_outside, token)
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

    grid = _draw(objects, bg, fg, corner)
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
    global _GEN_CALL_COUNT
    branch, with_outside = GEN_CASES[_GEN_CALL_COUNT % len(GEN_CASES)]
    corner = LEGEND_CORNERS[(_GEN_CALL_COUNT // len(GEN_CASES)) % len(LEGEND_CORNERS)]
    for _ in range(GEN_RETRIES):
        token = _GEN_CALL_COUNT + _
        grid = _gen_attempt(branch=branch, with_outside=with_outside, corner=corner, token=token)
        if grid is not None:
            _GEN_CALL_COUNT += 1
            return grid
    raise RuntimeError("failed to generate valid 2d0172a1 combo input")
