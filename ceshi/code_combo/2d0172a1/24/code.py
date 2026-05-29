# Combo variant of 2d0172a1 (index 24)
# combo_spec: D2/0+D16/2
# Name: 单像素色块+先去噪再规范化
#
# Design analysis for this combo variant
# - concretization:
#   The wobbly closed frames remain, but every semantic blob has collapsed to a single foreground pixel.
#   The input also contains 1-2 extra isolated foreground singletons. At least one such speck is injected
#   into a frame hole that already contains several true singleton blobs, so the solver must distinguish
#   "real child dot" from "noise dot" using relative placement rather than size.
# - step trace:
#   On a frame hole containing true dots at, say, (4,4) and (8,4), plus a noise speck at (6,7), the base
#   task would have used component size to recognize the blobs; the D16-only variant would delete *all*
#   singletons; neither is valid now. The fused pipeline is:
#   1) detect frames and partition singleton candidates by the deepest frame hole or the outside region;
#   2) inside each region, keep only the largest singleton subset that can still realize a legal sibling
#      arrangement together with any direct child frames; discard the rest as noise;
#   3) canonicalize the surviving tree exactly as the original task does.
# - complexity calibration:
#   The generator keeps the original small-tree feel (1-2 frames, 2-4 semantic dots, optional outside dot),
#   but every noise-bearing frame region is guaranteed to contain at least two true dots, and the legal
#   layouts stay in the original three families: vertical, horizontal, or diagonal. That keeps the new rule
#   visible without creating a combinatorial mess for humans or for the solver.
# - implicit assumptions audited:
#   The original `main()` assumed every non-frame foreground component was semantic, and the D16-only solver
#   assumed every singleton component was noise. Both assumptions fail here. The code paths that enumerate
#   components, attach children, and choose roots now rely on a new intermediate step: infer which singleton
#   candidates are consistent with the generator's legal layout grammar before building the final semantic
#   tree. `_build_layout()` itself remains valid because semantic dots still canonicalize to 1x1 masks.
# - placement legality:
#   Frames are still rendered as non-self-intersecting 4-connected loops with one hole. Semantic dots are
#   1x1 objects arranged by the same output-first layout as before. Noise is only injected onto background
#   cells whose 8-neighborhood is empty, and the generator validates each noisy sample with the same region
#   selector used by `main()`, so noise never merges with a frame/dot and never steals a semantic role.
# - observability and cleanliness:
#   Human solvers can now see two kinds of singleton pixels in the same color: those that line up with the
#   frame hierarchy survive as canonical dots, while off-pattern specks disappear. The sample distribution is
#   intentionally sparse: 1-2 noise pixels total, generous gaps, and no color remapping.
# - generator guarantees for the new assumptions:
#   `main()` now assumes every non-frame component is a singleton and every semantic singleton belongs either
#   to the deepest containing frame hole or to a valid outside slot of the outermost frame. The generator
#   enforces those assumptions and additionally checks that the subset selector recovers the exact true dot
#   set before accepting a sample.

from __future__ import annotations

import itertools
import random

import numpy as np

from arc_utils import find_components, get_bg, label_mask, paste


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


def _make_node(coords: np.ndarray) -> _Node:
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    return _Node(
        coords=coords.astype(int),
        bbox=(int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])),
        center=coords.mean(axis=0),
    )


def _node_shape(node: _Node) -> tuple[int, int]:
    r0, c0, r1, c1 = node.bbox
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _vertical_score(nodes: list[_Node]) -> tuple[int, int, int, int] | None:
    ordered = sorted(nodes, key=lambda node: (node.bbox[0], node.bbox[1], node.bbox[2], node.bbox[3]))
    gaps = []
    for prev, curr in zip(ordered, ordered[1:]):
        prev_h, _ = _node_shape(prev)
        gap = int(curr.bbox[0] - (prev.bbox[0] + prev_h))
        if not 2 <= gap <= 5:
            return None
        gaps.append(gap)

    min_c = min(node.bbox[1] for node in ordered)
    span_w = max(node.bbox[3] + 1 for node in ordered) - min_c
    max_w = max(_node_shape(node)[1] for node in ordered)
    slack = int(max_w + 4 - span_w)
    if slack < 0:
        return None

    doubled_centers = [int(2 * node.bbox[1] + _node_shape(node)[1]) for node in ordered]
    spread = max(doubled_centers) - min(doubled_centers)
    return slack, -spread, -sum(gaps), 0


def _horizontal_score(nodes: list[_Node]) -> tuple[int, int, int, int] | None:
    ordered = sorted(nodes, key=lambda node: (node.bbox[1], node.bbox[0], node.bbox[2], node.bbox[3]))
    gaps = []
    for prev, curr in zip(ordered, ordered[1:]):
        _, prev_w = _node_shape(prev)
        gap = int(curr.bbox[1] - (prev.bbox[1] + prev_w))
        if not 2 <= gap <= 5:
            return None
        gaps.append(gap)

    min_r = min(node.bbox[0] for node in ordered)
    span_h = max(node.bbox[2] + 1 for node in ordered) - min_r
    max_h = max(_node_shape(node)[0] for node in ordered)
    slack = int(max_h + 4 - span_h)
    if slack < 0:
        return None

    doubled_centers = [int(2 * node.bbox[0] + _node_shape(node)[0]) for node in ordered]
    spread = max(doubled_centers) - min(doubled_centers)
    return slack, -spread, -sum(gaps), -1


def _diagonal_score(nodes: list[_Node]) -> tuple[int, int, int, int] | None:
    by_row = sorted(nodes, key=lambda node: (node.bbox[0], node.bbox[1], node.bbox[2], node.bbox[3]))
    by_col = sorted(nodes, key=lambda node: (node.bbox[1], node.bbox[0], node.bbox[2], node.bbox[3]))
    if any(a is not b for a, b in zip(by_row, by_col)):
        return None

    gaps = []
    for prev, curr in zip(by_row, by_row[1:]):
        prev_h, prev_w = _node_shape(prev)
        row_gap = int(curr.bbox[0] - (prev.bbox[0] + prev_h))
        col_gap = int(curr.bbox[1] - (prev.bbox[1] + prev_w))
        if not (2 <= row_gap <= 5 and row_gap == col_gap):
            return None
        gaps.append(row_gap)
    return 0, 0, -sum(gaps), 1


def _group_fit_score(nodes: list[_Node]) -> tuple[int, int, int, int] | None:
    if not nodes:
        return None
    if len(nodes) == 1:
        return 0, 0, 0, 0

    scores = []
    for scorer in (_vertical_score, _horizontal_score, _diagonal_score):
        score = scorer(nodes)
        if score is not None:
            scores.append(score)
    if not scores:
        return None
    return max(scores)


def _best_frame_dot_subset(frame_children: list[_Node], dot_candidates: list[_Node]) -> tuple[list[_Node], int]:
    ordered = sorted(dot_candidates, key=lambda node: (node.bbox[0], node.bbox[1]))
    best_subset: tuple[_Node, ...] = ()
    best_key: tuple[int, int, int, int, int] | None = None
    best_ties = 0

    for size in range(len(ordered), -1, -1):
        found_for_size = False
        for subset in itertools.combinations(ordered, size):
            group = frame_children + list(subset)
            if not group:
                continue
            fit = _group_fit_score(group)
            if fit is None:
                continue
            found_for_size = True
            key = (size,) + fit
            if best_key is None or key > best_key:
                best_key = key
                best_subset = subset
                best_ties = 1
            elif key == best_key:
                best_ties += 1
        if found_for_size:
            break

    return list(best_subset), best_ties


def _outside_dot_score(frame: _Node, dot: _Node) -> tuple[int, int, int] | None:
    r = int(dot.coords[0, 0])
    c = int(dot.coords[0, 1])
    fr0, fc0, fr1, fc1 = frame.bbox
    scores = []

    if fr0 < r < fr1:
        right_gap = c - fc1 - 1
        left_gap = fc0 - c - 1
        if 2 <= right_gap <= 5:
            scores.append((5 - right_gap, -abs(2 * r - (fr0 + fr1)), 1))
        if 2 <= left_gap <= 5:
            scores.append((5 - left_gap, -abs(2 * r - (fr0 + fr1)), 0))
    if fc0 < c < fc1:
        bottom_gap = r - fr1 - 1
        top_gap = fr0 - r - 1
        if 2 <= bottom_gap <= 5:
            scores.append((5 - bottom_gap, -abs(2 * c - (fc0 + fc1)), 3))
        if 2 <= top_gap <= 5:
            scores.append((5 - top_gap, -abs(2 * c - (fc0 + fc1)), 2))

    if not scores:
        return None
    return max(scores)


def _select_outside_dots(root_frame: _Node, outside_candidates: list[_Node]) -> tuple[list[_Node], int]:
    best_node: _Node | None = None
    best_score: tuple[int, int, int] | None = None
    ties = 0
    for node in outside_candidates:
        score = _outside_dot_score(root_frame, node)
        if score is None:
            continue
        if best_score is None or score > best_score:
            best_node = node
            best_score = score
            ties = 1
        elif score == best_score:
            ties += 1
    return ([] if best_node is None else [best_node]), ties


def _extract_semantic_structure(grid: np.ndarray, bg: int, fg: int) -> tuple[list[_Node], list[_Node]]:
    comps = find_components(grid, bg=bg, connectivity=4, monochromatic=True, colors=[fg], return_format="dict")
    assert comps

    frame_nodes: list[_Node] = []
    singleton_nodes: list[_Node] = []
    holes_by_frame: dict[_Node, list[np.ndarray]] = {}
    for comp in comps:
        coords = np.asarray(comp["coords"], dtype=int)
        node = _make_node(coords)
        node_holes = _component_holes(grid.shape, coords)
        if node_holes:
            node.is_frame = True
            frame_nodes.append(node)
            holes_by_frame[node] = node_holes
        else:
            assert coords.shape[0] == 1
            singleton_nodes.append(node)

    assert frame_nodes
    for node_i, node in enumerate(frame_nodes):
        containers = []
        for frame_i, frame in enumerate(frame_nodes):
            if node_i == frame_i:
                continue
            if any(np.all(hole[tuple(node.coords.T)]) for hole in holes_by_frame[frame]):
                containers.append(frame)
        if containers:
            node.parent = min(containers, key=lambda item: int(item.coords.shape[0]))
            node.parent.children.append(node)

    root_frames = [node for node in frame_nodes if node.parent is None]
    assert len(root_frames) == 1
    root_frame = root_frames[0]

    candidates_by_frame = {frame: [] for frame in frame_nodes}
    outside_candidates: list[_Node] = []
    for node in singleton_nodes:
        r = int(node.coords[0, 0])
        c = int(node.coords[0, 1])
        containers = [frame for frame in frame_nodes if any(hole[r, c] for hole in holes_by_frame[frame])]
        if containers:
            host = min(containers, key=lambda item: int(item.coords.shape[0]))
            candidates_by_frame[host].append(node)
        else:
            outside_candidates.append(node)

    selected_dots: list[_Node] = []
    selected_by_frame: dict[_Node, list[_Node]] = {}
    for frame in sorted(frame_nodes, key=lambda node: int(node.coords.shape[0])):
        chosen, _ties = _best_frame_dot_subset(list(frame.children), candidates_by_frame[frame])
        selected_by_frame[frame] = chosen
        selected_dots.extend(chosen)

    outside_selected, _outside_ties = _select_outside_dots(root_frame, outside_candidates)
    selected_dots.extend(outside_selected)

    for frame in frame_nodes:
        frame.children.extend(selected_by_frame[frame])
        for dot in selected_by_frame[frame]:
            dot.parent = frame

    roots = [root_frame] + outside_selected
    return roots, selected_dots


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove off-pattern singleton specks, then normalize a rough nested frame-and-dot sketch.

    Input structure:
    - The grid uses one background color and one foreground color.
    - Foreground components are wobbly closed frames plus many singleton pixels.
    - Some singleton pixels are real blob-children that belong to the frame hierarchy, while 1-2 others are stray noise.

    Rule:
    - First detect all frame components and split singleton candidates by region: inside the deepest frame hole or outside the outer frame.
    - In each frame region, keep only the largest singleton subset that still forms a legal sibling arrangement with that frame's direct child frames; discard the remaining singleton specks as noise.
    - Keep at most one outside singleton if it sits in a legal outside-blob slot next to the outer frame.
    - Canonicalize the surviving semantic tree exactly as in the base task: frames become rectangular borders and semantic dots remain single pixels.

    Key insight: Denoising can no longer rely on component size, because true blobs and noise are both single pixels; the separator is whether a singleton participates in the same positional grammar as the frame hierarchy.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    fg_colors = sorted(set(map(int, g.flatten())) - {bg})
    assert len(fg_colors) == 1
    fg = int(fg_colors[0])

    roots, _selected_dots = _extract_semantic_structure(g, bg=bg, fg=fg)
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

# Frame geometry
FRAME_H_RANGE = (7, 20)
FRAME_W_RANGE = (7, 24)
FRAME_CORNER_MAX = 5
FRAME_MIN_EDGE = 2
FRAME_TURN_RATIO = (0.34, 0.43)
FRAME_RETRIES = 60

# Layout / arrangement
ARRANGE_GAP_COMPACT = (2, 3)
ARRANGE_GAP_NORMAL = (2, 5)
ARRANGE_EXTRA_SPAN = (0, 4)
FRAME_MARGIN_RANGE = (3, 5)
COMPACT_MARGIN = 3

# Outside dot placement
OUTSIDE_GAP_RANGE = (2, 5)

# Noise injection
NOISE_COUNT_RANGE = (1, 2)
NOISE_TRIES = 120

# Tree structure probabilities
DEPTH1_PROB = 0.4
OUTSIDE_DOT_PROB = 0.35
DEPTH1_INSIDE_RANGE = (2, 4)
DEPTH2_INNER_BLOBS = (2, 3)
DEPTH2_OUTER_BLOBS = (0, 2)

# Step-mix alternation bias
STEP_MIX_ALT_PROB = 0.7

# Generation retry budget
GEN_RETRIES = 400


class _Obj:
    def __init__(self, kind, children=None, outside_side=None, layout_mode=None):
        self.kind = kind
        self.children = list(children or [])
        self.mask: np.ndarray | None = None
        self.rel = (0, 0)
        self.pos = (0, 0)
        self.outside_side = outside_side
        self.layout_mode = layout_mode


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


def _dot_mask() -> np.ndarray:
    return np.ones((1, 1), dtype=bool)


def _arrange(
    children: list[_Obj], compact: bool = False, preferred_mode: str | None = None
) -> tuple[dict[_Obj, tuple[int, int]], int, int]:
    random.shuffle(children)
    dims = [child.mask.shape for child in children]
    k = len(children)
    if k == 1:
        return {children[0]: (0, 0)}, dims[0][0], dims[0][1]

    mode = preferred_mode or random.choice(("vertical", "horizontal", "diagonal"))
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
    if node.kind == "dot":
        node.mask = _dot_mask()
        return True

    for child in node.children:
        if not _prepare(child, compact=child.kind == "frame"):
            return False

    has_frame_child = any(child.kind == "frame" for child in node.children)
    tight = compact or has_frame_child
    placements, group_h, group_w = _arrange(
        node.children, compact=tight, preferred_mode=node.layout_mode
    )
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
    gap = random.randint(*OUTSIDE_GAP_RANGE)
    side = outside.outside_side
    if side == "right":
        outside.rel = (random.randint(1, h - 2), w + gap)
    elif side == "left":
        outside.rel = (random.randint(1, h - 2), -gap - 1)
    elif side == "bottom":
        outside.rel = (h + gap, random.randint(1, w - 2))
    else:
        outside.rel = (-gap - 1, random.randint(1, w - 2))


def _node_from_obj(obj: _Obj) -> _Node:
    assert obj.mask is not None
    r0, c0 = obj.pos
    coords = np.argwhere(obj.mask) + np.array([r0, c0])
    node = _make_node(coords.astype(int))
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


def _direct_blob_frames(root: _Obj) -> list[_Obj]:
    out = []

    def walk(node: _Obj) -> None:
        if node.kind == "frame" and sum(child.kind == "dot" for child in node.children) >= 2:
            out.append(node)
        for child in node.children:
            walk(child)

    walk(root)
    return out


def _frame_direct_region(shape: tuple[int, int], frame_obj: _Obj) -> np.ndarray:
    assert frame_obj.mask is not None
    coords = np.argwhere(frame_obj.mask) + np.array(frame_obj.pos, dtype=int)
    holes = _component_holes(shape, coords.astype(int))
    assert len(holes) == 1
    region = np.array(holes[0], copy=True)
    for child in frame_obj.children:
        if child.kind != "frame":
            continue
        child_coords = np.argwhere(child.mask) + np.array(child.pos, dtype=int)
        for hole in _component_holes(shape, child_coords.astype(int)):
            region &= ~hole
    return region


def _isolated_bg_candidates(grid: np.ndarray, bg: int, region: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    candidates = []
    for r, c in np.argwhere(region & (grid == bg)):
        r = int(r)
        c = int(c)
        ok = True
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w and grid[rr, cc] != bg:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            candidates.append((r, c))
    return candidates


def _true_dot_positions(root: _Obj, outside: _Obj | None) -> set[tuple[int, int]]:
    positions = set()

    def walk(node: _Obj) -> None:
        if node.kind == "dot":
            positions.add((int(node.pos[0]), int(node.pos[1])))
        for child in node.children:
            walk(child)

    walk(root)
    if outside is not None:
        positions.add((int(outside.pos[0]), int(outside.pos[1])))
    return positions


def _selected_dot_positions(grid: np.ndarray, bg: int, fg: int) -> set[tuple[int, int]] | None:
    try:
        _roots, dots = _extract_semantic_structure(grid, bg=bg, fg=fg)
    except Exception:
        return None
    return {tuple(map(int, dot.coords[0])) for dot in dots}


_BRANCH_PLAN_SEQUENCE = [
    {"depth": 1, "outside": False, "layout": "vertical"},
    {"depth": 1, "outside": False, "layout": "horizontal"},
    {"depth": 1, "outside": True, "layout": "diagonal"},
    {"depth": 2, "outside": False, "layout": "vertical"},
    {"depth": 2, "outside": False, "layout": "horizontal"},
    {"depth": 2, "outside": True, "layout": "diagonal"},
]
_branch_plan_index = 0


def _next_branch_plan() -> dict[str, object]:
    global _branch_plan_index
    plan = _BRANCH_PLAN_SEQUENCE[_branch_plan_index % len(_BRANCH_PLAN_SEQUENCE)]
    _branch_plan_index += 1
    return dict(plan)


def _inject_noise(grid: np.ndarray, root: _Obj, outside: _Obj | None, bg: int, fg: int) -> np.ndarray | None:
    true_dots = _true_dot_positions(root, outside)
    hosts = _direct_blob_frames(root)
    if not hosts:
        return None

    noise_target = random.randint(*NOISE_COUNT_RANGE)
    noisy = np.array(grid, copy=True)
    placed = []

    first_host = random.choice(hosts)
    first_region = _frame_direct_region(noisy.shape, first_host)
    first_candidates = _isolated_bg_candidates(noisy, bg=bg, region=first_region)
    random.shuffle(first_candidates)
    for pos in first_candidates[:NOISE_TRIES]:
        trial = np.array(noisy, copy=True)
        trial[pos] = fg
        if _selected_dot_positions(trial, bg=bg, fg=fg) == true_dots:
            noisy = trial
            placed.append(pos)
            break
    if not placed:
        return None

    all_regions = [_frame_direct_region(noisy.shape, host) for host in hosts]
    extra_region = np.logical_or.reduce(all_regions + [np.ones(noisy.shape, dtype=bool)])
    for _ in range(noise_target - 1):
        candidates = _isolated_bg_candidates(noisy, bg=bg, region=extra_region)
        random.shuffle(candidates)
        accepted = False
        for pos in candidates[:NOISE_TRIES]:
            trial = np.array(noisy, copy=True)
            trial[pos] = fg
            if _selected_dot_positions(trial, bg=bg, fg=fg) == true_dots:
                noisy = trial
                placed.append(pos)
                accepted = True
                break
        if not accepted:
            return None
    return noisy


def _make_tree(
    depth_mode: int | None = None,
    want_outside: bool | None = None,
    layout_mode: str | None = None,
) -> tuple[_Obj, _Obj | None]:
    if depth_mode is None:
        depth_mode = 1 if random.random() < DEPTH1_PROB else 2
    if want_outside is None:
        want_outside = random.random() < OUTSIDE_DOT_PROB

    root = _Obj("frame", layout_mode=layout_mode if depth_mode == 1 else None)
    outside = None
    sides = ("left", "right", "top", "bottom")
    if depth_mode == 1:
        inside_hi = min(3, DEPTH1_INSIDE_RANGE[1]) if want_outside else DEPTH1_INSIDE_RANGE[1]
        n_inside = random.randint(DEPTH1_INSIDE_RANGE[0], inside_hi)
        root.children = [_Obj("dot") for _ in range(n_inside)]
        if want_outside:
            outside = _Obj("dot", outside_side=random.choice(sides))
    else:
        inner_n = random.randint(*DEPTH2_INNER_BLOBS)
        max_outer = 0 if want_outside else (1 if inner_n == 3 else 2)
        outer_n = random.randint(0, max_outer)
        inner = _Obj("frame", layout_mode=layout_mode)
        inner.children = [_Obj("dot") for _ in range(inner_n)]
        root.children = [inner] + [_Obj("dot") for _ in range(outer_n)]
        if want_outside:
            outside = _Obj("dot", outside_side=random.choice(sides))
    return root, outside


def _draw(objects: list[_Obj], bg: int, fg: int) -> np.ndarray | None:
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


def _gen_attempt(plan: dict[str, object] | None = None) -> np.ndarray | None:
    bg, fg = random.sample(range(10), 2)
    if plan is None:
        root, outside = _make_tree()
    else:
        root, outside = _make_tree(
            depth_mode=int(plan["depth"]),
            want_outside=bool(plan["outside"]),
            layout_mode=str(plan["layout"]),
        )
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

    noisy_grid = _inject_noise(grid, root, outside, bg=bg, fg=fg)
    if noisy_grid is None:
        return None

    expected = _expected_output(root, outside, bg, fg)
    try:
        got = main(noisy_grid)
    except Exception:
        return None
    if not np.array_equal(got, expected):
        return None
    return noisy_grid


def generate_input() -> np.ndarray:
    plan = _next_branch_plan()
    focused_tries = max(1, GEN_RETRIES // 2)
    for _ in range(focused_tries):
        grid = _gen_attempt(plan)
        if grid is not None:
            return grid
    for _ in range(GEN_RETRIES - focused_tries):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 2d0172a1 combo input")
