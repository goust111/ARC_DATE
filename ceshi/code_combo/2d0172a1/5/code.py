# Combo variant of 2d0172a1 (index 5)
# combo_spec: D3/2+D16/2
# Name: 有缺口框架+先去噪再规范化
#
# Design analysis for this combo variant
# - concretization:
#   The base task shows one foreground drawing made of wobbly closed frames plus small filled blobs.
#   This combo changes two stages at once.
#   1. Each semantic frame is drawn with one or two tiny gaps on straight side segments. Every gap is
#      contiguous and 1-2 pixels wide, so the visible component is an almost-closed open path rather than
#      a strict loop.
#   2. The grid also contains 1-2 isolated singleton foreground pixels that are pure noise.
#   A concrete trace:
#     old pipeline on {outer frame, inner blob, singleton noise inside the enclosure} assumes every
#     foreground component is semantic and every frame already has a true hole. With a gap, the outer
#     component has no strict hole, and with the singleton still present the parser would either miss the
#     frame entirely or keep an extra fake blob. The combo pipeline instead deletes singleton components
#     first, then repairs each large near-loop by trying to fill back 1-2 missing adjacent pixels, and
#     only after that reconstructs the containment tree and canonical rectangles.
# - complexity calibration:
#   The two changes are both visually obvious but tightly controlled. Gaps are only 1-2 pixels wide and
#   always contiguous, which keeps the frame as a single readable object and avoids exploding the search
#   space into arbitrary broken scribbles. Noise is limited to 1-2 singleton pixels and at least one lies
#   inside a semantic frame interior, so the denoising step is consistently observable without clutter.
# - implicit assumption audit:
#   The original main() relied on three hidden assumptions:
#   1. every foreground component is semantic;
#   2. every semantic frame already contains a strict hole;
#   3. hole-based containment can be computed directly from raw components.
#   In this combo, (1) fails because of singleton noise, and (2)(3) fail because gapped frames have no
#   literal hole until repaired. The layout code remains valid once parsing restores the intended tree, so
#   only the preprocessing / frame classification stage needs to change.
# - placement legality:
#   The generator still builds a valid semantic tree first. It places children inside a closed wobbly frame,
#   then removes a 1-2 pixel contiguous segment from a straight run of the frame border. This preserves one
#   connected component, keeps the intended repaired hole well-defined, and never causes child overlap.
#   Noise is injected only onto 8-isolated background cells, so it cannot merge with any real object.
# - visual observability:
#   Across a few examples, a human can see both effects: tiny specks disappear, and slightly broken frame
#   outlines are still normalized into perfect rectangles. No hidden color-id logic is used; the evidence
#   is entirely geometric and spatial.
# - visual tidiness:
#   Restricting each frame to one or two short contiguous gaps keeps the drawing legible. Children preserve
#   the original sibling band structure, and isolated noise pixels stay sparse, so the scene still reads as
#   one coherent sketch rather than overlapping annotations.
# - color / rule inference:
#   Colors remain exactly bg + fg. The new reasoning burden is sequential and structural only: remove stray
#   specks, infer near-closed frames, then canonicalize as usual.
# - generator guarantees for new assumptions:
#   The repaired-frame detector in main() assumes that any semantic open frame differs from a valid simple
#   loop by at most two tiny straight-run gaps, each 1-2 pixels wide, and that non-frame blobs remain much
#   smaller than frames. generate_input() enforces that by carving one or two such gaps from straight border
#   runs and by keeping blobs in the original 4-11 pixel / 2-4 bbox regime.

from __future__ import annotations

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


def _remove_singleton_noise(grid: np.ndarray, bg: int, fg: int) -> np.ndarray:
    """Delete isolated 1-pixel foreground components before semantic parsing."""
    out = np.array(grid, copy=True)
    comps = find_components(out, bg=bg, connectivity=4, monochromatic=True, colors=[fg], return_format="dict")
    for comp in comps:
        coords = np.asarray(comp["coords"], dtype=int)
        if coords.shape[0] == 1:
            r, c = map(int, coords[0])
            out[r, c] = bg
    return out


def _loop_ok(mask: np.ndarray) -> bool:
    """True iff mask is one simple 4-connected loop with exactly one hole."""
    lab, n = label_mask(mask, connectivity=4)
    if n != 1:
        return False
    deg = np.zeros_like(mask, dtype=int)
    deg[1:, :] += mask[:-1, :]
    deg[:-1, :] += mask[1:, :]
    deg[:, 1:] += mask[:, :-1]
    deg[:, :-1] += mask[:, 1:]
    if not np.all(deg[mask] == 2):
        return False
    return len(_component_holes(mask.shape, np.argwhere(mask))) == 1


def _gap_components_ok(cells: set[tuple[int, int]]) -> bool:
    """True iff missing cells form one or two straight 1-2 pixel gap segments."""
    if not cells:
        return False
    pending = set(cells)
    n_comp = 0
    while pending:
        n_comp += 1
        if n_comp > 2:
            return False
        stack = [pending.pop()]
        comp = []
        while stack:
            cell = stack.pop()
            comp.append(cell)
            r, c = cell
            for nbr in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if nbr in pending:
                    pending.remove(nbr)
                    stack.append(nbr)
        if len(comp) > 2:
            return False
        if len(comp) == 2:
            (r0, c0), (r1, c1) = comp
            if abs(r0 - r1) + abs(c0 - c1) != 1:
                return False
            if r0 != r1 and c0 != c1:
                return False
    return True


def _candidate_gap_groups(candidates: list[tuple[int, int]]) -> list[tuple[tuple[int, int], ...]]:
    """Enumerate 1- and 2-cell straight candidate gap segments."""
    out = [((r, c),) for r, c in candidates]
    candidate_set = set(candidates)
    for r, c in candidates:
        if (r + 1, c) in candidate_set:
            out.append(((r, c), (r + 1, c)))
        if (r, c + 1) in candidate_set:
            out.append(((r, c), (r, c + 1)))
    return out


def _globalize_local_mask(shape: tuple[int, int], local_mask: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Embed a local padded mask back into a global-shape mask."""
    out = np.zeros(shape, dtype=bool)
    inner = local_mask[1:-1, 1:-1]
    r0, c0 = map(int, origin)
    out[r0 : r0 + inner.shape[0], c0 : c0 + inner.shape[1]] = inner
    return out


def _frame_hole(shape: tuple[int, int], coords: np.ndarray) -> np.ndarray | None:
    """
    Return the repaired/global hole mask for a frame component.

    Closed frames are accepted directly. Open near-frames are repaired by filling back
    one or two tiny gap segments, each of width 1-2, inside the bbox plus one-cell padding.
    """
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    h = int(hi[0] - lo[0] + 1)
    w = int(hi[1] - lo[1] + 1)
    if min(h, w) < 7 or coords.shape[0] < 18:
        return None

    local = np.zeros((h + 2, w + 2), dtype=bool)
    shifted = coords - lo + 1
    local[tuple(shifted.T)] = True

    if _loop_ok(local):
        hole = _component_holes(local.shape, np.argwhere(local))[0]
        return _globalize_local_mask(shape, hole, lo)

    candidates: list[tuple[int, int]] = []
    for r in range(1, h + 1):
        for c in range(1, w + 1):
            if local[r, c]:
                continue
            deg4 = int(local[r - 1, c]) + int(local[r + 1, c]) + int(local[r, c - 1]) + int(local[r, c + 1])
            if 1 <= deg4 <= 2:
                candidates.append((r, c))

    for r, c in candidates:
        trial = np.array(local, copy=True)
        trial[r, c] = True
        if _loop_ok(trial):
            hole = _component_holes(trial.shape, np.argwhere(trial))[0]
            return _globalize_local_mask(shape, hole, lo)

    groups = _candidate_gap_groups(candidates)
    for group in groups:
        trial = np.array(local, copy=True)
        for r, c in group:
            trial[r, c] = True
        if _loop_ok(trial):
            hole = _component_holes(trial.shape, np.argwhere(trial))[0]
            return _globalize_local_mask(shape, hole, lo)

    for i, group_a in enumerate(groups):
        set_a = set(group_a)
        for group_b in groups[i + 1 :]:
            merged = set_a | set(group_b)
            if len(merged) != len(group_a) + len(group_b):
                continue
            if not _gap_components_ok(merged):
                continue
            trial = np.array(local, copy=True)
            for r, c in merged:
                trial[r, c] = True
            if _loop_ok(trial):
                hole = _component_holes(trial.shape, np.argwhere(trial))[0]
                return _globalize_local_mask(shape, hole, lo)

    return None


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
    Remove stray singleton specks, then canonicalize a rough nested drawing with slightly broken frames.

    Input structure:
    - The grid has exactly one background color and one foreground color.
    - Meaningful foreground objects are small filled blobs or wobbly frame curves that may be missing one or two tiny border segments.
    - In addition, 1-2 isolated foreground singletons may appear as noise, including inside a frame's intended enclosed region.

    Rule:
    - First discard every singleton foreground 4-component as non-semantic noise.
    - Then classify each remaining large near-loop as a frame if filling back one or two tiny 1-2 pixel gap segments would make it a one-hole simple loop.
    - Use the repaired frame holes to recover the containment tree, collapse every blob to one pixel, and replace every frame by a rectangular border padded around its direct children.
    - Preserve sibling row/column ordering and keep one background cell between canonical objects, just as in the base task.

    Key insight: The transformation is now genuinely two-stage: tiny isolated pixels must be removed before parsing, and frame identity comes from near-enclosure rather than strict closure.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    fg_colors = sorted(set(map(int, g.flatten())) - {bg})
    assert len(fg_colors) == 1
    fg = int(fg_colors[0])
    g = _remove_singleton_noise(g, bg=bg, fg=fg)

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, colors=[fg], return_format="dict")
    assert comps

    nodes: list[_Node] = []
    holes: list[np.ndarray | None] = []
    for comp in comps:
        coords = np.asarray(comp["coords"], dtype=int)
        node = _Node(
            coords=coords,
            bbox=tuple(map(int, comp["bbox"])),
            center=coords.mean(axis=0),
        )
        hole = _frame_hole(g.shape, coords)
        node.is_frame = hole is not None
        nodes.append(node)
        holes.append(hole)

    assert any(node.is_frame for node in nodes)
    for node_i, node in enumerate(nodes):
        containers = []
        for frame_i, frame in enumerate(nodes):
            hole = holes[frame_i]
            if node_i == frame_i or hole is None:
                continue
            if np.all(hole[tuple(node.coords.T)]):
                containers.append(frame)
        if containers:
            node.parent = min(containers, key=lambda item: int(item.coords.shape[0]))
            node.parent.children.append(node)

    roots = [node for node in nodes if node.parent is None]
    mask = _layout_root(roots)
    out = np.full(mask.shape, bg, dtype=int)
    out[mask] = fg
    return out


# --------------- generator (combo variant) ---------------

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
GAP_WIDTH_CHOICES = (1, 2)
GAP_COUNT_CHOICES = (1, 2)

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

# Noise injection
NOISE_COUNT_RANGE = (1, 2)

# Tree structure probabilities
DEPTH1_PROB = 0.35
OUTSIDE_BLOB_PROB = 0.35
DEPTH1_INSIDE_RANGE = (1, 3)
DEPTH2_INNER_BLOBS = (1, 2)
DEPTH2_OUTER_BLOBS = (0, 2)
MAX_TOTAL_CHILDREN = 4

# Step-mix alternation bias
STEP_MIX_ALT_PROB = 0.7

# Generation retry budget
GEN_RETRIES = 300


class _Obj:
    def __init__(self, kind, children=None, outside_side=None):
        self.kind = kind
        self.children = list(children or [])
        self.mask = None
        self.closed_mask = None
        self.rel = (0, 0)
        self.pos = (0, 0)
        self.outside_side = outside_side


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


def _frame_mask_and_path(h, w):
    """Generate a wobbly simple closed curve mask of size (h, w), plus its ordered path."""
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
            return mask, path
    return None, None


def _gapped_mask(mask: np.ndarray, path: list[tuple[int, int]]) -> np.ndarray | None:
    """Remove one or two short straight-run segments while keeping one readable component."""
    n = len(path)
    one_gap = []
    two_gap = []
    for i in range(n):
        prev_dir = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        next_dir = (path[(i + 1) % n][0] - path[i][0], path[(i + 1) % n][1] - path[i][1])
        if prev_dir == next_dir:
            one_gap.append((i,))

    for i in range(n):
        d0 = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        d1 = (path[(i + 1) % n][0] - path[i][0], path[(i + 1) % n][1] - path[i][1])
        d2 = (
            path[(i + 2) % n][0] - path[(i + 1) % n][0],
            path[(i + 2) % n][1] - path[(i + 1) % n][1],
        )
        if d0 == d1 == d2:
            two_gap.append((i, (i + 1) % n))

    groups = []
    if 1 in GAP_WIDTH_CHOICES:
        groups.extend(one_gap)
    if 2 in GAP_WIDTH_CHOICES:
        groups.extend(two_gap)

    target_counts = [random.choice(GAP_COUNT_CHOICES)]
    if len(GAP_COUNT_CHOICES) > 1:
        target_counts.extend(count for count in GAP_COUNT_CHOICES if count != target_counts[0])

    def try_groups(group_list):
        out = np.array(mask, copy=True)
        for idxs in group_list:
            for idx in idxs:
                r, c = path[idx]
                out[r, c] = False
        _, n_comp = label_mask(out, connectivity=4)
        if n_comp != 1:
            return None
        if _component_holes(out.shape, np.argwhere(out)):
            return None
        if _frame_hole(out.shape, np.argwhere(out)) is None:
            return None
        return out

    for target_count in target_counts:
        if target_count == 1:
            options = list(groups)
            random.shuffle(options)
            for idxs in options:
                out = try_groups([idxs])
                if out is not None:
                    return out
            continue

        pairs = []
        for i, group_a in enumerate(groups):
            cells_a = {path[idx] for idx in group_a}
            for group_b in groups[i + 1 :]:
                merged = cells_a | {path[idx] for idx in group_b}
                if len(merged) != len(group_a) + len(group_b):
                    continue
                if not _gap_components_ok(merged):
                    continue
                pairs.append((group_a, group_b))
        random.shuffle(pairs)
        for group_a, group_b in pairs:
            out = try_groups([group_a, group_b])
            if out is not None:
                return out
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
    """Recursively build masks and place children within frames."""
    if node.kind == "blob":
        node.mask = _blob_mask()
        node.closed_mask = node.mask
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
    h_hi = min(FRAME_H_RANGE[1], min_h + 2) if tight else FRAME_H_RANGE[1]
    w_hi = min(FRAME_W_RANGE[1], min_w + 3) if tight else FRAME_W_RANGE[1]
    h = random.randint(min_h, h_hi)
    w = random.randint(min_w, w_hi)
    top = margin + random.randint(0, h - 2 * margin - group_h)
    left = margin + random.randint(0, w - 2 * margin - group_w)
    for child, (r, c) in placements.items():
        child.rel = (top + r, left + c)
    closed_mask, path = _frame_mask_and_path(h, w)
    if closed_mask is None or path is None:
        return False
    gapped = _gapped_mask(closed_mask, path)
    if gapped is None:
        return False
    node.closed_mask = closed_mask
    node.mask = gapped
    return True


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


def _isolated_bg_candidates(grid, bg, region):
    H, W = grid.shape
    candidates = []
    for r, c in np.argwhere(region & (grid == bg)):
        r = int(r)
        c = int(c)
        ok = True
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < H and 0 <= cc < W and grid[rr, cc] != bg:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            candidates.append((r, c))
    return candidates


def _semantic_hole_region(shape, objects):
    region = np.zeros(shape, dtype=bool)
    for obj in objects:
        if obj.kind != "frame":
            continue
        assert obj.closed_mask is not None
        coords = np.argwhere(obj.closed_mask) + np.array(obj.pos, dtype=int)
        holes = _component_holes(shape, coords.astype(int))
        assert len(holes) == 1
        region |= holes[0]
    return region


def _inject_noise(grid, objects, bg, fg):
    """Add 1-2 isolated singleton fg pixels, forcing the first into a semantic frame interior."""
    noisy = np.array(grid, copy=True)
    hole_region = _semantic_hole_region(grid.shape, objects)
    count = random.randint(*NOISE_COUNT_RANGE)
    for step in range(count):
        region = hole_region if step == 0 else np.ones(grid.shape, dtype=bool)
        candidates = _isolated_bg_candidates(noisy, bg=bg, region=region)
        if step == 0:
            candidates = [pos for pos in candidates if hole_region[pos]]
        if not candidates:
            return None
        pos = random.choice(candidates)
        noisy[pos] = fg
    return noisy


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


def _make_tree():
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
    noisy_grid = _inject_noise(grid, objects, bg, fg)
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


def generate_input():
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 2d0172a1 combo input")
