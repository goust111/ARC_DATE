import random

import numpy as np


BACKGROUND = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
MAX_GRID_SIDE = 30
GEN_RETRY_LIMIT = 400


def _bbox_corners(bbox):
    r0, c0, r1, c1 = bbox
    return ((r0, c0), (r0, c1), (r1, c0), (r1, c1))


def _point_in_bbox(point, bbox):
    r, c = point
    r0, c0, r1, c1 = bbox
    return r0 <= r <= r1 and c0 <= c <= c1


def _bboxes_overlap(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _has_diagonal_pair(visible, bbox):
    tl, tr, bl, br = _bbox_corners(bbox)
    return (tl in visible and br in visible) or (tr in visible and bl in visible)


def _extract_rectangles(grid):
    rects = []
    colors = sorted(int(color) for color in np.unique(grid) if int(color) != BACKGROUND)
    for color in colors:
        coords = np.argwhere(grid == color)
        if coords.size == 0:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        bbox = (int(r0), int(c0), int(r1), int(c1))
        corners = _bbox_corners(bbox)
        visible = {tuple(map(int, rc)) for rc in coords}
        if not visible.issubset(set(corners)):
            raise AssertionError("non-corner pixel found in a corner-marker group")
        if not (2 <= len(visible) <= 4):
            raise AssertionError("each rectangle must expose between two and four corners")
        if not _has_diagonal_pair(visible, bbox):
            raise AssertionError("each rectangle must keep at least one visible diagonal pair")
        rects.append(
            {
                "color": int(color),
                "bbox": bbox,
                "corners": corners,
                "visible": visible,
            }
        )
    rects.sort(key=lambda rect: (rect["bbox"], rect["color"]))
    return rects


def _infer_depth_edges(rects):
    edges = set()
    n = len(rects)
    for i in range(n):
        for j in range(i + 1, n):
            pair_edges = set()
            for src_idx, dst_idx in ((i, j), (j, i)):
                src = rects[src_idx]
                dst = rects[dst_idx]
                for corner in src["corners"]:
                    if not _point_in_bbox(corner, dst["bbox"]):
                        continue
                    if corner in src["visible"]:
                        pair_edges.add((dst_idx, src_idx))
                    else:
                        pair_edges.add((src_idx, dst_idx))
            if len(pair_edges) > 1:
                raise AssertionError("contradictory depth clues for one rectangle pair")
            edges.update(pair_edges)
    return edges


def _topological_order(rects, edges):
    n = len(rects)
    succ = {idx: set() for idx in range(n)}
    indegree = [0] * n
    for lower, upper in edges:
        if upper not in succ[lower]:
            succ[lower].add(upper)
            indegree[upper] += 1

    ready = [idx for idx in range(n) if indegree[idx] == 0]
    ready.sort(key=lambda idx: (rects[idx]["bbox"], rects[idx]["color"]))
    order = []
    while ready:
        idx = ready.pop(0)
        order.append(idx)
        for nxt in sorted(succ[idx], key=lambda k: (rects[k]["bbox"], rects[k]["color"])):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                ready.append(nxt)
        ready.sort(key=lambda idx: (rects[idx]["bbox"], rects[idx]["color"]))

    if len(order) != n:
        raise AssertionError("depth clues must form an acyclic order")
    return order


def _render_rectangles(shape, rects, order):
    out = np.zeros(shape, dtype=int)
    for idx in order:
        r0, c0, r1, c1 = rects[idx]["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = rects[idx]["color"]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct stacked solid rectangles from the surviving corner markers and redraw the full occlusion scene.

    Input structure:
    - Black background with one color per rectangle.
    - Each color appears only at some of that rectangle's four bbox corners.
    - Missing corners are exactly the ones covered by higher rectangles in the final stack.

    Rule:
    - Recover each rectangle bbox from the min/max row and column of its visible corner markers.
    - For any rectangle A corner that lies inside rectangle B's bbox: if that corner survives, A is above B; if it is missing, B is above A.
    - Collect all such pairwise depth clues, topologically sort the rectangles, then repaint every rectangle as a solid fill from bottom to top.

    Key insight:
    - The absent corners are not noise: a hidden corner says which overlapping rectangle must sit above it, so bbox recovery and corner-survival tests together fully encode the stack order.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")
    rects = _extract_rectangles(g)
    if not rects:
        return g.copy()
    edges = _infer_depth_edges(rects)
    order = _topological_order(rects, edges)
    return _render_rectangles(g.shape, rects, order)


def _flip_bbox(bbox, shape, flip_ud, flip_lr):
    h, w = shape
    r0, c0, r1, c1 = bbox
    if flip_ud:
        r0, r1 = h - 1 - r1, h - 1 - r0
    if flip_lr:
        c0, c1 = w - 1 - c1, w - 1 - c0
    return (r0, c0, r1, c1)


def _sample_relative_chain(rect_count):
    rects = []
    height = random.randint(4, 8)
    width = random.randint(4, 8)
    row = 0
    col = 0
    rects.append((row, col, height, width))

    for _ in range(1, rect_count):
        _, _, prev_h, prev_w = rects[-1]
        max_dr = min(prev_h - 2, 4)
        max_dc = min(prev_w - 2, 4)
        if max_dr < 1 or max_dc < 1:
            return None
        dr = random.randint(1, max_dr)
        dc = random.randint(1, max_dc)

        min_h = max(4, prev_h - dr)
        max_h = min(9, prev_h - dr + 3)
        min_w = max(4, prev_w - dc)
        max_w = min(9, prev_w - dc + 3)
        if min_h > max_h or min_w > max_w:
            return None

        height = random.randint(min_h, max_h)
        width = random.randint(min_w, max_w)
        row += dr
        col += dc
        rects.append((row, col, height, width))
    return rects


def _build_layered_rectangles():
    rect_count = random.randint(3, 5)
    rel_rects = _sample_relative_chain(rect_count)
    if rel_rects is None:
        return None

    scene_h = max(r + h for r, _, h, _ in rel_rects)
    scene_w = max(c + w for _, c, _, w in rel_rects)
    if scene_h > MAX_GRID_SIDE or scene_w > MAX_GRID_SIDE:
        return None

    extra_h = random.randint(0, min(5, MAX_GRID_SIDE - scene_h))
    extra_w = random.randint(0, min(5, MAX_GRID_SIDE - scene_w))
    grid_h = scene_h + extra_h
    grid_w = scene_w + extra_w
    offset_r = random.randint(0, grid_h - scene_h)
    offset_c = random.randint(0, grid_w - scene_w)
    flip_ud = random.random() < 0.5
    flip_lr = random.random() < 0.5
    colors = random.sample(PALETTE, rect_count)

    rects = []
    for color, (row, col, height, width) in zip(colors, rel_rects):
        bbox = (
            offset_r + row,
            offset_c + col,
            offset_r + row + height - 1,
            offset_c + col + width - 1,
        )
        rects.append(
            {
                "color": int(color),
                "bbox": _flip_bbox(bbox, (grid_h, grid_w), flip_ud, flip_lr),
            }
        )
    return (grid_h, grid_w), rects


def _make_input_from_scene(shape, rects):
    out = np.zeros(shape, dtype=int)
    for rect in rects:
        r0, c0, r1, c1 = rect["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = rect["color"]

    inp = np.zeros(shape, dtype=int)
    annotated = []
    for rect in rects:
        corners = _bbox_corners(rect["bbox"])
        visible = {corner for corner in corners if out[corner] == rect["color"]}
        annotated.append(
            {
                "color": rect["color"],
                "bbox": rect["bbox"],
                "corners": corners,
                "visible": visible,
            }
        )
        for r, c in visible:
            inp[r, c] = rect["color"]
    return inp, out, annotated


def _scene_constraints_hold(rects):
    for rect in rects:
        if not (2 <= len(rect["visible"]) <= 4):
            return False
        if not _has_diagonal_pair(rect["visible"], rect["bbox"]):
            return False

    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            bbox_i = rects[i]["bbox"]
            bbox_j = rects[j]["bbox"]
            if not _bboxes_overlap(bbox_i, bbox_j):
                continue
            if not (
                any(_point_in_bbox(corner, bbox_j) for corner in rects[i]["corners"])
                or any(_point_in_bbox(corner, bbox_i) for corner in rects[j]["corners"])
            ):
                return False
    return True


def _gen_attempt():
    built = _build_layered_rectangles()
    if built is None:
        return None
    shape, layered_rects = built
    inp, out, annotated = _make_input_from_scene(shape, layered_rects)
    if not _scene_constraints_hold(annotated):
        return None
    try:
        if np.array_equal(main(inp), out):
            return inp
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid corner-occlusion scene")
