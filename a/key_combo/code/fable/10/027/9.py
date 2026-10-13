import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

PALETTE = (BLUE, RED, GREEN, PINK, ORANGE, TEAL, MAROON)
GEN_ATTEMPTS = 400
MIN_SKELETON_SIZE = 7
MAX_SKELETON_SIZE = 18


def _bbox(mask: np.ndarray):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box):
    r0, c0, r1, c1 = box
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)


def _vertex_mask(proto: np.ndarray) -> np.ndarray:
    fg = proto != BLACK
    h, w = proto.shape
    out = np.zeros_like(fg, dtype=bool)
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for r, c in np.argwhere(fg):
        neighbors = []
        for dr, dc in dirs:
            rr, cc = int(r) + dr, int(c) + dc
            if 0 <= rr < h and 0 <= cc < w and fg[rr, cc]:
                neighbors.append((dr, dc))
        if len(neighbors) != 2:
            out[r, c] = True
            continue
        a, b = neighbors
        if (a[0] + b[0], a[1] + b[1]) != (0, 0):
            out[r, c] = True
    return out


def _axis_spans(length: int, vertices: list[int], anchors: list[int]) -> list[tuple[int, int]]:
    assert len(vertices) == len(anchors)
    spans = []
    anchor_by_vertex = dict(zip(vertices, anchors))
    for pos in range(length):
        if pos in anchor_by_vertex:
            anchor = anchor_by_vertex[pos]
            spans.append((anchor, anchor))
            continue
        hi = 0
        while hi < len(vertices) and vertices[hi] < pos:
            hi += 1
        assert 0 < hi < len(vertices)
        lo_v, hi_v = vertices[hi - 1], vertices[hi]
        lo_a, hi_a = anchors[hi - 1], anchors[hi]
        internal_count = hi_v - lo_v - 1
        gap_count = hi_a - lo_a - 1
        assert internal_count > 0
        assert gap_count >= internal_count
        k = pos - lo_v - 1
        start = lo_a + 1 + (k * gap_count) // internal_count
        end = lo_a + ((k + 1) * gap_count) // internal_count
        assert start <= end
        spans.append((start, end))
    return spans


def _paint(out: np.ndarray, row_span: tuple[int, int], col_span: tuple[int, int], color: int) -> None:
    r0, r1 = row_span
    c0, c1 = col_span
    block = out[r0 : r1 + 1, c0 : c1 + 1]
    assert not np.any((block != BLACK) & (block != color))
    block[:] = int(color)


def _adjacency_color(proto: np.ndarray, vertices: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> int:
    non_vertex_colors = {
        int(proto[r, c])
        for r, c in (a, b)
        if not bool(vertices[r, c])
    }
    if len(non_vertex_colors) == 1:
        return non_vertex_colors.pop()
    colors = {int(proto[r, c]) for r, c in (a, b)}
    assert len(colors) == 1
    return colors.pop()


def _match_rotation(skeleton: np.ndarray, proto: np.ndarray):
    vertices = _vertex_mask(proto)
    coords = np.argwhere(vertices)
    if coords.size == 0:
        return None
    vertex_rows = sorted(set(int(r) for r in coords[:, 0]))
    vertex_cols = sorted(set(int(c) for c in coords[:, 1]))
    anchor_rows = np.flatnonzero(skeleton.any(axis=1)).astype(int).tolist()
    anchor_cols = np.flatnonzero(skeleton.any(axis=0)).astype(int).tolist()
    if len(vertex_rows) != len(anchor_rows) or len(vertex_cols) != len(anchor_cols):
        return None
    expected = vertices[np.ix_(vertex_rows, vertex_cols)]
    sampled = skeleton[np.ix_(anchor_rows, anchor_cols)]
    if not np.array_equal(expected, sampled):
        return None
    row_spans = _axis_spans(proto.shape[0], vertex_rows, anchor_rows)
    col_spans = _axis_spans(proto.shape[1], vertex_cols, anchor_cols)
    return {
        "vertices": vertices,
        "vertex_rows": vertex_rows,
        "vertex_cols": vertex_cols,
        "anchor_rows": anchor_rows,
        "anchor_cols": anchor_cols,
        "row_spans": row_spans,
        "col_spans": col_spans,
    }


def _render_stretched(proto: np.ndarray, row_spans, col_spans, vertices: np.ndarray) -> np.ndarray:
    out_h = row_spans[-1][1] + 1
    out_w = col_spans[-1][1] + 1
    out = np.zeros((out_h, out_w), dtype=int)
    fg = proto != BLACK

    for r, c in np.argwhere(fg):
        _paint(out, row_spans[int(r)], col_spans[int(c)], int(proto[r, c]))

    h, w = proto.shape
    for r in range(h):
        for c in range(w):
            if not fg[r, c]:
                continue
            if r + 1 < h and fg[r + 1, c]:
                start = row_spans[r][1] + 1
                end = row_spans[r + 1][0] - 1
                if start <= end:
                    color = _adjacency_color(proto, vertices, (r, c), (r + 1, c))
                    _paint(out, (start, end), col_spans[c], color)
            if c + 1 < w and fg[r, c + 1]:
                start = col_spans[c][1] + 1
                end = col_spans[c + 1][0] - 1
                if start <= end:
                    color = _adjacency_color(proto, vertices, (r, c), (r, c + 1))
                    _paint(out, row_spans[r], (start, end), color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate the compact prototype to the unique anchor-compatible orientation, then stretch its vertex-aligned line drawing onto the yellow canvas.

    Input structure:
    - The grid has black background plus a grey separator/filler band splitting two panels.
    - One panel contains a compact multicolor orthogonal prototype made of horizontal and vertical line segments.
    - The other panel contains only sparse yellow anchors; their bbox is the output canvas.

    Rule:
    - Compute the prototype's graph vertices as all endpoints, corners, and crossings.
    - Try the four quarter-turn rotations of the prototype and keep the unique one whose sorted vertex rows and cols map monotonically to the yellow anchor rows and cols while preserving the sampled vertex pattern.
    - Under that rotation, map prototype rows and cols independently onto the anchor rows and cols, then redraw every prototype cell and every edge interval on the stretched canvas using the prototype's visible colors.

    Key insight:
    - The yellow anchors do not match the prototype until the correct rotation is chosen first; only then does the row/column-wise stretch become a valid vertex-coordinate mapping.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    yellow_box = _bbox(g == YELLOW)
    assert yellow_box is not None
    skeleton = _crop_bbox(g, yellow_box) == YELLOW

    proto_box = _bbox((g != BLACK) & (g != GREY) & (g != YELLOW))
    assert proto_box is not None
    proto = _crop_bbox(g, proto_box)

    matches = []
    for k in range(4):
        rotated = np.rot90(proto, k)
        match = _match_rotation(skeleton, rotated)
        if match is None:
            continue
        out = _render_stretched(rotated, match["row_spans"], match["col_spans"], match["vertices"])
        matches.append((k, out))

    assert len(matches) == 1, f"expected a unique rotation match, got {[k for k, _ in matches]}"
    return matches[0][1]


TEMPLATES = [
    (
        ((0, 0), (0, 2), (2, 2), (2, 1), (3, 1)),
        ((0, 2), (0, 4)),
    ),
    (
        ((0, 1), (0, 3)),
        ((0, 1), (2, 1), (2, 4), (3, 4)),
    ),
    (
        ((0, 0), (0, 2)),
        ((0, 0), (2, 0), (2, 3), (3, 3)),
    ),
    (
        ((0, 1), (1, 1), (1, 3)),
        ((1, 1), (2, 1), (2, 0)),
        ((2, 1), (2, 2), (3, 2)),
    ),
    (
        ((0, 2), (1, 2), (1, 0)),
        ((1, 2), (1, 4)),
        ((1, 2), (3, 2), (3, 1)),
    ),
    (
        ((0, 0), (0, 1), (0, 3)),
        ((0, 1), (2, 1), (2, 4), (3, 4)),
    ),
]


def _build_template(chains):
    vertices = set()
    edges = []
    for chain in chains:
        pts = [tuple(map(int, p)) for p in chain]
        vertices.update(pts)
        for a, b in zip(pts, pts[1:]):
            assert a[0] == b[0] or a[1] == b[1]
            edges.append((a, b))
    max_r = max(r for r, _ in vertices)
    max_c = max(c for _, c in vertices)
    return {"vertices": sorted(vertices), "edges": edges, "shape": (max_r + 1, max_c + 1)}


CANONICAL_TEMPLATES = [_build_template(chains) for chains in TEMPLATES]


def _sample_vertex_pattern(vertices: np.ndarray) -> np.ndarray:
    coords = np.argwhere(vertices)
    rows = sorted(set(int(r) for r in coords[:, 0]))
    cols = sorted(set(int(c) for c in coords[:, 1]))
    return vertices[np.ix_(rows, cols)]


def _rotation_signatures(proto: np.ndarray) -> list[tuple[tuple[int, ...], ...]]:
    signatures = []
    for k in range(4):
        vertices = _vertex_mask(np.rot90(proto, k))
        sampled = _sample_vertex_pattern(vertices).astype(int)
        signatures.append(tuple(tuple(int(x) for x in row) for row in sampled))
    return signatures


def _template_proto(template) -> np.ndarray:
    h = 2 * (template["shape"][0] - 1) + 1
    w = 2 * (template["shape"][1] - 1) + 1
    proto = np.zeros((h, w), dtype=int)

    edge_count = len(template["edges"])
    color_count = int(np.random.randint(2, min(5, len(PALETTE), edge_count) + 1))
    chosen_palette = list(np.random.choice(PALETTE, size=color_count, replace=False))
    edge_colors = [chosen_palette[int(np.random.randint(0, color_count))] for _ in range(edge_count)]
    if edge_count >= 3 and len(set(edge_colors)) < 2:
        edge_colors[0] = chosen_palette[0]
        edge_colors[1] = chosen_palette[-1]
    if edge_count >= 4 and color_count >= 3 and len(set(edge_colors)) < 3:
        edge_colors[2] = chosen_palette[1]

    incident = {vertex: [] for vertex in template["vertices"]}
    for (a, b), color in zip(template["edges"], edge_colors):
        incident[a].append(int(color))
        incident[b].append(int(color))
        ar, ac = 2 * a[0], 2 * a[1]
        br, bc = 2 * b[0], 2 * b[1]
        if ar == br:
            c0, c1 = sorted((ac, bc))
            proto[ar, c0 + 1 : c1] = int(color)
        else:
            r0, r1 = sorted((ar, br))
            proto[r0 + 1 : r1, ac] = int(color)

    for vertex, colors in incident.items():
        vr, vc = 2 * vertex[0], 2 * vertex[1]
        proto[vr, vc] = int(colors[int(np.random.randint(0, len(colors)))])

    expected_vertices = np.zeros_like(proto, dtype=bool)
    for r, c in template["vertices"]:
        expected_vertices[2 * r, 2 * c] = True
    box = _bbox(proto != BLACK)
    assert box is not None
    proto = _crop_bbox(proto, box)
    expected_vertices = _crop_bbox(expected_vertices.astype(int), box).astype(bool)
    if not np.array_equal(_vertex_mask(proto), expected_vertices):
        return None
    if len(set(_rotation_signatures(proto))) != 4:
        return None
    if np.count_nonzero(proto) <= 6:
        return None
    return proto


def _anchor_positions(vertex_coords: list[int]) -> list[int] | None:
    diffs = [vertex_coords[i + 1] - vertex_coords[i] for i in range(len(vertex_coords) - 1)]
    for _ in range(200):
        extras = [int(np.random.randint(0, 4)) for _ in diffs]
        if diffs and not any(extra > 0 for extra in extras):
            extras[int(np.random.randint(0, len(extras)))] = 1
        gaps = [diff + extra for diff, extra in zip(diffs, extras)]
        anchors = [0]
        for gap in gaps:
            anchors.append(anchors[-1] + int(gap))
        size = anchors[-1] + 1
        if MIN_SKELETON_SIZE <= size <= MAX_SKELETON_SIZE:
            return anchors
    return None


def _make_skeleton(rotated_proto: np.ndarray):
    vertices = _vertex_mask(rotated_proto)
    coords = np.argwhere(vertices)
    vertex_rows = sorted(set(int(r) for r in coords[:, 0]))
    vertex_cols = sorted(set(int(c) for c in coords[:, 1]))
    anchor_rows = _anchor_positions(vertex_rows)
    anchor_cols = _anchor_positions(vertex_cols)
    if anchor_rows is None or anchor_cols is None:
        return None
    if anchor_rows[-1] + 1 == rotated_proto.shape[0] and anchor_cols[-1] + 1 == rotated_proto.shape[1]:
        return None
    row_map = dict(zip(vertex_rows, anchor_rows))
    col_map = dict(zip(vertex_cols, anchor_cols))
    skeleton = np.zeros((anchor_rows[-1] + 1, anchor_cols[-1] + 1), dtype=int)
    for r, c in np.argwhere(vertices):
        skeleton[row_map[int(r)], col_map[int(c)]] = YELLOW
    return skeleton, vertices, vertex_rows, vertex_cols, anchor_rows, anchor_cols


def _compose_input(proto: np.ndarray, skeleton: np.ndarray) -> np.ndarray:
    proto_first = bool(np.random.randint(0, 2))
    horizontal = bool(np.random.randint(0, 2))
    ph, pw = proto.shape
    sh, sw = skeleton.shape

    if horizontal:
        h = max(ph, sh)
        w = pw + 1 + sw
        grid = np.zeros((h, w), dtype=int)
        left, right = (proto, skeleton) if proto_first else (skeleton, proto)
        lw = left.shape[1]
        rw = right.shape[1]
        grid[:, lw] = GREY
        grid[: left.shape[0], :lw] = left
        grid[: right.shape[0], lw + 1 : lw + 1 + rw] = right
        if left.shape[0] < h:
            grid[left.shape[0] :, :lw] = GREY
        if right.shape[0] < h:
            grid[right.shape[0] :, lw + 1 : lw + 1 + rw] = GREY
        return grid

    h = ph + 1 + sh
    w = max(pw, sw)
    grid = np.zeros((h, w), dtype=int)
    top, bottom = (proto, skeleton) if proto_first else (skeleton, proto)
    th, tw = top.shape
    bh, bw = bottom.shape
    grid[th, :] = GREY
    grid[:th, :tw] = top
    grid[th + 1 : th + 1 + bh, :bw] = bottom
    if tw < w:
        grid[:th, tw:] = GREY
    if bw < w:
        grid[th + 1 : th + 1 + bh, bw:] = GREY
    return grid


def _matching_rotations(skeleton: np.ndarray, proto: np.ndarray) -> list[int]:
    hits = []
    for k in range(4):
        if _match_rotation(skeleton, np.rot90(proto, k)) is not None:
            hits.append(k)
    return hits


def _gen_attempt():
    template = CANONICAL_TEMPLATES[int(np.random.randint(0, len(CANONICAL_TEMPLATES)))]
    base_proto = _template_proto(template)
    if base_proto is None:
        return None

    proto_rot = int(np.random.randint(0, 4))
    match_rot = int(np.random.randint(0, 4))
    proto = np.rot90(base_proto, proto_rot)
    rotated_target = np.rot90(proto, match_rot)

    skeleton_info = _make_skeleton(rotated_target)
    if skeleton_info is None:
        return None
    skeleton, vertices, vertex_rows, vertex_cols, anchor_rows, anchor_cols = skeleton_info
    row_spans = _axis_spans(rotated_target.shape[0], vertex_rows, anchor_rows)
    col_spans = _axis_spans(rotated_target.shape[1], vertex_cols, anchor_cols)
    expected = _render_stretched(rotated_target, row_spans, col_spans, vertices)

    if _matching_rotations(skeleton == YELLOW, proto) != [match_rot]:
        return None
    if expected.shape != skeleton.shape:
        return None

    grid = _compose_input(proto, skeleton)
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rotation-then-stretch ARC input")
