from bisect import bisect_left

import numpy as np

from arc_utils import Color, bbox, crop_bbox


def _vertex_mask(proto: np.ndarray) -> np.ndarray:
    fg = proto != Color.BLACK
    h, w = proto.shape
    vertices = np.zeros_like(fg, dtype=bool)
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for r, c in np.argwhere(fg):
        neighbors = []
        for dr, dc in dirs:
            rr, cc = int(r) + dr, int(c) + dc
            if 0 <= rr < h and 0 <= cc < w and fg[rr, cc]:
                neighbors.append((dr, dc))

        if len(neighbors) != 2:
            vertices[r, c] = True
        else:
            a, b = neighbors
            if (a[0] + b[0], a[1] + b[1]) != (0, 0):
                vertices[r, c] = True

    return vertices


def _axis_spans(length: int, vertices: list[int], anchors: list[int]) -> list[tuple[int, int]]:
    assert len(vertices) == len(anchors)
    assert all(vertices[i] < vertices[i + 1] for i in range(len(vertices) - 1))
    assert all(anchors[i] < anchors[i + 1] for i in range(len(anchors) - 1))

    anchor_by_vertex = dict(zip(vertices, anchors))
    spans: list[tuple[int, int]] = []

    for pos in range(length):
        if pos in anchor_by_vertex:
            anchor = anchor_by_vertex[pos]
            spans.append((anchor, anchor))
            continue

        hi_index = bisect_left(vertices, pos)
        assert 0 < hi_index < len(vertices)
        lo_v, hi_v = vertices[hi_index - 1], vertices[hi_index]
        lo_a, hi_a = anchors[hi_index - 1], anchors[hi_index]
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
    old = out[r0 : r1 + 1, c0 : c1 + 1]
    assert not np.any((old != Color.BLACK) & (old != color))
    old[:] = color


def _has_perpendicular_neighbor(fg: np.ndarray, r: int, c: int, axis: str) -> bool:
    h, w = fg.shape
    if axis == "vertical":
        return (c > 0 and fg[r, c - 1]) or (c + 1 < w and fg[r, c + 1])
    assert axis == "horizontal"
    return (r > 0 and fg[r - 1, c]) or (r + 1 < h and fg[r + 1, c])


def _edge_color(proto: np.ndarray, a: tuple[int, int], b: tuple[int, int], axis: str) -> int:
    fg = proto != Color.BLACK
    candidates = {
        int(proto[r, c])
        for r, c in (a, b)
        if not _has_perpendicular_neighbor(fg, r, c, axis)
    }
    if len(candidates) == 1:
        return candidates.pop()

    a_color = int(proto[a])
    b_color = int(proto[b])
    assert a_color == b_color
    return a_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stretch the compact colored line prototype onto the sparse yellow skeleton canvas.

    Input structure
    - The grid has black background with grey separator/filler cells.
    - One panel contains only a sparse yellow skeleton; its bounding box is the output canvas.
    - The other panel is a compact colored prototype made from horizontal and vertical line segments.

    Rule
    - Treat endpoints, corners, and crossings of the prototype foreground as graph vertices.
    - The yellow pixels are the same vertex pattern after independently stretching rows and columns.
    - Map sorted prototype vertex rows/columns to sorted yellow rows/columns.
    - Paint every prototype foreground cell and adjacent foreground edge through the mapped row/column spans.

    Key insight
    - The yellow pixels are anchors for the prototype's graph vertices, while the compact prototype supplies the colors for the stretched line segments.
    """
    g = np.asarray(grid, dtype=int)

    yellow_box = bbox(g == Color.YELLOW)
    assert yellow_box is not None
    skeleton = crop_bbox(g, yellow_box) == Color.YELLOW

    proto_box = bbox((g != Color.BLACK) & (g != Color.YELLOW) & (g != Color.GREY))
    assert proto_box is not None
    proto = crop_bbox(g, proto_box)
    assert not np.any((proto == Color.YELLOW) | (proto == Color.GREY))

    vertices = _vertex_mask(proto)
    vertex_coords = np.argwhere(vertices)
    assert len(vertex_coords) > 0
    vertex_rows = sorted(set(map(int, vertex_coords[:, 0])))
    vertex_cols = sorted(set(map(int, vertex_coords[:, 1])))
    anchor_rows = np.flatnonzero(skeleton.any(axis=1)).astype(int).tolist()
    anchor_cols = np.flatnonzero(skeleton.any(axis=0)).astype(int).tolist()
    assert len(vertex_rows) == len(anchor_rows)
    assert len(vertex_cols) == len(anchor_cols)

    sampled_skeleton = skeleton[np.ix_(anchor_rows, anchor_cols)]
    expected_skeleton = np.array(
        [[vertices[r, c] for c in vertex_cols] for r in vertex_rows],
        dtype=bool,
    )
    assert np.array_equal(sampled_skeleton, expected_skeleton)

    row_spans = _axis_spans(proto.shape[0], vertex_rows, anchor_rows)
    col_spans = _axis_spans(proto.shape[1], vertex_cols, anchor_cols)

    out = np.zeros(skeleton.shape, dtype=int)
    fg = proto != Color.BLACK
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
                    color = _edge_color(proto, (r, c), (r + 1, c), "vertical")
                    _paint(out, (start, end), col_spans[c], color)

            if c + 1 < w and fg[r, c + 1]:
                start = col_spans[c][1] + 1
                end = col_spans[c + 1][0] - 1
                if start <= end:
                    color = _edge_color(proto, (r, c), (r, c + 1), "horizontal")
                    _paint(out, row_spans[r], (start, end), color)

    return out

# --------------- generator (o5x) ---------------

# Prototype palette/structure.
PROTO_COLORS = (Color.BLUE, Color.RED, Color.GREEN)
PROTO_STRUCTURE_PAIRS = ((1, 2), (1, 3), (2, 1), (2, 2), (2, 3))
PROTO_LINE_STEP = 2

# Skeleton anchors and composed input bounds.
SKELETON_TARGET_COL_MIN, SKELETON_TARGET_COL_MAX_EXCLUSIVE = 7, 12
SKELETON_ROW_MIN, SKELETON_ROW_MAX = 7, 12
ANCHOR_GAP_MIN, ANCHOR_GAP_MAX = 1, 5
INPUT_COL_MIN, INPUT_COL_MAX = 11, 15
SEPARATOR_WIDTH = 1
GEN_RETRY_LIMIT = 300


def _proto_axis_len(line_count: int) -> int:
    return PROTO_LINE_STEP * line_count + 1


def _sample_line_colors(line_count: int) -> list[int]:
    colors = list(PROTO_COLORS)
    line_colors = colors + [colors[int(np.random.randint(0, len(colors)))] for _ in range(line_count - len(colors))]
    np.random.shuffle(line_colors)
    return line_colors


def _gen_line_proto(nh: int, nv: int) -> np.ndarray:
    line_colors = _sample_line_colors(nh + nv)
    h_colors = line_colors[:nh]
    v_colors = line_colors[nh:]

    proto = np.zeros((_proto_axis_len(nh), _proto_axis_len(nv)), dtype=int)
    h_rows = [PROTO_LINE_STEP * i + 1 for i in range(nh)]
    v_cols = [PROTO_LINE_STEP * i + 1 for i in range(nv)]

    for r, color in zip(h_rows, h_colors):
        proto[r, :] = int(color)
    for c, color in zip(v_cols, v_colors):
        proto[:, c] = int(color)
    for i, r in enumerate(h_rows):
        for j, c in enumerate(v_cols):
            proto[r, c] = int(h_colors[i] if np.random.randint(0, 2) == 0 else v_colors[j])

    return proto


def _gen_anchor_positions(vertices: list[int], min_len: int, max_len: int) -> list[int] | None:
    min_gaps = [
        max(ANCHOR_GAP_MIN, vertices[i + 1] - vertices[i])
        for i in range(len(vertices) - 1)
    ]
    lo = max(min_len, 1 + sum(min_gaps))
    hi = min(max_len, 1 + ANCHOR_GAP_MAX * len(min_gaps))
    if lo > hi:
        return None

    remaining = int(np.random.randint(lo, hi + 1)) - 1
    gaps = []
    for i, min_gap in enumerate(min_gaps):
        rest_mins = sum(min_gaps[i + 1 :])
        rest_maxs = ANCHOR_GAP_MAX * (len(min_gaps) - i - 1)
        gap_lo = max(min_gap, remaining - rest_maxs)
        gap_hi = min(ANCHOR_GAP_MAX, remaining - rest_mins)
        if gap_lo > gap_hi:
            return None
        gap = int(np.random.randint(gap_lo, gap_hi + 1))
        gaps.append(gap)
        remaining -= gap

    anchors = [0]
    for gap in gaps:
        anchors.append(anchors[-1] + gap)
    return anchors


def _gen_stretch_proto(
    proto: np.ndarray,
    vertex_rows: list[int],
    vertex_cols: list[int],
    anchor_rows: list[int],
    anchor_cols: list[int],
) -> np.ndarray:
    row_spans = _axis_spans(proto.shape[0], vertex_rows, anchor_rows)
    col_spans = _axis_spans(proto.shape[1], vertex_cols, anchor_cols)

    out = np.zeros((anchor_rows[-1] + 1, anchor_cols[-1] + 1), dtype=int)
    fg = proto != Color.BLACK
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
                    color = _edge_color(proto, (r, c), (r + 1, c), "vertical")
                    _paint(out, (start, end), col_spans[c], color)
            if c + 1 < w and fg[r, c + 1]:
                start = col_spans[c][1] + 1
                end = col_spans[c + 1][0] - 1
                if start <= end:
                    color = _edge_color(proto, (r, c), (r, c + 1), "horizontal")
                    _paint(out, row_spans[r], (start, end), color)

    return out


def _gen_skeleton(
    vertices: np.ndarray,
    vertex_rows: list[int],
    vertex_cols: list[int],
    anchor_rows: list[int],
    anchor_cols: list[int],
) -> np.ndarray:
    skeleton = np.zeros((anchor_rows[-1] + 1, anchor_cols[-1] + 1), dtype=int)
    skeleton[np.ix_(anchor_rows, anchor_cols)] = vertices[np.ix_(vertex_rows, vertex_cols)] * Color.YELLOW
    return skeleton


def _gen_compose_input(proto: np.ndarray, skeleton: np.ndarray, proto_left: bool) -> np.ndarray:
    h, skel_w = skeleton.shape
    proto_h, proto_w = proto.shape
    grid = np.zeros((h, proto_w + SEPARATOR_WIDTH + skel_w), dtype=int)

    if proto_left:
        grid[:proto_h, :proto_w] = proto
        grid[:, proto_w : proto_w + SEPARATOR_WIDTH] = Color.GREY
        grid[proto_h:, :proto_w] = Color.GREY
        grid[:, proto_w + SEPARATOR_WIDTH :] = skeleton
    else:
        grid[:, :skel_w] = skeleton
        grid[:, skel_w : skel_w + SEPARATOR_WIDTH] = Color.GREY
        grid[:proto_h, skel_w + SEPARATOR_WIDTH :] = proto
        grid[proto_h:, skel_w + SEPARATOR_WIDTH :] = Color.GREY

    return grid


def _choose_structure_pair(target_cols: int) -> tuple[int, int]:
    feasible_pairs = [
        (nh, nv)
        for nh, nv in PROTO_STRUCTURE_PAIRS
        if target_cols + _proto_axis_len(nv) + SEPARATOR_WIDTH <= INPUT_COL_MAX
    ]
    return feasible_pairs[int(np.random.randint(0, len(feasible_pairs)))]


def _gen_attempt() -> np.ndarray | None:
    target_cols = int(np.random.randint(SKELETON_TARGET_COL_MIN, SKELETON_TARGET_COL_MAX_EXCLUSIVE))
    nh, nv = _choose_structure_pair(target_cols)
    proto = _gen_line_proto(nh, nv)

    vertices = _vertex_mask(proto)
    vertex_coords = np.argwhere(vertices)
    vertex_rows = sorted(set(map(int, vertex_coords[:, 0])))
    vertex_cols = sorted(set(map(int, vertex_coords[:, 1])))

    anchor_rows = _gen_anchor_positions(vertex_rows, SKELETON_ROW_MIN, SKELETON_ROW_MAX)
    anchor_cols = _gen_anchor_positions(vertex_cols, target_cols, target_cols)
    if anchor_rows is None or anchor_cols is None:
        return None

    expected = _gen_stretch_proto(proto, vertex_rows, vertex_cols, anchor_rows, anchor_cols)
    skeleton = _gen_skeleton(vertices, vertex_rows, vertex_cols, anchor_rows, anchor_cols)
    inp = _gen_compose_input(proto, skeleton, bool(np.random.randint(0, 2)))

    rows, cols = inp.shape
    if rows < SKELETON_ROW_MIN or rows > SKELETON_ROW_MAX or cols < INPUT_COL_MIN or cols > INPUT_COL_MAX:
        return None
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRY_LIMIT} attempts")
