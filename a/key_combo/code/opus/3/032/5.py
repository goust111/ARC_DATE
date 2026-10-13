from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox, crop_bbox


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
LINE_COLORS = (Color.BLUE, Color.RED)
GEN_ATTEMPTS = 400
WORK_H_RANGE = (6, 8)
WORK_W_RANGE = (6, 9)
BASE_PATH_COUNT_RANGE = (2, 3)
EXTRA_SEGMENT_COUNT_RANGE = (1, 2)
REPAIR_SEGMENT_COUNT_CHOICES = (1, 2, 2, 3)
PATH_PADDING = 1
ANCHOR_STEP_MIN = 2
ANCHOR_STEP_MAX = 4


def _sgn(x: int) -> int:
    return (x > 0) - (x < 0)


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    if c0 == c1:
        step = 1 if r1 >= r0 else -1
        return [(r, c0) for r in range(r0, r1 + step, step)]
    raise ValueError("segment must be axis-aligned")


def _polyline_cells(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for a, b in zip(points, points[1:]):
        seg = _segment_cells(a, b)
        if cells and seg and seg[0] == cells[-1]:
            seg = seg[1:]
        cells.extend(seg)
    return cells or [points[0]]


def _vertex_mask(proto: np.ndarray) -> np.ndarray:
    fg = proto != Color.BLACK
    h, w = proto.shape
    vertices = np.zeros_like(fg, dtype=bool)
    for r, c in np.argwhere(fg):
        neighbors = []
        for dr, dc in DIRS4:
            rr, cc = int(r) + dr, int(c) + dc
            if 0 <= rr < h and 0 <= cc < w and fg[rr, cc]:
                neighbors.append((dr, dc))
        if len(neighbors) != 2:
            vertices[r, c] = True
            continue
        a, b = neighbors
        if (a[0] + b[0], a[1] + b[1]) != (0, 0):
            vertices[r, c] = True
    return vertices


def _axis_spans(length: int, vertices: list[int], anchors: list[int]) -> list[tuple[int, int]]:
    assert len(vertices) == len(anchors)
    anchor_by_vertex = dict(zip(vertices, anchors))
    spans: list[tuple[int, int]] = []
    for pos in range(length):
        if pos in anchor_by_vertex:
            anchor = int(anchor_by_vertex[pos])
            spans.append((anchor, anchor))
            continue
        hi_index = 0
        while hi_index < len(vertices) and vertices[hi_index] < pos:
            hi_index += 1
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
        spans.append((start, end))
    return spans


def _paint(out: np.ndarray, row_span: tuple[int, int], col_span: tuple[int, int], color: int) -> None:
    r0, r1 = row_span
    c0, c1 = col_span
    view = out[r0 : r1 + 1, c0 : c1 + 1]
    if np.any((view != Color.BLACK) & (view != color)):
        raise AssertionError("conflicting colors in stretched fragments")
    view[:] = color


def _stretch_fragments(
    proto: np.ndarray,
    shape: tuple[int, int],
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for r, c in np.argwhere(proto != Color.BLACK):
        _paint(out, row_spans[int(r)], col_spans[int(c)], int(proto[r, c]))
    return out


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _faces_gap(grid: np.ndarray, r: int, c: int, dr: int, dc: int) -> bool:
    color = int(grid[r, c])
    if color == Color.BLACK:
        return False
    h, w = grid.shape
    fr, fc = r + dr, c + dc
    if 0 <= fr < h and 0 <= fc < w and grid[fr, fc] == color:
        return False
    return True


def _scan_first_hit(
    grid: np.ndarray, r0: int, c0: int, dr: int, dc: int
) -> tuple[int, int, int, int] | None:
    h, w = grid.shape
    r, c = r0 + dr, c0 + dc
    dist = 1
    while 0 <= r < h and 0 <= c < w:
        value = int(grid[r, c])
        if value != Color.BLACK:
            return dist, r, c, value
        r += dr
        c += dc
        dist += 1
    return None


def _gap_fill_cells(
    grid: np.ndarray, r: int, c: int, dr: int, dc: int
) -> tuple[list[tuple[int, int]], tuple[int, int]] | None:
    color = int(grid[r, c])
    h, w = grid.shape
    rr, cc = r + dr, c + dc
    blanks: list[tuple[int, int]] = []
    while 0 <= rr < h and 0 <= cc < w:
        value = int(grid[rr, cc])
        if value == Color.BLACK:
            blanks.append((rr, cc))
            rr += dr
            cc += dc
            continue
        if value != color or not blanks:
            return None
        if not _faces_gap(grid, rr, cc, -dr, -dc):
            return None
        return blanks, (rr, cc)
    return None


def _marker_actions(
    base: np.ndarray, markers: list[tuple[int, int]]
) -> list[tuple[tuple[int, int], int, list[tuple[int, int]]]]:
    actions: list[tuple[tuple[int, int], int, list[tuple[int, int]]]] = []
    for r0, c0 in sorted(markers):
        candidates = []
        for dr, dc in DIRS4:
            hit = _scan_first_hit(base, r0, c0, dr, dc)
            if hit is None:
                continue
            dist, r, c, color = hit
            face_dr, face_dc = -dr, -dc
            if not _faces_gap(base, r, c, face_dr, face_dc):
                continue
            gap = _gap_fill_cells(base, r, c, face_dr, face_dc)
            if gap is None:
                continue
            fill_cells, far_endpoint = gap
            far_r, far_c = far_endpoint
            candidates.append((dist, r, c, color, face_dr, face_dc, far_r, far_c, fill_cells))

        if not candidates:
            raise AssertionError(f"marker at {(r0, c0)} does not see a repairable endpoint")
        candidates.sort(key=lambda item: item[:8])
        best = candidates[0]
        if len(candidates) > 1 and candidates[1][0] == best[0] and candidates[1][:8] != best[:8]:
            raise AssertionError(f"marker at {(r0, c0)} has ambiguous nearest endpoints")
        actions.append(((r0, c0), int(best[3]), list(best[8])))
    return actions


def _repair_marked_gaps(base: np.ndarray, markers: list[tuple[int, int]]) -> np.ndarray:
    out = base.copy()
    for _marker, color, fill_cells in _marker_actions(base, markers):
        for r, c in fill_cells:
            out[r, c] = color
    return out


def _extract_panels(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sep_cols = np.flatnonzero(np.all(grid == Color.GREY, axis=0))
    assert len(sep_cols) == 1, "expected exactly one full grey separator column"
    sep = int(sep_cols[0])
    left = grid[:, :sep]
    right = grid[:, sep + 1 :]
    proto_box = bbox((left != Color.BLACK) & (left != Color.GREY))
    assert proto_box is not None, "prototype panel must contain colored lines"
    proto = crop_bbox(left, proto_box)
    return proto, right.copy()


def _solve(proto: np.ndarray, right_panel: np.ndarray) -> np.ndarray:
    vertices = _vertex_mask(proto)
    vertex_coords = np.argwhere(vertices)
    assert len(vertex_coords) > 0
    vertex_rows = sorted(set(map(int, vertex_coords[:, 0])))
    vertex_cols = sorted(set(map(int, vertex_coords[:, 1])))

    anchor_mask = right_panel == Color.YELLOW
    anchor_rows = np.flatnonzero(anchor_mask.any(axis=1)).astype(int).tolist()
    anchor_cols = np.flatnonzero(anchor_mask.any(axis=0)).astype(int).tolist()
    assert len(vertex_rows) == len(anchor_rows)
    assert len(vertex_cols) == len(anchor_cols)

    sampled = anchor_mask[np.ix_(anchor_rows, anchor_cols)]
    expected = np.array(
        [[vertices[r, c] for c in vertex_cols] for r in vertex_rows],
        dtype=bool,
    )
    assert np.array_equal(sampled, expected)
    assert int(anchor_mask.sum()) == int(expected.sum())

    row_spans = _axis_spans(proto.shape[0], vertex_rows, anchor_rows)
    col_spans = _axis_spans(proto.shape[1], vertex_cols, anchor_cols)
    base = _stretch_fragments(proto, right_panel.shape, row_spans, col_spans)
    markers = [tuple(map(int, pt)) for pt in np.argwhere(right_panel == Color.GREEN)]
    return _repair_marked_gaps(base, markers)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stretch the compact line prototype onto the anchor panel, then repair only the green-marked stretch gaps.

    Input structure:
    - A grey full column splits the grid into a compact left prototype panel and a larger right anchor panel.
    - The left panel contains blue and red one-cell-thick orthogonal line fragments whose endpoints and corners act as graph vertices.
    - The right panel contains yellow anchor pixels matching the prototype vertex pattern after independent row/column stretching, plus green singleton markers inside selected empty gaps.

    Rule:
    - Detect prototype vertices as foreground cells with non-straight degree two or with degree other than two, then map sorted vertex rows and columns to the sorted yellow anchor rows and columns.
    - Stretch every prototype foreground cell onto the right panel using that row/column mapping, which keeps long segments continuous but can leave 1-3 cell gaps where adjacent prototype vertices were directly connected.
    - For each green marker, look along its row and column for the nearest stretched endpoint that faces the marker through empty cells, then fill that entire gap with the endpoint's color until the nearest opposite endpoint of the same color.
    - Replace the marker itself during filling, keep unmarked empty gaps black, and remove the left panel and separator from the output.

    Key insight:
    - The yellow anchors explain how to stretch the prototype, while the green markers do not add new geometry: they simply select which facing endpoint gaps created by the stretch should be repaired.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    proto, right_panel = _extract_panels(g)
    return _solve(proto, right_panel)


def _buffered_cells(cells: set[tuple[int, int]], shape: tuple[int, int], padding: int = PATH_PADDING) -> set[tuple[int, int]]:
    h, w = shape
    out = set(cells)
    for r, c in cells:
        for dr in range(-padding, padding + 1):
            for dc in range(-padding, padding + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _random_turning_path(
    shape: tuple[int, int],
    forbidden: set[tuple[int, int]],
    *,
    require_unit_segment: bool,
) -> dict | None:
    h, w = shape
    for _ in range(300):
        seg_count = random.choice(REPAIR_SEGMENT_COUNT_CHOICES) if require_unit_segment else random.randint(*EXTRA_SEGMENT_COUNT_RANGE)
        dirs = [random.choice(DIRS4)]
        for _j in range(1, seg_count):
            prev = dirs[-1]
            dirs.append(random.choice([(0, 1), (0, -1)] if prev[0] else [(1, 0), (-1, 0)]))

        lengths = [random.randint(2, 3) for _ in range(seg_count)]
        if require_unit_segment:
            unit_index = random.randrange(seg_count)
            lengths[unit_index] = 1
            if seg_count > 1 and random.random() < 0.35:
                other = random.randrange(seg_count)
                lengths[other] = 1

        rel_points = [(0, 0)]
        rr = cc = 0
        for (dr, dc), length in zip(dirs, lengths):
            rr += dr * length
            cc += dc * length
            rel_points.append((rr, cc))

        min_r = min(r for r, _ in rel_points)
        max_r = max(r for r, _ in rel_points)
        min_c = min(c for _, c in rel_points)
        max_c = max(c for _, c in rel_points)
        if max_r - min_r + 1 > h - 2 or max_c - min_c + 1 > w - 2:
            continue

        start_r = random.randint(1 - min_r, h - 2 - max_r)
        start_c = random.randint(1 - min_c, w - 2 - max_c)
        points = [(start_r + r, start_c + c) for r, c in rel_points]
        cells = _polyline_cells(points)
        if len(set(cells)) != len(cells):
            continue
        if any(cell in forbidden for cell in cells):
            continue
        return {
            "points": points,
            "cells": set(cells),
            "lengths": list(lengths),
        }
    return None


def _endpoint_outward_direction(path: dict, which: str) -> tuple[tuple[int, int], tuple[int, int]]:
    points = path["points"]
    if which == "start":
        a, b = points[0], points[1]
        step = (_sgn(b[0] - a[0]), _sgn(b[1] - a[1]))
        return a, (-step[0], -step[1])
    a, b = points[-2], points[-1]
    step = (_sgn(b[0] - a[0]), _sgn(b[1] - a[1]))
    return b, step


def _place_structural_gap_path(
    proto: np.ndarray,
    paths: list[dict],
) -> dict | None:
    h, w = proto.shape
    occupied = {tuple(map(int, pt)) for pt in np.argwhere(proto != Color.BLACK)}
    endpoint_specs = []
    for path in paths:
        for which in ("start", "end"):
            endpoint_specs.append((path, *_endpoint_outward_direction(path, which)))
    random.shuffle(endpoint_specs)

    for path, endpoint, direction in endpoint_specs:
        color = int(path["color"])
        dr, dc = direction
        for gap in random.sample([1, 2], 2):
            for seg_len in random.sample([2, 3], 2):
                start = (endpoint[0] + dr * (gap + 1), endpoint[1] + dc * (gap + 1))
                end = (start[0] + dr * seg_len, start[1] + dc * seg_len)
                points = [start, end]
                cells = _polyline_cells(points)
                if any(not _in_bounds(proto.shape, r, c) for r, c in cells):
                    continue
                if any(cell in occupied for cell in cells):
                    continue
                ok = True
                for r, c in cells:
                    for pdr, pdc in DIRS4:
                        nr, nc = r + pdr, c + pdc
                        if (nr, nc) in occupied and (nr, nc) != endpoint:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    continue
                gap_cells = [(endpoint[0] + dr * step, endpoint[1] + dc * step) for step in range(1, gap + 1)]
                if any(cell in occupied or not _in_bounds(proto.shape, *cell) for cell in gap_cells):
                    continue
                return {
                    "points": points,
                    "cells": set(cells),
                    "lengths": [seg_len],
                    "color": color,
                }
    return None


def _render_paths(shape: tuple[int, int], paths: list[dict]) -> np.ndarray:
    proto = np.zeros(shape, dtype=int)
    for path in paths:
        color = int(path["color"])
        for r, c in path["cells"]:
            proto[r, c] = color
    return proto


def _crop_proto_and_paths(proto: np.ndarray, paths: list[dict]) -> tuple[np.ndarray, list[dict]]:
    box = bbox(proto != Color.BLACK)
    assert box is not None
    r0, c0, _r1, _c1 = box
    cropped = crop_bbox(proto, box)
    shifted = []
    for path in paths:
        shifted.append(
            {
                "points": [(r - r0, c - c0) for r, c in path["points"]],
                "cells": {(r - r0, c - c0) for r, c in path["cells"]},
                "lengths": list(path["lengths"]),
                "color": int(path["color"]),
            }
        )
    return cropped, shifted


def _vertex_axes_valid(proto: np.ndarray) -> bool:
    vertices = _vertex_mask(proto)
    coords = np.argwhere(vertices)
    if len(coords) < 2:
        return False
    rows = sorted(set(map(int, coords[:, 0])))
    cols = sorted(set(map(int, coords[:, 1])))
    return all(b - a <= 4 for a, b in zip(rows, rows[1:])) and all(b - a <= 4 for a, b in zip(cols, cols[1:]))


def _sample_anchor_positions(vertex_positions: list[int]) -> list[int] | None:
    anchors = [0]
    for a, b in zip(vertex_positions, vertex_positions[1:]):
        min_step = max(ANCHOR_STEP_MIN, b - a)
        if min_step > ANCHOR_STEP_MAX:
            return None
        anchors.append(anchors[-1] + random.randint(min_step, ANCHOR_STEP_MAX))
    return anchors


def _repair_edges(paths: list[dict]) -> list[tuple[tuple[int, int], tuple[int, int], int]]:
    edges = []
    for path in paths:
        for a, b in zip(path["points"], path["points"][1:]):
            if abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1:
                edges.append((a, b, int(path["color"])))
    return edges


def _edge_gap_cells(
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    a: tuple[int, int],
    b: tuple[int, int],
) -> list[tuple[int, int]]:
    if a[0] == b[0]:
        row = row_spans[a[0]][0]
        c0 = col_spans[a[1]][1]
        c1 = col_spans[b[1]][0]
        if c0 > c1:
            c0, c1 = c1, c0
        return [(row, c) for c in range(c0 + 1, c1)]
    col = col_spans[a[1]][0]
    r0 = row_spans[a[0]][1]
    r1 = row_spans[b[0]][0]
    if r0 > r1:
        r0, r1 = r1, r0
    return [(r, col) for r in range(r0 + 1, r1)]


def _choose_marker_cell(blank_cells: list[tuple[int, int]]) -> tuple[int, int]:
    assert blank_cells
    if len(blank_cells) == 1:
        return blank_cells[0]
    return random.choice([blank_cells[0], blank_cells[-1]])


def _make_right_panel(
    proto: np.ndarray,
    paths: list[dict],
) -> tuple[np.ndarray, np.ndarray] | None:
    vertices = _vertex_mask(proto)
    coords = np.argwhere(vertices)
    vertex_rows = sorted(set(map(int, coords[:, 0])))
    vertex_cols = sorted(set(map(int, coords[:, 1])))
    anchor_rows = _sample_anchor_positions(vertex_rows)
    anchor_cols = _sample_anchor_positions(vertex_cols)
    if anchor_rows is None or anchor_cols is None:
        return None

    right_shape = (anchor_rows[-1] + 1, anchor_cols[-1] + 1)
    if right_shape[0] < proto.shape[0] or right_shape[1] < 6:
        return None

    row_spans = _axis_spans(proto.shape[0], vertex_rows, anchor_rows)
    col_spans = _axis_spans(proto.shape[1], vertex_cols, anchor_cols)
    base = _stretch_fragments(proto, right_shape, row_spans, col_spans)

    markers: list[tuple[int, int]] = []
    seen = set()
    for a, b, _color in _repair_edges(paths):
        gap_cells = _edge_gap_cells(row_spans, col_spans, a, b)
        if not gap_cells:
            return None
        marker = _choose_marker_cell(gap_cells)
        if marker in seen:
            return None
        seen.add(marker)
        markers.append(marker)

    if not markers:
        return None

    right = np.zeros(right_shape, dtype=int)
    right[np.ix_(anchor_rows, anchor_cols)] = vertices[np.ix_(vertex_rows, vertex_cols)] * Color.YELLOW
    for r, c in markers:
        if right[r, c] != Color.BLACK:
            return None
        right[r, c] = Color.GREEN

    try:
        repaired = _repair_marked_gaps(base, markers)
    except AssertionError:
        return None
    return right, repaired


def _compose_input(proto: np.ndarray, right: np.ndarray) -> np.ndarray:
    proto_h, proto_w = proto.shape
    right_h, right_w = right.shape
    assert proto_h <= right_h
    grid = np.zeros((right_h, proto_w + 1 + right_w), dtype=int)
    grid[:proto_h, :proto_w] = proto
    if proto_h < right_h:
        grid[proto_h:, :proto_w] = Color.GREY
    grid[:, proto_w] = Color.GREY
    grid[:, proto_w + 1 :] = right
    return grid


def _gen_attempt() -> np.ndarray | None:
    work_shape = (
        random.randint(*WORK_H_RANGE),
        random.randint(*WORK_W_RANGE),
    )
    paths: list[dict] = []
    forbidden: set[tuple[int, int]] = set()

    base_path_count = random.randint(*BASE_PATH_COUNT_RANGE)
    forced_colors = [Color.BLUE, Color.RED] if base_path_count >= 2 and random.random() < 0.7 else []
    while len(forced_colors) < base_path_count:
        forced_colors.append(random.choice(LINE_COLORS))
    random.shuffle(forced_colors)

    for i in range(base_path_count):
        path = _random_turning_path(
            work_shape,
            forbidden,
            require_unit_segment=(i == 0),
        )
        if path is None:
            return None
        path["color"] = int(forced_colors[i])
        paths.append(path)
        forbidden |= _buffered_cells(path["cells"], work_shape)

    proto = _render_paths(work_shape, paths)
    if random.random() < 0.7:
        structural = _place_structural_gap_path(proto, paths)
        if structural is not None:
            paths.append(structural)
            proto = _render_paths(work_shape, paths)

    proto, paths = _crop_proto_and_paths(proto, paths)
    if not _vertex_axes_valid(proto):
        return None

    built = _make_right_panel(proto, paths)
    if built is None:
        return None
    right, expected = built
    if max(right.shape) > 30 or proto.shape[1] + 1 + right.shape[1] > 30:
        return None

    inp = _compose_input(proto, right)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed")
