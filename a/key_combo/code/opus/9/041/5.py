from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
PALETTE = (1, 2, 3, 6, 7, 8, 9)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1]


def _vertex_mask(proto: np.ndarray) -> np.ndarray:
    fg = proto != BLACK
    h, w = proto.shape
    out = np.zeros_like(fg, dtype=bool)
    for r, c in np.argwhere(fg):
        neighbors: list[tuple[int, int]] = []
        for dr, dc in DIRS4:
            rr, cc = int(r) + dr, int(c) + dc
            if 0 <= rr < h and 0 <= cc < w and fg[rr, cc]:
                neighbors.append((dr, dc))
        if len(neighbors) != 2:
            out[int(r), int(c)] = True
            continue
        a, b = neighbors
        if (a[0] + b[0], a[1] + b[1]) != (0, 0):
            out[int(r), int(c)] = True
    return out


def _extract_edges(proto: np.ndarray) -> tuple[np.ndarray, list[tuple[tuple[int, int], tuple[int, int], int]]]:
    fg = proto != BLACK
    vertices = _vertex_mask(proto)
    h, w = proto.shape
    edges: list[tuple[tuple[int, int], tuple[int, int], int]] = []
    seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    for vr, vc in np.argwhere(vertices):
        start = (int(vr), int(vc))
        for dr, dc in DIRS4:
            r, c = int(vr) + dr, int(vc) + dc
            if not (0 <= r < h and 0 <= c < w and fg[r, c]):
                continue
            path = [start]
            while 0 <= r < h and 0 <= c < w and fg[r, c]:
                path.append((int(r), int(c)))
                if vertices[r, c]:
                    end = (int(r), int(c))
                    key = tuple(sorted((start, end)))
                    if key not in seen:
                        seen.add(key)
                        colors = {int(proto[pr, pc]) for pr, pc in path[1:-1] if int(proto[pr, pc]) != BLACK}
                        if not colors:
                            colors = {int(proto[start]), int(proto[end])}
                        assert len(colors) == 1, "each edge must have one visible color"
                        edges.append((start, end, int(next(iter(colors)))))
                    break
                r += dr
                c += dc
            else:
                raise AssertionError("found a dangling segment without a terminal vertex")

    return vertices, edges


def _paint_line_interior(out: np.ndarray, a: tuple[int, int], b: tuple[int, int], color: int) -> None:
    ar, ac = a
    br, bc = b
    if ar == br:
        c0, c1 = sorted((ac, bc))
        if c1 - c0 > 1:
            out[ar, c0 + 1 : c1] = int(color)
        return
    assert ac == bc
    r0, r1 = sorted((ar, br))
    if r1 - r0 > 1:
        out[r0 + 1 : r1, ac] = int(color)


def _stretch(proto: np.ndarray, anchor_mask: np.ndarray) -> np.ndarray:
    vertices, edges = _extract_edges(proto)
    vertex_coords = np.argwhere(vertices)
    assert len(vertex_coords) >= 4, "expected a nontrivial rectilinear graph"

    proto_rows = sorted(set(map(int, vertex_coords[:, 0])))
    proto_cols = sorted(set(map(int, vertex_coords[:, 1])))
    anchor_rows = np.flatnonzero(anchor_mask.any(axis=1)).astype(int).tolist()
    anchor_cols = np.flatnonzero(anchor_mask.any(axis=0)).astype(int).tolist()
    assert len(proto_rows) == len(anchor_rows)
    assert len(proto_cols) == len(anchor_cols)

    expected_pattern = np.array(
        [[bool(vertices[r, c]) for c in proto_cols] for r in proto_rows],
        dtype=bool,
    )
    sampled_pattern = anchor_mask[np.ix_(anchor_rows, anchor_cols)]
    assert np.array_equal(sampled_pattern, expected_pattern), "yellow anchors must encode the prototype vertices"

    row_map = dict(zip(proto_rows, anchor_rows))
    col_map = dict(zip(proto_cols, anchor_cols))
    out = np.zeros(anchor_mask.shape, dtype=int)

    for a, b, color in edges:
        aa = (row_map[a[0]], col_map[a[1]])
        bb = (row_map[b[0]], col_map[b[1]])
        _paint_line_interior(out, aa, bb, color)

    for r, c in np.argwhere(vertices):
        out[row_map[int(r)], col_map[int(c)]] = int(proto[int(r), int(c)])

    return out


def _compress_one_axis(stretched: np.ndarray, compress_rows: bool) -> np.ndarray:
    if compress_rows:
        keep = np.any(stretched != BLACK, axis=1)
        assert np.any(~keep), "row compression must remove visible empty rows"
        return stretched[keep, :]
    keep = np.any(stretched != BLACK, axis=0)
    assert np.any(~keep), "column compression must remove visible empty columns"
    return stretched[:, keep]


def _panel_from_grid(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    separator_cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    assert len(separator_cols) == 1, "expected one full grey separator column"
    sep = separator_cols[0]

    left = grid[:, :sep]
    right = grid[:, sep + 1 :]
    proto_box = _bbox((left != BLACK) & (left != GREY) & (left != YELLOW))
    right_box = _bbox(right != BLACK)
    assert proto_box is not None and right_box is not None
    return _crop(left, proto_box), _crop(right, right_box)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stretch the compact prototype onto the yellow anchor skeleton, then compress only the guide-selected axis.

    Input structure:
    - A single full grey separator column splits a compact colored rectilinear prototype on the left from a right panel.
    - The left prototype is a one-cell-thick graph of horizontal and vertical colored segments on black.
    - The right panel contains yellow anchor pixels encoding the prototype's vertex pattern plus two full grey guide bars.

    Rule:
    - Extract prototype vertices as endpoints, corners, and crossings, and extract each straight edge's color from its interior cells.
    - Map sorted prototype vertex rows to sorted yellow-anchor rows and sorted prototype vertex columns to sorted yellow-anchor columns.
    - Draw the stretched colored graph on the full right-panel canvas, then ignore the yellow anchors and grey guides.
    - If the two guide bars are horizontal, delete every all-black row; if the two guide bars are vertical, delete every all-black column.

    Key insight:
    - The yellow anchors decide where the prototype vertices land, but the guide-bar orientation decides which orthogonal empty margin axis is compressed away.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    proto, panel = _panel_from_grid(g)
    anchor_mask = panel == YELLOW
    assert np.any(anchor_mask)

    grey_mask = panel == GREY
    full_rows = np.flatnonzero(np.all(grey_mask, axis=1)).astype(int).tolist()
    full_cols = np.flatnonzero(np.all(grey_mask, axis=0)).astype(int).tolist()
    horizontal = len(full_rows) == 2 and len(full_cols) == 0
    vertical = len(full_cols) == 2 and len(full_rows) == 0
    assert horizontal ^ vertical, "expected exactly two parallel guide bars"

    stretched = _stretch(proto, anchor_mask)
    return _compress_one_axis(stretched, compress_rows=horizontal)


def _draw_segment(grid: np.ndarray, a: tuple[int, int], b: tuple[int, int], color: int) -> None:
    ar, ac = a
    br, bc = b
    if ar == br:
        c0, c1 = sorted((ac, bc))
        grid[ar, c0 : c1 + 1] = int(color)
    else:
        assert ac == bc
        r0, r1 = sorted((ar, br))
        grid[r0 : r1 + 1, ac] = int(color)


def _perpendicular_dirs(prev: tuple[int, int] | None) -> list[tuple[int, int]]:
    if prev is None:
        return DIRS4[:]
    if prev[0] != 0:
        return [(0, -1), (0, 1)]
    return [(-1, 0), (1, 0)]


def _build_polyline_segments(node_h: int, node_w: int, segment_count: int) -> list[tuple[tuple[int, int], tuple[int, int]]] | None:
    for _ in range(400):
        start = (random.randrange(node_h), random.randrange(node_w))
        used_nodes = {start}
        used_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
        segments: list[tuple[tuple[int, int], tuple[int, int]]] = []

        def dfs(curr: tuple[int, int], prev_dir: tuple[int, int] | None, depth: int) -> bool:
            if depth == segment_count:
                return True

            choices = []
            for dr, dc in _perpendicular_dirs(prev_dir):
                rr, cc = curr
                run: list[tuple[int, int]] = []
                while True:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < node_h and 0 <= nc < node_w):
                        break
                    edge = tuple(sorted(((rr, cc), (nr, nc))))
                    if edge in used_edges or (nr, nc) in used_nodes:
                        break
                    run.append((nr, nc))
                    choices.append(((dr, dc), run[:]))
                    rr, cc = nr, nc
            random.shuffle(choices)
            choices.sort(key=lambda item: len(item[1]), reverse=random.random() < 0.5)

            for direction, run in choices:
                end = run[-1]
                new_edges = [tuple(sorted((curr if i == 0 else run[i - 1], run[i]))) for i in range(len(run))]
                for node in run:
                    used_nodes.add(node)
                for edge in new_edges:
                    used_edges.add(edge)
                segments.append((curr, end))
                if dfs(end, direction, depth + 1):
                    return True
                segments.pop()
                for edge in new_edges:
                    used_edges.remove(edge)
                for node in run:
                    used_nodes.remove(node)
            return False

        if dfs(start, None, 0):
            return segments
    return None


def _choose_segment_colors(segment_count: int) -> list[int]:
    distinct = random.randint(2, min(4, segment_count))
    base = random.sample(PALETTE, distinct)
    colors = [base[i % distinct] for i in range(segment_count)]
    random.shuffle(colors)
    for i in range(1, segment_count):
        if colors[i] == colors[i - 1]:
            swap = next((j for j in range(i + 1, segment_count) if colors[j] != colors[i - 1]), None)
            if swap is not None:
                colors[i], colors[swap] = colors[swap], colors[i]
            else:
                choices = [c for c in base if c != colors[i - 1]]
                colors[i] = random.choice(choices)
    return colors


def _build_prototype() -> np.ndarray | None:
    for _ in range(300):
        node_h = random.randint(4, 5)
        node_w = random.randint(4, 5)
        segment_count = random.randint(4, 5)
        segments = _build_polyline_segments(node_h, node_w, segment_count)
        if segments is None:
            continue

        proto = np.zeros((2 * node_h - 1, 2 * node_w - 1), dtype=int)
        colors = _choose_segment_colors(len(segments))
        vertex_choices: dict[tuple[int, int], list[int]] = {}

        for color, (a_node, b_node) in zip(colors, segments):
            a = (2 * a_node[0], 2 * a_node[1])
            b = (2 * b_node[0], 2 * b_node[1])
            _draw_segment(proto, a, b, color)
            vertex_choices.setdefault(a, []).append(int(color))
            vertex_choices.setdefault(b, []).append(int(color))

        vertices = _vertex_mask(proto)
        actual_vertices = [tuple(map(int, rc)) for rc in np.argwhere(vertices)]
        if len(actual_vertices) < 5:
            continue

        for coord in actual_vertices:
            choices = vertex_choices.get(coord)
            if not choices:
                continue
            proto[coord] = int(random.choice(choices))

        vertices, edges = _extract_edges(proto)
        vertex_coords = np.argwhere(vertices)
        if len(edges) < 4:
            continue
        if len(set(map(int, proto[proto != BLACK]))) < 2:
            continue
        if len(sorted(set(map(int, vertex_coords[:, 0])))) < 3:
            continue
        if len(sorted(set(map(int, vertex_coords[:, 1])))) < 3:
            continue
        return proto
    return None


def _sample_anchor_positions(count: int) -> tuple[list[int], int]:
    margin_lo, margin_hi = 1, 2
    gap_lo, gap_hi = 2, 3
    first = random.randint(margin_lo, margin_hi)
    positions = [first]
    for _ in range(count - 1):
        positions.append(positions[-1] + random.randint(gap_lo, gap_hi))
    size = positions[-1] + random.randint(margin_lo, margin_hi) + 1
    return positions, size


def _assemble_right_panel(proto: np.ndarray, horizontal_guides: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices = _vertex_mask(proto)
    vertex_coords = np.argwhere(vertices)
    proto_rows = sorted(set(map(int, vertex_coords[:, 0])))
    proto_cols = sorted(set(map(int, vertex_coords[:, 1])))
    anchor_rows, panel_h = _sample_anchor_positions(len(proto_rows))
    anchor_cols, panel_w = _sample_anchor_positions(len(proto_cols))

    row_map = dict(zip(proto_rows, anchor_rows))
    col_map = dict(zip(proto_cols, anchor_cols))
    panel = np.zeros((panel_h, panel_w), dtype=int)

    expected_pattern = np.array(
        [[bool(vertices[r, c]) for c in proto_cols] for r in proto_rows],
        dtype=bool,
    )
    for rr, proto_r in enumerate(proto_rows):
        for cc, proto_c in enumerate(proto_cols):
            if expected_pattern[rr, cc]:
                panel[row_map[proto_r], col_map[proto_c]] = YELLOW

    if horizontal_guides:
        panel[0, :] = GREY
        panel[-1, :] = GREY
    else:
        panel[:, 0] = GREY
        panel[:, -1] = GREY

    stretched = _stretch(proto, panel == YELLOW)
    output = _compress_one_axis(stretched, compress_rows=horizontal_guides)
    return panel, stretched, output


def _alt_compress(stretched: np.ndarray, horizontal_guides: bool) -> np.ndarray:
    return _compress_one_axis(stretched, compress_rows=not horizontal_guides)


def _assemble_input(proto: np.ndarray, panel: np.ndarray) -> np.ndarray | None:
    proto_h, proto_w = proto.shape
    panel_h, panel_w = panel.shape
    top_pad = random.randint(1, 3)
    bottom_pad = random.randint(1, 3)
    left_pad = random.randint(0, 2)
    proto_right_pad = random.randint(1, 3)
    right_extra = random.randint(0, 2)

    total_h = panel_h + top_pad + bottom_pad
    left_w = left_pad + proto_w + proto_right_pad
    total_w = left_w + 1 + panel_w + right_extra
    if total_h > 30 or total_w > 30:
        return None

    grid = np.zeros((total_h, total_w), dtype=int)
    sep = left_w
    grid[:, sep] = GREY

    proto_top = random.randint(0, total_h - proto_h)
    proto_left = random.randint(0, left_w - proto_w)
    panel_top = top_pad
    panel_left = sep + 1

    grid[proto_top : proto_top + proto_h, proto_left : proto_left + proto_w] = proto
    existing = grid[panel_top : panel_top + panel_h, panel_left : panel_left + panel_w]
    assert not np.any(existing != BLACK)
    grid[panel_top : panel_top + panel_h, panel_left : panel_left + panel_w] = panel
    return grid


def _semantic_checks(inp: np.ndarray, proto: np.ndarray, panel: np.ndarray, stretched: np.ndarray, out: np.ndarray, horizontal_guides: bool) -> bool:
    try:
        actual = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual, out):
        return False

    alt = _alt_compress(stretched, horizontal_guides)
    if np.array_equal(alt, out):
        return False
    if alt.shape == out.shape and np.array_equal(alt, out):
        return False

    row_blank_count = int(np.sum(~np.any(stretched != BLACK, axis=1)))
    col_blank_count = int(np.sum(~np.any(stretched != BLACK, axis=0)))
    if row_blank_count < 2 or col_blank_count < 2:
        return False

    colors = set(map(int, proto[proto != BLACK]))
    if not (2 <= len(colors) <= 4):
        return False

    vertices = _vertex_mask(proto)
    vertex_coords = np.argwhere(vertices)
    if len(vertex_coords) < 5:
        return False
    if len(set(map(int, vertex_coords[:, 0]))) < 3 or len(set(map(int, vertex_coords[:, 1]))) < 3:
        return False

    guide_rows = np.flatnonzero(np.all(panel == GREY, axis=1)).tolist()
    guide_cols = np.flatnonzero(np.all(panel == GREY, axis=0)).tolist()
    if horizontal_guides:
        if len(guide_rows) != 2 or guide_cols:
            return False
    else:
        if len(guide_cols) != 2 or guide_rows:
            return False

    if max(inp.shape) > 30 or max(out.shape) > 30:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(400):
        proto = _build_prototype()
        if proto is None:
            continue
        horizontal_guides = bool(random.getrandbits(1))
        panel, stretched, out = _assemble_right_panel(proto, horizontal_guides)
        inp = _assemble_input(proto, panel)
        if inp is None:
            continue
        if _semantic_checks(inp, proto, panel, stretched, out, horizontal_guides):
            return inp
    raise RuntimeError("failed to generate a valid sample for new puzzle 5")
