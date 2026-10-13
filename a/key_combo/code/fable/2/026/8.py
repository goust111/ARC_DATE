import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
LATTICE_SHAPES = ((2, 3), (3, 2), (2, 4), (4, 2), (3, 3))
GEN_RETRY_LIMIT = 500


# Local copies of a few stable ARC helpers keep this file self-contained while
# following the shared (r, c) / bbox conventions from arc_utils.py.
def _bbox(mask: np.ndarray):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box):
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _draw_line(grid: np.ndarray, start, end, color: int) -> None:
    r0, c0 = map(int, start)
    r1, c1 = map(int, end)
    color = int(color)
    if r0 == r1:
        c_lo, c_hi = sorted((c0, c1))
        grid[r0, c_lo : c_hi + 1] = color
        return
    if c0 == c1:
        r_lo, r_hi = sorted((r0, r1))
        grid[r_lo : r_hi + 1, c0] = color
        return
    raise AssertionError("expected an orthogonal line")


def _has_adjacent_foreground(panel: np.ndarray) -> bool:
    fg = (panel != BLACK) & (panel != GREY)
    return bool(np.any(fg[:-1, :] & fg[1:, :]) or np.any(fg[:, :-1] & fg[:, 1:]))


def _split_panels(grid: np.ndarray):
    grey_rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == GREY)]
    grey_cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]

    if len(grey_rows) == 1 and not grey_cols:
        sep = grey_rows[0]
        first = grid[:sep, :]
        second = grid[sep + 1 :, :]
    elif len(grey_cols) == 1 and not grey_rows:
        sep = grey_cols[0]
        first = grid[:, :sep]
        second = grid[:, sep + 1 :]
    else:
        raise AssertionError("expected exactly one full grey separator")

    if _has_adjacent_foreground(first) and not _has_adjacent_foreground(second):
        proto_panel, canvas_panel = first, second
    elif _has_adjacent_foreground(second) and not _has_adjacent_foreground(first):
        proto_panel, canvas_panel = second, first
    else:
        raise AssertionError("failed to distinguish prototype and canvas panels")

    proto_box = _bbox((proto_panel != BLACK) & (proto_panel != GREY))
    assert proto_box is not None
    proto = _crop_bbox(proto_panel, proto_box)
    assert not np.any(proto == GREY)
    assert not np.any(canvas_panel == GREY)
    return proto, canvas_panel.copy()


def _vertex_mask(proto: np.ndarray) -> np.ndarray:
    fg = proto != BLACK
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


def _trace_prototype(proto: np.ndarray):
    fg = proto != BLACK
    vertices = _vertex_mask(proto)
    vertex_coords = [tuple(map(int, rc)) for rc in np.argwhere(vertices)]
    assert vertex_coords

    coord_to_index = {coord: idx for idx, coord in enumerate(vertex_coords)}
    vertex_colors = [int(proto[r, c]) for r, c in vertex_coords]
    assert len(set(vertex_colors)) == len(vertex_colors)

    h_neighbors = {idx: [] for idx in range(len(vertex_coords))}
    v_neighbors = {idx: [] for idx in range(len(vertex_coords))}
    edges = []
    seen = set()

    for idx, (r, c) in enumerate(vertex_coords):
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = r + dr, c + dc
            if not (0 <= rr < proto.shape[0] and 0 <= cc < proto.shape[1] and fg[rr, cc]):
                continue

            path = []
            while True:
                assert 0 <= rr < proto.shape[0] and 0 <= cc < proto.shape[1] and fg[rr, cc]
                if vertices[rr, cc]:
                    other = coord_to_index[(rr, cc)]
                    key = tuple(sorted((idx, other)))
                    if key not in seen:
                        seen.add(key)
                        assert path, "generator guarantees a visible segment color"
                        colors = {int(proto[pr, pc]) for pr, pc in path}
                        assert len(colors) == 1
                        axis = "v" if dr else "h"
                        color = colors.pop()
                        edges.append((idx, other, axis, color))
                        if axis == "h":
                            h_neighbors[idx].append(other)
                            h_neighbors[other].append(idx)
                        else:
                            v_neighbors[idx].append(other)
                            v_neighbors[other].append(idx)
                    break
                path.append((rr, cc))
                rr += dr
                cc += dc

    assert edges
    return vertex_coords, vertex_colors, h_neighbors, v_neighbors, edges


def _infer_positions(vertex_count: int, h_neighbors, v_neighbors, known_positions, max_rounds=None):
    positions = {int(k): (int(v[0]), int(v[1])) for k, v in known_positions.items()}
    depths = {int(k): 0 for k in positions}
    rounds = 0

    while True:
        if max_rounds is not None and rounds >= max_rounds:
            break

        newly = []
        for idx in range(vertex_count):
            if idx in positions:
                continue
            row_candidates = {positions[nbr][0] for nbr in h_neighbors[idx] if nbr in positions}
            col_candidates = {positions[nbr][1] for nbr in v_neighbors[idx] if nbr in positions}
            if len(row_candidates) > 1 or len(col_candidates) > 1:
                return None, None
            if len(row_candidates) == 1 and len(col_candidates) == 1:
                newly.append((idx, (next(iter(row_candidates)), next(iter(col_candidates)))))

        if not newly:
            break

        rounds += 1
        for idx, pos in newly:
            positions[idx] = pos
            depths[idx] = rounds

    return positions, depths


def _render_output(canvas_shape, positions, vertex_colors, edges):
    out = np.zeros(canvas_shape, dtype=int)
    for a, b, _axis, color in edges:
        _draw_line(out, positions[a], positions[b], color)
    for idx, (r, c) in positions.items():
        out[int(r), int(c)] = int(vertex_colors[idx])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the stretched colored skeleton on the large canvas by matching vertex colors and inferring missing anchors.

    Input structure:
    - A full grey separator row or column splits the grid into two panels.
    - One panel contains a compact orthogonal prototype made of horizontal and vertical colored segments on black.
    - Every prototype endpoint, corner, T-junction, and crossing is a uniquely colored vertex pixel.
    - The other panel is a larger black canvas showing only some isolated anchor pixels whose colors match a subset of the prototype vertices.

    Rule:
    - Use color equality to match every visible canvas anchor to its prototype vertex.
    - For each missing prototype vertex, infer its canvas position from the row of a known horizontally connected neighbor and the column of a known vertically connected neighbor, repeating until every vertex is located.
    - Draw every prototype segment on the canvas with the segment's own color between its two inferred endpoint positions.
    - Keep the vertex pixels themselves in their unique vertex colors and return only the completed canvas panel.

    Key insight:
    - Sorted row or column order is unreliable because the canvas independently permutes prototype rows and columns; only vertex colors identify the anchors, and missing vertices are recovered by orthogonal row/column constraints from adjacent graph neighbors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    proto, canvas = _split_panels(g)
    vertex_coords, vertex_colors, h_neighbors, v_neighbors, edges = _trace_prototype(proto)

    anchor_coords = [tuple(map(int, rc)) for rc in np.argwhere(canvas != BLACK)]
    anchor_colors = [int(canvas[r, c]) for r, c in anchor_coords]
    assert len(set(anchor_colors)) == len(anchor_colors)

    color_to_vertex = {color: idx for idx, color in enumerate(vertex_colors)}
    known_positions = {}
    for coord, color in zip(anchor_coords, anchor_colors):
        assert color in color_to_vertex
        known_positions[color_to_vertex[color]] = coord

    positions, _depths = _infer_positions(len(vertex_coords), h_neighbors, v_neighbors, known_positions)
    assert positions is not None and len(positions) == len(vertex_coords)
    assert len(set(positions.values())) == len(vertex_coords)
    return _render_output(canvas.shape, positions, vertex_colors, edges)


def _sample_connected_cells(rng, row_count: int, col_count: int, target_count: int):
    for _ in range(200):
        start = (int(rng.integers(row_count)), int(rng.integers(col_count)))
        cells = {start}
        while len(cells) < target_count:
            frontier = []
            for r, c in cells:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < row_count and 0 <= cc < col_count and (rr, cc) not in cells:
                        frontier.append((rr, cc))
            if not frontier:
                break
            cells.add(frontier[int(rng.integers(len(frontier)))])

        if len(cells) != target_count:
            continue
        if len(cells) == row_count * col_count:
            continue
        if any(all((r, c) not in cells for c in range(col_count)) for r in range(row_count)):
            continue
        if any(all((r, c) not in cells for r in range(row_count)) for c in range(col_count)):
            continue

        ordered = sorted(cells)
        _, _, h_neighbors, v_neighbors, _ = _graph_from_cells(ordered)
        if any((len(h_neighbors[i]) == 2 and not v_neighbors[i]) or (len(v_neighbors[i]) == 2 and not h_neighbors[i]) for i in range(len(ordered))):
            continue
        orth_count = sum(bool(h_neighbors[i]) and bool(v_neighbors[i]) for i in range(len(ordered)))
        if orth_count == 0:
            continue
        return ordered
    return None


def _graph_from_cells(cells):
    index = {cell: idx for idx, cell in enumerate(cells)}
    h_neighbors = {idx: [] for idx in range(len(cells))}
    v_neighbors = {idx: [] for idx in range(len(cells))}
    edges = []

    for idx, (r, c) in enumerate(cells):
        right = index.get((r, c + 1))
        down = index.get((r + 1, c))
        if right is not None:
            h_neighbors[idx].append(right)
            h_neighbors[right].append(idx)
            edges.append((idx, right, "h"))
        if down is not None:
            v_neighbors[idx].append(down)
            v_neighbors[down].append(idx)
            edges.append((idx, down, "v"))

    return index, cells, h_neighbors, v_neighbors, edges


def _choose_visible_subset(cells, h_neighbors, v_neighbors, rng):
    vertex_count = len(cells)
    direct = []
    recursive = []
    truth_positions = {idx: cells[idx] for idx in range(vertex_count)}

    for mask in range(1, (1 << vertex_count) - 1):
        visible = {idx for idx in range(vertex_count) if (mask >> idx) & 1}
        if len(visible) < 2:
            continue
        missing = vertex_count - len(visible)
        if missing > 4:
            continue
        positions, depths = _infer_positions(vertex_count, h_neighbors, v_neighbors, {i: truth_positions[i] for i in visible})
        if positions is None or len(positions) != vertex_count:
            continue
        max_depth = max(depths.values())
        payload = (sorted(visible), max_depth)
        if max_depth == 1:
            direct.append(payload)
        elif max_depth >= 2:
            recursive.append(payload)

    if recursive and int(rng.integers(100)) < 65:
        visible, max_depth = recursive[int(rng.integers(len(recursive)))]
        return visible, max_depth, "recursive"
    if direct:
        visible, max_depth = direct[int(rng.integers(len(direct)))]
        return visible, max_depth, "direct"
    if recursive:
        visible, max_depth = recursive[int(rng.integers(len(recursive)))]
        return visible, max_depth, "recursive"
    return None


def _axis_positions(count: int, rng, gap_min: int, gap_max: int, lead_max: int, trail_max: int):
    positions = [int(rng.integers(lead_max + 1))]
    for _ in range(count - 1):
        positions.append(positions[-1] + int(rng.integers(gap_min, gap_max + 1)))
    size = positions[-1] + 1 + int(rng.integers(trail_max + 1))
    return positions, size


def _render_prototype(cells, row_positions, col_positions, vertex_colors, edges, edge_colors):
    proto = np.zeros((row_positions[-1] + 1, col_positions[-1] + 1), dtype=int)
    for (a, b, _axis), color in zip(edges, edge_colors):
        ra, ca = cells[a]
        rb, cb = cells[b]
        _draw_line(proto, (row_positions[ra], col_positions[ca]), (row_positions[rb], col_positions[cb]), color)
    for idx, (r, c) in enumerate(cells):
        proto[row_positions[r], col_positions[c]] = int(vertex_colors[idx])
    return proto


def _render_canvas(canvas_shape, cells, row_map, col_map, visible, vertex_colors):
    canvas = np.zeros(canvas_shape, dtype=int)
    positions = {}
    for idx, (r, c) in enumerate(cells):
        positions[idx] = (int(row_map[r]), int(col_map[c]))
    for idx in visible:
        rr, cc = positions[idx]
        canvas[rr, cc] = int(vertex_colors[idx])
    return canvas, positions


def _wrong_sorted_output(canvas_shape, cells, visible, true_positions, h_neighbors, v_neighbors, vertex_colors, edges):
    visible = sorted(visible, key=lambda idx: cells[idx])
    anchors = sorted((true_positions[idx] for idx in visible))
    guessed = {idx: coord for idx, coord in zip(visible, anchors)}
    positions, _depths = _infer_positions(len(cells), h_neighbors, v_neighbors, guessed)
    if positions is None or len(positions) != len(cells):
        return None
    try:
        return _render_output(canvas_shape, positions, vertex_colors, edges)
    except AssertionError:
        return None


def _compose_input(proto: np.ndarray, canvas: np.ndarray, vertical: bool, proto_first: bool):
    proto_h, proto_w = proto.shape
    canvas_h, canvas_w = canvas.shape

    if vertical:
        assert canvas_h >= proto_h
        grid = np.zeros((canvas_h, proto_w + 1 + canvas_w), dtype=int)
        if proto_first:
            grid[:proto_h, :proto_w] = proto
            grid[proto_h:, :proto_w] = GREY
            grid[:, proto_w] = GREY
            grid[:, proto_w + 1 :] = canvas
        else:
            grid[:, :canvas_w] = canvas
            grid[:, canvas_w] = GREY
            grid[:proto_h, canvas_w + 1 :] = proto
            grid[proto_h:, canvas_w + 1 :] = GREY
        return grid

    assert canvas_w >= proto_w
    grid = np.zeros((proto_h + 1 + canvas_h, canvas_w), dtype=int)
    if proto_first:
        grid[:proto_h, :proto_w] = proto
        grid[:proto_h, proto_w:] = GREY
        grid[proto_h, :] = GREY
        grid[proto_h + 1 :, :] = canvas
    else:
        grid[:canvas_h, :] = canvas
        grid[canvas_h, :] = GREY
        grid[canvas_h + 1 : canvas_h + 1 + proto_h, :proto_w] = proto
        grid[canvas_h + 1 : canvas_h + 1 + proto_h, proto_w:] = GREY
    return grid


def _build_instance(rng):
    row_count, col_count = LATTICE_SHAPES[int(rng.integers(len(LATTICE_SHAPES)))]
    min_count = max(4, row_count + col_count - 1)
    max_count = min(8, row_count * col_count - 1)
    target_count = int(rng.integers(min_count, max_count + 1))
    cells = _sample_connected_cells(rng, row_count, col_count, target_count)
    if cells is None:
        return None

    _index, cells, h_neighbors, v_neighbors, edges = _graph_from_cells(cells)
    subset = _choose_visible_subset(cells, h_neighbors, v_neighbors, rng)
    if subset is None:
        return None
    visible, max_depth, mode = subset

    proto_rows = [0]
    for _ in range(row_count - 1):
        proto_rows.append(proto_rows[-1] + int(rng.integers(2, 4)))
    proto_cols = [0]
    for _ in range(col_count - 1):
        proto_cols.append(proto_cols[-1] + int(rng.integers(2, 4)))

    canvas_row_positions, canvas_h = _axis_positions(row_count, rng, 3, 4, 2, 2)
    canvas_col_positions, canvas_w = _axis_positions(col_count, rng, 3, 4, 2, 2)

    row_perm = rng.permutation(row_count)
    col_perm = rng.permutation(col_count)
    if np.array_equal(row_perm, np.arange(row_count)) and np.array_equal(col_perm, np.arange(col_count)):
        if col_count > 1:
            col_perm = col_perm[::-1]
        else:
            row_perm = row_perm[::-1]

    row_map = {r: int(canvas_row_positions[int(row_perm[r])]) for r in range(row_count)}
    col_map = {c: int(canvas_col_positions[int(col_perm[c])]) for c in range(col_count)}

    vertex_colors = rng.choice(PALETTE, size=len(cells), replace=False).astype(int).tolist()
    edge_palette_size = min(len(PALETTE), max(2, min(4, len(edges))))
    edge_palette = rng.choice(PALETTE, size=edge_palette_size, replace=False).astype(int).tolist()
    edge_colors = [int(edge_palette[int(rng.integers(len(edge_palette)))]) for _ in edges]
    if len(edges) > 1 and len(set(edge_colors)) == 1:
        edge_colors[int(rng.integers(len(edge_colors)))] = edge_palette[-1]

    proto = _render_prototype(cells, proto_rows, proto_cols, vertex_colors, edges, edge_colors)
    canvas, true_positions = _render_canvas((canvas_h, canvas_w), cells, row_map, col_map, visible, vertex_colors)
    expected = _render_output((canvas_h, canvas_w), true_positions, vertex_colors, [(a, b, axis, color) for (a, b, axis), color in zip(edges, edge_colors)])

    orientations = []
    if canvas_h >= proto.shape[0] and proto.shape[1] + 1 + canvas_w <= 30:
        orientations.append(True)
    if canvas_w >= proto.shape[1] and proto.shape[0] + 1 + canvas_h <= 30:
        orientations.append(False)
    if not orientations:
        return None

    vertical = bool(orientations[int(rng.integers(len(orientations)))])
    proto_first = bool(int(rng.integers(2)))
    inp = _compose_input(proto, canvas, vertical, proto_first)

    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return None
    if np.count_nonzero(expected) <= np.count_nonzero(canvas):
        return None
    if mode == "recursive":
        one_pass, _ = _infer_positions(len(cells), h_neighbors, v_neighbors, {idx: true_positions[idx] for idx in visible}, max_rounds=1)
        if one_pass is None or len(one_pass) == len(cells):
            return None

    wrong = _wrong_sorted_output(expected.shape, cells, visible, true_positions, h_neighbors, v_neighbors, vertex_colors, [(a, b, axis, color) for (a, b, axis), color in zip(edges, edge_colors)])
    if wrong is not None and np.array_equal(wrong, expected):
        return None

    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRY_LIMIT):
        grid = _build_instance(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRY_LIMIT} attempts")
