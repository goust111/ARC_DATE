from __future__ import annotations

import random

import numpy as np


NODE_SIZE = 5
LEGEND_H = 4
LEGEND_W = 9
WIDTH_OFFSETS = {
    1: (2,),
    2: (1, 2),
    3: (1, 2, 3),
}
SIDE_TO_VEC = {
    "N": (-1, 0),
    "E": (0, 1),
    "S": (1, 0),
    "W": (0, -1),
}
VEC_TO_SIDE = {value: key for key, value in SIDE_TO_VEC.items()}
OPPOSITE = {
    "N": "S",
    "E": "W",
    "S": "N",
    "W": "E",
}
DIGIT_MASKS = {
    1: np.array([[0, 1, 0], [1, 1, 0], [0, 1, 0]], dtype=bool),
    2: np.array([[1, 1, 1], [0, 1, 1], [1, 1, 0]], dtype=bool),
    3: np.array([[1, 1, 1], [0, 1, 1], [1, 1, 1]], dtype=bool),
}
ROTATIONS = {
    "W": 0,
    "N": 1,
    "E": 2,
    "S": 3,
}
TEMPLATES = ("fan", "split", "cross", "chain")
GRID_MIN = 22
GRID_MAX = 30
GAP_RANGE = (2, 4)
LEAF_RANGE = (2, 4)
GEN_ATTEMPTS = 400


def _bg_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[cnts.argmax()])


def _orth_neighbors(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in SIDE_TO_VEC.values():
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _find_pipe_color(grid: np.ndarray, bg: int) -> int:
    colors = [int(color) for color in np.unique(grid) if int(color) != bg]
    assert colors, "expected at least one non-background color"
    return max(colors, key=lambda color: int(np.sum(grid == color)))


def _corner_origins(h: int, w: int) -> list[tuple[int, int]]:
    return [
        (0, 0),
        (0, w - LEGEND_W),
        (h - LEGEND_H, 0),
        (h - LEGEND_H, w - LEGEND_W),
    ]


def _find_legend(grid: np.ndarray, bg: int, pipe_color: int) -> tuple[dict[int, int], tuple[int, int, int, int]]:
    h, w = grid.shape
    for r0, c0 in _corner_origins(h, w):
        if r0 < 0 or c0 < 0:
            continue
        region = grid[r0 : r0 + LEGEND_H, c0 : c0 + LEGEND_W]
        if region.shape != (LEGEND_H, LEGEND_W):
            continue

        swatches: list[int] = []
        ok = True
        for i in range(3):
            block = region[0, 3 * i : 3 * (i + 1)]
            if len(set(map(int, block.tolist()))) != 1:
                ok = False
                break
            color = int(block[0])
            if color in {bg, pipe_color}:
                ok = False
                break
            swatches.append(color)
        if not ok or len(set(swatches)) != 3:
            continue

        tail = region[1:, :]
        digit_colors = set(map(int, np.unique(tail))) - {bg}
        if len(digit_colors) != 1:
            continue
        digit_color = digit_colors.pop()
        if digit_color == pipe_color or digit_color in swatches:
            continue

        good = True
        for idx, digit in enumerate((1, 2, 3)):
            block = tail[:, 3 * idx : 3 * (idx + 1)]
            if not np.array_equal(block == digit_color, DIGIT_MASKS[digit]):
                good = False
                break
        if not good:
            continue

        return {1: swatches[0], 2: swatches[1], 3: swatches[2]}, (r0, c0, r0 + LEGEND_H, c0 + LEGEND_W)

    raise AssertionError("failed to locate the corner width legend")


def _find_entry(grid: np.ndarray, bg: int, pipe_color: int) -> tuple[tuple[int, int], tuple[int, int], str, int]:
    h, w = grid.shape
    border_cells = []
    for c in range(w):
        border_cells.append((0, c))
        border_cells.append((h - 1, c))
    for r in range(1, h - 1):
        border_cells.append((r, 0))
        border_cells.append((r, w - 1))

    seen = set()
    candidates = []
    for r, c in border_cells:
        if (r, c) in seen:
            continue
        seen.add((r, c))
        color = int(grid[r, c])
        if color in {bg, pipe_color}:
            continue
        nbrs = [(rr, cc) for rr, cc in _orth_neighbors(r, c, h, w) if int(grid[rr, cc]) == pipe_color]
        if len(nbrs) == 1:
            (sr, sc) = nbrs[0]
            candidates.append(((r, c), (sr, sc), VEC_TO_SIDE[(sr - r, sc - c)], color))

    assert len(candidates) == 1, "expected exactly one border entry pixel"
    return candidates[0]


def _find_nodes(pipe_mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = pipe_mask.shape
    nodes = []
    for r in range(h - NODE_SIZE + 1):
        for c in range(w - NODE_SIZE + 1):
            if bool(np.all(pipe_mask[r : r + NODE_SIZE, c : c + NODE_SIZE])):
                nodes.append((r, c))
    assert nodes, "expected at least one junction node"
    return nodes


def _node_index_grid(shape: tuple[int, int], nodes: list[tuple[int, int]]) -> np.ndarray:
    out = np.full(shape, -1, dtype=int)
    for idx, (r0, c0) in enumerate(nodes):
        window = out[r0 : r0 + NODE_SIZE, c0 : c0 + NODE_SIZE]
        assert np.all(window == -1), "node boxes may not overlap"
        window[:, :] = idx
    return out


def _runs(indices: list[int]) -> list[list[int]]:
    if not indices:
        return []
    chunks = [[indices[0]]]
    for value in indices[1:]:
        if value == chunks[-1][-1] + 1:
            chunks[-1].append(value)
        else:
            chunks.append([value])
    return chunks


def _node_openings(pipe_mask: np.ndarray, nodes: list[tuple[int, int]]) -> list[dict[str, list[tuple[int, int]]]]:
    h, w = pipe_mask.shape
    out: list[dict[str, list[tuple[int, int]]]] = []
    for r0, c0 in nodes:
        openings: dict[str, list[tuple[int, int]]] = {}
        checks = {
            "N": (r0 - 1, list(range(c0, c0 + NODE_SIZE))),
            "S": (r0 + NODE_SIZE, list(range(c0, c0 + NODE_SIZE))),
            "W": (c0 - 1, list(range(r0, r0 + NODE_SIZE))),
            "E": (c0 + NODE_SIZE, list(range(r0, r0 + NODE_SIZE))),
        }

        for side, data in checks.items():
            if side in {"N", "S"}:
                rr, cols = data
                if not (0 <= rr < h):
                    continue
                idxs = [i for i, cc in enumerate(cols) if bool(pipe_mask[rr, cc])]
                groups = _runs(idxs)
                if not groups:
                    continue
                assert len(groups) == 1 and 1 <= len(groups[0]) <= 3, "side opening must be a single width-1/2/3 run"
                openings[side] = [(rr, cols[i]) for i in groups[0]]
            else:
                cc, rows = data
                if not (0 <= cc < w):
                    continue
                idxs = [i for i, rr in enumerate(rows) if bool(pipe_mask[rr, cc])]
                groups = _runs(idxs)
                if not groups:
                    continue
                assert len(groups) == 1 and 1 <= len(groups[0]) <= 3, "side opening must be a single width-1/2/3 run"
                openings[side] = [(rows[i], cc) for i in groups[0]]
        out.append(openings)
    return out


def _trace_corridor(
    pipe_mask: np.ndarray,
    node_ids: np.ndarray,
    start_cells: list[tuple[int, int]],
    side: str,
) -> tuple[set[tuple[int, int]], int | None]:
    dr, dc = SIDE_TO_VEC[side]
    h, w = pipe_mask.shape
    current = list(start_cells)
    segment = set(current)

    while True:
        next_cells = [(r + dr, c + dc) for r, c in current]
        if any(not (0 <= rr < h and 0 <= cc < w) for rr, cc in next_cells):
            return segment, None

        hit_nodes = {int(node_ids[rr, cc]) for rr, cc in next_cells if int(node_ids[rr, cc]) >= 0}
        if hit_nodes:
            assert len(hit_nodes) == 1, "corridor should enter at most one node at a time"
            return segment, next(iter(hit_nodes))

        if not all(bool(pipe_mask[rr, cc]) for rr, cc in next_cells):
            return segment, None

        segment.update(next_cells)
        current = next_cells


def _paint_cells(out: np.ndarray, cells: set[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        out[r, c] = int(color)


def _propagate_from_node(
    out: np.ndarray,
    pipe_mask: np.ndarray,
    nodes: list[tuple[int, int]],
    node_ids: np.ndarray,
    openings: list[dict[str, list[tuple[int, int]]]],
    node_idx: int,
    incoming_side: str,
    current_color: int,
    width_colors: dict[int, int],
    seen: set[int],
) -> None:
    if node_idx in seen:
        return
    seen.add(node_idx)

    r0, c0 = nodes[node_idx]
    out[r0 : r0 + NODE_SIZE, c0 : c0 + NODE_SIZE] = int(current_color)

    for side, cells in openings[node_idx].items():
        if side == incoming_side:
            continue
        width = len(cells)
        branch_color = int(width_colors[width])
        segment, target = _trace_corridor(pipe_mask, node_ids, cells, side)
        _paint_cells(out, segment, branch_color)
        if target is not None:
            _propagate_from_node(
                out,
                pipe_mask,
                nodes,
                node_ids,
                openings,
                target,
                OPPOSITE[side],
                branch_color,
                width_colors,
                seen,
            )


def main(grid: np.ndarray) -> np.ndarray:
    """Propagate a signal through the grey pipe network and recolor each downstream branch by the legend color for that branch width.

    Input structure:
    - The background is the dominant color.
    - One non-background color forms a connected orthogonal pipe network whose channels have width 1, 2, or 3.
    - The network has a single border entry pixel of another color; the pipe cell next to it is the unique entrance endpoint.
    - A corner legend shows three colored swatches in left-to-right order, labeled by dot-matrix digits 1, 2, and 3 underneath.

    Rule:
    - Read the legend as the width-to-color mapping 1 -> left swatch, 2 -> middle swatch, 3 -> right swatch.
    - Starting from the entry pixel, follow the pipe inward and recolor the initial straight segment with the entry color.
    - Whenever the signal reaches a junction node, recolor that node with the incoming color, then continue into every outgoing branch.
    - For each outgoing branch, switch to the legend color that matches that branch's width and keep propagating until the next node or a dead end.

    Key insight:
    - The entry only determines the pre-branch color; after each junction, the visible branch width becomes the lookup key that selects the next signal color from the legend.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _bg_color(g)
    pipe_color = _find_pipe_color(g, bg)
    width_colors, _legend_bbox = _find_legend(g, bg, pipe_color)
    (_entry_r, _entry_c), start_pipe, start_side, entry_color = _find_entry(g, bg, pipe_color)

    pipe_mask = g == pipe_color
    nodes = _find_nodes(pipe_mask)
    node_ids = _node_index_grid(g.shape, nodes)
    openings = _node_openings(pipe_mask, nodes)

    out = g.copy()
    segment, first_node = _trace_corridor(pipe_mask, node_ids, [start_pipe], start_side)
    _paint_cells(out, segment, entry_color)

    if first_node is not None:
        _propagate_from_node(
            out,
            pipe_mask,
            nodes,
            node_ids,
            openings,
            first_node,
            OPPOSITE[start_side],
            entry_color,
            width_colors,
            set(),
        )

    assert not np.any((out == pipe_color) & pipe_mask), "all pipe pixels should be recolored"
    return out


def _node_box(top_left: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0 = top_left
    return {
        (r, c)
        for r in range(r0, r0 + NODE_SIZE)
        for c in range(c0, c0 + NODE_SIZE)
    }


def _shift_node(top_left: tuple[int, int], side: str, gap: int) -> tuple[int, int]:
    r0, c0 = top_left
    if side == "N":
        return r0 - NODE_SIZE - gap, c0
    if side == "S":
        return r0 + NODE_SIZE + gap, c0
    if side == "W":
        return r0, c0 - NODE_SIZE - gap
    return r0, c0 + NODE_SIZE + gap


def _corridor_cells_between(
    source: tuple[int, int],
    side: str,
    width: int,
    target: tuple[int, int] | None,
    leaf_len: int | None,
) -> set[tuple[int, int]]:
    offsets = WIDTH_OFFSETS[width]
    cells: set[tuple[int, int]] = set()
    sr, sc = source

    if side == "E":
        end_c = target[1] - 1 if target is not None else sc + NODE_SIZE + int(leaf_len) - 1
        for c in range(sc + NODE_SIZE, end_c + 1):
            for off in offsets:
                cells.add((sr + off, c))
    elif side == "W":
        start_c = target[1] + NODE_SIZE if target is not None else sc - int(leaf_len)
        for c in range(start_c, sc):
            for off in offsets:
                cells.add((sr + off, c))
    elif side == "S":
        end_r = target[0] - 1 if target is not None else sr + NODE_SIZE + int(leaf_len) - 1
        for r in range(sr + NODE_SIZE, end_r + 1):
            for off in offsets:
                cells.add((r, sc + off))
    else:
        start_r = target[0] + NODE_SIZE if target is not None else sr - int(leaf_len)
        for r in range(start_r, sr):
            for off in offsets:
                cells.add((r, sc + off))
    return cells


def _entry_corridor(root: tuple[int, int], length: int) -> tuple[set[tuple[int, int]], tuple[int, int], tuple[int, int]]:
    rr, rc = root
    row = rr + WIDTH_OFFSETS[1][0]
    pipe = {(row, c) for c in range(rc - length, rc)}
    start_pipe = (row, rc - length)
    entry = (row, rc - length - 1)
    return pipe, start_pipe, entry


def _rotate_point(point: tuple[int, int], turns: int) -> tuple[int, int]:
    r, c = point
    for _ in range(turns):
        r, c = c, -r
    return int(r), int(c)


def _rotate_points(points: set[tuple[int, int]], turns: int) -> set[tuple[int, int]]:
    return {_rotate_point(point, turns) for point in points}


def _legend_origin(corner: str, h: int, w: int) -> tuple[int, int]:
    if corner == "TL":
        return 0, 0
    if corner == "TR":
        return 0, w - LEGEND_W
    if corner == "BL":
        return h - LEGEND_H, 0
    return h - LEGEND_H, w - LEGEND_W


def _legend_cells(corner: str, h: int, w: int, margin: int = 0) -> set[tuple[int, int]]:
    r0, c0 = _legend_origin(corner, h, w)
    cells = set()
    for r in range(max(0, r0 - margin), min(h, r0 + LEGEND_H + margin)):
        for c in range(max(0, c0 - margin), min(w, c0 + LEGEND_W + margin)):
            cells.add((r, c))
    return cells


def _network_from_template() -> tuple[set[tuple[int, int]], tuple[int, int], tuple[int, int]]:
    template = random.choice(TEMPLATES)
    nodes = {"root": (0, 0)}
    edges: list[tuple[str, str, str | None]] = []

    if template == "fan":
        edges = [("root", "E", None), ("root", "N", None), ("root", "S", None)]
    elif template == "split":
        nodes["a"] = _shift_node(nodes["root"], "E", random.randint(*GAP_RANGE))
        root_leaf = random.choice(["N", "S"])
        edges = [("root", "E", "a"), ("root", root_leaf, None), ("a", "N", None), ("a", "S", None)]
    elif template == "cross":
        nodes["a"] = _shift_node(nodes["root"], "E", random.randint(*GAP_RANGE))
        root_leaf = random.choice(["N", "S"])
        edges = [("root", "E", "a"), ("root", root_leaf, None), ("a", "E", None), ("a", "N", None), ("a", "S", None)]
    else:
        nodes["a"] = _shift_node(nodes["root"], "E", random.randint(*GAP_RANGE))
        nodes["b"] = _shift_node(nodes["a"], "E", random.randint(*GAP_RANGE))
        root_leaf = random.choice(["N", "S"])
        mid_leaf = "S" if root_leaf == "N" else "N"
        edges = [("root", "E", "a"), ("root", root_leaf, None), ("a", "E", "b"), ("a", mid_leaf, None), ("b", "N", None), ("b", "S", None)]

    widths = [1, 2, 3] + [random.choice([1, 2, 3]) for _ in range(len(edges) - 3)]
    random.shuffle(widths)

    pipe = set()
    for pos in nodes.values():
        pipe |= _node_box(pos)

    for (src, side, target_name), width in zip(edges, widths, strict=False):
        source_pos = nodes[src]
        target_pos = nodes[target_name] if target_name is not None else None
        leaf_len = None if target_pos is not None else random.randint(*LEAF_RANGE)
        pipe |= _corridor_cells_between(source_pos, side, width, target_pos, leaf_len)

    entry_pipe, start_pipe, entry = _entry_corridor(nodes["root"], random.randint(*LEAF_RANGE))
    pipe |= entry_pipe
    assert entry not in pipe
    return pipe, start_pipe, entry


def _required_canvas(
    pipe: set[tuple[int, int]],
    entry: tuple[int, int],
    entry_side: str,
) -> tuple[int, int]:
    rows = [r for r, _ in pipe]
    cols = [c for _, c in pipe]
    er, ec = entry

    if entry_side == "W":
        need_h = max(rows) - min(rows) + 5
        need_w = max(cols) - ec + 3
    elif entry_side == "E":
        need_h = max(rows) - min(rows) + 5
        need_w = ec - min(cols) + 3
    elif entry_side == "N":
        need_h = max(rows) - er + 3
        need_w = max(cols) - min(cols) + 5
    else:
        need_h = er - min(rows) + 3
        need_w = max(cols) - min(cols) + 5

    need_h = max(need_h, GRID_MIN, LEGEND_H + 4)
    need_w = max(need_w, GRID_MIN, LEGEND_W + 4)
    return int(need_h), int(need_w)


def _place_network(
    pipe: set[tuple[int, int]],
    start_pipe: tuple[int, int],
    entry: tuple[int, int],
    entry_side: str,
    legend_corner: str,
    h: int,
    w: int,
) -> tuple[set[tuple[int, int]], tuple[int, int], tuple[int, int]] | None:
    rows = [r for r, _ in pipe]
    cols = [c for _, c in pipe]
    er, ec = entry

    if entry_side in {"W", "E"}:
        if entry_side == "W":
            tc = -ec
        else:
            tc = (w - 1) - ec
        low = 2 - min(rows)
        high = (h - 3) - max(rows)
        shifts = [(tr, tc) for tr in range(low, high + 1)]
    else:
        if entry_side == "N":
            tr = -er
        else:
            tr = (h - 1) - er
        low = 2 - min(cols)
        high = (w - 3) - max(cols)
        shifts = [(tr, tc) for tc in range(low, high + 1)]

    if not shifts:
        return None
    random.shuffle(shifts)
    blocked = _legend_cells(legend_corner, h, w, margin=1)

    for tr, tc in shifts:
        pipe2 = {(r + tr, c + tc) for r, c in pipe}
        start2 = (start_pipe[0] + tr, start_pipe[1] + tc)
        entry2 = (entry[0] + tr, entry[1] + tc)

        all_cells = set(pipe2)
        all_cells.add(entry2)
        if any(not (0 <= r < h and 0 <= c < w) for r, c in all_cells):
            continue
        if pipe2 & blocked or entry2 in blocked:
            continue
        if any(r in {0, h - 1} or c in {0, w - 1} for r, c in pipe2):
            continue

        edge_adjacent = {
            (r, c)
            for r, c in pipe2
            if min(r, c, h - 1 - r, w - 1 - c) == 1
        }
        if edge_adjacent != {start2}:
            continue
        return pipe2, start2, entry2

    return None


def _draw_legend(grid: np.ndarray, corner: str, digit_color: int, width_colors: dict[int, int], bg: int) -> None:
    r0, c0 = _legend_origin(corner, grid.shape[0], grid.shape[1])
    for idx, width in enumerate((1, 2, 3)):
        grid[r0, c0 + 3 * idx : c0 + 3 * (idx + 1)] = int(width_colors[width])
        mask = DIGIT_MASKS[width]
        block = np.full(mask.shape, int(bg), dtype=int)
        block[mask] = int(digit_color)
        grid[r0 + 1 : r0 + 1 + mask.shape[0], c0 + 3 * idx : c0 + 3 * (idx + 1)] = block


def _sample_colors() -> tuple[int, int, int, int, dict[int, int]]:
    bg, pipe, digit, entry, c1, c2, c3 = random.sample(range(10), 7)
    return bg, pipe, digit, entry, {1: c1, 2: c2, 3: c3}


def _generate_attempt() -> np.ndarray | None:
    pipe_local, start_local, entry_local = _network_from_template()
    entry_side = random.choice(["W", "N", "E", "S"])
    turns = ROTATIONS[entry_side]
    pipe_rot = _rotate_points(pipe_local, turns)
    start_rot = _rotate_point(start_local, turns)
    entry_rot = _rotate_point(entry_local, turns)

    min_h, min_w = _required_canvas(pipe_rot, entry_rot, entry_side)
    if min_h > GRID_MAX or min_w > GRID_MAX:
        return None

    h = random.randint(min_h, GRID_MAX)
    w = random.randint(min_w, GRID_MAX)
    legend_corner = random.choice(["TL", "TR", "BL", "BR"])
    placed = _place_network(pipe_rot, start_rot, entry_rot, entry_side, legend_corner, h, w)
    if placed is None:
        return None

    pipe_cells, start_pipe, entry = placed
    bg, pipe_color, digit_color, entry_color, width_colors = _sample_colors()

    grid = np.full((h, w), bg, dtype=int)
    for r, c in pipe_cells:
        grid[r, c] = int(pipe_color)
    grid[entry] = int(entry_color)
    _draw_legend(grid, legend_corner, digit_color, width_colors, bg)

    # The legend must not create any accidental alternate entry.
    extra_entries = 0
    for rr in range(h):
        for cc in range(w):
            if rr not in {0, h - 1} and cc not in {0, w - 1}:
                continue
            if int(grid[rr, cc]) in {bg, pipe_color}:
                continue
            nbrs = [(nr, nc) for nr, nc in _orth_neighbors(rr, cc, h, w) if int(grid[nr, nc]) == pipe_color]
            if len(nbrs) == 1:
                extra_entries += 1
    if extra_entries != 1:
        return None

    # Pipe color must remain the dominant non-background structure.
    non_bg_counts = {
        int(color): int(np.sum(grid == color))
        for color in np.unique(grid)
        if int(color) != bg
    }
    if max(non_bg_counts, key=non_bg_counts.get) != pipe_color:
        return None

    _ = start_pipe
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is None:
            continue
        try:
            out = main(grid)
        except Exception:
            continue
        if out.shape != grid.shape:
            continue
        return grid
    raise RuntimeError(f"failed to generate a valid pipe network after {GEN_ATTEMPTS} attempts")
