from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG_COLOR = 0
PIPE_COLOR = 5
WORK_SIZE = 30
MAX_ATTEMPTS = 300
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _background_color(grid: np.ndarray) -> int:
    corners = [int(grid[0, 0]), int(grid[0, -1]), int(grid[-1, 0]), int(grid[-1, -1])]
    return int(Counter(corners).most_common(1)[0][0])


def _runs(line: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for idx, value in enumerate(line):
        if bool(value) and start is None:
            start = idx
        elif not bool(value) and start is not None:
            runs.append((int(start), int(idx - 1)))
            start = None
    if start is not None:
        runs.append((int(start), int(len(line) - 1)))
    return runs


def _extract_horizontal_rects(mask: np.ndarray) -> list[dict]:
    h, _ = mask.shape
    rects: list[dict] = []
    active: dict[tuple[int, int], int] = {}

    for r in range(h + 1):
        current = set(_runs(mask[r])) if r < h else set()
        for interval, start_row in list(active.items()):
            if interval not in current:
                c0, c1 = interval
                height = int(r - start_row)
                width = int(c1 - c0 + 1)
                if height <= 3 and width > height:
                    rects.append({"orient": "H", "bbox": (int(start_row), int(c0), int(r - 1), int(c1)), "width": height})
                del active[interval]
        if r < h:
            for interval in current:
                active.setdefault(interval, int(r))

    return rects


def _extract_vertical_rects(mask: np.ndarray) -> list[dict]:
    _, w = mask.shape
    rects: list[dict] = []
    active: dict[tuple[int, int], int] = {}

    for c in range(w + 1):
        current = set(_runs(mask[:, c])) if c < w else set()
        for interval, start_col in list(active.items()):
            if interval not in current:
                r0, r1 = interval
                height = int(r1 - r0 + 1)
                width = int(c - start_col)
                if width <= 3 and height > width:
                    rects.append({"orient": "V", "bbox": (int(r0), int(start_col), int(r1), int(c - 1)), "width": width})
                del active[interval]
        if c < w:
            for interval in current:
                active.setdefault(interval, int(c))

    return rects


def _paint_bbox(out: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    out[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _extract_graph(pipe_mask: np.ndarray) -> tuple[dict[int, dict], dict[int, dict]]:
    hrects = _extract_horizontal_rects(pipe_mask)
    vrects = _extract_vertical_rects(pipe_mask)

    coverage = np.zeros_like(pipe_mask, dtype=bool)
    for rect in hrects + vrects:
        r0, c0, r1, c1 = rect["bbox"]
        coverage[r0 : r1 + 1, c0 : c1 + 1] = True
    assert np.array_equal(coverage, pipe_mask), "pipe network must decompose into orthogonal strips"

    nodes: dict[int, dict] = {}
    h_nodes: list[list[int]] = [[] for _ in hrects]
    v_nodes: list[list[int]] = [[] for _ in vrects]
    next_node_id = 0

    for hi, hrect in enumerate(hrects):
        hr0, hc0, hr1, hc1 = hrect["bbox"]
        for vi, vrect in enumerate(vrects):
            vr0, vc0, vr1, vc1 = vrect["bbox"]
            r0 = max(hr0, vr0)
            c0 = max(hc0, vc0)
            r1 = min(hr1, vr1)
            c1 = min(hc1, vc1)
            if r0 <= r1 and c0 <= c1:
                nodes[next_node_id] = {"type": "junction", "bbox": (int(r0), int(c0), int(r1), int(c1)), "edges": []}
                h_nodes[hi].append(next_node_id)
                v_nodes[vi].append(next_node_id)
                next_node_id += 1

    edges: dict[int, dict] = {}
    next_edge_id = 0

    def make_terminal() -> int:
        nonlocal next_node_id
        node_id = next_node_id
        nodes[node_id] = {"type": "terminal", "bbox": None, "edges": []}
        next_node_id += 1
        return node_id

    def add_edge(a: int, b: int, orient: str, width: int, bbox: tuple[int, int, int, int]) -> None:
        nonlocal next_edge_id
        edge_id = next_edge_id
        edges[edge_id] = {"a": int(a), "b": int(b), "orient": orient, "width": int(width), "bbox": bbox}
        nodes[a]["edges"].append(edge_id)
        nodes[b]["edges"].append(edge_id)
        next_edge_id += 1

    for idx, rect in enumerate(hrects):
        r0, c0, r1, c1 = rect["bbox"]
        ordered = sorted(h_nodes[idx], key=lambda node_id: nodes[node_id]["bbox"][1])
        if not ordered:
            left = make_terminal()
            right = make_terminal()
            add_edge(left, right, "H", rect["width"], (r0, c0, r1, c1))
            continue

        first = ordered[0]
        first_c0 = int(nodes[first]["bbox"][1])
        if c0 <= first_c0 - 1:
            left = make_terminal()
            add_edge(left, first, "H", rect["width"], (r0, c0, r1, first_c0 - 1))

        for left_node, right_node in zip(ordered, ordered[1:]):
            seg_c0 = int(nodes[left_node]["bbox"][3] + 1)
            seg_c1 = int(nodes[right_node]["bbox"][1] - 1)
            if seg_c0 <= seg_c1:
                add_edge(left_node, right_node, "H", rect["width"], (r0, seg_c0, r1, seg_c1))

        last = ordered[-1]
        last_c1 = int(nodes[last]["bbox"][3])
        if last_c1 + 1 <= c1:
            right = make_terminal()
            add_edge(last, right, "H", rect["width"], (r0, last_c1 + 1, r1, c1))

    for idx, rect in enumerate(vrects):
        r0, c0, r1, c1 = rect["bbox"]
        ordered = sorted(v_nodes[idx], key=lambda node_id: nodes[node_id]["bbox"][0])
        if not ordered:
            top = make_terminal()
            bottom = make_terminal()
            add_edge(top, bottom, "V", rect["width"], (r0, c0, r1, c1))
            continue

        first = ordered[0]
        first_r0 = int(nodes[first]["bbox"][0])
        if r0 <= first_r0 - 1:
            top = make_terminal()
            add_edge(top, first, "V", rect["width"], (r0, c0, first_r0 - 1, c1))

        for top_node, bottom_node in zip(ordered, ordered[1:]):
            seg_r0 = int(nodes[top_node]["bbox"][2] + 1)
            seg_r1 = int(nodes[bottom_node]["bbox"][0] - 1)
            if seg_r0 <= seg_r1:
                add_edge(top_node, bottom_node, "V", rect["width"], (seg_r0, c0, seg_r1, c1))

        last = ordered[-1]
        last_r1 = int(nodes[last]["bbox"][2])
        if last_r1 + 1 <= r1:
            bottom = make_terminal()
            add_edge(last, bottom, "V", rect["width"], (last_r1 + 1, c0, r1, c1))

    assert edges, "pipe network must contain at least one edge"
    return nodes, edges


def _other_end(edge: dict, node_id: int) -> int:
    return int(edge["b"] if int(edge["a"]) == int(node_id) else edge["a"])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate the source color through a grey pipe network while each junction only forwards dye into the narrowest outgoing pipe.

    Input structure:
    - The background is black.
    - Grey pixels form an orthogonal pipe network made from horizontal and vertical rectangular channels of width 1 to 3.
    - A single non-grey colored pixel sits at one pipe mouth and marks the dye source.

    Rule:
    - Starting from the source, recolor every reached grey pipe pixel to the source color.
    - At a junction, recolor the junction block itself, then ignore the incoming arm and compare the widths of all remaining connected arms.
    - Continue only through the narrowest remaining arm; if several outgoing arms share that minimum width, continue through all of them in parallel.
    - Pipe arms that are never entered stay grey, and propagation stops at dead ends or the grid boundary.

    Key insight:
    - The grey region must be interpreted as a graph of straight pipe strips and intersection blocks, because the branching rule depends on pipe width rather than ordinary flood-fill connectivity.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    counts = Counter(map(int, g.flat))
    non_bg = [color for color in counts if color != bg]
    assert len(non_bg) >= 2, "expected a pipe color and a source color"

    source_colors = [color for color in non_bg if counts[color] == 1]
    assert source_colors, "expected a singleton source pixel"
    source_color = int(source_colors[0])
    source_positions = np.argwhere(g == source_color)
    assert source_positions.shape == (1, 2), "expected exactly one source pixel"
    source_r, source_c = map(int, source_positions[0])

    pipe_candidates = [color for color in non_bg if color != source_color]
    pipe_color = int(max(pipe_candidates, key=lambda color: counts[color]))
    pipe_mask = g == pipe_color

    nodes, edges = _extract_graph(pipe_mask)

    pipe_neighbors = []
    for dr, dc in NEIGHBORS4:
        nr = int(source_r + dr)
        nc = int(source_c + dc)
        if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and bool(pipe_mask[nr, nc]):
            pipe_neighbors.append((nr, nc))
    assert pipe_neighbors, "source must touch the pipe network"
    nr, nc = pipe_neighbors[0]

    start_edge_id = None
    for edge_id, edge in edges.items():
        r0, c0, r1, c1 = edge["bbox"]
        if r0 <= nr <= r1 and c0 <= nc <= c1:
            start_edge_id = int(edge_id)
            break
    assert start_edge_id is not None, "failed to locate the source edge"

    start_edge = edges[start_edge_id]
    if start_edge["orient"] == "H":
        start_node = int(start_edge["a"] if source_c < nc else start_edge["b"])
    else:
        start_node = int(start_edge["a"] if source_r < nr else start_edge["b"])

    out = g.copy()
    painted_edges: set[int] = set()
    painted_nodes: set[int] = set()
    seen_states: set[tuple[int, int]] = set()

    def follow_edge(edge_id: int, from_node: int) -> None:
        edge = edges[edge_id]
        if edge_id not in painted_edges:
            _paint_bbox(out, edge["bbox"], source_color)
            painted_edges.add(edge_id)
        to_node = _other_end(edge, from_node)
        if nodes[to_node]["type"] == "junction":
            process_node(to_node, edge_id)

    def process_node(node_id: int, incoming_edge_id: int) -> None:
        state = (int(node_id), int(incoming_edge_id))
        if state in seen_states:
            return
        seen_states.add(state)

        if node_id not in painted_nodes:
            _paint_bbox(out, nodes[node_id]["bbox"], source_color)
            painted_nodes.add(node_id)

        outgoing = [edge_id for edge_id in nodes[node_id]["edges"] if edge_id != incoming_edge_id]
        if not outgoing:
            return

        min_width = min(int(edges[edge_id]["width"]) for edge_id in outgoing)
        for edge_id in outgoing:
            if int(edges[edge_id]["width"]) == min_width:
                follow_edge(edge_id, node_id)

    follow_edge(start_edge_id, start_node)
    return out


def _bbox_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = bbox
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _can_place(pipe_mask: np.ndarray, bbox: tuple[int, int, int, int], allowed_overlap: tuple[int, int, int, int] | None) -> bool:
    new_mask = _bbox_mask(pipe_mask.shape, bbox)
    overlap = new_mask & pipe_mask
    allowed_mask = np.zeros_like(pipe_mask, dtype=bool)
    if allowed_overlap is not None:
        allowed_mask = _bbox_mask(pipe_mask.shape, allowed_overlap)

    if not np.array_equal(overlap, allowed_mask):
        return False

    new_only = new_mask & ~allowed_mask
    existing_only = pipe_mask & ~allowed_mask
    return not bool((_dilate4(new_only) & existing_only).any())


def _add_strip(pipe_mask: np.ndarray, bbox: tuple[int, int, int, int], allowed_overlap: tuple[int, int, int, int] | None) -> bool:
    if not _can_place(pipe_mask, bbox, allowed_overlap):
        return False
    r0, c0, r1, c1 = bbox
    pipe_mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return True


def _relative_width(base_width: int, force_not_all_wider: bool = False) -> tuple[int, int]:
    options: list[tuple[int, int]] = [(base_width, 0)]
    if base_width > 1:
        options.append((base_width - 1, -1))
    if base_width < 3:
        options.append((base_width + 1, 1))
    if force_not_all_wider:
        options = [item for item in options if item[1] <= 0]
    return random.choice(options)


def _branch_slots(start: int, end: int, count: int) -> list[int] | None:
    slots = list(range(start + 3, end - 2, 4))
    if len(slots) < count:
        return None
    return sorted(random.sample(slots, count))


def _crop_with_margin(grid: np.ndarray) -> np.ndarray:
    coords = np.argwhere(grid != BG_COLOR)
    r0, c0 = map(int, coords.min(axis=0))
    r1, c1 = map(int, coords.max(axis=0))
    top = random.randint(0, min(2, r0))
    left = random.randint(0, min(2, c0))
    bottom = random.randint(0, min(2, grid.shape[0] - 1 - r1))
    right = random.randint(0, min(2, grid.shape[1] - 1 - c1))
    return grid[r0 - top : r1 + bottom + 1, c0 - left : c1 + right + 1].copy()


def _gen_attempt() -> np.ndarray | None:
    pipe_mask = np.zeros((WORK_SIZE, WORK_SIZE), dtype=bool)
    root_orient = random.choice(["H", "V"])
    root_width = random.choices([1, 2, 3], weights=[1, 2, 2], k=1)[0]
    branch_count = random.randint(1, 3)
    root_length = random.randint(max(11, 7 + 4 * branch_count), 21)

    if root_orient == "H":
        row_start = random.randint(8, 18 - root_width)
        col_start = random.randint(2, WORK_SIZE - root_length - 2)
        root_bbox = (row_start, col_start, row_start + root_width - 1, col_start + root_length - 1)
    else:
        row_start = random.randint(2, WORK_SIZE - root_length - 2)
        col_start = random.randint(8, 18 - root_width)
        root_bbox = (row_start, col_start, row_start + root_length - 1, col_start + root_width - 1)

    if not _add_strip(pipe_mask, root_bbox, None):
        return None

    slots = _branch_slots(root_bbox[1] if root_orient == "H" else root_bbox[0], root_bbox[3] if root_orient == "H" else root_bbox[2], branch_count)
    if slots is None:
        return None

    branch_specs: list[dict] = []
    for _ in range(branch_count):
        width, relation = _relative_width(root_width)
        branch_specs.append({"width": width, "relation": relation})
    if all(spec["relation"] == 1 for spec in branch_specs):
        idx = random.randrange(branch_count)
        width, relation = _relative_width(root_width, force_not_all_wider=True)
        branch_specs[idx] = {"width": width, "relation": relation}

    branches_placed = 0
    for slot, spec in zip(slots, branch_specs):
        branch_width = int(spec["width"])
        if branch_width == 1:
            band_start = slot
        elif branch_width == 2:
            band_start = slot - random.choice([0, 1])
        else:
            band_start = slot - 1

        if root_orient == "H":
            c0 = max(root_bbox[1] + 1, min(band_start, root_bbox[3] - branch_width))
            c1 = c0 + branch_width - 1
            room_up = root_bbox[0]
            room_down = WORK_SIZE - 1 - root_bbox[2]
            branch_type = random.choice(["T", "X"])
            if branch_type == "T":
                side = random.choice(["up", "down"])
                if side == "up":
                    if room_up < 3:
                        continue
                    up_len = room_up if random.random() < 0.3 else random.randint(3, min(room_up, 9))
                    bbox = (root_bbox[0] - up_len, c0, root_bbox[2], c1)
                else:
                    if room_down < 3:
                        continue
                    down_len = room_down if random.random() < 0.3 else random.randint(3, min(room_down, 9))
                    bbox = (root_bbox[0], c0, root_bbox[2] + down_len, c1)
            else:
                if room_up < 3 or room_down < 3:
                    continue
                up_len = room_up if random.random() < 0.2 else random.randint(3, min(room_up, 8))
                down_len = room_down if random.random() < 0.2 else random.randint(3, min(room_down, 8))
                bbox = (root_bbox[0] - up_len, c0, root_bbox[2] + down_len, c1)
            overlap = (root_bbox[0], c0, root_bbox[2], c1)
        else:
            r0 = max(root_bbox[0] + 1, min(band_start, root_bbox[2] - branch_width))
            r1 = r0 + branch_width - 1
            room_left = root_bbox[1]
            room_right = WORK_SIZE - 1 - root_bbox[3]
            branch_type = random.choice(["T", "X"])
            if branch_type == "T":
                side = random.choice(["left", "right"])
                if side == "left":
                    if room_left < 3:
                        continue
                    left_len = room_left if random.random() < 0.3 else random.randint(3, min(room_left, 9))
                    bbox = (r0, root_bbox[1] - left_len, r1, root_bbox[3])
                else:
                    if room_right < 3:
                        continue
                    right_len = room_right if random.random() < 0.3 else random.randint(3, min(room_right, 9))
                    bbox = (r0, root_bbox[1], r1, root_bbox[3] + right_len)
            else:
                if room_left < 3 or room_right < 3:
                    continue
                left_len = room_left if random.random() < 0.2 else random.randint(3, min(room_left, 8))
                right_len = room_right if random.random() < 0.2 else random.randint(3, min(room_right, 8))
                bbox = (r0, root_bbox[1] - left_len, r1, root_bbox[3] + right_len)
            overlap = (r0, root_bbox[1], r1, root_bbox[3])

        if _add_strip(pipe_mask, bbox, overlap):
            branches_placed += 1

    if branches_placed == 0:
        return None

    source_color = random.choice([1, 2, 3, 4, 6, 7, 8, 9])
    source_at_start = random.choice([True, False])
    if root_orient == "H":
        src_row = random.randint(root_bbox[0], root_bbox[2])
        src_col = root_bbox[1] - 1 if source_at_start else root_bbox[3] + 1
        if not (0 <= src_col < WORK_SIZE):
            return None
    else:
        src_row = root_bbox[0] - 1 if source_at_start else root_bbox[2] + 1
        src_col = random.randint(root_bbox[1], root_bbox[3])
        if not (0 <= src_row < WORK_SIZE):
            return None

    grid = np.full((WORK_SIZE, WORK_SIZE), BG_COLOR, dtype=int)
    grid[pipe_mask] = PIPE_COLOR
    grid[src_row, src_col] = int(source_color)
    cropped = _crop_with_margin(grid)

    try:
        out = main(cropped)
        nodes, _ = _extract_graph(cropped == PIPE_COLOR)
    except Exception:
        return None

    pipe_pixels = cropped == PIPE_COLOR
    if not bool(pipe_pixels.any()):
        return None
    if not bool(((out == source_color) & pipe_pixels).any()):
        return None
    if not bool(((cropped == PIPE_COLOR) & (out == PIPE_COLOR)).any()):
        return None
    if sum(1 for node in nodes.values() if node["type"] == "junction") < 1:
        return None

    return cropped


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pipe-network input")
