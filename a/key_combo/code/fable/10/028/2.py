from __future__ import annotations

import random
from collections import Counter

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


MARKER_COLOR = 5
GRID_LIMIT = 30


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_h(bbox: BBox) -> int:
    return int(bbox[2] - bbox[0] + 1)


def _bbox_w(bbox: BBox) -> int:
    return int(bbox[3] - bbox[1] + 1)


def _overlap_len(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0) + 1)


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _room_from_component(grid: np.ndarray, coords: np.ndarray, frame_color: int) -> dict | None:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 5 or w < 5:
        return None

    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    ring = np.zeros((h, w), dtype=bool)
    ring[0, :] = True
    ring[-1, :] = True
    ring[:, 0] = True
    ring[:, -1] = True
    if not np.array_equal(patch == frame_color, ring):
        return None

    interior = patch[1:-1, 1:-1]
    if np.any((interior != _get_bg(grid)) & (interior != MARKER_COLOR)):
        return None

    marker_count = int(np.sum(interior == MARKER_COLOR))
    if marker_count <= 0:
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "count": marker_count,
        "coords": coords,
    }


def _detect_rooms(grid: np.ndarray) -> tuple[int, int, int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    colors = sorted(set(map(int, g.flat)) - {bg, MARKER_COLOR})
    assert len(colors) == 1, "expected one frame color"
    frame_color = colors[0]

    rooms: list[dict] = []
    marker_cover = np.zeros(g.shape, dtype=int)
    for comp in _components_of_color(g, frame_color):
        room = _room_from_component(g, comp, frame_color)
        assert room is not None, "every frame-color component must be a one-cell-thick hollow rectangle"
        r0, c0, r1, c1 = room["bbox"]
        region = g[r0 + 1 : r1, c0 + 1 : c1] == MARKER_COLOR
        marker_cover[r0 + 1 : r1, c0 + 1 : c1] += region.astype(int)
        rooms.append(room)

    assert len(rooms) >= 2, "expected multiple rooms"
    assert np.all(marker_cover[g == MARKER_COLOR] == 1), "every marker must lie in exactly one room interior"
    counts = [room["count"] for room in rooms]
    assert len(counts) == len(set(counts)), "marker counts must be unique"
    return bg, frame_color, MARKER_COLOR, rooms


def _candidate_edge(grid: np.ndarray, bg: int, a: dict, b: dict, ia: int, ib: int) -> dict | None:
    ar0, ac0, ar1, ac1 = a["bbox"]
    br0, bc0, br1, bc1 = b["bbox"]

    if ac1 < bc0 or bc1 < ac0:
        if ac1 < bc0:
            left_i, right_i, left, right = ia, ib, a, b
        else:
            left_i, right_i, left, right = ib, ia, b, a
        lr0, lc0, lr1, lc1 = left["bbox"]
        rr0, rc0, rr1, rc1 = right["bbox"]
        r0 = max(lr0 + 1, rr0 + 1)
        r1 = min(lr1 - 1, rr1 - 1)
        c0 = lc1 + 1
        c1 = rc0 - 1
        if r0 <= r1 and c0 <= c1 and np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == bg):
            return {
                "kind": "h",
                "a": left_i,
                "b": right_i,
                "r0": r0,
                "r1": r1,
                "c0": c0,
                "c1": c1,
                "gap": c1 - c0 + 1,
            }

    if ar1 < br0 or br1 < ar0:
        if ar1 < br0:
            upper_i, lower_i, upper, lower = ia, ib, a, b
        else:
            upper_i, lower_i, upper, lower = ib, ia, b, a
        ur0, uc0, ur1, uc1 = upper["bbox"]
        lr0, lc0, lr1, lc1 = lower["bbox"]
        c0 = max(uc0 + 1, lc0 + 1)
        c1 = min(uc1 - 1, lc1 - 1)
        r0 = ur1 + 1
        r1 = lr0 - 1
        if c0 <= c1 and r0 <= r1 and np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == bg):
            return {
                "kind": "v",
                "a": upper_i,
                "b": lower_i,
                "r0": r0,
                "r1": r1,
                "c0": c0,
                "c1": c1,
                "gap": r1 - r0 + 1,
            }

    return None


def _all_candidate_edges(grid: np.ndarray, bg: int, rooms: list[dict]) -> list[dict]:
    edges: list[dict] = []
    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            edge = _candidate_edge(grid, bg, rooms[i], rooms[j], i, j)
            if edge is not None:
                edges.append(edge)
    return edges


def _apply_edge(out: np.ndarray, bg: int, frame_color: int, rooms: list[dict], edge: dict) -> None:
    r0, r1, c0, c1 = edge["r0"], edge["r1"], edge["c0"], edge["c1"]
    if edge["kind"] == "h":
        left = rooms[edge["a"]]["bbox"]
        right = rooms[edge["b"]]["bbox"]
        out[r0, c0 : c1 + 1] = frame_color
        out[r1, c0 : c1 + 1] = frame_color
        if r0 + 1 <= r1 - 1:
            out[r0 + 1 : r1, left[3]] = bg
            out[r0 + 1 : r1, right[1]] = bg
    else:
        upper = rooms[edge["a"]]["bbox"]
        lower = rooms[edge["b"]]["bbox"]
        out[r0 : r1 + 1, c0] = frame_color
        out[r0 : r1 + 1, c1] = frame_color
        if c0 + 1 <= c1 - 1:
            out[upper[2], c0 + 1 : c1] = bg
            out[lower[0], c0 + 1 : c1] = bg


def _frame_components_count(grid: np.ndarray, bg: int, frame_color: int) -> int:
    return sum(1 for _ in _components_of_color(grid, frame_color))


def main(grid: np.ndarray) -> np.ndarray:
    """Connect hollow rectangular rooms into a corridor chain ordered by their grey marker counts.

    Input structure:
    - The grid has a dominant background, one shared frame color, and several disconnected
      one-cell-thick hollow rectangular rooms of that frame color.
    - Each room interior contains some grey marker pixels, and the marker counts are all distinct.
    - For each count-adjacent room pair there is a clear horizontal or vertical gap with overlapping interiors.

    Rule:
    - Count the grey markers inside every room and sort the rooms from fewest to most markers.
    - Connect only consecutive rooms in that count order.
    - For a horizontal connection, paint the two boundary rows of the overlapping interior span across
      the gap, then erase the facing wall cells strictly between those rows; for a vertical connection, do the transposed operation.

    Key insight:
    - The correct corridor chain is determined by hidden marker counts inside the rooms, not by geometric proximity:
      shorter visible gaps can remain unused if their rooms are not adjacent in count order.
    """
    g = np.asarray(grid, dtype=int)
    bg, frame_color, _marker, rooms = _detect_rooms(g)
    ordered = sorted(range(len(rooms)), key=lambda i: rooms[i]["count"])
    edges_by_pair: dict[tuple[int, int], dict] = {}
    for edge in _all_candidate_edges(g, bg, rooms):
        edges_by_pair[(edge["a"], edge["b"])] = edge

    out = g.copy()
    for left_idx, right_idx in zip(ordered, ordered[1:]):
        pair = (min(left_idx, right_idx), max(left_idx, right_idx))
        assert pair in edges_by_pair, "count-adjacent rooms must have a visible gap"
        _apply_edge(out, bg, frame_color, rooms, edges_by_pair[pair])

    assert np.array_equal(out[g == MARKER_COLOR], g[g == MARKER_COLOR]), "markers must stay unchanged"
    assert _frame_components_count(out, bg, frame_color) == 1, "all rooms must become connected"
    return out


GEN_RETRIES = 500
GEN_BG = 0
GEN_FRAME_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_ROOM_OUTER = (5, 8)
GEN_COUNT_CHOICES = (1, 2, 3, 4, 5, 6)
GEN_CHAIN_ORDER_OPTIONS = (
    (0, 1, 3, 2),  # A-B-D-C, leaves A-C unused
    (0, 2, 3, 1),  # A-C-D-B, leaves A-B unused
)


def _boxes_overlap(a: BBox, b: BBox) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _interior_row_overlap(a: BBox, b: BBox) -> int:
    return _overlap_len(a[0] + 1, a[2] - 1, b[0] + 1, b[2] - 1)


def _interior_col_overlap(a: BBox, b: BBox) -> int:
    return _overlap_len(a[1] + 1, a[3] - 1, b[1] + 1, b[3] - 1)


def _try_make_cycle_layout() -> tuple[list[BBox], tuple[int, int, int, int], tuple[int, int, int, int]] | None:
    for _ in range(200):
        sizes = [
            (random.randint(*GEN_ROOM_OUTER), random.randint(*GEN_ROOM_OUTER))
            for _ in range(4)
        ]
        (h0, w0), (h1, w1), (h2, w2), (h3, w3) = sizes

        gap_ab = random.randint(1, 2)
        gap_ac = random.randint(1, 2)
        gap_bd = random.randint(2, 5)
        gap_cd = random.randint(2, 5)
        if random.random() < 0.5:
            gap_ab, gap_bd = gap_bd, gap_ab
        else:
            gap_ac, gap_cd = gap_cd, gap_ac

        a = (0, 0, h0 - 1, w0 - 1)

        r1_low = -(h1 - 3)
        r1_high = h0 - 3
        r1_candidates = []
        for r1 in range(r1_low, r1_high + 1):
            b = (r1, w0 + gap_ab, r1 + h1 - 1, w0 + gap_ab + w1 - 1)
            if _interior_row_overlap(a, b) >= 2:
                r1_candidates.append(r1)
        if not r1_candidates:
            continue
        r1 = random.choice(r1_candidates)
        b = (r1, w0 + gap_ab, r1 + h1 - 1, w0 + gap_ab + w1 - 1)

        c2_low = -(w2 - 3)
        c2_high = w0 - 3
        c2_candidates = []
        for c2 in range(c2_low, c2_high + 1):
            c = (h0 + gap_ac, c2, h0 + gap_ac + h2 - 1, c2 + w2 - 1)
            if _interior_col_overlap(a, c) >= 2:
                c2_candidates.append(c2)
        if not c2_candidates:
            continue
        c2 = random.choice(c2_candidates)
        c = (h0 + gap_ac, c2, h0 + gap_ac + h2 - 1, c2 + w2 - 1)

        d = (
            b[2] + gap_bd + 1,
            c[3] + gap_cd + 1,
            b[2] + gap_bd + h3,
            c[3] + gap_cd + w3,
        )

        rooms = [a, b, c, d]
        if any(_boxes_overlap(rooms[i], rooms[j]) for i in range(4) for j in range(i + 1, 4)):
            continue
        if _interior_row_overlap(c, d) < 2 or _interior_col_overlap(b, d) < 2:
            continue

        min_r = min(box[0] for box in rooms)
        min_c = min(box[1] for box in rooms)
        rooms = [
            (r0 - min_r + 1, c0 - min_c + 1, r1 - min_r + 1, c1 - min_c + 1)
            for (r0, c0, r1, c1) in rooms
        ]
        max_r = max(box[2] for box in rooms)
        max_c = max(box[3] for box in rooms)
        if max_r + 2 > GRID_LIMIT or max_c + 2 > GRID_LIMIT:
            continue
        return rooms, (gap_ab, gap_ac, gap_bd, gap_cd), sizes
    return None


def _draw_room_patch(grid: np.ndarray, bbox: BBox, color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _place_markers(grid: np.ndarray, bbox: BBox, count: int) -> None:
    r0, c0, r1, c1 = bbox
    cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    for r, c in random.sample(cells, count):
        grid[r, c] = MARKER_COLOR


def _transform_grid(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, copy=True)
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    k = random.randint(0, 3)
    if k:
        out = np.rot90(out, k)
    return np.array(out, dtype=int)


def _edge_signature(edge: dict) -> tuple[int, int]:
    return (min(edge["a"], edge["b"]), max(edge["a"], edge["b"]))


def _edge_opened(output: np.ndarray, input_grid: np.ndarray, bg: int, frame_color: int, rooms: list[dict], edge: dict) -> bool:
    r0, r1, c0, c1 = edge["r0"], edge["r1"], edge["c0"], edge["c1"]
    if edge["kind"] == "h":
        left = rooms[edge["a"]]["bbox"]
        right = rooms[edge["b"]]["bbox"]
        if not np.all(output[r0, c0 : c1 + 1] == frame_color):
            return False
        if not np.all(output[r1, c0 : c1 + 1] == frame_color):
            return False
        if r0 + 1 <= r1 - 1:
            if not np.all(output[r0 + 1 : r1, left[3]] == bg):
                return False
            if not np.all(output[r0 + 1 : r1, right[1]] == bg):
                return False
            if not np.all(input_grid[r0 + 1 : r1, left[3]] == frame_color):
                return False
            if not np.all(input_grid[r0 + 1 : r1, right[1]] == frame_color):
                return False
        return True

    upper = rooms[edge["a"]]["bbox"]
    lower = rooms[edge["b"]]["bbox"]
    if not np.all(output[r0 : r1 + 1, c0] == frame_color):
        return False
    if not np.all(output[r0 : r1 + 1, c1] == frame_color):
        return False
    if c0 + 1 <= c1 - 1:
        if not np.all(output[upper[2], c0 + 1 : c1] == bg):
            return False
        if not np.all(output[lower[0], c0 + 1 : c1] == bg):
            return False
        if not np.all(input_grid[upper[2], c0 + 1 : c1] == frame_color):
            return False
        if not np.all(input_grid[lower[0], c0 + 1 : c1] == frame_color):
            return False
    return True


def _validate_semantics(grid: np.ndarray) -> None:
    bg, frame_color, _marker, rooms = _detect_rooms(grid)
    edges = _all_candidate_edges(grid, bg, rooms)
    assert len(rooms) == 4, "generator should build four rooms"
    assert len(edges) == 4, "generator should expose exactly one distractor edge"

    ordered = sorted(range(len(rooms)), key=lambda i: rooms[i]["count"])
    chain_pairs = {
        (min(a, b), max(a, b))
        for a, b in zip(ordered, ordered[1:])
    }
    edge_map = {_edge_signature(edge): edge for edge in edges}
    assert chain_pairs <= set(edge_map), "all count-adjacent pairs must be visible"

    extras = [edge for edge in edges if _edge_signature(edge) not in chain_pairs]
    assert len(extras) == 1, "expected exactly one non-chain visible distractor"

    chain_edges = [edge_map[pair] for pair in chain_pairs]
    assert {edge["kind"] for edge in chain_edges} == {"h", "v"}, "chain must include both orientations"
    assert extras[0]["gap"] < max(edge["gap"] for edge in chain_edges), "distractor must be shorter than a chain link"

    out = main(grid)
    assert np.array_equal(out[grid == MARKER_COLOR], grid[grid == MARKER_COLOR]), "markers must stay fixed"
    assert _frame_components_count(out, bg, frame_color) == 1, "output must connect all rooms"
    for edge in chain_edges:
        assert _edge_opened(out, grid, bg, frame_color, rooms, edge), "chain corridor missing"
    for edge in extras:
        assert not _edge_opened(out, grid, bg, frame_color, rooms, edge), "distractor edge must remain closed"


def _gen_attempt() -> np.ndarray | None:
    layout = _try_make_cycle_layout()
    if layout is None:
        return None
    rooms, gaps, _sizes = layout
    gap_ab, gap_ac, gap_bd, gap_cd = gaps

    valid_orders = []
    if gap_ac < max(gap_ab, gap_bd, gap_cd):
        valid_orders.append((0, 1, 3, 2))
    if gap_ab < max(gap_ac, gap_bd, gap_cd):
        valid_orders.append((0, 2, 3, 1))
    if not valid_orders:
        return None
    chain_order = random.choice(valid_orders)

    frame_color = random.choice(GEN_FRAME_COLORS)
    counts = sorted(random.sample(GEN_COUNT_CHOICES, 4))
    room_counts = [0, 0, 0, 0]
    for rank, room_idx in enumerate(chain_order):
        room_counts[room_idx] = counts[rank]

    h = max(box[2] for box in rooms) + 2
    w = max(box[3] for box in rooms) + 2
    if h > GRID_LIMIT or w > GRID_LIMIT:
        return None

    grid = np.full((h, w), GEN_BG, dtype=int)
    for bbox, count in zip(rooms, room_counts):
        interior_area = (_bbox_h(bbox) - 2) * (_bbox_w(bbox) - 2)
        if count >= interior_area:
            return None
        _draw_room_patch(grid, bbox, frame_color)
        _place_markers(grid, bbox, count)

    grid = _transform_grid(grid)
    try:
        _validate_semantics(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid counting-corridor instance")
