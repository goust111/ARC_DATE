import random
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

CELL = 4
GAP = 3
STEP = CELL + GAP
GEN_RETRIES = 300
LAYOUTS = ((2, 2), (2, 3), (3, 2))


Piece = Dict[str, int]


def _components(mask: np.ndarray) -> List[List[Tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: List[List[Tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: List[Tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: Sequence[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_separator_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == GREY)]
    assert len(rows) == 1
    return rows[0]


def _find_frame_bbox(grid: np.ndarray, separator_row: int) -> Tuple[int, int, int, int]:
    lower = grid[separator_row + 1 :, :]
    candidates = []
    for comp in _components(lower == GREY):
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            continue
        if len(comp) != 2 * (h + w) - 4:
            continue
        ok = True
        comp_set = set(comp)
        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                on_border = rr in (r0, r1) or cc in (c0, c1)
                if on_border != ((rr, cc) in comp_set):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            candidates.append((r0 + separator_row + 1, c0, r1 + separator_row + 1, c1))
    assert len(candidates) == 1
    return candidates[0]


def _extract_pieces(grid: np.ndarray, separator_row: int) -> List[Piece]:
    top = grid[:separator_row, :]
    pieces: List[Piece] = []
    for idx, comp in enumerate(_components((top != BLACK) & (top != GREY))):
        r0, c0, r1, c1 = _bbox(comp)
        color = int(top[comp[0][0], comp[0][1]])
        assert all(int(top[r, c]) == color for r, c in comp)
        assert len(comp) == (r1 - r0 + 1) * (c1 - c0 + 1)
        pieces.append(
            {
                "id": idx,
                "color": color,
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
                "h": r1 - r0 + 1,
                "w": c1 - c0 + 1,
            }
        )
    assert pieces
    return pieces


def _marker_axis(piece_a: Piece, piece_b: Piece, marker: Tuple[int, int]) -> str:
    mr, mc = marker
    row_center = (piece_a["r0"] + piece_a["r1"] + piece_b["r0"] + piece_b["r1"]) / 4.0
    col_center = (piece_a["c0"] + piece_a["c1"] + piece_b["c0"] + piece_b["c1"]) / 4.0
    dr = abs(mr - row_center)
    dc = abs(mc - col_center)
    return "H" if dr <= dc else "V"


def _nearest_left(pieces: Sequence[Piece], row: int, col: int) -> Piece | None:
    candidates = [p for p in pieces if p["c1"] < col and p["r0"] <= row <= p["r1"]]
    return max(candidates, key=lambda p: p["c1"]) if candidates else None


def _nearest_right(pieces: Sequence[Piece], row: int, col: int) -> Piece | None:
    candidates = [p for p in pieces if p["c0"] > col and p["r0"] <= row <= p["r1"]]
    return min(candidates, key=lambda p: p["c0"]) if candidates else None


def _nearest_up(pieces: Sequence[Piece], row: int, col: int) -> Piece | None:
    candidates = [p for p in pieces if p["r1"] < row and p["c0"] <= col <= p["c1"]]
    return max(candidates, key=lambda p: p["r1"]) if candidates else None


def _nearest_down(pieces: Sequence[Piece], row: int, col: int) -> Piece | None:
    candidates = [p for p in pieces if p["r0"] > row and p["c0"] <= col <= p["c1"]]
    return min(candidates, key=lambda p: p["r0"]) if candidates else None


def _infer_relation(marker: Tuple[int, int], pieces: Sequence[Piece]) -> Tuple[str, int, int]:
    mr, mc = marker
    left = _nearest_left(pieces, mr, mc)
    right = _nearest_right(pieces, mr, mc)
    up = _nearest_up(pieces, mr, mc)
    down = _nearest_down(pieces, mr, mc)

    has_horizontal = left is not None and right is not None
    has_vertical = up is not None and down is not None

    if has_horizontal and not has_vertical:
        first, second = left, right
    elif has_vertical and not has_horizontal:
        first, second = up, down
    else:
        assert False, "each grey marker must connect exactly one nearest-neighbor pair"

    axis = _marker_axis(first, second, marker)
    return axis, first["id"], second["id"]


def _order_chain(node_ids: Sequence[int], edges: Iterable[Tuple[int, int]]) -> List[int]:
    node_set = set(node_ids)
    incoming = {node_id: 0 for node_id in node_ids}
    outgoing: Dict[int, int] = {}
    edge_count = 0
    for a, b in edges:
        if a not in node_set or b not in node_set:
            continue
        assert a not in outgoing
        outgoing[a] = b
        incoming[b] += 1
        edge_count += 1
    assert edge_count == len(node_ids) - 1
    starts = [node_id for node_id in node_ids if incoming[node_id] == 0]
    assert len(starts) == 1
    order = []
    seen = set()
    current = starts[0]
    while True:
        assert current not in seen
        seen.add(current)
        order.append(current)
        if current not in outgoing:
            break
        current = outgoing[current]
    assert len(order) == len(node_ids)
    return order


def _consensus(sequences: Sequence[Tuple[int, ...]]) -> Tuple[int, ...]:
    assert sequences
    first = sequences[0]
    assert all(seq == first for seq in sequences[1:])
    return first


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble the scattered monochrome rectangles and place the finished rectangle
    inside the lower grey target frame.
    """
    g = np.asarray(grid, dtype=int)
    separator_row = _find_separator_row(g)
    frame_r0, frame_c0, frame_r1, frame_c1 = _find_frame_bbox(g, separator_row)
    pieces = _extract_pieces(g, separator_row)
    piece_by_id = {piece["id"]: piece for piece in pieces}

    markers = [(r, c) for r in range(separator_row) for c in range(g.shape[1]) if int(g[r, c]) == GREY]
    assert markers

    horizontal_edges = set()
    vertical_edges = set()
    for marker in markers:
        axis, first_id, second_id = _infer_relation(marker, pieces)
        if axis == "H":
            horizontal_edges.add((first_id, second_id))
        else:
            vertical_edges.add((first_id, second_id))

    heights = {}
    widths = {}
    for piece in pieces:
        heights.setdefault(piece["h"], []).append(piece["id"])
        widths.setdefault(piece["w"], []).append(piece["id"])

    row_count = len(heights)
    col_count = len(widths)
    assert row_count * col_count == len(pieces)

    width_orders = []
    for node_ids in heights.values():
        assert len(node_ids) == col_count
        ordered_ids = _order_chain(node_ids, horizontal_edges)
        width_orders.append(tuple(piece_by_id[node_id]["w"] for node_id in ordered_ids))
    col_width_order = _consensus(width_orders)

    height_orders = []
    for node_ids in widths.values():
        assert len(node_ids) == row_count
        ordered_ids = _order_chain(node_ids, vertical_edges)
        height_orders.append(tuple(piece_by_id[node_id]["h"] for node_id in ordered_ids))
    row_height_order = _consensus(height_orders)

    assert sum(row_height_order) == frame_r1 - frame_r0 - 1
    assert sum(col_width_order) == frame_c1 - frame_c0 - 1

    row_index = {height: idx for idx, height in enumerate(row_height_order)}
    col_index = {width: idx for idx, width in enumerate(col_width_order)}

    out = g.copy()
    out[:separator_row, :] = BLACK
    out[frame_r0 + 1 : frame_r1, frame_c0 + 1 : frame_c1] = BLACK

    row_offsets = [frame_r0 + 1]
    for height in row_height_order[:-1]:
        row_offsets.append(row_offsets[-1] + height)
    col_offsets = [frame_c0 + 1]
    for width in col_width_order[:-1]:
        col_offsets.append(col_offsets[-1] + width)

    for piece in pieces:
        rr = row_index[piece["h"]]
        cc = col_index[piece["w"]]
        r0 = row_offsets[rr]
        c0 = col_offsets[cc]
        out[r0 : r0 + piece["h"], c0 : c0 + piece["w"]] = piece["color"]

    return out


def _pick_marker(
    piece_a: Piece,
    piece_b: Piece,
    visual_axis: str,
    final_axis: str,
    rng: random.Random,
) -> Tuple[int, int] | None:
    if visual_axis == "H":
        assert piece_a["c1"] < piece_b["c0"]
        rows = range(max(piece_a["r0"], piece_b["r0"]), min(piece_a["r1"], piece_b["r1"]) + 1)
        cols = range(piece_a["c1"] + 1, piece_b["c0"])
    else:
        assert piece_a["r1"] < piece_b["r0"]
        rows = range(piece_a["r1"] + 1, piece_b["r0"])
        cols = range(max(piece_a["c0"], piece_b["c0"]), min(piece_a["c1"], piece_b["c1"]) + 1)

    candidates = []
    for r in rows:
        for c in cols:
            if _marker_axis(piece_a, piece_b, (r, c)) == final_axis:
                candidates.append((r, c))
    return rng.choice(candidates) if candidates else None


def _expected_output(
    grid: np.ndarray,
    separator_row: int,
    frame_bbox: Tuple[int, int, int, int],
    row_heights: Sequence[int],
    col_widths: Sequence[int],
    piece_map: Dict[Tuple[int, int], Piece],
) -> np.ndarray:
    frame_r0, frame_c0, frame_r1, frame_c1 = frame_bbox
    out = np.asarray(grid, dtype=int).copy()
    out[:separator_row, :] = BLACK
    out[frame_r0 + 1 : frame_r1, frame_c0 + 1 : frame_c1] = BLACK

    row_start = frame_r0 + 1
    for r, height in enumerate(row_heights):
        col_start = frame_c0 + 1
        for c, width in enumerate(col_widths):
            piece = piece_map[(r, c)]
            out[row_start : row_start + height, col_start : col_start + width] = piece["color"]
            col_start += width
        row_start += height
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    rows, cols = rng.choice(LAYOUTS)
    row_heights = rng.sample([2, 3, 4], rows)
    col_widths = rng.sample([2, 3, 4], cols)
    colors = rng.sample(PALETTE, rows * cols)
    swap_axes = rng.choice([False, True])

    scatter_rows = cols if swap_axes else rows
    scatter_cols = rows if swap_axes else cols

    top_margin = rng.randint(0, 1)
    left_margin = rng.randint(0, 1)
    right_margin = rng.randint(0, 2)
    top_area_h = top_margin + (scatter_rows - 1) * STEP + CELL
    top_area_w = left_margin + (scatter_cols - 1) * STEP + CELL + right_margin

    piece_map: Dict[Tuple[int, int], Piece] = {}
    scatter_map: Dict[Tuple[int, int], Piece] = {}
    color_idx = 0
    for r in range(rows):
        for c in range(cols):
            sr, sc = (c, r) if swap_axes else (r, c)
            h = row_heights[r]
            w = col_widths[c]
            slot_r0 = top_margin + sr * STEP
            slot_c0 = left_margin + sc * STEP
            piece = {
                "id": color_idx,
                "row": r,
                "col": c,
                "color": colors[color_idx],
                "h": h,
                "w": w,
                "r0": slot_r0 + (CELL - h) // 2,
                "c0": slot_c0 + (CELL - w) // 2,
            }
            piece["r1"] = piece["r0"] + h - 1
            piece["c1"] = piece["c0"] + w - 1
            piece_map[(r, c)] = piece
            scatter_map[(sr, sc)] = piece
            color_idx += 1

    markers = []
    for sr in range(scatter_rows):
        for sc in range(scatter_cols - 1):
            left_piece = scatter_map[(sr, sc)]
            right_piece = scatter_map[(sr, sc + 1)]
            final_axis = "V" if swap_axes else "H"
            marker = _pick_marker(left_piece, right_piece, "H", final_axis, rng)
            if marker is None:
                return None
            markers.append(marker)

    for sr in range(scatter_rows - 1):
        for sc in range(scatter_cols):
            top_piece = scatter_map[(sr, sc)]
            bottom_piece = scatter_map[(sr + 1, sc)]
            final_axis = "H" if swap_axes else "V"
            marker = _pick_marker(top_piece, bottom_piece, "V", final_axis, rng)
            if marker is None:
                return None
            markers.append(marker)

    frame_h = sum(row_heights) + 2
    frame_w = sum(col_widths) + 2
    grid_w = max(top_area_w, frame_w) + rng.randint(0, 3)
    frame_c0 = rng.randint(0, grid_w - frame_w)

    separator_row = top_area_h + rng.randint(0, 1)
    frame_r0 = separator_row + 1 + rng.randint(0, 1)
    grid_h = frame_r0 + frame_h + rng.randint(0, 1)

    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    grid[separator_row, :] = GREY
    grid[frame_r0, frame_c0 : frame_c0 + frame_w] = GREY
    grid[frame_r0 + frame_h - 1, frame_c0 : frame_c0 + frame_w] = GREY
    grid[frame_r0 : frame_r0 + frame_h, frame_c0] = GREY
    grid[frame_r0 : frame_r0 + frame_h, frame_c0 + frame_w - 1] = GREY

    for piece in piece_map.values():
        grid[piece["r0"] : piece["r1"] + 1, piece["c0"] : piece["c1"] + 1] = piece["color"]

    for mr, mc in markers:
        if grid[mr, mc] != BLACK:
            return None
        grid[mr, mc] = GREY

    expected = _expected_output(
        grid,
        separator_row,
        (frame_r0, frame_c0, frame_r0 + frame_h - 1, frame_c0 + frame_w - 1),
        row_heights,
        col_widths,
        piece_map,
    )
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
