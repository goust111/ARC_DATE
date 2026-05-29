# Combo variant of 88bcf3b4 (index 25)
# combo_spec: D4/0+D6/0+D17/2
# Name: 直线源矩形尖角绕行
#
# Design analysis for this combo variant
# - D4/0: the source segment is a single straight line, not an irregular rope.
# - D6/0: the obstacle is a strict rectangle. This implementation uses squares, which are still rectangles.
# - D17/2: the output has no alongside phase. It is a sharp V made of two diagonal runs only.
# - Local picture (terminal at the bar end, forward points into the grid, side is left/right):
#     input source   = [(0,s), (0,2s), (0,3s), ...]
#     obstacle       = {(r, cs) | 1<=r<=k, k<=(-cs*s)<=2k-1}
#     output source  = [(0,s), (1,0), ..., (k,(1-k)s), (k+1,(2-k)s), ...]
#   The square sits on the opposite side of the source line, and its near corner fixes the V apex row.

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, paint_coords


def _components(g: np.ndarray, col: int, conn8: bool) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(g == col, connectivity=8 if conn8 else 4)]


def _is_straight_line(comp: list[tuple[int, int]]) -> bool:
    rs = sorted({r for r, _ in comp})
    cs = sorted({c for _, c in comp})
    if len(rs) == 1:
        return cs == list(range(cs[0], cs[-1] + 1))
    if len(cs) == 1:
        return rs == list(range(rs[0], rs[-1] + 1))
    return False


def _bar_endpoints(comp: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]]:
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    if len(set(cs)) == 1:
        c = cs[0]
        return (min(rs), c), (max(rs), c)
    r = rs[0]
    return (r, min(cs)), (r, max(cs))


def _touches_border(p: tuple[int, int], h: int, w: int) -> bool:
    r, c = p
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


EDGE_DATA = {
    "top": {"fwd": (1, 0), "side": (0, 1)},
    "bottom": {"fwd": (-1, 0), "side": (0, -1)},
    "left": {"fwd": (0, 1), "side": (-1, 0)},
    "right": {"fwd": (0, -1), "side": (1, 0)},
}


def _edge_for_bar(comp: list[tuple[int, int]], h: int, w: int) -> tuple[str, tuple[int, int]]:
    e0, e1 = _bar_endpoints(comp)
    touch0 = _touches_border(e0, h, w)
    touch1 = _touches_border(e1, h, w)
    assert touch0 != touch1, "bar must touch exactly one border endpoint"
    border_ep = e0 if touch0 else e1
    terminal = e1 if touch0 else e0
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    if len(set(cs)) == 1:
        edge = "top" if border_ep[0] == 0 else "bottom"
    else:
        edge = "left" if border_ep[1] == 0 else "right"
    return edge, terminal


def _to_local(
    p: tuple[int, int], terminal: tuple[int, int], fwd: tuple[int, int], side_vec: tuple[int, int]
) -> tuple[int, int]:
    dr = p[0] - terminal[0]
    dc = p[1] - terminal[1]
    return dr * fwd[0] + dc * fwd[1], dr * side_vec[0] + dc * side_vec[1]


def _from_local(
    p: tuple[int, int], terminal: tuple[int, int], fwd: tuple[int, int], side_vec: tuple[int, int]
) -> tuple[int, int]:
    lf, ls = p
    return (
        terminal[0] + lf * fwd[0] + ls * side_vec[0],
        terminal[1] + lf * fwd[1] + ls * side_vec[1],
    )


def _line_path_from_start(comp: list[tuple[int, int]], start: tuple[int, int]) -> list[tuple[int, int]] | None:
    if not _is_straight_line(comp):
        return None
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    if len(set(rs)) == 1:
        r = rs[0]
        path = [(r, c) for c in sorted(cs)]
    else:
        c = cs[0]
        path = [(r, c) for r in sorted(rs)]
    if path[0] == start:
        return path
    if path[-1] == start:
        return path[::-1]
    return None


def _is_local_source_path(path: list[tuple[int, int]]) -> bool:
    if not path:
        return False
    fwd_vals = {lf for lf, _ in path}
    if fwd_vals != {0}:
        return False
    side_vals = [ls for _, ls in path]
    sign = 1 if side_vals[0] > 0 else -1
    if any(ls == 0 or ls * sign <= 0 for ls in side_vals):
        return False
    signed = [ls * sign for ls in side_vals]
    return signed == list(range(1, len(path) + 1))


def _square_info(
    comp: list[tuple[int, int]], terminal: tuple[int, int], fwd: tuple[int, int], side_vec: tuple[int, int], side: int
) -> int | None:
    local = [_to_local(p, terminal, fwd, side_vec) for p in comp]
    rows = sorted({lf for lf, _ in local})
    cols = sorted({-ls * side for _, ls in local})
    if not rows or not cols:
        return None
    if rows[0] != 1:
        return None
    if any(v <= 0 for v in cols):
        return None
    if rows != list(range(1, rows[-1] + 1)):
        return None
    k = rows[-1]
    if cols != list(range(k, 2 * k)):
        return None
    expected = {(r, -side * c) for r in rows for c in cols}
    if set(local) != expected:
        return None
    return k


def _build_output_local(seg_len: int, side: int, square_size: int) -> list[tuple[int, int]]:
    assert 1 <= square_size <= seg_len - 2
    coords = [(0, side)]
    for _ in range(square_size):
        pr, pc = coords[-1]
        coords.append((pr + 1, pc - side))
    exit_len = seg_len - 1 - square_size
    for _ in range(exit_len):
        pr, pc = coords[-1]
        coords.append((pr + 1, pc + side))
    return coords


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each straight bar-attached source line by a same-length sharp V around the nearest rectangle.

    Input structure:
    - A border-perpendicular bar reaches into the grid.
    - A straight source line of another color is attached to the bar terminal and extends sideways.
    - A filled rectangle of a third color sits in front of that line on the same side.

    Rule:
    - Keep bars and rectangles fixed.
    - Erase the straight source line.
    - Redraw the source as a two-arm diagonal V. The first arm approaches the rectangle corner; the second
      arm immediately exits on the opposite diagonal, with no middle alongside run.
    - The rectangle size determines the apex location, and the source line length is preserved.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})
    out = g.copy()

    def bar_candidates():
        for col in colors:
            for comp in _components(g, col, conn8=False):
                if len(comp) < 2 or not _is_straight_line(comp):
                    continue
                e0, e1 = _bar_endpoints(comp)
                touch0 = _touches_border(e0, h, w)
                touch1 = _touches_border(e1, h, w)
                if touch0 == touch1:
                    continue
                rs = [r for r, _ in comp]
                cs = [c for _, c in comp]
                vertical = len(set(cs)) == 1
                border_ep = e0 if touch0 else e1
                if vertical and border_ep[0] not in (0, h - 1):
                    continue
                if (not vertical) and border_ep[1] not in (0, w - 1):
                    continue
                terminal_ep = e1 if touch0 else e0
                tr, tc = terminal_ep
                if not any(
                    0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) != bg and int(g[nr, nc]) != col
                    for nr, nc in neighbors4(tr, tc)
                ):
                    continue
                yield col, comp, terminal_ep

    moves: list[tuple[list[tuple[int, int]], int, list[tuple[int, int]]]] = []
    for bar_col, bar_comp, terminal in sorted(list(bar_candidates()), key=lambda x: (-len(x[1]), x[0], x[2])):
        edge, terminal = _edge_for_bar(bar_comp, h, w)
        fwd = EDGE_DATA[edge]["fwd"]
        side_vec = EDGE_DATA[edge]["side"]

        best_seg = None
        for nr, nc in neighbors4(terminal[0], terminal[1]):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            cand_col = int(g[nr, nc])
            if cand_col in (bg, bar_col):
                continue
            for comp in _components(g, cand_col, conn8=False):
                if (nr, nc) not in comp:
                    continue
                path = _line_path_from_start(comp, (nr, nc))
                if path is None or len(path) < 3:
                    break
                local_path = [_to_local(p, terminal, fwd, side_vec) for p in path]
                if not _is_local_source_path(local_path):
                    break
                side = 1 if local_path[0][1] > 0 else -1
                key = (len(path), cand_col, side, nr, nc)
                if best_seg is None or key > best_seg[0]:
                    best_seg = (key, cand_col, comp, path, side)
                break
        assert best_seg is not None, f"no valid straight source for bar at {terminal}"
        (_, seg_col, seg_comp, seg_path, side) = best_seg
        seg_len = len(seg_path)

        best_obs = None
        square_size = None
        for cand_col in colors:
            if cand_col in (bar_col, seg_col):
                continue
            for comp in _components(g, cand_col, conn8=False):
                k = _square_info(comp, terminal, fwd, side_vec, side)
                if k is None:
                    continue
                d = min(abs(r - terminal[0]) + abs(c - terminal[1]) for r, c in comp)
                mr = min(r for r, _ in comp)
                mc = min(c for _, c in comp)
                key = (d, mr, mc)
                if best_obs is None or key < best_obs:
                    best_obs = key
                    square_size = k
        assert square_size is not None, f"no valid rectangle for bar at {terminal}"
        assert 1 <= square_size <= seg_len - 2, f"invalid square/source relation: {(square_size, seg_len)}"

        local_route = _build_output_local(seg_len, side, square_size)
        new_pts = [_from_local(p, terminal, fwd, side_vec) for p in local_route]
        assert len(new_pts) == len(set(new_pts)) == seg_len
        assert all(0 <= r < h and 0 <= c < w for r, c in new_pts)

        moves.append((seg_comp, seg_col, new_pts))

    for seg_comp, _seg_col, _ in moves:
        for r, c in seg_comp:
            out[r, c] = bg
    for _, seg_col, new_pts in moves:
        for r, c in new_pts:
            if out[r, c] == bg:
                out[r, c] = seg_col
    return out


GRID_SIZE_MIN, GRID_SIZE_MAX = 9, 26
BAR_LEN_MIN, BAR_LEN_MAX = 3, 7
BAR_CLEARANCE = 4
SEG_LEN_MIN, SEG_LEN_MAX = 5, 9
SQUARE_MIN, SQUARE_MAX = 2, 4
MAX_RETRIES = 500
PLACEMENT_PAD = 1


def _terminal_for(edge: str, n: int, bar_len: int, pos: int) -> tuple[int, int]:
    if edge == "top":
        return (bar_len - 1, pos)
    if edge == "bottom":
        return (n - bar_len, pos)
    if edge == "left":
        return (pos, bar_len - 1)
    return (pos, n - bar_len)


def _transform(locals_: list[tuple[int, int]], edge: str, n: int, bar_len: int, pos: int) -> list[tuple[int, int]]:
    tr, tc = _terminal_for(edge, n, bar_len, pos)
    fr, fc = EDGE_DATA[edge]["fwd"]
    sr, sc = EDGE_DATA[edge]["side"]
    return [(tr + lf * fr + ls * sr, tc + lf * fc + ls * sc) for lf, ls in locals_]


def _make_local_triple(n: int) -> dict | None:
    max_bar = min(BAR_LEN_MAX, n - BAR_CLEARANCE)
    if max_bar < BAR_LEN_MIN:
        return None
    bar_len = random.randint(BAR_LEN_MIN, max_bar)

    max_seg = min(SEG_LEN_MAX, n - bar_len + 1)
    if max_seg < SEG_LEN_MIN:
        return None
    seg_len = random.randint(SEG_LEN_MIN, max_seg)

    max_square = min(SQUARE_MAX, seg_len - 2)
    if max_square < SQUARE_MIN:
        return None
    square_size = random.randint(SQUARE_MIN, max_square)
    side = random.choice((-1, 1))

    bar = [(-k, 0) for k in range(bar_len)]
    input_seg = [(0, side * k) for k in range(1, seg_len + 1)]
    output_seg = _build_output_local(seg_len, side, square_size)
    obstacle = [
        (r, -side * c)
        for r in range(1, square_size + 1)
        for c in range(square_size, 2 * square_size)
    ]

    if set(bar) & set(input_seg):
        return None
    if set(bar) & set(output_seg):
        return None
    if set(bar) & set(obstacle):
        return None
    if set(input_seg) & set(obstacle):
        return None
    if set(output_seg) & set(obstacle):
        return None

    union = bar + input_seg + output_seg + obstacle
    rs = [r for r, _ in union]
    cs = [c for _, c in union]
    return {
        "bar_len": bar_len,
        "bar": bar,
        "input_seg": input_seg,
        "output_seg": output_seg,
        "obstacle": obstacle,
        "bbox": (min(rs), min(cs), max(rs), max(cs)),
    }


def _expanded_bbox(coords: list[tuple[int, int]], n: int) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return (
        max(0, min(rs) - PLACEMENT_PAD),
        max(0, min(cs) - PLACEMENT_PAD),
        min(n - 1, max(rs) + PLACEMENT_PAD),
        min(n - 1, max(cs) + PLACEMENT_PAD),
    )


def _place_triple(triple: dict, edge: str, n: int, reserved: np.ndarray) -> dict | None:
    positions = list(range(n))
    random.shuffle(positions)
    union = triple["bar"] + triple["input_seg"] + triple["output_seg"] + triple["obstacle"]
    bar_len = triple["bar_len"]
    for pos in positions:
        union_g = _transform(union, edge, n, bar_len, pos)
        if any(r < 0 or c < 0 or r >= n or c >= n for r, c in union_g):
            continue
        r0, c0, r1, c1 = _expanded_bbox(union_g, n)
        if reserved[r0 : r1 + 1, c0 : c1 + 1].any():
            continue
        return {
            "bar": _transform(triple["bar"], edge, n, bar_len, pos),
            "input_seg": _transform(triple["input_seg"], edge, n, bar_len, pos),
            "output_seg": _transform(triple["output_seg"], edge, n, bar_len, pos),
            "obstacle": _transform(triple["obstacle"], edge, n, bar_len, pos),
            "terminal": _transform([(0, 0)], edge, n, bar_len, pos)[0],
            "reserve": (r0, c0, r1, c1),
        }
    return None


def _generate_pair() -> tuple[np.ndarray, np.ndarray] | None:
    n = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    if n < 12:
        max_triples = 1
    elif n < 17:
        max_triples = 2
    else:
        max_triples = 3
    triple_count = random.randint(1, max_triples)

    bg, bar_color, seg_color, obstacle_color = random.sample(range(10), 4)
    inp = np.full((n, n), bg, dtype=int)
    out = np.full((n, n), bg, dtype=int)
    reserved = np.zeros((n, n), dtype=bool)
    placed_triples = []

    for edge in random.sample(list(EDGE_DATA), triple_count):
        triple = _make_local_triple(n)
        if triple is None:
            return None
        placed = _place_triple(triple, edge, n, reserved)
        if placed is None:
            return None

        paint_coords(inp, placed["bar"], bar_color)
        paint_coords(inp, placed["input_seg"], seg_color)
        paint_coords(inp, placed["obstacle"], obstacle_color)

        paint_coords(out, placed["bar"], bar_color)
        paint_coords(out, placed["output_seg"], seg_color)
        paint_coords(out, placed["obstacle"], obstacle_color)

        r0, c0, r1, c1 = placed["reserve"]
        reserved[r0 : r1 + 1, c0 : c1 + 1] = True
        placed_triples.append(placed)

    if len(placed_triples) > 1:
        for i, placed in enumerate(placed_triples):
            tr, tc = placed["terminal"]
            own = min(abs(r - tr) + abs(c - tc) for r, c in placed["obstacle"])
            other = min(
                abs(r - tr) + abs(c - tc)
                for j, other_placed in enumerate(placed_triples)
                if j != i
                for r, c in other_placed["obstacle"]
            )
            if own >= other:
                return None

    try:
        if not np.array_equal(main(inp), out):
            return None
    except Exception:
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        pair = _generate_pair()
        if pair is not None:
            return pair[0]
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
