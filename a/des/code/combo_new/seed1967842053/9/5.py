import random
from collections import deque

import numpy as np


BG = 0
MARK = 9

DIRS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}
OPP = {"U": "D", "D": "U", "L": "R", "R": "L"}


def _edge_cell(side, offset, h, w):
    if side == "U":
        return 0, offset
    if side == "D":
        return h - 1, offset
    if side == "L":
        return offset, 0
    return offset, w - 1


def _outside_cell(side, offset, h, w):
    r, c = _edge_cell(side, offset, h, w)
    dr, dc = DIRS[side]
    return r + dr, c + dc


def _side_length(side, h, w):
    return w if side in ("U", "D") else h


def _scan_colors(grid):
    seen = set()
    colors = []
    for v in grid.ravel():
        v = int(v)
        if v not in (BG, MARK) and v not in seen:
            seen.add(v)
            colors.append(v)
    return colors


def _piece_records(grid):
    rows, cols = grid.shape
    pieces = []
    for color in _scan_colors(grid):
        rr, cc = np.where(grid == color)
        top, bottom = int(rr.min()), int(rr.max())
        left, right = int(cc.min()), int(cc.max())
        h, w = bottom - top + 1, right - left + 1
        protrusions = []
        indentations = []
        for side in ("U", "D", "L", "R"):
            for off in range(_side_length(side, h, w)):
                er, ec = _edge_cell(side, off, h, w)
                orr, occ = _outside_cell(side, off, h, w)
                gr, gc = top + er, left + ec
                mr, mc = top + orr, left + occ
                if not (0 <= mr < rows and 0 <= mc < cols):
                    continue
                if grid[mr, mc] != MARK:
                    continue
                if grid[gr, gc] == color:
                    protrusions.append(
                        {
                            "piece": len(pieces),
                            "side": side,
                            "offset": off,
                            "local": (orr, occ),
                        }
                    )
                elif grid[gr, gc] == BG:
                    indentations.append(
                        {
                            "piece": len(pieces),
                            "side": side,
                            "offset": off,
                            "local": (er, ec),
                        }
                    )
        pieces.append(
            {
                "color": color,
                "top": top,
                "left": left,
                "h": h,
                "w": w,
                "protrusions": protrusions,
                "indentations": indentations,
            }
        )
    return pieces


def main(grid: np.ndarray) -> np.ndarray:
    """Assemble scattered edge-lock blocks into their completed rectangle.

    Input structure:
    The grid has black background, several single-color rectangular blocks,
    and maroon edge markers.  A protruding lock is a maroon pixel just outside
    a colored edge cell.  A recessed lock is a black edge-cell gap with a
    maroon pixel immediately outside that gap.

    Rule:
    For every protruding marker, find the recessed marker facing the opposite
    direction with the same relative offset along its edge.  Translate blocks
    so paired locks coincide: the protruding cell lands in the recessed gap.
    Draw the completed rectangle using the block colors and remove all maroon
    markers.

    Key insight:
    The maroon pixels have two different roles; separating outside tabs from
    gap markers exposes a direction-and-offset matching graph that determines
    the block translations.
    """
    grid = np.asarray(grid)
    pieces = _piece_records(grid)
    if not pieces:
        return np.zeros((0, 0), dtype=grid.dtype)

    indent_by_key = {}
    for p in pieces:
        for ind in p["indentations"]:
            indent_by_key[(ind["side"], ind["offset"])] = ind

    edges = [[] for _ in pieces]
    paired_protrusions = []
    for p in pieces:
        for pro in p["protrusions"]:
            ind = indent_by_key.get((OPP[pro["side"]], pro["offset"]))
            if ind is None:
                continue
            a = pro["piece"]
            b = ind["piece"]
            ar, ac = pro["local"]
            br, bc = ind["local"]
            delta = (ar - br, ac - bc)
            edges[a].append((b, delta))
            edges[b].append((a, (-delta[0], -delta[1])))
            paired_protrusions.append(pro)

    origins = [None] * len(pieces)
    origins[0] = (0, 0)
    q = deque([0])
    while q:
        cur = q.popleft()
        cr, cc = origins[cur]
        for nxt, delta in edges[cur]:
            nr, nc = cr + delta[0], cc + delta[1]
            if origins[nxt] is None:
                origins[nxt] = (nr, nc)
                q.append(nxt)

    for i, origin in enumerate(origins):
        if origin is None:
            origins[i] = (0, 0)

    min_r = min(r for r, _ in origins)
    min_c = min(c for _, c in origins)
    max_r = max(r + pieces[i]["h"] for i, (r, _) in enumerate(origins))
    max_c = max(c + pieces[i]["w"] for i, (_, c) in enumerate(origins))
    out = np.zeros((max_r - min_r, max_c - min_c), dtype=grid.dtype)

    shifted = []
    for i, p in enumerate(pieces):
        r0 = origins[i][0] - min_r
        c0 = origins[i][1] - min_c
        shifted.append((r0, c0))
        out[r0 : r0 + p["h"], c0 : c0 + p["w"]] = p["color"]

    for pro in paired_protrusions:
        p = pieces[pro["piece"]]
        r0, c0 = shifted[pro["piece"]]
        lr, lc = pro["local"]
        rr, cc = r0 + lr, c0 + lc
        if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
            out[rr, cc] = p["color"]

    return out


def _choose_layout():
    choices = [(1, 2), (2, 1), (1, 3), (3, 1), (2, 2), (1, 4), (4, 1), (2, 3), (3, 2)]
    return random.choice(choices)


def _build_random_spec():
    for _ in range(200):
        rows, cols = _choose_layout()
        row_h = [random.randint(3, 5) for _ in range(rows)]
        col_w = [random.randint(3, 5) for _ in range(cols)]
        colors = list(range(1, 9))
        random.shuffle(colors)
        pieces = []
        y = 0
        idx = 0
        for r in range(rows):
            x = 0
            for c in range(cols):
                pieces.append(
                    {
                        "tile": (r, c),
                        "color": colors[idx],
                        "h": row_h[r],
                        "w": col_w[c],
                        "origin": (y, x),
                        "protrusions": [],
                        "indentations": [],
                    }
                )
                idx += 1
                x += col_w[c]
            y += row_h[r]

        by_tile = {p["tile"]: i for i, p in enumerate(pieces)}
        used = set()
        ok = True

        joints = []
        for r in range(rows):
            for c in range(cols - 1):
                joints.append(("V", by_tile[(r, c)], by_tile[(r, c + 1)], row_h[r]))
        for r in range(rows - 1):
            for c in range(cols):
                joints.append(("H", by_tile[(r, c)], by_tile[(r + 1, c)], col_w[c]))
        random.shuffle(joints)

        for kind, a, b, span in joints:
            candidates = []
            if kind == "V":
                candidates = [("R", a, "L", b), ("L", b, "R", a)]
            else:
                candidates = [("D", a, "U", b), ("U", b, "D", a)]
            random.shuffle(candidates)
            offsets = list(range(1, span - 1))
            random.shuffle(offsets)
            picked = None
            for pro_side, pro_piece, ind_side, ind_piece in candidates:
                for off in offsets:
                    if (pro_side, off) not in used:
                        picked = (pro_side, pro_piece, ind_side, ind_piece, off)
                        break
                if picked is not None:
                    break
            if picked is None:
                ok = False
                break
            pro_side, pro_piece, ind_side, ind_piece, off = picked
            used.add((pro_side, off))
            pieces[pro_piece]["protrusions"].append((pro_side, off))
            pieces[ind_piece]["indentations"].append((ind_side, off))

        if ok:
            return pieces
    return _build_random_spec()


def generate_input() -> np.ndarray:
    pieces = _build_random_spec()
    n = len(pieces)
    slot_rows = 2 if n > 3 else 1 + random.randint(0, 1)
    slot_cols = (n + slot_rows - 1) // slot_rows
    max_h = max(p["h"] for p in pieces) + 2
    max_w = max(p["w"] for p in pieces) + 2
    cell_h = max_h + 2
    cell_w = max_w + 2
    height = min(30, slot_rows * cell_h + random.randint(0, max(0, 30 - slot_rows * cell_h)))
    width = min(30, slot_cols * cell_w + random.randint(0, max(0, 30 - slot_cols * cell_w)))
    grid = np.zeros((height, width), dtype=int)

    slots = [(r, c) for r in range(slot_rows) for c in range(slot_cols)]
    random.shuffle(slots)

    for p, (sr, sc) in zip(pieces, slots):
        sy = sr * cell_h
        sx = sc * cell_w
        y_max = min((sr + 1) * cell_h, height) - p["h"] - 1
        x_max = min((sc + 1) * cell_w, width) - p["w"] - 1
        y = random.randint(sy + 1, max(sy + 1, y_max))
        x = random.randint(sx + 1, max(sx + 1, x_max))

        grid[y : y + p["h"], x : x + p["w"]] = p["color"]
        for side, off in p["indentations"]:
            er, ec = _edge_cell(side, off, p["h"], p["w"])
            mr, mc = _outside_cell(side, off, p["h"], p["w"])
            grid[y + er, x + ec] = BG
            grid[y + mr, x + mc] = MARK
        for side, off in p["protrusions"]:
            mr, mc = _outside_cell(side, off, p["h"], p["w"])
            grid[y + mr, x + mc] = MARK

    return grid
