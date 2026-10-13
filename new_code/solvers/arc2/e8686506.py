import numpy as np

from arc_utils import bbox_pts as _bbox_pts, cc4 as _cc4, get_bg as _mode_color


def _piece_sig(pts: list[tuple[int, int]]) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    r0, c0, r1, c1 = _bbox_pts(pts)
    coords = tuple(sorted((r - r0, c - c0) for r, c in pts))
    return (r1 - r0 + 1), (c1 - c0 + 1), coords


def _factors(n: int) -> list[tuple[int, int]]:
    out = []
    for w in range(1, n + 1):
        if n % w == 0:
            h = n // w
            out.append((h, w))
    return out


def _build_requirements(
    g: np.ndarray,
    bg: int,
    H: int,
    W: int,
    comps_by_color: dict[int, list[list[tuple[int, int]]]],
) -> np.ndarray:
    """
    Build per-cell required color constraints (req[r,c] == -1 means unconstrained).

    Heuristics supported by training examples:
      - If there is a color with exactly 4 singleton components, those are the four corners.
      - If there are full horizontal bars of length W-2 in the input, use them as the top/bottom edges
        between the corners (ordered by their vertical position in the input).
      - If there is a color with exactly two full vertical line components (width=1) in the input,
        place them on the left/right edges centered vertically.
    """
    req = np.full((H, W), -1, dtype=int)

    # 4 singleton components -> corners
    corner_color = None
    for col, comps in comps_by_color.items():
        if len(comps) == 4 and all(len(p) == 1 for p in comps):
            corner_color = col
            break
    if corner_color is not None:
        req[0, 0] = corner_color
        req[0, W - 1] = corner_color
        req[H - 1, 0] = corner_color
        req[H - 1, W - 1] = corner_color

    # Horizontal bar(s) of length W-2 (filled, height 1)
    bar_colors: list[tuple[int, int]] = []  # (min_input_row, color)
    if W >= 3 and corner_color is not None:
        for col, comps in comps_by_color.items():
            for pts in comps:
                r0, c0, r1, c1 = _bbox_pts(pts)
                hh, ww = r1 - r0 + 1, c1 - c0 + 1
                if hh == 1 and ww == (W - 2) and len(pts) == ww:
                    bar_colors.append((r0, col))
                    break
        if bar_colors:
            bar_colors.sort()
            top_col = bar_colors[0][1]
            bot_col = bar_colors[-1][1]
            req[0, 1 : W - 1] = top_col
            req[H - 1, 1 : W - 1] = bot_col

    # Two full vertical line components -> left/right edges, centered vertically
    for col, comps in comps_by_color.items():
        if len(comps) != 2:
            continue
        sigs = [_piece_sig(pts) for pts in comps]
        (h0, w0, coords0), (h1, w1, coords1) = sigs[0], sigs[1]
        if w0 == 1 and w1 == 1 and h0 == h1 and coords0 == coords1 and h0 > 1:
            L = h0
            top = (H - L) // 2
            if 0 <= top and top + L <= H:
                req[top : top + L, 0] = col
                req[top : top + L, W - 1] = col
            break

    return req


def _assemble(
    types: list[dict],
    H: int,
    W: int,
    req: np.ndarray,
    disallow_touch: set[int],
) -> np.ndarray | None:
    board = np.full((H, W), -1, dtype=int)
    counts = [int(t["cnt"]) for t in types]

    def mirror_ok(r: int, c: int, col: int) -> bool:
        mc = W - 1 - c
        v = board[r, mc]
        return (v == -1) or (v == col)

    def can_place(ti: int, r0: int, c0: int) -> bool:
        t = types[ti]
        col = int(t["col"])
        for dr, dc in t["coords"]:
            r, c = r0 + dr, c0 + dc
            if board[r, c] != -1:
                return False
            if req[r, c] != -1 and int(req[r, c]) != col:
                return False
            if not mirror_ok(r, c, col):
                return False
            if col in disallow_touch:
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= rr < H and 0 <= cc < W and board[rr, cc] == col:
                        return False
        return True

    def place(ti: int, r0: int, c0: int) -> list[tuple[int, int]]:
        t = types[ti]
        col = int(t["col"])
        changed = []
        for dr, dc in t["coords"]:
            r, c = r0 + dr, c0 + dc
            board[r, c] = col
            changed.append((r, c))
        return changed

    def unplace(changed: list[tuple[int, int]]) -> None:
        for r, c in changed:
            board[r, c] = -1

    def next_empty() -> tuple[int, int] | None:
        for r in range(H):
            for c in range(W):
                if board[r, c] == -1:
                    return r, c
        return None

    def bt() -> bool:
        pos = next_empty()
        if pos is None:
            # full; enforce exact symmetry
            return bool(np.array_equal(board, board[:, ::-1]))
        r, c = pos
        # If the mirror cell is already filled, it forces this cell's color, but it's empty now.
        forced = board[r, W - 1 - c]
        forced_col = int(forced) if forced != -1 else None
        req_col = int(req[r, c]) if req[r, c] != -1 else None
        if req_col is not None and forced_col is not None and req_col != forced_col:
            return False

        for ti, t in enumerate(types):
            if counts[ti] <= 0:
                continue
            col = int(t["col"])
            if req_col is not None and col != req_col:
                continue
            if forced_col is not None and col != forced_col:
                continue
            # try align each cell of the piece to (r,c)
            for dr, dc in t["coords"]:
                r0, c0 = r - dr, c - dc
                if r0 < 0 or c0 < 0 or r0 + int(t["h"]) > H or c0 + int(t["w"]) > W:
                    continue
                if not can_place(ti, r0, c0):
                    continue
                counts[ti] -= 1
                ch = place(ti, r0, c0)
                if bt():
                    return True
                unplace(ch)
                counts[ti] += 1
        return False

    if bt():
        return board
    return None


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered colored fragments into a left-right symmetric rectangle, jigsaw-style.

    Input structure:
    - bg = most common color; remaining colors form small sprite fragments.
    - Special roles: 4 singleton pixels of one color mark corners; horizontal bars mark top/bottom edges; paired vertical lines mark left/right edges.

    Rule:
    - Collect all non-bg connected components as jigsaw pieces.
    - Determine output rectangle dimensions from total piece area.
    - Pack pieces into the rectangle so it is exactly mirror-symmetric (left-right) and same-color pieces never 4-touch.

    Key insight: Left-right symmetry plus the no-same-color-touching constraint make the assembly unique; corner/edge markers anchor the solution.
    """
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)

    # Extract pieces (connected components per color)
    comps_by_color: dict[int, list[list[tuple[int, int]]]] = {}
    pieces = []
    color_piece_count: dict[int, int] = {}

    for col in sorted(set(map(int, np.unique(g)))):
        if col == bg:
            continue
        comps = _cc4(g == col)
        comps_by_color[col] = comps
        color_piece_count[col] = len(comps)
        for pts in comps:
            _, _, coords = _piece_sig(pts)
            pieces.append((col, coords))

    # Group identical pieces into types
    from collections import Counter

    ctr = Counter(pieces)
    types = []
    for (col, coords), cnt in sorted(ctr.items(), key=lambda kv: (-len(kv[0][1]), kv[0][0], kv[0][1])):
        arr = np.array(coords, dtype=int)
        h = int(arr[:, 0].max() + 1)
        w = int(arr[:, 1].max() + 1)
        types.append({"col": int(col), "coords": coords, "h": h, "w": w, "cnt": int(cnt)})

    total_area = sum(int(t["cnt"]) * len(t["coords"]) for t in types)
    if total_area == 0:
        return np.zeros((0, 0), dtype=int)

    # Disallow 4-neigh touching between different pieces of same color (preserve component counts)
    disallow_touch = {col for col, n in color_piece_count.items() if n > 1}

    # Try candidate output shapes (prefer close-to-square; prefer portrait h>=w)
    dims = [(h, w) for (h, w) in _factors(total_area) if h >= w]
    dims.sort(key=lambda hw: (abs(hw[0] - hw[1]), -hw[1], hw[0]))

    for H, W in dims:
        req = _build_requirements(g, bg, H, W, comps_by_color)
        sol = _assemble(types, H, W, req, disallow_touch)
        if sol is not None:
            return sol.astype(int)

    # Fallback: try landscape if needed
    dims = [(h, w) for (h, w) in _factors(total_area)]
    dims.sort(key=lambda hw: (abs(hw[0] - hw[1]), hw[0] + hw[1]))
    for H, W in dims:
        req = _build_requirements(g, bg, H, W, comps_by_color)
        sol = _assemble(types, H, W, req, disallow_touch)
        if sol is not None:
            return sol.astype(int)

    # If nothing found, return an empty canvas (should not happen for this task).
    return np.zeros((0, 0), dtype=int)
