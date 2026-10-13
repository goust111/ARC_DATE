import numpy as np
from collections import deque

BLACK = 0
RED = 2
BLUE = 1
TL_MARKER = 3
TR_MARKER = 4
BL_MARKER = 7
BR_MARKER = 6


def _piece_components(grid: np.ndarray, marker_colors: set[int]) -> list[dict]:
    h, w = grid.shape
    passable = (grid != BLACK) & (grid != RED) & (grid != BLUE)
    for c in marker_colors:
        passable &= grid != c

    visited = np.zeros_like(passable, dtype=bool)
    pieces: list[dict] = []
    for sr, sc in zip(*np.where(passable)):
        if visited[sr, sc]:
            continue

        q = deque([(sr, sc)])
        visited[sr, sc] = True
        cells: list[tuple[int, int]] = []

        while q:
            r, c = q.popleft()
            cells.append((r, c))
            if r > 0 and passable[r - 1, c] and not visited[r - 1, c]:
                visited[r - 1, c] = True
                q.append((r - 1, c))
            if r + 1 < h and passable[r + 1, c] and not visited[r + 1, c]:
                visited[r + 1, c] = True
                q.append((r + 1, c))
            if c > 0 and passable[r, c - 1] and not visited[r, c - 1]:
                visited[r, c - 1] = True
                q.append((r, c - 1))
            if c + 1 < w and passable[r, c + 1] and not visited[r, c + 1]:
                visited[r, c + 1] = True
                q.append((r, c + 1))

        rs = [x[0] for x in cells]
        cs = [x[1] for x in cells]
        min_r, max_r = min(rs), max(rs)
        min_c, max_c = min(cs), max(cs)
        ph = max_r - min_r + 1
        pw = max_c - min_c + 1
        rel = {(r - min_r, c - min_c) for r, c in cells}
        key = tuple(sorted(rel))
        center_r = (min_r + max_r) // 2
        center_c = (min_c + max_c) // 2

        pieces.append(
            {
                "cells_abs": cells,
                "cells_set_abs": set(cells),
                "cells_rel": rel,
                "min_r": min_r,
                "max_r": max_r,
                "min_c": min_c,
                "max_c": max_c,
                "h": ph,
                "w": pw,
                "center": (center_r, center_c),
                "key": key,
            }
        )

    return pieces


def _complement_signature(piece: dict) -> tuple[tuple[int, int], ...]:
    h = piece["h"]
    w = piece["w"]
    shape = piece["cells_rel"]
    comp = []
    for r in range(h):
        for c in range(w):
            if (r, c) not in shape:
                comp.append((r, c))
    if not comp:
        return ()
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    min_r = min(rs)
    min_c = min(cs)
    return tuple(sorted((r - min_r, c - min_c) for r, c in comp))


def _find_separators(grid: np.ndarray) -> tuple[int | None, int | None]:
    h, w = grid.shape
    red_score = np.sum(grid == RED, axis=1)
    blue_in_row = np.sum(grid == BLUE, axis=1)
    red_rows = np.where((blue_in_row <= 1) & (red_score >= w - 1))[0]

    blue_score = np.sum(grid == BLUE, axis=0)
    red_in_col = np.sum(grid == RED, axis=0)
    blue_cols = np.where((red_in_col <= 1) & (blue_score >= h - 1))[0]

    if red_rows.size == 0 or blue_cols.size == 0:
        return None, None
    if red_rows.size == 1 and blue_cols.size == 1:
        return int(red_rows[0]), int(blue_cols[0])

    return int(red_rows[np.argmax(red_score[red_rows])]), int(blue_cols[np.argmax(blue_score[blue_cols])])


def _marker_colors(grid: np.ndarray, red_row: int, blue_col: int) -> tuple[int, int, int, int]:
    h, w = grid.shape
    if red_row <= 0 or red_row >= h - 1 or blue_col <= 0 or blue_col >= w - 1:
        return TL_MARKER, TR_MARKER, BL_MARKER, BR_MARKER

    tl = int(grid[red_row - 1, blue_col - 1])
    tr = int(grid[red_row - 1, blue_col + 1])
    bl = int(grid[red_row + 1, blue_col - 1])
    br = int(grid[red_row + 1, blue_col + 1])

    seen = {tl, tr, bl, br}
    if len(seen) < 4:
        tl, tr, bl, br = TL_MARKER, TR_MARKER, BL_MARKER, BR_MARKER
    return tl, tr, bl, br


def _random_connected_mask(rng: np.random.Generator, h: int, w: int, area: int) -> set[tuple[int, int]]:
    start_r = int(rng.integers(h))
    start_c = int(rng.integers(w))
    shape = {(start_r, start_c)}

    while len(shape) < area:
        border = set()
        for r, c in shape:
            if r > 0 and (r - 1, c) not in shape:
                border.add((r - 1, c))
            if r + 1 < h and (r + 1, c) not in shape:
                border.add((r + 1, c))
            if c > 0 and (r, c - 1) not in shape:
                border.add((r, c - 1))
            if c + 1 < w and (r, c + 1) not in shape:
                border.add((r, c + 1))
        if not border:
            break
        idx = int(rng.integers(len(border)))
        r, c = list(border)[idx]
        shape.add((r, c))

    return shape


def _is_connected(mask: set[tuple[int, int]], h: int, w: int) -> bool:
    if not mask:
        return False

    start = next(iter(mask))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        if r > 0 and (r - 1, c) in mask and (r - 1, c) not in seen:
            seen.add((r - 1, c))
            q.append((r - 1, c))
        if r + 1 < h and (r + 1, c) in mask and (r + 1, c) not in seen:
            seen.add((r + 1, c))
            q.append((r + 1, c))
        if c > 0 and (r, c - 1) in mask and (r, c - 1) not in seen:
            seen.add((r, c - 1))
            q.append((r, c - 1))
        if c + 1 < w and (r, c + 1) in mask and (r, c + 1) not in seen:
            seen.add((r, c + 1))
            q.append((r, c + 1))
    return len(seen) == len(mask)


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    h = int(rng.integers(18, 29))
    w = int(rng.integers(18, 29))
    grid = np.zeros((h, w), dtype=int)

    red_row = int(rng.integers(4, h - 4))
    blue_col = int(rng.integers(4, w - 4))
    grid[red_row, :] = RED
    grid[:, blue_col] = BLUE

    marker_coords = [
        (red_row - 1, blue_col - 1),
        (red_row - 1, blue_col + 1),
        (red_row + 1, blue_col - 1),
        (red_row + 1, blue_col + 1),
    ]
    marker_colors = [TL_MARKER, TR_MARKER, BL_MARKER, BR_MARKER]
    for pos, color in zip(marker_coords, marker_colors):
        grid[pos] = color

    occupied = np.zeros((h, w), dtype=bool)
    for r, c in marker_coords:
        occupied[r, c] = True
    occupied[red_row, :] = True
    occupied[:, blue_col] = True

    total_area = int(rng.choice(np.array([8, 10, 12, 14, 16, 18, 20], dtype=int)))
    half_area = total_area // 2
    factors: list[tuple[int, int]] = []
    for side_r in range(2, total_area + 1):
        if total_area % side_r != 0:
            continue
        side_c = total_area // side_r
        if side_r < h and side_c < w:
            factors.append((side_r, side_c))

    if not factors:
        factors = [(4, 4)]

    num_pairs = int(rng.integers(1, 4))
    palette = [5, 8, 9]

    for _ in range(num_pairs):
        for _attempt in range(200):
            ph, pw = factors[int(rng.integers(len(factors)))]
            piece = _random_connected_mask(rng, ph, pw, half_area)
            if len(piece) < half_area:
                continue

            complement = {(r, c) for r in range(ph) for c in range(pw) if (r, c) not in piece}
            if not _is_connected(complement, ph, pw):
                continue

            color_a = int(rng.choice(palette))
            color_b_choices = [x for x in palette if x != color_a]
            color_b = int(rng.choice(color_b_choices)) if color_b_choices else color_a

            def place(mask: set[tuple[int, int]]) -> tuple[int, int] | None:
                for _try in range(350):
                    q = int(rng.integers(4))
                    if q == 0:
                        r_lo, r_hi = 0, red_row - 2
                        c_lo, c_hi = 0, blue_col - 2
                    elif q == 1:
                        r_lo, r_hi = 0, red_row - 2
                        c_lo, c_hi = blue_col + 2, w - 1
                    elif q == 2:
                        r_lo, r_hi = red_row + 2, h - 1
                        c_lo, c_hi = 0, blue_col - 2
                    else:
                        r_lo, r_hi = red_row + 2, h - 1
                        c_lo, c_hi = blue_col + 2, w - 1

                    if r_hi - ph + 1 < r_lo or c_hi - pw + 1 < c_lo:
                        continue

                    rr = int(rng.integers(r_lo, r_hi - ph + 2))
                    cc = int(rng.integers(c_lo, c_hi - pw + 2))
                    abs_cells = [(rr + r, cc + c) for r, c in mask]
                    if any(occupied[R, C] for R, C in abs_cells):
                        continue
                    return rr, cc
                return None

            p1 = place(piece)
            if p1 is None:
                continue

            p1_cells = [(p1[0] + r, p1[1] + c) for r, c in piece]
            for R, C in p1_cells:
                occupied[R, C] = True

            p2 = place(complement)
            if p2 is None:
                for R, C in p1_cells:
                    occupied[R, C] = False
                continue

            p2_cells = [(p2[0] + r, p2[1] + c) for r, c in complement]
            for R, C in p2_cells:
                occupied[R, C] = True

            for rr, cc in piece:
                grid[p1[0] + rr, p1[1] + cc] = color_a
            for rr, cc in complement:
                grid[p2[0] + rr, p2[1] + cc] = color_b
            break

    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Part 1: 解析输入，定位红色水平分隔线、蓝色垂直分隔线与四个角落标记色。
    Part 2: 识别非背景非分隔块并提取连通块，按外接矩形反掩膜匹配补片形状。
    Part 3: 按对匹配后，用两块外接矩形中心中点确定拼合矩形的放置起点（取整）。
    Part 4: 按拼合后碎片质心所在象限重绘为对应角落颜色，并返回仅含重绘矩形的网格。
    """
    g = np.asarray(grid)
    out = np.zeros_like(g)

    red_row, blue_col = _find_separators(g)
    if red_row is None or blue_col is None:
        return out

    tl_marker, tr_marker, bl_marker, br_marker = _marker_colors(g, red_row, blue_col)
    marker_set = {tl_marker, tr_marker, bl_marker, br_marker}

    pieces = _piece_components(g, marker_set)
    if len(pieces) < 2:
        return out

    index_by_piece: dict[int, list[int]] = {}
    for i, p in enumerate(pieces):
        index_by_piece.setdefault(hash(p["key"]), []).append(i)

    used = np.zeros(len(pieces), dtype=bool)
    pairs: list[tuple[int, int]] = []
    for i, piece_i in enumerate(pieces):
        if used[i]:
            continue
        sig = _complement_signature(piece_i)
        sig_hash = hash(sig)
        candidates = index_by_piece.get(sig_hash, [])
        chosen = None
        for j in candidates:
            if used[j] or j == i:
                continue
            if pieces[j]["key"] == sig:
                chosen = j
                break
        if chosen is None:
            continue
        used[i] = True
        used[chosen] = True
        pairs.append((i, chosen))

    for left_i, right_i in pairs:
        p1 = pieces[left_i]
        p2 = pieces[right_i]

        ph, pw = p1["h"], p1["w"]
        if ph != p2["h"] or pw != p2["w"]:
            ph = max(ph, p2["h"])
            pw = max(pw, p2["w"])

        if ph <= 0 or pw <= 0:
            continue

        mid_r = (p1["center"][0] + p2["center"][0]) // 2
        mid_c = (p1["center"][1] + p2["center"][1]) // 2

        top_r = mid_r - ph // 2
        top_c = mid_c - pw // 2

        H, W = g.shape
        top_r = max(0, min(top_r, H - ph))
        top_c = max(0, min(top_c, W - pw))

        def recolor_piece(piece: dict, color: int) -> None:
            rel = piece["cells_rel"]
            final = [(top_r + r, top_c + c) for r, c in rel]
            if not final:
                return
            cr = sum(r for r, _ in final) / len(final)
            cc = sum(c for _, c in final) / len(final)
            if cr < red_row:
                color = tl_marker if cc < blue_col else tr_marker
            else:
                color = bl_marker if cc < blue_col else br_marker
            for R, C in final:
                if 0 <= R < H and 0 <= C < W:
                    out[R, C] = color

        recolor_piece(p1, TL_MARKER)
        recolor_piece(p2, TR_MARKER)

    return out
