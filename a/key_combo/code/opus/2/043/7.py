import random
from collections import deque

import numpy as np

BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    vis = np.zeros((h, w), dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or vis[r, c]:
                continue
            q = deque([(r, c)])
            vis[r, c] = True
            comp = [(r, c)]
            while q:
                cr, cc = q.popleft()
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if _in_bounds((h, w), nr, nc) and mask[nr, nc] and not vis[nr, nc]:
                        vis[nr, nc] = True
                        q.append((nr, nc))
                        comp.append((nr, nc))
            out.append(comp)
    return out


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _normalize(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    r0 = min(r for r, _ in coords)
    c0 = min(c for _, c in coords)
    return sorted((r - r0, c - c0) for r, c in coords)


def _translate(coords: list[tuple[int, int]], tr: int, tc: int) -> list[tuple[int, int]]:
    return [(r + tr, c + tc) for r, c in coords]


def _transform_coords(coords: list[tuple[int, int]], t: int) -> list[tuple[int, int]]:
    # 8 orientations: 4 rotations + mirrored
    arr = [(int(r), int(c)) for r, c in coords]
    if t == 0:
        out = [(r, c) for r, c in arr]
    elif t == 1:
        out = [(-c, r) for r, c in arr]
    elif t == 2:
        out = [(-r, -c) for r, c in arr]
    elif t == 3:
        out = [(c, -r) for r, c in arr]
    elif t == 4:
        out = [(-r, c) for r, c in arr]
    elif t == 5:
        out = [(c, r) for r, c in arr]
    elif t == 6:
        out = [(r, -c) for r, c in arr]
    else:
        out = [(-c, -r) for r, c in arr]
    return _normalize(out)


def _match_piece_to_hole(piece_coords: list[tuple[int, int]], hole_coords: list[tuple[int, int]]) -> tuple[int, int] | None:
    pn = _normalize(piece_coords)
    hn = _normalize(hole_coords)
    hp = set(hn)
    for t in range(8):
        tr_coords = _transform_coords(pn, t)
        for p0, h0 in zip(tr_coords, hn):
            dr = h0[0] - p0[0]
            dc = h0[1] - p0[1]
            shifted = {(r + dr, c + dc) for r, c in tr_coords}
            if shifted == hp:
                return dr, dc
    return None


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int, copy=True)
    h, w = grid.shape

    # 1) frame
    comp_orange = _components(grid == ORANGE)
    if not comp_orange:
        return grid
    frame = max(comp_orange, key=len)
    fr0, fc0, fr1, fc1 = _bbox(frame)

    # 2) expected perimeter holes (cells where orange is missing)
    perimeter: set[tuple[int, int]] = set()
    for c in range(fc0, fc1 + 1):
        perimeter.add((fr0, c))
        perimeter.add((fr1, c))
    for r in range(fr0, fr1 + 1):
        perimeter.add((r, fc0))
        perimeter.add((r, fc1))

    hole_mask = np.zeros((h, w), dtype=bool)
    for r, c in perimeter:
        if grid[r, c] == BLACK:
            hole_mask[r, c] = True
    holes = [comp for comp in _components(hole_mask) if 2 <= len(comp) <= 4]

    # 3) gray pieces
    piece_mask = grid == GREY
    pieces = [comp for comp in _components(piece_mask) if 2 <= len(comp) <= 4]

    out = grid.copy()
    used_hole = set()
    used_piece = set()

    # 4) fit each piece onto a hole
    for pi, piece in enumerate(pieces):
        if pi in used_piece:
            continue
        for hi, hole in enumerate(holes):
            if hi in used_hole or len(piece) != len(hole):
                continue
            t = _match_piece_to_hole(piece, hole)
            if t is None:
                continue
            dr, dc = t
            for rr, cc in piece:
                tr, tc = rr + dr, cc + dc
                if not _in_bounds((h, w), tr, tc):
                    break
            else:
                # place
                for rr, cc in piece:
                    out[rr, cc] = BLACK
                    tr, tc = rr + dr, cc + dc
                    out[tr, tc] = BLUE
                used_hole.add(hi)
                used_piece.add(pi)
                break
    # 5) fill interior regions
    inside_black = np.zeros((h, w), dtype=bool)
    if fr1 - fr0 >= 2 and fc1 - fc0 >= 2:
        inside_black[fr0 + 1 : fr1, fc0 + 1 : fc1] = (
            out[fr0 + 1 : fr1, fc0 + 1 : fc1] == BLACK
        )
        comps = _components(inside_black)
        for comp in comps:
            touches_bottom_wall = False
            for r, c in comp:
                if r + 1 < h and out[r + 1, c] == ORANGE:
                    touches_bottom_wall = True
                    break
            fill_col = GREEN if touches_bottom_wall else YELLOW
            for r, c in comp:
                out[r, c] = fill_col
    return out


def _make_hole(shape_coords: list[tuple[int, int]], grid: np.ndarray) -> list[tuple[int, int]]:
    # remove chosen perimeter cells (mutates grid)
    for r, c in shape_coords:
        grid[r, c] = BLACK
    return shape_coords


def generate_input() -> np.ndarray:
    h = 15
    w = 15
    while True:
        grid = np.full((h, w), BLACK, dtype=int)

        top = random.randint(2, 3)
        left = random.randint(2, 3)
        height = random.randint(8, 10)
        width = random.randint(8, 10)
        bottom = min(h - 2, top + height)
        right = min(w - 2, left + width)

        # orange frame
        grid[top, left : right + 1] = ORANGE
        grid[bottom, left : right + 1] = ORANGE
        grid[top : bottom + 1, left] = ORANGE
        grid[top : bottom + 1, right] = ORANGE

        # red divider inside frame
        rr = random.randint(top + 2, max(top + 2, bottom - 2))
        cc = random.randint(left + 2, max(left + 2, right - 2))
        if random.random() < 0.5:
            cc0 = random.randint(left + 1, right - 1)
            cc1 = random.randint(left + 1, right - 1)
            c = max(cc0, cc1)
            for ccol in range(min(cc0, cc1), c + 1):
                grid[rr, ccol] = RED
        else:
            rr0 = random.randint(top + 1, bottom - 1)
            rr1 = random.randint(top + 1, bottom - 1)
            r0 = max(rr0, rr1)
            for r in range(min(rr0, rr1), r0 + 1):
                grid[r, cc] = RED

        # create 2~3 holes on perimeter
        n_holes = random.randint(2, 3)
        holes: list[list[tuple[int, int]]] = []
        candidate_cells = []
        for c in range(left + 1, right):
            candidate_cells.append((top, c))
            candidate_cells.append((bottom, c))
        for r in range(top + 1, bottom):
            candidate_cells.append((r, left))
            candidate_cells.append((r, right))

        random.shuffle(candidate_cells)
        idx = 0
        for _ in range(n_holes):
            if idx >= len(candidate_cells):
                break
            sr, sc = candidate_cells[idx]
            idx += 1
            orientation = random.choice([(1, 0), (0, 1)])  # horizontal/vertical segment
            length = random.randint(2, 4)
            dr, dc = orientation
            hole = []
            for k in range(length):
                rr = sr + dr * k
                cc = sc + dc * k
                if not _in_bounds((h, w), rr, cc) or rr not in (top, bottom, ) and cc not in (left, right):
                    break
                if not _in_bounds((h, w), rr, cc):
                    break
                if (rr, cc) in [(top, left), (top, right), (bottom, left), (bottom, right)]:
                    pass
                hole.append((rr, cc))
            if len(hole) >= 2:
                for r, c in hole:
                    if _in_bounds((h, w), r, c):
                        grid[r, c] = BLACK
                holes.append(hole)

        # place transformed gray pieces outside frame
        gray_cells = []
        piece_colors = [GREY]
        for hole in holes:
            shape = hole[:]
            transformed = _transform_coords(shape, random.randint(0, 7))
            # normalize shape absolute offsets
            tr, tc = transformed[0]
            max_r = max(r for r, c in transformed)
            max_c = max(c for r, c in transformed)
            # candidate top-left in outside area
            outside = []
            for r0 in range(h):
                for c0 in range(w):
                    if top <= r0 <= bottom and left <= c0 <= right:
                        continue
                    positions = [(r0 + r, c0 + c) for r, c in transformed]
                    if all(_in_bounds((h, w), rr, cc) and grid[rr, cc] == BLACK for rr, cc in positions):
                        outside.append((r0, c0))
            if not outside:
                continue
            base_r, base_c = random.choice(outside)
            piece = [(base_r + r, base_c + c) for r, c in transformed]
            for rr, cc in piece:
                grid[rr, cc] = GREY
            gray_cells.extend(piece)

        # ensure all outputs stay bounded and valid
        if len(gray_cells) == 0:
            # still accepted, but keep at least one piece
            grid[0, 0] = BLUE
            return grid
        return grid


_generate_input_single = generate_input


def generate_inputs() -> list[np.ndarray]:
    outputs: list[np.ndarray] = []
    attempts = 0
    while len(outputs) < 20 and attempts < 200:
        attempts += 1
        g = _generate_input_single()
        arr = np.array(g, dtype=int, copy=True)
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        elif arr.ndim == 1:
            arr = arr.reshape(1, -1)
        elif arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        if arr.ndim != 2 or arr.shape[0] > 30 or arr.shape[1] > 30:
            continue
        outputs.append(arr)

    while len(outputs) < 20:
        if outputs:
            outputs.append(outputs[-1].copy())
        else:
            outputs.append(np.zeros((1, 1), dtype=int))
    return outputs


    random.seed(0)
    print(main(generate_input()).shape)
