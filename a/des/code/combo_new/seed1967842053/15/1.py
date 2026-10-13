import random
from collections import Counter, deque

import numpy as np


BG = 0
GREY = 5
QUAD_COLORS = [2, 1, 3, 4]
PIECE_COLORS = [6, 7, 8, 9]


def _normalize(cells):
    cells = tuple(cells)
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _rotate_once(cells):
    return _normalize((c, -r) for r, c in cells)


def _rotations(cells):
    cur = _normalize(cells)
    out = []
    for _ in range(4):
        if cur not in out:
            out.append(cur)
        cur = _rotate_once(cur)
    return out


def _components(mask):
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    comps = []
    for sr in range(h):
        for sc in range(w):
            if not mask[sr, sc] or seen[sr, sc]:
                continue
            q = deque([(sr, sc)])
            seen[sr, sc] = True
            comp = []
            while q:
                r, c = q.popleft()
                comp.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _mode(values):
    vals = list(values)
    return Counter(vals).most_common(1)[0][0]


def _color_mode(block, forbidden):
    vals = [int(x) for x in block.ravel() if int(x) not in forbidden]
    return _mode(vals) if vals else BG


def _same_shape(a, b):
    target = _normalize(a)
    return any(rot == target for rot in _rotations(b))


def _find_frame(grid, bg):
    grey_mask = grid == GREY
    rs, cs = np.where(grey_mask)
    if len(rs) == 0:
        return None
    top, bottom = int(rs.min()), int(rs.max())
    left, right = int(cs.min()), int(cs.max())
    sub = grey_mask[top : bottom + 1, left : right + 1]
    row_counts = sub.sum(axis=1)
    col_counts = sub.sum(axis=0)
    if bottom - top < 4 or right - left < 4:
        return None
    inner_rows = list(range(top + 1, bottom))
    inner_cols = list(range(left + 1, right))
    split_r = max(inner_rows, key=lambda r: int(row_counts[r - top]))
    split_c = max(inner_cols, key=lambda c: int(col_counts[c - left]))
    return top, bottom, left, right, split_r, split_c


def main(grid: np.ndarray) -> np.ndarray:
    """Embed scattered shape-matched fragments into frame gaps and recolor them by quadrant.

    Input structure:
    A black-background grid contains one large grey rectangular one-pixel frame.
    A grey horizontal divider and a grey vertical divider split the frame interior
    into four colored quadrants.  Small connected background-colored gaps lie on
    the outer frame border, and same-shape single-color fragments are scattered
    outside the frame.

    Rule:
    Infer the frame, its two dividers, and the visible fill color of each
    quadrant.  For every connected background gap inside the frame box that
    touches the outer border, find an unused scattered fragment whose shape
    matches the gap under rotation only.  Remove that fragment from its scattered
    position and fill the gap cells with the color of the quadrant nearest to the
    gap center.  All other frame, divider, quadrant, and background cells remain
    unchanged.

    Key insight:
    The fragment color is only an identifier for matching; the final color is
    determined by the spatial quadrant where the matching shape is embedded.
    """
    grid = np.asarray(grid)
    out = grid.copy()
    bg = _mode(int(x) for x in grid.ravel())
    frame = _find_frame(grid, bg)
    if frame is None:
        return out

    top, bottom, left, right, split_r, split_c = frame
    forbidden = {bg, GREY}
    quad = {
        (0, 0): _color_mode(grid[top + 1 : split_r, left + 1 : split_c], forbidden),
        (0, 1): _color_mode(grid[top + 1 : split_r, split_c + 1 : right], forbidden),
        (1, 0): _color_mode(grid[split_r + 1 : bottom, left + 1 : split_c], forbidden),
        (1, 1): _color_mode(grid[split_r + 1 : bottom, split_c + 1 : right], forbidden),
    }

    inside_bg = grid[top : bottom + 1, left : right + 1] == bg
    gaps = []
    for comp in _components(inside_bg):
        abs_comp = [(r + top, c + left) for r, c in comp]
        if any(r in (top, bottom) or c in (left, right) for r, c in abs_comp):
            gaps.append(abs_comp)

    outside = np.ones(grid.shape, dtype=bool)
    outside[top : bottom + 1, left : right + 1] = False
    piece_mask = outside & (grid != bg) & (grid != GREY)
    seen = np.zeros(grid.shape, dtype=bool)
    pieces = []
    h, w = grid.shape
    for sr, sc in zip(*np.where(piece_mask)):
        if seen[sr, sc]:
            continue
        color = int(grid[sr, sc])
        q = deque([(int(sr), int(sc))])
        seen[sr, sc] = True
        comp = []
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < h
                    and 0 <= nc < w
                    and not seen[nr, nc]
                    and piece_mask[nr, nc]
                    and int(grid[nr, nc]) == color
                ):
                    seen[nr, nc] = True
                    q.append((nr, nc))
        pieces.append({"cells": comp, "shape": _normalize(comp), "used": False})

    gaps.sort(key=lambda comp: (sum(r for r, _ in comp) / len(comp), sum(c for _, c in comp) / len(comp)))
    for gap in gaps:
        gap_shape = _normalize(gap)
        match = None
        for piece in pieces:
            if not piece["used"] and _same_shape(gap_shape, piece["shape"]):
                match = piece
                break
        if match is None:
            continue
        match["used"] = True
        for r, c in match["cells"]:
            out[r, c] = bg
        center_r = sum(r for r, _ in gap) / len(gap)
        center_c = sum(c for _, c in gap) / len(gap)
        qr = 0 if center_r < split_r else 1
        qc = 0 if center_c < split_c else 1
        fill = quad[(qr, qc)]
        for r, c in gap:
            out[r, c] = fill
    return out


BASE_SHAPES = [
    ((0, 0), (0, 1)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 1), (0, 2), (1, 0), (1, 1)),
]


def _draw_frame(grid, top, bottom, left, right, split_r, split_c, colors):
    grid[top + 1 : split_r, left + 1 : split_c] = colors[0]
    grid[top + 1 : split_r, split_c + 1 : right] = colors[1]
    grid[split_r + 1 : bottom, left + 1 : split_c] = colors[2]
    grid[split_r + 1 : bottom, split_c + 1 : right] = colors[3]
    grid[top, left : right + 1] = GREY
    grid[bottom, left : right + 1] = GREY
    grid[top : bottom + 1, left] = GREY
    grid[top : bottom + 1, right] = GREY
    grid[split_r, left : right + 1] = GREY
    grid[top : bottom + 1, split_c] = GREY


def _touches(cells, occupied):
    for r, c in cells:
        for dr, dc in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
            if (r + dr, c + dc) in occupied:
                return True
    return False


def _place_gap(shape, side, top, bottom, left, right, split_r, split_c, used):
    for _ in range(250):
        cells = random.choice(_rotations(shape))
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        if side == "top":
            row0 = top - min(rs)
            col0 = random.randint(left + 2, right - (max(cs) - min(cs)) - 2) - min(cs)
        elif side == "bottom":
            row0 = bottom - max(rs)
            col0 = random.randint(left + 2, right - (max(cs) - min(cs)) - 2) - min(cs)
        elif side == "left":
            row0 = random.randint(top + 2, bottom - (max(rs) - min(rs)) - 2) - min(rs)
            col0 = left - min(cs)
        else:
            row0 = random.randint(top + 2, bottom - (max(rs) - min(rs)) - 2) - min(rs)
            col0 = right - max(cs)
        placed = tuple(sorted((r + row0, c + col0) for r, c in cells))
        if not all(top <= r <= bottom and left <= c <= right for r, c in placed):
            continue
        if any(r == split_r or c == split_c for r, c in placed):
            continue
        if any((r in (top, bottom) and c in (left, right)) for r, c in placed):
            continue
        if not any(r in (top, bottom) or c in (left, right) for r, c in placed):
            continue
        if _touches(placed, used):
            continue
        return placed
    return None


def _place_piece(grid, shape, color, top, bottom, left, right, occupied):
    h, w = grid.shape
    for _ in range(800):
        cells = random.choice(_rotations(shape))
        bh = max(r for r, _ in cells) + 1
        bw = max(c for _, c in cells) + 1
        row0 = random.randint(1, h - bh - 1)
        col0 = random.randint(1, w - bw - 1)
        placed = tuple(sorted((r + row0, c + col0) for r, c in cells))
        if any(top <= r <= bottom and left <= c <= right for r, c in placed):
            continue
        if any(grid[r, c] != BG for r, c in placed):
            continue
        if _touches(placed, occupied):
            continue
        for r, c in placed:
            grid[r, c] = color
        occupied.update(placed)
        return True
    return False


def generate_input() -> np.ndarray:
    for _ in range(200):
        h = random.randint(23, 30)
        w = random.randint(23, 30)
        frame_h = random.randint(12, min(18, h - 6))
        frame_w = random.randint(12, min(18, w - 6))
        top = random.randint(3, h - frame_h - 3)
        left = random.randint(3, w - frame_w - 3)
        bottom = top + frame_h - 1
        right = left + frame_w - 1
        split_r = random.randint(top + 4, bottom - 4)
        split_c = random.randint(left + 4, right - 4)

        grid = np.full((h, w), BG, dtype=int)
        quad_colors = QUAD_COLORS[:]
        random.shuffle(quad_colors)
        _draw_frame(grid, top, bottom, left, right, split_r, split_c, quad_colors)

        n = random.randint(2, 4)
        shapes = random.sample(BASE_SHAPES, n)
        sides = random.sample(["top", "bottom", "left", "right"], n)
        random.shuffle(sides)
        gap_cells = []
        used_gaps = set()
        ok = True
        for shape, side in zip(shapes, sides):
            placed = _place_gap(shape, side, top, bottom, left, right, split_r, split_c, used_gaps)
            if placed is None:
                ok = False
                break
            for r, c in placed:
                grid[r, c] = BG
            used_gaps.update(placed)
            gap_cells.append(placed)
        if not ok:
            continue

        piece_colors = random.sample(PIECE_COLORS, n)
        occupied_pieces = set()
        order = list(range(n))
        random.shuffle(order)
        for idx in order:
            if not _place_piece(grid, shapes[idx], piece_colors[idx], top, bottom, left, right, occupied_pieces):
                ok = False
                break
        if ok:
            return grid
    raise RuntimeError("failed to generate a valid grid")
