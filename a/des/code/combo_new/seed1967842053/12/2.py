import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    comps = []
    rows, cols = mask.shape
    for r in range(rows):
        for c in range(cols):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < rows and 0 <= nc < cols and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(cells)
    return comps


def _norm(cells):
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return frozenset((r - min_r, c - min_c) for r, c in cells)


def _shape_size(shape):
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _frame_cavities(grid):
    cavities = []
    for grey_cells in _components(grid == GREY):
        r0 = min(r for r, _ in grey_cells)
        r1 = max(r for r, _ in grey_cells)
        c0 = min(c for _, c in grey_cells)
        c1 = max(c for _, c in grey_cells)
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        for bg_cells in _components(sub == BLACK):
            touches_edge = any(
                r == 0 or c == 0 or r == sub.shape[0] - 1 or c == sub.shape[1] - 1
                for r, c in bg_cells
            )
            if not touches_edge:
                abs_cells = [(r0 + r, c0 + c) for r, c in bg_cells]
                cavities.append((_norm(abs_cells), abs_cells))
    return cavities


def main(grid: np.ndarray) -> np.ndarray:
    """Move each colored key fragment into the matching grey lock cavity.

    Input structure:
    The grid has a black background, several grey rectangular frame components,
    and separate single-color connected fragments.  Each grey frame contains one
    enclosed black connected cavity, and every colored fragment has exactly the
    same pixel outline as one cavity.  Fragments do not touch the grey frames.

    Rule:
    For each enclosed cavity inside a grey frame, compare its normalized pixel
    shape with the normalized shapes of the scattered colored fragments.  The
    fragment whose shape matches is removed from its original position and its
    color is placed into the cavity cells.  Grey pixels and all other background
    pixels remain unchanged.

    Key insight:
    The pairing is determined by exact outline equality after translation, not
    by color, size alone, or relative position.
    """
    out = np.array(grid, copy=True)
    cavities = _frame_cavities(grid)

    pieces = {}
    colored = (grid != BLACK) & (grid != GREY)
    for cells in _components(colored):
        color = int(grid[cells[0]])
        shape = _norm(cells)
        pieces.setdefault(shape, []).append((color, cells))

    for shape, cavity_cells in cavities:
        matches = pieces.get(shape, [])
        if not matches:
            continue
        color, piece_cells = matches.pop(0)
        for r, c in piece_cells:
            out[r, c] = BLACK
        for r, c in cavity_cells:
            out[r, c] = color

    return out


def _make_shape(rng, h, w):
    target = rng.randint(4, max(4, min(11, h * w - 1)))
    cells = {(rng.randrange(h), rng.randrange(w))}
    while len(cells) < target:
        r, c = rng.choice(tuple(cells))
        nbrs = [
            (r + dr, c + dc)
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
            if 0 <= r + dr < h and 0 <= c + dc < w
        ]
        cells.add(rng.choice(nbrs))

    # Trim empty margins so shape comparison is controlled by visible pixels.
    normed = _norm(cells)
    return normed


def _dilated_cells(top, left, shape, pad=1):
    cells = set()
    for r, c in shape:
        ar, ac = top + r, left + c
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                cells.add((ar + dr, ac + dc))
    return cells


def _rect_cells(top, left, h, w, pad=0):
    return {
        (r, c)
        for r in range(top - pad, top + h + pad)
        for c in range(left - pad, left + w + pad)
    }


def _fits(cells, rows, cols, occupied):
    return all(0 <= r < rows and 0 <= c < cols and (r, c) not in occupied for r, c in cells)


def _place_rect_with_hole(grid, top, left, shape):
    sh, sw = _shape_size(shape)
    rect_h, rect_w = sh + 4, sw + 4
    grid[top : top + rect_h, left : left + rect_w] = GREY
    hole_top, hole_left = top + 2, left + 2
    for r, c in shape:
        grid[hole_top + r, hole_left + c] = BLACK


def _place_piece(grid, top, left, shape, color):
    for r, c in shape:
        grid[top + r, left + c] = color


def _random_position(rng, rows, cols, h, w, occupied, as_rect, shape=None):
    candidates = []
    for top in range(1, rows - h):
        for left in range(1, cols - w):
            cells = _rect_cells(top, left, h, w, 1) if as_rect else _dilated_cells(top, left, shape)
            if _fits(cells, rows, cols, occupied):
                candidates.append((top, left, cells))
    if not candidates:
        return None
    return rng.choice(candidates)


def generate_input() -> np.ndarray:
    rows = random.randint(20, 30)
    cols = random.randint(20, 30)
    grid = np.zeros((rows, cols), dtype=int)
    rng = random
    colors = [1, 2, 3, 4, 6, 7, 8, 9]
    rng.shuffle(colors)

    for _ in range(80):
        n = rng.randint(2, 4)
        shapes = []
        seen = set()
        for _j in range(n):
            for _k in range(80):
                shape = _make_shape(rng, rng.randint(3, 5), rng.randint(3, 5))
                if shape not in seen:
                    seen.add(shape)
                    shapes.append(shape)
                    break
        if len(shapes) != n:
            continue

        trial = np.zeros((rows, cols), dtype=int)
        occupied = set()
        placements = []
        order = list(range(n))
        rng.shuffle(order)
        ok = True
        for idx in order:
            sh, sw = _shape_size(shapes[idx])
            rect_h, rect_w = sh + 4, sw + 4
            pos = _random_position(rng, rows, cols, rect_h, rect_w, occupied, True)
            if pos is None:
                ok = False
                break
            top, left, blocked = pos
            occupied.update(blocked)
            placements.append(("frame", idx, top, left))

        if not ok:
            continue

        piece_order = list(range(n))
        rng.shuffle(piece_order)
        for idx in piece_order:
            sh, sw = _shape_size(shapes[idx])
            pos = _random_position(rng, rows, cols, sh, sw, occupied, False, shapes[idx])
            if pos is None:
                ok = False
                break
            top, left, blocked = pos
            occupied.update(blocked)
            placements.append(("piece", idx, top, left))

        if not ok:
            continue

        for kind, idx, top, left in placements:
            if kind == "frame":
                _place_rect_with_hole(trial, top, left, shapes[idx])
            else:
                _place_piece(trial, top, left, shapes[idx], colors[idx])
        return trial

    # Conservative fallback, still randomized in color and placement order.
    grid = np.zeros((24, 24), dtype=int)
    shapes = [
        frozenset({(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)}),
        frozenset({(0, 1), (1, 0), (1, 1), (1, 2), (2, 2)}),
    ]
    _place_rect_with_hole(grid, 1, 1, shapes[0])
    _place_rect_with_hole(grid, 1, 12, shapes[1])
    _place_piece(grid, 16, 3, shapes[1], colors[1])
    _place_piece(grid, 16, 15, shapes[0], colors[0])
    return grid
