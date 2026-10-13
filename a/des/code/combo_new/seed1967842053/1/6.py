import random
from collections import deque

import numpy as np


def _components(grid):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] == 0 or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                y, x = q.popleft()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and grid[ny, nx] != 0:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            rs = [p[0] for p in cells]
            cs = [p[1] for p in cells]
            r0, r1 = min(rs), max(rs) + 1
            c0, c1 = min(cs), max(cs) + 1
            comps.append(grid[r0:r1, c0:c1].copy())
    return comps


def _solid_edge(arr, side):
    if side == "top":
        line = arr[0, :]
    elif side == "bottom":
        line = arr[-1, :]
    elif side == "left":
        line = arr[:, 0]
    else:
        line = arr[:, -1]
    if len(line) == 0 or np.any(line == 0):
        return None
    first = int(line[0])
    if np.all(line == first):
        return first, len(line)
    return None


def _matched_edges(pieces):
    sides = ("top", "bottom", "left", "right")
    opposite = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
    edge_info = {}
    for i, arr in enumerate(pieces):
        for side in sides:
            value = _solid_edge(arr, side)
            if value is not None:
                edge_info[(i, side)] = value

    compat = {edge: [] for edge in edge_info}
    for edge_a, (color_a, length_a) in edge_info.items():
        i, side = edge_a
        for edge_b, (color_b, length_b) in edge_info.items():
            j, other_side = edge_b
            if i == j or other_side != opposite[side]:
                continue
            if color_a == color_b and length_a == length_b:
                compat[edge_a].append(edge_b)

    matches = {}
    for edge, partners in compat.items():
        if len(partners) != 1:
            continue
        partner = partners[0]
        if len(compat.get(partner, [])) == 1 and compat[partner][0] == edge:
            matches[edge] = partner
    return matches


def _trim_piece(arr, matched_sides):
    top = 1 if "top" in matched_sides else 0
    bottom = arr.shape[0] - (1 if "bottom" in matched_sides else 0)
    left = 1 if "left" in matched_sides else 0
    right = arr.shape[1] - (1 if "right" in matched_sides else 0)
    return arr[top:bottom, left:right].copy()


def main(grid: np.ndarray) -> np.ndarray:
    """Assemble scattered jigsaw fragments by matching equal colored edge bands.

    Input structure:
    The input is a black background containing separated non-black rectangular
    fragments.  A fragment may have a one-pixel solid color band on its top,
    bottom, left, or right edge.  Matching connector bands appear on opposite
    sides of two different fragments with the same color and the same length.

    Rule:
    Detect every fragment, match each unambiguous pair of opposite edge bands,
    and place the fragments next to their matched neighbors.  Whenever two
    matched bands touch, remove both band rows or columns so the remaining
    fragment contents meet directly.  Return only the final assembled rectangle.

    Key insight:
    The solid edge colors are connector codes rather than picture content; the
    output is obtained by using those codes for adjacency and deleting the seam
    bands after the layout is recovered.
    """
    grid = np.asarray(grid)
    pieces = _components(grid)
    if not pieces:
        return np.zeros((0, 0), dtype=grid.dtype)

    matches = _matched_edges(pieces)
    sides_by_piece = [set() for _ in pieces]
    adj = {i: {} for i in range(len(pieces))}
    for (i, side), (j, other_side) in matches.items():
        sides_by_piece[i].add(side)
        sides_by_piece[j].add(other_side)
        adj[i][side] = j

    trimmed = [_trim_piece(arr, sides_by_piece[i]) for i, arr in enumerate(pieces)]
    pos = {0: (0, 0)}
    q = deque([0])
    while q:
        i = q.popleft()
        y, x = pos[i]
        th, tw = trimmed[i].shape
        for side, j in adj[i].items():
            if side == "right":
                np_pos = (y, x + tw)
            elif side == "left":
                np_pos = (y, x - trimmed[j].shape[1])
            elif side == "bottom":
                np_pos = (y + th, x)
            else:
                np_pos = (y - trimmed[j].shape[0], x)
            if j not in pos:
                pos[j] = np_pos
                q.append(j)

    if len(pos) != len(pieces):
        unused = [i for i in range(len(pieces)) if i not in pos]
        cursor_y = max(y + trimmed[i].shape[0] for i, (y, _) in pos.items()) if pos else 0
        for i in unused:
            pos[i] = (cursor_y, 0)
            cursor_y += trimmed[i].shape[0]

    min_y = min(y for y, _ in pos.values())
    min_x = min(x for _, x in pos.values())
    max_y = max(y + trimmed[i].shape[0] for i, (y, _) in pos.items())
    max_x = max(x + trimmed[i].shape[1] for i, (_, x) in pos.items())
    out = np.zeros((max_y - min_y, max_x - min_x), dtype=grid.dtype)
    for i, arr in enumerate(trimmed):
        y, x = pos[i]
        yy = y - min_y
        xx = x - min_x
        out[yy : yy + arr.shape[0], xx : xx + arr.shape[1]] = arr
    return out


def _checker(height, width, colors, phase):
    arr = np.zeros((height, width), dtype=int)
    for r in range(height):
        for c in range(width):
            arr[r, c] = colors[(r + c + phase) % len(colors)]
    return arr


def _make_horizontal(rng, count, colors):
    height = rng.randint(3, 7)
    inner_widths = [rng.randint(2, 6) for _ in range(count)]
    band_colors = colors[: count - 1]
    fill_colors = colors[count - 1 :]
    pieces = []
    for i in range(count):
        left_band = 1 if i > 0 else 0
        right_band = 1 if i < count - 1 else 0
        arr = _checker(height, inner_widths[i], fill_colors[i % len(fill_colors) : i % len(fill_colors) + 2] if i % len(fill_colors) + 2 <= len(fill_colors) else fill_colors[:2], rng.randint(0, 4))
        if arr.shape[1] < inner_widths[i]:
            arr = _checker(height, inner_widths[i], fill_colors[:2], rng.randint(0, 4))
        if left_band:
            arr = np.concatenate([np.full((height, 1), band_colors[i - 1], dtype=int), arr], axis=1)
        if right_band:
            arr = np.concatenate([arr, np.full((height, 1), band_colors[i], dtype=int)], axis=1)
        pieces.append(arr)
    return pieces


def _make_vertical(rng, count, colors):
    width = rng.randint(3, 7)
    inner_heights = [rng.randint(2, 6) for _ in range(count)]
    band_colors = colors[: count - 1]
    fill_colors = colors[count - 1 :]
    pieces = []
    for i in range(count):
        top_band = 1 if i > 0 else 0
        bottom_band = 1 if i < count - 1 else 0
        arr = _checker(inner_heights[i], width, fill_colors[:2], rng.randint(0, 4))
        if top_band:
            arr = np.concatenate([np.full((1, width), band_colors[i - 1], dtype=int), arr], axis=0)
        if bottom_band:
            arr = np.concatenate([arr, np.full((1, width), band_colors[i], dtype=int)], axis=0)
        pieces.append(arr)
    return pieces


def _fits(box, boxes, gap=1):
    r0, c0, r1, c1 = box
    for a0, b0, a1, b1 in boxes:
        if not (r1 + gap <= a0 or a1 + gap <= r0 or c1 + gap <= b0 or b1 + gap <= c0):
            return False
    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    count = rng.randint(2, 5)
    colors = list(range(1, 10))
    rng.shuffle(colors)
    orientation = rng.choice(["horizontal", "vertical"])
    if orientation == "horizontal":
        pieces = _make_horizontal(rng, count, colors)
    else:
        pieces = _make_vertical(rng, count, colors)

    for _ in range(200):
        rows = rng.randint(14, 30)
        cols = rng.randint(14, 30)
        grid = np.zeros((rows, cols), dtype=int)
        boxes = []
        order = list(range(len(pieces)))
        rng.shuffle(order)
        ok = True
        for idx in order:
            piece = pieces[idx]
            ph, pw = piece.shape
            if ph + 2 > rows or pw + 2 > cols:
                ok = False
                break
            placed = False
            for _ in range(300):
                r = rng.randint(1, rows - ph - 1)
                c = rng.randint(1, cols - pw - 1)
                box = (r, c, r + ph, c + pw)
                if _fits(box, boxes):
                    grid[r : r + ph, c : c + pw] = piece
                    boxes.append(box)
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if ok:
            return grid

    rows, cols = 30, 30
    grid = np.zeros((rows, cols), dtype=int)
    r = 1
    c = 1
    row_height = 0
    for piece in pieces:
        ph, pw = piece.shape
        if c + pw + 1 > cols:
            r += row_height + 2
            c = 1
            row_height = 0
        grid[r : r + ph, c : c + pw] = piece
        c += pw + 2
        row_height = max(row_height, ph)
    return grid
