import random
from collections import deque

import numpy as np


PIECE_COLORS = (1, 3, 4, 8)
MARKER_COLORS = (2, 5, 6, 7)


def _neighbors4(r, c, h, w):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(cells)
    return parts


def _reflect_piece(piece_cells, marker, adjacent):
    mr, mc = marker
    pr, pc = adjacent
    reflected = []
    if mr == pr:
        s = mc + pc
        for r, c in piece_cells:
            reflected.append((r, s - c))
    else:
        s = mr + pr
        for r, c in piece_cells:
            reflected.append((s - r, c))
    return reflected


def _component_to_rect(grid, cells):
    marker = next((r, c) for r, c in cells if grid[r, c] in MARKER_COLORS)
    piece_cells = [(r, c) for r, c in cells if grid[r, c] in PIECE_COLORS]
    piece_color = grid[piece_cells[0]]
    marker_color = int(grid[marker])
    adjacent = None
    for nr, nc in _neighbors4(marker[0], marker[1], grid.shape[0], grid.shape[1]):
        if (nr, nc) in set(piece_cells):
            adjacent = (nr, nc)
            break
    reflected = _reflect_piece(piece_cells, marker, adjacent)
    all_cells = piece_cells + reflected
    rows = [r for r, _ in all_cells]
    cols = [c for _, c in all_cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    rect = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in piece_cells:
        rect[r - r0, c - c0] = piece_color
    for r, c in reflected:
        rect[r - r0, c - c0] = marker_color
    prow = [r for r, _ in piece_cells]
    pcol = [c for _, c in piece_cells]
    pbbox = (min(pcol), min(prow))
    return pbbox, rect


def main(grid: np.ndarray) -> np.ndarray:
    """
    Summary:
    Use each singleton marker to define a mirror axis, reflect the adjacent fragment, and pack the original-reflection pairs into rectangles.

    Input Model:
    Every fragment is a 4-connected single-color piece from the foreground palette with one adjacent marker pixel from a separate palette that indicates both axis and reflected color.

    Algorithm:
    For each connected component, separate the marker from the fragment, reflect all fragment cells across the midpoint line between the marker and its adjacent fragment cell, rasterize the resulting solid rectangle, then order all rectangles by the original fragment position.

    Output:
    Return the packed row of reflected rectangles, top-aligned and separated by one black column, with all marker pixels removed.
    """
    grid = np.asarray(grid, dtype=int)
    rects = [_component_to_rect(grid, cells) for cells in _components(grid != 0)]
    rects.sort(key=lambda item: (item[0][0], item[0][1]))
    width = sum(rect.shape[1] for _, rect in rects) + max(0, len(rects) - 1)
    height = max(rect.shape[0] for _, rect in rects)
    out = np.zeros((height, width), dtype=int)
    c = 0
    for _, rect in rects:
        out[: rect.shape[0], c : c + rect.shape[1]] = rect
        c += rect.shape[1] + 1
    return out


def _generate_case(rng: random.Random):
    h = rng.randint(14, 22)
    w = rng.randint(16, 24)
    grid = np.zeros((h, w), dtype=int)
    n = rng.randint(2, 4)
    piece_colors = rng.sample(PIECE_COLORS, n)
    marker_colors = rng.sample(MARKER_COLORS, n)
    expected_parts = []
    placements = []
    for piece_color, marker_color in zip(piece_colors, marker_colors):
        vertical = rng.choice((True, False))
        options = [(1, 3), (2, 2), (2, 3), (3, 1)]
        ph, pw = rng.choice(options)
        if ph * pw < 3 or ph * pw > 7:
            ph, pw = 2, 2
        if vertical:
            piece = np.full((ph, pw), piece_color, dtype=int)
            marker = np.zeros((ph, pw + 1), dtype=int)
            marker[:, :pw] = piece
            marker[ph // 2, pw] = marker_color
            rect = np.zeros((ph, pw * 2), dtype=int)
            rect[:, :pw] = piece_color
            rect[:, pw:] = marker_color
        else:
            piece = np.full((ph, pw), piece_color, dtype=int)
            marker = np.zeros((ph + 1, pw), dtype=int)
            marker[:ph, :] = piece
            marker[ph, pw // 2] = marker_color
            rect = np.zeros((ph * 2, pw), dtype=int)
            rect[:ph, :] = piece_color
            rect[ph:, :] = marker_color
        placements.append(marker)
        expected_parts.append(rect)
    order = []
    for idx, shape in enumerate(placements):
        sh, sw = shape.shape
        for _ in range(500):
            r0 = rng.randint(0, h - sh)
            c0 = rng.randint(0, w - sw)
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(h, r0 + sh + 1)
            cc1 = min(w, c0 + sw + 1)
            if np.any(grid[rr0:rr1, cc0:cc1] != 0):
                continue
            grid[r0 : r0 + sh, c0 : c0 + sw] = shape
            order.append((idx, c0, r0))
            break
    order.sort(key=lambda item: (item[1], item[2]))
    sorted_parts = [expected_parts[idx] for idx, _, _ in order]
    out_w = sum(part.shape[1] for part in sorted_parts) + len(sorted_parts) - 1
    out_h = max(part.shape[0] for part in sorted_parts)
    expected = np.zeros((out_h, out_w), dtype=int)
    c = 0
    for part in sorted_parts:
        expected[: part.shape[0], c : c + part.shape[1]] = part
        c += part.shape[1] + 1
    return grid, expected


def generate_input() -> np.ndarray:
    return _generate_case(random.Random(random.randrange(1 << 30)))[0]
