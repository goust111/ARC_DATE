import random

import numpy as np


BG = 0
MARK = 2
FRAME = 5


STAMP_MASK = np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=int)


def _neighbors(cell):
    r, c = cell
    return [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]


def _trace_path(path_cells, start, end):
    path_set = set(path_cells)
    ordered = [start]
    prev = None
    cur = start
    while cur != end:
        nxts = [n for n in _neighbors(cur) if n in path_set and n != prev]
        nxt = nxts[0]
        ordered.append(nxt)
        prev, cur = cur, nxt
    return ordered


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow each path from its stamp to its target frame, rotating the carried stamp only at marked turns, then stamp the final orientation into the frame.

    Input structure:
    - The background is black.
    - A monochrome path connects a colored 3x3 stamp pattern to a grey hollow 3x3 frame.
    - Some path corners have a red marker on the outer diagonal of the turn.

    Rule:
    - Start from the stamp-adjacent end of the path and walk to the frame-adjacent end.
    - Every time the path changes direction, inspect the outer diagonal cell of that bend.
    - If that diagonal contains a red marker, rotate the carried stamp clockwise by 90 degrees; otherwise keep the orientation.
    - Replace the grey frame with the final rotated stamp pattern, leaving the original stamp and path intact.

    Key insight:
    - Turning alone does nothing; only turns with the extra outer-diagonal marker contribute to the cumulative stamp rotation.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    frame_pos = np.argwhere(g == FRAME)
    fr0, fc0 = frame_pos.min(axis=0)
    fr1, fc1 = frame_pos.max(axis=0)
    frame_cells = {(r, c) for r in range(fr0, fr1 + 1) for c in range(fc0, fc1 + 1) if g[r, c] == FRAME}
    colors = [
        int(v)
        for v in np.unique(g)
        if v not in (BG, MARK, FRAME)
        and any(any(n in frame_cells for n in _neighbors(tuple(cell))) for cell in np.argwhere(g == v))
    ]
    for path_color in colors:
        path_cells = [(r, c) for r, c in np.argwhere(g == path_color)]
        if not path_cells:
            continue
        stamp_cells = []
        for r in range(g.shape[0] - 2):
            for c in range(g.shape[1] - 2):
                sub = g[r : r + 3, c : c + 3]
                vals = {int(v) for v in np.unique(sub) if v != BG}
                if len(vals) == 1 and path_color not in vals and FRAME not in vals and MARK not in vals:
                    if any(
                        0 <= nr < g.shape[0]
                        and 0 <= nc < g.shape[1]
                        and g[nr, nc] == path_color
                        for rr in range(r, r + 3)
                        for cc in range(c, c + 3)
                        for nr, nc in _neighbors((rr, cc))
                    ):
                        stamp_cells = [(rr, cc) for rr in range(r, r + 3) for cc in range(c, c + 3) if g[rr, cc] != BG]
                        stamp_bbox = (r, c)
        if not stamp_cells:
            continue
        start = next(cell for cell in path_cells if any(n in stamp_cells for n in _neighbors(cell)))
        end = next(cell for cell in path_cells if any(n in frame_cells for n in _neighbors(cell)))
        ordered = _trace_path(path_cells, start, end)
        turns = 0
        for i in range(1, len(ordered) - 1):
            pr, pc = ordered[i - 1]
            cr, cc = ordered[i]
            nr, nc = ordered[i + 1]
            d1 = (cr - pr, cc - pc)
            d2 = (nr - cr, nc - cc)
            if d1 != d2:
                mr, mc = cr + d1[0] + d2[0], cc + d1[1] + d2[1]
                if 0 <= mr < g.shape[0] and 0 <= mc < g.shape[1] and g[mr, mc] == MARK:
                    turns += 1
        sr, sc = stamp_bbox
        stamp = g[sr : sr + 3, sc : sc + 3].copy()
        rotated = np.rot90(stamp, -turns % 4)
        out[fr0 : fr0 + 3, fc0 : fc0 + 3] = rotated
    return out


def _draw_stamp(grid, top, left, color):
    for r in range(3):
        for c in range(3):
            if STAMP_MASK[r, c]:
                grid[top + r, left + c] = color


def _draw_frame(grid, top, left):
    grid[top, left : left + 3] = FRAME
    grid[top + 2, left : left + 3] = FRAME
    grid[top : top + 3, left] = FRAME
    grid[top : top + 3, left + 2] = FRAME
    grid[top + 1, left + 1] = BG


def generate_input() -> np.ndarray:
    for _ in range(300):
        h, w = random.randint(16, 24), random.randint(16, 24)
        grid = np.full((h, w), BG, dtype=int)
        path_color = random.choice([1, 3, 4, 6, 7, 8, 9])
        stamp_color = random.choice([c for c in [1, 3, 4, 6, 7, 8, 9] if c != path_color])
        sr, sc = random.randint(2, h - 5), random.randint(1, 4)
        fr, fc = random.randint(2, h - 5), random.randint(w - 7, w - 4)
        _draw_stamp(grid, sr, sc, stamp_color)
        _draw_frame(grid, fr, fc)
        start = (sr + 2, sc + 3)
        end = (fr + 1, fc - 1)
        mid1 = (start[0], random.randint(start[1] + 2, max(start[1] + 2, end[1] - 2)))
        mid2 = (end[0], mid1[1])
        points = [start, mid1, mid2, end]
        for (r0, c0), (r1, c1) in zip(points, points[1:]):
            if r0 == r1:
                for c in range(min(c0, c1), max(c0, c1) + 1):
                    grid[r0, c] = path_color
            else:
                for r in range(min(r0, r1), max(r0, r1) + 1):
                    grid[r, c0] = path_color
        for i in range(1, len(points) - 1):
            pr, pc = points[i - 1]
            cr, cc = points[i]
            nr, nc = points[i + 1]
            d1 = (np.sign(cr - pr), np.sign(cc - pc))
            d2 = (np.sign(nr - cr), np.sign(nc - cc))
            if d1 != d2 and random.choice([True, False]):
                mr, mc = cr + d1[0] + d2[0], cc + d1[1] + d2[1]
                if 0 <= mr < h and 0 <= mc < w and grid[mr, mc] == BG:
                    grid[mr, mc] = MARK
        return grid
    raise RuntimeError("failed to generate input")
