import random

import numpy as np


BG = 0
DIV = 5


def _neighbors(cell):
    r, c = cell
    return [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy each left-panel path to the matching right-panel marker by replaying its horizontally mirrored step sequence.

    Input structure:
    - A vertical grey divider splits the grid into left and right panels on a black background.
    - In the left panel, a same-color path starts from a 2x1 edge marker and ends at another border.
    - In the right panel, the same color appears only as an isolated 2x1 edge marker.

    Rule:
    - For each color, identify the left path's 2x1 start marker and trace the path from its anchor cell through the interior to its endpoint.
    - Mirror each path step horizontally, which is equivalent to keeping vertical moves and reversing horizontal moves.
    - Start from the corresponding anchor cell of the right-panel 2x1 marker and draw the mirrored step sequence there.
    - Leave the original left panel unchanged.

    Key insight:
    - The right output is not a raw translation of pixels; it is a replay of the left path's abstract step program under horizontal reflection.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    divider = int(np.where(np.all(g == DIV, axis=0))[0][0])
    left = g[:, :divider]
    right = g[:, divider + 1 :]
    colors = [int(v) for v in np.unique(g) if v not in (BG, DIV)]

    for color in colors:
        left_cells = [tuple(x) for x in np.argwhere(left == color)]
        right_cells = [tuple(x) for x in np.argwhere(right == color)]
        if not left_cells or len(right_cells) != 2:
            continue
        marker_pair = sorted([cell for cell in left_cells if cell[1] == 0])
        if len(marker_pair) != 2:
            continue
        anchor = next(cell for cell in marker_pair if any(n in left_cells and n[1] > 0 for n in _neighbors(cell)))
        comp_set = set(left_cells)
        stub = next(cell for cell in marker_pair if cell != anchor)
        ordered = [anchor]
        prev, cur = stub, anchor
        while True:
            nxts = [n for n in _neighbors(cur) if n in comp_set and n != prev]
            if not nxts:
                break
            nxt = nxts[0]
            ordered.append(nxt)
            prev, cur = cur, nxt
            if cur[1] in (0, left.shape[1] - 1) or cur[0] in (0, left.shape[0] - 1):
                if cur not in marker_pair:
                    break
        steps = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(ordered, ordered[1:])]
        mirrored = [(dr, -dc) for dr, dc in steps]

        right_pair = sorted(right_cells)
        anchor_index = marker_pair.index(anchor)
        r_anchor = right_pair[anchor_index]
        rr, cc = r_anchor
        for dr, dc in mirrored:
            rr += dr
            cc += dc
            if 0 <= rr < right.shape[0] and 0 <= cc < right.shape[1]:
                out[rr, divider + 1 + cc] = color
    return out


def generate_input() -> np.ndarray:
    for _ in range(300):
        h = random.randint(14, 22)
        half_w = random.randint(8, 12)
        w = half_w * 2 + 1
        if w > 30:
            continue
        grid = np.full((h, w), BG, dtype=int)
        divider = half_w
        grid[:, divider] = DIV
        color = random.choice([1, 2, 3, 4, 6, 7, 8, 9])
        sr = random.randint(2, h - 6)
        # left marker
        grid[sr : sr + 2, 0] = color
        path = [(sr, 0), (sr, 1)]
        r, c = sr, 1
        segs = [
            (0, 1, random.randint(2, half_w - 4)),
            (random.choice([-1, 1]), 0, random.randint(2, h // 3)),
            (0, 1, random.randint(1, half_w - c - 3)),
        ]
        for dr, dc, steps in segs:
            for _ in range(steps):
                nr, nc = r + dr, c + dc
                if not (0 < nr < h - 1 and 0 < nc < half_w - 1):
                    break
                if (nr, nc) in path:
                    break
                path.append((nr, nc))
                r, c = nr, nc
        end_dir = -1 if r > h // 2 else 1
        while 0 < r < h - 1:
            nr = r + end_dir
            if (nr, c) in path:
                break
            path.append((nr, c))
            r = nr
            if r in (0, h - 1):
                break
        for pr, pc in path:
            grid[pr, pc] = color
        # right marker at mirrored rows on right edge
        grid[sr : sr + 2, -1] = color
        return grid
    raise RuntimeError("failed to generate input")
