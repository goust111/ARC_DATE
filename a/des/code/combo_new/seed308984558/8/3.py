import random
from collections import deque

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
GREY = 5
STAMP_COLOR = 4


def _rotate_patch(patch: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(patch, -turns % 4)


def _neighbors(r, c):
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        yield r + dr, c + dc


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the framed motif onto green path markers while turns accumulate rotation.

    Input structure:
    - A grey bordered rectangle contains a small non-symmetric colored motif.
    - A red 4-connected orthogonal path starts adjacent to one frame side; some path
      cells are replaced by green marker cells.
    - The rest of the grid is black.

    Rule:
    - Extract the motif from inside the grey frame.
    - Traverse the path from the endpoint touching the frame outward.
    - Every 90-degree turn increments the motif orientation by one clockwise quarter-
      turn cumulatively.
    - Whenever a green marker is encountered, stamp the current motif centered on that
      marker onto a black canvas; later stamps overwrite earlier ones.
    - The output removes the frame and the path, keeping only stamped motifs.

    Key insight:
    - The frame contains a reusable stamp, green cells are the trigger points, and the
      path's turning history controls a cumulative rotation state rather than a local
      per-marker orientation.
    """
    h, w = grid.shape
    grey = np.argwhere(grid == GREY)
    r0, c0 = grey.min(axis=0)
    r1, c1 = grey.max(axis=0)
    motif = grid[r0 + 1 : r1, c0 + 1 : c1].copy()
    motif[motif == GREY] = BLACK
    path_mask = np.isin(grid, [RED, GREEN]).copy()
    starts = []
    for r, c in np.argwhere(path_mask):
        deg = sum(0 <= nr < h and 0 <= nc < w and path_mask[nr, nc] for nr, nc in _neighbors(r, c))
        if deg == 1:
            if any(r0 <= nr <= r1 and c0 <= nc <= c1 for nr, nc in _neighbors(r, c)):
                starts.insert(0, (r, c))
            else:
                starts.append((r, c))
    if starts:
        start = starts[0]
    else:
        near = []
        for r, c in np.argwhere(path_mask):
            if any(r0 <= nr <= r1 and c0 <= nc <= c1 for nr, nc in _neighbors(r, c)):
                near.append((r, c))
        start = near[0]
    order = [start]
    prev = None
    cur = start
    while True:
        nxts = []
        for nr, nc in _neighbors(*cur):
            if 0 <= nr < h and 0 <= nc < w and path_mask[nr, nc] and (nr, nc) != prev:
                nxts.append((nr, nc))
        if not nxts:
            break
        nxt = nxts[0]
        order.append(nxt)
        prev, cur = cur, nxt

    out = np.zeros_like(grid)
    turns = 0
    prev_dir = None
    mh, mw = motif.shape
    cr0, cc0 = mh // 2, mw // 2
    for i, (r, c) in enumerate(order):
        if i:
            dr = r - order[i - 1][0]
            dc = c - order[i - 1][1]
            cur_dir = (dr, dc)
            if prev_dir is not None and cur_dir != prev_dir:
                turns += 1
            prev_dir = cur_dir
        if grid[r, c] != GREEN:
            continue
        patch = _rotate_patch(motif, turns)
        for pr in range(patch.shape[0]):
            for pc in range(patch.shape[1]):
                color = int(patch[pr, pc])
                if color == BLACK:
                    continue
                rr = r + pr - patch.shape[0] // 2
                cc = c + pc - patch.shape[1] // 2
                if 0 <= rr < h and 0 <= cc < w:
                    out[rr, cc] = color
    return out


def generate_input() -> np.ndarray:
    while True:
        h = random.randint(18, 26)
        w = random.randint(18, 26)
        grid = np.zeros((h, w), dtype=int)
        frame_r = random.randint(2, 5)
        frame_c = random.randint(2, 5)
        grid[frame_r : frame_r + 5, frame_c : frame_c + 5] = GREY
        grid[frame_r + 1 : frame_r + 4, frame_c + 1 : frame_c + 4] = BLACK
        motif = np.array([[STAMP_COLOR, 0, 0], [STAMP_COLOR, STAMP_COLOR, 0], [0, STAMP_COLOR, 0]])
        if random.random() < 0.5:
            motif = np.fliplr(motif)
        grid[frame_r + 1 : frame_r + 4, frame_c + 1 : frame_c + 4] = motif
        side = random.choice(["right", "left", "top", "bottom"])
        if side == "right":
            start = (frame_r + 2, frame_c + 5)
            direction = (0, 1)
        elif side == "left":
            start = (frame_r + 2, frame_c - 1)
            direction = (0, -1)
        elif side == "top":
            start = (frame_r - 1, frame_c + 2)
            direction = (-1, 0)
        else:
            start = (frame_r + 5, frame_c + 2)
            direction = (1, 0)
        if not (0 <= start[0] < h and 0 <= start[1] < w):
            continue
        path = [start]
        cur = start
        cur_dir = direction
        segments = random.randint(3, 5)
        for _ in range(segments):
            step_count = random.randint(2, 4)
            ok = True
            for _ in range(step_count):
                nr, nc = cur[0] + cur_dir[0], cur[1] + cur_dir[1]
                if not (1 <= nr < h - 1 and 1 <= nc < w - 1) or (nr, nc) in path:
                    ok = False
                    break
                if any((nr, nc) != cur and abs(nr - pr) + abs(nc - pc) == 1 for pr, pc in path[:-1]):
                    ok = False
                    break
                cur = (nr, nc)
                path.append(cur)
            if not ok:
                break
            cur_dir = (cur_dir[1], -cur_dir[0])
            if random.random() < 0.5:
                cur_dir = (-cur_dir[0], -cur_dir[1])
        if len(path) < 8:
            continue
        for r, c in path:
            grid[r, c] = RED
        markers = random.sample(path[2:-1], random.randint(2, min(4, len(path) - 3)))
        for r, c in markers:
            grid[r, c] = GREEN
        return grid
