import random
from itertools import permutations

import numpy as np


AZURE = 8
MAROON = 9


def _neighbors(r, c):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask):
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(cells)
    return comps


def _bbox(cells):
    rows = [p[0] for p in cells]
    cols = [p[1] for p in cells]
    return min(rows), max(rows), min(cols), max(cols)


def _norm(cells):
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _find_frames(grid):
    frames = []
    for cells in _components(grid == MAROON):
        r0, r1, c0, c1 = _bbox(cells)
        if r1 - r0 >= 3 and c1 - c0 >= 3:
            frames.append({"cells": set(cells), "bbox": (r0, r1, c0, c1)})
    frames.sort(key=lambda f: (f["bbox"][0], f["bbox"][2]))
    return frames


def _find_pieces(grid):
    pieces = []
    for color in sorted(int(x) for x in np.unique(grid) if x not in (AZURE, MAROON)):
        for cells in _components(grid == color):
            pieces.append(
                {
                    "color": color,
                    "cells": cells,
                    "shape": _norm(cells),
                }
            )
    pieces.sort(key=lambda p: (min(r for r, _ in p["cells"]), min(c for _, c in p["cells"])))
    return pieces


def _best_for_frame(grid, frame, shape):
    h, w = grid.shape
    r0, r1, c0, c1 = frame["bbox"]
    max_sr = max(r for r, _ in shape)
    max_sc = max(c for _, c in shape)
    best = None
    for ar in range(r0 + 1, r1 - max_sr + 1):
        for ac in range(c0 + 1, c1 - max_sc + 1):
            placed = [(ar + r, ac + c) for r, c in shape]
            if any(grid[rr, cc] != AZURE for rr, cc in placed):
                continue
            contacts = 0
            for rr, cc in placed:
                for nr, nc in _neighbors(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == MAROON:
                        contacts += 1
            if contacts == 0:
                continue
            top_contact = sum(1 for rr, cc in placed if rr > 0 and grid[rr - 1, cc] == MAROON)
            bottom_contact = sum(1 for rr, cc in placed if rr + 1 < h and grid[rr + 1, cc] == MAROON)
            left_contact = sum(1 for rr, cc in placed if cc > 0 and grid[rr, cc - 1] == MAROON)
            right_contact = sum(1 for rr, cc in placed if cc + 1 < w and grid[rr, cc + 1] == MAROON)
            side_fit = max(top_contact, bottom_contact, left_contact, right_contact)
            edge_cells = len(
                {
                    (rr, cc)
                    for rr, cc in placed
                    if any(
                        0 <= nr < h and 0 <= nc < w and grid[nr, nc] == MAROON
                        for nr, nc in _neighbors(rr, cc)
                    )
                }
            )
            score = contacts * 100 + side_fit * 10 + edge_cells
            candidate = (score, ar, ac, placed)
            if best is None or candidate[:3] > best[:3]:
                best = candidate
    return best


def _choose_moves(grid, pieces, frames):
    if not pieces or not frames:
        return {}
    options = []
    for piece in pieces:
        row = []
        for frame in frames:
            row.append(_best_for_frame(grid, frame, piece["shape"]))
        options.append(row)

    n = min(len(pieces), len(frames))
    best_assignment = None
    for frame_order in permutations(range(len(frames)), n):
        total = 0
        placements = {}
        ok = True
        for pi, fi in enumerate(frame_order):
            opt = options[pi][fi]
            if opt is None:
                ok = False
                break
            total += opt[0]
            placements[pi] = (fi, opt)
        if ok:
            key = (total, tuple(-x for x in frame_order))
            if best_assignment is None or key > best_assignment[0]:
                best_assignment = (key, placements)
    return {} if best_assignment is None else best_assignment[1]


def main(grid: np.ndarray) -> np.ndarray:
    """Slide each colored shard into the maroon frame whose inner sawteeth match its outline.

    Input structure:
    The grid has an azure background, several separate maroon rectangular frames with inward
    jagged wall pixels, and scattered 4-connected colored shards outside the frames. Each shard
    has one unique color, while frame interiors are otherwise azure.

    Rule:
    Treat every non-azure, non-maroon component as a movable shard. For each shard, test
    translations into each maroon frame interior without rotation or overlap. The valid placement
    whose shard boundary has the strongest maroon wall contact is the complementary socket.
    Move each shard to its uniquely matching frame and clear its original cells to azure.

    Key insight:
    The jagged maroon wall is a visible spatial key: the correct destination is determined by
    shape-to-wall complementarity, not by shard color, size ordering, or position.
    """
    grid = np.array(grid, copy=True)
    out = np.array(grid, copy=True)
    frames = _find_frames(grid)
    pieces = _find_pieces(grid)
    moves = _choose_moves(grid, pieces, frames)

    for piece in pieces:
        for r, c in piece["cells"]:
            out[r, c] = AZURE
    for pi, (_, opt) in moves.items():
        color = pieces[pi]["color"]
        for r, c in opt[3]:
            out[r, c] = color
    return out


def _profile(length, max_depth):
    vals = [random.randint(0, max_depth) for _ in range(length)]
    if 0 not in vals:
        vals[random.randrange(length)] = 0
    if max_depth not in vals:
        vals[random.randrange(length)] = max_depth
    return vals


def _make_frame():
    orient = random.choice(("top", "bottom", "left", "right"))
    max_depth = random.choice((2, 3))
    length = random.randint(4, 6)
    if orient in ("top", "bottom"):
        width = random.randint(max(8, length + 4), 11)
        height = random.randint(max_depth + 6, 11)
        span = width - 2
    else:
        height = random.randint(max(8, length + 4), 11)
        width = random.randint(max_depth + 6, 11)
        span = height - 2

    start = random.randint(1, max(1, span - length - 1))
    depths = _profile(length, max_depth)
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True

    piece = set()
    if orient == "top":
        for i, d in enumerate(depths):
            col = 1 + start + i
            if d:
                mask[1 : 1 + d, col] = True
            for rr in range(d, max_depth + 1):
                piece.add((rr, i))
        anchor = (1, 1 + start)
    elif orient == "bottom":
        for i, d in enumerate(depths):
            col = 1 + start + i
            if d:
                mask[height - 1 - d : height - 1, col] = True
            for rr in range(0, max_depth - d + 1):
                piece.add((rr, i))
        anchor = (height - 2 - max_depth, 1 + start)
    elif orient == "left":
        for i, d in enumerate(depths):
            row = 1 + start + i
            if d:
                mask[row, 1 : 1 + d] = True
            for cc in range(d, max_depth + 1):
                piece.add((i, cc))
        anchor = (1 + start, 1)
    else:
        for i, d in enumerate(depths):
            row = 1 + start + i
            if d:
                mask[row, width - 1 - d : width - 1] = True
            for cc in range(0, max_depth - d + 1):
                piece.add((i, cc))
        anchor = (1 + start, width - 2 - max_depth)

    return mask, _norm(piece), anchor


def _can_reserve(reserved, shape, top, left, margin=1):
    h, w = reserved.shape
    cells = [(top + r, left + c) for r, c in shape]
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        r0 = max(0, r - margin)
        r1 = min(h, r + margin + 1)
        c0 = max(0, c - margin)
        c1 = min(w, c + margin + 1)
        if reserved[r0:r1, c0:c1].any():
            return False
    return True


def _reserve(reserved, shape, top, left, margin=1):
    h, w = reserved.shape
    for r, c in shape:
        rr, cc = top + r, left + c
        r0 = max(0, rr - margin)
        r1 = min(h, rr + margin + 1)
        c0 = max(0, cc - margin)
        c1 = min(w, cc + margin + 1)
        reserved[r0:r1, c0:c1] = True


def _place_random(reserved, shape):
    h, w = reserved.shape
    max_r = max(r for r, _ in shape)
    max_c = max(c for _, c in shape)
    for _ in range(300):
        top = random.randint(0, h - max_r - 1)
        left = random.randint(0, w - max_c - 1)
        if _can_reserve(reserved, shape, top, left):
            _reserve(reserved, shape, top, left)
            return top, left
    return None


def generate_input() -> np.ndarray:
    rows = random.randint(10, 30)
    cols = random.randint(10, 30)
    return np.full((rows, cols), AZURE, dtype=int)
