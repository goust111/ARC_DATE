import random
from collections import deque

import numpy as np

BLACK = 0
GREEN = 3
YELLOW = 4


_DIRS = {
    "right": (0, 1),
    "left": (0, -1),
    "down": (1, 0),
    "up": (-1, 0),
}


def _arrow_offsets(direction, size):
    if direction == "right":
        pts = {(r, c) for r in range(size) for c in range(size)}
        pts.update((r, size) for r in range(size))
        mid = size // 2
        for k in range(2, size + 1):
            pts.add((mid, size + k - 1))
        return pts
    if direction == "left":
        width = 2 * size
        right = width - 1
        pts = {(r, right - c) for r in range(size) for c in range(size)}
        pts.update((r, size - 1) for r in range(size))
        mid = size // 2
        for k in range(2, size + 1):
            pts.add((mid, size - k))
        return pts
    if direction == "down":
        pts = {(r, c) for r in range(size) for c in range(size)}
        pts.update((size, c) for c in range(size))
        mid = size // 2
        for k in range(2, size + 1):
            pts.add((size + k - 1, mid))
        return pts
    if direction == "up":
        height = 2 * size
        bottom = height - 1
        pts = {(bottom - r, c) for r in range(size) for c in range(size)}
        pts.update((size - 1, c) for c in range(size))
        mid = size // 2
        for k in range(2, size + 1):
            pts.add((size - k, mid))
        return pts
    raise ValueError(direction)


_ARROW_TEMPLATES = []
for _size in (3, 5):
    for _direction in _DIRS:
        _pts = _arrow_offsets(_direction, _size)
        _min_r = min(r for r, _ in _pts)
        _min_c = min(c for _, c in _pts)
        _norm = frozenset((r - _min_r, c - _min_c) for r, c in _pts)
        _height = max(r for r, _ in _norm) + 1
        _width = max(c for _, c in _norm) + 1
        _ARROW_TEMPLATES.append((_norm, _direction, _size, _height, _width))


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
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


def _identify_arrow(comp):
    min_r = min(r for r, _ in comp)
    min_c = min(c for _, c in comp)
    norm = frozenset((r - min_r, c - min_c) for r, c in comp)
    for pts, direction, size, _height, _width in _ARROW_TEMPLATES:
        if norm == pts:
            return {
                "direction": direction,
                "size": size,
                "step": 1 if size == 3 else 2,
                "cells": set(comp),
                "tip": _tip_cell(comp, direction),
            }
    return None


def _tip_cell(cells, direction):
    if direction == "right":
        max_c = max(c for _, c in cells)
        rows = [r for r, c in cells if c == max_c]
        return rows[0], max_c
    if direction == "left":
        min_c = min(c for _, c in cells)
        rows = [r for r, c in cells if c == min_c]
        return rows[0], min_c
    if direction == "down":
        max_r = max(r for r, _ in cells)
        cols = [c for r, c in cells if r == max_r]
        return max_r, cols[0]
    min_r = min(r for r, _ in cells)
    cols = [c for r, c in cells if r == min_r]
    return min_r, cols[0]


def _block_key(block, direction, tip):
    rows = [r for r, _ in block]
    cols = [c for _, c in block]
    if direction == "right" and min(cols) > tip[1] and min(rows) <= tip[0] <= max(rows):
        return min(cols) - tip[1]
    if direction == "left" and max(cols) < tip[1] and min(rows) <= tip[0] <= max(rows):
        return tip[1] - max(cols)
    if direction == "down" and min(rows) > tip[0] and min(cols) <= tip[1] <= max(cols):
        return min(rows) - tip[0]
    if direction == "up" and max(rows) < tip[0] and min(cols) <= tip[1] <= max(cols):
        return tip[0] - max(rows)
    return None


def _target_block(arrow, yellow_blocks, used):
    best = None
    best_dist = None
    for i, block in enumerate(yellow_blocks):
        if i in used:
            continue
        dist = _block_key(block, arrow["direction"], arrow["tip"])
        if dist is not None and (best_dist is None or dist < best_dist):
            best = i
            best_dist = dist
    return best


def _slide_block(block, dr, dc, requested, arrow_cells, shape):
    h, w = shape
    block_set = set(block)
    distance = 0
    for step in range(1, requested + 1):
        shifted = {(r + dr * step, c + dc * step) for r, c in block_set}
        if any(r < 0 or r >= h or c < 0 or c >= w for r, c in shifted):
            break
        if shifted & arrow_cells:
            break
        distance = step
    return {(r + dr * distance, c + dc * distance) for r, c in block_set}


def main(grid: np.ndarray) -> np.ndarray:
    """Move each yellow block in the direction and distance indicated by its green arrow.

    Input structure:
    The grid has a black background, several green arrow components, and several yellow
    rectangular blocks of size 1x1 through 2x2.  Each arrow is either a small arrow made
    from a 3x3 body and triangular tip or a large arrow made from a 5x5 body and a larger
    matching tip.  Every yellow block lies unambiguously in the row or column pointed to
    by one arrow.

    Rule:
    Arrows remain fixed.  For each arrow, locate the yellow block directly ahead in the
    same row or column.  A small arrow pushes that block one cell and a large arrow pushes
    it two cells along the arrow direction.  If the requested movement would hit the grid
    edge or a green arrow, stop the block immediately before that obstacle.  The block's
    old cells become black and its new cells are painted yellow.

    Key insight:
    The arrow's orientation supplies the movement direction, while the arrow body's size
    independently supplies the movement distance.
    """
    out = np.array(grid, copy=True)
    arrows = []
    for comp in _components(grid == GREEN):
        arrow = _identify_arrow(comp)
        if arrow is not None:
            arrows.append(arrow)

    yellow_blocks = [set(comp) for comp in _components(grid == YELLOW)]
    arrow_cells = set()
    for arrow in arrows:
        arrow_cells.update(arrow["cells"])

    moves = []
    used = set()
    for arrow in arrows:
        idx = _target_block(arrow, yellow_blocks, used)
        if idx is None:
            continue
        used.add(idx)
        dr, dc = _DIRS[arrow["direction"]]
        new_block = _slide_block(yellow_blocks[idx], dr, dc, arrow["step"], arrow_cells, grid.shape)
        moves.append((yellow_blocks[idx], new_block))

    for old_block, _new_block in moves:
        for r, c in old_block:
            out[r, c] = BLACK
    for _old_block, new_block in moves:
        for r, c in new_block:
            out[r, c] = YELLOW
    return out


def _place(cells, top, left):
    return {(top + r, left + c) for r, c in cells}


def _fits(cells, h, w, occupied):
    return all(0 <= r < h and 0 <= c < w and (r, c) not in occupied for r, c in cells)


def _line_clear_between(a_cells, b_cells, direction, occupied):
    a_tip = _tip_cell(a_cells, direction)
    brs = [r for r, _ in b_cells]
    bcs = [c for _, c in b_cells]
    forbidden = occupied - a_cells - b_cells
    if direction == "right":
        r = a_tip[0]
        for c in range(a_tip[1] + 1, min(bcs)):
            if (r, c) in forbidden:
                return False
    elif direction == "left":
        r = a_tip[0]
        for c in range(max(bcs) + 1, a_tip[1]):
            if (r, c) in forbidden:
                return False
    elif direction == "down":
        c = a_tip[1]
        for r in range(a_tip[0] + 1, min(brs)):
            if (r, c) in forbidden:
                return False
    else:
        c = a_tip[1]
        for r in range(max(brs) + 1, a_tip[0]):
            if (r, c) in forbidden:
                return False
    return True


def _candidate_positions(h, w, direction, arrow_cells, block_shape, requested, stop_short):
    tip = _tip_cell(arrow_cells, direction)
    bh, bw = block_shape
    dr, dc = _DIRS[direction]
    positions = []
    if direction in ("right", "left"):
        row_tops = [tip[0]] if bh == 1 else [tip[0] - 1, tip[0]]
        for top in row_tops:
            if top < 0 or top + bh > h:
                continue
            if direction == "right":
                start = tip[1] + 2
                end = w - bw - requested
                if stop_short:
                    end = w - bw
                for left in range(start, end + 1):
                    positions.append((top, left))
            else:
                start = requested
                if stop_short:
                    start = 0
                end = tip[1] - bw - 1
                for left in range(start, end + 1):
                    positions.append((top, left))
    else:
        col_lefts = [tip[1]] if bw == 1 else [tip[1] - 1, tip[1]]
        for left in col_lefts:
            if left < 0 or left + bw > w:
                continue
            if direction == "down":
                start = tip[0] + 2
                end = h - bh - requested
                if stop_short:
                    end = h - bh
                for top in range(start, end + 1):
                    positions.append((top, left))
            else:
                start = requested
                if stop_short:
                    start = 0
                end = tip[0] - bh - 1
                for top in range(start, end + 1):
                    positions.append((top, left))
    random.shuffle(positions)
    return positions


def _would_stop_short(block_cells, direction, requested, arrow_cells, shape):
    dr, dc = _DIRS[direction]
    return _slide_block(block_cells, dr, dc, requested, arrow_cells, shape) != {
        (r + dr * requested, c + dc * requested) for r, c in block_cells
    }


def generate_input() -> np.ndarray:
    h = random.randint(18, 30)
    w = random.randint(18, 30)
    grid = np.zeros((h, w), dtype=int)
    occupied = set()
    arrows_to_place = random.randint(2, 5)
    desired = []
    base = [("right", 3, False), ("left", 3, True), ("down", 5, False), ("up", 5, True)]
    random.shuffle(base)
    desired.extend(base[: min(arrows_to_place, len(base))])
    while len(desired) < arrows_to_place:
        desired.append((random.choice(tuple(_DIRS)), random.choice((3, 5)), random.choice((False, True))))

    placed = 0
    for direction, size, stop_short in desired:
        pts = _arrow_offsets(direction, size)
        ah = max(r for r, _ in pts) + 1
        aw = max(c for _, c in pts) + 1
        arrow_positions = [(r, c) for r in range(1, h - ah) for c in range(1, w - aw)]
        random.shuffle(arrow_positions)
        made = False
        for ar, ac in arrow_positions[:250]:
            arrow_cells = _place(pts, ar, ac)
            if not _fits(arrow_cells, h, w, occupied):
                continue
            requested = 1 if size == 3 else 2
            bh = random.choice((1, 2))
            bw = random.choice((1, 2))
            positions = _candidate_positions(h, w, direction, arrow_cells, (bh, bw), requested, stop_short)
            for br, bc in positions:
                block = {(br + rr, bc + cc) for rr in range(bh) for cc in range(bw)}
                if not _fits(block, h, w, occupied | arrow_cells):
                    continue
                if not _line_clear_between(arrow_cells, block, direction, occupied | arrow_cells | block):
                    continue
                actual_stop = _would_stop_short(block, direction, requested, arrow_cells, (h, w))
                if actual_stop != stop_short:
                    continue
                for r, c in arrow_cells:
                    grid[r, c] = GREEN
                for r, c in block:
                    grid[r, c] = YELLOW
                occupied.update(arrow_cells)
                occupied.update(block)
                placed += 1
                made = True
                break
            if made:
                break
    if placed < 2:
        return generate_input()
    return grid
