import random
from collections import deque

import numpy as np


BLACK = 0
MAROON = 9
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
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
                for dr, dc in DIRS.values():
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _is_arrow_component(cells):
    if len(cells) != 3:
        return False
    rs = cells[:, 0]
    cs = cells[:, 1]
    return rs.max() - rs.min() == 1 and cs.max() - cs.min() == 1


def _frame_bbox(grid):
    loc = np.argwhere(grid == MAROON)
    if len(loc) == 0:
        return None
    return (
        int(loc[:, 0].min()),
        int(loc[:, 0].max()),
        int(loc[:, 1].min()),
        int(loc[:, 1].max()),
    )


def _arrow_direction(cells, box):
    top, bottom, left, right = box
    r0, r1 = int(cells[:, 0].min()), int(cells[:, 0].max())
    c0, c1 = int(cells[:, 1].min()), int(cells[:, 1].max())
    if c1 < left:
        return "right"
    if c0 > right:
        return "left"
    if r1 < top:
        return "down"
    if r0 > bottom:
        return "up"

    cr = (r0 + r1) / 2
    cc = (c0 + c1) / 2
    distances = [
        (abs(cc - left), "right" if cc < left else "left"),
        (abs(cr - top), "down" if cr < top else "up"),
    ]
    return min(distances)[1]


def _extract_objects(grid):
    box = _frame_bbox(grid)
    if box is None:
        return [], None
    objects = []
    colors = [int(v) for v in np.unique(grid) if v not in (BLACK, MAROON)]
    for color in colors:
        comps = _components(grid == color)
        arrows = [comp for comp in comps if _is_arrow_component(comp)]
        pieces = [comp for comp in comps if not _is_arrow_component(comp)]
        if not arrows or not pieces:
            continue
        for piece in pieces:
            pc = piece.mean(axis=0)
            arrow = min(arrows, key=lambda a: float(np.abs(a.mean(axis=0) - pc).sum()))
            objects.append(
                {
                    "color": color,
                    "cells": piece,
                    "area": len(piece),
                    "direction": _arrow_direction(arrow, box),
                    "order": (
                        int(piece[:, 0].min()),
                        int(piece[:, 1].min()),
                    ),
                }
            )
    objects.sort(key=lambda obj: (obj["area"], obj["order"]))
    return objects, box


def _inside_frame(cells, box):
    top, bottom, left, right = box
    return (
        np.all(cells[:, 0] > top)
        and np.all(cells[:, 0] < bottom)
        and np.all(cells[:, 1] > left)
        and np.all(cells[:, 1] < right)
    )


def _slide_position(cells, direction, out, box):
    dr, dc = DIRS[direction]
    limit = out.shape[0] + out.shape[1] + 8
    last = None
    seen_valid = False
    for step in range(limit + 1):
        moved = cells + np.array([dr * step, dc * step])
        if not _inside_frame(moved, box):
            if seen_valid:
                break
            continue
        rr = moved[:, 0]
        cc = moved[:, 1]
        if np.all(out[rr, cc] == BLACK):
            last = moved
            seen_valid = True
        elif seen_valid:
            break
    return last


def _solve(grid, report=False):
    objects, box = _extract_objects(grid)
    out = np.zeros_like(grid)
    out[grid == MAROON] = MAROON
    ok = True
    if box is None:
        return (out, False) if report else out
    for obj in objects:
        placed = _slide_position(obj["cells"], obj["direction"], out, box)
        if placed is None:
            ok = False
            continue
        out[placed[:, 0], placed[:, 1]] = obj["color"]
    return (out, ok and len(objects) > 0) if report else out


def main(grid: np.ndarray) -> np.ndarray:
    """Slide colored fragments into the maroon frame in increasing area order.

    Input structure: The grid has a black background, one hollow maroon
    rectangular frame, and several colored fragments outside the frame.  Each
    fragment is a connected component of one non-background color, paired with
    a separate same-colored three-pixel L-shaped arrow marker pointing from its
    side of the grid toward the frame.

    Rule: Remove the arrow markers and process the fragments from smallest area
    to largest area.  For each fragment, translate its original shape along the
    direction indicated by its arrow until its leading edge is blocked by the
    inner frame wall or by fragments already placed inside the frame.  Keep the
    maroon frame unchanged and leave all original outside positions empty.

    Key insight: The arrows give only the sliding direction; the hidden ordering
    signal is fragment area, so smaller fragments settle deeper in the frame and
    larger fragments stack nearer the entry side.
    """
    return _solve(grid)


def _normalize(cells):
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return sorted((r - min_r, c - min_c) for r, c in cells)


def _random_polyomino(area, max_span=4):
    for _ in range(200):
        cells = {(0, 0)}
        while len(cells) < area:
            r, c = random.choice(tuple(cells))
            dr, dc = random.choice(tuple(DIRS.values()))
            cells.add((r + dr, c + dc))
        norm = _normalize(cells)
        rs = [r for r, _ in norm]
        cs = [c for _, c in norm]
        if max(rs) + 1 <= max_span and max(cs) + 1 <= max_span:
            return np.array(norm, dtype=int)
    return np.array(_normalize(cells), dtype=int)


def _arrow_cells(direction, tail_r, tail_c, wing_sign):
    if direction == "right":
        return [(tail_r, tail_c), (tail_r, tail_c + 1), (tail_r + wing_sign, tail_c + 1)]
    if direction == "left":
        return [(tail_r, tail_c), (tail_r, tail_c - 1), (tail_r + wing_sign, tail_c - 1)]
    if direction == "down":
        return [(tail_r, tail_c), (tail_r + 1, tail_c), (tail_r + 1, tail_c + wing_sign)]
    return [(tail_r, tail_c), (tail_r - 1, tail_c), (tail_r - 1, tail_c + wing_sign)]


def _place_input_object(grid, frame, shape, color, direction):
    top, bottom, left, right = frame
    h, w = int(shape[:, 0].max()) + 1, int(shape[:, 1].max()) + 1
    ih, iw = bottom - top - 1, right - left - 1
    if h > ih or w > iw:
        return False

    for _ in range(80):
        wing = random.choice([-1, 1])
        if direction in ("right", "left"):
            base_r = random.randint(top + 1, bottom - h)
            if direction == "right":
                base_c = left - w - 3
                ar = base_r + h // 2
                ac = base_c + w + 1
            else:
                base_c = right + 4
                ar = base_r + h // 2
                ac = base_c - 2
        else:
            base_c = random.randint(left + 1, right - w)
            if direction == "down":
                base_r = top - h - 3
                ar = base_r + h + 1
                ac = base_c + w // 2
            else:
                base_r = bottom + 4
                ar = base_r - 2
                ac = base_c + w // 2

        piece = shape + np.array([base_r, base_c])
        arrow = np.array(_arrow_cells(direction, ar, ac, wing), dtype=int)
        all_cells = np.vstack([piece, arrow])
        if (
            np.any(all_cells[:, 0] < 0)
            or np.any(all_cells[:, 0] >= grid.shape[0])
            or np.any(all_cells[:, 1] < 0)
            or np.any(all_cells[:, 1] >= grid.shape[1])
        ):
            continue
        if np.all(grid[all_cells[:, 0], all_cells[:, 1]] == BLACK):
            grid[piece[:, 0], piece[:, 1]] = color
            grid[arrow[:, 0], arrow[:, 1]] = color
            return True
    return False


def generate_input() -> np.ndarray:
    for _ in range(300):
        rows = random.randint(24, 30)
        cols = random.randint(24, 30)
        max_fh = min(10, rows - 16)
        max_fw = min(10, cols - 16)
        if max_fh < 6 or max_fw < 6:
            continue
        inner_h = random.randint(6, max_fh)
        inner_w = random.randint(6, max_fw)
        top = random.randint(7, rows - inner_h - 9)
        left = random.randint(7, cols - inner_w - 9)
        bottom = top + inner_h + 1
        right = left + inner_w + 1

        grid = np.zeros((rows, cols), dtype=int)
        grid[top, left : right + 1] = MAROON
        grid[bottom, left : right + 1] = MAROON
        grid[top : bottom + 1, left] = MAROON
        grid[top : bottom + 1, right] = MAROON

        count = random.randint(2, 4)
        colors = random.sample([1, 2, 3, 4, 5, 6, 7, 8], count)
        areas = random.sample([4, 5, 6, 7, 8], count)
        directions = [random.choice(tuple(DIRS.keys())) for _ in range(count)]
        order = list(range(count))
        random.shuffle(order)
        frame = (top, bottom, left, right)

        good = True
        for idx in order:
            shape = _random_polyomino(areas[idx])
            if not _place_input_object(grid, frame, shape, colors[idx], directions[idx]):
                good = False
                break
        if not good:
            continue
        objects, _ = _extract_objects(grid)
        if len(objects) != count:
            continue
        _, ok = _solve(grid, report=True)
        if ok:
            return grid

    return generate_input()
