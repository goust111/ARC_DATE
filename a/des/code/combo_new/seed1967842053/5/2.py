import random
from collections import deque

import numpy as np


BG = 0
SIDES = ("top", "bottom", "left", "right")


def _bbox(cells):
    rs = [p[0] for p in cells]
    cs = [p[1] for p in cells]
    return min(rs), max(rs), min(cs), max(cs)


def _same_color_components(grid, bg):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c]:
                continue
            color = int(grid[r, c])
            if color == bg:
                seen[r, c] = True
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append({"color": color, "cells": cells, "bbox": _bbox(cells)})
    return comps


def _component_mask(comp):
    r0, r1, c0, c1 = comp["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in comp["cells"]:
        mask[r - r0, c - c0] = True
    return mask


def _compact(mask):
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return mask.copy(), (0, 0)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy(), (int(r0), int(c0))


def _mask_sig(mask):
    return mask.shape, mask.astype(np.uint8).tobytes()


def _find_notches(grid, bg, frame_box):
    fr0, fr1, fc0, fc1 = frame_box
    sub = grid[fr0 : fr1 + 1, fc0 : fc1 + 1] == bg
    h, w = sub.shape
    seen = np.zeros((h, w), dtype=bool)
    out = []
    side_names = ("top", "bottom", "left", "right")
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not sub[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            touches = [False, False, False, False]
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                touches[0] |= rr == 0
                touches[1] |= rr == h - 1
                touches[2] |= cc == 0
                touches[3] |= cc == w - 1
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and sub[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if sum(touches) != 1:
                continue
            side = side_names[touches.index(True)]
            lr0, lr1, lc0, lc1 = _bbox(cells)
            mask = np.zeros((lr1 - lr0 + 1, lc1 - lc0 + 1), dtype=bool)
            for rr, cc in cells:
                mask[rr - lr0, cc - lc0] = True
            out.append(
                {
                    "side": side,
                    "top": (fr0 + lr0, fc0 + lc0),
                    "mask": mask,
                    "sig": _mask_sig(mask),
                }
            )
    return out


def _piece_side(piece_box, frame_box):
    pr0, pr1, pc0, pc1 = piece_box
    fr0, fr1, fc0, fc1 = frame_box
    candidates = []
    if pr1 < fr0:
        candidates.append(("top", fr0 - pr1))
    if pr0 > fr1:
        candidates.append(("bottom", pr0 - fr1))
    if pc1 < fc0:
        candidates.append(("left", fc0 - pc1))
    if pc0 > fc1:
        candidates.append(("right", pc0 - fc1))
    if not candidates:
        return None
    return max(candidates, key=lambda x: x[1])[0]


def _protrusion(mask, side):
    h, w = mask.shape
    if side == "top":
        k = 0
        while k < h and mask[k, :].all():
            k += 1
        part = mask[k:, :]
        base = (k, 0)
    elif side == "bottom":
        k = 0
        while k < h and mask[h - 1 - k, :].all():
            k += 1
        part = mask[: h - k, :]
        base = (0, 0)
    elif side == "left":
        k = 0
        while k < w and mask[:, k].all():
            k += 1
        part = mask[:, k:]
        base = (0, k)
    else:
        k = 0
        while k < w and mask[:, w - 1 - k].all():
            k += 1
        part = mask[:, : w - k]
        base = (0, 0)
    compact, off = _compact(part)
    return compact, (base[0] + off[0], base[1] + off[1])


def main(grid: np.ndarray) -> np.ndarray:
    """Slide each outside toothed piece into the matching notch of the central frame.

    Input structure:
    The grid has black background, one large connected single-color central frame,
    and smaller one-color pieces outside the four sides of that frame.  The frame
    has background-colored recesses cut into its boundary, and each piece has a
    flat outer block with a colored protrusion aimed toward the frame.

    Rule:
    Identify the frame as the largest non-background component.  For every other
    colored component, determine whether it lies above, below, left, or right of
    the frame.  Extract the inward protrusion of that piece and match its visible
    shape to the unique background notch on the same side of the frame.  Remove
    the piece from its original location, then translate the whole piece straight
    toward the frame so the protrusion exactly covers the matched notch.

    Key insight:
    Direction gives the sliding axis, while the protrusion/notch shape gives the
    lateral alignment and the target slot.
    """
    grid = np.asarray(grid)
    comps = _same_color_components(grid, BG)
    if not comps:
        return grid.copy()
    frame = max(comps, key=lambda comp: len(comp["cells"]))
    frame_color = frame["color"]
    frame_box = frame["bbox"]
    notches = _find_notches(grid, BG, frame_box)
    used = set()
    pieces = [comp for comp in comps if comp is not frame and comp["color"] != frame_color]

    placements = []
    for comp in pieces:
        side = _piece_side(comp["bbox"], frame_box)
        if side is None:
            continue
        piece_mask = _component_mask(comp)
        tooth, offset = _protrusion(piece_mask, side)
        sig = _mask_sig(tooth)
        match = None
        for idx, notch in enumerate(notches):
            if idx not in used and notch["side"] == side and notch["sig"] == sig:
                match = idx
                break
        if match is None:
            continue
        used.add(match)
        nr, nc = notches[match]["top"]
        placements.append((comp, piece_mask, nr - offset[0], nc - offset[1]))

    out = grid.copy()
    for comp in pieces:
        for r, c in comp["cells"]:
            out[r, c] = BG
    for comp, piece_mask, top, left in placements:
        color = comp["color"]
        coords = np.argwhere(piece_mask)
        for rr, cc in coords:
            r, c = top + int(rr), left + int(cc)
            if 0 <= r < out.shape[0] and 0 <= c < out.shape[1]:
                out[r, c] = color
    return out


def _top_tab(depth, length, variant):
    mask = np.zeros((depth, length), dtype=bool)
    for r in range(depth):
        if variant == 0:
            lo, hi = 0, length
        elif variant == 1:
            lo, hi = min(r, length - 1), length
        elif variant == 2:
            lo, hi = 0, max(1, length - r)
        else:
            shrink = min(r, max(0, (length - 1) // 2))
            lo, hi = shrink, length - shrink
            if lo >= hi:
                mid = length // 2
                lo, hi = mid, mid + 1
        mask[r, lo:hi] = True
    return mask


def _tab_for_side(side, depth, length, variant):
    top = _top_tab(depth, length, variant)
    if side == "top":
        return top
    if side == "bottom":
        return top[::-1, :]
    left = top.T
    if side == "left":
        return left
    return left[:, ::-1]


def _piece_from_tab(side, tab, base, lead, trail):
    th, tw = tab.shape
    if side in ("top", "bottom"):
        out = np.zeros((base + th, lead + tw + trail), dtype=bool)
        if side == "top":
            out[:base, :] = True
            out[base : base + th, lead : lead + tw] |= tab
        else:
            out[th : th + base, :] = True
            out[:th, lead : lead + tw] |= tab
    else:
        out = np.zeros((lead + th + trail, base + tw), dtype=bool)
        if side == "left":
            out[:, :base] = True
            out[lead : lead + th, base : base + tw] |= tab
        else:
            out[:, tw : tw + base] = True
            out[lead : lead + th, :tw] |= tab
    return out


def _can_place(grid, mask, top, left):
    h, w = grid.shape
    mh, mw = mask.shape
    if top < 0 or left < 0 or top + mh > h or left + mw > w:
        return False
    block = grid[top : top + mh, left : left + mw]
    if np.any(block[mask] != BG):
        return False
    r0, r1 = max(0, top - 1), min(h, top + mh + 1)
    c0, c1 = max(0, left - 1), min(w, left + mw + 1)
    return not np.any(grid[r0:r1, c0:c1] != BG)


def _paint(grid, mask, top, left, color):
    block = grid[top : top + mask.shape[0], left : left + mask.shape[1]]
    block[mask] = color


def _choose_start(lo, hi, length, intervals):
    if hi < lo:
        return None
    for _ in range(100):
        start = random.randint(lo, hi)
        end = start + length - 1
        if all(end + 1 < a or start - 1 > b for a, b in intervals):
            intervals.append((start, end))
            return start
    for start in range(lo, hi + 1):
        end = start + length - 1
        if all(end + 1 < a or start - 1 > b for a, b in intervals):
            intervals.append((start, end))
            return start
    return None


def _place_outside(grid, side, mask, frame_box, color):
    h, w = grid.shape
    fr0, fr1, fc0, fc1 = frame_box
    mh, mw = mask.shape
    tries = []
    for _ in range(200):
        if side == "top":
            row = random.randint(0, max(0, fr0 - mh - 1))
            lo, hi = max(0, fc0 - 2), min(w - mw, fc1 + 2)
            col = random.randint(lo, max(lo, hi))
        elif side == "bottom":
            row = random.randint(min(h - mh, fr1 + 2), h - mh)
            lo, hi = max(0, fc0 - 2), min(w - mw, fc1 + 2)
            col = random.randint(lo, max(lo, hi))
        elif side == "left":
            col = random.randint(0, max(0, fc0 - mw - 1))
            lo, hi = max(0, fr0 - 2), min(h - mh, fr1 + 2)
            row = random.randint(lo, max(lo, hi))
        else:
            col = random.randint(min(w - mw, fc1 + 2), w - mw)
            lo, hi = max(0, fr0 - 2), min(h - mh, fr1 + 2)
            row = random.randint(lo, max(lo, hi))
        tries.append((row, col))

    for row in range(h - mh + 1):
        for col in range(w - mw + 1):
            if side == "top" and row + mh > fr0 - 1:
                continue
            if side == "bottom" and row < fr1 + 2:
                continue
            if side == "left" and col + mw > fc0 - 1:
                continue
            if side == "right" and col < fc1 + 2:
                continue
            tries.append((row, col))

    random.shuffle(tries)
    for row, col in tries:
        if _can_place(grid, mask, row, col):
            _paint(grid, mask, row, col, color)
            return True
    return False


def _build_input():
    height = random.randint(28, 30)
    width = random.randint(28, 30)
    top_margin = random.randint(6, 7)
    bottom_margin = random.randint(6, 7)
    left_margin = random.randint(6, 7)
    right_margin = random.randint(6, 7)
    fr0 = top_margin
    fr1 = height - bottom_margin - 1
    fc0 = left_margin
    fc1 = width - right_margin - 1
    frame_box = (fr0, fr1, fc0, fc1)

    grid = np.zeros((height, width), dtype=int)
    frame_color = 5
    grid[fr0 : fr1 + 1, fc0 : fc1 + 1] = frame_color

    counts = {side: 1 for side in SIDES}
    for side in random.sample(SIDES, random.randint(0, 4)):
        counts[side] += 1

    colors = [1, 2, 3, 4, 6, 7, 8, 9]
    random.shuffle(colors)
    intervals = {side: [] for side in SIDES}
    used_shapes = {side: set() for side in SIDES}
    pieces = []

    for side in SIDES:
        for _ in range(counts[side]):
            tab = None
            for _attempt in range(100):
                depth = random.randint(1, 3)
                length = random.randint(2, 4)
                variant = random.randint(0, 3)
                candidate = _tab_for_side(side, depth, length, variant)
                sig = _mask_sig(candidate)
                if sig not in used_shapes[side]:
                    used_shapes[side].add(sig)
                    tab = candidate
                    break
            if tab is None:
                return None

            lateral = tab.shape[1] if side in ("top", "bottom") else tab.shape[0]
            if side in ("top", "bottom"):
                lo, hi = fc0 + 2, fc1 - 2 - lateral + 1
            else:
                lo, hi = fr0 + 2, fr1 - 2 - lateral + 1
            start = _choose_start(lo, hi, lateral, intervals[side])
            if start is None:
                return None

            if side == "top":
                nr, nc = fr0, start
            elif side == "bottom":
                nr, nc = fr1 - tab.shape[0] + 1, start
            elif side == "left":
                nr, nc = start, fc0
            else:
                nr, nc = start, fc1 - tab.shape[1] + 1
            block = grid[nr : nr + tab.shape[0], nc : nc + tab.shape[1]]
            block[tab] = BG

            lead = random.randint(1, 2)
            trail = random.randint(1, 2)
            piece = _piece_from_tab(side, tab, 2, lead, trail)
            pieces.append((side, piece, colors.pop()))

    random.shuffle(pieces)
    for side, piece, color in pieces:
        if not _place_outside(grid, side, piece, frame_box, color):
            return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(200):
        grid = _build_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
