from __future__ import annotations

import numpy as np
from collections import deque

BLACK = 0
GREY = 5
YELLOW = 4
RED = 2
BLUE = 1
GREEN = 3


def _neighbors4(r: int, c: int, H: int, W: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < H:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < W:
        yield r, c + 1


def _components(mask: np.ndarray):
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r0 in range(H):
        for c0 in range(W):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            q = deque([(r0, c0)])
            seen[r0, c0] = True
            pts = []
            while q:
                r, c = q.popleft()
                pts.append((r, c))
                for rr, cc in _neighbors4(r, c, H, W):
                    if mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        q.append((rr, cc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _find_gray_frame(grid: np.ndarray):
    H, W = grid.shape
    comps = _components(grid == GREY)
    best = None
    for comp in comps:
        if len(comp) < 20:
            continue
        r0, c0 = int(comp[:, 0].min()), int(comp[:, 1].min())
        r1, c1 = int(comp[:, 0].max()), int(comp[:, 1].max())
        h, w = r1 - r0 + 1, c1 - c0 + 1
        if h < 4 or w < 8:
            continue
        if len(comp) != 2 * (h + w) - 4:
            continue
        for rr in range(r0, r1 + 1):
            if grid[rr, c0] != GREY or grid[rr, c1] != GREY:
                break
        else:
            for cc in range(c0, c1 + 1):
                if grid[r0, cc] != GREY or grid[r1, cc] != GREY:
                    break
            else:
                best = (r0, c0, r1, c1)
                break
    return best


def main(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, copy=True)
    H, W = out.shape

    frame = _find_gray_frame(out)
    if frame is None:
        return out

    r0, c0, r1, c1 = frame
    inner_top, inner_bottom = r0 + 1, r1 - 1
    inner_left, inner_right = c0 + 1, c1 - 1

    top_color = int(out[r0, inner_left])
    bottom_color = int(out[r1, inner_left])
    left_color = int(out[inner_top, c0])
    right_color = int(out[inner_top, c1])

    gray_mask = out == GREY
    frame_mask = np.zeros_like(gray_mask, dtype=bool)
    frame_mask[r0 : r1 + 1, c0 : c1 + 1] = True
    bar_mask = gray_mask & ~frame_mask
    bar_comps = _components(bar_mask)
    if not bar_comps:
        return out
    bar = max(bar_comps, key=len)

    br, bc = bar[:, 0], bar[:, 1]
    base_row = int(br.max())
    target_base = inner_bottom - 1
    dr = target_base - base_row

    marker_cols = [int(c) for r, c in np.argwhere((out == YELLOW) & (np.arange(H)[:, None] == inner_bottom))]

    # columns where bar has a cell and also a cell above it
    protrusions = set(int(c) for (r, c) in bar if int(r - 1) in set(br) and (r - 1, c) in set((int(rr), int(cc)) for rr, cc in bar))
    # find best horizontal shift
    delta_candidates = range(-12, 13)
    best_shift = 0
    best_score = -1
    marker_set = set(marker_cols)
    for d in delta_candidates:
        score = sum((c + d) in marker_set for c in protrusions)
        if score > best_score:
            best_shift = d
            best_score = score

    out[bar_mask] = BLACK
    shifted_bar = []
    for r, c in bar:
        nr = int(r) + dr
        nc = int(c) + best_shift
        if 0 <= nr < H and 0 <= nc < W:
            shifted_bar.append((nr, nc))
            out[nr, nc] = GREY

    bar_cols = set(c for _, c in shifted_bar)
    shifted_bar_set = set(shifted_bar)

    # move markers matching protrusion columns
    markers = []
    for r, c in np.argwhere(np.full_like(out, False, dtype=bool)):
        pass
    marker_rows = set()
    for c in marker_cols:
        if c in bar_cols:
            cols = [r for r, cc in shifted_bar if cc == c]
            top = min(cols)
            mr = top - 1
            if inner_top <= mr <= inner_bottom:
                marker_rows.add((mr, c + best_shift))
                continue
        marker_rows.add((inner_bottom, c))

    # clear original bottom-row markers
    for c in marker_cols:
        out[inner_bottom, c] = BLACK

    for mr, mc in marker_rows:
        if not (inner_top <= mr <= inner_bottom and inner_left <= mc <= inner_right):
            continue
        out[mr, mc] = YELLOW

        du = mr - inner_top
        dd = inner_bottom - mr
        dl = mc - inner_left
        dr = inner_right - mc
        mind = min(du, dd, dl, dr)
        cnt = 0
        if du == mind:
            cnt += 1
            chosen = top_color
        if dd == mind:
            cnt += 1
            chosen = bottom_color
        if dl == mind:
            cnt += 1
            chosen = left_color
        if dr == mind:
            cnt += 1
            chosen = right_color
        if cnt == 1:
            out[mr, mc] = chosen

    return out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    out = np.zeros((16, 16), dtype=int)
    r0, c0, r1, c1 = 1, 1, 13, 14
    ir_top, ir_bottom = r0 + 1, r1 - 1
    ir_left, ir_right = c0 + 1, c1 - 1

    top_color, bottom_color = RED, BLUE
    left_color, right_color = GREEN, YELLOW

    out[r0, c0 + 1 : c1] = top_color
    out[r1, c0 + 1 : c1] = bottom_color
    out[r0 + 1 : r1, c0] = left_color
    out[r0 + 1 : r1, c1] = right_color

    # markers at bottom inside row
    base_cols = [c for c in range(ir_left + 1, ir_right, 3)]
    marker_cols = rng.choice(base_cols, size=3, replace=False)
    for c in marker_cols:
        out[ir_bottom, int(c)] = YELLOW

    # bar component below
    base_row = r1 + 2
    cols = list(range(3, 13))
    rng.shuffle(cols)
    bar_cols = cols[:rng.integers(3, 6)]
    for c in bar_cols:
        out[base_row, c] = GREY
    for c in bar_cols:
        if rng.random() < 0.7:
            h = 2
            for rr in range(base_row - h, base_row):
                out[rr, c] = GREY
    return out
