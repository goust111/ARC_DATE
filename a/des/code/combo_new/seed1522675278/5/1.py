import random
from collections import deque

import numpy as np


BG = 0
GREY = 5


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
            cur = []
            while q:
                x, y = q.popleft()
                cur.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            comps.append(cur)
    return comps


def _frame_bbox(grid):
    colors = [c for c in np.unique(grid) if c not in (BG, GREY)]
    color = max(colors, key=lambda c: np.sum(grid == c))
    pts = np.argwhere(grid == color)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return color, r0, r1, c0, c1


def _sym_ok(arr, hole_mask, kind):
    h, w = arr.shape
    for r in range(h):
        for c in range(w):
            if hole_mask[r, c]:
                continue
            if kind == "h":
                rr, cc = h - 1 - r, c
            elif kind == "v":
                rr, cc = r, w - 1 - c
            else:
                rr, cc = c, h - 1 - r
                if not (0 <= rr < h and 0 <= cc < w):
                    return False
            if hole_mask[rr, cc]:
                continue
            if arr[r, c] != arr[rr, cc]:
                return False
    return True


def _expected_patch(arr, hole_mask, kind):
    h, w = arr.shape
    out = np.zeros_like(arr)
    hole = np.argwhere(hole_mask)
    r0, c0 = hole.min(axis=0)
    r1, c1 = hole.max(axis=0)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if kind == "h":
                rr, cc = h - 1 - r, c
            elif kind == "v":
                rr, cc = r, w - 1 - c
            else:
                rr, cc = c, h - 1 - r
            out[r, c] = arr[rr, cc]
    return out[r0 : r1 + 1, c0 : c1 + 1], (r0, r1, c0, c1)


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    divider = int(np.where(np.all(grid == GREY, axis=0))[0][0])
    left = grid[:, :divider].copy()
    right = grid[:, divider + 1 :]
    color, r0, r1, c0, c1 = _frame_bbox(left)
    inner = left[r0 + 1 : r1, c0 + 1 : c1]
    bg_comps = _components(inner == BG)
    expected = None
    hole_box = None
    for comp in bg_comps:
        rows = [p[0] for p in comp]
        cols = [p[1] for p in comp]
        rr0, rr1 = min(rows), max(rows)
        cc0, cc1 = min(cols), max(cols)
        if len(comp) != (rr1 - rr0 + 1) * (cc1 - cc0 + 1):
            continue
        hole_mask = np.zeros_like(inner, dtype=bool)
        hole_mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
        for kind in ("h", "v", "r"):
            if _sym_ok(inner, hole_mask, kind):
                expected, hole_box = _expected_patch(inner, hole_mask, kind)
                break
        if expected is not None:
            break
    if expected is None:
        return left
    matched = expected
    for comp in _components(right != BG):
        rows = [p[0] for p in comp]
        cols = [p[1] for p in comp]
        rr0, rr1 = min(rows), max(rows)
        cc0, cc1 = min(cols), max(cols)
        sub = right[rr0 : rr1 + 1, cc0 : cc1 + 1]
        if sub.shape[0] < 3 or sub.shape[1] < 3:
            continue
        inner_patch = sub[1:-1, 1:-1]
        for k in range(4):
            rot = np.rot90(inner_patch, k)
            if rot.shape == expected.shape and np.array_equal(rot, expected):
                matched = rot
                break
    hr0, hr1, hc0, hc1 = hole_box
    left[r0 + 1 + hr0 : r0 + 2 + hr1, c0 + 1 + hc0 : c0 + 2 + hc1] = matched
    return left


def generate_input() -> np.ndarray:
    h = random.randint(18, 24)
    w = random.randint(24, 30)
    divider = random.randint(13, 16)
    grid = np.zeros((h, w), dtype=int)
    grid[:, divider] = GREY
    frame_color = random.choice([1, 2, 3, 4])
    top = random.randint(2, 5)
    left = random.randint(1, 2)
    max_w = divider - left - 4
    if max_w < 6:
        divider = left + 10
        max_w = divider - left - 4
    if random.random() < 0.33:
        side = random.randint(6, min(8, max_w))
        ih = iw = side
    else:
        ih = random.randint(6, 8)
        iw = random.randint(6, min(8, max_w))
    bottom = top + ih + 1
    right = left + iw + 1
    grid[top : bottom + 1, left] = frame_color
    grid[top : bottom + 1, right] = frame_color
    grid[top, left : right + 1] = frame_color
    grid[bottom, left : right + 1] = frame_color
    inner = np.zeros((ih, iw), dtype=int)
    kind = random.choice(["h", "v", "r"] if ih == iw else ["h", "v"])
    if kind == "h":
        base = np.random.randint(0, 2, size=((ih + 1) // 2, iw))
        full = np.vstack([base, base[: ih // 2][::-1]])
    elif kind == "v":
        base = np.random.randint(0, 2, size=(ih, (iw + 1) // 2))
        full = np.hstack([base, base[:, : iw // 2][:, ::-1]])
    else:
        q = np.random.randint(0, 2, size=(ih, iw))
        q = np.triu(q)
        full = np.maximum.reduce([q, np.rot90(q), np.rot90(q, 2), np.rot90(q, 3)])
    inner[full == 1] = frame_color
    hole_h = random.randint(2, min(3, ih - 1))
    hole_w = random.randint(2, min(3, iw - 1))
    tries = 0
    while True:
        tries += 1
        hr = random.randint(0, ih - hole_h)
        hc = random.randint(0, iw - hole_w)
        hole_mask = np.zeros_like(inner, dtype=bool)
        hole_mask[hr : hr + hole_h, hc : hc + hole_w] = True
        if _sym_ok(inner, hole_mask, kind):
            break
        if tries > 100:
            hr, hc = 1, 1
            break
    expected, _ = _expected_patch(inner, hole_mask, kind)
    inner[hr : hr + hole_h, hc : hc + hole_w] = BG
    grid[top + 1 : bottom, left + 1 : right] = inner
    y = 2
    x = divider + 2
    patches = []
    good_rot = random.randint(0, 3)
    patches.append(np.rot90(expected, good_rot))
    for _ in range(2):
        bad = np.random.choice([BG, frame_color], size=expected.shape)
        if np.array_equal(bad, expected):
            bad[0, 0] = frame_color if bad[0, 0] == BG else BG
        patches.append(bad)
    random.shuffle(patches)
    for patch in patches:
        bh, bw = patch.shape[0] + 2, patch.shape[1] + 2
        border = random.choice([6, 7, 8, 9])
        if x + bw >= w:
            x = max(divider + 2, w - bw - 1)
        if y + bh >= h:
            y = max(1, h - bh - 1)
        grid[y : y + bh, x : x + bw] = border
        grid[y + 1 : y + bh - 1, x + 1 : x + bw - 1] = patch
        y += bh + 1
    return grid
