from collections import deque
import random

import numpy as np


BG = 0
RED = 2

DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _neighbors4(r, c):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask):
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    comps = []
    for sr, sc in zip(*np.where(mask & ~seen)):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        q = deque([(sr, sc)])
        seen[sr, sc] = True
        pts = []
        while q:
            r, c = q.popleft()
            pts.append((r, c))
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(pts)
    return comps


def _bbox(points):
    rows = [p[0] for p in points]
    cols = [p[1] for p in points]
    return min(rows), min(cols), max(rows), max(cols)


def _shape_color_at_marker(g, r, c):
    counts = {}
    h, w = g.shape
    for radius in (1,):
        for nr in range(max(0, r - radius), min(h, r + radius + 1)):
            for nc in range(max(0, c - radius), min(w, c + radius + 1)):
                if nr == r and nc == c:
                    continue
                v = int(g[nr, nc])
                if v not in (BG, RED):
                    counts[v] = counts.get(v, 0) + 1
        if counts:
            return max(counts, key=counts.get)
    raise AssertionError("red marker is not adjacent to any shape color")


def _shape_from_marker(g, marker):
    mr, mc = marker
    color = _shape_color_at_marker(g, mr, mc)
    h, w = g.shape
    allowed = (g == color) | (g == RED)
    q = deque([(mr, mc)])
    seen = {(mr, mc)}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and allowed[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return color, seen


def _direction_from_marker(shape, marker):
    r0, c0, r1, c1 = _bbox(shape)
    mr, mc = marker
    hits = []
    if mr == r0:
        hits.append("up")
    if mr == r1:
        hits.append("down")
    if mc == c0:
        hits.append("left")
    if mc == c1:
        hits.append("right")
    if len(hits) != 1:
        raise AssertionError("marker must lie on exactly one side of the shape")
    return hits[0]


def _classify_channel(points):
    r0, c0, r1, c1 = _bbox(points)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None
    pset = set(points)
    top_full = all((r0, c) in pset for c in range(c0, c1 + 1))
    bottom_full = all((r1, c) in pset for c in range(c0, c1 + 1))
    left_full = all((r, c0) in pset for r in range(r0, r1 + 1))
    right_full = all((r, c1) in pset for r in range(r0, r1 + 1))

    if left_full and right_full and (top_full ^ bottom_full):
        closed = "up" if top_full else "down"
        return {
            "closed": closed,
            "orientation": "vertical",
            "bbox": (r0, c0, r1, c1),
            "size": len(points),
        }
    if top_full and bottom_full and (left_full ^ right_full):
        closed = "left" if left_full else "right"
        return {
            "closed": closed,
            "orientation": "horizontal",
            "bbox": (r0, c0, r1, c1),
            "size": len(points),
        }
    return None


def _find_channel(g, color, shape, direction):
    mask = g == color
    for r, c in shape:
        mask[r, c] = False
    candidates = []
    for pts in _components(mask):
        info = _classify_channel(pts)
        if info is not None and info["closed"] == direction:
            candidates.append(info)
    if not candidates:
        raise AssertionError("no matching same-color channel found")
    candidates.sort(key=lambda item: item["size"], reverse=True)
    return candidates[0]


def _inside_channel(info, r, c):
    r0, c0, r1, c1 = info["bbox"]
    closed = info["closed"]
    if info["orientation"] == "vertical":
        if not (c0 < c < c1):
            return False
        return r0 < r <= r1 if closed == "up" else r0 <= r < r1
    if not (r0 < r < r1):
        return False
    return c0 < c <= c1 if closed == "left" else c0 <= c < c1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide marked colored shapes into their same-color dead-end channels.

    Input structure:
    - The background is black. Several colors form one-cell-thick rectangular tube channels; each channel has two long side walls, one closed end, and one open end.
    - Near each open end is a small 4-connected shape in the same color as that channel.
    - Each shape contains exactly one red marker pixel on one side of its bounding box.

    Rule:
    - For every red-marked shape, read the slide direction from the side that contains the red marker: top, bottom, left, or right.
    - Match the shape to the channel with the same outline color and the closed end in that direction.
    - Translate the whole shape straight in that direction until its leading edge is just inside the closed end.
    - Erase the original shape, repaint the moved shape entirely in the channel color, and leave the remaining black channel interior black.

    Key insight: The red marker encodes direction by relative position inside the moving shape, while color matching selects the channel it enters.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    markers = [(int(r), int(c)) for r, c in zip(*np.where(g == RED))]
    shapes = []
    used = set()
    for marker in markers:
        if marker in used:
            continue
        color, shape = _shape_from_marker(g, marker)
        used |= {p for p in shape if int(g[p]) == RED}
        direction = _direction_from_marker(shape, marker)
        channel = _find_channel(g, color, shape, direction)
        shapes.append((color, shape, direction, channel))

    for color, shape, direction, channel in shapes:
        sr0, sc0, sr1, sc1 = _bbox(shape)
        cr0, cc0, cr1, cc1 = channel["bbox"]
        if direction == "up":
            dr, dc = cr0 + 1 - sr0, 0
        elif direction == "down":
            dr, dc = cr1 - 1 - sr1, 0
        elif direction == "left":
            dr, dc = 0, cc0 + 1 - sc0
        else:
            dr, dc = 0, cc1 - 1 - sc1

        new_points = {(r + dr, c + dc) for r, c in shape}
        if not all(_inside_channel(channel, r, c) for r, c in new_points):
            raise AssertionError("translated shape does not fit the channel interior")

        for r, c in shape:
            out[r, c] = BG
        for r, c in new_points:
            out[r, c] = color

    return out


def _connected(mask):
    pts = list(zip(*np.where(mask)))
    if not pts:
        return False
    start = tuple(map(int, pts[0]))
    seen = {start}
    q = deque([start])
    h, w = mask.shape
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _keeps_full_bbox(mask):
    pts = list(zip(*np.where(mask)))
    if not pts:
        return False
    r0, c0, r1, c1 = _bbox([(int(r), int(c)) for r, c in pts])
    return (r0, c0, r1, c1) == (0, 0, mask.shape[0] - 1, mask.shape[1] - 1)


def _make_shape(direction, rng):
    if direction in ("up", "down"):
        h = rng.randint(2, 3)
        w = rng.randint(3, 4)
        marker = (0 if direction == "up" else h - 1, rng.randint(1, w - 2))
    else:
        h = 3
        w = rng.randint(2, 4)
        marker = (1, 0 if direction == "left" else w - 1)

    mask = np.ones((h, w), dtype=bool)
    for _ in range(rng.randint(0, 2)):
        cells = [(r, c) for r in range(h) for c in range(w) if (r, c) != marker and mask[r, c]]
        rng.shuffle(cells)
        for r, c in cells:
            test = mask.copy()
            test[r, c] = False
            if int(test.sum()) >= 4 and test[marker] and _connected(test) and _keeps_full_bbox(test):
                mask = test
                break
    return mask, marker


def _draw_channel(patch, color, direction, r0, c0, h, w):
    if direction == "up":
        patch[r0:r0 + h, c0] = color
        patch[r0:r0 + h, c0 + w - 1] = color
        patch[r0, c0:c0 + w] = color
    elif direction == "down":
        patch[r0:r0 + h, c0] = color
        patch[r0:r0 + h, c0 + w - 1] = color
        patch[r0 + h - 1, c0:c0 + w] = color
    elif direction == "left":
        patch[r0, c0:c0 + w] = color
        patch[r0 + h - 1, c0:c0 + w] = color
        patch[r0:r0 + h, c0] = color
    else:
        patch[r0, c0:c0 + w] = color
        patch[r0 + h - 1, c0:c0 + w] = color
        patch[r0:r0 + h, c0 + w - 1] = color


def _make_object(color, direction, rng):
    shape, marker = _make_shape(direction, rng)
    sh, sw = shape.shape
    gap = rng.randint(0, 2)

    if direction in ("up", "down"):
        inner_w = sw
        inner_len = rng.randint(max(5, sh + 3), 8)
        ch_h, ch_w = inner_len + 1, inner_w + 2
        if direction == "up":
            patch = np.zeros((ch_h + gap + sh, ch_w), dtype=int)
            ch_r0, ch_c0 = 0, 0
            sr0, sc0 = ch_h + gap, 1
        else:
            patch = np.zeros((sh + gap + ch_h, ch_w), dtype=int)
            ch_r0, ch_c0 = sh + gap, 0
            sr0, sc0 = 0, 1
    else:
        inner_h = sh
        inner_len = rng.randint(max(5, sw + 3), 8)
        ch_h, ch_w = inner_h + 2, inner_len + 1
        if direction == "left":
            patch = np.zeros((ch_h, ch_w + gap + sw), dtype=int)
            ch_r0, ch_c0 = 0, 0
            sr0, sc0 = 1, ch_w + gap
        else:
            patch = np.zeros((ch_h, sw + gap + ch_w), dtype=int)
            ch_r0, ch_c0 = 0, sw + gap
            sr0, sc0 = 1, 0

    _draw_channel(patch, color, direction, ch_r0, ch_c0, ch_h, ch_w)
    for rr, cc in zip(*np.where(shape)):
        rr, cc = int(rr), int(cc)
        patch[sr0 + rr, sc0 + cc] = RED if (rr, cc) == marker else color
    return patch


def _can_place(reserve, r0, c0, patch, pad=1):
    h, w = reserve.shape
    ph, pw = patch.shape
    if r0 < 0 or c0 < 0 or r0 + ph > h or c0 + pw > w:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h, r0 + ph + pad)
    cc1 = min(w, c0 + pw + pad)
    return not reserve[rr0:rr1, cc0:cc1].any()


def _paste_object(grid, reserve, patch, r0, c0):
    ph, pw = patch.shape
    mask = patch != BG
    grid[r0:r0 + ph, c0:c0 + pw][mask] = patch[mask]
    reserve[r0:r0 + ph, c0:c0 + pw] = True


def _gen_attempt(rng):
    h = rng.randint(16, 30)
    w = rng.randint(16, 30)
    n = rng.randint(2, 4)
    colors = rng.sample([1, 3, 4, 5, 6, 7, 8, 9], n)
    directions = [rng.choice(list(DIRS)) for _ in range(n)]

    grid = np.zeros((h, w), dtype=int)
    reserve = np.zeros((h, w), dtype=bool)
    objects = [_make_object(color, direction, rng) for color, direction in zip(colors, directions)]
    rng.shuffle(objects)

    for patch in objects:
        ph, pw = patch.shape
        positions = [
            (r, c)
            for r in range(1, h - ph)
            for c in range(1, w - pw)
            if _can_place(reserve, r, c, patch, pad=1)
        ]
        if not positions:
            return None
        r0, c0 = rng.choice(positions)
        _paste_object(grid, reserve, patch, r0, c0)

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(grid, out):
        return None
    if np.any(out == RED):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(500):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid channel-sliding input")
