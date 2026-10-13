import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
CORNERS = ("ul", "ur", "ll", "lr")


def _separator_row(grid):
    full = np.where(np.all(grid == GREY, axis=1))[0]
    if len(full):
        return int(full[len(full) // 2])
    return int(np.argmax(np.sum(grid == GREY, axis=1)))


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords):
    rows = [p[0] for p in coords]
    cols = [p[1] for p in coords]
    return min(rows), max(rows), min(cols), max(cols)


def _rect_component(grid, coords):
    r0, r1, c0, c1 = _bbox(coords)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    if len(coords) != h * w:
        return None
    color = int(grid[coords[0]])
    return {"coords": coords, "bbox": (r0, r1, c0, c1), "h": h, "w": w, "color": color}


def _l_component(grid, coords):
    r0, r1, c0, c1 = _bbox(coords)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    filled = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        filled[r - r0, c - c0] = True

    missing = ~filled
    if not np.any(missing):
        return None
    mr, mc = np.where(missing)
    zr0, zr1 = int(mr.min()), int(mr.max())
    zc0, zc1 = int(mc.min()), int(mc.max())
    expected = np.zeros_like(missing)
    expected[zr0 : zr1 + 1, zc0 : zc1 + 1] = True
    if not np.array_equal(missing, expected):
        return None

    corner = None
    if zr0 == 0 and zc0 == 0:
        corner = "ul"
    elif zr0 == 0 and zc1 == w - 1:
        corner = "ur"
    elif zr1 == h - 1 and zc0 == 0:
        corner = "ll"
    elif zr1 == h - 1 and zc1 == w - 1:
        corner = "lr"
    if corner is None:
        return None

    color = int(grid[coords[0]])
    return {
        "coords": coords,
        "bbox": (r0, r1, c0, c1),
        "mh": zr1 - zr0 + 1,
        "mw": zc1 - zc0 + 1,
        "corner": corner,
        "color": color,
    }


def _missing_slice(info):
    r0, r1, c0, c1 = info["bbox"]
    mh, mw = info["mh"], info["mw"]
    if info["corner"] == "ul":
        return slice(r0, r0 + mh), slice(c0, c0 + mw)
    if info["corner"] == "ur":
        return slice(r0, r0 + mh), slice(c1 - mw + 1, c1 + 1)
    if info["corner"] == "ll":
        return slice(r1 - mh + 1, r1 + 1), slice(c0, c0 + mw)
    return slice(r1 - mh + 1, r1 + 1), slice(c1 - mw + 1, c1 + 1)


def main(grid: np.ndarray) -> np.ndarray:
    """Move each lower complementary rectangle into the matching missing corner above.

    Input structure:
    The grid has black background and one full-width grey horizontal separator row. Above the row are colored L-shaped objects, each equal to a solid rectangle with one rectangular corner removed. Below the row are same-count colored solid rectangles; each lower rectangle is the exact size of one upper missing corner, and matching by this shape is unique.

    Rule:
    Clear the grey separator row to black. For every upper L-shape, identify the missing corner rectangle and find the lower solid rectangle with the same height and width. Clear that lower rectangle from its original position, then paint it into the missing corner of the upper shape, preserving the upper color and the moved lower color.

    Key insight:
    The correspondence is determined by complementary geometry rather than color or position; after pairing, the missing corner direction determines where the lower piece is inserted.
    """
    g = np.asarray(grid)
    out = np.array(g, copy=True)
    sep = _separator_row(g)
    out[sep, :] = BLACK

    mask = g != BLACK
    mask[sep, :] = False

    top_mask = np.zeros_like(mask, dtype=bool)
    top_mask[:sep, :] = mask[:sep, :]
    bottom_mask = np.zeros_like(mask, dtype=bool)
    bottom_mask[sep + 1 :, :] = mask[sep + 1 :, :]

    tops = []
    for coords in _components(top_mask):
        info = _l_component(g, coords)
        if info is not None:
            tops.append(info)

    bottoms_by_size = {}
    for coords in _components(bottom_mask):
        info = _rect_component(g, coords)
        if info is None:
            continue
        bottoms_by_size.setdefault((info["h"], info["w"]), []).append(info)

    for pieces in bottoms_by_size.values():
        pieces.sort(key=lambda x: x["bbox"])

    for top in sorted(tops, key=lambda x: x["bbox"]):
        key = (top["mh"], top["mw"])
        pieces = bottoms_by_size.get(key, [])
        if not pieces:
            continue
        bottom = pieces.pop(0)
        for r, c in bottom["coords"]:
            out[r, c] = BLACK
        rs, cs = _missing_slice(top)
        out[rs, cs] = bottom["color"]

    return out


def _can_place(reserved, r, c, h, w):
    rows, cols = reserved.shape
    r0, r1 = max(0, r - 1), min(rows, r + h + 1)
    c0, c1 = max(0, c - 1), min(cols, c + w + 1)
    return not np.any(reserved[r0:r1, c0:c1])


def _place_box(reserved, h, w):
    rows, cols = reserved.shape
    if h > rows or w > cols:
        return None
    candidates = [(r, c) for r in range(rows - h + 1) for c in range(cols - w + 1)]
    random.shuffle(candidates)
    for r, c in candidates:
        if _can_place(reserved, r, c, h, w):
            reserved[r : r + h, c : c + w] = True
            return r, c
    return None


def _erase_corner(grid, r, c, h, w, mh, mw, corner):
    if corner == "ul":
        grid[r : r + mh, c : c + mw] = BLACK
    elif corner == "ur":
        grid[r : r + mh, c + w - mw : c + w] = BLACK
    elif corner == "ll":
        grid[r + h - mh : r + h, c : c + mw] = BLACK
    else:
        grid[r + h - mh : r + h, c + w - mw : c + w] = BLACK


def _random_params(n, top_h, width):
    params = []
    used_missing = set()
    corners = random.sample(CORNERS, n)
    for i in range(n):
        for _ in range(200):
            h = random.randint(3, min(6, top_h))
            w = random.randint(3, min(7, width))
            mh = random.randint(1, h - 1)
            mw = random.randint(1, w - 1)
            if (mh, mw) in used_missing:
                continue
            used_missing.add((mh, mw))
            params.append((h, w, mh, mw, corners[i]))
            break
        else:
            return None
    return params


def generate_input() -> np.ndarray:
    for _ in range(1000):
        n = random.randint(2, 4)
        width = random.randint(max(18, n * 5), 30)
        top_h = random.randint(8, 14)
        bottom_h = random.randint(6, 14)
        if top_h + 1 + bottom_h > 30:
            continue

        params = _random_params(n, top_h, width)
        if params is None:
            continue

        grid = np.zeros((top_h + 1 + bottom_h, width), dtype=int)
        grid[top_h, :] = GREY
        top_reserved = np.zeros((top_h, width), dtype=bool)
        bottom_reserved = np.zeros((bottom_h, width), dtype=bool)

        colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], 2 * n)
        ok = True
        for i, (h, w, mh, mw, corner) in enumerate(params):
            top_pos = _place_box(top_reserved, h, w)
            bottom_pos = _place_box(bottom_reserved, mh, mw)
            if top_pos is None or bottom_pos is None:
                ok = False
                break

            tr, tc = top_pos
            br, bc = bottom_pos
            grid[tr : tr + h, tc : tc + w] = colors[i]
            _erase_corner(grid, tr, tc, h, w, mh, mw, corner)

            abs_br = top_h + 1 + br
            grid[abs_br : abs_br + mh, bc : bc + mw] = colors[n + i]

        if ok:
            return grid

    raise RuntimeError("could not generate a valid puzzle input")
