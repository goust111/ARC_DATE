from __future__ import annotations

import numpy as np

BLACK = 0
GREY = 5


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
            stack = [(r0, c0)]
            seen[r0, c0] = True
            pts = []
            while stack:
                r, c = stack.pop()
                pts.append((r, c))
                for rr, cc in _neighbors4(r, c, H, W):
                    if mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        stack.append((rr, cc))
                # tiny typo avoided by explicit condition above
            comps.append(np.array(pts, dtype=int))
    return comps


def _find_frame(grid: np.ndarray):
    H, W = grid.shape
    comps = _components(grid == GREY)
    for comp in comps:
        if len(comp) < 12:
            continue
        r0, c0 = comp[:, 0].min(), comp[:, 1].min()
        r1, c1 = comp[:, 0].max(), comp[:, 1].max()
        h, w = r1 - r0 + 1, c1 - c0 + 1
        if h < 3 or w < 3:
            continue
        perim = 2 * (h + w) - 4
        if len(comp) != perim:
            continue
        ok = True
        for rr in range(r0, r1 + 1):
            if grid[rr, c0] != GREY or grid[rr, c1] != GREY:
                ok = False
                break
        if not ok:
            continue
        for cc in range(c0, c1 + 1):
            if grid[r0, cc] != GREY or grid[r1, cc] != GREY:
                ok = False
                break
        if not ok:
            continue
        return (r0, c0, r1, c1)
    return None


def _count_layers(grid: np.ndarray, frame):
    r0, c0, r1, c1 = frame
    r0i, c0i = r0 + 1, c0 + 1
    r1i, c1i = r1 - 1, c1 - 1
    layers = 0
    while r0i <= r1i and c0i <= c1i:
        vals = []
        vals.extend(grid[r0i, c0i : c1i + 1].tolist())
        vals.extend(grid[r1i, c0i : c1i + 1].tolist())
        vals.extend(grid[r0i + 1 : r1i, c0i].tolist())
        vals.extend(grid[r0i + 1 : r1i, c1i].tolist())
        vals = [v for v in vals if v != BLACK]
        uniq = sorted(set(vals))
        if len(uniq) != 1:
            break
        layers += 1
        r0i += 1
        c0i += 1
        r1i -= 1
        c1i -= 1
    return layers


def _find_tile_region(grid: np.ndarray, frame):
    H, W = grid.shape
    frame_mask = np.zeros_like(grid, dtype=bool)
    r0, c0, r1, c1 = frame
    frame_mask[r0 : r1 + 1, c0 : c1 + 1] = True
    mask = (grid != BLACK) & ~frame_mask
    comps = _components(mask)
    if not comps:
        return None
    # choose largest component, likely the tile block
    comp = max(comps, key=len)
    r0t, c0t = comp[:, 0].min(), comp[:, 1].min()
    r1t, c1t = comp[:, 0].max(), comp[:, 1].max()
    # only accept rectangular components
    for rr in range(r0t, r1t + 1):
        for cc in range(c0t, c1t + 1):
            if not mask[rr, cc]:
                return None
    return (r0t, c0t, r1t, c1t)


def main(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, copy=True)
    frame = _find_frame(out)
    if frame is None:
        return out

    k = _count_layers(out, frame)
    if k <= 0:
        return out

    tile = _find_tile_region(out, frame)
    if tile is None:
        return out
    t0, l0, t1, l1 = tile
    th = t1 - t0 + 1
    tw = l1 - l0 + 1
    if th % k != 0 or tw % k != 0:
        return out
    hs = th // k
    ws = tw // k

    small = np.zeros((hs, ws), dtype=int)
    for rr in range(hs):
        for cc in range(ws):
            block = out[t0 + rr * k : t0 + (rr + 1) * k, l0 + cc * k : l0 + (cc + 1) * k]
            vals, counts = np.unique(block.ravel(), return_counts=True)
            small[rr, cc] = int(vals[np.argmax(counts)])

    r0i, c0i = frame[0] + 1, frame[1] + 1
    r1i, c1i = frame[2] - 1, frame[3] - 1
    inner_h, inner_w = r1i - r0i + 1, c1i - c0i + 1
    if inner_h != hs or inner_w != ws:
        return out
    out[r0i : r1i + 1, c0i : c1i + 1] = small

    # erase tile region
    out[t0 : t1 + 1, l0 : l1 + 1] = BLACK
    return out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    out = np.zeros((30, 30), dtype=int)

    k = int(rng.choice([2, 3, 4]))
    ih = int(rng.integers(max(3, k), 7))
    iw = int(rng.integers(max(3, k), 7))
    frame_r0, frame_c0 = 1, 1
    frame_r1, frame_c1 = frame_r0 + ih + 1, frame_c0 + iw + 1

    # concentric rectangles in frame interior with k layers
    for rr in range(frame_r0, frame_r1 + 1):
        out[rr, frame_c0] = GREY
        out[rr, frame_c1] = GREY
    for cc in range(frame_c0, frame_c1 + 1):
        out[frame_r0, cc] = GREY
        out[frame_r1, cc] = GREY

    colors = [1, 2, 3, 4]
    for layer in range(k):
        color = colors[layer]
        r0, c0 = frame_r0 + 1 + layer, frame_c0 + 1 + layer
        r1, c1 = frame_r1 - 1 - layer, frame_c1 - 1 - layer
        if r0 > r1 or c0 > c1:
            break
        for c in range(c0, c1 + 1):
            out[r0, c] = color
            out[r1, c] = color
        for r in range(r0 + 1, r1):
            if c0 <= c1:
                out[r, c0] = color
                out[r, c1] = color

    tile_h, tile_w = ih * k, iw * k
    candidates = [
        (frame_r0, frame_c1 + 2),
        (frame_r1 + 2, frame_c0),
        (30 - tile_h - 1, 30 - tile_w - 1),
    ]
    valid = [(r, c) for r, c in candidates if 0 <= r and 0 <= c and r + tile_h <= 30 and c + tile_w <= 30]
    if not valid:
        tr, tc = 10, 10
    else:
        tr, tc = valid[int(rng.integers(0, len(valid)))]

    low = rng.integers(0, 4, size=(tile_h // k, tile_w // k))
    for rr in range(tile_h):
        for cc in range(tile_w):
            out[tr + rr, tc + cc] = int(low[rr // k, cc // k]) + 1

    return out
