import numpy as np
from itertools import permutations


def _find_mapping(a, b):
    colors_a = [int(c) for c in np.unique(a) if c != 0]
    colors_b = [int(c) for c in np.unique(b) if c != 0]
    if not colors_a or len(colors_a) != len(colors_b):
        return {}
    masks_a = [a == c for c in colors_a]
    masks_b = [b == c for c in colors_b]
    for perm in permutations(range(len(colors_b))):
        mapping = {}
        ok = True
        for i, j in enumerate(perm):
            if not np.array_equal(masks_a[i], masks_b[j]):
                ok = False
                break
            mapping[colors_a[i]] = colors_b[j]
        if ok:
            return mapping
    return {}


def main(grid):
    """
    1) Compress upper area by 3x3 blocks: any-red ->2 else ->1.
    2) Detect two isomorphic 3x3 tiles in lower area and infer color mapping.
    3) Recolor compressed 3x3 map using mapping and output 3x3.
    4) Return mapped map only.
    """
    g = np.array(grid, dtype=int)
    h, w = g.shape
    rows = [i for i in range(h) if np.all(g[i] == 6)]
    if not rows:
        return g
    sep = int(rows[0])
    top = g[:sep, :]
    bot = g[sep + 1 :, :]

    if top.shape[0] % 3 != 0 or top.shape[1] % 3 != 0:
        return g
    bh = top.shape[0] // 3
    bw = top.shape[1] // 3

    comp = np.zeros((3, 3), dtype=int)
    for r in range(3):
        for c in range(3):
            blk = top[r * bh : (r + 1) * bh, c * bw : (c + 1) * bw]
            comp[r, c] = 2 if np.any(blk == 2) else 1

    # locate three tiles at fixed layout
    starts = []
    for c in (0, 4, 8):
        if bot.shape[1] >= c + 3 and np.any(bot[:, c : c + 3] != 0):
            starts.append(c)
    if len(starts) != 3:
        return comp

    # one row tiles (size 3)
    rs = 0
    while rs + 2 < bot.shape[0] and not np.any(bot[rs : rs + 3] != 0):
        rs += 1
        if rs + 3 > bot.shape[0]:
            return comp
    tiles = [bot[rs : rs + 3, c : c + 3] for c in starts]
    if any(t.shape != (3, 3) for t in tiles):
        return comp

    mapping = {}
    for i in range(3):
        for j in range(i + 1, 3):
            m = _find_mapping(tiles[i], tiles[j])
            if m and set(m.keys()) == {1, 2}:
                mapping = m
                break
            m = _find_mapping(tiles[j], tiles[i])
            if m and set(m.keys()) == {1, 2}:
                mapping = m
                break
        if mapping:
            break

    if not mapping:
        mapping = {1: 1, 2: 2}

    out = np.vectorize(lambda v: mapping.get(int(v), int(v)))(comp)
    return out


def generate_input(seed=0, variant=None):
    if variant is None:
        if not hasattr(generate_input, "_variant"):
            generate_input._variant = 0
        variant = generate_input._variant
        generate_input._variant += 1
    rng = np.random.default_rng(int(seed) * 1000003 + int(variant) * 23 + 29)
    top_h = 9
    top_w = 12
    H = 13
    W = 12
    g = np.zeros((H, W), dtype=int)

    top = np.ones((top_h, top_w), dtype=int)
    for y in range(top_h):
        for x in range(top_w):
            if rng.random() < 0.16:
                top[y, x] = 2
    g[:top_h, :] = top

    sep = top_h
    g[sep, :] = 6

    # shape and pair colors
    shape = np.array(
        [[1, 2, 1],
         [2, 1, 2],
         [1, 2, 1]],
        dtype=int,
    )
    if rng.random() < 0.5:
        src_colors = (1, 2)
        dst_colors = (3, 4)
    else:
        src_colors = (1, 2)
        dst_colors = (4, 3)

    tile_a = np.where(shape == 1, src_colors[0], src_colors[1])
    if rng.random() < 0.5:
        tile_b = np.where(shape == 1, dst_colors[0], dst_colors[1])
    else:
        tile_b = np.where(shape == 1, dst_colors[1], dst_colors[0])

    tile_c = np.array(
        [[0, 6, 0],
         [6, 0, 6],
         [0, 6, 0]],
        dtype=int,
    )
    if rng.random() < 0.5:
        order = [tile_a, tile_b, tile_c]
    else:
        order = [tile_b, tile_a, tile_c]

    r0 = sep + 1
    g[r0 : r0 + 3, 0:3] = order[0]
    g[r0 : r0 + 3, 4:7] = order[1]
    g[r0 : r0 + 3, 8:11] = order[2]

    return g
