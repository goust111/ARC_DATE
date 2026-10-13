import random
from collections import Counter

import numpy as np


def _bbox(mask):
    rs, cs = np.where(mask)
    return int(rs.min()), int(cs.min()), int(rs.max()) + 1, int(cs.max()) + 1


def _reduce_scaled(arr):
    h, w = arr.shape
    best = 1
    for s in range(2, min(h, w) + 1):
        if h % s or w % s:
            continue
        ok = True
        for r in range(0, h, s):
            for c in range(0, w, s):
                block = arr[r:r + s, c:c + s]
                if np.any(block != block[0, 0]):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            best = s
    if best == 1:
        return arr.copy(), 1
    small = np.zeros((h // best, w // best), dtype=arr.dtype)
    for r in range(small.shape[0]):
        for c in range(small.shape[1]):
            small[r, c] = arr[r * best, c * best]
    return small, best


def _scale(arr, s):
    return np.kron(arr, np.ones((s, s), dtype=arr.dtype))


def _stripe_bands(g):
    dom = []
    for c in range(g.shape[1]):
        vals = [int(x) for x in g[:, c] if x not in (5, 9)]
        dom.append(Counter(vals).most_common(1)[0][0])
    bands = []
    start = 0
    for c in range(1, len(dom) + 1):
        if c == len(dom) or dom[c] != dom[start]:
            bands.append((start, c, dom[start]))
            start = c
    return bands


def _sprite_info(band):
    mask = np.isin(band, [5, 9])
    r0, c0, r1, c1 = _bbox(mask)
    arr = band[r0:r1, c0:c1].copy()
    arr[~np.isin(arr, [5, 9])] = 0
    base, scale = _reduce_scaled(arr)
    return {
        "bbox": (r0, c0, r1, c1),
        "arr": arr,
        "base": base,
        "scale": scale,
    }


def _anchor_min(arr):
    rs, cs = np.where(arr == 5)
    return int(rs.min()), int(cs.min())


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, copy=True)
    bands = _stripe_bands(g)
    infos = []
    for c0, c1, color in bands:
        band = g[:, c0:c1].copy()
        info = _sprite_info(band)
        info.update({"c0": c0, "c1": c1, "color": color, "band": band})
        infos.append(info)

    unique = []
    for info in sorted(infos, key=lambda x: (x["base"].size, x["bbox"][0], x["c0"])):
        if not any(np.array_equal(info["base"], u) for u in unique):
            unique.append(info["base"])
        if len(unique) == 2:
            break
    if len(unique) < 2:
        return g

    for info in infos:
        info["group"] = 0 if np.array_equal(info["base"], unique[0]) else 1

    ordered = [x for x in infos if x["group"] == 0] + [x for x in infos if x["group"] == 1]
    out_bands = []
    for info in ordered:
        band = info["band"].copy()
        band[np.isin(band, [5, 9])] = info["color"]
        old_r0, old_c0, _, _ = info["bbox"]
        old_anchor = _anchor_min(info["arr"])
        replacement_base = unique[1 - info["group"]]
        repl = _scale(replacement_base, info["scale"])
        new_anchor = _anchor_min(repl)
        top = old_r0 + old_anchor[0] - new_anchor[0]
        left = old_c0 + old_anchor[1] - new_anchor[1]
        for r in range(repl.shape[0]):
            for c in range(repl.shape[1]):
                if repl[r, c] != 0:
                    rr, cc = top + r, left + c
                    if 0 <= rr < band.shape[0] and 0 <= cc < band.shape[1]:
                        band[rr, cc] = repl[r, c]
        out_bands.append(band)
    return np.concatenate(out_bands, axis=1)


P = np.array([[5, 5, 0], [9, 5, 9]], dtype=int)
Q = np.array([[5, 0, 9], [5, 9, 9]], dtype=int)


def generate_input() -> np.ndarray:
    n = random.choice([4, 5])
    width = 6
    h = random.choice([11, 12, 13])
    g = np.zeros((h, n * width), dtype=int)
    stripe_colors = random.sample([1, 2, 3, 4, 6, 7, 8], n)
    groups = [0, 1] + [random.randrange(2) for _ in range(n - 2)]
    random.shuffle(groups)
    for i in range(n):
        c0 = i * width
        g[:, c0:c0 + width] = stripe_colors[i]
        base = P if groups[i] == 0 else Q
        scale = random.choice([1, 2])
        sprite = _scale(base, scale)
        top = random.randint(1, h - sprite.shape[0] - 1)
        left = random.randint(0, width - sprite.shape[1])
        for r in range(sprite.shape[0]):
            for c in range(sprite.shape[1]):
                if sprite[r, c] != 0:
                    g[top + r, c0 + left + c] = sprite[r, c]
    return g
