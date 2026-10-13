import numpy as np

RNG = np.random.default_rng(505)

SPRITE_TEMPLATES = {
    3: np.array(
        [
            [0, 5, 5],
            [5, 5, 5],
            [0, 5, 0],
        ],
        dtype=int,
    ),
    4: np.array(
        [
            [0, 5, 5, 0],
            [5, 5, 0, 5],
            [5, 0, 5, 5],
            [0, 5, 5, 0],
        ],
        dtype=int,
    ),
}


def _components(mask: np.ndarray):
    h, w = mask.shape
    vis = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or vis[r, c]:
                continue
            st = [(r, c)]
            vis[r, c] = True
            comp = []
            while st:
                x, y = st.pop()
                comp.append((x, y))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = x + dr, y + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not vis[nr, nc]:
                        vis[nr, nc] = True
                        st.append((nr, nc))
            comps.append(np.array(comp, dtype=int))
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """Task 5: diagonal marker rectangles and upscaled sprite stenciled stamping.

Input:
    Gray sprite near top-left and marker pairs with diagonal corners.
Output:
    Each rectangle receives a tiled stencil copy derived from scaled sprite mask.

Reasoning:
    Detect sprite template from the gray component, then for each pair-marked rectangle
    detect scale m and stamp the sprite at every gray block of the scaled template.

Complexity:
    O(H*W * m^2 * SH*SW).
    """

    g = np.array(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    comps = _components(g == 5)
    if not comps:
        return out

    comps = sorted(comps, key=lambda c: (c[:, 0].min(), c[:, 1].min()))
    sprite_comp = comps[0]
    r0, r1 = int(sprite_comp[:, 0].min()), int(sprite_comp[:, 0].max())
    c0, c1 = int(sprite_comp[:, 1].min()), int(sprite_comp[:, 1].max())
    sprite = (g[r0 : r1 + 1, c0 : c1 + 1] == 5).astype(int)
    sh, sw = sprite.shape

    for color in (1, 2, 3):
        pos = list(zip(*np.where(g == color)))
        if len(pos) != 2:
            continue
        (r1m, c1m), (r2m, c2m) = pos
        top, bottom = min(r1m, r2m), max(r1m, r2m)
        left, right = min(c1m, c2m), max(c1m, c2m)
        H = bottom - top + 1
        W = right - left + 1

        if H % sh != 0 or W % sw != 0:
            continue
        m = H // sh
        if m <= 0 or W // sw != m:
            continue

        for sr in range(sh):
            for sc in range(sw):
                if sprite[sr, sc] == 0:
                    continue
                b_top = top + sr * m
                b_left = left + sc * m
                for by in range(m):
                    for bx in range(m):
                        base_r = b_top + by
                        base_c = b_left + bx
                        if base_r > bottom or base_c > right:
                            continue
                        for tr in range(sh):
                            for tc in range(sw):
                                if sprite[tr, tc] == 0:
                                    continue
                                rr = base_r + tr
                                cc = base_c + tc
                                if top <= rr <= bottom and left <= cc <= right and rr < h and cc < w:
                                    out[rr, cc] = 5

    return out


def generate_input() -> np.ndarray:
    h = 20
    w = 20
    g = np.zeros((h, w), dtype=int)

    sw = int(RNG.choice([3, 4]))
    sprite = SPRITE_TEMPLATES[sw]
    sh = sprite.shape[0]
    g[:sh, :sw] = sprite

    n_pairs = int(RNG.integers(2, 4))
    used = set(zip(*np.where(g != 0)))
    colors = RNG.choice([1, 2, 3], size=n_pairs, replace=False)

    for color in colors:
        m = int(RNG.choice([2, 3]))
        rh = sh * m
        rw = sw * m
        placed = False
        for _ in range(500):
            top = int(RNG.integers(3, h - rh + 1))
            left = int(RNG.integers(0, w - rw + 1))
            corners = [(top, left), (top + rh - 1, left + rw - 1)]
            if any((top <= r < top + rh and left <= c < left + rw) for r, c in used):
                continue
            if corners[0] in used or corners[1] in used:
                continue
            g[top, left] = color
            g[top + rh - 1, left + rw - 1] = color
            used.add((top, left))
            used.add((top + rh - 1, left + rw - 1))
            placed = True
            break

        if not placed:
            # fallback inside safe margin
            top = 3
            left = 3
            if left + rw < w and top + rh < h:
                g[top, left] = color
                g[top + rh - 1, left + rw - 1] = color

    return g
