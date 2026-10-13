import numpy as np

RNG = np.random.default_rng(808)


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
            comps.append(comp)
    return comps


def _rect_for_color(g: np.ndarray, color: int):
    comps = _components(g == color)
    if len(comps) != 2:
        return None, 0

    s0 = set(map(tuple, comps[0]))
    s1 = set(map(tuple, comps[1]))
    touches_left_0 = any(c == 0 for _, c in s0)
    touches_left_1 = any(c == 0 for _, c in s1)

    if touches_left_0 and not touches_left_1:
        move_set, anchor_set = s0, s1
    elif touches_left_1 and not touches_left_0:
        move_set, anchor_set = s1, s0
    else:
        return None, 0

    h, w = g.shape
    total = len(move_set) + len(anchor_set)
    found = None

    move_list = list(move_set)
    for dr in range(-h, h + 1):
        for dc in range(-w, w + 1):
            moved = {(r + dr, c + dc) for r, c in move_list}
            if any(r < 0 or c < 0 or r >= h or c >= w for r, c in moved):
                continue
            union = anchor_set | moved
            rs = [p[0] for p in union]
            cs = [p[1] for p in union]
            tr, br = min(rs), max(rs)
            lc, rc = min(cs), max(cs)
            if (br - tr + 1) * (rc - lc + 1) != len(union):
                continue
            if (br - tr + 1) * (rc - lc + 1) != total:
                continue
            found = (tr, br, lc, rc)
            break
        if found is not None:
            break

    return found, total


def main(grid: np.ndarray) -> np.ndarray:
    """Task 8: overlap-aware complementary rectangles with cross conflict.

Input:
    Two colors each with two fragments, one touching left edge.
Output:
    Two filled rectangles by area; overlap cells resolved by center-cross pattern.

Reasoning:
    Recover each rectangle from moving+anchor pair, fill independently,
    then resolve overlap by lines through each rectangle center.

Complexity:
    O(H*W).
    """

    g = np.array(grid, dtype=int)
    h, w = g.shape
    out = np.zeros((h, w), dtype=int)

    rects = {}
    sizes = {}
    for color in (2, 3):
        rect, area = _rect_for_color(g, color)
        if rect is None:
            return g.copy()
        rects[color] = rect
        sizes[color] = area

    r1t, r1b, r1l, r1r = rects[2]
    r2t, r2b, r2l, r2r = rects[3]

    out[r1t : r1b + 1, r1l : r1r + 1] = 2
    out[r2t : r2b + 1, r2l : r2r + 1] = 3

    top = max(r1t, r2t)
    left = max(r1l, r2l)
    bottom = min(r1b, r2b)
    right = min(r1r, r2r)

    if top <= bottom and left <= right:
        c1r, c1c = (r1t + r1b) // 2, (r1l + r1r) // 2
        c2r, c2c = (r2t + r2b) // 2, (r2l + r2r) // 2
        winner = 2 if sizes[2] >= sizes[3] else 3
        loser = 3 if winner == 2 else 2

        for rr in range(top, bottom + 1):
            for cc in range(left, right + 1):
                if rr == c1r or cc == c1c or rr == c2r or cc == c2c:
                    out[rr, cc] = winner
                else:
                    out[rr, cc] = loser

    return out


def generate_input() -> np.ndarray:
    h = 16
    w = 16
    for _ in range(200):
        g = np.zeros((h, w), dtype=int)
        rect_meta = {}

        for color in (2, 3):
            for _ in range(200):
                r_h = int(RNG.integers(4, 7))
                r_w = int(RNG.integers(5, 9))
                split = int(RNG.integers(1, r_w - 1))
                move_w = split
                anchor_w = r_w - split
                if anchor_w <= 0:
                    continue
                top = int(RNG.integers(0, h - r_h))
                anchor_left = int(RNG.integers(1, w - r_w + 1))

                if color == 3 and any(
                    np.any(g[rr, 0:anchor_left] != 0) for rr in range(top, top + r_h)
                ):
                    # keep both fragments mostly isolated before anchoring
                    pass

                move_cols = range(0, move_w)
                anchor_cols = range(anchor_left, anchor_left + anchor_w)

                for rr in range(top, top + r_h):
                    for cc in move_cols:
                        g[rr, cc] = color
                for rr in range(top, top + r_h):
                    for cc in anchor_cols:
                        g[rr, cc] = color

                area = r_h * r_w
                rect_meta[color] = (top, top + r_h - 1, anchor_left, anchor_left + r_w - 1, area)
                break

        if len(rect_meta) != 2:
            continue

        t1, b1, l1, r1, a1 = rect_meta[2]
        t2, b2, l2, r2, a2 = rect_meta[3]
        if a1 == a2:
            continue

        top = max(t1, t2)
        bottom = min(b1, b2)
        left = max(l1, l2)
        right = min(r1, r2)
        if top > bottom or left > right:
            continue
        if (bottom - top + 1) * (right - left + 1) < 4:
            continue
        if (t1 >= t2 and b1 <= b2 and l1 >= l2 and r1 <= r2) or (
            t2 >= t1 and b2 <= b1 and l2 >= l1 and r2 <= r1
        ):
            continue

        return g

    # fallback guaranteed-overlap case
    g = np.zeros((h, w), dtype=int)
    for rr in range(2, 9):
        g[rr, 0] = 2
        g[rr, 1:5] = 2
    for rr in range(3, 11):
        g[rr, 0] = 3
        g[rr, 5:8] = 3
    return g
