import numpy as np


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    H, W = mask.shape
    vis = np.zeros((H, W), dtype=np.uint8)
    out = []
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or vis[r, c]:
                continue
            q = [(r, c)]
            vis[r, c] = 1
            comp = []
            while q:
                x, y = q.pop()
                comp.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < H and 0 <= ny < W and mask[nx, ny] and not vis[nx, ny]:
                        vis[nx, ny] = 1
                        q.append((nx, ny))
            out.append(comp)
    return out


def _rotate_matrix_clockwise(mat: np.ndarray, k: int) -> np.ndarray:
    # k in {0,1,2,3}
    return np.rot90(mat, -k)


def _rotate_coord(r, c, h, w, k):
    k = k % 4
    if k == 0:
        return r, c, h, w
    if k == 1:
        return c, h - 1 - r, w, h
    if k == 2:
        return h - 1 - r, w - 1 - c, h, w
    return w - 1 - c, r, w, h


def _fit_pack(pieces: list[tuple[int, int, int, np.ndarray]], H: int, W: int):
    # pieces: list[(h,w,color,mat)] excluding anchored fixed at (0,0)
    occ = -np.ones((H, W), dtype=int)
    placements = []

    # anchor piece assumed first
    a_h, a_w, _, a_mat = pieces[0]
    if a_h > H or a_w > W:
        return None
    occ[:a_h, :a_w] = 0
    placements.append((0, 0, 0, a_h, a_w, a_mat))

    rem = list(range(1, len(pieces)))

    def first_empty():
        for r in range(H):
            for c in range(W):
                if occ[r, c] < 0:
                    return r, c
        return None

    def can_place(r, c, h, w):
        if r + h > H or c + w > W:
            return False
        for rr in range(r, r + h):
            if np.any(occ[rr, c:c + w] >= 0):
                return False
        return True

    def dfs():
        p = first_empty()
        if p is None:
            return True
        r0, c0 = p
        rem.sort(key=lambda idx: -(pieces[idx][0] * pieces[idx][1]))
        for k in range(len(rem)):
            idx = rem.pop(k)
            h, w, col, mat = pieces[idx]
            variants = [(h, w, mat)]
            if h != w:
                variants.append((w, h, np.rot90(mat)))
            for hh, ww, mm in variants:
                if not can_place(r0, c0, hh, ww):
                    continue
                for rr in range(r0, r0 + hh):
                    occ[rr, c0:c0 + ww] = idx
                placements.append((idx, r0, c0, hh, ww, mm))
                if dfs():
                    return True
                placements.pop()
                for rr in range(r0, r0 + hh):
                    occ[rr, c0:c0 + ww] = -1
            rem.insert(k, idx)
        return False

    if dfs():
        return placements
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Input:
        Scattered rectangular sprites with one magenta(6) marker corner pixel.
    Output:
        Sprites packed to a full rectangle, ranked row-major, then alternately shifted.
    Idea:
        Detect anchor by marker, choose rotation to place marker at top-left of anchor,
        exact-pack all rectangles into a bounded rectangle, sort by row-major anchors,
        shift odd index left and even index right by one.
    Complexity:
        Backtracking on tiny piece count, at most few dozen states per 2D cell.
    """
    inp = np.array(grid, dtype=int)
    comps = _components(inp != 0)
    if not comps:
        return inp.copy()

    anchor_idx = -1
    anchor_marker = None
    for i, comp in enumerate(comps):
        for p in comp:
            if inp[p[0], p[1]] == 6:
                anchor_idx = i
                anchor_marker = p
                break
        if anchor_idx >= 0:
            break
    if anchor_idx < 0:
        return inp.copy()

    # build piece info with matrices
    pieces: list[tuple[int, int, int, np.ndarray, tuple[int, int]]] = []
    marker_info = None
    for i, comp in enumerate(comps):
        rr = [p[0] for p in comp]
        cc = [p[1] for p in comp]
        r0, c0 = min(rr), min(cc)
        h, w = max(rr) - r0 + 1, max(cc) - c0 + 1
        mat = np.zeros((h, w), dtype=int)
        for r, c in comp:
            mat[r - r0, c - c0] = inp[r, c]
        if i == anchor_idx:
            mr, mc = anchor_marker[0] - r0, anchor_marker[1] - c0
            marker_info = (mr, mc, h, w, mat)
        pieces.append((h, w, inp[comp[0][0], comp[0][1]], mat, (r0, c0)))

    # orient anchor so marker at (0,0)
    mh, mw, mat = marker_info[:2] + (marker_info[3],)
    mh0, mw0 = mh, mw
    mr, mc = marker_info[0], marker_info[1]
    anchor_mat = pieces[anchor_idx][3]
    anchor_oriented = None
    anchor_h = anchor_w = 0
    for k in range(4):
        mm = _rotate_matrix_clockwise(pieces[anchor_idx][3], k)
        rr, cc = _rotate_coord(mr, mc, anchor_mat.shape[0], anchor_mat.shape[1], k)[:2]
        if rr == 0 and cc == 0:
            anchor_oriented = mm
            anchor_h, anchor_w = mm.shape
            break

    if anchor_oriented is None:
        return inp.copy()

    # prepare pieces as oriented rectangles
    desc = [(anchor_h, anchor_w, pieces[anchor_idx][2], anchor_oriented)]
    for i, (h, w, col, mat, _) in enumerate(pieces):
        if i == anchor_idx:
            continue
        desc.append((h, w, col, mat))

    total_area = sum(h * w for h, w, _, _ in desc)

    found_pack = None
    for H in range(max(h for h, _, _, _ in desc), total_area + 1):
        if total_area % H != 0:
            continue
        W = total_area // H
        if W < 3 or W > 30:
            continue
        if W < max(h for h, _, _, _ in desc):
            continue
        placements = _fit_pack(desc, H, W)
        if placements is not None:
            found_pack = (H, W, placements, desc)
            break

    if found_pack is None:
        return inp.copy()

    H, W, placements, desc = found_pack
    rank_items = [(r, c, idx) for (idx, r, c, h, w, mm) in placements]
    rank_items.sort(key=lambda x: (x[0], x[1]))

    out = np.zeros((H, W), dtype=int)
    placement_map = {idx: (r, c, mm) for idx, r, c, hh, ww, mm in placements}

    for rank, (r, c, idx) in enumerate(rank_items, start=1):
        h, w, col, mm = desc[idx]
        shift = -1 if (rank % 2 == 1) else 1
        rr0, cc0, pm = placement_map[idx]
        for i in range(mm.shape[0]):
            for j in range(mm.shape[1]):
                if mm[i, j] == 0:
                    continue
                nr = rr0 + i
                nc = cc0 + j + shift
                if 0 <= nr < H and 0 <= nc < W:
                    out[nr, nc] = int(mm[i, j])
    return out


def generate_input() -> np.ndarray:
    """
    Random generator for task 7.
    Returns:
        Scattered rectangular sprites with one marker in anchor sprite corner.
    """
    if not hasattr(generate_input, '_rng'):
        generate_input._rng = np.random.RandomState(7707)
        generate_input._i = 0
    rs = generate_input._rng
    generate_input._i += 1

    colors = [1, 2, 3, 4, 8]

    for _case in range(200):
        H = int(14 + rs.randint(0, 6))
        W = int(14 + rs.randint(0, 8))
        out = np.zeros((H, W), dtype=int)
        used = np.zeros((H, W), dtype=bool)
        rects = []
        n = int(3 + rs.randint(0, 3))

        for i in range(n):
            ph = int(rs.randint(2, 5))
            pw = int(rs.randint(2, 6))
            placed = False
            for _ in range(200):
                r = int(rs.randint(0, H - ph + 1))
                c = int(rs.randint(0, W - pw + 1))
                halo = used[max(0, r - 1): min(H, r + ph + 1),
                            max(0, c - 1): min(W, c + pw + 1)]
                if halo.any():
                    continue
                color = colors[i % len(colors)]
                out[r:r + ph, c:c + pw] = color
                used[r:r + ph, c:c + pw] = True
                rects.append((r, c, ph, pw))
                placed = True
                break
            if not placed:
                break

        if len(rects) != n:
            continue

        ridx = int(rs.randint(0, len(rects)))
        r, c, ph, pw = rects[ridx]
        corner = int(rs.randint(0, 4))
        marker = [
            (r, c),
            (r, c + pw - 1),
            (r + ph - 1, c),
            (r + ph - 1, c + pw - 1),
        ][corner]
        out[marker] = 6
        return out

    fallback = np.zeros((16, 16), dtype=int)
    fallback[1:4, 1:4] = 1
    fallback[6:9, 3:7] = 2
    fallback[2:6, 10:13] = 3
    fallback[1, 1] = 6
    return fallback
