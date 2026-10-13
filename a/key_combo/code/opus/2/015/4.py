import numpy as np

_RNG = np.random.default_rng(20260707)

def _rng(seed=None):
    return np.random.default_rng(seed) if seed is not None else _RNG


def _segment_indices(mask):
    segs = []
    i = 0
    n = len(mask)
    while i < n:
        if mask[i]:
            i += 1
            continue
        j = i
        while j < n and not mask[j]:
            j += 1
        segs.append((i, j))
        i = j
    return segs


def _find_regions(top, sep_row):
    top_area = top
    row_mask = np.all(top_area == 0, axis=1)
    col_mask = np.all(top_area == 0, axis=0)
    row_segments = [(a, b) for a, b in _segment_indices(row_mask) if b - a >= 4]
    col_segments = [(a, b) for a, b in _segment_indices(col_mask) if b - a >= 4]
    regions = []

    for rs in row_segments:
        for cs in col_segments:
            r0, r1 = rs
            c0, c1 = cs
            if r1 - r0 != 4 or c1 - c0 != 4:
                continue
            sub = top_area[r0:r1, c0:c1]
            if np.any(sub == 2):
                regions.append((r0, c0, sub))

    if len(regions) != 6:
        # 兜底扫描（用于轻微生成偏移）
        h, w = top_area.shape
        seen = set()
        for r in range(h - 3):
            for c in range(w - 3):
                sub = top_area[r:r + 4, c:c + 4]
                if np.any(sub == 2) and np.all(np.isin(sub, [0, 2])):
                    key = (r, c)
                    if key not in seen:
                        seen.add(key)
                        regions.append((r, c, sub))
        regions = [x for x in regions if x[2].shape == (4, 4)]
        regions = sorted(regions, key=lambda x: (x[0], x[1]))[:6]

    return regions[:6]


def _triple_from_masks(masks):
    n = len(masks)
    rels = [m for m in masks]
    for ci in range(n):
        cset = rels[ci]
        for ai in range(n):
            if ai == ci:
                continue
            for bi in range(ai + 1, n):
                if bi == ci:
                    continue
                uset = rels[ai] | rels[bi]
                if uset != cset:
                    continue
                if not (rels[ai] - rels[bi]):
                    continue
                if not (rels[bi] - rels[ai]):
                    continue
                return ai, bi, ci
    return None


def main(grid):
    """
1) 先识别上下灰色分隔行，上方是6个红色掩码模板，下方是待印章区。
2) 在6个4x4掩码中找唯一三元组 (A,B,C): A|B == C。
3) 按题意用青/绿重涂 A,B,C，取 C 的相对形状作为印章模板。
4) 在下方对每个4x4窗口进行形状匹配，匹配时将模板黑位点涂为绿返回。
"""

    g = np.array(grid, dtype=int, copy=True)
    h, w = g.shape

    if h < 2:
        return g
    gray_row = np.where(np.sum(g == 5, axis=1) == w)[0]
    if len(gray_row) == 0:
        return g
    sep = int(gray_row[0])

    top = g[:sep]
    regions = _find_regions(top, sep)
    if len(regions) != 6:
        return g

    masks = []
    for r0, c0, sub in regions:
        rs = set()
        for r in range(4):
            for c in range(4):
                if sub[r, c] == 2:
                    rs.add((r, c))
        masks.append(rs)

    triple = _triple_from_masks(masks)
    if triple is None:
        return g

    a_idx, b_idx, c_idx = triple
    for idx, region in enumerate(regions):
        r0, c0, _ = region
        for r in range(4):
            for c in range(4):
                rr, cc = r0 + r, c0 + c
                if idx in (a_idx, b_idx):
                    if g[rr, cc] == 2:
                        g[rr, cc] = 8
                elif idx == c_idx:
                    if g[rr, cc] == 2:
                        g[rr, cc] = 3

    template = masks[c_idx]

    lower = g[sep + 1 :]
    lh = lower.shape[0]
    for r in range(lh - 3):
        for c in range(w - 3):
            ok = True
            for rr in range(4):
                for cc in range(4):
                    rrg = r + rr
                    ccg = c + cc
                    if (rr, cc) in template:
                        if lower[rrg, ccg] != 0:
                            ok = False
                            break
                    else:
                        if lower[rrg, ccg] != 5:
                            ok = False
                            break
                if not ok:
                    break
            if not ok:
                continue
            for rr in range(4):
                for cc in range(4):
                    if (rr, cc) in template:
                        lower[r + rr, c + cc] = 3

    g[sep + 1 :] = lower
    return g


def _random_connected_pattern(rng, size):
    if size < 1:
        size = 1
    cells = {(0, 0)}
    while len(cells) < size:
        x, y = list(cells)[rng.integers(len(cells))]
        neigh = []
        if x > 0:
            neigh.append((x - 1, y))
        if x < 3:
            neigh.append((x + 1, y))
        if y > 0:
            neigh.append((x, y - 1))
        if y < 3:
            neigh.append((x, y + 1))
        nx, ny = neigh[rng.integers(len(neigh))]
        cells.add((nx, ny))
    return cells


def _count_triples(masks):
    n = len(masks)
    found = []
    for ci in range(n):
        c = masks[ci]
        for ai in range(n):
            if ai == ci:
                continue
            for bi in range(ai + 1, n):
                if bi == ci:
                    continue
                if c == masks[ai] | masks[bi]:
                    if (masks[ai] - masks[bi]) and (masks[bi] - masks[ai]):
                        found.append((ai, bi, ci))
    return found


def generate_input(seed=None):
    rng = _rng(seed)
    H, W = 15, 20
    sep = 9

    while True:
        top_positions = [(0, 0), (0, 5), (0, 10), (5, 0), (5, 5), (5, 10)]
        grid = np.zeros((H, W), dtype=int)

        grid[sep, :] = 5
        for r in range(sep + 1, H):
            grid[r, :] = 5

        masks = None
        for _ in range(500):
            trial = []
            sizes = [int(rng.integers(1, 9)) for _ in range(6)]
            for i in range(6):
                trial.append(_random_connected_pattern(rng, sizes[i]))
            # ensure one explicit union triple by override
            a, b = 0, 1
            trial[2] = trial[a] | trial[b]
            # force non-empty and size limit
            if len(trial[2]) == 0:
                continue

            # regenerate remaining to avoid conflict too often
            for i in [3, 4, 5]:
                if len(trial[i]) == 0:
                    trial[i] = _random_connected_pattern(rng, int(rng.integers(1, 9)))

            triples = _count_triples(trial)
            if len(triples) != 1:
                continue
            masks = trial
            break

        if masks is None:
            continue

        # 放置上区掩码
        for (r0, c0), m in zip(top_positions, masks):
            for rr, cc in m:
                grid[r0 + rr, c0 + cc] = 2

        lower_top = sep + 1
        # 预留两个匹配位和一个不匹配位
        template = masks[2]
        match1 = (0, 2)
        match2 = (1, 6)
        mismatch = (0, 13)

        # 先清理下区成灰框架，内部可见4x4槽位
        # 置灰后再刻出黑槽
        for r in range(lower_top, H):
            grid[r, 2:18] = 5

        for (rs, cs) in (match1, match2, mismatch):
            base = r + 0
            for rr in range(4):
                for cc in range(4):
                    rrg = lower_top + rs + rr
                    ccg = cs + cc
                    if 0 <= rrg < H and 0 <= ccg < W:
                        grid[rrg, ccg] = 5
            for rr, cc in template:
                rrg = lower_top + rs + rr
                ccg = cs + cc
                grid[rrg, ccg] = 0

        # mismatch: 打一个反例
        rr, cc = mismatch
        diff_cells = []
        for dr in range(4):
            for dc in range(4):
                rrg = lower_top + rr + dr
                ccg = cc + dc
                if 0 <= rrg < H and 0 <= ccg < W:
                    if (dr, dc) in template:
                        diff_cells.append((rrg, ccg, 5))
        if diff_cells:
            rr0, cc0, val = diff_cells[0]
            grid[rr0, cc0] = 5 if grid[rr0, cc0] == 0 else 0

        # 需要至少2个匹配3x4黑槽及1个非匹配
        return grid


if __name__ == "__main__":
    pass
