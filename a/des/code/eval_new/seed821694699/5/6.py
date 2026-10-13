import numpy as np
from collections import Counter


def _infer_background(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _dominant(values, fallback: int) -> int:
    if not values:
        return fallback
    return Counter(values).most_common(1)[0][0]


def _edge_scales(grid: np.ndarray, bg: int):
    h, w = grid.shape
    row_scale = np.full(h, bg, dtype=int)
    col_scale = np.full(w, bg, dtype=int)

    for r in range(1, h - 1):
        vals = []
        if grid[r, 0] != bg:
            vals.append(int(grid[r, 0]))
        if grid[r, w - 1] != bg:
            vals.append(int(grid[r, w - 1]))
        if vals:
            row_scale[r] = _dominant(vals, bg)

    for c in range(1, w - 1):
        vals = []
        if grid[0, c] != bg:
            vals.append(int(grid[0, c]))
        if grid[h - 1, c] != bg:
            vals.append(int(grid[h - 1, c]))
        if vals:
            col_scale[c] = _dominant(vals, bg)

    for axis, n in ((row_scale, h), (col_scale, w)):
        for i in range(1, n - 1):
            if axis[i] != bg:
                continue
            d = 1
            while i - d >= 1 or i + d < n - 1:
                if i - d >= 1 and axis[i - d] != bg:
                    axis[i] = axis[i - d]
                    break
                if i + d < n - 1 and axis[i + d] != bg:
                    axis[i] = axis[i + d]
                    break
                d += 1

    row_fb = _dominant([int(v) for v in row_scale if int(v) != bg], bg)
    col_fb = _dominant([int(v) for v in col_scale if int(v) != bg], bg)
    for i in range(1, h - 1):
        if row_scale[i] == bg:
            row_scale[i] = row_fb
    for j in range(1, w - 1):
        if col_scale[j] == bg:
            col_scale[j] = col_fb

    return row_scale, col_scale


def _connected_components(mask: np.ndarray):
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    comps = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue

            stack = [(r, c)]
            visited[r, c] = True
            comp = []

            while stack:
                cr, cc = stack.pop()
                comp.append((cr + 1, cc + 1))

                if cr > 0 and mask[cr - 1, cc] and not visited[cr - 1, cc]:
                    visited[cr - 1, cc] = True
                    stack.append((cr - 1, cc))
                if cr + 1 < h and mask[cr + 1, cc] and not visited[cr + 1, cc]:
                    visited[cr + 1, cc] = True
                    stack.append((cr + 1, cc))
                if cc > 0 and mask[cr, cc - 1] and not visited[cr, cc - 1]:
                    visited[cr, cc - 1] = True
                    stack.append((cr, cc - 1))
                if cc + 1 < w and mask[cr, cc + 1] and not visited[cr, cc + 1]:
                    visited[cr, cc + 1] = True
                    stack.append((cr, cc + 1))

            comps.append(comp)

    return comps


def _find_template_boxes(grid: np.ndarray, bg: int):
    h, w = grid.shape
    mask = grid[1 : h - 1, 1 : w - 1] != bg
    comps = _connected_components(mask)

    best_for_box = {}

    for comp in comps:
        if not comp:
            continue
        rs = [p[0] for p in comp]
        cs = [p[1] for p in comp]
        min_r, max_r = min(rs), max(rs)
        min_c, max_c = min(cs), max(cs)
        comp_size = len(comp)

        if (max_r - min_r + 1) > 4 or (max_c - min_c + 1) > 4:
            continue

        best = None
        best_box = None

        for s in (3, 4):
            top_min = max(1, max_r - s + 1)
            top_max = min(min_r, h - 1 - s)
            left_min = max(1, max_c - s + 1)
            left_max = min(min_c, w - 1 - s)
            if top_min > top_max or left_min > left_max:
                continue

            for top in range(top_min, top_max + 1):
                for left in range(left_min, left_max + 1):
                    block = grid[top : top + s, left : left + s]
                    non_bg = int(np.count_nonzero(block != bg))
                    if non_bg < s * s - 3 or non_bg > s * s:
                        continue

                    missing = s * s - non_bg
                    extra = non_bg - comp_size
                    score = (missing, extra, -non_bg)
                    if best is None or score < best:
                        best = score
                        best_box = (top, left, s)

        if best_box is not None:
            prior = best_for_box.get(best_box)
            if prior is None or best < prior[0]:
                best_for_box[best_box] = (best, comp_size)

    candidates = sorted([(box, info[0]) for box, info in best_for_box.items()], key=lambda x: x[1])

    def overlap(a, b):
        ar, ac, asz = a
        br, bc, bsz = b
        if ar + asz <= br or br + bsz <= ar:
            return False
        if ac + asz <= bc or bc + bsz <= ac:
            return False
        return True

    selected = []
    for box, _ in candidates:
        if any(overlap(box, prev) for prev in selected):
            continue
        selected.append(box)

    return selected


def _infer_box_for_missing(grid: np.ndarray, bg: int, r: int, c: int):
    h, w = grid.shape
    best = None
    best_box = None

    for s in (3, 4):
        top_min = max(1, r - s + 1)
        top_max = min(r, h - 1 - s)
        left_min = max(1, c - s + 1)
        left_max = min(c, w - 1 - s)
        if top_min > top_max or left_min > left_max:
            continue

        for top in range(top_min, top_max + 1):
            for left in range(left_min, left_max + 1):
                block = grid[top : top + s, left : left + s]
                non_bg = int(np.count_nonzero(block != bg))
                if non_bg < s * s - 3 or non_bg > s * s:
                    continue

                score = (s * s - non_bg, -non_bg, top, left)
                if best is None or score < best:
                    best = score
                    best_box = (top, left, s)

    return best_box


def _fill_box(template_ref: np.ndarray, out: np.ndarray, box, bg: int, row_scale: np.ndarray, col_scale: np.ndarray):
    top, left, s = box
    known = set(int(v) for v in template_ref[top : top + s, left : left + s].ravel() if v != bg)
    if not known:
        return

    for r in range(top, top + s):
        for c in range(left, left + s):
            if out[r, c] != bg:
                continue

            row_v = int(row_scale[r])
            col_v = int(col_scale[c])

            if row_v == col_v:
                fill_color = row_v
            else:
                has_row = row_v in known
                has_col = col_v in known
                if has_row:
                    fill_color = row_v
                elif has_col:
                    fill_color = col_v
                else:
                    fill_color = next(iter(known))

            if fill_color == bg:
                fill_color = next(iter(known))
            out[r, c] = fill_color


def main(grid: np.ndarray) -> np.ndarray:
    """Solve the ruler-indexed template completion puzzle.

    Overview
    - Recover every 3x3/4x4 sprite containing a few black holes.
    - Row/column ruler intersections decide candidate colors.

    Inputs
    - grid: np.ndarray using ARC integer colors; border stores ruler colors.

    Returns
    - np.ndarray same size, with template holes inferred and filled.

    Complexity
    - O(H*W) mask pass plus small constant-size window scans in O(H*W*16).
    """
    g = np.asarray(grid)
    g = np.array(g, dtype=int, copy=True)
    h, w = g.shape
    if h < 5 or w < 5:
        return g

    bg = _infer_background(g)
    row_scale, col_scale = _edge_scales(g, bg)

    template_ref = np.array(g, dtype=int, copy=True)
    out = np.array(g, dtype=int, copy=True)

    boxes = _find_template_boxes(template_ref, bg)
    for box in boxes:
        _fill_box(template_ref, out, box, bg, row_scale, col_scale)

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if out[r, c] != bg:
                continue
            box = _infer_box_for_missing(template_ref, bg, r, c)
            if box is None:
                continue
            _fill_box(template_ref, out, box, bg, row_scale, col_scale)

    return out


def _make_prototype(rng: np.random.Generator, s: int, palette: np.ndarray) -> np.ndarray:
    while True:
        proto = rng.choice(palette, size=(s, s)).astype(int)
        if len(np.unique(proto)) >= 2:
            return proto


def _template_missing_ok(candidate: np.ndarray, top: int, left: int, row_scale: np.ndarray, col_scale: np.ndarray, bg: int) -> bool:
    present = set(int(v) for v in candidate.ravel() if v != bg)
    if not present:
        return False
    s = candidate.shape[0]

    for dr in range(s):
        for dc in range(s):
            if candidate[dr, dc] != bg:
                continue
            row_v = int(row_scale[top + dr - 1])
            col_v = int(col_scale[left + dc - 1])
            if row_v != col_v and row_v not in present and col_v not in present:
                return False

    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    bg = 0
    h = int(rng.integers(10, 27))
    w = int(rng.integers(10, 27))

    s = int(rng.integers(3, 5))
    palette = rng.choice(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int), size=int(rng.integers(2, 5)), replace=False)

    proto = _make_prototype(rng, s, palette)

    row_scale = rng.choice(palette, size=h)
    col_scale = rng.choice(palette, size=w)

    board = np.full((h + 2, w + 2), bg, dtype=int)

    max_templates = max(2, min(7, (h * w) // ((s + 2) * (s + 2))))
    target = int(rng.integers(2, max_templates + 1))

    candidates = [(r, c) for r in range(1, h - s + 2) for c in range(1, w - s + 2)]
    rng.shuffle(candidates)
    occupied = np.zeros((h, w), dtype=bool)
    placements = []

    for top, left in candidates:
        if len(placements) >= target:
            break

        r0 = max(0, top - 2)
        r1 = min(h, top + s)
        c0 = max(0, left - 2)
        c1 = min(w, left + s)

        if occupied[r0:r1, c0:c1].any():
            continue

        occupied[r0:r1, c0:c1] = True
        placements.append((top, left))

    if not placements:
        placements.append(((h - s) // 2 + 1, (w - s) // 2 + 1))

    for top, left in placements:
        block = proto.copy()
        missing = int(rng.integers(1, 4))

        chosen = False
        for _ in range(120):
            idx = rng.choice(s * s, size=missing, replace=False)
            cand = block.copy()
            for x in idx:
                cand[x // s, x % s] = bg
            if _template_missing_ok(cand, top, left, row_scale, col_scale, bg):
                block = cand
                chosen = True
                break

        if not chosen:
            idx = rng.choice(s * s, size=missing, replace=False)
            for x in idx:
                block[x // s, x % s] = bg

        board[top : top + s, left : left + s] = block

    for r in range(1, h + 1):
        board[r, 0] = row_scale[r - 1]
        board[r, w + 1] = row_scale[r - 1]

    for c in range(1, w + 1):
        board[0, c] = col_scale[c - 1]
        board[h + 1, c] = col_scale[c - 1]

    return board
