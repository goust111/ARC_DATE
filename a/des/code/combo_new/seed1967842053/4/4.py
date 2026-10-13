import random
from collections import deque

import numpy as np


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    comps = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(cells)
    return comps


def _find_background(grid):
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _find_marker_color(grid, bg):
    comps = _components(grid != bg)
    candidates = []
    if len(comps) >= 2:
        for color in np.unique(grid):
            color = int(color)
            if color == bg:
                continue
            if all(sum(1 for r, c in comp if int(grid[r, c]) == color) == 1 for comp in comps):
                candidates.append(color)
        if candidates:
            return candidates[0]

    best_color = None
    best_count = -1
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        comps = _components(grid == color)
        if len(comps) < 2:
            continue
        if all(len(comp) == 1 for comp in comps):
            count = len(comps)
            if count > best_count:
                best_count = count
                best_color = color
    if best_color is not None:
        return best_color

    vals, counts = np.unique(grid[grid != bg], return_counts=True)
    if len(vals) == 0:
        return bg
    one_pixel = vals[counts == 1]
    if len(one_pixel):
        return int(one_pixel[0])
    return int(vals[np.argmin(counts)])


def _boxes_overlap(a, b):
    ar, ac, ah, aw = a
    br, bc, bh, bw = b
    return ar < br + bh and br < ar + ah and ac < bc + bw and bc < ac + aw


def _valid_layout(grid, bg, marker):
    marks = [tuple(int(x) for x in p) for p in np.argwhere(grid == marker)]
    if not (2 <= len(marks) <= 10):
        return None

    h_grid, w_grid = grid.shape
    active = [tuple(int(x) for x in p) for p in np.argwhere(grid != bg)]
    best = None

    for anchor_r in range(h_grid):
        if any(mr - anchor_r < 0 for mr, _ in marks):
            break
        max_h = min(h_grid - (mr - anchor_r) for mr, _ in marks)
        for anchor_c in range(w_grid):
            if any(mc - anchor_c < 0 for _, mc in marks):
                break
            max_w = min(w_grid - (mc - anchor_c) for _, mc in marks)

            for out_h in range(anchor_r + 1, max_h + 1):
                if best is not None and out_h * (anchor_c + 1) >= best[0] * best[1]:
                    continue
                for out_w in range(anchor_c + 1, max_w + 1):
                    if best is not None and out_h * out_w >= best[0] * best[1]:
                        continue

                    boxes = [(mr - anchor_r, mc - anchor_c, out_h, out_w) for mr, mc in marks]
                    bad = False
                    for i in range(len(boxes)):
                        for j in range(i + 1, len(boxes)):
                            if _boxes_overlap(boxes[i], boxes[j]):
                                bad = True
                                break
                        if bad:
                            break
                    if bad:
                        continue

                    edge_hits = [[False, False, False, False] for _ in boxes]
                    for r, c in active:
                        owner = -1
                        for i, (top, left, _, _) in enumerate(boxes):
                            if top <= r < top + out_h and left <= c < left + out_w:
                                if owner != -1:
                                    owner = -2
                                    break
                                owner = i
                        if owner < 0:
                            bad = True
                            break
                        top, left, _, _ = boxes[owner]
                        rr, cc = r - top, c - left
                        if rr == 0:
                            edge_hits[owner][0] = True
                        if rr == out_h - 1:
                            edge_hits[owner][1] = True
                        if cc == 0:
                            edge_hits[owner][2] = True
                        if cc == out_w - 1:
                            edge_hits[owner][3] = True
                    if bad or not all(all(hit) for hit in edge_hits):
                        continue
                    best = (out_h, out_w, anchor_r, anchor_c, boxes)
    return best


def _infer_layout(grid, bg):
    best = None
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        count = int(np.sum(grid == color))
        if not (2 <= count <= 10):
            continue
        layout = _valid_layout(grid, bg, color)
        if layout is None:
            continue
        score = (layout[0] * layout[1], -count)
        if best is None or score < best[0]:
            best = (score, color, layout)
    if best is None:
        return None
    return best[1], best[2]


def main(grid: np.ndarray) -> np.ndarray:
    """Align scattered fragments by their shared marker and merge them into one pattern.

    Input structure:
    The grid has a dominant background color and several separated rectangular fragments.
    Each fragment contains exactly one pixel of the same marker color, plus ordinary
    non-background pattern colors and possible background holes. All fragment boxes have
    the same size and are separated from one another by background.

    Rule:
    Detect the background, identify the shared one-pixel marker color, and split the
    non-background cells into fragments. Crop every fragment to its bounding box, align
    all crops so their marker pixels occupy the same output position, and overlay the
    ordinary colored pixels. Marker pixels are erased to background in the output.

    Key insight:
    The separate fragments are complementary pieces of one image; the repeated marker is
    the anchor that reveals how the pieces must be registered before the final merge.
    """
    grid = np.asarray(grid)
    bg = _find_background(grid)
    inferred = _infer_layout(grid, bg)
    if inferred is not None:
        marker, (out_h, out_w, anchor_r, anchor_c, boxes) = inferred
        out = np.full((out_h, out_w), bg, dtype=grid.dtype)
        for top, left, _, _ in boxes:
            sub = grid[top : top + out_h, left : left + out_w]
            mask = (sub != bg) & (sub != marker)
            out[mask] = sub[mask]
        out[anchor_r, anchor_c] = bg
        return out

    marker = _find_marker_color(grid, bg)
    active = grid != bg
    comps = _components(active)

    fragments = []
    for comp in comps:
        if not any(int(grid[r, c]) == marker for r, c in comp):
            continue
        rs = [r for r, _ in comp]
        cs = [c for _, c in comp]
        r0, r1 = min(rs), max(rs) + 1
        c0, c1 = min(cs), max(cs) + 1
        sub = grid[r0:r1, c0:c1]
        loc = np.argwhere(sub == marker)
        if len(loc) != 1:
            continue
        fragments.append((sub.copy(), tuple(int(x) for x in loc[0])))

    if not fragments:
        return np.array([[bg]], dtype=grid.dtype)

    out_h = max(sub.shape[0] for sub, _ in fragments)
    out_w = max(sub.shape[1] for sub, _ in fragments)
    anchor_r = max(pos[0] for _, pos in fragments)
    anchor_c = max(pos[1] for _, pos in fragments)
    out_h = max(out_h, max(anchor_r - pos[0] + sub.shape[0] for sub, pos in fragments))
    out_w = max(out_w, max(anchor_c - pos[1] + sub.shape[1] for sub, pos in fragments))
    out = np.full((out_h, out_w), bg, dtype=grid.dtype)

    for sub, pos in fragments:
        dr = anchor_r - pos[0]
        dc = anchor_c - pos[1]
        for r in range(sub.shape[0]):
            for c in range(sub.shape[1]):
                color = int(sub[r, c])
                if color != bg and color != marker:
                    out[dr + r, dc + c] = sub[r, c]
    out[anchor_r, anchor_c] = bg
    return out


def _random_pattern(h, w, colors, marker_pos):
    pat = np.zeros((h, w), dtype=int)
    cells = [(r, c) for r in range(h) for c in range(w) if (r, c) != marker_pos]
    random.shuffle(cells)
    target = random.randint(max(6, h + w), max(7, int(h * w * 0.65)))
    chosen = cells[: min(target, len(cells))]

    for r in range(h):
        pat[r, marker_pos[1]] = random.choice(colors)
    for c in range(w):
        pat[marker_pos[0], c] = random.choice(colors)
    for r, c in chosen:
        pat[r, c] = random.choice(colors)

    pat[marker_pos] = 0
    return pat


def generate_input() -> np.ndarray:
    bg = 0
    marker = 9
    palette = [1, 2, 3, 4, 5, 6, 7, 8]
    random.shuffle(palette)
    colors = palette[: random.randint(3, 5)]

    box_h = random.randint(6, 8)
    box_w = random.randint(6, 8)
    marker_pos = (random.randint(1, box_h - 2), random.randint(1, box_w - 2))
    fragment_count = random.randint(3, min(4, box_h - 2, box_w - 2))

    pattern = _random_pattern(box_h, box_w, colors, marker_pos)
    buckets = [[] for _ in range(fragment_count)]
    assigned = set()

    for i in range(fragment_count):
        forced = ((0, i + 1), (box_h - 1, i + 1), (i + 1, 0), (i + 1, box_w - 1))
        for cell in forced:
            if cell == marker_pos:
                continue
            pattern[cell] = random.choice(colors)
            if cell not in assigned:
                buckets[i].append(cell)
                assigned.add(cell)

    filled = [
        (r, c)
        for r in range(box_h)
        for c in range(box_w)
        if pattern[r, c] != bg and (r, c) not in assigned
    ]
    random.shuffle(filled)
    for i, cell in enumerate(filled):
        buckets[i % fragment_count].append(cell)

    fragments = []
    for bucket in buckets:
        frag = np.full((box_h, box_w), bg, dtype=int)
        for r, c in bucket:
            frag[r, c] = pattern[r, c]
        frag[marker_pos] = marker
        fragments.append(frag)

    cell_h = box_h + 1
    cell_w = box_w + 1
    rows = random.randint(1, min(fragment_count, 3))
    cols = (fragment_count + rows - 1) // rows
    if cols > 3:
        cols = 3
        rows = (fragment_count + cols - 1) // cols
    grid_h = rows * cell_h + 1
    grid_w = cols * cell_w + 1
    grid = np.full((grid_h, grid_w), bg, dtype=int)

    slots = [(rr, cc) for rr in range(rows) for cc in range(cols)]
    random.shuffle(slots)
    for frag, (sr, sc) in zip(fragments, slots):
        top = sr * cell_h + 1
        left = sc * cell_w + 1
        max_jitter_r = cell_h - box_h - 1
        max_jitter_c = cell_w - box_w - 1
        top += random.randint(0, max_jitter_r)
        left += random.randint(0, max_jitter_c)
        grid[top : top + box_h, left : left + box_w] = frag

    return grid
