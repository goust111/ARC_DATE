import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5


def _neighbors4(r, c, h, w):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _label_components(mask):
    h, w = mask.shape
    labels = np.full((h, w), -1, dtype=int)
    comps = []
    label = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != -1:
                continue
            cells = []
            q = deque([(r, c)])
            labels[r, c] = label
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and labels[nr, nc] == -1:
                        labels[nr, nc] = label
                        q.append((nr, nc))
            comps.append(cells)
            label += 1
    return labels, comps


def _adjacent4(mask):
    adj = np.zeros_like(mask, dtype=bool)
    adj[1:, :] |= mask[:-1, :]
    adj[:-1, :] |= mask[1:, :]
    adj[:, 1:] |= mask[:, :-1]
    adj[:, :-1] |= mask[:, 1:]
    return adj


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each closed grey boundary with its interior marker color and grow it according to the marker count.

    Input structure:
    The grid has a black background, with one-pixel grey closed outlines enclosing separate interior regions.
    Each enclosed interior contains exactly one non-black, non-grey marker color, repeated one to four times.
    Grey pixels that do not bound a marked enclosed region may also appear and are treated as unrelated structure.

    Rule:
    For every enclosed region, count its interior marker pixels and read their shared color C.
    Recolor the whole grey boundary of that region to C, then clear the marker pixels to black.
    One marker keeps the boundary one pixel wide, two markers add the outside black 4-neighbors, three markers add
    the inside 4-neighbors, and four markers add both inside and outside 4-neighbors.
    Grey pixels not associated with a valid marked enclosure are left unchanged.

    Key insight:
    The marker color and marker count encode independent information: the color supplies the repaint color, while
    the count selects whether the boundary stays fixed, grows outward, grows inward, or grows both ways.
    """
    arr = np.array(grid, copy=False)
    h, w = arr.shape
    grey = arr == GREY
    grey_labels, _ = _label_components(grey)
    free_labels, free_comps = _label_components(~grey)
    out = np.array(arr, copy=True)
    markers_to_clear = np.zeros((h, w), dtype=bool)

    for comp_id, cells in enumerate(free_comps):
        touches_edge = any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in cells)
        if touches_edge:
            continue

        rows = np.array([r for r, _ in cells], dtype=int)
        cols = np.array([c for _, c in cells], dtype=int)
        values = arr[rows, cols]
        marker_values = values[(values != BLACK) & (values != GREY)]
        if marker_values.size == 0:
            continue
        colors = np.unique(marker_values)
        if colors.size != 1 or marker_values.size not in (1, 2, 3, 4):
            continue

        color = int(colors[0])
        count = int(marker_values.size)
        interior = free_labels == comp_id
        touching_grey = _adjacent4(interior) & grey
        boundary_ids = np.unique(grey_labels[touching_grey])
        boundary_ids = boundary_ids[boundary_ids >= 0]
        if boundary_ids.size == 0:
            continue

        boundary = np.isin(grey_labels, boundary_ids)
        beside_boundary = _adjacent4(boundary)
        out[boundary] = color

        if count in (2, 4):
            outward = beside_boundary & (arr == BLACK) & ~interior
            out[outward] = color
        if count in (3, 4):
            inward = beside_boundary & interior
            out[inward] = color

        markers_to_clear |= interior & (arr != BLACK) & (arr != GREY)

    out[markers_to_clear] = BLACK
    return out


def _dilate8(mask, steps=1):
    result = mask.copy()
    for _ in range(steps):
        old = result.copy()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                src_r0 = max(0, -dr)
                src_r1 = old.shape[0] - max(0, dr)
                src_c0 = max(0, -dc)
                src_c1 = old.shape[1] - max(0, dc)
                dst_r0 = max(0, dr)
                dst_r1 = old.shape[0] - max(0, -dr)
                dst_c0 = max(0, dc)
                dst_c1 = old.shape[1] - max(0, -dc)
                result[dst_r0:dst_r1, dst_c0:dst_c1] |= old[src_r0:src_r1, src_c0:src_c1]
    return result


def _make_rectangle(rng):
    ih = rng.randint(5, 8)
    iw = rng.randint(5, 8)
    interior = np.zeros((ih + 2, iw + 2), dtype=bool)
    interior[1:-1, 1:-1] = True
    boundary = np.zeros_like(interior)
    boundary[0, :] = True
    boundary[-1, :] = True
    boundary[:, 0] = True
    boundary[:, -1] = True
    return interior, boundary


def _make_l_shape(rng):
    ih = rng.randint(6, 9)
    iw = rng.randint(6, 9)
    thick = rng.randint(3, min(5, ih, iw))
    base = np.zeros((ih, iw), dtype=bool)
    base[:, :thick] = True
    base[-thick:, :] = True
    base = np.rot90(base, rng.randint(0, 3))
    interior = np.pad(base, 1, constant_values=False)
    boundary = _dilate8(interior, 1) & ~interior
    return interior, boundary


def _make_object(rng):
    if rng.random() < 0.5:
        return _make_rectangle(rng)
    return _make_l_shape(rng)


def _place_markers(rng, grid, top, left, interior, boundary, color, count):
    inner_ring = _adjacent4(boundary) & interior
    candidates = np.argwhere(interior & ~inner_ring)
    if len(candidates) < count:
        candidates = np.argwhere(interior)
    chosen = rng.sample(range(len(candidates)), count)
    for idx in chosen:
        rr, cc = candidates[idx]
        grid[top + rr, left + cc] = color


def generate_input() -> np.ndarray:
    rng = random.Random()
    colors = [1, 2, 3, 4, 6, 7, 8, 9]

    for _ in range(50):
        h = rng.randint(20, 30)
        w = rng.randint(20, 30)
        grid = np.zeros((h, w), dtype=int)
        blocked = np.zeros((h, w), dtype=bool)
        object_count = rng.randint(1, 4)
        counts = [1, 2, 3, 4]
        rng.shuffle(counts)
        placed = 0

        for count in counts[:object_count]:
            for _attempt in range(200):
                interior, boundary = _make_object(rng)
                oh, ow = interior.shape
                if oh + 2 > h or ow + 2 > w:
                    continue
                top = rng.randint(1, h - oh - 1)
                left = rng.randint(1, w - ow - 1)
                full = interior | boundary
                global_full = np.zeros((h, w), dtype=bool)
                global_full[top : top + oh, left : left + ow] = full
                if np.any(blocked & global_full):
                    continue

                grid[top : top + oh, left : left + ow][boundary] = GREY
                color = rng.choice(colors)
                _place_markers(rng, grid, top, left, interior, boundary, color, count)
                blocked |= _dilate8(global_full, 3)
                placed += 1
                break

        if placed:
            return grid

    return generate_input()
