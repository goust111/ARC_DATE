import random
from collections import deque

import numpy as np


BACKGROUND = 0
GRAY = 5


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    found = []
    rows, cols = mask.shape
    for sr, sc in np.argwhere(mask):
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        q = deque([(int(sr), int(sc))])
        cells = []
        while q:
            r, c = q.popleft()
            cells.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        found.append(np.array(cells, dtype=int))
    return found


def _count_holes(cells):
    min_r, min_c = cells.min(axis=0)
    max_r, max_c = cells.max(axis=0)
    h = int(max_r - min_r + 1)
    w = int(max_c - min_c + 1)
    solid = np.zeros((h, w), dtype=bool)
    solid[cells[:, 0] - min_r, cells[:, 1] - min_c] = True
    empty = ~solid

    outside = np.zeros((h, w), dtype=bool)
    q = deque()
    for r in range(h):
        for c in (0, w - 1):
            if empty[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if empty[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and empty[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    seen = outside | solid
    holes = 0
    for sr, sc in np.argwhere(empty & ~outside):
        if seen[sr, sc]:
            continue
        holes += 1
        seen[sr, sc] = True
        q = deque([(int(sr), int(sc))])
        while q:
            r, c = q.popleft()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and empty[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
    return holes


def _mirror_for_axis(cells, axis):
    r = cells[:, 0]
    c = cells[:, 1]
    if axis == "left":
        pivot = c.min()
        return r, 2 * pivot - c
    if axis == "right":
        pivot = c.max()
        return r, 2 * pivot - c
    if axis == "top":
        pivot = r.min()
        return 2 * pivot - r, c
    pivot = r.max()
    return 2 * pivot - r, c


def _choose_axis(cells, shape):
    rows = cells[:, 0]
    cols = cells[:, 1]
    min_r, max_r = int(rows.min()), int(rows.max())
    min_c, max_c = int(cols.min()), int(cols.max())
    candidates = [
        ("left", int(np.count_nonzero(cols == min_c))),
        ("right", int(np.count_nonzero(cols == max_c))),
        ("top", int(np.count_nonzero(rows == min_r))),
        ("bottom", int(np.count_nonzero(rows == max_r))),
    ]
    valid = []
    h, w = shape
    for axis, score in candidates:
        rr, cc = _mirror_for_axis(cells, axis)
        if np.all((0 <= rr) & (rr < h) & (0 <= cc) & (cc < w)):
            valid.append((score, axis))
    if not valid:
        return None
    valid.sort(reverse=True)
    return valid[0][1]


def main(grid: np.ndarray) -> np.ndarray:
    """Complete each half object with a palette-colored mirror chosen by hole count.

    Input structure:
    The top row is a palette row made from isolated non-background pixels ordered
    left to right.  Below it, black background contains separate gray seed
    objects.  Each seed has one visible horizontal or vertical straight edge that
    acts as its symmetry axis, and closed black holes may appear inside the gray
    component.

    Rule:
    Remove the palette row.  For each gray component, count its enclosed black
    holes.  Use that count as a 1-based index into the palette row.  Reflect the
    component across its visible axis and draw only the new half in the selected
    palette color, leaving the original gray pixels unchanged.

    Key insight:
    The number of holes is the palette lookup key, while the longest straight
    edge of the half object identifies the direction in which the colored mirror
    must grow.
    """
    palette_cols = np.where(grid[0] != BACKGROUND)[0]
    palette = [int(grid[0, c]) for c in palette_cols]
    out = np.array(grid[1:], copy=True)
    if not palette:
        return out

    for cells in _components(out == GRAY):
        holes = _count_holes(cells)
        if holes < 1 or holes > len(palette):
            continue
        axis = _choose_axis(cells, out.shape)
        if axis is None:
            continue
        rr, cc = _mirror_for_axis(cells, axis)
        color = palette[holes - 1]
        same = (rr == cells[:, 0]) & (cc == cells[:, 1])
        rr = rr[~same]
        cc = cc[~same]
        blank = out[rr, cc] == BACKGROUND
        out[rr[blank], cc[blank]] = color
    return out


def _can_place(reserved, cells):
    rows, cols = reserved.shape
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if rr < 0 or rr >= rows or cc < 0 or cc >= cols or reserved[rr, cc]:
                    return False
    return True


def _mark(reserved, cells):
    for r, c in cells:
        reserved[r, c] = True


def _hole_candidates(mask):
    h, w = mask.shape
    taken = ~mask
    candidates = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if taken[r, c]:
                continue
            ok = True
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if taken[r + dr, c + dc]:
                    ok = False
                    break
            if ok:
                candidates.append((r, c))
    random.shuffle(candidates)
    return candidates


def _make_seed_mask(direction, holes):
    for _ in range(100):
        if direction in ("left", "right"):
            h = random.randint(max(7, holes * 2 + 3), 12)
            w = random.randint(5, 6)
            mask = np.ones((h, w), dtype=bool)
            notch_col = 0 if direction == "right" else w - 1
            notch_rows = random.sample(range(1, h - 1), random.randint(1, 3))
            for r in notch_rows:
                mask[r, notch_col] = False
        else:
            h = random.randint(5, 6)
            w = random.randint(max(7, holes * 2 + 3), 12)
            mask = np.ones((h, w), dtype=bool)
            notch_row = 0 if direction == "bottom" else h - 1
            notch_cols = random.sample(range(1, w - 1), random.randint(1, 3))
            for c in notch_cols:
                mask[notch_row, c] = False

        candidates = _hole_candidates(mask)
        selected = []
        for r, c in candidates:
            if all(abs(r - pr) + abs(c - pc) > 1 for pr, pc in selected):
                selected.append((r, c))
                if len(selected) == holes:
                    break
        if len(selected) == holes:
            for r, c in selected:
                mask[r, c] = False
            return mask
    raise RuntimeError("could not build seed mask")


def _footprint(mask, direction, top, left):
    local = np.argwhere(mask)
    cells = []
    h, w = mask.shape
    for r, c in local:
        ar, ac = int(top + r), int(left + c)
        cells.append((ar, ac))
        if direction == "right":
            cells.append((ar, int(left + 2 * (w - 1) - c)))
        elif direction == "left":
            cells.append((ar, int(left - c)))
        elif direction == "bottom":
            cells.append((int(top + 2 * (h - 1) - r), ac))
        else:
            cells.append((int(top - r), ac))
    return cells


def _placement_ranges(mask, direction, rows, cols):
    h, w = mask.shape
    if direction == "right":
        return range(0, rows - h + 1), range(0, cols - (2 * w - 1) + 1)
    if direction == "left":
        return range(0, rows - h + 1), range(w - 1, cols - w + 1)
    if direction == "bottom":
        return range(0, rows - (2 * h - 1) + 1), range(0, cols - w + 1)
    return range(h - 1, rows - h + 1), range(0, cols - w + 1)


def generate_input() -> np.ndarray:
    palette_colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], random.randint(3, 4))
    cols = random.randint(max(18, 2 * len(palette_colors) + 3), 30)
    rows = random.randint(18, 30)
    lower_rows = rows - 1

    for _ in range(200):
        grid = np.zeros((rows, cols), dtype=int)
        start = random.randint(0, min(2, cols - (2 * len(palette_colors) - 1)))
        for i, color in enumerate(palette_colors):
            grid[0, start + 2 * i] = color

        reserved = np.zeros((lower_rows, cols), dtype=bool)
        object_count = random.randint(1, 3)
        directions = random.sample(["left", "right", "top", "bottom"], object_count)
        placed = True
        for direction in directions:
            holes = random.randint(1, len(palette_colors))
            mask = _make_seed_mask(direction, holes)
            row_range, col_range = _placement_ranges(mask, direction, lower_rows, cols)
            positions = [(r, c) for r in row_range for c in col_range]
            random.shuffle(positions)
            for top, left in positions:
                cells = _footprint(mask, direction, top, left)
                if _can_place(reserved, cells):
                    _mark(reserved, cells)
                    for r, c in np.argwhere(mask):
                        grid[1 + top + int(r), left + int(c)] = GRAY
                    break
            else:
                placed = False
                break
        if placed:
            return grid

    return generate_input()
