import numpy as np


def _background_color(grid):
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors(r, c, h, w):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(grid, bg):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors(rr, cc, h, w):
                    if not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append((color, cells))
    return comps


def _hole_count(mask):
    h, w = mask.shape
    padded = np.zeros((h + 2, w + 2), dtype=bool)
    padded[1:h + 1, 1:w + 1] = mask
    outside = np.zeros_like(padded, dtype=bool)
    stack = [(0, 0)]
    outside[0, 0] = True
    ph, pw = padded.shape
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors(r, c, ph, pw):
            if not outside[nr, nc] and not padded[nr, nc]:
                outside[nr, nc] = True
                stack.append((nr, nc))

    holes = 0
    seen = outside | padded
    for r in range(1, h + 1):
        for c in range(1, w + 1):
            if seen[r, c]:
                continue
            holes += 1
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                for nr, nc in _neighbors(rr, cc, ph, pw):
                    if not seen[nr, nc] and not padded[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
    return holes


def main(grid: np.ndarray) -> np.ndarray:
    """Copy each object into the corner selected by its number of enclosed holes.

    Input structure:
    The grid has a most-common background color and several non-overlapping
    single-color 4-connected foreground objects. Objects have distinct colors,
    may contain zero, one, two, or three fully enclosed background holes, and
    the four corner placement regions are initially background only.

    Rule:
    For every foreground object, count its enclosed background holes. Hole
    counts 0, 1, 2, and 3 map respectively to the top-left, top-right,
    bottom-left, and bottom-right corners. Take the object's bounding-box
    contents and shrink them by sampling the top-left pixel of each 2x2 block,
    then place the thumbnail flush with the selected corner. If several objects
    select the same corner, order them by decreasing object area and step each
    next thumbnail away from the corner along that corner's diagonal.

    Key insight:
    The closed-hole count is the classification key; it must be computed from
    the complete object boundary before the independent corner-placement rule
    can be applied.
    """
    arr = np.asarray(grid)
    bg = _background_color(arr)
    out = arr.copy()
    h, w = arr.shape
    buckets = {0: [], 1: [], 2: [], 3: []}

    for color, cells in _components(arr, bg):
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        r0, r1 = min(rows), max(rows) + 1
        c0, c1 = min(cols), max(cols) + 1
        sub = arr[r0:r1, c0:c1]
        mask = sub == color
        holes = _hole_count(mask)
        if holes in buckets:
            thumb = sub[::2, ::2].copy()
            buckets[holes].append((len(cells), r0, c0, thumb))

    for holes in buckets:
        buckets[holes].sort(key=lambda item: (-item[0], item[1], item[2]))

    offsets = {0: 0, 1: 0, 2: 0, 3: 0}
    for holes, items in buckets.items():
        for _, _, _, thumb in items:
            th, tw = thumb.shape
            d = offsets[holes]
            if holes == 0:
                rr, cc = d, d
            elif holes == 1:
                rr, cc = d, w - d - tw
            elif holes == 2:
                rr, cc = h - d - th, d
            else:
                rr, cc = h - d - th, w - d - tw
            out[rr:rr + th, cc:cc + tw] = thumb
            offsets[holes] += max(th, tw)
    return out


def _choose_holes(rng, height, width, count):
    if count == 0:
        return []
    cells = [(r, c) for r in range(1, height - 1) for c in range(1, width - 1)]
    rng.shuffle(cells)
    chosen = []
    for cell in cells:
        if all(abs(cell[0] - other[0]) + abs(cell[1] - other[1]) > 1
               for other in chosen):
            chosen.append(cell)
            if len(chosen) == count:
                return chosen
    return [(1, 1), (1, width - 2), (height - 2, width // 2)][:count]


def _make_shape(rng, holes):
    height = int(rng.integers(5, 7))
    width = int(rng.integers(5, 7))
    mask = np.ones((height, width), dtype=bool)
    for r, c in _choose_holes(rng, height, width, holes):
        mask[r, c] = False

    corners = [(0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1)]
    rng.shuffle(corners)
    for r, c in corners[:int(rng.integers(0, 3))]:
        mask[r, c] = False
    return mask


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    height = int(rng.integers(26, 31))
    width = int(rng.integers(26, 31))
    bg = int(rng.integers(0, 10))
    colors = [c for c in range(10) if c != bg]
    rng.shuffle(colors)

    object_count = int(rng.integers(4, 7))
    holes_list = [0, 1, 2, 3]
    extras = [0, 1, 2, 3]
    rng.shuffle(extras)
    holes_list.extend(extras[:object_count - 4])
    rng.shuffle(holes_list)

    row_top = max(7, height // 2 - 6)
    row_bottom = min(height - 11, row_top + 7)
    col_left = 3
    col_mid = max(10, width // 2 - 3)
    col_right = max(col_mid + 7, width - 9)
    slots = [
        (row_top, col_left),
        (row_top, col_mid),
        (row_top, col_right),
        (row_bottom, col_left),
        (row_bottom, col_mid),
        (row_bottom, col_right),
    ]
    rng.shuffle(slots)

    grid = np.full((height, width), bg, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)
    for idx, holes in enumerate(holes_list):
        mask = _make_shape(rng, holes)
        mh, mw = mask.shape
        base_r, base_c = slots[idx]
        max_r = max(0, min(height - mh, base_r + int(rng.integers(-1, 2))))
        max_c = max(0, min(width - mw, base_c + int(rng.integers(-1, 2))))

        placed = False
        candidates = [(max_r, max_c)]
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                candidates.append((base_r + dr, base_c + dc))
        rng.shuffle(candidates)
        for r, c in candidates:
            r = max(0, min(height - mh, r))
            c = max(0, min(width - mw, c))
            pad_r0 = max(0, r - 1)
            pad_r1 = min(height, r + mh + 1)
            pad_c0 = max(0, c - 1)
            pad_c1 = min(width, c + mw + 1)
            if not occupied[pad_r0:pad_r1, pad_c0:pad_c1].any():
                color = colors[idx]
                grid[r:r + mh, c:c + mw][mask] = color
                occupied[r:r + mh, c:c + mw] |= mask
                placed = True
                break
        if not placed:
            color = colors[idx]
            r, c = slots[idx]
            r = max(0, min(height - mh, r))
            c = max(0, min(width - mw, c))
            grid[r:r + mh, c:c + mw][mask] = color
            occupied[r:r + mh, c:c + mw] |= mask
    return grid
