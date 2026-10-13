import numpy as np


BLACK = 0
GREY = 5
MAROON = 9


def _components(mask):
    rows, cols = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(rows):
        for c in range(cols):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < rows and 0 <= nc < cols and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _rotation_from_pointer(coords):
    rs = [p[0] for p in coords]
    cs = [p[1] for p in coords]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    for rr, cc, color in coords:
        if color != MAROON:
            continue
        if rr == r0:
            return 0
        if cc == c1:
            return 1
        if rr == r1:
            return 2
        return 3
    return 0


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each upper object according to its maroon pointer and carve the rotated shape into the matching lower slot.

    Input structure:
    - A grey horizontal separator splits the grid into an upper object area and a lower slot area.
    - Each upper object has one base color plus exactly one maroon pointer pixel attached to the object.
    - The lower area contains one solid rectangle slot for each base color, sized to the object's rotated bounding box.

    Rule:
    - Decode rotation from the pointer position on the object's bounding-box edge: top, right, bottom, left mean 0, 90, 180, and 270 degrees clockwise.
    - Restore the pointer cell to the object's base color, rotate the whole object, and locate the lower slot with the same base color.
    - Clear the slot to black and repaint only the rotated object cells inside that slot. Clear the original upper objects while preserving the grey separator.

    Key insight:
    - The maroon pixel is not part of the final shape: it is a positional instruction that must be decoded before the matching slot can be carved correctly.
    """
    out = grid.copy()
    sep = int(np.where(np.all(grid == GREY, axis=1))[0][0])
    top_mask = (grid[:sep] != BLACK) & (grid[:sep] != GREY)
    objects = []
    for comp in _components(top_mask):
        rows = [p[0] for p in comp]
        cols = [p[1] for p in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        arr = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        base = int(np.unique(arr[(arr != BLACK) & (arr != MAROON)])[0])
        arr[arr == MAROON] = base
        coords = [(rr, cc, int(grid[rr, cc])) for rr, cc in comp]
        rot = _rotation_from_pointer(coords)
        objects.append((base, np.rot90(arr, -rot), (r0, r1, c0, c1)))
        out[r0 : r1 + 1, c0 : c1 + 1] = BLACK
    for base, rotated, _ in objects:
        mask = grid[sep + 1 :] == base
        for comp in _components(mask):
            rows = [p[0] for p in comp]
            cols = [p[1] for p in comp]
            r0, r1 = min(rows) + sep + 1, max(rows) + sep + 1
            c0, c1 = min(cols), max(cols)
            if (r1 - r0 + 1, c1 - c0 + 1) == rotated.shape:
                out[r0 : r1 + 1, c0 : c1 + 1] = BLACK
                out[r0 : r1 + 1, c0 : c1 + 1][rotated == base] = base
                break
    return out


def _random_object(rng):
    h = int(rng.integers(3, 5))
    w = int(rng.integers(3, 5))
    side = rng.choice(["top", "right", "bottom", "left"])
    if side == "top":
        pr, pc = 0, w // 2
    elif side == "right":
        pr, pc = h // 2, w - 1
    elif side == "bottom":
        pr, pc = h - 1, w // 2
    else:
        pr, pc = h // 2, 0
    mask = np.zeros((h, w), dtype=bool)
    center = (h // 2, w // 2)
    rr, cc = center
    mask[rr, cc] = True
    while cc != pc:
        cc += 1 if pc > cc else -1
        mask[rr, cc] = True
    while rr != pr:
        rr += 1 if pr > rr else -1
        mask[rr, cc] = True
    frontier = list(zip(*np.where(mask)))
    target = int(rng.integers(max(4, mask.sum()), max(5, h * w)))
    while mask.sum() < target:
        rr, cc = frontier[int(rng.integers(0, len(frontier)))]
        candidates = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w:
                candidates.append((nr, nc))
        rng.shuffle(candidates)
        for nr, nc in candidates[:2]:
            if not mask[nr, nc]:
                mask[nr, nc] = True
                frontier.append((nr, nc))
    return mask, (pr, pc), {"top": 0, "right": 1, "bottom": 2, "left": 3}[side]


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    colors = list(rng.choice([1, 2, 3, 4, 6, 7, 8], size=int(rng.integers(2, 5)), replace=False))
    height = 24
    width = 28
    sep = 10
    grid = np.zeros((height, width), dtype=int)
    grid[sep, :] = GREY
    top_col = 1
    bottom_col = 1
    for color in colors:
        mask, (pr, pc), rot = _random_object(rng)
        h, w = mask.shape
        obj = np.zeros((h, w), dtype=int)
        obj[mask] = color
        obj[pr, pc] = MAROON
        grid[2 : 2 + h, top_col : top_col + w] = obj
        rotated = np.rot90(np.where(obj == MAROON, color, obj), -rot)
        rh, rw = rotated.shape
        grid[sep + 3 : sep + 3 + rh, bottom_col : bottom_col + rw] = color
        top_col += w + 3
        bottom_col += rw + 3
    return grid
