import numpy as np


def _components(mask):
    rows, cols = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    comps = []
    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        stack = [(sr, sc)]
        seen[sr, sc] = True
        coords = []
        while stack:
            r, c = stack.pop()
            coords.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < rows and 0 <= nc < cols and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.array(coords, dtype=np.int16))
    return comps


def _bbox(coords):
    r0 = int(coords[:, 0].min())
    r1 = int(coords[:, 0].max()) + 1
    c0 = int(coords[:, 1].min())
    c1 = int(coords[:, 1].max()) + 1
    return r0, r1, c0, c1


def main(grid: np.ndarray) -> np.ndarray:
    """Move each small solid object into its same-color large hollow object, leaving a hollow footprint.

    Input structure:
    - The background is black.
    - Every foreground color appears as two disconnected components.
    - For each color, one component has the smaller bounding box and is solid; the other is a larger hollow outline of the same scaled shape.
    - Different color pairs are separated, and the hollow object has empty interior space for the small object.
    Rule:
    - For each color pair, identify the smaller component by bounding-box area.
    - Erase that small component at its original location and draw a hollow rectangular frame over its old bounding box in the same color.
    - Copy the small component's exact mask into the center of the larger component's bounding box, using the upper-left centered position when centering is not exact.
    - Keep the large hollow outline otherwise unchanged.
    Key insight:
    - Color gives the pairing, while relative component size reveals which solid shape must move into the hollow scaled partner.
    """
    arr = np.asarray(grid)
    out = arr.copy()
    plans = []

    for color in np.unique(arr):
        color = int(color)
        if color == 0:
            continue
        comps = _components(arr == color)
        if len(comps) != 2:
            continue

        infos = []
        for coords in comps:
            r0, r1, c0, c1 = _bbox(coords)
            box_area = (r1 - r0) * (c1 - c0)
            infos.append((box_area, len(coords), (r0, r1, c0, c1), coords))
        infos.sort(key=lambda item: (item[0], item[1]))
        plans.append((color, infos[0], infos[1]))

    for color, small, large in plans:
        _, _, (sr0, sr1, sc0, sc1), small_coords = small
        _, _, (lr0, lr1, lc0, lc1), _ = large

        out[small_coords[:, 0], small_coords[:, 1]] = 0

        out[sr0:sr1, sc0] = color
        out[sr0:sr1, sc1 - 1] = color
        out[sr0, sc0:sc1] = color
        out[sr1 - 1, sc0:sc1] = color

        small_h = sr1 - sr0
        small_w = sc1 - sc0
        small_mask = np.zeros((small_h, small_w), dtype=bool)
        small_mask[small_coords[:, 0] - sr0, small_coords[:, 1] - sc0] = True

        large_h = lr1 - lr0
        large_w = lc1 - lc0
        top = lr0 + max(0, (large_h - small_h) // 2)
        left = lc0 + max(0, (large_w - small_w) // 2)
        target = out[top:top + small_h, left:left + small_w]
        if target.shape == small_mask.shape:
            target[small_mask] = color

    return out


def _rect_border(h, w):
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _random_origin(rng, rows, cols, h, w):
    if rows - h >= 2:
        top = int(rng.integers(1, rows - h))
    else:
        top = int(rng.integers(0, rows - h + 1))
    if cols - w >= 2:
        left = int(rng.integers(1, cols - w))
    else:
        left = int(rng.integers(0, cols - w + 1))
    return top, left


def _can_place(reserved, top, left, h, w):
    rows, cols = reserved.shape
    r0 = max(0, top - 1)
    r1 = min(rows, top + h + 1)
    c0 = max(0, left - 1)
    c1 = min(cols, left + w + 1)
    return not reserved[r0:r1, c0:c1].any()


def _place_mask(grid, reserved, mask, color, rng, tries=200):
    rows, cols = grid.shape
    h, w = mask.shape
    if h > rows or w > cols:
        return False
    for _ in range(tries):
        top, left = _random_origin(rng, rows, cols, h, w)
        if not _can_place(reserved, top, left, h, w):
            continue
        region = grid[top:top + h, left:left + w]
        region[mask] = color
        reserved[top:top + h, left:left + w] = True
        return True
    return False


def _choose_small_size(rng):
    exact_center = bool(rng.integers(0, 2))
    if exact_center:
        return int(rng.choice([4, 6])), int(rng.choice([4, 6]))

    choices = [3, 4, 5, 6]
    while True:
        h = int(rng.choice(choices))
        w = int(rng.choice(choices))
        if h % 2 == 1 or w % 2 == 1:
            return h, w


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()

    for _ in range(300):
        rows = int(rng.integers(18, 31))
        cols = int(rng.integers(18, 31))
        pair_count = int(rng.choice([1, 2, 3], p=[0.25, 0.45, 0.30]))
        colors = rng.choice(np.arange(1, 10), size=pair_count, replace=False)

        grid = np.zeros((rows, cols), dtype=np.int8)
        reserved = np.zeros((rows, cols), dtype=bool)
        ok = True

        for color in colors:
            color = int(color)
            placed_pair = False
            for _ in range(80):
                small_h, small_w = _choose_small_size(rng)
                large_mask = _rect_border(2 * small_h, 2 * small_w)
                small_mask = np.ones((small_h, small_w), dtype=bool)

                temp_grid = grid.copy()
                temp_reserved = reserved.copy()
                if rng.random() < 0.5:
                    first = _place_mask(temp_grid, temp_reserved, large_mask, color, rng)
                    second = first and _place_mask(temp_grid, temp_reserved, small_mask, color, rng)
                else:
                    first = _place_mask(temp_grid, temp_reserved, small_mask, color, rng)
                    second = first and _place_mask(temp_grid, temp_reserved, large_mask, color, rng)

                if second:
                    grid = temp_grid
                    reserved = temp_reserved
                    placed_pair = True
                    break

            if not placed_pair:
                ok = False
                break

        if ok:
            return grid

    grid = np.zeros((20, 20), dtype=np.int8)
    grid[2:10, 2] = 1
    grid[2:10, 9] = 1
    grid[2, 2:10] = 1
    grid[9, 2:10] = 1
    grid[13:17, 13:17] = 1
    return grid
