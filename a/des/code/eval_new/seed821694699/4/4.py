import numpy as np
from typing import List, Tuple

Cell = Tuple[int, int]


def _most_frequent_color(grid: np.ndarray) -> int:
    """Return the most frequent color in the grid."""
    flat = np.asarray(grid, dtype=np.int64).ravel()
    counts = np.bincount(flat)
    return int(np.argmax(counts))


def _extract_palette(grid: np.ndarray, background: int) -> List[int]:
    """Read palette colors from the bottom row left to right."""
    bottom = np.asarray(grid[-1])
    return [int(v) for v in bottom if int(v) != background]


def _connected_components(mask: np.ndarray) -> List[List[Cell]]:
    """List 4-connected components in a boolean mask."""
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: List[List[Cell]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue

            stack: List[Cell] = [(r, c)]
            visited[r, c] = True
            comp: List[Cell] = []

            while stack:
                y, x = stack.pop()
                comp.append((y, x))

                if y > 0 and mask[y - 1, x] and not visited[y - 1, x]:
                    visited[y - 1, x] = True
                    stack.append((y - 1, x))
                if y + 1 < h and mask[y + 1, x] and not visited[y + 1, x]:
                    visited[y + 1, x] = True
                    stack.append((y + 1, x))
                if x > 0 and mask[y, x - 1] and not visited[y, x - 1]:
                    visited[y, x - 1] = True
                    stack.append((y, x - 1))
                if x + 1 < w and mask[y, x + 1] and not visited[y, x + 1]:
                    visited[y, x + 1] = True
                    stack.append((y, x + 1))

            comps.append(comp)

    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """Solve shape recoloring by area order.

    Input
    -----
    grid : np.ndarray
        Integer ARC grid. Background color is the most frequent color.
        Bottom row stores palette pixels left-to-right, separated by background.

    Output
    ------
    np.ndarray
        Grid with all non-background components above the last row recolored by sorted
        area order, and the last row restored to background color.

    Algorithm
    ---------
    1. Determine background color as mode color.
    2. Extract palette from the bottom row.
    3. Find 4-connected non-background components above the bottom row.
    4. Sort components by area (pixel count), recolor in palette order.
    5. Set the bottom row to background.

    Complexity
    ----------
    O(H * W) time and O(H * W) extra space.
    """
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape

    background = _most_frequent_color(grid)
    palette = _extract_palette(grid, background)

    result = np.array(grid, copy=True)
    non_bg = grid[: h - 1] != background
    components = _connected_components(non_bg)
    components.sort(key=len)

    for color, comp in zip(palette, components):
        for y, x in comp:
            result[y, x] = color

    if h > 0:
        result[h - 1, :] = background
    return result


def _random_connected_mask(rng: np.random.Generator) -> np.ndarray:
    """Create a small random connected shape mask."""
    target = int(rng.integers(4, 10))
    active = {(0, 0)}
    frontier = [(0, 0)]

    while len(active) < target:
        y, x = frontier[int(rng.integers(0, len(frontier)))]
        candidates = []

        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if (ny, nx) not in active:
                candidates.append((ny, nx))

        if not candidates:
            y, x = frontier[int(rng.integers(0, len(frontier)))]
            continue

        ny, nx = candidates[int(rng.integers(0, len(candidates)))]
        active.add((ny, nx))
        frontier.append((ny, nx))

    ys, xs = zip(*active)
    ys = np.array(ys, dtype=int)
    xs = np.array(xs, dtype=int)
    ys -= ys.min()
    xs -= xs.min()
    mask = np.zeros((ys.max() + 1, xs.max() + 1), dtype=bool)
    mask[ys, xs] = True
    return mask


def _scaled_mask(mask: np.ndarray, k: int) -> np.ndarray:
    ys, xs = np.where(mask)
    scaled = np.zeros((mask.shape[0] * k, mask.shape[1] * k), dtype=bool)
    for y, x in zip(ys, xs):
        scaled[y * k:(y + 1) * k, x * k:(x + 1) * k] = True
    return scaled


def generate_input() -> np.ndarray:
    """Generate random ARC-like test input with size <= 30."""
    rng = np.random.default_rng()
    while True:
        background = int(rng.integers(0, 10))
        k = int(rng.integers(3, 7))

        base = _random_connected_mask(rng)
        scales = np.array(sorted(rng.choice(np.arange(1, 9), size=k, replace=False)), dtype=int)
        max_scale = int(scales.max())

        req_h = base.shape[0] * max_scale + 4
        req_w = base.shape[1] * max_scale + 4
        min_w = max(req_w, 2 * k + 1, 6)
        if req_h > 29 or min_w > 29:
            continue

        h = int(rng.integers(req_h + 1, 31))
        w = int(rng.integers(min_w, 31))

        grid = np.full((h, w), fill_value=background, dtype=int)
        used = np.zeros((h - 1, w), dtype=bool)

        shape_masks = [_scaled_mask(base, s) for s in scales]
        available_colors = [c for c in range(10) if c != background]
        if len(available_colors) < k:
            continue
        shape_colors = [int(c) for c in rng.choice(available_colors, size=k, replace=False)]

        placement_ok = True
        play_h = h - 1
        for mask, color in sorted(zip(shape_masks, shape_colors), key=lambda p: p[0].size, reverse=True):
            mh, mw = mask.shape
            found = False
            for _ in range(500):
                y = int(rng.integers(0, play_h - mh + 1))
                x = int(rng.integers(0, w - mw + 1))
                y0 = max(0, y - 1)
                x0 = max(0, x - 1)
                y1 = min(play_h, y + mh + 1)
                x1 = min(w, x + mw + 1)
                window = used[y0:y1, x0:x1]
                if np.any(window):
                    continue
                used[y : y + mh, x : x + mw] |= mask
                region = grid[y : y + mh, x : x + mw]
                region[mask] = color
                found = True
                break
            if not found:
                placement_ok = False
                break

        if not placement_ok:
            continue

        palette_colors = [int(c) for c in rng.choice(available_colors, size=k, replace=False)]
        start = int(rng.integers(0, w - (2 * k - 1) + 1))
        for i, color in enumerate(palette_colors):
            grid[h - 1, start + 2 * i] = color

        return grid
