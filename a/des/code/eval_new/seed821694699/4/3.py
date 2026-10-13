import numpy as np
from typing import List


def _connected_components(mask: np.ndarray) -> List[np.ndarray]:
    """Find 4-neighbor connected components in a boolean mask."""
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    components: List[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            coords: List[tuple[int, int]] = []

            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))

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

            components.append(np.asarray(coords, dtype=int))

    return components


def _line_descriptor(points: np.ndarray) -> tuple[str, int, int, int] | None:
    """Return (orientation, fixed_axis_value, span_start, span_end) for a straight segment."""
    if points.size == 0:
        return None

    rows = points[:, 0]
    cols = points[:, 1]

    if np.all(rows == rows[0]):
        return ("H", int(rows[0]), int(cols.min()), int(cols.max()))
    if np.all(cols == cols[0]):
        return ("V", int(cols[0]), int(rows.min()), int(rows.max()))
    return None


def _choose_grey_color(bg: int, pair_colors: list[int]) -> int:
    """Pick a deterministic gray color not used by line colors and not equal to background."""
    used = set(pair_colors)
    for c in range(10):
        if c != bg and c not in used:
            return int(c)
    return 0 if bg != 0 else 1


def main(grid: np.ndarray) -> np.ndarray:
    """Solve pairing-and-checkerboard rectangles.

    Input:
        A 2D integer ndarray grid where the most frequent color is background, and every other color
        appears on exactly two parallel straight segments (horizontal or vertical).

    Output:
        A grid with the same shape where each color-defined pair induces a rectangle whose interior is
        checkerboard-filled using that pair color and gray, while all original segment pixels remain unchanged.

    Method:
        1) Infer background as the mode color.
        2) For each non-background color, extract connected components and keep only the case with two straight
           components of the same orientation (both H or both V).
        3) Fill the rectangle interior between the two segments with checkerboard parity ((r + c) % 2) and
           a chosen gray color for odd parity, pair color for even parity.

    Complexity:
        O(H*W) for full scan plus linear component traversal; memory O(H*W).
    """
    working = np.array(grid, dtype=int, copy=True)
    if working.ndim != 2 or working.size == 0:
        return working

    counts = np.bincount(working.reshape(-1))
    bg = int(np.argmax(counts))
    pair_colors = [int(c) for c in np.unique(working) if c != bg]

    pair_info = {}
    for color in pair_colors:
        comps = _connected_components(working == color)
        if len(comps) != 2:
            continue

        d0 = _line_descriptor(comps[0])
        d1 = _line_descriptor(comps[1])
        if d0 is None or d1 is None:
            continue
        if d0[0] != d1[0]:
            continue

        pair_info[color] = (d0, d1)

    if not pair_info:
        return working

    grey = _choose_grey_color(bg, list(pair_info.keys()))

    for color, (d1, d2) in pair_info.items():
        orient, axis1, s1, e1 = d1
        _, axis2, s2, e2 = d2

        if orient == "H":
            top = min(axis1, axis2)
            bottom = max(axis1, axis2)
            if bottom - top < 2:
                continue
            left = min(s1, s2)
            right = max(e1, e2)
            if left > right:
                continue

            cols = np.arange(left, right + 1, dtype=int)
            for r in range(top + 1, bottom):
                parity = ((r + cols) & 1) == 0
                working[r, cols] = np.where(parity, color, grey)

        else:  # "V"
            left = min(axis1, axis2)
            right = max(axis1, axis2)
            if right - left < 2:
                continue
            top = min(s1, s2)
            bottom = max(e1, e2)
            if top > bottom:
                continue

            rows = np.arange(top, bottom + 1, dtype=int)
            for c in range(left + 1, right):
                parity = ((rows + c) & 1) == 0
                working[rows, c] = np.where(parity, color, grey)

    return working


def generate_input() -> np.ndarray:
    """Generate one valid random ARC-like input grid (size <= 30) for this puzzle rule set."""
    rng = np.random.default_rng()
    h = int(rng.integers(12, 31))
    w = int(rng.integers(12, 31))

    bg = 0
    grid = np.full((h, w), bg, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    colors = [1, 2, 3, 4, 5, 6, 7, 8]
    rng.shuffle(colors)
    max_pairs = min(len(colors), 4, (h - 2) // 2, (w - 2) // 2)
    if max_pairs <= 0:
        return grid

    target_pairs = int(rng.integers(1, max_pairs + 1))

    placed = 0
    for ci in range(target_pairs):
        color = colors[ci]
        for _ in range(500):
            if bool(rng.integers(0, 2)):
                top = int(rng.integers(0, h - 2))
                bottom = int(rng.integers(top + 2, h))

                max_len = min(12, w - 1)
                if max_len < 2:
                    continue
                width = int(rng.integers(2, max_len + 1))
                left = int(rng.integers(0, w - width + 1))
                right = left + width - 1
                top, bottom = min(top, bottom), max(top, bottom)
                if np.any(occupied[top : bottom + 1, left : right + 1]):
                    continue

                grid[top, left : right + 1] = color
                grid[bottom, left : right + 1] = color
                occupied[top : bottom + 1, left : right + 1] = True
                placed += 1
                break
            else:
                left = int(rng.integers(0, w - 2))
                right = int(rng.integers(left + 2, w))

                max_len = min(12, h - 1)
                if max_len < 2:
                    continue
                height = int(rng.integers(2, max_len + 1))
                top = int(rng.integers(0, h - height + 1))
                bottom = top + height - 1
                left, right = left, right
                if np.any(occupied[top : bottom + 1, left : right + 1]):
                    continue

                grid[top : bottom + 1, left] = color
                grid[top : bottom + 1, right] = color
                occupied[top : bottom + 1, left : right + 1] = True
                placed += 1
                break
    return grid


if __name__ == "__main__":
    sample = generate_input()
    result = main(sample)

    assert sample.shape == result.shape
    bg = int(np.bincount(sample.reshape(-1)).argmax())
    assert np.all(sample[sample != bg] == result[sample != bg])

    changed = sample != result
    if np.any(changed):
        assert np.all(sample[changed] == bg)

    print("self-check ok", sample.shape, sample.shape[0] * sample.shape[1])
