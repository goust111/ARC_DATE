import numpy as np
from collections import deque


def main(grid: np.ndarray) -> np.ndarray:
    """Solve.
Input:
    Input is an integer-color 2D ARC grid. Black is 0, Grey is 5.
Output:
    Returns a same-size grid with corner templates stamped at anchor-pair midpoints.
Method:
    1) Find all non-black color components with at least one Grey pixel as templates.
    2) Sort templates by left-to-right (minimum column), and anchor color pairs by midpoint row.
    3) For each pair-template match, rotate template by direction (0/90) and recolor it with pair color.
    4) Remove original corner templates, keep anchors, and overlay recolored stamps.
Complexity:
    Let HxW be grid size and K the number of filled cells, then O(HW + K) memory/time.
"""
    arr = np.array(grid, dtype=int).copy()
    if arr.ndim != 2:
        return arr

    templates = _extract_templates(arr)
    pairs = _extract_anchor_pairs(arr)
    if not templates or not pairs:
        return arr

    # Build one canvas and delete all found templates first.
    result = arr.copy()
    for t in templates:
        result[t.rows, t.cols] = 0

    count = min(len(templates), len(pairs))
    if count == 0:
        return result

    ordered_templates = sorted(templates, key=lambda t: t.left)
    ordered_pairs = sorted(pairs, key=lambda p: (p.center_row, p.center_col))

    for idx in range(count):
        pair = ordered_pairs[idx]
        stamp = ordered_templates[idx].stamp
        rotated = _rotate_stamp(stamp, pair)
        _stamp_to_center(
            result,
            rotated,
            pair.color,
            center_row=pair.center_row,
            center_col=pair.center_col,
        )

    return result


def generate_input() -> np.ndarray:
    """Generate random ARC-like inputs matching the described anchor-template puzzle."""
    rng = np.random.default_rng()
    h = int(rng.integers(15, 30))
    w = int(rng.integers(15, 30))
    grid = np.zeros((h, w), dtype=int)

    palette = [1, 2, 3, 4, 6, 7, 8, 9]
    rng.shuffle(palette)

    # Place template stamps in a corner, separated by black and arranged left-to-right.
    template_specs = []
    cursor = 1
    base_row = 1
    template_count = int(rng.integers(2, 5))
    for i in range(template_count):
        size = int(rng.choice([3, 5]))
        marker_color = palette[(i * 2) % len(palette)]
        stamp = _random_template_stamp(rng, size, marker_color)
        if cursor + size + 2 >= w:
            break
        if base_row + size >= h:
            break
        col = cursor
        grid[base_row : base_row + size, col : col + size] = stamp
        template_specs.append((base_row, col, stamp))
        cursor += size + int(rng.integers(1, 3))

    template_count = len(template_specs)
    if template_count == 0:
        return grid

    # Keep a margin below templates for anchor placement.
    start_row = base_row + 6
    if start_row >= h - 2:
        start_row = 2

    # Create colored anchor pairs; each color appears exactly twice.
    for i in range(template_count):
        color = palette[(i + template_count) % len(palette)]
        attempts = 80
        placed = False
        for _ in range(attempts):
            horizontal = bool(rng.integers(0, 2))
            if horizontal:
                dist = int(rng.integers(2, 8))
                if w < 8:
                    continue
                r = int(rng.integers(start_row, h - 1))
                c1 = int(rng.integers(1, w - dist - 1))
                c2 = c1 + dist
            else:
                dist = int(rng.integers(2, 8))
                if h - start_row < dist + 2:
                    continue
                r1 = int(rng.integers(start_row, h - dist - 1))
                r2 = r1 + dist
                c1 = int(rng.integers(1, w - 1))
                c2 = c1

            if _cell_safe(grid, r if horizontal else r1, c1) and _cell_safe(
                grid, r if horizontal else r2, c2
            ):
                if horizontal:
                    grid[r, c1] = color
                    grid[r, c2] = color
                else:
                    grid[r1, c1] = color
                    grid[r2, c2] = color
                placed = True
                break
        if not placed:
            # Fallback: skip pair if no valid placement was found.
            continue

    return grid


class _Template:
    def __init__(self, rows, cols, stamp, left: int, right: int):
        self.rows = rows
        self.cols = cols
        self.stamp = stamp
        self.left = left
        self.right = right


class _Pair:
    def __init__(self, color, p1, p2, center_row: float, center_col: float):
        self.color = color
        self.p1 = p1
        self.p2 = p2
        self.center_row = center_row
        self.center_col = center_col


def _extract_templates(arr: np.ndarray):
    h, w = arr.shape
    visited = np.zeros((h, w), dtype=bool)
    templates = []
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))
    for r in range(h):
        for c in range(w):
            if arr[r, c] == 0 or visited[r, c]:
                continue

            queue = deque([(r, c)])
            visited[r, c] = True
            comp = []
            has_grey = False
            while queue:
                cr, cc = queue.popleft()
                comp.append((cr, cc))
                if arr[cr, cc] == 5:
                    has_grey = True
                for dr, dc in dirs:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and arr[nr, nc] != 0:
                        visited[nr, nc] = True
                        queue.append((nr, nc))

            if has_grey and len(comp) > 1:
                rows = np.array([x[0] for x in comp], dtype=int)
                cols = np.array([x[1] for x in comp], dtype=int)
                r0, r1 = int(rows.min()), int(rows.max())
                c0, c1 = int(cols.min()), int(cols.max())
                stamp = arr[r0 : r1 + 1, c0 : c1 + 1].copy()
                templates.append(_Template(rows, cols, stamp, c0, c1))

    return templates


def _extract_anchor_pairs(arr: np.ndarray):
    h, w = arr.shape
    colors = {}
    for r in range(h):
        for c in range(w):
            v = int(arr[r, c])
            if v == 0 or v == 5:
                continue
            colors.setdefault(v, []).append((r, c))

    pairs = []
    for color, pts in colors.items():
        if len(pts) == 2:
            (r1, c1), (r2, c2) = pts
            pairs.append(
                _Pair(
                    color=color,
                    p1=(r1, c1),
                    p2=(r2, c2),
                    center_row=(r1 + r2) / 2.0,
                    center_col=(c1 + c2) / 2.0,
                )
            )
        else:
            # Skip non-paired colors to avoid corrupting inference.
            continue

    return pairs


def _rotate_stamp(stamp: np.ndarray, pair: _Pair) -> np.ndarray:
    (r1, c1), (r2, c2) = pair.p1, pair.p2
    dr = abs(r2 - r1)
    dc = abs(c2 - c1)
    if dr > dc:
        return np.rot90(stamp)
    return stamp


def _stamp_to_center(
    grid: np.ndarray,
    stamp: np.ndarray,
    color: int,
    center_row: float,
    center_col: float,
) -> None:
    sr, sc = stamp.shape
    top = int(round(center_row)) - sr // 2
    left = int(round(center_col)) - sc // 2
    h, w = grid.shape
    for rr in range(sr):
        gr = top + rr
        if gr < 0 or gr >= h:
            continue
        for cc in range(sc):
            gc = left + cc
            if gc < 0 or gc >= w:
                continue
            if stamp[rr, cc] != 0:
                grid[gr, gc] = color


def _cell_safe(grid: np.ndarray, r: int, c: int) -> bool:
    if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]):
        return False
    return grid[r, c] == 0


def _random_template_stamp(rng: np.random.Generator, size: int, marker_color: int) -> np.ndarray:
    stamp = np.zeros((size, size), dtype=int)
    # Start from center and grow a connected grey region.
    center = (size // 2, size // 2)
    visited = {center}
    cells = [center]
    target = int(rng.integers(size, max(3, size * size - 1)))
    target = min(target, size * size)

    for _ in range(target - 1):
        r0, c0 = rng.choice(cells)
        for _ in range(8):
            dr, dc = rng.choice(np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=int))
            nr, nc = int(r0 + dr), int(c0 + dc)
            if 0 <= nr < size and 0 <= nc < size and (nr, nc) not in visited:
                visited.add((nr, nc))
                cells.append((nr, nc))
                break

    for r, c in cells:
        stamp[r, c] = 5
    stamp[center] = marker_color
    return stamp


if __name__ == "__main__":
    _sample = generate_input()
    _out = main(_sample)
    print(_sample.shape, _out.shape)
