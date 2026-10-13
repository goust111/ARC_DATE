import numpy as np


def _mode_color(grid):
    colors, counts = np.unique(grid, return_counts=True)
    return colors[int(np.argmax(counts))]


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(comp, dtype=int))
    return comps


def _is_line(coords):
    if len(coords) <= 1:
        return True
    return np.all(coords[:, 0] == coords[0, 0]) or np.all(coords[:, 1] == coords[0, 1])


def _body_and_markers(grid, coords):
    vals = grid[coords[:, 0], coords[:, 1]]
    colors, counts = np.unique(vals, return_counts=True)
    if len(colors) == 1:
        return colors[0], np.empty((0, 2), dtype=int)

    candidates = []
    for color, count in zip(colors, counts):
        pos = coords[vals == color]
        if 1 <= count <= 4 and _is_line(pos) and len(coords) - count > count:
            candidates.append(color)

    if len(candidates) == 1:
        marker = candidates[0]
        body = colors[colors != marker][0]
    else:
        body = colors[int(np.argmax(counts))]
    return body, coords[vals != body]


def main(grid: np.ndarray) -> np.ndarray:
    """Copy each marked object as many times as its internal chromosome pixels indicate.

    Input structure:
    The most common color is background. Each non-background connected component is one
    single-body-color object whose interior contains one to four pixels of a shared
    marker color. The marker pixels in an object form either one horizontal segment or
    one vertical segment; objects are separated and all resulting copies fit in bounds.

    Rule:
    For each object, count its marker pixels. That count is the number of copies. If the
    marker segment is horizontal, place those copies left to right with one background
    column between adjacent bounding boxes; if it is vertical, place them top to bottom
    with one background row between adjacent bounding boxes. The first copy stays at the
    original position. All marker pixels are repainted to the object's body color.

    Key insight:
    The small differently colored segment is not noise: its length gives the copy count,
    and its orientation gives the direction in which the restored copies are laid out.
    """
    grid = np.asarray(grid)
    bg = _mode_color(grid)
    out = np.full_like(grid, bg)

    for coords in _components(grid != bg):
        body, markers = _body_and_markers(grid, coords)
        n = max(1, len(markers))

        top, left = coords.min(axis=0)
        bottom, right = coords.max(axis=0)
        height = bottom - top + 1
        width = right - left + 1

        if len(markers) <= 1:
            vertical = False
        else:
            vertical = np.all(markers[:, 1] == markers[0, 1])

        for i in range(n):
            dr = i * (height + 1) if vertical else 0
            dc = i * (width + 1) if not vertical else 0
            rr = coords[:, 0] + dr
            cc = coords[:, 1] + dc
            in_bounds = (0 <= rr) & (rr < out.shape[0]) & (0 <= cc) & (cc < out.shape[1])
            out[rr[in_bounds], cc[in_bounds]] = body

    return out


def _reserve(cells, reserved):
    h, w = reserved.shape
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    reserved[nr, nc] = True


def _clear_for(cells, reserved):
    return all(not reserved[r, c] for r, c in cells)


def _make_mask(rng, height, width, marker_rel):
    mask = np.ones((height, width), dtype=bool)
    marker_set = set(marker_rel)
    if height >= 3 and width >= 3:
        corners = [(0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1)]
        removable = [p for p in corners if p not in marker_set]
        rng.shuffle(removable)
        for r, c in removable[: int(rng.integers(0, min(3, len(removable) + 1)))]:
            mask[r, c] = False
    return mask


def _random_object(rng, rows, cols):
    orient_horizontal = bool(rng.integers(0, 2))
    n = int(rng.integers(1, 5))

    if orient_horizontal:
        max_width = (cols - (n - 1)) // n
        if max_width < max(2, n):
            return None
        width = int(rng.integers(max(2, n), min(6, max_width) + 1))
        height = int(rng.integers(2, min(6, rows) + 1))
        if height * width <= 2 * n:
            return None
        marker_row = int(rng.integers(1, height - 1)) if height >= 3 else int(rng.integers(0, height))
        start = int(rng.integers(0, width - n + 1))
        marker_rel = [(marker_row, start + j) for j in range(n)]
    else:
        max_height = (rows - (n - 1)) // n
        if max_height < max(2, n):
            return None
        height = int(rng.integers(max(2, n), min(6, max_height) + 1))
        width = int(rng.integers(2, min(6, cols) + 1))
        if height * width <= 2 * n:
            return None
        marker_col = int(rng.integers(1, width - 1)) if width >= 3 else int(rng.integers(0, width))
        start = int(rng.integers(0, height - n + 1))
        marker_rel = [(start + j, marker_col) for j in range(n)]

    mask = _make_mask(rng, height, width, marker_rel)
    if np.count_nonzero(mask) - n <= n:
        return None
    return orient_horizontal, n, mask, marker_rel


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    colors = list(range(10))

    for _ in range(200):
        rows = int(rng.integers(14, 31))
        cols = int(rng.integers(14, 31))
        bg = int(rng.choice(colors))
        marker = int(rng.choice([c for c in colors if c != bg]))
        body_pool = [c for c in colors if c not in (bg, marker)]
        rng.shuffle(body_pool)
        target_objects = int(rng.integers(2, 5))

        grid = np.full((rows, cols), bg, dtype=int)
        reserved = np.zeros((rows, cols), dtype=bool)
        placed = 0

        for body in body_pool[:target_objects]:
            success = False
            for _attempt in range(120):
                obj = _random_object(rng, rows, cols)
                if obj is None:
                    continue
                horizontal, n, mask, marker_rel = obj
                height, width = mask.shape
                total_h = height if horizontal else height + (n - 1) * (height + 1)
                total_w = width + (n - 1) * (width + 1) if horizontal else width
                if total_h > rows or total_w > cols:
                    continue
                top = int(rng.integers(0, rows - total_h + 1))
                left = int(rng.integers(0, cols - total_w + 1))

                rel_cells = np.argwhere(mask)
                all_copy_cells = []
                for i in range(n):
                    dr = 0 if horizontal else i * (height + 1)
                    dc = i * (width + 1) if horizontal else 0
                    for rr, cc in rel_cells:
                        all_copy_cells.append((top + dr + int(rr), left + dc + int(cc)))
                if not _clear_for(all_copy_cells, reserved):
                    continue

                for rr, cc in rel_cells:
                    grid[top + int(rr), left + int(cc)] = body
                for rr, cc in marker_rel:
                    grid[top + rr, left + cc] = marker

                _reserve(all_copy_cells, reserved)
                placed += 1
                success = True
                break
            if not success:
                break

        if placed == target_objects:
            vals, counts = np.unique(grid, return_counts=True)
            bg_count = counts[vals == bg][0]
            if bg_count == counts.max():
                return grid

    grid = np.full((18, 18), 0, dtype=int)
    grid[2:5, 2:6] = 3
    grid[3, 3:5] = 4
    grid[9:13, 10:13] = 7
    grid[10:13, 11] = 4
    return grid
