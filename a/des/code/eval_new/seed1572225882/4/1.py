import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def _find_separator(grid):
    rows, cols = grid.shape
    for c in range(cols - 1):
        if np.all(grid[:, c] == GREY) and np.any(grid[:, c + 1 :] != BLACK):
            return "right", c
    for r in range(rows - 1):
        if np.all(grid[r, :] == GREY) and np.any(grid[r + 1 :, :] != BLACK):
            return "bottom", r
    return None, None


def _grey_components(mask):
    rows, cols = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    for sr in range(rows):
        for sc in range(cols):
            if seen[sr, sc] or not mask[sr, sc]:
                continue
            stack = [(sr, sc)]
            seen[sr, sc] = True
            cells = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            yield cells


def _parse_legend(legend):
    patterns = {}
    for cells in _grey_components(legend == GREY):
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        h, w = r1 - r0 + 1, c1 - c0 + 1
        p = h - 2
        if p not in (2, 3) or w != p + 3:
            continue
        block = legend[r0 : r1 + 1, c0 : c1 + 1]
        border_ok = (
            np.all(block[0, :] == GREY)
            and np.all(block[-1, :] == GREY)
            and np.all(block[:, 0] == GREY)
            and np.all(block[:, -1] == GREY)
        )
        if not border_ok:
            continue
        key = int(block[1, 1])
        pattern = block[1 : 1 + p, 2 : 2 + p]
        if key in (BLACK, GREY):
            continue
        if pattern.shape == (p, p) and np.all(pattern != BLACK) and np.all(pattern != GREY):
            patterns[key] = pattern.copy()
    return patterns


def _fill_regions(out, work_shape, patterns):
    rows, cols = work_shape
    seen = np.zeros((rows, cols), dtype=bool)
    for sr in range(rows):
        for sc in range(cols):
            if seen[sr, sc] or out[sr, sc] == GREY:
                continue
            queue = deque([(sr, sc)])
            seen[sr, sc] = True
            cells = []
            seeds = []
            touches_edge = False
            while queue:
                r, c = queue.popleft()
                cells.append((r, c))
                if r == 0 or c == 0 or r == rows - 1 or c == cols - 1:
                    touches_edge = True
                if out[r, c] not in (BLACK, GREY):
                    seeds.append((r, c, int(out[r, c])))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and not seen[nr, nc] and out[nr, nc] != GREY:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            if touches_edge or len(seeds) != 1:
                continue
            anchor_r, anchor_c, key = seeds[0]
            pattern = patterns.get(key)
            if pattern is None:
                continue
            ph, pw = pattern.shape
            for r, c in cells:
                out[r, c] = pattern[(r - anchor_r) % ph, (c - anchor_c) % pw]


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each closed room with the legend pattern selected by its seed color.

    Input structure:
    The grid has a black background, grey closed boundaries, and one colored seed inside each closed non-grey region.  A grey full-height or full-width divider separates a legend area on the right or bottom.  The legend contains grey-framed entries: each entry has a colored key at its inner upper-left corner and a 2x2 or 3x3 pattern immediately to the right of that key.

    Rule:
    Locate the divider, parse every legend entry into a mapping from key color to pattern, then find every enclosed non-grey component outside the legend.  If a component contains exactly one seed whose color appears in the legend, tile the matching pattern across the whole component, including over the seed, using the seed cell as the pattern anchor.  Grey boundaries and the entire legend area stay unchanged.

    Key insight:
    The seed color is only a lookup key; the actual fill comes from the legend pattern and must be aligned periodically to the seed position.
    """
    out = np.array(grid, copy=True)
    side, sep = _find_separator(out)
    if side == "right":
        patterns = _parse_legend(out[:, sep + 1 :])
        _fill_regions(out, (out.shape[0], sep), patterns)
    elif side == "bottom":
        patterns = _parse_legend(out[sep + 1 :, :])
        _fill_regions(out, (sep, out.shape[1]), patterns)
    return out


def _connected(mask):
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    start = tuple(coords[0])
    seen = {start}
    stack = [start]
    rows, cols = mask.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _make_interior(rng):
    h = rng.randint(3, 6)
    w = rng.randint(3, 6)
    mask = np.ones((h, w), dtype=bool)
    candidates = [(r, c) for r in range(h) for c in range(w) if r in (0, h - 1) or c in (0, w - 1)]
    rng.shuffle(candidates)
    removals = rng.randint(0, min(5, len(candidates)))
    removed = 0
    for r, c in candidates:
        if removed >= removals:
            break
        trial = mask.copy()
        trial[r, c] = False
        if trial.sum() >= 6 and _connected(trial) and np.all(trial.sum(axis=0) > 0) and np.all(trial.sum(axis=1) > 0):
            mask = trial
            removed += 1
    return mask


def _outline(interior_mask):
    h, w = interior_mask.shape
    interior = np.zeros((h + 2, w + 2), dtype=bool)
    interior[1 : h + 1, 1 : w + 1] = interior_mask
    boundary = np.zeros_like(interior)
    rows, cols = interior.shape
    for r in range(rows):
        for c in range(cols):
            if interior[r, c]:
                continue
            found = False
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and interior[nr, nc]:
                        found = True
            boundary[r, c] = found
    return interior, boundary


def _make_pattern(rng, size):
    a, b = rng.sample(COLORS, 2)
    style = rng.choice(("checker", "rows", "cols", "diag", "anti", "corner"))
    pattern = np.empty((size, size), dtype=int)
    for r in range(size):
        for c in range(size):
            if style == "checker":
                use_a = (r + c) % 2 == 0
            elif style == "rows":
                use_a = r % 2 == 0
            elif style == "cols":
                use_a = c % 2 == 0
            elif style == "diag":
                use_a = r == c
            elif style == "anti":
                use_a = r + c == size - 1
            else:
                use_a = r == 0 or c == 0
            pattern[r, c] = a if use_a else b
    return pattern


def _draw_legend_entry(grid, top, left, key, pattern):
    p = pattern.shape[0]
    h, w = p + 2, p + 3
    grid[top, left : left + w] = GREY
    grid[top + h - 1, left : left + w] = GREY
    grid[top : top + h, left] = GREY
    grid[top : top + h, left + w - 1] = GREY
    grid[top + 1, left + 1] = key
    grid[top + 1 : top + 1 + p, left + 2 : left + 2 + p] = pattern


def _place_regions(rng, grid, work_shape, keys):
    rows, cols = work_shape
    occupied = np.zeros((rows, cols), dtype=bool)
    for key in keys:
        placed = False
        for _ in range(200):
            interior_mask = _make_interior(rng)
            interior, boundary = _outline(interior_mask)
            footprint = interior | boundary
            h, w = footprint.shape
            if rows - h <= 2 or cols - w <= 2:
                continue
            top = rng.randint(1, rows - h - 1)
            left = rng.randint(1, cols - w - 1)
            pad_top = max(0, top - 1)
            pad_left = max(0, left - 1)
            pad_bottom = min(rows, top + h + 1)
            pad_right = min(cols, left + w + 1)
            if np.any(occupied[pad_top:pad_bottom, pad_left:pad_right]):
                continue
            view = grid[top : top + h, left : left + w]
            view[boundary] = GREY
            seed_cells = np.argwhere(interior)
            seed_r, seed_c = seed_cells[rng.randrange(len(seed_cells))]
            view[seed_r, seed_c] = key
            occupied[top : top + h, left : left + w] |= footprint
            placed = True
            break
        if not placed:
            return False
    return True


def generate_input() -> np.ndarray:
    rng = random
    for _ in range(300):
        count = rng.randint(2, 4)
        keys = rng.sample(COLORS, count)
        sizes = [rng.choice((2, 3)) for _ in range(count)]
        patterns = [_make_pattern(rng, p) for p in sizes]
        frame_heights = [p + 2 for p in sizes]
        frame_widths = [p + 3 for p in sizes]
        side = rng.choice(("right", "bottom"))

        if side == "right":
            legend_w = max(frame_widths) + 2
            legend_content_h = sum(frame_heights) + count - 1
            legend_h = legend_content_h + 2
            total_h = rng.randint(max(17, legend_h), 30)
            work_w = rng.randint(16, 30 - 1 - legend_w)
            total_w = work_w + 1 + legend_w
            grid = np.zeros((total_h, total_w), dtype=int)
            grid[:, work_w] = GREY
            top = rng.randint(1, total_h - legend_content_h - 1)
            left = work_w + 2
            for key, pattern in zip(keys, patterns):
                _draw_legend_entry(grid, top, left, key, pattern)
                top += pattern.shape[0] + 3
            if _place_regions(rng, grid, (total_h, work_w), keys):
                return grid
        else:
            legend_h = max(frame_heights) + 2
            legend_content_w = sum(frame_widths) + count - 1
            legend_w = legend_content_w + 2
            total_w = rng.randint(max(17, legend_w), 30)
            work_h = rng.randint(16, 30 - 1 - legend_h)
            total_h = work_h + 1 + legend_h
            grid = np.zeros((total_h, total_w), dtype=int)
            grid[work_h, :] = GREY
            top = work_h + 2
            left = rng.randint(1, total_w - legend_content_w - 1)
            for key, pattern in zip(keys, patterns):
                _draw_legend_entry(grid, top, left, key, pattern)
                left += pattern.shape[0] + 4
            if _place_regions(rng, grid, (work_h, total_w), keys):
                return grid
    raise RuntimeError("failed to generate a valid grid")
