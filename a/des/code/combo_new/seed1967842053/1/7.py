import random

import numpy as np


BLACK = 0
GREY = 5


def _find_colored_rectangles(area, forbidden):
    h, w = area.shape
    seen = np.zeros_like(area, dtype=bool)
    rects = []
    for r in range(h):
        for c in range(w):
            color = int(area[r, c])
            if color in forbidden or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(area[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rs = [p[0] for p in cells]
            cs = [p[1] for p in cells]
            r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
            if r1 - r0 >= 2 and c1 - c0 >= 2:
                border = np.concatenate(
                    [
                        area[r0, c0 : c1 + 1],
                        area[r1, c0 : c1 + 1],
                        area[r0 : r1 + 1, c0],
                        area[r0 : r1 + 1, c1],
                    ]
                )
                if np.all(border == color):
                    rects.append((r0, r1, c0, c1, color))
    return rects


def _separator_row(grid):
    has_sample_key = []
    for r in range(grid.shape[0]):
        row = grid[r]
        nonzero = row[row != BLACK]
        if len(nonzero) and np.any(nonzero != GREY):
            has_sample_key.append(r)
    if not has_sample_key:
        return 0
    last_sample = max(r for r in has_sample_key if r < grid.shape[0] - 1)
    for r in range(last_sample + 1, grid.shape[0]):
        if np.all(grid[r] == BLACK):
            return r
    return last_sample + 1


def _extract_patterns(grid, sep):
    patterns = {}
    top = grid[:sep]
    for r0, r1, c0, c1, color in _find_colored_rectangles(top, {BLACK, GREY}):
        inner = top[r0 + 1 : r1, c0 + 1 : c1 + 1]
        if inner.size:
            patterns[color] = inner.copy()
    return patterns


def _find_fences(grid, start_row):
    area = grid[start_row:]
    mask = area == GREY
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    fences = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and mask[nr, nc]
                        and not seen[nr, nc]
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rs = [p[0] for p in cells]
            cs = [p[1] for p in cells]
            r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
            if r1 - r0 >= 2 and c1 - c0 >= 2:
                border_count = 2 * (r1 - r0 + 1) + 2 * (c1 - c0 + 1) - 4
                if len(cells) == border_count:
                    fences.append((start_row + r0, start_row + r1, c0, c1))
    return fences


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each grey fenced region with the texture keyed by its seed color.

    Input structure:
    The background is black.  The top rows contain one or more small texture
    swatches, each enclosed by a one-pixel colored rectangular border whose
    color is the key for that swatch.  A black separator row follows the swatch
    area.  Below it are single-pixel-wide grey rectangular fences; each fence
    contains exactly one colored seed pixel and otherwise black interior cells.

    Rule:
    For every grey fence, read the seed color inside the rectangle, find the
    swatch whose border has that same color, and extract the swatch interior as
    the repeating texture.  Starting at the top-left interior cell of the fence,
    tile that texture across the whole interior, replacing both black cells and
    the seed.  The swatches, separator row, and grey fence borders are preserved.

    Key insight:
    The visible border color in the swatch area is an index key, and each seed
    uses that key to select the texture that must be tiled into its enclosure.
    """
    out = grid.copy()
    sep = _separator_row(grid)
    patterns = _extract_patterns(grid, sep)
    for r0, r1, c0, c1 in _find_fences(grid, min(sep + 1, grid.shape[0])):
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        values = inner[(inner != BLACK) & (inner != GREY)]
        if len(values) == 0:
            continue
        seed = int(values[0])
        pat = patterns.get(seed)
        if pat is None or pat.size == 0:
            continue
        ph, pw = pat.shape
        ih, iw = inner.shape
        tiled = np.fromfunction(lambda rr, cc: pat[rr % ph, cc % pw], (ih, iw), dtype=int)
        out[r0 + 1 : r1, c0 + 1 : c1] = tiled
    return out


def _place_sample(grid, top, left, key, pattern):
    ph, pw = pattern.shape
    grid[top : top + ph + 2, left] = key
    grid[top : top + ph + 2, left + pw + 1] = key
    grid[top, left : left + pw + 2] = key
    grid[top + ph + 1, left : left + pw + 2] = key
    grid[top + 1 : top + ph + 1, left + 1 : left + pw + 1] = pattern


def _place_fence(grid, top, left, height, width, seed):
    bottom = top + height - 1
    right = left + width - 1
    grid[top, left : right + 1] = GREY
    grid[bottom, left : right + 1] = GREY
    grid[top : bottom + 1, left] = GREY
    grid[top : bottom + 1, right] = GREY
    sr = random.randrange(top + 1, bottom)
    sc = random.randrange(left + 1, right)
    grid[sr, sc] = seed


def _overlaps(rect, rects, pad=1):
    r0, r1, c0, c1 = rect
    for a0, a1, b0, b1 in rects:
        if not (r1 + pad < a0 or a1 + pad < r0 or c1 + pad < b0 or b1 + pad < c0):
            return True
    return False


def generate_input() -> np.ndarray:
    sample_count = random.randint(2, 4)
    keys = random.sample([1, 2, 3, 4, 6, 7, 8, 9], sample_count)
    fill_colors = [c for c in [1, 2, 3, 4, 6, 7, 8, 9] if c not in keys]

    patterns = []
    sample_widths = []
    sample_heights = []
    max_sample_h = 0
    for _ in range(sample_count):
        ph = random.choice([2, 3])
        pw = random.choice([2, 3])
        max_sample_h = max(max_sample_h, ph + 2)
        colors = random.sample(fill_colors, min(len(fill_colors), random.choice([2, 3])))
        pat = np.array([[random.choice(colors) for _c in range(pw)] for _r in range(ph)], dtype=int)
        patterns.append(pat)
        sample_widths.append(pw + 2)
        sample_heights.append(ph + 2)

    sample_gap = 1
    sample_total_w = sum(sample_widths) + sample_gap * (sample_count - 1)
    cols = random.randint(max(sample_total_w, 18), 30)
    sample_top = random.randint(0, 1)
    sep = sample_top + max_sample_h + random.randint(0, 1)
    main_top = sep + 1

    fence_count = random.randint(sample_count, min(sample_count + 3, 6))
    for _attempt in range(300):
        rows = random.randint(max(main_top + 8, 18), 30)
        placed = []
        fences = []
        for _ in range(fence_count):
            ok = False
            for _try in range(100):
                fh = random.randint(4, 7)
                fw = random.randint(4, 8)
                if rows - fh <= main_top or cols - fw <= 0:
                    continue
                top = random.randint(main_top, rows - fh)
                left = random.randint(0, cols - fw)
                rect = (top, top + fh - 1, left, left + fw - 1)
                if not _overlaps(rect, placed, pad=1):
                    key = random.choice(keys)
                    placed.append(rect)
                    fences.append((top, left, fh, fw, key))
                    ok = True
                    break
            if not ok:
                break
        if len(fences) == fence_count:
            break
    else:
        rows = 30
        fences = []
        y = main_top
        for i in range(fence_count):
            fh, fw = 4, 5
            x = (i % 4) * 7
            if y + fh > rows or x + fw > cols:
                y += 6
                x = 0
            fences.append((y, x, fh, fw, random.choice(keys)))

    grid = np.zeros((rows, cols), dtype=int)
    x = random.randint(0, cols - sample_total_w)
    for key, pat, sw, sh in zip(keys, patterns, sample_widths, sample_heights):
        top = sample_top + random.randint(0, max_sample_h - sh)
        _place_sample(grid, top, x, key, pat)
        x += sw + sample_gap
    grid[sep, :] = BLACK

    for top, left, fh, fw, key in fences:
        _place_fence(grid, top, left, fh, fw, key)
    return grid
