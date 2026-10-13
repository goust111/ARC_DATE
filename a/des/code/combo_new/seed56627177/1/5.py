from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_separator(grid: np.ndarray, bg: int) -> tuple[int, int]:
    candidates = [r for r in range(grid.shape[0]) if grid[r, 0] != bg and np.all(grid[r, :] == grid[r, 0])]
    assert len(candidates) == 1, "expected exactly one full-width non-background separator row"
    sep_r = int(candidates[0])
    return sep_r, int(grid[sep_r, 0])


def _extract_stamp(lower: np.ndarray, stamp_color: int) -> list[tuple[int, int]]:
    comps = _components(lower == stamp_color)
    assert len(comps) == 1, "expected exactly one stamp template"
    pts = comps[0]
    r0, c0, _r1, _c1 = _bbox(pts)
    offsets = sorted((r - r0, c - c0) for r, c in pts)
    return offsets


def _path_adjacency(path_mask: np.ndarray) -> dict[tuple[int, int], list[tuple[int, int]]]:
    coords = [tuple(map(int, pt)) for pt in np.argwhere(path_mask)]
    cells = set(coords)
    adj: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in coords:
        adj[(r, c)] = [(nr, nc) for nr, nc in _neighbors4(r, c) if (nr, nc) in cells]
    return adj


def _trace_path(adj: dict[tuple[int, int], list[tuple[int, int]]], start: tuple[int, int]) -> list[tuple[int, int]]:
    seq = [start]
    prev: tuple[int, int] | None = None
    cur = start
    while True:
        nxt = [cell for cell in adj[cur] if cell != prev]
        assert len(nxt) <= 1, "path must not branch"
        if not nxt:
            break
        prev, cur = cur, nxt[0]
        seq.append(cur)
    assert len(seq) == len(adj), "path must be a single simple chain"
    return seq


def _color_pattern(seq: list[tuple[int, int]], grid: np.ndarray) -> tuple[int, ...]:
    mapping: dict[int, int] = {}
    next_label = 0
    pattern: list[int] = []
    for r, c in seq:
        color = int(grid[r, c])
        if color not in mapping:
            mapping[color] = next_label
            next_label += 1
        pattern.append(mapping[color])
    return tuple(pattern)


def _path_signature(seq: list[tuple[int, int]], grid: np.ndarray) -> tuple[tuple[int, int, int], ...]:
    r0, c0 = seq[0]
    pattern = _color_pattern(seq, grid)
    return tuple((r - r0, c - c0, pattern[i]) for i, (r, c) in enumerate(seq))


def _ordered_path(path_grid: np.ndarray) -> list[tuple[int, int]]:
    adj = _path_adjacency(path_grid != BG)
    endpoints = sorted(cell for cell, nbrs in adj.items() if len(nbrs) == 1)
    assert len(endpoints) == 2, "path must have exactly two endpoints"
    seq_a = _trace_path(adj, endpoints[0])
    seq_b = _trace_path(adj, endpoints[1])
    sig_a = _path_signature(seq_a, path_grid)
    sig_b = _path_signature(seq_b, path_grid)
    if sig_a < sig_b:
        return seq_a
    if sig_b < sig_a:
        return seq_b
    return seq_a if endpoints[0] <= endpoints[1] else seq_b


def _segment_direction(
    seq: list[tuple[int, int]],
    start: int,
    end: int,
) -> tuple[int, int]:
    if end - start >= 2:
        dr = seq[start + 1][0] - seq[start][0]
        dc = seq[start + 1][1] - seq[start][1]
    elif end < len(seq):
        dr = seq[end][0] - seq[start][0]
        dc = seq[end][1] - seq[start][1]
    else:
        dr = seq[start][0] - seq[start - 1][0]
        dc = seq[start][1] - seq[start - 1][1]
    assert (dr, dc) in DIRS, "segment direction must be cardinal"
    for i in range(start + 1, end):
        step = (seq[i][0] - seq[i - 1][0], seq[i][1] - seq[i - 1][1])
        assert step == (dr, dc), "same-color segment must be straight"
    return int(dr), int(dc)


def _segments(seq: list[tuple[int, int]], path_grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    out: list[tuple[int, int, int, int]] = []
    i = 0
    while i < len(seq):
        color = int(path_grid[seq[i]])
        j = i + 1
        while j < len(seq) and int(path_grid[seq[j]]) == color:
            j += 1
        dr, dc = _segment_direction(seq, i, j)
        out.append((color, dr, dc, j - i))
        i = j
    return out


def _stamp(canvas: np.ndarray, offsets: list[tuple[int, int]], anchor: tuple[int, int], color: int) -> None:
    base_r, base_c = anchor
    h, w = canvas.shape
    for dr, dc in offsets:
        rr = base_r + dr
        cc = base_c + dc
        if 0 <= rr < h and 0 <= cc < w:
            canvas[rr, cc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Walk the lower-half color path and use each segment to stamp the grey template onto the upper canvas.

    Input structure:
    - A black grid is split by one full-width grey horizontal row into an upper canvas and a lower clue area.
    - The upper area is an empty black canvas.
    - In the lower area, a small grey connected shape near the lower-left is the stamp template, and the remaining foreground is a single 4-connected multicolor path.

    Rule:
    - Identify the stamp template from the grey component below the separator and identify the unique path from the colored pixels.
    - Choose a path endpoint as the start; when the two endpoint readings are symmetric, use the upper-left endpoint.
    - Traverse the path, splitting it whenever the color changes; each same-color straight subpath gives one step, with its cardinal direction taken from the subpath orientation and its length taken from its pixel count.
    - Maintain a cursor on the upper canvas starting at the canvas upper-left corner; after each step, move the cursor by that direction and length, then stamp the template there using the finished segment's color, clipping to the canvas and letting later stamps overwrite earlier ones.

    Key insight:
    - The path encodes movement in two channels at once: geometry gives the cursor motion, while color-change boundaries define the stamping steps and their paint colors.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    sep_r, grey = _find_separator(g, bg)
    canvas = np.full((sep_r, g.shape[1]), bg, dtype=int)
    lower = g[sep_r + 1 :, :]

    stamp_offsets = _extract_stamp(lower, grey)
    path_mask = (lower != bg) & (lower != grey)
    comps = _components(path_mask)
    assert len(comps) == 1, "expected exactly one colored path"

    path_grid = np.where(path_mask, lower, bg)
    ordered = _ordered_path(path_grid)
    cursor_r = 0
    cursor_c = 0
    for color, dr, dc, length in _segments(ordered, path_grid):
        cursor_r += dr * length
        cursor_c += dc * length
        _stamp(canvas, stamp_offsets, (cursor_r, cursor_c), color)
    return canvas


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _random_connected_mask(height: int, width: int, area: int) -> np.ndarray:
    while True:
        mask = np.zeros((height, width), dtype=bool)
        start = (random.randrange(height), random.randrange(width))
        cells = {start}
        mask[start] = True
        while len(cells) < area:
            frontier = []
            for r, c in list(cells):
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            pick = random.choice(frontier)
            cells.add(pick)
            mask[pick] = True
        if int(mask.sum()) != area:
            continue
        cropped = _crop_mask(mask)
        if cropped.shape[0] >= 2 and cropped.shape[1] >= 2:
            return cropped


def _expand_blocked(blocked: np.ndarray) -> np.ndarray:
    h, w = blocked.shape
    out = blocked.copy()
    for r in range(h):
        for c in range(w):
            if not blocked[r, c]:
                continue
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr, nc] = True
    return out


def _can_add_cell(
    cell: tuple[int, int],
    prev: tuple[int, int],
    occupied: set[tuple[int, int]],
    allowed: np.ndarray,
) -> bool:
    r, c = cell
    if not (0 <= r < allowed.shape[0] and 0 <= c < allowed.shape[1]):
        return False
    if not allowed[r, c] or cell in occupied:
        return False
    for nr, nc in _neighbors4(r, c):
        if (nr, nc) in occupied and (nr, nc) != prev:
            return False
    return True


def _segment_candidates(
    current: tuple[int, int],
    occupied: set[tuple[int, int]],
    allowed: np.ndarray,
    seg_idx: int,
    prev_dir: tuple[int, int] | None,
) -> list[tuple[tuple[int, int], int, list[tuple[int, int]]]]:
    dirs = DIRS[:]
    random.shuffle(dirs)
    out: list[tuple[tuple[int, int], int, list[tuple[int, int]]]] = []
    for dr, dc in dirs:
        if prev_dir is not None and (dr, dc) in {prev_dir, (-prev_dir[0], -prev_dir[1])}:
            continue
        new_cells: list[tuple[int, int]] = []
        temp_occupied = set(occupied)
        prev = current
        pos = current
        max_added = random.randint(3, 6)
        for _ in range(max_added):
            pos = (pos[0] + dr, pos[1] + dc)
            if not _can_add_cell(pos, prev, temp_occupied, allowed):
                break
            new_cells.append(pos)
            temp_occupied.add(pos)
            prev = pos
            total_len = len(new_cells) + (1 if seg_idx == 0 else 0)
            if total_len >= 2:
                out.append(((dr, dc), total_len, new_cells.copy()))
    random.shuffle(out)
    return out


def _build_path_pixels(
    allowed: np.ndarray,
    n_segments: int,
) -> list[tuple[int, int]] | None:
    starts = [tuple(map(int, pt)) for pt in np.argwhere(allowed)]
    random.shuffle(starts)
    for start in starts[: min(len(starts), 50)]:
        pixels = [start]
        occupied = {start}

        def rec(
            seg_idx: int,
            current: tuple[int, int],
            prev_dir: tuple[int, int] | None,
            path_pixels: list[tuple[int, int]],
            used_dirs: list[tuple[int, int]],
        ) -> list[tuple[int, int]] | None:
            if seg_idx == n_segments:
                if len({d for d in used_dirs}) < 2:
                    return None
                return path_pixels
            options = _segment_candidates(current, occupied, allowed, seg_idx, prev_dir)
            for direction, _length, new_cells in options:
                old_size = len(path_pixels)
                path_pixels.extend(new_cells)
                for cell in new_cells:
                    occupied.add(cell)
                used_dirs.append(direction)
                result = rec(seg_idx + 1, new_cells[-1], direction, path_pixels, used_dirs)
                if result is not None:
                    return result
                used_dirs.pop()
                for cell in new_cells:
                    occupied.remove(cell)
                del path_pixels[old_size:]
            return None

        built = rec(0, start, None, pixels, [])
        if built is not None:
            return built
    return None


def _assign_segment_colors(n_segments: int) -> list[int]:
    color_count = random.randint(2, min(4, n_segments))
    base_colors = random.sample(PALETTE, color_count)
    colors: list[int] = []
    unused = base_colors[:]
    for i in range(n_segments):
        choices = [c for c in base_colors if c != (colors[-1] if colors else None)]
        if unused:
            forced = [c for c in unused if c in choices]
            color = random.choice(forced if forced else choices)
            if color in unused:
                unused.remove(color)
        else:
            color = random.choice(choices)
        colors.append(color)
    return colors


def _paint_path(
    lower: np.ndarray,
    pixels: list[tuple[int, int]],
    seg_lengths: list[int],
    seg_colors: list[int],
) -> None:
    idx = 0
    for seg_i, length in enumerate(seg_lengths):
        count = length if seg_i == 0 else length
        for _ in range(count):
            r, c = pixels[idx]
            lower[r, c] = seg_colors[seg_i]
            idx += 1
    assert idx == len(pixels)


def _segment_lengths_from_pixels(pixels: list[tuple[int, int]]) -> list[int]:
    lengths: list[int] = []
    start = 0
    while start < len(pixels):
        if start == len(pixels) - 1:
            return []
        dr = pixels[start + 1][0] - pixels[start][0]
        dc = pixels[start + 1][1] - pixels[start][1]
        end = start + 1
        while end < len(pixels):
            step = (pixels[end][0] - pixels[end - 1][0], pixels[end][1] - pixels[end - 1][1])
            if step != (dr, dc):
                break
            end += 1
        lengths.append(end - start)
        start = end
    return lengths


def _place_stamp_template(lower: np.ndarray) -> tuple[list[tuple[int, int]], np.ndarray]:
    while True:
        base_h = random.randint(2, 5)
        base_w = random.randint(2, 5)
        max_area = max(4, min(base_h * base_w, 8))
        area = random.randint(3, max_area)
        mask = _random_connected_mask(base_h, base_w, area)
        th, tw = mask.shape
        left_margin = random.randint(0, 1)
        bottom_margin = random.randint(0, 1)
        top = lower.shape[0] - th - bottom_margin
        left = left_margin
        if top < 0 or left + tw > lower.shape[1]:
            continue
        lower[top : top + th, left : left + tw][mask] = GREY
        blocked = np.zeros_like(lower, dtype=bool)
        blocked[top : top + th, left : left + tw] |= mask
        return [(top + r, left + c) for r, c in zip(*np.where(mask))], blocked


def _render_input() -> np.ndarray | None:
    canvas_h = random.randint(8, 14)
    lower_h = random.randint(7, 14)
    width = random.randint(14, 28)
    if canvas_h + 1 + lower_h > 30:
        return None

    lower = np.full((lower_h, width), BG, dtype=int)
    _stamp_cells, stamp_blocked = _place_stamp_template(lower)
    blocked = _expand_blocked(stamp_blocked)

    path_col_min = max(4, int(np.where(blocked.any(axis=0))[0].max()) + 2)
    if path_col_min >= width - 3:
        return None

    allowed = np.ones_like(lower, dtype=bool)
    allowed[:, :path_col_min] = False
    allowed &= ~blocked

    n_segments = random.randint(3, 6)
    pixels = _build_path_pixels(allowed, n_segments)
    if pixels is None:
        return None

    seg_lengths = _segment_lengths_from_pixels(pixels)
    if len(seg_lengths) != n_segments or min(seg_lengths) < 2:
        return None

    seg_colors = _assign_segment_colors(n_segments)
    _paint_path(lower, pixels, seg_lengths, seg_colors)

    grid = np.full((canvas_h + 1 + lower_h, width), BG, dtype=int)
    grid[canvas_h, :] = GREY
    grid[canvas_h + 1 :, :] = lower

    out = main(grid)
    visible_colors = {int(v) for v in np.unique(out) if int(v) != BG}
    if len(visible_colors) < 2:
        return None
    if not np.any(out != BG):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _render_input()
        if grid is None:
            continue
        try:
            out = main(grid)
        except AssertionError:
            continue
        if grid.ndim == 2 and out.ndim == 2 and max(grid.shape) <= 30 and max(out.shape) <= 30:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
