from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
WALL = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _mode_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _parse_legend(grid: np.ndarray, bg: int) -> dict[int, np.ndarray]:
    h, w = grid.shape
    assert h >= 3, "legend requires a bottom 3-row strip"
    top = h - 3
    mapping: dict[int, np.ndarray] = {}
    c = 0
    while c <= w - 3:
        if int(grid[top, c]) == bg:
            c += 1
            continue
        tile = np.asarray(grid[top : top + 3, c : c + 3], dtype=int)
        assert tile.shape == (3, 3), "legend tile must be 3x3"
        seed = int(tile[0, 0])
        mapping[seed] = tile.copy()
        c += 4
    assert mapping, "legend mapping must be non-empty"
    return mapping


def _border_connected(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if mask[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if mask[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return seen


def _enclosed_regions(grid: np.ndarray, bg: int, wall: int) -> list[list[tuple[int, int]]]:
    top = grid.shape[0] - 3
    body = np.asarray(grid[:top], dtype=int)
    open_mask = body != wall
    outside = _border_connected(open_mask)
    visited = outside.copy()
    regions: list[list[tuple[int, int]]] = []
    h, w = body.shape

    for r in range(h):
        for c in range(w):
            if not open_mask[r, c] or visited[r, c]:
                continue
            visited[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and open_mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        q.append((nr, nc))
            regions.append(cells)
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed seeded region with the legend texture keyed by its seed color.

    Input structure:
    - bg is the most common color, and the bottom 3 rows form a legend strip of 3x3 texture tiles.
    - Above the legend, grey 1-pixel outlines surround several 4-connected enclosed regions on the background.
    - Each enclosed region contains exactly one non-grey seed pixel whose color also appears as the top-left pixel of one legend tile.

    Rule:
    - Read the bottom legend strip and build a seed-color -> 3x3 texture lookup.
    - Treat grey cells as walls; among all non-wall cells above the legend, discard the border-connected outside area and keep the remaining enclosed components.
    - For each enclosed component, find its unique seed pixel, look up the matching 3x3 texture, and tile that texture across the whole component with the seed position acting as texture coordinate (0,0).
    - Leave the grey outlines, the outside background, and the legend strip unchanged.

    Key insight:
    - The seed pixel is both a lookup key and a phase anchor: it selects the texture and fixes the tiling offset used inside its surrounding closed region.
    """
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)
    body = g[:-3]
    counts = Counter(map(int, body.flatten()))
    wall_candidates = [color for color in counts if color != bg]
    assert wall_candidates, "expected non-background wall color above legend"
    wall = max(wall_candidates, key=lambda color: counts[color])

    mapping = _parse_legend(g, bg)
    regions = _enclosed_regions(g, bg, wall)
    out = g.copy()

    for cells in regions:
        seeds = [(r, c) for r, c in cells if int(g[r, c]) not in (bg, wall)]
        assert len(seeds) == 1, "each enclosed region must contain exactly one seed"
        sr, sc = seeds[0]
        seed_color = int(g[sr, sc])
        assert seed_color in mapping, "seed color must appear in the legend"
        texture = mapping[seed_color]
        for r, c in cells:
            out[r, c] = int(texture[(r - sr) % 3, (c - sc) % 3])
    return out


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    return np.asarray(mask[rows][:, cols], dtype=bool)


def _is_connected(mask: np.ndarray) -> bool:
    pts = list(map(tuple, np.argwhere(mask)))
    if not pts:
        return False
    seen = {pts[0]}
    q: deque[tuple[int, int]] = deque([pts[0]])
    h, w = mask.shape
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(int(r), int(c)):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _has_hole(mask: np.ndarray) -> bool:
    empty = ~mask
    padded = np.pad(empty, 1, constant_values=True)
    outside = _border_connected(padded)
    return bool(np.any(padded & ~outside))


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _rect_mask() -> np.ndarray:
    return np.ones((random.randint(3, 7), random.randint(3, 7)), dtype=bool)


def _notched_mask() -> np.ndarray:
    h = random.randint(4, 8)
    w = random.randint(4, 8)
    mask = np.ones((h, w), dtype=bool)
    notch_h = random.randint(1, h - 2)
    notch_w = random.randint(1, w - 2)
    corner = random.choice(("tl", "tr", "bl", "br"))
    if corner == "tl":
        mask[:notch_h, :notch_w] = False
    elif corner == "tr":
        mask[:notch_h, w - notch_w :] = False
    elif corner == "bl":
        mask[h - notch_h :, :notch_w] = False
    else:
        mask[h - notch_h :, w - notch_w :] = False
    return _trim_mask(mask)


def _union_rects_mask() -> np.ndarray | None:
    h = random.randint(5, 8)
    w = random.randint(5, 8)
    mask = np.zeros((h, w), dtype=bool)

    r1h = random.randint(2, h - 1)
    r1w = random.randint(2, w - 1)
    r1r = random.randint(0, h - r1h)
    r1c = random.randint(0, w - r1w)
    mask[r1r : r1r + r1h, r1c : r1c + r1w] = True

    for _ in range(60):
        r2h = random.randint(2, h - 1)
        r2w = random.randint(2, w - 1)
        r2r = random.randint(0, h - r2h)
        r2c = random.randint(0, w - r2w)
        rect2 = np.zeros((h, w), dtype=bool)
        rect2[r2r : r2r + r2h, r2c : r2c + r2w] = True
        if not np.any(rect2 & mask) or not np.any(rect2 & ~mask):
            continue
        union = mask | rect2
        union = _trim_mask(union)
        if _is_connected(union) and not _has_hole(union):
            return union
    return None


def _polyomino_mask() -> np.ndarray | None:
    h = random.randint(4, 8)
    w = random.randint(4, 8)
    target = random.randint(6, min(h * w - 1, 18))
    start = (random.randrange(h), random.randrange(w))
    cells = {start}

    while len(cells) < target:
        frontier = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        cells.add(random.choice(frontier))
        if random.random() < 0.18 and frontier:
            cells.add(random.choice(frontier))

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    mask = _trim_mask(mask)
    if not _is_connected(mask) or _has_hole(mask):
        return None
    return mask


def _random_region_mask() -> np.ndarray | None:
    factories = (_rect_mask, _notched_mask, _union_rects_mask, _polyomino_mask)
    for _ in range(80):
        mask = random.choice(factories)()
        if mask is None:
            continue
        area = int(mask.sum())
        if area < 6 or area > 24:
            continue
        if min(mask.shape) < 2:
            continue
        if not _is_connected(mask) or _has_hole(mask):
            continue
        return mask
    return None


def _random_texture(seed_color: int) -> np.ndarray:
    tile = np.full((3, 3), BG, dtype=int)
    tile[0, 0] = seed_color
    others = [(r, c) for r in range(3) for c in range(3) if (r, c) != (0, 0)]
    extra_count = random.randint(1, 5)
    for r, c in random.sample(others, extra_count):
        tile[r, c] = seed_color
    return tile


def _build_region_patch(mask: np.ndarray, seed_color: int) -> np.ndarray:
    inner = np.pad(mask, 1, constant_values=False)
    wall_mask = _dilate4(inner) & ~inner
    patch = np.full(inner.shape, BG, dtype=int)
    patch[wall_mask] = WALL
    cells = list(map(tuple, np.argwhere(inner)))
    sr, sc = random.choice(cells)
    patch[int(sr), int(sc)] = seed_color
    return patch


def _build_legend(colors: list[int], textures: dict[int, np.ndarray]) -> np.ndarray:
    order = colors[:]
    random.shuffle(order)
    width = 4 * len(order) - 1
    legend = np.full((3, width), BG, dtype=int)
    c = 0
    for color in order:
        legend[:, c : c + 3] = textures[color]
        c += 4
    return legend


def _compose_body(patches: list[np.ndarray]) -> np.ndarray | None:
    shuffled = patches[:]
    random.shuffle(shuffled)
    n = len(shuffled)
    if n <= 2:
        row_count = 1 if random.random() < 0.7 else 2
    elif n == 3:
        row_count = random.choice((1, 2))
    else:
        row_count = 2

    rows: list[list[np.ndarray]] = [[] for _ in range(row_count)]
    est_widths = [0] * row_count
    for patch in sorted(shuffled, key=lambda arr: arr.shape[1], reverse=True):
        idx = min(range(row_count), key=lambda i: est_widths[i])
        rows[idx].append(patch)
        est_widths[idx] += patch.shape[1] + 3

    rows = [row for row in rows if row]
    row_heights = [max(p.shape[0] for p in row) for row in rows]
    row_gaps = [[random.randint(2, 4) for _ in range(len(row) - 1)] for row in rows]
    row_widths = [
        sum(p.shape[1] for p in row) + sum(gaps)
        for row, gaps in zip(rows, row_gaps)
    ]

    top_pad = random.randint(1, 3)
    bottom_pad = random.randint(1, 3)
    left_pad = random.randint(1, 3)
    right_pad = random.randint(1, 3)
    between_rows = [random.randint(2, 4) for _ in range(len(rows) - 1)]

    body_h = top_pad + sum(row_heights) + sum(between_rows) + bottom_pad
    body_w = left_pad + max(row_widths) + right_pad
    if body_h > 27 or body_w > 30:
        return None

    body = np.full((body_h, body_w), BG, dtype=int)
    y = top_pad
    for row, row_h, gaps in zip(rows, row_heights, row_gaps):
        x = left_pad
        for i, patch in enumerate(row):
            jitter = random.randint(0, row_h - patch.shape[0])
            body[y + jitter : y + jitter + patch.shape[0], x : x + patch.shape[1]] = patch
            x += patch.shape[1]
            if i < len(gaps):
                x += gaps[i]
        y += row_h
        if between_rows:
            y += between_rows.pop(0)
    return body


def _is_valid_candidate(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if _mode_color(grid) != BG:
        return False
    body = grid[:-3]
    counts = Counter(map(int, body.flatten()))
    non_bg = [color for color in counts if color != BG]
    if not non_bg or max(non_bg, key=lambda color: counts[color]) != WALL:
        return False
    try:
        out = main(grid)
    except Exception:
        return False
    return bool(np.any(out != grid))


def _gen_attempt() -> np.ndarray | None:
    n_regions = random.randint(2, 5)
    colors = random.sample(SEED_COLORS, n_regions)

    textures: dict[int, np.ndarray] = {}
    seen_patterns: set[tuple[int, ...]] = set()
    for color in colors:
        for _ in range(50):
            texture = _random_texture(color)
            pattern = tuple(int(x == color) for x in texture.flatten())
            if pattern in seen_patterns:
                continue
            seen_patterns.add(pattern)
            textures[color] = texture
            break
        else:
            return None

    patches = []
    for color in colors:
        mask = _random_region_mask()
        if mask is None:
            return None
        patches.append(_build_region_patch(mask, color))

    body = _compose_body(patches)
    if body is None:
        return None

    legend = _build_legend(colors, textures)
    min_width = max(body.shape[1], legend.shape[1] + random.randint(0, 4))
    if min_width > 30:
        return None
    final_w = random.randint(min_width, 30)
    body_left = random.randint(0, final_w - body.shape[1])
    legend_left = random.randint(0, final_w - legend.shape[1])

    grid = np.full((body.shape[0] + 3, final_w), BG, dtype=int)
    grid[: body.shape[0], body_left : body_left + body.shape[1]] = body
    grid[body.shape[0] :, legend_left : legend_left + legend.shape[1]] = legend

    if not _is_valid_candidate(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
