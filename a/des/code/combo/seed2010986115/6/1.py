from __future__ import annotations

from collections import Counter, deque

import numpy as np


def _mode_color(values: np.ndarray) -> int:
    colors, counts = np.unique(values, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(pts)
    return comps


def _runs(mask_1d: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    n = int(mask_1d.shape[0])
    i = 0
    while i < n:
        if not bool(mask_1d[i]):
            i += 1
            continue
        j = i
        while j < n and bool(mask_1d[j]):
            j += 1
        runs.append((int(i), int(j)))
        i = j
    return runs


def _parse_legend(grid: np.ndarray, bg: int) -> tuple[dict[int, np.ndarray], int]:
    h = int(grid.shape[0])
    r0 = 0
    while r0 < h and bool(np.all(grid[r0] == bg)):
        r0 += 1
    assert r0 < h

    r1 = r0
    while r1 < h and not bool(np.all(grid[r1] == bg)):
        r1 += 1
    assert r1 > r0

    legend = grid[r0:r1]
    col_runs = _runs(np.any(legend != bg, axis=0))
    assert col_runs
    widths = {int(c1 - c0) for c0, c1 in col_runs}
    assert len(widths) == 1
    cell_w = widths.pop()
    assert legend.shape[0] == 2 * cell_w

    mapping: dict[int, np.ndarray] = {}
    for c0, c1 in col_runs:
        block = legend[:, c0:c1]
        top = block[:cell_w]
        bottom = block[cell_w:]
        seed_color = int(top[0, 0])
        assert seed_color != bg
        assert bool((top == seed_color).all())
        mapping[seed_color] = bottom.copy()

    assert mapping
    return mapping, int(r1)


def _frame_bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    perimeter = set()
    for c in range(c0, c1 + 1):
        perimeter.add((r0, c))
        perimeter.add((r1, c))
    for r in range(r0 + 1, r1):
        perimeter.add((r, c0))
        perimeter.add((r, c1))

    if len(perimeter) != len(points):
        return None
    if any((r, c) not in perimeter for r, c in points):
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _find_frames(grid: np.ndarray, bg: int, start_row: int) -> tuple[int, list[tuple[int, int, int, int]]]:
    best_color = -1
    best_frames: list[tuple[int, int, int, int]] = []
    best_score = (-1, -1)
    lower = grid[start_row:]

    for color in np.unique(lower):
        color = int(color)
        if color == bg:
            continue
        frames: list[tuple[int, int, int, int]] = []
        for comp in _components(lower == color):
            shifted = [(int(r + start_row), int(c)) for r, c in comp]
            bbox = _frame_bbox(shifted)
            if bbox is None:
                continue
            r0, c0, r1, c1 = bbox
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            frames.append(bbox)
        score = (len(frames), sum((r1 - r0 + 1) * (c1 - c0 + 1) for r0, c0, r1, c1 in frames))
        if score > best_score:
            best_score = score
            best_color = color
            best_frames = frames

    assert best_color != -1 and best_frames
    best_frames.sort()
    return best_color, best_frames


def _tile_pattern(pattern: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    ph, pw = map(int, pattern.shape)
    h, w = map(int, shape)
    reps = ((h + ph - 1) // ph, (w + pw - 1) // pw)
    return np.tile(pattern, reps)[:h, :w]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use each region's seed color as a key into the top legend, then tile the matched texture.
    """
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)

    texture_map, region_start = _parse_legend(g, bg)
    frame_color, frames = _find_frames(g, bg, region_start)

    out = g.copy()
    for r0, c0, r1, c1 in frames:
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        seeds = np.argwhere((interior != bg) & (interior != frame_color))
        assert len(seeds) == 1
        sr, sc = map(int, seeds[0])
        seed_color = int(interior[sr, sc])
        assert seed_color in texture_map
        pattern = texture_map[seed_color]
        out[r0 + 1 : r1, c0 + 1 : c1] = _tile_pattern(pattern, interior.shape)

    return out


_BG = 0
_FRAME = 5
_VISIBLE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
_PATTERNS_2 = [
    np.array([[1, 0], [0, 1]], dtype=int),
    np.array([[0, 1], [1, 0]], dtype=int),
    np.array([[1, 0], [1, 0]], dtype=int),
    np.array([[1, 1], [0, 0]], dtype=int),
]
_PATTERNS_3 = [
    np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=int),
    np.array([[1, 0, 1], [1, 0, 1], [1, 0, 1]], dtype=int),
    np.array([[1, 1, 1], [0, 0, 0], [1, 1, 1]], dtype=int),
    np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=int),
    np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=int),
]


def _random_texture(rng: np.random.Generator, k: int, seed_color: int) -> np.ndarray:
    bank = _PATTERNS_2 if k == 2 else _PATTERNS_3
    base = bank[int(rng.integers(len(bank)))]
    choices = [c for c in _VISIBLE_COLORS if c != seed_color]
    a, b = map(int, rng.choice(choices, size=2, replace=False))
    texture = np.where(base == 1, a, b).astype(int)
    texture = np.rot90(texture, int(rng.integers(4)))
    if bool(rng.integers(2)):
        texture = np.fliplr(texture)
    return texture


def _draw_frame(grid: np.ndarray, r0: int, c0: int, h: int, w: int, seed_color: int, rng: np.random.Generator) -> None:
    grid[r0, c0 : c0 + w] = _FRAME
    grid[r0 + h - 1, c0 : c0 + w] = _FRAME
    grid[r0 : r0 + h, c0] = _FRAME
    grid[r0 : r0 + h, c0 + w - 1] = _FRAME

    sr = int(rng.integers(r0 + 1, r0 + h - 1))
    sc = int(rng.integers(c0 + 1, c0 + w - 1))
    grid[sr, sc] = seed_color


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    k = int(rng.choice([2, 3]))
    legend_h = 2 * k
    sep_rows = int(rng.integers(1, 3))
    area_top = legend_h + sep_rows

    entry_count = int(rng.integers(2, 5))
    gap = int(rng.integers(1, 3))
    legend_w = entry_count * k + (entry_count - 1) * gap

    seed_colors = list(map(int, rng.choice(_VISIBLE_COLORS, size=entry_count, replace=False)))
    textures = {seed: _random_texture(rng, k, seed) for seed in seed_colors}

    region_count = int(rng.integers(2, 5))
    used_count = int(rng.integers(1, min(entry_count, region_count) + 1))
    used_colors = list(map(int, rng.choice(seed_colors, size=used_count, replace=False)))
    region_colors = used_colors.copy()
    while len(region_colors) < region_count:
        region_colors.append(int(rng.choice(used_colors)))
    rng.shuffle(region_colors)

    specs: list[tuple[int, int, int]] = []
    for color in region_colors:
        inner_h = int(rng.integers(2, 2 * k + 3))
        inner_w = int(rng.integers(2, 2 * k + 3))
        specs.append((inner_h, inner_w, color))
    specs.sort(key=lambda item: item[0] * item[1], reverse=True)

    max_h = max(inner_h + 2 for inner_h, _, _ in specs)
    max_w = max(inner_w + 2 for _, inner_w, _ in specs)
    padded_area = sum((inner_h + 4) * (inner_w + 4) for inner_h, inner_w, _ in specs)
    body_side = int(np.ceil(np.sqrt(padded_area))) + int(rng.integers(1, 4))
    grid_h = area_top + max(max_h + 2, body_side + int(rng.integers(0, 4)))
    grid_w = max(legend_w + int(rng.integers(0, 5)), max_w + 2, body_side + int(rng.integers(0, 4)))
    if grid_h > 30 or grid_w > 30:
        return None

    grid = np.full((grid_h, grid_w), _BG, dtype=int)
    left = int(rng.integers(0, grid_w - legend_w + 1))
    order = list(rng.permutation(entry_count))
    for idx, entry_idx in enumerate(order):
        c0 = left + idx * (k + gap)
        seed_color = seed_colors[entry_idx]
        texture = textures[seed_color]
        grid[:k, c0 : c0 + k] = seed_color
        grid[k : 2 * k, c0 : c0 + k] = texture

    occ = np.zeros_like(grid, dtype=bool)
    occ[:area_top, :] = True
    occ[0, :] = True
    occ[-1, :] = True
    occ[:, 0] = True
    occ[:, -1] = True

    for inner_h, inner_w, seed_color in specs:
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        placed = False
        for _ in range(200):
            r0 = int(rng.integers(area_top, grid_h - outer_h + 1))
            c0 = int(rng.integers(0, grid_w - outer_w + 1))
            pr0 = r0 - 1
            pc0 = c0 - 1
            pr1 = r0 + outer_h + 1
            pc1 = c0 + outer_w + 1
            if bool(occ[pr0:pr1, pc0:pc1].any()):
                continue
            occ[pr0:pr1, pc0:pc1] = True
            _draw_frame(grid, r0, c0, outer_h, outer_w, seed_color, rng)
            placed = True
            break
        if not placed:
            return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(300):
        grid = _gen_attempt(rng)
        if grid is None:
            continue
        try:
            out = main(grid)
        except AssertionError:
            continue
        if bool(np.array_equal(out, grid)):
            continue
        return grid
    raise RuntimeError("failed to generate a valid input")
