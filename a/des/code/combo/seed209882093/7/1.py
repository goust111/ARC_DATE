from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
MARKER = 5
GEN_RETRIES = 400
PLACEMENT_RETRIES = 60


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _touches_edge(comp: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp)


def _nonbg_runs(row: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for c, value in enumerate(row):
        if int(value) != BG and start is None:
            start = c
        elif int(value) == BG and start is not None:
            runs.append((start, c - 1))
            start = None
    if start is not None:
        runs.append((start, len(row) - 1))
    return runs


def _find_palette(grid: np.ndarray) -> tuple[np.ndarray, list[int]]:
    h, w = grid.shape
    candidates: list[tuple[int, int, int, list[int]]] = []
    for r in (0, h - 1):
        for c0, c1 in _nonbg_runs(grid[r]):
            colors = [int(v) for v in grid[r, c0 : c1 + 1]]
            if len(colors) != len(set(colors)):
                continue
            candidates.append((len(colors), r, c0, colors))
    assert candidates, "expected a unique palette run on the top or bottom edge"
    _, row, c0, colors = max(candidates, key=lambda item: item[0])
    mask = np.zeros((h, w), dtype=bool)
    mask[row, c0 : c0 + len(colors)] = True
    return mask, colors


def _find_border_color(grid: np.ndarray, palette_mask: np.ndarray) -> int:
    palette_colors = {int(v) for v in grid[palette_mask]}
    candidates = [
        color
        for color in sorted({int(v) for v in np.unique(grid)})
        if color != BG and color not in palette_colors
    ]
    best_color = None
    best_score = (-1, -1)
    h, w = grid.shape
    for color in candidates:
        traversable = (grid != color) & (~palette_mask)
        score_regions = 0
        score_area = 0
        for comp in _components(traversable):
            if _touches_edge(comp, h, w):
                continue
            score_regions += 1
            score_area += len(comp)
        if (score_regions, score_area) > best_score:
            best_score = (score_regions, score_area)
            best_color = color
    assert best_color is not None and best_score[0] > 0, "failed to infer the border color"
    return int(best_color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill enclosed regions using a horizontal edge palette indexed by marker count.

    Input structure:
    - Black background.
    - Several closed regions drawn in one border color.
    - Grey marker pixels scattered inside each region.
    - A horizontal palette bar on the top or bottom edge, with distinct colors.

    Rule:
    - Count grey markers inside each enclosed region.
    - Use that count k as a 1-based index into the palette.
    - Fill the region's black and grey interior pixels with the k-th palette color.
    - Leave the border and palette unchanged.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    palette_mask, palette_colors = _find_palette(g)
    border_color = _find_border_color(g, palette_mask)

    traversable = (g != border_color) & (~palette_mask)
    h, w = g.shape
    for comp in _components(traversable):
        if _touches_edge(comp, h, w):
            continue
        marker_count = sum(1 for r, c in comp if int(g[r, c]) != BG)
        assert 1 <= marker_count <= len(palette_colors), "marker count must index the palette"
        fill_color = int(palette_colors[marker_count - 1])
        for r, c in comp:
            out[r, c] = fill_color

    return out


def _grow_connected_cells(rng: random.Random, h: int, w: int, area: int) -> set[tuple[int, int]]:
    seed = (rng.randrange(h), rng.randrange(w))
    cells = {seed}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        cells.add(rng.choice(frontier))
    return cells


def _transform_cells(
    cells: set[tuple[int, int]], h: int, w: int, rng: random.Random
) -> tuple[set[tuple[int, int]], int, int]:
    pts = list(cells)
    if rng.random() < 0.35:
        pts = [(c, r) for r, c in pts]
        h, w = w, h
    if rng.random() < 0.5:
        pts = [(r, w - 1 - c) for r, c in pts]
    if rng.random() < 0.5:
        pts = [(h - 1 - r, c) for r, c in pts]
    return set(pts), h, w


def _make_cavity(rng: random.Random, marker_count: int) -> tuple[set[tuple[int, int]], int, int]:
    if rng.random() < 0.4:
        inner_h = rng.randint(2, 5)
        inner_w = rng.randint(2, 6)
        while inner_h * inner_w <= marker_count:
            if inner_h <= inner_w:
                inner_h += 1
            else:
                inner_w += 1
        cells = {(r, c) for r in range(inner_h) for c in range(inner_w)}
        return _transform_cells(cells, inner_h, inner_w, rng)

    for _ in range(80):
        inner_h = rng.randint(2, 6)
        inner_w = rng.randint(2, 6)
        min_area = max(marker_count + 1, max(4, (inner_h * inner_w + 1) // 2))
        max_area = min(inner_h * inner_w, 14)
        if min_area > max_area:
            continue
        area = rng.randint(min_area, max_area)
        cells = _grow_connected_cells(rng, inner_h, inner_w, area)
        rows = {r for r, _ in cells}
        cols = {c for _, c in cells}
        if len(rows) < 2 or len(cols) < 2:
            continue
        return _transform_cells(cells, inner_h, inner_w, rng)

    inner_h = marker_count + 1
    inner_w = 2
    cells = {(r, c) for r in range(inner_h) for c in range(inner_w)}
    return _transform_cells(cells, inner_h, inner_w, rng)


def _build_region_patch(
    cells: set[tuple[int, int]],
    inner_h: int,
    inner_w: int,
    border_color: int,
    marker_count: int,
    rng: random.Random,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    patch_h = inner_h + 2
    patch_w = inner_w + 2
    patch = np.full((patch_h, patch_w), BG, dtype=int)

    cavity = {(r + 1, c + 1) for r, c in cells}
    border: set[tuple[int, int]] = set()
    for r, c in cavity:
        for nr, nc in _neighbors8(r, c):
            if 0 <= nr < patch_h and 0 <= nc < patch_w and (nr, nc) not in cavity:
                border.add((nr, nc))
    for r, c in border:
        patch[r, c] = border_color

    cavity_list = sorted(cavity)
    marker_cells = set(rng.sample(cavity_list, marker_count))
    for r, c in marker_cells:
        patch[r, c] = MARKER
    return patch, cavity_list


def _place_patch(
    grid: np.ndarray,
    blocked: np.ndarray,
    patch: np.ndarray,
    rng: random.Random,
) -> tuple[int, int] | None:
    h, w = grid.shape
    ph, pw = patch.shape
    candidates = []
    for r0 in range(h - ph + 1):
        for c0 in range(w - pw + 1):
            if blocked[r0 : r0 + ph, c0 : c0 + pw].any():
                continue
            candidates.append((r0, c0))
    if not candidates:
        return None
    r0, c0 = rng.choice(candidates)
    mask = patch != BG
    grid[r0 : r0 + ph, c0 : c0 + pw][mask] = patch[mask]
    blocked[max(0, r0 - 1) : min(h, r0 + ph + 1), max(0, c0 - 1) : min(w, c0 + pw + 1)] = True
    return r0, c0


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    max_palette_len = 6
    h = rng.randint(14, 24)
    w = rng.randint(16, 28)
    n_regions = rng.randint(2, 5)
    max_count = rng.randint(2, max_palette_len)
    counts = [rng.randint(1, max_count) for _ in range(n_regions - 1)] + [max_count]
    rng.shuffle(counts)

    border_color = rng.choice([c for c in range(1, 10) if c != MARKER])
    palette_pool = [c for c in range(1, 10) if c not in (MARKER, border_color)]
    if max_count > len(palette_pool):
        return None
    palette_colors = rng.sample(palette_pool, max_count)

    if w < max_count + 2:
        return None

    grid = np.full((h, w), BG, dtype=int)
    out = grid.copy()
    blocked = np.zeros((h, w), dtype=bool)
    blocked[0, :] = True
    blocked[-1, :] = True
    blocked[:, 0] = True
    blocked[:, -1] = True

    palette_side = rng.choice(["top", "bottom"])
    start_c = rng.randint(1, w - max_count - 1)
    row = 0 if palette_side == "top" else h - 1
    grid[row, start_c : start_c + max_count] = palette_colors
    out[row, start_c : start_c + max_count] = palette_colors
    if palette_side == "top":
        blocked[1, :] = True
    else:
        blocked[h - 2, :] = True

    regions: list[tuple[list[tuple[int, int]], int]] = []
    for marker_count in counts:
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            cells, inner_h, inner_w = _make_cavity(rng, marker_count)
            patch, cavity = _build_region_patch(cells, inner_h, inner_w, border_color, marker_count, rng)
            pos = _place_patch(grid, blocked, patch, rng)
            if pos is None:
                continue
            r0, c0 = pos
            abs_cavity = [(r0 + r, c0 + c) for r, c in cavity]
            regions.append((abs_cavity, marker_count))
            placed = True
            break
        if not placed:
            return None

    out[:, :] = grid
    for abs_cavity, marker_count in regions:
        fill_color = palette_colors[marker_count - 1]
        for r, c in abs_cavity:
            out[r, c] = fill_color

    if np.array_equal(main(grid), out):
        return grid
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
