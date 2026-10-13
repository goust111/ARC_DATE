from __future__ import annotations

from collections import deque
import random

import numpy as np


BACKGROUND = 0
SEED_COLOR = 8
BORDER_COLORS = (1, 2, 3, 4, 5, 6, 7, 9)

GRID_MIN = 20
GRID_MAX = 28
MAX_GENERATE_ATTEMPTS = 400
REGION_PLACEMENT_ATTEMPTS = 120
REGION_PATCH_ATTEMPTS = 120
PATTERN_ATTEMPTS = 120

SHAPE_KINDS = ("rect", "notch_br", "notch_tr", "notch_bl")


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


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


def _touches_edge(cells: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in cells)


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _draw_segment(grid: np.ndarray, p0: tuple[int, int], p1: tuple[int, int], color: int) -> None:
    r0, c0 = p0
    r1, c1 = p1
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        grid[r0, lo : hi + 1] = color
        return
    if c0 == c1:
        lo, hi = sorted((r0, r1))
        grid[lo : hi + 1, c0] = color
        return
    raise AssertionError("polygon edges must be axis-aligned")


def _draw_polygon(grid: np.ndarray, vertices: list[tuple[int, int]], color: int) -> None:
    for a, b in zip(vertices, vertices[1:] + vertices[:1], strict=False):
        _draw_segment(grid, a, b, color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find each enclosed region, extract the azure seed pattern in its top-left corner,
    tile that pattern across the region, and recolor the tiled seed cells to the
    enclosing border color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    passable = (g == BACKGROUND) | (g == SEED_COLOR)
    out = g.copy()
    enclosed_count = 0

    for comp in _components(passable):
        if _touches_edge(comp, h, w):
            continue

        seed_cells = [(r, c) for r, c in comp if int(g[r, c]) == SEED_COLOR]
        assert seed_cells, "each enclosed region must contain a seed"

        boundary_colors = set()
        for r, c in comp:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w:
                    value = int(g[nr, nc])
                    if value not in (BACKGROUND, SEED_COLOR):
                        boundary_colors.add(value)
        assert len(boundary_colors) == 1, "each region must have exactly one border color"
        fill_color = int(next(iter(boundary_colors)))

        sr0, sc0, sr1, sc1 = _bbox(seed_cells)
        pattern = g[sr0 : sr1 + 1, sc0 : sc1 + 1] == SEED_COLOR
        assert np.any(pattern), "seed bbox must contain at least one seed pixel"

        ph, pw = pattern.shape
        for r, c in comp:
            if pattern[(r - sr0) % ph, (c - sc0) % pw]:
                out[r, c] = fill_color
            else:
                out[r, c] = BACKGROUND

        enclosed_count += 1

    assert enclosed_count > 0, "expected at least one enclosed region"
    return out


def _random_pattern(size: int) -> np.ndarray:
    min_on = 2 if size == 2 else 3
    max_on = size * size
    for _ in range(PATTERN_ATTEMPTS):
        target_on = random.randint(min_on, max_on)
        cells = [(r, c) for r in range(size) for c in range(size)]
        chosen = set(random.sample(cells, target_on))
        pat = np.zeros((size, size), dtype=bool)
        for r, c in chosen:
            pat[r, c] = True
        if not np.any(pat[0, :]) or not np.any(pat[-1, :]):
            continue
        if not np.any(pat[:, 0]) or not np.any(pat[:, -1]):
            continue
        return pat
    raise RuntimeError("failed to sample a seed pattern")


def _rect_vertices(height: int, width: int) -> list[tuple[int, int]]:
    return [(0, 0), (0, width - 1), (height - 1, width - 1), (height - 1, 0)]


def _notch_br_vertices(height: int, width: int, seed_size: int) -> list[tuple[int, int]]:
    cut_r = random.randint(seed_size + 1, height - 3)
    cut_c = random.randint(seed_size + 2, width - 3)
    return [(0, 0), (0, width - 1), (cut_r, width - 1), (cut_r, cut_c), (height - 1, cut_c), (height - 1, 0)]


def _notch_tr_vertices(height: int, width: int, seed_size: int) -> list[tuple[int, int]]:
    cut_r = random.randint(2, height - 3)
    cut_c = random.randint(seed_size + 1, width - 3)
    return [(0, 0), (0, cut_c), (cut_r, cut_c), (cut_r, width - 1), (height - 1, width - 1), (height - 1, 0)]


def _notch_bl_vertices(height: int, width: int, seed_size: int) -> list[tuple[int, int]]:
    cut_r = random.randint(seed_size + 1, height - 3)
    cut_c = random.randint(2, width - 3)
    return [(0, 0), (0, width - 1), (height - 1, width - 1), (height - 1, cut_c), (cut_r, cut_c), (cut_r, 0)]


def _make_vertices(shape_kind: str, height: int, width: int, seed_size: int) -> list[tuple[int, int]]:
    if shape_kind == "rect":
        return _rect_vertices(height, width)
    if shape_kind == "notch_br":
        return _notch_br_vertices(height, width, seed_size)
    if shape_kind == "notch_tr":
        return _notch_tr_vertices(height, width, seed_size)
    if shape_kind == "notch_bl":
        return _notch_bl_vertices(height, width, seed_size)
    raise AssertionError(f"unknown shape {shape_kind}")


def _interior_mask(boundary_patch: np.ndarray) -> np.ndarray:
    bg_mask = boundary_patch == BACKGROUND
    comps = _components(bg_mask)
    interior = np.zeros_like(bg_mask, dtype=bool)
    for comp in comps:
        if not _touches_edge(comp, *bg_mask.shape):
            for r, c in comp:
                interior[r, c] = True
    return interior


def _make_region_patch(border_color: int, shape_kind: str, seed_size: int) -> np.ndarray | None:
    min_height = seed_size + 4
    min_width = seed_size + 4
    if shape_kind == "notch_br":
        min_width = seed_size + 5
    elif shape_kind == "notch_tr":
        min_width = seed_size + 4
    elif shape_kind == "notch_bl":
        min_height = seed_size + 4
    for _ in range(REGION_PATCH_ATTEMPTS):
        height = random.randint(min_height, 12)
        width = random.randint(min_width, 12)
        patch = np.zeros((height, width), dtype=int)
        vertices = _make_vertices(shape_kind, height, width, seed_size)
        _draw_polygon(patch, vertices, border_color)

        interior = _interior_mask(patch)
        comps = _components(interior)
        if len(comps) != 1:
            continue
        cells = comps[0]
        ir0, ic0, ir1, ic1 = _bbox(cells)
        if (ir1 - ir0 + 1) < seed_size + 1 or (ic1 - ic0 + 1) < seed_size + 1:
            continue
        anchor = (ir0, ic0)
        fits = True
        for dr in range(seed_size):
            for dc in range(seed_size):
                rr = anchor[0] + dr
                cc = anchor[1] + dc
                if rr >= height or cc >= width or not interior[rr, cc]:
                    fits = False
                    break
            if not fits:
                break
        if not fits:
            continue

        pattern = _random_pattern(seed_size)
        for dr in range(seed_size):
            for dc in range(seed_size):
                if pattern[dr, dc]:
                    patch[anchor[0] + dr, anchor[1] + dc] = SEED_COLOR

        # Keep only one enclosed passable component after inserting the seed.
        passable = (patch == BACKGROUND) | (patch == SEED_COLOR)
        enclosed = [comp for comp in _components(passable) if not _touches_edge(comp, height, width)]
        if len(enclosed) != 1:
            continue
        seed_cells = [(r, c) for r, c in enclosed[0] if patch[r, c] == SEED_COLOR]
        if not seed_cells:
            continue

        sr0, sc0, sr1, sc1 = _bbox(seed_cells)
        if (sr1 - sr0 + 1, sc1 - sc0 + 1) != (seed_size, seed_size):
            continue
        return patch
    return None


def _can_place(reserved: np.ndarray, top: int, left: int, patch: np.ndarray) -> bool:
    h, w = reserved.shape
    ph, pw = patch.shape
    rr0 = max(0, top - 1)
    cc0 = max(0, left - 1)
    rr1 = min(h, top + ph + 1)
    cc1 = min(w, left + pw + 1)
    return not np.any(reserved[rr0:rr1, cc0:cc1])


def _place_patch(grid: np.ndarray, reserved: np.ndarray, top: int, left: int, patch: np.ndarray) -> None:
    ph, pw = patch.shape
    region = grid[top : top + ph, left : left + pw]
    mask = patch != BACKGROUND
    assert not np.any(region[mask] != BACKGROUND)
    region[mask] = patch[mask]

    rr0 = max(0, top - 1)
    cc0 = max(0, left - 1)
    rr1 = min(grid.shape[0], top + ph + 1)
    cc1 = min(grid.shape[1], left + pw + 1)
    reserved[rr0:rr1, cc0:cc1] = True


def _target_region_count(height: int, width: int) -> int:
    limit = min(height, width)
    if limit >= 25:
        return random.randint(2, 4)
    if limit >= 22:
        return random.randint(1, 3)
    return random.randint(1, 2)


def _structural_region_count(grid: np.ndarray) -> int:
    passable = (grid == BACKGROUND) | (grid == SEED_COLOR)
    return sum(1 for comp in _components(passable) if not _touches_edge(comp, *grid.shape))


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(GRID_MIN, GRID_MAX)
    width = random.randint(GRID_MIN, GRID_MAX)
    n_regions = _target_region_count(height, width)
    border_colors = random.sample(BORDER_COLORS, n_regions)

    grid = np.full((height, width), BACKGROUND, dtype=int)
    reserved = np.zeros((height, width), dtype=bool)

    if n_regions >= 2:
        seed_sizes = [2, 3]
        while len(seed_sizes) < n_regions:
            seed_sizes.append(random.choice((2, 3)))
        random.shuffle(seed_sizes)
    else:
        seed_sizes = [random.choice((2, 3))]

    shape_choices = list(SHAPE_KINDS)
    random.shuffle(shape_choices)
    while len(shape_choices) < n_regions:
        shape_choices.append(random.choice(SHAPE_KINDS))
    random.shuffle(shape_choices)

    for idx in range(n_regions):
        border_color = border_colors[idx]
        seed_size = seed_sizes[idx]
        shape_kind = shape_choices[idx]

        patch = _make_region_patch(border_color, shape_kind, seed_size)
        if patch is None:
            return None

        ph, pw = patch.shape
        if ph + 2 > height or pw + 2 > width:
            return None

        top_max = height - ph - 1
        left_max = width - pw - 1
        if top_max < 1 or left_max < 1:
            return None

        for _ in range(REGION_PLACEMENT_ATTEMPTS):
            top = random.randint(1, top_max)
            left = random.randint(1, left_max)
            if _can_place(reserved, top, left, patch):
                _place_patch(grid, reserved, top, left, patch)
                break
        else:
            return None

    if _structural_region_count(grid) != n_regions:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    if np.any(out == SEED_COLOR):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
