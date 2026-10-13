from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]
GEN_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(pts)
    return out


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _find_legend(grid: np.ndarray) -> tuple[tuple[int, int, int, int], set[int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[tuple[int, tuple[int, int, int, int], set[int]]] = []
    for color0 in np.unique(g):
        color = int(color0)
        if color == BG:
            continue
        for pts in _components(g == color):
            if (0, 0) not in pts:
                continue
            r0, c0, r1, c1 = _bbox(pts)
            if r0 != 0 or c0 != 0:
                continue
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            sub = g[r0 : r1 + 1, c0 : c1 + 1]
            border = np.zeros(sub.shape, dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            if not np.array_equal(sub == color, border):
                continue
            inside_colors = {int(v) for v in sub[1:-1, 1:-1].ravel() if int(v) != BG}
            if not inside_colors:
                continue
            area = (r1 - r0 + 1) * (c1 - c0 + 1)
            candidates.append((area, (r0, c0, r1, c1), inside_colors))
    assert len(candidates) == 1, f"expected one top-left legend, found {len(candidates)}"
    _, box, target_colors = candidates[0]
    return box, target_colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Apply the legend's target-color set as a deletion filter on multi-color objects.

    The top-left legend is a single-color rectangular frame. Every non-background color
    inside that frame belongs to the target set. Outside the legend, each 4-connected
    foreground object is examined as a whole. Objects whose color set contains every
    target color are erased; all others remain unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    (r0, c0, r1, c1), target_colors = _find_legend(g)
    work_mask = g != BG
    work_mask[r0 : r1 + 1, c0 : c1 + 1] = False

    out = g.copy()
    for pts in _components(work_mask):
        colors = {int(g[r, c]) for r, c in pts}
        if target_colors <= colors:
            for r, c in pts:
                out[r, c] = BG
    return out


def _sample_connected_shape(rng: random.Random, h: int, w: int, area: int) -> set[tuple[int, int]]:
    start = (rng.randrange(h), rng.randrange(w))
    cells = {start}
    frontier = set()
    for dr, dc in NEIGHBORS4:
        nr, nc = start[0] + dr, start[1] + dc
        if 0 <= nr < h and 0 <= nc < w:
            frontier.add((nr, nc))
    while len(cells) < area and frontier:
        cell = rng.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        cr, cc = cell
        for dr, dc in NEIGHBORS4:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                frontier.add((nr, nc))
    return cells


def _color_object(
    rng: random.Random,
    cells: list[tuple[int, int]],
    colors: list[int],
) -> dict[tuple[int, int], int]:
    shuffled = cells[:]
    rng.shuffle(shuffled)
    k = len(colors)
    assignment: dict[tuple[int, int], int] = {}
    for idx, cell in enumerate(shuffled):
        if idx < k:
            assignment[cell] = colors[idx]
        else:
            assignment[cell] = rng.choice(colors)
    return assignment


def _legend_patch(
    rng: random.Random, frame_color: int, target_colors: list[int]
) -> np.ndarray:
    inner_h = rng.randint(2, 4)
    inner_w = rng.randint(max(2, len(target_colors)), max(4, len(target_colors) + 1))
    patch = np.full((inner_h + 2, inner_w + 2), BG, dtype=int)
    patch[0, :] = frame_color
    patch[-1, :] = frame_color
    patch[:, 0] = frame_color
    patch[:, -1] = frame_color
    interior = [(r, c) for r in range(1, inner_h + 1) for c in range(1, inner_w + 1)]
    rng.shuffle(interior)
    for color, (r, c) in zip(target_colors, interior):
        patch[r, c] = color
    extra_slots = interior[len(target_colors) :]
    for r, c in extra_slots:
        if rng.random() < 0.35:
            patch[r, c] = rng.choice(target_colors)
    return patch


def _can_place(
    occ: np.ndarray,
    top: int,
    left: int,
    shape_h: int,
    shape_w: int,
    margin: int = 1,
) -> bool:
    h, w = occ.shape
    if top < 0 or left < 0 or top + shape_h > h or left + shape_w > w:
        return False
    r0 = max(0, top - margin)
    c0 = max(0, left - margin)
    r1 = min(h, top + shape_h + margin)
    c1 = min(w, left + shape_w + margin)
    return not occ[r0:r1, c0:c1].any()


def _paint_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    grid[top : top + h, left : left + w] = patch


def _attempt_generate(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(14, 23)
    w = rng.randint(14, 23)
    grid = np.zeros((h, w), dtype=int)
    occ = np.zeros((h, w), dtype=bool)

    target_size = rng.randint(2, 4)
    palette = rng.sample(COLOR_POOL, rng.randint(max(5, target_size + 2), 8))
    target_colors = rng.sample(palette, target_size)
    frame_color_choices = [c for c in palette if c not in target_colors]
    frame_color = rng.choice(frame_color_choices)

    legend = _legend_patch(rng, frame_color, target_colors)
    lh, lw = legend.shape
    if lh >= h or lw >= w:
        return None
    _paint_patch(grid, legend, 0, 0)
    occ[:lh, :lw] = legend != BG

    object_count = rng.randint(4, 7)
    full_match_count = rng.randint(1, max(1, object_count - 2))
    full_match_indices = set(rng.sample(range(object_count), full_match_count))

    for idx in range(object_count):
        shp_h = rng.randint(2, 5)
        shp_w = rng.randint(2, 5)
        min_area = max(4, shp_h + shp_w - 1)
        max_area = shp_h * shp_w
        area = rng.randint(min_area, max_area)
        cells = list(_sample_connected_shape(rng, shp_h, shp_w, area))
        if len(cells) < area:
            return None

        if idx in full_match_indices:
            extra_pool = [c for c in palette if c not in target_colors]
            extra_count = rng.randint(0, min(2, len(extra_pool)))
            colors = target_colors + rng.sample(extra_pool, extra_count)
        else:
            missing_count = rng.randint(1, min(len(target_colors), 2))
            present_targets = rng.sample(target_colors, len(target_colors) - missing_count)
            extras_needed = rng.randint(1, max(1, 4 - len(present_targets)))
            extra_pool = [c for c in palette if c not in target_colors]
            if len(extra_pool) < extras_needed:
                return None
            colors = present_targets + rng.sample(extra_pool, extras_needed)
        colors = list(dict.fromkeys(colors))
        if len(colors) < 2 or len(cells) < len(colors):
            return None

        assignment = _color_object(rng, cells, colors)
        tries = 0
        placed = False
        while tries < 120 and not placed:
            top = rng.randint(0, h - shp_h)
            left = rng.randint(0, w - shp_w)
            tries += 1
            if top == 0 and left == 0:
                continue
            if not _can_place(occ, top, left, shp_h, shp_w, margin=1):
                continue
            for (dr, dc), color in assignment.items():
                grid[top + dr, left + dc] = color
                occ[top + dr, left + dc] = True
            placed = True
        if not placed:
            return None

    out = main(grid)
    changed = np.any(out != grid)
    unchanged = np.any((grid != BG) & (out == grid) & ~np.pad(np.ones((lh, lw), dtype=bool), ((0, h - lh), (0, w - lw)), constant_values=False))
    if not changed or not unchanged:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        rng = random.Random()
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
