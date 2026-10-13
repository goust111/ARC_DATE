from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_SIDE_ORDER = [("up", -1, 0), ("down", 1, 0), ("left", 0, -1), ("right", 0, 1)]

_MAX_GEN_ATTEMPTS = 400
_COLOR_POOL = tuple(range(10))
_MIN_CANVAS = 16
_MAX_CANVAS = 30
_MIN_OBJECTS = 1
_MAX_OBJECTS = 3
_MIN_SHAPE_SIZE = 7
_MAX_SHAPE_SIZE = 11


def _components(mask: np.ndarray) -> list[tuple[list[tuple[int, int]], bool]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[tuple[list[tuple[int, int]], bool]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        q: deque[tuple[int, int]] = deque([(r, c)])
        seen[r, c] = True
        cells: list[tuple[int, int]] = []
        touches_border = False
        while q:
            cr, cc = q.popleft()
            cells.append((cr, cc))
            if cr == 0 or cc == 0 or cr == h - 1 or cc == w - 1:
                touches_border = True
            for dr, dc in _DIRS4:
                nr = cr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        parts.append((cells, touches_border))
    return parts


def _outer_border_color(grid: np.ndarray) -> int:
    h, w = grid.shape
    if h == 1 or w == 1:
        vals, counts = np.unique(grid, return_counts=True)
        return int(vals[counts.argmax()])
    border = np.concatenate([grid[0, :], grid[-1, :], grid[1:-1, 0], grid[1:-1, -1]])
    vals, counts = np.unique(border, return_counts=True)
    return int(vals[counts.argmax()])


def _outline_color(grid: np.ndarray, bg: int) -> int:
    fg = grid[grid != bg]
    assert fg.size > 0, "expected outline and seed pixels"
    vals, counts = np.unique(fg, return_counts=True)
    return int(vals[counts.argmax()])


def _adjacent_sides(boundary: np.ndarray, r: int, c: int) -> list[str]:
    h, w = boundary.shape
    sides: list[str] = []
    for name, dr, dc in _SIDE_ORDER:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w and boundary[nr, nc]:
            sides.append(name)
    return sides


def _paint_parallel_strip(
    out: np.ndarray,
    boundary: np.ndarray,
    region: set[tuple[int, int]],
    seed: tuple[int, int],
    seed_color: int,
    side: str,
) -> None:
    if side in {"left", "right"}:
        walk_dirs = [(-1, 0), (1, 0)]
        wall_offset = (0, -1) if side == "left" else (0, 1)
    else:
        walk_dirs = [(0, -1), (0, 1)]
        wall_offset = (-1, 0) if side == "up" else (1, 0)

    sr, sc = seed
    out[sr, sc] = seed_color
    for dr, dc in walk_dirs:
        cr, cc = sr, sc
        while True:
            nr = cr + dr
            nc = cc + dc
            if (nr, nc) not in region:
                break
            wr = nr + wall_offset[0]
            wc = nc + wall_offset[1]
            if not (0 <= wr < boundary.shape[0] and 0 <= wc < boundary.shape[1] and boundary[wr, wc]):
                break
            out[nr, nc] = seed_color
            cr, cc = nr, nc


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _outer_border_color(g)
    outline = _outline_color(g, bg)
    boundary = g == outline

    out = g.copy()
    for cells, touches_border in _components(~boundary):
        if touches_border:
            continue

        seeds = [(r, c) for r, c in cells if int(g[r, c]) != bg]
        assert len(seeds) == 1, "each enclosed region must contain exactly one seed"
        seed = seeds[0]
        seed_color = int(g[seed])
        region = set(cells)
        sides = _adjacent_sides(boundary, *seed)

        if not sides:
            for r, c in cells:
                if int(g[r, c]) == bg:
                    out[r, c] = seed_color
            continue

        for side in dict.fromkeys(sides):
            _paint_parallel_strip(out, boundary, region, seed, seed_color, side)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed region either with a wall-parallel stripe or with a full flood fill, depending on how its seed touches the boundary.

    Input structure:
    - The grid has a background color plus one shared outline color forming one or
      more simple closed orthogonal boundaries.
    - Each enclosed interior contains exactly one seed pixel of a unique non-background,
      non-outline color, and the rest of the interior is background.
    - Boundaries do not overlap, and the outside background remains connected to the
      grid border.

    Rule:
    - For each enclosed region, inspect the seed pixel's 4-neighbor contact with the
      outline.
    - If the seed is not 4-adjacent to any outline pixel, flood-fill all background
      cells in that region's interior with the seed color.
    - If the seed is 4-adjacent to one wall, draw a 1-cell-wide segment through the
      seed parallel to that wall, extending in both directions until the wall turns.
      If the seed touches two walls at an interior corner, draw one such segment for
      each touched wall. Outline pixels stay unchanged.

    Key insight:
    - The same seed-inside-boundary setup splits into two distinct behaviors based on
      whether the seed is touching the outline: touching seeds trace the wall's inner
      side, while detached seeds trigger a full interior fill.
    """
    return _transform(grid)


def _outline_from_area(area: np.ndarray) -> np.ndarray:
    h, w = area.shape
    boundary = np.zeros_like(area, dtype=bool)
    for r, c in np.argwhere(area):
        r = int(r)
        c = int(c)
        for dr, dc in _DIRS4:
            nr = r + dr
            nc = c + dc
            if not (0 <= nr < h and 0 <= nc < w and area[nr, nc]):
                boundary[r, c] = True
                break
    return boundary


def _shape_is_valid(area: np.ndarray) -> bool:
    if not area.any():
        return False
    if len(_components(area)) != 1:
        return False
    for _cells, touches_border in _components(~area):
        if not touches_border:
            return False
    boundary = _outline_from_area(area)
    if len(_components(boundary)) != 1:
        return False
    return True


def _cell_sides(boundary: np.ndarray) -> dict[tuple[int, int], list[str]]:
    inside = np.argwhere(~boundary)
    sides: dict[tuple[int, int], list[str]] = {}
    for r, c in inside:
        labels = _adjacent_sides(boundary, int(r), int(c))
        if labels:
            sides[(int(r), int(c))] = labels
    return sides


def _classify_candidates(area: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    boundary = _outline_from_area(area)
    interior = area & ~boundary
    side_map = {
        (r, c): _adjacent_sides(boundary, r, c)
        for r, c in [tuple(map(int, rc)) for rc in np.argwhere(interior)]
    }
    edge_cells = [cell for cell, sides in side_map.items() if sides]
    deep_cells = [cell for cell, sides in side_map.items() if not sides]
    corner_cells = [cell for cell, sides in side_map.items() if len(sides) >= 2]
    single_wall_cells = [cell for cell, sides in side_map.items() if len(sides) == 1]
    return boundary, deep_cells, single_wall_cells, corner_cells


def _random_cut(area: np.ndarray) -> bool:
    h, w = area.shape
    side = random.choice(["top", "bottom", "left", "right"])
    if side in {"top", "bottom"}:
        max_depth = h - 4
        max_span = w - 3
        if max_depth < 1 or max_span < 1:
            return False
        depth = random.randint(1, max_depth)
        span = random.randint(1, max_span)
        start = random.randint(0, w - span)
        rows = slice(0, depth) if side == "top" else slice(h - depth, h)
        cols = slice(start, start + span)
    else:
        max_depth = w - 4
        max_span = h - 3
        if max_depth < 1 or max_span < 1:
            return False
        depth = random.randint(1, max_depth)
        span = random.randint(1, max_span)
        start = random.randint(0, h - span)
        rows = slice(start, start + span)
        cols = slice(0, depth) if side == "left" else slice(w - depth, w)
    area[rows, cols] = False
    return True


def _make_area() -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]] | None:
    for _ in range(80):
        h = random.randint(_MIN_SHAPE_SIZE, _MAX_SHAPE_SIZE)
        w = random.randint(_MIN_SHAPE_SIZE, _MAX_SHAPE_SIZE)
        area = np.ones((h, w), dtype=bool)
        cut_count = random.choices([0, 1, 2], weights=[2, 4, 3])[0]
        for _ in range(cut_count):
            trial = area.copy()
            if not _random_cut(trial):
                continue
            if trial.any():
                area = trial
        if not _shape_is_valid(area):
            continue
        boundary, deep_cells, single_wall_cells, corner_cells = _classify_candidates(area)
        if not deep_cells:
            continue
        if not single_wall_cells and not corner_cells:
            continue
        return area, boundary, deep_cells, single_wall_cells, corner_cells
    return None


def _render_object(
    bg: int,
    outline: int,
    seed_color: int,
    branch: str,
) -> np.ndarray | None:
    shape = _make_area()
    if shape is None:
        return None
    area, boundary, deep_cells, single_wall_cells, corner_cells = shape

    if branch == "internal":
        seed = random.choice(deep_cells)
    else:
        prefer_corner = bool(random.getrandbits(1))
        if prefer_corner and corner_cells:
            seed = random.choice(corner_cells)
        elif single_wall_cells:
            seed = random.choice(single_wall_cells)
        elif corner_cells:
            seed = random.choice(corner_cells)
        else:
            return None

    patch = np.full(area.shape, bg, dtype=int)
    patch[boundary] = outline
    patch[seed] = seed_color
    return patch


def _branch_plan(count: int) -> list[str]:
    if count == 1:
        return ["edge" if random.random() < 0.5 else "internal"]
    if count == 2:
        branches = ["edge", "internal"]
        random.shuffle(branches)
        return branches
    branches = ["edge", "internal", "internal"] if random.random() < 0.5 else ["edge", "edge", "internal"]
    random.shuffle(branches)
    return branches


def _place_patches(patches: list[np.ndarray], bg: int) -> np.ndarray | None:
    heights = [patch.shape[0] for patch in patches]
    widths = [patch.shape[1] for patch in patches]
    min_h = max(max(heights) + 3, _MIN_CANVAS)
    min_w = max(max(widths) + 3, _MIN_CANVAS)
    if min_h > _MAX_CANVAS or min_w > _MAX_CANVAS:
        return None

    for _ in range(60):
        h = random.randint(min_h, _MAX_CANVAS)
        w = random.randint(min_w, _MAX_CANVAS)
        grid = np.full((h, w), bg, dtype=int)
        occupied = np.zeros((h, w), dtype=bool)
        order = list(range(len(patches)))
        random.shuffle(order)
        positions: list[tuple[int, int] | None] = [None] * len(patches)
        ok = True

        for idx in order:
            patch = patches[idx]
            ph, pw = patch.shape
            candidates: list[tuple[int, int]] = []
            for r0 in range(1, h - ph):
                for c0 in range(1, w - pw):
                    rr0 = r0 - 1
                    rr1 = r0 + ph + 1
                    cc0 = c0 - 1
                    cc1 = c0 + pw + 1
                    if occupied[rr0:rr1, cc0:cc1].any():
                        continue
                    candidates.append((r0, c0))
            if not candidates:
                ok = False
                break
            positions[idx] = random.choice(candidates)
            r0, c0 = positions[idx]
            occupied[r0 : r0 + ph, c0 : c0 + pw] = True

        if not ok:
            continue

        for patch, pos in zip(patches, positions, strict=False):
            assert pos is not None
            r0, c0 = pos
            ph, pw = patch.shape
            mask = patch != bg
            region = grid[r0 : r0 + ph, c0 : c0 + pw]
            region[mask] = patch[mask]

        return grid
    return None


def _seed_colors_for_objects(count: int, bg: int, outline: int) -> list[int]:
    colors = [color for color in _COLOR_POOL if color not in {bg, outline}]
    random.shuffle(colors)
    return colors[:count]


def _input_respects_assumptions(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    bg = _outer_border_color(grid)
    outline = _outline_color(grid, bg)
    if np.count_nonzero(grid == outline) <= np.count_nonzero(grid != bg) // 4:
        return False
    boundary = grid == outline
    enclosed = [(cells, touch) for cells, touch in _components(~boundary) if not touch]
    if not enclosed:
        return False
    seen_seed_colors: set[int] = set()
    for cells, _touch in enclosed:
        seeds = [(r, c) for r, c in cells if int(grid[r, c]) != bg]
        if len(seeds) != 1:
            return False
        seed_color = int(grid[seeds[0]])
        if seed_color in seen_seed_colors or seed_color == outline:
            return False
        seen_seed_colors.add(seed_color)
    return True


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GEN_ATTEMPTS):
        bg, outline = random.sample(_COLOR_POOL, 2)
        obj_count = random.randint(_MIN_OBJECTS, _MAX_OBJECTS)
        branches = _branch_plan(obj_count)
        seed_colors = _seed_colors_for_objects(obj_count, bg, outline)

        patches: list[np.ndarray] = []
        for branch, seed_color in zip(branches, seed_colors, strict=False):
            patch = _render_object(bg, outline, seed_color, branch)
            if patch is None:
                break
            patches.append(patch)
        if len(patches) != obj_count:
            continue

        inp = _place_patches(patches, bg)
        if inp is None or not _input_respects_assumptions(inp):
            continue

        try:
            _ = main(inp)
        except Exception:
            continue
        return inp
    raise RuntimeError("failed to generate a valid input")
