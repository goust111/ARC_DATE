from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


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


def _component_coords(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                nbrs = _neighbors4(rr, cc) if connectivity == 4 else _neighbors8(rr, cc)
                for nr, nc in nbrs:
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _border_reachable_zeros(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    exterior = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if grid[r, c] == BLACK and not exterior[r, c]:
                exterior[r, c] = True
                queue.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if grid[r, c] == BLACK and not exterior[r, c]:
                exterior[r, c] = True
                queue.append((r, c))
    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(grid.shape, nr, nc) and grid[nr, nc] == BLACK and not exterior[nr, nc]:
                exterior[nr, nc] = True
                queue.append((nr, nc))
    return exterior


def _outline_mask(shape_mask: np.ndarray, exterior: np.ndarray) -> np.ndarray:
    out = np.zeros(shape_mask.shape, dtype=bool)
    for r, c in np.argwhere(shape_mask):
        for nr, nc in _neighbors8(int(r), int(c)):
            if _in_bounds(shape_mask.shape, nr, nc) and exterior[nr, nc]:
                out[nr, nc] = True
    return out


def _find_palette_column(grid: np.ndarray) -> int:
    for c in range(grid.shape[1]):
        col = grid[:, c]
        if np.any((col != BLACK) & (col != GREY)):
            return c
    raise AssertionError("expected a non-grey palette column")


def _extract_palette(grid: np.ndarray, palette_col: int) -> list[int]:
    palette = [int(v) for v in grid[:, palette_col] if int(v) != BLACK]
    assert len(palette) >= 2, "palette must have at least two colors"
    assert len(set(palette)) == len(palette), "palette colors must be distinct"
    assert GREY not in palette, "palette cannot use the foreground grey"
    return palette


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    palette_col = _find_palette_column(g)
    palette = _extract_palette(g, palette_col)
    work = g.copy()
    work[:, palette_col] = BLACK

    shape_mask = work == GREY
    shapes = _component_coords(shape_mask, connectivity=4)
    assert 3 <= len(shapes) <= 4, "expected 3-4 separated grey shapes"

    exterior = _border_reachable_zeros(work)
    holes = (work == BLACK) & ~exterior
    out = work.copy()
    all_outline = np.zeros(g.shape, dtype=bool)

    for coords in shapes:
        r0, c0, r1, c1 = _bbox(coords)
        obj_mask = np.zeros(g.shape, dtype=bool)
        obj_mask[coords[:, 0], coords[:, 1]] = True
        obj_outline = _outline_mask(obj_mask, exterior)
        local_holes = holes[r0 : r1 + 1, c0 : c1 + 1]
        hole_components = _component_coords(local_holes, connectivity=4)
        hole_count = len(hole_components)
        outline_color = palette[hole_count % len(palette)]
        fill_color = palette[(hole_count + 1) % len(palette)]
        out[obj_outline] = outline_color
        all_outline |= obj_outline
        for hole_coords in hole_components:
            rr = hole_coords[:, 0] + r0
            cc = hole_coords[:, 1] + c0
            out[rr, cc] = fill_color

    assert not np.any(all_outline & holes), "outside outline cannot overlap holes"
    out[shape_mask] = GREY
    out[:, palette_col] = BLACK
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color each grey shape's outside outline and enclosed holes by indexing a left-side palette with that shape's hole count.

    Input structure:
    - The grid has a black background and one left-side palette column made of isolated non-black single pixels.
    - Away from that column, there are 3 or 4 mutually separated grey 4-connected shapes.
    - Each grey shape may contain black enclosed holes, and different shapes have different hole counts.

    Rule:
    - Read the palette column from top to bottom as an ordered list P.
    - Remove the palette column from the scene by turning that whole column black.
    - For each grey shape, treat border-reachable black cells as exterior background and the remaining black cells inside the shape as holes.
    - Let H be that shape's number of enclosed hole components; paint its exterior 8-neighbor outline with P[H mod len(P)] and paint its holes with P[(H + 1) mod len(P)].
    - Keep the grey shape pixels unchanged.

    Key insight:
    - The colors are not fixed: each object's topology, specifically its hole count, selects an adjacent pair of palette entries for outline and hole fill.
    """
    return _solve(grid)


def _count_holes(mask: np.ndarray) -> tuple[int, np.ndarray]:
    bg = ~mask
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    holes_mask = np.zeros((h, w), dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if not bg[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            touches_border = False
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                if rr in (0, h - 1) or cc in (0, w - 1):
                    touches_border = True
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and bg[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            if not touches_border:
                count += 1
                for rr, cc in coords:
                    holes_mask[rr, cc] = True
    return count, holes_mask


def _foreground_connected(mask: np.ndarray) -> bool:
    comps = _component_coords(mask, connectivity=4)
    return len(comps) == 1


def _shape_quality(mask: np.ndarray, hole_count: int) -> bool:
    if not mask.any():
        return False
    if not _foreground_connected(mask):
        return False
    area = int(mask.sum())
    if not 12 <= area <= 30:
        return False
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    bh = r1 - r0 + 1
    bw = c1 - c0 + 1
    if bh < 4 or bw < 4 or bh > 7 or bw > 7:
        return False
    actual_holes, _ = _count_holes(mask)
    if actual_holes != hole_count:
        return False
    if area == bh * bw:
        return False
    return True


def _sample_hole_cells(height: int, width: int, hole_count: int) -> list[tuple[int, int]] | None:
    if hole_count == 0:
        return []
    interior = [(r, c) for r in range(1, height - 1) for c in range(1, width - 1)]
    random.shuffle(interior)
    chosen: list[tuple[int, int]] = []
    for r, c in interior:
        if all(abs(r - rr) + abs(c - cc) > 1 for rr, cc in chosen):
            chosen.append((r, c))
            if len(chosen) == hole_count:
                return chosen
    return None


def _border_fg_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for r, c in np.argwhere(mask):
        if any(_in_bounds(mask.shape, nr, nc) and not mask[nr, nc] for nr, nc in _neighbors4(int(r), int(c))):
            cells.append((int(r), int(c)))
    return cells


def _adjacent_bg_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    cells = set()
    for r, c in np.argwhere(mask):
        for nr, nc in _neighbors4(int(r), int(c)):
            if _in_bounds(mask.shape, nr, nc) and not mask[nr, nc]:
                cells.add((int(nr), int(nc)))
    return list(cells)


def _generate_shape(hole_count: int) -> np.ndarray:
    for _ in range(250):
        canvas = np.zeros((8, 8), dtype=bool)
        min_side = 5 if hole_count else 4
        base_h = random.randint(min_side, 6)
        base_w = random.randint(min_side, 6)
        r0 = random.randint(1, canvas.shape[0] - base_h - 1)
        c0 = random.randint(1, canvas.shape[1] - base_w - 1)
        canvas[r0 : r0 + base_h, c0 : c0 + base_w] = True
        hole_cells = _sample_hole_cells(base_h, base_w, hole_count)
        if hole_cells is None:
            continue
        for hr, hc in hole_cells:
            canvas[r0 + hr, c0 + hc] = False
        if hole_count == 0:
            notch = random.choice(
                [
                    (r0, c0),
                    (r0, c0 + base_w - 1),
                    (r0 + base_h - 1, c0),
                    (r0 + base_h - 1, c0 + base_w - 1),
                ]
            )
            canvas[notch] = False
        if not _shape_quality(canvas, hole_count):
            continue

        target_edits = random.randint(1, 4)
        accepted = 0
        for _ in range(40):
            proposal = canvas.copy()
            remove_mode = random.random() < 0.55
            candidates = _border_fg_cells(proposal) if remove_mode else _adjacent_bg_cells(proposal)
            if not candidates:
                continue
            rr, cc = random.choice(candidates)
            proposal[rr, cc] = not remove_mode
            if _shape_quality(proposal, hole_count):
                canvas = proposal
                accepted += 1
                if accepted >= target_edits:
                    break

        coords = np.argwhere(canvas)
        r0, c0, r1, c1 = _bbox(coords)
        cropped = canvas[r0 : r1 + 1, c0 : c1 + 1].copy()
        if _shape_quality(cropped, hole_count):
            return cropped
    raise RuntimeError(f"failed to generate shape with {hole_count} holes")


def _dilate_chebyshev(mask: np.ndarray, radius: int) -> np.ndarray:
    out = mask.copy()
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr = int(r) + dr
                nc = int(c) + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr, nc] = True
    return out


def _place_shapes(grid: np.ndarray, palette_mask: np.ndarray, shapes: list[tuple[int, np.ndarray]]) -> bool:
    blocked = _dilate_chebyshev(palette_mask, radius=2)
    order = sorted(shapes, key=lambda item: int(item[1].sum()), reverse=True)
    for _hole_count, patch in order:
        ph, pw = patch.shape
        starts: list[tuple[int, int]] = []
        for r0 in range(2, grid.shape[0] - ph - 1):
            for c0 in range(4, grid.shape[1] - pw - 1):
                if np.any(blocked[r0 : r0 + ph, c0 : c0 + pw] & patch):
                    continue
                starts.append((r0, c0))
        random.shuffle(starts)
        if not starts:
            return False
        r0, c0 = starts[0]
        grid[r0 : r0 + ph, c0 : c0 + pw][patch] = GREY
        placed = np.zeros_like(grid, dtype=bool)
        placed[r0 : r0 + ph, c0 : c0 + pw] = patch
        blocked |= _dilate_chebyshev(placed, radius=2)
    return True


def _oracle(grid: np.ndarray) -> np.ndarray:
    return _solve(grid)


def _semantic_checks(grid: np.ndarray) -> bool:
    palette_col = _find_palette_column(grid)
    palette = _extract_palette(grid, palette_col)
    work = grid.copy()
    work[:, palette_col] = BLACK
    shapes = _component_coords(work == GREY, connectivity=4)
    if len(shapes) not in (3, 4):
        return False

    exterior = _border_reachable_zeros(work)
    holes = (work == BLACK) & ~exterior
    hole_counts: list[int] = []
    areas_by_h: dict[int, int] = {}
    left_by_h: dict[int, int] = {}
    outline_union = np.zeros(grid.shape, dtype=bool)

    for coords in shapes:
        r0, c0, r1, c1 = _bbox(coords)
        local_holes = holes[r0 : r1 + 1, c0 : c1 + 1]
        hole_count = len(_component_coords(local_holes, connectivity=4))
        hole_counts.append(hole_count)
        areas_by_h[hole_count] = int(coords.shape[0])
        left_by_h[hole_count] = c0
        obj_mask = np.zeros(grid.shape, dtype=bool)
        obj_mask[coords[:, 0], coords[:, 1]] = True
        outline = _outline_mask(obj_mask, exterior)
        if np.any(outline & outline_union):
            return False
        if np.any(outline & holes):
            return False
        outline_union |= outline

    n_shapes = len(shapes)
    if sorted(hole_counts) != list(range(n_shapes)):
        return False
    if len(palette) != n_shapes + 1:
        return False

    areas = [areas_by_h[h] for h in range(n_shapes)]
    if all(areas[i] < areas[i + 1] for i in range(n_shapes - 1)) or all(areas[i] > areas[i + 1] for i in range(n_shapes - 1)):
        return False

    lefts = [left_by_h[h] for h in range(n_shapes)]
    if lefts == sorted(lefts) or lefts == sorted(lefts, reverse=True):
        return False

    actual = main(grid)
    expected = _oracle(grid)
    if not np.array_equal(actual, expected):
        return False

    swapped = grid.copy()
    palette_rows = [r for r in range(grid.shape[0]) if grid[r, palette_col] != BLACK]
    r0, r1 = palette_rows[0], palette_rows[1]
    swapped[r0, palette_col], swapped[r1, palette_col] = swapped[r1, palette_col], swapped[r0, palette_col]
    if np.array_equal(main(swapped), actual):
        return False
    return True


def _attempt_instance() -> np.ndarray | None:
    n_shapes = random.randint(3, 4)
    hole_counts = list(range(n_shapes))
    shapes = [(hole_count, _generate_shape(hole_count)) for hole_count in hole_counts]

    grid_h = random.randint(18, 24)
    grid_w = random.randint(19, 27)
    grid = np.zeros((grid_h, grid_w), dtype=int)

    palette = random.sample(PALETTE_COLORS, n_shapes + 1)
    palette_col = random.choice([1, 2])
    palette_mask = np.zeros_like(grid, dtype=bool)
    gap = random.choice([1, 2])
    needed = len(palette) + (len(palette) - 1) * gap
    if needed + 2 > grid_h:
        return None
    top = random.randint(1, grid_h - needed - 1)
    for idx, color in enumerate(palette):
        r = top + idx * (gap + 1)
        grid[r, palette_col] = color
        palette_mask[r, palette_col] = True

    random.shuffle(shapes)
    if not _place_shapes(grid, palette_mask, shapes):
        return None
    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_instance()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid topology-palette outline/fill instance")
