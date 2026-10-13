from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(cells)
    return comps


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    assert len(pts) > 0, "cannot crop an empty mask"
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _count_holes(mask: np.ndarray) -> int:
    shape = _crop_mask(mask)
    bg = ~shape
    h, w = bg.shape
    seen = np.zeros_like(bg, dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if not bg[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            touches_edge = False
            while queue:
                rr, cc = queue.popleft()
                if rr in (0, h - 1) or cc in (0, w - 1):
                    touches_edge = True
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bg[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            if not touches_edge:
                holes += 1
    return holes


def _is_connected(mask: np.ndarray) -> bool:
    return len(_components(mask)) == 1


def _random_transform(mask: np.ndarray) -> np.ndarray:
    out = np.rot90(mask, random.randint(0, 3))
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    return _crop_mask(out)


def _add_protrusions(canvas: np.ndarray, count: int) -> None:
    for _ in range(count):
        pts = np.argwhere(canvas)
        if len(pts) == 0:
            return
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        options: list[tuple[int, int, int, int]] = []
        if r0 > 0:
            for c in range(int(c0), int(c1) + 1):
                if canvas[int(r0), c]:
                    options.append((int(r0), c, -1, 0))
        if r1 < canvas.shape[0] - 1:
            for c in range(int(c0), int(c1) + 1):
                if canvas[int(r1), c]:
                    options.append((int(r1), c, 1, 0))
        if c0 > 0:
            for r in range(int(r0), int(r1) + 1):
                if canvas[r, int(c0)]:
                    options.append((r, int(c0), 0, -1))
        if c1 < canvas.shape[1] - 1:
            for r in range(int(r0), int(r1) + 1):
                if canvas[r, int(c1)]:
                    options.append((r, int(c1), 0, 1))

        random.shuffle(options)
        for r, c, dr, dc in options:
            run = 0
            rr = r + dr
            cc = c + dc
            while 0 <= rr < canvas.shape[0] and 0 <= cc < canvas.shape[1] and not canvas[rr, cc] and run < 2:
                run += 1
                rr += dr
                cc += dc
            if run == 0:
                continue
            length = random.randint(1, run)
            rr = r
            cc = c
            for _ in range(length):
                rr += dr
                cc += dc
                canvas[rr, cc] = True
            break


def _carve_boundary_notch(canvas: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    side = random.choice(["top", "bottom", "left", "right"])
    if side in {"top", "bottom"} and h >= 3 and w >= 3:
        depth = random.randint(1, min(2, h - 1))
        length = random.randint(1, w - 2)
        start = random.randint(1, w - length - 1)
        if side == "top":
            canvas[r0 : r0 + depth, c0 + start : c0 + start + length] = False
        else:
            canvas[r0 + h - depth : r0 + h, c0 + start : c0 + start + length] = False
    elif side in {"left", "right"} and h >= 3 and w >= 3:
        depth = random.randint(1, min(2, w - 1))
        length = random.randint(1, h - 2)
        start = random.randint(1, h - length - 1)
        if side == "left":
            canvas[r0 + start : r0 + start + length, c0 : c0 + depth] = False
        else:
            canvas[r0 + start : r0 + start + length, c0 + w - depth : c0 + w] = False


def _make_zero_hole_shape(max_h: int, max_w: int) -> np.ndarray | None:
    if max_h < 4 or max_w < 4:
        return None
    for _ in range(200):
        canvas = np.zeros((max_h, max_w), dtype=bool)
        h = random.randint(4, max_h)
        w = random.randint(4, max_w)
        r0 = random.randint(0, max_h - h)
        c0 = random.randint(0, max_w - w)
        canvas[r0 : r0 + h, c0 : c0 + w] = True
        for _ in range(random.randint(1, 2)):
            _carve_boundary_notch(canvas, r0, c0, h, w)
        _add_protrusions(canvas, random.randint(1, 3))
        shape = _crop_mask(canvas)
        if _is_connected(shape) and _count_holes(shape) == 0 and shape.sum() >= 7 and not np.all(shape):
            return _random_transform(shape)
    return None


def _make_holey_shape(num_holes: int, max_h: int, max_w: int) -> np.ndarray | None:
    if max_h < 5 or max_w < 5:
        return None
    for _ in range(200):
        canvas = np.zeros((max_h, max_w), dtype=bool)
        h = random.randint(5, max_h)
        w = random.randint(5, max_w)
        r0 = random.randint(0, max_h - h)
        c0 = random.randint(0, max_w - w)
        canvas[r0 : r0 + h, c0 : c0 + w] = True

        interior = [(r, c) for r in range(r0 + 1, r0 + h - 1) for c in range(c0 + 1, c0 + w - 1)]
        if num_holes == 1:
            hole_h = random.choice([1, 1, 2])
            hole_w = random.choice([1, 1, 2])
            if h - 2 < hole_h or w - 2 < hole_w:
                continue
            hr = random.randint(r0 + 1, r0 + h - hole_h - 1)
            hc = random.randint(c0 + 1, c0 + w - hole_w - 1)
            canvas[hr : hr + hole_h, hc : hc + hole_w] = False
        else:
            random.shuffle(interior)
            pair = None
            for i, a in enumerate(interior):
                for b in interior[i + 1 :]:
                    if abs(a[0] - b[0]) + abs(a[1] - b[1]) > 1:
                        pair = (a, b)
                        break
                if pair is not None:
                    break
            if pair is None:
                continue
            for rr, cc in pair:
                canvas[rr, cc] = False

        _add_protrusions(canvas, random.randint(1, 3))
        shape = _crop_mask(canvas)
        if _is_connected(shape) and _count_holes(shape) == num_holes and shape.sum() >= 10:
            return _random_transform(shape)
    return None


def _make_object_mask(num_holes: int, qh: int, qw: int) -> np.ndarray:
    min_h = 5 if num_holes > 0 else 4
    min_w = 5 if num_holes > 0 else 4
    max_h = random.randint(min_h, min(8, qh - 1))
    max_w = random.randint(min_w, min(8, qw - 1))
    for _ in range(200):
        if num_holes == 0:
            shape = _make_zero_hole_shape(max_h, max_w)
        else:
            shape = _make_holey_shape(num_holes, max_h, max_w)
        if shape is not None and shape.shape[0] <= qh and shape.shape[1] <= qw:
            return shape
    raise RuntimeError("failed to generate an object mask")


def _legend_local_mask(qh: int, qw: int) -> np.ndarray:
    legend = np.zeros((qh, qw), dtype=bool)
    legend[:3, 0] = True
    return legend


def _legend_guard(qh: int, qw: int) -> np.ndarray:
    guard = _legend_local_mask(qh, qw)
    for r, c in [(0, 0), (1, 0), (2, 0)]:
        for dr, dc in DIR4:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < qh and 0 <= cc < qw:
                guard[rr, cc] = True
    return guard


def _embed(mask: np.ndarray, qh: int, qw: int, r0: int, c0: int) -> np.ndarray:
    out = np.zeros((qh, qw), dtype=bool)
    h, w = mask.shape
    out[r0 : r0 + h, c0 : c0 + w] = mask
    return out


def _pick_placements(masks: dict[str, np.ndarray], qh: int, qw: int) -> dict[str, np.ndarray] | None:
    legend = _legend_local_mask(qh, qw)
    guard = _legend_guard(qh, qw)
    candidates: dict[str, list[np.ndarray]] = {}
    for quad, mask in masks.items():
        h, w = mask.shape
        placed_options: list[np.ndarray] = []
        for r0 in range(qh - h + 1):
            for c0 in range(qw - w + 1):
                placed = _embed(mask, qh, qw, r0, c0)
                if quad == "br" and np.any(placed & guard):
                    continue
                placed_options.append(placed)
        if not placed_options:
            return None
        candidates[quad] = placed_options

    for _ in range(1000):
        placed = {quad: random.choice(opts) for quad, opts in candidates.items()}
        if np.any(np.fliplr(placed["ul"]) & placed["ur"]):
            continue
        if np.any(np.flipud(placed["ur"]) & placed["br"]):
            continue
        if np.any(np.flipud(placed["ur"]) & legend):
            continue
        if np.any(np.fliplr(placed["br"]) & placed["ll"]):
            continue
        if np.any(np.flipud(placed["ll"]) & placed["ul"]):
            continue
        return placed
    return None


def _find_center_lines(grid: np.ndarray) -> tuple[int, int]:
    row_hits = np.where(np.all(grid == GREY, axis=1))[0]
    col_hits = np.where(np.all(grid == GREY, axis=0))[0]
    assert len(row_hits) == 1 and len(col_hits) == 1, "expected one full grey row and one full grey column"
    return int(row_hits[0]), int(col_hits[0])


def _legend_cells(center_r: int, center_c: int, shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    cells = [(center_r + 1 + i, center_c + 1) for i in range(3)]
    assert cells[-1][0] < h and cells[0][1] < w, "legend must fit inside the lower-right quadrant"
    return cells


def _quadrant_bounds(name: str, center_r: int, center_c: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    if name == "ul":
        return 0, center_r, 0, center_c
    if name == "ur":
        return 0, center_r, center_c + 1, w
    if name == "ll":
        return center_r + 1, h, 0, center_c
    if name == "br":
        return center_r + 1, h, center_c + 1, w
    raise ValueError(name)


def _extract_quadrant_object(grid: np.ndarray, center_r: int, center_c: int, name: str) -> tuple[list[tuple[int, int]], int]:
    r0, r1, c0, c1 = _quadrant_bounds(name, center_r, center_c, grid.shape)
    region = grid[r0:r1, c0:c1].copy()
    if name == "br":
        region[:3, 0] = BLACK
    fg = (region != BLACK) & (region != GREY)
    comps = _components(fg)
    assert len(comps) == 1, f"expected one object in quadrant {name}"
    cells = comps[0]
    colors = {int(region[r, c]) for r, c in cells}
    assert len(colors) == 1, f"quadrant {name} object must be monochrome"
    mask = np.zeros_like(region, dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    holes = _count_holes(mask)
    global_cells = [(r0 + r, c0 + c) for r, c in cells]
    return global_cells, holes


def _mirror_pixel(r: int, c: int, center_r: int, center_c: int, quadrant: str) -> tuple[int, int]:
    if quadrant in {"ul", "br"}:
        return r, 2 * center_c - c
    return 2 * center_r - r, c


def main(grid: np.ndarray) -> np.ndarray:
    """Mirror each quadrant object toward the next quadrant and recolor the projection by its hole count.

    Input structure
    - The background is black.
    - One full grey row and one full grey column intersect at the grid center and split the canvas into four quadrants.
    - Each quadrant contains exactly one monochrome 4-connected object; the lower-right quadrant also has a 3x1 legend just below-right of the center.
    - The legend's three colors encode the projection colors for objects with 0, 1, and 2 enclosed black holes.

    Rule
    - Count the enclosed black 4-connected holes inside each quadrant object.
    - Use that hole count to select a projection color from the legend.
    - Mirror the upper-left object into the upper-right, the upper-right into the lower-right, the lower-right into the lower-left, and the lower-left into the upper-left.
    - Draw mirrored pixels only onto black background cells, keep the original objects unchanged, and erase the legend to black.

    Key insight
    - The destination depends on the object's quadrant, while the projection color is controlled independently by the object's number of enclosed holes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    center_r, center_c = _find_center_lines(g)
    legend_cells = _legend_cells(center_r, center_c, g.shape)
    legend = [int(g[r, c]) for r, c in legend_cells]
    assert len(set(legend)) == 3 and all(color not in {BLACK, GREY} for color in legend)

    out = g.copy()
    for quadrant in ("ul", "ur", "br", "ll"):
        cells, holes = _extract_quadrant_object(g, center_r, center_c, quadrant)
        assert 0 <= holes <= 2, "hole count must match the 3-entry legend"
        color = legend[holes]
        for r, c in cells:
            rr, cc = _mirror_pixel(r, c, center_r, center_c, quadrant)
            if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1] and int(g[rr, cc]) == BLACK and int(out[rr, cc]) == BLACK:
                out[rr, cc] = color

    for r, c in legend_cells:
        out[r, c] = BLACK
    return out


def _build_grid(
    h: int,
    w: int,
    legend_colors: list[int],
    object_colors: dict[str, int],
    placements: dict[str, np.ndarray],
) -> np.ndarray:
    grid = np.full((h, w), BLACK, dtype=int)
    center_r = h // 2
    center_c = w // 2
    grid[center_r, :] = GREY
    grid[:, center_c] = GREY

    legend_cells = _legend_cells(center_r, center_c, grid.shape)
    for (r, c), color in zip(legend_cells, legend_colors):
        grid[r, c] = color

    for quadrant in ("ul", "ur", "ll", "br"):
        r0, r1, c0, c1 = _quadrant_bounds(quadrant, center_r, center_c, grid.shape)
        region = grid[r0:r1, c0:c1]
        region[placements[quadrant]] = object_colors[quadrant]
    return grid


def _validate_generated_input(grid: np.ndarray) -> bool:
    center_r, center_c = _find_center_lines(grid)
    qh = center_r
    qw = center_c
    assert qh >= 3 and qw >= 1
    legend = [int(grid[r, c]) for r, c in _legend_cells(center_r, center_c, grid.shape)]
    assert len(set(legend)) == 3
    for quadrant in ("ul", "ur", "ll", "br"):
        _, holes = _extract_quadrant_object(grid, center_r, center_c, quadrant)
        assert holes in {0, 1, 2}
    out = main(grid)
    assert out.shape == grid.shape
    for r, c in _legend_cells(center_r, center_c, grid.shape):
        assert int(out[r, c]) == BLACK
    assert np.any(out != grid)
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randrange(19, 30, 2)
    w = random.randrange(19, 30, 2)
    center_r = h // 2
    center_c = w // 2
    qh = center_r
    qw = center_c
    if qh < 9 or qw < 9:
        return None

    palette = random.sample(COLORS, 7)
    legend_colors = palette[:3]
    object_colors = dict(zip(("ul", "ur", "ll", "br"), palette[3:]))

    hole_counts = [0, 1, 2, random.randint(0, 2)]
    random.shuffle(hole_counts)
    holes_by_quad = dict(zip(("ul", "ur", "ll", "br"), hole_counts))

    try:
        masks = {quad: _make_object_mask(holes_by_quad[quad], qh, qw) for quad in ("ul", "ur", "ll", "br")}
    except RuntimeError:
        return None

    placements = _pick_placements(masks, qh, qw)
    if placements is None:
        return None

    grid = _build_grid(h, w, legend_colors, object_colors, placements)
    if _validate_generated_input(grid):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
