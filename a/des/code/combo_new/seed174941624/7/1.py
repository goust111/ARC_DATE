import random
from collections import deque

import numpy as np


BACKGROUND = 0
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
SIDES = ("top", "bottom", "left", "right")

# Generator tuning.
GEN_RETRIES = 300
OBJECT_RETRIES = 60
PACK_RETRIES = 80
PLACE_RETRIES = 200
OBJECT_COUNT_CHOICES = (1, 2, 3, 4)
INNER_MIN = 4
RESERVE_GAP = 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Return 4-connected components from a boolean mask."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        seen[rr, cc] = True
        queue = deque([(rr, cc)])
        pts: list[tuple[int, int]] = []
        while queue:
            cr, cc = queue.popleft()
            pts.append((cr, cc))
            for dr, dc in DIR4:
                nr = cr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _perimeter_points(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    pts: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        pts.add((r0, c))
        pts.add((r1, c))
    for r in range(r0, r1 + 1):
        pts.add((r, c0))
        pts.add((r, c1))
    return pts


def _touching_sides(shape_mask: np.ndarray) -> list[str]:
    sides: list[str] = []
    if shape_mask[0, :].any():
        sides.append("top")
    if shape_mask[-1, :].any():
        sides.append("bottom")
    if shape_mask[:, 0].any():
        sides.append("left")
    if shape_mask[:, -1].any():
        sides.append("right")
    return sides


def _project_cell(
    r: int,
    c: int,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    side: str,
) -> tuple[int, int]:
    if side == "top":
        return 2 * r0 - r, c
    if side == "bottom":
        return 2 * r1 - r, c
    if side == "left":
        return r, 2 * c0 - c
    if side == "right":
        return r, 2 * c1 - c
    raise ValueError(f"unknown side: {side}")


def _try_parse_fence(
    grid: np.ndarray, points: list[tuple[int, int]], fence_color: int
) -> dict[str, object] | None:
    r0, c0, r1, c1 = _bbox(points)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    border = _perimeter_points(r0, c0, r1, c1)
    if len(points) != len(border) or set(points) != border:
        return None

    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    if inner.size == 0 or not np.any(inner == BACKGROUND):
        return None

    shape_colors = set(map(int, np.unique(inner))) - {BACKGROUND}
    if len(shape_colors) != 1:
        return None

    shape_color = int(next(iter(shape_colors)))
    shape_mask = inner == shape_color
    comps = _components(shape_mask)
    if len(comps) != 1:
        return None

    sides = _touching_sides(shape_mask)
    if len(sides) != 1:
        return None

    shape_abs = [(r0 + 1 + r, c0 + 1 + c) for r, c in comps[0]]
    return {
        "fence_color": fence_color,
        "shape_color": shape_color,
        "bbox": (r0, c0, r1, c1),
        "side": sides[0],
        "shape_abs": shape_abs,
        "covered": border | set(shape_abs),
    }


def _find_fences(grid: np.ndarray) -> list[dict[str, object]]:
    infos: list[dict[str, object]] = []
    covered: set[tuple[int, int]] = set()
    for color in sorted(set(map(int, grid.flat)) - {BACKGROUND}):
        for points in _components(grid == color):
            info = _try_parse_fence(grid, points, int(color))
            if info is None:
                continue
            infos.append(info)
            covered |= info["covered"]

    assert infos, "expected at least one fenced object"
    non_bg = {(int(r), int(c)) for r, c in np.argwhere(grid != BACKGROUND)}
    assert covered == non_bg, "expected all non-background pixels to belong to valid fences"
    return infos


def main(grid: np.ndarray) -> np.ndarray:
    """Mirror each interior colored region outside its fence using the fence color.

    Input structure:
    - The background is black.
    - The grid contains one or more one-pixel-thick rectangular fences in non-black colors.
    - Inside each fence is one connected colored region of a different non-black color, and that region touches exactly one fence side.

    Rule:
    - For each fence, detect which side of the interior region touches the fence.
    - Reflect the region across that touched fence side and place the reflected copy immediately outside the fence.
    - Recolor the reflected copy with the fence color while keeping the original fence and the original interior region unchanged.

    Key insight:
    - The contacted fence side determines the mirror direction, while the fence color rather than the interior color determines the projected copy's color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert 0 < g.shape[0] <= 30 and 0 < g.shape[1] <= 30, "ARC grids must stay within 30x30"

    out = g.copy()
    for info in _find_fences(g):
        r0, c0, r1, c1 = info["bbox"]
        side = info["side"]
        fence_color = int(info["fence_color"])
        for r, c in info["shape_abs"]:
            rr, cc = _project_cell(int(r), int(c), r0, c0, r1, c1, side)
            assert 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1], "projection left the canvas"
            out[rr, cc] = fence_color
    return out


def _allowed_mask(inner_h: int, inner_w: int, side: str) -> np.ndarray:
    mask = np.ones((inner_h, inner_w), dtype=bool)
    if side != "top":
        mask[0, :] = False
    if side != "bottom":
        mask[-1, :] = False
    if side != "left":
        mask[:, 0] = False
    if side != "right":
        mask[:, -1] = False
    return mask


def _contact_cells(inner_h: int, inner_w: int, side: str) -> list[tuple[int, int]]:
    if side == "top":
        return [(0, c) for c in range(1, inner_w - 1)]
    if side == "bottom":
        return [(inner_h - 1, c) for c in range(1, inner_w - 1)]
    if side == "left":
        return [(r, 0) for r in range(1, inner_h - 1)]
    if side == "right":
        return [(r, inner_w - 1) for r in range(1, inner_h - 1)]
    raise ValueError(f"unknown side: {side}")


def _depth_from_side(cell: tuple[int, int], inner_h: int, inner_w: int, side: str) -> int:
    r, c = cell
    if side == "top":
        return r
    if side == "bottom":
        return inner_h - 1 - r
    if side == "left":
        return c
    if side == "right":
        return inner_w - 1 - c
    raise ValueError(f"unknown side: {side}")


def _shape_neighbors(
    shape: set[tuple[int, int]], allowed: np.ndarray
) -> list[tuple[int, int]]:
    h, w = allowed.shape
    frontier: set[tuple[int, int]] = set()
    for r, c in shape:
        for dr, dc in DIR4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and allowed[nr, nc] and (nr, nc) not in shape:
                frontier.add((nr, nc))
    return list(frontier)


def _adjacent_count(shape: set[tuple[int, int]], cell: tuple[int, int]) -> int:
    r, c = cell
    return sum((r + dr, c + dc) in shape for dr, dc in DIR4)


def _generate_shape(inner_h: int, inner_w: int, side: str) -> list[tuple[int, int]] | None:
    allowed = _allowed_mask(inner_h, inner_w, side)
    contacts = _contact_cells(inner_h, inner_w, side)
    if allowed.sum() < 3 or not contacts:
        return None

    anchor_len = random.randint(1, min(3, len(contacts)))
    start = random.randint(0, len(contacts) - anchor_len)
    shape: set[tuple[int, int]] = set(contacts[start : start + anchor_len])
    target_low = max(3, len(shape))
    target_high = min(int(allowed.sum()), max(target_low, int(round(allowed.sum() * random.uniform(0.35, 0.7)))))
    target = random.randint(target_low, target_high)
    mode = random.choice(("blob", "snake"))

    while len(shape) < target:
        frontier = _shape_neighbors(shape, allowed)
        if not frontier:
            break
        if mode == "snake":
            tail = random.choice(tuple(shape))
            tail_frontier = []
            for dr, dc in DIR4:
                nr = tail[0] + dr
                nc = tail[1] + dc
                if 0 <= nr < inner_h and 0 <= nc < inner_w and allowed[nr, nc] and (nr, nc) not in shape:
                    tail_frontier.append((nr, nc))
            choice_pool = tail_frontier or frontier
            cell = random.choice(choice_pool)
        else:
            frontier.sort(
                key=lambda cell: (
                    _adjacent_count(shape, cell),
                    _depth_from_side(cell, inner_h, inner_w, side),
                    random.random(),
                ),
                reverse=True,
            )
            keep = max(1, len(frontier) // 2)
            cell = random.choice(frontier[:keep])
        shape.add(cell)

    deepest = max(_depth_from_side(cell, inner_h, inner_w, side) for cell in shape)
    if deepest == 0:
        deeper = [
            cell
            for cell in _shape_neighbors(shape, allowed)
            if _depth_from_side(cell, inner_h, inner_w, side) > 0
        ]
        if deeper:
            shape.add(random.choice(deeper))

    mask = np.zeros((inner_h, inner_w), dtype=bool)
    for r, c in shape:
        mask[r, c] = True
    if len(_components(mask)) != 1 or _touching_sides(mask) != [side]:
        return None
    return sorted(shape)


def _build_object(
    side: str, fence_color: int, shape_color: int, object_count: int
) -> dict[str, object] | None:
    max_dim = 8 if object_count == 1 else 7 if object_count == 2 else 6
    for _ in range(OBJECT_RETRIES):
        inner_h = random.randint(INNER_MIN, max_dim)
        inner_w = random.randint(INNER_MIN, max_dim)
        shape = _generate_shape(inner_h, inner_w, side)
        if shape is None:
            continue

        patch_h = inner_h + 2
        patch_w = inner_w + 2
        input_cells: list[tuple[int, int, int]] = []
        for r in range(patch_h):
            for c in range(patch_w):
                if r in (0, patch_h - 1) or c in (0, patch_w - 1):
                    input_cells.append((r, c, fence_color))
        shape_abs = [(r + 1, c + 1) for r, c in shape]
        for r, c in shape_abs:
            input_cells.append((r, c, shape_color))

        proj_cells = [
            _project_cell(r, c, 0, 0, patch_h - 1, patch_w - 1, side) for r, c in shape_abs
        ]
        rows = [0, patch_h - 1] + [r for r, _ in proj_cells]
        cols = [0, patch_w - 1] + [c for _, c in proj_cells]
        rmin = min(rows)
        rmax = max(rows)
        cmin = min(cols)
        cmax = max(cols)

        return {
            "input_cells": input_cells,
            "reserve_bbox": (rmin, cmin, rmax, cmax),
            "reserve_area": (rmax - rmin + 1) * (cmax - cmin + 1),
        }
    return None


def _validate_generated_input(grid: np.ndarray) -> bool:
    try:
        out = main(grid)
    except AssertionError:
        return False
    return (
        isinstance(out, np.ndarray)
        and out.shape == grid.shape
        and max(grid.shape) <= 30
        and max(out.shape) <= 30
        and np.any(out != grid)
    )


def _gen_attempt() -> np.ndarray | None:
    object_count = random.choice(OBJECT_COUNT_CHOICES)
    sides = random.sample(SIDES, k=object_count)
    palette = random.sample(list(range(1, 10)), 2 * object_count)
    fence_colors = palette[:object_count]
    shape_colors = palette[object_count:]

    objects: list[dict[str, object]] = []
    for side, fence_color, shape_color in zip(sides, fence_colors, shape_colors):
        obj = _build_object(side, fence_color, shape_color, object_count)
        if obj is None:
            return None
        objects.append(obj)

    objects.sort(key=lambda obj: int(obj["reserve_area"]), reverse=True)
    reserve_sizes = [
        (
            int(obj["reserve_bbox"][2]) - int(obj["reserve_bbox"][0]) + 1,
            int(obj["reserve_bbox"][3]) - int(obj["reserve_bbox"][1]) + 1,
        )
        for obj in objects
    ]
    total_padded_area = sum((h + 2 * RESERVE_GAP) * (w + 2 * RESERVE_GAP) for h, w in reserve_sizes)
    min_dim = max(
        max(h for h, _ in reserve_sizes) + 2,
        max(w for _, w in reserve_sizes) + 2,
        12,
        int(np.ceil(np.sqrt(total_padded_area))) + 1,
    )
    if min_dim > 30:
        return None

    for _ in range(PACK_RETRIES):
        height = random.randint(min_dim, 30)
        width = random.randint(min_dim, 30)
        if height * width < total_padded_area + 8:
            continue

        grid = np.zeros((height, width), dtype=int)
        blocked = np.zeros((height, width), dtype=bool)
        success = True

        for obj in objects:
            rmin, cmin, rmax, cmax = obj["reserve_bbox"]
            lo_r = -int(rmin)
            hi_r = height - 1 - int(rmax)
            lo_c = -int(cmin)
            hi_c = width - 1 - int(cmax)
            if lo_r > hi_r or lo_c > hi_c:
                success = False
                break

            placed = False
            for _ in range(PLACE_RETRIES):
                base_r = random.randint(lo_r, hi_r)
                base_c = random.randint(lo_c, hi_c)
                pad_r0 = max(0, base_r + int(rmin) - RESERVE_GAP)
                pad_c0 = max(0, base_c + int(cmin) - RESERVE_GAP)
                pad_r1 = min(height - 1, base_r + int(rmax) + RESERVE_GAP)
                pad_c1 = min(width - 1, base_c + int(cmax) + RESERVE_GAP)
                if blocked[pad_r0 : pad_r1 + 1, pad_c0 : pad_c1 + 1].any():
                    continue

                for r, c, color in obj["input_cells"]:
                    assert grid[base_r + r, base_c + c] == BACKGROUND
                    grid[base_r + r, base_c + c] = color
                blocked[pad_r0 : pad_r1 + 1, pad_c0 : pad_c1 + 1] = True
                placed = True
                break

            if not placed:
                success = False
                break

        if success and _validate_generated_input(grid):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fenced-mirror ARC input")
