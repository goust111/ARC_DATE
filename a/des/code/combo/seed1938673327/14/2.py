from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
SEPARATOR = 5
BLUE = 1
GREEN = 3
ORANGE = 7

INPUT_COLORS = [2, 4, 6, 8, 9]
ASPECT_KINDS = ("square", "tall", "wide")

GRID_SIZE_RANGE = (13, 19)
OBJECT_COUNT_RANGE = (3, 5)
SEPARATOR_MARGIN = 4
PLACEMENT_PADDING = 1
GENERATION_ATTEMPTS = 400
OBJECT_ATTEMPTS = 80


def _find_separator(g: np.ndarray) -> tuple[str, int]:
    row_axes = [r for r in range(g.shape[0]) if np.all(g[r, :] == SEPARATOR)]
    col_axes = [c for c in range(g.shape[1]) if np.all(g[:, c] == SEPARATOR)]
    assert len(row_axes) + len(col_axes) == 1, "expected exactly one full grey separator"
    if row_axes:
        return "row", int(row_axes[0])
    return "col", int(col_axes[0])


def _components_of_color(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue: deque[tuple[int, int]] = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(comp)
    return components


def _foreground_components(g: np.ndarray) -> list[list[tuple[int, int]]]:
    colors = [
        int(color)
        for color in np.unique(g)
        if int(color) not in {BG, SEPARATOR}
    ]
    components: list[list[tuple[int, int]]] = []
    for color in colors:
        components.extend(_components_of_color(g == color))
    assert components, "expected at least one foreground object"
    return components


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _encoded_color(height: int, width: int) -> int:
    if height == width:
        return ORANGE
    if height > width:
        return GREEN
    return BLUE


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each foreground object across the grey separator, recoloring the copy by bbox aspect ratio.

    Objects keep their original colors. The mirrored copy uses orange for square
    bounding boxes, green for taller-than-wide boxes, and blue for wider-than-tall boxes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    axis_kind, axis = _find_separator(g)
    fg_mask = (g != BG) & (g != SEPARATOR)
    assert not np.any(fg_mask[axis, :]) if axis_kind == "row" else not np.any(fg_mask[:, axis])

    if axis_kind == "row":
        has_before = bool(np.any(fg_mask[:axis, :]))
        has_after = bool(np.any(fg_mask[axis + 1 :, :]))
    else:
        has_before = bool(np.any(fg_mask[:, :axis]))
        has_after = bool(np.any(fg_mask[:, axis + 1 :]))
    assert has_before != has_after, "foreground must lie on exactly one side of the separator"

    out = g.copy()
    h, w = g.shape
    for comp in _foreground_components(g):
        r0, c0, r1, c1 = _bbox(comp)
        if axis_kind == "row":
            assert r1 < axis or r0 > axis, "objects cannot cross the separator"
            fill = _encoded_color(r1 - r0 + 1, c1 - c0 + 1)
            mirrored = [(2 * axis - r, c) for r, c in comp]
        else:
            assert c1 < axis or c0 > axis, "objects cannot cross the separator"
            fill = _encoded_color(r1 - r0 + 1, c1 - c0 + 1)
            mirrored = [(r, 2 * axis - c) for r, c in comp]

        for rr, cc in mirrored:
            assert 0 <= rr < h and 0 <= cc < w, "mirrored object must stay in bounds"
            assert int(out[rr, cc]) == BG, "mirrored copy cannot overwrite existing content"
            out[rr, cc] = fill

    return out


def _corner_path(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    if h == 1 and w == 1:
        mask[0, 0] = True
        return mask

    if random.random() < 0.5:
        r, c = 0, 0
        dc = 1
    else:
        r, c = 0, w - 1
        dc = -1
    dr = 1

    mask[r, c] = True
    moves = [(dr, 0)] * (h - 1) + [(0, dc)] * (w - 1)
    random.shuffle(moves)
    for step_r, step_c in moves:
        r += step_r
        c += step_c
        mask[r, c] = True
    return mask


def _grow_mask(mask: np.ndarray, extra: int) -> np.ndarray:
    h, w = mask.shape
    for _ in range(extra):
        frontier = []
        for r in range(h):
            for c in range(w):
                if mask[r, c]:
                    continue
                if any(
                    0 <= r + dr < h
                    and 0 <= c + dc < w
                    and mask[r + dr, c + dc]
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                ):
                    frontier.append((r, c))
        if not frontier:
            break
        rr, cc = random.choice(frontier)
        mask[rr, cc] = True
    return mask


def _make_object_mask(height: int, width: int) -> np.ndarray:
    mask = _corner_path(height, width)
    free = height * width - int(mask.sum())
    if free:
        extra = random.randint(0, free)
        mask = _grow_mask(mask, extra)
    return mask


def _sample_dims(kind: str, axis_kind: str, normal_depth: int) -> tuple[int, int]:
    if kind == "square":
        side = random.randint(2, min(4, normal_depth))
        return side, side
    if kind == "tall":
        height = random.randint(3, min(4, normal_depth)) if axis_kind == "row" else random.randint(2, 4)
        width = random.randint(1, min(3, height - 1))
        return height, width
    width = random.randint(3, min(5, normal_depth)) if axis_kind == "col" else random.randint(3, 5)
    height = random.randint(1, min(3, width - 1))
    return height, width


def _valid_positions(
    occupied: np.ndarray,
    row_range: tuple[int, int],
    col_range: tuple[int, int],
    height: int,
    width: int,
) -> list[tuple[int, int]]:
    h, w = occupied.shape
    positions = []
    row_start, row_end = row_range
    col_start, col_end = col_range
    for r in range(row_start, row_end - height + 2):
        for c in range(col_start, col_end - width + 2):
            r0 = max(0, r - PLACEMENT_PADDING)
            r1 = min(h, r + height + PLACEMENT_PADDING)
            c0 = max(0, c - PLACEMENT_PADDING)
            c1 = min(w, c + width + PLACEMENT_PADDING)
            if occupied[r0:r1, c0:c1].any():
                continue
            positions.append((r, c))
    return positions


def _paint_object(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    height, width = mask.shape
    patch = np.full((height, width), BG, dtype=int)
    patch[mask] = color
    region = grid[top : top + height, left : left + width]
    region[mask] = patch[mask]


def _mark_occupied(occupied: np.ndarray, top: int, left: int, height: int, width: int) -> None:
    h, w = occupied.shape
    r0 = max(0, top - PLACEMENT_PADDING)
    r1 = min(h, top + height + PLACEMENT_PADDING)
    c0 = max(0, left - PLACEMENT_PADDING)
    c1 = min(w, left + width + PLACEMENT_PADDING)
    occupied[r0:r1, c0:c1] = True


def _build_scene() -> np.ndarray | None:
    height = random.randint(*GRID_SIZE_RANGE)
    width = random.randint(*GRID_SIZE_RANGE)
    grid = np.full((height, width), BG, dtype=int)

    axis_kind = random.choice(("row", "col"))
    if axis_kind == "row":
        axis = random.randint(SEPARATOR_MARGIN, height - SEPARATOR_MARGIN - 1)
        grid[axis, :] = SEPARATOR
        use_before = bool(random.getrandbits(1))
        normal_depth = min(axis, height - axis - 1)
        if normal_depth < 4:
            return None
        if use_before:
            row_range = (axis - normal_depth, axis - 1)
        else:
            row_range = (axis + 1, axis + normal_depth)
        col_range = (0, width - 1)
    else:
        axis = random.randint(SEPARATOR_MARGIN, width - SEPARATOR_MARGIN - 1)
        grid[:, axis] = SEPARATOR
        use_before = bool(random.getrandbits(1))
        normal_depth = min(axis, width - axis - 1)
        if normal_depth < 5:
            return None
        row_range = (0, height - 1)
        if use_before:
            col_range = (axis - normal_depth, axis - 1)
        else:
            col_range = (axis + 1, axis + normal_depth)

    count = random.randint(*OBJECT_COUNT_RANGE)
    kinds = ["square", "tall", "wide"] + [random.choice(ASPECT_KINDS) for _ in range(count - 3)]
    random.shuffle(kinds)
    colors = random.sample(INPUT_COLORS, count)
    occupied = np.zeros_like(grid, dtype=bool)
    occupied[grid == SEPARATOR] = True

    for kind, color in zip(kinds, colors, strict=False):
        placed = False
        for _ in range(OBJECT_ATTEMPTS):
            obj_h, obj_w = _sample_dims(kind, axis_kind, normal_depth)
            if obj_h > row_range[1] - row_range[0] + 1 or obj_w > col_range[1] - col_range[0] + 1:
                continue
            mask = _make_object_mask(obj_h, obj_w)
            positions = _valid_positions(occupied, row_range, col_range, obj_h, obj_w)
            if not positions:
                continue
            top, left = random.choice(positions)
            _paint_object(grid, mask, top, left, color)
            _mark_occupied(occupied, top, left, obj_h, obj_w)
            placed = True
            break
        if not placed:
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _build_scene()
        if grid is None:
            continue
        out = main(grid)
        if not (np.any(out == ORANGE) and np.any(out == GREEN) and np.any(out == BLUE)):
            continue
        return grid
    raise RuntimeError("failed to generate a valid sample")
