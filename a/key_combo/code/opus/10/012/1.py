from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
YELLOW = 4
GREY = 5
BODY_COLORS = [1, 2, 3, 6, 7, 8, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(coords)
    return components


def _find_separator_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == GREY)]
    if len(rows) != 1:
        raise ValueError("expected exactly one full-width grey separator row")
    return rows[0]


def _transform_patch(patch: np.ndarray, marker_count: int) -> np.ndarray:
    if marker_count == 0:
        return np.flipud(patch)
    if marker_count == 1:
        return np.fliplr(patch)
    if marker_count == 2:
        return np.rot90(patch, 2)
    raise ValueError("marker count must be 0, 1, or 2")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transform each marked sprite by a marker-count-selected symmetry and mirror it below the separator.

    Input structure:
    - The grid has black background and one full-width grey horizontal separator row.
    - The upper half contains several separated 4-connected sprites made of one body color plus zero, one, or two yellow marker pixels.
    - The lower half starts empty, and each sprite fits inside a bounding box no larger than 4x4.

    Rule:
    - Find each upper-half sprite as a connected component of non-black, non-grey pixels.
    - Count its yellow pixels: 0 means vertical flip, 1 means horizontal flip, and 2 means 180-degree rotation.
    - Before transforming, treat yellow marker cells as body-color cells; then place the transformed sprite in the vertically mirrored position across the grey separator, keeping the same columns and preserving the original upper-half sprites.

    Key insight:
    - The yellow pixels are not output decorations; their count is a control code that selects which geometric transform to apply before mirrored placement.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    sep = _find_separator_row(g)
    upper_mask = (g[:sep] != BG) & (g[:sep] != GREY)
    components = _connected_components(upper_mask)
    components.sort(key=lambda coords: _bbox(coords))

    out = g.copy()
    for coords in components:
        r0, c0, r1, c1 = _bbox(coords)
        colors = {int(g[r, c]) for r, c in coords if int(g[r, c]) != YELLOW}
        if len(colors) != 1:
            raise ValueError("each sprite must have exactly one non-yellow body color")
        body_color = colors.pop()
        marker_count = sum(int(g[r, c] == YELLOW) for r, c in coords)
        if marker_count > 2:
            raise ValueError("marker count must be at most 2")

        patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
        for r, c in coords:
            patch[r - r0, c - c0] = body_color
        transformed = _transform_patch(patch, marker_count)

        out_r0 = 2 * sep - r1
        out_r1 = out_r0 + transformed.shape[0]
        if not (sep < out_r0 and out_r1 <= g.shape[0]):
            raise ValueError("mirrored sprite falls out of bounds")
        view = out[out_r0:out_r1, c0 : c0 + transformed.shape[1]]
        view[transformed != BG] = transformed[transformed != BG]

    return out


def _normalize_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _cells_to_patch(cells: set[tuple[int, int]], color: int = 1) -> np.ndarray:
    max_r = max(r for r, _ in cells)
    max_c = max(c for _, c in cells)
    patch = np.zeros((max_r + 1, max_c + 1), dtype=int)
    for r, c in cells:
        patch[r, c] = color
    return patch


def _random_polyomino(area: int) -> set[tuple[int, int]] | None:
    cells: set[tuple[int, int]] = {(0, 0)}
    while len(cells) < area:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in DIRS4:
                cand = (r + dr, c + dc)
                if cand in cells or cand in frontier:
                    continue
                trial = _normalize_cells(cells | {cand})
                h = 1 + max(r0 for r0, _ in trial)
                w = 1 + max(c0 for _, c0 in trial)
                if h <= 4 and w <= 4:
                    frontier.append(cand)
        if not frontier:
            return None
        cells.add(random.choice(frontier))
    return _normalize_cells(cells)


def _random_shape(marker_count: int) -> np.ndarray:
    for _ in range(200):
        area = random.randint(3, 6)
        cells = _random_polyomino(area)
        if cells is None:
            continue
        patch = _cells_to_patch(cells, color=1)
        if min(patch.shape) < 2:
            continue
        transformed = _transform_patch(patch, marker_count)
        if np.array_equal(patch, transformed):
            continue
        return patch
    raise RuntimeError("failed to sample a non-degenerate sprite shape")


def _random_composition(total: int, parts: int) -> list[int]:
    return np.random.multinomial(total, [1.0 / parts] * parts).tolist()


def _pack_intervals(widths: list[int], total_width: int, *, min_gap: int = 2, edge_gap: int = 1) -> list[tuple[int, int]]:
    required = sum(widths) + min_gap * (len(widths) - 1) + 2 * edge_gap
    extra = total_width - required
    if extra < 0:
        raise ValueError("width too small for requested packing")

    slack = _random_composition(extra, len(widths) + 1)
    intervals: list[tuple[int, int]] = []
    pos = edge_gap + slack[0]
    for i, width in enumerate(widths):
        intervals.append((pos, pos + width - 1))
        pos += width
        if i + 1 < len(widths):
            pos += min_gap + slack[i + 1]
    return intervals


def _paint_sprite(grid: np.ndarray, patch: np.ndarray, top: int, left: int, body_color: int, marker_count: int) -> None:
    coords = list(zip(*np.where(patch != BG)))
    marker_coords = set(random.sample(coords, marker_count))
    for r, c in coords:
        grid[top + r, left + c] = YELLOW if (r, c) in marker_coords else body_color


def generate_input() -> np.ndarray:
    for _ in range(400):
        sprite_count = random.randint(3, 5)
        marker_counts = [0, 1, 2] + [random.randint(0, 2) for _ in range(sprite_count - 3)]
        random.shuffle(marker_counts)

        shapes = [_random_shape(marker_count) for marker_count in marker_counts]
        widths = [int(shape.shape[1]) for shape in shapes]

        min_width = sum(widths) + 2 * (sprite_count - 1) + 2
        if min_width > 30:
            continue
        width_low = max(14, min_width)
        width_high = min(30, width_low + 6)
        width = random.randint(width_low, width_high)

        height = random.choice([15, 17, 19])
        sep = height // 2
        grid = np.zeros((height, width), dtype=int)
        grid[sep, :] = GREY

        intervals = _pack_intervals(widths, width, min_gap=2, edge_gap=1)
        colors = random.sample(BODY_COLORS, sprite_count)

        ok = True
        used_upper = np.zeros((sep, width), dtype=bool)
        for patch, body_color, marker_count, (c0, c1) in zip(shapes, colors, marker_counts, intervals):
            h, w = patch.shape
            max_top = sep - 1 - h
            if max_top < 1:
                ok = False
                break
            top = random.randint(1, max_top)
            bottom = top + h - 1
            mirror_top = 2 * sep - bottom
            mirror_bottom = mirror_top + h - 1
            if mirror_bottom >= height:
                ok = False
                break

            bbox_rows = slice(max(0, top - 1), min(sep, bottom + 2))
            bbox_cols = slice(max(0, c0 - 1), min(width, c1 + 2))
            if used_upper[bbox_rows, bbox_cols].any():
                ok = False
                break
            used_upper[top : bottom + 1, c0 : c1 + 1] |= patch != BG

            _paint_sprite(grid, patch, top, c0, body_color, marker_count)

        if not ok:
            continue

        try:
            out = main(grid)
        except Exception:
            continue

        if not np.all(grid[sep + 1 :, :] == BG):
            continue
        if np.array_equal(out, grid):
            continue
        return grid

    raise RuntimeError("failed to generate a valid input")
