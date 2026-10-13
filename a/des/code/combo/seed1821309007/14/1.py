from __future__ import annotations

import random

import numpy as np


_BG = 0
_DIRS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 8-connected components as coordinate arrays."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in _DIRS8:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _centered_top_left(r0: int, c0: int, old_shape: tuple[int, int], new_shape: tuple[int, int]) -> tuple[int, int]:
    h, w = old_shape
    nh, nw = new_shape
    nr0 = (2 * int(r0) + int(h) - int(nh)) // 2
    nc0 = (2 * int(c0) + int(w) - int(nw)) // 2
    return int(nr0), int(nc0)


def _analyze_object(grid: np.ndarray, coords: np.ndarray) -> tuple[np.ndarray, str, int]:
    r0, c0, r1, c1 = _bbox(coords)
    obj = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
    colors = [int(color) for color in np.unique(obj) if int(color) != _BG]
    assert 2 <= len(colors) <= 4, "each object must contain 2-4 colors"

    rects = []
    for color in colors:
        color_coords = np.argwhere(obj == int(color))
        assert color_coords.size > 0
        cr0, cc0, cr1, cc1 = _bbox(color_coords)
        block = obj[cr0 : cr1 + 1, cc0 : cc1 + 1]
        assert np.all(block == int(color)), "each color block must be a solid rectangle"
        rects.append((int(cr0), int(cc0), int(cr1), int(cc1), int(color)))

    horizontal = all(cr0 == rects[0][0] and cr1 == rects[0][2] for cr0, _, cr1, _, _ in rects)
    vertical = all(cc0 == rects[0][1] and cc1 == rects[0][3] for _, cc0, _, cc1, _ in rects)
    assert horizontal ^ vertical, "color blocks must be laid out in exactly one axis"

    if horizontal:
        rects.sort(key=lambda item: item[1])
        cursor = 0
        for _, cc0, _, cc1, _ in rects:
            assert cc0 == cursor
            cursor = cc1 + 1
        assert cursor == obj.shape[1]
        direction = "horizontal"
    else:
        rects.sort(key=lambda item: item[0])
        cursor = 0
        for cr0, _, cr1, _, _ in rects:
            assert cr0 == cursor
            cursor = cr1 + 1
        assert cursor == obj.shape[0]
        direction = "vertical"

    return obj, direction, len(rects)


def _transform_object(obj: np.ndarray, direction: str, count: int) -> np.ndarray:
    if direction == "horizontal":
        return np.rot90(obj, -int(count % 4))
    assert direction == "vertical"
    return np.fliplr(obj) if count % 2 == 1 else obj.copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transform each multicolor object according to its internal block direction and block count.

    Horizontal block order -> rotate the whole object clockwise by 90 degrees count times.
    Vertical block order -> flip the whole object horizontally count times.
    Each transformed object stays centered on its original bounding-box center, and out-of-bounds
    pixels are clipped.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    out = np.zeros_like(g)
    for coords in _components(g != _BG):
        r0, c0, r1, c1 = _bbox(coords)
        obj, direction, count = _analyze_object(g, coords)
        transformed = _transform_object(obj, direction, count)
        nr0, nc0 = _centered_top_left(r0, c0, obj.shape, transformed.shape)

        for rr, cc in np.argwhere(transformed != _BG):
            tr = nr0 + int(rr)
            tc = nc0 + int(cc)
            if 0 <= tr < h and 0 <= tc < w:
                out[tr, tc] = int(transformed[rr, cc])

    return out


def _randint(lo: int, hi: int) -> int:
    return random.randint(int(lo), int(hi))


def _sample_colors(k: int) -> list[int]:
    return random.sample(list(range(1, 10)), int(k))


def _mark_dilated(mask: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = mask.shape
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    mask[nr, nc] = True


def _build_object(direction: str, count: int) -> np.ndarray:
    colors = _sample_colors(count)

    if direction == "horizontal":
        for _ in range(100):
            band_h = _randint(2, 4)
            widths = [_randint(1, 4) for _ in range(count)]
            total_w = sum(widths)
            if 3 <= total_w <= 10 and total_w % 2 == band_h % 2:
                obj = np.zeros((band_h, total_w), dtype=int)
                c0 = 0
                for color, width in zip(colors, widths):
                    obj[:, c0 : c0 + width] = int(color)
                    c0 += width
                return obj
        raise RuntimeError("failed to build horizontal object")

    for _ in range(100):
        band_w = _randint(2, 4)
        heights = [_randint(1, 4) for _ in range(count)]
        total_h = sum(heights)
        if 3 <= total_h <= 10 and total_h % 2 == band_w % 2:
            obj = np.zeros((total_h, band_w), dtype=int)
            r0 = 0
            for color, height in zip(colors, heights):
                obj[r0 : r0 + height, :] = int(color)
                r0 += height
            return obj
    raise RuntimeError("failed to build vertical object")


def _in_bounds_cells(top: tuple[int, int], obj: np.ndarray, shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    r0, c0 = top
    cells: list[tuple[int, int]] = []
    for rr, cc in np.argwhere(obj != _BG):
        tr = r0 + int(rr)
        tc = c0 + int(cc)
        if 0 <= tr < h and 0 <= tc < w:
            cells.append((tr, tc))
    return cells


def _try_place_object(
    grid: np.ndarray,
    input_blocked: np.ndarray,
    output_blocked: np.ndarray,
    obj: np.ndarray,
    transformed: np.ndarray,
) -> bool:
    h, w = grid.shape
    oh, ow = obj.shape

    positions = [(r, c) for r in range(h - oh + 1) for c in range(w - ow + 1)]
    random.shuffle(positions)

    for r0, c0 in positions:
        in_cells = [(r0 + int(rr), c0 + int(cc)) for rr, cc in np.argwhere(obj != _BG)]
        if any(input_blocked[r, c] for r, c in in_cells):
            continue

        nr0, nc0 = _centered_top_left(r0, c0, obj.shape, transformed.shape)
        out_cells = _in_bounds_cells((nr0, nc0), transformed, grid.shape)
        if not out_cells:
            continue
        if any(output_blocked[r, c] for r, c in out_cells):
            continue

        for rr, cc in np.argwhere(obj != _BG):
            grid[r0 + int(rr), c0 + int(cc)] = int(obj[rr, cc])
        _mark_dilated(input_blocked, in_cells)
        _mark_dilated(output_blocked, out_cells)
        return True

    return False


def _direction_plan(n_objects: int) -> list[str]:
    if n_objects == 2:
        dirs = ["horizontal", "vertical"]
    elif n_objects == 3:
        dirs = ["horizontal", "horizontal", "vertical"] if random.random() < 0.5 else ["horizontal", "vertical", "vertical"]
    else:
        dirs = ["horizontal", "horizontal", "vertical", "vertical"]
    random.shuffle(dirs)
    return dirs


def _gen_attempt() -> np.ndarray | None:
    h = _randint(16, 24)
    w = _randint(16, 24)
    n_objects = _randint(2, 4)

    grid = np.zeros((h, w), dtype=int)
    input_blocked = np.zeros((h, w), dtype=bool)
    output_blocked = np.zeros((h, w), dtype=bool)

    counts = [_randint(2, 4) for _ in range(n_objects)]
    directions = _direction_plan(n_objects)

    specs = list(zip(directions, counts))
    random.shuffle(specs)
    for direction, count in specs:
        obj = _build_object(direction, count)
        transformed = _transform_object(obj, direction, count)
        if not _try_place_object(grid, input_blocked, output_blocked, obj, transformed):
            return None

    comps = _components(grid != _BG)
    if len(comps) != n_objects:
        return None
    if not {"horizontal", "vertical"}.issubset(set(directions)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is None:
            continue
        out = main(grid)
        if isinstance(out, np.ndarray) and out.shape == grid.shape and np.any(grid != _BG):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
