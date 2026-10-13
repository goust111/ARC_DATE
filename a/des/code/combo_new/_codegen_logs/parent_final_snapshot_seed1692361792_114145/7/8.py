from __future__ import annotations

import random

import numpy as np


_BG = 0
_RED = 2
_YELLOW = 4
_OBJECT_COLORS = (1, 3, 5, 6, 7, 8, 9)
_CARDINALS = ((1, 0), (-1, 0), (0, 1), (0, -1))

_SHAPES = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (1, 0), (1, 1), (2, 1)),
    ((0, 1), (1, 0), (1, 1), (2, 0)),
    ((0, 1), (1, 0), (1, 1), (2, 1)),
    ((1, 0), (0, 1), (1, 1), (2, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
)

_MIN_HALF = 9
_MAX_HALF = 14
_GEN_ATTEMPTS = 400
_QUADRANT_ATTEMPTS = 150
_OBJECT_ATTEMPTS = 80


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in _CARDINALS:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.asarray(coords, dtype=int))
    return parts


def _cross_indices(grid: np.ndarray) -> tuple[int, int]:
    red_rows = np.where(np.all(grid == _RED, axis=1))[0]
    red_cols = np.where(np.all(grid == _RED, axis=0))[0]
    assert len(red_rows) == 1 and len(red_cols) == 1, "expected one solid red row and one solid red column"
    cross_r = int(red_rows[0])
    cross_c = int(red_cols[0])
    h, w = grid.shape
    assert 2 * cross_r + 1 == h and 2 * cross_c + 1 == w, "red cross must split the grid into four centered quadrants"
    return cross_r, cross_c


def _anchor_positions(height: int, width: int) -> dict[str, tuple[int, int]]:
    return {
        "tl": (0, 0),
        "tr": (0, width - 1),
        "bl": (height - 1, 0),
        "br": (height - 1, width - 1),
    }


def _quadrant_slices(cross_r: int, cross_c: int, height: int, width: int) -> dict[str, tuple[slice, slice]]:
    return {
        "tl": (slice(0, cross_r), slice(0, cross_c)),
        "tr": (slice(0, cross_r), slice(cross_c + 1, width)),
        "bl": (slice(cross_r + 1, height), slice(0, cross_c)),
        "br": (slice(cross_r + 1, height), slice(cross_c + 1, width)),
    }


def _to_local(name: str, row: int, col: int, height: int, width: int) -> tuple[int, int]:
    if name == "tl":
        return row, col
    if name == "tr":
        return row, width - 1 - col
    if name == "bl":
        return height - 1 - row, col
    return height - 1 - row, width - 1 - col


def _from_local(name: str, row: int, col: int, height: int, width: int) -> tuple[int, int]:
    if name == "tl":
        return row, col
    if name == "tr":
        return row, width - 1 - col
    if name == "bl":
        return height - 1 - row, col
    return height - 1 - row, width - 1 - col


def _quadrant_components(grid: np.ndarray, subgrid: np.ndarray) -> list[np.ndarray]:
    movable = (subgrid != _BG) & (subgrid != _RED) & (subgrid != _YELLOW)
    comps = _components(movable)
    assert 1 <= len(comps) <= 3, "each quadrant must contain between one and three objects"

    seen_colors: set[int] = set()
    for comp in comps:
        colors = {int(subgrid[r, c]) for r, c in comp}
        assert len(colors) == 1, "each object must be monochrome"
        color = colors.pop()
        assert color not in seen_colors, "objects in the same quadrant must have different colors"
        seen_colors.add(color)
        rows = comp[:, 0]
        cols = comp[:, 1]
        assert int(rows.max() - rows.min() + 1) <= 3, "object bbox too tall"
        assert int(cols.max() - cols.min() + 1) <= 3, "object bbox too wide"
    return comps


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = g.shape
    assert max(height, width) <= 30, "ARC grids must stay within 30x30"

    cross_r, cross_c = _cross_indices(g)
    anchors = _anchor_positions(height, width)
    assert set(map(tuple, np.argwhere(g == _YELLOW))) == set(anchors.values()), "expected four yellow corner anchors"

    out = g.copy()
    movable_mask = (g != _BG) & (g != _RED) & (g != _YELLOW)
    out[movable_mask] = _BG

    for name, (row_slice, col_slice) in _quadrant_slices(cross_r, cross_c, height, width).items():
        subgrid = g[row_slice, col_slice]
        _quadrant_components(g, subgrid)

        local_pixels: list[tuple[int, int, int]] = []
        for local_r, local_c in np.argwhere((subgrid != _BG) & (subgrid != _RED) & (subgrid != _YELLOW)):
            global_r = row_slice.start + int(local_r)
            global_c = col_slice.start + int(local_c)
            inward_r, inward_c = _to_local(name, global_r, global_c, height, width)
            local_pixels.append((inward_r, inward_c, int(g[global_r, global_c])))

        assert local_pixels, "quadrant must contain at least one movable pixel"
        min_r = min(pixel[0] for pixel in local_pixels)
        min_c = min(pixel[1] for pixel in local_pixels)

        for inward_r, inward_c, color in local_pixels:
            target_r, target_c = _from_local(name, inward_r - min_r, inward_c - min_c, height, width)
            assert out[target_r, target_c] == _BG, "translated pattern overlaps a fixed marker or another object"
            out[target_r, target_c] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group each quadrant's small colored objects and redraw the quadrant composite at its yellow corner anchor.

    Input structure:
    - The background is black.
    - One solid red horizontal line and one solid red vertical line cross at the center and split the grid into four quadrants.
    - Each quadrant contains exactly one yellow corner anchor plus 1 to 3 small 4-connected monochrome objects whose bounding boxes are at most 3x3.

    Rule:
    - Solve each quadrant independently while keeping the red cross and yellow anchor fixed.
    - Collect all non-anchor object pixels in that quadrant and preserve their internal shapes and mutual relative offsets.
    - Translate the whole quadrant arrangement toward the outer corner so that the composite bounding box aligns with the yellow anchor.
    - Clear the original object locations and redraw only the translated composite.

    Key insight:
    - The red cross is only a partition: each quadrant is a local scene whose colored objects must be merged into one preserved arrangement and then packed against its fixed corner anchor.
    """
    return _transform(grid)


def _shape_size(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    return max(rows) + 1, max(cols) + 1


def _expanded_cells(cells: set[tuple[int, int]], height: int, width: int) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for row, col in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = row + dr
                nc = col + dc
                if 0 <= nr < height and 0 <= nc < width:
                    expanded.add((nr, nc))
    return expanded


def _generate_quadrant(local_h: int, local_w: int, n_objects: int) -> list[tuple[int, set[tuple[int, int]]]] | None:
    for _ in range(_QUADRANT_ATTEMPTS):
        colors = random.sample(_OBJECT_COLORS, n_objects)
        placements: list[tuple[int, set[tuple[int, int]]]] = []
        blocked: set[tuple[int, int]] = set()
        success = True

        for color in colors:
            placed = False
            for _ in range(_OBJECT_ATTEMPTS):
                shape = random.choice(_SHAPES)
                shape_h, shape_w = _shape_size(shape)
                max_r0 = local_h - shape_h - 2
                max_c0 = local_w - shape_w - 2
                if max_r0 < 1 or max_c0 < 1:
                    success = False
                    break

                top = random.randint(1, max_r0)
                left = random.randint(1, max_c0)
                cells = {(top + dr, left + dc) for dr, dc in shape}
                if cells & blocked:
                    continue

                placements.append((color, cells))
                blocked |= _expanded_cells(cells, local_h, local_w)
                placed = True
                break
            if not placed:
                success = False
                break

        if not success:
            continue

        all_cells = set().union(*(cells for _, cells in placements))
        min_row = min(row for row, _ in all_cells)
        min_col = min(col for _, col in all_cells)
        if (min_row, min_col) in all_cells:
            continue
        return placements
    return None


def _render_quadrant(grid: np.ndarray, name: str, placements: list[tuple[int, set[tuple[int, int]]]]) -> None:
    height, width = grid.shape
    for color, cells in placements:
        for local_r, local_c in cells:
            global_r, global_c = _from_local(name, local_r, local_c, height, width)
            assert grid[global_r, global_c] == _BG
            grid[global_r, global_c] = color


def _gen_attempt() -> np.ndarray | None:
    half_h = random.randint(_MIN_HALF, _MAX_HALF)
    half_w = random.randint(_MIN_HALF, _MAX_HALF)
    height = 2 * half_h + 1
    width = 2 * half_w + 1

    grid = np.full((height, width), _BG, dtype=int)
    grid[half_h, :] = _RED
    grid[:, half_w] = _RED

    for row, col in _anchor_positions(height, width).values():
        grid[row, col] = _YELLOW

    quadrant_specs: dict[str, list[tuple[int, set[tuple[int, int]]]]] = {}
    for name in ("tl", "tr", "bl", "br"):
        count = random.randint(1, 3)
        placements = _generate_quadrant(half_h, half_w, count)
        if placements is None:
            return None
        quadrant_specs[name] = placements

    for name, placements in quadrant_specs.items():
        _render_quadrant(grid, name, placements)

    try:
        out = _transform(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for quadrant-anchor composite puzzle")
