import random
from collections import deque

import numpy as np


BLACK = 0
AZURE = 8
COLORS = [1, 2, 3, 4, 5, 6, 7, 9]


def _mirror_info(grid):
    row_hits = np.where(np.all(grid == AZURE, axis=1))[0]
    if len(row_hits):
        return "h", int(row_hits[0])
    col_hits = np.where(np.all(grid == AZURE, axis=0))[0]
    if len(col_hits):
        return "v", int(col_hits[0])
    raise ValueError("no full azure mirror line found")


def _object_side(grid, orientation, mirror):
    mask = (grid != BLACK) & (grid != AZURE)
    if orientation == "h":
        before = int(np.count_nonzero(mask[:mirror, :]))
        after = int(np.count_nonzero(mask[mirror + 1 :, :]))
    else:
        before = int(np.count_nonzero(mask[:, :mirror]))
        after = int(np.count_nonzero(mask[:, mirror + 1 :]))
    return -1 if before >= after else 1


def _on_side(r, c, orientation, mirror, side):
    if orientation == "h":
        return r < mirror if side < 0 else r > mirror
    return c < mirror if side < 0 else c > mirror


def _band_cells(grid, orientation, mirror, side):
    h, w = grid.shape
    candidates = []

    for r in (0, h - 1):
        coords = []
        for c in range(w):
            if _on_side(r, c, orientation, mirror, side):
                val = int(grid[r, c])
                if val not in (BLACK, AZURE):
                    coords.append((r, c))
        if coords:
            candidates.append(("row", coords))

    for c in (0, w - 1):
        coords = []
        for r in range(h):
            if _on_side(r, c, orientation, mirror, side):
                val = int(grid[r, c])
                if val not in (BLACK, AZURE):
                    coords.append((r, c))
        if coords:
            candidates.append(("col", coords))

    if not candidates:
        return []

    axis, coords = max(candidates, key=lambda item: len(item[1]))
    if axis == "row":
        coords.sort(key=lambda rc: rc[1])
    else:
        coords.sort(key=lambda rc: rc[0])
    return coords


def _same_color_components(grid, orientation, mirror, side, excluded):
    h, w = grid.shape
    seen = set(excluded)
    comps = []
    for r in range(h):
        for c in range(w):
            val = int(grid[r, c])
            if (
                val in (BLACK, AZURE)
                or (r, c) in seen
                or not _on_side(r, c, orientation, mirror, side)
            ):
                continue
            color = val
            cells = []
            q = deque([(r, c)])
            seen.add((r, c))
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if (nr, nc) in seen:
                        continue
                    if not _on_side(nr, nc, orientation, mirror, side):
                        continue
                    if int(grid[nr, nc]) != color:
                        continue
                    seen.add((nr, nc))
                    q.append((nr, nc))
            comps.append(cells)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """Reflect same-side objects across an azure mirror and recolor the reflections from an edge band.

    Input structure:
    The background is black. A full horizontal or vertical azure line is the mirror.
    All colored single-color 4-connected objects lie on one side of that mirror, and
    the opposite side is empty. On an outer edge of the object side, a color band
    contains one visible marker color for each object; these marker colors are not
    object colors.

    Rule:
    Find the objects on the occupied side, excluding the mirror and the edge color
    band. Sort the objects by their nearest pixel distance to the mirror, from
    nearest to farthest. Assign the band colors in visible order, left-to-right for
    an edge row or top-to-bottom for an edge column. Draw each object's mirror image
    on the empty side using its assigned band color while leaving the original
    objects and mirror unchanged.

    Key insight:
    The reflected shape comes from ordinary axial symmetry, but its color is keyed
    by distance order against the visible edge band, not by the original object color.
    """
    out = np.array(grid, copy=True)
    orientation, mirror = _mirror_info(grid)
    side = _object_side(grid, orientation, mirror)
    band = _band_cells(grid, orientation, mirror, side)
    band_colors = [int(grid[r, c]) for r, c in band]
    comps = _same_color_components(grid, orientation, mirror, side, set(band))

    def comp_key(cells):
        if orientation == "h":
            dist = min(abs(r - mirror) for r, _ in cells)
        else:
            dist = min(abs(c - mirror) for _, c in cells)
        return dist, min(cells)

    comps.sort(key=comp_key)
    h, w = grid.shape
    for color, cells in zip(band_colors, comps):
        for r, c in cells:
            if orientation == "h":
                nr, nc = 2 * mirror - r, c
            else:
                nr, nc = r, 2 * mirror - c
            if 0 <= nr < h and 0 <= nc < w:
                out[nr, nc] = color
    return out


def _random_shape(rng):
    cells = {(0, 0)}
    target = rng.randint(2, 6)
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    while len(cells) < target:
        r, c = rng.choice(tuple(cells))
        dr, dc = rng.choice(dirs)
        cells.add((r + dr, c + dc))
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return sorted((r - min_r, c - min_c) for r, c in cells)


def _can_place(cells, occupied):
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (r + dr, c + dc) in occupied:
                    return False
    return True


def _place_objects(grid, rng, orientation, mirror, side, object_colors):
    h, w = grid.shape
    occupied = set()
    used_distances = set()

    for color in object_colors:
        placed = False
        for _ in range(800):
            shape = _random_shape(rng)
            max_r = max(r for r, _ in shape)
            max_c = max(c for _, c in shape)

            if orientation == "h":
                row_min = 1 if side < 0 else mirror + 1
                row_max = mirror - 1 - max_r if side < 0 else h - 2 - max_r
                col_min, col_max = 1, w - 2 - max_c
            else:
                row_min, row_max = 1, h - 2 - max_r
                col_min = 1 if side < 0 else mirror + 1
                col_max = mirror - 1 - max_c if side < 0 else w - 2 - max_c

            if row_min > row_max or col_min > col_max:
                continue

            ar = rng.randint(row_min, row_max)
            ac = rng.randint(col_min, col_max)
            cells = [(ar + r, ac + c) for r, c in shape]

            if orientation == "h":
                dist = min(abs(r - mirror) for r, _ in cells)
                mirrored = [(2 * mirror - r, c) for r, c in cells]
            else:
                dist = min(abs(c - mirror) for _, c in cells)
                mirrored = [(r, 2 * mirror - c) for r, c in cells]

            if dist in used_distances:
                continue
            if any(not (0 <= r < h and 0 <= c < w) for r, c in mirrored):
                continue
            if not _can_place(cells, occupied):
                continue

            for r, c in cells:
                grid[r, c] = color
            occupied.update(cells)
            used_distances.add(dist)
            placed = True
            break

        if not placed:
            return False
    return True


def generate_input() -> np.ndarray:
    rng = random
    for _ in range(1000):
        h = rng.randint(12, 30)
        w = rng.randint(12, 30)
        orientation = rng.choice(("h", "v"))
        side = rng.choice((-1, 1))

        if orientation == "h":
            mirror = rng.randint(4, h - 5)
        else:
            mirror = rng.randint(4, w - 5)

        n_objects = rng.randint(2, 4)
        palette = rng.sample(COLORS, 2 * n_objects)
        object_colors = palette[:n_objects]
        band_colors = palette[n_objects:]

        grid = np.zeros((h, w), dtype=int)
        if orientation == "h":
            grid[mirror, :] = AZURE
            band_row = 0 if side < 0 else h - 1
            start = rng.randint(1, w - n_objects - 1)
            for i, color in enumerate(band_colors):
                grid[band_row, start + i] = color
        else:
            grid[:, mirror] = AZURE
            band_col = 0 if side < 0 else w - 1
            start = rng.randint(1, h - n_objects - 1)
            for i, color in enumerate(band_colors):
                grid[start + i, band_col] = color

        rng.shuffle(object_colors)
        if _place_objects(grid, rng, orientation, mirror, side, object_colors):
            return grid

    raise RuntimeError("could not generate a valid puzzle input")
