import random

import numpy as np


_BG = 0
_COLOR_POOL = tuple(range(1, 10))
_GEN_MAX_ATTEMPTS = 400
_INNER_H_RANGE = (7, 19)
_INNER_W_RANGE = (7, 19)
_SELECTION_MODES = ("sparse", "medium", "dense")
_SHAPE_MODES = ("rectangle", "ell", "blob")


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    return mask[rows][:, cols]


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _choose_kept_indices(length: int, mode: str) -> list[int]:
    bands = {
        "sparse": (0.25, 0.45),
        "medium": (0.45, 0.65),
        "dense": (0.65, 0.82),
    }
    frac_lo, frac_hi = bands[mode]
    min_count = max(2, int(np.ceil(length * frac_lo)))
    max_count = min(length - 1, max(min_count, int(np.floor(length * frac_hi))))
    for _ in range(200):
        count = random.randint(min_count, max_count)
        picked = sorted(random.sample(range(length), count))
        if picked[-1] - picked[0] + 1 > len(picked):
            return picked
    return list(range(0, length, 2))


def _rectangle_mask() -> np.ndarray:
    while True:
        h = random.randint(1, 4)
        w = random.randint(1, 5)
        if h * w >= 3:
            return np.ones((h, w), dtype=bool)


def _ell_mask() -> np.ndarray:
    h = random.randint(3, 5)
    w = random.randint(3, 5)
    mask = np.zeros((h, w), dtype=bool)
    mask[:, 0] = True
    mask[-1, :] = True
    if random.random() < 0.5 and h >= 4:
        mask[h // 2 :, min(1, w - 1)] = True
    if random.random() < 0.5 and w >= 4:
        mask[max(0, h - 2), : w // 2 + 1] = True
    if random.random() < 0.5:
        mask = np.fliplr(mask)
    if random.random() < 0.5:
        mask = np.flipud(mask)
    return _trim_mask(mask)


def _blob_mask() -> np.ndarray:
    h = random.randint(2, 5)
    w = random.randint(2, 5)
    max_area = min(h * w, 9)
    min_area = min(max_area, 3)
    area = random.randint(min_area, max_area)
    mask = np.zeros((h, w), dtype=bool)
    seed = (random.randrange(h), random.randrange(w))
    mask[seed] = True
    filled = {seed}
    while len(filled) < area:
        frontier: set[tuple[int, int]] = set()
        for r, c in filled:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.add((nr, nc))
        if not frontier:
            break
        cell = random.choice(list(frontier))
        mask[cell] = True
        filled.add(cell)
    return _trim_mask(mask)


def _random_object_mask() -> np.ndarray:
    mode = random.choice(_SHAPE_MODES)
    if mode == "rectangle":
        return _rectangle_mask()
    if mode == "ell":
        return _ell_mask()
    return _blob_mask()


def _object_cells(mask: np.ndarray, top: int, left: int) -> list[tuple[int, int]]:
    return [
        (top + int(r), left + int(c))
        for r, c in np.argwhere(mask)
    ]


def _place_objects(
    inner_h: int, inner_w: int, masks: list[np.ndarray], colors: list[int]
) -> tuple[np.ndarray, list[list[tuple[int, int]]]] | None:
    interior = np.full((inner_h, inner_w), _BG, dtype=int)
    blocked = np.zeros((inner_h, inner_w), dtype=bool)
    objects: list[list[tuple[int, int]] | None] = [None] * len(masks)
    order = sorted(range(len(masks)), key=lambda idx: int(masks[idx].sum()), reverse=True)
    for idx in order:
        mask = masks[idx]
        h, w = mask.shape
        candidates: list[tuple[int, int]] = []
        for top in range(inner_h - h + 1):
            for left in range(inner_w - w + 1):
                cells = _object_cells(mask, top, left)
                if all(not blocked[r, c] for r, c in cells):
                    candidates.append((top, left))
        if not candidates:
            return None
        top, left = random.choice(candidates)
        cells = _object_cells(mask, top, left)
        for r, c in cells:
            interior[r, c] = colors[idx]
            for nr in range(max(0, r - 1), min(inner_h, r + 2)):
                for nc in range(max(0, c - 1), min(inner_w, c + 2)):
                    blocked[nr, nc] = True
        objects[idx] = cells
    return interior, [obj for obj in objects if obj is not None]


def _structure_is_valid(interior: np.ndarray, expected_count: int) -> bool:
    fg_mask = interior != _BG
    comps = _connected_components(fg_mask)
    if len(comps) != expected_count:
        return False
    for comp in comps:
        colors = {int(interior[r, c]) for r, c in comp}
        if len(colors) != 1:
            return False
    return True


def _scene_has_meaningful_crop(
    objects: list[list[tuple[int, int]]], keep_rows: set[int], keep_cols: set[int]
) -> bool:
    has_kept_pixels = False
    has_partial_object = False
    row_mask_matters = False
    col_mask_matters = False
    for cells in objects:
        kept = [(r, c) for r, c in cells if r in keep_rows and c in keep_cols]
        if kept:
            has_kept_pixels = True
            if len(kept) < len(cells):
                has_partial_object = True
            if any(r not in keep_rows for r, _c in cells):
                row_mask_matters = True
            if any(c not in keep_cols for _r, c in cells):
                col_mask_matters = True
    return has_kept_pixels and has_partial_object and row_mask_matters and col_mask_matters


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the interior scene by row and column rulers, then compact the retained cells into a smaller grid.

    Input structure:
    - The background is black.
    - The leftmost column and top row act as rulers with non-background marker pixels.
    - Marker pixels in the leftmost column indicate which interior rows to keep, and marker pixels in the top row indicate which interior columns to keep.
    - The interior area (excluding the ruler row and ruler column) contains several separate monochrome 4-connected objects.

    Rule:
    - Read the marked row positions from the leftmost column and the marked column positions from the top row.
    - Remove the ruler row and ruler column themselves from consideration.
    - Keep only interior pixels whose row is marked and whose column is marked.
    - Remove skipped row and column gaps so the retained cells become a compact output grid in their original order.

    Key insight:
    - The edge markers are not ordinary objects; together they define a row-mask and a column-mask whose intersection selects a compact subgrid of the whole interior scene.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2 and g.shape[1] >= 2

    bg = _most_common_color(g)
    keep_rows = np.flatnonzero(g[1:, 0] != bg)
    keep_cols = np.flatnonzero(g[0, 1:] != bg)
    assert keep_rows.size > 0 and keep_cols.size > 0

    interior = g[1:, 1:]
    return interior[np.ix_(keep_rows, keep_cols)].copy()


def _gen_attempt() -> np.ndarray | None:
    inner_h = random.randint(*_INNER_H_RANGE)
    inner_w = random.randint(*_INNER_W_RANGE)
    selection_mode = random.choice(_SELECTION_MODES)
    keep_rows = _choose_kept_indices(inner_h, selection_mode)
    keep_cols = _choose_kept_indices(inner_w, selection_mode)

    max_objects = min(5, len(_COLOR_POOL) - 2, max(2, (inner_h * inner_w) // 28))
    obj_count = random.randint(2, max_objects)
    colors = random.sample(list(_COLOR_POOL), obj_count + 2)
    row_marker_color, col_marker_color = colors[:2]
    object_colors = colors[2:]
    masks = [_random_object_mask() for _ in range(obj_count)]

    placed = _place_objects(inner_h, inner_w, masks, object_colors)
    if placed is None:
        return None
    interior, objects = placed

    grid = np.full((inner_h + 1, inner_w + 1), _BG, dtype=int)
    grid[1:, 1:] = interior
    grid[1 + np.array(keep_rows), 0] = row_marker_color
    grid[0, 1 + np.array(keep_cols)] = col_marker_color

    if np.count_nonzero(grid != _BG) >= grid.size // 2:
        return None
    if not _structure_is_valid(interior, obj_count):
        return None
    if not _scene_has_meaningful_crop(objects, set(keep_rows), set(keep_cols)):
        return None

    expected = interior[np.ix_(keep_rows, keep_cols)]
    if expected.size == 0 or np.all(expected == _BG):
        return None
    if expected.shape == interior.shape:
        return None
    if not np.array_equal(expected, main(grid)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ruler-crop ARC input")
