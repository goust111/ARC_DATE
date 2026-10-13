from __future__ import annotations

import numpy as np


CoordArray = np.ndarray

_NEIGHBORS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]
_NEIGHBORS4 = [(-1, 0), (0, -1), (0, 1), (1, 0)]

_BG = 0
_PALETTE = np.arange(1, 10, dtype=int)
_GEN_ATTEMPTS = 300
_PLACE_ATTEMPTS = 120


def _find_components(grid: np.ndarray, bg: int = _BG, connectivity: int = 8) -> list[CoordArray]:
    neighbors = _NEIGHBORS8 if connectivity == 8 else _NEIGHBORS4
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[CoordArray] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in neighbors:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != bg:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))

    comps.sort(key=lambda coords: _bbox(coords))
    return comps


def _bbox(coords: CoordArray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return (
        int(rows.min()),
        int(cols.min()),
        int(rows.max()),
        int(cols.max()),
    )


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Deconstruct each multi-color 8-connected object into row/column projections on its bounding box border.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    for coords in _find_components(g, bg=_BG, connectivity=8):
        r0, c0, r1, c1 = _bbox(coords)
        out[r0 : r1 + 1, c0 : c1 + 1] = _BG

        order = np.lexsort((coords[:, 1], coords[:, 0]))
        for idx in order:
            r = int(coords[idx, 0])
            c = int(coords[idx, 1])
            color = int(g[r, c])
            out[r0, c] = color
            out[r1, c] = color
            out[r, c0] = color
            out[r, c1] = color

    return out


def _mask_connected(mask: np.ndarray) -> bool:
    return len(_find_components(mask.astype(int), bg=0, connectivity=8)) == 1


def _has_hole(mask: np.ndarray) -> bool:
    h, w = mask.shape
    bg = ~mask
    seen = np.zeros((h, w), dtype=bool)
    stack: list[tuple[int, int]] = []

    for r in range(h):
        for c in (0, w - 1):
            if bg[r, c] and not seen[r, c]:
                seen[r, c] = True
                stack.append((int(r), int(c)))
    for c in range(w):
        for r in (0, h - 1):
            if bg[r, c] and not seen[r, c]:
                seen[r, c] = True
                stack.append((int(r), int(c)))

    while stack:
        r, c = stack.pop()
        for dr, dc in _NEIGHBORS4:
            nr = int(r + dr)
            nc = int(c + dc)
            if 0 <= nr < h and 0 <= nc < w and bg[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))

    return bool(np.any(bg & ~seen))


def _valid_mask(mask: np.ndarray) -> bool:
    if int(mask.sum()) < 5:
        return False
    if not _mask_connected(mask):
        return False
    if np.any(mask.sum(axis=0) == 0) or np.any(mask.sum(axis=1) == 0):
        return False
    if not np.any(mask[1:-1, 1:-1]):
        return False
    if not np.any(mask.sum(axis=0) >= 2) or not np.any(mask.sum(axis=1) >= 2):
        return False
    if _has_hole(mask):
        return False
    return True


def _make_mask(rng: np.random.Generator, h: int, w: int) -> np.ndarray | None:
    area_min = max(5, h + w - 1, int(np.ceil(0.45 * h * w)))
    area_max = min(h * w - 1, int(np.floor(0.8 * h * w)))
    if area_min > area_max:
        return None

    target = int(rng.integers(area_min, area_max + 1))
    mask = np.ones((h, w), dtype=bool)

    while int(mask.sum()) > target:
        coords = np.argwhere(mask)
        removable: list[tuple[int, int]] = []
        for idx in rng.permutation(len(coords)):
            r, c = coords[int(idx)]
            mask[r, c] = False
            if _valid_mask(mask):
                removable.append((int(r), int(c)))
            mask[r, c] = True
        if not removable:
            return None
        rr, cc = removable[int(rng.integers(len(removable)))]
        mask[rr, cc] = False

    return mask if _valid_mask(mask) else None


def _choose_witnesses(
    mask: np.ndarray, rng: np.random.Generator
) -> tuple[int, int, int, int, int, int]:
    row_options = []
    for r in range(mask.shape[0]):
        cols = np.flatnonzero(mask[r])
        if len(cols) >= 2:
            row_options.append((int(r), int(cols[0]), int(cols[-1])))

    col_options = []
    for c in range(mask.shape[1]):
        rows = np.flatnonzero(mask[:, c])
        if len(rows) >= 2:
            col_options.append((int(c), int(rows[0]), int(rows[-1])))

    rng.shuffle(row_options)
    rng.shuffle(col_options)

    best = None
    best_score = -1
    for r, left, right in row_options:
        for c, top, bottom in col_options:
            score = len({(r, left), (r, right), (top, c), (bottom, c)})
            if score > best_score:
                best = (r, left, right, c, top, bottom)
                best_score = score
            if score == 4:
                return best

    assert best is not None
    return best


def _row_has_color_conflict(mask: np.ndarray, patch: np.ndarray) -> bool:
    for r in range(mask.shape[0]):
        cols = np.flatnonzero(mask[r])
        if len(cols) >= 2 and len({int(patch[r, c]) for c in cols}) >= 2:
            return True
    return False


def _col_has_color_conflict(mask: np.ndarray, patch: np.ndarray) -> bool:
    for c in range(mask.shape[1]):
        rows = np.flatnonzero(mask[:, c])
        if len(rows) >= 2 and len({int(patch[r, c]) for r in rows}) >= 2:
            return True
    return False


def _colorize_mask(rng: np.random.Generator, mask: np.ndarray) -> np.ndarray | None:
    coords = [tuple(map(int, coord)) for coord in np.argwhere(mask)]
    r, left, right, c, top, bottom = _choose_witnesses(mask, rng)

    for _ in range(40):
        n_colors = 2 + int(rng.integers(2))
        colors = [int(x) for x in rng.choice(_PALETTE, size=n_colors, replace=False)]
        patch = np.zeros(mask.shape, dtype=int)
        patch[mask] = colors[0]

        protected = {(r, left), (r, right), (top, c), (bottom, c)}
        free_cells = [cell for cell in coords if cell not in protected]
        rng.shuffle(free_cells)
        for rr, cc in free_cells:
            patch[rr, cc] = int(colors[int(rng.integers(n_colors))])

        patch[top, c] = colors[1] if n_colors == 2 else colors[2]
        patch[bottom, c] = colors[0]
        patch[r, left] = colors[0]
        patch[r, right] = colors[1]

        present = {int(patch[rr, cc]) for rr, cc in coords}
        missing = [color for color in colors if color not in present]
        fill_candidates = free_cells + [cell for cell in coords if cell not in free_cells]
        for color, (rr, cc) in zip(missing, fill_candidates, strict=False):
            patch[rr, cc] = int(color)

        present = {int(patch[rr, cc]) for rr, cc in coords}
        if len(present) != n_colors:
            continue
        if not _row_has_color_conflict(mask, patch):
            continue
        if not _col_has_color_conflict(mask, patch):
            continue
        return patch

    return None


def _make_object_patch(rng: np.random.Generator) -> np.ndarray | None:
    for _ in range(60):
        h = int(rng.integers(3, 8))
        w = int(rng.integers(3, 8))
        mask = _make_mask(rng, h, w)
        if mask is None:
            continue
        patch = _colorize_mask(rng, mask)
        if patch is not None:
            return patch
    return None


def _place_patch(
    rng: np.random.Generator,
    grid: np.ndarray,
    patch: np.ndarray,
    reserved: list[tuple[int, int, int, int]],
) -> bool:
    h, w = patch.shape
    gh, gw = grid.shape
    if gh - h < 2 or gw - w < 2:
        return False

    mask = patch != _BG
    for _ in range(_PLACE_ATTEMPTS):
        r0 = int(rng.integers(1, gh - h))
        c0 = int(rng.integers(1, gw - w))
        box = (r0 - 1, c0 - 1, r0 + h, c0 + w)
        if any(_boxes_overlap(box, other) for other in reserved):
            continue
        region = grid[r0 : r0 + h, c0 : c0 + w]
        region[mask] = patch[mask]
        reserved.append(box)
        return True
    return False


def _component_patch(grid: np.ndarray, coords: CoordArray) -> tuple[np.ndarray, np.ndarray]:
    r0, c0, r1, c1 = _bbox(coords)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    mask = patch != _BG
    return patch, mask


def _scene_is_valid(grid: np.ndarray, n_objects: int) -> bool:
    comps = _find_components(grid, bg=_BG, connectivity=8)
    if len(comps) != n_objects:
        return False
    if np.array_equal(main(grid), grid):
        return False

    for coords in comps:
        patch, mask = _component_patch(grid, coords)
        colors = {int(patch[r, c]) for r, c in np.argwhere(mask)}
        if len(colors) not in (2, 3):
            return False
        if not _row_has_color_conflict(mask, patch):
            return False
        if not _col_has_color_conflict(mask, patch):
            return False

    return True


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(18, 31))
    w = int(rng.integers(18, 31))
    n_objects = int(rng.integers(2, 5))
    grid = np.zeros((h, w), dtype=int)
    reserved: list[tuple[int, int, int, int]] = []

    for _ in range(n_objects):
        patch = _make_object_patch(rng)
        if patch is None or not _place_patch(rng, grid, patch, reserved):
            return None

    if not _scene_is_valid(grid, n_objects):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_GEN_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
