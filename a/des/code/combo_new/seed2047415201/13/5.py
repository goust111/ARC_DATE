from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


AZURE = 8
PALETTE = [0, 1, 2, 3, 4, 5, 6, 7, 9]
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue

        cells: list[tuple[int, int]] = []
        q: deque[tuple[int, int]] = deque([(sr, sc)])
        seen[sr, sc] = True
        while q:
            r, c = q.popleft()
            cells.append((r, c))
            for dr, dc in NEIGHBORS4:
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    q.append((rr, cc))

        comps.append(np.array(cells, dtype=int))

    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn each multicolor foreground object into a horizontal band in diagonal scan order.

    Input structure:
    - The background is Azure.
    - Several multicolor 4-connected foreground objects are scattered on the grid.
    - Each object contains a mixture of two to four non-background colors in an irregular shape.

    Rule:
    - For each foreground object, read its pixels by increasing row+column, then by increasing row.
    - Clear the original object from the grid.
    - Draw the read color sequence left-to-right on the first row of the object's original bounding box.
    - Start at the bounding box's left column and truncate the band if it reaches the right grid edge.

    Key insight: The 2D object is recoded as a 1D color sequence, and the hidden order is diagonal scan rather than row-major or column-major order.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    objects = []
    for coords in _connected_components(g != AZURE):
        rows = coords[:, 0]
        cols = coords[:, 1]
        r0 = int(rows.min())
        c0 = int(cols.min())
        scan_cells = sorted(
            ((int(r), int(c)) for r, c in coords),
            key=lambda rc: (rc[0] + rc[1], rc[0]),
        )
        colors = [int(g[r, c]) for r, c in scan_cells]
        color_counts = Counter(colors)
        assert 2 <= len(color_counts) <= 4, "expected each object to use two to four colors"
        objects.append((r0, c0, coords, colors))

    for _r0, _c0, coords, _colors in objects:
        out[coords[:, 0], coords[:, 1]] = AZURE

    for r0, c0, _coords, colors in sorted(objects, key=lambda item: (item[0], item[1])):
        run_len = min(len(colors), w - c0)
        if run_len > 0:
            out[r0, c0 : c0 + run_len] = np.asarray(colors[:run_len], dtype=int)

    assert h <= 30 and w <= 30
    return out


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.nonzero(mask)
    return mask[int(rows.min()) : int(rows.max()) + 1, int(cols.min()) : int(cols.max()) + 1]


def _orderings_differ(mask: np.ndarray) -> bool:
    coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    diag = sorted(coords, key=lambda rc: (rc[0] + rc[1], rc[0]))
    row = sorted(coords, key=lambda rc: (rc[0], rc[1]))
    col = sorted(coords, key=lambda rc: (rc[1], rc[0]))
    return diag != row and diag != col


def _random_polyomino(area: int) -> np.ndarray | None:
    for _ in range(80):
        max_side = min(7, area)
        height = random.randint(2, max_side)
        width = random.randint(2, max_side)
        if height * width < area:
            continue

        cells = {(random.randrange(height), random.randrange(width))}
        while len(cells) < area:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for dr, dc in NEIGHBORS4:
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < height and 0 <= cc < width and (rr, cc) not in cells:
                        frontier.append((rr, cc))
            if not frontier:
                break
            cells.add(random.choice(frontier))

        if len(cells) != area:
            continue
        mask = np.zeros((height, width), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _crop_mask(mask)
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if bool(mask.all()):
            continue
        if not _orderings_differ(mask):
            continue
        return mask

    return None


def _make_patch(area: int) -> np.ndarray | None:
    mask = _random_polyomino(area)
    if mask is None:
        return None

    coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    diag = sorted(coords, key=lambda rc: (rc[0] + rc[1], rc[0]))
    row = sorted(coords, key=lambda rc: (rc[0], rc[1]))
    col = sorted(coords, key=lambda rc: (rc[1], rc[0]))

    color_count = random.randint(2, min(4, len(coords)))
    colors = random.sample(PALETTE, color_count)

    for _ in range(80):
        values = colors[:] + [random.choice(colors) for _ in range(len(coords) - color_count)]
        random.shuffle(values)
        patch = np.full(mask.shape, AZURE, dtype=int)
        for (r, c), value in zip(coords, values):
            patch[r, c] = int(value)

        diag_seq = [int(patch[r, c]) for r, c in diag]
        row_seq = [int(patch[r, c]) for r, c in row]
        col_seq = [int(patch[r, c]) for r, c in col]
        if diag_seq != row_seq and diag_seq != col_seq and len(Counter(diag_seq)) == color_count:
            return patch

    return None


def _can_place(blocked: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    mask = patch != AZURE
    ph, pw = patch.shape
    return not bool(np.any(blocked[r0 : r0 + ph, c0 : c0 + pw] & mask))


def _mark_blocked(blocked: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = blocked.shape
    for dr, dc in np.argwhere(patch != AZURE):
        r = int(r0 + dr)
        c = int(c0 + dc)
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                blocked[rr, cc] = True


def _place_patch(
    grid: np.ndarray,
    blocked: np.ndarray,
    band_used: np.ndarray,
    patch: np.ndarray,
    want_truncation: bool,
) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    area = int(np.count_nonzero(patch != AZURE))
    candidates: list[tuple[int, int]] = []

    if ph > h or pw > w:
        return False

    for r0 in range(0, h - ph + 1):
        for c0 in range(0, w - pw + 1):
            is_truncated = area > w - c0
            if is_truncated != want_truncation:
                continue
            band_end = min(w, c0 + area)
            if bool(band_used[r0, c0:band_end].any()):
                continue
            if _can_place(blocked, patch, r0, c0):
                candidates.append((r0, c0))

    if not candidates:
        return False

    r0, c0 = random.choice(candidates)
    mask = patch != AZURE
    rr, cc = np.nonzero(mask)
    grid[r0 + rr, c0 + cc] = patch[rr, cc]
    band_used[r0, c0 : min(w, c0 + area)] = True
    _mark_blocked(blocked, patch, r0, c0)
    return True


def _generated_grid_is_valid(grid: np.ndarray, require_truncation: bool) -> bool:
    comps = _connected_components(grid != AZURE)
    if len(comps) < 2:
        return False

    saw_truncation = False
    for coords in comps:
        rows = coords[:, 0]
        cols = coords[:, 1]
        area = int(coords.shape[0])
        c0 = int(cols.min())
        if area > grid.shape[1] - c0:
            saw_truncation = True
        colors = [int(grid[int(r), int(c)]) for r, c in coords]
        if not 2 <= len(Counter(colors)) <= 4:
            return False

    if saw_truncation != require_truncation:
        return False

    try:
        out = main(grid)
    except Exception:
        return False
    return out.ndim == 2 and max(out.shape) <= 30


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(16, 28)
    width = random.randint(14, 30)
    object_count = random.randint(2, 5)
    require_truncation = random.random() < 0.5

    grid = np.full((height, width), AZURE, dtype=int)
    blocked = np.zeros((height, width), dtype=bool)
    band_used = np.zeros((height, width), dtype=bool)

    trunc_flags = [require_truncation] + [False] * (object_count - 1)
    for want_truncation in trunc_flags:
        if want_truncation:
            area_min = max(8, min(width + 1, 18))
            area_max = min(24, max(area_min, width + 8))
        else:
            area_min = 5
            area_max = min(18, width - 2)
        if area_max < area_min:
            return None

        placed = False
        for _ in range(80):
            area = random.randint(area_min, area_max)
            patch = _make_patch(area)
            if patch is None:
                continue
            if _place_patch(grid, blocked, band_used, patch, want_truncation):
                placed = True
                break
        if not placed:
            return None

    if _generated_grid_is_valid(grid, require_truncation):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after 300 attempts")
