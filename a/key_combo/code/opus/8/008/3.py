from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BACKGROUND = 0
RECOLOR = 7
FOREGROUND_COLORS = (1, 2, 3, 4, 5, 6, 8, 9)
MOTIF_SIZES = (3, 4, 5)
MIN_PIXELS = 6
MAX_PIXELS = 12
MAX_GENERATION_ATTEMPTS = 600


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _crop_to_bbox(grid: np.ndarray, mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, coords[0]))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _fills_bbox(mask: np.ndarray) -> bool:
    rows, cols = np.where(mask)
    return (
        rows.size > 0
        and int(rows.min()) == 0
        and int(rows.max()) == mask.shape[0] - 1
        and int(cols.min()) == 0
        and int(cols.max()) == mask.shape[1] - 1
    )


def _background_reachable_from_border(grid: np.ndarray, bg: int) -> np.ndarray:
    h, w = grid.shape
    reachable = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if int(grid[r, c]) == bg and not reachable[r, c]:
                reachable[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if int(grid[r, c]) == bg and not reachable[r, c]:
                reachable[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == bg and not reachable[nr, nc]:
                reachable[nr, nc] = True
                q.append((nr, nc))

    return reachable


def _expand_motif(motif: np.ndarray, fg_color: int, bg: int) -> np.ndarray:
    k = motif.shape[0]
    out = np.full((k * k, k * k), bg, dtype=int)
    stamp = np.where(motif == fg_color, fg_color, bg)
    for block_r, block_c in np.argwhere(motif == fg_color):
        r0 = int(block_r) * k
        c0 = int(block_c) * k
        out[r0 : r0 + k, c0 : c0 + k] = stamp
    return out


def _has_enclosed_background(grid: np.ndarray, bg: int) -> bool:
    reachable = _background_reachable_from_border(grid, bg)
    return bool(np.any((grid == bg) & ~reachable))


def _random_connected_mask(k: int, count: int) -> np.ndarray:
    mask = np.zeros((k, k), dtype=bool)
    seed = (random.randrange(k), random.randrange(k))
    mask[seed] = True
    frontier = set()
    for nr, nc in _neighbors4(*seed):
        if 0 <= nr < k and 0 <= nc < k:
            frontier.add((nr, nc))

    while int(mask.sum()) < count:
        if not frontier:
            return mask
        r, c = random.choice(tuple(frontier))
        frontier.remove((r, c))
        mask[r, c] = True
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < k and 0 <= nc < k and not mask[nr, nc]:
                frontier.add((nr, nc))

    return mask


def _generate_mask(k: int, want_original_hole: bool) -> np.ndarray | None:
    min_count = MIN_PIXELS
    max_count = min(MAX_PIXELS, k * k)
    if want_original_hole:
        min_count = max(min_count, 8)

    for _ in range(250):
        count = random.randint(min_count, max_count)
        mask = _random_connected_mask(k, count)
        if int(mask.sum()) != count:
            continue
        if not _fills_bbox(mask):
            continue
        if not _connected(mask):
            continue
        if _has_enclosed_background(mask.astype(int), 0) != want_original_hole:
            continue
        if not _has_enclosed_background(_expand_motif(mask.astype(int), 1, 0), 0):
            continue
        return mask
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a compact motif into a self-similar block pattern, then recolor the emergent enclosed background regions.

    Input structure:
    - The grid has a black background and a single-color compact motif in the upper-left area.
    - The motif's foreground cells form one 4-connected object whose bounding box is a 3x3, 4x4, or 5x5 square.
    - The full input canvas is the square of that bounding-box side length.

    Rule:
    - Crop the motif to its square bounding box.
    - Build a K^2 x K^2 canvas and stamp the full KxK motif into each KxK block whose block position corresponds to a foreground cell in the motif.
    - Flood-fill the expanded black background from the grid boundary using 4-connectivity.
    - Recolor every remaining black cell that was not boundary-reachable to orange.

    Key insight:
    - The important holes are not read from the original motif; they are created by the motif's self-similar expansion and must be detected after stamping.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _most_common_color(g)
    fg_colors = [color for color in np.unique(g) if int(color) != bg]
    assert len(fg_colors) == 1, "expected exactly one foreground color"
    fg_color = int(fg_colors[0])

    fg_mask = g == fg_color
    assert np.any(fg_mask), "expected a foreground motif"
    motif = _crop_to_bbox(g, fg_mask)
    assert motif.shape[0] == motif.shape[1], "expected a square motif bounding box"

    expanded = _expand_motif(motif, fg_color, bg)
    reachable = _background_reachable_from_border(expanded, bg)
    out = expanded.copy()
    out[(expanded == bg) & ~reachable] = RECOLOR
    return out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        k = random.choice(MOTIF_SIZES)
        want_original_hole = random.random() < 0.35
        mask = _generate_mask(k, want_original_hole)
        if mask is None:
            continue

        color = random.choice(FOREGROUND_COLORS)
        grid = np.full((k * k, k * k), BACKGROUND, dtype=int)
        grid[:k, :k] = np.where(mask, color, BACKGROUND)

        out = main(grid)
        if not np.any(out == RECOLOR):
            continue
        return grid

    raise RuntimeError("failed to generate a valid input")
