from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
YELLOW = 4

MASK_SIDE_MIN = 3
MASK_SIDE_MAX = 5
MAX_TEMPLATE_ATTEMPTS = 200
MAX_DIFF_CELLS = 6
GENERATION_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _touches_all_bbox_sides(mask: np.ndarray) -> bool:
    return (
        bool(np.any(mask[0, :]))
        and bool(np.any(mask[-1, :]))
        and bool(np.any(mask[:, 0]))
        and bool(np.any(mask[:, -1]))
    )


def _grow_connected_shape(h: int, w: int, area: int) -> np.ndarray | None:
    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    frontier = set(_neighbors4(start[0], start[1], h, w))

    while len(cells) < area and frontier:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        for nxt in _neighbors4(cell[0], cell[1], h, w):
            if nxt not in cells:
                frontier.add(nxt)

    if len(cells) != area:
        return None

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _random_template(h: int, w: int, allow_full: bool) -> np.ndarray | None:
    area_total = h * w
    max_area = area_total if allow_full else area_total - 1
    if max_area < 1:
        return None

    min_area = 1 if area_total <= 2 or random.random() < 0.2 else 2
    min_area = min(min_area, max_area)

    for _ in range(MAX_TEMPLATE_ATTEMPTS):
        area = random.randint(min_area, max_area)
        connected = random.random() < 0.85
        if connected:
            mask = _grow_connected_shape(h, w, area)
            if mask is None:
                continue
        else:
            mask = np.zeros((h, w), dtype=bool)
            picks = random.sample([(r, c) for r in range(h) for c in range(w)], area)
            for r, c in picks:
                mask[r, c] = True

        if not _touches_all_bbox_sides(mask):
            continue
        if not allow_full and bool(np.all(mask)):
            continue
        return mask
    return None


def _connected_subset(pool: set[tuple[int, int]], count: int) -> list[tuple[int, int]] | None:
    if not pool:
        return None

    start = random.choice(tuple(pool))
    chosen = {start}
    frontier = {cell for cell in _neighbors4(start[0], start[1], 99, 99) if cell in pool}

    while len(chosen) < count and frontier:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        chosen.add(cell)
        for nxt in _neighbors4(cell[0], cell[1], 99, 99):
            if nxt in pool and nxt not in chosen:
                frontier.add(nxt)

    if len(chosen) != count:
        return None
    return list(chosen)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read two aligned red masks twice: their overlap gives the motif and their disagreements give the stamp locations.

    Input structure:
    - One solid yellow separator row splits the grid into equally sized upper and lower masks.
    - Both masks use only black background and red active cells.
    - The two masks have at least one overlapping red cell.

    Rule:
    - Compute the cellwise intersection of the upper and lower red masks.
    - Crop that intersection to its tight bounding box; the cropped red pattern is the stamp motif.
    - Compute the cellwise symmetric difference of the two masks and treat it as a coarse placement mask.
    - Output a black canvas whose coarse cells match the mask size; each true coarse cell is expanded into one copy of the red motif, and each false coarse cell stays black.

    Key insight:
    - The same pair of masks is decoded in two ways: shared red cells define what to stamp, while mismatched red cells define where to stamp it.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    separator_rows = np.flatnonzero(np.all(g == YELLOW, axis=1))
    assert len(separator_rows) == 1, "expected exactly one solid yellow separator row"
    sep = int(separator_rows[0])

    top = g[:sep]
    bottom = g[sep + 1 :]
    assert top.shape == bottom.shape and top.size > 0, "expected equal non-empty masks"

    mask_colors = set(map(int, np.unique(np.concatenate([top.ravel(), bottom.ravel()]))))
    assert mask_colors <= {BLACK, RED}, "masks must contain only black and red"

    top_mask = top == RED
    bottom_mask = bottom == RED
    overlap = top_mask & bottom_mask
    diff = top_mask ^ bottom_mask

    box = _bbox(overlap)
    assert box is not None, "expected a non-empty overlap motif"
    r0, c0, r1, c1 = box
    motif = overlap[r0 : r1 + 1, c0 : c1 + 1].astype(int)

    out = np.kron(diff.astype(int), motif) * RED
    return out.astype(int)


def _sample_shared_mask(h: int, w: int) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    for _ in range(MAX_TEMPLATE_ATTEMPTS):
        box_h = random.randint(1, h)
        box_w = random.randint(1, w)
        r0 = random.randint(0, h - box_h)
        c0 = random.randint(0, w - box_w)

        allow_full = not (box_h == h and box_w == w)
        motif = _random_template(box_h, box_w, allow_full=allow_full)
        if motif is None:
            continue

        shared = np.zeros((h, w), dtype=bool)
        shared[r0 : r0 + box_h, c0 : c0 + box_w] = motif
        if int(shared.sum()) > h * w - 2:
            continue
        return shared, (r0, c0, r0 + box_h - 1, c0 + box_w - 1)
    return None


def _sample_diff_positions(
    h: int, w: int, shared: np.ndarray, shared_box: tuple[int, int, int, int]
) -> list[tuple[int, int]] | None:
    r0, c0, r1, c1 = shared_box
    all_free = [(r, c) for r in range(h) for c in range(w) if not shared[r, c]]
    outer_free = [(r, c) for r, c in all_free if not (r0 <= r <= r1 and c0 <= c <= c1)]
    pool = outer_free if len(outer_free) >= 2 else all_free
    if len(pool) < 2:
        return None

    count = random.randint(2, min(MAX_DIFF_CELLS, len(pool)))
    if random.random() < 0.5:
        connected = _connected_subset(set(pool), count)
        if connected is not None:
            return connected
    return random.sample(pool, count)


def _build_masks(h: int, w: int) -> np.ndarray | None:
    shared_data = _sample_shared_mask(h, w)
    if shared_data is None:
        return None
    shared, shared_box = shared_data

    diff_positions = _sample_diff_positions(h, w, shared, shared_box)
    if diff_positions is None:
        return None

    random.shuffle(diff_positions)
    split = random.randint(1, len(diff_positions) - 1)
    top_only = diff_positions[:split]
    bottom_only = diff_positions[split:]

    top = shared.copy()
    bottom = shared.copy()
    for r, c in top_only:
        top[r, c] = True
    for r, c in bottom_only:
        bottom[r, c] = True

    top_grid = np.where(top, RED, BLACK)
    bottom_grid = np.where(bottom, RED, BLACK)
    separator = np.full((1, w), YELLOW, dtype=int)
    inp = np.vstack([top_grid, separator, bottom_grid]).astype(int)

    try:
        out = main(inp)
    except AssertionError:
        return None

    motif_box = _bbox(shared)
    assert motif_box is not None
    mr0, mc0, mr1, mc1 = motif_box
    motif_h = mr1 - mr0 + 1
    motif_w = mc1 - mc0 + 1
    expected_shape = (h * motif_h, w * motif_w)
    if out.shape != expected_shape or max(out.shape) > 30:
        return None
    if not np.any(out == RED):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        h = random.randint(MASK_SIDE_MIN, MASK_SIDE_MAX)
        w = random.randint(MASK_SIDE_MIN, MASK_SIDE_MAX)
        grid = _build_masks(h, w)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
