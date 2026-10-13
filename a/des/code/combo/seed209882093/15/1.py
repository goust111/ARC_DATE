from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np


BG = 0
BORDER = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Generator configuration.
GRID_H_RANGE = (18, 28)
GRID_W_RANGE = (18, 28)
REGION_COUNT_CHOICES = (1, 2, 2, 3, 3, 4)
REGION_TRIES = 80
STRIPE_TRIES = 120
GEN_RETRIES = 400


def neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _touches_edge(cells: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in cells)


def _classify_stripe(comp: list[tuple[int, int]]) -> tuple[str, int] | None:
    if len(comp) < 2:
        return None
    rows = sorted({r for r, _ in comp})
    cols = sorted({c for _, c in comp})
    if len(rows) == 1:
        ordered = sorted(c for _, c in comp)
        if ordered == list(range(ordered[0], ordered[0] + len(comp))):
            return ("h", len(comp))
    if len(cols) == 1:
        ordered = sorted(r for r, _ in comp)
        if ordered == list(range(ordered[0], ordered[0] + len(comp))):
            return ("v", len(comp))
    return None


def _stripe_map(grid: np.ndarray) -> dict[int, tuple[str, int]]:
    stripes: dict[int, tuple[str, int]] = {}
    for color in map(int, np.unique(grid)):
        if color in (BG, BORDER):
            continue
        stripe = None
        for comp in _components_mask(grid == color):
            info = _classify_stripe(comp)
            if info is not None:
                stripe = info
                break
        if stripe is not None:
            stripes[color] = stripe
    return stripes


def _fill_with_stripe(
    out: np.ndarray,
    cells: list[tuple[int, int]],
    color: int,
    orientation: str,
    length: int,
) -> None:
    # The stripe is monochrome, so respecting its axis/period still colors every
    # reachable interior cell with the same visible color.
    if orientation == "h":
        by_row: dict[int, list[int]] = defaultdict(list)
        for r, c in cells:
            by_row[r].append(c)
        for r, cols in by_row.items():
            ordered = sorted(cols)
            for i in range(0, len(ordered), length):
                for c in ordered[i : i + length]:
                    out[r, c] = color
    else:
        by_col: dict[int, list[int]] = defaultdict(list)
        for r, c in cells:
            by_col[c].append(r)
        for c, rows in by_col.items():
            ordered = sorted(rows)
            for i in range(0, len(ordered), length):
                for r in ordered[i : i + length]:
                    out[r, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each interior seed color to the unique same-color outside stripe and
    use that stripe's axis/length as the tiling rule for the enclosed region.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    stripes = _stripe_map(g)

    # Every non-grey component that does not touch the grid edge is one hollow
    # region interior together with its single seed pixel.
    for comp in _components_mask(g != BORDER):
        if _touches_edge(comp, h, w):
            continue
        colors = {int(g[r, c]) for r, c in comp if int(g[r, c]) != BG}
        if len(colors) != 1:
            continue
        seed_color = next(iter(colors))
        stripe = stripes.get(seed_color)
        if stripe is None:
            continue
        orientation, length = stripe
        _fill_with_stripe(out, comp, seed_color, orientation, length)
    return out


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _random_blob_mask(rng: random.Random) -> np.ndarray:
    for _ in range(30):
        h = rng.randint(5, 8)
        w = rng.randint(5, 8)
        mask = np.zeros((h, w), dtype=bool)
        rh = rng.randint(2, min(4, h))
        rw = rng.randint(2, min(4, w))
        top = rng.randint(0, h - rh)
        left = rng.randint(0, w - rw)
        mask[top : top + rh, left : left + rw] = True
        base_area = int(mask.sum())
        target = rng.randint(base_area + 2, min(h * w, base_area + 8))
        while int(mask.sum()) < target:
            frontier: dict[tuple[int, int], int] = {}
            for r, c in np.argwhere(mask):
                for nr, nc in neighbors4(int(r), int(c)):
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                        frontier[(nr, nc)] = frontier.get((nr, nc), 0) + 1
            if not frontier:
                break
            weighted: list[tuple[int, int]] = []
            for cell, degree in frontier.items():
                weighted.extend([cell] * (degree * degree))
            rr, cc = rng.choice(weighted)
            mask[rr, cc] = True
        trimmed = _trim_mask(mask)
        if trimmed.shape[0] >= 3 and trimmed.shape[1] >= 3 and int(trimmed.sum()) >= 8:
            return trimmed
    raise RuntimeError("failed to build blob mask")


def _random_region_mask(rng: random.Random) -> np.ndarray:
    kind = rng.choice(["rect", "notch", "blob", "blob"])
    if kind == "rect":
        return np.ones((rng.randint(3, 6), rng.randint(3, 6)), dtype=bool)
    if kind == "notch":
        h = rng.randint(4, 7)
        w = rng.randint(4, 7)
        mask = np.ones((h, w), dtype=bool)
        notch_h = rng.randint(1, h - 2)
        notch_w = rng.randint(1, w - 2)
        corner = rng.choice(["tl", "tr", "bl", "br"])
        if corner == "tl":
            mask[:notch_h, :notch_w] = False
        elif corner == "tr":
            mask[:notch_h, w - notch_w :] = False
        elif corner == "bl":
            mask[h - notch_h :, :notch_w] = False
        else:
            mask[h - notch_h :, w - notch_w :] = False
        return mask
    return _random_blob_mask(rng)


def _shell(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    border: set[tuple[int, int]] = set()
    for r, c in cells:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                border.add((nr, nc))
    return border


def _mark_reserved(reserved: np.ndarray, cells: set[tuple[int, int]]) -> None:
    h, w = reserved.shape
    for r, c in cells:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if 0 <= nr < h and 0 <= nc < w:
                    reserved[nr, nc] = True


def _pick_seed(mask: np.ndarray, rng: random.Random) -> tuple[int, int]:
    best_cells: list[tuple[int, int]] = []
    best_degree = -1
    for r, c in np.argwhere(mask):
        degree = 0
        for nr, nc in neighbors4(int(r), int(c)):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                degree += 1
        if degree > best_degree:
            best_degree = degree
            best_cells = [(int(r), int(c))]
        elif degree == best_degree:
            best_cells.append((int(r), int(c)))
    return rng.choice(best_cells)


def _place_region(
    grid: np.ndarray,
    reserved: np.ndarray,
    mask: np.ndarray,
    color: int,
    rng: random.Random,
) -> bool:
    h, w = grid.shape
    mh, mw = mask.shape
    if mh + 2 >= h or mw + 2 >= w:
        return False

    seed_r, seed_c = _pick_seed(mask, rng)
    local_cells = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    for _ in range(REGION_TRIES):
        top = rng.randint(1, h - mh - 1)
        left = rng.randint(1, w - mw - 1)
        interior = {(top + r, left + c) for r, c in local_cells}
        border = _shell(interior, h, w)
        footprint = interior | border
        if any(reserved[r, c] for r, c in footprint):
            continue
        for r, c in border:
            grid[r, c] = BORDER
        grid[top + seed_r, left + seed_c] = color
        _mark_reserved(reserved, footprint)
        return True
    return False


def _place_stripe(
    grid: np.ndarray,
    reserved: np.ndarray,
    color: int,
    orientation: str,
    length: int,
    rng: random.Random,
) -> bool:
    h, w = grid.shape
    for _ in range(STRIPE_TRIES):
        if orientation == "h":
            r = rng.randint(1, h - 2)
            c = rng.randint(1, w - length - 1)
            cells = {(r, c + i) for i in range(length)}
        else:
            r = rng.randint(1, h - length - 1)
            c = rng.randint(1, w - 2)
            cells = {(r + i, c) for i in range(length)}
        if any(reserved[rr, cc] for rr, cc in cells):
            continue
        for rr, cc in cells:
            grid[rr, cc] = color
        _mark_reserved(reserved, cells)
        return True
    return False


def _validate_instance(grid: np.ndarray, expected_regions: int, colors: set[int]) -> bool:
    present = {int(v) for v in np.unique(grid)} - {BG, BORDER}
    if present != colors:
        return False

    stripes = _stripe_map(grid)
    if set(stripes) != colors:
        return False

    h, w = grid.shape
    enclosed = [comp for comp in _components_mask(grid != BORDER) if not _touches_edge(comp, h, w)]
    if len(enclosed) != expected_regions:
        return False
    for comp in enclosed:
        seen_colors = {int(grid[r, c]) for r, c in comp if int(grid[r, c]) != BG}
        if len(seen_colors) != 1:
            return False

    out = main(grid)
    return bool(np.any(out != grid))


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    n_regions = rng.choice(REGION_COUNT_CHOICES)
    colors = rng.sample(SEED_COLORS, n_regions)

    grid = np.full((h, w), BG, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    for color in colors:
        placed = False
        for _ in range(REGION_TRIES):
            mask = _random_region_mask(rng)
            if _place_region(grid, reserved, mask, color, rng):
                placed = True
                break
        if not placed:
            return None

    orientations = [rng.choice(["h", "v"]) for _ in colors]
    if n_regions >= 2 and len(set(orientations)) == 1:
        flip_idx = rng.randrange(n_regions)
        orientations[flip_idx] = "v" if orientations[flip_idx] == "h" else "h"

    lengths = [rng.randint(2, 4) for _ in colors]
    order = list(range(n_regions))
    rng.shuffle(order)
    for idx in order:
        if not _place_stripe(grid, reserved, colors[idx], orientations[idx], lengths[idx], rng):
            return None

    if not _validate_instance(grid, n_regions, set(colors)):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted retries")
