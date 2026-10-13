from __future__ import annotations

import random
from collections import deque
from typing import Iterable

import numpy as np


BG_COLOR = 7
SHADOW_COLOR = 5
OBJECT_PALETTE = tuple(color for color in range(10) if color not in (BG_COLOR, SHADOW_COLOR))

GRID_H_RANGE = (8, 16)
GRID_W_RANGE = (9, 18)
OBJECT_COUNT_RANGE = (2, 5)
RANDOM_SHAPE_H_RANGE = (1, 5)
RANDOM_SHAPE_W_RANGE = (1, 6)
RANDOM_SHAPE_CELL_RANGE = (2, 10)
PLACEMENT_RETRIES = 120
GENERATION_RETRIES = 400


def _neighbors4(r: int, c: int, h: int, w: int) -> Iterable[tuple[int, int]]:
    if r > 0:
        yield (r - 1, c)
    if r + 1 < h:
        yield (r + 1, c)
    if c > 0:
        yield (r, c - 1)
    if c + 1 < w:
        yield (r, c + 1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _control_direction(grid: np.ndarray) -> tuple[int, tuple[int, int]]:
    coords = np.argwhere(grid == SHADOW_COLOR)
    assert coords.shape[0] == 1, "expected exactly one grey control pixel"
    r, c = map(int, coords[0])
    if r == 0:
        return 1, (r, c)
    if r == grid.shape[0] - 1:
        return -1, (r, c)
    raise AssertionError("control pixel must be on the top or bottom edge")


def _shadow_cells(coords: np.ndarray, direction: int) -> set[tuple[int, int]]:
    cols = np.unique(coords[:, 1])
    shadow: set[tuple[int, int]] = set()
    for c in cols:
        rows = coords[coords[:, 1] == c][:, 0]
        anchor = int(rows.max()) if direction > 0 else int(rows.min())
        shadow.add((anchor + direction, int(c)))
    return shadow


def main(grid: np.ndarray) -> np.ndarray:
    """
    Place a grey one-cell-thick horizontal projection immediately above or below each foreground object.

    The single grey edge pixel controls the global direction:
    - top edge -> project downward
    - bottom edge -> project upward

    For each monochromatic 4-connected foreground object, the shadow follows the
    object's per-column outer contour rather than its full bounding box.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h > 0 and w > 0

    direction, _control = _control_direction(g)
    out = g.copy()

    colors = [int(color) for color in np.unique(g) if int(color) not in (BG_COLOR, SHADOW_COLOR)]
    for color in colors:
        for coords in _components(g == color):
            shadow = _shadow_cells(coords, direction)
            for r, c in shadow:
                assert 0 <= r < h and 0 <= c < w, "shadow must stay inside the grid"
                out[r, c] = SHADOW_COLOR

    return out


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _random_connected_mask(rng: random.Random) -> np.ndarray:
    for _ in range(200):
        h = rng.randint(*RANDOM_SHAPE_H_RANGE)
        w = rng.randint(*RANDOM_SHAPE_W_RANGE)
        upper = min(RANDOM_SHAPE_CELL_RANGE[1], h * w)
        lower = min(RANDOM_SHAPE_CELL_RANGE[0], upper)
        if lower > upper:
            continue

        target = rng.randint(lower, upper)
        mask = np.zeros((h, w), dtype=bool)
        start = (rng.randrange(h), rng.randrange(w))
        mask[start] = True
        cells = {start}

        while len(cells) < target:
            frontier: set[tuple[int, int]] = set()
            for rr, cc in cells:
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if not mask[nr, nc]:
                        frontier.add((nr, nc))
            if not frontier:
                break
            choice = rng.choice(tuple(frontier))
            mask[choice] = True
            cells.add(choice)

        if int(mask.sum()) != target:
            continue
        cropped = _crop_mask(mask)
        if int(cropped.sum()) >= 2:
            return cropped
    raise RuntimeError("failed to sample a connected shape")


def _profiled_mask(direction: int, rng: random.Random) -> np.ndarray:
    h = rng.randint(2, 5)
    w = rng.randint(2, 6)
    heights = [rng.randint(1, h) for _ in range(w)]
    if len(set(heights)) == 1:
        idx = rng.randrange(w)
        heights[idx] = h if heights[idx] < h else 1

    mask = np.zeros((h, w), dtype=bool)
    if direction > 0:
        for c, height in enumerate(heights):
            mask[:height, c] = True
    else:
        for c, height in enumerate(heights):
            mask[h - height :, c] = True
    assert len(set(_profile_signature(mask, direction))) > 1
    return mask


def _profile_signature(mask: np.ndarray, direction: int) -> tuple[int, ...]:
    coords = np.argwhere(mask)
    signature: list[int] = []
    for c in range(mask.shape[1]):
        rows = coords[coords[:, 1] == c][:, 0]
        anchor = int(rows.max()) if direction > 0 else int(rows.min())
        signature.append(anchor)
    return tuple(signature)


def _global_cells(mask: np.ndarray, top: int, left: int) -> np.ndarray:
    coords = np.argwhere(mask)
    return coords + np.array([top, left])


def _dilate8(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _fits_clear(
    object_cells: set[tuple[int, int]],
    shadow_cells: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
) -> bool:
    future = object_cells | shadow_cells
    return not bool(future & blocked)


def _place_objects(
    h: int,
    w: int,
    direction: int,
    masks: list[np.ndarray],
    colors: list[int],
    rng: random.Random,
) -> list[tuple[int, set[tuple[int, int]]]] | None:
    order = sorted(range(len(masks)), key=lambda idx: -(int(masks[idx].sum()) + len(_profile_signature(masks[idx], direction))))
    placements: list[tuple[int, set[tuple[int, int]]]] = []
    occupied_output: set[tuple[int, int]] = set()
    blocked = _dilate8(occupied_output, h, w)

    for idx in order:
        mask = masks[idx]
        mh, mw = mask.shape
        rows = list(range(1, h - mh))
        cols = list(range(0, w - mw + 1))
        if not rows or not cols:
            return None
        rng.shuffle(rows)
        rng.shuffle(cols)

        placed = False
        attempts = 0
        for top in rows:
            for left in cols:
                attempts += 1
                if attempts > PLACEMENT_RETRIES:
                    break
                coords = _global_cells(mask, top, left)
                object_cells = {(int(r), int(c)) for r, c in coords}
                shadow_cells = _shadow_cells(coords, direction)
                if not _fits_clear(object_cells, shadow_cells, blocked):
                    continue
                future = object_cells | shadow_cells
                occupied_output |= future
                blocked = _dilate8(occupied_output, h, w)
                placements.append((colors[idx], object_cells))
                placed = True
                break
            if placed or attempts > PLACEMENT_RETRIES:
                break
        if not placed:
            return None
    return placements


def _background_dominant(grid: np.ndarray) -> bool:
    bg_count = int(np.sum(grid == BG_COLOR))
    return bg_count > grid.size // 2


def _oracle_accepts(grid: np.ndarray) -> bool:
    try:
        out = main(grid)
    except Exception:
        return False
    return isinstance(out, np.ndarray) and out.shape == grid.shape


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    direction = rng.choice((1, -1))

    object_count = rng.randint(*OBJECT_COUNT_RANGE)
    colors = rng.sample(OBJECT_PALETTE, object_count)

    featured = rng.randrange(object_count)
    masks: list[np.ndarray] = []
    for idx in range(object_count):
        if idx == featured:
            masks.append(_profiled_mask(direction, rng))
        else:
            masks.append(_random_connected_mask(rng))

    if not any(len(set(_profile_signature(mask, direction))) > 1 for mask in masks if mask.shape[1] > 1):
        return None

    placements = _place_objects(h, w, direction, masks, colors, rng)
    if placements is None:
        return None

    occupied_object_cells = set().union(*(cells for _color, cells in placements))
    projected_cells: set[tuple[int, int]] = set()
    for color, cells in placements:
        coords = np.array(sorted(cells), dtype=int)
        projected_cells |= _shadow_cells(coords, direction)

    blocked = _dilate8(occupied_object_cells | projected_cells, h, w)
    control_row = 0 if direction > 0 else h - 1
    control_slots = [(control_row, c) for c in range(w) if (control_row, c) not in blocked]
    if not control_slots:
        return None
    control = rng.choice(control_slots)

    grid = np.full((h, w), BG_COLOR, dtype=int)
    grid[control] = SHADOW_COLOR
    for color, cells in placements:
        for r, c in cells:
            grid[r, c] = color

    if np.count_nonzero(grid == SHADOW_COLOR) != 1:
        return None
    if not _background_dominant(grid):
        return None
    if not _oracle_accepts(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
