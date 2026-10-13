from __future__ import annotations

import numpy as np


PALETTE_ORDER = (5, 2, 8, 9, 6)  # Grey, Red, Teal, Maroon, Magenta
BACKGROUND_POOL = (0, 1, 3, 4, 7)

MIN_TOTAL_OBJECTS = 5
MAX_TOTAL_OBJECTS = 14
MAX_COLOR_COUNT = 6

MIN_GRID_SIDE = 14
MAX_GRID_SIDE = 28
PLACEMENT_TRIES = 120
COUNT_SAMPLING_TRIES = 120
GENERATION_TRIES = 400


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []

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
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr = cr + dr
                        nc = cc + dc
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
            components.append(comp)

    return components


def _object_counts(grid: np.ndarray) -> list[int]:
    return [len(_components_mask(grid == color)) for color in PALETTE_ORDER]


def _render_histogram(bg: int, counts: list[int]) -> np.ndarray:
    height = max(counts)
    out = np.full((height, len(PALETTE_ORDER)), bg, dtype=int)
    for col, (color, count) in enumerate(zip(PALETTE_ORDER, counts)):
        if count > 0:
            out[height - count :, col] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Count 8-connected objects per palette color and render bottom-aligned bars."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _dominant_color(g)
    assert bg not in PALETTE_ORDER

    counts = _object_counts(g)
    assert max(counts) > 0
    return _render_histogram(bg, counts)


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    return mask[rows][:, cols]


def _grown_blob(rng: np.random.Generator) -> np.ndarray:
    h = int(rng.integers(2, 5))
    w = int(rng.integers(2, 5))
    mask = np.zeros((h, w), dtype=bool)
    start = (int(rng.integers(h)), int(rng.integers(w)))
    mask[start] = True
    frontier = [start]
    target = int(rng.integers(3, min(7, h * w) + 1))

    while int(mask.sum()) < target:
        seed_r, seed_c = frontier[int(rng.integers(len(frontier)))]
        candidates = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr = seed_r + dr
                nc = seed_c + dc
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    candidates.append((nr, nc))
        if not candidates:
            frontier = list(zip(*np.where(mask), strict=False))
            continue
        nr, nc = candidates[int(rng.integers(len(candidates)))]
        mask[nr, nc] = True
        frontier.append((nr, nc))

    return _trim_mask(mask)


def _random_blob(rng: np.random.Generator) -> np.ndarray:
    kind = int(rng.integers(8))

    if kind == 0:
        return np.array([[True]], dtype=bool)

    if kind == 1:
        length = int(rng.integers(2, 5))
        return np.ones((1, length), dtype=bool)

    if kind == 2:
        length = int(rng.integers(2, 5))
        return np.ones((length, 1), dtype=bool)

    if kind == 3:
        h = int(rng.integers(2, 4))
        w = int(rng.integers(2, 4))
        return np.ones((h, w), dtype=bool)

    if kind == 4:
        h = int(rng.integers(2, 5))
        w = int(rng.integers(2, 5))
        mask = np.zeros((h, w), dtype=bool)
        orientation = int(rng.integers(4))
        if orientation == 0:
            mask[:, 0] = True
            mask[-1, :] = True
        elif orientation == 1:
            mask[:, -1] = True
            mask[-1, :] = True
        elif orientation == 2:
            mask[:, 0] = True
            mask[0, :] = True
        else:
            mask[:, -1] = True
            mask[0, :] = True
        return mask

    if kind == 5:
        mask = np.zeros((3, 3), dtype=bool)
        mask[1, :] = True
        mask[:, 1] = True
        return mask

    if kind == 6:
        length = int(rng.integers(2, 5))
        mask = np.zeros((length, length), dtype=bool)
        if rng.random() < 0.5:
            for i in range(length):
                mask[i, i] = True
        else:
            for i in range(length):
                mask[i, length - 1 - i] = True
        return mask

    return _grown_blob(rng)


def _shape_coords(mask: np.ndarray, r0: int, c0: int) -> list[tuple[int, int]]:
    points = np.argwhere(mask)
    return [(int(r0 + r), int(c0 + c)) for r, c in points]


def _reserve(blocked: np.ndarray, coords: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    blocked[nr, nc] = True


def _place_object(
    grid: np.ndarray,
    blocked: np.ndarray,
    color: int,
    mask: np.ndarray,
    rng: np.random.Generator,
) -> bool:
    h, w = grid.shape
    mh, mw = mask.shape
    if mh > h or mw > w:
        return False

    for _ in range(PLACEMENT_TRIES):
        r0 = int(rng.integers(0, h - mh + 1))
        c0 = int(rng.integers(0, w - mw + 1))
        coords = _shape_coords(mask, r0, c0)
        if any(blocked[r, c] for r, c in coords):
            continue
        for r, c in coords:
            grid[r, c] = color
        _reserve(blocked, coords)
        return True

    return False


def _sample_counts(rng: np.random.Generator) -> list[int] | None:
    n_colors = len(PALETTE_ORDER)
    for _ in range(COUNT_SAMPLING_TRIES):
        used_count = int(rng.integers(2, n_colors + 1))
        used = rng.choice(n_colors, size=used_count, replace=False)
        values = rng.integers(1, MAX_COLOR_COUNT + 1, size=used_count)
        if int(values.max()) < 2:
            values[int(rng.integers(used_count))] = 2
        total = int(values.sum())
        if not (MIN_TOTAL_OBJECTS <= total <= MAX_TOTAL_OBJECTS):
            continue

        counts = [0] * n_colors
        for idx, value in zip(used, values, strict=False):
            counts[int(idx)] = int(value)
        return counts

    return None


def _sample_grid_shape(counts: list[int], rng: np.random.Generator) -> tuple[int, int]:
    total = sum(counts)
    tallest = max(counts)
    base = max(MIN_GRID_SIDE, 11 + total // 2 + tallest // 2)
    low = min(MAX_GRID_SIDE, base)
    high = min(MAX_GRID_SIDE, base + 4)
    h = int(rng.integers(low, high + 1))
    w = int(rng.integers(low, high + 1))
    return h, w


def _unique_dominant_background(grid: np.ndarray, bg: int) -> bool:
    values, counts = np.unique(grid, return_counts=True)
    order = np.argsort(counts)[::-1]
    if int(values[order[0]]) != bg:
        return False
    if len(order) > 1 and int(counts[order[0]]) == int(counts[order[1]]):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    counts = _sample_counts(rng)
    if counts is None:
        return None

    bg = int(rng.choice(BACKGROUND_POOL))
    h, w = _sample_grid_shape(counts, rng)
    grid = np.full((h, w), bg, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    colors_to_place = []
    for color, count in zip(PALETTE_ORDER, counts, strict=False):
        colors_to_place.extend([color] * count)
    rng.shuffle(colors_to_place)

    for color in colors_to_place:
        if not _place_object(grid, blocked, int(color), _random_blob(rng), rng):
            return None

    if not _unique_dominant_background(grid, bg):
        return None
    if _object_counts(grid) != counts:
        return None
    if not np.array_equal(main(grid), _render_histogram(bg, counts)):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_TRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GENERATION_TRIES} attempts")
