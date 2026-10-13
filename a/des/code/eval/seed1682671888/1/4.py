import random
from collections import deque

import numpy as np


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
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
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _count_holes(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            touches_border = r == 0 or r == h - 1 or c == 0 or c == w - 1
            while q:
                cr, cc = q.popleft()
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        if nr == 0 or nr == h - 1 or nc == 0 or nc == w - 1:
                            touches_border = True
                        q.append((nr, nc))
            if not touches_border:
                holes += 1
    return holes


def _find_palette(grid: np.ndarray, bg: int) -> tuple[list[int], np.ndarray]:
    h, w = grid.shape
    row = grid[h - 1]
    candidates: list[tuple[int, int, int, list[int]]] = []
    c = 0
    while c < w:
        if int(row[c]) == bg:
            c += 1
            continue
        start = c
        while c < w and int(row[c]) != bg:
            c += 1
        end = c
        runs: list[tuple[int, int]] = []
        i = start
        while i < end:
            color = int(row[i])
            j = i + 1
            while j < end and int(row[j]) == color:
                j += 1
            runs.append((color, j - i))
            i = j
        lengths = {length for _, length in runs}
        colors = [color for color, _ in runs]
        if 3 <= len(runs) <= 5 and len(lengths) == 1 and len(set(colors)) == len(colors):
            candidates.append((len(runs), start, end, colors))
    assert candidates, "palette not found on bottom row"
    candidates.sort(key=lambda item: (item[0], item[2] - item[1]), reverse=True)
    _count, start, end, colors = candidates[0]
    mask = np.zeros_like(grid, dtype=bool)
    mask[h - 1, start:end] = True
    return colors, mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each object using the bottom palette, indexed by its hole count.

    The bottom row contains a contiguous palette strip with equal-length color
    runs. Every other non-background component is a monochrome object. The
    object's number of enclosed background regions selects its replacement color
    from the palette.
    """
    g = np.asarray(grid, dtype=int)
    bg = 0
    palette, palette_mask = _find_palette(g, bg)

    object_mask = (g != bg) & (~palette_mask)
    comps = _components(object_mask)
    out = g.copy()
    for comp in comps:
        r0, c0, r1, c1 = _bbox(comp)
        local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in comp:
            local[r - r0, c - c0] = True
        hole_count = _count_holes(local)
        assert 0 <= hole_count < len(palette), "hole count must index the palette"
        new_color = int(palette[hole_count])
        for r, c in comp:
            out[r, c] = new_color
    return out


GEN_RETRIES = 400
BG = 0
PALETTE_COUNT_RANGE = (3, 5)
SEGMENT_LENGTHS = (2, 3)
COLOR_POOL = tuple(range(1, 10))


def _sample_distinct(rng: random.Random, count: int) -> list[int]:
    return rng.sample(list(COLOR_POOL), count)


def _make_shape(rng: random.Random) -> tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]]]:
    hole_size = rng.choice((1, 2))
    border = rng.choice((1, 2))
    row_gap = rng.randint(1, 3)
    col_gap = rng.randint(1, 4)
    h = border * 2 + hole_size * 2 + row_gap
    w = border * 2 + hole_size * 2 + col_gap
    slots = [
        (border, border),
        (border, border + hole_size + col_gap),
        (border + hole_size + row_gap, border),
        (border + hole_size + row_gap, border + hole_size + col_gap),
    ]
    return (h, w), (hole_size, hole_size), slots


def _object_mask(shape_hw: tuple[int, int], hole_hw: tuple[int, int], slots: list[tuple[int, int]], count: int, order: list[int]) -> np.ndarray:
    h, w = shape_hw
    hh, hw = hole_hw
    mask = np.ones((h, w), dtype=bool)
    for idx in order[:count]:
        r0, c0 = slots[idx]
        mask[r0 : r0 + hh, c0 : c0 + hw] = False
    return mask


def _try_place_objects(
    rng: random.Random,
    grid: np.ndarray,
    object_masks: list[np.ndarray],
    object_colors: list[int],
) -> bool:
    h, w = grid.shape
    occupied = np.zeros((h, w), dtype=bool)
    for mask, color in zip(object_masks, object_colors):
        oh, ow = mask.shape
        placed = False
        for _ in range(200):
            r0 = rng.randint(0, h - oh - 1)
            c0 = rng.randint(0, w - ow)
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(h, r0 + oh + 1)
            cc1 = min(w, c0 + ow + 1)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            grid[r0 : r0 + oh, c0 : c0 + ow][mask] = color
            occupied[r0 : r0 + oh, c0 : c0 + ow] |= mask
            placed = True
            break
        if not placed:
            return False
    return True


def _validate_generated(grid: np.ndarray, palette: list[int], object_count: int) -> bool:
    found_palette, palette_mask = _find_palette(grid, BG)
    if found_palette != palette:
        return False
    comps = _components((grid != BG) & (~palette_mask))
    if len(comps) != object_count:
        return False
    hole_counts = []
    for comp in comps:
        r0, c0, r1, c1 = _bbox(comp)
        local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in comp:
            local[r - r0, c - c0] = True
        hole_counts.append(_count_holes(local))
    return sorted(hole_counts) == list(range(object_count))


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    palette_count = rng.randint(*PALETTE_COUNT_RANGE)
    segment_len = rng.choice(SEGMENT_LENGTHS)
    palette = _sample_distinct(rng, palette_count)
    object_colors = _sample_distinct(rng, palette_count)
    if all(object_colors[i] == palette[i] for i in range(palette_count)):
        return None

    shape_hw, hole_hw, slots = _make_shape(rng)
    slot_order = list(range(4))
    rng.shuffle(slot_order)
    object_masks = [_object_mask(shape_hw, hole_hw, slots, count, slot_order) for count in range(palette_count)]

    max_oh = max(mask.shape[0] for mask in object_masks)
    max_ow = max(mask.shape[1] for mask in object_masks)
    palette_width = palette_count * segment_len

    min_h = max(max_oh + 5, 12)
    max_h = max(min_h, max_oh * 2 + 10)
    min_w = max(max_ow + 4, palette_width + 4, 14)
    max_w = max(min_w, max_ow * palette_count + 10)
    grid_h = rng.randint(min_h, min(max_h, 24))
    grid_w = rng.randint(min_w, min(max_w, 30))

    grid = np.zeros((grid_h, grid_w), dtype=int)

    if not _try_place_objects(rng, grid[:-1], object_masks, object_colors):
        return None

    palette_start = rng.randint(0, grid_w - palette_width)
    for i, color in enumerate(palette):
        c0 = palette_start + i * segment_len
        grid[-1, c0 : c0 + segment_len] = color

    if np.count_nonzero(grid == BG) <= grid.size // 2:
        return None
    if not _validate_generated(grid, palette, palette_count):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to create a valid sample")
