import numpy as np

from arc_utils import bbox, components_mask, crop_bbox, get_bg, paste
import random


def main(grid: np.ndarray) -> np.ndarray:
    """Crop the non-background rectangle and mirror it left-to-right.

    Input structure
    A mostly background grid contains one filled rectangular foreground region,
    made from two non-background colors.

    Rule
    Find the tight bounding box of all non-background pixels, crop that region,
    and return the crop with its columns reversed.

    Key insight
    The outer placement in the input is irrelevant; the output is the foreground
    rectangle viewed in a horizontal mirror.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    assert np.count_nonzero(g == bg) > g.size // 2, "background must be dominant"

    fg = g != bg
    b = bbox(fg)
    assert b is not None, "foreground rectangle must exist"

    crop = crop_bbox(g, b)
    assert np.all(crop != bg), "foreground bbox must be filled"
    return crop[:, ::-1].copy()

# --------------- generator (o5x) ---------------

# Canvas and placed rectangle size bounds.
_GEN_CANVAS_H_RANGE, _GEN_CANVAS_W_RANGE = (10, 17), (15, 17)
_GEN_RECT_H_RANGE, _GEN_RECT_W_RANGE, _GEN_PLACEMENT_BORDER = (4, 7), (5, 8), 1

# Accent blob fill percent, component counts, and component sizes.
_GEN_ACCENT_PERCENT_RANGE, _GEN_PERCENT_DENOMINATOR = (16, 25), 100
_GEN_BLOB_COUNT_RANGE, _GEN_BLOB_SIZE_RANGE = (1, 4), (1, 5)
_GEN_DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

# Colors and retry limits.
_GEN_BG_COLOR, _GEN_MAX_ATTEMPTS, _GEN_COLOR_IDS = 0, 300, range(1, 10)


def _gen_blob_sizes(total):
    min_count, max_count = _GEN_BLOB_COUNT_RANGE
    min_size, max_size = _GEN_BLOB_SIZE_RANGE
    valid_counts = [k for k in range(min_count, max_count + 1) if min_size * k <= total <= max_size * k]
    k = random.choice(valid_counts)
    sizes = []
    remaining = total
    for slots in range(k, 0, -1):
        lo = max(min_size, remaining - max_size * (slots - 1))
        hi = min(max_size, remaining - min_size * (slots - 1))
        size = random.randint(lo, hi)
        sizes.append(size)
        remaining -= size
    random.shuffle(sizes)
    return sizes


def _gen_prior_block(mask, reserved):
    blocked = reserved.copy()
    blocked |= mask
    blocked[1:, :] |= mask[:-1, :]
    blocked[:-1, :] |= mask[1:, :]
    blocked[:, 1:] |= mask[:, :-1]
    blocked[:, :-1] |= mask[:, 1:]
    return blocked


def _gen_grow_blob(available, size, seed):
    h, w = available.shape
    chosen = np.zeros_like(available, dtype=bool)
    cells = [seed]
    chosen[seed] = True
    while len(cells) < size:
        frontier = []
        for r, c in cells:
            for dr, dc in _GEN_DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and available[nr, nc] and not chosen[nr, nc]:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        cell = random.choice(frontier)
        chosen[cell] = True
        cells.append(cell)
    return cells


def _gen_seed_asymmetric_blob(mask, size):
    rh, rw = mask.shape
    first_seeds = [(r, c) for r in range(rh) for c in range(rw) if c != rw - 1 - c]
    random.shuffle(first_seeds)
    for seed in first_seeds:
        reserved = np.zeros_like(mask, dtype=bool)
        reserved[seed[0], rw - 1 - seed[1]] = True
        cells = _gen_grow_blob(~reserved, size, seed)
        if cells is None:
            continue
        mask[tuple(zip(*cells))] = True
        return reserved
    return None


def _gen_add_separated_blob(mask, reserved, size):
    available = ~_gen_prior_block(mask, reserved)
    regions = [coords for coords in components_mask(available, connectivity=4) if len(coords) >= size]
    if not regions:
        return False
    region = random.choice(regions)
    seed = tuple(int(x) for x in region[random.randrange(len(region))])
    cells = _gen_grow_blob(available, size, seed)
    if cells is None:
        return False
    mask[tuple(zip(*cells))] = True
    return True


def _gen_accent_mask(rh, rw, total):
    sizes = _gen_blob_sizes(total)
    mask = np.zeros((rh, rw), dtype=bool)
    reserved = _gen_seed_asymmetric_blob(mask, sizes[0])
    if reserved is None:
        return None
    for size in sizes[1:]:
        if not _gen_add_separated_blob(mask, reserved, size):
            return None
    if int(mask.sum()) != total or np.array_equal(mask, mask[:, ::-1]):
        return None
    return mask


def _gen_output_rect(rh, rw):
    area = rh * rw
    lo_pct, hi_pct = _GEN_ACCENT_PERCENT_RANGE
    lo = (lo_pct * area + _GEN_PERCENT_DENOMINATOR - 1) // _GEN_PERCENT_DENOMINATOR
    hi = (hi_pct * area) // _GEN_PERCENT_DENOMINATOR
    accent_count = random.randint(lo, hi)
    dominant, accent = random.sample(_GEN_COLOR_IDS, 2)
    rect = np.full((rh, rw), dominant, dtype=int)
    accent_mask = _gen_accent_mask(rh, rw, accent_count)
    if accent_mask is None:
        return None
    rect[accent_mask] = accent
    return rect


def _gen_place_input_rect(h, w, rect):
    rh, rw = rect.shape
    grid = np.full((h, w), _GEN_BG_COLOR, dtype=int)
    r0 = random.randint(_GEN_PLACEMENT_BORDER, h - rh - _GEN_PLACEMENT_BORDER)
    c0 = random.randint(_GEN_PLACEMENT_BORDER, w - rw - _GEN_PLACEMENT_BORDER)
    paste(grid, rect[:, ::-1].copy(), r0, c0)
    return grid


def _gen_attempt():
    h, w = random.randint(*_GEN_CANVAS_H_RANGE), random.randint(*_GEN_CANVAS_W_RANGE)
    rh, rw = random.randint(*_GEN_RECT_H_RANGE), random.randint(*_GEN_RECT_W_RANGE)
    rect = _gen_output_rect(rh, rw)
    if rect is None:
        return None
    grid = _gen_place_input_rect(h, w, rect)
    try:
        if np.array_equal(main(grid), rect):
            return grid
    except Exception:
        return None
    return None


def generate_input():
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 7468f01a input")
