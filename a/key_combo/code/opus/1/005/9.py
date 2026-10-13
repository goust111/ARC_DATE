from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
MASK = 9

RNG = np.random.default_rng()

GEN_ATTEMPTS = 800
HEIGHT_CHOICES = (13, 15, 17, 19)
WIDTH_CHOICES = (13, 15, 17, 19)
MASK_HEIGHT_RANGE = (3, 5)
MASK_WIDTH_RANGE = (4, 6)
ROW_PART_CHOICES = (2, 3)
COL_PART_CHOICES = (2, 3)
WANT_ZERO_IN_CROP_PROB = 0.55


def _solid_mask_bbox(g: np.ndarray) -> tuple[int, int, int, int]:
    mask = g == MASK
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected at least one mask cell"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    area = int((r1 - r0 + 1) * (c1 - c0 + 1))
    assert int(mask.sum()) == area, "mask color must form one solid rectangle"
    assert bool(np.all(mask[r0 : r1 + 1, c0 : c1 + 1])), "mask rectangle must be filled"
    return int(r0), int(c0), int(r1), int(c1)


def _best_row_axis_sum(g: np.ndarray) -> int:
    h, _w = g.shape
    best_score = -1
    best_sums: list[int] = []
    for axis_sum in range(2 * h - 1):
        ok = True
        score = 0
        for r in range(h):
            rr = axis_sum - r
            if rr < 0 or rr >= h or r > rr:
                continue
            left = g[r]
            right = g[rr]
            known = (left != MASK) & (right != MASK)
            if np.any(left[known] != right[known]):
                ok = False
                break
            weight = 1 if r == rr else 2
            score += weight * int(known.sum())
        if not ok:
            continue
        if score > best_score:
            best_score = score
            best_sums = [axis_sum]
        elif score == best_score:
            best_sums.append(axis_sum)
    assert best_score >= 0 and len(best_sums) == 1, "expected a unique horizontal symmetry axis"
    return int(best_sums[0])


def _best_col_axis_sum(g: np.ndarray) -> int:
    h, w = g.shape
    best_score = -1
    best_sums: list[int] = []
    for axis_sum in range(2 * w - 1):
        ok = True
        score = 0
        for c in range(w):
            cc = axis_sum - c
            if cc < 0 or cc >= w or c > cc:
                continue
            top = g[:, c]
            bot = g[:, cc]
            known = (top != MASK) & (bot != MASK)
            if np.any(top[known] != bot[known]):
                ok = False
                break
            weight = 1 if c == cc else 2
            score += weight * int(known.sum())
        if not ok:
            continue
        if score > best_score:
            best_score = score
            best_sums = [axis_sum]
        elif score == best_score:
            best_sums.append(axis_sum)
    assert best_score >= 0 and len(best_sums) == 1, "expected a unique vertical symmetry axis"
    return int(best_sums[0])


def _restore_mask(g: np.ndarray, row_axis_sum: int, col_axis_sum: int) -> np.ndarray:
    h, w = g.shape
    out = g.copy()
    for r, c in np.argwhere(g == MASK):
        r = int(r)
        c = int(c)
        candidates: set[int] = set()
        rr = row_axis_sum - r
        cc = col_axis_sum - c
        if 0 <= rr < h:
            v = int(g[rr, c])
            if v != MASK:
                candidates.add(v)
        if 0 <= cc < w:
            v = int(g[r, cc])
            if v != MASK:
                candidates.add(v)
        if 0 <= rr < h and 0 <= cc < w:
            v = int(g[rr, cc])
            if v != MASK:
                candidates.add(v)
        assert len(candidates) == 1, "each mask cell must have one recoverable reflected color"
        out[r, c] = candidates.pop()
    return out


def _swap_patch_colors(patch: np.ndarray) -> np.ndarray:
    out = patch.copy()
    blue = patch == BLUE
    red = patch == RED
    out[blue] = RED
    out[red] = BLUE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore the masked quarter-symmetric pattern, crop the mask rectangle, and swap the two pattern colors inside that crop.

    Input structure:
    - A black background contains a red/blue pattern that is symmetric under both horizontal and vertical reflection when the mask is ignored.
    - Exactly one solid maroon rectangle marks the occluded area and lies fully inside a single quadrant relative to the two symmetry axes.
    - The masked region may cover red, blue, or black cells from the hidden symmetric pattern.

    Rule:
    - Find the unique solid maroon rectangle and record its bounding box.
    - Infer the horizontal and vertical reflection axes from the visible non-mask cells.
    - Replace every mask cell by the unique visible color in its horizontal, vertical, or double-reflection counterpart set.
    - Crop the restored grid to the mask bounding box, then swap blue and red inside that cropped patch while leaving black unchanged.

    Key insight:
    - The maroon rectangle serves two roles at once: it identifies the missing symmetric region to reconstruct and also defines the exact crop that must be color-swapped.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    r0, c0, r1, c1 = _solid_mask_bbox(g)
    row_axis_sum = _best_row_axis_sum(g)
    col_axis_sum = _best_col_axis_sum(g)
    restored = _restore_mask(g, row_axis_sum, col_axis_sum)
    patch = restored[r0 : r1 + 1, c0 : c1 + 1]
    return _swap_patch_colors(patch)


def _random_composition(total: int, parts: int, min_size: int) -> np.ndarray | None:
    extra = int(total) - int(parts) * int(min_size)
    if extra < 0:
        return None
    return np.random.multinomial(extra, [1.0 / parts] * parts) + int(min_size)


def _bands(total: int, choices: tuple[int, ...], min_size: int) -> list[int] | None:
    options: list[np.ndarray] = []
    for parts in choices:
        sizes = _random_composition(total, parts, min_size)
        if sizes is not None:
            options.append(sizes)
    if not options:
        return None
    picked = options[int(RNG.integers(len(options)))]
    return [int(v) for v in picked]


def _tile_boundaries(sizes: list[int]) -> list[tuple[int, int]]:
    bounds = []
    start = 0
    for size in sizes:
        bounds.append((start, start + int(size)))
        start += int(size)
    return bounds


def _build_quarter(qh: int, qw: int) -> np.ndarray | None:
    row_sizes = _bands(qh, ROW_PART_CHOICES, 2)
    col_sizes = _bands(qw, COL_PART_CHOICES, 2)
    if row_sizes is None or col_sizes is None:
        return None

    tile_rows = len(row_sizes)
    tile_cols = len(col_sizes)
    want_zero = RNG.random() < WANT_ZERO_IN_CROP_PROB
    palette = np.array([BLACK, BLUE, RED], dtype=int)
    probs = np.array([0.36, 0.32, 0.32])

    for _ in range(80):
        tile_colors = RNG.choice(palette, size=(tile_rows, tile_cols), p=probs)
        colors = set(map(int, tile_colors.ravel()))
        if BLUE not in colors or RED not in colors:
            continue
        if want_zero and BLACK not in colors:
            continue
        if len(colors) < 2:
            continue

        quarter = np.zeros((qh, qw), dtype=int)
        for (r0, r1), row_idx in zip(_tile_boundaries(row_sizes), range(tile_rows)):
            for (c0, c1), col_idx in zip(_tile_boundaries(col_sizes), range(tile_cols)):
                quarter[r0:r1, c0:c1] = int(tile_colors[row_idx, col_idx])
        return quarter
    return None


def _mirrored_sequence(length: int, zero_prob: float) -> np.ndarray:
    half = length // 2
    seq = np.zeros(length, dtype=int)
    weights = np.array([zero_prob, (1.0 - zero_prob) / 2.0, (1.0 - zero_prob) / 2.0])
    base = RNG.choice(np.array([BLACK, BLUE, RED], dtype=int), size=half + 1, p=weights)
    seq[: half + 1] = base
    seq[half + 1 :] = base[:-1][::-1]
    return seq


def _build_clean_grid(height: int, width: int) -> np.ndarray | None:
    axis_r = height // 2
    axis_c = width // 2
    quarter = _build_quarter(axis_r, axis_c)
    if quarter is None:
        return None

    clean = np.zeros((height, width), dtype=int)
    clean[:axis_r, :axis_c] = quarter
    clean[:axis_r, axis_c + 1 :] = quarter[:, ::-1]
    clean[axis_r + 1 :, :axis_c] = quarter[::-1, :]
    clean[axis_r + 1 :, axis_c + 1 :] = quarter[::-1, ::-1]

    row_seq = _mirrored_sequence(width, zero_prob=0.48)
    col_seq = _mirrored_sequence(height, zero_prob=0.48)
    center_color = int(RNG.choice(np.array([BLACK, BLUE, RED], dtype=int), p=[0.30, 0.35, 0.35]))
    row_seq[axis_c] = center_color
    col_seq[axis_r] = center_color
    clean[axis_r, :] = row_seq
    clean[:, axis_c] = col_seq

    if BLUE not in clean or RED not in clean:
        return None
    return clean


def _crop_positions(
    height: int,
    width: int,
    axis_r: int,
    axis_c: int,
    box_h: int,
    box_w: int,
) -> list[tuple[int, int, int, int]]:
    quadrants = [
        (0, axis_r - box_h + 1, 0, axis_c - box_w + 1),
        (0, axis_r - box_h + 1, axis_c + 1, width - box_w + 1),
        (axis_r + 1, height - box_h + 1, 0, axis_c - box_w + 1),
        (axis_r + 1, height - box_h + 1, axis_c + 1, width - box_w + 1),
    ]
    out = []
    for r_start, r_stop, c_start, c_stop in quadrants:
        for r0 in range(r_start, r_stop):
            for c0 in range(c_start, c_stop):
                out.append((int(r0), int(c0), int(r0 + box_h - 1), int(c0 + box_w - 1)))
    return out


def _clean_expected(clean: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return _swap_patch_colors(clean[r0 : r1 + 1, c0 : c1 + 1])


def _unique_solver_axes(g: np.ndarray) -> bool:
    row_sum = _best_row_axis_sum(g)
    col_sum = _best_col_axis_sum(g)
    return row_sum == g.shape[0] - 1 and col_sum == g.shape[1] - 1


def _find_mask_candidates(clean: np.ndarray, require_zero: bool) -> list[tuple[int, int, int, int]]:
    height, width = clean.shape
    axis_r = height // 2
    axis_c = width // 2
    candidates: list[tuple[int, int, int, int]] = []
    max_h = min(MASK_HEIGHT_RANGE[1], axis_r)
    max_w = min(MASK_WIDTH_RANGE[1], axis_c)
    for box_h in range(MASK_HEIGHT_RANGE[0], max_h + 1):
        for box_w in range(MASK_WIDTH_RANGE[0], max_w + 1):
            for bbox in _crop_positions(height, width, axis_r, axis_c, box_h, box_w):
                r0, c0, r1, c1 = bbox
                patch = clean[r0 : r1 + 1, c0 : c1 + 1]
                colors = set(map(int, np.unique(patch)))
                if BLUE not in colors or RED not in colors:
                    continue
                if require_zero and BLACK not in colors:
                    continue
                candidates.append(bbox)
    return candidates


def _make_input(clean: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    inp = clean.copy()
    r0, c0, r1, c1 = bbox
    inp[r0 : r1 + 1, c0 : c1 + 1] = MASK
    return inp


def _build_case() -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]] | None:
    height = int(RNG.choice(HEIGHT_CHOICES))
    width = int(RNG.choice(WIDTH_CHOICES))
    clean = _build_clean_grid(height, width)
    if clean is None:
        return None

    require_zero = RNG.random() < WANT_ZERO_IN_CROP_PROB
    candidates = _find_mask_candidates(clean, require_zero=require_zero)
    if not candidates:
        if require_zero:
            candidates = _find_mask_candidates(clean, require_zero=False)
        if not candidates:
            return None

    bbox = candidates[int(RNG.integers(len(candidates)))]
    inp = _make_input(clean, bbox)
    visible_colors = set(map(int, np.unique(inp[inp != MASK])))
    if BLUE not in visible_colors or RED not in visible_colors:
        return None
    if not _unique_solver_axes(inp):
        return None

    expected = _clean_expected(clean, bbox)
    if np.array_equal(expected, clean[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1]):
        return None
    if not np.array_equal(main(inp), expected):
        return None

    return inp, clean, bbox


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        built = _build_case()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid symmetric crop-and-swap input")
