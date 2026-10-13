from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


LEFT_W = 12
BG = 0
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
_TARGET_BAG: list[int] = []


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _scale_sprite(sprite: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(np.asarray(sprite, dtype=int), np.ones((factor, factor), dtype=int))


def _prefix_len(row: np.ndarray, bg: int = BG) -> int:
    nz = np.flatnonzero(row != bg)
    assert nz.size >= 2, "each active row must contain a visible prefix"
    assert int(nz[0]) == 0, "every prefix must start at the left edge"
    return int(nz[-1] + 1)


def _extend_prefix_row(row: np.ndarray, *, mode: str = "full", bg: int = BG) -> np.ndarray:
    """
    mode='full': prefix + fill + right copy
    mode='fill': prefix + fill only
    mode='copy': prefix + right copy only
    """
    out = np.full(LEFT_W, bg, dtype=int)
    prefix_len = _prefix_len(row, bg)
    prefix = np.asarray(row[:prefix_len], dtype=int)
    boundary = int(prefix[0])
    assert int(prefix[-1]) == boundary, "the prefix must close with its boundary color"

    suffix_len = prefix_len - 1
    out[:prefix_len] = prefix
    if mode in {"full", "fill"}:
        out[prefix_len : LEFT_W - suffix_len] = boundary
    if mode in {"full", "copy"}:
        out[LEFT_W - suffix_len :] = prefix[1:]
    return out


def _transform_left(left: np.ndarray, *, mode: str = "full", bg: int = BG) -> np.ndarray:
    return np.asarray([_extend_prefix_row(row, mode=mode, bg=bg) for row in left], dtype=int)


def _count_enclosed_regions(grid: np.ndarray, *, bg: int = BG) -> int:
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if int(grid[r, c]) == bg and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if int(grid[r, c]) == bg and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == bg and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))

    enclosed = (grid == bg) & ~seen
    count = 0
    for r, c in np.argwhere(enclosed):
        rr = int(r)
        cc = int(c)
        if not enclosed[rr, cc]:
            continue
        count += 1
        enclosed[rr, cc] = False
        q = deque([(rr, cc)])
        while q:
            cr, cc2 = q.popleft()
            for nr, nc in ((cr - 1, cc2), (cr + 1, cc2), (cr, cc2 - 1), (cr, cc2 + 1)):
                if 0 <= nr < h and 0 <= nc < w and enclosed[nr, nc]:
                    enclosed[nr, nc] = False
                    q.append((nr, nc))
    return count


def _structural_row(mask: tuple[int, ...]) -> np.ndarray:
    row = np.zeros(LEFT_W, dtype=int)
    vals = [2, *[1 if bit else 0 for bit in mask], 2]
    row[: len(vals)] = vals
    return row


NEUTRAL_MASKS = tuple((1,) * n for n in range(1, 5))
ALL_MASKS = tuple(
    tuple((bits >> idx) & 1 for idx in range(length))
    for length in range(1, 5)
    for bits in range(1 << length)
)
HOLEY_MASKS = tuple(mask for mask in ALL_MASKS if 0 in mask)


def _classify_masks() -> tuple[list[tuple[int, ...]], list[tuple[int, ...]], list[tuple[tuple[int, ...], ...]]]:
    neutral = _structural_row((1,))
    pair_masks: list[tuple[int, ...]] = []
    quad_masks: list[tuple[int, ...]] = []
    triple_cores: list[tuple[tuple[int, ...], ...]] = []

    for mask in HOLEY_MASKS:
        left = np.asarray([neutral, _structural_row(mask), neutral], dtype=int)
        full = _count_enclosed_regions(_transform_left(left, mode="full"))
        raw = _count_enclosed_regions(left)
        fill_only = _count_enclosed_regions(_transform_left(left, mode="fill"))
        copy_only = _count_enclosed_regions(_transform_left(left, mode="copy"))
        if raw != full and fill_only != full and copy_only != full:
            if full == 2:
                pair_masks.append(mask)
            elif full == 4:
                quad_masks.append(mask)

    for a in ALL_MASKS:
        for b in ALL_MASKS:
            for c in ALL_MASKS:
                if a not in NEUTRAL_MASKS and c not in NEUTRAL_MASKS:
                    continue
                left = np.asarray(
                    [_structural_row(a), _structural_row(b), _structural_row(c)],
                    dtype=int,
                )
                full = _count_enclosed_regions(_transform_left(left, mode="full"))
                if full != 3:
                    continue
                raw = _count_enclosed_regions(left)
                fill_only = _count_enclosed_regions(_transform_left(left, mode="fill"))
                copy_only = _count_enclosed_regions(_transform_left(left, mode="copy"))
                if raw != 3 and fill_only != 3 and copy_only != 3:
                    triple_cores.append((a, b, c))

    assert pair_masks and quad_masks and triple_cores
    return pair_masks, quad_masks, triple_cores


PAIR_MASKS, QUAD_MASKS, TRIPLE_CORES = _classify_masks()


def _materialize_prefix(mask: tuple[int, ...]) -> np.ndarray:
    boundary = random.choice(PALETTE)
    accents = [c for c in PALETTE if c != boundary]
    accent_a = random.choice(accents)
    accent_b = random.choice(accents)

    vals = [boundary]
    one_positions = [i for i, bit in enumerate(mask) if bit == 1]
    second_color_positions = set()
    if len(one_positions) >= 2 and random.random() < 0.6:
        second_color_positions = set(random.sample(one_positions, k=random.randint(1, len(one_positions) - 1)))

    for idx, bit in enumerate(mask):
        if bit == 0:
            vals.append(BG)
        else:
            vals.append(accent_b if idx in second_color_positions else accent_a)
    vals.append(boundary)

    row = np.zeros(LEFT_W, dtype=int)
    row[: len(vals)] = np.asarray(vals, dtype=int)
    return row


def _build_left_box(target: int) -> np.ndarray:
    for _ in range(400):
        top_pad = random.randint(0, 2)
        bottom_pad = random.randint(0, 2)

        if target == 2:
            band_mask = random.choice(PAIR_MASKS)
            band_height = random.randint(1, 3)
            core_masks = [random.choice(NEUTRAL_MASKS), *([band_mask] * band_height), random.choice(NEUTRAL_MASKS)]
        elif target == 4:
            band_mask = random.choice(QUAD_MASKS)
            band_height = random.randint(1, 2)
            core_masks = [random.choice(NEUTRAL_MASKS), *([band_mask] * band_height), random.choice(NEUTRAL_MASKS)]
        else:
            core_masks = list(random.choice(TRIPLE_CORES))

        active = np.asarray([_materialize_prefix(mask) for mask in core_masks], dtype=int)
        if len({tuple(map(int, row)) for row in active}) < 2:
            continue
        if not any(0 in row[1 : _prefix_len(row) - 1] for row in active):
            continue
        if not any(np.any(row[1 : _prefix_len(row) - 1] != 0) for row in active):
            continue

        full = _transform_left(active, mode="full")
        raw = _count_enclosed_regions(active)
        fill_only = _count_enclosed_regions(_transform_left(active, mode="fill"))
        copy_only = _count_enclosed_regions(_transform_left(active, mode="copy"))
        if _count_enclosed_regions(full) != target:
            continue
        if raw == target or fill_only == target or copy_only == target:
            continue
        if sum(0 in row[1 : _prefix_len(row) - 1] for row in active) == target:
            continue

        left = np.zeros((top_pad + len(active) + bottom_pad, LEFT_W), dtype=int)
        left[top_pad : top_pad + len(active)] = active
        return left

    raise RuntimeError("failed to build a valid left box")


def _random_template(target: int) -> np.ndarray:
    size = random.choice([2, 3])
    assert size * target <= 12
    colors = random.sample(list(PALETTE), random.choice([2, 3]))
    for _ in range(100):
        template = np.asarray(
            [random.choice(colors) for _ in range(size * size)],
            dtype=int,
        ).reshape(size, size)
        if len(set(map(int, template.ravel()))) < 2:
            continue
        if np.array_equal(template, np.flipud(template)) and np.array_equal(template, np.fliplr(template)):
            continue
        return template
    raise RuntimeError("failed to build a non-degenerate template")


def _next_target() -> int:
    if not _TARGET_BAG:
        _TARGET_BAG.extend([2, 3, 4])
        random.shuffle(_TARGET_BAG)
    return int(_TARGET_BAG.pop())


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend the left prefix rows, decode the resulting enclosed-region count, and use it to scale the right template.

    Input structure:
    - The background is black, with a left 12-column prefix area and a separate right miniature template.
    - In the left area, each active row starts at column 0 and ends with the same boundary color it starts with.
    - The prefix interior may mix black cells and other colors, and all cells after the prefix are background.

    Rule:
    - Restrict to the active left-row band and perform prefix closure on each row: keep the prefix, fill the middle gap
      with the row's boundary color, and copy the prefix without its first cell into the right end of the 12-column area.
    - In that extended left rectangle, flood-fill black from the rectangle boundary and count the remaining enclosed
      4-connected black regions.
    - Crop the right-side template to its bounding box and scale it by that enclosed-region count with nearest-neighbor replication.

    Key insight:
    - The left rows are not the output object: after closure they encode an integer as the number of enclosed black pockets,
      and that decoded integer becomes the scale factor for the independent template on the right.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    assert bg == BG, "this task uses black background"

    left_activity = (g[:, :LEFT_W] != bg).any(axis=1)
    active_rows = np.flatnonzero(left_activity)
    assert active_rows.size >= 3, "expected a visible left row band"
    r0 = int(active_rows[0])
    r1 = int(active_rows[-1])
    assert np.array_equal(active_rows, np.arange(r0, r1 + 1)), "left rows must form one contiguous band"

    left = g[r0 : r1 + 1, :LEFT_W]
    extended_left = _transform_left(left, mode="full", bg=bg)
    factor = _count_enclosed_regions(extended_left, bg=bg)
    assert 2 <= factor <= 4, "enclosed-region count must be the intended scale factor"

    right_mask = (g != bg).copy()
    right_mask[:, :LEFT_W] = False
    template_box = _bbox(right_mask)
    assert template_box is not None, "expected a right-side template"
    tr0, tc0, tr1, tc1 = template_box
    template = g[tr0 : tr1 + 1, tc0 : tc1 + 1]
    return _scale_sprite(template, factor)


def generate_input() -> np.ndarray:
    for _ in range(400):
        target = _next_target()
        left = _build_left_box(target)
        template = _random_template(target)

        gap = random.choice([1, 2])
        left_top = random.randint(0, 2)
        template_top = random.randint(0, 6)
        right_margin = random.randint(1, 4)
        bottom_margin = random.randint(0, 3)

        h = max(left_top + left.shape[0], template_top + template.shape[0]) + bottom_margin
        w = LEFT_W + gap + template.shape[1] + right_margin
        grid = np.zeros((h, w), dtype=int)

        grid[left_top : left_top + left.shape[0], :LEFT_W] = left
        template_col = LEFT_W + gap
        grid[template_top : template_top + template.shape[0], template_col : template_col + template.shape[1]] = template

        out = main(grid)
        left_rows = np.flatnonzero((grid[:, :LEFT_W] != 0).any(axis=1))
        active_left = grid[int(left_rows[0]) : int(left_rows[-1]) + 1, :LEFT_W]
        extended = _transform_left(active_left)
        raw = _count_enclosed_regions(active_left)
        fill_only = _count_enclosed_regions(_transform_left(active_left, mode="fill"))
        copy_only = _count_enclosed_regions(_transform_left(active_left, mode="copy"))
        factor = _count_enclosed_regions(extended)

        if factor != target or raw == target or fill_only == target or copy_only == target:
            continue
        if factor == template.shape[0] == template.shape[1]:
            continue
        if out.shape != (template.shape[0] * target, template.shape[1] * target):
            continue
        if max(grid.shape) > 30 or max(out.shape) > 30:
            continue
        return grid

    raise RuntimeError("failed to generate a valid input")
