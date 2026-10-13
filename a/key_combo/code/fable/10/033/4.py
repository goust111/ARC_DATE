from __future__ import annotations

import random
from collections import deque

import numpy as np


COLOR_IDS = tuple(range(10))
GEN_ATTEMPTS = 600
MASK_ATTEMPTS = 120
MIN_HALF = 3
MAX_HALF = 5
MIN_BRANCH_CELLS = 2


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _component_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
    return count


def _parse_halves(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    rows, cols = g.shape
    assert rows % 2 == 0, "input height must be even"

    half = rows // 2
    top = g[:half, :]
    bottom = g[half:, :]

    top_colors = set(map(int, np.unique(top)))
    bottom_colors = set(map(int, np.unique(bottom)))
    common = top_colors & bottom_colors
    assert len(common) == 1, "halves must share exactly one background color"
    bg = common.pop()

    top_fg = sorted(top_colors - {bg})
    bottom_fg = sorted(bottom_colors - {bg})
    assert len(top_fg) == 1, "top half must use exactly one foreground color"
    assert len(bottom_fg) == 1, "bottom half must use exactly one foreground color"
    assert top_fg[0] != bottom_fg[0], "top and bottom foreground colors must differ"
    return top, bottom, bg, top_fg[0], bottom_fg[0]


def _render_output(top: np.ndarray, bottom: np.ndarray, bg: int) -> np.ndarray:
    h, w = top.shape
    out = np.full((h * h, w * w), bg, dtype=int)
    top_mask = top != bg
    bottom_mask = bottom != bg
    for r in range(h):
        for c in range(w):
            r0 = r * h
            c0 = c * w
            block = out[r0 : r0 + h, c0 : c0 + w]
            if top_mask[r, c] and not bottom_mask[r, c]:
                block[:] = top
            elif bottom_mask[r, c] and not top_mask[r, c]:
                block[:] = bottom
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Expand each exclusive foreground cell into a full half-pattern stamp chosen by its provenance.

    Input structure:
    - The input height is even and is split into two aligned halves of the same shape.
    - The two halves share one background color; the top half has one foreground color and the bottom half has a different foreground color.
    - Each half is simultaneously a placement mask and the full stamp pattern associated with that half.

    Rule:
    - Compare the two halves cell by cell.
    - If only the top cell is foreground, place a full copy of the entire top-half pattern into the corresponding output block.
    - If only the bottom cell is foreground, place a full copy of the entire bottom-half pattern into the corresponding output block.
    - If both aligned cells are background or both are foreground, leave that output block entirely background.

    Key insight:
    - The symmetric difference decides where blocks appear, but each block's content is determined by which half contributed the exclusive foreground at that location.
    """
    top, bottom, bg, _top_fg, _bottom_fg = _parse_halves(grid)
    return _render_output(top, bottom, bg)


def _occupied_span_count(mask: np.ndarray, axis: int) -> int:
    return int(np.count_nonzero(mask.any(axis=axis)))


def _mask_ok(mask: np.ndarray) -> bool:
    h, w = mask.shape
    area = h * w
    filled = int(mask.sum())
    if filled < 3 or filled > area - 2:
        return False
    if _occupied_span_count(mask, axis=1) < 2 or _occupied_span_count(mask, axis=0) < 2:
        return False
    comps = _component_count(mask)
    if comps < 1 or comps > 3:
        return False
    return True


def _random_walk_mask(h: int, w: int) -> np.ndarray:
    area = h * w
    target = random.randint(3, max(3, area - 2))
    cells = {(random.randrange(h), random.randrange(w))}
    while len(cells) < target:
        r, c = random.choice(tuple(cells))
        dr, dc = random.choice(((1, 0), (-1, 0), (0, 1), (0, -1)))
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            cells.add((nr, nc))

    if random.random() < 0.35 and len(cells) <= area - 4:
        for _ in range(20):
            seed_r, seed_c = random.randrange(h), random.randrange(w)
            if (seed_r, seed_c) in cells:
                continue
            arm = [(seed_r, seed_c)]
            if random.random() < 0.5 and seed_c + 1 < w and (seed_r, seed_c + 1) not in cells:
                arm.append((seed_r, seed_c + 1))
            elif seed_r + 1 < h and (seed_r + 1, seed_c) not in cells:
                arm.append((seed_r + 1, seed_c))
            for cell in arm:
                cells.add(cell)
            break

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _rect_union_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    part_count = random.randint(1, 3)
    for _ in range(part_count):
        rh = random.randint(1, h)
        rw = random.randint(1, w)
        if rh == h and rw == w:
            if random.random() < 0.5 and rh > 1:
                rh -= 1
            elif rw > 1:
                rw -= 1
        r0 = random.randint(0, h - rh)
        c0 = random.randint(0, w - rw)
        mask[r0 : r0 + rh, c0 : c0 + rw] = True
    return mask


def _random_mask(h: int, w: int) -> np.ndarray:
    for _ in range(MASK_ATTEMPTS):
        if random.random() < 0.5:
            mask = _random_walk_mask(h, w)
        else:
            mask = _rect_union_mask(h, w)
        if _mask_ok(mask):
            return mask
    raise RuntimeError("failed to build a valid half-mask")


def _wrong_uniform_output(top: np.ndarray, bottom: np.ndarray, bg: int, use_top: bool) -> np.ndarray:
    h, w = top.shape
    stamp = top if use_top else bottom
    out = np.full((h * h, w * w), bg, dtype=int)
    xor_mask = (top != bg) ^ (bottom != bg)
    for r, c in np.argwhere(xor_mask):
        out[r * h : (r + 1) * h, c * w : (c + 1) * w] = stamp
    return out


def _wrong_swapped_output(top: np.ndarray, bottom: np.ndarray, bg: int) -> np.ndarray:
    h, w = top.shape
    out = np.full((h * h, w * w), bg, dtype=int)
    top_only = (top != bg) & (bottom == bg)
    bottom_only = (bottom != bg) & (top == bg)
    for r, c in np.argwhere(top_only):
        out[r * h : (r + 1) * h, c * w : (c + 1) * w] = bottom
    for r, c in np.argwhere(bottom_only):
        out[r * h : (r + 1) * h, c * w : (c + 1) * w] = top
    return out


def _semantic_ok(grid: np.ndarray, top: np.ndarray, bottom: np.ndarray, bg: int, top_fg: int, bottom_fg: int) -> bool:
    expected = _render_output(top, bottom, bg)
    if not np.array_equal(main(grid), expected):
        return False

    top_mask = top != bg
    bottom_mask = bottom != bg
    top_only = top_mask & ~bottom_mask
    bottom_only = bottom_mask & ~top_mask
    overlap = top_mask & bottom_mask
    both_bg = ~top_mask & ~bottom_mask

    if int(top_only.sum()) < MIN_BRANCH_CELLS or int(bottom_only.sum()) < MIN_BRANCH_CELLS:
        return False
    if int(overlap.sum()) < 1 or int(both_bg.sum()) < 1:
        return False
    if np.array_equal(top_mask, bottom_mask):
        return False
    if int(np.count_nonzero(top_mask ^ bottom_mask)) < max(3, (top.size // 4)):
        return False
    if int(expected.size) == 0 or not np.any(expected == top_fg) or not np.any(expected == bottom_fg):
        return False
    if not np.any(expected == bg):
        return False

    uniform_top = _wrong_uniform_output(top, bottom, bg, use_top=True)
    uniform_bottom = _wrong_uniform_output(top, bottom, bg, use_top=False)
    swapped = _wrong_swapped_output(top, bottom, bg)
    if np.array_equal(expected, uniform_top) or np.array_equal(expected, uniform_bottom):
        return False
    if np.array_equal(expected, swapped):
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(MIN_HALF, MAX_HALF)
    w = random.randint(MIN_HALF, MAX_HALF)

    bg, top_fg, bottom_fg = random.sample(COLOR_IDS, 3)
    top_mask = _random_mask(h, w)
    bottom_mask = _random_mask(h, w)

    if np.array_equal(top_mask, bottom_mask):
        return None

    top = np.where(top_mask, top_fg, bg).astype(int)
    bottom = np.where(bottom_mask, bottom_fg, bg).astype(int)
    grid = np.vstack([top, bottom]).astype(int)

    if not _semantic_ok(grid, top, bottom, bg, top_fg, bottom_fg):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stamp-provenance ARC input")
