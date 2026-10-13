from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

TEMPLATE_COLORS = [BLUE, GREEN, YELLOW, PINK, ORANGE, TEAL, MAROON]
GEN_SIZES = [3, 4]
GEN_SCALES = [2, 3]
GEN_RETRIES = 800

_MASK_CACHE: dict[int, list[np.ndarray]] = {}


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _rotate_cw(grid: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(grid, k=-(turns % 4))


def _paste_nonzero(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, preserve_existing: bool) -> None:
    h, w = patch.shape
    for rr in range(h):
        for cc in range(w):
            value = int(patch[rr, cc])
            if value == BG:
                continue
            dr = r0 + rr
            dc = c0 + cc
            if preserve_existing and int(dst[dr, dc]) != BG:
                continue
            dst[dr, dc] = value


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    stack = [start]
    seen = {start}
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _component_count(mask: np.ndarray) -> int:
    remaining = {tuple(map(int, rc)) for rc in np.argwhere(mask)}
    count = 0
    while remaining:
        count += 1
        start = remaining.pop()
        stack = [start]
        while stack:
            r, c = stack.pop()
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (nr, nc) in remaining:
                    remaining.remove((nr, nc))
                    stack.append((nr, nc))
    return count


def _occupied_columns(sprite: np.ndarray) -> tuple[int, ...]:
    cols = np.where(np.any(sprite != BG, axis=0))[0]
    return tuple(map(int, cols))


def _match_marker_columns(marker_cols: tuple[int, ...], occupied_cols: tuple[int, ...], scale: int, offset: int) -> bool:
    intervals = [(offset + c * scale, offset + (c + 1) * scale - 1) for c in occupied_cols]
    covered = [False] * len(intervals)
    for marker in marker_cols:
        hit = False
        for idx, (lo, hi) in enumerate(intervals):
            if lo <= marker <= hi:
                covered[idx] = True
                hit = True
                break
        if not hit:
            return False
    return all(covered)


def _find_solutions(template: np.ndarray, marker_cols: tuple[int, ...], width: int) -> list[tuple[int, int, int, np.ndarray]]:
    solutions: list[tuple[int, int, int, np.ndarray]] = []
    for turns in range(4):
        rotated = _rotate_cw(template, turns)
        occupied_cols = _occupied_columns(rotated)
        if not occupied_cols:
            continue
        max_scale = width // rotated.shape[1]
        for scale in range(1, max_scale + 1):
            max_offset = width - rotated.shape[1] * scale
            for offset in range(max_offset + 1):
                if _match_marker_columns(marker_cols, occupied_cols, scale, offset):
                    solutions.append((turns, scale, offset, rotated))
    return solutions


def _apply_solution(
    grid: np.ndarray,
    template_box: tuple[int, int, int, int],
    marker_cols: tuple[int, ...],
    solution: tuple[int, int, int, np.ndarray],
) -> np.ndarray:
    _turns, scale, offset, rotated = solution
    out = np.asarray(grid, dtype=int).copy()
    h = out.shape[0]

    out[h - 1, list(marker_cols)] = BG
    scaled = np.kron(rotated, np.ones((scale, scale), dtype=int))
    _paste_nonzero(out, scaled, 0, offset, preserve_existing=True)

    r0, c0, r1, c1 = template_box
    original_template = grid[r0 : r1 + 1, c0 : c1 + 1]
    _paste_nonzero(out, original_template, r0, c0, preserve_existing=False)

    for c in marker_cols:
        for r in range(h - 2, -1, -1):
            if int(out[r, c]) == BG:
                out[r, c] = RED
            else:
                break
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the template rotation from bottom markers, place the rotated scaled copy at the top, and draw upward hit rays.

    Input structure:
    - A black background contains one small multicolor template in the upper-left region and 2-4 gray marker pixels on the bottom edge.
    - The template occupies a tight 3x3 or 4x4 bbox with internal background allowed, while the rest of the grid is empty.

    Rule:
    - Crop the small template, try its four clockwise rotations, and find the unique rotation, integer scale, and horizontal offset whose occupied template columns can explain the bottom marker columns after block scaling.
    - Place that rotated template at the top edge as a kxk block-scaled copy, erase the bottom markers, and for each former marker column paint red upward through background until the first non-background pixel is reached.
    - Keep the original small template unchanged.

    Key insight:
    - The bottom markers do not point to a fixed target directly; they encode which rotated template columns must be occupied after scaling, so the correct rotation must be inferred before anything can be drawn.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    marker_cols = tuple(sorted(int(c) for c in np.where(g[h - 1] == GRAY)[0]))
    assert 2 <= len(marker_cols) <= 4, "expected 2-4 bottom markers"

    template_mask = (g != BG) & (g != GRAY)
    template_box = _bbox(template_mask)
    assert template_box is not None, "expected a colored template"
    r0, c0, r1, c1 = template_box
    template = _crop(g, template_box)
    assert 3 <= template.shape[0] <= 4 and 3 <= template.shape[1] <= 4, "template bbox must be 3x3 or 4x4"

    solutions = _find_solutions(template, marker_cols, w)
    assert len(solutions) == 1, "rotation / scale / offset must be uniquely determined"
    return _apply_solution(g, (r0, c0, r1, c1), marker_cols, solutions[0])


def _candidate_masks(n: int) -> list[np.ndarray]:
    if n in _MASK_CACHE:
        return _MASK_CACHE[n]

    total = n * n
    min_area = max(4, int(np.ceil(0.45 * total)))
    max_area = min(total - 1, int(np.floor(0.8 * total)))
    out: list[np.ndarray] = []

    for bits in range(1, 1 << total):
        area = bits.bit_count()
        if area < min_area or area > max_area:
            continue
        flat = np.array([(bits >> idx) & 1 for idx in range(total)], dtype=bool)
        mask = flat.reshape(n, n)
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not (rows[0] and rows[-1] and cols[0] and cols[-1]):
            continue
        components = _component_count(mask)
        if components > 3:
            continue
        signatures = {_occupied_columns(_rotate_cw(mask.astype(int), turns)) for turns in range(4)}
        if len(signatures) < 2:
            continue
        if max(len(sig) for sig in signatures) < 2:
            continue
        out.append(mask)

    _MASK_CACHE[n] = out
    return out


def _color_template(mask: np.ndarray) -> np.ndarray:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    color_count = random.randint(2, min(3, len(coords)))
    palette = random.sample(TEMPLATE_COLORS, color_count)
    assignments = palette[:]
    while len(assignments) < len(coords):
        assignments.append(random.choice(palette))
    random.shuffle(assignments)

    template = np.zeros(mask.shape, dtype=int)
    for (r, c), color in zip(coords, assignments):
        template[r, c] = int(color)
    return template


def _build_attempt() -> np.ndarray | None:
    size = random.choice(GEN_SIZES)
    mask = random.choice(_candidate_masks(size))
    template = _color_template(mask)

    turns = random.randrange(4)
    scale = random.choice(GEN_SCALES)
    rotated = _rotate_cw(template, turns)
    occupied_cols = _occupied_columns(rotated)
    if not (2 <= len(occupied_cols) <= 4):
        return None

    tr = random.randint(1, 4)
    tc = random.randint(0, 2)
    gap = random.randint(2, 4)

    min_h = max(15, rotated.shape[0] * scale + 5, tr + size + 1)
    max_h = 24
    if min_h > max_h:
        return None
    height = random.randint(min_h, max_h)

    min_w = max(16, tc + size + gap + rotated.shape[1] * scale + 1)
    max_w = 26
    if min_w > max_w:
        return None
    width = random.randint(min_w, max_w)

    min_offset = tc + size + gap
    max_offset = width - rotated.shape[1] * scale
    if min_offset > max_offset:
        return None
    offset = random.randint(min_offset, max_offset)

    marker_cols = []
    for c in occupied_cols:
        lo = offset + c * scale
        hi = offset + (c + 1) * scale - 1
        marker_cols.append(random.randint(lo, hi))
    marker_cols_t = tuple(sorted(marker_cols))

    grid = np.zeros((height, width), dtype=int)
    _paste_nonzero(grid, template, tr, tc, preserve_existing=False)
    grid[height - 1, list(marker_cols_t)] = GRAY

    solutions = _find_solutions(template, marker_cols_t, width)
    if len(solutions) != 1:
        return None
    sol_turns, sol_scale, sol_offset, _sol_rotated = solutions[0]
    if (sol_turns, sol_scale, sol_offset) != (turns, scale, offset):
        return None

    expected = _apply_solution(grid, (tr, tc, tr + size - 1, tc + size - 1), marker_cols_t, solutions[0])
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle 7 input")
