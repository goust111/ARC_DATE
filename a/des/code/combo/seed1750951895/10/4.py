import random
from collections import Counter

import numpy as np


def get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _axis_score(coords: list[tuple[int, int]], width: int, k: int) -> int:
    pts = {(int(r), int(c)) for r, c in coords}
    score = 0
    for r, c in pts:
        cc = int(k) - int(c)
        if 0 <= cc < width and (int(r), int(cc)) in pts:
            score += 1
    return int(score)


def _best_axis(coords: list[tuple[int, int]], width: int) -> tuple[int, int, int]:
    best_k = 0
    best_score = -1
    second_score = -1
    for k in range(0, 2 * (width - 1) + 1):
        score = _axis_score(coords, width, k)
        if score > best_score:
            second_score = best_score
            best_score = score
            best_k = k
        elif score > second_score:
            second_score = score
    return int(best_k), int(best_score), int(second_score)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Clean each non-background color layer by enforcing its own best vertical mirror symmetry.

    For each color independently, search all vertical axes, keep the axis maximizing
    mirrored same-color matches, then delete pixels without a mirror counterpart.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h > 0 and w > 0

    bg = get_bg(g)
    out = g.copy()
    colors = [int(c) for c in np.unique(g) if int(c) != int(bg)]
    assert colors

    for color in colors:
        pts_arr = np.argwhere(g == int(color))
        if pts_arr.size == 0:
            continue
        pts = [(int(r), int(c)) for r, c in pts_arr]
        best_k, _best, _second = _best_axis(pts, w)

        for r, c in pts:
            cc = int(best_k) - int(c)
            if not (0 <= cc < w and int(g[r, cc]) == int(color)):
                out[r, c] = int(bg)

    return out


# --------------- generator ---------------

GRID_H_RANGE = (14, 27)
GRID_W_RANGE = (14, 27)
COLOR_COUNT_RANGE = (2, 4)
RECT_SIZE_RANGE = (5, 10)
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
BG_COLOR = 0
OUTER_RETRIES = 400
RECT_PLACEMENT_RETRIES = 120
SHAPE_RETRIES = 80
NOISE_RETRIES = 40


def _randint(lo: int, hi: int) -> int:
    return int(np.random.randint(lo, hi + 1))


def _sample_rectangles(h: int, w: int, n: int) -> list[tuple[int, int, int, int]] | None:
    rects: list[tuple[int, int, int, int]] = []
    for _ in range(RECT_PLACEMENT_RETRIES):
        rects.clear()
        ok = True
        for _idx in range(n):
            rh = _randint(RECT_SIZE_RANGE[0], min(RECT_SIZE_RANGE[1], h - 2))
            rw = _randint(RECT_SIZE_RANGE[0], min(RECT_SIZE_RANGE[1], w - 2))
            placed = False
            for _ in range(80):
                r0 = _randint(1, h - rh - 1)
                c0 = _randint(1, w - rw - 1)
                r1 = r0 + rh
                c1 = c0 + rw
                overlap = False
                for rr0, cc0, rr1, cc1 in rects:
                    if not (r1 + 1 <= rr0 or rr1 + 1 <= r0 or c1 + 1 <= cc0 or cc1 + 1 <= c0):
                        overlap = True
                        break
                if not overlap:
                    rects.append((r0, c0, r1, c1))
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if ok:
            return list(rects)
    return None


def _coords_from_mask(mask: np.ndarray, r0: int, c0: int) -> list[tuple[int, int]]:
    pts = np.argwhere(mask)
    return [(int(r0 + r), int(c0 + c)) for r, c in pts]


def _build_symmetric_mask(rh: int, rw: int) -> tuple[np.ndarray, int] | None:
    for _ in range(SHAPE_RETRIES):
        use_integer_axis = bool(random.getrandbits(1))
        if use_integer_axis:
            axis_col = _randint(1, rw - 2)
            k = 2 * axis_col
        else:
            if rw < 4:
                continue
            left_col = _randint(1, rw - 3)
            k = 2 * left_col + 1

        mask = np.zeros((rh, rw), dtype=bool)
        pair_slots: list[tuple[int, int, int]] = []
        fixed_slots: list[tuple[int, int]] = []
        for r in range(rh):
            for c in range(rw):
                cc = k - c
                if not (0 <= cc < rw):
                    continue
                if c < cc:
                    pair_slots.append((r, c, cc))
                elif c == cc:
                    fixed_slots.append((r, c))

        if len(pair_slots) < 3:
            continue

        pair_count = _randint(3, max(3, min(len(pair_slots), (rh * rw) // 4 + 2)))
        random.shuffle(pair_slots)
        chosen_pairs = pair_slots[:pair_count]
        for r, c1, c2 in chosen_pairs:
            mask[r, c1] = True
            mask[r, c2] = True

        if fixed_slots and random.random() < 0.5:
            extra = _randint(1, min(len(fixed_slots), max(1, rh // 2)))
            random.shuffle(fixed_slots)
            for r, c in fixed_slots[:extra]:
                mask[r, c] = True

        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        if rows.size == 0 or cols.size == 0:
            continue
        if int(rows[-1] - rows[0] + 1) < 3 or int(cols[-1] - cols[0] + 1) < 3:
            continue
        if int(mask.sum()) < 7:
            continue
        return mask, int(k)
    return None


def _add_noise(mask: np.ndarray, axis_k: int) -> np.ndarray | None:
    rh, rw = mask.shape
    base = mask.copy()
    empty = [(r, c) for r in range(rh) for c in range(rw) if not bool(base[r, c])]
    random.shuffle(empty)

    clean_coords = _coords_from_mask(base, 0, 0)
    _best_k, clean_best, clean_second = _best_axis(clean_coords, rw)
    if clean_best - clean_second < 2:
        return None

    max_noise = min(4, max(1, len(empty) // 8))
    for _ in range(NOISE_RETRIES):
        noisy = base.copy()
        noise_count = _randint(1, max_noise)
        added = 0
        for r, c in empty:
            cc = axis_k - c
            if 0 <= cc < rw and bool(base[r, cc]):
                continue
            noisy[r, c] = True
            added += 1
            if added == noise_count:
                break
        if added == 0:
            continue
        pts = _coords_from_mask(noisy, 0, 0)
        best_k, best_score, second_score = _best_axis(pts, rw)
        if best_k != axis_k:
            continue
        if best_score - second_score < 2:
            continue
        if np.array_equal(noisy, base):
            continue
        return noisy
    return None


def _gen_attempt() -> np.ndarray | None:
    h = _randint(*GRID_H_RANGE)
    w = _randint(*GRID_W_RANGE)
    n_colors = _randint(*COLOR_COUNT_RANGE)

    rects = _sample_rectangles(h, w, n_colors)
    if rects is None:
        return None

    colors = random.sample(PALETTE, n_colors)
    grid = np.full((h, w), BG_COLOR, dtype=int)

    any_even_axis = False
    any_odd_axis = False
    for color, (r0, c0, r1, c1) in zip(colors, rects):
        rh = int(r1 - r0)
        rw = int(c1 - c0)
        built = _build_symmetric_mask(rh, rw)
        if built is None:
            return None
        clean_mask, local_k = built
        noisy_mask = _add_noise(clean_mask, local_k)
        if noisy_mask is None:
            return None

        global_k = 2 * c0 + local_k
        coords = _coords_from_mask(noisy_mask, r0, c0)
        best_k, best_score, second_score = _best_axis(coords, w)
        if best_k != global_k or best_score - second_score < 2:
            return None

        clean_coords = _coords_from_mask(clean_mask, r0, c0)
        clean_set = set(clean_coords)
        noisy_set = set(coords)
        if not (noisy_set - clean_set):
            return None

        for r, c in coords:
            if grid[r, c] != BG_COLOR:
                return None
            grid[r, c] = int(color)

        any_even_axis = any_even_axis or (local_k % 2 == 0)
        any_odd_axis = any_odd_axis or (local_k % 2 == 1)

    if not any_even_axis and not any_odd_axis:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if not np.any(out != BG_COLOR):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(OUTER_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
