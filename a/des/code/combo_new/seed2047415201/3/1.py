from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
TEAL = 8
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
CROSS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _main_colors(g: np.ndarray) -> tuple[int, int]:
    counts: dict[int, int] = {}
    for value in map(int, g.ravel()):
        if value != BLACK:
            counts[value] = counts.get(value, 0) + 1
    assert len(counts) >= 2, "expected at least two non-black foreground colors"
    colors = list(counts)
    colors.sort(key=lambda color: counts[color], reverse=True)
    return int(colors[0]), int(colors[1])


def _neighbor_values(g: np.ndarray, r: int, c: int) -> list[int]:
    vals: list[int] = []
    for dr, dc in NEIGHBORS4:
        rr, cc = int(r) + dr, int(c) + dc
        if _in_bounds(g.shape, rr, cc):
            vals.append(int(g[rr, cc]))
    return vals


def _has_same_2x2(g: np.ndarray, r: int, c: int, color: int) -> bool:
    h, w = g.shape
    for dr in (-1, 0):
        for dc in (-1, 0):
            rr = int(r) + dr
            cc = int(c) + dc
            if 0 <= rr < h - 1 and 0 <= cc < w - 1:
                if bool(np.all(g[rr : rr + 2, cc : cc + 2] == int(color))):
                    return True
    return False


def _clean_foreground(g: np.ndarray, colors: tuple[int, int]) -> np.ndarray:
    a, b = map(int, colors)
    out = g.copy()

    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            v = int(g[r, c])
            if v == BLACK:
                continue
            if v not in colors:
                out[r, c] = BLACK
                continue

            other = b if v == a else a
            vals = _neighbor_values(g, r, c)
            same4 = sum(x == v for x in vals)
            other4 = sum(x == other for x in vals)

            if len(vals) == 4 and other4 == 4:
                out[r, c] = other
            elif same4 <= 1 and not _has_same_2x2(g, r, c, v):
                out[r, c] = BLACK
            else:
                out[r, c] = v

    return out


def _sub3(g: np.ndarray, r: int, c: int) -> np.ndarray:
    h, w = g.shape
    return g[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)]


def _defect_centers(original: np.ndarray, clean: np.ndarray, colors: tuple[int, int]) -> list[tuple[int, int, int]]:
    centers: list[tuple[int, int, int]] = []
    a, b = map(int, colors)

    for r_, c_ in zip(*np.where(original == BLACK), strict=False):
        r, c = int(r_), int(c_)
        if int(clean[r, c]) != BLACK:
            continue

        sub = _sub3(clean, r, c)
        count_a = int(np.sum(sub == a))
        count_b = int(np.sum(sub == b))
        if count_a == count_b:
            continue

        host = a if count_a > count_b else b
        other_count = count_b if host == a else count_a
        cardinal = 0
        for dr, dc in NEIGHBORS4:
            rr, cc = r + dr, c + dc
            if _in_bounds(clean.shape, rr, cc) and int(clean[rr, cc]) == host:
                cardinal += 1

        if max(count_a, count_b) >= 5 and max(count_a, count_b) > other_count and cardinal >= 3:
            centers.append((r, c, int(host)))

    return centers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark black defects inside two foreground regions with opposite-color crosses.

    Input structure:
    - Black is the background, and the two largest non-black colors form large separated foreground regions.
    - The regions are arranged on opposite sides of the grid, either left/right or top/bottom, and may contain single-cell black defects.
    - A few one-cell boundary protrusions or isolated wrong-color cells may appear on or near the regions.

    Rule:
    - First clean the foreground: erase a same-color one-neighbor protrusion to Black, and recolor a foreground pixel fully 4-surrounded by the other foreground color to that surrounding color.
    - For each original Black cell whose cleaned 3x3 neighborhood is mostly one foreground color, treat it as a defect in that color's region.
    - Draw a 3x3 cross at each defect using the other foreground color, then recolor the defect center itself to Teal (8).

    Key insight:
    - The Teal centers reveal the original Black defects, while each surrounding cross borrows its color from the opposite foreground region after boundary noise has been ignored.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    colors = _main_colors(g)
    clean = _clean_foreground(g, colors)
    centers = _defect_centers(g, clean, colors)

    out = clean.copy()
    for r, c, host in centers:
        other = colors[1] if int(host) == int(colors[0]) else colors[0]
        for dr, dc in CROSS:
            rr, cc = r + dr, c + dc
            if _in_bounds(out.shape, rr, cc):
                out[rr, cc] = int(other)

    for r, c, _host in centers:
        out[r, c] = TEAL

    return out


# --------------- generator ---------------

GRID_MIN, GRID_MAX = 18, 30
REGION_MIN_SIDE = 7
REGION_GAP_MIN = 4
MAX_GEN_ATTEMPTS = 500
DENT_SEPARATION = 3
DENTS_PER_REGION = (1, 4)
PROTRUSION_RANGE = (1, 4)
ISOLATED_WRONG_RANGE = (1, 4)
SPECK_RANGE = (0, 3)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((int(rr), int(cc)))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _shape_has_sturdy_boundary(mask: np.ndarray) -> bool:
    if len(_components(mask)) != 1:
        return False
    h, w = mask.shape
    for r, c in zip(*np.where(mask), strict=False):
        same = 0
        for dr, dc in NEIGHBORS4:
            rr, cc = int(r) + dr, int(c) + dc
            if 0 <= rr < h and 0 <= cc < w and bool(mask[rr, cc]):
                same += 1
        if same < 2:
            return False
    return True


def _make_blob(height: int, width: int, rng: random.Random) -> np.ndarray | None:
    if height < REGION_MIN_SIDE or width < REGION_MIN_SIDE:
        return None

    for _ in range(80):
        intervals: list[tuple[int, int]] = []
        center = rng.randint(width // 2 - 1, width // 2 + 1)
        half = rng.randint(width // 3, max(width // 3 + 1, width // 2 - 1))

        for r in range(height):
            if r:
                center += rng.choice([-1, 0, 0, 1])
                half += rng.choice([-1, 0, 0, 1])
            half = max(3, min(half, width // 2))
            center = max(half, min(width - half - 1, center))
            left = center - half
            right = center + half
            if intervals:
                pl, pr = intervals[-1]
                if min(pr, right) - max(pl, left) + 1 < 4:
                    left, right = pl, pr
            intervals.append((left, right))

        mask = np.zeros((height, width), dtype=bool)
        for r, (left, right) in enumerate(intervals):
            mask[r, left : right + 1] = True

        if _shape_has_sturdy_boundary(mask):
            return mask
    return None


def _paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    for rr, cc in zip(*np.where(mask), strict=False):
        grid[int(r0 + rr), int(c0 + cc)] = int(color)


def _layout_regions(rng: random.Random, color_a: int, color_b: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    for _ in range(100):
        h = rng.randint(GRID_MIN, GRID_MAX)
        w = rng.randint(GRID_MIN, GRID_MAX)
        vertical = rng.random() < 0.5

        if vertical:
            reg_h = rng.randint(max(REGION_MIN_SIDE, h - 9), h - 3)
            max_reg_w = min(10, (w - REGION_GAP_MIN - 4) // 2)
            if max_reg_w < REGION_MIN_SIDE:
                continue
            reg_w = rng.randint(REGION_MIN_SIDE, max_reg_w)
            shape = _make_blob(reg_h, reg_w, rng)
            if shape is None:
                continue
            left_shape = shape
            right_shape = np.fliplr(shape) if rng.random() < 0.75 else shape.copy()

            r0 = rng.randint(1, h - reg_h - 1)
            r1 = min(max(1, r0 + rng.choice([-1, 0, 0, 1])), h - reg_h - 1)
            max_left_c = max(1, w // 2 - REGION_GAP_MIN // 2 - reg_w)
            if max_left_c < 1:
                continue
            c0 = rng.randint(1, max_left_c)
            min_right_c = max(c0 + reg_w + REGION_GAP_MIN, w // 2 + REGION_GAP_MIN // 2)
            max_right_c = w - reg_w - 2
            if min_right_c > max_right_c:
                continue
            c1 = rng.randint(min_right_c, max_right_c)

            grid = np.zeros((h, w), dtype=int)
            mask_a = np.zeros((h, w), dtype=bool)
            mask_b = np.zeros((h, w), dtype=bool)
            mask_a[r0 : r0 + reg_h, c0 : c0 + reg_w] = left_shape
            mask_b[r1 : r1 + reg_h, c1 : c1 + reg_w] = right_shape
        else:
            reg_w = rng.randint(max(REGION_MIN_SIDE, w - 9), w - 3)
            max_reg_h = min(10, (h - REGION_GAP_MIN - 4) // 2)
            if max_reg_h < REGION_MIN_SIDE:
                continue
            reg_h = rng.randint(REGION_MIN_SIDE, max_reg_h)
            shape = _make_blob(reg_h, reg_w, rng)
            if shape is None:
                continue
            top_shape = shape
            bottom_shape = np.flipud(shape) if rng.random() < 0.75 else shape.copy()

            c0 = rng.randint(1, w - reg_w - 1)
            c1 = min(max(1, c0 + rng.choice([-1, 0, 0, 1])), w - reg_w - 1)
            max_top_r = max(1, h // 2 - REGION_GAP_MIN // 2 - reg_h)
            if max_top_r < 1:
                continue
            r0 = rng.randint(1, max_top_r)
            min_bottom_r = max(r0 + reg_h + REGION_GAP_MIN, h // 2 + REGION_GAP_MIN // 2)
            max_bottom_r = h - reg_h - 2
            if min_bottom_r > max_bottom_r:
                continue
            r1 = rng.randint(min_bottom_r, max_bottom_r)

            grid = np.zeros((h, w), dtype=int)
            mask_a = np.zeros((h, w), dtype=bool)
            mask_b = np.zeros((h, w), dtype=bool)
            mask_a[r0 : r0 + reg_h, c0 : c0 + reg_w] = top_shape
            mask_b[r1 : r1 + reg_h, c1 : c1 + reg_w] = bottom_shape

        if bool(np.any(mask_a & mask_b)):
            continue
        grid[mask_a] = int(color_a)
        grid[mask_b] = int(color_b)
        if _defect_centers(grid, grid, (color_a, color_b)):
            continue
        return grid, mask_a, mask_b

    return None


def _solid_3x3_candidates(grid: np.ndarray, mask: np.ndarray, color: int) -> list[tuple[int, int]]:
    h, w = grid.shape
    candidates: list[tuple[int, int]] = []
    for r, c in zip(*np.where(mask), strict=False):
        rr, cc = int(r), int(c)
        if rr <= 0 or cc <= 0 or rr >= h - 1 or cc >= w - 1:
            continue
        if bool(np.all(grid[rr - 1 : rr + 2, cc - 1 : cc + 2] == int(color))):
            candidates.append((rr, cc))
    return candidates


def _choose_spaced(candidates: list[tuple[int, int]], count: int, chosen: list[tuple[int, int]], rng: random.Random) -> list[tuple[int, int]] | None:
    rng.shuffle(candidates)
    picked: list[tuple[int, int]] = []
    for r, c in candidates:
        all_points = chosen + picked
        if all(max(abs(r - rr), abs(c - cc)) >= DENT_SEPARATION for rr, cc in all_points):
            picked.append((int(r), int(c)))
            if len(picked) == int(count):
                return picked
    return None


def _stamp_crosses(clean_with_dents: np.ndarray, dents: list[tuple[int, int, int]], colors: tuple[int, int]) -> np.ndarray:
    out = clean_with_dents.copy()
    for r, c, host in dents:
        other = colors[1] if int(host) == int(colors[0]) else colors[0]
        for dr, dc in CROSS:
            rr, cc = int(r) + dr, int(c) + dc
            if _in_bounds(out.shape, rr, cc):
                out[rr, cc] = int(other)
    for r, c, _host in dents:
        out[int(r), int(c)] = TEAL
    return out


def _near_any(r: int, c: int, points: list[tuple[int, int]], distance: int) -> bool:
    return any(max(abs(int(r) - rr), abs(int(c) - cc)) <= int(distance) for rr, cc in points)


def _add_protrusions(inp: np.ndarray, base: np.ndarray, color: int, count: int, protected: list[tuple[int, int]], rng: random.Random) -> None:
    h, w = inp.shape
    candidates: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if int(base[r, c]) != BLACK or _near_any(r, c, protected, 2):
                continue
            same = 0
            touches_other = False
            for dr, dc in NEIGHBORS4:
                rr, cc = r + dr, c + dc
                if not _in_bounds(base.shape, rr, cc):
                    continue
                if int(base[rr, cc]) == int(color):
                    same += 1
                elif int(base[rr, cc]) != BLACK:
                    touches_other = True
            if same == 1 and not touches_other:
                candidates.append((r, c))

    rng.shuffle(candidates)
    for r, c in candidates[: int(count)]:
        inp[r, c] = int(color)


def _add_isolated_wrong(inp: np.ndarray, base: np.ndarray, host: int, wrong: int, count: int, protected: list[tuple[int, int]], rng: random.Random) -> None:
    h, w = inp.shape
    candidates: list[tuple[int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if _near_any(r, c, protected, 2):
                continue
            if int(inp[r, c]) != int(host):
                continue
            if all(int(inp[r + dr, c + dc]) == int(host) for dr, dc in NEIGHBORS4):
                candidates.append((r, c))

    rng.shuffle(candidates)
    for r, c in candidates[: int(count)]:
        inp[r, c] = int(wrong)


def _add_background_specks(inp: np.ndarray, base: np.ndarray, colors: tuple[int, int], count: int, protected: list[tuple[int, int]], rng: random.Random) -> None:
    h, w = inp.shape
    candidates: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if int(base[r, c]) != BLACK or int(inp[r, c]) != BLACK or _near_any(r, c, protected, 2):
                continue
            if all(
                not _in_bounds(inp.shape, r + dr, c + dc) or int(inp[r + dr, c + dc]) == BLACK
                for dr, dc in NEIGHBORS4
            ):
                candidates.append((r, c))

    rng.shuffle(candidates)
    for r, c in candidates[: int(count)]:
        inp[r, c] = int(rng.choice(colors))


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    color_pool = [1, 2, 3, 4, 5, 6, 7, 9]
    color_a, color_b = rng.sample(color_pool, 2)
    layout = _layout_regions(rng, color_a, color_b)
    if layout is None:
        return None
    base, mask_a, mask_b = layout

    candidates_a = _solid_3x3_candidates(base, mask_a, color_a)
    candidates_b = _solid_3x3_candidates(base, mask_b, color_b)
    if len(candidates_a) < 4 or len(candidates_b) < 4:
        return None

    n_a = rng.randint(*DENTS_PER_REGION)
    n_b = rng.randint(*DENTS_PER_REGION)
    dents_a = _choose_spaced(candidates_a, n_a, [], rng)
    if dents_a is None:
        return None
    dents_b = _choose_spaced(candidates_b, n_b, dents_a, rng)
    if dents_b is None:
        return None

    dents = [(r, c, color_a) for r, c in dents_a] + [(r, c, color_b) for r, c in dents_b]
    protected = dents_a + dents_b
    clean_with_dents = base.copy()
    for r, c, _host in dents:
        clean_with_dents[r, c] = BLACK

    expected = _stamp_crosses(clean_with_dents, dents, (color_a, color_b))
    inp = clean_with_dents.copy()

    _add_protrusions(inp, base, color_a, rng.randint(*PROTRUSION_RANGE), protected, rng)
    _add_protrusions(inp, base, color_b, rng.randint(*PROTRUSION_RANGE), protected, rng)
    _add_isolated_wrong(inp, base, color_a, color_b, rng.randint(*ISOLATED_WRONG_RANGE), protected, rng)
    _add_isolated_wrong(inp, base, color_b, color_a, rng.randint(*ISOLATED_WRONG_RANGE), protected, rng)
    _add_background_specks(inp, base, (color_a, color_b), rng.randint(*SPECK_RANGE), protected, rng)

    try:
        produced = main(inp)
    except Exception:
        return None
    if not np.array_equal(produced, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid grid")
