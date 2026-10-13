import random
from collections import Counter

import numpy as np


BLACK = 0
TEAL = 8
GEN_RETRIES = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _color_components(grid: np.ndarray, color: int):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _principal_colors(grid: np.ndarray) -> tuple[int, int]:
    counts = Counter(int(v) for v in grid.flat if int(v) != BLACK)
    assert len(counts) >= 2, "expected at least two foreground colors"
    colors = [color for color, _ in counts.most_common()]
    return colors[0], colors[1]


def _cleanup(grid: np.ndarray, colors: tuple[int, int]) -> np.ndarray:
    out = grid.copy()
    h, w = grid.shape
    for color in colors:
        other = colors[1] if color == colors[0] else colors[0]
        for comp in _color_components(grid, color):
            if len(comp) != 1:
                continue
            r, c = comp[0]
            around = []
            for nr, nc in _neighbors8(r, c):
                if 0 <= nr < h and 0 <= nc < w:
                    around.append(int(grid[nr, nc]))
            other_count = sum(v == other for v in around)
            same_orth = 0
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == other:
                    same_orth += 1
            if other_count >= 5 and same_orth >= 3:
                out[r, c] = other
            else:
                out[r, c] = BLACK
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    color_a, color_b = _principal_colors(g)
    colors = (color_a, color_b)
    cleaned = _cleanup(g, colors)
    out = cleaned.copy()
    h, w = g.shape

    dents = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(g[r, c]) != BLACK or int(cleaned[r, c]) != BLACK:
                continue
            ring = [int(cleaned[nr, nc]) for nr, nc in _neighbors8(r, c)]
            orth = [int(cleaned[nr, nc]) for nr, nc in _neighbors4(r, c)]
            for host, fill in ((color_a, color_b), (color_b, color_a)):
                if all(v == host for v in orth) and sum(v == host for v in ring) >= 7 and sum(v == fill for v in ring) == 0:
                    dents.append((r, c, fill))
                    break

    for r, c, fill in dents:
        out[r - 1 : r + 2, c - 1 : c + 2] = fill
        out[r, c] = TEAL
    return out


def _rand_color_pair(rng: random.Random) -> tuple[int, int]:
    colors = [1, 2, 3, 4, 5, 6, 7, 9]
    rng.shuffle(colors)
    return colors[0], colors[1]


def _bbox(mask: np.ndarray):
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _add_rectangle(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int):
    mask[r0 : r1 + 1, c0 : c1 + 1] = True


def _rand_inclusive(rng: random.Random, lo: int, hi: int) -> int | None:
    if lo > hi:
        return None
    return rng.randint(lo, hi)


def _make_region_mask(h: int, w: int, rng: random.Random, orient: str, side: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    if orient == "horizontal":
        left_limit = w // 2 - 2
        right_start = w // 2 + 2
        left_room = left_limit - 2 + 1
        right_room = (w - 3) - right_start + 1
        if left_room < 6 or right_room < 6:
            return mask
        rect_h = rng.randint(7, max(7, h - 6))
        rect_w = rng.randint(6, max(6, min(left_room, right_room) - 1))
        r0 = _rand_inclusive(rng, 2, h - rect_h - 3)
        if r0 is None:
            return mask
        if side == 0:
            c0 = _rand_inclusive(rng, 2, left_limit - rect_w)
        else:
            c0 = _rand_inclusive(rng, right_start, w - rect_w - 3)
        if c0 is None:
            return mask
    else:
        top_limit = h // 2 - 2
        bottom_start = h // 2 + 2
        top_room = top_limit - 2 + 1
        bottom_room = (h - 3) - bottom_start + 1
        if top_room < 6 or bottom_room < 6:
            return mask
        rect_h = rng.randint(6, max(6, min(top_room, bottom_room) - 1))
        rect_w = rng.randint(7, max(7, w - 6))
        c0 = _rand_inclusive(rng, 2, w - rect_w - 3)
        if c0 is None:
            return mask
        if side == 0:
            r0 = _rand_inclusive(rng, 2, top_limit - rect_h)
        else:
            r0 = _rand_inclusive(rng, bottom_start, h - rect_h - 3)
        if r0 is None:
            return mask
    r1 = r0 + rect_h - 1
    c1 = c0 + rect_w - 1
    _add_rectangle(mask, r0, c0, r1, c1)

    for _ in range(rng.randint(1, 3)):
        edge = rng.choice(["top", "bottom", "left", "right"])
        if edge in ("top", "bottom"):
            ext_h = rng.randint(1, 2)
            ext_w = rng.randint(3, max(3, rect_w - 1))
            sc0 = rng.randint(c0, c1 - ext_w + 1)
            if edge == "top" and r0 - ext_h >= 1:
                _add_rectangle(mask, r0 - ext_h, sc0, r0 - 1, sc0 + ext_w - 1)
            if edge == "bottom" and r1 + ext_h <= h - 2:
                _add_rectangle(mask, r1 + 1, sc0, r1 + ext_h, sc0 + ext_w - 1)
        else:
            ext_h = rng.randint(3, max(3, rect_h - 1))
            ext_w = rng.randint(1, 2)
            sr0 = rng.randint(r0, r1 - ext_h + 1)
            if edge == "left" and c0 - ext_w >= 1:
                _add_rectangle(mask, sr0, c0 - ext_w, sr0 + ext_h - 1, c0 - 1)
            if edge == "right" and c1 + ext_w <= w - 2:
                _add_rectangle(mask, sr0, c1 + 1, sr0 + ext_h - 1, c1 + ext_w)

    r0, c0, r1, c1 = _bbox(mask)
    for _ in range(rng.randint(0, 2)):
        corner = rng.choice(["tl", "tr", "bl", "br"])
        cut_h = rng.randint(1, 2)
        cut_w = rng.randint(1, 2)
        if corner == "tl":
            mask[r0 : r0 + cut_h, c0 : c0 + cut_w] = False
        elif corner == "tr":
            mask[r0 : r0 + cut_h, c1 - cut_w + 1 : c1 + 1] = False
        elif corner == "bl":
            mask[r1 - cut_h + 1 : r1 + 1, c0 : c0 + cut_w] = False
        else:
            mask[r1 - cut_h + 1 : r1 + 1, c1 - cut_w + 1 : c1 + 1] = False

    comps = _color_components(mask.astype(int), 1)
    if len(comps) != 1:
        return np.zeros((h, w), dtype=bool)
    return mask


def _dent_candidates(mask: np.ndarray):
    h, w = mask.shape
    pts = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if np.all(mask[r - 1 : r + 2, c - 1 : c + 2]):
                pts.append((r, c))
    return pts


def _place_regions(h: int, w: int, rng: random.Random):
    orient = rng.choice(["horizontal", "vertical"])
    a = _make_region_mask(h, w, rng, orient, 0)
    b = _make_region_mask(h, w, rng, orient, 1)
    if not a.any() or not b.any():
        return None
    if np.any(a & b):
        return None
    ar0, ac0, ar1, ac1 = _bbox(a)
    br0, bc0, br1, bc1 = _bbox(b)
    if not (ar1 + 2 < br0 or br1 + 2 < ar0 or ac1 + 2 < bc0 or bc1 + 2 < ac0):
        return None
    if a.sum() < 35 or b.sum() < 35:
        return None
    if len(_dent_candidates(a)) < 1 or len(_dent_candidates(b)) < 1:
        return None
    return a, b


def _sample_separated(points, k: int, rng: random.Random):
    rng.shuffle(points)
    chosen = []
    for r, c in points:
        if all(max(abs(r - rr), abs(c - cc)) >= 3 for rr, cc in chosen):
            chosen.append((r, c))
            if len(chosen) == k:
                break
    return chosen


def _apply_noise(grid: np.ndarray, region_mask: np.ndarray, region_color: int, other_color: int, rng: random.Random):
    candidates = _dent_candidates(region_mask)
    protrusions = _sample_separated(candidates[:], rng.randint(0, 2), rng)
    for r, c in protrusions:
        if int(grid[r, c]) == region_color:
            grid[r, c] = other_color


def _place_speckles(grid: np.ndarray, rng: random.Random, colors: tuple[int, int]):
    h, w = grid.shape
    for _ in range(rng.randint(0, 3)):
        spots = []
        for r in range(1, h - 1):
            for c in range(1, w - 1):
                if int(grid[r, c]) != BLACK:
                    continue
                if any(int(grid[nr, nc]) != BLACK for nr, nc in _neighbors8(r, c)):
                    continue
                spots.append((r, c))
        if not spots:
            break
        r, c = rng.choice(spots)
        grid[r, c] = rng.choice(colors)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(16, 24)
    w = rng.randint(16, 24)
    region_data = _place_regions(h, w, rng)
    if region_data is None:
        return None
    mask_a, mask_b = region_data
    color_a, color_b = _rand_color_pair(rng)
    grid = np.zeros((h, w), dtype=int)
    grid[mask_a] = color_a
    grid[mask_b] = color_b

    dents_total = 0
    for mask, color, other in ((mask_a, color_a, color_b), (mask_b, color_b, color_a)):
        count = rng.randint(1, 3)
        chosen = _sample_separated(_dent_candidates(mask), count, rng)
        if len(chosen) != count:
            return None
        for r, c in chosen:
            grid[r, c] = BLACK
        dents_total += len(chosen)
        _apply_noise(grid, mask, color, other, rng)

    if dents_total < 2:
        return None

    _place_speckles(grid, rng, (color_a, color_b))

    out = main(grid)
    teal_count = int(np.count_nonzero(out == TEAL))
    if teal_count != dents_total:
        return None
    if len(_color_components(grid, color_a)) < 1 or len(_color_components(grid, color_b)) < 1:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
