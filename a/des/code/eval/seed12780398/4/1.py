import random
from collections import Counter

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
GEN_RETRIES = 400
PLACEMENT_TRIES = 300


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
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
                for nr, nc in ((cr + 1, cc), (cr - 1, cc), (cr, cc + 1), (cr, cc - 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _find_seed_bbox(grid: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    best_bbox = None
    best_key = None
    for comp in _components(grid != bg):
        r0, c0, r1, c1 = _bbox(comp)
        colors = {int(grid[r, c]) for r, c in comp}
        if len(colors) < 3:
            continue
        key = (r0 + c0, r0, c0, (r1 - r0 + 1) * (c1 - c0 + 1))
        if best_key is None or key < best_key:
            best_key = key
            best_bbox = (r0, c0, r1, c1)
    if best_bbox is None:
        raise ValueError("seed region not found")
    return best_bbox


def _seed_order(seed: np.ndarray) -> list[int]:
    h, w = seed.shape
    top = 0
    left = 0
    bottom = h - 1
    right = w - 1
    order: list[int] = []
    while top <= bottom and left <= right:
        color = int(seed[top, left])
        if not order or color != order[-1]:
            order.append(color)
        top += 1
        left += 1
        bottom -= 1
        right -= 1
    return order


def _factor_pairs(area: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    limit = int(area ** 0.5)
    for h in range(1, limit + 1):
        if area % h == 0:
            pairs.append((h, area // h))
    return pairs


def _best_factor_pair(area: int) -> tuple[int, int]:
    pairs = _factor_pairs(area)
    return min(pairs, key=lambda hw: (hw[0] + hw[1], hw[1] - hw[0], hw[0]))


def _reconstruct_structure(
    areas: list[int], max_h: int | None = None, max_w: int | None = None
) -> dict[str, int | tuple[int, ...]]:
    if not areas or any(area <= 0 for area in areas):
        raise ValueError("invalid layer areas")

    candidates = []
    inner_area = int(areas[-1])
    for inner_h, inner_w in _factor_pairs(inner_area):
        cur_h = int(inner_h)
        cur_w = int(inner_w)
        thick_rev: list[int] = []
        valid = True
        for frame_area in areas[-2::-1]:
            found = None
            limit = int(np.sqrt(frame_area)) + max(cur_h, cur_w) + 2
            for thickness in range(1, limit + 1):
                used = 2 * thickness * (cur_h + cur_w) + 4 * thickness * thickness
                if used == frame_area:
                    found = thickness
                    break
            if found is None:
                valid = False
                break
            thick_rev.append(found)
            cur_h += 2 * found
            cur_w += 2 * found

        if not valid:
            continue
        if max_h is not None and cur_h > max_h:
            continue
        if max_w is not None and cur_w > max_w:
            continue

        thicknesses = tuple(reversed(thick_rev))
        candidates.append(
            (
                cur_h + cur_w,
                cur_w - cur_h,
                cur_h,
                cur_w,
                inner_h + inner_w,
                inner_w - inner_h,
                thicknesses,
                inner_h,
                inner_w,
            )
        )

    if not candidates:
        raise ValueError("no valid concentric structure")

    candidates.sort()
    _, _, outer_h, outer_w, _, _, thicknesses, inner_h, inner_w = candidates[0]
    return {
        "outer_h": int(outer_h),
        "outer_w": int(outer_w),
        "inner_h": int(inner_h),
        "inner_w": int(inner_w),
        "thicknesses": tuple(int(t) for t in thicknesses),
    }


def _render_structure(
    shape: tuple[int, int],
    bg: int,
    colors: list[int],
    outer_h: int,
    outer_w: int,
    thicknesses: tuple[int, ...],
) -> np.ndarray:
    h, w = shape
    out = np.full((h, w), bg, dtype=int)
    top = (h - outer_h) // 2
    left = (w - outer_w) // 2
    bottom = top + outer_h - 1
    right = left + outer_w - 1

    for idx, color in enumerate(colors):
        if idx == len(colors) - 1:
            out[top : bottom + 1, left : right + 1] = int(color)
            break

        thickness = int(thicknesses[idx])
        out[top : top + thickness, left : right + 1] = int(color)
        out[bottom - thickness + 1 : bottom + 1, left : right + 1] = int(color)
        out[top : bottom + 1, left : left + thickness] = int(color)
        out[top : bottom + 1, right - thickness + 1 : right + 1] = int(color)
        top += thickness
        left += thickness
        bottom -= thickness
        right -= thickness

    return out


def _seedless_color_areas(
    grid: np.ndarray, seed_bbox: tuple[int, int, int, int], colors: list[int]
) -> list[int]:
    r0, c0, r1, c1 = seed_bbox
    mask = np.ones(grid.shape, dtype=bool)
    mask[r0 : r1 + 1, c0 : c1 + 1] = False
    return [int(np.sum((grid == color) & mask)) for color in colors]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the nested-color order from the seed region, sum same-color fragment areas,
    then rebuild a centered concentric frame structure whose layer areas match those sums.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _background_color(g)

    seed_bbox = _find_seed_bbox(g, bg)
    r0, c0, r1, c1 = seed_bbox
    colors = _seed_order(g[r0 : r1 + 1, c0 : c1 + 1])
    areas = _seedless_color_areas(g, seed_bbox, colors)
    spec = _reconstruct_structure(areas, max_h=h, max_w=w)
    return _render_structure(
        (h, w),
        bg,
        colors,
        int(spec["outer_h"]),
        int(spec["outer_w"]),
        tuple(spec["thicknesses"]),
    )


def _areas_from_spec(center_h: int, center_w: int, thicknesses: list[int]) -> tuple[list[int], int, int]:
    cur_h = int(center_h)
    cur_w = int(center_w)
    center_area = cur_h * cur_w
    frame_areas_rev: list[int] = []
    for thickness in reversed(thicknesses):
        frame_area = 2 * thickness * (cur_h + cur_w) + 4 * thickness * thickness
        frame_areas_rev.append(int(frame_area))
        cur_h += 2 * thickness
        cur_w += 2 * thickness
    areas = list(reversed(frame_areas_rev)) + [int(center_area)]
    return areas, cur_h, cur_w


def _build_seed(colors: list[int]) -> np.ndarray:
    size = 2 * len(colors) - 1
    seed = np.full((size, size), int(colors[-1]), dtype=int)
    for idx, color in enumerate(colors[:-1]):
        lo = idx
        hi = size - 1 - idx
        seed[lo : hi + 1, lo] = int(color)
        seed[lo : hi + 1, hi] = int(color)
        seed[lo, lo : hi + 1] = int(color)
        seed[hi, lo : hi + 1] = int(color)
    return seed


def _split_base_rectangle(height: int, width: int, pieces: int, rng: random.Random) -> list[tuple[int, int]]:
    rects = [(int(height), int(width))]
    while len(rects) < pieces:
        splittable = [idx for idx, (rh, rw) in enumerate(rects) if rh > 1 or rw > 1]
        if not splittable:
            break
        idx = rng.choice(splittable)
        rh, rw = rects.pop(idx)

        if rh == 1:
            cut = rng.randint(1, rw - 1)
            rects.extend([(1, cut), (1, rw - cut)])
        elif rw == 1:
            cut = rng.randint(1, rh - 1)
            rects.extend([(cut, 1), (rh - cut, 1)])
        elif rng.random() < 0.5:
            cut = rng.randint(1, rh - 1)
            rects.extend([(cut, rw), (rh - cut, rw)])
        else:
            cut = rng.randint(1, rw - 1)
            rects.extend([(rh, cut), (rh, rw - cut)])

    out: list[tuple[int, int]] = []
    for rh, rw in rects:
        if rng.random() < 0.5:
            rh, rw = rw, rh
        out.append((int(rh), int(rw)))
    return out


def _fragments_for_area(area: int, rng: random.Random) -> list[tuple[int, int]]:
    base_h, base_w = _best_factor_pair(area)
    max_pieces = min(4, area)
    if area <= 3:
        pieces = 1
    else:
        pieces = rng.randint(1, max_pieces)
    return _split_base_rectangle(base_h, base_w, pieces, rng)


def _choose_grid_shape(
    outer_h: int,
    outer_w: int,
    seed_h: int,
    seed_w: int,
    pieces: list[tuple[int, int, int]],
    total_area: int,
    rng: random.Random,
) -> tuple[int, int] | None:
    inflated = sum((ph + 2) * (pw + 2) for ph, pw, _ in pieces) + (seed_h + 2) * (seed_w + 2)
    h_min = max(18, outer_h + 4, seed_h + 2, max(ph for ph, _, _ in pieces) + 4)
    w_min = max(18, outer_w + 8, seed_w + 6, max(pw for _, pw, _ in pieces) + 4)
    if h_min > 30 or w_min > 30:
        return None

    for _ in range(80):
        h = rng.randint(h_min, 30)
        w = rng.randint(w_min, 30)
        if total_area + seed_h * seed_w >= (h * w) // 2:
            continue
        if inflated > h * w:
            continue
        return h, w
    return None


def _place_fragments(
    h: int,
    w: int,
    seed: np.ndarray,
    seed_r: int,
    seed_c: int,
    pieces: list[tuple[int, int, int]],
    rng: random.Random,
) -> np.ndarray | None:
    grid = np.full((h, w), BG, dtype=int)
    occ = np.zeros((h, w), dtype=bool)
    sh, sw = seed.shape
    grid[seed_r : seed_r + sh, seed_c : seed_c + sw] = seed
    occ[seed_r : seed_r + sh, seed_c : seed_c + sw] = True

    for ph, pw, color in sorted(pieces, key=lambda item: item[0] * item[1], reverse=True):
        placed = False
        if ph >= h - 1 or pw >= w - 1:
            return None
        for _ in range(PLACEMENT_TRIES):
            r = rng.randint(1, h - ph - 1)
            c = rng.randint(1, w - pw - 1)
            if r < seed_r + sh + 2 and c < seed_c + sw + 2:
                continue
            r0 = max(0, r - 1)
            c0 = max(0, c - 1)
            r1 = min(h, r + ph + 1)
            c1 = min(w, c + pw + 1)
            if occ[r0:r1, c0:c1].any():
                continue
            grid[r : r + ph, c : c + pw] = int(color)
            occ[r : r + ph, c : c + pw] = True
            placed = True
            break
        if not placed:
            return None

    return grid


def _sample_structure(rng: random.Random) -> tuple[list[int], list[int], int, int, list[int]] | None:
    layer_count = rng.choice((3, 4))
    colors = rng.sample(PALETTE, layer_count)

    for _ in range(120):
        center_h = rng.randint(2, 4 if layer_count == 4 else 5)
        center_w = rng.randint(center_h, 6 if layer_count == 4 else 7)
        if _best_factor_pair(center_h * center_w) != (center_h, center_w):
            continue

        thicknesses = [rng.choice((1, 2)) for _ in range(layer_count - 1)]
        areas, outer_h, outer_w = _areas_from_spec(center_h, center_w, thicknesses)
        if outer_h > 16 or outer_w > 16:
            continue
        if sum(areas) > 170:
            continue

        spec = _reconstruct_structure(areas)
        recovered = (
            int(spec["inner_h"]),
            int(spec["inner_w"]),
            tuple(spec["thicknesses"]),
            int(spec["outer_h"]),
            int(spec["outer_w"]),
        )
        expected = (center_h, center_w, tuple(thicknesses), outer_h, outer_w)
        if recovered != expected:
            continue
        return colors, areas, outer_h, outer_w, thicknesses

    return None


def generate_input() -> np.ndarray:
    rng = random.Random()

    for _ in range(GEN_RETRIES):
        sampled = _sample_structure(rng)
        if sampled is None:
            continue
        colors, areas, outer_h, outer_w, thicknesses = sampled

        seed = _build_seed(colors)
        seed_h, seed_w = seed.shape
        pieces: list[tuple[int, int, int]] = []
        for color, area in zip(colors, areas):
            for ph, pw in _fragments_for_area(area, rng):
                pieces.append((ph, pw, int(color)))

        shape = _choose_grid_shape(outer_h, outer_w, seed_h, seed_w, pieces, sum(areas), rng)
        if shape is None:
            continue
        h, w = shape

        seed_r = rng.randint(0, 1)
        seed_c = rng.randint(0, 1)
        if seed_r + seed_h > h or seed_c + seed_w > w:
            continue

        grid = _place_fragments(h, w, seed, seed_r, seed_c, pieces, rng)
        if grid is None:
            continue

        expected = _render_structure((h, w), BG, colors, outer_h, outer_w, tuple(thicknesses))
        try:
            actual = main(grid)
        except Exception:
            continue
        if not np.array_equal(actual, expected):
            continue

        counts = Counter(map(int, grid.reshape(-1)))
        if counts[BG] <= max(counts[color] for color in colors):
            continue

        return grid

    raise RuntimeError("generate_input: exceeded retries")
