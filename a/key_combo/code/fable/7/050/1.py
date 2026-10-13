from __future__ import annotations

import random
from collections import Counter

import numpy as np


BBox = tuple[int, int, int, int]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _bbox(mask: np.ndarray) -> BBox | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_area(box: BBox) -> int:
    r0, c0, r1, c1 = box
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _rotate180(r: int, c: int, sr: int, sc: int) -> tuple[int, int]:
    return int(sr - r), int(sc - c)


def _solid_rectangle_candidates(grid: np.ndarray, bg: int) -> list[BBox]:
    candidates: list[BBox] = []
    for color in sorted(set(map(int, grid.flat)) - {int(bg)}):
        mask = grid == int(color)
        box = _bbox(mask)
        if box is None:
            continue
        if int(mask.sum()) != _bbox_area(box):
            continue
        candidates.append(box)
    candidates.sort(key=_bbox_area, reverse=True)
    return candidates


def _try_center(grid: np.ndarray, domain: BBox, occ: BBox, sr: int, sc: int) -> tuple[np.ndarray, int] | None:
    r0, c0, r1, c1 = domain
    or0, oc0, or1, oc1 = occ
    occ_mask = np.zeros_like(grid, dtype=bool)
    occ_mask[or0 : or1 + 1, oc0 : oc1 + 1] = True

    mapping: dict[int, int] = {}
    visible_pairs = 0

    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            rr, cc = _rotate180(r, c, sr, sc)
            if not (r0 <= rr <= r1 and c0 <= cc <= c1):
                return None
            if occ_mask[r, c]:
                continue
            if occ_mask[rr, cc]:
                continue

            a = int(grid[r, c])
            b = int(grid[rr, cc])
            prev = mapping.get(a)
            if prev is None:
                mapping[a] = b
            elif prev != b:
                return None
            visible_pairs += 1

    if not mapping:
        return None

    for a, b in mapping.items():
        if mapping.get(b, a) != a:
            return None

    patch = np.empty((or1 - or0 + 1, oc1 - oc0 + 1), dtype=int)
    for r in range(or0, or1 + 1):
        for c in range(oc0, oc1 + 1):
            rr, cc = _rotate180(r, c, sr, sc)
            if not (r0 <= rr <= r1 and c0 <= cc <= c1):
                return None
            if occ_mask[rr, cc]:
                return None
            src = int(grid[rr, cc])
            if src not in mapping:
                return None
            patch[r - or0, c - oc0] = mapping[src]

    score = visible_pairs + 5 * len(mapping)
    return patch, score


def _solve(grid: np.ndarray) -> np.ndarray:
    bg = _get_bg(grid)
    domain = _bbox(grid != bg)
    assert domain is not None, "grid must contain a non-background pattern"
    dr0, dc0, dr1, dc1 = domain

    solutions: list[tuple[int, bytes, np.ndarray]] = []
    for occ in _solid_rectangle_candidates(grid, bg):
        or0, oc0, or1, oc1 = occ
        if not (dr0 <= or0 <= or1 <= dr1 and dc0 <= oc0 <= oc1 <= dc1):
            continue
        for sr in range(2 * dr0, 2 * dr1 + 1):
            for sc in range(2 * dc0, 2 * dc1 + 1):
                solved = _try_center(grid, domain, occ, sr, sc)
                if solved is None:
                    continue
                patch, score = solved
                solutions.append((score, patch.tobytes(), patch))

    assert solutions, "failed to infer a valid occluder patch"
    best_score = max(score for score, _sig, _patch in solutions)
    best = [patch for score, _sig, patch in solutions if score == best_score]
    unique: dict[bytes, np.ndarray] = {patch.tobytes(): patch for patch in best}
    assert len(unique) == 1, "solution should be unique"
    return next(iter(unique.values()))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the occluded rectangle patch by undoing a hidden 180-degree rotate-plus-recolor symmetry.

    Input structure:
    - One solid monochrome rectangle is an occluder whose color appears nowhere else.
    - The remaining colored area belongs to one hidden rectangular motif that is symmetric under a 180-degree rotation combined with an involutive color swap.
    - The symmetry center is not marked and may sit on a cell center or on a half-cell intersection.

    Rule:
    - Find the occluder rectangle candidate, the hidden rotation center, and the color involution from visible rotated pixel pairs.
    - For each occluded cell, rotate it through the hidden center to its visible partner and then apply the inferred color mapping.
    - Output only the recovered occluder patch.

    Key insight:
    - Geometry alone is insufficient because opposite cells can swap colors, and color swapping alone is insufficient because the paired locations depend on the hidden center; both must be inferred together from visible pairings.
    """
    return _solve(np.asarray(grid, dtype=int))


def _random_involution(colors: list[int], require_fixed: bool) -> dict[int, int]:
    palette = list(colors)
    random.shuffle(palette)
    max_pairs = len(palette) // 2
    if require_fixed and 2 * max_pairs == len(palette):
        max_pairs -= 1
    max_pairs = max(1, max_pairs)
    pair_count = random.randint(1, max_pairs)

    mapping = {color: color for color in palette}
    swapped = random.sample(palette, 2 * pair_count)
    for i in range(0, len(swapped), 2):
        a = swapped[i]
        b = swapped[i + 1]
        mapping[a] = b
        mapping[b] = a

    if require_fixed:
        assert any(mapping[color] == color for color in palette)
    assert any(mapping[color] != color for color in palette)
    return mapping


def _build_support(h: int, w: int, colors: list[int], mapping: dict[int, int]) -> np.ndarray:
    fixed = [color for color in colors if mapping[color] == color]
    support = np.empty((h, w), dtype=int)

    for r in range(h):
        for c in range(w):
            rr = h - 1 - r
            cc = w - 1 - c
            if (r > rr) or (r == rr and c > cc):
                continue
            if r == rr and c == cc:
                support[r, c] = random.choice(fixed)
                continue
            color = random.choice(colors)
            support[r, c] = color
            support[rr, cc] = mapping[color]

    return support


def _rotated_local_box(h: int, w: int, box: BBox) -> BBox:
    r0, c0, r1, c1 = box
    return h - 1 - r1, w - 1 - c1, h - 1 - r0, w - 1 - c0


def _boxes_disjoint(a: BBox, b: BBox) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0


def _occluder_options(h: int, w: int) -> list[BBox]:
    options: list[BBox] = []
    max_h = min(4, h - 1)
    max_w = min(4, w - 1)
    for oh in range(1, max_h + 1):
        for ow in range(1, max_w + 1):
            if oh * ow < 2:
                continue
            for r0 in range(0, h - oh + 1):
                for c0 in range(0, w - ow + 1):
                    box = (r0, c0, r0 + oh - 1, c0 + ow - 1)
                    if _boxes_disjoint(box, _rotated_local_box(h, w, box)):
                        options.append(box)
    return options


def _needs_recolor(local_patch: np.ndarray, support: np.ndarray, box: BBox) -> bool:
    h, w = support.shape
    r0, c0, r1, c1 = box
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            rr = h - 1 - r
            cc = w - 1 - c
            if int(local_patch[r - r0, c - c0]) != int(support[rr, cc]):
                return True
    return False


def _place_support(bg: int, support: np.ndarray) -> tuple[np.ndarray, int, int]:
    h, w = support.shape
    for _ in range(80):
        H = random.randint(h + 6, min(30, h + 14))
        W = random.randint(w + 6, min(30, w + 14))
        if H * W <= 2 * h * w:
            continue
        top = random.randint(0, H - h)
        left = random.randint(0, W - w)
        if 2 * top + h - 1 == H - 1 and 2 * left + w - 1 == W - 1 and random.random() < 0.85:
            continue
        grid = np.full((H, W), bg, dtype=int)
        grid[top : top + h, left : left + w] = support
        return grid, top, left
    raise RuntimeError("failed to place support off-center")


def _gen_attempt() -> np.ndarray | None:
    bg = random.randrange(10)
    h = random.randint(6, 12)
    w = random.randint(6, 12)
    require_fixed = (h % 2 == 1) and (w % 2 == 1)

    usable = [color for color in range(10) if color != bg]
    occluder_color = random.choice(usable)
    palette_pool = [color for color in usable if color != occluder_color]
    palette_size = random.randint(3, min(6, len(palette_pool)))
    pattern_colors = random.sample(palette_pool, palette_size)

    mapping = _random_involution(pattern_colors, require_fixed)
    support = _build_support(h, w, pattern_colors, mapping)

    occluders = _occluder_options(h, w)
    if not occluders:
        return None
    occ = random.choice(occluders)
    r0, c0, r1, c1 = occ
    expected = support[r0 : r1 + 1, c0 : c1 + 1].copy()

    if len(set(map(int, expected.flat))) < 2:
        return None
    if random.random() < 0.75 and not _needs_recolor(expected, support, occ):
        return None

    inp_support = support.copy()
    inp_support[r0 : r1 + 1, c0 : c1 + 1] = occluder_color

    grid, top, left = _place_support(bg, inp_support)
    expected = expected.copy()

    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None

    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return None

    if _get_bg(grid) != bg:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hidden-center recolor-completion grid")
