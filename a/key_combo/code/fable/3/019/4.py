from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
PALETTE = tuple(range(10))
NON_BLACK = tuple(range(1, 10))
MAX_GEN_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _orbit_from_ul(size: int, r: int, c: int) -> list[tuple[int, int]]:
    n = int(size)
    return [
        (int(r), int(c)),
        (int(c), n - 1 - int(r)),
        (n - 1 - int(r), n - 1 - int(c)),
        (n - 1 - int(c), int(r)),
    ]


def _fold_defects(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h == w and h % 2 == 0, "expected an even square grid"

    q = h // 2
    folded = np.zeros((q, q), dtype=int)
    defect_count = 0

    for r in range(q):
        for c in range(q):
            orbit = _orbit_from_ul(h, r, c)
            colors = [int(g[rr, cc]) for rr, cc in orbit]
            counts = Counter(colors)
            if len(counts) == 1:
                continue

            freq = sorted(counts.values())
            assert freq == [1, 3], "each rotational orbit must be either clean or have one odd pixel"
            defect_color = next(color for color, count in counts.items() if count == 1)
            folded[r, c] = defect_color
            defect_count += 1

    assert defect_count > 0, "expected at least one defect orbit"
    return folded


def _mirror_complete(left_half: np.ndarray) -> np.ndarray:
    return np.concatenate([left_half, np.fliplr(left_half)], axis=1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the odd-color rotational fragments, fold them into the upper-left quarter, then mirror-complete the hidden figure.

    Input structure:
    - The input is an even-sized square grid whose clean underlying pattern is 90-degree rotationally symmetric.
    - Every 4-cell rotation orbit is either perfectly clean or has exactly one odd-colored defect cell against three matching base cells.
    - Folding each defect orbit back into the upper-left quadrant produces the left half of the target figure.

    Rule:
    - For every orbit under 90/180/270-degree rotation around the grid center, ignore clean 4-of-a-kind orbits.
    - In every 3-vs-1 orbit, keep the odd color and place it at the unique upper-left-quadrant representative of that orbit.
    - Crop the folded fragment set to its tight bounding box, mirror it horizontally to reconstruct the full bilateral figure, and return the result on black.

    Key insight:
    - The visible noise pixels are not independent: majority vote across each rotation orbit reveals a hidden left half, and a second symmetry step turns that half into the final object.
    """
    folded = _fold_defects(np.asarray(grid, dtype=int))
    bbox = _bbox(folded != BLACK)
    assert bbox is not None, "folded fragment set must be nonempty"
    left_half = _crop_bbox(folded, bbox)
    return _mirror_complete(left_half)


def _paint_patch(quarter: np.ndarray, color: int) -> None:
    q = quarter.shape[0]
    kind = random.choice(["rect", "hband", "vband", "diag"])
    if kind == "rect":
        h = random.randint(1, max(1, q // 2))
        w = random.randint(1, max(1, q // 2))
        r0 = random.randint(0, q - h)
        c0 = random.randint(0, q - w)
        quarter[r0 : r0 + h, c0 : c0 + w] = color
        return

    if kind == "hband":
        h = random.randint(1, max(1, q // 3))
        r0 = random.randint(0, q - h)
        c0 = random.randint(0, q - 1)
        c1 = random.randint(c0, q - 1)
        quarter[r0 : r0 + h, c0 : c1 + 1] = color
        return

    if kind == "vband":
        w = random.randint(1, max(1, q // 3))
        c0 = random.randint(0, q - w)
        r0 = random.randint(0, q - 1)
        r1 = random.randint(r0, q - 1)
        quarter[r0 : r1 + 1, c0 : c0 + w] = color
        return

    offset = random.randint(-(q - 1), q - 1)
    thickness = random.randint(0, max(0, q // 5))
    rr, cc = np.indices(quarter.shape)
    if random.random() < 0.5:
        mask = np.abs(rr - cc - offset) <= thickness
    else:
        mask = np.abs(rr + cc - (q - 1 + offset)) <= thickness
    quarter[mask] = color


def _generate_quarter(q: int, palette: list[int]) -> np.ndarray:
    bg = random.choice(palette)
    quarter = np.full((q, q), bg, dtype=int)
    accents = [color for color in palette if color != bg]
    assert accents, "palette must contain at least two colors"

    forced = random.sample(accents, k=random.randint(1, len(accents)))
    for color in forced:
        _paint_patch(quarter, color)

    for _ in range(random.randint(2, 5)):
        _paint_patch(quarter, random.choice(accents))

    if len(set(map(int, quarter.flatten()))) == 1:
        _paint_patch(quarter, accents[0])
    return quarter


def _grow_connected_mask(h: int, w: int, target: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    start = (random.randrange(h), random.randrange(w))
    mask[start] = True

    while int(mask.sum()) < target:
        frontier: list[tuple[int, int]] = []
        weights: list[int] = []
        for r in range(h):
            for c in range(w):
                if mask[r, c]:
                    continue
                adj = sum(1 for nr, nc in _neighbors4(r, c, h, w) if mask[nr, nc])
                if adj == 0:
                    continue
                frontier.append((r, c))
                weights.append(adj * adj)
        if not frontier:
            break
        rr, cc = random.choices(frontier, weights=weights, k=1)[0]
        mask[rr, cc] = True

    extra = random.randint(0, min(3, h * w - int(mask.sum())))
    for _ in range(extra):
        frontier = [
            (r, c)
            for r in range(h)
            for c in range(w)
            if not mask[r, c] and any(mask[nr, nc] for nr, nc in _neighbors4(r, c, h, w))
        ]
        if not frontier:
            break
        rr, cc = random.choice(frontier)
        mask[rr, cc] = True
    return mask


def _colorize_mask(mask: np.ndarray, palette: list[int]) -> np.ndarray:
    coords = [tuple(map(int, coord)) for coord in np.argwhere(mask)]
    n_colors = random.randint(1, min(3, len(palette), len(coords)))
    colors = random.sample(palette, k=n_colors)

    out = np.zeros(mask.shape, dtype=int)
    if n_colors == 1:
        out[mask] = colors[0]
        return out

    seeds = random.sample(coords, k=n_colors)
    for r, c in coords:
        best_idx = min(
            range(n_colors),
            key=lambda idx: (abs(r - seeds[idx][0]) + abs(c - seeds[idx][1]), idx),
        )
        out[r, c] = colors[best_idx]
    return out


def _generate_left_half(q: int, palette: list[int]) -> np.ndarray | None:
    for _ in range(200):
        h = random.randint(3, q)
        w = random.randint(2, q)
        area = h * w
        lo = max(5, area // 4)
        hi = max(lo, min(area - 1, (2 * area) // 3))
        mask = _grow_connected_mask(h, w, random.randint(lo, hi))
        bbox = _bbox(mask)
        if bbox is None:
            continue
        mask = _crop_bbox(mask.astype(int), bbox).astype(bool)
        hh, ww = mask.shape
        if hh < 2 or ww < 2 or int(mask.sum()) < 5:
            continue
        return _colorize_mask(mask, palette)
    return None


def _quarter_to_full(quarter: np.ndarray) -> np.ndarray:
    q = quarter.shape[0]
    n = 2 * q
    full = np.zeros((n, n), dtype=int)
    for r in range(q):
        for c in range(q):
            color = int(quarter[r, c])
            for rr, cc in _orbit_from_ul(n, r, c):
                full[rr, cc] = color
    return full


def _sample_palette() -> list[int]:
    while True:
        palette = random.sample(PALETTE, k=random.randint(3, 5))
        if sum(color != BLACK for color in palette) >= 2:
            return palette


def _build_scene() -> tuple[np.ndarray, np.ndarray] | None:
    q = random.randint(5, 15)
    palette = _sample_palette()
    quarter = _generate_quarter(q, palette)

    used_non_black = sorted({int(color) for color in quarter.flatten() if int(color) != BLACK})
    if not used_non_black:
        return None

    left_half = _generate_left_half(q, used_non_black)
    if left_half is None:
        return None

    hh, hw = left_half.shape
    off_r = random.randint(0, q - hh)
    off_c = random.randint(0, q - hw)

    folded_canvas = np.zeros((q, q), dtype=int)
    folded_canvas[off_r : off_r + hh, off_c : off_c + hw] = left_half

    for r_local, c_local in np.argwhere(folded_canvas != BLACK):
        r = int(r_local)
        c = int(c_local)
        color = int(folded_canvas[r, c])
        if int(quarter[r, c]) == color:
            choices = [candidate for candidate in palette if candidate != color]
            quarter[r, c] = random.choice(choices)

    inp = _quarter_to_full(quarter)
    n = inp.shape[0]
    for r_local, c_local in np.argwhere(folded_canvas != BLACK):
        r = int(r_local)
        c = int(c_local)
        color = int(folded_canvas[r, c])
        orbit = _orbit_from_ul(n, r, c)
        rr, cc = random.choice(orbit)
        inp[rr, cc] = color

    expected = _mirror_complete(left_half)
    return inp, expected


def _gen_attempt() -> np.ndarray | None:
    built = _build_scene()
    if built is None:
        return None
    inp, expected = built

    if inp.shape[0] != inp.shape[1] or inp.shape[0] % 2 != 0 or max(inp.shape) > 30:
        return None
    if max(expected.shape) > 30:
        return None

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(actual, np.zeros_like(actual)):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid folded-fragment symmetry puzzle")
