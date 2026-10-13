import random
from typing import Optional

import numpy as np


BG = 7
COLOR_POOL = (1, 2, 3, 4, 5, 6, 8, 9)
GEN_ATTEMPTS = 300
HEIGHT_RANGE = (12, 24)
WIDTH_RANGE = (12, 24)
OBJECT_COUNT_RANGE = (2, 6)
SIDE_CHOICES = (3, 5, 7, 9)
GAP = 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-colored aligned pixels and replace each pair by a centered solid square.

    The two input pixels for a color define the square center and size:
    they lie on one row or one column with a background midpoint between them.
    The inclusive span between the two pixels is therefore odd and is used as
    the square side length. The original pixels are erased to background.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"
    out = g.copy()

    colors = sorted(int(c) for c in np.unique(g) if int(c) != BG)
    for color in colors:
        pts = np.argwhere(g == color)
        assert pts.shape == (2, 2), "each foreground color must appear exactly twice"
        (r0, c0), (r1, c1) = pts.tolist()
        assert r0 == r1 or c0 == c1, "paired pixels must be aligned"

        if r0 == r1:
            span = abs(c1 - c0) + 1
            center_r = r0
            center_c = (c0 + c1) // 2
        else:
            span = abs(r1 - r0) + 1
            center_r = (r0 + r1) // 2
            center_c = c0

        assert span % 2 == 1, "aligned pair must have an integer midpoint cell"
        half = span // 2

        out[r0, c0] = BG
        out[r1, c1] = BG

        rr0 = max(0, center_r - half)
        rr1 = min(g.shape[0], center_r + half + 1)
        cc0 = max(0, center_c - half)
        cc1 = min(g.shape[1], center_c + half + 1)
        out[rr0:rr1, cc0:cc1] = color

    return out


def _bbox_intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _sample_center(
    h: int, w: int, side: int, horizontal: bool, clip: bool, rng: random.Random
) -> Optional[tuple[int, int]]:
    half = side // 2
    if horizontal:
        c = rng.randint(half, w - 1 - half)
        if clip and h > side:
            if rng.random() < 0.5:
                r = rng.randint(0, half - 1)
            else:
                r = rng.randint(h - half, h - 1)
        else:
            r = rng.randint(half, h - 1 - half)
        return (r, c)

    r = rng.randint(half, h - 1 - half)
    if clip and w > side:
        if rng.random() < 0.5:
            c = rng.randint(0, half - 1)
        else:
            c = rng.randint(w - half, w - 1)
    else:
        c = rng.randint(half, w - 1 - half)
    return (r, c)


def _try_place_object(
    h: int,
    w: int,
    side: int,
    horizontal: bool,
    clip: bool,
    reserved: list[tuple[int, int, int, int]],
    endpoints: set[tuple[int, int]],
    rng: random.Random,
) -> Optional[dict]:
    half = side // 2
    for _ in range(120):
        center = _sample_center(h, w, side, horizontal, clip, rng)
        if center is None:
            continue
        r, c = center
        if horizontal:
            p0 = (r, c - half)
            p1 = (r, c + half)
        else:
            p0 = (r - half, c)
            p1 = (r + half, c)

        rr0 = max(0, r - half)
        rr1 = min(h - 1, r + half)
        cc0 = max(0, c - half)
        cc1 = min(w - 1, c + half)
        bbox = (rr0, cc0, rr1, cc1)

        if p0 in endpoints or p1 in endpoints:
            continue
        if any(_bbox_intersects(bbox, prev, pad=GAP) for prev in reserved):
            continue
        return {
            "side": side,
            "horizontal": horizontal,
            "center": center,
            "endpoints": (p0, p1),
            "bbox": bbox,
            "clipped": clip and (rr0 == 0 or rr1 == h - 1 or cc0 == 0 or cc1 == w - 1),
        }
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_ATTEMPTS} attempts")


def _gen_attempt() -> Optional[np.ndarray]:
    rng = random.Random()
    h = rng.randint(*HEIGHT_RANGE)
    w = rng.randint(*WIDTH_RANGE)
    side_limit = min(max(SIDE_CHOICES), h if h % 2 == 1 else h - 1, w if w % 2 == 1 else w - 1)
    valid_sides = [side for side in SIDE_CHOICES if side <= side_limit]
    if not valid_sides:
        return None

    max_objects = min(OBJECT_COUNT_RANGE[1], len(COLOR_POOL), (h * w) // 20)
    min_objects = min(OBJECT_COUNT_RANGE[0], max_objects)
    if min_objects == 0:
        return None
    n = rng.randint(min_objects, max_objects)

    colors = list(COLOR_POOL)
    rng.shuffle(colors)
    colors = colors[:n]

    horizontal_count = n // 2
    orientations = [True] * horizontal_count + [False] * (n - horizontal_count)
    rng.shuffle(orientations)

    want_clipped = rng.random() < 0.5
    clipped_index = rng.randrange(n) if want_clipped else None

    specs = []
    reserved: list[tuple[int, int, int, int]] = []
    endpoints: set[tuple[int, int]] = set()

    for idx, horizontal in enumerate(orientations):
        choices = valid_sides[:]
        rng.shuffle(choices)
        placed = None
        for side in choices:
            clip = idx == clipped_index
            candidate = _try_place_object(h, w, side, horizontal, clip, reserved, endpoints, rng)
            if candidate is None:
                continue
            placed = candidate
            break
        if placed is None:
            return None
        specs.append(placed)
        reserved.append(placed["bbox"])
        endpoints.update(placed["endpoints"])

    if want_clipped and not any(spec["clipped"] for spec in specs):
        return None
    if not any(spec["horizontal"] for spec in specs):
        return None
    if not any(not spec["horizontal"] for spec in specs):
        return None

    grid = np.full((h, w), BG, dtype=int)
    for color, spec in zip(colors, specs, strict=False):
        (r0, c0), (r1, c1) = spec["endpoints"]
        grid[r0, c0] = color
        grid[r1, c1] = color

        cr, cc = spec["center"]
        if grid[cr, cc] != BG:
            return None

    if len(np.unique(grid)) != n + 1:
        return None
    return grid
