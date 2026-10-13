import random

import numpy as np


BG = 8
YELLOW = 4
MAROON = 9
MARKER_COLORS = (0, 1, 2, 3, 5, 6, 7)
GEN_RETRIES = 400


def _border_mask(shape: tuple[int, int], rect: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = rect
    mask = np.zeros(shape, dtype=bool)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _cover_counts(shape: tuple[int, int], rects: list[tuple[int, int, int, int]]) -> np.ndarray:
    counts = np.zeros(shape, dtype=int)
    for r0, c0, r1, c1 in rects:
        counts[r0 : r1 + 1, c0 : c1 + 1] += 1
    return counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group scattered singleton markers by color, infer each color's bounding-box rectangle,
    then color rectangle overlaps.

    Input structure:
    - Background is azure.
    - Each foreground color appears as isolated single-pixel markers.
    - Yellow and maroon do not appear in the input.

    Rule:
    - Each color defines a virtual rectangle via the bbox of all its markers.
    - Keep the original markers.
    - Draw each rectangle's one-pixel border on background cells in that color.
    - Background cells covered by exactly two rectangles become yellow.
    - Background cells covered by three or more rectangles become maroon.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    rects: list[tuple[int, int, int, int, int]] = []
    for color in np.unique(g):
        color = int(color)
        if color == BG:
            continue
        pts = np.argwhere(g == color)
        assert len(pts) >= 2, "each marker color must appear at least twice"
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        rects.append((color, int(r0), int(c0), int(r1), int(c1)))

    counts = _cover_counts(g.shape, [(r0, c0, r1, c1) for _, r0, c0, r1, c1 in rects])
    bg_mask = g == BG

    overlap2 = bg_mask & (counts == 2)
    overlap3 = bg_mask & (counts >= 3)
    out[overlap2] = YELLOW
    out[overlap3] = MAROON

    single_cover = bg_mask & (counts == 1)
    for color, r0, c0, r1, c1 in rects:
        border = _border_mask(g.shape, (r0, c0, r1, c1))
        out[single_cover & border] = color

    return out


def _expanded_intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0 - pad
        or br1 + pad < ar0 - pad
        or ac1 + pad < bc0 - pad
        or bc1 + pad < ac0 - pad
    )


def _sample_anchor_margins(
    rng: random.Random, anchor: str, high: tuple[int, int], low: tuple[int, int]
) -> tuple[int, int, int, int]:
    hi0, hi1 = high
    lo0, lo1 = low

    def hi() -> int:
        return rng.randint(hi0, hi1)

    def lo() -> int:
        return rng.randint(lo0, lo1)

    if anchor == "nw":
        return hi(), hi(), lo(), lo()
    if anchor == "ne":
        return hi(), lo(), lo(), hi()
    if anchor == "sw":
        return lo(), hi(), hi(), lo()
    if anchor == "se":
        return lo(), lo(), hi(), hi()
    raise ValueError(anchor)


def _rect_from_core(
    core: tuple[int, int, int, int], margins: tuple[int, int, int, int]
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = core
    up, left, down, right = margins
    return (r0 - up, c0 - left, r1 + down, c1 + right)


def _place_disjoint_rect(
    rng: random.Random,
    h: int,
    w: int,
    keepouts: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    max_h = min(10, h - 2)
    max_w = min(10, w - 2)
    if max_h < 4 or max_w < 4:
        return None

    for _ in range(200):
        rh = rng.randint(4, max_h)
        rw = rng.randint(4, max_w)
        r0 = rng.randint(1, h - rh - 1)
        c0 = rng.randint(1, w - rw - 1)
        rect = (r0, c0, r0 + rh - 1, c0 + rw - 1)
        if all(not _expanded_intersects(rect, other, pad=1) for other in keepouts):
            return rect
    return None


def _gen_none_rects(rng: random.Random, h: int, w: int) -> list[tuple[int, int, int, int]] | None:
    n_rects = rng.randint(2, 4)
    rects: list[tuple[int, int, int, int]] = []
    for _ in range(n_rects):
        rect = _place_disjoint_rect(rng, h, w, rects)
        if rect is None:
            return None
        rects.append(rect)
    return rects


def _gen_double_rects(rng: random.Random, h: int, w: int) -> list[tuple[int, int, int, int]] | None:
    for _ in range(200):
        core_h = rng.randint(2, 4)
        core_w = rng.randint(2, 4)
        anchors = rng.choice([("nw", "se"), ("ne", "sw")])
        margins = [_sample_anchor_margins(rng, anchor, (2, 4), (0, 2)) for anchor in anchors]

        max_up = max(m[0] for m in margins)
        max_left = max(m[1] for m in margins)
        max_down = max(m[2] for m in margins)
        max_right = max(m[3] for m in margins)

        r0_min = 1 + max_up
        c0_min = 1 + max_left
        r0_max = h - 2 - core_h - max_down
        c0_max = w - 2 - core_w - max_right
        if r0_min > r0_max or c0_min > c0_max:
            continue

        cr0 = rng.randint(r0_min, r0_max)
        cc0 = rng.randint(c0_min, c0_max)
        core = (cr0, cc0, cr0 + core_h - 1, cc0 + core_w - 1)
        rects = [_rect_from_core(core, m) for m in margins]

        counts = _cover_counts((h, w), rects)
        if counts.max() != 2 or not np.any(counts == 2):
            continue

        if rng.random() < 0.4:
            extra = _place_disjoint_rect(rng, h, w, rects)
            if extra is None:
                continue
            rects.append(extra)

        return rects
    return None


def _gen_triple_rects(rng: random.Random, h: int, w: int) -> list[tuple[int, int, int, int]] | None:
    for _ in range(300):
        core_h = rng.randint(2, 4)
        core_w = rng.randint(2, 4)
        anchors = rng.sample(("nw", "ne", "sw", "se"), 3)
        margins = [_sample_anchor_margins(rng, anchor, (2, 4), (0, 2)) for anchor in anchors]

        max_up = max(m[0] for m in margins)
        max_left = max(m[1] for m in margins)
        max_down = max(m[2] for m in margins)
        max_right = max(m[3] for m in margins)

        r0_min = 1 + max_up
        c0_min = 1 + max_left
        r0_max = h - 2 - core_h - max_down
        c0_max = w - 2 - core_w - max_right
        if r0_min > r0_max or c0_min > c0_max:
            continue

        cr0 = rng.randint(r0_min, r0_max)
        cc0 = rng.randint(c0_min, c0_max)
        core = (cr0, cc0, cr0 + core_h - 1, cc0 + core_w - 1)
        rects = [_rect_from_core(core, m) for m in margins]

        counts = _cover_counts((h, w), rects)
        if not np.any(counts >= 3) or not np.any(counts == 2):
            continue

        if rng.random() < 0.3:
            extra = _place_disjoint_rect(rng, h, w, rects)
            if extra is not None:
                rects.append(extra)

        return rects
    return None


def _marker_patterns(rect: tuple[int, int, int, int]) -> list[list[tuple[int, int]]]:
    r0, c0, r1, c1 = rect
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    tl = (r0, c0)
    tr = (r0, c1)
    bl = (r1, c0)
    br = (r1, c1)
    patterns = [
        [tl, br],
        [tr, bl],
        [tl, tr, bl],
        [tl, tr, br],
        [tl, bl, br],
        [tr, bl, br],
        [tl, tr, bl, br],
    ]
    if h >= 5 and w >= 5:
        mid_r = (r0 + r1) // 2
        mid_c = (c0 + c1) // 2
        patterns.append([(r0, mid_c), (r1, mid_c), (mid_r, c0), (mid_r, c1)])
    return patterns


def _is_isolated(cell: tuple[int, int], taken: set[tuple[int, int]]) -> bool:
    r, c = cell
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if (r + dr, c + dc) in taken:
                return False
    return True


def _paint_markers(
    rng: random.Random,
    rects: list[tuple[int, int, int, int]],
    colors: list[int],
    shape: tuple[int, int],
) -> np.ndarray | None:
    grid = np.full(shape, BG, dtype=int)
    taken: set[tuple[int, int]] = set()

    for color, rect in zip(colors, rects):
        patterns = _marker_patterns(rect)
        rng.shuffle(patterns)
        placed = None
        for pattern in patterns:
            if len(set(pattern)) != len(pattern):
                continue
            if all(_is_isolated(cell, taken) for cell in pattern):
                local_taken = set()
                ok = True
                for cell in pattern:
                    if not _is_isolated(cell, local_taken):
                        ok = False
                        break
                    local_taken.add(cell)
                if ok:
                    placed = pattern
                    break
        if placed is None:
            return None
        for r, c in placed:
            grid[r, c] = color
            taken.add((r, c))

    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(14, 24)
    w = rng.randint(14, 24)
    scenario = rng.choice(("none", "double", "triple"))

    if scenario == "none":
        rects = _gen_none_rects(rng, h, w)
    elif scenario == "double":
        rects = _gen_double_rects(rng, h, w)
    else:
        rects = _gen_triple_rects(rng, h, w)
    if rects is None:
        return None

    colors = rng.sample(list(MARKER_COLORS), len(rects))
    grid = _paint_markers(rng, rects, colors, (h, w))
    if grid is None:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None

    if scenario == "double" and not np.any(out == YELLOW):
        return None
    if scenario == "triple" and (not np.any(out == YELLOW) or not np.any(out == MAROON)):
        return None
    if np.any(grid == YELLOW) or np.any(grid == MAROON):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
