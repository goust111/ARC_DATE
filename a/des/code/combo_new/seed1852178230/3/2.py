from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
GEN_FRAME_COLOR = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
QUADRANTS = ("tl", "tr", "bl", "br")
MAX_GENERATION_ATTEMPTS = 300
PLACEMENT_ATTEMPTS = 80


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each seeded quadrant of a gray rectangular room with the seed's color.

    Input structure:
    - The background is black.
    - One-pixel-thick gray rectangular frames form several disjoint closed rooms.
    - Inside each room there are one or more colored seed pixels, and seeds in the
      same room occupy distinct quadrants of the room interior.

    Rule:
    - For every gray rectangle, split its interior into four quadrants.
    - When the interior width or height is odd, the middle column belongs to the
      left quadrants and the middle row belongs to the top quadrants.
    - For each quadrant containing at least one seed, recolor all black pixels in
      that quadrant to the seed color.
    - Leave the gray borders, the seed pixels themselves, and every unseeded
      quadrant unchanged.

    Key insight:
    - A seed does not spread from its exact position; it only labels one whole
      quadrant of its enclosing rectangle, and that quadrant is what gets filled.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frame_color = _find_frame_color(g)
    frames = _components(g == frame_color)
    assert frames, "expected at least one rectangular frame"

    out = g.copy()
    for comp in frames:
        r0, c0, r1, c1 = _bbox(comp)
        _assert_rect_frame(g, frame_color, r0, c0, r1, c1, comp)

        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        inner_h, inner_w = interior.shape
        row_mid = (inner_h + 1) // 2
        col_mid = (inner_w + 1) // 2

        quadrant_colors: dict[str, int] = {}
        for rr, cc in np.argwhere((interior != BG) & (interior != frame_color)):
            quad = _quadrant_of(int(rr), int(cc), row_mid, col_mid)
            color = int(interior[int(rr), int(cc)])
            prev = quadrant_colors.get(quad)
            assert prev is None or prev == color, "conflicting seed colors in one quadrant"
            quadrant_colors[quad] = color

        assert quadrant_colors, "each frame interior must contain at least one seed"

        for quad, color in quadrant_colors.items():
            rs, cs = _quadrant_slice(quad, row_mid, col_mid, inner_h, inner_w)
            patch = out[
                slice(r0 + 1 + rs.start, r0 + 1 + rs.stop),
                slice(c0 + 1 + cs.start, c0 + 1 + cs.stop),
            ]
            mask = patch == BG
            patch[mask] = color

    return out


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _assert_rect_frame(
    grid: np.ndarray,
    frame_color: int,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    comp: list[tuple[int, int]],
) -> None:
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    assert h >= 4 and w >= 4, "frames must enclose a non-empty 2x2+ interior"

    expected = set()
    for r in range(r0, r1 + 1):
        expected.add((r, c0))
        expected.add((r, c1))
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))

    comp_set = set(comp)
    assert comp_set == expected, "frame color components must be rectangular borders"
    assert np.all(grid[r0, c0 : c1 + 1] == frame_color)
    assert np.all(grid[r1, c0 : c1 + 1] == frame_color)
    assert np.all(grid[r0 : r1 + 1, c0] == frame_color)
    assert np.all(grid[r0 : r1 + 1, c1] == frame_color)


def _find_frame_color(grid: np.ndarray) -> int:
    candidates: list[tuple[int, int]] = []
    for color in map(int, np.unique(grid)):
        if color == BG:
            continue
        comps = _components(grid == color)
        if not comps:
            continue
        total = 0
        ok = True
        for comp in comps:
            r0, c0, r1, c1 = _bbox(comp)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h < 4 or w < 4:
                ok = False
                break
            expected_area = 2 * (h + w) - 4
            if len(comp) != expected_area:
                ok = False
                break
            comp_set = set(comp)
            for r in range(r0, r1 + 1):
                if (r, c0) not in comp_set or (r, c1) not in comp_set:
                    ok = False
                    break
            if not ok:
                break
            for c in range(c0, c1 + 1):
                if (r0, c) not in comp_set or (r1, c) not in comp_set:
                    ok = False
                    break
            if not ok:
                break
            total += len(comp)
        if ok:
            candidates.append((total, color))

    assert candidates, "could not identify the frame color"
    return max(candidates)[1]


def _quadrant_of(rr: int, cc: int, row_mid: int, col_mid: int) -> str:
    top = rr < row_mid
    left = cc < col_mid
    if top and left:
        return "tl"
    if top and not left:
        return "tr"
    if not top and left:
        return "bl"
    return "br"


def _quadrant_slice(
    quad: str,
    row_mid: int,
    col_mid: int,
    inner_h: int,
    inner_w: int,
) -> tuple[slice, slice]:
    if quad == "tl":
        return slice(0, row_mid), slice(0, col_mid)
    if quad == "tr":
        return slice(0, row_mid), slice(col_mid, inner_w)
    if quad == "bl":
        return slice(row_mid, inner_h), slice(0, col_mid)
    if quad == "br":
        return slice(row_mid, inner_h), slice(col_mid, inner_w)
    raise ValueError(f"unknown quadrant {quad}")


def _sample_dimension(rng: random.Random, limit: int) -> int:
    parity = rng.choice((0, 1))
    options = [v for v in range(2, limit + 1) if v % 2 == parity]
    if not options:
        options = list(range(2, limit + 1))
    return rng.choice(options)


def _can_place(used: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1) -> bool:
    h, w = used.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    return not bool(used[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _mark_used(used: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1) -> None:
    h, w = used.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    used[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _paint_frame(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int) -> None:
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _quadrant_bounds(inner_h: int, inner_w: int, quad: str) -> tuple[int, int, int, int]:
    row_mid = (inner_h + 1) // 2
    col_mid = (inner_w + 1) // 2
    rs, cs = _quadrant_slice(quad, row_mid, col_mid, inner_h, inner_w)
    return rs.start, rs.stop, cs.start, cs.stop


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(12, 30)
    width = rng.randint(12, 30)
    grid = np.zeros((height, width), dtype=int)
    used = np.zeros((height, width), dtype=bool)

    max_rectangles = min(5, max(2, (height * width) // 110 + 1))
    target_rectangles = rng.randint(2, max_rectangles)

    placed = 0
    inner_dim_cap = min(10, height - 3, width - 3)
    if inner_dim_cap < 2:
        return None

    for _ in range(target_rectangles):
        placed_here = False
        for _ in range(PLACEMENT_ATTEMPTS):
            inner_h_limit = min(inner_dim_cap, height - 4)
            inner_w_limit = min(inner_dim_cap, width - 4)
            if inner_h_limit < 2 or inner_w_limit < 2:
                return None

            inner_h = _sample_dimension(rng, inner_h_limit)
            inner_w = _sample_dimension(rng, inner_w_limit)
            outer_h = inner_h + 2
            outer_w = inner_w + 2

            if outer_h + 2 > height or outer_w + 2 > width:
                continue

            r0 = rng.randint(1, height - outer_h - 1)
            c0 = rng.randint(1, width - outer_w - 1)
            r1 = r0 + outer_h - 1
            c1 = c0 + outer_w - 1
            if not _can_place(used, r0, c0, r1, c1, pad=1):
                continue

            _paint_frame(grid, r0, c0, r1, c1, GEN_FRAME_COLOR)
            _mark_used(used, r0, c0, r1, c1, pad=1)

            seed_count = rng.randint(1, 4)
            quads = rng.sample(QUADRANTS, seed_count)
            colors = rng.sample(SEED_COLORS, seed_count)
            for quad, color in zip(quads, colors):
                qr0, qr1, qc0, qc1 = _quadrant_bounds(inner_h, inner_w, quad)
                sr = r0 + 1 + rng.randint(qr0, qr1 - 1)
                sc = c0 + 1 + rng.randint(qc0, qc1 - 1)
                grid[sr, sc] = color

            placed_here = True
            placed += 1
            break
        if not placed_here:
            break

    if placed < 2:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
