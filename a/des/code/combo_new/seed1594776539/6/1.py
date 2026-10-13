from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

GRID_H_RANGE = (16, 30)
GRID_W_RANGE = (16, 30)
REGION_COUNT_RANGE = (3, 6)
PLACEMENT_TRIES = 240
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _touches_boundary(points: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in points)


def _enclosed_regions(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    regions = []
    for comp in _components(grid != GREY):
        if not _touches_boundary(comp, h, w):
            regions.append(comp)
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint each grey-bounded room's inner border with its seed color, then erase the seed marker.

    Input structure:
    - The background is black.
    - One-pixel-thick grey rectangular frames form closed rooms.
    - Each room interior contains exactly one non-black, non-grey single-pixel seed.
    - Rooms may be separated by black gaps or have grey borders directly adjacent to each other.

    Rule:
    - Keep every grey border unchanged.
    - Treat grey as walls and find each enclosed interior region that does not touch the grid boundary.
    - Read the region's unique seed color C, recolor every interior cell that is 4-adjacent to grey with C, and set every other interior cell to black.
    - If the interior is only one cell, or if the seed already lies on that inner border, that cell remains color C after recoloring.

    Key insight: The seed is only a color signal; the actual output is the one-cell-thick contour immediately inside each closed grey room.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    out = g.copy()
    for region in _enclosed_regions(g):
        seeds = [(r, c) for r, c in region if int(g[r, c]) not in (BLACK, GREY)]
        assert len(seeds) == 1, "each enclosed region must contain exactly one seed"
        seed_r, seed_c = seeds[0]
        color = int(g[seed_r, seed_c])

        for r, c in region:
            out[r, c] = BLACK
        for r, c in region:
            if any(0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) == GREY for nr, nc in _neighbors4(r, c)):
                out[r, c] = color

    return out


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return not (a1 < b0 or b1 < a0)


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _corner_touch(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    row_overlap = _intervals_overlap(a[0], a[2], b[0], b[2])
    col_overlap = _intervals_overlap(a[1], a[3], b[1], b[3])
    row_gap = max(b[0] - a[2] - 1, a[0] - b[2] - 1, 0)
    col_gap = max(b[1] - a[3] - 1, a[1] - b[3] - 1, 0)
    return not row_overlap and not col_overlap and row_gap == 0 and col_gap == 0


def _has_side_adjacency(boxes: list[tuple[int, int, int, int]]) -> bool:
    for i, a in enumerate(boxes):
        for b in boxes[i + 1 :]:
            same_rows = _intervals_overlap(a[0], a[2], b[0], b[2])
            same_cols = _intervals_overlap(a[1], a[3], b[1], b[3])
            if same_rows and (a[3] + 1 == b[1] or b[3] + 1 == a[1]):
                return True
            if same_cols and (a[2] + 1 == b[0] or b[2] + 1 == a[0]):
                return True
    return False


def _sample_outer_size(kind: str, h: int, w: int, rng: random.Random) -> tuple[int, int] | None:
    if kind == "tiny":
        return (3, 3)
    if kind == "ring":
        max_h = min(9, h - 2)
        max_w = min(11, w - 2)
        if max_h < 4 or max_w < 4:
            return None
        return (rng.randint(4, max_h), rng.randint(4, max_w))
    max_h = min(10, h - 2)
    max_w = min(12, w - 2)
    if max_h < 5 or max_w < 5:
        return None
    return (rng.randint(5, max_h), rng.randint(5, max_w))


def _place_box(
    boxes: list[tuple[int, int, int, int]],
    h: int,
    w: int,
    outer_h: int,
    outer_w: int,
    rng: random.Random,
) -> tuple[int, int, int, int] | None:
    for _ in range(PLACEMENT_TRIES):
        if boxes and rng.random() < 0.6:
            base = rng.choice(boxes)
            side = rng.choice(("up", "down", "left", "right"))
            gap = rng.choice((0, 0, 1, 2))
            if side in ("left", "right"):
                c0 = base[1] - gap - outer_w if side == "left" else base[3] + 1 + gap
                r_min = max(1, base[0] - outer_h + 1)
                r_max = min(h - outer_h - 1, base[2])
                if r_min > r_max:
                    continue
                r0 = rng.randint(r_min, r_max)
            else:
                r0 = base[0] - gap - outer_h if side == "up" else base[2] + 1 + gap
                c_min = max(1, base[1] - outer_w + 1)
                c_max = min(w - outer_w - 1, base[3])
                if c_min > c_max:
                    continue
                c0 = rng.randint(c_min, c_max)
        else:
            if h - outer_h - 1 < 1 or w - outer_w - 1 < 1:
                return None
            r0 = rng.randint(1, h - outer_h - 1)
            c0 = rng.randint(1, w - outer_w - 1)

        r1 = r0 + outer_h - 1
        c1 = c0 + outer_w - 1
        if r0 < 1 or c0 < 1 or r1 >= h - 1 or c1 >= w - 1:
            continue

        cand = (r0, c0, r1, c1)
        if any(_boxes_overlap(cand, other) or _corner_touch(cand, other) for other in boxes):
            continue
        return cand
    return None


def _seed_position(
    box: tuple[int, int, int, int], kind: str, rng: random.Random
) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    if kind == "tiny":
        return (r0 + 1, c0 + 1)

    inner_rows = range(r0 + 1, r1)
    inner_cols = range(c0 + 1, c1)
    if kind == "ring":
        ring_cells = [
            (r, c)
            for r in inner_rows
            for c in inner_cols
            if r in (r0 + 1, r1 - 1) or c in (c0 + 1, c1 - 1)
        ]
        return rng.choice(ring_cells)

    return (rng.randint(r0 + 2, r1 - 2), rng.randint(c0 + 2, c1 - 2))


def _draw_frame(grid: np.ndarray, box: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY


def _render_expected(shape: tuple[int, int], specs: list[tuple[tuple[int, int, int, int], int]]) -> np.ndarray:
    out = np.full(shape, BLACK, dtype=int)
    for box, color in specs:
        _draw_frame(out, box)
        r0, c0, r1, c1 = box
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                if r in (r0 + 1, r1 - 1) or c in (c0 + 1, c1 - 1):
                    out[r, c] = color
    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        h = rng.randint(*GRID_H_RANGE)
        w = rng.randint(*GRID_W_RANGE)
        region_count = rng.randint(*REGION_COUNT_RANGE)
        want_touching = rng.random() < 0.5

        kinds = ["tiny", "ring", "inner"]
        for _extra in range(region_count - 3):
            kinds.append(rng.choice(("tiny", "ring", "inner")))
        rng.shuffle(kinds)

        if region_count > len(SEED_COLORS):
            continue
        colors = rng.sample(SEED_COLORS, region_count)

        boxes: list[tuple[int, int, int, int]] = []
        placements: list[tuple[str, tuple[int, int, int, int], tuple[int, int], int]] = []
        ok = True
        for kind, color in zip(kinds, colors, strict=False):
            size = _sample_outer_size(kind, h, w, rng)
            if size is None:
                ok = False
                break
            box = _place_box(boxes, h, w, size[0], size[1], rng)
            if box is None:
                ok = False
                break
            seed = _seed_position(box, kind, rng)
            boxes.append(box)
            placements.append((kind, box, seed, color))
        if not ok:
            continue

        if _has_side_adjacency(boxes) != want_touching:
            continue

        inp = np.full((h, w), BLACK, dtype=int)
        expected_specs: list[tuple[tuple[int, int, int, int], int]] = []
        for _kind, box, seed, color in placements:
            _draw_frame(inp, box)
            inp[seed] = color
            expected_specs.append((box, color))

        regions = _enclosed_regions(inp)
        if len(regions) != region_count:
            continue
        if any(sum(int(inp[r, c]) not in (BLACK, GREY) for r, c in region) != 1 for region in regions):
            continue

        expected = _render_expected(inp.shape, expected_specs)
        try:
            actual = main(inp)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return inp

    raise RuntimeError("failed to generate a valid ARC input")
