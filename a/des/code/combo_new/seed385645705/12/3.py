import random
from typing import List, Tuple

import numpy as np


BG = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 300


BBox = Tuple[int, int, int, int]


def _fill_rect(grid: np.ndarray, bbox: BBox, color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = color


def _intersection_area(a: BBox, b: BBox) -> int:
    ir0 = max(a[0], b[0])
    ic0 = max(a[1], b[1])
    ir1 = min(a[2], b[2])
    ic1 = min(a[3], b[3])
    if ir0 > ir1 or ic0 > ic1:
        return 0
    return (ir1 - ir0 + 1) * (ic1 - ic0 + 1)


def _legend_colors(grid: np.ndarray) -> List[int]:
    mask = (grid != BG) & (grid != GREY)
    coords = np.argwhere(mask)
    assert coords.shape[0] == 2, "expected exactly two legend pixels"
    cols = {int(c) for _r, c in coords}
    assert len(cols) == 1, "legend must be a single vertical column"

    col = int(coords[0, 1])
    rows = sorted(int(r) for r, _c in coords)
    assert rows[1] == rows[0] + 1, "legend pixels must be vertically adjacent"
    return [int(grid[rows[0], col]), int(grid[rows[1], col])]


def _corner_candidates(mask: np.ndarray) -> List[Tuple[BBox, str]]:
    h, w = mask.shape
    out: List[Tuple[BBox, str]] = []
    seen = set()

    def add(bbox: BBox, orient: str) -> None:
        key = (bbox, orient)
        if key not in seen:
            seen.add(key)
            out.append((bbox, orient))

    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue

            if (r == 0 or not mask[r - 1, c]) and (c == 0 or not mask[r, c - 1]):
                c1 = c
                while c1 + 1 < w and mask[r, c1 + 1] and (r == 0 or not mask[r - 1, c1 + 1]):
                    c1 += 1
                r1 = r
                while r1 + 1 < h and mask[r1 + 1, c] and (c == 0 or not mask[r1 + 1, c - 1]):
                    r1 += 1
                add((r, c, r1, c1), "TL")

            if (r == 0 or not mask[r - 1, c]) and (c == w - 1 or not mask[r, c + 1]):
                c0 = c
                while c0 - 1 >= 0 and mask[r, c0 - 1] and (r == 0 or not mask[r - 1, c0 - 1]):
                    c0 -= 1
                r1 = r
                while r1 + 1 < h and mask[r1 + 1, c] and (c == w - 1 or not mask[r1 + 1, c + 1]):
                    r1 += 1
                add((r, c0, r1, c), "TR")

            if (r == h - 1 or not mask[r + 1, c]) and (c == 0 or not mask[r, c - 1]):
                c1 = c
                while c1 + 1 < w and mask[r, c1 + 1] and (r == h - 1 or not mask[r + 1, c1 + 1]):
                    c1 += 1
                r0 = r
                while r0 - 1 >= 0 and mask[r0 - 1, c] and (c == 0 or not mask[r0 - 1, c - 1]):
                    r0 -= 1
                add((r0, c, r, c1), "BL")

            if (r == h - 1 or not mask[r + 1, c]) and (c == w - 1 or not mask[r, c + 1]):
                c0 = c
                while c0 - 1 >= 0 and mask[r, c0 - 1] and (r == h - 1 or not mask[r + 1, c0 - 1]):
                    c0 -= 1
                r0 = r
                while r0 - 1 >= 0 and mask[r0 - 1, c] and (c == w - 1 or not mask[r0 - 1, c + 1]):
                    r0 -= 1
                add((r0, c0, r, c), "BR")

    return out


def _pair_union_matches(mask: np.ndarray, a: BBox, b: BBox) -> bool:
    trial = np.zeros_like(mask)
    _fill_rect(trial, a, True)
    _fill_rect(trial, b, True)
    return np.array_equal(trial, mask)


def _solve_rectangles(grid: np.ndarray) -> Tuple[BBox, BBox]:
    mask = grid == GREY
    candidates = _corner_candidates(mask)
    by_orient = {"TL": [], "TR": [], "BL": [], "BR": []}
    for bbox, orient in candidates:
        by_orient[orient].append(bbox)

    solutions: List[Tuple[BBox, BBox]] = []
    for tl in by_orient["TL"]:
        for br in by_orient["BR"]:
            if _intersection_area(tl, br) == 0:
                continue
            if _pair_union_matches(mask, tl, br):
                solutions.append((tl, br))
    for tr in by_orient["TR"]:
        for bl in by_orient["BL"]:
            if _intersection_area(tr, bl) == 0:
                continue
            if _pair_union_matches(mask, tr, bl):
                solutions.append((tr, bl))

    assert len(solutions) == 1, "expected a unique opposite-corner rectangle pair"
    top_rect, bottom_rect = solutions[0]
    return top_rect, bottom_rect


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct two partially hidden grey rectangles and repaint them by stack depth using the right-edge legend.

    Input structure:
    - The background is black. Two solid grey rectangles overlap, so only one complete outer corner of each rectangle remains fully visible.
    - The visible corners are opposite corners of the final overlap silhouette: either top-left plus bottom-right, or top-right plus bottom-left.
    - A vertical two-pixel legend on the right edge gives the depth colors from top layer to bottom layer.

    Rule:
    - Read the two legend colors from top to bottom and erase the legend in the output.
    - From each fully visible grey corner, extend its complete horizontal and vertical exposed edges to infer the full rectangle bbox.
    - Pair the opposite-corner rectangles whose union exactly recreates the grey silhouette.
    - Repaint the lower rectangle with the second legend color, then repaint the upper rectangle with the first legend color so the overlap shows the top-layer color.

    Key insight:
    - The grey silhouette is the union of two full rectangles; each rectangle is recoverable from one intact corner, and the opposite-corner pairing identifies the two hidden full bboxes.
    """
    g = np.asarray(grid, dtype=int)
    colors = _legend_colors(g)
    top_rect, bottom_rect = _solve_rectangles(g)

    out = np.full_like(g, BG)
    _fill_rect(out, bottom_rect, colors[1])
    _fill_rect(out, top_rect, colors[0])
    return out


def _expected_output(shape: Tuple[int, int], top_rect: BBox, bottom_rect: BBox, colors: List[int]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    _fill_rect(out, bottom_rect, colors[1])
    _fill_rect(out, top_rect, colors[0])
    return out


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    pair = rng.choice((("TL", "BR"), ("TR", "BL")))
    h = rng.randint(13, 22)
    main_w = rng.randint(11, 20)
    w = main_w + 2  # one empty separator column plus one legend column

    for _ in range(200):
        core_h = rng.randint(2, 4)
        core_w = rng.randint(2, 4)
        if h - core_h - 5 < 4 or main_w - core_w - 5 < 4:
            continue

        core_r0 = rng.randint(4, h - core_h - 5)
        core_c0 = rng.randint(4, main_w - core_w - 5)
        core_r1 = core_r0 + core_h - 1
        core_c1 = core_c0 + core_w - 1

        if pair == ("TL", "BR"):
            top_rect = (
                rng.randint(0, core_r0 - 2),
                rng.randint(0, core_c0 - 2),
                core_r1,
                core_c1,
            )
            bottom_rect = (
                core_r0,
                core_c0,
                rng.randint(core_r1 + 2, h - 1),
                rng.randint(core_c1 + 2, main_w - 1),
            )
        else:
            top_rect = (
                rng.randint(0, core_r0 - 2),
                core_c0,
                core_r1,
                rng.randint(core_c1 + 2, main_w - 1),
            )
            bottom_rect = (
                core_r0,
                rng.randint(0, core_c0 - 2),
                rng.randint(core_r1 + 2, h - 1),
                core_c1,
            )

        colors = list(rng.sample(PALETTE, 2))
        legend_r0 = rng.randint(0, h - 2)

        grid = np.full((h, w), BG, dtype=int)
        _fill_rect(grid, top_rect, GREY)
        _fill_rect(grid, bottom_rect, GREY)
        grid[legend_r0, w - 1] = colors[0]
        grid[legend_r0 + 1, w - 1] = colors[1]

        try:
            solved_top, solved_bottom = _solve_rectangles(grid)
        except AssertionError:
            continue
        if (solved_top, solved_bottom) != (top_rect, bottom_rect):
            continue

        expected = _expected_output(grid.shape, top_rect, bottom_rect, colors)
        if not np.array_equal(main(grid), expected):
            continue
        return grid

    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stacked-rectangle input")
