import random

import numpy as np


BLACK = 0
GREY = 5
RECT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

# --------------- solver ---------------


def _recover_rectangles(grid: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    rects: list[tuple[int, int, int, int, int]] = []
    for color in np.unique(grid):
        color = int(color)
        if color in (BLACK, GREY):
            continue
        coords = np.argwhere(grid == color)
        assert len(coords) > 0
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        rects.append((color, int(r0), int(c0), int(r1), int(c1)))
    rects.sort()
    return rects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey diagonal-path pixel by the color of the rectangle it covers.

    Input structure
    - Black background.
    - Several non-overlapping solid rectangles in distinct colors.
    - One-pixel grey diagonal paths are drawn on top of the rectangles/background.

    Rule
    - For each grey pixel, recover which rectangle lies underneath it.
    - If the grey pixel falls inside a rectangle, restore that rectangle color.
    - Otherwise keep the grey pixel unchanged.

    Key insight
    - Since each rectangle uses a unique color, the visible pixels of that color
      still determine the rectangle's original bbox even after grey diagonal
      pixels are drawn over it.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    rects = _recover_rectangles(g)
    out = g.copy()

    for color, r0, c0, r1, c1 in rects:
        sub = out[r0 : r1 + 1, c0 : c1 + 1]
        mask = sub == GREY
        if np.any(mask):
            sub = sub.copy()
            sub[mask] = color
            out[r0 : r1 + 1, c0 : c1 + 1] = sub

    return out


# --------------- generator ---------------

GRID_H_RANGE = (12, 20)
GRID_W_RANGE = (12, 20)
RECT_COUNT_RANGE = (2, 5)
RECT_H_RANGE = (2, 6)
RECT_W_RANGE = (2, 7)
RECT_MARGIN = 1
RECT_GAP = 1

MIN_PATH_LEN = 5
MIN_BG_ON_PATH = 2
MIN_PATH_HIT_CELLS = 2
MAX_PATH_COUNT = 3

RECT_PLACEMENT_ATTEMPTS = 80
PATH_SELECTION_ATTEMPTS = 80
GENERATION_ATTEMPTS = 400


def _boxes_too_close(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0
    )


def _place_rectangles(rng: random.Random, h: int, w: int) -> list[dict[str, int]]:
    count = rng.randint(*RECT_COUNT_RANGE)
    colors = rng.sample(RECT_COLORS, count)
    rects: list[dict[str, int]] = []

    for color in colors:
        placed = False
        for _ in range(RECT_PLACEMENT_ATTEMPTS):
            rh = rng.randint(*RECT_H_RANGE)
            rw = rng.randint(*RECT_W_RANGE)
            if rh + 2 * RECT_MARGIN > h or rw + 2 * RECT_MARGIN > w:
                continue

            r0 = rng.randint(RECT_MARGIN, h - RECT_MARGIN - rh)
            c0 = rng.randint(RECT_MARGIN, w - RECT_MARGIN - rw)
            box = (r0, c0, r0 + rh - 1, c0 + rw - 1)
            if any(
                _boxes_too_close(
                    box, (rect["r0"], rect["c0"], rect["r1"], rect["c1"]), RECT_GAP
                )
                for rect in rects
            ):
                continue

            rects.append(
                {
                    "color": color,
                    "r0": r0,
                    "c0": c0,
                    "r1": r0 + rh - 1,
                    "c1": c0 + rw - 1,
                }
            )
            placed = True
            break
        if not placed:
            return []

    return rects


def _render_base(h: int, w: int, rects: list[dict[str, int]]) -> np.ndarray:
    grid = np.full((h, w), BLACK, dtype=int)
    for rect in rects:
        grid[rect["r0"] : rect["r1"] + 1, rect["c0"] : rect["c1"] + 1] = rect["color"]
    return grid


def _diagonal_lines(h: int, w: int) -> list[tuple[list[tuple[int, int]], int]]:
    lines: list[tuple[list[tuple[int, int]], int]] = []

    for c0 in range(w):
        cells = []
        r, c = 0, c0
        while r < h and c < w:
            cells.append((r, c))
            r += 1
            c += 1
        lines.append((cells, 1))
    for r0 in range(1, h):
        cells = []
        r, c = r0, 0
        while r < h and c < w:
            cells.append((r, c))
            r += 1
            c += 1
        lines.append((cells, 1))

    for c0 in range(w):
        cells = []
        r, c = 0, c0
        while r < h and c >= 0:
            cells.append((r, c))
            r += 1
            c -= 1
        lines.append((cells, -1))
    for r0 in range(1, h):
        cells = []
        r, c = r0, w - 1
        while r < h and c >= 0:
            cells.append((r, c))
            r += 1
            c -= 1
        lines.append((cells, -1))

    return lines


def _path_candidates(
    rng: random.Random, base: np.ndarray, rects: list[dict[str, int]]
) -> list[dict[str, object]]:
    h, w = base.shape
    color_to_idx = {rect["color"]: i for i, rect in enumerate(rects)}
    candidates: list[dict[str, object]] = []

    for cells, slope in _diagonal_lines(h, w):
        if len(cells) < MIN_PATH_LEN:
            continue

        hit_counts: dict[int, int] = {}
        bg_count = 0
        for r, c in cells:
            color = int(base[r, c])
            if color == BLACK:
                bg_count += 1
            else:
                rect_idx = color_to_idx[color]
                hit_counts[rect_idx] = hit_counts.get(rect_idx, 0) + 1

        if not hit_counts or bg_count < MIN_BG_ON_PATH:
            continue

        total_hits = sum(hit_counts.values())
        if total_hits < MIN_PATH_HIT_CELLS or max(hit_counts.values()) < 2:
            continue

        candidates.append(
            {
                "cells": cells,
                "cell_set": set(cells),
                "slope": slope,
                "hit_rects": tuple(sorted(hit_counts)),
                "hit_counts": hit_counts,
            }
        )

    rng.shuffle(candidates)
    return candidates


def _compatible(
    cand: dict[str, object], used_rects: set[int], used_cells: set[tuple[int, int]]
) -> bool:
    hit_rects = cand["hit_rects"]
    cell_set = cand["cell_set"]
    return set(hit_rects).isdisjoint(used_rects) and cell_set.isdisjoint(used_cells)


def _pick_candidate(
    rng: random.Random,
    candidates: list[dict[str, object]],
    used_rects: set[int],
    used_cells: set[tuple[int, int]],
    slope: int | None = None,
) -> dict[str, object] | None:
    options = [
        cand
        for cand in candidates
        if _compatible(cand, used_rects, used_cells)
        and (slope is None or cand["slope"] == slope)
    ]
    if not options:
        return None

    multi_rect = [cand for cand in options if len(cand["hit_rects"]) >= 2]
    if multi_rect and rng.random() < 0.45:
        options = multi_rect

    return rng.choice(options)


def _select_paths(
    rng: random.Random, base: np.ndarray, rects: list[dict[str, int]]
) -> list[dict[str, object]] | None:
    candidates = _path_candidates(rng, base, rects)
    if not candidates:
        return None

    max_count = min(MAX_PATH_COUNT, len(rects))
    for _ in range(PATH_SELECTION_ATTEMPTS):
        target_count = rng.randint(1, max_count)
        chosen: list[dict[str, object]] = []
        used_rects: set[int] = set()
        used_cells: set[tuple[int, int]] = set()

        positive_exists = any(cand["slope"] == 1 for cand in candidates)
        negative_exists = any(cand["slope"] == -1 for cand in candidates)
        require_both = (
            target_count >= 2
            and positive_exists
            and negative_exists
            and rng.random() < 0.5
        )

        if require_both:
            slopes = [1, -1]
            rng.shuffle(slopes)
            ok = True
            for slope in slopes:
                cand = _pick_candidate(rng, candidates, used_rects, used_cells, slope)
                if cand is None:
                    ok = False
                    break
                chosen.append(cand)
                used_rects.update(cand["hit_rects"])
                used_cells.update(cand["cell_set"])
            if not ok:
                continue

        while len(chosen) < target_count:
            cand = _pick_candidate(rng, candidates, used_rects, used_cells)
            if cand is None:
                break
            chosen.append(cand)
            used_rects.update(cand["hit_rects"])
            used_cells.update(cand["cell_set"])

        if len(chosen) == target_count:
            return chosen

    return None


def _expected_output(base: np.ndarray, paths: list[dict[str, object]]) -> np.ndarray:
    out = base.copy()
    for path in paths:
        for r, c in path["cells"]:
            if out[r, c] == BLACK:
                out[r, c] = GREY
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)

    rects = _place_rectangles(rng, h, w)
    if len(rects) < RECT_COUNT_RANGE[0]:
        return None

    base = _render_base(h, w, rects)
    paths = _select_paths(rng, base, rects)
    if not paths:
        return None

    grid = base.copy()
    covered_inside = 0
    covered_background = 0
    for path in paths:
        for r, c in path["cells"]:
            if base[r, c] == BLACK:
                covered_background += 1
            else:
                covered_inside += 1
            grid[r, c] = GREY

    if covered_inside == 0 or covered_background == 0:
        return None

    recovered = _recover_rectangles(grid)
    target = sorted(
        (rect["color"], rect["r0"], rect["c0"], rect["r1"], rect["c1"]) for rect in rects
    )
    if recovered != target:
        return None

    if not np.array_equal(main(grid), _expected_output(base, paths)):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
