import random
from typing import Optional

import numpy as np


BG = 0
RED = 2
DEFAULT_CLUE = 5
MAX_GENERATION_ATTEMPTS = 400


def _prefix_sums(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cell_ps = np.pad(mask.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    row_ps = np.pad(mask.astype(int), ((0, 0), (1, 0))).cumsum(1)
    return cell_ps, row_ps


def _rect_sum(ps: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    if r0 > r1 or c0 > c1:
        return 0
    return int(ps[r1 + 1, c1 + 1] - ps[r0, c1 + 1] - ps[r1 + 1, c0] + ps[r0, c0])


def _row_sum(row_ps: np.ndarray, r: int, c0: int, c1: int) -> int:
    return int(row_ps[r, c1 + 1] - row_ps[r, c0])


def _col_sum(cell_ps: np.ndarray, c: int, r0: int, r1: int) -> int:
    return _rect_sum(cell_ps, r0, c, r1, c)


def _clue_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid[grid != BG], return_counts=True)
    assert len(colors) > 0, "expected at least one foreground clue color"
    return int(colors[np.argmax(counts)])


def _candidate_key(bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int, int, int]:
    r0, c0, r1, c1 = bbox
    area = (r1 - r0 + 1) * (c1 - c0 + 1)
    perimeter = 2 * ((r1 - r0 + 1) + (c1 - c0 + 1))
    return (area, perimeter, -(r1 - r0), -(c1 - c0), -r0, -c0)


def _find_best_rectangle(mask: np.ndarray) -> tuple[int, int, int, int]:
    h, w = mask.shape
    cell_ps, row_ps = _prefix_sums(mask)
    candidates: list[tuple[int, int, int, int]] = []
    best_key: Optional[tuple[int, int, int, int, int, int]] = None

    for r0 in range(h - 2):
        for r1 in range(r0 + 2, h):
            for c0 in range(w - 2):
                for c1 in range(c0 + 2, w):
                    if _rect_sum(cell_ps, r0 + 1, c0 + 1, r1 - 1, c1 - 1) != 0:
                        continue
                    if _row_sum(row_ps, r0, c0, c1) < 2:
                        continue
                    if _row_sum(row_ps, r1, c0, c1) < 2:
                        continue
                    if _col_sum(cell_ps, c0, r0, r1) < 2:
                        continue
                    if _col_sum(cell_ps, c1, r0, r1) < 2:
                        continue

                    bbox = (r0, c0, r1, c1)
                    key = _candidate_key(bbox)
                    if best_key is None or key > best_key:
                        best_key = key
                        candidates = [bbox]
                    elif key == best_key:
                        candidates.append(bbox)

    assert best_key is not None, "no valid invisible rectangle found"
    assert len(candidates) == 1, f"expected a unique best rectangle, found {len(candidates)}"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify the largest hinted axis-aligned rectangle and draw its missing border in red.

    Input structure:
    - Background is black.
    - One foreground clue color appears on the border of one or more possible rectangles.
    - The target rectangle is the unique largest rectangle whose four sides each contain at least
      two clue pixels and whose strict interior contains no clue pixels.

    Rule:
    - Keep every existing clue pixel unchanged.
    - Draw a one-pixel-thick red border for the target rectangle, but only on background cells.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    clue = _clue_color(g)
    mask = g == clue
    r0, c0, r1, c1 = _find_best_rectangle(mask)

    out = g.copy()
    for c in range(c0, c1 + 1):
        if out[r0, c] == BG:
            out[r0, c] = RED
        if out[r1, c] == BG:
            out[r1, c] = RED
    for r in range(r0, r1 + 1):
        if out[r, c0] == BG:
            out[r, c0] = RED
        if out[r, c1] == BG:
            out[r, c1] = RED
    return out


def _paint_rectangle_hints(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    color: int,
    rng: random.Random,
) -> None:
    r0, c0, r1, c1 = bbox
    side_points = {
        "top": [(r0, c) for c in range(c0, c1 + 1)],
        "bottom": [(r1, c) for c in range(c0, c1 + 1)],
        "left": [(r, c0) for r in range(r0, r1 + 1)],
        "right": [(r, c1) for r in range(r0, r1 + 1)],
    }
    for pts in side_points.values():
        take = rng.randint(2, min(len(pts), 5))
        for r, c in rng.sample(pts, take):
            grid[r, c] = color

    extra_prob = rng.uniform(0.1, 0.35)
    border = set(side_points["top"] + side_points["bottom"] + side_points["left"] + side_points["right"])
    for r, c in border:
        if grid[r, c] == BG and rng.random() < extra_prob:
            grid[r, c] = color


def _bbox_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0)


def _sample_bbox(
    h: int,
    w: int,
    rng: random.Random,
    min_h: int,
    max_h: int,
    min_w: int,
    max_w: int,
    forbidden: list[tuple[int, int, int, int]],
) -> Optional[tuple[int, int, int, int]]:
    for _ in range(80):
        rect_h = rng.randint(min_h, max_h)
        rect_w = rng.randint(min_w, max_w)
        if rect_h >= h or rect_w >= w:
            continue
        r0 = rng.randint(0, h - rect_h)
        c0 = rng.randint(0, w - rect_w)
        bbox = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
        if any(_bbox_overlap(bbox, other, pad=1) for other in forbidden):
            continue
        return bbox
    return None


def _add_noise(grid: np.ndarray, target: tuple[int, int, int, int], rng: random.Random) -> None:
    h, w = grid.shape
    r0, c0, r1, c1 = target
    cells = []
    for r in range(h):
        for c in range(w):
            if r0 < r < r1 and c0 < c < c1:
                continue
            cells.append((r, c))

    rng.shuffle(cells)
    budget = rng.randint(max(4, (h * w) // 30), max(8, (h * w) // 12))
    for r, c in cells[:budget]:
        if grid[r, c] == BG and rng.random() < 0.7:
            grid[r, c] = DEFAULT_CLUE


def _gen_attempt() -> Optional[np.ndarray]:
    rng = random.Random()
    h = rng.randint(12, 24)
    w = rng.randint(12, 24)
    grid = np.zeros((h, w), dtype=int)

    target_h = rng.randint(6, h - 2)
    target_w = rng.randint(6, w - 2)
    target_r0 = rng.randint(0, h - target_h)
    target_c0 = rng.randint(0, w - target_w)
    target = (target_r0, target_c0, target_r0 + target_h - 1, target_c0 + target_w - 1)

    _paint_rectangle_hints(grid, target, DEFAULT_CLUE, rng)

    distractors: list[tuple[int, int, int, int]] = [target]
    distractor_count = rng.choice([0, 1, 2])
    for _ in range(distractor_count):
        bbox = _sample_bbox(
            h,
            w,
            rng,
            min_h=4,
            max_h=max(4, min(target_h - 1, h - 1)),
            min_w=4,
            max_w=max(4, min(target_w - 1, w - 1)),
            forbidden=distractors,
        )
        if bbox is None:
            continue
        if (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1) >= target_h * target_w:
            continue
        _paint_rectangle_hints(grid, bbox, DEFAULT_CLUE, rng)
        distractors.append(bbox)

    _add_noise(grid, target, rng)

    try:
        found = _find_best_rectangle(grid == DEFAULT_CLUE)
    except AssertionError:
        return None
    if found != target:
        return None

    out = main(grid)
    border_mask = np.zeros_like(grid, dtype=bool)
    tr0, tc0, tr1, tc1 = target
    border_mask[tr0, tc0 : tc1 + 1] = True
    border_mask[tr1, tc0 : tc1 + 1] = True
    border_mask[tr0 : tr1 + 1, tc0] = True
    border_mask[tr0 : tr1 + 1, tc1] = True
    newly_red = (out == RED) & (grid == BG)
    if not np.any(newly_red & border_mask):
        return None
    if np.any(newly_red & ~border_mask):
        return None
    if np.any(grid[tr0 + 1 : tr1, tc0 + 1 : tc1] == DEFAULT_CLUE):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rectangle-hint input")
