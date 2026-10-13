import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

GRID_MIN = 18
GRID_MAX = 24
REGION_COUNT_CHOICES = (2, 3)
REGION_SIZE_RANGE = (6, 8)
BAR_SHORT_SIDES = (1, 2)
BAR_LONG_RANGE = (3, 5)
GEN_RETRIES = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_of_value(grid: np.ndarray, value: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != value:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == value:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _extract_seed_bar(grid: np.ndarray) -> tuple[int, int, int, int]:
    red_components = _components_of_value(grid, RED)
    assert len(red_components) == 1, "expected exactly one red seed bar"
    comp = red_components[0]
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    assert h != w, "seed bar must be non-square"
    assert len(comp) == h * w, "seed bar must be a solid rectangle"
    assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == RED), "seed bar bbox must be solid red"
    return r0, c0, h, w


def _stamp_rect_skip_color(
    out: np.ndarray,
    top: int,
    left: int,
    height: int,
    width: int,
    color: int,
    blocked_color: int | None,
) -> None:
    h, w = out.shape
    rr0 = max(0, top)
    cc0 = max(0, left)
    rr1 = min(h, top + height)
    cc1 = min(w, left + width)
    if rr0 >= rr1 or cc0 >= cc1:
        return
    if blocked_color is None:
        out[rr0:rr1, cc0:cc1] = color
        return
    view = out[rr0:rr1, cc0:cc1]
    mask = view != blocked_color
    view[mask] = color


def _apply_zigzag_tiling(
    out: np.ndarray,
    seed_top: int,
    seed_left: int,
    bar_h: int,
    bar_w: int,
    blocked_color: int | None,
) -> np.ndarray:
    period = bar_h + bar_w
    limit = out.shape[0] + out.shape[1] + period + 2
    for n in range(-limit, limit + 1):
        offset = n * period
        _stamp_rect_skip_color(out, seed_top + offset, seed_left + offset, bar_h, bar_w, RED, blocked_color)
        _stamp_rect_skip_color(out, seed_top + offset, seed_left - offset, bar_h, bar_w, RED, blocked_color)
        _stamp_rect_skip_color(
            out,
            seed_top + bar_h + offset,
            seed_left + bar_w + offset,
            bar_w,
            bar_h,
            GREEN,
            blocked_color,
        )
        _stamp_rect_skip_color(
            out,
            seed_top + bar_h + offset,
            seed_left - bar_h - offset,
            bar_w,
            bar_h,
            GREEN,
            blocked_color,
        )
    return out


def _enclosed_black_regions(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    regions: list[list[tuple[int, int]]] = []
    for comp in _components_of_value(grid, BLACK):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        regions.append(comp)
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend the red seed bar into diagonal zigzag tiling, let blue walls clip only wall cells, then fill the untouched gaps inside enclosed regions with yellow.

    Input structure:
    - The grid uses black background, one red solid non-square rectangle, and several blue rectangular wall loops.
    - Each blue loop encloses a black interior region, and the red seed bar is separate from all walls.
    - No green or yellow pixels appear in the input.

    Rule:
    - Use the red rectangle as a seed bar and repeat it along both 45-degree diagonals with period height+width.
    - Insert 90-degree rotated copies between adjacent red copies and recolor those rotated copies green.
    - Stamp the tiling across the whole grid with clipping, but never overwrite blue wall pixels; then fill only the still-black cells inside each enclosed blue region with yellow.

    Key insight:
    - The blue walls do not stop the zigzag pattern globally; they only punch holes in the wall cells themselves, so the region interior ends up as an interference pattern of surviving red/green tiles plus yellow gap fill.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    seed_top, seed_left, bar_h, bar_w = _extract_seed_bar(g)

    out = g.copy()
    _apply_zigzag_tiling(out, seed_top, seed_left, bar_h, bar_w, BLUE)

    for region in _enclosed_black_regions(g):
        for r, c in region:
            if int(out[r, c]) == BLACK:
                out[r, c] = YELLOW

    return out


def _bar_dimension_options(max_side: int) -> list[tuple[int, int]]:
    options: list[tuple[int, int]] = []
    for short in BAR_SHORT_SIDES:
        for long_side in range(BAR_LONG_RANGE[0], BAR_LONG_RANGE[1] + 1):
            if short == long_side:
                continue
            options.append((short, long_side))
            options.append((long_side, short))
    filtered = []
    for h, w in options:
        if max_side / (h + w) >= 3.0:
            filtered.append((h, w))
    return filtered


def _bbox_far_enough(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    gap: int,
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return bool(ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0)


def _ring_coverage_mask(shape: tuple[int, int], top: int, left: int, height: int, width: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    bottom = top + height - 1
    right = left + width - 1
    mask[top, left : right + 1] = True
    mask[bottom, left : right + 1] = True
    mask[top : bottom + 1, left] = True
    mask[top : bottom + 1, right] = True
    return mask


def _candidate_regions(
    tiling: np.ndarray,
    seed_box: tuple[int, int, int, int],
) -> list[dict]:
    h, w = tiling.shape
    candidates: list[dict] = []
    for region_h in range(REGION_SIZE_RANGE[0], REGION_SIZE_RANGE[1] + 1):
        for region_w in range(REGION_SIZE_RANGE[0], REGION_SIZE_RANGE[1] + 1):
            for top in range(1, h - region_h):
                for left in range(1, w - region_w):
                    box = (top, left, top + region_h - 1, left + region_w - 1)
                    if not _bbox_far_enough(seed_box, box, gap=1):
                        continue
                    interior = tiling[top + 1 : top + region_h - 1, left + 1 : left + region_w - 1]
                    covered = int(np.count_nonzero(interior))
                    if covered < 2 or interior.size - covered < 2:
                        continue
                    colors_inside = set(map(int, np.unique(interior))) - {BLACK}
                    ring_mask = _ring_coverage_mask(tiling.shape, top, left, region_h, region_w)
                    ring_covered = int(np.count_nonzero(tiling[ring_mask]))
                    if ring_covered < 1:
                        continue
                    candidates.append(
                        {
                            "box": box,
                            "colors_inside": colors_inside,
                            "covered": covered,
                            "uncovered": int(interior.size - covered),
                            "ring_covered": ring_covered,
                        }
                    )
    return candidates


def _pick_regions(rng: random.Random, candidates: list[dict], count: int) -> list[dict] | None:
    if len(candidates) < count:
        return None
    for _ in range(300):
        chosen = rng.sample(candidates, count)
        if all(
            _bbox_far_enough(chosen[i]["box"], chosen[j]["box"], gap=1)
            for i in range(count)
            for j in range(i + 1, count)
        ):
            colors_seen = set().union(*(cand["colors_inside"] for cand in chosen))
            if colors_seen >= {RED, GREEN}:
                return chosen
    return None


def _draw_region_walls(grid: np.ndarray, box: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = BLUE
    grid[r1, c0 : c1 + 1] = BLUE
    grid[r0 : r1 + 1, c0] = BLUE
    grid[r0 : r1 + 1, c1] = BLUE


def _oracle_accepts(inp: np.ndarray) -> bool:
    try:
        out = main(inp)
    except Exception:
        return False

    if out.shape != inp.shape:
        return False
    if not np.any(out == GREEN) or not np.any(out == YELLOW):
        return False
    if not np.array_equal(out[inp == BLUE], inp[inp == BLUE]):
        return False
    if not np.array_equal(main(inp.copy()), out):
        return False

    regions = _enclosed_black_regions(inp)
    if len(regions) not in REGION_COUNT_CHOICES:
        return False
    for region in regions:
        vals = {int(out[r, c]) for r, c in region}
        if YELLOW not in vals:
            return False
        if not ({RED, GREEN} & vals):
            return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(GRID_MIN, GRID_MAX)
    w = rng.randint(GRID_MIN, GRID_MAX)
    dims = _bar_dimension_options(max(h, w))
    if not dims:
        return None
    bar_h, bar_w = rng.choice(dims)
    seed_top = rng.randint(0, h - bar_h)
    seed_left = rng.randint(0, w - bar_w)
    seed_box = (seed_top, seed_left, seed_top + bar_h - 1, seed_left + bar_w - 1)

    tiling = np.zeros((h, w), dtype=int)
    _apply_zigzag_tiling(tiling, seed_top, seed_left, bar_h, bar_w, None)
    candidates = _candidate_regions(tiling, seed_box)
    if len(candidates) < min(REGION_COUNT_CHOICES):
        return None

    region_count = rng.choice(REGION_COUNT_CHOICES)
    chosen = _pick_regions(rng, candidates, region_count)
    if chosen is None:
        return None

    inp = np.zeros((h, w), dtype=int)
    inp[seed_top : seed_top + bar_h, seed_left : seed_left + bar_w] = RED
    for region in chosen:
        _draw_region_walls(inp, region["box"])

    if not _oracle_accepts(inp):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for task 10")
