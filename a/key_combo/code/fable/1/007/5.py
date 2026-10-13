from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
RED = 2
PALETTE = (1, 3, 4, 5, 6, 7, 8, 9)
SIDES = ("top", "bottom", "left", "right")
MAX_GENERATION_ATTEMPTS = 400


Coord = tuple[int, int]
Rect = tuple[int, int, int, int]  # (r0, c0, h, w)


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)


def _reachable_non_red(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    reachable = np.zeros((h, w), dtype=bool)
    queue: deque[Coord] = deque()

    def add(r: int, c: int) -> None:
        if 0 <= r < h and 0 <= c < w and int(grid[r, c]) != RED and not reachable[r, c]:
            reachable[r, c] = True
            queue.append((r, c))

    for r in range(h):
        add(r, 0)
        add(r, w - 1)
    for c in range(w):
        add(0, c)
        add(h - 1, c)

    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            add(nr, nc)
    return reachable


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue: deque[Coord] = deque([(r, c)])
            seen[r, c] = True
            cells: list[Coord] = []
            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _draw_closed_rect(grid: np.ndarray, rect: Rect) -> None:
    r0, c0, h, w = rect
    grid[r0, c0 : c0 + w] = RED
    grid[r0 + h - 1, c0 : c0 + w] = RED
    grid[r0 : r0 + h, c0] = RED
    grid[r0 : r0 + h, c0 + w - 1] = RED


def _draw_open_rect(
    grid: np.ndarray,
    rect: Rect,
    side: str,
    gap_start: int,
    gap_len: int,
) -> None:
    _draw_closed_rect(grid, rect)
    r0, c0, h, w = rect
    if side == "top":
        grid[r0, c0 + gap_start : c0 + gap_start + gap_len] = BLACK
    elif side == "bottom":
        grid[r0 + h - 1, c0 + gap_start : c0 + gap_start + gap_len] = BLACK
    elif side == "left":
        grid[r0 + gap_start : r0 + gap_start + gap_len, c0] = BLACK
    else:
        grid[r0 + gap_start : r0 + gap_start + gap_len, c0 + w - 1] = BLACK


def _reserve(occupied: np.ndarray, rect: Rect, padding: int = 1) -> None:
    h, w = occupied.shape
    r0, c0, rh, rw = rect
    rr0 = max(0, r0 - padding)
    cc0 = max(0, c0 - padding)
    rr1 = min(h, r0 + rh + padding)
    cc1 = min(w, c0 + rw + padding)
    occupied[rr0:rr1, cc0:cc1] = True


def _free_box_positions(
    occupied: np.ndarray,
    box_h: int,
    box_w: int,
    border: int = 1,
    padding: int = 1,
) -> list[Coord]:
    h, w = occupied.shape
    positions: list[Coord] = []
    for r0 in range(border, h - box_h - border + 1):
        for c0 in range(border, w - box_w - border + 1):
            rr0 = max(0, r0 - padding)
            cc0 = max(0, c0 - padding)
            rr1 = min(h, r0 + box_h + padding)
            cc1 = min(w, c0 + box_w + padding)
            if not bool(occupied[rr0:rr1, cc0:cc1].any()):
                positions.append((r0, c0))
    return positions


def _pick_top_level_rect(
    occupied: np.ndarray,
    h_range: tuple[int, int],
    w_range: tuple[int, int],
    border: int = 1,
    padding: int = 1,
) -> Rect | None:
    for _ in range(80):
        box_h = random.randint(*h_range)
        box_w = random.randint(*w_range)
        positions = _free_box_positions(occupied, box_h, box_w, border=border, padding=padding)
        if positions:
            r0, c0 = random.choice(positions)
            rect = (r0, c0, box_h, box_w)
            _reserve(occupied, rect, padding=padding)
            return rect
    return None


def _pick_inner_rect(outer: Rect) -> Rect | None:
    r0, c0, h, w = outer
    max_h = min(6, h - 4)
    max_w = min(6, w - 4)
    if max_h < 4 or max_w < 4:
        return None
    for _ in range(60):
        inner_h = random.randint(4, max_h)
        inner_w = random.randint(4, max_w)
        top_min = r0 + 2
        top_max = r0 + h - inner_h - 2
        left_min = c0 + 2
        left_max = c0 + w - inner_w - 2
        if top_min > top_max or left_min > left_max:
            continue
        inner_r0 = random.randint(top_min, top_max)
        inner_c0 = random.randint(left_min, left_max)
        return inner_r0, inner_c0, inner_h, inner_w
    return None


def _rect_interior(rect: Rect) -> list[Coord]:
    r0, c0, h, w = rect
    return [(r, c) for r in range(r0 + 1, r0 + h - 1) for c in range(c0 + 1, c0 + w - 1)]


def _sample_pair_counts(available: int, low_total: int, high_total: int) -> tuple[int, int] | None:
    max_total = min(high_total, available)
    if max_total < low_total:
        return None
    total = random.randint(low_total, max_total)
    left = random.randint(2, total - 2)
    return left, total - left


def _paint_pair(
    grid: np.ndarray,
    cells: list[Coord],
    pair: tuple[int, int],
    low_total: int,
    high_total: int,
) -> bool:
    available = [cell for cell in cells if int(grid[cell]) == BLACK]
    counts = _sample_pair_counts(len(available), low_total, high_total)
    if counts is None:
        return False
    left_count, right_count = counts
    picked = random.sample(available, left_count + right_count)
    random.shuffle(picked)
    left, right = pair
    for cell in picked[:left_count]:
        grid[cell] = left
    for cell in picked[left_count:]:
        grid[cell] = right
    return True


def _build_scene() -> tuple[np.ndarray, list[list[Coord]]] | None:
    h = random.randint(22, 30)
    w = random.randint(22, 30)
    grid = np.full((h, w), BLACK, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    outer = _pick_top_level_rect(occupied, (11, 15), (11, 15), border=1, padding=1)
    if outer is None:
        return None
    inner = _pick_inner_rect(outer)
    if inner is None:
        return None

    _draw_closed_rect(grid, outer)
    _draw_closed_rect(grid, inner)

    extra_closed: list[Rect] = []
    for _ in range(random.randint(0, 2)):
        rect = _pick_top_level_rect(occupied, (5, 8), (5, 8), border=1, padding=1)
        if rect is None:
            break
        extra_closed.append(rect)
        _draw_closed_rect(grid, rect)

    open_interiors: list[list[Coord]] = []
    for _ in range(random.randint(1, 2)):
        rect = _pick_top_level_rect(occupied, (6, 10), (6, 10), border=1, padding=1)
        if rect is None:
            if not open_interiors:
                return None
            break
        side = random.choice(SIDES)
        if side in {"top", "bottom"}:
            max_gap = min(2, rect[3] - 3)
            gap_len = 1 if max_gap <= 1 else random.choice((1, 1, 2))
            gap_start = random.randint(1, rect[3] - gap_len - 1)
        else:
            max_gap = min(2, rect[2] - 3)
            gap_len = 1 if max_gap <= 1 else random.choice((1, 1, 2))
            gap_start = random.randint(1, rect[2] - gap_len - 1)
        _draw_open_rect(grid, rect, side, gap_start, gap_len)
        open_interiors.append(_rect_interior(rect))

    return grid, open_interiors


def _semantic_checks(grid: np.ndarray, open_interiors: list[list[Coord]]) -> bool:
    reachable = _reachable_non_red(grid)
    closed_components = _components((grid != RED) & ~reachable)
    if len(closed_components) < 2 or not open_interiors:
        return False

    chamber_pairs: list[tuple[int, int]] = []
    chamber_colors: list[set[int]] = []
    for comp in closed_components:
        values = {int(v) for v in grid[comp[:, 0], comp[:, 1]] if int(v) != BLACK}
        if len(values) != 2:
            return False
        pair = tuple(sorted(values))
        chamber_pairs.append(pair)
        chamber_colors.append(values)

    reachable_color_mask = reachable & (grid != BLACK) & (grid != RED)
    if not bool(reachable_color_mask.any()):
        return False

    open_mask = np.zeros(grid.shape, dtype=bool)
    for interior in open_interiors:
        if not interior:
            return False
        if not any(int(grid[cell]) != BLACK for cell in interior):
            return False
        for cell in interior:
            if not reachable[cell]:
                return False
            open_mask[cell] = True

    if not bool((reachable_color_mask & ~open_mask).any()):
        return False

    shared_with_reachable = set(map(int, grid[reachable_color_mask])) & set().union(*chamber_colors)
    if not shared_with_reachable:
        return False

    mates_by_color: dict[int, set[int]] = {}
    for left, right in chamber_pairs:
        mates_by_color.setdefault(left, set()).add(right)
        mates_by_color.setdefault(right, set()).add(left)
    if not any(len(mates) >= 2 for mates in mates_by_color.values()):
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    if not np.array_equal(out[grid == RED], grid[grid == RED]):
        return False
    if np.any(out[reachable_color_mask] != BLACK):
        return False

    for comp in closed_components:
        rr = comp[:, 0]
        cc = comp[:, 1]
        values = sorted({int(v) for v in grid[rr, cc] if int(v) != BLACK})
        left, right = values
        left_mask = grid[rr, cc] == left
        right_mask = grid[rr, cc] == right
        black_mask = grid[rr, cc] == BLACK
        if np.any(out[rr[left_mask], cc[left_mask]] != right):
            return False
        if np.any(out[rr[right_mask], cc[right_mask]] != left):
            return False
        if np.any(out[rr[black_mask], cc[black_mask]] != BLACK):
            return False
    return True


def _attempt() -> np.ndarray | None:
    built = _build_scene()
    if built is None:
        return None
    grid, open_interiors = built

    reachable = _reachable_non_red(grid)
    chambers = sorted(_components((grid != RED) & ~reachable), key=lambda comp: -int(comp.shape[0]))
    if len(chambers) < 2:
        return None

    a, b, c, d = random.sample(PALETTE, 4)
    pair_cycle = [(a, b), (b, c), (a, d), (c, d), (a, c), (b, d)]
    chamber_pairs = [pair_cycle[i % len(pair_cycle)] for i in range(len(chambers))]

    for comp, pair in zip(chambers, chamber_pairs):
        cells = [tuple(map(int, rc)) for rc in comp]
        if not _paint_pair(grid, cells, pair, low_total=4, high_total=10):
            return None

    for i, interior in enumerate(open_interiors):
        pair = chamber_pairs[i % len(chamber_pairs)]
        if not _paint_pair(grid, interior, pair, low_total=4, high_total=8):
            return None

    open_mask = np.zeros(grid.shape, dtype=bool)
    for interior in open_interiors:
        for cell in interior:
            open_mask[cell] = True

    outside_cells = [
        (r, c)
        for r in range(grid.shape[0])
        for c in range(grid.shape[1])
        if reachable[r, c] and not open_mask[r, c] and int(grid[r, c]) == BLACK
    ]
    if not _paint_pair(grid, outside_cells, chamber_pairs[0], low_total=4, high_total=8):
        return None

    extra_outside = [cell for cell in outside_cells if int(grid[cell]) == BLACK]
    if extra_outside:
        bonus_color = random.choice(chamber_pairs[1])
        bonus_count = random.randint(1, min(4, len(extra_outside)))
        for cell in random.sample(extra_outside, bonus_count):
            grid[cell] = bonus_color

    if not _semantic_checks(grid, open_interiors):
        return None
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """Use red walls as topology barriers: swap the two colors inside each sealed chamber and erase all reachable colors.

    Input structure
    - A black background contains one-cell-thick red wall segments that may form
      fully closed chambers, leaky near-chambers with gaps, and nested walls.
    - Non-red, non-black colored pixels are scattered both inside sealed chambers
      and in boundary-reachable space, including inside leaky outlines.
    - Only red acts as a wall; every sealed chamber contains exactly two colors.

    Rule
    - Treat red as an impassable 4-connected barrier and flood from the grid
      boundary through every non-red cell that can reach the outside.
    - In each remaining unreachable non-red-connected chamber, swap its two
      colors simultaneously, chamber by chamber.
    - Recolor every boundary-reachable colored pixel to black and leave all red
      wall pixels unchanged.

    Key insight
    - The scope is determined by true reachability, not by visual enclosure:
      a tiny wall gap makes its interior evaporate, while nested sealed chambers
      still swap independently even when they reuse the same colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D ARC grid"

    reachable = _reachable_non_red(g)
    out = g.copy()
    reachable_colors = reachable & (g != BLACK) & (g != RED)
    out[reachable_colors] = BLACK

    chamber_mask = (g != RED) & ~reachable
    for comp in _components(chamber_mask):
        rr = comp[:, 0]
        cc = comp[:, 1]
        colors = sorted({int(v) for v in g[rr, cc] if int(v) != BLACK})
        assert len(colors) == 2, "each sealed chamber must contain exactly two colors"
        left, right = colors
        left_mask = g[rr, cc] == left
        right_mask = g[rr, cc] == right
        out[rr[left_mask], cc[left_mask]] = right
        out[rr[right_mask], cc[right_mask]] = left

    return out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid puzzle after {MAX_GENERATION_ATTEMPTS} attempts")
