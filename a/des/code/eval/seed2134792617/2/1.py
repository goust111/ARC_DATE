from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


@dataclass(frozen=True)
class RectSpec:
    top: int
    left: int
    bottom: int
    right: int
    gap_r: int
    gap_c: int
    dr: int
    dc: int
    seed_r: int
    seed_c: int
    seed_color: int

    @property
    def area(self) -> int:
        return int((self.bottom - self.top - 1) * (self.right - self.left - 1))


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _rectangle_boundary(top: int, left: int, bottom: int, right: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for c in range(left, right + 1):
        cells.add((top, c))
        cells.add((bottom, c))
    for r in range(top, bottom + 1):
        cells.add((r, left))
        cells.add((r, right))
    return cells


def _extract_rectangles(grid: np.ndarray, bg: int) -> list[RectSpec]:
    rectangles: list[RectSpec] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for comp in _components(grid == color):
            rows = [r for r, _ in comp]
            cols = [c for _, c in comp]
            top, bottom = min(rows), max(rows)
            left, right = min(cols), max(cols)
            if bottom - top < 2 or right - left < 2:
                continue

            comp_set = set(comp)
            boundary = _rectangle_boundary(top, left, bottom, right)
            if len(comp_set) != len(boundary) - 1:
                continue
            if not comp_set <= boundary:
                continue

            missing = list(boundary - comp_set)
            if len(missing) != 1:
                continue
            gap_r, gap_c = missing[0]

            on_horizontal = gap_r in (top, bottom)
            on_vertical = gap_c in (left, right)
            if on_horizontal == on_vertical:
                continue

            if gap_r == top:
                dr, dc = -1, 0
            elif gap_r == bottom:
                dr, dc = 1, 0
            elif gap_c == left:
                dr, dc = 0, -1
            else:
                dr, dc = 0, 1

            interior = grid[top + 1 : bottom, left + 1 : right]
            if interior.size == 0:
                continue
            non_bg = np.argwhere(interior != bg)
            if non_bg.shape[0] != 1:
                continue
            seed_rr, seed_cc = map(int, non_bg[0])
            seed_r = top + 1 + seed_rr
            seed_c = left + 1 + seed_cc
            seed_color = int(grid[seed_r, seed_c])
            if seed_color == color:
                continue

            rectangles.append(
                RectSpec(
                    top=top,
                    left=left,
                    bottom=bottom,
                    right=right,
                    gap_r=gap_r,
                    gap_c=gap_c,
                    dr=dr,
                    dc=dc,
                    seed_r=seed_r,
                    seed_c=seed_c,
                    seed_color=seed_color,
                )
            )
    return rectangles


def _branch_cells(start_r: int, start_c: int, dr: int, dc: int, length: int, h: int, w: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    r, c = int(start_r), int(start_c)
    for _ in range(int(length)):
        if not (0 <= r < h and 0 <= c < w):
            break
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow a colored branch outward from each rectangle gap.

    Input structure:
    - Black background.
    - One or more 1px-thick rectangular frames, each missing exactly one border pixel.
    - Exactly one colored seed pixel inside each frame.

    Rule:
    - The missing border pixel sets the outward direction.
    - The branch length equals the rectangle's interior area.
    - The branch color is the seed color.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _most_common_color(g)
    out = g.copy()

    rectangles = _extract_rectangles(g, bg)
    assert rectangles, "expected at least one gapped rectangle"

    for rect in rectangles:
        for r, c in _branch_cells(rect.gap_r, rect.gap_c, rect.dr, rect.dc, rect.area, h, w):
            out[r, c] = rect.seed_color

    return out


# --------------- generator ---------------

BG = 0
BORDER = 5
SEED_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GRID_H_RANGE = (14, 26)
GRID_W_RANGE = (14, 28)
INTERIOR_H_CHOICES = [1, 2, 2, 3, 3, 4, 5]
INTERIOR_W_CHOICES = [1, 2, 2, 3, 3, 4, 5, 6]
GEN_ATTEMPTS = 400
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _expanded_box(top: int, left: int, bottom: int, right: int, h: int, w: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for r in range(max(0, top - 1), min(h - 1, bottom + 1) + 1):
        for c in range(max(0, left - 1), min(w - 1, right + 1) + 1):
            cells.add((r, c))
    return cells


def _halo(cells: list[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    zone: set[tuple[int, int]] = set()
    for r, c in cells:
        for nr in range(max(0, r - 1), min(h - 1, r + 1) + 1):
            for nc in range(max(0, c - 1), min(w - 1, c + 1) + 1):
                zone.add((nr, nc))
    return zone


def _direction_cycle(n: int) -> list[tuple[int, int]]:
    order: list[tuple[int, int]] = []
    while len(order) < n:
        chunk = DIRECTIONS[:]
        random.shuffle(chunk)
        order.extend(chunk)
    return order[:n]


def _truncation_cycle(n: int) -> list[bool]:
    order: list[bool] = []
    while len(order) < n:
        chunk = [False, True]
        random.shuffle(chunk)
        order.extend(chunk)
    return order[:n]


def _sample_object_count(h: int, w: int) -> int:
    area = h * w
    if area < 260:
        return random.choice([1, 2, 2, 3])
    if area < 420:
        return random.choice([1, 2, 2, 3, 3, 4])
    return random.choice([2, 2, 3, 3, 4, 4, 5])


def _free_length(top: int, left: int, bottom: int, right: int, dr: int, dc: int, h: int, w: int) -> int:
    if dr == -1:
        return int(top + 1)
    if dr == 1:
        return int(h - bottom)
    if dc == -1:
        return int(left + 1)
    return int(w - right)


def _candidate_gaps(top: int, left: int, bottom: int, right: int, dr: int, dc: int) -> list[tuple[int, int]]:
    if dr == -1:
        gaps = [(top, c) for c in range(left + 1, right)]
    elif dr == 1:
        gaps = [(bottom, c) for c in range(left + 1, right)]
    elif dc == -1:
        gaps = [(r, left) for r in range(top + 1, bottom)]
    else:
        gaps = [(r, right) for r in range(top + 1, bottom)]
    random.shuffle(gaps)
    return gaps


def _find_placement(
    h: int,
    w: int,
    blocked: set[tuple[int, int]],
    seed_color: int,
    dr: int,
    dc: int,
    want_truncation: bool,
) -> tuple[RectSpec, set[tuple[int, int]]] | None:
    dims = [(ih, iw) for ih in INTERIOR_H_CHOICES for iw in INTERIOR_W_CHOICES]
    random.shuffle(dims)
    for ih, iw in dims:
        top_positions = list(range(0, h - (ih + 2) + 1))
        left_positions = list(range(0, w - (iw + 2) + 1))
        random.shuffle(top_positions)
        random.shuffle(left_positions)

        for top in top_positions:
            bottom = top + ih + 1
            for left in left_positions:
                right = left + iw + 1
                free = _free_length(top, left, bottom, right, dr, dc, h, w)
                area = ih * iw
                is_truncated = area > free
                if is_truncated != want_truncation:
                    continue

                bbox_zone = _expanded_box(top, left, bottom, right, h, w)
                if bbox_zone & blocked:
                    continue

                for gap_r, gap_c in _candidate_gaps(top, left, bottom, right, dr, dc):
                    branch = _branch_cells(gap_r, gap_c, dr, dc, area, h, w)
                    reserve = bbox_zone | _halo(branch, h, w)
                    if reserve & blocked:
                        continue

                    seed_r = random.randint(top + 1, bottom - 1)
                    seed_c = random.randint(left + 1, right - 1)
                    spec = RectSpec(
                        top=top,
                        left=left,
                        bottom=bottom,
                        right=right,
                        gap_r=gap_r,
                        gap_c=gap_c,
                        dr=dr,
                        dc=dc,
                        seed_r=seed_r,
                        seed_c=seed_c,
                        seed_color=seed_color,
                    )
                    return spec, reserve
    return None


def _render_input(h: int, w: int, specs: list[RectSpec]) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for spec in specs:
        for r, c in _rectangle_boundary(spec.top, spec.left, spec.bottom, spec.right):
            if (r, c) != (spec.gap_r, spec.gap_c):
                grid[r, c] = BORDER
        grid[spec.seed_r, spec.seed_c] = spec.seed_color
    return grid


def _expected_output(inp: np.ndarray, specs: list[RectSpec]) -> np.ndarray:
    out = inp.copy()
    h, w = out.shape
    for spec in specs:
        for r, c in _branch_cells(spec.gap_r, spec.gap_c, spec.dr, spec.dc, spec.area, h, w):
            out[r, c] = spec.seed_color
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    object_count = min(_sample_object_count(h, w), len(SEED_COLORS))

    colors = random.sample(SEED_COLORS, object_count)
    directions = _direction_cycle(object_count)
    truncations = _truncation_cycle(object_count)

    blocked: set[tuple[int, int]] = set()
    specs: list[RectSpec] = []

    for idx in range(object_count):
        seed_color = colors[idx]
        dr, dc = directions[idx]
        want_truncation = truncations[idx]

        placed = _find_placement(h, w, blocked, seed_color, dr, dc, want_truncation)
        if placed is None:
            placed = _find_placement(h, w, blocked, seed_color, dr, dc, not want_truncation)
        if placed is None:
            return None

        spec, reserve = placed
        specs.append(spec)
        blocked |= reserve

    inp = _render_input(h, w, specs)
    expected = _expected_output(inp, specs)

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gapped-rectangle puzzle")
