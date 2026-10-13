from __future__ import annotations

import random
from collections import deque

import numpy as np


BACKGROUND = 0
FRAME_COLOR = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left

# Generator tuning.
GRID_SIDE_RANGE = (20, 28)
SHAPES = [(2, 2), (2, 3), (2, 4), (3, 2), (3, 3), (3, 4), (4, 2), (4, 3), (4, 4)]
GEN_ATTEMPTS = 400
RECT_ATTEMPTS = 300


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in DIRS:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _perimeter_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c1))
    for c in range(c1, c0 - 1, -1):
        cells.append((r1, c))
    for r in range(r1 - 1, r0, -1):
        cells.append((r, c0))
    return cells


def _gap_direction(gap: tuple[int, int], r0: int, c0: int, r1: int, c1: int) -> int:
    gr, gc = gap
    if gr == r0 and c0 < gc < c1:
        return 0
    if gc == c1 and r0 < gr < r1:
        return 1
    if gr == r1 and c0 < gc < c1:
        return 2
    if gc == c0 and r0 < gr < r1:
        return 3
    raise AssertionError("gap must lie on a non-corner border cell")


def _find_rectangles(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    rects: list[dict[str, object]] = []
    used_colors: set[int] = set()

    for comp in _components(g == FRAME_COLOR):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        assert r1 - r0 >= 3 and c1 - c0 >= 3, "frame must have non-empty interior"

        perimeter = _perimeter_cells(r0, c0, r1, c1)
        perimeter_set = set(perimeter)
        comp_set = set(comp)
        gap_cells = [(r, c) for r, c in perimeter if int(g[r, c]) != FRAME_COLOR]

        assert len(gap_cells) == 1, "each frame must have exactly one colored gap"
        gap = gap_cells[0]
        gap_color = int(g[gap])
        assert gap_color not in (BACKGROUND, FRAME_COLOR)
        assert comp_set == perimeter_set - {gap}, "grey component must be the rectangle border minus the gap"

        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        seed_mask = (interior != BACKGROUND) & (interior != FRAME_COLOR)
        seed_locs = np.argwhere(seed_mask)
        assert len(seed_locs) == 1, "interior must contain exactly one seed"
        sr, sc = map(int, seed_locs[0])
        sr += r0 + 1
        sc += c0 + 1
        seed_color = int(g[sr, sc])
        assert seed_color == gap_color, "seed color must match the gap color"
        assert np.count_nonzero(interior != BACKGROUND) == 1, "interior must contain only the seed"
        assert seed_color not in used_colors, "seed colors must be unique"
        used_colors.add(seed_color)

        rects.append(
            {
                "bbox": (r0, c0, r1, c1),
                "gap": gap,
                "dir": _gap_direction(gap, r0, c0, r1, c1),
                "color": seed_color,
                "area": int((r1 - r0 - 1) * (c1 - c0 - 1)),
            }
        )

    assert rects, "expected at least one framed rectangle"
    return rects


def _draw_vine(out: np.ndarray, gap: tuple[int, int], direction: int, color: int, length: int) -> None:
    h, w = out.shape
    current = gap
    facing = direction
    for _ in range(length):
        for _turn in range(4):
            dr, dc = DIRS[facing]
            nr, nc = current[0] + dr, current[1] + dc
            if 0 <= nr < h and 0 <= nc < w and int(out[nr, nc]) == BACKGROUND:
                out[nr, nc] = color
                current = (nr, nc)
                break
            facing = (facing + 1) % 4
        else:
            raise AssertionError("vine became trapped before reaching its full length")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend a colored vine from each framed region's border gap.
    The vine length equals the rectangle's interior area.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    rects = sorted(_find_rectangles(g), key=lambda item: tuple(item["bbox"]))  # deterministic traversal
    for rect in rects:
        _draw_vine(
            out,
            gap=rect["gap"],  # type: ignore[arg-type]
            direction=int(rect["dir"]),
            color=int(rect["color"]),
            length=int(rect["area"]),
        )
    return out


def _expanded_clear(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> bool:
    h, w = mask.shape
    rr0 = max(0, r0)
    cc0 = max(0, c0)
    rr1 = min(h - 1, r1)
    cc1 = min(w - 1, c1)
    return not bool(mask[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _sample_bbox(
    rng: random.Random,
    height: int,
    width: int,
    outer_h: int,
    outer_w: int,
    direction: int,
    area: int,
    want_turn: bool,
) -> tuple[int, int] | None:
    max_r0 = height - outer_h - 1
    max_c0 = width - outer_w - 1
    if max_r0 < 1 or max_c0 < 1:
        return None

    if direction == 0:
        lo = 1
        hi = min(max_r0, area - 1) if want_turn else max_r0
        if not want_turn:
            lo = area
    elif direction == 1:
        lo = max(1, width - outer_w - area + 1) if want_turn else 1
        hi = max_c0 if want_turn else min(max_c0, width - outer_w - area)
    elif direction == 2:
        lo = max(1, height - outer_h - area + 1) if want_turn else 1
        hi = max_r0 if want_turn else min(max_r0, height - outer_h - area)
    else:
        lo = 1
        hi = min(max_c0, area - 1) if want_turn else max_c0
        if not want_turn:
            lo = area

    if lo > hi:
        return None

    if direction in (0, 2):
        r0 = rng.randint(lo, hi)
        c0 = rng.randint(1, max_c0)
    else:
        r0 = rng.randint(1, max_r0)
        c0 = rng.randint(lo, hi)
    return r0, c0


def _sample_gap(
    rng: random.Random,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    direction: int,
) -> tuple[int, int]:
    if direction == 0:
        return r0, rng.randint(c0 + 1, c1 - 1)
    if direction == 1:
        return rng.randint(r0 + 1, r1 - 1), c1
    if direction == 2:
        return r1, rng.randint(c0 + 1, c1 - 1)
    return rng.randint(r0 + 1, r1 - 1), c0


def _paint_rectangle(
    grid: np.ndarray,
    color: int,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    gap: tuple[int, int],
    seed: tuple[int, int],
) -> None:
    for r, c in _perimeter_cells(r0, c0, r1, c1):
        grid[r, c] = FRAME_COLOR
    grid[gap] = color
    grid[seed] = color


def _simulate_vine(
    occupied: np.ndarray,
    gap: tuple[int, int],
    direction: int,
    length: int,
) -> tuple[list[tuple[int, int]], int] | None:
    h, w = occupied.shape
    path: list[tuple[int, int]] = []
    current = gap
    facing = direction
    turns = 0
    work = occupied.copy()
    for _ in range(length):
        moved = False
        for _turn in range(4):
            dr, dc = DIRS[facing]
            nr, nc = current[0] + dr, current[1] + dc
            if 0 <= nr < h and 0 <= nc < w and not work[nr, nc]:
                work[nr, nc] = True
                path.append((nr, nc))
                current = (nr, nc)
                moved = True
                break
            facing = (facing + 1) % 4
            turns += 1
        if not moved:
            return None
    return path, turns


def _place_one(
    rng: random.Random,
    grid: np.ndarray,
    reserved: np.ndarray,
    color: int,
    direction: int,
    want_turn: bool,
) -> np.ndarray | None:
    height, width = grid.shape
    for _ in range(RECT_ATTEMPTS):
        inner_h, inner_w = rng.choice(SHAPES)
        area = inner_h * inner_w
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        bbox = _sample_bbox(rng, height, width, outer_h, outer_w, direction, area, want_turn)
        if bbox is None:
            continue
        r0, c0 = bbox
        r1 = r0 + outer_h - 1
        c1 = c0 + outer_w - 1

        if not _expanded_clear(reserved, r0 - 1, c0 - 1, r1 + 1, c1 + 1):
            continue

        gap = _sample_gap(rng, r0, c0, r1, c1, direction)
        seed = (rng.randint(r0 + 1, r1 - 1), rng.randint(c0 + 1, c1 - 1))

        occupied = reserved.copy()
        valid = True
        for cell in _perimeter_cells(r0, c0, r1, c1) + [seed]:
            if occupied[cell]:
                valid = False
                break
            occupied[cell] = True
        if not valid:
            continue

        simulated = _simulate_vine(occupied, gap, direction, area)
        if simulated is None:
            continue
        path, turns = simulated
        if want_turn != (turns > 0):
            continue

        _paint_rectangle(grid, color, r0, c0, r1, c1, gap, seed)
        for r, c in path:
            occupied[r, c] = True
        return occupied
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(*GRID_SIDE_RANGE)
    width = rng.randint(*GRID_SIDE_RANGE)
    grid = np.full((height, width), BACKGROUND, dtype=int)
    reserved = np.zeros((height, width), dtype=bool)

    max_objects = min(4, max(1, (height * width) // 170))
    count = rng.randint(1, max_objects)
    colors = rng.sample(COLOR_POOL, count)

    directions = list(range(4))
    rng.shuffle(directions)
    directions = directions[:count]

    if count == 1:
        turn_targets = [bool(rng.randint(0, 1))]
    else:
        turn_targets = [idx < (count // 2) for idx in range(count)]
        if not any(turn_targets):
            turn_targets[0] = True
        if all(turn_targets):
            turn_targets[-1] = False
        rng.shuffle(turn_targets)

    order = list(range(count))
    rng.shuffle(order)
    for idx in order:
        updated = _place_one(
            rng=rng,
            grid=grid,
            reserved=reserved,
            color=colors[idx],
            direction=directions[idx],
            want_turn=turn_targets[idx],
        )
        if updated is None:
            return None
        reserved = updated

    rects = _find_rectangles(grid)
    out = main(grid)
    added = int(np.count_nonzero((out != grid) & (grid == BACKGROUND)))
    expected = sum(int(rect["area"]) for rect in rects)
    assert added == expected, "generated vines must be disjoint and match encoded lengths"
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        try:
            grid = _gen_attempt(rng)
        except AssertionError:
            continue
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
