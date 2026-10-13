from __future__ import annotations

import random
from collections import Counter

import numpy as np

BG = 0
BLOCKER = 4
SQUARE = 5
PAIR_COLORS = (1, 2, 3, 7)
DIRECTIONS = ("up", "right", "down", "left")


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _find_components(mask: np.ndarray) -> list[set[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[set[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: set[tuple[int, int]] = set()
            while stack:
                rr, cc = stack.pop()
                comp.add((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _bbox_center(cells: set[tuple[int, int]]) -> tuple[float, float]:
    r0, c0, r1, c1 = _bbox(cells)
    return (r0 + r1) / 2.0, (c0 + c1) / 2.0


def _sq_dist(p: tuple[float, float], q: tuple[float, float]) -> float:
    return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2


def _solid_rectangle(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    r0, c0, r1, c1 = _bbox(cells)
    area = (r1 - r0 + 1) * (c1 - c0 + 1)
    return len(cells) == area


def _pair_direction(components: list[set[tuple[int, int]]], center: tuple[float, float]) -> str:
    all_cells = set().union(*components)
    pr, pc = _bbox_center(all_cells)
    dr = pr - center[0]
    dc = pc - center[1]
    if abs(dc) > abs(dr):
        return "right" if dc > 0 else "left"
    return "down" if dr > 0 else "up"


def _toward_center(direction: str) -> tuple[int, int]:
    return {
        "up": (1, 0),
        "down": (-1, 0),
        "left": (0, 1),
        "right": (0, -1),
    }[direction]


def _shift_cells(cells: set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in cells}


def _in_bounds(shape: tuple[int, int], cells: set[tuple[int, int]]) -> bool:
    h, w = shape
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _find_square(grid: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid == SQUARE)
    assert coords.shape[0] == 9, "expected one grey 3x3 square"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    assert (int(r1 - r0 + 1), int(c1 - c0 + 1)) == (3, 3)
    assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == SQUARE)
    return int(r0), int(c0), int(r1), int(c1)


def _ray_lengths(grid: np.ndarray, square_bbox: tuple[int, int, int, int]) -> dict[str, int]:
    h, w = grid.shape
    r0, c0, r1, c1 = square_bbox
    rm = r0 + 1
    cm = c0 + 1
    starts = {
        "up": (r0 - 1, cm, -1, 0),
        "down": (r1 + 1, cm, 1, 0),
        "left": (rm, c0 - 1, 0, -1),
        "right": (rm, c1 + 1, 0, 1),
    }
    lengths: dict[str, int] = {}
    for direction, (r, c, dr, dc) in starts.items():
        dist = 0
        while 0 <= r < h and 0 <= c < w and int(grid[r, c]) == BG:
            dist += 1
            r += dr
            c += dc
        assert 0 <= r < h and 0 <= c < w and int(grid[r, c]) == BLOCKER
        lengths[direction] = int(dist)
    return lengths


def _enumerate_candidate_shifts(
    anchor: set[tuple[int, int]],
    mover: set[tuple[int, int]],
    step: tuple[int, int],
    shape: tuple[int, int],
) -> list[int]:
    out: list[int] = []
    dr, dc = step
    for amount in range(1, max(shape) + 1):
        shifted = _shift_cells(mover, dr * amount, dc * amount)
        if not _in_bounds(shape, shifted):
            continue
        if anchor & shifted:
            continue
        if _solid_rectangle(anchor | shifted):
            out.append(amount)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each outer fragment by the blocker-encoded ray length so it completes its same-colored anchor rectangle.

    Input structure
    - The background is black, with one grey 3x3 solid square near the center and one yellow blocker on each cardinal axis of that square.
    - Each blocker sits on the same row or column as the square center, so the empty run from the square edge to the blocker defines one ray length.
    - Four non-background/non-control colors each appear as exactly two 4-connected fragments, with one same-colored pair in each side-region around the square.

    Rule
    - Measure the four ray lengths as the counts of background cells between the square edge-center and the first blocker in the up, right, down, and left directions.
    - For each same-colored pair, identify the fragment whose bounding-box center is farther from the square center; that fragment is the movable one.
    - Classify the pair by which side-region it occupies, then translate the movable fragment toward the square by the matching ray length and leave the nearer anchor fragment fixed.
    - Erase the movable fragment's original cells; keep the square and blockers unchanged.

    Key insight
    - The blockers do not mark destination cells directly: their clipped ray lengths are the numeric translation parameters that disambiguate how each fragment pair should complete a solid rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    assert bg == BG
    square_bbox = _find_square(g)
    r0, c0, _r1, _c1 = square_bbox
    center = (r0 + 1.0, c0 + 1.0)
    ray_lengths = _ray_lengths(g, square_bbox)

    active_colors = [int(c) for c in np.unique(g) if int(c) not in (bg, BLOCKER, SQUARE)]
    assert len(active_colors) == 4

    assignments: dict[str, tuple[int, set[tuple[int, int]], set[tuple[int, int]]]] = {}
    protected = {(r, c) for r, c in np.argwhere((g != bg))}

    for color in active_colors:
        components = _find_components(g == color)
        assert len(components) == 2, "each active color must form one fragment pair"
        direction = _pair_direction(components, center)
        assert direction not in assignments, "expected exactly one pair per direction"

        c0_cells, c1_cells = components
        d0 = _sq_dist(_bbox_center(c0_cells), center)
        d1 = _sq_dist(_bbox_center(c1_cells), center)
        anchor, mover = (c0_cells, c1_cells) if d0 < d1 else (c1_cells, c0_cells)

        amount = ray_lengths[direction]
        step = _toward_center(direction)
        shifted = _shift_cells(mover, step[0] * amount, step[1] * amount)

        assert _in_bounds(g.shape, shifted)
        assert not (anchor & shifted)
        assert _solid_rectangle(anchor | shifted)
        blocked = protected - mover
        assert not (blocked & shifted), "translated fragment must land on background"
        assignments[direction] = (color, mover, shifted)

    assert set(assignments) == set(DIRECTIONS)

    out = g.copy()
    for color, mover, shifted in assignments.values():
        for r, c in mover:
            out[r, c] = bg
        for r, c in shifted:
            out[r, c] = int(color)
    return out


def _bar_cells(axis: str, start_r: int, start_c: int, length: int) -> set[tuple[int, int]]:
    if axis == "v":
        return {(start_r + k, start_c) for k in range(length)}
    return {(start_r, start_c + k) for k in range(length)}


def _place_pair(
    grid: np.ndarray,
    center: tuple[int, int],
    direction: str,
    color: int,
    ray_length: int,
    rect_side: int,
    bar_len: int,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]] | None:
    h, w = grid.shape
    rm, cm = center
    axis = "v" if direction in ("left", "right") else "h"
    sign = 1 if direction in ("right", "down") else -1
    gap = ray_length + rect_side
    if gap < 2:
        return None

    for _ in range(80):
        if direction == "right":
            row_lo = max(0, rm - 3)
            row_hi = min(h - bar_len, rm + 2)
            col_lo = cm + 4
            col_hi = w - gap - 2
            if row_lo > row_hi or col_lo > col_hi:
                return None
            start_r = random.randint(row_lo, row_hi)
            anchor_axis = random.randint(col_lo, col_hi)
            anchor = _bar_cells(axis, start_r, anchor_axis, bar_len)
            mover = _shift_cells(anchor, 0, gap)
        elif direction == "left":
            row_lo = max(0, rm - 3)
            row_hi = min(h - bar_len, rm + 2)
            col_lo = gap + 1
            col_hi = cm - 4
            if row_lo > row_hi or col_lo > col_hi:
                return None
            start_r = random.randint(row_lo, row_hi)
            anchor_axis = random.randint(col_lo, col_hi)
            anchor = _bar_cells(axis, start_r, anchor_axis, bar_len)
            mover = _shift_cells(anchor, 0, -gap)
        elif direction == "down":
            col_lo = max(0, cm - 3)
            col_hi = min(w - bar_len, cm + 2)
            row_lo = rm + 4
            row_hi = h - gap - 2
            if col_lo > col_hi or row_lo > row_hi:
                return None
            start_c = random.randint(col_lo, col_hi)
            anchor_axis = random.randint(row_lo, row_hi)
            anchor = _bar_cells(axis, anchor_axis, start_c, bar_len)
            mover = _shift_cells(anchor, gap, 0)
        else:
            col_lo = max(0, cm - 3)
            col_hi = min(w - bar_len, cm + 2)
            row_lo = gap + 1
            row_hi = rm - 4
            if col_lo > col_hi or row_lo > row_hi:
                return None
            start_c = random.randint(col_lo, col_hi)
            anchor_axis = random.randint(row_lo, row_hi)
            anchor = _bar_cells(axis, anchor_axis, start_c, bar_len)
            mover = _shift_cells(anchor, -gap, 0)

        if not _in_bounds(grid.shape, anchor | mover):
            continue
        if any(int(grid[r, c]) != BG for r, c in anchor | mover):
            continue

        pair_direction = _pair_direction([anchor, mover], (float(rm), float(cm)))
        if pair_direction != direction:
            continue

        shifted = _shift_cells(mover, *_scale_step(_toward_center(direction), ray_length))
        if not _in_bounds(grid.shape, shifted):
            continue
        if anchor & shifted:
            continue
        if any(int(grid[r, c]) not in (BG, color) for r, c in shifted):
            continue
        if not _solid_rectangle(anchor | shifted):
            continue

        candidates = _enumerate_candidate_shifts(anchor, mover, _toward_center(direction), grid.shape)
        if candidates != sorted(set(candidates)):
            continue
        if ray_length not in candidates or len(candidates) != 2:
            continue

        for r, c in anchor | mover:
            grid[r, c] = int(color)
        return anchor, mover, shifted

    return None


def _scale_step(step: tuple[int, int], amount: int) -> tuple[int, int]:
    return step[0] * amount, step[1] * amount


def _semantic_ok(
    grid: np.ndarray,
    expected: dict[str, tuple[int, set[tuple[int, int]], set[tuple[int, int]], int]],
) -> bool:
    try:
        square_bbox = _find_square(grid)
        r0, c0, _r1, _c1 = square_bbox
        center = (r0 + 1.0, c0 + 1.0)
        rays = _ray_lengths(grid, square_bbox)
        if len(set(rays.values())) != 4:
            return False
        seen: set[str] = set()
        active_colors = [int(c) for c in np.unique(grid) if int(c) not in (BG, BLOCKER, SQUARE)]
        if len(active_colors) != 4:
            return False
        for color in active_colors:
            components = _find_components(grid == color)
            if len(components) != 2:
                return False
            direction = _pair_direction(components, center)
            if direction in seen:
                return False
            seen.add(direction)
            anchor, mover = components
            if _sq_dist(_bbox_center(anchor), center) > _sq_dist(_bbox_center(mover), center):
                anchor, mover = mover, anchor
            candidates = _enumerate_candidate_shifts(anchor, mover, _toward_center(direction), grid.shape)
            if len(candidates) != 2 or rays[direction] not in candidates:
                return False
            moved = _shift_cells(mover, *_scale_step(_toward_center(direction), rays[direction]))
            if not _solid_rectangle(anchor | moved):
                return False
            exp_color, exp_anchor, exp_shifted, exp_len = expected[direction]
            if exp_color != color or exp_len != rays[direction]:
                return False
            if anchor != exp_anchor or moved != exp_shifted:
                return False
        if seen != set(DIRECTIONS):
            return False
        produced = main(grid)
        manual = grid.copy()
        for direction, (color, anchor, shifted, _length) in expected.items():
            comps = _find_components(grid == color)
            mover = max(comps, key=lambda cells: _sq_dist(_bbox_center(cells), center))
            for r, c in mover:
                manual[r, c] = BG
            for r, c in shifted:
                manual[r, c] = color
        return np.array_equal(produced, manual)
    except Exception:
        return False


def _attempt() -> np.ndarray | None:
    h = random.randint(24, 28)
    w = random.randint(24, 28)
    grid = np.full((h, w), BG, dtype=int)

    rm = max(9, min(h - 10, h // 2 + random.randint(-2, 2)))
    cm = max(9, min(w - 10, w // 2 + random.randint(-2, 2)))
    grid[rm - 1 : rm + 2, cm - 1 : cm + 2] = SQUARE

    ray_lengths = dict(zip(DIRECTIONS, random.sample(list(range(2, 7)), 4)))
    blockers = {
        "up": (rm - ray_lengths["up"] - 2, cm),
        "right": (rm, cm + ray_lengths["right"] + 2),
        "down": (rm + ray_lengths["down"] + 2, cm),
        "left": (rm, cm - ray_lengths["left"] - 2),
    }
    for r, c in blockers.values():
        grid[r, c] = BLOCKER

    colors = list(PAIR_COLORS)
    random.shuffle(colors)
    color_by_direction = dict(zip(DIRECTIONS, colors))

    rect_side_by_direction: dict[str, int] = {}
    eligible = [d for d in DIRECTIONS if ray_lengths[d] >= 3]
    forced_inner = random.choice(eligible) if eligible else None
    for direction in DIRECTIONS:
        choices = [1]
        if ray_lengths[direction] >= 3:
            choices.append(-1)
        rect_side_by_direction[direction] = random.choice(choices)
    if forced_inner is not None:
        rect_side_by_direction[forced_inner] = -1

    expected: dict[str, tuple[int, set[tuple[int, int]], set[tuple[int, int]], int]] = {}
    for direction in DIRECTIONS:
        placement = _place_pair(
            grid,
            (rm, cm),
            direction,
            color_by_direction[direction],
            ray_lengths[direction],
            rect_side_by_direction[direction],
            random.randint(2, 3),
        )
        if placement is None:
            return None
        anchor, mover, shifted = placement
        expected[direction] = (color_by_direction[direction], anchor, shifted, ray_lengths[direction])

    if not _semantic_ok(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ray-length fragment-translation input")
