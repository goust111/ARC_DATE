from __future__ import annotations

import random
from collections import Counter

import numpy as np


Direction = tuple[int, int]
Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


ARROW_PATTERNS: dict[frozenset[Coord], tuple[Direction, Coord]] = {
    frozenset({(0, 2), (1, 0), (1, 1), (1, 2), (2, 2)}): ((0, 1), (1, 2)),
    frozenset({(0, 0), (1, 0), (1, 1), (1, 2), (2, 0)}): ((0, -1), (1, 0)),
    frozenset({(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)}): ((-1, 0), (0, 1)),
    frozenset({(0, 1), (1, 1), (2, 0), (2, 1), (2, 2)}): ((1, 0), (2, 1)),
}

DIRS: tuple[Direction, ...] = ((0, 1), (0, -1), (-1, 0), (1, 0))
COLOR_COUNT = 10
GENERATION_ATTEMPTS = 500
GRID_SIZE_RANGE = (12, 30)
PAIR_COUNT_RANGE = (1, 4)
GAP_RANGE = (1, 7)
PAIR_PADDING = 1


def _most_common_color(grid: np.ndarray) -> int:
    counts = Counter(int(v) for v in grid.ravel())
    return int(counts.most_common(1)[0][0])


def _same_color_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []

    for sr, sc in np.argwhere(grid != int(bg)):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue

        color = int(grid[sr, sc])
        stack = [(sr, sc)]
        seen[sr, sc] = True
        cells: list[Coord] = []

        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if not (0 <= rr < h and 0 <= cc < w):
                    continue
                if seen[rr, cc] or int(grid[rr, cc]) != color:
                    continue
                seen[rr, cc] = True
                stack.append((int(rr), int(cc)))

        rows = [r for r, _c in cells]
        cols = [c for _r, c in cells]
        comps.append(
            {
                "color": color,
                "coords": cells,
                "bbox": (min(rows), min(cols), max(rows), max(cols)),
                "area": len(cells),
            }
        )

    return comps


def _classify_arrow(comp: dict[str, object]) -> tuple[Direction, Coord] | None:
    if int(comp["area"]) != 5:
        return None
    r0, c0, r1, c1 = map(int, comp["bbox"])
    if (r1 - r0 + 1, c1 - c0 + 1) != (3, 3):
        return None

    local = frozenset((int(r) - r0, int(c) - c0) for r, c in comp["coords"])
    spec = ARROW_PATTERNS.get(local)
    if spec is None:
        return None

    direction, tip_offset = spec
    tip = (int(r0 + tip_offset[0]), int(c0 + tip_offset[1]))
    return direction, tip


def _is_target_block(comp: dict[str, object]) -> bool:
    if int(comp["area"]) != 9:
        return False
    r0, c0, r1, c1 = map(int, comp["bbox"])
    if (r1 - r0 + 1, c1 - c0 + 1) != (3, 3):
        return False

    return {
        (int(r) - r0, int(c) - c0)
        for r, c in comp["coords"]
    } == {(r, c) for r in range(3) for c in range(3)}


def _parse_objects(grid: np.ndarray, bg: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    arrows: list[dict[str, object]] = []
    targets: list[dict[str, object]] = []

    for comp in _same_color_components(grid, bg):
        arrow = _classify_arrow(comp)
        if arrow is not None:
            direction, tip = arrow
            item = dict(comp)
            item["direction"] = direction
            item["tip"] = tip
            arrows.append(item)
            continue
        if _is_target_block(comp):
            targets.append(comp)

    assert arrows, "expected at least one arrow"
    assert len(targets) >= len(arrows), "expected target blocks for the arrows"
    return arrows, targets


def _target_lookup(targets: list[dict[str, object]]) -> dict[Coord, int]:
    lookup: dict[Coord, int] = {}
    for tid, target in enumerate(targets):
        for r, c in target["coords"]:
            lookup[(int(r), int(c))] = int(tid)
    return lookup


def _ray_target(
    grid: np.ndarray,
    bg: int,
    targets: list[dict[str, object]],
    target_at: dict[Coord, int],
    tip: Coord,
    direction: Direction,
) -> tuple[int, Coord]:
    h, w = map(int, grid.shape)
    dr, dc = map(int, direction)
    rr, cc = int(tip[0] + dr), int(tip[1] + dc)

    while 0 <= rr < h and 0 <= cc < w:
        if int(grid[rr, cc]) == int(bg):
            rr += dr
            cc += dc
            continue

        hit = (int(rr), int(cc))
        assert hit in target_at, "the arrow ray must hit a target block first"
        tid = int(target_at[hit])
        tr0, tc0, tr1, tc1 = map(int, targets[tid]["bbox"])

        if direction == (0, 1):
            assert cc == tc0 and tr0 <= rr <= tr1
            return tid, (int(tip[0]), int(tc0 - 1))
        if direction == (0, -1):
            assert cc == tc1 and tr0 <= rr <= tr1
            return tid, (int(tip[0]), int(tc1 + 1))
        if direction == (1, 0):
            assert rr == tr0 and tc0 <= cc <= tc1
            return tid, (int(tr0 - 1), int(tip[1]))

        assert direction == (-1, 0)
        assert rr == tr1 and tc0 <= cc <= tc1
        return tid, (int(tr1 + 1), int(tip[1]))

    raise AssertionError("each arrow must see one target block in its pointing direction")


def _paint_target_stripe(out: np.ndarray, target: dict[str, object], arrow_color: int) -> None:
    target_color = int(target["color"])
    r0, c0, r1, c1 = map(int, target["bbox"])
    out[r0 : r1 + 1, c0 : c1 + 1] = target_color
    out[r0 + 1, c0 : c1 + 1] = int(arrow_color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each directional arrow along its ray until it docks beside a target block, then stripe the target.

    Input structure:
    - The most common color is the background.
    - Foreground objects are monochromatic 3x3 arrows and solid 3x3 square
      target blocks, separated by background.
    - Each arrow is the five-pixel right-arrow sprite, possibly rotated by a
      quarter turn, and exactly one target block lies ahead of its tip through
      background cells.

    Rule:
    - Decode each arrow direction from its rotated 3x3 shape and cast a ray from
      the arrow tip in that direction.
    - Move the arrow forward until its tip is adjacent to the near edge of the
      first target block on that ray, clearing the arrow's old cells to the
      background.
    - Recolor the hit target as horizontal stripes: its top and bottom rows keep
      the original target color, while its middle row takes the arrow color.

    Key insight:
    - The small arrow shape supplies both the moving object and its direction;
      the visible target on the same row or column supplies the stopping point
      and receives the arrow color as the middle stripe.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _most_common_color(g)

    arrows, targets = _parse_objects(g, bg)
    target_at = _target_lookup(targets)
    out = g.copy()

    moves: list[tuple[dict[str, object], int, int, int]] = []
    activated: set[int] = set()
    for arrow in arrows:
        direction = arrow["direction"]
        tip = arrow["tip"]
        tid, new_tip = _ray_target(g, bg, targets, target_at, tip, direction)
        assert tid not in activated, "a target block must be activated by only one arrow"
        activated.add(int(tid))
        assert int(targets[tid]["color"]) != int(arrow["color"])

        shift_r = int(new_tip[0] - tip[0])
        shift_c = int(new_tip[1] - tip[1])
        dr, dc = map(int, direction)
        assert (shift_r, shift_c) == (dr * max(abs(shift_r), abs(shift_c)), dc * max(abs(shift_r), abs(shift_c)))
        moves.append((arrow, int(tid), int(shift_r), int(shift_c)))

    for arrow, _tid, _sr, _sc in moves:
        for r, c in arrow["coords"]:
            out[int(r), int(c)] = int(bg)

    for arrow, tid, _sr, _sc in moves:
        _paint_target_stripe(out, targets[int(tid)], int(arrow["color"]))

    h, w = map(int, g.shape)
    for arrow, _tid, shift_r, shift_c in moves:
        color = int(arrow["color"])
        for r, c in arrow["coords"]:
            rr = int(r + shift_r)
            cc = int(c + shift_c)
            assert 0 <= rr < h and 0 <= cc < w
            out[rr, cc] = color

    return out


# --------------- generator ---------------


def _dominant_bg(grid: np.ndarray, bg: int) -> bool:
    counts = Counter(int(v) for v in grid.ravel())
    bg_count = int(counts.get(int(bg), 0))
    return all(bg_count > int(count) for color, count in counts.items() if int(color) != int(bg))


def _arrow_offsets(direction: Direction) -> frozenset[Coord]:
    for pattern, (pat_dir, _tip) in ARROW_PATTERNS.items():
        if pat_dir == direction:
            return pattern
    raise ValueError("unknown arrow direction")


def _pair_patch(
    bg: int,
    arrow_color: int,
    target_color: int,
    direction: Direction,
    gap: int,
) -> np.ndarray:
    if direction in ((0, 1), (0, -1)):
        patch = np.full((3, int(gap) + 6), int(bg), dtype=int)
        arrow_origin = (0, 0) if direction == (0, 1) else (0, int(gap) + 3)
        target_origin = (0, int(gap) + 3) if direction == (0, 1) else (0, 0)
    else:
        patch = np.full((int(gap) + 6, 3), int(bg), dtype=int)
        arrow_origin = (0, 0) if direction == (1, 0) else (int(gap) + 3, 0)
        target_origin = (int(gap) + 3, 0) if direction == (1, 0) else (0, 0)

    ar, ac = map(int, arrow_origin)
    for dr, dc in _arrow_offsets(direction):
        patch[ar + int(dr), ac + int(dc)] = int(arrow_color)

    tr, tc = map(int, target_origin)
    patch[tr : tr + 3, tc : tc + 3] = int(target_color)
    return patch


def _can_place(blocked: np.ndarray, row: int, col: int, height: int, width: int) -> bool:
    h, w = map(int, blocked.shape)
    r0 = max(0, int(row) - PAIR_PADDING)
    c0 = max(0, int(col) - PAIR_PADDING)
    r1 = min(h, int(row) + int(height) + PAIR_PADDING)
    c1 = min(w, int(col) + int(width) + PAIR_PADDING)
    return not bool(blocked[r0:r1, c0:c1].any())


def _reserve(blocked: np.ndarray, row: int, col: int, height: int, width: int) -> None:
    h, w = map(int, blocked.shape)
    r0 = max(0, int(row) - PAIR_PADDING)
    c0 = max(0, int(col) - PAIR_PADDING)
    r1 = min(h, int(row) + int(height) + PAIR_PADDING)
    c1 = min(w, int(col) + int(width) + PAIR_PADDING)
    blocked[r0:r1, c0:c1] = True


def _place_patch(grid: np.ndarray, blocked: np.ndarray, patch: np.ndarray, bg: int) -> bool:
    rows, cols = map(int, grid.shape)
    ph, pw = map(int, patch.shape)
    locs: list[Coord] = []
    for r in range(rows - ph + 1):
        for c in range(cols - pw + 1):
            if _can_place(blocked, r, c, ph, pw):
                locs.append((int(r), int(c)))
    if not locs:
        return False

    r0, c0 = random.choice(locs)
    mask = patch != int(bg)
    grid[r0 : r0 + ph, c0 : c0 + pw][mask] = patch[mask]
    _reserve(blocked, r0, c0, ph, pw)
    return True


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(*GRID_SIZE_RANGE)
    cols = random.randint(*GRID_SIZE_RANGE)
    bg = random.randrange(COLOR_COUNT)

    max_pairs_by_area = max(1, min(PAIR_COUNT_RANGE[1], (rows * cols) // 85))
    pair_count = random.randint(PAIR_COUNT_RANGE[0], max_pairs_by_area)

    grid = np.full((rows, cols), int(bg), dtype=int)
    blocked = np.zeros((rows, cols), dtype=bool)
    palette = [color for color in range(COLOR_COUNT) if color != int(bg)]
    directions = random.sample(DIRS, int(pair_count))

    for direction in directions:
        placed = False
        for _ in range(80):
            arrow_color = random.choice(palette)
            target_color = random.choice([c for c in palette if int(c) != int(arrow_color)])
            gap = random.randint(*GAP_RANGE)
            patch = _pair_patch(bg, arrow_color, target_color, direction, gap)
            if patch.shape[0] > rows or patch.shape[1] > cols:
                continue
            if _place_patch(grid, blocked, patch, bg):
                placed = True
                break
        if not placed:
            return None

    if not _dominant_bg(grid, bg):
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    if out.ndim != 2 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arrow docking input")
