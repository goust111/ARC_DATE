from __future__ import annotations

import random
from collections import defaultdict

import numpy as np

Coord = tuple[int, int]
Direction = tuple[int, int]

BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
DIRS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]
ARROW_OFFSETS: dict[Direction, tuple[Coord, Coord, Coord]] = {
    (-1, 0): ((0, 0), (1, -1), (1, 1)),
    (0, 1): ((0, 0), (-1, -1), (1, -1)),
    (1, 0): ((0, 0), (-1, -1), (-1, 1)),
    (0, -1): ((0, 0), (-1, 1), (1, 1)),
}
ARROW_PATTERNS: dict[frozenset[Coord], tuple[Direction, Coord]] = {
    frozenset({(0, 1), (1, 0), (1, 2)}): ((-1, 0), (0, 1)),
    frozenset({(0, 0), (1, 1), (2, 0)}): ((0, 1), (1, 1)),
    frozenset({(0, 0), (0, 2), (1, 1)}): ((1, 0), (1, 1)),
    frozenset({(0, 1), (1, 0), (2, 1)}): ((0, -1), (1, 0)),
}

GRID_H_RANGE = (15, 29)
GRID_W_RANGE = (15, 29)
PAIR_COUNT_RANGE = (2, 4)
SUCCESS_TARGET_DIST = (3, 9)
BLOCKER_DIST = (3, 7)
TARGET_AFTER_BLOCKER = (2, 6)
SCENE_ATTEMPTS = 300


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []

            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS8:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "area": int(arr.shape[0]),
                }
            )
    return components


def _detect_arrow(coords: np.ndarray) -> tuple[Direction, Coord] | None:
    if int(coords.shape[0]) != 3:
        return None
    r0, c0 = coords.min(axis=0)
    norm = frozenset((int(r - r0), int(c - c0)) for r, c in coords)
    found = ARROW_PATTERNS.get(norm)
    if found is None:
        return None
    direction, tip_norm = found
    return direction, (int(r0 + tip_norm[0]), int(c0 + tip_norm[1]))


def _arrow_cells(tip: Coord, direction: Direction) -> set[Coord]:
    tr, tc = tip
    return {(tr + dr, tc + dc) for dr, dc in ARROW_OFFSETS[direction]}


def _forward_aligned(tip: Coord, target: Coord, direction: Direction) -> bool:
    dr, dc = direction
    tr, tc = tip
    rr, cc = target
    if dr != 0:
        return cc == tc and (rr - tr) * dr > 0
    return rr == tr and (cc - tc) * dc > 0


def _simulate_arrow(
    shape: np.ndarray,
    tip: Coord,
    direction: Direction,
    target: Coord,
    occupied: set[Coord],
    shape_hw: tuple[int, int],
) -> tuple[bool, np.ndarray]:
    if not _forward_aligned(tip, target, direction):
        return False, shape.copy()

    h, w = shape_hw
    dr, dc = direction
    current = shape.copy()
    current_tip = tip

    while True:
        next_tip = (current_tip[0] + dr, current_tip[1] + dc)
        if next_tip == target:
            return True, current

        shifted = current + np.array([dr, dc], dtype=int)
        blocked = False
        for rr, cc in shifted:
            r_i, c_i = int(rr), int(cc)
            if not (0 <= r_i < h and 0 <= c_i < w):
                blocked = True
                break
            if (r_i, c_i) in occupied:
                blocked = True
                break
        if blocked:
            return False, current

        current = shifted
        current_tip = next_tip


def _draw_box(out: np.ndarray, center: Coord, color: int) -> None:
    h, w = out.shape
    cr, cc = center
    for rr in range(cr - 1, cr + 2):
        for cc2 in range(cc - 1, cc + 2):
            if 0 <= rr < h and 0 <= cc2 < w and (rr != cr or cc2 != cc):
                out[rr, cc2] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each three-pixel arrow toward its same-color dot, removing arrivals and boxing the target.

    Input structure:
    - The background is black and the non-background pixels form hollow triangular arrows and single-pixel colored dots.
    - Each arrow is a three-pixel triangle: one tip pixel plus two base-corner pixels whose line is perpendicular to the arrow direction.
    - Every moving arrow color has exactly one same-color target dot; additional unmatched dots may appear as blockers.

    Rule:
    - Detect each arrow's direction from which side the lone tip pixel sits relative to the two-pixel base.
    - Slide the whole arrow step by step along that direction until the next tip position would be its same-color target dot.
    - If another non-background object blocks the next translated arrow position first, keep the arrow at the last free position and do not draw a box.
    - Otherwise erase the arrow and draw a same-color 3x3 frame around the target dot, leaving the center target in place.

    Key insight:
    - The three-pixel triangle encodes motion direction, while color matches each moving arrow to the one dot that can trigger its disappearance and surrounding frame.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)

    components = _find_components(g, bg)
    arrows_by_color: dict[int, list[dict]] = defaultdict(list)
    dots_by_color: dict[int, list[Coord]] = defaultdict(list)

    for comp in components:
        color = int(comp["color"])
        coords = comp["coords"]
        if int(comp["area"]) == 1:
            dots_by_color[color].append((int(coords[0, 0]), int(coords[0, 1])))
            continue

        found = _detect_arrow(coords)
        if found is None:
            continue
        direction, tip = found
        arrows_by_color[color].append(
            {
                "coords": coords,
                "direction": direction,
                "tip": tip,
            }
        )

    occupied = set(map(tuple, np.argwhere(g != bg)))
    out = g.copy()
    paired: list[tuple[int, dict, Coord]] = []

    for color, arrows in arrows_by_color.items():
        dots = dots_by_color.get(color, [])
        if len(arrows) != 1 or len(dots) != 1:
            continue
        arrow = arrows[0]
        paired.append((color, arrow, dots[0]))
        for rr, cc in arrow["coords"]:
            out[int(rr), int(cc)] = bg

    for color, arrow, target in paired:
        arrow_cells = set((int(r), int(c)) for r, c in arrow["coords"])
        success, final_coords = _simulate_arrow(
            arrow["coords"],
            arrow["tip"],
            arrow["direction"],
            target,
            occupied - arrow_cells,
            g.shape,
        )
        if success:
            _draw_box(out, target, color)
        else:
            out[final_coords[:, 0], final_coords[:, 1]] = color

    return out


def _shift_cells(cells: set[Coord], shift: Coord) -> set[Coord]:
    sr, sc = shift
    return {(r + sr, c + sc) for r, c in cells}


def _relative_path_cells(direction: Direction, steps: int) -> set[Coord]:
    dr, dc = direction
    rel: set[Coord] = set()
    for step in range(steps + 1):
        for rr, cc in ARROW_OFFSETS[direction]:
            rel.add((rr + step * dr, cc + step * dc))
    return rel


def _relative_square(center_rel: Coord) -> set[Coord]:
    cr, cc = center_rel
    return {(cr + dr, cc + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _make_statuses(pair_count: int) -> list[bool]:
    if pair_count == 2:
        statuses = [True, False]
    elif pair_count == 3:
        statuses = [True, True, False] if random.random() < 0.5 else [True, False, False]
    else:
        statuses = [True, True, False, False]
    random.shuffle(statuses)
    return statuses


def _translate_rel(rel_cells: set[Coord], tip: Coord) -> set[Coord]:
    return _shift_cells(rel_cells, tip)


def _choose_tip(h: int, w: int, rel_reserved: set[Coord], reserved: set[Coord]) -> Coord | None:
    rel_rows = [r for r, _ in rel_reserved]
    rel_cols = [c for _, c in rel_reserved]
    min_r, max_r = min(rel_rows), max(rel_rows)
    min_c, max_c = min(rel_cols), max(rel_cols)

    r_lo = -min_r
    r_hi = h - 1 - max_r
    c_lo = -min_c
    c_hi = w - 1 - max_c
    if r_lo > r_hi or c_lo > c_hi:
        return None

    tips = [(r, c) for r in range(r_lo, r_hi + 1) for c in range(c_lo, c_hi + 1)]
    random.shuffle(tips)
    for tip in tips:
        if _translate_rel(rel_reserved, tip).isdisjoint(reserved):
            return tip
    return None


def _build_pair_candidate(
    h: int,
    w: int,
    color: int,
    success: bool,
    blocker_color: int | None,
    reserved: set[Coord],
) -> dict | None:
    directions = list(ARROW_OFFSETS)
    random.shuffle(directions)

    for direction in directions:
        dr, dc = direction
        if success:
            dist_choices = list(range(SUCCESS_TARGET_DIST[0], SUCCESS_TARGET_DIST[1] + 1))
            random.shuffle(dist_choices)
            for target_dist in dist_choices:
                target_rel = (dr * target_dist, dc * target_dist)
                rel_path = _relative_path_cells(direction, target_dist - 1)
                rel_box = _relative_square(target_rel)
                rel_reserved = rel_path | rel_box
                tip = _choose_tip(h, w, rel_reserved, reserved)
                if tip is None:
                    continue
                return {
                    "color": color,
                    "direction": direction,
                    "start_tip": tip,
                    "start_cells": _arrow_cells(tip, direction),
                    "target": (tip[0] + target_rel[0], tip[1] + target_rel[1]),
                    "success": True,
                    "reserved": _translate_rel(rel_reserved, tip),
                }
        else:
            block_choices = list(range(BLOCKER_DIST[0], BLOCKER_DIST[1] + 1))
            random.shuffle(block_choices)
            for block_dist in block_choices:
                extra_choices = list(range(TARGET_AFTER_BLOCKER[0], TARGET_AFTER_BLOCKER[1] + 1))
                random.shuffle(extra_choices)
                for extra in extra_choices:
                    target_dist = block_dist + extra
                    blocker_rel = (dr * block_dist, dc * block_dist)
                    target_rel = (dr * target_dist, dc * target_dist)
                    rel_path = _relative_path_cells(direction, block_dist - 1)
                    rel_reserved = rel_path | {blocker_rel, target_rel}
                    tip = _choose_tip(h, w, rel_reserved, reserved)
                    if tip is None:
                        continue
                    final_tip = (tip[0] + dr * (block_dist - 1), tip[1] + dc * (block_dist - 1))
                    return {
                        "color": color,
                        "direction": direction,
                        "start_tip": tip,
                        "start_cells": _arrow_cells(tip, direction),
                        "target": (tip[0] + target_rel[0], tip[1] + target_rel[1]),
                        "success": False,
                        "blocker": (tip[0] + blocker_rel[0], tip[1] + blocker_rel[1]),
                        "blocker_color": blocker_color,
                        "final_tip": final_tip,
                        "final_cells": _arrow_cells(final_tip, direction),
                        "reserved": _translate_rel(rel_reserved, tip),
                    }
    return None


def _render_scene(h: int, w: int, pairs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)

    for pair in pairs:
        color = int(pair["color"])
        for r, c in pair["start_cells"]:
            inp[r, c] = color
        tr, tc = pair["target"]
        inp[tr, tc] = color
        if not pair["success"]:
            br, bc = pair["blocker"]
            inp[br, bc] = int(pair["blocker_color"])

    out[:] = inp
    for pair in pairs:
        color = int(pair["color"])
        for r, c in pair["start_cells"]:
            out[r, c] = BG
        if pair["success"]:
            _draw_box(out, pair["target"], color)
        else:
            for r, c in pair["final_cells"]:
                out[r, c] = color

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    statuses = _make_statuses(pair_count)
    blocker_count = sum(not status for status in statuses)
    need_colors = pair_count + blocker_count
    if need_colors > len(PALETTE):
        return None

    colors = random.sample(PALETTE, need_colors)
    pair_colors = colors[:pair_count]
    blocker_colors = colors[pair_count:]
    blocker_iter = iter(blocker_colors)

    reserved: set[Coord] = set()
    pairs: list[dict] = []

    order = list(range(pair_count))
    random.shuffle(order)
    for idx in order:
        blocker_color = next(blocker_iter) if not statuses[idx] else None
        pair = _build_pair_candidate(
            h,
            w,
            pair_colors[idx],
            statuses[idx],
            blocker_color,
            reserved,
        )
        if pair is None:
            return None
        pairs.append(pair)
        reserved |= set(pair["reserved"])

    inp, expected = _render_scene(h, w, pairs)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(SCENE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arrow-target scene")
