from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]

DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

# Tip is always at (0, 0). The two bar pixels sit one step behind it and are
# separated along the perpendicular axis, which makes orientation unambiguous.
ARROW_OFFSETS = {
    DIRS["up"]: [(0, 0), (1, -1), (1, 1)],
    DIRS["down"]: [(0, 0), (-1, -1), (-1, 1)],
    DIRS["left"]: [(0, 0), (-1, 1), (1, 1)],
    DIRS["right"]: [(0, 0), (-1, -1), (1, -1)],
}

ARROW_PATTERNS: dict[frozenset[tuple[int, int]], tuple[tuple[int, int], tuple[int, int]]] = {}
for direction, rels in ARROW_OFFSETS.items():
    rmin = min(r for r, _ in rels)
    cmin = min(c for _, c in rels)
    norm = frozenset((r - rmin, c - cmin) for r, c in rels)
    tip_offset = (-rmin, -cmin)
    ARROW_PATTERNS[norm] = (direction, tip_offset)


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int) -> Iterable[tuple[int, int]]:
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _components(mask: np.ndarray, conn8: bool) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    neigh = _neighbors8 if conn8 else _neighbors4
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in neigh(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _copy_shapes(shapes: list[dict]) -> list[dict]:
    return [{"cells": set(shape["cells"]), "color": int(shape["color"])} for shape in shapes]


def _shape_bbox(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _rect_cells(top: int, left: int, height: int, width: int) -> set[tuple[int, int]]:
    return {(top + dr, left + dc) for dr in range(height) for dc in range(width)}


def _translate(cells: set[tuple[int, int]], dr: int, dc: int, steps: int) -> set[tuple[int, int]]:
    return {(r + dr * steps, c + dc * steps) for r, c in cells}


def _distance_to_target_edge(tip: tuple[int, int], direction: tuple[int, int], h: int, w: int) -> int:
    r, c = tip
    dr, dc = direction
    if dr == -1:
        return r
    if dr == 1:
        return h - 1 - r
    if dc == -1:
        return c
    return w - 1 - c


def _sort_arrows(arrows: list[dict], h: int, w: int) -> list[dict]:
    return sorted(
        arrows,
        key=lambda arrow: (
            _distance_to_target_edge(arrow["tip"], arrow["dir"], h, w),
            arrow["tip"][0],
            arrow["tip"][1],
        ),
    )


def _find_arrows(grid: np.ndarray) -> list[dict]:
    arrows: list[dict] = []
    colors = [int(color) for color in np.unique(grid) if int(color) not in (BG, GREY)]
    for color in colors:
        for comp in _components(grid == color, conn8=True):
            if len(comp) != 3:
                continue
            rmin, cmin, rmax, cmax = _shape_bbox(set(comp))
            norm = frozenset((r - rmin, c - cmin) for r, c in comp)
            if norm not in ARROW_PATTERNS:
                continue
            direction, tip_offset = ARROW_PATTERNS[norm]
            tip = (rmin + tip_offset[0], cmin + tip_offset[1])
            arrows.append(
                {
                    "color": color,
                    "dir": direction,
                    "tip": tip,
                    "cells": set(comp),
                }
            )
    return arrows


def _parse_scene(grid: np.ndarray) -> tuple[list[dict], list[dict], dict[tuple[int, int], int]]:
    g = np.asarray(grid, dtype=int)
    arrows = _find_arrows(g)
    arrow_cells = set().union(*(arrow["cells"] for arrow in arrows)) if arrows else set()

    shape_mask = (g != BG).copy()
    for r, c in arrow_cells:
        shape_mask[r, c] = False

    shapes: list[dict] = []
    dots: dict[tuple[int, int], int] = {}
    for comp in _components(shape_mask, conn8=False):
        colors = {int(g[r, c]) for r, c in comp}
        if len(colors) != 1:
            raise AssertionError("non-arrow objects must be monochrome")
        color = next(iter(colors))
        if len(comp) == 1:
            dots[comp[0]] = color
        else:
            shapes.append({"cells": set(comp), "color": color})
    return arrows, shapes, dots


def _render_scene(h: int, w: int, shapes: list[dict], dots: dict[tuple[int, int], int]) -> np.ndarray:
    out = np.full((h, w), BG, dtype=int)
    for shape in shapes:
        for r, c in shape["cells"]:
            out[r, c] = int(shape["color"])
    for (r, c), color in dots.items():
        out[r, c] = int(color)
    return out


def _simulate_scene(
    h: int,
    w: int,
    shapes: list[dict],
    arrows: list[dict],
    dots: dict[tuple[int, int], int],
    order: list[dict] | None = None,
) -> tuple[list[dict], dict[tuple[int, int], int], list[dict]]:
    cur_shapes = _copy_shapes(shapes)
    cur_dots = dict(dots)
    ordered = _sort_arrows(arrows, h, w) if order is None else list(order)
    trace: list[dict] = []

    for arrow in ordered:
        occ: dict[tuple[int, int], int] = {}
        for idx, shape in enumerate(cur_shapes):
            for cell in shape["cells"]:
                occ[cell] = idx

        dr, dc = arrow["dir"]
        rr, cc = arrow["tip"][0] + dr, arrow["tip"][1] + dc
        target_idx = None
        while 0 <= rr < h and 0 <= cc < w:
            if (rr, cc) in occ:
                target_idx = occ[(rr, cc)]
                break
            rr += dr
            cc += dc

        event = {
            "arrow": dict(arrow),
            "target_idx": target_idx,
            "move_steps": 0,
            "swept": set(),
            "covered": set(),
            "same_color_hits": 0,
        }
        if target_idx is None:
            trace.append(event)
            continue

        target = cur_shapes[target_idx]
        other_occ = {cell for idx, shape in enumerate(cur_shapes) if idx != target_idx for cell in shape["cells"]}

        steps = 0
        while True:
            moved = _translate(target["cells"], dr, dc, steps + 1)
            if any(not (0 <= r < h and 0 <= c < w) for r, c in moved):
                break
            if any(cell in other_occ for cell in moved):
                break
            steps += 1

        swept: set[tuple[int, int]] = set()
        covered: set[tuple[int, int]] = set()
        for step in range(1, steps + 1):
            moved = _translate(target["cells"], dr, dc, step)
            swept |= moved
            covered |= {cell for cell in moved if cell in cur_dots}

        same_color_hits = sum(1 for cell in covered if cur_dots[cell] == arrow["color"])
        for cell in covered:
            del cur_dots[cell]

        target["cells"] = _translate(target["cells"], dr, dc, steps)
        if same_color_hits > 0:
            target["color"] = int(arrow["color"])

        event["move_steps"] = steps
        event["swept"] = swept
        event["covered"] = covered
        event["same_color_hits"] = same_color_hits
        trace.append(event)

    return cur_shapes, cur_dots, trace


def main(grid: np.ndarray) -> np.ndarray:
    """
    Multiple colored arrows point in different directions. Each arrow pushes the
    first figure in front of it, the figure slides until blocked by another
    figure or the grid edge, and swept same-color dots recolor the final figure.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    arrows, shapes, dots = _parse_scene(g)
    final_shapes, final_dots, _trace = _simulate_scene(h, w, shapes, arrows, dots)
    return _render_scene(h, w, final_shapes, final_dots)


def _arrow_cells(tip: tuple[int, int], direction: tuple[int, int]) -> set[tuple[int, int]]:
    tr, tc = tip
    return {(tr + dr, tc + dc) for dr, dc in ARROW_OFFSETS[direction]}


def _chain_scene(direction: tuple[int, int], h: int, w: int, count: int) -> tuple[list[dict], list[dict]] | None:
    if direction in (DIRS["right"], DIRS["left"]):
        thickness = random.randint(2, 4)
        if h < thickness + 2:
            return None
        top = random.randint(1, h - thickness - 1)
        anchor_row = top + random.randint(0, thickness - 1)
        lengths = [random.randint(2, 4) for _ in range(count)]
        outer_gap = random.randint(2, 4)
        front_move = random.randint(3, 5)
        inter_gaps = [random.randint(5, 8) for _ in range(count - 1)]
        arrow_gaps = [random.randint(1, 2) for _ in range(count)]
        lefts = [0] * count

        if direction == DIRS["right"]:
            lefts[-1] = w - lengths[-1] - front_move
            for idx in range(count - 2, -1, -1):
                lefts[idx] = lefts[idx + 1] - inter_gaps[idx] - lengths[idx]
            if lefts[0] - outer_gap - 2 < 0:
                return None
            tip_cols = [0] * count
            tip_cols[0] = lefts[0] - outer_gap - 1
            for idx in range(1, count):
                tip_cols[idx] = lefts[idx] - arrow_gaps[idx] - 1
        else:
            lefts[0] = front_move
            for idx in range(1, count):
                lefts[idx] = lefts[idx - 1] + lengths[idx - 1] + inter_gaps[idx - 1]
            right_last = lefts[-1] + lengths[-1] - 1
            if right_last + outer_gap + 2 >= w:
                return None
            tip_cols = [0] * count
            tip_cols[-1] = right_last + outer_gap + 1
            for idx in range(count - 1):
                right_edge = lefts[idx] + lengths[idx] - 1
                tip_cols[idx] = right_edge + arrow_gaps[idx] + 1

        shapes = []
        arrows = []
        for idx in range(count):
            cells = _rect_cells(top, lefts[idx], thickness, lengths[idx])
            shapes.append({"cells": cells, "color": GREY})
            arrows.append(
                {
                    "color": PALETTE[idx],
                    "dir": direction,
                    "tip": (anchor_row, tip_cols[idx]),
                    "cells": _arrow_cells((anchor_row, tip_cols[idx]), direction),
                }
            )
    else:
        thickness = random.randint(2, 4)
        if w < thickness + 2:
            return None
        left = random.randint(1, w - thickness - 1)
        anchor_col = left + random.randint(0, thickness - 1)
        lengths = [random.randint(2, 4) for _ in range(count)]
        outer_gap = random.randint(2, 4)
        front_move = random.randint(3, 5)
        inter_gaps = [random.randint(5, 8) for _ in range(count - 1)]
        arrow_gaps = [random.randint(1, 2) for _ in range(count)]
        tops = [0] * count

        if direction == DIRS["down"]:
            tops[-1] = h - lengths[-1] - front_move
            for idx in range(count - 2, -1, -1):
                tops[idx] = tops[idx + 1] - inter_gaps[idx] - lengths[idx]
            if tops[0] - outer_gap - 2 < 0:
                return None
            tip_rows = [0] * count
            tip_rows[0] = tops[0] - outer_gap - 1
            for idx in range(1, count):
                tip_rows[idx] = tops[idx] - arrow_gaps[idx] - 1
        else:
            tops[0] = front_move
            for idx in range(1, count):
                tops[idx] = tops[idx - 1] + lengths[idx - 1] + inter_gaps[idx - 1]
            bottom_last = tops[-1] + lengths[-1] - 1
            if bottom_last + outer_gap + 2 >= h:
                return None
            tip_rows = [0] * count
            tip_rows[-1] = bottom_last + outer_gap + 1
            for idx in range(count - 1):
                bottom_edge = tops[idx] + lengths[idx] - 1
                tip_rows[idx] = bottom_edge + arrow_gaps[idx] + 1

        shapes = []
        arrows = []
        for idx in range(count):
            cells = _rect_cells(tops[idx], left, lengths[idx], thickness)
            shapes.append({"cells": cells, "color": GREY})
            arrows.append(
                {
                    "color": PALETTE[idx],
                    "dir": direction,
                    "tip": (tip_rows[idx], anchor_col),
                    "cells": _arrow_cells((tip_rows[idx], anchor_col), direction),
                }
            )

    all_shape_cells = set().union(*(shape["cells"] for shape in shapes))
    all_arrow_cells = set().union(*(arrow["cells"] for arrow in arrows))
    if all_shape_cells & all_arrow_cells:
        return None
    if len(all_shape_cells) != sum(len(shape["cells"]) for shape in shapes):
        return None
    if len(all_arrow_cells) != 3 * len(arrows):
        return None
    if any(not (0 <= r < h and 0 <= c < w) for r, c in all_shape_cells | all_arrow_cells):
        return None

    return shapes, arrows


def _input_grid(h: int, w: int, shapes: list[dict], arrows: list[dict], dots: dict[tuple[int, int], int]) -> np.ndarray:
    grid = _render_scene(h, w, shapes, dots)
    for arrow in arrows:
        for r, c in arrow["cells"]:
            grid[r, c] = int(arrow["color"])
    return grid


def _far_enough(cell: tuple[int, int], occupied: set[tuple[int, int]]) -> bool:
    r, c = cell
    for rr, cc in occupied:
        if max(abs(r - rr), abs(c - cc)) <= 1:
            return False
    return True


def _pick_spaced_cells(
    candidates: list[tuple[int, int]],
    blocked: set[tuple[int, int]],
    count: int,
) -> list[tuple[int, int]] | None:
    picks: list[tuple[int, int]] = []
    pool = candidates[:]
    random.shuffle(pool)
    local_blocked = set(blocked)
    for cell in pool:
        if not _far_enough(cell, local_blocked):
            continue
        picks.append(cell)
        local_blocked.add(cell)
        if len(picks) == count:
            return picks
    return None


def _generate_once() -> np.ndarray | None:
    h = random.randint(14, 21)
    w = random.randint(14, 21)
    count = random.choice([2, 3])
    direction = random.choice(list(DIRS.values()))
    color_order = random.sample(PALETTE, count)

    scene = _chain_scene(direction, h, w, count)
    if scene is None:
        return None
    shapes, arrows = scene
    for arrow, color in zip(arrows, color_order):
        arrow["color"] = color

    empty_dots: dict[tuple[int, int], int] = {}
    sorted_arrows = _sort_arrows(arrows, h, w)
    _, _, dry_trace = _simulate_scene(h, w, shapes, sorted_arrows, empty_dots, order=sorted_arrows)

    if len(dry_trace) != count or any(event["target_idx"] is None or event["move_steps"] <= 0 for event in dry_trace):
        return None

    base_grid = _input_grid(h, w, shapes, arrows, {})
    background_cells = {(r, c) for r in range(h) for c in range(w) if base_grid[r, c] == BG}
    shape_cells = set().union(*(shape["cells"] for shape in shapes))
    arrow_cells_by_color = {int(arrow["color"]): set(arrow["cells"]) for arrow in arrows}

    absorb_flags = [False] * count
    true_count = 1 if count == 2 else random.choice([1, 2])
    for idx in random.sample(range(count), true_count):
        absorb_flags[idx] = True

    dots: dict[tuple[int, int], int] = {}
    blocked_for_dots = set(shape_cells)
    earlier_swept: set[tuple[int, int]] = set()
    all_swept = set().union(*(event["swept"] for event in dry_trace))

    for event, absorb in zip(dry_trace, absorb_flags):
        color = int(event["arrow"]["color"])
        candidates = [
            cell
            for cell in event["swept"]
            if cell in background_cells
            and cell not in earlier_swept
            and _far_enough(cell, shape_cells)
            and _far_enough(cell, arrow_cells_by_color[color])
        ]
        if absorb:
            if not candidates:
                return None
            need = random.randint(1, min(3, len(candidates)))
            picks = _pick_spaced_cells(candidates, blocked_for_dots, need)
            if picks is None:
                return None
            for cell in picks:
                dots[cell] = color
                blocked_for_dots.add(cell)
        earlier_swept |= event["swept"]

    stray_count = random.randint(0, 2)
    if stray_count:
        for _ in range(stray_count):
            color = random.choice(color_order)
            stray_candidates = [
                (r, c)
                for r in range(h)
                for c in range(w)
                if (r, c) in background_cells
                and (r, c) not in all_swept
                and _far_enough((r, c), shape_cells)
                and _far_enough((r, c), arrow_cells_by_color[color])
            ]
            stray_pick = _pick_spaced_cells(stray_candidates, blocked_for_dots, 1)
            if stray_pick is None:
                break
            dots[stray_pick[0]] = color
            blocked_for_dots.add(stray_pick[0])

    grid = _input_grid(h, w, shapes, arrows, dots)
    out = main(grid)

    sim_shapes, sim_dots, real_trace = _simulate_scene(h, w, shapes, arrows, dots)
    expected = _render_scene(h, w, sim_shapes, sim_dots)
    if not np.array_equal(out, expected):
        return None

    same_hits = [event["same_color_hits"] for event in real_trace]
    if not any(hit > 0 for hit in same_hits):
        return None
    if not any(hit == 0 for hit in same_hits):
        return None

    reversed_shapes, reversed_dots, _ = _simulate_scene(
        h,
        w,
        shapes,
        arrows,
        dots,
        order=list(reversed(sorted_arrows)),
    )
    reversed_out = _render_scene(h, w, reversed_shapes, reversed_dots)
    if np.array_equal(out, reversed_out):
        return None

    if np.array_equal(grid, out):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _generate_once()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after 500 attempts")
