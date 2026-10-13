from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
YELLOW = 4
GREY = 5
BORDER_COLORS = (1, 2, 3, 6, 7, 8, 9)
SIDE_TO_DIRS = {
    "top": [(-1, -1), (-1, 1)],
    "bottom": [(1, -1), (1, 1)],
    "left": [(-1, -1), (1, -1)],
    "right": [(-1, 1), (1, 1)],
}
SIDE_ORDER = ("top", "right", "bottom", "left")
GEN_COUNTER = 0


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if not (dr == 0 and dc == 0)
    ]


def _most_common(values: np.ndarray) -> int:
    return int(Counter(map(int, values.ravel())).most_common(1)[0][0])


def _interior_bg(grid: np.ndarray) -> int:
    interior = grid[1:-1, 1:-1]
    return _most_common(interior if interior.size else grid)


def _touch_side(tip: tuple[int, int], shape: tuple[int, int]) -> str | None:
    r, c = tip
    h, w = shape
    sides = []
    if r == 1:
        sides.append("top")
    if r == h - 2:
        sides.append("bottom")
    if c == 1:
        sides.append("left")
    if c == w - 2:
        sides.append("right")
    if len(sides) != 1:
        return None
    return sides[0]


def _border_color_at(grid: np.ndarray, side: str, tip: tuple[int, int]) -> int:
    r, c = tip
    if side == "top":
        return int(grid[0, c])
    if side == "bottom":
        return int(grid[-1, c])
    if side == "left":
        return int(grid[r, 0])
    if side == "right":
        return int(grid[r, -1])
    raise AssertionError("unknown border side")


def _object_cells_from_tip(tip: tuple[int, int], direction: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    tr, tc = tip
    dr, dc = direction
    return tuple((tr - k * dr, tc - k * dc) for k in range(4))


def _shift_cells(cells: tuple[tuple[int, int], ...], dr: int, dc: int) -> tuple[tuple[int, int], ...]:
    return tuple((r + dr, c + dc) for r, c in cells)


def _all_in_interior(cells: tuple[tuple[int, int], ...], shape: tuple[int, int]) -> bool:
    h, w = shape
    return all(1 <= r <= h - 2 and 1 <= c <= w - 2 for r, c in cells)


def _components8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
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
                for nr, nc in _neighbors8(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _extract_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = _interior_bg(g)
    assert g.shape[0] >= 7 and g.shape[1] >= 7, "grid must leave room for border bars and diagonal objects"
    assert np.all(g[0, :] != bg) and np.all(g[-1, :] != bg), "top and bottom border bars must be non-background"
    assert np.all(g[:, 0] != bg) and np.all(g[:, -1] != bg), "left and right border bars must be non-background"

    mask = np.zeros_like(g, dtype=bool)
    mask[1:-1, 1:-1] = g[1:-1, 1:-1] != bg
    comps = _components8(mask)
    assert comps, "expected at least one interior object"

    objects: list[dict] = []
    for comp in comps:
        coords = tuple(sorted(comp))
        colors = [int(g[r, c]) for r, c in coords]
        assert colors.count(YELLOW) == 1 and colors.count(GREY) == 3, "each object must be one yellow tip plus three grey diagonal cells"

        tip = next((r, c) for r, c in coords if int(g[r, c]) == YELLOW)
        grey_neighbors = [
            (r, c)
            for r, c in _neighbors8(*tip)
            if 0 <= r < g.shape[0] and 0 <= c < g.shape[1] and int(g[r, c]) == GREY and (r, c) in coords
        ]
        assert len(grey_neighbors) == 1, "yellow tip must touch exactly one grey continuation cell"
        neck = grey_neighbors[0]
        direction = (tip[0] - neck[0], tip[1] - neck[1])
        dr, dc = direction
        assert abs(dr) == 1 and abs(dc) == 1, "tip direction must be diagonal"

        expected = set(_object_cells_from_tip(tip, direction))
        assert expected == set(coords), "object cells must lie on one contiguous diagonal line"
        objects.append({"coords": coords, "tip": tip, "direction": direction})
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide every grey diagonal segment along its yellow polarity tip and recolor it by the edge-bar segment it lands against.

    Input structure:
    - The outermost row and column form 1-cell-wide colored edge bars, each split into multiple visible color segments.
    - The interior is black except for several diagonal 4-pixel launchers made of three grey cells plus one yellow tip cell on one endpoint.
    - Each launcher is an 8-connected straight diagonal chain, and the yellow tip indicates which diagonal direction it travels.

    Rule:
    - For each launcher, move the whole 4-pixel object one diagonal step at a time in the yellow tip's direction.
    - Stop when the yellow tip first enters an interior cell that is directly adjacent to exactly one edge bar.
    - Clear the launcher's original position, then repaint all four of its landed cells with the color of the contacted edge-bar segment.
    - Keep all edge bars unchanged.

    Key insight:
    - The yellow endpoint is a polarity marker, not decoration: it fixes the travel direction, while the final color is determined only by the exact border segment touched at the tip's landing position.
    """
    g = np.asarray(grid, dtype=int)
    bg = _interior_bg(g)
    out = g.copy()

    objects = _extract_objects(g)
    placements: list[tuple[tuple[tuple[int, int], ...], int]] = []
    for obj in objects:
        tip = obj["tip"]
        dr, dc = obj["direction"]
        coords = obj["coords"]
        step = 0
        cur_tip = tip
        while _touch_side(cur_tip, g.shape) is None:
            step += 1
            cur_tip = (tip[0] + step * dr, tip[1] + step * dc)
            assert 1 <= cur_tip[0] <= g.shape[0] - 2 and 1 <= cur_tip[1] <= g.shape[1] - 2, "tip must land before leaving the interior"
        side = _touch_side(cur_tip, g.shape)
        assert side is not None
        landed = tuple((r + step * dr, c + step * dc) for r, c in coords)
        color = _border_color_at(g, side, cur_tip)
        placements.append((landed, color))
        for r, c in coords:
            out[r, c] = bg

    occupied: set[tuple[int, int]] = set()
    for landed, color in placements:
        for r, c in landed:
            assert 0 <= r < g.shape[0] and 0 <= c < g.shape[1]
            assert (r, c) not in occupied, "landed objects may not overlap"
            occupied.add((r, c))
            out[r, c] = color
    return out


def _make_edge_segments(length: int) -> list[int]:
    seg_count = random.randint(2, min(4, max(2, length // 2)))
    cut_pool = list(range(2, length))
    cuts = sorted(random.sample(cut_pool, seg_count - 1))
    points = [0] + cuts + [length]
    colors: list[int] = []
    prev = None
    for _ in range(seg_count):
        choices = [color for color in BORDER_COLORS if color != prev]
        color = random.choice(choices)
        colors.append(color)
        prev = color

    values: list[int] = []
    for idx in range(seg_count):
        values.extend([colors[idx]] * (points[idx + 1] - points[idx]))
    return values


def _build_border_grid(height: int, width: int) -> tuple[np.ndarray, dict[str, list[int]]]:
    grid = np.full((height, width), BG, dtype=int)
    top_inner = _make_edge_segments(width - 2)
    bottom_inner = _make_edge_segments(width - 2)
    left_inner = _make_edge_segments(height - 2)
    right_inner = _make_edge_segments(height - 2)

    top = [top_inner[0]] + top_inner + [top_inner[-1]]
    bottom = [bottom_inner[0]] + bottom_inner + [bottom_inner[-1]]
    left = [left_inner[0]] + left_inner + [left_inner[-1]]
    right = [right_inner[0]] + right_inner + [right_inner[-1]]

    grid[0, :] = top
    grid[-1, :] = bottom
    grid[1:-1, 0] = left[1:-1]
    grid[1:-1, -1] = right[1:-1]
    return grid, {"top": top, "bottom": bottom, "left": left, "right": right}


def _axis_family(direction: tuple[int, int]) -> str:
    dr, dc = direction
    return "backslash" if dr == dc else "slash"


def _distances_to_edges(tip: tuple[int, int], height: int, width: int) -> dict[str, int]:
    r, c = tip
    return {
        "top": r - 1,
        "bottom": height - 2 - r,
        "left": c - 1,
        "right": width - 2 - c,
    }


def _plan_path_cells(input_cells: tuple[tuple[int, int], ...], direction: tuple[int, int], steps: int) -> set[tuple[int, int]]:
    dr, dc = direction
    path: set[tuple[int, int]] = set()
    for step in range(steps + 1):
        for r, c in input_cells:
            path.add((r + step * dr, c + step * dc))
    return path


def _enumerate_plans(height: int, width: int, border: dict[str, list[int]]) -> list[dict]:
    plans: list[dict] = []
    for side in SIDE_ORDER:
        positions: list[tuple[int, int]] = []
        if side == "top":
            positions = [(1, c) for c in range(1, width - 1)]
        elif side == "bottom":
            positions = [(height - 2, c) for c in range(1, width - 1)]
        elif side == "left":
            positions = [(r, 1) for r in range(1, height - 1)]
        else:
            positions = [(r, width - 2) for r in range(1, height - 1)]

        for direction in SIDE_TO_DIRS[side]:
            dr, dc = direction
            for tip_out in positions:
                if _touch_side(tip_out, (height, width)) != side:
                    continue
                output_cells = _object_cells_from_tip(tip_out, direction)
                if not _all_in_interior(output_cells, (height, width)):
                    continue

                max_steps = 0
                while max_steps < 8:
                    shifted = _shift_cells(output_cells, -(max_steps + 1) * dr, -(max_steps + 1) * dc)
                    if not _all_in_interior(shifted, (height, width)):
                        break
                    max_steps += 1

                for steps in range(1, max_steps + 1):
                    input_cells = _shift_cells(output_cells, -steps * dr, -steps * dc)
                    tip_in = input_cells[0]
                    dists = _distances_to_edges(tip_in, height, width)
                    target_dist = dists[side]
                    misleading = min(dist for name, dist in dists.items() if name != side) < target_dist
                    plans.append(
                        {
                            "side": side,
                            "direction": direction,
                            "axis": _axis_family(direction),
                            "steps": steps,
                            "tip_in": tip_in,
                            "tip_out": tip_out,
                            "input_cells": input_cells,
                            "output_cells": output_cells,
                            "path_cells": _plan_path_cells(input_cells, direction, steps),
                            "landing_color": border[side][tip_out[1] if side in {"top", "bottom"} else tip_out[0]],
                            "misleading": misleading,
                        }
                    )
    return plans


def _separated_from(cells: tuple[tuple[int, int], ...], occupied: set[tuple[int, int]]) -> bool:
    for r, c in cells:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if (nr, nc) in occupied:
                    return False
    return True


def _compatible(plan: dict, occ_in: set[tuple[int, int]], occ_out: set[tuple[int, int]], occ_path: set[tuple[int, int]]) -> bool:
    return (
        _separated_from(plan["input_cells"], occ_in)
        and _separated_from(plan["output_cells"], occ_out)
        and plan["path_cells"].isdisjoint(occ_path)
    )


def _selection_ok(selected: list[dict], count: int, mandatory_side: str) -> bool:
    if len(selected) != count:
        return False
    if not any(plan["misleading"] for plan in selected):
        return False
    if mandatory_side not in {plan["side"] for plan in selected}:
        return False
    if len({plan["direction"] for plan in selected}) < min(2, count):
        return False
    if len({plan["landing_color"] for plan in selected}) < min(2, count):
        return False
    if count >= 2 and len({plan["axis"] for plan in selected}) < 2:
        return False
    if count >= 3 and len({plan["side"] for plan in selected}) < 2:
        return False
    return True


def _plan_score(plan: dict, selected: list[dict], mandatory_side: str) -> float:
    sides = {item["side"] for item in selected}
    dirs = {item["direction"] for item in selected}
    colors = {item["landing_color"] for item in selected}
    axes = {item["axis"] for item in selected}

    score = 0.0
    if not selected and plan["side"] == mandatory_side:
        score += 6.0
    if not any(item["misleading"] for item in selected) and plan["misleading"]:
        score += 5.0
    if plan["side"] not in sides:
        score += 2.0
    if plan["direction"] not in dirs:
        score += 2.0
    if plan["landing_color"] not in colors:
        score += 2.0
    if plan["axis"] not in axes:
        score += 1.0
    score += min(plan["steps"], 4) * 0.5
    if plan["side"] == mandatory_side:
        score += 0.5
    return score + random.random()


def _choose_plans(plans: list[dict], count: int, mandatory_side: str) -> list[dict] | None:
    indexed = list(enumerate(plans))
    side_seed = [pair for pair in indexed if pair[1]["side"] == mandatory_side and pair[1]["misleading"]]
    if not side_seed:
        side_seed = [pair for pair in indexed if pair[1]["side"] == mandatory_side]
    if not side_seed:
        return None

    for _ in range(500):
        selected: list[dict] = []
        used: set[int] = set()
        occ_in: set[tuple[int, int]] = set()
        occ_out: set[tuple[int, int]] = set()
        occ_path: set[tuple[int, int]] = set()

        seed_idx, seed_plan = random.choice(side_seed)
        if not _compatible(seed_plan, occ_in, occ_out, occ_path):
            continue
        selected.append(seed_plan)
        used.add(seed_idx)
        occ_in.update(seed_plan["input_cells"])
        occ_out.update(seed_plan["output_cells"])
        occ_path.update(seed_plan["path_cells"])

        while len(selected) < count:
            candidates = [
                (idx, plan)
                for idx, plan in indexed
                if idx not in used and _compatible(plan, occ_in, occ_out, occ_path)
            ]
            if not candidates:
                break
            candidates.sort(key=lambda pair: _plan_score(pair[1], selected, mandatory_side), reverse=True)
            choice_pool = candidates[: min(12, len(candidates))]
            idx, plan = random.choice(choice_pool)
            selected.append(plan)
            used.add(idx)
            occ_in.update(plan["input_cells"])
            occ_out.update(plan["output_cells"])
            occ_path.update(plan["path_cells"])

        if _selection_ok(selected, count, mandatory_side):
            return selected
    return None


def _paint_input(border_grid: np.ndarray, plans: list[dict]) -> np.ndarray:
    grid = border_grid.copy()
    for plan in plans:
        tip = plan["tip_in"]
        for cell in plan["input_cells"]:
            grid[cell] = YELLOW if cell == tip else GREY
    return grid


def _paint_output(border_grid: np.ndarray, plans: list[dict]) -> np.ndarray:
    grid = border_grid.copy()
    for plan in plans:
        for cell in plan["output_cells"]:
            grid[cell] = plan["landing_color"]
    return grid


def _segment_count(edge_values: list[int]) -> int:
    return 1 + sum(edge_values[i] != edge_values[i - 1] for i in range(1, len(edge_values)))


def _validate_generated(inp: np.ndarray, out: np.ndarray, plans: list[dict], border: dict[str, list[int]]) -> None:
    assert inp.shape == out.shape
    assert inp.ndim == 2 and max(inp.shape) <= 30
    assert np.array_equal(main(inp), out), "solver output must match the constructed oracle output"
    assert len(plans) == int(np.count_nonzero(inp == YELLOW)), "each object must contribute exactly one yellow tip"
    assert not np.array_equal(inp, out), "transform must be non-trivial"
    for side in SIDE_ORDER:
        values = border[side][1:-1]
        assert _segment_count(values) >= 2, "each edge bar must contain multiple visible color segments"
    assert any(plan["misleading"] for plan in plans), "need at least one object close to the wrong edge"
    assert len({plan["direction"] for plan in plans}) >= min(2, len(plans)), "need multiple diagonal directions across objects"
    assert len({plan["landing_color"] for plan in plans}) >= min(2, len(plans)), "need multiple landing colors"


def _generate_attempt(call_index: int) -> np.ndarray | None:
    height = random.randint(12, 20)
    width = random.randint(12, 20)
    border_grid, border = _build_border_grid(height, width)
    plans = _enumerate_plans(height, width, border)
    if len(plans) < 8:
        return None

    mandatory_side = SIDE_ORDER[call_index % len(SIDE_ORDER)]
    base_count = 2 if min(height, width) < 15 else 3
    count = random.randint(base_count, 5)
    chosen = _choose_plans(plans, count, mandatory_side)
    if chosen is None:
        return None

    inp = _paint_input(border_grid, chosen)
    out = _paint_output(border_grid, chosen)
    _validate_generated(inp, out, chosen, border)
    return inp


def generate_input() -> np.ndarray:
    global GEN_COUNTER
    call_index = GEN_COUNTER
    GEN_COUNTER += 1
    for _ in range(400):
        grid = _generate_attempt(call_index)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid polarity-slide puzzle input")
