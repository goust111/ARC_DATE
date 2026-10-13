from __future__ import annotations

import random
from collections import deque

import numpy as np


YELLOW = 4
GREY = 5
OBJ_COLORS = (1, 2, 3, 6)
ALL_OBJECT_COLORS = set(OBJ_COLORS)

SHAPES: tuple[tuple[tuple[int, int], ...], ...] = (
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0)),
    ((0, 0), (1, 0), (1, 1), (2, 1), (2, 2)),
    ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)),
)

SHAPES_BY_AREA: dict[int, list[tuple[tuple[int, int], ...]]] = {}
for _shape in SHAPES:
    SHAPES_BY_AREA.setdefault(len(_shape), []).append(_shape)


def _checkerboard(height: int, width: int) -> np.ndarray:
    rows, cols = np.indices((height, width))
    return np.where((rows + cols) % 2 == 0, YELLOW, GREY).astype(int)


def _shape_bbox(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    max_r = max(r for r, _ in shape)
    max_c = max(c for _, c in shape)
    return max_r + 1, max_c + 1


def _translate(
    shape: tuple[tuple[int, int], ...], anchor: tuple[int, int]
) -> tuple[tuple[int, int], ...]:
    ar, ac = anchor
    return tuple((ar + r, ac + c) for r, c in shape)


def _bbox_of_cells(cells: tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _touching4(
    cells_a: tuple[tuple[int, int], ...], cells_b: tuple[tuple[int, int], ...]
) -> bool:
    bset = set(cells_b)
    for r, c in cells_a:
        if (r - 1, c) in bset or (r + 1, c) in bset or (r, c - 1) in bset or (r, c + 1) in bset:
            return True
    return False


def _row_overlap(
    bbox_a: tuple[int, int, int, int], bbox_b: tuple[int, int, int, int]
) -> bool:
    return not (bbox_a[2] < bbox_b[0] or bbox_b[2] < bbox_a[0])


def _col_overlap(
    bbox_a: tuple[int, int, int, int], bbox_b: tuple[int, int, int, int]
) -> bool:
    return not (bbox_a[3] < bbox_b[1] or bbox_b[3] < bbox_a[1])


def _min_manhattan_between(
    cells_a: tuple[tuple[int, int], ...], cells_b: tuple[tuple[int, int], ...]
) -> int:
    best = 10**9
    for r1, c1 in cells_a:
        for r2, c2 in cells_b:
            best = min(best, abs(r1 - r2) + abs(c1 - c2))
    return best


def _far_enough_input(
    new_cells: tuple[tuple[int, int], ...],
    existing: list[dict],
) -> bool:
    for plan in existing:
        if _min_manhattan_between(new_cells, plan["input_cells"]) < 3:
            return False
    return True


def _far_enough_final(
    new_cells: tuple[tuple[int, int], ...],
    existing: list[dict],
) -> bool:
    for plan in existing:
        if _min_manhattan_between(new_cells, plan["final_cells"]) < 3:
            return False
    return True


def _parity_shift_candidates(
    direction: str, final_anchor: tuple[int, int]
) -> list[int]:
    r, c = final_anchor
    if direction == "right":
        max_shift = c
        target_parity = (r + c) % 2
        return [d for d in range(1, max_shift + 1) if d % 2 == target_parity]
    max_shift = r
    target_parity = (r + c - 1) % 2
    return [d for d in range(1, max_shift + 1) if d % 2 == target_parity]


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _extract_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    fg_mask = (g != YELLOW) & (g != GREY)
    seen = np.zeros_like(fg_mask, dtype=bool)
    objects: list[dict] = []

    for r in range(height):
        for c in range(width):
            if not fg_mask[r, c] or seen[r, c]:
                continue
            color = int(g[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                assert int(g[rr, cc]) == color
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < height and 0 <= nc < width and fg_mask[nr, nc] and not seen[nr, nc]:
                        assert int(g[nr, nc]) == color
                        seen[nr, nc] = True
                        q.append((nr, nc))

            cells_t = tuple(sorted(cells))
            bbox = _bbox_of_cells(cells_t)
            anchor = (bbox[0], bbox[1])
            assert anchor in set(cells_t)
            objects.append(
                {
                    "color": color,
                    "cells": cells_t,
                    "bbox": bbox,
                    "anchor": anchor,
                    "area": len(cells_t),
                }
            )

    return objects


def _direction_for(anchor: tuple[int, int]) -> str:
    return "right" if (anchor[0] + anchor[1]) % 2 == 0 else "down"


def _simulate_objects(grid: np.ndarray) -> list[dict]:
    height, width = grid.shape
    objects = _extract_objects(grid)
    for idx, obj in enumerate(objects):
        obj["id"] = idx
        obj["direction"] = _direction_for(obj["anchor"])

    right = sorted(
        [obj for obj in objects if obj["direction"] == "right"],
        key=lambda obj: (-obj["anchor"][1], obj["anchor"][0], obj["anchor"][1], obj["id"]),
    )
    down = sorted(
        [obj for obj in objects if obj["direction"] == "down"],
        key=lambda obj: (-obj["anchor"][0], obj["anchor"][1], obj["anchor"][0], obj["id"]),
    )

    stopped: list[dict] = []
    occupied = np.zeros((height, width), dtype=bool)

    for group, dr, dc in ((right, 0, 1), (down, 1, 0)):
        for obj in group:
            cells = np.array(obj["cells"], dtype=int)
            moved = 0
            stop_reason = "boundary"
            while True:
                nxt = cells + np.array([dr, dc], dtype=int)
                if np.any(nxt[:, 0] < 0) or np.any(nxt[:, 0] >= height) or np.any(nxt[:, 1] < 0) or np.any(nxt[:, 1] >= width):
                    stop_reason = "boundary"
                    break
                if occupied[nxt[:, 0], nxt[:, 1]].any():
                    stop_reason = "blocked"
                    break
                cells = nxt
                moved += 1

            final_cells = tuple((int(r), int(c)) for r, c in cells.tolist())
            final_bbox = _bbox_of_cells(final_cells)
            placed = dict(obj)
            placed["final_cells"] = final_cells
            placed["final_cell_set"] = set(final_cells)
            placed["final_bbox"] = final_bbox
            placed["final_anchor"] = (final_bbox[0], final_bbox[1])
            placed["moved"] = moved
            placed["stop_reason"] = stop_reason
            stopped.append(placed)
            occupied[cells[:, 0], cells[:, 1]] = True

    return stopped


def _losers_from_final_state(final_objects: list[dict]) -> tuple[set[int], list[dict]]:
    losers: set[int] = set()
    pairs: list[dict] = []
    for i in range(len(final_objects)):
        for j in range(i + 1, len(final_objects)):
            a = final_objects[i]
            b = final_objects[j]
            if not _touching4(a["final_cells"], b["final_cells"]):
                continue
            if a["area"] < b["area"]:
                loser = a
                winner = b
                tie = False
            elif a["area"] > b["area"]:
                loser = b
                winner = a
                tie = False
            else:
                tie = True
                key_a = (a["final_bbox"][0], a["final_bbox"][1], a["id"])
                key_b = (b["final_bbox"][0], b["final_bbox"][1], b["id"])
                if key_a <= key_b:
                    winner = a
                    loser = b
                else:
                    winner = b
                    loser = a
            losers.add(loser["id"])
            pairs.append(
                {
                    "a_id": a["id"],
                    "b_id": b["id"],
                    "a_area": a["area"],
                    "b_area": b["area"],
                    "a_top": a["final_bbox"][0],
                    "b_top": b["final_bbox"][0],
                    "winner": winner["id"],
                    "loser": loser["id"],
                    "tie": tie,
                }
            )
    return losers, pairs


def _render_output_from_state(grid: np.ndarray, final_objects: list[dict], losers: set[int]) -> np.ndarray:
    out = _checkerboard(*grid.shape)
    for obj in final_objects:
        if obj["id"] in losers:
            continue
        for r, c in obj["final_cells"]:
            out[r, c] = int(obj["color"])
    return out


def _oracle_transform(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    objects = _extract_objects(g)
    placed: list[dict] = []
    stopped_cells: set[tuple[int, int]] = set()

    for idx, obj in enumerate(objects):
        obj = dict(obj)
        obj["id"] = idx
        obj["direction"] = _direction_for(obj["anchor"])
        placed.append(obj)

    right = sorted(
        [obj for obj in placed if obj["direction"] == "right"],
        key=lambda obj: (-obj["anchor"][1], obj["anchor"][0], obj["id"]),
    )
    down = sorted(
        [obj for obj in placed if obj["direction"] == "down"],
        key=lambda obj: (-obj["anchor"][0], obj["anchor"][1], obj["id"]),
    )

    resolved: list[dict] = []
    for group in (right, down):
        for obj in group:
            dr, dc = (0, 1) if obj["direction"] == "right" else (1, 0)
            current = set(obj["cells"])
            moved = 0
            while True:
                nxt = {(r + dr, c + dc) for r, c in current}
                if any(r < 0 or r >= height or c < 0 or c >= width for r, c in nxt):
                    stop_reason = "boundary"
                    break
                if nxt & stopped_cells:
                    stop_reason = "blocked"
                    break
                current = nxt
                moved += 1
            final_cells = tuple(sorted(current))
            final_bbox = _bbox_of_cells(final_cells)
            resolved_obj = dict(obj)
            resolved_obj["final_cells"] = final_cells
            resolved_obj["final_bbox"] = final_bbox
            resolved_obj["moved"] = moved
            resolved_obj["stop_reason"] = stop_reason
            resolved.append(resolved_obj)
            stopped_cells |= current

    losers, pairs = _losers_from_final_state(resolved)
    out = _render_output_from_state(g, resolved, losers)
    info = {
        "objects": resolved,
        "collision_pairs": pairs,
        "losers": losers,
        "right_count": sum(obj["direction"] == "right" for obj in resolved),
        "down_count": sum(obj["direction"] == "down" for obj in resolved),
        "all_moved": all(obj["moved"] > 0 for obj in resolved),
        "has_block_stop": any(obj["stop_reason"] == "blocked" for obj in resolved),
        "has_tie_collision": any(pair["tie"] for pair in pairs),
        "has_size_collision": any(not pair["tie"] for pair in pairs),
        "ambiguous_tie": any(
            pair["tie"] and pair["a_top"] == pair["b_top"] and pair["a_area"] == pair["b_area"]
            for pair in pairs
        ),
    }
    return out, info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode each object's anchor parity into a slide direction, then remove the weaker object from every final side-contact.

    Input structure:
    - The grid is a yellow/grey checkerboard anchored at the top-left corner: cells with even `(r+c)` are yellow and odd `(r+c)` are grey.
    - Several small monochrome 4-connected objects sit on top of that checkerboard, with each object's bbox top-left cell occupied by the object itself.
    - Object colors are foreground colors distinct from the two checkerboard colors.

    Rule:
    - For each object, inspect the checkerboard phase under its anchor pixel: an even-parity anchor slides the whole object right, and an odd-parity anchor slides it down.
    - Process all right-sliding objects first from right to left, then all down-sliding objects from bottom to top; each object advances one step at a time until its next step would leave the grid or overlap an already stopped object.
    - After all objects stop, compare every pair of objects that is 4-neighbor adjacent: erase the smaller object back to checkerboard background, and if sizes tie keep the one whose final top row is higher.

    Key insight:
    - The checkerboard phase is not decorative background: it is the hidden code that assigns each object's motion direction before the later sliding and collision-elimination stages.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    final_objects = _simulate_objects(g)
    losers, _pairs = _losers_from_final_state(final_objects)
    return _render_output_from_state(g, final_objects, losers)


def _make_plan(
    shape: tuple[tuple[int, int], ...],
    color: int,
    direction: str,
    input_anchor: tuple[int, int],
    final_anchor: tuple[int, int],
) -> dict:
    input_cells = _translate(shape, input_anchor)
    final_cells = _translate(shape, final_anchor)
    return {
        "shape": shape,
        "color": color,
        "direction": direction,
        "input_anchor": input_anchor,
        "final_anchor": final_anchor,
        "input_cells": input_cells,
        "final_cells": final_cells,
        "input_bbox": _bbox_of_cells(input_cells),
        "final_bbox": _bbox_of_cells(final_cells),
        "area": len(shape),
    }


def _right_pair_plan(height: int, width: int, tie_mode: bool) -> list[dict] | None:
    if tie_mode:
        area = random.randint(2, 6)
        shape_a = random.choice(SHAPES_BY_AREA[area])
        shape_b = random.choice(SHAPES_BY_AREA[area])
    else:
        area_a, area_b = random.sample(range(2, 7), 2)
        shape_a = random.choice(SHAPES_BY_AREA[area_a])
        shape_b = random.choice(SHAPES_BY_AREA[area_b])

    ha, wa = _shape_bbox(shape_a)
    hb, wb = _shape_bbox(shape_b)
    c_a = width - wa
    if c_a <= 1:
        return None

    color_a = random.choice(OBJ_COLORS)
    color_b = random.choice(OBJ_COLORS)

    candidates: list[list[dict]] = []
    row_a_order = list(range(0, height - ha + 1))
    row_b_order = list(range(0, height - hb + 1))
    col_b_order = list(range(0, c_a))
    random.shuffle(row_a_order)
    random.shuffle(row_b_order)
    random.shuffle(col_b_order)

    for r_a in row_a_order:
        a_final = _translate(shape_a, (r_a, c_a))
        a_set = set(a_final)
        for r_b in row_b_order:
            if tie_mode and r_b == r_a:
                continue
            for c_b in col_b_order:
                b_final = _translate(shape_b, (r_b, c_b))
                b_shift = {(r, c + 1) for r, c in b_final}
                if any(c >= width for _, c in b_shift):
                    continue
                if set(b_final) & a_set:
                    continue
                if not (b_shift & a_set):
                    continue
                if not _touching4(a_final, b_final):
                    continue

                da = _parity_shift_candidates("right", (r_a, c_a))
                db = _parity_shift_candidates("right", (r_b, c_b))
                random.shuffle(da)
                random.shuffle(db)
                for shift_a in da:
                    for shift_b in db:
                        plan_a = _make_plan(
                            shape_a,
                            color_a,
                            "right",
                            (r_a, c_a - shift_a),
                            (r_a, c_a),
                        )
                        plan_b = _make_plan(
                            shape_b,
                            color_b,
                            "right",
                            (r_b, c_b - shift_b),
                            (r_b, c_b),
                        )
                        if not _far_enough_input(plan_b["input_cells"], [plan_a]):
                            continue
                        candidates.append([plan_a, plan_b])
                        break
                    if candidates:
                        break
                if candidates:
                    break
            if candidates:
                break
        if candidates:
            break

    if not candidates:
        return None
    return random.choice(candidates)


def _down_pair_plan(height: int, width: int, tie_mode: bool) -> list[dict] | None:
    if tie_mode:
        area = random.randint(2, 6)
        shape_a = random.choice(SHAPES_BY_AREA[area])
        shape_b = random.choice(SHAPES_BY_AREA[area])
    else:
        area_a, area_b = random.sample(range(2, 7), 2)
        shape_a = random.choice(SHAPES_BY_AREA[area_a])
        shape_b = random.choice(SHAPES_BY_AREA[area_b])

    ha, wa = _shape_bbox(shape_a)
    hb, wb = _shape_bbox(shape_b)
    r_a = height - ha
    if r_a <= 1:
        return None

    color_a = random.choice(OBJ_COLORS)
    color_b = random.choice(OBJ_COLORS)

    candidates: list[list[dict]] = []
    col_a_order = list(range(0, width - wa + 1))
    col_b_order = list(range(0, width - wb + 1))
    row_b_order = list(range(0, r_a))
    random.shuffle(col_a_order)
    random.shuffle(col_b_order)
    random.shuffle(row_b_order)

    for c_a in col_a_order:
        a_final = _translate(shape_a, (r_a, c_a))
        a_set = set(a_final)
        for c_b in col_b_order:
            for r_b in row_b_order:
                b_final = _translate(shape_b, (r_b, c_b))
                b_shift = {(r + 1, c) for r, c in b_final}
                if any(r >= height for r, _ in b_shift):
                    continue
                if set(b_final) & a_set:
                    continue
                if not (b_shift & a_set):
                    continue
                if not _touching4(a_final, b_final):
                    continue

                da = _parity_shift_candidates("down", (r_a, c_a))
                db = _parity_shift_candidates("down", (r_b, c_b))
                random.shuffle(da)
                random.shuffle(db)
                for shift_a in da:
                    for shift_b in db:
                        plan_a = _make_plan(
                            shape_a,
                            color_a,
                            "down",
                            (r_a - shift_a, c_a),
                            (r_a, c_a),
                        )
                        plan_b = _make_plan(
                            shape_b,
                            color_b,
                            "down",
                            (r_b - shift_b, c_b),
                            (r_b, c_b),
                        )
                        if not _far_enough_input(plan_b["input_cells"], [plan_a]):
                            continue
                        candidates.append([plan_a, plan_b])
                        break
                    if candidates:
                        break
                if candidates:
                    break
            if candidates:
                break
        if candidates:
            break

    if not candidates:
        return None
    return random.choice(candidates)


def _plan_boundary_object(
    height: int,
    width: int,
    direction: str,
    existing: list[dict],
) -> dict | None:
    shapes = list(SHAPES)
    random.shuffle(shapes)
    for shape in shapes:
        h, w = _shape_bbox(shape)
        color = random.choice(OBJ_COLORS)
        if direction == "right":
            c_final = width - w
            row_order = list(range(0, height - h + 1))
            random.shuffle(row_order)
            for r_final in row_order:
                final_cells = _translate(shape, (r_final, c_final))
                final_bbox = _bbox_of_cells(final_cells)
                if not _far_enough_final(final_cells, existing):
                    continue
                if any(_row_overlap(final_bbox, plan["final_bbox"]) for plan in existing):
                    continue
                shifts = _parity_shift_candidates("right", (r_final, c_final))
                random.shuffle(shifts)
                for shift in shifts:
                    plan = _make_plan(
                        shape,
                        color,
                        "right",
                        (r_final, c_final - shift),
                        (r_final, c_final),
                    )
                    if _far_enough_input(plan["input_cells"], existing):
                        return plan
        else:
            r_final = height - h
            col_order = list(range(0, width - w + 1))
            random.shuffle(col_order)
            for c_final in col_order:
                final_cells = _translate(shape, (r_final, c_final))
                final_bbox = _bbox_of_cells(final_cells)
                if not _far_enough_final(final_cells, existing):
                    continue
                if any(_col_overlap(final_bbox, plan["final_bbox"]) for plan in existing):
                    continue
                shifts = _parity_shift_candidates("down", (r_final, c_final))
                random.shuffle(shifts)
                for shift in shifts:
                    plan = _make_plan(
                        shape,
                        color,
                        "down",
                        (r_final - shift, c_final),
                        (r_final, c_final),
                    )
                    if _far_enough_input(plan["input_cells"], existing):
                        return plan
    return None


def _render_input(height: int, width: int, plans: list[dict]) -> np.ndarray:
    grid = _checkerboard(height, width)
    for plan in plans:
        for r, c in plan["input_cells"]:
            grid[r, c] = int(plan["color"])
    return grid


def _scene_is_valid(grid: np.ndarray, scenario: str, tie_mode: bool) -> bool:
    objects = _extract_objects(grid)
    if not (3 <= len(objects) <= 5):
        return False
    if any(obj["area"] < 2 or obj["area"] > 6 for obj in objects):
        return False
    if any((obj["color"] not in ALL_OBJECT_COLORS) for obj in objects):
        return False

    expected, info = _oracle_transform(grid)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False
    if info["right_count"] == 0 or info["down_count"] == 0:
        return False
    if not info["all_moved"]:
        return False
    if info["ambiguous_tie"]:
        return False

    if scenario == "none":
        if info["collision_pairs"]:
            return False
    else:
        if not info["collision_pairs"]:
            return False
        if not info["has_block_stop"]:
            return False
        if tie_mode and not info["has_tie_collision"]:
            return False
        if not tie_mode and not info["has_size_collision"]:
            return False

    return True


def _build_scene() -> np.ndarray | None:
    height = random.randint(13, 16)
    width = random.randint(13, 16)
    scenario = random.choices(
        ["right_pair", "down_pair", "none"], weights=[4, 4, 2], k=1
    )[0]
    tie_mode = scenario != "none" and random.random() < 0.35
    target_count = random.randint(3, 5)

    plans: list[dict] = []
    if scenario == "right_pair":
        pair = _right_pair_plan(height, width, tie_mode)
        if pair is None:
            return None
        plans.extend(pair)
    elif scenario == "down_pair":
        pair = _down_pair_plan(height, width, tie_mode)
        if pair is None:
            return None
        plans.extend(pair)

    needed_dirs: list[str] = []
    has_right = any(plan["direction"] == "right" for plan in plans)
    has_down = any(plan["direction"] == "down" for plan in plans)
    if not has_right:
        needed_dirs.append("right")
    if not has_down:
        needed_dirs.append("down")

    while len(plans) < target_count:
        if needed_dirs:
            direction = needed_dirs.pop(0)
        else:
            direction = random.choice(("right", "down"))
        filler = _plan_boundary_object(height, width, direction, plans)
        if filler is None:
            return None
        plans.append(filler)

    grid = _render_input(height, width, plans)
    if not _scene_is_valid(grid, scenario, tie_mode):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _build_scene()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid checkerboard-slide scene")
