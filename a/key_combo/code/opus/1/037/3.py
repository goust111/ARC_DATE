from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3

DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _find_components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            components.append(
                {
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "area": len(coords),
                }
            )
    return components


def _component_mask(component: dict) -> np.ndarray:
    r0, c0, r1, c1 = map(int, component["bbox"])
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in component["coords"]:
        mask[r - r0, c - c0] = True
    return mask


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in mask.astype(int))


def _blue_template_mask(grid: np.ndarray) -> np.ndarray:
    components = _find_components_of_color(grid, BLUE)
    assert components, "expected blue sprites"
    masks = [_component_mask(component) for component in components]
    signatures = [_mask_signature(mask) for mask in masks]
    counts = Counter(signatures)
    unique_signatures = [sig for sig, n in counts.items() if n == 1]
    assert len(unique_signatures) == 1, "expected exactly one singleton blue crop"
    unique_mask = np.array(unique_signatures[0], dtype=bool)
    assert any(n == 2 for n in counts.values()), "expected duplicated distractor crops"
    return unique_mask


def _green_squares(grid: np.ndarray) -> list[dict]:
    squares: list[dict] = []
    for component in _find_components_of_color(grid, GREEN):
        r0, c0, r1, c1 = map(int, component["bbox"])
        side_h = r1 - r0 + 1
        side_w = c1 - c0 + 1
        assert side_h == side_w, "green component must be square"
        assert int(component["area"]) == side_h * side_h, "green square must be solid"
        squares.append({"r0": r0, "c0": c0, "side": side_h, "bbox": component["bbox"]})
    return sorted(squares, key=lambda sq: (int(sq["side"]), int(sq["r0"]), int(sq["c0"])))


def _scaled_nonbg_cells(
    template_mask: np.ndarray,
    scale: int,
    top: int,
    left: int,
    shape: tuple[int, int],
) -> tuple[set[tuple[int, int]], bool]:
    h, w = map(int, shape)
    cells: set[tuple[int, int]] = set()
    out_of_bounds = False
    for tr, tc in np.argwhere(template_mask):
        tr = int(tr)
        tc = int(tc)
        base_r = int(top) + tr * int(scale)
        base_c = int(left) + tc * int(scale)
        for dr in range(scale):
            for dc in range(scale):
                rr = base_r + dr
                cc = base_c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    cells.add((rr, cc))
                else:
                    out_of_bounds = True
    return cells, out_of_bounds


def _paint_square_black(canvas: np.ndarray, square: dict) -> None:
    r0 = int(square["r0"])
    c0 = int(square["c0"])
    side = int(square["side"])
    canvas[r0 : r0 + side, c0 : c0 + side] = BG


def _run_process(
    grid: np.ndarray,
    template_mask: np.ndarray,
    squares: list[dict],
) -> tuple[np.ndarray, list[dict]]:
    canvas = np.asarray(grid, dtype=int).copy()
    trace: list[dict] = []
    for square in squares:
        cells, out_of_bounds = _scaled_nonbg_cells(
            template_mask,
            int(square["side"]),
            int(square["r0"]),
            int(square["c0"]),
            canvas.shape,
        )
        hit_colors = {int(canvas[r, c]) for r, c in cells if int(canvas[r, c]) != BG}
        if out_of_bounds:
            reason = "out_of_bounds"
        elif not hit_colors:
            reason = "success"
        elif hit_colors == {BLUE}:
            reason = "blue_overlap"
        elif hit_colors == {GREEN}:
            reason = "green_overlap"
        elif hit_colors == {RED}:
            reason = "red_overlap"
        else:
            reason = "mixed_overlap"

        applied = (reason == "success")
        if applied:
            _paint_square_black(canvas, square)
            for r, c in cells:
                canvas[r, c] = RED

        trace.append(
            {
                "square": {"r0": int(square["r0"]), "c0": int(square["c0"]), "side": int(square["side"])},
                "cells": cells,
                "out_of_bounds": out_of_bounds,
                "hit_colors": hit_colors,
                "reason": reason,
                "applied": applied,
            }
        )
    return canvas, trace


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the singleton blue sprite crop as a scaled stamp for the green squares, but only when the stamp fits into empty space.

    Input structure:
    - A black background contains several separated blue 4-connected sprites and two or three detached solid green squares.
    - The blue sprites come in duplicate bounding-box crops except for one unique crop, which is the outlier template.
    - Each green object is a solid k x k square whose top-left corner is the placement origin for a scaled copy of that template.

    Rule:
    - Find every blue connected component, crop each one to its bounding box, and select the only crop pattern that appears once.
    - Sort the green squares by side length, then by top-left row and column.
    - For each square, scale every non-background template pixel to a k x k block aligned to the square's top-left corner; if any stamped cell would leave the grid or hit any current non-black cell, skip that square, otherwise paint the stamp red and erase the green square.

    Key insight:
    - The unique blue crop is not the output itself; it becomes a reusable stamp whose success depends on scale, placement, and the canvas state created by earlier smaller squares.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    template_mask = _blue_template_mask(g)
    squares = _green_squares(g)
    out, _trace = _run_process(g, template_mask, squares)
    return out


def _connected_mask(mask: np.ndarray) -> bool:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    if not coords:
        return False
    seen = {coords[0]}
    stack = [coords[0]]
    cells = set(coords)
    while stack:
        r, c = stack.pop()
        for dr, dc in DIRS4:
            nb = (r + dr, c + dc)
            if nb in cells and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == len(cells)


def _build_mask_catalog() -> dict[int, list[np.ndarray]]:
    catalog: dict[int, list[np.ndarray]] = {5: [], 6: []}
    for bits in range(1 << 9):
        mask = np.zeros((3, 3), dtype=bool)
        area = 0
        for idx in range(9):
            if (bits >> idx) & 1:
                mask[idx // 3, idx % 3] = True
                area += 1
        if area not in catalog:
            continue
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        if len(rows) == 0 or len(cols) == 0:
            continue
        if (int(rows[0]), int(cols[0]), int(rows[-1]), int(cols[-1])) != (0, 0, 2, 2):
            continue
        if _connected_mask(mask):
            catalog[area].append(mask.copy())
    return catalog


MASK_CATALOG = _build_mask_catalog()

GRID_HEIGHT_RANGE = (20, 24)
GRID_WIDTH_RANGE = (20, 24)
BLUE_SPRITE_COUNT_CHOICES = (5, 7)
GREEN_COUNT_WEIGHTS = ((2, 0.72), (3, 0.28))
FAILURE_MODE_WEIGHTS = (("red_overlap", 0.45), ("blue_overlap", 0.3), ("out_of_bounds", 0.25))
THIRD_GOAL_WEIGHTS = (("success", 0.55), ("blue_overlap", 0.25), ("out_of_bounds", 0.2))
GEN_MAX_ATTEMPTS = 400


def _weighted_choice(weighted_values: tuple[tuple[object, float], ...]):
    total = sum(weight for _value, weight in weighted_values)
    pick = random.random() * total
    running = 0.0
    for value, weight in weighted_values:
        running += weight
        if pick <= running:
            return value
    return weighted_values[-1][0]


def _rect_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    return a[2] + gap < b[0] or b[2] + gap < a[0] or a[3] + gap < b[1] or b[3] + gap < a[1]


def _coarse_sprite_slots(height: int, width: int) -> list[tuple[int, int]]:
    row_offset = random.randint(0, 1)
    col_offset = random.randint(0, 1)
    rows = list(range(row_offset, height - 2, 4))
    cols = list(range(col_offset, width - 2, 4))
    return [(r, c) for r in rows for c in cols]


def _choose_blue_masks() -> tuple[np.ndarray, list[np.ndarray]]:
    area = random.choice((5, 6))
    outlier_pool = [mask for mask in MASK_CATALOG[area] if not bool(mask[0, 0])]
    outlier = random.choice(outlier_pool).copy()
    pair_count = 2 if random.random() < 0.55 else 3

    same_corner_pool = [
        mask
        for mask in MASK_CATALOG[area]
        if not bool(mask[0, 0]) and not np.array_equal(mask, outlier)
    ]
    other_pool = [mask for mask in MASK_CATALOG[area] if not np.array_equal(mask, outlier)]

    chosen_pairs: list[np.ndarray] = []
    if same_corner_pool:
        chosen_pairs.append(random.choice(same_corner_pool).copy())
    while len(chosen_pairs) < pair_count:
        candidate = random.choice(other_pool).copy()
        if any(np.array_equal(candidate, existing) for existing in chosen_pairs):
            continue
        chosen_pairs.append(candidate)

    masks = [outlier.copy()]
    for pair_mask in chosen_pairs:
        masks.extend([pair_mask.copy(), pair_mask.copy()])
    random.shuffle(masks)
    return outlier, masks


def _place_blue_sprites(height: int, width: int, masks: list[np.ndarray]) -> tuple[np.ndarray, list[tuple[int, int, int, int]]] | None:
    slots = _coarse_sprite_slots(height, width)
    if len(slots) < len(masks):
        return None
    chosen_slots = random.sample(slots, len(masks))
    grid = np.zeros((height, width), dtype=int)
    blue_bboxes: list[tuple[int, int, int, int]] = []
    for mask, (r0, c0) in zip(masks, chosen_slots):
        grid[r0 : r0 + 3, c0 : c0 + 3][mask] = BLUE
        blue_bboxes.append((r0, c0, r0 + 2, c0 + 2))
    return grid, blue_bboxes


def _square_bbox(top: int, left: int, side: int) -> tuple[int, int, int, int]:
    return (top, left, top + side - 1, left + side - 1)


def _square_cells(square: dict) -> set[tuple[int, int]]:
    top = int(square["r0"])
    left = int(square["c0"])
    side = int(square["side"])
    return {(r, c) for r in range(top, top + side) for c in range(left, left + side)}


def _candidate_square_positions(
    height: int,
    width: int,
    side: int,
    occupied_rects: list[tuple[int, int, int, int]],
    forbidden_cells: set[tuple[int, int]],
) -> list[dict]:
    candidates: list[dict] = []
    for r0 in range(height - side + 1):
        for c0 in range(width - side + 1):
            bbox = _square_bbox(r0, c0, side)
            if any(not _rect_separated(bbox, other, gap=1) for other in occupied_rects):
                continue
            cells = {(r, c) for r in range(r0, r0 + side) for c in range(c0, c0 + side)}
            if cells & forbidden_cells:
                continue
            candidates.append({"r0": r0, "c0": c0, "side": side, "bbox": bbox})
    random.shuffle(candidates)
    return candidates


def _eval_placement(canvas: np.ndarray, template_mask: np.ndarray, square: dict) -> dict:
    cells, out_of_bounds = _scaled_nonbg_cells(
        template_mask,
        int(square["side"]),
        int(square["r0"]),
        int(square["c0"]),
        canvas.shape,
    )
    hit_colors = {int(canvas[r, c]) for r, c in cells if int(canvas[r, c]) != BG}
    if out_of_bounds:
        clean_reason = "out_of_bounds" if not hit_colors else "out_of_bounds_mixed"
    elif not hit_colors:
        clean_reason = "success"
    elif hit_colors == {BLUE}:
        clean_reason = "blue_overlap"
    elif hit_colors == {GREEN}:
        clean_reason = "green_overlap"
    elif hit_colors == {RED}:
        clean_reason = "red_overlap"
    else:
        clean_reason = "mixed_overlap"
    return {"cells": cells, "out_of_bounds": out_of_bounds, "hit_colors": hit_colors, "reason": clean_reason}


def _apply_success_on_canvas(canvas: np.ndarray, square: dict, cells: set[tuple[int, int]]) -> np.ndarray:
    updated = canvas.copy()
    _paint_square_black(updated, square)
    for r, c in cells:
        updated[r, c] = RED
    return updated


def _choose_square_count() -> int:
    return int(_weighted_choice(GREEN_COUNT_WEIGHTS))


def _square_sizes(square_count: int) -> list[int]:
    return [2, 3] if square_count == 2 else [1, 2, 3]


def _choose_first_success(
    blue_grid: np.ndarray,
    blue_bboxes: list[tuple[int, int, int, int]],
    template_mask: np.ndarray,
    first_side: int,
    target_failure: str,
) -> tuple[dict, dict] | None:
    candidates = []
    for square in _candidate_square_positions(blue_grid.shape[0], blue_grid.shape[1], first_side, blue_bboxes, set()):
        info = _eval_placement(blue_grid, template_mask, square)
        if info["reason"] == "success":
            span_r = min(int(square["r0"]), int(blue_grid.shape[0]) - (int(square["r0"]) + 3 * first_side))
            span_c = min(int(square["c0"]), int(blue_grid.shape[1]) - (int(square["c0"]) + 3 * first_side))
            candidates.append((span_r + span_c, square, info))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=(target_failure == "red_overlap"))
    best_band = candidates[: max(1, len(candidates) // 2)] if target_failure == "red_overlap" else candidates
    _score, square, info = random.choice(best_band)
    return square, info


def _choose_clean_candidate(
    canvas: np.ndarray,
    blue_grid: np.ndarray,
    template_mask: np.ndarray,
    side: int,
    occupied_rects: list[tuple[int, int, int, int]],
    forbidden_cells: set[tuple[int, int]],
    target_reason: str,
    require_pre_red_success: bool = False,
) -> tuple[dict, dict] | None:
    candidates = _candidate_square_positions(canvas.shape[0], canvas.shape[1], side, occupied_rects, forbidden_cells)
    filtered: list[tuple[dict, dict]] = []
    for square in candidates:
        info = _eval_placement(canvas, template_mask, square)
        if info["reason"] != target_reason:
            continue
        if require_pre_red_success:
            pre_info = _eval_placement(blue_grid, template_mask, square)
            if pre_info["reason"] != "success":
                continue
        filtered.append((square, info))
    if not filtered:
        return None
    return random.choice(filtered)


def _build_square_plan(
    blue_grid: np.ndarray,
    blue_bboxes: list[tuple[int, int, int, int]],
    template_mask: np.ndarray,
    square_count: int,
    target_failure: str,
) -> list[dict] | None:
    sizes = _square_sizes(square_count)
    first_choice = _choose_first_success(blue_grid, blue_bboxes, template_mask, sizes[0], target_failure)
    if first_choice is None:
        return None
    first_square, first_info = first_choice
    plan = [{"square": first_square, "goal": "success"}]
    occupied_rects = list(blue_bboxes) + [first_square["bbox"]]
    forbidden_cells = set(first_info["cells"])
    canvas_after_first = _apply_success_on_canvas(blue_grid, first_square, first_info["cells"])

    second_choice = _choose_clean_candidate(
        canvas_after_first,
        blue_grid,
        template_mask,
        sizes[1],
        occupied_rects,
        forbidden_cells,
        target_failure,
        require_pre_red_success=(target_failure == "red_overlap"),
    )
    if second_choice is None:
        return None
    second_square, second_info = second_choice
    plan.append({"square": second_square, "goal": target_failure})
    occupied_rects.append(second_square["bbox"])
    forbidden_cells |= set(second_info["cells"])

    if square_count == 3:
        third_goal = str(_weighted_choice(THIRD_GOAL_WEIGHTS))
        third_choice = _choose_clean_candidate(
            canvas_after_first,
            blue_grid,
            template_mask,
            sizes[2],
            occupied_rects,
            forbidden_cells,
            third_goal,
            require_pre_red_success=False,
        )
        if third_choice is None:
            return None
        third_square, _third_info = third_choice
        plan.append({"square": third_square, "goal": third_goal})

    return plan


def _render_input(
    blue_grid: np.ndarray,
    square_plan: list[dict],
) -> np.ndarray:
    grid = blue_grid.copy()
    for item in square_plan:
        square = item["square"]
        r0 = int(square["r0"])
        c0 = int(square["c0"])
        side = int(square["side"])
        grid[r0 : r0 + side, c0 : c0 + side] = GREEN
    return grid


def _validate_generated_case(
    grid: np.ndarray,
    intended_template: np.ndarray,
    square_plan: list[dict],
    target_failure: str,
) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    if max(grid.shape) > 30:
        return False
    if not set(map(int, np.unique(grid))) <= {BG, BLUE, GREEN}:
        return False

    blue_components = _find_components_of_color(grid, BLUE)
    if len(blue_components) not in BLUE_SPRITE_COUNT_CHOICES:
        return False
    blue_signatures = [_mask_signature(_component_mask(component)) for component in blue_components]
    counts = Counter(blue_signatures)
    if sorted(counts.values()) != [1] + [2] * ((len(blue_components) - 1) // 2):
        return False
    singleton_sigs = [sig for sig, n in counts.items() if n == 1]
    if len(singleton_sigs) != 1:
        return False
    if singleton_sigs[0] != _mask_signature(intended_template):
        return False

    squares = _green_squares(grid)
    planned_squares = sorted(
        [{"r0": int(item["square"]["r0"]), "c0": int(item["square"]["c0"]), "side": int(item["square"]["side"])} for item in square_plan],
        key=lambda sq: (sq["side"], sq["r0"], sq["c0"]),
    )
    detected_squares = [{"r0": int(sq["r0"]), "c0": int(sq["c0"]), "side": int(sq["side"])} for sq in squares]
    if planned_squares != detected_squares:
        return False

    template_mask = _blue_template_mask(grid)
    if _mask_signature(template_mask) != _mask_signature(intended_template):
        return False
    out, trace = _run_process(grid, template_mask, squares)
    if not np.array_equal(out, main(grid)):
        return False

    reasons = [entry["reason"] for entry in trace]
    if "success" not in reasons:
        return False
    if not any(reason != "success" for reason in reasons):
        return False
    if target_failure not in reasons:
        return False
    for entry in trace:
        if entry["reason"] in {"green_overlap", "mixed_overlap"}:
            return False
        if entry["reason"] == "out_of_bounds" and entry["hit_colors"]:
            return False

    if target_failure == "red_overlap":
        red_idx = reasons.index("red_overlap")
        if red_idx == 0:
            return False
        blue_only = grid.copy()
        blue_only[blue_only == GREEN] = BG
        red_square = squares[red_idx]
        red_pre = _eval_placement(blue_only, template_mask, red_square)
        if red_pre["reason"] != "success":
            return False

    applied = [entry for entry in trace if entry["applied"]]
    if not applied:
        return False
    if any(int(component["area"]) not in (5, 6) for component in blue_components):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)

    outlier_mask, blue_masks = _choose_blue_masks()
    placed = _place_blue_sprites(height, width, blue_masks)
    if placed is None:
        return None
    blue_grid, blue_bboxes = placed

    square_count = _choose_square_count()
    target_failure = str(_weighted_choice(FAILURE_MODE_WEIGHTS))
    square_plan = _build_square_plan(blue_grid, blue_bboxes, outlier_mask, square_count, target_failure)
    if square_plan is None:
        return None

    grid = _render_input(blue_grid, square_plan)
    if not _validate_generated_case(grid, outlier_mask, square_plan, target_failure):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
