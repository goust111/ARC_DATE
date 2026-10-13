from __future__ import annotations

import random

import numpy as np


BG = 0
RED = 2
GREY = 5
OBJECT_COLORS = [1, 3, 4, 6, 7, 8, 9]

ARROW_PATTERNS = {
    "right": np.array([[0, 0, 1], [1, 1, 1], [0, 0, 1]], dtype=int),
    "left": np.array([[1, 0, 0], [1, 1, 1], [1, 0, 0]], dtype=int),
    "up": np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=int),
    "down": np.array([[0, 1, 0], [0, 1, 0], [1, 1, 1]], dtype=int),
}
DIRS = {
    "right": (0, 1),
    "left": (0, -1),
    "up": (-1, 0),
    "down": (1, 0),
}
TIP_OFFSETS = {
    "right": (1, 2),
    "left": (1, 0),
    "up": (0, 1),
    "down": (2, 1),
}
ARROW_CELLS = {
    name: [tuple(map(int, cell)) for cell in np.argwhere(pattern == 1)]
    for name, pattern in ARROW_PATTERNS.items()
}


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(int(cr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _extract_arrows(grid: np.ndarray) -> list[dict]:
    arrows: list[dict] = []
    for cells in _components(grid == RED):
        if len(cells) != 5:
            continue
        r0, c0, r1, c1 = _bbox(cells)
        if r1 - r0 != 2 or c1 - c0 != 2:
            continue
        patch = np.zeros((3, 3), dtype=int)
        for r, c in cells:
            patch[r - r0, c - c0] = 1
        matched = None
        for name, pattern in ARROW_PATTERNS.items():
            if np.array_equal(patch, pattern):
                matched = name
                break
        assert matched is not None, "red components must be valid 3x3 arrows"
        tr, tc = TIP_OFFSETS[matched]
        arrows.append(
            {
                "name": matched,
                "direction": DIRS[matched],
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "cells": [(int(r), int(c)) for r, c in cells],
                "tip": (int(r0 + tr), int(c0 + tc)),
            }
        )
    arrows.sort(key=lambda a: (a["bbox"][0], a["bbox"][1], a["name"]))
    return arrows


def _extract_rectangles(grid: np.ndarray) -> list[dict]:
    rectangles: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid)):
        if color in (BG, RED, GREY):
            continue
        for cells in _components(grid == color):
            r0, c0, r1, c1 = _bbox(cells)
            area = (r1 - r0 + 1) * (c1 - c0 + 1)
            assert len(cells) == area, "each colored object must be a solid rectangle"
            rectangles.append(
                {
                    "color": int(color),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )
    rectangles.sort(key=lambda rect: (rect["bbox"][0], rect["bbox"][1], rect["color"]))
    return rectangles


def _cell_to_rect(rectangles: list[dict], shape: tuple[int, int]) -> np.ndarray:
    idx = np.full(shape, -1, dtype=int)
    for rid, rect in enumerate(rectangles):
        r0, c0, r1, c1 = rect["bbox"]
        idx[r0 : r1 + 1, c0 : c1 + 1] = int(rid)
    return idx


def _ray_hits(
    tip: tuple[int, int],
    direction: tuple[int, int],
    cell_to_rect: np.ndarray,
) -> tuple[int | None, int | None, set[int]]:
    h, w = cell_to_rect.shape
    dr, dc = direction
    r, c = tip
    first_id: int | None = None
    first_distance: int | None = None
    seen_ids: set[int] = set()
    step = 0
    while True:
        step += 1
        r += dr
        c += dc
        if not (0 <= r < h and 0 <= c < w):
            break
        rid = int(cell_to_rect[r, c])
        if rid >= 0:
            seen_ids.add(int(rid))
            if first_id is None:
                first_id = int(rid)
                first_distance = int(step)
    return first_id, first_distance, seen_ids


def _shift_bbox_to_edge(
    bbox: tuple[int, int, int, int],
    direction_name: str,
    shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    h, w = shape
    r0, c0, r1, c1 = bbox
    if direction_name == "right":
        delta = (w - 1) - c1
        return r0, c0 + delta, r1, c1 + delta
    if direction_name == "left":
        delta = c0
        return r0, c0 - delta, r1, c1 - delta
    if direction_name == "down":
        delta = (h - 1) - r1
        return r0 + delta, c0, r1 + delta, c1
    assert direction_name == "up"
    delta = r0
    return r0 - delta, c0, r1 - delta, c1


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    arrows = _extract_arrows(g)
    rectangles = _extract_rectangles(g)
    rect_index = _cell_to_rect(rectangles, g.shape)

    best_by_rect: dict[int, dict] = {}
    for arrow_id, arrow in enumerate(arrows):
        target_id, distance, seen_ids = _ray_hits(arrow["tip"], arrow["direction"], rect_index)
        assert target_id is not None and distance is not None, "each arrow must point to a rectangle"
        assert seen_ids == {int(target_id)}, "each arrow ray must contain exactly one rectangle"
        candidate = {
            "arrow_id": int(arrow_id),
            "rect_id": int(target_id),
            "distance": int(distance),
            "direction_name": arrow["name"],
        }
        prev = best_by_rect.get(int(target_id))
        if prev is None or (candidate["distance"], candidate["arrow_id"]) < (prev["distance"], prev["arrow_id"]):
            best_by_rect[int(target_id)] = candidate

    out = g.copy()
    for arrow in arrows:
        for r, c in arrow["cells"]:
            out[r, c] = BG

    for rect_id, winner in best_by_rect.items():
        r0, c0, r1, c1 = rectangles[int(rect_id)]["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = BG
        nr0, nc0, nr1, nc1 = _shift_bbox_to_edge((r0, c0, r1, c1), winner["direction_name"], (h, w))
        out[nr0 : nr1 + 1, nc0 : nc1 + 1] = rectangles[int(rect_id)]["color"]

    return out


def _bbox_overlaps(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    *,
    pad: int = 0,
) -> bool:
    return not (
        a[2] < b[0] - pad
        or b[2] < a[0] - pad
        or a[3] < b[1] - pad
        or b[3] < a[1] - pad
    )


def _bbox_cells(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _moat(cells: list[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = int(r + dr)
                nc = int(c + dc)
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((int(nr), int(nc)))
    return out


def _object_layout(
    h: int,
    w: int,
    color: int,
    winner_dir: str,
    existing: list[dict],
) -> dict | None:
    for _ in range(120):
        obj_h = random.randint(2, min(5, h - 6))
        obj_w = random.randint(2, min(5, w - 6))
        if winner_dir in ("left", "right"):
            max_shift = w - obj_w - 4
        else:
            max_shift = h - obj_h - 4
        if max_shift < 2:
            continue
        shift = random.randint(2, min(7, max_shift))

        if winner_dir == "right":
            r0 = random.randint(0, h - obj_h)
            bbox = (r0, w - obj_w - shift, r0 + obj_h - 1, w - shift - 1)
            final_bbox = (r0, w - obj_w, r0 + obj_h - 1, w - 1)
        elif winner_dir == "left":
            r0 = random.randint(0, h - obj_h)
            bbox = (r0, shift, r0 + obj_h - 1, shift + obj_w - 1)
            final_bbox = (r0, 0, r0 + obj_h - 1, obj_w - 1)
        elif winner_dir == "down":
            c0 = random.randint(0, w - obj_w)
            bbox = (h - obj_h - shift, c0, h - shift - 1, c0 + obj_w - 1)
            final_bbox = (h - obj_h, c0, h - 1, c0 + obj_w - 1)
        else:
            c0 = random.randint(0, w - obj_w)
            bbox = (shift, c0, shift + obj_h - 1, c0 + obj_w - 1)
            final_bbox = (0, c0, obj_h - 1, c0 + obj_w - 1)

        if any(_bbox_overlaps(bbox, other["bbox"], pad=1) for other in existing):
            continue
        if any(_bbox_overlaps(final_bbox, other["final_bbox"], pad=1) for other in existing):
            continue
        return {
            "color": int(color),
            "bbox": tuple(map(int, bbox)),
            "final_bbox": tuple(map(int, final_bbox)),
            "winner_dir": winner_dir,
        }
    return None


def _object_occupancy(shape: tuple[int, int], objects: list[dict]) -> np.ndarray:
    occ = np.full(shape, -1, dtype=int)
    for idx, obj in enumerate(objects):
        r0, c0, r1, c1 = obj["bbox"]
        occ[r0 : r1 + 1, c0 : c1 + 1] = int(idx)
    return occ


def _arrow_top_left(
    bbox: tuple[int, int, int, int],
    direction_name: str,
    lane: int,
    gap: int,
) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    if direction_name == "right":
        return lane - 1, c0 - gap - 3
    if direction_name == "left":
        return lane - 1, c1 + gap + 1
    if direction_name == "up":
        return r1 + gap + 1, lane - 1
    assert direction_name == "down"
    return r0 - gap - 3, lane - 1


def _arrow_candidate_cells(top: int, left: int, direction_name: str) -> list[tuple[int, int]]:
    return [(int(top + dr), int(left + dc)) for dr, dc in ARROW_CELLS[direction_name]]


def _candidate_arrows_for_object(
    obj_id: int,
    objects: list[dict],
    shape: tuple[int, int],
    blocked_by_objects: set[tuple[int, int]],
) -> list[dict]:
    h, w = shape
    obj = objects[int(obj_id)]
    r0, c0, r1, c1 = obj["bbox"]
    occ = _object_occupancy(shape, objects)
    candidates: list[dict] = []

    for direction_name in ("right", "left", "up", "down"):
        if direction_name in ("right", "left"):
            lanes = list(range(r0, r1 + 1))
            max_gap = c0 - 3 if direction_name == "right" else w - c1 - 4
        else:
            lanes = list(range(c0, c1 + 1))
            max_gap = h - r1 - 4 if direction_name == "up" else r0 - 3
        if max_gap < 1:
            continue

        random.shuffle(lanes)
        gaps = list(range(1, min(5, max_gap) + 1))
        random.shuffle(gaps)
        for lane in lanes:
            for gap in gaps:
                top, left = _arrow_top_left(obj["bbox"], direction_name, int(lane), int(gap))
                if not (0 <= top <= h - 3 and 0 <= left <= w - 3):
                    continue
                cells = _arrow_candidate_cells(top, left, direction_name)
                if any(cell in blocked_by_objects for cell in cells):
                    continue
                tip_r, tip_c = TIP_OFFSETS[direction_name]
                tip = (int(top + tip_r), int(left + tip_c))
                target_id, distance, seen_ids = _ray_hits(tip, DIRS[direction_name], occ)
                if target_id != obj_id or seen_ids != {int(obj_id)} or distance is None:
                    continue
                candidates.append(
                    {
                        "target": int(obj_id),
                        "dir_name": direction_name,
                        "distance": int(distance),
                        "top": int(top),
                        "left": int(left),
                        "tip": tip,
                        "cells": cells,
                    }
                )
    return candidates


def _build_input_grid(shape: tuple[int, int], objects: list[dict], arrows: list[dict]) -> np.ndarray:
    h, w = shape
    grid = np.full((h, w), BG, dtype=int)
    for obj in objects:
        r0, c0, r1, c1 = obj["bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = obj["color"]
    for arrow in arrows:
        for r, c in arrow["cells"]:
            grid[r, c] = RED
    return grid


def _planned_output(shape: tuple[int, int], objects: list[dict], arrows: list[dict]) -> np.ndarray | None:
    out = np.full(shape, BG, dtype=int)
    winners: dict[int, dict] = {}
    for order, arrow in enumerate(arrows):
        arrow = dict(arrow)
        arrow["order"] = int(order)
        prev = winners.get(int(arrow["target"]))
        if prev is None or (arrow["distance"], arrow["order"]) < (prev["distance"], prev["order"]):
            winners[int(arrow["target"])] = arrow

    for obj_id, obj in enumerate(objects):
        winner = winners.get(int(obj_id))
        bbox = obj["bbox"] if winner is None else _shift_bbox_to_edge(obj["bbox"], winner["dir_name"], shape)
        r0, c0, r1, c1 = bbox
        if np.any(out[r0 : r1 + 1, c0 : c1 + 1] != BG):
            return None
        out[r0 : r1 + 1, c0 : c1 + 1] = obj["color"]
    return out


def _ray_object_colors(grid: np.ndarray, tip: tuple[int, int], direction: tuple[int, int]) -> list[int]:
    h, w = grid.shape
    dr, dc = direction
    r, c = tip
    colors: list[int] = []
    while True:
        r += dr
        c += dc
        if not (0 <= r < h and 0 <= c < w):
            break
        value = int(grid[r, c])
        if value != BG:
            colors.append(int(value))
    return colors


def _validate_generated_case(
    grid: np.ndarray,
    objects: list[dict],
    arrows: list[dict],
    *,
    require_competition: bool,
) -> bool:
    extracted_arrows = _extract_arrows(grid)
    extracted_rectangles = _extract_rectangles(grid)
    if len(extracted_arrows) != len(arrows) or len(extracted_rectangles) != len(objects):
        return False
    if len({rect["color"] for rect in extracted_rectangles}) != len(extracted_rectangles):
        return False

    occ = _cell_to_rect(extracted_rectangles, grid.shape)
    target_counts: dict[int, int] = {}
    for arrow in extracted_arrows:
        target_id, _distance, seen_ids = _ray_hits(arrow["tip"], arrow["direction"], occ)
        if target_id is None or seen_ids != {int(target_id)}:
            return False
        ray_colors = _ray_object_colors(grid, arrow["tip"], arrow["direction"])
        if not ray_colors or RED in ray_colors or GREY in ray_colors or len(set(ray_colors)) != 1:
            return False
        target_counts[int(target_id)] = target_counts.get(int(target_id), 0) + 1

    has_competition = any(count > 1 for count in target_counts.values())
    if has_competition != require_competition:
        return False

    expected = _planned_output(grid.shape, objects, arrows)
    if expected is None or np.array_equal(expected, grid):
        return False
    return np.array_equal(main(grid), expected)


def _choose_arrows_for_objects(objects: list[dict], shape: tuple[int, int], require_competition: bool) -> list[dict] | None:
    blocked_by_objects = _moat(
        [cell for obj in objects for cell in _bbox_cells(obj["bbox"])],
        shape,
    )
    all_candidates = {
        idx: _candidate_arrows_for_object(idx, objects, shape, blocked_by_objects)
        for idx in range(len(objects))
    }
    if any(not candidates for candidates in all_candidates.values()):
        return None

    order = list(range(len(objects)))
    if require_competition:
        random.shuffle(order)
        competitive_obj = order[0]
    else:
        competitive_obj = None

    selection_order = [competitive_obj] + [idx for idx in order if idx != competitive_obj] if competitive_obj is not None else order
    arrow_blocked: set[tuple[int, int]] = set()
    chosen: list[dict] = []
    primary_by_object: dict[int, dict] = {}

    for obj_id in selection_order:
        assert obj_id is not None
        winner_dir = objects[int(obj_id)]["winner_dir"]
        candidates = [
            cand
            for cand in all_candidates[int(obj_id)]
            if cand["dir_name"] == winner_dir and not any(cell in arrow_blocked for cell in cand["cells"])
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda cand: (cand["distance"], cand["top"], cand["left"]))
        keep = candidates[: min(3, len(candidates))]
        primary = random.choice(keep)
        primary_by_object[int(obj_id)] = primary
        chosen.append(primary)
        arrow_blocked |= _moat(primary["cells"], shape)

    if competitive_obj is not None:
        winner = primary_by_object[int(competitive_obj)]
        rivals = [
            cand
            for cand in all_candidates[int(competitive_obj)]
            if cand["distance"] > winner["distance"] and not any(cell in arrow_blocked for cell in cand["cells"])
        ]
        if not rivals:
            return None
        different_dir = [cand for cand in rivals if cand["dir_name"] != winner["dir_name"]]
        pool = different_dir if different_dir else rivals
        pool.sort(key=lambda cand: (cand["distance"], cand["top"], cand["left"]))
        rival = random.choice(pool[: min(4, len(pool))])
        chosen.append(rival)

    random.shuffle(chosen)
    return chosen


def generate_input() -> np.ndarray:
    for _ in range(400):
        h = random.randint(14, 22)
        w = random.randint(14, 22)
        n_objects = random.randint(2, 4)
        colors = random.sample(OBJECT_COLORS, n_objects)
        require_competition = random.random() < 0.5

        objects: list[dict] = []
        ok = True
        for color in colors:
            winner_dir = random.choice(list(DIRS))
            obj = _object_layout(h, w, int(color), winner_dir, objects)
            if obj is None:
                ok = False
                break
            objects.append(obj)
        if not ok:
            continue

        arrows = _choose_arrows_for_objects(objects, (h, w), require_competition)
        if arrows is None:
            continue

        grid = _build_input_grid((h, w), objects, arrows)
        if _validate_generated_case(grid, objects, arrows, require_competition=require_competition):
            return grid

    raise RuntimeError("failed to generate a valid ARC instance")
