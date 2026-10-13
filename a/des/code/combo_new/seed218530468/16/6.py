from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
CORNER_NAMES = ("tl", "tr", "bl", "br")
ALL_LOCAL_CELLS = ((0, 0), (0, 1), (1, 0), (1, 1))
MISSING_BY_CORNER = {
    "tl": (0, 0),
    "tr": (0, 1),
    "bl": (1, 0),
    "br": (1, 1),
}
LOCAL_CELLS_BY_CORNER = {
    corner: tuple(cell for cell in ALL_LOCAL_CELLS if cell != missing)
    for corner, missing in MISSING_BY_CORNER.items()
}
MIN_SIDE = 10
MAX_SIDE = 20
RETRY_LIMIT = 400
DISTANCE_MARGIN = 8

_CANDIDATE_CACHE: dict[int, list[dict[str, object]]] = {}
_PAIR_CACHE: dict[int, tuple[list[tuple[int, int]], list[tuple[int, int]]]] = {}


def _corner_anchor(side: int, corner: str) -> tuple[int, int]:
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, side - 1
    if corner == "bl":
        return side - 1, 0
    return side - 1, side - 1


def _corner_distance4(r0: int, c0: int, side: int, corner: str) -> int:
    center_r4 = 2 * r0 + 1
    center_c4 = 2 * c0 + 1
    cr, cc = _corner_anchor(side, corner)
    corner_r4 = 2 * cr
    corner_c4 = 2 * cc
    dr = center_r4 - corner_r4
    dc = center_c4 - corner_c4
    return int(dr * dr + dc * dc)


def _nearest_corner(r0: int, c0: int, side: int) -> tuple[str, int, int]:
    scored = sorted(
        (_corner_distance4(r0, c0, side, corner), corner) for corner in CORNER_NAMES
    )
    best_dist, best_corner = scored[0]
    gap = int(scored[1][0] - best_dist)
    return best_corner, int(best_dist), gap


def _slide_origin(r0: int, c0: int, side: int, corner: str) -> tuple[int, int, int]:
    if corner == "tl":
        steps = min(r0, c0)
        return r0 - steps, c0 - steps, steps
    if corner == "tr":
        steps = min(r0, side - 2 - c0)
        return r0 - steps, c0 + steps, steps
    if corner == "bl":
        steps = min(side - 2 - r0, c0)
        return r0 + steps, c0 - steps, steps
    steps = min(side - 2 - r0, side - 2 - c0)
    return r0 + steps, c0 + steps, steps


def _bbox_cells(top_left: tuple[int, int]) -> frozenset[tuple[int, int]]:
    r0, c0 = top_left
    return frozenset((r0 + dr, c0 + dc) for dr, dc in ALL_LOCAL_CELLS)


def _cells_from_origin_and_corner(
    top_left: tuple[int, int], corner: str
) -> frozenset[tuple[int, int]]:
    r0, c0 = top_left
    return frozenset((r0 + dr, c0 + dc) for dr, dc in LOCAL_CELLS_BY_CORNER[corner])


def _cells_from_origin_and_missing(
    top_left: tuple[int, int], missing: tuple[int, int]
) -> frozenset[tuple[int, int]]:
    r0, c0 = top_left
    return frozenset((r0 + dr, c0 + dc) for dr, dc in ALL_LOCAL_CELLS if (dr, dc) != missing)


def _boxes_overlap(a: tuple[int, int], b: tuple[int, int], margin: int = 0) -> bool:
    ar, ac = a
    br, bc = b
    return not (
        ar + 1 + margin < br
        or br + 1 + margin < ar
        or ac + 1 + margin < bc
        or bc + 1 + margin < ac
    )


def _missing_corner(cells: Iterable[tuple[int, int]]) -> tuple[int, int]:
    present = set(cells)
    for cell in ALL_LOCAL_CELLS:
        if cell not in present:
            return cell
    raise AssertionError("invalid L triomino: no missing corner")


def _extract_objects(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    side = int(g.shape[0])
    colors = sorted(int(color) for color in np.unique(g) if int(color) != BG)
    objects: list[dict[str, object]] = []

    for color in colors:
        pts = np.argwhere(g == color)
        assert int(pts.shape[0]) == 3
        r0, c0 = map(int, pts.min(axis=0))
        r1, c1 = map(int, pts.max(axis=0))
        assert (r1 - r0 + 1, c1 - c0 + 1) == (2, 2)
        local_cells = frozenset((int(r - r0), int(c - c0)) for r, c in pts)
        missing = _missing_corner(local_cells)
        target, distance4, gap = _nearest_corner(r0, c0, side)
        assert gap > 0
        fr, fc, _steps = _slide_origin(r0, c0, side, target)
        objects.append(
            {
                "color": int(color),
                "origin": (r0, c0),
                "missing": missing,
                "target": target,
                "distance4": int(distance4),
                "final": (fr, fc),
            }
        )

    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each colored L triomino toward its nearest corner and slide it diagonally to that corner-side boundary.

    Input structure:
    - The grid is square and has a black background.
    - Each non-black color appears exactly once as a monochrome L triomino occupying three cells of a 2x2 box.
    - Different triomino colors are unique, so each color identifies one object.

    Rule:
    - For each triomino, compute the center of its 2x2 bounding box and find the nearest grid corner.
    - Rotate the triomino so the missing corner of the 2x2 box points toward that nearest corner.
    - Move the rotated triomino along the matching diagonal until its 2x2 box first touches a grid edge.
    - If moved triominoes overlap, the one whose original box center was farther from its target corner is drawn on top.

    Key insight:
    - The nearest corner is the shared hidden parameter: it determines both the L opening after rotation and the diagonal slide direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    side, width = map(int, g.shape)
    assert side == width

    out = np.zeros_like(g)
    objects = _extract_objects(g)
    for obj in sorted(objects, key=lambda item: (int(item["distance4"]), int(item["color"]))):
        fr, fc = obj["final"]
        for dr, dc in LOCAL_CELLS_BY_CORNER[str(obj["target"])]:
            out[fr + dr, fc + dc] = int(obj["color"])
    return out


def _candidate_data(side: int) -> tuple[list[dict[str, object]], list[tuple[int, int]], list[tuple[int, int]]]:
    cached_candidates = _CANDIDATE_CACHE.get(side)
    cached_pairs = _PAIR_CACHE.get(side)
    if cached_candidates is not None and cached_pairs is not None:
        return cached_candidates, cached_pairs[0], cached_pairs[1]

    candidates: list[dict[str, object]] = []
    for r0 in range(1, side - 2):
        for c0 in range(1, side - 2):
            target, distance4, gap = _nearest_corner(r0, c0, side)
            if gap < DISTANCE_MARGIN:
                continue
            fr, fc, steps = _slide_origin(r0, c0, side, target)
            if steps <= 0:
                continue
            candidates.append(
                {
                    "origin": (r0, c0),
                    "target": target,
                    "distance4": int(distance4),
                    "final": (fr, fc),
                    "input_box": _bbox_cells((r0, c0)),
                    "final_box": _bbox_cells((fr, fc)),
                    "final_cells": _cells_from_origin_and_corner((fr, fc), target),
                }
            )

    partial_pairs: list[tuple[int, int]] = []
    full_pairs: list[tuple[int, int]] = []
    for i in range(len(candidates)):
        a = candidates[i]
        for j in range(i + 1, len(candidates)):
            b = candidates[j]
            if a["target"] != b["target"]:
                continue
            if _boxes_overlap(a["origin"], b["origin"]):
                continue
            overlap = a["final_cells"] & b["final_cells"]
            if not overlap:
                continue
            if len(overlap) < 3:
                partial_pairs.append((i, j))
            else:
                full_pairs.append((i, j))

    _CANDIDATE_CACHE[side] = candidates
    _PAIR_CACHE[side] = (partial_pairs, full_pairs)
    return candidates, partial_pairs, full_pairs


def _select_layout(side: int, count: int, force_overlap: bool) -> list[dict[str, object]] | None:
    candidates, partial_pairs, full_pairs = _candidate_data(side)
    if len(candidates) < count:
        return None

    selected_indices: list[int] = []
    used_origins: list[tuple[int, int]] = []
    used_final_boxes: list[tuple[int, int]] = []
    used_output_cells: set[tuple[int, int]] = set()

    if force_overlap:
        pair_pool = partial_pairs if partial_pairs else full_pairs
        if not pair_pool:
            return None
        first_idx, second_idx = random.choice(pair_pool)
        selected_indices.extend([first_idx, second_idx])
        for idx in (first_idx, second_idx):
            cand = candidates[idx]
            used_origins.append(cand["origin"])
            used_final_boxes.append(cand["final"])
            used_output_cells.update(cand["final_cells"])

    remaining = list(range(len(candidates)))
    random.shuffle(remaining)
    chosen = set(selected_indices)

    for idx in remaining:
        if idx in chosen:
            continue
        cand = candidates[idx]
        if any(_boxes_overlap(cand["origin"], origin) for origin in used_origins):
            continue
        if any(_boxes_overlap(cand["final"], final) for final in used_final_boxes):
            continue
        if cand["final_cells"] & used_output_cells:
            continue
        selected_indices.append(idx)
        chosen.add(idx)
        used_origins.append(cand["origin"])
        used_final_boxes.append(cand["final"])
        used_output_cells.update(cand["final_cells"])
        if len(selected_indices) == count:
            break

    if len(selected_indices) != count:
        return None

    selected = [candidates[idx] for idx in selected_indices]
    distinct_corners = {str(cand["target"]) for cand in selected}
    if count >= 3 and len(distinct_corners) < 2:
        return None
    return selected


def _paint_input(objects: list[dict[str, object]], side: int) -> np.ndarray:
    grid = np.zeros((side, side), dtype=int)
    colors = random.sample(PALETTE, len(objects))
    for obj, color in zip(objects, colors):
        missing = random.choice(ALL_LOCAL_CELLS)
        for r, c in _cells_from_origin_and_missing(obj["origin"], missing):
            grid[r, c] = int(color)
    return grid


def _has_overlap(objects: list[dict[str, object]]) -> bool:
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            if objects[i]["final_cells"] & objects[j]["final_cells"]:
                return True
    return False


def _validate_generated(grid: np.ndarray, expect_overlap: bool) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2
    side, width = map(int, grid.shape)
    assert side == width
    assert side <= 30

    objects = _extract_objects(grid)
    assert 2 <= len(objects) <= len(PALETTE)

    overlap_pairs = 0
    for i in range(len(objects)):
        ai = objects[i]
        atarget = str(ai["target"])
        afcells = _cells_from_origin_and_corner(ai["final"], atarget)
        for j in range(i + 1, len(objects)):
            bj = objects[j]
            btarget = str(bj["target"])
            bfcells = _cells_from_origin_and_corner(bj["final"], btarget)
            if afcells & bfcells:
                overlap_pairs += 1
                assert atarget == btarget
                assert int(ai["distance4"]) != int(bj["distance4"])

    if expect_overlap:
        assert overlap_pairs >= 1
    else:
        assert overlap_pairs == 0

    out = main(grid)
    assert out.shape == grid.shape


def generate_input() -> np.ndarray:
    for _ in range(RETRY_LIMIT):
        side = random.randint(MIN_SIDE, MAX_SIDE)
        count_max = min(6, max(2, side // 3 + 1))
        force_overlap = bool(random.getrandbits(1))
        if force_overlap and count_max < 2:
            force_overlap = False

        count_min = 3 if force_overlap and count_max >= 3 else 2
        count = random.randint(count_min, count_max)
        layout = _select_layout(side, count, force_overlap)
        if layout is None:
            continue

        grid = _paint_input(layout, side)
        try:
            _validate_generated(grid, expect_overlap=force_overlap)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid input for puzzle 6")
