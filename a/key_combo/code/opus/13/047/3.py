from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
RED = 2
GREY = 5

INNER = 3
BOX = INNER + 2
GAP = 3
PITCH = BOX + GAP

TEMPLATE_COLORS = [1, 3, 4, 6, 7, 8, 9]
SIDE_ORDER = ("left", "top", "right", "bottom")
SIDE_TO_ORIENTATION = {
    "left": "V",
    "right": "V",
    "top": "H",
    "bottom": "H",
}
ROTATION_FOR_SIDE = {
    "left": 0,
    "top": -1,
    "right": 2,
    "bottom": 1,
}


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _coord_set(coords: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, rc)) for rc in coords}


def _is_rectangular_ring(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if len(coords) != 2 * (h + w) - 4:
        return False
    pts = _coord_set(coords)
    for c in range(c0, c1 + 1):
        if (r0, c) not in pts or (r1, c) not in pts:
            return False
    for r in range(r0, r1 + 1):
        if (r, c0) not in pts or (r, c1) not in pts:
            return False
    return True


def _all_rotations_distinct(patch: np.ndarray) -> bool:
    variants = [np.rot90(patch, k) for k in range(4)]
    for i in range(4):
        for j in range(i + 1, 4):
            if np.array_equal(variants[i], variants[j]):
                return False
    return True


def _classify_cells(grid: np.ndarray):
    grey_comps = _components_from_mask(grid == GREY)
    cell_comps = [comp for comp in grey_comps if _is_rectangular_ring(comp)]
    path_comps = [comp for comp in grey_comps if not _is_rectangular_ring(comp)]

    cells = []
    template = None
    targets = []

    for idx, comp in enumerate(cell_comps):
        r0, c0, r1, c1 = _bbox(comp)
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        if inner.shape[0] < 3 or inner.shape[1] < 3:
            continue
        center = ((r0 + r1) // 2, (c0 + c1) // 2)
        cr, cc = center
        if grid[cr, cc] != RED:
            cells.append({"id": idx, "bbox": (r0, c0, r1, c1), "center": center})
            continue

        patch = grid[cr - 1 : cr + 2, cc - 1 : cc + 2]
        if patch.shape != (3, 3):
            cells.append({"id": idx, "bbox": (r0, c0, r1, c1), "center": center})
            continue

        extras = int(np.count_nonzero((patch != BG) & (patch != RED) & (patch != GREY)))
        non_bg_inner = int(np.count_nonzero(inner != BG))

        cell = {
            "id": idx,
            "bbox": (r0, c0, r1, c1),
            "center": center,
            "patch": patch.copy(),
            "inner": inner.copy(),
        }
        cells.append(cell)

        if extras in (2, 3) and non_bg_inner == extras + 1 and _all_rotations_distinct(patch):
            assert template is None, "expected exactly one template cell"
            template = cell
        elif extras == 0 and non_bg_inner == 1:
            targets.append(cell)

    assert template is not None, "template cell not found"
    assert 3 <= len(targets) <= 5, "expected 3-5 target cells"
    return cells, template, targets, path_comps


def _side_point(cell: dict, side: str) -> tuple[int, int] | None:
    r0, c0, r1, c1 = cell["bbox"]
    cr, cc = cell["center"]
    if side == "left":
        return (cr, c0 - 2) if c0 - 2 >= 0 else None
    if side == "right":
        return (cr, c1 + 2)
    if side == "top":
        return (r0 - 2, cc) if r0 - 2 >= 0 else None
    if side == "bottom":
        return (r1 + 2, cc)
    raise ValueError(side)


def _endpoint_role(cells: list[dict], endpoint: tuple[int, int]):
    matches = []
    for cell in cells:
        for side in SIDE_ORDER:
            if _side_point(cell, side) == endpoint:
                matches.append((cell, side))
    assert len(matches) <= 1, "path endpoint must identify at most one red-centered cell side"
    return matches[0] if matches else None


def _path_endpoints(coords: np.ndarray) -> list[tuple[int, int]]:
    pts = _coord_set(coords)
    endpoints = []
    for r, c in pts:
        degree = sum((nr, nc) in pts for nr, nc in _neighbors4(r, c))
        if degree == 1:
            endpoints.append((r, c))
        else:
            assert degree == 2, "paths must be non-branching"
    assert len(endpoints) == 2, "each path must have exactly two endpoints"
    return endpoints


def _analyze_paths(grid: np.ndarray):
    cells, template, targets, path_comps = _classify_cells(grid)
    red_cells = [template, *targets]
    target_by_id = {cell["id"]: cell for cell in targets}

    target_arrival_side: dict[int, str] = {}
    path_shapes: list[str] = []

    for comp in path_comps:
        endpoints = _path_endpoints(comp)
        m0 = _endpoint_role(red_cells, endpoints[0])
        m1 = _endpoint_role(red_cells, endpoints[1])
        assert m0 is not None and m1 is not None, "each path must connect two red-centered cell sides"
        (cell0, side0), (cell1, side1) = m0, m1

        if cell0["id"] == template["id"]:
            tpl_side, other_cell, other_side = side0, cell1, side1
        elif cell1["id"] == template["id"]:
            tpl_side, other_cell, other_side = side1, cell0, side0
        else:
            raise AssertionError("every path must touch the template")

        assert other_cell["id"] in target_by_id, "template paths may only reach targets"
        assert other_cell["id"] not in target_arrival_side, "target should be connected by at most one path"
        target_arrival_side[other_cell["id"]] = other_side

        shape = "straight" if endpoints[0][0] == endpoints[1][0] or endpoints[0][1] == endpoints[1][1] else "L"
        path_shapes.append(shape)

        # Orientation consistency: same-orientation endpoints can only connect straight.
        orient0 = SIDE_TO_ORIENTATION[tpl_side]
        orient1 = SIDE_TO_ORIENTATION[other_side]
        if orient0 == orient1:
            assert shape == "straight"
        else:
            assert shape == "L"

    return {
        "cells": cells,
        "template": template,
        "targets": targets,
        "target_arrival_side": target_arrival_side,
        "path_shapes": path_shapes,
    }


def _stamp_template(out: np.ndarray, target: dict, patch: np.ndarray) -> None:
    cr, cc = target["center"]
    r0, c0, r1, c1 = target["bbox"]
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            rr = cr + dr
            cc2 = cc + dc
            if not (0 <= rr < out.shape[0] and 0 <= cc2 < out.shape[1]):
                continue
            if not (r0 + 1 <= rr <= r1 - 1 and c0 + 1 <= cc2 <= c1 - 1):
                continue
            out[rr, cc2] = int(patch[dr + 1, dc + 1])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp one asymmetric template onto path-reached targets, rotating it by the path's arrival direction.

    Input structure:
    - The grid is partitioned into regular grey box cells on a black background.
    - One boxed cell contains a 3x3 asymmetric template centered on a red pixel, while 3-5 other boxed cells contain only a red center pixel.
    - Separate thin grey path components run through the gaps between boxed cells, each touching the template side-center and optionally one target side-center.

    Rule:
    - Find the unique template cell: its red center has 2-3 extra non-black, non-grey neighbors in its 3x3 neighborhood, and the 3x3 pattern is distinct under all four quarter-turn rotations.
    - For each target cell with only a red center, check whether a grey path connects one template side-center to one target side-center.
    - Rotate the template by the target's arrival side: left -> 0 degrees, top -> clockwise 90 degrees, right -> 180 degrees, bottom -> clockwise 270 degrees, then stamp that rotated 3x3 patch onto the target with centers aligned and clipping to the target cell interior.
    - Grey paths, grey cell borders, the template cell, and any unconnected targets stay unchanged.

    Key insight:
    - The grey paths do not merely mark which targets are active; the side from which a path reaches a target is the hidden rotation code for the copied template.
    """
    g = np.asarray(grid, dtype=int)
    analysis = _analyze_paths(g)
    template = analysis["template"]
    targets = analysis["targets"]
    target_arrival_side = analysis["target_arrival_side"]

    out = g.copy()
    for target in targets:
        side = target_arrival_side.get(target["id"])
        if side is None:
            continue
        rotated = np.rot90(template["patch"], ROTATION_FOR_SIDE[side])
        _stamp_template(out, target, rotated)
    return out


def _cell_bbox(cell_rc: tuple[int, int]) -> tuple[int, int, int, int]:
    rr, cc = cell_rc
    r0 = rr * PITCH
    c0 = cc * PITCH
    return r0, c0, r0 + BOX - 1, c0 + BOX - 1


def _cell_center(cell_rc: tuple[int, int]) -> tuple[int, int]:
    r0, c0, _r1, _c1 = _cell_bbox(cell_rc)
    return r0 + BOX // 2, c0 + BOX // 2


def _draw_ring(grid: np.ndarray, cell_rc: tuple[int, int]) -> None:
    r0, c0, r1, c1 = _cell_bbox(cell_rc)
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY


def _internal_sides(rows: int, cols: int, cell_rc: tuple[int, int]) -> list[str]:
    r, c = cell_rc
    sides = []
    if c > 0:
        sides.append("left")
    if r > 0:
        sides.append("top")
    if c < cols - 1:
        sides.append("right")
    if r < rows - 1:
        sides.append("bottom")
    return sides


def _neighbor(cell_rc: tuple[int, int], side: str) -> tuple[int, int]:
    r, c = cell_rc
    if side == "left":
        return r, c - 1
    if side == "right":
        return r, c + 1
    if side == "top":
        return r - 1, c
    if side == "bottom":
        return r + 1, c
    raise ValueError(side)


def _endpoint(cell_rc: tuple[int, int], side: str) -> tuple[int, int]:
    r0, c0, r1, c1 = _cell_bbox(cell_rc)
    cr, cc = _cell_center(cell_rc)
    if side == "left":
        return cr, c0 - 2
    if side == "right":
        return cr, c1 + 2
    if side == "top":
        return r0 - 2, cc
    if side == "bottom":
        return r1 + 2, cc
    raise ValueError(side)


def _line(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    if a[0] == b[0]:
        step = 1 if b[1] >= a[1] else -1
        return [(a[0], c) for c in range(a[1], b[1] + step, step)]
    if a[1] == b[1]:
        step = 1 if b[0] >= a[0] else -1
        return [(r, a[1]) for r in range(a[0], b[0] + step, step)]
    raise ValueError("line endpoints must be axis-aligned")


def _route(
    base: np.ndarray,
    start: tuple[int, int],
    start_side: str,
    end: tuple[int, int],
    end_side: str,
    used: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], str] | None:
    orient0 = SIDE_TO_ORIENTATION[start_side]
    orient1 = SIDE_TO_ORIENTATION[end_side]

    candidates: list[tuple[list[tuple[int, int]], str]] = []
    if orient0 == orient1:
        if orient0 == "V" and start[1] == end[1]:
            candidates.append((_line(start, end), "straight"))
        elif orient0 == "H" and start[0] == end[0]:
            candidates.append((_line(start, end), "straight"))
    elif orient0 == "V" and orient1 == "H":
        corner = (end[0], start[1])
        path = _line(start, corner) + _line(corner, end)[1:]
        candidates.append((path, "L"))
    else:
        corner = (start[0], end[1])
        path = _line(start, corner) + _line(corner, end)[1:]
        candidates.append((path, "L"))

    for path, shape in candidates:
        if any(not (0 <= r < base.shape[0] and 0 <= c < base.shape[1]) for r, c in path):
            continue
        if any(base[r, c] != BG for r, c in path):
            continue
        if any((r, c) in used for r, c in path):
            continue
        interior = set(path[1:-1])
        if interior & forbidden:
            continue
        return path, shape
    return None


def _design_template_patch() -> np.ndarray:
    for _ in range(200):
        patch = np.zeros((3, 3), dtype=int)
        patch[1, 1] = RED
        n_extra = random.choice([2, 3])
        positions = random.sample([(r, c) for r in range(3) for c in range(3) if (r, c) != (1, 1)], n_extra)
        colors = random.sample(TEMPLATE_COLORS, n_extra)
        for (r, c), color in zip(positions, colors):
            patch[r, c] = color
        if _all_rotations_distinct(patch):
            return patch
    raise RuntimeError("failed to build an asymmetric template")


def _red_side_points(rows: int, cols: int, red_cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    pts = set()
    for cell in red_cells:
        for side in _internal_sides(rows, cols, cell):
            pts.add(_endpoint(cell, side))
    return pts


def _assign_paths(
    base: np.ndarray,
    rows: int,
    cols: int,
    template_cell: tuple[int, int],
    connected_targets: list[tuple[int, int]],
    available_template_sides: list[str],
    red_cells: set[tuple[int, int]],
) -> list[tuple[tuple[int, int], str, list[tuple[int, int]], str]] | None:
    forbidden = _red_side_points(rows, cols, red_cells)
    results: list[tuple[tuple[int, int], str, list[tuple[int, int]], str]] = []

    def rec(
        idx: int,
        remaining_template_sides: tuple[str, ...],
        used_cells: set[tuple[int, int]],
    ) -> bool:
        if idx == len(connected_targets):
            return True

        target = connected_targets[idx]
        target_sides = _internal_sides(rows, cols, target)
        random.shuffle(target_sides)
        candidate_template_sides = list(remaining_template_sides)
        random.shuffle(candidate_template_sides)

        for tpl_side in candidate_template_sides:
            start = _endpoint(template_cell, tpl_side)
            for tgt_side in target_sides:
                if _neighbor(target, tgt_side) in red_cells:
                    continue
                end = _endpoint(target, tgt_side)
                route = _route(base, start, tpl_side, end, tgt_side, used_cells, forbidden - {start, end})
                if route is None:
                    continue
                path, shape = route
                results.append((target, tgt_side, path, shape))
                if rec(
                    idx + 1,
                    tuple(side for side in remaining_template_sides if side != tpl_side),
                    used_cells | set(path),
                ):
                    return True
                results.pop()
        return False

    if rec(0, tuple(available_template_sides), set()):
        return results
    return None


def _validate_semantics(grid: np.ndarray) -> None:
    analysis = _analyze_paths(grid)
    template = analysis["template"]
    targets = analysis["targets"]
    target_arrival_side = analysis["target_arrival_side"]
    path_shapes = analysis["path_shapes"]

    assert 2 <= len(target_arrival_side) <= 4
    assert len(target_arrival_side) < len(targets)
    assert len(set(target_arrival_side.values())) >= 2, "need multiple arrival directions"
    if len(path_shapes) >= 3:
        assert "straight" in path_shapes and "L" in path_shapes, "need both straight and L paths"

    out = main(grid)
    assert out.shape == grid.shape
    assert not np.array_equal(out, grid)
    assert np.array_equal(out == GREY, grid == GREY), "grey structure must be preserved"
    assert np.array_equal(out[template["bbox"][0] : template["bbox"][2] + 1, template["bbox"][1] : template["bbox"][3] + 1],
                          grid[template["bbox"][0] : template["bbox"][2] + 1, template["bbox"][1] : template["bbox"][3] + 1]), "template cell must stay unchanged"

    for target in targets:
        r0, c0, r1, c1 = target["bbox"]
        got = out[r0 + 1 : r1, c0 + 1 : c1]
        side = target_arrival_side.get(target["id"])
        if side is None:
            want = np.zeros((INNER, INNER), dtype=int)
            want[INNER // 2, INNER // 2] = RED
        else:
            want = np.rot90(template["patch"], ROTATION_FOR_SIDE[side])
        assert np.array_equal(got, want)


def generate_input() -> np.ndarray:
    for _ in range(600):
        rows = random.choice([3, 4])
        cols = random.choice([3, 4])
        total_cells = rows * cols
        num_targets = random.randint(3, min(5, total_cells - 1))
        connected_count = random.randint(2, min(4, num_targets - 1))

        H = rows * BOX + (rows - 1) * GAP
        W = cols * BOX + (cols - 1) * GAP
        if max(H, W) > 30:
            continue

        cells = [(r, c) for r in range(rows) for c in range(cols)]
        template_candidates = [cell for cell in cells if len(_internal_sides(rows, cols, cell)) >= connected_count]
        if not template_candidates:
            continue
        template_cell = random.choice(template_candidates)

        others = [cell for cell in cells if cell != template_cell]
        random.shuffle(others)
        target_cells = others[:num_targets]
        red_cells = {template_cell, *target_cells}

        available_template_sides = [
            side
            for side in _internal_sides(rows, cols, template_cell)
            if _neighbor(template_cell, side) not in red_cells
        ]
        if len(available_template_sides) < connected_count:
            continue

        random.shuffle(target_cells)
        connected_targets = target_cells[:connected_count]

        base = np.zeros((H, W), dtype=int)
        for cell in cells:
            _draw_ring(base, cell)

        template_patch = _design_template_patch()
        tr, tc = _cell_center(template_cell)
        base[tr - 1 : tr + 2, tc - 1 : tc + 2] = template_patch

        for target in target_cells:
            rr, cc = _cell_center(target)
            base[rr, cc] = RED

        assignment = _assign_paths(base, rows, cols, template_cell, connected_targets, available_template_sides, red_cells)
        if assignment is None:
            continue

        for _target, _side, path, _shape in assignment:
            for r, c in path:
                base[r, c] = GREY

        try:
            _validate_semantics(base)
        except AssertionError:
            continue
        return base

    raise RuntimeError("failed to generate a valid puzzle instance")

