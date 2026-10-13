from __future__ import annotations

import itertools
import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

GRID_H_RANGE = (11, 18)
GRID_W_RANGE = (11, 18)
AXIS_MARGIN = 3
OBJECT_DEPTH_RANGE = (2, 4)
OBJECT_SPAN_RANGE = (5, 7)
OBJECT_AREA_RANGE = (2, 6)
OBJECT_COUNT_RANGE = (2, 4)
MASK_DENSITY_RANGE = (0.32, 0.55)
GENERATION_ATTEMPTS = 400
LOCAL_MASK_ATTEMPTS = 250
MARKER_PLAN_ATTEMPTS = 120


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[counts.argmax()])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components_from_mask(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        deltas = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]

    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            q = deque([(r, c)])
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in deltas:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _color_components(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    return _components_from_mask(np.asarray(grid, dtype=int) == int(color), connectivity=4)


def _find_hidden_axis(grid: np.ndarray, bg: int) -> tuple[str, int, tuple[int, int, int, int]]:
    box = _bbox(grid != bg)
    assert box is not None, "expected non-background foreground"
    r0, c0, r1, c1 = box
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]

    empty_rows = [r0 + rr for rr in range(1, sub.shape[0] - 1) if np.all(sub[rr, :] == bg)]
    empty_cols = [c0 + cc for cc in range(1, sub.shape[1] - 1) if np.all(sub[:, cc] == bg)]
    separators = [("row", idx) for idx in empty_rows] + [("col", idx) for idx in empty_cols]
    assert len(separators) == 1, "foreground bbox must contain exactly one empty interior line"
    axis_kind, axis_index = separators[0]
    return axis_kind, int(axis_index), box


def _reflect_coords(coords: list[tuple[int, int]], axis_kind: str, axis_index: int) -> set[tuple[int, int]]:
    if axis_kind == "row":
        return {(2 * axis_index - r, c) for r, c in coords}
    return {(r, 2 * axis_index - c) for r, c in coords}


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    axis_kind, axis_index, box = _find_hidden_axis(g, bg)

    grey_components = [sorted(comp) for comp in _color_components(g, GREY)]
    assert grey_components, "expected grey source objects"
    assert all(len(comp) > 1 for comp in grey_components), "grey objects must be multi-pixel"

    markers: dict[tuple[int, int], int] = {}
    for color in sorted(set(map(int, np.unique(g))) - {bg, GREY}):
        for comp in _color_components(g, color):
            assert len(comp) == 1, "colored markers must be singleton pixels"
            markers[comp[0]] = int(color)
    assert markers, "expected visible singleton markers"

    grey_side_signs = set()
    marker_side_signs = set()
    if axis_kind == "row":
        for comp in grey_components:
            signs = {int(np.sign(r - axis_index)) for r, _c in comp}
            assert len(signs) == 1 and 0 not in signs
            grey_side_signs |= signs
        for r, _c in markers:
            sign = int(np.sign(r - axis_index))
            assert sign != 0
            marker_side_signs.add(sign)
    else:
        for comp in grey_components:
            signs = {int(np.sign(c - axis_index)) for _r, c in comp}
            assert len(signs) == 1 and 0 not in signs
            grey_side_signs |= signs
        for _r, c in markers:
            sign = int(np.sign(c - axis_index))
            assert sign != 0
            marker_side_signs.add(sign)
    assert len(grey_side_signs) == 1 and len(marker_side_signs) == 1
    assert next(iter(grey_side_signs)) == -next(iter(marker_side_signs))

    return {
        "grid": g,
        "bg": bg,
        "axis_kind": axis_kind,
        "axis_index": axis_index,
        "bbox": box,
        "grey_components": grey_components,
        "markers": markers,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold each grey object across the hidden empty axis and recolor it from any overlapped singleton marker.

    Input structure:
    - A black background contains several multi-pixel grey objects on one side of a hidden fold axis and colored singleton markers on the other side.
    - The overall foreground bounding box contains exactly one completely empty interior row or column; that invisible line is the fold axis.
    - Each colored marker is a single pixel, while each grey object is a separate multi-pixel component.

    Rule:
    - Find the unique all-background row or column strictly inside the foreground bounding box.
    - For each grey object, compute its mirror image across that hidden axis without drawing the mirror itself.
    - If the mirrored footprint covers a colored singleton marker, recolor the entire original grey object to that marker's color; otherwise keep it grey.
    - Erase all singleton markers from the output.

    Key insight:
    - The decisive evidence is not any visible contact: each object's color is determined by whether its unseen reflection lands on a marker after the empty separator line is treated as the fold axis.
    """
    info = _analyze(grid)
    g = info["grid"]
    bg = info["bg"]
    axis_kind = info["axis_kind"]
    axis_index = info["axis_index"]
    markers = info["markers"]

    out = np.full_like(g, bg)
    for comp in info["grey_components"]:
        mirrored = _reflect_coords(comp, axis_kind, axis_index)
        for r, c in mirrored:
            assert 0 <= r < g.shape[0] and 0 <= c < g.shape[1], "mirrored objects must stay in bounds"
        hits = [markers[cell] for cell in mirrored if cell in markers]
        assert len(hits) <= 1, "each mirrored grey object may cover at most one marker"
        color = hits[0] if hits else GREY
        for r, c in comp:
            out[r, c] = color
    return out


def _components_well_separated(comps: list[list[tuple[int, int]]]) -> bool:
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            for r1, c1 in comps[i]:
                for r2, c2 in comps[j]:
                    if max(abs(r1 - r2), abs(c1 - c2)) <= 1:
                        return False
    return True


def _comp_bbox(comp: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in comp]
    cols = [c for _r, c in comp]
    return min(rows), min(cols), max(rows), max(cols)


def _generate_object_mask(depth: int, span: int) -> list[list[tuple[int, int]]] | None:
    cells = [(r, c) for r in range(depth) for c in range(span)]
    area = depth * span
    min_fill = max(int(np.ceil(area * MASK_DENSITY_RANGE[0])), depth + span - 3)
    max_fill = min(int(np.floor(area * MASK_DENSITY_RANGE[1])), area - 1)
    if min_fill > max_fill:
        return None

    for _ in range(LOCAL_MASK_ATTEMPTS):
        count = random.randint(min_fill, max_fill)
        chosen = set(random.sample(cells, count))
        mask = np.zeros((depth, span), dtype=bool)
        for cell in chosen:
            mask[cell] = True

        if not np.all(mask.any(axis=0)) or not np.all(mask.any(axis=1)):
            continue

        comps = _components_from_mask(mask, connectivity=4)
        if not (OBJECT_COUNT_RANGE[0] <= len(comps) <= OBJECT_COUNT_RANGE[1]):
            continue
        areas = [len(comp) for comp in comps]
        if any(not (OBJECT_AREA_RANGE[0] <= area <= OBJECT_AREA_RANGE[1]) for area in areas):
            continue
        if not any(area >= 3 for area in areas):
            continue
        if not any((_comp_bbox(comp)[2] - _comp_bbox(comp)[0] >= 1) and (_comp_bbox(comp)[3] - _comp_bbox(comp)[1] >= 1) for comp in comps):
            continue
        if not _components_well_separated(comps):
            continue
        return [sorted(comp) for comp in comps]
    return None


def _well_separated_singletons(cells: list[tuple[int, int]]) -> bool:
    for i in range(len(cells)):
        for j in range(i + 1, len(cells)):
            if max(abs(cells[i][0] - cells[j][0]), abs(cells[i][1] - cells[j][1])) <= 1:
                return False
    return True


def _build_marker_plan(
    mirrors: list[set[tuple[int, int]]],
    row_low: int,
    row_high: int,
    col_low: int,
    col_high: int,
) -> tuple[dict[int, tuple[int, int]], list[tuple[int, int]]] | None:
    mirror_union = set().union(*mirrors)
    all_safe = [
        (r, c)
        for r in range(row_low, row_high + 1)
        for c in range(col_low, col_high + 1)
        if (r, c) not in mirror_union
    ]

    comp_indices = list(range(len(mirrors)))
    mark_choices = list(range(1, len(mirrors)))
    random.shuffle(mark_choices)
    for marked_count in mark_choices:
        for _ in range(MARKER_PLAN_ATTEMPTS):
            marked = sorted(random.sample(comp_indices, marked_count))
            option_lists = [list(mirrors[idx]) for idx in marked]
            for options in option_lists:
                random.shuffle(options)

            products = itertools.product(*option_lists)
            candidates = []
            for chosen in products:
                chosen_cells = list(chosen)
                if len(set(chosen_cells)) != len(chosen_cells):
                    continue
                if not _well_separated_singletons(chosen_cells):
                    continue
                used_rows = {r for r, _c in chosen_cells}
                row_min = min(used_rows)
                row_max = max(used_rows)
                missing_rows = [r for r in range(row_min, row_max + 1) if r not in used_rows]

                fillers: list[tuple[int, int]] = []
                taken = list(chosen_cells)
                ok = True
                for row in missing_rows:
                    row_candidates = [
                        cell
                        for cell in all_safe
                        if cell[0] == row and cell not in taken and _well_separated_singletons(taken + [cell])
                    ]
                    if not row_candidates:
                        ok = False
                        break
                    pick = random.choice(row_candidates)
                    fillers.append(pick)
                    taken.append(pick)
                if ok:
                    score = (row_max - row_min + 1, len(fillers), random.random())
                    candidates.append((score, dict(zip(marked, chosen_cells)), fillers, taken))

            if not candidates:
                continue

            candidates.sort(key=lambda item: item[0])
            _score, used_map, fillers, taken = candidates[0]

            extras: list[tuple[int, int]] = []
            extra_candidates = [cell for cell in all_safe if cell not in taken]
            random.shuffle(extra_candidates)
            extra_target = random.randint(1, 2)
            for cell in extra_candidates:
                if _well_separated_singletons(taken + extras + [cell]):
                    extras.append(cell)
                    if len(extras) >= extra_target:
                        break

            all_markers = list(used_map.values()) + fillers + extras
            if extras and len(all_markers) <= len(MARKER_COLORS):
                return used_map, fillers + extras
    return None


def _make_row_case() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)

    axis = random.randint(AXIS_MARGIN, h - AXIS_MARGIN - 1)
    if h % 2 == 1 and axis == h // 2 and random.random() < 0.75:
        axis = random.choice([idx for idx in range(AXIS_MARGIN, h - AXIS_MARGIN) if idx != h // 2])

    objects_on_top = random.choice([True, False])
    near_space = axis if objects_on_top else h - axis - 1
    far_space = h - axis - 1 if objects_on_top else axis
    max_depth = min(OBJECT_DEPTH_RANGE[1], near_space, far_space)
    if max_depth < OBJECT_DEPTH_RANGE[0]:
        return None
    depth = random.randint(OBJECT_DEPTH_RANGE[0], max_depth)

    span_max = min(OBJECT_SPAN_RANGE[1], w - 2)
    if span_max < OBJECT_SPAN_RANGE[0]:
        return None
    span = random.randint(OBJECT_SPAN_RANGE[0], span_max)
    left = random.randint(1, w - span - 1)

    local_components = _generate_object_mask(depth, span)
    if local_components is None:
        return None

    row_offset = axis - depth if objects_on_top else axis + 1
    global_components: list[list[tuple[int, int]]] = []
    for comp in local_components:
        global_components.append(sorted((row_offset + r, left + c) for r, c in comp))

    mirrors = [_reflect_coords(comp, "row", axis) for comp in global_components]
    if objects_on_top:
        marker_row_low, marker_row_high = axis + 1, h - 1
    else:
        marker_row_low, marker_row_high = 0, axis - 1
    plan = _build_marker_plan(mirrors, marker_row_low, marker_row_high, left, left + span - 1)
    if plan is None:
        return None
    used_map, distractors = plan

    marker_positions = list(used_map.values()) + distractors
    marker_colors = random.sample(MARKER_COLORS, len(marker_positions))
    color_by_marker = dict(zip(marker_positions, marker_colors))

    grid = np.full((h, w), BG, dtype=int)
    for comp in global_components:
        for r, c in comp:
            grid[r, c] = GREY
    for marker in marker_positions:
        grid[marker] = color_by_marker[marker]

    return grid


def _validate_generated_case(grid: np.ndarray) -> None:
    info = _analyze(grid)
    overlaps = []
    markers = info["markers"]
    marker_hits = {cell: 0 for cell in markers}
    for comp in info["grey_components"]:
        mirrored = _reflect_coords(comp, info["axis_kind"], info["axis_index"])
        hits = [cell for cell in mirrored if cell in markers]
        assert len(hits) <= 1
        overlaps.append(len(hits))
        for cell in hits:
            marker_hits[cell] += 1

    assert any(v == 1 for v in overlaps), "need at least one recolored object"
    assert any(v == 0 for v in overlaps), "need at least one untouched grey object"
    assert all(count <= 1 for count in marker_hits.values()), "marker ownership must be unambiguous"
    assert any(count == 0 for count in marker_hits.values()), "include distractor markers to block shortcut rules"

    out = main(grid)
    assert out.shape == grid.shape
    assert not np.any(out == BG) or out.size > 0
    assert not np.any((grid != BG) & (grid != GREY) & (out != BG)), "all markers must be erased"


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _make_row_case()
        if grid is None:
            continue
        if random.random() < 0.5:
            grid = grid.T.copy()
        try:
            _validate_generated_case(grid)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid hidden-axis mirror recolor puzzle")
