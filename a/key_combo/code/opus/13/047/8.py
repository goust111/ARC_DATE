from __future__ import annotations

from fractions import Fraction
from itertools import permutations
import random

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
MARKER_COLORS = (1, 2, 3, 6, 7, 8, 9)
OFFSETS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
OFFSETS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


def _neighbors(r: int, c: int, connectivity: int) -> list[tuple[int, int]]:
    return OFFSETS4 if connectivity == 4 else OFFSETS8


def _label_mask(mask: np.ndarray, connectivity: int) -> tuple[np.ndarray, int]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    n = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            n += 1
            stack = [(r, c)]
            labels[r, c] = n
            while stack:
                rr, cc = stack.pop()
                for dr, dc in _neighbors(rr, cc, connectivity):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = n
                        stack.append((nr, nc))
    return labels, n


def _components_mask(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    labels, n = _label_mask(mask, connectivity)
    return [np.argwhere(labels == idx) for idx in range(1, n + 1)]


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_union(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _relative_marker(pt: tuple[int, int], box: tuple[int, int, int, int]) -> tuple[Fraction, Fraction]:
    r, c = pt
    r0, c0, r1, c1 = box
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    assert h > 1 and w > 1
    return Fraction(r - r0, h), Fraction(c - c0, w)


def _relative_center(box: tuple[int, int, int, int], ref_box: tuple[int, int, int, int]) -> tuple[Fraction, Fraction]:
    r0, c0, r1, c1 = box
    rr0, cc0, rr1, cc1 = ref_box
    h = rr1 - rr0 + 1
    w = cc1 - cc0 + 1
    assert h > 1 and w > 1
    return (
        Fraction(r0 + r1 - 2 * rr0, 2 * h),
        Fraction(c0 + c1 - 2 * cc0, 2 * w),
    )


def _dist_l1(a: tuple[Fraction, Fraction], b: tuple[Fraction, Fraction]) -> Fraction:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _unique_assignment(
    src_pos: list[tuple[Fraction, Fraction]],
    dst_pos: list[tuple[Fraction, Fraction]],
) -> tuple[int, ...]:
    assert len(src_pos) == len(dst_pos) and src_pos
    scored: list[tuple[Fraction, tuple[int, ...]]] = []
    for perm in permutations(range(len(dst_pos))):
        total = sum(_dist_l1(src_pos[i], dst_pos[perm[i]]) for i in range(len(src_pos)))
        scored.append((total, perm))
    best_score = min(score for score, _ in scored)
    best = [perm for score, perm in scored if score == best_score]
    assert len(best) == 1, "matching must have a unique minimum total L1 distance"
    return best[0]


def _legend_box(g: np.ndarray) -> tuple[int, int, int, int]:
    comps = _components_mask(g == GREY, connectivity=4)
    assert comps, "expected grey components"
    sizes = [coords.shape[0] for coords in comps]
    max_size = max(sizes)
    biggest = [coords for coords in comps if coords.shape[0] == max_size]
    assert len(biggest) == 1, "legend grey component must be uniquely largest"
    return _bbox_coords(biggest[0])


def _detect_axis_and_control(
    g: np.ndarray, coords: np.ndarray
) -> tuple[str, int, tuple[int, int]]:
    yellow = [(int(r), int(c)) for r, c in coords if int(g[r, c]) == YELLOW]
    assert len(yellow) >= 3, "each object needs an axis and one control pixel"

    candidates: list[tuple[str, int, int]] = []
    row_map: dict[int, list[int]] = {}
    col_map: dict[int, list[int]] = {}
    for r, c in yellow:
        row_map.setdefault(r, []).append(c)
        col_map.setdefault(c, []).append(r)

    for row, cols in row_map.items():
        cols = sorted(cols)
        if len(cols) >= 2 and cols == list(range(cols[0], cols[-1] + 1)):
            candidates.append(("h", row, len(cols)))
    for col, rows in col_map.items():
        rows = sorted(rows)
        if len(rows) >= 2 and rows == list(range(rows[0], rows[-1] + 1)):
            candidates.append(("v", col, len(rows)))

    assert candidates, "expected one contiguous yellow axis"
    longest = max(length for _orient, _axis, length in candidates)
    best = [cand for cand in candidates if cand[2] == longest]
    assert len(best) == 1, "axis must be uniquely determined"
    orient, axis, _ = best[0]

    if orient == "h":
        controls = [(r, c) for r, c in yellow if r != axis]
    else:
        controls = [(r, c) for r, c in yellow if c != axis]
    assert len(controls) == 1, "expected exactly one off-axis yellow control pixel"
    return orient, int(axis), controls[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match legend markers to incomplete grey objects, mirror each trusted half, and recolor the completed object.

    Input structure:
    - The background is black.
    - One solid grey rectangle acts as a legend and contains 3-4 distinct one-pixel colored markers.
    - Outside the legend, each half-finished object has one 8-connected grey component plus yellow pixels forming a contiguous horizontal or vertical axis and one off-axis control pixel.

    Rule:
    - Identify the largest grey component as the legend body and collect its internal colored markers.
    - Identify the legend-external grey components as the half-finished objects, compute each grey bbox center, and normalize those centers inside the union bbox of all grey objects.
    - Normalize each legend marker position inside the legend bbox, then find the unique minimum-total-Manhattan one-to-one matching between grey objects and markers.
    - For each matched object, use the side containing the off-axis yellow control pixel as the trusted side, reflect every non-black source-side pixel across the yellow axis, and then recolor every grey or yellow pixel of the completed object to the matched marker color.

    Key insight:
    - The positional match is decided from the incomplete grey geometry before completion, but the visible output shows only the completed recolored objects, so the solver must separate matching time from painting time.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    legend_box = _legend_box(g)
    lr0, lc0, lr1, lc1 = legend_box
    legend_rect = np.zeros(g.shape, dtype=bool)
    legend_rect[lr0 : lr1 + 1, lc0 : lc1 + 1] = True

    marker_cells = [(int(r), int(c)) for r, c in np.argwhere(legend_rect & (g != BLACK) & (g != GREY))]
    marker_colors = [int(g[r, c]) for r, c in marker_cells]
    assert 3 <= len(marker_cells) <= 4
    assert len(set(marker_colors)) == len(marker_colors)

    grey_objects = _components_mask((g == GREY) & (~legend_rect), connectivity=8)
    assert len(grey_objects) == len(marker_cells), "marker/object counts must match"

    outside_nonblack = (g != BLACK) & (~legend_rect)
    labels, _ = _label_mask(outside_nonblack, connectivity=8)

    object_labels: list[int] = []
    full_objects: list[np.ndarray] = []
    for grey_coords in grey_objects:
        labs = {int(labels[r, c]) for r, c in grey_coords}
        assert len(labs) == 1 and next(iter(labs)) != 0
        label = next(iter(labs))
        object_labels.append(label)
        full_objects.append(np.argwhere(labels == label))
    assert len(set(object_labels)) == len(object_labels)

    grey_boxes = [_bbox_coords(coords) for coords in grey_objects]
    grey_union = _bbox_union(grey_boxes)
    object_pos = [_relative_center(box, grey_union) for box in grey_boxes]
    marker_pos = [_relative_marker(cell, legend_box) for cell in marker_cells]
    assignment = _unique_assignment(object_pos, marker_pos)

    out = g.copy()
    for full_coords, marker_idx in zip(full_objects, assignment):
        orient, axis, control = _detect_axis_and_control(g, full_coords)
        completed = {tuple(map(int, rc)) for rc in full_coords}

        if orient == "h":
            source_sign = -1 if control[0] < axis else 1
            for r, c in completed.copy():
                if r == axis:
                    continue
                if (-1 if r < axis else 1) != source_sign:
                    continue
                dst = (2 * axis - r, c)
                assert 0 <= dst[0] < g.shape[0]
                out[dst] = g[r, c]
                completed.add(dst)
        else:
            source_sign = -1 if control[1] < axis else 1
            for r, c in completed.copy():
                if c == axis:
                    continue
                if (-1 if c < axis else 1) != source_sign:
                    continue
                dst = (r, 2 * axis - c)
                assert 0 <= dst[1] < g.shape[1]
                out[dst] = g[r, c]
                completed.add(dst)

        target_color = marker_colors[marker_idx]
        for r, c in completed:
            if out[r, c] in (GREY, YELLOW):
                out[r, c] = target_color

    return out


def _connected_mask(h: int, w: int, target: int, anchor: tuple[int, int]) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[anchor] = True
    while int(mask.sum()) < target:
        frontier: set[tuple[int, int]] = set()
        for r, c in np.argwhere(mask):
            for dr, dc in OFFSETS8:
                nr, nc = int(r) + dr, int(c) + dc
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.add((nr, nc))
        pick = random.choice(tuple(frontier))
        mask[pick] = True
    return mask


def _make_horizontal_object(source_top: bool) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(80):
        depth = random.randint(2, 4)
        width = random.randint(5, 8)
        axis_row = depth
        axis_len = random.randint(3, width)
        axis_c0 = random.randint(0, width - axis_len)
        axis_c1 = axis_c0 + axis_len - 1
        anchor_col = random.randint(axis_c0, axis_c1)
        grey_count = random.randint(4, min(8, depth * width - 1))

        source_mask = _connected_mask(depth, width, grey_count, (depth - 1, anchor_col))
        ctrl_offset = random.randint(1, min(2, depth))
        ctrl_local_row = depth - ctrl_offset
        ctrl_candidates = [
            c
            for c in range(axis_c0, axis_c1 + 1)
            if not source_mask[ctrl_local_row, c] and (ctrl_offset > 1 or c != anchor_col)
        ]
        if not ctrl_candidates:
            continue
        ctrl_col = random.choice(ctrl_candidates)

        inp = np.zeros((2 * depth + 1, width), dtype=int)
        if source_top:
            inp[:depth, :][source_mask] = GREY
            inp[axis_row, axis_c0 : axis_c1 + 1] = YELLOW
            inp[axis_row - ctrl_offset, ctrl_col] = YELLOW
        else:
            inp[axis_row + 1 : axis_row + 1 + depth, :][source_mask] = GREY
            inp[axis_row, axis_c0 : axis_c1 + 1] = YELLOW
            inp[axis_row + ctrl_offset, ctrl_col] = YELLOW

        if len(_components_mask(inp == GREY, connectivity=8)) != 1:
            continue
        if len(_components_mask(inp != BLACK, connectivity=8)) != 1:
            continue

        completed = inp.copy()
        coords = np.argwhere(inp != BLACK)
        for r, c in coords:
            rr = int(r)
            cc = int(c)
            if rr == axis_row:
                continue
            if source_top and rr < axis_row:
                completed[2 * axis_row - rr, cc] = inp[rr, cc]
            if (not source_top) and rr > axis_row:
                completed[2 * axis_row - rr, cc] = inp[rr, cc]
        return inp, completed
    raise AssertionError("failed to build horizontal object")


def _make_vertical_object(source_left: bool) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(80):
        height = random.randint(5, 8)
        depth = random.randint(2, 4)
        axis_col = depth
        axis_len = random.randint(3, height)
        axis_r0 = random.randint(0, height - axis_len)
        axis_r1 = axis_r0 + axis_len - 1
        anchor_row = random.randint(axis_r0, axis_r1)
        grey_count = random.randint(4, min(8, height * depth - 1))

        source_mask = _connected_mask(height, depth, grey_count, (anchor_row, depth - 1))
        ctrl_offset = random.randint(1, min(2, depth))
        ctrl_local_col = depth - ctrl_offset
        ctrl_candidates = [
            r
            for r in range(axis_r0, axis_r1 + 1)
            if not source_mask[r, ctrl_local_col] and (ctrl_offset > 1 or r != anchor_row)
        ]
        if not ctrl_candidates:
            continue
        ctrl_row = random.choice(ctrl_candidates)

        inp = np.zeros((height, 2 * depth + 1), dtype=int)
        if source_left:
            inp[:, :depth][source_mask] = GREY
            inp[axis_r0 : axis_r1 + 1, axis_col] = YELLOW
            inp[ctrl_row, axis_col - ctrl_offset] = YELLOW
        else:
            inp[:, axis_col + 1 : axis_col + 1 + depth][source_mask] = GREY
            inp[axis_r0 : axis_r1 + 1, axis_col] = YELLOW
            inp[ctrl_row, axis_col + ctrl_offset] = YELLOW

        if len(_components_mask(inp == GREY, connectivity=8)) != 1:
            continue
        if len(_components_mask(inp != BLACK, connectivity=8)) != 1:
            continue

        completed = inp.copy()
        coords = np.argwhere(inp != BLACK)
        for r, c in coords:
            rr = int(r)
            cc = int(c)
            if cc == axis_col:
                continue
            if source_left and cc < axis_col:
                completed[rr, 2 * axis_col - cc] = inp[rr, cc]
            if (not source_left) and cc > axis_col:
                completed[rr, 2 * axis_col - cc] = inp[rr, cc]
        return inp, completed
    raise AssertionError("failed to build vertical object")


def _dilate(mask: np.ndarray, padding: int) -> np.ndarray:
    out = mask.copy()
    cur = mask.copy()
    for _ in range(padding):
        nxt = cur.copy()
        for dr, dc in OFFSETS8:
            shifted = np.zeros_like(cur)
            r0 = max(0, dr)
            r1 = cur.shape[0] + min(0, dr)
            c0 = max(0, dc)
            c1 = cur.shape[1] + min(0, dc)
            shifted[r0:r1, c0:c1] = cur[r0 - dr : r1 - dr, c0 - dc : c1 - dc]
            nxt |= shifted
        cur = nxt
        out = nxt
    return out


def _random_place(zone_h: int, zone_w: int, patch: np.ndarray, occupied: np.ndarray, padding: int) -> tuple[int, int]:
    h, w = patch.shape
    candidates: list[tuple[int, int]] = []
    dilated = _dilate(occupied, padding)
    patch_mask = patch != BLACK
    for r0 in range(zone_h - h + 1):
        for c0 in range(zone_w - w + 1):
            if not np.any(dilated[r0 : r0 + h, c0 : c0 + w] & patch_mask):
                candidates.append((r0, c0))
    if not candidates:
        raise ValueError("no free placement")
    return random.choice(candidates)


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    m = patch != BLACK
    view = dst[r0 : r0 + h, c0 : c0 + w]
    view[m] = patch[m]


def _sample_marker_cells(
    legend_box: tuple[int, int, int, int], count: int
) -> list[tuple[int, int]] | None:
    r0, c0, r1, c1 = legend_box
    cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    random.shuffle(cells)
    chosen: list[tuple[int, int]] = []
    for cell in cells:
        if all(max(abs(cell[0] - rr), abs(cell[1] - cc)) >= 2 for rr, cc in chosen):
            chosen.append(cell)
            if len(chosen) == count:
                rows = {r for r, _ in chosen}
                cols = {c for _, c in chosen}
                if len(rows) >= 2 and len(cols) >= 2:
                    return chosen
                break
    return None


def _abs_boxes_from_objects(
    objects: list[dict],
) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
    grey_boxes: list[tuple[int, int, int, int]] = []
    full_boxes: list[tuple[int, int, int, int]] = []
    complete_boxes: list[tuple[int, int, int, int]] = []

    for obj in objects:
        r0, c0 = obj["pos"]
        grey_comps = _components_mask(obj["inp"] == GREY, connectivity=8)
        assert len(grey_comps) == 1
        grey_coords = grey_comps[0] + np.array([r0, c0])
        full_coords = np.argwhere(obj["inp"] != BLACK) + np.array([r0, c0])
        complete_coords = np.argwhere(obj["completed"] != BLACK) + np.array([r0, c0])
        grey_boxes.append(_bbox_coords(grey_coords))
        full_boxes.append(_bbox_coords(full_coords))
        complete_boxes.append(_bbox_coords(complete_coords))

    return grey_boxes, full_boxes, complete_boxes


def _assignment_for_boxes(
    boxes: list[tuple[int, int, int, int]],
    marker_cells: list[tuple[int, int]],
    legend_box: tuple[int, int, int, int],
) -> tuple[int, ...]:
    union = _bbox_union(boxes)
    assert union[0] < union[2] and union[1] < union[3]
    src = [_relative_center(box, union) for box in boxes]
    dst = [_relative_marker(cell, legend_box) for cell in marker_cells]
    return _unique_assignment(src, dst)


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    count = random.choice([3, 4])
    colors = random.sample(MARKER_COLORS, count)

    orientations = ["h", "v"] + [random.choice(["h", "v"]) for _ in range(count - 2)]
    random.shuffle(orientations)
    objects: list[dict] = []
    for orient in orientations:
        if orient == "h":
            source_top = bool(random.getrandbits(1))
            inp_patch, completed_patch = _make_horizontal_object(source_top)
        else:
            source_left = bool(random.getrandbits(1))
            inp_patch, completed_patch = _make_vertical_object(source_left)
        objects.append({"inp": inp_patch, "completed": completed_patch})

    h = random.randint(18, 28)
    w = random.randint(18, 28)
    legend_h = random.randint(5, 6)
    legend_w = random.randint(5, 6)
    side = random.choice(["left", "right", "top", "bottom"])
    gap = 3
    border = 1

    if side == "left":
        legend_r0 = random.randint(border, h - legend_h - border)
        legend_c0 = border
        zone = (border, legend_c0 + legend_w + gap, h - border, w - border)
    elif side == "right":
        legend_r0 = random.randint(border, h - legend_h - border)
        legend_c0 = w - border - legend_w
        zone = (border, border, h - border, legend_c0 - gap)
    elif side == "top":
        legend_r0 = border
        legend_c0 = random.randint(border, w - legend_w - border)
        zone = (legend_r0 + legend_h + gap, border, h - border, w - border)
    else:
        legend_r0 = h - border - legend_h
        legend_c0 = random.randint(border, w - legend_w - border)
        zone = (border, border, legend_r0 - gap, w - border)

    zr0, zc0, zr1, zc1 = zone
    zone_h = zr1 - zr0
    zone_w = zc1 - zc0
    assert zone_h >= 8 and zone_w >= 8

    occupied = np.zeros((zone_h, zone_w), dtype=bool)
    for obj in sorted(objects, key=lambda d: d["completed"].shape[0] * d["completed"].shape[1], reverse=True):
        r0, c0 = _random_place(zone_h, zone_w, obj["completed"], occupied, padding=1)
        obj["pos"] = (zr0 + r0, zc0 + c0)
        hh, ww = obj["completed"].shape
        occupied[r0 : r0 + hh, c0 : c0 + ww] |= obj["completed"] != BLACK

    legend_box = (legend_r0, legend_c0, legend_r0 + legend_h - 1, legend_c0 + legend_w - 1)
    marker_cells = None
    grey_assignment = None
    full_assignment = None
    complete_assignment = None
    legend_rect = np.zeros((h, w), dtype=bool)
    legend_rect[legend_box[0] : legend_box[2] + 1, legend_box[1] : legend_box[3] + 1] = True

    inp = np.zeros((h, w), dtype=int)
    inp[legend_box[0] : legend_box[2] + 1, legend_box[1] : legend_box[3] + 1] = GREY
    for obj in objects:
        _paste(inp, obj["inp"], *obj["pos"])

    grey_boxes, full_boxes, complete_boxes = _abs_boxes_from_objects(objects)

    for _ in range(200):
        candidate = _sample_marker_cells(legend_box, count)
        if candidate is None:
            continue
        try:
            ga = _assignment_for_boxes(grey_boxes, candidate, legend_box)
            fa = _assignment_for_boxes(full_boxes, candidate, legend_box)
            ca = _assignment_for_boxes(complete_boxes, candidate, legend_box)
        except AssertionError:
            continue
        if ga != fa and ga != ca:
            marker_cells = candidate
            grey_assignment, full_assignment, complete_assignment = ga, fa, ca
            break

    assert marker_cells is not None
    for marker_idx, cell in enumerate(marker_cells):
        inp[cell] = colors[marker_idx]

    out = np.zeros((h, w), dtype=int)
    out[legend_box[0] : legend_box[2] + 1, legend_box[1] : legend_box[3] + 1] = GREY
    for marker_idx, cell in enumerate(marker_cells):
        out[cell] = colors[marker_idx]
    for obj, marker_idx in zip(objects, grey_assignment):
        recolored = np.where(obj["completed"] != BLACK, colors[marker_idx], BLACK)
        _paste(out, recolored, *obj["pos"])

    solved = main(inp)
    assert np.array_equal(solved, out)
    assert grey_assignment != full_assignment
    assert grey_assignment != complete_assignment
    assert any(gb != fb for gb, fb in zip(sorted(grey_boxes), sorted(full_boxes)))
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(400):
        try:
            inp, _out = _generate_case()
            return inp
        except (AssertionError, ValueError):
            continue
    raise RuntimeError("failed to generate a valid puzzle instance")
