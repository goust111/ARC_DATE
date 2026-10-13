from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
USABLE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            parts.append(coords)
    return parts


def _coords_bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _perimeter_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = box
    out[r0, c0 : c1 + 1] = True
    out[r1, c0 : c1 + 1] = True
    out[r0 : r1 + 1, c0] = True
    out[r0 : r1 + 1, c1] = True
    return out


def _find_frame_and_inner_colors(
    grid: np.ndarray,
) -> tuple[int, int, tuple[int, int, int, int]]:
    colors = sorted(int(x) for x in np.unique(grid) if int(x) not in (BLACK, GREY))
    assert len(colors) == 2, "expected exactly two non-black non-grey colors"

    candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    for color in colors:
        mask = grid == color
        box = _bbox(mask)
        if box is None:
            continue
        r0, c0, r1, c1 = box
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        if np.array_equal(mask, _perimeter_mask(grid.shape, box)):
            candidates.append((color, box))

    assert len(candidates) == 1, "expected one full rectangular frame color"
    frame_color, box = candidates[0]
    inner_color = colors[0] if colors[1] == frame_color else colors[1]
    return frame_color, inner_color, box


def _extract_axes(
    grid: np.ndarray,
    frame_box: tuple[int, int, int, int],
    frame_color: int,
    inner_color: int,
) -> tuple[list[int], list[int], list[tuple[int, int]]]:
    r0, c0, r1, c1 = frame_box
    outside = np.argwhere(grid == inner_color)
    vertical_axes: list[int] = []
    horizontal_axes: list[int] = []
    leaves: list[tuple[int, int]] = []

    for rr_raw, cc_raw in outside:
        rr = int(rr_raw)
        cc = int(cc_raw)
        if r0 <= rr <= r1 and c0 <= cc <= c1:
            continue
        if rr == r0 - 1 and c0 + 1 <= cc <= c1 - 1:
            assert grid[r0, cc] == frame_color
            vertical_axes.append(cc)
            leaves.append((rr, cc))
        elif rr == r1 + 1 and c0 + 1 <= cc <= c1 - 1:
            assert grid[r1, cc] == frame_color
            vertical_axes.append(cc)
            leaves.append((rr, cc))
        elif cc == c0 - 1 and r0 + 1 <= rr <= r1 - 1:
            assert grid[rr, c0] == frame_color
            horizontal_axes.append(rr)
            leaves.append((rr, cc))
        elif cc == c1 + 1 and r0 + 1 <= rr <= r1 - 1:
            assert grid[rr, c1] == frame_color
            horizontal_axes.append(rr)
            leaves.append((rr, cc))
        else:
            raise AssertionError("inner-color pixels outside the frame must be leaf markers")

    assert 2 <= len(leaves) <= 4, "expected 2 to 4 leaves"
    assert len(set(leaves)) == len(leaves), "leaf cells must be unique"
    assert vertical_axes or horizontal_axes, "expected at least one axis"
    return vertical_axes, horizontal_axes, leaves


def _assign_axis_kind(
    region_box: tuple[int, int, int, int],
    vertical_axes: list[int],
    horizontal_axes: list[int],
) -> tuple[str, int]:
    r0, c0, r1, c1 = region_box
    center2_r = r0 + r1
    center2_c = c0 + c1
    distances: list[tuple[int, str, int]] = []
    for axis in vertical_axes:
        distances.append((abs(center2_c - 2 * axis), "vertical", axis))
    for axis in horizontal_axes:
        distances.append((abs(center2_r - 2 * axis), "horizontal", axis))
    distances.sort(key=lambda item: (item[0], item[1], item[2]))
    assert distances, "expected at least one candidate axis"
    if len(distances) >= 2:
        assert distances[0][0] != distances[1][0], "nearest axis must be unique"
    _dist, kind, axis = distances[0]
    return kind, axis


def _transform_crop(
    crop: np.ndarray,
    region_boxes: list[tuple[int, int, int, int]],
    assignments: list[tuple[str, int]],
) -> np.ndarray:
    out = crop.copy()
    for (lr0, lc0, lr1, lc1), (kind, _axis) in zip(region_boxes, assignments):
        rr0, cc0 = 1 + lr0, 1 + lc0
        rr1, cc1 = 1 + lr1, 1 + lc1
        patch = out[rr0 : rr1 + 1, cc0 : cc1 + 1].copy()
        assert not np.any(patch == GREY)
        if kind == "vertical":
            out[rr0 : rr1 + 1, cc0 : cc1 + 1] = np.fliplr(patch)
        else:
            out[rr0 : rr1 + 1, cc0 : cc1 + 1] = np.flipud(patch)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Mirror each wall-delimited subregion of the framed rectangle using the nearest leaf-marked axis.

    Input structure
    - A black background contains one rectangular foreground object whose outer ring is color A.
    - The rectangle interior contains color-B texture pixels, black holes, and one-cell-thick grey wall segments.
    - Two to four single-cell color-B leaves sit just outside the frame and mark candidate mirror axes.

    Rule
    - Detect the color-A frame bbox; a leaf on the top or bottom edge marks a vertical axis at its column, and a leaf on the left or right edge marks a horizontal axis at its row.
    - Use the grey walls to split the frame interior into 4-connected rectangular subregions.
    - For each subregion, choose the uniquely nearest leaf axis by centroid distance, then mirror that subregion inside its own bbox: vertical axis means left-right flip, horizontal axis means up-down flip.
    - Keep the frame and grey walls unchanged, erase leaves, and return only the original frame bbox crop.

    Key insight
    - The leaves do not emit rays; they only choose a flip orientation for each wall-bounded interior region, so both the wall partition and the nearest-axis comparison are necessary.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frame_color, inner_color, frame_box = _find_frame_and_inner_colors(g)
    r0, c0, r1, c1 = frame_box
    vertical_axes, horizontal_axes, _leaves = _extract_axes(g, frame_box, frame_color, inner_color)

    crop = _crop_bbox(g, frame_box)
    assert np.all(crop[0, :] == frame_color)
    assert np.all(crop[-1, :] == frame_color)
    assert np.all(crop[:, 0] == frame_color)
    assert np.all(crop[:, -1] == frame_color)

    interior = crop[1:-1, 1:-1]
    wall_mask = interior == GREY
    assert wall_mask.any(), "expected at least one internal grey wall"
    region_mask = ~wall_mask
    components = _components(region_mask)
    assert 2 <= len(components) <= 4, "expected 2 to 4 subregions"

    region_boxes: list[tuple[int, int, int, int]] = []
    assignments: list[tuple[str, int]] = []
    for coords in components:
        box = _coords_bbox(coords)
        br0, bc0, br1, bc1 = box
        assert len(coords) == (br1 - br0 + 1) * (bc1 - bc0 + 1), "subregions must be rectangular"
        global_box = (r0 + 1 + br0, c0 + 1 + bc0, r0 + 1 + br1, c0 + 1 + bc1)
        region_boxes.append(box)
        assignments.append(_assign_axis_kind(global_box, vertical_axes, horizontal_axes))

    return _transform_crop(crop, region_boxes, assignments)


def _sample_spaced_positions(lo: int, hi: int, count: int, min_gap: int = 2) -> list[int] | None:
    candidates = list(range(lo, hi + 1))
    for _ in range(50):
        random.shuffle(candidates)
        picked: list[int] = []
        for value in candidates:
            if all(abs(value - other) >= min_gap for other in picked):
                picked.append(value)
                if len(picked) == count:
                    return sorted(picked)
    return None


def _sample_partition(ih: int, iw: int) -> tuple[set[tuple[int, int]], list[tuple[int, int, int, int]]] | None:
    templates = [
        "v2",
        "h2",
        "t3_left",
        "t3_right",
        "t3_top",
        "t3_bottom",
        "cross4",
    ]
    template = random.choice(templates)
    if ih < 7 or iw < 7:
        return None

    vc = random.randint(3, iw - 4)
    hr = random.randint(3, ih - 4)
    walls: set[tuple[int, int]] = set()
    regions: list[tuple[int, int, int, int]]

    if template == "v2":
        walls = {(r, vc) for r in range(ih)}
        regions = [(0, 0, ih - 1, vc - 1), (0, vc + 1, ih - 1, iw - 1)]
    elif template == "h2":
        walls = {(hr, c) for c in range(iw)}
        regions = [(0, 0, hr - 1, iw - 1), (hr + 1, 0, ih - 1, iw - 1)]
    elif template == "t3_left":
        walls = {(r, vc) for r in range(ih)} | {(hr, c) for c in range(vc)}
        regions = [(0, 0, hr - 1, vc - 1), (hr + 1, 0, ih - 1, vc - 1), (0, vc + 1, ih - 1, iw - 1)]
    elif template == "t3_right":
        walls = {(r, vc) for r in range(ih)} | {(hr, c) for c in range(vc + 1, iw)}
        regions = [(0, 0, ih - 1, vc - 1), (0, vc + 1, hr - 1, iw - 1), (hr + 1, vc + 1, ih - 1, iw - 1)]
    elif template == "t3_top":
        walls = {(hr, c) for c in range(iw)} | {(r, vc) for r in range(hr)}
        regions = [(0, 0, hr - 1, vc - 1), (0, vc + 1, hr - 1, iw - 1), (hr + 1, 0, ih - 1, iw - 1)]
    elif template == "t3_bottom":
        walls = {(hr, c) for c in range(iw)} | {(r, vc) for r in range(hr + 1, ih)}
        regions = [(0, 0, hr - 1, iw - 1), (hr + 1, 0, ih - 1, vc - 1), (hr + 1, vc + 1, ih - 1, iw - 1)]
    else:
        walls = {(r, vc) for r in range(ih)} | {(hr, c) for c in range(iw)}
        regions = [
            (0, 0, hr - 1, vc - 1),
            (0, vc + 1, hr - 1, iw - 1),
            (hr + 1, 0, ih - 1, vc - 1),
            (hr + 1, vc + 1, ih - 1, iw - 1),
        ]

    for rr0, cc0, rr1, cc1 in regions:
        if rr1 - rr0 + 1 < 2 or cc1 - cc0 + 1 < 2:
            return None
    return walls, regions


def _sample_axes_for_regions(
    frame_box: tuple[int, int, int, int],
    region_boxes: list[tuple[int, int, int, int]],
) -> tuple[list[int], list[int], list[tuple[str, int]], list[tuple[int, int]]] | None:
    r0, c0, r1, c1 = frame_box
    v_count, h_count = random.choice([(1, 1), (2, 1), (1, 2), (2, 2)])
    v_axes = _sample_spaced_positions(c0 + 1, c1 - 1, v_count)
    h_axes = _sample_spaced_positions(r0 + 1, r1 - 1, h_count)
    if v_axes is None or h_axes is None:
        return None

    assignments: list[tuple[str, int]] = []
    for lr0, lc0, lr1, lc1 in region_boxes:
        global_box = (r0 + 1 + lr0, c0 + 1 + lc0, r0 + 1 + lr1, c0 + 1 + lc1)
        try:
            assignments.append(_assign_axis_kind(global_box, v_axes, h_axes))
        except AssertionError:
            return None

    used_kinds = {kind for kind, _axis in assignments}
    if used_kinds != {"vertical", "horizontal"}:
        return None

    leaves: list[tuple[int, int]] = []
    used_leaf_cells: set[tuple[int, int]] = set()
    for axis in v_axes:
        side = random.choice(["top", "bottom"])
        cell = (r0 - 1, axis) if side == "top" else (r1 + 1, axis)
        if cell in used_leaf_cells:
            return None
        used_leaf_cells.add(cell)
        leaves.append(cell)
    for axis in h_axes:
        side = random.choice(["left", "right"])
        cell = (axis, c0 - 1) if side == "left" else (axis, c1 + 1)
        if cell in used_leaf_cells:
            return None
        used_leaf_cells.add(cell)
        leaves.append(cell)

    return v_axes, h_axes, assignments, leaves


def _sample_region_pattern(height: int, width: int) -> np.ndarray | None:
    area = height * width
    max_fill = max(3, min(area - 1, int(round(area * 0.6))))
    for _ in range(200):
        count = random.randint(3, max_fill)
        coords = random.sample(range(area), count)
        mask = np.zeros((height, width), dtype=bool)
        for idx in coords:
            mask[idx // width, idx % width] = True
        if not mask.any() or mask.all():
            continue
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        if np.array_equal(mask, np.flipud(mask)):
            continue
        return mask
    return None


def _oracle_from_spec(grid: np.ndarray, spec: dict, force_kind: str | None = None) -> np.ndarray:
    crop = _crop_bbox(grid, spec["frame_box"])
    assignments = spec["assignments"]
    if force_kind is not None:
        assignments = [(force_kind, -1) for _ in assignments]
    return _transform_crop(crop, spec["region_boxes"], assignments)


def _validate_instance(grid: np.ndarray, spec: dict) -> bool:
    frame_box = spec["frame_box"]
    expected = _oracle_from_spec(grid, spec)
    if not np.array_equal(main(grid), expected):
        return False

    frame_color, inner_color, detected_box = _find_frame_and_inner_colors(grid)
    if frame_color != spec["frame_color"] or inner_color != spec["inner_color"] or detected_box != frame_box:
        return False

    non_black_box = _bbox(grid != BLACK)
    if non_black_box is None or non_black_box == frame_box:
        return False

    crop = _crop_bbox(grid, frame_box)
    if np.array_equal(crop, expected):
        return False

    if np.array_equal(_oracle_from_spec(grid, spec, force_kind="vertical"), expected):
        return False
    if np.array_equal(_oracle_from_spec(grid, spec, force_kind="horizontal"), expected):
        return False

    for (rr0, cc0, rr1, cc1), (kind, _axis) in zip(spec["region_boxes"], spec["assignments"]):
        patch = crop[1 + rr0 : 2 + rr1, 1 + cc0 : 2 + cc1]
        if int(np.count_nonzero(patch == spec["inner_color"])) < 3:
            return False
        transformed = np.fliplr(patch) if kind == "vertical" else np.flipud(patch)
        other = np.flipud(patch) if kind == "vertical" else np.fliplr(patch)
        if np.array_equal(patch, transformed):
            return False
        if np.array_equal(patch, other):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(14, 18)
    grid_w = random.randint(14, 18)
    rect_h = random.randint(10, min(13, grid_h - 4))
    rect_w = random.randint(10, min(13, grid_w - 4))
    r0 = random.randint(2, grid_h - rect_h - 2)
    c0 = random.randint(2, grid_w - rect_w - 2)
    frame_box = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)

    sampled = _sample_partition(rect_h - 2, rect_w - 2)
    if sampled is None:
        return None
    walls, region_boxes = sampled

    axes = _sample_axes_for_regions(frame_box, region_boxes)
    if axes is None:
        return None
    vertical_axes, horizontal_axes, assignments, leaves = axes

    frame_color, inner_color = random.sample(USABLE_COLORS, 2)
    grid = np.zeros((grid_h, grid_w), dtype=int)
    r1 = r0 + rect_h - 1
    c1 = c0 + rect_w - 1
    grid[r0 : r1 + 1, c0 : c1 + 1] = frame_color
    grid[r0 + 1 : r1, c0 + 1 : c1] = BLACK

    for wr, wc in walls:
        grid[r0 + 1 + wr, c0 + 1 + wc] = GREY

    for region_box, assignment in zip(region_boxes, assignments):
        rr0, cc0, rr1, cc1 = region_box
        pattern = _sample_region_pattern(rr1 - rr0 + 1, cc1 - cc0 + 1)
        if pattern is None:
            return None
        patch = grid[r0 + 1 + rr0 : r0 + 2 + rr1, c0 + 1 + cc0 : c0 + 2 + cc1]
        patch[pattern] = inner_color
        if assignment[0] == "vertical":
            if np.array_equal(pattern, np.fliplr(pattern)):
                return None
        else:
            if np.array_equal(pattern, np.flipud(pattern)):
                return None

    for lr, lc in leaves:
        grid[lr, lc] = inner_color

    spec = {
        "frame_box": frame_box,
        "frame_color": frame_color,
        "inner_color": inner_color,
        "vertical_axes": vertical_axes,
        "horizontal_axes": horizontal_axes,
        "region_boxes": region_boxes,
        "assignments": assignments,
        "leaves": leaves,
    }
    if not _validate_instance(grid, spec):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid leaf-axis mirror input")
