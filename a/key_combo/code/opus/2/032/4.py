from __future__ import annotations

from collections import Counter
from itertools import combinations
import random

import numpy as np


BG = 0
TEMPLATE_A_COLOR = 1
TEMPLATE_B_COLOR = 3
MARKER_COLOR = 5

TEMPLATE_H = 3
TEMPLATE_W = 4
TEMPLATE_AREA_RANGE = (4, 6)

MARKER_COUNT_RANGE = (5, 8)
GRID_H_LIMIT = 30
GRID_W_LIMIT = 30
GENERATION_RETRIES = 400
MARKER_RETRIES = 300

_NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_NEIGHBORS8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, bbox)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def bbox_h(bbox: tuple[int, int, int, int]) -> int:
    return int(bbox[2] - bbox[0] + 1)


def bbox_w(bbox: tuple[int, int, int, int]) -> int:
    return int(bbox[3] - bbox[1] + 1)


def components_of_color(
    grid: np.ndarray, color: int, *, connectivity: int
) -> list[dict[str, np.ndarray | tuple[int, int, int, int] | int]]:
    mask = np.asarray(grid == int(color), dtype=bool)
    visited = np.zeros_like(mask, dtype=bool)
    neighbors = _NEIGHBORS4 if int(connectivity) == 4 else _NEIGHBORS8
    comps: list[dict[str, np.ndarray | tuple[int, int, int, int] | int]] = []

    for r0, c0 in np.argwhere(mask):
        r = int(r0)
        c = int(c0)
        if visited[r, c]:
            continue
        stack = [(r, c)]
        visited[r, c] = True
        coords: list[tuple[int, int]] = []
        while stack:
            rr, cc = stack.pop()
            coords.append((int(rr), int(cc)))
            for dr, dc in neighbors:
                nr = int(rr + dr)
                nc = int(cc + dc)
                if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    stack.append((nr, nc))
        arr = np.array(coords, dtype=int)
        comps.append({"coords": arr, "bbox": bbox_coords(arr), "area": int(arr.shape[0])})

    return comps


def _mask_is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    seen = {tuple(map(int, coords[0]))}
    stack = [tuple(map(int, coords[0]))]
    coord_set = {tuple(map(int, rc)) for rc in coords}
    while stack:
        r, c = stack.pop()
        for dr, dc in _NEIGHBORS4:
            nxt = (int(r + dr), int(c + dc))
            if nxt in coord_set and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(coord_set)


def _build_template_catalog() -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    cell_count = TEMPLATE_H * TEMPLATE_W
    for area in range(TEMPLATE_AREA_RANGE[0], TEMPLATE_AREA_RANGE[1] + 1):
        for choice in combinations(range(cell_count), area):
            mask = np.zeros((TEMPLATE_H, TEMPLATE_W), dtype=bool)
            for idx in choice:
                mask[idx // TEMPLATE_W, idx % TEMPLATE_W] = True
            rows, cols = np.where(mask)
            if rows.min() != 0 or rows.max() != TEMPLATE_H - 1 or cols.min() != 0 or cols.max() != TEMPLATE_W - 1:
                continue
            if _mask_is_connected(mask):
                masks.append(mask)
    return masks


_TEMPLATE_MASKS = _build_template_catalog()


def _choose_template_pair() -> tuple[np.ndarray, np.ndarray]:
    for _ in range(500):
        mask_a = random.choice(_TEMPLATE_MASKS)
        mask_b = random.choice(_TEMPLATE_MASKS)
        if np.array_equal(mask_a, mask_b):
            continue
        overlap = mask_a & mask_b
        if not overlap.any():
            continue
        if not (mask_a & ~mask_b).any():
            continue
        if not (mask_b & ~mask_a).any():
            continue
        return mask_a.copy(), mask_b.copy()
    raise RuntimeError("failed to choose compatible templates")


def _paint_mask(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = int(color)
    return patch


def _marker_pixels(spec: dict[str, int | str]) -> list[tuple[int, int]]:
    row_index = int(spec["row_index"])
    anchor_col = int(spec["anchor_col"])
    length = int(spec["length"])
    base_r = int(row_index * TEMPLATE_H)
    base_c = int(anchor_col * TEMPLATE_W)
    if spec["orientation"] == "nw_se":
        return [(int(base_r + i), int(base_c + i)) for i in range(length)]
    return [(int(base_r + i), int(base_c - i)) for i in range(length)]


def _marker_bbox(spec: dict[str, int | str]) -> tuple[int, int, int, int]:
    coords = np.array(_marker_pixels(spec), dtype=int)
    return bbox_coords(coords)


def _marker_orientation(component: dict[str, np.ndarray | tuple[int, int, int, int] | int]) -> str:
    bbox = component["bbox"]
    assert isinstance(bbox, tuple)
    coords = component["coords"]
    assert isinstance(coords, np.ndarray)
    r0, c0, r1, c1 = map(int, bbox)
    assert bbox_h(bbox) == bbox_w(bbox), "marker bbox must be square"
    coord_set = {tuple(map(int, rc)) for rc in coords}
    length = int(coords.shape[0])
    if all((int(r0 + i), int(c0 + i)) in coord_set for i in range(length)):
        return "nw_se"
    if all((int(r0 + i), int(c1 - i)) in coord_set for i in range(length)):
        return "ne_sw"
    raise AssertionError("marker must be a short diagonal segment")


def _stamp(
    cover: np.ndarray,
    out: np.ndarray,
    patch: np.ndarray,
    r0: int,
    c0: int,
    *,
    bg: int,
) -> None:
    rows, cols = out.shape
    for rr, cc in np.argwhere(patch != int(bg)):
        r = int(r0 + rr)
        c = int(c0 + cc)
        assert 0 <= r < rows and 0 <= c < cols, "stamp placement must stay in bounds"
        cover[r, c] += 1
        if int(cover[r, c]) == 1:
            out[r, c] = int(patch[rr, cc])
        else:
            out[r, c] = int(bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Classify short diagonal markers by orientation, stamp the majority with one template and the minority with the other, then erase overlaps.

    Input structure:
    - The background is black.
    - The upper-left area contains exactly two disconnected monochrome templates:
      a blue template and a green template, both with the same bounding-box size.
    - The lower-right area contains several gray 8-connected components, each a
      2- or 3-pixel diagonal segment in either the NW-SE or NE-SW direction.

    Rule:
    - Crop the blue and green templates from their bounding boxes and record the
      shared template size `(H, W)` and the top-left corner of the whole template area.
    - Classify every gray marker component by its diagonal orientation and count
      how many markers belong to each orientation; the larger group is the majority.
    - Convert each marker bbox's coarse offset from the top-left of the marker
      area into lattice coordinates by integer-dividing row and column offsets by `(H, W)`.
    - Stamp the blue template at majority-marker lattice positions and the green
      template at minority-marker lattice positions, anchored from the template area's start.
    - If two stamped foreground pixels land on the same output cell, erase that
      cell back to black. The original templates and markers do not remain.

    Key insight:
    - The diagonal orientation no longer chooses an output color directly; it
      chooses which template shape gets stamped, and the coarse lattice offsets
      of the marker bboxes determine where those template copies land.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(get_bg(g))

    blue = components_of_color(g, TEMPLATE_A_COLOR, connectivity=4)
    green = components_of_color(g, TEMPLATE_B_COLOR, connectivity=4)
    markers = components_of_color(g, MARKER_COLOR, connectivity=8)

    assert len(blue) == 1 and len(green) == 1
    assert len(markers) >= 2

    bbox_a = blue[0]["bbox"]
    bbox_b = green[0]["bbox"]
    assert isinstance(bbox_a, tuple) and isinstance(bbox_b, tuple)
    assert (bbox_h(bbox_a), bbox_w(bbox_a)) == (bbox_h(bbox_b), bbox_w(bbox_b))

    template_a = crop_bbox(g, bbox_a)
    template_b = crop_bbox(g, bbox_b)
    stamp_h = int(bbox_h(bbox_a))
    stamp_w = int(bbox_w(bbox_a))

    template_start_r = min(int(bbox_a[0]), int(bbox_b[0]))
    template_start_c = min(int(bbox_a[1]), int(bbox_b[1]))
    marker_origin_r = min(int(comp["bbox"][0]) for comp in markers)  # type: ignore[index]
    marker_origin_c = min(int(comp["bbox"][1]) for comp in markers)  # type: ignore[index]

    orientations = [_marker_orientation(comp) for comp in markers]
    counts = Counter(orientations)
    assert len(counts) == 2
    majority_orientation, majority_count = counts.most_common(1)[0]
    minority_count = min(counts.values())
    assert int(majority_count) > int(minority_count)

    out = np.full_like(g, int(bg))
    cover = np.zeros_like(g, dtype=int)

    for comp, orientation in zip(markers, orientations):
        bbox = comp["bbox"]
        assert isinstance(bbox, tuple)
        grid_r = (int(bbox[0]) - int(marker_origin_r)) // int(stamp_h)
        grid_c = (int(bbox[1]) - int(marker_origin_c)) // int(stamp_w)
        patch = template_a if orientation == majority_orientation else template_b
        _stamp(
            cover,
            out,
            patch,
            int(template_start_r + grid_r * stamp_h),
            int(template_start_c + grid_c * stamp_w),
            bg=bg,
        )

    return out


def _choose_marker_specs(total_markers: int, majority_orientation: str) -> list[dict[str, int | str]] | None:
    minority_orientation = "ne_sw" if majority_orientation == "nw_se" else "nw_se"
    majority_count = random.randint(total_markers // 2 + 1, total_markers - 1)
    counts = {"nw_se": 1, "ne_sw": 1}
    need_major = int(majority_count - counts[majority_orientation])
    need_minor = int((total_markers - majority_count) - counts[minority_orientation])
    if need_major < 0 or need_minor < 0:
        return None

    forced = [
        {"row_index": 0, "anchor_col": 0, "mapped_col": 0, "orientation": "nw_se", "length": 2},
        {"row_index": 0, "anchor_col": 1, "mapped_col": 0, "orientation": "ne_sw", "length": 2},
    ]

    remaining_orientations = [majority_orientation] * need_major + [minority_orientation] * need_minor
    mapped_cells = [(row_index, mapped_col) for row_index in (0, 2, 4) for mapped_col in (1, 2, 3)]

    for _ in range(MARKER_RETRIES):
        random.shuffle(remaining_orientations)
        chosen_cells = random.sample(mapped_cells, len(remaining_orientations))
        lengths = [random.choice((2, 3)) for _ in remaining_orientations]
        if lengths and 3 not in lengths and random.random() < 0.7:
            lengths[random.randrange(len(lengths))] = 3

        specs = list(forced)
        for (row_index, mapped_col), orientation, length in zip(chosen_cells, remaining_orientations, lengths):
            anchor_col = int(mapped_col if orientation == "nw_se" else mapped_col + 1)
            specs.append(
                {
                    "row_index": int(row_index),
                    "anchor_col": int(anchor_col),
                    "mapped_col": int(mapped_col),
                    "orientation": str(orientation),
                    "length": int(length),
                }
            )

        anchor_keys = {(int(spec["row_index"]), int(spec["anchor_col"])) for spec in specs}
        if len(anchor_keys) != len(specs):
            continue

        if not _marker_specs_are_valid(specs):
            continue

        mapped_keys = [(int(spec["row_index"]), int(spec["mapped_col"])) for spec in specs]
        mapped_counts = Counter(mapped_keys)
        if len(mapped_counts) != total_markers - 1:
            continue
        if mapped_counts[(0, 0)] != 2:
            continue
        if any(count != 1 for key, count in mapped_counts.items() if key != (0, 0)):
            continue

        return specs

    return None


def _marker_specs_are_valid(specs: list[dict[str, int | str]]) -> bool:
    pixels = []
    for spec in specs:
        coords = _marker_pixels(spec)
        if min(c for _r, c in coords) < 0:
            return False
        pixels.append(coords)

    all_coords = [coord for coords in pixels for coord in coords]
    if len(set(all_coords)) != len(all_coords):
        return False

    height = max(r for r, _c in all_coords) + 1
    width = max(c for _r, c in all_coords) + 1
    canvas = np.full((height, width), BG, dtype=int)
    for coords in pixels:
        for r, c in coords:
            canvas[r, c] = MARKER_COLOR

    comps = components_of_color(canvas, MARKER_COLOR, connectivity=8)
    return len(comps) == len(specs)


def _render_marker_grid(specs: list[dict[str, int | str]]) -> tuple[np.ndarray, list[dict[str, np.ndarray | tuple[int, int, int, int] | int]]]:
    all_coords = [coord for spec in specs for coord in _marker_pixels(spec)]
    height = max(r for r, _c in all_coords) + 1
    width = max(c for _r, c in all_coords) + 1
    grid = np.full((height, width), BG, dtype=int)
    for spec in specs:
        for r, c in _marker_pixels(spec):
            grid[r, c] = MARKER_COLOR
    return grid, components_of_color(grid, MARKER_COLOR, connectivity=8)


def _generate_input_once() -> np.ndarray | None:
    template_mask_a, template_mask_b = _choose_template_pair()
    template_a = _paint_mask(template_mask_a, TEMPLATE_A_COLOR)
    template_b = _paint_mask(template_mask_b, TEMPLATE_B_COLOR)

    total_markers = random.randint(*MARKER_COUNT_RANGE)
    majority_orientation = random.choice(("nw_se", "ne_sw"))
    specs = _choose_marker_specs(total_markers, majority_orientation)
    if specs is None:
        return None

    marker_grid, marker_components = _render_marker_grid(specs)
    marker_origin_r = min(int(comp["bbox"][0]) for comp in marker_components)  # type: ignore[index]
    marker_origin_c = min(int(comp["bbox"][1]) for comp in marker_components)  # type: ignore[index]
    max_mapped_r = max((int(comp["bbox"][0]) - marker_origin_r) // TEMPLATE_H for comp in marker_components)  # type: ignore[index]
    max_mapped_c = max((int(comp["bbox"][1]) - marker_origin_c) // TEMPLATE_W for comp in marker_components)  # type: ignore[index]

    template_start_r = random.randint(1, 2)
    template_start_c = random.randint(1, 2)
    template_gap = random.randint(2, 3)
    template_b_r = template_start_r + TEMPLATE_H + template_gap
    template_b_c = template_start_c
    template_union_r1 = template_b_r + TEMPLATE_H - 1
    template_union_c1 = template_start_c + TEMPLATE_W - 1

    output_needed_h = template_start_r + (max_mapped_r + 1) * TEMPLATE_H
    output_needed_w = template_start_c + (max_mapped_c + 1) * TEMPLATE_W
    marker_needed_h = template_union_r1 + 4 + marker_grid.shape[0]
    marker_needed_w = template_union_c1 + 4 + marker_grid.shape[1]
    min_grid_h = max(output_needed_h, marker_needed_h)
    min_grid_w = max(output_needed_w, marker_needed_w)

    if min_grid_h > GRID_H_LIMIT or min_grid_w > GRID_W_LIMIT:
        return None

    grid_h = random.randint(min_grid_h, GRID_H_LIMIT)
    grid_w = random.randint(min_grid_w, GRID_W_LIMIT)

    anchor_r_min = template_union_r1 + 4
    anchor_c_min = template_union_c1 + 4
    anchor_r_max = grid_h - marker_grid.shape[0]
    anchor_c_max = grid_w - marker_grid.shape[1]
    if anchor_r_min > anchor_r_max or anchor_c_min > anchor_c_max:
        return None

    anchor_r = random.randint(anchor_r_min, anchor_r_max)
    anchor_c = random.randint(anchor_c_min, anchor_c_max)

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    grid[template_start_r : template_start_r + TEMPLATE_H, template_start_c : template_start_c + TEMPLATE_W] = template_a
    grid[template_b_r : template_b_r + TEMPLATE_H, template_b_c : template_b_c + TEMPLATE_W] = template_b

    marker_rows, marker_cols = np.where(marker_grid == MARKER_COLOR)
    grid[anchor_r + marker_rows, anchor_c + marker_cols] = MARKER_COLOR

    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape != grid.shape:
        return None
    if not set(map(int, np.unique(out))).issubset({BG, TEMPLATE_A_COLOR, TEMPLATE_B_COLOR}):
        return None
    if not (out == BG).all() and np.array_equal(out, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _generate_input_once()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
