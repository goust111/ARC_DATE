from __future__ import annotations

import hashlib
import random

import numpy as np


BG = 0
MARKER = 8
SPRITE_COLORS = (1, 2, 3, 4, 5, 6, 7, 9)
GRID_SIDE_RANGE = (15, 20)
OBJECT_COUNT_RANGE = (5, 7)
SELECTED_COUNT_RANGE = (3, 4)
AREA_RANGE = tuple(range(3, 9))
MAX_GENERATION_ATTEMPTS = 500

_SEEN_INPUT_SIGS: set[str] = set()
_SEEN_OUTPUT_SIGS: set[str] = set()


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _normalize(cells: set[tuple[int, int]] | list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    r0 = min(rows)
    c0 = min(cols)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _bbox_from_cells(cells: tuple[tuple[int, int], ...] | np.ndarray) -> tuple[int, int, int, int]:
    if isinstance(cells, np.ndarray):
        rows = cells[:, 0]
        cols = cells[:, 1]
    else:
        rows = [r for r, _c in cells]
        cols = [c for _r, c in cells]
    return int(min(rows)), int(min(cols)), int(max(rows)), int(max(cols))


def _is_connected(cells: tuple[tuple[int, int], ...]) -> bool:
    cell_set = set(cells)
    seen = {cells[0]}
    stack = [cells[0]]
    while stack:
        r, c = stack.pop()
        for rr, cc in _neighbors4(r, c):
            if (rr, cc) in cell_set and (rr, cc) not in seen:
                seen.add((rr, cc))
                stack.append((rr, cc))
    return len(seen) == len(cell_set)


def _marker_candidates(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    cell_set = set(cells)
    counts: dict[tuple[int, int], int] = {}
    for r, c in cells:
        for rr, cc in _neighbors4(r, c):
            if (rr, cc) in cell_set:
                continue
            counts[(rr, cc)] = counts.get((rr, cc), 0) + 1
    return tuple(sorted(pos for pos, count in counts.items() if count == 1))


def _build_shape_library() -> dict[int, list[dict[str, object]]]:
    by_area = {area: [] for area in AREA_RANGE}
    seen_by_area = {area: set() for area in AREA_RANGE}
    for bits in range(1 << 16):
        area = bits.bit_count()
        if area not in by_area:
            continue
        cells = []
        for idx in range(16):
            if (bits >> idx) & 1:
                cells.append((idx // 4, idx % 4))
        norm = _normalize(cells)
        if norm in seen_by_area[area]:
            continue
        if not _is_connected(norm):
            continue
        seen_by_area[area].add(norm)
        r0, c0, r1, c1 = _bbox_from_cells(norm)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        by_area[area].append(
            {
                "cells": norm,
                "area": int(area),
                "h": int(h),
                "w": int(w),
                "irregular": bool(area < h * w),
                "marker_candidates": _marker_candidates(norm),
            }
        )
    return by_area


SHAPES_BY_AREA = _build_shape_library()
SELECTABLE_SHAPES_BY_AREA = {
    area: [shape for shape in shapes if shape["marker_candidates"]]
    for area, shapes in SHAPES_BY_AREA.items()
}


def _sprite_patch(shape: dict[str, object], color: int) -> np.ndarray:
    patch = np.zeros((int(shape["h"]), int(shape["w"])), dtype=int)
    for r, c in shape["cells"]:
        patch[int(r), int(c)] = int(color)
    return patch


def _marked_patch(shape: dict[str, object], color: int, marker_pos: tuple[int, int]) -> np.ndarray:
    all_cells = list(shape["cells"]) + [marker_pos]
    r0, c0, r1, c1 = _bbox_from_cells(tuple(all_cells))
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in shape["cells"]:
        patch[int(r - r0), int(c - c0)] = int(color)
    patch[int(marker_pos[0] - r0), int(marker_pos[1] - c0)] = MARKER
    return patch


def _stack_patches(patches: list[np.ndarray]) -> np.ndarray:
    width = max(p.shape[1] for p in patches)
    height = sum(p.shape[0] for p in patches) + (len(patches) - 1)
    out = np.zeros((height, width), dtype=int)
    row = 0
    for patch in patches:
        h, w = patch.shape
        out[row : row + h, :w] = patch
        row += h + 1
    return out


def _component_data(grid: np.ndarray) -> tuple[list[dict[str, object]], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    mask = (g != BG) & (g != MARKER)
    labels = np.zeros((h, w), dtype=int)
    comps: list[dict[str, object]] = []
    next_id = 1
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            stack = [(r, c)]
            labels[r, c] = next_id
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds((h, w), nr, nc) and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = next_id
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0, c0, r1, c1 = _bbox_from_cells(arr)
            colors = sorted({int(g[rr, cc]) for rr, cc in coords})
            comps.append(
                {
                    "id": int(next_id),
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(arr.shape[0]),
                    "colors": colors,
                    "color": int(colors[0]) if len(colors) == 1 else -1,
                }
            )
            next_id += 1
    return comps, labels


def _selected_components(grid: np.ndarray) -> tuple[list[dict[str, object]], list[tuple[int, int]]]:
    comps, labels = _component_data(grid)
    comp_by_id = {int(comp["id"]): comp for comp in comps}
    markers = [tuple(map(int, pos)) for pos in np.argwhere(np.asarray(grid) == MARKER)]
    selected_ids = []
    for r, c in markers:
        neighbors = []
        for rr, cc in _neighbors4(r, c):
            if _in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) not in (BG, MARKER):
                neighbors.append((rr, cc))
        assert len(neighbors) == 1
        comp_id = int(labels[neighbors[0]])
        assert comp_id > 0
        selected_ids.append(comp_id)
    assert len(selected_ids) == len(set(selected_ids))
    return [comp_by_id[idx] for idx in selected_ids], markers


def main(grid: np.ndarray) -> np.ndarray:
    """Select the teal-marked sprites, sort them by area, and stack their cropped shapes vertically.

    Input structure:
    - Black background with 5-7 small monochrome 4-connected sprites.
    - Exactly 3-4 sprites each have one teal marker pixel that is 4-neighbor-adjacent to exactly one sprite pixel.
    - Unmarked sprites are distractors and do not participate in the output.

    Rule:
    - For every teal marker, find the unique adjacent non-black non-teal sprite component.
    - Count each selected sprite's own pixels, excluding the marker.
    - Crop each selected sprite to its minimal bounding box, keep its original color on sprite cells, and keep black elsewhere inside the crop.
    - Sort the selected crops by increasing sprite area and stack them top-to-bottom with one black separator row, left-aligned and padded to the maximum crop width.

    Key insight:
    - The teal marker is both a selector and an exclusion cue: it chooses which sprites matter, but the marker itself is discarded before counting, sorting, and assembling.
    """
    g = np.asarray(grid, dtype=int)
    selected, _markers = _selected_components(g)
    ordered = sorted(selected, key=lambda comp: (int(comp["area"]), int(comp["bbox"][0]), int(comp["bbox"][1])))
    patches = []
    for comp in ordered:
        r0, c0, r1, c1 = comp["bbox"]
        patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
        for r, c in comp["coords"]:
            patch[int(r - r0), int(c - c0)] = int(g[int(r), int(c)])
        patches.append(patch)
    return _stack_patches(patches)


def _can_place(occupied: np.ndarray, patch: np.ndarray, top: int, left: int) -> bool:
    mask = patch != BG
    h, w = patch.shape
    H, W = occupied.shape
    if top < 0 or left < 0 or top + h > H or left + w > W:
        return False
    for pr, pc in np.argwhere(mask):
        rr = int(top + pr)
        cc = int(left + pc)
        r0 = max(0, rr - 1)
        r1 = min(H, rr + 2)
        c0 = max(0, cc - 1)
        c1 = min(W, cc + 2)
        if occupied[r0:r1, c0:c1].any():
            return False
    return True


def _place_patch(grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray) -> bool:
    h, w = patch.shape
    H, W = grid.shape
    positions = [(r, c) for r in range(H - h + 1) for c in range(W - w + 1)]
    random.shuffle(positions)
    for top, left in positions:
        if not _can_place(occupied, patch, top, left):
            continue
        mask = patch != BG
        view = grid[top : top + h, left : left + w]
        view[mask] = patch[mask]
        occupied[top : top + h, left : left + w] |= mask
        return True
    return False


def _choose_selected_shapes(areas: list[int]) -> list[dict[str, object]] | None:
    for _ in range(80):
        shapes = [random.choice(SELECTABLE_SHAPES_BY_AREA[area]) for area in areas]
        widths = {int(shape["w"]) for shape in shapes}
        if len(widths) < 2:
            continue
        if not any(bool(shape["irregular"]) for shape in shapes):
            continue
        return shapes
    return None


def _scene_invariants(grid: np.ndarray) -> tuple[bool, np.ndarray | None]:
    try:
        comps, _labels = _component_data(grid)
        selected, markers = _selected_components(grid)
    except Exception:
        return False, None

    if not (OBJECT_COUNT_RANGE[0] <= len(comps) <= OBJECT_COUNT_RANGE[1]):
        return False, None
    if not (SELECTED_COUNT_RANGE[0] <= len(selected) <= SELECTED_COUNT_RANGE[1]):
        return False, None
    if len(markers) != len(selected):
        return False, None

    selected_ids = {int(comp["id"]) for comp in selected}
    marker_counts: dict[int, int] = {comp_id: 0 for comp_id in selected_ids}
    for r, c in markers:
        neighbors = [(rr, cc) for rr, cc in _neighbors4(r, c) if _in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) not in (BG, MARKER)]
        if len(neighbors) != 1:
            return False, None
        _, labels = _component_data(grid)
        comp_id = int(labels[neighbors[0]])
        marker_counts[comp_id] = marker_counts.get(comp_id, 0) + 1
        non_bg_neighbors = [(rr, cc) for rr, cc in _neighbors4(r, c) if _in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) != BG]
        if len(non_bg_neighbors) != 1:
            return False, None

    if any(count != 1 for count in marker_counts.values()):
        return False, None

    for comp in comps:
        if len(comp["colors"]) != 1:
            return False, None
        if not (3 <= int(comp["area"]) <= 8):
            return False, None
        r0, c0, r1, c1 = comp["bbox"]
        if r1 - r0 + 1 > 4 or c1 - c0 + 1 > 4:
            return False, None

    for i, comp_a in enumerate(comps):
        a0, b0, a1, b1 = comp_a["bbox"]
        for comp_b in comps[i + 1 :]:
            c0, d0, c1, d1 = comp_b["bbox"]
            separated = a1 + 1 < c0 or c1 + 1 < a0 or b1 + 1 < d0 or d1 + 1 < b0
            if not separated:
                return False, None

    selected_areas = [int(comp["area"]) for comp in selected]
    if len(set(selected_areas)) != len(selected_areas):
        return False, None

    scan_order = [int(comp["id"]) for comp in sorted(selected, key=lambda comp: (int(comp["bbox"][0]), int(comp["bbox"][1])))]
    area_order = [int(comp["id"]) for comp in sorted(selected, key=lambda comp: int(comp["area"]))]
    if scan_order == area_order:
        return False, None

    distractors = [comp for comp in comps if int(comp["id"]) not in selected_ids]
    if not distractors:
        return False, None
    if not any(int(comp["area"]) in selected_areas for comp in distractors):
        return False, None

    all_colors = [int(comp["color"]) for comp in comps]
    if len(set(all_colors)) == len(all_colors):
        return False, None
    selected_colors = {int(comp["color"]) for comp in selected}
    if not any(int(comp["color"]) in selected_colors for comp in distractors):
        return False, None

    out = main(grid)
    widths = [int(comp["bbox"][3] - comp["bbox"][1] + 1) for comp in selected]
    if len(set(widths)) < 2:
        return False, None
    if out.ndim != 2 or max(out.shape) > 30:
        return False, None
    return True, out


def _build_objects() -> tuple[list[dict[str, object]], np.ndarray] | None:
    n_objects = random.randint(*OBJECT_COUNT_RANGE)
    n_selected = random.randint(*SELECTED_COUNT_RANGE)
    n_distractors = n_objects - n_selected

    selected_areas = random.sample(list(AREA_RANGE), n_selected)
    selected_shapes = _choose_selected_shapes(selected_areas)
    if selected_shapes is None:
        return None

    distractor_areas = [random.choice(selected_areas)]
    while len(distractor_areas) < n_distractors:
        distractor_areas.append(random.choice(AREA_RANGE))
    random.shuffle(distractor_areas)
    distractor_shapes = [random.choice(SHAPES_BY_AREA[area]) for area in distractor_areas]

    palette_size = random.randint(2, min(4, n_objects))
    palette = random.sample(list(SPRITE_COLORS), palette_size)

    selected_colors = [random.choice(palette) for _ in range(n_selected)]
    distractor_colors = [random.choice(selected_colors)]
    while len(distractor_colors) < n_distractors:
        distractor_colors.append(random.choice(palette))
    random.shuffle(distractor_colors)

    objects: list[dict[str, object]] = []
    selected_patches = []

    for shape, color, area in zip(selected_shapes, selected_colors, selected_areas):
        marker_pos = random.choice(shape["marker_candidates"])
        sprite_patch = _sprite_patch(shape, int(color))
        placed_patch = _marked_patch(shape, int(color), marker_pos)
        objects.append(
            {
                "selected": True,
                "area": int(area),
                "shape": shape,
                "color": int(color),
                "patch": placed_patch,
            }
        )
        selected_patches.append((int(area), sprite_patch))

    for shape, color, area in zip(distractor_shapes, distractor_colors, distractor_areas):
        objects.append(
            {
                "selected": False,
                "area": int(area),
                "shape": shape,
                "color": int(color),
                "patch": _sprite_patch(shape, int(color)),
            }
        )

    expected = _stack_patches([patch for _area, patch in sorted(selected_patches, key=lambda item: item[0])])
    random.shuffle(objects)
    return objects, expected


def _signature(grid: np.ndarray) -> str:
    return hashlib.md5(grid.tobytes() + str(grid.shape).encode()).hexdigest()


def _gen_attempt() -> np.ndarray | None:
    built = _build_objects()
    if built is None:
        return None
    objects, expected = built

    H = random.randint(*GRID_SIDE_RANGE)
    W = random.randint(*GRID_SIDE_RANGE)
    grid = np.zeros((H, W), dtype=int)
    occupied = np.zeros((H, W), dtype=bool)

    random.shuffle(objects)
    for obj in objects:
        if not _place_patch(grid, occupied, obj["patch"]):
            return None

    ok, out = _scene_invariants(grid)
    if not ok or out is None:
        return None
    if not np.array_equal(out, expected):
        return None

    input_sig = _signature(grid)
    output_sig = _signature(out)
    if input_sig in _SEEN_INPUT_SIGS or output_sig in _SEEN_OUTPUT_SIGS:
        return None
    _SEEN_INPUT_SIGS.add(input_sig)
    _SEEN_OUTPUT_SIGS.add(output_sig)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample for puzzle 8")
