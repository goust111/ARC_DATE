from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
GREY = 5
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
CORNERS = ("tl", "tr", "bl", "br")


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _corner_coords(r0: int, c0: int, r1: int, c1: int) -> dict[str, tuple[int, int]]:
    return {
        "tl": (int(r0), int(c0)),
        "tr": (int(r0), int(c1)),
        "bl": (int(r1), int(c0)),
        "br": (int(r1), int(c1)),
    }


def _place_for_corner(canvas_shape: tuple[int, int], patch_shape: tuple[int, int], corner: str) -> tuple[int, int]:
    out_h, out_w = map(int, canvas_shape)
    h, w = map(int, patch_shape)
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, out_w - w
    if corner == "bl":
        return out_h - h, 0
    if corner == "br":
        return out_h - h, out_w - w
    raise ValueError(f"unknown corner: {corner}")


def _transparent_paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    mask = patch != BLACK
    view[mask] = patch[mask]


def _parse_objects(grid: np.ndarray) -> list[dict[str, object]]:
    objects: list[dict[str, object]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in {BLACK, GREY}):
        coords = np.argwhere(grid == color)
        assert coords.shape == (3, 2), "each marker color must appear exactly three times"
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "marker box must have a non-empty strict interior"
        corners = _corner_coords(r0, c0, r1, c1)
        seen = {tuple(map(int, rc)) for rc in coords}
        missing = [name for name, rc in corners.items() if rc not in seen]
        assert len(missing) == 1, "markers must occupy exactly three bbox corners"
        missing_corner = missing[0]

        box = grid[r0 : r1 + 1, c0 : c1 + 1]
        allowed = {BLACK, GREY, color}
        assert set(map(int, np.unique(box))) <= allowed, "unexpected colors inside object box"
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        patch = np.zeros_like(interior)
        patch[interior == GREY] = color
        area = int((r1 - r0 + 1) * (c1 - c0 + 1))
        objects.append(
            {
                "color": int(color),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "missing_corner": missing_corner,
                "patch": patch,
                "area": area,
                "shape": patch.shape,
            }
        )

    assert objects, "expected at least one object"
    areas = [int(obj["area"]) for obj in objects]
    assert len(set(areas)) == len(areas), "object bbox areas must be distinct"
    return objects


def _render_output_from_objects(objects: Iterable[dict[str, object]]) -> np.ndarray:
    items = list(objects)
    largest = max(items, key=lambda obj: int(obj["area"]))
    out_h, out_w = map(int, largest["patch"].shape)
    out = np.zeros((out_h, out_w), dtype=int)

    for obj in sorted(items, key=lambda item: int(item["area"]), reverse=True):
        patch = np.asarray(obj["patch"], dtype=int)
        corner = str(obj["missing_corner"])
        rr, cc = _place_for_corner(out.shape, patch.shape, corner)
        _transparent_paste(out, patch, rr, cc)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each gray interior and stack it into the canvas corner named by its missing marker corner.

    Input structure:
    - A black background contains several separated objects.
    - Each object has a gray interior pattern inside a rectangular marker box.
    - Exactly three corners of that box contain same-colored marker pixels; the fourth corner is empty.

    Rule:
    - For each object, use the three marker pixels to recover the full marker-box bounding box.
    - Crop the strict interior of that box, recolor every gray pattern pixel to the marker color,
      and treat black cells as transparent.
    - The missing marker corner tells which output corner to align that interior pattern to.
    - The output canvas size is the strict interior size of the largest marker box, and objects are
      pasted from larger boxes to smaller boxes so later smaller ones overwrite earlier overlaps.

    Key insight:
    - The absent corner is not damage to ignore: it is the anchor instruction that chooses the
      destination corner, while box size determines the visible layer order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    objects = _parse_objects(g)
    return _render_output_from_objects(objects)


def _grow_connected_mask(h: int, w: int, target_area: int, required: set[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    cells = {(int(r), int(c)) for r, c in required}
    cells.add(random.choice(sorted(cells)))
    if len(cells) > target_area:
        target_area = len(cells)

    frontier = set(cells)
    while len(cells) < target_area:
        candidates = set()
        for r, c in frontier:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                    candidates.add((rr, cc))
        if not candidates:
            empties = [(r, c) for r in range(h) for c in range(w) if (r, c) not in cells]
            if not empties:
                break
            candidates = set(empties)
        pick = random.choice(sorted(candidates))
        cells.add(pick)
        frontier = {pick}

    for r, c in cells:
        mask[r, c] = True
    return mask


def _random_mask(h: int, w: int, anchor_corner: str, dense: bool) -> np.ndarray:
    area = h * w
    required = {
        "tl": {(0, 0)},
        "tr": {(0, w - 1)},
        "bl": {(h - 1, 0)},
        "br": {(h - 1, w - 1)},
    }[anchor_corner]
    if dense:
        probs = np.random.uniform(0.58, 0.82, size=(h, w))
        mask = np.random.random((h, w)) < probs
        mask[0, 0] = True
        mask[0, w - 1] = True
        mask[h - 1, 0] = True
        mask[h - 1, w - 1] = True
        if mask.all():
            rr = random.randrange(h)
            cc = random.randrange(w)
            mask[rr, cc] = False
            for r, c in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)):
                mask[r, c] = True
        return mask

    min_area = max(3, (area + 3) // 4)
    max_area = max(min_area, int(round(area * 0.7)))
    target_area = random.randint(min_area, max_area)
    mask = _grow_connected_mask(h, w, target_area, required)
    if mask.sum() == area:
        extras = np.argwhere(mask)
        if len(extras) > 1:
            rr, cc = extras[random.randrange(len(extras))]
            if (rr, cc) not in required:
                mask[rr, cc] = False
    return mask


def _render_object(inner_mask: np.ndarray, color: int, missing_corner: str) -> np.ndarray:
    h, w = inner_mask.shape
    obj = np.zeros((h + 2, w + 2), dtype=int)
    corners = {
        "tl": (0, 0),
        "tr": (0, w + 1),
        "bl": (h + 1, 0),
        "br": (h + 1, w + 1),
    }
    for name, (r, c) in corners.items():
        if name != missing_corner:
            obj[r, c] = color
    obj[1 : h + 1, 1 : w + 1][inner_mask] = GREY
    return obj


def _boxes_separated(boxes: list[tuple[int, int, int, int]], candidate: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = candidate
    for a0, b0, a1, b1 in boxes:
        if not (r1 + 1 < a0 or a1 + 1 < r0 or c1 + 1 < b0 or b1 + 1 < c0):
            return False
    return True


def _render_scene(specs: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray]:
    objects = specs[:]
    random.shuffle(objects)
    patches = [np.asarray(obj["input_patch"], dtype=int) for obj in objects]
    top_h = max(int(patches[0].shape[0]), int(patches[1].shape[0]))
    bottom_h = max(int(patches[2].shape[0]), int(patches[3].shape[0]))
    left_w = max(int(patches[0].shape[1]), int(patches[2].shape[1]))
    right_w = max(int(patches[1].shape[1]), int(patches[3].shape[1]))

    min_h = top_h + bottom_h + 1
    min_w = left_w + right_w + 1
    if min_h > 30 or min_w > 30:
        raise RuntimeError("scene does not fit ARC size limit")

    canvas_h = random.randint(min_h, 30)
    canvas_w = random.randint(min_w, 30)
    grid = np.zeros((canvas_h, canvas_w), dtype=int)

    extra_h = canvas_h - min_h
    extra_w = canvas_w - min_w
    top_margin = random.randint(0, extra_h)
    bottom_margin = extra_h - top_margin
    left_margin = random.randint(0, extra_w)
    right_margin = extra_w - left_margin

    row_split = top_margin + top_h
    col_split = left_margin + left_w
    row_gap = canvas_h - top_margin - bottom_margin - top_h - bottom_h
    col_gap = canvas_w - left_margin - right_margin - left_w - right_w
    assert row_gap >= 1 and col_gap >= 1

    slots = [
        (top_margin, left_margin, row_split - 1, col_split - 1),
        (top_margin, col_split + col_gap, row_split - 1, canvas_w - right_margin - 1),
        (row_split + row_gap, left_margin, canvas_h - bottom_margin - 1, col_split - 1),
        (row_split + row_gap, col_split + col_gap, canvas_h - bottom_margin - 1, canvas_w - right_margin - 1),
    ]
    used: list[tuple[int, int, int, int]] = []
    for obj, patch, slot in zip(objects, patches, slots):
        sr0, sc0, sr1, sc1 = slot
        h, w = patch.shape
        r0 = random.randint(sr0, sr1 - h + 1)
        c0 = random.randint(sc0, sc1 - w + 1)
        box = (r0, c0, r0 + h - 1, c0 + w - 1)
        assert _boxes_separated(used, box)
        grid[r0 : r0 + h, c0 : c0 + w] = patch
        used.append(box)

    objects_for_output = [
        {
            "color": int(obj["color"]),
            "patch": np.asarray(obj["output_patch"], dtype=int),
            "missing_corner": str(obj["missing_corner"]),
            "area": int(obj["outer_area"]),
        }
        for obj in specs
    ]
    expected = _render_output_from_objects(objects_for_output)
    return grid, expected


def _alternative_anchor_changes_output(specs: list[dict[str, object]], expected: np.ndarray) -> bool:
    for idx, obj in enumerate(specs):
        if obj["output_patch"].shape == expected.shape:
            continue
        for alt in CORNERS:
            if alt == obj["missing_corner"]:
                continue
            altered = []
            for j, item in enumerate(specs):
                altered.append(
                    {
                        "color": int(item["color"]),
                        "patch": np.asarray(item["output_patch"], dtype=int),
                        "missing_corner": alt if j == idx else str(item["missing_corner"]),
                        "area": int(item["outer_area"]),
                    }
                )
            if not np.array_equal(_render_output_from_objects(altered), expected):
                return True
    return False


def _wrong_layer_order_changes_output(specs: list[dict[str, object]], expected: np.ndarray) -> bool:
    largest = max(specs, key=lambda item: int(item["outer_area"]))
    out = np.zeros_like(expected)
    ascending = sorted(specs, key=lambda item: int(item["outer_area"]))
    for item in ascending:
        patch = np.asarray(item["output_patch"], dtype=int)
        rr, cc = _place_for_corner(largest["output_patch"].shape, patch.shape, str(item["missing_corner"]))
        _transparent_paste(out, patch, rr, cc)
    return not np.array_equal(out, expected)


def _has_meaningful_overlap(specs: list[dict[str, object]], expected_shape: tuple[int, int]) -> bool:
    cover = np.zeros(expected_shape, dtype=int)
    for item in specs:
        patch = np.asarray(item["output_patch"], dtype=int)
        rr, cc = _place_for_corner(expected_shape, patch.shape, str(item["missing_corner"]))
        mask = patch != BLACK
        cover[rr : rr + patch.shape[0], cc : cc + patch.shape[1]] += mask.astype(int)
    return int(cover.max()) >= 2


def _build_specs() -> list[dict[str, object]]:
    out_h = random.randint(6, 10)
    out_w = random.randint(6, 10)
    largest_area = (out_h + 2) * (out_w + 2)

    smaller_sizes: list[tuple[int, int]] = []
    seen_areas = {largest_area}
    pool = [(h, w) for h in range(2, out_h + 1) for w in range(2, out_w + 1)]
    random.shuffle(pool)
    for h, w in pool:
        area = (h + 2) * (w + 2)
        if area in seen_areas:
            continue
        seen_areas.add(area)
        smaller_sizes.append((h, w))
        if len(smaller_sizes) == 3:
            break
    if len(smaller_sizes) < 3:
        raise RuntimeError("insufficient smaller size choices")

    sizes = [(out_h, out_w)] + sorted(smaller_sizes, key=lambda item: (item[0] + 2) * (item[1] + 2), reverse=True)
    duplicate_corner = random.choice(CORNERS)
    other_corners = [corner for corner in CORNERS if corner != duplicate_corner]
    random.shuffle(other_corners)
    anchors = [other_corners[0], other_corners[1], duplicate_corner, duplicate_corner]
    if random.random() < 0.5:
        anchors[0], anchors[1] = anchors[1], anchors[0]

    colors = random.sample(MARKER_COLORS, 4)
    specs: list[dict[str, object]] = []
    for idx, ((h, w), color, corner) in enumerate(zip(sizes, colors, anchors)):
        dense = idx == 0
        mask = _random_mask(h, w, corner, dense=dense)
        output_patch = np.zeros((h, w), dtype=int)
        output_patch[mask] = color
        specs.append(
            {
                "color": color,
                "missing_corner": corner,
                "output_patch": output_patch,
                "input_patch": _render_object(mask, color, corner),
                "outer_area": (h + 2) * (w + 2),
            }
        )
    return specs


def _attempt_generate() -> np.ndarray | None:
    specs = _build_specs()
    grid, expected = _render_scene(specs)

    if not np.array_equal(main(grid), expected):
        return None
    if len({str(item["missing_corner"]) for item in specs}) < 3:
        return None
    if not _has_meaningful_overlap(specs, expected.shape):
        return None
    if not _alternative_anchor_changes_output(specs, expected):
        return None
    if not _wrong_layer_order_changes_output(specs, expected):
        return None
    if np.count_nonzero(expected) < 6:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid missing-corner-anchor puzzle")
