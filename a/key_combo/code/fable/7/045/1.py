from __future__ import annotations

import itertools
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _bbox_coords(coords: list[Coord] | np.ndarray) -> BBox:
    arr = np.asarray(coords, dtype=int)
    r0, c0 = arr.min(axis=0)
    r1, c1 = arr.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_h(bbox: BBox) -> int:
    return int(bbox[2] - bbox[0] + 1)


def _bbox_w(bbox: BBox) -> int:
    return int(bbox[3] - bbox[1] + 1)


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    seen = np.zeros((H, W), dtype=bool)
    components: list[dict] = []

    for r in range(H):
        for c in range(W):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            color = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox_coords(coords),
                    "area": len(coords),
                }
            )

    return components


def _three_corner_rectangle(
    points: tuple[Coord, Coord, Coord], inner_h: int, inner_w: int
) -> tuple[BBox, Coord] | None:
    pts = {tuple(map(int, point)) for point in points}
    if len(pts) != 3:
        return None

    rows = sorted({r for r, _ in pts})
    cols = sorted({c for _, c in pts})
    if len(rows) != 2 or len(cols) != 2:
        return None

    r0, r1 = rows
    c0, c1 = cols
    if int(r1 - r0 - 1) != int(inner_h) or int(c1 - c0 - 1) != int(inner_w):
        return None

    corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    if not pts.issubset(corners):
        return None

    missing = next(iter(corners - pts))
    return (int(r0), int(c0), int(r1), int(c1)), (int(missing[0]), int(missing[1]))


def _matching_rectangles(marker_coords: list[Coord], inner_h: int, inner_w: int) -> list[dict]:
    matches: list[dict] = []
    for triple in itertools.combinations(marker_coords, 3):
        found = _three_corner_rectangle(triple, inner_h, inner_w)
        if found is None:
            continue
        bbox, missing = found
        matches.append({"bbox": bbox, "missing": missing, "present": set(triple)})
    return matches


def _all_three_corner_rectangles(marker_coords: list[Coord]) -> list[dict]:
    found: list[dict] = []
    for triple in itertools.combinations(marker_coords, 3):
        pts = {tuple(map(int, point)) for point in triple}
        rows = sorted({r for r, _ in pts})
        cols = sorted({c for _, c in pts})
        if len(pts) != 3 or len(rows) != 2 or len(cols) != 2:
            continue
        r0, r1 = rows
        c0, c1 = cols
        corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
        if not pts.issubset(corners):
            continue
        missing = next(iter(corners - pts))
        found.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "missing": (int(missing[0]), int(missing[1])),
                "present": pts,
            }
        )
    return found


def _paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    for rr, cc in np.argwhere(mask):
        grid[int(r0 + rr), int(c0 + cc)] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """Complete the unique size-matching marker rectangle and move the object into its interior.

    Input structure:
    - The background is black.
    - There is one monochrome 4-connected foreground object and several isolated marker pixels.
    - Exactly one set of three markers forms three corners of an axis-aligned rectangle whose
      interior height and width match the object's bounding box.
    - Other markers may also form three-corner rectangles, but only at different interior sizes.

    Rule:
    - Read the moving object's bounding-box height and width.
    - Search all marker triples for the unique rectangle whose interior matches those dimensions.
    - Add the missing fourth corner in the marker color, erase the object from its source, and
      translate the object so its bounding box fills that rectangle's interior exactly.

    Key insight:
    - The correct destination is not the only partial rectangle on the board; the object's bbox
      dimensions are the filter that identify which three markers matter.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    components = _find_components(g, bg=bg)
    large_components = [comp for comp in components if int(comp["area"]) > 1]
    marker_components = [comp for comp in components if int(comp["area"]) == 1]

    assert len(large_components) == 1, "expected one moving object"
    assert marker_components, "expected marker pixels"

    obj = large_components[0]
    marker_color = int(marker_components[0]["color"])
    assert all(int(comp["color"]) == marker_color for comp in marker_components), "markers must share one color"

    obj_bbox = obj["bbox"]
    inner_h = _bbox_h(obj_bbox)
    inner_w = _bbox_w(obj_bbox)
    marker_coords = [tuple(comp["coords"][0]) for comp in marker_components]

    matches = _matching_rectangles(marker_coords, inner_h, inner_w)
    assert len(matches) == 1, "expected exactly one size-matching three-corner rectangle"

    target_bbox = matches[0]["bbox"]
    missing = matches[0]["missing"]
    target_r0 = int(target_bbox[0] + 1)
    target_c0 = int(target_bbox[1] + 1)
    source_r0, source_c0, _source_r1, _source_c1 = obj_bbox
    dr = int(target_r0 - source_r0)
    dc = int(target_c0 - source_c0)

    out = g.copy()
    for r, c in obj["coords"]:
        out[int(r), int(c)] = bg

    mr, mc = missing
    assert int(out[mr, mc]) == bg, "missing corner must be empty"
    out[mr, mc] = marker_color

    obj_color = int(obj["color"])
    H, W = out.shape
    for r, c in obj["coords"]:
        nr = int(r + dr)
        nc = int(c + dc)
        assert 0 <= nr < H and 0 <= nc < W, "translated object must stay in bounds"
        assert int(out[nr, nc]) == bg, "translated object must land on empty cells"
        out[nr, nc] = obj_color

    return out


def _mask_is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False

    start = tuple(map(int, coords[0]))
    stack = [start]
    seen = {start}
    H, W = mask.shape

    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < H and 0 <= nc < W and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))

    return len(seen) == int(mask.sum())


def _mask_spans_bbox(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    r0, c0, r1, c1 = _bbox_coords(coords)
    H, W = mask.shape
    return (r0, c0, r1, c1) == (0, 0, H - 1, W - 1)


def _generate_object_mask(h: int, w: int) -> np.ndarray | None:
    min_area = max(4, h + w - 1, (h * w + 1) // 2)
    max_area = h * w
    if min_area > max_area:
        return None

    for _ in range(200):
        target_area = random.randint(min_area, max_area)
        if max_area > min_area and random.random() < 0.8:
            target_area = random.randint(min_area, max_area - 1)

        mask = np.ones((h, w), dtype=bool)
        cells = [(r, c) for r in range(h) for c in range(w)]
        random.shuffle(cells)

        for r, c in cells:
            if int(mask.sum()) <= target_area:
                break
            mask[r, c] = False
            if not _mask_spans_bbox(mask) or not _mask_is_connected(mask):
                mask[r, c] = True

        if int(mask.sum()) == target_area and _mask_spans_bbox(mask) and _mask_is_connected(mask):
            return mask

    return None


def _boxes_too_close(a: BBox, b: BBox, pad: int = 0) -> bool:
    return not (
        int(a[2]) + pad < int(b[0])
        or int(b[2]) + pad < int(a[0])
        or int(a[3]) + pad < int(b[1])
        or int(b[3]) + pad < int(a[1])
    )


def _rectangle_candidates(
    H: int, W: int, inner_h: int, inner_w: int, used_boxes: list[BBox], used_rows: set[int], used_cols: set[int]
) -> list[BBox]:
    outer_h = int(inner_h + 2)
    outer_w = int(inner_w + 2)
    candidates: list[BBox] = []

    for r0 in range(0, H - outer_h + 1):
        r1 = int(r0 + inner_h + 1)
        if r0 in used_rows or r1 in used_rows:
            continue
        for c0 in range(0, W - outer_w + 1):
            c1 = int(c0 + inner_w + 1)
            if c0 in used_cols or c1 in used_cols:
                continue
            bbox = (int(r0), int(c0), int(r1), int(c1))
            if any(_boxes_too_close(bbox, other, pad=1) for other in used_boxes):
                continue
            candidates.append(bbox)

    random.shuffle(candidates)
    return candidates


def _source_candidates(H: int, W: int, h: int, w: int, forbidden_boxes: list[BBox]) -> list[Coord]:
    candidates: list[Coord] = []
    for r0 in range(0, H - h + 1):
        for c0 in range(0, W - w + 1):
            bbox = (int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1))
            if any(_boxes_too_close(bbox, other, pad=1) for other in forbidden_boxes):
                continue
            candidates.append((int(r0), int(c0)))
    random.shuffle(candidates)
    return candidates


def _corners_from_bbox(bbox: BBox) -> list[Coord]:
    r0, c0, r1, c1 = bbox
    return [(int(r0), int(c0)), (int(r0), int(c1)), (int(r1), int(c0)), (int(r1), int(c1))]


def _render_expected_output(
    inp: np.ndarray, obj_mask: np.ndarray, obj_color: int, source_pos: Coord, target_bbox: BBox, marker_color: int, missing: Coord
) -> np.ndarray:
    out = inp.copy()
    sr0, sc0 = source_pos
    for rr, cc in np.argwhere(obj_mask):
        out[int(sr0 + rr), int(sc0 + cc)] = 0

    out[int(missing[0]), int(missing[1])] = int(marker_color)
    _paint_mask(out, obj_mask, int(target_bbox[0] + 1), int(target_bbox[1] + 1), obj_color)
    return out


def _generate_attempt() -> np.ndarray | None:
    inner_h = random.randint(2, 6)
    inner_w = random.randint(2, 6)
    obj_mask = _generate_object_mask(inner_h, inner_w)
    if obj_mask is None:
        return None

    decoy_count = random.randint(1, 2)
    decoy_dims: list[tuple[int, int]] = []
    for _ in range(decoy_count):
        for _retry in range(50):
            dims = (random.randint(2, 6), random.randint(2, 6))
            if dims != (inner_h, inner_w):
                decoy_dims.append(dims)
                break
        else:
            return None

    min_h = max(12, inner_h + 7, max(dh for dh, _ in decoy_dims) + 7)
    min_w = max(12, inner_w + 7, max(dw for _, dw in decoy_dims) + 7)
    H = random.randint(min_h, min(30, min_h + 8))
    W = random.randint(min_w, min(30, min_w + 8))

    used_boxes: list[BBox] = []
    used_rows: set[int] = set()
    used_cols: set[int] = set()

    target_choices = _rectangle_candidates(H, W, inner_h, inner_w, used_boxes, used_rows, used_cols)
    if not target_choices:
        return None
    target_bbox = target_choices[0]
    used_boxes.append(target_bbox)
    used_rows.update((int(target_bbox[0]), int(target_bbox[2])))
    used_cols.update((int(target_bbox[1]), int(target_bbox[3])))

    decoy_boxes: list[BBox] = []
    for decoy_h, decoy_w in sorted(decoy_dims, key=lambda dims: (dims[0] + 2) * (dims[1] + 2), reverse=True):
        choices = _rectangle_candidates(H, W, decoy_h, decoy_w, used_boxes, used_rows, used_cols)
        if not choices:
            return None
        bbox = choices[0]
        decoy_boxes.append(bbox)
        used_boxes.append(bbox)
        used_rows.update((int(bbox[0]), int(bbox[2])))
        used_cols.update((int(bbox[1]), int(bbox[3])))

    source_choices = _source_candidates(H, W, inner_h, inner_w, used_boxes)
    if not source_choices:
        return None
    source_pos = source_choices[0]

    colors = random.sample(range(1, 10), 2)
    obj_color, marker_color = int(colors[0]), int(colors[1])

    grid = np.zeros((H, W), dtype=int)
    _paint_mask(grid, obj_mask, source_pos[0], source_pos[1], obj_color)

    marker_coords: set[Coord] = set()
    target_corners = _corners_from_bbox(target_bbox)
    target_missing = random.choice(target_corners)
    marker_coords.update(coord for coord in target_corners if coord != target_missing)

    for bbox in decoy_boxes:
        corners = _corners_from_bbox(bbox)
        missing = random.choice(corners)
        marker_coords.update(coord for coord in corners if coord != missing)

    for r, c in marker_coords:
        if int(grid[r, c]) != 0:
            return None
        grid[r, c] = marker_color

    marker_list = sorted(marker_coords)
    correct_matches = _matching_rectangles(marker_list, inner_h, inner_w)
    all_rectangles = _all_three_corner_rectangles(marker_list)
    wrong_matches = [
        rect
        for rect in all_rectangles
        if (int(rect["bbox"][2] - rect["bbox"][0] - 1), int(rect["bbox"][3] - rect["bbox"][1] - 1)) != (inner_h, inner_w)
    ]
    if len(correct_matches) != 1 or not wrong_matches:
        return None

    components = _find_components(grid, bg=0)
    if len([comp for comp in components if int(comp["area"]) > 1]) != 1:
        return None
    if any(int(comp["area"]) != 1 for comp in components if int(comp["color"]) == marker_color):
        return None

    expected = _render_expected_output(grid, obj_mask, obj_color, source_pos, target_bbox, marker_color, target_missing)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 1")
