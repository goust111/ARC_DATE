from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BG = 0
FRAGMENT = 8
RANK_COLORS = (1, 2, 3, 4)
GEN_ATTEMPTS = 600
LOCAL_ATTEMPTS = 120
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _neighbors4(r: int, c: int):
    for dr, dc in NEIGHBORS4:
        yield r + dr, c + dc


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _spans_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(int(a0), int(b0)) <= min(int(a1), int(b1))


def _bboxes_share_row_or_col(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return _spans_overlap(ar0, ar1, br0, br1) or _spans_overlap(ac0, ac1, bc0, bc1)


def _find_fragments(mask: np.ndarray) -> list[dict]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    fragments: list[dict] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            fragments.append(
                {
                    "coords": arr,
                    "bbox": _bbox_from_coords(arr),
                    "area": int(arr.shape[0]),
                }
            )
    return fragments


def _merge_fragment_groups(fragments: list[dict]) -> list[dict]:
    parent = list(range(len(fragments)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(fragments)):
        for j in range(i + 1, len(fragments)):
            if _bboxes_share_row_or_col(fragments[i]["bbox"], fragments[j]["bbox"]):
                union(i, j)

    grouped: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(fragments)):
        grouped[find(idx)].append(idx)

    groups: list[dict] = []
    for member_ids in grouped.values():
        coords = np.vstack([fragments[idx]["coords"] for idx in member_ids])
        groups.append(
            {
                "coords": coords,
                "bbox": _bbox_from_coords(coords),
                "fragments": [fragments[idx] for idx in member_ids],
                "fragment_count": len(member_ids),
                "pixel_count": int(coords.shape[0]),
            }
        )
    return groups


def _outer_adjacent_cells(
    bbox: tuple[int, int, int, int], shape: tuple[int, int]
) -> list[tuple[int, int]]:
    h, w = shape
    r0, c0, r1, c1 = bbox
    cells = []
    if r0 - 1 >= 0:
        for c in range(c0, c1 + 1):
            cells.append((r0 - 1, c))
    if r1 + 1 < h:
        for c in range(c0, c1 + 1):
            cells.append((r1 + 1, c))
    if c0 - 1 >= 0:
        for r in range(r0, r1 + 1):
            cells.append((r, c0 - 1))
    if c1 + 1 < w:
        for r in range(r0, r1 + 1):
            cells.append((r, c1 + 1))
    return cells


def _ranked_groups(groups: list[dict], key_name: str = "area") -> list[dict]:
    enriched = []
    for group in groups:
        r0, c0, r1, c1 = group["bbox"]
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        area = height * width
        perimeter = 2 * (height + width)
        enriched.append(
            {
                **group,
                "height": height,
                "width": width,
                "area": area,
                "perimeter": perimeter,
            }
        )
    return sorted(enriched, key=lambda g: (-int(g[key_name]), g["bbox"]))


def _render_from_groups(groups: list[dict], shape: tuple[int, int], key_name: str = "area") -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    ordered = _ranked_groups(groups, key_name=key_name)
    for rank, group in enumerate(ordered):
        color = RANK_COLORS[rank]
        for r, c in _outer_adjacent_cells(group["bbox"], shape):
            if out[r, c] == BG:
                out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge teal fragments into full rectangles, erase them, and color each rectangle's outside contour by area rank.

    Input structure:
    - The grid is mostly black background with 3-4 groups of teal fragments.
    - Teal cells form several 4-connected fragments; fragments from the same hidden rectangle
      may be disconnected but have overlapping row spans or column spans in their bboxes.
    - Distinct rectangle groups do not share any row span or any column span, and the completed
      rectangle areas are all different.

    Rule:
    - Find every 4-connected teal fragment.
    - Merge fragments transitively whenever their bounding boxes overlap in row span or in column span.
    - Treat each merged group's bbox as one complete rectangle, rank those rectangles by bbox area
      from largest to smallest, and map ranks to colors blue, red, green, yellow.
    - Erase all original teal cells and paint only the outside 4-neighbor cells adjacent to each
      rectangle boundary with that rectangle's rank color; do not color the rectangle interior or boundary.

    Key insight:
    - The visible teal pieces are not the target objects: they must first be merged into hidden rectangles,
      and only those reconstructed rectangle bboxes determine both the contour location and the ranking colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    fragments = _find_fragments(g == FRAGMENT)
    assert fragments, "expected teal fragments"

    groups = _merge_fragment_groups(fragments)
    return _render_from_groups(groups, g.shape, key_name="area")


def _fill_box(mask: np.ndarray, box: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = box
    mask[r0 : r1 + 1, c0 : c1 + 1] = True


def _corner_chain_mask(h: int, w: int) -> np.ndarray | None:
    max_h = min(3, h - 2)
    max_w = min(3, w - 2)
    if max_h < 1 or max_w < 1:
        return None

    tl_h = random.randint(1, max_h)
    tl_w = random.randint(1, max_w)
    tr_h = random.randint(1, max_h)
    tr_w = random.randint(1, max_w)
    br_h = random.randint(1, max_h)
    br_w = random.randint(1, max_w)

    if tl_w + tr_w > w - 1 or tr_h + br_h > h - 1:
        return None

    mask = np.zeros((h, w), dtype=bool)
    _fill_box(mask, (0, 0, tl_h - 1, tl_w - 1))
    _fill_box(mask, (0, w - tr_w, tr_h - 1, w - 1))
    _fill_box(mask, (h - br_h, w - br_w, h - 1, w - 1))

    if random.choice((False, True)):
        mask = np.fliplr(mask)
    if random.choice((False, True)):
        mask = np.flipud(mask)
    return mask


def _four_corners_mask(h: int, w: int) -> np.ndarray | None:
    max_h = min(2, h - 2)
    max_w = min(2, w - 2)
    if max_h < 1 or max_w < 1:
        return None

    dims = {
        "tl": (random.randint(1, max_h), random.randint(1, max_w)),
        "tr": (random.randint(1, max_h), random.randint(1, max_w)),
        "bl": (random.randint(1, max_h), random.randint(1, max_w)),
        "br": (random.randint(1, max_h), random.randint(1, max_w)),
    }
    if dims["tl"][1] + dims["tr"][1] > w - 1:
        return None
    if dims["bl"][1] + dims["br"][1] > w - 1:
        return None
    if dims["tl"][0] + dims["bl"][0] > h - 1:
        return None
    if dims["tr"][0] + dims["br"][0] > h - 1:
        return None

    mask = np.zeros((h, w), dtype=bool)
    _fill_box(mask, (0, 0, dims["tl"][0] - 1, dims["tl"][1] - 1))
    _fill_box(mask, (0, w - dims["tr"][1], dims["tr"][0] - 1, w - 1))
    _fill_box(mask, (h - dims["bl"][0], 0, h - 1, dims["bl"][1] - 1))
    _fill_box(mask, (h - dims["br"][0], w - dims["br"][1], h - 1, w - 1))
    return mask


def _top_bottom_center_mask(h: int, w: int) -> np.ndarray | None:
    if h < 5 or w < 4:
        return None

    top_h = random.randint(1, min(2, h - 3))
    bottom_h_max = min(2, h - top_h - 2)
    if bottom_h_max < 1:
        return None
    bottom_h = random.randint(1, bottom_h_max)

    min_strip_w = max(2, (w + 1) // 2)
    top_w = random.randint(min_strip_w, w)
    bottom_w = random.randint(min_strip_w, w)
    if top_w + bottom_w < w + 1:
        return None

    overlap_start = w - bottom_w
    overlap_end = top_w - 1
    if overlap_start > overlap_end:
        return None

    center_w = random.randint(1, min(2, overlap_end - overlap_start + 1))
    center_c0 = random.randint(overlap_start, overlap_end - center_w + 1)
    center_r0_lo = top_h + 1
    center_r0_hi = h - bottom_h - 2
    if center_r0_lo > center_r0_hi:
        return None
    center_r0 = random.randint(center_r0_lo, center_r0_hi)
    center_r1 = random.randint(center_r0, center_r0_hi)

    mask = np.zeros((h, w), dtype=bool)
    _fill_box(mask, (0, 0, top_h - 1, top_w - 1))
    _fill_box(mask, (h - bottom_h, w - bottom_w, h - 1, w - 1))
    _fill_box(mask, (center_r0, center_c0, center_r1, center_c0 + center_w - 1))

    if random.choice((False, True)):
        mask = np.fliplr(mask)
    return mask


def _left_right_center_mask(h: int, w: int) -> np.ndarray | None:
    if h < 4 or w < 5:
        return None

    left_w = random.randint(1, min(2, w - 3))
    right_w_max = min(2, w - left_w - 2)
    if right_w_max < 1:
        return None
    right_w = random.randint(1, right_w_max)
    if w - left_w - right_w < 3:
        return None

    min_strip_h = max(2, (h + 1) // 2)
    left_h = random.randint(min_strip_h, h)
    right_h = random.randint(min_strip_h, h)
    if left_h + right_h < h + 1:
        return None

    overlap_start = h - right_h
    overlap_end = left_h - 1
    if overlap_start > overlap_end:
        return None

    center_h = random.randint(1, min(2, overlap_end - overlap_start + 1))
    center_r0 = random.randint(overlap_start, overlap_end - center_h + 1)
    center_c0_lo = left_w + 1
    center_c0_hi = w - right_w - 2
    if center_c0_lo > center_c0_hi:
        return None
    center_c0 = random.randint(center_c0_lo, center_c0_hi)
    center_c1 = random.randint(center_c0, center_c0_hi)

    mask = np.zeros((h, w), dtype=bool)
    _fill_box(mask, (0, 0, left_h - 1, left_w - 1))
    _fill_box(mask, (h - right_h, w - right_w, h - 1, w - 1))
    _fill_box(mask, (center_r0, center_c0, center_r0 + center_h - 1, center_c1))

    if random.choice((False, True)):
        mask = np.flipud(mask)
    return mask


def _transitive_merge_needed(fragments: list[dict]) -> bool:
    if len(fragments) < 3:
        return False
    for i in range(len(fragments)):
        for j in range(i + 1, len(fragments)):
            if not _bboxes_share_row_or_col(fragments[i]["bbox"], fragments[j]["bbox"]):
                return True
    return False


def _build_fragment_mask(h: int, w: int) -> np.ndarray | None:
    builders = [_corner_chain_mask, _four_corners_mask, _top_bottom_center_mask, _left_right_center_mask]
    for _ in range(LOCAL_ATTEMPTS):
        builder = random.choice(builders)
        mask = builder(h, w)
        if mask is None:
            continue
        fragments = _find_fragments(mask)
        if len(fragments) < 3:
            continue
        merged = _merge_fragment_groups(fragments)
        if len(merged) != 1:
            continue
        if merged[0]["bbox"] != (0, 0, h - 1, w - 1):
            continue
        coverage = float(mask.sum()) / float(h * w)
        if not (0.18 <= coverage <= 0.68):
            continue
        if not _transitive_merge_needed(fragments):
            continue
        return mask
    return None


def _generate_rectangles() -> tuple[list[dict], tuple[int, int]] | None:
    count = random.choice((3, 4))
    if count == 3:
        h_range = (4, 6)
        w_range = (4, 7)
    else:
        h_range = (4, 5)
        w_range = (4, 5)

    top_pad = random.randint(1, 3)
    left_pad = random.randint(1, 3)
    bottom_pad = random.randint(1, 3)
    right_pad = random.randint(1, 3)
    row_gaps = [random.randint(2, 4) for _ in range(count - 1)]
    col_gaps = [random.randint(2, 4) for _ in range(count - 1)]
    heights = [random.randint(*h_range) for _ in range(count)]
    widths = [random.randint(*w_range) for _ in range(count)]
    areas = [heights[i] * widths[i] for i in range(count)]

    if len(set(areas)) != count:
        return None

    rects = []
    r = top_pad
    c = left_pad
    for idx in range(count):
        h = heights[idx]
        w = widths[idx]
        rects.append({"bbox": (r, c, r + h - 1, c + w - 1), "height": h, "width": w})
        if idx < count - 1:
            r += h + row_gaps[idx]
            c += w + col_gaps[idx]

    grid_h = rects[-1]["bbox"][2] + 1 + bottom_pad
    grid_w = rects[-1]["bbox"][3] + 1 + right_pad
    if max(grid_h, grid_w) > 30 or min(grid_h, grid_w) < 16:
        return None

    for rect in rects:
        r0, c0, r1, c1 = rect["bbox"]
        rect["area"] = (r1 - r0 + 1) * (c1 - c0 + 1)
        rect["perimeter"] = 2 * ((r1 - r0 + 1) + (c1 - c0 + 1))

    area_order = sorted(range(count), key=lambda i: (-rects[i]["area"], i))
    height_order = sorted(range(count), key=lambda i: (-rects[i]["height"], i))
    width_order = sorted(range(count), key=lambda i: (-rects[i]["width"], i))
    perimeter_order = sorted(range(count), key=lambda i: (-rects[i]["perimeter"], i))
    if area_order == height_order or area_order == width_order or area_order == perimeter_order:
        return None

    return rects, (grid_h, grid_w)


def _validate_grid(grid: np.ndarray, intended_rects: list[dict]) -> bool:
    try:
        fragments = _find_fragments(grid == FRAGMENT)
        groups = _merge_fragment_groups(fragments)
    except Exception:
        return False

    if len(groups) != len(intended_rects):
        return False

    solved = main(grid)
    area_out = _render_from_groups(groups, grid.shape, key_name="area")
    if not np.array_equal(solved, area_out):
        return False

    expected_bboxes = sorted(rect["bbox"] for rect in intended_rects)
    found_bboxes = sorted(group["bbox"] for group in groups)
    if expected_bboxes != found_bboxes:
        return False

    areas = []
    pixel_counts = []
    perimeters = []
    heights = []
    widths = []
    contour_cells: set[tuple[int, int]] = set()
    has_transitive_group = False

    for group in groups:
        if group["fragment_count"] < 3:
            return False
        if not _transitive_merge_needed(group["fragments"]):
            return False
        has_transitive_group = True

        bbox = group["bbox"]
        r0, c0, r1, c1 = bbox
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        heights.append(height)
        widths.append(width)
        areas.append(height * width)
        perimeters.append(2 * (height + width))
        pixel_counts.append(group["pixel_count"])

        for frag in group["fragments"]:
            if frag["bbox"] == bbox:
                return False

        local_cells = _outer_adjacent_cells(bbox, grid.shape)
        for cell in local_cells:
            if cell in contour_cells:
                return False
            contour_cells.add(cell)

    if not has_transitive_group:
        return False
    if len(set(areas)) != len(areas):
        return False

    count = len(groups)
    area_order = sorted(range(count), key=lambda i: (-areas[i], i))
    alt_orders = [
        sorted(range(count), key=lambda i: (-heights[i], i)),
        sorted(range(count), key=lambda i: (-widths[i], i)),
        sorted(range(count), key=lambda i: (-perimeters[i], i)),
        sorted(range(count), key=lambda i: (-pixel_counts[i], i)),
    ]
    if any(area_order == alt for alt in alt_orders):
        return False

    if any(np.array_equal(solved, _render_from_groups(groups, grid.shape, key_name=key)) for key in ("height", "width", "perimeter", "pixel_count")):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        layout = _generate_rectangles()
        if layout is None:
            continue
        rects, shape = layout
        grid = np.zeros(shape, dtype=int)
        ok = True
        for rect in rects:
            h = rect["height"]
            w = rect["width"]
            mask = _build_fragment_mask(h, w)
            if mask is None:
                ok = False
                break
            r0, c0, r1, c1 = rect["bbox"]
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            patch[mask] = FRAGMENT
        if not ok:
            continue
        if _validate_grid(grid, rects):
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
