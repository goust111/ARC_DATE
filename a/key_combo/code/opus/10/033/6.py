from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
YELLOW = 4
ORANGE = 7
TEAL = 8

EDGE_RETRY_LIMIT = 120
INTERNAL_RETRY_LIMIT = 180
GEN_RETRY_LIMIT = 400


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def touching_edge_side(coords: np.ndarray, shape: tuple[int, int]) -> str | None:
    h, w = shape
    rows = coords[:, 0]
    cols = coords[:, 1]
    sides = []
    if np.any(rows == 0):
        sides.append("top")
    if np.any(rows == h - 1):
        sides.append("bottom")
    if np.any(cols == 0):
        sides.append("left")
    if np.any(cols == w - 1):
        sides.append("right")
    if not sides:
        return None
    assert len(sides) == 1, "touching-edge objects must touch exactly one edge"
    return sides[0]


def contiguous_span(indices: np.ndarray) -> tuple[int, int]:
    assert indices.size > 0, "expected a non-empty span"
    lo = int(indices[0])
    hi = int(indices[-1])
    assert np.array_equal(indices, np.arange(lo, hi + 1)), "edge run must be contiguous"
    return lo, hi


def centered_span(lo: int, hi: int, length: int) -> tuple[int, int]:
    center2 = int(lo) + int(hi)
    assert length > 0
    assert (center2 - (length - 1)) % 2 == 0
    out_lo = (center2 - (length - 1)) // 2
    out_hi = (center2 + (length - 1)) // 2
    return int(out_lo), int(out_hi)


def paint_shrinking_rows(mask: np.ndarray, row: int, step: int, c0: int, c1: int) -> None:
    width = int(c1) - int(c0) + 1
    dist = 1
    length = width - 2
    while length > 0:
        lo, hi = centered_span(c0, c1, length)
        rr = int(row) + int(step) * dist
        if 0 <= rr < mask.shape[0]:
            mask[rr, max(0, lo) : min(mask.shape[1] - 1, hi) + 1] = True
        dist += 1
        length -= 2


def paint_shrinking_cols(mask: np.ndarray, col: int, step: int, r0: int, r1: int) -> None:
    height = int(r1) - int(r0) + 1
    dist = 1
    length = height - 2
    while length > 0:
        lo, hi = centered_span(r0, r1, length)
        cc = int(col) + int(step) * dist
        if 0 <= cc < mask.shape[1]:
            mask[max(0, lo) : min(mask.shape[0] - 1, hi) + 1, cc] = True
        dist += 1
        length -= 2


def shrinking_fill_mask(coords: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    obj_mask = np.zeros((h, w), dtype=bool)
    obj_mask[coords[:, 0], coords[:, 1]] = True
    r0, c0, r1, c1 = bbox_coords(coords)

    top_c0, top_c1 = contiguous_span(np.flatnonzero(obj_mask[r0, :]))
    bot_c0, bot_c1 = contiguous_span(np.flatnonzero(obj_mask[r1, :]))
    left_r0, left_r1 = contiguous_span(np.flatnonzero(obj_mask[:, c0]))
    right_r0, right_r1 = contiguous_span(np.flatnonzero(obj_mask[:, c1]))

    ext = np.zeros((h, w), dtype=bool)
    paint_shrinking_rows(ext, r0, -1, top_c0, top_c1)
    paint_shrinking_rows(ext, r1, 1, bot_c0, bot_c1)
    paint_shrinking_cols(ext, c0, -1, left_r0, left_r1)
    paint_shrinking_cols(ext, c1, 1, right_r0, right_r1)
    ext &= ~obj_mask
    return ext


def rectangle_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = bbox
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def moving_object_masks(
    coords: np.ndarray, shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    h, w = shape
    side = touching_edge_side(coords, shape)
    assert side is not None
    r0, c0, r1, c1 = bbox_coords(coords)
    rect_h = r1 - r0 + 1
    rect_w = c1 - c0 + 1
    assert coords.shape[0] == rect_h * rect_w, "edge-touching object must be a solid rectangle"

    if side == "top":
        dst = (h - rect_h, c0, h - 1, c1)
        trail = np.zeros(shape, dtype=bool)
        trail[0 : h - rect_h, c0 : c1 + 1] = True
    elif side == "bottom":
        dst = (0, c0, rect_h - 1, c1)
        trail = np.zeros(shape, dtype=bool)
        trail[rect_h:h, c0 : c1 + 1] = True
    elif side == "left":
        dst = (r0, w - rect_w, r1, w - 1)
        trail = np.zeros(shape, dtype=bool)
        trail[r0 : r1 + 1, 0 : w - rect_w] = True
    else:
        assert side == "right"
        dst = (r0, 0, r1, rect_w - 1)
        trail = np.zeros(shape, dtype=bool)
        trail[r0 : r1 + 1, rect_w:w] = True
    return trail, rectangle_mask(shape, dst), dst


def _sorted_components(coords_list: list[np.ndarray]) -> list[np.ndarray]:
    return sorted(coords_list, key=lambda pts: tuple(map(int, pts.min(axis=0))))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split teal objects by edge contact: slide edge-touching rectangles across the grid and extend interior rounded objects with shrinking fills.

    Input structure:
    - The background is black.
    - All foreground pixels are teal and form 3-5 separate 4-connected objects.
    - Each object either touches exactly one grid edge as a compact rectangle, or stays fully interior with contiguous top/bottom/left/right side runs.

    Rule:
    - Classify each teal object by whether it touches a grid edge.
    - Interior objects stay in place and emit yellow shrinking fills from their top, bottom, left, and right outer side runs, reducing the run length by 2 each step.
    - Edge-touching rectangles are cleared from their source, slide straight to the opposite edge, and leave an orange swept trail excluding the final rectangle.
    - Combine all results with priority: moved teal rectangles, then stationary teal interiors, then orange trails, then yellow fills, over black background.

    Key insight:
    - The same teal object type splits into two transformation pipelines based only on the spatial predicate "touches the outer border", and orange slide trails can overwrite yellow shrinking fills where they cross.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    fg_mask = g == TEAL
    comps = _sorted_components(components_mask(fg_mask))
    assert 3 <= len(comps) <= 5, "expected 3-5 teal objects"

    out = np.zeros((h, w), dtype=int)
    moving: list[np.ndarray] = []
    stationary: list[np.ndarray] = []

    for coords in comps:
        side = touching_edge_side(coords, g.shape)
        if side is None:
            stationary.append(coords)
        else:
            moving.append(coords)

    assert moving and stationary, "expected both moving and stationary objects"

    for coords in stationary:
        ext = shrinking_fill_mask(coords, g.shape)
        out[ext] = YELLOW

    for coords in moving:
        trail, _dst_mask, _dst_bbox = moving_object_masks(coords, g.shape)
        out[trail] = ORANGE

    for coords in stationary:
        out[coords[:, 0], coords[:, 1]] = TEAL

    for coords in moving:
        _trail, dst_mask, _dst_bbox = moving_object_masks(coords, g.shape)
        out[dst_mask] = TEAL

    return out


def rounded_mask(height: int, width: int, radius: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    for r in range(height):
        edge_distance = min(r, height - 1 - r, radius)
        inset = radius - edge_distance
        mask[r, inset : width - inset] = True
    return mask


def dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    pts = np.argwhere(mask)
    for r, c in pts:
        rr = int(r)
        cc = int(c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr, nc] = True
    return out


def bbox_from_mask_at(mask: np.ndarray, r0: int, c0: int) -> tuple[int, int, int, int]:
    h, w = mask.shape
    return int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)


def placed_mask(shape: tuple[int, int], mask: np.ndarray, r0: int, c0: int) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    h, w = mask.shape
    out[r0 : r0 + h, c0 : c0 + w] = mask
    return out


def mask_coords(mask: np.ndarray) -> np.ndarray:
    return np.argwhere(mask).astype(int)


def sample_internal_shape(for_conflict: bool) -> np.ndarray:
    if for_conflict:
        radius = random.choice([1, 1, 2])
        min_h = 2 * radius + 3
        min_w = 2 * radius + 3
        height = random.randint(min_h, min_h + 2)
        width = random.randint(min_w, min_w + 3)
    else:
        radius = random.choice([1, 1, 2])
        min_h = 2 * radius + 3
        min_w = 2 * radius + 3
        height = random.randint(min_h, min(min_h + 3, 8))
        width = random.randint(min_w, min(min_w + 4, 9))
    return rounded_mask(height, width, radius)


def interval_overlaps(a0: int, a1: int, b0: int, b1: int, gap: int = 0) -> bool:
    return not (a1 + gap < b0 or b1 + gap < a0)


def build_vertical_edge_object(
    shape: tuple[int, int],
    used_cols: list[tuple[int, int]],
    require_conflict_space: bool,
) -> dict | None:
    h, w = shape
    for _ in range(EDGE_RETRY_LIMIT):
        rect_h = random.randint(2, min(4, max(2, h // 4)))
        rect_w = random.randint(2, 3)
        c0_lo = 1
        c0_hi = w - rect_w - 1
        if require_conflict_space:
            c0_lo = 4
            c0_hi = w - rect_w - 4
        if c0_lo > c0_hi:
            return None
        c0 = random.randint(c0_lo, c0_hi)
        c1 = c0 + rect_w - 1
        if any(interval_overlaps(c0, c1, u0, u1, gap=1) for u0, u1 in used_cols):
            continue
        side = random.choice(["top", "bottom"])
        if side == "top":
            src = (0, c0, rect_h - 1, c1)
            dst = (h - rect_h, c0, h - 1, c1)
            trail = np.zeros(shape, dtype=bool)
            trail[0 : h - rect_h, c0 : c1 + 1] = True
        else:
            src = (h - rect_h, c0, h - 1, c1)
            dst = (0, c0, rect_h - 1, c1)
            trail = np.zeros(shape, dtype=bool)
            trail[rect_h:h, c0 : c1 + 1] = True
        return {
            "kind": "edge",
            "axis": "vertical",
            "side": side,
            "src_bbox": src,
            "dst_bbox": dst,
            "trail_mask": trail,
            "reserved_mask": trail | rectangle_mask(shape, src) | rectangle_mask(shape, dst),
            "object_mask": rectangle_mask(shape, src),
        }
    return None


def build_horizontal_edge_object(
    shape: tuple[int, int],
    used_rows: list[tuple[int, int]],
    require_conflict_space: bool,
) -> dict | None:
    h, w = shape
    for _ in range(EDGE_RETRY_LIMIT):
        rect_h = random.randint(2, 3)
        rect_w = random.randint(2, min(4, max(2, w // 4)))
        r0_lo = 1
        r0_hi = h - rect_h - 1
        if require_conflict_space:
            r0_lo = 4
            r0_hi = h - rect_h - 4
        if r0_lo > r0_hi:
            return None
        r0 = random.randint(r0_lo, r0_hi)
        r1 = r0 + rect_h - 1
        if any(interval_overlaps(r0, r1, u0, u1, gap=1) for u0, u1 in used_rows):
            continue
        side = random.choice(["left", "right"])
        if side == "left":
            src = (r0, 0, r1, rect_w - 1)
            dst = (r0, w - rect_w, r1, w - 1)
            trail = np.zeros(shape, dtype=bool)
            trail[r0 : r1 + 1, 0 : w - rect_w] = True
        else:
            src = (r0, w - rect_w, r1, w - 1)
            dst = (r0, 0, r1, rect_w - 1)
            trail = np.zeros(shape, dtype=bool)
            trail[r0 : r1 + 1, rect_w:w] = True
        return {
            "kind": "edge",
            "axis": "horizontal",
            "side": side,
            "src_bbox": src,
            "dst_bbox": dst,
            "trail_mask": trail,
            "reserved_mask": trail | rectangle_mask(shape, src) | rectangle_mask(shape, dst),
            "object_mask": rectangle_mask(shape, src),
        }
    return None


def place_conflict_internal(shape: tuple[int, int], edge_obj: dict, blocked: np.ndarray) -> dict | None:
    h, w = shape
    for _ in range(INTERNAL_RETRY_LIMIT):
        local = sample_internal_shape(for_conflict=True)
        lh, lw = local.shape
        if edge_obj["axis"] == "vertical":
            er0, ec0, er1, ec1 = edge_obj["src_bbox"]
            place_left = random.choice([True, False])
            options = [place_left, not place_left]
            for use_left in options:
                c0 = ec0 - lw if use_left else ec1 + 1
                if not (1 <= c0 and c0 + lw <= w - 1):
                    continue
                if edge_obj["side"] == "top":
                    r_lo = 2
                    r_hi = max(r_lo, h - lh - edge_obj["src_bbox"][2] - 2)
                else:
                    rect_h = edge_obj["src_bbox"][2] - edge_obj["src_bbox"][0] + 1
                    r_lo = rect_h + 1
                    r_hi = h - lh - 2
                if r_lo > r_hi:
                    continue
                r0 = random.randint(r_lo, r_hi)
                placed = placed_mask(shape, local, r0, c0)
                if np.any(placed & blocked):
                    continue
                return {
                    "kind": "internal",
                    "local_mask": local,
                    "object_mask": placed,
                    "bbox": bbox_from_mask_at(local, r0, c0),
                }
        else:
            er0, ec0, er1, ec1 = edge_obj["src_bbox"]
            place_up = random.choice([True, False])
            options = [place_up, not place_up]
            for use_up in options:
                r0 = er0 - lh if use_up else er1 + 1
                if not (1 <= r0 and r0 + lh <= h - 1):
                    continue
                if edge_obj["side"] == "left":
                    c_lo = 2
                    c_hi = max(c_lo, w - lw - (edge_obj["src_bbox"][3] - edge_obj["src_bbox"][1]) - 2)
                else:
                    rect_w = edge_obj["src_bbox"][3] - edge_obj["src_bbox"][1] + 1
                    c_lo = rect_w + 1
                    c_hi = w - lw - 2
                if c_lo > c_hi:
                    continue
                c0 = random.randint(c_lo, c_hi)
                placed = placed_mask(shape, local, r0, c0)
                if np.any(placed & blocked):
                    continue
                return {
                    "kind": "internal",
                    "local_mask": local,
                    "object_mask": placed,
                    "bbox": bbox_from_mask_at(local, r0, c0),
                }
    return None


def place_random_internal(shape: tuple[int, int], blocked: np.ndarray) -> dict | None:
    h, w = shape
    for _ in range(INTERNAL_RETRY_LIMIT):
        local = sample_internal_shape(for_conflict=False)
        lh, lw = local.shape
        if h - lh - 1 < 1 or w - lw - 1 < 1:
            continue
        r0 = random.randint(1, h - lh - 1)
        c0 = random.randint(1, w - lw - 1)
        placed = placed_mask(shape, local, r0, c0)
        if np.any(placed & blocked):
            continue
        return {
            "kind": "internal",
            "local_mask": local,
            "object_mask": placed,
            "bbox": bbox_from_mask_at(local, r0, c0),
        }
    return None


def classify_objects(grid: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    comps = _sorted_components(components_mask(grid == TEAL))
    edge, internal = [], []
    for coords in comps:
        if touching_edge_side(coords, grid.shape) is None:
            internal.append(coords)
        else:
            edge.append(coords)
    return edge, internal


def semantic_checks(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    edge_objs, internal_objs = classify_objects(grid)
    if not edge_objs or not internal_objs:
        return False
    if len(edge_objs) + len(internal_objs) < 3 or len(edge_objs) + len(internal_objs) > 5:
        return False

    all_fg = np.zeros(grid.shape, dtype=bool)
    for coords in edge_objs:
        r0, c0, r1, c1 = bbox_coords(coords)
        if coords.shape[0] != (r1 - r0 + 1) * (c1 - c0 + 1):
            return False
        if min(r1 - r0 + 1, c1 - c0 + 1) < 2:
            return False
        side = touching_edge_side(coords, grid.shape)
        if side is None:
            return False
        trail, dst_mask, _dst_bbox = moving_object_masks(coords, grid.shape)
        source_mask = np.zeros(grid.shape, dtype=bool)
        source_mask[coords[:, 0], coords[:, 1]] = True
        blockers = (grid == TEAL) & ~source_mask
        if np.any(trail & blockers) or np.any(dst_mask & blockers):
            return False
        all_fg |= source_mask

    ext_union = np.zeros(grid.shape, dtype=bool)
    trail_union = np.zeros(grid.shape, dtype=bool)
    conflict = np.zeros(grid.shape, dtype=bool)
    for coords in internal_objs:
        side = touching_edge_side(coords, grid.shape)
        if side is not None:
            return False
        ext = shrinking_fill_mask(coords, grid.shape)
        if not np.any(ext):
            return False
        obj_mask = np.zeros(grid.shape, dtype=bool)
        obj_mask[coords[:, 0], coords[:, 1]] = True
        top = np.flatnonzero(obj_mask[bbox_coords(coords)[0], :])
        bottom = np.flatnonzero(obj_mask[bbox_coords(coords)[2], :])
        left = np.flatnonzero(obj_mask[:, bbox_coords(coords)[1]])
        right = np.flatnonzero(obj_mask[:, bbox_coords(coords)[3]])
        if min(len(top), len(bottom), len(left), len(right)) < 3:
            return False
        ext_union |= ext
        all_fg |= obj_mask

    for coords in edge_objs:
        trail, _dst_mask, _dst_bbox = moving_object_masks(coords, grid.shape)
        conflict |= trail & ext_union
        trail_union |= trail

    if not np.any(conflict):
        return False

    out = main(grid)
    if not ({TEAL, ORANGE, YELLOW} <= set(map(int, np.unique(out)))):
        return False
    if not np.all(out[conflict] == ORANGE):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        total = random.randint(3, 5)
        edge_count = random.randint(1, min(3, total - 1))
        internal_count = total - edge_count

        axis = random.choice(["vertical", "horizontal"])
        if axis == "vertical":
            h = random.randint(max(13, 8 + internal_count), 20)
            w = random.randint(max(13, 7 + 3 * edge_count), 20)
        else:
            h = random.randint(max(13, 7 + 3 * edge_count), 20)
            w = random.randint(max(13, 8 + internal_count), 20)
        shape = (h, w)

        edge_specs: list[dict] = []
        used_cols: list[tuple[int, int]] = []
        used_rows: list[tuple[int, int]] = []

        ok = True
        for i in range(edge_count):
            if axis == "vertical":
                spec = build_vertical_edge_object(shape, used_cols, require_conflict_space=(i == 0))
                if spec is None:
                    ok = False
                    break
                c0 = spec["src_bbox"][1]
                c1 = spec["src_bbox"][3]
                used_cols.append((c0, c1))
            else:
                spec = build_horizontal_edge_object(shape, used_rows, require_conflict_space=(i == 0))
                if spec is None:
                    ok = False
                    break
                r0 = spec["src_bbox"][0]
                r1 = spec["src_bbox"][2]
                used_rows.append((r0, r1))
            edge_specs.append(spec)
        if not ok:
            continue

        blocked = np.zeros(shape, dtype=bool)
        for spec in edge_specs:
            blocked |= dilate8(spec["object_mask"])
            blocked |= spec["trail_mask"]
            blocked |= spec["dst_bbox"] and dilate8(rectangle_mask(shape, spec["dst_bbox"]))

        internal_specs: list[dict] = []
        conflict_internal = place_conflict_internal(shape, edge_specs[0], blocked)
        if conflict_internal is None:
            continue
        internal_specs.append(conflict_internal)
        blocked |= dilate8(conflict_internal["object_mask"])

        for _k in range(internal_count - 1):
            spec = place_random_internal(shape, blocked)
            if spec is None:
                ok = False
                break
            internal_specs.append(spec)
            blocked |= dilate8(spec["object_mask"])
        if not ok:
            continue

        grid = np.zeros(shape, dtype=int)
        for spec in edge_specs:
            grid[spec["object_mask"]] = TEAL
        for spec in internal_specs:
            grid[spec["object_mask"]] = TEAL

        if semantic_checks(grid):
            return grid

    raise RuntimeError("failed to generate a valid input")
