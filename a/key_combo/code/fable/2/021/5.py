from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


BG = 0
ALL_COLORS = tuple(range(1, 10))
GENERATION_ATTEMPTS = 400


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class MarkerRect:
    color: int
    bbox: BBox
    corners: tuple[Coord, Coord, Coord, Coord]


def _corners(rect: BBox) -> tuple[Coord, Coord, Coord, Coord]:
    r0, c0, r1, c1 = rect
    return ((r0, c0), (r0, c1), (r1, c0), (r1, c1))


def _rect_area(rect: BBox) -> int:
    r0, c0, r1, c1 = rect
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _inside(rect: BBox, cell: Coord) -> bool:
    r0, c0, r1, c1 = rect
    r, c = cell
    return r0 <= r <= r1 and c0 <= c <= c1


def _sort_rects(rects: list[BBox]) -> list[BBox]:
    return sorted(rects, key=lambda rect: (rect[0], rect[1], rect[2], rect[3]))


def _center2(rect: BBox) -> tuple[int, int]:
    r0, c0, r1, c1 = rect
    return r0 + r1, c0 + c1


def _overlap_bbox(a: BBox, b: BBox) -> BBox | None:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 > r1 or c0 > c1:
        return None
    return (r0, c0, r1, c1)


def _overlap_area(a: BBox, b: BBox) -> int:
    ov = _overlap_bbox(a, b)
    return 0 if ov is None else _rect_area(ov)


def _rotate_point(rect: BBox, cell: Coord) -> Coord:
    r0, c0, r1, c1 = rect
    r, c = cell
    return (r0 + r1 - r, c0 + c1 - c)


def _rotate_cells(rect: BBox, cells: tuple[Coord, ...]) -> tuple[Coord, ...]:
    return tuple(sorted(_rotate_point(rect, cell) for cell in cells))


def _find_marker_rectangles(grid: np.ndarray) -> list[MarkerRect]:
    g = np.asarray(grid, dtype=int)
    rects: list[MarkerRect] = []
    for color in sorted(int(color) for color in np.unique(g) if int(color) != BG):
        coords = np.argwhere(g == color)
        if coords.shape[0] != 4:
            continue
        rows = sorted({int(r) for r in coords[:, 0]})
        cols = sorted({int(c) for c in coords[:, 1]})
        if len(rows) != 2 or len(cols) != 2:
            continue
        if rows[0] == rows[1] or cols[0] == cols[1]:
            continue
        expected = set(_corners((rows[0], cols[0], rows[1], cols[1])))
        actual = {tuple(map(int, coord)) for coord in coords.tolist()}
        if actual != expected:
            continue
        rects.append(
            MarkerRect(
                color=color,
                bbox=(rows[0], cols[0], rows[1], cols[1]),
                corners=tuple(sorted(expected)),
            )
        )
    rects.sort(key=lambda item: (item.bbox[0], item.bbox[1], item.bbox[2], item.bbox[3]))
    assert rects, "expected at least one marker rectangle"
    top_lefts = {(rect.bbox[0], rect.bbox[1]) for rect in rects}
    assert len(top_lefts) == len(rects), "marker rectangles must have unique top-left corners"
    return rects


def _marker_mask(shape: tuple[int, int], rects: list[MarkerRect] | list[BBox]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if rects and isinstance(rects[0], MarkerRect):
        corners_iter = [rect.corners for rect in rects]  # type: ignore[arg-type]
    else:
        corners_iter = [_corners(rect) for rect in rects]  # type: ignore[arg-type]
    for corners in corners_iter:
        for r, c in corners:
            mask[r, c] = True
    return mask


def _apply_one_rotation(grid: np.ndarray, rect: BBox, marker_mask: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = rect
    out = np.array(grid, copy=True)
    region = grid[r0 : r1 + 1, c0 : c1 + 1]
    rotated = np.rot90(region, 2)
    marker_region = marker_mask[r0 : r1 + 1, c0 : c1 + 1]
    incoming_marker = np.rot90(marker_region, 2)
    dst = out[r0 : r1 + 1, c0 : c1 + 1]
    movable = ~marker_region
    dst[movable] = BG
    fill_mask = movable & ~incoming_marker
    dst[fill_mask] = rotated[fill_mask]
    return out


def _apply_one_rotation_slow(grid: np.ndarray, rect: BBox, marker_mask: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = rect
    out = np.array(grid, copy=True)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if marker_mask[r, c]:
                continue
            src = _rotate_point(rect, (r, c))
            out[r, c] = BG if marker_mask[src] else int(grid[src])
    return out


def _apply_order(grid: np.ndarray, rects: list[BBox], marker_mask: np.ndarray, *, slow: bool = False) -> np.ndarray:
    out = np.array(grid, copy=True)
    for rect in rects:
        out = _apply_one_rotation_slow(out, rect, marker_mask) if slow else _apply_one_rotation(out, rect, marker_mask)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sequentially 180-rotate each implied marker rectangle while keeping the corner markers fixed.

    Input structure:
    - A black background contains scattered colored single pixels and small colored fragments.
    - Some colors appear exactly four times, and those four same-color pixels are the unseen corners of one axis-aligned rectangle.
    - Different corner-marker groups use different colors, and the rectangles may partially overlap.

    Rule:
    - Identify every same-color four-corner rectangle and sort the rectangles by the row-major order of their top-left corners.
    - Process the rectangles in that order; for each one, 180-rotate the current contents of its closed rectangle back into the same rectangle.
    - Corner-marker pixels never move and are never treated as content, so only non-marker cells inside the rectangle are rotated; everything outside all rectangles stays unchanged.

    Key insight:
    Overlapping rectangles create a relay: two different 180-degree turns around different centers compose into a translation, so the prescribed top-left order is essential.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    rects = _find_marker_rectangles(g)
    marker_mask = _marker_mask(g.shape, rects)
    return _apply_order(g, [rect.bbox for rect in rects], marker_mask)


def _normalize_shape(cells: tuple[Coord, ...]) -> tuple[Coord, ...]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _rot90_shape(cells: tuple[Coord, ...]) -> tuple[Coord, ...]:
    return _normalize_shape(tuple((c, -r) for r, c in cells))


def _build_shapes() -> list[tuple[Coord, ...]]:
    base = [
        ((0, 0), (0, 1), (1, 0)),
        ((0, 0), (0, 1), (1, 1)),
        ((0, 0), (0, 1), (0, 2)),
        ((0, 0), (1, 0), (2, 0)),
        ((0, 0), (0, 1)),
        ((0, 0), (1, 0)),
        ((0, 0),),
    ]
    seen: set[tuple[Coord, ...]] = set()
    shapes: list[tuple[Coord, ...]] = []
    for shape in base:
        cur = _normalize_shape(shape)
        for _ in range(4):
            if cur not in seen:
                seen.add(cur)
                shapes.append(cur)
            cur = _rot90_shape(cur)
    return sorted(shapes, key=lambda shape: (-len(shape), shape))


SHAPES = _build_shapes()


def _pair_geometry_ok(a: BBox, b: BBox) -> bool:
    if _center2(a) == _center2(b):
        return False
    overlap = _overlap_bbox(a, b)
    if overlap is None:
        return True
    for corner in _corners(a):
        if _inside(b, corner):
            return False
    for corner in _corners(b):
        if _inside(a, corner):
            return False
    ov_area = _rect_area(overlap)
    if ov_area < 4:
        return False
    if ov_area * 100 > 45 * min(_rect_area(a), _rect_area(b)):
        return False
    return True


def _overlap_graph_connected(rects: list[BBox]) -> bool:
    seen = {0}
    stack = [0]
    while stack:
        i = stack.pop()
        for j in range(len(rects)):
            if j in seen or i == j:
                continue
            if _overlap_area(rects[i], rects[j]) > 0:
                seen.add(j)
                stack.append(j)
    return len(seen) == len(rects)


def _cover_counts(shape: tuple[int, int], rects: list[BBox]) -> np.ndarray:
    counts = np.zeros(shape, dtype=int)
    for r0, c0, r1, c1 in rects:
        counts[r0 : r1 + 1, c0 : c1 + 1] += 1
    return counts


def _stable_local_zone(index: int, rects: list[BBox], marker_mask: np.ndarray, cover: np.ndarray) -> set[Coord]:
    rect = rects[index]
    later = rects[index + 1 :]
    zone: set[Coord] = set()
    for r in range(rect[0], rect[2] + 1):
        for c in range(rect[1], rect[3] + 1):
            if marker_mask[r, c] or cover[r, c] != 1:
                continue
            dst = _rotate_point(rect, (r, c))
            if marker_mask[dst]:
                continue
            if any(_inside(other, dst) for other in later):
                continue
            zone.add((r, c))
    return zone


def _relay_zone(a: BBox, b: BBox, marker_mask: np.ndarray) -> set[Coord]:
    overlap = _overlap_bbox(a, b)
    if overlap is None:
        return set()
    zone: set[Coord] = set()
    r0, c0, r1, c1 = overlap
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if marker_mask[r, c]:
                continue
            p = (r, c)
            after_a = _rotate_point(a, p)
            after_b = _rotate_point(b, p)
            if marker_mask[after_a] or marker_mask[after_b]:
                continue
            if not _inside(b, after_a):
                continue
            if not _inside(a, after_b):
                continue
            zone.add(p)
    return zone


def _sample_rectangles(height: int, width: int, count: int) -> list[BBox] | None:
    if count != 2:
        return None
    for _ in range(250):
        hh = random.randint(5, 7)
        wh = random.randint(10, min(15, width - 4))
        hv = random.randint(10, min(15, height - 4))
        wv = random.randint(5, 7)
        if hh + 4 > hv or wv + 4 > wh:
            continue

        rv0 = random.randint(1, height - hv - 1)
        cv0 = random.randint(1, width - wv - 1)
        rh0_lo = max(1, rv0 + 2)
        rh0_hi = min(height - hh - 1, rv0 + hv - hh - 2)
        ch0_lo = max(1, cv0 + wv + 1 - wh)
        ch0_hi = min(width - wh - 1, cv0 - 2)
        if rh0_lo > rh0_hi or ch0_lo > ch0_hi:
            continue

        rh0 = random.randint(rh0_lo, rh0_hi)
        ch0 = random.randint(ch0_lo, ch0_hi)
        horizontal = (rh0, ch0, rh0 + hh - 1, ch0 + wh - 1)
        vertical = (rv0, cv0, rv0 + hv - 1, cv0 + wv - 1)
        rects = _sort_rects([horizontal, vertical])

        marker_mask = _marker_mask((height, width), rects)
        cover = _cover_counts((height, width), rects)
        local_zones = [_stable_local_zone(i, rects, marker_mask, cover) for i in range(len(rects))]
        if any(len(zone) < 4 for zone in local_zones):
            continue
        if len(_relay_zone(rects[0], rects[1], marker_mask)) < 3:
            continue
        return rects
    return None


def _neighbors8(cell: Coord, shape: tuple[int, int]) -> list[Coord]:
    h, w = shape
    r, c = cell
    out: list[Coord] = []
    for nr in range(max(0, r - 1), min(h, r + 2)):
        for nc in range(max(0, c - 1), min(w, c + 2)):
            if nr == r and nc == c:
                continue
            out.append((nr, nc))
    return out


def _can_place(cells: tuple[Coord, ...], allowed: set[Coord], occupied: np.ndarray) -> bool:
    cell_set = set(cells)
    h, w = occupied.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if (r, c) not in allowed or occupied[r, c]:
            return False
        for nr in range(max(0, r - 1), min(h, r + 2)):
            for nc in range(max(0, c - 1), min(w, c + 2)):
                if (nr, nc) in cell_set:
                    continue
                if occupied[nr, nc]:
                    return False
    return True


def _place_shape(
    grid: np.ndarray,
    occupied: np.ndarray,
    allowed: set[Coord],
    color: int,
    *,
    motion_rect: BBox | None = None,
) -> tuple[Coord, ...] | None:
    anchors = list(allowed)
    random.shuffle(anchors)
    size_groups = sorted({len(shape) for shape in SHAPES}, reverse=True)
    for size in size_groups:
        shapes = [shape for shape in SHAPES if len(shape) == size]
        random.shuffle(shapes)
        placements: list[tuple[Coord, ...]] = []
        for shape in shapes:
            for ar, ac in anchors:
                cells = tuple(sorted((ar + dr, ac + dc) for dr, dc in shape))
                if motion_rect is not None and _rotate_cells(motion_rect, cells) == cells:
                    continue
                if _can_place(cells, allowed, occupied):
                    placements.append(cells)
        if placements:
            chosen = random.choice(placements)
            for r, c in chosen:
                grid[r, c] = int(color)
                occupied[r, c] = True
            return chosen
    return None


def _outside_zone(height: int, width: int, rects: list[BBox], marker_mask: np.ndarray) -> set[Coord]:
    zone: set[Coord] = set()
    for r in range(height):
        for c in range(width):
            if marker_mask[r, c]:
                continue
            if any(_inside(rect, (r, c)) for rect in rects):
                continue
            zone.add((r, c))
    return zone


def _sort_marker_rects(rects: list[MarkerRect]) -> list[MarkerRect]:
    return sorted(rects, key=lambda rect: (rect.bbox[0], rect.bbox[1], rect.bbox[2], rect.bbox[3], rect.color))


def _rot90_point(cell: Coord, shape: tuple[int, int]) -> tuple[Coord, tuple[int, int]]:
    h, w = shape
    r, c = cell
    return (c, h - 1 - r), (w, h)


def _rotate_rects(rects: list[MarkerRect], shape: tuple[int, int], k: int) -> list[MarkerRect]:
    current_rects = rects
    current_shape = shape
    for _ in range(k % 4):
        next_rects: list[MarkerRect] = []
        for rect in current_rects:
            rotated_corners: list[Coord] = []
            for corner in rect.corners:
                rotated, _ = _rot90_point(corner, current_shape)
                rotated_corners.append(rotated)
            rows = [r for r, _ in rotated_corners]
            cols = [c for _, c in rotated_corners]
            next_rects.append(
                MarkerRect(
                    color=rect.color,
                    bbox=(min(rows), min(cols), max(rows), max(cols)),
                    corners=tuple(sorted(rotated_corners)),
                )
            )
        current_shape = (current_shape[1], current_shape[0])
        current_rects = next_rects
    return _sort_marker_rects(current_rects)


def _generate_candidate() -> tuple[np.ndarray, list[MarkerRect]] | None:
    rect_count = 2
    height = random.randint(20, 28)
    width = random.randint(20, 28)
    rect_bboxes = _sample_rectangles(height, width, rect_count)
    if rect_bboxes is None:
        return None

    marker_colors = random.sample(ALL_COLORS, rect_count)
    remaining = [color for color in ALL_COLORS if color not in marker_colors]
    outside_count = random.randint(1, 2)
    shape_color_count = rect_count + 1 + outside_count
    shape_colors = random.sample(remaining, shape_color_count)

    rects = [
        MarkerRect(color=color, bbox=rect, corners=_corners(rect))
        for color, rect in zip(marker_colors, rect_bboxes)
    ]

    grid = np.zeros((height, width), dtype=int)
    for rect in rects:
        for r, c in rect.corners:
            grid[r, c] = rect.color

    marker_mask = _marker_mask(grid.shape, rects)
    cover = _cover_counts(grid.shape, [rect.bbox for rect in rects])
    occupied = marker_mask.copy()

    relay_cells = _place_shape(
        grid,
        occupied,
        _relay_zone(rect_bboxes[-2], rect_bboxes[-1], marker_mask),
        shape_colors[0],
    )
    if relay_cells is None:
        return None

    for index, rect in enumerate(rect_bboxes):
        zone = _stable_local_zone(index, rect_bboxes, marker_mask, cover)
        placed = _place_shape(
            grid,
            occupied,
            zone,
            shape_colors[1 + index],
            motion_rect=rect,
        )
        if placed is None:
            return None

    zone = _outside_zone(height, width, rect_bboxes, marker_mask)
    for offset in range(outside_count):
        placed = _place_shape(
            grid,
            occupied,
            zone,
            shape_colors[1 + rect_count + offset],
        )
        if placed is None:
            return None

    rot_k = random.randint(0, 3)
    if rot_k:
        grid = np.rot90(grid, rot_k)
        rects = _rotate_rects(rects, (height, width), rot_k)

    return grid, rects


def _validate_instance(grid: np.ndarray, rects: list[MarkerRect]) -> None:
    detected = _find_marker_rectangles(grid)
    assert {(rect.color, rect.bbox) for rect in detected} == {(rect.color, rect.bbox) for rect in rects}

    ordered = [rect.bbox for rect in detected]
    marker_mask = _marker_mask(grid.shape, detected)

    out_main = main(grid)
    out_slow = _apply_order(grid, ordered, marker_mask, slow=True)
    assert np.array_equal(out_main, out_slow), "main must match the independent slow simulator"
    assert not np.array_equal(out_main, grid), "instance must actually change under the rule"

    overlap_content = False
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r, c] == BG or marker_mask[r, c]:
                continue
            hits = sum(_inside(rect, (r, c)) for rect in ordered)
            if hits >= 2:
                overlap_content = True
                break
        if overlap_content:
            break
    assert overlap_content, "instance must place visible content in an overlap relay zone"

    step_grid = np.array(grid, copy=True)
    for rect in ordered:
        next_grid = _apply_one_rotation_slow(step_grid, rect, marker_mask)
        assert not np.array_equal(next_grid, step_grid), "each rectangle step must matter on its own turn"
        step_grid = next_grid

    for index in range(len(ordered)):
        skipped = _apply_order(grid, ordered[:index] + ordered[index + 1 :], marker_mask, slow=True)
        assert not np.array_equal(skipped, out_main), "removing any rectangle must change the result"

    reversed_out = _apply_order(grid, list(reversed(ordered)), marker_mask, slow=True)
    assert not np.array_equal(reversed_out, out_main), "reversing the processing order must change the result"


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        candidate = _generate_candidate()
        if candidate is None:
            continue
        grid, rects = candidate
        try:
            _validate_instance(grid, rects)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a non-degenerate relay-rotation instance")
