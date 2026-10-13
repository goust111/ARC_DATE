import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
GREEN = 3
GREY = 5
TEAL = 8

SIDES_LR = ("left", "right")
SIDES_TB = ("top", "bottom")
ALL_SIDES = SIDES_LR + SIDES_TB

GRID_ROWS = (16, 20)
GRID_COLS = (16, 20)
MAX_ATTEMPTS = 500

VALID_OBJECT_DIMS = tuple(
    (h, w)
    for h in range(3, 6)
    for w in range(3, 6)
    if 8 <= 2 * h + 2 * w - 5 <= 15
)
WIDE_OBJECT_DIMS = tuple((h, w) for h, w in VALID_OBJECT_DIMS if w >= 4)

OVERLAP_RECT_DIMS = ((2, 2), (3, 2))
GENERAL_RECT_DIMS = ((2, 2), (2, 3), (3, 2), (3, 3))


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _bbox_h(bbox: tuple[int, int, int, int]) -> int:
    r0, _c0, r1, _c1 = bbox
    return r1 - r0 + 1


def _bbox_w(bbox: tuple[int, int, int, int]) -> int:
    _r0, c0, _r1, c1 = bbox
    return c1 - c0 + 1


def _bbox_in_bounds(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> bool:
    h, w = shape
    r0, c0, r1, c1 = bbox
    return 0 <= r0 <= r1 < h and 0 <= c0 <= c1 < w


def _bbox_overlaps(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    margin: int = 0,
) -> bool:
    return not (
        a[2] + margin < b[0]
        or b[2] + margin < a[0]
        or a[3] + margin < b[1]
        or b[3] + margin < a[1]
    )


def _bbox_clear(
    bbox: tuple[int, int, int, int],
    others: list[tuple[int, int, int, int]],
    margin: int = 0,
) -> bool:
    return all(not _bbox_overlaps(bbox, other, margin=margin) for other in others)


def _bbox_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = bbox
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _connected_components(grid: np.ndarray, color: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append({"coords": coords, "bbox": _bbox_from_coords(coords), "area": len(coords)})
    return components


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _tab_side(patch: np.ndarray) -> str:
    mask = patch == GREEN
    counts = {
        "top": int(mask[0, :].sum()),
        "bottom": int(mask[-1, :].sum()),
        "left": int(mask[:, 0].sum()),
        "right": int(mask[:, -1].sum()),
    }
    sides = [side for side, count in counts.items() if count == 1]
    assert len(sides) == 1, f"expected exactly one tab side, got {counts}"
    return sides[0]


def _projection_bbox(
    bbox: tuple[int, int, int, int],
    side: str,
) -> tuple[tuple[int, int, int, int], int]:
    r0, c0, r1, c1 = bbox
    h = _bbox_h(bbox)
    w = _bbox_w(bbox)
    if side == "left":
        return (r0, c1 + 2, r1, c1 + 1 + w), BLUE
    if side == "right":
        return (r0, c0 - w - 1, r1, c0 - 2), BLUE
    if side == "top":
        return (r1 + 2, c0, r1 + 1 + h, c1), TEAL
    if side == "bottom":
        return (r0 - h - 1, c0, r0 - 2, c1), TEAL
    raise ValueError(f"unknown side: {side}")


def _tab_coord(shape: tuple[int, int], side: str) -> tuple[int, int]:
    h, w = shape
    if side == "left":
        return h // 2, 0
    if side == "right":
        return h // 2, w - 1
    if side == "top":
        return 0, w // 2
    if side == "bottom":
        return h - 1, w // 2
    raise ValueError(f"unknown side: {side}")


def _mask_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return len(seen) == int(mask.sum())


def _draw_rect_outline(patch: np.ndarray, top: int, left: int, height: int, width: int) -> None:
    bottom = top + height - 1
    right = left + width - 1
    patch[top, left : right + 1] = GREEN
    patch[bottom, left : right + 1] = GREEN
    patch[top : bottom + 1, left] = GREEN
    patch[top : bottom + 1, right] = GREEN


def _open_patch(patch: np.ndarray, side: str) -> np.ndarray:
    if random.random() >= 0.5:
        return patch
    tab = _tab_coord(patch.shape, side)
    candidates = [tuple(map(int, rc)) for rc in np.argwhere(patch == GREEN) if tuple(map(int, rc)) != tab]
    random.shuffle(candidates)
    for r, c in candidates:
        trial = patch.copy()
        trial[r, c] = BLACK
        if int((trial == GREEN).sum()) < 8:
            continue
        if not _mask_connected(trial == GREEN):
            continue
        try:
            if _tab_side(trial) == side:
                return trial
        except AssertionError:
            continue
    return patch


def _make_outline_patch(h: int, w: int, side: str) -> np.ndarray:
    patch = np.zeros((h, w), dtype=int)
    if side == "left":
        _draw_rect_outline(patch, 0, 1, h, w - 1)
    elif side == "right":
        _draw_rect_outline(patch, 0, 0, h, w - 1)
    elif side == "top":
        _draw_rect_outline(patch, 1, 0, h - 1, w)
    elif side == "bottom":
        _draw_rect_outline(patch, 0, 0, h - 1, w)
    else:
        raise ValueError(f"unknown side: {side}")

    tr, tc = _tab_coord((h, w), side)
    patch[tr, tc] = GREEN
    patch = _open_patch(patch, side)
    assert 8 <= int((patch == GREEN).sum()) <= 15
    assert _tab_side(patch) == side
    assert _mask_connected(patch == GREEN)
    return patch


def _object_bbox_for_zone(
    zone_bbox: tuple[int, int, int, int],
    side: str,
) -> tuple[int, int, int, int]:
    zr0, zc0, zr1, zc1 = zone_bbox
    h = _bbox_h(zone_bbox)
    w = _bbox_w(zone_bbox)
    if side == "left":
        return zr0, zc0 - w - 1, zr1, zc0 - 2
    if side == "right":
        return zr0, zc1 + 2, zr1, zc1 + w + 1
    if side == "top":
        return zr0 - h - 1, zc0, zr0 - 2, zc1
    if side == "bottom":
        return zr1 + 2, zc0, zr1 + h + 1, zc1
    raise ValueError(f"unknown side: {side}")


def _coverage_matrix(shape: tuple[int, int], zones: list[tuple[int, int, int, int]]) -> np.ndarray:
    coverage = np.zeros(shape, dtype=int)
    for r0, c0, r1, c1 in zones:
        coverage[r0 : r1 + 1, c0 : c1 + 1] += 1
    return coverage


def _find_rect_in_mask(
    mask: np.ndarray,
    blocked_bboxes: list[tuple[int, int, int, int]],
    size_options: tuple[tuple[int, int], ...],
) -> tuple[int, int, int, int] | None:
    h, w = mask.shape
    dims = list(size_options)
    random.shuffle(dims)
    candidates: list[tuple[int, int, int, int]] = []
    for rh, rw in dims:
        positions = [(r0, c0) for r0 in range(h - rh + 1) for c0 in range(w - rw + 1)]
        random.shuffle(positions)
        for r0, c0 in positions:
            r1 = r0 + rh - 1
            c1 = c0 + rw - 1
            bbox = (r0, c0, r1, c1)
            if not mask[r0 : r1 + 1, c0 : c1 + 1].all():
                continue
            if not _bbox_clear(bbox, blocked_bboxes, margin=1):
                continue
            candidates.append(bbox)
    return random.choice(candidates) if candidates else None


def _place_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    grid[top : top + h, left : left + w] = np.where(patch == GREEN, GREEN, grid[top : top + h, left : left + w])


def _paint_rect(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = color


def _rect_category(coverage: np.ndarray, bbox: tuple[int, int, int, int]) -> str:
    r0, c0, r1, c1 = bbox
    values = np.unique(coverage[r0 : r1 + 1, c0 : c1 + 1])
    assert len(values) == 1
    count = int(values[0])
    if count == 0:
        return "none"
    if count == 1:
        return "single"
    return "multi"


def _build_primary_layout(
    shape: tuple[int, int],
) -> tuple[list[dict], list[tuple[int, int, int, int]]] | None:
    h, w = shape
    for _ in range(200):
        zh_h, zh_w = random.choice(WIDE_OBJECT_DIMS)
        zv_h, zv_w = random.choice(WIDE_OBJECT_DIMS)
        side_h = random.choice(SIDES_LR)
        side_v = random.choice(SIDES_TB)
        union_h = max(zh_h, zv_h)
        union_w = zh_w + zv_w - 2
        positions = [(r, c) for r in range(h - union_h + 1) for c in range(w - union_w + 1)]
        random.shuffle(positions)
        for top, left in positions:
            zone_h = (top, left, top + zh_h - 1, left + zh_w - 1)
            zone_v = (top, left + zh_w - 2, top + zv_h - 1, left + zh_w + zv_w - 3)
            obj_h = _object_bbox_for_zone(zone_h, side_h)
            obj_v = _object_bbox_for_zone(zone_v, side_v)
            if not _bbox_in_bounds(obj_h, shape) or not _bbox_in_bounds(obj_v, shape):
                continue
            if not _bbox_clear(obj_h, [obj_v], margin=1):
                continue
            return (
                [
                    {"zone": zone_h, "side": side_h, "bbox": obj_h, "patch": _make_outline_patch(zh_h, zh_w, side_h)},
                    {"zone": zone_v, "side": side_v, "bbox": obj_v, "patch": _make_outline_patch(zv_h, zv_w, side_v)},
                ],
                [zone_h, zone_v],
            )
    return None


def _build_extra_object(
    shape: tuple[int, int],
    objects: list[dict],
    zones: list[tuple[int, int, int, int]],
) -> dict | None:
    h, w = shape
    for _ in range(200):
        obj_h, obj_w = random.choice(VALID_OBJECT_DIMS)
        side = random.choice(ALL_SIDES)
        positions = [(r, c) for r in range(h - obj_h + 1) for c in range(w - obj_w + 1)]
        random.shuffle(positions)
        for zr0, zc0 in positions:
            zone = (zr0, zc0, zr0 + obj_h - 1, zc0 + obj_w - 1)
            bbox = _object_bbox_for_zone(zone, side)
            if not _bbox_in_bounds(bbox, shape):
                continue
            if not _bbox_clear(zone, zones, margin=1):
                continue
            if not _bbox_clear(bbox, [obj["bbox"] for obj in objects], margin=1):
                continue
            return {"zone": zone, "side": side, "bbox": bbox, "patch": _make_outline_patch(obj_h, obj_w, side)}
    return None


def _generate_example() -> np.ndarray | None:
    shape = (
        random.randint(*GRID_ROWS),
        random.randint(*GRID_COLS),
    )
    primary = _build_primary_layout(shape)
    if primary is None:
        return None
    objects, zones = primary

    green_count = random.choice((2, 3))
    if green_count == 3:
        extra = _build_extra_object(shape, objects, zones)
        if extra is None:
            return None
        objects.append(extra)
        zones.append(extra["zone"])

    coverage = _coverage_matrix(shape, zones)
    blocked = [obj["bbox"] for obj in objects]

    overlap_mask = coverage == 2
    overlap_rect = _find_rect_in_mask(overlap_mask, blocked, OVERLAP_RECT_DIMS)
    if overlap_rect is None:
        return None

    rects = [overlap_rect]
    zone_masks = [_bbox_mask(shape, zone) & (coverage == 1) for zone in zones]
    zone_order = list(range(len(zones)))
    random.shuffle(zone_order)

    single_rect = None
    for idx in zone_order:
        single_rect = _find_rect_in_mask(zone_masks[idx], blocked + rects, GENERAL_RECT_DIMS)
        if single_rect is not None:
            break
    if single_rect is None:
        return None
    rects.append(single_rect)

    none_rect = _find_rect_in_mask(coverage == 0, blocked + rects, GENERAL_RECT_DIMS)
    if none_rect is None:
        return None
    rects.append(none_rect)

    gray_count = random.choice((3, 4))
    if gray_count == 4:
        extra_rect = None
        if len(zones) == 3:
            extra_rect = _find_rect_in_mask(zone_masks[2], blocked + rects, GENERAL_RECT_DIMS)
        if extra_rect is None:
            extra_masks = zone_masks + [coverage == 0]
            random.shuffle(extra_masks)
            for mask in extra_masks:
                extra_rect = _find_rect_in_mask(mask, blocked + rects, GENERAL_RECT_DIMS)
                if extra_rect is not None:
                    break
        if extra_rect is None:
            return None
        rects.append(extra_rect)

    categories = {_rect_category(coverage, rect) for rect in rects}
    if not {"single", "multi", "none"} <= categories:
        return None

    grid = np.zeros(shape, dtype=int)
    for obj in objects:
        r0, c0, _r1, _c1 = obj["bbox"]
        _place_patch(grid, obj["patch"], r0, c0)
    for rect in rects:
        _paint_rect(grid, rect, GREY)

    if set(map(int, np.unique(grid))) - {BLACK, GREEN, GREY}:
        return None
    if len(_connected_components(grid, GREEN)) != len(objects):
        return None

    output = main(grid)
    if output.shape != grid.shape:
        return None

    for rect in rects:
        category = _rect_category(coverage, rect)
        r0, c0, r1, c1 = rect
        block = output[r0 : r1 + 1, c0 : c1 + 1]
        if category == "none" and not np.all(block == GREY):
            return None
        if category == "multi" and not np.all(block == BLUE):
            return None
        if category == "single":
            colors = {int(v) for v in np.unique(block)}
            if len(colors) != 1 or colors.pop() not in {BLUE, TEAL}:
                return None

    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor gray rectangles where they fall inside the invisible projection boxes of mirrored green tabbed outlines.

    Input structure:
    - The grid has a black background, 2-3 green 4-connected outline objects with exactly one one-pixel tab on a bbox side, and 2-4 solid gray rectangles.
    - Each green object is read through its tight bounding box; the unique side containing exactly one green boundary pixel is the tab side.
    - Gray rectangles never overlap the green outlines, but they may lie inside zero, one, or several invisible projection boxes.

    Rule:
    - For each green outline, detect the tab side from its bbox boundary counts.
    - Left/right tabs imply a horizontal mirror and define a blue projection box on the opposite side with a one-cell gap; top/bottom tabs imply a vertical mirror and define a teal projection box on the opposite side with a one-cell gap.
    - Recolor each gray pixel by projection coverage: keep gray outside every projection box, use the lone box color under exactly one box, and use blue whenever multiple projection boxes overlap on that pixel.

    Key insight:
    - The mirrored copies are never drawn; only their axis-aligned bounding boxes matter, so the visible gray recoloring encodes invisible mirror placement rather than the mirrored outline shapes themselves.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    components = _connected_components(g, GREEN)
    assert components, "expected at least one green outline"

    zones: list[tuple[tuple[int, int, int, int], int]] = []
    for comp in components:
        patch = _crop_bbox(g, comp["bbox"])
        side = _tab_side(patch)
        zones.append(_projection_bbox(comp["bbox"], side))

    out = g.copy()
    gray_coords = np.argwhere(g == GREY)
    for r, c in gray_coords:
        hits = 0
        color = GREY
        for (r0, c0, r1, c1), zone_color in zones:
            if r0 <= r <= r1 and c0 <= c <= c1:
                hits += 1
                color = zone_color
        if hits == 1:
            out[r, c] = color
        elif hits > 1:
            out[r, c] = BLUE
    return out


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_example()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
