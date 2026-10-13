from __future__ import annotations

import numpy as np


BG = 0
SPRITE = 2
FILL = 3
_RNG = np.random.default_rng()
_MAX_GEN_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _components8(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr = cr + dr
                        nc = cc + dc
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int).copy()


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int | None = None) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    if transparent is None:
        view[:, :] = patch
    else:
        mask = patch != int(transparent)
        view[mask] = patch[mask]


def _pack_crops(crops: list[np.ndarray], axis: str, bg: int) -> np.ndarray:
    if axis == "h":
        out = np.full((crops[0].shape[0], len(crops) * crops[0].shape[1]), bg, dtype=int)
        for i, crop in enumerate(crops):
            _paste(out, crop, 0, i * crop.shape[1])
        return out
    out = np.full((len(crops) * crops[0].shape[0], crops[0].shape[1]), bg, dtype=int)
    for i, crop in enumerate(crops):
        _paste(out, crop, i * crop.shape[0], 0)
    return out


def _rect_sum(prefix: np.ndarray, r0: int, c0: int, h: int, w: int) -> int:
    r1 = r0 + h
    c1 = c0 + w
    return int(prefix[r1, c1] - prefix[r0, c1] - prefix[r1, c0] + prefix[r0, c0])


def _largest_black_rect(
    grid: np.ndarray, bg: int, *, min_side: int = 2
) -> tuple[tuple[int, int, int, int] | None, int]:
    black = (np.asarray(grid, dtype=int) == int(bg)).astype(int)
    h, w = black.shape
    prefix = np.pad(black, ((1, 0), (1, 0))).cumsum(axis=0).cumsum(axis=1)

    best: tuple[int, int, int, int] | None = None
    best_area = 0
    best_count = 0
    for r0 in range(h - min_side + 1):
        for c0 in range(w - min_side + 1):
            for rh in range(min_side, h - r0 + 1):
                for rw in range(min_side, w - c0 + 1):
                    area = rh * rw
                    if area < best_area:
                        continue
                    if _rect_sum(prefix, r0, c0, rh, rw) != area:
                        continue
                    if area > best_area:
                        best = (r0, c0, rh, rw)
                        best_area = area
                        best_count = 1
                    else:
                        best_count += 1
                        assert best is not None
                        if (r0, c0) < (best[0], best[1]):
                            best = (r0, c0, rh, rw)
    return best, best_count


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop separated sprites, pack them tightly along their dominant layout axis, then fill the largest emergent empty rectangle.

    Input structure:
    - The grid has black background and 3-5 separated red 8-connected sprites.
    - Every sprite has the same bounding-box shape, but its red occupancy can differ and may contain black holes.
    - Sprite centers are spread much more along rows or columns, defining a vertical or horizontal layout axis.

    Rule:
    - Crop each sprite to its bounding box.
    - Compare the spans of sprite-center rows and columns to choose the layout axis.
    - Sort sprites by center coordinate along that axis and concatenate the crops with zero gap.
    - In the packed canvas, enumerate all all-black rectangles with height and width at least 2, choose the maximum-area one, and break ties by the top-left corner.
    - Fill that rectangle green and return the packed canvas.

    Key insight:
    - The target rectangle is not read directly from the dispersed input; it appears only after tight packing changes which sprite holes touch each other.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    comps = _components8(g != bg)
    assert 3 <= len(comps) <= 5

    items: list[tuple[tuple[int, int, int, int], np.ndarray, tuple[float, float]]] = []
    shapes = set()
    for comp in comps:
        box = _bbox(comp)
        crop = _crop(g, box)
        shapes.add(tuple(crop.shape))
        r0, c0, r1, c1 = box
        items.append((box, crop, ((r0 + r1) / 2.0, (c0 + c1) / 2.0)))
    assert len(shapes) == 1

    row_span = max(center[0] for _, _, center in items) - min(center[0] for _, _, center in items)
    col_span = max(center[1] for _, _, center in items) - min(center[1] for _, _, center in items)
    axis = "h" if col_span > row_span else "v"
    order_key = 1 if axis == "h" else 0
    crops = [crop for _, crop, _ in sorted(items, key=lambda item: (item[2][order_key], item[2][1 - order_key]))]

    packed = _pack_crops(crops, axis, bg)
    rect, _count = _largest_black_rect(packed, bg, min_side=2)
    assert rect is not None

    r0, c0, rh, rw = rect
    out = packed.copy()
    out[r0 : r0 + rh, c0 : c0 + rw] = FILL
    return out


def _normalized_crop(kind: str, stem_h: int, band_top: int, band_h: int) -> np.ndarray:
    crop = np.full((stem_h, 3), SPRITE, dtype=int)

    if kind == "target_left":
        crop[band_top : band_top + band_h, 1:3] = BG
    elif kind == "target_right":
        crop[band_top : band_top + band_h, 0:2] = BG
    elif kind == "left_small":
        top = 1 if band_top + band_h >= stem_h - 1 else stem_h - 3
        crop[top : top + 2, 0] = BG
        crop[0, 1] = BG
    elif kind == "right_small":
        top = stem_h - 3 if band_top <= 1 else 1
        crop[top : top + 2, 2] = BG
        crop[stem_h - 1, 1] = BG
    elif kind == "center_bar":
        top = 1 if band_top + band_h >= stem_h - 1 else band_top + band_h
        if top >= stem_h - 1:
            top = max(1, band_top - 2)
        crop[top : top + 2, 1] = BG
        crop[0, 1] = BG
    elif kind == "outer_left":
        crop[1:3, 0:2] = BG
        crop[stem_h - 2, 1] = BG
    elif kind == "outer_right":
        crop[stem_h - 3 : stem_h - 1, 1:3] = BG
        crop[1, 1] = BG
    else:
        raise ValueError(f"unknown crop kind: {kind}")

    assert np.any(crop == BG)
    assert len(_components8(crop == SPRITE)) == 1
    return crop


def _split_packed(packed: np.ndarray, axis: str) -> list[np.ndarray]:
    if axis == "h":
        return [packed[:, i : i + 3].copy() for i in range(0, packed.shape[1], 3)]
    return [packed[i : i + 3, :].copy() for i in range(0, packed.shape[0], 3)]


def _expected_target_rect(axis: str, boundary_index: int, band_top: int, band_h: int) -> tuple[int, int, int, int]:
    if axis == "h":
        return band_top, boundary_index * 3 + 1, band_h, 4
    return boundary_index * 3 + 1, band_top, 4, band_h


def _scatter_input(crops: list[np.ndarray], axis: str) -> np.ndarray:
    n = len(crops)
    piece = crops[0].shape[1] if axis == "h" else crops[0].shape[0]
    cross = crops[0].shape[0] if axis == "h" else crops[0].shape[1]

    base_axis = 1 + n * piece + (n - 1) * 3 + 1
    axis_len = int(min(30, base_axis + int(_RNG.integers(0, max(1, 31 - base_axis)))))
    extra = axis_len - base_axis
    extras = [0] * (n + 1)
    for _ in range(extra):
        extras[int(_RNG.integers(0, n + 1))] += 1

    starts = []
    pos = 1 + extras[0]
    starts.append(pos)
    for i in range(1, n):
        pos += piece + 3 + extras[i]
        starts.append(pos)

    cross_len = int(min(30, cross + 2 + int(_RNG.integers(0, 5))))
    max_cross_start = cross_len - cross - 1
    if max_cross_start <= 1:
        cross_starts = [1] * n
    else:
        base = int(_RNG.integers(1, max_cross_start + 1))
        span = int(min(4, max_cross_start - 1))
        cross_starts = []
        for _ in range(n):
            delta = int(_RNG.integers(0, span + 1)) if span > 0 else 0
            cross_starts.append(min(max_cross_start, base + delta))
        if n >= 3 and len(set(cross_starts)) == 1 and max_cross_start > 1:
            cross_starts[-1] = 1 if cross_starts[-1] != 1 else max_cross_start

    if axis == "h":
        grid = np.zeros((cross_len, axis_len), dtype=int)
        for crop, r0, c0 in zip(crops, cross_starts, starts):
            _paste(grid, crop, r0, c0, transparent=BG)
    else:
        grid = np.zeros((axis_len, cross_len), dtype=int)
        for crop, r0, c0 in zip(crops, starts, cross_starts):
            _paste(grid, crop, r0, c0, transparent=BG)
    return grid


def _max_rect_area(grid: np.ndarray) -> int:
    rect, _count = _largest_black_rect(grid, BG, min_side=2)
    if rect is None:
        return 0
    return rect[2] * rect[3]


def _semantic_ok(
    grid: np.ndarray,
    axis: str,
    boundary_index: int,
    band_top: int,
    band_h: int,
    target_area: int,
) -> bool:
    comps = _components8(grid == SPRITE)
    if not (3 <= len(comps) <= 5):
        return False

    boxes = [_bbox(comp) for comp in comps]
    shapes = {(r1 - r0 + 1, c1 - c0 + 1) for r0, c0, r1, c1 in boxes}
    if len(shapes) != 1:
        return False

    centers = [((r0 + r1) / 2.0, (c0 + c1) / 2.0) for r0, c0, r1, c1 in boxes]
    row_span = max(r for r, _ in centers) - min(r for r, _ in centers)
    col_span = max(c for _, c in centers) - min(c for _, c in centers)
    inferred_axis = "h" if col_span > row_span else "v"
    if inferred_axis != axis:
        return False

    packed = main(grid).copy()
    packed[packed == FILL] = BG
    rect, count = _largest_black_rect(packed, BG, min_side=2)
    if rect != _expected_target_rect(axis, boundary_index, band_top, band_h):
        return False
    if count != 1 or rect is None or rect[2] * rect[3] != target_area:
        return False

    if axis == "h":
        boundary = 3 * (boundary_index + 1)
        if not (rect[1] < boundary <= rect[1] + rect[3] - 1):
            return False
    else:
        boundary = 3 * (boundary_index + 1)
        if not (rect[0] < boundary <= rect[0] + rect[2] - 1):
            return False

    for crop in _split_packed(packed, axis):
        if _max_rect_area(crop) >= target_area:
            return False

    solved = main(grid)
    expected = packed.copy()
    expected[rect[0] : rect[0] + rect[2], rect[1] : rect[1] + rect[3]] = FILL
    return np.array_equal(solved, expected)


def _generate_attempt() -> np.ndarray | None:
    axis = "h" if int(_RNG.integers(0, 2)) == 0 else "v"
    n = int(_RNG.integers(3, 6))
    stem_h = int(_RNG.choice(np.array([5, 6], dtype=int)))
    band_h = int(_RNG.choice(np.array([2, 3], dtype=int)))
    if band_h > stem_h - 2:
        band_h = stem_h - 2
    band_top = int(_RNG.integers(1, stem_h - band_h))
    boundary_index = int(_RNG.integers(0, n - 1))

    distractor_pool = np.array(
        ["left_small", "right_small", "center_bar", "outer_left", "outer_right"], dtype=object
    )
    normalized: list[np.ndarray] = []
    for idx in range(n):
        if idx == boundary_index:
            normalized.append(_normalized_crop("target_left", stem_h, band_top, band_h))
        elif idx == boundary_index + 1:
            normalized.append(_normalized_crop("target_right", stem_h, band_top, band_h))
        else:
            kind = str(_RNG.choice(distractor_pool))
            normalized.append(_normalized_crop(kind, stem_h, band_top, band_h))

    crops = normalized if axis == "h" else [crop.T.copy() for crop in normalized]
    packed = _pack_crops(crops, axis, BG)
    rect, count = _largest_black_rect(packed, BG, min_side=2)
    expected_rect = _expected_target_rect(axis, boundary_index, band_top, band_h)
    target_area = expected_rect[2] * expected_rect[3]
    if rect != expected_rect or count != 1:
        return None

    for crop in crops:
        if _max_rect_area(crop) >= target_area:
            return None

    grid = _scatter_input(crops, axis)
    if max(grid.shape) > 30:
        return None
    if not _semantic_ok(grid, axis, boundary_index, band_top, band_h, target_area):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 10")
