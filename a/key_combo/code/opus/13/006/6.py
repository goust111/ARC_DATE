from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
MAROON = 9


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox_from_coords(coords: list[Coord]) -> BBox:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _bbox_from_mask(mask: np.ndarray) -> BBox | None:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return None
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(arr: np.ndarray, bbox: BBox) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return arr[r0 : r1 + 1, c0 : c1 + 1].copy()


def _find_components(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q: deque[Coord] = deque([(r, c)])
            coords: list[Coord] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(coords)
    return components


def _scale_bool(mask: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(mask.astype(int), np.ones((factor, factor), dtype=int)).astype(bool)


def _match_indicator(
    local_yellow: np.ndarray,
    template_yellows: list[np.ndarray],
) -> tuple[int, BBox, BBox]:
    local_bbox = _bbox_from_mask(local_yellow)
    assert local_bbox is not None, "each local object must contain yellow indicator pixels"
    local_crop = _crop_bbox(local_yellow, local_bbox)

    for template_yellow in template_yellows:
        template_bbox = _bbox_from_mask(template_yellow)
        if template_bbox is None:
            continue
        template_crop = _crop_bbox(template_yellow, template_bbox)
        th, tw = template_crop.shape
        lh, lw = local_crop.shape
        if lh % th != 0 or lw % tw != 0:
            continue
        factor_h = lh // th
        factor_w = lw // tw
        if factor_h != factor_w or factor_h <= 0:
            continue
        factor = int(factor_h)
        if np.array_equal(_scale_bool(template_crop, factor), local_crop):
            scaled_full = _scale_bool(template_yellow, factor)
            scaled_bbox = _bbox_from_mask(scaled_full)
            assert scaled_bbox is not None
            return factor, scaled_bbox, local_bbox

    raise AssertionError("failed to match a local indicator region to the template indicator")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the NOR of the two top templates as a completion mask, then scale and align it to each local object by matching yellow indicators.

    Input structure:
    - A black background contains two top template sprites separated by one maroon vertical divider.
    - The left/right templates are 4-connected blue-or-yellow and red-or-yellow objects with equal tight bbox size.
    - One or two lower local objects are 4-connected green/yellow components whose yellow pixels encode scale and alignment.

    Rule:
    - Extract the two template bboxes and convert them to same-sized binary masks using non-black as foreground.
    - Compute the pixelwise NOR mask: a cell is part of the completion only where both template masks are empty.
    - For each lower local object, match its yellow indicator shape against the template yellow shape to infer the integer scale and alignment.
    - Scale the NOR mask by nearest-neighbor replication, paste it so the indicators line up, recolor local yellow pixels to green, and erase the templates plus the maroon divider.

    Key insight:
    - The missing shape is not stored in either template alone; it is the shared absence of both templates, revealed only by their NOR.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    separator_cols = [c for c in range(w) if np.all(g[:, c] == MAROON)]
    assert len(separator_cols) == 1, "expected one full-height maroon separator"
    sep = separator_cols[0]

    fg_mask = (g != BLACK) & (g != MAROON)
    components = _find_components(fg_mask)

    templates: list[dict[str, object]] = []
    locals_: list[dict[str, object]] = []
    for coords in components:
        bbox = _bbox_from_coords(coords)
        crop = _crop_bbox(g, bbox)
        colors = {int(g[r, c]) for r, c in coords}
        info = {"coords": coords, "bbox": bbox, "crop": crop, "colors": colors}
        if BLUE in colors or RED in colors:
            templates.append(info)
        elif colors <= {GREEN, YELLOW}:
            locals_.append(info)

    assert len(templates) == 2, "expected exactly two template components"
    assert 1 <= len(locals_) <= 2, "expected one or two local objects"

    template_crops = [info["crop"] for info in templates]
    assert template_crops[0].shape == template_crops[1].shape, "template bboxes must match"
    mask_a = template_crops[0] != BLACK
    mask_b = template_crops[1] != BLACK
    nor_mask = (~mask_a) & (~mask_b)
    assert np.any(nor_mask), "NOR completion mask must be non-empty"

    template_yellows = [crop == YELLOW for crop in template_crops if np.any(crop == YELLOW)]
    assert template_yellows, "expected yellow indicators inside the templates"

    out = g.copy()
    out[:, sep] = BLACK
    for info in templates:
        for r, c in info["coords"]:
            out[r, c] = BLACK

    for info in locals_:
        bbox = info["bbox"]
        crop = info["crop"]
        local_yellow = crop == YELLOW
        factor, scaled_indicator_bbox, local_indicator_bbox = _match_indicator(local_yellow, template_yellows)

        local_r0, local_c0, _, _ = bbox
        ind_r0, ind_c0, _, _ = local_indicator_bbox
        scaled_ind_r0, scaled_ind_c0, _, _ = scaled_indicator_bbox
        paste_r0 = local_r0 + ind_r0 - scaled_ind_r0
        paste_c0 = local_c0 + ind_c0 - scaled_ind_c0

        scaled_nor = _scale_bool(nor_mask, factor)
        sh, sw = scaled_nor.shape
        assert 0 <= paste_r0 and 0 <= paste_c0
        assert paste_r0 + sh <= h and paste_c0 + sw <= w

        for r, c in info["coords"]:
            out[r, c] = GREEN
        rr, cc = np.where(scaled_nor)
        out[paste_r0 + rr, paste_c0 + cc] = GREEN

    return out


def _build_base_masks(h: int, w: int, rotation: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    template_a = np.zeros((h, w), dtype=bool)
    template_b = np.zeros((h, w), dtype=bool)
    indicators = np.zeros((h, w), dtype=bool)

    template_a[0, :] = True
    template_a[:, 0] = True
    template_b[-1, :] = True
    template_b[:, -1] = True
    indicators[0, -1] = True
    indicators[-1, 0] = True

    nor_mask = ~(template_a | template_b)
    for _ in range(rotation % 4):
        template_a = np.rot90(template_a)
        template_b = np.rot90(template_b)
        indicators = np.rot90(indicators)
        nor_mask = np.rot90(nor_mask)
    return template_a, template_b, indicators, nor_mask


def _render_template(mask: np.ndarray, indicators: np.ndarray, color: int) -> np.ndarray:
    sprite = np.zeros(mask.shape, dtype=int)
    sprite[mask] = color
    sprite[indicators] = YELLOW
    return sprite


def _render_local_object(ring_mask: np.ndarray, indicators: np.ndarray, factor: int) -> np.ndarray:
    scaled_ring = _scale_bool(ring_mask, factor)
    scaled_indicators = _scale_bool(indicators, factor)
    obj = np.zeros(scaled_ring.shape, dtype=int)
    obj[scaled_ring] = GREEN
    obj[scaled_indicators] = YELLOW
    return obj


def _paste(dst: np.ndarray, src: np.ndarray, top: int, left: int) -> None:
    h, w = src.shape
    mask = src != BLACK
    block = dst[top : top + h, left : left + w]
    block[mask] = src[mask]


def _gen_case() -> tuple[np.ndarray, np.ndarray]:
    base_h, base_w = random.choice([(4, 4), (4, 5)])
    rotation = random.randrange(4)
    template_a_mask, template_b_mask, indicators, nor_mask = _build_base_masks(base_h, base_w, rotation)
    h, w = template_a_mask.shape
    ring_mask = ~nor_mask

    local_count = random.choice([1, 2])
    if local_count == 1:
        scales = [random.choice([1, 2, 3])]
        sides = [random.choice(["left", "right"])]
    else:
        scales = [random.choice([1, 2]), random.choice([1, 2])]
        sides = ["left", "right"]
        if random.random() < 0.5:
            random.shuffle(scales)

    left_local_w = max([w * s for s, side in zip(scales, sides) if side == "left"], default=0)
    right_local_w = max([w * s for s, side in zip(scales, sides) if side == "right"], default=0)
    max_local_h = max(h * s for s in scales)

    left_region = max(w, left_local_w) + random.randint(2, 4)
    right_region = max(w, right_local_w) + random.randint(2, 4)
    grid_w = left_region + 1 + right_region
    lower_band = random.randint(3, 5)
    grid_h = h + max_local_h + lower_band + random.randint(2, 4)
    if grid_w > 30 or grid_h > 30:
        raise ValueError("grid too large")

    separator_col = left_region
    grid = np.zeros((grid_h, grid_w), dtype=int)
    expected = np.zeros_like(grid)
    grid[:, separator_col] = MAROON

    top_row = random.randint(1, 2)
    left_template = _render_template(template_a_mask, indicators, BLUE)
    right_template = _render_template(template_b_mask, indicators, RED)

    left_template_col = random.randint(0, left_region - w)
    right_template_col = separator_col + 1 + random.randint(0, right_region - w)
    _paste(grid, left_template, top_row, left_template_col)
    _paste(grid, right_template, top_row, right_template_col)

    min_local_row = max(grid_h // 2, top_row + h + 2)
    placements: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    for factor, side in zip(scales, sides):
        obj = _render_local_object(ring_mask, indicators, factor)
        solid = np.full(obj.shape, GREEN, dtype=int)
        oh, ow = obj.shape
        if side == "left":
            c0 = random.randint(0, left_region - ow)
        else:
            c0 = separator_col + 1 + random.randint(0, right_region - ow)
        r0 = random.randint(min_local_row, grid_h - oh - 1)
        placements.append((r0, c0, obj, solid))

    if len(placements) == 2:
        (r0a, c0a, obj_a, solid_a), (r0b, c0b, obj_b, solid_b) = placements
        ah, aw = obj_a.shape
        bh, bw = obj_b.shape
        overlap = not (r0a + ah + 1 < r0b or r0b + bh + 1 < r0a or c0a + aw + 1 < c0b or c0b + bw + 1 < c0a)
        if overlap:
            raise ValueError("local objects too close")

    for r0, c0, obj, solid in placements:
        _paste(grid, obj, r0, c0)
        _paste(expected, solid, r0, c0)

    produced = main(grid)
    if not np.array_equal(produced, expected):
        raise AssertionError("generated sample does not match solver output")
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(500):
        try:
            grid, _expected = _gen_case()
            return grid
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid input for puzzle 6")
