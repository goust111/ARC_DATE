from __future__ import annotations

import random
from collections import Counter, defaultdict

import numpy as np


BACKGROUND = 0
PALETTE = (1, 2, 3, 4, 5, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int = BACKGROUND) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    mask = patch != transparent
    view[mask] = patch[mask]


def _find_foreground_components(grid: np.ndarray, *, bg: int = BACKGROUND) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors8(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != bg:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            bbox = _bbox_from_coords(coords)
            components.append(
                {
                    "coords": coords,
                    "bbox": bbox,
                    "crop": _crop_bbox(grid, bbox),
                }
            )
    return components


def _shape_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    binary = np.asarray(mask, dtype=np.uint8)
    return tuple(map(int, binary.shape)), binary.tobytes()


def _component_mask(component: dict, *, bg: int = BACKGROUND) -> np.ndarray:
    return component["crop"] != bg


def _partition_roles(components: list[dict], *, bg: int = BACKGROUND) -> tuple[list[dict], dict]:
    groups: dict[tuple[tuple[int, int], bytes], list[dict]] = defaultdict(list)
    for component in components:
        groups[_shape_signature(_component_mask(component, bg=bg))].append(component)

    repeated_groups = [items for items in groups.values() if len(items) > 1]
    unique_groups = [items for items in groups.values() if len(items) == 1]
    assert len(repeated_groups) == 1, "expected exactly one repeated shape group"
    assert len(unique_groups) == 1, "expected exactly one unique shape group"
    assert len(groups) == 2, "expected exactly two shape identities"
    return repeated_groups[0], unique_groups[0][0]


def _defect_info(crop: np.ndarray, *, bg: int = BACKGROUND) -> tuple[int, int, int]:
    fg = crop[crop != bg]
    counts = Counter(map(int, fg))
    defect_colors = [color for color, count in counts.items() if count == 1]
    assert len(counts) == 2, "repeated shape must use exactly two foreground colors"
    assert len(defect_colors) == 1, "expected a single defect color"

    defect_color = int(defect_colors[0])
    coords = np.argwhere(crop == defect_color)
    assert coords.shape == (1, 2), "defect color must occupy exactly one pixel"
    r, c = map(int, coords[0])
    return r, c, defect_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Aggregate defect coordinates from repeated copies to address recolored stamp placements built from the unique shape.

    Input structure:
    - The background is black.
    - One foreground shape appears multiple times when colors are ignored; each copy has one body color plus exactly one defect pixel of a different color.
    - Another foreground shape appears exactly once and serves as the stamp content.
    - The stamp shape shares its color with one repeated copy, so color frequency does not determine its role.

    Rule:
    - Split the foreground into connected components and compare their cropped binary masks, ignoring colors.
    - The mask that occurs multiple times is the address shape; the mask that occurs once is the stamp shape S.
    - For each repeated copy, read the defect pixel's local bbox coordinate (r, c) and its color d, then paste S recolored to d into output block (r, c).
    - The output grid has one p-by-q stamp block for every cell in the repeated shape bbox, so its size is (h*p) by (w*q), and all unused blocks stay black.

    Key insight:
    - The repeated copies do not stamp themselves; each copy contributes exactly one address and one color through its lone defect pixel, while the unique shape provides the shared stamp mask.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    components = _find_foreground_components(g, bg=BACKGROUND)
    assert len(components) >= 3, "expected repeated copies plus one unique stamp"

    repeaters, stamp = _partition_roles(components, bg=BACKGROUND)
    repeater_mask = _component_mask(repeaters[0], bg=BACKGROUND)
    stamp_mask = _component_mask(stamp, bg=BACKGROUND)
    rh, rw = map(int, repeater_mask.shape)
    sh, sw = map(int, stamp_mask.shape)

    out = np.full((rh * sh, rw * sw), BACKGROUND, dtype=int)
    used_addresses: set[tuple[int, int]] = set()
    for repeater in repeaters:
        local_r, local_c, defect_color = _defect_info(repeater["crop"], bg=BACKGROUND)
        assert repeater_mask.shape == _component_mask(repeater, bg=BACKGROUND).shape
        assert np.array_equal(repeater_mask, _component_mask(repeater, bg=BACKGROUND))
        assert (local_r, local_c) not in used_addresses, "defect addresses must be unique"
        used_addresses.add((local_r, local_c))

        stamp_patch = np.where(stamp_mask, defect_color, BACKGROUND).astype(int)
        _paste(out, stamp_patch, local_r * sh, local_c * sw, transparent=BACKGROUND)

    return out


def _sample_shape(
    *,
    min_h: int,
    max_h: int,
    min_w: int,
    max_w: int,
    min_area: int,
    max_area: int,
    forbidden: set[tuple[tuple[int, int], bytes]] | None = None,
) -> np.ndarray | None:
    forbidden = forbidden or set()
    for _ in range(200):
        h = random.randint(min_h, max_h)
        w = random.randint(min_w, max_w)
        area_lo = max(min_area, max(h, w))
        area_hi = min(max_area, h * w - 1)
        if area_lo > area_hi:
            continue
        target_area = random.randint(area_lo, area_hi)

        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        frontier = {nb for nb in _neighbors8(*start) if 0 <= nb[0] < h and 0 <= nb[1] < w}
        while len(cells) < target_area and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            for nb in _neighbors8(*cell):
                if 0 <= nb[0] < h and 0 <= nb[1] < w and nb not in cells:
                    frontier.add(nb)

        if len(cells) != target_area:
            continue

        mask = np.zeros((h, w), dtype=bool)
        rows, cols = zip(*cells)
        mask[np.array(rows), np.array(cols)] = True

        used_rows = np.flatnonzero(mask.any(axis=1))
        used_cols = np.flatnonzero(mask.any(axis=0))
        if used_rows[0] != 0 or used_rows[-1] != h - 1 or used_cols[0] != 0 or used_cols[-1] != w - 1:
            continue

        signature = _shape_signature(mask)
        if signature in forbidden:
            continue
        return mask
    return None


def _render_repeater(mask: np.ndarray, main_color: int, defect_pos: tuple[int, int], defect_color: int) -> np.ndarray:
    sprite = np.where(mask, main_color, BACKGROUND).astype(int)
    sprite[defect_pos] = defect_color
    return sprite


def _render_stamp(mask: np.ndarray, color: int) -> np.ndarray:
    return np.where(mask, color, BACKGROUND).astype(int)


def _expected_output(
    repeater_mask: np.ndarray,
    stamp_mask: np.ndarray,
    defect_positions: list[tuple[int, int]],
    defect_colors: list[int],
) -> np.ndarray:
    rh, rw = repeater_mask.shape
    sh, sw = stamp_mask.shape
    out = np.zeros((rh * sh, rw * sw), dtype=int)
    for (r, c), color in zip(defect_positions, defect_colors):
        patch = _render_stamp(stamp_mask, color)
        _paste(out, patch, int(r) * sh, int(c) * sw, transparent=BACKGROUND)
    return out


def _place_components(crops: list[np.ndarray]) -> np.ndarray | None:
    max_h = max(int(crop.shape[0]) for crop in crops)
    max_w = max(int(crop.shape[1]) for crop in crops)
    padded_area = sum((int(crop.shape[0]) + 2) * (int(crop.shape[1]) + 2) for crop in crops)

    for _ in range(80):
        base = max(int(np.ceil(np.sqrt(padded_area * random.uniform(1.5, 2.2)))), max_h + 2, max_w + 2)
        h_min = min(30, max(10, base - 3))
        w_min = min(30, max(10, base - 3))
        if h_min > 30 or w_min > 30:
            continue
        h = random.randint(h_min, 30)
        w = random.randint(w_min, 30)

        grid = np.zeros((h, w), dtype=int)
        blocked = np.zeros((h, w), dtype=bool)
        order = list(range(len(crops)))
        random.shuffle(order)
        placed = True

        for idx in order:
            crop = crops[idx]
            ch, cw = map(int, crop.shape)
            success = False
            for _ in range(200):
                r0 = random.randint(0, h - ch)
                c0 = random.randint(0, w - cw)
                br0 = max(0, r0 - 1)
                bc0 = max(0, c0 - 1)
                br1 = min(h, r0 + ch + 1)
                bc1 = min(w, c0 + cw + 1)
                if blocked[br0:br1, bc0:bc1].any():
                    continue
                _paste(grid, crop, r0, c0, transparent=BACKGROUND)
                blocked[br0:br1, bc0:bc1] = True
                success = True
                break
            if not success:
                placed = False
                break

        if placed:
            return grid
    return None


def _gen_attempt() -> np.ndarray | None:
    repeater_mask = _sample_shape(
        min_h=2,
        max_h=5,
        min_w=2,
        max_w=5,
        min_area=4,
        max_area=10,
    )
    if repeater_mask is None:
        return None

    rh, rw = repeater_mask.shape
    stamp_mask = _sample_shape(
        min_h=2,
        max_h=min(5, 30 // rh),
        min_w=2,
        max_w=min(5, 30 // rw),
        min_area=3,
        max_area=9,
        forbidden={_shape_signature(repeater_mask)},
    )
    if stamp_mask is None:
        return None

    sh, sw = stamp_mask.shape
    if rh * sh > 30 or rw * sw > 30:
        return None

    defect_slots = [tuple(map(int, rc)) for rc in np.argwhere(repeater_mask)]
    max_copies = min(4, len(defect_slots), len(PALETTE) // 2)
    if max_copies < 2:
        return None
    copy_count = random.randint(2, max_copies)

    defect_positions = random.sample(defect_slots, copy_count)
    main_colors = random.sample(PALETTE, copy_count)
    remaining = [color for color in PALETTE if color not in main_colors]
    defect_colors = random.sample(remaining, copy_count)
    stamp_color = random.choice(main_colors)

    repeater_crops = [
        _render_repeater(repeater_mask, main_color, defect_pos, defect_color)
        for main_color, defect_pos, defect_color in zip(main_colors, defect_positions, defect_colors)
    ]
    stamp_crop = _render_stamp(stamp_mask, stamp_color)

    grid = _place_components([*repeater_crops, stamp_crop])
    if grid is None:
        return None

    expected = _expected_output(repeater_mask, stamp_mask, defect_positions, defect_colors)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for task 5")
