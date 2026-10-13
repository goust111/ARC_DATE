from __future__ import annotations

from collections import Counter

import numpy as np


_RNG = np.random.default_rng()
_BG = 0
_COLORS = np.arange(1, 10, dtype=int)
_GEN_RETRIES = 600


def _background_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors(r: int, c: int, h: int, w: int, connectivity: int):
    if connectivity == 4:
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        deltas = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]
    for dr, dc in deltas:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _component_coords(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
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
                for nr, nc in _neighbors(cr, cc, h, w, connectivity):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _find_sprites(grid: np.ndarray, bg: int) -> list[dict]:
    sprites = []
    for color in sorted(set(map(int, grid.ravel())) - {int(bg)}):
        comps = _component_coords(grid == color, connectivity=8)
        for coords in comps:
            bbox = _bbox_from_coords(coords)
            sprites.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": bbox,
                    "crop": _crop_bbox(grid, bbox),
                }
            )
    return sprites


def _count_holes(crop: np.ndarray, fg_color: int, bg: int) -> tuple[int, list[int]]:
    cavity = crop == bg
    h, w = crop.shape
    seen = np.zeros_like(cavity, dtype=bool)
    hole_areas: list[int] = []
    for r in range(h):
        for c in range(w):
            if not cavity[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            area = 0
            touches_border = False
            while stack:
                cr, cc = stack.pop()
                area += 1
                if cr in (0, h - 1) or cc in (0, w - 1):
                    touches_border = True
                for nr, nc in _neighbors(cr, cc, h, w, 4):
                    if cavity[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            if not touches_border:
                hole_areas.append(area)
    return len(hole_areas), sorted(hole_areas)


def _recolor_crop(crop: np.ndarray, bg: int, color: int) -> np.ndarray:
    out = np.full_like(crop, int(bg))
    out[crop != bg] = int(color)
    return out


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop same-bbox sprites and repack them along the layout axis, keeping colors in positional order but shapes in hole-count order.

    Input structure:
    - A background grid contains several separated monochrome sprites with identical bounding-box dimensions.
    - Each sprite has a unique color and a unique number of enclosed holes.
    - The sprite centers spread farther in rows or in columns, defining a vertical or horizontal layout axis.

    Rule:
    - Crop every sprite to its bounding box.
    - Sort the sprites by position along the layout axis, and separately sort them by enclosed-hole count from fewest to most.
    - Tightly concatenate bbox crops along the layout axis: output slot i uses the shape from hole-rank i, but recolors that entire sprite body with the color from position-rank i.

    Key insight:
    - The bodies are reordered by hole count while the colors stay attached to positional order, so the output is a zipper of shape order and color order rather than a direct pack-by-position.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    sprites = _find_sprites(g, bg)
    assert len(sprites) >= 2

    bbox_shapes = set()
    centers = []
    hole_counts = []
    colors = []
    for sprite in sprites:
        bbox = sprite["bbox"]
        r0, c0, r1, c1 = bbox
        bbox_shapes.add(sprite["crop"].shape)
        centers.append(((r0 + r1) / 2.0, (c0 + c1) / 2.0))
        holes, hole_areas = _count_holes(sprite["crop"], sprite["color"], bg)
        sprite["holes"] = int(holes)
        sprite["hole_areas"] = hole_areas
        hole_counts.append(int(holes))
        colors.append(int(sprite["color"]))

    assert len(bbox_shapes) == 1
    assert len(set(hole_counts)) == len(hole_counts)
    assert len(set(colors)) == len(colors)

    row_span = max(r for r, _ in centers) - min(r for r, _ in centers)
    col_span = max(c for _, c in centers) - min(c for _, c in centers)
    assert row_span != col_span

    horizontal = col_span > row_span
    if horizontal:
        pos_sorted = sorted(sprites, key=lambda s: (s["bbox"][1], s["bbox"][0]))
    else:
        pos_sorted = sorted(sprites, key=lambda s: (s["bbox"][0], s["bbox"][1]))
    hole_sorted = sorted(sprites, key=lambda s: s["holes"])

    box_h, box_w = next(iter(bbox_shapes))
    if horizontal:
        out = np.full((box_h, box_w * len(sprites)), bg, dtype=int)
        for i, (shape_sprite, color_sprite) in enumerate(zip(hole_sorted, pos_sorted)):
            block = _recolor_crop(shape_sprite["crop"], bg, color_sprite["color"])
            _paste(out, block, 0, i * box_w)
    else:
        out = np.full((box_h * len(sprites), box_w), bg, dtype=int)
        for i, (shape_sprite, color_sprite) in enumerate(zip(hole_sorted, pos_sorted)):
            block = _recolor_crop(shape_sprite["crop"], bg, color_sprite["color"])
            _paste(out, block, i * box_h, 0)

    return out


def _sample_nonadjacent_seeds(inner_h: int, inner_w: int, count: int) -> list[tuple[int, int]] | None:
    cells = [(r, c) for r in range(1, inner_h + 1) for c in range(1, inner_w + 1)]
    _RNG.shuffle(cells)
    seeds: list[tuple[int, int]] = []
    for r, c in cells:
        if all(abs(r - rr) + abs(c - cc) > 1 for rr, cc in seeds):
            seeds.append((r, c))
            if len(seeds) == count:
                return seeds
    return None


def _grow_holes(box_h: int, box_w: int, hole_count: int) -> tuple[np.ndarray, list[int], int] | None:
    color = 1
    sprite = np.full((box_h, box_w), color, dtype=int)
    if hole_count == 0:
        return sprite, [], 0

    inner_h, inner_w = box_h - 2, box_w - 2
    if inner_h <= 0 or inner_w <= 0:
        return None
    seeds = _sample_nonadjacent_seeds(inner_h, inner_w, hole_count)
    if seeds is None:
        return None

    components = [{seed} for seed in seeds]
    occupied = set(seeds)
    sizes = []
    for i in range(hole_count):
        upper = 5 if hole_count <= 2 else 4
        sizes.append(int(_RNG.integers(1, upper + 1)))
    # Bias at least one low-count example toward a visibly larger hole.
    if hole_count == 1:
        sizes[0] = max(sizes[0], int(_RNG.integers(2, 6)))

    for comp_index, comp in enumerate(components):
        target = sizes[comp_index]
        while len(comp) < target:
            frontier = []
            for r, c in list(comp):
                for nr, nc in _neighbors(r, c, box_h, box_w, 4):
                    if nr in (0, box_h - 1) or nc in (0, box_w - 1):
                        continue
                    if (nr, nc) in occupied:
                        continue
                    bad = False
                    for orow, ocol in occupied:
                        if (orow, ocol) in comp:
                            continue
                        if abs(nr - orow) + abs(nc - ocol) <= 1:
                            bad = True
                            break
                    if not bad:
                        frontier.append((nr, nc))
            if not frontier:
                break
            pick = frontier[int(_RNG.integers(len(frontier)))]
            comp.add(pick)
            occupied.add(pick)

    for comp in components:
        for r, c in comp:
            sprite[r, c] = _BG

    holes, hole_areas = _count_holes(sprite, color, _BG)
    if holes != hole_count:
        return None
    return sprite, hole_areas, int(sum(hole_areas))


def _make_shape_set(count: int, box_h: int, box_w: int) -> list[dict] | None:
    shapes = []
    for holes in range(count):
        built = _grow_holes(box_h, box_w, holes)
        if built is None:
            return None
        sprite, hole_areas, total_area = built
        shapes.append(
            {
                "holes": holes,
                "mask_crop": sprite,
                "hole_areas": hole_areas,
                "total_hole_area": total_area,
            }
        )

    total_areas = [item["total_hole_area"] for item in shapes]
    area_order = sorted(range(count), key=lambda i: (total_areas[i], i))
    if area_order == list(range(count)) or area_order == list(range(count - 1, -1, -1)):
        return None
    if not any(any(area > 1 for area in item["hole_areas"]) for item in shapes if item["holes"] > 0):
        return None
    return shapes


def _colorize(mask_crop: np.ndarray, color: int) -> np.ndarray:
    out = np.zeros_like(mask_crop, dtype=int)
    out[mask_crop != 0] = int(color)
    return out


def _sample_axis_positions(box_len: int, count: int, limit: int) -> list[int] | None:
    min_len = 2 + count * box_len + (count - 1)
    if min_len > limit:
        return None
    length = int(_RNG.integers(min_len, limit + 1))
    extra = length - min_len
    buckets = np.zeros(count + 1, dtype=int)
    for _ in range(extra):
        buckets[int(_RNG.integers(0, count + 1))] += 1
    starts = [1 + int(buckets[0])]
    for i in range(1, count):
        gap = 1 + int(buckets[i])
        starts.append(starts[-1] + box_len + gap)
    if starts[-1] + box_len > length - 1:
        return None
    return starts


def _sample_cross_positions(box_len: int, count: int, limit: int) -> list[int] | None:
    if box_len + 2 > limit:
        return None
    max_start = limit - box_len - 1
    min_start = 1
    span_max = min(3, max_start - min_start)
    if span_max < 0:
        return None
    span = int(_RNG.integers(0, span_max + 1))
    if count >= 3 and span == 0 and span_max > 0:
        span = 1
    base = int(_RNG.integers(min_start, max_start - span + 2))
    starts = [base + int(_RNG.integers(0, span + 1)) for _ in range(count)]
    if span > 0 and len(set(starts)) == 1:
        starts[-1] = min(max_start, starts[-1] + span)
    return starts


def _expected_output(box_h: int, box_w: int, horizontal: bool, shapes: list[dict], pos_colors: list[int]) -> np.ndarray:
    if horizontal:
        out = np.full((box_h, box_w * len(shapes)), _BG, dtype=int)
        for i, (shape, color) in enumerate(zip(shapes, pos_colors)):
            _paste(out, _colorize(shape["mask_crop"], color), 0, i * box_w)
    else:
        out = np.full((box_h * len(shapes), box_w), _BG, dtype=int)
        for i, (shape, color) in enumerate(zip(shapes, pos_colors)):
            _paste(out, _colorize(shape["mask_crop"], color), i * box_h, 0)
    return out


def _build_input() -> tuple[np.ndarray, np.ndarray] | None:
    count = int(_RNG.integers(3, 5))
    box_h = int(_RNG.integers(5, 8))
    box_w = int(_RNG.integers(5, 8))
    shapes = _make_shape_set(count, box_h, box_w)
    if shapes is None:
        return None

    perm = list(_RNG.permutation(count))
    if perm == list(range(count)):
        return None

    horizontal = bool(_RNG.integers(0, 2))
    axis_box = box_w if horizontal else box_h
    cross_box = box_h if horizontal else box_w
    min_axis_len = 2 + count * axis_box + (count - 1)
    min_cross_len = cross_box + 2
    if min_axis_len > 30 or min_cross_len > 30:
        return None

    axis_len = int(_RNG.integers(min_axis_len, min(30, min_axis_len + 8) + 1))
    cross_len = int(_RNG.integers(min_cross_len, min(18, min_cross_len + 6, 30) + 1))

    if horizontal:
        grid_h, grid_w = cross_len, axis_len
        axis_starts = _sample_axis_positions(box_w, count, grid_w)
        cross_starts = _sample_cross_positions(box_h, count, grid_h)
    else:
        grid_h, grid_w = axis_len, cross_len
        axis_starts = _sample_axis_positions(box_h, count, grid_h)
        cross_starts = _sample_cross_positions(box_w, count, grid_w)
    if axis_starts is None or cross_starts is None:
        return None

    colors = [int(c) for c in _RNG.choice(_COLORS, size=count, replace=False)]
    inp = np.zeros((grid_h, grid_w), dtype=int)
    pos_colors = []
    for pos_index, shape_index in enumerate(perm):
        color = colors[pos_index]
        pos_colors.append(color)
        patch = _colorize(shapes[shape_index]["mask_crop"], color)
        if horizontal:
            _paste(inp, patch, cross_starts[pos_index], axis_starts[pos_index])
        else:
            _paste(inp, patch, axis_starts[pos_index], cross_starts[pos_index])

    centers = []
    for pos_index in range(count):
        if horizontal:
            r0 = cross_starts[pos_index]
            c0 = axis_starts[pos_index]
        else:
            r0 = axis_starts[pos_index]
            c0 = cross_starts[pos_index]
        centers.append((r0 + (box_h - 1) / 2.0, c0 + (box_w - 1) / 2.0))
    row_span = max(r for r, _ in centers) - min(r for r, _ in centers)
    col_span = max(c for _, c in centers) - min(c for _, c in centers)
    if horizontal and not (col_span > row_span):
        return None
    if (not horizontal) and not (row_span > col_span):
        return None

    expected = _expected_output(box_h, box_w, horizontal, shapes, pos_colors)

    sprite_info = _find_sprites(inp, _BG)
    if len(sprite_info) != count:
        return None
    pos_sorted = sorted(
        sprite_info,
        key=(lambda s: (s["bbox"][1], s["bbox"][0])) if horizontal else (lambda s: (s["bbox"][0], s["bbox"][1])),
    )
    inferred_perm = []
    inferred_counts = []
    for sprite in pos_sorted:
        holes, _ = _count_holes(sprite["crop"], sprite["color"], _BG)
        inferred_perm.append(int(holes))
        inferred_counts.append(int(holes))
    if inferred_perm == list(range(count)):
        return None

    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        built = _build_input()
        if built is None:
            continue
        inp, expected = built
        try:
            out = main(inp)
        except Exception:
            continue
        if not np.array_equal(out, expected):
            continue
        if inp.ndim != 2 or out.ndim != 2:
            continue
        if max(inp.shape) > 30 or max(out.shape) > 30:
            continue
        return inp
    raise RuntimeError("failed to generate a valid input")
