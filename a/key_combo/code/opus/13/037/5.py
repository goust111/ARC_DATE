from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
GREY = 5
BAND_H = 3
SPRITE_COLORS = (1, 2, 3, 4, 7)
MAIN_HEIGHT_OPTIONS = (16, 17)
GRID_WIDTH_RANGE = (19, 24)
GEN_RETRIES = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(cells, dtype=int))
    out.sort(key=lambda coords: _bbox_from_coords(coords)[:2])
    return out


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in mask.astype(int))


def _crop_patch(grid: np.ndarray, coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste_transparent(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    sub = dst[r0 : r0 + h, c0 : c0 + w]
    mask = patch != BG
    sub[mask] = patch[mask]


def _shade_patch(patch: np.ndarray, extra_count: int) -> np.ndarray:
    shaded = patch.copy()
    occupied = np.argwhere(shaded != BG)
    assert 0 <= extra_count < len(occupied), "grey replacements must leave a visible sprite remainder"
    for r, c in occupied[:extra_count]:
        shaded[int(r), int(c)] = GREY
    return shaded


def _analyze_input(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] > BAND_H and g.shape[1] >= 15
    assert np.all(g[-BAND_H:, :] == GREY), "bottom catalog band must be solid grey in the input"
    assert not np.any(g[:-BAND_H, :] == GREY), "grey is reserved for the catalog band in the input"

    main_area = g[:-BAND_H, :]
    components = _components4(main_area != BG)
    assert 10 <= len(components) <= 18, "sprite count must match the puzzle spec"

    grouped: dict[tuple[int, tuple[tuple[int, ...], ...]], dict] = {}
    for coords in components:
        patch = _crop_patch(main_area, coords)
        colors = sorted(set(int(v) for v in patch.ravel()) - {BG})
        assert len(colors) == 1, "each sprite must be monochrome"
        color = colors[0]
        assert color in SPRITE_COLORS
        h, w = patch.shape
        assert h <= 3 and w <= 3, "sprite bbox must fit inside 3x3"
        key = (color, _mask_key(patch != BG))
        if key not in grouped:
            grouped[key] = {
                "key": key,
                "color": color,
                "patch": patch,
                "count": 0,
                "first_bbox": _bbox_from_coords(coords),
            }
        grouped[key]["count"] += 1
        if _bbox_from_coords(coords)[:2] < grouped[key]["first_bbox"][:2]:
            grouped[key]["first_bbox"] = _bbox_from_coords(coords)

    types = list(grouped.values())
    assert 3 <= len(types) <= 4, "there must be 3-4 distinct sprite types"

    counts = [info["count"] for info in types]
    assert min(counts) >= 2 and max(counts) <= 6
    assert len(counts) == len(set(counts)), "type frequencies must be unique"
    return types


def _render_catalog_output(types: list[dict], shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    out = np.zeros((h, w), dtype=int)
    out[-BAND_H:, :] = GREY

    min_count = min(info["count"] for info in types)
    ordered = sorted(types, key=lambda info: info["count"])
    col = 0
    for info in ordered:
        patch = _shade_patch(info["patch"], info["count"] - min_count)
        ph, pw = patch.shape
        assert ph <= BAND_H and pw <= 3
        assert col + 3 <= w, "catalog band is too narrow"
        _paste_transparent(out, patch, h - BAND_H, col)
        col += 4
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Deduplicate scattered sprites, sort the types by frequency, and stamp the ordered catalog into the bottom grey strip.

    Input structure:
    - Black is the background, and the lowest 3 rows are a full-width grey catalog band.
    - Above the band, 10-18 small monochrome 4-connected sprites are scattered with bounding boxes no larger than 3x3.
    - There are 3-4 distinct sprite types, where a type is defined by its normalized bbox mask together with its color.

    Rule:
    - Detect every 4-connected sprite in the main area and normalize each one to its bbox crop.
    - Group identical normalized colored sprites, count the occurrences of each type, and sort the unique types by ascending count.
    - Clear the main area to black, then stamp one copy of each unique type into the bottom grey band from left to right in 3-column slots separated by 1 grey column.
    - The least frequent type keeps its original color; every extra occurrence beyond that minimum recolors one additional row-major occupied cell of its catalog stamp to grey.

    Key insight:
    - The output is a catalog assembled from all sprite types, so solving requires both deduplication and global frequency counting; the hidden grey replacements also depend on those same counts.
    """
    types = _analyze_input(grid)
    return _render_catalog_output(types, np.asarray(grid, dtype=int).shape)


def _shape_pool() -> list[np.ndarray]:
    pool: dict[tuple[tuple[int, ...], ...], np.ndarray] = {}
    for bits in range(1, 1 << 9):
        mask = np.zeros((3, 3), dtype=bool)
        for idx in range(9):
            if bits & (1 << idx):
                mask[idx // 3, idx % 3] = True
        area = int(mask.sum())
        if not (5 <= area <= 8):
            continue
        if len(_components4(mask)) != 1:
            continue
        coords = np.argwhere(mask)
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        cropped = mask[r0 : r1 + 1, c0 : c1 + 1]
        pool.setdefault(_mask_key(cropped), cropped.copy())
    return list(pool.values())


SHAPE_POOL = _shape_pool()
COUNT_OPTIONS = [
    tuple(counts)
    for k in (3, 4)
    for counts in itertools.combinations(range(2, 7), k)
    if 10 <= sum(counts) <= 18
]


def _patch_from_shape(shape: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(shape.shape, dtype=int)
    patch[shape] = int(color)
    return patch


def _pick_type_specs(k: int) -> tuple[list[dict], str]:
    mode = random.choice(("distinct", "same_shape", "same_color"))

    if mode == "same_shape":
        colors = random.sample(SPRITE_COLORS, k)
        shared = random.choice(SHAPE_POOL)
        others = random.sample([shape for shape in SHAPE_POOL if _mask_key(shape) != _mask_key(shared)], k - 2)
        shapes = [shared, shared, *others]
        random.shuffle(shapes)
        specs = [{"shape": shape.copy(), "color": color} for shape, color in zip(shapes, colors)]
    elif mode == "same_color":
        shared_color = random.choice(SPRITE_COLORS)
        other_colors = random.sample([c for c in SPRITE_COLORS if c != shared_color], k - 2)
        colors = [shared_color, shared_color, *other_colors]
        random.shuffle(colors)
        shapes = random.sample(SHAPE_POOL, k)
        specs = [{"shape": shape.copy(), "color": color} for shape, color in zip(shapes, colors)]
    else:
        colors = random.sample(SPRITE_COLORS, k)
        shapes = random.sample(SHAPE_POOL, k)
        specs = [{"shape": shape.copy(), "color": color} for shape, color in zip(shapes, colors)]

    keys = {(spec["color"], _mask_key(spec["shape"])) for spec in specs}
    assert len(keys) == k
    return specs, mode


def _catalog_tiles_from_types(types: list[dict]) -> list[np.ndarray]:
    min_count = min(info["count"] for info in types)
    tiles = []
    for info in sorted(types, key=lambda item: item["count"]):
        tiles.append(_shade_patch(info["patch"], info["count"] - min_count))
    return tiles


def _place_instances(specs: list[dict], main_h: int, width: int) -> np.ndarray | None:
    grid = np.zeros((main_h + BAND_H, width), dtype=int)
    grid[-BAND_H:, :] = GREY

    slots = [(r, c) for r in range(0, main_h - 2, 4) for c in range(0, width - 2, 4)]
    total_instances = sum(spec["count"] for spec in specs)
    if len(slots) < total_instances:
        return None

    chosen_slots = random.sample(slots, total_instances)
    instances = [idx for idx, spec in enumerate(specs) for _ in range(spec["count"])]
    random.shuffle(instances)

    for (r0, c0), idx in zip(chosen_slots, instances):
        spec = specs[idx]
        patch = _patch_from_shape(spec["shape"], spec["color"])
        h, w = patch.shape
        _paste_transparent(grid, patch, r0, c0)
        spec.setdefault("bboxes", []).append((r0, c0, r0 + h - 1, c0 + w - 1))
    return grid


def _type_order_signature(types: list[dict], key_name: str) -> tuple[tuple[int, tuple[tuple[int, ...], ...]], ...]:
    if key_name == "count":
        ordered = sorted(types, key=lambda info: info["count"])
    elif key_name == "area":
        ordered = sorted(types, key=lambda info: int((info["patch"] != BG).sum()))
    else:
        ordered = sorted(types, key=lambda info: info["first_bbox"][:2])
    return tuple(info["key"] for info in ordered)


def _validate_semantics(grid: np.ndarray, expected: np.ndarray, mode: str) -> None:
    types = _analyze_input(grid)
    assert np.array_equal(_render_catalog_output(types, grid.shape), expected)

    freq_order = _type_order_signature(types, "count")
    first_order = _type_order_signature(types, "first")
    assert freq_order != first_order, "reject inputs where first-seen order matches frequency order"

    areas = [int((info["patch"] != BG).sum()) for info in types]
    if len(set(areas)) == len(areas):
        area_order = _type_order_signature(types, "area")
        assert freq_order != area_order, "reject inputs where area order matches frequency order"
    else:
        same_shape_diff_color = any(
            types[i]["key"][1] == types[j]["key"][1] and types[i]["color"] != types[j]["color"]
            for i in range(len(types))
            for j in range(i + 1, len(types))
        )
        same_color_diff_shape = any(
            types[i]["color"] == types[j]["color"] and types[i]["key"][1] != types[j]["key"][1]
            for i in range(len(types))
            for j in range(i + 1, len(types))
        )
        assert same_shape_diff_color or same_color_diff_shape

    tiles = _catalog_tiles_from_types(types)
    tile_keys = []
    for tile in tiles:
        slot = np.full((3, 3), GREY, dtype=int)
        slot[: tile.shape[0], : tile.shape[1]] = np.where(tile != BG, tile, slot[: tile.shape[0], : tile.shape[1]])
        tile_keys.append(tuple(tuple(int(v) for v in row) for row in slot))
    assert len(tile_keys) == len(set(tile_keys)), "catalog entries must remain visually distinct"

    if mode == "same_shape":
        assert any(
            types[i]["key"][1] == types[j]["key"][1] and types[i]["color"] != types[j]["color"]
            for i in range(len(types))
            for j in range(i + 1, len(types))
        )
    elif mode == "same_color":
        assert any(
            types[i]["color"] == types[j]["color"] and types[i]["key"][1] != types[j]["key"][1]
            for i in range(len(types))
            for j in range(i + 1, len(types))
        )


def _generate_once() -> np.ndarray | None:
    main_h = random.choice(MAIN_HEIGHT_OPTIONS)
    width = random.randint(*GRID_WIDTH_RANGE)
    k = random.choice((3, 4))
    counts = list(random.choice([counts for counts in COUNT_OPTIONS if len(counts) == k]))
    specs, mode = _pick_type_specs(k)
    random.shuffle(specs)
    for spec, count in zip(specs, counts):
        area = int(spec["shape"].sum())
        if count - min(counts) >= area:
            return None
        spec["count"] = int(count)

    if 4 * k - 1 > width:
        return None

    grid = _place_instances(specs, main_h, width)
    if grid is None:
        return None

    expected_types = []
    for spec in specs:
        expected_types.append(
            {
                "key": (spec["color"], _mask_key(spec["shape"])),
                "color": spec["color"],
                "patch": _patch_from_shape(spec["shape"], spec["color"]),
                "count": spec["count"],
                "first_bbox": min(spec["bboxes"]),
            }
        )
    expected = _render_catalog_output(expected_types, grid.shape)

    try:
        _validate_semantics(grid, expected, mode)
        if not np.array_equal(main(grid), expected):
            return None
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_once()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
