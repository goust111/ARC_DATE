from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

_TOOLBOX = Path("/home/lixinyu/snap/new/solvers")
if _TOOLBOX.exists():
    toolbox_str = str(_TOOLBOX)
    if toolbox_str not in sys.path:
        sys.path.insert(0, toolbox_str)

from arc_utils import bbox_coords, components_mask, crop_bbox, find_components, get_bg, paste


BG = 0
FRAME = 5
WALL = 6
SPRITE_COLORS = (1, 2, 3, 4)
GEN_ATTEMPTS = 300


def _shape_variants(mask: np.ndarray) -> list[np.ndarray]:
    seen: set[tuple[tuple[int, int], bytes]] = set()
    variants: list[np.ndarray] = []
    base = mask.astype(bool)
    for flip in (False, True):
        flipped = np.fliplr(base) if flip else base
        for k in range(4):
            rotated = np.rot90(flipped, k)
            key = (rotated.shape, rotated.tobytes())
            if key not in seen:
                seen.add(key)
                variants.append(rotated.copy())
    return variants


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return min((variant.shape, variant.tobytes()) for variant in _shape_variants(mask))


def _mask_from_coords(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    r0, c0, r1, c1 = bbox_coords(coords)
    rel = coords - np.array([r0, c0])
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[rel[:, 0], rel[:, 1]] = True
    return mask, (int(r0), int(c0))


def _find_frame(grid: np.ndarray) -> dict:
    comps = find_components(grid, bg=BG, connectivity=4, monochromatic=True, colors=[FRAME], return_format="dict")
    assert comps, "expected a grey template frame"
    frame = max(comps, key=lambda comp: int(comp["area"]))
    return frame


def _extract_holes(grid: np.ndarray, frame_bbox: tuple[int, int, int, int]) -> list[dict]:
    sub = crop_bbox(grid, frame_bbox)
    holes = []
    for coords in components_mask(sub == BG, connectivity=4):
        mask, (r0, c0) = _mask_from_coords(coords)
        holes.append(
            {
                "mask": mask,
                "local_top_left": (int(r0), int(c0)),
                "key": _shape_key(mask),
                "area": int(mask.sum()),
                "bbox": (frame_bbox[0] + int(r0), frame_bbox[1] + int(c0), frame_bbox[0] + int(r0) + mask.shape[0] - 1, frame_bbox[1] + int(c0) + mask.shape[1] - 1),
            }
        )
    return holes


def _find_wall_columns(grid: np.ndarray, frame_top: int) -> list[int]:
    wall_zone = grid[:frame_top, :]
    cols = []
    for comp in find_components(wall_zone, bg=BG, connectivity=4, monochromatic=True, colors=[WALL], return_format="dict"):
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if c0 == c1 and int(comp["area"]) >= 3:
            cols.append(c0)
    cols = sorted(set(cols))
    assert cols, "expected vertical wall segments"
    return cols


def _channel_intervals(width: int, wall_cols: list[int]) -> list[tuple[int, int]]:
    bounds = [-1, *wall_cols, width]
    return [(bounds[i] + 1, bounds[i + 1] - 1) for i in range(len(bounds) - 1) if bounds[i] + 1 <= bounds[i + 1] - 1]


def _holes_by_channel(holes: list[dict], intervals: list[tuple[int, int]]) -> list[dict]:
    channel_records = []
    for left, right in intervals:
        channel_holes = []
        for hole in holes:
            _r0, c0, _r1, c1 = hole["bbox"]
            if left <= c0 and c1 <= right:
                channel_holes.append(hole)
        if channel_holes:
            assert len(channel_holes) == 1, "each channel must correspond to exactly one hole"
            channel_records.append({"interval": (int(left), int(right)), "hole": channel_holes[0]})
    assert channel_records, "expected channels covering holes"
    return channel_records


def _find_sprites(grid: np.ndarray, frame_bbox: tuple[int, int, int, int], channels: list[dict]) -> dict[int, dict]:
    r0, _c0, _r1, _c1 = frame_bbox
    sprites_by_channel: dict[int, dict] = {}
    comps = find_components(grid, bg=BG, connectivity=4, monochromatic=True, return_format="dict")
    for comp in comps:
        color = int(comp["color"])
        if color in (FRAME, WALL):
            continue
        br0, bc0, br1, bc1 = map(int, comp["bbox"])
        if br0 >= r0:
            continue
        if int(comp["area"]) <= 3:
            continue
        channel_index = None
        for idx, channel in enumerate(channels):
            left, right = channel["interval"]
            if left <= bc0 and bc1 <= right:
                channel_index = idx
                break
        assert channel_index is not None, "sprite must sit inside a wall-defined channel"
        assert channel_index not in sprites_by_channel, "at most one sprite per channel"
        mask, (sr0, sc0) = _mask_from_coords(comp["coords"])
        sprites_by_channel[channel_index] = {
            "color": color,
            "coords": comp["coords"],
            "mask": mask,
            "top_left": (int(sr0), int(sc0)),
            "key": _shape_key(mask),
        }
    return sprites_by_channel


def _drop_top_left(grid: np.ndarray, sprite_mask: np.ndarray, top_left: tuple[int, int]) -> tuple[int, int]:
    sr, sc = map(int, top_left)
    clear = np.array(grid, copy=True)
    patch = np.full(sprite_mask.shape, BG, dtype=int)
    paste(clear, patch, sr, sc, mask=sprite_mask)

    dist = 0
    while True:
        nr = sr + dist + 1
        if nr + sprite_mask.shape[0] > clear.shape[0]:
            break
        target = clear[nr : nr + sprite_mask.shape[0], sc : sc + sprite_mask.shape[1]]
        if np.any(target[sprite_mask] != BG):
            break
        dist += 1
    return sr + dist, sc


def _apply_rule_with_objects(grid: np.ndarray, channels: list[dict], sprites_by_channel: dict[int, dict]) -> np.ndarray:
    out = np.array(grid, copy=True)
    for idx, channel in enumerate(channels):
        sprite = sprites_by_channel.get(idx)
        if sprite is None:
            continue

        sr, sc = sprite["top_left"]
        erase = np.full(sprite["mask"].shape, BG, dtype=int)
        paste(out, erase, sr, sc, mask=sprite["mask"])

        if sprite["key"] == channel["hole"]["key"]:
            hr0, hc0 = channel["hole"]["local_top_left"]
            hole_color_patch = np.full(channel["hole"]["mask"].shape, int(sprite["color"]), dtype=int)
            frame_r0, frame_c0, _frame_r1, _frame_c1 = channel["hole"]["bbox"]
            paste(out, hole_color_patch, frame_r0, frame_c0, mask=channel["hole"]["mask"])
        else:
            dr, dc = _drop_top_left(out, sprite["mask"], (sr, sc))
            sprite_patch = np.full(sprite["mask"].shape, int(sprite["color"]), dtype=int)
            paste(out, sprite_patch, dr, dc, mask=sprite["mask"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop each sprite straight down inside its wall-defined channel and fill only the matching hole directly below it.

    Input structure:
    - The background is black, with one large grey template frame in the lower part of the grid.
    - Inside that frame are several black 4-connected holes of different shapes.
    - Above the frame are colored sprite components and one-cell-wide magenta vertical wall segments that split the upper area into channels.

    Rule:
    - Identify the grey frame, its internal holes, and the wall columns that partition the upper area into left-to-right channels.
    - Each sprite belongs to exactly one channel and may only interact with the single hole whose columns lie directly below that channel.
    - Remove the sprite from its start position, drop it vertically until one more row would hit the frame, a wall, or the grid bottom, then compare its shape with the channel's hole under all rotations and mirror flips.
    - If the shapes match, erase the sprite and fill the hole with the sprite's color; otherwise leave the sprite at its landing position. Walls and the grey frame stay unchanged.

    Key insight:
    - The magenta walls do more than block space: they assign each sprite to one reachable hole, so shape matching is local channel-by-channel rather than a global search over all holes.
    """
    g = np.asarray(grid, dtype=int)
    frame = _find_frame(g)
    frame_bbox = tuple(map(int, frame["bbox"]))
    holes = _extract_holes(g, frame_bbox)
    wall_cols = _find_wall_columns(g, frame_bbox[0])
    channels = _holes_by_channel(holes, _channel_intervals(g.shape[1], wall_cols))
    sprites_by_channel = _find_sprites(g, frame_bbox, channels)
    return _apply_rule_with_objects(g, channels, sprites_by_channel)


def _make_mask(lines: list[str]) -> np.ndarray:
    return np.array([[ch == "#" for ch in row] for row in lines], dtype=bool)


_RAW_LIBRARY = [
    ["##", "##"],
    ["#.", "##", "#."],
    ["###", ".#."],
    [".##", "##."],
    ["###", "#.."],
    ["###", "#.#"],
    ["###", ".#.", ".#."],
    ["###", "##."],
    ["##.", "###"],
    ["#..", "###", ".#."],
    [".#.", "###", ".#."],
    ["###", "###"],
    ["###", "##.", "#.."],
    ["###", "#.#", "#.."],
    [".##", "###", ".#."],
    ["###", "###", ".#."],
]


def _build_shape_library() -> list[np.ndarray]:
    seen: set[tuple[tuple[int, int], bytes]] = set()
    library: list[np.ndarray] = []
    for lines in _RAW_LIBRARY:
        mask = _make_mask(lines)
        area = int(mask.sum())
        if not (4 <= area <= 8):
            continue
        key = _shape_key(mask)
        if key not in seen:
            seen.add(key)
            library.append(mask)
    return library


SHAPE_LIBRARY = _build_shape_library()
LIBRARY_BY_AREA: dict[int, list[np.ndarray]] = {}
for shape in SHAPE_LIBRARY:
    LIBRARY_BY_AREA.setdefault(int(shape.sum()), []).append(shape)


def _random_oriented(mask: np.ndarray) -> np.ndarray:
    return random.choice(_shape_variants(mask))


def _choose_hole_shapes(n_channels: int) -> list[np.ndarray]:
    area = random.choice([a for a, shapes in LIBRARY_BY_AREA.items() if len(shapes) >= 2])
    hole_shapes = random.sample(LIBRARY_BY_AREA[area], 2)
    used = {_shape_key(shape) for shape in hole_shapes}
    while len(hole_shapes) < n_channels:
        candidate = random.choice(SHAPE_LIBRARY)
        key = _shape_key(candidate)
        if key not in used:
            used.add(key)
            hole_shapes.append(candidate)
    random.shuffle(hole_shapes)
    return hole_shapes


def _choose_sprite_bases(hole_bases: list[np.ndarray], match_indices: set[int], mismatch_same_area_idx: int) -> list[np.ndarray]:
    hole_keys = {_shape_key(shape) for shape in hole_bases}
    sprite_bases: list[np.ndarray] = []
    for idx, hole in enumerate(hole_bases):
        if idx in match_indices:
            sprite_bases.append(hole)
            continue
        area = int(hole.sum())
        pool = [shape for shape in LIBRARY_BY_AREA[area] if _shape_key(shape) not in hole_keys]
        if not pool:
            pool = [shape for shape in SHAPE_LIBRARY if _shape_key(shape) not in hole_keys]
        if idx == mismatch_same_area_idx:
            assert pool, "expected a non-hole mismatch shape with the same area"
        sprite_bases.append(random.choice(pool))
    return sprite_bases


def _place_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    patch = np.full(mask.shape, int(color), dtype=int)
    paste(grid, patch, int(top), int(left), mask=mask)


def _build_instance() -> tuple[np.ndarray, np.ndarray, dict]:
    n_channels = random.randint(3, 4)
    hole_bases = _choose_hole_shapes(n_channels)
    match_count = random.randint(1, n_channels - 1)
    match_indices = set(random.sample(range(n_channels), match_count))
    mismatch_indices = [idx for idx in range(n_channels) if idx not in match_indices]
    mismatch_same_area_idx = random.choice(mismatch_indices)
    sprite_bases = _choose_sprite_bases(hole_bases, match_indices, mismatch_same_area_idx)

    hole_masks = [_random_oriented(shape) for shape in hole_bases]
    sprite_masks = [_random_oriented(shape) for shape in sprite_bases]
    slot_widths = [max(hole_masks[i].shape[1], sprite_masks[i].shape[1]) + 2 for i in range(n_channels)]
    inner_width = sum(slot_widths) + (n_channels - 1)
    frame_w = inner_width + 2
    max_hole_h = max(mask.shape[0] for mask in hole_masks)
    max_sprite_h = max(mask.shape[0] for mask in sprite_masks)
    inner_h = max(max_hole_h + 3, 6)
    frame_h = inner_h + 2

    frame_top_min = max(max_sprite_h + 4, 8)
    frame_top_max = min(13, 29 - frame_h)
    assert frame_top_min <= frame_top_max
    frame_top = random.randint(frame_top_min, frame_top_max)

    left_margin = random.randint(1, 2)
    right_margin = random.randint(1, 2)
    width = left_margin + frame_w + right_margin
    assert width <= 30
    bottom_margin = random.randint(1, min(3, 30 - (frame_top + frame_h)))
    height = frame_top + frame_h + bottom_margin
    assert height <= 30

    frame_left = left_margin
    frame_bbox = (frame_top, frame_left, frame_top + frame_h - 1, frame_left + frame_w - 1)
    grid = np.zeros((height, width), dtype=int)
    grid[frame_bbox[0] : frame_bbox[2] + 1, frame_bbox[1] : frame_bbox[3] + 1] = FRAME

    channels = []
    cursor = frame_left + 1
    wall_cols = []
    for idx, slot_w in enumerate(slot_widths):
        left = cursor
        right = cursor + slot_w - 1
        hole = hole_masks[idx]
        max_h_offset = slot_w - hole.shape[1] - 1
        hole_left = left + random.randint(1, max_h_offset)
        max_v_offset = inner_h - hole.shape[0] - 1
        hole_top = frame_top + 1 + random.randint(1, max_v_offset)
        hole_patch = np.full(hole.shape, BG, dtype=int)
        paste(grid, hole_patch, hole_top, hole_left, mask=hole)
        channel = {
            "interval": (left, right),
            "hole": {
                "mask": hole,
                "bbox": (hole_top, hole_left, hole_top + hole.shape[0] - 1, hole_left + hole.shape[1] - 1),
                "local_top_left": (hole_top - frame_top, hole_left - frame_left),
                "key": _shape_key(hole),
                "area": int(hole.sum()),
            },
            "index": idx,
        }
        channels.append(channel)
        cursor = right + 1
        if idx != n_channels - 1:
            wall_cols.append(cursor)
            cursor += 1

    for wall_col in wall_cols:
        wall_top = random.randint(0, min(2, frame_top - 4))
        wall_bottom = frame_top - 2
        grid[wall_top : wall_bottom + 1, wall_col] = WALL

    sprite_colors = random.sample(list(SPRITE_COLORS), n_channels)
    sprites_by_channel: dict[int, dict] = {}
    for idx, channel in enumerate(channels):
        mask = sprite_masks[idx]
        left, right = channel["interval"]
        min_c = left + 1
        max_c = right - mask.shape[1]
        assert min_c <= max_c
        sc = random.randint(min_c, max_c)
        max_r = frame_top - mask.shape[0] - 3
        assert max_r >= 0
        sr = random.randint(0, max_r)
        color = sprite_colors[idx]
        _place_mask(grid, mask, sr, sc, color)
        sprites_by_channel[idx] = {
            "mask": mask,
            "top_left": (sr, sc),
            "color": color,
            "key": _shape_key(mask),
        }

    expected = _apply_rule_with_objects(grid, channels, sprites_by_channel)
    meta = {
        "frame_bbox": frame_bbox,
        "channels": channels,
        "sprites_by_channel": sprites_by_channel,
        "wall_cols": wall_cols,
        "match_indices": match_indices,
        "mismatch_same_area_idx": mismatch_same_area_idx,
    }
    return grid, expected, meta


def _derived_nonlocal_variant(grid: np.ndarray, meta: dict) -> tuple[np.ndarray, np.ndarray] | None:
    channels = meta["channels"]
    sprites = meta["sprites_by_channel"]
    source_idx = random.choice(sorted(meta["match_indices"]))
    targets = [idx for idx in range(len(channels)) if idx != source_idx]
    random.shuffle(targets)
    source_shape = channels[source_idx]["hole"]["mask"]
    for target_idx in targets:
        target_hole_key = channels[target_idx]["hole"]["key"]
        if _shape_key(source_shape) == target_hole_key:
            continue
        source_variant = random.choice(_shape_variants(source_shape))
        left, right = channels[target_idx]["interval"]
        if source_variant.shape[1] + 2 > (right - left + 1):
            continue
        variant_grid = np.array(grid, copy=True)
        variant_sprites = {}
        for idx, sprite in sprites.items():
            erase = np.full(sprite["mask"].shape, BG, dtype=int)
            paste(variant_grid, erase, sprite["top_left"][0], sprite["top_left"][1], mask=sprite["mask"])
            variant_sprites[idx] = dict(sprite)
        min_c = left + 1
        max_c = right - source_variant.shape[1]
        if min_c > max_c:
            continue
        new_left = random.randint(min_c, max_c)
        new_top = 0
        while new_top + source_variant.shape[0] < meta["frame_bbox"][0] - 2:
            target = variant_grid[new_top : new_top + source_variant.shape[0], new_left : new_left + source_variant.shape[1]]
            if np.all(target[source_variant] == BG):
                break
            new_top += 1
        if new_top + source_variant.shape[0] >= meta["frame_bbox"][0] - 1:
            continue
        variant_sprites[target_idx] = {
            "mask": source_variant,
            "top_left": (new_top, new_left),
            "color": sprites[target_idx]["color"],
            "key": _shape_key(source_variant),
        }
        for sprite in variant_sprites.values():
            _place_mask(variant_grid, sprite["mask"], sprite["top_left"][0], sprite["top_left"][1], sprite["color"])
        expected = _apply_rule_with_objects(variant_grid, channels, variant_sprites)
        return variant_grid, expected
    return None


def _validate_instance(grid: np.ndarray, expected: np.ndarray, meta: dict) -> None:
    assert grid.ndim == 2 and expected.ndim == 2
    assert max(grid.shape) <= 30 and max(expected.shape) <= 30
    assert np.array_equal(main(grid), expected)

    channels = meta["channels"]
    sprites = meta["sprites_by_channel"]
    assert len(channels) == len(sprites)
    assert meta["match_indices"]
    assert len(meta["match_indices"]) < len(channels)

    hole_areas = [int(channel["hole"]["area"]) for channel in channels]
    assert len(set(hole_areas)) < len(hole_areas), "need at least one equal-area different-shape hole pair"

    mismatch_idx = meta["mismatch_same_area_idx"]
    assert mismatch_idx not in meta["match_indices"]
    assert sprites[mismatch_idx]["mask"].sum() == channels[mismatch_idx]["hole"]["area"]
    hole_keys = {channel["hole"]["key"] for channel in channels}
    assert sprites[mismatch_idx]["key"] not in hole_keys

    variant = _derived_nonlocal_variant(grid, meta)
    assert variant is not None, "expected a derived counterexample variant"
    variant_grid, variant_expected = variant
    assert np.array_equal(main(variant_grid), variant_expected)
    assert not np.array_equal(variant_grid, grid)
    assert not np.array_equal(variant_expected, expected)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        try:
            grid, expected, meta = _build_instance()
            _validate_instance(grid, expected, meta)
            return grid
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid wall-channel hole-filling instance")
