from __future__ import annotations

import random

import numpy as np


MAX_GENERATION_ATTEMPTS = 400
_GEN_COUNTER = 0


def _most_common_color(grid: np.ndarray) -> int:
    vals = np.asarray(grid, dtype=int).ravel()
    return int(np.bincount(vals, minlength=10).argmax())


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    arr = np.asarray(mask, dtype=int)
    return tuple(arr.shape), tuple(int(v) for v in arr.ravel())


def _flip_for_axis(mask: np.ndarray, orient: str) -> np.ndarray:
    return np.fliplr(mask) if orient == "v" else np.flipud(mask)


def _detect_separator(grid: np.ndarray, bg: int) -> tuple[str, int, int]:
    h, w = grid.shape
    candidates: list[tuple[str, int, int]] = []
    for r in range(h):
        color = int(grid[r, 0])
        if color != bg and bool(np.all(grid[r, :] == color)):
            candidates.append(("h", r, color))
    for c in range(w):
        color = int(grid[0, c])
        if color != bg and bool(np.all(grid[:, c] == color)):
            candidates.append(("v", c, color))
    assert len(candidates) == 1, "expected exactly one full separator line"
    return candidates[0]


def _mirror_bbox(
    bbox: tuple[int, int, int, int], orient: str, axis: int
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    if orient == "v":
        return int(r0), int(2 * axis - c1), int(r1), int(2 * axis - c0)
    return int(2 * axis - r1), int(c0), int(2 * axis - r0), int(c1)


def _extract_sprites(
    grid: np.ndarray, bg: int, sep_color: int, orient: str, axis: int, source_side: str
) -> list[dict]:
    sprites: list[dict] = []
    for color in [int(c) for c in np.unique(grid) if int(c) not in {bg, sep_color}]:
        for coords in _components_mask(grid == color):
            if orient == "v":
                if source_side == "left":
                    assert bool(np.all(coords[:, 1] < axis))
                else:
                    assert bool(np.all(coords[:, 1] > axis))
            else:
                if source_side == "top":
                    assert bool(np.all(coords[:, 0] < axis))
                else:
                    assert bool(np.all(coords[:, 0] > axis))
            bbox = _bbox_coords(coords)
            r0, c0, r1, c1 = bbox
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            local = coords - np.array([r0, c0], dtype=int)
            mask[local[:, 0], local[:, 1]] = True
            sprites.append({"color": color, "bbox": bbox, "mask": mask})
    assert len(sprites) >= 2, "expected at least two sprites"
    return sprites


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    orient, axis, sep_color = _detect_separator(g, bg)
    if orient == "v":
        left = g[:, :axis]
        right = g[:, axis + 1 :]
        left_has = bool(np.any(left != bg))
        right_has = bool(np.any(right != bg))
        assert left_has ^ right_has, "sprites must appear on exactly one side of the separator"
        source_side = "left" if left_has else "right"
        blank_side = right if source_side == "left" else left
    else:
        top = g[:axis, :]
        bottom = g[axis + 1 :, :]
        top_has = bool(np.any(top != bg))
        bottom_has = bool(np.any(bottom != bg))
        assert top_has ^ bottom_has, "sprites must appear on exactly one side of the separator"
        source_side = "top" if top_has else "bottom"
        blank_side = bottom if source_side == "top" else top
    assert bool(np.all(blank_side == bg)), "the mirrored side must be pure background in the input"

    sprites = _extract_sprites(g, bg, sep_color, orient, axis, source_side)
    shapes = {tuple(sprite["mask"].shape) for sprite in sprites}
    assert len(shapes) == 1, "all sprite bboxes must have the same size"

    prototypes: dict[tuple[tuple[int, int], tuple[int, ...]], np.ndarray] = {}
    counts: dict[tuple[tuple[int, int], tuple[int, ...]], int] = {}
    for sprite in sprites:
        key = _shape_key(sprite["mask"])
        prototypes.setdefault(key, sprite["mask"])
        counts[key] = counts.get(key, 0) + 1
    assert len(prototypes) == 2, "expected exactly two prototype shapes"
    keys = list(prototypes)
    other = {keys[0]: prototypes[keys[1]], keys[1]: prototypes[keys[0]]}

    return {
        "grid": g,
        "bg": bg,
        "sep_color": sep_color,
        "orient": orient,
        "axis": axis,
        "source_side": source_side,
        "sprites": sprites,
        "prototype_keys": keys,
        "prototype_masks": prototypes,
        "prototype_counts": counts,
        "other_by_key": other,
    }


def _render_swapped_reflection(scene: dict) -> np.ndarray:
    g = scene["grid"]
    out = g.copy()
    bg = int(scene["bg"])
    orient = str(scene["orient"])
    axis = int(scene["axis"])
    for sprite in scene["sprites"]:
        src_key = _shape_key(sprite["mask"])
        target_mask = scene["other_by_key"][src_key]
        color = int(sprite["color"])
        mr0, mc0, mr1, mc1 = _mirror_bbox(sprite["bbox"], orient, axis)
        patch = np.full(target_mask.shape, bg, dtype=int)
        patch[target_mask] = color
        view = out[mr0 : mr1 + 1, mc0 : mc1 + 1]
        assert view.shape == patch.shape, "mirrored bbox must stay in-bounds"
        assert bool(np.all(view[target_mask] == bg)), "mirrored sprites must not overlap"
        view[target_mask] = color
    return out


def _render_ordinary_mirror(scene: dict) -> np.ndarray:
    g = scene["grid"]
    out = g.copy()
    bg = int(scene["bg"])
    orient = str(scene["orient"])
    axis = int(scene["axis"])
    for sprite in scene["sprites"]:
        target_mask = _flip_for_axis(sprite["mask"], orient)
        color = int(sprite["color"])
        mr0, mc0, mr1, mc1 = _mirror_bbox(sprite["bbox"], orient, axis)
        patch = np.full(target_mask.shape, bg, dtype=int)
        patch[target_mask] = color
        view = out[mr0 : mr1 + 1, mc0 : mc1 + 1]
        assert view.shape == patch.shape
        assert bool(np.all(view[target_mask] == bg))
        view[target_mask] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Mirror each sprite across the separator while swapping it to the other prototype shape.

    Input structure:
    - One non-background row or column is a full one-pixel separator line.
    - Exactly one side of that line contains several monochrome sprite components;
      the opposite side is pure background.
    - Every sprite is an instance of one of exactly two prototype shapes, and both
      prototypes share the same bounding-box size.

    Rule:
    - Keep the original sprite side and the separator line unchanged.
    - Reflect each sprite's bounding box across the separator onto the blank side.
    - At that mirrored location, redraw the sprite in its original color but using
      the other prototype's shape instead of its own.

    Key insight:
    - The separator mirrors positions, but the mirror also swaps prototype identity:
      a type-A sprite becomes type-B on the opposite side, and vice versa.
    """
    scene = _parse_scene(np.asarray(grid, dtype=int))
    return _render_swapped_reflection(scene)


def _touches_all_edges(mask: np.ndarray) -> bool:
    return (
        bool(mask[0, :].any())
        and bool(mask[-1, :].any())
        and bool(mask[:, 0].any())
        and bool(mask[:, -1].any())
    )


def _random_prototype(h: int, w: int) -> np.ndarray:
    min_area = max(4, (h + w) // 2)
    for _ in range(500):
        mask = np.random.rand(h, w) < random.uniform(0.38, 0.68)
        mask[0, random.randrange(w)] = True
        mask[h - 1, random.randrange(w)] = True
        mask[random.randrange(h), 0] = True
        mask[random.randrange(h), w - 1] = True
        area = int(mask.sum())
        if area < min_area or area >= h * w:
            continue
        if not _touches_all_edges(mask):
            continue
        if len(_components_mask(mask)) != 1:
            continue
        return mask
    raise RuntimeError("failed to sample prototype")


def _sample_prototypes(h: int, w: int, orient: str, decoy: bool) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(500):
        proto_a = _random_prototype(h, w)
        if decoy:
            proto_b = _flip_for_axis(proto_a, orient)
            if np.array_equal(proto_a, proto_b):
                continue
            return proto_a, proto_b
        proto_b = _random_prototype(h, w)
        if np.array_equal(proto_a, proto_b):
            continue
        if np.array_equal(proto_b, _flip_for_axis(proto_a, orient)):
            continue
        return proto_a, proto_b
    raise RuntimeError("failed to sample prototype pair")


def _rects_far_enough(
    rect_a: tuple[int, int, int, int], rect_b: tuple[int, int, int, int], gap: int = 1
) -> bool:
    ar0, ac0, ar1, ac1 = rect_a
    br0, bc0, br1, bc1 = rect_b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _random_boxes(area_h: int, area_w: int, box_h: int, box_w: int, count: int) -> list[tuple[int, int, int, int]] | None:
    candidates = [(r, c) for r in range(area_h - box_h + 1) for c in range(area_w - box_w + 1)]
    for _ in range(250):
        random.shuffle(candidates)
        placed: list[tuple[int, int, int, int]] = []
        for r, c in candidates:
            rect = (r, c, r + box_h - 1, c + box_w - 1)
            if all(_rects_far_enough(rect, prev, gap=1) for prev in placed):
                placed.append(rect)
                if len(placed) == count:
                    return placed
    return None


def _sample_source_dims(count: int, box_h: int, box_w: int, orient: str) -> tuple[int, int]:
    gap = 1
    options: list[tuple[int, int]] = []
    if orient == "v":
        for cols in range(1, count + 1):
            rows = (count + cols - 1) // cols
            min_h = rows * box_h + (rows + 1) * gap
            min_half_w = cols * box_w + (cols + 1) * gap
            if min_h <= 29 and min_half_w <= 14:
                max_h = min(29, min_h + 4)
                max_half_w = min(14, min_half_w + 3)
                for h in range(min_h, max_h + 1):
                    for half_w in range(min_half_w, max_half_w + 1):
                        options.append((h, half_w))
    else:
        for rows in range(1, count + 1):
            cols = (count + rows - 1) // rows
            min_half_h = rows * box_h + (rows + 1) * gap
            min_w = cols * box_w + (cols + 1) * gap
            if min_half_h <= 14 and min_w <= 29:
                max_half_h = min(14, min_half_h + 3)
                max_w = min(29, min_w + 4)
                for half_h in range(min_half_h, max_half_h + 1):
                    for w in range(min_w, max_w + 1):
                        options.append((half_h, w))
    assert options, "no feasible source dimensions"
    return random.choice(options)


def _paint_sprite(
    grid: np.ndarray, bbox: tuple[int, int, int, int], mask: np.ndarray, color: int
) -> None:
    r0, c0, r1, c1 = bbox
    view = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert view.shape == mask.shape
    view[mask] = int(color)


def _build_input(call_index: int) -> tuple[np.ndarray, bool]:
    source_side = ["left", "right", "top", "bottom"][call_index % 4]
    orient = "v" if source_side in {"left", "right"} else "h"
    decoy = call_index % 3 == 0
    count = random.randint(3, 6)
    box_h = random.randint(3, 5)
    box_w = random.randint(3, 5)
    proto_a, proto_b = _sample_prototypes(box_h, box_w, orient, decoy)

    if orient == "v":
        area_h, half_w = _sample_source_dims(count, box_h, box_w, orient)
        boxes = _random_boxes(area_h, half_w, box_h, box_w, count)
        assert boxes is not None
        axis = half_w
        grid = np.full((area_h, 2 * half_w + 1), -1, dtype=int)
    else:
        half_h, area_w = _sample_source_dims(count, box_h, box_w, orient)
        boxes = _random_boxes(half_h, area_w, box_h, box_w, count)
        assert boxes is not None
        axis = half_h
        grid = np.full((2 * half_h + 1, area_w), -1, dtype=int)

    colors = list(range(10))
    random.shuffle(colors)
    bg = colors.pop()
    sep_color = colors.pop()
    sprite_colors = colors[:count]
    grid.fill(bg)
    if orient == "v":
        grid[:, axis] = sep_color
    else:
        grid[axis, :] = sep_color

    zeros = random.randint(1, count - 1)
    type_ids = [0] * zeros + [1] * (count - zeros)
    random.shuffle(type_ids)

    for rect, color, type_id in zip(boxes, sprite_colors, type_ids):
        r0, c0, r1, c1 = rect
        if orient == "v":
            if source_side == "right":
                c0 += axis + 1
                c1 += axis + 1
        else:
            if source_side == "bottom":
                r0 += axis + 1
                r1 += axis + 1
        mask = proto_a if type_id == 0 else proto_b
        _paint_sprite(grid, (r0, c0, r1, c1), mask, color)

    return grid, decoy


def _validate_scene(grid: np.ndarray, expected_decoy: bool) -> bool:
    scene = _parse_scene(grid)
    out = main(grid)
    ordinary = _render_ordinary_mirror(scene)
    assert out.shape == grid.shape
    assert not np.array_equal(out, grid)
    assert len(scene["prototype_masks"]) == 2
    assert min(scene["prototype_counts"].values()) >= 1

    orient = scene["orient"]
    axis = scene["axis"]
    bg = scene["bg"]
    sep_color = scene["sep_color"]
    g = scene["grid"]
    if orient == "v":
        assert bool(np.all(out[:, axis] == sep_color))
        if scene["source_side"] == "left":
            assert np.array_equal(out[:, : axis + 1], g[:, : axis + 1])
        else:
            assert np.array_equal(out[:, axis:], g[:, axis:])
    else:
        assert bool(np.all(out[axis, :] == sep_color))
        if scene["source_side"] == "top":
            assert np.array_equal(out[: axis + 1, :], g[: axis + 1, :])
        else:
            assert np.array_equal(out[axis:, :], g[axis:, :])

    painted = np.zeros_like(grid, dtype=bool)
    for sprite in scene["sprites"]:
        src_key = _shape_key(sprite["mask"])
        target_mask = scene["other_by_key"][src_key]
        color = sprite["color"]
        mr0, mc0, mr1, mc1 = _mirror_bbox(sprite["bbox"], orient, axis)
        patch = _crop_bbox(out, (mr0, mc0, mr1, mc1))
        expected = np.full(target_mask.shape, bg, dtype=int)
        expected[target_mask] = color
        assert np.array_equal(patch, expected)
        ordinary_expected = np.full(target_mask.shape, bg, dtype=int)
        ordinary_expected[_flip_for_axis(sprite["mask"], orient)] = color
        if not expected_decoy:
            assert not np.array_equal(expected, ordinary_expected)
        painted[mr0 : mr1 + 1, mc0 : mc1 + 1] |= target_mask

    if orient == "v":
        if scene["source_side"] == "left":
            blank = out[:, axis + 1 :]
            blank_painted = painted[:, axis + 1 :]
        else:
            blank = out[:, :axis]
            blank_painted = painted[:, :axis]
    else:
        if scene["source_side"] == "top":
            blank = out[axis + 1 :, :]
            blank_painted = painted[axis + 1 :, :]
        else:
            blank = out[:axis, :]
            blank_painted = painted[:axis, :]
    assert bool(np.all(blank[~blank_painted] == bg))
    if expected_decoy:
        assert np.array_equal(out, ordinary)
    else:
        assert not np.array_equal(out, ordinary)
    return True


def generate_input() -> np.ndarray:
    global _GEN_COUNTER
    call_index = _GEN_COUNTER
    _GEN_COUNTER += 1
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            grid, decoy = _build_input(call_index)
            _validate_scene(grid, expected_decoy=decoy)
            return grid
        except AssertionError:
            continue
        except RuntimeError:
            continue
    raise RuntimeError("failed to generate a valid puzzle instance")
