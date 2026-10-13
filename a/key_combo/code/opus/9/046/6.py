from __future__ import annotations

import math
import random

import numpy as np


BG = 0
FOUR_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_GENERATION_ATTEMPTS = 500


def _neighbors4(r: int, c: int):
    for dr, dc in FOUR_DIRS:
        yield r + dr, c + dc


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.array(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int, copy=True)


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int | None = None) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    if transparent is None:
        view[:] = patch
        return
    mask = patch != int(transparent)
    view[mask] = patch[mask]


def _scale_sprite(sprite: np.ndarray, factor: int) -> np.ndarray:
    k = int(factor)
    if k <= 0:
        raise ValueError("scale factor must be positive")
    return np.kron(sprite, np.ones((k, k), dtype=int)).astype(int)


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _find_monochrome_components(grid: np.ndarray, bg: int = BG) -> list[dict]:
    comps: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != int(bg)):
        for coords in _component_coords(grid == int(color)):
            comps.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": _bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def _is_frame_component(grid: np.ndarray, component: dict, bg: int = BG) -> bool:
    r0, c0, r1, c1 = component["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h != w or h < 3:
        return False
    if int(component["area"]) != 4 * h - 4:
        return False
    coords = {tuple(map(int, rc)) for rc in component["coords"]}
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_border = r in (r0, r1) or c in (c0, c1)
            if on_border:
                if (r, c) not in coords:
                    return False
            elif int(grid[r, c]) != int(bg):
                return False
    return True


def _extract_objects(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = BG
    frames: list[dict] = []
    sprites: list[dict] = []
    for component in _find_monochrome_components(g, bg=bg):
        r0, c0, r1, c1 = component["bbox"]
        patch = _crop_bbox(g, component["bbox"])
        if _is_frame_component(g, component, bg=bg):
            side = int(r1 - r0 + 1)
            frames.append(
                {
                    **component,
                    "outer_side": side,
                    "inner_side": side - 2,
                    "inner_bbox": (r0 + 1, c0 + 1, r1 - 1, c1 - 1),
                    "patch": patch,
                }
            )
        else:
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h != w:
                raise AssertionError("sprite bounding boxes must be square")
            sprites.append(
                {
                    **component,
                    "bbox_side": h,
                    "pixel_area": int(component["area"]),
                    "patch": patch,
                }
            )
    return frames, sprites


def _render_from_sorted_lists(grid: np.ndarray, frames_sorted: list[dict], sprites_sorted: list[dict]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for sprite in sprites_sorted:
        out[sprite["coords"][:, 0], sprite["coords"][:, 1]] = BG
    if len(frames_sorted) != len(sprites_sorted):
        raise AssertionError("frame/sprite counts must match")
    for frame, sprite in zip(frames_sorted, sprites_sorted):
        inner = int(frame["inner_side"])
        side = int(sprite["bbox_side"])
        if inner % side != 0:
            raise AssertionError("frame interior must be an integer multiple of sprite bbox")
        factor = inner // side
        scaled = _scale_sprite(sprite["patch"], factor)
        if scaled.shape != (inner, inner):
            raise AssertionError("scaled sprite must exactly match frame interior")
        r0, c0, _r1, _c1 = frame["inner_bbox"]
        _paste(out, scaled, r0, c0, transparent=BG)
    return out


def _render_pairing(grid: np.ndarray, frames: list[dict], sprites: list[dict]) -> np.ndarray:
    frames_sorted = sorted(frames, key=lambda obj: int(obj["inner_side"]))
    sprites_sorted = sorted(sprites, key=lambda obj: int(obj["bbox_side"]))
    return _render_from_sorted_lists(grid, frames_sorted, sprites_sorted)


def main(grid: np.ndarray) -> np.ndarray:
    """Pair sprites and hollow square frames by ascending area rank, then scale each sprite to fill its matched frame.

    Input structure:
    - The grid is a black background containing 2 or 3 single-color hollow square frames with one-cell-thick borders.
    - Each frame encloses an empty black square interior, and each frame uses a color different from every sprite color.
    - Elsewhere on the grid are the same number of monochrome 4-connected sprites, each with a square bounding box.

    Rule:
    - Detect all hollow square frames and sort them by interior area from smallest to largest.
    - Detect all non-frame sprite components and sort them by their square bounding-box area from smallest to largest.
    - Pair the two sorted lists rank by rank, scale each sprite's bounding-box crop by the integer factor implied by its matched frame interior, erase the original sprite, and paste the scaled result into that frame.

    Key insight:
    - Colors and proximity are distractors; the only pairing signal is the parallel area ordering of the sprite set and the frame set.
    """
    g = np.asarray(grid, dtype=int)
    frames, sprites = _extract_objects(g)
    return _render_pairing(g, frames, sprites)


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    return coords.size > 0 and len(_component_coords(mask)) == 1


def _touches_all_bbox_sides(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _is_frame_like_mask(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if h != w or h < 3:
        return False
    border = np.zeros_like(mask, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(np.array_equal(mask, border))


def _generate_sprite_mask(size: int, target_area: int, rng: random.Random) -> np.ndarray | None:
    if target_area < size or target_area > size * size:
        return None
    for _ in range(200):
        mask = np.ones((size, size), dtype=bool)
        cells = [(r, c) for r in range(size) for c in range(size)]
        rng.shuffle(cells)
        to_remove = size * size - target_area
        for r, c in cells:
            if to_remove == 0:
                break
            mask[r, c] = False
            if not _touches_all_bbox_sides(mask) or not _is_connected(mask):
                mask[r, c] = True
                continue
            to_remove -= 1
        if (
            int(mask.sum()) == int(target_area)
            and _touches_all_bbox_sides(mask)
            and _is_connected(mask)
            and not _is_frame_like_mask(mask)
        ):
            return mask
    return None


def _sample_sizes(k: int, rng: random.Random) -> tuple[int, ...]:
    if k == 2:
        return rng.choice(((3, 4), (4, 5)))
    return rng.choice(((3, 4, 5), (2, 4, 5)))


def _sample_frame_inner_sizes(sprite_sizes: tuple[int, ...], rng: random.Random) -> tuple[int, ...]:
    candidates: list[tuple[int, ...]] = []
    for scales in np.ndindex(*(2,) * len(sprite_sizes)):
        actual_scales = tuple(2 + s for s in scales)
        inner_sizes = tuple(size * scale for size, scale in zip(sprite_sizes, actual_scales))
        if list(inner_sizes) == sorted(inner_sizes) and len(set(inner_sizes)) == len(inner_sizes) and max(inner_sizes) <= 12:
            candidates.append(inner_sizes)
    if not candidates:
        raise AssertionError("no valid frame sizes")
    return rng.choice(candidates)


def _sample_target_areas(sprite_sizes: tuple[int, ...], rng: random.Random) -> tuple[int, ...]:
    ranges = [(max(size + 1, math.ceil(size * size * 0.45)), size * size) for size in sprite_sizes]
    inversion_spots = [i for i in range(len(ranges) - 1) if ranges[i][1] > ranges[i + 1][0]]
    if not inversion_spots:
        raise RuntimeError("sprite sizes do not allow pixel-area inversion")

    for _ in range(200):
        areas: list[int | None] = [None] * len(sprite_sizes)
        inv = rng.choice(inversion_spots)

        left_lo, left_hi = ranges[inv]
        right_lo, right_hi = ranges[inv + 1]
        left_min = max(left_lo, right_lo + 1)
        if left_min > left_hi:
            continue
        left_area = rng.randint(left_min, left_hi)
        right_area = rng.randint(right_lo, min(right_hi, left_area - 1))
        areas[inv] = left_area
        areas[inv + 1] = right_area

        used = {left_area, right_area}
        ok = True
        for idx, (lo, hi) in enumerate(ranges):
            if areas[idx] is not None:
                continue
            choices = [value for value in range(lo, hi + 1) if value not in used]
            if not choices:
                ok = False
                break
            value = rng.choice(choices)
            areas[idx] = value
            used.add(value)
        if not ok:
            continue

        out = tuple(int(value) for value in areas)
        order_by_pixels = tuple(sorted(range(len(out)), key=lambda idx: out[idx]))
        if order_by_pixels != tuple(range(len(out))):
            return out
    raise RuntimeError("failed to construct non-monotone sprite areas")


def _build_frame_patch(inner_side: int, color: int) -> np.ndarray:
    outer = inner_side + 2
    patch = np.full((outer, outer), int(color), dtype=int)
    patch[1:-1, 1:-1] = BG
    return patch


def _build_sprite_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = int(color)
    return patch


def _can_place(used: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    H, W = used.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(H, r0 + h + pad)
    cc1 = min(W, c0 + w + pad)
    return not bool(used[rr0:rr1, cc0:cc1].any())


def _reserve(used: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> None:
    H, W = used.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(H, r0 + h + pad)
    cc1 = min(W, c0 + w + pad)
    used[rr0:rr1, cc0:cc1] = True


def _rect_center(r0: int, c0: int, side: int) -> tuple[float, float]:
    half = (side - 1) / 2.0
    return r0 + half, c0 + half


def _manhattan(a: tuple[float, float], b: tuple[float, float]) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _place_frames(H: int, W: int, outer_sides: list[int], rng: random.Random) -> list[tuple[int, int]] | None:
    used = np.zeros((H, W), dtype=bool)
    positions: list[tuple[int, int] | None] = [None] * len(outer_sides)
    for idx in sorted(range(len(outer_sides)), key=lambda i: outer_sides[i], reverse=True):
        side = outer_sides[idx]
        candidates = [(r, c) for r in range(H - side + 1) for c in range(W - side + 1) if _can_place(used, r, c, side, side)]
        rng.shuffle(candidates)
        if not candidates:
            return None
        r0, c0 = candidates[0]
        positions[idx] = (r0, c0)
        _reserve(used, r0, c0, side, side)
    return [pos for pos in positions if pos is not None]


def _place_sprites(
    H: int,
    W: int,
    sprite_sides: list[int],
    frame_positions: list[tuple[int, int]],
    frame_outer_sides: list[int],
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    used = np.zeros((H, W), dtype=bool)
    for (r0, c0), side in zip(frame_positions, frame_outer_sides):
        _reserve(used, r0, c0, side, side)

    frame_centers = [_rect_center(r0, c0, side) for (r0, c0), side in zip(frame_positions, frame_outer_sides)]
    positions: list[tuple[int, int] | None] = [None] * len(sprite_sides)

    for idx in sorted(range(len(sprite_sides)), key=lambda i: sprite_sides[i], reverse=True):
        side = sprite_sides[idx]
        candidates: list[tuple[int, int]] = []
        for r0 in range(H - side + 1):
            for c0 in range(W - side + 1):
                if not _can_place(used, r0, c0, side, side):
                    continue
                sprite_center = _rect_center(r0, c0, side)
                paired_dist = _manhattan(sprite_center, frame_centers[idx])
                wrong_dist = min(_manhattan(sprite_center, frame_centers[j]) for j in range(len(frame_centers)) if j != idx)
                if wrong_dist < paired_dist:
                    candidates.append((r0, c0))
        rng.shuffle(candidates)
        if not candidates:
            return None
        r0, c0 = candidates[0]
        positions[idx] = (r0, c0)
        _reserve(used, r0, c0, side, side)
    return [pos for pos in positions if pos is not None]


def _row_major_order(positions: list[tuple[int, int]]) -> tuple[int, ...]:
    return tuple(sorted(range(len(positions)), key=lambda i: (positions[i][0], positions[i][1])))


def _make_case(rng: random.Random) -> np.ndarray | None:
    k = rng.choice((2, 3))
    sprite_sizes = _sample_sizes(k, rng)
    frame_inner_sizes = _sample_frame_inner_sizes(sprite_sizes, rng)
    target_areas = _sample_target_areas(sprite_sizes, rng)

    all_colors = rng.sample(range(1, 10), 2 * k)
    frame_colors = all_colors[:k]
    sprite_colors = all_colors[k:]

    sprite_masks = []
    for side, area in zip(sprite_sizes, target_areas):
        mask = _generate_sprite_mask(side, area, rng)
        if mask is None:
            return None
        sprite_masks.append(mask)

    frame_outer_sides = [inner + 2 for inner in frame_inner_sizes]
    min_side = max(max(frame_outer_sides) + 4, 18)
    H = rng.randint(min_side, 28)
    W = rng.randint(min_side, 28)

    frame_positions = _place_frames(H, W, frame_outer_sides, rng)
    if frame_positions is None:
        return None
    if _row_major_order(frame_positions) == tuple(range(k)):
        return None

    sprite_positions = _place_sprites(H, W, list(sprite_sizes), frame_positions, frame_outer_sides, rng)
    if sprite_positions is None:
        return None
    if _row_major_order(sprite_positions) == tuple(range(k)):
        return None

    grid = np.zeros((H, W), dtype=int)
    frame_records: list[dict] = []
    sprite_records: list[dict] = []

    for idx in range(k):
        frame_patch = _build_frame_patch(frame_inner_sizes[idx], frame_colors[idx])
        fr, fc = frame_positions[idx]
        _paste(grid, frame_patch, fr, fc)
        frame_records.append(
            {
                "color": int(frame_colors[idx]),
                "inner_side": int(frame_inner_sizes[idx]),
                "outer_side": int(frame_outer_sides[idx]),
                "inner_bbox": (fr + 1, fc + 1, fr + frame_inner_sizes[idx], fc + frame_inner_sizes[idx]),
                "bbox": (fr, fc, fr + frame_outer_sides[idx] - 1, fc + frame_outer_sides[idx] - 1),
                "patch": frame_patch,
            }
        )

    for idx in range(k):
        sprite_patch = _build_sprite_patch(sprite_masks[idx], sprite_colors[idx])
        sr, sc = sprite_positions[idx]
        _paste(grid, sprite_patch, sr, sc, transparent=BG)
        coords = np.argwhere(sprite_patch != BG) + np.array([sr, sc], dtype=int)
        sprite_records.append(
            {
                "color": int(sprite_colors[idx]),
                "bbox_side": int(sprite_sizes[idx]),
                "pixel_area": int(target_areas[idx]),
                "bbox": (sr, sc, sr + sprite_sizes[idx] - 1, sc + sprite_sizes[idx] - 1),
                "coords": coords,
                "patch": sprite_patch,
            }
        )

    expected = _render_pairing(grid, frame_records, sprite_records)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    parsed_frames, parsed_sprites = _extract_objects(grid)
    if len(parsed_frames) != k or len(parsed_sprites) != k:
        return None
    if len({frame["inner_side"] for frame in parsed_frames}) != k:
        return None
    if len({sprite["bbox_side"] for sprite in parsed_sprites}) != k:
        return None

    pixel_order = tuple(sorted(range(k), key=lambda idx: sprite_records[idx]["pixel_area"]))
    if pixel_order == tuple(range(k)):
        return None

    try:
        alt = _render_from_sorted_lists(
            grid,
            sorted(frame_records, key=lambda obj: int(obj["inner_side"])),
            [sprite_records[idx] for idx in pixel_order],
        )
    except AssertionError:
        alt = None
    if alt is not None and np.array_equal(alt, expected):
        return None

    frame_centers = [_rect_center(r0, c0, side) for (r0, c0), side in zip(frame_positions, frame_outer_sides)]
    proximity_mismatch = False
    for idx, (sr, sc) in enumerate(sprite_positions):
        sprite_center = _rect_center(sr, sc, sprite_sizes[idx])
        paired = _manhattan(sprite_center, frame_centers[idx])
        wrong = min(_manhattan(sprite_center, frame_centers[j]) for j in range(k) if j != idx)
        if wrong < paired:
            proximity_mismatch = True
            break
    if not proximity_mismatch:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _make_case(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC sample")
