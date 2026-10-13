from __future__ import annotations

import itertools
import random

import numpy as np


BG = 8
PALETTE = [0, 1, 2, 3, 4, 5, 6, 7, 9]
SIDES = ("top", "bottom", "left", "right")
GENERATE_MAX_RETRIES = 400

_GAP_CACHE: dict[tuple[int, int], dict[tuple[str, bool], list[tuple[tuple[int, int], ...]]]] = {}


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == int(bg) or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _border_mask(size: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _frame_info(grid: np.ndarray, comp: dict, bg: int) -> dict | None:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    size_h = int(r1 - r0 + 1)
    size_w = int(c1 - c0 + 1)
    if size_h != size_w or size_h < 5:
        return None

    sub = np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
    color = int(comp["color"])
    if np.any((sub != int(bg)) & (sub != color)):
        return None
    if np.any(sub[1:-1, 1:-1] != int(bg)):
        return None
    if any(int(v) == int(bg) for v in (sub[0, 0], sub[0, -1], sub[-1, 0], sub[-1, -1])):
        return None

    border = _border_mask(size_h)
    local_mask = np.zeros((size_h, size_h), dtype=bool)
    local_coords = np.asarray(comp["coords"], dtype=int) - np.array([[r0, c0]], dtype=int)
    local_mask[local_coords[:, 0], local_coords[:, 1]] = True
    if np.any(local_mask & ~border):
        return None
    if not np.array_equal(local_mask, border & (sub == color)):
        return None

    gaps: list[tuple[int, int, str]] = []
    for cc in range(1, size_h - 1):
        if int(sub[0, cc]) == int(bg):
            gaps.append((r0, c0 + cc, "top"))
        if int(sub[-1, cc]) == int(bg):
            gaps.append((r1, c0 + cc, "bottom"))
    for rr in range(1, size_h - 1):
        if int(sub[rr, 0]) == int(bg):
            gaps.append((r0 + rr, c0, "left"))
        if int(sub[rr, -1]) == int(bg):
            gaps.append((r0 + rr, c1, "right"))

    gap_count = len(gaps)
    if not (1 <= gap_count <= 3):
        return None

    side_counts = {side: 0 for side in SIDES}
    for _gr, _gc, side in gaps:
        side_counts[side] += 1
    best = max(side_counts.values())
    best_sides = [side for side in SIDES if side_counts[side] == best]
    if len(best_sides) == 1:
        target_side = best_sides[0]
    else:
        target_side = min(gaps, key=lambda item: (int(item[0]), int(item[1])))[2]

    return {
        "bbox": (r0, c0, r1, c1),
        "size": size_h,
        "color": color,
        "gap_count": gap_count,
        "target_side": target_side,
        "gaps": gaps,
    }


def _sprite_info(grid: np.ndarray, comp: dict, bg: int) -> dict | None:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    patch = np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
    color = int(comp["color"])
    if np.any((patch != int(bg)) & (patch != color)):
        return None
    mask = patch == color
    area = int(mask.sum())
    if area != int(comp["area"]):
        return None
    if not (1 <= area <= 3):
        return None
    if patch.shape[0] > 3 or patch.shape[1] > 3:
        return None
    return {
        "bbox": (r0, c0, r1, c1),
        "coords": np.asarray(comp["coords"], dtype=int),
        "mask": mask,
        "color": color,
        "area": area,
    }


def _rotation_key(mask: np.ndarray, side: str) -> tuple[float, float, float]:
    pts = np.argwhere(mask)
    center_r = 0.5 * float(mask.shape[0] - 1)
    center_c = 0.5 * float(mask.shape[1] - 1)
    mean_r = float(pts[:, 0].mean())
    mean_c = float(pts[:, 1].mean())

    if side == "top":
        bias = center_r - mean_r
        axis_span = float(pts[:, 0].max() - pts[:, 0].min()) if pts.shape[0] > 1 else 0.0
        ortho = abs(mean_c - center_c)
    elif side == "bottom":
        bias = mean_r - center_r
        axis_span = float(pts[:, 0].max() - pts[:, 0].min()) if pts.shape[0] > 1 else 0.0
        ortho = abs(mean_c - center_c)
    elif side == "left":
        bias = center_c - mean_c
        axis_span = float(pts[:, 1].max() - pts[:, 1].min()) if pts.shape[0] > 1 else 0.0
        ortho = abs(mean_r - center_r)
    else:
        bias = mean_c - center_c
        axis_span = float(pts[:, 1].max() - pts[:, 1].min()) if pts.shape[0] > 1 else 0.0
        ortho = abs(mean_r - center_r)

    return (bias, axis_span, -ortho)


def _rotate_for_side(mask: np.ndarray, side: str) -> np.ndarray:
    best_mask = np.asarray(mask, dtype=bool)
    best_key = _rotation_key(best_mask, side)
    for k in range(1, 4):
        rotated = np.rot90(mask, k)
        key = _rotation_key(rotated, side)
        if key > best_key:
            best_key = key
            best_mask = rotated
    return np.asarray(best_mask, dtype=bool)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match sprites to frames by gap-count, rotate toward the frame's opening side,
    and center the rotated sprite inside the frame.
    """
    g = np.asarray(grid, dtype=int)
    comps = _find_components(g, BG)

    frames: list[dict] = []
    frame_component_ids: set[int] = set()
    for idx, comp in enumerate(comps):
        info = _frame_info(g, comp, BG)
        if info is not None:
            frames.append(info)
            frame_component_ids.add(int(idx))

    sprites: list[dict] = []
    for idx, comp in enumerate(comps):
        if idx in frame_component_ids:
            continue
        info = _sprite_info(g, comp, BG)
        if info is not None:
            sprites.append(info)

    assert frames, "expected at least one frame"
    assert sprites, "expected at least one sprite"

    out = np.array(g, copy=True)
    for sprite in sprites:
        coords = np.asarray(sprite["coords"], dtype=int)
        out[coords[:, 0], coords[:, 1]] = int(BG)

    frames_by_count: dict[int, list[dict]] = {}
    for frame in frames:
        frames_by_count.setdefault(int(frame["gap_count"]), []).append(frame)

    sprites_by_count: dict[int, list[dict]] = {}
    for sprite in sprites:
        sprites_by_count.setdefault(int(sprite["area"]), []).append(sprite)

    for count, frame_group in frames_by_count.items():
        sprite_group = sprites_by_count.get(int(count), [])
        assert len(sprite_group) == len(frame_group), "gap-count to pixel-count matching must be unambiguous"
        frame_group.sort(key=lambda item: (int(item["bbox"][0]), int(item["bbox"][1])))
        sprite_group.sort(key=lambda item: (int(item["bbox"][0]), int(item["bbox"][1])))
        for frame, sprite in zip(frame_group, sprite_group):
            rotated = _rotate_for_side(np.asarray(sprite["mask"], dtype=bool), str(frame["target_side"]))
            color = int(sprite["color"])
            inner_size = int(frame["size"] - 2)
            h, w = rotated.shape
            rr = int(frame["bbox"][0] + 1 + (inner_size - h) // 2)
            cc = int(frame["bbox"][1] + 1 + (inner_size - w) // 2)
            target = out[rr : rr + h, cc : cc + w]
            target[rotated] = color

    return out


def _gap_side_from_rel(size: int, gap: tuple[int, int]) -> str:
    r, c = gap
    if r == 0 and 0 < c < size - 1:
        return "top"
    if r == size - 1 and 0 < c < size - 1:
        return "bottom"
    if c == 0 and 0 < r < size - 1:
        return "left"
    if c == size - 1 and 0 < r < size - 1:
        return "right"
    raise ValueError("gap must lie on a non-corner border cell")


def _infer_side_from_rel_gaps(size: int, gaps: tuple[tuple[int, int], ...]) -> tuple[str, bool]:
    side_counts = {side: 0 for side in SIDES}
    annotated: list[tuple[int, int, str]] = []
    for gap in gaps:
        side = _gap_side_from_rel(size, gap)
        side_counts[side] += 1
        annotated.append((int(gap[0]), int(gap[1]), side))
    best = max(side_counts.values())
    best_sides = [side for side in SIDES if side_counts[side] == best]
    is_tie = len(best_sides) > 1
    if not is_tie:
        return best_sides[0], False
    return min(annotated, key=lambda item: (int(item[0]), int(item[1])))[2], True


def _gap_patterns(size: int, count: int) -> dict[tuple[str, bool], list[tuple[tuple[int, int], ...]]]:
    key = (size, count)
    cached = _GAP_CACHE.get(key)
    if cached is not None:
        return cached

    slots: list[tuple[int, int]] = []
    for cc in range(1, size - 1):
        slots.append((0, cc))
        slots.append((size - 1, cc))
    for rr in range(1, size - 1):
        slots.append((rr, 0))
        slots.append((rr, size - 1))

    patterns: dict[tuple[str, bool], list[tuple[tuple[int, int], ...]]] = {}
    for gaps in itertools.combinations(slots, count):
        frame_mask = _border_mask(size)
        for r, c in gaps:
            frame_mask[int(r), int(c)] = False
        if not _single_component(frame_mask):
            continue
        side, is_tie = _infer_side_from_rel_gaps(size, gaps)
        patterns.setdefault((side, is_tie), []).append(gaps)

    _GAP_CACHE[key] = patterns
    return patterns


def _choose_gaps(size: int, count: int) -> tuple[list[tuple[int, int]], str]:
    patterns = _gap_patterns(size, count)
    keys = list(patterns.keys())
    assert keys, "gap pattern cache must not be empty"
    if count >= 2:
        unique_keys = [key for key in keys if not key[1]]
        tie_keys = [key for key in keys if key[1]]
        if unique_keys and tie_keys:
            key_pool = tie_keys if random.choice([False, True]) else unique_keys
        else:
            key_pool = keys
    else:
        key_pool = keys
    target_key = random.choice(key_pool)
    gaps = list(random.choice(patterns[target_key]))
    return gaps, str(target_key[0])


def _single_component(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _build_frame_patch(size: int, color: int, gaps: list[tuple[int, int]]) -> np.ndarray:
    patch = np.full((size, size), int(BG), dtype=int)
    patch[0, :] = int(color)
    patch[-1, :] = int(color)
    patch[:, 0] = int(color)
    patch[:, -1] = int(color)
    for r, c in gaps:
        patch[int(r), int(c)] = int(BG)
    assert _single_component(patch == int(color))
    return patch


def _random_sprite_mask(count: int) -> np.ndarray:
    if count == 1:
        return np.array([[True]], dtype=bool)
    if count == 2:
        base = np.array([[True, True]], dtype=bool)
        return np.rot90(base, random.randrange(4))
    assert count == 3
    base = np.array([[True, False], [True, True]], dtype=bool)
    return np.rot90(base, random.randrange(4))


def _build_sprite_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, int(BG), dtype=int)
    patch[np.asarray(mask, dtype=bool)] = int(color)
    return patch


def _can_place(occupied: np.ndarray, r: int, c: int, h: int, w: int, pad: int) -> bool:
    hh, ww = occupied.shape
    r0 = max(0, int(r - pad))
    c0 = max(0, int(c - pad))
    r1 = min(hh, int(r + h + pad))
    c1 = min(ww, int(c + w + pad))
    return not bool(occupied[r0:r1, c0:c1].any())


def _place_patch(grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray, r: int, c: int, pad: int) -> None:
    h, w = patch.shape
    grid[r : r + h, c : c + w] = patch
    hh, ww = occupied.shape
    r0 = max(0, int(r - pad))
    c0 = max(0, int(c - pad))
    r1 = min(hh, int(r + h + pad))
    c1 = min(ww, int(c + w + pad))
    occupied[r0:r1, c0:c1] = True


def _random_location(occupied: np.ndarray, h: int, w: int, pad: int) -> tuple[int, int] | None:
    hh, ww = occupied.shape
    positions = [(r, c) for r in range(hh - h + 1) for c in range(ww - w + 1)]
    random.shuffle(positions)
    for r, c in positions:
        if _can_place(occupied, int(r), int(c), int(h), int(w), int(pad)):
            return int(r), int(c)
    return None


def _gen_attempt() -> np.ndarray | None:
    frame_count = random.choice([2, 3])
    counts = random.sample([1, 2, 3], frame_count)
    frame_size = random.randint(5, 8)

    min_side = max(16, frame_size * 2 + 4)
    grid_h = random.randint(min_side, min_side + 8)
    grid_w = random.randint(min_side, min_side + 8)
    grid = np.full((grid_h, grid_w), int(BG), dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    colors = random.sample(PALETTE, frame_count * 2)
    frame_colors = colors[:frame_count]
    sprite_colors = colors[frame_count:]

    frame_specs = []
    for count, color in zip(counts, frame_colors):
        gaps, target_side = _choose_gaps(frame_size, count)
        frame_specs.append(
            {
                "count": int(count),
                "target_side": str(target_side),
                "patch": _build_frame_patch(frame_size, int(color), gaps),
            }
        )

    sprite_specs = []
    for count, color in zip(counts, sprite_colors):
        mask = _random_sprite_mask(int(count))
        sprite_specs.append(
            {
                "count": int(count),
                "patch": _build_sprite_patch(mask, int(color)),
            }
        )

    for spec in frame_specs:
        patch = np.asarray(spec["patch"], dtype=int)
        loc = _random_location(occupied, int(patch.shape[0]), int(patch.shape[1]), pad=1)
        if loc is None:
            return None
        _place_patch(grid, occupied, patch, int(loc[0]), int(loc[1]), pad=1)

    random.shuffle(sprite_specs)
    for spec in sprite_specs:
        patch = np.asarray(spec["patch"], dtype=int)
        loc = _random_location(occupied, int(patch.shape[0]), int(patch.shape[1]), pad=1)
        if loc is None:
            return None
        _place_patch(grid, occupied, patch, int(loc[0]), int(loc[1]), pad=1)

    try:
        _ = main(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GENERATE_MAX_RETRIES} attempts")
