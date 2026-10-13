from __future__ import annotations

import random
from itertools import combinations

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
INPUT_SIZE_RANGE = (16, 22)
GEN_RETRIES = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _trim_bool(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_connected4(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(coords)


def _extract_components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or g[r, c] == BG:
                continue
            color = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(g[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0, c0, r1, c1 = _bbox_from_coords(arr)
            sprite = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
            sprite[arr[:, 0] - r0, arr[:, 1] - c0] = color
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (r0, c0, r1, c1),
                    "sprite": sprite,
                    "area": int(len(coords)),
                    "bbox_area": int(sprite.shape[0] * sprite.shape[1]),
                    "max_side": int(max(sprite.shape)),
                }
            )
    return comps


def _assemble_output(components: list[dict]) -> np.ndarray:
    ordered = sorted(components, key=lambda comp: comp["area"])
    max_side = max(comp["max_side"] for comp in ordered)
    out = np.zeros((2 * max_side, 2 * max_side), dtype=int)
    placements = [
        (max_side, max_side, 0),
        (0, max_side, -1),
        (0, 0, 2),
        (max_side, 0, 1),
    ]
    for comp, (r0, c0, rot_k) in zip(ordered, placements):
        rotated = np.rot90(comp["sprite"], rot_k)
        h, w = rotated.shape
        out[r0 : r0 + h, c0 : c0 + w] = rotated
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort the four scattered sprites by pixel area and tile their cropped forms into rotated quadrants.

    Input structure:
    - The grid is a black background with exactly four separated non-black sprites.
    - Each sprite is monochromatic, 4-connected, irregular, and has a tight bounding box no larger than 4x4.
    - The four sprite areas are all different, while input positions are arbitrary and irrelevant.

    Rule:
    - Detect the four sprites and compute their pixel areas, ordering them from smallest to largest.
    - Crop each sprite to its tight bounding box and let M be the largest side length among those cropped boxes.
    - Create a 2M x 2M black output split into four MxM quadrants.
    - Place the sorted sprites into quadrants after rotation, aligned to each quadrant's top-left corner:
      smallest -> lower-right unchanged, second -> upper-right rotated 90 degrees clockwise,
      third -> upper-left rotated 180 degrees, largest -> lower-left rotated 90 degrees counterclockwise.

    Key insight:
    - Area rank is the only signal that determines both destination quadrant and rotation; the input layout itself carries no positional meaning.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    components = _extract_components(g)
    assert len(components) == 4
    areas = [comp["area"] for comp in components]
    assert len(set(areas)) == 4
    return _assemble_output(components)


def _enumerate_candidate_masks() -> list[dict]:
    seen = set()
    candidates = []
    for bits in range(1, 1 << 16):
        area = bits.bit_count()
        if area < 3 or area > 10:
            continue
        mask = np.zeros((4, 4), dtype=bool)
        for idx in range(16):
            if (bits >> idx) & 1:
                mask[idx // 4, idx % 4] = True
        if not _is_connected4(mask):
            continue
        cropped = _trim_bool(mask)
        h, w = cropped.shape
        if min(h, w) < 2 or area == h * w:
            continue
        if any(np.array_equal(cropped, _trim_bool(np.rot90(cropped, k))) for k in (1, 2, 3)):
            continue
        key = tuple(map(tuple, cropped.astype(int)))
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            {
                "mask": cropped,
                "area": area,
                "bbox_area": int(h * w),
                "max_side": int(max(h, w)),
            }
        )
    return candidates


CANDIDATES = _enumerate_candidate_masks()
SEEN_OUTPUT_SIGNATURES: set[tuple[tuple[int, int], bytes]] = set()


def _has_strict_bbox_inversion(objs: list[dict]) -> bool:
    for left, right in combinations(objs, 2):
        if left["area"] < right["area"] and left["bbox_area"] > right["bbox_area"]:
            return True
        if right["area"] < left["area"] and right["bbox_area"] > left["bbox_area"]:
            return True
    return False


def _max_side_is_not_enough(objs: list[dict]) -> bool:
    for left, right in combinations(objs, 2):
        if left["area"] < right["area"] and left["max_side"] >= right["max_side"]:
            return True
        if right["area"] < left["area"] and right["max_side"] >= left["max_side"]:
            return True
    return False


def _order_by(items: list[dict], key) -> list[int]:
    return [idx for idx, _ in sorted(enumerate(items), key=lambda pair: key(pair[1]))]


def _shape_set_valid(objs: list[dict]) -> bool:
    if len({obj["area"] for obj in objs}) != 4:
        return False
    if not any(obj["max_side"] == 3 for obj in objs) and not any(obj["max_side"] == 4 for obj in objs):
        return False
    return _has_strict_bbox_inversion(objs) and _max_side_is_not_enough(objs)


def _choose_shape_set(target_max_side: int) -> list[dict] | None:
    pool = [cand for cand in CANDIDATES if cand["max_side"] <= target_max_side]
    exact = [cand for cand in pool if cand["max_side"] == target_max_side]
    if not exact:
        return None
    for _ in range(GEN_RETRIES):
        chosen = [random.choice(exact)]
        used_areas = {chosen[0]["area"]}
        while len(chosen) < 4:
            cand = random.choice(pool)
            if cand["area"] in used_areas:
                continue
            chosen.append(cand)
            used_areas.add(cand["area"])
        if _shape_set_valid(chosen):
            random.shuffle(chosen)
            return [dict(obj) for obj in chosen]
    return None


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _place_sprites(sprites: list[np.ndarray], height: int, width: int) -> np.ndarray | None:
    grid = np.zeros((height, width), dtype=int)
    occupied = np.zeros((height, width), dtype=bool)
    order = list(range(len(sprites)))
    random.shuffle(order)
    for idx in order:
        sprite = sprites[idx]
        mask = sprite != BG
        sh, sw = sprite.shape
        blocked = _dilate8(occupied)
        positions = [
            (r0, c0)
            for r0 in range(1, height - sh)
            for c0 in range(1, width - sw)
            if not np.any(blocked[r0 : r0 + sh, c0 : c0 + sw] & mask)
        ]
        if not positions:
            return None
        r0, c0 = random.choice(positions)
        grid[r0 : r0 + sh, c0 : c0 + sw][mask] = sprite[mask]
        occupied[r0 : r0 + sh, c0 : c0 + sw] |= mask
    return grid


def _rank_signature(items: list[dict], key) -> tuple[int, ...]:
    return tuple(idx for idx, _ in sorted(enumerate(items), key=lambda pair: key(pair[1])))


def _validate_scene(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False
    components = _extract_components(grid)
    if len(components) != 4:
        return False
    areas = [comp["area"] for comp in components]
    colors = [comp["color"] for comp in components]
    if len(set(areas)) != 4 or len(set(colors)) != 4:
        return False
    if not _has_strict_bbox_inversion(components):
        return False
    if not _max_side_is_not_enough(components):
        return False
    for comp in components:
        sprite = comp["sprite"]
        h, w = sprite.shape
        if max(h, w) > 4 or min(h, w) < 2:
            return False
        if comp["area"] == h * w:
            return False
        if any(np.array_equal(sprite, _trim_bool(np.rot90(sprite != BG, k)).astype(int) * comp["color"]) for k in (1, 2, 3)):
            return False

    area_order = _rank_signature(components, lambda comp: comp["area"])
    row_major_order = _rank_signature(components, lambda comp: (comp["bbox"][0], comp["bbox"][1]))
    col_major_order = _rank_signature(components, lambda comp: (comp["bbox"][1], comp["bbox"][0]))
    color_order = _rank_signature(components, lambda comp: comp["color"])
    if area_order in {row_major_order, col_major_order, color_order, color_order[::-1]}:
        return False

    actual = main(grid)
    expected = _assemble_output(components)
    if not np.array_equal(actual, expected):
        return False
    max_side = max(comp["max_side"] for comp in components)
    if actual.shape != (2 * max_side, 2 * max_side):
        return False
    return True


def _make_colored_sprites(shape_set: list[dict]) -> list[np.ndarray]:
    colors = random.sample(PALETTE, 4)
    sprites = []
    for shape, color in zip(shape_set, colors):
        sprite = np.zeros_like(shape["mask"], dtype=int)
        sprite[shape["mask"]] = color
        enriched = dict(shape)
        enriched["color"] = color
        enriched["sprite"] = sprite
        sprites.append(enriched)
    random.shuffle(sprites)
    return [sprite["sprite"] for sprite in sprites]


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        target_max_side = random.choice([3, 4])
        shape_set = _choose_shape_set(target_max_side)
        if shape_set is None:
            continue
        sprites = _make_colored_sprites(shape_set)
        height = random.randint(*INPUT_SIZE_RANGE)
        width = random.randint(*INPUT_SIZE_RANGE)
        grid = _place_sprites(sprites, height, width)
        if grid is None:
            continue
        if not _validate_scene(grid):
            continue
        out = main(grid)
        signature = (tuple(out.shape), out.tobytes())
        if signature in SEEN_OUTPUT_SIGNATURES:
            continue
        SEEN_OUTPUT_SIGNATURES.add(signature)
        return grid
    raise RuntimeError("failed to generate a valid unique input")
