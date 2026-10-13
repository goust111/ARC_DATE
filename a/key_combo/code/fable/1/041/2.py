from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: BBox) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.asarray(grid, dtype=int)[r0 : r1 + 1, c0 : c1 + 1].copy()


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a non-empty mask"
    r0, c0, r1, c1 = _bbox_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    H, W = mask.shape
    seen = np.zeros((H, W), dtype=bool)
    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]

    comps: list[np.ndarray] = []
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[Coord] = deque([(r, c)])
            seen[r, c] = True
            coords: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in steps:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _find_components(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    connectivity: int = 4,
) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g) if bg is None else int(bg)
    comps: list[dict] = []
    for color in sorted(set(map(int, g.ravel())) - {bg}):
        for coords in _components_mask(g == color, connectivity=connectivity):
            comps.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": _bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def _mask_from_component(comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = np.asarray(comp["coords"], dtype=int) - np.array([r0, c0], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True
    return mask


def _variants(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    out: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for flip in (False, True):
        cur = np.flipud(mask) if flip else mask
        for k in range(4):
            var = np.rot90(cur, k)
            key = (tuple(map(int, var.shape)), var.tobytes())
            if key not in seen:
                seen.add(key)
                out.append(var.copy())
    return out


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return min((tuple(map(int, var.shape)), var.tobytes()) for var in _variants(mask))


def _extract_instance(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)

    comps = _find_components(g, bg=bg, connectivity=4)
    assert comps, "expected a frame and sprites"
    max_area = max(int(comp["area"]) for comp in comps)
    frame_candidates = [comp for comp in comps if int(comp["area"]) == max_area]
    assert len(frame_candidates) == 1, "expected a unique largest frame component"
    frame = frame_candidates[0]
    frame_color = int(frame["color"])
    frame_bbox = frame["bbox"]
    sub = _crop_bbox(g, frame_bbox)

    hole_mask = sub != frame_color
    hole_coords = _components_mask(hole_mask, connectivity=4)
    assert hole_coords, "expected at least one hole"
    holes = []
    for coords in hole_coords:
        hb = _bbox_coords(coords)
        r0, c0, r1, c1 = hb
        local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        rel = coords - np.array([r0, c0], dtype=int)
        local[rel[:, 0], rel[:, 1]] = True
        holes.append(
            {
                "bbox": hb,
                "mask": local,
                "key": _shape_key(_tight_mask(local)),
                "comp_key": _shape_key(_tight_mask(~local)),
            }
        )

    r0, c0, r1, c1 = frame_bbox
    outside = g.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = bg
    sprites = []
    for comp in _find_components(outside, bg=bg, connectivity=4):
        mask = _mask_from_component(comp)
        sprites.append(
            {
                "color": int(comp["color"]),
                "bbox": comp["bbox"],
                "mask": mask,
                "key": _shape_key(mask),
            }
        )

    return {
        "grid": g,
        "bg": bg,
        "frame_color": frame_color,
        "frame_bbox": frame_bbox,
        "frame_crop": sub,
        "holes": holes,
        "sprites": sprites,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each frame hole with the color of the outside sprite that complements its bounding rectangle.

    Input structure:
    - The background is the most common color.
    - One large monochrome template frame contains several hole regions made of non-frame pixels.
    - Outside the frame are small monochrome sprites, including same-shape decoys.

    Rule:
    - Find the unique largest frame component and crop to its bounding box.
    - For each hole, take its local bounding rectangle and compute the complementary shape obtained by removing the hole from that rectangle.
    - Match that complementary shape to an outside sprite up to rotation or reflection, then repaint the whole hole with the matched sprite's color while leaving the rest of the cropped frame color unchanged.

    Key insight:
    - The correct key is not the sprite congruent to the hole itself, but the sprite that joins with the hole to make a solid rectangle.
    """
    parsed = _extract_instance(grid)
    frame_color = int(parsed["frame_color"])
    out = np.full(parsed["frame_crop"].shape, frame_color, dtype=int)

    sprites_by_key: dict[tuple[tuple[int, int], bytes], Counter] = {}
    for sprite in parsed["sprites"]:
        bucket = sprites_by_key.setdefault(sprite["key"], Counter())
        bucket[int(sprite["color"])] += 1

    for hole in parsed["holes"]:
        key = hole["comp_key"]
        assert key in sprites_by_key, "missing complementary key sprite"
        color = max(sprites_by_key[key].items(), key=lambda kv: (kv[1], -kv[0]))[0]
        r0, c0, r1, c1 = hole["bbox"]
        target = out[r0 : r1 + 1, c0 : c1 + 1]
        target[hole["mask"]] = int(color)

    return out


def _mask(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


_BASE_HOLES = [
    _mask(["111", "100"]),
    _mask(["1111", "1000"]),
    _mask(["111", "100", "100"]),
    _mask(["111", "110", "100"]),
    _mask(["111", "101", "100"]),
    _mask(["1111", "1100", "1000"]),
    _mask(["1111", "1010", "1000"]),
    _mask(["1111", "1100", "1000", "1000"]),
    _mask(["1111", "1110", "1000", "1000"]),
    _mask(["1111", "1101", "1001", "1000"]),
]


def _build_library() -> list[dict]:
    infos = []
    for base in _BASE_HOLES:
        hole_key = _shape_key(base)
        comp_key = _shape_key(_tight_mask(~base))
        assert hole_key != comp_key
        infos.append(
            {
                "mask": base,
                "shape": tuple(map(int, base.shape)),
                "hole_key": hole_key,
                "comp_key": comp_key,
            }
        )
    return infos


_HOLE_LIBRARY = _build_library()


def _compatible_specs(specs: list[dict]) -> bool:
    hole_keys = [spec["hole_key"] for spec in specs]
    comp_keys = [spec["comp_key"] for spec in specs]
    return (
        len(hole_keys) == len(set(hole_keys))
        and len(comp_keys) == len(set(comp_keys))
        and not (set(hole_keys) & set(comp_keys))
    )


def _choose_hole_specs() -> list[dict]:
    groups: dict[tuple[int, int], list[dict]] = {}
    for info in _HOLE_LIBRARY:
        groups.setdefault(info["shape"], []).append(info)

    for _ in range(200):
        count = random.randint(2, 4)
        chosen: list[dict] = []
        if count >= 2 and random.random() < 0.55:
            same_size_groups = [grp for grp in groups.values() if len(grp) >= 2]
            grp = random.choice(same_size_groups)
            chosen.extend(random.sample(grp, 2))

        chosen_keys = {(info["hole_key"], info["comp_key"]) for info in chosen}
        remaining = [info for info in _HOLE_LIBRARY if (info["hole_key"], info["comp_key"]) not in chosen_keys]
        needed = count - len(chosen)
        chosen.extend(random.sample(remaining, needed))
        random.shuffle(chosen)
        if _compatible_specs(chosen):
            return chosen
    raise ValueError("failed to choose a compatible hole set")


def _random_box_location(occ: np.ndarray, h: int, w: int, *, padding: int, border: int) -> Coord | None:
    H, W = occ.shape
    candidates: list[Coord] = []
    for r in range(border, H - h - border + 1):
        for c in range(border, W - w - border + 1):
            rr0 = max(0, r - padding)
            cc0 = max(0, c - padding)
            rr1 = min(H, r + h + padding)
            cc1 = min(W, c + w + padding)
            if not occ[rr0:rr1, cc0:cc1].any():
                candidates.append((r, c))
    if not candidates:
        return None
    return random.choice(candidates)


def _paint_mask(dst: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    view = dst[top : top + h, left : left + w]
    view[np.asarray(mask, dtype=bool)] = int(color)


def _render_sprite(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    patch = np.full(mask.shape, int(bg), dtype=int)
    patch[np.asarray(mask, dtype=bool)] = int(color)
    return patch


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    bg = random.randint(0, 9)
    frame_color = random.choice([c for c in range(10) if c != bg])
    hole_specs = _choose_hole_specs()

    hole_masks = [random.choice(_variants(spec["mask"])) for spec in hole_specs]
    forced_idx = random.randrange(len(hole_masks))
    key_masks = []
    decoy_masks = []
    for idx, hole_mask in enumerate(hole_masks):
        comp_mask = _tight_mask(~hole_mask)
        key_variants = _variants(comp_mask)
        non_identity = [var for var in key_variants if var.shape != comp_mask.shape or not np.array_equal(var, comp_mask)]
        if idx == forced_idx and non_identity:
            key_masks.append(random.choice(non_identity))
        else:
            key_masks.append(random.choice(key_variants))

        hole_variants = _variants(hole_mask)
        hole_non_identity = [var for var in hole_variants if var.shape != hole_mask.shape or not np.array_equal(var, hole_mask)]
        decoy_masks.append(random.choice(hole_non_identity or hole_variants))

    sprite_count = 2 * len(hole_specs)
    remaining_colors = [c for c in range(10) if c not in {bg, frame_color}]
    sprite_colors = random.sample(remaining_colors, sprite_count)
    match_colors = sprite_colors[: len(hole_specs)]
    decoy_colors = sprite_colors[len(hole_specs) :]

    hole_order = sorted(range(len(hole_masks)), key=lambda i: hole_masks[i].shape[0] * hole_masks[i].shape[1], reverse=True)
    inner_h = random.randint(8, 14)
    inner_w = random.randint(8, 14)
    occ = np.zeros((inner_h, inner_w), dtype=bool)
    placements: list[Coord | None] = [None] * len(hole_masks)
    for idx in hole_order:
        h, w = hole_masks[idx].shape
        pos = _random_box_location(occ, h, w, padding=1, border=0)
        if pos is None:
            raise ValueError("failed to place hole")
        r, c = pos
        occ[r : r + h, c : c + w] = True
        placements[idx] = pos

    frame_input = np.full((inner_h + 2, inner_w + 2), frame_color, dtype=int)
    expected = frame_input.copy()
    for idx, (hole_mask, pos) in enumerate(zip(hole_masks, placements)):
        assert pos is not None
        r, c = pos
        top = r + 1
        left = c + 1
        _paint_mask(frame_input, hole_mask, top, left, bg)
        _paint_mask(expected, hole_mask, top, left, match_colors[idx])

    frame_h, frame_w = frame_input.shape
    extra_h = random.randint(8, min(12, 30 - frame_h))
    extra_w = random.randint(8, min(12, 30 - frame_w))
    H = frame_h + extra_h
    W = frame_w + extra_w
    grid = np.full((H, W), bg, dtype=int)

    frame_top = random.randint(1, H - frame_h - 1)
    frame_left = random.randint(1, W - frame_w - 1)
    grid[frame_top : frame_top + frame_h, frame_left : frame_left + frame_w] = frame_input

    sprite_items = []
    for idx, key_mask in enumerate(key_masks):
        sprite_items.append({"mask": key_mask, "color": match_colors[idx], "kind": "key", "idx": idx})
    for idx, decoy_mask in enumerate(decoy_masks):
        sprite_items.append({"mask": decoy_mask, "color": decoy_colors[idx], "kind": "decoy", "idx": idx})
    sprite_items.sort(key=lambda item: item["mask"].shape[0] * item["mask"].shape[1], reverse=True)

    occ_grid = np.zeros((H, W), dtype=bool)
    occ_grid[frame_top : frame_top + frame_h, frame_left : frame_left + frame_w] = True
    for item in sprite_items:
        mask = item["mask"]
        h, w = mask.shape
        pos = _random_box_location(occ_grid, h, w, padding=1, border=1)
        if pos is None:
            raise ValueError("failed to place sprite")
        r, c = pos
        occ_grid[r : r + h, c : c + w] = True
        patch = _render_sprite(mask, int(item["color"]), bg)
        sprite_region = grid[r : r + h, c : c + w]
        sprite_region[mask] = patch[mask]

    if _get_bg(grid) != bg:
        raise ValueError("background stopped being dominant")

    return grid, expected


def _semantic_check(grid: np.ndarray, expected: np.ndarray) -> bool:
    parsed = _extract_instance(grid)
    if parsed["frame_crop"].shape != expected.shape:
        return False

    hole_comp_keys = [hole["comp_key"] for hole in parsed["holes"]]
    if len(hole_comp_keys) != len(set(hole_comp_keys)):
        return False

    sprite_key_counts = Counter(sprite["key"] for sprite in parsed["sprites"])
    if any(sprite_key_counts[key] != 1 for key in hole_comp_keys):
        return False

    decoy_hits = sum(sprite_key_counts[hole["key"]] for hole in parsed["holes"])
    if decoy_hits < 1:
        return False

    if any(hole["key"] in set(hole_comp_keys) for hole in parsed["holes"]):
        return False

    return np.array_equal(main(grid), expected)


def generate_input() -> np.ndarray:
    for _ in range(400):
        try:
            grid, expected = _generate_case()
        except Exception:
            continue
        if _semantic_check(grid, expected):
            return grid
    raise RuntimeError("failed to generate a valid complement-key sample")
