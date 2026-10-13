from __future__ import annotations

import random
from typing import Iterable

import numpy as np


GREY = 5
BG = 0
PALETTE = (1, 2, 3, 4)
DIR_TO_VEC = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
DIR_TO_SIDE = {
    "up": "top",
    "down": "bottom",
    "left": "left",
    "right": "right",
}
SIDE_TO_DIR = {
    "top": "up",
    "bottom": "down",
    "left": "left",
    "right": "right",
}
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _coords_to_mask(coords: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray]:
    r0, c0, r1, c1 = _bbox_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0])
    mask[rel[:, 0], rel[:, 1]] = True
    return (r0, c0, r1, c1), mask


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _find_components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    comps = []
    for coords in _components_mask(grid == int(color)):
        bbox, mask = _coords_to_mask(coords)
        comps.append(
            {
                "color": int(color),
                "coords": coords,
                "bbox": bbox,
                "mask": mask,
                "area": int(coords.shape[0]),
            }
        )
    return comps


def _unique_transforms(mask: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for k in range(4):
        rotated = np.rot90(mask, k)
        for transformed in (rotated, np.fliplr(rotated)):
            key = (tuple(map(int, transformed.shape)), transformed.tobytes())
            if key in seen:
                continue
            seen.add(key)
            out.append(transformed.copy())
    return out


def _canonical_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return min((tuple(map(int, t.shape)), t.tobytes()) for t in _unique_transforms(mask))


def _paint_mask(grid: np.ndarray, anchor: tuple[int, int], mask: np.ndarray, color: int) -> None:
    r0, c0 = map(int, anchor)
    coords = np.argwhere(mask)
    grid[r0 + coords[:, 0], c0 + coords[:, 1]] = int(color)


def _border_color_to_direction(grid: np.ndarray) -> dict[int, str]:
    h, w = map(int, grid.shape)
    assert h >= 6 and w >= 6
    assert int(grid[0, 0]) == 0 and int(grid[0, w - 1]) == 0
    assert int(grid[h - 1, 0]) == 0 and int(grid[h - 1, w - 1]) == 0

    def _side_color(vals: Iterable[int]) -> int:
        colors = {int(v) for v in vals if int(v) != 0}
        assert len(colors) == 1
        return colors.pop()

    top = _side_color(grid[0, 1 : w - 1])
    bottom = _side_color(grid[h - 1, 1 : w - 1])
    left = _side_color(grid[1 : h - 1, 0])
    right = _side_color(grid[1 : h - 1, w - 1])
    assert len({top, bottom, left, right}) == 4
    return {
        int(top): "up",
        int(bottom): "down",
        int(left): "left",
        int(right): "right",
    }


def _extract_holes(grid: np.ndarray) -> list[dict]:
    holes: list[dict] = []
    for comp in _find_components_of_color(grid, GREY):
        r0, c0, r1, c1 = comp["bbox"]
        region = grid[r0 : r1 + 1, c0 : c1 + 1]
        assert 4 <= region.shape[0] <= 6 and 4 <= region.shape[1] <= 6
        colors = {int(v) for v in np.unique(region)}
        assert colors <= {BG, GREY}
        zero_comps = []
        for zcoords in _components_mask(region == BG):
            if np.any(zcoords[:, 0] == 0) or np.any(zcoords[:, 0] == region.shape[0] - 1):
                continue
            if np.any(zcoords[:, 1] == 0) or np.any(zcoords[:, 1] == region.shape[1] - 1):
                continue
            zero_comps.append(zcoords)
        assert len(zero_comps) == 1
        hole_rel = zero_comps[0]
        (hr0, hc0, hr1, hc1), hole_mask = _coords_to_mask(hole_rel)
        assert 2 <= int(hole_mask.sum()) <= 4
        abs_r0 = r0 + hr0
        abs_c0 = c0 + hc0
        abs_coords = np.argwhere(hole_mask) + np.array([abs_r0, abs_c0])
        holes.append(
            {
                "container_bbox": (r0, c0, r1, c1),
                "bbox": (abs_r0, abs_c0, abs_r0 + hole_mask.shape[0] - 1, abs_c0 + hole_mask.shape[1] - 1),
                "anchor": (abs_r0, abs_c0),
                "mask": hole_mask,
                "coords": abs_coords,
                "key": (tuple(map(int, hole_mask.shape)), hole_mask.tobytes()),
            }
        )
    return holes


def _extract_sprites(grid: np.ndarray, color_to_dir: dict[int, str]) -> list[dict]:
    h, w = map(int, grid.shape)
    sprites: list[dict] = []
    for color in sorted(color_to_dir):
        for comp in _find_components_of_color(grid, color):
            r0, c0, r1, c1 = comp["bbox"]
            if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
                continue
            assert 2 <= comp["area"] <= 4
            sprites.append(
                {
                    "color": int(color),
                    "direction": color_to_dir[int(color)],
                    "bbox": comp["bbox"],
                    "mask": comp["mask"],
                    "coords": comp["coords"],
                }
            )
    sprites.sort(key=lambda s: (s["bbox"][0], s["bbox"][1]))
    return sprites


def _path_match_events(sprite: dict, holes: list[dict], available: set[int] | None = None) -> list[tuple[int, int]]:
    htrans = _unique_transforms(sprite["mask"])
    lookup = {}
    for idx, hole in enumerate(holes):
        lookup[(hole["anchor"], hole["key"])] = idx

    dr, dc = DIR_TO_VEC[sprite["direction"]]
    r, c = sprite["bbox"][0], sprite["bbox"][1]
    height, width = sprite["grid_shape"]
    step = 0
    events: list[tuple[int, int]] = []
    while True:
        step += 1
        r += dr
        c += dc
        if not (0 <= r < height and 0 <= c < width):
            break
        hits = set()
        for transformed in htrans:
            idx = lookup.get(((r, c), (tuple(map(int, transformed.shape)), transformed.tobytes())))
            if idx is not None and (available is None or idx in available):
                hits.add(int(idx))
        if hits:
            assert len(hits) == 1
            events.append((step, next(iter(hits))))
    return events


def _simulate(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    color_to_dir = _border_color_to_direction(g)
    holes = _extract_holes(g)
    sprites = _extract_sprites(g, color_to_dir)
    for sprite in sprites:
        sprite["grid_shape"] = g.shape

    out = g.copy()
    remaining = set(range(len(holes)))
    for sprite in sprites:
        out[sprite["coords"][:, 0], sprite["coords"][:, 1]] = BG
        events = _path_match_events(sprite, holes, remaining)
        if not events:
            continue
        _step, hole_idx = min(events, key=lambda x: x[0])
        hole = holes[hole_idx]
        _paint_mask(out, hole["anchor"], hole["mask"], sprite["color"])
        remaining.remove(hole_idx)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each color-keyed sprite along the border-indicated direction until it first shape-matches a container hole, then fill that hole.

    Input structure:
    - Black background with a 1-pixel outer border whose four sides have four distinct colors and black corners.
    - Inside the border are 2 to 4 grey rectangular containers; each is solid grey except for one internal black 4-connected hole.
    - Also inside the border are 2 to 4 small monochrome sprites colored with one of the border-side colors.

    Rule:
    - Read the border colors to get a color-to-direction mapping: top, bottom, left, and right side colors mean up, down, left, and right motion.
    - Process sprites in top-to-bottom, then left-to-right order of their bounding-box top-left corners.
    - For each sprite, translate its bounding-box anchor one cell at a time in its direction; at each new anchor, check whether any rigid transform of the sprite exactly matches a container hole at that anchor.
    - On the first match, erase the sprite from its start position and fill that hole with the sprite color; if no match occurs before leaving the grid, just erase the sprite.

    Key insight:
    - The sprite color determines motion direction via the border, while the stopping point is determined independently by transformed shape equality with a hole.
    """
    return _simulate(np.asarray(grid, dtype=int))


SHAPE_LIBRARY = [
    np.array([[1, 1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
]


def _reserve_ok(reserved: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    height, width = reserved.shape
    rr0 = max(1, r0 - pad)
    cc0 = max(1, c0 - pad)
    rr1 = min(height - 2, r0 + h - 1 + pad)
    cc1 = min(width - 2, c0 + w - 1 + pad)
    return not bool(np.any(reserved[rr0 : rr1 + 1, cc0 : cc1 + 1]))


def _reserve_rect(reserved: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> None:
    height, width = reserved.shape
    rr0 = max(1, r0 - pad)
    cc0 = max(1, c0 - pad)
    rr1 = min(height - 2, r0 + h - 1 + pad)
    cc1 = min(width - 2, c0 + w - 1 + pad)
    reserved[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _all_margin_specs(hh: int, hw: int) -> list[tuple[int, int, int, int]]:
    out = []
    for ch in range(max(4, hh + 2), 7):
        for cw in range(max(4, hw + 2), 7):
            for mt in range(1, ch - hh):
                mb = ch - hh - mt
                if mb < 1:
                    continue
                for ml in range(1, cw - hw):
                    mr = cw - hw - ml
                    if mr < 1:
                        continue
                    out.append((mt, mb, ml, mr))
    return out


def _candidate_anchors(height: int, width: int, hole_mask: np.ndarray) -> list[tuple[int, int]]:
    hh, hw = hole_mask.shape
    return [(r, c) for r in range(2, height - hh - 1) for c in range(2, width - hw - 1)]


def _candidate_starts(height: int, width: int, direction: str, target_anchor: tuple[int, int], start_mask: np.ndarray) -> list[tuple[int, int]]:
    th, tw = start_mask.shape
    tr, tc = target_anchor
    out = []
    if direction == "up":
        if not (1 <= tc <= width - tw - 1):
            return out
        for sr in range(tr + 2, height - th):
            out.append((sr, tc))
    elif direction == "down":
        if not (1 <= tc <= width - tw - 1):
            return out
        for sr in range(1, tr - 1):
            out.append((sr, tc))
    elif direction == "left":
        if not (1 <= tr <= height - th - 1):
            return out
        for sc in range(tc + 2, width - tw):
            out.append((tr, sc))
    else:
        if not (1 <= tr <= height - th - 1):
            return out
        for sc in range(1, tc - 1):
            out.append((tr, sc))
    return out


def _render_input(height: int, width: int, side_colors: dict[str, int], hole_specs: list[dict], sprite_specs: list[dict]) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    grid[0, 1:-1] = side_colors["top"]
    grid[height - 1, 1:-1] = side_colors["bottom"]
    grid[1:-1, 0] = side_colors["left"]
    grid[1:-1, width - 1] = side_colors["right"]
    for hole in hole_specs:
        r0, c0, r1, c1 = hole["container_bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = GREY
        _paint_mask(grid, hole["anchor"], hole["mask"], BG)
    for sprite in sprite_specs:
        _paint_mask(grid, sprite["start_anchor"], sprite["input_mask"], sprite["color"])
    return grid


def _expected_from_spec(grid: np.ndarray, hole_specs: list[dict], sprite_specs: list[dict]) -> np.ndarray:
    out = grid.copy()
    for sprite in sprite_specs:
        coords = np.argwhere(sprite["input_mask"]) + np.array(sprite["start_anchor"])
        out[coords[:, 0], coords[:, 1]] = BG
        if sprite["target_idx"] is not None:
            hole = hole_specs[sprite["target_idx"]]
            _paint_mask(out, hole["anchor"], hole["mask"], sprite["color"])
    return out


def _build_targeted_sprite(
    reserved: np.ndarray,
    height: int,
    width: int,
    direction: str,
    canonical_shape: np.ndarray,
) -> tuple[dict, dict] | None:
    hole_mask = random.choice(_unique_transforms(canonical_shape))
    input_mask = random.choice(_unique_transforms(canonical_shape))
    anchors = _candidate_anchors(height, width, hole_mask)
    random.shuffle(anchors)
    for anchor in anchors:
        start_candidates = _candidate_starts(height, width, direction, anchor, input_mask)
        if not start_candidates:
            continue
        random.shuffle(start_candidates)
        margin_specs = _all_margin_specs(*hole_mask.shape)
        random.shuffle(margin_specs)
        for mt, mb, ml, mr in margin_specs:
            cr0 = anchor[0] - mt
            cc0 = anchor[1] - ml
            ch = mt + hole_mask.shape[0] + mb
            cw = ml + hole_mask.shape[1] + mr
            if not _reserve_ok(reserved, cr0, cc0, ch, cw):
                continue
            for start_anchor in start_candidates:
                if not _reserve_ok(reserved, start_anchor[0], start_anchor[1], input_mask.shape[0], input_mask.shape[1]):
                    continue
                hole_spec = {
                    "anchor": anchor,
                    "mask": hole_mask,
                    "container_bbox": (cr0, cc0, cr0 + ch - 1, cc0 + cw - 1),
                    "canonical_key": _canonical_key(canonical_shape),
                }
                sprite_spec = {
                    "direction": direction,
                    "input_mask": input_mask,
                    "start_anchor": start_anchor,
                    "canonical_key": _canonical_key(canonical_shape),
                }
                _reserve_rect(reserved, cr0, cc0, ch, cw)
                _reserve_rect(reserved, start_anchor[0], start_anchor[1], input_mask.shape[0], input_mask.shape[1])
                return hole_spec, sprite_spec
    return None


def _place_aligned_distractor(
    reserved: np.ndarray,
    height: int,
    width: int,
    sprite_spec: dict,
    target_hole: dict,
    canonical_shape: np.ndarray,
) -> dict | None:
    hole_mask = random.choice(_unique_transforms(canonical_shape))
    direction = sprite_spec["direction"]
    sr, sc = sprite_spec["start_anchor"]
    tr, tc = target_hole["anchor"]
    if direction == "up":
        anchors = [(r, tc) for r in range(tr + 1, sr)]
    elif direction == "down":
        anchors = [(r, tc) for r in range(sr + 1, tr)]
    elif direction == "left":
        anchors = [(tr, c) for c in range(tc + 1, sc)]
    else:
        anchors = [(tr, c) for c in range(sc + 1, tc)]
    random.shuffle(anchors)
    for anchor in anchors:
        if anchor[0] < 2 or anchor[1] < 2:
            continue
        if anchor[0] + hole_mask.shape[0] > height - 2 or anchor[1] + hole_mask.shape[1] > width - 2:
            continue
        margin_specs = _all_margin_specs(*hole_mask.shape)
        random.shuffle(margin_specs)
        for mt, mb, ml, mr in margin_specs:
            cr0 = anchor[0] - mt
            cc0 = anchor[1] - ml
            ch = mt + hole_mask.shape[0] + mb
            cw = ml + hole_mask.shape[1] + mr
            if not _reserve_ok(reserved, cr0, cc0, ch, cw):
                continue
            hole_spec = {
                "anchor": anchor,
                "mask": hole_mask,
                "container_bbox": (cr0, cc0, cr0 + ch - 1, cc0 + cw - 1),
                "canonical_key": _canonical_key(canonical_shape),
            }
            _reserve_rect(reserved, cr0, cc0, ch, cw)
            return hole_spec
    return None


def _place_free_distractor_container(
    reserved: np.ndarray,
    height: int,
    width: int,
    canonical_shape: np.ndarray,
) -> dict | None:
    hole_mask = random.choice(_unique_transforms(canonical_shape))
    anchors = _candidate_anchors(height, width, hole_mask)
    random.shuffle(anchors)
    for anchor in anchors:
        margin_specs = _all_margin_specs(*hole_mask.shape)
        random.shuffle(margin_specs)
        for mt, mb, ml, mr in margin_specs:
            cr0 = anchor[0] - mt
            cc0 = anchor[1] - ml
            ch = mt + hole_mask.shape[0] + mb
            cw = ml + hole_mask.shape[1] + mr
            if not _reserve_ok(reserved, cr0, cc0, ch, cw):
                continue
            hole_spec = {
                "anchor": anchor,
                "mask": hole_mask,
                "container_bbox": (cr0, cc0, cr0 + ch - 1, cc0 + cw - 1),
                "canonical_key": _canonical_key(canonical_shape),
            }
            _reserve_rect(reserved, cr0, cc0, ch, cw)
            return hole_spec
    return None


def _place_unmatched_sprite(
    reserved: np.ndarray,
    height: int,
    width: int,
    direction: str,
    canonical_shape: np.ndarray,
) -> dict | None:
    input_mask = random.choice(_unique_transforms(canonical_shape))
    h, w = input_mask.shape
    anchors = [(r, c) for r in range(1, height - h) for c in range(1, width - w)]
    random.shuffle(anchors)
    for anchor in anchors:
        if not _reserve_ok(reserved, anchor[0], anchor[1], h, w):
            continue
        _reserve_rect(reserved, anchor[0], anchor[1], h, w)
        return {
            "direction": direction,
            "input_mask": input_mask,
            "start_anchor": anchor,
            "canonical_key": _canonical_key(canonical_shape),
        }
    return None


def _validate_generated_instance(grid: np.ndarray, hole_specs: list[dict], sprite_specs: list[dict], expected: np.ndarray) -> bool:
    try:
        color_to_dir = _border_color_to_direction(grid)
        holes = _extract_holes(grid)
        sprites = _extract_sprites(grid, color_to_dir)
    except Exception:
        return False

    if not (2 <= len(holes) <= 4 and 2 <= len(sprites) <= 4):
        return False

    hole_by_anchor = {tuple(h["anchor"]): i for i, h in enumerate(holes)}
    target_count = 0
    seen_targets: set[int] = set()
    sprite_specs_sorted = sorted(sprite_specs, key=lambda s: (s["start_anchor"][0], s["start_anchor"][1]))
    if len(sprite_specs_sorted) != len(sprites):
        return False

    for parsed, spec in zip(sprites, sprite_specs_sorted):
        if parsed["direction"] != spec["direction"]:
            return False
        if (parsed["bbox"][0], parsed["bbox"][1]) != tuple(spec["start_anchor"]):
            return False
        parsed["grid_shape"] = grid.shape
        events = _path_match_events(parsed, holes)
        unique_hits = {hole_idx for _step, hole_idx in events}
        if len(unique_hits) > 1:
            return False
        spec_target = spec["target_idx"]
        if spec_target is None:
            if unique_hits:
                return False
        else:
            target_count += 1
            target_anchor = tuple(hole_specs[spec_target]["anchor"])
            if target_anchor not in hole_by_anchor:
                return False
            parsed_idx = hole_by_anchor[target_anchor]
            if unique_hits != {parsed_idx}:
                return False
            if parsed_idx in seen_targets:
                return False
            seen_targets.add(parsed_idx)

    if target_count < 2:
        return False
    try:
        actual = main(grid)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(18, 24)
    width = random.randint(18, 24)

    shuffled = list(PALETTE)
    random.shuffle(shuffled)
    side_colors = {
        "top": shuffled[0],
        "bottom": shuffled[1],
        "left": shuffled[2],
        "right": shuffled[3],
    }
    dir_to_color = {SIDE_TO_DIR[side]: color for side, color in side_colors.items()}

    matched_count = random.randint(2, 3)
    unmatched_count = 1 if matched_count < 4 and random.random() < 0.45 else 0
    total_sprites = matched_count + unmatched_count
    total_containers = min(4, matched_count + random.randint(0, 1))
    if total_containers < 2 or total_sprites < 2:
        return None

    used_shapes = random.sample(SHAPE_LIBRARY, total_containers + unmatched_count)
    matched_shapes = used_shapes[:matched_count]
    distractor_shapes = used_shapes[matched_count:total_containers]
    unmatched_shapes = used_shapes[total_containers:]

    directions = list(DIR_TO_VEC)
    if matched_count == 2:
        if random.random() < 0.5:
            chosen_dirs = [random.choice(directions)] * 2
        else:
            chosen_dirs = random.sample(directions, 2)
    else:
        repeated = random.choice(directions)
        others = random.sample([d for d in directions if d != repeated], 2)
        chosen_dirs = [repeated, repeated, random.choice(others)]
        random.shuffle(chosen_dirs)

    reserved = np.zeros((height, width), dtype=bool)
    hole_specs: list[dict] = []
    sprite_specs: list[dict] = []

    for idx in range(matched_count):
        placed = _build_targeted_sprite(reserved, height, width, chosen_dirs[idx], matched_shapes[idx])
        if placed is None:
            return None
        hole_spec, sprite_spec = placed
        sprite_spec["color"] = dir_to_color[sprite_spec["direction"]]
        sprite_spec["target_idx"] = len(hole_specs)
        hole_specs.append(hole_spec)
        sprite_specs.append(sprite_spec)

    if distractor_shapes:
        prefer_aligned = random.random() < 0.75
        placed_hole = None
        if prefer_aligned:
            order = list(range(matched_count))
            random.shuffle(order)
            for idx in order:
                placed_hole = _place_aligned_distractor(
                    reserved,
                    height,
                    width,
                    sprite_specs[idx],
                    hole_specs[sprite_specs[idx]["target_idx"]],
                    distractor_shapes[0],
                )
                if placed_hole is not None:
                    break
        if placed_hole is None:
            placed_hole = _place_free_distractor_container(reserved, height, width, distractor_shapes[0])
        if placed_hole is None:
            return None
        hole_specs.append(placed_hole)

    for idx in range(unmatched_count):
        direction = random.choice(directions)
        placed = _place_unmatched_sprite(reserved, height, width, direction, unmatched_shapes[idx])
        if placed is None:
            return None
        placed["color"] = dir_to_color[placed["direction"]]
        placed["target_idx"] = None
        sprite_specs.append(placed)

    grid = _render_input(height, width, side_colors, hole_specs, sprite_specs)
    expected = _expected_from_spec(grid, hole_specs, sprite_specs)
    if not _validate_generated_instance(grid, hole_specs, sprite_specs, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sliding-hole ARC instance")
