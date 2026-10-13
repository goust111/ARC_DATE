from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
MAX_GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _transform_array(arr: np.ndarray, code: int) -> np.ndarray:
    out = arr.copy()
    if code >= 4:
        out = np.fliplr(out)
    return np.rot90(out, code % 4)


def _unique_variants(arr: np.ndarray, eye_color: int, bg: int) -> list[dict]:
    seen = set()
    variants = []
    for code in range(8):
        var = _transform_array(arr, code)
        key = (tuple(var.shape), var.tobytes())
        if key in seen:
            continue
        seen.add(key)
        eye = np.argwhere(var == eye_color)
        assert eye.shape == (1, 2)
        mask = np.argwhere(var != bg)
        variants.append(
            {
                "arr": var.copy(),
                "eye": (int(eye[0, 0]), int(eye[0, 1])),
                "mask": tuple((int(r), int(c)) for r, c in mask.tolist()),
            }
        )
    return variants


def _find_holes_for_container(
    grid: np.ndarray, coords: np.ndarray, color: int, bg: int
) -> list[dict]:
    r0, c0, r1, c1 = _bbox(coords)
    local = grid[r0 : r1 + 1, c0 : c1 + 1]
    object_mask = local == color
    non_object = ~object_mask
    h, w = non_object.shape
    exterior = np.zeros_like(non_object, dtype=bool)
    stack = []
    for r in range(h):
        for c in (0, w - 1):
            if non_object[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if non_object[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                stack.append((r, c))
    while stack:
        rr, cc = stack.pop()
        for nr, nc in _neighbors4(rr, cc):
            if 0 <= nr < h and 0 <= nc < w and non_object[nr, nc] and not exterior[nr, nc]:
                exterior[nr, nc] = True
                stack.append((nr, nc))
    hole_mask = non_object & ~exterior
    holes = []
    for local_coords in _components(hole_mask):
        abs_coords = local_coords + np.array([r0, c0])
        bg_count = sum(1 for rc in abs_coords.tolist() if int(grid[tuple(rc)]) == bg)
        if bg_count == 0:
            continue
        anchor_coords = [tuple(map(int, rc)) for rc in abs_coords.tolist() if int(grid[tuple(rc)]) != bg]
        if len(anchor_coords) != 1:
            continue
        anchor = anchor_coords[0]
        holes.append(
            {
                "cells": frozenset(tuple(map(int, rc)) for rc in abs_coords.tolist()),
                "anchor": anchor,
                "anchor_color": int(grid[anchor]),
            }
        )
    return holes


def _parse_scene(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)

    containers = []
    container_mask = np.zeros_like(g, dtype=bool)
    for color in [int(v) for v in np.unique(g) if int(v) != bg]:
        for coords in _components(g == color):
            holes = _find_holes_for_container(g, coords, color, bg)
            if holes:
                containers.append({"color": color, "coords": coords, "holes": holes})
                container_mask[coords[:, 0], coords[:, 1]] = True

    holes = [hole for container in containers for hole in container["holes"]]
    remaining_mask = (g != bg) & ~container_mask
    sprites = []
    anchors = []
    for coords in _components(remaining_mask):
        values = g[coords[:, 0], coords[:, 1]]
        counts = Counter(map(int, values.tolist()))
        if len(coords) == 1:
            anchors.append((int(coords[0, 0]), int(coords[0, 1])))
            continue
        if len(counts) != 2 or sorted(counts.values()) != [1, len(coords) - 1]:
            raise AssertionError("unexpected non-container component")
        eye_color = next(color for color, cnt in counts.items() if cnt == 1)
        body_color = next(color for color, cnt in counts.items() if cnt > 1)
        r0, c0, r1, c1 = _bbox(coords)
        sprite = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
        for r, c in coords.tolist():
            sprite[int(r) - r0, int(c) - c0] = int(g[int(r), int(c)])
        eye = np.argwhere(sprite == eye_color)
        assert eye.shape == (1, 2)
        sprites.append(
            {
                "coords": coords,
                "arr": sprite,
                "body_color": int(body_color),
                "eye_color": int(eye_color),
                "eye": (int(eye[0, 0]), int(eye[0, 1])),
            }
        )

    return {
        "bg": bg,
        "containers": containers,
        "holes": holes,
        "sprites": sprites,
        "anchors": anchors,
        "container_mask": container_mask,
    }


def _placement_results(sprite: dict, hole: dict, bg: int) -> list[tuple[tuple[int, int, int], ...]]:
    results = []
    for variant in _unique_variants(sprite["arr"], sprite["eye_color"], bg):
        dr = hole["anchor"][0] - variant["eye"][0]
        dc = hole["anchor"][1] - variant["eye"][1]
        placed_cells = frozenset((dr + r, dc + c) for r, c in variant["mask"])
        if placed_cells != hole["cells"]:
            continue
        placed = []
        arr = variant["arr"]
        for r, c in np.argwhere(arr != bg):
            placed.append((dr + int(r), dc + int(c), int(arr[int(r), int(c)])))
        results.append(tuple(sorted(placed)))
    dedup = []
    seen = set()
    for res in results:
        if res not in seen:
            seen.add(res)
            dedup.append(res)
    return dedup


def _shape_only_eye_positions(sprite: dict, hole: dict, bg: int) -> set[tuple[int, int]]:
    positions = set()
    hole_cells = hole["cells"]
    for variant in _unique_variants(sprite["arr"], sprite["eye_color"], bg):
        mask = variant["mask"]
        max_r = max(r for r, _ in mask)
        max_c = max(c for _, c in mask)
        hole_r0 = min(r for r, _ in hole_cells)
        hole_c0 = min(c for _, c in hole_cells)
        hole_r1 = max(r for r, _ in hole_cells)
        hole_c1 = max(c for _, c in hole_cells)
        for dr in range(hole_r0 - max_r, hole_r1 + 1):
            for dc in range(hole_c0 - max_c, hole_c1 + 1):
                placed_cells = frozenset((dr + r, dc + c) for r, c in mask)
                if placed_cells == hole_cells:
                    positions.add((dr + variant["eye"][0], dc + variant["eye"][1]))
    return positions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fit each eye-marked sprite into the matching enclosed hole so the eye lands exactly on the anchor pixel.

    Input structure:
    - The background is uniform.
    - One or more large monochrome container objects each contain enclosed hole regions.
    - Every hole contains exactly one anchor pixel whose color matches the body color of one outside sprite.
    - Each outside sprite is a 4-connected two-color object: one body color plus one single-pixel eye.

    Rule:
    - Detect the enclosed holes inside the monochrome containers, counting the anchor cell as part of its hole.
    - Pair each hole with the unique sprite whose body color matches the hole anchor color.
    - Rotate and/or mirror that sprite, then translate it so its full mask exactly covers the hole cells and its eye lands on the anchor.
    - Clear the original sprite locations and replace the hole with the transformed sprite, while leaving the container walls unchanged.

    Key insight:
    - The anchor pixel serves two roles at once: its color selects which sprite belongs in the hole, and its position fixes the only valid painted eye location when symmetric sprite masks could fit in several orientations.
    """
    scene = _parse_scene(grid)
    bg = scene["bg"]
    holes = scene["holes"]
    sprites = scene["sprites"]
    assert holes and sprites

    holes_by_color = {hole["anchor_color"]: hole for hole in holes}
    sprites_by_color = {sprite["body_color"]: sprite for sprite in sprites}
    assert len(holes_by_color) == len(holes) == len(sprites) == len(sprites_by_color)
    assert set(holes_by_color) == set(sprites_by_color)

    out = np.asarray(grid, dtype=int).copy()
    for sprite in sprites:
        coords = sprite["coords"]
        out[coords[:, 0], coords[:, 1]] = bg

    for color in sorted(holes_by_color):
        hole = holes_by_color[color]
        sprite = sprites_by_color[color]
        results = _placement_results(sprite, hole, bg)
        assert len(results) == 1
        out[hole["anchor"]] = bg
        for r, c, value in results[0]:
            out[r, c] = value
    return out


TEMPLATES = [
    {
        "name": "rect23",
        "coords": {(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)},
        "eye": (0, 0),
        "symmetric": True,
    },
    {
        "name": "plus5",
        "coords": {(1, 0), (0, 1), (1, 1), (2, 1), (1, 2)},
        "eye": (0, 1),
        "symmetric": True,
    },
    {
        "name": "bar4",
        "coords": {(0, 0), (0, 1), (0, 2), (0, 3)},
        "eye": (0, 0),
        "symmetric": True,
    },
    {
        "name": "T5",
        "coords": {(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)},
        "eye": (0, 0),
        "symmetric": True,
    },
    {
        "name": "L4",
        "coords": {(0, 0), (1, 0), (2, 0), (2, 1)},
        "eye": (0, 0),
        "symmetric": False,
    },
    {
        "name": "zig4",
        "coords": {(0, 1), (0, 2), (1, 0), (1, 1)},
        "eye": (0, 2),
        "symmetric": False,
    },
    {
        "name": "hook5",
        "coords": {(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)},
        "eye": (0, 0),
        "symmetric": False,
    },
]


def _coords_to_array(coords: set[tuple[int, int]], eye: tuple[int, int], body_color: int, eye_color: int) -> np.ndarray:
    pts = np.array(sorted(coords), dtype=int)
    r0, c0, r1, c1 = _bbox(pts)
    arr = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in coords:
        arr[r - r0, c - c0] = body_color
    arr[eye[0] - r0, eye[1] - c0] = eye_color
    return arr


def _relative_transform_kind(src: np.ndarray, dst: np.ndarray) -> str:
    if src.shape == dst.shape and np.array_equal(src, dst):
        return "same"
    for k in range(4):
        if np.array_equal(np.rot90(src, k), dst):
            return "rotate"
    for k in range(4):
        if np.array_equal(np.rot90(np.fliplr(src), k), dst):
            return "mirror"
    return "other"


def _rectangles_overlap(a, b) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _expand_rect(rect, pad, h, w):
    r0, c0, r1, c1 = rect
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _arrange_holes(hole_arrays: list[np.ndarray]) -> tuple[np.ndarray, list[tuple[int, int]]] | None:
    dims = [arr.shape for arr in hole_arrays]
    max_h = max(h for h, _ in dims)
    max_w = max(w for _, w in dims)
    sum_h = sum(h for h, _ in dims)
    sum_w = sum(w for _, w in dims)
    for _ in range(120):
        min_h = max(max_h + 2, 6)
        min_w = max(max_w + 2, 6)
        if min_h > 14 or min_w > 14:
            return None
        max_h_bound = max(max_h + 2, sum_h + len(dims) + 1, min_h)
        max_w_bound = max(max_w + 2, sum_w + len(dims) + 1, min_w)
        interior_h = random.randint(min_h, min(14, max_h_bound))
        interior_w = random.randint(min_w, min(14, max_w_bound))
        placed_rects = []
        offsets = [None] * len(hole_arrays)
        order = list(range(len(hole_arrays)))
        order.sort(key=lambda idx: hole_arrays[idx].shape[0] * hole_arrays[idx].shape[1], reverse=True)
        ok = True
        for idx in order:
            arr = hole_arrays[idx]
            hh, ww = arr.shape
            candidates = [
                (r, c)
                for r in range(interior_h - hh + 1)
                for c in range(interior_w - ww + 1)
            ]
            random.shuffle(candidates)
            for r, c in candidates:
                rect = (r, c, r + hh - 1, c + ww - 1)
                if any(
                    _rectangles_overlap(
                        _expand_rect(rect, 1, interior_h, interior_w),
                        _expand_rect(other, 1, interior_h, interior_w),
                    )
                    for other in placed_rects
                ):
                    continue
                placed_rects.append(rect)
                offsets[idx] = (r, c)
                break
            else:
                ok = False
                break
        if not ok:
            continue
        return np.zeros((interior_h + 2, interior_w + 2), dtype=int), offsets
    return None


def _place_object(grid: np.ndarray, arr: np.ndarray, occupied: list[tuple[int, int, int, int]], pad: int) -> tuple[int, int] | None:
    h, w = grid.shape
    ah, aw = arr.shape
    candidates = [(r, c) for r in range(h - ah + 1) for c in range(w - aw + 1)]
    random.shuffle(candidates)
    for r, c in candidates:
        rect = (r, c, r + ah - 1, c + aw - 1)
        if any(
            _rectangles_overlap(_expand_rect(rect, pad, h, w), _expand_rect(other, 0, h, w))
            for other in occupied
        ):
            continue
        if np.any((arr != BG) & (grid[r : r + ah, c : c + aw] != BG)):
            continue
        grid[r : r + ah, c : c + aw][arr != BG] = arr[arr != BG]
        occupied.append(rect)
        return r, c
    return None


def _make_container(color: int, specs: list[dict]) -> dict | None:
    hole_arrays = [spec["hole_arr"] for spec in specs]
    arranged = _arrange_holes(hole_arrays)
    if arranged is None:
        return None
    canvas, offsets = arranged
    canvas[:] = color
    holes = []
    for spec, offset in zip(specs, offsets):
        assert offset is not None
        arr = spec["hole_arr"]
        rr, cc = offset
        rr += 1
        cc += 1
        hole_mask = arr != BG
        canvas[rr : rr + arr.shape[0], cc : cc + arr.shape[1]][hole_mask] = BG
        eye = np.argwhere(arr == spec["eye_color"])
        assert eye.shape == (1, 2)
        anchor = (rr + int(eye[0, 0]), cc + int(eye[0, 1]))
        holes.append(
            {
                "body_color": spec["body_color"],
                "eye_color": spec["eye_color"],
                "arr": arr,
                "anchor_local": anchor,
            }
        )
    return {"arr": canvas, "holes": holes, "color": color}


def _make_expected(grid: np.ndarray, sprite_places: list[dict], container_places: list[dict]) -> np.ndarray:
    out = grid.copy()
    for spec in sprite_places:
        arr = spec["arr"]
        r, c = spec["top_left"]
        region = out[r : r + arr.shape[0], c : c + arr.shape[1]]
        region[arr != BG] = BG
    for container in container_places:
        base_r, base_c = container["top_left"]
        for hole in container["holes"]:
            anchor_r = base_r + hole["anchor_local"][0]
            anchor_c = base_c + hole["anchor_local"][1]
            out[anchor_r, anchor_c] = BG
            arr = hole["arr"]
            for r, c in np.argwhere(arr != BG):
                out[base_r + hole["offsets"][0] + int(r), base_c + hole["offsets"][1] + int(c)] = int(arr[int(r), int(c)])
    return out


def _semantic_checks(inp: np.ndarray, expected: np.ndarray, meta: dict) -> bool:
    if np.array_equal(inp, expected):
        return False
    if not np.array_equal(main(inp), expected):
        return False
    scene = _parse_scene(inp)
    if len(scene["holes"]) != len(scene["sprites"]) or len(scene["holes"]) < 2:
        return False
    has_eye_needed = False
    has_visible_transform = False
    for hole in scene["holes"]:
        sprite = next(s for s in scene["sprites"] if s["body_color"] == hole["anchor_color"])
        results = _placement_results(sprite, hole, scene["bg"])
        if len(results) != 1:
            return False
        if len(_shape_only_eye_positions(sprite, hole, scene["bg"])) >= 2:
            has_eye_needed = True
    for rel in meta["relations"]:
        if rel != "same":
            has_visible_transform = True
    return has_eye_needed and has_visible_transform


def _gen_attempt() -> np.ndarray | None:
    num_containers = 1 if random.random() < 0.45 else 2
    num_sprites = random.randint(max(2, num_containers + 1), 5)
    container_colors = random.sample(COLORS, num_containers)
    body_colors = random.sample([c for c in COLORS if c not in container_colors], num_sprites)

    template_pool = TEMPLATES.copy()
    chosen_templates = [random.choice(template_pool) for _ in range(num_sprites)]
    if not any(t["symmetric"] for t in chosen_templates):
        chosen_templates[0] = random.choice([t for t in TEMPLATES if t["symmetric"]])
    if all(t["symmetric"] for t in chosen_templates):
        chosen_templates[-1] = random.choice([t for t in TEMPLATES if not t["symmetric"]])

    sprite_specs = []
    relations = []
    for idx, (template, body_color) in enumerate(zip(chosen_templates, body_colors)):
        eye_pool = [c for c in COLORS if c != body_color and c not in container_colors]
        eye_color = random.choice(eye_pool)
        base = _coords_to_array(template["coords"], template["eye"], body_color, eye_color)
        source_code = random.randrange(8)
        hole_code = random.randrange(8)
        source_arr = _transform_array(base, source_code)
        hole_arr = _transform_array(base, hole_code)
        sprite_specs.append(
            {
                "body_color": body_color,
                "eye_color": eye_color,
                "source_arr": source_arr,
                "hole_arr": hole_arr,
                "template": template["name"],
            }
        )
        relations.append(_relative_transform_kind(source_arr, hole_arr))

    if all(rel == "same" for rel in relations):
        candidates = [i for i, t in enumerate(chosen_templates) if not t["symmetric"]]
        if not candidates:
            candidates = list(range(num_sprites))
        idx = random.choice(candidates)
        spec = sprite_specs[idx]
        tpl = next(t for t in TEMPLATES if t["name"] == spec["template"])
        base = _coords_to_array(tpl["coords"], tpl["eye"], spec["body_color"], spec["eye_color"])
        alternatives = []
        for code in range(8):
            candidate = _transform_array(base, code)
            if not np.array_equal(candidate, spec["source_arr"]):
                alternatives.append(candidate)
        if not alternatives:
            return None
        spec["hole_arr"] = random.choice(alternatives)
        relations[idx] = _relative_transform_kind(spec["source_arr"], spec["hole_arr"])

    indices = list(range(num_sprites))
    random.shuffle(indices)
    groups = []
    remaining = num_sprites
    start = 0
    for i in range(num_containers):
        groups_left = num_containers - i
        if groups_left == 1:
            take = remaining
        else:
            take = random.randint(1, remaining - groups_left + 1)
        groups.append(indices[start : start + take])
        start += take
        remaining -= take

    containers = []
    for color, group in zip(container_colors, groups):
        specs = [
            {
                "body_color": sprite_specs[idx]["body_color"],
                "eye_color": sprite_specs[idx]["eye_color"],
                "hole_arr": sprite_specs[idx]["hole_arr"],
            }
            for idx in group
        ]
        container = _make_container(color, specs)
        if container is None:
            return None
        container["group"] = group
        containers.append(container)

    object_arrays = [container["arr"] for container in containers] + [spec["source_arr"] for spec in sprite_specs]
    max_obj_h = max(arr.shape[0] for arr in object_arrays)
    max_obj_w = max(arr.shape[1] for arr in object_arrays)
    total_area = sum(int(np.count_nonzero(arr)) for arr in object_arrays)
    min_h = max(18, max_obj_h + 2)
    min_w = max(18, max_obj_w + 2)
    max_h = min(30, max(min_h, int(np.ceil(np.sqrt(total_area))) + 14))
    max_w = min(30, max(min_w, int(np.ceil(np.sqrt(total_area))) + 14))
    if min_h > max_h or min_w > max_w:
        return None
    h = random.randint(min_h, max_h)
    w = random.randint(min_w, max_w)
    grid = np.zeros((h, w), dtype=int)
    occupied: list[tuple[int, int, int, int]] = []

    container_places = []
    for container in containers:
        loc = _place_object(grid, container["arr"], occupied, pad=2)
        if loc is None:
            return None
        base_r, base_c = loc
        holes = []
        for idx in container["group"]:
            hole_arr = sprite_specs[idx]["hole_arr"]
            found = False
            for hole in container["holes"]:
                if hole["body_color"] == sprite_specs[idx]["body_color"]:
                    anchor_r = base_r + hole["anchor_local"][0]
                    anchor_c = base_c + hole["anchor_local"][1]
                    grid[anchor_r, anchor_c] = sprite_specs[idx]["body_color"]
                    hh, ww = hole_arr.shape
                    start_r = hole["anchor_local"][0] - int(np.argwhere(hole_arr == sprite_specs[idx]["eye_color"])[0, 0])
                    start_c = hole["anchor_local"][1] - int(np.argwhere(hole_arr == sprite_specs[idx]["eye_color"])[0, 1])
                    holes.append(
                        {
                            "body_color": sprite_specs[idx]["body_color"],
                            "arr": hole_arr,
                            "anchor_local": hole["anchor_local"],
                            "offsets": (start_r, start_c),
                        }
                    )
                    found = True
                    break
            if not found:
                return None
        container_places.append({"top_left": loc, "holes": holes})

    sprite_places = []
    for spec in sprite_specs:
        loc = _place_object(grid, spec["source_arr"], occupied, pad=1)
        if loc is None:
            return None
        sprite_places.append({"arr": spec["source_arr"], "top_left": loc})

    if _get_bg(grid) != BG:
        return None
    expected = _make_expected(grid, sprite_places, container_places)
    if _get_bg(expected) != BG:
        return None
    if not _semantic_checks(grid, expected, {"relations": relations}):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
