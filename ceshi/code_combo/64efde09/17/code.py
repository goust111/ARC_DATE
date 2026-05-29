# Combo variant of 64efde09 (index 17)
# combo_spec: D3/3+D6/0
# Name: 端点涂色+擦除标记

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import crop_bbox, find_components, get_bg


# Design analysis for the fused variant:
# - Behavior trace: in the base task, a labeled active-edge port paints every background cell
#   on its outward ray and the source marker remains visible. In this combo, the same label is
#   still inferred from the input marker, but the output paints only the last reachable background
#   cell on each corresponding ray, then erases the original marker pixel. Example: if a left-facing
#   port sees a boundary marker at col 0 with three background cells in between, the base task
#   colors all three cells and keeps the marker; this combo colors only the cell adjacent to the
#   marker and turns the marker itself back into background.
# - Complexity calibration: endpoint-only output is much sparser, so the generator keeps 3 markers,
#   3-4 sprites, spreads marker carriers across at least 2 sprites, increases boundary clearance,
#   and rejects scenes whose rays are all too short. That preserves repeated structural evidence
#   while making the trajectory inversion necessary.
# - Assumption audit: the original assumptions still needed by the solver are dense rectangular
#   sprites, a unique active long side, unique marker colors, and a unique template match per
#   sprite orientation. The new solver additionally assumes differently colored endpoint pixels do
#   not collide; the generator guarantees this by validating candidate scenes through the solver and
#   rejecting weak endpoint-only cases.
# - Placement / observability: sprites remain padded and within bounds, each active side still faces
#   a grid boundary, markers stay on clear boundary-aligned rays, and no numeric color ordering is
#   used. The repeated disappearance of the boundary markers plus the repeated endpoint pattern is
#   the intended human-visible cue for the fused rule.


def _dihedral_variants(template: np.ndarray) -> list[tuple[str, np.ndarray, np.ndarray]]:
    ids = np.arange(template.size, dtype=int).reshape(template.shape)
    variants: list[tuple[str, np.ndarray, np.ndarray]] = []
    for k in range(4):
        rotated = np.rot90(template, k)
        rotated_ids = np.rot90(ids, k)
        variants.append((f"rot{k}", rotated, rotated_ids))
        variants.append((f"rot{k}_fliplr", np.fliplr(rotated), np.fliplr(rotated_ids)))
    return variants


def _active_side(sub: np.ndarray) -> tuple[tuple[int, int], set[tuple[int, int]]]:
    h, w = sub.shape
    assert h != w, "sprites must be non-square rectangles"

    if h > w:
        sides = [
            ((0, -1), [(r, 0) for r in range(h)]),
            ((0, 1), [(r, w - 1) for r in range(h)]),
        ]
    else:
        sides = [
            ((-1, 0), [(0, c) for c in range(w)]),
            ((1, 0), [(h - 1, c) for c in range(w)]),
        ]

    scored = []
    for direction, coords in sides:
        vals = [int(sub[r, c]) for r, c in coords]
        scored.append((max(Counter(vals).values()), direction, set(coords)))

    scored.sort(key=lambda item: item[0], reverse=True)
    assert scored[0][0] > scored[1][0], "active side must be structurally unique"
    return scored[0][1], scored[0][2]


def _first_marker_on_ray(
    g: np.ndarray,
    bg: int,
    markers: dict[tuple[int, int], int],
    r: int,
    c: int,
    dr: int,
    dc: int,
) -> int | None:
    H, W = g.shape
    r += dr
    c += dc
    while 0 <= r < H and 0 <= c < W:
        if int(g[r, c]) != bg:
            return markers.get((r, c))
        r += dr
        c += dc
    return None


def _trace_bg_cells_on_ray(
    g: np.ndarray,
    bg: int,
    r: int,
    c: int,
    dr: int,
    dc: int,
) -> list[tuple[int, int]]:
    H, W = g.shape
    r += dr
    c += dc
    cells: list[tuple[int, int]] = []
    while 0 <= r < H and 0 <= c < W:
        if int(g[r, c]) != bg:
            break
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _extract_scene(g: np.ndarray) -> tuple[int, list[dict], dict[tuple[int, int], int]]:
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=False)

    sprites_raw: list[dict] = []
    markers: dict[tuple[int, int], int] = {}
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        area = int(comp["area"])
        if area == 1:
            r, c = map(int, comp["coords"][0])
            markers[(r, c)] = int(g[r, c])
        else:
            assert area == h * w, "sprite components must be dense rectangles"
            sprites_raw.append({"bbox": (r0, c0, r1, c1), "sub": crop_bbox(g, (r0, c0, r1, c1))})

    assert len(sprites_raw) >= 2, "expected multiple transformed rectangular sprites"
    assert markers, "expected isolated control markers"
    assert len(set(markers.values())) == len(markers), "control marker colors must be unique"
    return bg, sprites_raw, markers


def _match_sprites(sprites_raw: list[dict]) -> list[dict]:
    variants = _dihedral_variants(sprites_raw[0]["sub"])
    sprites: list[dict] = []
    for sprite in sprites_raw:
        matches = [
            (ids, name)
            for name, sub, ids in variants
            if sub.shape == sprite["sub"].shape and np.array_equal(sub, sprite["sub"])
        ]
        assert len(matches) == 1, "each sprite must match the template in exactly one orientation"
        ids, _name = matches[0]
        direction, active = _active_side(sprite["sub"])
        sprites.append({**sprite, "ids": ids, "direction": direction, "active": active})
    return sprites


def _infer_labels(
    g: np.ndarray,
    bg: int,
    sprites: list[dict],
    markers: dict[tuple[int, int], int],
) -> dict[int, int]:
    color_to_template_id: dict[int, int] = {}
    template_id_to_color: dict[int, int] = {}
    for sprite in sprites:
        r0, c0, _r1, _c1 = sprite["bbox"]
        dr, dc = sprite["direction"]
        for lr, lc in sprite["active"]:
            marker_color = _first_marker_on_ray(g, bg, markers, r0 + lr, c0 + lc, dr, dc)
            if marker_color is None:
                continue
            template_id = int(sprite["ids"][lr, lc])
            if marker_color in color_to_template_id:
                assert color_to_template_id[marker_color] == template_id
            if template_id in template_id_to_color:
                assert template_id_to_color[template_id] == marker_color
            color_to_template_id[marker_color] = template_id
            template_id_to_color[template_id] = marker_color

    assert set(color_to_template_id) == set(markers.values()), "every control marker must label one active-edge cell"
    return color_to_template_id


def _scene_analysis(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg, sprites_raw, markers = _extract_scene(g)
    sprites = _match_sprites(sprites_raw)
    color_to_template_id = _infer_labels(g, bg, sprites, markers)

    rays: list[dict] = []
    for sprite in sprites:
        r0, c0, _r1, _c1 = sprite["bbox"]
        dr, dc = sprite["direction"]
        for color, template_id in color_to_template_id.items():
            locs = np.argwhere(sprite["ids"] == template_id)
            assert locs.shape[0] == 1
            lr, lc = map(int, locs[0])
            assert (lr, lc) in sprite["active"], "mapped port must land on the sprite active edge"
            cells = _trace_bg_cells_on_ray(g, bg, r0 + lr, c0 + lc, dr, dc)
            rays.append(
                {
                    "color": color,
                    "template_id": template_id,
                    "endpoint": cells[-1] if cells else None,
                    "length": len(cells),
                    "cells": cells,
                    "bbox": sprite["bbox"],
                    "direction": (dr, dc),
                }
            )

    return {
        "grid": g,
        "bg": bg,
        "markers": markers,
        "sprites": sprites,
        "color_to_template_id": color_to_template_id,
        "rays": rays,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer hidden port labels across matching rectangle sprites, but keep only each ray's final landing pixel.

    Input structure:
    - The background is the most common color.
    - Several dense 2-by-N or N-by-2 rectangles are dihedral variants of one shared template.
    - Single-pixel markers of unique colors lie on outward boundary-facing rays from some active-edge cells.

    Rule:
    - For each sprite, detect the active long edge as the side with the stronger same-color majority.
    - A marker first seen along an outward active-edge ray labels the corresponding template port for all matching sprites.
    - In the output, erase the original marker pixels, then for every labeled port on every sprite color only the final reachable background cell on that outward ray, stopping before the first obstacle or before leaving the grid.

    Key insight:
    - The visible boundary markers are only clues in the input: they identify template-relative ports, but the output removes them and preserves just the ray endpoints that those hidden labels imply across all orientations.
    """
    scene = _scene_analysis(grid)
    g = scene["grid"]
    bg = int(scene["bg"])
    out = g.copy()

    for r, c in scene["markers"]:
        out[r, c] = bg

    for ray in scene["rays"]:
        endpoint = ray["endpoint"]
        if endpoint is None:
            continue
        rr, cc = endpoint
        color = int(ray["color"])
        assert int(out[rr, cc]) in (bg, color), "endpoint rays must not conflict"
        out[rr, cc] = color

    return out


def _continuous_output(grid: np.ndarray) -> np.ndarray:
    scene = _scene_analysis(grid)
    g = scene["grid"]
    bg = int(scene["bg"])
    out = g.copy()

    for ray in scene["rays"]:
        color = int(ray["color"])
        for rr, cc in ray["cells"]:
            assert int(out[rr, cc]) in (bg, color), "latent full rays must not conflict"
            out[rr, cc] = color

    return out


# --------------- generator (combo) ---------------

_BG = 8
_NON_BG_COLORS = [0, 1, 2, 3, 4, 5, 6, 7, 9]
_GRID_H_RANGE = (21, 29)
_GRID_W_RANGE = (27, 29)
_TEMPLATE_N_RANGE = (7, 9)
_TEMPLATE_COLOR_COUNT = (3, 4)
_ACTIVE_MAJORITY_RANGE = (5, 9)
_SPRITE_COUNT_RANGE = (3, 4)
_NUM_MARKERS = 3
_PLACEMENT_PAD = 3
_PLACEMENT_CLEARANCE = 4
_PLACEMENT_RETRIES = 160
_GEN_RETRIES = 500
_MIN_MARKER_CARRIERS = 2
_MIN_MAX_RAY_LEN = 3
_MIN_TOTAL_PATH_SAVINGS = 10


def _build_column(n: int, colors: list[int], majority: int, dominant: int) -> list[int]:
    col = [dominant] * majority
    others = [c for c in colors if c != dominant]
    fill = others if others else colors
    for _ in range(n - majority):
        col.append(random.choice(fill))
    random.shuffle(col)
    return col


def _design_template(N: int) -> tuple[np.ndarray, np.ndarray, list[int]] | None:
    num_colors = random.randint(*_TEMPLATE_COLOR_COUNT)
    template_colors = random.sample(_NON_BG_COLORS, num_colors)

    active_majority = random.randint(_ACTIVE_MAJORITY_RANGE[0], min(N, _ACTIVE_MAJORITY_RANGE[1]))
    left_col = _build_column(N, template_colors, active_majority, random.choice(template_colors))

    max_right_majority = active_majority - 1
    right_majority = random.randint(max(2, N // num_colors), max_right_majority)
    right_dominant = random.choice(template_colors)
    if not [c for c in template_colors if c != right_dominant]:
        return None
    right_col = _build_column(N, template_colors, right_majority, right_dominant)

    if max(Counter(left_col).values()) <= max(Counter(right_col).values()):
        return None

    template = np.column_stack([left_col, right_col]).astype(int)
    if len(set(template.flat)) < num_colors:
        return None

    ids_base = np.arange(N * 2, dtype=int).reshape(N, 2)
    return template, ids_base, template_colors


def _deduplicated_variants(
    template: np.ndarray,
    ids_base: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray, tuple[int, int], list[tuple[int, int]]]]:
    raw = _dihedral_variants(template)
    raw_ids = _dihedral_variants(ids_base)

    seen: list[np.ndarray] = []
    result = []
    for (_, sub, _), (_, _, ids_arr) in zip(raw, raw_ids):
        if any(s.shape == sub.shape and np.array_equal(s, sub) for s in seen):
            continue
        try:
            direction, active_set = _active_side(sub)
        except AssertionError:
            continue
        seen.append(sub)
        result.append((sub, ids_arr, direction, sorted(active_set)))
    return result


def _random_sprite_position(H: int, W: int, h: int, w: int, dr: int, dc: int) -> tuple[int, int]:
    pad = _PLACEMENT_CLEARANCE
    if dr == 0 and dc == -1:
        return random.randint(0, H - h), random.randint(pad, W - w)
    if dr == 0 and dc == 1:
        return random.randint(0, H - h), random.randint(0, W - w - pad - 1)
    if dr == -1 and dc == 0:
        return random.randint(pad, H - h), random.randint(0, W - w)
    return random.randint(0, H - h - pad - 1), random.randint(0, W - w)


def _occupied_set(placements: list[tuple]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for r0, c0, sub, *_ in placements:
        h, w = sub.shape
        for r in range(h):
            for c in range(w):
                cells.add((r0 + r, c0 + c))
    return cells


def _place_sprites(H: int, W: int, variants) -> list[tuple] | None:
    placements: list[tuple] = []
    pad = _PLACEMENT_PAD
    for sub, ids_arr, direction, active_coords in variants:
        h, w = sub.shape
        dr, dc = direction
        placed = False
        for _ in range(_PLACEMENT_RETRIES):
            r0, c0 = _random_sprite_position(H, W, h, w, dr, dc)
            occ = _occupied_set(placements)
            if any((r0 + r, c0 + c) in occ for r in range(-pad, h + pad) for c in range(-pad, w + pad)):
                continue
            placements.append((r0, c0, sub, ids_arr, direction, active_coords))
            placed = True
            break
        if not placed:
            return None
    return placements


def _boundary_position(r: int, c: int, dr: int, dc: int, H: int, W: int) -> tuple[int, int]:
    if dr == 0 and dc == -1:
        return r, 0
    if dr == 0 and dc == 1:
        return r, W - 1
    if dr == -1 and dc == 0:
        return 0, c
    return H - 1, c


def _place_markers(
    grid: np.ndarray,
    sprite_placements: list[tuple],
    ids_base: np.ndarray,
    template_colors: list[int],
) -> bool:
    H, W = grid.shape
    N = ids_base.shape[0]
    num_sprites = len(sprite_placements)

    active_template_ids = [int(ids_base[r, 0]) for r in range(N)]
    if len(active_template_ids) < _NUM_MARKERS:
        return False
    labeled_ids = random.sample(active_template_ids, _NUM_MARKERS)

    available = [c for c in range(10) if c != _BG and c not in template_colors]
    if len(available) < _NUM_MARKERS:
        return False
    marker_colors = random.sample(available, _NUM_MARKERS)

    carrier_count = random.randint(_MIN_MARKER_CARRIERS, min(_NUM_MARKERS, num_sprites))
    carriers = random.sample(range(num_sprites), carrier_count)
    sprite_indices = carriers.copy()
    while len(sprite_indices) < _NUM_MARKERS:
        sprite_indices.append(random.choice(carriers))
    random.shuffle(sprite_indices)

    for color, tid, si in zip(marker_colors, labeled_ids, sprite_indices):
        r0, c0, _sub, ids_arr, direction, active_coords = sprite_placements[si]
        dr, dc = direction

        locs = np.argwhere(ids_arr == tid)
        if locs.shape[0] != 1:
            return False
        lr, lc = map(int, locs[0])
        if (lr, lc) not in set(active_coords):
            return False

        marker_r, marker_c = _boundary_position(r0 + lr, c0 + lc, dr, dc, H, W)

        cr, cc = r0 + lr + dr, c0 + lc + dc
        ray_len = 0
        while (cr, cc) != (marker_r, marker_c) and 0 <= cr < H and 0 <= cc < W:
            if grid[cr, cc] != _BG:
                return False
            ray_len += 1
            cr += dr
            cc += dc

        if ray_len < 1:
            return False
        if grid[marker_r, marker_c] != _BG:
            return False
        grid[marker_r, marker_c] = color
    return True


def _combo_signal_metrics(grid: np.ndarray) -> dict:
    scene = _scene_analysis(grid)
    ray_lengths = [int(ray["length"]) for ray in scene["rays"]]
    endpoints = [tuple(ray["endpoint"]) for ray in scene["rays"] if ray["endpoint"] is not None]
    all_ray_cells = [cell for ray in scene["rays"] for cell in ray["cells"]]
    endpoint_count = sum(length > 0 for length in ray_lengths)
    total_path_savings = sum(max(0, length - 1) for length in ray_lengths)
    return {
        "marker_count": len(scene["markers"]),
        "ray_lengths": ray_lengths,
        "endpoint_count": endpoint_count,
        "distinct_endpoint_count": len(set(endpoints)),
        "disjoint_ray_cells": len(set(all_ray_cells)) == len(all_ray_cells),
        "max_ray_len": max(ray_lengths) if ray_lengths else 0,
        "total_path_savings": total_path_savings,
    }


def _gen_attempt() -> np.ndarray | None:
    N = random.randint(*_TEMPLATE_N_RANGE)
    H = random.randint(*_GRID_H_RANGE)
    W = random.randint(*_GRID_W_RANGE)

    design = _design_template(N)
    if design is None:
        return None
    template, ids_base, template_colors = design

    unique_variants = _deduplicated_variants(template, ids_base)
    num_sprites = random.randint(*_SPRITE_COUNT_RANGE)
    if len(unique_variants) < num_sprites:
        return None
    chosen = random.sample(unique_variants, num_sprites)
    if len({("h" if direction[0] == 0 else "v") for *_rest, direction, _active in chosen}) < 2:
        return None

    placements = _place_sprites(H, W, chosen)
    if placements is None:
        return None

    grid = np.full((H, W), _BG, dtype=int)
    for r0, c0, sub, *_ in placements:
        h, w = sub.shape
        grid[r0:r0 + h, c0:c0 + w] = sub

    if not _place_markers(grid, placements, ids_base, template_colors):
        return None

    try:
        main(grid)
        _continuous_output(grid)
    except Exception:
        return None

    metrics = _combo_signal_metrics(grid)
    if metrics["marker_count"] != _NUM_MARKERS:
        return None
    if metrics["endpoint_count"] != len(metrics["ray_lengths"]):
        return None
    if metrics["distinct_endpoint_count"] != metrics["endpoint_count"]:
        return None
    if not metrics["disjoint_ray_cells"]:
        return None
    if metrics["max_ray_len"] < _MIN_MAX_RAY_LEN:
        return None
    if metrics["total_path_savings"] < _MIN_TOTAL_PATH_SAVINGS:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
