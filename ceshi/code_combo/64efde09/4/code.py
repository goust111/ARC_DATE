# Combo variant of 64efde09 (index 4)
# combo_spec: D5/0+D6/3
# Name: 穿透攻击射线

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import crop_bbox, find_components, get_bg


# Design analysis for combo D5/0 + D6/3
# --------------------------------------
# a) Concrete behavior:
# - Start from the source task grammar: dense rectangular sprites share one Nx2 template, and
#   boundary markers label active-edge template ports.
# - D5/0 changes both marker reading and ray travel: sprites are transparent to the ray logic.
#   A boundary marker may sit "behind" another sprite, and an output ray keeps stepping after
#   entering a different sprite instead of stopping there.
# - D6/3 adds an attack consequence on top of that transparency: every sprite touched by another
#   sprite's ray disappears from the output. The important fused behavior is not "ray ignores
#   sprites" in isolation, but "ray crosses a sprite, that sprite is erased, and the same beam
#   continues through the newly emptied region".
# - Small trace: suppose two left-facing copies share the same rows, with a blocker at col 6..7
#   and an emitter at col 24..25. A marker for row 3 sits at col 0. Old logic stops the marker
#   scan at the blocker and never discovers the marker. D5-only logic finds the marker and paints
#   through bg gaps while leaving the blocker intact. The fused logic finds the marker, marks row 3
#   as an active port, erases the blocker sprite when that row is crossed, and paints the beam all
#   the way to the boundary, including the cells that became empty after erasure.
#
# b) Complexity calibration:
# - A fully free placement would create many accidental ray intersections, so the generator uses
#   horizontal-only attack lanes. This keeps the task readable while still showing repeated
#   evidence: 3 marker colors, 3-4 sprites, and 1-2 destroyed targets per sample.
# - The layout keeps at least one untouched emitter so the solver sees both outcomes in the same
#   output: some sprites survive, some vanish, and the same labeled ports still replicate across
#   all copies.
# - The rays stay long enough to be visually obvious, and each marker labels a repeated template
#   row rather than a one-off absolute coordinate.
#
# c) Broken assumptions from the source solver:
# - `_first_marker_on_ray(...)` used to treat "first non-bg" as decisive. That fails because a
#   blocker sprite may appear before the marker. The fused version must scan to the boundary and
#   only care about marker cells.
# - The drawing loop used to break on any non-bg cell. That fails twice here: sprites no longer
#   stop rays, and hit sprites must be removed instead of preserved.
# - `out = g.copy()` followed by painting bg cells is insufficient. We first have to determine
#   which sprites are hit from the original scene, erase those sprites in the output, and only
#   then replay the ray paths so the beam can occupy the destroyed lanes.
# - Using the raw 8 dihedral transforms can produce duplicate matches for symmetric templates.
#   The fused solver deduplicates equivalent variants before orientation matching.
#
# d) Placement legality:
# - Sprites remain dense non-overlapping rectangles fully in bounds.
# - Top-band markers always lie on the boundary cell directly in front of the rightmost emitter's
#   active edge, with a left-side blocker between them. This guarantees true D5-style occluded
#   labeling.
# - Horizontal attack lanes are separated by bands, so different colors never need to share one
#   background cell.
#
# e) Visual observability:
# - The input already advertises transparency because the only plausible marker for the top-right
#   emitter sits behind another sprite.
# - The output makes the attack semantics visible because the crossed blocker rectangle disappears
#   while the beam continues beyond it. This is much stronger than a subtle recolor and can be
#   inferred from 2-3 examples.
#
# f) Visual cleanliness:
# - All generated attack rays are horizontal, which avoids orthogonal beam clutter.
# - Destroyed sprites are isolated blocks, so the "missing rectangle plus continuing beam" pattern
#   stays easy to parse by eye.
#
# g) Color inference:
# - Marker colors are unique and never overlap template colors.
# - Only 3 marker colors are used, and each repeats across all sprite copies via the same relative
#   active-edge rows. The color rule is therefore tied to visible template structure, not to
#   numeric color ids.
#
# Post-change solver assumptions enforced by the generator:
# - There is at least one marker whose sight-line crosses another sprite before reaching the
#   labeling emitter.
# - There is at least one sprite hit by a ray from a different sprite.
# - Rays of different colors never need to overlap on the same non-marker cell.
# - Every hit sprite is dense, so erasing its full bbox is equivalent to erasing the object.


def _dihedral_variants(template: np.ndarray) -> list[tuple[str, np.ndarray, np.ndarray]]:
    ids = np.arange(template.size, dtype=int).reshape(template.shape)
    variants: list[tuple[str, np.ndarray, np.ndarray]] = []
    for k in range(4):
        rotated = np.rot90(template, k)
        rotated_ids = np.rot90(ids, k)
        variants.append((f"rot{k}", rotated, rotated_ids))
        variants.append((f"rot{k}_fliplr", np.fliplr(rotated), np.fliplr(rotated_ids)))
    return variants


def _apply_grid_dihedral(grid: np.ndarray, name: str) -> np.ndarray:
    if name == "rot0":
        return grid.copy()
    if name == "rot1":
        return np.rot90(grid, 1).copy()
    if name == "rot2":
        return np.rot90(grid, 2).copy()
    if name == "rot3":
        return np.rot90(grid, 3).copy()
    if name == "rot0_fliplr":
        return np.fliplr(grid).copy()
    if name == "rot1_fliplr":
        return np.fliplr(np.rot90(grid, 1)).copy()
    if name == "rot2_fliplr":
        return np.fliplr(np.rot90(grid, 2)).copy()
    if name == "rot3_fliplr":
        return np.fliplr(np.rot90(grid, 3)).copy()
    raise ValueError(f"unknown dihedral transform: {name}")


def _unique_variants(template: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    unique: list[tuple[np.ndarray, np.ndarray]] = []
    for _name, sub, ids in _dihedral_variants(template):
        if any(prev.shape == sub.shape and np.array_equal(prev, sub) for prev, _ in unique):
            continue
        unique.append((sub.copy(), ids.copy()))
    return unique


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


def _parse_scene(g: np.ndarray) -> tuple[int, list[dict], dict[tuple[int, int], int]]:
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


def _orient_sprites(sprites_raw: list[dict]) -> list[dict]:
    variants = _unique_variants(sprites_raw[0]["sub"])
    sprites = []
    for sprite in sprites_raw:
        matches = [(sub, ids) for sub, ids in variants if sub.shape == sprite["sub"].shape and np.array_equal(sub, sprite["sub"])]
        assert len(matches) == 1, "each sprite must match one unique template orientation"
        _sub, ids = matches[0]
        direction, active = _active_side(sprite["sub"])
        sprites.append({**sprite, "ids": ids, "direction": direction, "active": active})
    return sprites


def _sprite_cell_map(sprites: list[dict]) -> dict[tuple[int, int], int]:
    cell_to_sprite: dict[tuple[int, int], int] = {}
    for idx, sprite in enumerate(sprites):
        r0, c0, r1, c1 = sprite["bbox"]
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                cell_to_sprite[(r, c)] = idx
    return cell_to_sprite


def _scan_marker_ray(
    H: int,
    W: int,
    markers: dict[tuple[int, int], int],
    cell_to_sprite: dict[tuple[int, int], int],
    r: int,
    c: int,
    dr: int,
    dc: int,
) -> tuple[tuple[int, int] | None, int | None, bool]:
    r += dr
    c += dc
    crossed_sprite = False
    while 0 <= r < H and 0 <= c < W:
        if (r, c) in markers:
            return (r, c), markers[(r, c)], crossed_sprite
        if (r, c) in cell_to_sprite:
            crossed_sprite = True
        r += dr
        c += dc
    return None, None, crossed_sprite


def _recover_port_labels(
    g: np.ndarray,
    sprites: list[dict],
    markers: dict[tuple[int, int], int],
    cell_to_sprite: dict[tuple[int, int], int],
) -> tuple[dict[int, int], set[tuple[int, int]]]:
    H, W = g.shape
    color_to_template_id: dict[int, int] = {}
    template_id_to_color: dict[int, int] = {}
    bridged_markers: set[tuple[int, int]] = set()

    for sprite in sprites:
        r0, c0, _r1, _c1 = sprite["bbox"]
        dr, dc = sprite["direction"]
        for lr, lc in sprite["active"]:
            marker_pos, marker_color, crossed_sprite = _scan_marker_ray(
                H,
                W,
                markers,
                cell_to_sprite,
                r0 + lr,
                c0 + lc,
                dr,
                dc,
            )
            if marker_color is None:
                continue
            template_id = int(sprite["ids"][lr, lc])
            if marker_color in color_to_template_id:
                assert color_to_template_id[marker_color] == template_id
            if template_id in template_id_to_color:
                assert template_id_to_color[template_id] == marker_color
            color_to_template_id[marker_color] = template_id
            template_id_to_color[template_id] = marker_color
            if crossed_sprite:
                assert marker_pos is not None
                bridged_markers.add(marker_pos)

    assert set(color_to_template_id) == set(markers.values()), "every control marker must label one active-edge cell"
    return color_to_template_id, bridged_markers


def _trace_attack_rays(
    g: np.ndarray,
    sprites: list[dict],
    color_to_template_id: dict[int, int],
    cell_to_sprite: dict[tuple[int, int], int],
) -> tuple[list[dict], set[int], int]:
    H, W = g.shape
    rays: list[dict] = []
    hit_sprites: set[int] = set()
    through_other_sprite_cells = 0

    for source_idx, sprite in enumerate(sprites):
        r0, c0, _r1, _c1 = sprite["bbox"]
        dr, dc = sprite["direction"]
        for color, template_id in color_to_template_id.items():
            locs = np.argwhere(sprite["ids"] == template_id)
            assert locs.shape[0] == 1
            lr, lc = map(int, locs[0])
            assert (lr, lc) in sprite["active"], "mapped port must land on the sprite active edge"

            path: list[tuple[int, int]] = []
            r = r0 + lr + dr
            c = c0 + lc + dc
            while 0 <= r < H and 0 <= c < W:
                path.append((r, c))
                target_idx = cell_to_sprite.get((r, c))
                if target_idx is not None and target_idx != source_idx:
                    hit_sprites.add(target_idx)
                    through_other_sprite_cells += 1
                r += dr
                c += dc
            rays.append({"color": color, "path": path, "source_idx": source_idx})

    return rays, hit_sprites, through_other_sprite_cells


def _solve_and_analyze(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    bg, sprites_raw, markers = _parse_scene(g)
    sprites = _orient_sprites(sprites_raw)
    cell_to_sprite = _sprite_cell_map(sprites)
    color_to_template_id, bridged_markers = _recover_port_labels(g, sprites, markers, cell_to_sprite)
    rays, hit_sprites, through_other_sprite_cells = _trace_attack_rays(g, sprites, color_to_template_id, cell_to_sprite)

    out = g.copy()
    for idx in hit_sprites:
        r0, c0, r1, c1 = sprites[idx]["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = bg

    for ray in rays:
        color = int(ray["color"])
        for r, c in ray["path"]:
            if (r, c) in markers:
                assert int(out[r, c]) == color, "a ray may only terminate on its own marker color"
                continue

            target_idx = cell_to_sprite.get((r, c))
            if target_idx is not None:
                assert target_idx in hit_sprites, "any sprite crossed by a ray must be erased"

            assert int(out[r, c]) in (bg, color), "attack rays must not conflict"
            out[r, c] = color

    metrics = {
        "marker_count": len(markers),
        "bridged_marker_count": len(bridged_markers),
        "hit_sprite_count": len(hit_sprites),
        "through_other_sprite_cells": through_other_sprite_cells,
        "sprite_count": len(sprites),
    }
    return out, metrics


def fusion_metrics(grid: np.ndarray) -> dict:
    _, metrics = _solve_and_analyze(grid)
    return metrics


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read boundary port labels through intervening sprites, then fire attack rays that erase the sprites they cross.

    Input structure:
    - The majority color is background.
    - Several dense non-square rectangles are copies of one shared Nx2 template under dihedral transforms.
    - Single-pixel boundary markers label some active-edge ports, and those marker sight-lines may pass through other sprites before reaching the labeled copy.

    Rule:
    - For each sprite, find the active long edge by the stronger repeated-color majority.
    - Scan outward from every active-edge cell, ignoring sprite pixels until a boundary marker is found; that marker color labels the corresponding template port.
    - From every sprite's matching port, send that color straight outward to the boundary. Rays do not stop at sprites: any sprite touched by such a ray is erased from the output, and the ray continues through the cleared lane.

    Key insight:
    - The markers still describe shared template ports, but the fused behavior is now laser-like: sprites are transparent for detection, and being crossed by a beam means disappearing from the final scene.
    """
    out, _metrics = _solve_and_analyze(grid)
    return out


# --------------- generator ---------------

_BG = 8
_NON_BG_COLORS = [0, 1, 2, 3, 4, 5, 6, 7, 9]
_GRID_H_RANGE = (21, 29)
_GRID_W_RANGE = (27, 29)
_TEMPLATE_N_RANGE = (7, 9)
_TEMPLATE_COLOR_COUNT = (3, 4)
_ACTIVE_MAJORITY_RANGE = (5, 9)
_SPRITE_COUNT_RANGE = (3, 4)
_NUM_MARKERS = 3
_GEN_RETRIES = 300


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
    if len(template_colors) < 2:
        return None
    right_col = _build_column(N, template_colors, right_majority, right_dominant)

    if max(Counter(left_col).values()) <= max(Counter(right_col).values()):
        return None

    template = np.column_stack([left_col, right_col]).astype(int)
    if len(set(map(int, template.flat))) < num_colors:
        return None
    return template, np.arange(N * 2, dtype=int).reshape(N, 2), template_colors


def _directional_variants(
    template: np.ndarray,
    ids_base: np.ndarray,
) -> dict[tuple[int, int], list[tuple[np.ndarray, np.ndarray, tuple[int, int], list[tuple[int, int]]]]]:
    raw_template = _dihedral_variants(template)
    raw_ids = _dihedral_variants(ids_base)

    by_direction: dict[tuple[int, int], list[tuple[np.ndarray, np.ndarray, tuple[int, int], list[tuple[int, int]]]]] = {}
    seen: list[np.ndarray] = []
    for (_, sub, _), (_, _, ids_arr) in zip(raw_template, raw_ids):
        if any(prev.shape == sub.shape and np.array_equal(prev, sub) for prev in seen):
            continue
        try:
            direction, active_set = _active_side(sub)
        except AssertionError:
            continue
        seen.append(sub.copy())
        by_direction.setdefault(direction, []).append((sub, ids_arr, direction, sorted(active_set)))
    return by_direction


def _make_sprite(
    r0: int,
    c0: int,
    sub: np.ndarray,
    ids_arr: np.ndarray,
    direction: tuple[int, int],
    active_coords: list[tuple[int, int]],
) -> dict:
    h, w = sub.shape
    return {
        "bbox": (r0, c0, r0 + h - 1, c0 + w - 1),
        "sub": sub,
        "ids": ids_arr,
        "direction": direction,
        "active": set(active_coords),
    }


def _place_attack_layout(
    H: int,
    W: int,
    N: int,
    left_variant: tuple[np.ndarray, np.ndarray, tuple[int, int], list[tuple[int, int]]],
    num_sprites: int,
) -> list[dict] | None:
    sub, ids_arr, direction, active_coords = left_variant
    h, w = sub.shape
    assert direction == (0, -1), "generator expects left-facing vertical sprites"

    upper_top = random.randint(1, 2)
    lower_top = H - h - random.randint(1, 2)
    if lower_top - upper_top < h + 3:
        return None

    blocker_top_c = random.randint(5, 7)
    emitter_top_c = W - w - random.randint(2, 3)
    emitter_bottom_c = W - w - random.randint(2, 4)
    if emitter_top_c - blocker_top_c < 10 or emitter_bottom_c - blocker_top_c < 10:
        return None

    placements = [
        _make_sprite(upper_top, blocker_top_c, sub, ids_arr, direction, active_coords),
        _make_sprite(upper_top, emitter_top_c, sub, ids_arr, direction, active_coords),
        _make_sprite(lower_top, emitter_bottom_c, sub, ids_arr, direction, active_coords),
    ]
    if num_sprites == 4:
        blocker_bottom_c = random.randint(5, 7)
        if emitter_bottom_c - blocker_bottom_c < 10:
            return None
        placements.append(_make_sprite(lower_top, blocker_bottom_c, sub, ids_arr, direction, active_coords))
    return placements


def _paint_sprites(H: int, W: int, placements: list[dict]) -> np.ndarray | None:
    grid = np.full((H, W), _BG, dtype=int)
    for sprite in placements:
        r0, c0, r1, c1 = sprite["bbox"]
        if r0 < 0 or c0 < 0 or r1 >= H or c1 >= W:
            return None
        if np.any(grid[r0 : r1 + 1, c0 : c1 + 1] != _BG):
            return None
        grid[r0 : r1 + 1, c0 : c1 + 1] = sprite["sub"]
    return grid


def _boundary_position(r: int, c: int, dr: int, dc: int, H: int, W: int) -> tuple[int, int]:
    if dr == 0 and dc == -1:
        return r, 0
    if dr == 0 and dc == 1:
        return r, W - 1
    if dr == -1 and dc == 0:
        return 0, c
    return H - 1, c


def _place_markers(grid: np.ndarray, placements: list[dict], template_colors: list[int]) -> bool:
    emitter = placements[1]
    H, W = grid.shape

    active_template_ids = [int(emitter["ids"][r, c]) for r, c in sorted(emitter["active"])]
    if len(active_template_ids) < _NUM_MARKERS:
        return False
    labeled_ids = random.sample(active_template_ids, _NUM_MARKERS)

    available = [c for c in _NON_BG_COLORS if c not in template_colors]
    if len(available) < _NUM_MARKERS:
        return False
    marker_colors = random.sample(available, _NUM_MARKERS)

    used_boundary_cells: set[tuple[int, int]] = set()
    r0, c0, _r1, _c1 = emitter["bbox"]
    dr, dc = emitter["direction"]
    for color, tid in zip(marker_colors, labeled_ids):
        locs = np.argwhere(emitter["ids"] == tid)
        if locs.shape[0] != 1:
            return False
        lr, lc = map(int, locs[0])
        if (lr, lc) not in emitter["active"]:
            return False
        mr, mc = _boundary_position(r0 + lr, c0 + lc, dr, dc, H, W)
        if (mr, mc) in used_boundary_cells or int(grid[mr, mc]) != _BG:
            return False
        grid[mr, mc] = color
        used_boundary_cells.add((mr, mc))
    return True


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*_GRID_H_RANGE)
    W = random.randint(*_GRID_W_RANGE)
    N = random.randint(*_TEMPLATE_N_RANGE)
    num_sprites = random.randint(*_SPRITE_COUNT_RANGE)

    design = _design_template(N)
    if design is None:
        return None
    template, ids_base, template_colors = design

    by_direction = _directional_variants(template, ids_base)
    left_choices = by_direction.get((0, -1), [])
    if not left_choices:
        return None
    left_variant = random.choice(left_choices)

    placements = _place_attack_layout(H, W, N, left_variant, num_sprites)
    if placements is None:
        return None

    grid = _paint_sprites(H, W, placements)
    if grid is None:
        return None
    if not _place_markers(grid, placements, template_colors):
        return None

    transform_names = ["rot0", "rot2", "rot0_fliplr", "rot2_fliplr"]
    if _GRID_W_RANGE[0] <= H <= _GRID_W_RANGE[1]:
        transform_names += ["rot1", "rot3", "rot1_fliplr", "rot3_fliplr"]
    transform_name = random.choice(transform_names)
    grid = _apply_grid_dihedral(grid, transform_name)

    try:
        _out, metrics = _solve_and_analyze(grid)
    except Exception:
        return None

    if metrics["bridged_marker_count"] < 1:
        return None
    if metrics["hit_sprite_count"] < 1:
        return None
    if metrics["through_other_sprite_cells"] < 1:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
