# Combo variant of 67e490f4 (index 18)
# combo_spec: D1/0+D3/2
# Name: 颜色标记配对+距离消歧

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np
from scipy import ndimage

from arc_utils import bbox_coords, components_mask, crop_bbox, find_components, get_bg, paste


# Design analysis for the fused variant:
# - Concrete behavior:
#   Original 67e490f4 fills each hole by finding an outside sprite with the same shape.
#   This variant removes that geometric key. Each hole stays as a bg component inside the
#   frame, but one adjacent frame pixel is recolored into a tag. Outside the frame, every
#   sprite is a tiny 2x2 multicolor object with one tag pixel and three body-color pixels.
#   The hole pairs to sprites by tag color, then chooses the nearest sprite among those with
#   the same tag; the sprite body color is written into the hole.
# - Why this is not a trivial simplification:
#   If the tag color itself were the fill color, distance would do nothing. To keep both
#   dimensions active, the tag is only a key, while the body color carries the answer.
#   Each tag appears on at least two sprites with different body colors, so distance really
#   changes the output. Multiple different hole shapes also reuse the same tag, so shape
#   can no longer explain the mapping.
# - Assumption audit against the original solver:
#   1. `sub != frame_col` no longer isolates holes, because marker pixels also differ from
#      the frame color. The new solver therefore extracts holes as `sub == bg`.
#   2. Outside components are no longer monochrome. The new solver treats each sprite as a
#      multicolor component and decodes `least frequent color -> tag`, `most frequent color
#      -> fill/body`.
#   3. Shape keys are no longer meaningful. Matching is done by marker color and distance.
#   4. The frame detector still assumes the frame-colored region remains the unique largest
#      monochrome non-bg component. The generator guarantees this by placing only one marker
#      per hole and keeping the global frame dense and 8-connected.
# - Placement legality:
#   Hole markers are chosen from frame cells that are adjacent to exactly one hole, so a
#   marker cannot ambiguously belong to two holes. Sprites are placed fully outside the
#   frame bbox with one-cell Chebyshev padding and with a strict distance gap between the
#   nearest and second-nearest sprite of the same tag.
# - Visual observability:
#   Humans can infer the rule from a few examples because the repeated cue is explicit:
#   same edge tag -> look for same tagged outside sprites -> pick the nearest one -> use its
#   other color. The generator reuses tags across distinct hole shapes to make it visible
#   that shape is no longer the matching key.


def _all_rot_flip(mask: np.ndarray):
    m0 = mask.astype(bool)
    for flip in (False, True):
        m1 = np.flipud(m0) if flip else m0
        for k in range(4):
            m2 = np.rot90(m1, k)
            yield (m2.shape, m2.tobytes()), m2


def _canonical_key(mask: np.ndarray):
    return min(key for key, _ in _all_rot_flip(mask))


def _point_rect_dist2(r: int, c: int, bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = map(int, bbox)
    if r < r0:
        dr = int(r0 - r)
    elif r > r1:
        dr = int(r - r1)
    else:
        dr = 0
    if c < c0:
        dc = int(c0 - c)
    elif c > c1:
        dc = int(c - c1)
    else:
        dc = 0
    return int(dr * dr + dc * dc)


def _comp_dist2(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> int:
    return min(_point_rect_dist2(int(r), int(c), bbox) for r, c in coords)


def _frame_and_subgrid(g: np.ndarray, bg: int):
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    assert comps
    frame_area = max(int(comp["area"]) for comp in comps)
    frame_candidates = [comp for comp in comps if int(comp["area"]) == frame_area]
    assert len(frame_candidates) == 1
    frame = frame_candidates[0]
    frame_col = int(frame["color"])
    fb = tuple(map(int, frame["bbox"]))
    return frame_col, fb, crop_bbox(g, fb)


def _extract_holes_and_tags(sub: np.ndarray, frame_col: int, bg: int):
    th, tw = sub.shape
    hole_mask = sub == int(bg)
    holes = []
    for coords in components_mask(hole_mask, connectivity=4):
        hr0, hc0, hr1, hc1 = bbox_coords(coords)
        rel = coords - np.array([hr0, hc0])
        mask = np.zeros((int(hr1 - hr0 + 1), int(hc1 - hc0 + 1)), dtype=bool)
        mask[rel[:, 0], rel[:, 1]] = True

        tag_colors = set()
        for rr, cc in coords:
            for nr, nc in ((int(rr) - 1, int(cc)), (int(rr) + 1, int(cc)), (int(rr), int(cc) - 1), (int(rr), int(cc) + 1)):
                if 0 <= nr < th and 0 <= nc < tw:
                    val = int(sub[int(nr), int(nc)])
                    if val != int(frame_col) and val != int(bg):
                        tag_colors.add(val)
        assert len(tag_colors) == 1
        holes.append(
            {
                "origin": (int(hr0), int(hc0)),
                "mask": mask,
                "tag": int(next(iter(tag_colors))),
            }
        )
    return holes


def _extract_sprites(g: np.ndarray, bg: int, frame_bbox: tuple[int, int, int, int]):
    r0, c0, r1, c1 = map(int, frame_bbox)
    outside = g.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
    comps = find_components(outside, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    sprites = []
    for comp in comps:
        coords = comp["coords"]
        vals = [int(v) for v in g[coords[:, 0], coords[:, 1]] if int(v) != int(bg)]
        counts = Counter(vals)
        assert len(counts) >= 2
        body_col, _ = max(counts.items(), key=lambda kv: (kv[1], -kv[0]))
        tag_col, _ = min(counts.items(), key=lambda kv: (kv[1], kv[0]))
        assert body_col != tag_col
        sprites.append(
            {
                "tag": int(tag_col),
                "body": int(body_col),
                "dist2": _comp_dist2(coords, frame_bbox),
            }
        )
    return sprites


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each hole inside a large frame by reading a nearby tagged sprite outside it.

    Input structure:
    - bg is the dominant color. One large monochrome frame encloses several bg holes.
    - Each hole has a single colored marker pixel embedded on its bordering frame edge.
    - Outside the frame are small multicolor sprites; each sprite contains one marker-colored
      tag pixel plus a larger body region of another color.

    Rule:
    - Crop to the frame bounding box and start from a solid frame-colored canvas.
    - For every hole, read its adjacent marker color and collect outside sprites with the
      same tag color.
    - If several sprites share that tag, choose the one with the smallest Euclidean distance
      to the frame boundary, then fill the hole with that sprite's body color.

    Key insight: The edge marker tells you which sprite group to inspect, but not the answer
    color itself; the answer comes from the nearest sprite carrying that same marker.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    frame_col, frame_bbox, sub = _frame_and_subgrid(g, bg)
    holes = _extract_holes_and_tags(sub, frame_col, bg)
    sprites = _extract_sprites(g, bg, frame_bbox)

    by_tag: dict[int, list[dict]] = defaultdict(list)
    for spr in sprites:
        by_tag[int(spr["tag"])].append(spr)

    out = np.full(sub.shape, int(frame_col), dtype=int)
    for hole in holes:
        candidates = by_tag.get(int(hole["tag"]))
        assert candidates
        best = min(candidates, key=lambda d: (int(d["dist2"]), int(d["body"])))
        patch = np.full(hole["mask"].shape, int(best["body"]), dtype=int)
        hr0, hc0 = hole["origin"]
        paste(out, patch, int(hr0), int(hc0), mask=hole["mask"])

    return out


# --------------- generator ---------------

NUM_COLORS = 10
GRID_SIZE = 30

FRAME_H_RANGE = (8, 13)
FRAME_W_RANGE = (8, 18)
FRAME_MARGIN = 3

SHAPE_COUNT_RANGE = (3, 6)
MIN_SHAPES = 3
SHAPE_MAX_AREA = 8
FRAME_DENSITY_RANGE = (0.58, 0.90)
MIN_TOTAL_HOLES = 5
PLACEMENT_RETRIES = 80
MULT_EVEN_WEIGHTS = ([1, 2], [4, 1])
MULT_ODD_WEIGHTS = ([1, 2, 3], [3, 2, 1])

TAG_COUNT_RANGE = (2, 2)
SPRITE_PADDING = 1
MAX_GEN_ATTEMPTS = 400


def _build_shape_library():
    raw = [
        [[1]],
        [[1, 1]],
        [[1], [1]],
        [[1, 1, 1]],
        [[1, 1], [1, 1]],
        [[1, 1, 1], [1, 1, 1]],
        [[1, 0], [1, 1]],
        [[1, 0], [1, 0], [1, 1]],
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
        [[1, 1, 1], [0, 1, 0], [0, 1, 0]],
        [[1, 1, 1, 1]],
        [[1, 1], [1, 0]],
        [[1, 1, 1, 1, 1]],
        [[1, 0], [1, 1], [0, 1]],
        [[1, 1, 1], [1, 0, 0]],
        [[1, 1], [0, 1], [0, 1]],
    ]
    seen, dedup = set(), []
    for s in raw:
        m = np.array(s, dtype=bool)
        k = _canonical_key(m)
        if k not in seen:
            seen.add(k)
            dedup.append(m)
    return dedup


SHAPE_LIBRARY = _build_shape_library()


def _mirror_pos(r, c, sh, sw, th, tw):
    return list(
        {
            (r, c),
            (r, tw - 1 - c - sw + 1),
            (th - 1 - r - sh + 1, c),
            (th - 1 - r - sh + 1, tw - 1 - c - sw + 1),
        }
    )


def _validate_positions(occ, shape, positions, th, tw):
    sh, sw = shape.shape
    new_cells = set()
    for pr, pc in positions:
        for dr in range(sh):
            for dc in range(sw):
                if shape[dr, dc]:
                    new_cells.add((pr + dr, pc + dc))

    for pr, pc in positions:
        if pr < 1 or pc < 1 or pr + sh > th - 1 or pc + sw > tw - 1:
            return None
        for dr in range(sh):
            for dc in range(sw):
                if not shape[dr, dc]:
                    continue
                rr, cc = pr + dr, pc + dc
                if not occ[rr, cc]:
                    return None
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < th and 0 <= nc < tw and not occ[nr, nc]:
                        if (nr, nc) not in new_cells:
                            return None
    return positions


def _try_place_quadrant(occ, shape, th, tw):
    sh, sw = shape.shape
    r_max, c_max = th // 2 - sh, tw // 2 - sw
    if r_max < 1 or c_max < 1:
        return None
    r, c = random.randint(1, r_max), random.randint(1, c_max)
    return _validate_positions(occ, shape, _mirror_pos(r, c, sh, sw, th, tw), th, tw)


def _try_place_hcenter(occ, shape, th, tw):
    if th % 2 == 0:
        return None
    sh, sw = shape.shape
    r = th // 2 - sh // 2
    if r < 1 or r + sh > th - 1:
        return None
    c_max = tw // 2 - sw
    if c_max < 1:
        return None
    c = random.randint(1, c_max)
    c_m = tw - 1 - c - sw + 1
    return _validate_positions(occ, shape, list({(r, c), (r, c_m)}), th, tw)


def _try_place_vcenter(occ, shape, th, tw):
    if tw % 2 == 0:
        return None
    sh, sw = shape.shape
    c = tw // 2 - sw // 2
    if c < 1 or c + sw > tw - 1:
        return None
    r_max = th // 2 - sh
    if r_max < 1:
        return None
    r = random.randint(1, r_max)
    r_m = th - 1 - r - sh + 1
    return _validate_positions(occ, shape, list({(r, c), (r_m, c)}), th, tw)


def _try_place_center(occ, shape, th, tw):
    if th % 2 == 0 or tw % 2 == 0:
        return None
    sh, sw = shape.shape
    r, c = th // 2 - sh // 2, tw // 2 - sw // 2
    if r < 1 or c < 1 or r + sh > th - 1 or c + sw > tw - 1:
        return None
    return _validate_positions(occ, shape, [(r, c)], th, tw)


def _try_place_symmetric(occ, shape, th, tw):
    placers = [_try_place_quadrant]
    if th % 2 == 1:
        placers.append(_try_place_hcenter)
    if tw % 2 == 1:
        placers.append(_try_place_vcenter)
    if th % 2 == 1 and tw % 2 == 1:
        placers.append(_try_place_center)
    return random.choice(placers)(occ, shape, th, tw)


def _carve_hole(occ, shape, positions):
    sh, sw = shape.shape
    for pr, pc in positions:
        for dr in range(sh):
            for dc in range(sw):
                if shape[dr, dc]:
                    occ[pr + dr, pc + dc] = False


def _build_hole_layout(th, tw):
    max_sh, max_sw = max(1, (th - 2) // 2), max(1, (tw - 2) // 2)
    available = [
        s
        for s in SHAPE_LIBRARY
        if s.shape[0] <= max_sh and s.shape[1] <= max_sw and int(np.sum(s)) <= SHAPE_MAX_AREA
    ]
    if len(available) < MIN_SHAPES:
        return None

    k_target = random.randint(SHAPE_COUNT_RANGE[0], min(SHAPE_COUNT_RANGE[1], len(available)))
    random.shuffle(available)
    even_dims = th % 2 == 0 or tw % 2 == 0
    if even_dims:
        available.sort(key=lambda s: int(np.sum(s)))

    occ = np.ones((th, tw), dtype=bool)
    hole_groups = []

    mult_vals, mult_wts = MULT_EVEN_WEIGHTS if even_dims else MULT_ODD_WEIGHTS
    for shape in available:
        if len(hole_groups) >= k_target:
            break
        mult = random.choices(mult_vals, weights=mult_wts)[0]
        placed = []
        for _ in range(mult):
            for _ in range(PLACEMENT_RETRIES):
                positions = _try_place_symmetric(occ, shape, th, tw)
                if positions is not None:
                    _carve_hole(occ, shape, positions)
                    placed.append(positions)
                    break
        if placed:
            flat_positions = [tuple(map(int, pos)) for positions in placed for pos in positions]
            hole_groups.append({"shape": shape, "positions": flat_positions})

    if len(hole_groups) < MIN_SHAPES:
        return None

    frame_density = 1.0 - int(np.sum(~occ)) / float(th * tw)
    if not (FRAME_DENSITY_RANGE[0] <= frame_density <= FRAME_DENSITY_RANGE[1]):
        return None

    total_holes = sum(len(group["positions"]) for group in hole_groups)
    if total_holes < MIN_TOTAL_HOLES:
        return None

    return occ, hole_groups


def _assign_tags_and_colors(hole_groups, bg: int, frame_col: int):
    palette = [c for c in range(NUM_COLORS) if c != bg and c != frame_col]
    n_groups = len(hole_groups)
    # Every tag should be reusable on multiple hole groups; otherwise a singleton tag
    # weakens the visible "color marker as pairing key" pattern for human solvers.
    max_tags = min(TAG_COUNT_RANGE[1], n_groups // 2, len(palette) // 2)
    min_tags = min(TAG_COUNT_RANGE[0], max_tags)
    if max_tags < 2:
        return None
    n_tags = 2
    tag_colors = random.sample(palette, n_tags)
    body_pool = [c for c in palette if c not in tag_colors]
    if len(body_pool) < n_tags + 1:
        return None

    winner_colors = random.sample(body_pool, n_tags)
    tag_to_winner = {int(tag): int(win) for tag, win in zip(tag_colors, winner_colors)}
    tag_to_loser = {}
    for tag, win in tag_to_winner.items():
        losers = [c for c in body_pool if c != win]
        tag_to_loser[int(tag)] = int(random.choice(losers))

    shape_keys = [_canonical_key(group["shape"]) for group in hole_groups]
    base_counts = [2] * n_tags
    for _ in range(n_groups - 2 * n_tags):
        base_counts[random.randrange(n_tags)] += 1

    assigned = False
    for _ in range(40):
        group_order = list(range(n_groups))
        random.shuffle(group_order)
        tag_order = list(range(n_tags))
        random.shuffle(tag_order)
        assignments = []
        for idx in tag_order:
            assignments.extend([tag_colors[idx]] * base_counts[idx])
        random.shuffle(assignments)

        shapes_by_tag = defaultdict(set)
        for gid, tag in zip(group_order, assignments):
            shapes_by_tag[int(tag)].add(shape_keys[int(gid)])
        if all(len(v) >= 2 for v in shapes_by_tag.values()):
            for gid, tag in zip(group_order, assignments):
                hole_groups[gid]["tag"] = int(tag)
                hole_groups[gid]["fill"] = int(tag_to_winner[int(tag)])
            assigned = True
            break

    if not assigned:
        return None

    return tag_to_winner, tag_to_loser


def _choose_markers(occ: np.ndarray, hole_groups):
    th, tw = occ.shape
    hole_id = np.full((th, tw), -1, dtype=int)
    instances = []

    hid = 0
    for group in hole_groups:
        shape = group["shape"]
        for pr, pc in group["positions"]:
            cells = []
            for dr in range(shape.shape[0]):
                for dc in range(shape.shape[1]):
                    if shape[dr, dc]:
                        rr, cc = int(pr + dr), int(pc + dc)
                        hole_id[rr, cc] = hid
                        cells.append((rr, cc))
            instances.append({"hid": hid, "tag": int(group["tag"]), "cells": cells})
            hid += 1

    def marker_candidates(inst, used):
        out = []
        hid_local = int(inst["hid"])
        for rr, cc in inst["cells"]:
            for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                if not (0 <= nr < th and 0 <= nc < tw):
                    continue
                if not bool(occ[nr, nc]):
                    continue
                if (nr, nc) in used:
                    continue
                adj_ids = set()
                for ar, ac in ((nr - 1, nc), (nr + 1, nc), (nr, nc - 1), (nr, nc + 1)):
                    if 0 <= ar < th and 0 <= ac < tw and int(hole_id[ar, ac]) >= 0:
                        adj_ids.add(int(hole_id[ar, ac]))
                if adj_ids != {hid_local}:
                    continue
                border_penalty = 1 if nr in (0, th - 1) or nc in (0, tw - 1) else 0
                out.append((border_penalty, (int(nr), int(nc))))
        out.sort(key=lambda item: item[0])
        dedup = []
        seen = set()
        for item in out:
            if item[1] not in seen:
                seen.add(item[1])
                dedup.append(item)
        return dedup

    chosen = {}
    used = set()
    remaining = instances[:]
    while remaining:
        options = []
        for inst in remaining:
            cands = marker_candidates(inst, used)
            if not cands:
                return None
            options.append((len(cands), cands, inst))
        options.sort(key=lambda item: item[0])
        _, cands, inst = options[0]
        best_penalty = cands[0][0]
        best_cells = [cell for penalty, cell in cands if penalty == best_penalty]
        cell = random.choice(best_cells)
        chosen[int(inst["hid"])] = (cell, int(inst["tag"]))
        used.add(cell)
        remaining = [x for x in remaining if int(x["hid"]) != int(inst["hid"])]

    return chosen


def _make_sprite_patch(tag: int, body: int) -> np.ndarray:
    patch = np.array([[int(tag), int(body)], [int(body), int(body)]], dtype=int)
    return np.rot90(patch, random.randint(0, 3))


def _rectangles_overlap(a, b):
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _enumerate_sprite_positions(placement: np.ndarray, patch: np.ndarray, bg: int, frame_bbox):
    H, W = placement.shape
    h, w = patch.shape
    frame_r0, frame_c0, frame_r1, frame_c1 = map(int, frame_bbox)
    occ = placement != int(bg)
    padded = ndimage.binary_dilation(occ, structure=np.ones((3, 3), dtype=bool), iterations=SPRITE_PADDING)
    cells = np.argwhere(patch != int(bg))
    result = []

    for r0 in range(0, H - h + 1):
        for c0 in range(0, W - w + 1):
            sprite_bbox = (int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1))
            if _rectangles_overlap(sprite_bbox, (frame_r0, frame_c0, frame_r1, frame_c1)):
                continue
            window = padded[r0 : r0 + h, c0 : c0 + w]
            if bool((window & (patch != int(bg))).any()):
                continue
            coords = cells + np.array([r0, c0])
            dist2 = _comp_dist2(coords, frame_bbox)
            result.append((int(dist2), int(r0), int(c0)))
    return result


def _place_one_sprite(inp: np.ndarray, placement: np.ndarray, frame_bbox, tag: int, body: int, bg: int, *, max_d2=None, min_d2=None):
    patch = _make_sprite_patch(tag, body)
    candidates = _enumerate_sprite_positions(placement, patch, bg, frame_bbox)
    if max_d2 is not None:
        candidates = [cand for cand in candidates if int(cand[0]) <= int(max_d2)]
    if min_d2 is not None:
        candidates = [cand for cand in candidates if int(cand[0]) >= int(min_d2)]
    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])
    if max_d2 is not None:
        pool = candidates
    elif min_d2 is None:
        best = candidates[0][0]
        pool = [cand for cand in candidates if int(cand[0]) <= int(best + 5)]
    else:
        pool = candidates[: min(25, len(candidates))]

    dist2, r0, c0 = random.choice(pool)
    paste(placement, patch, int(r0), int(c0))
    paste(inp, patch, int(r0), int(c0))
    return int(dist2)


def _render_output(occ: np.ndarray, hole_groups, frame_col: int):
    th, tw = occ.shape
    output = np.full((th, tw), int(frame_col), dtype=int)
    for group in hole_groups:
        shape = group["shape"]
        fill = int(group["fill"])
        patch = np.full(shape.shape, int(fill), dtype=int)
        for pr, pc in group["positions"]:
            paste(output, patch, int(pr), int(pc), mask=shape)
    return output


def _build_input_canvas(occ: np.ndarray, hole_groups, marker_map, frame_col: int, bg: int):
    th, tw = occ.shape
    frame = np.where(occ, int(frame_col), int(bg)).astype(int)
    hid = 0
    for group in hole_groups:
        for _ in group["positions"]:
            (mr, mc), tag = marker_map[int(hid)]
            frame[int(mr), int(mc)] = int(tag)
            hid += 1
    return frame


def _scatter_sprites(inp: np.ndarray, frame_bbox, tag_to_winner, tag_to_loser, bg: int):
    placement = inp.copy()
    sprite_info = {}
    for tag, winner in tag_to_winner.items():
        win_d2 = _place_one_sprite(inp, placement, frame_bbox, int(tag), int(winner), bg, max_d2=16)
        if win_d2 is None:
            return None
        lose_d2 = _place_one_sprite(inp, placement, frame_bbox, int(tag), int(tag_to_loser[int(tag)]), bg, min_d2=int(win_d2 + 5))
        if lose_d2 is None:
            return None
        sprite_info[int(tag)] = {
            "winner_dist2": int(win_d2),
            "loser_dist2": int(lose_d2),
        }
        if random.random() < 0.4:
            extra_choices = [c for c in tag_to_winner.values() if int(c) != int(winner)]
            if extra_choices:
                extra_body = int(random.choice(extra_choices))
                extra_d2 = _place_one_sprite(inp, placement, frame_bbox, int(tag), extra_body, bg, min_d2=int(lose_d2 + 4))
                if extra_d2 is not None:
                    sprite_info[int(tag)]["extra_dist2"] = int(extra_d2)
    return sprite_info


def _gen_attempt() -> np.ndarray | None:
    bg = random.randint(0, NUM_COLORS - 1)
    frame_col = random.choice([c for c in range(NUM_COLORS) if c != bg])
    th = random.randint(*FRAME_H_RANGE)
    tw = random.randint(*FRAME_W_RANGE)

    layout = _build_hole_layout(th, tw)
    if layout is None:
        return None
    occ, hole_groups = layout

    colors = _assign_tags_and_colors(hole_groups, bg, frame_col)
    if colors is None:
        return None
    tag_to_winner, tag_to_loser = colors

    marker_map = _choose_markers(occ, hole_groups)
    if marker_map is None:
        return None

    output = _render_output(occ, hole_groups, frame_col)

    if GRID_SIZE - th - FRAME_MARGIN < FRAME_MARGIN or GRID_SIZE - tw - FRAME_MARGIN < FRAME_MARGIN:
        return None
    fr = random.randint(FRAME_MARGIN, GRID_SIZE - th - FRAME_MARGIN)
    fc = random.randint(FRAME_MARGIN, GRID_SIZE - tw - FRAME_MARGIN)
    frame_bbox = (int(fr), int(fc), int(fr + th - 1), int(fc + tw - 1))

    inp = np.full((GRID_SIZE, GRID_SIZE), int(bg), dtype=int)
    frame_patch = _build_input_canvas(occ, hole_groups, marker_map, frame_col, bg)
    paste(inp, frame_patch, int(fr), int(fc))

    if _scatter_sprites(inp, frame_bbox, tag_to_winner, tag_to_loser, bg) is None:
        return None

    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {MAX_GEN_ATTEMPTS} attempts")
