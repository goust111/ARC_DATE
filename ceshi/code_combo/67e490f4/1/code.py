# Combo variant of 67e490f4 (index 1)
# combo_spec: D2/0+D3/2
# Name: 严格方向+距离消歧
#
# Design analysis for this combo
# - Source-task baseline:
#   The original task used one large frame with background-colored holes and outside sprites as
#   shape-coded color keys. Matching was tolerant to rotation/flip, and when several colors matched
#   the same hole family, the plurality color won.
# - Concrete fused behavior:
#   This combo changes both parts of that lookup. A hole is keyed by its exact bitmap as drawn, so a
#   rotated or flipped cousin is a different key. Among exact-match sprites, color is chosen from the
#   sprite that sits closest to the frame bbox, measured by minimum Euclidean distance from any sprite
#   cell to the frame rectangle.
# - Small trace:
#   Suppose the frame contains two L-shaped holes:
#   `A = [[1,0],[1,1]]` and `B = [[1,1],[1,0]]`.
#   Original rule: sprites of A and B are merged into one rotation family, then the most frequent
#   color across that family fills both holes.
#   Combo rule: A only looks at A-sprites and B only looks at B-sprites. If A has one red sprite near
#   the frame and three teal A-sprites farther away, A becomes red despite teal being more frequent.
#   A nearby B-sprite still does not count for A because its facing is wrong.
# - Complexity calibration:
#   If the generator used only symmetric shapes or let the nearest exact-match color also be the most
#   frequent color, the combo would visually collapse back toward the source task. To keep both edits
#   observable, every sample contains:
#   1) at least one canonical family shown in two different orientations with different output colors;
#   2) exact-match color conflicts where the nearest sprite color differs from the majority color.
# - Assumption audit against the old `main()`:
#   Step 1 (largest non-bg monochrome component is the frame) still holds.
#   Step 2 already extracts exact hole masks, so it can stay exact-orientation based.
#   Step 3 no longer expands each sprite into 8 transformed keys; doing so would violate D2/0.
#   Step 4 can no longer vote by color frequency; it must instead keep the nearest sprite per exact
#   key. The generator must therefore guarantee at least one exact-match sprite per hole orientation
#   and avoid ambiguous equal-distance ties across colors.
# - Placement legality:
#   Holes remain strictly inside the frame border and 4-disconnected. Sprites are fully outside the
#   frame bbox and Chebyshev-separated from the frame and from each other. The custom placement search
#   only accepts legal top-left positions before pasting a sprite.
# - Visual observability:
#   The paired orientation family makes "direction matters" visible because two rotated look-alikes
#   produce different hole colors. Distance matters because each oriented group has a close winner and
#   farther exact-match distractors in another color, so the output cannot be explained by counting.
# - Visual cleanliness:
#   The generator uses 3-4 oriented groups, monochrome sprites, small asymmetric masks, and shared
#   distractor colors outside the output palette. This keeps the frame interior legible while making
#   the near-vs-far contrast easy to inspect around the frame.
# - New implicit assumptions introduced here:
#   The solver assumes the nearest exact-match sprite color is unique after distance comparison, with
#   color/top-left fallbacks only as robustness. The generator enforces a visible distance margin so
#   those fallbacks are rarely relevant and never needed for the intended reasoning path.

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import (
    bbox_coords,
    components_mask,
    crop_bbox,
    find_components,
    get_bg,
    paste,
)


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    m = mask.astype(bool)
    return (m.shape, m.tobytes())


def _all_rot_flip(mask: np.ndarray):
    """Yield (key, oriented_mask) for all 8 rotation/flip variants of a bool mask."""
    m0 = mask.astype(bool)
    for flip in (False, True):
        m1 = np.flipud(m0) if flip else m0
        for k in range(4):
            m2 = np.rot90(m1, k)
            yield _mask_key(m2), m2


def _canonical_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return min(key for key, _ in _all_rot_flip(mask))


def _all_orientations(mask: np.ndarray) -> list[np.ndarray]:
    seen: set[tuple[tuple[int, int], bytes]] = set()
    result: list[np.ndarray] = []
    for key, oriented in _all_rot_flip(mask):
        if key not in seen:
            seen.add(key)
            result.append(oriented)
    return result


def _distance_coords_to_frame(coords: np.ndarray, frame_bbox: tuple[int, int, int, int]) -> float:
    """Minimum Euclidean distance from component cells to the closed frame rectangle."""
    r0, c0, r1, c1 = map(int, frame_bbox)
    rows = coords[:, 0]
    cols = coords[:, 1]
    dr = np.where(rows < r0, r0 - rows, np.where(rows > r1, rows - r1, 0))
    dc = np.where(cols < c0, c0 - cols, np.where(cols > c1, cols - c1, 0))
    return float(np.min(np.hypot(dr, dc)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the holes inside a frame by reading strictly oriented sprites around it.

    Input structure:
    - The background is the most common color, and one large monochrome object forms the frame.
    - Inside the frame bbox are several 4-connected holes cut out of the frame color.
    - Outside the frame bbox are small sprites on the background, usually monochrome and separated.

    Rule:
    - Crop to the frame bounding box and begin from a solid rectangle of the frame color.
    - A hole only matches sprites whose bitmap is identical with the same facing; no rotation or flip
      is allowed during matching.
    - If several exact-match sprites have different colors, use the color of the one nearest to the
      frame boundary by Euclidean distance.

    Key insight:
    - The outside sprites now encode two things at once: exact orientation decides whether a sprite is
      even eligible, and proximity to the frame decides which eligible color wins.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    # STEP 1) Find the template frame: largest connected monochrome non-bg component.
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    assert comps
    frame_area = max(int(comp["area"]) for comp in comps)
    frame_candidates = [comp for comp in comps if int(comp["area"]) == frame_area]
    assert len(frame_candidates) == 1
    frame = frame_candidates[0]
    frame_col = int(frame["color"])
    fb = frame["bbox"]
    r0, c0, r1, c1 = map(int, fb)

    sub = crop_bbox(g, fb)
    th, tw = sub.shape

    # STEP 2) Extract hole components inside the bbox, keyed by exact mask orientation.
    hole_mask = sub != int(frame_col)
    holes: list[tuple[tuple[tuple[int, int], bytes], tuple[int, int], np.ndarray]] = []
    for coords in components_mask(hole_mask, connectivity=4):
        hr0, hc0, hr1, hc1 = bbox_coords(coords)
        hh = int(hr1 - hr0 + 1)
        ww = int(hc1 - hc0 + 1)
        rel = coords - np.array([hr0, hc0])
        mask = np.zeros((hh, ww), dtype=bool)
        mask[rel[:, 0], rel[:, 1]] = True
        holes.append((_mask_key(mask), (int(hr0), int(hc0)), mask))

    # STEP 3) Extract outside sprites and retain the nearest sprite for each exact key.
    outside = g.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
    spr_comps = find_components(outside, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    best_sprite_by_key: dict[tuple[tuple[int, int], bytes], tuple[float, int, int, int]] = {}
    for comp in spr_comps:
        coords = comp["coords"]
        sr0, sc0, sr1, sc1 = map(int, comp["bbox"])
        hh = int(sr1 - sr0 + 1)
        ww = int(sc1 - sc0 + 1)

        vals = [int(v) for v in g[coords[:, 0], coords[:, 1]]]
        scol = int(Counter(vals).most_common(1)[0][0])

        rel = coords - np.array([sr0, sc0])
        mask = np.zeros((hh, ww), dtype=bool)
        mask[rel[:, 0], rel[:, 1]] = True
        key = _mask_key(mask)
        dist = _distance_coords_to_frame(coords, (r0, c0, r1, c1))
        candidate = (dist, int(scol), int(sr0), int(sc0))
        prev = best_sprite_by_key.get(key)
        if prev is None or candidate < prev:
            best_sprite_by_key[key] = candidate

    # STEP 4) Fill each hole from the closest exact-match sprite.
    out = np.full((th, tw), int(frame_col), dtype=int)
    for key, (hr0, hc0), hmask in holes:
        choice = best_sprite_by_key.get(key)
        assert choice is not None
        scol = int(choice[1])
        paste(out, np.full(hmask.shape, scol, dtype=int), int(hr0), int(hc0), mask=hmask)

    return out


# --------------- generator ---------------

NUM_COLORS = 10
GRID_SIZE = 30

FRAME_H_RANGE = (7, 13)
FRAME_W_RANGE = (7, 18)

SHAPE_COUNT_RANGE = (3, 4)
MIN_SHAPES = 3
SHAPE_MAX_AREA = 8
FRAME_DENSITY_RANGE = (0.55, 0.90)
MIN_TOTAL_HOLES = 5
PLACEMENT_RETRIES = 90
MULT_EVEN_WEIGHTS = ([1, 2], [4, 1])
MULT_ODD_WEIGHTS = ([1, 2, 3], [3, 2, 1])

LOSER_COUNT_RANGE = (2, 3)
DISTANCE_MARGIN = 2.0
NEAR_POOL_SIZE = 12
FAR_TAIL_FRACTION = 0.45
EXTRA_ALT_DECOYS = (0, 1)

MAX_GEN_ATTEMPTS = 600


def _build_shape_library() -> list[np.ndarray]:
    raw = [
        [[1, 0], [1, 1]],
        [[1, 0], [1, 0], [1, 1]],
        [[1, 1, 1], [1, 0, 0]],
        [[1, 1, 1], [0, 1, 0], [0, 1, 0]],
        [[1, 0], [1, 1], [0, 1]],
        [[1, 1, 0], [0, 1, 1]],
        [[1, 1, 1], [0, 0, 1]],
        [[1, 0, 0], [1, 1, 1], [0, 0, 1]],
        [[1, 1, 1], [1, 0, 1]],
        [[1, 1, 0], [1, 0, 0], [1, 1, 1]],
    ]
    seen: set[tuple[tuple[int, int], bytes]] = set()
    dedup: list[np.ndarray] = []
    for arr in raw:
        mask = np.array(arr, dtype=bool)
        key = _canonical_key(mask)
        if key not in seen:
            seen.add(key)
            if len(_all_orientations(mask)) > 1 and int(np.sum(mask)) <= SHAPE_MAX_AREA:
                dedup.append(mask)
    return dedup


SHAPE_LIBRARY = _build_shape_library()


def _mirror_pos(r: int, c: int, sh: int, sw: int, th: int, tw: int) -> list[tuple[int, int]]:
    return list(
        {
            (r, c),
            (r, tw - sw - c),
            (th - sh - r, c),
            (th - sh - r, tw - sw - c),
        }
    )


def _validate_positions(
    occ: np.ndarray,
    shape: np.ndarray,
    positions: list[tuple[int, int]],
    th: int,
    tw: int,
) -> list[tuple[int, int]] | None:
    sh, sw = shape.shape
    new_cells: set[tuple[int, int]] = set()
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
                    if 0 <= nr < th and 0 <= nc < tw and not occ[nr, nc] and (nr, nc) not in new_cells:
                        return None
    return positions


def _try_place_quadrant(occ: np.ndarray, shape: np.ndarray, th: int, tw: int):
    sh, sw = shape.shape
    r_max, c_max = th // 2 - sh, tw // 2 - sw
    if r_max < 1 or c_max < 1:
        return None
    r = random.randint(1, r_max)
    c = random.randint(1, c_max)
    return _validate_positions(occ, shape, _mirror_pos(r, c, sh, sw, th, tw), th, tw)


def _try_place_hcenter(occ: np.ndarray, shape: np.ndarray, th: int, tw: int):
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
    return _validate_positions(occ, shape, list({(r, c), (r, tw - sw - c)}), th, tw)


def _try_place_vcenter(occ: np.ndarray, shape: np.ndarray, th: int, tw: int):
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
    return _validate_positions(occ, shape, list({(r, c), (th - sh - r, c)}), th, tw)


def _try_place_center(occ: np.ndarray, shape: np.ndarray, th: int, tw: int):
    if th % 2 == 0 or tw % 2 == 0:
        return None
    sh, sw = shape.shape
    r = th // 2 - sh // 2
    c = tw // 2 - sw // 2
    if r < 1 or c < 1 or r + sh > th - 1 or c + sw > tw - 1:
        return None
    return _validate_positions(occ, shape, [(r, c)], th, tw)


def _try_place_symmetric(occ: np.ndarray, shape: np.ndarray, th: int, tw: int):
    placers = [_try_place_quadrant]
    if th % 2 == 1:
        placers.append(_try_place_hcenter)
    if tw % 2 == 1:
        placers.append(_try_place_vcenter)
    if th % 2 == 1 and tw % 2 == 1:
        placers.append(_try_place_center)
    return random.choice(placers)(occ, shape, th, tw)


def _carve_hole(occ: np.ndarray, shape: np.ndarray, positions: list[tuple[int, int]]) -> None:
    sh, sw = shape.shape
    for pr, pc in positions:
        for dr in range(sh):
            for dc in range(sw):
                if shape[dr, dc]:
                    occ[pr + dr, pc + dc] = False


def _fitting_orientations(shape: np.ndarray, max_sh: int, max_sw: int) -> list[np.ndarray]:
    return [
        oriented
        for oriented in _all_orientations(shape)
        if oriented.shape[0] <= max_sh and oriented.shape[1] <= max_sw and int(np.sum(oriented)) <= SHAPE_MAX_AREA
    ]


def _select_oriented_groups(th: int, tw: int):
    max_sh = max(1, (th - 2) // 2)
    max_sw = max(1, (tw - 2) // 2)

    families: list[tuple[tuple[tuple[int, int], bytes], list[np.ndarray]]] = []
    for base in SHAPE_LIBRARY:
        orientations = _fitting_orientations(base, max_sh, max_sw)
        if orientations:
            families.append((_canonical_key(base), orientations))

    paired_candidates = [(fk, oris) for fk, oris in families if len(oris) >= 2]
    if not paired_candidates:
        return None

    min_groups = max(MIN_SHAPES, SHAPE_COUNT_RANGE[0])
    max_groups = min(SHAPE_COUNT_RANGE[1], len(families) + 1)
    if max_groups < min_groups:
        return None

    target_groups = random.randint(min_groups, max_groups)
    paired_key, paired_oris = random.choice(paired_candidates)
    paired_pick = random.sample(paired_oris, 2)
    groups = [{"family": paired_key, "shape": shape, "paired": True} for shape in paired_pick]

    remaining = [(fk, oris) for fk, oris in families if fk != paired_key]
    random.shuffle(remaining)
    extras_needed = target_groups - len(groups)
    if len(remaining) < extras_needed:
        return None
    for fk, oris in remaining[:extras_needed]:
        groups.append({"family": fk, "shape": random.choice(oris), "paired": False})
    return groups


def _build_hole_layout(th: int, tw: int):
    oriented_groups = _select_oriented_groups(th, tw)
    if oriented_groups is None or len(oriented_groups) < MIN_SHAPES:
        return None

    occ = np.ones((th, tw), dtype=bool)
    placed_groups = []
    mult_vals, mult_wts = MULT_EVEN_WEIGHTS if (th % 2 == 0 or tw % 2 == 0) else MULT_ODD_WEIGHTS

    for group in oriented_groups:
        shape = group["shape"]
        mult = random.choices(mult_vals, weights=mult_wts)[0]
        placements = []
        for _ in range(mult):
            for _ in range(PLACEMENT_RETRIES):
                positions = _try_place_symmetric(occ, shape, th, tw)
                if positions is not None:
                    _carve_hole(occ, shape, positions)
                    placements.append(positions)
                    break
        if not placements:
            return None
        placed_groups.append(
            {
                "family": group["family"],
                "shape": shape,
                "paired": group["paired"],
                "placements": placements,
            }
        )

    frame_density = 1.0 - int(np.sum(~occ)) / float(th * tw)
    if not (FRAME_DENSITY_RANGE[0] <= frame_density <= FRAME_DENSITY_RANGE[1]):
        return None

    total_holes = sum(len(pos) for group in placed_groups for pos in group["placements"])
    if total_holes < MIN_TOTAL_HOLES:
        return None

    if sum(1 for group in placed_groups if group["paired"]) < 2:
        return None

    return occ, placed_groups


def _build_output(occ: np.ndarray, placed_groups: list[dict], frame_col: int, bg: int):
    avail_colors = [c for c in range(NUM_COLORS) if c != bg and c != frame_col]
    if len(avail_colors) < len(placed_groups):
        return None

    fill_colors = random.sample(avail_colors, len(placed_groups))
    output = np.full(occ.shape, frame_col, dtype=int)
    shape_info = []

    for group, fill_color in zip(placed_groups, fill_colors):
        shape = group["shape"]
        all_pos = []
        for positions in group["placements"]:
            for pr, pc in positions:
                paste(output, np.full(shape.shape, int(fill_color), dtype=int), pr, pc, mask=shape)
                all_pos.append((pr, pc))
        shape_info.append(
            {
                "family": group["family"],
                "shape": shape,
                "fill_color": int(fill_color),
                "positions": all_pos,
                "paired": bool(group["paired"]),
            }
        )

    paired_colors = {info["fill_color"] for info in shape_info if info["paired"]}
    if len(paired_colors) < 2:
        return None

    return output, shape_info


def _sample_frame_origin(th: int, tw: int) -> tuple[int, int]:
    row_lo, row_hi = 2, GRID_SIZE - th - 2
    col_lo, col_hi = 2, GRID_SIZE - tw - 2
    assert row_lo <= row_hi and col_lo <= col_hi
    return random.randint(row_lo, row_hi), random.randint(col_lo, col_hi)


def _bbox_has_padding(
    r: int,
    c: int,
    sh: int,
    sw: int,
    frame_bbox: tuple[int, int, int, int],
) -> bool:
    fr0, fc0, fr1, fc1 = frame_bbox
    return (r + sh - 1 < fr0 - 1) or (r > fr1 + 1) or (c + sw - 1 < fc0 - 1) or (c > fc1 + 1)


def _can_place_shape(
    placement: np.ndarray,
    shape: np.ndarray,
    r: int,
    c: int,
    bg: int,
    frame_bbox: tuple[int, int, int, int],
) -> bool:
    h, w = placement.shape
    sh, sw = shape.shape
    if r < 0 or c < 0 or r + sh > h or c + sw > w:
        return False
    if not _bbox_has_padding(r, c, sh, sw, frame_bbox):
        return False

    fg = np.argwhere(shape)
    for dr, dc in fg:
        rr = int(r + dr)
        cc = int(c + dc)
        for nr in range(max(0, rr - 1), min(h, rr + 2)):
            for nc in range(max(0, cc - 1), min(w, cc + 2)):
                if int(placement[nr, nc]) != int(bg):
                    return False
    return True


def _candidate_positions(
    placement: np.ndarray,
    shape: np.ndarray,
    bg: int,
    frame_bbox: tuple[int, int, int, int],
) -> list[tuple[float, int, int]]:
    h, w = placement.shape
    sh, sw = shape.shape
    rel = np.argwhere(shape)
    candidates: list[tuple[float, int, int]] = []
    for r in range(h - sh + 1):
        for c in range(w - sw + 1):
            if not _can_place_shape(placement, shape, r, c, bg, frame_bbox):
                continue
            coords = rel + np.array([r, c])
            dist = _distance_coords_to_frame(coords, frame_bbox)
            candidates.append((dist, int(r), int(c)))
    candidates.sort()
    return candidates


def _place_sprite_at(
    inp: np.ndarray,
    placement: np.ndarray,
    shape: np.ndarray,
    color: int,
    r: int,
    c: int,
    bg: int,
) -> None:
    patch = np.full(shape.shape, int(bg), dtype=int)
    patch[shape] = int(color)
    paste(placement, patch, int(r), int(c), transparent=int(bg))
    paste(inp, patch, int(r), int(c), transparent=int(bg))


def _choose_near_candidate(candidates: list[tuple[float, int, int]]):
    if not candidates:
        return None
    pool = candidates[: min(NEAR_POOL_SIZE, len(candidates))]
    return random.choice(pool)


def _choose_far_candidate(candidates: list[tuple[float, int, int]], min_dist: float):
    far = [cand for cand in candidates if cand[0] >= min_dist]
    if not far:
        return None
    start = max(0, int(len(far) * (1.0 - FAR_TAIL_FRACTION)))
    pool = far[start:] or far
    return random.choice(pool)


def _place_distance_conflict_group(
    inp: np.ndarray,
    placement: np.ndarray,
    info: dict,
    winner_color: int,
    loser_color: int,
    bg: int,
    frame_bbox: tuple[int, int, int, int],
) -> bool:
    shape = info["shape"]
    candidates = _candidate_positions(placement, shape, bg, frame_bbox)
    winner_pick = _choose_near_candidate(candidates)
    if winner_pick is None:
        return False

    winner_dist, wr, wc = winner_pick
    _place_sprite_at(inp, placement, shape, winner_color, wr, wc, bg)

    loser_target = random.randint(*LOSER_COUNT_RANGE)
    placed_losers = 0
    while placed_losers < loser_target:
        candidates = _candidate_positions(placement, shape, bg, frame_bbox)
        loser_pick = _choose_far_candidate(candidates, winner_dist + DISTANCE_MARGIN)
        if loser_pick is None:
            break
        _, lr, lc = loser_pick
        _place_sprite_at(inp, placement, shape, loser_color, lr, lc, bg)
        placed_losers += 1

    if placed_losers < LOSER_COUNT_RANGE[0]:
        return False

    return True


def _place_alt_orientation_decoys(
    inp: np.ndarray,
    placement: np.ndarray,
    family_to_infos: dict[tuple[tuple[int, int], bytes], list[dict]],
    distractor_colors: list[int],
    bg: int,
    frame_bbox: tuple[int, int, int, int],
) -> None:
    for infos in family_to_infos.values():
        if not any(info["paired"] for info in infos):
            continue
        family_oris = _all_orientations(infos[0]["shape"])
        used_keys = {_mask_key(info["shape"]) for info in infos}
        unused = [ori for ori in family_oris if _mask_key(ori) not in used_keys]
        if not unused or not distractor_colors:
            continue
        for _ in range(random.randint(*EXTRA_ALT_DECOYS)):
            shape = random.choice(unused)
            candidates = _candidate_positions(placement, shape, bg, frame_bbox)
            pick = _choose_near_candidate(candidates)
            if pick is None:
                break
            _, r, c = pick
            color = random.choice(distractor_colors)
            _place_sprite_at(inp, placement, shape, color, r, c, bg)


def _scatter_sprites(
    inp: np.ndarray,
    shape_info: list[dict],
    bg: int,
    frame_col: int,
    frame_bbox: tuple[int, int, int, int],
) -> bool:
    placement = inp.copy()

    used_fill = {int(info["fill_color"]) for info in shape_info}
    distractor_colors = [c for c in range(NUM_COLORS) if c not in {bg, frame_col} and c not in used_fill]
    if not distractor_colors:
        return False

    family_to_infos: dict[tuple[tuple[int, int], bytes], list[dict]] = {}
    for info in shape_info:
        family_to_infos.setdefault(info["family"], []).append(info)

    for idx, info in enumerate(shape_info):
        loser_color = distractor_colors[idx % len(distractor_colors)]
        if not _place_distance_conflict_group(
            inp,
            placement,
            info,
            int(info["fill_color"]),
            int(loser_color),
            int(bg),
            frame_bbox,
        ):
            return False

    _place_alt_orientation_decoys(inp, placement, family_to_infos, distractor_colors, int(bg), frame_bbox)
    return True


def _gen_attempt() -> np.ndarray | None:
    bg = random.randint(0, NUM_COLORS - 1)
    frame_col = random.choice([c for c in range(NUM_COLORS) if c != bg])
    th = random.randint(*FRAME_H_RANGE)
    tw = random.randint(*FRAME_W_RANGE)

    layout = _build_hole_layout(th, tw)
    if layout is None:
        return None
    occ, placed_groups = layout

    built = _build_output(occ, placed_groups, frame_col, bg)
    if built is None:
        return None
    output, shape_info = built

    inp = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
    fr, fc = _sample_frame_origin(th, tw)
    frame_bbox = (fr, fc, fr + th - 1, fc + tw - 1)
    paste(inp, np.where(occ, frame_col, bg).astype(int), fr, fc)

    if not _scatter_sprites(inp, shape_info, bg, frame_col, frame_bbox):
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
