# Variant of 67e490f4
# Dimension: D8 — 目标结构：框架的对称性 (dst)
# Change: 框架内洞的布局具有 LR+UD+rot180 对称性（四重对称），通过在左上象限放置后镜像到其他象限实现... → 无对称性约束（自由布局）

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
    random_free_location,
)

# Design analysis for the D8 variant
# 1) Concrete behavior and trace:
#    - Old task: holes were placed by mirroring one seed placement across LR, UD, and 180-degree rotation,
#      so a hole in the upper-left implied matching copies in the other quadrants.
#    - New task: the frame is still a single monochrome rectangle with internal holes, and outside sprites
#      still serve as shape-coded color clues, but the holes can appear anywhere inside the frame interior.
#      Example: an L-hole may appear near the upper-left border while another T-hole appears only near the
#      lower-right; no mirrored partner is required. The solver behavior stays "detect each hole, match its
#      shape against outside sprites, fill with the plurality color", but the generator must stop relying on
#      mirrored placement groups.
# 2) Complexity calibration:
#    - Keep 3-6 distinct canonical shapes, each reused across multiple holes, so shape-to-color evidence is
#      repeated and human-solvable.
#    - Keep 6-13 total holes and moderate hole density, so the frame is visually structured rather than sparse
#      noise, but not cluttered enough to cause accidental touching or ambiguous components.
#    - Keep winning sprite color as a strict plurality for each shape, so color choice is readable without
#      relying on the solver's tie-break.
# 3) Implicit-assumption audit of main():
#    - Largest 8-connected monochrome non-bg component is assumed to be the frame. Generator therefore keeps
#      the frame much larger than any sprite and separates all sprites from it.
#    - Inside the frame bbox, every non-frame-color region is treated as a hole. Generator therefore allows
#      only bg cells inside holes and forbids sprites from entering the bbox.
#    - Hole matching uses the exact hole mask, while sprites are indexed under all D8 transforms. Generator may
#      freely rotate/flip hole instances because the solver still matches them through transformed sprite keys.
#    - Each hole key must exist in sprites_by_key. Generator ensures every canonical hole shape gets sprites.
# 4) Placement legality:
#    - Holes stay at least one cell away from the frame border, never overlap, and remain 4-disconnected from
#      one another.
#    - Sprite placement is computed on a canvas where the full frame bbox is marked occupied, so sprites are
#      guaranteed to stay outside the frame and at least one bg cell away from each other.
# 5) Visual observability:
#    - The key D8 change must be visible in 2-3 examples, so layouts are explicitly rejected if the binary hole
#      mask is symmetric under LR, UD, or 180-degree rotation. This prevents silent regression to the original.
# 6) Visual cleanliness:
#    - No extra distractor system is added. The scene contains one frame plus separated outside sprites, keeping
#      the puzzle readable despite the freer hole arrangement.
# 7) Color inferability:
#    - Each canonical shape gets one fill color, reused across all of its hole instances. Other sprite colors of
#      the same shape may appear, but never as ties or hidden arithmetic on color ids; humans can infer the rule
#      by counting repeated sprite colors for the same shape.
# 8) New main() assumptions after the variant:
#    - The solver now depends even more on shape matching rather than layout regularity. Generator guarantees
#      shape uniqueness up to D8 canonicalization across the selected hole types, so one sprite family maps to
#      exactly one semantic hole type.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct a frame by coloring its freely placed holes from outside shape clues.

    Input structure:
    - The background is the most common color.
    - One large monochrome rectangular frame sits on that background, with irregular holes cut anywhere inside.
    - Outside the frame, small disconnected sprites appear on the background and encode shape-to-color evidence.

    Rule:
    - Locate the largest frame, crop to its bounding box, and start from a solid block of the frame color.
    - Treat each internal hole as a 4-connected shape, then find outside sprites with the same shape up to
      rotation or reflection.
    - Fill every hole with the most frequent color among its matching sprites.

    Key insight:
    - The informative signal is shape equivalence between holes and outside sprites; the hole layout itself is
      arbitrary, so symmetry provides no shortcut in this variant.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

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

    hole_mask = sub != frame_col
    holes: list[tuple[tuple[tuple[int, int], bytes], tuple[int, int], np.ndarray]] = []
    for coords in components_mask(hole_mask, connectivity=4):
        hr0, hc0, hr1, hc1 = bbox_coords(coords)
        hh = int(hr1 - hr0 + 1)
        ww = int(hc1 - hc0 + 1)
        rel = coords - np.array([hr0, hc0])
        m = np.zeros((hh, ww), dtype=bool)
        m[rel[:, 0], rel[:, 1]] = True
        holes.append(((m.shape, m.tobytes()), (int(hr0), int(hc0)), m))

    outside = g.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = bg
    spr_comps = find_components(outside, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    sprites_by_key: dict[tuple[tuple[int, int], bytes], dict[int, int]] = {}

    def _all_mask_keys(mask: np.ndarray):
        m0 = mask.astype(bool)
        for flip in (False, True):
            m1 = np.flipud(m0) if flip else m0
            for k in range(4):
                m2 = np.rot90(m1, k)
                yield (m2.shape, m2.tobytes())

    for comp in spr_comps:
        coords = comp["coords"]
        sr0, sc0, sr1, sc1 = map(int, comp["bbox"])
        hh = int(sr1 - sr0 + 1)
        ww = int(sc1 - sc0 + 1)

        vals = [int(v) for v in g[coords[:, 0], coords[:, 1]]]
        scol = int(Counter(vals).most_common(1)[0][0])

        rel = coords - np.array([sr0, sc0])
        m = np.zeros((hh, ww), dtype=bool)
        m[rel[:, 0], rel[:, 1]] = True

        for key in _all_mask_keys(m):
            bucket = sprites_by_key.setdefault(key, {})
            bucket[scol] = int(bucket.get(scol, 0)) + 1

    out = np.full((th, tw), frame_col, dtype=int)
    for key, (hr0, hc0), hmask in holes:
        cmap = sprites_by_key.get(key)
        assert cmap is not None
        scol = max(cmap.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        patch = np.full(hmask.shape, scol, dtype=int)
        paste(out, patch, hr0, hc0, mask=hmask)

    return out


NUM_COLORS = 10
GRID_SIZE = 30

FRAME_H_RANGE = (7, 13)
FRAME_W_RANGE = (7, 20)

SHAPE_COUNT_RANGE = (3, 6)
MIN_SHAPES = 3
SHAPE_MAX_AREA = 8
MIN_TOTAL_HOLES = 6
MAX_TOTAL_HOLES = 13
TARGET_HOLE_FILL_RANGE = (0.16, 0.34)
FRAME_DENSITY_RANGE = (0.60, 0.86)

WINNING_SPRITE_RANGE = (2, 5)
OTHER_COLOR_MAX = 2
SPRITE_PADDING = 1

MAX_GEN_ATTEMPTS = 300


def _all_rot_flip(mask: np.ndarray):
    m0 = mask.astype(bool)
    for flip in (False, True):
        m1 = np.flipud(m0) if flip else m0
        for k in range(4):
            m2 = np.rot90(m1, k)
            yield (m2.shape, m2.tobytes()), m2


def _canonical_key(mask: np.ndarray):
    return min(key for key, _ in _all_rot_flip(mask))


def _all_orientations(mask: np.ndarray):
    seen: set[tuple[tuple[int, int], bytes]] = set()
    result: list[np.ndarray] = []
    for key, m in _all_rot_flip(mask):
        if key not in seen:
            seen.add(key)
            result.append(m)
    return result


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
    seen: set[tuple[tuple[int, int], bytes]] = set()
    dedup: list[np.ndarray] = []
    for s in raw:
        m = np.array(s, dtype=bool)
        k = _canonical_key(m)
        if k not in seen:
            seen.add(k)
            dedup.append(m)
    return dedup


SHAPE_LIBRARY = _build_shape_library()


def _symmetry_flags(mask: np.ndarray) -> tuple[bool, bool, bool]:
    return (
        bool(np.array_equal(mask, np.fliplr(mask))),
        bool(np.array_equal(mask, np.flipud(mask))),
        bool(np.array_equal(mask, np.rot90(mask, 2))),
    )


def _breaks_all_old_symmetries(mask: np.ndarray) -> bool:
    lr, ud, rot180 = _symmetry_flags(mask)
    return not lr and not ud and not rot180


def _shape_area(shape: np.ndarray) -> int:
    return int(np.sum(shape))


def _can_place(occ: np.ndarray, shape: np.ndarray, pr: int, pc: int) -> bool:
    th, tw = occ.shape
    sh, sw = shape.shape
    if pr < 1 or pc < 1 or pr + sh > th - 1 or pc + sw > tw - 1:
        return False

    cells = {(pr + dr, pc + dc) for dr, dc in np.argwhere(shape)}
    for rr, cc in cells:
        if not occ[rr, cc]:
            return False
        for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
            if 0 <= nr < th and 0 <= nc < tw and not occ[nr, nc] and (nr, nc) not in cells:
                return False
    return True


def _list_valid_placements(occ: np.ndarray, shape: np.ndarray):
    th, tw = occ.shape
    valid: list[tuple[np.ndarray, int, int]] = []
    for ori in _all_orientations(shape):
        sh, sw = ori.shape
        for pr in range(1, th - sh):
            for pc in range(1, tw - sw):
                if _can_place(occ, ori, pr, pc):
                    valid.append((ori, pr, pc))
    return valid


def _carve_hole(occ: np.ndarray, shape: np.ndarray, pr: int, pc: int) -> None:
    for dr, dc in np.argwhere(shape):
        occ[pr + dr, pc + dc] = False


def _build_hole_layout(th: int, tw: int):
    max_sh, max_sw = max(1, th - 2), max(1, tw - 2)
    available = [
        s
        for s in SHAPE_LIBRARY
        if s.shape[0] <= max_sh and s.shape[1] <= max_sw and _shape_area(s) <= SHAPE_MAX_AREA
    ]
    if len(available) < MIN_SHAPES:
        return None

    k_target = random.randint(SHAPE_COUNT_RANGE[0], min(SHAPE_COUNT_RANGE[1], len(available)))
    chosen = random.sample(available, k_target)
    chosen.sort(key=lambda s: (-_shape_area(s), -max(s.shape), -min(s.shape)))

    occ = np.ones((th, tw), dtype=bool)
    hole_groups: list[tuple[np.ndarray, list[tuple[np.ndarray, tuple[int, int]]]]] = []
    total_holes = 0
    hole_area = 0

    for shape in chosen:
        valid = _list_valid_placements(occ, shape)
        if not valid:
            return None
        ori, pr, pc = random.choice(valid)
        _carve_hole(occ, ori, pr, pc)
        hole_groups.append((shape, [(ori, (pr, pc))]))
        total_holes += 1
        hole_area += _shape_area(ori)

    bbox_area = th * tw
    low_area = max(hole_area, int(np.ceil(bbox_area * TARGET_HOLE_FILL_RANGE[0])))
    high_area = max(low_area, int(np.floor(bbox_area * TARGET_HOLE_FILL_RANGE[1])))
    target_hole_area = random.randint(low_area, high_area)

    stalled = 0
    while (hole_area < target_hole_area or total_holes < MIN_TOTAL_HOLES) and total_holes < MAX_TOTAL_HOLES:
        idx = random.randrange(len(hole_groups))
        shape, instances = hole_groups[idx]
        valid = _list_valid_placements(occ, shape)
        if not valid:
            stalled += 1
            if stalled >= 12:
                break
            continue
        ori, pr, pc = random.choice(valid)
        _carve_hole(occ, ori, pr, pc)
        instances.append((ori, (pr, pc)))
        total_holes += 1
        hole_area += _shape_area(ori)
        stalled = 0

    while total_holes < MAX_TOTAL_HOLES and hole_area < bbox_area * TARGET_HOLE_FILL_RANGE[1] and random.random() < 0.35:
        idx = random.randrange(len(hole_groups))
        shape, instances = hole_groups[idx]
        valid = _list_valid_placements(occ, shape)
        if not valid:
            break
        ori, pr, pc = random.choice(valid)
        _carve_hole(occ, ori, pr, pc)
        instances.append((ori, (pr, pc)))
        total_holes += 1
        hole_area += _shape_area(ori)

    frame_density = 1.0 - hole_area / bbox_area
    if not (FRAME_DENSITY_RANGE[0] <= frame_density <= FRAME_DENSITY_RANGE[1]):
        return None
    if total_holes < MIN_TOTAL_HOLES:
        return None

    hole_mask = ~occ
    if not _breaks_all_old_symmetries(hole_mask):
        return None

    used_rows = int(np.sum(hole_mask.any(axis=1)))
    used_cols = int(np.sum(hole_mask.any(axis=0)))
    if used_rows < 3 or used_cols < 3:
        return None

    return occ, hole_groups


def _build_output(occ: np.ndarray, hole_groups, frame_col: int, bg: int):
    th, tw = occ.shape
    avail_colors = [c for c in range(NUM_COLORS) if c != bg and c != frame_col]
    if len(avail_colors) < len(hole_groups):
        return None

    shape_colors = random.sample(avail_colors, len(hole_groups))
    output = np.full((th, tw), frame_col, dtype=int)
    shape_info = []
    for (shape, instances), fill_col in zip(hole_groups, shape_colors):
        for ori, (pr, pc) in instances:
            patch = np.full(ori.shape, fill_col, dtype=int)
            paste(output, patch, pr, pc, mask=ori)
        shape_info.append((shape, fill_col, instances))
    return output, shape_info


def _place_sprite(inp: np.ndarray, placement: np.ndarray, ori: np.ndarray, color: int, bg: int) -> bool:
    patch = np.full(ori.shape, bg, dtype=int)
    patch[ori] = color
    try:
        r, c = random_free_location(placement, patch, transparent=bg, padding=SPRITE_PADDING)
    except ValueError:
        return False
    paste(placement, patch, r, c, transparent=bg)
    paste(inp, patch, r, c, transparent=bg)
    return True


def _scatter_sprites(inp: np.ndarray, shape_info, bg: int, frame_col: int, frame_bbox) -> bool:
    placement = inp.copy()
    fr0, fc0, fr1, fc1 = map(int, frame_bbox)
    placement[fr0 : fr1 + 1, fc0 : fc1 + 1] = frame_col
    other_pool = [c for c in range(NUM_COLORS) if c != bg and c != frame_col]

    for shape, fill_color, _instances in shape_info:
        orientations = _all_orientations(shape)
        n_winning = random.randint(*WINNING_SPRITE_RANGE)
        losers = [c for c in other_pool if c != fill_color]
        n_other = random.randint(1, min(OTHER_COLOR_MAX, len(losers))) if losers else 0
        oc_list = random.sample(losers, n_other)

        for _ in range(n_winning):
            if not _place_sprite(inp, placement, random.choice(orientations), fill_color, bg):
                return False

        for oc in oc_list:
            n_oc = random.randint(1, n_winning - 1)
            for _ in range(n_oc):
                if not _place_sprite(inp, placement, random.choice(orientations), oc, bg):
                    break
    return True


def _gen_attempt() -> np.ndarray | None:
    bg = random.randint(0, NUM_COLORS - 1)
    frame_col = random.choice([c for c in range(NUM_COLORS) if c != bg])
    th = random.randint(*FRAME_H_RANGE)
    tw = random.randint(*FRAME_W_RANGE)

    layout = _build_hole_layout(th, tw)
    if layout is None:
        return None
    occ, hole_groups = layout

    result = _build_output(occ, hole_groups, frame_col, bg)
    if result is None:
        return None
    output, shape_info = result

    fr = random.randint(0, GRID_SIZE - th)
    fc = random.randint(0, GRID_SIZE - tw)
    inp = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
    paste(inp, np.where(occ, frame_col, bg).astype(int), fr, fc)

    frame_bbox = (fr, fc, fr + th - 1, fc + tw - 1)
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
