# Variant of 67e490f4
# Dimension: D7 — 目标结构：洞的形态与连通性 (dst)
# Change: 洞为 4-连通背景色区域，形状 bbox 1x1 到 5x6，面积 1-8 像素，3-6 种不同形状，每种 1-5 个实例
#         → 洞内含部分框架色像素（分区洞）

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import components_mask, crop_bbox, find_components, get_bg, paste, random_free_location


# Design analysis for the D7 variant
# ----------------------------------
# a) Concrete behavior:
#    In the original task, every hole is one 4-connected bg component, so taking bg-components
#    inside the frame bbox is enough to recover the template keys. In this variant, a hole is an
#    isolated small bbox whose internal pattern mixes bg cells and frame-colored divider cells.
#    Example:
#        frame input window      matched sprite
#        7 0 7                  7 3 7
#        7 0 7        <->       7 3 7
#    The full 2x3 window is the key. The two bg columns are filled, but the middle column stays
#    frame-colored. Solver impact: "bg-component == hole" is no longer true. We must instead
#    detect isolated patterned windows and use their whole bg-vs-frame layout as the shape key.
#
# b) Complexity calibration:
#    All hole patterns are genuinely partitioned (their bg mask has >=2 4-components) so the D7
#    change is visible in every sample. To avoid ambiguous mergers or accidental subwindow matches,
#    the pattern library is limited to bboxes 2x2..5x6 whose bg mask touches all four sides and
#    whose canonical key is repeated across multiple instances via symmetry.
#
# c) Old assumptions that break:
#    - `hole_mask = (sub != frame_col)` and `components_mask(..., connectivity=4)` assumed one hole
#      == one connected non-frame region. That fails here because divider strips split a single hole
#      into several bg pieces.
#    - Filling with `paste(..., mask=hmask)` used the whole detected component as writable area.
#      That would overwrite divider strips if we recovered the wrong mask. New solver only pastes
#      onto bg cells inside each detected window.
#    - Outside sprites used to be monochrome. Now they are multicolor rectangles: fill cells carry
#      the voting color, divider cells carry `frame_col`. Solver therefore extracts the dominant
#      non-frame color from each multi-color sprite while keeping the full window mask as the key.
#
# d) Placement legality:
#    Generator reserves entire hole bboxes, not just bg cells, and keeps a two-cell gap between
#    bboxes. This prevents two neighboring holes from being misread as one larger partitioned hole.
#    Every hole bbox also keeps a one-cell frame border so the solver can detect it by its outer ring.
#
# e) Visual observability:
#    Every sample contains multiple partitioned holes and matching outside sprites that visibly carry
#    frame-colored divider strips. Humans can see that the divider remains unchanged in the output
#    while the bg subregions inherit the majority sprite color for that full window pattern.
#
# f) Visual cleanliness:
#    Hole windows are isolated rectangles, sprites are also isolated rectangles, and color voting is
#    by repeated exact window pattern. This keeps the input readable despite the extra divider detail.
#
# g) Color inference:
#    No invisible ordering rule is used. Generator guarantees a unique plurality color for each
#    pattern, so the intended fill color is recoverable from repeated sprite occurrences alone.
#
# h) Reference task:
#    No source_task was provided, so no additional reference solver is consulted.
#
# i) New solver assumptions audited against generator:
#    - Assumption: every true hole bbox is fully isolated by a one-cell outer ring of frame color.
#      Guaranteed by bbox placement with border >=1 and gap >=2.
#    - Assumption: the union bbox of a hole's bg cells equals the intended hole bbox.
#      Guaranteed because every pattern touches all four sides.
#    - Assumption: every detected canonical pattern appears among outside sprites.
#      Guaranteed by output-first generation plus oracle validation.
#    - Assumption: per-pattern color plurality is unique.
#      Guaranteed by sprite-count construction.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor patterned windows inside a large frame using matching outside keys.

    Input structure:
    - The background is the most common color. One large monochrome frame object encloses the target area.
    - Inside the frame bbox, each target hole is a small isolated window: some cells are bg, while some
      internal cells stay in the frame color as divider strips.
    - Outside the frame, small rectangular sprites show the same window pattern; divider cells use the
      frame color and fillable cells use a sprite color.

    Rule:
    - Crop to the frame bbox and start from a solid frame-colored canvas.
    - For each isolated internal window, match its full bg-vs-divider pattern to outside sprites
      up to rotation/flip, then pick the most frequent sprite color for that pattern.
    - Fill only the bg cells of the window with that color; keep divider-strip cells unchanged.

    Key insight:
    The thing to match is not each bg subregion separately, but the whole small window pattern.
    Divider strips belong to the frame and survive in the output; they only partition where the
    matched color is allowed to spread.
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

    outside = g.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
    spr_comps = find_components(outside, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    sprites_by_key: dict[tuple[tuple[int, int], bytes], Counter[int]] = {}
    valid_shapes = set()

    for comp in spr_comps:
        sr0, sc0, sr1, sc1 = map(int, comp["bbox"])
        patch = outside[sr0 : sr1 + 1, sc0 : sc1 + 1]
        fill_mask = (patch != int(bg)) & (patch != int(frame_col))
        if not bool(fill_mask.any()):
            continue
        key = _canonical_key(fill_mask)
        valid_shapes.add(fill_mask.shape)
        colors = [int(v) for v in patch[fill_mask]]
        scol = int(Counter(colors).most_common(1)[0][0])
        sprites_by_key.setdefault(key, Counter())[scol] += 1

    assert sprites_by_key

    candidates = []
    used_shape_set = set(valid_shapes)
    for hh, ww in sorted(used_shape_set, key=lambda s: (s[0] * s[1], s[0], s[1]), reverse=True):
        if hh > th - 2 or ww > tw - 2:
            continue
        for hr0 in range(1, th - hh):
            for hc0 in range(1, tw - ww):
                window = sub[hr0 : hr0 + hh, hc0 : hc0 + ww]
                fill_mask = window == int(bg)
                if not bool(fill_mask.any()):
                    continue
                if not _touches_all_sides(fill_mask):
                    continue
                if not _outer_ring_is_frame(sub, hr0, hc0, hh, ww, int(frame_col)):
                    continue
                key = _canonical_key(fill_mask)
                if key not in sprites_by_key:
                    continue
                candidates.append((hr0, hc0, fill_mask, key))

    occupied = np.zeros_like(sub, dtype=bool)
    holes = []
    for hr0, hc0, fill_mask, key in candidates:
        hh, ww = fill_mask.shape
        if bool(occupied[hr0 : hr0 + hh, hc0 : hc0 + ww].any()):
            continue
        occupied[hr0 : hr0 + hh, hc0 : hc0 + ww] = True
        holes.append((hr0, hc0, fill_mask, key))

    assert holes
    covered_bg = np.zeros_like(sub, dtype=bool)
    for hr0, hc0, fill_mask, _ in holes:
        hh, ww = fill_mask.shape
        covered_bg[hr0 : hr0 + hh, hc0 : hc0 + ww] |= fill_mask
    assert np.array_equal(covered_bg, sub == int(bg))

    out = np.full((th, tw), int(frame_col), dtype=int)
    for hr0, hc0, fill_mask, key in holes:
        cmap = sprites_by_key.get(key)
        assert cmap
        scol = int(cmap.most_common(1)[0][0])
        patch = np.full(fill_mask.shape, scol, dtype=int)
        paste(out, patch, int(hr0), int(hc0), mask=fill_mask)

    return out


# --------------- generator ---------------

NUM_COLORS = 10
GRID_SIZE = 30

FRAME_H_RANGE = (9, 13)
FRAME_W_RANGE = (10, 20)

SHAPE_COUNT_RANGE = (3, 5)
MIN_SHAPES = 3
MIN_TOTAL_HOLES = 6
FRAME_DENSITY_RANGE = (0.58, 0.9)
PLACEMENT_RETRIES = 80
HOLE_GAP = 1

WINNING_SPRITE_RANGE = (2, 5)
OTHER_COLOR_MAX = 2
SPRITE_PADDING = 1

MAX_GEN_ATTEMPTS = 400


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
    seen = set()
    result = []
    for key, m in _all_rot_flip(mask):
        if key not in seen:
            seen.add(key)
            result.append(m)
    return result


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(
        mask[0, :].any()
        and mask[-1, :].any()
        and mask[:, 0].any()
        and mask[:, -1].any()
    )


def _component_count(mask: np.ndarray) -> int:
    return len(components_mask(mask.astype(bool), connectivity=4))


def _build_pattern_library():
    raw = [
        [[1, 0], [0, 1]],
        [[1, 0, 1], [1, 0, 1]],
        [[1, 1], [0, 0], [1, 1]],
        [[1, 0, 1], [1, 0, 1], [1, 0, 1]],
        [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
        [[1, 1, 0], [0, 0, 0], [0, 1, 1]],
        [[1, 0, 1, 0], [1, 0, 1, 0]],
        [[1, 0, 1, 0, 1], [1, 0, 1, 0, 1]],
        [[1, 1, 1], [0, 0, 0], [1, 0, 1]],
        [[1, 0, 1], [0, 0, 0], [1, 0, 1]],
        [[1, 0, 0], [0, 0, 0], [0, 0, 1]],
        [[1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 1, 1]],
    ]
    seen = set()
    lib = []
    for pat in raw:
        m = np.array(pat, dtype=bool)
        if not _touches_all_sides(m):
            continue
        area = int(np.sum(m))
        if not (2 <= area <= 8):
            continue
        if _component_count(m) < 2:
            continue
        key = _canonical_key(m)
        if key in seen:
            continue
        seen.add(key)
        lib.append(m)
    return lib


PATTERN_LIBRARY = _build_pattern_library()


def _outer_ring_is_frame(sub: np.ndarray, r: int, c: int, h: int, w: int, frame_col: int) -> bool:
    ring = sub[r - 1 : r + h + 1, c - 1 : c + w + 1]
    mask = np.ones_like(ring, dtype=bool)
    mask[1:-1, 1:-1] = False
    return bool(np.all(ring[mask] == int(frame_col)))


def _mirror_pos(r: int, c: int, sh: int, sw: int, th: int, tw: int):
    return list(
        {
            (r, c),
            (r, tw - sw - c),
            (th - sh - r, c),
            (th - sh - r, tw - sw - c),
        }
    )


def _mark_used(used: np.ndarray, r: int, c: int, h: int, w: int):
    rr0 = max(0, r - HOLE_GAP)
    cc0 = max(0, c - HOLE_GAP)
    rr1 = min(used.shape[0], r + h + HOLE_GAP)
    cc1 = min(used.shape[1], c + w + HOLE_GAP)
    used[rr0:rr1, cc0:cc1] = True


def _can_place(used: np.ndarray, r: int, c: int, h: int, w: int, th: int, tw: int) -> bool:
    if r < 1 or c < 1 or r + h > th - 1 or c + w > tw - 1:
        return False
    rr0 = max(0, r - HOLE_GAP)
    cc0 = max(0, c - HOLE_GAP)
    rr1 = min(th, r + h + HOLE_GAP)
    cc1 = min(tw, c + w + HOLE_GAP)
    return not bool(used[rr0:rr1, cc0:cc1].any())


def _try_place_quadrant(used: np.ndarray, shape: np.ndarray, th: int, tw: int):
    sh, sw = shape.shape
    r_max = th // 2 - sh
    c_max = tw // 2 - sw
    if r_max < 1 or c_max < 1:
        return None
    for _ in range(PLACEMENT_RETRIES):
        r = random.randint(1, r_max)
        c = random.randint(1, c_max)
        positions = _mirror_pos(r, c, sh, sw, th, tw)
        if all(_can_place(used, pr, pc, sh, sw, th, tw) for pr, pc in positions):
            return positions
    return None


def _try_place_hcenter(used: np.ndarray, shape: np.ndarray, th: int, tw: int):
    if th % 2 == 0:
        return None
    sh, sw = shape.shape
    r = th // 2 - sh // 2
    if r < 1 or r + sh > th - 1:
        return None
    c_max = tw // 2 - sw
    if c_max < 1:
        return None
    for _ in range(PLACEMENT_RETRIES):
        c = random.randint(1, c_max)
        positions = list({(r, c), (r, tw - sw - c)})
        if len(positions) != 2:
            continue
        if all(_can_place(used, pr, pc, sh, sw, th, tw) for pr, pc in positions):
            return positions
    return None


def _try_place_vcenter(used: np.ndarray, shape: np.ndarray, th: int, tw: int):
    if tw % 2 == 0:
        return None
    sh, sw = shape.shape
    c = tw // 2 - sw // 2
    if c < 1 or c + sw > tw - 1:
        return None
    r_max = th // 2 - sh
    if r_max < 1:
        return None
    for _ in range(PLACEMENT_RETRIES):
        r = random.randint(1, r_max)
        positions = list({(r, c), (th - sh - r, c)})
        if len(positions) != 2:
            continue
        if all(_can_place(used, pr, pc, sh, sw, th, tw) for pr, pc in positions):
            return positions
    return None


def _try_place_center(used: np.ndarray, shape: np.ndarray, th: int, tw: int):
    if th % 2 == 0 or tw % 2 == 0:
        return None
    sh, sw = shape.shape
    r = th // 2 - sh // 2
    c = tw // 2 - sw // 2
    if not _can_place(used, r, c, sh, sw, th, tw):
        return None
    return [(r, c)]


def _try_place_group(used: np.ndarray, shape: np.ndarray, th: int, tw: int):
    placers = []
    quad = _try_place_quadrant
    if th // 2 - shape.shape[0] >= 1 and tw // 2 - shape.shape[1] >= 1:
        placers.extend([quad, quad, quad])
    if th % 2 == 1:
        placers.append(_try_place_hcenter)
    if tw % 2 == 1:
        placers.append(_try_place_vcenter)
    if th % 2 == 1 and tw % 2 == 1:
        placers.append(_try_place_center)
    random.shuffle(placers)
    for placer in placers:
        positions = placer(used, shape, th, tw)
        if positions is not None:
            return positions
    return None


def _build_hole_layout(th: int, tw: int):
    max_h = min(5, th - 2)
    max_w = min(6, tw - 2)
    available = [p for p in PATTERN_LIBRARY if p.shape[0] <= max_h and p.shape[1] <= max_w]
    if len(available) < MIN_SHAPES:
        return None

    target = random.randint(SHAPE_COUNT_RANGE[0], min(SHAPE_COUNT_RANGE[1], len(available)))
    random.shuffle(available)

    used = np.zeros((th, tw), dtype=bool)
    placements = []
    for shape in available:
        if len(placements) >= target:
            break
        positions = _try_place_group(used, shape, th, tw)
        if positions is None:
            continue
        for pr, pc in positions:
            _mark_used(used, pr, pc, shape.shape[0], shape.shape[1])
        placements.append((shape, positions))

    if len(placements) < MIN_SHAPES:
        return None

    total_holes = sum(len(pos_list) for _, pos_list in placements)
    if total_holes < MIN_TOTAL_HOLES:
        return None

    fill_area = sum(int(np.sum(shape)) * len(pos_list) for shape, pos_list in placements)
    frame_density = 1.0 - fill_area / float(th * tw)
    if not (FRAME_DENSITY_RANGE[0] <= frame_density <= FRAME_DENSITY_RANGE[1]):
        return None

    return placements


def _build_output(placements, frame_col: int, bg: int):
    max_colors = len([c for c in range(NUM_COLORS) if c not in (bg, frame_col)])
    if max_colors < len(placements):
        return None
    canvas_h = max(pr + shape.shape[0] for shape, pos_list in placements for pr, _ in pos_list)
    canvas_w = max(pc + shape.shape[1] for shape, pos_list in placements for _, pc in pos_list)
    output = np.full((canvas_h, canvas_w), frame_col, dtype=int)

    avail_colors = [c for c in range(NUM_COLORS) if c != bg and c != frame_col]
    shape_colors = random.sample(avail_colors, len(placements))
    shape_info = []
    for (shape, pos_list), fill_col in zip(placements, shape_colors):
        for pr, pc in pos_list:
            patch = np.full(shape.shape, fill_col, dtype=int)
            paste(output, patch, pr, pc, mask=shape)
        shape_info.append((shape, fill_col, list(pos_list)))
    return output, shape_info


def _place_sprite(inp: np.ndarray, placement: np.ndarray, ori: np.ndarray, color: int, bg: int, frame_col: int):
    patch = np.full(ori.shape, frame_col, dtype=int)
    patch[ori] = color
    try:
        r, c = random_free_location(placement, patch, transparent=bg, padding=SPRITE_PADDING)
    except ValueError:
        return False
    paste(placement, patch, r, c, transparent=bg)
    paste(inp, patch, r, c, transparent=bg)
    return True


def _scatter_sprites(inp: np.ndarray, shape_info, bg: int, frame_col: int):
    placement = inp.copy()
    other_pool = [c for c in range(NUM_COLORS) if c != bg and c != frame_col]

    for shape, fill_color, _ in shape_info:
        orientations = _all_orientations(shape)
        n_winning = random.randint(*WINNING_SPRITE_RANGE)
        losers = [c for c in other_pool if c != fill_color]
        n_other = random.randint(1, min(OTHER_COLOR_MAX, len(losers))) if losers else 0
        oc_list = random.sample(losers, n_other) if n_other else []

        for _ in range(n_winning):
            if not _place_sprite(inp, placement, random.choice(orientations), fill_color, bg, frame_col):
                return False

        for oc in oc_list:
            n_oc = random.randint(1, max(1, n_winning - 1))
            for _ in range(n_oc):
                if not _place_sprite(inp, placement, random.choice(orientations), oc, bg, frame_col):
                    break
    return True


def _gen_attempt() -> np.ndarray | None:
    bg = random.randint(0, NUM_COLORS - 1)
    frame_col = random.choice([c for c in range(NUM_COLORS) if c != bg])
    th = random.randint(*FRAME_H_RANGE)
    tw = random.randint(*FRAME_W_RANGE)

    placements = _build_hole_layout(th, tw)
    if placements is None:
        return None

    output = np.full((th, tw), frame_col, dtype=int)
    avail_colors = [c for c in range(NUM_COLORS) if c != bg and c != frame_col]
    if len(avail_colors) < len(placements):
        return None
    shape_colors = random.sample(avail_colors, len(placements))
    shape_info = []
    for (shape, pos_list), fill_col in zip(placements, shape_colors):
        for pr, pc in pos_list:
            patch = np.full(shape.shape, fill_col, dtype=int)
            paste(output, patch, pr, pc, mask=shape)
        shape_info.append((shape, fill_col, list(pos_list)))

    inp = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
    template = np.full((th, tw), frame_col, dtype=int)
    for shape, _, pos_list in shape_info:
        for pr, pc in pos_list:
            patch = np.full(shape.shape, bg, dtype=int)
            paste(template, patch, pr, pc, mask=shape)

    fr = random.randint(0, GRID_SIZE - th)
    fc = random.randint(0, GRID_SIZE - tw)
    paste(inp, template, fr, fc)

    if not _scatter_sprites(inp, shape_info, bg, frame_col):
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
