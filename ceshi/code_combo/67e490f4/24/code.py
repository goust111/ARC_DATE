# Combo variant of 67e490f4 (index 24)
# combo_spec: D6/2+D7/3+D12/2
# Name: 缺口框架+分区洞+假洞干扰

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import components_mask, crop_bbox, find_components, get_bg, paste, random_free_location


# Design analysis for the fused variant
# -------------------------------------
# a) Concrete behavior and execution trace:
#    Original task:
#    - Treat every non-frame region inside the frame bbox as a fillable hole.
#    - Match each hole's exact connected shape to outside sprites.
#    - Repaint the whole hole with the plurality sprite color.
#    Fused task:
#    - Border bg notches are part of the frame silhouette and must stay bg in the output.
#    - A true hole is an interior rectangular window whose cells are only {bg, frame_col}; inside that
#      window, bg cells can be split by frame-colored divider strips, so one hole can contain multiple
#      bg 4-components.
#    - A fake hole uses the same kind of small window geometry, but its fill cells are a special non-bg
#      decoy color instead of bg, so it visually resembles a target while remaining unchanged.
#    Example 3x3 interior window:
#        input true hole     matching sprite      output
#        7 0 7              7 3 7                7 3 7
#        0 7 0      <->     3 7 3        ->      3 7 3
#        7 0 7              7 3 7                7 3 7
#      A fake hole may show the same pattern with color 8 instead of bg:
#        7 8 7
#        8 7 8
#        7 8 7
#      That decoy stays 8 in the output. A border notch like a 1x2 bg gap on the top edge also stays bg.
#
# b) Complexity calibration:
#    - Every accepted sample contains all three visible signals: at least one border notch, at least one
#      partitioned true hole (bg mask with >=2 components), and at least one fake-hole decoy color inside
#      the frame.
#    - True holes come from a small repeated library of partitioned masks so the matching rule is learnable
#      from a few examples instead of depending on one-off shapes.
#    - Fake holes reuse the same visual language as true holes, but only true holes use bg as the writable
#      material. That keeps the fusion non-trivial without making it arbitrary.
#
# c) Broken old assumptions and fused replacement:
#    - Old assumption: every `sub != frame_col` component is a hole. Broken twice: edge-connected bg can be
#      a notch, and one logical hole can split into several bg components because of dividers.
#    - Old assumption: output can start as solid frame color. Broken by preserved notches and preserved fake
#      holes, so the new solver starts from the cropped frame window itself and only repaints confirmed true
#      bg cells.
#    - Old assumption: inside-hole shape can be recovered directly from connected bg components. Broken by
#      partitioned holes, so the solver matches whole interior windows against sprite patterns instead.
#    - New fused rule: a true target must simultaneously satisfy three predicates:
#      1. interior to the bbox with a full outer ring of frame color,
#      2. exact bg-vs-frame window pattern matching an outside sprite up to rotation/flip,
#      3. no decoy colors inside the window.
#
# d) Placement legality:
#    - Hole and fake-hole windows are placed with a one-cell outer frame ring and a one-cell inter-window gap.
#    - True/fake windows stay at least two cells from the bbox border, while notches are only one cell deep;
#      this prevents border notches from merging with true holes.
#    - Outside sprites are placed using a placement canvas that marks the entire frame bbox as occupied, so
#      they cannot land in notches or other empty cells inside the frame bbox.
#
# e) Visual observability:
#    - Humans can separately see three categories inside the cropped frame: edge-attached bg bites, interior
#      bg windows with frame dividers, and interior colored decoy windows. Only the second category changes.
#    - The outside sprites visibly carry the same divider strips as the true holes, so the "match whole window
#      pattern, then color only bg cells" rule is directly supported by the examples.
#
# f) Visual cleanliness:
#    - Fake holes use a single repeated decoy color per grid, reducing palette noise.
#    - Window placement is isolated and symmetric-by-group, avoiding merged clutter.
#    - Notches are shallow boundary cuts, so the main frame remains easy to parse.
#
# g) Color inference and learnability:
#    - True hole colors come from repeated outside sprites only; fake-hole colors are already present inside
#      the frame and never need inference. This prevents the user from having to guess among too many color
#      roles at once.
#    - The generator enforces a strict plurality winner for every true hole pattern, so the intended fill color
#      is observable from frequency, not hidden tie rules.
#
# h) Fused solver assumptions audited against the generator:
#    - Assumption: every true hole window contains only bg and frame_col. Generator paints decoys with a third
#      color, so fake holes are excluded by construction.
#    - Assumption: the union of all non-edge-connected bg cells inside the frame equals the union of writable
#      cells of detected true holes. Generator enforces margin-2 hole placement and one-cell-deep notches.
#    - Assumption: every true hole canonical pattern appears among outside sprites. Generator builds sprites
#      directly from the chosen true-hole shapes and validates with the solver.
#    - Assumption: every detected window is isolated by a frame-colored outer ring. Generator reserves whole
#      bboxes plus a gap, so candidate windows do not overlap or bleed together.
#
# i) Degeneracy guard:
#    - The generator rejects samples without fake-hole cells, without border notches, or without partitioned
#      true holes. This prevents regression back to either the original task or a single-axis variant.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill only the true partitioned holes inside a notched frame using outside sprite keys.

    Input structure:
    - The dominant color is bg. One large monochrome frame component defines a rectangular bbox, but some bbox-edge cells are bg notches rather than solid frame.
    - Inside that bbox, true holes are interior windows made only of bg and frame color: bg cells are the writable regions, and frame-colored divider strips split them into subregions.
    - The same bbox can also contain fake holes, which use a special non-bg decoy color in those window cells; outside the frame, small sprite rectangles repeat the true-hole window patterns.

    Rule:
    - Find the largest monochrome frame component, crop to its bbox, and keep border-connected bg gaps and decoy-colored windows unchanged.
    - Match each interior bg-vs-frame window pattern to outside sprites up to rotation/flip, using the most frequent matching sprite color.
    - Recolor only the bg cells of those matched windows; divider strips stay frame-colored.

    Key insight:
    The target is the whole interior window pattern, not each bg component by itself. Border-connected bg belongs to frame notches, decoy-colored windows are fake holes, and only interior bg cells inside a matched bg-vs-frame pattern should be repainted.
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
    r0, c0, r1, c1 = map(int, frame["bbox"])

    sub = crop_bbox(g, frame["bbox"])
    th, tw = sub.shape

    outside = g.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
    spr_comps = find_components(outside, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    sprites_by_key: dict[tuple[tuple[int, int], bytes], Counter[int]] = {}
    pattern_variants: dict[tuple[tuple[int, int], bytes], dict[tuple[tuple[int, int], bytes], np.ndarray]] = {}

    for comp in spr_comps:
        sr0, sc0, sr1, sc1 = map(int, comp["bbox"])
        patch = outside[sr0 : sr1 + 1, sc0 : sc1 + 1]
        fill_mask = (patch != int(bg)) & (patch != int(frame_col))
        if not bool(fill_mask.any()):
            continue

        key = _canonical_key(fill_mask)
        colors = [int(v) for v in patch[fill_mask]]
        scol = int(Counter(colors).most_common(1)[0][0])
        sprites_by_key.setdefault(key, Counter())[scol] += 1

        bucket = pattern_variants.setdefault(key, {})
        for ori in _all_orientations(fill_mask):
            bucket[(ori.shape, ori.tobytes())] = ori

    assert sprites_by_key

    candidates = []
    for key, variants in pattern_variants.items():
        for fill_mask in sorted(
            variants.values(),
            key=lambda m: (m.shape[0] * m.shape[1], m.shape[0], m.shape[1]),
            reverse=True,
        ):
            hh, ww = fill_mask.shape
            if hh > th - 2 or ww > tw - 2:
                continue
            for hr0 in range(1, th - hh):
                for hc0 in range(1, tw - ww):
                    window = sub[hr0 : hr0 + hh, hc0 : hc0 + ww]
                    if not bool(fill_mask.any()):
                        continue
                    if not bool(np.all((window == int(bg)) | (window == int(frame_col)))):
                        continue
                    if not np.array_equal(window == int(bg), fill_mask):
                        continue
                    if not _outer_ring_is_frame(sub, hr0, hc0, hh, ww, int(frame_col)):
                        continue
                    candidates.append((hh * ww, hr0, hc0, fill_mask, key))

    occupied = np.zeros_like(sub, dtype=bool)
    holes = []
    for _area, hr0, hc0, fill_mask, key in sorted(candidates, reverse=True):
        hh, ww = fill_mask.shape
        if bool(occupied[hr0 : hr0 + hh, hc0 : hc0 + ww].any()):
            continue
        occupied[hr0 : hr0 + hh, hc0 : hc0 + ww] = True
        holes.append((hr0, hc0, fill_mask, key))

    assert holes

    interior_bg = np.zeros_like(sub, dtype=bool)
    for coords in components_mask(sub == int(bg), connectivity=4):
        touches_border = bool(
            np.any(coords[:, 0] == 0)
            or np.any(coords[:, 0] == th - 1)
            or np.any(coords[:, 1] == 0)
            or np.any(coords[:, 1] == tw - 1)
        )
        if touches_border:
            continue
        interior_bg[coords[:, 0], coords[:, 1]] = True

    covered_bg = np.zeros_like(sub, dtype=bool)
    for hr0, hc0, fill_mask, _key in holes:
        hh, ww = fill_mask.shape
        covered_bg[hr0 : hr0 + hh, hc0 : hc0 + ww] |= fill_mask
    assert np.array_equal(covered_bg, interior_bg)

    out = sub.copy()
    for hr0, hc0, fill_mask, key in holes:
        cmap = sprites_by_key.get(key)
        assert cmap is not None
        scol = max(cmap.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        patch = np.full(fill_mask.shape, int(scol), dtype=int)
        paste(out, patch, int(hr0), int(hc0), mask=fill_mask)

    return out


NUM_COLORS = 10
GRID_SIZE = 30

FRAME_H_RANGE = (12, 13)
FRAME_W_RANGE = (18, 20)

SHAPE_COUNT_RANGE = (2, 4)
MIN_SHAPES = 2
MIN_TOTAL_HOLES = 5
TRUE_MULT_RANGE = (1, 3)
FAKE_SHAPE_COUNT_RANGE = (1, 2)
MIN_TOTAL_FAKE_WINDOWS = 2
FAKE_MULT_RANGE = (1, 3)
FRAME_DENSITY_RANGE = (0.55, 0.88)
PLACEMENT_RETRIES = 90
HOLE_MARGIN = 2
WINDOW_GAP = 1

TOP_BOTTOM_NOTCH_COUNT_RANGE = (1, 2)
LEFT_RIGHT_NOTCH_COUNT_RANGE = (1, 2)
NOTCH_LEN_RANGE = (1, 3)

WINNING_SPRITE_RANGE = (2, 5)
OTHER_COLOR_MAX = 1
SPRITE_PADDING = 1

MAX_GEN_ATTEMPTS = 800


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
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _component_count(mask: np.ndarray) -> int:
    return len(components_mask(mask.astype(bool), connectivity=4))


def _outer_ring_is_frame(sub: np.ndarray, r: int, c: int, h: int, w: int, frame_col: int) -> bool:
    ring = sub[r - 1 : r + h + 1, c - 1 : c + w + 1]
    mask = np.ones_like(ring, dtype=bool)
    mask[1:-1, 1:-1] = False
    return bool(np.all(ring[mask] == int(frame_col)))


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
        area = int(np.sum(m))
        if not (2 <= area <= 8):
            continue
        if not _touches_all_sides(m):
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


def _mirror_pos(r: int, c: int, h: int, w: int, th: int, tw: int):
    return list(
        {
            (r, c),
            (r, tw - w - c),
            (th - h - r, c),
            (th - h - r, tw - w - c),
        }
    )


def _mark_used(used: np.ndarray, r: int, c: int, h: int, w: int):
    rr0 = max(0, r - WINDOW_GAP)
    cc0 = max(0, c - WINDOW_GAP)
    rr1 = min(used.shape[0], r + h + WINDOW_GAP)
    cc1 = min(used.shape[1], c + w + WINDOW_GAP)
    used[rr0:rr1, cc0:cc1] = True


def _can_place(used: np.ndarray, r: int, c: int, h: int, w: int, th: int, tw: int) -> bool:
    if r < HOLE_MARGIN or c < HOLE_MARGIN:
        return False
    if r + h > th - HOLE_MARGIN or c + w > tw - HOLE_MARGIN:
        return False
    rr0 = max(0, r - WINDOW_GAP)
    cc0 = max(0, c - WINDOW_GAP)
    rr1 = min(th, r + h + WINDOW_GAP)
    cc1 = min(tw, c + w + WINDOW_GAP)
    return not bool(used[rr0:rr1, cc0:cc1].any())


def _try_place_quadrant(used: np.ndarray, shape: np.ndarray, th: int, tw: int):
    h, w = shape.shape
    r_max = th // 2 - h
    c_max = tw // 2 - w
    if r_max < HOLE_MARGIN or c_max < HOLE_MARGIN:
        return None
    for _ in range(PLACEMENT_RETRIES):
        r = random.randint(HOLE_MARGIN, r_max)
        c = random.randint(HOLE_MARGIN, c_max)
        positions = _mirror_pos(r, c, h, w, th, tw)
        if all(_can_place(used, pr, pc, h, w, th, tw) for pr, pc in positions):
            return positions
    return None


def _try_place_hcenter(used: np.ndarray, shape: np.ndarray, th: int, tw: int):
    if th % 2 == 0:
        return None
    h, w = shape.shape
    r = th // 2 - h // 2
    if r < HOLE_MARGIN or r + h > th - HOLE_MARGIN:
        return None
    c_max = tw // 2 - w
    if c_max < HOLE_MARGIN:
        return None
    for _ in range(PLACEMENT_RETRIES):
        c = random.randint(HOLE_MARGIN, c_max)
        positions = list({(r, c), (r, tw - w - c)})
        if all(_can_place(used, pr, pc, h, w, th, tw) for pr, pc in positions):
            return positions
    return None


def _try_place_vcenter(used: np.ndarray, shape: np.ndarray, th: int, tw: int):
    if tw % 2 == 0:
        return None
    h, w = shape.shape
    c = tw // 2 - w // 2
    if c < HOLE_MARGIN or c + w > tw - HOLE_MARGIN:
        return None
    r_max = th // 2 - h
    if r_max < HOLE_MARGIN:
        return None
    for _ in range(PLACEMENT_RETRIES):
        r = random.randint(HOLE_MARGIN, r_max)
        positions = list({(r, c), (th - h - r, c)})
        if all(_can_place(used, pr, pc, h, w, th, tw) for pr, pc in positions):
            return positions
    return None


def _try_place_center(used: np.ndarray, shape: np.ndarray, th: int, tw: int):
    if th % 2 == 0 or tw % 2 == 0:
        return None
    h, w = shape.shape
    r = th // 2 - h // 2
    c = tw // 2 - w // 2
    if not _can_place(used, r, c, h, w, th, tw):
        return None
    return [(r, c)]


def _try_place_group(used: np.ndarray, shape: np.ndarray, th: int, tw: int):
    placers = []
    if th // 2 - shape.shape[0] >= HOLE_MARGIN and tw // 2 - shape.shape[1] >= HOLE_MARGIN:
        placers.extend([_try_place_quadrant, _try_place_quadrant, _try_place_quadrant])
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


def _try_place_single(used: np.ndarray, shape: np.ndarray, th: int, tw: int):
    h, w = shape.shape
    r_max = th - HOLE_MARGIN - h
    c_max = tw - HOLE_MARGIN - w
    if r_max < HOLE_MARGIN or c_max < HOLE_MARGIN:
        return None
    for _ in range(PLACEMENT_RETRIES * 2):
        r = random.randint(HOLE_MARGIN, r_max)
        c = random.randint(HOLE_MARGIN, c_max)
        if _can_place(used, r, c, h, w, th, tw):
            return int(r), int(c)
    return None


def _build_layout(th: int, tw: int):
    max_h = min(5, th - 2 * HOLE_MARGIN)
    max_w = min(6, tw - 2 * HOLE_MARGIN)
    available = [p for p in PATTERN_LIBRARY if p.shape[0] <= max_h and p.shape[1] <= max_w]
    if len(available) < MIN_SHAPES:
        return None

    used = np.zeros((th, tw), dtype=bool)
    true_entries = []
    true_target = random.randint(SHAPE_COUNT_RANGE[0], min(SHAPE_COUNT_RANGE[1], len(available)))
    for base in random.sample(available, k=len(available)):
        if len(true_entries) >= true_target:
            break
        shape = random.choice(_all_orientations(base))
        mult_target = random.randint(*TRUE_MULT_RANGE)
        positions = []
        for _ in range(mult_target):
            pos = _try_place_single(used, shape, th, tw)
            if pos is None:
                break
            pr, pc = pos
            _mark_used(used, pr, pc, shape.shape[0], shape.shape[1])
            positions.append((pr, pc))
        if len(positions) < 1:
            continue
        true_entries.append((shape, positions))

    if len(true_entries) < MIN_SHAPES:
        return None

    total_holes = sum(len(pos_list) for _shape, pos_list in true_entries)
    if total_holes < MIN_TOTAL_HOLES:
        return None

    fake_entries = []
    fake_target = random.randint(*FAKE_SHAPE_COUNT_RANGE)
    for base in random.sample(available, k=len(available)):
        if len(fake_entries) >= fake_target:
            break
        shape = random.choice(_all_orientations(base))
        mult_target = random.randint(*FAKE_MULT_RANGE)
        positions = []
        for _ in range(mult_target):
            pos = _try_place_single(used, shape, th, tw)
            if pos is None:
                break
            pr, pc = pos
            _mark_used(used, pr, pc, shape.shape[0], shape.shape[1])
            positions.append((pr, pc))
        if len(positions) < 1:
            continue
        fake_entries.append((shape, positions))

    total_fake = sum(len(pos_list) for _shape, pos_list in fake_entries)
    if total_fake < MIN_TOTAL_FAKE_WINDOWS:
        return None

    if not any(_component_count(shape) >= 2 for shape, _pos_list in true_entries):
        return None

    return true_entries, fake_entries


def _add_notches(th: int, tw: int):
    notch = np.zeros((th, tw), dtype=bool)

    def carve_top_bottom(c0: int, length: int):
        notch[0, c0 : c0 + length] = True
        notch[th - 1, c0 : c0 + length] = True

    def carve_left_right(r0: int, length: int):
        notch[r0 : r0 + length, 0] = True
        notch[r0 : r0 + length, tw - 1] = True

    if tw >= 5:
        max_len = min(NOTCH_LEN_RANGE[1], max(1, tw - 4))
        n_tb = random.randint(*TOP_BOTTOM_NOTCH_COUNT_RANGE)
        used_cols: list[tuple[int, int]] = []
        for _ in range(n_tb):
            for _ in range(40):
                length = random.randint(NOTCH_LEN_RANGE[0], max_len)
                if tw - length - 1 <= 1:
                    continue
                c0 = random.randint(1, tw - length - 1)
                if any(not (c0 + length <= a or b <= c0) for a, b in used_cols):
                    continue
                carve_top_bottom(c0, length)
                used_cols.append((c0, c0 + length))
                break

    if th >= 5:
        max_len = min(NOTCH_LEN_RANGE[1], max(1, th - 4))
        n_lr = random.randint(*LEFT_RIGHT_NOTCH_COUNT_RANGE)
        used_rows: list[tuple[int, int]] = []
        for _ in range(n_lr):
            for _ in range(40):
                length = random.randint(NOTCH_LEN_RANGE[0], max_len)
                if th - length - 1 <= 1:
                    continue
                r0 = random.randint(1, th - length - 1)
                if any(not (r0 + length <= a or b <= r0) for a, b in used_rows):
                    continue
                carve_left_right(r0, length)
                used_rows.append((r0, r0 + length))
                break

    if not bool(notch.any()):
        return None
    return notch


def _render_input(
    th: int,
    tw: int,
    frame_col: int,
    bg: int,
    notch_mask: np.ndarray,
    true_info,
    fake_info,
):
    canvas = np.full((th, tw), int(frame_col), dtype=int)
    canvas[notch_mask] = int(bg)
    for shape, _fill_color, pos_list in true_info:
        patch = np.full(shape.shape, int(frame_col), dtype=int)
        patch[shape] = int(bg)
        for pr, pc in pos_list:
            paste(canvas, patch, int(pr), int(pc))
    for shape, fake_color, pos_list in fake_info:
        patch = np.full(shape.shape, int(frame_col), dtype=int)
        patch[shape] = int(fake_color)
        for pr, pc in pos_list:
            paste(canvas, patch, int(pr), int(pc))
    return canvas


def _render_output(
    th: int,
    tw: int,
    frame_col: int,
    bg: int,
    notch_mask: np.ndarray,
    true_info,
    fake_info,
):
    canvas = np.full((th, tw), int(frame_col), dtype=int)
    canvas[notch_mask] = int(bg)
    for shape, fill_color, pos_list in true_info:
        patch = np.full(shape.shape, int(frame_col), dtype=int)
        patch[shape] = int(fill_color)
        for pr, pc in pos_list:
            paste(canvas, patch, int(pr), int(pc))
    for shape, fake_color, pos_list in fake_info:
        patch = np.full(shape.shape, int(frame_col), dtype=int)
        patch[shape] = int(fake_color)
        for pr, pc in pos_list:
            paste(canvas, patch, int(pr), int(pc))
    return canvas


def _place_sprite(
    inp: np.ndarray,
    placement: np.ndarray,
    ori: np.ndarray,
    color: int,
    bg: int,
    frame_col: int,
):
    patch = np.full(ori.shape, int(frame_col), dtype=int)
    patch[ori] = int(color)
    try:
        r, c = random_free_location(placement, patch, transparent=bg, padding=SPRITE_PADDING)
    except ValueError:
        return False
    paste(placement, patch, int(r), int(c), transparent=bg)
    paste(inp, patch, int(r), int(c), transparent=bg)
    return True


def _scatter_sprites(inp: np.ndarray, placement: np.ndarray, shape_info, bg: int, frame_col: int, loser_pool):
    for shape, fill_color, _pos_list in shape_info:
        orientations = _all_orientations(shape)
        n_winning = random.randint(*WINNING_SPRITE_RANGE)
        for _ in range(n_winning):
            if not _place_sprite(inp, placement, random.choice(orientations), fill_color, bg, frame_col):
                return False

        available_losers = [c for c in loser_pool if c != fill_color]
        remaining_budget = n_winning - 1
        max_other = min(OTHER_COLOR_MAX, len(available_losers), remaining_budget)
        n_other = random.randint(0, max_other) if max_other > 0 else 0
        oc_list = random.sample(available_losers, n_other) if n_other > 0 else []

        for idx, oc in enumerate(oc_list):
            max_for_color = remaining_budget - max(0, len(oc_list) - idx - 1)
            n_oc = random.randint(1, max_for_color)
            remaining_budget -= n_oc
            for _ in range(n_oc):
                if not _place_sprite(inp, placement, random.choice(orientations), oc, bg, frame_col):
                    break
    return True


def _count_notch_components(sub: np.ndarray, bg: int) -> int:
    count = 0
    th, tw = sub.shape
    for coords in components_mask(sub == int(bg), connectivity=4):
        if bool(
            np.any(coords[:, 0] == 0)
            or np.any(coords[:, 0] == th - 1)
            or np.any(coords[:, 1] == 0)
            or np.any(coords[:, 1] == tw - 1)
        ):
            count += 1
    return count


def _gen_attempt() -> np.ndarray | None:
    bg = random.randint(0, NUM_COLORS - 1)
    frame_col = random.choice([c for c in range(NUM_COLORS) if c != bg])
    th = random.randint(*FRAME_H_RANGE)
    tw = random.randint(*FRAME_W_RANGE)

    layout = _build_layout(th, tw)
    if layout is None:
        return None
    true_entries, fake_entries = layout

    notch_mask = _add_notches(th, tw)
    if notch_mask is None or int(notch_mask.sum()) == 0:
        return None

    avail_colors = [c for c in range(NUM_COLORS) if c not in (bg, frame_col)]
    if len(avail_colors) < len(true_entries) + 1:
        return None
    random.shuffle(avail_colors)
    true_colors = avail_colors[: len(true_entries)]
    fake_color = avail_colors[len(true_entries)]
    loser_pool = [c for c in avail_colors[len(true_entries) + 1 :] if c != fake_color]

    true_info = [(shape, int(fill_color), list(pos_list)) for (shape, pos_list), fill_color in zip(true_entries, true_colors)]
    fake_info = [(shape, int(fake_color), list(pos_list)) for shape, pos_list in fake_entries]

    input_frame = _render_input(th, tw, frame_col, bg, notch_mask, true_info, fake_info)
    frame_density = float(np.mean(input_frame == int(frame_col)))
    if not (FRAME_DENSITY_RANGE[0] <= frame_density <= FRAME_DENSITY_RANGE[1]):
        return None

    output = _render_output(th, tw, frame_col, bg, notch_mask, true_info, fake_info)

    if _count_notch_components(input_frame, bg) == 0:
        return None
    if not bool(((input_frame != int(bg)) & (input_frame != int(frame_col))).any()):
        return None

    inp = np.full((GRID_SIZE, GRID_SIZE), int(bg), dtype=int)
    fr = random.randint(0, GRID_SIZE - th)
    fc = random.randint(0, GRID_SIZE - tw)
    paste(inp, input_frame, fr, fc)

    placement = inp.copy()
    placement[fr : fr + th, fc : fc + tw] = int(frame_col)
    if not _scatter_sprites(inp, placement, true_info, bg, frame_col, loser_pool):
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
