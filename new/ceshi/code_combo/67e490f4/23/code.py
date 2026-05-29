# Combo variant of 67e490f4 (index 23)
# combo_spec: D7/3+D9/0
# Name: 分区洞+图案搬运

from __future__ import annotations

import random

import numpy as np

from arc_utils import crop_bbox, find_components, get_bg, paste, random_free_location


# ---------------------------------------------------------------------------
# Design analysis for combo D7/3 + D9/0
#
# a) Concrete behavior:
#    The original task had plain bg holes inside a large frame, and outside
#    sprites only voted for one fill color per hole shape. This combo changes
#    both sides of that contract.
#    - D7/3: each "hole" is now a small isolated window whose interior mixes
#      bg cells with frame-colored divider strips. The divider splits the
#      writable area into multiple visual subregions, but matching still uses
#      the whole window shape.
#    - D9/0: the outside sprite is no longer summarized by one color. Its full
#      aligned pattern is copied into the hole, but only onto the writable bg
#      cells; the divider strip remains frame-colored in the output.
#    Tiny trace:
#        input hole window      matched outside sprite      output window
#        7 0 7                 7 2 7                      7 2 7
#        7 0 0       match     7 3 4          stamp       7 3 4
#    The middle/top divider cell stays 7. Only former bg cells receive 2/3/4.
#
# b) Complexity calibration:
#    If the winning sprite were monochrome, the task would collapse toward the
#    D7-only variant. If holes were connected, it would collapse toward D9-only.
#    The generator therefore enforces both:
#    - every library hole mask has at least two 4-connected bg components;
#    - every winning sprite uses at least two distinct fill colors.
#    Shape count stays at 3-5 with repeated sprite evidence, so the rule is
#    visible without creating too many one-off correspondences.
#
# c) Hidden assumptions from the original solver that break:
#    - `components(sub != frame_col)` no longer returns one component per hole;
#      divider strips split a hole into several bg islands.
#    - "one shape -> one scalar color" no longer holds; exact aligned patch
#      frequency matters.
#    - Pasting onto the entire hole bbox would incorrectly overwrite divider
#      strips. The combo solver must only paste onto the writable mask cells.
#    The new solver therefore detects isolated rectangular windows, not bg
#    components, and it buckets full sprite patches by oriented window key.
#
# d) Placement legality:
#    The generator reserves whole hole bboxes with a one-cell frame ring and a
#    gap between windows, so no two holes overlap or touch. Sprites are placed
#    fully outside the frame bbox with Chebyshev padding >= 1, so multicolor
#    sprite components do not merge. Every fill mask touches all four bbox
#    sides, making the intended hole window bbox recoverable.
#
# e) Visual observability:
#    The divider strips are visible in both input and output, and they survive
#    the transformation. The copied sprite patterns are multicolor, so the
#    result is visibly "stamped through partitions" rather than flat-filled.
#    A human can see that the same exterior motif reappears inside matching
#    windows, clipped by the divider strips.
#
# f) Visual cleanliness:
#    Only three layers exist in the frame area: bg writable cells, frame-colored
#    divider/border cells, and stamped colors. There is no extra annotation.
#    Outside sprites are isolated full rectangles using the same divider style,
#    which keeps the correspondence readable.
#
# g) Inferability of the change:
#    No hidden color ordering is used. The winner is determined by repeated
#    identical sprite patterns, and every winning pattern is multicolor. That
#    means a flat-color rule disagrees with the generated targets, while the
#    partitioned-window rule is directly observable from the surviving divider
#    strips.
#
# h) Anti-degeneration criterion used later in testing:
#    On generated samples, all detected holes must be partitioned
#    (`component_count(fill_mask) >= 2`) and at least one hole per sample must
#    remain multicolor after stamping. We also compare the combo output against
#    a flat-fill baseline; equality would indicate regression toward D7-only.
#
# i) New solver assumptions audited against the generator:
#    - Assumption: every true hole window is surrounded by a one-cell ring of
#      frame color. Generator: enforced by bbox placement with border >= 1.
#    - Assumption: exact oriented hole keys can be recovered from outside
#      sprites. Generator: only uses masks whose 8 dihedral transforms are all
#      distinct, so alignment is unambiguous.
#    - Assumption: each oriented key has a strict plurality winning patch.
#      Generator: winning sprite count is always greater than each losing patch
#      count for that shape.
#    - Assumption: all non-frame cells inside the frame belong to detected hole
#      windows. Generator: every hole window touches all four sides, and oracle
#      validation rejects any sample where this coverage invariant fails.
# ---------------------------------------------------------------------------


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

PATTERN_POOL_RANGE = (4, 5)
WINNING_SPRITE_RANGE = (3, 5)
LOSING_PATTERN_MAX = 2
LOSING_SPRITE_RANGE = (1, 2)
SPRITE_PADDING = 1
PATTERN_RETRIES = 80

MAX_GEN_ATTEMPTS = 400


def _all_rot_flip_masks(mask: np.ndarray):
    m0 = mask.astype(bool)
    for flip in (False, True):
        m1 = np.flipud(m0) if flip else m0
        for k in range(4):
            m2 = np.rot90(m1, k)
            yield (m2.shape, m2.tobytes()), m2


def _all_rot_flip_patches(patch: np.ndarray):
    seen = set()
    p0 = np.array(patch, dtype=int)
    for flip in (False, True):
        p1 = np.flipud(p0) if flip else p0
        for k in range(4):
            p2 = np.rot90(p1, k)
            sig = (p2.shape, p2.tobytes())
            if sig in seen:
                continue
            seen.add(sig)
            yield sig, p2


def _canonical_key(mask: np.ndarray):
    return min(key for key, _ in _all_rot_flip_masks(mask))


def _canonical_patch_signature(patch: np.ndarray):
    return min(sig for sig, _ in _all_rot_flip_patches(patch))


def _orientation_count(mask: np.ndarray) -> int:
    return len({key for key, _ in _all_rot_flip_masks(mask)})


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(
        mask[0, :].any()
        and mask[-1, :].any()
        and mask[:, 0].any()
        and mask[:, -1].any()
    )


def _component_count(mask: np.ndarray) -> int:
    mask = mask.astype(bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
    return count


def _outer_ring_is_frame(sub: np.ndarray, r: int, c: int, h: int, w: int, frame_col: int) -> bool:
    ring = sub[r - 1 : r + h + 1, c - 1 : c + w + 1]
    mask = np.ones_like(ring, dtype=bool)
    mask[1:-1, 1:-1] = False
    return bool(np.all(ring[mask] == int(frame_col)))


def _build_pattern_library():
    raw = [
        [[1, 0, 1], [1, 0, 0]],
        [[0, 1, 1], [1, 0, 0]],
        [[1, 0, 1], [1, 1, 0]],
        [[1, 0, 0, 1], [1, 0, 0, 0]],
        [[0, 1, 0, 1], [1, 0, 0, 0]],
        [[1, 1, 0, 1], [1, 0, 0, 0]],
        [[0, 0, 1, 1], [1, 0, 0, 0]],
        [[1, 0, 1, 1], [1, 0, 0, 0]],
        [[0, 1, 1, 1], [1, 0, 0, 0]],
        [[1, 0, 0, 1], [0, 1, 0, 0]],
        [[1, 1, 0, 1], [0, 1, 0, 0]],
        [[1, 0, 1, 1], [0, 1, 0, 0]],
        [[1, 0, 0, 1], [1, 0, 1, 0]],
        [[1, 1, 0, 1], [1, 0, 1, 0]],
        [[1, 0, 0, 1], [1, 1, 1, 0]],
        [[1, 0, 1, 1], [1, 1, 0, 0]],
    ]
    seen = set()
    library = []
    for pat in raw:
        mask = np.array(pat, dtype=bool)
        area = int(np.sum(mask))
        if not (2 <= area <= 8):
            continue
        if not _touches_all_sides(mask):
            continue
        if _component_count(mask) < 2:
            continue
        if _orientation_count(mask) != 8:
            continue
        key = _canonical_key(mask)
        if key in seen:
            continue
        seen.add(key)
        library.append(mask)
    return library


PATTERN_LIBRARY = _build_pattern_library()
VALID_HOLE_KEYS = {
    key for mask in PATTERN_LIBRARY for key, _ in _all_rot_flip_masks(mask)
}


def _detect_holes(sub: np.ndarray, frame_col: int, valid_keys) -> list[tuple[int, int, np.ndarray, tuple]]:
    th, tw = sub.shape
    candidate_sizes = sorted({key[0] for key in valid_keys}, key=lambda s: (s[0] * s[1], s[0], s[1]), reverse=True)
    candidates = []
    for hh, ww in candidate_sizes:
        if hh > th - 2 or ww > tw - 2:
            continue
        for hr0 in range(1, th - hh):
            for hc0 in range(1, tw - ww):
                window = sub[hr0 : hr0 + hh, hc0 : hc0 + ww]
                fill_mask = window != int(frame_col)
                if not bool(fill_mask.any()):
                    continue
                if not _touches_all_sides(fill_mask):
                    continue
                if _component_count(fill_mask) < 2:
                    continue
                if not _outer_ring_is_frame(sub, hr0, hc0, hh, ww, int(frame_col)):
                    continue
                key = (fill_mask.shape, fill_mask.tobytes())
                if key not in valid_keys:
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

    covered = np.zeros_like(sub, dtype=bool)
    for hr0, hc0, fill_mask, _key in holes:
        hh, ww = fill_mask.shape
        covered[hr0 : hr0 + hh, hc0 : hc0 + ww] |= fill_mask
    assert np.array_equal(covered, sub != int(frame_col))
    return holes


def _extract_sprite_catalog(grid: np.ndarray, frame_bbox, bg: int, frame_col: int):
    r0, c0, r1, c1 = map(int, frame_bbox)
    outside = np.array(grid, dtype=int)
    outside[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
    spr_comps = find_components(outside, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    sprites_by_key: dict[tuple, dict[tuple, list[object]]] = {}
    for comp in spr_comps:
        sr0, sc0, sr1, sc1 = map(int, comp["bbox"])
        patch = outside[sr0 : sr1 + 1, sc0 : sc1 + 1]
        fill_mask = (patch != int(bg)) & (patch != int(frame_col))
        if not bool(fill_mask.any()):
            continue
        if _component_count(fill_mask) < 2 or not _touches_all_sides(fill_mask):
            continue

        variants_by_key: dict[tuple, dict[tuple, np.ndarray]] = {}
        for sig, oriented_patch in _all_rot_flip_patches(patch):
            oriented_fill = (oriented_patch != int(bg)) & (oriented_patch != int(frame_col))
            key = (oriented_fill.shape, oriented_fill.tobytes())
            variants_by_key.setdefault(key, {})[sig] = oriented_patch

        for key, sig_map in variants_by_key.items():
            assert len(sig_map) == 1
            sig, aligned_patch = next(iter(sig_map.items()))
            bucket = sprites_by_key.setdefault(key, {})
            if sig not in bucket:
                bucket[sig] = [0, aligned_patch]
            bucket[sig][0] = int(bucket[sig][0]) + 1

    assert sprites_by_key
    return sprites_by_key


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy patterned outside stamps into partitioned holes inside a frame.

    Input structure:
    - The most common color is the background. One large monochrome frame object encloses the target area.
    - Inside the frame bbox, each target is a small isolated window whose writable cells are non-frame gaps,
      while some internal cells remain in the frame color as divider strips.
    - Outside the frame, separated multicolor sprites show those same window layouts; divider cells use the
      frame color and only the writable cells carry the copied pattern colors.

    Rule:
    - Crop to the frame bbox and begin with a solid frame-colored canvas.
    - Detect each internal window by its full writable-mask shape, match it to outside sprites up to
      rotation/flip, and identify the most frequent aligned full sprite patch for that exact window orientation.
    - Paste only the writable cells of that winning patch into the window; leave divider-strip cells untouched.

    Key insight:
    The exterior sprites are literal stamps for whole partitioned windows, not simple color votes. The frame-
    colored divider belongs to the template and survives, so the copied pattern appears clipped into multiple
    subregions inside the hole.
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

    sub = crop_bbox(g, fb)
    th, tw = sub.shape

    sprites_by_key = _extract_sprite_catalog(g, fb, int(bg), int(frame_col))
    holes = _detect_holes(sub, int(frame_col), sprites_by_key.keys())
    assert holes

    out = np.full((th, tw), int(frame_col), dtype=int)
    for hr0, hc0, fill_mask, key in holes:
        patch_map = sprites_by_key.get(key)
        assert patch_map is not None
        best_count = max(int(info[0]) for info in patch_map.values())
        winners = [info[1] for info in patch_map.values() if int(info[0]) == best_count]
        assert len(winners) == 1
        paste(out, winners[0], int(hr0), int(hc0), mask=fill_mask)

    return out


def _all_patch_orientations(patch: np.ndarray):
    return [p for _sig, p in _all_rot_flip_patches(patch)]


def _mirror_pos(r: int, c: int, h: int, w: int, th: int, tw: int):
    return list({(r, c), (r, tw - w - c), (th - h - r, c), (th - h - r, tw - w - c)})


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
    h, w = shape.shape
    r_max = th // 2 - h
    c_max = tw // 2 - w
    if r_max < 1 or c_max < 1:
        return None
    for _ in range(PLACEMENT_RETRIES):
        r = random.randint(1, r_max)
        c = random.randint(1, c_max)
        positions = _mirror_pos(r, c, h, w, th, tw)
        if all(_can_place(used, pr, pc, h, w, th, tw) for pr, pc in positions):
            return positions
    return None


def _try_place_hcenter(used: np.ndarray, shape: np.ndarray, th: int, tw: int):
    if th % 2 == 0:
        return None
    h, w = shape.shape
    r = th // 2 - h // 2
    if r < 1 or r + h > th - 1:
        return None
    c_max = tw // 2 - w
    if c_max < 1:
        return None
    for _ in range(PLACEMENT_RETRIES):
        c = random.randint(1, c_max)
        positions = list({(r, c), (r, tw - w - c)})
        if len(positions) == 2 and all(_can_place(used, pr, pc, h, w, th, tw) for pr, pc in positions):
            return positions
    return None


def _try_place_vcenter(used: np.ndarray, shape: np.ndarray, th: int, tw: int):
    if tw % 2 == 0:
        return None
    h, w = shape.shape
    c = tw // 2 - w // 2
    if c < 1 or c + w > tw - 1:
        return None
    r_max = th // 2 - h
    if r_max < 1:
        return None
    for _ in range(PLACEMENT_RETRIES):
        r = random.randint(1, r_max)
        positions = list({(r, c), (th - h - r, c)})
        if len(positions) == 2 and all(_can_place(used, pr, pc, h, w, th, tw) for pr, pc in positions):
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
    if th // 2 - shape.shape[0] >= 1 and tw // 2 - shape.shape[1] >= 1:
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


def _build_hole_layout(th: int, tw: int):
    max_h = min(5, th - 2)
    max_w = min(6, tw - 2)
    available = [shape for shape in PATTERN_LIBRARY if shape.shape[0] <= max_h and shape.shape[1] <= max_w]
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

    total_holes = sum(len(pos_list) for _shape, pos_list in placements)
    if total_holes < MIN_TOTAL_HOLES:
        return None

    fill_area = sum(int(np.sum(shape)) * len(pos_list) for shape, pos_list in placements)
    frame_density = 1.0 - fill_area / float(th * tw)
    if not (FRAME_DENSITY_RANGE[0] <= frame_density <= FRAME_DENSITY_RANGE[1]):
        return None

    return placements


def _random_pattern(fill_mask: np.ndarray, palette: list[int], frame_col: int, banned: set[tuple]):
    cells = [tuple(map(int, rc)) for rc in np.argwhere(fill_mask)]
    max_colors = min(3, len(cells), len(palette))
    if max_colors < 2:
        return None

    for _ in range(PATTERN_RETRIES):
        n_cols = random.randint(2, max_colors)
        cols = random.sample(palette, n_cols)
        patch = np.full(fill_mask.shape, int(frame_col), dtype=int)
        random.shuffle(cells)
        for (rr, cc), col in zip(cells[:n_cols], cols):
            patch[rr, cc] = int(col)
        for rr, cc in cells[n_cols:]:
            patch[rr, cc] = int(random.choice(cols))

        fill_vals = [int(patch[rr, cc]) for rr, cc in cells]
        if len(set(fill_vals)) < 2:
            continue
        sig = _canonical_patch_signature(patch)
        if sig in banned:
            continue
        return patch, sig
    return None


def _build_output(th: int, tw: int, placements, frame_col: int, bg: int):
    avail_colors = [c for c in range(NUM_COLORS) if c not in (bg, frame_col)]
    palette_size = random.randint(PATTERN_POOL_RANGE[0], min(PATTERN_POOL_RANGE[1], len(avail_colors)))
    palette = random.sample(avail_colors, palette_size)

    output = np.full((th, tw), int(frame_col), dtype=int)
    shape_info = []
    for shape, pos_list in placements:
        built = _random_pattern(shape, palette, int(frame_col), set())
        if built is None:
            return None
        winning_patch, winning_sig = built

        banned = {winning_sig}
        losing_patches = []
        n_losing = random.randint(1, LOSING_PATTERN_MAX)
        for _ in range(n_losing):
            alt = _random_pattern(shape, palette, int(frame_col), banned)
            if alt is None:
                return None
            losing_patch, losing_sig = alt
            banned.add(losing_sig)
            losing_patches.append(losing_patch)

        for pr, pc in pos_list:
            paste(output, winning_patch, pr, pc, mask=shape)
        shape_info.append((shape, winning_patch, losing_patches, list(pos_list)))

    return output, shape_info


def _place_sprite(inp: np.ndarray, placement: np.ndarray, patch: np.ndarray, bg: int):
    try:
        r, c = random_free_location(placement, patch, transparent=bg, padding=SPRITE_PADDING)
    except ValueError:
        return False
    paste(placement, patch, r, c, transparent=bg)
    paste(inp, patch, r, c, transparent=bg)
    return True


def _scatter_sprites(inp: np.ndarray, shape_info, bg: int):
    placement = inp.copy()
    for _shape, winning_patch, losing_patches, _positions in shape_info:
        winning_orients = _all_patch_orientations(winning_patch)
        n_winning = random.randint(*WINNING_SPRITE_RANGE)
        for _ in range(n_winning):
            if not _place_sprite(inp, placement, random.choice(winning_orients), bg):
                return False

        for losing_patch in losing_patches:
            losing_orients = _all_patch_orientations(losing_patch)
            n_losing = random.randint(LOSING_SPRITE_RANGE[0], min(LOSING_SPRITE_RANGE[1], n_winning - 1))
            placed = 0
            for _ in range(n_losing):
                if not _place_sprite(inp, placement, random.choice(losing_orients), bg):
                    break
                placed += 1
            if placed == 0:
                return False
    return True


def _gen_attempt() -> np.ndarray | None:
    bg = random.randint(0, NUM_COLORS - 1)
    frame_col = random.choice([c for c in range(NUM_COLORS) if c != bg])
    th = random.randint(*FRAME_H_RANGE)
    tw = random.randint(*FRAME_W_RANGE)

    placements = _build_hole_layout(th, tw)
    if placements is None:
        return None

    built = _build_output(th, tw, placements, int(frame_col), int(bg))
    if built is None:
        return None
    output, shape_info = built

    template = np.full((th, tw), int(frame_col), dtype=int)
    for shape, _winning_patch, _losing_patches, pos_list in shape_info:
        for pr, pc in pos_list:
            hole_patch = np.full(shape.shape, int(bg), dtype=int)
            paste(template, hole_patch, pr, pc, mask=shape)

    inp = np.full((GRID_SIZE, GRID_SIZE), int(bg), dtype=int)
    fr = random.randint(0, GRID_SIZE - th)
    fc = random.randint(0, GRID_SIZE - tw)
    paste(inp, template, fr, fc)

    if not _scatter_sprites(inp, shape_info, int(bg)):
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
