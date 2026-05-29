# Combo variant of b10624e5 (index 42)
# combo_spec: D5/0+D10/0+D12/3
# Name: 锚块颜色编码变换类型 + 旋转替代反射 + 边缘像素编码旋转角度

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


"""
Design analysis for the combo variant.

a) Concrete behavior and trace
   - The 27x27 board, divider cross, TL source motif, and anchor-size scaling all stay from the
     original task.
   - The changed part is now hierarchical:
       red 2x2 TL anchor      -> source only
       green target anchor    -> reflection family (legacy quadrant-based mirror)
       teal target anchor     -> rotation family; an attached black edge ray gives 90/180/270
   - Example trace:
       source patch = asymmetric 6x4 motif with the red 2x2 block near its upper-left.
       Original TR target: mirror left-right because it is on the right.
       Combo TR target:
         if its anchor is green, still mirror left-right;
         if its anchor is teal with a 3-pixel ray, rotate 270 degrees clockwise instead.
       So the solver must first read anchor color to choose the transform family, and only then
       count the ray length when the family is rotation.

b) Complexity calibration
   - A fully free color->transform dictionary would be too hard to infer from a few ARC examples.
   - This combo therefore uses only two target families:
       green(3) = reflection family
       teal(8)  = rotation family
   - Every generated sample has exactly one green target anchor and exactly two teal target anchors.
     All three target anchors also carry the same style of black edge ray, so "marker present"
     alone is not enough; color must still be read first.
   - The two teal targets use different marker counts, while the green target always uses a fixed
     1-pixel ray that is visually similar but semantically inert. This keeps the hierarchy active:
     color chooses the family, then count refines only the rotation branch.

c) Hidden assumptions that change in main()
   - Original anchor search assumed every anchor was red. This now fails because target anchors can
     be green or teal, so anchor extraction must scan a fixed anchor-color codebook.
   - Original transform selection depended only on quadrant position. That is now only valid for
     the green reflection family; teal anchors must branch into marker-decoded rotations.
   - Original stamping skipped only red cells. That would overwrite green/teal anchors and black
     marker rays, so stamping now writes only onto background cells in target quadrants.
   - Marker decoding assumes a teal target quadrant contains exactly one anchor-plus-marker
     component and no extra paint in the input. The generator enforces that by clearing non-TL
     quadrants before re-placing anchors and markers.

d) Placement legality
   - The generator computes the transformed+scaled patch before choosing anchor positions.
   - Green anchors are placed where the full reflected patch fits inside the quadrant.
   - Teal anchors are placed where both the full rotated patch and the whole marker ray fit.
   - Marker side is chosen only from sides that stay disjoint from the future stamped motif, so
     anchor, symbol, and output paint do not overlap.

e) Visual observability
   - Color is behaviorally active because each sample mixes green and teal target anchors that can
     even share the same 1-pixel ray length while producing different transforms.
   - Marker counts are behaviorally active because the two teal targets use different counts.
   - The source motif is filtered to be asymmetric under flips and quarter-turn rotations, and the
     realized rotated outputs are checked against the legacy all-reflection rule so the combo does
     not collapse back into the original puzzle.

f) Visual cleanliness
   - Motif colors exclude the divider color, both target-anchor colors, and the marker color.
   - Non-TL input quadrants contain only an isolated square anchor plus one short 1-cell-thick ray.
     The output overlays exactly one transformed motif per quadrant.

g) Inferability of the change
   - The semantic load is split across visibly different channels:
       size   -> scale
       color  -> transform family
       count  -> rotation angle, only for the rotation family
   - That is the minimum needed to make D10 and D12 both non-degenerate on the same sample without
     introducing a large arbitrary color vocabulary.
"""


BG = 4
DIV = 1
SOURCE_ANCHOR = 2
REFLECT_ANCHOR = 3
MARK = 0
ROTATE_ANCHOR = 8

GRID_SIZE = 27
MID = 13
QUADS = {
    (0, 0): (0, 12, 0, 12),
    (0, 1): (0, 12, 14, 26),
    (1, 0): (14, 26, 0, 12),
    (1, 1): (14, 26, 14, 26),
}
TARGET_KEYS = ((0, 1), (1, 0), (1, 1))
ANCHOR_COLORS = (SOURCE_ANCHOR, REFLECT_ANCHOR, ROTATE_ANCHOR)
TRANSFORM_KIND_BY_COLOR = {
    REFLECT_ANCHOR: "reflect",
    ROTATE_ANCHOR: "rotate",
}
MARKER_SIDES = ("top", "bottom", "left", "right")
REFLECT_MARK_LEN = 1

SOURCE_ANCHOR_SIZE = 2
BLOCK_DIMS = [(2, 3), (3, 2), (3, 3)]
FILL_RATIO_MIN = 0.5
FILL_RATIO_MAX = 0.8
MOTIF_COLORS_MIN = 3
MOTIF_COLORS_MAX = 5
MOTIF_EXTRA_COLORS = [5, 6, 7, 9]
TARGET_SIZE_RANGE = (1, 4)
MAX_RETRIES = 700


def _find_dividers(g: np.ndarray, bg: int) -> tuple[int, int]:
    """Return the unique full-row/full-col divider intersection."""
    h, w = g.shape
    full_rows = [(r, int(g[r, 0])) for r in range(h) if np.all(g[r, :] == g[r, 0]) and int(g[r, 0]) != bg]
    full_cols = [(c, int(g[0, c])) for c in range(w) if np.all(g[:, c] == g[0, c]) and int(g[0, c]) != bg]
    matches = []
    for r, vr in full_rows:
        for c, vc in full_cols:
            if vr == vc:
                matches.append((int(r), int(c)))
    assert len(matches) == 1, "grid must have exactly one matching full-row/full-col divider"
    return matches[0]


def _resize_nn(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    src_h, src_w = img.shape
    if new_h <= 0 or new_w <= 0:
        return img.copy()
    rr = np.clip(np.floor(np.arange(new_h) * src_h / new_h).astype(int), 0, src_h - 1)
    cc = np.clip(np.floor(np.arange(new_w) * src_w / new_w).astype(int), 0, src_w - 1)
    return img[rr[:, None], cc[None, :]]


def _rotate_cw(img: np.ndarray, turns: int) -> np.ndarray:
    turns = int(turns) % 4
    if turns == 0:
        return img.copy()
    return np.rot90(img, -turns)


def _reflect_by_quadrant(patch: np.ndarray, qb: int, qr: int) -> np.ndarray:
    out = patch
    if qr == 1:
        out = np.fliplr(out)
    if qb == 1:
        out = np.flipud(out)
    return out


def _apply_target_transform(
    patch: np.ndarray, key: tuple[int, int], color: int, turns: int | None = None
) -> np.ndarray:
    if TRANSFORM_KIND_BY_COLOR[color] == "reflect":
        return _reflect_by_quadrant(patch, key[0], key[1])
    assert turns in (1, 2, 3), "rotation family needs a 90/180/270 code"
    return _rotate_cw(patch, turns)


def _square_component(g: np.ndarray, pts: np.ndarray, color: int) -> tuple[int, int, int] | None:
    r0, c0, r1, c1 = bbox_coords(pts)
    side_r = r1 - r0 + 1
    side_c = c1 - c0 + 1
    if side_r != side_c:
        return None
    if not np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color):
        return None
    return int(r0), int(c0), int(side_r)


def _find_anchors(g: np.ndarray, mid_r: int, mid_c: int) -> dict[tuple[int, int], tuple[int, int, int, int]]:
    anchors: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for color in ANCHOR_COLORS:
        for pts in components_mask(g == color, connectivity=4):
            square = _square_component(g, pts, color)
            if square is None:
                continue
            r0, c0, size = square
            rr = r0 + size // 2
            cc = c0 + size // 2
            qb = 0 if rr < mid_r else 1
            qr = 0 if cc < mid_c else 1
            key = (qb, qr)
            assert key not in anchors, "each quadrant must contain exactly one square anchor"
            anchors[key] = (r0, c0, size, color)
    assert set(anchors) == {(0, 0), (0, 1), (1, 0), (1, 1)}, "must find one anchor in each quadrant"
    return anchors


def _extract_source_patch(g: np.ndarray, bg: int, mid_r: int, mid_c: int) -> np.ndarray:
    tl = g[0:mid_r, 0:mid_c]
    patch_bb = None
    for pts in components_mask(tl != bg, connectivity=4):
        if any(tl[r, c] == SOURCE_ANCHOR for r, c in pts):
            r0, c0, r1, c1 = bbox_coords(pts)
            patch_bb = (int(r0), int(c0), int(r1), int(c1))
            break
    assert patch_bb is not None, "top-left quadrant must contain a motif component with the source anchor"
    pr0, pc0, pr1, pc1 = patch_bb
    return tl[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()


def _quadrant_bounds(shape: tuple[int, int], mid_r: int, mid_c: int, key: tuple[int, int]) -> tuple[int, int, int, int]:
    qb, qr = key
    h, w = shape
    r0 = 0 if qb == 0 else mid_r + 1
    r1 = mid_r - 1 if qb == 0 else h - 1
    c0 = 0 if qr == 0 else mid_c + 1
    c1 = mid_c - 1 if qr == 0 else w - 1
    return int(r0), int(r1), int(c0), int(c1)


def _decode_rotation_turns(
    g: np.ndarray, bg: int, mid_r: int, mid_c: int, key: tuple[int, int], anchor: tuple[int, int, int, int]
) -> int:
    q_r0, q_r1, q_c0, q_c1 = _quadrant_bounds(g.shape, mid_r, mid_c, key)
    quad = g[q_r0 : q_r1 + 1, q_c0 : q_c1 + 1]
    local_r = anchor[0] - q_r0
    local_c = anchor[1] - q_c0
    match = None
    for pts in components_mask(quad != bg, connectivity=4):
        if any(local_r <= r < local_r + anchor[2] and local_c <= c < local_c + anchor[2] for r, c in pts):
            match = pts
            break
    assert match is not None, "rotation target must contain an anchor-plus-marker component"
    marker_cells = [(r, c) for r, c in match if int(quad[r, c]) == MARK]
    assert all(int(quad[r, c]) in (anchor[3], MARK) for r, c in match), "rotation component must be anchor plus marker only"
    turns = len(marker_cells)
    assert turns in (1, 2, 3), "marker length must encode 90/180/270 degrees"
    return int(turns)


def _scaled_patch(
    patch: np.ndarray, key: tuple[int, int], color: int, target_size: int, turns: int | None = None
) -> np.ndarray:
    patch_t = _apply_target_transform(patch, key, color, turns)
    ratio = target_size / SOURCE_ANCHOR_SIZE
    new_h = max(1, int(round(patch_t.shape[0] * ratio)))
    new_w = max(1, int(round(patch_t.shape[1] * ratio)))
    return _resize_nn(patch_t, new_h, new_w)


def _marker_rel_cells(red_r0: int, red_c0: int, size: int, side: str, length: int) -> list[tuple[int, int]]:
    mid_r = red_r0 + (size - 1) // 2
    mid_c = red_c0 + (size - 1) // 2
    if side == "top":
        return [(red_r0 - 1 - k, mid_c) for k in range(length)]
    if side == "bottom":
        return [(red_r0 + size + k, mid_c) for k in range(length)]
    if side == "left":
        return [(mid_r, red_c0 - 1 - k) for k in range(length)]
    if side == "right":
        return [(mid_r, red_c0 + size + k) for k in range(length)]
    raise ValueError(f"unknown marker side: {side}")


def _target_marker_cells(anchor_r: int, anchor_c: int, size: int, side: str, length: int) -> list[tuple[int, int]]:
    mid_r = anchor_r + (size - 1) // 2
    mid_c = anchor_c + (size - 1) // 2
    if side == "top":
        return [(anchor_r - 1 - k, mid_c) for k in range(length)]
    if side == "bottom":
        return [(anchor_r + size + k, mid_c) for k in range(length)]
    if side == "left":
        return [(mid_r, anchor_c - 1 - k) for k in range(length)]
    if side == "right":
        return [(mid_r, anchor_c + size + k) for k in range(length)]
    raise ValueError(f"unknown marker side: {side}")


def _valid_marker_sides(patch_s: np.ndarray, size: int, length: int) -> list[str]:
    rr, cc = np.where(patch_s == SOURCE_ANCHOR)
    red_r0, red_c0 = int(rr.min()), int(cc.min())
    sides = []
    for side in MARKER_SIDES:
        ok = True
        for mr, mc in _marker_rel_cells(red_r0, red_c0, size, side, length):
            if 0 <= mr < patch_s.shape[0] and 0 <= mc < patch_s.shape[1] and patch_s[mr, mc] != BG:
                ok = False
                break
        if ok:
            sides.append(side)
    return sides


def _choose_anchor(
    key: tuple[int, int], size: int, patch_s: np.ndarray, side: str | None = None, length: int = 0
) -> tuple[int, int] | None:
    q_r0, q_r1, q_c0, q_c1 = QUADS[key]
    rr, cc = np.where(patch_s == SOURCE_ANCHOR)
    red_r0, red_c0 = int(rr.min()), int(cc.min())

    fit_r0 = q_r0 + red_r0
    fit_r1 = q_r1 - patch_s.shape[0] + 1 + red_r0
    fit_c0 = q_c0 + red_c0
    fit_c1 = q_c1 - patch_s.shape[1] + 1 + red_c0

    anch_r0, anch_r1 = q_r0, q_r1 - size + 1
    anch_c0, anch_c1 = q_c0, q_c1 - size + 1

    if side is not None:
        if side == "top":
            anch_r0 = max(anch_r0, q_r0 + length)
        elif side == "bottom":
            anch_r1 = min(anch_r1, q_r1 - size - length + 1)
        elif side == "left":
            anch_c0 = max(anch_c0, q_c0 + length)
        elif side == "right":
            anch_c1 = min(anch_c1, q_c1 - size - length + 1)
        else:
            raise ValueError(f"unknown marker side: {side}")

    r_lo, r_hi = max(fit_r0, anch_r0), min(fit_r1, anch_r1)
    c_lo, c_hi = max(fit_c0, anch_c0), min(fit_c1, anch_c1)
    if r_lo > r_hi or c_lo > c_hi:
        return None
    return random.randint(r_lo, r_hi), random.randint(c_lo, c_hi)


def _paint_marker(out: np.ndarray, anchor_r: int, anchor_c: int, size: int, side: str, length: int) -> None:
    for rr, cc in _target_marker_cells(anchor_r, anchor_c, size, side, length):
        out[rr, cc] = MARK


def _stamp(out: np.ndarray, patch_s: np.ndarray, anchor_r: int, anchor_c: int) -> None:
    rr_red, cc_red = np.where(patch_s == SOURCE_ANCHOR)
    place_r0 = anchor_r - int(rr_red.min())
    place_c0 = anchor_c - int(cc_red.min())

    src_rr, src_cc = np.where(patch_s != BG)
    tgt_rr = src_rr + place_r0
    tgt_cc = src_cc + place_c0
    valid = (
        (tgt_rr >= 0)
        & (tgt_rr < GRID_SIZE)
        & (tgt_cc >= 0)
        & (tgt_cc < GRID_SIZE)
        & (tgt_rr != MID)
        & (tgt_cc != MID)
    )
    tgt_rr = tgt_rr[valid]
    tgt_cc = tgt_cc[valid]
    src_rr = src_rr[valid]
    src_cc = src_cc[valid]
    bg_only = out[tgt_rr, tgt_cc] == BG
    out[tgt_rr[bg_only], tgt_cc[bg_only]] = patch_s[src_rr[bg_only], src_cc[bg_only]]


def _anchor_signature(patch: np.ndarray) -> tuple[tuple[int, ...], ...]:
    view = patch.copy()
    view[view == SOURCE_ANCHOR] = BG
    return tuple(tuple(int(x) for x in row) for row in view.tolist())


def _grow_4conn(h: int, w: int, seeds: set[tuple[int, int]], n_fill: int) -> set[tuple[int, int]]:
    filled = set(seeds)
    frontier = set()
    for r, c in seeds:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in filled:
                frontier.add((nr, nc))
    while len(filled) < n_fill and frontier:
        cell = random.choice(list(frontier))
        frontier.discard(cell)
        filled.add(cell)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = cell[0] + dr, cell[1] + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in filled:
                frontier.add((nr, nc))
    return filled


def _assign_colors(filled: set[tuple[int, int]], anchor_cell: tuple[int, int], h: int, w: int) -> np.ndarray | None:
    rows = {r for r, _ in filled}
    cols = {c for _, c in filled}
    if rows != set(range(h)) or cols != set(range(w)):
        return None
    non_anchor = [p for p in filled if p != anchor_cell]
    if len(non_anchor) < MOTIF_COLORS_MIN - 1:
        return None
    max_colors = min(MOTIF_COLORS_MAX, len(non_anchor) + 1)
    if max_colors < MOTIF_COLORS_MIN:
        return None
    n_colors = random.randint(MOTIF_COLORS_MIN, max_colors)
    palette = MOTIF_EXTRA_COLORS[:]
    random.shuffle(palette)
    palette = palette[: n_colors - 1]

    logical = np.full((h, w), BG, dtype=int)
    logical[anchor_cell] = SOURCE_ANCHOR
    random.shuffle(non_anchor)
    for i, (r, c) in enumerate(non_anchor):
        logical[r, c] = palette[i] if i < len(palette) else random.choice(palette)
    return logical


def _build_source_patch() -> tuple[np.ndarray, int, int] | None:
    logical_h, logical_w = random.choice(BLOCK_DIMS)
    n_cells = logical_h * logical_w
    min_fill = max(4, int(np.ceil(FILL_RATIO_MIN * n_cells)))
    max_fill = max(min_fill, int(np.floor(FILL_RATIO_MAX * n_cells)))
    anchor_cell = (random.randrange(logical_h), random.randrange(logical_w))
    filled = _grow_4conn(logical_h, logical_w, {anchor_cell}, random.randint(min_fill, max_fill))
    logical = _assign_colors(filled, anchor_cell, logical_h, logical_w)
    if logical is None:
        return None

    patch = np.repeat(np.repeat(logical, SOURCE_ANCHOR_SIZE, axis=0), SOURCE_ANCHOR_SIZE, axis=1)
    signatures = {
        "flip_lr": _anchor_signature(np.fliplr(patch)),
        "flip_ud": _anchor_signature(np.flipud(patch)),
        "rot90": _anchor_signature(_rotate_cw(patch, 1)),
        "rot180": _anchor_signature(_rotate_cw(patch, 2)),
        "rot270": _anchor_signature(_rotate_cw(patch, 3)),
    }
    if len(set(signatures.values())) != len(signatures):
        return None

    anchor_r = anchor_cell[0] * SOURCE_ANCHOR_SIZE
    anchor_c = anchor_cell[1] * SOURCE_ANCHOR_SIZE
    return patch, anchor_r, anchor_c


def _target_sizes() -> dict[tuple[int, int], int]:
    while True:
        sizes = {(0, 0): SOURCE_ANCHOR_SIZE}
        for key in TARGET_KEYS:
            sizes[key] = random.randint(*TARGET_SIZE_RANGE)
        non_tl = [sizes[key] for key in TARGET_KEYS]
        if sum(size >= 2 for size in non_tl) >= 2 and any(size != SOURCE_ANCHOR_SIZE for size in non_tl):
            return sizes


def _legacy_scaled_patch(patch: np.ndarray, key: tuple[int, int], target_size: int) -> np.ndarray:
    p = _reflect_by_quadrant(patch, key[0], key[1])
    ratio = target_size / SOURCE_ANCHOR_SIZE
    h = max(1, int(round(p.shape[0] * ratio)))
    w = max(1, int(round(p.shape[1] * ratio)))
    return _resize_nn(p, h, w)


def _build_output(
    patch: np.ndarray,
    tl_r0: int,
    tl_c0: int,
    target_specs: dict[tuple[int, int], dict[str, object]],
    anchors: dict[tuple[int, int], tuple[int, int]],
    realized: dict[tuple[int, int], np.ndarray],
) -> np.ndarray:
    out = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    out[MID, :] = DIV
    out[:, MID] = DIV
    paste(out, patch, tl_r0, tl_c0)

    for key, spec in target_specs.items():
        ar, ac = anchors[key]
        paste(out, np.full((spec["size"], spec["size"]), spec["color"], dtype=int), ar, ac)
        _paint_marker(out, ar, ac, spec["size"], spec["side"], spec["marker_len"])
    for key, patch_s in realized.items():
        _stamp(out, patch_s, *anchors[key])
    return out


def _build_input(
    out: np.ndarray,
    target_specs: dict[tuple[int, int], dict[str, object]],
    anchors: dict[tuple[int, int], tuple[int, int]],
) -> np.ndarray:
    inp = out.copy()
    for key, spec in target_specs.items():
        q_r0, q_r1, q_c0, q_c1 = QUADS[key]
        inp[q_r0 : q_r1 + 1, q_c0 : q_c1 + 1] = BG
        ar, ac = anchors[key]
        paste(inp, np.full((spec["size"], spec["size"]), spec["color"], dtype=int), ar, ac)
        _paint_marker(inp, ar, ac, spec["size"], spec["side"], spec["marker_len"])
    return inp


def _spec_signature(spec: dict[str, object], patch_s: np.ndarray) -> tuple[object, ...]:
    return (
        spec["kind"],
        spec["color"],
        spec["size"],
        spec.get("turns", 0),
        _anchor_signature(patch_s),
    )


def _combo_visible(
    patch: np.ndarray,
    target_specs: dict[tuple[int, int], dict[str, object]],
    realized: dict[tuple[int, int], np.ndarray],
) -> bool:
    rotate_keys = [key for key, spec in target_specs.items() if spec["kind"] == "rotate"]
    reflect_keys = [key for key, spec in target_specs.items() if spec["kind"] == "reflect"]
    if len(rotate_keys) != 2 or len(reflect_keys) != 1:
        return False

    # Marker channel must matter: the two rotation targets should realize different motif signatures.
    rot_sigs = [_anchor_signature(realized[key]) for key in rotate_keys]
    if len(set(rot_sigs)) != len(rot_sigs):
        return False

    # Rotation family must differ from the old all-reflection rule on at least one target.
    changed_vs_legacy = 0
    for key in rotate_keys:
        legacy = _legacy_scaled_patch(patch, key, target_specs[key]["size"])
        if _anchor_signature(legacy) != _anchor_signature(realized[key]):
            changed_vs_legacy += 1
    if changed_vs_legacy == 0:
        return False

    # Color channel must matter: a green target with the same 1-pixel ray as one teal target should
    # still realize a different transform family.
    reflect_key = reflect_keys[0]
    same_len_rotation = [key for key in rotate_keys if int(target_specs[key]["marker_len"]) == REFLECT_MARK_LEN]
    if len(same_len_rotation) != 1:
        return False
    if _anchor_signature(realized[reflect_key]) == _anchor_signature(realized[same_len_rotation[0]]):
        return False

    # Family channel must stay readable: the reflection target should not accidentally realize the
    # same transformed motif as both rotation targets.
    all_sigs = [_spec_signature(target_specs[key], realized[key]) for key in TARGET_KEYS]
    return len(set(all_sigs)) == len(all_sigs)


def _gen_attempt() -> np.ndarray | None:
    source = _build_source_patch()
    if source is None:
        return None
    patch, anchor_r_in_patch, anchor_c_in_patch = source

    target_sizes = _target_sizes()
    reflect_key = random.choice(TARGET_KEYS)
    rotate_keys = [key for key in TARGET_KEYS if key != reflect_key]
    rotation_turns = [1, random.choice((2, 3))]
    random.shuffle(rotation_turns)

    target_specs: dict[tuple[int, int], dict[str, object]] = {}
    for key in TARGET_KEYS:
        if key == reflect_key:
            target_specs[key] = {
                "kind": "reflect",
                "color": REFLECT_ANCHOR,
                "size": target_sizes[key],
                "marker_len": REFLECT_MARK_LEN,
            }
        else:
            turns = rotation_turns.pop()
            target_specs[key] = {
                "kind": "rotate",
                "color": ROTATE_ANCHOR,
                "size": target_sizes[key],
                "turns": turns,
                "marker_len": turns,
            }

    realized: dict[tuple[int, int], np.ndarray] = {}
    for key, spec in target_specs.items():
        patch_s = _scaled_patch(
            patch,
            key,
            int(spec["color"]),
            int(spec["size"]),
            None if spec["kind"] == "reflect" else int(spec["turns"]),
        )
        patch_t = _apply_target_transform(
            patch, key, int(spec["color"]), None if spec["kind"] == "reflect" else int(spec["turns"])
        )
        roundtrip = _resize_nn(patch_s, patch_t.shape[0], patch_t.shape[1])
        if not np.array_equal(roundtrip, patch_t):
            return None
        sides = _valid_marker_sides(patch_s, int(spec["size"]), int(spec["marker_len"]))
        if not sides:
            return None
        spec["side"] = random.choice(sides)
        realized[key] = patch_s

    if not _combo_visible(patch, target_specs, realized):
        return None

    anchors: dict[tuple[int, int], tuple[int, int]] = {}
    tl_r0 = random.randint(0, MID - patch.shape[0])
    tl_c0 = random.randint(0, MID - patch.shape[1])
    anchors[(0, 0)] = (tl_r0 + anchor_r_in_patch, tl_c0 + anchor_c_in_patch)
    for key, spec in target_specs.items():
        anchor = _choose_anchor(
            key,
            int(spec["size"]),
            realized[key],
            str(spec["side"]),
            int(spec["marker_len"]),
        )
        if anchor is None:
            return None
        anchors[key] = anchor

    out = _build_output(patch, tl_r0, tl_c0, target_specs, anchors, realized)
    inp = _build_input(out, target_specs, anchors)
    return inp if np.array_equal(main(inp), out) else None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the top-left motif into the other quadrants by splitting transform control across
    anchor size, anchor color, and edge-marker count.

    Input structure:
    - One solid non-background row and one solid non-background column divide the 27x27 board into four 13x13 quadrants.
    - The TL quadrant contains the full source motif as one connected non-background component with a red 2x2 source anchor inside it.
    - Every non-TL quadrant contains one square target anchor plus a short black ray attached to one edge.
    - Green anchors belong to the reflection family, while teal anchors belong to the rotation family; only the teal branch interprets the ray length.

    Rule:
    - Extract the TL source motif and keep using target-anchor size divided by source-anchor size as the scale factor.
    - For each green target anchor, use the legacy quadrant-based reflection rule and ignore the attached ray as a non-operative family label.
    - For each teal target anchor, first read the anchor color as "rotation family", then count the attached black pixels and rotate clockwise by 90/180/270 degrees accordingly before scaling.
    - Align the transformed source red area onto the target anchor and stamp only onto background cells, leaving divider cells, anchors, and marker rays intact.

    Key insight:
    - The old single cue has been split into a hierarchy: anchor size still gives scale, anchor color chooses the transform family, and only the rotation family consults the edge-pixel count to decode the exact angle.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    bg = get_bg(g)
    mid_r, mid_c = _find_dividers(g, bg)
    anchors = _find_anchors(g, mid_r, mid_c)

    sr0, sc0, source_size, source_color = anchors[(0, 0)]
    assert source_color == SOURCE_ANCHOR, "top-left source anchor must stay red"
    assert source_size == SOURCE_ANCHOR_SIZE, "top-left source anchor must be 2x2"

    patch = _extract_source_patch(g, bg, mid_r, mid_c)

    for key, anchor in anchors.items():
        if key == (0, 0):
            continue

        tr0, tc0, target_size, color = anchor
        kind = TRANSFORM_KIND_BY_COLOR[color]
        turns = None if kind == "reflect" else _decode_rotation_turns(g, bg, mid_r, mid_c, key, anchor)
        patch_t = _apply_target_transform(patch, key, color, turns)

        ratio = float(target_size) / float(source_size)
        new_h = max(1, int(round(patch_t.shape[0] * ratio)))
        new_w = max(1, int(round(patch_t.shape[1] * ratio)))
        patch_s = _resize_nn(patch_t, new_h, new_w)
        roundtrip = _resize_nn(patch_s, patch_t.shape[0], patch_t.shape[1])
        assert np.array_equal(roundtrip, patch_t), "scaling must preserve motif information"

        rr_red, cc_red = np.where(patch_s == SOURCE_ANCHOR)
        assert rr_red.size > 0, "transformed patch must still contain the source anchor region"
        place_r0 = tr0 - int(rr_red.min())
        place_c0 = tc0 - int(cc_red.min())

        rr_p, cc_p = np.where(patch_s != bg)
        for rr0, cc0 in zip(rr_p.tolist(), cc_p.tolist(), strict=False):
            rr_t = place_r0 + rr0
            cc_t = place_c0 + cc0
            if not (0 <= rr_t < h and 0 <= cc_t < w):
                continue
            if rr_t == mid_r or cc_t == mid_c:
                continue
            if out[rr_t, cc_t] != bg:
                continue
            out[rr_t, cc_t] = int(patch_s[rr0, cc0])

    return out


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid combo b10624e5 input")
