# Combo variant of a32d8b75 (index 16)
# combo_spec: D1/0+D3/1+D5/1
# Name: 形状匹配嵌入+多色框架+旋转翻转

import random

import numpy as np
from arc_utils import bbox_coords, neighbors4
from scipy.ndimage import label as _scipy_label


# Design analysis
# ---------------
# a) Concrete fused behavior and trace:
#    - Keep the original left/right legend -> canvas layout, but replace the old tiled mask logic with a 5-column plaque.
#    - The unit sprite is fixed at 3x3 so the legend parser can still read the whole panel from a 5-column slice.
#    - The plaque contains three enclosed zero-color cavities stacked vertically.
#    - Two cavities deliberately share the same silhouette, but their surrounding frame colors differ:
#        8 => use the rotation-family coloring for that silhouette
#        9 => use the reflection-family coloring for that same silhouette
#      This makes D3 non-redundant: shape alone no longer determines the filled colors.
#    - A third cavity uses a different silhouette, so the solver still has to perform actual shape matching instead of only
#      reading frame color.
#    - After all cavities are filled, the completed plaque is transformed again by the marker patch:
#      hand direction = quarter-turn, center color = optional extra horizontal / vertical flip.
#    - Micro trace:
#      old: read frame mask -> tile one sprite everywhere -> rotate -> place.
#      fused: read 3x3 sprite -> group all dihedral variants by cavity silhouette and by family (rot / reflect) ->
#      fill each cavity using both its shape and its frame color -> rotate whole plaque -> optional flip whole plaque -> place.
#
# b) Complexity calibration:
#    - Three cavities per plaque give repeated evidence: one same-shape rot-vs-reflect pair plus one different shape.
#    - The frame uses exactly two fixed semantic colors (8, 9), and the marker uses exactly three fixed center colors
#      for {none, hflip, vflip}, keeping the hypothesis space small.
#    - The generator searches for unit sprites where at least one cavity silhouette is shared by both transform families
#      but produces different color layouts, so the frame colors visibly matter.
#
# c) Hidden-assumption audit against the original main():
#    - `frame != 0` is no longer enough: the frame colors carry local rules, so the solver preserves the full plaque colors.
#    - Internal zeros are no longer irrelevant padding; they are the actual cavity instructions and must stay intact.
#    - Shape matching can no longer choose a single exact dihedral transform by silhouette alone. The solver groups transforms
#      by silhouette and then uses the surrounding frame color to choose the family-specific colored fill.
#    - Global transform is no longer a single rotation integer; it is `(rot_k, flip_mode)` applied after local cavity filling.
#    - The generator must guarantee each used cavity/family pair resolves to one unique colored fill, or the solver would be
#      ambiguous. The search below enforces that.
#
# d) Placement legality:
#    - The generator is output-first: build the completed plaque, apply the global transform, then assign anchors with full
#      in-bounds and non-overlap checks before drawing the input.
#    - Full plaque pixels, including frame colors, count as occupied during overlap checks.
#
# e) Visual observability:
#    - Humans can compare two equal-shaped cavities inside one plaque: same hole outline, different surrounding color,
#      different internal color arrangement. That exposes the D3 rule directly.
#    - The third cavity has a different silhouette, so the D1 shape-matching step is also visible from a few examples.
#    - The marker patch is structurally pinned below the anchor square, so the extra flip cue is inferable without relying
#      on color id ordering or any non-visual convention.
#
# f) Visual cleanliness:
#    - All added semantics stay inside the legend/plaque structure; the canvas is still just background bands plus placed plaques.
#    - Each plaque uses one centered cavity lane, so holes never merge and the picture remains legible.
#
# g) Non-degeneration target:
#    - For every legend, at least one pair of same-shape cavities must differ between frame-color 8 and 9 fills.
#    - For every sample, at least one legend must use a non-zero global flip, and that flip must change the placed plaque.


SPRITE_N = 3
LEGEND_W = 5
GRID_H_RANGE = (28, 30)
GRID_W = 30
LEGEND_COUNT_WEIGHTS = [1, 1]  # keep single-legend and dual-legend cases both visible across samples
MIN_CANVAS_W = 8
MARKER_PANEL_HEIGHT = 8
BAND_WIDTH_MIN = 4
GEN_RETRIES = 500
LEGEND_DESIGN_RETRIES = 500
FOUR_CONN = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
EIGHT_CONN = np.ones((3, 3), dtype=int)

FRAME_ROT_COLOR = 8
FRAME_REFLECT_COLOR = 9
FRAME_RULE_COLORS = (FRAME_ROT_COLOR, FRAME_REFLECT_COLOR)

ROTATION_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up/right/down/left
FLIP_NONE = 0
FLIP_H = 1
FLIP_V = 2
MARKER_COLOR_TO_FLIP = {1: FLIP_NONE, 2: FLIP_H, 3: FLIP_V}
FLIP_TO_MARKER_COLOR = {v: k for k, v in MARKER_COLOR_TO_FLIP.items()}

COLOR_POOL = [1, 2, 3, 4, 5, 7]


def _separator_runs(g: np.ndarray) -> list[tuple[int, int]]:
    cols6 = [c for c in range(g.shape[1]) if np.all(g[:, c] == 6)]
    if not cols6:
        return []
    runs: list[tuple[int, int]] = []
    start = end = cols6[0]
    for c in cols6[1:]:
        if c == end + 1:
            end = c
        else:
            runs.append((start, end))
            start = end = c
    runs.append((start, end))
    return runs


def _split_canvas(g: np.ndarray) -> tuple[tuple[int, int], tuple[int, int] | None]:
    runs = _separator_runs(g)
    assert runs, "must have at least one color-6 separator run"
    left = runs[0]
    right = None
    for start, end in reversed(runs[1:]):
        if start >= g.shape[1] - 10 and start > left[1] + 1:
            right = (start, end)
            break
    return left, right


def _all_marker_panel_squares(g: np.ndarray) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    h, w = g.shape
    for r0 in range(h - 7):
        for c0 in range(w - 4):
            sub = g[r0 : r0 + 5, c0 : c0 + 5]
            if not (np.all(sub[0, :] == 6) and np.all(sub[4, :] == 6) and np.all(sub[:, 0] == 6) and np.all(sub[:, 4] == 6)):
                continue
            inner = sub[1:4, 1:4]
            if np.count_nonzero((inner != 0) & (inner != 6)) != 1:
                continue
            rot = g[r0 + 5 : r0 + 8, c0 + 1 : c0 + 4]
            if rot.shape != (3, 3) or not np.any((rot != 0) & (rot != 6)):
                continue
            out.append((int(r0), int(c0)))
    out.sort()
    return out


def _extract_unit_sprite(legend: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    mask = (legend != 0) & (legend != 6)
    pts_all = np.argwhere(mask)
    if pts_all.size == 0:
        return None
    rmin = int(pts_all[:, 0].min())
    cseed = int(np.argwhere(mask[rmin, :])[0, 0])
    h, w = legend.shape
    vis = np.zeros((h, w), dtype=bool)
    stack = [(rmin, cseed)]
    vis[rmin, cseed] = True
    pts: list[tuple[int, int]] = []
    while stack:
        r, c = stack.pop()
        pts.append((r, c))
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not vis[rr, cc]:
                vis[rr, cc] = True
                stack.append((rr, cc))
    r0, c0, r1, c1 = bbox_coords(np.asarray(pts))
    spr = legend[r0 : r1 + 1, c0 : c1 + 1].copy()
    cols = sorted({int(x) for x in np.unique(spr) if int(x) not in (0, 6)})
    if len(cols) == 2:
        a, b = cols
        swapped = spr.copy()
        swapped[spr == a] = -1
        swapped[spr == b] = a
        swapped[swapped == -1] = b
        spr = swapped
    return spr, (r0, c0, r1, c1)


def _extract_frame_bbox(legend: np.ndarray, unit_bb: tuple[int, int, int, int], border6_r0: int) -> tuple[int, int, int, int] | None:
    r_start = unit_bb[2] + 1
    r_end = border6_r0 - 1
    if r_start > r_end:
        return None
    sub = legend[r_start : r_end + 1, :]
    mask = (sub != 0) & (sub != 6)
    pts = np.argwhere(mask)
    if pts.size == 0:
        return None
    pts[:, 0] += r_start
    r0 = int(pts[:, 0].min())
    r1 = int(pts[:, 0].max())
    c0 = int(pts[:, 1].min())
    c1 = int(pts[:, 1].max())
    return (r0, c0, r1, c1)


def _position_indices_from_border_square(g: np.ndarray, border6_r0: int, border6_c0: int) -> tuple[int, int] | None:
    inner = g[border6_r0 + 1 : border6_r0 + 4, border6_c0 + 1 : border6_c0 + 4]
    pts = np.argwhere((inner != 0) & (inner != 6))
    if pts.size == 0:
        return None
    return int(pts[0, 0]), int(pts[0, 1])


def _rotation_flip_from_marker(g: np.ndarray, border6_r0: int, border6_c0: int) -> tuple[int, int]:
    sub = g[border6_r0 + 5 : border6_r0 + 8, border6_c0 + 1 : border6_c0 + 4]
    assert sub.shape == (3, 3), "rotation marker patch out of bounds"
    center_color = int(sub[1, 1])
    assert center_color in MARKER_COLOR_TO_FLIP, "invalid global flip marker color"
    pts = np.argwhere((sub != 0) & (sub != 6))
    endpoint = [(int(r), int(c)) for r, c in pts if not (int(r) == 1 and int(c) == 1)]
    assert len(endpoint) == 1, "marker must contain center plus exactly one endpoint"
    rr, cc = endpoint[0]
    dr, dc = rr - 1, cc - 1
    if abs(dr) >= abs(dc):
        rot_k = 0 if dr < 0 else 2
    else:
        rot_k = 1 if dc > 0 else 3
    return rot_k, MARKER_COLOR_TO_FLIP[center_color]


def _anchor_to_top_left(anchor_idx: int, size: int, place_size: int) -> int:
    if anchor_idx <= 0:
        canvas_anchor = 0
        obj_anchor = 0
    elif anchor_idx >= 2:
        canvas_anchor = max(0, size - 1)
        obj_anchor = max(0, place_size - 1)
    else:
        canvas_anchor = (size - 1) // 2 if size > 1 else 0
        obj_anchor = (place_size - 1) // 2 if place_size > 1 else 0
    return canvas_anchor - obj_anchor


def _apply_dihedral(arr: np.ndarray, is_reflect: bool, rot_k: int) -> np.ndarray:
    base = np.fliplr(arr) if is_reflect else arr
    return np.rot90(base, rot_k).copy()


def _apply_global_transform(arr: np.ndarray, rot_k: int, flip_mode: int) -> np.ndarray:
    out = np.rot90(arr, (-rot_k) % 4)
    if flip_mode == FLIP_H:
        out = np.fliplr(out)
    elif flip_mode == FLIP_V:
        out = np.flipud(out)
    return out


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    return mask.shape, tuple(mask.astype(int).flatten())


def _sprite_key(sprite: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    return sprite.shape, tuple(int(x) for x in sprite.flatten())


def _variant_records(unit_sprite: np.ndarray) -> list[dict]:
    records: list[dict] = []
    for is_reflect in (False, True):
        for rot_k in range(4):
            spr = _apply_dihedral(unit_sprite, is_reflect, rot_k)
            mask = spr != 0
            records.append(
                {
                    "is_reflect": is_reflect,
                    "rot_k": rot_k,
                    "sprite": spr,
                    "mask": mask,
                    "mask_key": _mask_key(mask),
                    "sprite_key": _sprite_key(spr),
                }
            )
    return records


def _family_fill_lookup(unit_sprite: np.ndarray) -> dict[tuple[tuple[int, int], tuple[int, ...]], dict[bool, np.ndarray]]:
    lookup: dict[tuple[tuple[int, int], tuple[int, ...]], dict[bool, np.ndarray]] = {}
    by_key: dict[tuple[tuple[int, int], tuple[int, ...]], dict[bool, set[tuple[tuple[int, int], tuple[int, ...]]]]] = {}
    reps: dict[tuple[tuple[tuple[int, int], tuple[int, ...]], bool], np.ndarray] = {}
    for rec in _variant_records(unit_sprite):
        mkey = rec["mask_key"]
        fam = bool(rec["is_reflect"])
        by_key.setdefault(mkey, {False: set(), True: set()})[fam].add(rec["sprite_key"])
        reps.setdefault((mkey, fam), rec["sprite"])
    for mkey, fam_map in by_key.items():
        fam_out: dict[bool, np.ndarray] = {}
        for fam, sprite_keys in fam_map.items():
            if not sprite_keys:
                continue
            assert len(sprite_keys) == 1, "family+shape must map to exactly one colored fill"
            fam_out[fam] = reps[(mkey, fam)]
        lookup[mkey] = fam_out
    return lookup


def _frame_cavities(frame: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
    zero_mask = frame == 0
    labs, nlab = _scipy_label(zero_mask, structure=FOUR_CONN)
    cavities: list[tuple[int, int, np.ndarray]] = []
    h, w = frame.shape
    for idx in range(1, nlab + 1):
        pts = np.argwhere(labs == idx)
        if pts.size == 0:
            continue
        if np.any(pts[:, 0] == 0) or np.any(pts[:, 0] == h - 1) or np.any(pts[:, 1] == 0) or np.any(pts[:, 1] == w - 1):
            continue
        r0, c0, r1, c1 = bbox_coords(pts)
        cavities.append((r0, c0, labs[r0 : r1 + 1, c0 : c1 + 1] == idx))
    cavities.sort(key=lambda item: (item[0], item[1]))
    return cavities


def _cavity_rule_color(frame: np.ndarray, r0: int, c0: int, cav_mask: np.ndarray) -> int:
    colors: set[int] = set()
    h, w = frame.shape
    ch, cw = cav_mask.shape
    for rr in range(ch):
        for cc in range(cw):
            if not cav_mask[rr, cc]:
                continue
            gr = r0 + rr
            gc = c0 + cc
            for dc in (-1, 1):
                scan = gc + dc
                while 0 <= scan < w and frame[gr, scan] == 0:
                    scan += dc
                if 0 <= scan < w:
                    val = int(frame[gr, scan])
                    if val not in (0, 6):
                        colors.add(val)
    assert colors <= set(FRAME_RULE_COLORS), f"unexpected cavity rule colors: {colors}"
    assert len(colors) == 1, f"cavity must have one rule color, got {colors}"
    return next(iter(colors))


def _assemble_plaque(unit_sprite: np.ndarray, frame: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    lookup = _family_fill_lookup(unit_sprite)
    comp = frame.copy()
    matched: list[dict] = []
    cavities = _frame_cavities(frame)
    assert len(cavities) == 3, "frame must contain exactly three enclosed cavities"
    for r0, c0, cav_mask in cavities:
        rule_color = _cavity_rule_color(frame, r0, c0, cav_mask)
        fam = rule_color == FRAME_REFLECT_COLOR
        mkey = _mask_key(cav_mask)
        assert mkey in lookup and fam in lookup[mkey], "cavity shape/family not supported by unit sprite"
        fill = lookup[mkey][fam]
        fill_mask = fill != 0
        assert fill_mask.shape == cav_mask.shape and np.array_equal(fill_mask, cav_mask), "fill mask mismatch"
        sub = comp[r0 : r0 + cav_mask.shape[0], c0 : c0 + cav_mask.shape[1]]
        sub[cav_mask] = fill[fill_mask]
        matched.append({"r0": r0, "c0": c0, "rule_color": rule_color, "family_reflect": fam, "mask_key": mkey})
    return comp, matched


def _output_from_legend(unit_sprite: np.ndarray, frame: np.ndarray, rot_k: int, flip_mode: int) -> np.ndarray:
    plaque, _ = _assemble_plaque(unit_sprite, frame)
    return _apply_global_transform(plaque, rot_k, flip_mode)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode legend plaques with color-coded cavity families, then place globally rotated/flipped results on the canvas.

    Input structure:
    - Full-height color-6 columns still split the grid into a left legend, the canvas, and an optional right legend.
    - Each legend panel contains a color-swapped 3x3 unit sprite, then a 5-column plaque with three enclosed cavities, then a 5x5 anchor square and a 3x3 transform marker.
    - Plaque colors 8 and 9 are both active frame pixels, but they mark different cavity-fill families; the marker center color separately encodes whether the finished plaque gets an extra horizontal or vertical flip.

    Rule:
    - Read the unit sprite and enumerate its dihedral variants grouped by cavity silhouette and by family (rotation-family vs reflection-family).
    - For each enclosed cavity in the plaque, match its zero-shape silhouette and use the surrounding frame color to choose which family-specific colored fill to insert.
    - After all cavities are filled, rotate the whole plaque from the hand direction, optionally flip it from the marker center color, and place it on the canvas according to the anchor square.

    Key insight:
    - Two cavities can share the same hole shape while producing different internal color layouts, because the frame color chooses between the rotation-family and reflection-family fill for that silhouette before the whole plaque is transformed again globally.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert w >= 2, "grid width must be >= 2"

    left_sep, right_sep = _split_canvas(g)
    canvas_c0 = left_sep[1] + 1
    canvas_c1 = right_sep[0] if right_sep is not None else w
    canvas = g[:, canvas_c0:canvas_c1].copy()

    marker_squares = _all_marker_panel_squares(g)
    assert marker_squares, "no marker panel found"

    eff_h, eff_w = canvas.shape
    full6_cols = [c for c in range(eff_w) if np.all(canvas[:, c] == 6)]
    if full6_cols:
        eff_w = min(full6_cols)
    full6_rows = [r for r in range(eff_h) if np.all(canvas[r, :] == 6)]
    if full6_rows:
        eff_h = min(full6_rows)

    out = canvas.copy()
    composite_written = np.zeros(out.shape, dtype=bool)

    for marker_r0, marker_c0 in marker_squares:
        if canvas_c0 <= marker_c0 < canvas_c1:
            continue
        if marker_c0 + 5 > w:
            continue

        panel = g[:, marker_c0 : marker_c0 + 5]
        unit = _extract_unit_sprite(panel)
        if unit is None:
            continue
        unit_sprite, unit_bb = unit

        frame_bb = _extract_frame_bbox(panel, unit_bb, marker_r0)
        if frame_bb is None:
            continue
        fr0, fc0, fr1, fc1 = frame_bb
        frame = panel[fr0 : fr1 + 1, fc0 : fc1 + 1].copy()

        frame_active = (frame != 0) & (frame != 6)
        _, n8 = _scipy_label(frame_active, structure=EIGHT_CONN)
        assert n8 == 1, f"frame must be 8-connected, got {n8} components"
        frame_colors = {int(x) for x in np.unique(frame[frame_active])}
        assert frame_colors == set(FRAME_RULE_COLORS), f"frame must use both rule colors, got {frame_colors}"

        rot_k, flip_mode = _rotation_flip_from_marker(g, marker_r0, marker_c0)
        comp = _output_from_legend(unit_sprite, frame, rot_k, flip_mode)

        pos = _position_indices_from_border_square(g, marker_r0, marker_c0)
        if pos is None:
            continue
        pr, pc = pos
        top = _anchor_to_top_left(pr, eff_h, comp.shape[0])
        left = _anchor_to_top_left(pc, eff_w, comp.shape[1])

        ch, cw = comp.shape
        for r in range(ch):
            rr = top + r
            assert 0 <= rr < out.shape[0], "plaque row out of canvas bounds"
            for c in range(cw):
                cc = left + c
                assert 0 <= cc < out.shape[1], "plaque col out of canvas bounds"
                if out[rr, cc] == 6:
                    continue
                val = int(comp[r, c])
                if val != 0 and val != 6:
                    assert not composite_written[rr, cc], "plaques must not overlap"
                    composite_written[rr, cc] = True
                    out[rr, cc] = val
    return out


def _color_swap(sprite: np.ndarray, c1: int, c2: int) -> np.ndarray:
    out = sprite.copy()
    out[sprite == c1] = -1
    out[sprite == c2] = c1
    out[out == -1] = c2
    return out


def _is_connected(mask: np.ndarray) -> bool:
    _, nlab = _scipy_label(mask.astype(int), structure=FOUR_CONN)
    return nlab == 1


def _mask_groups(mask: np.ndarray) -> dict[tuple[tuple[int, int], tuple[int, ...]], dict[bool, list[np.ndarray]]]:
    groups: dict[tuple[tuple[int, int], tuple[int, ...]], dict[bool, list[np.ndarray]]] = {}
    for is_reflect in (False, True):
        for rot_k in range(4):
            m = _apply_dihedral(mask.astype(int), is_reflect, rot_k).astype(bool)
            mkey = _mask_key(m)
            groups.setdefault(mkey, {False: [], True: []})[is_reflect].append(m)
    return groups


def _generate_unit_sprite(c1: int, c2: int) -> tuple[np.ndarray, np.ndarray, tuple[tuple[int, int], tuple[int, ...]], tuple[tuple[int, int], tuple[int, ...]]]:
    for _ in range(LEGEND_DESIGN_RETRIES):
        mask = np.zeros((SPRITE_N, SPRITE_N), dtype=bool)
        cells = random.randint(4, 6)
        for idx in random.sample(range(SPRITE_N * SPRITE_N), cells):
            mask[idx // SPRITE_N, idx % SPRITE_N] = True
        if not _is_connected(mask):
            continue
        if not np.all(mask.any(axis=0)) or not np.all(mask.any(axis=1)):
            continue

        mask_groups = _mask_groups(mask)
        if len(mask_groups) < 2:
            continue
        shared_keys = [k for k, fam_map in mask_groups.items() if fam_map[False] and fam_map[True]]
        if not shared_keys:
            continue

        sprite = np.zeros((SPRITE_N, SPRITE_N), dtype=int)
        true_ids = [idx for idx in range(SPRITE_N * SPRITE_N) if mask.flat[idx]]
        num_c2 = random.randint(1, len(true_ids) - 1)
        c2_ids = set(random.sample(true_ids, num_c2))
        for idx in true_ids:
            sprite.flat[idx] = c2 if idx in c2_ids else c1

        lookup = _family_fill_lookup(sprite)
        valid_shared = []
        for key in shared_keys:
            fam_map = lookup.get(key, {})
            if False not in fam_map or True not in fam_map:
                continue
            if _sprite_key(fam_map[False]) == _sprite_key(fam_map[True]):
                continue
            valid_shared.append(key)
        if not valid_shared:
            continue

        shared_key = random.choice(valid_shared)
        other_keys = [k for k in lookup.keys() if k != shared_key]
        if not other_keys:
            continue
        other_key = random.choice(other_keys)
        legend_sprite = _color_swap(sprite, c1, c2)
        return sprite, legend_sprite, shared_key, other_key
    raise RuntimeError("failed to generate a unit sprite with shared-shape family contrast")


def _choose_cavity_specs(
    sprite: np.ndarray,
    shared_key: tuple[tuple[int, int], tuple[int, ...]],
    other_key: tuple[tuple[int, int], tuple[int, ...]],
) -> list[dict]:
    lookup = _family_fill_lookup(sprite)
    assert shared_key in lookup and False in lookup[shared_key] and True in lookup[shared_key]
    assert other_key in lookup

    other_families = [fam for fam in (False, True) if fam in lookup[other_key]]
    other_family = random.choice(other_families)
    specs = [
        {"mask_key": shared_key, "is_reflect": False, "rule_color": FRAME_ROT_COLOR},
        {"mask_key": shared_key, "is_reflect": True, "rule_color": FRAME_REFLECT_COLOR},
        {"mask_key": other_key, "is_reflect": other_family, "rule_color": FRAME_REFLECT_COLOR if other_family else FRAME_ROT_COLOR},
    ]
    random.shuffle(specs)
    return specs


def _frame_mask_from_key(mask_key_data: tuple[tuple[int, int], tuple[int, ...]]) -> np.ndarray:
    shape, flat = mask_key_data
    return np.array(flat, dtype=bool).reshape(shape)


def _build_frame(cavity_specs: list[dict], inner_gap: int) -> np.ndarray:
    frame_h = 2 + len(cavity_specs) * SPRITE_N + (len(cavity_specs) - 1) * inner_gap
    frame = np.full((frame_h, LEGEND_W), FRAME_ROT_COLOR, dtype=int)

    top_color = cavity_specs[0]["rule_color"]
    bottom_color = cavity_specs[-1]["rule_color"]
    frame[0, :] = top_color
    frame[-1, :] = bottom_color

    row = 1
    for idx, spec in enumerate(cavity_specs):
        rule_color = spec["rule_color"]
        frame[row : row + SPRITE_N, :] = rule_color
        cav_mask = _frame_mask_from_key(spec["mask_key"])
        assert cav_mask.shape == (SPRITE_N, SPRITE_N)
        frame[row : row + SPRITE_N, 1:4][cav_mask] = 0
        row += SPRITE_N
        if idx < len(cavity_specs) - 1:
            frame[row : row + inner_gap, :] = rule_color
            row += inner_gap
    return frame


def _pick_global_transform(local_plaque: np.ndarray) -> tuple[int, int, np.ndarray]:
    options: list[tuple[int, int, np.ndarray]] = []
    for rot_k in range(4):
        rotated = _apply_global_transform(local_plaque, rot_k, FLIP_NONE)
        options.append((rot_k, FLIP_NONE, rotated))
        flipped_h = _apply_global_transform(local_plaque, rot_k, FLIP_H)
        if not np.array_equal(rotated, flipped_h):
            options.append((rot_k, FLIP_H, flipped_h))
        flipped_v = _apply_global_transform(local_plaque, rot_k, FLIP_V)
        if not np.array_equal(rotated, flipped_v):
            options.append((rot_k, FLIP_V, flipped_v))
    non_none = [opt for opt in options if opt[1] != FLIP_NONE]
    if non_none:
        return random.choices(non_none + options, weights=[4] * len(non_none) + [1] * len(options))[0]
    return random.choice(options)


def _allocate_colors(num_legends: int):
    pool = list(COLOR_POOL)
    random.shuffle(pool)
    if len(pool) < 2 * num_legends + 2:
        return None
    unit_color_sets = []
    for _ in range(num_legends):
        unit_color_sets.append((pool.pop(), pool.pop()))
    bg_colors = (pool.pop(), pool.pop())
    return unit_color_sets, bg_colors


def _design_legend(leg: dict, unit_colors: tuple[int, int], marker_r0: int) -> bool:
    c1, c2 = unit_colors
    avail = marker_r0 - (SPRITE_N + 1)
    if avail < 13:
        return False

    sprite, legend_sprite, shared_key, other_key = _generate_unit_sprite(c1, c2)
    cavity_specs = _choose_cavity_specs(sprite, shared_key, other_key)

    gap_options = [gap for gap in (1, 2) if 2 + 3 * SPRITE_N + 2 * gap <= avail]
    if not gap_options:
        return False
    inner_gap = random.choice(gap_options)
    frame = _build_frame(cavity_specs, inner_gap)

    local_plaque, matches = _assemble_plaque(sprite, frame)
    diff_pair = None
    for a in matches:
        for b in matches:
            if a["mask_key"] == b["mask_key"] and a["family_reflect"] != b["family_reflect"]:
                diff_pair = (a, b)
                break
        if diff_pair is not None:
            break
    if diff_pair is None:
        return False

    rot_k, flip_mode, transformed_plaque = _pick_global_transform(local_plaque)
    leg.update(
        sprite=sprite,
        legend_sprite=legend_sprite,
        frame=frame,
        cavity_specs=cavity_specs,
        local_plaque=local_plaque,
        transformed_plaque=transformed_plaque,
        rot_k=rot_k,
        flip_mode=flip_mode,
        matches=matches,
    )
    return True


def _comp_size(leg: dict) -> tuple[int, int]:
    return leg["transformed_plaque"].shape


def _assign_anchors(legends: list[dict], grid_h: int, canvas_w: int) -> bool:
    if len(legends) == 1:
        ch, cw = _comp_size(legends[0])
        choices = []
        for pr in range(3):
            for pc in range(3):
                top = _anchor_to_top_left(pr, grid_h, ch)
                left = _anchor_to_top_left(pc, canvas_w, cw)
                if top < 0 or left < 0 or top + ch > grid_h or left + cw > canvas_w:
                    continue
                choices.append((pr, pc, top, left))
        if not choices:
            return False
        pr, pc, top, left = random.choice(choices)
        legends[0].update(pr=pr, pc=pc, comp_top=top, comp_left=left)
        return True

    a, b = legends
    a_h, a_w = _comp_size(a)
    b_h, b_w = _comp_size(b)
    valid_pairs = []
    for apr in range(3):
        for apc in range(3):
            at = _anchor_to_top_left(apr, grid_h, a_h)
            al = _anchor_to_top_left(apc, canvas_w, a_w)
            if at < 0 or al < 0 or at + a_h > grid_h or al + a_w > canvas_w:
                continue
            for bpr in range(3):
                for bpc in range(3):
                    bt = _anchor_to_top_left(bpr, grid_h, b_h)
                    bl = _anchor_to_top_left(bpc, canvas_w, b_w)
                    if bt < 0 or bl < 0 or bt + b_h > grid_h or bl + b_w > canvas_w:
                        continue
                    if at < bt + b_h and bt < at + a_h and al < bl + b_w and bl < al + a_w:
                        continue
                    valid_pairs.append((apr, apc, at, al, bpr, bpc, bt, bl))
    if not valid_pairs:
        return False
    apr, apc, at, al, bpr, bpc, bt, bl = random.choice(valid_pairs)
    a.update(pr=apr, pc=apc, comp_top=at, comp_left=al)
    b.update(pr=bpr, pc=bpc, comp_top=bt, comp_left=bl)
    return True


def _build_banded_canvas(grid_h: int, canvas_w: int, bg1: int, bg2: int) -> np.ndarray:
    canvas = np.zeros((grid_h, canvas_w), dtype=int)
    band_dir = random.choice(["h", "v"])
    dim = grid_h if band_dir == "h" else canvas_w
    bw = random.randint(BAND_WIDTH_MIN, max(BAND_WIDTH_MIN, dim // 2))
    band_ids = np.arange(dim) // bw % 2
    colors = np.array([bg1, bg2], dtype=int)
    if band_dir == "h":
        canvas[:] = colors[band_ids][:, None]
    else:
        canvas[:] = colors[band_ids][None, :]
    return canvas


def _overlay_composites(canvas: np.ndarray, legends: list[dict]) -> np.ndarray:
    out = canvas.copy()
    for leg in legends:
        comp = leg["transformed_plaque"]
        t, l = leg["comp_top"], leg["comp_left"]
        region = out[t : t + comp.shape[0], l : l + comp.shape[1]]
        write_mask = comp != 0
        region[write_mask] = comp[write_mask]
    return out


def _place_legend(grid: np.ndarray, leg: dict, col_start: int, sep_col: int, marker_r0: int, is_right: bool) -> None:
    grid[:, sep_col] = 6

    mc0 = col_start
    grid[marker_r0 : marker_r0 + 5, mc0 : mc0 + 5] = 0
    grid[marker_r0, mc0 : mc0 + 5] = 6
    grid[marker_r0 + 4, mc0 : mc0 + 5] = 6
    grid[marker_r0 : marker_r0 + 5, mc0] = 6
    grid[marker_r0 : marker_r0 + 5, mc0 + 4] = 6
    grid[marker_r0 + 1 + leg["pr"], mc0 + 1 + leg["pc"]] = 4

    rr0 = marker_r0 + 5
    grid[rr0 : rr0 + 3, mc0 : mc0 + 5] = 0
    grid[rr0 : rr0 + 3, mc0] = 6
    grid[rr0 : rr0 + 3, mc0 + 4] = 6
    cx_r, cx_c = rr0 + 1, mc0 + 2
    dr, dc = ROTATION_DIRS[leg["rot_k"]]
    grid[cx_r, cx_c] = FLIP_TO_MARKER_COLOR[leg["flip_mode"]]
    grid[cx_r + dr, cx_c + dc] = 7

    grid[1 : 1 + SPRITE_N, mc0 + 1 : mc0 + 4] = leg["legend_sprite"]

    frame_h = leg["frame"].shape[0]
    avail = marker_r0 - (SPRITE_N + 1)
    gap = random.randint(1, max(1, avail - frame_h))
    fr_start = SPRITE_N + 1 + gap
    grid[fr_start : fr_start + frame_h, mc0 : mc0 + 5] = leg["frame"]


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_H_RANGE)
    num_legends = random.choices([1, 2], weights=LEGEND_COUNT_WEIGHTS)[0]

    left_total = LEGEND_W + 1
    right_total = (1 + LEGEND_W) if num_legends == 2 else 0
    canvas_w = GRID_W - left_total - right_total
    if canvas_w < MIN_CANVAS_W:
        return None

    colors = _allocate_colors(num_legends)
    if colors is None:
        return None
    unit_color_sets, (bg1, bg2) = colors

    marker_r0 = grid_h - MARKER_PANEL_HEIGHT
    legends = [{} for _ in range(num_legends)]
    for li in range(num_legends):
        if not _design_legend(legends[li], unit_color_sets[li], marker_r0):
            return None

    if not any(int(leg["flip_mode"]) != FLIP_NONE for leg in legends):
        return None

    if not _assign_anchors(legends, grid_h, canvas_w):
        return None

    canvas = _build_banded_canvas(grid_h, canvas_w, bg1, bg2)
    output = _overlay_composites(canvas, legends)

    grid = np.zeros((grid_h, GRID_W), dtype=int)
    canvas_c0 = left_total
    grid[:, canvas_c0 : canvas_c0 + canvas_w] = canvas

    _place_legend(grid, legends[0], col_start=0, sep_col=LEGEND_W, marker_r0=marker_r0, is_right=False)
    if num_legends == 2:
        r_sep = canvas_c0 + canvas_w
        _place_legend(grid, legends[1], col_start=r_sep + 1, sep_col=r_sep, marker_r0=marker_r0, is_right=True)

    try:
        if np.array_equal(main(grid), output):
            return grid
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
