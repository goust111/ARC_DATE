# Variant of a32d8b75
# Dimension: D1 — 匹配条件：精灵组装方式 (match_cond)
# Change: s1（单元精灵）替换 d1（框架模板）中每个活跃像素，形成分形式的放大复合精灵
#   -> 形状匹配嵌入：单元精灵形状必须匹配框架中的凹槽形状

import random

import numpy as np
from arc_utils import bbox_coords, neighbors4
from scipy.ndimage import label as _scipy_label


# Design analysis
# ---------------
# variant:
#   core_change:
#     - The old frame was a coarse mask whose 1-cells were expanded into repeated NxN tiles.
#     - The new frame is a solid 5-column plaque containing 2-3 internal zero-color cavities.
#     - Each cavity is exactly one dihedral transform (rotation / mirror) of the same 3x3 unit sprite.
#     - The solver must detect the sprite's binary silhouette, enumerate its transforms,
#       and fill every cavity with the matching transformed colored sprite.
#   concretization:
#     - The legend still has: unit sprite -> gap -> frame/plaque -> 5x5 anchor square -> 3x3 rotation patch.
#     - To keep the original 5-column legend-reading path viable, the unit sprite is fixed at 3x3.
#     - The plaque stays 5 columns wide; cavities occupy the centered 3x3 lane and stack vertically.
#     - Output remains the canvas only. Each legend produces one assembled plaque, then the whole plaque
#       is rotated by the clock-hand marker and anchored by the 3x3 marker square exactly as before.
#   behavior_trace:
#     - Old local step: for every active frame pixel, stamp the full unit sprite into a scaled block.
#     - New local step: for every enclosed zero component inside the plaque, find which transformed unit
#       mask has exactly the same shape, then insert only that transformed sprite into the hole.
#     - Difference: the output size no longer scales by frame-cell count; it stays at plaque size, and
#       correctness depends on exact shape matching rather than uniform tiling.
#   complexity_calibration:
#     - Every sample uses at least 2 cavities.
#     - At least one cavity per legend uses a non-identity transform; mirrored variants are strongly favored.
#     - The unit sprite is required to be asymmetric in both shape and 2-color arrangement so transforms
#       are visually observable instead of collapsing into equivalent copies.
#   hidden_assumptions_checked:
#     - Original code assumed all nonzero frame pixels form one 8-connected component. That still holds.
#     - Original code treated frame zeros as irrelevant padding. That no longer holds; internal zero
#       components are the actual instructions and must be preserved during assembly.
#     - Original code rotated the frame mask before expansion. Here the solver must fill cavities first,
#       then rotate the finished plaque, otherwise cavity matching would be performed in the wrong basis.
#     - Original extraction only reads a 5-column panel per legend; the new generator therefore keeps
#       the sprite at 3x3 and the plaque at width 5 so the solver can still parse the full legend.
#   placement_legality:
#     - The generator computes the rotated plaque size first, then assigns anchors with full in-bounds
#       checks and pairwise non-overlap checks for the 2-legend case.
#     - Cavities are separated by frame-colored rows, so zero components never merge accidentally.
#   visual_observability:
#     - Humans can see a sparse 3x3 sprite in the legend and matching empty silhouettes inside the plaque.
#     - The output shows those silhouettes filled by rotated / mirrored colored copies, which is a direct
#       visual cue for the new rule from only a few examples.


SPRITE_N = 3
GRID_H_RANGE = (22, 30)
GRID_W = 30
LEGEND_COUNT_WEIGHTS = [1, 4]
MIN_CANVAS_W = 8
MARKER_PANEL_HEIGHT = 8
BAND_WIDTH_MIN = 4
GEN_RETRIES = 400
USABLE_COLORS = [1, 2, 3, 5, 7, 8, 9]
ROTATION_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
FOUR_CONN = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)


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


def _position_indices_from_border_square(legend: np.ndarray, border6_r0: int, border6_c0: int) -> tuple[int, int] | None:
    inner = legend[border6_r0 + 1 : border6_r0 + 4, border6_c0 + 1 : border6_c0 + 4]
    pts = np.argwhere((inner != 0) & (inner != 6))
    if pts.size == 0:
        return None
    return int(pts[0, 0]), int(pts[0, 1])


def _rotation_from_clockhand(legend: np.ndarray, border6_r0: int, border6_c0: int) -> int:
    sub = legend[border6_r0 + 5 : border6_r0 + 8, border6_c0 + 1 : border6_c0 + 4]
    pts = np.argwhere((sub != 0) & (sub != 6))
    assert pts.size > 0, "no rotation marker pixel found"
    center = np.array([1, 1], dtype=int)
    dists = np.abs(pts - center).sum(axis=1)
    p = pts[int(dists.argmax())]
    dr, dc = int(p[0] - 1), int(p[1] - 1)
    if abs(dr) >= abs(dc):
        return 0 if dr < 0 else 2
    return 1 if dc > 0 else 3


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


def _apply_dihedral(arr: np.ndarray, flip: bool, rot_k: int) -> np.ndarray:
    base = np.fliplr(arr) if flip else arr
    return np.rot90(base, rot_k).copy()


def _sprite_variants_by_mask(unit_sprite: np.ndarray) -> list[dict]:
    variants: list[dict] = []
    seen: set[tuple[tuple[int, int], tuple[int, ...]]] = set()
    for flip in (False, True):
        for rot_k in range(4):
            spr = _apply_dihedral(unit_sprite, flip, rot_k)
            mask = (spr != 0) & (spr != 6)
            key = (mask.shape, tuple(mask.astype(int).flatten()))
            if key in seen:
                continue
            seen.add(key)
            variants.append({"flip": flip, "rot_k": rot_k, "sprite": spr, "mask": mask})
    return variants


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


def _assemble_plaque(unit_sprite: np.ndarray, frame: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    variants = _sprite_variants_by_mask(unit_sprite)
    comp = frame.copy()
    matched: list[dict] = []
    cavities = _frame_cavities(frame)
    assert len(cavities) >= 1, "frame must contain at least one enclosed cavity"
    for r0, c0, cav_mask in cavities:
        fits = [v for v in variants if v["mask"].shape == cav_mask.shape and np.array_equal(v["mask"], cav_mask)]
        assert len(fits) == 1, "cavity shape must match exactly one sprite transform"
        fit = fits[0]
        sub = comp[r0 : r0 + cav_mask.shape[0], c0 : c0 + cav_mask.shape[1]]
        sub[cav_mask] = fit["sprite"][fit["mask"]]
        matched.append({"r0": r0, "c0": c0, "flip": fit["flip"], "rot_k": fit["rot_k"]})
    return comp, matched


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble shape-matched plaques from legend panels and place them onto the canvas.

    Input structure:
    - Color-6 separator columns still split the grid into a left legend, the main canvas, and an optional right legend.
    - Each legend panel contains a sparse 3x3 unit sprite, then a solid plaque whose internal zero-color cavities encode transformed copies of that sprite, then a 5x5 anchor square and a 3x3 rotation marker.

    Rule:
    - Read the unit sprite, enumerate its rotations and mirrors, and match those transformed silhouettes to every enclosed cavity in the plaque.
    - Fill each cavity with the corresponding transformed colored sprite, rotate the fully assembled plaque according to the clock-hand, and place it on the canvas using the anchor position from the 5x5 marker square.

    Key insight:
    - The zero cells inside the plaque are not background padding; they are exact shape slots. Solving the task requires matching the sprite's silhouette under dihedral transforms before the final plaque rotation is applied.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert w >= 2, "grid width must be >= 2"

    left_sep, right_sep = _split_canvas(g)
    canvas_c0 = left_sep[1] + 1
    canvas_c1 = right_sep[0] if right_sep is not None else w
    main_canvas = g[:, canvas_c0:canvas_c1].copy()

    marker_squares = _all_marker_panel_squares(g)
    assert marker_squares, "no marker panel found"

    eff_h, eff_w = main_canvas.shape
    full6_cols = [c for c in range(eff_w) if np.all(main_canvas[:, c] == 6)]
    if full6_cols:
        eff_w = min(full6_cols)
    full6_rows = [r for r in range(eff_h) if np.all(main_canvas[r, :] == 6)]
    if full6_rows:
        eff_h = min(full6_rows)

    out = main_canvas.copy()
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
        frame_mask = (frame != 0) & (frame != 6)
        _, n8 = _scipy_label(frame_mask, structure=np.ones((3, 3), dtype=int))
        assert n8 == 1, f"frame must be 8-connected, got {n8} components"

        plaque, _matches = _assemble_plaque(unit_sprite, frame)
        dir_k = _rotation_from_clockhand(g, marker_r0, marker_c0)
        comp = np.rot90(plaque, (-dir_k) % 4)

        pos = _position_indices_from_border_square(g, marker_r0, marker_c0)
        if pos is None:
            continue
        pr, pc = pos
        top = _anchor_to_top_left(pr, eff_h, comp.shape[0])
        left = _anchor_to_top_left(pc, eff_w, comp.shape[1])

        ch, cw = comp.shape
        for r in range(ch):
            rr = top + r
            assert 0 <= rr < out.shape[0], "sprite row out of canvas bounds"
            for c in range(cw):
                cc = left + c
                assert 0 <= cc < out.shape[1], "sprite col out of canvas bounds"
                if out[rr, cc] == 6:
                    continue
                val = int(comp[r, c])
                if val != 0 and val != 6:
                    assert not composite_written[rr, cc], "composites must not overlap"
                    composite_written[rr, cc] = True
                    out[rr, cc] = val
    return out


def _allocate_colors(num_legends: int):
    pool = list(USABLE_COLORS)
    random.shuffle(pool)
    if len(pool) < 2 * num_legends + 2:
        return None
    unit_color_sets = []
    for _ in range(num_legends):
        unit_color_sets.append((pool.pop(), pool.pop()))
    bg_colors = (pool.pop(), pool.pop())
    if not pool:
        return None
    frame_colors = [random.choice(pool) for _ in range(num_legends)]
    return unit_color_sets, bg_colors, frame_colors


def _color_swap(sprite: np.ndarray, c1: int, c2: int) -> np.ndarray:
    out = sprite.copy()
    out[sprite == c1] = -1
    out[sprite == c2] = c1
    out[out == -1] = c2
    return out


def _is_connected(mask: np.ndarray) -> bool:
    labs, nlab = _scipy_label(mask.astype(int), structure=FOUR_CONN)
    return nlab == 1


def _generate_unit_sprite(c1: int, c2: int) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    for _ in range(500):
        mask = np.zeros((SPRITE_N, SPRITE_N), dtype=bool)
        cells = random.randint(4, 6)
        for idx in random.sample(range(SPRITE_N * SPRITE_N), cells):
            mask[idx // SPRITE_N, idx % SPRITE_N] = True
        if not _is_connected(mask):
            continue
        if not np.all(mask.any(axis=0)) or not np.all(mask.any(axis=1)):
            continue
        mask_variants = _sprite_variants_by_mask(mask.astype(int))
        if len(mask_variants) < 6:
            continue

        true_ids = [idx for idx in range(SPRITE_N * SPRITE_N) if mask.flat[idx]]
        num_c2 = random.randint(1, len(true_ids) - 1)
        c2_ids = set(random.sample(true_ids, num_c2))
        sprite = np.zeros((SPRITE_N, SPRITE_N), dtype=int)
        for idx in true_ids:
            sprite.flat[idx] = c2 if idx in c2_ids else c1

        sprite_variants = _sprite_variants_by_mask(sprite)
        if len(sprite_variants) < 6:
            continue
        rot_only = {
            (v["sprite"].shape, tuple(v["sprite"].flatten()))
            for v in sprite_variants
            if not v["flip"]
        }
        if not any((v["sprite"].shape, tuple(v["sprite"].flatten())) not in rot_only for v in sprite_variants if v["flip"]):
            continue

        return sprite, _color_swap(sprite, c1, c2), sprite_variants
    raise RuntimeError("failed to generate asymmetric 3x3 unit sprite")


def _choose_transforms(variants: list[dict], cavity_count: int) -> list[int]:
    identity = None
    for idx, var in enumerate(variants):
        if not var["flip"] and var["rot_k"] == 0:
            identity = idx
            break
    assert identity is not None
    flip_ids = [idx for idx, var in enumerate(variants) if var["flip"]]
    non_identity = [idx for idx in range(len(variants)) if idx != identity]
    for _ in range(200):
        chosen = []
        if flip_ids:
            chosen.append(random.choice(flip_ids))
        while len(chosen) < cavity_count:
            cand = random.choice(non_identity if len(chosen) == 0 else list(range(len(variants))))
            if cand not in chosen:
                chosen.append(cand)
        if len(set(chosen)) < cavity_count:
            continue
        if all(idx == identity for idx in chosen):
            continue
        return chosen
    raise RuntimeError("failed to choose distinct cavity transforms")


def _build_frame(frame_color: int, variants: list[dict], transform_ids: list[int], inner_gap: int) -> np.ndarray:
    frame_h = 2 + len(transform_ids) * SPRITE_N + (len(transform_ids) - 1) * inner_gap
    frame = np.full((frame_h, 5), frame_color, dtype=int)
    row = 1
    for idx in transform_ids:
        mask = variants[idx]["mask"]
        frame[row : row + SPRITE_N, 1:4][mask] = 0
        row += SPRITE_N + inner_gap
    return frame


def _design_legend(leg: dict, unit_colors: tuple[int, int], frame_color: int, marker_r0: int) -> bool:
    c1, c2 = unit_colors
    sprite, legend_sprite, variants = _generate_unit_sprite(c1, c2)

    avail = marker_r0 - (SPRITE_N + 1)
    options: list[tuple[int, int]] = []
    for cavity_count in (3, 2):
        for inner_gap in (1, 2):
            frame_h = 2 + cavity_count * SPRITE_N + (cavity_count - 1) * inner_gap
            if avail >= frame_h + 1:
                options.append((cavity_count, inner_gap))
    if not options:
        return False

    cavity_count, inner_gap = random.choice(options)
    transform_ids = _choose_transforms(variants, cavity_count)
    frame = _build_frame(frame_color, variants, transform_ids, inner_gap)
    plaque, matched = _assemble_plaque(sprite, frame)

    leg.update(
        sprite=sprite,
        legend_sprite=legend_sprite,
        frame=frame,
        frame_color=frame_color,
        variants=variants,
        transform_ids=transform_ids,
        cavity_count=cavity_count,
        matched=matched,
        rot_k=random.randint(0, 3),
        plaque=plaque,
    )
    leg["rotated_plaque"] = np.rot90(plaque, (-leg["rot_k"]) % 4)
    return True


def _comp_size(leg: dict) -> tuple[int, int]:
    return leg["rotated_plaque"].shape


def _assign_anchors(legends: list[dict], grid_h: int, canvas_w: int) -> bool:
    if len(legends) == 1:
        pr, pc = random.randint(0, 2), random.randint(0, 2)
        ch, cw = _comp_size(legends[0])
        top = _anchor_to_top_left(pr, grid_h, ch)
        left = _anchor_to_top_left(pc, canvas_w, cw)
        if top < 0 or left < 0 or top + ch > grid_h or left + cw > canvas_w:
            return False
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
        comp = leg["rotated_plaque"]
        t, l = leg["comp_top"], leg["comp_left"]
        region = out[t : t + comp.shape[0], l : l + comp.shape[1]]
        write_mask = comp != 0
        region[write_mask] = comp[write_mask]
    return out


def _place_legend(grid: np.ndarray, leg: dict, col_start: int, sep_col: int, marker_r0: int, is_right: bool) -> None:
    grid[:, sep_col] = 6

    mc0 = col_start if not is_right else sep_col
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
    grid[cx_r, cx_c] = 7
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

    legend_w = SPRITE_N + 2
    left_total = legend_w + 1
    right_total = (1 + legend_w) if num_legends == 2 else 0
    canvas_w = GRID_W - left_total - right_total
    if canvas_w < MIN_CANVAS_W:
        return None

    colors = _allocate_colors(num_legends)
    if colors is None:
        return None
    unit_color_sets, (bg1, bg2), frame_colors = colors

    marker_r0 = grid_h - MARKER_PANEL_HEIGHT
    legends = [{} for _ in range(num_legends)]
    for li in range(num_legends):
        if not _design_legend(legends[li], unit_color_sets[li], frame_colors[li], marker_r0):
            return None

    if not _assign_anchors(legends, grid_h, canvas_w):
        return None

    canvas = _build_banded_canvas(grid_h, canvas_w, bg1, bg2)
    output = _overlay_composites(canvas, legends)

    grid = np.zeros((grid_h, GRID_W), dtype=int)
    canvas_c0 = left_total
    grid[:, canvas_c0 : canvas_c0 + canvas_w] = canvas
    _place_legend(grid, legends[0], col_start=0, sep_col=legend_w, marker_r0=marker_r0, is_right=False)
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
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
