# Combo variant of a25697e4 (index 8)
# combo_spec: D6/3+D14/0
# Name: 异色框架颜色重映射
#
# Design analysis:
# - Concrete behavior trace:
#   Original task docks a 2-color tool into a border frame and only changes colors via
#   a flip-dependent swap. Here the frame itself has two visible sub-components: a full
#   rectangle `bar` and a notched rectangle `notch`, with different colors. After docking,
#   the tool no longer keeps either original color: the region that exactly fills the cavity
#   is recolored to the notch-component color, while the protruding handle is recolored to
#   the bar-component color.
# - Example difference:
#   If an input tool uses colors {red, green} and the target frame uses {yellow bar,
#   blue notch}, the output is not a red/green docking and not a red<->green swap.
#   Instead the cavity-filling region becomes blue and the outward handle becomes yellow.
# - Complexity calibration:
#   The geometry stays close to the source task: 1-2 border frames, staircase cavities,
#   and free 2-color tools with unique plug areas. The visible novelty is strong because
#   every successful docking introduces both frame colors into the tool footprint.
# - Hidden assumptions audited:
#   The original solver implicitly assumed a single frame color and carried a reflection
#   branch that swaps the two tool colors. Both assumptions break here. The new solver
#   therefore:
#   1. identifies the two frame sub-components separately,
#   2. infers which one is the full `bar` and which one is the `notch` piece,
#   3. removes flip-based color swapping entirely,
#   4. recolors by structural role (`plug -> notch`, `handle -> bar`) after placement.
# - Placement legality:
#   The generator still uses output-first construction, verifies in-bounds non-overlap,
#   keeps tools away from borders in the input, and rechecks each sample with `main`.
# - Visual observability:
#   Each pair uses four distinct non-background colors: bar, notch, plug, handle. This
#   makes the fusion rule inferable from a few examples because the output tool colors are
#   systematically tied to frame sub-component roles, not to arbitrary palette values.
# - New solver assumptions and generator guarantees:
#   Every frame group has exactly one full-rectangle sub-component and one non-rectangular
#   staircase sub-component; each movable tool has exactly two colors; one color-region is
#   D4-congruent to exactly one cavity. The generator constructs samples to satisfy those
#   conditions directly.

from __future__ import annotations

from collections import Counter
import itertools
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, get_bg, paste, random_free_location, random_sprite


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock detached two-color tools into two-color border frames with staircase cavities.

    Input structure:
    - Background is the most common color.
    - Border-touching foreground components form anchor frames, each made from two nearby
      sub-components of different colors: one solid bar and one notched piece.
    - Non-border foreground components are movable 2-color tools; one color-region matches
      a frame cavity under rotation/flip, while the other color-region is the handle.

    Rule:
    - Match each tool to the frame whose cavity is shape-compatible with one tool color-region.
    - Rotate/flip the whole tool so that the matching region fills the cavity exactly and the
      remaining region protrudes outward from the frame.
    - Recolor the docked result by frame role: plug pixels take the notched sub-component's
      color and handle pixels take the solid bar's color. Remove the tool from its original spot.

    Key insight: The frame's two differently colored sub-components define the output palette:
    the cavity-filling region inherits the notch color, and the protruding region inherits the bar color.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    def bbox_gap(a, b) -> int:
        (ar0, ac0, ar1, ac1) = a
        (br0, bc0, br1, bc1) = b

        def gap1(x0, x1, y0, y1):
            if y1 < x0:
                return int(x0 - y1)
            if x1 < y0:
                return int(y0 - x1)
            return 0

        return gap1(ar0, ar1, br0, br1) + gap1(ac0, ac1, bc0, bc1)

    def normalize_shape(coords) -> frozenset[tuple[int, int]]:
        pts = list(coords)
        r0 = min(r for r, _ in pts)
        c0 = min(c for _, c in pts)
        return frozenset((int(r - r0), int(c - c0)) for r, c in pts)

    def rot90_shape(shape: frozenset[tuple[int, int]]) -> frozenset[tuple[int, int]]:
        return normalize_shape((c, -r) for r, c in shape)

    def flip_shape(shape: frozenset[tuple[int, int]]) -> frozenset[tuple[int, int]]:
        return normalize_shape((r, -c) for r, c in shape)

    def dihedral_shapes(shape: frozenset[tuple[int, int]]) -> set[frozenset[tuple[int, int]]]:
        out: set[frozenset[tuple[int, int]]] = set()
        cur = normalize_shape(shape)
        for _ in range(4):
            out.add(cur)
            out.add(flip_shape(cur))
            cur = rot90_shape(cur)
        return out

    fg = g != bg
    lab, n = label(fg, structure=STRUCT4)
    fixed_mask = np.zeros((H, W), dtype=bool)
    fixed_components: list[dict] = []
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == cid)
        pts_list = [(int(r), int(c)) for r, c in pts]
        touches = any(r == 0 or r == H - 1 or c == 0 or c == W - 1 for r, c in pts_list)
        if not touches:
            continue
        bbox = (
            int(pts[:, 0].min()),
            int(pts[:, 1].min()),
            int(pts[:, 0].max()),
            int(pts[:, 1].max()),
        )
        fixed_components.append(
            {
                "pts": pts_list,
                "bbox": bbox,
                "color": int(g[pts_list[0]]),
                "area": int(len(pts_list)),
            }
        )
        for r, c in pts_list:
            fixed_mask[r, c] = True

    assert fixed_components, "expected at least one border frame component"

    parent = list(range(len(fixed_components)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(fixed_components)):
        for j in range(i + 1, len(fixed_components)):
            if bbox_gap(fixed_components[i]["bbox"], fixed_components[j]["bbox"]) <= 2:
                union(i, j)

    grouped: dict[int, list[int]] = {}
    for i in range(len(fixed_components)):
        grouped.setdefault(find(i), []).append(i)

    anchor_groups: list[dict] = []
    for idxs in grouped.values():
        pts = [p for i in idxs for p in fixed_components[i]["pts"]]
        rs = [r for r, _ in pts]
        cs = [c for _, c in pts]
        bbox = (min(rs), min(cs), max(rs), max(cs))

        mask = np.zeros((H, W), dtype=bool)
        for r, c in pts:
            mask[r, c] = True

        sock = np.zeros((H, W), dtype=bool)
        for r in range(bbox[0], bbox[2] + 1):
            for c in range(bbox[1], bbox[3] + 1):
                if int(g[r, c]) != bg:
                    continue
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < H and 0 <= cc < W and mask[rr, cc]:
                        sock[r, c] = True
                        break

        cavity = np.zeros((H, W), dtype=bool)
        sub_bg = g[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1] == bg
        sub_lab, sub_n = label(sub_bg, structure=STRUCT4)
        best_id = None
        best_key = None
        for cid in range(1, int(sub_n) + 1):
            ptsc = np.argwhere(sub_lab == cid)
            area = int(len(ptsc))
            adj = False
            for rr, cc in ptsc:
                r = int(bbox[0] + rr)
                c = int(bbox[1] + cc)
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    r2, c2 = r + dr, c + dc
                    if bbox[0] <= r2 <= bbox[2] and bbox[1] <= c2 <= bbox[3] and mask[r2, c2]:
                        adj = True
                        break
                if adj:
                    break
            if not adj:
                continue
            key = (area, -cid)
            if best_key is None or key > best_key:
                best_key = key
                best_id = int(cid)
        assert best_id is not None, "expected a cavity per frame"
        for rr, cc in np.argwhere(sub_lab == best_id):
            cavity[int(bbox[0] + rr), int(bbox[1] + cc)] = True

        cavity_coords = {(int(r), int(c)) for r, c in np.argwhere(cavity)}
        cavity_shape = normalize_shape(cavity_coords)
        cavity_size = int(len(cavity_coords))

        sub_infos = []
        for i in idxs:
            comp = fixed_components[i]
            r0, c0, r1, c1 = comp["bbox"]
            bbox_area = int((r1 - r0 + 1) * (c1 - c0 + 1))
            sub_infos.append(
                {
                    "color": int(comp["color"]),
                    "area": int(comp["area"]),
                    "bbox": comp["bbox"],
                    "rect_full": int(comp["area"]) == bbox_area,
                }
            )
        assert len(sub_infos) == 2, "generator guarantees exactly two frame sub-components"
        bar_info = next((x for x in sub_infos if x["rect_full"]), None)
        notch_info = next((x for x in sub_infos if not x["rect_full"]), None)
        assert bar_info is not None and notch_info is not None, "need one solid bar and one notched piece"

        dir_list = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        dir_counts = [0, 0, 0, 0]
        for r, c in pts:
            for di, (dr, dc) in enumerate(dir_list):
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == bg:
                    dir_counts[di] += 1
        open_dir = dir_list[int(np.argmax(dir_counts))]

        anchor_groups.append(
            {
                "mask": mask,
                "bbox": bbox,
                "socket_mask": sock,
                "cavity_mask": cavity,
                "cavity_coords": cavity_coords,
                "cavity_shape": cavity_shape,
                "cavity_size": cavity_size,
                "open_dir": open_dir,
                "bar_color": int(bar_info["color"]),
                "notch_color": int(notch_info["color"]),
                "size": int(mask.sum()),
            }
        )

    anchor_groups.sort(key=lambda d: -d["size"])

    mov_mask = fg & (~fixed_mask)
    mov_lab, mov_n = label(mov_mask, structure=STRUCT4)
    mov_comps: list[dict] = []
    for cid in range(1, int(mov_n) + 1):
        pts = np.argwhere(mov_lab == cid)
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        m = mov_lab[r0 : r1 + 1, c0 : c1 + 1] == cid
        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        patch[~m] = 0
        colors = sorted({int(x) for x in patch[m].flatten() if int(x) != 0})
        color_shapes = {
            col: normalize_shape((int(rr), int(cc)) for rr, cc in np.argwhere((patch == int(col)) & m))
            for col in colors
        }
        mov_comps.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "mask": m,
                "patch": patch,
                "size": int(len(pts)),
                "colors": colors,
                "color_shapes": color_shapes,
            }
        )
    mov_comps.sort(key=lambda d: -d["size"])

    def choose_group(bbox) -> int:
        best_i = 0
        best_d = None
        for i, grp in enumerate(anchor_groups):
            d = bbox_gap(bbox, grp["bbox"])
            if best_d is None or d < best_d:
                best_d = d
                best_i = i
        return int(best_i)

    def compatible(ci: int, gi: int) -> bool:
        cavity_shape = anchor_groups[gi]["cavity_shape"]
        for shape in mov_comps[ci]["color_shapes"].values():
            if cavity_shape in dihedral_shapes(shape):
                return True
        return False

    def gen_variants(patch: np.ndarray, mask: np.ndarray):
        base: list[tuple[np.ndarray, np.ndarray, bool]] = []
        for k in range(4):
            p = np.rot90(patch, k)
            m = np.rot90(mask, k)
            base.append((p, m, False))
            base.append((np.fliplr(p), np.fliplr(m), True))
        seen = set()
        uniq: list[tuple[np.ndarray, np.ndarray, bool]] = []
        for p, m, is_flip in base:
            key = (m.tobytes(), p[m].tobytes())
            if key in seen:
                continue
            seen.add(key)
            uniq.append((p, m, is_flip))
        return uniq

    out = np.full_like(g, bg)
    out[fixed_mask] = g[fixed_mask]
    placed_global = fixed_mask.copy()
    placed_groups = [grp["mask"].copy() for grp in anchor_groups]

    if len(mov_comps) == len(anchor_groups) and len(anchor_groups) > 1:
        best_perm = None
        best_score = None
        for perm in itertools.permutations(range(len(anchor_groups))):
            dist_sum = 0
            ok = True
            for ci, gi in enumerate(perm):
                if not compatible(ci, gi):
                    ok = False
                    break
                dist_sum += bbox_gap(mov_comps[ci]["bbox"], anchor_groups[gi]["bbox"])
            if not ok:
                continue
            score = (-int(dist_sum),)
            if best_score is None or score > best_score:
                best_score = score
                best_perm = perm
        if best_perm is not None:
            group_for_comp = [int(x) for x in best_perm]
        else:
            group_for_comp = []
            for ci, comp in enumerate(mov_comps):
                best_g = None
                best_d = None
                for gi, grp in enumerate(anchor_groups):
                    if not compatible(ci, gi):
                        continue
                    d = bbox_gap(comp["bbox"], grp["bbox"])
                    if best_d is None or d < best_d:
                        best_d = d
                        best_g = gi
                group_for_comp.append(int(best_g) if best_g is not None else int(choose_group(comp["bbox"])))
    else:
        group_for_comp = []
        for ci, comp in enumerate(mov_comps):
            best_g = None
            best_d = None
            for gi, grp in enumerate(anchor_groups):
                if not compatible(ci, gi):
                    continue
                d = bbox_gap(comp["bbox"], grp["bbox"])
                if best_d is None or d < best_d:
                    best_d = d
                    best_g = gi
            group_for_comp.append(int(best_g) if best_g is not None else int(choose_group(comp["bbox"])))

    for comp, gi in zip(mov_comps, group_for_comp):
        grp = anchor_groups[gi]
        placed_grp = placed_groups[gi]
        cav = grp["cavity_mask"]
        cavity_coords = grp["cavity_coords"]
        sock = grp["socket_mask"]
        odr, odc = grp["open_dir"]
        br0, bc0, br1, bc1 = grp["bbox"]

        best_state = None
        best_score = None
        for p, m, _is_flip in gen_variants(comp["patch"], comp["mask"]):
            h, w = m.shape
            coords = np.argwhere(m)
            cols_here = sorted({int(x) for x in p[m].flatten() if int(x) != 0})
            for r0 in range(0, H - h + 1):
                for c0 in range(0, W - w + 1):
                    if any(placed_global[int(r0 + rr), int(c0 + cc)] for rr, cc in coords):
                        continue
                    adj = 0
                    cavity_hits = 0
                    socket_hits = 0
                    outside = 0
                    for rr, cc in coords:
                        r = int(r0 + rr)
                        c = int(c0 + cc)
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            r2, c2 = r + dr, c + dc
                            if 0 <= r2 < H and 0 <= c2 < W and placed_grp[r2, c2]:
                                adj += 1
                        if cav[r, c]:
                            cavity_hits += 1
                        if sock[r, c]:
                            socket_hits += 1
                        if odc == -1 and c < bc0:
                            outside += 1
                        elif odc == 1 and c > bc1:
                            outside += 1
                        elif odr == -1 and r < br0:
                            outside += 1
                        elif odr == 1 and r > br1:
                            outside += 1
                    if adj == 0:
                        continue

                    plug_best = None
                    for plug_col in cols_here:
                        plug_cells = {(int(r0 + rr), int(c0 + cc)) for rr, cc in coords if int(p[rr, cc]) == int(plug_col)}
                        plug_hits = int(len(plug_cells & cavity_coords))
                        plug_outside = int(len(plug_cells - cavity_coords))
                        nonplug_in_cav = 0
                        for rr, cc in coords:
                            if int(p[rr, cc]) == int(plug_col):
                                continue
                            if cav[int(r0 + rr), int(c0 + cc)]:
                                nonplug_in_cav += 1
                        plug_full = int(plug_cells == cavity_coords and nonplug_in_cav == 0)
                        plug_score = (plug_full, plug_hits, -nonplug_in_cav, -plug_outside, int(plug_col))
                        if plug_best is None or plug_score > plug_best:
                            plug_best = plug_score

                    dist = bbox_gap(grp["bbox"], (int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)))
                    score = tuple(plug_best[:-1]) + (
                        int(cavity_hits),
                        int(adj),
                        int(outside),
                        int(socket_hits),
                        -int(dist),
                        -int(r0),
                        -int(c0),
                    )
                    if best_score is None or score > best_score:
                        best_score = score
                        best_state = (p.copy(), m.copy(), int(r0), int(c0), int(plug_best[-1]))

        assert best_state is not None, "must find a valid docking placement"
        p, m, r0, c0, plug_col = best_state
        for rr, cc in np.argwhere(m):
            r = int(r0 + rr)
            c = int(c0 + cc)
            out[r, c] = int(grp["notch_color"] if int(p[rr, cc]) == plug_col else grp["bar_color"])
            placed_global[r, c] = True
            placed_grp[r, c] = True

    return out


from arc_utils import bbox as _bbox

# Canvas
GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (25, 30)
NUM_PAIRS_CHOICES = [1, 2]

# Frame geometry
FRAME_W_RANGE = (4, 8)
NOTCH_N_RANGE = (1, 4)
TOP_H_RANGE = (1, 4)
BOT_H_RANGE = (2, 4)
PLUG_AREA_BOUNDS = (5, 8)

# Tool sizing
HANDLE_AREA_RANGE = (4, 10)
TOOL_TOTAL_BOUNDS = (10, 15)
HANDLE_SPRITE_DIM = (2, 5)

# Placement
UNION_BBOX_GAP_MIN = 2
FRAME_BBOX_GAP_MIN = 3
TOOL_BORDER = 1
TOOL_PADDING = 3
TOOL_PADDING_CONN = 4

# Retries
PAIR_SPEC_RETRIES = 300
HANDLE_SPRITE_RETRIES = 30
GEN_ATTEMPT_RETRIES = 300


def _bbox_gap(a, b) -> int:
    (ar0, ac0, ar1, ac1) = a
    (br0, bc0, br1, bc1) = b

    def _gap1(x0, x1, y0, y1):
        if y1 < x0:
            return int(x0 - y1)
        if x1 < y0:
            return int(y0 - x1)
        return 0

    return _gap1(ar0, ar1, br0, br1) + _gap1(ac0, ac1, bc0, bc1)


def _transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.rot90(arr, k)
    if flip:
        out = np.fliplr(out)
    return out


def _orient_left_to_side(arr: np.ndarray, side: str) -> np.ndarray:
    if side == "left":
        return arr.copy()
    if side == "right":
        return np.fliplr(arr)
    if side == "top":
        return np.rot90(arr, 3)
    if side == "bottom":
        return np.rot90(arr, 1)
    raise ValueError(side)


def _notch_offsets(notch_n: int, fw: int, bot_h: int) -> list[tuple[int, int]]:
    if notch_n == 1:
        return [(0, fw - 1)]
    if notch_n == 2:
        return [(0, fw - 2), (0, fw - 1)]
    if notch_n == 3:
        if bot_h >= 2:
            return [(0, fw - 2), (0, fw - 1), (1, fw - 1)]
        return [(0, fw - 3), (0, fw - 2), (0, fw - 1)]
    raise ValueError(notch_n)


def _make_handle_sprite(handle_area: int, bg: int, handle_color: int) -> np.ndarray | None:
    lo, hi = HANDLE_SPRITE_DIM
    dims = [
        (h, w)
        for h in range(lo, hi)
        for w in range(lo, hi)
        if handle_area <= h * w <= handle_area * 2
    ]
    if not dims:
        return None
    for _ in range(HANDLE_SPRITE_RETRIES):
        h, w = random.choice(dims)
        sprite = random_sprite(
            h,
            w,
            density=handle_area / (h * w),
            color_palette=[handle_color],
            connectivity=4,
            background=bg,
        )
        if int(np.sum(sprite != bg)) == handle_area:
            return sprite
    return None


def _build_left_pair(
    *,
    bg: int,
    bar_color: int,
    notch_color: int,
    plug_color: int,
    handle_color: int,
    fw: int,
    top_h: int,
    bot_h: int,
    notch_n: int,
    handle_area: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    fh = top_h + 1 + bot_h
    frame = np.full((fh, fw), bg, dtype=int)
    frame[:top_h, :] = int(bar_color)
    frame[top_h + 1 :, :] = int(notch_color)
    for dr, dc in _notch_offsets(notch_n, fw, bot_h):
        frame[top_h + 1 + dr, dc] = int(bg)

    cavity = frame == int(bg)
    attach_rows = {r for r in range(fh) if cavity[r, fw - 1]}

    handle_sprite = _make_handle_sprite(handle_area, bg, handle_color)
    if handle_sprite is None:
        return None

    hs_h, hs_w = handle_sprite.shape
    sprite_left_rows = {r for r in range(hs_h) if handle_sprite[r, 0] != bg}
    valid_offsets = [voff for voff in range(fh) if any((voff + sr) in attach_rows for sr in sprite_left_rows)]
    if not valid_offsets:
        return None

    voff = random.choice(valid_offsets)
    total_h = max(fh, voff + hs_h)
    total_w = fw + hs_w

    frame_full = np.full((total_h, total_w), bg, dtype=int)
    frame_full[:fh, :fw] = frame

    tool = np.full((total_h, total_w), bg, dtype=int)
    tool[:fh, :fw][cavity] = int(plug_color)
    handle_mask = handle_sprite != bg
    tool[voff : voff + hs_h, fw : fw + hs_w][handle_mask] = int(handle_color)
    return frame_full, tool


def _build_param_options() -> list[tuple[int, int, int, int]]:
    fw_lo, fw_hi = FRAME_W_RANGE
    nn_lo, nn_hi = NOTCH_N_RANGE
    th_lo, th_hi = TOP_H_RANGE
    bh_lo, bh_hi = BOT_H_RANGE
    pa_lo, pa_hi = PLUG_AREA_BOUNDS
    return [
        (fw, top_h, bot_h, notch_n)
        for fw in range(fw_lo, fw_hi)
        for notch_n in range(nn_lo, nn_hi)
        if pa_lo <= fw + notch_n <= pa_hi
        for top_h in range(th_lo, th_hi)
        for bot_h in range(bh_lo, bh_hi)
    ]


PARAM_OPTIONS = _build_param_options()


def _pick_pair_specs(num_pairs: int) -> list[dict]:
    ha_lo, ha_hi = HANDLE_AREA_RANGE
    tt_lo, tt_hi = TOOL_TOTAL_BOUNDS
    specs: list[dict] = []
    for _ in range(PAIR_SPEC_RETRIES):
        specs.clear()
        used_plug_areas: set[int] = set()
        for _ in range(num_pairs):
            candidates = [x for x in PARAM_OPTIONS if x[0] + x[3] not in used_plug_areas]
            if not candidates:
                specs = []
                break
            fw, top_h, bot_h, notch_n = random.choice(candidates)
            plug_area = int(fw + notch_n)
            specs.append(
                {
                    "fw": int(fw),
                    "top_h": int(top_h),
                    "bot_h": int(bot_h),
                    "notch_n": int(notch_n),
                    "plug_area": plug_area,
                }
            )
            used_plug_areas.add(plug_area)
        if len(specs) != num_pairs:
            continue
        all_plug_areas = {int(s["plug_area"]) for s in specs}
        ok = True
        for spec in specs:
            pa = int(spec["plug_area"])
            handle_choices = [
                h
                for h in range(ha_lo, ha_hi)
                if tt_lo <= pa + h <= tt_hi and h != pa and h not in all_plug_areas
            ]
            if not handle_choices:
                ok = False
                break
            spec["handle_area"] = int(random.choice(handle_choices))
        if ok:
            return [dict(s) for s in specs]
    return []


def _candidate_origins(side: str, ph: int, pw: int, H: int, W: int) -> list[tuple[int, int]]:
    if side == "left":
        origins = [(r0, 0) for r0 in range(H - ph + 1)]
    elif side == "right":
        origins = [(r0, W - pw) for r0 in range(H - ph + 1)]
    elif side == "top":
        origins = [(0, c0) for c0 in range(W - pw + 1)]
    elif side == "bottom":
        origins = [(H - ph, c0) for c0 in range(W - pw + 1)]
    else:
        raise ValueError(side)
    random.shuffle(origins)
    return origins


def _variant_choices(patch: np.ndarray) -> list[np.ndarray]:
    seen = set()
    out: list[np.ndarray] = []
    mask = patch != 0
    for k in range(4):
        for flip in (False, True):
            q = _transform(patch, k, flip)
            m = _transform(mask.astype(np.uint8), k, flip).astype(bool)
            key = (m.tobytes(), q[m].tobytes())
            if key in seen:
                continue
            seen.add(key)
            out.append(q)
    random.shuffle(out)
    return out


def _place_docked_pair(
    i: int,
    spec: dict,
    colors: list[int],
    side: str,
    bg: int,
    H: int,
    W: int,
    output: np.ndarray,
    input_grid: np.ndarray,
    union_bboxes: list,
    frame_bboxes: list,
) -> dict | None:
    base = 4 * i
    bar_color = int(colors[base])
    notch_color = int(colors[base + 1])
    plug_color = int(colors[base + 2])
    handle_color = int(colors[base + 3])

    result = _build_left_pair(
        bg=bg,
        bar_color=bar_color,
        notch_color=notch_color,
        plug_color=plug_color,
        handle_color=handle_color,
        fw=int(spec["fw"]),
        top_h=int(spec["top_h"]),
        bot_h=int(spec["bot_h"]),
        notch_n=int(spec["notch_n"]),
        handle_area=int(spec["handle_area"]),
    )
    if result is None:
        return None
    frame_left, tool_left = result

    docked_left = tool_left.copy()
    docked_left[tool_left == plug_color] = notch_color
    docked_left[tool_left == handle_color] = bar_color
    scene_left = np.where(docked_left != bg, docked_left, frame_left)

    frame_scene = _orient_left_to_side(frame_left, side)
    tool_scene = _orient_left_to_side(tool_left, side)
    scene = _orient_left_to_side(scene_left, side)
    ph, pw = scene.shape
    fr0, fc0, fr1, fc1 = _bbox(frame_scene != bg)

    origin = None
    for r0, c0 in _candidate_origins(side, ph, pw, H, W):
        union_bbox = (int(r0), int(c0), int(r0 + ph - 1), int(c0 + pw - 1))
        frame_bbox = (int(r0 + fr0), int(c0 + fc0), int(r0 + fr1), int(c0 + fc1))
        if side in ("left", "right") and (frame_bbox[0] == 0 or frame_bbox[2] == H - 1):
            continue
        if side in ("top", "bottom") and (frame_bbox[1] == 0 or frame_bbox[3] == W - 1):
            continue
        if any(_bbox_gap(union_bbox, prev) <= UNION_BBOX_GAP_MIN - 1 for prev in union_bboxes):
            continue
        if any(_bbox_gap(frame_bbox, prev) <= FRAME_BBOX_GAP_MIN - 1 for prev in frame_bboxes):
            continue
        if np.any((output[r0 : r0 + ph, c0 : c0 + pw] != bg) & (scene != bg)):
            continue
        origin = (int(r0), int(c0))
        break
    if origin is None:
        return None

    r0, c0 = origin
    paste(output, scene, r0, c0, transparent=bg)
    paste(input_grid, frame_scene, r0, c0, transparent=bg)

    union_bboxes.append((int(r0), int(c0), int(r0 + ph - 1), int(c0 + pw - 1)))
    frame_bboxes.append((int(r0 + fr0), int(c0 + fc0), int(r0 + fr1), int(c0 + fc1)))
    return {"tool": tool_scene}


def _undock_tool(info: dict, bg: int, input_grid: np.ndarray, placement_canvas: np.ndarray) -> bool:
    for patch in _variant_choices(info["tool"]):
        try:
            r0, c0 = random_free_location(
                placement_canvas,
                patch,
                transparent=bg,
                border=TOOL_BORDER,
                padding=TOOL_PADDING,
                padding_connectivity=TOOL_PADDING_CONN,
            )
        except ValueError:
            continue
        paste(input_grid, patch, r0, c0, transparent=bg)
        paste(placement_canvas, patch, r0, c0, transparent=bg)
        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    bg = int(random.choice(list(range(10))))
    num_pairs = int(random.choice(NUM_PAIRS_CHOICES))

    pair_specs = _pick_pair_specs(num_pairs)
    if len(pair_specs) != num_pairs:
        return None

    colors = [c for c in range(10) if c != bg]
    random.shuffle(colors)
    need = 4 * num_pairs
    if len(colors) < need:
        return None
    colors = colors[:need]

    sides = random.sample(["left", "right", "top", "bottom"], k=num_pairs)
    output = np.full((H, W), bg, dtype=int)
    input_grid = np.full((H, W), bg, dtype=int)
    union_bboxes: list[tuple[int, int, int, int]] = []
    frame_bboxes: list[tuple[int, int, int, int]] = []
    placed_tools: list[dict] = []

    for i, spec in enumerate(pair_specs):
        info = _place_docked_pair(
            i,
            spec,
            colors,
            sides[i],
            bg,
            H,
            W,
            output,
            input_grid,
            union_bboxes,
            frame_bboxes,
        )
        if info is None:
            return None
        placed_tools.append(info)

    placement_canvas = output.copy()
    for info in placed_tools:
        if not _undock_tool(info, bg, input_grid, placement_canvas):
            return None

    if np.array_equal(main(input_grid), output):
        return input_grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPT_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPT_RETRIES} attempts")
