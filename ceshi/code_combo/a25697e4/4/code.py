# Combo variant of a25697e4 (index 4)
# combo_spec: D2/3+D6/2
# Name: 标记编码旋转奇偶变色
#
# Design analysis:
# - Concrete behavior trace: in the base task, a tool may use any D4 orientation to fill a frame cavity,
#   and only a reflection can trigger a color swap. Here that search is replaced by a visible code:
#   each frame carries 1-3 marker pixels in one border corner, and the marker count is exactly the
#   number of 90-degree turns the solver must apply to the free tool before docking it. After that
#   forced rotation, odd counts swap the two tool colors and even counts keep them unchanged.
# - Small example: if a tool's plug already has the right cavity area but is rotated 90 degrees away
#   from the socket, a frame with two corner markers means "rotate twice". The solver must not search
#   flips; it rotates the tool by 180 degrees. Because 2 is even, the cavity is filled by the same
#   color it had in the input. A different frame with three markers rotates by 270 degrees, and the
#   plug geometry stays the same while the plug/handle colors exchange roles in the output.
# - Complexity calibration: the geometry stays close to the original task so plug-area matching still
#   disambiguates tool-to-frame pairing, but the added marker strip is visually obvious and repeated
#   on every frame. The generator always uses two frame-tool pairs and assigns one odd rotation and
#   one even rotation so a single training grid already shows both parity outcomes.
# - Hidden-assumption audit: the original solver assumed that any rotation/flip might be needed and
#   that the winning transform itself reveals whether to swap colors. That assumption is removed.
#   The new solver instead assumes each anchor group has exactly one dominant frame color plus 1-3
#   non-dominant marker pixels; the generator guarantees this by recoloring existing border-corner
#   frame cells rather than adding detached decorations.
# - Placement legality: the generator still builds the docked output first, then undocks each tool by
#   inverse rotation only. Random free placement keeps tools away from borders and other objects, and
#   pair layouts still satisfy the original non-overlap and spacing constraints.
# - Visual observability and cleanliness: the markers reuse one dedicated color and sit in a single
#   corner run on the border-touching side of each frame, so they do not clutter the cavity region.
#   The odd/even color behavior is inferable because one pair in each sample swaps colors and the
#   other does not, while both still obey the same cavity-filling geometry.

from __future__ import annotations

from collections import Counter
import itertools
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, get_bg, paste, random_free_location, random_sprite
from arc_utils import bbox as _bbox


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


def _swap_colors(patch: np.ndarray, a: int, b: int) -> np.ndarray:
    out = patch.copy()
    ma = out == int(a)
    mb = out == int(b)
    out[ma] = int(b)
    out[mb] = int(a)
    return out


def _trim_patch(patch: np.ndarray, bg: int) -> np.ndarray:
    bb = _bbox(patch != int(bg))
    if bb is None:
        return patch.copy()
    r0, c0, r1, c1 = bb
    return patch[r0 : r1 + 1, c0 : c1 + 1].copy()


def _extract_anchor_groups(grid: np.ndarray, bg: int):
    H, W = grid.shape
    fg = grid != int(bg)
    lab, n = label(fg, structure=STRUCT4)
    fixed_mask = np.zeros((H, W), dtype=bool)
    fixed_components = []
    for cid in range(1, n + 1):
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
        fixed_components.append({"pts": pts_list, "bbox": bbox})
        for r, c in pts_list:
            fixed_mask[r, c] = True

    assert fixed_components, "expected at least one border-anchored frame"

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
            if _bbox_gap(fixed_components[i]["bbox"], fixed_components[j]["bbox"]) <= 2:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(len(fixed_components)):
        groups.setdefault(find(i), []).append(i)

    anchor_groups = []
    for idxs in groups.values():
        pts = [p for i in idxs for p in fixed_components[i]["pts"]]
        rs = [r for r, _ in pts]
        cs = [c for _, c in pts]
        bbox = (min(rs), min(cs), max(rs), max(cs))
        mask = np.zeros((H, W), dtype=bool)
        for r, c in pts:
            mask[r, c] = True

        cavity = np.zeros((H, W), dtype=bool)
        sub_bg = grid[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1] == int(bg)
        sub_lab, sub_n = label(sub_bg, structure=STRUCT4)
        best = None
        best_id = None
        for cid in range(1, sub_n + 1):
            ptsc = np.argwhere(sub_lab == cid)
            area = int(len(ptsc))
            adjacent = False
            for rr, cc in ptsc:
                r = int(bbox[0] + rr)
                c = int(bbox[1] + cc)
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    r2 = r + dr
                    c2 = c + dc
                    if bbox[0] <= r2 <= bbox[2] and bbox[1] <= c2 <= bbox[3] and mask[r2, c2]:
                        adjacent = True
                        break
                if adjacent:
                    break
            if not adjacent:
                continue
            cand = (area, -cid)
            if best is None or cand > best:
                best = cand
                best_id = cid
        if best_id is not None:
            for rr, cc in np.argwhere(sub_lab == best_id):
                cavity[int(bbox[0] + rr), int(bbox[1] + cc)] = True

        socket_mask = np.zeros((H, W), dtype=bool)
        for r in range(bbox[0], bbox[2] + 1):
            for c in range(bbox[1], bbox[3] + 1):
                if int(grid[r, c]) != int(bg):
                    continue
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < H and 0 <= cc < W and mask[rr, cc]:
                        socket_mask[r, c] = True
                        break

        dir_list = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        dir_counts = [0, 0, 0, 0]
        for r, c in pts:
            for i, (dr, dc) in enumerate(dir_list):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < H and 0 <= cc < W and int(grid[rr, cc]) == int(bg):
                    dir_counts[i] += 1
        open_dir = dir_list[int(np.argmax(dir_counts))]

        color_counts = Counter(int(x) for x in grid[mask].tolist())
        frame_color = int(color_counts.most_common(1)[0][0])
        marker_count = int(sum(cnt for col, cnt in color_counts.items() if int(col) != frame_color))

        anchor_groups.append(
            {
                "mask": mask,
                "bbox": bbox,
                "socket_mask": socket_mask,
                "cavity_mask": cavity,
                "open_dir": open_dir,
                "cavity_size": int(cavity.sum()),
                "size": int(mask.sum()),
                "frame_color": frame_color,
                "rotation_steps": marker_count % 4,
                "marker_count": marker_count,
            }
        )

    anchor_groups.sort(key=lambda d: -d["size"])
    return fixed_mask, anchor_groups


def _extract_movable_components(grid: np.ndarray, bg: int, fixed_mask: np.ndarray):
    fg = grid != int(bg)
    mov_mask = fg & (~fixed_mask)
    mov_lab, mov_n = label(mov_mask, structure=STRUCT4)
    mov_comps = []
    for cid in range(1, mov_n + 1):
        pts = np.argwhere(mov_lab == cid)
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        mask = mov_lab[r0 : r1 + 1, c0 : c1 + 1] == cid
        patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        patch[~mask] = 0
        mov_comps.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "mask": mask,
                "patch": patch,
                "size": int(len(pts)),
            }
        )
    mov_comps.sort(key=lambda d: -d["size"])
    return mov_comps


def _match_groups(mov_comps, anchor_groups):
    comp_color_counts = []
    for comp in mov_comps:
        vals = [int(x) for x in comp["patch"][comp["mask"]].flatten() if int(x) != 0]
        comp_color_counts.append(dict(Counter(vals)))

    def compatible(ci: int, gi: int) -> bool:
        cav_sz = int(anchor_groups[gi]["cavity_size"])
        return any(int(v) == cav_sz for v in comp_color_counts[ci].values())

    if len(mov_comps) == len(anchor_groups) and len(anchor_groups) > 1:
        best_perm = None
        best_score = None
        for perm in itertools.permutations(range(len(anchor_groups))):
            ok = True
            dist_sum = 0
            for ci, gi in enumerate(perm):
                if not compatible(ci, gi):
                    ok = False
                    break
                dist_sum += _bbox_gap(mov_comps[ci]["bbox"], anchor_groups[gi]["bbox"])
            if not ok:
                continue
            score = -int(dist_sum)
            if best_score is None or score > best_score:
                best_score = score
                best_perm = perm
        assert best_perm is not None, "must find a compatible tool-to-frame permutation"
        return [int(x) for x in best_perm]

    group_for_comp = []
    for ci, comp in enumerate(mov_comps):
        best_g = None
        best_d = None
        for gi, grp in enumerate(anchor_groups):
            if not compatible(ci, gi):
                continue
            dist = _bbox_gap(comp["bbox"], grp["bbox"])
            if best_d is None or dist < best_d:
                best_d = dist
                best_g = gi
        assert best_g is not None, "every tool must match some cavity by color-region count"
        group_for_comp.append(int(best_g))
    return group_for_comp


def _apply_rotation_and_parity_swap(patch: np.ndarray, mask: np.ndarray, rotation_steps: int) -> np.ndarray:
    p = np.rot90(patch, rotation_steps)
    m = np.rot90(mask, rotation_steps)
    colors = sorted({int(x) for x in p[m].flatten() if int(x) != 0})
    if rotation_steps % 2 == 1 and len(colors) == 2:
        p = _swap_colors(p, colors[0], colors[1])
    return p


def _best_docking_state(
    comp: dict,
    grp: dict,
    placed_global: np.ndarray,
    placed_grp: np.ndarray,
    bg: int,
):
    H, W = placed_global.shape
    cav = grp["cavity_mask"]
    sock = grp["socket_mask"]
    odr, odc = grp["open_dir"]
    cav_sz = int(grp["cavity_size"])
    rotation_steps = int(grp["rotation_steps"])
    m = np.rot90(comp["mask"], rotation_steps)
    p = _apply_rotation_and_parity_swap(comp["patch"], comp["mask"], rotation_steps)
    h, w = m.shape
    coords = np.argwhere(m)
    cols_here = sorted({int(x) for x in p[m].flatten() if int(x) != 0})
    plug_colors = [int(col) for col in cols_here if int(((p == int(col)) & m).sum()) == cav_sz]
    assert plug_colors, "rotated tool must still expose one cavity-sized color region"

    best_state = None
    best_score = None
    for r0 in range(0, H - h + 1):
        for c0 in range(0, W - w + 1):
            overlaps = False
            for rr, cc in coords:
                r = int(r0 + rr)
                c = int(c0 + cc)
                if placed_global[r, c]:
                    overlaps = True
                    break
            if overlaps:
                continue

            adj = 0
            for rr, cc in coords:
                r = int(r0 + rr)
                c = int(c0 + cc)
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    r2 = r + dr
                    c2 = c + dc
                    if 0 <= r2 < H and 0 <= c2 < W and placed_grp[r2, c2]:
                        adj += 1
            if adj == 0:
                continue

            socket_hits = 0
            cavity_hits = 0
            outside = 0
            br0, bc0, br1, bc1 = grp["bbox"]
            for rr, cc in coords:
                r = int(r0 + rr)
                c = int(c0 + cc)
                if sock[r, c]:
                    socket_hits += 1
                if cav[r, c]:
                    cavity_hits += 1
                if odc == -1 and c < bc0:
                    outside += 1
                elif odc == 1 and c > bc1:
                    outside += 1
                elif odr == -1 and r < br0:
                    outside += 1
                elif odr == 1 and r > br1:
                    outside += 1

            plug_best = None
            for plug_col in plug_colors:
                plug_hits = 0
                plug_outside = 0
                nonplug_in_cav = 0
                for rr, cc in coords:
                    r = int(r0 + rr)
                    c = int(c0 + cc)
                    in_cav = bool(cav[r, c])
                    col = int(p[rr, cc])
                    if col == int(plug_col):
                        if in_cav:
                            plug_hits += 1
                        else:
                            plug_outside += 1
                    elif in_cav:
                        nonplug_in_cav += 1
                plug_full = int(plug_hits == cav_sz and plug_outside == 0 and nonplug_in_cav == 0)
                plug_score = (plug_full, plug_hits, -nonplug_in_cav, -plug_outside)
                if plug_best is None or plug_score > plug_best:
                    plug_best = plug_score

            dist = _bbox_gap(grp["bbox"], (int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)))
            score = tuple(plug_best) + (
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
                best_state = (p, m, int(r0), int(c0))

    assert best_state is not None, "must find a valid placement for every movable tool"
    return best_state


def describe_fusion_input(grid: np.ndarray) -> dict:
    bg = get_bg(grid)
    fixed_mask, anchor_groups = _extract_anchor_groups(grid, bg)
    movable = _extract_movable_components(grid, bg, fixed_mask)
    return {
        "bg": int(bg),
        "rotation_steps": [int(g["rotation_steps"]) for g in anchor_groups],
        "marker_counts": [int(g["marker_count"]) for g in anchor_groups],
        "odd_rotations": int(sum(int(g["rotation_steps"]) % 2 for g in anchor_groups)),
        "num_frames": int(len(anchor_groups)),
        "num_tools": int(len(movable)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock each free two-color tool into a border frame by reading a corner marker code first.

    Input structure:
    - The background is the most frequent color.
    - Border-touching components form frame groups; each frame is mostly one color but has 1-3
      corner marker pixels in a second color, and its interior contains a staircase cavity.
    - Non-border foreground components are movable tools with exactly two color regions.

    Rule:
    - Pair every tool with the frame whose cavity size matches one of the tool's two color-region sizes.
    - Count the frame's corner markers to determine the required number of 90-degree rotations, dock
      the tool using that rotation only, and do not use flips.
    - After rotating, swap the tool's two colors iff the rotation count is odd; then place the cavity-
      sized color region into the socket and let the other region protrude outward.

    Key insight: the corner markers encode both geometry and color behavior, because the same decoded
    rotation count fixes the tool orientation and simultaneously decides whether its two colors trade roles.
    """
    g = grid.copy()
    bg = get_bg(g)
    fixed_mask, anchor_groups = _extract_anchor_groups(g, bg)
    mov_comps = _extract_movable_components(g, bg, fixed_mask)

    out = np.full_like(g, int(bg))
    out[fixed_mask] = g[fixed_mask]
    placed_global = fixed_mask.copy()
    placed_groups = [grp["mask"].copy() for grp in anchor_groups]

    group_for_comp = _match_groups(mov_comps, anchor_groups)
    for comp, gi in zip(mov_comps, group_for_comp):
        grp = anchor_groups[gi]
        p, m, r0, c0 = _best_docking_state(comp, grp, placed_global, placed_groups[gi], bg)
        for rr, cc in np.argwhere(m):
            r = int(r0 + rr)
            c = int(c0 + cc)
            out[r, c] = int(p[rr, cc])
            placed_global[r, c] = True
            placed_groups[gi][r, c] = True

    return out


# --------------- generator ---------------

GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (25, 30)
NUM_PAIRS_CHOICES = [2]

FRAME_W_RANGE = (4, 8)
NOTCH_N_RANGE = (1, 4)
TOP_H_RANGE = (1, 4)
BOT_H_RANGE = (2, 4)
PLUG_AREA_BOUNDS = (5, 8)

HANDLE_AREA_RANGE = (4, 10)
TOOL_TOTAL_BOUNDS = (10, 15)
HANDLE_SPRITE_DIM = (2, 5)

UNION_BBOX_GAP_MIN = 2
FRAME_BBOX_GAP_MIN = 3
TOOL_BORDER = 1
TOOL_PADDING = 3
TOOL_PADDING_CONN = 4

PAIR_SPEC_RETRIES = 300
HANDLE_SPRITE_RETRIES = 30
GEN_ATTEMPT_RETRIES = 300


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
    frame_color: int,
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
    frame[:top_h, :] = frame_color
    frame[top_h + 1 :, :] = frame_color
    for dr, dc in _notch_offsets(notch_n, fw, bot_h):
        frame[top_h + 1 + dr, dc] = bg

    cavity = frame == bg
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
    tool[:fh, :fw][cavity] = plug_color
    handle_mask = handle_sprite != bg
    tool[voff : voff + hs_h, fw : fw + hs_w][handle_mask] = handle_color
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
            plug_area = fw + notch_n
            specs.append(
                {
                    "fw": fw,
                    "top_h": top_h,
                    "bot_h": bot_h,
                    "notch_n": notch_n,
                    "plug_area": plug_area,
                }
            )
            used_plug_areas.add(plug_area)
        if len(specs) != num_pairs:
            continue
        all_plug_areas = {s["plug_area"] for s in specs}
        ok = True
        for spec in specs:
            pa = spec["plug_area"]
            handle_choices = [
                h
                for h in range(ha_lo, ha_hi)
                if tt_lo <= pa + h <= tt_hi and h != pa and h not in all_plug_areas
            ]
            if not handle_choices:
                ok = False
                break
            spec["handle_area"] = random.choice(handle_choices)
        if ok:
            return [dict(s) for s in specs]
    return []


def _pick_rotation_steps(num_pairs: int) -> list[int]:
    if num_pairs == 1:
        return [random.choice([1, 2, 3])]
    steps = [2, random.choice([1, 3])]
    random.shuffle(steps)
    return steps


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


def _mark_frame_corner(frame_scene: np.ndarray, side: str, frame_color: int, marker_color: int, count: int, bg: int) -> np.ndarray:
    out = frame_scene.copy()
    H, W = out.shape
    if side == "left":
        border_cells = [(r, 0) for r in range(H) if int(out[r, 0]) == int(frame_color)]
    elif side == "right":
        border_cells = [(r, W - 1) for r in range(H) if int(out[r, W - 1]) == int(frame_color)]
    elif side == "top":
        border_cells = [(0, c) for c in range(W) if int(out[0, c]) == int(frame_color)]
    elif side == "bottom":
        border_cells = [(H - 1, c) for c in range(W) if int(out[H - 1, c]) == int(frame_color)]
    else:
        raise ValueError(side)
    assert len(border_cells) >= count >= 1
    for r, c in border_cells[:count]:
        out[r, c] = int(marker_color)
    assert np.any(out != int(bg))
    return out


def _place_docked_pair(
    i: int,
    spec: dict,
    marker_color: int,
    colors: list[int],
    side: str,
    rotation_steps: int,
    bg: int,
    H: int,
    W: int,
    output: np.ndarray,
    input_grid: np.ndarray,
    union_bboxes: list,
    frame_bboxes: list,
) -> dict | None:
    frame_color = colors[3 * i]
    plug_color = colors[3 * i + 1]
    handle_color = colors[3 * i + 2]

    result = _build_left_pair(
        bg=bg,
        frame_color=frame_color,
        plug_color=plug_color,
        handle_color=handle_color,
        fw=spec["fw"],
        top_h=spec["top_h"],
        bot_h=spec["bot_h"],
        notch_n=spec["notch_n"],
        handle_area=spec["handle_area"],
    )
    if result is None:
        return None
    frame_left, tool_left = result

    frame_scene = _orient_left_to_side(frame_left, side)
    tool_base = _orient_left_to_side(tool_left, side)
    frame_scene = _mark_frame_corner(frame_scene, side, frame_color, marker_color, rotation_steps, bg)

    tool_final = tool_base.copy()
    if rotation_steps % 2 == 1:
        tool_final = _swap_colors(tool_final, plug_color, handle_color)
    scene = np.where(tool_final != bg, tool_final, frame_scene)

    ph, pw = scene.shape
    frame_bb_local = _bbox(frame_scene != bg)
    assert frame_bb_local is not None
    fr0, fc0, fr1, fc1 = frame_bb_local

    origin = None
    for r0, c0 in _candidate_origins(side, ph, pw, H, W):
        union_bbox = (r0, c0, r0 + ph - 1, c0 + pw - 1)
        frame_bbox = (r0 + fr0, c0 + fc0, r0 + fr1, c0 + fc1)
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
        origin = (r0, c0)
        break
    if origin is None:
        return None

    r0, c0 = origin
    paste(output, scene, r0, c0, transparent=bg)
    paste(input_grid, frame_scene, r0, c0, transparent=bg)
    union_bboxes.append((r0, c0, r0 + ph - 1, c0 + pw - 1))
    frame_bboxes.append((r0 + fr0, c0 + fc0, r0 + fr1, c0 + fc1))

    tool_input = _trim_patch(np.rot90(tool_base, -rotation_steps), bg)
    return {
        "tool_input": tool_input,
        "rotation_steps": int(rotation_steps),
        "plug_color": int(plug_color),
        "handle_color": int(handle_color),
    }


def _undock_tool(info: dict, bg: int, input_grid: np.ndarray, placement_canvas: np.ndarray) -> bool:
    patch = info["tool_input"]
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
        return False
    paste(input_grid, patch, r0, c0, transparent=bg)
    paste(placement_canvas, patch, r0, c0, transparent=bg)
    return True


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    bg = random.choice(list(range(10)))
    num_pairs = random.choice(NUM_PAIRS_CHOICES)

    pair_specs = _pick_pair_specs(num_pairs)
    if len(pair_specs) != num_pairs:
        return None

    colors = [c for c in range(1, 10) if c != bg]
    random.shuffle(colors)
    need = 1 + 3 * num_pairs
    if len(colors) < need:
        return None
    marker_color = colors[0]
    pair_colors = colors[1 : 1 + 3 * num_pairs]

    sides = random.sample(["left", "right", "top", "bottom"], k=num_pairs)
    rotation_steps = _pick_rotation_steps(num_pairs)

    output = np.full((H, W), bg, dtype=int)
    input_grid = np.full((H, W), bg, dtype=int)
    union_bboxes: list[tuple[int, int, int, int]] = []
    frame_bboxes: list[tuple[int, int, int, int]] = []
    placed_tools: list[dict] = []

    for i, spec in enumerate(pair_specs):
        info = _place_docked_pair(
            i,
            spec,
            marker_color,
            pair_colors,
            sides[i],
            rotation_steps[i],
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
