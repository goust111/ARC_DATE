# Combo variant of a25697e4 (index 22)
# combo_spec: D8/3+D14/0
# Name: 异色框架凹槽外翻

from __future__ import annotations

from collections import Counter
import itertools
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, bbox as arc_bbox, get_bg, paste


# Design analysis for combo D8/3 + D14/0
# ---------------------------------------
# a) concretization_and_trace:
#    Start from the source left-border frame. Originally the two frame slabs share
#    one color, the gap row plus staircase notch forms a cavity whose usable side
#    is the grid-interior side, and the docked handle sticks farther toward the
#    interior. The fused variant keeps the same "two slabs with one gap row" family
#    but mirrors the staircase notch onto the border-facing side and colors the two
#    slabs differently. Example: for a left-border frame with width 5 and notch size
#    2, the original bottom slab would miss its two rightmost top cells, so the tool
#    plugs from right to left only because the cavity opens inward. Here the same
#    bottom slab misses its two leftmost top cells instead, so the cavity touches
#    the left border and the tool plugs from the interior toward the edge. The top
#    slab is the full bar color; the bottom notched slab has a second color; the
#    plug region in the loose tool uses exactly that notched-slab color. Output
#    still erases the loose copy and keeps only frame + docked tool.
# b) complexity_calibration:
#    If cavity sizes stayed unique, D14/0 would be visually present but logically
#    irrelevant because the source task could still pair by area alone. To prevent
#    that collapse, generation always makes two pairs with the same cavity geometry
#    and crosses their loose-tool positions so distance heuristics point to the
#    wrong frame. The only stable pairing cue is now "plug color == notched slab
#    color". This keeps the combo readable without creating too many regimes: there
#    are always two frames, each still made from the same simple two-slab template.
# c) implicit_assumptions_audit:
#    - Source `open_dir` inference from background exposure is no longer reliable.
#      The cavity touches the border-facing side, so the solver must derive the
#      anchor side from the fixed sub-components and then invert it to get the
#      handle direction.
#    - Source compatibility only checked color-region size against cavity area.
#      That misses D14/0 entirely. The combo solver additionally requires the
#      candidate plug color to equal the notched sub-component color.
#    - Source cavity extraction picked the largest interior-adjacent background
#      component in the frame bbox. With the notch mirrored outward, the intended
#      cavity is the interior-adjacent background component that also touches the
#      anchor-side edge of the group bbox; the solver enforces that.
#    - Source generator allowed random flips when undocking. That can swap the two
#      tool colors and hide the D14/0 color cue from a human. The combo generator
#      therefore uses rotation-only undocking so the fused color rule stays visible,
#      while `main()` still supports reflected docking states for robustness.
# d) placement_validity:
#    Every frame scene is built docked-first, then pasted on one border with no
#    overlap. The loose tools are re-placed strictly inside the grid, with a one
#    cell halo so they do not merge with frames or with each other. Opposite-side
#    placement keeps them closer to the wrong frame, which is deliberate.
# e) visual_observability:
#    In 2-3 examples a human can see two repeated facts: the frame half with the
#    staircase notch has a distinct color, and the loose tool region of that same
#    color is the part that eventually fills the cavity. The cavity opening also
#    visibly points at the border, while the handle ends up extending back into the
#    grid, so D8/3 is directly readable.
# f) visual_cleanliness:
#    The layout uses only two opposite border frames and two free tools. Shapes stay
#    compact, inter-object spacing stays at least one background cell, and each pair
#    uses three colors only, so the border object, the matching plug, and the handle
#    remain easy to separate by eye.
# g) color_inference:
#    The solver never relies on numeric color ordering. The only color logic is
#    structural: among the two frame slab colors, the color carried by the notched
#    slab is the one whose equal-area region must plug the cavity. Generation uses
#    repeated same-size pairs so that this color relation is necessary and visible.
# new_solver_assumptions_after_edit:
#    - Each anchor group contains exactly two border-touching 4-connected components.
#    - Exactly one of those components is a full rectangle and the other is the
#      notched rectangle, so the notched color is identifiable from area < bbox area.
#    - Each loose tool has exactly two color regions, and exactly one region has both
#      the cavity size and the notched-slab color of its target frame.
#    - The intended cavity touches the frame's anchor-side edge; generate_input()
#      constructs frames to guarantee that this is true.


GRID_H_RANGE = (20, 28)
GRID_W_RANGE = (26, 30)
PAIR_SIDES = [("left", "right")]
FRAME_W_RANGE = (4, 8)
NOTCH_N_RANGE = (1, 4)
TOP_H_RANGE = (1, 4)
BOT_H_RANGE = (2, 4)
PLUG_AREA_BOUNDS = (5, 8)
HANDLE_AREA_RANGE = (4, 9)
TOOL_TOTAL_BOUNDS = (10, 15)
GEN_RETRIES = 200

OPPOSITE = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}


def _bbox_gap(a, b) -> int:
    (ar0, ac0, ar1, ac1) = a
    (br0, bc0, br1, bc1) = b

    def gap1(x0, x1, y0, y1):
        if y1 < x0:
            return int(x0 - y1)
        if x1 < y0:
            return int(y0 - x1)
        return 0

    return gap1(ar0, ar1, br0, br1) + gap1(ac0, ac1, bc0, bc1)


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


def _swap_colors(patch: np.ndarray, a: int, b: int) -> np.ndarray:
    out = patch.copy()
    ma = out == int(a)
    mb = out == int(b)
    out[ma] = int(b)
    out[mb] = int(a)
    return out


def _notch_offsets_left(notch_n: int, bot_h: int) -> list[tuple[int, int]]:
    if notch_n == 1:
        return [(0, 0)]
    if notch_n == 2:
        return [(0, 0), (0, 1)]
    if notch_n == 3:
        if bot_h >= 2:
            return [(0, 0), (0, 1), (1, 0)]
        return [(0, 0), (0, 1), (0, 2)]
    raise ValueError(notch_n)


def _make_handle_sprite(handle_area: int, bg: int, handle_color: int) -> np.ndarray:
    height = 2 if handle_area >= 5 else 1
    width = (handle_area + height - 1) // height
    sprite = np.full((height, width), int(bg), dtype=int)
    need = int(handle_area)
    for c in range(width):
        for r in range(height):
            if need <= 0:
                break
            sprite[r, c] = int(handle_color)
            need -= 1
    assert need == 0
    return sprite


def _build_left_pair(
    *,
    bg: int,
    bar_color: int,
    notch_color: int,
    handle_color: int,
    fw: int,
    top_h: int,
    bot_h: int,
    notch_n: int,
    handle_area: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build one docked pair in the left-border canonical orientation.

    The cavity opens toward the left border. The handle extends to the right,
    which is the grid-interior direction for this canonical orientation.
    """
    fh = top_h + 1 + bot_h
    frame = np.full((fh, fw), int(bg), dtype=int)
    frame[:top_h, :] = int(bar_color)
    frame[top_h + 1 :, :] = int(notch_color)
    for dr, dc in _notch_offsets_left(notch_n, bot_h):
        frame[top_h + 1 + dr, dc] = int(bg)

    cavity = frame == int(bg)
    attach_rows = [r for r in range(fh) if cavity[r, fw - 1]]
    assert attach_rows

    handle_sprite = _make_handle_sprite(handle_area, bg, handle_color)
    hs_h, hs_w = handle_sprite.shape
    sprite_left_rows = [r for r in range(hs_h) if handle_sprite[r, 0] != int(bg)]
    valid_offsets = [
        voff for voff in range(fh - hs_h + 1)
        if any((voff + sr) in attach_rows for sr in sprite_left_rows)
    ]
    assert valid_offsets
    voff = random.choice(valid_offsets)

    total_h = fh
    total_w = fw + hs_w
    tool = np.full((total_h, total_w), int(bg), dtype=int)
    tool[:, :fw][cavity] = int(notch_color)
    handle_mask = handle_sprite != int(bg)
    tool[voff : voff + hs_h, fw : fw + hs_w][handle_mask] = int(handle_color)

    frame_full = np.full((total_h, total_w), int(bg), dtype=int)
    frame_full[:, :fw] = frame
    return frame_full, tool


def _build_param_options() -> list[tuple[int, int, int, int]]:
    out: list[tuple[int, int, int, int]] = []
    for fw in range(*FRAME_W_RANGE):
        for notch_n in range(*NOTCH_N_RANGE):
            plug_area = fw + notch_n
            if not (PLUG_AREA_BOUNDS[0] <= plug_area <= PLUG_AREA_BOUNDS[1]):
                continue
            for top_h in range(*TOP_H_RANGE):
                for bot_h in range(*BOT_H_RANGE):
                    if notch_n == 3 and bot_h < 3:
                        continue
                    out.append((fw, top_h, bot_h, notch_n))
    return out


PARAM_OPTIONS = _build_param_options()


def _pick_shared_spec() -> dict:
    fw, top_h, bot_h, notch_n = random.choice(PARAM_OPTIONS)
    plug_area = fw + notch_n
    handle_choices = [
        h
        for h in range(*HANDLE_AREA_RANGE)
        if TOOL_TOTAL_BOUNDS[0] <= plug_area + h <= TOOL_TOTAL_BOUNDS[1] and h != plug_area
    ]
    assert handle_choices
    return {
        "fw": fw,
        "top_h": top_h,
        "bot_h": bot_h,
        "notch_n": notch_n,
        "plug_area": plug_area,
        "handle_area": random.choice(handle_choices),
    }


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


def _touch_sides(pts: np.ndarray, H: int, W: int) -> set[str]:
    sides: set[str] = set()
    if np.any(pts[:, 0] == 0):
        sides.add("top")
    if np.any(pts[:, 0] == H - 1):
        sides.add("bottom")
    if np.any(pts[:, 1] == 0):
        sides.add("left")
    if np.any(pts[:, 1] == W - 1):
        sides.add("right")
    return sides


def _gen_variants(patch: np.ndarray, mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, bool]]:
    base: list[tuple[np.ndarray, np.ndarray, bool]] = []
    for k in range(4):
        p = np.rot90(patch, k)
        m = np.rot90(mask, k)
        base.append((p, m, False))
        base.append((np.fliplr(p), np.fliplr(m), True))
    uniq: list[tuple[np.ndarray, np.ndarray, bool]] = []
    seen = set()
    for p, m, is_flip in base:
        key = (m.tobytes(), p[m].tobytes())
        if key in seen:
            continue
        seen.add(key)
        uniq.append((p, m, is_flip))
    return uniq


def _rotation_variants_only(patch: np.ndarray) -> list[np.ndarray]:
    seen = set()
    out: list[np.ndarray] = []
    mask = patch != 0
    for k in range(4):
        q = np.rot90(patch, k)
        m = np.rot90(mask, k)
        key = (m.tobytes(), q[m].tobytes())
        if key in seen:
            continue
        seen.add(key)
        out.append(q)
    random.shuffle(out)
    return out


def _frame_component_color(grid: np.ndarray, pts: np.ndarray) -> int:
    vals = {int(grid[int(r), int(c)]) for r, c in pts}
    assert len(vals) == 1
    return int(next(iter(vals)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock two-color tools into border frames whose staircase cavities open toward the grid edge.

    Input structure:
    - The dominant color is background.
    - Border-touching pixels form frames made from exactly two separated sub-components.
    - The two frame sub-components have different colors: one is a solid rectangle, one is the notched rectangle.
    - Interior non-border components are the movable two-color tools.

    Rule:
    - For each frame, identify the notched sub-component and treat its color as the required plug color.
    - Match a tool only when one of its color regions both has the cavity area and has that required color.
    - Rotate or flip the tool so that the matching color fills the border-facing cavity exactly and the other color extends back toward the grid interior.
    - If the chosen docking transform includes a reflection and the two color counts differ, swap the two tool colors after placement.

    Key insight:
    The frame's two colors are not decoration: the color on the notched slab tells you which tool region is the plug, while the shared border side of the two slabs tells you that the handle must finish on the opposite, inward side.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    fg = g != bg
    lab, n = label(fg, structure=STRUCT4)

    fixed_mask = np.zeros((H, W), dtype=bool)
    fixed_components: list[dict] = []
    for cid in range(1, n + 1):
        pts = np.argwhere(lab == cid)
        sides = _touch_sides(pts, H, W)
        if not sides:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        fixed_components.append(
            {
                "pts": pts.copy(),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "area": int(len(pts)),
                "bbox_area": int((r1 - r0 + 1) * (c1 - c0 + 1)),
                "touches": sides,
                "color": _frame_component_color(g, pts),
            }
        )
        fixed_mask[lab == cid] = True

    assert fixed_components

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

    anchor_groups: list[dict] = []
    for idxs in groups.values():
        comps = [fixed_components[i] for i in idxs]
        assert len(comps) == 2
        common_sides = comps[0]["touches"] & comps[1]["touches"]
        assert len(common_sides) == 1
        border_side = next(iter(common_sides))
        notch_candidates = [comp for comp in comps if comp["area"] < comp["bbox_area"]]
        assert len(notch_candidates) == 1
        notch_comp = notch_candidates[0]

        pts = np.concatenate([comp["pts"] for comp in comps], axis=0)
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        bbox = (int(r0), int(c0), int(r1), int(c1))
        mask = np.zeros((H, W), dtype=bool)
        for rr, cc in pts:
            mask[int(rr), int(cc)] = True

        sock = np.zeros((H, W), dtype=bool)
        for r in range(bbox[0], bbox[2] + 1):
            for c in range(bbox[1], bbox[3] + 1):
                if int(g[r, c]) != bg:
                    continue
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < H and 0 <= cc < W and mask[rr, cc]:
                        sock[r, c] = True
                        break

        cavity = np.zeros((H, W), dtype=bool)
        sub = g[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1]
        sub_bg = sub == bg
        sub_lab, sub_n = label(sub_bg, structure=STRUCT4)
        best_id = None
        best_area = -1
        for cid in range(1, sub_n + 1):
            ptsc = np.argwhere(sub_lab == cid)
            area = int(len(ptsc))
            touches_anchor = False
            touches_open_edge = False
            for rr, cc in ptsc:
                if border_side == "left" and int(cc) == 0:
                    touches_open_edge = True
                elif border_side == "right" and int(cc) == sub.shape[1] - 1:
                    touches_open_edge = True
                elif border_side == "top" and int(rr) == 0:
                    touches_open_edge = True
                elif border_side == "bottom" and int(rr) == sub.shape[0] - 1:
                    touches_open_edge = True
                ar = int(bbox[0] + rr)
                ac = int(bbox[1] + cc)
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    r2 = ar + dr
                    c2 = ac + dc
                    if bbox[0] <= r2 <= bbox[2] and bbox[1] <= c2 <= bbox[3] and mask[r2, c2]:
                        touches_anchor = True
                        break
                if touches_anchor and touches_open_edge:
                    break
            if touches_anchor and touches_open_edge and area > best_area:
                best_area = area
                best_id = cid
        assert best_id is not None
        for rr, cc in np.argwhere(sub_lab == best_id):
            cavity[int(bbox[0] + rr), int(bbox[1] + cc)] = True

        anchor_groups.append(
            {
                "mask": mask,
                "bbox": bbox,
                "socket_mask": sock,
                "cavity_mask": cavity,
                "cavity_size": int(cavity.sum()),
                "notch_color": int(notch_comp["color"]),
                "border_side": border_side,
                "inward_dir": OPPOSITE[border_side],
                "size": int(mask.sum()),
            }
        )

    anchor_groups.sort(key=lambda d: -d["size"])

    mov_mask = fg & (~fixed_mask)
    mov_lab, mov_n = label(mov_mask, structure=STRUCT4)
    mov_comps: list[dict] = []
    comp_color_counts: list[dict[int, int]] = []
    for cid in range(1, mov_n + 1):
        pts = np.argwhere(mov_lab == cid)
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        mask = mov_lab[r0 : r1 + 1, c0 : c1 + 1] == cid
        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        patch[~mask] = 0
        vals = [int(x) for x in patch[mask].flatten() if int(x) != 0]
        counts = dict(Counter(vals))
        mov_comps.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "mask": mask,
                "patch": patch,
                "size": int(len(pts)),
            }
        )
        comp_color_counts.append(counts)
    mov_comps.sort(key=lambda d: -d["size"])
    comp_color_counts = [
        dict(Counter([int(x) for x in comp["patch"][comp["mask"]].flatten() if int(x) != 0]))
        for comp in mov_comps
    ]
    assert len(mov_comps) == len(anchor_groups)

    def compatible(ci: int, gi: int) -> bool:
        grp = anchor_groups[gi]
        return any(
            int(col) == int(grp["notch_color"]) and int(cnt) == int(grp["cavity_size"])
            for col, cnt in comp_color_counts[ci].items()
        )

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
        score = (-int(dist_sum),)
        if best_score is None or score > best_score:
            best_score = score
            best_perm = perm
    assert best_perm is not None

    out = np.full_like(g, int(bg))
    out[fixed_mask] = g[fixed_mask]
    placed_global = fixed_mask.copy()

    for comp, gi in zip(mov_comps, best_perm):
        grp = anchor_groups[int(gi)]
        cav = grp["cavity_mask"]
        sock = grp["socket_mask"]
        br0, bc0, br1, bc1 = grp["bbox"]
        desired_color = int(grp["notch_color"])
        cav_sz = int(grp["cavity_size"])
        comp_counts = Counter([int(x) for x in comp["patch"][comp["mask"]].flatten() if int(x) != 0])

        best_state = None
        best_score = None
        for p, m, is_flip in _gen_variants(comp["patch"], comp["mask"]):
            coords = np.argwhere(m)
            h, w = m.shape
            plug_colors = [
                int(col)
                for col in sorted({int(x) for x in p[m].flatten() if int(x) != 0})
                if int(col) == desired_color and int(((p == int(col)) & m).sum()) == cav_sz
            ]
            if not plug_colors:
                continue
            for r0 in range(H - h + 1):
                for c0 in range(W - w + 1):
                    blocked = False
                    adj = 0
                    socket_hits = 0
                    outside_inward = 0
                    for rr, cc in coords:
                        r = int(r0 + rr)
                        c = int(c0 + cc)
                        if placed_global[r, c]:
                            blocked = True
                            break
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            r2 = r + dr
                            c2 = c + dc
                            if 0 <= r2 < H and 0 <= c2 < W and grp["mask"][r2, c2]:
                                adj += 1
                        if sock[r, c]:
                            socket_hits += 1
                        inward = grp["inward_dir"]
                        if inward == "right" and c > bc1:
                            outside_inward += 1
                        elif inward == "left" and c < bc0:
                            outside_inward += 1
                        elif inward == "bottom" and r > br1:
                            outside_inward += 1
                        elif inward == "top" and r < br0:
                            outside_inward += 1
                    if blocked or adj == 0:
                        continue

                    plug_best = None
                    for plug_col in plug_colors:
                        plug_hits = 0
                        plug_outside = 0
                        nonplug_in_cav = 0
                        for rr, cc in coords:
                            r = int(r0 + rr)
                            c = int(c0 + cc)
                            col = int(p[rr, cc])
                            in_cav = bool(cav[r, c])
                            if col == plug_col:
                                if in_cav:
                                    plug_hits += 1
                                else:
                                    plug_outside += 1
                            elif in_cav:
                                nonplug_in_cav += 1
                        plug_score = (
                            int(plug_hits == cav_sz and plug_outside == 0 and nonplug_in_cav == 0),
                            int(plug_hits),
                            -int(nonplug_in_cav),
                            -int(plug_outside),
                        )
                        if plug_best is None or plug_score > plug_best:
                            plug_best = plug_score
                    dist = _bbox_gap(grp["bbox"], (int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)))
                    score = tuple(plug_best) + (int(outside_inward), int(adj), int(socket_hits), -int(dist), -int(r0), -int(c0))
                    if best_score is None or score > best_score:
                        best_score = score
                        best_state = (p.copy(), m.copy(), int(r0), int(c0), bool(is_flip))

        assert best_state is not None
        p, m, r0, c0, is_flip = best_state
        if is_flip and len(comp_counts) == 2 and len(set(comp_counts.values())) == 2:
            a, b = sorted(comp_counts.keys())
            p = _swap_colors(p, a, b)

        for rr, cc in np.argwhere(m):
            r = int(r0 + rr)
            c = int(c0 + cc)
            out[r, c] = int(p[rr, cc])
            placed_global[r, c] = True

    return out


def _place_pair_on_border(
    *,
    side: str,
    spec: dict,
    bg: int,
    bar_color: int,
    notch_color: int,
    handle_color: int,
    H: int,
    W: int,
    output: np.ndarray,
    input_grid: np.ndarray,
    union_bboxes: list[tuple[int, int, int, int]],
    frame_bboxes: list[tuple[int, int, int, int]],
) -> dict | None:
    frame_left, tool_left = _build_left_pair(
        bg=bg,
        bar_color=bar_color,
        notch_color=notch_color,
        handle_color=handle_color,
        fw=spec["fw"],
        top_h=spec["top_h"],
        bot_h=spec["bot_h"],
        notch_n=spec["notch_n"],
        handle_area=spec["handle_area"],
    )
    scene_left = np.where(tool_left != int(bg), tool_left, frame_left)

    frame_scene = _orient_left_to_side(frame_left, side)
    tool_scene = _orient_left_to_side(tool_left, side)
    scene = _orient_left_to_side(scene_left, side)
    ph, pw = scene.shape
    fr0, fc0, fr1, fc1 = arc_bbox(frame_scene != int(bg))

    origin = None
    for r0, c0 in _candidate_origins(side, ph, pw, H, W):
        union_bbox = (int(r0), int(c0), int(r0 + ph - 1), int(c0 + pw - 1))
        frame_bbox = (int(r0 + fr0), int(c0 + fc0), int(r0 + fr1), int(c0 + fc1))
        if side in ("left", "right") and (frame_bbox[0] == 0 or frame_bbox[2] == H - 1):
            continue
        if side in ("top", "bottom") and (frame_bbox[1] == 0 or frame_bbox[3] == W - 1):
            continue
        if any(_bbox_gap(union_bbox, prev) <= 1 for prev in union_bboxes):
            continue
        if any(_bbox_gap(frame_bbox, prev) <= 2 for prev in frame_bboxes):
            continue
        if np.any((output[r0 : r0 + ph, c0 : c0 + pw] != int(bg)) & (scene != int(bg))):
            continue
        origin = (int(r0), int(c0))
        break
    if origin is None:
        return None

    r0, c0 = origin
    paste(output, scene, r0, c0, transparent=bg)
    paste(input_grid, frame_scene, r0, c0, transparent=bg)
    union_bboxes.append((r0, c0, r0 + ph - 1, c0 + pw - 1))
    frame_bbox = (r0 + fr0, c0 + fc0, r0 + fr1, c0 + fc1)
    frame_bboxes.append(frame_bbox)
    return {
        "side": side,
        "tool": tool_scene,
        "plug_color": int(notch_color),
        "handle_color": int(handle_color),
        "plug_area": int(spec["plug_area"]),
        "handle_area": int(spec["handle_area"]),
        "frame_bbox": frame_bbox,
    }


def _can_place_patch(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> bool:
    H, W = canvas.shape
    ph, pw = patch.shape
    if r0 < 1 or c0 < 1 or r0 + ph > H - 1 or c0 + pw > W - 1:
        return False
    mask = patch != int(bg)
    if np.any((canvas[r0 : r0 + ph, c0 : c0 + pw] != int(bg)) & mask):
        return False
    halo = canvas[max(0, r0 - 1) : min(H, r0 + ph + 1), max(0, c0 - 1) : min(W, c0 + pw + 1)]
    if np.any(halo != int(bg)):
        return False
    return True


def _place_tool_near_wrong_frame(
    *,
    info: dict,
    opp_info: dict,
    bg: int,
    input_grid: np.ndarray,
    occupancy: np.ndarray,
) -> bool:
    H, W = input_grid.shape
    for patch in _rotation_variants_only(info["tool"]):
        ph, pw = patch.shape
        best = None
        best_score = None
        for r0 in range(1, H - ph):
            for c0 in range(1, W - pw):
                if not _can_place_patch(occupancy, patch, r0, c0, bg):
                    continue
                pb = (int(r0), int(c0), int(r0 + ph - 1), int(c0 + pw - 1))
                d_own = _bbox_gap(pb, info["frame_bbox"])
                d_opp = _bbox_gap(pb, opp_info["frame_bbox"])
                score = (int(d_opp < d_own), -int(d_opp), int(d_own), -int(r0), -int(c0))
                if best_score is None or score > best_score:
                    best_score = score
                    best = (int(r0), int(c0))
        if best is not None and best_score is not None and best_score[0] == 1:
            r0, c0 = best
            paste(input_grid, patch, r0, c0, transparent=bg)
            paste(occupancy, patch, r0, c0, transparent=bg)
            return True
    return False


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    bg = random.randint(0, 9)
    sides = random.choice(PAIR_SIDES)
    spec = _pick_shared_spec()

    colors = [c for c in range(1, 10) if c != bg]
    random.shuffle(colors)
    bar_colors = [colors[0], colors[1]]
    notch_colors = [colors[2], colors[3]]
    handle_colors = [colors[4], colors[5]]

    output = np.full((H, W), int(bg), dtype=int)
    input_grid = np.full((H, W), int(bg), dtype=int)
    union_bboxes: list[tuple[int, int, int, int]] = []
    frame_bboxes: list[tuple[int, int, int, int]] = []
    infos: list[dict] = []

    for i, side in enumerate(sides):
        info = _place_pair_on_border(
            side=side,
            spec=spec,
            bg=bg,
            bar_color=bar_colors[i],
            notch_color=notch_colors[i],
            handle_color=handle_colors[i],
            H=H,
            W=W,
            output=output,
            input_grid=input_grid,
            union_bboxes=union_bboxes,
            frame_bboxes=frame_bboxes,
        )
        if info is None:
            return None
        infos.append(info)

    occupancy = input_grid.copy()
    for i, info in enumerate(infos):
        opp = infos[1 - i]
        if not _place_tool_near_wrong_frame(
            info=info,
            opp_info=opp,
            bg=bg,
            input_grid=input_grid,
            occupancy=occupancy,
        ):
            return None

    solved = main(input_grid)
    if np.array_equal(solved, output):
        return input_grid, output
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result[0]
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
