# Combo variant of a25697e4 (index 13)
# combo_spec: D1/0+D3/0+D14/0
# Name: 三色工具异色框架颜色闭环

from __future__ import annotations

from collections import Counter
import itertools
import random

import numpy as np
from scipy import ndimage
from scipy.ndimage import label

from arc_utils import STRUCT4, STRUCT8, bbox as _bbox, get_bg, paste


# Design analysis for the fused variant:
# a) Concrete behavior trace:
#    - Original task: each movable tool has 2 colors, frames are monochrome, and pairing is inferred from
#      cavity area. After docking, a reflected placement swaps the two tool colors.
#    - Fused task: each frame is now two-tone. One subcomponent is a solid rectangle (bar color B), while the
#      other is the notched subcomponent (color A). The tool is 3-color: plug, handle, hinge.
#    - Pixel-level trace with two pairs:
#      input frame 1 = {bar color 6, notch color 2}, input tool 1 = {plug 2, handle 7, hinge 5}
#      input frame 2 = {bar color 8, notch color 3}, input tool 2 = {plug 3, handle 4, hinge 5}
#      Both cavities and both tool masks share the same geometry, so area/shape no longer disambiguate pairing.
#      The only stable pairing clue is that tool 1 contains color 2 and tool 2 contains color 3.
#      After docking, the color-2/3 region fills the notch-shaped cavity; the hinge remains visible just outside
#      the frame. If one tool must be mirrored to redock and plug/handle sizes differ, only those two colors swap.
#    - Conflict resolution across source variants: standalone D3/0 erased the hinge in the output, but the fused
#      spec explicitly says "铰链色保持不变", so this combo keeps the hinge visible in both input and output.
# b) Complexity calibration:
#    - To make D1/0 genuinely active, every sample uses exactly 2 frame-tool pairs that share one global cavity
#      spec and one global tool-shape template. This kills the original area-based one-to-one cue.
#    - To make the reflection rule observable, one pair is forced to use a reflected input orientation and the
#      other is forced to use a non-reflected one. Plug and handle areas are always unequal.
#    - Colors are still capped tightly: bg + 2 notch colors + 2 bar colors + 2 handle colors + 1 shared hinge
#      color. This is enough to expose the new rule without turning the grid into noise.
# c) Hidden-assumption audit against the original solver:
#    - "A movable tool has exactly 2 colors" no longer holds; the solver now expects 3 colors and identifies
#      hinge vs handle by region size among the non-plug colors.
#    - "Frame color is uniform" no longer holds; the solver must inspect each fixed subcomponent separately and
#      use the non-rectangular component's color as the plug-matching color.
#    - "Tool-frame pairing can be inferred from cavity size" no longer holds; all cavities are generated with the
#      same area. Pairing is now driven by the shared color between a tool region and the notched frame piece.
#    - "Reflection swap can be undone from input colors" is intentionally not used here. Input tools keep their
#      canonical colors so the color-link remains visible; the swap is applied only when docking into the output.
# d) Placement legality:
#    - Frames are placed from an output-first construction on distinct borders with bbox-gap checks, so groups do
#      not merge and all frame pixels stay in bounds.
#    - Free tools are placed only at in-bounds interior locations with padding from every frame/tool already on
#      the canvas, plus a minimum bbox gap from their target frame, so they stay visually detached.
# e) Visual observability:
#    - Humans can see two same-shaped tools and two same-sized cavities in one grid, so the original pairing cue
#      is visibly broken. The repeated evidence is the notch-color loop: frame notch color -> tool plug color ->
#      docked cavity color (or swapped handle color in the reflected case).
# f) Visual cleanliness:
#    - The hinge is kept to 1-2 cells and always sits in the open direction outside the frame bbox, so it reads
#      as a connector instead of cluttering the cavity.
#    - Only one border frame lives on each used side, and free tools are padded apart from each other.
# g) New solver assumptions and generator guarantees:
#    - `main()` assumes each anchor group contains exactly one rectangular fixed subcomponent and one non-rectangular
#      fixed subcomponent. `generate_input()` guarantees that by construction.
#    - `main()` assumes each movable component contains exactly one frame-matching color and exactly 3 colors total.
#      `generate_input()` uses globally unique notch colors and 3-color connected tools.
#    - `main()` assumes hinge area is the smaller non-plug region and remains unchanged by the swap rule.
#      `generate_input()` enforces hinge area 1-2 and handle area 5, distinct from plug area 6-8.


GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (25, 30)
NUM_PAIRS = 2

FRAME_SPECS = [
    {"fw": 4, "top_h": 2, "bot_h": 2, "notch_n": 2},
    {"fw": 5, "top_h": 1, "bot_h": 2, "notch_n": 2},
    {"fw": 5, "top_h": 2, "bot_h": 2, "notch_n": 3},
]
HANDLE_TEMPLATES = [
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1], [0, 0, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=bool),
]
HINGE_LEN_CHOICES = [1, 2]

UNION_BBOX_GAP_MIN = 2
FRAME_BBOX_GAP_MIN = 3
FREE_TOOL_BORDER = 1
FREE_TOOL_PADDING = 3
FREE_TOOL_TARGET_GAP = 4
GEN_ATTEMPT_RETRIES = 400

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _bbox_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    (ar0, ac0, ar1, ac1) = a
    (br0, bc0, br1, bc1) = b

    def _gap1(x0: int, x1: int, y0: int, y1: int) -> int:
        if y1 < x0:
            return int(x0 - y1)
        if x1 < y0:
            return int(y0 - x1)
        return 0

    return _gap1(ar0, ar1, br0, br1) + _gap1(ac0, ac1, bc0, bc1)


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


def _variant_choices(patch: np.ndarray, bg: int) -> list[tuple[np.ndarray, bool]]:
    mask = patch != int(bg)
    out: list[tuple[np.ndarray, bool]] = []
    seen = set()
    for p, m, is_flip in _gen_variants(patch, mask):
        key = (m.tobytes(), p[m].tobytes())
        if key in seen:
            continue
        seen.add(key)
        out.append((p, bool(is_flip)))
    random.shuffle(out)
    return out


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


def _build_left_pair(
    *,
    bg: int,
    bar_color: int,
    plug_color: int,
    handle_color: int,
    hinge_color: int,
    spec: dict,
    handle_mask: np.ndarray,
    hinge_len: int,
) -> dict | None:
    fw = int(spec["fw"])
    top_h = int(spec["top_h"])
    bot_h = int(spec["bot_h"])
    notch_n = int(spec["notch_n"])

    fh = top_h + 1 + bot_h
    frame = np.full((fh, fw), bg, dtype=int)
    frame[:top_h, :] = int(bar_color)
    frame[top_h + 1 :, :] = int(plug_color)
    for dr, dc in _notch_offsets(notch_n, fw, bot_h):
        frame[top_h + 1 + dr, dc] = int(bg)

    cavity = frame == int(bg)
    attach_rows = [int(r) for r in range(fh) if cavity[r, fw - 1]]
    if not attach_rows:
        return None

    hh, hw = handle_mask.shape
    left_rows = [int(r) for r in range(hh) if handle_mask[r, 0]]
    alignments: list[tuple[int, int]] = []
    for attach_row in attach_rows:
        for left_row in left_rows:
            voff = int(attach_row - left_row)
            if voff < 0:
                continue
            alignments.append((voff, attach_row))
    if not alignments:
        return None

    alignments.sort()
    voff, attach_row = alignments[len(alignments) // 2]
    total_h = max(fh, voff + hh)
    total_w = fw + int(hinge_len) + hw

    frame_full = np.full((total_h, total_w), bg, dtype=int)
    frame_full[:fh, :fw] = frame

    tool = np.full((total_h, total_w), bg, dtype=int)
    tool[:fh, :fw][cavity] = int(plug_color)
    tool[attach_row, fw : fw + hinge_len] = int(hinge_color)
    tool[voff : voff + hh, fw + hinge_len : fw + hinge_len + hw][handle_mask] = int(handle_color)

    return {
        "frame": frame_full,
        "tool": tool,
        "plug_area": int(cavity.sum()),
        "handle_area": int(handle_mask.sum()),
        "hinge_area": int(hinge_len),
    }


def _candidate_free_locations(canvas: np.ndarray, patch: np.ndarray, bg: int) -> list[tuple[int, int]]:
    H, W = canvas.shape
    h, w = patch.shape
    occ = canvas != int(bg)
    if FREE_TOOL_PADDING > 0:
        occ = ndimage.binary_dilation(
            occ,
            iterations=int(FREE_TOOL_PADDING),
            structure=STRUCT4,
        )
    mask = patch != int(bg)
    out: list[tuple[int, int]] = []
    for r0 in range(FREE_TOOL_BORDER, H - h - FREE_TOOL_BORDER + 1):
        for c0 in range(FREE_TOOL_BORDER, W - w - FREE_TOOL_BORDER + 1):
            if bool((occ[r0 : r0 + h, c0 : c0 + w] & mask).any()):
                continue
            out.append((int(r0), int(c0)))
    random.shuffle(out)
    return out


def _choose_global_geometry() -> tuple[dict, np.ndarray, int]:
    spec = dict(random.choice(FRAME_SPECS))
    handle_mask = random.choice(HANDLE_TEMPLATES).copy()
    hinge_len = int(random.choice(HINGE_LEN_CHOICES))
    plug_area = int(spec["fw"] + spec["notch_n"])
    handle_area = int(handle_mask.sum())
    total = plug_area + handle_area + hinge_len
    if handle_area == plug_area or total > 15:
        return _choose_global_geometry()
    return spec, handle_mask, hinge_len


def _build_anchor_groups(g: np.ndarray, bg: int) -> tuple[np.ndarray, list[dict]]:
    H, W = g.shape
    fg = g != int(bg)
    lab, n = label(fg, structure=STRUCT4)

    fixed_mask = np.zeros((H, W), dtype=bool)
    fixed_components = []
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        if pts.size == 0:
            continue
        pts_list = [(int(r), int(c)) for r, c in pts]
        if not any(r == 0 or r == H - 1 or c == 0 or c == W - 1 for r, c in pts_list):
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        comp_mask = np.zeros((H, W), dtype=bool)
        comp_mask[lab == int(cid)] = True
        col = int(g[lab == int(cid)][0])
        fixed_components.append(
            {
                "pts": pts_list,
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "mask": comp_mask,
                "color": col,
                "is_rect": int(len(pts)) == int((r1 - r0 + 1) * (c1 - c0 + 1)),
            }
        )
        fixed_mask[lab == int(cid)] = True

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
        pts = [p for i in idxs for p in fixed_components[i]["pts"]]
        rs = [r for r, _ in pts]
        cs = [c for _, c in pts]
        bbox = (min(rs), min(cs), max(rs), max(cs))
        mask = np.zeros((H, W), dtype=bool)
        for r, c in pts:
            mask[r, c] = True

        socket = np.zeros((H, W), dtype=bool)
        for r in range(bbox[0], bbox[2] + 1):
            for c in range(bbox[1], bbox[3] + 1):
                if int(g[r, c]) != int(bg):
                    continue
                for dr, dc in DIRS4:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < H and 0 <= cc < W and mask[rr, cc]:
                        socket[r, c] = True
                        break

        cavity = np.zeros((H, W), dtype=bool)
        sub_bg = g[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1] == int(bg)
        sub_lab, sub_n = label(sub_bg, structure=STRUCT4)
        best_id = None
        best_score = None
        for cid in range(1, int(sub_n) + 1):
            ptsc = np.argwhere(sub_lab == int(cid))
            if ptsc.size == 0:
                continue
            area = int(len(ptsc))
            adj = False
            for rr, cc in ptsc:
                r = int(bbox[0] + rr)
                c = int(bbox[1] + cc)
                for dr, dc in DIRS4:
                    r2, c2 = r + dr, c + dc
                    if bbox[0] <= r2 <= bbox[2] and bbox[1] <= c2 <= bbox[3] and mask[r2, c2]:
                        adj = True
                        break
                if adj:
                    break
            if not adj:
                continue
            score = (area, -int(cid))
            if best_score is None or score > best_score:
                best_score = score
                best_id = int(cid)
        assert best_id is not None
        for rr, cc in np.argwhere(sub_lab == int(best_id)):
            cavity[int(bbox[0] + rr), int(bbox[1] + cc)] = True
        cav_pts = np.argwhere(cavity)
        cav_bbox = tuple(int(x) for x in (*cav_pts.min(axis=0), *cav_pts.max(axis=0)))
        edge_counts = {
            (-1, 0): int(cavity[bbox[0], bbox[1] : bbox[3] + 1].sum()),
            (1, 0): int(cavity[bbox[2], bbox[1] : bbox[3] + 1].sum()),
            (0, -1): int(cavity[bbox[0] : bbox[2] + 1, bbox[1]].sum()),
            (0, 1): int(cavity[bbox[0] : bbox[2] + 1, bbox[3]].sum()),
        }
        open_dir = max(edge_counts, key=edge_counts.get)

        sub_infos = [fixed_components[i] for i in idxs]
        plug_source = [sub for sub in sub_infos if not bool(sub["is_rect"])]
        if plug_source:
            plug_match_color = int(plug_source[0]["color"])
        else:
            plug_match_color = int(sub_infos[0]["color"])

        anchor_groups.append(
            {
                "mask": mask,
                "bbox": bbox,
                "socket_mask": socket,
                "cavity_mask": cavity,
                "cavity_bbox": cav_bbox,
                "open_dir": open_dir,
                "plug_match_color": plug_match_color,
                "cavity_size": int(cavity.sum()),
                "size": int(mask.sum()),
            }
        )
    anchor_groups.sort(key=lambda d: (-int(d["size"]), d["bbox"]))
    return fixed_mask, anchor_groups


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock three-color tools into two-tone border frames using the notch color as the pairing cue.

    Input structure:
    - bg is the most frequent color.
    - Border-touching foreground clusters form frames made of two separated 4-connected subcomponents with different colors; the non-rectangular subcomponent is the notched piece.
    - Interior foreground components are 3-color tools containing a plug region, a handle region, and a 1-2 cell hinge region.

    Rule:
    - Pair each tool with the frame whose notched-subcomponent color also appears on that tool; that shared color marks the plug.
    - Rotate and optionally flip the tool so the plug fills the frame cavity exactly, while hinge and handle stay outside in the frame's open direction.
    - If the chosen docking transform uses a reflection and the plug and handle areas are unequal, swap only those two colors; the hinge color is preserved.

    Key insight:
    - Tool-frame matching is no longer determined by cavity size. The decisive clue is the color of the non-rectangular frame piece, and the visible hinge lets the solver treat the third tool color separately from the reflection-triggered swap.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    fixed_mask, anchor_groups = _build_anchor_groups(g, bg)

    fg = g != int(bg)
    mov_mask = fg & (~fixed_mask)
    mov_lab, mov_n = label(mov_mask, structure=STRUCT4)
    mov_comps: list[dict] = []
    for cid in range(1, int(mov_n) + 1):
        pts = np.argwhere(mov_lab == int(cid))
        if pts.size == 0:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        mask = mov_lab[r0 : r1 + 1, c0 : c1 + 1] == int(cid)
        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        patch[~mask] = 0
        vals = [int(x) for x in patch[mask].flatten()]
        counts = dict(Counter(vals))
        mov_comps.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "mask": mask,
                "patch": patch,
                "counts": counts,
                "size": int(len(pts)),
            }
        )
    mov_comps.sort(key=lambda d: (-int(d["size"]), d["bbox"]))

    def compatible(ci: int, gi: int) -> bool:
        return int(anchor_groups[gi]["plug_match_color"]) in mov_comps[ci]["counts"]

    if len(mov_comps) == len(anchor_groups) and len(anchor_groups) > 1:
        best_perm = None
        best_score = None
        for perm in itertools.permutations(range(len(anchor_groups))):
            if not all(compatible(ci, gi) for ci, gi in enumerate(perm)):
                continue
            dist_sum = sum(_bbox_gap(mov_comps[ci]["bbox"], anchor_groups[gi]["bbox"]) for ci, gi in enumerate(perm))
            score = (-int(dist_sum),)
            if best_score is None or score > best_score:
                best_score = score
                best_perm = perm
        assert best_perm is not None
        group_for_comp = [int(x) for x in best_perm]
    else:
        group_for_comp = []
        for ci, comp in enumerate(mov_comps):
            candidates = [gi for gi in range(len(anchor_groups)) if compatible(ci, gi)]
            assert candidates
            gi = min(candidates, key=lambda j: _bbox_gap(comp["bbox"], anchor_groups[j]["bbox"]))
            group_for_comp.append(int(gi))

    out = np.full_like(g, int(bg))
    out[fixed_mask] = g[fixed_mask]
    placed_global = fixed_mask.copy()
    placed_groups = [grp["mask"].copy() for grp in anchor_groups]

    for comp, gi in zip(mov_comps, group_for_comp):
        grp = anchor_groups[gi]
        placed_grp = placed_groups[gi]
        cav = grp["cavity_mask"]
        sock = grp["socket_mask"]
        plug_color = int(grp["plug_match_color"])
        counts = {int(k): int(v) for k, v in comp["counts"].items()}
        assert plug_color in counts
        others = sorted((col for col in counts if col != plug_color), key=lambda col: (counts[col], col))
        assert len(others) == 2
        hinge_color = int(others[0])
        handle_color = int(others[1])
        plug_area = int(counts[plug_color])
        handle_area = int(counts[handle_color])
        cav_sz = int(grp["cavity_size"])

        best_state = None
        best_score = None
        for p, m, is_flip in _gen_variants(comp["patch"], comp["mask"]):
            h, w = m.shape
            coords = np.argwhere(m)
            for r0 in range(H - h + 1):
                for c0 in range(W - w + 1):
                    if any(placed_global[int(r0 + rr), int(c0 + cc)] for rr, cc in coords):
                        continue
                    adj = 0
                    plug_hits = 0
                    plug_outside = 0
                    nonplug_in_cavity = 0
                    nonplug_outside = 0
                    nonplug_wrong_side = 0
                    socket_hits = 0
                    for rr, cc in coords:
                        r = int(r0 + rr)
                        c = int(c0 + cc)
                        col = int(p[rr, cc])
                        in_cav = bool(cav[r, c])
                        if sock[r, c]:
                            socket_hits += 1
                        for dr, dc in DIRS4:
                            r2, c2 = r + dr, c + dc
                            if 0 <= r2 < H and 0 <= c2 < W and placed_grp[r2, c2]:
                                adj += 1
                        if col == plug_color:
                            if in_cav:
                                plug_hits += 1
                            else:
                                plug_outside += 1
                        else:
                            if in_cav:
                                nonplug_in_cavity += 1
                            br0, bc0, br1, bc1 = grp["bbox"]
                            odr, odc = grp["open_dir"]
                            on_open_side = False
                            if odc == -1 and c < bc0:
                                on_open_side = True
                            elif odc == 1 and c > bc1:
                                on_open_side = True
                            elif odr == -1 and r < br0:
                                on_open_side = True
                            elif odr == 1 and r > br1:
                                on_open_side = True
                            if on_open_side:
                                nonplug_outside += 1
                            else:
                                nonplug_wrong_side += 1
                    if adj == 0:
                        continue
                    plug_full = int(plug_hits == cav_sz and plug_outside == 0 and nonplug_in_cavity == 0)
                    dist = _bbox_gap(grp["bbox"], (int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)))
                    score = (
                        plug_full,
                        int(plug_hits),
                        -int(nonplug_in_cavity),
                        -int(plug_outside),
                        int(nonplug_outside),
                        -int(nonplug_wrong_side),
                        int(socket_hits),
                        int(adj),
                        -int(dist),
                        -int(r0),
                        -int(c0),
                    )
                    if best_score is None or score > best_score:
                        best_score = score
                        best_state = (p.copy(), m.copy(), int(r0), int(c0), bool(is_flip), hinge_color, handle_color)

        assert best_state is not None
        p, m, r0, c0, is_flip, hinge_color, handle_color = best_state
        if is_flip and plug_area != handle_area:
            p = _swap_colors(p, plug_color, handle_color)
        for rr, cc in np.argwhere(m):
            r = int(r0 + rr)
            c = int(c0 + cc)
            out[r, c] = int(p[rr, cc])
            placed_global[r, c] = True
            placed_grp[r, c] = True

    return out


def _place_pair_on_border(
    *,
    output: np.ndarray,
    input_grid: np.ndarray,
    union_bboxes: list[tuple[int, int, int, int]],
    frame_bboxes: list[tuple[int, int, int, int]],
    side: str,
    bg: int,
    bar_color: int,
    plug_color: int,
    handle_color: int,
    hinge_color: int,
    spec: dict,
    handle_mask: np.ndarray,
    hinge_len: int,
    force_flip: bool,
) -> dict | None:
    built = _build_left_pair(
        bg=bg,
        bar_color=bar_color,
        plug_color=plug_color,
        handle_color=handle_color,
        hinge_color=hinge_color,
        spec=spec,
        handle_mask=handle_mask,
        hinge_len=hinge_len,
    )
    if built is None:
        return None

    frame_scene = _orient_left_to_side(built["frame"], side)
    tool_scene = _orient_left_to_side(built["tool"], side)
    variants = _variant_choices(tool_scene, bg)
    candidates = [item for item in variants if bool(item[1]) == bool(force_flip)]
    if not candidates:
        return None
    input_patch, is_flip = random.choice(candidates)

    output_tool = tool_scene.copy()
    if is_flip and int(built["plug_area"]) != int(built["handle_area"]):
        output_tool = _swap_colors(output_tool, plug_color, handle_color)
    output_scene = np.where(output_tool != int(bg), output_tool, frame_scene)

    ph, pw = output_scene.shape
    frame_bbox_local = _bbox(frame_scene != int(bg))
    assert frame_bbox_local is not None
    fr0, fc0, fr1, fc1 = frame_bbox_local

    origin = None
    for r0, c0 in _candidate_origins(side, ph, pw, output.shape[0], output.shape[1]):
        union_bbox = (int(r0), int(c0), int(r0 + ph - 1), int(c0 + pw - 1))
        frame_bbox = (int(r0 + fr0), int(c0 + fc0), int(r0 + fr1), int(c0 + fc1))
        if side in ("left", "right") and (frame_bbox[0] == 0 or frame_bbox[2] == output.shape[0] - 1):
            continue
        if side in ("top", "bottom") and (frame_bbox[1] == 0 or frame_bbox[3] == output.shape[1] - 1):
            continue
        if any(_bbox_gap(union_bbox, prev) < UNION_BBOX_GAP_MIN for prev in union_bboxes):
            continue
        if any(_bbox_gap(frame_bbox, prev) < FRAME_BBOX_GAP_MIN for prev in frame_bboxes):
            continue
        if np.any((output[r0 : r0 + ph, c0 : c0 + pw] != int(bg)) & (output_scene != int(bg))):
            continue
        origin = (int(r0), int(c0))
        break
    if origin is None:
        return None

    r0, c0 = origin
    paste(output, output_scene, r0, c0, transparent=bg)
    paste(input_grid, frame_scene, r0, c0, transparent=bg)

    union_bboxes.append((int(r0), int(c0), int(r0 + ph - 1), int(c0 + pw - 1)))
    frame_bbox = (int(r0 + fr0), int(c0 + fc0), int(r0 + fr1), int(c0 + fc1))
    frame_bboxes.append(frame_bbox)

    return {
        "input_patch": input_patch,
        "target_frame_bbox": frame_bbox,
    }


def _place_free_tool(
    input_grid: np.ndarray,
    canvas: np.ndarray,
    patch: np.ndarray,
    bg: int,
    target_frame_bbox: tuple[int, int, int, int],
    all_frame_bboxes: list[tuple[int, int, int, int]],
) -> bool:
    h, w = patch.shape
    for r0, c0 in _candidate_free_locations(canvas, patch, bg):
        bbox = (int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1))
        if _bbox_gap(bbox, target_frame_bbox) < FREE_TOOL_TARGET_GAP:
            continue
        if any(_bbox_gap(bbox, fb) < 2 for fb in all_frame_bboxes):
            continue
        paste(input_grid, patch, r0, c0, transparent=bg)
        paste(canvas, patch, r0, c0, transparent=bg)
        return True
    return False


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    bg = random.choice(list(range(10)))
    spec, handle_mask, hinge_len = _choose_global_geometry()

    colors = [c for c in range(10) if c != bg]
    random.shuffle(colors)
    plug_colors = [int(colors[0]), int(colors[1])]
    bar_colors = [int(colors[2]), int(colors[3])]
    handle_colors = [int(colors[4]), int(colors[5])]
    hinge_color = int(colors[6])

    sides = random.sample(["left", "right", "top", "bottom"], k=NUM_PAIRS)
    flip_plan = [True, False]
    random.shuffle(flip_plan)

    output = np.full((H, W), bg, dtype=int)
    input_grid = np.full((H, W), bg, dtype=int)
    union_bboxes: list[tuple[int, int, int, int]] = []
    frame_bboxes: list[tuple[int, int, int, int]] = []
    tool_infos: list[dict] = []

    for i in range(NUM_PAIRS):
        info = _place_pair_on_border(
            output=output,
            input_grid=input_grid,
            union_bboxes=union_bboxes,
            frame_bboxes=frame_bboxes,
            side=sides[i],
            bg=bg,
            bar_color=bar_colors[i],
            plug_color=plug_colors[i],
            handle_color=handle_colors[i],
            hinge_color=hinge_color,
            spec=spec,
            handle_mask=handle_mask,
            hinge_len=hinge_len,
            force_flip=flip_plan[i],
        )
        if info is None:
            return None
        tool_infos.append(info)

    canvas = input_grid.copy()
    for info in tool_infos:
        if not _place_free_tool(
            input_grid,
            canvas,
            info["input_patch"],
            bg,
            info["target_frame_bbox"],
            frame_bboxes,
        ):
            return None

    return input_grid, output


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPT_RETRIES):
        built = _gen_attempt()
        if built is None:
            continue
        inp, out = built
        if np.array_equal(main(inp), out):
            return inp
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPT_RETRIES} attempts")
