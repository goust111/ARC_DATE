from __future__ import annotations
from collections import Counter, defaultdict, deque
import numpy as np
from scipy.ndimage import label
from arc_utils import STRUCT4, STRUCT8, get_bg, paste, random_free_location, random_sprite
import random

def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock movable tools into border-anchored frames by filling cavities.

    Input structure:
    - bg = most common color.
    - Border-touching components cluster into anchor frames with interior bg cavities.
    - Non-border 2-color components are movable tools.

    Rule:
    - Match each tool to a frame whose cavity size equals one of its color-region counts.
    - Rotate/flip the tool so one color fills the cavity exactly; the other stays outside.
    - If the docking orientation uses a reflection, swap the tool's two colors.

    Key insight: Cavity size determines tool–frame pairing; reflection triggers color swap.
    """
    inp = grid
    g = inp.copy()
    bg = get_bg(g)
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

    fg = g != int(bg)
    lab, n = label(fg, structure=STRUCT4)
    fixed_mask = np.zeros((H, W), dtype=bool)
    fixed_components = []
    for cid in range(1, n + 1):
        pts = np.argwhere(lab == cid)
        pts_list = [(int(r), int(c)) for r, c in pts]
        touches = any(r == 0 or r == H - 1 or c == 0 or c == W - 1 for r, c in pts_list)
        if not touches:
            continue
        fixed_components.append({"pts": pts_list, "bbox": (int(pts[:,0].min()), int(pts[:,1].min()), int(pts[:,0].max()), int(pts[:,1].max()))})
        for r, c in pts_list:
            fixed_mask[r, c] = True

    assert not (not fixed_components)

    # Cluster fixed components into anchor frames.
    parent = list(range(len(fixed_components)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(fixed_components)):
        for j in range(i + 1, len(fixed_components)):
            if bbox_gap(fixed_components[i]["bbox"], fixed_components[j]["bbox"]) <= 2:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(len(fixed_components)):
        groups.setdefault(find(i), []).append(i)

    anchor_groups = []
    for idxs in groups.values():
        pts = [p for i in idxs for p in fixed_components[i]["pts"]]
        rs = [r for r, _c in pts]
        cs = [c for _r, c in pts]
        bbox = (min(rs), min(cs), max(rs), max(cs))
        mask = np.zeros((H, W), dtype=bool)
        for r, c in pts:
            mask[r, c] = True
        sock = np.zeros((H, W), dtype=bool)
        for r in range(bbox[0], bbox[2] + 1):
            for c in range(bbox[1], bbox[3] + 1):
                if int(g[r, c]) != int(bg):
                    continue
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < H and 0 <= cc < W and mask[rr, cc]:
                        sock[r, c] = True
                        break

        # Cavity = largest bg component inside bbox that is adjacent to the anchor pixels.
        cavity = np.zeros((H, W), dtype=bool)
        sub_bg = (g[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1] == int(bg))
        sub_lab, sub_n = label(sub_bg, structure=STRUCT4)
        best_c = None  # (area, cid)
        for cid in range(1, sub_n + 1):
            ptsc = np.argwhere(sub_lab == cid)
            area = int(len(ptsc))
            # check adjacency to anchor within bbox
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
            cand = (area, -cid)
            if best_c is None or cand > best_c:
                best_c = cand
                best_id = cid
        if best_c is not None:
            ptsc = np.argwhere(sub_lab == best_id)
            for rr, cc in ptsc:
                cavity[int(bbox[0] + rr), int(bbox[1] + cc)] = True
        cav_bbox = None
        if cavity.any():
            ptsc2 = np.argwhere(cavity)
            r0c, c0c = ptsc2.min(axis=0)
            r1c, c1c = ptsc2.max(axis=0)
            cav_bbox = (int(r0c), int(c0c), int(r1c), int(c1c))

        # swap rule for this anchor group: look for a color that appears in >=2 fixed sub-components
        # (within the group) with different sizes and larger-above-smaller.
        swap_needed = False
        for col in sorted({int(x) for x in np.unique(g[mask])} - {int(bg)}):
            sub = (g == int(col)) & mask
            labc, nc = label(sub, structure=STRUCT4)
            if nc < 2:
                continue
            comps = []
            for cid in range(1, nc + 1):
                ptsc = np.argwhere(labc == cid)
                comps.append((int(len(ptsc)), float(ptsc[:, 0].mean()), float(ptsc[:, 1].mean())))
            comps.sort(reverse=True)
            (s0, r0, _c0), (s1, r1, _c1) = comps[0], comps[1]
            # Swap only when the larger component is vertically above the smaller one.
            if s0 != s1 and r0 < r1:
                swap_needed = True
                break

        # Dominant open direction: which side has the most bg neighbors.
        dir_list = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # U,D,L,R
        dir_counts = [0, 0, 0, 0]
        for r, c in pts:
            for di, (dr, dc) in enumerate(dir_list):
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(bg):
                    dir_counts[di] += 1
        open_dir = dir_list[int(np.argmax(dir_counts))]
        _labg, _ng = label(mask, structure=STRUCT4)
        single_piece = int(_ng) == 1
        cavity_size = int(cavity.sum())

        anchor_groups.append(
            {
                "mask": mask,
                "bbox": bbox,
                "socket_mask": sock,
                "cavity_mask": cavity,
                "cavity_bbox": cav_bbox,
                "open_dir": open_dir,
                "swap_needed": swap_needed,
                "single_piece": single_piece,
                "cavity_size": cavity_size,
                "size": int(mask.sum()),
            }
        )
    anchor_groups.sort(key=lambda d: -d["size"])

    # Movable multi-color components are the remaining foreground pixels.
    mov_mask = fg & (~fixed_mask)
    mov_lab, mov_n = label(mov_mask, structure=STRUCT4)
    mov_comps = []
    for cid in range(1, mov_n + 1):
        pts = np.argwhere(mov_lab == cid)
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        m = mov_lab[r0 : r1 + 1, c0 : c1 + 1] == cid
        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        patch[~m] = 0
        mov_comps.append({"bbox": (int(r0), int(c0), int(r1), int(c1)), "mask": m, "patch": patch, "size": int(len(pts))})
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

    def gen_variants(patch: np.ndarray, mask: np.ndarray):
        """All unique dihedral variants, with a reflection flag."""
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

    out = np.full_like(g, int(bg))
    out[fixed_mask] = g[fixed_mask]
    placed_global = fixed_mask.copy()
    placed_groups = [grp["mask"].copy() for grp in anchor_groups]

    group_for_comp: list[int]
    # Match condition (key): a tool can dock into a frame only if one of its two color-regions has
    # pixel-count equal to the frame's cavity size. Use this to assign tools to frames (one-to-one
    # when counts match), falling back to distance only if no such match exists.
    comp_color_counts: list[dict[int, int]] = []
    for comp in mov_comps:
        vals = [int(x) for x in comp["patch"][comp["mask"]].flatten() if int(x) != 0]
        comp_color_counts.append(dict(Counter(vals)))

    def compatible(ci: int, gi: int) -> bool:
        cav_sz = int(anchor_groups[gi]["cavity_size"])
        return any(int(v) == cav_sz for v in comp_color_counts[ci].values())

    if len(mov_comps) == len(anchor_groups) and len(anchor_groups) > 1:
        import itertools

        best_perm = None
        best_score = None
        for perm in itertools.permutations(range(len(anchor_groups))):
            ok = True
            hits = 0
            dist_sum = 0
            for ci, gi in enumerate(perm):
                if not compatible(ci, gi):
                    ok = False
                    break
                hits += 1
                dist_sum += bbox_gap(mov_comps[ci]["bbox"], anchor_groups[gi]["bbox"])
            if not ok:
                continue
            score = (int(hits), -int(dist_sum))
            if best_score is None or score > best_score:
                best_score = score
                best_perm = perm
        assert best_perm is not None, 'must find a compatible tool-to-frame permutation'
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

    for comp, gi in zip(mov_comps, group_for_comp):
        grp = anchor_groups[gi]
        placed_grp = placed_groups[gi]
        sock = grp["socket_mask"]
        cav = grp["cavity_mask"]
        cav_bbox = grp["cavity_bbox"]
        odr, odc = grp["open_dir"]
        # Swap rule (more principled): swap the two tool colors iff the *docking transform* uses a
        # reflection (flip). We'll detect this from the winning variant. (Ignore equal-area 2-color
        # tools: swapping is ambiguous and not needed.)
        comp_vals = [int(x) for x in comp["patch"][comp["mask"]].flatten() if int(x) != 0]
        comp_counts = Counter(comp_vals)

        best_state = None
        # Score prioritizes *filling the cavity* with exactly one tool color region.
        # (plug_full, plug_hits, -nonplug_in_cavity, -plug_outside, cavity_hits, adj, outside, socket_hits, -dist, -r0, -c0)
        best_score = None
        cav_sz = int(grp["cavity_size"])
        for p, m, is_flip in gen_variants(comp["patch"], comp["mask"]):
            h, w = m.shape
            coords = np.argwhere(m)
            cols_here = sorted({int(x) for x in p[m].flatten() if int(x) != 0})
            plug_colors = [int(col) for col in cols_here if int(((p == int(col)) & m).sum()) == cav_sz]
            assert plug_colors
            for r0 in range(0, H - h + 1):
                for c0 in range(0, W - w + 1):
                    ok = True
                    for rr, cc in coords:
                        r = int(r0 + rr)
                        c = int(c0 + cc)
                        if placed_global[r, c]:
                            ok = False
                            break
                    if not ok:
                        continue
                    adj = 0
                    for rr, cc in coords:
                        r = int(r0 + rr)
                        c = int(c0 + cc)
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            r2, c2 = r + dr, c + dc
                            if 0 <= r2 < H and 0 <= c2 < W and placed_grp[r2, c2]:
                                adj += 1
                    if adj == 0:
                        continue
                    socket_hits = 0
                    cavity_hits = 0
                    outside = 0
                    # Best plug-match for this placement among candidate plug colors.
                    plug_best = None  # (plug_full, plug_hits, -nonplug_in_cavity, -plug_outside)
                    for rr, cc in coords:
                        r = int(r0 + rr)
                        c = int(c0 + cc)
                        if sock[r, c]:
                            socket_hits += 1
                        if cav[r, c]:
                            cavity_hits += 1
                        # Count pixels placed outside the anchor bbox in the dominant open direction.
                        br0, bc0, br1, bc1 = grp["bbox"]
                        if odc == -1 and c < bc0:
                            outside += 1
                        elif odc == 1 and c > bc1:
                            outside += 1
                        elif odr == -1 and r < br0:
                            outside += 1
                        elif odr == 1 and r > br1:
                            outside += 1

                    for plug_col in plug_colors:
                        if plug_col is None:
                            plug_score = (0, 0, 0, 0)
                        else:
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
                                else:
                                    if in_cav:
                                        nonplug_in_cav += 1
                            plug_full = int(plug_hits == cav_sz and plug_outside == 0 and nonplug_in_cav == 0)
                            plug_score = (int(plug_full), int(plug_hits), -int(nonplug_in_cav), -int(plug_outside))
                        if plug_best is None or plug_score > plug_best:
                            plug_best = plug_score
                    dist = bbox_gap(grp["bbox"], (int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)))
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
                        best_state = (p, m, int(r0), int(c0), bool(is_flip))

        assert best_state is not None, 'must find a valid placement for every movable tool'
        p, m, r0, c0, is_flip = best_state
        if is_flip and len(comp_counts) == 2 and len(set(comp_counts.values())) == 2:
            a, b = sorted(comp_counts.keys())
            q = p.copy()
            q[(p == a) & m] = b
            q[(p == b) & m] = a
            p = q
        for rr, cc in np.argwhere(m):
            r = int(r0 + rr)
            c = int(c0 + cc)
            out[r, c] = int(p[rr, cc])
            placed_global[r, c] = True
            placed_grp[r, c] = True

    return out

# --------------- generator (O4) ---------------

from arc_utils import bbox as _bbox

# ── CFG: canvas ──
GRID_H_RANGE = (18, 30)          # output grid height
GRID_W_RANGE = (25, 30)          # output grid width
NUM_PAIRS_CHOICES = [1, 2]       # how many frame-tool pairs

# ── CFG: frame geometry ──
FRAME_W_RANGE = (4, 8)           # frame width (exclusive upper)
NOTCH_N_RANGE = (1, 4)           # notch cell count (exclusive upper)
TOP_H_RANGE = (1, 4)             # top bar height (exclusive upper)
BOT_H_RANGE = (2, 4)             # bottom bar height (exclusive upper)
PLUG_AREA_BOUNDS = (5, 8)        # valid plug area = fw + notch_n

# ── CFG: tool sizing ──
HANDLE_AREA_RANGE = (4, 10)      # handle pixel count (exclusive upper)
TOOL_TOTAL_BOUNDS = (10, 15)     # plug + handle total pixels
HANDLE_SPRITE_DIM = (2, 5)      # handle sprite bbox side (exclusive upper)

# ── CFG: placement ──
UNION_BBOX_GAP_MIN = 2           # min gap between pair union bboxes
FRAME_BBOX_GAP_MIN = 3           # min gap between frame bboxes
TOOL_BORDER = 1                  # border margin for free tool placement
TOOL_PADDING = 3                 # padding around existing content
TOOL_PADDING_CONN = 4            # connectivity for padding check

# ── CFG: retry limits ──
PAIR_SPEC_RETRIES = 300          # retries for picking compatible pair specs
HANDLE_SPRITE_RETRIES = 30       # retries for generating a valid handle sprite
GEN_ATTEMPT_RETRIES = 300        # outer retries in generate_input()


def _bbox_gap(a, b) -> int:
    """Manhattan gap between two axis-aligned bboxes (r0,c0,r1,c1)."""
    (ar0, ac0, ar1, ac1) = a
    (br0, bc0, br1, bc1) = b

    def _gap1(x0, x1, y0, y1):
        if y1 < x0:
            return x0 - y1
        if x1 < y0:
            return y0 - x1
        return 0

    return _gap1(ar0, ar1, br0, br1) + _gap1(ac0, ac1, bc0, bc1)


def _transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    """Apply rot90(k) then optional horizontal flip."""
    out = np.rot90(arr, k)
    if flip:
        out = np.fliplr(out)
    return out


def _orient_left_to_side(arr: np.ndarray, side: str) -> np.ndarray:
    """Rotate/flip an array built for the left border to target the given side."""
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
    """Swap two colors in a patch, returning a new array."""
    out = patch.copy()
    ma, mb = out == a, out == b
    out[ma] = b
    out[mb] = a
    return out


def _notch_offsets(notch_n: int, fw: int, bot_h: int) -> list[tuple[int, int]]:
    """Return (row, col) offsets within the bottom bar to carve out a staircase notch."""
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
    """Generate a random 4-connected handle sprite with exactly handle_area pixels."""
    lo, hi = HANDLE_SPRITE_DIM
    dims = [(h, w) for h in range(lo, hi) for w in range(lo, hi)
            if handle_area <= h * w <= handle_area * 2]
    if not dims:
        return None
    for _ in range(HANDLE_SPRITE_RETRIES):
        h, w = random.choice(dims)
        sprite = random_sprite(
            h, w, density=handle_area / (h * w),
            color_palette=[handle_color], connectivity=4, background=bg,
        )
        if int(np.sum(sprite != bg)) == handle_area:
            return sprite
    return None


def _build_left_pair(
    *, bg: int, frame_color: int, plug_color: int, handle_color: int,
    fw: int, top_h: int, bot_h: int, notch_n: int, handle_area: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Build a frame + docked tool pair oriented for the left border.

    Returns (frame_array, tool_array) on the same canvas, or None on failure.
    """
    fh = top_h + 1 + bot_h
    frame = np.full((fh, fw), bg, dtype=int)
    frame[:top_h, :] = frame_color
    frame[top_h + 1:, :] = frame_color
    for dr, dc in _notch_offsets(notch_n, fw, bot_h):
        frame[top_h + 1 + dr, dc] = bg

    cavity = frame == bg
    attach_rows = {r for r in range(fh) if cavity[r, fw - 1]}

    handle_sprite = _make_handle_sprite(handle_area, bg, handle_color)
    if handle_sprite is None:
        return None

    hs_h, hs_w = handle_sprite.shape
    sprite_left_rows = {r for r in range(hs_h) if handle_sprite[r, 0] != bg}

    valid_offsets = [
        voff for voff in range(fh)
        if any((voff + sr) in attach_rows for sr in sprite_left_rows)
    ]
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
    tool[voff:voff + hs_h, fw:fw + hs_w][handle_mask] = handle_color

    return frame_full, tool


def _build_param_options() -> list[tuple[int, int, int, int]]:
    """Enumerate all valid (fw, top_h, bot_h, notch_n) combinations."""
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
    """Pick frame geometry specs for each pair, ensuring distinct plug areas."""
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
            specs.append({"fw": fw, "top_h": top_h, "bot_h": bot_h,
                          "notch_n": notch_n, "plug_area": plug_area})
            used_plug_areas.add(plug_area)
        if len(specs) != num_pairs:
            continue
        all_plug_areas = {s["plug_area"] for s in specs}
        ok = True
        for spec in specs:
            pa = spec["plug_area"]
            handle_choices = [
                h for h in range(ha_lo, ha_hi)
                if tt_lo <= pa + h <= tt_hi and h != pa and h not in all_plug_areas
            ]
            if not handle_choices:
                ok = False
                break
            spec["handle_area"] = random.choice(handle_choices)
        if ok:
            return [dict(s) for s in specs]
    return []


def _candidate_origins(side: str, ph: int, pw: int, H: int, W: int) -> list[tuple[int, int]]:
    """Return shuffled candidate (r0, c0) origins for placing a patch on the given border side."""
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


def _variant_choices(patch: np.ndarray) -> list[tuple[np.ndarray, bool]]:
    """Return shuffled list of unique D4 variants of patch, each with its flip flag."""
    seen = set()
    out: list[tuple[np.ndarray, bool]] = []
    mask = patch != 0
    for k in range(4):
        for flip in (False, True):
            q = _transform(patch, k, flip)
            m = _transform(mask.astype(np.uint8), k, flip).astype(bool)
            key = (m.tobytes(), q[m].tobytes())
            if key in seen:
                continue
            seen.add(key)
            out.append((q, flip))
    random.shuffle(out)
    return out


def _place_docked_pair(
    i: int, spec: dict, colors: list[int], side: str, bg: int,
    H: int, W: int, output: np.ndarray, input_grid: np.ndarray,
    union_bboxes: list, frame_bboxes: list,
) -> dict | None:
    """Build one frame-tool pair, dock it onto the output, and record frame on input_grid.

    Returns tool info dict for later undocking, or None on failure.
    """
    frame_color = colors[3 * i]
    plug_color = colors[3 * i + 1]
    handle_color = colors[3 * i + 2]

    result = _build_left_pair(
        bg=bg, frame_color=frame_color, plug_color=plug_color,
        handle_color=handle_color, fw=spec["fw"], top_h=spec["top_h"],
        bot_h=spec["bot_h"], notch_n=spec["notch_n"],
        handle_area=spec["handle_area"],
    )
    if result is None:
        return None
    frame_left, tool_left = result

    # Combine frame + tool into a single docked scene
    scene_left = np.where(tool_left != bg, tool_left, frame_left)

    frame_scene = _orient_left_to_side(frame_left, side)
    tool = _orient_left_to_side(tool_left, side)
    scene = _orient_left_to_side(scene_left, side)
    ph, pw = scene.shape
    fr0, fc0, fr1, fc1 = _bbox(frame_scene != bg)

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
        if np.any((output[r0:r0 + ph, c0:c0 + pw] != bg) & (scene != bg)):
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
    return {
        "tool": tool, "plug_color": plug_color, "handle_color": handle_color,
        "plug_area": spec["plug_area"], "handle_area": spec["handle_area"],
    }


def _undock_tool(info: dict, bg: int, input_grid: np.ndarray,
                 placement_canvas: np.ndarray) -> bool:
    """Pick a random D4 variant of the tool and place it at a free location on the grid.

    Returns True on success, False if no placement found.
    """
    for patch, is_flip in _variant_choices(info["tool"]):
        if is_flip and info["plug_area"] != info["handle_area"]:
            patch = _swap_colors(patch, info["plug_color"], info["handle_color"])
        try:
            r0, c0 = random_free_location(
                placement_canvas, patch, transparent=bg,
                border=TOOL_BORDER, padding=TOOL_PADDING,
                padding_connectivity=TOOL_PADDING_CONN,
            )
        except ValueError:
            continue
        paste(input_grid, patch, r0, c0, transparent=bg)
        paste(placement_canvas, patch, r0, c0, transparent=bg)
        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid input grid."""
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    bg = random.choice(list(range(10)))
    num_pairs = random.choice(NUM_PAIRS_CHOICES)

    pair_specs = _pick_pair_specs(num_pairs)
    if len(pair_specs) != num_pairs:
        return None

    colors = [c for c in range(1, 10) if c != bg]
    random.shuffle(colors)
    need = 1 + (3 * num_pairs)
    if len(colors) < need:
        return None

    sides = random.sample(["left", "right", "top", "bottom"], k=num_pairs)
    output = np.full((H, W), bg, dtype=int)
    input_grid = np.full((H, W), bg, dtype=int)
    union_bboxes: list[tuple[int, int, int, int]] = []
    frame_bboxes: list[tuple[int, int, int, int]] = []
    placed_tools: list[dict] = []

    for i, spec in enumerate(pair_specs):
        info = _place_docked_pair(
            i, spec, colors, sides[i], bg, H, W,
            output, input_grid, union_bboxes, frame_bboxes,
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
    raise RuntimeError("failed to generate valid input after %d attempts" % GEN_ATTEMPT_RETRIES)
