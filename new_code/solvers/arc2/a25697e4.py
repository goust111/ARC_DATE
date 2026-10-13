from __future__ import annotations
from collections import Counter, defaultdict, deque
import numpy as np
from scipy.ndimage import label
from arc_utils import STRUCT4, STRUCT8, get_bg

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
        if best_perm is not None:
            group_for_comp = [int(x) for x in best_perm]
        else:
            group_for_comp = [choose_group(comp["bbox"]) for comp in mov_comps]
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

        if best_state is None:
            continue
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


