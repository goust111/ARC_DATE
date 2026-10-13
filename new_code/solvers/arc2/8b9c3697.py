from __future__ import annotations
from collections import defaultdict
import numpy as np
from arc_utils import Color, bbox_coords, components_mask, get_bg

def main(grid: np.ndarray) -> np.ndarray:
    """
    Insert red components into interior sockets of large structures, leaving a black trail.

    Input structure:
    - bg = most common. Large non-bg, non-red structures with interior cavities.
    - Red components outside structures.

    Rule:
    - Interior sockets: bg cells inside a structure's bbox, adjacent to structure pixels.
    - For each structure, find the direction where rays from sockets hit a red component.
    - Slide that component into the socket; paint traversed cells black.
    - Unmatched reds vanish.

    Key insight: Docking direction is inferred by ray-casting from interior sockets to the nearest red component.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    RED = int(Color.RED)
    BLACK = int(Color.BLACK)

    assert not (int(RED) not in set(map(int, np.unique(g))))

    # --- connected components: red blobs ---
    red_comps = components_mask(g == int(RED), connectivity=4)
    assert not (not red_comps)

    red_coords: list[list[tuple[int, int]]] = []
    red_bbox: list[tuple[int, int, int, int]] = []
    red_at: dict[tuple[int, int], int] = {}

    for rid, coords in enumerate(red_comps):
        pts = [(int(r), int(c)) for r, c in coords]
        red_coords.append(pts)
        red_bbox.append(tuple(map(int, bbox_coords(coords))))
        for rc in pts:
            red_at[rc] = int(rid)

    # --- connected components: structures (non-bg, non-red) ---
    struct_mask = (g != int(bg)) & (g != int(RED))
    struct_comps = components_mask(struct_mask, connectivity=4)
    if not struct_comps:
        # Nothing to dock into: reds disappear.
        out = g.copy()
        out[out == int(RED)] = int(bg)
        return out

    struct_coords: list[set[tuple[int, int]]] = []
    struct_bbox: list[tuple[int, int, int, int]] = []
    for coords in struct_comps:
        pts = {(int(r), int(c)) for r, c in coords}
        struct_coords.append(pts)
        struct_bbox.append(tuple(map(int, bbox_coords(coords))))

    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # For each structure, find its best (red, dir, anchor) triple.
    struct_best: list[tuple[tuple[int, int], int, tuple[int, int], tuple[int, int], int]] = []
    # entries: ((hit_count, -min_dist), rid, dir, anchor, sid)

    for sid, S in enumerate(struct_coords):
        sr0, sc0, sr1, sc1 = map(int, struct_bbox[sid])
        best = None  # (hit_count, -min_dist, rid, dir, anchor)

        for dr, dc in dirs:
            # rid -> [count, min_dist, best_anchor]
            hits: dict[int, list] = defaultdict(lambda: [0, 10**9, None])
            for (r, c) in S:
                nr, nc = int(r + dr), int(c + dc)
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                if int(g[nr, nc]) != int(bg):
                    continue
                # Socket must lie strictly inside the structure bbox.
                if not (sr0 < nr < sr1 and sc0 < nc < sc1):
                    continue

                rr, cc = int(nr), int(nc)
                while 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(bg):
                    rr += int(dr)
                    cc += int(dc)

                if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(RED):
                    rid = int(red_at[(int(rr), int(cc))])
                    rec = hits[rid]
                    rec[0] += 1
                    dist = abs(int(rr) - int(r)) + abs(int(cc) - int(c))
                    if int(dist) < int(rec[1]):
                        rec[1] = int(dist)
                        rec[2] = (int(r), int(c))

            if not hits:
                continue

            rid_best = max(hits.keys(), key=lambda k: (int(hits[k][0]), -int(hits[k][1])))
            cnt, mind, anchor = hits[rid_best]
            cand = (int(cnt), -int(mind), int(rid_best), (int(dr), int(dc)), anchor)
            if best is None or cand > best:
                best = cand

        if best is not None:
            cnt, neg_mind, rid, dirv, anchor = best
            struct_best.append(((int(cnt), int(neg_mind)), int(rid), (int(dirv[0]), int(dirv[1])), anchor, int(sid)))

    # Assign each red component to at most one structure (greedy best-first).
    struct_best.sort(reverse=True, key=lambda t: t[0])
    used_red: set[int] = set()
    assignments: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    for _score, rid, (dr, dc), anchor, _sid in struct_best:
        if int(rid) in used_red or anchor is None:
            continue
        used_red.add(int(rid))
        assignments.append((int(rid), (int(dr), int(dc)), (int(anchor[0]), int(anchor[1]))))

    # --- build output ---
    out = g.copy()
    out[out == int(RED)] = int(bg)  # all reds vanish unless re-drawn by an assignment

    for rid, (dr, dc), (ar, ac) in assignments:
        coords = red_coords[int(rid)]
        r0, c0, r1, c1 = map(int, red_bbox[int(rid)])
        tr, tc = int(ar + dr), int(ac + dc)  # socket cell

        # Shift so the component face nearest the structure lands on the socket.
        if dr == 0 and dc == 1:  # east; align min col
            shift_r, shift_c = 0, int(tc - c0)
        elif dr == 0 and dc == -1:  # west; align max col
            shift_r, shift_c = 0, int(tc - c1)
        elif dr == 1 and dc == 0:  # south; align min row
            shift_r, shift_c = int(tr - r0), 0
        else:  # north; align max row
            shift_r, shift_c = int(tr - r1), 0

        steps = int(max(abs(int(shift_r)), abs(int(shift_c))))
        step_dr = 0 if shift_r == 0 else (1 if shift_r > 0 else -1)
        step_dc = 0 if shift_c == 0 else (1 if shift_c > 0 else -1)

        # Paint trail black for intermediate placements (including the original),
        # but excluding the final placement where the red ends up.
        for step in range(0, steps):
            off_r = int(step * step_dr)
            off_c = int(step * step_dc)
            for r, c in coords:
                rr = int(r + off_r)
                cc = int(c + off_c)
                if 0 <= rr < H and 0 <= cc < W:
                    out[rr, cc] = int(BLACK)

        # Final placement of the red component.
        for r, c in coords:
            rr = int(r + shift_r)
            cc = int(c + shift_c)
            if 0 <= rr < H and 0 <= cc < W:
                out[rr, cc] = int(RED)

    return out



