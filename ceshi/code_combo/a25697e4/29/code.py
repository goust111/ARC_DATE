# Combo variant of a25697e4 (index 29)
# combo_spec: D5/0+D13/1
# Name: 面积差匹配矩形凹槽

from __future__ import annotations

from collections import Counter
import itertools
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, get_bg, paste, random_free_location, random_sprite
from arc_utils import bbox as _bbox

# Design analysis for the fused variant:
# - Concrete behavior: if a frame has a 2x2 rectangular cavity (area 4), the docked tool contains
#   a 2x2 plug-colored core that fills that cavity plus one extra same-color nub outside the frame.
#   The other color then occupies 9 cells, so the two tool colors have areas 5 and 9 and
#   |5 - 9| = 4. This prevents the task from collapsing back to the original "one color count
#   equals cavity size" rule while preserving a clearly visible rectangular plug core.
# - Complexity calibration: cavity areas are restricted to a small visible set {3, 4, 5} and are
#   unique within one sample. This keeps the arithmetic legible and avoids ambiguous many-to-many
#   matches when there are two frame-tool pairs.
# - Implicit assumptions revised: the old staircase-notch assumption is removed. The solver now
#   relies on a rectangular cavity and on exact cavity coverage by one tool color after some D4
#   transform; the matching stage uses the absolute difference of the two color counts instead of a
#   direct equality test. Reflection-based color swapping remains, but only matters because the two
#   color areas are intentionally unequal in every generated tool.
# - Placement legality: the generator builds output first with border-anchored C-frames, then
#   undocks each tool via a random D4 transform and a free-location search that enforces no border
#   contact, no overlap, and visible separation from the anchored frames.
# - Visual observability: rectangular cavities look materially different from the source task's
#   staircase cavities, and the tools repeatedly exhibit a "small rectangular core + tiny nub +
#   larger handle" structure. The rectangle docks into the rectangle, while the total color counts
#   only match through their difference, which is now necessary rather than redundant.


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


def _swap_colors(patch: np.ndarray, a: int, b: int) -> np.ndarray:
    out = patch.copy()
    ma = out == int(a)
    mb = out == int(b)
    out[ma] = int(b)
    out[mb] = int(a)
    return out


def _gen_variants(patch: np.ndarray, mask: np.ndarray):
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
    seen = set()
    out: list[tuple[np.ndarray, bool]] = []
    mask = patch != int(bg)
    for k in range(4):
        for flip in (False, True):
            q = _transform(patch, k, flip)
            m = _transform(mask.astype(np.uint8), k, flip).astype(bool)
            key = (m.tobytes(), q[m].tobytes())
            if key in seen:
                continue
            seen.add(key)
            out.append((q, bool(flip)))
    random.shuffle(out)
    return out


def _extract_anchor_groups(g: np.ndarray, bg: int):
    H, W = g.shape
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
        fixed_components.append(
            {
                "pts": pts_list,
                "bbox": (
                    int(pts[:, 0].min()),
                    int(pts[:, 1].min()),
                    int(pts[:, 0].max()),
                    int(pts[:, 1].max()),
                ),
            }
        )
        for r, c in pts_list:
            fixed_mask[r, c] = True

    assert fixed_components, "need at least one border-touching frame"

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
        sub_bg = g[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1] == int(bg)
        sub_lab, sub_n = label(sub_bg, structure=STRUCT4)
        best_id = None
        best_score = None
        for cid in range(1, sub_n + 1):
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
            score = (area, -cid)
            if best_score is None or score > best_score:
                best_score = score
                best_id = cid

        assert best_id is not None, "every generated frame exposes one bg cavity"
        ptsc = np.argwhere(sub_lab == best_id)
        for rr, cc in ptsc:
            cavity[int(bbox[0] + rr), int(bbox[1] + cc)] = True

        cav_pts = np.argwhere(cavity)
        r0c, c0c = cav_pts.min(axis=0)
        r1c, c1c = cav_pts.max(axis=0)
        cav_bbox = (int(r0c), int(c0c), int(r1c), int(c1c))
        cavity_size = int(cavity.sum())
        rect_area = (cav_bbox[2] - cav_bbox[0] + 1) * (cav_bbox[3] - cav_bbox[1] + 1)
        is_rectangle = cavity_size == int(rect_area)
        target_mask = np.zeros((H, W), dtype=bool)
        if is_rectangle:
            target_mask[cav_bbox[0] : cav_bbox[2] + 1, cav_bbox[1] : cav_bbox[3] + 1] = True
        else:
            target_mask = cavity.copy()

        edge_counts = [
            int(cavity[bbox[0], bbox[1] : bbox[3] + 1].sum()),
            int(cavity[bbox[2], bbox[1] : bbox[3] + 1].sum()),
            int(cavity[bbox[0] : bbox[2] + 1, bbox[1]].sum()),
            int(cavity[bbox[0] : bbox[2] + 1, bbox[3]].sum()),
        ]
        open_dir = [(-1, 0), (1, 0), (0, -1), (0, 1)][int(np.argmax(edge_counts))]

        anchor_groups.append(
            {
                "mask": mask,
                "bbox": bbox,
                "cavity_mask": target_mask,
                "cavity_bbox": cav_bbox,
                "cavity_size": cavity_size,
                "open_dir": open_dir,
                "size": int(mask.sum()),
                "is_rectangle": bool(is_rectangle),
            }
        )

    anchor_groups.sort(key=lambda d: -d["size"])
    return anchor_groups, fixed_mask


def _extract_movable(g: np.ndarray, fixed_mask: np.ndarray, bg: int):
    fg = g != int(bg)
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
        mov_comps.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "mask": m,
                "patch": patch,
                "size": int(len(pts)),
            }
        )
    mov_comps.sort(key=lambda d: -d["size"])
    return mov_comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock loose two-color tools into border frames with rectangular cavities.

    Input structure:
    - The background is the most frequent color.
    - Border-touching same-color structures act as anchored frames; each frame exposes one bg
      cavity whose occupied bbox is a rectangle.
    - Non-border foreground components are movable tools, each made of exactly two colors.

    Rule:
    - Pair a tool with the frame whose cavity area equals the absolute difference of the tool's two
      color-region areas.
    - Rotate and/or reflect the tool until one color occupies the cavity rectangle exactly; that
      same color may have a tiny extra nub outside the frame, while the other color forms the
      outward handle.
    - If the winning docking transform uses a reflection and the two color counts are unequal,
      swap the two tool colors before painting the output.

    Key insight: the cavity rectangle identifies the plug-colored core geometrically, but no color
    count matches the cavity size directly; only the difference between the two color totals does.
    """
    g = grid.copy()
    bg = get_bg(g)
    H, W = g.shape

    anchor_groups, fixed_mask = _extract_anchor_groups(g, bg)
    mov_comps = _extract_movable(g, fixed_mask, bg)

    out = np.full_like(g, int(bg))
    out[fixed_mask] = g[fixed_mask]

    def choose_group(bbox) -> int:
        best_i = 0
        best_d = None
        for i, grp in enumerate(anchor_groups):
            d = _bbox_gap(bbox, grp["bbox"])
            if best_d is None or d < best_d:
                best_d = d
                best_i = i
        return int(best_i)

    comp_color_counts: list[dict[int, int]] = []
    for comp in mov_comps:
        vals = [int(x) for x in comp["patch"][comp["mask"]].flatten() if int(x) != 0]
        comp_color_counts.append(dict(Counter(vals)))

    def compatible(ci: int, gi: int) -> bool:
        counts = list(comp_color_counts[ci].values())
        if len(counts) != 2:
            return False
        return abs(int(counts[0]) - int(counts[1])) == int(anchor_groups[gi]["cavity_size"])

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
            score = (-int(dist_sum),)
            if best_score is None or score > best_score:
                best_score = score
                best_perm = perm
        group_for_comp = [int(x) for x in best_perm] if best_perm is not None else []
    else:
        group_for_comp = []

    if len(group_for_comp) != len(mov_comps):
        group_for_comp = []
        for ci, comp in enumerate(mov_comps):
            best_g = None
            best_d = None
            for gi, grp in enumerate(anchor_groups):
                if not compatible(ci, gi):
                    continue
                d = _bbox_gap(comp["bbox"], grp["bbox"])
                if best_d is None or d < best_d:
                    best_d = d
                    best_g = gi
            group_for_comp.append(int(best_g) if best_g is not None else int(choose_group(comp["bbox"])))

    candidate_lists: list[list[dict]] = []
    for ci, (comp, gi) in enumerate(zip(mov_comps, group_for_comp)):
        grp = anchor_groups[gi]
        cav = grp["cavity_mask"]
        cav_sz = int(grp["cavity_size"])
        odr, odc = grp["open_dir"]
        br0, bc0, br1, bc1 = grp["bbox"]
        comp_candidates: list[dict] = []
        for p, m, is_flip in _gen_variants(comp["patch"], comp["mask"]):
            h, w = m.shape
            coords = np.argwhere(m)
            cols_here = sorted({int(x) for x in p[m].flatten() if int(x) != 0})
            if len(cols_here) != 2:
                continue
            for r0 in range(0, H - h + 1):
                for c0 in range(0, W - w + 1):
                    blocked = False
                    for rr, cc in coords:
                        r = int(r0 + rr)
                        c = int(c0 + cc)
                        if fixed_mask[r, c]:
                            blocked = True
                            break
                    if blocked:
                        continue

                    adj = 0
                    cavity_hits = 0
                    outside = 0
                    for rr, cc in coords:
                        r = int(r0 + rr)
                        c = int(c0 + cc)
                        if cav[r, c]:
                            cavity_hits += 1
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            r2, c2 = r + dr, c + dc
                            if 0 <= r2 < H and 0 <= c2 < W and grp["mask"][r2, c2]:
                                adj += 1
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
                        plug_hits = 0
                        plug_outside = 0
                        nonplug_in_cavity = 0
                        for rr, cc in coords:
                            r = int(r0 + rr)
                            c = int(c0 + cc)
                            in_cavity = bool(cav[r, c])
                            col = int(p[rr, cc])
                            if col == int(plug_col):
                                if in_cavity:
                                    plug_hits += 1
                                else:
                                    plug_outside += 1
                            elif in_cavity:
                                nonplug_in_cavity += 1
                        plug_full = int(plug_hits == cav_sz and nonplug_in_cavity == 0)
                        plug_score = (
                            int(plug_full),
                            int(plug_hits),
                            -int(nonplug_in_cavity),
                            -int(plug_outside),
                        )
                        if plug_best is None or plug_score > plug_best:
                            plug_best = plug_score

                    dist = _bbox_gap(
                        grp["bbox"], (int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1))
                    )
                    score = tuple(plug_best) + (
                        int(cavity_hits),
                        int(outside),
                        int(adj),
                        -int(dist),
                        -int(r0),
                        -int(c0),
                    )
                    if int(score[0]) != 1:
                        continue

                    occ_mask = np.zeros((H, W), dtype=bool)
                    for rr, cc in coords:
                        occ_mask[int(r0 + rr), int(c0 + cc)] = True
                    comp_candidates.append(
                        {
                            "patch": p.copy(),
                            "mask": m.copy(),
                            "r0": int(r0),
                            "c0": int(c0),
                            "is_flip": bool(is_flip),
                            "score": score,
                            "occ_mask": occ_mask,
                        }
                    )

        assert comp_candidates, "tool must fully fill one cavity"
        comp_candidates.sort(key=lambda cand: cand["score"], reverse=True)
        candidate_lists.append(comp_candidates[:64])

    score_len = len(candidate_lists[0][0]["score"])
    order = sorted(range(len(candidate_lists)), key=lambda idx: len(candidate_lists[idx]))
    occupied = np.zeros((H, W), dtype=bool)
    chosen_states: list[dict | None] = [None] * len(mov_comps)
    best_choice: list[dict] | None = None
    best_total: tuple[int, ...] | None = None

    def search_combo(depth: int, total: tuple[int, ...]) -> None:
        nonlocal best_choice, best_total
        if depth == len(order):
            if best_total is None or total > best_total:
                best_total = total
                best_choice = [cand for cand in chosen_states if cand is not None]
            return

        ci = order[depth]
        for cand in candidate_lists[ci]:
            if np.any(occupied & cand["occ_mask"]):
                continue
            occupied[cand["occ_mask"]] = True
            chosen_states[ci] = cand
            next_total = tuple(total[i] + int(cand["score"][i]) for i in range(score_len))
            search_combo(depth + 1, next_total)
            chosen_states[ci] = None
            occupied[cand["occ_mask"]] = False

    search_combo(0, tuple(0 for _ in range(score_len)))
    assert best_choice is not None and len(best_choice) == len(mov_comps), "must place all movable tools"

    for ci, cand in enumerate(best_choice):
        comp_counts = Counter(
            int(x) for x in mov_comps[ci]["patch"][mov_comps[ci]["mask"]].flatten() if int(x) != 0
        )
        p = cand["patch"].copy()
        m = cand["mask"]
        r0 = int(cand["r0"])
        c0 = int(cand["c0"])
        is_flip = bool(cand["is_flip"])
        if is_flip and len(comp_counts) == 2 and len(set(comp_counts.values())) == 2:
            a, b = sorted(comp_counts.keys())
            p = _swap_colors(p, a, b)

        for rr, cc in np.argwhere(m):
            r = int(r0 + rr)
            c = int(c0 + cc)
            out[r, c] = int(p[rr, cc])

    return out


# --------------- generator ---------------

GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (25, 30)
NUM_PAIRS_CHOICES = [1, 2]

CAVITY_OPTIONS = {
    3: [(1, 3)],
    4: [(2, 2), (1, 4)],
    5: [(1, 5)],
}
STEM_W_CHOICES = [1, 2]
BAR_H_CHOICES = [1, 2]
HANDLE_SPRITE_DIM = (2, 6)
PLUG_EXTRA_CELLS = 1

UNION_BBOX_GAP_MIN = 2
FRAME_BBOX_GAP_MIN = 3
TOOL_BORDER = 1
TOOL_PADDING = 3
TOOL_PADDING_CONN = 4

PAIR_SPEC_RETRIES = 200
HANDLE_SPRITE_RETRIES = 40
GEN_ATTEMPT_RETRIES = 300


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
        try:
            sprite = random_sprite(
                h,
                w,
                density=handle_area / (h * w),
                color_palette=[handle_color],
                connectivity=4,
                background=bg,
            )
        except AssertionError:
            continue
        if int(np.sum(sprite != bg)) != handle_area:
            continue
        if int(np.sum(sprite[:, 0] != bg)) == 0:
            continue
        return sprite
    return None


def _build_left_pair(
    *,
    bg: int,
    frame_color: int,
    plug_color: int,
    handle_color: int,
    cav_h: int,
    cav_w: int,
    stem_w: int,
    top_h: int,
    bot_h: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    frame_h = top_h + cav_h + bot_h
    frame_w = stem_w + cav_w

    frame = np.full((frame_h, frame_w), bg, dtype=int)
    frame[:, :stem_w] = frame_color
    frame[:top_h, :] = frame_color
    frame[top_h + cav_h :, :] = frame_color

    plug_mask = np.zeros((frame_h, frame_w), dtype=bool)
    plug_mask[top_h : top_h + cav_h, stem_w : stem_w + cav_w] = True

    plug_area = int(plug_mask.sum())
    plug_total_area = plug_area + PLUG_EXTRA_CELLS
    handle_area = plug_total_area + plug_area
    handle_sprite = _make_handle_sprite(handle_area, bg, handle_color)
    if handle_sprite is None:
        return None

    hs_h, hs_w = handle_sprite.shape
    plug_rows = set(range(top_h, top_h + cav_h))
    nub_row = random.choice(sorted(plug_rows))
    handle_left_rows = {r for r in range(hs_h) if handle_sprite[r, 0] != bg}
    valid_offsets = [
        voff
        for voff in range(max(1, frame_h - hs_h + cav_h + 1))
        if any((voff + sr) == nub_row for sr in handle_left_rows)
    ]
    if not valid_offsets:
        return None
    voff = random.choice(valid_offsets)

    total_h = max(frame_h, voff + hs_h)
    total_w = frame_w + 1 + hs_w
    frame_full = np.full((total_h, total_w), bg, dtype=int)
    frame_full[:frame_h, :frame_w] = frame

    tool = np.full((total_h, total_w), bg, dtype=int)
    tool[:frame_h, :frame_w][plug_mask] = plug_color
    tool[nub_row, frame_w] = plug_color
    handle_mask = handle_sprite != bg
    tool[voff : voff + hs_h, frame_w + 1 : frame_w + 1 + hs_w][handle_mask] = handle_color
    return frame_full, tool


def _pick_pair_specs(num_pairs: int) -> list[dict]:
    areas = list(CAVITY_OPTIONS.keys())
    for _ in range(PAIR_SPEC_RETRIES):
        chosen_areas = random.sample(areas, k=num_pairs)
        specs = []
        for area in chosen_areas:
            cav_h, cav_w = random.choice(CAVITY_OPTIONS[area])
            specs.append(
                {
                    "cav_h": cav_h,
                    "cav_w": cav_w,
                    "plug_area": area,
                    "handle_area": 2 * area + PLUG_EXTRA_CELLS,
                    "stem_w": random.choice(STEM_W_CHOICES),
                    "top_h": random.choice(BAR_H_CHOICES),
                    "bot_h": random.choice(BAR_H_CHOICES),
                }
            )
        if len({spec["plug_area"] for spec in specs}) == num_pairs:
            return specs
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
    frame_color = colors[3 * i]
    plug_color = colors[3 * i + 1]
    handle_color = colors[3 * i + 2]

    result = _build_left_pair(
        bg=bg,
        frame_color=frame_color,
        plug_color=plug_color,
        handle_color=handle_color,
        cav_h=spec["cav_h"],
        cav_w=spec["cav_w"],
        stem_w=spec["stem_w"],
        top_h=spec["top_h"],
        bot_h=spec["bot_h"],
    )
    if result is None:
        return None
    frame_left, tool_left = result
    scene_left = np.where(tool_left != bg, tool_left, frame_left)

    frame_scene = _orient_left_to_side(frame_left, side)
    tool = _orient_left_to_side(tool_left, side)
    scene = _orient_left_to_side(scene_left, side)
    ph, pw = scene.shape
    frame_bbox_local = _bbox(frame_scene != bg)
    assert frame_bbox_local is not None
    fr0, fc0, fr1, fc1 = frame_bbox_local

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
    return {
        "tool": tool,
        "plug_color": plug_color,
        "handle_color": handle_color,
        "plug_area": spec["plug_area"],
        "handle_area": spec["handle_area"],
    }


def _undock_tool(info: dict, bg: int, input_grid: np.ndarray, placement_canvas: np.ndarray) -> bool:
    for patch, is_flip in _variant_choices(info["tool"], bg):
        q = patch
        if is_flip and info["plug_area"] != info["handle_area"]:
            q = _swap_colors(q, info["plug_color"], info["handle_color"])
        try:
            r0, c0 = random_free_location(
                placement_canvas,
                q,
                transparent=bg,
                border=TOOL_BORDER,
                padding=TOOL_PADDING,
                padding_connectivity=TOOL_PADDING_CONN,
            )
        except ValueError:
            continue
        paste(input_grid, q, r0, c0, transparent=bg)
        paste(placement_canvas, q, r0, c0, transparent=bg)
        return True
    return False


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
    if len(colors) < 3 * num_pairs:
        return None

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
