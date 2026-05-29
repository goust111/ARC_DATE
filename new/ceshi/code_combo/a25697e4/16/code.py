# Combo variant of a25697e4 (index 16)
# combo_spec: D1/4+D9/2+D18/2
# Name: 多对面积排序尺寸分化
#
# Design analysis:
# - Concrete behavior trace:
#   - Original task: 1-2 detached two-color tools are paired with border frames by a local condition:
#     one tool color-region has the same area as one cavity and, under some D4 transform, that region
#     fills the cavity while the other region protrudes outward.
#   - Fused variant: 3-4 pairs are always present. Frames now come in a visibly strong size ladder
#     (roughly 3x4, 5x6, 7x8, 10x10 frame bboxes), so the cavities are also strictly area-ranked.
#     Tools are matched globally by total tool area rank: largest tool -> largest cavity, second
#     largest -> second largest, ..., smallest -> smallest. After that rank pairing is fixed, the old
#     docking transform logic still applies unchanged: rotate/flip the matched tool until one color
#     region fills the matched cavity exactly, and if the winning docking transform uses a reflection
#     and the two color counts differ, swap the tool's two colors.
#   - Small example: suppose the four cavities have areas [14, 11, 8, 5] and the four detached tools
#     have total areas [32, 25, 19, 13]. Even if the 19-pixel tool is physically closest to the 14-cell
#     frame and one handle area happens to equal 11, it still docks only into the 8-cell frame because
#     it is the third-largest tool. The cavity-fill transform is then searched only within that ranked
#     pair.
# - Complexity calibration:
#   - The frame/tool geometry family stays the same as the source task, so the fusion is readable:
#     staircase cavity, two-color connected tool, D4 docking, reflection-dependent swap.
#   - The new difficulty comes from jointly tracking 3-4 objects and ranking both sides by a human-
#     visible quantity. Cavity areas are all distinct, tool totals are all distinct, and the largest
#     frame is much larger than the smallest one, so the ranking cue is obvious rather than hidden.
#   - To stop the scene from collapsing back into a purely local match, the generator deliberately
#     places detached tools near the wrong frames and gives several handles the same area as the next
#     larger cavity. That creates visible decoys without making the intended output ambiguous.
# - Hidden-assumption audit against the original solver:
#   - Original assumption: "compatible iff one tool color count equals one cavity size" is enough to
#     choose the pair. Broken here. The new solver must instead:
#     1. recover every border frame and measure its cavity area,
#     2. recover every detached tool and measure its total foreground area,
#     3. sort those two lists independently and pair by rank,
#     4. only then run the old placement search inside each matched pair.
#   - Original assumption: with at most two pairs, incidental distance heuristics are harmless.
#     Broken here because 3-4 tools are intentionally spatially scrambled; the generator makes nearest-
#     frame pairing disagree with rank pairing in most samples.
#   - Original assumption: frame sizes live in a narrow band. Broken here because large-vs-small frame
#     disparity is part of the visible rule, so the generator uses a fixed multi-scale ladder.
# - Placement legality:
#   - Output is still built first. Every border frame is placed in-bounds on a distinct side with bbox
#     gap constraints; free tools are then undocked and re-placed at non-overlapping interior locations.
#   - The generator revalidates each sample with `main(input) == output` before accepting it.
# - Visual observability and cleanliness:
#   - The solver never uses numeric color ids as a ranking key; only visible pixel counts and shapes.
#   - The frame-size ladder is intentionally coarse, so humans can tell "tiny / medium / large / huge"
#     by eye from 2-4 examples.
#   - Detached tools are separated by padding and border margin, frames sit on different sides, and no
#     extra markings are added, keeping the grid readable despite having 3-4 pairs.
# - New solver assumptions and generator guarantees:
#   - Each frame group is the union of exactly two nearby monochrome border-touching sub-components with
#     one staircase cavity. The generator constructs frames only from that template family.
#   - Detached tools are single 4-connected, exactly two-color components. The generator enforces
#     distinct total areas and distinct cavity areas, and preserves the same D4 docking family as the
#     original task.

from __future__ import annotations

from collections import Counter
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, get_bg, paste, random_sprite


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


# ---- canvas / pair-count configuration ----

GRID_H_RANGE = (36, 46)
GRID_W_RANGE = (40, 50)
PAIR_CONFIGS = [
    (3, [0, 1, 3]),
    (3, [0, 2, 3]),
    (4, [0, 1, 2, 3]),
    (4, [0, 1, 2, 3]),
]


# ---- frame size ladder (D18/2) ----

RANKED_FRAME_SPECS = [
    {"fw": 4, "top_h": 1, "bot_h": 1, "notch_n": 1},   # bbox 3x4, cavity 5
    {"fw": 6, "top_h": 2, "bot_h": 2, "notch_n": 2},   # bbox 5x6, cavity 8
    {"fw": 8, "top_h": 3, "bot_h": 3, "notch_n": 3},   # bbox 7x8, cavity 11
    {"fw": 10, "top_h": 4, "bot_h": 5, "notch_n": 4},  # bbox 10x10, cavity 14
]


# ---- tool sizing / placement ----

HANDLE_EXTRA_FOR_LARGEST = (4, 6)
UNION_BBOX_GAP_MIN = 3
FRAME_BBOX_GAP_MIN = 4
TOOL_BORDER = 1
TOOL_PADDING = 3
TOOL_PADDING_CONN = 4
TARGET_TOOL_FRAME_GAP = 5
GEN_ATTEMPT_RETRIES = 240
HANDLE_SPRITE_RETRIES = 40


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


def _notch_offsets(notch_n: int, fw: int, bot_h: int) -> list[tuple[int, int]]:
    if notch_n == 1:
        return [(0, fw - 1)]
    if notch_n == 2:
        return [(0, fw - 2), (0, fw - 1)]
    if notch_n == 3:
        if bot_h >= 2:
            return [(0, fw - 2), (0, fw - 1), (1, fw - 1)]
        return [(0, fw - 3), (0, fw - 2), (0, fw - 1)]
    if notch_n == 4:
        if bot_h >= 3:
            return [(0, fw - 3), (0, fw - 2), (0, fw - 1), (1, fw - 1)]
        if bot_h >= 2:
            return [(0, fw - 3), (0, fw - 2), (1, fw - 2), (1, fw - 1)]
        return [(0, fw - 4), (0, fw - 3), (0, fw - 2), (0, fw - 1)]
    raise ValueError(notch_n)


def _make_shape_sprite(local_coords: list[tuple[int, int]], bg: int, color: int) -> np.ndarray:
    rs = [r for r, _ in local_coords]
    cs = [c for _, c in local_coords]
    h = max(rs) + 1
    w = max(cs) + 1
    out = np.full((h, w), int(bg), dtype=int)
    for r, c in local_coords:
        out[int(r), int(c)] = int(color)
    return out


def _make_handle_sprite(
    handle_area: int,
    bg: int,
    handle_color: int,
    template_coords: list[tuple[int, int]] | None,
) -> np.ndarray | None:
    if template_coords is not None:
        return _make_shape_sprite(template_coords, bg, handle_color)

    dims = [(h, w) for h in range(2, 7) for w in range(2, 8) if handle_area <= h * w <= handle_area * 2]
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
        if int(np.sum(sprite != bg)) == int(handle_area):
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
    handle_template_coords: list[tuple[int, int]] | None,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]] | None:
    """
    Build a docked pair in canonical left-border orientation.

    Returns:
        (frame_array, tool_array, cavity_local_coords)
    """
    fh = top_h + 1 + bot_h
    frame = np.full((fh, fw), int(bg), dtype=int)
    frame[:top_h, :] = int(frame_color)
    frame[top_h + 1 :, :] = int(frame_color)
    for dr, dc in _notch_offsets(notch_n, fw, bot_h):
        frame[top_h + 1 + dr, dc] = int(bg)

    cavity = frame == int(bg)
    cavity_local = [(int(r), int(c)) for r, c in np.argwhere(cavity)]
    attach_rows = {int(r) for r in range(fh) if cavity[r, fw - 1]}

    handle_sprite = _make_handle_sprite(handle_area, bg, handle_color, handle_template_coords)
    if handle_sprite is None:
        return None

    hs_h, hs_w = handle_sprite.shape
    sprite_left_rows = {int(r) for r in range(hs_h) if handle_sprite[r, 0] != int(bg)}
    valid_offsets = [
        int(voff)
        for voff in range(max(1, fh + hs_h))
        if any((voff + sr) in attach_rows for sr in sprite_left_rows)
    ]
    if not valid_offsets:
        return None

    voff = random.choice(valid_offsets)
    total_h = max(fh, voff + hs_h)
    total_w = fw + hs_w
    frame_full = np.full((total_h, total_w), int(bg), dtype=int)
    frame_full[:fh, :fw] = frame

    tool = np.full((total_h, total_w), int(bg), dtype=int)
    tool[:fh, :fw][cavity] = int(plug_color)
    handle_mask = handle_sprite != int(bg)
    tool[voff : voff + hs_h, fw : fw + hs_w][handle_mask] = int(handle_color)
    return frame_full, tool, cavity_local


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


def _patch_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    pts = np.argwhere(mask)
    return (
        int(pts[:, 0].min()),
        int(pts[:, 1].min()),
        int(pts[:, 0].max()),
        int(pts[:, 1].max()),
    )


def _extract_scene(grid: np.ndarray) -> dict[str, object]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape
    fg = g != int(bg)
    lab, n = label(fg, structure=STRUCT4)

    fixed_mask = np.zeros((H, W), dtype=bool)
    fixed_components: list[dict[str, object]] = []
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
        fixed_components.append({"pts": pts_list, "bbox": bbox})
        for r, c in pts_list:
            fixed_mask[r, c] = True

    assert fixed_components, "expected border-touching frame components"
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

    anchor_groups: list[dict[str, object]] = []
    for idxs in groups.values():
        pts = [p for i in idxs for p in fixed_components[i]["pts"]]
        rs = [r for r, _ in pts]
        cs = [c for _, c in pts]
        bbox = (min(rs), min(cs), max(rs), max(cs))

        mask = np.zeros((H, W), dtype=bool)
        for r, c in pts:
            mask[r, c] = True

        socket_mask = np.zeros((H, W), dtype=bool)
        for r in range(bbox[0], bbox[2] + 1):
            for c in range(bbox[1], bbox[3] + 1):
                if int(g[r, c]) != int(bg):
                    continue
                for dr, dc in DIRS4:
                    rr = int(r + dr)
                    cc = int(c + dc)
                    if 0 <= rr < H and 0 <= cc < W and mask[rr, cc]:
                        socket_mask[r, c] = True
                        break

        cavity_mask = np.zeros((H, W), dtype=bool)
        sub_bg = g[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1] == int(bg)
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
                for dr, dc in DIRS4:
                    r2 = int(r + dr)
                    c2 = int(c + dc)
                    if bbox[0] <= r2 <= bbox[2] and bbox[1] <= c2 <= bbox[3] and mask[r2, c2]:
                        adj = True
                        break
                if adj:
                    break
            if not adj:
                continue
            key = (int(area), -int(cid))
            if best_key is None or key > best_key:
                best_key = key
                best_id = int(cid)
        assert best_id is not None, "expected one cavity per frame"
        for rr, cc in np.argwhere(sub_lab == best_id):
            cavity_mask[int(bbox[0] + rr), int(bbox[1] + cc)] = True

        cavity_size = int(cavity_mask.sum())

        dir_counts = [0, 0, 0, 0]
        for r, c in pts:
            for di, (dr, dc) in enumerate(DIRS4):
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(bg):
                    dir_counts[di] += 1
        open_dir = DIRS4[int(np.argmax(dir_counts))]

        anchor_groups.append(
            {
                "mask": mask,
                "bbox": bbox,
                "socket_mask": socket_mask,
                "cavity_mask": cavity_mask,
                "cavity_size": int(cavity_size),
                "open_dir": open_dir,
                "frame_area": int(mask.sum()),
            }
        )

    mov_mask = fg & (~fixed_mask)
    mov_lab, mov_n = label(mov_mask, structure=STRUCT4)
    mov_comps: list[dict[str, object]] = []
    for cid in range(1, int(mov_n) + 1):
        pts = np.argwhere(mov_lab == cid)
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        m = mov_lab[r0 : r1 + 1, c0 : c1 + 1] == int(cid)
        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        patch[~m] = 0
        colors = [int(x) for x in patch[m].flatten()]
        mov_comps.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "mask": m,
                "patch": patch,
                "size": int(len(pts)),
                "color_counts": dict(Counter(colors)),
            }
        )

    return {
        "bg": bg,
        "fixed_mask": fixed_mask,
        "anchor_groups": anchor_groups,
        "mov_comps": mov_comps,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock several detached tools into a multi-scale set of border frames by global size rank.

    Input structure:
    - Background is the most common color.
    - Border-touching foreground components cluster into 3-4 anchor frames; each frame contains a
      staircase-shaped cavity, and different frames have strongly different overall sizes.
    - The remaining foreground components are detached 2-color tools placed away from the border.

    Rule:
    - Sort the frames by cavity area from largest to smallest, and sort the detached tools by total
      foreground area from largest to smallest.
    - Pair the k-th largest tool with the k-th largest frame. For each paired tool/frame, rotate or
      flip the tool so that one color-region fills that frame's cavity exactly and the other region
      protrudes outward.
    - If the winning docking transform uses a reflection and the two tool colors have different pixel
      counts, swap those two colors in the docked result. Remove each tool from its original position.

    Key insight: Local shape matching is no longer the pairing key; the pairing is determined first by
    the global ranking of tool total area against frame cavity area, and only then by the old docking transform.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    scene = _extract_scene(g)
    bg = int(scene["bg"])
    fixed_mask = np.array(scene["fixed_mask"], dtype=bool)
    anchor_groups = list(scene["anchor_groups"])
    mov_comps = list(scene["mov_comps"])

    anchor_groups.sort(
        key=lambda grp: (
            -int(grp["cavity_size"]),
            -int(grp["frame_area"]),
            grp["bbox"][0],
            grp["bbox"][1],
        )
    )
    mov_comps.sort(
        key=lambda comp: (
            -int(comp["size"]),
            comp["bbox"][0],
            comp["bbox"][1],
        )
    )
    assert len(anchor_groups) == len(mov_comps) and 3 <= len(anchor_groups) <= 4, "expected 3-4 ranked pairs"

    def gen_variants(patch: np.ndarray, mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, bool]]:
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
            uniq.append((p, m, bool(is_flip)))
        return uniq

    out = np.full_like(g, int(bg))
    out[fixed_mask] = g[fixed_mask]
    placed_global = fixed_mask.copy()
    placed_groups = [np.array(grp["mask"], dtype=bool).copy() for grp in anchor_groups]

    for comp, grp, placed_grp in zip(mov_comps, anchor_groups, placed_groups):
        sock = np.array(grp["socket_mask"], dtype=bool)
        cav = np.array(grp["cavity_mask"], dtype=bool)
        odr, odc = grp["open_dir"]
        comp_counts = Counter(int(x) for x in comp["patch"][comp["mask"]].flatten())

        best_state = None
        best_score = None
        cav_sz = int(grp["cavity_size"])
        for p, m, is_flip in gen_variants(np.array(comp["patch"]), np.array(comp["mask"], dtype=bool)):
            h, w = m.shape
            coords = np.argwhere(m)
            cols_here = sorted({int(x) for x in p[m].flatten()})
            plug_colors = [int(col) for col in cols_here if int(((p == int(col)) & m).sum()) == cav_sz]
            if not plug_colors:
                continue
            for r0 in range(0, H - h + 1):
                for c0 in range(0, W - w + 1):
                    if any(placed_global[int(r0 + rr), int(c0 + cc)] for rr, cc in coords):
                        continue

                    adj = 0
                    for rr, cc in coords:
                        r = int(r0 + rr)
                        c = int(c0 + cc)
                        for dr, dc in DIRS4:
                            r2 = int(r + dr)
                            c2 = int(c + dc)
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
                        plug_score = (plug_full, int(plug_hits), -int(nonplug_in_cav), -int(plug_outside))
                        if plug_best is None or plug_score > plug_best:
                            plug_best = plug_score
                    score = tuple(plug_best) + (
                        int(cavity_hits),
                        int(adj),
                        int(outside),
                        int(socket_hits),
                        -int(_bbox_gap(grp["bbox"], (int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)))),
                        -int(r0),
                        -int(c0),
                    )
                    if best_score is None or score > best_score:
                        best_score = score
                        best_state = (p, m, int(r0), int(c0), bool(is_flip))

        assert best_state is not None, "must find a valid ranked docking placement"
        p, m, r0, c0, is_flip = best_state
        if is_flip and len(comp_counts) == 2 and len(set(comp_counts.values())) == 2:
            a, b = sorted(comp_counts.keys())
            p = _swap_colors(p, int(a), int(b))

        for rr, cc in np.argwhere(m):
            r = int(r0 + rr)
            c = int(c0 + cc)
            out[r, c] = int(p[rr, cc])
            placed_global[r, c] = True
            placed_grp[r, c] = True

    return out


def _prepare_specs(num_pairs: int, idxs: list[int]) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    for idx in idxs:
        base = dict(RANKED_FRAME_SPECS[idx])
        base["rank_index"] = int(idx)
        base["plug_area"] = int(base["fw"] + base["notch_n"])
        specs.append(base)
    specs.sort(key=lambda spec: int(spec["plug_area"]))
    if num_pairs == 3:
        for i, spec in enumerate(specs):
            if i == 0:
                spec["handle_area"] = int(specs[1]["plug_area"])
            elif i == 1:
                cands = [int(spec["plug_area"]) + 2, int(spec["plug_area"]) + 3, int(specs[2]["plug_area"]) - 1]
                used = {int(s["plug_area"]) for s in specs}
                handle_area = next((x for x in cands if x > int(spec["plug_area"]) and x not in used), int(spec["plug_area"]) + 3)
                spec["handle_area"] = int(handle_area)
            else:
                extra = random.randint(*HANDLE_EXTRA_FOR_LARGEST)
                spec["handle_area"] = int(spec["plug_area"] + extra)
        return specs

    for i, spec in enumerate(specs):
        if i < len(specs) - 1:
            spec["handle_area"] = int(specs[i + 1]["plug_area"])
        else:
            extra = random.randint(*HANDLE_EXTRA_FOR_LARGEST)
            spec["handle_area"] = int(spec["plug_area"] + extra)
    return specs

def _color_triples(bg: int, num_pairs: int) -> list[tuple[int, int, int]]:
    avail = [c for c in range(10) if c != int(bg)]
    random.shuffle(avail)
    frame_cols = avail[:num_pairs]
    other_cols = avail[num_pairs:] + avail[:num_pairs]
    triples: list[tuple[int, int, int]] = []
    cursor = 0
    for i in range(num_pairs):
        frame_color = int(frame_cols[i])
        plug_color = int(other_cols[cursor % len(other_cols)])
        cursor += 1
        while plug_color == frame_color:
            plug_color = int(other_cols[cursor % len(other_cols)])
            cursor += 1
        handle_color = int(other_cols[cursor % len(other_cols)])
        cursor += 1
        while handle_color in {frame_color, plug_color}:
            handle_color = int(other_cols[cursor % len(other_cols)])
            cursor += 1
        triples.append((frame_color, plug_color, handle_color))
    return triples


def _place_docked_pair(
    pair_idx: int,
    spec: dict[str, object],
    colors: tuple[int, int, int],
    side: str,
    bg: int,
    H: int,
    W: int,
    output: np.ndarray,
    input_grid: np.ndarray,
    union_bboxes: list[tuple[int, int, int, int]],
    frame_bboxes: list[tuple[int, int, int, int]],
) -> dict[str, object] | None:
    frame_color, plug_color, handle_color = colors

    result = _build_left_pair(
        bg=bg,
        frame_color=frame_color,
        plug_color=plug_color,
        handle_color=handle_color,
        fw=int(spec["fw"]),
        top_h=int(spec["top_h"]),
        bot_h=int(spec["bot_h"]),
        notch_n=int(spec["notch_n"]),
        handle_area=int(spec["handle_area"]),
        handle_template_coords=None,
    )
    if result is None:
        return None
    frame_left, tool_left, cavity_local = result
    scene_left = np.where(tool_left != bg, tool_left, frame_left)

    frame_scene = _orient_left_to_side(frame_left, side)
    tool = _orient_left_to_side(tool_left, side)
    scene = _orient_left_to_side(scene_left, side)
    ph, pw = scene.shape
    fr0, fc0, fr1, fc1 = _patch_bbox(frame_scene != bg)

    origin = None
    for r0, c0 in _candidate_origins(side, ph, pw, H, W):
        union_bbox = (int(r0), int(c0), int(r0 + ph - 1), int(c0 + pw - 1))
        frame_bbox = (int(r0 + fr0), int(c0 + fc0), int(r0 + fr1), int(c0 + fc1))
        if side in ("left", "right") and (frame_bbox[0] == 0 or frame_bbox[2] == H - 1):
            continue
        if side in ("top", "bottom") and (frame_bbox[1] == 0 or frame_bbox[3] == W - 1):
            continue
        if any(_bbox_gap(union_bbox, prev) < UNION_BBOX_GAP_MIN for prev in union_bboxes):
            continue
        if any(_bbox_gap(frame_bbox, prev) < FRAME_BBOX_GAP_MIN for prev in frame_bboxes):
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
    union_bboxes.append((int(r0), int(c0), int(r0 + ph - 1), int(c0 + pw - 1)))
    frame_bbox = (int(r0 + fr0), int(c0 + fc0), int(r0 + fr1), int(c0 + fc1))
    frame_bboxes.append(frame_bbox)

    return {
        "pair_idx": int(pair_idx),
        "tool": tool,
        "plug_color": int(plug_color),
        "handle_color": int(handle_color),
        "plug_area": int(spec["plug_area"]),
        "handle_area": int(spec["handle_area"]),
        "true_frame_bbox": frame_bbox,
        "frame_side": side,
        "tool_size": int(np.sum(tool != bg)),
        "cavity_local": cavity_local,
    }


def _tool_placement_candidates(
    canvas: np.ndarray,
    patch: np.ndarray,
    bg: int,
    preferred_bbox: tuple[int, int, int, int],
    true_bbox: tuple[int, int, int, int],
) -> list[tuple[int, int, int, int]]:
    H, W = canvas.shape
    h, w = patch.shape
    mask = patch != int(bg)
    occupied = canvas != int(bg)
    candidates: list[tuple[int, int, int, int]] = []
    for r0 in range(TOOL_BORDER, H - h + 1 - TOOL_BORDER):
        for c0 in range(TOOL_BORDER, W - w + 1 - TOOL_BORDER):
            if np.any(occupied[r0 : r0 + h, c0 : c0 + w] & mask):
                continue
            rr0 = max(0, r0 - TOOL_PADDING)
            cc0 = max(0, c0 - TOOL_PADDING)
            rr1 = min(H, r0 + h + TOOL_PADDING)
            cc1 = min(W, c0 + w + TOOL_PADDING)
            if np.any(occupied[rr0:rr1, cc0:cc1]):
                continue
            placed_bbox = (int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1))
            gap_true = _bbox_gap(placed_bbox, true_bbox)
            if gap_true < TARGET_TOOL_FRAME_GAP:
                continue
            gap_pref = _bbox_gap(placed_bbox, preferred_bbox)
            candidates.append((int(gap_pref), -int(gap_true), int(r0), int(c0)))
    candidates.sort()
    return candidates


def _undock_tool(
    info: dict[str, object],
    bg: int,
    input_grid: np.ndarray,
    placement_canvas: np.ndarray,
    preferred_bbox: tuple[int, int, int, int],
) -> bool:
    for patch, is_flip in _variant_choices(np.array(info["tool"]), int(bg)):
        if bool(is_flip) and int(info["plug_area"]) != int(info["handle_area"]):
            patch = _swap_colors(patch, int(info["plug_color"]), int(info["handle_color"]))
        candidates = _tool_placement_candidates(
            placement_canvas,
            patch,
            int(bg),
            preferred_bbox,
            tuple(info["true_frame_bbox"]),
        )
        if not candidates:
            continue
        top_band = candidates[: min(12, len(candidates))]
        _gap_pref, _neg_gap_true, r0, c0 = random.choice(top_band)
        paste(input_grid, patch, int(r0), int(c0), transparent=bg)
        paste(placement_canvas, patch, int(r0), int(c0), transparent=bg)
        return True
    return False


def _gen_attempt(num_pairs: int, rank_indices: list[int]) -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    bg = random.randint(0, 9)
    specs = _prepare_specs(num_pairs, rank_indices)
    colors = _color_triples(bg, num_pairs)
    sides = random.sample(["left", "right", "top", "bottom"], k=num_pairs)
    output = np.full((H, W), int(bg), dtype=int)
    input_grid = np.full((H, W), int(bg), dtype=int)
    union_bboxes: list[tuple[int, int, int, int]] = []
    frame_bboxes: list[tuple[int, int, int, int]] = []
    placed_tools: list[dict[str, object]] = []

    for i, spec in enumerate(specs):
        info = _place_docked_pair(
            i,
            spec,
            colors[i],
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

    placed_tools.sort(key=lambda info: int(info["tool_size"]))
    rank_frame_bboxes = [tuple(info["true_frame_bbox"]) for info in sorted(placed_tools, key=lambda info: int(info["tool_size"]))]
    # Prefer a wrong frame of adjacent rank; fall back to a cyclic side-order derangement if needed.
    preferred_bboxes: list[tuple[int, int, int, int]] = []
    if len(rank_frame_bboxes) == 3:
        perm = [1, 2, 0]
    else:
        perm = [1, 2, 3, 0]
    for i in range(len(rank_frame_bboxes)):
        preferred_bboxes.append(rank_frame_bboxes[perm[i]])

    placement_canvas = output.copy()
    for info, preferred_bbox in zip(placed_tools, preferred_bboxes):
        if not _undock_tool(info, bg, input_grid, placement_canvas, preferred_bbox):
            return None

    if not np.array_equal(main(input_grid), output):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    num_pairs, rank_indices = random.choice(PAIR_CONFIGS)
    for _ in range(GEN_ATTEMPT_RETRIES):
        grid = _gen_attempt(int(num_pairs), list(rank_indices))
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate input after {GEN_ATTEMPT_RETRIES} attempts")
