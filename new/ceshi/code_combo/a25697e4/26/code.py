# Combo variant of a25697e4 (index 26)
# combo_spec: D7/0+D11/1
# Name: 链式对接轨迹保留
#
# Design analysis for the fused variant
# a) Concrete behavior:
#    Start from one free 2-color tool plus two border frames. The tool has region A and region B
#    with different areas, and frame 1 / frame 2 have cavity sizes matching those two areas.
#    Original a25697e4 would use one tool independently on one frame and erase the source. Here the
#    nearest compatible frame is used first, but the source copy stays visible. After docking into
#    frame 1, the placed result itself becomes the input to the second docking: its other color
#    region now fills frame 2. Output therefore shows three stages at once: source, intermediate,
#    final. Reflection-triggered color swapping is applied independently at each docking step.
# b) Complexity calibration:
#    To keep the chain legible, generation uses exactly one tool and exactly two frames. The two
#    cavity sizes are distinct and equal to the two color-region sizes, so each frame consumes a
#    different region of the same tool. Layout is limited to horizontal or vertical opposite-border
#    chains, and the source tool is sampled from a small family of clean pre-rotated shapes so both
#    docking steps exercise reflection and non-reflection cases without making the chain ambiguous.
# c) Hidden assumptions audit:
#    The original solver assumed multiple movable tools could be matched to multiple frames and
#    then disappear from their sources. That no longer holds. This combo solver instead assumes:
#    exactly one non-border 2-color component; exactly two border frame groups; source-to-frame
#    distance breaks the chain order; each docking has a unique exact cavity match; the second
#    docking is computed from the actual first-stage result, including any reflection-induced swap.
# d) Placement legality:
#    Generator builds the two frames first, computes both docked stages with the same exact docking
#    routine used by `main`, then searches a non-overlapping interior location for the preserved
#    source copy. This guarantees in-bounds placements and avoids accidental overlaps.
# e) Visual observability:
#    The output visibly contains a three-step trajectory. Distinct cavity sizes force the first and
#    second dockings to use different color regions, so the chain is not reducible to two unrelated
#    independent copies. The first frame is also the one closer to the source in the input, making
#    the order human-visible without relying on color ids or other hidden metadata.
# f) Visual cleanliness:
#    Only four colors are used: background, a shared frame color, and the two tool colors. Frames
#    are placed on opposite borders, and the preserved source is placed away from both docked
#    copies, so the three visible stages are easy to separate by eye.
# g) New solver assumptions checked by the generator:
#    No equal-area color regions; no distance ties between source and the two frames; exactly one
#    frame is first by distance; each frame touches exactly one border; the two cavity sizes are
#    unique; and the source is not on the border.

from __future__ import annotations

from collections import Counter
import random
from typing import Iterable

import numpy as np
from scipy.ndimage import label

from arc_utils import get_bg


STRUCT4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)

GRID_H_RANGE = (20, 28)
GRID_W_RANGE = (24, 30)
GEN_ATTEMPT_RETRIES = 300

FRAME_W_RANGE = (4, 7)      # exclusive upper bound
TOP_H_RANGE = (1, 3)
BOT_H_RANGE = (2, 4)
NOTCH_N_RANGE = (1, 4)
PLUG_AREA_BOUNDS = (5, 9)   # exclusive upper bound on upper end

SOURCE_BORDER = 1
SOURCE_GAP_MIN = 2

TOOL_FAMILIES = {
    "horizontal": [
        {"combiner": "horizontal", "shape_a": "id", "shape_b": "lr"},
        {"combiner": "horizontal", "shape_a": "id", "shape_b": "r2"},
        {"combiner": "horizontal", "shape_a": "ud", "shape_b": "lr"},
        {"combiner": "horizontal", "shape_a": "ud", "shape_b": "r2"},
    ],
    "vertical": [
        {"combiner": "horizontal", "shape_a": "id", "shape_b": "lr"},
        {"combiner": "horizontal", "shape_a": "id", "shape_b": "r2"},
        {"combiner": "horizontal", "shape_a": "ud", "shape_b": "lr"},
        {"combiner": "horizontal", "shape_a": "ud", "shape_b": "r2"},
    ],
}


def _bbox_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    (ar0, ac0, ar1, ac1) = a
    (br0, bc0, br1, bc1) = b

    def gap1(x0: int, x1: int, y0: int, y1: int) -> int:
        if y1 < x0:
            return int(x0 - y1)
        if x1 < y0:
            return int(y0 - x1)
        return 0

    return gap1(ar0, ar1, br0, br1) + gap1(ac0, ac1, bc0, bc1)


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_patch(patch: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r0, c0, r1, c1 = _mask_bbox(mask)
    return patch[r0 : r1 + 1, c0 : c1 + 1].copy(), mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _transform_patch(patch: np.ndarray, mask: np.ndarray, k: int, flip: bool) -> tuple[np.ndarray, np.ndarray]:
    p = np.rot90(patch, k)
    m = np.rot90(mask, k)
    if flip:
        p = np.fliplr(p)
        m = np.fliplr(m)
    return p, m


def _gen_variants(patch: np.ndarray, mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, bool]]:
    out: list[tuple[np.ndarray, np.ndarray, bool]] = []
    seen = set()
    for k in range(4):
        for flip in (False, True):
            p, m = _transform_patch(patch, mask, k, flip)
            q, mq = _crop_patch(p, m)
            key = (mq.tobytes(), q[mq].tobytes())
            if key in seen:
                continue
            seen.add(key)
            out.append((q, mq, bool(flip)))
    return out


def _swap_two_colors(patch: np.ndarray, mask: np.ndarray, a: int, b: int) -> np.ndarray:
    out = patch.copy()
    ma = (out == int(a)) & mask
    mb = (out == int(b)) & mask
    out[ma] = int(b)
    out[mb] = int(a)
    return out


def _color_counts(patch: np.ndarray, mask: np.ndarray) -> dict[int, int]:
    vals = [int(v) for v in patch[mask].tolist()]
    return dict(Counter(vals))


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


def _build_left_frame_array(bg: int, frame_color: int, fw: int, top_h: int, bot_h: int, notch_n: int) -> np.ndarray:
    fh = top_h + 1 + bot_h
    frame = np.full((fh, fw), int(bg), dtype=int)
    frame[:top_h, :] = int(frame_color)
    frame[top_h + 1 :, :] = int(frame_color)
    for dr, dc in _notch_offsets(notch_n, fw, bot_h):
        frame[top_h + 1 + dr, dc] = int(bg)
    return frame


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


def _side_open_dir(side: str) -> tuple[int, int]:
    return {
        "left": (0, 1),
        "right": (0, -1),
        "top": (1, 0),
        "bottom": (-1, 0),
    }[side]


def _side_from_bbox(bbox: tuple[int, int, int, int], H: int, W: int) -> str:
    r0, c0, r1, c1 = bbox
    if c0 == 0:
        return "left"
    if c1 == W - 1:
        return "right"
    if r0 == 0:
        return "top"
    if r1 == H - 1:
        return "bottom"
    raise AssertionError("frame must touch exactly one border")


def _cavity_shape_from_spec(spec: dict) -> np.ndarray:
    frame = _build_left_frame_array(0, 1, spec["fw"], spec["top_h"], spec["bot_h"], spec["notch_n"])
    cavity = frame == 0
    r0, c0, r1, c1 = _mask_bbox(cavity)
    return cavity[r0 : r1 + 1, c0 : c1 + 1].copy()


def _combine_shapes_horizontal(shape_a: np.ndarray, shape_b: np.ndarray, color_a: int, color_b: int, bg: int) -> tuple[np.ndarray, np.ndarray]:
    a_rows = {int(r) for r in range(shape_a.shape[0]) if bool(shape_a[r, -1])}
    b_rows = {int(r) for r in range(shape_b.shape[0]) if bool(shape_b[r, 0])}
    valid_offsets = []
    for voff in range(-shape_b.shape[0] + 1, shape_a.shape[0]):
        if any((voff + br) in a_rows for br in b_rows):
            valid_offsets.append(int(voff))
    assert valid_offsets
    valid_offsets.sort(key=lambda x: (abs(x), x))
    voff = valid_offsets[0]
    top = min(0, voff)
    bottom = max(shape_a.shape[0], voff + shape_b.shape[0])
    H = bottom - top
    W = shape_a.shape[1] + shape_b.shape[1]
    patch = np.full((H, W), int(bg), dtype=int)
    mask = np.zeros((H, W), dtype=bool)
    a_r0 = -top
    b_r0 = voff - top
    patch[a_r0 : a_r0 + shape_a.shape[0], : shape_a.shape[1]][shape_a] = int(color_a)
    mask[a_r0 : a_r0 + shape_a.shape[0], : shape_a.shape[1]][shape_a] = True
    patch[b_r0 : b_r0 + shape_b.shape[0], shape_a.shape[1] :][shape_b] = int(color_b)
    mask[b_r0 : b_r0 + shape_b.shape[0], shape_a.shape[1] :][shape_b] = True
    return _crop_patch(patch, mask)


def _combine_shapes_vertical(shape_a: np.ndarray, shape_b: np.ndarray, color_a: int, color_b: int, bg: int) -> tuple[np.ndarray, np.ndarray]:
    a_cols = {int(c) for c in range(shape_a.shape[1]) if bool(shape_a[-1, c])}
    b_cols = {int(c) for c in range(shape_b.shape[1]) if bool(shape_b[0, c])}
    valid_offsets = []
    for hoff in range(-shape_b.shape[1] + 1, shape_a.shape[1]):
        if any((hoff + bc) in a_cols for bc in b_cols):
            valid_offsets.append(int(hoff))
    assert valid_offsets
    valid_offsets.sort(key=lambda x: (abs(x), x))
    hoff = valid_offsets[0]
    left = min(0, hoff)
    right = max(shape_a.shape[1], hoff + shape_b.shape[1])
    H = shape_a.shape[0] + shape_b.shape[0]
    W = right - left
    patch = np.full((H, W), int(bg), dtype=int)
    mask = np.zeros((H, W), dtype=bool)
    a_c0 = -left
    b_c0 = hoff - left
    patch[: shape_a.shape[0], a_c0 : a_c0 + shape_a.shape[1]][shape_a] = int(color_a)
    mask[: shape_a.shape[0], a_c0 : a_c0 + shape_a.shape[1]][shape_a] = True
    patch[shape_a.shape[0] :, b_c0 : b_c0 + shape_b.shape[1]][shape_b] = int(color_b)
    mask[shape_a.shape[0] :, b_c0 : b_c0 + shape_b.shape[1]][shape_b] = True
    return _crop_patch(patch, mask)


def _apply_shape_transform(shape: np.ndarray, name: str) -> np.ndarray:
    if name == "id":
        return shape.copy()
    if name == "lr":
        return np.fliplr(shape)
    if name == "ud":
        return np.flipud(shape)
    if name == "r2":
        return np.rot90(shape, 2)
    raise ValueError(name)


def _build_base_tool(
    layout: str,
    shape_a: np.ndarray,
    shape_b: np.ndarray,
    color_a: int,
    color_b: int,
    bg: int,
) -> tuple[np.ndarray, np.ndarray]:
    family = random.choice(TOOL_FAMILIES[layout])
    ta = _apply_shape_transform(shape_a, family["shape_a"])
    tb = _apply_shape_transform(shape_b, family["shape_b"])
    if family["combiner"] == "horizontal":
        return _combine_shapes_horizontal(ta, tb, color_a, color_b, bg)
    if family["combiner"] == "vertical":
        return _combine_shapes_vertical(ta, tb, color_a, color_b, bg)
    raise ValueError(family["combiner"])


def _extract_frames(grid: np.ndarray, bg: int) -> tuple[np.ndarray, list[dict]]:
    H, W = grid.shape
    fg = grid != int(bg)
    lab, n = label(fg, structure=STRUCT4)
    fixed_mask = np.zeros((H, W), dtype=bool)
    fixed_components: list[dict] = []
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        if pts.size == 0:
            continue
        touches = bool((pts[:, 0] == 0).any() or (pts[:, 0] == H - 1).any() or (pts[:, 1] == 0).any() or (pts[:, 1] == W - 1).any())
        if not touches:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        fixed_components.append({"pts": pts.copy(), "bbox": (int(r0), int(c0), int(r1), int(c1))})
        fixed_mask[lab == int(cid)] = True
    assert fixed_components

    parent = list(range(len(fixed_components)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return int(x)

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

    frames: list[dict] = []
    for idxs in groups.values():
        mask = np.zeros((H, W), dtype=bool)
        pts = []
        for i in idxs:
            for r, c in fixed_components[i]["pts"]:
                r = int(r)
                c = int(c)
                mask[r, c] = True
                pts.append((r, c))
        rs = [r for r, _ in pts]
        cs = [c for _, c in pts]
        bbox = (min(rs), min(cs), max(rs), max(cs))
        side = _side_from_bbox(bbox, H, W)

        cavity = np.zeros((H, W), dtype=bool)
        sub_bg = grid[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1] == int(bg)
        sub_lab, sub_n = label(sub_bg, structure=STRUCT4)
        best_id = None
        best_area = -1
        for cid in range(1, int(sub_n) + 1):
            comp = sub_lab == int(cid)
            ptsc = np.argwhere(comp)
            if ptsc.size == 0:
                continue
            area = int(ptsc.shape[0])
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
            if adj and area > best_area:
                best_area = int(area)
                best_id = int(cid)
        assert best_id is not None
        cavity[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1] = sub_lab == int(best_id)
        frame_colors = [int(v) for v in grid[mask].tolist()]
        frame_color = Counter(frame_colors).most_common(1)[0][0]
        frames.append(
            {
                "mask": mask,
                "bbox": bbox,
                "side": side,
                "open_dir": _side_open_dir(side),
                "cavity_mask": cavity,
                "cavity_size": int(cavity.sum()),
                "frame_color": int(frame_color),
            }
        )
    frames.sort(key=lambda d: d["bbox"])
    return fixed_mask, frames


def _extract_tool(grid: np.ndarray, fixed_mask: np.ndarray, bg: int) -> dict:
    mov_mask = (grid != int(bg)) & (~fixed_mask)
    mov_lab, mov_n = label(mov_mask, structure=STRUCT4)
    comps = []
    for cid in range(1, int(mov_n) + 1):
        pts = np.argwhere(mov_lab == int(cid))
        if pts.size == 0:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        mask = mov_lab[r0 : r1 + 1, c0 : c1 + 1] == int(cid)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        patch[~mask] = int(bg)
        comps.append({"bbox": (int(r0), int(c0), int(r1), int(c1)), "patch": patch, "mask": mask, "size": int(mask.sum())})
    assert len(comps) == 1
    comp = comps[0]
    colors = sorted(set(int(v) for v in comp["patch"][comp["mask"]].tolist()))
    assert len(colors) == 2
    counts = _color_counts(comp["patch"], comp["mask"])
    assert len(set(counts.values())) == 2
    return comp


def _shifted_coords(coords: Iterable[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(int(r + dr), int(c + dc)) for r, c in coords}


def _dock_into_frame(
    patch: np.ndarray,
    mask: np.ndarray,
    frame: dict,
    occupied: np.ndarray,
) -> dict:
    H, W = occupied.shape
    cavity_coords = {(int(r), int(c)) for r, c in np.argwhere(frame["cavity_mask"])}
    cav_rmin = min(r for r, _ in cavity_coords)
    cav_cmin = min(c for _, c in cavity_coords)
    counts = _color_counts(patch, mask)
    unequal = len(set(counts.values())) == 2
    bbox = frame["bbox"]
    odr, odc = frame["open_dir"]
    best = None
    best_score = None

    for p, m, is_flip in _gen_variants(patch, mask):
        colors = sorted(set(int(v) for v in p[m].tolist()))
        for plug_color in colors:
            plug_mask = (p == int(plug_color)) & m
            plug_count = int(plug_mask.sum())
            if plug_count != int(frame["cavity_size"]):
                continue
            plug_coords = [(int(r), int(c)) for r, c in np.argwhere(plug_mask)]
            pr_min = min(r for r, _ in plug_coords)
            pc_min = min(c for _, c in plug_coords)
            dr = int(cav_rmin - pr_min)
            dc = int(cav_cmin - pc_min)
            shifted_plug = _shifted_coords(plug_coords, dr, dc)
            if shifted_plug != cavity_coords:
                continue
            full_coords = [(int(r + dr), int(c + dc)) for r, c in np.argwhere(m)]
            if any(not (0 <= r < H and 0 <= c < W) for r, c in full_coords):
                continue
            if any(bool(occupied[r, c]) for r, c in full_coords):
                continue
            nonplug = [(int(r + dr), int(c + dc)) for r, c in np.argwhere(m & (~plug_mask))]
            if odc == 1 and not all(c > bbox[3] for _r, c in nonplug):
                continue
            if odc == -1 and not all(c < bbox[1] for _r, c in nonplug):
                continue
            if odr == 1 and not all(r > bbox[2] for r, _c in nonplug):
                continue
            if odr == -1 and not all(r < bbox[0] for r, _c in nonplug):
                continue
            placed = p.copy()
            if is_flip and unequal:
                a, b = colors
                placed = _swap_two_colors(placed, m, a, b)
            coords_mask = np.zeros((H, W), dtype=bool)
            for r, c in full_coords:
                coords_mask[r, c] = True
            r0, c0, r1, c1 = _mask_bbox(coords_mask)
            cand = {
                "patch": placed,
                "mask": m.copy(),
                "top": int(dr),
                "left": int(dc),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
            }
            score = (-_bbox_gap(cand["bbox"], bbox), int(is_flip), -int(len(nonplug)))
            if best_score is None or score > best_score:
                best_score = score
                best = cand
    assert best is not None
    return best


def _stamp(out: np.ndarray, patch: np.ndarray, mask: np.ndarray, top: int, left: int) -> np.ndarray:
    res = out.copy()
    for rr, cc in np.argwhere(mask):
        res[int(top + rr), int(left + cc)] = int(patch[int(rr), int(cc)])
    return res


def _apply_stage(out: np.ndarray, occupied: np.ndarray, stage: dict) -> tuple[np.ndarray, np.ndarray]:
    out2 = _stamp(out, stage["patch"], stage["mask"], stage["top"], stage["left"])
    occ2 = occupied.copy()
    for rr, cc in np.argwhere(stage["mask"]):
        occ2[int(stage["top"] + rr), int(stage["left"] + cc)] = True
    return out2, occ2


def _build_output(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    fixed_mask, frames = _extract_frames(g, bg)
    assert len(frames) == 2
    tool = _extract_tool(g, fixed_mask, bg)
    counts = _color_counts(tool["patch"], tool["mask"])
    frame_sizes = sorted(int(fr["cavity_size"]) for fr in frames)
    assert sorted(counts.values()) == frame_sizes

    compatible = [fr for fr in frames if int(fr["cavity_size"]) in counts.values()]
    assert len(compatible) == 2
    first = min(compatible, key=lambda fr: _bbox_gap(tool["bbox"], fr["bbox"]))
    second = compatible[0] if compatible[1] is first else compatible[1]
    assert _bbox_gap(tool["bbox"], first["bbox"]) != _bbox_gap(tool["bbox"], second["bbox"])

    out = g.copy()
    occupied = fixed_mask.copy()
    occupied[tool["bbox"][0] : tool["bbox"][2] + 1, tool["bbox"][1] : tool["bbox"][3] + 1] |= tool["mask"]

    stage1 = _dock_into_frame(tool["patch"], tool["mask"], first, occupied)
    out, occ1 = _apply_stage(out, occupied, stage1)
    stage2 = _dock_into_frame(stage1["patch"], stage1["mask"], second, occ1)
    out, _occ2 = _apply_stage(out, occ1, stage2)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy one two-color tool through a two-step docking chain between border frames.

    Input structure:
    - bg is the most common color.
    - Exactly two border-touching frame groups each contain a staircase cavity.
    - Exactly one non-border 2-color tool appears in the interior, with its two region sizes
      matching the two frame cavity sizes.

    Rule:
    - Keep the original tool in place.
    - Dock the tool into the nearer compatible frame by rotating/flipping it so one color region
      fills that cavity exactly and the other protrudes outward.
    - Treat that docked result as a new tool and dock it again into the remaining frame, again
      applying the reflection-triggered color swap when the docking transform is a flip and the two
      color-region sizes are unequal.

    Key insight: The output is not just "source plus one docking"; it preserves the full chain
    source -> intermediate -> final, where the first docked copy becomes the second step's tool.
    """
    return _build_output(np.array(grid, dtype=int))


def _param_options() -> list[dict]:
    out: list[dict] = []
    for fw in range(*FRAME_W_RANGE):
        for top_h in range(*TOP_H_RANGE):
            for bot_h in range(*BOT_H_RANGE):
                for notch_n in range(*NOTCH_N_RANGE):
                    plug_area = fw + notch_n
                    if PLUG_AREA_BOUNDS[0] <= plug_area < PLUG_AREA_BOUNDS[1]:
                        out.append(
                            {
                                "fw": int(fw),
                                "top_h": int(top_h),
                                "bot_h": int(bot_h),
                                "notch_n": int(notch_n),
                                "plug_area": int(plug_area),
                            }
                        )
    return out


PARAM_OPTIONS = _param_options()


def _pick_specs() -> tuple[dict, dict]:
    specs = random.sample(PARAM_OPTIONS, k=2)
    while specs[0]["plug_area"] == specs[1]["plug_area"]:
        specs = random.sample(PARAM_OPTIONS, k=2)
    return dict(specs[0]), dict(specs[1])


def _candidate_origins_for_side(side: str, ph: int, pw: int, H: int, W: int) -> list[tuple[int, int]]:
    if side == "left":
        out = [(r0, 0) for r0 in range(2, H - ph - 1)]
    elif side == "right":
        out = [(r0, W - pw) for r0 in range(2, H - ph - 1)]
    elif side == "top":
        out = [(0, c0) for c0 in range(2, W - pw - 1)]
    elif side == "bottom":
        out = [(H - ph, c0) for c0 in range(2, W - pw - 1)]
    else:
        raise ValueError(side)
    random.shuffle(out)
    return out


def _place_frame(grid: np.ndarray, frame_arr: np.ndarray, side: str) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    H, W = grid.shape
    ph, pw = frame_arr.shape
    bg = int(get_bg(grid))
    for r0, c0 in _candidate_origins_for_side(side, ph, pw, H, W):
        sub = grid[r0 : r0 + ph, c0 : c0 + pw]
        mask = frame_arr != int(bg)
        if np.any((sub != int(bg)) & mask):
            continue
        out = grid.copy()
        out[r0 : r0 + ph, c0 : c0 + pw][mask] = frame_arr[mask]
        return out, (int(r0), int(c0), int(r0 + ph - 1), int(c0 + pw - 1))
    return None


def _random_tool_source_variant(base_patch: np.ndarray, base_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    variants = _gen_variants(base_patch, base_mask)
    p, m, is_flip = random.choice(variants)
    counts = _color_counts(p, m)
    if is_flip and len(set(counts.values())) == 2:
        cols = sorted(counts)
        p = _swap_two_colors(p, m, cols[0], cols[1])
    return p, m


def _place_source(
    H: int,
    W: int,
    bg: int,
    source_patch: np.ndarray,
    source_mask: np.ndarray,
    occupied: np.ndarray,
    first_bbox: tuple[int, int, int, int],
    second_bbox: tuple[int, int, int, int],
) -> tuple[int, int] | None:
    h, w = source_patch.shape
    candidates = []
    for r0 in range(SOURCE_BORDER, H - h - SOURCE_BORDER + 1):
        for c0 in range(SOURCE_BORDER, W - w - SOURCE_BORDER + 1):
            bbox = (int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1))
            if any(bool(occupied[int(r0 + rr), int(c0 + cc)]) for rr, cc in np.argwhere(source_mask)):
                continue
            d1 = _bbox_gap(bbox, first_bbox)
            d2 = _bbox_gap(bbox, second_bbox)
            if d1 + SOURCE_GAP_MIN > d2:
                continue
            if d1 < 3:
                continue
            candidates.append(((int(d2 - d1), -int(d1)), (int(r0), int(c0))))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    best_score = candidates[0][0]
    best = [pos for score, pos in candidates if score == best_score]
    return random.choice(best)


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    bg = random.choice(list(range(10)))
    colors = [c for c in range(10) if c != bg]
    random.shuffle(colors)
    frame_color, color_a, color_b = colors[:3]

    spec_a, spec_b = _pick_specs()
    shape_a = _cavity_shape_from_spec(spec_a)
    shape_b = _cavity_shape_from_spec(spec_b)

    layout = random.choice(["horizontal", "vertical"])
    if layout == "horizontal":
        sides = ("left", "right")
    else:
        sides = ("top", "bottom")
    base_patch, base_mask = _build_base_tool(layout, shape_a, shape_b, color_a, color_b, bg)

    input_grid = np.full((H, W), int(bg), dtype=int)
    frame1 = _orient_left_to_side(_build_left_frame_array(bg, frame_color, spec_a["fw"], spec_a["top_h"], spec_a["bot_h"], spec_a["notch_n"]), sides[0])
    frame2 = _orient_left_to_side(_build_left_frame_array(bg, frame_color, spec_b["fw"], spec_b["top_h"], spec_b["bot_h"], spec_b["notch_n"]), sides[1])
    if layout == "horizontal":
        shared_r = random.randint(3, H - max(frame1.shape[0], frame2.shape[0]) - 4)
        input_grid[shared_r : shared_r + frame1.shape[0], 0 : frame1.shape[1]][frame1 != int(bg)] = frame1[frame1 != int(bg)]
        input_grid[shared_r : shared_r + frame2.shape[0], W - frame2.shape[1] : W][frame2 != int(bg)] = frame2[frame2 != int(bg)]
    else:
        shared_c = random.randint(3, W - max(frame1.shape[1], frame2.shape[1]) - 4)
        input_grid[0 : frame1.shape[0], shared_c : shared_c + frame1.shape[1]][frame1 != int(bg)] = frame1[frame1 != int(bg)]
        input_grid[H - frame2.shape[0] : H, shared_c : shared_c + frame2.shape[1]][frame2 != int(bg)] = frame2[frame2 != int(bg)]

    fixed_mask, frames = _extract_frames(input_grid, bg)
    if len(frames) != 2:
        return None
    first = [fr for fr in frames if fr["side"] == sides[0]][0]
    second = [fr for fr in frames if fr["side"] == sides[1]][0]

    occ_frames = fixed_mask.copy()
    try:
        stage1 = _dock_into_frame(base_patch, base_mask, first, occ_frames)
    except AssertionError:
        return None
    canvas_after_stage1 = _stamp(input_grid.copy(), stage1["patch"], stage1["mask"], stage1["top"], stage1["left"])
    occ1 = occ_frames.copy()
    for rr, cc in np.argwhere(stage1["mask"]):
        occ1[int(stage1["top"] + rr), int(stage1["left"] + cc)] = True
    try:
        stage2 = _dock_into_frame(stage1["patch"], stage1["mask"], second, occ1)
    except AssertionError:
        return None
    output_grid = _stamp(canvas_after_stage1, stage2["patch"], stage2["mask"], stage2["top"], stage2["left"])

    occ_output = occ1.copy()
    for rr, cc in np.argwhere(stage2["mask"]):
        occ_output[int(stage2["top"] + rr), int(stage2["left"] + cc)] = True

    source_patch, source_mask = _random_tool_source_variant(base_patch, base_mask)
    pos = _place_source(H, W, bg, source_patch, source_mask, occ_output, first["bbox"], second["bbox"])
    if pos is None:
        return None
    sr, sc = pos
    input_with_source = _stamp(input_grid, source_patch, source_mask, sr, sc)
    output_with_source = _stamp(output_grid, source_patch, source_mask, sr, sc)

    try:
        solved = main(input_with_source)
    except Exception:
        return None
    if not np.array_equal(solved, output_with_source):
        return None
    return input_with_source


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPT_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPT_RETRIES} attempts")
