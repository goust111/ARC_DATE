# Combo variant of a25697e4 (index 11)
# combo_spec: D9/1+D10/1
# Name: 单工具多框架分离输出
#
# Design analysis:
# a) Concrete behavior trace:
#    - Original task: each detached 2-color tool is matched to exactly one border frame, docked back
#      into the original canvas, and the output keeps the global input size.
#    - Fused task: there is exactly one detached tool, but 2-3 border frames all share the same
#      staircase cavity family. The tool is conceptually copied and docked into every compatible
#      frame. Each docked result is then cropped to its own tight bbox and the crops are stacked as
#      separate panels with divider rows.
#    - Example: if the input has one free tool and three border frames on left/top/right, the output
#      is not one large scene with three insertions. It is three small docked subgrids, each showing
#      that same tool fitted to one frame, ordered by the frame positions in reading order.
#    - The phrase "各框架凹槽不同" is realized as visually different orientations of the same cavity
#      class. This keeps D9/1 valid: one tool matches many frames, while the frames still look
#      different because they sit on different borders and require different D4 transforms.
# b) Complexity calibration:
#    - The geometry stays close to the source task: 2 same-color frame pieces per frame, cavity area
#      5-8, one 2-color connected tool, handle area different from plug area, and optional
#      reflection-triggered color swap.
#    - To keep the multi-frame relation obvious, every sample has 2 or 3 frames but only 1 movable
#      tool. All frames are compatible with the same plug shape, and frame colors are distinct so the
#      separate panels remain easy to read.
# c) Hidden-assumption audit against the original solver:
#    - Original assumption "number of tools ~= number of frames" breaks. The new solver must gather
#      all compatible frames for the single tool instead of solving a permutation assignment.
#    - Original assumption "output lives on the original canvas" breaks. The new solver must render
#      local docked panels and pack them into a new vertically separated grid.
#    - Original placement search used global collision checks between different tools. That no longer
#      applies because each output panel is independent and contains only one frame copy plus one tool
#      copy.
#    - Open direction is still derived from border anchoring, but now only for deciding how the local
#      panel is rendered; the original global placement score is unnecessary.
# d) Placement legality:
#    - Generator places each frame flush to exactly one border, rejects corner-touching placements on
#      other borders, enforces bbox gaps between frames, and places the free tool via
#      `random_free_location` with padding so it stays detached and non-overlapping.
# e) Visual observability:
#    - Humans can see the fused rule from a few examples because the input always contains one tool
#      versus multiple matching frames, while the output clearly splits into the same number of docked
#      panels. This is a visible cardinality change plus a visible output-layout change.
# f) Visual cleanliness:
#    - Using vertical panel packing with a single solid divider row avoids clutter. Each panel is
#      independently cropped, so no large empty canvas remains.
# g) New solver assumptions and generator guarantees:
#    - `main()` assumes there is exactly one detached 2-color tool and at least two border frames,
#      each with one cavity that is D4-congruent to one tool color-region. `generate_input()` builds
#      samples directly to satisfy that.
#    - `main()` assumes there is at least one unused palette color available for panel dividers.
#      The generator leaves such a color unused by keeping the visible palette small.

from __future__ import annotations

from collections import Counter
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, bbox as _bbox, get_bg, paste, random_free_location, random_sprite


GRID_H_RANGE = (22, 30)
GRID_W_RANGE = (25, 30)
NUM_FRAMES_CHOICES = [2, 3]

FRAME_W_RANGE = (4, 8)
NOTCH_N_RANGE = (1, 4)
TOP_H_RANGE = (1, 4)
BOT_H_RANGE = (2, 4)
PLUG_AREA_BOUNDS = (5, 8)

HANDLE_AREA_RANGE = (4, 10)
TOOL_TOTAL_BOUNDS = (10, 15)
HANDLE_SPRITE_DIM = (2, 5)

FRAME_BBOX_GAP_MIN = 3
TOOL_BORDER = 1
TOOL_PADDING = 4
TOOL_PADDING_CONN = 4

SPEC_RETRIES = 300
HANDLE_SPRITE_RETRIES = 30
GEN_ATTEMPT_RETRIES = 300
DIVIDER_THICKNESS = 1

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


def _transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.rot90(arr, int(k))
    if flip:
        out = np.fliplr(out)
    return out


def _swap_colors(patch: np.ndarray, a: int, b: int) -> np.ndarray:
    out = patch.copy()
    ma = out == int(a)
    mb = out == int(b)
    out[ma] = int(b)
    out[mb] = int(a)
    return out


def _crop_to_mask(arr: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return (
        arr[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy(),
        mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy(),
    )


def _crop_non_bg(arr: np.ndarray, bg: int) -> np.ndarray:
    b = _bbox(arr != int(bg))
    assert b is not None
    r0, c0, r1, c1 = b
    return arr[r0 : r1 + 1, c0 : c1 + 1].copy()


def _normalize_shape(coords) -> frozenset[tuple[int, int]]:
    pts = [(int(r), int(c)) for r, c in coords]
    r0 = min(r for r, _ in pts)
    c0 = min(c for _, c in pts)
    return frozenset((int(r - r0), int(c - c0)) for r, c in pts)


def _rot90_shape(shape: frozenset[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    return _normalize_shape((c, -r) for r, c in shape)


def _flip_shape(shape: frozenset[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    return _normalize_shape((r, -c) for r, c in shape)


def _dihedral_shapes(shape: frozenset[tuple[int, int]]) -> set[frozenset[tuple[int, int]]]:
    out: set[frozenset[tuple[int, int]]] = set()
    cur = _normalize_shape(shape)
    for _ in range(4):
        out.add(cur)
        out.add(_flip_shape(cur))
        cur = _rot90_shape(cur)
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
        pp, mm = _crop_to_mask(p, m)
        key = (mm.tobytes(), pp[mm].tobytes())
        if key in seen:
            continue
        seen.add(key)
        uniq.append((pp, mm, is_flip))
    return uniq


def _variant_choices(patch: np.ndarray, bg: int) -> list[tuple[np.ndarray, bool]]:
    mask = patch != int(bg)
    out: list[tuple[np.ndarray, bool]] = []
    seen = set()
    for k in range(4):
        for flip in (False, True):
            q = _transform(patch, k, flip)
            m = _transform(mask.astype(np.uint8), k, flip).astype(bool)
            qq, mm = _crop_to_mask(q, m)
            key = (mm.tobytes(), qq[mm].tobytes())
            if key in seen:
                continue
            seen.add(key)
            out.append((qq, bool(flip)))
    random.shuffle(out)
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
    dims = [(h, w) for h in range(lo, hi) for w in range(lo, hi) if handle_area <= h * w <= handle_area * 2]
    if not dims:
        return None
    for _ in range(HANDLE_SPRITE_RETRIES):
        h, w = random.choice(dims)
        sprite = random_sprite(
            h,
            w,
            density=handle_area / (h * w),
            color_palette=[int(handle_color)],
            connectivity=4,
            background=int(bg),
        )
        if int(np.sum(sprite != int(bg))) == int(handle_area):
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
    frame = np.full((fh, fw), int(bg), dtype=int)
    frame[:top_h, :] = int(frame_color)
    frame[top_h + 1 :, :] = int(frame_color)
    for dr, dc in _notch_offsets(notch_n, fw, bot_h):
        frame[top_h + 1 + dr, dc] = int(bg)

    cavity = frame == int(bg)
    attach_rows = {int(r) for r in range(fh) if cavity[r, fw - 1]}

    handle_sprite = _make_handle_sprite(handle_area, bg, handle_color)
    if handle_sprite is None:
        return None

    hs_h, hs_w = handle_sprite.shape
    sprite_left_rows = {int(r) for r in range(hs_h) if handle_sprite[r, 0] != int(bg)}
    valid_offsets = [voff for voff in range(fh) if any((voff + sr) in attach_rows for sr in sprite_left_rows)]
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


def _pick_scene_spec() -> dict | None:
    ha_lo, ha_hi = HANDLE_AREA_RANGE
    tt_lo, tt_hi = TOOL_TOTAL_BOUNDS
    for _ in range(SPEC_RETRIES):
        fw, top_h, bot_h, notch_n = random.choice(PARAM_OPTIONS)
        plug_area = fw + notch_n
        handle_choices = [h for h in range(ha_lo, ha_hi) if tt_lo <= plug_area + h <= tt_hi and h != plug_area]
        if not handle_choices:
            continue
        return {
            "fw": fw,
            "top_h": top_h,
            "bot_h": bot_h,
            "notch_n": notch_n,
            "plug_area": plug_area,
            "handle_area": random.choice(handle_choices),
        }
    return None


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


def _place_frame_patch(
    canvas: np.ndarray,
    frame_patch: np.ndarray,
    side: str,
    bg: int,
    frame_bboxes: list[tuple[int, int, int, int]],
) -> tuple[int, int] | None:
    H, W = canvas.shape
    ph, pw = frame_patch.shape
    for r0, c0 in _candidate_origins(side, ph, pw, H, W):
        bbox = (r0, c0, r0 + ph - 1, c0 + pw - 1)
        if side in ("left", "right") and (bbox[0] == 0 or bbox[2] == H - 1):
            continue
        if side in ("top", "bottom") and (bbox[1] == 0 or bbox[3] == W - 1):
            continue
        if any(_bbox_gap(bbox, prev) <= FRAME_BBOX_GAP_MIN - 1 for prev in frame_bboxes):
            continue
        if np.any((canvas[r0 : r0 + ph, c0 : c0 + pw] != int(bg)) & (frame_patch != int(bg))):
            continue
        paste(canvas, frame_patch, r0, c0, transparent=int(bg))
        frame_bboxes.append(bbox)
        return int(r0), int(c0)
    return None


def _choose_divider_color(grid: np.ndarray, bg: int) -> int:
    used = set(map(int, grid.ravel()))
    for col in range(10):
        if col != int(bg) and col not in used:
            return int(col)
    raise AssertionError("expected an unused divider color")


def _pack_panels_vertically(panels: list[np.ndarray], bg: int, divider_color: int) -> np.ndarray:
    assert panels
    max_w = max(int(p.shape[1]) for p in panels)
    total_h = sum(int(p.shape[0]) for p in panels) + DIVIDER_THICKNESS * (len(panels) - 1)
    out = np.full((total_h, max_w), int(bg), dtype=int)
    r = 0
    for i, panel in enumerate(panels):
        h, w = panel.shape
        c0 = (max_w - w) // 2
        paste(out, panel, r, c0, transparent=None)
        r += h
        if i + 1 < len(panels):
            out[r : r + DIVIDER_THICKNESS, :] = int(divider_color)
            r += DIVIDER_THICKNESS
    return out


def _count_divider_rows(grid: np.ndarray, bg: int) -> int:
    count = 0
    for r in range(grid.shape[0]):
        vals = {int(x) for x in grid[r, :]}
        if len(vals) == 1:
            col = next(iter(vals))
            if col != int(bg):
                count += 1
    return int(count)


def _extract_frame_groups(g: np.ndarray, bg: int) -> tuple[list[dict], np.ndarray]:
    H, W = g.shape
    fg = g != int(bg)
    lab, n = label(fg, structure=STRUCT4)
    fixed_mask = np.zeros((H, W), dtype=bool)
    fixed_components: list[dict] = []
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
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

    assert fixed_components, "expected border frames"

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

    grouped: dict[int, list[int]] = {}
    for i in range(len(fixed_components)):
        grouped.setdefault(find(i), []).append(i)

    groups: list[dict] = []
    for idxs in grouped.values():
        pts = [p for i in idxs for p in fixed_components[i]["pts"]]
        rs = [r for r, _ in pts]
        cs = [c for _, c in pts]
        bbox = (min(rs), min(cs), max(rs), max(cs))

        mask = np.zeros((H, W), dtype=bool)
        for r, c in pts:
            mask[r, c] = True

        sub_bg = g[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1] == int(bg)
        sub_lab, sub_n = label(sub_bg, structure=STRUCT4)
        best_id = None
        best_key = None
        for cid in range(1, int(sub_n) + 1):
            ptsc = np.argwhere(sub_lab == int(cid))
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
            key = (area, -cid)
            if best_key is None or key > best_key:
                best_key = key
                best_id = int(cid)
        assert best_id is not None, "expected a frame cavity"

        cavity_local = sub_lab == int(best_id)
        frame_patch = g[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1].copy()
        frame_mask_local = mask[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1].copy()
        cavity_coords = {(int(r), int(c)) for r, c in np.argwhere(cavity_local)}
        cavity_shape = _normalize_shape(cavity_coords)
        border_hits = []
        fh, fw = frame_patch.shape
        if bbox[1] == 0:
            border_hits.append((0, 1))
        if bbox[3] == W - 1:
            border_hits.append((0, -1))
        if bbox[0] == 0:
            border_hits.append((1, 0))
        if bbox[2] == H - 1:
            border_hits.append((-1, 0))
        assert len(border_hits) == 1, "each frame must touch exactly one border"

        groups.append(
            {
                "bbox": bbox,
                "frame_patch": frame_patch,
                "frame_mask_local": frame_mask_local,
                "cavity_local": cavity_local,
                "cavity_coords": cavity_coords,
                "cavity_shape": cavity_shape,
                "cavity_size": int(len(cavity_coords)),
                "open_dir": border_hits[0],
            }
        )
    groups.sort(key=lambda d: (d["bbox"][0], d["bbox"][1]))
    return groups, fixed_mask


def _extract_tool(g: np.ndarray, bg: int, fixed_mask: np.ndarray) -> dict:
    mov_mask = (g != int(bg)) & (~fixed_mask)
    mov_lab, mov_n = label(mov_mask, structure=STRUCT4)
    mov_comps: list[dict] = []
    for cid in range(1, int(mov_n) + 1):
        pts = np.argwhere(mov_lab == int(cid))
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        mask = mov_lab[r0 : r1 + 1, c0 : c1 + 1] == int(cid)
        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        patch[~mask] = 0
        vals = [int(x) for x in patch[mask].flatten() if int(x) != 0]
        counts = Counter(vals)
        mov_comps.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "mask": mask,
                "patch": patch,
                "counts": counts,
                "size": int(len(pts)),
            }
        )
    assert len(mov_comps) == 1, "fused variant expects exactly one free tool"
    comp = mov_comps[0]
    assert len(comp["counts"]) == 2, "tool must have exactly two colors"
    return comp


def _outside_open_dir(r: int, c: int, fh: int, fw: int, open_dir: tuple[int, int]) -> bool:
    dr, dc = open_dir
    if (dr, dc) == (0, 1):
        return c >= fw
    if (dr, dc) == (0, -1):
        return c < 0
    if (dr, dc) == (1, 0):
        return r >= fh
    if (dr, dc) == (-1, 0):
        return r < 0
    raise ValueError(open_dir)


def _find_docking_state(tool: dict, frame: dict):
    patch = tool["patch"]
    mask = tool["mask"]
    fh, fw = frame["frame_patch"].shape
    cavity_shape = frame["cavity_shape"]
    cavity_shapes = _dihedral_shapes(cavity_shape)
    cavity_coords = frame["cavity_coords"]
    cavity_size = int(frame["cavity_size"])
    frame_mask_local = frame["frame_mask_local"]
    cav_min_r = min(r for r, _ in cavity_coords)
    cav_min_c = min(c for _, c in cavity_coords)

    best_state = None
    best_score = None
    for p, m, is_flip in _gen_variants(patch, mask):
        colors_here = sorted({int(x) for x in p[m].flatten() if int(x) != 0})
        for plug_color in colors_here:
            plug_mask = (p == int(plug_color)) & m
            if int(plug_mask.sum()) != cavity_size:
                continue
            plug_coords = [(int(r), int(c)) for r, c in np.argwhere(plug_mask)]
            plug_shape = _normalize_shape(plug_coords)
            if plug_shape not in cavity_shapes:
                continue
            plug_min_r = min(r for r, _ in plug_coords)
            plug_min_c = min(c for _, c in plug_coords)
            shift_r = cav_min_r - plug_min_r
            shift_c = cav_min_c - plug_min_c

            translated_plug = {(int(r + shift_r), int(c + shift_c)) for r, c in plug_coords}
            if translated_plug != cavity_coords:
                continue

            other_pixels: list[tuple[int, int, int]] = []
            ok = True
            for rr, cc in np.argwhere(m):
                r = int(rr + shift_r)
                c = int(cc + shift_c)
                col = int(p[rr, cc])
                if col == int(plug_color):
                    continue
                if (r, c) in cavity_coords:
                    ok = False
                    break
                if 0 <= r < fh and 0 <= c < fw and bool(frame_mask_local[r, c]):
                    ok = False
                    break
                if not _outside_open_dir(r, c, fh, fw, frame["open_dir"]):
                    ok = False
                    break
                other_pixels.append((r, c, col))
            if not ok:
                continue

            contact = 0
            for r, c, _col in other_pixels:
                for dr, dc in DIRS4:
                    if (r + dr, c + dc) in cavity_coords:
                        contact += 1
            score = (int(contact), -int(is_flip), -abs(shift_r), -abs(shift_c))
            if best_score is None or score > best_score:
                best_score = score
                best_state = (p.copy(), m.copy(), int(shift_r), int(shift_c), bool(is_flip))
    return best_state


def _dock_panel(tool: dict, frame: dict, bg: int, dock_state=None) -> np.ndarray:
    counts = tool["counts"]
    best_state = _find_docking_state(tool, frame) if dock_state is None else dock_state
    assert best_state is not None, "expected a valid docking transform for every frame"
    fh, fw = frame["frame_patch"].shape
    p, m, shift_r, shift_c, is_flip = best_state
    if is_flip and len(counts) == 2 and len(set(counts.values())) == 2:
        a, b = sorted(int(x) for x in counts.keys())
        p = _swap_colors(p, a, b)

    placed_coords = [(int(rr + shift_r), int(cc + shift_c), int(p[rr, cc])) for rr, cc in np.argwhere(m)]
    min_r = min([0] + [r for r, _, _ in placed_coords])
    min_c = min([0] + [c for _, c, _ in placed_coords])
    max_r = max([fh - 1] + [r for r, _, _ in placed_coords])
    max_c = max([fw - 1] + [c for _, c, _ in placed_coords])
    out = np.full((max_r - min_r + 1, max_c - min_c + 1), int(bg), dtype=int)
    paste(out, frame["frame_patch"], -min_r, -min_c, transparent=int(bg))
    for r, c, col in placed_coords:
        out[int(r - min_r), int(c - min_c)] = int(col)
    return out


def _build_expected_output(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    frames, fixed_mask = _extract_frame_groups(g, bg)
    tool = _extract_tool(g, bg, fixed_mask)
    dockable = []
    for frame in frames:
        dock_state = _find_docking_state(tool, frame)
        if dock_state is not None:
            dockable.append((frame, dock_state))
    assert len(dockable) >= 2, "expected one tool docking into multiple frames"
    panels = [_dock_panel(tool, frame, bg, dock_state=dock_state) for frame, dock_state in dockable]
    divider_color = _choose_divider_color(g, bg)
    return _pack_panels_vertically(panels, bg, divider_color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy one detached two-color tool into every compatible border frame and emit separate panels.

    Input structure:
    - Background is the most common color.
    - Border-touching foreground components form 2-3 anchor frames, each made from nearby
      same-color pieces whose union encloses one staircase cavity.
    - Exactly one non-border connected component is a movable 2-color tool; one of its color
      regions is D4-congruent to every frame cavity, while the other region is the handle.

    Rule:
    - For each compatible frame, independently rotate/flip the single tool so one color region
      fills that frame's cavity exactly and the other region protrudes away from the touched border.
    - If the chosen docking transform uses a reflection and the two tool colors have different
      pixel counts, swap those two colors in that panel's docked copy.
    - Crop each docked result to its own tight bbox and stack the panels vertically with a solid
      divider row between neighboring panels.

    Key insight: The free tool is not consumed by one frame; it must be replicated across all
    matching frames, and the true output space is the list of local docked scenes rather than the
    original global canvas.
    """
    return _build_expected_output(grid)


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    bg = random.choice(list(range(10)))
    num_frames = random.choice(NUM_FRAMES_CHOICES)

    spec = _pick_scene_spec()
    if spec is None:
        return None

    colors = [c for c in range(10) if c != bg]
    random.shuffle(colors)
    need = num_frames + 3  # frame colors + plug + handle + divider reserve
    if len(colors) < need:
        return None

    frame_colors = colors[:num_frames]
    plug_color = colors[num_frames]
    handle_color = colors[num_frames + 1]
    result = _build_left_pair(
        bg=bg,
        frame_color=frame_colors[0],
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
    _frame_left, tool_left = result
    tool_base = _crop_non_bg(tool_left, bg)

    input_grid = np.full((H, W), int(bg), dtype=int)
    frame_bboxes: list[tuple[int, int, int, int]] = []

    sides = random.sample(["left", "right", "top", "bottom"], k=num_frames)
    for frame_color, side in zip(frame_colors, sides):
        built = _build_left_pair(
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
        if built is None:
            return None
        frame_scene_left, _tool_scene_left = built
        frame_patch = _crop_non_bg(_orient_left_to_side(frame_scene_left, side), bg)

        origin = _place_frame_patch(input_grid, frame_patch, side, bg, frame_bboxes)
        if origin is None:
            return None

    placement_canvas = input_grid.copy()
    placed_tool = False
    for tool_patch, is_flip in _variant_choices(tool_base, bg):
        patch = tool_patch.copy()
        if is_flip and spec["plug_area"] != spec["handle_area"]:
            patch = _swap_colors(patch, plug_color, handle_color)
        try:
            r0, c0 = random_free_location(
                placement_canvas,
                patch,
                transparent=int(bg),
                border=TOOL_BORDER,
                padding=TOOL_PADDING,
                padding_connectivity=TOOL_PADDING_CONN,
            )
        except ValueError:
            continue
        paste(input_grid, patch, r0, c0, transparent=int(bg))
        paste(placement_canvas, patch, r0, c0, transparent=int(bg))
        placed_tool = True
        break
    if not placed_tool:
        return None

    try:
        actual = main(input_grid)
    except Exception:
        return None

    frame_groups, _ = _extract_frame_groups(input_grid, bg)
    if len(frame_groups) != num_frames:
        return None
    if _count_divider_rows(actual, bg) != num_frames - 1:
        return None
    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPT_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPT_RETRIES} attempts")
