# Combo variant of a25697e4 (index 25)
# combo_spec: D7/3+D10/0
# Name: 裁剪输出工具提取
#
# Design analysis
# a) 具体化与行为追踪:
#    - 原题: 输入里有边框框架和远处的双色工具; 输出把工具对接进框架, 保留框架, 输出尺寸不变。
#    - 本融合: 对接动作本身不变, 仍靠 cavity 像素数匹配 plug 区, 仍允许 D4 旋转/翻转, 仍保留“翻转且两色面积不等时交换两色”。
#      变化发生在对接后的显示阶段: 框架全部擦除, 只留下完成对接后的工具形态; 然后把整张图 tight-crop 到所有非背景像素的最小包围盒。
#    - 小例子: 若左边框架的凹槽面积为 6, 远处工具颜色 A 有 6 像素、颜色 B 有 5 像素, solver 先把 A 对准凹槽、B 伸出框外。
#      原题输出会看到“框架 + A/B 工具”; 本题输出只保留变换后的 A/B 工具, 框架像素全部消失, 并裁掉外围空白。
# b) 复杂度校准:
#    - 若直接保留原 generator 的边缘分散摆放, 两个 docked 工具 tight-crop 后仍可能横跨整张画布, D10/0 的视觉差异会变弱。
#    - 因此生成器在 2-pair 时把框架收拢到同一角附近的相邻两条边, 保持“可区分的多对象”同时让裁剪后的输出明显变小。
#    - plug_area 继续强制互异, 避免多工具多框架时匹配歧义。
# c) 隐式假设审查:
#    - 原 solver 假设输出先复制 fixed_mask 上的框架像素; 本变体不成立, 所以 out 改成纯背景, 仅在找到 docking 位置后绘制工具像素。
#    - 原 solver 的 placed_global=fixed_mask 仍然成立, 因为它约束的是“工具不能覆盖框架实体”, 而不是“框架必须可见”。
#    - 原 generator 用 output=scene 作为最终标签; 本变体不成立, 必须拆成 docked_scene(占位/避碰) 与 tool_only_output(监督标签)。
#    - 原 generator 只校验 main(input)==same-shape output; 本变体新增 tight-crop, 所以监督输出必须先裁剪后再比较。
# d) 放置合法性:
#    - 框架仍沿边放置, 互不重叠, frame_bbox 不触及第二条边。
#    - docked_scene 用于约束框架和已 docked 工具不重叠; undocked 工具再通过 random_free_location 放到远离现有内容的位置。
#    - tool_only_output 只负责记录可见结果, 不参与输入合法性。
# e) 视觉可观察性:
#    - 人类能直接看到: 输入有框架和远处工具; 输出里框架没了, 只剩 docked 后的工具, 而且图像被裁小。
#    - 颜色交换不是凭颜色值决定, 而是跟“是否使用翻转且两区面积不等”这一几何条件绑定, 样例中会通过形状对接关系提供线索。
# f) 可视化整洁度:
#    - 输出没有框架层, 视觉更干净; 2-pair 时又限制在同角附近, 避免 crop 后留下大块难解释空白。
# g) 变化的可推断性:
#    - 颜色变化仍只发生在翻转情形且仅限双色工具, 颜色数少且有 plug/handle 分区结构支撑。
#    - “擦除框架 + 紧凑裁剪”是所有样例都稳定出现的全局规律, 不依赖任何不可视属性。

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np
from scipy.ndimage import label

ROOT_DIR = Path(__file__).resolve().parents[4]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from arc_utils import STRUCT4, bbox as _bbox, get_bg, paste, random_free_location, random_sprite


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


def _tight_crop(grid: np.ndarray, bg: int) -> np.ndarray:
    nz = np.argwhere(grid != int(bg))
    assert nz.size > 0, "tight crop requires at least one non-background pixel"
    r0, c0 = nz.min(axis=0)
    r1, c1 = nz.max(axis=0)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


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
    raise ValueError(notch_n)


def _build_left_pair(
    *, bg: int, frame_color: int, plug_color: int, handle_color: int,
    fw: int, top_h: int, bot_h: int, notch_n: int, handle_area: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    fh = top_h + 1 + bot_h
    frame = np.full((fh, fw), bg, dtype=int)
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
    valid_offsets = [
        voff
        for voff in range(fh)
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
    tool[:fh, :fw][cavity] = int(plug_color)
    handle_mask = handle_sprite != int(bg)
    tool[voff : voff + hs_h, fw : fw + hs_w][handle_mask] = int(handle_color)
    return frame_full, tool


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
            color_palette=[int(handle_color)],
            connectivity=4,
            background=int(bg),
        )
        if int(np.sum(sprite != int(bg))) == int(handle_area):
            return sprite
    return None


def _variant_choices(patch: np.ndarray) -> list[tuple[np.ndarray, bool]]:
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
            out.append((q, bool(flip)))
    random.shuffle(out)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock each loose two-color tool into a border frame, then keep only the tool silhouettes.

    Input structure:
    - `bg` is the most frequent color.
    - Border-touching same-color pieces cluster into anchor frames with staircase cavities.
    - Interior 2-color connected components are movable tools whose two color-region sizes identify a target cavity.

    Rule:
    - Pair every tool with a frame whose cavity size matches exactly one of the tool's two color-region areas.
    - Search over rotations/reflections and placements so one color region plugs the cavity exactly while the other protrudes outward.
    - If the winning docking transform uses a reflection and the two color-region sizes are unequal, swap the two tool colors.
    - After docking, erase the frame entirely, keep only the transformed tool pixels, and tight-crop the result to the minimal non-background bbox.

    Key insight: The cavity area still determines which tool docks where; the new twist is that the frame is only a hidden alignment scaffold, not part of the final visible output.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    fg = g != int(bg)
    lab, n = label(fg, structure=STRUCT4)
    fixed_mask = np.zeros((H, W), dtype=bool)
    fixed_components = []
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
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

    assert fixed_components, "expected at least one border frame"

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

    anchor_groups = []
    for idxs in groups.values():
        pts = [p for i in idxs for p in fixed_components[i]["pts"]]
        rs = [r for r, _c in pts]
        cs = [c for _r, c in pts]
        bbox = (min(rs), min(cs), max(rs), max(cs))
        mask = np.zeros((H, W), dtype=bool)
        for r, c in pts:
            mask[r, c] = True

        socket_mask = np.zeros((H, W), dtype=bool)
        for r in range(bbox[0], bbox[2] + 1):
            for c in range(bbox[1], bbox[3] + 1):
                if int(g[r, c]) != int(bg):
                    continue
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < H and 0 <= cc < W and mask[rr, cc]:
                        socket_mask[r, c] = True
                        break

        cavity = np.zeros((H, W), dtype=bool)
        sub_bg = g[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1] == int(bg)
        sub_lab, sub_n = label(sub_bg, structure=STRUCT4)
        best_choice = None
        best_id = None
        for cid in range(1, int(sub_n) + 1):
            ptsc = np.argwhere(sub_lab == int(cid))
            area = int(len(ptsc))
            adjacent = False
            for rr, cc in ptsc:
                r = int(bbox[0] + rr)
                c = int(bbox[1] + cc)
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    r2, c2 = r + dr, c + dc
                    if bbox[0] <= r2 <= bbox[2] and bbox[1] <= c2 <= bbox[3] and mask[r2, c2]:
                        adjacent = True
                        break
                if adjacent:
                    break
            if not adjacent:
                continue
            choice = (int(area), -int(cid))
            if best_choice is None or choice > best_choice:
                best_choice = choice
                best_id = int(cid)
        if best_id is not None:
            for rr, cc in np.argwhere(sub_lab == int(best_id)):
                cavity[int(bbox[0] + rr), int(bbox[1] + cc)] = True

        dir_list = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        dir_counts = [0, 0, 0, 0]
        for r, c in pts:
            for di, (dr, dc) in enumerate(dir_list):
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(bg):
                    dir_counts[di] += 1
        open_dir = dir_list[int(np.argmax(dir_counts))]

        anchor_groups.append(
            {
                "mask": mask,
                "bbox": bbox,
                "socket_mask": socket_mask,
                "cavity_mask": cavity,
                "open_dir": open_dir,
                "cavity_size": int(cavity.sum()),
                "size": int(mask.sum()),
            }
        )
    anchor_groups.sort(key=lambda d: -int(d["size"]))

    mov_mask = fg & (~fixed_mask)
    mov_lab, mov_n = label(mov_mask, structure=STRUCT4)
    mov_comps = []
    for cid in range(1, int(mov_n) + 1):
        pts = np.argwhere(mov_lab == int(cid))
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        mask = mov_lab[r0 : r1 + 1, c0 : c1 + 1] == int(cid)
        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        patch[~mask] = 0
        mov_comps.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "mask": mask,
                "patch": patch,
                "size": int(len(pts)),
            }
        )
    mov_comps.sort(key=lambda d: -int(d["size"]))

    def choose_group(bbox) -> int:
        best_i = 0
        best_d = None
        for i, grp in enumerate(anchor_groups):
            dist = _bbox_gap(bbox, grp["bbox"])
            if best_d is None or dist < best_d:
                best_d = dist
                best_i = i
        return int(best_i)

    comp_color_counts: list[dict[int, int]] = []
    for comp in mov_comps:
        vals = [int(x) for x in comp["patch"][comp["mask"]].flatten() if int(x) != 0]
        comp_color_counts.append(dict(Counter(vals)))

    def compatible(ci: int, gi: int) -> bool:
        cav_sz = int(anchor_groups[gi]["cavity_size"])
        return any(int(v) == cav_sz for v in comp_color_counts[ci].values())

    group_for_comp: list[int]
    if len(mov_comps) == len(anchor_groups) and len(anchor_groups) > 1:
        import itertools

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
        assert best_perm is not None, "expected a compatible tool-to-frame assignment"
        group_for_comp = [int(x) for x in best_perm]
    else:
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
            group_for_comp.append(int(best_g) if best_g is not None else int(choose_group(comp["bbox"])))

    out = np.full_like(g, int(bg))
    placed_global = fixed_mask.copy()
    placed_groups = [grp["mask"].copy() for grp in anchor_groups]

    for comp, gi in zip(mov_comps, group_for_comp):
        grp = anchor_groups[gi]
        placed_grp = placed_groups[gi]
        socket_mask = grp["socket_mask"]
        cavity_mask = grp["cavity_mask"]
        cav_sz = int(grp["cavity_size"])
        odr, odc = grp["open_dir"]
        comp_vals = [int(x) for x in comp["patch"][comp["mask"]].flatten() if int(x) != 0]
        comp_counts = Counter(comp_vals)

        best_state = None
        best_score = None
        for p, m, is_flip in _gen_variants(comp["patch"], comp["mask"]):
            h, w = m.shape
            coords = np.argwhere(m)
            cols_here = sorted({int(x) for x in p[m].flatten() if int(x) != 0})
            plug_colors = [int(col) for col in cols_here if int(((p == int(col)) & m).sum()) == cav_sz]
            assert plug_colors, "expected one color region to match cavity size"
            for r0 in range(0, H - h + 1):
                for c0 in range(0, W - w + 1):
                    if any(placed_global[int(r0 + rr), int(c0 + cc)] for rr, cc in coords):
                        continue

                    adj = 0
                    socket_hits = 0
                    cavity_hits = 0
                    outside = 0
                    br0, bc0, br1, bc1 = grp["bbox"]
                    for rr, cc in coords:
                        r = int(r0 + rr)
                        c = int(c0 + cc)
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            r2, c2 = r + dr, c + dc
                            if 0 <= r2 < H and 0 <= c2 < W and placed_grp[r2, c2]:
                                adj += 1
                        if socket_mask[r, c]:
                            socket_hits += 1
                        if cavity_mask[r, c]:
                            cavity_hits += 1
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
                    for plug_col in plug_colors:
                        plug_hits = 0
                        plug_outside = 0
                        nonplug_in_cavity = 0
                        for rr, cc in coords:
                            r = int(r0 + rr)
                            c = int(c0 + cc)
                            in_cavity = bool(cavity_mask[r, c])
                            col = int(p[rr, cc])
                            if col == int(plug_col):
                                if in_cavity:
                                    plug_hits += 1
                                else:
                                    plug_outside += 1
                            elif in_cavity:
                                nonplug_in_cavity += 1
                        plug_full = int(
                            plug_hits == cav_sz and plug_outside == 0 and nonplug_in_cavity == 0
                        )
                        plug_score = (
                            int(plug_full),
                            int(plug_hits),
                            -int(nonplug_in_cavity),
                            -int(plug_outside),
                        )
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
                        best_state = (p, m, int(r0), int(c0), bool(is_flip))

        assert best_state is not None, "expected a valid docking placement for every tool"
        p, m, r0, c0, is_flip = best_state
        if is_flip and len(comp_counts) == 2 and len(set(comp_counts.values())) == 2:
            a, b = sorted(comp_counts.keys())
            p = _swap_colors(p, a, b)
        for rr, cc in np.argwhere(m):
            r = int(r0 + rr)
            c = int(c0 + cc)
            out[r, c] = int(p[rr, cc])
            placed_global[r, c] = True
            placed_grp[r, c] = True

    return _tight_crop(out, bg)


def _gen_variants(patch: np.ndarray, mask: np.ndarray):
    base: list[tuple[np.ndarray, np.ndarray, bool]] = []
    for k in range(4):
        p = np.rot90(patch, k)
        m = np.rot90(mask, k)
        base.append((p, m, False))
        base.append((np.fliplr(p), np.fliplr(m), True))
    uniq = []
    seen = set()
    for p, m, is_flip in base:
        key = (m.tobytes(), p[m].tobytes())
        if key in seen:
            continue
        seen.add(key)
        uniq.append((p, m, bool(is_flip)))
    return uniq


# --------------- generator ---------------

GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (25, 30)
NUM_PAIRS_CHOICES = [1, 2]

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
        for _pair in range(num_pairs):
            candidates = [x for x in PARAM_OPTIONS if x[0] + x[3] not in used_plug_areas]
            if not candidates:
                specs = []
                break
            fw, top_h, bot_h, notch_n = random.choice(candidates)
            plug_area = int(fw + notch_n)
            specs.append(
                {
                    "fw": int(fw),
                    "top_h": int(top_h),
                    "bot_h": int(bot_h),
                    "notch_n": int(notch_n),
                    "plug_area": int(plug_area),
                }
            )
            used_plug_areas.add(int(plug_area))
        if len(specs) != num_pairs:
            continue
        all_plug_areas = {int(s["plug_area"]) for s in specs}
        ok = True
        for spec in specs:
            pa = int(spec["plug_area"])
            handle_choices = [
                h
                for h in range(ha_lo, ha_hi)
                if tt_lo <= pa + h <= tt_hi and h != pa and h not in all_plug_areas
            ]
            if not handle_choices:
                ok = False
                break
            spec["handle_area"] = int(random.choice(handle_choices))
        if ok:
            return [dict(s) for s in specs]
    return []


def _pick_sides_and_corner(num_pairs: int) -> tuple[list[str], str]:
    corner = random.choice(["tl", "tr", "bl", "br"])
    if num_pairs == 1:
        return [random.choice(["left", "right", "top", "bottom"])], corner
    corner_sides = {
        "tl": ["top", "left"],
        "tr": ["top", "right"],
        "bl": ["bottom", "left"],
        "br": ["bottom", "right"],
    }
    sides = corner_sides[corner][:]
    random.shuffle(sides)
    return sides, corner


def _candidate_origins(side: str, ph: int, pw: int, H: int, W: int, corner_hint: str) -> list[tuple[int, int]]:
    if side == "left":
        origins = [(r0, 0) for r0 in range(H - ph + 1)]
        target_r = 0 if corner_hint in ("tl", "tr") else H - ph
        origins.sort(key=lambda rc: abs(rc[0] - target_r))
    elif side == "right":
        origins = [(r0, W - pw) for r0 in range(H - ph + 1)]
        target_r = 0 if corner_hint in ("tl", "tr") else H - ph
        origins.sort(key=lambda rc: abs(rc[0] - target_r))
    elif side == "top":
        origins = [(0, c0) for c0 in range(W - pw + 1)]
        target_c = 0 if corner_hint in ("tl", "bl") else W - pw
        origins.sort(key=lambda rc: abs(rc[1] - target_c))
    elif side == "bottom":
        origins = [(H - ph, c0) for c0 in range(W - pw + 1)]
        target_c = 0 if corner_hint in ("tl", "bl") else W - pw
        origins.sort(key=lambda rc: abs(rc[1] - target_c))
    else:
        raise ValueError(side)

    # Within the preferred band, keep some randomness so the task is not template-fixed.
    band = min(len(origins), 8 if len(origins) > 1 else 1)
    head = origins[:band]
    tail = origins[band:]
    random.shuffle(head)
    random.shuffle(tail)
    return head + tail


def _place_docked_pair(
    i: int,
    spec: dict,
    colors: list[int],
    side: str,
    corner_hint: str,
    bg: int,
    H: int,
    W: int,
    docked_scene: np.ndarray,
    tool_only_output: np.ndarray,
    input_grid: np.ndarray,
    union_bboxes: list,
    frame_bboxes: list,
) -> dict | None:
    frame_color = int(colors[3 * i])
    plug_color = int(colors[3 * i + 1])
    handle_color = int(colors[3 * i + 2])

    result = _build_left_pair(
        bg=int(bg),
        frame_color=frame_color,
        plug_color=plug_color,
        handle_color=handle_color,
        fw=int(spec["fw"]),
        top_h=int(spec["top_h"]),
        bot_h=int(spec["bot_h"]),
        notch_n=int(spec["notch_n"]),
        handle_area=int(spec["handle_area"]),
    )
    if result is None:
        return None
    frame_left, tool_left = result
    scene_left = np.where(tool_left != int(bg), tool_left, frame_left)

    frame_scene = _orient_left_to_side(frame_left, side)
    tool = _orient_left_to_side(tool_left, side)
    scene = _orient_left_to_side(scene_left, side)
    ph, pw = scene.shape
    fr0, fc0, fr1, fc1 = _bbox(frame_scene != int(bg))

    origin = None
    for r0, c0 in _candidate_origins(side, ph, pw, H, W, corner_hint):
        union_bbox = (int(r0), int(c0), int(r0 + ph - 1), int(c0 + pw - 1))
        frame_bbox = (int(r0 + fr0), int(c0 + fc0), int(r0 + fr1), int(c0 + fc1))
        if side in ("left", "right") and (frame_bbox[0] == 0 or frame_bbox[2] == H - 1):
            continue
        if side in ("top", "bottom") and (frame_bbox[1] == 0 or frame_bbox[3] == W - 1):
            continue
        if any(_bbox_gap(union_bbox, prev) <= UNION_BBOX_GAP_MIN - 1 for prev in union_bboxes):
            continue
        if any(_bbox_gap(frame_bbox, prev) <= FRAME_BBOX_GAP_MIN - 1 for prev in frame_bboxes):
            continue
        if np.any((docked_scene[r0 : r0 + ph, c0 : c0 + pw] != int(bg)) & (scene != int(bg))):
            continue
        origin = (int(r0), int(c0))
        break
    if origin is None:
        return None

    r0, c0 = origin
    paste(docked_scene, scene, r0, c0, transparent=int(bg))
    paste(tool_only_output, tool, r0, c0, transparent=int(bg))
    paste(input_grid, frame_scene, r0, c0, transparent=int(bg))

    union_bboxes.append((int(r0), int(c0), int(r0 + ph - 1), int(c0 + pw - 1)))
    frame_bboxes.append((int(r0 + fr0), int(c0 + fc0), int(r0 + fr1), int(c0 + fc1)))
    return {
        "tool": tool,
        "plug_color": int(plug_color),
        "handle_color": int(handle_color),
        "plug_area": int(spec["plug_area"]),
        "handle_area": int(spec["handle_area"]),
    }


def _undock_tool(info: dict, bg: int, input_grid: np.ndarray, placement_canvas: np.ndarray) -> bool:
    for patch, is_flip in _variant_choices(info["tool"]):
        if is_flip and int(info["plug_area"]) != int(info["handle_area"]):
            patch = _swap_colors(patch, int(info["plug_color"]), int(info["handle_color"]))
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

    colors = [c for c in range(1, 10) if c != int(bg)]
    random.shuffle(colors)
    if len(colors) < 3 * num_pairs:
        return None

    sides, corner_hint = _pick_sides_and_corner(num_pairs)
    docked_scene = np.full((H, W), int(bg), dtype=int)
    tool_only_output = np.full((H, W), int(bg), dtype=int)
    input_grid = np.full((H, W), int(bg), dtype=int)
    union_bboxes: list[tuple[int, int, int, int]] = []
    frame_bboxes: list[tuple[int, int, int, int]] = []
    placed_tools: list[dict] = []

    for i, spec in enumerate(pair_specs):
        info = _place_docked_pair(
            i,
            spec,
            colors,
            sides[i],
            corner_hint,
            int(bg),
            H,
            W,
            docked_scene,
            tool_only_output,
            input_grid,
            union_bboxes,
            frame_bboxes,
        )
        if info is None:
            return None
        placed_tools.append(info)

    placement_canvas = docked_scene.copy()
    for info in placed_tools:
        if not _undock_tool(info, int(bg), input_grid, placement_canvas):
            return None

    if not np.any(tool_only_output != int(bg)):
        return None
    final_output = _tight_crop(tool_only_output, int(bg))

    # Enforce that the fused variant is visually distinct from the base task.
    if final_output.shape == input_grid.shape:
        return None
    if final_output.size >= int(0.8 * input_grid.size):
        return None
    if int(np.sum(docked_scene != int(bg))) <= int(np.sum(tool_only_output != int(bg))):
        return None

    if np.array_equal(main(input_grid), final_output):
        return input_grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPT_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPT_RETRIES} attempts")
