# Combo variant of a25697e4 (index 14)
# combo_spec: D3/0+D6/3+D14/0
# Name: 三色工具异色框架颜色重映射

from __future__ import annotations

from collections import Counter
import itertools
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, bbox as _bbox, get_bg, paste, random_free_location, random_sprite


# ---------------------------------------------------------------------------
# Design analysis for the combo variant
# ---------------------------------------------------------------------------
#
# design_analysis:
#   variant_summary: |
#     这是把原题的“按 cavity 对接工具”主链，和三个变化真正耦合后的版本：
#     1) frame 不再是单色，而是两个不同颜色的边框子组件；
#     2) tool 不再是两色，而是 plug / handle / hinge 三色；
#     3) dock 后不是保留或交换原工具颜色，而是按 frame 两个子组件的角色重映射：
#        plug -> notch 子组件色，handle -> bar 子组件色，hinge -> 背景色。
#     所以输出既依赖几何匹配，也依赖 frame 内部的“哪一块带 notch、哪一块是实心 bar”。
#   a_concretization_and_trace:
#     concrete_example: |
#       设一个 left-border frame 的上方 bar 是蓝色(1)，下方带 notch 的部件是红色(2)。
#       输入中的自由工具三色分别为 plug=绿色(3)、handle=黄色(4)、hinge=粉色(6)。
#       原题逻辑只会把某一块颜色填进 cavity，并保留/交换工具本身颜色。
#       本融合版则是：
#       - 先把工具做某个 D4 变换，让绿色 plug 恰好填满阶梯 cavity；
#       - 黄色 handle 保持在 frame 外侧；
#       - 粉色 hinge 是输入里连接 plug 与 handle 的 1-2 像素，但在输出里直接消失成背景；
#       - 输出中 cavity 里的整块绿色会变成红色，因为它对应 notch 子组件；
#       - 输出中外侧黄色 handle 会变成蓝色，因为它对应实心 bar 子组件。
#       差异不只是“多一色”或“换颜色”，而是三色结构、双色 frame、颜色重映射同时参与同一对接结果。
#   b_complexity_calibration:
#     controls:
#       - 仍只保留 1-2 对 frame/tool，维持原题的观察负担。
#       - 每个样例里所有工具共享同一组三色输入配色，frame 颜色按 pair 区分，方便肉眼看出“工具整体换成 frame 配色”。
#       - plug area 仍限制在 5-8，且多 pair 时各 pair 的 plug area 互异，避免配对歧义。
#       - hinge 长度只取 1 或 2，且始终是沿开口方向的一小段直线，变化可见但不花哨。
#     anti_degeneracy:
#       - 强制 tool 三色都出现且 hinge 长度 >= 1。
#       - 强制 plug 输入色 != notch 色、handle 输入色 != bar 色，防止“重映射看不出来”。
#       - 强制 frame 两子组件颜色不同，防止退化回原题的单色框架。
#   c_implicit_assumption_audit:
#     original_assumptions_broken:
#       - 原 solver 默认 movable component 恰好两色；融合后必须显式识别三色并区分 plug / handle / hinge。
#       - 原 solver 用“反射时交换两色”处理颜色；融合后颜色完全来自 frame 的两个子组件角色。
#       - 原 solver 只需要知道 cavity 大小；融合后还需要知道 frame 组内哪一个组件是 notch 组件。
#       - 原 solver 输出时直接把工具颜色写回；融合后 hinge 不能写回，必须消失为背景。
#     line_level_policy: |
#       因此 solver 改成两段推理：
#       先从 border-touching 组件恢复双色 frame，并通过“bbox 是否被填满”判定谁是 notch 组件；
#       再对每个三色工具枚举 D4 变换，用 plug 形状精确匹配 cavity，并在渲染阶段做角色驱动的重着色。
#   d_placement_legality:
#     generator_strategy: |
#       继续采用 output-first。先在输出态构造“已经 dock 好”的 frame + recolored tool，
#       然后把 tri-color tool 从 docked 状态中拆出来，随机旋转/翻转后远离 frame 放回输入。
#     legality_guards:
#       - frame 组之间保持 bbox 间距，scene 不重叠、不越界。
#       - 自由工具通过 `random_free_location` 放到非边界位置，并与其他内容保留 padding。
#       - hinge 放在 frame 开口外侧，保证其消失后不会破坏 cavity 填充。
#   e_visual_observability:
#     rationale: |
#       解题者会同时观察到三件稳定事实：
#       1) 输入里的自由工具总是三色，而输出里只剩 frame 的两种颜色；
#       2) 消失的总是夹在 plug 与 handle 之间的细小第三色 hinge；
#       3) cavity 内颜色总是变成带 notch 的那块 frame 颜色，而外侧 handle 变成另一块 frame 颜色。
#       这让颜色变化带有明确的结构锚点，而不是任意 recolor。
#   f_visual_cleanliness:
#     rationale: |
#       画面仍然只有 frame 和 tool 两类物体，没有额外符号层。hinge 很短且只出现在输入工具内部，
#       输出中直接消失，减少 dock 后的局部堆叠。frame 内部两块都是大块纯色，角色区分清晰。
#   g_color_change_inferability:
#     rationale: |
#       颜色重映射不是按数值大小、排序或隐式标签，而是按可见结构角色：
#       cavity 对应的 plug 变成 notch 组件色，外侧 handle 变成 bar 组件色，hinge 消失。
#       即使 2-3 个样例，也能从“颜色去向”和“所在结构位置”归纳出规则。
#   h_main_new_assumptions_to_check:
#     solver_assumptions:
#       - 每个 frame group 恰有两个 border-touching 4 连通子组件，且颜色不同。
#       - 其中恰有一个组件不是满 bbox 的实心矩形；该组件就是 notch 组件。
#       - 每个 movable tool 恰有三种颜色，其中一色像素数等于某个 cavity 大小，另一色面积为 1-2。
#       - plug 的某个 D4 变换能与某个 cavity 形状完全吻合，且 handle/hinge 落在 frame 开口外侧。
#     generator_support:
#       - `_build_left_pair` 直接按这些结构约束构造 docked 态。
#       - `_pick_pair_specs` 保证多 pair 时 plug area 互异且 handle area 不与任何 plug area 冲突。
#       - `_gen_attempt` 用 `main(input) == output` 做 oracle 检查，并且所有颜色集合按融合规则强制分离。


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


def _normalize(coords: np.ndarray) -> tuple[frozenset[tuple[int, int]], tuple[int, int]]:
    coords = np.asarray(coords, dtype=int)
    assert coords.size > 0
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    shape = frozenset((int(r - r0), int(c - c0)) for r, c in coords)
    return shape, (r0, c0)


def _outside_in_open_dir(r: int, c: int, bbox, open_dir) -> bool:
    r0, c0, r1, c1 = bbox
    dr, dc = open_dir
    if (dr, dc) == (0, 1):
        return int(c) > int(c1)
    if (dr, dc) == (0, -1):
        return int(c) < int(c0)
    if (dr, dc) == (1, 0):
        return int(r) > int(r1)
    if (dr, dc) == (-1, 0):
        return int(r) < int(r0)
    raise ValueError(open_dir)


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
    dims = [
        (h, w)
        for h in range(HANDLE_SPRITE_DIM[0], HANDLE_SPRITE_DIM[1])
        for w in range(HANDLE_SPRITE_DIM[0], HANDLE_SPRITE_DIM[1])
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
            color_palette=[handle_color],
            connectivity=4,
            background=bg,
        )
        if int(np.sum(sprite != bg)) == int(handle_area):
            return sprite
    return None


def _gen_variants(patch: np.ndarray, mask: np.ndarray):
    base: list[tuple[np.ndarray, np.ndarray]] = []
    for k in range(4):
        p = np.rot90(patch, k)
        m = np.rot90(mask, k)
        base.append((p, m))
        base.append((np.fliplr(p), np.fliplr(m)))
    seen = set()
    out = []
    for p, m in base:
        key = (m.tobytes(), p[m].tobytes())
        if key in seen:
            continue
        seen.add(key)
        out.append((p, m))
    return out


def _component_bbox(pts: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _largest_adjacent_cavity(grid: np.ndarray, union_mask: np.ndarray, bbox, bg: int) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    sub_bg = grid[r0 : r1 + 1, c0 : c1 + 1] == int(bg)
    sub_lab, sub_n = label(sub_bg, structure=STRUCT4)
    best_id = None
    best_key = None
    for cid in range(1, int(sub_n) + 1):
        pts = np.argwhere(sub_lab == int(cid))
        area = int(len(pts))
        adj = False
        for rr, cc in pts:
            rr2 = int(r0 + rr)
            cc2 = int(c0 + cc)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                r3 = rr2 + dr
                c3 = cc2 + dc
                if r0 <= r3 <= r1 and c0 <= c3 <= c1 and bool(union_mask[r3, c3]):
                    adj = True
                    break
            if adj:
                break
        if not adj:
            continue
        key = (area, -int(cid))
        if best_key is None or key > best_key:
            best_key = key
            best_id = int(cid)
    cavity = np.zeros_like(union_mask, dtype=bool)
    if best_id is None:
        return cavity
    pts = np.argwhere(sub_lab == int(best_id))
    for rr, cc in pts:
        cavity[int(r0 + rr), int(c0 + cc)] = True
    return cavity


def _open_dir_from_mask(mask: np.ndarray) -> tuple[int, int]:
    H, W = mask.shape
    pts = np.argwhere(mask)
    sides = set()
    for r, c in pts:
        if int(c) == 0:
            sides.add("left")
        if int(c) == W - 1:
            sides.add("right")
        if int(r) == 0:
            sides.add("top")
        if int(r) == H - 1:
            sides.add("bottom")
    assert len(sides) == 1
    side = next(iter(sides))
    return {
        "left": (0, 1),
        "right": (0, -1),
        "top": (1, 0),
        "bottom": (-1, 0),
    }[side]


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


def _variant_choices(patch: np.ndarray) -> list[np.ndarray]:
    mask = patch != 0
    seen = set()
    out = []
    for k in range(4):
        for flip in (False, True):
            q = _transform(patch, k, flip)
            m = _transform(mask.astype(np.uint8), k, flip).astype(bool)
            key = (m.tobytes(), q[m].tobytes())
            if key in seen:
                continue
            seen.add(key)
            out.append(q)
    random.shuffle(out)
    return out


def _build_left_pair(
    *,
    bg: int,
    notch_color: int,
    bar_color: int,
    plug_color: int,
    handle_color: int,
    hinge_color: int,
    fw: int,
    top_h: int,
    bot_h: int,
    notch_n: int,
    handle_area: int,
    hinge_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    fh = top_h + 1 + bot_h
    frame = np.full((fh, fw), bg, dtype=int)
    frame[:top_h, :] = int(bar_color)
    frame[top_h + 1 :, :] = int(notch_color)
    for dr, dc in _notch_offsets(notch_n, fw, bot_h):
        frame[top_h + 1 + dr, dc] = int(bg)

    cavity = frame == int(bg)
    attach_rows = [int(r) for r in range(fh) if bool(cavity[r, fw - 1])]
    if not attach_rows:
        return None

    handle_sprite = _make_handle_sprite(handle_area, bg, handle_color)
    if handle_sprite is None:
        return None
    hs_h, hs_w = handle_sprite.shape
    sprite_left_rows = [int(r) for r in range(hs_h) if int(handle_sprite[r, 0]) != int(bg)]
    if not sprite_left_rows:
        return None

    hinge_row = random.choice(attach_rows)
    valid_offsets = [
        voff
        for voff in range(max(0, hinge_row - hs_h + 1), hinge_row + 1)
        if any(int(voff + sr) == int(hinge_row) for sr in sprite_left_rows)
    ]
    if not valid_offsets:
        return None
    voff = random.choice(valid_offsets)

    total_h = max(fh, voff + hs_h)
    total_w = fw + hinge_len + hs_w

    frame_full = np.full((total_h, total_w), bg, dtype=int)
    frame_full[:fh, :fw] = frame

    docked_tool = np.full((total_h, total_w), bg, dtype=int)
    docked_tool[:fh, :fw][cavity] = int(plug_color)
    for dc in range(hinge_len):
        docked_tool[hinge_row, fw + dc] = int(hinge_color)
    handle_mask = handle_sprite != int(bg)
    docked_tool[voff : voff + hs_h, fw + hinge_len : fw + hinge_len + hs_w][handle_mask] = int(handle_color)

    output_tool = np.full((total_h, total_w), bg, dtype=int)
    output_tool[:fh, :fw][cavity] = int(notch_color)
    output_tool[voff : voff + hs_h, fw + hinge_len : fw + hinge_len + hs_w][handle_mask] = int(bar_color)
    return frame_full, docked_tool, output_tool


def _parse_anchor_groups(grid: np.ndarray, bg: int):
    H, W = grid.shape
    fg = grid != int(bg)
    lab, n = label(fg, structure=STRUCT4)
    fixed_mask = np.zeros((H, W), dtype=bool)
    fixed_components = []
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        touches = any(
            int(r) == 0 or int(r) == H - 1 or int(c) == 0 or int(c) == W - 1
            for r, c in pts
        )
        if not touches:
            continue
        bbox = _component_bbox(pts)
        colors = sorted({int(grid[int(r), int(c)]) for r, c in pts} - {int(bg)})
        assert len(colors) == 1
        for r, c in pts:
            fixed_mask[int(r), int(c)] = True
        fixed_components.append(
            {
                "pts": pts,
                "bbox": bbox,
                "color": int(colors[0]),
                "area": int(len(pts)),
            }
        )
    assert fixed_components

    parent = list(range(len(fixed_components)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
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
        comps = [fixed_components[i] for i in idxs]
        assert len(comps) == 2
        # D14/0: the two frame subcomponents must be different colors.
        assert int(comps[0]["color"]) != int(comps[1]["color"])
        all_pts = np.vstack([comp["pts"] for comp in comps])
        bbox = _component_bbox(all_pts)
        mask = np.zeros((H, W), dtype=bool)
        for r, c in all_pts:
            mask[int(r), int(c)] = True
        cavity = _largest_adjacent_cavity(grid, mask, bbox, bg)
        cav_pts = np.argwhere(cavity)
        assert len(cav_pts) > 0
        cav_shape, cav_anchor = _normalize(cav_pts)
        open_dir = _open_dir_from_mask(mask)

        roles = []
        for comp in comps:
            r0, c0, r1, c1 = comp["bbox"]
            bbox_area = int((r1 - r0 + 1) * (c1 - c0 + 1))
            roles.append(
                {
                    "color": int(comp["color"]),
                    "bbox_area": bbox_area,
                    "area": int(comp["area"]),
                }
            )
        notch_parts = [x for x in roles if int(x["area"]) < int(x["bbox_area"])]
        bar_parts = [x for x in roles if int(x["area"]) == int(x["bbox_area"])]
        assert len(notch_parts) == 1 and len(bar_parts) == 1
        anchor_groups.append(
            {
                "bbox": bbox,
                "mask": mask,
                "cavity_mask": cavity,
                "cavity_size": int(len(cav_pts)),
                "cavity_shape": cav_shape,
                "cavity_anchor": cav_anchor,
                "open_dir": open_dir,
                "notch_color": int(notch_parts[0]["color"]),
                "bar_color": int(bar_parts[0]["color"]),
                "size": int(mask.sum()),
            }
        )
    anchor_groups.sort(key=lambda d: -int(d["size"]))
    return fixed_mask, anchor_groups


def _parse_movable_components(grid: np.ndarray, bg: int, fixed_mask: np.ndarray):
    fg = grid != int(bg)
    mov_mask = fg & (~fixed_mask)
    mov_lab, mov_n = label(mov_mask, structure=STRUCT4)
    comps = []
    for cid in range(1, int(mov_n) + 1):
        pts = np.argwhere(mov_lab == int(cid))
        r0, c0, r1, c1 = _component_bbox(pts)
        mask = mov_lab[r0 : r1 + 1, c0 : c1 + 1] == int(cid)
        patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
        patch[mask] = grid[r0 : r1 + 1, c0 : c1 + 1][mask]
        colors = sorted({int(x) for x in patch[mask].flatten()} - {int(bg)})
        assert len(colors) == 3
        counts = {int(col): int(((patch == int(col)) & mask).sum()) for col in colors}
        comps.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "mask": mask,
                "patch": patch,
                "colors": colors,
                "counts": counts,
                "size": int(len(pts)),
            }
        )
    comps.sort(key=lambda d: -int(d["size"]))
    return comps


def _find_dock_state(comp: dict, grp: dict, fixed_mask: np.ndarray, bg: int, shape) -> dict | None:
    H, W = shape
    cavity_coords = np.argwhere(grp["cavity_mask"])
    cavity_set = {(int(r), int(c)) for r, c in cavity_coords}
    for p, m in _gen_variants(comp["patch"], comp["mask"]):
        colors = sorted({int(x) for x in p[m].flatten()} - {int(bg)})
        if len(colors) != 3:
            continue
        counts = {int(col): int(((p == int(col)) & m).sum()) for col in colors}
        plug_candidates = [int(col) for col in colors if int(counts[col]) == int(grp["cavity_size"])]
        hinge_candidates = [int(col) for col in colors if int(counts[col]) in (1, 2)]
        if len(plug_candidates) != 1 or len(hinge_candidates) != 1:
            continue
        plug_color = int(plug_candidates[0])
        hinge_color = int(hinge_candidates[0])
        handle_candidates = [int(col) for col in colors if int(col) not in (plug_color, hinge_color)]
        if len(handle_candidates) != 1:
            continue
        handle_color = int(handle_candidates[0])

        plug_local = np.argwhere((p == int(plug_color)) & m)
        plug_shape, plug_anchor = _normalize(plug_local)
        if plug_shape != grp["cavity_shape"]:
            continue
        dr = int(grp["cavity_anchor"][0] - plug_anchor[0])
        dc = int(grp["cavity_anchor"][1] - plug_anchor[1])

        shifted = []
        for rr, cc in np.argwhere(m):
            r = int(rr + dr)
            c = int(cc + dc)
            if not (0 <= r < H and 0 <= c < W):
                shifted = []
                break
            shifted.append((int(r), int(c), int(p[rr, cc])))
        if not shifted:
            continue

        plug_pts = {(int(r), int(c)) for r, c, col in shifted if int(col) == int(plug_color)}
        hinge_pts = {(int(r), int(c)) for r, c, col in shifted if int(col) == int(hinge_color)}
        handle_pts = {(int(r), int(c)) for r, c, col in shifted if int(col) == int(handle_color)}
        if plug_pts != cavity_set:
            continue
        if any(bool(fixed_mask[r, c]) for r, c in hinge_pts | handle_pts):
            continue
        if not hinge_pts or not handle_pts:
            continue
        if not all(_outside_in_open_dir(r, c, grp["bbox"], grp["open_dir"]) for r, c in hinge_pts):
            continue
        if not all(_outside_in_open_dir(r, c, grp["bbox"], grp["open_dir"]) for r, c in handle_pts):
            continue

        dock_bbox = (
            min(r for r, _c, _col in shifted),
            min(c for _r, c, _col in shifted),
            max(r for r, _c, _col in shifted),
            max(c for _r, c, _col in shifted),
        )
        score = (
            int(len(handle_pts)),
            -int(_bbox_gap(comp["bbox"], grp["bbox"])),
            -int(_bbox_gap(dock_bbox, grp["bbox"])),
        )
        return {
            "plug_pts": plug_pts,
            "hinge_pts": hinge_pts,
            "handle_pts": handle_pts,
            "score": score,
        }
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock each free tri-color tool into the matching border frame, then repaint it with the frame palette.

    Input structure:
    - `bg` is the dominant color.
    - Border-touching foreground splits into frame groups of two differently colored 4-connected components:
      one solid rectangle (`bar`), one rectangle with a staircase bite (`notch`).
    - Interior foreground components are 3-color tools made of a cavity-shaped `plug`, an outer `handle`,
      and a tiny 1-2 pixel `hinge` linking them.

    Rule:
    - Match a tool to the frame whose cavity matches the plug region under some rotation/flip.
    - Place the transformed tool so the plug fills the cavity exactly and the rest stays on the open side.
    - Recolor the placed plug to the notch-component color, recolor the handle to the bar-component color,
      and erase the hinge back to background.

    Key insight: Geometry identifies which tool docks where, but the final colors come from the frame's two
    subcomponent roles rather than from the tool's original three-color palette.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    fixed_mask, anchor_groups = _parse_anchor_groups(g, bg)
    mov_comps = _parse_movable_components(g, bg, fixed_mask)
    assert len(mov_comps) == len(anchor_groups)

    compat: dict[tuple[int, int], dict | None] = {}
    for ci, comp in enumerate(mov_comps):
        for gi, grp in enumerate(anchor_groups):
            if int(grp["cavity_size"]) not in {int(v) for v in comp["counts"].values()}:
                compat[(ci, gi)] = None
                continue
            compat[(ci, gi)] = _find_dock_state(comp, grp, fixed_mask, bg, (H, W))

    best_perm = None
    best_score = None
    for perm in itertools.permutations(range(len(anchor_groups))):
        states = []
        dist_sum = 0
        ok = True
        for ci, gi in enumerate(perm):
            state = compat[(ci, gi)]
            if state is None:
                ok = False
                break
            states.append(state)
            dist_sum += _bbox_gap(mov_comps[ci]["bbox"], anchor_groups[gi]["bbox"])
        if not ok:
            continue
        score = (-int(dist_sum),)
        if best_score is None or score > best_score:
            best_score = score
            best_perm = tuple(int(x) for x in perm)
    assert best_perm is not None

    out = np.full_like(g, int(bg))
    out[fixed_mask] = g[fixed_mask]
    for ci, gi in enumerate(best_perm):
        grp = anchor_groups[gi]
        state = compat[(ci, gi)]
        assert state is not None
        for r, c in state["plug_pts"]:
            out[int(r), int(c)] = int(grp["notch_color"])
        for r, c in state["handle_pts"]:
            out[int(r), int(c)] = int(grp["bar_color"])
    return out


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (25, 30)
NUM_PAIRS_CHOICES = [1, 2]

FRAME_W_RANGE = (4, 8)
NOTCH_N_RANGE = (1, 4)
TOP_H_RANGE = (1, 4)
BOT_H_RANGE = (2, 4)
PLUG_AREA_BOUNDS = (5, 8)

HANDLE_AREA_RANGE = (4, 9)
HINGE_LEN_CHOICES = [1, 2]
TOOL_TOTAL_BOUNDS = (11, 15)
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
    out = []
    for fw in range(FRAME_W_RANGE[0], FRAME_W_RANGE[1]):
        for notch_n in range(NOTCH_N_RANGE[0], NOTCH_N_RANGE[1]):
            plug_area = int(fw + notch_n)
            if not (PLUG_AREA_BOUNDS[0] <= plug_area <= PLUG_AREA_BOUNDS[1]):
                continue
            for top_h in range(TOP_H_RANGE[0], TOP_H_RANGE[1]):
                for bot_h in range(BOT_H_RANGE[0], BOT_H_RANGE[1]):
                    out.append((fw, top_h, bot_h, notch_n))
    return out


PARAM_OPTIONS = _build_param_options()


def _pick_pair_specs(num_pairs: int) -> list[dict]:
    specs: list[dict] = []
    for _ in range(PAIR_SPEC_RETRIES):
        specs.clear()
        used_plug_areas: set[int] = set()
        for _j in range(num_pairs):
            candidates = [x for x in PARAM_OPTIONS if int(x[0] + x[3]) not in used_plug_areas]
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
            candidates = []
            for handle_area in range(HANDLE_AREA_RANGE[0], HANDLE_AREA_RANGE[1]):
                for hinge_len in HINGE_LEN_CHOICES:
                    total = int(pa + handle_area + hinge_len)
                    if not (TOOL_TOTAL_BOUNDS[0] <= total <= TOOL_TOTAL_BOUNDS[1]):
                        continue
                    if int(handle_area) in all_plug_areas or int(handle_area) == int(pa):
                        continue
                    candidates.append((int(handle_area), int(hinge_len)))
            if not candidates:
                ok = False
                break
            handle_area, hinge_len = random.choice(candidates)
            spec["handle_area"] = int(handle_area)
            spec["hinge_len"] = int(hinge_len)
        if ok:
            return [dict(s) for s in specs]
    return []


def _place_docked_pair(
    i: int,
    spec: dict,
    frame_colors: list[int],
    tool_colors: tuple[int, int, int],
    side: str,
    bg: int,
    H: int,
    W: int,
    output: np.ndarray,
    input_grid: np.ndarray,
    union_bboxes: list,
    frame_bboxes: list,
) -> dict | None:
    notch_color = int(frame_colors[2 * i])
    bar_color = int(frame_colors[2 * i + 1])
    plug_color, handle_color, hinge_color = map(int, tool_colors)

    built = _build_left_pair(
        bg=bg,
        notch_color=notch_color,
        bar_color=bar_color,
        plug_color=plug_color,
        handle_color=handle_color,
        hinge_color=hinge_color,
        fw=int(spec["fw"]),
        top_h=int(spec["top_h"]),
        bot_h=int(spec["bot_h"]),
        notch_n=int(spec["notch_n"]),
        handle_area=int(spec["handle_area"]),
        hinge_len=int(spec["hinge_len"]),
    )
    if built is None:
        return None
    frame_left, tool_left, output_tool_left = built

    frame_scene = _orient_left_to_side(frame_left, side)
    tool = _orient_left_to_side(tool_left, side)
    output_tool = _orient_left_to_side(output_tool_left, side)
    scene = np.where(output_tool != int(bg), output_tool, frame_scene)
    ph, pw = scene.shape
    frame_bbox_local = _bbox(frame_scene != int(bg))
    assert frame_bbox_local is not None
    fr0, fc0, fr1, fc1 = frame_bbox_local

    origin = None
    for r0, c0 in _candidate_origins(side, ph, pw, H, W):
        union_bbox = (int(r0), int(c0), int(r0 + ph - 1), int(c0 + pw - 1))
        frame_bbox = (int(r0 + fr0), int(c0 + fc0), int(r0 + fr1), int(c0 + fc1))
        if side in ("left", "right"):
            if int(frame_bbox[0]) == 0 or int(frame_bbox[2]) == H - 1:
                continue
            if int(union_bbox[0]) == 0 or int(union_bbox[2]) == H - 1:
                continue
        else:
            if int(frame_bbox[1]) == 0 or int(frame_bbox[3]) == W - 1:
                continue
            if int(union_bbox[1]) == 0 or int(union_bbox[3]) == W - 1:
                continue
        if any(_bbox_gap(union_bbox, prev) <= UNION_BBOX_GAP_MIN - 1 for prev in union_bboxes):
            continue
        if any(_bbox_gap(frame_bbox, prev) <= FRAME_BBOX_GAP_MIN - 1 for prev in frame_bboxes):
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
    frame_bboxes.append((int(r0 + fr0), int(c0 + fc0), int(r0 + fr1), int(c0 + fc1)))
    return {
        "tool": tool,
        "plug_area": int(spec["plug_area"]),
        "hinge_len": int(spec["hinge_len"]),
        "frame_colors": (int(notch_color), int(bar_color)),
        "tool_colors": (int(plug_color), int(handle_color), int(hinge_color)),
    }


def _undock_tool(info: dict, bg: int, input_grid: np.ndarray, placement_canvas: np.ndarray) -> bool:
    for patch in _variant_choices(info["tool"]):
        try:
            r0, c0 = random_free_location(
                placement_canvas,
                patch,
                transparent=bg,
                border=TOOL_BORDER,
                padding=TOOL_PADDING,
                padding_connectivity=TOOL_PADDING_CONN,
            )
        except ValueError:
            continue
        paste(input_grid, patch, r0, c0, transparent=bg)
        paste(placement_canvas, patch, r0, c0, transparent=bg)
        return True
    return False


def _color_sets_separated(placed_tools: list[dict], bg: int) -> bool:
    used = {int(bg)}
    for info in placed_tools:
        notch_color, bar_color = info["frame_colors"]
        plug_color, handle_color, hinge_color = info["tool_colors"]
        if len({notch_color, bar_color, plug_color, handle_color, hinge_color, int(bg)}) != 6:
            return False
        used |= {int(notch_color), int(bar_color), int(plug_color), int(handle_color), int(hinge_color)}
    return len(used) >= 6


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    bg = random.choice(list(range(10)))
    num_pairs = random.choice(NUM_PAIRS_CHOICES)

    specs = _pick_pair_specs(num_pairs)
    if len(specs) != num_pairs:
        return None

    colors = [int(c) for c in range(10) if int(c) != int(bg)]
    random.shuffle(colors)
    if len(colors) < int(3 + 2 * num_pairs):
        return None
    tool_colors = (int(colors[0]), int(colors[1]), int(colors[2]))
    frame_colors = [int(c) for c in colors[3 : 3 + 2 * num_pairs]]

    output = np.full((H, W), int(bg), dtype=int)
    input_grid = np.full((H, W), int(bg), dtype=int)
    union_bboxes: list[tuple[int, int, int, int]] = []
    frame_bboxes: list[tuple[int, int, int, int]] = []
    placed_tools: list[dict] = []
    sides = random.sample(["left", "right", "top", "bottom"], k=num_pairs)

    for i, spec in enumerate(specs):
        info = _place_docked_pair(
            i,
            spec,
            frame_colors,
            tool_colors,
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

    if not _color_sets_separated(placed_tools, bg):
        return None

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
