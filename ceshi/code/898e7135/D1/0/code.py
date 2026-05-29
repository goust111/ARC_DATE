# Variant of 898e7135
# Dimension: D1 — 匹配条件：块级形状匹配（含旋转/翻转） (match_cond)
# Change: shape_match + rotation_flip — 精灵下采样为块级掩码后，与洞的形状在二面体群（4次旋转 × 水平翻转）下比较
#         颜色匹配替代形状匹配：每个洞区域的边缘墙壁像素染成该洞的专属颜色，外部精灵用同色表示；
#         solver 先按颜色配对，再只在该已配对精灵内部恢复能贴合洞形的旋转/翻转朝向。

from __future__ import annotations

"""
Design analysis for the D1 variant
==================================

spec_variant:
  preserved_core:
    - 输入仍然有一个包围若干洞区域的矩形 frame，洞仍然是背景色 0。
    - 外部 pieces 仍然是洞形状按 k x k 放大的精灵，并且仍可随机旋转/翻转。
    - 输出仍然是裁剪后的放大 frame，背景为 frame 的基础墙色，piece 被填回对应洞。
    - k 仍由最小的实心 k x k 方块 piece 推断。
  changed_dimension_D1:
    old:
      - 洞与 piece 的对应关系由块级形状在 D4 群下匹配得到。
    new:
      - 每个洞四邻接触到的墙像素都染成该洞的专属颜色。
      - 外部 piece 的实体像素也全部使用该颜色。
      - solver 不再在所有 piece 中做形状搜索，而是先用颜色锁定唯一 piece，
        再只对这一个 piece 测试 D4 朝向以恢复正确摆放姿态。
  pixel_level_realization:
    - frame 现在是多色连通体：基础墙色 + 若干洞边标记色。
    - 洞本身仍是 0，因此原题依赖的“从 frame bbox 中提取 hole mask”依旧成立。
    - 为避免退化回旧题，生成器允许重复洞形；同形不同色的 slots 会迫使解题者使用颜色而不是形状配对。

behavior_trace:
  local_example:
    - 原逻辑：两个 L 形洞分别对应红/绿两个外部 L 形 piece。solver 先枚举每个 piece 的所有 D4 变体，
      再在所有洞中找形状相等者；颜色对配对过程没有作用。
    - 变体逻辑：两个 L 形洞的洞边墙像素分别染成红/绿。solver 先看到“红洞边 -> 红 piece、绿洞边 -> 绿 piece”，
      然后仅在红 piece 内部测试哪一个 D4 朝向能贴合红洞。
    - 差异：几何比较从“跨 piece 的身份判定”退化为“已知身份后的姿态恢复”。若两个洞形完全相同，
      原逻辑会歧义或任意交换，而新逻辑仍可稳定输出。

complexity_calibration:
  - 不使用颜色值大小做任何排序或映射，颜色只作为可见类别标签。
  - slot 颜色总数 <= 6，且每个样例中都至少有一个重复形状组（两块 unit slot 或一组重复非 unit 形状），
    使颜色配对成为必要信息，而不是可有可无的附加线索。
  - 仍保留 piece 的随机旋转/翻转，因此样例不是“纯颜色拷贝”；解题者仍需理解放大与姿态恢复。

implicit_assumption_audit:
  - 原 main 用 frame 组件的主色作为输出背景，默认 frame 单色。该前提在变体中失效。
    新实现改为：先抽取所有外部 piece 颜色，再把 frame 组件中“不属于任何 piece 的那一个颜色”
    视为基础墙色；生成器保证此颜色唯一。
  - 原 main 对 slot 和 piece 的匹配完全依赖 area + 下采样形状相等。该前提改为：
    颜色先给出唯一 piece；area/形状只用于验证和恢复朝向。
  - 原 generator 允许 unit slots 共用颜色，因为它们可由形状/面积区分。变体中这会导致颜色歧义，
    因此 generator 改为所有 slots 颜色唯一。
  - 原 generator 强制非 unit shapes 在 D4 下互异。变体中不再需要，反而要经常制造重复形状以体现新规则。

placement_legality:
  - 仍要求 pieces 与 frame、彼此之间至少隔 1 个背景像素。
  - 生成 layout 时继续保持洞之间 8 分离，并额外避免两个不同洞共享同一个四邻墙像素，
    这样“墙色 -> 洞”的关系是单值的，solver 可安全提取每个洞的唯一标记色。
  - 洞仍整体触及 tight-crop 的四条边，保证 solver 从 hole bbox 裁剪后得到的 layout 与生成器一致。

visual_observability:
  - 输入里 frame 内部明显出现“彩色墙环 + 同色外部精灵”的重复结构，人眼可直接注意到颜色配对。
  - 输出中这些墙色标记被移除，只保留基础墙色和填入的彩色 pieces，规则从少量样例可归纳为：
    “看洞边颜色，把同色块放回去，并按洞姿态旋正。”

new_solver_assumptions_and_generator_guarantees:
  - solver 假设每个 slot 的相邻非背景墙色与某一个 piece 颜色唯一相交。generator 通过唯一 slot 颜色保证。
  - solver 假设每个 piece 仍是合法的 k 块缩放图形，且其某个 D4 变体可下采样回 slot mask。
    generator 仍按 slot mask 直接放大后再随机 D4 变换，保证成立。
  - solver 假设 frame 组件中恰有一个“不是任何 piece 颜色”的墙色作为基础墙色。
    generator 通过独立选择 frame_color、distractor_color、slot colors 保证该条件。
"""

from collections import Counter
import random

import numpy as np

from arc_utils import bbox, bbox_coords, components_mask, crop_bbox, find_components, get_bg, paste, random_free_location


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _canonical_shape_key(mask: np.ndarray) -> bytes:
    variants = []
    for kk in range(4):
        r = np.rot90(mask, kk)
        variants.append(r.tobytes() + bytes([r.shape[0], r.shape[1]]))
        fr = np.fliplr(r)
        variants.append(fr.tobytes() + bytes([fr.shape[0], fr.shape[1]]))
    return min(variants)


def _downsample_blocks(mask: np.ndarray, k: int) -> np.ndarray | None:
    hh, ww = mask.shape
    if hh % k != 0 or ww % k != 0:
        return None
    out_h, out_w = int(hh // k), int(ww // k)
    out = np.zeros((out_h, out_w), dtype=bool)
    for r in range(out_h):
        for c in range(out_w):
            block = mask[int(r * k) : int((r + 1) * k), int(c * k) : int((c + 1) * k)]
            if bool(block.all()):
                out[r, c] = True
            elif bool(block.any()):
                return None
    return out


def _dihedral_variants(patch: np.ndarray, mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    base = []
    for kk in range(4):
        p = np.rot90(patch, int(kk))
        m = np.rot90(mask, int(kk))
        base.append((p, m))
        base.append((np.fliplr(p), np.fliplr(m)))
    uniq = []
    seen = set()
    for p, m in base:
        key = m.tobytes() + bytes([m.shape[0], m.shape[1]])
        if key in seen:
            continue
        seen.add(key)
        uniq.append((p, m))
    return uniq


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps, "input must contain foreground components"

    frame_comp = max(comps, key=lambda d: int(d["area"]))

    pieces = []
    for comp in comps:
        if comp is frame_comp:
            continue
        coords = comp["coords"]
        if coords.size == 0 or int(coords.shape[0]) <= 1:
            continue
        pr0, pc0, pr1, pc1 = map(int, comp["bbox"])
        patch = crop_bbox(g, (pr0, pc0, pr1, pc1))
        ph, pw = map(int, patch.shape)
        pm = np.zeros((ph, pw), dtype=bool)
        pm[coords[:, 0] - pr0, coords[:, 1] - pc0] = True
        pieces.append(
            {
                "color": int(g[int(coords[0, 0]), int(coords[0, 1])]),
                "area": int(pm.sum()),
                "mask": pm,
                "patch": patch,
            }
        )

    assert pieces, "at least one non-frame piece must exist"
    piece_colors = {int(p["color"]) for p in pieces}
    assert len(piece_colors) == len(pieces), "slot colors must uniquely identify pieces"

    # k is still encoded by the unit-slot pieces, exactly as in the original task.
    k = None
    for p in pieces:
        h, w = map(int, p["mask"].shape)
        if h == w and int(p["mask"].sum()) == int(h * w):
            k = int(h) if k is None else min(int(k), int(h))
    k = int(k or 1)

    frame_color_counts = Counter(int(g[int(r), int(c)]) for r, c in frame_comp["coords"])
    frame_color_candidates = [col for col in frame_color_counts if int(col) not in piece_colors]
    assert frame_color_candidates, "frame must have a dedicated base wall color"
    bgc = max(frame_color_candidates, key=lambda col: int(frame_color_counts[int(col)]))

    fr0, fc0, fr1, fc1 = map(int, frame_comp["bbox"])
    sub = crop_bbox(g, (fr0, fc0, fr1, fc1))
    hole = sub == int(bg)
    hole_box = bbox(hole)
    assert hole_box is not None, "frame must enclose at least one bg hole"
    lr0, lc0, lr1, lc1 = map(int, hole_box)
    layout = hole[lr0 : lr1 + 1, lc0 : lc1 + 1]

    slots = []
    for coords in components_mask(layout, connectivity=4):
        rr0, cc0, rr1, cc1 = bbox_coords(coords)
        rr0, cc0, rr1, cc1 = map(int, (rr0, cc0, rr1, cc1))
        h = int(rr1 - rr0 + 1)
        w = int(cc1 - cc0 + 1)
        sm = np.zeros((h, w), dtype=bool)
        sm[coords[:, 0] - rr0, coords[:, 1] - cc0] = True

        adj_colors = set()
        for r, c in coords:
            sr = int(lr0 + r)
            sc = int(lc0 + c)
            for dr, dc in DIR4:
                nr = int(sr + dr)
                nc = int(sc + dc)
                if 0 <= nr < sub.shape[0] and 0 <= nc < sub.shape[1]:
                    val = int(sub[nr, nc])
                    if val != int(bg):
                        adj_colors.add(val)
        slot_piece_colors = adj_colors & piece_colors
        assert len(slot_piece_colors) == 1, "each slot must be marked by exactly one piece color"
        slots.append(
            {
                "area": int(sm.sum()),
                "r0": int(rr0),
                "c0": int(cc0),
                "mask": sm,
                "color": int(next(iter(slot_piece_colors))),
            }
        )
    slots.sort(key=lambda s: (-int(s["area"]), int(s["r0"]), int(s["c0"])))

    return {
        "grid": g,
        "bg": bg,
        "frame_comp": frame_comp,
        "frame_color": int(bgc),
        "pieces": pieces,
        "piece_colors": piece_colors,
        "k": k,
        "layout": layout,
        "slots": slots,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Inflate the frame layout and fill each slot with the piece that shares its wall-marker color.

    Compared with the original task, the only semantic change is the matching condition:
    slot identity comes from wall color, while D4 shape testing is now only used to restore
    the matched piece's orientation before placement.
    """
    scene = _parse_scene(np.array(grid, dtype=int))
    pieces = scene["pieces"]
    slots = scene["slots"]
    k = int(scene["k"])
    layout = scene["layout"]
    bgc = int(scene["frame_color"])

    assert slots, "frame must contain at least one slot"

    piece_by_color = {int(p["color"]): p for p in pieces}
    assert len(piece_by_color) == len(pieces), "piece colors must be unique"

    placements: list[tuple[dict, dict]] = []
    for slot in slots:
        color = int(slot["color"])
        assert color in piece_by_color, "every slot marker color must have a matching piece"
        piece = piece_by_color[color]
        need_area = int(slot["area"]) * int(k) * int(k)
        assert int(piece["area"]) == int(need_area), "piece area must match scaled slot area"

        found = None
        for vp, vm in _dihedral_variants(piece["patch"], piece["mask"]):
            um = _downsample_blocks(vm, k)
            if um is None:
                continue
            if um.shape == slot["mask"].shape and np.array_equal(um, slot["mask"]):
                found = {"vp": vp, "vm": vm}
                break
        assert found is not None, "matched piece must admit a D4 orientation fitting the slot"
        placements.append((slot, found))

    out_h = (int(layout.shape[0]) + 2) * int(k)
    out_w = (int(layout.shape[1]) + 2) * int(k)
    out = np.full((out_h, out_w), int(bgc), dtype=int)

    for slot, cand in placements:
        top = (int(slot["r0"]) + 1) * int(k)
        left = (int(slot["c0"]) + 1) * int(k)
        paste(out, cand["vp"], int(top), int(left), mask=cand["vm"], overwrite=True)

    return out


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

K_CHOICES = [2, 3]
LAYOUT_H_RANGE = (4, 12)
LAYOUT_W_RANGE = (4, 8)
N_SLOTS_RANGE = (3, 6)
N_UNITS_MAX = 2
NON_UNIT_SIZE_CHOICES = [2, 3, 4, 5, 6, 7]
DUPLICATE_SIZE_CHOICES = [2, 3, 4, 5]

HOLE_DENSITY_RANGE = (0.20, 0.50)
PLACE_RETRIES = 80

CANVAS_DIM_RANGE = (15, 30)
AREA_BUFFER_MULT = 4
AREA_BUFFER_CONST = 50

DISTRACTOR_RANGE = (8, 13)
DISTRACTOR_RETRY_MULT = 10
MIN_DISTRACTORS = 8

GENERATE_MAX_RETRIES = 300


def _random_polyomino(target_size: int) -> np.ndarray:
    chosen = {(0, 0)}
    frontier = {(1, 0), (-1, 0), (0, 1), (0, -1)}
    while len(chosen) < int(target_size):
        nr, nc = random.choice(tuple(frontier))
        frontier.discard((nr, nc))
        chosen.add((nr, nc))
        for dr, dc in DIR4:
            nxt = (int(nr + dr), int(nc + dc))
            if nxt not in chosen:
                frontier.add(nxt)
    coords = np.array(sorted(chosen))
    r0, c0 = coords.min(axis=0)
    coords[:, 0] -= int(r0)
    coords[:, 1] -= int(c0)
    r1, c1 = coords.max(axis=0)
    mask = np.zeros((int(r1 + 1), int(c1 + 1)), dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _dihedral_masks(mask: np.ndarray) -> list[np.ndarray]:
    out = []
    seen = set()
    for kk in range(4):
        r = np.rot90(mask, kk)
        for cand in (r, np.fliplr(r)):
            key = cand.tobytes() + bytes([cand.shape[0], cand.shape[1]])
            if key in seen:
                continue
            seen.add(key)
            out.append(cand.copy())
    return out


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[:-1, :] |= mask[1:, :]
    out[1:, :] |= mask[:-1, :]
    out[:, :-1] |= mask[:, 1:]
    out[:, 1:] |= mask[:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[:-1, 1:] |= mask[1:, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    return out


def _forbid_coords(forbidden: np.ndarray, coords: np.ndarray) -> None:
    cell_mask = np.zeros_like(forbidden)
    cell_mask[coords[:, 0], coords[:, 1]] = True
    forbidden |= _dilate8(cell_mask)


def _wall_neighbors_layout(coords: np.ndarray, h: int, w: int) -> np.ndarray:
    occupied = {tuple(map(int, rc)) for rc in coords.tolist()}
    walls = set()
    for r, c in occupied:
        for dr, dc in DIR4:
            nr = int(r + dr)
            nc = int(c + dc)
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in occupied:
                walls.add((nr, nc))
    if not walls:
        return np.zeros((0, 2), dtype=int)
    return np.array(sorted(walls), dtype=int)


def _place_mask(
    base_mask: np.ndarray, layout_h: int, layout_w: int, forbidden: np.ndarray, claimed_walls: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    variants = _dihedral_masks(base_mask)
    random.shuffle(variants)
    for vm in variants:
        pts = np.argwhere(vm)
        vh, vw = map(int, vm.shape)
        positions = [(r0, c0) for r0 in range(layout_h - vh + 1) for c0 in range(layout_w - vw + 1)]
        random.shuffle(positions)
        for r0, c0 in positions[:PLACE_RETRIES]:
            coords = pts + np.array([[int(r0), int(c0)]], dtype=int)
            if bool(forbidden[coords[:, 0], coords[:, 1]].any()):
                continue
            walls = _wall_neighbors_layout(coords, layout_h, layout_w)
            if walls.size and bool(claimed_walls[walls[:, 0], walls[:, 1]].any()):
                continue
            return coords, vm
    return None


def _gen_layout(h: int, w: int, n_slots: int, n_units: int) -> list[dict] | None:
    forbidden = np.zeros((h, w), dtype=bool)
    claimed_walls = np.zeros((h, w), dtype=bool)
    slots: list[dict] = []

    n_non_unit = int(n_slots - n_units)
    plans: list[dict] = []
    for _ in range(n_units):
        plans.append({"is_unit": True, "base_mask": np.ones((1, 1), dtype=bool)})

    if n_non_unit > 0:
        force_duplicate = bool(n_non_unit >= 2)
        if force_duplicate:
            dup_mask = _random_polyomino(random.choice(DUPLICATE_SIZE_CHOICES))
            plans.append({"is_unit": False, "base_mask": dup_mask})
            plans.append({"is_unit": False, "base_mask": dup_mask})
            remaining = int(n_non_unit - 2)
        else:
            remaining = int(n_non_unit)
        for _ in range(remaining):
            plans.append({"is_unit": False, "base_mask": _random_polyomino(random.choice(NON_UNIT_SIZE_CHOICES))})

    # Place larger masks first for better packing, but keep enough randomness in tie-breaking.
    random.shuffle(plans)
    plans.sort(key=lambda x: -int(x["base_mask"].sum()))

    for plan in plans:
        placed = _place_mask(plan["base_mask"], h, w, forbidden, claimed_walls)
        if placed is None:
            return None
        coords, actual_mask = placed
        walls = _wall_neighbors_layout(coords, h, w)
        if walls.size:
            claimed_walls[walls[:, 0], walls[:, 1]] = True
        _forbid_coords(forbidden, coords)
        shape_key = _canonical_shape_key(actual_mask)
        slots.append(
            {
                "coords": coords,
                "area": int(coords.shape[0]),
                "mask": actual_mask,
                "shape_key": shape_key,
                "is_unit": bool(plan["is_unit"]),
            }
        )

    layout_mask = np.zeros((h, w), dtype=bool)
    for slot in slots:
        layout_mask[slot["coords"][:, 0], slot["coords"][:, 1]] = True

    density = float(layout_mask.sum()) / float(h * w)
    lo, hi = HOLE_DENSITY_RANGE
    if not (float(lo) <= density <= float(hi)):
        return None
    if not (bool(layout_mask[0].any()) and bool(layout_mask[-1].any()) and bool(layout_mask[:, 0].any()) and bool(layout_mask[:, -1].any())):
        return None

    shape_counts = Counter(slot["shape_key"] for slot in slots)
    if max(shape_counts.values()) < 2:
        return None

    return slots


def _make_piece(slot: dict, color: int, k: int, apply_dihedral: bool) -> dict:
    if slot["area"] == 1:
        patch = np.full((k, k), color, dtype=int)
        mask = np.ones((k, k), dtype=bool)
        return {
            "color": color,
            "slot": slot,
            "out_patch": patch,
            "out_mask": mask,
            "in_patch": patch,
            "in_mask": mask,
        }

    out_mask = np.kron(slot["mask"], np.ones((k, k), dtype=bool))
    out_patch = np.where(out_mask, color, 0).astype(int)

    if apply_dihedral:
        rot = random.randint(0, 3)
        in_mask = np.rot90(out_mask, rot)
        in_patch = np.rot90(out_patch, rot)
        if random.random() < 0.5:
            in_mask = np.fliplr(in_mask)
            in_patch = np.fliplr(in_patch)
    else:
        in_mask, in_patch = out_mask, out_patch

    return {
        "color": color,
        "slot": slot,
        "out_patch": out_patch,
        "out_mask": out_mask,
        "in_patch": in_patch,
        "in_mask": in_mask,
    }


def _build_expected_output(pieces: list[dict], layout_h: int, layout_w: int, k: int, frame_color: int) -> np.ndarray:
    out = np.full(((layout_h + 2) * k, (layout_w + 2) * k), frame_color, dtype=int)
    for p in pieces:
        coords = p["slot"]["coords"]
        top = (int(coords[:, 0].min()) + 1) * k
        left = (int(coords[:, 1].min()) + 1) * k
        paste(out, p["out_patch"], top, left, mask=p["out_mask"], overwrite=True)
    return out


def _slot_wall_neighbors_frame(slot: dict, layout_mask: np.ndarray, layout_h: int, layout_w: int) -> list[tuple[int, int]]:
    walls = set()
    for r, c in slot["coords"]:
        rr = int(r + 1)
        cc = int(c + 1)
        for dr, dc in DIR4:
            nr = int(rr + dr)
            nc = int(cc + dc)
            if not (0 <= nr < layout_h + 2 and 0 <= nc < layout_w + 2):
                continue
            if 1 <= nr <= layout_h and 1 <= nc <= layout_w and bool(layout_mask[int(nr - 1), int(nc - 1)]):
                continue
            walls.add((nr, nc))
    return sorted(walls)


def _build_input_canvas(
    pieces: list[dict], layout_mask: np.ndarray, layout_h: int, layout_w: int, k: int, frame_color: int, distractor_color: int
) -> np.ndarray | None:
    frame_h, frame_w = layout_h + 2, layout_w + 2
    total_piece_area = sum(int(p["in_mask"].sum()) for p in pieces)
    needed_cells = frame_h * frame_w + AREA_BUFFER_MULT * total_piece_area + AREA_BUFFER_CONST

    min_h = max(CANVAS_DIM_RANGE[0], frame_h + 2)
    min_w = max(CANVAS_DIM_RANGE[0], frame_w + 2)
    canvas_h = random.randint(min_h, CANVAS_DIM_RANGE[1])
    canvas_w = random.randint(min_w, CANVAS_DIM_RANGE[1])
    while canvas_h * canvas_w < needed_cells and (canvas_h < CANVAS_DIM_RANGE[1] or canvas_w < CANVAS_DIM_RANGE[1]):
        if canvas_h <= canvas_w and canvas_h < CANVAS_DIM_RANGE[1]:
            canvas_h += 1
        elif canvas_w < CANVAS_DIM_RANGE[1]:
            canvas_w += 1
        else:
            break
    if canvas_h * canvas_w < needed_cells:
        return None

    canvas = np.zeros((canvas_h, canvas_w), dtype=int)

    frame_img = np.full((frame_h, frame_w), frame_color, dtype=int)
    frame_img[1:-1, 1:-1] = frame_color
    frame_img[1:-1, 1:-1][layout_mask] = 0
    for piece in pieces:
        for wr, wc in _slot_wall_neighbors_frame(piece["slot"], layout_mask, layout_h, layout_w):
            frame_img[int(wr), int(wc)] = int(piece["color"])

    fr_max, fc_max = canvas_h - frame_h - 1, canvas_w - frame_w - 1
    if fr_max < 1 or fc_max < 1:
        return None
    paste(canvas, frame_img, random.randint(1, fr_max), random.randint(1, fc_max))

    for p in sorted(pieces, key=lambda p: -int(p["in_mask"].sum())):
        try:
            pr, pc = random_free_location(
                canvas,
                p["in_patch"],
                transparent=0,
                border=0,
                padding=1,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(canvas, p["in_patch"], pr, pc, mask=p["in_mask"], overwrite=True)

    n_distractors = random.randint(*DISTRACTOR_RANGE)
    placed = 0
    for _ in range(n_distractors * DISTRACTOR_RETRY_MULT):
        if placed >= n_distractors:
            break
        rr = random.randint(0, canvas_h - 1)
        cc = random.randint(0, canvas_w - 1)
        if int(canvas[rr, cc]) != 0:
            continue
        r_lo, r_hi = max(0, rr - 1), min(canvas_h, rr + 2)
        c_lo, c_hi = max(0, cc - 1), min(canvas_w, cc + 2)
        if bool(np.any(canvas[r_lo:r_hi, c_lo:c_hi] != 0)):
            continue
        canvas[rr, cc] = int(distractor_color)
        placed += 1
    if placed < MIN_DISTRACTORS:
        return None

    vals, counts = np.unique(canvas, return_counts=True)
    if int(vals[np.argmax(counts)]) != 0:
        return None

    return canvas


def _gen_attempt() -> np.ndarray | None:
    k = random.choice(K_CHOICES)
    layout_h = random.randint(*LAYOUT_H_RANGE)
    layout_w = random.randint(*LAYOUT_W_RANGE)
    n_slots = random.randint(*N_SLOTS_RANGE)
    if n_slots == 3:
        n_units = 1
    else:
        n_units = random.randint(1, min(N_UNITS_MAX, n_slots - 1))

    palette = list(range(1, 10))
    random.shuffle(palette)
    frame_color = palette.pop()
    distractor_color = palette.pop()
    if len(palette) < n_slots:
        return None
    slot_colors = palette[:n_slots]

    slots = _gen_layout(layout_h, layout_w, n_slots, n_units)
    if slots is None:
        return None

    random.shuffle(slots)
    for slot, color in zip(slots, slot_colors):
        slot["color"] = int(color)

    layout_mask = np.zeros((layout_h, layout_w), dtype=bool)
    for slot in slots:
        layout_mask[slot["coords"][:, 0], slot["coords"][:, 1]] = True

    pieces = []
    for slot in slots:
        pieces.append(_make_piece(slot, int(slot["color"]), k, apply_dihedral=bool(slot["area"] > 1)))

    expected_out = _build_expected_output(pieces, layout_h, layout_w, k, frame_color)
    canvas = _build_input_canvas(pieces, layout_mask, layout_h, layout_w, k, frame_color, distractor_color)
    if canvas is None:
        return None

    try:
        produced = main(canvas)
    except Exception:
        return None
    if not np.array_equal(produced, expected_out):
        return None

    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_MAX_RETRIES):
        res = _gen_attempt()
        if res is not None:
            return res
    raise RuntimeError(f"generate_input: failed to generate after {GENERATE_MAX_RETRIES} attempts")


def variant_metrics(grid: np.ndarray) -> dict:
    """
    Lightweight diagnostic helper used outside the solver to confirm this variant
    is genuinely color-driven rather than silently collapsing back to shape matching.
    """
    scene = _parse_scene(np.array(grid, dtype=int))
    shape_counts = Counter(_canonical_shape_key(slot["mask"]) for slot in scene["slots"])
    return {
        "slot_count": int(len(scene["slots"])),
        "unit_slot_count": int(sum(int(slot["area"]) == 1 for slot in scene["slots"])),
        "duplicate_shape_groups": int(sum(int(v) >= 2 for v in shape_counts.values())),
        "max_duplicate_group": int(max(shape_counts.values())),
        "unique_slot_colors": int(len({int(slot["color"]) for slot in scene["slots"]})),
        "k": int(scene["k"]),
    }
