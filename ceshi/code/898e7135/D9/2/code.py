# Variant of 898e7135
# Dimension: D9 — 洞区域的空间约束 (meta)
# Change: 洞区域 4-连通且彼此 8-隔离（相邻洞之间至少隔 1 像素墙壁，含对角线方向），面积 1-7 像素
# ... → 洞面积范围扩大（含大面积洞）
#
# Design analysis
# ===============
# variant_spec:
#   unchanged_core_rule:
#     - 输入仍然包含一个单色矩形框架，框内是背景色洞区域，框外散落着若干按 k×k 块放大的精灵。
#     - 输出仍然是把框内洞布局按同一 k 放大，并把每个精灵填回其匹配洞位置后的裁剪结果。
#   D9_change:
#     - 非单元洞面积从 2-7 扩到 2-12。
#     - 为了让变化肉眼可见，生成器强制每题至少包含 1 个面积 8-12 的非单元洞。
#     - 大洞不允许退化成整块矩形或 1 像素粗长条；至少有一个大洞必须是 bbox 内带缺口的不规则形。
#   concrete_pixel_behavior:
#     - 例：原任务里 3 格 L 形洞会变成 2x2 或 3x3 块拼出的 L 形精灵。
#     - 现在允许 9 格、10 格、12 格这类更大的 4 连通洞；对应输入里会出现更大的块状精灵，
#       例如 3x4 bbox 中缺 2 格的块状形状，旋转/翻转后散落在框外。
#     - solver 仍然把每个精灵按 k×k 分块下采样回布尔形状，再与框内洞区域做 D4 匹配。
#   behavior_trace:
#     - 原逻辑：若某洞面积为 6，则 piece 面积只能是 6*k*k；matching 时按面积和块级 mask 精确相等配对。
#     - 变体逻辑：若某洞面积为 10，则 piece 面积变成 10*k*k；同一 matching 逻辑仍然成立，
#       只是 generator 现在真的会构造这类大洞与大精灵。
#     - 差异不在 solver 算法类型，而在 generator 的可生成形状空间和 solver 需覆盖的实例规模。
#   complexity_calibration:
#     - 若只是把 size choice 改成 2-12，但仍经常采到 <=7，则会退化成原题。
#     - 因此生成器要求至少 1 个 area>=8 的洞，并在 k=2 时偶尔允许第二个大洞。
#     - 同时保留 1-2 个 unit 洞，用于稳定推断 k；颜色规则保持原样，避免引入新维度。
#   implicit_assumption_review_on_main:
#     - find_components(..., monochromatic=False): 依赖“整圈框架 + 内部墙壁”作为最大的 4 连通前景组件。
#       大洞只会减少框内背景，不会破坏该前提；generator 保证框和所有 piece 至少隔 1 格背景。
#     - layout = hole tight crop: 原代码默认洞的紧致 bbox 就是待放大的布局。
#       变体继续沿用这个定义；generator 先在工作网格上造洞，再按洞 bbox 裁成最终 layout。
#     - slot extraction by 4-connectivity: 依赖每个洞内部 4 连通、洞与洞 8 隔离。
#       D9 没改这一点，generator 仍用 8 邻域禁放保证它成立。
#     - k inference from smallest solid square piece: 依赖至少存在 unit-slot 对应的 k×k 实心块。
#       生成器继续强制 1-2 个 unit 洞，因此该推断仍稳定。
#     - matching by exact block mask equality under D4: 依赖每个非单元洞的 D4 形状唯一，避免同形冲突。
#       生成器继续去重 canonical D4 shape key。
#   placement_legality:
#     - 先在固定 12x8 工作布局上放洞，放置时把每个洞及其 8 邻域都标成 forbidden，保证洞间至少隔 1 格墙。
#     - 再对洞整体 tight-crop，得到真实框内布局；这样既便于放置大洞，也保持 solver 的裁剪定义。
#     - 画布容量按 piece 总面积估算；k=3 时减少 slot 数和总洞面积预算，避免大精灵放不下。
#   visual_observability:
#     - 单靠“允许更大面积”但样本里不出现大洞，人类看不出变体；因此至少一个 area>=8 是硬约束。
#     - 若大洞是纯矩形，旋转/翻转挑战几乎不可见；因此至少一个大洞必须在 bbox 中留缺口。
#   visual_cleanliness:
#     - 保持单色 piece、单色框、单色干扰点，避免同一区域叠层。
#     - piece 与 frame/彼此之间仍留 >=1 格背景，单像素干扰点仍独立散布。
#   color_inference:
#     - 颜色不参与排序或变换判定；solver 只用形状、面积和位置。
#     - 非单元 piece 颜色各不相同，unit piece 允许共享一色，和原题一致，颜色规律可从结构上看懂。
#   new_solver_assumptions_after_variant:
#     - main 新增的实际假设只有“至少存在一个 unit piece 来推断 k”，这一点由 generator 保证。
#     - main 并不依赖面积上限 7，因此 D9 改动不需要改 solver 推理链，只需要 generator 确保新样本确实覆盖 area 8-12。

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import (
    bbox,
    bbox_coords,
    components_mask,
    crop_bbox,
    find_components,
    get_bg,
    paste,
    random_free_location,
)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Inflate a frame's hole-layout and fill each hole with its matching piece.

    D9 variant note:
    The matching logic is unchanged; it already supports arbitrarily shaped slots.
    The important difference is that the generator now supplies larger 4-connected
    holes (up to area 12), so the solver must continue to match larger block masks.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps, "input must contain a frame component"
    frame_comp = max(comps, key=lambda d: int(d["area"]))
    fr0, fc0, fr1, fc1 = map(int, frame_comp["bbox"])

    frame_cols = [int(g[int(r), int(c)]) for r, c in frame_comp["coords"]]
    bgc = int(Counter(frame_cols).most_common(1)[0][0])

    sub = crop_bbox(g, (fr0, fc0, fr1, fc1))
    hole = sub == bg
    hole_box = bbox(hole)
    assert hole_box is not None, "frame must enclose at least one bg hole"
    lr0, lc0, lr1, lc1 = map(int, hole_box)
    layout = hole[lr0 : lr1 + 1, lc0 : lc1 + 1]

    slots = []
    for coords in components_mask(layout, connectivity=4):
        rr0, cc0, rr1, cc1 = map(int, bbox_coords(coords))
        sm = np.zeros((rr1 - rr0 + 1, cc1 - cc0 + 1), dtype=bool)
        sm[coords[:, 0] - rr0, coords[:, 1] - cc0] = True
        slots.append({"area": int(sm.sum()), "r0": rr0, "c0": cc0, "mask": sm})
    slots.sort(key=lambda s: (-int(s["area"]), int(s["r0"]), int(s["c0"])))
    assert slots, "frame must contain at least one slot"

    pieces = []
    for comp in comps:
        if comp is frame_comp:
            continue
        coords = comp["coords"]
        if coords.size == 0 or int(coords.shape[0]) <= 1:
            continue
        pr0, pc0, pr1, pc1 = map(int, comp["bbox"])
        patch = crop_bbox(g, (pr0, pc0, pr1, pc1))
        pm = np.zeros_like(patch, dtype=bool)
        pm[coords[:, 0] - pr0, coords[:, 1] - pc0] = True
        pieces.append({"area": int(pm.sum()), "mask": pm, "patch": patch})
    assert pieces, "at least one non-frame piece must exist"

    k = None
    for p in pieces:
        h, w = map(int, p["mask"].shape)
        if h == w and int(p["mask"].sum()) == h * w:
            k = h if k is None else min(int(k), h)
    k = int(k or 1)

    def downsample_blocks(mask: np.ndarray) -> np.ndarray | None:
        hh, ww = map(int, mask.shape)
        if hh % k != 0 or ww % k != 0:
            return None
        out = np.zeros((hh // k, ww // k), dtype=bool)
        for r in range(out.shape[0]):
            for c in range(out.shape[1]):
                block = mask[r * k : (r + 1) * k, c * k : (c + 1) * k]
                if bool(block.all()):
                    out[r, c] = True
                elif bool(block.any()):
                    return None
        return out

    def dihedral_variants(
        patch: np.ndarray, mask: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        base = []
        for kk in range(4):
            rp = np.rot90(patch, kk)
            rm = np.rot90(mask, kk)
            base.append((rp, rm))
            base.append((np.fliplr(rp), np.fliplr(rm)))
        uniq = []
        seen: set[tuple[tuple[int, int], bytes]] = set()
        for vp, vm in base:
            key = (tuple(map(int, vm.shape)), vm.tobytes())
            if key in seen:
                continue
            seen.add(key)
            uniq.append((vp, vm))
        return uniq

    candidates = []
    for pi, p in enumerate(pieces):
        for vp, vm in dihedral_variants(p["patch"], p["mask"]):
            um = downsample_blocks(vm)
            if um is None:
                continue
            candidates.append({"pi": int(pi), "area": int(p["area"]), "um": um, "vp": vp, "vm": vm})

    used_piece: set[int] = set()
    placements: list[tuple[dict, dict]] = []
    for s in slots:
        need_area = int(s["area"]) * k * k
        found = None
        for cand in candidates:
            if int(cand["pi"]) in used_piece:
                continue
            if int(cand["area"]) != need_area:
                continue
            if cand["um"].shape != s["mask"].shape:
                continue
            if np.array_equal(cand["um"], s["mask"]):
                found = cand
                break
        assert found is not None, "every slot must match a piece"
        used_piece.add(int(found["pi"]))
        placements.append((s, found))

    out = np.full(((layout.shape[0] + 2) * k, (layout.shape[1] + 2) * k), bgc, dtype=int)
    for s, cand in placements:
        paste(out, cand["vp"], (int(s["r0"]) + 1) * k, (int(s["c0"]) + 1) * k, mask=cand["vm"], overwrite=True)
    return out


K_CHOICES = [2, 3]
WORK_LAYOUT_SHAPE = (12, 8)
FINAL_LAYOUT_H_RANGE = (4, 12)
FINAL_LAYOUT_W_RANGE = (4, 8)
N_SLOTS_RANGE_K2 = (3, 6)
N_SLOTS_RANGE_K3 = (3, 4)
N_UNITS_MAX = 2

SMALL_NON_UNIT_CHOICES = [2, 3, 4, 5, 6, 7]
BIG_NON_UNIT_CHOICES = [8, 9, 10, 11, 12]
SECOND_BIG_CHOICES = [8, 9, 10]
TOTAL_SLOT_AREA_BUDGET = {2: 34, 3: 20}

HOLE_DENSITY_RANGE = (0.20, 0.50)
UNIT_SLOT_RETRIES = 40
NON_UNIT_SLOT_RETRIES = 120
GROW_SEED_TRIES = 20

CANVAS_DIM_RANGE = (15, 30)
AREA_BUFFER_MULT = 4
AREA_BUFFER_CONST = 50

DISTRACTOR_RANGE = (8, 13)
DISTRACTOR_RETRY_MULT = 12
MIN_DISTRACTORS = 8

GENERATE_MAX_RETRIES = 500


def _tight_mask_from_coords(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    r0, c0, r1, c1 = map(int, bbox_coords(coords))
    sm = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    sm[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return sm, (r0, c0)


def _canonical_shape_key(mask: np.ndarray) -> bytes:
    variants = []
    for kk in range(4):
        rot = np.rot90(mask, kk)
        variants.append(bytes([rot.shape[0], rot.shape[1]]) + rot.tobytes())
        flip = np.fliplr(rot)
        variants.append(bytes([flip.shape[0], flip.shape[1]]) + flip.tobytes())
    return min(variants)


def _is_irregular_large_shape(mask: np.ndarray) -> bool:
    h, w = map(int, mask.shape)
    area = int(mask.sum())
    return area >= 8 and h >= 3 and w >= 3 and area < h * w


def _grow_slot(forbidden: np.ndarray, target_size: int, *, want_irregular_large: bool) -> np.ndarray | None:
    H, W = map(int, forbidden.shape)
    free_cells = np.argwhere(~forbidden)
    if free_cells.size == 0:
        return None
    free_list = [tuple(map(int, cell)) for cell in free_cells]
    random.shuffle(free_list)
    for r0, c0 in free_list[:GROW_SEED_TRIES]:
        chosen = {(r0, c0)}
        frontier = {(r0 - 1, c0), (r0 + 1, c0), (r0, c0 - 1), (r0, c0 + 1)}
        while len(chosen) < target_size and frontier:
            cands = []
            for nr, nc in frontier:
                if 0 <= nr < H and 0 <= nc < W and not forbidden[nr, nc] and (nr, nc) not in chosen:
                    freedom = 0
                    for ar, ac in ((nr - 1, nc), (nr + 1, nc), (nr, nc - 1), (nr, nc + 1)):
                        if 0 <= ar < H and 0 <= ac < W and not forbidden[ar, ac] and (ar, ac) not in chosen:
                            freedom += 1
                    cands.append((freedom, nr, nc))
            if not cands:
                break
            cands.sort(reverse=True)
            top = cands[: min(4, len(cands))]
            _, nr, nc = random.choice(top)
            chosen.add((nr, nc))
            frontier.discard((nr, nc))
            frontier.update({(nr - 1, nc), (nr + 1, nc), (nr, nc - 1), (nr, nc + 1)})
        if len(chosen) != target_size:
            continue
        coords = np.array(sorted(chosen), dtype=int)
        if want_irregular_large:
            mask, _ = _tight_mask_from_coords(coords)
            if not _is_irregular_large_shape(mask):
                continue
        return coords
    return None


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


def _choose_slot_sizes(k: int) -> tuple[int, list[int]] | None:
    n_slots = random.randint(*(N_SLOTS_RANGE_K2 if k == 2 else N_SLOTS_RANGE_K3))
    n_units = random.randint(1, min(N_UNITS_MAX, n_slots - 1))
    non_unit_count = n_slots - n_units
    if non_unit_count <= 0:
        return None

    budget = int(TOTAL_SLOT_AREA_BUDGET[int(k)])
    sizes = [random.choice(BIG_NON_UNIT_CHOICES)]

    if k == 2 and non_unit_count >= 3 and random.random() < 0.35:
        candidate = random.choice(SECOND_BIG_CHOICES)
        if candidate + n_units + sum(sizes) + 2 * (non_unit_count - 2) <= budget:
            sizes.append(candidate)

    while len(sizes) < non_unit_count:
        remaining = non_unit_count - len(sizes)
        max_allowed = budget - n_units - sum(sizes) - 2 * (remaining - 1)
        if max_allowed < 2:
            return None
        choices = [s for s in SMALL_NON_UNIT_CHOICES if s <= max_allowed]
        if not choices:
            return None
        if k == 3:
            choices = choices[: min(4, len(choices))]
        sizes.append(random.choice(choices))

    random.shuffle(sizes)
    return n_units, sizes


def _generate_slots(slot_sizes: list[int], n_units: int) -> list[dict] | None:
    forbidden = np.zeros(WORK_LAYOUT_SHAPE, dtype=bool)
    slots: list[dict] = []
    used_shape_keys: set[bytes] = set()
    need_irregular_large = True

    for size in sorted(slot_sizes, reverse=True):
        want_irregular = need_irregular_large and size >= 8
        for _ in range(NON_UNIT_SLOT_RETRIES):
            coords = _grow_slot(forbidden, size, want_irregular_large=want_irregular)
            if coords is None:
                continue
            mask, _ = _tight_mask_from_coords(coords)
            key = _canonical_shape_key(mask)
            if key in used_shape_keys:
                continue
            used_shape_keys.add(key)
            slots.append({"coords": coords, "area": int(size), "mask": mask, "shape_key": key})
            _forbid_coords(forbidden, coords)
            if want_irregular:
                need_irregular_large = False
            break
        else:
            return None

    unit_slots: list[dict] = []
    for _ in range(n_units):
        for _ in range(UNIT_SLOT_RETRIES):
            coords = _grow_slot(forbidden, 1, want_irregular_large=False)
            if coords is None:
                continue
            unit_slots.append({"coords": coords, "area": 1})
            _forbid_coords(forbidden, coords)
            break
        else:
            return None

    if need_irregular_large:
        return None
    return unit_slots + slots


def _crop_layout(slots: list[dict]) -> tuple[list[dict], np.ndarray] | None:
    full_mask = np.zeros(WORK_LAYOUT_SHAPE, dtype=bool)
    for s in slots:
        full_mask[s["coords"][:, 0], s["coords"][:, 1]] = True
    hole_box = bbox(full_mask)
    if hole_box is None:
        return None
    r0, c0, r1, c1 = map(int, hole_box)
    layout_mask = full_mask[r0 : r1 + 1, c0 : c1 + 1]
    h, w = map(int, layout_mask.shape)
    if not (FINAL_LAYOUT_H_RANGE[0] <= h <= FINAL_LAYOUT_H_RANGE[1]):
        return None
    if not (FINAL_LAYOUT_W_RANGE[0] <= w <= FINAL_LAYOUT_W_RANGE[1]):
        return None
    density = float(layout_mask.sum()) / float(h * w)
    if not (HOLE_DENSITY_RANGE[0] <= density <= HOLE_DENSITY_RANGE[1]):
        return None

    cropped_slots = []
    for s in slots:
        coords = s["coords"].copy()
        coords[:, 0] -= r0
        coords[:, 1] -= c0
        entry = {"coords": coords, "area": int(s["area"])}
        if int(s["area"]) > 1:
            mask, _ = _tight_mask_from_coords(coords)
            entry["mask"] = mask
            entry["shape_key"] = _canonical_shape_key(mask)
        cropped_slots.append(entry)
    return cropped_slots, layout_mask


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
        paste(out, p["out_patch"], (int(coords[:, 0].min()) + 1) * k, (int(coords[:, 1].min()) + 1) * k, mask=p["out_mask"], overwrite=True)
    return out


def _build_input_canvas(
    pieces: list[dict],
    layout_mask: np.ndarray,
    layout_h: int,
    layout_w: int,
    k: int,
    frame_color: int,
    distractor_color: int,
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
    frame_img[1:-1, 1:-1] = np.where(layout_mask, 0, frame_color)
    fr_max, fc_max = canvas_h - frame_h - 1, canvas_w - frame_w - 1
    if fr_max < 1 or fc_max < 1:
        return None
    paste(canvas, frame_img, random.randint(1, fr_max), random.randint(1, fc_max))

    for p in sorted(pieces, key=lambda item: -int(item["in_mask"].sum())):
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
        if canvas[rr, cc] != 0:
            continue
        r_lo, r_hi = max(0, rr - 1), min(canvas_h, rr + 2)
        c_lo, c_hi = max(0, cc - 1), min(canvas_w, cc + 2)
        if np.any(canvas[r_lo:r_hi, c_lo:c_hi] != 0):
            continue
        canvas[rr, cc] = distractor_color
        placed += 1
    if placed < MIN_DISTRACTORS:
        return None

    vals, counts = np.unique(canvas, return_counts=True)
    if int(vals[np.argmax(counts)]) != 0:
        return None
    return canvas


def _gen_attempt() -> np.ndarray | None:
    k = random.choice(K_CHOICES)
    size_plan = _choose_slot_sizes(k)
    if size_plan is None:
        return None
    n_units, non_unit_sizes = size_plan

    palette = list(range(1, 10))
    random.shuffle(palette)
    frame_color = palette.pop()
    distractor_color = palette.pop()
    needed_piece_colors = len(non_unit_sizes) + 1
    if len(palette) < needed_piece_colors:
        return None
    non_unit_colors = palette[: len(non_unit_sizes)]
    unit_color = palette[len(non_unit_sizes)]

    slots = _generate_slots(non_unit_sizes, n_units)
    if slots is None:
        return None
    cropped = _crop_layout(slots)
    if cropped is None:
        return None
    slots, layout_mask = cropped

    unit_slots = [s for s in slots if int(s["area"]) == 1]
    non_unit_slots = [s for s in slots if int(s["area"]) > 1]
    if not unit_slots or not any(int(s["area"]) >= 8 for s in non_unit_slots):
        return None
    if not any(_is_irregular_large_shape(s["mask"]) for s in non_unit_slots if int(s["area"]) >= 8):
        return None

    pieces = [_make_piece(s, non_unit_colors[i], k, apply_dihedral=True) for i, s in enumerate(non_unit_slots)]
    pieces += [_make_piece(s, unit_color, k, apply_dihedral=False) for s in unit_slots]

    layout_h, layout_w = map(int, layout_mask.shape)
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
    raise RuntimeError(f"generate_input: failed after {GENERATE_MAX_RETRIES} attempts")
